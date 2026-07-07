"""tamil-talk backend: FastAPI app serving the single-page frontend and
the /talk WebSocket that orchestrates STT -> LLM -> TTS.
"""
import json
import os

import anyio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tamil_talk.parler_synth import make_parler_synth
from tamil_talk.whisper_stt import make_whisper_stt
from tamil_talk.llm import append_assistant_turn, append_user_turn, accumulate_sentences, make_llm

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


async def _aiter_sync(sync_iter):
    """Yield from a blocking generator without blocking the event loop.

    Cleans up (closes) the underlying generator on cancellation, e.g. a
    client disconnecting mid-stream, so an in-flight TTS generator can
    release resources.
    """
    sentinel = object()
    it = iter(sync_iter)
    try:
        while True:
            chunk = await anyio.to_thread.run_sync(lambda: next(it, sentinel))
            if chunk is sentinel:
                break
            yield chunk
    finally:
        if hasattr(it, "close"):
            it.close()


def build_app() -> FastAPI:
    transcribe = make_whisper_stt()
    synth = make_parler_synth()
    chat_stream = make_llm()

    app = FastAPI()
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.websocket("/talk")
    async def talk(ws: WebSocket):
        await ws.accept()
        history = []
        try:
            while True:
                buf = bytearray()
                think = False
                sample_rate = 16000
                system_prompt = ""
                while True:
                    msg = await ws.receive()
                    if msg.get("bytes") is not None:
                        buf.extend(msg["bytes"])
                    elif msg.get("text") is not None:
                        evt = json.loads(msg["text"])
                        if evt.get("event") == "end":
                            think = bool(evt.get("think", False))
                            sample_rate = int(evt.get("sample_rate", 16000))
                            system_prompt = str(evt.get("system_prompt", ""))
                            break
                    else:
                        return  # client disconnected

                if not buf:
                    await ws.send_text(json.dumps({"event": "error", "detail": "empty audio"}))
                    continue

                text = await anyio.to_thread.run_sync(transcribe, bytes(buf), sample_rate)
                await ws.send_text(json.dumps({"event": "transcript", "text": text}))

                history = append_user_turn(history, text)

                reply_sentences = []
                audio_started = False
                sentence_iter = accumulate_sentences(chat_stream(history, think, system_prompt))
                index = 0
                async for sentence in _aiter_sync(sentence_iter):
                    reply_sentences.append(sentence)
                    await ws.send_text(json.dumps({
                        "event": "response_sentence", "text": sentence, "index": index,
                    }))
                    if not audio_started:
                        await ws.send_text(json.dumps({"event": "audio_start", "sample_rate": synth.sample_rate}))
                        audio_started = True
                    async for chunk in _aiter_sync(synth(sentence, "tamil_female", 0.3, 1.3, 1200, 1.0)):
                        await ws.send_bytes(chunk)
                    index += 1

                reply = " ".join(reply_sentences)
                history = append_assistant_turn(history, reply)
                await ws.send_text(json.dumps({"event": "done"}))
        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await ws.send_text(json.dumps({"event": "error", "detail": str(e)}))
            except Exception:
                pass

    return app


app = build_app()
