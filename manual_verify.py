# manual_verify.py
"""Standalone check for the /talk pipeline, no frontend needed. Sends a short
pre-recorded Tamil WAV, prints the JSON events, saves the TTS reply to a WAV.

Run: python manual_verify.py path/to/input.wav
(record a short Tamil sentence yourself, e.g. with `arecord -f S16_LE -r 16000
-c 1 input.wav`, or reuse an existing sample under tests/fixtures/*.wav)
"""
import asyncio
import json
import sys
import wave

import websockets

WS_URL = "ws://localhost:9099/talk"


async def run(wav_path: str, think: bool = False):
    with wave.open(wav_path, "rb") as w:
        sample_rate = w.getframerate()
        pcm = w.readframes(w.getnframes())

    out_frames = bytearray()
    out_rate = None
    async with websockets.connect(WS_URL) as ws:
        await ws.send(pcm)
        await ws.send(json.dumps({"event": "end", "think": think, "sample_rate": sample_rate}))
        async for msg in ws:
            if isinstance(msg, bytes):
                out_frames.extend(msg)
            else:
                evt = json.loads(msg)
                print(evt)
                if evt.get("event") == "audio_start":
                    out_rate = evt["sample_rate"]
                if evt.get("event") in ("done", "error"):
                    break

    if out_frames and out_rate:
        with wave.open("manual_verify_out.wav", "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(out_rate)
            w.writeframes(bytes(out_frames))
        print(f"Saved reply audio to manual_verify_out.wav ({len(out_frames)} bytes @ {out_rate}Hz)")


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1], think="--think" in sys.argv))
