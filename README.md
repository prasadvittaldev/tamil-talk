# tamil-talk

A live Tamil voice-chat demo: push-to-talk mic → Tamil speech-to-text →
a Tamil-finetuned Gemma 4 chat model → Tamil text-to-speech → audio playback,
streamed sentence-by-sentence so it starts talking before the whole reply is
ready. Built by [Prasad Vittaldev](https://linkedin.com/in/prasadvittaldev/).

## How it works

```
 mic (browser)  --PCM-->  WebSocket /talk  --> Whisper STT (Tamil)
                                              --> Gemma 4 Tamil (Ollama, streamed)
                                              --> sentence-boundary detection
                                              --> Parler-TTS (Tamil), per sentence
                                              --> PCM audio  --WebSocket-->  browser
```

- **STT:** [`vasista22/whisper-tamil-large-v2`](https://huggingface.co/vasista22/whisper-tamil-large-v2)
  via `faster-whisper`.
- **LLM:** a Tamil-finetuned Gemma 4 model
  ([`prasadvittaldev/gemma4-tamil-e4b-it-GGUF`](https://huggingface.co/prasadvittaldev/gemma4-tamil-e4b-it-GGUF)),
  served through [Ollama](https://ollama.com) for streaming + a `think`/no-think
  toggle. If Ollama isn't reachable, the app falls back to loading the GGUF
  directly via `llama-cpp-python` (slower to start, no `think` toggle).
- **TTS:** [`ai4bharat/indic-parler-tts`](https://huggingface.co/ai4bharat/indic-parler-tts)'s
  Tamil speaker ("Jaya"), streamed per sentence as the LLM generates them —
  so playback starts after the first sentence, not the whole reply.

The frontend is a single static page (`tamil_talk/static/`): a push-to-talk
mic button, a "think mode" toggle, an optional persona/system-prompt field,
a live transcript with per-sentence highlighting synced to playback, and a
circular waveform animation during TTS.

## Prerequisites

- A CUDA GPU. This is not realistic to run on CPU — Whisper + the LLM +
  Parler-TTS all need to run in real time for a usable demo.
- Python 3.10+.
- [Ollama](https://ollama.com/download) installed and running (`ollama serve`,
  or it may already run as a background service depending on your install).

## Setup

1. **Install torch first**, matching your GPU's CUDA version — see
   [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
   (Tested with `torch==2.11.0+cu128` on an RTX 5060 Ti / Blackwell, which
   specifically needs the `cu128` build — older `cu121` wheels fail silently
   on that architecture. Match your own GPU's requirements instead of
   copying this blindly.)

2. **Install the rest of the dependencies:**

   ```bash
   pip install -r requirements.txt
   # or, to also run the test suite:
   pip install -r requirements-dev.txt
   ```

3. **Pull the Tamil LLM into Ollama:**

   ```bash
   ollama pull hf.co/prasadvittaldev/gemma4-tamil-e4b-it-GGUF:Q5_K_M
   ```

   If you'd rather skip Ollama entirely, install `llama-cpp-python` instead —
   the app will fall back to loading the GGUF directly the first time Ollama
   isn't reachable (slower cold start, no `think` toggle support).

## Run

```bash
python -m uvicorn tamil_talk.server:app --host 0.0.0.0 --port 9099
```

Open `http://localhost:9099/` in a browser (use `localhost`, not a bare LAN
IP — `getUserMedia` requires a secure context, so a plain `http://` LAN
address won't be allowed to use the mic; tunnel with something like `ngrok`
if you need to test from another device).

**Using it:** click and hold the mic button, speak a Tamil sentence, release
to send. Toggle "Think mode" to compare direct replies vs. the model's
reasoning mode. Type into the persona field to give the assistant a system
prompt/character — it applies to every turn from then on.

## Testing

```bash
pytest tests/ -v
```

Covers the LLM-facing pure logic (`tamil_talk/llm.py`): message building,
sentence-boundary streaming, and Ollama request/retry behavior (all mocked
at the HTTP boundary — no live model calls). There's no automated test for
the WebSocket pipeline or frontend; use these two scripts instead:

```bash
# No browser needed -- sends a WAV over the /talk socket, prints the JSON
# events, saves the spoken reply to manual_verify_out.wav
python manual_verify.py path/to/some_tamil_audio.wav
```

## Project layout

```
tamil_talk/
  server.py         FastAPI app + the /talk WebSocket (STT -> LLM -> TTS)
  llm.py            Ollama/GGUF chat client, streaming, sentence-boundary detection
  parler_synth.py   Parler-TTS synth backend
  whisper_stt.py    faster-whisper STT backend
  leveler.py        per-sentence loudness leveling + silence trimming
  resampler.py      PCM resampling (mic sample rate -> 16kHz for STT)
  text_chunk.py     sentence/clause splitting shared by the TTS path
  static/           frontend (index.html + app.js)
tests/
  test_llm.py
manual_verify.py    no-browser pipeline check
```

## Known limitations

- The `/talk` WebSocket is **unauthenticated** — fine for a local/demo
  deployment, not something to expose directly on the open internet without
  adding your own auth in front of it.
- The Gemma 4 Tamil model is licensed **CC-BY-NC-4.0** (non-commercial) —
  see its [model card](https://huggingface.co/prasadvittaldev/gemma4-tamil-e4b-it-GGUF)
  for why and what that means for your use.
- Sentence-sequential TTS streaming reduces *time-to-first-audio*, not total
  turn latency — the LLM still has to finish generating each sentence before
  it's spoken.
