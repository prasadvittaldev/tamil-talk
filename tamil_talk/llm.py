"""LLM stage for Tamil-talkies: Ollama chat client (primary) with a direct
GGUF/llama-cpp-python fallback for when Ollama isn't reachable. See
docs/superpowers/specs/2026-07-07-tamil-talkies-design.md for the full design.
"""
import json
import re
import requests

OLLAMA_MODEL = "hf.co/prasadvittaldev/gemma4-tamil-e4b-it-GGUF:Q5_K_M"
GGUF_REPO_ID = "prasadvittaldev/gemma4-tamil-e4b-it-GGUF"
GGUF_FILENAME = "gemma4-tamil-e4b-it-Q5_K_M.gguf"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# Mirrors serving/llamacpp_synth.py's _SENT_SPLIT: split on sentence-ending
# punctuation followed by whitespace, or a newline.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.?!;।])\s+|\n+")


def check_ollama_reachable(base_url: str = DEFAULT_OLLAMA_BASE_URL) -> bool:
    try:
        resp = requests.get(f"{base_url}/api/version", timeout=2)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def append_user_turn(history: list, text: str) -> list:
    return history + [{"role": "user", "content": text}]


def append_assistant_turn(history: list, text: str) -> list:
    return history + [{"role": "assistant", "content": text}]


NO_THINK_SYSTEM_MESSAGE = (
    "Respond directly in Tamil without any preliminary thinking, "
    "planning, or reasoning steps."
)


def build_chat_messages(history: list, think: bool, system_prompt: str = "") -> list:
    """Shared by both the Ollama and GGUF-fallback chat paths: combine an
    optional user-supplied persona/system prompt with the no-thinking
    instruction (when think=False) into ONE system message -- not two
    separate system-role entries, since not every chat template handles
    multiple system turns well. Needed on the Ollama path too because
    Ollama hard-rejects think=True for some model imports (see
    ollama_chat_stream) -- without the no-think instruction, both toggle
    states would silently produce identical requests whenever that
    happens, making the UI toggle a no-op.
    """
    parts = []
    if system_prompt:
        parts.append(system_prompt)
    if not think:
        parts.append(NO_THINK_SYSTEM_MESSAGE)
    if not parts:
        return history
    return [{"role": "system", "content": "\n\n".join(parts)}] + history


def accumulate_sentences(delta_iter):
    """Buffer a stream of text deltas and yield each completed sentence as
    soon as a sentence-ending boundary appears, flushing any remainder
    once the delta stream ends."""
    buf = ""
    for delta in delta_iter:
        buf += delta
        while True:
            match = _SENTENCE_BOUNDARY.search(buf)
            if not match:
                break
            sentence = buf[:match.start()].strip()
            buf = buf[match.end():]
            if sentence:
                yield sentence
    remainder = buf.strip()
    if remainder:
        yield remainder


def ollama_chat_stream(history: list, think: bool, system_prompt: str = "",
                       model: str = OLLAMA_MODEL,
                       base_url: str = DEFAULT_OLLAMA_BASE_URL):
    messages = build_chat_messages(history, think, system_prompt)
    resp = requests.post(
        f"{base_url}/api/chat",
        json={"model": model, "messages": messages, "think": think, "stream": True},
        timeout=120,
        stream=True,
    )
    if think and resp.status_code == 400 and "does not support thinking" in resp.text:
        # Some Ollama model imports (this GGUF included -- confirmed via
        # `ollama show`, which lists only tools/completion capabilities, no
        # thinking) hard-reject think=True even though the underlying model's
        # own chat template supports a reasoning mode. Retry as a normal
        # (non-thinking) streaming call rather than surfacing an error to
        # the user for what is, from their perspective, just flipping a UI
        # toggle. Messages are unchanged from the think=True attempt (no
        # suppression prompt), since the intent when think=True is "let it
        # reason", not suppress it.
        resp = requests.post(
            f"{base_url}/api/chat",
            json={"model": model, "messages": messages, "think": False, "stream": True},
            timeout=120,
            stream=True,
        )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content
        if chunk.get("done"):
            break


def make_gguf_chat_stream(repo_id: str = GGUF_REPO_ID, filename: str = GGUF_FILENAME):
    from llama_cpp import Llama

    llm = Llama.from_pretrained(repo_id=repo_id, filename=filename, n_ctx=4096, verbose=False)

    def chat_stream(history: list, think: bool, system_prompt: str = ""):
        messages = build_chat_messages(history, think, system_prompt)
        for chunk in llm.create_chat_completion(messages=messages, stream=True):
            content = chunk["choices"][0]["delta"].get("content")
            if content:
                yield content

    return chat_stream


def make_llm(ollama_model: str = OLLAMA_MODEL, gguf_repo_id: str = GGUF_REPO_ID,
            gguf_filename: str = GGUF_FILENAME,
            ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL):
    if check_ollama_reachable(ollama_base_url):
        def chat_stream(history: list, think: bool, system_prompt: str = ""):
            yield from ollama_chat_stream(
                history, think, system_prompt, model=ollama_model, base_url=ollama_base_url
            )
        return chat_stream
    gguf_chat_stream = make_gguf_chat_stream(gguf_repo_id, gguf_filename)

    def chat_stream(history: list, think: bool, system_prompt: str = ""):
        yield from gguf_chat_stream(history, think, system_prompt)
    return chat_stream
