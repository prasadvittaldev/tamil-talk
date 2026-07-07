# Web Search Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the tamil-talk assistant a `web_search` tool (backed by Tavily) it can call mid-conversation for current-information questions, gated behind a UI toggle, plus a layout fix so the header (now containing the waveform) and footer stay pinned while only the conversation scrolls.

**Architecture:** Native Ollama tool-calling, empirically validated against the real model/GGUF before this plan was written (see the design spec). Because Ollama's streaming API returns a tool-call turn as a single non-incremental chunk, `chat_stream`'s output changes from plain text strings to small tagged event dicts (`{"type": "content", "text": ...}` / `{"type": "searching", "query": ...}`) — the `{"type": "searching", ...}` event is yielded (pausing the generator) *before* the blocking Tavily call executes, which is what lets server.py relay it to the client in real time without any cross-thread callback machinery. `accumulate_sentences` is extended to pass non-content events through unchanged while still grouping content text into sentences exactly as before.

**Tech Stack:** Same as the rest of `tamil-talk` — FastAPI, `requests`, vanilla JS. No new Python dependencies (`tavily_search.py` only needs `requests`, already a dependency).

## Global Constraints

- Repo: `tamil-talk` only (github.com/prasadvittaldev/tamil-talk, `main` branch). The monorepo's `tamil_talkies/` is untouched.
- Web search only wires into the **Ollama path**. The GGUF-fallback path ignores `web_search`/`tavily_api_key` entirely — documented as an explicit limitation, not silently missing.
- Bounded to **2 search rounds per turn**; the 3rd (forced-final) call omits the `tools` parameter entirely.
- Tavily results: **top 3**, each formatted as `"<title>: <snippet, truncated to 300 chars>"`, joined with newlines.
- `tavily_search.web_search()` **never raises** — always returns a plain string (result text or an error string like `"search unavailable: ..."`).
- If the model's response contains more than one tool call at once, only the **first** is executed; the rest are ignored (not a silent bug — a stated choice).
- The Tavily API key comes from the **UI** (`localStorage`, sent per-turn in the `end` event as `tavily_api_key`) — there is no server-side env var for it.
- Persona input becomes a header button opening a `<dialog>` with a `<textarea>`, same wire format (`system_prompt` in the `end` event) as today — only the widget changes.
- No unit tests for WS orchestration or frontend — verified live, matching this project's established convention.

---

### Task 1: `tamil_talk/tavily_search.py` — Tavily search backend

**Files:**
- Create: `tamil_talk/tavily_search.py`
- Test: `tests/test_tavily_search.py`

**Interfaces:**
- Produces: `web_search(query: str, api_key: str) -> str` — never raises.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tavily_search.py`:

```python
from unittest.mock import patch, MagicMock

import requests

from tamil_talk.tavily_search import web_search


def _fake_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    if status_code == 200:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status_code}")
    return resp


def test_web_search_returns_error_string_when_api_key_missing():
    result = web_search("chennai weather", "")
    assert result == "search unavailable: no API key configured"


def test_web_search_formats_top_3_results_as_title_snippet_lines():
    data = {
        "results": [
            {"title": "Weather Today", "content": "Sunny, 32C, humidity 70%."},
            {"title": "Forecast Tomorrow", "content": "Rain expected in the evening."},
            {"title": "Extended Outlook", "content": "Cooler by the weekend."},
            {"title": "Should be dropped", "content": "This is the 4th result."},
        ]
    }
    fake_resp = _fake_response(data)
    with patch("tamil_talk.tavily_search.requests.post", return_value=fake_resp) as mock_post:
        result = web_search("chennai weather", "real-key")
    assert result == (
        "Weather Today: Sunny, 32C, humidity 70%.\n"
        "Forecast Tomorrow: Rain expected in the evening.\n"
        "Extended Outlook: Cooler by the weekend."
    )
    mock_post.assert_called_once_with(
        "https://api.tavily.com/search",
        json={"api_key": "real-key", "query": "chennai weather", "max_results": 3},
        timeout=10,
    )


def test_web_search_truncates_long_snippets_to_300_chars():
    long_content = "x" * 500
    data = {"results": [{"title": "T", "content": long_content}]}
    fake_resp = _fake_response(data)
    with patch("tamil_talk.tavily_search.requests.post", return_value=fake_resp):
        result = web_search("q", "key")
    assert result == "T: " + ("x" * 300)


def test_web_search_returns_error_string_on_empty_results():
    fake_resp = _fake_response({"results": []})
    with patch("tamil_talk.tavily_search.requests.post", return_value=fake_resp):
        result = web_search("obscure query", "key")
    assert result == "search unavailable: no results found"


def test_web_search_returns_error_string_on_network_error():
    with patch("tamil_talk.tavily_search.requests.post", side_effect=requests.ConnectionError("boom")):
        result = web_search("q", "key")
    assert result.startswith("search unavailable:")


def test_web_search_returns_error_string_on_http_error():
    fake_resp = _fake_response({}, status_code=401)
    with patch("tamil_talk.tavily_search.requests.post", return_value=fake_resp):
        result = web_search("q", "bad-key")
    assert result.startswith("search unavailable:")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk && /home/prasad/Desktop/suryantts/.venv-parler/bin/python -m pytest tests/test_tavily_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tamil_talk.tavily_search'`

- [ ] **Step 3: Implement `tamil_talk/tavily_search.py`**

```python
"""Tavily web search backend for the assistant's web_search tool. Never
raises -- any failure (missing key, network error, API error, no results)
returns a plain error string as the "result", so the caller can feed it
back to the LLM as a tool result and let the model answer from its own
knowledge instead of the whole turn failing.
"""
import requests

TAVILY_API_URL = "https://api.tavily.com/search"
MAX_RESULTS = 3
SNIPPET_CHARS = 300


def web_search(query: str, api_key: str) -> str:
    if not api_key:
        return "search unavailable: no API key configured"

    try:
        resp = requests.post(
            TAVILY_API_URL,
            json={"api_key": api_key, "query": query, "max_results": MAX_RESULTS},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return f"search unavailable: {e}"

    results = data.get("results", [])[:MAX_RESULTS]
    if not results:
        return "search unavailable: no results found"

    lines = []
    for r in results:
        title = (r.get("title") or "").strip()
        snippet = (r.get("content") or "").strip()[:SNIPPET_CHARS]
        if title and snippet:
            lines.append(f"{title}: {snippet}")
        elif title or snippet:
            lines.append(title or snippet)
    return "\n".join(lines) if lines else "search unavailable: no results found"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk && /home/prasad/Desktop/suryantts/.venv-parler/bin/python -m pytest tests/test_tavily_search.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk
git add tamil_talk/tavily_search.py tests/test_tavily_search.py
git commit -m "feat: tavily_search.py -- web search backend for the web_search tool"
```

---

### Task 2: `tamil_talk/llm.py` — `ollama_chat_stream` gains an optional `tools` param

**Files:**
- Modify: `tamil_talk/llm.py`
- Modify: `tests/test_llm.py`

**Interfaces:**
- Produces: `ollama_chat_stream(history, think, system_prompt="", model=OLLAMA_MODEL, base_url=DEFAULT_OLLAMA_BASE_URL, tools=None)` — same as today when `tools` is omitted (byte-identical request payload, all 23 existing tests must still pass unmodified). When `tools` is provided, it's included in the request JSON. When the model calls a tool instead of producing text, the generator yields **no content** and its `return` value (accessible via `StopIteration.value`, or `x = yield from ollama_chat_stream(...)` in a caller) is the `tool_calls` list from that response; a normal text-only response's `return` value is `None`.

This task is purely additive — the existing 23 tests in `tests/test_llm.py` must keep passing with zero changes, since `tools=None` (the default) produces an identical request to today.

- [ ] **Step 1: Write the new failing tests**

Add to `tests/test_llm.py` (after the existing `ollama_chat_stream` tests, before `test_build_chat_messages_passes_through_when_think_true_and_no_persona`):

```python
def test_ollama_chat_stream_includes_tools_in_payload_when_provided():
    lines = [json.dumps({"message": {"content": "hi"}, "done": True})]
    fake_resp = _fake_ndjson_response(lines)
    tools = [{"type": "function", "function": {"name": "web_search"}}]
    with patch("tamil_talk.llm.requests.post", return_value=fake_resp) as mock_post:
        list(ollama_chat_stream([], think=False, tools=tools, model="m", base_url="http://x:1"))
    assert mock_post.call_args.kwargs["json"]["tools"] == tools


def test_ollama_chat_stream_omits_tools_key_when_not_provided():
    lines = [json.dumps({"message": {"content": "hi"}, "done": True})]
    fake_resp = _fake_ndjson_response(lines)
    with patch("tamil_talk.llm.requests.post", return_value=fake_resp) as mock_post:
        list(ollama_chat_stream([], think=False, model="m", base_url="http://x:1"))
    assert "tools" not in mock_post.call_args.kwargs["json"]


def test_ollama_chat_stream_returns_tool_calls_via_stopiteration_and_yields_no_content():
    tool_calls = [{"function": {"name": "web_search", "arguments": {"query": "weather"}}}]
    lines = [json.dumps({"message": {"content": "", "tool_calls": tool_calls}, "done": True})]
    fake_resp = _fake_ndjson_response(lines)
    with patch("tamil_talk.llm.requests.post", return_value=fake_resp):
        gen = ollama_chat_stream([], think=False, model="m", base_url="http://x:1")
        collected = []
        returned = "not set"
        try:
            while True:
                collected.append(next(gen))
        except StopIteration as stop:
            returned = stop.value
    assert collected == []
    assert returned == tool_calls


def test_ollama_chat_stream_returns_none_for_a_normal_text_response():
    lines = [json.dumps({"message": {"content": "hi"}, "done": True})]
    fake_resp = _fake_ndjson_response(lines)
    with patch("tamil_talk.llm.requests.post", return_value=fake_resp):
        gen = ollama_chat_stream([], think=False, model="m", base_url="http://x:1")
        collected = []
        returned = "not set"
        try:
            while True:
                collected.append(next(gen))
        except StopIteration as stop:
            returned = stop.value
    assert collected == ["hi"]
    assert returned is None
```

- [ ] **Step 2: Run tests to verify the new ones fail, old ones still pass**

Run: `cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk && /home/prasad/Desktop/suryantts/.venv-parler/bin/python -m pytest tests/test_llm.py -v -k "ollama_chat_stream"`
Expected: the 4 new tests FAIL (`tools` isn't a parameter yet; `TypeError: unexpected keyword argument 'tools'`), the 6 pre-existing `ollama_chat_stream` tests still PASS.

- [ ] **Step 3: Update `ollama_chat_stream` in `tamil_talk/llm.py`**

Replace the existing `ollama_chat_stream` function with:

```python
def ollama_chat_stream(history: list, think: bool, system_prompt: str = "",
                       model: str = OLLAMA_MODEL,
                       base_url: str = DEFAULT_OLLAMA_BASE_URL,
                       tools: list = None):
    """Yields content text deltas. If the model calls a tool instead of
    producing text, no content is yielded and this generator's return value
    (accessible via StopIteration.value, or `x = yield from ollama_chat_stream(...)`
    in a caller) is the tool_calls list from that response; a normal
    text-only response returns None."""
    messages = build_chat_messages(history, think, system_prompt)
    payload = {"model": model, "messages": messages, "think": think, "stream": True}
    if tools:
        payload["tools"] = tools
    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=120, stream=True)
    if think and resp.status_code == 400 and "does not support thinking" in resp.text:
        # See build_chat_messages' docstring / the non-streaming history in
        # this file: some Ollama model imports hard-reject think=True even
        # though the underlying model's own chat template supports a
        # reasoning mode. Retry as a normal (non-thinking) streaming call
        # rather than surfacing an error to the user for what is, from
        # their perspective, just flipping a UI toggle. Messages/tools are
        # unchanged from the think=True attempt, since the intent when
        # think=True is "let it reason", not suppress it.
        payload = dict(payload, think=False)
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=120, stream=True)
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        tool_calls = chunk.get("message", {}).get("tool_calls")
        if tool_calls:
            return tool_calls
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content
        if chunk.get("done"):
            break
    return None
```

- [ ] **Step 4: Run the full test file to verify everything passes**

Run: `cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk && /home/prasad/Desktop/suryantts/.venv-parler/bin/python -m pytest tests/test_llm.py -v`
Expected: PASS (27 tests: the pre-existing 23 + 4 new).

- [ ] **Step 5: Commit**

```bash
cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk
git add tamil_talk/llm.py tests/test_llm.py
git commit -m "feat: llm.py -- ollama_chat_stream() gains an optional tools param + tool_calls signaling"
```

---

### Task 3: `tamil_talk/llm.py` — `WEB_SEARCH_TOOL` schema + `ollama_chat_stream_with_tools` orchestration

**Files:**
- Modify: `tamil_talk/llm.py`
- Modify: `tests/test_llm.py`

**Interfaces:**
- Consumes: `ollama_chat_stream(...)` (Task 2), `tamil_talk.tavily_search.web_search(query, api_key) -> str` (Task 1).
- Produces: `WEB_SEARCH_TOOL` (a module-level dict, the tool's JSON schema). `ollama_chat_stream_with_tools(history, think, system_prompt, web_search, tavily_api_key, model=OLLAMA_MODEL, base_url=DEFAULT_OLLAMA_BASE_URL, max_search_rounds=2) -> Iterator[dict]` — yields `{"type": "content", "text": str}` and `{"type": "searching", "query": str}` events.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_llm.py` (after the `ollama_chat_stream` tests added in Task 2):

```python
def _tool_call_gen(tool_calls):
    return tool_calls
    yield  # pragma: no cover -- unreachable; makes this a generator function whose StopIteration carries tool_calls


def _content_gen(*texts):
    for t in texts:
        yield t


def test_with_tools_no_search_bypasses_tool_orchestration_entirely():
    with patch("tamil_talk.llm.ollama_chat_stream", return_value=iter(["Hel", "lo"])) as mock_stream:
        events = list(ollama_chat_stream_with_tools(
            [], think=False, system_prompt="", web_search=False, tavily_api_key="",
            model="m", base_url="http://x:1"))
    assert events == [{"type": "content", "text": "Hel"}, {"type": "content", "text": "lo"}]
    mock_stream.assert_called_once_with([], False, "", model="m", base_url="http://x:1")


def test_with_tools_direct_answer_no_tool_call():
    with patch("tamil_talk.llm.ollama_chat_stream", return_value=_content_gen("Hello", " world.")) as mock_stream:
        events = list(ollama_chat_stream_with_tools(
            [], think=False, system_prompt="", web_search=True, tavily_api_key="key",
            model="m", base_url="http://x:1"))
    assert events == [{"type": "content", "text": "Hello"}, {"type": "content", "text": " world."}]
    assert mock_stream.call_count == 1
    assert mock_stream.call_args.kwargs["tools"] == [WEB_SEARCH_TOOL]


def test_with_tools_executes_search_and_reinvokes():
    tool_calls = [{"function": {"name": "web_search", "arguments": {"query": "chennai weather"}}}]
    with patch("tamil_talk.llm.ollama_chat_stream",
               side_effect=[_tool_call_gen(tool_calls), _content_gen("It is sunny.")]) as mock_stream, \
         patch("tamil_talk.llm.tavily_search.web_search", return_value="Chennai: sunny, 30C") as mock_search:
        events = list(ollama_chat_stream_with_tools(
            [{"role": "user", "content": "hi"}], think=False, system_prompt="", web_search=True,
            tavily_api_key="key", model="m", base_url="http://x:1"))

    assert {"type": "searching", "query": "chennai weather"} in events
    assert {"type": "content", "text": "It is sunny."} in events
    mock_search.assert_called_once_with("chennai weather", "key")
    assert mock_stream.call_count == 2
    second_call_history = mock_stream.call_args_list[1].args[0]
    assert second_call_history[-2] == {"role": "assistant", "content": "", "tool_calls": tool_calls}
    assert second_call_history[-1] == {"role": "tool", "content": "Chennai: sunny, 30C"}


def test_with_tools_caps_at_max_search_rounds_then_forces_final_answer_without_tools():
    tool_calls = [{"function": {"name": "web_search", "arguments": {"query": "q1"}}}]
    with patch("tamil_talk.llm.ollama_chat_stream",
               side_effect=[_tool_call_gen(tool_calls), _tool_call_gen(tool_calls), _content_gen("Final answer.")]) as mock_stream, \
         patch("tamil_talk.llm.tavily_search.web_search", return_value="result"):
        events = list(ollama_chat_stream_with_tools(
            [], think=False, system_prompt="", web_search=True, tavily_api_key="key",
            model="m", base_url="http://x:1", max_search_rounds=2))

    assert mock_stream.call_count == 3
    assert mock_stream.call_args_list[0].kwargs["tools"] == [WEB_SEARCH_TOOL]
    assert mock_stream.call_args_list[1].kwargs["tools"] == [WEB_SEARCH_TOOL]
    assert mock_stream.call_args_list[2].kwargs["tools"] is None
    assert {"type": "content", "text": "Final answer."} in events


def test_with_tools_handles_missing_query_gracefully():
    tool_calls = [{"function": {"name": "web_search", "arguments": {}}}]  # no "query" key
    with patch("tamil_talk.llm.ollama_chat_stream",
               side_effect=[_tool_call_gen(tool_calls), _content_gen("I could not search.")]), \
         patch("tamil_talk.llm.tavily_search.web_search") as mock_search:
        events = list(ollama_chat_stream_with_tools(
            [], think=False, system_prompt="", web_search=True, tavily_api_key="key",
            model="m", base_url="http://x:1"))
    mock_search.assert_not_called()
    assert any(e.get("type") == "searching" and e.get("query") == "" for e in events)


def test_with_tools_only_executes_first_tool_call_when_multiple_present():
    tool_calls = [
        {"function": {"name": "web_search", "arguments": {"query": "first"}}},
        {"function": {"name": "web_search", "arguments": {"query": "second"}}},
    ]
    with patch("tamil_talk.llm.ollama_chat_stream",
               side_effect=[_tool_call_gen(tool_calls), _content_gen("done")]), \
         patch("tamil_talk.llm.tavily_search.web_search", return_value="r") as mock_search:
        list(ollama_chat_stream_with_tools(
            [], think=False, system_prompt="", web_search=True, tavily_api_key="key",
            model="m", base_url="http://x:1"))
    mock_search.assert_called_once_with("first", "key")
```

Update the import line at the top of `tests/test_llm.py` to add the new names:

```python
from tamil_talk.llm import (
    check_ollama_reachable, append_user_turn, append_assistant_turn,
    build_chat_messages, accumulate_sentences, ollama_chat_stream,
    ollama_chat_stream_with_tools, WEB_SEARCH_TOOL,
    NO_THINK_SYSTEM_MESSAGE, make_llm,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk && /home/prasad/Desktop/suryantts/.venv-parler/bin/python -m pytest tests/test_llm.py -v -k "with_tools"`
Expected: FAIL with `ImportError: cannot import name 'ollama_chat_stream_with_tools'`

- [ ] **Step 3: Implement `WEB_SEARCH_TOOL` and `ollama_chat_stream_with_tools` in `tamil_talk/llm.py`**

Add `import tamil_talk.tavily_search as tavily_search` to the imports at the top of the file (alongside `import json`, `import re`, `import requests`).

Add this after `ollama_chat_stream` (from Task 2) and before `make_gguf_chat_stream`:

```python
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current, up-to-date information not known "
            "to the model, such as weather, news, or recent events."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
        },
    },
}


def ollama_chat_stream_with_tools(history: list, think: bool, system_prompt: str,
                                  web_search: bool, tavily_api_key: str,
                                  model: str = OLLAMA_MODEL,
                                  base_url: str = DEFAULT_OLLAMA_BASE_URL,
                                  max_search_rounds: int = 2):
    """Wraps ollama_chat_stream with tool-calling: when web_search is True,
    offers the web_search tool and, if the model calls it, executes the
    search and re-invokes the model with the result appended to history --
    bounded to max_search_rounds rounds (the final, forced call omits the
    tools parameter entirely so the model must answer with what it has).
    Yields {"type": "content", "text": str} for text deltas and
    {"type": "searching", "query": str} right before each search executes
    -- yielded (not just returned) specifically so a caller iterating this
    generator one item at a time sees it in real time, before the blocking
    Tavily call runs, not after.
    """
    if not web_search:
        for text in ollama_chat_stream(history, think, system_prompt, model=model, base_url=base_url):
            yield {"type": "content", "text": text}
        return

    current_history = history
    round_num = 0
    while True:
        use_tools = [WEB_SEARCH_TOOL] if round_num < max_search_rounds else None
        gen = ollama_chat_stream(current_history, think, system_prompt,
                                 tools=use_tools, model=model, base_url=base_url)
        tool_calls = None
        while True:
            try:
                text = next(gen)
            except StopIteration as stop:
                tool_calls = stop.value
                break
            yield {"type": "content", "text": text}

        if not tool_calls:
            return

        call = tool_calls[0]  # only the first is executed if the model returns more than one
        query = call.get("function", {}).get("arguments", {}).get("query", "")
        yield {"type": "searching", "query": query}
        if query:
            result = tavily_search.web_search(query, tavily_api_key)
        else:
            result = "search unavailable: no query provided"
        current_history = current_history + [
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
            {"role": "tool", "content": result},
        ]
        round_num += 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk && /home/prasad/Desktop/suryantts/.venv-parler/bin/python -m pytest tests/test_llm.py -v -k "with_tools"`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk
git add tamil_talk/llm.py tests/test_llm.py
git commit -m "feat: llm.py -- WEB_SEARCH_TOOL schema + ollama_chat_stream_with_tools orchestration"
```

---

### Task 4: `tamil_talk/llm.py` — `accumulate_sentences` handles tagged content/control events

**Files:**
- Modify: `tamil_talk/llm.py`
- Modify: `tests/test_llm.py`

**Interfaces:**
- Produces: `accumulate_sentences(event_iter: Iterator[dict]) -> Iterator[str | dict]` — **input contract changes** from "iterator of plain strings" to "iterator of `{\"type\": ..., ...}` dicts" (matching `ollama_chat_stream_with_tools`'s output from Task 3). For `{"type": "content", "text": ...}` events, groups text into completed sentences exactly as before (still yielded as plain strings). Any other event type is yielded through **unchanged**, immediately (no buffering delay) — flushing whatever partial sentence text is currently buffered first, so it's never silently dropped or reordered around a passthrough event.

This is a deliberate breaking change to `accumulate_sentences`'s existing (already-shipped, in this same repo) input format — all 4 of its existing tests are rewritten to use the new dict-event input shape, not left as-is.

- [ ] **Step 1: Replace the existing `accumulate_sentences` tests, add new ones**

In `tests/test_llm.py`, find and replace these four existing tests:

```python
def test_accumulate_sentences_yields_completed_sentences_as_deltas_arrive():
    deltas = ["Hello", " world.", " How are you?", " I am fine."]
    result = list(accumulate_sentences(iter(deltas)))
    assert result == ["Hello world.", "How are you?", "I am fine."]


def test_accumulate_sentences_flushes_remainder_with_no_trailing_punctuation():
    deltas = ["Hello world.", " No period at the end"]
    result = list(accumulate_sentences(iter(deltas)))
    assert result == ["Hello world.", "No period at the end"]


def test_accumulate_sentences_handles_multiple_sentences_in_one_delta():
    deltas = ["First. Second. Third."]
    result = list(accumulate_sentences(iter(deltas)))
    assert result == ["First.", "Second.", "Third."]


def test_accumulate_sentences_empty_input_yields_nothing():
    assert list(accumulate_sentences(iter([]))) == []
```

with:

```python
def _content_events(*texts):
    return [{"type": "content", "text": t} for t in texts]


def test_accumulate_sentences_yields_completed_sentences_as_deltas_arrive():
    events = _content_events("Hello", " world.", " How are you?", " I am fine.")
    result = list(accumulate_sentences(iter(events)))
    assert result == ["Hello world.", "How are you?", "I am fine."]


def test_accumulate_sentences_flushes_remainder_with_no_trailing_punctuation():
    events = _content_events("Hello world.", " No period at the end")
    result = list(accumulate_sentences(iter(events)))
    assert result == ["Hello world.", "No period at the end"]


def test_accumulate_sentences_handles_multiple_sentences_in_one_delta():
    events = _content_events("First. Second. Third.")
    result = list(accumulate_sentences(iter(events)))
    assert result == ["First.", "Second.", "Third."]


def test_accumulate_sentences_empty_input_yields_nothing():
    assert list(accumulate_sentences(iter([]))) == []


def test_accumulate_sentences_passes_through_non_content_events_immediately():
    events = [
        {"type": "searching", "query": "weather"},
        {"type": "content", "text": "It is sunny."},
    ]
    result = list(accumulate_sentences(iter(events)))
    assert result == [{"type": "searching", "query": "weather"}, "It is sunny."]


def test_accumulate_sentences_flushes_partial_buffer_before_passthrough_event():
    events = [
        {"type": "content", "text": "partial without punctuation"},
        {"type": "searching", "query": "q"},
        {"type": "content", "text": "Final sentence."},
    ]
    result = list(accumulate_sentences(iter(events)))
    assert result == ["partial without punctuation", {"type": "searching", "query": "q"}, "Final sentence."]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk && /home/prasad/Desktop/suryantts/.venv-parler/bin/python -m pytest tests/test_llm.py -v -k "accumulate_sentences"`
Expected: FAIL — `accumulate_sentences` still expects plain strings (`buf += delta` will raise `TypeError` when `delta` is a dict).

- [ ] **Step 3: Update `accumulate_sentences` in `tamil_talk/llm.py`**

Replace the existing `accumulate_sentences` function with:

```python
def accumulate_sentences(event_iter):
    """Buffer a stream of {"type": "content", "text": str} events and yield
    each completed sentence as soon as a sentence-ending boundary appears,
    flushing any remainder once the event stream ends. Any event whose
    type isn't "content" (e.g. {"type": "searching", ...}) is passed
    through unchanged, immediately -- first flushing whatever partial
    sentence text is currently buffered so it's never silently dropped or
    reordered around the passthrough event."""
    buf = ""
    for evt in event_iter:
        if evt["type"] != "content":
            if buf.strip():
                yield buf.strip()
                buf = ""
            yield evt
            continue
        buf += evt["text"]
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk && /home/prasad/Desktop/suryantts/.venv-parler/bin/python -m pytest tests/test_llm.py -v -k "accumulate_sentences"`
Expected: PASS (6 tests: 4 rewritten + 2 new)

- [ ] **Step 5: Commit**

```bash
cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk
git add tamil_talk/llm.py tests/test_llm.py
git commit -m "feat: llm.py -- accumulate_sentences() handles tagged content/control events"
```

---

### Task 5: `tamil_talk/llm.py` — `make_llm()` returns a web-search-aware, uniformly-tagged `chat_stream`

**Files:**
- Modify: `tamil_talk/llm.py`
- Modify: `tests/test_llm.py`

**Interfaces:**
- Consumes: `ollama_chat_stream_with_tools` (Task 3), `make_gguf_chat_stream` (unchanged), `accumulate_sentences` (Task 4).
- Produces: `make_llm(...) -> chat_stream` where `chat_stream(history, think, system_prompt="", web_search=False, tavily_api_key="") -> Iterator[dict]` — yields `{"type": "content"|"searching", ...}` events on **both** branches (the GGUF-fallback branch never emits `"searching"`, since it ignores `web_search`/`tavily_api_key` entirely, but wraps its plain-string output in the same `{"type": "content", "text": ...}` shape for a uniform contract).

- [ ] **Step 1: Replace the `make_llm`-referencing tests**

In `tests/test_llm.py`, find and replace these three existing tests:

```python
def test_make_llm_uses_ollama_when_reachable():
    with patch("tamil_talk.llm.check_ollama_reachable", return_value=True), \
         patch("tamil_talk.llm.ollama_chat_stream", return_value=iter(["ollama", " reply"])) as mock_chat, \
         patch("tamil_talk.llm.make_gguf_chat_stream") as mock_gguf:
        chat_stream = make_llm(ollama_model="m", ollama_base_url="http://x:1")
        result = list(chat_stream([{"role": "user", "content": "hi"}], True))
    assert result == ["ollama", " reply"]
    mock_chat.assert_called_once_with(
        [{"role": "user", "content": "hi"}], True, "", model="m", base_url="http://x:1"
    )
    mock_gguf.assert_not_called()


def test_make_llm_falls_back_to_gguf_when_ollama_unreachable():
    fake_gguf_chat_stream = MagicMock(return_value=iter(["gguf", " reply"]))
    with patch("tamil_talk.llm.check_ollama_reachable", return_value=False), \
         patch("tamil_talk.llm.make_gguf_chat_stream", return_value=fake_gguf_chat_stream) as mock_make_gguf:
        chat_stream = make_llm(gguf_repo_id="r", gguf_filename="f")
        result = list(chat_stream([{"role": "user", "content": "hi"}], False))
    assert result == ["gguf", " reply"]
    mock_make_gguf.assert_called_once_with("r", "f")
    fake_gguf_chat_stream.assert_called_once_with([{"role": "user", "content": "hi"}], False, "")


def test_make_llm_passes_system_prompt_through_to_ollama_stream():
    with patch("tamil_talk.llm.check_ollama_reachable", return_value=True), \
         patch("tamil_talk.llm.ollama_chat_stream", return_value=iter(["hi"])) as mock_chat:
        chat_stream = make_llm(ollama_model="m", ollama_base_url="http://x:1")
        list(chat_stream([{"role": "user", "content": "hi"}], False, "You are a pirate."))
    mock_chat.assert_called_once_with(
        [{"role": "user", "content": "hi"}], False, "You are a pirate.", model="m", base_url="http://x:1"
    )
```

with:

```python
def test_make_llm_uses_ollama_when_reachable():
    with patch("tamil_talk.llm.check_ollama_reachable", return_value=True), \
         patch("tamil_talk.llm.ollama_chat_stream_with_tools",
               return_value=iter([{"type": "content", "text": "ollama"}, {"type": "content", "text": " reply"}])) as mock_chat, \
         patch("tamil_talk.llm.make_gguf_chat_stream") as mock_gguf:
        chat_stream = make_llm(ollama_model="m", ollama_base_url="http://x:1")
        result = list(chat_stream([{"role": "user", "content": "hi"}], True))
    assert result == [{"type": "content", "text": "ollama"}, {"type": "content", "text": " reply"}]
    mock_chat.assert_called_once_with(
        [{"role": "user", "content": "hi"}], True, "", False, "", model="m", base_url="http://x:1"
    )
    mock_gguf.assert_not_called()


def test_make_llm_falls_back_to_gguf_when_ollama_unreachable():
    fake_gguf_chat_stream = MagicMock(return_value=iter(["gguf", " reply"]))
    with patch("tamil_talk.llm.check_ollama_reachable", return_value=False), \
         patch("tamil_talk.llm.make_gguf_chat_stream", return_value=fake_gguf_chat_stream) as mock_make_gguf:
        chat_stream = make_llm(gguf_repo_id="r", gguf_filename="f")
        result = list(chat_stream([{"role": "user", "content": "hi"}], False))
    assert result == [{"type": "content", "text": "gguf"}, {"type": "content", "text": " reply"}]
    mock_make_gguf.assert_called_once_with("r", "f")
    fake_gguf_chat_stream.assert_called_once_with([{"role": "user", "content": "hi"}], False, "")


def test_make_llm_passes_system_prompt_and_search_params_through_to_ollama():
    with patch("tamil_talk.llm.check_ollama_reachable", return_value=True), \
         patch("tamil_talk.llm.ollama_chat_stream_with_tools",
               return_value=iter([{"type": "content", "text": "hi"}])) as mock_chat:
        chat_stream = make_llm(ollama_model="m", ollama_base_url="http://x:1")
        list(chat_stream([{"role": "user", "content": "hi"}], False, "You are a pirate.", True, "tavily-key"))
    mock_chat.assert_called_once_with(
        [{"role": "user", "content": "hi"}], False, "You are a pirate.", True, "tavily-key",
        model="m", base_url="http://x:1",
    )


def test_make_llm_gguf_fallback_ignores_web_search_params():
    fake_gguf_chat_stream = MagicMock(return_value=iter(["gguf reply"]))
    with patch("tamil_talk.llm.check_ollama_reachable", return_value=False), \
         patch("tamil_talk.llm.make_gguf_chat_stream", return_value=fake_gguf_chat_stream):
        chat_stream = make_llm(gguf_repo_id="r", gguf_filename="f")
        result = list(chat_stream([{"role": "user", "content": "hi"}], False, "", True, "some-key"))
    assert result == [{"type": "content", "text": "gguf reply"}]
    fake_gguf_chat_stream.assert_called_once_with([{"role": "user", "content": "hi"}], False, "")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk && /home/prasad/Desktop/suryantts/.venv-parler/bin/python -m pytest tests/test_llm.py -v -k "make_llm"`
Expected: FAIL — `make_llm`'s current implementation still calls the old `ollama_chat_stream` directly and yields plain strings, not dicts; doesn't accept `web_search`/`tavily_api_key`.

- [ ] **Step 3: Update `make_llm` in `tamil_talk/llm.py`**

Replace the existing `make_llm` function with:

```python
def make_llm(ollama_model: str = OLLAMA_MODEL, gguf_repo_id: str = GGUF_REPO_ID,
            gguf_filename: str = GGUF_FILENAME,
            ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL):
    if check_ollama_reachable(ollama_base_url):
        def chat_stream(history: list, think: bool, system_prompt: str = "",
                        web_search: bool = False, tavily_api_key: str = ""):
            yield from ollama_chat_stream_with_tools(
                history, think, system_prompt, web_search, tavily_api_key,
                model=ollama_model, base_url=ollama_base_url,
            )
        return chat_stream
    gguf_chat_stream = make_gguf_chat_stream(gguf_repo_id, gguf_filename)

    def chat_stream(history: list, think: bool, system_prompt: str = "",
                    web_search: bool = False, tavily_api_key: str = ""):
        # GGUF fallback doesn't support web search -- web_search/tavily_api_key
        # are accepted (so callers don't need to know which backend is
        # active) but silently ignored, matching this repo's documented
        # scope for the fallback path.
        for text in gguf_chat_stream(history, think, system_prompt):
            yield {"type": "content", "text": text}
    return chat_stream
```

- [ ] **Step 4: Run the full test file to verify everything passes**

Run: `cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk && /home/prasad/Desktop/suryantts/.venv-parler/bin/python -m pytest tests/ -v`
Expected: PASS (all tests — 6 `tavily_search` + everything in `test_llm.py`: 3 `check_ollama_reachable` + 3 `append_*_turn` + 6 `accumulate_sentences` + 4 `build_chat_messages` + 6 `ollama_chat_stream` + 4 `tools`-param tests from Task 2 + 7 `with_tools` + 4 `make_llm` = 43 tests total in `test_llm.py`, 49 overall).

- [ ] **Step 5: Commit**

```bash
cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk
git add tamil_talk/llm.py tests/test_llm.py
git commit -m "feat: llm.py -- make_llm() returns a web-search-aware, uniformly-tagged chat_stream"
```

---

### Task 6: `tamil_talk/server.py` — WS protocol wiring for web search

**Files:**
- Modify: `tamil_talk/server.py`

**Interfaces:**
- Consumes: `make_llm()`'s `chat_stream(history, think, system_prompt="", web_search=False, tavily_api_key="") -> Iterator[dict]` and `accumulate_sentences(event_iter) -> Iterator[str | dict]` (both from Task 5/4).
- Produces: updated `/talk` WS protocol — client `end` event gains `web_search: bool` and `tavily_api_key: str` (both optional, default `False`/`""`); server gains `{"event": "searching", "query": str}`, sent only when the model actually calls the tool.

No unit tests for this task (async WS orchestration, matching this repo's established convention) — verified live in Task 10.

- [ ] **Step 1: Update the `/talk` handler**

In `tamil_talk/server.py`, inside the `talk` websocket handler, update the `end`-event parsing block:

```python
                buf = bytearray()
                think = False
                sample_rate = 16000
                system_prompt = ""
                web_search = False
                tavily_api_key = ""
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
                            web_search = bool(evt.get("web_search", False))
                            tavily_api_key = str(evt.get("tavily_api_key", ""))
                            break
                    else:
                        return  # client disconnected
```

Then update the sentence/audio loop (replacing the existing `sentence_iter = accumulate_sentences(chat_stream(history, think, system_prompt))` and the `async for sentence in _aiter_sync(sentence_iter):` block):

```python
                reply_sentences = []
                audio_started = False
                sentence_iter = accumulate_sentences(
                    chat_stream(history, think, system_prompt, web_search, tavily_api_key)
                )
                index = 0
                async for item in _aiter_sync(sentence_iter):
                    if isinstance(item, dict):
                        await ws.send_text(json.dumps({"event": "searching", "query": item["query"]}))
                        continue
                    sentence = item
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
```

The rest of the handler (transcript send, history bookkeeping, `done`/`error` events) is unchanged.

- [ ] **Step 2: Verify the file still parses**

```bash
cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk
/home/prasad/Desktop/suryantts/.venv-parler/bin/python -m py_compile tamil_talk/server.py && echo "compiles OK"
```

Expected: `compiles OK`.

- [ ] **Step 3: Commit**

```bash
cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk
git add tamil_talk/server.py
git commit -m "feat: server.py -- web_search/tavily_api_key in end event, searching event forwarded"
```

---

### Task 7: `tamil_talk/static/index.html` — layout fix + Web search checkbox + persona/search-settings modals

**Files:**
- Modify: `tamil_talk/static/index.html`

**Interfaces:**
- Produces: DOM elements Task 8 (`app.js`) binds to, in addition to the existing ones (`#mic-btn`, `#think-toggle`, `#status`, `#conversation`, `#waveform`): `#web-search-toggle` (checkbox), `#persona-btn` (button), `#persona-dialog` (`<dialog>`), `#persona-textarea`, `#persona-save-btn`, `#persona-cancel-btn`, `#search-settings-btn` (button), `#search-settings-dialog` (`<dialog>`), `#tavily-key-input`, `#search-settings-save-btn`, `#search-settings-cancel-btn`.

This task also fixes a layout defect surfaced while designing this feature: `main`'s flex/overflow setup doesn't correctly constrain scrolling to just the conversation area (missing `min-height: 0` on the flex child — a classic flexbox gotcha where an overflowing flex item refuses to shrink below its content's natural height unless told to), and the waveform moves into the header so it's part of the "always visible, never scrolls away" zone. `scrollToBottom()`'s actual bug (scrolling `#conversation`, which has no `overflow` set, instead of `main`, which is the real scroll container) is fixed in Task 8, but this task's CSS changes are what make that fix meaningful (giving `main` an actual bounded, scrollable box).

- [ ] **Step 1: Replace the file contents**

```html
<!doctype html>
<html lang="ta">
<head>
  <meta charset="utf-8">
  <title>tamil-talk</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    html, body { height: 100%; margin: 0; }
    body {
      font-family: sans-serif; max-width: 640px; margin: 0 auto; padding: 0 1rem;
      display: flex; flex-direction: column; height: 100vh;
    }
    header { flex: 0 0 auto; padding: 1rem 0; border-bottom: 1px solid #ddd; }
    header h1 { margin: 0 0 0.5rem 0; font-size: 1.4rem; }
    header .controls { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; margin-top: 0.5rem; }
    #mic-btn {
      font-size: 2rem; width: 4rem; height: 4rem; border-radius: 50%; border: none;
      background: #d33; color: white; cursor: pointer;
    }
    #mic-btn.recording { background: #3a3; }
    #mic-btn:disabled { background: #999; cursor: not-allowed; }
    #status { color: #666; margin: 0.5rem 0; }
    #waveform {
      display: block; width: 140px; height: 140px; max-width: 100%;
      margin: 0.5rem auto 0; background: #10151a; border-radius: 50%;
    }
    main {
      flex: 1 1 auto; min-height: 0; overflow-y: auto; padding: 1rem 0;
    }
    .turn { margin: 1rem 0; padding: 0.5rem; border-left: 3px solid #ccc; }
    .turn.user { border-color: #06c; }
    .turn.assistant { border-color: #3a3; }
    .turn.assistant span.speaking { background: #dfd; font-weight: bold; }
    label { display: inline-flex; align-items: center; gap: 0.4rem; }
    footer { flex: 0 0 auto; padding: 1rem 0; border-top: 1px solid #ddd; color: #666; font-size: 0.9rem; }
    footer a { color: #06c; }
    dialog { border: none; border-radius: 6px; padding: 1.5rem; max-width: 30rem; width: 90%; }
    dialog::backdrop { background: rgba(0, 0, 0, 0.4); }
    dialog textarea, dialog input[type="password"] { width: 100%; box-sizing: border-box; padding: 0.5rem; margin: 0.5rem 0 1rem; }
    dialog textarea { min-height: 8rem; font-family: inherit; }
    dialog .dialog-actions { display: flex; justify-content: flex-end; gap: 0.5rem; }
    dialog p.hint { font-size: 0.85rem; color: #666; }
  </style>
</head>
<body>
  <header>
    <h1>tamil-talk</h1>
    <canvas id="waveform"></canvas>
    <div class="controls">
      <label><input type="checkbox" id="think-toggle"> Think mode</label>
      <label><input type="checkbox" id="web-search-toggle"> Web search</label>
      <button id="persona-btn" type="button">Persona</button>
      <button id="search-settings-btn" type="button">Search settings</button>
      <button id="mic-btn">🎤</button>
    </div>
    <div id="status">Click and hold to talk</div>
  </header>
  <main>
    <div id="conversation"></div>
  </main>
  <footer>
    a demo by <a href="https://linkedin.com/in/prasadvittaldev/" target="_blank" rel="noopener">Prasad Vittaldev</a>
  </footer>

  <dialog id="persona-dialog">
    <h2>Persona / system prompt</h2>
    <textarea id="persona-textarea" placeholder="Optional: describe a persona or give instructions for how the assistant should respond."></textarea>
    <div class="dialog-actions">
      <button id="persona-cancel-btn" type="button">Cancel</button>
      <button id="persona-save-btn" type="button">Save</button>
    </div>
  </dialog>

  <dialog id="search-settings-dialog">
    <h2>Search settings</h2>
    <p class="hint">Needs a free Tavily API key -- get one at
      <a href="https://tavily.com" target="_blank" rel="noopener">tavily.com</a>.
      Stored only in this browser (localStorage), sent to this server per-turn
      when Web search is on.</p>
    <input type="password" id="tavily-key-input" placeholder="Tavily API key">
    <div class="dialog-actions">
      <button id="search-settings-cancel-btn" type="button">Cancel</button>
      <button id="search-settings-save-btn" type="button">Save</button>
    </div>
  </dialog>

  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk
git add tamil_talk/static/index.html
git commit -m "feat: index.html -- pin header/footer + move waveform into header, web search checkbox, persona/search-settings dialogs"
```

---

### Task 8: `tamil_talk/static/app.js` — wire the new controls, fix `scrollToBottom()`, handle `searching`

**Files:**
- Modify: `tamil_talk/static/app.js`

**Interfaces:**
- Consumes: DOM elements from Task 7.
- Consumes: the updated `/talk` protocol from Task 6 (`searching` event; `end` event gains `web_search`/`tavily_api_key`).

- [ ] **Step 1: Replace the file contents**

```javascript
// tamil_talk/static/app.js
const micBtn = document.getElementById("mic-btn");
const thinkToggle = document.getElementById("think-toggle");
const webSearchToggle = document.getElementById("web-search-toggle");
const statusEl = document.getElementById("status");
const conversationEl = document.getElementById("conversation");
const mainEl = document.querySelector("main");
const waveformCanvas = document.getElementById("waveform");
const waveformCtx = waveformCanvas.getContext("2d");

const personaBtn = document.getElementById("persona-btn");
const personaDialog = document.getElementById("persona-dialog");
const personaTextarea = document.getElementById("persona-textarea");
const personaSaveBtn = document.getElementById("persona-save-btn");
const personaCancelBtn = document.getElementById("persona-cancel-btn");

const searchSettingsBtn = document.getElementById("search-settings-btn");
const searchSettingsDialog = document.getElementById("search-settings-dialog");
const tavilyKeyInput = document.getElementById("tavily-key-input");
const searchSettingsSaveBtn = document.getElementById("search-settings-save-btn");
const searchSettingsCancelBtn = document.getElementById("search-settings-cancel-btn");

let ws = null;
let audioCtx = null;
let processorNode = null;
let sourceNode = null;
let mediaStream = null;
let recording = false;

// Playback state for the current turn's TTS reply.
let playbackCtx = null;
let playbackTime = 0;
let analyserNode = null;
let waveformRAF = null;

// The <div class="turn assistant"> for the in-progress reply, and the
// currently-"speaking" sentence <span> inside it, for this turn.
let currentAssistantTurn = null;
let currentSpeakingSpan = null;

// Persona text and Tavily key are held here (not read live from an inline
// input) since both now live behind a Save/Cancel dialog.
let systemPrompt = "";
let tavilyApiKey = localStorage.getItem("tamil_talk_tavily_key") || "";

function scrollToBottom() {
  // main (not #conversation) is the actual scroll container -- #conversation
  // itself has no overflow set, so scrolling it directly is a no-op.
  mainEl.scrollTop = mainEl.scrollHeight;
}

function addTurn(role, text) {
  const div = document.createElement("div");
  div.className = `turn ${role}`;
  div.textContent = text;
  conversationEl.appendChild(div);
  scrollToBottom();
}

function startAssistantTurn() {
  currentAssistantTurn = document.createElement("div");
  currentAssistantTurn.className = "turn assistant";
  conversationEl.appendChild(currentAssistantTurn);
}

function addResponseSentence(text) {
  if (!currentAssistantTurn) startAssistantTurn();
  if (currentSpeakingSpan) currentSpeakingSpan.classList.remove("speaking");
  const span = document.createElement("span");
  span.className = "speaking";
  span.textContent = text + " ";
  currentAssistantTurn.appendChild(span);
  currentSpeakingSpan = span;
  scrollToBottom();
}

function endAssistantTurn() {
  if (currentSpeakingSpan) currentSpeakingSpan.classList.remove("speaking");
  currentAssistantTurn = null;
  currentSpeakingSpan = null;
}

function ensureSocket() {
  if (ws && ws.readyState === WebSocket.OPEN) return;
  const wsScheme = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${wsScheme}//${location.host}/talk`);
  ws.binaryType = "arraybuffer";
  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      const evt = JSON.parse(event.data);
      handleEvent(evt);
    } else {
      playChunk(new Int16Array(event.data));
    }
  };
  ws.onclose = () => { statusEl.textContent = "Disconnected — reload to reconnect"; };
}

function reenableMicNow() {
  micBtn.disabled = false;
  stopWaveform();
}

function reenableMicWhenPlaybackEnds() {
  if (!playbackCtx) {
    reenableMicNow();
    return;
  }
  const remaining = playbackTime - playbackCtx.currentTime;
  if (remaining <= 0) {
    reenableMicNow();
  } else {
    setTimeout(reenableMicNow, remaining * 1000);
  }
}

function handleEvent(evt) {
  if (evt.event === "transcript") {
    addTurn("user", evt.text);
  } else if (evt.event === "searching") {
    statusEl.textContent = `Searching: "${evt.query}"...`;
  } else if (evt.event === "response_sentence") {
    addResponseSentence(evt.text);
  } else if (evt.event === "audio_start") {
    if (playbackCtx && playbackCtx.state !== "closed") {
      playbackCtx.close();
    }
    playbackCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: evt.sample_rate });
    playbackTime = playbackCtx.currentTime;
    analyserNode = playbackCtx.createAnalyser();
    analyserNode.fftSize = 256;
    analyserNode.connect(playbackCtx.destination);
    startWaveform();
  } else if (evt.event === "done") {
    endAssistantTurn();
    statusEl.textContent = "Click and hold to talk";
    reenableMicWhenPlaybackEnds();
  } else if (evt.event === "error") {
    statusEl.textContent = `Error: ${evt.detail}`;
    endAssistantTurn();
    reenableMicNow();
  }
}

function startWaveform() {
  if (waveformRAF) cancelAnimationFrame(waveformRAF);
  const data = new Uint8Array(analyserNode.frequencyBinCount);
  const draw = () => {
    analyserNode.getByteTimeDomainData(data);
    const w = waveformCanvas.width = waveformCanvas.clientWidth;
    const h = waveformCanvas.height = waveformCanvas.clientHeight;
    waveformCtx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2;
    const baseRadius = Math.min(w, h) / 2 * 0.5;
    const amplitude = Math.min(w, h) / 2 * 0.42;

    waveformCtx.beginPath();
    for (let i = 0; i <= data.length; i++) {
      const idx = i % data.length;
      const amp = (data[idx] - 128) / 128;
      const angle = (i / data.length) * Math.PI * 2 - Math.PI / 2;
      const radius = baseRadius + amp * amplitude;
      const x = cx + radius * Math.cos(angle);
      const y = cy + radius * Math.sin(angle);
      if (i === 0) waveformCtx.moveTo(x, y);
      else waveformCtx.lineTo(x, y);
    }
    waveformCtx.closePath();
    waveformCtx.lineWidth = 3;
    waveformCtx.strokeStyle = "#4f4";
    waveformCtx.shadowColor = "#4f4";
    waveformCtx.shadowBlur = 14;
    waveformCtx.stroke();
    waveformCtx.shadowBlur = 0;

    waveformRAF = requestAnimationFrame(draw);
  };
  draw();
}

function stopWaveform() {
  if (waveformRAF) {
    cancelAnimationFrame(waveformRAF);
    waveformRAF = null;
  }
  waveformCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
}

function playChunk(int16) {
  if (!playbackCtx) return;
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;
  const buffer = playbackCtx.createBuffer(1, float32.length, playbackCtx.sampleRate);
  buffer.copyToChannel(float32, 0);
  const src = playbackCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(analyserNode || playbackCtx.destination);
  const startAt = Math.max(playbackTime, playbackCtx.currentTime);
  src.start(startAt);
  playbackTime = startAt + buffer.duration;
}

async function startRecording() {
  ensureSocket();
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    statusEl.textContent = `Mic error: ${err.message}`;
    return;
  }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  sourceNode = audioCtx.createMediaStreamSource(mediaStream);
  processorNode = audioCtx.createScriptProcessor(4096, 1, 1);
  processorNode.onaudioprocess = (e) => {
    if (!recording || ws.readyState !== WebSocket.OPEN) return;
    const input = e.inputBuffer.getChannelData(0);
    const int16 = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      const s = Math.max(-1, Math.min(1, input[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    ws.send(int16.buffer);
  };
  sourceNode.connect(processorNode);
  processorNode.connect(audioCtx.destination);
  recording = true;
  micBtn.classList.add("recording");
  statusEl.textContent = "Listening...";
}

function sendEndEvent() {
  ws.send(JSON.stringify({
    event: "end",
    think: thinkToggle.checked,
    sample_rate: audioCtx.sampleRate,
    system_prompt: systemPrompt,
    web_search: webSearchToggle.checked,
    tavily_api_key: tavilyApiKey,
  }));
}

function stopRecording() {
  if (!recording) return;
  recording = false;
  micBtn.classList.remove("recording");
  micBtn.disabled = true;
  statusEl.textContent = "Thinking...";
  processorNode.disconnect();
  sourceNode.disconnect();
  mediaStream.getTracks().forEach((t) => t.stop());
  if (ws.readyState === WebSocket.OPEN) {
    sendEndEvent();
  } else if (ws.readyState === WebSocket.CONNECTING) {
    // Fast tap: the WS may not have finished connecting yet (this is most
    // likely on the very first press, before any socket exists to reuse).
    // Send "end" as soon as it opens instead of dropping the turn silently.
    ws.addEventListener("open", sendEndEvent, { once: true });
  } else {
    statusEl.textContent = "Connection lost — reload to try again";
    micBtn.disabled = false;
  }
  audioCtx.close();
}

micBtn.addEventListener("mousedown", startRecording);
micBtn.addEventListener("touchstart", (e) => { e.preventDefault(); startRecording(); });
micBtn.addEventListener("mouseup", stopRecording);
micBtn.addEventListener("mouseleave", () => { if (recording) stopRecording(); });
micBtn.addEventListener("touchend", (e) => { e.preventDefault(); stopRecording(); });

function updatePersonaBtnLabel() {
  personaBtn.textContent = systemPrompt ? "Persona ✓" : "Persona";
}

personaBtn.addEventListener("click", () => {
  personaTextarea.value = systemPrompt;
  personaDialog.showModal();
});
personaSaveBtn.addEventListener("click", () => {
  systemPrompt = personaTextarea.value;
  updatePersonaBtnLabel();
  personaDialog.close();
});
personaCancelBtn.addEventListener("click", () => personaDialog.close());

searchSettingsBtn.addEventListener("click", () => {
  tavilyKeyInput.value = tavilyApiKey;
  searchSettingsDialog.showModal();
});
searchSettingsSaveBtn.addEventListener("click", () => {
  tavilyApiKey = tavilyKeyInput.value;
  localStorage.setItem("tamil_talk_tavily_key", tavilyApiKey);
  searchSettingsDialog.close();
});
searchSettingsCancelBtn.addEventListener("click", () => searchSettingsDialog.close());
```

- [ ] **Step 2: Verify the script has no syntax errors**

```bash
node --check /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk/tamil_talk/static/app.js
echo "syntax OK"
```

Expected: `syntax OK`.

- [ ] **Step 3: Commit**

```bash
cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk
git add tamil_talk/static/app.js
git commit -m "feat: app.js -- web search checkbox, persona/search-settings dialogs, fix scrollToBottom(), searching event"
```

---

### Task 9: `README.md` — document web search

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a "Web search" section**

In `README.md`, after the existing "## Testing" section and before "## Project layout", add:

```markdown
## Web search

Toggle "Web search" on to let the assistant search the web for current
information (weather, news, recent events) via [Tavily](https://tavily.com).

1. Get a free Tavily API key at [tavily.com](https://tavily.com).
2. Click "Search settings" in the header and paste it in — it's saved in
   your browser's `localStorage` (sent to this server per-turn only while
   the toggle is on; never sent anywhere else).
3. Turn on "Web search" and ask something that needs current info.

If the toggle is on but no key is configured (or the search fails), the
assistant just answers from its own knowledge instead of the turn failing.

This only works on the Ollama path — if Ollama is unreachable and the app
falls back to the direct GGUF path, the toggle has no effect.
```

Update the "Project layout" tree to add the new file:

```
tamil_talk/
  server.py         FastAPI app + the /talk WebSocket (STT -> LLM -> TTS)
  llm.py            Ollama/GGUF chat client, streaming, sentence-boundary detection, tool-calling
  tavily_search.py  Tavily web search backend for the web_search tool
  parler_synth.py   Parler-TTS synth backend
  whisper_stt.py    faster-whisper STT backend
  leveler.py        per-sentence loudness leveling + silence trimming
  resampler.py      PCM resampling (mic sample rate -> 16kHz for STT)
  text_chunk.py     sentence/clause splitting shared by the TTS path
  static/           frontend (index.html + app.js)
tests/
  test_llm.py
  test_tavily_search.py
manual_verify.py    no-browser pipeline check
```

- [ ] **Step 2: Commit**

```bash
cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk
git add README.md
git commit -m "docs: README -- document the web search toggle and Tavily key setup"
```

---

### Task 10: End-to-end live verification

**Files:** none (verification only)

This repo isn't currently running as a service anywhere (unlike the monorepo's `tamil_talkies` demo on port 9099, which stays untouched). Verification here starts a **separate, temporary** instance of this repo's own server, on a different port, using the same already-running Ollama instance and local model caches.

No real Tavily API key is available in this environment. Verification is scoped to what's genuinely checkable without one: the tool-calling wiring, the `searching` event, and the graceful-failure path (an invalid key still exercises a real HTTP round-trip to Tavily and back). A note is included below for the user to do a full "real results" pass once they have a real key.

- [ ] **Step 1: Start a temporary server instance on port 9100**

```bash
cd /home/prasad/.claude/jobs/b22015a3/tmp/tamil-talk
nohup /home/prasad/Desktop/suryantts/.venv-parler/bin/python -m uvicorn tamil_talk.server:app --host 0.0.0.0 --port 9100 > /tmp/tamil_talk_verify.log 2>&1 &
disown
```

- [ ] **Step 2: Wait for it to become healthy**

```bash
for i in $(seq 1 60); do
  out=$(curl -s http://localhost:9100/health --max-time 2 2>/dev/null)
  if echo "$out" | grep -q ok; then
    echo "healthy after $((i*3))s"
    break
  fi
  sleep 3
done
curl -s http://localhost:9100/health
```

Expected: `{"status":"ok"}`. If it never becomes healthy, check `/tmp/tamil_talk_verify.log`.

- [ ] **Step 3: Confirm the rebrand/layout landed**

```bash
curl -s http://localhost:9100/ | grep -o "<title>.*</title>"
curl -s http://localhost:9100/ | grep -o 'id="[a-z-]*"'
```

Expected: `<title>tamil-talk</title>`, and the id list includes `think-toggle`,
`web-search-toggle`, `persona-btn`, `persona-dialog`, `persona-textarea`,
`search-settings-btn`, `search-settings-dialog`, `tavily-key-input`,
`mic-btn`, `status`, `waveform`, `conversation`.

- [ ] **Step 4: Verify the no-search path is unaffected (regression check)**

```bash
cat > /tmp/verify_no_search.py << 'PYEOF'
import asyncio, json, sys
import wave
import websockets

async def run():
    wav_path = "/home/prasad/Desktop/suryantts/tamil/mixed_test.wav"
    with wave.open(wav_path, "rb") as w:
        sample_rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
    async with websockets.connect("ws://localhost:9100/talk") as ws:
        await ws.send(pcm)
        await ws.send(json.dumps({"event": "end", "think": False, "sample_rate": sample_rate}))
        saw_searching = False
        async for msg in ws:
            if isinstance(msg, bytes):
                continue
            evt = json.loads(msg)
            print(evt)
            if evt.get("event") == "searching":
                saw_searching = True
            if evt.get("event") in ("done", "error"):
                break
    print(f"\n--- summary: saw_searching={saw_searching} (must be False) ---")

asyncio.run(run())
PYEOF
/home/prasad/Desktop/suryantts/.venv-parler/bin/python /tmp/verify_no_search.py
```

Expected: normal `transcript` -> `response_sentence` (one or more) -> `audio_start` -> `done` events, `saw_searching=False`. This confirms the web-search machinery doesn't fire at all when the toggle is off — behavior identical to before this feature existed.

- [ ] **Step 5: Verify the web-search path (with an invalid key, since no real key is available)**

```bash
cat > /tmp/verify_with_search.py << 'PYEOF'
import asyncio, json, sys
import wave
import websockets

async def run():
    wav_path = "/home/prasad/Desktop/suryantts/tamil/mixed_test.wav"
    with wave.open(wav_path, "rb") as w:
        sample_rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
    async with websockets.connect("ws://localhost:9100/talk") as ws:
        await ws.send(pcm)
        await ws.send(json.dumps({
            "event": "end", "think": False, "sample_rate": sample_rate,
            "web_search": True, "tavily_api_key": "invalid-test-key-123",
        }))
        saw_searching = False
        got_final_audio = False
        async for msg in ws:
            if isinstance(msg, bytes):
                got_final_audio = True
                continue
            evt = json.loads(msg)
            print(evt)
            if evt.get("event") == "searching":
                saw_searching = True
            if evt.get("event") in ("done", "error"):
                break
    print(f"\n--- summary: saw_searching={saw_searching}, got_final_audio={got_final_audio} ---")

asyncio.run(run())
PYEOF
/home/prasad/Desktop/suryantts/.venv-parler/bin/python /tmp/verify_with_search.py
```

This test input (a Tamil phone-number sentence) likely won't itself trigger a
search, since it doesn't need current information — this step is checking
that turning the toggle on doesn't break the ordinary path, and that *if*
the model happens to call the tool, the graceful-failure path (invalid key
-> `tavily_search.web_search` returns an error string -> model still
produces a final spoken answer) works rather than crashing the turn. If
`saw_searching` is `False` here, that's an acceptable outcome (the model
correctly decided this particular sentence didn't need a search) — the
important thing is `got_final_audio=True` either way (turn completed and
was spoken, no `error` event).

- [ ] **Step 6: Stop the temporary verification server**

```bash
ps aux | grep "[u]vicorn tamil_talk.server" | awk '{print $2}' | xargs -r kill
sleep 2
ps aux | grep "[u]vicorn tamil_talk.server" || echo stopped
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Expected: `stopped`, GPU memory back near baseline. This is a temporary
verification instance, not a long-running demo — it should not be left
running afterward (unlike the monorepo's `tamil_talkies` demo, which stays
up).

No commit for this task (verification only). Report to the user: confirmation
the layout fix and new controls landed, confirmation the no-search path is
byte-identical to before, confirmation the search-path wiring round-trips
correctly end-to-end (including the graceful-failure branch, genuinely
exercised via a real failed Tavily call). Flag plainly that a full "real
search results" pass needs the user to test live in a browser once they
have a real Tavily key — that specific scenario (a real, successful search
changing the spoken answer) cannot be verified without one.

## Self-Review Notes

- **Spec coverage:** validated feasibility (already done pre-plan) —
  n/a, informational. Scope (Ollama-only) — Task 5's GGUF branch explicitly
  ignores web_search/tavily_api_key, tested. Turn flow (direct answer /
  tool call / bounded rounds) — Task 3, tested. Protocol additions — Task 6.
  Frontend (checkbox, persona dialog, search-settings dialog, searching
  status) — Tasks 7-8. Backend (`tavily_search.py`, `llm.py` orchestration,
  `server.py` wiring) — Tasks 1-6. Error handling (missing key, network/API
  error, malformed query, multiple tool calls) — Task 1 (tavily_search) +
  Task 3 (llm.py orchestration), all tested. Testing section's own
  requirements (unit tests for tavily_search + llm.py orchestration, no
  unit tests for WS/frontend, live verification) — followed exactly.
  Known limitations (GGUF fallback, no citations read aloud, English
  queries, 2-round cap, localStorage key storage) — all either enforced by
  code (GGUF ignoring the params, 2-round cap) or inherently true of the
  implementation (no citation-reading code was added; queries are whatever
  the model generates) — nothing further to build for these.
- **Placeholder scan:** none found.
- **Type consistency:** `chat_stream(history, think, system_prompt="", web_search=False, tavily_api_key="") -> Iterator[dict]`
  is identical between Task 5's two branches and Task 6's `server.py` call
  site. `accumulate_sentences(event_iter) -> Iterator[str | dict]` matches
  between Task 4's definition and Task 6's usage. `ollama_chat_stream_with_tools(...)`'s
  event shape (`{"type": "content", "text": ...}` / `{"type": "searching", "query": ...}`)
  matches between Task 3's definition, Task 4's `accumulate_sentences`
  passthrough handling, and Task 5's direct `yield from` in the Ollama
  branch of `make_llm`. `web_search(query, api_key) -> str` matches between
  Task 1's definition and Task 3's `tavily_search.web_search(query, tavily_api_key)`
  call.
