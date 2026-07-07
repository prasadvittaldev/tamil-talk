# Web search tool for tamil-talk — design

## Goal

Let the assistant answer questions that need current, real-world information
(weather, news, recent events) by giving it a `web_search` tool it can call
mid-conversation, gated behind a UI toggle so search-API usage stays under
the user's control.

## Validated feasibility

Before designing, the model's actual tool-calling behavior was tested
directly against the running Ollama instance (not assumed from
documentation):

- `ollama show hf.co/prasadvittaldev/gemma4-tamil-e4b-it-GGUF:Q5_K_M` lists
  `tools` as a registered capability (alongside `completion`; notably not
  `thinking`, consistent with prior findings in this project).
- A live `/api/chat` call with a `web_search` tool definition and a Tamil
  weather question produced a correct, well-formed tool call
  (`{"name": "web_search", "arguments": {"query": "current weather Chennai
  today"}}`) — the model decided on its own that the question needed a
  search, and chose an English query (search engines work better with
  English queries; this is expected, not a bug).
- Feeding a synthetic tool result back as a `tool`-role message produced a
  fluent, accurate Tamil answer synthesizing that data, unprompted (no
  extra system-prompt engineering needed).
- Ollama's **streaming** API returns a tool-call turn as a single
  `done: true` chunk with empty `content` — no incremental text deltas.
  This means the existing sentence-streaming pipeline (built for
  text-only replies) cannot be reused unmodified for a tool-call turn; the
  turn needs a distinct branch (see Turn Flow below).

This is empirical grounding, not a hopeful assumption: the core mechanism
this design relies on has already been exercised against the real model.

## Scope

**Ollama path only.** `llama-cpp-python`'s `create_chat_completion` also
accepts a `tools` parameter (confirmed via `inspect.signature`), but whether
it produces correct tool-calls for this model/architecture on the
llama.cpp side is unverified, and that GGUF-fallback path is only exercised
when Ollama is unreachable. Web search is silently unavailable on the
fallback path for now — the toggle simply has no effect there. This is an
explicit, documented limitation, not a silent gap.

This work lands in the `tamil-talk` repo
(github.com/prasadvittaldev/tamil-talk) only — the monorepo's
`tamil_talkies/` stays frozen as a pre-carve-out snapshot.

## Architecture: why native tool-calling over the alternatives

- **Native Ollama tool-calling (chosen):** proven above to work correctly
  with zero extra finetuning. Reuses Ollama's own structured tool-call
  parsing rather than inventing a parallel mechanism.
- *Prompt-engineered marker* (have the model emit a `[SEARCH: query]`
  string in plain text, detected via regex): rejected — fragile compared to
  a structured mechanism that already works, and it would reinvent
  something Ollama does natively and correctly.
- *Client-side search* (browser JS calls the search API directly):
  rejected — would leak the user's Tavily key to anything with devtools
  access to the page, and loses the model's own judgment about *when* a
  search is actually needed (would need a client-side heuristic instead).

## Turn flow when "Web search" is toggled on

1. Client sends the `end` event with `web_search: true` and (if configured)
   `tavily_api_key: "<key>"`.
2. Server calls Ollama with the `web_search` tool definition attached to
   the request.
3. **Model answers directly** (no tool call): identical to today's
   pipeline — the response streams through `accumulate_sentences()` →
   per-sentence TTS, unchanged.
4. **Model calls the tool**: server sends a new
   `{"event": "searching", "query": "<the model's query>"}` (frontend shows
   it as a status line, same slot as "Listening..."/"Thinking...") → server
   calls Tavily with the user-supplied key → appends the result as a
   `tool`-role message → re-invokes the LLM, tools still attached → that
   response is handled per steps 3/4 again.
5. **Bounded to 2 search rounds per turn.** Step 4 can repeat once more
   (round 2 gets its own fresh `searching` event with round 2's query, same
   as round 1) if the model calls the tool again. If it still wants to
   search after round 2, the 3rd (forced-final) call omits the `tools`
   parameter entirely — the model literally cannot call the tool again, so
   it must produce a text answer from whatever's already in history.

## Protocol additions

- Client → server `end` event gains two new optional fields:
  `web_search: bool` (default `false`) and `tavily_api_key: str` (default
  `""`).
- Server → client gains `{"event": "searching", "query": str}`, sent only
  when the model actually calls the tool (never sent on a toggle-off turn,
  or a turn where the model answers directly).

## Frontend changes

- **New "Web search" checkbox** next to the existing "Think mode" toggle,
  read into `sendEndEvent()`'s payload the same way `think` is.
- **Two header buttons open native `<dialog>` modals**, replacing the
  current inline persona `<input>`:
  - **"Persona"**: a `<textarea>` (multi-line — the old single-line input
    was cramped for real persona text) + Save/Cancel. Button label
    reflects whether a persona is currently set (e.g. "Persona" vs.
    "Persona ✓") so state is visible without reopening it. The saved text
    lives in a JS variable, sent in the `end` event exactly as today —
    only the input widget changes, not the wire format.
  - **"Search settings"**: a masked (`type="password"`) Tavily API key
    field + Save/Cancel, with a short hint linking to where to get a free
    key. Saved to `localStorage` so it survives page reloads, and read into
    `sendEndEvent()`'s payload as `tavily_api_key`.
- `handleEvent()` gains a `searching` branch → sets `#status` to
  `Searching: "<query>"...`.

**Explicit tradeoff, stated plainly:** storing a third-party API key in
`localStorage` means it's visible to anything with devtools/storage access
to that browser. Reasonable for a personal demo; would need real secret
handling before this became a multi-user service.

## Backend changes

- **`tamil_talk/tavily_search.py`** (new): `web_search(query: str,
  api_key: str) -> str`. Calls Tavily's API, formats the top 3 results into
  a plain-text block — one `"<title>: <snippet, truncated to 300 chars>"`
  line per result, joined with newlines (roughly 900 chars total, well
  under the context budget). Never raises — a missing/empty
  `api_key`, a network error, or an API error all produce a plain error
  string (e.g. `"search unavailable: no API key configured"` /
  `"search unavailable: <reason>"`) returned as the tool result, so the
  model answers from its own knowledge instead of the turn failing.
  Malformed tool-call arguments (missing/empty `query`) are handled the
  same way, one level up in `llm.py`, before `tavily_search.web_search` is
  even called.
- **`tamil_talk/llm.py`**:
  - The `web_search` tool's JSON schema (name, description, `query`
    parameter) lives here.
  - `make_llm()`'s returned `chat_stream` gains the two-phase-with-cap
    orchestration described above: detect `tool_calls` in the (otherwise
    single-chunk) streamed response; if present, resolve the search,
    append the tool result, and recurse into another `chat_stream` call
    (tools attached, unless the 2-round cap has been hit, in which case
    tools are omitted) — otherwise it's a normal streamed reply, passed
    through unchanged.
  - If the model's response contains more than one tool call at once
    (unlikely, since only one tool is offered, but possible): only the
    first is executed; the rest are ignored. Explicit choice, not a silent
    mishandling.
- **`tamil_talk/server.py`**: reads `web_search`/`tavily_api_key` off the
  `end` event, passes them through to `chat_stream`, and forwards the new
  `searching` event to the client. A mid-search client disconnect is
  handled by the same outer `except WebSocketDisconnect` this file already
  has — no new handling needed.

## Testing

- Unit tests (mocked at the HTTP boundary, matching `test_llm.py`'s
  existing style):
  - `tavily_search.py`: successful-result formatting, missing-key
    fallback, and network/API-error fallback — all as plain strings, never
    raising.
  - `llm.py`: tool-call detected → search executed → result appended →
    second call made with tools still attached; round-cap forcing (3rd
    call omits `tools`); malformed tool-call arguments handled gracefully;
    multiple-tool-calls-in-one-response only executes the first.
- No unit tests for the WS orchestration or frontend — verified live,
  matching this project's established convention.
- Live verification: toggle on + ask something needing current info
  (confirm a real Tavily call fires, a `searching` event arrives, and the
  spoken answer reflects real search results); toggle off (confirm the
  tool is never offered — byte-identical behavior to before this feature
  existed).

## Known limitations (explicit, not gaps to silently fix)

- GGUF-fallback path doesn't get web search — Ollama-only for now.
- The reply never reads out URLs/citations — the model paraphrases
  naturally in Tamil (as tested), fitting a voice-first experience; source
  links aren't surfaced anywhere in the UI either.
- Search queries the model generates are typically English even
  mid-Tamil-conversation — expected behavior (search engines work better
  that way), not a bug.
- The 2-round cap could occasionally truncate a genuinely multi-step
  research need — acceptable tradeoff for a demo.
- Tavily key lives in browser `localStorage`, in the clear — acceptable for
  a personal demo, not for a multi-user deployment (see Frontend changes).
