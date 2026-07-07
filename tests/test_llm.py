import json
from unittest.mock import patch, MagicMock

import requests

from tamil_talk.llm import (
    check_ollama_reachable, append_user_turn, append_assistant_turn,
    build_chat_messages, accumulate_sentences, ollama_chat_stream,
    NO_THINK_SYSTEM_MESSAGE, make_llm,
)


def test_check_ollama_reachable_true_on_200():
    fake_resp = MagicMock(status_code=200)
    with patch("tamil_talk.llm.requests.get", return_value=fake_resp) as mock_get:
        assert check_ollama_reachable("http://localhost:11434") is True
        mock_get.assert_called_once_with("http://localhost:11434/api/version", timeout=2)


def test_check_ollama_reachable_false_on_non_200():
    fake_resp = MagicMock(status_code=500)
    with patch("tamil_talk.llm.requests.get", return_value=fake_resp):
        assert check_ollama_reachable("http://localhost:11434") is False


def test_check_ollama_reachable_false_on_connection_error():
    with patch("tamil_talk.llm.requests.get", side_effect=requests.ConnectionError()):
        assert check_ollama_reachable("http://localhost:11434") is False


def test_append_user_turn_adds_message_without_mutating_input():
    history = [{"role": "assistant", "content": "hi"}]
    out = append_user_turn(history, "hello")
    assert out == [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "hello"},
    ]
    assert history == [{"role": "assistant", "content": "hi"}]  # unmutated


def test_append_assistant_turn_adds_message_without_mutating_input():
    history = [{"role": "user", "content": "hello"}]
    out = append_assistant_turn(history, "hi there")
    assert out == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    assert history == [{"role": "user", "content": "hello"}]  # unmutated


def test_append_turns_compose_across_multiple_calls():
    history = []
    history = append_user_turn(history, "turn 1 q")
    history = append_assistant_turn(history, "turn 1 a")
    history = append_user_turn(history, "turn 2 q")
    assert history == [
        {"role": "user", "content": "turn 1 q"},
        {"role": "assistant", "content": "turn 1 a"},
        {"role": "user", "content": "turn 2 q"},
    ]


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


def _fake_ndjson_response(lines, status_code=200):
    """A minimal stand-in for requests.Response supporting the subset of
    the API ollama_chat_stream uses: .status_code, .text, .raise_for_status(),
    .iter_lines()."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = "\n".join(lines)
    if status_code == 200:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status_code}")
    resp.iter_lines.return_value = iter([line.encode() for line in lines])
    return resp


def test_ollama_chat_stream_yields_content_deltas_and_stops_at_done():
    lines = [
        json.dumps({"message": {"content": "Hel"}, "done": False}),
        json.dumps({"message": {"content": "lo"}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True}),
    ]
    fake_resp = _fake_ndjson_response(lines)
    history = [{"role": "user", "content": "hi"}]
    with patch("tamil_talk.llm.requests.post", return_value=fake_resp) as mock_post:
        result = list(ollama_chat_stream(history, think=False, model="test-model", base_url="http://x:1"))
    assert result == ["Hel", "lo"]
    mock_post.assert_called_once_with(
        "http://x:1/api/chat",
        json={
            "model": "test-model",
            "messages": [{"role": "system", "content": NO_THINK_SYSTEM_MESSAGE}] + history,
            "think": False,
            "stream": True,
        },
        timeout=120,
        stream=True,
    )


def test_ollama_chat_stream_think_true_does_not_prepend_no_think_message():
    lines = [json.dumps({"message": {"content": "hi"}, "done": True})]
    fake_resp = _fake_ndjson_response(lines)
    history = [{"role": "user", "content": "hi"}]
    with patch("tamil_talk.llm.requests.post", return_value=fake_resp) as mock_post:
        list(ollama_chat_stream(history, think=True, model="m", base_url="http://x:1"))
    assert mock_post.call_args.kwargs["json"]["messages"] == history


def test_ollama_chat_stream_raises_on_http_error():
    fake_resp = _fake_ndjson_response([], status_code=500)
    with patch("tamil_talk.llm.requests.post", return_value=fake_resp):
        try:
            list(ollama_chat_stream([], think=True, model="m", base_url="http://x:1"))
            assert False, "expected HTTPError"
        except requests.HTTPError:
            pass


def test_ollama_chat_stream_retries_without_think_when_model_lacks_thinking_capability():
    no_think_resp = MagicMock()
    no_think_resp.status_code = 400
    no_think_resp.text = '{"error":"\\"m\\" does not support thinking"}'

    ok_lines = [json.dumps({"message": {"content": "reply"}, "done": True})]
    ok_resp = _fake_ndjson_response(ok_lines)

    history = [{"role": "user", "content": "hi"}]
    with patch("tamil_talk.llm.requests.post", side_effect=[no_think_resp, ok_resp]) as mock_post:
        result = list(ollama_chat_stream(history, think=True, model="m", base_url="http://x:1"))

    assert result == ["reply"]
    assert mock_post.call_count == 2
    first_call, second_call = mock_post.call_args_list
    assert first_call.kwargs["json"]["think"] is True
    assert second_call.kwargs["json"] == {
        "model": "m", "messages": history, "think": False, "stream": True,
    }


def test_ollama_chat_stream_does_not_retry_when_think_false_and_error_unrelated():
    fake_resp = _fake_ndjson_response([], status_code=500)
    with patch("tamil_talk.llm.requests.post", return_value=fake_resp) as mock_post:
        try:
            list(ollama_chat_stream([], think=False, model="m", base_url="http://x:1"))
            assert False, "expected HTTPError"
        except requests.HTTPError:
            pass
    assert mock_post.call_count == 1


def test_ollama_chat_stream_does_not_retry_when_think_true_and_400_is_unrelated():
    fake_resp = MagicMock()
    fake_resp.status_code = 400
    fake_resp.text = '{"error":"invalid request: messages field is required"}'
    fake_resp.raise_for_status.side_effect = requests.HTTPError("boom")
    with patch("tamil_talk.llm.requests.post", return_value=fake_resp) as mock_post:
        try:
            list(ollama_chat_stream([], think=True, model="m", base_url="http://x:1"))
            assert False, "expected HTTPError"
        except requests.HTTPError:
            pass
    assert mock_post.call_count == 1


def test_build_chat_messages_passes_through_when_think_true_and_no_persona():
    history = [{"role": "user", "content": "hi"}]
    assert build_chat_messages(history, think=True) == history


def test_build_chat_messages_prepends_no_think_instruction_when_think_false():
    history = [{"role": "user", "content": "hi"}]
    out = build_chat_messages(history, think=False)
    assert out[0]["role"] == "system"
    assert "directly" in out[0]["content"].lower()
    assert out[1:] == history


def test_build_chat_messages_prepends_persona_when_think_true():
    history = [{"role": "user", "content": "hi"}]
    out = build_chat_messages(history, think=True, system_prompt="You are a pirate.")
    assert out[0] == {"role": "system", "content": "You are a pirate."}
    assert out[1:] == history


def test_build_chat_messages_combines_persona_and_no_think_into_one_message():
    history = [{"role": "user", "content": "hi"}]
    out = build_chat_messages(history, think=False, system_prompt="You are a pirate.")
    assert len(out) == len(history) + 1
    assert out[0]["role"] == "system"
    assert "You are a pirate." in out[0]["content"]
    assert "directly" in out[0]["content"].lower()
    assert out[1:] == history


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
