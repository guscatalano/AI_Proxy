"""A response the upstream billed for and then delivered nothing of.

Measured on this box: 81 of 241 gemma4 replies in one six-hour window generated tokens and
sent no content, no reasoning and no tool call. Status 200, healthy-looking token counts, a
clean stop reason, no error — so every existing view called it a success, and to whoever was
driving the agent it read as the model stopping for no reason.

Replaying a failing request five times produced a valid tool call every time, so this is
intermittent and upstream. That is what the column is for: it cannot be reproduced on demand,
so it has to be recorded when it happens.
"""
from ai_proxy import proxy as P


def _sse(*objs):
    import json
    return "".join("data: %s\n" % json.dumps(o) for o in objs)


# --- what counts as delivered --------------------------------------------------------------


def test_the_real_failing_stream_reads_as_nothing_delivered():
    """Verbatim shape of the 23:27:55 response: an empty content chunk, a stop, and a usage
    block claiming 1,312 tokens."""
    blob = _sse(
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"completion_tokens": 1312}},
    )
    assert P._response_delivered_chars(None, blob) == 0


def test_text_counts_as_delivered():
    assert P._response_delivered_chars(None, _sse(
        {"choices": [{"delta": {"content": "hello"}}]})) == 5


def test_reasoning_alone_counts_as_delivered():
    """A thinking model that answered in its reasoning field did give the client something.
    Counting only `content` would call that empty."""
    assert P._response_delivered_chars(None, _sse(
        {"choices": [{"delta": {"reasoning": "pondering"}}]})) == 9


def test_the_ollama_native_shape_counts():
    assert P._response_delivered_chars(None, _sse(
        {"message": {"content": "hi", "thinking": "hmm"}})) == 5


def test_the_anthropic_shape_counts():
    assert P._response_delivered_chars(None, _sse(
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}})) == 2


def test_a_non_streaming_body_counts():
    import json
    body = json.dumps({"choices": [{"message": {"content": "abcd"}}]})
    assert P._response_delivered_chars(body, None) == 4


def test_junk_does_not_raise():
    assert P._response_delivered_chars("not json", "data: also not json\n") == 0


# --- what gets recorded ---------------------------------------------------------------------


def _row(rid):
    conn = P.db()
    r = conn.execute("SELECT empty_output, inflight_at_finish, completion_tokens, tool_calls "
                     "FROM requests WHERE id=?", (rid,)).fetchone()
    conn.execute("DELETE FROM requests WHERE id=?", (rid,))
    conn.commit()
    conn.close()
    return dict(r) if r else None


def _seed(rid):
    conn = P.db()
    conn.execute("INSERT INTO requests (id, ts, method, path, upstream_url, model, is_stream, "
                 "client_ip, client_app, upstream) VALUES (?,?,?,?,?,?,?,?,?,?)",
                 (rid, 1_800_000_000, "POST", "/v1/chat/completions",
                  "http://localhost:11434/v1/chat/completions", "gemma4:26b", 1,
                  "127.0.0.1", "test", "ollama"))
    conn.commit()
    conn.close()


def test_an_empty_billed_response_is_flagged(client):
    _seed("eo_empty")
    blob = _sse({"choices": [{"delta": {"content": ""}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"completion_tokens": 1312}})
    P._save_finish("eo_empty", 200, {}, None, blob, 1000.0, None)
    row = _row("eo_empty")
    assert row["empty_output"] == 1
    assert row["completion_tokens"] == 1312, "the tokens really were billed"


def test_a_normal_response_is_not_flagged(client):
    _seed("eo_ok")
    blob = _sse({"choices": [{"delta": {"content": "here you go"}}]},
                {"choices": [], "usage": {"completion_tokens": 12}})
    P._save_finish("eo_ok", 200, {}, None, blob, 1000.0, None)
    assert _row("eo_ok")["empty_output"] is None


def test_a_tool_call_is_not_flagged(client):
    """A turn whose whole output is a tool call delivered plenty — it just has no prose."""
    _seed("eo_tool")
    blob = _sse({"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "type": "function",
                     "function": {"name": "Write", "arguments": '{"path":"a"}'}}]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
                {"choices": [], "usage": {"completion_tokens": 40}})
    P._save_finish("eo_tool", 200, {}, None, blob, 1000.0, None)
    assert _row("eo_tool")["empty_output"] is None


def test_a_zero_token_response_that_terminated_properly_is_not_flagged(client):
    """Nothing billed, nothing delivered — but the stream ended cleanly, so the model simply
    had nothing to say. That is different from a stream that stopped mid-flight."""
    _seed("eo_zero")
    blob = (_sse({"choices": [{"delta": {}, "finish_reason": "stop"}]},
                 {"choices": [], "usage": {"completion_tokens": 0}}) + "data: [DONE]\n")
    P._save_finish("eo_zero", 200, {}, None, blob, 1000.0, None)
    assert _row("eo_zero")["empty_output"] is None


def test_an_error_response_is_not_flagged(client):
    """A 500 already reports itself; the column is for failures that look like successes."""
    _seed("eo_err")
    P._save_finish("eo_err", 500, {}, None, None, 1000.0, "boom")
    assert _row("eo_err")["empty_output"] is None


def test_the_list_api_exposes_it(client):
    _seed("eo_api")
    blob = _sse({"choices": [{"delta": {"content": ""}}]},
                {"choices": [], "usage": {"completion_tokens": 900}})
    P._save_finish("eo_api", 200, {}, None, blob, 1000.0, None)
    try:
        items = client.get("/__proxy/api/requests?limit=80").json()["items"]
        row = next((i for i in items if i["id"] == "eo_api"), None)
        assert row and row["empty_output"] == 1, "the badge has nothing to render from"
    finally:
        _row("eo_api")


# --- the stream that died mid-flight ---------------------------------------------------------
#
# The production failure, verbatim. One chunk, a control token in the reasoning field, and then
# nothing: no finish_reason, no usage block, no [DONE]. The client sees a 2xx with an empty
# assistant turn and Claude Code answers it with "[Your previous response had no visible
# output. Please continue...]". Seen 3 times in 24 hours, on gemma4 AND on nemotron.

_DEAD_STREAM = ('data: {"id":"chatcmpl-660","object":"chat.completion.chunk",'
                '"model":"gemma4:26b","choices":[{"index":0,"delta":{"role":"assistant",'
                '"content":"","reasoning":"<|channel|>"},"finish_reason":null}]}\n')

_HEALTHY = ('data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n'
            'data: [DONE]\n')


def test_a_control_token_is_not_delivered_content():
    """<|channel|> is protocol debris. Counted as text it reads as eleven characters
    successfully delivered, and the failure hides behind its own symptom."""
    assert P._response_delivered_chars(None, _DEAD_STREAM) == 0


def test_a_stray_control_token_does_not_discard_a_real_answer():
    blob = ('data: {"choices":[{"delta":{"content":"real answer<|end|>"},'
            '"finish_reason":"stop"}]}\ndata: [DONE]\n')
    assert P._response_delivered_chars(None, blob) == len("real answer")


def test_a_stream_without_a_finish_reason_did_not_finish():
    assert P._extract_finish_reason(_DEAD_STREAM) is None
    assert P._extract_finish_reason(_HEALTHY) == "stop"


def test_the_ollama_native_done_reason_counts_as_finishing():
    assert P._extract_finish_reason('data: {"done_reason":"stop"}\n') == "stop"


def test_the_anthropic_stop_reason_counts_as_finishing():
    assert P._extract_finish_reason(
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n') == "end_turn"


def test_a_dead_stream_is_flagged_even_though_nothing_was_billed(client):
    """The first version required completion_tokens > 0 and missed every one of these: a
    stream that dies never gets a usage block, so nothing is billed."""
    _seed("eo_dead")
    P._save_finish("eo_dead", 200, {}, None, _DEAD_STREAM, 6200.0, None)
    row = _row("eo_dead")
    assert row["empty_output"] == 1
    assert row["completion_tokens"] in (None, 0), "a dead stream reports no usage"


def test_a_healthy_stream_is_not_flagged(client):
    _seed("eo_live")
    P._save_finish("eo_live", 200, {}, None, _HEALTHY, 1000.0, None)
    assert _row("eo_live")["empty_output"] is None


def test_a_slow_but_complete_stream_is_not_flagged(client):
    """Duration is not the signal — a 180-second reply that arrives is fine."""
    _seed("eo_slow")
    P._save_finish("eo_slow", 200, {}, None, _HEALTHY, 180_000.0, None)
    assert _row("eo_slow")["empty_output"] is None
