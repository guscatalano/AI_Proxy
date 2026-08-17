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


def test_a_zero_token_response_is_not_flagged(client):
    """Nothing billed, nothing delivered, nothing wrong."""
    _seed("eo_zero")
    P._save_finish("eo_zero", 200, {}, None,
                   _sse({"choices": [], "usage": {"completion_tokens": 0}}), 1000.0, None)
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
