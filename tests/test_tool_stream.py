"""Ollama's /v1 endpoint loses tool calls when streaming, and the proxy puts them back.

The model decides to call a tool, the stream delivers {"delta":{"content":""},"finish_reason":
"stop"}, and the call itself never arrives. The client receives a well-formed turn carrying
nothing and reads it as the agent giving up. Measured over 72 hours on this box: 6 of 4,523
Ollama streams against 0 of 671 on vLLM, on gemma4 and nemotron alike — which is why the
default keys on the upstream rather than the model, and why a model quirk can override it.

The recovery is a non-streaming re-issue, not a plain retry: only the streaming path drops the
call, so asking the same question the other way answers it. Sending stream:false in the first
place was measured and rejected — Ollama withholds response headers until a non-streaming
generation is completely finished, and 15.9% of tool-bearing requests here run past the ~30s
a client will wait in silence.
"""
import json

import httpx
import pytest

from ai_proxy import proxy as P


# --- which requests are covered ----------------------------------------------------------


def test_ollama_buffers_by_default(client):
    assert P._tool_stream_mode("gemma4:26b", "ollama") == "buffer"


def test_vllm_is_left_alone(client):
    """The same models served over vLLM never lost a call in 671 streams."""
    assert P._tool_stream_mode("nemotron-vllm", "vllm") == "passthrough"


def test_an_unnamed_model_on_ollama_is_still_covered(client):
    """The defect belongs to the endpoint, so a freshly pulled model inherits the fix rather
    than the bug — the opposite of what a model-name allowlist would do."""
    assert P._tool_stream_mode("something-pulled-yesterday:latest", "ollama") == "buffer"


def test_a_quirk_can_opt_a_model_out():
    assert P._tool_stream_mode("x", "ollama", {"tool_stream": "passthrough"}) == "passthrough"


def test_a_quirk_can_opt_a_model_in_on_another_upstream():
    assert P._tool_stream_mode("x", "vllm", {"tool_stream": "buffer"}) == "buffer"


def test_an_unrelated_quirk_does_not_decide_it(client):
    assert P._tool_stream_mode("x", "ollama", {"thinking": "force_off"}) == "buffer"


def test_the_rule_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(P, "load_rules_config",
                        lambda: {"tool_stream": {"enabled": False, "upstreams": ["ollama"]}})
    assert P._tool_stream_mode("gemma4:26b", "ollama") == "passthrough"


@pytest.mark.parametrize("body,expected", [
    ({"tools": [{"type": "function"}]}, True),
    ({"functions": [{"name": "f"}]}, True),
    ({"tools": []}, False),
    ({"tools": None}, False),
    ({}, False),
    ("not a dict", False),
])
def test_which_bodies_offer_tools(body, expected):
    assert P._request_has_tools(body) is expected


# --- the failure, end to end -------------------------------------------------------------


_LOST = ('data: {"id":"c1","object":"chat.completion.chunk","model":"gemma4:26b","choices":'
         '[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":"stop"}]}\n\n'
         'data: [DONE]\n\n')

_RECOVERED = {
    "id": "c1", "object": "chat.completion", "model": "gemma4:26b",
    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "Read", "arguments": '{"path":"/etc/hosts"}'}}]}}],
    "usage": {"completion_tokens": 40},
}

_REQUEST = {
    "model": "gemma4:26b", "stream": True,
    "messages": [{"role": "user", "content": "read /etc/hosts"}],
    "tools": [{"type": "function", "function": {"name": "Read", "parameters": {}}}],
}


def _streaming_response(text: str) -> httpx.Response:
    """An SSE response the proxy can actually iterate.

    MockTransport with `content=` hands back a body httpx has already read, so the proxy's
    aiter_raw() raises StreamConsumed and the whole response arrives empty — the mock fails in
    a way that looks like a proxy bug. An async generator leaves the stream unread.
    """
    async def _gen():
        yield text.encode()

    return httpx.Response(200, content=_gen(), headers={"content-type": "text/event-stream"})


class _Upstream:
    """Serves the broken stream first and a healthy object to the non-streaming re-issue,
    which is exactly how the real endpoint behaves."""

    def __init__(self, streamed_body=_LOST, json_body=None):
        self.streamed_body = streamed_body
        self.json_body = _RECOVERED if json_body is None else json_body
        self.calls: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        try:
            sent = json.loads(request.content or b"{}")
        except ValueError:
            sent = {}
        self.calls.append(sent)
        if sent.get("stream"):
            return _streaming_response(self.streamed_body)
        return httpx.Response(200, json=self.json_body,
                              headers={"content-type": "application/json"})


@pytest.fixture
def upstream(client, monkeypatch):
    """Swap the app's shared httpx client for a mocked upstream, restoring it afterwards so
    the session-scoped `client` fixture is not left poisoned for other test modules."""
    def _install(up: _Upstream):
        real = client.app.state.client
        mocked = httpx.AsyncClient(transport=httpx.MockTransport(up.handler),
                                   timeout=httpx.Timeout(None, pool=15.0))
        client.app.state.client = mocked
        monkeypatch.setattr(client.app.state, "client", mocked, raising=False)

        def _restore():
            client.app.state.client = real
        request_finalizers.append(_restore)
        return up

    request_finalizers: list = []
    try:
        yield _install
    finally:
        for fn in request_finalizers:
            fn()


def _sse_events(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload and payload != "[DONE]":
            try:
                out.append(json.loads(payload))
            except ValueError:
                pass
    return out


def test_a_lost_tool_call_is_re_fetched_and_delivered(client, upstream):
    up = upstream(_Upstream())
    r = client.post("/v1/chat/completions", json=_REQUEST)
    assert r.status_code == 200

    assert len(up.calls) == 2, "the broken stream should have triggered exactly one re-issue"
    assert up.calls[0]["stream"] is True
    assert up.calls[1]["stream"] is False, "the re-issue is what dodges the streaming defect"
    assert "stream_options" not in up.calls[1], \
        "a streaming-only field on a stream:false body makes Ollama reject the request outright"

    names = [(d.get("function") or {}).get("name")
             for ev in _sse_events(r.text)
             for ch in (ev.get("choices") or [])
             for d in ((ch.get("delta") or {}).get("tool_calls") or [])]
    assert "Read" in names, "the client must receive the tool call the stream dropped"


def test_the_client_never_sees_the_broken_response(client, upstream):
    """Buffering is what makes the re-issue safe: a client already handed a finish_reason will
    not reliably accept a second message after it, so the failed attempt must not be relayed."""
    upstream(_Upstream())
    r = client.post("/v1/chat/completions", json=_REQUEST)
    finishes = [ch.get("finish_reason")
                for ev in _sse_events(r.text) for ch in (ev.get("choices") or [])
                if ch.get("finish_reason")]
    assert finishes == ["tool_calls"], f"exactly one ending, from the good response: {finishes}"


def test_a_healthy_stream_is_passed_through_untouched(client, upstream):
    good = ('data: {"choices":[{"index":0,"delta":{"content":"here you go"}}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            'data: [DONE]\n\n')
    up = upstream(_Upstream(streamed_body=good))
    r = client.post("/v1/chat/completions", json=_REQUEST)
    assert r.status_code == 200
    assert len(up.calls) == 1, "nothing was wrong, so nothing should have been re-issued"
    assert "here you go" in r.text


def test_a_stream_carrying_only_a_tool_call_is_not_treated_as_empty(client, upstream):
    """A turn whose entire output is a tool call delivered no prose and is still a success —
    counting characters alone would re-issue a perfectly good response."""
    tc = ('data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1",'
          '"type":"function","function":{"name":"Read","arguments":"{}"}}]}}]}\n\n'
          'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
          'data: [DONE]\n\n')
    up = upstream(_Upstream(streamed_body=tc))
    client.post("/v1/chat/completions", json=_REQUEST)
    assert len(up.calls) == 1


def test_a_request_without_tools_is_not_buffered(client, upstream):
    """Prose keeps streaming token by token — it is tool calls that go missing, and covering
    everything would cost live delivery on the majority of traffic for nothing."""
    up = upstream(_Upstream())
    body = dict(_REQUEST)
    body.pop("tools")
    client.post("/v1/chat/completions", json=body)
    assert len(up.calls) == 1, "no tools, no re-issue"


def test_a_failed_re_issue_leaves_the_original_response_alone(client, upstream):
    """The upstream is already misbehaving; a broken recovery must not turn a bad turn into a
    500. The client still gets the empty turn, and empty_output still records it."""
    class _Broken(_Upstream):
        def handler(self, request):
            try:
                sent = json.loads(request.content or b"{}")
            except ValueError:
                sent = {}
            self.calls.append(sent)
            if sent.get("stream"):
                return _streaming_response(_LOST)
            return httpx.Response(500, json={"error": "upstream fell over"})

    up = upstream(_Broken())
    r = client.post("/v1/chat/completions", json=_REQUEST)
    assert r.status_code == 200
    assert len(up.calls) == 2


# --- reading a tool call, which has no characters to count ---------------------------------
#
# The first version of the audit reported every successful recovery as a failure, because it
# judged success by counting delivered characters and a tool call has none. In production that
# printed recovered=false next to a response that had in fact been rescued — the one number
# anyone would use to decide whether this works at all.


def test_a_delta_tool_call_is_seen():
    assert P._stream_has_tool_call(
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"Read"}}]}}]}\n')


def test_an_assembled_tool_call_is_seen():
    assert P._stream_has_tool_call(
        'data: {"choices":[{"message":{"tool_calls":[{"function":{"name":"Read"}}]}}]}\n')


def test_an_anthropic_tool_use_block_is_seen():
    assert P._stream_has_tool_call(
        'data: {"type":"content_block_start","content_block":{"type":"tool_use","name":"Read"}}\n')


def test_prose_is_not_mistaken_for_a_tool_call():
    assert not P._stream_has_tool_call(
        'data: {"choices":[{"delta":{"content":"I could call Read here"}}]}\n')


def test_an_empty_tool_calls_list_is_not_a_tool_call():
    assert not P._stream_has_tool_call('data: {"choices":[{"delta":{"tool_calls":[]}}]}\n')


def test_junk_does_not_raise():
    assert not P._stream_has_tool_call("data: not json\ndata: [DONE]\n")
    assert not P._stream_has_tool_call(None)


def test_a_rescued_tool_call_is_reported_as_recovered(client, upstream):
    """A recovery that worked must not be filed as a failure."""
    upstream(_Upstream())
    client.post("/v1/chat/completions", json=_REQUEST)
    conn = P.db()
    try:
        row = conn.execute("SELECT gate_details FROM requests WHERE gate_rule='tool_stream' "
                           "ORDER BY ts DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    assert json.loads(row["gate_details"])["recovered"] is True


def test_an_empty_re_issue_is_reported_as_not_recovered(client, upstream):
    """Empty twice is not the streaming defect — gemma4:26b is the MoE build, which has its own
    reported habit of returning nothing on long prompts. The flag is what separates the two."""
    empty = {"id": "c1", "choices": [{"index": 0, "finish_reason": "stop", "message": {
        "role": "assistant", "content": ""}}]}
    upstream(_Upstream(json_body=empty))
    client.post("/v1/chat/completions", json=_REQUEST)
    conn = P.db()
    try:
        row = conn.execute("SELECT gate_details, gate_reason FROM requests "
                           "WHERE gate_rule='tool_stream' ORDER BY ts DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    assert json.loads(row["gate_details"])["recovered"] is False
    assert "not the streaming defect" in row["gate_reason"]


def test_the_re_issue_is_recorded_for_the_audit(client, upstream):
    """A response the client never saw was replaced by one it did — that has to be visible, or
    the next person debugging a latency spike has no idea a second generation happened."""
    upstream(_Upstream())
    client.post("/v1/chat/completions", json=_REQUEST)
    conn = P.db()
    try:
        row = conn.execute(
            "SELECT gate_rule FROM requests WHERE gate_rule='tool_stream' "
            "ORDER BY ts DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    assert row is not None, "the re-issue should have left a tool_stream gate rule"
