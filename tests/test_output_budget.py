"""Clamp an output reservation the client will never use.

OpenAI-compatible engines count requested output against the context window. Claude Code asks
for max_tokens=64000, so vLLM rejected the request outright once the transcript passed 198,144
tokens: "maximum context length is 262144 tokens. However, you requested 64000 output tokens and
your prompt contains at least 198145 input tokens". A 262,144 window was behaving like a 198,144
one, and every claude-code turn 400'd.

Replies measured on this box run 26-1,952 tokens, so the reservation bought nothing.
"""
import httpx
import pytest

from ai_proxy import proxy as P


class _Recorder:
    def __init__(self):
        self.bodies = []

    def handler(self, request: httpx.Request):
        import json
        try:
            self.bodies.append(json.loads(request.content or b"{}"))
        except ValueError:
            self.bodies.append({})
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"completion_tokens": 1}}, headers={"content-type": "application/json"})


@pytest.fixture
def rec(client):
    r = _Recorder()
    real = client.app.state.client
    client.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(r.handler))
    try:
        yield r
    finally:
        client.app.state.client = real


def _send(client, max_tokens):
    client.post("/v1/chat/completions", json={
        "model": "m", "stream": False, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "hi"}]},
        headers={"x-proxy-no-router": "1"})


def test_an_oversized_reservation_is_clamped(client, rec):
    _send(client, 64000)
    assert rec.bodies, "no upstream request captured"
    assert rec.bodies[-1]["max_tokens"] == 8192


def test_a_reasonable_reservation_is_left_alone(client, rec):
    _send(client, 2048)
    assert rec.bodies[-1]["max_tokens"] == 2048


def test_a_small_reservation_is_left_alone(client, rec):
    """Claude Code's cheap side-calls ask for 64 tokens; clamping must never raise a value."""
    _send(client, 64)
    assert rec.bodies[-1]["max_tokens"] == 64


def test_the_clamp_survives_serialisation(client, rec):
    """The mutation has to reach the wire. Assigning it after body_mutated was computed left the
    client's original bytes going upstream — the same ordering trap the thinking quirk hit."""
    _send(client, 64000)
    assert rec.bodies[-1]["max_tokens"] != 64000


def test_it_can_be_disabled(client, rec, monkeypatch):
    cfg = dict(P.load_rules_config())
    cfg["output_budget"] = {"enabled": False, "max_tokens": 8192}
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    _send(client, 64000)
    assert rec.bodies[-1]["max_tokens"] == 64000


def test_the_cap_is_configurable(client, rec, monkeypatch):
    cfg = dict(P.load_rules_config())
    cfg["output_budget"] = {"enabled": True, "max_tokens": 1024}
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    _send(client, 64000)
    assert rec.bodies[-1]["max_tokens"] == 1024
