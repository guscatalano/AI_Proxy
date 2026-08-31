"""The protocol bridge must go where the router says, not always to Ollama.

The destination was hardwired to OLLAMA_URL, which silently outranked model_router. Pointing the
catch-all rule at a vLLM-served model moved every client except the only one that speaks
Anthropic — claude-code, whose lost tool calls were the reason for moving in the first place. It
answered 404 on every request ("The model `gemma4-vllm` does not exist") while the very same
model served /v1/chat/completions perfectly, which made it look like a vLLM fault rather than a
routing one.

Ollama stays the default, so an unrouted Anthropic request behaves exactly as before.
"""
import json

import httpx
import pytest

from ai_proxy import proxy as P


_ANTHROPIC_REQUEST = {
    "model": "claude-sonnet-4",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "hello"}],
}


class _Recorder:
    """Answers any upstream and records which base URL was dialled."""

    def __init__(self):
        self.urls: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "model": "test",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"completion_tokens": 1},
        }, headers={"content-type": "application/json"})


@pytest.fixture
def recorder(client):
    rec = _Recorder()
    real = client.app.state.client
    client.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(rec.handler))
    try:
        yield rec
    finally:
        client.app.state.client = real


def _route_all_to(monkeypatch, model, upstream=None):
    rule = {"if": {}, "then": model}
    if upstream:
        rule["upstream"] = upstream
    base = dict(P.load_rules_config())
    base["model_router"] = {"enabled": True, "rules": [rule], "aliases": {}, "advertise": {}}
    base["protocol_bridge"] = {"enabled": True}
    monkeypatch.setattr(P, "load_rules_config", lambda: base)


def test_the_bridge_follows_the_router_to_vllm(client, recorder, monkeypatch):
    _route_all_to(monkeypatch, "gemma4-vllm", "vllm")
    client.post("/v1/messages", json=_ANTHROPIC_REQUEST)
    assert recorder.urls, "no upstream request was made at all"
    assert P.VLLM_URL in recorder.urls[0], (
        f"bridge ignored the router and dialled {recorder.urls[0]!r}")


def test_the_bridge_still_defaults_to_ollama(client, recorder, monkeypatch):
    """No upstream named on the rule: behaviour must be exactly what it always was."""
    _route_all_to(monkeypatch, "some-ollama-model")
    client.post("/v1/messages", json=_ANTHROPIC_REQUEST)
    assert recorder.urls
    assert P.OLLAMA_URL in recorder.urls[0]


def test_a_header_pin_overrides_the_rule(client, recorder, monkeypatch):
    _route_all_to(monkeypatch, "some-model")
    client.post("/v1/messages", json=_ANTHROPIC_REQUEST,
                headers={"x-proxy-upstream": "vllm"})
    assert recorder.urls
    assert P.VLLM_URL in recorder.urls[0]


def test_anthropic_is_never_a_bridge_target(client, recorder, monkeypatch):
    """The translation only runs one way. Sending an OpenAI-shape body to Anthropic would 400,
    so a rule naming it must fall back rather than be obeyed."""
    _route_all_to(monkeypatch, "some-model", "anthropic")
    client.post("/v1/messages", json=_ANTHROPIC_REQUEST)
    assert recorder.urls
    assert P.OLLAMA_URL in recorder.urls[0]


def test_an_unknown_upstream_falls_back_rather_than_failing(client, recorder, monkeypatch):
    _route_all_to(monkeypatch, "some-model", "not-a-real-backend")
    client.post("/v1/messages", json=_ANTHROPIC_REQUEST)
    assert recorder.urls
    assert P.OLLAMA_URL in recorder.urls[0]


def test_the_translated_body_is_openai_shaped(client, recorder, monkeypatch):
    """Whatever the destination, the bridge's job is still the translation."""
    _route_all_to(monkeypatch, "gemma4-vllm", "vllm")
    client.post("/v1/messages", json=_ANTHROPIC_REQUEST)
    assert "v1/chat/completions" in recorder.urls[0]
