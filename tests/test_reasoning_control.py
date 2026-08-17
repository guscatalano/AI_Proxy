"""Turning thinking off with the knob the engine actually reads.

Engines disagree, and picking the wrong knob fails silently: the request is accepted, the field
ignored, and the model thinks anyway. Measured on Ollama 0.32.13 with gemma4:26b, one question,
max_tokens=200:

    no knob                  -> 0 chars content, 918 reasoning, budget exhausted
    chat_template_kwargs     -> 0 chars content, 861 reasoning   (ignored outright)
    reasoning_effort="none"  -> 175 chars content, 0 reasoning, 32 tokens

The first two are the same result — nothing the client can use. In production this cost a
claude-code turn 1,952 tokens over 100 seconds that were entirely reasoning: the Anthropic
bridge drops it, so the user waited a minute and a half for an empty reply. The quirk names its
lever so the same config cannot mean two different things on two backends.
"""
from ai_proxy import proxy as P


def _quirk_for(name):
    return P._model_quirk(name) or {}


# --- the quirk itself ----------------------------------------------------------------------


def test_gemma4_is_baked_in_not_left_to_the_file(client):
    """model_quirks.json lives inside the package and every deploy overwrites it; a copy placed
    one directory up never loaded at all. A quirk that matters belongs in code."""
    assert "gemma4" in P.DEFAULT_MODEL_QUIRKS


def test_gemma4_names_the_lever_ollama_reads(client):
    q = _quirk_for("gemma4:26b")
    assert q.get("reasoning_control") == "reasoning_effort"
    assert q.get("thinking") == "default_off_optin"


def test_ornith_keeps_the_lever_vllm_reads(client):
    """vLLM reads chat_template_kwargs. Changing the default would silently switch Ornith to a
    knob its engine ignores — the failure this whole field exists to prevent."""
    assert _quirk_for("ornith-nvfp4").get("reasoning_control") == "chat_template_kwargs.enable_thinking"


def test_the_prefix_match_covers_the_served_names(client):
    for name in ("gemma4:26b", "gemma4:latest", "gemma4-vllm"):
        assert _quirk_for(name).get("reasoning_control") == "reasoning_effort", name


def test_an_unrelated_model_gets_no_reasoning_quirk(client):
    assert not _quirk_for("qwen3-coder-next").get("reasoning_control")


# --- what actually goes out on the wire -----------------------------------------------------
#
# The quirk being present proves nothing: the previous version was configured correctly and
# still had no effect, because it wrote a field the engine ignores. These assert on the body.


import httpx  # noqa: E402
import pytest  # noqa: E402


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
            "id": "c1", "object": "chat.completion", "model": "gemma4:26b",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"completion_tokens": 1}}, headers={"content-type": "application/json"})


@pytest.fixture
def recorder(client):
    rec = _Recorder()
    real = client.app.state.client
    client.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(rec.handler))
    try:
        yield rec
    finally:
        client.app.state.client = real


def _send(client, **extra):
    client.post("/v1/chat/completions", json={
        "model": "gemma4:26b", "stream": False, "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}], **extra},
        headers={"x-proxy-no-router": "1"})


def test_reasoning_effort_none_is_sent_for_gemma4(client, recorder):
    _send(client)
    assert recorder.bodies, "no upstream request captured"
    assert recorder.bodies[0].get("reasoning_effort") == "none"


def test_the_ignored_knob_is_not_used_for_gemma4(client, recorder):
    """chat_template_kwargs on Ollama is the silent no-op that hid this bug for a day."""
    _send(client)
    assert "enable_thinking" not in (recorder.bodies[0].get("chat_template_kwargs") or {})


def test_an_explicit_request_for_reasoning_is_respected(client, recorder):
    """A caller naming its own budget is deciding; the proxy must not overrule it."""
    _send(client, reasoning_effort="high")
    assert recorder.bodies[0].get("reasoning_effort") == "high"
