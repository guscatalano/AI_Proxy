"""Record the context window the backend actually loaded.

A preload at num_ctx=65536 survives exactly until a request arrives without a context hint,
at which point Ollama reloads at OLLAMA_CONTEXT_LENGTH — 262144 on this host. That happened
mid-benchmark and no report could show it, because the only context the report knew came from
the catalogue, which records what a backend offers rather than what it did.
"""
import json

import pytest

from ai_proxy import proxy


class _Resp:
    def __init__(self, payload): self._p = payload
    def json(self): return self._p


class _Client:
    def __init__(self, payload): self._p = payload
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def get(self, url, **kw): return _Resp(self._p)


def _patch(monkeypatch, payload):
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda *a, **k: _Client(payload))


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_ollama_reports_the_context_it_loaded(monkeypatch):
    _patch(monkeypatch, {"models": [{"name": "qwen3.8:27b", "context_length": 262144}]})
    assert await proxy._bench_loaded_context("qwen3.8:27b", "ollama") == 262144


@pytest.mark.anyio
async def test_the_reloaded_value_is_what_gets_recorded(monkeypatch):
    """Asked for 65536, got 262144 — the report must be able to say so."""
    _patch(monkeypatch, {"models": [{"name": "m", "context_length": 262144}]})
    got = await proxy._bench_loaded_context("m", "ollama")
    assert got == 262144 and got != 65536


@pytest.mark.anyio
async def test_a_model_that_is_not_loaded_yields_nothing(monkeypatch):
    _patch(monkeypatch, {"models": [{"name": "other", "context_length": 8192}]})
    assert await proxy._bench_loaded_context("mine", "ollama") is None


@pytest.mark.anyio
async def test_vllm_reports_max_model_len(monkeypatch):
    _patch(monkeypatch, {"data": [{"id": "nemotron-vllm", "max_model_len": 65536}]})
    assert await proxy._bench_loaded_context("nemotron-vllm", "vllm") == 65536


@pytest.mark.anyio
async def test_a_sole_vllm_model_matches_even_under_a_served_alias(monkeypatch):
    """vLLM is launched with --served-model-name, so the id need not match what was asked."""
    _patch(monkeypatch, {"data": [{"id": "some-served-alias", "max_model_len": 262144}]})
    assert await proxy._bench_loaded_context("whatever", "vllm") == 262144


@pytest.mark.anyio
async def test_an_unreachable_backend_costs_the_metric_not_the_run(monkeypatch):
    class Boom:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **k): raise proxy.httpx.RequestError("down")
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda *a, **k: Boom())
    assert await proxy._bench_loaded_context("m", "ollama") is None


@pytest.mark.anyio
async def test_an_unknown_backend_is_not_guessed_at(monkeypatch):
    _patch(monkeypatch, {"models": []})
    assert await proxy._bench_loaded_context("m", "lmstudio") is None
