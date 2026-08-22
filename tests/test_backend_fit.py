"""Refuse a request that would kill the backend, instead of forwarding it.

Ollama allocates weights plus KV cache on load. A model that does not fit gets the service
OOM-killed, and the next request kills the restarted one: 303 restarts in an afternoon on this
box, while the client only ever saw "ollama ended the response early". One refusal costs one
request; forwarding costs the backend and everything queued behind it.

The arithmetic already existed for benchmarks, measured against TOTAL memory because a bench
owns the box. A live request does not — with vLLM holding 85 of 121 GB, what matters is what is
free.
"""
import asyncio
import json

import httpx
import pytest

from ai_proxy import proxy as P


def _fake_ollama(*, resident=(), size_gb=51.0, ctx=262144, arch="qwen3next"):
    """Stand in for Ollama's /api/ps, /api/show and /api/tags."""
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/api/ps"):
            return httpx.Response(200, json={"models": [{"name": n} for n in resident]})
        if path.endswith("/api/show"):
            return httpx.Response(200, json={
                "model_info": {"general.architecture": arch,
                               f"{arch}.context_length": ctx,
                               f"{arch}.block_count": 64,
                               f"{arch}.attention.head_count_kv": 8,
                               f"{arch}.attention.key_length": 128,
                               f"{arch}.attention.value_length": 128,
                               f"{arch}.embedding_length": 5120},
                "parameters": ""})
        if path.endswith("/api/tags"):
            return httpx.Response(200, json={"models": [
                {"name": "big-model:latest", "size": int(size_gb * 1024 ** 3)}]})
        return httpx.Response(404)
    return handler


@pytest.fixture
def ollama(monkeypatch):
    def _install(handler, avail_gb):
        real = httpx.AsyncClient

        class _Client(real):
            def __init__(self, *a, **k):
                k["transport"] = httpx.MockTransport(handler)
                super().__init__(*a, **k)

        monkeypatch.setattr(httpx, "AsyncClient", _Client)
        monkeypatch.setattr(P, "_mem_snapshot",
                            lambda: {"total_mb": 121 * 1024, "avail_mb": int(avail_gb * 1024),
                                     "used_mb": int((121 - avail_gb) * 1024)})
        P._OLLAMA_FIT_CACHE.clear()
    return _install


def test_a_model_that_cannot_fit_is_refused(client, ollama):
    """51 GB of weights against 35 GB free — the case that produced the crash loop."""
    ollama(_fake_ollama(size_gb=51.0), avail_gb=35)
    msg = asyncio.run(P._ollama_fit_refusal("big-model:latest"))
    assert msg and "cannot be loaded right now" in msg
    assert "51" in msg and "GB" in msg, "the arithmetic has to be in the message"


def test_a_model_that_fits_is_allowed(client, ollama):
    ollama(_fake_ollama(size_gb=8.0, ctx=8192), avail_gb=100)
    assert asyncio.run(P._ollama_fit_refusal("big-model:latest")) is None


def test_a_resident_model_is_always_allowed(client, ollama):
    """It is already loaded, so no allocation is coming however large it is."""
    ollama(_fake_ollama(resident=("big-model:latest",), size_gb=51.0), avail_gb=2)
    assert asyncio.run(P._ollama_fit_refusal("big-model:latest")) is None


def test_an_unreachable_backend_does_not_block(client, ollama):
    """Not knowing must never mean refusing — that would turn a blip into an outage."""
    def dead(request):
        raise httpx.ConnectError("nothing listening")
    ollama(dead, avail_gb=35)
    assert asyncio.run(P._ollama_fit_refusal("big-model:latest")) is None


def test_an_unknown_model_does_not_block(client, ollama):
    def not_found(request):
        return httpx.Response(404)
    ollama(not_found, avail_gb=35)
    assert asyncio.run(P._ollama_fit_refusal("big-model:latest")) is None


def test_the_verdict_is_cached(client, ollama):
    """It runs per request; /api/show on every call would add load to a struggling backend."""
    calls = {"n": 0}
    base = _fake_ollama(size_gb=51.0)

    def counting(request):
        calls["n"] += 1
        return base(request)
    ollama(counting, avail_gb=35)
    asyncio.run(P._ollama_fit_refusal("big-model:latest"))
    first = calls["n"]
    asyncio.run(P._ollama_fit_refusal("big-model:latest"))
    assert calls["n"] == first, "second call should have used the cache"


def test_the_rule_is_on_by_default(client):
    cfg = P.load_rules_config().get("backend_fit") or {}
    assert cfg.get("enabled") is True
    assert "ollama" in cfg.get("upstreams", [])


# --- the guard has to work when the backend is already broken -----------------------------------
#
# The first version failed exactly when it mattered. Once a model has crashed Ollama, /api/tags
# stops answering, so the guard could not learn what the model weighed, and its "never block on
# ignorance" rule let the next request through to kill the restarted process. Remembering what
# was learned while the backend was healthy is what breaks the loop.


def test_it_still_refuses_once_the_backend_has_stopped_answering(client, ollama):
    ollama(_fake_ollama(size_gb=51.0), avail_gb=35)
    first = asyncio.run(P._ollama_fit_refusal("big-model:latest"))
    assert first, "should refuse while healthy"

    P._OLLAMA_FIT_CACHE.clear()          # force a fresh decision

    def dead(request):
        raise httpx.ConnectError("crash looping")
    ollama(dead, avail_gb=35)
    # The size cache is deliberately not cleared: that is the state after a crash.
    again = asyncio.run(P._ollama_fit_refusal("big-model:latest"))
    assert again, "must keep refusing using what it learned before the crash"


def test_a_model_never_seen_healthy_is_not_blocked(client, ollama):
    """Refusing something it has never been able to measure would be guessing."""
    P._OLLAMA_META_CACHE.clear()

    def dead(request):
        raise httpx.ConnectError("nothing listening")
    ollama(dead, avail_gb=35)
    assert asyncio.run(P._ollama_fit_refusal("never-seen:latest")) is None
