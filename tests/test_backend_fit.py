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


def _fake_ollama(*, resident=(), size_gb=51.0, ctx=262144, arch="qwen3next",
                 kv_measurable=True):
    """Stand in for Ollama's /api/ps, /api/show and /api/tags."""
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/api/ps"):
            return httpx.Response(200, json={"models": [{"name": n} for n in resident]})
        if path.endswith("/api/show"):
            info = {"general.architecture": arch, f"{arch}.context_length": ctx}
            if kv_measurable:
                info.update({f"{arch}.block_count": 64,
                             f"{arch}.attention.head_count_kv": 8,
                             f"{arch}.attention.key_length": 128,
                             f"{arch}.attention.value_length": 128,
                             f"{arch}.embedding_length": 5120})
            return httpx.Response(200, json={"model_info": info, "parameters": ""})
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


def test_weights_alone_are_enough_to_refuse(client, ollama):
    """_bench_ollama_kv_mb returns None for architectures it does not recognise — qwen3next
    among them — and that was letting the real crash-looping model through. 48 GB of weights
    into 36 GB free is disqualifying whatever the KV cache turns out to be."""
    handler = _fake_ollama(size_gb=48.0, kv_measurable=False)
    ollama(handler, avail_gb=36)
    P._OLLAMA_META_CACHE.clear()
    msg = asyncio.run(P._ollama_fit_refusal("big-model:latest"))
    assert msg, "must refuse on weights alone when the KV size is unknown"
    assert "could not size" in msg, "the message should admit what it could not measure"


# --- the gate and the context pin must agree -------------------------------------------------
#
# The fit gate runs before ollama_options rewrites the request to a context-pinned derivative,
# so it sees the base name. Pricing that name at the 262k server default refuses a model that
# was about to be loaded at 16k: qwen3.6:35b-a3b costs ~655 GB at the default and ~32 GB pinned,
# and only the second number describes the load that would actually happen.


def test_a_configured_pin_is_what_the_gate_prices(monkeypatch):
    monkeypatch.setattr(P, "load_rules_config", lambda: {
        "ollama_options": {"enabled": True, "per_model": {"big": {"num_ctx": 16384}}}})
    assert P._configured_num_ctx("big") == 16384


def test_an_unpinned_model_gets_no_pin(monkeypatch):
    monkeypatch.setattr(P, "load_rules_config", lambda: {
        "ollama_options": {"enabled": True, "per_model": {"other": {"num_ctx": 16384}}}})
    assert P._configured_num_ctx("big") is None


def test_a_disabled_rule_pins_nothing(monkeypatch):
    """A pin that will not be applied must not be priced as though it were."""
    monkeypatch.setattr(P, "load_rules_config", lambda: {
        "ollama_options": {"enabled": False, "per_model": {"big": {"num_ctx": 16384}}}})
    assert P._configured_num_ctx("big") is None


def test_a_global_default_pins_every_model(monkeypatch):
    monkeypatch.setattr(P, "load_rules_config", lambda: {
        "ollama_options": {"enabled": True, "defaults": {"num_ctx": 32768}}})
    assert P._configured_num_ctx("anything") == 32768


# --- the window priced must be the window loaded ---------------------------------------------
#
# The pin only exists on the OpenAI-compat path, where ollama_options swaps the request for a
# context-pinned derivative. A native /api/chat request keeps the model it named, so pricing it
# at the pin would wave through the 262k allocation the pin was meant to avoid.


def _fit_body(pin):
    """The refusal text quotes the window it priced, which is what makes this observable."""
    P._OLLAMA_FIT_CACHE.clear()
    return pin


def test_the_gate_prices_a_pinned_request_at_its_pin(client, monkeypatch):
    seen = []

    async def _spy(model, assume_avail_mb=0, pin=None):
        seen.append(pin)
        return None

    monkeypatch.setattr(P, "_ollama_fit_refusal", _spy)
    monkeypatch.setattr(P, "load_rules_config", lambda: {
        "backend_fit": {"enabled": True, "upstreams": ["ollama"]},
        "ollama_options": {"enabled": True, "per_model": {"pinned": {"num_ctx": 16384}}}})
    client.post("/v1/chat/completions", json={"model": "pinned", "messages": [{"role": "user",
                                                                              "content": "hi"}]})
    assert 16384 in seen


def test_a_native_request_is_priced_at_the_num_ctx_it_carries(client, monkeypatch):
    """Ollama honours num_ctx on /api/chat, so that — not the config pin — is the real window."""
    seen = []

    async def _spy(model, assume_avail_mb=0, pin=None):
        seen.append(pin)
        return None

    monkeypatch.setattr(P, "_ollama_fit_refusal", _spy)
    monkeypatch.setattr(P, "load_rules_config", lambda: {
        "backend_fit": {"enabled": True, "upstreams": ["ollama"]},
        "ollama_options": {"enabled": True, "per_model": {"pinned": {"num_ctx": 16384}}}})
    client.post("/api/chat", json={"model": "pinned", "messages": [{"role": "user",
                                                                    "content": "hi"}],
                                   "options": {"num_ctx": 65536}})
    assert seen and seen[0] == 65536, "the config pin does not apply to a native request"


def test_a_native_request_without_num_ctx_is_priced_at_the_server_default(client, monkeypatch):
    seen = []

    async def _spy(model, assume_avail_mb=0, pin=None):
        seen.append(pin)
        return None

    monkeypatch.setattr(P, "_ollama_fit_refusal", _spy)
    monkeypatch.setattr(P, "load_rules_config", lambda: {
        "backend_fit": {"enabled": True, "upstreams": ["ollama"]},
        "ollama_options": {"enabled": True, "per_model": {"pinned": {"num_ctx": 16384}}}})
    client.post("/api/chat", json={"model": "pinned", "messages": [{"role": "user",
                                                                    "content": "hi"}]})
    assert seen and seen[0] is None, "nothing named a smaller window, so the default stands"
