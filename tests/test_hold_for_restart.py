"""A backend that drops the connection is usually restarting, not gone.

The incident: qwen3.6's vLLM engine died on a GDN kernel fault (Xid 13 / 31, misaligned
address). docker's restart policy brought the container straight back, but reloading 20 GiB of
weights takes minutes — and every request that arrived in that window was rejected in under two
seconds with "upstream unreachable". Five requests failed that would all have succeeded had the
proxy waited.
"""
import asyncio

import httpx

from ai_proxy import proxy as P


class _Prov:
    def __init__(self, ready_result=True):
        self.control = "docker"
        self.base_url = "http://localhost:8002"
        self.ready_calls = []
        self._ready = ready_result

    async def ready(self, timeout_s=900.0):
        self.ready_calls.append(timeout_s)
        return self._ready


def _cfg(monkeypatch, **hold):
    cfg = dict(P.load_rules_config())
    mc = dict(cfg.get("model_control") or {})
    mc["hold_for_restart"] = {"enabled": True, "timeout_s": 0, **hold}
    cfg["model_control"] = mc
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)


def test_a_dropped_connection_waits_for_the_backend(client, monkeypatch):
    prov = _Prov(ready_result=True)
    monkeypatch.setitem(P.PROVIDERS, "vllm2", prov)
    _cfg(monkeypatch)
    held = asyncio.run(P._wait_out_backend_restart("vllm2", httpx.ReadError("")))
    assert held is True
    assert prov.ready_calls, "it must actually wait on the provider, not just return"


def test_it_gives_up_when_the_backend_stays_down(client, monkeypatch):
    """Provider.ready() consults died(), so a container that exited for good fails fast."""
    prov = _Prov(ready_result=False)
    monkeypatch.setitem(P.PROVIDERS, "vllm2", prov)
    _cfg(monkeypatch)
    assert asyncio.run(P._wait_out_backend_restart("vllm2", httpx.ConnectError(""))) is False


def test_a_slow_backend_is_not_retried(client, monkeypatch):
    """ReadTimeout means it accepted and is still thinking. Retrying would put a second copy of
    an expensive prefill on a box that is already busy."""
    prov = _Prov(ready_result=True)
    monkeypatch.setitem(P.PROVIDERS, "vllm2", prov)
    _cfg(monkeypatch)
    assert asyncio.run(P._wait_out_backend_restart("vllm2", httpx.ReadTimeout(""))) is False
    assert prov.ready_calls == [], "must not even wait for a backend that is merely slow"


def test_on_demand_backends_are_not_held(client, monkeypatch):
    """Ollama and LM Studio load on demand — a failure there is not a restart to wait out."""
    _cfg(monkeypatch)
    for name in ("ollama", "lmstudio"):
        assert asyncio.run(P._wait_out_backend_restart(name, httpx.ReadError(""))) is False


def test_both_vllm_slots_and_llamacpp_qualify(client, monkeypatch):
    _cfg(monkeypatch)
    for name in ("vllm", "vllm2", "llamacpp"):
        prov = _Prov(ready_result=True)
        monkeypatch.setitem(P.PROVIDERS, name, prov)
        assert asyncio.run(P._wait_out_backend_restart(name, httpx.ReadError(""))) is True


def test_it_can_be_turned_off(client, monkeypatch):
    prov = _Prov(ready_result=True)
    monkeypatch.setitem(P.PROVIDERS, "vllm2", prov)
    _cfg(monkeypatch, enabled=False)
    assert asyncio.run(P._wait_out_backend_restart("vllm2", httpx.ReadError(""))) is False
    assert prov.ready_calls == []


def test_zero_timeout_falls_back_to_the_ready_timeout(client, monkeypatch):
    """0 means "use vllm_ready_timeout_s", not "wait zero seconds"."""
    prov = _Prov(ready_result=True)
    monkeypatch.setitem(P.PROVIDERS, "vllm2", prov)
    _cfg(monkeypatch, timeout_s=0)
    asyncio.run(P._wait_out_backend_restart("vllm2", httpx.ReadError("")))
    assert prov.ready_calls == [P._vllm_ready_timeout()]


def test_an_explicit_timeout_is_honoured(client, monkeypatch):
    prov = _Prov(ready_result=True)
    monkeypatch.setitem(P.PROVIDERS, "vllm2", prov)
    _cfg(monkeypatch, timeout_s=45)
    asyncio.run(P._wait_out_backend_restart("vllm2", httpx.ReadError("")))
    assert prov.ready_calls == [45.0]


def _failing_client(monkeypatch, sends, fail_times=2):
    """Swap the proxy's upstream client for one that records sends and raises."""
    real = P.app.state.client

    class _C:
        def build_request(self, *a, **k):
            return real.build_request(*a, **k)

        async def send(self, req, **k):
            sends.append(req)
            if len(sends) <= fail_times:
                raise httpx.ReadError("engine died")
            raise httpx.ReadError("still down")

        def __getattr__(self, name):
            return getattr(real, name)

    P.app.state.client = _C()
    monkeypatch.setattr(P.app.state, "client", _C(), raising=False)
    return real


def test_the_send_path_actually_retries_after_a_hold(client, monkeypatch):
    """The helper being correct proves nothing — this asserts the forward path calls it and
    re-sends. Without the retry loop, a dropped connection was one send and an instant 502."""
    sends = []
    real = _failing_client(monkeypatch, sends)

    async def held(upstream, exc):
        return True
    monkeypatch.setattr(P, "_wait_out_backend_restart", held)
    try:
        r = client.post("/v1/chat/completions",
                        json={"model": "nemotron", "messages": [{"role": "user", "content": "hi"}]},
                        headers={"x-client-name": "hold-e2e"})
    finally:
        P.app.state.client = real
    assert len(sends) == 2, f"expected one retry after the hold, got {len(sends)} send(s)"
    assert r.status_code == 502, "the second failure must still surface honestly"


def test_no_hold_means_no_retry(client, monkeypatch):
    """A backend that is genuinely gone must fail fast, exactly as before."""
    sends = []
    real = _failing_client(monkeypatch, sends)

    async def not_held(upstream, exc):
        return False
    monkeypatch.setattr(P, "_wait_out_backend_restart", not_held)
    try:
        r = client.post("/v1/chat/completions",
                        json={"model": "nemotron", "messages": [{"role": "user", "content": "hi"}]},
                        headers={"x-client-name": "nohold-e2e"})
    finally:
        P.app.state.client = real
    assert len(sends) == 1, "must not retry when the backend is not coming back"
    assert r.status_code == 502
