"""Free memory by stopping an idle backend — but only deliberately.

Ollama evicts its own models and holds nothing when idle, so "make room" here means stopping the
vLLM container that is serving production. That is a real cost: a vLLM cold start on this box
measured 400-500s, so an evicting request takes production down for the best part of ten minutes
and it does not return until something asks for it.

Hence off by default, and guarded by idleness. A refusal that declines to evict says why, so the
setting never looks like it did nothing.
"""
import asyncio
import time

from ai_proxy import proxy as P


def _cfg(monkeypatch, *, enabled, idle_s=600):
    base = dict(P.load_rules_config())
    base["backend_fit"] = {"enabled": True, "upstreams": ["ollama"],
                           "evict": {"enabled": enabled, "idle_s": idle_s,
                                     "evict_upstreams": ["vllm"]}}
    monkeypatch.setattr(P, "load_rules_config", lambda: base)


def _last_vllm_request(client, ago_s):
    conn = P.db()
    conn.execute("DELETE FROM requests WHERE id='evict_probe'")
    # Other tests in this session post through the client and leave real vllm rows behind, which
    # would make the idleness check see traffic seconds old whatever this helper inserts.
    conn.execute("UPDATE requests SET upstream='_test_parked' WHERE upstream='vllm'")
    conn.execute("INSERT INTO requests (id, ts, method, path, upstream_url, model, is_stream, "
                 "client_ip, client_app, upstream) VALUES (?,?,?,?,?,?,?,?,?,?)",
                 ("evict_probe", time.time() - ago_s, "POST", "/v1/chat/completions",
                  "http://x", "m", 0, "127.0.0.1", "test", "vllm"))
    conn.commit()
    conn.close()


def _cleanup():
    conn = P.db()
    conn.execute("DELETE FROM requests WHERE id='evict_probe'")
    conn.commit()
    conn.close()


def test_it_is_off_by_default(client):
    cfg = (P.load_rules_config().get("backend_fit") or {}).get("evict") or {}
    assert cfg.get("enabled") is False, "stopping production must be opted into"


def test_disabled_declines_and_says_so(client, monkeypatch):
    _cfg(monkeypatch, enabled=False)
    why = asyncio.run(P._evict_for_fit("refusal", "some-model"))
    assert why and "eviction is off" in why


def test_a_recently_used_backend_is_not_evicted(client, monkeypatch):
    """One stalled request is better than everybody's requests stalling."""
    _cfg(monkeypatch, enabled=True, idle_s=600)
    _last_vllm_request(client, ago_s=30)
    try:
        why = asyncio.run(P._evict_for_fit("refusal", "some-model"))
        assert why and "served a request" in why and "active use" in why
    finally:
        _cleanup()


def test_an_idle_backend_is_evicted(client, monkeypatch):
    _cfg(monkeypatch, enabled=True, idle_s=60)
    _last_vllm_request(client, ago_s=3600)
    stopped = []

    class _Fake:
        async def stop(self):
            stopped.append("vllm")

    monkeypatch.setitem(P.PROVIDERS, "vllm", _Fake())
    monkeypatch.setattr(P, "_mem_snapshot", lambda: {"avail_mb": 100 * 1024,
                                                     "total_mb": 121 * 1024, "used_mb": 21 * 1024})
    try:
        assert asyncio.run(P._evict_for_fit("refusal", "some-model")) is None
        assert stopped == ["vllm"]
    finally:
        _cleanup()


def test_a_failed_stop_is_reported_not_swallowed(client, monkeypatch):
    _cfg(monkeypatch, enabled=True, idle_s=60)
    _last_vllm_request(client, ago_s=3600)

    class _Broken:
        async def stop(self):
            raise RuntimeError("docker is unhappy")

    monkeypatch.setitem(P.PROVIDERS, "vllm", _Broken())
    try:
        why = asyncio.run(P._evict_for_fit("refusal", "some-model"))
        assert why and "failed" in why
    finally:
        _cleanup()


# --- a sweep is the case eviction exists for ------------------------------------------------------
#
# The idle guard defeated its own purpose: a sweep moves between models, so the backend it needs
# to evict is the one IT was using seconds ago. Every swap would be refused, and eviction would be
# useless for the only workload that wants it. A run with exclusive=true has already quiesced live
# traffic through panic mode, so idleness has nobody left to protect.


def _bench_running(client, running=True):
    conn = P.db()
    conn.execute("DELETE FROM bench_runs WHERE id='evict_bench'")
    if running:
        conn.execute("INSERT INTO bench_runs (id, ts, model, status, progress, progress_total, "
                     "config_json) VALUES (?,?,?,?,?,?,?)",
                     ("evict_bench", time.time(), "m", "running", 1, 10, "{}"))
    conn.commit()
    conn.close()


def _bench_cleanup():
    conn = P.db()
    conn.execute("DELETE FROM bench_runs WHERE id='evict_bench'")
    conn.commit()
    conn.close()


def test_a_running_bench_evicts_a_just_used_backend(client, monkeypatch):
    _cfg(monkeypatch, enabled=True, idle_s=600)
    _last_vllm_request(client, ago_s=5)        # the model the sweep just finished measuring
    _bench_running(client, True)
    stopped = []

    class _Fake:
        async def stop(self):
            stopped.append("vllm")

    monkeypatch.setitem(P.PROVIDERS, "vllm", _Fake())
    monkeypatch.setattr(P, "_mem_snapshot", lambda: {"avail_mb": 100 * 1024,
                                                     "total_mb": 121 * 1024, "used_mb": 21 * 1024})
    try:
        assert asyncio.run(P._evict_for_fit("refusal", "next-model")) is None
        assert stopped == ["vllm"], "a sweep must be able to swap backends"
    finally:
        _bench_cleanup()
        _cleanup()


def test_without_a_bench_the_idle_guard_still_holds(client, monkeypatch):
    """Live traffic keeps its protection — this is not a blanket exemption."""
    _cfg(monkeypatch, enabled=True, idle_s=600)
    _last_vllm_request(client, ago_s=5)
    _bench_running(client, False)
    try:
        why = asyncio.run(P._evict_for_fit("refusal", "next-model"))
        assert why and "active use" in why
    finally:
        _bench_cleanup()
        _cleanup()


# --- a benchmark driven from outside this proxy ---------------------------------------------------
#
# An external sweep has no row in bench_runs, so the internal owner check cannot see it. Whoever
# is driving the sweep is the only party who knows the box is theirs, so they declare it per
# request with x-proxy-evict.


def test_a_caller_can_declare_ownership(client, monkeypatch):
    _cfg(monkeypatch, enabled=True, idle_s=600)
    _last_vllm_request(client, ago_s=5)        # the model the external sweep just measured
    stopped = []

    class _Fake:
        async def stop(self):
            stopped.append("vllm")

    monkeypatch.setitem(P.PROVIDERS, "vllm", _Fake())
    monkeypatch.setattr(P, "_mem_snapshot", lambda: {"avail_mb": 100 * 1024,
                                                     "total_mb": 121 * 1024, "used_mb": 21 * 1024})
    try:
        assert asyncio.run(P._evict_for_fit("r", "next", caller_owns=True)) is None
        assert stopped == ["vllm"]
    finally:
        _cleanup()


def test_without_the_header_the_guard_still_applies(client, monkeypatch):
    _cfg(monkeypatch, enabled=True, idle_s=600)
    _last_vllm_request(client, ago_s=5)
    try:
        why = asyncio.run(P._evict_for_fit("r", "next", caller_owns=False))
        assert why and "active use" in why
    finally:
        _cleanup()


# --- the blocker is often Ollama itself ------------------------------------------------------------
#
# Eviction only ever stopped vLLM, and missed that Ollama parks up to OLLAMA_MAX_LOADED_MODELS
# for OLLAMA_KEEP_ALIVE — three models for two hours here. A sweep that had already visited two
# models was holding 47 GB of its own before the third was requested, so the request was refused
# with vLLM already stopped and nothing left for eviction to do.


def test_ollama_residents_are_released_before_stopping_a_container(client, monkeypatch):
    """Releasing them is free; stopping vLLM costs a 400-500s restart. Try the cheap one first."""
    _cfg(monkeypatch, enabled=True, idle_s=60)
    _last_vllm_request(client, ago_s=3600)
    stopped, released = [], []

    class _Fake:
        async def stop(self):
            stopped.append("vllm")

    async def _release(reason=""):
        released.append(reason)
        return ["some-resident:latest"]

    monkeypatch.setitem(P.PROVIDERS, "vllm", _Fake())
    monkeypatch.setattr(P, "_free_ollama_models", _release)
    # Once the residents are gone the model fits, so no container should be stopped.
    monkeypatch.setattr(P, "_ollama_fit_refusal", lambda m, **k: _none())
    try:
        assert asyncio.run(P._evict_for_fit("r", "wanted", caller_owns=True)) is None
        assert released, "ollama residents should be released"
        assert stopped == [], "no container stop needed once the residents were freed"
    finally:
        _cleanup()


async def _none():
    return None


# --- eviction buys memory, and some models cannot be bought room ------------------------------
#
# qwen3.6:35b-a3b prices at ~340 GB of KV on a 122 GB box. Asking for it stopped the container
# serving production to chase a load that could never succeed, and then held the request for the
# full release timeout before refusing anyway. One refusal became eight minutes of outage.


def test_a_model_bigger_than_the_machine_never_triggers_eviction(client, monkeypatch):
    _cfg(monkeypatch, enabled=True, idle_s=60)
    _last_vllm_request(client, ago_s=3600)      # idle, so nothing else would stop it
    stopped = []

    class _Fake:
        async def stop(self):
            stopped.append("vllm")

    monkeypatch.setitem(P.PROVIDERS, "vllm", _Fake())
    try:
        why = asyncio.run(P._evict_for_fit("r", "huge-model", caller_owns=True, _impossible=True))
        assert why and "no amount of eviction" in why
        assert stopped == [], "nothing may be stopped for a load that cannot succeed"
    finally:
        _cleanup()


def test_a_model_that_merely_needs_room_still_evicts(client, monkeypatch):
    """The guard must not become a blanket refusal — this is the case eviction exists for."""
    _cfg(monkeypatch, enabled=True, idle_s=60)
    _last_vllm_request(client, ago_s=3600)
    stopped = []

    class _Fake:
        async def stop(self):
            stopped.append("vllm")

    async def _fits(model, assume_avail_mb=0, pin=None):
        return None

    monkeypatch.setitem(P.PROVIDERS, "vllm", _Fake())
    monkeypatch.setattr(P, "_free_ollama_models", lambda reason="": _empty_list())
    monkeypatch.setattr(P, "_ollama_fit_refusal", _fits)
    monkeypatch.setattr(P, "_mem_snapshot", lambda: {"avail_mb": 100 * 1024,
                                                     "total_mb": 121 * 1024, "used_mb": 21 * 1024})
    try:
        assert asyncio.run(P._evict_for_fit("r", "big", caller_owns=True, _impossible=False)) is None
    finally:
        _cleanup()


async def _empty_list():
    return []


# --- a backend that has only just started is not idle -----------------------------------------
#
# It has served nothing, which reads as maximally idle, so the first request it cannot fit stops
# it — throwing away a load that takes five to seven minutes. This happened to vLLM a minute
# after it came back up from the previous incident.


def test_a_freshly_started_backend_is_not_evicted(client, monkeypatch):
    _cfg(monkeypatch, enabled=True, idle_s=600)
    stopped = []

    class _Fake:
        async def stop(self):
            stopped.append("vllm")

    async def _uptime(name):
        return 30.0

    monkeypatch.setitem(P.PROVIDERS, "vllm", _Fake())
    monkeypatch.setattr(P, "_backend_uptime_s", _uptime)
    monkeypatch.setattr(P, "_free_ollama_models", lambda reason="": _empty_list())
    try:
        why = asyncio.run(P._evict_for_fit("r", "wanted"))
        assert why and "coming up" in why
        assert stopped == []
    finally:
        _cleanup()


def test_a_long_running_idle_backend_is_still_evictable(client, monkeypatch):
    _cfg(monkeypatch, enabled=True, idle_s=600)
    stopped = []

    class _Fake:
        async def stop(self):
            stopped.append("vllm")

    async def _uptime(name):
        return 40000.0

    async def _fits(model, assume_avail_mb=0, pin=None):
        return None

    monkeypatch.setitem(P.PROVIDERS, "vllm", _Fake())
    monkeypatch.setattr(P, "_backend_uptime_s", _uptime)
    monkeypatch.setattr(P, "_free_ollama_models", lambda reason="": _empty_list())
    monkeypatch.setattr(P, "_ollama_fit_refusal", _fits)
    try:
        assert asyncio.run(P._evict_for_fit("r", "wanted")) is None
        assert stopped == ["vllm"]
    finally:
        _cleanup()


def test_an_unknown_uptime_does_not_block_eviction(client, monkeypatch):
    """None means "cannot tell". Treating it as young would disable eviction wherever docker
    is not reachable."""
    _cfg(monkeypatch, enabled=True, idle_s=600)
    stopped = []

    class _Fake:
        async def stop(self):
            stopped.append("vllm")

    async def _uptime(name):
        return None

    async def _fits(model, assume_avail_mb=0, pin=None):
        return None

    monkeypatch.setitem(P.PROVIDERS, "vllm", _Fake())
    monkeypatch.setattr(P, "_backend_uptime_s", _uptime)
    monkeypatch.setattr(P, "_free_ollama_models", lambda reason="": _empty_list())
    monkeypatch.setattr(P, "_ollama_fit_refusal", _fits)
    try:
        assert asyncio.run(P._evict_for_fit("r", "wanted")) is None
        assert stopped == ["vllm"]
    finally:
        _cleanup()


def test_a_bench_may_still_swap_a_backend_it_just_started(client, monkeypatch):
    """A sweep starts and stops backends by design; the guard would refuse every swap."""
    _cfg(monkeypatch, enabled=True, idle_s=600)
    stopped = []

    class _Fake:
        async def stop(self):
            stopped.append("vllm")

    async def _uptime(name):
        return 5.0

    async def _fits(model, assume_avail_mb=0, pin=None):
        return None

    monkeypatch.setitem(P.PROVIDERS, "vllm", _Fake())
    monkeypatch.setattr(P, "_backend_uptime_s", _uptime)
    monkeypatch.setattr(P, "_free_ollama_models", lambda reason="": _empty_list())
    monkeypatch.setattr(P, "_ollama_fit_refusal", _fits)
    try:
        assert asyncio.run(P._evict_for_fit("r", "wanted", caller_owns=True)) is None
        assert stopped == ["vllm"]
    finally:
        _cleanup()
