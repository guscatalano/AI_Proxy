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
