"""Tests for the performance guards: body-size cap, analytics offloading, caching,
and system-history downsampling. See the perf investigation that motivated these."""
import inspect
import time

from ai_proxy import proxy as P


def test_truncate_none_and_small_unchanged():
    assert P._truncate_for_store(None) is None
    assert P._truncate_for_store("") == ""
    assert P._truncate_for_store("hello") == "hello"


def test_truncate_over_cap_marks(monkeypatch):
    monkeypatch.setattr(P, "MAX_STORED_BODY", 100)
    out = P._truncate_for_store("x" * 500)
    assert out.startswith("x" * 100)
    assert len(out) < 500
    assert "truncated by ai-proxy" in out


def test_truncate_disabled_passes_through(monkeypatch):
    monkeypatch.setattr(P, "MAX_STORED_BODY", 0)
    big = "x" * 1000
    assert P._truncate_for_store(big) == big


def test_analytics_handlers_are_sync():
    # These must be plain `def` so Starlette runs them in a threadpool; an `async def`
    # here would run their blocking SQLite queries on the event loop and stall proxying.
    for name in ("stats", "suggestions", "system_history", "audit",
                 "list_conversations", "get_conversation"):
        fn = getattr(P, name)
        assert not inspect.iscoroutinefunction(fn), f"{name} must be sync (threadpool)"


def test_analytics_cache_roundtrip(monkeypatch):
    monkeypatch.setattr(P, "ANALYTICS_CACHE_TTL_S", 30)
    P._ANALYTICS_CACHE.clear()
    assert P._analytics_cache_get("k") is None
    P._analytics_cache_put("k", {"v": 1})
    assert P._analytics_cache_get("k") == {"v": 1}


def test_analytics_cache_respects_ttl(monkeypatch):
    monkeypatch.setattr(P, "ANALYTICS_CACHE_TTL_S", 0)  # disabled
    P._ANALYTICS_CACHE.clear()
    P._analytics_cache_put("k", 1)
    assert P._analytics_cache_get("k") is None


def test_system_history_downsampled(client):
    conn = P.db()
    now = time.time()
    conn.executemany(
        "INSERT OR REPLACE INTO system_metrics (ts, cpu_pct) VALUES (?, ?)",
        [(now - i, 1.0) for i in range(1000)],
    )
    conn.commit()
    conn.close()
    r = client.get("/__proxy/api/system/history?minutes=60")
    assert r.status_code == 200
    body = r.json()
    assert body["total_samples"] >= 1000       # all rows in the window were counted
    assert len(body["samples"]) <= 800         # but the payload is downsampled
