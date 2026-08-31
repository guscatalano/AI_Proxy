"""Connection-pool hygiene.

On 2026-08-08 the proxy stopped serving anything for 25 minutes. Nothing looked wrong:
systemd said active, Ollama answered in 0.8ms, vLLM in 11ms, and /__proxy/api/health in
3ms — because the management routes don't share the upstream client. 101 leaked sockets to
vLLM had filled httpx's default 100-connection pool, and with no pool timeout every proxied
request queued for a slot forever. These tests cover the three things that let it happen:
the pool was unbounded in wait and small in size, finalizing a row threw away the only
reference to an unclosed socket, and nothing anywhere reported occupancy.
"""
import time

import httpx
import pytest

import ai_proxy
from ai_proxy import proxy


def test_upstream_client_bounds_its_pool_wait(client):
    """timeout=Timeout(None) is what made exhaustion silent and infinite."""
    c = ai_proxy.app.state.client
    assert c.timeout.pool is not None, (
        "pool wait must be bounded — an unbounded one turns exhaustion into an "
        "indefinite hang with no error, no log, and a passing health check"
    )
    assert c.timeout.read is None, (
        "read must stay unbounded: a large prefill legitimately takes minutes"
    )


def test_upstream_client_raises_the_connection_cap(client):
    pool = ai_proxy.app.state.client._transport._pool
    assert pool._max_connections > 100, "default cap of 100 is what we hit"


def test_pool_stats_reports_occupancy(client):
    s = proxy._pool_stats()
    assert s, "pool stats must be readable — this is the only signal for this failure"
    assert s["limit"] > 100
    assert s["active"] == s["connections"] - s["idle"]


def test_pool_stats_survives_httpx_internals_moving(monkeypatch):
    """It reaches into private httpcore attributes; losing the metric is acceptable,
    raising from a health check is not."""
    class Exploding:
        @property
        def _transport(self):
            raise AttributeError("httpx moved the furniture")

    monkeypatch.setattr(ai_proxy.app.state, "client", Exploding(), raising=False)
    assert proxy._pool_stats() == {}
    proxy._pool_watch()          # must not raise either


def test_health_exposes_pool_occupancy(client):
    r = client.get("/__proxy/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "upstream_pool" in body
    assert body["upstream_pool"]["limit"] > 100
    assert "inflight" in body


class _FakeResp:
    def __init__(self, closed):
        self.is_closed = closed
        self.aclosed = False

    async def aclose(self):
        self.aclosed = True
        self.is_closed = True


def test_finalizing_a_row_does_not_strand_an_open_socket(client):
    """_save_finish pops the registry — the last reference to the response. If the socket
    is still open at that moment it becomes unreclaimable AND invisible, which is how 101
    of them accumulated without anything noticing."""
    proxy._ORPHANED_RESPONSES.clear()
    resp = _FakeResp(closed=False)
    proxy._INFLIGHT_REQUESTS["pool-test-open"] = {
        "ts": time.time(), "upstream_resp": resp, "upstream": "vllm", "cancelled": False}

    proxy._save_finish("pool-test-open", 200, {}, "{}", None, 1.0, None)

    assert "pool-test-open" not in proxy._INFLIGHT_REQUESTS
    assert resp in proxy._ORPHANED_RESPONSES, "open socket must be handed off for closing"


def test_already_closed_socket_is_not_queued(client):
    proxy._ORPHANED_RESPONSES.clear()
    resp = _FakeResp(closed=True)
    proxy._INFLIGHT_REQUESTS["pool-test-closed"] = {
        "ts": time.time(), "upstream_resp": resp, "upstream": "vllm", "cancelled": False}

    proxy._save_finish("pool-test-closed", 200, {}, "{}", None, 1.0, None)

    assert proxy._ORPHANED_RESPONSES == [], "the happy path must not grow the queue"


@pytest.mark.anyio
async def test_reaper_drains_orphaned_sockets():
    """The queue is only useful if something empties it."""
    resp = _FakeResp(closed=False)
    proxy._ORPHANED_RESPONSES.clear()
    proxy._ORPHANED_RESPONSES.append(resp)

    while proxy._ORPHANED_RESPONSES:          # the drain loop, as the killer runs it
        await proxy._ORPHANED_RESPONSES.pop().aclose()

    assert resp.aclosed


def test_pool_watch_warns_near_the_cap(monkeypatch, capsys):
    monkeypatch.setattr(proxy, "_pool_stats",
                        lambda: {"connections": 210, "idle": 0, "active": 210, "limit": 256})
    proxy._POOL_WARNED["at"] = 0.0
    proxy._pool_watch()
    assert "210/256" in capsys.readouterr().out


def test_pool_watch_is_quiet_below_the_threshold(monkeypatch, capsys):
    monkeypatch.setattr(proxy, "_pool_stats",
                        lambda: {"connections": 10, "idle": 8, "active": 2, "limit": 256})
    proxy._POOL_WARNED["at"] = 0.0
    proxy._pool_watch()
    assert capsys.readouterr().out == ""


def test_pool_watch_does_not_flood_the_journal(monkeypatch, capsys):
    monkeypatch.setattr(proxy, "_pool_stats",
                        lambda: {"connections": 256, "idle": 0, "active": 256, "limit": 256})
    proxy._POOL_WARNED["at"] = 0.0
    proxy._pool_watch()
    capsys.readouterr()
    proxy._pool_watch()          # a full pool stays full; the second tick must stay silent
    assert capsys.readouterr().out == ""
