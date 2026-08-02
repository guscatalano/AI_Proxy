"""The metrics collector must not do blocking work on the event loop.

Symptom: the dashboard stops answering for ten seconds at a time while a benchmark runs, then
recovers on its own — so it reads as load rather than a bug. The proxy was still listening;
it just was not *accepting*, and connections queued in the socket backlog.

Cause: _collect_once is `async def` and called _gpu_snapshot() directly. That shells out to
nvidia-smi three times with a 3s timeout each, and nvidia-smi is slowest exactly when the GPU
is busy — so it stalled hardest when someone was most likely to be watching.
"""
import asyncio
import inspect
import time

from ai_proxy import proxy as P


def test_the_collector_offloads_every_blocking_call(client):
    src = inspect.getsource(P._collect_once)
    for direct in ("_gpu_snapshot()", "_cpu_pct()", "_mem_snapshot()", "db()"):
        assert direct not in src, f"{direct} still runs on the event loop"
    assert "to_thread(_host_snapshot)" in src
    assert "to_thread(_write_metrics_sample" in src


def test_a_slow_nvidia_smi_does_not_stall_the_loop(client, monkeypatch):
    """The real failure, reproduced: nvidia-smi wedged for longer than a request can wait.
    The loop has to stay free to serve while that happens."""
    async def probe(c):
        return {"reachable": True}

    for b in list(P.PROVIDERS.values()) + list(P.SIDE_SERVICES.values()):
        monkeypatch.setattr(b, "probe", probe)
    monkeypatch.setattr(P, "_gpu_snapshot", lambda: time.sleep(0.6) or [])
    monkeypatch.setattr(P, "_artifact_sweep", lambda: None)

    async def drive():
        gaps = []

        async def heartbeat():
            last = time.perf_counter()
            for _ in range(80):
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        hb = asyncio.create_task(heartbeat())
        # Let the heartbeat actually start: measuring from before its first tick would miss an
        # inline call made before the first await, which is where this one sits.
        await asyncio.sleep(0.05)
        await P._collect_once(P.app)
        await hb
        return max(gaps)

    # Inline, the 0.6s snapshot shows up as one 0.6s gap between heartbeats.
    assert asyncio.run(drive()) < 0.25, "the event loop stalled while the GPU was polled"


def test_the_sample_still_lands(client, monkeypatch):
    async def probe(c):
        return {"reachable": True, "marker": "yes"}

    for b in list(P.PROVIDERS.values()) + list(P.SIDE_SERVICES.values()):
        monkeypatch.setattr(b, "probe", probe)
    monkeypatch.setattr(P, "_gpu_snapshot", lambda: [{"idx": 0, "name": "test"}])
    monkeypatch.setattr(P, "_artifact_sweep", lambda: None)

    conn = P.db()
    conn.execute("DELETE FROM system_metrics")
    conn.commit()
    conn.close()

    asyncio.run(P._collect_once(P.app))

    conn = P.db()
    row = conn.execute(
        "SELECT gpu_json, backends_json, cpu_pct FROM system_metrics ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None, "no sample was written"
    assert "test" in (row["gpu_json"] or "")
    assert "marker" in (row["backends_json"] or "")
