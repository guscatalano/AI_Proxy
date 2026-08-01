"""The bench frees the GPU before measuring, and puts back what it found.

Quiescing used to mean "gate proxy traffic and evict Ollama". vLLM holds ~99 GB whether or not
anything is asking it, so a run started with ~19 GB free on this box — less than the smallest
model worth benchmarking. It either failed to load or measured a contended GPU.
"""
import asyncio
import json

from ai_proxy import proxy as P


def _stub(monkeypatch, *, vllm_running=True, svc_running=True, ollama=("qwen3:4b",),
          free_mb=20000):
    """Stand in for docker, systemctl, ollama and /proc so nothing here needs a real box."""
    calls = []

    cfg = dict(P.load_rules_config())
    mc = dict(cfg.get("model_control") or {})
    mc["services"] = {"comfyui": {"unit": "comfyui.service", "scope": "user"}}
    cfg["model_control"] = mc
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)

    async def container():
        return "qwen-vllm"

    async def run(args, timeout=120.0, max_chars=800, keep_tail=False, env=None):
        calls.append(list(args))
        if "inspect" in args:
            return 0, "true" if vllm_running else "false"
        if "is-active" in args:
            return 0, "active" if svc_running else "inactive"
        if "is-enabled" in args:
            return 0, "enabled"
        return 0, ""

    monkeypatch.setattr(P, "_vllm_container", container)
    monkeypatch.setattr(P, "_docker_bin", lambda: "/usr/bin/docker")
    monkeypatch.setattr(P, "_run_cmd", run)
    monkeypatch.setattr(P, "_free_mem_mb", lambda: free_mb)

    async def evict(keep=""):
        calls.append(["evict-ollama", keep])
        return [m for m in ollama if m != keep]
    monkeypatch.setattr(P, "_bench_evict_ollama", evict)

    class _Resp:
        def json(self):
            return {"models": [{"name": m} for m in ollama]}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, *a, **kw):
            return _Resp()

        async def post(self, url, json=None, **kw):
            calls.append(["ollama-post", (json or {}).get("model"),
                          (json or {}).get("keep_alive")])
            return _Resp()

    monkeypatch.setattr(P.httpx, "AsyncClient", lambda *a, **kw: _Client())
    return calls


def test_snapshot_discovers_what_is_running(client, monkeypatch):
    _stub(monkeypatch)
    snap = asyncio.run(P._bench_residency_snapshot())
    assert snap["vllm"] == {"container": "qwen-vllm", "was_running": True}
    assert snap["services"] == [{"name": "comfyui", "was_running": True}]
    assert snap["ollama"] == ["qwen3:4b"]


def test_snapshot_records_what_was_already_down(client, monkeypatch):
    # Restoring must not start something the bench found stopped.
    _stub(monkeypatch, vllm_running=False, svc_running=False, ollama=())
    snap = asyncio.run(P._bench_residency_snapshot())
    assert snap["vllm"]["was_running"] is False
    assert snap["services"][0]["was_running"] is False
    assert snap["ollama"] == []


def test_freeing_stops_everything_that_was_up(client, monkeypatch):
    calls = _stub(monkeypatch)
    snap = asyncio.run(P._bench_residency_snapshot())
    calls.clear()
    out = asyncio.run(P._bench_free_gpu(snap, keep="target", want_free_mb=0))
    assert ["systemctl", "--user", "stop", "comfyui.service"] in calls
    assert ["/usr/bin/docker", "stop", "qwen-vllm"] in calls
    assert ["evict-ollama", "target"] in calls
    assert out["stopped_vllm"]["ok"] is True


def test_freeing_leaves_alone_what_was_already_down(client, monkeypatch):
    calls = _stub(monkeypatch, vllm_running=False, svc_running=False)
    snap = asyncio.run(P._bench_residency_snapshot())
    calls.clear()
    asyncio.run(P._bench_free_gpu(snap, want_free_mb=0))
    assert not any("stop" in c for c in calls if c and c[0] != "evict-ollama")


def test_freeing_waits_for_the_memory_to_actually_come_back(client, monkeypatch):
    # docker stop returns when the process is signalled, not when ~99 GB has been unmapped.
    _stub(monkeypatch, free_mb=1000)
    readings = iter([1000, 1000, 90000, 90000, 90000])
    monkeypatch.setattr(P, "_free_mem_mb", lambda: next(readings, 90000))
    snap = asyncio.run(P._bench_residency_snapshot())
    out = asyncio.run(P._bench_free_gpu(snap, want_free_mb=80000, timeout_s=30))
    assert out["reached_target"] is True
    assert out["free_mb_after"] >= 80000


def test_freeing_reports_when_the_memory_never_arrives(client, monkeypatch):
    _stub(monkeypatch, free_mb=1000)
    snap = asyncio.run(P._bench_residency_snapshot())
    out = asyncio.run(P._bench_free_gpu(snap, want_free_mb=80000, timeout_s=4))
    assert out["reached_target"] is False       # said so rather than benchmarking anyway


def test_restore_puts_back_exactly_what_was_found(client, monkeypatch):
    calls = _stub(monkeypatch)
    monkeypatch.setattr(P, "_vllm_ready", lambda t: asyncio.sleep(0, result=True))
    snap = asyncio.run(P._bench_residency_snapshot())
    calls.clear()
    res = asyncio.run(P._bench_restore_residency(snap))
    assert ["/usr/bin/docker", "start", "qwen-vllm"] in calls
    assert ["systemctl", "--user", "start", "comfyui.service"] in calls
    assert ["ollama-post", "qwen3:4b", "30m"] in calls
    assert res["started_vllm"]["ready"] is True


def test_restore_does_not_start_what_was_not_running(client, monkeypatch):
    calls = _stub(monkeypatch, vllm_running=False, svc_running=False, ollama=())
    snap = asyncio.run(P._bench_residency_snapshot())
    calls.clear()
    asyncio.run(P._bench_restore_residency(snap))
    assert calls == []


def test_restore_waits_for_vllm_to_answer(client, monkeypatch):
    # vLLM measured ~9 minutes to reload; reporting "restored" while it still refuses
    # connections would be worse than saying nothing.
    _stub(monkeypatch)
    waited = []

    async def ready(t):
        waited.append(t)
        return False
    monkeypatch.setattr(P, "_vllm_ready", ready)
    snap = asyncio.run(P._bench_residency_snapshot())
    res = asyncio.run(P._bench_restore_residency(snap))
    assert waited, "did not wait for readiness"
    assert res["started_vllm"]["ready"] is False


def test_quiesce_persists_the_snapshot_until_restored(client, monkeypatch):
    # A bench that stops the daily driver and dies would leave it stopped indefinitely.
    _stub(monkeypatch)
    monkeypatch.setattr(P, "_vllm_ready", lambda t: asyncio.sleep(0, result=True))
    state = asyncio.run(P._bench_quiesce(True, keep=""))
    pending = P.get_setting(P._RESIDENCY_SETTING)
    assert pending, "snapshot was not persisted before anything was stopped"
    assert json.loads(pending["value"])["vllm"]["container"] == "qwen-vllm"

    asyncio.run(P._bench_quiesce(False, state))
    assert not P.get_setting(P._RESIDENCY_SETTING), "snapshot outlived the restore"


def test_startup_finishes_an_interrupted_restore(client, monkeypatch):
    calls = _stub(monkeypatch)
    monkeypatch.setattr(P, "_vllm_ready", lambda t: asyncio.sleep(0, result=True))
    snap = asyncio.run(P._bench_residency_snapshot())
    P._save_pending_residency(snap)          # as if the process died mid-bench
    calls.clear()
    res = asyncio.run(P._restore_pending_residency())
    assert res is not None
    assert ["/usr/bin/docker", "start", "qwen-vllm"] in calls
    assert not P.get_setting(P._RESIDENCY_SETTING)


def test_startup_is_a_no_op_with_nothing_pending(client):
    P._save_pending_residency(None)
    assert asyncio.run(P._restore_pending_residency()) is None


def test_a_corrupt_pending_snapshot_is_discarded(client):
    P.set_setting(P._RESIDENCY_SETTING, "not json")
    assert asyncio.run(P._restore_pending_residency()) is None
    assert not P.get_setting(P._RESIDENCY_SETTING)
