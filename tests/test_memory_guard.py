"""Memory guards on every path that can start a vLLM container.

The incident these pin: qwen-vllm at --gpu-memory-utilization 0.80 booted beside a
resident 22 GB gemma4. The auto-loader priced the boot at its checkpoint size — usually
unknown for containers, so want=0 and the free-memory check never ran — while vLLM
actually pre-allocates utilization × total (~97 GB of 121). The box hit the memory wall:
sshd and the proxy starved (anything needing new allocations hung) while the resident
backends kept answering. A start the box cannot absorb must be REFUSED, because once it
happens nobody can reach the machine to undo it.
"""
import asyncio
import json

from ai_proxy import proxy as P


def _claim_env(monkeypatch, *, util="0.80", total_mb=124610, free_mb=20000.0):
    async def cfgs(*a, **k):
        return [{"container": "qwen-vllm", "serves_port": True, "running": False,
                 "model": "qwen3-coder-next",
                 "gpu_memory_utilization": util}]

    monkeypatch.setattr(P, "_vllm_configs", cfgs)
    monkeypatch.setattr(P, "_mem_snapshot", lambda: {"total_mb": total_mb,
                                                     "avail_mb": free_mb})
    monkeypatch.setattr(P, "_free_mem_mb", lambda: free_mb)


def test_boot_claim_is_utilization_times_total(monkeypatch):
    _claim_env(monkeypatch, util="0.80")
    claim = asyncio.run(P._vllm_boot_claim_mb("qwen-vllm"))
    # util × total, not weights — plus the driver reserve, because utilization turned out
    # not to bound what vLLM actually takes.
    assert round(claim) == round(124610 * 0.80 + P._vllm_boot_reserve_mb(124610))


def test_boot_claim_defaults_to_vllms_own_090(monkeypatch):
    _claim_env(monkeypatch, util=None)
    claim = asyncio.run(P._vllm_boot_claim_mb("qwen-vllm"))
    assert round(claim) == round(124610 * 0.90 + P._vllm_boot_reserve_mb(124610))


def test_load_refuses_a_start_the_box_cannot_absorb(monkeypatch):
    _claim_env(monkeypatch, free_mb=20000.0)     # 20 GB free vs ~97 GB claim

    async def must_not_start(cmd, timeout, **kw):
        raise AssertionError(f"docker must not run: {cmd}")

    monkeypatch.setattr(P, "_run_cmd", must_not_start)
    prov = P.PROVIDERS["vllm"]
    res = asyncio.run(prov.load({"container": "qwen-vllm"}, "qwen3-coder-next"))
    assert res.status_code == 409
    body = json.loads(res.body)
    assert "force:true" in body["error"]
    assert body["claim_mb"] > body["free_mb"]


def test_force_overrides_the_load_guard(client, monkeypatch):
    _claim_env(monkeypatch, free_mb=20000.0)
    ran = []

    async def fake_run(cmd, timeout, **kw):
        ran.append(cmd)
        return 0, ""

    async def ready(*a, **k):
        return True

    monkeypatch.setattr(P, "_run_cmd", fake_run)
    monkeypatch.setattr(P, "_vllm_ready", ready)
    monkeypatch.setattr(P, "_docker_bin", lambda: "docker")
    prov = P.PROVIDERS["vllm"]
    res = asyncio.run(prov.load({"container": "qwen-vllm", "force": True}, "m"))
    assert isinstance(res, dict) and res["ok"] is True
    assert any("start" in c for c in ran)


def test_load_passes_when_the_box_has_room(client, monkeypatch):
    _claim_env(monkeypatch, util="0.62", free_mb=110000.0)   # ~77 GB claim, 110 free
    ran = []

    async def fake_run(cmd, timeout, **kw):
        ran.append(cmd)
        return 0, ""

    async def ready(*a, **k):
        return True

    monkeypatch.setattr(P, "_run_cmd", fake_run)
    monkeypatch.setattr(P, "_vllm_ready", ready)
    monkeypatch.setattr(P, "_docker_bin", lambda: "docker")
    prov = P.PROVIDERS["vllm"]
    res = asyncio.run(prov.load({"container": "qwen-vllm"}, "m"))
    assert isinstance(res, dict) and res["ok"] is True


def test_control_backend_docker_start_is_guarded(monkeypatch):
    _claim_env(monkeypatch, free_mb=20000.0)

    async def must_not_start(cmd, timeout, **kw):
        raise AssertionError(f"docker must not run: {cmd}")

    monkeypatch.setattr(P, "_run_cmd", must_not_start)
    monkeypatch.setattr(P, "_docker_bin", lambda: "docker")
    b = P.backend("vllm")
    res = asyncio.run(P._control_backend(b, "start", container="qwen-vllm"))
    assert res["ok"] is False and res["via"] == "memory-guard"
    assert "free memory first" in res["detail"]
    # stop is never guarded — stopping is how memory comes back
    stopped = []

    async def fake_stop(cmd, timeout, **kw):
        stopped.append(cmd)
        return 0, ""

    monkeypatch.setattr(P, "_run_cmd", fake_stop)
    res = asyncio.run(P._control_backend(b, "stop", container="qwen-vllm"))
    assert res["ok"] is True and stopped


def test_auto_load_refuses_when_eviction_cannot_make_room(client, monkeypatch):
    """End of the auto-load path: eviction ran, the memory still is not there — the
    request is refused with the arithmetic rather than the box being started into a
    wedge. (The 503 carries the numbers so the client log says WHY.)"""
    monkeypatch.setattr(P, "_auto_load_cfg",
                        lambda: {"enabled": True, "min_hold_s": 0, "drain_s": 0,
                                 "ready_timeout_s": 5})
    meta = {"startable": True, "container": "qwen-vllm", "loaded": False, "size_mb": None}

    async def index():
        return {}

    monkeypatch.setattr(P, "_bench_model_index", index)
    monkeypatch.setattr(P, "_bench_resolve_model", lambda i, m, u: dict(meta))

    async def claim(c):
        return 97000.0

    async def running(*a, **k):
        return None

    async def snap():
        return {"backends": []}

    async def free_gpu(*a, **k):
        return {"stopped": [], "evicted_ollama": []}

    monkeypatch.setattr(P, "_vllm_boot_claim_mb", claim)
    monkeypatch.setattr(P, "_vllm_container", running)
    monkeypatch.setattr(P, "_bench_residency_snapshot", snap)
    monkeypatch.setattr(P, "_bench_free_gpu", free_gpu)
    monkeypatch.setattr(P, "_free_mem_mb", lambda: 20000.0)
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.commit()
    conn.close()
    P._AUTO_LOAD_LAST.update(ts=0.0, target=None)
    res = asyncio.run(P._ensure_model_served("qwen3-coder-next", "vllm"))
    assert res is not None and res["status"] == 503
    assert "Refusing" in res["error"] and "95 GB" in res["error"]
