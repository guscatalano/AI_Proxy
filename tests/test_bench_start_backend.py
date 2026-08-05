"""Benchmarking a model whose container is configured but stopped.

vLLM serves exactly what its process was launched with, so a stopped container looked identical
to a model that does not exist: the bench refused it, and the only way through was to open the
System tab, start the right container by hand, come back, and hope. The proxy already knew both
what the container would serve and how to start it.
"""
import asyncio
import json
import time

from ai_proxy import proxy as P


def _configs(running=False):
    async def fn():
        return [{"container": "qwen-vllm", "image": "vllm/vllm-openai", "running": running,
                 "serves_port": True, "model": "qwen3-coder", "checkpoint": "/models/qwen3",
                 "quant": "fp8", "max_model_len": "262144", "mounts": []},
                # Publishes a different port: starting it would not make it reachable here.
                {"container": "stray-vllm", "image": "vllm/vllm-openai", "running": False,
                 "serves_port": False, "model": "other", "max_model_len": "8192", "mounts": []}]
    return fn


def test_a_stopped_container_is_offered_as_a_model(client, monkeypatch):
    monkeypatch.setattr(P, "_vllm_configs", _configs())
    monkeypatch.setattr(P, "system_now", lambda: {})
    idx = asyncio.run(P._bench_model_index())
    rec = idx.get("vllm:qwen3-coder")
    assert rec, f"stopped container missing from the picker: {sorted(idx)}"
    assert rec["loaded"] is False and rec["startable"] is True
    assert rec["container"] == "qwen-vllm"
    assert rec["max_context"] == 262144


def test_a_container_on_the_wrong_port_is_not_offered(client, monkeypatch):
    """Starting it would succeed and still leave nothing reachable through this proxy."""
    monkeypatch.setattr(P, "_vllm_configs", _configs())
    monkeypatch.setattr(P, "system_now", lambda: {})
    assert "vllm:other" not in asyncio.run(P._bench_model_index())


def test_a_running_container_is_not_duplicated_as_startable(client, monkeypatch):
    """'Running' excludes the startable entry only when the live probe actually answers for
    its model. A running-but-still-booting container must stay visible (qwen-vllm spent
    minutes loading shards and its cells died at preflight); a genuinely serving one must
    not appear twice."""
    monkeypatch.setattr(P, "_vllm_configs", _configs(running=True))
    monkeypatch.setattr(P, "system_now", lambda: {
        "vllm": {"reachable": True,
                 "available": [{"id": "qwen3-coder", "state": "loaded"}]}})
    idx = asyncio.run(P._bench_model_index())
    rec = idx.get("vllm:qwen3-coder")
    assert rec and rec["loaded"] is True and not rec.get("startable")
    # And while booting (live probe silent), the same running container IS offered.
    monkeypatch.setattr(P, "system_now", lambda: {})
    booting = asyncio.run(P._bench_model_index()).get("vllm:qwen3-coder")
    assert booting and booting["startable"] is True and booting["loaded"] is False


def test_preflight_allows_a_startable_model(client):
    meta = {"model": "qwen3-coder", "upstream": "vllm", "loaded": False, "startable": True}
    idx = {"vllm:qwen3-coder": meta}
    assert P._bench_preflight("qwen3-coder", meta, "vllm", idx) is None


def test_preflight_still_refuses_a_model_that_is_merely_absent(client):
    idx = {"vllm:other": {"model": "other", "upstream": "vllm", "loaded": True}}
    assert P._bench_preflight("qwen3-coder", {}, "vllm", idx) is not None


def _quiet_residency(monkeypatch):
    """Stub the free/restore handshake so these tests are about starting, not about docker."""
    async def snap():
        return {"backends": [], "ollama": []}

    async def free(s, keep="", want_free_mb=0, timeout_s=240.0, **kw):
        return {"stopped": [], "evicted_ollama": []}

    async def restore(s):
        return {"started": []}

    monkeypatch.setattr(P, "_bench_residency_snapshot", snap)
    monkeypatch.setattr(P, "_bench_free_gpu", free)
    monkeypatch.setattr(P, "_bench_restore_residency", restore)
    monkeypatch.setattr(P, "_save_pending_residency", lambda s: None)


def test_starting_passes_the_container_through(client, monkeypatch):
    seen = []
    _quiet_residency(monkeypatch)

    async def load(payload, name):
        seen.append(dict(payload))
        return {"ok": True, "started_container": payload.get("container"), "ready": True}

    monkeypatch.setattr(P.PROVIDERS["vllm"], "load", load)
    res = asyncio.run(P._bench_start_backend(
        {"model": "qwen3-coder", "upstream": "vllm", "container": "qwen-vllm"}))
    assert res["ok"] and res["started"]
    assert seen[0]["container"] == "qwen-vllm", "the wrong container would serve wrong numbers"
    assert res["restore"]["upstream"] == "vllm" and res["restore"]["stop"] is True


def test_a_start_that_never_became_ready_is_still_stopped(client, monkeypatch):
    """The bug that left a container crash-looping for thirteen minutes. vLLM's load starts the
    container and *then* waits for readiness, so a timeout leaves it running -- and under
    restart=unless-stopped it kept coming back to fight for the memory it had just failed to
    get. "Did it work" and "was the box changed" are different questions."""
    _quiet_residency(monkeypatch)

    async def load(payload, name):
        return {"ok": False, "started_container": "qwen-vllm", "ready": False,
                "error": "container started but the server did not become ready in time"}

    monkeypatch.setattr(P.PROVIDERS["vllm"], "load", load)
    res = asyncio.run(P._bench_start_backend({"upstream": "vllm", "container": "qwen-vllm"}))
    assert res["ok"] is False
    assert res["started"] is True
    assert res["restore"]["stop"] is True, "nothing would ever stop the container"


def test_room_is_made_before_the_backend_starts(client, monkeypatch):
    """vLLM wants ~99 GB on a 121 GB box, so llama.cpp holding 90 GB means the start can only
    ever time out -- and the symptom is indistinguishable from a slow load."""
    order = []

    async def snap():
        order.append("snapshot")
        return {"backends": [{"name": "llamacpp", "was_running": True, "control": "unit"}]}

    async def free(s, keep="", want_free_mb=0, timeout_s=240.0, **kw):
        order.append("free")
        return {"stopped": [{"name": "llamacpp"}]}

    async def load(payload, name):
        order.append("start")
        return {"ok": True, "started_container": "qwen-vllm", "ready": True}

    monkeypatch.setattr(P, "_bench_residency_snapshot", snap)
    monkeypatch.setattr(P, "_bench_free_gpu", free)
    monkeypatch.setattr(P, "_save_pending_residency", lambda s: None)
    monkeypatch.setattr(P.PROVIDERS["vllm"], "load", load)

    res = asyncio.run(P._bench_start_backend({"upstream": "vllm", "container": "qwen-vllm"}))
    assert order == ["snapshot", "free", "start"], order
    assert res["restore"]["residency"]["backends"][0]["name"] == "llamacpp"


def test_restore_stops_the_new_backend_before_reviving_the_old(client, monkeypatch):
    """llama.cpp cannot reload 90 GB of weights until vLLM gives its memory back."""
    order = []

    async def stop():
        order.append("stop-vllm")
        return {"ok": True}

    async def restore(s):
        order.append("restore-residency")
        return {"started": ["llamacpp"]}

    monkeypatch.setattr(P.PROVIDERS["vllm"], "stop", stop)
    monkeypatch.setattr(P, "_bench_restore_residency", restore)
    monkeypatch.setattr(P, "_save_pending_residency", lambda s: None)
    asyncio.run(P._bench_restore_backend(
        {"upstream": "vllm", "stop": True, "residency": {"backends": []}}))
    assert order == ["stop-vllm", "restore-residency"], order


def test_what_the_bench_started_is_stopped_again(client, monkeypatch):
    """vLLM holds ~99 GB whether or not anything is asking it, so leaving it up after a run
    quietly takes the box away from whatever comes next."""
    stopped = []

    async def stop():
        stopped.append(1)
        return {"ok": True, "detail": "", "via": "docker"}

    monkeypatch.setattr(P.PROVIDERS["vllm"], "stop", stop)
    res = asyncio.run(P._bench_restore_backend({"upstream": "vllm", "stop": True}))
    assert res["stopped"]["ok"] is True
    assert stopped == [1]


def test_a_backend_that_was_already_up_is_left_alone(client):
    # It is the user's, not the bench's. Stopping it would take away something in use.
    assert asyncio.run(P._bench_restore_backend(None)) is None


# ---- eviction is no longer a choice -------------------------------------------------------

def test_every_run_evicts_whether_or_not_it_was_asked(client):
    """It stopped being a checkbox because it is not a preference: a run that leaves other
    models resident measures whatever the box happened to be holding, not the model."""
    import inspect
    src = inspect.getsource(P._bench_execute)
    assert 'cfg.get("evict_others")' not in src, "eviction is still conditional on a flag"
    assert "_bench_evict_ollama(keep=model)" in src


def test_what_was_evicted_is_reloaded(client, monkeypatch):
    """Opt-in, losing them was the point. Unconditional, a three-minute bench would leave the
    daily driver cold with nothing saying why."""
    import inspect
    src = inspect.getsource(P._bench_execute)
    assert "_bench_reload_ollama(evicted)" in src

    reloaded = []

    class _R:
        status_code = 200

    class _C:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, **kw):
            reloaded.append(json["model"])
            assert json["keep_alive"] != 0, "reload must not unload it again"
            return _R()

    monkeypatch.setattr(P.httpx, "AsyncClient", lambda *a, **kw: _C())
    assert asyncio.run(P._bench_reload_ollama(["qwen3:4b", "llama3"])) == ["qwen3:4b", "llama3"]
    assert reloaded == ["qwen3:4b", "llama3"]


def test_an_eviction_failure_is_not_swallowed(client, monkeypatch, capsys):
    """An empty list means "the GPU is clear" to the caller, so a failure here makes the bench
    measure a contended box believing it is quiet."""
    class _C:
        async def __aenter__(self):
            raise RuntimeError("ollama unreachable")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(P.httpx, "AsyncClient", lambda *a, **kw: _C())
    assert asyncio.run(P._bench_evict_ollama()) == []
    assert "eviction failed" in capsys.readouterr().out


# ---- mixed-backend sweeps -----------------------------------------------------------------

def test_a_stopped_llamacpp_still_offers_its_model(client, monkeypatch):
    """Once the bench stopped llama.cpp to make room for vLLM, llama.cpp's own cells could not
    resolve a model and failed preflight with "not serving anything". The unit is configured
    with exactly one model; a stopped one has to be offered the way a stopped container is."""
    cfg = dict(P.load_rules_config())
    mc = dict(cfg.get("model_control") or {})
    mc["llamacpp"] = {"unit": "llamacpp.service", "binary": "/x/llama-server",
                      "model": "/models/ds4-flash-UD-IQ2_XXS.gguf"}
    cfg["model_control"] = mc
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    monkeypatch.setattr(P, "_vllm_configs", _configs())
    monkeypatch.setattr(P, "system_now", lambda: {"llamacpp": {"reachable": False}})

    idx = asyncio.run(P._bench_model_index())
    rec = idx.get("llamacpp:/models/ds4-flash-UD-IQ2_XXS.gguf")
    assert rec, f"stopped llama.cpp offers nothing: {sorted(idx)}"
    assert rec["startable"] is True and rec["loaded"] is False


def test_cells_are_grouped_by_backend(client):
    """Two backends rarely fit on one box, so each switch means stopping one and loading the
    other. Interleaved, a mixed sweep pays that per cell -- and one ran llama.cpp's cell while
    it was stopped for vLLM's."""
    models = [{"model": "a", "upstream": "vllm"}, {"model": "b", "upstream": "llamacpp"}]
    cells = P._bench_expand_matrix(models, {"cache": ["cold", "cached"]})
    ups = [c["upstream"] for c in cells]
    assert ups == sorted(ups), f"backends interleaved: {ups}"
    # ...and the cache axis still runs in order inside each backend's group.
    for u in ("llamacpp", "vllm"):
        assert [c["cache"] for c in cells if c["upstream"] == u] == ["cold", "cached"]


def test_a_cell_inside_a_sweep_does_not_overwrite_the_sweeps_snapshot(client, monkeypatch):
    """The sweep records the box before its first cell. A cell's own snapshot is of the box
    mid-run, so persisting it would restore whatever the previous cell happened to leave."""
    saved = []
    _quiet_residency(monkeypatch)
    monkeypatch.setattr(P, "_save_pending_residency", lambda s: saved.append(s))

    async def load(payload, name):
        return {"ok": True, "started_container": "qwen-vllm", "ready": True}

    monkeypatch.setattr(P.PROVIDERS["vllm"], "load", load)
    res = asyncio.run(P._bench_start_backend({"upstream": "vllm", "container": "qwen-vllm"},
                                             persist=False))
    assert saved == [], "a cell overwrote the sweep's record of the original state"
    assert res["restore"]["residency"] is None
    assert res["restore"]["stop"] is True, "it still has to stop what it started"


# ---- waiting long enough, safely -----------------------------------------------------------

def test_a_bench_start_waits_far_longer_than_the_default(client, monkeypatch):
    """qwen3-coder-next was stopped mid-load at the 420s default and recorded as a failure,
    while ornith-nvfp4 -- a 4-bit checkpoint -- loaded inside it. A bench has hours to spend."""
    seen = []
    _quiet_residency(monkeypatch)

    async def load(payload, name):
        seen.append(dict(payload))
        return {"ok": True, "started_container": "qwen-vllm", "ready": True}

    monkeypatch.setattr(P.PROVIDERS["vllm"], "load", load)
    asyncio.run(P._bench_start_backend({"upstream": "vllm", "container": "qwen-vllm"}))
    assert seen[0]["wait_s"] == P._BENCH_START_READY_S
    assert P._BENCH_START_READY_S >= 1800, "still too tight for a large checkpoint"


def test_a_dead_container_is_noticed_without_waiting_it_out(client, monkeypatch):
    """What makes the long wait safe. Without this, raising the ceiling would just mean a
    crashed container burns thirty minutes instead of seven."""
    async def container():
        return "qwen-vllm"

    async def run(args, timeout=120.0, max_chars=800, keep_tail=False, env=None):
        if "inspect" in args:
            return 0, "false"
        if "logs" in args:
            return 0, "CUDA out of memory"
        return 0, ""

    monkeypatch.setattr(P, "_vllm_container", container)
    monkeypatch.setattr(P, "_docker_bin", lambda: "/usr/bin/docker")
    monkeypatch.setattr(P, "_run_cmd", run)
    why = asyncio.run(P.PROVIDERS["vllm"].died())
    assert why and "not running" in why
    assert "out of memory" in why, "the reason has to come with it"


def test_a_running_container_is_not_reported_as_dead(client, monkeypatch):
    async def container():
        return "qwen-vllm"

    async def run(args, timeout=120.0, max_chars=800, keep_tail=False, env=None):
        return (0, "true") if "inspect" in args else (0, "")

    monkeypatch.setattr(P, "_vllm_container", container)
    monkeypatch.setattr(P, "_docker_bin", lambda: "/usr/bin/docker")
    monkeypatch.setattr(P, "_run_cmd", run)
    assert asyncio.run(P.PROVIDERS["vllm"].died()) is None


def test_the_environment_snapshot_does_not_await_a_sync_function(client):
    """system_now is sync so Starlette runs it in the threadpool. Awaiting it raised TypeError
    into a blanket except, so every bench run recorded env['error'] instead of the GPU, memory
    and engine state — which is why reports said "GPU: not reported" on a machine with a
    perfectly detectable GB10. Same mistake _bench_model_index already had."""
    import inspect
    src = inspect.getsource(P._bench_env_snapshot)
    assert "await system_now()" not in src
    assert "to_thread(system_now)" in src


def test_the_environment_snapshot_captures_the_gpu(client, monkeypatch):
    monkeypatch.setattr(P, "system_now", lambda: {
        "gpus": [{"name": "NVIDIA GB10", "mem_total_mb": 124610, "mem_used_mb": 170,
                  "util_pct": 0}],
        "mem": {"total_mb": 124610}})
    env = asyncio.run(P._bench_env_snapshot())
    assert "error" not in env, env.get("error")
    assert env["gpus"][0]["name"] == "NVIDIA GB10"
    assert env["gpus"][0]["mem_total_mb"] == 124610
