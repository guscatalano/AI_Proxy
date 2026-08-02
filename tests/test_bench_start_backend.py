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
    monkeypatch.setattr(P, "_vllm_configs", _configs(running=True))
    monkeypatch.setattr(P, "system_now", lambda: {})
    assert "vllm:qwen3-coder" not in asyncio.run(P._bench_model_index())


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

    async def free(s, keep="", want_free_mb=0, timeout_s=240.0):
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
    assert seen == [{"container": "qwen-vllm"}], "the wrong container would serve wrong numbers"
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

    async def free(s, keep="", want_free_mb=0, timeout_s=240.0):
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
