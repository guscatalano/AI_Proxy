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


def test_starting_passes_the_container_through(client, monkeypatch):
    seen = []

    async def load(payload, name):
        seen.append(dict(payload))
        return {"ok": True, "started_container": payload.get("container"), "ready": True}

    monkeypatch.setattr(P.PROVIDERS["vllm"], "load", load)
    res = asyncio.run(P._bench_start_backend(
        {"model": "qwen3-coder", "upstream": "vllm", "container": "qwen-vllm"}))
    assert res["ok"] and res["started"]
    assert seen == [{"container": "qwen-vllm"}], "the wrong container would serve wrong numbers"
    assert res["restore"] == {"upstream": "vllm"}


def test_a_failed_start_hands_back_no_restore(client, monkeypatch):
    async def load(payload, name):
        return P.JSONResponse({"error": "no such container"}, status_code=404)

    monkeypatch.setattr(P.PROVIDERS["vllm"], "load", load)
    res = asyncio.run(P._bench_start_backend({"upstream": "vllm", "container": "gone"}))
    assert res["ok"] is False and res["restore"] is None
    assert "no such container" in res["detail"]


def test_what_the_bench_started_is_stopped_again(client, monkeypatch):
    """vLLM holds ~99 GB whether or not anything is asking it, so leaving it up after a run
    quietly takes the box away from whatever comes next."""
    stopped = []

    async def stop():
        stopped.append(1)
        return {"ok": True, "detail": "", "via": "docker"}

    monkeypatch.setattr(P.PROVIDERS["vllm"], "stop", stop)
    assert asyncio.run(P._bench_restore_backend({"upstream": "vllm"}))["ok"] is True
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
