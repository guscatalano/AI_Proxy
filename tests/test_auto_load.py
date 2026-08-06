"""Auto-load: Ollama-style on-demand loading for the fixed backends, strictly opt-in.

The contract, in order: disabled means untouched; a loaded model means untouched; a running
bench owns the box; the vLLM twins swap by stopping the port-holder first; other models are
evicted ONLY when the target does not fit in free memory; and a fresh switch holds for
min_hold_s so two clients cannot ping-pong the port.
"""
import asyncio
import json
import time

from ai_proxy import proxy as P


class _FakeProv:
    def __init__(self):
        self.stopped = False
        self.loaded_with = None

    async def stop(self):
        self.stopped = True
        return {"ok": True}

    async def load(self, payload, name):
        self.loaded_with = payload
        return {"ok": True, "ready": True}


def _setup(monkeypatch, *, enabled=True, loaded=False, startable=True, size_mb=40000,
           free_mb=100000, bench_busy=False, running_container="ornith-vllm"):
    prov = _FakeProv()
    freed = {"called": False}

    monkeypatch.setattr(P, "load_rules_config", lambda: {
        "model_control": {"auto_load": {"enabled": enabled, "ready_timeout_s": 5,
                                        "min_hold_s": 120}}})

    async def fake_index():
        meta = {"model": "qwen3-coder-next", "upstream": "vllm", "loaded": loaded,
                "size_mb": size_mb}
        if startable:
            meta.update(startable=True, container="qwen-vllm")
        return {"vllm:qwen3-coder-next": meta}

    async def fake_container():
        return running_container

    async def fake_snap():
        return {"backends": [], "ollama": []}

    async def fake_free(snap, keep="", spare="", want_free_mb=0, bench_id=""):
        freed["called"] = True
        return {"reached_target": True}

    monkeypatch.setattr(P, "_bench_model_index", fake_index)
    monkeypatch.setattr(P, "_vllm_container", fake_container)
    monkeypatch.setattr(P, "_bench_residency_snapshot", fake_snap)
    monkeypatch.setattr(P, "_bench_free_gpu", fake_free)
    monkeypatch.setattr(P, "_free_mem_mb", lambda: free_mb)
    monkeypatch.setattr(P, "PROVIDERS", {"vllm": prov})
    monkeypatch.setattr(P, "_AUTO_LOAD_LAST", {"ts": 0.0, "target": ""})

    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    if bench_busy:
        conn.execute("INSERT INTO bench_runs (id, ts, model, config_json, status) "
                     "VALUES ('b_x', ?, 'm', '{}', 'running')", (time.time(),))
    conn.commit()
    conn.close()
    return prov, freed


def test_disabled_is_a_no_op(client, monkeypatch):
    prov, _ = _setup(monkeypatch, enabled=False)
    assert asyncio.run(P._ensure_model_served("qwen3-coder-next", "vllm")) is None
    assert not prov.stopped and prov.loaded_with is None


def test_a_loaded_model_is_untouched(client, monkeypatch):
    prov, _ = _setup(monkeypatch, loaded=True)
    assert asyncio.run(P._ensure_model_served("qwen3-coder-next", "vllm")) is None
    assert not prov.stopped


def test_swaps_the_twin_and_waits(client, monkeypatch):
    prov, freed = _setup(monkeypatch)
    assert asyncio.run(P._ensure_model_served("qwen3-coder-next", "vllm")) is None
    assert prov.stopped, "the port-holding twin must stop first"
    assert prov.loaded_with["container"] == "qwen-vllm"
    assert not freed["called"], "fits in free memory — nothing else may be evicted"
    assert P._AUTO_LOAD_LAST["target"] == "vllm:qwen3-coder-next"


def test_evicts_only_when_it_does_not_fit(client, monkeypatch):
    prov, freed = _setup(monkeypatch, size_mb=90000, free_mb=20000)
    assert asyncio.run(P._ensure_model_served("qwen3-coder-next", "vllm")) is None
    assert freed["called"], "no room means the bench-grade free path runs"


def test_a_running_bench_owns_the_box(client, monkeypatch):
    prov, _ = _setup(monkeypatch, bench_busy=True)
    err = asyncio.run(P._ensure_model_served("qwen3-coder-next", "vllm"))
    assert err and err["status"] == 503 and "benchmark" in err["error"]
    assert not prov.stopped


def test_min_hold_prevents_ping_pong(client, monkeypatch):
    prov, _ = _setup(monkeypatch)
    monkeypatch.setattr(P, "_AUTO_LOAD_LAST",
                        {"ts": time.time(), "target": "vllm:ornith-nvfp4"})
    err = asyncio.run(P._ensure_model_served("qwen3-coder-next", "vllm"))
    assert err and "holds it" in err["error"]
    assert not prov.stopped


def test_an_unknown_model_is_left_to_404_honestly(client, monkeypatch):
    prov, _ = _setup(monkeypatch, startable=False)
    assert asyncio.run(P._ensure_model_served("qwen3-coder-next", "vllm")) is None
    assert prov.loaded_with is None
