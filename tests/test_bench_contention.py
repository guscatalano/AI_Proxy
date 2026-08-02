"""Every cell gets the box to itself.

Observed: an eighteen-model sweep whose first cell measured llama.cpp legitimately, and whose
every subsequent Ollama cell then inherited a machine with ~90 GB already spoken for. Ollama
fitted 28 of codellama:70b's 38 GB into VRAM and spilled the rest; one request took 24 minutes
instead of ten seconds. The run was measuring memory pressure, not the model — and would have
taken 21 hours for that one cell.

The cause was wiring the free to "am I starting a backend". An Ollama cell starts nothing.
"""
import asyncio

from ai_proxy import proxy as P

# Bound before anything shadows it: a replacement that calls asyncio.sleep by name calls
# itself.
_REAL_SLEEP = asyncio.sleep


def _snap(**running):
    return {"backends": [{"name": n, "was_running": r, "control": "unit"}
                         for n, r in running.items()], "ollama": []}


def test_the_backend_being_measured_is_not_stopped(client, monkeypatch):
    stopped = []

    class _B:
        def __init__(self, name):
            self.name = name

        async def stop(self):
            stopped.append(self.name)
            return {"ok": True}

    monkeypatch.setattr(P, "backend", lambda n: _B(n))
    monkeypatch.setattr(P, "_bench_evict_ollama", lambda keep="": _REAL_SLEEP(0, result=[]))
    monkeypatch.setattr(P, "_free_mem_mb", lambda: 100000)
    monkeypatch.setattr(P.asyncio, "sleep", lambda *a, **kw: _REAL_SLEEP(0))

    asyncio.run(P._bench_free_gpu(_snap(llamacpp=True, comfyui=True, vllm=True),
                                  spare="llamacpp"))
    assert "llamacpp" not in stopped, "stopped the backend it was about to measure"
    assert set(stopped) == {"comfyui", "vllm"}


def test_everything_else_is_stopped_for_an_ollama_cell(client, monkeypatch):
    """Ollama is not a backend the proxy starts or stops, so nothing is spared and llama.cpp
    has to go — which is exactly what did not happen."""
    stopped = []

    class _B:
        def __init__(self, name):
            self.name = name

        async def stop(self):
            stopped.append(self.name)
            return {"ok": True}

    monkeypatch.setattr(P, "backend", lambda n: _B(n))
    monkeypatch.setattr(P, "_bench_evict_ollama", lambda keep="": _REAL_SLEEP(0, result=[]))
    monkeypatch.setattr(P, "_free_mem_mb", lambda: 100000)
    monkeypatch.setattr(P.asyncio, "sleep", lambda *a, **kw: _REAL_SLEEP(0))

    asyncio.run(P._bench_free_gpu(_snap(llamacpp=True), spare="ollama"))
    assert stopped == ["llamacpp"]


def test_what_was_already_down_is_not_started_by_freeing(client, monkeypatch):
    stopped = []

    class _B:
        def __init__(self, name):
            self.name = name

        async def stop(self):
            stopped.append(self.name)
            return {"ok": True}

    monkeypatch.setattr(P, "backend", lambda n: _B(n))
    monkeypatch.setattr(P, "_bench_evict_ollama", lambda keep="": _REAL_SLEEP(0, result=[]))
    monkeypatch.setattr(P, "_free_mem_mb", lambda: 100000)
    monkeypatch.setattr(P.asyncio, "sleep", lambda *a, **kw: _REAL_SLEEP(0))
    asyncio.run(P._bench_free_gpu(_snap(llamacpp=False, vllm=True), spare=""))
    assert stopped == ["vllm"]


def test_every_cell_frees_not_only_ones_that_start_something(client):
    """The bug in one line: the free lived inside _bench_start_backend, which an Ollama cell
    never calls."""
    import inspect
    src = inspect.getsource(P._bench_execute)
    i_free = src.index("_bench_free_gpu(snap_now")
    i_start = src.index("_bench_start_backend(model_meta")
    assert i_free < i_start, "freeing still happens only as part of starting a backend"
    assert 'spare=this_up' in src, "the backend under test must be spared"


def test_a_standalone_run_puts_the_others_back(client):
    """A sweep's suite restores at the end; a single run has no suite to do it."""
    import inspect
    src = inspect.getsource(P._bench_execute)
    assert 'if row["parent_id"] is None:\n                    resid_snap = snap_now' in src
    assert "_bench_restore_residency(resid_snap)" in src


# ---- waiting for the memory, not for the signal ---------------------------------------------

def test_it_waits_until_the_memory_is_actually_back(client, monkeypatch):
    """`docker stop` and `systemctl stop` return when the process is signalled, not when its
    memory is unmapped. Releasing ~90 GB takes appreciably longer, and the bench used to sleep a
    flat three seconds and start loading into a machine still handing memory back."""
    # A ramp that settles, rather than a fixed list: the number of reads is an implementation
    # detail and a test that pins it breaks on every refactor.
    seq = [10_000, 20_000, 40_000, 80_000]
    calls = {"n": 0}

    def mem():
        i = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        return seq[i]

    monkeypatch.setattr(P, "_free_mem_mb", mem)
    monkeypatch.setattr(P, "backend", lambda n: None)
    monkeypatch.setattr(P, "_bench_evict_ollama", lambda keep="": _REAL_SLEEP(0, result=[]))
    monkeypatch.setattr(P.asyncio, "sleep", lambda *a, **kw: _REAL_SLEEP(0))

    out = asyncio.run(P._bench_free_gpu(_snap(llamacpp=True), want_free_mb=70_000))
    assert out["reached_target"] is True
    assert out["free_mb_after"] >= 70_000


def test_it_gives_up_rather_than_waiting_forever(client, monkeypatch):
    """A target that never arrives must end the wait and be recorded, not stall the run."""
    monkeypatch.setattr(P, "_free_mem_mb", lambda: 5_000)
    monkeypatch.setattr(P, "backend", lambda n: None)
    monkeypatch.setattr(P, "_bench_evict_ollama", lambda keep="": _REAL_SLEEP(0, result=[]))
    monkeypatch.setattr(P.asyncio, "sleep", lambda *a, **kw: _REAL_SLEEP(0))

    out = asyncio.run(P._bench_free_gpu(_snap(llamacpp=True), want_free_mb=70_000,
                                        timeout_s=0.05))
    assert out["reached_target"] is False
    assert out["wanted_mb"] == 70_000


def test_the_wait_reports_what_it_is_waiting_for(client, monkeypatch):
    """Otherwise it is another silent multi-minute step that reads as a hang."""
    said = []
    seq = [10_000, 10_000, 90_000]
    calls = {"n": 0}

    def mem():
        i = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        return seq[i]

    monkeypatch.setattr(P, "_free_mem_mb", mem)
    monkeypatch.setattr(P, "backend", lambda n: None)
    monkeypatch.setattr(P, "_bench_evict_ollama", lambda keep="": _REAL_SLEEP(0, result=[]))
    monkeypatch.setattr(P.asyncio, "sleep", lambda *a, **kw: _REAL_SLEEP(0))
    monkeypatch.setattr(P, "_bench_phase", lambda bid, txt: said.append(txt))

    # 70 GiB expressed in MB, so the rendered message reads in whole GB.
    asyncio.run(P._bench_free_gpu(_snap(llamacpp=True), want_free_mb=70 * 1024, bench_id="b_x"))
    assert any("waiting for memory" in (t or "") for t in said), said
    assert any("need 70 GB" in (t or "") for t in said), said


def test_the_bench_asks_for_what_the_model_needs(client):
    """The wait needs a target or it does not happen at all — both call sites passed none, so
    the loop that existed for exactly this never ran."""
    import inspect
    src = inspect.getsource(P._bench_execute)
    assert "want_free_mb=_want" in src, "the cell still frees without a target"
    assert "_BENCH_FIT_OVERHEAD" in src, "the target must allow for more than the weights"
    src2 = inspect.getsource(P._bench_start_backend)
    assert "want_free_mb=" in src2


def test_falling_short_is_recorded_beside_the_numbers(client):
    """A model running partly offloaded still produces a number; it is just a number about
    memory pressure. That has to be visible, not inferred afterwards."""
    import inspect
    src = inspect.getsource(P._bench_execute)
    assert 'env["memory_warning"]' in src
