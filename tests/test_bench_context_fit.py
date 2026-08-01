"""The bench sizes the window it needs, and refuses runs that cannot produce a measurement.

Two failures this file exists to prevent, both observed on a real sweep:

- A 32,000-token prompt against llama.cpp serving a 32,768-token slot. Every request returns an
  empty completion, the run "succeeds", and the numbers are of nothing.
- A model benched against a backend that does not serve it. `_bench_resolve_model` returns {},
  the run proceeds anyway, and 54 requests go out. With `exclusive` set, every other proxy
  client is queued behind a run that was doomed at submission.
"""
import asyncio
import json
import time

from ai_proxy import proxy as P


# ---- window arithmetic ------------------------------------------------------------------

def test_needed_and_budget_are_inverses():
    """The planner asks "does it fit" and the resizer asks "how big" — if these two disagree,
    a cell is resized to a window it is then skipped for."""
    for prompt, max_tokens in ((32000, 256), (1000, 4096), (131072, 512)):
        need = P._bench_ctx_needed(prompt, max_tokens)
        assert P._bench_ctx_budget(need, max_tokens) >= prompt


def test_the_window_leaves_room_for_the_reply():
    # Sized to the prompt alone, every request returns an empty completion.
    assert P._bench_ctx_needed(32000, 256) > 32000 + 256


# ---- llama.cpp resizing -----------------------------------------------------------------

def _llamacpp(monkeypatch, *, n_ctx=32768, parallel=4, load_ok=True):
    """Stand in for a running llama-server and its systemd drop-in."""
    seen = []

    async def snap(client):
        return {"reachable": True, "n_ctx": n_ctx, "parallel": parallel}

    async def load(payload, name):
        seen.append(dict(payload))
        if not load_ok:
            return P.JSONResponse({"error": "llama-server exited 1"}, status_code=502)
        return {"ok": True, "upstream": "llamacpp", "ready": True,
                "context_length": payload.get("context_length"),
                "parallel": payload.get("parallel")}

    monkeypatch.setattr(P, "_llamacpp_snapshot", snap)
    monkeypatch.setattr(P.PROVIDERS["llamacpp"], "load", load)
    return seen


def test_lowering_parallel_is_what_buys_the_window(monkeypatch, client):
    """--ctx-size is the whole KV pool divided across --parallel slots, so a bigger per-request
    window at concurrency 1 costs *less* memory, not more: 1x36k beats 4x32k."""
    seen = _llamacpp(monkeypatch, n_ctx=32768, parallel=4)
    res = asyncio.run(P.PROVIDERS["llamacpp"].resize_context(35840, concurrency=1))
    assert res["ok"] and res["changed"]
    assert seen == [{"context_length": 35840, "parallel": 1}]
    assert seen[0]["context_length"] < 32768 * 4, "asked for a bigger pool than it needed"


def test_the_pool_is_sized_to_the_concurrency_being_measured(monkeypatch, client):
    seen = _llamacpp(monkeypatch, n_ctx=32768, parallel=4)
    asyncio.run(P.PROVIDERS["llamacpp"].resize_context(40000, concurrency=4))
    assert seen == [{"context_length": 160000, "parallel": 4}]


def test_a_big_enough_server_is_left_alone(monkeypatch, client):
    # Restarting a 90 GB model to change nothing costs minutes.
    seen = _llamacpp(monkeypatch, n_ctx=131072, parallel=4)
    res = asyncio.run(P.PROVIDERS["llamacpp"].resize_context(35840, concurrency=1))
    assert res["ok"] and res["changed"] is False
    assert seen == [], "restarted a server that already fit"
    assert res["previous"] is None, "nothing changed, so nothing needs restoring"


def test_a_wider_sweep_still_resizes_even_when_the_window_fits(monkeypatch, client):
    # Window is fine but there are only 2 slots for a concurrency-4 cell.
    seen = _llamacpp(monkeypatch, n_ctx=131072, parallel=2)
    res = asyncio.run(P.PROVIDERS["llamacpp"].resize_context(40000, concurrency=4))
    assert res["changed"] is True and seen[0]["parallel"] == 4


def test_a_resize_that_kills_the_server_is_rolled_back(monkeypatch, client):
    """A box left with nothing on port 8080 is worse than a bench that did not run."""
    seen = _llamacpp(monkeypatch, n_ctx=32768, parallel=4, load_ok=False)
    res = asyncio.run(P.PROVIDERS["llamacpp"].resize_context(262144, concurrency=1))
    assert res["ok"] is False
    assert len(seen) == 2, "did not attempt a rollback"
    assert seen[1] == {"context_length": 32768 * 4, "parallel": 4}, "rolled back to the wrong size"
    assert res["previous"] is None, "a failed resize must not hand back a restore token"


def test_vllm_declares_that_it_cannot_resize(client):
    # max_model_len is baked into the container's arguments; changing it means recreating the
    # container, which is a different and much less safe operation than a restart.
    assert P.PROVIDERS["vllm"].resizable_context is False
    assert P.PROVIDERS["llamacpp"].resizable_context is True
    res = asyncio.run(P._bench_fit_context("vllm", 40000, 1))
    assert res["ok"] is False and "cannot change its context window" in res["detail"]


def test_the_window_is_put_back_afterwards(monkeypatch, client):
    """A bench that permanently reconfigures the box silently changes what the next bench
    measures, and the difference reads as a property of the model."""
    seen = _llamacpp(monkeypatch, n_ctx=32768, parallel=4)
    fit = asyncio.run(P._bench_fit_context("llamacpp", 35840, 1))
    assert fit["restore"] == {"upstream": "llamacpp", "context_length": 131072, "parallel": 4}
    asyncio.run(P._bench_restore_context(fit["restore"]))
    assert seen[-1] == {"context_length": 131072, "parallel": 4}


def test_restoring_nothing_is_not_an_error(client):
    assert asyncio.run(P._bench_restore_context(None)) is None


# ---- fail-fast --------------------------------------------------------------------------

_INDEX = {
    "llamacpp:ds4-flash": {"model": "ds4-flash", "upstream": "llamacpp", "loaded": True},
    "ollama:qwen3:4b": {"model": "qwen3:4b", "upstream": "ollama", "loaded": False},
}


def test_a_model_the_backend_does_not_serve_is_refused(client):
    why = P._bench_preflight("qwen3:4b", {}, "llamacpp", _INDEX)
    assert why and "llamacpp does not serve" in why
    assert "ds4-flash" in why, "should say what it does serve"


def test_an_unreachable_backend_says_so_rather_than_naming_a_model(client):
    why = P._bench_preflight("anything", {}, "llamacpp", {})
    assert why and "not serving anything" in why


def test_a_model_no_backend_serves_is_refused(client):
    why = P._bench_preflight("gpt-9", {}, "", _INDEX)
    assert why and "no reachable backend serves" in why
    assert "ds4-flash" in why


def test_a_servable_model_passes(client):
    assert P._bench_preflight("ds4-flash", _INDEX["llamacpp:ds4-flash"], "llamacpp",
                              _INDEX) is None


def _queue(conn, bench_id, cfg, model, parent_id=None):
    conn.execute(
        "INSERT INTO bench_runs (id, ts, model, config_json, status, parent_id) "
        "VALUES (?, ?, ?, ?, 'pending', ?)",
        (bench_id, time.time(), model, json.dumps(cfg), parent_id))
    conn.commit()


def test_a_doomed_run_sends_no_requests(client, monkeypatch):
    """The point of failing fast. Before this, the run went ahead: 54 requests of empty
    completions, and under `exclusive` every other proxy client queued behind them."""
    sent = []

    async def index():
        return dict(_INDEX)

    async def run_one(*a, **kw):
        sent.append(1)
        return {"ok": True}

    monkeypatch.setattr(P, "_bench_model_index", index)
    monkeypatch.setattr(P, "_bench_run_one", run_one)

    bid = "b_doomed"
    conn = P.db()
    _queue(conn, bid, {"upstream": "llamacpp", "runs": 54, "warmup": False}, "qwen3:4b")
    conn.close()

    asyncio.run(P._bench_execute(bid, P.app))

    conn = P.db()
    row = conn.execute("SELECT status, error FROM bench_runs WHERE id=?", (bid,)).fetchone()
    conn.close()
    assert row["status"] == "failed"
    assert "does not serve" in row["error"]
    assert sent == [], f"sent {len(sent)} requests for a run that could never work"


def test_a_run_too_big_for_a_fixed_backend_is_refused_not_attempted(client, monkeypatch):
    """vLLM cannot be resized, so this is the case that must still fail rather than run."""
    sent = []

    async def index():
        return {"vllm:qwen": {"model": "qwen", "upstream": "vllm", "loaded": True,
                              "loaded_context": 8192, "max_context": 8192}}

    async def run_one(*a, **kw):
        sent.append(1)
        return {"ok": True}

    monkeypatch.setattr(P, "_bench_model_index", index)
    monkeypatch.setattr(P, "_bench_run_one", run_one)

    bid = "b_toobig"
    conn = P.db()
    _queue(conn, bid, {"upstream": "vllm", "runs": 3, "prompt_tokens": 32000,
                       "warmup": False}, "qwen")
    conn.close()

    asyncio.run(P._bench_execute(bid, P.app))

    conn = P.db()
    row = conn.execute("SELECT status, error FROM bench_runs WHERE id=?", (bid,)).fetchone()
    conn.close()
    assert row["status"] == "failed"
    assert "32,000 prompt tokens need" in row["error"]
    assert "8,192" in row["error"], "should say what the backend actually serves"
    assert sent == []
