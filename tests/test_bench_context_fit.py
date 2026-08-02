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
    assert seen == [{"context_length": 35840, "parallel": 1,
                     "ready_timeout_s": P._BENCH_RESIZE_READY_S}]
    assert seen[0]["context_length"] < 32768 * 4, "asked for a bigger pool than it needed"


def test_the_pool_is_sized_to_the_concurrency_being_measured(monkeypatch, client):
    seen = _llamacpp(monkeypatch, n_ctx=32768, parallel=4)
    asyncio.run(P.PROVIDERS["llamacpp"].resize_context(40000, concurrency=4))
    assert seen == [{"context_length": 160000, "parallel": 4,
                     "ready_timeout_s": P._BENCH_RESIZE_READY_S}]


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


def test_a_resize_waits_a_bounded_time_before_giving_up(monkeypatch, client):
    """load() defaults to a 1800s ready poll. The usual reason a resize never comes back is
    that the pool did not fit in memory, and waiting half an hour to learn that — before even
    starting the rollback — is the whole cost of the mistake."""
    seen = _llamacpp(monkeypatch, n_ctx=32768, parallel=4)
    asyncio.run(P.PROVIDERS["llamacpp"].resize_context(35840, concurrency=1))
    assert seen[0]["ready_timeout_s"] == P._BENCH_RESIZE_READY_S
    assert P._BENCH_RESIZE_READY_S < 1800


def test_a_resize_that_kills_the_server_is_rolled_back(monkeypatch, client):
    """A box left with nothing on port 8080 is worse than a bench that did not run."""
    seen = _llamacpp(monkeypatch, n_ctx=32768, parallel=4, load_ok=False)
    res = asyncio.run(P.PROVIDERS["llamacpp"].resize_context(262144, concurrency=1))
    assert res["ok"] is False
    assert len(seen) == 2, "did not attempt a rollback"
    assert seen[1]["context_length"] == 32768 * 4, "rolled back to the wrong size"
    assert seen[1]["parallel"] == 4, "rolled back to the wrong slot count"
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
    assert seen[-1]["context_length"] == 131072 and seen[-1]["parallel"] == 4


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


# ---- history grouping -------------------------------------------------------------------

def _sweep(conn, parent, kids):
    """A sweep as _bench_execute_suite writes one: a parent plus a row per cell."""
    now = time.time()
    conn.execute(
        "INSERT INTO bench_runs (id, ts, model, config_json, status) VALUES (?,?,?,?,'done')",
        (parent, now, "ds4-flash", json.dumps({"models": ["ds4-flash"], "runs": 3})))
    for i, status in enumerate(kids):
        conn.execute(
            "INSERT INTO bench_runs (id, ts, model, config_json, status, parent_id, label) "
            "VALUES (?,?,?,?,?,?,?)",
            (f"{parent}_c{i}", now, "ds4-flash", "{}", status, parent, f"cell {i}"))
    conn.commit()


def test_one_click_is_one_history_entry(client):
    """A sweep's cells are rows in this table too, so a single click showed up as 25 entries —
    and each cell was already rendered inside its parent's detail view, so it was duplication
    as well as noise."""
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    _sweep(conn, "b_sweep", ["done", "done", "failed", "skipped"])
    conn.close()

    items = client.get("/__proxy/api/bench/runs").json()["items"]
    assert [i["id"] for i in items] == ["b_sweep"]


def test_the_row_says_how_the_cells_landed(client):
    # "done" alone hides a matrix that skipped or failed half of itself.
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    _sweep(conn, "b_sweep", ["done", "done", "failed", "skipped"])
    conn.close()

    it = client.get("/__proxy/api/bench/runs").json()["items"][0]
    assert it["cells"] == {"total": 4, "done": 2, "failed": 1, "skipped": 1}


def test_cells_can_still_be_listed_when_asked_for(client):
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    _sweep(conn, "b_sweep", ["done", "done"])
    conn.close()

    items = client.get("/__proxy/api/bench/runs?include_children=1").json()["items"]
    assert len(items) == 3


def test_a_sweep_no_longer_crowds_out_earlier_runs(client):
    """The limit counted cells, so two 24-cell sweeps pushed every earlier run off the end."""
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    _sweep(conn, "b_big", ["done"] * 24)
    conn.execute(
        "INSERT INTO bench_runs (id, ts, model, config_json, status) VALUES (?,?,?,?,'done')",
        ("b_solo", time.time() - 60, "qwen3:4b", "{}"))
    conn.commit()
    conn.close()

    items = client.get("/__proxy/api/bench/runs?limit=2").json()["items"]
    assert {i["id"] for i in items} == {"b_big", "b_solo"}


def test_a_standalone_run_has_no_cell_rollup(client):
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.execute(
        "INSERT INTO bench_runs (id, ts, model, config_json, status) VALUES (?,?,?,?,'done')",
        ("b_solo", time.time(), "qwen3:4b", "{}"))
    conn.commit()
    conn.close()

    assert "cells" not in client.get("/__proxy/api/bench/runs").json()["items"][0]


# ---- not waiting on a dead server -------------------------------------------------------

def test_ready_stops_polling_a_process_that_exited(client, monkeypatch):
    """The usual reason a resize never becomes ready is that the KV pool did not fit. Polling
    a corpse for the full timeout turns that fast, explicable failure into 15 minutes of
    silence — and the rollback cannot start until it ends."""
    polls = []

    class _Resp:
        status_code = 503

    class _C:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, *a, **kw):
            polls.append(url)
            return _Resp()

    monkeypatch.setattr(P.httpx, "AsyncClient", lambda *a, **kw: _C())
    # Bind the real sleep before shadowing it, or the replacement calls itself.
    _real_sleep = asyncio.sleep
    monkeypatch.setattr(P.asyncio, "sleep", lambda *a, **kw: _real_sleep(0))

    async def died(self=None):
        return "llamacpp.service is failed: failed to allocate KV cache"
    monkeypatch.setattr(P.PROVIDERS["llamacpp"], "died", died)

    assert asyncio.run(P.PROVIDERS["llamacpp"].ready(900.0)) is False
    assert len(polls) <= 2, f"kept polling a dead server {len(polls)} times"


def test_a_still_starting_unit_is_not_mistaken_for_a_dead_one(client, monkeypatch):
    # Type=simple reports 'activating' briefly; aborting there would kill every slow load.
    async def run(args, timeout=120.0, max_chars=800, keep_tail=False, env=None):
        if "is-active" in args:
            return 0, "activating"
        return 0, "enabled"

    cfg = dict(P.load_rules_config())
    mc = dict(cfg.get("model_control") or {})
    mc["services"] = {"llamacpp": {"unit": "llamacpp.service", "scope": "user"}}
    cfg["model_control"] = mc
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    monkeypatch.setattr(P, "_run_cmd", run)
    assert asyncio.run(P.PROVIDERS["llamacpp"].died()) is None


def test_a_failed_unit_reports_why(client, monkeypatch):
    async def run(args, timeout=120.0, max_chars=800, keep_tail=False, env=None):
        if "is-active" in args:
            return 0, "failed"
        if "status" in args:
            return 0, "llama-server: failed to allocate KV cache: out of memory"
        return 0, "enabled"

    cfg = dict(P.load_rules_config())
    mc = dict(cfg.get("model_control") or {})
    mc["services"] = {"llamacpp": {"unit": "llamacpp.service", "scope": "user"}}
    cfg["model_control"] = mc
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    monkeypatch.setattr(P, "_run_cmd", run)
    why = asyncio.run(P.PROVIDERS["llamacpp"].died())
    assert why and "out of memory" in why


def test_an_unmanaged_backend_never_claims_to_have_died(client):
    # died() gates an early abort, so an uncertain answer must be None.
    assert asyncio.run(P.PROVIDERS["ollama"].died()) is None


def test_the_running_cell_is_visible_on_the_parent(client):
    """Grouping the cells removed the only place per-cell progress showed. The parent's own
    counter ticks once per finished cell, so a graded 262k sweep sits at 0/6 for many minutes
    and reads as stuck."""
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    now = time.time()
    conn.execute(
        "INSERT INTO bench_runs (id, ts, model, config_json, status, progress, progress_total) "
        "VALUES (?,?,?,?,'running',0,6)",
        ("b_sweep", now, "ds4-flash", json.dumps({"models": ["ds4-flash"]})))
    conn.execute(
        "INSERT INTO bench_runs (id, ts, model, config_json, status, parent_id, label, "
        "progress, progress_total) VALUES (?,?,?,'{}','done',?,?,?,?)",
        ("b_sweep_c0", now, "ds4-flash", "b_sweep", "cell 0", 12, 12))
    conn.execute(
        "INSERT INTO bench_runs (id, ts, model, config_json, status, parent_id, label, "
        "progress, progress_total) VALUES (?,?,?,'{}','running',?,?,?,?)",
        ("b_sweep_c1", now, "ds4-flash", "b_sweep", "ds4 · 131k ctx · cold", 5, 12))
    conn.commit()
    conn.close()

    it = client.get("/__proxy/api/bench/runs").json()["items"][0]
    assert it["cells"]["now"] == {"label": "ds4 · 131k ctx · cold",
                                 "progress": 5, "progress_total": 12}


def test_a_finished_sweep_has_no_running_cell(client):
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    _sweep(conn, "b_sweep", ["done", "done"])
    conn.close()
    assert "now" not in client.get("/__proxy/api/bench/runs").json()["items"][0]["cells"]


# ---- one bench at a time ----------------------------------------------------------------

def _submit(client, model="qwen3:4b"):
    return client.post("/__proxy/api/bench/run",
                       json={"models": [{"model": model, "upstream": "ollama"}], "runs": 1})


def test_a_second_bench_is_refused_while_one_runs(client):
    """_BENCH_SEM would queue it instead. "Eventually" is the problem: a bench takes the box
    exclusively, resizes context windows and evicts residents, so a run that starts an hour
    later measures a machine configured by whatever ran before it — and nothing in its results
    would say so."""
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.execute(
        "INSERT INTO bench_runs (id, ts, model, config_json, status, label) "
        "VALUES (?,?,?,'{}','running',?)",
        ("b_busy", time.time(), "ds4-flash", "ds4 · 262k ctx"))
    conn.commit()
    conn.close()

    r = _submit(client)
    assert r.status_code == 409
    d = r.json()
    assert "already running" in d["error"]
    assert "262k" in d["error"], "should name what is running"
    assert d["running"]["id"] == "b_busy"

    conn = P.db()
    n = conn.execute("SELECT COUNT(*) c FROM bench_runs").fetchone()["c"]
    conn.close()
    assert n == 1, "the refused submission was queued anyway"


def test_a_running_cell_blocks_even_if_its_parent_row_is_gone(client):
    # A suite that failed mid-way can leave cells behind; they still own the box.
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.execute(
        "INSERT INTO bench_runs (id, ts, model, config_json, status, parent_id, label) "
        "VALUES (?,?,?,'{}','running',?,?)",
        ("b_cell", time.time(), "ds4-flash", "b_gone", "cell 3"))
    conn.commit()
    conn.close()
    assert _submit(client).status_code == 409


def test_a_finished_bench_does_not_block(client):
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.execute(
        "INSERT INTO bench_runs (id, ts, model, config_json, status) VALUES (?,?,?,'{}','done')",
        ("b_old", time.time() - 600, "qwen3:4b"))
    conn.commit()
    conn.close()
    assert _submit(client).status_code == 200


def test_an_interrupted_bench_does_not_block_forever(client):
    """The guard reads the same rows the restart-recovery clears, so a crash mid-bench cannot
    leave the endpoint permanently refusing."""
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.execute(
        "INSERT INTO bench_runs (id, ts, model, config_json, status) "
        "VALUES (?,?,?,'{}','running')", ("b_stranded", time.time() - 3600, "qwen3:4b"))
    conn.commit()
    # Exactly what startup recovery does.
    conn.execute("UPDATE bench_runs SET status='failed', error='interrupted' "
                 "WHERE status IN ('pending','running')")
    conn.commit()
    conn.close()
    assert _submit(client).status_code == 200
