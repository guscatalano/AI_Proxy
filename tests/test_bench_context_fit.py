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
    assert it["cells"]["now"]["label"] == "ds4 · 131k ctx · cold"
    assert it["cells"]["now"]["progress"] == 5
    assert it["cells"]["now"]["progress_total"] == 12


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


# ---- server_context: the window as an axis in its own right -----------------------------

def test_server_context_is_its_own_axis(client):
    """prompt_tokens is what a request sends; server_context is what the backend was launched
    with. Conflating them is why a sweep of [32k, 131k, 262k] measured one thing six times."""
    cells = P._bench_expand_matrix([{"model": "m", "upstream": "llamacpp"}],
                                   {"server_context": [32768, 262144], "prompt_tokens": 0})
    assert [c["server_context"] for c in cells] == [32768, 262144]


def test_cells_are_grouped_so_each_window_is_set_once(client):
    # Each distinct window costs a restart and a full model reload.
    cells = P._bench_expand_matrix(
        [{"model": "m", "upstream": "llamacpp"}],
        {"server_context": [262144, 32768], "cache": ["cold", "cached"]})
    ctxs = [c["server_context"] for c in cells]
    assert ctxs == sorted(ctxs), "cells not grouped by window; that is a restart per cell"
    # ...and the cache axis still runs cold before cached inside each group.
    for ctx in (32768, 262144):
        grp = [c["cache"] for c in cells if c["server_context"] == ctx]
        assert grp == ["cold", "cached"], grp


def test_the_label_distinguishes_prompt_from_window(client):
    lab = P._bench_cell_label({"model": "/models/big/DeepSeek-V4-Flash-0731-UD-IQ2_XXS.gguf",
                               "upstream": "llamacpp", "prompt_tokens": 32000,
                               "server_context": 262144})
    assert "32k prompt" in lab, "a prompt size labelled 'ctx' is what caused the confusion"
    assert "256k ctx" in lab
    assert "/models/" not in lab, "the full path makes every table unreadable"


def test_a_model_path_is_shortened_for_display(client):
    d = P._bench_model_display(
        "/home/crimson/models/ds4-flash/UD-IQ2_XXS/"
        "DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf")
    assert d == "DeepSeek-V4-Flash-0731-UD-IQ2_XXS"


def test_shortening_leaves_ordinary_names_alone(client):
    for name in ("qwen3:4b", "ornith-nvfp4", "gpt-oss-120b"):
        assert P._bench_model_display(name) == name


def test_an_exact_resize_will_shrink_the_window(client, monkeypatch):
    """Fitting a prompt only ever grows the window. An axis has to be able to set 32k when the
    server is at 292k, or half the comparison is impossible."""
    seen = _llamacpp(monkeypatch, n_ctx=291840, parallel=1)
    res = asyncio.run(P.PROVIDERS["llamacpp"].resize_context(32768, 1, exact=True))
    assert res["ok"] and res["changed"]
    assert seen[0]["context_length"] == 32768

    # Without exact=, the same call is a no-op because 291840 already "fits".
    seen2 = _llamacpp(monkeypatch, n_ctx=291840, parallel=1)
    assert asyncio.run(P.PROVIDERS["llamacpp"].resize_context(32768, 1))["changed"] is False
    assert seen2 == []


def test_a_resize_is_refused_when_the_current_window_cannot_be_read(client, monkeypatch):
    """The bug that left llama.cpp at 291,784/1 overnight: the probe came back empty, the
    resize went ahead anyway, and the restore had nothing to restore to."""
    seen = _llamacpp(monkeypatch, n_ctx=0, parallel=1)
    res = asyncio.run(P.PROVIDERS["llamacpp"].resize_context(65536, 1))
    assert res["ok"] is False
    assert "could not be undone" in res["detail"]
    assert seen == [], "changed the box without knowing how to change it back"


def test_every_sweepable_axis_survives_submission(client):
    """The submit endpoint rebuilds config from an explicit whitelist, and the matrix expander
    reads whatever it finds there. server_context was added to the expander and not the
    whitelist, so the UI sent it, the server dropped it, and a 'long context' sweep ran as two
    short cells with nothing anywhere saying why. This pins the two lists together."""
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.commit()
    conn.close()

    payload = {"models": [{"model": "ds4", "upstream": "llamacpp"}], "runs": 1,
               "prompt_tokens": [0, 1000], "thinking": ["off", "on"],
               "temperature": [0.0, 0.7], "cache": ["cold", "cached"],
               "concurrency": [1, 2], "server_context": [32768, 65536]}
    r = client.post("/__proxy/api/bench/run", json=payload)
    assert r.status_code == 200, r.text

    conn = P.db()
    cfg = json.loads(conn.execute(
        "SELECT config_json FROM bench_runs WHERE id=?", (r.json()["id"],)).fetchone()[0])
    conn.close()
    for axis in ("prompt_tokens", "thinking", "temperature", "cache", "concurrency",
                 "server_context"):
        assert isinstance(cfg.get(axis), list), f"{axis} did not survive submission"
    # And the expander agrees: 2^6 combinations.
    assert len(P._bench_expand_matrix(payload["models"], cfg)) == 64


def test_a_long_context_sweep_is_not_submitted_as_short_cells(client):
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.commit()
    conn.close()
    r = client.post("/__proxy/api/bench/run", json={
        "models": [{"model": "ds4", "upstream": "llamacpp"}],
        "suite": "coding-v1", "prompt_tokens": 0, "cache": ["cold", "cached"],
        "server_context": [32768, 131072, 262144], "runs": 1})
    assert r.status_code == 200, r.text
    assert r.json()["cells"] == 6, "the window axis collapsed"

    # Cells are written by a background task, so read the plan the parent stored rather than
    # racing it.
    conn = P.db()
    cfg = json.loads(conn.execute(
        "SELECT config_json FROM bench_runs WHERE id=?", (r.json()["id"],)).fetchone()[0])
    conn.close()
    labels = [P._bench_cell_label(c) for c in P._bench_expand_matrix(cfg["models"], cfg)]
    assert any("32k ctx" in l for l in labels), labels
    assert any("256k ctx" in l for l in labels), labels
    assert not all(l.endswith("short") for l in labels), "every cell came out short"


# ---- ETA ---------------------------------------------------------------------------------

def _cell(status, started, finished, prog=0, total=36):
    return {"status": status, "started_ts": started, "finished_ts": finished,
            "progress": prog, "progress_total": total}


def test_eta_counts_the_gap_between_cells(client):
    """The cost a naive estimate misses. On a server_context sweep the gap between cells is a
    backend restart and a full model reload -- often longer than the cell itself -- so an ETA
    built from cell durations alone reads far too optimistic."""
    now = 10_000.0
    cells = [_cell("done", 0, 400, 36),          # 400s of work
             _cell("done", 800, 1200, 36),       # ...preceded by a 400s reload
             _cell("pending", None, None)]
    eta = P._bench_eta_s(cells, now)
    assert eta is not None
    assert 750 <= eta <= 850, f"expected ~800s (work + reload), got {eta}"


def test_eta_projects_the_running_cell_from_its_own_pace(client):
    now = 1_000.0
    # Half done after 300s -> 300s left, and nothing pending.
    cells = [_cell("running", 700, None, prog=18, total=36)]
    # No completed cell yet, so there is no basis at all.
    assert P._bench_eta_s(cells, now) is None
    cells = [_cell("done", 0, 600, 36), _cell("running", 700, None, prog=18, total=36)]
    eta = P._bench_eta_s(cells, now)
    assert 250 <= eta <= 350, eta


def test_a_finished_sweep_has_no_eta(client):
    assert P._bench_eta_s([_cell("done", 0, 400, 36), _cell("done", 500, 900, 36)],
                          10_000.0) is None


def test_skipped_and_failed_cells_are_not_waited_for(client):
    """A cell that already failed is not work still to come. Cells are laid out contiguously so
    no gap is observed either way, leaving the count of remaining cells as the only difference."""
    now = 10_000.0
    two_pending = [_cell("done", 0, 400, 36),
                   _cell("pending", None, None), _cell("pending", None, None)]
    one_dead = [_cell("done", 0, 400, 36),
                _cell("failed", 400, 500), _cell("pending", None, None)]
    assert P._bench_eta_s(two_pending, now) == 800
    assert P._bench_eta_s(one_dead, now) == 400


def test_the_eta_reaches_the_history_row(client):
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    now = time.time()
    conn.execute("INSERT INTO bench_runs (id, ts, model, config_json, status) "
                 "VALUES ('b_p', ?, 'm', '{}', 'running')", (now - 900,))
    conn.execute("INSERT INTO bench_runs (id, ts, model, config_json, status, parent_id, "
                 "started_ts, finished_ts, progress, progress_total) "
                 "VALUES ('b_p1', ?, 'm', '{}', 'done', 'b_p', ?, ?, 36, 36)",
                 (now - 900, now - 900, now - 500))
    conn.execute("INSERT INTO bench_runs (id, ts, model, config_json, status, parent_id, "
                 "label, progress, progress_total) "
                 "VALUES ('b_p2', ?, 'm', '{}', 'pending', 'b_p', 'cell 2', 0, 36)", (now - 900,))
    conn.commit()
    conn.close()
    it = client.get("/__proxy/api/bench/runs").json()["items"][0]
    assert it["cells"]["eta_s"] > 0


def test_the_window_axis_does_not_expand_where_it_cannot_apply(client):
    """One tick of "Long context" produced 102 cells, 84 of which failed instantly with
    "ollama cannot change its context window" — and the three per model would have been
    identical anyway. An axis a backend cannot honour is not a failure for it, it is not an
    axis for it."""
    models = [{"model": "ds4", "upstream": "llamacpp"},
              {"model": "qwen3:4b", "upstream": "ollama"},
              {"model": "qwen3-coder-next", "upstream": "vllm"}]
    cells = P._bench_expand_matrix(models, {"server_context": [32768, 131072, 262144],
                                            "cache": ["cold", "cached"]})
    per = {}
    for c in cells:
        per.setdefault(c["upstream"], []).append(c)
    assert len(per["llamacpp"]) == 6, "the backend that can resize keeps the axis"
    assert len(per["ollama"]) == 2, f"ollama got {len(per['ollama'])} cells, expected 2"
    assert len(per["vllm"]) == 2, "vLLM bakes max_model_len into its container arguments"
    assert all("server_context" not in c for c in per["ollama"])
    assert sorted(c["server_context"] for c in per["llamacpp"]) == [
        32768, 32768, 131072, 131072, 262144, 262144]


def test_a_window_sweep_of_one_backend_is_unchanged(client):
    cells = P._bench_expand_matrix([{"model": "ds4", "upstream": "llamacpp"}],
                                   {"server_context": [32768, 262144]})
    assert [c["server_context"] for c in cells] == [32768, 262144]
