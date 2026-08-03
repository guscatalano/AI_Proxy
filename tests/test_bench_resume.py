"""Reusing cells an earlier run already measured.

A "test everything" sweep across every model on this box is a two-to-four hour job. Any proxy
restart marks every pending cell interrupted and the whole thing is lost — and this codebase
gets deployed several times an hour. Without reuse, a long sweep is a bet that nobody ships
anything.
"""
import json
import time

from ai_proxy import proxy as P


def _done_cell(conn, cid, model, cfg, ts=None, results='{"summary": {"n_success": 36}}'):
    ts = ts if ts is not None else time.time()
    conn.execute(
        "INSERT INTO bench_runs (id, ts, model, config_json, status, results_json, env_json, "
        "started_ts, finished_ts, progress, progress_total) "
        "VALUES (?,?,?,?,'done',?,'{}',?,?,36,36)",
        (cid, ts, model, json.dumps(cfg), results, ts, ts + 300))
    conn.commit()


_CFG = {"upstream": "ollama", "suite": "coding-v1", "runs": 2, "max_tokens": 512,
        "prompt_tokens": 0, "thinking": "off", "cache": "cold", "concurrency": 1}


def test_identical_settings_produce_the_same_signature(client):
    assert P._bench_cell_sig("m", dict(_CFG)) == P._bench_cell_sig("m", dict(_CFG))


def test_any_setting_that_changes_the_measurement_changes_the_signature(client):
    base = P._bench_cell_sig("m", dict(_CFG))
    for key, val in (("runs", 5), ("suite", "other"), ("thinking", "on"), ("cache", "cached"),
                     ("max_tokens", 64), ("concurrency", 4), ("server_context", 32768),
                     ("prompt_tokens", 32000), ("upstream", "vllm")):
        cfg = dict(_CFG)
        cfg[key] = val
        assert P._bench_cell_sig("m", cfg) != base, f"{key} did not change the signature"
    assert P._bench_cell_sig("other-model", dict(_CFG)) != base


def test_a_completed_cell_is_found_by_its_signature(client):
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    _done_cell(conn, "b_old", "qwen3:4b", _CFG)
    found = P._bench_completed_cells(conn)
    conn.close()
    assert P._bench_cell_sig("qwen3:4b", _CFG) in found


def test_a_sweep_parent_is_not_offered_as_a_measurement(client):
    """It measures nothing itself; its cells do."""
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    _done_cell(conn, "b_parent", "qwen3:4b", dict(_CFG, models=[{"model": "qwen3:4b"}]))
    found = P._bench_completed_cells(conn)
    conn.close()
    assert found == {}


def test_the_most_recent_measurement_wins(client):
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    now = time.time()
    _done_cell(conn, "b_older", "qwen3:4b", _CFG, ts=now - 5000)
    _done_cell(conn, "b_newer", "qwen3:4b", _CFG, ts=now - 10)
    found = P._bench_completed_cells(conn)
    conn.close()
    assert found[P._bench_cell_sig("qwen3:4b", _CFG)]["id"] == "b_newer"


def test_history_prices_a_model_from_what_it_actually_took(client):
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    _done_cell(conn, "b_1", "slowpoke", _CFG)          # 300s by construction
    cost = P._bench_history_cost(conn)
    conn.close()
    assert 290 <= cost["slowpoke"] <= 310


def test_reuse_is_on_unless_turned_off(client):
    """Default-on. The signature covers every setting that changes what a cell measures, so a
    reused cell is one that would have produced the same numbers anyway — and three multi-hour
    runs went out unprotected because this had to be remembered."""
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.commit()
    conn.close()
    r = client.post("/__proxy/api/bench/run",
                    json={"models": [{"model": "qwen3:4b", "upstream": "ollama"}], "runs": 1})
    assert r.status_code == 200
    conn = P.db()
    cfg = json.loads(conn.execute("SELECT config_json FROM bench_runs WHERE id=?",
                                  (r.json()["id"],)).fetchone()[0])
    conn.close()
    assert cfg["resume"] is True


def test_reuse_can_still_be_turned_off(client):
    """Forcing a fresh measurement of everything has to stay possible."""
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.commit()
    conn.close()
    r = client.post("/__proxy/api/bench/run",
                    json={"models": [{"model": "qwen3:4b", "upstream": "ollama"}],
                          "runs": 1, "resume": False})
    conn = P.db()
    cfg = json.loads(conn.execute("SELECT config_json FROM bench_runs WHERE id=?",
                                  (r.json()["id"],)).fetchone()[0])
    conn.close()
    assert cfg["resume"] is False


def test_resume_survives_submission(client):
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.commit()
    conn.close()
    r = client.post("/__proxy/api/bench/run",
                    json={"models": [{"model": "qwen3:4b", "upstream": "ollama"}],
                          "runs": 1, "resume": True})
    conn = P.db()
    cfg = json.loads(conn.execute("SELECT config_json FROM bench_runs WHERE id=?",
                                  (r.json()["id"],)).fetchone()[0])
    conn.close()
    assert cfg["resume"] is True


def test_a_resumed_sweep_copies_the_cell_instead_of_running_it(client, monkeypatch):
    """The whole point: after an interruption, resubmitting the same sweep re-measures only
    what is missing."""
    import asyncio
    ran = []

    async def fake_execute(cid, app):
        ran.append(cid)
        conn = P.db()
        conn.execute("UPDATE bench_runs SET status='done' WHERE id=?", (cid,))
        conn.commit()
        conn.close()

    monkeypatch.setattr(P, "_bench_execute", fake_execute)
    monkeypatch.setattr(P, "_bench_residency_snapshot",
                        lambda: asyncio.sleep(0, result={"backends": []}))

    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    # qwen3:4b was measured before; llama3 was not.
    cell_cfg = dict(_CFG, cache="cold", resume=True, randomize=True, warmup=True,
                    temperature=None, server_context=None)
    _done_cell(conn, "b_prev", "qwen3:4b", cell_cfg, results='{"summary":{"n_success":36}}')
    parent_cfg = dict(cell_cfg)
    parent_cfg["models"] = [{"model": "qwen3:4b", "upstream": "ollama"},
                            {"model": "llama3", "upstream": "ollama"}]
    conn.execute(
        "INSERT INTO bench_runs (id, ts, model, config_json, status) "
        "VALUES ('b_new', ?, 'qwen3:4b', ?, 'pending')", (time.time(), json.dumps(parent_cfg)))
    conn.commit()
    conn.close()

    asyncio.run(P._bench_execute_suite("b_new", P.app))

    conn = P.db()
    kids = conn.execute("SELECT model, status, env_json FROM bench_runs WHERE parent_id='b_new' "
                        "ORDER BY model").fetchall()
    conn.close()
    got = {k["model"]: k["status"] for k in kids}
    assert got == {"llama3": "done", "qwen3:4b": "done"}, got
    assert len(ran) == 1, f"re-ran a cell it already had: {ran}"

    # The reused one records where its numbers came from.
    env = json.loads(next(k["env_json"] for k in kids if k["model"] == "qwen3:4b") or "{}")
    assert env.get("reused_from") == "b_prev"


def test_without_resume_every_cell_runs(client, monkeypatch):
    import asyncio
    ran = []

    async def fake_execute(cid, app):
        ran.append(cid)
        conn = P.db()
        conn.execute("UPDATE bench_runs SET status='done' WHERE id=?", (cid,))
        conn.commit()
        conn.close()

    monkeypatch.setattr(P, "_bench_execute", fake_execute)
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    cell_cfg = dict(_CFG, resume=False)
    _done_cell(conn, "b_prev", "qwen3:4b", cell_cfg)
    parent_cfg = dict(cell_cfg)
    parent_cfg["models"] = [{"model": "qwen3:4b", "upstream": "ollama"}]
    conn.execute("INSERT INTO bench_runs (id, ts, model, config_json, status) "
                 "VALUES ('b_new2', ?, 'qwen3:4b', ?, 'pending')",
                 (time.time(), json.dumps(parent_cfg)))
    conn.commit()
    conn.close()
    asyncio.run(P._bench_execute_suite("b_new2", P.app))
    assert len(ran) == 1, "reuse happened without being asked for"


# ---- phases -------------------------------------------------------------------------------

def test_a_phase_is_readable_while_the_step_is_still_running(client):
    """The whole point is the minutes in between. A phase that only lands when the step
    finishes describes the past."""
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.execute("INSERT INTO bench_runs (id, ts, model, config_json, status) "
                 "VALUES ('b_ph', ?, 'm', '{}', 'running')", (time.time(),))
    conn.commit()
    # Deliberately left open: the writer must not depend on this caller committing.
    P._bench_phase("b_ph", "starting vllm and waiting for it to serve")
    got = conn.execute("SELECT phase FROM bench_runs WHERE id='b_ph'").fetchone()[0]
    conn.close()
    assert got == "starting vllm and waiting for it to serve"


def test_the_phase_reaches_the_history_row(client):
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.execute("INSERT INTO bench_runs (id, ts, model, config_json, status, phase) "
                 "VALUES ('b_ph2', ?, 'm', '{}', 'running', 'freeing memory for vllm')",
                 (time.time(),))
    conn.commit()
    conn.close()
    it = client.get("/__proxy/api/bench/runs").json()["items"][0]
    assert it["phase"] == "freeing memory for vllm"


def test_a_sweeps_running_cell_carries_its_phase(client):
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    now = time.time()
    conn.execute("INSERT INTO bench_runs (id, ts, model, config_json, status) "
                 "VALUES ('b_pp', ?, 'm', '{}', 'running')", (now,))
    conn.execute("INSERT INTO bench_runs (id, ts, model, config_json, status, parent_id, "
                 "label, progress, progress_total, phase) "
                 "VALUES ('b_pc', ?, 'm', '{}', 'running', 'b_pp', 'cell', 0, 36, ?)",
                 (now, "restarting llamacpp at a 291,840-token window and reloading its weights"))
    conn.commit()
    conn.close()
    now_cell = client.get("/__proxy/api/bench/runs").json()["items"][0]["cells"]["now"]
    assert "reloading its weights" in now_cell["phase"]


def test_a_finished_run_stops_describing_a_step(client):
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.execute("INSERT INTO bench_runs (id, ts, model, config_json, status, phase) "
                 "VALUES ('b_ph3', ?, 'm', '{}', 'running', 'loading')", (time.time(),))
    conn.commit()
    conn.close()
    P._bench_phase("b_ph3", None)
    conn = P.db()
    got = conn.execute("SELECT phase FROM bench_runs WHERE id='b_ph3'").fetchone()[0]
    conn.close()
    assert got is None


# ---- eviction thrash ----------------------------------------------------------------------

def test_a_sweep_cell_does_not_reload_what_the_next_cell_will_evict(client):
    """Fifteen Ollama models, one cell each. Reloading per cell means cell 1 evicts fourteen,
    measures, loads all fourteen back — so cell 2 can evict them again. Minutes of thrash per
    cell, and they cannot all be resident anyway."""
    import inspect
    src = inspect.getsource(P._bench_execute)
    assert 'if evicted and row["parent_id"] is None:' in src, \
        "cells reload Ollama models the next cell is about to evict"


def test_a_sweep_always_records_what_was_resident(client):
    """Every cell evicts now, so only a snapshot taken before the first one knows what to put
    back — whether or not the sweep also switches backends."""
    import inspect
    src = inspect.getsource(P._bench_execute_suite)
    i = src.index("orig_residency = await _bench_residency_snapshot()")
    # No condition between the phase setup and the snapshot: it is taken unconditionally.
    assert "if start_meta is not None or len(" not in src[:i], \
        "the snapshot is still conditional on a backend switch"


def test_the_model_load_itself_reports_a_phase(client):
    """The one long step that had none. For Ollama the warm-up is where a 36 GB model is pulled
    into memory, and the warm-up is deliberately not counted — so the unit counter sits at 0 for
    minutes and the run looks wedged exactly when it is doing the most work."""
    import inspect
    src = inspect.getsource(P._bench_execute)
    i_warm = src.index("if warmup:")
    i_load = src.index('into memory"')
    assert i_load > i_warm, "the load phase must be set inside the warm-up branch"
    # ...and the clear must come after it, not before.
    i_clear = src.index("_bench_phase(bench_id, None)      # measuring from here")
    assert i_clear > i_load, "the phase is cleared before the load can announce itself"


def test_the_load_phase_names_the_size(client):
    """"Loading" for four minutes is a mystery; "loading (36 GB)" is an explanation."""
    import inspect
    src = inspect.getsource(P._bench_execute)
    assert '_sz / 1024:.0f} GB' in src


def test_load_time_and_footprint_actually_reach_the_database(client, monkeypatch):
    """They were captured into the env dict and never persisted — and because the report only
    renders Load/Resident when data exists, the loss was silent: a design meant to avoid
    columns of dashes made missing data look like a layout choice. Caught when a user asked
    'are we saving the load times?' and the live run said no."""
    import asyncio

    async def fake_run_one(client_, base, model, max_tokens, prompt, seq, cfg=None,
                           capture_text=False):
        return {"seq": seq, "ttft_ms": 10.0, "ttfc_ms": 10.0, "total_ms": 50.0,
                "completion_tokens": 5, "reasoning_tokens": None, "decode_tps": 100.0,
                "error": None, "served_model": "qwen3:4b"}

    async def fake_index():
        return {"ollama:qwen3:4b": {"model": "qwen3:4b", "upstream": "ollama",
                                    "loaded": True, "size_mb": 2400}}

    async def fake_resident(model, upstream):
        return 2300.0

    monkeypatch.setattr(P, "_bench_run_one", fake_run_one)
    monkeypatch.setattr(P, "_bench_model_index", fake_index)
    monkeypatch.setattr(P, "_bench_resident_mb", fake_resident)
    monkeypatch.setattr(P, "_bench_evict_ollama", lambda keep="": asyncio.sleep(0, result=[]))
    monkeypatch.setattr(P, "_bench_residency_snapshot",
                        lambda: asyncio.sleep(0, result={"backends": []}))

    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.execute(
        "INSERT INTO bench_runs (id, ts, model, config_json, status) VALUES (?,?,?,?,?)",
        ("b_load", time.time(), "qwen3:4b",
         json.dumps({"upstream": "ollama", "runs": 1, "warmup": True, "resume": False}),
         "pending"))
    conn.commit()
    conn.close()

    asyncio.run(P._bench_execute("b_load", P.app))

    conn = P.db()
    env = json.loads(conn.execute("SELECT env_json FROM bench_runs WHERE id='b_load'")
                     .fetchone()[0] or "{}")
    conn.close()
    assert env.get("load_ms") is not None, "load time never reached the database"
    assert env.get("resident_mb") == 2300.0, "resident footprint never reached the database"
