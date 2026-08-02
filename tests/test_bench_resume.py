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


def test_reuse_is_off_unless_asked_for(client):
    """A benchmark that quietly hands back old numbers is worse than a slow one."""
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
