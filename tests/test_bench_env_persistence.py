"""What a cell's env_json must contain by the time anyone reads it.

Two regressions live here. A suite cell (child of a sweep) must persist load_ms exactly like a
plain cell does. And a graded cell on a machine missing a toolchain must record the skipped
languages rather than crash: the recording originally ran ten lines before `env` existed, so it
raised UnboundLocalError on precisely the machines the portability feature was built for —
and passed silently everywhere the suite ran complete.
"""
import asyncio
import json
import time

from ai_proxy import proxy as P


def _fakes(monkeypatch):
    # _BENCH_SEM is module-global and binds to the first event loop that acquires it; a fresh
    # one per test keeps these independent of whichever test file ran an execute before us.
    monkeypatch.setattr(P, "_BENCH_SEM", asyncio.Semaphore(1))

    async def fake_run_one(client_, base, model, max_tokens, prompt, seq, cfg=None,
                           capture_text=False):
        return {"seq": seq, "ttft_ms": 10.0, "ttfc_ms": 10.0, "total_ms": 50.0,
                "completion_tokens": 5, "reasoning_tokens": None, "decode_tps": 100.0,
                "error": None, "served_model": "qwen3:4b",
                "text": "def f():\n    return 1"}

    async def fake_index():
        return {"ollama:qwen3:4b": {"model": "qwen3:4b", "upstream": "ollama",
                                    "loaded": True, "size_mb": 2400}}

    async def fake_resident(model, upstream):
        return 2300.0

    async def fake_grade(text, task, timeout):
        return {"passed": 0, "total": 1}

    monkeypatch.setattr(P, "_bench_run_one", fake_run_one)
    monkeypatch.setattr(P, "_bench_model_index", fake_index)
    monkeypatch.setattr(P, "_bench_resident_mb", fake_resident)
    monkeypatch.setattr(P, "_bench_grade", fake_grade)
    monkeypatch.setattr(P, "_bench_evict_ollama", lambda keep="": asyncio.sleep(0, result=[]))
    monkeypatch.setattr(P, "_bench_residency_snapshot",
                        lambda: asyncio.sleep(0, result={"backends": []}))
    monkeypatch.setattr(P, "_bench_ollama_kv_preflight",
                        lambda model, meta, cfg: asyncio.sleep(0, result=None))


def _seed(cfg):
    conn = P.db()
    conn.execute("DELETE FROM bench_runs")
    conn.execute(
        "INSERT INTO bench_runs (id, ts, model, config_json, status, parent_id) "
        "VALUES (?,?,?,?,?,?)",
        ("b_env", time.time(), "qwen3:4b", json.dumps(cfg), "pending", "b_parent"))
    conn.commit()
    conn.close()


def _row():
    conn = P.db()
    st, err, envj = conn.execute(
        "SELECT status, error, env_json FROM bench_runs WHERE id='b_env'").fetchone()
    conn.close()
    return st, err, json.loads(envj or "{}")


def test_suite_child_persists_load_ms(client, monkeypatch):
    _fakes(monkeypatch)
    _seed({"upstream": "ollama", "runs": 1, "warmup": True, "resume": False,
           "suite": "coding-v1", "cache": "cached", "exclusive": False})
    asyncio.run(P._bench_execute("b_env", P.app))
    st, err, env = _row()
    assert st == "done", err
    assert env.get("load_ms") is not None, "suite cell lost load_ms"
    assert env.get("resident_mb") == 2300.0


def test_missing_toolchains_are_recorded_not_fatal(client, monkeypatch):
    """coding-v2 on a box with no compilers: the cell must run its Python/HTML/CSS tasks and
    write the skipped languages into env for the report's method section."""
    _fakes(monkeypatch)
    from ai_proxy import bench_graders as G
    monkeypatch.setattr(G, "_bench_tool", lambda name: None)
    _seed({"upstream": "ollama", "runs": 1, "warmup": True, "resume": False,
           "suite": "coding-v2", "cache": "cached", "exclusive": False})
    asyncio.run(P._bench_execute("b_env", P.app))
    st, err, env = _row()
    assert st == "done", err
    skipped = env.get("skipped_languages") or {}
    assert set(skipped) == {"js", "c", "cpp", "rust", "csharp", "php"}, skipped
    assert env.get("load_ms") is not None
