"""Tool names are extracted once, at write time, instead of re-read from bodies on every call.

Counting them used to mean pulling 500 response bodies out of the blob table per stats call:
19-27 MB of scattered reads, measured at 5.1s against cold pages where the same rows without
bodies took 0.001s. That was the entire cost of the endpoint.
"""
import json
import time

from ai_proxy import proxy as P


def _resp(*names):
    return json.dumps({"choices": [{"message": {"tool_calls": [
        {"type": "function", "function": {"name": n, "arguments": "{}"}} for n in names]}}]})


def _pending(rid, ts=None, model="m", body=None, stream=None):
    """A row as it exists before _save_finish, with its blobs already stored."""
    conn = P.db()
    conn.execute("INSERT OR REPLACE INTO requests (id, ts, method, path, upstream_url, model) "
                 "VALUES (?, ?, 'POST', '/v1/chat/completions', 'http://x', ?)",
                 (rid, ts if ts is not None else time.time(), model))
    conn.execute("INSERT OR REPLACE INTO request_blobs (id, response_body, stream_chunks) "
                 "VALUES (?, ?, ?)", (rid, body, stream))
    conn.commit()
    conn.close()


def _col(rid):
    conn = P.db()
    try:
        return conn.execute("SELECT tool_calls FROM requests WHERE id=?", (rid,)).fetchone()[0]
    finally:
        conn.close()


def test_save_finish_records_the_tool_names(client):
    _pending("tc-1")
    P._save_finish("tc-1", 200, {}, _resp("terminal", "read_file"), None, 10.0, None)
    assert json.loads(_col("tc-1")) == ["terminal", "read_file"]


def test_no_tool_calls_stores_nothing(client):
    _pending("tc-2")
    P._save_finish("tc-2", 200, {}, json.dumps({"choices": [{"message": {"content": "hi"}}]}),
                   None, 10.0, None)
    assert _col("tc-2") is None


def test_streamed_calls_are_recorded_too(client):
    chunks = "\n".join([
        "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "run", "arguments": ""}}]}}]}),
        "data: [DONE]",
    ])
    _pending("tc-3")
    P._save_finish("tc-3", 200, {}, None, chunks, 10.0, None)
    assert json.loads(_col("tc-3")) == ["run"]


def test_a_malformed_body_does_not_break_the_save(client):
    _pending("tc-4")
    P._save_finish("tc-4", 200, {}, "{not json", None, 10.0, None)
    assert _col("tc-4") is None
    conn = P.db()
    assert conn.execute("SELECT status FROM requests WHERE id='tc-4'").fetchone()[0] == 200
    conn.close()


def test_stats_counts_from_the_column_without_touching_bodies(client, monkeypatch):
    conn = P.db()
    conn.execute("DELETE FROM requests")
    conn.execute("DELETE FROM request_blobs")
    conn.commit()
    conn.close()
    now = time.time()
    _pending("s-1", ts=now, model="alpha")
    _pending("s-2", ts=now, model="beta")
    conn = P.db()
    conn.executemany("UPDATE requests SET tool_calls=?, status=200 WHERE id=?",
                     [(json.dumps(["terminal", "terminal"]), "s-1"),
                      (json.dumps(["terminal", "grep"]), "s-2")])
    conn.commit()
    conn.close()

    # If stats still read bodies, this would fire.
    def boom(*a, **kw):
        raise AssertionError("stats re-read response bodies to count tools")
    monkeypatch.setattr(P, "_extract_tool_calls", boom)

    P._STATS_CACHE.update({"ts": 0.0, "data": None})   # stats has its own 10s cache
    tools = {t["name"]: t for t in P.stats()["by_tool"]}
    assert tools["terminal"]["calls"] == 3
    assert tools["grep"]["calls"] == 1
    assert tools["terminal"]["by_model"] == {"alpha": 2, "beta": 1}


def test_stats_survives_a_corrupt_column(client):
    conn = P.db()
    conn.execute("DELETE FROM requests")
    conn.commit()
    conn.close()
    _pending("s-bad", ts=time.time())
    conn = P.db()
    conn.execute("UPDATE requests SET tool_calls='not json' WHERE id='s-bad'")
    conn.commit()
    conn.close()
    P._STATS_CACHE.update({"ts": 0.0, "data": None})   # stats has its own 10s cache
    assert P.stats()["by_tool"] == []


def test_backfill_fills_historical_rows(client):
    conn = P.db()
    conn.execute("DELETE FROM requests")
    conn.execute("DELETE FROM request_blobs")
    conn.commit()
    conn.close()
    old = time.time() - 3600
    _pending("b-1", ts=old, body=_resp("terminal"))
    _pending("b-2", ts=old, body=_resp("grep", "grep"))
    _pending("b-3", ts=old, body=json.dumps({"choices": [{"message": {"content": "no tools"}}]}))

    assert P._backfill_tool_calls(chunk=2, pause_s=0) == 3
    assert json.loads(_col("b-1")) == ["terminal"]
    assert json.loads(_col("b-2")) == ["grep", "grep"]
    # Marked as checked rather than left NULL, or every pass would re-read it forever.
    assert json.loads(_col("b-3")) == []


def test_backfill_is_idempotent_and_terminates(client):
    conn = P.db()
    conn.execute("DELETE FROM requests")
    conn.execute("DELETE FROM request_blobs")
    conn.commit()
    conn.close()
    _pending("b-4", ts=time.time() - 3600, body=_resp("terminal"))
    assert P._backfill_tool_calls(chunk=10, pause_s=0) == 1
    assert P._backfill_tool_calls(chunk=10, pause_s=0) == 0     # nothing left to do


def test_backfill_ignores_rows_outside_its_window(client):
    conn = P.db()
    conn.execute("DELETE FROM requests")
    conn.execute("DELETE FROM request_blobs")
    conn.commit()
    conn.close()
    _pending("b-old", ts=time.time() - 30 * 86400, body=_resp("terminal"))   # older than 7d
    _pending("b-new", ts=time.time() + 60, body=_resp("terminal"))           # after this run
    assert P._backfill_tool_calls(chunk=10, pause_s=0) == 0
    assert _col("b-old") is None and _col("b-new") is None
