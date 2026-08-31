"""Disk accounting and the shape of the database reset.

Clearing a 35 GB proxy exposed four gaps at once: the System tab reported 7.7 GB because it
only ever showed the main file; the reset ran one giant transaction that locked out live
traffic (metrics collection failed ten times over and a real /v1/chat/completions returned
500); the status had no progress inside a step, so seven minutes of "deleting request bodies"
was indistinguishable from a hang; and nothing warned any of this would happen.
"""
import sqlite3

import ai_proxy
from ai_proxy import proxy


def test_disk_listing_counts_the_write_ahead_logs_not_just_the_database(client):
    """The WAL is routinely larger than the database it belongs to during a delete, and it
    was invisible: 'reclaimed 17.6 GB' against 33 GB actually freed.

    Takes `client` because the database file only exists once the app's lifespan has run."""
    d = proxy._db_files()
    labels = {f["label"] for f in d["files"]}
    assert "database" in labels
    assert d["total_bytes"] >= 0
    assert d["total_bytes"] == sum(f["bytes"] for f in d["files"])


def test_disk_listing_skips_files_that_do_not_exist():
    """Most proxies have no archive and no pre-migration backup; absent is normal, not an
    error, and must not appear as a zero-byte row."""
    for f in proxy._db_files()["files"]:
        assert f["bytes"] >= 0
        assert f["path"]


def test_health_reports_every_file(client):
    body = client.get("/__proxy/api/health").json()
    disk = body["db"]["disk"]
    assert "files" in disk and "total_bytes" in disk
    assert any(f["label"] == "database" for f in disk["files"])


def _rows(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _fixture_db(tmp_path, n):
    path = tmp_path / "chunk.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (id TEXT PRIMARY KEY, body TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(str(i), "x" * 20) for i in range(n)])
    conn.commit()
    return conn


def test_chunked_delete_removes_everything(tmp_path):
    conn = _fixture_db(tmp_path, 250)
    removed = proxy._chunked_delete(conn, "t", chunk=40, pause_s=0, on_progress=lambda n: None)
    assert removed == 250
    assert _rows(conn, "t") == 0
    conn.close()


def test_chunked_delete_commits_between_batches(tmp_path):
    """The whole point: a second connection must see progress while the delete is still
    running. One `DELETE FROM requests` holds the write lock for the entire operation, which
    is what returned 500s to live traffic."""
    conn = _fixture_db(tmp_path, 200)
    observer = sqlite3.connect(str(tmp_path / "chunk.db"), timeout=5)
    seen = []

    def watch(done):
        seen.append(_rows(observer, "t"))

    proxy._chunked_delete(conn, "t", chunk=50, pause_s=0, on_progress=watch)
    assert seen, "progress callback never fired"
    assert seen[0] < 200, f"first batch was not visible to another connection: {seen}"
    assert seen[-1] == 0
    observer.close()
    conn.close()


def test_chunked_delete_reports_running_totals(tmp_path):
    conn = _fixture_db(tmp_path, 100)
    seen = []
    proxy._chunked_delete(conn, "t", chunk=25, pause_s=0, on_progress=seen.append)
    assert seen == sorted(seen), "progress must be monotonic"
    assert seen[-1] == 100
    conn.close()


def test_chunked_delete_on_an_empty_table_is_a_no_op(tmp_path):
    conn = _fixture_db(tmp_path, 0)
    calls = []
    assert proxy._chunked_delete(conn, "t", 50, 0, calls.append) == 0
    assert calls == [], "an empty table should not report progress"
    conn.close()


def test_a_step_resets_progress_so_it_cannot_read_as_stuck_at_100():
    """Carrying the previous step's numbers into a step with no countable work is worse than
    showing none: it looks finished while it is still working."""
    proxy._DB_JOB.clear()
    proxy._DB_JOB.update({"state": "running", "started": 1.0})

    def step(name, total=None):
        proxy._DB_JOB["step"] = name
        proxy._DB_JOB["step_started"] = 1.0
        proxy._DB_JOB["progress"] = {"done": 0, "total": total}

    step("deleting requests", 500)
    proxy._DB_JOB["progress"]["done"] = 500
    step("compacting the database")
    assert proxy._DB_JOB["progress"] == {"done": 0, "total": None}
    proxy._DB_JOB.clear()
    proxy._DB_JOB.update({"state": "idle"})


def test_status_carries_disk_while_running():
    """Deletes land in the WAL first, so disk climbs during a clear. Without the number
    visible, watching it grow reads as failure."""
    proxy._DB_JOB.clear()
    proxy._DB_JOB.update({"state": "running", "started": 1.0, "step": "deleting requests"})
    s = proxy._db_job_status()
    assert "disk" in s and "total_bytes" in s["disk"]
    proxy._DB_JOB.clear()
    proxy._DB_JOB.update({"state": "idle"})


def test_status_is_quiet_when_idle():
    proxy._DB_JOB.clear()
    proxy._DB_JOB.update({"state": "idle"})
    assert "disk" not in proxy._db_job_status(), "no need to stat files when nothing is running"
