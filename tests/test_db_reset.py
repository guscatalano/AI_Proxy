"""The database wipe: one at a time, reports progress, and never on the event loop.

Deleting rows and rewriting a multi-gigabyte file takes minutes. Two of those at once contend
for the same write lock, and the whole job used to run inside an `async def`, which stopped the
proxy serving anything at all for the duration.
"""
import threading
import time

from ai_proxy import proxy as P


def _seed(client, n=5):
    conn = P.db()
    # Other modules share this database; start from a known count or the deleted totals below
    # are whatever ran before this file did.
    conn.execute("DELETE FROM requests")
    conn.execute("DELETE FROM request_blobs")
    conn.executemany(
        "INSERT INTO requests (id, ts, method, path, upstream_url) "
        "VALUES (?, ?, 'POST', '/v1/messages', 'http://x')",
        [(f"dr-{i}", time.time() - i) for i in range(n)])
    conn.executemany("INSERT INTO request_blobs (id, request_body) VALUES (?, ?)",
                     [(f"dr-{i}", "x" * 100) for i in range(n)])
    conn.commit()
    conn.close()


def test_reset_deletes_and_reports_counts(client):
    _seed(client, 4)
    d = client.post("/__proxy/api/db/reset",
                    json={"targets": ["requests"], "vacuum": False}).json()
    assert d["ok"] is True
    assert d["deleted"]["requests"] == 4
    conn = P.db()
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
    conn.close()


def test_reset_takes_the_blobs_with_the_requests(client):
    # Bodies live in their own table. Wiping requests alone would strand gigabytes of them with
    # nothing pointing at the rows.
    _seed(client, 3)
    client.post("/__proxy/api/db/reset", json={"targets": ["requests"], "vacuum": False})
    conn = P.db()
    assert conn.execute("SELECT COUNT(*) FROM request_blobs").fetchone()[0] == 0
    conn.close()


def test_reset_leaves_untargeted_tables_alone(client):
    _seed(client, 2)
    conn = P.db()
    conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('keep', '1')")
    conn.commit()
    conn.close()
    client.post("/__proxy/api/db/reset", json={"targets": ["requests"], "vacuum": False})
    conn = P.db()
    assert conn.execute("SELECT COUNT(*) FROM settings WHERE key='keep'").fetchone()[0] == 1
    conn.close()


def test_status_reports_a_finished_job(client):
    _seed(client, 2)
    client.post("/__proxy/api/db/reset", json={"targets": ["requests"], "vacuum": False})
    s = client.get("/__proxy/api/db/reset/status").json()
    assert s["state"] == "done"
    assert s["step"] == "finished"
    assert s["elapsed_s"] >= 0
    assert s["deleted"]["requests"] == 2
    assert s["size_after"] is not None


def test_second_wipe_is_refused_while_one_runs(client):
    # The lock is what stops a DELETE racing a VACUUM on the same file.
    P._DB_JOB_LOCK.acquire()
    try:
        P._DB_JOB.update({"state": "running", "step": "compacting the database",
                          "started": time.time(), "finished": None})
        r = client.post("/__proxy/api/db/reset", json={"targets": ["requests"]})
        assert r.status_code == 409
        body = r.json()
        assert "already running" in body["error"].lower()
        assert body["status"]["step"] == "compacting the database"
    finally:
        P._DB_JOB.update({"state": "idle", "step": "", "started": None})
        P._DB_JOB_LOCK.release()


def test_lock_is_released_after_a_failure(client):
    # A job that dies holding the lock would wedge the endpoint until the next restart.
    def boom(*_a, **_kw):
        raise sqlite_error()

    def sqlite_error():
        import sqlite3
        return sqlite3.OperationalError("disk exploded")

    orig = P.db
    P.db = boom
    try:
        d = client.post("/__proxy/api/db/reset",
                        json={"targets": ["requests"], "vacuum": False}).json()
        assert d["ok"] is False
        assert "disk exploded" in d["status"]["error"]
    finally:
        P.db = orig
    assert P._DB_JOB_LOCK.acquire(blocking=False), "lock still held after a failed job"
    P._DB_JOB_LOCK.release()
    # And the endpoint still works afterwards.
    assert client.post("/__proxy/api/db/reset",
                       json={"targets": ["metrics"], "vacuum": False}).json()["ok"] is True


def test_vacuum_runs_and_is_reported(client):
    _seed(client, 3)
    d = client.post("/__proxy/api/db/reset",
                    json={"targets": ["requests"], "vacuum": True}).json()
    assert d["vacuumed"] is True
    assert d["status"]["reclaimed_bytes"] is not None


def test_vacuum_compacts_the_archive_too(client):
    # Emptying the rows leaves the archive's pages on its own free list. Compacting only main
    # would report space reclaimed while gigabytes stayed on disk in the second file.
    #
    # Deliberately leaves _ARCHIVE_ACTIVE False before the wipe: the job has to reach the
    # archive on its own, or "wipe requests" silently strands every archived body.
    P._ensure_archive_file()
    P._ARCHIVE_ACTIVE = True
    conn = P.db()
    conn.execute("DELETE FROM requests")
    conn.execute("DELETE FROM arch.request_blobs")
    conn.executemany("INSERT INTO arch.request_blobs (id, request_body) VALUES (?, ?)",
                     [(f"v-{i}", "x" * 20000) for i in range(200)])
    conn.commit()
    conn.close()
    grown = P.Path(P.ARCHIVE_DB_PATH).stat().st_size
    assert grown > 1_000_000, "archive did not actually grow; test proves nothing"

    P._ARCHIVE_ACTIVE = False        # as if nothing had been archived this run
    d = client.post("/__proxy/api/db/reset",
                    json={"targets": ["requests"], "vacuum": True}).json()
    assert d["status"].get("archive_error") is None
    shrunk = P.Path(P.ARCHIVE_DB_PATH).stat().st_size
    assert shrunk < grown / 2, f"archive not compacted: {grown} -> {shrunk}"


def test_reported_size_covers_both_files(client):
    P._ensure_archive_file()
    assert P._db_size() >= P.Path(P.DB_PATH).stat().st_size
    assert P._db_size() >= P.Path(P.ARCHIVE_DB_PATH).stat().st_size


def test_table_sizes_serve_stale_rather_than_rescan(client):
    # The System tab asks for these every load. A full-database scan on each one is what made
    # the tab lag; a stale answer now returns instantly and refreshes behind the response.
    P._DBSTAT_CACHE.update({"ts": time.time(), "value": {"requests": 4242}, "refreshing": False})
    calls = []
    orig = P._scan_table_sizes

    def counted():
        calls.append(1)
        time.sleep(0.05)
        return {"requests": 1}

    P._scan_table_sizes = counted
    try:
        assert P._table_sizes() == {"requests": 4242}      # fresh cache: no scan at all
        assert calls == []
        P._DBSTAT_CACHE["ts"] = time.time() - P._dbstat_ttl() - 1
        assert P._table_sizes() == {"requests": 4242}      # stale: still serves the old value
        for _ in range(40):                                # ...and refreshed behind it
            if calls:
                break
            time.sleep(0.02)
        assert calls, "stale cache never triggered a background refresh"
        for _ in range(40):
            if not P._DBSTAT_CACHE.get("refreshing"):
                break
            time.sleep(0.02)
        assert P._DBSTAT_CACHE["value"] == {"requests": 1}
    finally:
        P._scan_table_sizes = orig
        P._DBSTAT_CACHE.update({"ts": 0.0, "value": {}, "refreshing": False, "cost_s": 0.0})


def test_first_call_returns_immediately_rather_than_scanning(client):
    # A cold cache used to block the caller for the length of a full scan — 111 seconds on the
    # 6 GB production database, for a panel folded inside a <details>.
    P._DBSTAT_CACHE.update({"ts": 0.0, "value": {}, "refreshing": False, "cost_s": 0.0})
    started = time.time()
    orig = P._scan_table_sizes

    def slow():
        time.sleep(0.4)
        return {"requests": 7}

    P._scan_table_sizes = slow
    try:
        assert P._table_sizes() == {}                   # nothing known yet, and no waiting
        assert time.time() - started < 0.2
        for _ in range(60):
            if not P._DBSTAT_CACHE.get("refreshing"):
                break
            time.sleep(0.02)
        assert P._table_sizes() == {"requests": 7}      # arrives on the next call
    finally:
        P._scan_table_sizes = orig
        P._DBSTAT_CACHE.update({"ts": 0.0, "value": {}, "refreshing": False, "cost_s": 0.0})


def test_health_flags_a_scan_in_progress(client):
    P._DBSTAT_CACHE.update({"ts": 0.0, "value": {}, "refreshing": True, "cost_s": 0.0})
    try:
        assert client.get("/__proxy/api/health?tables=1").json()["db"]["table_sizes_pending"]
        # Without ?tables=1 nothing is measured, so nothing is pending.
        assert not client.get("/__proxy/api/health").json()["db"]["table_sizes_pending"]
    finally:
        P._DBSTAT_CACHE.update({"ts": 0.0, "value": {}, "refreshing": False, "cost_s": 0.0})


def test_scan_interval_scales_with_what_the_scan_costs(client):
    # A fixed 5-minute TTL on a 6 GB database means a 100-second scan every 5 minutes: 30% of
    # wall-clock spent walking pages nobody asked to see.
    P._DBSTAT_CACHE["cost_s"] = 0.0
    assert P._dbstat_ttl() == P._DBSTAT_TTL_S          # small database keeps the floor
    P._DBSTAT_CACHE["cost_s"] = 100.0
    assert P._dbstat_ttl() == 6000                     # 100s scan -> once per 100 minutes
    P._DBSTAT_CACHE["cost_s"] = 0.0


def test_health_does_not_block_the_event_loop(client):
    """`health` must be a sync handler so Starlette runs it in the threadpool.

    As `async def` its five aggregate queries — and the full-database dbstat scan behind
    ?tables=1 — ran on the event loop, so opening the System tab stalled request proxying.
    """
    import inspect
    for fn in (P.health, P.system_now, P.system_history):
        assert not inspect.iscoroutinefunction(fn), f"{fn.__name__} would block the event loop"


def test_concurrent_posts_never_both_run(client):
    _seed(client, 2)
    seen = []
    orig = P._db_reset_job

    def slow(targets, vacuum):
        seen.append(P._DB_JOB_LOCK.locked())
        time.sleep(0.3)
        return orig(targets, vacuum)

    P._db_reset_job = slow
    results = []
    try:
        def post():
            r = client.post("/__proxy/api/db/reset",
                            json={"targets": ["requests"], "vacuum": False})
            results.append(r.status_code)

        threads = [threading.Thread(target=post) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
    finally:
        P._db_reset_job = orig
    assert sorted(results) == [200, 409], f"expected exactly one to run, got {results}"
