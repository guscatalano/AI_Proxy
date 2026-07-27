"""Cold request bodies move to a second file, and still read back.

Bodies are ~340 KB each arriving at ~3 GB/day, against a ~1.4 KB row. Nothing analytical reads
one any more, so they are archival weight in a file that has to be backed up and vacuumed. The
point of the design is that the requests_v view spans both files: a moved body reads back
through exactly the same query, so no caller had to change.
"""
import json
import time

from ai_proxy import proxy as P


def _req(rid, ts, body="x" * 500):
    conn = P.db()
    conn.execute("INSERT OR REPLACE INTO requests (id, ts, method, path, upstream_url, model) "
                 "VALUES (?, ?, 'POST', '/v1/chat/completions', 'http://x', 'm')", (rid, ts))
    conn.execute("INSERT OR REPLACE INTO request_blobs (id, request_body, response_body) "
                 "VALUES (?, ?, ?)", (rid, body, json.dumps({"ok": rid})))
    conn.commit()
    conn.close()


def _wipe():
    # Force the attach so the archive is reachable regardless of what the last test left the
    # active flag set to.
    P._ensure_archive_file()
    P._ARCHIVE_ACTIVE = True
    conn = P.db()
    conn.execute("DELETE FROM requests")
    conn.execute("DELETE FROM request_blobs")
    try:
        conn.execute("DELETE FROM arch.request_blobs")
    except Exception:
        pass
    conn.commit()
    conn.close()


def _where(rid):
    """Which file currently holds this body."""
    conn = P.db()
    try:
        m = conn.execute("SELECT 1 FROM main.request_blobs WHERE id=?", (rid,)).fetchone()
        a = conn.execute("SELECT 1 FROM arch.request_blobs WHERE id=?", (rid,)).fetchone()
        return ("main" if m else "") + ("arch" if a else "")
    finally:
        conn.close()


def _view(rid):
    conn = P.db()
    try:
        return conn.execute(
            "SELECT request_body, response_body FROM requests_v WHERE id=?", (rid,)).fetchone()
    finally:
        conn.close()


def test_the_view_spans_both_files(client):
    _wipe()
    old, new = time.time() - 30 * 86400, time.time()
    _req("a-old", old, body="OLD BODY")
    _req("a-new", new, body="NEW BODY")
    P._archive_sweep(force=True)

    assert _where("a-old") == "arch"      # moved
    assert _where("a-new") == "main"      # still hot
    # The whole point: identical query, same answer, wherever the bytes live.
    assert _view("a-old")["request_body"] == "OLD BODY"
    assert _view("a-new")["request_body"] == "NEW BODY"
    assert json.loads(_view("a-old")["response_body"]) == {"ok": "a-old"}


def test_sweep_respects_the_age_window(client, monkeypatch):
    _wipe()
    cfg = dict(P.load_rules_config())
    cfg["archive"] = {"enabled": True, "after_days": 7, "chunk": 100, "pause_s": 0}
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    _req("w-8d", time.time() - 8 * 86400)
    _req("w-6d", time.time() - 6 * 86400)
    P._archive_sweep()
    assert _where("w-8d") == "arch"
    assert _where("w-6d") == "main"


def test_disabled_does_nothing_unless_forced(client, monkeypatch):
    _wipe()
    cfg = dict(P.load_rules_config())
    cfg["archive"] = {"enabled": False, "after_days": 1, "chunk": 100, "pause_s": 0}
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    _req("d-1", time.time() - 30 * 86400)
    assert P._archive_sweep()["ok"] is False
    assert _where("d-1") == "main"
    assert P._archive_sweep(force=True)["ok"] is True
    assert _where("d-1") == "arch"


def test_sweep_is_chunked_and_moves_everything(client, monkeypatch):
    _wipe()
    cfg = dict(P.load_rules_config())
    cfg["archive"] = {"enabled": True, "after_days": 1, "chunk": 3, "pause_s": 0}
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    old = time.time() - 10 * 86400
    for i in range(10):
        _req(f"c-{i}", old + i)
    res = P._archive_sweep()
    assert res["moved"] == 10
    assert all(_where(f"c-{i}") == "arch" for i in range(10))


def test_sweep_is_idempotent(client, monkeypatch):
    _wipe()
    cfg = dict(P.load_rules_config())
    cfg["archive"] = {"enabled": True, "after_days": 1, "chunk": 50, "pause_s": 0}
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    _req("i-1", time.time() - 10 * 86400)
    assert P._archive_sweep()["moved"] == 1
    assert P._archive_sweep()["moved"] == 0


def test_a_body_is_never_in_neither_file(client):
    # Insert and delete share one transaction, so a failure can duplicate but never drop.
    _wipe()
    _req("t-1", time.time() - 30 * 86400)
    P._archive_sweep(force=True)
    assert _where("t-1") in ("arch", "mainarch")
    assert _view("t-1")["request_body"] is not None


def test_status_reports_what_would_move(client, monkeypatch):
    _wipe()
    cfg = dict(P.load_rules_config())
    cfg["archive"] = {"enabled": True, "after_days": 7, "chunk": 100, "pause_s": 0}
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    _req("st-old", time.time() - 30 * 86400, body="y" * 1000)
    _req("st-new", time.time())
    s = P._archive_status()
    assert s["pending_rows"] == 1
    assert s["pending_bytes"] > 1000
    assert s["archived_rows"] == 0
    P._archive_sweep()
    s2 = P._archive_status()
    assert s2["pending_rows"] == 0 and s2["archived_rows"] == 1


def test_wiping_requests_also_clears_the_archive(client):
    # Otherwise "wipe requests" leaves gigabytes of bodies behind in the second file.
    _wipe()
    _req("w-1", time.time() - 30 * 86400)
    P._archive_sweep(force=True)
    assert _where("w-1") == "arch"
    client.post("/__proxy/api/db/reset", json={"targets": ["requests"], "vacuum": False})
    conn = P.db()
    assert conn.execute("SELECT COUNT(*) FROM arch.request_blobs").fetchone()[0] == 0
    conn.close()


def test_endpoints(client):
    _wipe()
    _req("e-1", time.time() - 30 * 86400)
    s = client.get("/__proxy/api/archive").json()
    assert "pending_rows" in s and "main_db_bytes" in s and "archive_db_bytes" in s

    r = client.post("/__proxy/api/archive/run", json={"force": True}).json()
    assert r["ok"] is True and r["moved"] == 1
    assert r["status"]["archived_rows"] == 1
    assert _view("e-1")["request_body"] is not None


def test_sweeps_run_periodically_only_when_enabled(client, monkeypatch):
    # Without a periodic trigger the rule would only ever do anything when someone pressed a
    # button, which is not what "archive after 7 days" means.
    import ai_proxy.proxy as mod
    calls = []
    monkeypatch.setattr(mod.threading, "Thread",
                        lambda target=None, **kw: type("T", (), {"start": lambda s: calls.append(target)})())
    cfg = dict(P.load_rules_config())

    cfg["archive"] = {"enabled": False}
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    monkeypatch.setattr(P, "_last_archive_sweep", 0.0)
    P._maybe_sweep_archive(time.time())
    assert calls == []

    cfg["archive"] = {"enabled": True}
    monkeypatch.setattr(P, "_last_archive_sweep", 0.0)
    P._maybe_sweep_archive(time.time())
    assert calls == [P._archive_sweep]

    # ...and not again within the hour.
    P._maybe_sweep_archive(time.time())
    assert len(calls) == 1


def test_health_reports_the_second_file(client):
    d = client.get("/__proxy/api/health").json()["db"]
    assert "archive_bytes" in d and "archive_path" in d


def test_connections_skip_the_attach_until_there_is_something_to_read(client, monkeypatch):
    # Attaching and building the spanning view costs ~1.3ms per connection, and db() runs on
    # every save and poll. Nobody who has never archived should pay for it.
    _wipe()
    cfg = dict(P.load_rules_config())
    cfg["archive"] = {"enabled": False, "after_days": 7, "chunk": 50, "pause_s": 0}
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    monkeypatch.setattr(P, "_ARCHIVE_ACTIVE", False)
    conn = P.db()
    try:
        assert "arch" not in [r[1] for r in conn.execute("PRAGMA database_list")]
        conn.execute("SELECT request_body FROM requests_v LIMIT 1")   # still works, main-only
    finally:
        conn.close()

    # ...and starts paying the moment a sweep puts something there.
    _req("lz-1", time.time() - 30 * 86400)
    P._archive_sweep(force=True)
    assert P._ARCHIVE_ACTIVE is True
    conn = P.db()
    try:
        assert "arch" in [r[1] for r in conn.execute("PRAGMA database_list")]
    finally:
        conn.close()


def test_active_flag_survives_a_restart(client, monkeypatch):
    _wipe()
    cfg = dict(P.load_rules_config())
    cfg["archive"] = {"enabled": False, "after_days": 7, "chunk": 50, "pause_s": 0}
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    _req("rs-1", time.time() - 30 * 86400)
    P._archive_sweep(force=True)
    monkeypatch.setattr(P, "_ARCHIVE_ACTIVE", False)      # as if freshly started
    assert P._refresh_archive_active() is True            # rows exist, so attach again
    assert _view("rs-1")["request_body"] is not None
