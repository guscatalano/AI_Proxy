#!/usr/bin/env python3
"""Backfill the requests.surface column from stored request bodies, using the current
_detect_surface fingerprints. The forward path labels new traffic at ingest; this covers
history. Like relabel_clients.py, this is a one-off script rather than a startup migration:
request bodies live in the request_blobs side table, which isn't cleanly queryable early in
init_db, and a failing backfill there crash-loops startup.

Usage:  python -m scripts.backfill_surface [/path/to/proxy.db]
        (defaults to $PROXY_DB, else ./proxy.db). Run from the install dir with its venv so
        the proxy module is importable.
"""
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.getcwd())
from ai_proxy.proxy import _detect_surface  # noqa: E402


def main() -> None:
    db = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PROXY_DB", "proxy.db")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT r.id AS id, r.request_headers AS h, b.request_body AS body "
        "FROM requests r JOIN request_blobs b ON b.id = r.id "
        "WHERE r.surface IS NULL AND r.shadow_of IS NULL").fetchall()
    changed: dict[str, int] = {}
    for r in rows:
        try:
            hdr = json.loads(r["h"]) if r["h"] else None
        except (TypeError, ValueError):
            hdr = None
        try:
            body = json.loads(r["body"]) if r["body"] else None
        except (TypeError, ValueError):
            body = None
        surface = _detect_surface(hdr, body if isinstance(body, dict) else None)
        if surface:
            conn.execute("UPDATE requests SET surface=? WHERE id=?", (surface, r["id"]))
            changed[surface] = changed.get(surface, 0) + 1
    conn.commit()
    conn.close()
    print(f"scanned {len(rows)} unlabeled rows; tagged: {dict(changed) or 'none'}")


if __name__ == "__main__":
    main()
