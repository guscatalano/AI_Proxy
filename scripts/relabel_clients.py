#!/usr/bin/env python3
"""Reclassify client_app for rows currently tagged with a *generic* label, using the current
_detect_client_app logic (including the system-prompt fingerprint layer). Run this after adding
new fingerprints to relabel historical rows — the forward path already labels new traffic.

This is a one-off maintenance script rather than a startup migration on purpose: request bodies
live in the request_blobs side table, which isn't cleanly queryable early in init_db, and a
failing backfill there crash-loops startup.

Usage:  python -m scripts.relabel_clients [/path/to/proxy.db]
        (defaults to $PROXY_DB, else ./proxy.db). Run from the install dir with its venv so the
        proxy module is importable.
"""
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.getcwd())
from ai_proxy.proxy import _detect_client_app  # noqa: E402

GENERIC = ("unknown", "openai-sdk", "openai-python", "openai-node",
           "anthropic-sdk", "anthropic-python", "anthropic-node", "httpx", "requests")


def main() -> None:
    db = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PROXY_DB", "proxy.db")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    ph = ",".join("?" * len(GENERIC))
    rows = conn.execute(
        "SELECT r.id AS id, r.client_app AS client_app, r.request_headers AS h, "
        "b.request_body AS body FROM requests r JOIN request_blobs b ON b.id = r.id "
        "WHERE r.client_app IN (" + ph + ")", GENERIC).fetchall()
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
        app = _detect_client_app(hdr, body if isinstance(body, dict) else None)
        if app and app != r["client_app"]:
            conn.execute("UPDATE requests SET client_app=? WHERE id=?", (app, r["id"]))
            changed[app] = changed.get(app, 0) + 1
    conn.commit()
    conn.close()
    print(f"scanned {len(rows)} generic-labeled rows; relabeled: {dict(changed) or 'none'}")


if __name__ == "__main__":
    main()
