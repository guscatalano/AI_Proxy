#!/usr/bin/env python3
"""Remove credentials already written to the request log, and backfill their fingerprints.

Until the redaction landed, every request stored its complete header set verbatim — bearer
tokens and api keys included. On a box whose backends accept a placeholder that was harmless;
the moment a real cloud key is routed through, the same code path writes it into a file that
gets backed up and copied. This rewrites history: each stored credential becomes '[redacted]'
and its fingerprint (sha256 prefix + last 4) goes into api_key_fp, so nothing that could
identify a caller is lost and nothing that could authenticate as one remains.

Idempotent, and safe to run against a live proxy: rows are rewritten in small committed
batches so the write lock is never held for long.

VACUUM is opt-in and deliberately not the default. Rewriting a row leaves the old bytes in
the database's free pages, so a real secret is not gone until the file is rebuilt — but this
database is 7.6 GB on the box it was written for, and rebuilding it takes an exclusive lock
for minutes plus its own size again in free disk. That is a maintenance window, not a gap
between requests. Scrub now, VACUUM when you can afford it:

    python -m scripts.scrub_credentials [/path/to/proxy.db]           # safe any time
    python -m scripts.scrub_credentials [/path/to/proxy.db] --vacuum  # needs a window
"""
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.getcwd())
from ai_proxy.proxy import (_CREDENTIAL_HEADERS, _CRED_REDACTED,  # noqa: E402
                            _credential_fingerprint, _redact_credential_headers)


BATCH = 500


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_vacuum = "--vacuum" in sys.argv
    db = args[0] if args else os.environ.get("PROXY_DB", "proxy.db")
    conn = sqlite3.connect(db, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("ALTER TABLE requests ADD COLUMN api_key_fp TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass          # already migrated by the proxy
    rows = conn.execute(
        "SELECT id, request_headers, api_key_fp FROM requests "
        "WHERE request_headers IS NOT NULL").fetchall()
    scrubbed = fingerprinted = 0
    pending = []
    for r in rows:
        try:
            h = json.loads(r["request_headers"] or "{}")
        except (TypeError, ValueError):
            continue
        if not isinstance(h, dict):
            continue
        if not any(k.lower() in _CREDENTIAL_HEADERS and h[k] != _CRED_REDACTED for k in h):
            continue
        fp = r["api_key_fp"] or _credential_fingerprint(h)
        pending.append((json.dumps(_redact_credential_headers(h)), fp, r["id"]))
        scrubbed += 1
        fingerprinted += 1 if fp else 0
        if len(pending) >= BATCH:
            # Committed in batches so a live proxy is never blocked on one long write.
            conn.executemany(
                "UPDATE requests SET request_headers=?, api_key_fp=? WHERE id=?", pending)
            conn.commit()
            pending.clear()
    if pending:
        conn.executemany(
            "UPDATE requests SET request_headers=?, api_key_fp=? WHERE id=?", pending)
        conn.commit()
    print(f"scanned {len(rows)} rows; scrubbed {scrubbed}; "
          f"fingerprints recorded for {fingerprinted}")
    if do_vacuum:
        size_gb = os.path.getsize(db) / 1e9
        print(f"VACUUM on {size_gb:.1f} GB — exclusive lock, needs that much free disk again")
        conn.execute("VACUUM")
        print("VACUUM done; the old values are gone from the free pages")
    elif scrubbed:
        print("the scrubbed values remain in free pages until a VACUUM — rerun with "
              "--vacuum during a maintenance window if any of them were real")
    conn.close()


if __name__ == "__main__":
    main()
