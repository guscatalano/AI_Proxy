#!/usr/bin/env python3
"""Remove credentials already written to the request log, and backfill their fingerprints.

Until the redaction landed, every request stored its complete header set verbatim — bearer
tokens and api keys included. On a box whose backends accept a placeholder that was harmless;
the moment a real cloud key is routed through, the same code path writes it into a file that
gets backed up and copied. This rewrites history: each stored credential becomes '[redacted]'
and its fingerprint (sha256 prefix + last 4) goes into api_key_fp, so nothing that could
identify a caller is lost and nothing that could authenticate as one remains.

Idempotent. Run it on the proxy host with its venv:
    python -m scripts.scrub_credentials [/path/to/proxy.db]
"""
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.getcwd())
from ai_proxy.proxy import (_CREDENTIAL_HEADERS, _CRED_REDACTED,  # noqa: E402
                            _credential_fingerprint, _redact_credential_headers)


def main() -> None:
    db = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PROXY_DB", "proxy.db")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("ALTER TABLE requests ADD COLUMN api_key_fp TEXT")
    except sqlite3.OperationalError:
        pass          # already migrated by the proxy
    rows = conn.execute(
        "SELECT id, request_headers, api_key_fp FROM requests "
        "WHERE request_headers IS NOT NULL").fetchall()
    scrubbed = fingerprinted = 0
    for r in rows:
        try:
            h = json.loads(r["request_headers"] or "{}")
        except (TypeError, ValueError):
            continue
        if not isinstance(h, dict):
            continue
        present = [k for k in h if k.lower() in _CREDENTIAL_HEADERS
                   and h[k] != _CRED_REDACTED]
        if not present:
            continue
        fp = r["api_key_fp"] or _credential_fingerprint(h)
        conn.execute("UPDATE requests SET request_headers=?, api_key_fp=? WHERE id=?",
                     (json.dumps(_redact_credential_headers(h)), fp, r["id"]))
        scrubbed += 1
        fingerprinted += 1 if fp else 0
    conn.commit()
    conn.execute("VACUUM")     # the old values live on in free pages until this runs
    conn.close()
    print(f"scanned {len(rows)} rows; scrubbed {scrubbed}; "
          f"fingerprints recorded for {fingerprinted}")


if __name__ == "__main__":
    main()
