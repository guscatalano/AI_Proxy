"""Seconds since the last real client request through the proxy.

The bench's own traffic is excluded — during a run it is the only client, and counting it
would make the box look permanently busy. Requests the proxy makes to itself for health
probes never land in `requests`, so nothing else needs filtering.
"""
import sqlite3
import time

db = sqlite3.connect("/home/crimson/ai_proxy/proxy.db")
row = db.execute(
    "SELECT COALESCE(MAX(ts), 0) FROM requests "
    "WHERE COALESCE(client_app, '') != 'ai-proxy-bench'").fetchone()
last = row[0] or 0
idle = int(time.time() - last) if last else 999999
busy = db.execute(
    "SELECT COUNT(*) FROM bench_runs WHERE status IN ('pending','running')").fetchone()[0]
print(f"{idle} {busy}")
