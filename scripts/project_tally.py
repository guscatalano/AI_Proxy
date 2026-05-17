#!/usr/bin/env python3
"""One-shot: tally Claude Code traffic by project (from the cwd hint in the system prompt).
Splits local-via-bridge vs remote-to-Anthropic based on response-content sniffing."""
import sqlite3
import re
import sys
from collections import defaultdict

DB = sys.argv[1] if len(sys.argv) > 1 else "proxy.db"

# Two patterns: claude-code's own env block has "Primary working directory:" preceded by
# either a literal backslash or a JSON-escaped one. Match both.
PAT = re.compile(r"Primary working directory:\s*([^\n\"]+?)(?=\\n|\")")

# Manual aliases — fold sub-projects / variants into a single bucket.
# Example: {"MyProject.WinUI": "MyProject", "MyProject.Tests": "MyProject"}
PROJECT_ALIASES: dict = {}

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

agg: dict[str, dict] = defaultdict(lambda: {
    "reqs": 0, "prompt": 0, "completion": 0, "ms": 0,
    "first_ts": None, "last_ts": None, "local": 0, "remote": 0,
})

rows = conn.execute(
    "SELECT id, ts, request_body, stream_chunks, response_body, "
    "prompt_tokens, completion_tokens, duration_ms "
    "FROM requests WHERE client_app = 'claude-code' AND request_body IS NOT NULL"
).fetchall()

for r in rows:
    body = r["request_body"] or ""
    m = PAT.search(body)
    if not m:
        continue
    path = m.group(1).replace("\\\\", "\\").strip()
    parts = re.split(r"[\\/]", path)
    proj = parts[-1] if parts and parts[-1] else "(no project)"
    if proj.lower() == "repos":
        proj = "(no project)"
    proj = PROJECT_ALIASES.get(proj, proj)
    sc = r["stream_chunks"] or ""
    rb = r["response_body"] or ""
    is_local = ("fp_ollama" in sc) or ("fp_ollama" in rb) or ("chat.completion.chunk" in sc)
    a = agg[proj]
    a["reqs"] += 1
    a["prompt"] += r["prompt_tokens"] or 0
    a["completion"] += r["completion_tokens"] or 0
    a["ms"] += r["duration_ms"] or 0
    if is_local:
        a["local"] += 1
    else:
        a["remote"] += 1
    if a["first_ts"] is None or r["ts"] < a["first_ts"]:
        a["first_ts"] = r["ts"]
    if a["last_ts"] is None or r["ts"] > a["last_ts"]:
        a["last_ts"] = r["ts"]

items = sorted(agg.items(), key=lambda kv: -kv[1]["reqs"])

print(f"{'project':<32} {'reqs':>6} {'local':>6} {'remote':>7} {'prompt_tok':>14} {'compl_tok':>11} {'min':>9}")
print("-" * 95)
for proj, a in items[:30]:
    print(
        f"{proj[:32]:<32} {a['reqs']:>6} {a['local']:>6} {a['remote']:>7} "
        f"{a['prompt']:>14,} {a['completion']:>11,} {a['ms']/60000:>9.1f}"
    )
conn.close()
