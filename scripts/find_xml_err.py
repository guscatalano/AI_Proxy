#!/usr/bin/env python3
"""Find malformed XML in a specific conversation's responses, so we can verify the
fix mode covers the real-world error shapes."""
import os
import sqlite3
import sys
import re

DB = os.environ.get("AI_PROXY_DB", "proxy.db")
CONV = sys.argv[1] if len(sys.argv) > 1 else None
if not CONV:
    print("usage: find_xml_err.py <conversation_id_prefix>", file=sys.stderr)
    sys.exit(2)

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT id, ts, request_body, response_body, stream_chunks "
    "FROM requests WHERE conversation_id LIKE ? ORDER BY ts",
    (CONV + "%",),
).fetchall()

print(f"Conversation {CONV}: {len(rows)} requests\n")

# Re-assemble assistant content from each request's stream_chunks (OpenAI SSE).
def assemble_text(stream_chunks: str) -> str:
    parts = []
    for line in (stream_chunks or "").split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            continue
        try:
            import json as _j
            j = _j.loads(data)
            for c in (j.get("choices") or []):
                d = c.get("delta") or {}
                if isinstance(d.get("content"), str):
                    parts.append(d["content"])
        except Exception:
            pass
    return "".join(parts)

# Scan for likely XML-shaped content and obvious malformations.
tag_re = re.compile(r"<\s*/?\s*([A-Za-z][\w.\-:]*)\b[^>]*?>", re.DOTALL)

flagged = 0
for r in rows:
    text = assemble_text(r["stream_chunks"] or "")
    if not text:
        continue
    tags = tag_re.findall(text)
    if not tags:
        # Also look for raw `<` in text that isn't escaped — might be tool-call-style XML
        if "<" in text and "<" in text.replace("&lt;", ""):
            # Try a regex for tool_use-style XML common in some prompts
            if re.search(r"<[a-zA-Z_]+>[^<]*</[a-zA-Z_]+>|<[a-zA-Z_]+\s+[^>]*/?\s*>", text):
                print(f"=== {r['id']}: has raw < but my detector missed ===")
                print(text[:600])
                print("---")
        continue
    # Quick stack walk to see if it balances.
    stack = []
    issues = []
    for m in tag_re.finditer(text):
        name = m.group(1).lower()
        is_close = m.group(0).lstrip().startswith("</")
        is_self = m.group(0).rstrip(">").rstrip().endswith("/")
        if is_self:
            continue
        if is_close:
            if not stack:
                issues.append(f"stray </{name}>")
            elif stack[-1] != name:
                issues.append(f"mismatched: expected </{stack[-1]}>, got </{name}>")
                if name in stack:
                    while stack and stack[-1] != name:
                        issues.append(f"unclosed <{stack.pop()}>")
                    if stack:
                        stack.pop()
            else:
                stack.pop()
        else:
            stack.append(name)
    for t in stack:
        issues.append(f"unclosed <{t}>")
    if issues:
        flagged += 1
        print(f"--- {r['id']} (tags={len(tags)}, {len(issues)} issues) ---")
        for issue in issues[:10]:
            print(f"  • {issue}")
        # Print a snippet showing some context around the first tag of a suspect kind.
        first_tag = next((t for t in tags), "")
        idx = text.find(f"<{first_tag}") if first_tag else 0
        if idx >= 0:
            print(f"  snippet (around pos {idx}): {text[max(0,idx-50):idx+250]!r}")
        print()
    if flagged >= 10:
        break

if flagged == 0:
    print("No XML issues detected in this conversation.")
