import os
import re
import sys
import json
import time
import base64
import binascii
import sqlite3
import uuid
import hashlib
import asyncio
import shutil
import subprocess
import collections
import gzip
import ipaddress
import math
import datetime
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request

from ._version import __version__
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, Response

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
# Floor below which the System tab flags an upgrade. Bump this when a release lands that
# materially changes performance (new engine, MoE routing rewrite, scheduler change).
# Override with `PROXY_OLLAMA_MIN_VERSION` env var if you disagree with my pick.
OLLAMA_RECOMMENDED_VERSION = os.environ.get("PROXY_OLLAMA_MIN_VERSION", "0.21.0").strip()
SD_URL = os.environ.get("SD_URL", "http://localhost:8188").rstrip("/")
SD_MODEL = os.environ.get("SD_MODEL", "").strip()
ANTHROPIC_URL = os.environ.get("ANTHROPIC_URL", "https://api.anthropic.com").rstrip("/")

# PII redaction: when on, the API redacts request/response bodies, headers, and other content
# fields unless the viewer's IP matches the originator's IP or shares its subnet (default /24
# for v4, /64 for v6). Loopback viewers (127.0.0.1) always see everything (admin local access).
REDACT_PII_ENABLED = os.environ.get("PROXY_REDACT_PII", "1").lower() in ("1", "true", "yes", "on")
try:
    REDACT_SUBNET_BITS_V4 = int(os.environ.get("PROXY_REDACT_SUBNET_BITS", "24") or 24)
except ValueError:
    REDACT_SUBNET_BITS_V4 = 24
REDACT_PLACEHOLDER = "[REDACTED — PII hidden: viewer IP not on same subnet as originator]"
# Admin IPs bypass PII redaction entirely (see everything, regardless of subnet). Comma-separated.
ADMIN_IPS = {ip.strip() for ip in os.environ.get("PROXY_ADMIN_IPS", "192.168.6.113").split(",") if ip.strip()}
LMSTUDIO_URL = os.environ.get("LMSTUDIO_URL", "http://localhost:1234").rstrip("/")
PROXY_HOST = os.environ.get("PROXY_HOST", "0.0.0.0")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8000"))


def _user_state_dir() -> Path:
    """Per-user, writable directory for runtime state (DB, rules, generated images).

    Used as the default when the tool is installed globally (pip/pipx/npm) and the
    install directory or CWD isn't a good place to write. Override with PROXY_STATE_DIR.
    """
    env = os.environ.get("PROXY_STATE_DIR")
    if env:
        return Path(env)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "ai-proxy"
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "ai-proxy"
    return Path.home() / ".local" / "share" / "ai-proxy"


def _resolve_state_path(env_var: str, filename: str, *legacy: object) -> str:
    """Resolve a writable state file. Precedence: explicit env var, then any existing
    legacy location (so upgrades and source/systemd installs keep their data in place),
    else the per-user state dir (created on demand)."""
    v = os.environ.get(env_var)
    if v:
        return v
    for cand in legacy:
        if cand and Path(cand).exists():
            return str(cand)
    d = _user_state_dir()
    d.mkdir(parents=True, exist_ok=True)
    return str(d / filename)


# Legacy default was "./proxy.db" (CWD). Keep honoring an existing ./proxy.db so source
# and systemd installs (which also pin PROXY_DB explicitly) are unaffected.
DB_PATH = _resolve_state_path("PROXY_DB", "proxy.db", "proxy.db")

# When frozen by PyInstaller, bundled data lives under sys._MEIPASS, not next to the
# (compiled-away) source module. The spec adds static/ at "ai_proxy/static".
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    STATIC_DIR = Path(sys._MEIPASS) / "ai_proxy" / "static"
else:
    STATIC_DIR = Path(__file__).parent / "static"
METRICS_INTERVAL_S = float(os.environ.get("PROXY_METRICS_INTERVAL", "5"))
METRICS_RETENTION_S = float(os.environ.get("PROXY_METRICS_RETENTION", str(24 * 3600)))
MCP_ALLOW_WRITE = os.environ.get("MCP_ALLOW_WRITE", "false").lower() in ("1", "true", "yes")
MCP_API_KEY = os.environ.get("MCP_API_KEY", "").strip() or None
# Performance guards (see the analytics endpoints and the request save path):
#   - MAX_STORED_BODY caps how much of each request/response body is persisted, so a few
#     huge turns can't bloat the DB and slow the analytics scans. 0 = unlimited.
#   - REQUEST_RETENTION_DAYS prunes old rows periodically. 0 = keep forever.
#   - ANALYTICS_CACHE_TTL_S memoizes the expensive stats/suggestions endpoints briefly so
#     rapid UI polling doesn't recompute them every time.
MAX_STORED_BODY = int(os.environ.get("PROXY_MAX_STORED_BODY", str(4 * 1024 * 1024)) or 0)
REQUEST_RETENTION_DAYS = float(os.environ.get("PROXY_REQUEST_RETENTION_DAYS", "30") or 0)
ANALYTICS_CACHE_TTL_S = float(os.environ.get("PROXY_ANALYTICS_CACHE_TTL", "5") or 0)
# Cap uvicorn's graceful shutdown so a lingering connection (e.g. the UI's live SSE feed
# or an in-flight stream) can't hang a `systemctl restart` forever. 0 = wait indefinitely.
GRACEFUL_SHUTDOWN_S = int(float(os.environ.get("PROXY_GRACEFUL_SHUTDOWN", "10") or 0))
_last_request_prune = 0.0
_ANALYTICS_CACHE: dict = {}


def _truncate_for_store(s):
    """Cap an oversized body before persisting. Keeps the head (usually the useful part)
    and appends a marker. Returns the input unchanged when under the cap or capping is off."""
    if not s or not MAX_STORED_BODY or len(s) <= MAX_STORED_BODY:
        return s
    return s[:MAX_STORED_BODY] + f"\n…[truncated by ai-proxy: {len(s) - MAX_STORED_BODY} of {len(s)} bytes omitted]"


def _analytics_cache_get(key):
    e = _ANALYTICS_CACHE.get(key)
    if e and ANALYTICS_CACHE_TTL_S and (time.time() - e[0]) < ANALYTICS_CACHE_TTL_S:
        return e[1]
    return None


def _analytics_cache_put(key, val):
    if ANALYTICS_CACHE_TTL_S:
        _ANALYTICS_CACHE[key] = (time.time(), val)
    return val

# Rolling window of per-request proxy overhead (ms). Used by /__proxy/api/health.
_overhead_samples: collections.deque = collections.deque(maxlen=2000)
_PROCESS_START_TS = time.time()

# Panic mode kill-switch — set from the phone PWA when something goes rogue. While on,
# every proxied request short-circuits to 503 (the proxy itself stays up so you can disable
# it again). Backed by the settings table so it survives restarts.
_PANIC_MODE: bool = False

# Tracks recent phone-PWA chat sends so /api/control/await can correlate the resulting
# upstream request and stream its response back to the phone. Layout:
#   {send_id: {ts, target_name, target_ip, prompt_hint}}
# Auto-pruned by age in control_await; bounded by the phone polling client.
_CONTROL_SENDS: dict = {}

# In-memory pending tool-call queue. Each entry waits for a phone decision.
# Layout:
#   {pending_id: {ts, source(extension name), tool_name, arguments, summary, decision, decided_ts}}
# decision: None | 'allow' | 'deny' | 'always_allow' | 'always_deny'
_PENDING_TOOLS: dict = {}

# In-flight upstream responses, keyed by req_id. Populated when client.send() returns and
# cleared in a finally block when the response is fully consumed. The cancel endpoint
# looks up the entry and calls .aclose() on it to drop the upstream connection mid-stream
# (which signals Ollama to abort generation).
#   {req_id: {ts, upstream_resp, cancel_evt}}
_INFLIGHT_REQUESTS: dict = {}


# request_dedup: when two identical streaming requests arrive close together (some clients —
# notably claude-code — fan out parallel duplicates), the second one subscribes to the first's
# stream and gets the same bytes tee'd to it. Saves the GPU doing the same work twice.
#
# Map: signature_hash → StreamFanout. Stale entries are cleaned up when their TTL expires.
import hashlib as _hashlib
_REQUEST_FANOUT: dict = {}


class StreamFanout:
    """Single-producer / multi-consumer byte fanout for tee-ing a streamed response to
    duplicate requests. The producer (the primary's response generator) calls push()
    for each emitted chunk and finish() when done. Consumers (the duplicates' response
    generators) call subscribe() to get an asyncio.Queue that receives all past chunks
    immediately + future chunks live + a final None sentinel."""

    def __init__(self, primary_id: str):
        self.primary_id = primary_id
        self.created_ts = time.time()
        self.finished_ts: float | None = None
        self.history: list[bytes] = []
        self.subscribers: list[asyncio.Queue] = []
        self.error: str | None = None
        self.done = False
        self.total_bytes = 0

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        for chunk in self.history:
            q.put_nowait(chunk)
        if self.done:
            q.put_nowait(None)
        self.subscribers.append(q)
        return q

    def push(self, chunk: bytes):
        if self.done or not chunk:
            return
        self.history.append(chunk)
        self.total_bytes += len(chunk)
        for q in self.subscribers:
            try:
                q.put_nowait(chunk)
            except Exception:
                pass

    def finish(self, error: str | None = None):
        if self.done:
            return
        self.done = True
        self.error = error
        self.finished_ts = time.time()
        for q in self.subscribers:
            try:
                q.put_nowait(None)
            except Exception:
                pass


def _dedup_signature(client_ip: str, full_path: str, body: bytes) -> str:
    """Hash that identifies a request as a duplicate. Includes client IP + path + full
    body bytes. Same client sending same bytes within the dedup window = duplicate."""
    h = _hashlib.sha256()
    h.update((client_ip or "").encode("utf-8", errors="replace"))
    h.update(b"\x00")
    h.update(full_path.encode("utf-8", errors="replace"))
    h.update(b"\x00")
    h.update(body)
    return h.hexdigest()


def _dedup_gc():
    """Drop stale fanouts. Called opportunistically on each new request."""
    cfg = (load_rules_config().get("request_dedup") or {})
    ttl = float(cfg.get("ttl_s", 60))
    now = time.time()
    stale = [k for k, fo in _REQUEST_FANOUT.items()
             if fo.done and fo.finished_ts and (now - fo.finished_ts) > ttl]
    for k in stale:
        _REQUEST_FANOUT.pop(k, None)


def _load_panic_mode() -> bool:
    global _PANIC_MODE
    s = get_setting("panic_mode")
    _PANIC_MODE = bool(s and s.get("value") == "on")
    return _PANIC_MODE


def _set_panic_mode(on: bool) -> None:
    global _PANIC_MODE
    _PANIC_MODE = bool(on)
    if _PANIC_MODE:
        set_setting("panic_mode", "on")
    else:
        delete_setting("panic_mode")

# Per-request in-flight token state. While a request is pending the DB has no usage yet,
# but for streaming responses (especially Anthropic) the upstream tells us input_tokens in
# the message_start event and updates output_tokens via message_delta as generation
# progresses. Keep a small in-memory mirror so the UI can show live counts. Cleared on
# _save_finish. Layout: {req_id: {prompt: int|None, completion: int|None, est_prompt: int}}.
_LIVE_STREAMS: dict = {}

SCHEMA_TABLE = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_ts REAL
);

CREATE TABLE IF NOT EXISTS system_metrics (
    ts REAL PRIMARY KEY,
    cpu_pct REAL,
    load_1m REAL,
    mem_total_mb INTEGER,
    mem_used_mb INTEGER,
    mem_avail_mb INTEGER,
    gpu_json TEXT,
    ollama_json TEXT,
    lmstudio_json TEXT
);

CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    upstream_url TEXT NOT NULL,
    request_headers TEXT,
    request_body TEXT,
    model TEXT,
    is_stream INTEGER DEFAULT 0,
    status INTEGER,
    response_headers TEXT,
    response_body TEXT,
    stream_chunks TEXT,
    duration_ms REAL,
    error TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    client_ip TEXT
);
-- Large request/response payloads live here, not inline in `requests`, so that list/count/
-- aggregate scans over `requests` stay fast (they never touch these multi-MB blobs). The
-- blob-split migration in init_db() moves existing data here and DROPs the old columns.
CREATE TABLE IF NOT EXISTS request_blobs (
    id TEXT PRIMARY KEY,
    request_body TEXT,
    response_body TEXT,
    stream_chunks TEXT,
    images_data TEXT
);
CREATE TABLE IF NOT EXISTS proxy_memory (
    conversation_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_ts REAL,
    updated_ts REAL,
    PRIMARY KEY (conversation_id, key)
);

CREATE TABLE IF NOT EXISTS proxy_todos (
    conversation_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | in_progress | completed
    created_ts REAL,
    updated_ts REAL,
    PRIMARY KEY (conversation_id, idx)
);

CREATE TABLE IF NOT EXISTS control_endpoints (
    name TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    token TEXT,
    kind TEXT,
    registered_ts REAL,
    last_seen_ts REAL,
    source TEXT
);

CREATE TABLE IF NOT EXISTS conversation_labels (
    conversation_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    updated_ts REAL
);

CREATE TABLE IF NOT EXISTS tool_permissions (
    -- Persistent allow/deny rules. `pattern` is the literal tool_name or "tool_name:argpfx".
    pattern TEXT PRIMARY KEY,
    decision TEXT NOT NULL,  -- 'allow' | 'deny'
    created_ts REAL
);

CREATE TABLE IF NOT EXISTS bench_runs (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    model TEXT NOT NULL,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL,                -- 'pending' | 'running' | 'done' | 'failed' | 'cancelled'
    progress INTEGER DEFAULT 0,
    progress_total INTEGER,
    results_json TEXT,
    error TEXT,
    started_ts REAL,
    finished_ts REAL,
    creator_ip TEXT
);

CREATE TABLE IF NOT EXISTS proxy_personalities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    created_ts REAL NOT NULL,
    updated_ts REAL NOT NULL,
    creator_ip TEXT
);

CREATE TABLE IF NOT EXISTS proxy_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT NOT NULL,
    mode TEXT NOT NULL,                 -- 'chat' | 'agent'
    target_endpoint TEXT,               -- agent mode: registered VS Code endpoint name
    model TEXT,                         -- chat mode: model id
    status TEXT NOT NULL,               -- 'pending' | 'running' | 'done' | 'failed' | 'cancelled'
    result TEXT,
    error TEXT,
    created_ts REAL NOT NULL,
    started_ts REAL,
    finished_ts REAL,
    schedule TEXT,                      -- cron expr, 'every Nm/Nh/Nd', or null = one-shot
    next_run_ts REAL,                   -- when scheduler should fire it next; null for completed one-shots
    parent_task_id INTEGER,             -- when a recurring task fires it spawns a one-shot child
    tool_approval_mode TEXT,            -- 'rules-only' | 'notify-phone' | 'yolo' (agent mode only)
    enabled INTEGER NOT NULL DEFAULT 1, -- recurring tasks: 0 to pause without deleting
    creator_ip TEXT                     -- caller IP at create time, for subnet-scoped visibility
);
"""

SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts DESC);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model);
CREATE INDEX IF NOT EXISTS idx_requests_client ON requests(client_ip);
CREATE INDEX IF NOT EXISTS idx_requests_verdict ON requests(gate_verdict);
CREATE INDEX IF NOT EXISTS idx_requests_conversation ON requests(conversation_id, ts);
CREATE INDEX IF NOT EXISTS idx_metrics_ts ON system_metrics(ts DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON proxy_tasks(status, created_ts);
CREATE INDEX IF NOT EXISTS idx_tasks_schedule ON proxy_tasks(schedule, enabled, next_run_ts) WHERE schedule IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tasks_parent ON proxy_tasks(parent_task_id);
"""

MIGRATIONS = [
    "ALTER TABLE requests ADD COLUMN prompt_tokens INTEGER",
    "ALTER TABLE requests ADD COLUMN completion_tokens INTEGER",
    "ALTER TABLE requests ADD COLUMN total_tokens INTEGER",
    "ALTER TABLE requests ADD COLUMN client_ip TEXT",
    "ALTER TABLE requests ADD COLUMN gate_verdict TEXT",
    "ALTER TABLE requests ADD COLUMN gate_rule TEXT",
    "ALTER TABLE requests ADD COLUMN gate_reason TEXT",
    "ALTER TABLE requests ADD COLUMN gate_details TEXT",
    "ALTER TABLE requests ADD COLUMN conversation_id TEXT",
    "ALTER TABLE requests ADD COLUMN turn_index INTEGER",
    "ALTER TABLE requests ADD COLUMN client_app TEXT",
    "ALTER TABLE requests ADD COLUMN shadow_of TEXT",
    "ALTER TABLE proxy_tasks ADD COLUMN creator_ip TEXT",
    "ALTER TABLE requests ADD COLUMN proxy_tool_log TEXT",
    "ALTER TABLE system_metrics ADD COLUMN comfyui_json TEXT",
    "ALTER TABLE requests ADD COLUMN ttft_ms REAL",
    "ALTER TABLE requests ADD COLUMN upstream TEXT",
    # Chars-based estimate of the prompt token count, stored at request time. Compared against
    # the evaluated prompt_tokens (what the upstream actually prefilled) to derive a cache
    # hit/miss verdict cheaply in the requests list — no body re-parsing needed.
    "ALTER TABLE requests ADD COLUMN est_prompt_tokens INTEGER",
    # 1 if the request carried image/non-text content, so the list can badge it without
    # re-parsing the (large) body per row.
    "ALTER TABLE requests ADD COLUMN has_images INTEGER",
    # NOTE: images_data (full-fidelity image payloads) now lives in the request_blobs side
    # table, not `requests`. Do NOT re-add it here — the blob-split migration DROPs it from
    # `requests`, and re-adding would collide with request_blobs.images_data in the requests_v
    # view. Fresh installs get it via the request_blobs CREATE TABLE.
]


def db():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    conn = db()
    conn.executescript(SCHEMA_TABLE)
    for stmt in MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.executescript(SCHEMA_INDEXES)
    conn.commit()
    # Backfills run only once per migration (sentinel stored in settings table).
    done_marker = conn.execute(
        "SELECT value FROM settings WHERE key='backfill_v1'"
    ).fetchone()
    if not done_marker:
        rows = conn.execute(
            """SELECT id, response_body, stream_chunks FROM requests
               WHERE total_tokens IS NULL AND prompt_tokens IS NULL AND completion_tokens IS NULL
                     AND (response_body IS NOT NULL OR stream_chunks IS NOT NULL)"""
        ).fetchall()
        for r in rows:
            pt, ct, tt = _extract_usage(r["response_body"], r["stream_chunks"])
            if pt is not None or ct is not None or tt is not None:
                conn.execute(
                    "UPDATE requests SET prompt_tokens=?, completion_tokens=?, total_tokens=? WHERE id=?",
                    (pt, ct, tt, r["id"]),
                )
        rows = conn.execute(
            """SELECT id, request_body FROM requests
               WHERE conversation_id IS NULL AND request_body IS NOT NULL"""
        ).fetchall()
        for r in rows:
            try:
                body = json.loads(r["request_body"])
            except (TypeError, json.JSONDecodeError):
                continue
            cid = _conversation_id(body)
            turn = _turn_index(body)
            if cid is not None or turn is not None:
                conn.execute(
                    "UPDATE requests SET conversation_id=?, turn_index=? WHERE id=?",
                    (cid, turn, r["id"]),
                )
        conn.execute(
            "INSERT INTO settings (key, value, updated_ts) VALUES ('backfill_v1', '1', ?)",
            (time.time(),),
        )
    # Backfill v2: re-compute conversation_id for rows whose request_body contains the
    # Anthropic billing-header line. Older rows hashed the volatile `cch=<hex>;` value,
    # so they didn't aggregate. Rerun _conversation_id with the new normalizer.
    done_v2 = conn.execute("SELECT value FROM settings WHERE key='backfill_v2'").fetchone()
    if not done_v2:
        rows = conn.execute(
            """SELECT id, request_body FROM requests
               WHERE request_body IS NOT NULL AND request_body LIKE '%x-anthropic-billing-header%'"""
        ).fetchall()
        for r in rows:
            try:
                body = json.loads(r["request_body"])
            except (TypeError, json.JSONDecodeError):
                continue
            cid = _conversation_id(body)
            if cid is not None:
                conn.execute(
                    "UPDATE requests SET conversation_id=? WHERE id=?",
                    (cid, r["id"]),
                )
        conn.execute(
            "INSERT INTO settings (key, value, updated_ts) VALUES ('backfill_v2', '1', ?)",
            (time.time(),),
        )
    # Backfill v3: detect client_app for rows that don't have it yet (any historical row).
    done_v3 = conn.execute("SELECT value FROM settings WHERE key='backfill_v3'").fetchone()
    if not done_v3:
        rows = conn.execute(
            """SELECT id, request_headers, request_body FROM requests WHERE client_app IS NULL"""
        ).fetchall()
        for r in rows:
            headers = None
            body = None
            try:
                headers = json.loads(r["request_headers"]) if r["request_headers"] else None
            except (TypeError, json.JSONDecodeError):
                pass
            try:
                body = json.loads(r["request_body"]) if r["request_body"] else None
            except (TypeError, json.JSONDecodeError):
                pass
            app = _detect_client_app(headers, body if isinstance(body, dict) else None)
            conn.execute("UPDATE requests SET client_app=? WHERE id=?", (app, r["id"]))
        conn.execute(
            "INSERT INTO settings (key, value, updated_ts) VALUES ('backfill_v3', '1', ?)",
            (time.time(),),
        )
    # Backfill v4: re-run the detector on rows currently labeled 'unknown'. Detector keeps
    # gaining new patterns (vscode-copilot via GitHubCopilotChat UA, cline, aider, etc.); this
    # gives historical 'unknown' rows another shot without disturbing correctly-classified ones.
    done_v4 = conn.execute("SELECT value FROM settings WHERE key='backfill_v4'").fetchone()
    if not done_v4:
        rows = conn.execute(
            """SELECT id, request_headers, request_body FROM requests WHERE client_app = 'unknown'"""
        ).fetchall()
        for r in rows:
            headers = None
            body = None
            try:
                headers = json.loads(r["request_headers"]) if r["request_headers"] else None
            except (TypeError, json.JSONDecodeError):
                pass
            try:
                body = json.loads(r["request_body"]) if r["request_body"] else None
            except (TypeError, json.JSONDecodeError):
                pass
            app = _detect_client_app(headers, body if isinstance(body, dict) else None)
            if app != "unknown":
                conn.execute("UPDATE requests SET client_app=? WHERE id=?", (app, r["id"]))
        conn.execute(
            "INSERT INTO settings (key, value, updated_ts) VALUES ('backfill_v4', '1', ?)",
            (time.time(),),
        )
    # Backfill v6: re-classify rows still tagged 'unknown' using the latest detector patterns
    # (browsers, additional SDKs added since backfill_v4).
    done_v6 = conn.execute("SELECT value FROM settings WHERE key='backfill_v6'").fetchone()
    if not done_v6:
        rows = conn.execute(
            """SELECT id, request_headers, request_body FROM requests WHERE client_app = 'unknown'"""
        ).fetchall()
        for r in rows:
            headers = None
            body = None
            try:
                headers = json.loads(r["request_headers"]) if r["request_headers"] else None
            except (TypeError, json.JSONDecodeError):
                pass
            try:
                body = json.loads(r["request_body"]) if r["request_body"] else None
            except (TypeError, json.JSONDecodeError):
                pass
            app = _detect_client_app(headers, body if isinstance(body, dict) else None)
            if app != "unknown":
                conn.execute("UPDATE requests SET client_app=? WHERE id=?", (app, r["id"]))
        conn.execute(
            "INSERT INTO settings (key, value, updated_ts) VALUES ('backfill_v6', '1', ?)",
            (time.time(),),
        )
    # Backfill v5: decompress historical gzipped stream_chunks/response_body and re-extract
    # token usage. Anthropic was returning gzipped SSE for some requests; we now force identity
    # encoding, but old rows still have the compressed bytes (and NULL token counts).
    done_v5 = conn.execute("SELECT value FROM settings WHERE key='backfill_v5'").fetchone()
    if not done_v5:
        rows = conn.execute(
            """SELECT id, response_body, stream_chunks, total_tokens FROM requests
               WHERE stream_chunks LIKE char(31) || char(139) || '%'
                  OR response_body LIKE char(31) || char(139) || '%'"""
        ).fetchall()
        for r in rows:
            new_stream = _maybe_gunzip(r["stream_chunks"]) if r["stream_chunks"] else None
            new_body = _maybe_gunzip(r["response_body"]) if r["response_body"] else None
            updates = []
            params: list = []
            if isinstance(new_stream, str) and new_stream != r["stream_chunks"]:
                updates.append("stream_chunks=?")
                params.append(new_stream)
            if isinstance(new_body, str) and new_body != r["response_body"]:
                updates.append("response_body=?")
                params.append(new_body)
            pt, ct, tt = _extract_usage(new_body, new_stream)
            if pt is not None or ct is not None or tt is not None:
                updates.extend(["prompt_tokens=?", "completion_tokens=?", "total_tokens=?"])
                params.extend([pt, ct, tt])
            if updates:
                params.append(r["id"])
                conn.execute(f"UPDATE requests SET {', '.join(updates)} WHERE id=?", params)
        conn.execute(
            "INSERT INTO settings (key, value, updated_ts) VALUES ('backfill_v5', '1', ?)",
            (time.time(),),
        )
    # Sweep abandoned in-flight rows from a previous process. _save_pending writes a row at
    # the start of every request; if the proxy died (or was restarted) before _save_finish
    # ran, those rows are stuck as "pending" forever. At startup nothing else is writing yet,
    # so it's safe to mark any null-status row as abandoned.
    abandoned = conn.execute(
        """UPDATE requests
           SET status = 0,
               error = COALESCE(error, 'abandoned: proxy restarted before response completed'),
               duration_ms = COALESCE(duration_ms, (?  - ts) * 1000)
           WHERE status IS NULL AND error IS NULL""",
        (time.time(),),
    )
    if abandoned.rowcount:
        try:
            print(f"[init] marked {abandoned.rowcount} pending request(s) as abandoned")
        except Exception:
            pass
    # Tasks that were 'running' when the proxy died are now orphaned. Mark them failed
    # rather than silently re-running, so the user decides whether to retry.
    orphaned = conn.execute(
        """UPDATE proxy_tasks
           SET status = 'failed',
               error = COALESCE(error, 'orphaned: proxy restarted while task was running'),
               finished_ts = COALESCE(finished_ts, ?)
           WHERE status = 'running'""",
        (time.time(),),
    )
    if orphaned.rowcount:
        try:
            print(f"[init] marked {orphaned.rowcount} task(s) as orphaned/failed")
        except Exception:
            pass
    # Backfill v8: re-compute conversation_id for vscode-* rows. Earlier conversation
    # hashes included rotating content (Copilot Chat's <environment_info> terminal IDs,
    # cwd, exit codes), fragmenting one chat session across many cids. The hash now strips
    # XML wrappers before hashing, so re-running over historical rows merges the fragments.
    done_v8 = conn.execute("SELECT value FROM settings WHERE key='backfill_v8'").fetchone()
    if not done_v8:
        rows = conn.execute(
            """SELECT id, request_body FROM requests
               WHERE request_body IS NOT NULL
                 AND client_app IN ('vscode-copilot', 'github-copilot', 'continue.dev', 'cursor', 'vscode')"""
        ).fetchall()
        for r in rows:
            try:
                body = json.loads(r["request_body"])
            except (TypeError, json.JSONDecodeError):
                continue
            cid = _conversation_id(body)
            if cid is not None:
                conn.execute(
                    "UPDATE requests SET conversation_id=? WHERE id=?",
                    (cid, r["id"]),
                )
        conn.execute(
            "INSERT INTO settings (key, value, updated_ts) VALUES ('backfill_v8', '1', ?)",
            (time.time(),),
        )
    # Backfill v11: fix bridged-request rows where path is /v1/messages but the response
    # actually came from Ollama (protocol_bridge fired but earlier code didn't update the
    # upstream column at that time, or backfill_v10 used the path heuristic). Sniff
    # 'fp_ollama' / 'chat.completion.chunk' markers in response_body/stream_chunks to
    # recognize a bridged request and re-tag it as upstream='ollama'.
    done_v11 = conn.execute("SELECT value FROM settings WHERE key='backfill_v11'").fetchone()
    if not done_v11:
        conn.execute(
            """UPDATE requests SET upstream='ollama'
               WHERE upstream != 'ollama'
                 AND path LIKE '/v1/messages%'
                 AND (stream_chunks LIKE '%fp_ollama%'
                      OR response_body LIKE '%fp_ollama%'
                      OR stream_chunks LIKE '%chat.completion.chunk%')"""
        )
        conn.execute(
            "INSERT INTO settings (key, value, updated_ts) VALUES ('backfill_v11', '1', ?)",
            (time.time(),),
        )
    # Backfill v10: populate upstream column on historical rows. We can derive it from the
    # path: /v1/messages* → anthropic, /api/* and /v1/chat/completions/* → ollama, the rest
    # → 'unknown'. Future rows are populated at request time.
    done_v10 = conn.execute("SELECT value FROM settings WHERE key='backfill_v10'").fetchone()
    if not done_v10:
        # Set upstream from path heuristic for any row missing it.
        conn.execute(
            "UPDATE requests SET upstream='anthropic' WHERE upstream IS NULL AND path LIKE '/v1/messages%'"
        )
        conn.execute(
            "UPDATE requests SET upstream='ollama' WHERE upstream IS NULL AND "
            "(path LIKE '/api/%' OR path LIKE '/v1/chat/%' OR path LIKE '/v1/completions%' OR path = '/v1/models')"
        )
        conn.execute(
            "UPDATE requests SET upstream='other' WHERE upstream IS NULL"
        )
        conn.execute(
            "INSERT INTO settings (key, value, updated_ts) VALUES ('backfill_v10', '1', ?)",
            (time.time(),),
        )
    # Backfill v9: re-classify rows still tagged 'unknown' using the latest detector
    # (now honors x-client-name and extracts a label from arbitrary 'Name/Version' UAs).
    done_v9 = conn.execute("SELECT value FROM settings WHERE key='backfill_v9'").fetchone()
    if not done_v9:
        rows = conn.execute(
            """SELECT id, request_headers, request_body FROM requests WHERE client_app = 'unknown'"""
        ).fetchall()
        for r in rows:
            headers = None
            body = None
            try:
                headers = json.loads(r["request_headers"]) if r["request_headers"] else None
            except (TypeError, json.JSONDecodeError):
                pass
            try:
                body = json.loads(r["request_body"]) if r["request_body"] else None
            except (TypeError, json.JSONDecodeError):
                pass
            app = _detect_client_app(headers, body if isinstance(body, dict) else None)
            if app != "unknown":
                conn.execute("UPDATE requests SET client_app=? WHERE id=?", (app, r["id"]))
        conn.execute(
            "INSERT INTO settings (key, value, updated_ts) VALUES ('backfill_v9', '1', ?)",
            (time.time(),),
        )
    # Backfill v7: re-extract usage for Anthropic rows that have prompt-cache fields. Old
    # rows recorded only `input_tokens` (the fresh portion); we now sum fresh + cache_read +
    # cache_create so the prompt_tokens reflect the true prompt size.
    done_v7 = conn.execute("SELECT value FROM settings WHERE key='backfill_v7'").fetchone()
    if not done_v7:
        rows = conn.execute(
            """SELECT id, response_body, stream_chunks FROM requests
               WHERE response_body LIKE '%cache_read_input_tokens%'
                  OR response_body LIKE '%cache_creation_input_tokens%'
                  OR stream_chunks LIKE '%cache_read_input_tokens%'
                  OR stream_chunks LIKE '%cache_creation_input_tokens%'"""
        ).fetchall()
        for r in rows:
            pt, ct, tt = _extract_usage(r["response_body"], r["stream_chunks"])
            if pt is not None or ct is not None or tt is not None:
                conn.execute(
                    "UPDATE requests SET prompt_tokens=?, completion_tokens=?, total_tokens=? WHERE id=?",
                    (pt, ct, tt, r["id"]),
                )
        conn.execute(
            "INSERT INTO settings (key, value, updated_ts) VALUES ('backfill_v7', '1', ?)",
            (time.time(),),
        )
    # One-time blob split: move the multi-MB payload columns out of `requests` into
    # `request_blobs` and DROP them from `requests`, so list/count/aggregate scans over the
    # hot table never read blob pages. Robust to a partial prior run (checks table_info so it
    # never SELECTs an already-dropped column). VACUUM (after close) reclaims the freed space.
    _did_blob_split = False
    if not conn.execute("SELECT value FROM settings WHERE key='blob_split_v1'").fetchone():
        _req_cols = {r[1] for r in conn.execute("PRAGMA table_info(requests)").fetchall()}
        _blob_cols = [c for c in ("request_body", "response_body", "stream_chunks", "images_data") if c in _req_cols]
        if _blob_cols:
            _sel = ", ".join(_blob_cols)
            conn.execute(
                f"""INSERT OR IGNORE INTO request_blobs (id, {_sel})
                    SELECT id, {_sel} FROM requests
                    WHERE {" OR ".join(f"{c} IS NOT NULL" for c in _blob_cols)}"""
            )
            conn.commit()  # persist the moved data before we start dropping columns
            for _col in _blob_cols:
                try:
                    conn.execute(f"ALTER TABLE requests DROP COLUMN {_col}")
                except sqlite3.OperationalError:
                    pass  # DROP COLUMN needs SQLite >= 3.35; confirmed before deploy
        conn.execute(
            "INSERT INTO settings (key, value, updated_ts) VALUES ('blob_split_v1', '1', ?)",
            (time.time(),),
        )
        _did_blob_split = True
    # Compatibility view: presents `requests` joined with its blobs as one row shape, so the
    # (few) blob-reading queries use `requests_v` unchanged while fast scans stay on the lean
    # `requests`. Recreated every startup (cheap); r.* is clean because the blob columns were
    # dropped above. Needs SQLite >= 3.35 (DROP COLUMN) — confirmed before deploy.
    try:
        conn.execute("DROP VIEW IF EXISTS requests_v")
        conn.execute(
            "CREATE VIEW requests_v AS SELECT r.*, b.request_body, b.response_body, "
            "b.stream_chunks, b.images_data FROM requests r LEFT JOIN request_blobs b ON b.id = r.id"
        )
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()
    if _did_blob_split:
        try:  # VACUUM can't run inside a transaction — use a fresh autocommit connection
            _vc = sqlite3.connect(DB_PATH, timeout=120.0, isolation_level=None)
            _vc.execute("VACUUM")
            _vc.close()
        except sqlite3.OperationalError:
            pass


_BLOB_COLS = ("request_body", "response_body", "stream_chunks", "images_data")

def _blobs_upsert(conn, req_id, **cols):
    """Write large payload columns to the request_blobs side table (id PK). Only the columns
    passed are set; the rest are left intact via ON CONFLICT upsert. Callers use this instead
    of writing request_body/response_body/stream_chunks/images_data into `requests` directly."""
    cols = {k: v for k, v in cols.items() if k in _BLOB_COLS}
    if not cols:
        return
    names = list(cols)
    ph = ", ".join("?" for _ in names)
    upd = ", ".join(f"{n}=excluded.{n}" for n in names)
    conn.execute(
        f"INSERT INTO request_blobs (id, {', '.join(names)}) VALUES (?, {ph}) "
        f"ON CONFLICT(id) DO UPDATE SET {upd}",
        (req_id, *[cols[n] for n in names]),
    )


def _maybe_gunzip(data):
    """If `data` is gzip-compressed, return decompressed text. Otherwise return data as-is.
    Accepts either bytes or str. Used because some upstreams (Anthropic) ignore our header
    stripping and send gzipped SSE anyway, leaving compressed bytes in our DB."""
    if data is None:
        return data
    if isinstance(data, str):
        # Detect gzipped data hiding in a string from a prior latin-1/utf-8 decode round-trip.
        if len(data) >= 2 and data[:2] == "\x1f\x8b":
            try:
                return gzip.decompress(data.encode("latin-1", errors="replace")).decode("utf-8", errors="replace")
            except (OSError, EOFError):
                return data
        return data
    if isinstance(data, (bytes, bytearray)):
        if len(data) >= 2 and data[:2] == b"\x1f\x8b":
            try:
                return gzip.decompress(bytes(data)).decode("utf-8", errors="replace")
            except (OSError, EOFError):
                return data.decode("utf-8", errors="replace")
        return bytes(data).decode("utf-8", errors="replace")
    return data


def _extract_usage(body_text, stream_text):
    """Pull (prompt, completion, total) tokens out of a JSON body or SSE stream chunks.
    Handles four shapes: OpenAI (usage.prompt_tokens/completion_tokens/total_tokens),
    Ollama native (prompt_eval_count/eval_count), Anthropic /v1/messages
    (usage.input_tokens/output_tokens, sent via message_start + message_delta on streams),
    and Anthropic /v1/messages/count_tokens (top-level input_tokens with no output)."""
    pt = ct = tt = None
    body_text = _maybe_gunzip(body_text) if body_text else body_text
    stream_text = _maybe_gunzip(stream_text) if stream_text else stream_text

    def _anthropic_input_total(u):
        """Sum fresh + cache-read + cache-create. Anthropic prompt caching otherwise makes
        `input_tokens` undercount the true prompt size by 10-100× on cache hits."""
        if not isinstance(u, dict):
            return None
        parts = []
        for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            v = u.get(k)
            if isinstance(v, int):
                parts.append(v)
        return sum(parts) if parts else None

    def from_obj(j):
        if not isinstance(j, dict):
            return None, None, None
        u = j.get("usage")
        if isinstance(u, dict):
            if "prompt_tokens" in u or "completion_tokens" in u:
                return u.get("prompt_tokens"), u.get("completion_tokens"), u.get("total_tokens")
            if any(k in u for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")):
                return _anthropic_input_total(u), u.get("output_tokens"), None
        # Anthropic /v1/messages/count_tokens utility endpoint: {"input_tokens": N, ...}.
        if "input_tokens" in j and "messages" not in j and "content" not in j:
            return _anthropic_input_total(j), j.get("output_tokens"), None
        # Anthropic stream message_start wraps usage under "message"
        m = j.get("message")
        if isinstance(m, dict):
            mu = m.get("usage")
            if isinstance(mu, dict) and any(k in mu for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")):
                return _anthropic_input_total(mu), mu.get("output_tokens"), None
        # Ollama native (/api/chat, /api/generate)
        if "prompt_eval_count" in j or "eval_count" in j:
            return j.get("prompt_eval_count"), j.get("eval_count"), None
        return None, None, None

    if body_text:
        try:
            pt, ct, tt = from_obj(json.loads(body_text))
        except json.JSONDecodeError:
            pass

    if stream_text and (pt is None or ct is None):
        for line in stream_text.split("\n"):
            data = None
            if line.startswith("data: "):
                data = line[6:]
            elif line.strip().startswith("{"):
                data = line.strip()  # NDJSON
            if not data or data == "[DONE]":
                continue
            try:
                p, c, t = from_obj(json.loads(data))
            except json.JSONDecodeError:
                continue
            # Anthropic: input arrives in message_start, output in message_delta. Take the latest non-None for each.
            if p is not None: pt = p
            if c is not None: ct = c
            if t is not None: tt = t

    if tt is None and (pt is not None or ct is not None):
        tt = (pt or 0) + (ct or 0)
    return pt, ct, tt


def _live_update_from_chunk(req_id: str, chunk_text: str) -> None:
    """Walk a freshly arrived streaming chunk for usage hints. Cheap per-line scan: only
    parses lines that look like they could carry usage. Updates _LIVE_STREAMS in place."""
    if not chunk_text or req_id not in _LIVE_STREAMS:
        return
    state = _LIVE_STREAMS[req_id]
    for line in chunk_text.split("\n"):
        data = None
        if line.startswith("data: "):
            data = line[6:]
        elif line.lstrip().startswith("{"):
            data = line.strip()
        if not data or data == "[DONE]":
            continue
        # Cheap pre-filter — skip JSON parse unless the line could carry usage OR text content.
        if ("tokens" not in data and "usage" not in data and "eval_count" not in data
                and '"content"' not in data and '"text"' not in data and '"reasoning' not in data
                and '"tool_calls"' not in data and '"function"' not in data):
            continue
        try:
            j = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(j, dict):
            continue
        # Live streaming text: keep a rolling ~1.4KB tail so the Live View can show the reply
        # forming in real time without unbounded memory growth on a long response.
        _piece = ""
        _ch0 = j.get("choices")
        if isinstance(_ch0, list) and _ch0 and isinstance(_ch0[0], dict):
            _d = _ch0[0].get("delta") or {}
            if isinstance(_d.get("content"), str):
                _piece = _d["content"]
            elif isinstance(_d.get("reasoning_content"), str):   # thinking-model reasoning stream
                _piece = _d["reasoning_content"]
            elif isinstance(_d.get("reasoning"), str):
                _piece = _d["reasoning"]
            for _tc in (_d.get("tool_calls") or []):      # tool-call name + streaming arguments
                if not isinstance(_tc, dict):
                    continue
                _fn = _tc.get("function") or {}
                if _fn.get("name"):
                    _piece += ("\n🔧 " if (state.get("text") or _piece) else "🔧 ") + _fn["name"] + " "
                if isinstance(_fn.get("arguments"), str):
                    _piece += _fn["arguments"]
        if not _piece and j.get("type") == "content_block_delta":
            _dd = j.get("delta") or {}
            if _dd.get("type") == "text_delta" and isinstance(_dd.get("text"), str):
                _piece = _dd["text"]
        if _piece:
            state["text"] = ((state.get("text") or "") + _piece)[-6000:]
        def _anth_total(u):
            parts = [u.get(k) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens") if isinstance(u.get(k), int)]
            return sum(parts) if parts else None

        # OpenAI-style usage block (only sent when stream_options.include_usage=true).
        u = j.get("usage")
        if isinstance(u, dict):
            if "prompt_tokens" in u and u["prompt_tokens"] is not None:
                state["prompt"] = u["prompt_tokens"]
            if "completion_tokens" in u and u["completion_tokens"] is not None:
                state["completion"] = u["completion_tokens"]
            anth_in = _anth_total(u)
            if anth_in is not None:
                state["prompt"] = anth_in
            if "output_tokens" in u and u["output_tokens"] is not None:
                state["completion"] = u["output_tokens"]
        # Anthropic message_start wraps the initial usage under "message".
        m = j.get("message")
        if isinstance(m, dict):
            mu = m.get("usage")
            if isinstance(mu, dict):
                anth_in = _anth_total(mu)
                if anth_in is not None:
                    state["prompt"] = anth_in
                if mu.get("output_tokens") is not None:
                    state["completion"] = mu["output_tokens"]
        # Ollama native /api/chat final chunk.
        if j.get("prompt_eval_count") is not None:
            state["prompt"] = j["prompt_eval_count"]
        if j.get("eval_count") is not None:
            state["completion"] = j["eval_count"]


def _live_snapshot(req_id: str) -> dict | None:
    """Return a public-shape live token snapshot for the requests-list API, or None."""
    s = _LIVE_STREAMS.get(req_id)
    if not s:
        return None
    pt = s.get("prompt")
    ct = s.get("completion")
    est = s.get("est_prompt")
    # If the upstream hasn't given us prompt yet, fall back to our chars/3.5 estimate.
    if pt is None and est is not None:
        pt = est
        est_used = True
    else:
        est_used = False
    if pt is None and ct is None:
        return None
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": (pt or 0) + (ct or 0) if (pt is not None or ct is not None) else None,
        "estimated": est_used,
    }


def _is_anthropic_path(path: str) -> bool:
    """True for Anthropic-shaped endpoints (/v1/messages*, /v1/complete*)."""
    p = path if path.startswith("/") else "/" + path
    return p.startswith("/v1/messages") or p.startswith("/v1/complete")


def _pick_upstream(full_path: str) -> tuple[str, str]:
    """Path-based routing. Anthropic for /v1/messages* and /v1/complete*; everything else
    (OpenAI-compat /v1/chat/completions, /v1/models, embeddings, plus all Ollama /api/*)
    goes to OLLAMA_URL. Returns (base_url, label)."""
    if _is_anthropic_path(full_path):
        return ANTHROPIC_URL, "anthropic"
    return OLLAMA_URL, "ollama"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _load_panic_mode()
    app.state.started_at = time.time()
    app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(None))
    app.state.metrics_client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
    app.state.metrics_task = asyncio.create_task(_metrics_loop(app))
    app.state.task_worker = asyncio.create_task(_task_worker_loop(app))
    app.state.zombie_killer = asyncio.create_task(_inflight_zombie_killer(app))
    try:
        yield
    finally:
        for t_attr in ("metrics_task", "task_worker", "zombie_killer"):
            t = getattr(app.state, t_attr, None)
            if t is not None:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        await app.state.client.aclose()
        await app.state.metrics_client.aclose()


# -------- System metrics collector --------

_cpu_prev: dict = {"total": 0, "idle": 0}


def _read_proc_stat_cpu():
    """Sample /proc/stat once. Returns (total_jiffies, idle_jiffies) for the aggregate cpu line."""
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        parts = line.split()
        if not parts or parts[0] != "cpu":
            return None
        nums = [int(x) for x in parts[1:]]
        # user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        total = sum(nums)
        return total, idle
    except OSError:
        return None


def _cpu_pct() -> float | None:
    sample = _read_proc_stat_cpu()
    if sample is None:
        return None
    total, idle = sample
    prev_total = _cpu_prev["total"]
    prev_idle = _cpu_prev["idle"]
    _cpu_prev["total"] = total
    _cpu_prev["idle"] = idle
    if prev_total == 0:
        return None
    dt = total - prev_total
    di = idle - prev_idle
    if dt <= 0:
        return None
    return round(100.0 * (dt - di) / dt, 1)


def _read_proc_meminfo() -> dict:
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                value = rest.strip().split()
                if value and value[0].isdigit():
                    out[key.strip()] = int(value[0])  # kB
    except OSError:
        pass
    return out


def _mem_snapshot() -> dict:
    info = _read_proc_meminfo()
    if not info:
        return {}
    total_mb = info.get("MemTotal", 0) // 1024
    avail_mb = info.get("MemAvailable", info.get("MemFree", 0)) // 1024
    used_mb = total_mb - avail_mb
    return {"total_mb": total_mb, "used_mb": used_mb, "avail_mb": avail_mb}


def _load_avg() -> float | None:
    try:
        with open("/proc/loadavg") as f:
            return float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


_NVIDIA_SMI = shutil.which("nvidia-smi")


def _maybe_int(s):
    """Parse an nvidia-smi numeric field, treating '[N/A]'/'[Not Supported]'/'' as None."""
    if s is None:
        return None
    s = s.strip()
    if not s or s.startswith("["):
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return None


def _gpu_snapshot() -> list:
    """Returns [{idx, name, util_pct, mem_total_mb, mem_used_mb, temp_c, unified, processes:[{pid, name, mem_mb}]}].
    Unified-memory GPUs (e.g. DGX Spark / GB10) report [N/A] for memory.total — we fall back to system RAM total
    and sum per-process VRAM for "used"."""
    if not _NVIDIA_SMI:
        return []
    try:
        gpu_csv = subprocess.run(
            [_NVIDIA_SMI, "--query-gpu=index,name,utilization.gpu,memory.total,memory.used,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        proc_csv = subprocess.run(
            [_NVIDIA_SMI, "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        idx_csv = subprocess.run(
            [_NVIDIA_SMI, "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if gpu_csv.returncode != 0:
        return []

    uuid_to_idx = {}
    for line in idx_csv.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                uuid_to_idx[parts[1]] = int(parts[0])
            except ValueError:
                pass

    procs_by_idx: dict[int, list] = {}
    for line in proc_csv.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpu_uuid, pid_s, name, mem_s = parts[0], parts[1], parts[2], parts[3]
        idx = uuid_to_idx.get(gpu_uuid)
        if idx is None:
            continue
        try:
            procs_by_idx.setdefault(idx, []).append({
                "pid": int(pid_s), "name": name,
                "mem_mb": _maybe_int(mem_s) or 0,
            })
        except ValueError:
            pass

    gpus = []
    for line in gpu_csv.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        mem_total = _maybe_int(parts[3])
        mem_used = _maybe_int(parts[4])
        procs = procs_by_idx.get(idx, [])
        unified = False
        # Fall back for unified-memory GPUs: total = system RAM, used = sum of per-process VRAM.
        if mem_total is None:
            sys_mem = _read_proc_meminfo()
            if sys_mem.get("MemTotal"):
                mem_total = sys_mem["MemTotal"] // 1024  # kB -> MB
                unified = True
        if mem_used is None:
            mem_used = sum(p.get("mem_mb", 0) for p in procs) if procs else 0
            unified = unified or any(p.get("mem_mb", 0) > 0 for p in procs)
        gpus.append({
            "idx": idx,
            "name": parts[1],
            "util_pct": _maybe_int(parts[2]),
            "mem_total_mb": mem_total,
            "mem_used_mb": mem_used,
            "temp_c": _maybe_int(parts[5]),
            "unified": unified,
            "processes": procs,
        })
    return gpus


def _find_ollama_pid() -> int | None:
    """Walk /proc looking for a process whose comm is 'ollama'."""
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    for name in entries:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/comm") as f:
                if f.read().strip() == "ollama":
                    return int(name)
        except OSError:
            continue
    return None


def _read_ollama_env(pid: int) -> dict:
    """Read /proc/<pid>/environ, return only OLLAMA_* keys (safe to surface).
    Requires same UID as the ollama process or root."""
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            data = f.read()
    except OSError:
        return {}
    env: dict = {}
    for entry in data.split(b"\x00"):
        if b"=" not in entry:
            continue
        k, v = entry.split(b"=", 1)
        ks = k.decode("utf-8", errors="replace")
        if ks.startswith("OLLAMA_"):
            env[ks] = v.decode("utf-8", errors="replace")
    return env


_OLLAMA_UNIT = os.environ.get("OLLAMA_SYSTEMD_UNIT", "ollama.service")


def _read_systemd_env() -> dict:
    """Fall-back: pull Environment= entries from the ollama systemd unit. No sudo required.
    Used when the proxy runs as a different user than the ollama process and so can't read /proc/<pid>/environ."""
    if not shutil.which("systemctl"):
        return {}
    try:
        out = subprocess.run(
            ["systemctl", "show", _OLLAMA_UNIT, "--property=Environment", "--no-pager"],
            capture_output=True, text=True, timeout=2,
        )
    except (subprocess.TimeoutExpired, OSError):
        return {}
    if out.returncode != 0:
        return {}
    line = out.stdout.strip()
    if not line.startswith("Environment="):
        return {}
    raw = line[len("Environment="):]
    import shlex
    try:
        tokens = shlex.split(raw, posix=True)
    except ValueError:
        tokens = raw.split()
    env: dict = {}
    for tok in tokens:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k.startswith("OLLAMA_"):
            env[k] = v
    return env


def _semver_tuple(v: str) -> tuple:
    """Parse a dotted version like '0.21.0' into a tuple of ints for comparison. Anything
    non-numeric becomes 0 so we don't crash on pre-release suffixes."""
    if not v:
        return (0, 0, 0)
    parts = re.split(r"[.\-+]", str(v).lstrip("v"))
    out: list[int] = []
    for p in parts[:3]:
        try:
            out.append(int(re.sub(r"[^\d].*$", "", p) or "0"))
        except (ValueError, TypeError):
            out.append(0)
    while len(out) < 3:
        out.append(0)
    return tuple(out)


# Architectures Ollama refuses to batch — it returns "model architecture does not currently
# support parallel requests" and serves them one at a time. The whole Qwen3 family
# (qwen3, qwen3moe, qwen3next, qwen35, etc.) is serial under Ollama.
# This is runtime-specific: LM Studio / llama.cpp continuous batching parallelizes the same
# models fine, which is why we route heavy qwen traffic there.
_OLLAMA_SERIAL_ARCH_PREFIXES = ("qwen3",)


def _model_parallelism(runtime: str, arch: str | None = None, name: str | None = None) -> dict:
    """Best-effort answer to 'can this model serve concurrent requests on this runtime?'.
    Returns {"parallel": True|False|None, "reason": str}. None = couldn't determine."""
    a = (arch or "").lower()
    n = (name or "").lower()
    if runtime == "lmstudio":
        return {"parallel": True,
                "reason": "LM Studio (llama.cpp continuous batching) batches all architectures"}
    if runtime == "ollama":
        hay = a or n
        if hay and any(hay.startswith(p) or (p in hay) for p in _OLLAMA_SERIAL_ARCH_PREFIXES):
            return {"parallel": False,
                    "reason": "Ollama serializes this architecture — 'does not support parallel requests'"}
        if not hay:
            return {"parallel": None, "reason": "architecture unknown"}
        return {"parallel": True,
                "reason": "Ollama batches this architecture up to OLLAMA_NUM_PARALLEL"}
    return {"parallel": None, "reason": "unknown runtime"}


async def _ollama_snapshot(client: httpx.AsyncClient) -> dict:
    out: dict = {"reachable": False, "ps": [], "tags": [], "env": {}, "env_source": None,
                 "pid": None, "version": None, "recommended_version": OLLAMA_RECOMMENDED_VERSION,
                 "outdated": False}
    pid = _find_ollama_pid()
    if pid:
        out["pid"] = pid
        env = _read_ollama_env(pid)
        if env:
            out["env"] = env
            out["env_source"] = "/proc/environ"
    if not out["env"]:
        # Different user owns the ollama process — fall back to its systemd unit definition.
        env = _read_systemd_env()
        if env:
            out["env"] = env
            out["env_source"] = f"systemctl show {_OLLAMA_UNIT}"
    try:
        r = await client.get(f"{OLLAMA_URL}/api/version")
        if r.status_code == 200:
            out["version"] = (r.json() or {}).get("version")
    except (httpx.RequestError, ValueError):
        pass
    if out["version"] and OLLAMA_RECOMMENDED_VERSION:
        out["outdated"] = _semver_tuple(out["version"]) < _semver_tuple(OLLAMA_RECOMMENDED_VERSION)
    try:
        r = await client.get(f"{OLLAMA_URL}/api/ps")
        if r.status_code == 200:
            out["reachable"] = True
            data = r.json()
            for m in (data.get("models") or []):
                _fam = (m.get("details") or {}).get("family")
                _par = _model_parallelism("ollama", _fam, m.get("name"))
                out["ps"].append({
                    "name": m.get("name"),
                    "model": m.get("model"),
                    "size_mb": (m.get("size") or 0) // (1024 * 1024) if m.get("size") else None,
                    "size_vram_mb": (m.get("size_vram") or 0) // (1024 * 1024) if m.get("size_vram") else None,
                    "expires_at": m.get("expires_at"),
                    "parameter_size": (m.get("details") or {}).get("parameter_size"),
                    "family": _fam,
                    "parallel": _par["parallel"],
                    "parallel_reason": _par["reason"],
                })
    except (httpx.RequestError, ValueError):
        return out
    try:
        r = await client.get(f"{OLLAMA_URL}/api/tags")
        if r.status_code == 200:
            data = r.json()
            for m in (data.get("models") or []):
                _fam = (m.get("details") or {}).get("family")
                _par = _model_parallelism("ollama", _fam, m.get("name"))
                out["tags"].append({
                    "name": m.get("name"),
                    "size_mb": (m.get("size") or 0) // (1024 * 1024) if m.get("size") else None,
                    "modified_at": m.get("modified_at"),
                    "parameter_size": (m.get("details") or {}).get("parameter_size"),
                    "family": _fam,
                    "parallel": _par["parallel"],
                    "parallel_reason": _par["reason"],
                })
    except (httpx.RequestError, ValueError):
        pass
    return out


async def _lmstudio_snapshot(client: httpx.AsyncClient) -> dict:
    """Try LM Studio's REST v0 API first (richer info), fall back to OpenAI-compat /v1/models."""
    out = {"reachable": False, "loaded": [], "available": []}
    try:
        r = await client.get(f"{LMSTUDIO_URL}/api/v0/models")
        if r.status_code == 200:
            out["reachable"] = True
            data = r.json()
            entries = data.get("data") if isinstance(data, dict) else data
            for m in (entries or []):
                _par = _model_parallelism("lmstudio", m.get("arch"), m.get("id"))
                rec = {
                    "id": m.get("id"),
                    "type": m.get("type"),
                    "state": m.get("state"),
                    "loaded_context_length": m.get("loaded_context_length"),
                    "max_context_length": m.get("max_context_length"),
                    "arch": m.get("arch"),
                    "publisher": m.get("publisher"),
                    "quant": m.get("quantization"),
                    "compat_type": m.get("compatibility_type"),
                    "parallel": _par["parallel"],
                    "parallel_reason": _par["reason"],
                }
                out["available"].append(rec)
                if (m.get("state") or "").lower() in ("loaded", "ready"):
                    out["loaded"].append(rec)
            return out
    except (httpx.RequestError, ValueError):
        pass
    # Fallback: OpenAI-compat — only tells us models exist, not whether loaded.
    try:
        r = await client.get(f"{LMSTUDIO_URL}/v1/models")
        if r.status_code == 200:
            out["reachable"] = True
            data = r.json()
            for m in (data.get("data") or []):
                out["available"].append({"id": m.get("id"), "type": "unknown"})
    except (httpx.RequestError, ValueError):
        pass
    return out


# Priority queue: per-bucket semaphores. Initialized lazily on first request after config
# load so OLLAMA_NUM_PARALLEL is readable. Reset whenever rules.json changes via the API.
_PRIORITY_SEMS: dict[str, asyncio.Semaphore | None] = {}
_PRIORITY_CAPS_USED: dict[str, int | None] = {}


def _ollama_num_parallel() -> int | None:
    """Best-effort read of OLLAMA_NUM_PARALLEL from /proc env or systemd unit. Returns None
    if not detectable (in which case priority caps fall back to hardcoded defaults)."""
    try:
        env: dict = {}
        pid = _find_ollama_pid()
        if pid:
            env = _read_ollama_env(pid)
        if not env:
            env = _read_systemd_env()
        v = env.get("OLLAMA_NUM_PARALLEL")
        return int(v) if v else None
    except (ValueError, TypeError, OSError):
        return None


def _resolve_priority_caps() -> dict[str, int | None]:
    """Return {priority: cap} per current config. cap=None means unlimited."""
    cfg = (load_rules_config().get("request_priority") or {})
    caps_cfg = cfg.get("caps") or {}
    np = _ollama_num_parallel() or 1
    out: dict[str, int | None] = {}
    for name, default in (("high", None),
                          ("normal", max(1, np - 1) if np > 1 else None),
                          ("low", 1)):
        if name in caps_cfg:
            v = caps_cfg[name]
            out[name] = int(v) if isinstance(v, (int, float)) and v > 0 else None
        else:
            out[name] = default
    return out


def _ensure_priority_sems():
    """Lazily build (or rebuild on cap-change) the per-priority semaphores."""
    caps = _resolve_priority_caps()
    if caps == _PRIORITY_CAPS_USED and _PRIORITY_SEMS:
        return
    _PRIORITY_CAPS_USED.clear()
    _PRIORITY_CAPS_USED.update(caps)
    _PRIORITY_SEMS.clear()
    for name, cap in caps.items():
        _PRIORITY_SEMS[name] = asyncio.Semaphore(cap) if cap and cap > 0 else None


def _resolve_request_priority(headers: dict, client_app: str | None) -> str:
    """Pick priority for a request. Header > client_priority map > default_priority."""
    cfg = (load_rules_config().get("request_priority") or {})
    h = (headers.get("x-priority") or "").lower().strip()
    if h in ("high", "normal", "low"):
        return h
    cmap = cfg.get("client_priority") or {}
    if client_app and client_app in cmap and cmap[client_app] in ("high", "normal", "low"):
        return cmap[client_app]
    default = cfg.get("default_priority") or "normal"
    return default if default in ("high", "normal", "low") else "normal"


# Cache for the Ollama-registry update-check fan-out. Keeps us from hammering the
# registry every time the user opens the System tab. TTL = 1 hour.
_OLLAMA_UPDATE_CACHE: dict[str, dict] = {}
_OLLAMA_UPDATE_CACHE_TTL_S = 3600


async def _check_ollama_registry(client: httpx.AsyncClient, name: str, local_digest: str) -> dict:
    """For a single model `name:tag`, ask the Ollama registry for its current manifest
    and compare config-blob digests. Returns a result dict with status
    'up_to_date' | 'outdated' | 'not_in_registry' | 'error' | 'unknown'."""
    if ":" in name:
        base, tag = name.rsplit(":", 1)
    else:
        base, tag = name, "latest"
    # Default namespace is `library/`. User-namespaced models keep their `<user>/<model>` form.
    ns_path = base if "/" in base else f"library/{base}"
    url = f"https://registry.ollama.ai/v2/{ns_path}/manifests/{tag}"
    headers = {"Accept": "application/vnd.docker.distribution.manifest.v2+json"}
    base_out = {"name": name, "tag": tag, "local_digest": local_digest, "remote_digest": None}
    try:
        # HEAD is enough — we only need the `ollama-content-digest` response header. Note
        # this is a custom Ollama header, NOT the standard Docker-Content-Digest. The
        # manifest body has its own `config.digest` but that's a different blob and won't
        # match what Ollama stores in /api/tags.
        r = await client.head(url, headers=headers, timeout=httpx.Timeout(10.0))
    except Exception as e:
        return {**base_out, "status": "error", "error": str(e)}
    if r.status_code == 404:
        return {**base_out, "status": "not_in_registry"}
    if r.status_code != 200:
        return {**base_out, "status": "error",
                "error": f"HTTP {r.status_code}"}
    remote = (r.headers.get("ollama-content-digest") or "").removeprefix("sha256:").strip().lower()
    if not remote:
        return {**base_out, "status": "unknown",
                "error": "registry response missing ollama-content-digest header"}
    if not local_digest:
        return {**base_out, "remote_digest": remote, "status": "unknown",
                "error": "no local digest"}
    status = "up_to_date" if remote == local_digest.lower() else "outdated"
    return {**base_out, "remote_digest": remote, "status": status}


async def _comfyui_snapshot(client: httpx.AsyncClient) -> dict:
    """Probe ComfyUI's /system_stats and /queue. Returns reachability, VRAM/RAM usage,
    queue depth, and checkpoint count. Quiet on failure — ComfyUI being down is normal."""
    out: dict = {"reachable": False, "url": SD_URL}
    try:
        s_task = client.get(f"{SD_URL}/system_stats", timeout=3.0)
        q_task = client.get(f"{SD_URL}/queue", timeout=3.0)
        s_resp, q_resp = await asyncio.gather(s_task, q_task, return_exceptions=True)
    except Exception:
        return out
    if isinstance(s_resp, httpx.Response) and s_resp.status_code == 200:
        out["reachable"] = True
        try:
            sj = s_resp.json()
            sysinfo = sj.get("system") or {}
            out["python"] = sysinfo.get("python_version")
            out["os"] = sysinfo.get("os")
            out["ram_total"] = sysinfo.get("ram_total")
            out["ram_free"] = sysinfo.get("ram_free")
            out["pytorch_version"] = sysinfo.get("pytorch_version")
            out["embedded_python"] = sysinfo.get("embedded_python")
            out["devices"] = []
            for d in (sj.get("devices") or []):
                out["devices"].append({
                    "name": d.get("name"),
                    "type": d.get("type"),
                    "vram_total": d.get("vram_total"),
                    "vram_free": d.get("vram_free"),
                    "torch_vram_total": d.get("torch_vram_total"),
                    "torch_vram_free": d.get("torch_vram_free"),
                })
        except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
            pass
    if isinstance(q_resp, httpx.Response) and q_resp.status_code == 200:
        out["reachable"] = True
        try:
            qj = q_resp.json()
            out["queue_running"] = len(qj.get("queue_running") or [])
            out["queue_pending"] = len(qj.get("queue_pending") or [])
        except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
            pass
    return out


async def _collect_once(app: FastAPI):
    cpu = _cpu_pct()
    mem = _mem_snapshot()
    gpus = _gpu_snapshot()
    ollama, lmstudio, comfyui = await asyncio.gather(
        _ollama_snapshot(app.state.metrics_client),
        _lmstudio_snapshot(app.state.metrics_client),
        _comfyui_snapshot(app.state.metrics_client),
        return_exceptions=False,
    )
    ts = time.time()
    conn = db()
    conn.execute(
        """INSERT OR REPLACE INTO system_metrics
           (ts, cpu_pct, load_1m, mem_total_mb, mem_used_mb, mem_avail_mb, gpu_json,
            ollama_json, lmstudio_json, comfyui_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ts,
            cpu,
            _load_avg(),
            mem.get("total_mb"),
            mem.get("used_mb"),
            mem.get("avail_mb"),
            json.dumps(gpus) if gpus else None,
            json.dumps(ollama),
            json.dumps(lmstudio),
            json.dumps(comfyui),
        ),
    )
    # Retention
    conn.execute("DELETE FROM system_metrics WHERE ts < ?", (ts - METRICS_RETENTION_S,))
    # Prune old requests periodically (throttled; indexed on ts so it's cheap) to bound
    # the DB size. Freed pages are reused, so the file stays roughly stable.
    global _last_request_prune
    if REQUEST_RETENTION_DAYS > 0 and (ts - _last_request_prune) > 600:
        _last_request_prune = ts
        conn.execute("DELETE FROM requests WHERE ts < ?", (ts - REQUEST_RETENTION_DAYS * 86400,))
    conn.commit()
    conn.close()


async def _metrics_loop(app: FastAPI):
    # Prime CPU sample so the first real reading is meaningful.
    _cpu_pct()
    await asyncio.sleep(1)
    while True:
        try:
            await _collect_once(app)
        except Exception:
            pass  # Telemetry must never crash the proxy.
        await asyncio.sleep(METRICS_INTERVAL_S)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


# -------- Rule engine --------

RULES_REGISTRY: dict = {}

DEFAULT_RULES_CONFIG = {
    "loop_detector": {
        "enabled": True,
        "action": "block",       # "block" | "warn"
        "max_repeats": 4,         # signature appearing this many times in window blocks
        "window": 10,             # how many trailing assistant messages to inspect
        "tail_consecutive": 3,    # OR: this many identical signatures at the tail blocks
    },
    "schema_validator": {
        "enabled": False,
        "action": "block",
        "strict_types": True,
        "reject_unknown_fields": False,
    },
    "tool_call_xml_retry": {
        # When Ollama's tool-call parser rejects a model output with "XML syntax error"
        # (typically a missing </parameter> close from qwen-style models), we transparently
        # retry the same request with an extra system message telling the model exactly
        # what it screwed up. Manual retry doesn't help because the model has no signal
        # that anything went wrong on the prior attempt; the corrective hint is what makes
        # the retry land. Off by default; enable per-model if you hit this a lot.
        "enabled": False,
        "max_retries": 1,
        "error_patterns": [
            "XML syntax error",
            "element <parameter> closed by",
        ],
        "applies_to_upstream": ["ollama"],
        "corrective_hint": (
            "Your previous tool call output had malformed XML — most likely a missing "
            "</parameter> close tag before </function>. Output complete, properly-closed "
            "XML for every tool call. Each <parameter name=\"...\"> must be closed with "
            "</parameter> before the enclosing </function>."
        ),
    },
    "xml_autofix": {
        # Post-flight: when the model emits XML-ish content in its text response that
        # doesn't parse cleanly (unclosed tags, unescaped &, etc.), record findings
        # in the audit log. v1 is audit-only — no payload mutation. Future `fix` mode
        # would auto-repair before the response leaves the proxy.
        "enabled": False,
        "action": "audit",   # 'audit' = log only; 'silent' = detect but don't log; 'fix' = (future)
        "min_tags": 2,       # skip responses with fewer than this many XML-ish tags
        "ignore_tags": ["br", "hr", "img", "input", "meta", "link"],  # HTML void tags, never need closing
    },
    "tool_args_autofix": {
        # Post-flight: when the model emits a tool_call missing required fields,
        # patch the arguments with configured defaults BEFORE forwarding to the client.
        # Order: per-tool defaults > "*" wildcard > schema's own "default" property.
        # Runs before schema_validator — fixed args may then satisfy validation.
        "enabled": False,
        "action": "audit",   # "audit" = record in audit log; "silent" = don't
        "defaults": {        # { "tool_name" or "*": { "field": value } }
            # Example:
            # "*": {"startLine": 1, "endLine": 1000},
            # "read_file": {"startLine": 1}
        },
    },
    "hallucinated_tool": {
        "enabled": False,
        "action": "block",
    },
    "tool_failure_breaker": {
        "enabled": False,
        "action": "block",
        "max_errors": 3,
        "window": 5,
    },
    "model_router": {
        "enabled": False,
        # Static aliases applied first: { "from_model": "to_model" }
        "aliases": {},
        # Conditional rules; first match wins (after aliases). Each rule:
        #   { "if": { ...conditions... }, "then": "target_model_name", "upstream": "lmstudio" }
        # Conditions (any combination, all must match):
        #   from_model: string | [strings]      match the model name (post-alias)
        #   from_model_prefix: string | [str]   match a model-name prefix (e.g. "qwen")
        #   from_client: string | [strings]     exact client IP
        #   from_client_prefix: string          IP prefix
        #   prompt_chars_lt: int                total chars across all message content
        #   prompt_chars_gt: int                total chars across all message content
        #   has_tools: bool                     request includes a non-empty tools[] array
        #   has_images: bool                    a message carries image content (vision request)
        #   path_prefix: string                 match URL path prefix
        # Optional "upstream": "ollama" | "lmstudio" sends the (rewritten) model to that backend
        # instead of the path-based default — both speak the /v1 shape so no translation needed.
        # Lets you route e.g. qwen to LM Studio (parallel-capable) while everything else stays on
        # Ollama, all still logged through the proxy.
        "rules": [],
        # When True, rewrites the `model` field in upstream responses (and streamed chunks)
        # back to the original requested model name. Hides the rewrite from clients that
        # hard-check the response model. Off by default — clients see the actual upstream
        # model and an X-Proxy-Model-Rewrite response header indicating the swap.
        "preserve_response_model_name": False,
    },
    "ollama_options": {
        # Inject generation parameters into outgoing requests when the client didn't set them.
        # For /v1/* (OpenAI-compat) keys go top-level; for /api/* (Ollama native) most keys go
        # into body.options (special keys 'keep_alive' and 'format' stay top-level on /api/*).
        # Existing client-set values are NEVER overwritten — this is fill-in-defaults semantics.
        "enabled": False,
        "defaults": {},          # global, e.g. {"num_ctx": 8192, "temperature": 0.7}
        "per_model": {},          # {"qwen2.5:14b": {"num_ctx": 16384}}
        "per_client": {},         # {"10.0.0.20": {"temperature": 0.2}}
        "rules": [],              # [{"if": {...}, "set": {"num_ctx": 16384}}] (first match)
        # Hard ceiling on num_ctx, applied even to client-set values (unlike the fill-in
        # options above). Null = no cap. Set e.g. 32768 to stop a client forcing a huge
        # context (which balloons the KV cache and kills parallelism). Independent of
        # `enabled` — the cap applies whenever it's a positive int.
        "num_ctx_max": None,
    },
    "context_overflow_guard": {
        # Pre-flight transform: estimates prompt token count and reacts when it exceeds num_ctx.
        # Without this, Ollama silently truncates prompts that exceed num_ctx, ruining responses.
        # Runs AFTER ollama_options so it sees the effective num_ctx that will be sent upstream.
        "enabled": False,
        # action: "warn" (log only), "bump" (raise num_ctx in the request),
        #         "trim" (drop oldest non-system messages), or "block" (return 413).
        "action": "warn",
        "chars_per_token": 3.5,         # heuristic: avg chars per token (cl100k ≈ 3.8, qwen ≈ 3.5)
        "headroom_ratio": 0.95,         # treat as overflow when est_tokens > num_ctx * this
        "assumed_default_num_ctx": 4096,  # used only if no num_ctx is set in body or by options
        "max_ctx": 131072,               # bump cap; raise if your model supports more
        "bump_to": None,                 # explicit bump target, else next pow2 ≥ est_tokens
        "min_keep_messages": 4,          # trim never drops below this many recent non-system msgs
    },
    "tool_pruner": {
        # Pre-flight transform: drops tool definitions the model has been offered repeatedly
        # but never invoked in this conversation. Cuts prompt tokens AND tightens tool selection.
        # Runs after ollama_options, before context_overflow_guard, so pruning reduces the
        # token estimate the overflow guard sees.
        "enabled": False,
        # action: "prune" (drop unused tools from the request) or "warn" (audit only).
        "action": "prune",
        "min_turns_offered": 3,          # tool must have been offered this many prior turns
        "min_history_turns": 2,          # only act once the conversation has this many prior turns
        "always_keep": [],                # tool names that are never pruned (e.g. ["read_file"])
        "max_prune_ratio": 0.8,          # cap: never drop more than this fraction of tools[]
        "include_hint": True,             # append a system note listing pruned tool names
    },
    "protocol_bridge": {
        # Translate Anthropic /v1/messages requests to OpenAI /v1/chat/completions when the
        # (post-router) target model isn't a Claude model. Lets Anthropic clients (Claude Code,
        # Anthropic SDKs) drive any OpenAI-compatible backend (Ollama, LM Studio, vLLM, etc.).
        # Triggers automatically when model_router rewrites the request's model from a Claude
        # name to a non-Claude one. Disable to leave Anthropic traffic untranslated.
        "enabled": True,
        # When True, always bridge Anthropic-shape traffic regardless of the target model —
        # even Claude model names get translated to OpenAI shape and routed to OLLAMA_URL.
        # Useful when the OpenAI-compatible backend is itself serving a Claude model.
        "force": False,
    },
    "shadow_router": {
        # Fan out incoming requests to one or more SHADOW models in parallel. The client
        # receives the primary response unchanged; shadow responses are stored to the DB linked
        # via `shadow_of` for side-by-side comparison in the UI.
        # Shadows are best-effort — failures don't affect the primary. Always non-streaming.
        # Anthropic-shape primaries with non-Claude shadow targets are auto-translated.
        # Each rule has the same `if` shape as model_router; the matching target is in `shadow_to`.
        # Example:
        #   {"if": {"from_model": "claude-opus-4-7"}, "shadow_to": "qwen3-coder-next:latest"}
        "enabled": False,
        "rules": [],
    },
    "request_dedup": {
        # When the same client sends two identical streaming requests within `ttl_s`, the
        # second one is "tee'd" from the first — it subscribes to the first's response
        # stream and gets the same bytes without re-running upstream. Saves GPU work when
        # clients (e.g. claude-code) fan out parallel duplicates. Only applies to streaming
        # requests (is_stream=true); non-stream requests pass through.
        "enabled": False,
        "ttl_s": 60,  # how long after a primary finishes to keep its bytes available for dedup
    },
    "request_priority": {
        # Soft priority via per-bucket concurrency caps. Each request is assigned a priority
        # ('high' | 'normal' | 'low'), which selects an asyncio.Semaphore. Requests beyond
        # the cap wait in line at the proxy instead of piling onto the upstream. Caps are
        # auto-set from OLLAMA_NUM_PARALLEL when not specified explicitly:
        #   high   = unlimited (always go through)
        #   normal = OLLAMA_NUM_PARALLEL - 1 (leave one slot for high-priority traffic)
        #   low    = 1
        # Override `caps` to taste; null = unlimited. Priority is resolved per request:
        #   1. X-Priority header ('high'|'normal'|'low')
        #   2. client_priority map (e.g. claude-code → high, background-job → low)
        #   3. default_priority
        # `max_wait_s` caps how long a request will queue for a priority slot. After this,
        # the request proceeds without one — protecting streaming clients whose read
        # timeouts would otherwise fire before they see any bytes. Set to None for the
        # strict-queueing behavior (slot or wait forever).
        "enabled": False,
        "caps": {},  # {} = derive from OLLAMA_NUM_PARALLEL; explicit dict overrides
        "default_priority": "normal",
        "client_priority": {},  # e.g. {"claude-code": "high", "background-job": "low"}
        "max_wait_s": 0,
    },
    "compaction_nudge": {
        # When a request's estimated prompt size crosses `threshold_pct`% of the effective
        # num_ctx, nudge the client/model to compact. Strategy depends on client_app:
        #   "system_reminder"        → inject a <system-reminder> tag into the system msg
        #                              (Claude Code is trained to respect this tag strongly)
        #   "system_reminder_plain"  → inject a plain English reminder into the system msg
        #                              (works on any model; less reliable than the tag)
        #   "synthetic_response"     → short-circuit with a synthetic assistant message
        #                              telling the user to summarize/start a new chat
        #                              (only used for non-streaming; streams fall back to
        #                              system_reminder_plain)
        # Plus an `X-Proxy-Suggest: compact` response header on every nudged request, so
        # clients that read response headers can render their own banner.
        "enabled": False,
        "threshold_pct": 70,
        "chars_per_token": 3.5,
        "assumed_default_num_ctx": 32768,
        "default_strategy": "synthetic_response",
        "client_strategies": {
            "claude-code": "system_reminder",
            "vscode-copilot": "system_reminder_plain",
            "github-copilot": "system_reminder_plain",
            "cursor": "system_reminder_plain",
            "continue.dev": "system_reminder_plain",
            "ai-proxy-chat": "system_reminder_plain",
        },
    },
    "tool_injector": {
        # Adds proxy-owned tools to outgoing requests and executes them server-side. The model
        # sees them in tools[] like any other tool; when it calls one, the proxy intercepts
        # the response, runs the handler, appends a tool_result, and re-calls upstream so the
        # model sees the answer and continues. Capped at max_iterations to prevent runaway.
        #
        # Bundles:
        #   memory: per-conversation key-value store. Tools: remember, recall, list_memory, forget.
        #   todos:  per-conversation task list. Tools: set_todos, get_todos, add_todo, complete_todo.
        #
        # Per-client scopes: `scopes` is an evaluated-in-order list. The first entry whose
        # `match` clause matches the inbound request wins, and its overrides are merged onto
        # the root config. match keys (all AND'd; at least one required):
        #   ip:         exact client IP match (after X-Forwarded-For resolution)
        #   ip_cidr:    "10.0.0.0/24" / "fd00::/64" — client IP within network
        #   user_agent: case-insensitive substring of User-Agent header
        #   client_app: exact match against the detected label (e.g. "myapp", "claude-code")
        # Scope overrides may set: enabled, memory, todos, max_iterations.
        "enabled": False,
        "memory": True,
        "todos": True,
        "max_iterations": 4,
        # Example:
        #   "scopes": [
        #     {"match": {"client_app": "myapp"}, "enabled": true, "memory": true, "todos": false},
        #     {"match": {"ip_cidr": "10.0.0.0/24", "user_agent": "claude-code"}, "enabled": true},
        #   ]
        "scopes": [],
    },
}

RULES_FILE = _resolve_state_path(
    "PROXY_RULES_FILE", "rules.json", Path(__file__).parent / "rules.json", "rules.json"
)


def get_setting(key: str):
    conn = db()
    row = conn.execute("SELECT value, updated_ts FROM settings WHERE key = ?", (key,)).fetchone()
    conn.close()
    return dict(row) if row else None


def set_setting(key: str, value: str):
    conn = db()
    conn.execute(
        """INSERT INTO settings (key, value, updated_ts) VALUES (?, ?, ?)
           ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_ts=excluded.updated_ts""",
        (key, value, time.time()),
    )
    conn.commit()
    conn.close()


def delete_setting(key: str):
    conn = db()
    conn.execute("DELETE FROM settings WHERE key = ?", (key,))
    conn.commit()
    conn.close()


def _rules_source() -> tuple[str, str | None]:
    """Returns (source, raw_text). Source ∈ {'db', 'file', 'defaults'}.
    Migrates the legacy rules.json into the DB on first read so the file is no longer authoritative."""
    stored = get_setting("rules")
    if stored:
        return "db", stored["value"]
    p = Path(RULES_FILE)
    if p.exists():
        try:
            text = p.read_text()
            # One-time migration into DB.
            set_setting("rules", text)
            return "db", text
        except OSError:
            return "file", None
    return "defaults", None


def load_rules_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_RULES_CONFIG))
    _src, raw = _rules_source()
    if not raw:
        return cfg
    try:
        user_cfg = json.loads(raw)
    except json.JSONDecodeError:
        return cfg
    for k, v in (user_cfg or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def rule(name):
    def decorate(fn):
        RULES_REGISTRY[name] = fn
        return fn
    return decorate


def _normalize_args(args) -> str:
    """Stable string form of tool-call arguments for fuzzy comparison."""
    if args is None:
        return ""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return args.strip()[:1000]
    try:
        return json.dumps(args, sort_keys=True, separators=(",", ":"))[:1000]
    except (TypeError, ValueError):
        return str(args)[:1000]


@rule("loop_detector")
def _rule_loop_detector(body: dict, cfg: dict):
    """Block when the same tool call (name + normalized args) appears too often or consecutively in recent assistant turns."""
    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return None

    assistant_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
    window = max(1, int(cfg.get("window", 10)))
    recent = assistant_msgs[-window:]

    sigs: list[tuple[str, str]] = []
    for m in recent:
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            name = fn.get("name") or "?"
            args = _normalize_args(fn.get("arguments"))
            sigs.append((name, args))

    if not sigs:
        return None

    # Total count in window
    counts: dict = {}
    for s in sigs:
        counts[s] = counts.get(s, 0) + 1
    top_sig, top_count = max(counts.items(), key=lambda kv: kv[1])

    max_repeats = max(2, int(cfg.get("max_repeats", 4)))
    tail_n = max(2, int(cfg.get("tail_consecutive", 3)))

    if top_count >= max_repeats:
        return {
            "reason": f"Tool call {top_sig[0]!r} repeated {top_count}× in last {len(recent)} assistant turn(s)",
            "details": {
                "trigger": "count_in_window",
                "tool_name": top_sig[0],
                "count": top_count,
                "window_size": len(recent),
                "args_preview": top_sig[1][:300],
            },
        }

    # Consecutive tail
    if len(sigs) >= tail_n and all(s == sigs[-1] for s in sigs[-tail_n:]):
        return {
            "reason": f"Tool call {sigs[-1][0]!r} called {tail_n}× consecutively at end of conversation",
            "details": {
                "trigger": "tail_consecutive",
                "tool_name": sigs[-1][0],
                "count": tail_n,
                "args_preview": sigs[-1][1][:300],
            },
        }

    return None


@rule("tool_failure_breaker")
def _rule_tool_failure_breaker(body: dict, cfg: dict):
    """Block when the latest N tool results for the same tool name are all errors."""
    pairs = _tool_results_in_body(body)
    if not pairs:
        return None
    window = max(1, int(cfg.get("window", 5)))
    max_errors = max(1, int(cfg.get("max_errors", 3)))
    recent = pairs[-window:]
    # Group consecutive results by tool name from the tail.
    streaks: dict[str, int] = {}
    last_error_msg: dict[str, str | None] = {}
    for name, content in reversed(recent):
        is_err, excerpt = _is_tool_error(content)
        if not is_err:
            # Non-error breaks the streak for this tool.
            if name in streaks:
                continue  # already counted; don't extend
            continue
        streaks[name] = streaks.get(name, 0) + 1
        last_error_msg.setdefault(name, excerpt)
    for name, count in streaks.items():
        if count >= max_errors:
            return {
                "reason": f"Tool {name!r} has failed {count} consecutive times. Latest: {last_error_msg.get(name)!r}",
                "details": {
                    "tool_name": name,
                    "consecutive_errors": count,
                    "window": window,
                    "last_error": last_error_msg.get(name),
                },
            }
    return None


def _matches_type(value, type_name) -> bool:
    if isinstance(type_name, list):
        return any(_matches_type(value, t) for t in type_name)
    if type_name == "string": return isinstance(value, str)
    if type_name == "integer": return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number": return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean": return isinstance(value, bool)
    if type_name == "object": return isinstance(value, dict)
    if type_name == "array": return isinstance(value, list)
    if type_name == "null": return value is None
    return True


def _validate_args(args, schema: dict, strict_types: bool, reject_unknown: bool) -> list[str]:
    """Lightweight JSON Schema validation. Returns list of human-readable errors."""
    errs: list[str] = []
    if not isinstance(schema, dict):
        return errs
    if isinstance(args, str):
        if args.strip() == "":
            args = {}
        else:
            try:
                args = json.loads(args)
            except json.JSONDecodeError as e:
                return [f"arguments are not valid JSON: {e}"]
    if not isinstance(args, dict):
        return ["arguments must be a JSON object"]
    for r in (schema.get("required") or []):
        if r not in args:
            errs.append(f"missing required field {r!r}")
    properties = schema.get("properties") or {}
    for key, value in args.items():
        if key not in properties:
            if reject_unknown:
                errs.append(f"unknown field {key!r}")
            continue
        prop_schema = properties[key]
        if not isinstance(prop_schema, dict):
            continue
        expected = prop_schema.get("type")
        if expected and not _matches_type(value, expected):
            actual = type(value).__name__
            msg = f"field {key!r} should be {expected}, got {actual}"
            if strict_types:
                errs.append(msg)
    return errs


def _build_tool_schemas(request_body) -> dict[str, dict]:
    """name -> parameters schema, from the request's tools[]"""
    out: dict[str, dict] = {}
    if not isinstance(request_body, dict):
        return out
    for t in (request_body.get("tools") or []):
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or t
        if isinstance(fn, dict):
            name = fn.get("name")
            params = fn.get("parameters") or {}
            if isinstance(name, str):
                out[name] = params if isinstance(params, dict) else {}
    return out


def _validate_response_tool_calls(response_obj: dict, request_body) -> list[dict]:
    """Inspect a parsed OpenAI-format response (or Ollama native one) for tool_call problems.
    Returns list of {tool_name, arguments, errors:[...], kind:'hallucinated'|'invalid_args'}."""
    cfg = load_rules_config()
    sv = cfg.get("schema_validator") or {}
    ht = cfg.get("hallucinated_tool") or {}
    sv_on = bool(sv.get("enabled"))
    ht_on = bool(ht.get("enabled"))
    if not (sv_on or ht_on):
        return []

    tools = _build_tool_schemas(request_body)
    findings: list[dict] = []

    def inspect_tool_call(tc):
        if not isinstance(tc, dict):
            return
        fn = tc.get("function") or {}
        name = fn.get("name")
        args = fn.get("arguments")
        if not isinstance(name, str):
            return
        # Hallucination check: tools[] was declared and this name isn't in it.
        if ht_on and tools and name not in tools:
            findings.append({"tool_name": name, "arguments": args, "kind": "hallucinated", "errors": [f"tool {name!r} was not declared in tools[]"]})
            return
        if sv_on and name in tools:
            errs = _validate_args(args, tools[name], bool(sv.get("strict_types", True)), bool(sv.get("reject_unknown_fields", False)))
            if errs:
                findings.append({"tool_name": name, "arguments": args, "kind": "invalid_args", "errors": errs})

    if not isinstance(response_obj, dict):
        return findings

    for choice in (response_obj.get("choices") or []):
        msg = (choice.get("message") if isinstance(choice, dict) else None) or {}
        for tc in (msg.get("tool_calls") or []):
            inspect_tool_call(tc)
    # Ollama native /api/chat
    msg = response_obj.get("message")
    if isinstance(msg, dict):
        for tc in (msg.get("tool_calls") or []):
            inspect_tool_call(tc)
    return findings


def _assemble_streaming_response(stream_text: str) -> dict:
    """Reconstruct the final assistant message from an OpenAI SSE stream so we can validate it."""
    by_idx_choice: dict[int, dict] = {}
    completion_id = None
    model = None
    created = None
    for line in stream_text.split("\n"):
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]" or not data.strip():
            continue
        try:
            j = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(j, dict):
            continue
        completion_id = completion_id or j.get("id")
        model = model or j.get("model")
        created = created or j.get("created")
        for c in (j.get("choices") or []):
            idx = c.get("index", 0)
            slot = by_idx_choice.setdefault(idx, {"index": idx, "message": {"role": "assistant", "content": "", "tool_calls": {}}, "finish_reason": None})
            if c.get("finish_reason"):
                slot["finish_reason"] = c["finish_reason"]
            delta = c.get("delta") or {}
            if isinstance(delta.get("content"), str):
                slot["message"]["content"] += delta["content"]
            for tc in (delta.get("tool_calls") or []):
                tcidx = tc.get("index", len(slot["message"]["tool_calls"]))
                bucket = slot["message"]["tool_calls"].setdefault(tcidx, {"id": None, "type": "function", "function": {"name": None, "arguments": ""}})
                if tc.get("id"):
                    bucket["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    bucket["function"]["name"] = fn["name"]
                if isinstance(fn.get("arguments"), str):
                    bucket["function"]["arguments"] += fn["arguments"]
    # Flatten tool_calls dict to list ordered by index
    choices = []
    for idx in sorted(by_idx_choice):
        slot = by_idx_choice[idx]
        tcs = slot["message"].pop("tool_calls", {})
        if tcs:
            slot["message"]["tool_calls"] = [tcs[k] for k in sorted(tcs)]
        else:
            slot["message"].pop("tool_calls", None)
        if slot["message"]["content"] == "":
            slot["message"]["content"] = None
        choices.append(slot)
    return {
        "id": completion_id or f"proxy-synth-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": created or int(time.time()),
        "model": model,
        "choices": choices,
    }


def _format_intercept_message(findings: list[dict]) -> str:
    """Build the assistant content that replaces a bad tool-call response."""
    parts = ["[AI Proxy] Your last tool call was rejected before execution:"]
    for f in findings:
        if f["kind"] == "hallucinated":
            parts.append(f"- {f['tool_name']}: tool not declared in this request's tools[]. Pick a valid tool.")
        else:
            joined = "; ".join(f["errors"])
            parts.append(f"- {f['tool_name']}: {joined}.")
    parts.append("Please retry with corrected arguments. The original tool was not executed.")
    return "\n".join(parts)


def _synth_correction_response(message: str, completion_id: str, model: str | None) -> dict:
    """Build a non-streaming OpenAI-format response that delivers the correction message."""
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": message},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "x_proxy_intercepted": True,
    }


def _synth_correction_stream(message: str, completion_id: str, model: str | None) -> bytes:
    """Build a synthetic SSE stream that delivers the correction and ends cleanly."""
    base = {"id": completion_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
    role_chunk = dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}])
    text_chunk = dict(base, choices=[{"index": 0, "delta": {"content": message}, "finish_reason": None}])
    end_chunk = dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])
    out = (
        f"data: {json.dumps(role_chunk)}\n\n"
        f"data: {json.dumps(text_chunk)}\n\n"
        f"data: {json.dumps(end_chunk)}\n\n"
        "data: [DONE]\n\n"
    )
    return out.encode("utf-8")


def _post_flight_active(request_body) -> bool:
    """Quick check: is post-flight intercept (validate or autofix) relevant for this request?"""
    if not isinstance(request_body, dict):
        return False
    if not request_body.get("tools"):
        return False
    cfg = load_rules_config()
    sv = (cfg.get("schema_validator") or {}).get("enabled")
    ht = (cfg.get("hallucinated_tool") or {}).get("enabled")
    af = (cfg.get("tool_args_autofix") or {}).get("enabled")
    xa = (cfg.get("xml_autofix") or {}).get("enabled")
    return bool(sv or ht or af or xa)


_XML_TAG_RE = re.compile(r"<\s*/?\s*([A-Za-z][\w.\-:]*)\b[^>]*?(/?)>", re.DOTALL)
_XML_ENTITY_OR_AMP_RE = re.compile(r"&(?:#\d+|#x[0-9A-Fa-f]+|[A-Za-z]+);")


def _xml_detect_errors(text: str) -> list[dict]:
    """Best-effort scan for malformed XML in an assistant text response. Walks tags with a
    stack to find unclosed/mismatched. Reports unescaped `&` outside known entities.
    Returns a list of {kind, detail, ...}; empty if the content looks clean (or has too
    few XML tags to bother checking)."""
    if not text or not isinstance(text, str):
        return []
    cfg = (load_rules_config().get("xml_autofix") or {})
    min_tags = max(1, int(cfg.get("min_tags", 2) or 2))
    ignore = {str(t).lower() for t in (cfg.get("ignore_tags") or [])}
    tags = list(_XML_TAG_RE.finditer(text))
    if len(tags) < min_tags:
        return []
    out: list[dict] = []
    stack: list[tuple[str, int]] = []  # (tag_name, position)
    for m in tags:
        full = m.group(0)
        name = (m.group(1) or "").lower()
        self_close = m.group(2) == "/" or full.startswith("</")
        is_close = full.lstrip().startswith("</")
        is_self = m.group(2) == "/"
        if name in ignore or is_self:
            continue
        if is_close:
            if not stack:
                out.append({"kind": "stray_close", "tag": name, "pos": m.start()})
                continue
            top_name, _ = stack[-1]
            if top_name == name:
                stack.pop()
            else:
                out.append({"kind": "mismatched_close",
                            "expected": top_name, "got": name, "pos": m.start()})
                # Try to recover: if the close tag matches anything in the stack, unwind
                # everything above it. Otherwise treat as stray.
                if any(t == name for t, _ in stack):
                    while stack and stack[-1][0] != name:
                        unclosed_name, unclosed_pos = stack.pop()
                        out.append({"kind": "unclosed_tag", "tag": unclosed_name,
                                    "pos": unclosed_pos})
                    if stack:
                        stack.pop()
        else:
            stack.append((name, m.start()))
    for name, pos in stack:
        out.append({"kind": "unclosed_tag", "tag": name, "pos": pos})
    # Unescaped `&` not part of an entity. Common LLM mistake when emitting HTML/XML.
    if "&" in text:
        for idx, ch in enumerate(text):
            if ch != "&":
                continue
            tail = text[idx:idx + 12]
            if _XML_ENTITY_OR_AMP_RE.match(tail):
                continue
            # Skip if it's inside code-fences (rough heuristic — count fences before idx).
            fences_before = text[:idx].count("```")
            if fences_before % 2 == 1:
                continue
            out.append({"kind": "unescaped_amp", "pos": idx,
                        "context": text[max(0, idx-10):idx+20]})
    return out


def _extract_assistant_text(resp_obj: dict) -> str:
    """Pull the assistant's text content from an OpenAI or Anthropic response object."""
    if not isinstance(resp_obj, dict):
        return ""
    parts: list[str] = []
    for c in (resp_obj.get("choices") or []):
        msg = c.get("message") or {}
        if isinstance(msg.get("content"), str):
            parts.append(msg["content"])
    if resp_obj.get("type") == "message":
        for blk in (resp_obj.get("content") or []):
            if isinstance(blk, dict) and blk.get("type") == "text" and isinstance(blk.get("text"), str):
                parts.append(blk["text"])
    return "\n".join(parts)


def _xml_apply_fixes(text: str) -> tuple[str, list[dict]]:
    """Apply repairs to malformed XML text. Returns (fixed_text, fixes_list).
    Fixes applied:
      - unescaped & not part of a known entity → &amp;
      - unclosed tags → append closing tags in reverse stack order at end of text
    Stray-close and mismatched-close findings are noted but not auto-repaired — they
    usually indicate the model meant a different tag, and we don't want to guess."""
    if not text or not isinstance(text, str):
        return text, []
    fixes: list[dict] = []
    out = text
    # Step 1: escape stray &. Walk the string, replace each & that isn't already part of
    # an entity. Skip code fences to avoid mangling code that legitimately uses & in
    # operators (`a && b`, etc.).
    new_chars: list[str] = []
    i = 0
    in_fence = False
    amp_fixed = 0
    while i < len(out):
        # Toggle fence state on ``` boundaries.
        if out[i:i+3] == "```":
            in_fence = not in_fence
            new_chars.append("```")
            i += 3
            continue
        ch = out[i]
        if ch == "&" and not in_fence:
            if _XML_ENTITY_OR_AMP_RE.match(out[i:i+12]):
                new_chars.append("&")
            else:
                new_chars.append("&amp;")
                amp_fixed += 1
            i += 1
        else:
            new_chars.append(ch)
            i += 1
    if amp_fixed:
        out = "".join(new_chars)
        fixes.append({"kind": "escape_amp", "count": amp_fixed})

    # Step 2: append closing tags for unclosed. Re-walk after the & fixes (positions
    # changed). Build the same stack the detector uses, then append closes for what's
    # left at the end.
    cfg = (load_rules_config().get("xml_autofix") or {})
    ignore = {str(t).lower() for t in (cfg.get("ignore_tags") or [])}
    stack: list[str] = []
    for m in _XML_TAG_RE.finditer(out):
        full = m.group(0)
        name = (m.group(1) or "").lower()
        is_close = full.lstrip().startswith("</")
        is_self = m.group(2) == "/"
        if name in ignore or is_self:
            continue
        if is_close:
            # Pop matching tag if present (mismatches just get stray-closed; we don't try
            # to fix them, just unwind to keep our stack accurate for later tags).
            if name in stack:
                while stack and stack[-1] != name:
                    stack.pop()
                if stack:
                    stack.pop()
        else:
            stack.append(name)
    if stack:
        closes = "".join(f"</{t}>" for t in reversed(stack))
        out = out + closes
        fixes.append({"kind": "close_tags", "tags": list(reversed(stack))})
    return out, fixes


def _xml_fix_resp_obj(resp_obj: dict) -> list[dict]:
    """Apply XML fixes in place to assistant text within a response object. Returns the
    aggregated list of fixes applied across all text blocks (empty if nothing changed)."""
    if not isinstance(resp_obj, dict):
        return []
    all_fixes: list[dict] = []
    # OpenAI shape: choices[i].message.content
    for c in (resp_obj.get("choices") or []):
        msg = c.get("message") or {}
        if isinstance(msg.get("content"), str):
            fixed, applied = _xml_apply_fixes(msg["content"])
            if applied:
                msg["content"] = fixed
                all_fixes.extend(applied)
    # Anthropic shape: content[i] where type=='text'
    if resp_obj.get("type") == "message":
        for blk in (resp_obj.get("content") or []):
            if isinstance(blk, dict) and blk.get("type") == "text" and isinstance(blk.get("text"), str):
                fixed, applied = _xml_apply_fixes(blk["text"])
                if applied:
                    blk["text"] = fixed
                    all_fixes.extend(applied)
    return all_fixes


def _autofix_tool_calls(response_obj: dict, request_body) -> list[dict]:
    """For each tool_call, fill in missing required fields from configured defaults.
    Mutates function.arguments in place. Returns list of {tool_name, fixed_fields}."""
    cfg = (load_rules_config().get("tool_args_autofix") or {})
    if not cfg.get("enabled"):
        return []
    user_defaults = cfg.get("defaults") or {}
    tools = _build_tool_schemas(request_body)
    fixes: list[dict] = []

    def fix_call(tc):
        if not isinstance(tc, dict):
            return
        fn = tc.get("function") or {}
        name = fn.get("name")
        if not isinstance(name, str):
            return
        schema = tools.get(name) or {}
        required = schema.get("required") or []
        properties = schema.get("properties") or {}

        args_raw = fn.get("arguments")
        try:
            if isinstance(args_raw, str):
                args = json.loads(args_raw) if args_raw.strip() else {}
            else:
                args = args_raw if isinstance(args_raw, dict) else {}
        except json.JSONDecodeError:
            return  # unparseable JSON — leave for schema_validator to surface
        if not isinstance(args, dict):
            return

        per_tool = user_defaults.get(name) or {}
        wildcard = user_defaults.get("*") or {}

        applied: dict = {}
        for r in required:
            if r in args:
                continue
            if r in per_tool:
                args[r] = per_tool[r]; applied[r] = per_tool[r]
            elif r in wildcard:
                args[r] = wildcard[r]; applied[r] = wildcard[r]
            elif "default" in (properties.get(r) or {}):
                args[r] = properties[r]["default"]; applied[r] = properties[r]["default"]
        if applied:
            fn["arguments"] = json.dumps(args)
            fixes.append({"tool_name": name, "fixed_fields": applied})

    if isinstance(response_obj, dict):
        for c in (response_obj.get("choices") or []):
            msg = (c.get("message") if isinstance(c, dict) else None) or {}
            for tc in (msg.get("tool_calls") or []):
                fix_call(tc)
        msg = response_obj.get("message")
        if isinstance(msg, dict):
            for tc in (msg.get("tool_calls") or []):
                fix_call(tc)
    return fixes


def _synth_response_stream(response_obj: dict) -> bytes:
    """Convert a non-streaming OpenAI-format response into a self-contained SSE chunk sequence.
    Used when we've mutated the response (e.g. via autofix) and need to emit the fixed version
    to a client that originally requested streaming."""
    base_id = response_obj.get("id") or f"proxy-{uuid.uuid4().hex[:12]}"
    base_model = response_obj.get("model")
    base_created = response_obj.get("created") or int(time.time())
    base = {"id": base_id, "object": "chat.completion.chunk", "created": base_created, "model": base_model}
    out_chunks: list[str] = []

    def emit(delta, idx, finish=None):
        chunk = dict(base, choices=[{"index": idx, "delta": delta, "finish_reason": finish}])
        out_chunks.append(f"data: {json.dumps(chunk)}\n\n")

    for c in (response_obj.get("choices") or []):
        if not isinstance(c, dict):
            continue
        idx = c.get("index", 0)
        msg = c.get("message") or {}
        finish = c.get("finish_reason") or "stop"
        emit({"role": "assistant"}, idx)
        if isinstance(msg.get("content"), str) and msg["content"]:
            emit({"content": msg["content"]}, idx)
        for i, tc in enumerate(msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            emit({
                "tool_calls": [{
                    "index": i,
                    "id": tc.get("id"),
                    "type": tc.get("type", "function"),
                    "function": {"name": fn.get("name"), "arguments": fn.get("arguments", "")},
                }]
            }, idx)
        emit({}, idx, finish)

    out_chunks.append("data: [DONE]\n\n")
    return "".join(out_chunks).encode("utf-8")


def _prompt_total_chars(body: dict) -> int:
    n = 0
    for m in body.get("messages") or []:
        c = m.get("content")
        if isinstance(c, str):
            n += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    n += len(part["text"])
    if isinstance(body.get("prompt"), str):
        n += len(body["prompt"])
    return n


def _body_has_images(body: dict) -> bool:
    """True if any message carries image content. Handles OpenAI multimodal
    (content list with {"type":"image_url"}) and Anthropic ({"type":"image"}).
    Lets the router send vision requests to a multimodal model instead of a
    text-only one (which would 400 with 'model does not support images')."""
    if not isinstance(body, dict):
        return False
    for m in (body.get("messages") or []):
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image", "input_image"):
                    return True
    return False


# Base64 image blobs sometimes arrive embedded in a tool's TEXT output rather than as a standard
# vision part — e.g. a screen-capture tool returning {"screenshot_png_b64": "iVBORw0KGgo…"}. A text
# model can't decode these (and they must NOT trigger vision routing — that's why this is kept out
# of _body_has_images), but the dashboard can still surface them. We spot them by the PNG
# (iVBORw0KGgo…) and JPEG (/9j/…) base64 magic prefixes and pull the contiguous run.
_EMBEDDED_B64_IMG_RE = re.compile(r"iVBORw0KGgo[A-Za-z0-9+/]{200,}={0,2}|/9j/[A-Za-z0-9+/]{200,}={0,2}")


def _embedded_b64_images(s):
    """Yield (media_type, base64) for each PNG/JPEG blob embedded in a string message body."""
    if not isinstance(s, str) or ("iVBORw0KGgo" not in s and "/9j/" not in s):
        return
    for mm in _EMBEDDED_B64_IMG_RE.finditer(s):
        blob = mm.group(0)
        yield ("image/png" if blob[0] == "i" else "image/jpeg", blob)


def _iter_request_images(body: dict):
    """Yield (index, media_type, kind, payload) for each image in a chat request, in order.
    kind='data' → payload is base64 (from a data: URL or Anthropic source); kind='url' →
    payload is an external URL. Handles OpenAI (image_url) and Anthropic (image) shapes."""
    if not isinstance(body, dict):
        return
    idx = 0
    for m in (body.get("messages") or []):
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            for (mt, blob) in _embedded_b64_images(c):
                yield (idx, mt, "data", blob); idx += 1
            continue
        if not isinstance(c, list):
            continue
        for part in c:
            if not isinstance(part, dict):
                continue
            t = part.get("type")
            if t == "image_url":
                url = (part.get("image_url") or {}).get("url") or ""
                if url.startswith("data:"):
                    head, _, b64 = url.partition(",")
                    mt = "image/png"
                    mm = re.match(r"data:([^;,]+)", head)
                    if mm:
                        mt = mm.group(1)
                    yield (idx, mt, "data", b64); idx += 1
                elif url:
                    yield (idx, "", "url", url); idx += 1
            elif t in ("image", "input_image"):
                src = part.get("source") if isinstance(part.get("source"), dict) else {}
                if src.get("data"):
                    yield (idx, src.get("media_type") or "image/png", "data", src["data"]); idx += 1
                elif src.get("url") or part.get("image_url"):
                    yield (idx, "", "url", src.get("url") or part.get("image_url")); idx += 1


def _strip_image_data(body: dict) -> int:
    """Replace inline base64 image payloads with a short placeholder so the DISPLAYED body isn't
    a multi-hundred-KB blob. Mutates in place; returns how many were replaced. The actual bytes
    are still reconstructable from the untouched DB row via the image endpoint."""
    n = 0
    if not isinstance(body, dict):
        return 0
    for m in (body.get("messages") or []):
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            if "iVBORw0KGgo" in c or "/9j/" in c:
                new, cnt = _EMBEDDED_B64_IMG_RE.subn(
                    lambda mm: f"…[{len(mm.group(0)) * 3 // 4} bytes image — see Images section]", c)
                if cnt:
                    m["content"] = new
                    n += cnt
            continue
        if not isinstance(c, list):
            continue
        for part in c:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                iu = part.get("image_url")
                url = (iu or {}).get("url") or ""
                if isinstance(iu, dict) and url.startswith("data:"):
                    head, _, b64 = url.partition(",")
                    iu["url"] = f"{head},…[{len(b64) * 3 // 4} bytes — see Images section]"
                    n += 1
            elif part.get("type") in ("image", "input_image"):
                src = part.get("source") if isinstance(part.get("source"), dict) else None
                if src and src.get("data"):
                    src["data"] = f"…[{len(src['data']) * 3 // 4} bytes — see Images section]"
                    n += 1
    return n


def _load_images_data(images_data_json):
    """Parse the images_data column (JSON array of {media_type, data}) → list, or []."""
    if not images_data_json:
        return []
    try:
        v = json.loads(images_data_json)
        return v if isinstance(v, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _match_router_cond(cond: dict, body: dict, ctx: dict) -> bool:
    """Return True iff every key in `cond` matches the request."""
    if not isinstance(cond, dict):
        return False

    def as_list(v):
        return v if isinstance(v, list) else [v]

    if "from_model" in cond:
        if (body.get("model") or "") not in as_list(cond["from_model"]):
            return False
    if "from_model_prefix" in cond:
        m = body.get("model") or ""
        if not any(m.startswith(str(p)) for p in as_list(cond["from_model_prefix"])):
            return False
    if "from_client" in cond:
        if (ctx.get("client_ip") or "") not in as_list(cond["from_client"]):
            return False
    if "from_client_prefix" in cond:
        ip = ctx.get("client_ip") or ""
        if not ip.startswith(str(cond["from_client_prefix"])):
            return False
    if "path_prefix" in cond:
        if not (ctx.get("path") or "").startswith(str(cond["path_prefix"])):
            return False
    if "has_tools" in cond:
        actual = bool(body.get("tools"))
        if bool(cond["has_tools"]) != actual:
            return False
    if "has_images" in cond:
        if bool(cond["has_images"]) != _body_has_images(body):
            return False
    if "prompt_chars_lt" in cond or "prompt_chars_gt" in cond:
        n = _prompt_total_chars(body)
        if "prompt_chars_lt" in cond and not (n < int(cond["prompt_chars_lt"])):
            return False
        if "prompt_chars_gt" in cond and not (n > int(cond["prompt_chars_gt"])):
            return False
    return True


def evaluate_router(body: dict, ctx: dict) -> dict | None:
    """Apply model_router config. Returns {from, to, via, condition} or None.
    Mutates `body['model']` in place when a rewrite fires."""
    if not isinstance(body, dict):
        return None
    cfg = (load_rules_config().get("model_router") or {})
    if not cfg.get("enabled", False):
        return None
    original = body.get("model")
    if not original:
        return None

    target = None
    via = None
    matched_cond = None
    upstream = None  # optional per-rule upstream override ("ollama" | "lmstudio")

    aliases = cfg.get("aliases") or {}
    if isinstance(aliases, dict) and original in aliases:
        target = aliases[original]
        via = "alias"

    for r in (cfg.get("rules") or []):
        if not isinstance(r, dict):
            continue
        cond = r.get("if") or {}
        # Evaluate against the (possibly already aliased) name so chains work intuitively.
        body_for_match = dict(body)
        body_for_match["model"] = target or original
        if _match_router_cond(cond, body_for_match, ctx):
            target = r.get("then")
            via = "rule"
            matched_cond = cond
            upstream = r.get("upstream")  # e.g. "lmstudio" to send this model elsewhere
            break

    # A rule may switch upstream without renaming the model, so a bare upstream override
    # (target == original) still counts as a rewrite.
    if not target:
        return None
    if target == original and not upstream:
        return None
    body["model"] = target
    return {"from": original, "to": target, "via": via, "condition": matched_cond,
            "upstream": upstream}


_NATIVE_TOP_LEVEL_KEYS = {"keep_alive", "format"}


def _cap_num_ctx(body: dict, cap: int) -> int | None:
    """Clamp a request's num_ctx down to `cap`, wherever it sits — top-level (OpenAI-compat
    /v1/*) or nested under options (Ollama-native /api/*). Overrides a client-set value
    (unlike ollama_options' fill-in-only injection). Returns the new value if it changed
    anything, else None.

    Purpose: a client (e.g. an agent that auto-sets num_ctx to the model's full training
    context, like 262144) can otherwise force an enormous KV cache that slows every request
    and starves parallelism. This caps that centrally.
    """
    if not isinstance(body, dict) or not isinstance(cap, int) or cap <= 0:
        return None
    changed = None
    v = body.get("num_ctx")
    if isinstance(v, int) and v > cap:
        body["num_ctx"] = cap
        changed = cap
    opts = body.get("options")
    if isinstance(opts, dict):
        ov = opts.get("num_ctx")
        if isinstance(ov, int) and ov > cap:
            opts["num_ctx"] = cap
            changed = cap
    return changed


def evaluate_ollama_options(body: dict, ctx: dict) -> dict | None:
    """Inject generation parameters per the ollama_options config. Mutates body in place.
    Never overwrites a value the client already set. Returns {applied:{...}, sources:[...]} or None."""
    if not isinstance(body, dict):
        return None
    cfg = (load_rules_config().get("ollama_options") or {})
    if not cfg.get("enabled", False):
        return None

    candidate: dict = {}
    sources: list[str] = []

    defaults = cfg.get("defaults") or {}
    if isinstance(defaults, dict) and defaults:
        for k, v in defaults.items():
            candidate.setdefault(k, v)
        sources.append("defaults")

    per_model = (cfg.get("per_model") or {}).get(body.get("model"))
    if isinstance(per_model, dict) and per_model:
        for k, v in per_model.items():
            candidate[k] = v  # per-model wins over global defaults
        sources.append(f"per_model:{body.get('model')}")

    per_client = (cfg.get("per_client") or {}).get(ctx.get("client_ip") or "")
    if isinstance(per_client, dict) and per_client:
        for k, v in per_client.items():
            candidate[k] = v
        sources.append(f"per_client:{ctx.get('client_ip')}")

    for r in (cfg.get("rules") or []):
        if not isinstance(r, dict):
            continue
        if _match_router_cond(r.get("if") or {}, body, ctx):
            for k, v in (r.get("set") or {}).items():
                candidate[k] = v
            sources.append("rule")
            break

    if not candidate:
        return None

    is_native = "/api/" in (ctx.get("path") or "")
    applied: dict = {}

    if is_native:
        if not isinstance(body.get("options"), dict):
            body["options"] = {}
        for k, v in candidate.items():
            if k in _NATIVE_TOP_LEVEL_KEYS:
                if k not in body:
                    body[k] = v
                    applied[k] = v
            else:
                if k not in body["options"]:
                    body["options"][k] = v
                    applied[k] = v
    else:
        for k, v in candidate.items():
            if k in _NATIVE_TOP_LEVEL_KEYS:
                continue  # not applicable to OpenAI-compat path
            if k not in body:
                body[k] = v
                applied[k] = v

    if not applied:
        return None
    return {"applied": applied, "sources": sources}


def _estimate_prompt_tokens(body: dict, chars_per_token: float) -> int:
    """Cheap heuristic: count chars across messages, tools, prompt/system, divide by chars_per_token.
    Adds a small per-message overhead for chat formatting tokens. Off by ±15% vs tiktoken; good enough
    for an overflow check. Handles OpenAI/Ollama and Anthropic shapes."""
    if not isinstance(body, dict):
        return 0
    total_chars = 0
    msgs = body.get("messages") or []
    msg_count = 0
    for m in msgs:
        if not isinstance(m, dict):
            continue
        msg_count += 1
        c = m.get("content")
        if isinstance(c, str):
            total_chars += len(c)
        elif isinstance(c, list):
            for p in c:
                if not isinstance(p, dict):
                    continue
                if isinstance(p.get("text"), str):
                    total_chars += len(p["text"])
                t = p.get("type")
                if t == "tool_use" and p.get("input") is not None:
                    try:
                        total_chars += len(json.dumps(p["input"]))
                    except (TypeError, ValueError):
                        pass
                elif t == "tool_result":
                    inner = p.get("content")
                    if isinstance(inner, str):
                        total_chars += len(inner)
                    elif isinstance(inner, list):
                        for ip in inner:
                            if isinstance(ip, dict) and isinstance(ip.get("text"), str):
                                total_chars += len(ip["text"])
        for tc in (m.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if isinstance(fn.get("name"), str):
                total_chars += len(fn["name"])
            args = fn.get("arguments")
            if isinstance(args, str):
                total_chars += len(args)
            elif args is not None:
                try:
                    total_chars += len(json.dumps(args))
                except (TypeError, ValueError):
                    pass
    for t in (body.get("tools") or []):
        if isinstance(t, dict):
            try:
                total_chars += len(json.dumps(t))
            except (TypeError, ValueError):
                pass
    if isinstance(body.get("prompt"), str):
        total_chars += len(body["prompt"])
    sys_field = body.get("system")
    if isinstance(sys_field, str):
        total_chars += len(sys_field)
    elif isinstance(sys_field, list):
        for p in sys_field:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                total_chars += len(p["text"])
    if total_chars == 0:
        return 0
    return int(total_chars / max(0.5, chars_per_token)) + 4 * msg_count


def _get_num_ctx(body: dict, is_native: bool) -> int | None:
    if is_native:
        opts = body.get("options")
        if isinstance(opts, dict):
            v = opts.get("num_ctx")
            if isinstance(v, int):
                return v
    v = body.get("num_ctx")
    if isinstance(v, int):
        return v
    return None


def _set_num_ctx(body: dict, is_native: bool, value: int) -> None:
    if is_native:
        if not isinstance(body.get("options"), dict):
            body["options"] = {}
        body["options"]["num_ctx"] = value
    else:
        body["num_ctx"] = value


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _trim_messages_to_fit(body: dict, est_tokens: int, target_tokens: int, chars_per_token: float, min_keep: int) -> tuple[int, int]:
    """Drop oldest non-system messages until estimated tokens fit under `target_tokens`.
    Always preserves system messages and the last `min_keep` non-system messages.
    Inserts one synthetic system note explaining the trim. Returns (new_estimated_tokens, count_trimmed)."""
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return est_tokens, 0
    non_system = [i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") != "system"]
    if len(non_system) <= min_keep:
        return est_tokens, 0
    trimmable = non_system[:-min_keep]
    if not trimmable:
        return est_tokens, 0
    drop_set: set[int] = set()
    current_tokens = est_tokens
    for idx in trimmable:
        m = msgs[idx]
        c = m.get("content")
        chars = 0
        if isinstance(c, str):
            chars = len(c)
        elif isinstance(c, list):
            chars = sum(len(p.get("text", "")) for p in c if isinstance(p, dict) and isinstance(p.get("text"), str))
        for tc in (m.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                chars += len(args)
            elif args is not None:
                try:
                    chars += len(json.dumps(args))
                except (TypeError, ValueError):
                    pass
        msg_tokens = int(chars / max(0.5, chars_per_token)) + 4
        drop_set.add(idx)
        current_tokens -= msg_tokens
        if current_tokens <= target_tokens:
            break
    if not drop_set:
        return est_tokens, 0
    placeholder = {
        "role": "system",
        "content": f"[AI Proxy] {len(drop_set)} earlier message(s) trimmed to fit num_ctx. The conversation continues below.",
    }
    new_msgs: list = []
    placeholder_inserted = False
    for i, m in enumerate(msgs):
        if i in drop_set:
            continue
        if not placeholder_inserted and isinstance(m, dict) and m.get("role") != "system":
            new_msgs.append(placeholder)
            placeholder_inserted = True
        new_msgs.append(m)
    if not placeholder_inserted:
        new_msgs.append(placeholder)
    body["messages"] = new_msgs
    return current_tokens, len(drop_set)


def evaluate_context_overflow(body, ctx) -> dict | None:
    """Detect prompt-token overflow vs effective num_ctx; warn / bump / trim / block per config.
    Mutates body in place when action is bump or trim. Returns a result dict (or None when no action)."""
    if not isinstance(body, dict):
        return None
    cfg = (load_rules_config().get("context_overflow_guard") or {})
    if not cfg.get("enabled", False):
        return None
    action = (cfg.get("action") or "warn").lower()
    if action not in ("warn", "bump", "trim", "block"):
        action = "warn"
    chars_per_token = float(cfg.get("chars_per_token", 3.5) or 3.5)
    headroom_ratio = float(cfg.get("headroom_ratio", 0.95) or 0.95)
    if not (0.1 < headroom_ratio <= 1.0):
        headroom_ratio = 0.95
    max_ctx = int(cfg.get("max_ctx", 131072) or 131072)
    bump_to = cfg.get("bump_to")
    min_keep = max(1, int(cfg.get("min_keep_messages", 4) or 4))

    is_native = "/api/" in (ctx.get("path") or "")
    est_tokens = _estimate_prompt_tokens(body, chars_per_token)
    if est_tokens <= 0:
        return None

    num_ctx = _get_num_ctx(body, is_native)
    if num_ctx is None:
        num_ctx = int(cfg.get("assumed_default_num_ctx", 4096) or 4096)
    threshold = int(num_ctx * headroom_ratio)
    if est_tokens <= threshold:
        return None

    base = {
        "estimated_tokens": est_tokens,
        "num_ctx": num_ctx,
        "headroom_ratio": headroom_ratio,
        "chars_per_token": chars_per_token,
        "threshold": threshold,
    }

    if action == "bump":
        target = int(bump_to) if isinstance(bump_to, (int, float)) and bump_to else _next_pow2(int(est_tokens / headroom_ratio) + 1)
        target = min(target, max_ctx)
        if target <= num_ctx:
            return {**base, "action": "warn",
                    "reason": f"prompt ~{est_tokens} tok exceeds num_ctx={num_ctx}; bump capped at max_ctx={max_ctx}"}
        _set_num_ctx(body, is_native, target)
        return {**base, "action": "bump", "num_ctx_after": target,
                "reason": f"bumped num_ctx {num_ctx} → {target} for prompt of ~{est_tokens} tok"}

    if action == "trim":
        kept_tokens, trimmed = _trim_messages_to_fit(body, est_tokens, threshold, chars_per_token, min_keep)
        if trimmed == 0:
            return {**base, "action": "warn",
                    "reason": f"prompt ~{est_tokens} tok exceeds num_ctx={num_ctx}; nothing safe to trim (min_keep_messages={min_keep})"}
        return {**base, "action": "trim", "trimmed_count": trimmed, "estimated_tokens_after": kept_tokens,
                "reason": f"trimmed {trimmed} oldest message(s) to fit num_ctx={num_ctx} (~{est_tokens} → ~{kept_tokens} tok)"}

    if action == "block":
        return {**base, "action": "block",
                "reason": f"prompt ~{est_tokens} tok exceeds num_ctx={num_ctx} ({int(headroom_ratio*100)}% threshold = {threshold}); blocked to prevent silent truncation"}

    return {**base, "action": "warn",
            "reason": f"prompt ~{est_tokens} tok exceeds num_ctx={num_ctx} (will be silently truncated by Ollama)"}


def evaluate_compaction_nudge(body: dict, ctx: dict) -> dict | None:
    """When the prompt is creeping toward num_ctx, return an action describing how to nudge
    the client/model to compact. Strategy depends on client_app — Claude Code respects the
    <system-reminder> tag strongly; other clients get either a plain reminder or (when not
    streaming) a synthetic assistant response asking the user to summarize."""
    if not isinstance(body, dict):
        return None
    cfg = load_rules_config().get("compaction_nudge") or {}
    if not cfg.get("enabled", False):
        return None
    threshold_pct = float(cfg.get("threshold_pct", 70) or 70)
    chars_per_token = float(cfg.get("chars_per_token", 3.5) or 3.5)
    is_native = "/api/" in (ctx.get("path") or "")
    est = _estimate_prompt_tokens(body, chars_per_token)
    if est <= 0:
        return None
    num_ctx = _get_num_ctx(body, is_native)
    if num_ctx is None:
        num_ctx = int(cfg.get("assumed_default_num_ctx", 32768) or 32768)
    threshold = int(num_ctx * threshold_pct / 100.0)
    if est < threshold:
        return None
    client_app = ctx.get("client_app") or "unknown"
    strategies = cfg.get("client_strategies") or {}
    strategy = strategies.get(client_app, cfg.get("default_strategy") or "synthetic_response")
    is_stream = bool(body.get("stream"))
    if strategy == "synthetic_response" and is_stream:
        strategy = "system_reminder_plain"  # synthetic only works for non-streaming
    pct = int(est * 100 / max(1, num_ctx))
    base = {
        "estimated_tokens": est, "num_ctx": num_ctx, "threshold_pct": threshold_pct,
        "client_app": client_app, "pct": pct, "strategy": strategy,
    }
    if strategy == "system_reminder":
        text = (
            f"<system-reminder>This conversation has used ~{est} tokens "
            f"(~{pct}% of the {num_ctx}-token context window). Strongly suggest the user "
            f"run /compact to summarize older turns before continuing, or start a new chat.</system-reminder>"
        )
        return {**base, "action": "system_reminder", "text": text,
                "reason": f"prompt ~{est} tok ({pct}%) — injecting <system-reminder> for {client_app}"}
    if strategy == "system_reminder_plain":
        text = (
            f"NOTE: this conversation has used ~{est} tokens (~{pct}% of the available "
            f"context). Before continuing, advise the user to summarize earlier turns or "
            f"start a fresh conversation. Long conversations cause silent truncation and "
            f"degrade response quality."
        )
        return {**base, "action": "system_reminder", "text": text,
                "reason": f"prompt ~{est} tok ({pct}%) — injecting plain reminder for {client_app}"}
    if strategy == "synthetic_response":
        msg = (
            f"⚠ This conversation has grown to ~{est} tokens (~{pct}% of the context window). "
            f"To preserve response quality, please summarize what we've covered so far and "
            f"start a new chat with that summary — or use your client's compact/summarize "
            f"feature. (This message comes from the AI Proxy's compaction_nudge rule, not "
            f"from the model.)"
        )
        return {**base, "action": "synthetic_response", "text": msg,
                "reason": f"prompt ~{est} tok ({pct}%) — synthetic nudge for {client_app}"}
    return None


def _inject_system_reminder(body: dict, reminder: str) -> bool:
    """Append `reminder` to the body's system message. Returns True if mutated. Handles
    OpenAI shape (messages[0] with role='system') and Anthropic shape (body['system'] as
    str or list of content blocks)."""
    if not isinstance(body, dict) or not reminder:
        return False
    # Anthropic shape: top-level `system`
    if "system" in body or "messages" in body and isinstance(body.get("system"), (str, list)):
        sys_field = body.get("system")
        if isinstance(sys_field, str):
            body["system"] = sys_field.rstrip() + "\n\n" + reminder
            return True
        if isinstance(sys_field, list):
            sys_field.append({"type": "text", "text": reminder})
            return True
    # OpenAI shape: prepend or append to first system message in messages[]
    msgs = body.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "system":
                c = m.get("content")
                if isinstance(c, str):
                    m["content"] = c.rstrip() + "\n\n" + reminder
                    return True
                if isinstance(c, list):
                    c.append({"type": "text", "text": reminder})
                    return True
        # No system message — insert one at the front.
        msgs.insert(0, {"role": "system", "content": reminder})
        return True
    return False


def _build_synthetic_response(body: dict, text: str) -> tuple[bytes, str]:
    """Build a response body in the upstream's shape that looks like a normal assistant
    message containing `text`. Returns (raw_bytes, content_type)."""
    is_anthropic = isinstance(body, dict) and ("system" in body or "max_tokens" in body and "messages" in body and "model" in body and not body.get("messages") is None)
    # Heuristic: presence of `system` or `max_tokens` typically marks Anthropic shape.
    if isinstance(body, dict) and ("system" in body or ("max_tokens" in body and "tools" not in body and "stream_options" not in body)):
        resp = {
            "id": "msg_proxy_" + uuid.uuid4().hex[:16],
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": body.get("model") or "proxy-synthetic",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    else:
        resp = {
            "id": "chatcmpl-proxy-" + uuid.uuid4().hex[:12],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model") if isinstance(body, dict) else "proxy-synthetic",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    return json.dumps(resp).encode("utf-8"), "application/json"


def _tool_names_from_body(body) -> list[str]:
    """Names of tools declared in body['tools']. Handles OpenAI (tools[i].function.name) and
    Anthropic (tools[i].name)."""
    out: list[str] = []
    for t in (body.get("tools") or []) if isinstance(body, dict) else []:
        if not isinstance(t, dict):
            continue
        # OpenAI: nested under "function"; Anthropic: name lives at the top
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            out.append(fn["name"])
    return out


def _conversation_tool_history(conv_id: str, exclude_req_id: str | None) -> tuple[dict[str, int], dict[str, int], int]:
    """Walk this conversation's prior requests; return (offered_counts, invoked_counts, prior_turn_count).
    `offered_counts[tool] = N` is how many prior turns declared this tool in tools[].
    `invoked_counts[tool] = M` is how many prior turns actually called the tool in their response."""
    offered: dict[str, int] = {}
    invoked: dict[str, int] = {}
    if not conv_id:
        return offered, invoked, 0
    conn = db()
    try:
        if exclude_req_id:
            rows = conn.execute(
                "SELECT id, request_body, response_body, stream_chunks FROM requests_v "
                "WHERE conversation_id = ? AND id != ? ORDER BY ts ASC",
                (conv_id, exclude_req_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, request_body, response_body, stream_chunks FROM requests_v "
                "WHERE conversation_id = ? ORDER BY ts ASC",
                (conv_id,),
            ).fetchall()
    finally:
        conn.close()
    turn_count = 0
    for r in rows:
        body = None
        try:
            body = json.loads(r["request_body"] or "")
        except (json.JSONDecodeError, TypeError):
            pass
        if not isinstance(body, dict):
            continue
        turn_count += 1
        for name in _tool_names_from_body(body):
            offered[name] = offered.get(name, 0) + 1
        for name in _extract_tool_calls(r["response_body"], r["stream_chunks"]):
            invoked[name] = invoked.get(name, 0) + 1
    return offered, invoked, turn_count


def evaluate_tool_pruner(body, ctx) -> dict | None:
    """Drop tool definitions that have been offered repeatedly in this conversation but never
    invoked. Mutates body['tools'] in place when action is 'prune'. Returns audit info or None."""
    if not isinstance(body, dict):
        return None
    cfg = (load_rules_config().get("tool_pruner") or {})
    if not cfg.get("enabled", False):
        return None
    tools = body.get("tools")
    if not isinstance(tools, list) or len(tools) < 2:
        return None
    conv_id = _conversation_id(body)
    if not conv_id:
        return None

    action = (cfg.get("action") or "prune").lower()
    if action not in ("prune", "warn"):
        action = "prune"
    min_turns_offered = max(1, int(cfg.get("min_turns_offered", 3) or 3))
    min_history_turns = max(0, int(cfg.get("min_history_turns", 2) or 2))
    always_keep = set(cfg.get("always_keep") or [])
    max_prune_ratio = float(cfg.get("max_prune_ratio", 0.8) or 0.8)
    if not (0.0 < max_prune_ratio <= 1.0):
        max_prune_ratio = 0.8
    include_hint = bool(cfg.get("include_hint", True))

    offered, invoked, prior_turns = _conversation_tool_history(conv_id, ctx.get("req_id"))
    if prior_turns < min_history_turns:
        return None

    current_names = _tool_names_from_body(body)
    candidates = []
    for name in current_names:
        if name in always_keep:
            continue
        if invoked.get(name, 0) > 0:
            continue
        if offered.get(name, 0) < min_turns_offered:
            continue
        candidates.append(name)

    if not candidates:
        return None

    # Cap by max_prune_ratio: keep at least ceil(len(tools) * (1 - max_prune_ratio)) tools.
    keep_floor = max(1, math.ceil(len(tools) * (1.0 - max_prune_ratio)))
    max_drop = max(0, len(tools) - keep_floor)
    if max_drop <= 0:
        return None
    pruned_names = candidates[:max_drop]
    if not pruned_names:
        return None

    base = {
        "conversation_id": conv_id,
        "prior_turns": prior_turns,
        "tools_before": len(tools),
        "candidates": candidates,
        "pruned": pruned_names,
        "kept": [n for n in current_names if n not in pruned_names],
        "max_prune_ratio": max_prune_ratio,
    }

    if action == "warn":
        return {**base, "action": "warn",
                "reason": f"would prune {len(pruned_names)} of {len(tools)} tools never invoked in {prior_turns} prior turns: {', '.join(pruned_names[:6])}{'…' if len(pruned_names) > 6 else ''}"}

    pruned_set = set(pruned_names)
    new_tools = []
    for t in tools:
        if not isinstance(t, dict):
            new_tools.append(t)
            continue
        fn = t.get("function") or t
        name = fn.get("name") if isinstance(fn, dict) else None
        if name in pruned_set:
            continue
        new_tools.append(t)
    body["tools"] = new_tools

    if include_hint:
        msgs = body.get("messages")
        if isinstance(msgs, list):
            hint = {
                "role": "system",
                "content": (
                    f"[AI Proxy] Pruned {len(pruned_names)} tool(s) unused in this conversation: "
                    f"{', '.join(pruned_names)}. Re-request them if needed."
                ),
            }
            insert_at = 0
            for i, m in enumerate(msgs):
                if isinstance(m, dict) and m.get("role") == "system":
                    insert_at = i + 1
                else:
                    break
            msgs.insert(insert_at, hint)

    return {**base, "action": "prune", "tools_after": len(new_tools),
            "reason": f"pruned {len(pruned_names)} of {base['tools_before']} tool(s) unused after {prior_turns} prior turns: {', '.join(pruned_names[:6])}{'…' if len(pruned_names) > 6 else ''}"}


def _looks_complex_prompt(body) -> bool:
    """Quick heuristics: does this prompt look 'hard' for a small model to handle?"""
    if not isinstance(body, dict):
        return False
    has_code = False
    total = 0
    msgs = body.get("messages") or []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
            if "```" in c:
                has_code = True
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    total += len(p["text"])
                    if "```" in p["text"]:
                        has_code = True
    if isinstance(body.get("prompt"), str):
        total += len(body["prompt"])
        if "```" in body["prompt"]:
            has_code = True
    if body.get("response_format"):
        return True  # structured output requirement
    return has_code or total > 4000


def _completion_chars(response_body, stream_chunks) -> int:
    n = 0
    if response_body:
        try:
            j = json.loads(response_body)
            if isinstance(j, dict):
                for c in (j.get("choices") or []):
                    if isinstance(c, dict):
                        msg = c.get("message") or {}
                        if isinstance(msg.get("content"), str):
                            n += len(msg["content"])
                msg = j.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    n += len(msg["content"])
        except (json.JSONDecodeError, TypeError):
            pass
    if stream_chunks:
        for line in stream_chunks.split("\n"):
            data = None
            if line.startswith("data: "):
                data = line[6:]
            elif line.strip().startswith("{"):
                data = line.strip()
            if not data or data == "[DONE]":
                continue
            try:
                j = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                continue
            for c in (j.get("choices") or []):
                if isinstance(c, dict):
                    delta = c.get("delta") or c.get("message") or {}
                    if isinstance(delta.get("content"), str):
                        n += len(delta["content"])
            msg = j.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                n += len(msg["content"])
    return n


def _downsize_score(body, response_body, stream_chunks, prompt_chars: int, duration_ms: float | None) -> tuple[int, list[str]]:
    """0-100 score: higher = more likely a smaller model would have sufficed.
    Returns (score, list_of_reasons_for_score)."""
    score = 0
    reasons: list[str] = []

    completion_chars = _completion_chars(response_body, stream_chunks)
    if completion_chars < 200:
        score += 35; reasons.append(f"short response ({completion_chars} chars)")
    elif completion_chars < 800:
        score += 25; reasons.append(f"medium response ({completion_chars} chars)")
    elif completion_chars < 2000:
        score += 10

    invoked = _extract_tool_calls(response_body, stream_chunks)
    if not invoked:
        score += 25; reasons.append("no tool calls invoked")

    if prompt_chars < 500:
        score += 20; reasons.append(f"short prompt ({prompt_chars} chars)")
    elif prompt_chars < 2000:
        score += 12; reasons.append(f"moderate prompt ({prompt_chars} chars)")
    elif prompt_chars < 5000:
        score += 5

    if not _looks_complex_prompt(body):
        score += 10; reasons.append("prompt is plain text (no code, no structured output)")

    if duration_ms is not None:
        if duration_ms < 1500:
            score += 5
        elif duration_ms < 4000:
            score += 2

    return min(100, max(0, score)), reasons


def evaluate_rules(body_json) -> dict:
    """Run all enabled rules against the parsed request body. Returns the first non-allow verdict."""
    if not isinstance(body_json, dict):
        return {"verdict": "allow", "rule": None, "reason": None, "details": None}
    cfg = load_rules_config()
    for name, fn in RULES_REGISTRY.items():
        rcfg = cfg.get(name) or {}
        if not rcfg.get("enabled", False):
            continue
        try:
            result = fn(body_json, rcfg)
        except Exception as e:
            # Rule errors should never block traffic; record as a warn for visibility.
            return {
                "verdict": "warn",
                "rule": name,
                "reason": f"rule error: {e!r}",
                "details": None,
            }
        if result:
            return {
                "verdict": rcfg.get("action", "block"),
                "rule": name,
                "reason": result["reason"],
                "details": result.get("details"),
            }
    return {"verdict": "allow", "rule": None, "reason": None, "details": None}


def _save_gate(req_id: str, gate: dict):
    conn = db()
    conn.execute(
        "UPDATE requests SET gate_verdict=?, gate_rule=?, gate_reason=?, gate_details=? WHERE id=?",
        (
            gate.get("verdict"),
            gate.get("rule"),
            gate.get("reason"),
            json.dumps(gate.get("details")) if gate.get("details") is not None else None,
            req_id,
        ),
    )
    conn.commit()
    conn.close()


# -------- Web UI + management API (registered first so they win over catch-all) --------

@app.get("/__proxy/api/info")
async def info():
    return {"version": __version__, "upstream": OLLAMA_URL, "anthropic": ANTHROPIC_URL, "lmstudio": LMSTUDIO_URL, "port": PROXY_PORT}


def _read_proc_self_status() -> dict:
    out: dict = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if ":" not in line:
                    continue
                k, _, v = line.partition(":")
                out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


@app.get("/__proxy/api/health")
async def health():
    db_size = 0
    try:
        db_size = Path(DB_PATH).stat().st_size
    except OSError:
        pass

    conn = db()
    request_count = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    metrics_count = conn.execute("SELECT COUNT(*) FROM system_metrics").fetchone()[0]
    settings_count = conn.execute("SELECT COUNT(*) FROM settings").fetchone()[0]
    oldest = conn.execute("SELECT MIN(ts) FROM requests").fetchone()[0]
    newest = conn.execute("SELECT MAX(ts) FROM requests").fetchone()[0]
    # Per-table sizes for SQLite (approximate; uses dbstat if compiled in, else best-effort).
    table_sizes: dict = {}
    try:
        for tbl, sz in conn.execute(
            "SELECT name, SUM(pgsize) FROM dbstat GROUP BY name ORDER BY 2 DESC"
        ).fetchall():
            table_sizes[tbl] = sz
    except sqlite3.OperationalError:
        # dbstat virtual table not available — approximate via row counts only.
        pass
    conn.close()

    proc = _read_proc_self_status()
    rss_kb = None
    try:
        if "VmRSS" in proc:
            rss_kb = int(proc["VmRSS"].split()[0])
    except ValueError:
        pass

    samples = list(_overhead_samples)
    samples_sorted = sorted(samples)
    n = len(samples_sorted)

    def pct(p):
        if not n:
            return 0.0
        idx = max(0, min(n - 1, int(n * p)))
        return round(samples_sorted[idx], 3)

    overhead = {
        "samples": n,
        "avg_ms": round(sum(samples) / n, 3) if n else 0,
        "p50_ms": pct(0.5),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "max_ms": round(samples_sorted[-1], 3) if n else 0,
    }

    return {
        "db": {
            "path": DB_PATH,
            "size_bytes": db_size,
            "size_mb": round(db_size / (1024 * 1024), 3),
            "request_count": request_count,
            "metric_count": metrics_count,
            "settings_count": settings_count,
            "oldest_ts": oldest,
            "newest_ts": newest,
            "table_sizes": table_sizes,
        },
        "process": {
            "pid": os.getpid(),
            "rss_kb": rss_kb,
            "rss_mb": round(rss_kb / 1024, 2) if rss_kb else None,
            "threads": int(proc.get("Threads", "0").split()[0]) if "Threads" in proc else None,
            "uptime_s": int(time.time() - getattr(app.state, "started_at", time.time())),
            "managed_by_systemd": bool(os.environ.get("INVOCATION_ID")),
        },
        "overhead_ms": overhead,
    }


@app.post("/__proxy/api/db/reset")
async def db_reset(request: Request):
    """Wipe captured request/metric data. Settings (rules, etc.) are preserved unless explicitly listed."""
    try:
        payload = await request.json() if (await request.body()) else {}
    except Exception:
        payload = {}
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        targets = ["requests", "metrics"]
    valid = {"requests", "metrics", "settings"}
    targets = [t for t in targets if t in valid]
    counts: dict = {}
    conn = db()
    if "requests" in targets:
        counts["requests"] = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        conn.execute("DELETE FROM requests")
    if "metrics" in targets:
        counts["metrics"] = conn.execute("SELECT COUNT(*) FROM system_metrics").fetchone()[0]
        conn.execute("DELETE FROM system_metrics")
    if "settings" in targets:
        counts["settings"] = conn.execute("SELECT COUNT(*) FROM settings").fetchone()[0]
        conn.execute("DELETE FROM settings")
    conn.commit()
    conn.close()
    # VACUUM must run outside any transaction.
    if payload.get("vacuum", True):
        v = sqlite3.connect(DB_PATH, timeout=10.0)
        v.isolation_level = None
        try:
            v.execute("VACUUM")
        finally:
            v.close()
    _overhead_samples.clear()
    return {"ok": True, "deleted": counts, "vacuumed": payload.get("vacuum", True)}


@app.get("/__proxy/api/system/now")
async def system_now():
    conn = db()
    row = conn.execute(
        """SELECT ts, cpu_pct, load_1m, mem_total_mb, mem_used_mb, mem_avail_mb,
                  gpu_json, ollama_json, lmstudio_json, comfyui_json
           FROM system_metrics ORDER BY ts DESC LIMIT 1"""
    ).fetchone()
    conn.close()
    if not row:
        return {"ts": None, "cpu_pct": None, "mem": None, "gpus": [], "ollama": {}, "lmstudio": {}, "comfyui": {}}
    d = dict(row)
    return {
        "ts": d["ts"],
        "cpu_pct": d["cpu_pct"],
        "load_1m": d["load_1m"],
        "mem": {"total_mb": d["mem_total_mb"], "used_mb": d["mem_used_mb"], "avail_mb": d["mem_avail_mb"]},
        "gpus": json.loads(d["gpu_json"]) if d["gpu_json"] else [],
        "ollama": json.loads(d["ollama_json"]) if d["ollama_json"] else {},
        "lmstudio": json.loads(d["lmstudio_json"]) if d["lmstudio_json"] else {},
        "comfyui": json.loads(d["comfyui_json"]) if d["comfyui_json"] else {},
    }


def _last_user_snippet(body_json, limit=6000):
    """Pull the newest user message text from a request body, for the Live View tile."""
    if not isinstance(body_json, dict):
        return ""
    for m in reversed(body_json.get("messages") or []):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c[:limit]
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") in ("text", "input_text") and part.get("text"):
                    return part["text"][:limit]
    return ""


def _response_snippet(response_body, stream_chunks, limit=6000):
    """Reconstruct the assistant's reply text (OpenAI or Anthropic shape) from a finished
    request's stored response, for the Live View tile. Cheap: stops once past the limit."""
    txt = ""
    if stream_chunks:
        parts = []
        for line in stream_chunks.split("\n"):
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                j = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(j, dict):
                continue
            ch = j.get("choices")
            if isinstance(ch, list) and ch and isinstance(ch[0], dict):
                d = ch[0].get("delta") or ch[0].get("message") or {}
                if isinstance(d.get("content"), str):
                    parts.append(d["content"])
                for tc in (d.get("tool_calls") or []):
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        parts.append(("\n🔧 " if parts else "🔧 ") + fn["name"] + " ")
                    if isinstance(fn.get("arguments"), str):
                        parts.append(fn["arguments"])
            if j.get("type") == "content_block_delta":
                dd = j.get("delta") or {}
                if dd.get("type") == "text_delta" and isinstance(dd.get("text"), str):
                    parts.append(dd["text"])
            if sum(len(p) for p in parts) > limit * 2:
                break
        txt = "".join(parts)
    if not txt and response_body:
        try:
            j = json.loads(response_body)
            ch = j.get("choices")
            if isinstance(ch, list) and ch and isinstance(ch[0], dict):
                m = ch[0].get("message") or {}
                if isinstance(m.get("content"), str):
                    txt = m["content"]
                if not txt:
                    for tc in (m.get("tool_calls") or []):
                        fn = (tc or {}).get("function") or {}
                        if fn.get("name"):
                            txt += ("\n🔧 " if txt else "🔧 ") + fn["name"] + " " + (fn.get("arguments") or "")
            if not txt and isinstance(j.get("content"), list):   # Anthropic
                txt = "".join(b.get("text", "") for b in j["content"]
                              if isinstance(b, dict) and b.get("type") == "text")
        except (json.JSONDecodeError, TypeError):
            pass
    return txt[:limit].strip()


@app.get("/__proxy/api/live")
def live_view(request: Request):  # sync → threadpool
    """Snapshot for the Live View tab: one tile per active/recent conversation. Merges the
    in-flight registry (live elapsed + tokens) with each request's saved DB row (model, client,
    prompt, image) and the recently-finished rows so tiles linger a moment after completing."""
    now = time.time()
    # Only treat genuinely-live requests as active: skip entries the zombie-killer flagged
    # cancelled, and cap the age (a streamer wedged on a dead client socket would otherwise show
    # as a fake multi-minute "active" tile). 900s is well past any real prefill.
    inflight = {k: v.get("ts", now) for k, v in dict(_INFLIGHT_REQUESTS).items()
                if not v.get("cancelled") and (now - v.get("ts", now)) < 900}
    conn = db()
    placeholders = ",".join("?" for _ in inflight) or "''"
    rows = conn.execute(
        f"""SELECT id, ts, conversation_id, turn_index, client_app, client_ip, model, upstream, path,
                   status, est_prompt_tokens, prompt_tokens, completion_tokens, duration_ms,
                   has_images, request_body, response_body, stream_chunks, ttft_ms
            FROM requests_v
            WHERE id IN ({placeholders}) OR ts > ?
            ORDER BY ts ASC""",
        (*inflight.keys(), now - 32),   # finished tiles linger ~30s so the response is readable
    ).fetchall()
    conn.close()
    viewer = _client_ip(request)
    byconv = {}
    for r in rows:
        byconv[r["conversation_id"] or r["id"]] = r   # ASC order → last write is newest per conv
    tiles = []
    for cid, r in byconv.items():
        rid = r["id"]
        active = rid in inflight
        started = inflight.get(rid, r["ts"])
        live = _live_snapshot(rid) if active else None
        ptok = (live or {}).get("prompt_tokens") or r["prompt_tokens"] or r["est_prompt_tokens"] or 0
        otok = (live or {}).get("completion_tokens") or r["completion_tokens"] or 0
        try:
            bj = json.loads(r["request_body"]) if r["request_body"] else None
        except (json.JSONDecodeError, TypeError):
            bj = None
        has_img = bool(r["has_images"])
        live_text = ((_LIVE_STREAMS.get(rid) or {}).get("text") or "") if active else ""
        if not active:
            state = "DONE"
        elif has_img:
            state = "VISION"
        elif otok or live_text:      # usage only lands in the final chunk, so key off live text too
            state = "STREAMING"
        else:
            state = "THINKING"
        # active → live elapsed; done → the request's actual duration (so a quick request that
        # finished before we polled still shows how long it really took, not its age since).
        elapsed_ms = int(r["duration_ms"]) if (not active and r["duration_ms"]) else int((now - started) * 1000)
        tps = round(otok / (elapsed_ms / 1000), 1) if (otok and elapsed_ms > 500) else None
        # Server-side PII gate: only expose the prompt text / image to a viewer allowed to see
        # this originator's data (same IP, same subnet, or an admin IP). Metadata (model, timing,
        # tokens, state) stays visible — mirrors the Requests view's redaction contract.
        viewable = _can_view_pii(viewer, r["client_ip"])
        cache = None
        if not active:
            try:
                _cp, cache = _cache_verdict(r["prompt_tokens"], r["est_prompt_tokens"],
                                            r["ttft_ms"], r["prompt_tokens"], r["upstream"])
            except Exception:
                cache = None
        # "fresh" = finished within the last 10s → keep it highlighted (not dimmed) so a quick
        # request that completed before we could catch it mid-flight still stands out.
        fresh = (not active) and ((now - (r["ts"] + (r["duration_ms"] or 0) / 1000.0)) < 10)
        tiles.append({
            "cache": cache, "key": cid or rid, "fresh": fresh,
            "req_id": rid, "conv": (cid or "")[:6], "client": r["client_app"] or "?",
            "model": r["model"] or "?", "upstream": r["upstream"], "path": r["path"],
            "state": state, "done": not active, "elapsed_ms": elapsed_ms,
            "ptok": ptok, "otok": otok, "tps": tps, "turn": r["turn_index"],
            "itps": (round(ptok / (elapsed_ms / 1000)) if (active and state == "THINKING" and ptok and elapsed_ms > 400) else None),
            "prompt": _last_user_snippet(bj) if viewable else "",
            "response": ((live_text if active else _response_snippet(r["response_body"], r["stream_chunks"]))
                         if viewable else ""),
            "has_image": has_img and viewable,
            "image_url": f"/__proxy/api/requests/{rid}/image/0" if (has_img and viewable) else None,
            "redacted": not viewable,
        })
    tiles.sort(key=lambda t: (t["done"], -t["elapsed_ms"]))
    return {"ts": now, "tiles": tiles[:16]}


def _norm_model_id(mid: str) -> str:
    """Normalize a model id for cross-runtime matching: drop any publisher/ prefix and :tag suffix."""
    s = (mid or "").split("/")[-1]
    return s.split(":")[0].lower()


@app.get("/v1/models")
async def list_models_enriched():
    """OpenAI /v1/models, enriched with each model's real context window.

    Ollama's /v1/models omits any context field, so a client that auto-discovers the window
    (instead of hardcoding it) can't tell how large it is and falls back to a conservative
    default — often 128k — then compacts/stops early even though the model is loaded much larger.
    We fill `context_length` (plus `max_context_length` and `max_model_len` aliases, since
    different clients read different keys) from the cached LM Studio snapshot — the runtime qwen
    actually routes to — matched by normalized model name. Best-effort: if the snapshot is missing
    or a model isn't matched, the entry is returned unchanged.
    """
    client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
    try:
        r = await client.get(OLLAMA_URL + "/v1/models")
        data = r.json() if r.status_code == 200 else {"object": "list", "data": []}
    except Exception:
        data = {"object": "list", "data": []}
    finally:
        await client.aclose()
    ctxmap: dict[str, int] = {}
    try:
        conn = db()
        row = conn.execute(
            "SELECT lmstudio_json FROM system_metrics ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        conn.close()
        lm = json.loads(row["lmstudio_json"]) if row and row["lmstudio_json"] else {}
        for m in (lm.get("available") or []):
            ctx = m.get("loaded_context_length") or m.get("max_context_length")
            if isinstance(ctx, int) and ctx > 0:
                ctxmap[_norm_model_id(m.get("id"))] = ctx
    except Exception:
        pass
    if isinstance(data, dict) and ctxmap:
        for m in (data.get("data") or []):
            if not isinstance(m, dict) or "context_length" in m:
                continue
            ctx = ctxmap.get(_norm_model_id(m.get("id")))
            if ctx:
                m["context_length"] = ctx
                m["max_context_length"] = ctx
                m["max_model_len"] = ctx
    return JSONResponse(data)


@app.get("/__proxy/api/ollama/update-check")
async def ollama_update_check(force: int = 0):
    """Compare local model digests against the Ollama registry. Cached for an hour;
    pass `?force=1` to skip the cache. Returns `{items: [...], checked_ts}`."""
    now = time.time()
    client = httpx.AsyncClient(timeout=httpx.Timeout(15.0))
    try:
        try:
            r = await client.get(OLLAMA_URL + "/api/tags", timeout=httpx.Timeout(5.0))
            tags = ((r.json() or {}).get("models") or []) if r.status_code == 200 else []
        except Exception as e:
            return {"items": [], "error": f"could not reach ollama at {OLLAMA_URL}: {e}"}
        items: list[dict] = []
        to_check: list[tuple[str, str]] = []
        for m in tags:
            name = m.get("name") or m.get("model") or ""
            digest = (m.get("digest") or "").removeprefix("sha256:")
            if not name:
                continue
            cached = _OLLAMA_UPDATE_CACHE.get(name)
            if cached and not force and (now - cached.get("checked_ts", 0) < _OLLAMA_UPDATE_CACHE_TTL_S):
                items.append(cached)
                continue
            to_check.append((name, digest))
        if to_check:
            results = await asyncio.gather(
                *[_check_ollama_registry(client, n, d) for n, d in to_check],
                return_exceptions=False,
            )
            for res in results:
                res["checked_ts"] = now
                _OLLAMA_UPDATE_CACHE[res["name"]] = res
                items.append(res)
        items.sort(key=lambda x: (
            {"outdated": 0, "up_to_date": 1, "unknown": 2,
             "not_in_registry": 3, "error": 4}.get(x.get("status", "error"), 5),
            x.get("name", ""),
        ))
        return {"items": items, "checked_ts": now}
    finally:
        await client.aclose()


@app.get("/__proxy/api/system/history")
# Sync handler: runs in Starlette's threadpool so its blocking DB query can't stall the
# event loop (and thus in-flight request proxying). See the analytics endpoints below.
def system_history(minutes: int = 60):
    minutes = max(1, min(int(minutes), 1440))
    cutoff = time.time() - minutes * 60
    conn = db()
    rows = conn.execute(
        """SELECT ts, cpu_pct, load_1m, mem_used_mb, mem_total_mb, gpu_json
           FROM system_metrics
           WHERE ts > ?
           ORDER BY ts ASC""",
        (cutoff,),
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        gpus = json.loads(d["gpu_json"]) if d["gpu_json"] else []
        # Reduce per-tick GPU payload for the time series — just primary GPU's util/mem.
        primary = gpus[0] if gpus else None
        out.append({
            "ts": d["ts"],
            "cpu_pct": d["cpu_pct"],
            "load_1m": d["load_1m"],
            "mem_used_mb": d["mem_used_mb"],
            "mem_total_mb": d["mem_total_mb"],
            "gpu_util_pct": primary["util_pct"] if primary else None,
            "gpu_mem_used_mb": primary["mem_used_mb"] if primary else None,
            "gpu_mem_total_mb": primary["mem_total_mb"] if primary else None,
        })
    # Downsample so a wide window doesn't ship tens of thousands of points (a multi-MB
    # payload) to the browser on every poll — the charts don't need finer resolution.
    total = len(out)
    if total > 800:
        stride = total // 800 + 1
        out = out[::stride]
    return {"minutes": minutes, "samples": out, "total_samples": total}


# Real prompt prefill is bounded by the accelerator's prompt-processing throughput. An implied
# prefill rate (prompt tokens / TTFT) far above what the hardware can actually prefill means the
# prompt was served from the KV cache (prefix reused) rather than re-evaluated. Large local
# models typically cold-prefill at a few hundred tok/s, while a cache hit implies several
# thousand — so this threshold sits several× above real cold prefill. Note TTFT includes some
# scheduling/first-token overhead, which deflates the rate, so we keep the bar reachable. Tune
# it (or _CACHE_MIN_PROMPT_TOKENS) to your hardware if the hit/miss labels look off.
_PREFILL_SKIP_TOK_PER_S = 2500.0
_CACHE_MIN_PROMPT_TOKENS = 400   # below this, timing is too noisy to judge reuse


def _cache_verdict(evaluated, est, ttft_ms=None, prompt_tokens=None, upstream=None):
    """Cache-reuse verdict. Returns (cache_pct, verdict) where verdict is 'hit'|'partial'|'miss',
    or (None, None) when we can't tell (in flight, tiny prompt, or an upstream we can't measure).

    Two signals, in priority order:
      1. Timing (upstream-agnostic): if the effective prefill rate (prompt tokens / TTFT) is
         implausibly high, the prefill was skipped → hit. This is the ONLY reliable signal for
         OpenAI-semantics upstreams like LM Studio, which report the full prompt_tokens and no
         cached_tokens regardless of reuse.
      2. Token counts (Ollama only): Ollama's prompt_tokens is the *evaluated-only* count, so
         evaluated < estimated ⇒ a prefix was reused. Invalid for LM Studio (full count), so it's
         applied only to Ollama rows to avoid crying 'miss' on real hits."""
    pt = prompt_tokens or evaluated or est or 0
    # 1) Timing-based, works across upstreams (needs streamed TTFT + a non-trivial prompt).
    if ttft_ms and ttft_ms > 0 and pt >= _CACHE_MIN_PROMPT_TOKENS:
        if pt / (ttft_ms / 1000.0) >= _PREFILL_SKIP_TOK_PER_S:
            return None, "hit"
    # 2) Token-based, valid only where prompt_tokens == evaluated-only count (Ollama).
    if upstream in (None, "ollama"):
        e = evaluated or 0
        s = est or 0
        if s > 0 and e > 0:
            pct = round(100.0 * (1 - e / s), 1) if e < s else 0.0
            return pct, ("hit" if pct >= 50 else ("partial" if pct >= 10 else "miss"))
    return None, None


@app.get("/__proxy/api/requests")
def list_requests(request: Request, limit: int = 200, offset: int = 0, include_shadows: bool = False, client: str = ""):  # sync → threadpool
    viewer = _client_ip(request)
    conn = db()
    conds, params = [], []
    if not include_shadows:
        conds.append("shadow_of IS NULL")
    if client:
        # Match either the detected app (e.g. "claude-code") or the raw client IP.
        conds.append("(client_app = ? OR client_ip = ?)")
        params += [client, client]
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    rows = conn.execute(
        f"""SELECT id, ts, method, path, model, is_stream, status, duration_ms, error,
                  prompt_tokens, completion_tokens, total_tokens, est_prompt_tokens, ttft_ms,
                  upstream, client_ip, client_app, gate_verdict, gate_rule, gate_reason, shadow_of,
                  has_images
           FROM requests {where} ORDER BY ts DESC LIMIT ? OFFSET ?""",
        (*params, limit, offset),
    ).fetchall()
    total = conn.execute(
        f"SELECT COUNT(*) FROM requests {where}", tuple(params)
    ).fetchone()[0]
    # Distinct clients (for the filter dropdown), independent of the current client filter.
    client_rows = conn.execute(
        "SELECT client_app AS app, COUNT(*) AS count FROM requests WHERE client_app IS NOT NULL"
        + ("" if include_shadows else " AND shadow_of IS NULL")
        + " GROUP BY client_app ORDER BY count DESC"
    ).fetchall()
    conn.close()
    items = []
    for r in rows:
        d = dict(r)
        if d.get("status") is None and d.get("total_tokens") is None:
            live = _live_snapshot(d["id"])
            if live:
                d["prompt_tokens"] = live["prompt_tokens"]
                d["completion_tokens"] = live["completion_tokens"]
                d["total_tokens"] = live["total_tokens"]
                d["tokens_live"] = True
                d["tokens_estimated"] = live["estimated"]
        # Cache verdict: cheap (timing + token arithmetic), no body parse.
        _cpct, _cverdict = _cache_verdict(d.get("prompt_tokens"), d.get("est_prompt_tokens"),
                                          d.get("ttft_ms"), d.get("prompt_tokens"), d.get("upstream"))
        d["cache_pct"] = _cpct
        d["cache_verdict"] = _cverdict
        items.append(_redact_row(d, viewer))
    return {"total": total, "items": items,
            "clients": [dict(r) for r in client_rows], "redacted": REDACT_PII_ENABLED}


@app.get("/__proxy/api/cache")
def cache_stats(request: Request, limit: int = 50):  # sync → threadpool
    """Prompt-cache diagnostic. Ollama reports how many prompt tokens it actually had to
    *evaluate* (prefill) — stored here as prompt_tokens. Comparing that against the estimated
    total prompt size shows whether the KV cache was reused: if a request sends ~85k tokens
    but only ~2k were evaluated, the shared prefix was cached (hit); if it evaluated ~all of
    them, no reuse (miss). Lets you SEE if caching is doing anything instead of guessing."""
    limit = max(1, min(int(limit or 50), 200))
    conn = db()
    rows = conn.execute(
        """SELECT id, ts, client_app, model, request_body, prompt_tokens, duration_ms,
                  ttft_ms, upstream
           FROM requests_v
           WHERE prompt_tokens IS NOT NULL AND request_body IS NOT NULL AND shadow_of IS NULL
                 AND (path LIKE '%chat%' OR path LIKE '%generate%' OR path LIKE '%messages%')
           ORDER BY ts DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    items = []
    for r in rows:
        try:
            body = json.loads(r["request_body"])
        except (json.JSONDecodeError, TypeError):
            body = {}
        est = _estimate_prompt_tokens(body, 3.5) if isinstance(body, dict) else 0
        evaluated = r["prompt_tokens"] or 0
        # Shared verdict: timing-based for OpenAI-semantics upstreams (LM Studio), token-based
        # for Ollama. See _cache_verdict. Falls back to a plain token pct when it returns None.
        pct, verdict = _cache_verdict(evaluated, est, r["ttft_ms"], evaluated, r["upstream"])
        if verdict is None:
            pct = round(100.0 * (1 - evaluated / est), 1) if est > 0 and evaluated < est else 0.0
            verdict = "hit" if pct >= 50 else ("partial" if pct >= 10 else "miss")
        items.append({
            "id": r["id"], "ts": r["ts"], "client_app": r["client_app"], "model": r["model"],
            "evaluated_tokens": evaluated, "est_prompt_tokens": est, "upstream": r["upstream"],
            "cache_pct": pct, "verdict": verdict, "duration_ms": r["duration_ms"],
        })
    n = len(items)
    return {
        "items": items,
        "summary": {
            "count": n,
            "hits": sum(1 for i in items if i["verdict"] == "hit"),
            "misses": sum(1 for i in items if i["verdict"] == "miss"),
            "avg_cache_pct": round(sum(i["cache_pct"] for i in items) / n, 1) if n else 0.0,
        },
        "note": "evaluated_tokens = tokens Ollama actually prefilled (from usage). Much lower "
                "than est_prompt_tokens => KV cache reused the shared prefix.",
    }


@app.get("/__proxy/api/audit")
def audit(request: Request, limit: int = 200, offset: int = 0, include_allow: bool = False):  # sync → threadpool
    viewer = _client_ip(request)
    limit = max(1, min(int(limit or 200), 1000))
    offset = max(0, int(offset or 0))
    conn = db()
    where = ("gate_verdict IS NOT NULL" if include_allow
             else "gate_verdict IN ('block', 'warn', 'rewrite', 'intercept')")
    rows = conn.execute(
        f"""SELECT id, ts, method, path, model, client_ip, gate_verdict, gate_rule, gate_reason, gate_details
           FROM requests WHERE {where}
           ORDER BY ts DESC LIMIT ? OFFSET ?""",
        (limit, offset),
    ).fetchall()
    total = conn.execute(f"SELECT COUNT(*) FROM requests WHERE {where}").fetchone()[0]
    counts = {}
    for v, n in conn.execute(
        "SELECT COALESCE(gate_verdict, '(none)'), COUNT(*) FROM requests GROUP BY gate_verdict"
    ).fetchall():
        counts[v] = n
    conn.close()
    items = [_redact_row(dict(r), viewer) for r in rows]
    return {"counts": counts, "items": items, "total": total, "redacted": REDACT_PII_ENABLED}


@app.get("/__proxy/api/conversations")
def list_conversations(request: Request, limit: int = 100):  # sync → threadpool
    viewer = _client_ip(request)
    conn = db()
    rows = conn.execute(
        """SELECT conversation_id,
                  MIN(ts) AS first_ts,
                  MAX(ts) AS last_ts,
                  COUNT(*) AS turns,
                  COALESCE(SUM(total_tokens), 0) AS total_tokens,
                  COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                  COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                  GROUP_CONCAT(DISTINCT model) AS models,
                  GROUP_CONCAT(DISTINCT client_ip) AS clients,
                  GROUP_CONCAT(DISTINCT client_app) AS apps,
                  SUM(CASE WHEN gate_verdict = 'block' THEN 1 ELSE 0 END) AS blocks,
                  SUM(CASE WHEN gate_verdict = 'rewrite' THEN 1 ELSE 0 END) AS rewrites
           FROM requests
           WHERE conversation_id IS NOT NULL
           GROUP BY conversation_id
           ORDER BY last_ts DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    items = []
    for r in rows:
        d = dict(r)
        preview = conn.execute(
            """SELECT request_body FROM requests_v
               WHERE conversation_id = ? ORDER BY ts ASC LIMIT 1""",
            (d["conversation_id"],),
        ).fetchone()
        if preview and preview["request_body"]:
            try:
                j = json.loads(preview["request_body"])
                first_user = next((m for m in (j.get("messages") or []) if isinstance(m, dict) and m.get("role") == "user"), None)
                if first_user:
                    text = _msg_text(first_user)
                    d["preview"] = text[:200]
            except json.JSONDecodeError:
                pass
        # Conversation aggregates may include multiple client IPs — show preview if any
        # contributing client is viewable.
        contributing_ips = [ip.strip() for ip in (d.get("clients") or "").split(",") if ip.strip()]
        items.append(_redact_row(d, viewer, originator_ips=contributing_ips or [None]))
    conn.close()
    return {"items": items, "redacted": REDACT_PII_ENABLED}


@app.get("/__proxy/api/conversations/{conv_id}")
def get_conversation(conv_id: str, request: Request):  # sync → threadpool
    viewer = _client_ip(request)
    conn = db()
    rows = conn.execute(
        """SELECT id, ts, model, turn_index, request_body, response_body, stream_chunks,
                  prompt_tokens, completion_tokens, total_tokens, duration_ms,
                  status, error, gate_verdict, gate_rule, gate_reason, gate_details,
                  client_ip, is_stream
           FROM requests_v
           WHERE conversation_id = ?
           ORDER BY ts ASC""",
        (conv_id,),
    ).fetchall()
    conn.close()
    turns = [_redact_row(dict(r), viewer) for r in rows]
    return {"conversation_id": conv_id, "turns": turns, "redacted": REDACT_PII_ENABLED}


@app.get("/__proxy/api/suggestions")
def suggestions():  # sync → threadpool
    """Scan recent traffic and surface config tuning recommendations."""
    cached = _analytics_cache_get("suggestions")
    if cached is not None:
        return cached
    conn = db()
    cutoff = time.time() - 30 * 86400
    rows = conn.execute(
        """SELECT ts, model, request_body, response_body, stream_chunks, prompt_tokens, completion_tokens,
                  total_tokens, duration_ms, client_ip, gate_verdict, gate_rule
           FROM requests_v
           WHERE ts > ?
           ORDER BY ts DESC
           LIMIT 5000""",
        (cutoff,),
    ).fetchall()
    conn.close()

    cfg = load_rules_config()

    out: list[dict] = []
    by_model: dict[str, list[dict]] = {}
    tools_defined: dict[str, int] = {}
    tools_invoked: dict[str, int] = {}
    by_client_tokens: dict[str, int] = {}
    near_miss_sigs: dict[tuple[str, str], int] = {}

    loop_threshold = max(2, int((cfg.get("loop_detector") or {}).get("max_repeats", 4)))

    for r in rows:
        body_json = None
        if r["request_body"]:
            try:
                body_json = json.loads(r["request_body"])
            except json.JSONDecodeError:
                pass
        if isinstance(body_json, dict):
            chars = _prompt_total_chars(body_json)
            has_tools = bool(body_json.get("tools"))
            if r["model"]:
                by_model.setdefault(r["model"], []).append({
                    "chars": chars,
                    "has_tools": has_tools,
                    "completion": r["completion_tokens"] or 0,
                    "duration_ms": r["duration_ms"] or 0,
                    "prompt_tokens": r["prompt_tokens"] or 0,
                })
            for t in (body_json.get("tools") or []):
                fn = (t.get("function") if isinstance(t, dict) else None) or t
                if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                    tools_defined[fn["name"]] = tools_defined.get(fn["name"], 0) + 1

            # Loop near-misses: count top tool-call signature in conversation history
            assistant_msgs = [m for m in (body_json.get("messages") or []) if isinstance(m, dict) and m.get("role") == "assistant"]
            recent = assistant_msgs[-10:]
            sig_counts: dict[tuple[str, str], int] = {}
            for m in recent:
                for tc in (m.get("tool_calls") or []):
                    fn = tc.get("function") or {}
                    sig = (fn.get("name") or "?", _normalize_args(fn.get("arguments")))
                    sig_counts[sig] = sig_counts.get(sig, 0) + 1
            if sig_counts:
                top_sig, top_count = max(sig_counts.items(), key=lambda kv: kv[1])
                if top_count == loop_threshold - 1:
                    near_miss_sigs[top_sig] = near_miss_sigs.get(top_sig, 0) + 1

        for n in _extract_tool_calls(r["response_body"], r["stream_chunks"]):
            tools_invoked[n] = tools_invoked.get(n, 0) + 1

        if r["client_ip"]:
            by_client_tokens[r["client_ip"]] = by_client_tokens.get(r["client_ip"], 0) + (r["total_tokens"] or 0)

    # Suggestion 1: route short non-tool prompts on heavy model to smaller
    for model, items in by_model.items():
        if len(items) < 5:
            continue
        short = [x for x in items if x["chars"] < 500 and not x["has_tools"]]
        ratio = len(short) / len(items)
        if ratio >= 0.5 and len(short) >= 5:
            out.append({
                "id": f"route-short-{model}",
                "severity": "suggest",
                "title": f"Route short prompts on {model} to a smaller model",
                "detail": f"{len(short)} of {len(items)} ({int(ratio*100)}%) requests on {model} had < 500 prompt chars and no tools. A smaller local model would likely suffice.",
                "snippet": {
                    "model_router": {
                        "enabled": True,
                        "rules": [{"if": {"from_model": model, "prompt_chars_lt": 500, "has_tools": False}, "then": "<small-model>"}]
                    }
                },
            })

    # Suggestion 1b: per-model downsize-candidate ratio from heuristic
    downsize_per_model: dict[str, list[int]] = {}
    for r in rows:
        if not r["model"] or not r["request_body"]:
            continue
        try:
            bj = json.loads(r["request_body"])
        except (json.JSONDecodeError, TypeError):
            continue
        score, _ = _downsize_score(bj, r["response_body"], r["stream_chunks"],
                                    _prompt_total_chars(bj), r["duration_ms"])
        downsize_per_model.setdefault(r["model"], []).append(score)
    for model, scores in downsize_per_model.items():
        if len(scores) < 5:
            continue
        high = [s for s in scores if s >= 70]
        if len(high) >= 5 and len(high) / len(scores) >= 0.4:
            avg_high = sum(high) / len(high)
            out.append({
                "id": f"downsize-heuristic-{model}",
                "severity": "perf",
                "title": f"~{int(len(high)/len(scores)*100)}% of requests on {model} look downsize-able",
                "detail": f"{len(high)} of {len(scores)} requests scored ≥70 on the downsize heuristic (avg {avg_high:.0f}/100). Short responses, no tool calls, simple prompts — a smaller model would likely produce equivalent results.",
                "snippet": {
                    "model_router": {
                        "enabled": True,
                        "rules": [
                            {"if": {"from_model": model, "prompt_chars_lt": 1500, "has_tools": False}, "then": "<small-model>"}
                        ]
                    }
                },
            })

    # Suggestion 2: slow on short prompts
    for model, items in by_model.items():
        slow_short = [x for x in items if x["duration_ms"] > 5000 and x["prompt_tokens"] and x["prompt_tokens"] < 200]
        if len(slow_short) >= 5:
            avg = sum(x["duration_ms"] for x in slow_short) / len(slow_short)
            out.append({
                "id": f"slow-short-{model}",
                "severity": "perf",
                "title": f"{model} is slow on short prompts",
                "detail": f"{len(slow_short)} requests with <200 prompt tokens averaged {int(avg)}ms on {model}. Routing those to a faster model would cut latency.",
                "snippet": {
                    "model_router": {
                        "enabled": True,
                        "rules": [{"if": {"from_model": model, "prompt_chars_lt": 800, "has_tools": False}, "then": "<faster-model>"}]
                    }
                },
            })

    # Suggestion 3: unused tools
    for name, defined in tools_defined.items():
        if defined >= 10 and tools_invoked.get(name, 0) == 0:
            out.append({
                "id": f"unused-tool-{name}",
                "severity": "cleanup",
                "title": f"Tool {name!r} is defined but never invoked",
                "detail": f"Defined in {defined} requests; the model never called it. Removing it from the client's tool list would save prompt tokens on every request.",
            })

    # Suggestion 4: loop near-misses
    if near_miss_sigs:
        top = sorted(near_miss_sigs.items(), key=lambda kv: kv[1], reverse=True)[:3]
        for (name, _args), count in top:
            if count >= 3:
                out.append({
                    "id": f"loop-near-miss-{name}",
                    "severity": "tune",
                    "title": f"Tool {name!r} repeatedly hits the loop-detector threshold − 1",
                    "detail": f"{count} requests had {loop_threshold - 1} repeats of the same {name!r} call in their history (one short of blocking). Consider lowering loop_detector.max_repeats.",
                    "snippet": {"loop_detector": {"max_repeats": max(2, loop_threshold - 1)}},
                })

    # Suggestion 4b: tool error rates. Use the latest request per conversation to avoid
    # double-counting the same historical tool result that appears in many sequential requests.
    conn = db()
    err_rows = conn.execute(
        """SELECT r.request_body
           FROM requests_v r
           INNER JOIN (
               SELECT conversation_id, MAX(ts) AS max_ts
               FROM requests
               WHERE conversation_id IS NOT NULL AND ts > ? AND request_body IS NOT NULL
               GROUP BY conversation_id
           ) lr ON r.conversation_id = lr.conversation_id AND r.ts = lr.max_ts
           LIMIT 500""",
        (cutoff,),
    ).fetchall()
    conn.close()
    tool_err_stats: dict[str, dict] = {}
    for r in err_rows:
        try:
            body = json.loads(r["request_body"])
        except (json.JSONDecodeError, TypeError):
            continue
        for name, content in _tool_results_in_body(body):
            entry = tool_err_stats.setdefault(name, {"total": 0, "errors": 0, "last": None})
            entry["total"] += 1
            is_err, excerpt = _is_tool_error(content)
            if is_err:
                entry["errors"] += 1
                entry["last"] = excerpt
    for name, e in tool_err_stats.items():
        if e["total"] >= 5 and e["errors"] / e["total"] >= 0.2:
            pct = int(e["errors"] / e["total"] * 100)
            out.append({
                "id": f"tool-error-rate-{name}",
                "severity": "tune",
                "title": f"Tool {name!r} fails {pct}% of the time",
                "detail": f"{e['errors']} of {e['total']} calls returned errors across recent conversations. Latest error: {e['last']!r}",
            })

    # Suggestion 5: token-skewed clients
    total_tokens = sum(by_client_tokens.values())
    if total_tokens > 0 and len(by_client_tokens) > 2:
        top_ip, top_tokens = max(by_client_tokens.items(), key=lambda kv: kv[1])
        if top_tokens / total_tokens > 0.5:
            out.append({
                "id": f"heavy-client-{top_ip}",
                "severity": "policy",
                "title": f"Client {top_ip} consumes {int(top_tokens/total_tokens*100)}% of tokens",
                "detail": f"{top_ip} used {top_tokens:,} of {total_tokens:,} tokens across {len(by_client_tokens)} clients. Consider per-client routing or rate limiting.",
                "snippet": {"model_router": {"rules": [{"if": {"from_client": top_ip}, "then": "<dedicated-model>"}]}},
            })

    # Suggestion 6: OLLAMA_NUM_PARALLEL is too low for the proxy's enabled features.
    # Each parallel slot reserves num_ctx of GPU memory, so the bump isn't free, but with
    # only N slots, N+1 concurrent conversations cause cache eviction and full prefill on
    # every interleaved turn — the exact thing prompt caching is supposed to prevent.
    try:
        ollama_env: dict = {}
        pid = _find_ollama_pid()
        if pid:
            ollama_env = _read_ollama_env(pid)
        if not ollama_env:
            ollama_env = _read_systemd_env()
        cur_parallel = int(ollama_env.get("OLLAMA_NUM_PARALLEL") or 1)
    except (ValueError, TypeError, OSError):
        cur_parallel = None

    if cur_parallel is not None:
        # Estimate concurrent slot demand from recent activity:
        #   - one slot per Ollama-bound model that's been active recently
        #   - one extra slot if shadow_router has fired (shadow + primary share concurrency)
        #   - one extra slot per distinct conversation seen on Ollama paths in the last hour
        recent_cutoff = time.time() - 3600
        ollama_models = set()
        ollama_convs = set()
        # rows is already in scope from the top of suggestions().
        for r in rows:
            if r["ts"] is not None and r["ts"] < recent_cutoff:
                continue
            if r["model"] and r["model"] != "(none)" and not _is_claude_model(r["model"]):
                ollama_models.add(r["model"])
            # request body might give us the conv id, but we don't have ts comparison done
        # Pull active conversations from DB directly (cheap).
        try:
            cn = db()
            ollama_convs_count = cn.execute(
                """SELECT COUNT(DISTINCT conversation_id) FROM requests
                   WHERE ts > ? AND conversation_id IS NOT NULL
                     AND model IS NOT NULL AND model NOT LIKE 'claude-%'""",
                (recent_cutoff,),
            ).fetchone()[0] or 0
            cn.close()
        except sqlite3.OperationalError:
            ollama_convs_count = 0

        shadow_active = bool((cfg.get("shadow_router") or {}).get("enabled")
                              and (cfg.get("shadow_router") or {}).get("rules"))
        bridge_active = bool((cfg.get("protocol_bridge") or {}).get("enabled", True)
                              and (cfg.get("model_router") or {}).get("enabled")
                              and (cfg.get("model_router") or {}).get("rules"))

        # Demand model: # of distinct concurrent conversations + 1 for shadow overhead.
        recommended = max(len(ollama_models), ollama_convs_count, 1)
        if shadow_active:
            recommended += 1  # shadow runs alongside its primary
        if bridge_active and recommended < 2:
            recommended = 2  # at minimum allow one redirect + one passthrough

        if recommended > cur_parallel:
            reasons: list[str] = []
            if shadow_active: reasons.append("shadow_router is enabled")
            if bridge_active: reasons.append("protocol_bridge redirects active")
            if ollama_convs_count > cur_parallel: reasons.append(f"{ollama_convs_count} distinct conversations on Ollama in last hour")
            if len(ollama_models) > 1: reasons.append(f"{len(ollama_models)} Ollama models active")
            why = "; ".join(reasons) if reasons else "active concurrency exceeds slot count"
            out.append({
                "id": "ollama-num-parallel-low",
                "severity": "perf",
                "title": f"OLLAMA_NUM_PARALLEL={cur_parallel} too low — recommend ≥{recommended}",
                "detail": (
                    f"With {cur_parallel} KV-cache slot(s), interleaved conversations evict each other "
                    f"and pay full prefill on every turn (defeats prompt caching). {why}. "
                    f"Raise to {recommended} on the Ollama systemd unit, then restart Ollama. "
                    f"Each slot reserves num_ctx of GPU memory, so size accordingly."
                ),
                "snippet": {
                    "_action": "edit /etc/systemd/system/ollama.service.d/override.conf",
                    "_command": f"sudo systemctl edit ollama  # then add: Environment=OLLAMA_NUM_PARALLEL={recommended}",
                    "_after": "sudo systemctl daemon-reload && sudo systemctl restart ollama",
                },
            })

    return _analytics_cache_put("suggestions", {"sample_size": len(rows), "items": out})


@app.get("/__proxy/api/rules")
async def get_rules():
    cfg = load_rules_config()
    src, raw = _rules_source()
    setting = get_setting("rules")
    # Show every known rule/transform — pre-flight (registry), transforms, and post-flight.
    known_extras = ["model_router", "ollama_options", "context_overflow_guard", "tool_pruner", "protocol_bridge", "shadow_router", "tool_injector", "compaction_nudge", "request_priority", "request_dedup", "schema_validator", "hallucinated_tool", "tool_args_autofix", "xml_autofix", "tool_call_xml_retry"]
    seen: set = set()
    registered: list[str] = []
    for n in list(RULES_REGISTRY.keys()) + known_extras:
        if n not in seen:
            seen.add(n)
            registered.append(n)
    return {
        "registered": registered,
        "config": cfg,
        "defaults": DEFAULT_RULES_CONFIG,
        "source": src,
        "stored": (json.loads(raw) if raw else None),
        "updated_ts": setting.get("updated_ts") if setting else None,
        "rules_file": RULES_FILE,
        "rules_file_exists": Path(RULES_FILE).exists(),
    }


@app.post("/__proxy/api/rules")
async def update_rules(request: Request):
    try:
        payload = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"invalid JSON body: {e}"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "rules config must be a JSON object"}, status_code=400)
    set_setting("rules", json.dumps(payload, indent=2))
    cfg = load_rules_config()
    return {"ok": True, "config": cfg, "source": "db"}


@app.post("/__proxy/api/rules/reset")
async def reset_rules():
    delete_setting("rules")
    return {"ok": True, "config": load_rules_config(), "source": "defaults"}


_STATS_CACHE: dict = {"ts": 0.0, "data": None}
_STATS_CACHE_TTL_S = 10.0


_PERF_CACHE: dict = {}


@app.get("/__proxy/api/perf")
def perf(window_h: float = 6.0, bucket_min: float = 5.0):  # sync → threadpool
    """Time-bucketed performance trends for the Stats tab: prefill (TTFT) p50/p95, decode rate,
    cache-hit %, prompt size, concurrency, and errors per bucket, plus a latency-vs-size scatter.
    SQLite has no percentile aggregate, so we pull the window's rows and bucket in Python."""
    now = time.time()
    window_h = max(0.25, min(168.0, window_h))
    bucket_min = max(1.0, min(120.0, bucket_min))
    _ck = (round(window_h, 2), round(bucket_min, 2))
    _cc = _PERF_CACHE.get(_ck)
    if _cc and (now - _cc[0]) < 8.0:
        return _cc[1]
    since = now - window_h * 3600.0
    bsec = bucket_min * 60.0
    nb = max(1, int(round(window_h * 60.0 / bucket_min)))
    conn = db()
    rows = conn.execute(
        """SELECT ts, ttft_ms, duration_ms, prompt_tokens, est_prompt_tokens, completion_tokens,
                  status, error, upstream
           FROM requests WHERE ts >= ? ORDER BY ts""",
        (since,),
    ).fetchall()
    conn.close()

    buckets = [{"ttft": [], "decode": [], "hit": 0, "ctot": 0, "ps": [], "conc": 0, "err": 0, "n": 0}
               for _ in range(nb)]
    scatter = []
    for r in rows:
        bi = int((r["ts"] - since) / bsec)
        if bi < 0 or bi >= nb:
            continue
        b = buckets[bi]
        b["n"] += 1
        pt = r["prompt_tokens"] or r["est_prompt_tokens"]
        if pt:
            b["ps"].append(pt)
        if r["ttft_ms"] and r["ttft_ms"] > 0:
            b["ttft"].append(r["ttft_ms"] / 1000.0)
        if (r["completion_tokens"] and r["ttft_ms"] and r["duration_ms"]
                and r["duration_ms"] > r["ttft_ms"]):
            dt = (r["duration_ms"] - r["ttft_ms"]) / 1000.0
            if dt > 0:
                b["decode"].append(r["completion_tokens"] / dt)
        try:
            _cp, cv = _cache_verdict(r["prompt_tokens"], r["est_prompt_tokens"],
                                     r["ttft_ms"], r["prompt_tokens"], r["upstream"])
        except Exception:
            cv = None
        if cv:
            b["ctot"] += 1
            if cv == "hit":
                b["hit"] += 1
        if (r["status"] is not None and (r["status"] >= 400 or r["status"] == 0)) or r["error"]:
            b["err"] += 1
        if r["duration_ms"]:
            be = min(nb - 1, int((r["ts"] + r["duration_ms"] / 1000.0 - since) / bsec))
            for j in range(bi, be + 1):
                buckets[j]["conc"] += 1
        if pt and r["ttft_ms"] and r["ttft_ms"] > 0:
            scatter.append([round(pt / 1000.0, 1), round(r["ttft_ms"] / 1000.0, 2),
                            1 if cv == "hit" else 0])

    def pctl(vals, p):
        if not vals:
            return None
        s = sorted(vals)
        return round(s[min(len(s) - 1, int(p * len(s)))], 2)

    series = []
    for i, b in enumerate(buckets):
        series.append({
            "t": round(since + i * bsec),
            "ttft_p50": pctl(b["ttft"], 0.50),
            "ttft_p95": pctl(b["ttft"], 0.95),
            "decode": pctl(b["decode"], 0.50),
            "cache_pct": round(100.0 * b["hit"] / b["ctot"]) if b["ctot"] else None,
            "psize_k": round(pctl(b["ps"], 0.50) / 1000.0, 1) if b["ps"] else None,
            "conc": b["conc"],
            "err": b["err"],
            "n": b["n"],
        })
    _res = {"since": since, "now": now, "bucket_s": bsec, "series": series, "scatter": scatter[-220:]}
    _PERF_CACHE[_ck] = (now, _res)
    return _res


@app.get("/__proxy/api/stats")
def stats():  # sync → threadpool
    # Cheap TTL cache. Stats are expensive (5000-row table scans) and the UI auto-refreshes
    # every 1.5s; without this we re-query the whole world ~40×/min per viewer.
    now = time.time()
    if _STATS_CACHE["data"] is not None and (now - _STATS_CACHE["ts"]) < _STATS_CACHE_TTL_S:
        return _STATS_CACHE["data"]
    conn = db()
    overall = dict(conn.execute(
        """SELECT COUNT(*) AS count,
                  COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                  COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                  COALESCE(SUM(total_tokens), 0) AS total_tokens,
                  COALESCE(AVG(duration_ms), 0) AS avg_ms,
                  COALESCE(SUM(duration_ms), 0) AS total_ms,
                  SUM(CASE WHEN (status >= 400) OR (status = 0) OR (error IS NOT NULL) THEN 1 ELSE 0 END) AS errors,
                  SUM(CASE WHEN is_stream = 1 THEN 1 ELSE 0 END) AS streams,
                  MIN(ts) AS first_ts, MAX(ts) AS last_ts
           FROM requests"""
    ).fetchone())

    by_model = [dict(r) for r in conn.execute(
        """SELECT COALESCE(model, '(none)') AS model,
                  COUNT(*) AS count,
                  COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                  COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                  COALESCE(SUM(total_tokens), 0) AS total_tokens,
                  COALESCE(AVG(duration_ms), 0) AS avg_ms,
                  COALESCE(SUM(duration_ms), 0) AS total_ms,
                  SUM(CASE WHEN is_stream = 1 THEN 1 ELSE 0 END) AS streams,
                  SUM(CASE WHEN (status >= 400) OR (status = 0) OR (error IS NOT NULL) THEN 1 ELSE 0 END) AS errors,
                  MAX(ts) AS last_ts
           FROM requests
           GROUP BY model
           ORDER BY count DESC"""
    ).fetchall()]

    by_path = [dict(r) for r in conn.execute(
        """SELECT path,
                  COUNT(*) AS count,
                  COALESCE(AVG(duration_ms), 0) AS avg_ms,
                  COALESCE(SUM(total_tokens), 0) AS total_tokens
           FROM requests
           GROUP BY path
           ORDER BY count DESC
           LIMIT 25"""
    ).fetchall()]

    by_status = [dict(r) for r in conn.execute(
        """SELECT COALESCE(status, 0) AS status, COUNT(*) AS count
           FROM requests GROUP BY status ORDER BY status"""
    ).fetchall()]

    by_upstream = [dict(r) for r in conn.execute(
        """SELECT COALESCE(upstream, '(unknown)') AS upstream,
                  COUNT(*) AS count,
                  COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                  COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                  COALESCE(SUM(total_tokens), 0) AS total_tokens,
                  COALESCE(AVG(duration_ms), 0) AS avg_ms,
                  COALESCE(SUM(duration_ms), 0) AS total_ms,
                  COALESCE(AVG(ttft_ms), 0) AS avg_ttft_ms,
                  SUM(CASE WHEN ttft_ms IS NOT NULL THEN 1 ELSE 0 END) AS with_ttft,
                  SUM(CASE WHEN is_stream = 1 THEN 1 ELSE 0 END) AS streams,
                  MAX(ts) AS last_ts
           FROM requests
           GROUP BY upstream
           ORDER BY count DESC"""
    ).fetchall()]

    by_client = [dict(r) for r in conn.execute(
        """SELECT COALESCE(client_ip, '(unknown)') AS client_ip,
                  COUNT(*) AS count,
                  COALESCE(SUM(total_tokens), 0) AS total_tokens,
                  COALESCE(AVG(duration_ms), 0) AS avg_ms,
                  MAX(ts) AS last_ts
           FROM requests
           GROUP BY client_ip
           ORDER BY count DESC
           LIMIT 50"""
    ).fetchall()]

    by_app = [dict(r) for r in conn.execute(
        """SELECT COALESCE(client_app, '(unknown)') AS client_app,
                  COUNT(*) AS count,
                  COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                  COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                  COALESCE(SUM(total_tokens), 0) AS total_tokens,
                  COALESCE(AVG(duration_ms), 0) AS avg_ms,
                  COUNT(DISTINCT conversation_id) AS conversations,
                  COUNT(DISTINCT model) AS models,
                  GROUP_CONCAT(DISTINCT model) AS models_seen,
                  MAX(ts) AS last_ts
           FROM requests
           GROUP BY client_app
           ORDER BY count DESC"""
    ).fetchall()]

    # Tool usage requires parsing response bodies — scan up to a bounded set.
    # Trimmed from 5000 → 500 and limited to the last 7 days. With multi-MB response
    # bodies, the old query was loading hundreds of MB and JSON-parsing every row, which
    # dominated the stats endpoint latency. The most recent 500 is representative.
    _tool_cutoff = time.time() - 7 * 86400
    tool_rows = conn.execute(
        """SELECT model, response_body, stream_chunks FROM requests_v
           WHERE ts > ? AND (response_body IS NOT NULL OR stream_chunks IS NOT NULL)
           ORDER BY ts DESC LIMIT 500""",
        (_tool_cutoff,),
    ).fetchall()
    by_tool: dict[str, dict] = {}
    for r in tool_rows:
        names = _extract_tool_calls(r["response_body"], r["stream_chunks"])
        for n in names:
            entry = by_tool.setdefault(n, {"name": n, "calls": 0, "by_model": {}})
            entry["calls"] += 1
            if r["model"]:
                entry["by_model"][r["model"]] = entry["by_model"].get(r["model"], 0) + 1
    by_tool_list = sorted(
        ({"name": v["name"], "calls": v["calls"], "by_model": v["by_model"]} for v in by_tool.values()),
        key=lambda x: x["calls"],
        reverse=True,
    )

    # Throughput percentile distributions, split into:
    #   - generation rate: completion_tokens / (duration - ttft)  (pure decoding speed)
    #   - processing rate: prompt_tokens / ttft                  (prefill speed)
    #   - combined:       completion_tokens / duration           (legacy single-number metric;
    #                                                             includes prefill in denominator)
    # Filter zero-duration rows and errors. For split rates, also require a recorded TTFT.
    tps_rows = conn.execute(
        """SELECT prompt_tokens, completion_tokens, duration_ms, ttft_ms, upstream FROM requests
           WHERE completion_tokens IS NOT NULL AND completion_tokens > 0
             AND duration_ms IS NOT NULL AND duration_ms > 50
             AND (status IS NULL OR status < 400) AND error IS NULL"""
    ).fetchall()
    combined_values: list[float] = []
    generation_values: list[float] = []
    processing_values: list[float] = []
    # Anthropic batches its SSE output, so (duration − ttft) is transfer time, not
    # decoding time. Exclude it from the generation-rate distribution to keep that metric
    # meaningful. Processing-rate (prefill) and combined-throughput are still valid.
    GENERATION_RATE_UPSTREAMS = {"ollama", "lmstudio"}
    for r in tps_rows:
        ct = r["completion_tokens"] or 0
        pt = r["prompt_tokens"] or 0
        dur = r["duration_ms"] or 0
        ttft = r["ttft_ms"]
        ups = r["upstream"]
        if dur > 0 and ct > 0:
            combined_values.append(ct / (dur / 1000.0))
        if (ttft is not None and ttft > 0 and dur > ttft and ct > 0
                and (ups in GENERATION_RATE_UPSTREAMS or ups is None)):
            # Older rows pre-backfill have ups=None; let those through so we don't lose data.
            generation_values.append(ct / ((dur - ttft) / 1000.0))
        if ttft is not None and ttft > 50 and pt > 0:
            processing_values.append(pt / (ttft / 1000.0))
    combined_values.sort()
    generation_values.sort()
    processing_values.sort()

    def _pct(values, p):
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        idx = (len(values) - 1) * p / 100.0
        lo = int(idx)
        hi = min(lo + 1, len(values) - 1)
        frac = idx - lo
        return values[lo] * (1 - frac) + values[hi] * frac

    def _dist(values):
        return {
            "samples": len(values),
            "min": values[0] if values else None,
            "max": values[-1] if values else None,
            "p10": _pct(values, 10),
            "p25": _pct(values, 25),
            "p50": _pct(values, 50),
            "p75": _pct(values, 75),
            "p90": _pct(values, 90),
        }

    overall["throughput"] = _dist(combined_values)            # legacy / combined
    overall["throughput_generation"] = _dist(generation_values)  # decode rate
    overall["throughput_processing"] = _dist(processing_values)  # prefill rate

    conn.close()
    result = {
        "overall": overall,
        "by_model": by_model,
        "by_path": by_path,
        "by_status": by_status,
        "by_upstream": by_upstream,
        "by_client": by_client,
        "by_app": by_app,
        "by_tool": by_tool_list,
    }
    _STATS_CACHE["ts"] = now
    _STATS_CACHE["data"] = result
    return result


@app.get("/__proxy/api/requests/{req_id}")
async def get_request(req_id: str, request: Request):
    viewer = _client_ip(request)
    conn = db()
    row = conn.execute("SELECT * FROM requests_v WHERE id = ?", (req_id,)).fetchone()
    if not row:
        conn.close()
        return JSONResponse({"error": "not found"}, status_code=404)
    d = dict(row)

    # If this row is itself a shadow, its visibility rides on the primary's client_ip
    # (shadows are derived comparisons, not independent user requests).
    if d.get("shadow_of"):
        parent = conn.execute("SELECT client_ip FROM requests WHERE id=?", (d["shadow_of"],)).fetchone()
        if parent and parent["client_ip"]:
            d["client_ip"] = parent["client_ip"]  # use primary's IP for the redaction gate

    if d.get("request_body"):
        try:
            bj = json.loads(d["request_body"])
            score, reasons = _downsize_score(
                bj, d.get("response_body"), d.get("stream_chunks"),
                _prompt_total_chars(bj),
                d.get("duration_ms"),
            )
            d["downsize_score"] = score
            d["downsize_reasons"] = reasons
        except (json.JSONDecodeError, TypeError):
            pass

    # Attach shadow runs (rows where shadow_of = this id). Shadows ride on the primary's
    # visibility: if the viewer can see the primary's content, they see the shadows verbatim;
    # if not, the shadows are redacted using the PRIMARY's client_ip (not their own).
    shadow_rows = conn.execute(
        """SELECT id, ts, model, status, duration_ms, error,
                  prompt_tokens, completion_tokens, total_tokens,
                  upstream_url, request_body, response_body, stream_chunks
           FROM requests_v WHERE shadow_of = ? ORDER BY ts ASC""",
        (req_id,),
    ).fetchall()
    conn.close()
    primary_visible = _can_view_pii(viewer, d.get("client_ip"))
    if primary_visible:
        shadows = [dict(s) for s in shadow_rows]
    else:
        shadows = [_redact_row(dict(s), viewer, originator_ips=[d.get("client_ip")]) for s in shadow_rows]
    d["shadows"] = shadows
    # Cache verdict for the detail panel (same logic as the list): timing-based for
    # OpenAI-semantics upstreams, token-based for Ollama.
    _cpct, _cverdict = _cache_verdict(d.get("prompt_tokens"), d.get("est_prompt_tokens"),
                                      d.get("ttft_ms"), d.get("prompt_tokens"), d.get("upstream"))
    d["cache_pct"] = _cpct
    d["cache_verdict"] = _cverdict
    # Images: surface each as metadata (served on demand via .../image/{idx}). Prefer the
    # full-fidelity images_data column; fall back to parsing the body for pre-images_data rows.
    d["images"] = []
    _imgs_col = _load_images_data(d.get("images_data"))
    if _imgs_col:
        for i, im in enumerate(_imgs_col):
            b64 = im.get("data") or ""
            d["images"].append({"index": i, "media_type": im.get("media_type") or "image/png",
                                "kind": "data", "size_bytes": len(b64) * 3 // 4})
    elif d.get("request_body"):
        try:
            _bj = json.loads(d["request_body"])
        except (json.JSONDecodeError, TypeError):
            _bj = None
        if isinstance(_bj, dict):
            for (i, mt, kind, payload) in _iter_request_images(_bj):
                if kind == "data":
                    d["images"].append({"index": i, "media_type": mt, "kind": "data",
                                        "size_bytes": len(payload or "") * 3 // 4})
                else:
                    d["images"].append({"index": i, "media_type": mt or "", "kind": "url",
                                        "url": payload})
            if _strip_image_data(_bj):
                d["request_body"] = json.dumps(_bj)
    # Don't ship the raw image column to the client (it's large; images load via the endpoint).
    d.pop("images_data", None)
    return _redact_row(d, viewer)


def _request_image_refs(row):
    """Image (media_type, kind, payload) tuples for a request row, preferring the full-fidelity
    images_data column and falling back to parsing the (possibly stripped/truncated) body for
    rows saved before images_data existed."""
    imgs = _load_images_data(row["images_data"] if "images_data" in row.keys() else None)
    if imgs:
        return [(i, im.get("media_type") or "image/png", "data", im.get("data") or "")
                for i, im in enumerate(imgs)]
    try:
        bj = json.loads(row["request_body"]) if row["request_body"] else None
    except (json.JSONDecodeError, TypeError):
        return None  # unparseable (e.g. truncated) and no images_data → unrecoverable
    return list(_iter_request_images(bj)) if isinstance(bj, dict) else []


@app.get("/__proxy/api/requests/{req_id}/image/{idx}")
def get_request_image(req_id: str, idx: int, request: Request):  # sync → threadpool
    """Reconstruct and serve the idx-th image, from the full-fidelity images_data column."""
    conn = db()
    row = conn.execute("SELECT request_body, images_data, client_ip FROM requests_v WHERE id = ?", (req_id,)).fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "not found"}, status_code=404)
    # Same visibility gate as the request detail: only a viewer who may see this client's content.
    if not _can_view_pii(_client_ip(request), row["client_ip"]):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    refs = _request_image_refs(row)
    if refs is None:
        return JSONResponse({"error": "image not stored (request predates image capture and the body was truncated)"},
                            status_code=422)
    for (i, mt, kind, payload) in refs:
        if i != idx:
            continue
        if kind == "url":
            return JSONResponse({"url": payload}, status_code=200)  # external — client links directly
        try:
            raw = base64.b64decode((payload or "") + "=" * (-len(payload or "") % 4))
        except (ValueError, binascii.Error):
            return JSONResponse({"error": "image data truncated or not valid base64 — full bytes weren't stored"},
                                status_code=422)
        return Response(content=raw, media_type=mt or "image/png",
                        headers={"Cache-Control": "private, max-age=300"})
    return JSONResponse({"error": "image index out of range"}, status_code=404)


@app.post("/__proxy/api/clear")
async def clear_requests():
    conn = db()
    conn.execute("DELETE FROM requests")
    conn.commit()
    conn.close()
    return {"ok": True}


@app.get("/__proxy/api/restart/info")
async def restart_info():
    """Tells the UI whether a restart will actually re-launch the process. systemd sets
    INVOCATION_ID; if absent we're either standalone or under a different supervisor."""
    return {
        "managed_by_systemd": bool(os.environ.get("INVOCATION_ID")),
        "pid": os.getpid(),
        "started_at": _PROCESS_START_TS,
    }


@app.post("/__proxy/api/restart")
async def restart_proxy(request: Request):
    """Exit the process so the supervisor (systemd Restart=on-failure) brings it back up.
    Requires header `X-Confirm: restart-now` to guard against accidental browser GETs / CSRF."""
    if request.headers.get("x-confirm") != "restart-now":
        return JSONResponse(
            {"error": "missing or wrong X-Confirm header (expected 'restart-now')"},
            status_code=400,
        )
    managed = bool(os.environ.get("INVOCATION_ID"))

    async def _exit_soon():
        # Give the response a moment to flush before we tear down.
        await asyncio.sleep(0.5)
        os._exit(1)  # non-zero so systemd Restart=on-failure triggers a relaunch

    asyncio.create_task(_exit_soon())
    return JSONResponse(
        {"ok": True, "managed_by_systemd": managed, "pid": os.getpid()},
        status_code=202,
    )


# -------- Remote control: self-registered endpoints + phone PWA bridge + panic mode --------

@app.post("/__proxy/api/control/register")
async def control_register(request: Request):
    """Self-registration endpoint for control targets (e.g. the VS Code companion extension).
    Idempotent upsert keyed by `name`. Re-call as a heartbeat (last_seen_ts gets updated)."""
    try:
        payload = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"invalid JSON: {e}"}, status_code=400)
    name = (payload.get("name") or "").strip()
    url = (payload.get("url") or "").strip()
    if not name or not url:
        return JSONResponse({"error": "'name' and 'url' are required"}, status_code=400)
    token = payload.get("token") or None
    kind = (payload.get("kind") or "vscode-chat").strip()
    now = time.time()
    conn = db()
    conn.execute(
        """INSERT INTO control_endpoints (name, url, token, kind, registered_ts, last_seen_ts, source)
           VALUES (?, ?, ?, ?, ?, ?, 'auto')
           ON CONFLICT(name) DO UPDATE SET
             url = excluded.url,
             token = COALESCE(excluded.token, control_endpoints.token),
             kind = excluded.kind,
             last_seen_ts = excluded.last_seen_ts""",
        (name, url, token, kind, now, now),
    )
    conn.commit()
    conn.close()
    return {"ok": True, "name": name, "registered_ts": now}


@app.get("/__proxy/api/control/endpoints")
async def control_list_endpoints():
    """Lists all registered control endpoints (auto + manual)."""
    conn = db()
    rows = conn.execute(
        """SELECT name, url, kind, registered_ts, last_seen_ts, source,
                  CASE WHEN token IS NOT NULL THEN 1 ELSE 0 END AS has_token
           FROM control_endpoints ORDER BY last_seen_ts DESC"""
    ).fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows], "panic_mode": _PANIC_MODE}


@app.delete("/__proxy/api/control/endpoints/{name}")
async def control_delete_endpoint(name: str):
    conn = db()
    cur = conn.execute("DELETE FROM control_endpoints WHERE name = ?", (name,))
    conn.commit()
    conn.close()
    return {"ok": True, "removed": cur.rowcount}


@app.post("/__proxy/api/control/chat")
async def control_chat(request: Request):
    """Forward a chat prompt to a registered VS Code endpoint. The phone PWA hits this; the
    proxy looks up the endpoint and POSTs to its /chat with the stored token."""
    try:
        payload = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"invalid JSON: {e}"}, status_code=400)
    prompt = payload.get("prompt") or payload.get("query")
    if not isinstance(prompt, str) or not prompt.strip():
        return JSONResponse({"error": "'prompt' is required"}, status_code=400)
    target = payload.get("target")
    conn = db()
    if target:
        row = conn.execute(
            "SELECT name, url, token FROM control_endpoints WHERE name = ?", (target,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT name, url, token FROM control_endpoints ORDER BY last_seen_ts DESC LIMIT 1"
        ).fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "no control endpoints registered"}, status_code=404)
    fwd_url = row["url"].rstrip("/") + "/chat"
    fwd_headers = {"content-type": "application/json"}
    if row["token"]:
        fwd_headers["authorization"] = f"Bearer {row['token']}"
    fwd_body = {"prompt": prompt}
    for k in ("command", "location", "isPartialQuery", "attachScreenshot",
              "newChat", "submit", "newChatCommand", "submitCommand"):
        if payload.get(k) is not None:
            fwd_body[k] = payload[k]
    client_http: httpx.AsyncClient = request.app.state.client
    send_id = uuid.uuid4().hex[:16]
    send_ts = time.time()
    try:
        r = await client_http.post(
            fwd_url, headers=fwd_headers,
            content=json.dumps(fwd_body).encode("utf-8"),
            timeout=httpx.Timeout(10.0),
        )
        try:
            ext_payload = json.loads(r.text)
        except (json.JSONDecodeError, TypeError):
            ext_payload = {"raw": r.text}
        # Record the send so /api/control/await can correlate the resulting LLM call.
        if r.is_success:
            _CONTROL_SENDS[send_id] = {
                "ts": send_ts,
                "target_name": row["name"],
                "target_ip": _control_target_ip(row["url"]),
                "prompt_hint": prompt[:120],
            }
        return JSONResponse(
            {"ok": r.is_success, "target": row["name"], "status": r.status_code,
             "endpoint_url": row["url"], "extension": ext_payload, "send_id": send_id},
            status_code=r.status_code,
        )
    except Exception as e:
        return JSONResponse(
            {"ok": False, "target": row["name"], "error": str(e),
             "endpoint_url": row["url"], "send_id": send_id},
            status_code=502,
        )


def _extract_response_text(body_text, stream_text) -> str:
    """Best-effort assistant text extraction from either OpenAI or Anthropic shape.
    Used by /api/control/await to stream live responses back to the phone PWA."""
    parts: list[str] = []
    if body_text:
        try:
            j = json.loads(body_text)
            if isinstance(j, dict):
                # OpenAI chat.completion
                for c in (j.get("choices") or []):
                    msg = c.get("message") or {}
                    if isinstance(msg.get("content"), str):
                        parts.append(msg["content"])
                # Anthropic /v1/messages
                if j.get("type") == "message":
                    for blk in (j.get("content") or []):
                        if isinstance(blk, dict) and blk.get("type") == "text" and isinstance(blk.get("text"), str):
                            parts.append(blk["text"])
                # Ollama native
                m = j.get("message")
                if isinstance(m, dict) and isinstance(m.get("content"), str):
                    parts.append(m["content"])
        except (json.JSONDecodeError, TypeError):
            pass
    if stream_text:
        text = _maybe_gunzip(stream_text)
        for line in text.split("\n") if isinstance(text, str) else []:
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                continue
            try:
                j = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                continue
            # OpenAI streaming delta
            for c in (j.get("choices") or []):
                d = c.get("delta") or c.get("message") or {}
                if isinstance(d.get("content"), str):
                    parts.append(d["content"])
            # Anthropic streaming
            if j.get("type") == "content_block_delta":
                d = j.get("delta") or {}
                if d.get("type") == "text_delta" and isinstance(d.get("text"), str):
                    parts.append(d["text"])
            elif j.get("type") == "content_block_start":
                blk = j.get("content_block") or {}
                if blk.get("type") == "text" and isinstance(blk.get("text"), str):
                    parts.append(blk["text"])
            # Ollama streaming
            elif isinstance(j.get("message"), dict) and isinstance(j["message"].get("content"), str):
                parts.append(j["message"]["content"])
    return "".join(parts)


def _extract_response_tool_calls_full(body_text, stream_text) -> list[dict]:
    """Extract tool calls (name + args) from a response in either shape, in arrival order.
    Used by the phone PWA to render 'Read welcome.md' / 'Created welcome.md' style events."""
    out: list[dict] = []
    if body_text:
        try:
            j = json.loads(body_text)
            if isinstance(j, dict):
                for c in (j.get("choices") or []):
                    msg = c.get("message") or {}
                    for tc in (msg.get("tool_calls") or []):
                        fn = (tc.get("function") or {})
                        out.append({
                            "id": tc.get("id"),
                            "name": fn.get("name") or "?",
                            "arguments": fn.get("arguments") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments") or {}),
                        })
                m = j.get("message")
                if isinstance(m, dict):
                    for tc in (m.get("tool_calls") or []):
                        fn = (tc.get("function") or {})
                        out.append({
                            "name": fn.get("name") or "?",
                            "arguments": json.dumps(fn.get("arguments") or {}),
                        })
                if j.get("type") == "message" and isinstance(j.get("content"), list):
                    for blk in j["content"]:
                        if isinstance(blk, dict) and blk.get("type") == "tool_use":
                            out.append({
                                "id": blk.get("id"),
                                "name": blk.get("name") or "?",
                                "arguments": json.dumps(blk.get("input") or {}),
                            })
        except (json.JSONDecodeError, TypeError):
            pass
    if stream_text:
        text = _maybe_gunzip(stream_text)
        if isinstance(text, str):
            tcs: dict = {}
            order: list = []
            for line in text.split("\n"):
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    continue
                try:
                    j = json.loads(data)
                except (json.JSONDecodeError, TypeError):
                    continue
                # OpenAI streaming
                for c in (j.get("choices") or []):
                    delta = c.get("delta") or c.get("message") or {}
                    for tc in (delta.get("tool_calls") or []):
                        idx = ("oai", tc.get("index", len(order)))
                        if idx not in tcs:
                            tcs[idx] = {"name": "", "arguments": "", "id": tc.get("id")}
                            order.append(idx)
                        slot = tcs[idx]
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
                # Anthropic streaming tool_use
                if j.get("type") == "content_block_start":
                    blk = j.get("content_block") or {}
                    if blk.get("type") == "tool_use":
                        idx = ("anth", j.get("index", len(order)))
                        tcs[idx] = {"name": blk.get("name") or "?", "arguments": "", "id": blk.get("id")}
                        order.append(idx)
                elif j.get("type") == "content_block_delta":
                    idx = ("anth", j.get("index"))
                    if idx in tcs:
                        d = j.get("delta") or {}
                        if d.get("type") == "input_json_delta":
                            tcs[idx]["arguments"] += d.get("partial_json") or ""
            for k in order:
                out.append(tcs[k])
    return out


def _control_target_ip(url_str: str | None) -> str | None:
    """Extract host portion of an endpoint URL for client_ip correlation."""
    if not url_str:
        return None
    try:
        from urllib.parse import urlparse
        return urlparse(url_str).hostname
    except (ValueError, TypeError):
        return None


# Context wrappers that VS Code Copilot Chat / Claude Code / Continue inject into the user
# message envelope. Stripped by _clean_user_prompt so the mirror shows what the user actually
# typed instead of the whole context bundle. Order matters — more-specific names checked first.
_USER_PROMPT_TAGS = (
    "userPrompt", "user_prompt",
    "userRequest", "user_request",
    "userMessage", "user_message",
    "userQuery", "user_query",
    "userInput", "user_input",
    "userQuestion", "user_question",
    "currentRequest", "current_request",
    "currentMessage", "current_message",
    "user", "input", "prompt", "question", "request", "message", "query",
)
_CONTEXT_WRAPPERS = (
    # Generic
    "context", "instructions", "tools", "history", "memory", "session",
    # File / code context
    "file", "files", "code_block", "code", "snippet", "selectedText",
    "currentSelection", "selection", "attachment", "attachments",
    "active_file", "activeFile", "open_files", "openFiles", "editor_state",
    "editorState", "currentFile", "current_file",
    # Environment / workspace
    "environment", "workspace", "workspaceState", "workspace_state",
    "repoSummary", "repo_summary", "repositoryContext",
    # Reminders / instructions (Copilot Chat wraps a lot in these)
    "reminder", "system-reminder", "system_reminder", "systemReminder",
    "reminderInstructions", "reminder_instructions",
    "customInstructions", "custom_instructions",
    "userInstructions", "user_instructions",
    "additionalInstructions", "additional_instructions",
    "agentInstructions", "agent_instructions",
    "policyInstructions", "policy_instructions",
    "guidance", "policy",
    # Diagnostics / git / tasks
    "diagnostics", "problems", "errors", "git_status", "gitStatus",
    "git_log", "gitLog", "task", "tasks", "recent_changes", "recentChanges",
    "lintIssues", "lint_issues",
    # Conversation framing
    "conversationContext", "conversation_context", "previousResponse",
    "previous_response", "thoughts", "scratchpad",
)


_BLOCK_TAG_RE = re.compile(
    # Match a paired XML-like block: <Tag attr="...">content</Tag>. Non-greedy so nested
    # blocks of different tag names work via iterative passes.
    r"<\s*([a-zA-Z_][a-zA-Z0-9_:-]*)\b[^>]*?>(?:.*?)</\s*\1\s*>",
    re.DOTALL,
)
_SELF_CLOSING_TAG_RE = re.compile(r"<\s*[a-zA-Z_][a-zA-Z0-9_:-]*\b[^>]*/\s*>")


def _clean_user_prompt(text: str | None) -> tuple[str, list[str], str | None]:
    """Extract the actual user-typed text from a chat-extension-wrapped envelope.
    Strategy: (1) check for explicit user-input markers, (2) protect markdown code blocks,
    (3) iteratively strip every paired XML-like block <Tag>...</Tag> until no more changes,
    (4) strip self-closing tags, (5) restore code blocks. Returns (cleaned, stripped_tags, raw)."""
    if not text:
        return "", [], None
    raw = text

    # 1. Explicit user-input markers — strongest signal.
    for tag in _USER_PROMPT_TAGS:
        m = re.search(
            rf"<\s*{tag}\b[^>]*>(.*?)</\s*{tag}\s*>",
            text, re.DOTALL | re.IGNORECASE,
        )
        if m:
            inner = m.group(1).strip()
            if inner:
                return inner[:4000], ["(extracted from <" + tag + ">)"], raw

    # 2. Protect markdown code blocks (``` fences and `inline`) so we don't strip XML
    # the user actually typed inside their code samples.
    code_blocks: list[str] = []
    def _protect(m):
        code_blocks.append(m.group(0))
        return f"\x00CODEBLK{len(code_blocks) - 1}\x00"
    protected = re.sub(r"```[a-zA-Z0-9_+-]*\n?[\s\S]*?```", _protect, text)
    protected = re.sub(r"`[^`\n]{1,200}`", _protect, protected)

    # 3. Iteratively strip every paired XML-like block, REMEMBERING each block's content
    # so we can fall back to it if all the text was inside wrappers.
    stripped: list[str] = []
    block_contents: dict[str, str] = {}    # name(lower) -> last-seen inner text
    out = protected
    for _ in range(8):
        before = out
        for m in _BLOCK_TAG_RE.finditer(out):
            tname = m.group(1).lower()
            stripped.append(tname)
            # Capture inner content so we can recover it if everything got stripped.
            full = m.group(0)
            inner = full[full.index(">") + 1 : full.rindex("<")]
            if inner.strip():
                block_contents[tname] = inner.strip()
        out = _BLOCK_TAG_RE.sub("", out)
        if out == before:
            break

    # 4. Self-closing tags.
    out = _SELF_CLOSING_TAG_RE.sub("", out)

    # 5. Restore protected code blocks.
    def _restore(m):
        idx = int(m.group(1))
        return code_blocks[idx] if 0 <= idx < len(code_blocks) else m.group(0)
    out = re.sub(r"\x00CODEBLK(\d+)\x00", _restore, out)

    out = re.sub(r"\n[ \t]*\n[ \t]*(\n[ \t]*)+", "\n\n", out).strip()

    # Dedup stripped-tags list (preserve first-seen order).
    seen: list[str] = []
    for s in stripped:
        if s not in seen:
            seen.append(s)

    # If text outside wrappers exists, that's the prompt.
    if out:
        return out[:4000], seen, raw

    # Otherwise try to recover from a captured wrapper. Prefer ones that look like user input.
    user_priority = [t.lower() for t in _USER_PROMPT_TAGS]
    candidates = [t for t in user_priority if t in block_contents] + \
                 [t for t in seen if t not in user_priority and t in block_contents]
    for cand in candidates:
        return block_contents[cand][:4000], seen + [f"(extracted from <{cand}>)"], raw

    # Truly nothing — return empty so the UI can render a clear "no prompt" placeholder
    # rather than dumping the raw envelope.
    if seen:
        return "", seen, raw
    # No wrappers were detected at all; return the original.
    return text.strip()[:4000], [], raw


def _classify_turn_origin(req_ts: float, target_ip: str | None) -> str:
    """Was this chat turn triggered from the phone (recent control send) or typed locally?
    Wide window because Copilot Chat can take 30+ seconds between the user pressing send
    and the actual upstream API call firing (especially on cold-start). Underclassifying as
    'local' would break the phone PWA's placeholder-reconcile logic (visible as echo)."""
    if not target_ip:
        return "local"
    for info in _CONTROL_SENDS.values():
        if info.get("target_ip") == target_ip:
            sent_ts = info.get("ts", 0)
            # Match if the feed turn arrived within 60s after the send (or within 5s before,
            # to absorb minor clock skew).
            if -5 <= (req_ts - sent_ts) <= 60:
                return "phone"
    return "local"


_FEED_APPS = ("vscode-copilot", "claude-code", "github-copilot", "continue.dev", "cursor", "vscode")


@app.get("/__proxy/api/control/feed")
async def control_feed(target: str | None = None, since: float | None = None,
                        limit: int = 30, conversation_id: str | None = None):
    """Live-mirror feed of recent chat turns from registered editor endpoints.

    Each item is one round-trip (request → response) shaped as a chat turn. Filterable by
    `conversation_id` so the PWA can pin to a single session. The `sessions` field in the
    response lists every conversation_id seen in the broader window so the UI can offer a
    session picker."""
    conn = db()
    target_ip: str | None = None
    target_url: str | None = None
    if target:
        row = conn.execute("SELECT url FROM control_endpoints WHERE name = ?", (target,)).fetchone()
        if row:
            target_url = row["url"]
            target_ip = _control_target_ip(row["url"])
    where = ["shadow_of IS NULL", f"client_app IN ({','.join('?' * len(_FEED_APPS))})"]
    params: list = list(_FEED_APPS)
    if target_ip:
        where.append("client_ip = ?")
        params.append(target_ip)

    # Session picker window: last 24 hours so sessions you've worked on across the day all
    # appear. Capped at 30 entries to keep the dropdown manageable.
    sess_window = time.time() - 24 * 3600
    sess_where = list(where) + ["ts > ?", "conversation_id IS NOT NULL"]
    sess_params = list(params) + [sess_window]
    sess_sql = (
        "SELECT conversation_id, MAX(ts) AS last_ts, MIN(ts) AS first_ts, "
        "       COUNT(*) AS turn_count, "
        "       (SELECT request_body FROM requests_v "
        "        WHERE conversation_id = r.conversation_id AND request_body IS NOT NULL "
        "        ORDER BY ts ASC LIMIT 1) AS first_body "
        "FROM requests r WHERE " + " AND ".join(sess_where)
        + " GROUP BY conversation_id ORDER BY last_ts DESC LIMIT 30"
    )
    sess_rows = conn.execute(sess_sql, sess_params).fetchall()
    sessions = []
    for sr in sess_rows:
        first_user = ""
        try:
            body = json.loads(sr["first_body"]) if sr["first_body"] else None
        except (json.JSONDecodeError, TypeError):
            body = None
        if isinstance(body, dict):
            # Prefer the first user-typed prompt (skips Copilot's <environment_info>-only
            # first message); fall back to cleaned first user content for non-Copilot shapes.
            typed = _first_typed_user_prompt(body)
            if typed:
                first_user = typed[:120]
            else:
                for m in (body.get("messages") or []):
                    if isinstance(m, dict) and m.get("role") == "user":
                        cleaned, _, _ = _clean_user_prompt(_msg_text(m))
                        if cleaned:
                            first_user = cleaned[:120]
                        break
        # Honor custom labels set via /api/control/session-label.
        label_row = conn.execute(
            "SELECT label FROM conversation_labels WHERE conversation_id = ?",
            (sr["conversation_id"],),
        ).fetchone()
        custom_label = label_row["label"] if label_row else None
        sessions.append({
            "conversation_id": sr["conversation_id"],
            "first_ts": sr["first_ts"],
            "last_ts": sr["last_ts"],
            "turn_count": sr["turn_count"],
            "first_user": first_user or "(no user prompt)",
            "label": custom_label,
        })

    # Now the actual feed query.
    if conversation_id:
        where.append("conversation_id = ?")
        params.append(conversation_id)
    if since is not None:
        where.append("ts > ?")
        params.append(float(since))
    # Only chat round-trips, not utility endpoints (model discovery, embeddings, etc.).
    where.append(
        "(path LIKE '%/v1/chat/completions%' OR path LIKE '%/v1/messages%' "
        "OR path LIKE '%/v1/complete%' OR path LIKE '/api/chat%')"
    )
    sql = (
        "SELECT id, ts, model, status, prompt_tokens, completion_tokens, total_tokens, "
        "duration_ms, request_body, response_body, stream_chunks, error, client_app, "
        "client_ip, conversation_id "
        "FROM requests_v WHERE " + " AND ".join(where) + " ORDER BY ts ASC LIMIT ?"
    )
    params.append(max(1, min(int(limit), 100)))
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    items = []
    newest_ts = since or 0.0
    for r in rows:
        last_user = ""
        try:
            body = json.loads(r["request_body"]) if r["request_body"] else None
        except (json.JSONDecodeError, TypeError):
            body = None
        if isinstance(body, dict):
            msgs = body.get("messages") or []
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("role") == "user":
                    last_user = _msg_text(m)
                    break
        cleaned, stripped, raw = _clean_user_prompt(last_user)
        text = _extract_response_text(r["response_body"], r["stream_chunks"])
        tool_calls = _extract_response_tool_calls_full(r["response_body"], r["stream_chunks"])
        if r["status"] is None and not r["error"]:
            state = "streaming"
        elif r["error"]:
            state = "error"
        else:
            state = "complete"
        items.append({
            "id": r["id"],
            "ts": r["ts"],
            "model": r["model"],
            "client_app": r["client_app"],
            "conversation_id": r["conversation_id"],
            "status": state,
            "via": _classify_turn_origin(r["ts"] or 0.0, target_ip),
            "prompt": cleaned,
            "prompt_raw": raw if raw and raw != cleaned else None,
            "context_blocks": stripped,
            "text": text,
            "tool_calls": tool_calls,
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "total_tokens": r["total_tokens"],
            "duration_ms": r["duration_ms"],
            "error": r["error"],
        })
        if (r["ts"] or 0.0) > newest_ts:
            newest_ts = r["ts"]
    return {
        "items": items,
        "newest_ts": newest_ts,
        "target_url": target_url,
        "sessions": sessions,
        "active_session": conversation_id,
        "panic_mode": _PANIC_MODE,
    }


@app.get("/__proxy/api/control/await/{send_id}")
async def control_await(send_id: str):
    """Polled by the phone PWA after a /api/control/chat send. Looks for the LLM API call
    that VS Code's chat triggered as a result, and returns its response (live or final).
    Returns one of: {state: 'pending'|'streaming'|'complete'|'error'|'unknown'}."""
    # Prune sends older than 10 minutes so the in-memory map can't grow unbounded.
    cutoff = time.time() - 600
    stale = [k for k, v in _CONTROL_SENDS.items() if v.get("ts", 0) < cutoff]
    for k in stale:
        _CONTROL_SENDS.pop(k, None)

    info = _CONTROL_SENDS.get(send_id)
    if not info:
        return JSONResponse({"state": "unknown", "error": "send_id not found or expired"}, status_code=404)
    target_ip = info.get("target_ip")
    elapsed = time.time() - info["ts"]
    conn = db()
    rows = conn.execute(
        """SELECT id, ts, status, model, prompt_tokens, completion_tokens, total_tokens,
                  response_body, stream_chunks, error, duration_ms
           FROM requests_v
           WHERE ts > ? AND client_ip = ?
             AND client_app IN ('vscode-copilot','claude-code','github-copilot','continue.dev','cursor')
             AND shadow_of IS NULL
           ORDER BY ts ASC LIMIT 1""",
        (info["ts"] - 1, target_ip or ""),
    ).fetchall()
    conn.close()
    if not rows:
        return {"state": "pending", "elapsed_s": round(elapsed, 1), "target_ip": target_ip}
    r = rows[0]
    text = _extract_response_text(r["response_body"], r["stream_chunks"])
    tool_calls = _extract_response_tool_calls_full(r["response_body"], r["stream_chunks"])
    if r["status"] is None and not r["error"]:
        return {
            "state": "streaming", "request_id": r["id"], "model": r["model"],
            "elapsed_s": round(elapsed, 1), "text": text, "tool_calls": tool_calls,
        }
    return {
        "state": "error" if r["error"] else "complete",
        "request_id": r["id"], "model": r["model"],
        "duration_ms": r["duration_ms"],
        "prompt_tokens": r["prompt_tokens"],
        "completion_tokens": r["completion_tokens"],
        "total_tokens": r["total_tokens"],
        "text": text,
        "tool_calls": tool_calls,
        "error": r["error"],
        "elapsed_s": round(elapsed, 1),
    }


# -------- Tool permission queue (used by the @proxy chat participant) --------

def _tool_permission_lookup(tool_name: str, args_str: str) -> str | None:
    """Check persistent allow/deny rules. Returns 'allow', 'deny', or None.
    Patterns: exact tool name match, or 'tool_name:<argprefix>' for command-prefix rules."""
    if not tool_name:
        return None
    conn = db()
    rows = conn.execute(
        "SELECT pattern, decision FROM tool_permissions WHERE pattern = ? OR pattern LIKE ? || '%'",
        (tool_name, tool_name + ":"),
    ).fetchall()
    conn.close()
    for r in rows:
        pat = r["pattern"]
        if pat == tool_name:
            return r["decision"]
        # Prefix rules: pattern is "bash:npm test", argument starts with "npm test"
        if pat.startswith(tool_name + ":"):
            prefix = pat[len(tool_name) + 1:]
            if isinstance(args_str, str) and args_str.startswith(prefix):
                return r["decision"]
    return None


@app.post("/__proxy/api/control/pending-tool")
async def control_pending_tool_register(request: Request):
    """Extension calls this when its LLM emits a tool call requiring approval. Returns a
    pending_id; the extension then polls /pending-tool/{id} until decided. Hits a persistent
    allow/deny rule first (auto-decided immediately if matched)."""
    try:
        payload = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"invalid JSON: {e}"}, status_code=400)
    name = (payload.get("tool_name") or "").strip()
    args = payload.get("arguments")
    if isinstance(args, dict):
        args_str = json.dumps(args)
    elif args is None:
        args_str = ""
    else:
        args_str = str(args)
    summary = (payload.get("summary") or "").strip() or f"{name}({args_str[:120]})"
    if not name:
        return JSONResponse({"error": "'tool_name' required"}, status_code=400)
    source = (payload.get("source") or "extension").strip()
    pending_id = uuid.uuid4().hex[:16]

    # Auto-decide via persistent rules.
    rule = _tool_permission_lookup(name, args_str)
    decision = rule if rule in ("allow", "deny") else None
    auto_rule_label = rule if rule else None
    # If this tool call originates while a queued agent task is running, fall back to
    # that task's tool_approval_mode when no persistent rule applied.
    if decision is None and _CURRENT_AGENT_TASK:
        mode = _CURRENT_AGENT_TASK.get("mode")
        if mode == "yolo":
            decision = "allow"
            auto_rule_label = "task:yolo"
        elif mode == "rules-only":
            decision = "deny"
            auto_rule_label = "task:rules-only"
        # 'notify-phone' (and unset) → fall through to existing wait-for-approval flow.

    _PENDING_TOOLS[pending_id] = {
        "id": pending_id,
        "ts": time.time(),
        "source": source,
        "tool_name": name,
        "arguments": args_str,
        "summary": summary[:300],
        "decision": decision,
        "decided_ts": time.time() if decision else None,
        "auto_rule": auto_rule_label,
    }
    return {"id": pending_id, "decision": decision, "auto_rule": auto_rule_label}


@app.get("/__proxy/api/control/pending-tool/{pending_id}")
async def control_pending_tool_status(pending_id: str):
    info = _PENDING_TOOLS.get(pending_id)
    if not info:
        return JSONResponse({"error": "not found"}, status_code=404)
    return info


@app.get("/__proxy/api/control/pending-tools")
async def control_pending_tools_list():
    """List undecided pending tool calls (for the phone PWA to show approve/deny prompts).
    Also auto-prunes entries older than 10 minutes."""
    cutoff = time.time() - 600
    for k in list(_PENDING_TOOLS.keys()):
        info = _PENDING_TOOLS[k]
        if info.get("ts", 0) < cutoff and info.get("decision") is not None:
            _PENDING_TOOLS.pop(k, None)
    pending = [v for v in _PENDING_TOOLS.values() if v.get("decision") is None]
    pending.sort(key=lambda v: v.get("ts", 0))
    return {"items": pending}


@app.post("/__proxy/api/control/tool-decision/{pending_id}")
async def control_tool_decision(pending_id: str, request: Request):
    """Phone PWA POSTs the user's decision. Body: {"decision": "allow"|"deny"|"always_allow"|"always_deny"}.
    'always_*' also writes a persistent rule to tool_permissions."""
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    decision = (payload.get("decision") or "").strip()
    if decision not in ("allow", "deny", "always_allow", "always_deny"):
        return JSONResponse({"error": "decision must be allow/deny/always_allow/always_deny"}, status_code=400)
    info = _PENDING_TOOLS.get(pending_id)
    if not info:
        return JSONResponse({"error": "pending_id not found"}, status_code=404)
    short = "allow" if decision in ("allow", "always_allow") else "deny"
    info["decision"] = short
    info["decided_ts"] = time.time()
    info["decided_via"] = decision
    if decision in ("always_allow", "always_deny"):
        # Persistent rule. Pattern format: "tool_name" or "tool_name:argprefix" — for now,
        # use exact-tool-name rules; argument-prefix rules can be added by /tool-permissions.
        conn = db()
        conn.execute(
            """INSERT OR REPLACE INTO tool_permissions (pattern, decision, created_ts)
               VALUES (?, ?, ?)""",
            (info["tool_name"], short, time.time()),
        )
        conn.commit()
        conn.close()
    return {"ok": True, "decision": short}


@app.get("/__proxy/api/control/tool-permissions")
async def control_tool_permissions_list():
    conn = db()
    rows = conn.execute(
        "SELECT pattern, decision, created_ts FROM tool_permissions ORDER BY created_ts DESC"
    ).fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows]}


@app.delete("/__proxy/api/control/tool-permissions/{pattern}")
async def control_tool_permissions_delete(pattern: str):
    conn = db()
    cur = conn.execute("DELETE FROM tool_permissions WHERE pattern = ?", (pattern,))
    conn.commit()
    conn.close()
    return {"ok": True, "removed": cur.rowcount}


# -------- Custom session labels (extension scrapes / user renames) --------

@app.get("/__proxy/api/control/session-label/{conversation_id}")
async def control_session_label_get(conversation_id: str):
    conn = db()
    row = conn.execute(
        "SELECT label, updated_ts FROM conversation_labels WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else {"label": None}


@app.post("/__proxy/api/control/session-label/{conversation_id}")
async def control_session_label_set(conversation_id: str, request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    label = (payload.get("label") or "").strip()
    conn = db()
    if label:
        conn.execute(
            """INSERT OR REPLACE INTO conversation_labels (conversation_id, label, updated_ts)
               VALUES (?, ?, ?)""",
            (conversation_id, label[:200], time.time()),
        )
    else:
        conn.execute("DELETE FROM conversation_labels WHERE conversation_id = ?", (conversation_id,))
    conn.commit()
    conn.close()
    return {"ok": True, "label": label or None}


@app.get("/__proxy/api/control/panic")
async def control_panic_status():
    return {"panic": _PANIC_MODE}


@app.get("/__proxy/api/control/inflight")
async def control_inflight_list():
    """List in-flight requests (those currently streaming from upstream). Used by the UI's
    Kill button to know which requests can still be cancelled."""
    now = time.time()
    items = []
    for rid, info in _INFLIGHT_REQUESTS.items():
        items.append({
            "id": rid,
            "ts": info.get("ts"),
            "elapsed_s": round(now - info.get("ts", now), 1),
            "cancelled": bool(info.get("cancelled")),
        })
    return {"items": sorted(items, key=lambda x: x["ts"] or 0, reverse=True)}


@app.post("/__proxy/api/control/cancel/{req_id}")
async def control_cancel_request(req_id: str):
    """Kill a specific in-flight request by closing its upstream connection. Ollama 0.21+
    notices the client drop and aborts generation, freeing the GPU slot. The proxy's
    streamer loop sees the close and ends — the client gets a truncated stream (or an
    abrupt connection close if no bytes were sent yet)."""
    info = _INFLIGHT_REQUESTS.get(req_id)
    if not info:
        return JSONResponse({"error": "not in-flight (already completed or never existed)"}, status_code=404)
    if info.get("cancelled"):
        return {"ok": True, "already_cancelled": True}
    info["cancelled"] = True
    upstream_resp = info.get("upstream_resp")
    if upstream_resp is not None:
        try:
            await upstream_resp.aclose()
        except Exception:
            pass
    return {"ok": True, "req_id": req_id}


@app.post("/__proxy/api/control/panic")
async def control_panic_set(request: Request):
    """Toggle panic mode. Body: {"on": bool}. While on, every proxied request returns 503
    (the proxy itself stays up so this endpoint remains reachable to disable it)."""
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    on = bool(payload.get("on", not _PANIC_MODE))  # if `on` not provided, toggle current
    _set_panic_mode(on)
    return {"ok": True, "panic": _PANIC_MODE}


# -------- Task queue REST API --------

_VALID_TASK_MODES = {"chat", "agent"}
_VALID_APPROVAL_MODES = {"rules-only", "notify-phone", "yolo"}


def _task_visible_to(viewer_ip: str | None, creator_ip: str | None) -> bool:
    """Same subnet visibility rule as PII redaction: a viewer sees a task only if the
    viewer and the creator share a subnet (or either is loopback). Tasks with no creator_ip
    (legacy rows) are visible to everyone — they predate this column."""
    if not creator_ip:
        return True
    return _ips_share_subnet(viewer_ip, creator_ip)


def _task_check_or_404(task_id: int, request: Request) -> JSONResponse | None:
    """Return a 404 JSONResponse if the task doesn't exist OR isn't visible to the caller's
    subnet (mirrors the PII redaction policy). Returns None if the caller may proceed."""
    conn = db()
    row = conn.execute("SELECT creator_ip FROM proxy_tasks WHERE id=?", (task_id,)).fetchone()
    conn.close()
    if not row or not _task_visible_to(_client_ip(request), row["creator_ip"]):
        return JSONResponse({"error": "not found"}, status_code=404)
    return None


def _task_row_to_dict(r) -> dict:
    if not r:
        return {}
    d = dict(r)
    d["enabled"] = bool(d.get("enabled"))
    return d


@app.post("/__proxy/api/control/tasks")
async def control_task_create(request: Request):
    """Create a queued task. Body: {prompt, mode, target_endpoint?, model?, schedule?,
    tool_approval_mode?}. mode='chat'|'agent'. schedule null = one-shot, else cron expr or
    'every Nm/Nh/Nd' for recurring."""
    try:
        payload = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"invalid JSON: {e}"}, status_code=400)
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"error": "'prompt' is required"}, status_code=400)
    mode = (payload.get("mode") or "chat").strip().lower()
    if mode not in _VALID_TASK_MODES:
        return JSONResponse({"error": f"mode must be one of {sorted(_VALID_TASK_MODES)}"}, status_code=400)
    target_endpoint = (payload.get("target_endpoint") or "").strip() or None
    model = (payload.get("model") or "").strip() or None
    schedule = (payload.get("schedule") or "").strip() or None
    approval = (payload.get("tool_approval_mode") or "").strip() or None
    if approval and approval not in _VALID_APPROVAL_MODES:
        return JSONResponse(
            {"error": f"tool_approval_mode must be one of {sorted(_VALID_APPROVAL_MODES)}"},
            status_code=400,
        )
    if mode == "agent" and not approval:
        approval = "notify-phone"
    now = time.time()
    next_run = _task_compute_next_run(schedule, now) if schedule else None
    if schedule and next_run is None:
        return JSONResponse(
            {"error": f"schedule {schedule!r} could not be parsed (use 'every 30m', 'every 2h', "
                      f"'every 1d', or a 5-field cron expr if croniter is installed)"},
            status_code=400,
        )
    creator_ip = _client_ip(request)
    conn = db()
    cur = conn.execute(
        """INSERT INTO proxy_tasks
           (prompt, mode, target_endpoint, model, status, created_ts,
            schedule, next_run_ts, tool_approval_mode, enabled, creator_ip)
           VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, 1, ?)""",
        (prompt, mode, target_endpoint, model, now, schedule, next_run, approval, creator_ip),
    )
    task_id = cur.lastrowid
    conn.commit()
    row = conn.execute("SELECT * FROM proxy_tasks WHERE id=?", (task_id,)).fetchone()
    conn.close()
    return _task_row_to_dict(row)


@app.get("/__proxy/api/control/tasks")
async def control_task_list(request: Request, status: str = "", recurring_only: int = 0, limit: int = 100):
    """List tasks, newest first. Query: status (comma-separated filter), recurring_only=1
    to show only schedule-bearing parents, limit (max 500). Same subnet visibility rule as
    PII redaction: each viewer only sees tasks created by clients in their subnet."""
    limit = max(1, min(int(limit or 100), 500))
    where = []
    params: list = []
    if status:
        statuses = [s.strip() for s in status.split(",") if s.strip()]
        if statuses:
            where.append("status IN (" + ",".join("?" for _ in statuses) + ")")
            params.extend(statuses)
    if recurring_only:
        where.append("schedule IS NOT NULL")
    sql = "SELECT * FROM proxy_tasks"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_ts DESC LIMIT ?"
    params.append(limit)
    conn = db()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    viewer = _client_ip(request)
    visible = [r for r in rows if _task_visible_to(viewer, r["creator_ip"])]
    return {"items": [_task_row_to_dict(r) for r in visible]}


@app.get("/__proxy/api/control/tasks/{task_id}")
async def control_task_get(task_id: int, request: Request):
    conn = db()
    row = conn.execute("SELECT * FROM proxy_tasks WHERE id=?", (task_id,)).fetchone()
    children_rows = []
    if row and row["schedule"]:
        children_rows = conn.execute(
            "SELECT * FROM proxy_tasks WHERE parent_task_id=? ORDER BY created_ts DESC LIMIT 50",
            (task_id,),
        ).fetchall()
    conn.close()
    if not row:
        return JSONResponse({"error": "not found"}, status_code=404)
    viewer = _client_ip(request)
    if not _task_visible_to(viewer, row["creator_ip"]):
        return JSONResponse({"error": "not found"}, status_code=404)
    out = _task_row_to_dict(row)
    children = [_task_row_to_dict(r) for r in children_rows if _task_visible_to(viewer, r["creator_ip"])]
    if children:
        out["children"] = children
    return out


@app.delete("/__proxy/api/control/tasks/{task_id}")
async def control_task_delete(task_id: int, request: Request):
    blocked = _task_check_or_404(task_id, request)
    if blocked: return blocked
    conn = db()
    cur = conn.execute("DELETE FROM proxy_tasks WHERE id=?", (task_id,))
    conn.commit()
    conn.close()
    return {"ok": True, "removed": cur.rowcount}


@app.post("/__proxy/api/control/tasks/{task_id}/cancel")
async def control_task_cancel(task_id: int, request: Request):
    """Mark a pending or running task as cancelled. (Running agent task: the worker
    coroutine doesn't get aborted mid-call, but its result will be discarded by the
    final-status check inside _task_execute.)"""
    blocked = _task_check_or_404(task_id, request)
    if blocked: return blocked
    conn = db()
    cur = conn.execute(
        """UPDATE proxy_tasks SET status='cancelled', finished_ts=?
           WHERE id=? AND status IN ('pending','running')""",
        (time.time(), task_id),
    )
    conn.commit()
    conn.close()
    return {"ok": True, "updated": cur.rowcount}


@app.post("/__proxy/api/control/tasks/{task_id}/pause")
async def control_task_pause(task_id: int, request: Request):
    blocked = _task_check_or_404(task_id, request)
    if blocked: return blocked
    conn = db()
    cur = conn.execute("UPDATE proxy_tasks SET enabled=0 WHERE id=?", (task_id,))
    conn.commit()
    conn.close()
    return {"ok": True, "updated": cur.rowcount}


@app.post("/__proxy/api/control/tasks/{task_id}/resume")
async def control_task_resume(task_id: int, request: Request):
    blocked = _task_check_or_404(task_id, request)
    if blocked: return blocked
    conn = db()
    cur = conn.execute("UPDATE proxy_tasks SET enabled=1 WHERE id=?", (task_id,))
    conn.commit()
    conn.close()
    return {"ok": True, "updated": cur.rowcount}


@app.post("/__proxy/api/control/tasks/{task_id}/run-now")
async def control_task_run_now(task_id: int, request: Request):
    """Force a recurring task to fire immediately. Spawns a one-shot child without
    advancing the parent's next_run_ts."""
    blocked = _task_check_or_404(task_id, request)
    if blocked: return blocked
    conn = db()
    parent = conn.execute("SELECT * FROM proxy_tasks WHERE id=?", (task_id,)).fetchone()
    if not parent:
        conn.close()
        return JSONResponse({"error": "not found"}, status_code=404)
    if not parent["schedule"]:
        conn.close()
        return JSONResponse({"error": "task is not recurring"}, status_code=400)
    cur = conn.execute(
        """INSERT INTO proxy_tasks
           (prompt, mode, target_endpoint, model, status, created_ts,
            parent_task_id, tool_approval_mode, creator_ip)
           VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?)""",
        (parent["prompt"], parent["mode"], parent["target_endpoint"],
         parent["model"], time.time(), parent["id"], parent["tool_approval_mode"],
         parent["creator_ip"]),
    )
    child_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {"ok": True, "child_task_id": child_id}


# -------- Benchmark API --------

def _bench_visible_to(viewer_ip: str | None, creator_ip: str | None) -> bool:
    if not creator_ip:
        return True
    return _ips_share_subnet(viewer_ip, creator_ip)


@app.post("/__proxy/api/bench/run")
async def bench_run(request: Request):
    """Queue a benchmark. Body: {model, runs?, max_tokens?, prompt_tokens?, concurrency?,
    randomize?, exclusive?, drain_seconds?}. Returns the bench id; poll /api/bench/runs/{id}
    for progress and results."""
    try:
        payload = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"invalid JSON: {e}"}, status_code=400)
    model = (payload.get("model") or "").strip()
    if not model:
        return JSONResponse({"error": "'model' is required"}, status_code=400)
    config = {
        "runs": int(payload.get("runs", 5) or 5),
        "max_tokens": int(payload.get("max_tokens", 256) or 256),
        "prompt_tokens": int(payload.get("prompt_tokens", 0) or 0),
        "concurrency": int(payload.get("concurrency", 1) or 1),
        "randomize": bool(payload.get("randomize", False)),
        "exclusive": bool(payload.get("exclusive", False)),
        "drain_seconds": float(payload.get("drain_seconds", 5.0) or 0.0),
    }
    bench_id = "b_" + uuid.uuid4().hex[:12]
    creator_ip = _client_ip(request)
    conn = db()
    conn.execute(
        """INSERT INTO bench_runs (id, ts, model, config_json, status, creator_ip)
           VALUES (?, ?, ?, ?, 'pending', ?)""",
        (bench_id, time.time(), model, json.dumps(config), creator_ip),
    )
    conn.commit()
    conn.close()
    asyncio.create_task(_bench_execute(bench_id, request.app))
    return {"id": bench_id, "model": model, "config": config}


@app.get("/__proxy/api/bench/runs")
async def bench_runs_list(request: Request, limit: int = 50):
    limit = max(1, min(int(limit or 50), 200))
    conn = db()
    rows = conn.execute(
        "SELECT id, ts, model, config_json, status, progress, progress_total, "
        "started_ts, finished_ts, error, creator_ip "
        "FROM bench_runs ORDER BY ts DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    viewer = _client_ip(request)
    items = []
    for r in rows:
        if not _bench_visible_to(viewer, r["creator_ip"]):
            continue
        d = dict(r)
        try:
            d["config"] = json.loads(d.pop("config_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            d["config"] = {}
        items.append(d)
    return {"items": items}


@app.get("/__proxy/api/bench/runs/{bench_id}")
async def bench_run_get(bench_id: str, request: Request):
    conn = db()
    row = conn.execute("SELECT * FROM bench_runs WHERE id=?", (bench_id,)).fetchone()
    conn.close()
    if not row or not _bench_visible_to(_client_ip(request), row["creator_ip"]):
        return JSONResponse({"error": "not found"}, status_code=404)
    d = dict(row)
    try:
        d["config"] = json.loads(d.pop("config_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        d["config"] = {}
    try:
        d["results"] = json.loads(d.pop("results_json") or "null")
    except (json.JSONDecodeError, TypeError):
        d["results"] = None
    return d


@app.delete("/__proxy/api/bench/runs/{bench_id}")
async def bench_run_delete(bench_id: str, request: Request):
    conn = db()
    row = conn.execute("SELECT creator_ip FROM bench_runs WHERE id=?", (bench_id,)).fetchone()
    if not row or not _bench_visible_to(_client_ip(request), row["creator_ip"]):
        conn.close()
        return JSONResponse({"error": "not found"}, status_code=404)
    conn.execute("DELETE FROM bench_runs WHERE id=?", (bench_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


# -------- Chat personalities (server-side, subnet-scoped) --------

def _personality_visible_to(viewer_ip: str | None, creator_ip: str | None) -> bool:
    """Same subnet-visibility rule as tasks/PII redaction."""
    if not creator_ip:
        return True
    return _ips_share_subnet(viewer_ip, creator_ip)


@app.get("/__proxy/api/memory/{scope:path}")
async def memory_get(scope: str):
    """Return all memory entries for a scope. Sorted by most recently updated."""
    if not scope or len(scope) > 200:
        return JSONResponse({"error": "invalid scope"}, status_code=400)
    conn = db()
    rows = conn.execute(
        "SELECT key, value, created_ts, updated_ts FROM proxy_memory "
        "WHERE conversation_id = ? ORDER BY updated_ts DESC",
        (scope,),
    ).fetchall()
    conn.close()
    items = [dict(r) for r in rows]
    return {"scope": scope, "count": len(items), "items": items}


@app.delete("/__proxy/api/memory/{scope:path}/key/{key:path}")
async def memory_delete_key(scope: str, key: str):
    """Delete a single memory entry by key under a scope."""
    if not scope or not key or len(scope) > 200 or len(key) > 500:
        return JSONResponse({"error": "invalid scope or key"}, status_code=400)
    conn = db()
    cur = conn.execute(
        "DELETE FROM proxy_memory WHERE conversation_id = ? AND key = ?", (scope, key)
    )
    conn.commit()
    conn.close()
    return {"ok": True, "removed": cur.rowcount}


@app.delete("/__proxy/api/memory/{scope:path}")
async def memory_delete(scope: str):
    """Erase all memory entries for a scope (e.g. 'pers:p_abc12345' wipes a personality)."""
    if not scope or len(scope) > 200:
        return JSONResponse({"error": "invalid scope"}, status_code=400)
    conn = db()
    cur = conn.execute(
        "DELETE FROM proxy_memory WHERE conversation_id = ?", (scope,)
    )
    conn.commit()
    conn.close()
    return {"ok": True, "removed": cur.rowcount}


@app.get("/__proxy/api/personalities")
async def personalities_list(request: Request):
    viewer = _client_ip(request)
    conn = db()
    rows = conn.execute(
        "SELECT id, name, prompt, created_ts, updated_ts, creator_ip "
        "FROM proxy_personalities ORDER BY name COLLATE NOCASE"
    ).fetchall()
    conn.close()
    items = [dict(r) for r in rows if _personality_visible_to(viewer, r["creator_ip"])]
    return {"items": items}


@app.post("/__proxy/api/personalities")
async def personalities_create(request: Request):
    """Create a personality. Body: {name, prompt, id?}. id is optional — if supplied (used
    by the localStorage migration path), it must match [a-zA-Z0-9_-]{1,64}; if absent, a
    new id is generated."""
    try:
        payload = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"invalid JSON: {e}"}, status_code=400)
    name = (payload.get("name") or "").strip()
    prompt_text = (payload.get("prompt") or "").strip()
    if not name:
        return JSONResponse({"error": "'name' is required"}, status_code=400)
    if not prompt_text:
        return JSONResponse({"error": "'prompt' is required"}, status_code=400)
    pid = (payload.get("id") or "").strip() or ("p_" + uuid.uuid4().hex[:8])
    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", pid):
        return JSONResponse({"error": "invalid id (allowed: a-z, A-Z, 0-9, _, -)"}, status_code=400)
    creator_ip = _client_ip(request)
    now = time.time()
    conn = db()
    try:
        conn.execute(
            """INSERT INTO proxy_personalities (id, name, prompt, created_ts, updated_ts, creator_ip)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (pid, name[:200], prompt_text, now, now, creator_ip),
        )
    except sqlite3.IntegrityError:
        conn.close()
        return JSONResponse({"error": f"id {pid!r} already exists"}, status_code=409)
    conn.commit()
    row = conn.execute("SELECT * FROM proxy_personalities WHERE id=?", (pid,)).fetchone()
    conn.close()
    return dict(row)


@app.put("/__proxy/api/personalities/{pid}")
async def personalities_update(pid: str, request: Request):
    try:
        payload = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"invalid JSON: {e}"}, status_code=400)
    conn = db()
    row = conn.execute("SELECT creator_ip FROM proxy_personalities WHERE id=?", (pid,)).fetchone()
    if not row or not _personality_visible_to(_client_ip(request), row["creator_ip"]):
        conn.close()
        return JSONResponse({"error": "not found"}, status_code=404)
    fields, params = [], []
    if isinstance(payload.get("name"), str) and payload["name"].strip():
        fields.append("name=?")
        params.append(payload["name"].strip()[:200])
    if isinstance(payload.get("prompt"), str) and payload["prompt"].strip():
        fields.append("prompt=?")
        params.append(payload["prompt"].strip())
    if not fields:
        conn.close()
        return JSONResponse({"error": "nothing to update"}, status_code=400)
    fields.append("updated_ts=?")
    params.append(time.time())
    params.append(pid)
    conn.execute(f"UPDATE proxy_personalities SET {','.join(fields)} WHERE id=?", params)
    conn.commit()
    out = dict(conn.execute("SELECT * FROM proxy_personalities WHERE id=?", (pid,)).fetchone())
    conn.close()
    return out


@app.delete("/__proxy/api/personalities/{pid}")
async def personalities_delete(pid: str, request: Request):
    conn = db()
    row = conn.execute("SELECT creator_ip FROM proxy_personalities WHERE id=?", (pid,)).fetchone()
    if not row or not _personality_visible_to(_client_ip(request), row["creator_ip"]):
        conn.close()
        return JSONResponse({"error": "not found"}, status_code=404)
    conn.execute("DELETE FROM proxy_personalities WHERE id=?", (pid,))
    conn.commit()
    conn.close()
    return {"ok": True}


# -------- Stable Diffusion bridge (/imagine) --------

# Generated images are written at runtime, so they live in the writable state dir rather
# than the (possibly read-only, when pip/npm-installed) packaged static dir.
GENERATED_DIR = _user_state_dir() / "generated"
_GENERATED_NAME_RE = re.compile(r"^[a-f0-9]{8,64}\.(png|jpg|jpeg|webp)$")


async def _comfyui_pick_checkpoint(client: httpx.AsyncClient) -> str | None:
    """Discover the first available checkpoint by querying ComfyUI's /object_info."""
    try:
        r = await client.get(SD_URL + "/object_info", timeout=10.0)
        j = r.json()
        ckpts = j["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        return ckpts[0] if ckpts else None
    except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError, TypeError):
        return None


def _comfyui_workflow(prompt: str, neg: str, ckpt: str, steps: int, cfg: float,
                      w: int, h: int, seed: int, sampler: str) -> dict:
    """Minimal txt2img workflow: CheckpointLoader → CLIPTextEncode (pos+neg) → KSampler →
    VAEDecode → SaveImage. Node IDs match ComfyUI's "Default" workflow numbering."""
    return {
        "3": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler,
            "scheduler": "normal", "denoise": 1.0,
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["5", 0],
        }},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {
            "width": w, "height": h, "batch_size": 1,
        }},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {
            "filename_prefix": "ai_proxy", "images": ["8", 0],
        }},
    }


@app.post("/__proxy/api/sd/txt2img")
async def sd_txt2img(request: Request):
    """Submit a txt2img prompt to ComfyUI at SD_URL (default http://localhost:8188). Builds
    a minimal workflow, queues it via /prompt, polls /history/{prompt_id} until the run
    finishes, then downloads each output via /view and saves it under static/generated/."""
    try:
        payload = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"invalid JSON: {e}"}, status_code=400)
    prompt_text = (payload.get("prompt") or "").strip()
    if not prompt_text:
        return JSONResponse({"error": "'prompt' is required"}, status_code=400)
    neg = payload.get("negative_prompt") or ""
    steps = int(payload.get("steps") or 20)
    cfg = float(payload.get("cfg_scale") or payload.get("cfg") or 7.0)
    w = int(payload.get("width") or 512)
    h = int(payload.get("height") or 512)
    sampler = payload.get("sampler_name") or "euler"
    seed = int(payload.get("seed")) if payload.get("seed") is not None else int.from_bytes(os.urandom(4), "big")
    ckpt = (payload.get("model") or SD_MODEL or "").strip()
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            if not ckpt:
                ckpt = await _comfyui_pick_checkpoint(client)
                if not ckpt:
                    return JSONResponse(
                        {"error": "no checkpoints found in ComfyUI",
                         "hint": "Drop a model file into ComfyUI/models/checkpoints/, or set SD_MODEL env var."},
                        status_code=502,
                    )
            wf = _comfyui_workflow(prompt_text, neg, ckpt, steps, cfg, w, h, seed, sampler)
            client_id = uuid.uuid4().hex
            try:
                r = await client.post(
                    SD_URL + "/prompt",
                    json={"prompt": wf, "client_id": client_id},
                )
            except Exception as e:
                return JSONResponse(
                    {"error": f"ComfyUI unreachable at {SD_URL}: {e}",
                     "hint": "Start ComfyUI or set SD_URL env var to the right host:port."},
                    status_code=502,
                )
            if r.status_code != 200:
                return JSONResponse(
                    {"error": f"ComfyUI /prompt HTTP {r.status_code}: {r.text[:400]}"},
                    status_code=502,
                )
            try:
                pj = r.json()
            except (json.JSONDecodeError, ValueError):
                return JSONResponse({"error": "ComfyUI /prompt returned non-JSON"}, status_code=502)
            pid = pj.get("prompt_id")
            if not pid:
                return JSONResponse(
                    {"error": "ComfyUI didn't return prompt_id", "raw": pj},
                    status_code=502,
                )
            # Poll /history/{pid} until outputs land, capped at 5 minutes.
            deadline = time.time() + 300
            outputs = None
            while time.time() < deadline:
                await asyncio.sleep(1.0)
                try:
                    hr = await client.get(SD_URL + f"/history/{pid}", timeout=10.0)
                except Exception:
                    continue
                if hr.status_code != 200:
                    continue
                try:
                    hj = hr.json()
                except (json.JSONDecodeError, ValueError):
                    continue
                entry = hj.get(pid) or {}
                if entry.get("outputs"):
                    outputs = entry["outputs"]
                    break
            if not outputs:
                return JSONResponse({"error": "ComfyUI run timed out after 5 minutes"}, status_code=504)
            # Collect filenames from any node that produced images (typically SaveImage).
            image_descs = []
            for _, out in outputs.items():
                for img in (out.get("images") or []):
                    image_descs.append(img)
            if not image_descs:
                return JSONResponse(
                    {"error": "ComfyUI produced no images", "outputs": outputs},
                    status_code=502,
                )
            GENERATED_DIR.mkdir(parents=True, exist_ok=True)
            files = []
            for img in image_descs:
                params = {
                    "filename": img.get("filename") or "",
                    "type": img.get("type") or "output",
                    "subfolder": img.get("subfolder") or "",
                }
                if not params["filename"]:
                    continue
                try:
                    img_r = await client.get(SD_URL + "/view", params=params, timeout=30.0)
                except Exception:
                    continue
                if img_r.status_code != 200:
                    continue
                fname = uuid.uuid4().hex + ".png"
                (GENERATED_DIR / fname).write_bytes(img_r.content)
                files.append("/__proxy/generated/" + fname)
    except Exception as e:
        return JSONResponse({"error": f"unexpected: {e}"}, status_code=500)
    if not files:
        return JSONResponse({"error": "ComfyUI generated images but none could be downloaded"}, status_code=502)
    return {
        "ok": True,
        "files": files,
        "prompt": prompt_text,
        "ms": int((time.time() - t0) * 1000),
        "seed": seed,
        "sampler": sampler,
        "model": ckpt,
        "backend": "comfyui",
    }


@app.get("/__proxy/generated/{fname}")
async def generated_image(fname: str):
    """Serve a previously-generated image by uuid filename. Locked to a strict pattern so
    the request can't traverse outside the generated directory."""
    if not _GENERATED_NAME_RE.match(fname):
        return JSONResponse({"error": "invalid filename"}, status_code=400)
    p = GENERATED_DIR / fname
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p)


@app.get("/__proxy/remote")
async def remote_pwa():
    """Mobile-first PWA for sending prompts to registered control endpoints + panic toggle."""
    return FileResponse(STATIC_DIR / "remote.html")


@app.get("/__proxy")
@app.get("/__proxy/")
async def ui_index():
    return FileResponse(STATIC_DIR / "index.html")


# The proxy's logo: two arrows (request out / response back). Same SVG the UI uses for its
# inline favicon — served here so a browser's automatic /favicon.ico request resolves instead
# of falling through to the catch-all and getting proxied to Ollama (404).
_FAVICON_SVG = (
    b"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    b"<rect width='64' height='64' rx='12' fill='#1a1d26'/>"
    b"<path d='M12 22h32m-8-8 8 8-8 8' stroke='#88c0d0' stroke-width='5' fill='none' stroke-linecap='round' stroke-linejoin='round'/>"
    b"<path d='M52 42H20m8-8-8 8 8 8' stroke='#a3be8c' stroke-width='5' fill='none' stroke-linecap='round' stroke-linejoin='round'/>"
    b"</svg>"
)


@app.get("/favicon.ico")
async def favicon():
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})


# -------- Benchmark runner --------
#
# In-proxy benchmark feature: queues a configurable streaming workload (N requests at a
# given prompt size and concurrency), records TTFT + decode-rate + total per request, and
# stores rolled-up percentiles. Optional exclusive-mode pauses all non-bench traffic for
# clean uncontended numbers.
#
# Exclusive mode mechanics: a process-wide asyncio.Event that's set by default (allows
# traffic). When a bench with exclusive=True begins, the event clears, making every
# non-bench request `await` it. Bench requests bypass via x-client-name=ai-proxy-bench.
# Hard 5-minute safety cap so a hung bench can't lock the proxy forever.

_BENCH_TRAFFIC_OK = asyncio.Event()
_BENCH_TRAFFIC_OK.set()
_BENCH_SEM = asyncio.Semaphore(1)  # one bench at a time globally
_BENCH_EXCLUSIVE_DEADLINE: float = 0.0  # if > now, exclusive mode is active until this ts


# Realistic long-context filler (mirrors scripts/bench.py so the in-proxy bench produces
# comparable numbers to the standalone CLI version).
_BENCH_FILLER = '''
# Module: data_pipeline/transforms.py
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Callable, TypeVar, Generic, Optional
import logging
import time

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class TransformResult(Generic[T]):
    """Outcome of running a transform: the value plus diagnostics."""
    value: Optional[T] = None
    duration_ms: float = 0.0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class Pipeline(Generic[T, U]):
    """Composes a sequence of stages into a single callable. Each stage takes a T and
    returns a U; the pipeline propagates results through the stages, short-circuiting on
    None or on the first error encountered."""

    def __init__(self, stages: list):
        self.stages = stages
        self._call_count = 0

    def run(self, item):
        result = TransformResult(value=item)
        start = time.perf_counter()
        for i, stage in enumerate(self.stages):
            if result.value is None or result.error:
                break
            try:
                result.value = stage(result.value)
            except Exception as e:
                result.error = f"stage {i}: {e}"
                break
        result.duration_ms = (time.perf_counter() - start) * 1000
        self._call_count += 1
        return result


def chunked(items, size):
    """Group items into lists of `size`. Final batch may be smaller."""
    if size <= 0:
        raise ValueError("size must be positive")
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
'''

_BENCH_BASE_TASK = (
    "Write a clear, well-commented Python implementation of binary search over a "
    "sorted list of integers. Include type hints, a docstring, and a small test block "
    "at the bottom that exercises edge cases. Be thorough."
)


def _bench_build_prompt(prompt_tokens: int, randomize: bool, seq: int) -> str:
    """Build a prompt of approximately prompt_tokens. randomize=True salts each call so
    prompt-cache hits don't make later runs trivially fast (which would mask the real
    prefill cost we're trying to measure)."""
    if prompt_tokens <= 0:
        salt = f" [run #{seq}]" if randomize else ""
        return _BENCH_BASE_TASK + salt
    target_chars = int(prompt_tokens * 3.5)
    salt = f"// nonce: {uuid.uuid4().hex}\n" if randomize else ""
    head = f"Below is a code module. After the module, you'll be given a task.\n\n{salt}<CODE>\n"
    tail = f"\n</CODE>\n\nTask: {_BENCH_BASE_TASK}"
    body = ""
    while len(head) + len(body) + len(tail) < target_chars:
        body += _BENCH_FILLER
    return head + body + tail


async def _bench_run_one(client: httpx.AsyncClient, base: str, model: str,
                         max_tokens: int, prompt: str, run_seq: int) -> dict:
    """Issue one streaming chat-completion request via the proxy and collect timings."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
    }).encode("utf-8")
    headers = {
        "content-type": "application/json",
        "x-client-name": "ai-proxy-bench",
        "x-priority": "high",
    }
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    upstream_ct: int | None = None
    chunk_ct = 0
    err: str | None = None
    try:
        async with client.stream("POST", base + "/v1/chat/completions",
                                 headers=headers, content=body, timeout=httpx.Timeout(600.0)) as resp:
            if resp.status_code != 200:
                err = f"HTTP {resp.status_code}: {(await resp.aread()).decode('utf-8', errors='replace')[:300]}"
            else:
                async for line in resp.aiter_lines():
                    line = (line or "").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        j = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    for ch in (j.get("choices") or []):
                        delta = ch.get("delta") or {}
                        if delta.get("content"):
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - t0) * 1000
                            chunk_ct += len(re.findall(r"\S+", delta["content"]))
                    u = j.get("usage")
                    if isinstance(u, dict) and u.get("completion_tokens"):
                        upstream_ct = u["completion_tokens"]
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    total_ms = (time.perf_counter() - t0) * 1000
    completion_tokens = upstream_ct if upstream_ct is not None else chunk_ct
    decode_tps: float | None = None
    if (ttft_ms is not None and total_ms > ttft_ms and completion_tokens > 0):
        decode_tps = completion_tokens / ((total_ms - ttft_ms) / 1000.0)
    return {
        "seq": run_seq,
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "completion_tokens": completion_tokens,
        "decode_tps": decode_tps,
        "error": err,
    }


def _bench_pct(values: list, p: float):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    idx = (len(s) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _bench_summarize(rows: list[dict]) -> dict:
    """Roll up per-request timings into min / p50 / p90 / max for TTFT and decode."""
    successes = [r for r in rows if not r.get("error")]
    ttfts = [r["ttft_ms"] for r in successes if r["ttft_ms"] is not None]
    decodes = [r["decode_tps"] for r in successes if r["decode_tps"] is not None]
    totals = [r["total_ms"] for r in successes]
    return {
        "n_total": len(rows),
        "n_success": len(successes),
        "ttft_ms": {
            "min": min(ttfts) if ttfts else None,
            "p50": _bench_pct(ttfts, 50),
            "p90": _bench_pct(ttfts, 90),
            "max": max(ttfts) if ttfts else None,
        },
        "decode_tps": {
            "min": min(decodes) if decodes else None,
            "p50": _bench_pct(decodes, 50),
            "p90": _bench_pct(decodes, 90),
            "max": max(decodes) if decodes else None,
            "mean": (sum(decodes) / len(decodes)) if decodes else None,
        },
        "total_ms": {
            "min": min(totals) if totals else None,
            "p50": _bench_pct(totals, 50),
            "max": max(totals) if totals else None,
        },
    }


async def _bench_execute(bench_id: str, app: FastAPI):
    """Run a queued bench. Acquires the global bench semaphore so only one runs at a time.
    If config.exclusive=True, also pauses non-bench traffic via _BENCH_TRAFFIC_OK."""
    global _BENCH_EXCLUSIVE_DEADLINE
    async with _BENCH_SEM:
        conn = db()
        row = conn.execute("SELECT * FROM bench_runs WHERE id=?", (bench_id,)).fetchone()
        if not row or row["status"] != "pending":
            conn.close()
            return
        try:
            cfg = json.loads(row["config_json"])
        except (json.JSONDecodeError, TypeError):
            conn.execute("UPDATE bench_runs SET status='failed', error='invalid config JSON', finished_ts=? WHERE id=?",
                         (time.time(), bench_id))
            conn.commit()
            conn.close()
            return
        runs = max(1, min(int(cfg.get("runs", 5)), 50))
        max_tokens = max(16, min(int(cfg.get("max_tokens", 256)), 4096))
        prompt_tokens = max(0, min(int(cfg.get("prompt_tokens", 0)), 65536))
        concurrency = max(1, min(int(cfg.get("concurrency", 1)), 8))
        randomize = bool(cfg.get("randomize", False))
        exclusive = bool(cfg.get("exclusive", False))
        drain_seconds = max(0.0, min(float(cfg.get("drain_seconds", 5.0)), 30.0))
        model = row["model"]
        conn.execute(
            "UPDATE bench_runs SET status='running', started_ts=?, progress=0, progress_total=? WHERE id=?",
            (time.time(), runs, bench_id),
        )
        conn.commit()
        conn.close()
        # Exclusive mode: clear the gate, optionally drain in-flight requests, run, then re-set.
        gate_held = False
        if exclusive:
            _BENCH_TRAFFIC_OK.clear()
            _BENCH_EXCLUSIVE_DEADLINE = time.time() + 300  # 5-minute safety cap
            gate_held = True
            if drain_seconds > 0:
                await asyncio.sleep(drain_seconds)
        try:
            base = f"http://127.0.0.1:{PROXY_PORT}"
            client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
            try:
                rows: list[dict] = []
                # Run in batches of `concurrency`. e.g. concurrency=4, runs=8 → 2 waves of 4.
                seq_counter = 0
                while seq_counter < runs:
                    wave_size = min(concurrency, runs - seq_counter)
                    wave_seqs = list(range(seq_counter + 1, seq_counter + wave_size + 1))
                    coros = [
                        _bench_run_one(client, base, model, max_tokens,
                                       _bench_build_prompt(prompt_tokens, randomize, s), s)
                        for s in wave_seqs
                    ]
                    results = await asyncio.gather(*coros, return_exceptions=False)
                    rows.extend(results)
                    seq_counter += wave_size
                    # Persist progress incrementally so the UI can poll.
                    conn = db()
                    conn.execute(
                        "UPDATE bench_runs SET progress=?, results_json=? WHERE id=?",
                        (seq_counter, json.dumps({"rows": rows}), bench_id),
                    )
                    conn.commit()
                    conn.close()
                summary = _bench_summarize(rows)
                final = {"rows": rows, "summary": summary, "config_used": cfg}
                conn = db()
                conn.execute(
                    "UPDATE bench_runs SET status='done', results_json=?, finished_ts=? WHERE id=?",
                    (json.dumps(final), time.time(), bench_id),
                )
                conn.commit()
                conn.close()
            finally:
                await client.aclose()
        except Exception as e:
            conn = db()
            conn.execute(
                "UPDATE bench_runs SET status='failed', error=?, finished_ts=? WHERE id=?",
                (f"{type(e).__name__}: {e}", time.time(), bench_id),
            )
            conn.commit()
            conn.close()
        finally:
            if gate_held:
                _BENCH_TRAFFIC_OK.set()
                _BENCH_EXCLUSIVE_DEADLINE = 0.0


# -------- Task queue (one-shots + cron/interval recurring) --------
#
# Lets the user queue prompts (chat or agent mode) from the phone PWA and run them in the
# background. Recurring rows fire on a cron expression or "every Nm/Nh/Nd" interval and
# spawn a one-shot child each time. Agent tasks reuse the existing /api/control/chat +
# /await flow; chat tasks hit OLLAMA_URL directly with non-streaming completions.

# Concurrency: agent tasks are serialized (one VS Code session); chat tasks run up to 3
# in parallel. Semaphores live module-global so the worker tick can fire-and-forget.
_TASK_AGENT_SEM = asyncio.Semaphore(1)
_TASK_CHAT_SEM = asyncio.Semaphore(3)
# Tracks which task (if any) is currently driving an agent run, so /pending-tool/register
# can apply that task's tool_approval_mode when no persistent rule matches.
_CURRENT_AGENT_TASK: dict = {}


def _task_compute_next_run(schedule: str | None, now_ts: float) -> float | None:
    """Parse `schedule` and return the next-fire timestamp after `now_ts`. Supported:
      - 'every Nm' / 'every Nh' / 'every Nd' (case-insensitive, also accepts 's')
      - 5-field cron exprs ('m h dom mon dow') iff `croniter` is installed.
    Returns None on parse failure (worker will mark the recurring row as failed)."""
    if not schedule or not isinstance(schedule, str):
        return None
    s = schedule.strip().lower()
    m = re.match(r"^every\s+(\d+)\s*([smhd])$", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        secs = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit] * n
        if secs <= 0:
            return None
        return now_ts + secs
    try:
        from croniter import croniter  # optional dep
    except ImportError:
        return None
    try:
        return float(croniter(schedule, datetime.datetime.fromtimestamp(now_ts)).get_next(float))
    except (ValueError, KeyError):
        return None


async def _task_run_chat(task_id: int, prompt: str, model: str | None) -> tuple[str | None, str | None]:
    """Run a chat-mode task against OLLAMA_URL. Returns (result_text, error_text)."""
    if not model:
        # Pick the first currently-loaded Ollama model; fall back to first available tag.
        try:
            client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
            try:
                r = await client.get(OLLAMA_URL + "/api/ps")
                ps = (r.json() or {}).get("models") or []
                if ps:
                    model = ps[0].get("name") or ps[0].get("model")
                if not model:
                    r2 = await client.get(OLLAMA_URL + "/api/tags")
                    tags = (r2.json() or {}).get("models") or []
                    if tags:
                        model = tags[0].get("name")
            finally:
                await client.aclose()
        except Exception as e:
            return None, f"could not pick a model: {e}"
    if not model:
        return None, "no model specified and none discoverable from Ollama"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    try:
        client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
        try:
            r = await client.post(OLLAMA_URL + "/v1/chat/completions", json=payload)
        finally:
            await client.aclose()
    except Exception as e:
        return None, f"upstream request failed: {e}"
    if r.status_code >= 400:
        return None, f"upstream HTTP {r.status_code}: {r.text[:500]}"
    try:
        j = r.json()
        choices = j.get("choices") or []
        if choices:
            content = ((choices[0] or {}).get("message") or {}).get("content")
            if isinstance(content, str):
                return content, None
        return r.text, None
    except (json.JSONDecodeError, ValueError):
        return r.text, None


async def _task_run_agent(task_id: int, prompt: str, target: str | None,
                          tool_approval_mode: str | None) -> tuple[str | None, str | None]:
    """Run an agent-mode task by POSTing to /api/control/chat (an existing registered VS
    Code endpoint) and polling /await for the resulting LLM call. Honors tool_approval_mode
    via _CURRENT_AGENT_TASK so the @proxy participant respects rules-only / yolo / phone."""
    _CURRENT_AGENT_TASK.clear()
    _CURRENT_AGENT_TASK.update({"task_id": task_id, "mode": tool_approval_mode or "notify-phone"})
    try:
        conn = db()
        if target:
            row = conn.execute(
                "SELECT name, url, token FROM control_endpoints WHERE name = ?", (target,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT name, url, token FROM control_endpoints ORDER BY last_seen_ts DESC LIMIT 1"
            ).fetchone()
        conn.close()
        if not row:
            return None, "no control endpoints registered"
        fwd_url = row["url"].rstrip("/") + "/chat"
        fwd_headers = {"content-type": "application/json"}
        if row["token"]:
            fwd_headers["authorization"] = f"Bearer {row['token']}"
        send_id = uuid.uuid4().hex[:16]
        send_ts = time.time()
        client = httpx.AsyncClient(timeout=httpx.Timeout(None))
        try:
            try:
                r = await client.post(
                    fwd_url, headers=fwd_headers,
                    content=json.dumps({"prompt": prompt, "newChat": True}).encode("utf-8"),
                    timeout=httpx.Timeout(15.0),
                )
            except Exception as e:
                return None, f"could not reach control endpoint {row['name']}: {e}"
            if not r.is_success:
                return None, f"control endpoint returned HTTP {r.status_code}: {r.text[:300]}"
            _CONTROL_SENDS[send_id] = {
                "ts": send_ts,
                "target_name": row["name"],
                "target_ip": _control_target_ip(row["url"]),
                "prompt_hint": prompt[:120],
            }
        finally:
            await client.aclose()
        # Poll /await for up to 30 minutes (matches typical agent run length).
        deadline = time.time() + 1800
        last_text = ""
        while time.time() < deadline:
            await asyncio.sleep(2.0)
            target_ip = _CONTROL_SENDS.get(send_id, {}).get("target_ip")
            conn = db()
            rows = conn.execute(
                """SELECT id, status, response_body, stream_chunks, error
                   FROM requests_v
                   WHERE ts > ? AND client_ip = ?
                     AND client_app IN ('vscode-copilot','claude-code','github-copilot','continue.dev','cursor')
                     AND shadow_of IS NULL
                   ORDER BY ts ASC LIMIT 1""",
                (send_ts - 1, target_ip or ""),
            ).fetchall()
            conn.close()
            if not rows:
                continue
            r0 = rows[0]
            text = _extract_response_text(r0["response_body"], r0["stream_chunks"])
            if text:
                last_text = text
            if r0["error"]:
                return last_text or None, r0["error"]
            if r0["status"] is not None:
                return last_text or "(no text)", None
        return last_text or None, "timed out after 30 minutes waiting for agent response"
    finally:
        _CURRENT_AGENT_TASK.clear()


async def _task_execute(task_id: int):
    """Claim a task row, dispatch by mode, write result. Runs inside the appropriate sema."""
    conn = db()
    row = conn.execute("SELECT * FROM proxy_tasks WHERE id=?", (task_id,)).fetchone()
    if not row:
        conn.close()
        return
    if row["status"] != "pending":
        conn.close()
        return
    cur = conn.execute(
        "UPDATE proxy_tasks SET status='running', started_ts=? WHERE id=? AND status='pending'",
        (time.time(), task_id),
    )
    conn.commit()
    conn.close()
    if cur.rowcount != 1:
        return  # another tick claimed it first
    sem = _TASK_AGENT_SEM if row["mode"] == "agent" else _TASK_CHAT_SEM
    async with sem:
        # Re-check that we weren't cancelled while waiting on the semaphore.
        conn = db()
        cur_row = conn.execute("SELECT status FROM proxy_tasks WHERE id=?", (task_id,)).fetchone()
        conn.close()
        if not cur_row or cur_row["status"] != "running":
            return
        try:
            if row["mode"] == "agent":
                result, err = await _task_run_agent(
                    task_id, row["prompt"], row["target_endpoint"],
                    row["tool_approval_mode"],
                )
            else:
                result, err = await _task_run_chat(task_id, row["prompt"], row["model"])
        except Exception as e:
            result, err = None, f"worker exception: {e}"
        conn = db()
        conn.execute(
            """UPDATE proxy_tasks
               SET status=?, result=?, error=?, finished_ts=?
               WHERE id=? AND status='running'""",
            ("failed" if err else "done", result, err, time.time(), task_id),
        )
        conn.commit()
        conn.close()


async def _inflight_zombie_killer(app: FastAPI):
    """Periodically cancel in-flight requests that have been running longer than the
    configured max. Safety net for streamer coroutines that wedge on a dead client socket
    while upstream bytes pile up in kernel buffers — they don't naturally recover, so we
    kill them so the GPU slot frees up. Default cap: 30 minutes. Override via env var
    PROXY_INFLIGHT_MAX_S."""
    max_age = float(os.environ.get("PROXY_INFLIGHT_MAX_S", "1800") or "1800")
    while True:
        try:
            await asyncio.sleep(60)
            now = time.time()
            for req_id in list(_INFLIGHT_REQUESTS.keys()):
                info = _INFLIGHT_REQUESTS.get(req_id)
                if not info or info.get("cancelled"):
                    continue
                if (now - info.get("ts", now)) > max_age:
                    info["cancelled"] = True
                    upstream_resp = info.get("upstream_resp")
                    if upstream_resp is not None:
                        try:
                            await upstream_resp.aclose()
                        except Exception:
                            pass
                    try:
                        print(f"[zombie_killer] cancelled in-flight req {req_id} "
                              f"(elapsed {int(now - info.get('ts', now))}s)")
                    except Exception:
                        pass
                    # Remove it even if the streamer is wedged on a dead client socket (closing the
                    # upstream doesn't unblock a client-write wedge, so it would never pop itself →
                    # the entry would leak in the registry until the next restart).
                    _INFLIGHT_REQUESTS.pop(req_id, None)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            try:
                print(f"[zombie_killer] tick error: {e}")
            except Exception:
                pass


async def _task_worker_loop(app: FastAPI):
    """Tick every 5s. Spawn child rows for due recurring tasks; pick up pending one-shots
    and dispatch them. Each dispatch is an asyncio.create_task — the loop never blocks."""
    while True:
        try:
            await asyncio.sleep(5.0)
            now = time.time()
            conn = db()
            due = conn.execute(
                """SELECT * FROM proxy_tasks
                   WHERE schedule IS NOT NULL AND enabled=1 AND status='pending'
                     AND (next_run_ts IS NULL OR next_run_ts <= ?)""",
                (now,),
            ).fetchall()
            for parent in due:
                # Don't pile up: skip if a previous child of this parent is still pending/running.
                still = conn.execute(
                    """SELECT 1 FROM proxy_tasks
                       WHERE parent_task_id=? AND status IN ('pending','running') LIMIT 1""",
                    (parent["id"],),
                ).fetchone()
                if not still:
                    conn.execute(
                        """INSERT INTO proxy_tasks
                           (prompt, mode, target_endpoint, model, status, created_ts,
                            parent_task_id, tool_approval_mode, creator_ip)
                           VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?)""",
                        (parent["prompt"], parent["mode"], parent["target_endpoint"],
                         parent["model"], now, parent["id"], parent["tool_approval_mode"],
                         parent["creator_ip"]),
                    )
                next_ts = _task_compute_next_run(parent["schedule"], now)
                if next_ts is None:
                    conn.execute(
                        "UPDATE proxy_tasks SET enabled=0, error=? WHERE id=?",
                        (f"unparseable schedule: {parent['schedule']!r}", parent["id"]),
                    )
                else:
                    conn.execute(
                        "UPDATE proxy_tasks SET next_run_ts=? WHERE id=?",
                        (next_ts, parent["id"]),
                    )
            conn.commit()
            # Pick up pending one-shots (NOT recurring parent rows — those have schedule).
            pending = conn.execute(
                """SELECT id FROM proxy_tasks
                   WHERE status='pending' AND schedule IS NULL
                   ORDER BY created_ts ASC LIMIT 25"""
            ).fetchall()
            conn.close()
            for r in pending:
                asyncio.create_task(_task_execute(r["id"]))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            try:
                print(f"[task_worker] tick error: {e}")
            except Exception:
                pass


# -------- Transparent proxy --------

REQ_HOP_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}


_MODEL_FIELD_RE = re.compile(r'"model"\s*:\s*"[^"]*"')


def _override_model_in_text(text: str, new_model: str) -> str:
    """Replace every `"model": "..."` value in a JSON or SSE blob with `new_model`. Used
    only when model_router.preserve_response_model_name is enabled — clients then see the
    model name they originally requested, hiding the proxy's rewrite. Off by default."""
    if not text or not new_model:
        return text
    safe = json.dumps(new_model)  # handles quoting/escaping
    return _MODEL_FIELD_RE.sub(f'"model": {safe}', text)
RESP_HOP_HEADERS = {"content-length", "content-encoding", "transfer-encoding", "connection"}


def _filter(headers, drop):
    return [(k, v) for k, v in headers.items() if k.lower() not in drop]


def _msg_text(m) -> str:
    if not isinstance(m, dict):
        return ""
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        # Handles OpenAI multimodal blocks ({type:"text",text}) and Anthropic content blocks
        # ({type:"text",text}, {type:"tool_use",input}, {type:"tool_result",content}).
        parts = []
        for p in c:
            if not isinstance(p, dict):
                continue
            if isinstance(p.get("text"), str):
                parts.append(p["text"])
                continue
            t = p.get("type")
            if t == "tool_use" and p.get("input") is not None:
                try:
                    parts.append(json.dumps(p["input"]))
                except (TypeError, ValueError):
                    parts.append(str(p["input"]))
            elif t == "tool_result":
                inner = p.get("content")
                if isinstance(inner, str):
                    parts.append(inner)
                elif isinstance(inner, list):
                    for ip in inner:
                        if isinstance(ip, dict) and isinstance(ip.get("text"), str):
                            parts.append(ip["text"])
        return "\n".join(parts)
    if c is None:
        return ""
    try:
        return json.dumps(c, sort_keys=True)
    except (TypeError, ValueError):
        return str(c)


_CID_VOLATILE_PATTERNS = [
    # Claude Code injects a per-request billing header line into the system prompt with a
    # rotating `cch=<hex>;` value that would otherwise break conversation grouping.
    re.compile(r"^x-anthropic-billing-header:.*$", re.MULTILINE | re.IGNORECASE),
    # Generic: any standalone "cch=<hex>;" fragment that survives the line strip above.
    re.compile(r"\bcch=[0-9a-f]+;?", re.IGNORECASE),
]


def _normalize_for_cid(text: str) -> str:
    """Strip per-request volatile content (timestamps, billing hashes, session ids) before
    hashing so the conversation_id remains stable across turns of the same conversation."""
    if not text:
        return ""
    out = text
    for pat in _CID_VOLATILE_PATTERNS:
        out = pat.sub("", out)
    return out.strip()


def _hash_input_for_cid(text: str) -> str:
    """Reduce a string to its stable identity for conversation grouping. Strips per-request
    volatile patterns AND all XML-like wrappers."""
    if not text:
        return ""
    cleaned, _, _ = _clean_user_prompt(text)
    return _normalize_for_cid(cleaned or text)


def _first_typed_user_prompt(body) -> str:
    """Scan ALL user messages for the FIRST user-input marker (<userRequest>, <userPrompt>,
    etc.) and return its inner text. Copilot Chat puts <environment_info> in the very first
    user message and the actual typed prompt only in later messages — so scanning by index
    instead of relying on message[0] gives us a stable conversation identity that survives
    the rotating env wrappers."""
    if not isinstance(body, dict):
        return ""
    for m in (body.get("messages") or []):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        text = _msg_text(m)
        if not text:
            continue
        for tag in _USER_PROMPT_TAGS:
            mm = re.search(
                rf"<\s*{tag}\b[^>]*>(.*?)</\s*{tag}\s*>",
                text, re.DOTALL | re.IGNORECASE,
            )
            if mm:
                inner = mm.group(1).strip()
                if inner:
                    return inner
    return ""


def _conversation_id(body) -> str | None:
    """Stable conversation hash. Three-layer strategy for finding session identity:
      1. If any user message has an explicit <userRequest>/<userPrompt>/etc. marker, the
         FIRST such block's content is the session identity (Copilot Chat pattern — the
         first user message is just env_info; actual typed prompt lives in later turns).
      2. Otherwise, hash system + cleaned first-user-message (Claude Code, Anthropic SDKs).
      3. Volatile content (XML wrappers, billing headers, env IDs) is normalized out at
         every layer so turns of the same chat share an id even when env rotates."""
    if not isinstance(body, dict):
        return None
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    # Layer 1: explicit user-input marker anywhere in the messages list.
    typed = _first_typed_user_prompt(body)
    if typed:
        sys_msg = next((m for m in messages if isinstance(m, dict) and m.get("role") == "system"), None)
        sys_text = _hash_input_for_cid(_msg_text(sys_msg)) if sys_msg else ""
        if not sys_text:
            top_sys = body.get("system")
            if isinstance(top_sys, str):
                sys_text = _hash_input_for_cid(top_sys)
            elif isinstance(top_sys, list):
                joined = "\n".join(p.get("text", "") for p in top_sys if isinstance(p, dict))
                sys_text = _hash_input_for_cid(joined)
        # Use just the first 400 chars of cleaned system + typed prompt as identity.
        # Short system prefix dampens collisions for short prompts ("test") across different
        # client setups, while staying stable across turns.
        ident = (sys_text[:400] + "\n---\n" + typed).encode("utf-8")
        return hashlib.sha256(ident).hexdigest()[:16]

    # Layer 2: legacy path — system + cleaned first-user.
    sys_msg = next((m for m in messages if isinstance(m, dict) and m.get("role") == "system"), None)
    first_user = next((m for m in messages if isinstance(m, dict) and m.get("role") == "user"), None)
    parts: list[str] = []
    if sys_msg:
        parts.append("system:" + _hash_input_for_cid(_msg_text(sys_msg)))
    else:
        top_sys = body.get("system")
        sys_text = ""
        if isinstance(top_sys, str):
            sys_text = top_sys
        elif isinstance(top_sys, list):
            sys_text = "\n".join(p.get("text", "") for p in top_sys if isinstance(p, dict))
        if sys_text:
            parts.append("system:" + _hash_input_for_cid(sys_text))
    if first_user:
        parts.append("user:" + _hash_input_for_cid(_msg_text(first_user)))
    if parts:
        return hashlib.sha256("\n---\n".join(parts).encode("utf-8")).hexdigest()[:16]
    return None


def _turn_index(body) -> int | None:
    if not isinstance(body, dict):
        return None
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return None
    return sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "user")


def _detect_client_app(headers: dict | None, body: dict | None) -> str:
    """Heuristic identification of the client SDK / app behind a request. Walks headers
    (User-Agent, x-stainless-*, Editor-Version, etc.) and falls back to fingerprints in
    the system prompt (Claude Code embeds a billing header). Returns a short label."""
    h = {k.lower(): str(v) for k, v in (headers or {}).items()} if isinstance(headers, dict) else {}
    ua = (h.get("user-agent") or "").lower()

    # Claude Code is the only thing that injects this billing header line into the prompt.
    sys_text = ""
    if isinstance(body, dict):
        sys_field = body.get("system")
        if isinstance(sys_field, str):
            sys_text = sys_field
        elif isinstance(sys_field, list):
            sys_text = " ".join(p.get("text", "") for p in sys_field if isinstance(p, dict) and isinstance(p.get("text"), str))
        if not sys_text:
            for m in (body.get("messages") or []):
                if isinstance(m, dict) and m.get("role") == "system":
                    c = m.get("content")
                    if isinstance(c, str):
                        sys_text = c
                    elif isinstance(c, list):
                        sys_text = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and isinstance(p.get("text"), str))
                    break
    if "cc_entrypoint=cli" in sys_text or "claude-code" in ua or "claudecode" in ua:
        return "claude-code"

    # Caller-supplied identification: if a client sets x-client-name (or x-app-id /
    # x-application-name), trust it. This lets arbitrary scripts label themselves
    # without us having to maintain a UA whitelist.
    for hk in ("x-client-name", "x-client-app", "x-app-id", "x-application-name"):
        v = (h.get(hk) or "").strip()
        if v:
            slug = re.sub(r"[^a-z0-9._-]+", "-", v.lower())[:40].strip("-")
            if slug:
                return slug

    # Editor / IDE integrations
    ev = (h.get("editor-version") or "").lower()
    epv = (h.get("editor-plugin-version") or "").lower()
    on_behalf = (h.get("x-onbehalf-extension-id") or "").lower()
    vscode_marker = bool(h.get("x-vscode-user-agent-library-version") or h.get("x-vscode-machineid"))
    if h.get("copilot-integration-id") or "github.copilot" in on_behalf or "githubcopilotchat" in ua:
        if "vscode" in ev or vscode_marker or "github.copilot-chat" in on_behalf:
            return "vscode-copilot"
        if "jetbrains" in ev or "intellij" in ev:
            return "jetbrains-copilot"
        if "neovim" in ev or "nvim" in ev:
            return "neovim-copilot"
        return "github-copilot"
    if vscode_marker:
        return "vscode"
    if "vscode" in ev or "visual studio code" in ua:
        return "vscode"
    if "cursor" in ua or "cursor" in ev:
        return "cursor"
    if "continue" in ua or "continue.dev" in ua:
        return "continue.dev"
    if "cline" in ua:
        return "cline"
    if "aider" in ua:
        return "aider"
    if "zed" in ua:
        return "zed"

    # Anthropic / OpenAI official SDKs (Stainless-generated)
    if h.get("anthropic-version") or "anthropic" in (h.get("x-stainless-package-version") or "").lower():
        if "python" in (h.get("x-stainless-lang") or "").lower(): return "anthropic-python"
        if "js" in (h.get("x-stainless-lang") or "").lower() or "node" in (h.get("x-stainless-lang") or "").lower(): return "anthropic-node"
        return "anthropic-sdk"
    if h.get("openai-organization") or "openai" in (h.get("x-stainless-package-version") or "").lower():
        if "python" in (h.get("x-stainless-lang") or "").lower(): return "openai-python"
        if "js" in (h.get("x-stainless-lang") or "").lower() or "node" in (h.get("x-stainless-lang") or "").lower(): return "openai-node"
        return "openai-sdk"

    # Free-form User-Agent fallbacks
    for pat, label in (
        ("openai-python", "openai-python"), ("openai/", "openai-sdk"),
        ("anthropic-python", "anthropic-python"), ("anthropic/", "anthropic-sdk"),
        ("langchain", "langchain"), ("llamaindex", "llamaindex"),
        ("ollama", "ollama-cli"), ("lm-studio", "lm-studio"), ("lmstudio", "lm-studio"),
        ("axios/", "axios"), ("python-httpx", "httpx"), ("httpx/", "httpx"),
        ("python-requests", "requests"), ("node-fetch", "node-fetch"),
        ("aiohttp", "aiohttp"), ("curl/", "curl"), ("wget/", "wget"),
        ("postman", "postman"), ("insomnia", "insomnia"), ("httpie", "httpie"),
    ):
        if pat in ua:
            return label

    # Browsers (typically hitting the UI directly, not the LLM API). Order matters:
    # check Edge before Chrome, Chrome before Safari (UA strings overlap).
    if "firefox/" in ua and "mozilla" in ua:
        return "browser-firefox"
    if "edg/" in ua:
        return "browser-edge"
    if "chrome/" in ua and "safari/" in ua:
        return "browser-chrome"
    if "safari/" in ua and "version/" in ua:
        return "browser-safari"
    if ua.startswith("mozilla/"):
        return "browser-other"

    # Final fallback: extract a "name" from a "Name/Version" or "Name (...)" UA so
    # arbitrary scripts ("MyApp/1.0", "MyTool/2.3 (+url)") get a label
    # instead of being lumped into 'unknown'. Skip if too short or too generic.
    raw_ua = (h.get("user-agent") or "").strip()
    if raw_ua:
        m = re.match(r"^([A-Za-z][\w.\-]{2,40})", raw_ua)
        if m:
            slug = re.sub(r"[^a-z0-9._-]+", "-", m.group(1).lower()).strip("-")
            generic = {"python", "client", "http", "app", "test", "mozilla", "user-agent"}
            if slug and slug not in generic and len(slug) >= 3:
                return slug

    return "unknown"


def _client_ip(request: Request) -> str | None:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    xri = request.headers.get("x-real-ip")
    if xri:
        return xri.strip()
    return request.client.host if request.client else None


def _ips_share_subnet(a: str | None, b: str | None) -> bool:
    """True iff `a` and `b` parse as IPs and live in the same subnet (default /24 for v4, /64 for v6).
    Loopback either side counts as a match (admin local access)."""
    if not a or not b:
        return False
    try:
        ip_a = ipaddress.ip_address(a)
        ip_b = ipaddress.ip_address(b)
    except (ValueError, TypeError):
        return False
    if ip_a.version != ip_b.version:
        return False
    if ip_a.is_loopback or ip_b.is_loopback:
        return True
    bits = REDACT_SUBNET_BITS_V4 if ip_a.version == 4 else 64
    try:
        net = ipaddress.ip_network(f"{ip_a}/{bits}", strict=False)
        return ip_b in net
    except (ValueError, TypeError):
        return False


def _can_view_pii(viewer_ip: str | None, originator_ip: str | None) -> bool:
    """When redaction is on, PII visible iff viewer and originator are the same IP or share a subnet.
    When redaction is off, everything is visible."""
    if not REDACT_PII_ENABLED:
        return True
    if viewer_ip and viewer_ip in ADMIN_IPS:
        return True
    if viewer_ip and originator_ip and viewer_ip == originator_ip:
        return True
    return _ips_share_subnet(viewer_ip, originator_ip)


# Field sets for redaction. Splitting into "body" (full-content fields) and "summary" (short
# fields like gate_reason that may quote prompt content) so per-endpoint we redact appropriately.
_PII_BODY_FIELDS = (
    "request_body", "response_body", "stream_chunks",
    "request_headers", "response_headers",
)
_PII_SUMMARY_FIELDS = ("gate_reason", "gate_details", "preview", "downsize_reasons")


def _redact_row(row: dict, viewer_ip: str | None, *, originator_ip_field: str = "client_ip",
                originator_ips: list[str | None] | None = None) -> dict:
    """Mutate `row` to scrub PII fields if the viewer can't view this row's originator.
    For aggregate rows (e.g. conversations grouping multiple clients) pass `originator_ips` —
    if any of them are viewable we keep the data."""
    if not REDACT_PII_ENABLED:
        return row
    visible = False
    if originator_ips is not None:
        visible = any(_can_view_pii(viewer_ip, ip) for ip in originator_ips)
    else:
        visible = _can_view_pii(viewer_ip, row.get(originator_ip_field))
    if visible:
        return row
    for key in _PII_BODY_FIELDS:
        if key in row and row[key] is not None:
            row[key] = REDACT_PLACEHOLDER
    for key in _PII_SUMMARY_FIELDS:
        if key in row and row[key] is not None:
            row[key] = REDACT_PLACEHOLDER
    row["_pii_redacted"] = True
    return row


def _save_pending(req_id: str, request: Request, full_path: str, upstream_url: str, body_text: str, body_json, model, is_stream: bool, upstream: str | None = None):
    conv_id = _conversation_id(body_json)
    turn = _turn_index(body_json)
    headers_dict = dict(request.headers)
    client_app = _detect_client_app(headers_dict, body_json if isinstance(body_json, dict) else None)
    # Chars-based prompt-token estimate, persisted so the requests list can derive a cache
    # verdict (evaluated vs estimated) without re-parsing the body on every load.
    est = _estimate_prompt_tokens(body_json, 3.5) if isinstance(body_json, dict) else 0
    # has_images flags standard vision parts (image_url / Anthropic image) AND base64 blobs
    # embedded in tool-output text (e.g. screenshot_png_b64). The router keys off _body_has_images()
    # directly (standard parts only), so flagging an embedded blob here surfaces it in the dashboard
    # viewer WITHOUT routing a text request to the vision model (which couldn't consume it anyway).
    _embedded_imgs = isinstance(body_text, str) and ("iVBORw0KGgo" in body_text or "/9j/" in body_text)
    has_imgs = 1 if (_body_has_images(body_json) or _embedded_imgs) else None
    # Pull inline base64 images into their own column at full size, then store the text body
    # with them stripped — so the 256KB body cap can't truncate (and corrupt the JSON of) a
    # ~700KB screenshot. The image bytes stay reconstructable from images_data.
    images_data = None
    store_body_text = body_text
    if isinstance(body_json, dict) and has_imgs:
        imgs = [{"media_type": mt, "data": payload}
                for (_i, mt, _k, payload) in _iter_request_images(body_json) if _k == "data"]
        if imgs:
            images_data = json.dumps(imgs)
            body_copy = json.loads(json.dumps(body_json))   # don't mutate the body we forward
            _strip_image_data(body_copy)
            store_body_text = json.dumps(body_copy)
    conn = db()
    conn.execute(
        """INSERT INTO requests (id, ts, method, path, upstream_url, request_headers, model, is_stream, client_ip, conversation_id, turn_index, client_app, upstream, est_prompt_tokens, has_images)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            req_id,
            time.time(),
            request.method,
            "/" + full_path,
            upstream_url,
            json.dumps(headers_dict),
            model,
            int(is_stream),
            _client_ip(request),
            conv_id,
            turn,
            client_app,
            upstream,
            est or None,
            has_imgs,
        ),
    )
    _blobs_upsert(conn, req_id,
                  request_body=_truncate_for_store(store_body_text),
                  images_data=images_data)
    conn.commit()
    conn.close()
    # Seed live state with the estimate so the UI shows something immediately, even before
    # the upstream confirms the real input_tokens.
    _LIVE_STREAMS[req_id] = {"prompt": None, "completion": None, "est_prompt": est or None}


_TOOL_ERROR_PATTERNS = (
    "must have required property",
    "validation error",
    "is not allowed",
    "is required",
    "is invalid",
    "could not be",
    "failed to",
    "traceback (most recent call",
)


def _is_tool_error(text) -> tuple[bool, str | None]:
    """Detect whether a tool-result content string looks like an error. Returns (is_error, excerpt)."""
    if not text:
        return False, None
    s = str(text).strip()
    if not s:
        return False, None
    # JSON shape: {error: ...} or {success: false} or {isError: true}
    try:
        j = json.loads(s)
        if isinstance(j, dict):
            if j.get("error") or j.get("success") is False or j.get("isError"):
                msg = j.get("error") if isinstance(j.get("error"), str) else (j.get("message") or json.dumps(j.get("error") or j)[:200])
                return True, str(msg)[:240]
    except (json.JSONDecodeError, TypeError):
        pass
    first_line = s.split("\n", 1)[0]
    fl_lower = first_line.lower()
    if fl_lower.startswith("error") or fl_lower.startswith("exception"):
        return True, first_line[:240]
    s_lower = s.lower()
    for pat in _TOOL_ERROR_PATTERNS:
        if pat in s_lower:
            return True, first_line[:240]
    return False, None


def _tool_results_in_body(body) -> list[tuple[str, str]]:
    """Returns list of (tool_name, content) pairs from tool-result messages, in order.
    OpenAI/Ollama: messages[i].role=='tool' with tool_call_id linking back to assistant tool_calls.
    Anthropic: messages[i].role=='user' with content[] containing tool_result blocks linked via
    tool_use_id to assistant content[] tool_use blocks."""
    if not isinstance(body, dict):
        return []
    msgs = body.get("messages") or []
    id_to_name: dict[str, str] = {}
    for m in msgs:
        if not isinstance(m, dict):
            continue
        # OpenAI/Ollama: assistant.tool_calls list
        for tc in (m.get("tool_calls") or []):
            tid = tc.get("id")
            fn = (tc.get("function") or {}).get("name")
            if tid and fn:
                id_to_name[tid] = fn
        # Anthropic: assistant.content[] with tool_use blocks
        if m.get("role") == "assistant" and isinstance(m.get("content"), list):
            for blk in m["content"]:
                if isinstance(blk, dict) and blk.get("type") == "tool_use":
                    tid = blk.get("id")
                    name = blk.get("name")
                    if tid and name:
                        id_to_name[tid] = name
    pairs: list[tuple[str, str]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        # OpenAI/Ollama: role==tool
        if m.get("role") == "tool":
            tid = m.get("tool_call_id") or ""
            name = id_to_name.get(tid) or m.get("name") or "?"
            content = m.get("content")
            if isinstance(content, list):
                content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
            elif content is None:
                content = ""
            elif not isinstance(content, str):
                try:
                    content = json.dumps(content)
                except (TypeError, ValueError):
                    content = str(content)
            pairs.append((name, content))
            continue
        # Anthropic: tool_result blocks live inside user messages
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for blk in m["content"]:
                if not isinstance(blk, dict) or blk.get("type") != "tool_result":
                    continue
                tid = blk.get("tool_use_id") or ""
                name = id_to_name.get(tid) or "?"
                inner = blk.get("content")
                if isinstance(inner, list):
                    text = "\n".join(p.get("text", "") for p in inner if isinstance(p, dict))
                elif isinstance(inner, str):
                    text = inner
                elif inner is None:
                    text = ""
                else:
                    try:
                        text = json.dumps(inner)
                    except (TypeError, ValueError):
                        text = str(inner)
                # Anthropic explicitly marks errors via is_error=true; surface those even if the
                # content text doesn't otherwise look like an error.
                if blk.get("is_error") and text and not text.lower().startswith("error"):
                    text = "Error: " + text
                pairs.append((name, text))
    return pairs


def _extract_tool_calls(response_body, stream_text):
    """Return list of tool function names actually invoked in a response.
    Handles OpenAI (choices[*].message.tool_calls), Ollama native (message.tool_calls),
    and Anthropic (content[*].type=='tool_use' on non-streaming, content_block_start
    events on streaming)."""
    names: list[str] = []

    def from_choices(choices):
        for c in choices or []:
            for src in (c.get("message"), c.get("delta")):
                if not isinstance(src, dict):
                    continue
                for tc in (src.get("tool_calls") or []):
                    fn = (tc.get("function") or {}).get("name")
                    if fn:
                        names.append(fn)

    def from_anthropic_content(content):
        for blk in content or []:
            if isinstance(blk, dict) and blk.get("type") == "tool_use" and isinstance(blk.get("name"), str):
                names.append(blk["name"])

    if response_body:
        try:
            j = json.loads(response_body)
            if isinstance(j, dict):
                from_choices(j.get("choices") or [])
                # Ollama native /api/chat
                msg = j.get("message")
                if isinstance(msg, dict):
                    for tc in (msg.get("tool_calls") or []):
                        fn = (tc.get("function") or {}).get("name")
                        if fn:
                            names.append(fn)
                # Anthropic /v1/messages
                if j.get("type") == "message" and isinstance(j.get("content"), list):
                    from_anthropic_content(j["content"])
        except json.JSONDecodeError:
            pass

    if stream_text:
        # Streaming tool calls arrive as deltas keyed by index; the name appears once per index.
        per_idx: dict[int, str] = {}
        for line in stream_text.split("\n"):
            data = None
            if line.startswith("data: "):
                data = line[6:]
            elif line.strip().startswith("{"):
                data = line.strip()
            if not data or data == "[DONE]":
                continue
            try:
                j = json.loads(data)
            except json.JSONDecodeError:
                continue
            for c in j.get("choices") or []:
                delta = c.get("delta") or c.get("message") or {}
                for tc in (delta.get("tool_calls") or []):
                    idx = tc.get("index", len(per_idx))
                    fn = (tc.get("function") or {}).get("name")
                    if fn:
                        per_idx[idx] = fn
            # Ollama native streaming has message.tool_calls in the final chunk
            msg = j.get("message")
            if isinstance(msg, dict):
                for tc in (msg.get("tool_calls") or []):
                    fn = (tc.get("function") or {}).get("name")
                    if fn:
                        names.append(fn)
            # Anthropic SSE: tool_use names arrive in content_block_start events
            if j.get("type") == "content_block_start":
                blk = j.get("content_block") or {}
                if blk.get("type") == "tool_use" and isinstance(blk.get("name"), str):
                    names.append(blk["name"])
        names.extend(per_idx.values())

    return names


# -------- Protocol bridge: Anthropic ↔ OpenAI translation --------
# Lets Anthropic-shape clients (e.g. Claude Code) drive OpenAI-compatible backends
# (Ollama, LM Studio, vLLM, etc.) transparently, and vice versa.

_FINISH_OPENAI_TO_ANTHROPIC = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "stop_sequence",
    "function_call": "tool_use",
}


def _is_claude_model(name) -> bool:
    """Heuristic: does this model name belong to a Claude (Anthropic) model?"""
    if not isinstance(name, str):
        return False
    n = name.lower()
    return n.startswith("claude-") or n.startswith("claude/") or n.startswith("anthropic/")


def _strip_volatile_prefix(body: dict) -> int:
    """Walks a request body and removes per-request volatile content (Claude Code's
    `x-anthropic-billing-header: ...; cch=<hex>;` line, etc.) from message contents and
    the top-level `system` field. Stable prefixes let llama.cpp's KV cache reuse work
    across turns of the same conversation. Returns the number of fields touched."""
    if not isinstance(body, dict):
        return 0
    touched = 0

    def scrub_str(s):
        nonlocal touched
        if not isinstance(s, str):
            return s
        out = s
        for pat in _CID_VOLATILE_PATTERNS:
            new = pat.sub("", out)
            if new != out:
                out = new
                touched += 1
        return out

    # Top-level system (Anthropic shape).
    sys_field = body.get("system")
    if isinstance(sys_field, str):
        body["system"] = scrub_str(sys_field)
    elif isinstance(sys_field, list):
        for p in sys_field:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                p["text"] = scrub_str(p["text"])

    for m in (body.get("messages") or []):
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str):
            m["content"] = scrub_str(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    p["text"] = scrub_str(p["text"])
    return touched


def _anthropic_to_openai_request(body: dict) -> dict:
    """Translate an Anthropic /v1/messages request body to OpenAI /v1/chat/completions shape.
    Returns a new dict; does not mutate the input. The model field is preserved (typically
    set by model_router before this runs)."""
    out: dict = {}
    if body.get("model") is not None:
        out["model"] = body["model"]
    if body.get("max_tokens") is not None:
        out["max_tokens"] = body["max_tokens"]
    for k in ("temperature", "top_p", "top_k", "stream", "user"):
        if body.get(k) is not None:
            out[k] = body[k]
    if body.get("stop_sequences") is not None:
        out["stop"] = body["stop_sequences"]
    # Ask upstream to include usage in stream chunks so back-translation has accurate counts.
    if out.get("stream"):
        out["stream_options"] = {"include_usage": True}

    msgs: list[dict] = []

    # System: top-level field becomes a leading system message.
    sys_field = body.get("system")
    sys_text = ""
    if isinstance(sys_field, str):
        sys_text = sys_field
    elif isinstance(sys_field, list):
        sys_text = "\n".join(p.get("text", "") for p in sys_field
                              if isinstance(p, dict) and isinstance(p.get("text"), str))
    if sys_text:
        msgs.append({"role": "system", "content": sys_text})

    for m in (body.get("messages") or []):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")

        if role == "user":
            if isinstance(content, str):
                msgs.append({"role": "user", "content": content})
                continue
            if not isinstance(content, list):
                continue
            text_parts: list[str] = []
            tool_results: list[dict] = []
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                t = blk.get("type")
                if t == "text" and isinstance(blk.get("text"), str):
                    text_parts.append(blk["text"])
                elif t == "tool_result":
                    inner = blk.get("content")
                    if isinstance(inner, list):
                        inner_text = "\n".join(p.get("text", "") for p in inner
                                                if isinstance(p, dict) and isinstance(p.get("text"), str))
                    elif isinstance(inner, str):
                        inner_text = inner
                    elif inner is None:
                        inner_text = ""
                    else:
                        try:
                            inner_text = json.dumps(inner)
                        except (TypeError, ValueError):
                            inner_text = str(inner)
                    if blk.get("is_error") and inner_text and not inner_text.lower().startswith("error"):
                        inner_text = "Error: " + inner_text
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": blk.get("tool_use_id") or "",
                        "content": inner_text,
                    })
                # Other block types (image, document) are dropped — most OpenAI backends
                # for Ollama don't support them anyway.
            # Tool results must precede the next user-text message in OpenAI ordering;
            # they belong to the prior assistant turn.
            msgs.extend(tool_results)
            if text_parts:
                msgs.append({"role": "user", "content": "\n".join(text_parts)})
            elif not tool_results:
                # Empty user message — unusual but preserve to keep turn count.
                msgs.append({"role": "user", "content": ""})

        elif role == "assistant":
            if isinstance(content, str):
                msgs.append({"role": "assistant", "content": content})
                continue
            if not isinstance(content, list):
                continue
            text_parts = []
            tool_calls: list[dict] = []
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                t = blk.get("type")
                if t == "text" and isinstance(blk.get("text"), str):
                    text_parts.append(blk["text"])
                elif t == "tool_use":
                    try:
                        args_str = json.dumps(blk.get("input") or {})
                    except (TypeError, ValueError):
                        args_str = "{}"
                    tool_calls.append({
                        "id": blk.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": blk.get("name") or "",
                            "arguments": args_str,
                        },
                    })
            asst_msg: dict = {"role": "assistant"}
            asst_msg["content"] = "\n".join(text_parts) if text_parts else None
            if tool_calls:
                asst_msg["tool_calls"] = tool_calls
            msgs.append(asst_msg)

        elif role == "system":
            text = _msg_text(m)
            if text:
                msgs.append({"role": "system", "content": text})

    out["messages"] = msgs

    # Tools: {name, description, input_schema} → {type:"function", function:{name, description, parameters}}
    if isinstance(body.get("tools"), list):
        out_tools = []
        for t in body["tools"]:
            if not isinstance(t, dict):
                continue
            out_tools.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {"type": "object"},
                },
            })
        if out_tools:
            out["tools"] = out_tools

    tc = body.get("tool_choice")
    if isinstance(tc, dict):
        ttype = tc.get("type")
        if ttype == "auto":
            out["tool_choice"] = "auto"
        elif ttype == "any":
            out["tool_choice"] = "required"
        elif ttype == "tool" and tc.get("name"):
            out["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
        elif ttype == "none":
            out["tool_choice"] = "none"

    # Strip per-request volatile prefix content so llama.cpp's prefix KV cache can reuse
    # across turns. Without this, Claude Code's billing header changes every request and
    # invalidates the cache for everything after it.
    _strip_volatile_prefix(out)
    return out


def _openai_to_anthropic_response(o: dict, fallback_model: str | None = None) -> dict:
    """Translate a non-streaming OpenAI chat.completion response to Anthropic /v1/messages shape."""
    if not isinstance(o, dict):
        return {}
    choice = (o.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content_blocks: list[dict] = []
    text = msg.get("content")
    if isinstance(text, str) and text:
        content_blocks.append({"type": "text", "text": text})
    elif isinstance(text, list):
        for p in text:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                content_blocks.append({"type": "text", "text": p["text"]})
    for tc in (msg.get("tool_calls") or []):
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        try:
            inp = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            inp = {"_raw_arguments": fn.get("arguments")}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:12]}",
            "name": fn.get("name") or "",
            "input": inp,
        })

    finish = choice.get("finish_reason")
    stop_reason = _FINISH_OPENAI_TO_ANTHROPIC.get(finish, finish or "end_turn")

    usage_out: dict = {}
    u = o.get("usage") or {}
    if isinstance(u, dict):
        usage_out["input_tokens"] = u.get("prompt_tokens", 0) or 0
        usage_out["output_tokens"] = u.get("completion_tokens", 0) or 0

    return {
        "id": f"msg_{(o.get('id') or uuid.uuid4().hex)[:24]}",
        "type": "message",
        "role": "assistant",
        "model": o.get("model") or fallback_model or "",
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage_out,
    }


class IncrementalAnthropicBridge:
    """Stateful, chunk-at-a-time translator that converts an OpenAI-format SSE stream into
    Anthropic-format SSE events. Unlike the batch translator below, this one emits events
    AS the upstream chunks arrive — keeps clients with completion-event timeouts happy on
    long generations (claude-code, opencode, Anthropic SDK all have this).

    Usage:
        bridge = IncrementalAnthropicBridge(fallback_model="claude-opus-4-7")
        async for chunk in upstream:
            out = bridge.feed(chunk)
            if out: yield out
        out = bridge.flush()
        if out: yield out
    """

    def __init__(self, fallback_model: str | None = None):
        self.fallback_model = fallback_model or ""
        self.model = fallback_model or ""
        self._buf = ""
        self._msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        self._started = False
        self._finished = False
        # Block state — Anthropic protocol requires a strict open/close pattern.
        self._cur_block_idx = -1
        self._cur_block_type: str | None = None  # 'text' | 'tool_use' | None
        # Tool-call accumulator: openai-tc-index → {block_idx, id, name, started}
        self._tool_calls: dict = {}
        self._next_block_idx = 0
        self._finish_reason: str | None = None
        self._input_tokens = 0
        self._output_tokens = 0

    @staticmethod
    def _event(name: str, payload: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

    def _emit_start(self, out: list, prompt_tokens: int = 0):
        if self._started:
            return
        self._started = True
        out.append(self._event("message_start", {
            "type": "message_start",
            "message": {
                "id": self._msg_id, "type": "message", "role": "assistant",
                "model": self.model, "content": [],
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
            },
        }))

    def _close_current_block(self, out: list):
        if self._cur_block_type is None:
            return
        out.append(self._event("content_block_stop", {
            "type": "content_block_stop", "index": self._cur_block_idx,
        }))
        self._cur_block_type = None

    def _open_text_block(self, out: list):
        if self._cur_block_type == "text":
            return
        self._close_current_block(out)
        self._cur_block_idx = self._next_block_idx
        self._next_block_idx += 1
        out.append(self._event("content_block_start", {
            "type": "content_block_start", "index": self._cur_block_idx,
            "content_block": {"type": "text", "text": ""},
        }))
        self._cur_block_type = "text"

    def _process_chunk(self, j: dict, out: list):
        if isinstance(j.get("model"), str) and j["model"]:
            self.model = j["model"]
        if not self._started:
            usage = j.get("usage") or {}
            self._emit_start(out, usage.get("prompt_tokens", 0) or 0)
        for c in (j.get("choices") or []):
            if c.get("finish_reason"):
                self._finish_reason = c["finish_reason"]
            d = c.get("delta") or {}
            content = d.get("content")
            if isinstance(content, str) and content:
                self._open_text_block(out)
                out.append(self._event("content_block_delta", {
                    "type": "content_block_delta", "index": self._cur_block_idx,
                    "delta": {"type": "text_delta", "text": content},
                }))
            for tc in (d.get("tool_calls") or []):
                tc_idx = tc.get("index", 0)
                slot = self._tool_calls.get(tc_idx)
                if slot is None:
                    slot = {"block_idx": None, "id": "", "name": "", "started": False}
                    self._tool_calls[tc_idx] = slot
                if tc.get("id") and not slot["id"]:
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                new_name = fn.get("name")
                if new_name and not slot["name"]:
                    slot["name"] = new_name
                if slot["name"] and not slot["started"]:
                    # First time we know the tool's name → open the tool_use block.
                    self._close_current_block(out)
                    slot["block_idx"] = self._next_block_idx
                    self._next_block_idx += 1
                    self._cur_block_idx = slot["block_idx"]
                    self._cur_block_type = "tool_use"
                    out.append(self._event("content_block_start", {
                        "type": "content_block_start", "index": slot["block_idx"],
                        "content_block": {
                            "type": "tool_use",
                            "id": slot["id"] or f"toolu_{uuid.uuid4().hex[:12]}",
                            "name": slot["name"], "input": {},
                        },
                    }))
                    slot["started"] = True
                args_delta = fn.get("arguments")
                if args_delta and slot["started"]:
                    # If this tool isn't the currently-open block (interleaved), close cur
                    # and re-open this one. Most providers emit one tool's args fully then
                    # the next, so this rarely fires.
                    if self._cur_block_idx != slot["block_idx"]:
                        self._close_current_block(out)
                        self._cur_block_idx = slot["block_idx"]
                        self._cur_block_type = "tool_use"
                    out.append(self._event("content_block_delta", {
                        "type": "content_block_delta", "index": slot["block_idx"],
                        "delta": {"type": "input_json_delta", "partial_json": args_delta},
                    }))
        u = j.get("usage")
        if isinstance(u, dict):
            if u.get("completion_tokens"):
                self._output_tokens = u["completion_tokens"]
            if u.get("prompt_tokens") and not self._input_tokens:
                self._input_tokens = u["prompt_tokens"]

    def feed(self, raw: bytes | str) -> bytes:
        """Process incoming OpenAI SSE bytes, return Anthropic SSE bytes ready to forward.
        Returns b'' if no complete events were finished by this chunk."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        self._buf += raw
        # SSE event boundaries are blank lines (\n\n). Split, keep last partial in buf.
        segments = self._buf.split("\n\n")
        self._buf = segments[-1]
        out: list = []
        for seg in segments[:-1]:
            for line in seg.split("\n"):
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    j = json.loads(data)
                except (json.JSONDecodeError, TypeError):
                    continue
                self._process_chunk(j, out)
        return "".join(out).encode("utf-8") if out else b""

    def flush(self) -> bytes:
        """Emit final events: close any open block + message_delta + message_stop."""
        if self._finished:
            return b""
        self._finished = True
        out: list = []
        if not self._started:
            self._emit_start(out)
        self._close_current_block(out)
        stop_reason = _FINISH_OPENAI_TO_ANTHROPIC.get(
            self._finish_reason, self._finish_reason or "end_turn"
        )
        out.append(self._event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": self._output_tokens},
        }))
        out.append(self._event("message_stop", {"type": "message_stop"}))
        return "".join(out).encode("utf-8")


def _openai_sse_to_anthropic_events(openai_chunks: bytes | str, fallback_model: str | None = None) -> bytes:
    """Translate a complete buffered OpenAI SSE stream into an Anthropic SSE event stream.
    Buffered (not chunk-by-chunk): we accumulate the OpenAI deltas, then emit Anthropic events.
    The trade-off is one round-trip latency for protocol fidelity — acceptable since most agent
    clients buffer tool-call responses anyway."""
    text = openai_chunks.decode("utf-8", errors="replace") if isinstance(openai_chunks, bytes) else str(openai_chunks)

    accum_text = ""
    tcs: dict[int, dict] = {}  # index -> {id, name, args}
    tool_index_order: list[int] = []
    finish_reason: str | None = None
    model = fallback_model or ""
    usage: dict = {}

    for line in text.split("\n"):
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            continue
        try:
            j = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(j.get("model"), str):
            model = j["model"]
        for c in (j.get("choices") or []):
            if c.get("finish_reason"):
                finish_reason = c["finish_reason"]
            d = c.get("delta") or {}
            if isinstance(d.get("content"), str):
                accum_text += d["content"]
            for tc in (d.get("tool_calls") or []):
                idx = tc.get("index", 0)
                if idx not in tcs:
                    tcs[idx] = {"id": "", "name": "", "args": ""}
                    tool_index_order.append(idx)
                slot = tcs[idx]
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
        u = j.get("usage")
        if isinstance(u, dict):
            usage = u

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    input_tokens = usage.get("prompt_tokens", 0) or 0
    output_tokens = usage.get("completion_tokens", 0) or 0
    stop_reason = _FINISH_OPENAI_TO_ANTHROPIC.get(finish_reason, finish_reason or "end_turn")

    events: list[str] = []

    def evt(name: str, payload: dict) -> None:
        events.append(f"event: {name}\ndata: {json.dumps(payload)}\n\n")

    evt("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    })

    block_idx = 0
    if accum_text:
        evt("content_block_start", {
            "type": "content_block_start",
            "index": block_idx,
            "content_block": {"type": "text", "text": ""},
        })
        evt("content_block_delta", {
            "type": "content_block_delta",
            "index": block_idx,
            "delta": {"type": "text_delta", "text": accum_text},
        })
        evt("content_block_stop", {"type": "content_block_stop", "index": block_idx})
        block_idx += 1

    for idx in tool_index_order:
        tc = tcs[idx]
        try:
            input_obj = json.loads(tc["args"]) if tc["args"] else {}
        except (json.JSONDecodeError, TypeError):
            input_obj = {}
        tu_id = tc["id"] or f"toolu_{uuid.uuid4().hex[:12]}"
        evt("content_block_start", {
            "type": "content_block_start",
            "index": block_idx,
            "content_block": {"type": "tool_use", "id": tu_id, "name": tc["name"], "input": {}},
        })
        # Emit the input as one delta — we already have the full JSON.
        evt("content_block_delta", {
            "type": "content_block_delta",
            "index": block_idx,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(input_obj)},
        })
        evt("content_block_stop", {"type": "content_block_stop", "index": block_idx})
        block_idx += 1

    evt("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    evt("message_stop", {"type": "message_stop"})

    return "".join(events).encode("utf-8")


def _pick_shadows(body: dict, ctx: dict) -> list[dict]:
    """Returns a list of shadow target dicts: [{target_model, upstream_base, upstream_label}].
    Each entry will be fanned out asynchronously by _run_shadow."""
    if not isinstance(body, dict):
        return []
    cfg = (load_rules_config().get("shadow_router") or {})
    if not cfg.get("enabled", False):
        return []
    out: list[dict] = []
    seen: set = set()
    for r in (cfg.get("rules") or []):
        if not isinstance(r, dict):
            continue
        if not _match_router_cond(r.get("if") or {}, body, ctx):
            continue
        target = r.get("shadow_to")
        if not isinstance(target, str) or not target:
            continue
        if target in seen:
            continue
        seen.add(target)
        is_claude = _is_claude_model(target)
        out.append({
            "target_model": target,
            "upstream_label": "anthropic" if is_claude else "ollama",
            "upstream_base": ANTHROPIC_URL if is_claude else OLLAMA_URL,
        })
    return out


async def _run_shadow(primary_id: str, primary_body: dict, primary_path: str,
                      target: dict, viewer_ip: str | None, app) -> None:
    """Best-effort: send a parallel non-streaming request to a shadow target and store the
    response. Body is translated if shapes differ. Failures are swallowed (recorded in the
    shadow row's error field). Never raises — must not affect the primary request."""
    try:
        shadow_id = uuid.uuid4().hex
        shadow_body = json.loads(json.dumps(primary_body))  # deep copy; safe to mutate
        shadow_body["model"] = target["target_model"]
        shadow_body["stream"] = False
        shadow_body.pop("stream_options", None)

        is_primary_anthropic = primary_path.startswith("/v1/messages")
        is_shadow_claude = _is_claude_model(target["target_model"])
        translated = False
        shadow_path = primary_path.lstrip("/")
        if is_primary_anthropic and not is_shadow_claude:
            shadow_body = _anthropic_to_openai_request(shadow_body)
            shadow_body["stream"] = False
            shadow_body.pop("stream_options", None)
            shadow_path = "v1/chat/completions"
            translated = True
        elif (not is_primary_anthropic) and is_shadow_claude:
            # OpenAI primary → Claude shadow needs a reverse translator we haven't built yet.
            return

        # Apply ollama_options to the shadow body when the target is an Ollama-style upstream,
        # so per_model defaults like num_ctx reach the shadow without having to duplicate config.
        if target["upstream_label"] != "anthropic":
            shadow_router_ctx = {
                "client_ip": viewer_ip,
                "path": "/" + shadow_path,
                "req_id": shadow_id,
                "upstream": target["upstream_label"],
            }
            try:
                evaluate_ollama_options(shadow_body, shadow_router_ctx)
            except Exception:
                pass
            # Strip per-request volatile prefix so llama.cpp's KV cache survives between turns.
            # (For translated bodies _anthropic_to_openai_request already did this; this catches
            # passthrough OpenAI-shape shadows.)
            try:
                _strip_volatile_prefix(shadow_body)
            except Exception:
                pass

        upstream_url = f"{target['upstream_base']}/{shadow_path}"
        body_text = json.dumps(shadow_body)
        conv_id = _conversation_id(primary_body)
        turn = _turn_index(primary_body)

        conn = db()
        conn.execute(
            """INSERT INTO requests (id, ts, method, path, upstream_url, request_headers,
                                       model, is_stream, client_ip,
                                       conversation_id, turn_index, client_app, shadow_of)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                shadow_id, time.time(), "POST", "/" + shadow_path, upstream_url,
                json.dumps({"x-proxy-shadow-of": primary_id, "x-proxy-translated": translated}),
                target["target_model"], 0, viewer_ip,
                conv_id, turn, "shadow", primary_id,
            ),
        )
        _blobs_upsert(conn, shadow_id, request_body=body_text)
        conn.commit()
        conn.close()
        _LIVE_STREAMS[shadow_id] = {
            "prompt": None, "completion": None,
            "est_prompt": _estimate_prompt_tokens(shadow_body, 3.5) or None,
        }

        start = time.perf_counter()
        err: str | None = None
        body_resp: str | None = None
        status = 0
        resp_headers: dict = {}
        try:
            client_http: httpx.AsyncClient = app.state.client
            req_headers = {"content-type": "application/json", "accept-encoding": "identity"}
            # Anthropic shadows need the API key forwarded; reuse the env var if set.
            if target["upstream_label"] == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if api_key:
                    req_headers["x-api-key"] = api_key
                    req_headers["anthropic-version"] = "2023-06-01"
            upstream_req = client_http.build_request(
                "POST", upstream_url, headers=req_headers, content=body_text.encode("utf-8"),
                timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
            )
            resp = await client_http.send(upstream_req)
            status = resp.status_code
            resp_headers = dict(resp.headers)
            body_resp = resp.text
            await resp.aclose()
        except Exception as e:
            err = f"shadow upstream error: {e!r}"
        elapsed = (time.perf_counter() - start) * 1000
        _save_finish(shadow_id, status, resp_headers, body_resp, None, elapsed, err)
    except Exception as e:
        # Defensive — shadow runner must never propagate errors.
        try:
            print(f"[shadow] uncaught error for primary {primary_id}: {e!r}")
        except Exception:
            pass


# -------- Proxy-owned tools (memory, etc.) --------
# Tool defs are provider-agnostic (name/description/parameters); injection emits the
# right shape per request. Keep this list small and high-signal — every entry costs
# tokens on every request that has tool_injector enabled.

PROXY_TOOLS_MEMORY: list[dict] = [
    {
        "name": "remember",
        "description": (
            "Store a value in proxy-managed memory for this conversation. The value persists "
            "across turns of the same conversation. Use when the user asks you to remember "
            "something or you identify a fact worth retaining for later."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short identifier (e.g. 'preferred_language')"},
                "value": {"type": "string", "description": "The value to remember"},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "recall",
        "description": "Retrieve a previously remembered value from proxy memory by key.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "The key to look up"}},
            "required": ["key"],
        },
    },
    {
        "name": "list_memory",
        "description": "List all keys currently stored in proxy memory for this conversation.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "forget",
        "description": "Remove a key from proxy memory.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
]

PROXY_TOOLS_TODOS: list[dict] = [
    {
        "name": "set_todos",
        "description": (
            "Replace this conversation's entire todo list. Use to plan multi-step work or "
            "track progress. Each todo has 'text' (the task) and optional 'status' "
            "(pending | in_progress | completed; default pending). Pass an empty list to clear."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        },
                        "required": ["text"],
                    },
                },
            },
            "required": ["todos"],
        },
    },
    {
        "name": "get_todos",
        "description": "Get this conversation's current todo list with statuses.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "add_todo",
        "description": "Append a new pending todo to this conversation's list.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "complete_todo",
        "description": (
            "Mark a todo as completed. Provide either 'idx' (1-based index from get_todos) "
            "or 'text' (substring match against existing todo text)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "idx": {"type": "integer"},
                "text": {"type": "string"},
            },
        },
    },
]


PROXY_TOOLS_WEB: list[dict] = [
    {
        "name": "web_search",
        "description": (
            "Search the web for up-to-date information. Returns the top results with title, "
            "URL, and snippet. Use when the user asks about current events, recent docs, or "
            "facts you may not be sure about — not for opinions or things obvious from context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Max results (1-10, default 5)"},
            },
            "required": ["query"],
        },
    },
]


def _resolve_tool_injector_cfg(ctx: dict | None = None) -> dict:
    """Return the effective tool_injector config for a request. Walks `scopes` (a list of
    {match: {...}, ...overrides}) in order — the first match's overrides are merged onto
    the root config. ctx keys: `client_ip`, `user_agent`, `client_app`. With no ctx (or no
    scopes configured), returns the root config as-is.

    A scope match is ALL of: ip (exact), ip_cidr, user_agent (substring, case-insensitive),
    client_app (exact). Empty match {} matches nothing (avoids accidentally global rules)."""
    cfg = dict(load_rules_config().get("tool_injector") or {})
    # Per-request override via X-Proxy-Tools header. Wins over both root config and scopes —
    # explicit opt-in from the caller. Empty list = explicit no-tools (still enables nothing).
    if ctx and isinstance(ctx.get("tool_bundles_override"), list):
        bundles = set(ctx["tool_bundles_override"])
        out = {**cfg, "enabled": True, "memory": False, "todos": False, "web_search": False,
               "_override": True}
        if "memory" in bundles: out["memory"] = True
        if "todos" in bundles: out["todos"] = True
        if "web_search" in bundles or "web" in bundles: out["web_search"] = True
        if not (out["memory"] or out["todos"] or out["web_search"]):
            out["enabled"] = False
        return out
    scopes = cfg.get("scopes") or []
    if not isinstance(scopes, list) or not ctx:
        return cfg
    ip = (ctx.get("client_ip") or "")
    ua = (ctx.get("user_agent") or "").lower()
    app = (ctx.get("client_app") or "")
    for sc in scopes:
        if not isinstance(sc, dict):
            continue
        m = sc.get("match") or {}
        if not isinstance(m, dict) or not m:
            continue
        ok = True
        if m.get("ip"):
            if str(m["ip"]) != ip:
                ok = False
        if ok and m.get("ip_cidr"):
            try:
                if ipaddress.ip_address(ip) not in ipaddress.ip_network(m["ip_cidr"], strict=False):
                    ok = False
            except (ValueError, TypeError):
                ok = False
        if ok and m.get("user_agent"):
            if str(m["user_agent"]).lower() not in ua:
                ok = False
        if ok and m.get("client_app"):
            if str(m["client_app"]) != app:
                ok = False
        if ok:
            merged = dict(cfg)
            for k, v in sc.items():
                if k != "match":
                    merged[k] = v
            merged["_scope_matched"] = m
            return merged
    return cfg


def _active_proxy_tools(ctx: dict | None = None) -> list[dict]:
    """Return the active set of proxy-owned tool definitions per the effective config
    (root config + any matching scope override)."""
    cfg = _resolve_tool_injector_cfg(ctx)
    out: list[dict] = []
    # When override is in effect, default-off; otherwise default-on for memory/todos
    # (preserves prior behavior where memory/todos were on by default once enabled).
    is_override = bool(cfg.get("_override"))
    if cfg.get("memory", not is_override):
        out.extend(PROXY_TOOLS_MEMORY)
    if cfg.get("todos", not is_override):
        out.extend(PROXY_TOOLS_TODOS)
    if cfg.get("web_search", False):
        out.extend(PROXY_TOOLS_WEB)
    return out


# Combined name set used by post-flight detection. Static (covers everything we might inject).
PROXY_TOOLS: list[dict] = PROXY_TOOLS_MEMORY + PROXY_TOOLS_TODOS + PROXY_TOOLS_WEB
PROXY_TOOL_NAMES: set[str] = {t["name"] for t in PROXY_TOOLS}


async def _exec_web_search(query: str, num: int) -> tuple[str, bool]:
    """Web search. Uses Brave Search API when BRAVE_SEARCH_API_KEY is set, otherwise falls
    back to scraping DuckDuckGo HTML. Returns (text_block, is_error)."""
    if not query:
        return ("Error: 'query' is required.", True)
    num = max(1, min(int(num or 5), 10))
    api_key = (os.environ.get("BRAVE_SEARCH_API_KEY") or "").strip()
    if api_key:
        return await _exec_web_search_brave(query, num, api_key)
    return await _exec_web_search_ddg(query, num)


async def _exec_web_search_brave(query: str, num: int, api_key: str) -> tuple[str, bool]:
    """Brave Search Web API. https://api.search.brave.com/res/v1/web/search"""
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params = {"q": query, "count": str(num)}
    try:
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as cli:
            r = await cli.get("https://api.search.brave.com/res/v1/web/search", params=params)
    except Exception as e:
        return (f"Brave search failed: {e}", True)
    if r.status_code != 200:
        return (f"Brave search failed: HTTP {r.status_code}: {r.text[:200]}", True)
    try:
        j = r.json()
    except (json.JSONDecodeError, ValueError):
        return ("Brave search: invalid JSON response", True)
    results = ((j.get("web") or {}).get("results") or [])[:num]
    if not results:
        return (f"No results for '{query}'.", False)
    lines = []
    for i, item in enumerate(results, start=1):
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        desc = re.sub(r"<[^>]+>", "", item.get("description") or "").strip()
        lines.append(f"{i}. {title}\n   {url}\n   {desc[:300]}")
    return ("\n\n".join(lines), False)


async def _exec_web_search_ddg(query: str, num: int) -> tuple[str, bool]:
    """DuckDuckGo HTML scrape fallback. No API key needed but rate-limited and brittle."""
    import urllib.parse as _up
    import html as _html
    url = "https://html.duckduckgo.com/html/?q=" + _up.quote(query)
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        "accept-language": "en-US,en;q=0.5",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0, headers=headers, follow_redirects=True) as cli:
            r = await cli.get(url)
    except Exception as e:
        return (f"Search failed: {e}", True)
    if r.status_code != 200:
        return (f"Search failed: HTTP {r.status_code}", True)
    pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    results = []
    for m in pattern.finditer(r.text):
        if len(results) >= num:
            break
        href = _html.unescape(m.group(1))
        title = _html.unescape(re.sub(r'<[^>]+>', '', m.group(2))).strip()
        snippet = _html.unescape(re.sub(r'<[^>]+>', '', m.group(3))).strip()
        if "uddg=" in href:
            try:
                params = _up.parse_qs(_up.urlparse(href).query)
                if "uddg" in params:
                    href = _up.unquote(params["uddg"][0])
            except (ValueError, TypeError):
                pass
        if href.startswith("//"):
            href = "https:" + href
        results.append((title, href, snippet[:300]))
    if not results:
        return (f"No results for '{query}'.", False)
    out = "\n\n".join(f"{i+1}. {t}\n   {u}\n   {s}" for i, (t, u, s) in enumerate(results))
    return (out, False)


async def _exec_proxy_tool(name: str, args, conversation_id: str | None,
                           memory_scope: str | None = None) -> tuple[str, bool]:
    """Run a proxy-owned tool. Returns (result_text, is_error). Async because some tools
    (web_search) make outbound HTTP. memory_scope (when set) overrides conversation_id for
    memory tools only — lets the chat UI scope memory by personality so it persists across
    conversations."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            args = {}
    if not isinstance(args, dict):
        args = {}
    if name == "web_search":
        return await _exec_web_search(
            (args.get("query") or "").strip(),
            args.get("num_results") or 5,
        )
    # Memory tools route through memory_scope when present; todos always use conversation_id.
    if name in ("remember", "recall", "list_memory", "forget"):
        return _exec_proxy_tool_sync(name, args, memory_scope or conversation_id)
    return _exec_proxy_tool_sync(name, args, conversation_id)


def _exec_proxy_tool_sync(name: str, args: dict, conversation_id: str | None) -> tuple[str, bool]:
    """Sync handler for DB-only tools (memory, todos)."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            args = {}
    if not isinstance(args, dict):
        args = {}
    cid = conversation_id or "_global"
    conn = db()
    try:
        if name == "remember":
            key = (args.get("key") or "").strip()
            val = args.get("value")
            if not key:
                return ("Error: 'key' is required.", True)
            if val is None:
                return ("Error: 'value' is required.", True)
            val_str = val if isinstance(val, str) else json.dumps(val)
            now = time.time()
            conn.execute(
                """INSERT INTO proxy_memory (conversation_id, key, value, created_ts, updated_ts)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(conversation_id, key) DO UPDATE SET
                     value = excluded.value, updated_ts = excluded.updated_ts""",
                (cid, key, val_str, now, now),
            )
            conn.commit()
            return (f"Stored '{key}'.", False)
        if name == "recall":
            key = (args.get("key") or "").strip()
            if not key:
                return ("Error: 'key' is required.", True)
            row = conn.execute(
                "SELECT value FROM proxy_memory WHERE conversation_id = ? AND key = ?",
                (cid, key),
            ).fetchone()
            if not row:
                return (f"No memory found for key '{key}'.", False)
            return (row["value"], False)
        if name == "list_memory":
            rows = conn.execute(
                "SELECT key, length(value) AS sz, updated_ts FROM proxy_memory "
                "WHERE conversation_id = ? ORDER BY updated_ts DESC",
                (cid,),
            ).fetchall()
            if not rows:
                return ("(no memory stored for this conversation)", False)
            return (
                "Stored keys:\n" + "\n".join(f"  - {r['key']} ({r['sz']} chars)" for r in rows),
                False,
            )
        if name == "forget":
            key = (args.get("key") or "").strip()
            if not key:
                return ("Error: 'key' is required.", True)
            cur = conn.execute(
                "DELETE FROM proxy_memory WHERE conversation_id = ? AND key = ?",
                (cid, key),
            )
            conn.commit()
            return ((f"Removed '{key}'.", False) if cur.rowcount else (f"No memory found for key '{key}'.", False))

        # ─── Todos ───
        if name == "set_todos":
            todos = args.get("todos")
            if not isinstance(todos, list):
                return ("Error: 'todos' must be a list.", True)
            now = time.time()
            conn.execute("DELETE FROM proxy_todos WHERE conversation_id = ?", (cid,))
            valid_status = {"pending", "in_progress", "completed"}
            for i, t in enumerate(todos, start=1):
                if not isinstance(t, dict):
                    continue
                text = (t.get("text") or "").strip()
                if not text:
                    continue
                status = t.get("status") or "pending"
                if status not in valid_status:
                    status = "pending"
                conn.execute(
                    """INSERT INTO proxy_todos (conversation_id, idx, text, status, created_ts, updated_ts)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (cid, i, text[:500], status, now, now),
                )
            conn.commit()
            count = conn.execute(
                "SELECT COUNT(*) c FROM proxy_todos WHERE conversation_id = ?", (cid,)
            ).fetchone()["c"]
            return (f"Saved {count} todo(s).", False)

        if name == "get_todos":
            rows = conn.execute(
                "SELECT idx, text, status FROM proxy_todos WHERE conversation_id = ? ORDER BY idx ASC",
                (cid,),
            ).fetchall()
            if not rows:
                return ("(no todos for this conversation)", False)
            icons = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
            lines = [f"{r['idx']}. {icons.get(r['status'], '[?]')} {r['text']}" for r in rows]
            return ("\n".join(lines), False)

        if name == "add_todo":
            text = (args.get("text") or "").strip()
            if not text:
                return ("Error: 'text' is required.", True)
            now = time.time()
            row = conn.execute(
                "SELECT COALESCE(MAX(idx), 0) + 1 AS next_idx FROM proxy_todos WHERE conversation_id = ?",
                (cid,),
            ).fetchone()
            next_idx = row["next_idx"]
            conn.execute(
                """INSERT INTO proxy_todos (conversation_id, idx, text, status, created_ts, updated_ts)
                   VALUES (?, ?, ?, 'pending', ?, ?)""",
                (cid, next_idx, text[:500], now, now),
            )
            conn.commit()
            return (f"Added todo #{next_idx}: {text[:80]}", False)

        if name == "complete_todo":
            idx = args.get("idx")
            text_match = (args.get("text") or "").strip().lower()
            target_idx = None
            if isinstance(idx, int) and idx > 0:
                target_idx = idx
            elif text_match:
                # Find first pending/in-progress todo whose text contains the substring.
                row = conn.execute(
                    """SELECT idx FROM proxy_todos
                       WHERE conversation_id = ? AND status != 'completed'
                         AND lower(text) LIKE ?
                       ORDER BY idx ASC LIMIT 1""",
                    (cid, f"%{text_match}%"),
                ).fetchone()
                if row:
                    target_idx = row["idx"]
            if target_idx is None:
                return ("Error: provide 'idx' or 'text' matching an existing todo.", True)
            cur = conn.execute(
                """UPDATE proxy_todos SET status = 'completed', updated_ts = ?
                   WHERE conversation_id = ? AND idx = ?""",
                (time.time(), cid, target_idx),
            )
            conn.commit()
            if cur.rowcount:
                row = conn.execute(
                    "SELECT text FROM proxy_todos WHERE conversation_id = ? AND idx = ?",
                    (cid, target_idx),
                ).fetchone()
                return (f"Completed #{target_idx}: {row['text'][:80] if row else ''}", False)
            return (f"No todo at index {target_idx}.", True)

        return (f"Unknown proxy tool: {name}", True)
    except Exception as e:
        return (f"Internal error: {e!r}", True)
    finally:
        conn.close()


def _detect_body_shape(body: dict) -> str:
    """'anthropic' or 'openai' — used to inject the right tool shape."""
    if not isinstance(body, dict):
        return "openai"
    # Anthropic carries a top-level `system` field and tools without a `function` wrapper.
    if "system" in body and not isinstance(body.get("messages") or [], type(None)):
        return "anthropic"
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        t0 = tools[0]
        if isinstance(t0, dict):
            if t0.get("type") == "function":
                return "openai"
            if "input_schema" in t0:
                return "anthropic"
    return "openai"


def _inject_proxy_tools(body: dict, ctx: dict | None = None) -> int:
    """Append proxy-owned tool definitions to body['tools']. Returns count of injected tools.
    Skips clients that don't already declare tools[] (they likely don't expect tool calls)
    UNLESS the caller explicitly opted in via X-Proxy-Tools header (override path). Skips
    any tool whose name collides with a client-declared tool."""
    if not isinstance(body, dict):
        return 0
    cfg = _resolve_tool_injector_cfg(ctx)
    if not cfg.get("enabled", False):
        return 0
    active = _active_proxy_tools(ctx)
    if not active:
        return 0
    tools = body.get("tools")
    if not isinstance(tools, list):
        if cfg.get("_override"):
            tools = []
            body["tools"] = tools
        else:
            return 0
    elif not tools and not cfg.get("_override"):
        return 0
    existing = set(_tool_names_from_body(body))
    shape = _detect_body_shape(body)
    added = 0
    for t in active:
        if t["name"] in existing:
            continue
        if shape == "anthropic":
            tools.append({
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            })
        else:
            tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            })
        added += 1
    return added


def _extract_proxy_tool_calls(resp_obj: dict) -> list[dict]:
    """Find tool calls naming a proxy-owned tool in either an Anthropic or OpenAI response.
    Returns [{id, name, input(dict)}] in arrival order."""
    out: list[dict] = []
    if not isinstance(resp_obj, dict):
        return out
    # Anthropic /v1/messages shape
    if resp_obj.get("type") == "message" and isinstance(resp_obj.get("content"), list):
        for blk in resp_obj["content"]:
            if isinstance(blk, dict) and blk.get("type") == "tool_use" and blk.get("name") in PROXY_TOOL_NAMES:
                out.append({
                    "id": blk.get("id") or f"toolu_{uuid.uuid4().hex[:12]}",
                    "name": blk["name"],
                    "input": blk.get("input") or {},
                })
        return out
    # OpenAI chat.completion shape
    for c in (resp_obj.get("choices") or []):
        msg = c.get("message") or {}
        for tc in (msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if fn.get("name") in PROXY_TOOL_NAMES:
                args = fn.get("arguments")
                try:
                    parsed = json.loads(args) if isinstance(args, str) else (args or {})
                except (json.JSONDecodeError, TypeError):
                    parsed = {"_raw": args}
                out.append({
                    "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "name": fn["name"],
                    "input": parsed,
                })
    return out


def _build_followup_body_anthropic(orig_body: dict, resp_obj: dict, executed: list[dict]) -> dict:
    """Construct a new Anthropic request body that appends the assistant turn + tool_results."""
    new_body = json.loads(json.dumps(orig_body))
    msgs = new_body.get("messages") or []
    msgs.append({"role": "assistant", "content": resp_obj.get("content") or []})
    tr_blocks = []
    for ex in executed:
        tr_blocks.append({
            "type": "tool_result",
            "tool_use_id": ex["id"],
            "content": [{"type": "text", "text": ex["result"]}],
            "is_error": ex.get("is_error", False),
        })
    msgs.append({"role": "user", "content": tr_blocks})
    new_body["messages"] = msgs
    return new_body


def _build_followup_body_openai(orig_body: dict, resp_obj: dict, executed: list[dict]) -> dict:
    """Construct a new OpenAI request body that appends the assistant turn + tool messages."""
    new_body = json.loads(json.dumps(orig_body))
    msgs = new_body.get("messages") or []
    choice = (resp_obj.get("choices") or [{}])[0]
    asst_msg = choice.get("message") or {"role": "assistant", "content": ""}
    msgs.append(asst_msg)
    for ex in executed:
        msgs.append({
            "role": "tool",
            "tool_call_id": ex["id"],
            "content": ex["result"],
        })
    new_body["messages"] = msgs
    return new_body


def _extract_ttft_ms(body_text: str | None, stream_text: str | None) -> float | None:
    """Pull a precise TTFT (time to first token) from upstream response data when available.
    For Ollama native API (/api/chat, /api/generate), parse `prompt_eval_duration` (ns) —
    that's exact prefill time. For other shapes, return None and caller falls back to the
    streaming-first-chunk timestamp."""
    for source in (body_text, stream_text):
        if not source:
            continue
        if "prompt_eval_duration" not in source:
            continue
        # Ollama native: top-level field on the final response object. Try plain JSON first;
        # if it's NDJSON (streaming), find the line with the field.
        try:
            j = json.loads(source)
            if isinstance(j, dict) and isinstance(j.get("prompt_eval_duration"), (int, float)):
                return j["prompt_eval_duration"] / 1_000_000.0
        except (json.JSONDecodeError, TypeError):
            pass
        for line in source.split("\n"):
            line = line.strip()
            if not line or "prompt_eval_duration" not in line:
                continue
            if line.startswith("data: "):
                line = line[6:]
            try:
                j = json.loads(line)
                if isinstance(j, dict) and isinstance(j.get("prompt_eval_duration"), (int, float)):
                    return j["prompt_eval_duration"] / 1_000_000.0
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def _save_finish(req_id: str, status: int, resp_headers: dict, body_text: str | None,
                 stream_text: str | None, elapsed_ms: float, error: str | None,
                 ttft_ms: float | None = None):
    pt, ct, tt = _extract_usage(body_text, stream_text)
    # Prefer the precise Ollama-native TTFT when present; else fall back to whatever the
    # caller sniffed from the SSE first-chunk arrival time.
    parsed_ttft = _extract_ttft_ms(body_text, stream_text)
    if parsed_ttft is not None:
        ttft_ms = parsed_ttft
    conn = db()
    conn.execute(
        """UPDATE requests
           SET status=?, response_headers=?, duration_ms=?, error=?,
               prompt_tokens=?, completion_tokens=?, total_tokens=?, ttft_ms=?
           WHERE id=?""",
        (
            status,
            json.dumps(resp_headers) if resp_headers else None,
            elapsed_ms,
            error,
            pt, ct, tt, ttft_ms,
            req_id,
        ),
    )
    _blobs_upsert(conn, req_id,
                  response_body=_truncate_for_store(body_text),
                  stream_chunks=_truncate_for_store(stream_text))
    conn.commit()
    conn.close()
    _LIVE_STREAMS.pop(req_id, None)
    _INFLIGHT_REQUESTS.pop(req_id, None)


@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy(full_path: str, request: Request):
    # Defensive: never proxy our own UI/API namespace.
    if full_path.startswith("__proxy"):
        return JSONResponse({"error": "not found"}, status_code=404)

    # Panic kill-switch: refuse all upstream traffic until disabled. Proxy stays up so the
    # UI / phone PWA / control endpoints remain reachable to flip it off again.
    if _PANIC_MODE:
        return JSONResponse(
            {"error": {"message": "AI Proxy is in PANIC MODE — all upstream traffic blocked. Disable via the phone PWA or POST /__proxy/api/control/panic with {\"on\": false}.",
                       "type": "proxy_panic", "code": "panic_mode"}},
            status_code=503,
            headers={"X-Proxy-Panic": "1"},
        )

    # Bench exclusive-mode gate: when an exclusive bench is running, hold all non-bench
    # requests until it finishes. Bench requests self-identify with x-client-name.
    if not _BENCH_TRAFFIC_OK.is_set():
        if (request.headers.get("x-client-name") or "").lower() != "ai-proxy-bench":
            # Hard cap so a hung bench can't lock the proxy forever.
            wait_until = _BENCH_EXCLUSIVE_DEADLINE or (time.time() + 300)
            timeout = max(1.0, wait_until - time.time())
            try:
                await asyncio.wait_for(_BENCH_TRAFFIC_OK.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                # Safety release: bench overran its window; fail open.
                _BENCH_TRAFFIC_OK.set()

    req_id = uuid.uuid4().hex
    start = time.perf_counter()
    body = await request.body()

    # Path normalization: some clients construct the URL as {ollama_base}/api + /v1/models,
    # yielding /api/v1/models — which Ollama doesn't serve (404). Ollama's OpenAI-compat
    # surface lives at /v1/*, so collapse a stray `api/v1/...` prefix to `v1/...` so these
    # resolve instead of 404ing.
    if full_path.startswith("api/v1/"):
        full_path = full_path[len("api/"):]

    upstream_base, upstream_label = _pick_upstream(full_path)
    upstream_url = f"{upstream_base}/{full_path}"
    if request.url.query:
        upstream_url += f"?{request.url.query}"

    body_text = body.decode("utf-8", errors="replace") if body else ""
    body_json = None
    model = None
    is_stream = False
    if body_text:
        try:
            body_json = json.loads(body_text)
            if isinstance(body_json, dict):
                model = body_json.get("model")
                is_stream = bool(body_json.get("stream"))
        except json.JSONDecodeError:
            pass

    _save_pending(req_id, request, full_path, upstream_url, body_text, body_json, model, is_stream, upstream=upstream_label)

    # request_dedup: if an identical request from the same client is already streaming,
    # subscribe to its response stream instead of issuing our own. Saves the GPU from doing
    # the same work twice when clients like claude-code fan out parallel duplicates.
    _dedup_cfg = (load_rules_config().get("request_dedup") or {})
    if _dedup_cfg.get("enabled") and is_stream and body:
        _dedup_gc()
        _client_ip_str = _client_ip(request) or ""
        _dedup_sig = _dedup_signature(_client_ip_str, full_path, body)
        _existing_fanout = _REQUEST_FANOUT.get(_dedup_sig)
        if _existing_fanout is not None and _existing_fanout.primary_id != req_id:
            # We're a duplicate — tee from the primary.
            _save_gate(req_id, {
                "verdict": "intercept",
                "rule": "request_dedup",
                "reason": f"duplicate of primary {_existing_fanout.primary_id} — tee'd from its stream",
                "details": {
                    "primary_id": _existing_fanout.primary_id,
                    "history_chunks": len(_existing_fanout.history),
                    "primary_total_bytes": _existing_fanout.total_bytes,
                },
            })
            q = _existing_fanout.subscribe()
            _replay_content_type = "text/event-stream"

            async def replay_from_fanout():
                _replay_start = time.perf_counter()
                _replay_bytes = bytearray()
                try:
                    while True:
                        chunk = await q.get()
                        if chunk is None:
                            break
                        _replay_bytes.extend(chunk)
                        yield chunk
                finally:
                    elapsed = (time.perf_counter() - _replay_start) * 1000
                    _save_finish(req_id, 200, {"content-type": _replay_content_type},
                                 None, _replay_bytes.decode("utf-8", errors="replace"),
                                 elapsed, _existing_fanout.error)

            return StreamingResponse(
                replay_from_fanout(),
                status_code=200,
                media_type=_replay_content_type,
                headers={"x-proxy-rule": "request_dedup",
                         "x-proxy-dedup-primary": _existing_fanout.primary_id},
            )
        # We're the primary — register a fanout so future duplicates can subscribe.
        _dedup_fanout: StreamFanout | None = StreamFanout(primary_id=req_id)
        _REQUEST_FANOUT[_dedup_sig] = _dedup_fanout
    else:
        _dedup_fanout = None

    # Snapshot the body before any rule mutates it — shadows always see the original request.
    body_snapshot_for_shadows = json.loads(body_text) if isinstance(body_json, dict) else None
    snapshot_path_for_shadows = "/" + full_path

    # Phase 1: transforms — model_router rewrite + ollama_options injection (both mutate body_json).
    # Anthropic requests skip ollama_options entirely (num_ctx etc. don't apply).
    router_ctx = {"client_ip": _client_ip(request), "path": "/" + full_path, "req_id": req_id, "upstream": upstream_label}
    rewrite = evaluate_router(body_json, router_ctx) if isinstance(body_json, dict) else None

    # Per-rule upstream override: a model_router rule may carry {"upstream": "lmstudio"} to
    # send that (rewritten) model to a different backend than the path-based default. This is
    # how e.g. qwen traffic reaches LM Studio's OpenAI endpoint *through* the proxy so it's
    # inspected/logged like everything else. Only OpenAI-compat backends are valid targets —
    # both Ollama and LM Studio speak the same /v1 shape, so no body translation is needed.
    # "anthropic" is deliberately NOT a target: forwarding an OpenAI-shape body there without
    # the protocol bridge (which only goes anthropic→ollama) would produce 400s.
    _UPSTREAM_BASES = {"ollama": OLLAMA_URL, "lmstudio": LMSTUDIO_URL}
    if (rewrite and rewrite.get("upstream")
            and upstream_label != "anthropic"):
        _new_label = str(rewrite["upstream"]).lower()
        _new_base = _UPSTREAM_BASES.get(_new_label)
        if _new_base and _new_label != upstream_label:
            upstream_label = _new_label
            upstream_base = _new_base
            upstream_url = f"{upstream_base}/{full_path}"
            if request.url.query:
                upstream_url += f"?{request.url.query}"
            router_ctx["upstream"] = upstream_label
            # Keep the saved row in sync so audit/stats attribute the bytes to the real
            # destination (mirrors the protocol-bridge fix-up below).
            try:
                _conn = db()
                _conn.execute(
                    "UPDATE requests SET upstream=?, upstream_url=? WHERE id=?",
                    (upstream_label, upstream_url, req_id),
                )
                _conn.commit()
                _conn.close()
            except Exception:
                pass

    # Protocol bridge: when an Anthropic-shape request is routed (via model_router) to a
    # non-Claude model, translate body to OpenAI shape and route to OLLAMA_URL. Response
    # gets translated back on the way out (see streaming/non-streaming branches below).
    bridge_active = False
    bridge_original_model: str | None = None
    _bridge_cfg = load_rules_config().get("protocol_bridge") or {}
    if (isinstance(body_json, dict)
            and _bridge_cfg.get("enabled", True)
            and upstream_label == "anthropic"
            and (_bridge_cfg.get("force", False)
                 or not _is_claude_model(body_json.get("model")))):
        bridge_original_model = body_json.get("model")
        body_json = _anthropic_to_openai_request(body_json)
        bridge_active = True
        upstream_label = "ollama"
        upstream_base = OLLAMA_URL
        full_path = "v1/chat/completions"
        upstream_url = f"{upstream_base}/{full_path}"
        if request.url.query:
            upstream_url += f"?{request.url.query}"
        router_ctx["path"] = "/" + full_path
        router_ctx["upstream"] = upstream_label
        is_stream = bool(body_json.get("stream"))
        model = body_json.get("model")
        # Bridge changed which upstream we're talking to — keep the saved row in sync
        # for BOTH `upstream` and `upstream_url`. Without this, audit/stats queries that
        # join by upstream report the original (anthropic) destination even though the
        # bytes flowed to ollama.
        try:
            _conn = db()
            _conn.execute(
                "UPDATE requests SET upstream=?, upstream_url=? WHERE id=?",
                (upstream_label, upstream_url, req_id),
            )
            _conn.commit()
            _conn.close()
        except Exception:
            pass

    options_inject = (evaluate_ollama_options(body_json, router_ctx)
                      if isinstance(body_json, dict) and upstream_label != "anthropic" else None)
    # tool_pruner runs after model/options choice, before overflow guard, so pruning shrinks
    # the token estimate the overflow guard sees.
    pruned = evaluate_tool_pruner(body_json, router_ctx) if isinstance(body_json, dict) else None
    # context_overflow_guard runs AFTER ollama_options so it sees the effective num_ctx.
    overflow = evaluate_context_overflow(body_json, router_ctx) if isinstance(body_json, dict) else None
    overflow_blocks = bool(overflow and overflow.get("action") == "block")
    # tool_injector runs AFTER all transforms (incl. bridge translation) so the proxy tools
    # are added in whatever shape the body will actually be sent in. Returns count of injected.
    # Build a scope-match context (ip / UA / detected client_app) so per-client scopes apply.
    _ti_ctx = {
        "client_ip": _client_ip(request),
        "user_agent": request.headers.get("user-agent", ""),
        "client_app": _detect_client_app(dict(request.headers), body_json if isinstance(body_json, dict) else None),
    }
    # X-Proxy-Tools: comma-separated bundle list (memory,todos,web_search). Caller-driven
    # opt-in for which proxy tools to inject this request — overrides config + scopes.
    _xpt = request.headers.get("x-proxy-tools")
    if _xpt is not None:
        _ti_ctx["tool_bundles_override"] = [t.strip().lower() for t in _xpt.split(",") if t.strip()]
    # X-Proxy-Memory-Scope: optional override for the memory-tool storage key. The chat UI
    # uses `pers:<personality_id>` so memory persists across conversations under the same
    # personality (useful for things like name/preferences/PII you want a personality to keep).
    _xms = (request.headers.get("x-proxy-memory-scope") or "").strip()
    if _xms:
        _ti_ctx["memory_scope"] = _xms[:200]
    tool_injection_count = _inject_proxy_tools(body_json, _ti_ctx) if isinstance(body_json, dict) else 0
    tool_injection_active = tool_injection_count > 0
    # compaction_nudge: per-client strategy. Returns either a 'system_reminder' (mutates
    # body, forwards upstream) or 'synthetic_response' (short-circuits with a synthetic
    # assistant message — handled below).
    compaction = (evaluate_compaction_nudge(body_json, {**router_ctx, "client_app": _ti_ctx.get("client_app")})
                  if isinstance(body_json, dict) else None)
    compaction_reminder_injected = False
    if compaction and compaction.get("action") == "system_reminder":
        compaction_reminder_injected = _inject_system_reminder(body_json, compaction.get("text") or "")
        if compaction_reminder_injected:
            _save_gate(req_id, {
                "verdict": "rewrite",
                "rule": "compaction_nudge",
                "reason": compaction.get("reason"),
                "details": compaction,
            })
    # num_ctx ceiling: clamp an over-large context (client-set or injected) for Ollama-bound
    # traffic, independent of ollama_options.enabled. Applied after all option injection so
    # it wins over everything.
    _ctx_cap = (load_rules_config().get("ollama_options") or {}).get("num_ctx_max")
    ctx_capped = None
    if (isinstance(body_json, dict) and upstream_label != "anthropic"
            and isinstance(_ctx_cap, int) and _ctx_cap > 0):
        ctx_capped = _cap_num_ctx(body_json, _ctx_cap)

    # Streaming OpenAI-compat responses only carry a usage block when the request opts in via
    # stream_options.include_usage. Ollama's native /api stream always includes eval_count, but
    # once qwen started routing to LM Studio's /v1 endpoint, streamed requests stopped reporting
    # usage — so the dashboard's per-request token count went blank. Inject the opt-in for
    # streaming /v1 chat/completions|completions (the stream parser already reads the usage chunk).
    # Respect a client-set include_usage (true or false) rather than override it.
    usage_injected = False
    if (isinstance(body_json, dict) and body_json.get("stream")
            and upstream_label != "anthropic"
            and full_path in ("v1/chat/completions", "v1/completions")):
        _so = body_json.get("stream_options")
        if not isinstance(_so, dict):
            _so = {}
            body_json["stream_options"] = _so
        if "include_usage" not in _so:
            _so["include_usage"] = True
            usage_injected = True

    body_mutated = bool(
        rewrite or options_inject or bridge_active or tool_injection_active
        or compaction_reminder_injected
        or (pruned and pruned.get("action") == "prune")
        or (overflow and overflow.get("action") in ("bump", "trim"))
        or ctx_capped or usage_injected
    )
    if body_mutated:
        # Re-serialize the body so all mutations are forwarded upstream.
        body = json.dumps(body_json).encode("utf-8")
        # Update the saved request_body so the request inspector shows what we actually
        # sent upstream (with tools injected, options merged, model rewritten, etc.) rather
        # than the original client-sent body. Without this, the inspector hides anything
        # the proxy added — which makes "did my tool get injected?" hard to verify.
        # Strip inline images from the SAVED copy (they're preserved in images_data) so a big
        # screenshot doesn't blow the body cap; the upstream `body` above keeps the real image.
        save_json = json.loads(json.dumps(body_json))
        _strip_image_data(save_json)
        conn = db()
        _blobs_upsert(conn, req_id, request_body=_truncate_for_store(json.dumps(save_json)))
        if rewrite:
            model = body_json.get("model")
            conn.execute("UPDATE requests SET model=? WHERE id=?", (model, req_id))
        conn.commit()
        conn.close()

    # compaction_nudge synthetic short-circuit: build a fake assistant response and return
    # it directly, never hitting upstream. Only used for non-streaming requests with
    # unknown clients (the evaluator already swapped to system_reminder_plain for streams).
    if compaction and compaction.get("action") == "synthetic_response":
        synth_body, synth_ct = _build_synthetic_response(body_json or {}, compaction.get("text") or "")
        elapsed = (time.perf_counter() - start) * 1000
        _save_gate(req_id, {
            "verdict": "intercept",
            "rule": "compaction_nudge",
            "reason": compaction.get("reason"),
            "details": compaction,
        })
        _save_finish(req_id, 200, {"content-type": synth_ct}, synth_body.decode("utf-8", errors="replace"), None, elapsed, None)
        return Response(
            content=synth_body, status_code=200,
            headers={"content-type": synth_ct, "x-proxy-suggest": "compact",
                     "x-proxy-rule": "compaction_nudge"},
        )

    # Phase 2: block/warn rules.
    gate = evaluate_rules(body_json)
    # context_overflow_guard with action=block short-circuits before evaluate_rules verdicts.
    if overflow_blocks:
        gate = {
            "verdict": "block",
            "rule": "context_overflow_guard",
            "reason": overflow.get("reason"),
            "details": overflow,
        }

    # Combine: rewrite gets folded in; block beats rewrite, rewrite beats warn beats allow.
    if gate["verdict"] == "block":
        if rewrite:
            details = gate.get("details") or {}
            details["rewrite"] = rewrite
            gate["details"] = details
        _save_gate(req_id, gate)
        elapsed = (time.perf_counter() - start) * 1000
        status_code = 413 if gate.get("rule") == "context_overflow_guard" else 400
        err_payload = {
            "error": {
                "message": f"Request blocked by AI Proxy rule {gate['rule']!r}: {gate['reason']}",
                "type": "proxy_block",
                "code": gate["rule"],
            }
        }
        err_text = json.dumps(err_payload)
        _save_finish(req_id, status_code, {"content-type": "application/json", "x-proxy-block": gate["rule"]}, err_text, None, elapsed, f"blocked: {gate['reason']}")
        return JSONResponse(err_payload, status_code=status_code, headers={"X-Proxy-Block": gate["rule"], "X-Proxy-Reason": (gate["reason"] or "")[:200]})

    pruner_warn = bool(pruned and pruned.get("action") == "warn")
    overflow_warn = bool(overflow and overflow.get("action") == "warn")
    if body_mutated or pruner_warn or overflow_warn:
        # A transform (or warn-mode rule) fired; record as 'rewrite' (or 'warn' if no body changed).
        rule_names: list[str] = []
        reason_parts: list[str] = []
        details: dict = {}
        if rewrite:
            rule_names.append("model_router")
            reason_parts.append(f"{rewrite['from']} → {rewrite['to']} (via {rewrite['via']})")
            details["rewrite"] = rewrite
        if options_inject:
            rule_names.append("ollama_options")
            kv = ", ".join(f"{k}={v}" for k, v in options_inject["applied"].items())
            reason_parts.append(f"options set: {kv}")
            details["options"] = options_inject
        if pruned:
            rule_names.append("tool_pruner")
            reason_parts.append(pruned.get("reason") or "")
            details["pruned"] = pruned
        if overflow:
            rule_names.append("context_overflow_guard")
            reason_parts.append(overflow.get("reason") or "")
            details["overflow"] = overflow
        if bridge_active:
            rule_names.append("protocol_bridge")
            reason_parts.append(f"anthropic→openai: {bridge_original_model} → {body_json.get('model')}")
            details["bridge"] = {
                "direction": "anthropic_to_openai",
                "original_model": bridge_original_model,
                "target_model": body_json.get("model"),
                "upstream": upstream_label,
            }
        if gate["verdict"] == "warn":
            details["warn"] = {"rule": gate["rule"], "reason": gate["reason"], "details": gate.get("details")}
            reason_parts.append(f"also warn: {gate['reason']}")
        verdict = "rewrite" if body_mutated else "warn"
        gate = {"verdict": verdict, "rule": "+".join(rule_names), "reason": "; ".join(p for p in reason_parts if p), "details": details}

    _save_gate(req_id, gate)

    # Fan out shadows BEFORE upstream send — they run concurrently with the primary request.
    # Best-effort: failures are recorded on the shadow row, never affect the primary.
    if isinstance(body_snapshot_for_shadows, dict):
        viewer_ip = _client_ip(request)
        for shadow_target in _pick_shadows(body_snapshot_for_shadows, router_ctx):
            asyncio.create_task(_run_shadow(req_id, body_snapshot_for_shadows,
                                            snapshot_path_for_shadows, shadow_target,
                                            viewer_ip, request.app))

    drop_headers = set(REQ_HOP_HEADERS)
    if bridge_active:
        # Strip Anthropic-specific headers so they don't leak the API key to Ollama logs.
        drop_headers |= {"x-api-key", "anthropic-version", "anthropic-beta", "anthropic-dangerous-direct-browser-access"}
    headers_out = _filter(request.headers, drop_headers)
    # Force identity encoding upstream so the audit DB stores plaintext SSE/JSON. Without
    # this httpx auto-adds gzip/deflate/br, and Anthropic returns gzipped streams that our
    # token parser, live token tracker, and SSE intercept can't read.
    headers_out = [(k, v) for (k, v) in headers_out if k.lower() != "accept-encoding"]
    headers_out.append(("accept-encoding", "identity"))
    client: httpx.AsyncClient = request.app.state.client

    # Pre-flight overhead = everything up to handing off to upstream.
    _overhead_samples.append((time.perf_counter() - start) * 1000)

    # request_priority: soft concurrency cap per priority bucket. Held until upstream
    # returns and (for streaming) the body is fully consumed. Released in the streamer's
    # finally block, or below for the non-streaming path.
    _pri_cfg = (load_rules_config().get("request_priority") or {})
    _pri_sem = None
    _pri_label = "off"
    _pri_acquired = False
    if _pri_cfg.get("enabled", False):
        _ensure_priority_sems()
        _pri_label = _resolve_request_priority(dict(request.headers), _ti_ctx.get("client_app"))
        _pri_sem = _PRIORITY_SEMS.get(_pri_label)
    if _pri_sem is not None:
        # Try to acquire with a bounded wait. If the slot doesn't free up in time, proceed
        # WITHOUT the slot so streaming clients don't time out before seeing any bytes.
        # max_wait_s=None means classic blocking behavior. 0 means non-blocking try.
        max_wait_s = _pri_cfg.get("max_wait_s", 3.0)
        if max_wait_s is None:
            await _pri_sem.acquire()
            _pri_acquired = True
        elif max_wait_s <= 0:
            _pri_acquired = _pri_sem.locked() is False and _pri_sem._value > 0
            if _pri_acquired:
                await _pri_sem.acquire()
        else:
            try:
                await asyncio.wait_for(_pri_sem.acquire(), timeout=float(max_wait_s))
                _pri_acquired = True
            except asyncio.TimeoutError:
                _pri_acquired = False  # proceed without the slot
    _pri_released = False
    def _release_pri_slot():
        nonlocal _pri_released
        if _pri_acquired and _pri_sem is not None and not _pri_released:
            _pri_sem.release()
            _pri_released = True

    try:
        upstream_req = client.build_request(
            request.method, upstream_url, headers=headers_out, content=body or None
        )
        upstream_resp = await client.send(upstream_req, stream=True)
    except Exception as e:
        _release_pri_slot()
        elapsed = (time.perf_counter() - start) * 1000
        _save_finish(req_id, 0, {}, None, None, elapsed, f"upstream error: {e!r}")
        return JSONResponse(
            {"error": "upstream unreachable", "upstream": upstream_base, "detail": str(e)},
            status_code=502,
        )

    # Register this in-flight request so /api/control/cancel/{req_id} can find it.
    _INFLIGHT_REQUESTS[req_id] = {"ts": time.time(), "upstream_resp": upstream_resp,
                                  "cancelled": False}

    # tool_call_xml_retry: when Ollama returns 500 with an XML-parse error in the body
    # (qwen-style models occasionally emit `<parameter>...</function>` without closing the
    # parameter tag), append a corrective system message to the request and re-send. A
    # plain manual retry doesn't help because the model has no context that anything went
    # wrong; the corrective hint is what makes the retry succeed.
    _tcr_cfg = (load_rules_config().get("tool_call_xml_retry") or {})
    _tcr_attempted = False
    if (_tcr_cfg.get("enabled") and not _tcr_attempted
            and upstream_resp.status_code == 500
            and upstream_label in (_tcr_cfg.get("applies_to_upstream") or ["ollama"])
            and isinstance(body_json, dict) and isinstance(body_json.get("messages"), list)):
        # Read the error body up-front so we can inspect the message. This eats a stream
        # response but it's a 500 — there's nothing useful to stream anyway.
        err_body = (await upstream_resp.aread()).decode("utf-8", errors="replace")
        await upstream_resp.aclose()
        patterns = _tcr_cfg.get("error_patterns") or ["XML syntax error"]
        if any(p in err_body for p in patterns):
            hint = _tcr_cfg.get("corrective_hint") or (
                "Your previous tool call had malformed XML. Output complete, "
                "properly-closed XML for every tool call."
            )
            retry_body = json.loads(json.dumps(body_json))  # deep copy
            retry_body["messages"] = list(retry_body["messages"]) + [
                {"role": "system", "content": hint}
            ]
            retry_bytes = json.dumps(retry_body).encode("utf-8")
            try:
                retry_req = client.build_request(
                    request.method, upstream_url, headers=headers_out, content=retry_bytes
                )
                retry_resp = await client.send(retry_req, stream=True)
                _tcr_attempted = True
                _save_gate(req_id, {
                    "verdict": "rewrite",
                    "rule": "tool_call_xml_retry",
                    "reason": f"retried after Ollama XML parse error ({len(err_body)} byte body)",
                    "details": {
                        "first_status": 500,
                        "retry_status": retry_resp.status_code,
                        "error_excerpt": err_body[:300],
                    },
                })
                upstream_resp = retry_resp
                body = retry_bytes  # so re-serialize / audit reflects what actually went up
            except Exception as e:
                # Retry failed entirely; reconstruct a synthetic response carrying the
                # original error so the client at least sees what happened.
                _release_pri_slot()
                elapsed = (time.perf_counter() - start) * 1000
                _save_finish(req_id, 500, {"content-type": "application/json"}, err_body,
                             None, elapsed, f"xml-retry-failed: {e!r}")
                return Response(content=err_body.encode("utf-8"), status_code=500,
                                media_type="application/json",
                                headers={"x-proxy-rule": "tool_call_xml_retry"})
        else:
            # Not the XML-parse error we know about — return the original 500 as-is.
            _release_pri_slot()
            elapsed = (time.perf_counter() - start) * 1000
            _save_finish(req_id, 500, {"content-type": "application/json"}, err_body,
                         None, elapsed, None)
            return Response(content=err_body.encode("utf-8"), status_code=500,
                            media_type="application/json")

    resp_headers_full = dict(upstream_resp.headers)
    out_headers = dict(_filter(upstream_resp.headers, RESP_HOP_HEADERS))
    content_type = upstream_resp.headers.get("content-type", "")
    treat_as_stream = ("text/event-stream" in content_type) or ("application/x-ndjson" in content_type) or is_stream

    # Surface model rewrites to the client. The X-Proxy-Model-Rewrite header is informational
    # (clients can choose to act on it or ignore); the body's `model` field still carries the
    # upstream's actual model unless preserve_response_model_name is enabled.
    _rewrite_from: str | None = None
    _rewrite_to: str | None = None
    if rewrite:
        _rewrite_from = rewrite.get("from")
        _rewrite_to = rewrite.get("to")
    elif bridge_active and bridge_original_model:
        _rewrite_from = bridge_original_model
        _rewrite_to = body_json.get("model") if isinstance(body_json, dict) else None
    if _rewrite_from and _rewrite_to and _rewrite_from != _rewrite_to:
        # ASCII-only value: HTTP header values must be latin-1 encodable. A Unicode arrow
        # (U+2192) here crashes response serialization (Starlette/uvicorn encode headers as
        # latin-1), which broke every request where a model rewrite actually changed the model.
        out_headers["x-proxy-model-rewrite"] = f"{_rewrite_from}->{_rewrite_to}"
    _preserve_model = bool((load_rules_config().get("model_router") or {}).get("preserve_response_model_name", False))
    _restore_model_name = _rewrite_from if (_preserve_model and _rewrite_from and _rewrite_to) else None

    # Helper used by both streaming and non-streaming paths: iteratively call upstream,
    # execute proxy-owned tools, append tool_results, and re-call until the model stops
    # using proxy tools (or max_iterations). Mutates nothing visible — returns the final
    # parsed response object plus the count of iterations consumed.
    async def _proxy_tool_loop(start_resp_obj: dict, current_body: dict) -> tuple[dict, int]:
        # Use the same scoped config as injection so per-client max_iterations applies.
        cfg = _resolve_tool_injector_cfg(_ti_ctx)
        max_iter = max(1, int(cfg.get("max_iterations", 4) or 4))
        shape = "anthropic" if (start_resp_obj.get("type") == "message") else "openai"
        cur_resp = start_resp_obj
        cur_body = current_body
        cur_conv = _conversation_id(body_json)
        followup_headers = list(headers_out)
        # Accumulated log of every proxy-tool call across all iterations. Written to the
        # requests table at the end so the detail view can show what each tool returned.
        tool_log: list[dict] = []

        def _persist_log():
            if not tool_log:
                return
            try:
                conn = db()
                conn.execute(
                    "UPDATE requests SET proxy_tool_log=? WHERE id=?",
                    (json.dumps(tool_log), req_id),
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

        for i in range(max_iter):
            calls = _extract_proxy_tool_calls(cur_resp)
            if not calls:
                _persist_log()
                return cur_resp, i
            executed = []
            for c in calls:
                t0 = time.time()
                txt, is_err = await _exec_proxy_tool(
                    c["name"], c["input"], cur_conv,
                    memory_scope=_ti_ctx.get("memory_scope"),
                )
                executed.append({"id": c["id"], "name": c["name"], "result": txt, "is_error": is_err})
                tool_log.append({
                    "iteration": i + 1,
                    "name": c["name"],
                    "input": c["input"],
                    "result": txt,
                    "is_error": is_err,
                    "ts": t0,
                    "duration_ms": round((time.time() - t0) * 1000, 1),
                })
            if shape == "anthropic":
                cur_body = _build_followup_body_anthropic(cur_body, cur_resp, executed)
            else:
                cur_body = _build_followup_body_openai(cur_body, cur_resp, executed)
            cur_body["stream"] = False  # follow-ups are non-streaming for simplicity
            cur_body.pop("stream_options", None)
            try:
                fu_req = client.build_request(
                    "POST", upstream_url, headers=followup_headers,
                    content=json.dumps(cur_body).encode("utf-8"),
                )
                fu_resp = await client.send(fu_req)
                fu_text = (await fu_resp.aread()).decode("utf-8", errors="replace")
                await fu_resp.aclose()
                fu_obj = json.loads(fu_text)
                cur_resp = fu_obj if isinstance(fu_obj, dict) else cur_resp
            except Exception:
                _persist_log()
                return cur_resp, i + 1
        _persist_log()
        return cur_resp, max_iter

    if treat_as_stream:
        do_intercept = _post_flight_active(body_json) and 200 <= upstream_resp.status_code < 300
        # Bridge translation AND tool-injection both require the full upstream stream first.
        do_buffer = do_intercept or bridge_active or tool_injection_active

        # Emit keepalive bytes every N seconds while we're waiting on upstream. Prevents
        # client-side read timeouts during long prefills (claude-code times out after ~30s
        # with no data — the keepalive resets that clock).
        # Shape matters: bare SSE comments work for vanilla clients but Anthropic's SDK
        # requires actual event-shaped messages to recognize the stream as alive, so when
        # the client is talking Anthropic (path=/v1/messages*, OR bridge_active which
        # delivers Anthropic-shape output), we emit a proper `event: ping\ndata: {...}`
        # block. Otherwise a generic SSE comment line.
        _ka_is_sse = "text/event-stream" in (content_type or "")
        _ka_anthropic = bridge_active or full_path.startswith("v1/messages")
        if _ka_anthropic:
            _ka_payload = b'event: ping\ndata: {"type":"ping"}\n\n'
        else:
            _ka_payload = b": ai-proxy keepalive\n\n"
        # Claude Code's SDK has a read timeout somewhere around 10-13 seconds based on
        # observed retry intervals, so 15s was too slow. 5s gives us 2-3 keepalives before
        # any realistic client times out.
        _ka_interval_s = 5.0

        # Incremental bridge translator: when bridge_active AND no post-flight transforms
        # (tool injection, intercept) are active, translate each upstream chunk to
        # Anthropic SSE events as it arrives instead of buffering the whole stream and
        # translating at the end. Without this, clients with completion-event timeouts
        # (Anthropic SDK, claude-code, opencode) give up during long generations because
        # they see only keepalives — no actual content events.
        _do_incr_bridge = bridge_active and not do_intercept and not tool_injection_active
        _incr_bridge = IncrementalAnthropicBridge(bridge_original_model) if _do_incr_bridge else None

        async def streamer():
            # REVERTED to simple async-for. The earlier asyncio.wait keepalive pattern
            # broke bridged claude-code requests starting ~2026-05-15 (success rate went
            # from ~97% to 0%). Keepalive bytes are now sent ONCE immediately and once
            # before generation starts; the simple async-for handles the actual upstream
            # reads without the create_task/wait dance that was wedging.
            chunks: list[bytes] = []
            err: str | None = None
            first_chunk_ms: float | None = None
            if _ka_is_sse:
                yield _ka_payload  # immediate "byte on wire" so the client doesn't TTFB-timeout
            if not bridge_active:
                # Non-bridged SSE (e.g. Hermes → LM Studio): run the upstream read in a task and
                # interleave keepalives during long silences (a big-prompt prefill emits no tokens
                # for 30-60s) by racing the QUEUE get — never the socket read — with the keepalive
                # timer. Racing/cancelling the raw read is what wedged bridged requests before, so
                # those stay on the simple async-for in the else branch.
                _q: "asyncio.Queue" = asyncio.Queue()

                async def _reader():
                    try:
                        async for _c in upstream_resp.aiter_raw():
                            await _q.put(("c", _c))
                    except Exception as _e:
                        await _q.put(("e", f"stream error: {_e!r}"))
                    await _q.put(("d", None))

                _rt = asyncio.create_task(_reader())
                try:
                    while True:
                        try:
                            _kind, _val = await asyncio.wait_for(_q.get(), _ka_interval_s)
                        except asyncio.TimeoutError:
                            if _ka_is_sse:
                                yield _ka_payload      # heartbeat during the prefill wait
                            continue
                        if _kind == "c":
                            if first_chunk_ms is None and _val:
                                first_chunk_ms = (time.perf_counter() - start) * 1000
                            chunks.append(_val)
                            try:
                                _live_update_from_chunk(req_id, _val.decode("utf-8", errors="replace"))
                            except Exception:
                                pass
                            if not do_buffer:
                                yield _val
                        elif _kind == "e":
                            err = _val
                            break
                        else:
                            break
                finally:
                    _rt.cancel()
                    try:
                        await _rt
                    except Exception:
                        pass
            else:
                try:
                    async for chunk in upstream_resp.aiter_raw():
                        if first_chunk_ms is None and chunk:
                            first_chunk_ms = (time.perf_counter() - start) * 1000
                        chunks.append(chunk)
                        try:
                            _live_update_from_chunk(req_id, chunk.decode("utf-8", errors="replace"))
                        except Exception:
                            pass
                        if _do_incr_bridge and _incr_bridge is not None:
                            translated_chunk = _incr_bridge.feed(chunk)
                            if translated_chunk:
                                yield translated_chunk
                        elif not do_buffer:
                            yield chunk
                except Exception as e:
                    err = f"stream error: {e!r}"
            try:
                await upstream_resp.aclose()
            except Exception:
                pass

            full = b"".join(chunks).decode("utf-8", errors="replace")
            findings: list[dict] = []
            fixes: list[dict] = []
            replaced = False

            # Phase 2: post-flight on the assembled OpenAI response (autofix → validate).
            # Runs BEFORE bridge translation, so validators see the OpenAI tool_call shape they expect.
            xml_findings: list[dict] = []
            xml_applied_s: list[dict] = []
            if do_intercept and not err:
                resp_obj = _assemble_streaming_response(full)
                fixes = _autofix_tool_calls(resp_obj, body_json)
                findings = _validate_response_tool_calls(resp_obj, body_json)
                # XML autofix.
                _xa_cfg = (load_rules_config().get("xml_autofix") or {})
                _xa_act_s = _xa_cfg.get("action", "audit")
                if _xa_cfg.get("enabled") and _xa_act_s != "silent":
                    if _xa_act_s == "fix":
                        xml_applied_s = _xml_fix_resp_obj(resp_obj)
                        if xml_applied_s:
                            # Replace buffered SSE with a synthetic stream of the fixed response.
                            synth_bytes = _synth_response_stream(resp_obj)
                            full = synth_bytes.decode("utf-8", errors="replace")
                            replaced = True
                            if not bridge_active:
                                yield synth_bytes
                    else:
                        xml_findings = _xml_detect_errors(_extract_assistant_text(resp_obj))
                if findings:
                    completion_id = resp_obj.get("id") or f"proxy-{uuid.uuid4().hex[:12]}"
                    model_name = resp_obj.get("model") or model
                    correction_msg = _format_intercept_message(findings)
                    synth_bytes = _synth_correction_stream(correction_msg, completion_id, model_name)
                    full = synth_bytes.decode("utf-8", errors="replace")
                    replaced = True
                    if not bridge_active:
                        yield synth_bytes
                elif fixes:
                    synth_bytes = _synth_response_stream(resp_obj)
                    full = synth_bytes.decode("utf-8", errors="replace")
                    replaced = True
                    if not bridge_active:
                        yield synth_bytes
                elif not bridge_active:
                    for chunk in chunks:
                        yield chunk

            # Phase 2.5: proxy-owned tool execution loop. If the model called a proxy-injected
            # tool (e.g. memory.recall), run it, append a tool_result, and re-call upstream until
            # the model produces a non-tool response (capped at max_iterations).
            tool_iters = 0
            if tool_injection_active and not err:
                # Parse buffered upstream response into an object the loop can iterate on.
                start_obj = _assemble_streaming_response(full) if not bridge_active else _assemble_streaming_response(full)
                if isinstance(start_obj, dict):
                    final_obj, tool_iters = await _proxy_tool_loop(start_obj, body_json)
                    if tool_iters > 0 and isinstance(final_obj, dict):
                        # Replace buffered SSE with a synthetic stream of the final response.
                        synth_bytes = _synth_response_stream(final_obj)
                        full = synth_bytes.decode("utf-8", errors="replace")
                        replaced = True
                        if not bridge_active:
                            yield synth_bytes

            # Phase 3: protocol bridge — translate the (possibly intercepted) OpenAI stream into
            # Anthropic SSE events for the client. Two paths:
            #   - Incremental: we already yielded translated events chunk-by-chunk above.
            #     Just flush the final message_delta + message_stop.
            #   - Batch: post-flight transforms (tool injection, intercept) needed the whole
            #     stream before we could translate. Yield the translated lump now.
            if bridge_active and not err:
                if _do_incr_bridge and _incr_bridge is not None:
                    final = _incr_bridge.flush()
                    if final:
                        yield final
                else:
                    translated = _openai_sse_to_anthropic_events(full, fallback_model=bridge_original_model)
                    yield translated
            elif bridge_active and err:
                # Surface upstream error as an Anthropic-shape error event to the client.
                err_payload = {
                    "type": "error",
                    "error": {"type": "api_error", "message": err},
                }
                yield f"event: error\ndata: {json.dumps(err_payload)}\n\n".encode("utf-8")
            elif do_buffer and not do_intercept and not tool_iters:
                # Buffered for some reason but nothing transformed it — replay raw chunks.
                for chunk in chunks:
                    yield chunk

            if tool_iters > 0:
                _save_gate(req_id, {
                    "verdict": "intercept",
                    "rule": "tool_injector",
                    "reason": f"executed {tool_iters} proxy-tool round(s)",
                    "details": {"iterations": tool_iters, "streaming": True},
                })

            elapsed = (time.perf_counter() - start) * 1000
            if findings:
                _save_gate(req_id, {
                    "verdict": "intercept",
                    "rule": "schema_validator" if any(f["kind"] == "invalid_args" for f in findings) else "hallucinated_tool",
                    "reason": "; ".join(f"{f['tool_name']}: {', '.join(f['errors'])}" for f in findings),
                    "details": {"findings": findings, "fixes": fixes, "replaced": replaced, "streaming": True},
                })
            elif fixes and (load_rules_config().get("tool_args_autofix") or {}).get("action", "audit") != "silent":
                _save_gate(req_id, {
                    "verdict": "rewrite",
                    "rule": "tool_args_autofix",
                    "reason": "; ".join(f"{f['tool_name']}: filled {', '.join(f['fixed_fields'].keys())}" for f in fixes),
                    "details": {"fixes": fixes, "streaming": True, "replaced": replaced},
                })
            if xml_findings and (load_rules_config().get("xml_autofix") or {}).get("action", "audit") == "audit":
                _kinds: dict = {}
                for f in xml_findings:
                    _kinds[f["kind"]] = _kinds.get(f["kind"], 0) + 1
                _save_gate(req_id, {
                    "verdict": "warn",
                    "rule": "xml_autofix",
                    "reason": f"{len(xml_findings)} XML issue(s): " + ", ".join(f"{k}={v}" for k, v in _kinds.items()),
                    "details": {"findings": xml_findings[:50], "streaming": True},
                })
            elif xml_applied_s:
                _save_gate(req_id, {
                    "verdict": "rewrite",
                    "rule": "xml_autofix",
                    "reason": f"{len(xml_applied_s)} XML fix(es): " + ", ".join(
                        f"{f['kind']}" + (f"({f.get('count')})" if f.get('count') else "")
                        for f in xml_applied_s
                    ),
                    "details": {"fixes": xml_applied_s, "streaming": True, "replaced": True},
                })
            _save_finish(req_id, upstream_resp.status_code, resp_headers_full, None, full, elapsed, err, ttft_ms=first_chunk_ms)

        # Wrap streamer to (a) release the priority slot even on disconnect/error, (b)
        # optionally rewrite the `model` field in every emitted chunk when the user asked
        # for it (preserve_response_model_name), and (c) tee each chunk into the dedup
        # fanout so concurrent duplicates receive the same bytes.
        async def _gen_with_release():
            stream_err: str | None = None
            try:
                async for chunk in streamer():
                    if _restore_model_name and chunk:
                        try:
                            chunk = _override_model_in_text(
                                chunk.decode("utf-8", errors="replace"), _restore_model_name
                            ).encode("utf-8")
                        except Exception:
                            pass
                    if _dedup_fanout is not None:
                        _dedup_fanout.push(chunk)
                    yield chunk
            except Exception as e:
                stream_err = f"{type(e).__name__}: {e}"
                raise
            finally:
                _release_pri_slot()
                if _dedup_fanout is not None:
                    _dedup_fanout.finish(stream_err)

        return StreamingResponse(
            _gen_with_release(),
            status_code=upstream_resp.status_code,
            headers=out_headers,
            media_type=content_type or "text/event-stream",
        )

    body_bytes = await upstream_resp.aread()
    await upstream_resp.aclose()
    _release_pri_slot()
    elapsed = (time.perf_counter() - start) * 1000
    body_text_resp = body_bytes.decode("utf-8", errors="replace")

    # Post-flight intercept (non-streaming): autofix → validate → replace if still invalid.
    if _post_flight_active(body_json) and 200 <= upstream_resp.status_code < 300:
        try:
            resp_obj = json.loads(body_text_resp)
        except (json.JSONDecodeError, TypeError):
            resp_obj = None
        if isinstance(resp_obj, dict):
            fixes = _autofix_tool_calls(resp_obj, body_json)
            findings = _validate_response_tool_calls(resp_obj, body_json)
            if findings:
                completion_id = resp_obj.get("id") or f"proxy-{uuid.uuid4().hex[:12]}"
                model_name = resp_obj.get("model") or model
                correction_msg = _format_intercept_message(findings)
                synth = _synth_correction_response(correction_msg, completion_id, model_name)
                body_text_resp = json.dumps(synth)
                body_bytes = body_text_resp.encode("utf-8")
                _save_gate(req_id, {
                    "verdict": "intercept",
                    "rule": "schema_validator" if any(f["kind"] == "invalid_args" for f in findings) else "hallucinated_tool",
                    "reason": "; ".join(f"{f['tool_name']}: {', '.join(f['errors'])}" for f in findings),
                    "details": {"findings": findings, "fixes": fixes, "replaced": True, "streaming": False},
                })
            elif fixes:
                # Autofix patched missing fields and validation now passes — emit the fixed response.
                body_text_resp = json.dumps(resp_obj)
                body_bytes = body_text_resp.encode("utf-8")
                if (load_rules_config().get("tool_args_autofix") or {}).get("action", "audit") != "silent":
                    _save_gate(req_id, {
                        "verdict": "rewrite",
                        "rule": "tool_args_autofix",
                        "reason": "; ".join(f"{f['tool_name']}: filled {', '.join(f['fixed_fields'].keys())}" for f in fixes),
                        "details": {"fixes": fixes, "streaming": False},
                    })
            # XML autofix (non-streaming).
            _xa_cfg_n = (load_rules_config().get("xml_autofix") or {})
            _xa_act = _xa_cfg_n.get("action", "audit")
            if _xa_cfg_n.get("enabled") and _xa_act != "silent":
                if _xa_act == "fix":
                    xml_applied = _xml_fix_resp_obj(resp_obj)
                    if xml_applied:
                        body_text_resp = json.dumps(resp_obj)
                        body_bytes = body_text_resp.encode("utf-8")
                        _save_gate(req_id, {
                            "verdict": "rewrite",
                            "rule": "xml_autofix",
                            "reason": f"{len(xml_applied)} XML fix(es): " + ", ".join(
                                f"{f['kind']}" + (f"({f.get('count')})" if f.get('count') else "")
                                for f in xml_applied
                            ),
                            "details": {"fixes": xml_applied, "streaming": False},
                        })
                else:
                    xml_findings_n = _xml_detect_errors(_extract_assistant_text(resp_obj))
                    if xml_findings_n:
                        _kinds_n: dict = {}
                        for f in xml_findings_n:
                            _kinds_n[f["kind"]] = _kinds_n.get(f["kind"], 0) + 1
                        _save_gate(req_id, {
                            "verdict": "warn",
                            "rule": "xml_autofix",
                            "reason": f"{len(xml_findings_n)} XML issue(s): " + ", ".join(f"{k}={v}" for k, v in _kinds_n.items()),
                            "details": {"findings": xml_findings_n[:50], "streaming": False},
                        })

    # Proxy-tool execution loop (non-streaming): if the model called a proxy-injected tool,
    # run it, append a tool_result, and re-call upstream until done (capped at max_iterations).
    if tool_injection_active and 200 <= upstream_resp.status_code < 300:
        try:
            cur_obj = json.loads(body_text_resp)
        except (json.JSONDecodeError, TypeError):
            cur_obj = None
        if isinstance(cur_obj, dict):
            final_obj, tool_iters = await _proxy_tool_loop(cur_obj, body_json)
            if tool_iters > 0 and isinstance(final_obj, dict):
                body_text_resp = json.dumps(final_obj)
                body_bytes = body_text_resp.encode("utf-8")
                _save_gate(req_id, {
                    "verdict": "intercept",
                    "rule": "tool_injector",
                    "reason": f"executed {tool_iters} proxy-tool round(s)",
                    "details": {"iterations": tool_iters, "streaming": False},
                })

    # Protocol bridge (non-streaming): translate the OpenAI response back to Anthropic shape
    # before returning to the client. We store the translated form so audit/UI parsers see what
    # the client actually received.
    if bridge_active and 200 <= upstream_resp.status_code < 300:
        try:
            o_obj = json.loads(body_text_resp)
        except (json.JSONDecodeError, TypeError):
            o_obj = None
        if isinstance(o_obj, dict):
            anthropic_obj = _openai_to_anthropic_response(o_obj, fallback_model=bridge_original_model)
            body_text_resp = json.dumps(anthropic_obj)
            body_bytes = body_text_resp.encode("utf-8")

    # preserve_response_model_name: rewrite the body's `model` field back to what the
    # client originally requested. Off by default; flips to "hide the rewrite" mode.
    if _restore_model_name and body_bytes:
        try:
            body_bytes = _override_model_in_text(
                body_bytes.decode("utf-8", errors="replace"), _restore_model_name
            ).encode("utf-8")
        except Exception:
            pass

    _save_finish(req_id, upstream_resp.status_code, resp_headers_full, body_text_resp, None, elapsed, None)

    return Response(
        content=body_bytes,
        status_code=upstream_resp.status_code,
        headers=out_headers,
        media_type=content_type or "application/octet-stream",
    )


# -------- Export / digest --------

_REDACT_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), "[api-key]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "[github-pat]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}"), "[github-pat]"),
    (re.compile(r"/(?:home|Users)/[^/\s\"']+"), "/[user]"),
]


def _redact(text):
    if not isinstance(text, str):
        return text
    for pat, rep in _REDACT_PATTERNS:
        text = pat.sub(rep, text)
    return text


def _truncate(text, n=500):
    if not isinstance(text, str):
        return text
    if len(text) <= n:
        return text
    return text[:n] + f"\n…[truncated {len(text)-n} chars]"


async def _gather_digest_data(samples: int = 10, since_minutes: int = 1440):
    cutoff = time.time() - since_minutes * 60
    conn = db()
    overall = dict(conn.execute(
        """SELECT COUNT(*) AS count,
                  COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                  COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                  COALESCE(SUM(total_tokens), 0) AS total_tokens,
                  COALESCE(AVG(duration_ms), 0) AS avg_ms,
                  SUM(CASE WHEN gate_verdict='block' THEN 1 ELSE 0 END) AS blocks,
                  SUM(CASE WHEN gate_verdict='warn' THEN 1 ELSE 0 END) AS warns,
                  SUM(CASE WHEN gate_verdict='rewrite' THEN 1 ELSE 0 END) AS rewrites,
                  SUM(CASE WHEN gate_verdict='intercept' THEN 1 ELSE 0 END) AS intercepts,
                  SUM(CASE WHEN gate_verdict='allow' THEN 1 ELSE 0 END) AS allows
           FROM requests WHERE ts > ?""",
        (cutoff,),
    ).fetchone())

    by_model = [dict(r) for r in conn.execute(
        """SELECT COALESCE(model,'(none)') AS model, COUNT(*) AS count,
                  COALESCE(SUM(total_tokens),0) AS total_tokens,
                  COALESCE(AVG(duration_ms),0) AS avg_ms
           FROM requests WHERE ts > ? GROUP BY model ORDER BY count DESC""",
        (cutoff,),
    ).fetchall()]

    by_client = [dict(r) for r in conn.execute(
        """SELECT COALESCE(client_ip,'(unknown)') AS client_ip, COUNT(*) AS count,
                  COALESCE(SUM(total_tokens),0) AS total_tokens
           FROM requests WHERE ts > ? GROUP BY client_ip ORDER BY count DESC LIMIT 20""",
        (cutoff,),
    ).fetchall()]

    audit_events = [dict(r) for r in conn.execute(
        """SELECT id, ts, model, path, client_ip, gate_verdict, gate_rule, gate_reason
           FROM requests
           WHERE ts > ? AND gate_verdict IN ('block','warn','intercept','rewrite')
           ORDER BY ts DESC LIMIT 50""",
        (cutoff,),
    ).fetchall()]

    sample_rows = [dict(r) for r in conn.execute(
        """SELECT id, ts, method, path, model, client_ip, is_stream, duration_ms,
                  prompt_tokens, completion_tokens, total_tokens,
                  gate_verdict, gate_rule, gate_reason,
                  request_body, response_body, stream_chunks
           FROM requests_v
           WHERE ts > ? AND request_body IS NOT NULL
           ORDER BY ts DESC LIMIT ?""",
        (cutoff, max(0, samples)),
    ).fetchall()]
    conn.close()

    suggestions_data = await suggestions()
    rules_data = await get_rules()
    system_data = await system_now()

    return {
        "generated_at": time.time(),
        "window_minutes": since_minutes,
        "overall": overall,
        "by_model": by_model,
        "by_client": by_client,
        "audit_events": audit_events,
        "samples": sample_rows,
        "suggestions": suggestions_data.get("items") or [],
        "rules": rules_data,
        "system": system_data,
    }


def _summarize_request_body(body_text: str, redact: bool, include_bodies: bool) -> dict:
    if not body_text:
        return {}
    try:
        body = json.loads(body_text)
    except (json.JSONDecodeError, TypeError):
        return {"raw_truncated": _truncate(body_text, 200)}
    if not isinstance(body, dict):
        return {}
    msgs = body.get("messages") or []
    role_counts: dict = {}
    last_user = None
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        role_counts[role] = role_counts.get(role, 0) + 1
        if role == "user":
            c = m.get("content")
            if isinstance(c, str):
                last_user = c
    tools = []
    for t in (body.get("tools") or []):
        fn = (t.get("function") if isinstance(t, dict) else None) or t
        if isinstance(fn, dict):
            tools.append(fn.get("name"))
    if last_user and redact:
        last_user = _redact(last_user)
    if last_user and not include_bodies:
        last_user = _truncate(last_user, 400)
    return {
        "model": body.get("model"),
        "stream": bool(body.get("stream")),
        "messages_count": len(msgs),
        "role_counts": role_counts,
        "tools_defined": tools,
        "tool_choice": body.get("tool_choice"),
        "temperature": body.get("temperature"),
        "max_tokens": body.get("max_tokens"),
        "last_user_message": last_user,
    }


def _summarize_response(response_body: str, stream_chunks: str, redact: bool, include_bodies: bool) -> dict:
    out: dict = {"finish_reason": None, "tool_calls": [], "content_chars": 0, "content": None}
    if response_body:
        try:
            j = json.loads(response_body)
            for c in (j.get("choices") or []):
                if isinstance(c, dict):
                    if c.get("finish_reason"):
                        out["finish_reason"] = c["finish_reason"]
                    msg = c.get("message") or {}
                    if isinstance(msg.get("content"), str):
                        out["content_chars"] += len(msg["content"])
                        out["content"] = msg["content"]
                    for tc in (msg.get("tool_calls") or []):
                        fn = tc.get("function") or {}
                        out["tool_calls"].append({"name": fn.get("name"), "arguments": fn.get("arguments")})
        except (json.JSONDecodeError, TypeError):
            pass
    if stream_chunks:
        for line in stream_chunks.split("\n"):
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]" or not data.strip():
                continue
            try:
                j = json.loads(data)
            except json.JSONDecodeError:
                continue
            for c in (j.get("choices") or []):
                if isinstance(c, dict):
                    if c.get("finish_reason"):
                        out["finish_reason"] = c["finish_reason"]
                    delta = c.get("delta") or {}
                    if isinstance(delta.get("content"), str):
                        out["content_chars"] += len(delta["content"])
                        out["content"] = (out["content"] or "") + delta["content"]
    if out["content"] and redact:
        out["content"] = _redact(out["content"])
    if out["content"] and not include_bodies:
        out["content"] = _truncate(out["content"], 400)
    if out["tool_calls"] and not include_bodies:
        for tc in out["tool_calls"]:
            if tc.get("arguments"):
                tc["arguments"] = _truncate(tc["arguments"], 300)
    return out


def _render_digest_markdown(data: dict, samples: int, include_bodies: bool, redact: bool) -> str:
    lines: list[str] = []
    lines.append("# AI Proxy Traffic Digest")
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data['generated_at']))} · last {data['window_minutes']} min_")
    lines.append("")
    lines.append("## Instructions for the AI reviewer")
    lines.append("You are reviewing this AI Proxy's recent traffic to recommend rule changes. The proxy supports these rules (all configured via a single JSON object):")
    lines.append("")
    lines.append("- `loop_detector` (pre-flight): block when the same tool call signature repeats too often. Keys: `enabled`, `action` (block|warn), `max_repeats`, `window`, `tail_consecutive`.")
    lines.append("- `tool_failure_breaker` (pre-flight): block when the same tool has N consecutive error results. Keys: `enabled`, `action`, `max_errors`, `window`.")
    lines.append("- `model_router` (transform): rewrite the model name. Keys: `enabled`, `aliases` ({from: to}), `rules` ([{if: {from_model, from_client, prompt_chars_lt, prompt_chars_gt, has_tools, path_prefix}, then: <model>}]).")
    lines.append("- `ollama_options` (transform): inject generation defaults (`num_ctx`, `temperature`, `keep_alive`, etc) when the client didn't set them. Keys: `enabled`, `defaults`, `per_model`, `per_client`, `rules`.")
    lines.append("- `context_overflow_guard` (transform): estimate prompt tokens; warn / bump `num_ctx` / trim oldest messages / block when prompt exceeds the effective context window (prevents Ollama silent truncation). Keys: `enabled`, `action` (warn|bump|trim|block), `chars_per_token`, `headroom_ratio`, `max_ctx`, `bump_to`, `min_keep_messages`, `assumed_default_num_ctx`.")
    lines.append("- `compaction_nudge` (transform + intercept): when prompt size crosses `threshold_pct`% of num_ctx, nudge the client/model to compact. Strategy is per-client: `system_reminder` injects a `<system-reminder>` tag (Claude Code respects this strongly), `system_reminder_plain` injects a plain text reminder, `synthetic_response` short-circuits non-streaming requests with a synthetic assistant message. Streams fall back to `system_reminder_plain`. Adds `X-Proxy-Suggest: compact` response header on synthetic responses. Keys: `enabled`, `threshold_pct`, `chars_per_token`, `assumed_default_num_ctx`, `default_strategy`, `client_strategies` (map of client_app → strategy).")
    lines.append("- `tool_pruner` (transform): drop tool definitions the model has been offered repeatedly in this conversation but never invoked, cutting prompt tokens and reducing tool-selection noise. Keys: `enabled`, `action` (prune|warn), `min_turns_offered`, `min_history_turns`, `always_keep` (names), `max_prune_ratio`, `include_hint`.")
    lines.append("- `protocol_bridge` (transform): when an Anthropic-shape request gets routed (via `model_router`) to a non-Claude model, translate the body to OpenAI shape, send to OLLAMA_URL, and translate the response back. Lets Claude Code & Anthropic SDKs drive any OpenAI-compatible backend. Keys: `enabled`, `force` (when true, always bridge regardless of the target model — even Claude model names).")
    lines.append("- `tool_injector` (transform + post-flight): inject proxy-owned tools (memory: `remember`/`recall`/`list_memory`/`forget`; todos: `set_todos`/`get_todos`/`add_todo`/`complete_todo`) into outgoing requests, then intercept tool_use of those names, execute server-side, append tool_result, and re-call upstream so the model sees the answer and continues. Capped at `max_iterations`. Keys: `enabled`, `memory`, `todos`, `max_iterations`, `scopes`. `scopes` is a list of `{match, enabled?, memory?, todos?, max_iterations?}`; first matching scope wins. `match` accepts `ip`, `ip_cidr`, `user_agent` (substring), `client_app` (exact label).")
    lines.append("- `schema_validator` (post-flight): validate tool_call args against the request's `tools[].parameters` schema; replace bad calls with a corrective assistant message. Keys: `enabled`, `action`, `strict_types`, `reject_unknown_fields`.")
    lines.append("- `hallucinated_tool` (post-flight): same intercept for tool names not declared in the request.")
    lines.append("")
    lines.append("Recommend concrete `rules.json` edits (additions, threshold tweaks, removals). Cite the data sections below as evidence.")
    lines.append("")
    lines.append("## Current rules")
    lines.append("```json")
    lines.append(json.dumps(data["rules"].get("config") or {}, indent=2))
    lines.append("```")
    lines.append("")

    o = data["overall"] or {}
    lines.append("## Overall")
    lines.append(f"- requests: **{o.get('count', 0)}** · allows {o.get('allows',0)} · rewrites {o.get('rewrites',0)} · warns {o.get('warns',0)} · blocks {o.get('blocks',0)} · intercepts {o.get('intercepts',0)}")
    lines.append(f"- tokens: prompt {o.get('prompt_tokens',0):,} · completion {o.get('completion_tokens',0):,} · total {o.get('total_tokens',0):,}")
    lines.append(f"- avg latency: {round(o.get('avg_ms') or 0)} ms")
    lines.append("")

    lines.append("## Models")
    lines.append("| model | count | total_tokens | avg_ms |")
    lines.append("|---|---:|---:|---:|")
    for m in (data["by_model"] or [])[:25]:
        lines.append(f"| `{m['model']}` | {m['count']} | {int(m['total_tokens']):,} | {round(m['avg_ms'] or 0)} |")
    lines.append("")

    lines.append("## Clients (top 20)")
    lines.append("| client | count | total_tokens |")
    lines.append("|---|---:|---:|")
    for c in (data["by_client"] or []):
        lines.append(f"| `{c['client_ip']}` | {c['count']} | {int(c['total_tokens']):,} |")
    lines.append("")

    lines.append("## Auditor suggestions")
    sugg = data.get("suggestions") or []
    if not sugg:
        lines.append("_(none — not enough traffic or no patterns matched)_")
    for s in sugg:
        lines.append(f"### {s.get('title','(no title)')} _({s.get('severity','')})_")
        lines.append(s.get("detail", ""))
        if s.get("snippet"):
            lines.append("```json")
            lines.append(json.dumps(s["snippet"], indent=2))
            lines.append("```")
    lines.append("")

    lines.append("## Recent gate events")
    if not data["audit_events"]:
        lines.append("_(none)_")
    else:
        lines.append("| ts | verdict | rule | model | client | reason |")
        lines.append("|---|---|---|---|---|---|")
        for e in data["audit_events"][:30]:
            ts = time.strftime("%H:%M:%S", time.localtime(e["ts"])) if e.get("ts") else ""
            reason = (e.get("gate_reason") or "")[:140].replace("|", "\\|")
            lines.append(f"| {ts} | {e.get('gate_verdict','')} | `{e.get('gate_rule') or ''}` | `{e.get('model') or ''}` | `{e.get('client_ip') or ''}` | {reason} |")
    lines.append("")

    lines.append(f"## Sample requests (n={len(data['samples'])})")
    for i, r in enumerate(data["samples"], 1):
        lines.append(f"### Sample {i} · `{r.get('id','')[:8]}` · `{r.get('model') or ''}` · {round(r.get('duration_ms') or 0)} ms")
        if r.get("gate_verdict") and r["gate_verdict"] != "allow":
            lines.append(f"- gate: **{r['gate_verdict']}** · `{r.get('gate_rule','')}` — {r.get('gate_reason','')}")
        req_summary = _summarize_request_body(r.get("request_body", ""), redact, include_bodies)
        lines.append("**Request**")
        lines.append("```json")
        lines.append(json.dumps(req_summary, indent=2))
        lines.append("```")
        resp_summary = _summarize_response(r.get("response_body") or "", r.get("stream_chunks") or "", redact, include_bodies)
        lines.append("**Response**")
        lines.append("```json")
        lines.append(json.dumps(resp_summary, indent=2))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def _render_ndjson(rows: list[dict], redact: bool) -> str:
    out_lines = []
    for r in rows:
        rec = {
            "id": r.get("id"),
            "ts": r.get("ts"),
            "method": r.get("method"),
            "path": r.get("path"),
            "model": r.get("model"),
            "client_ip": r.get("client_ip"),
            "is_stream": bool(r.get("is_stream")),
            "duration_ms": r.get("duration_ms"),
            "prompt_tokens": r.get("prompt_tokens"),
            "completion_tokens": r.get("completion_tokens"),
            "total_tokens": r.get("total_tokens"),
            "gate_verdict": r.get("gate_verdict"),
            "gate_rule": r.get("gate_rule"),
            "gate_reason": r.get("gate_reason"),
            "request_summary": _summarize_request_body(r.get("request_body", "") or "", redact, False),
            "response_summary": _summarize_response(r.get("response_body") or "", r.get("stream_chunks") or "", redact, False),
        }
        out_lines.append(json.dumps(rec))
    return "\n".join(out_lines)


@app.get("/__proxy/api/export")
async def export_data(
    request: Request,
    format: str = "markdown",
    samples: int = 10,
    minutes: int = 1440,
    include_bodies: bool = False,
    redact: bool = True,
):
    # The export aggregates traffic across all clients; when PII redaction is on, only
    # loopback viewers may export to avoid leaking other clients' bodies.
    if REDACT_PII_ENABLED:
        viewer = _client_ip(request)
        try:
            if not (viewer and ipaddress.ip_address(viewer).is_loopback):
                return JSONResponse(
                    {"error": "PII redaction is enabled; export is restricted to loopback viewers"},
                    status_code=403,
                )
        except (ValueError, TypeError):
            return JSONResponse({"error": "PII redaction is enabled; export restricted"}, status_code=403)
    samples = max(0, min(int(samples), 100))
    minutes = max(1, min(int(minutes), 30 * 1440))
    data = await _gather_digest_data(samples=samples, since_minutes=minutes)
    if format == "json":
        return JSONResponse(data)
    if format == "ndjson":
        text = _render_ndjson(data["samples"], redact=redact)
        return Response(content=text, media_type="application/x-ndjson")
    text = _render_digest_markdown(data, samples=samples, include_bodies=include_bodies, redact=redact)
    return Response(content=text, media_type="text/markdown; charset=utf-8")


# -------- MCP server (Streamable HTTP transport) --------

MCP_TOOLS: list[dict] = []
MCP_HANDLERS: dict = {}


def mcp_tool(name: str, description: str, schema: dict, write: bool = False):
    """Register an MCP tool. write=True tools are gated by MCP_ALLOW_WRITE."""
    def decorate(fn):
        MCP_TOOLS.append({"name": name, "description": description, "inputSchema": schema, "_write": write})
        MCP_HANDLERS[name] = fn
        return fn
    return decorate


def _mcp_text_result(payload) -> dict:
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2, default=str)
    return {"content": [{"type": "text", "text": text}]}


@mcp_tool(
    "list_recent_requests",
    "List recent proxied requests with summary fields (id, ts, model, status, tokens, latency, gate verdict). Use this to scan traffic.",
    {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500},
            "since_minutes": {"type": "integer", "default": 1440, "minimum": 1, "maximum": 43200},
            "model": {"type": "string", "description": "Filter by model name."},
            "verdict": {"type": "string", "enum": ["allow", "block", "warn", "rewrite", "intercept"]},
        },
    },
)
async def _mcp_list_recent(args: dict, request: Request | None = None):
    viewer = _client_ip(request) if request else None
    limit = max(1, min(int(args.get("limit", 50)), 500))
    cutoff = time.time() - max(1, int(args.get("since_minutes", 1440))) * 60
    where = ["ts > ?"]
    params: list = [cutoff]
    if args.get("model"):
        where.append("model = ?")
        params.append(args["model"])
    if args.get("verdict"):
        where.append("gate_verdict = ?")
        params.append(args["verdict"])
    sql = f"""SELECT id, ts, model, path, client_ip, is_stream, duration_ms,
                     prompt_tokens, completion_tokens, total_tokens,
                     gate_verdict, gate_rule, gate_reason
              FROM requests WHERE {' AND '.join(where)}
              ORDER BY ts DESC LIMIT ?"""
    params.append(limit)
    conn = db()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return _mcp_text_result([_redact_row(dict(r), viewer) for r in rows])


@mcp_tool(
    "get_request_detail",
    "Get the full record for a request, including parsed request/response, downsize heuristic score, and gate details.",
    {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
)
async def _mcp_get_request(args: dict, request: Request | None = None):
    return _mcp_text_result(await get_request(args["id"], request))


@mcp_tool(
    "list_conversations",
    "List recent conversations (groups of related requests sharing system+first-user prefix).",
    {"type": "object", "properties": {"limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 200}}},
)
async def _mcp_list_convs(args: dict, request: Request | None = None):
    return _mcp_text_result(await list_conversations(request, int(args.get("limit", 50))))


@mcp_tool(
    "get_conversation",
    "Return the full turn-by-turn timeline of a conversation by id.",
    {"type": "object", "properties": {"conversation_id": {"type": "string"}}, "required": ["conversation_id"]},
)
async def _mcp_get_conv(args: dict, request: Request | None = None):
    return _mcp_text_result(await get_conversation(args["conversation_id"], request))


@mcp_tool(
    "get_stats",
    "Aggregate traffic stats: per-model, per-client, per-tool counts and tokens.",
    {"type": "object", "properties": {}},
)
async def _mcp_stats(args: dict, request: Request | None = None):
    return _mcp_text_result(await stats())


@mcp_tool(
    "get_audit",
    "Recent gate events (block/warn/intercept/rewrite). Use to find failure patterns.",
    {"type": "object", "properties": {"limit": {"type": "integer", "default": 100}, "include_allow": {"type": "boolean", "default": False}}},
)
async def _mcp_audit(args: dict, request: Request | None = None):
    return _mcp_text_result(await audit(request, int(args.get("limit", 100)), bool(args.get("include_allow", False))))


@mcp_tool(
    "get_suggestions",
    "Auditor's automated config recommendations based on recent traffic patterns.",
    {"type": "object", "properties": {}},
)
async def _mcp_sugg(args: dict, request: Request | None = None):
    return _mcp_text_result(await suggestions())


@mcp_tool(
    "get_rules",
    "Get current rules config (effective + stored override + defaults + source).",
    {"type": "object", "properties": {}},
)
async def _mcp_get_rules(args: dict, request: Request | None = None):
    return _mcp_text_result(await get_rules())


@mcp_tool(
    "get_system_metrics",
    "Latest system snapshot: CPU, memory, GPU, loaded models on Ollama and LM Studio, and Ollama server config (env vars).",
    {"type": "object", "properties": {}},
)
async def _mcp_sys(args: dict, request: Request | None = None):
    return _mcp_text_result(await system_now())


@mcp_tool(
    "export_digest",
    "Build a markdown digest summarizing recent traffic, current rules, suggestions, audit events, and sample requests. Use this to assemble context before recommending rule changes.",
    {
        "type": "object",
        "properties": {
            "samples": {"type": "integer", "default": 10, "minimum": 0, "maximum": 50},
            "minutes": {"type": "integer", "default": 1440, "minimum": 1, "maximum": 43200},
            "include_bodies": {"type": "boolean", "default": False, "description": "Include longer prompt/response excerpts (vs 400-char truncation)."},
            "redact": {"type": "boolean", "default": True, "description": "Strip emails / API keys / user paths from sample text."},
        },
    },
)
async def _mcp_export(args: dict, request: Request | None = None):
    # The export aggregates traffic across all clients; under PII redaction, restrict to loopback.
    if REDACT_PII_ENABLED:
        viewer = _client_ip(request) if request else None
        try:
            if not (viewer and ipaddress.ip_address(viewer).is_loopback):
                return _mcp_text_result({"error": "PII redaction is enabled; export_digest is restricted to loopback callers"})
        except (ValueError, TypeError):
            return _mcp_text_result({"error": "PII redaction is enabled; export_digest restricted"})
    data = await _gather_digest_data(
        samples=max(0, min(int(args.get("samples", 10)), 50)),
        since_minutes=max(1, min(int(args.get("minutes", 1440)), 43200)),
    )
    md = _render_digest_markdown(data, samples=int(args.get("samples", 10)),
                                  include_bodies=bool(args.get("include_bodies", False)),
                                  redact=bool(args.get("redact", True)))
    return _mcp_text_result(md)


@mcp_tool(
    "update_rules",
    "Replace the stored rules config. Top-level must be a JSON object. Takes effect on the next request. WRITE TOOL — gated by MCP_ALLOW_WRITE.",
    {"type": "object", "properties": {"rules": {"type": "object", "description": "Full rules config object."}}, "required": ["rules"]},
    write=True,
)
async def _mcp_update_rules(args: dict, request: Request | None = None):
    rules = args.get("rules")
    if not isinstance(rules, dict):
        raise ValueError("rules must be a JSON object")
    set_setting("rules", json.dumps(rules, indent=2))
    return _mcp_text_result({"ok": True, "config": load_rules_config()})


def _mcp_check_auth(request: Request) -> bool:
    if not MCP_API_KEY:
        return True
    auth = request.headers.get("authorization", "")
    return auth == f"Bearer {MCP_API_KEY}"


@app.post("/__proxy/mcp")
async def mcp_endpoint(request: Request):
    if not _mcp_check_auth(request):
        return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32001, "message": "unauthorized"}}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}, status_code=400)

    method = body.get("method")
    rid = body.get("id")
    params = body.get("params") or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": rid,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "ai-proxy", "version": "1.0"},
                "capabilities": {"tools": {"listChanged": False}},
            },
        }
    if method == "notifications/initialized":
        return Response(status_code=204)
    if method == "tools/list":
        # Hide write tools when write isn't allowed.
        visible = [t for t in MCP_TOOLS if MCP_ALLOW_WRITE or not t.get("_write")]
        # Strip our internal _write key from the response.
        clean = [{k: v for k, v in t.items() if not k.startswith("_")} for t in visible]
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": clean}}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        tool = next((t for t in MCP_TOOLS if t["name"] == name), None)
        if not tool:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown tool: {name}"}}
        if tool.get("_write") and not MCP_ALLOW_WRITE:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32001, "message": "write tools disabled (set MCP_ALLOW_WRITE=true)"}}
        handler = MCP_HANDLERS[name]
        try:
            result = await handler(args, request)
            return {"jsonrpc": "2.0", "id": rid, "result": result}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": rid, "result": {"isError": True, "content": [{"type": "text", "text": f"error: {e!r}"}]}}

    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"method not found: {method}"}}


@app.get("/__proxy/mcp")
async def mcp_get(request: Request):
    """Some MCP clients probe with GET. Return the same listing as tools/list when authorized."""
    if not _mcp_check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    visible = [{k: v for k, v in t.items() if not k.startswith("_")} for t in MCP_TOOLS if MCP_ALLOW_WRITE or not t.get("_write")]
    return {
        "name": "ai-proxy",
        "version": "1.0",
        "transport": "streamable-http",
        "tools": visible,
        "write_enabled": MCP_ALLOW_WRITE,
        "auth_required": bool(MCP_API_KEY),
    }


# Routes registered after the catch-all `/{full_path:path}` would otherwise be shadowed by it.
# Move the catch-all to the end of the route list so all specific routes win.
def _promote_catchall_to_end():
    catch_all = None
    for r in list(app.router.routes):
        if getattr(r, "path", None) == "/{full_path:path}":
            catch_all = r
            break
    if catch_all is not None:
        app.router.routes.remove(catch_all)
        app.router.routes.append(catch_all)


_promote_catchall_to_end()


def main():
    """Console entry point (``ai-proxy``). Boots the proxy, optionally with a second
    HTTPS listener sharing the same app/lifespan."""
    import sys
    import uvicorn

    # Banner/log lines contain non-ASCII (e.g. the → arrow). On Windows a redirected or
    # piped stdout defaults to cp1252 and would raise UnicodeEncodeError before we bind.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print(f"AI Proxy → forwarding to {OLLAMA_URL}")
    print(f"Listening on http://{PROXY_HOST}:{PROXY_PORT}")
    print(f"UI: http://{PROXY_HOST}:{PROXY_PORT}/__proxy/")

    # Optional HTTPS listener that runs concurrently with HTTP, sharing the same FastAPI app
    # and lifespan state. Enabled when PROXY_SSL_CERT + PROXY_SSL_KEY are set AND
    # PROXY_HTTPS_PORT > 0. Useful for serving the phone PWA over HTTPS so iOS Safari will
    # honor "Add to Home Screen", microphone permissions, service workers, etc.
    ssl_cert = os.environ.get("PROXY_SSL_CERT", "").strip()
    ssl_key = os.environ.get("PROXY_SSL_KEY", "").strip()
    ssl_key_pw = os.environ.get("PROXY_SSL_KEY_PASSWORD") or None
    try:
        https_port = int(os.environ.get("PROXY_HTTPS_PORT", "0") or "0")
    except ValueError:
        https_port = 0
    https_enabled = bool(ssl_cert and ssl_key and https_port > 0)

    if not https_enabled:
        uvicorn.run("ai_proxy.proxy:app", host=PROXY_HOST, port=PROXY_PORT, reload=False, timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S or None)
    else:
        if not Path(ssl_cert).exists():
            print(f"WARNING: PROXY_SSL_CERT does not exist at {ssl_cert!r}; HTTPS disabled.")
            uvicorn.run("ai_proxy.proxy:app", host=PROXY_HOST, port=PROXY_PORT, reload=False, timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S or None)
        elif not Path(ssl_key).exists():
            print(f"WARNING: PROXY_SSL_KEY does not exist at {ssl_key!r}; HTTPS disabled.")
            uvicorn.run("ai_proxy.proxy:app", host=PROXY_HOST, port=PROXY_PORT, reload=False, timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S or None)
        else:
            print(f"Also listening on https://{PROXY_HOST}:{https_port}")
            print(f"HTTPS UI: https://{PROXY_HOST}:{https_port}/__proxy/")
            print(f"  cert: {ssl_cert}")
            print(f"  key:  {ssl_key}")

            async def _serve_both():
                # Single FastAPI app, two listeners. The HTTP server owns the lifespan
                # (init_db, app.state.client setup); the HTTPS server runs with
                # lifespan="off" and shares the same in-process state. Requests to either
                # listener hit the same handlers and the same SQLite database.
                http_cfg = uvicorn.Config(
                    "ai_proxy.proxy:app", host=PROXY_HOST, port=PROXY_PORT,
                    log_level="info", timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S or None,
                )
                https_cfg = uvicorn.Config(
                    "ai_proxy.proxy:app", host=PROXY_HOST, port=https_port,
                    ssl_certfile=ssl_cert,
                    ssl_keyfile=ssl_key,
                    ssl_keyfile_password=ssl_key_pw,
                    lifespan="off",
                    log_level="info", timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S or None,
                )
                http_server = uvicorn.Server(http_cfg)
                https_server = uvicorn.Server(https_cfg)
                http_task = asyncio.create_task(http_server.serve())
                # Wait for HTTP server lifespan startup to complete before opening HTTPS,
                # so the shared httpx client / DB are ready when HTTPS requests arrive.
                for _ in range(50):
                    await asyncio.sleep(0.1)
                    if getattr(http_server, "started", False):
                        break
                https_task = asyncio.create_task(https_server.serve())
                try:
                    await asyncio.gather(http_task, https_task)
                except (KeyboardInterrupt, asyncio.CancelledError):
                    pass

            try:
                asyncio.run(_serve_both())
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
