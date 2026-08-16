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
import tempfile
import collections
import gzip
import ssl
import struct
import threading
import ipaddress
import math
import datetime
from contextlib import asynccontextmanager
from html.parser import HTMLParser
from pathlib import Path

import httpx
from fastapi import FastAPI, Request

try:
    from . import bench_report as _bench_report_mod
except ImportError:
    import bench_report as _bench_report_mod  # flat-script launch
try:
    from . import bench_graders as _bench_graders_mod
except ImportError:
    import bench_graders as _bench_graders_mod  # flat-script launch
try:
    from . import bench_suites as _bench_suites_mod
except ImportError:
    import bench_suites as _bench_suites_mod  # flat-script launch
try:
    from ._version import __version__
except ImportError:
    # Deployed as a flat script (scp proxy.py + systemd ExecStart python proxy.py) rather
    # than as the installed package. The version string is cosmetic; refusing to boot the
    # proxy over it took production down in a restart loop.
    __version__ = "0.0.0+standalone"
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
# Artifact capture records which files, URLs and directories a model touched via tool calls —
# i.e. the paths of the user's own source tree. That's more than some deployments want stored,
# so it's a single switch: off means nothing is extracted and no artifact rows are written.
ARTIFACTS_ENABLED = os.environ.get("PROXY_ARTIFACTS", "1").lower() in ("1", "true", "yes", "on")
try:
    REDACT_SUBNET_BITS_V4 = int(os.environ.get("PROXY_REDACT_SUBNET_BITS", "24") or 24)
except ValueError:
    REDACT_SUBNET_BITS_V4 = 24
REDACT_PLACEHOLDER = "[REDACTED — PII hidden: viewer IP not on same subnet as originator]"
# Admin IPs bypass PII redaction entirely (see everything, regardless of subnet). Comma-separated.
ADMIN_IPS = {ip.strip() for ip in os.environ.get("PROXY_ADMIN_IPS", "").split(",") if ip.strip()}
LMSTUDIO_URL = os.environ.get("LMSTUDIO_URL", "http://localhost:1234").rstrip("/")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8001").rstrip("/")
# llama.cpp's own server. Its own slot rather than borrowing LM Studio's port: both speak the
# same OpenAI shape, so squatting works — and then every log row, bench result and report says
# "lmstudio" for numbers that came from llama.cpp. Mislabelled measurements are worse than none.
LLAMACPP_URL = os.environ.get("LLAMACPP_URL", "http://localhost:8080").rstrip("/")
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
# How long the HTTPS listener waits for the HTTP listener's lifespan (init_db, migrations,
# backfills, shared httpx client) before giving up. Generous by default: a slow start is a
# reason to keep waiting, not a reason to start serving against half-built state.
STARTUP_TIMEOUT_S = float(os.environ.get("PROXY_STARTUP_TIMEOUT_S", "120") or 120)
_last_request_prune = 0.0
_last_archive_sweep = 0.0
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

# In-flight upstream responses, keyed by req_id. Populated when client.send() returns and
# cleared in a finally block when the response is fully consumed. The cancel endpoint
# looks up the entry and calls .aclose() on it to drop the upstream connection mid-stream
# (which signals Ollama to abort generation).
#   {req_id: {ts, upstream_resp, cancel_evt}}
_INFLIGHT_REQUESTS: dict = {}

# Upstream responses whose request row has already been finalized but whose socket was never
# closed. _save_finish is sync (it runs in the threadpool), so it cannot await aclose() itself —
# and popping the registry entry throws away the ONLY reference to the response, after which the
# connection can never be returned to the pool and nothing can even see that it is held. That is
# how 101 sockets to vLLM accumulated unnoticed and wedged every proxied request on 2026-08-08.
# Hand them here instead; the zombie killer drains this on its next tick.
_ORPHANED_RESPONSES: list = []


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
-- Shared working notes. Deliberately not scoped to a conversation, a client or a subnet: the
-- point is a board every machine on the network sees the same view of. Unlike request bodies,
-- nothing here arrives from a third party, so the PII gate that redacts cross-subnet viewers
-- would only hide the notes from the people who wrote them.
CREATE TABLE IF NOT EXISTS scratchboard (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    author TEXT,
    text TEXT NOT NULL,
    client_ip TEXT,
    pinned INTEGER DEFAULT 0,
    -- Replies hang off a root note. One level only: a note and the answers to it. Deeper
    -- nesting turns a board into a forum, and the thing being replied to here is usually a
    -- question with an answer — "run this command" / "here is what it printed".
    parent_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_scratchboard_ts ON scratchboard(ts DESC);
-- Attachments live in the database rather than on disk. A board that is 500 notes deep with a
-- hard per-file cap cannot grow the way request bodies did, and keeping the bytes in SQLite
-- means deletes cascade and there are no filesystem paths to get wrong — which, given the week
-- this file has had, is worth something on its own.
CREATE TABLE IF NOT EXISTS scratchboard_files (
    id TEXT PRIMARY KEY,
    note_id TEXT NOT NULL,
    ts REAL NOT NULL,
    name TEXT NOT NULL,
    mime TEXT,
    size INTEGER,
    data BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scratchboard_files_note ON scratchboard_files(note_id);
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

CREATE TABLE IF NOT EXISTS conversation_labels (
    conversation_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    updated_ts REAL
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
    creator_ip TEXT,
    parent_id TEXT,                      -- set on children of a matrix suite
    axes_json TEXT,                      -- the axis values this cell represents
    env_json TEXT,                       -- machine/engine snapshot taken at run time
    label TEXT,                           -- human-readable cell label
    phase TEXT                            -- what it is doing between measurements
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

-- Artifact tracking: one row per distinct file/url/image touch extracted from tool calls in the
-- traffic. Metadata only (no content) — the `sig` dedups the same action re-appearing across a
-- conversation's resent history, and `request_id` links back to the request that shows the content.
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sig TEXT UNIQUE,                    -- sha256(conv + tool + op + path + content_hash): dedup key
    ts REAL NOT NULL,
    request_id TEXT,                    -- the request this was first seen in (for the content view)
    conversation_id TEXT,
    client_ip TEXT,                     -- originator, for PII/subnet gating
    kind TEXT,                          -- file | url | image | skill | dir
    op TEXT,                            -- read | write | edit | create | fetch | search | list | vision
    path TEXT,                          -- file path or URL (indexed)
    tool_name TEXT,
    size_bytes INTEGER,                 -- of the content arg, if any
    content_hash TEXT,                  -- sha256(content) prefix, if the tool carried content
    meta_json TEXT
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
CREATE INDEX IF NOT EXISTS idx_artifacts_path ON artifacts(path, ts);
CREATE INDEX IF NOT EXISTS idx_artifacts_conv ON artifacts(conversation_id, ts);
CREATE INDEX IF NOT EXISTS idx_artifacts_ts ON artifacts(ts DESC);
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
    # Which surface an agentic client was driven from (discord / cron / cli). Detected at
    # ingest from system-prompt fingerprints — see _detect_surface. NULL when unknown.
    "ALTER TABLE requests ADD COLUMN surface TEXT",
    # A non-reversible id for the credential the caller presented (sha256 prefix + last 4).
    # Distinguishes callers that share an IP and a client label without putting a usable
    # secret in a file that gets copied, archived and shared. See _credential_fingerprint.
    "ALTER TABLE requests ADD COLUMN api_key_fp TEXT",
    "ALTER TABLE requests ADD COLUMN shadow_of TEXT",
    "ALTER TABLE proxy_tasks ADD COLUMN creator_ip TEXT",
    "ALTER TABLE requests ADD COLUMN proxy_tool_log TEXT",
    "ALTER TABLE system_metrics ADD COLUMN comfyui_json TEXT",
    "ALTER TABLE system_metrics ADD COLUMN vllm_json TEXT",
    "ALTER TABLE system_metrics ADD COLUMN llamacpp_json TEXT",
    # One blob keyed by backend name, so adding a backend is a registry entry rather than
    # a migration plus four edits to the INSERT — the shape that silently dropped the
    # llamacpp value and stopped telemetry dead for an hour.
    "ALTER TABLE system_metrics ADD COLUMN backends_json TEXT",
    "ALTER TABLE requests ADD COLUMN ttft_ms REAL",
    "ALTER TABLE requests ADD COLUMN upstream TEXT",
    # Chars-based estimate of the prompt token count, stored at request time. Compared against
    # the evaluated prompt_tokens (what the upstream actually prefilled) to derive a cache
    # hit/miss verdict cheaply in the requests list — no body re-parsing needed.
    "ALTER TABLE requests ADD COLUMN est_prompt_tokens INTEGER",
    # 1 if the request carried image/non-text content, so the list can badge it without
    # re-parsing the (large) body per row.
    "ALTER TABLE requests ADD COLUMN has_images INTEGER",
    # Tool names invoked in the response, as a JSON array, extracted once at write time.
    # Counting them used to mean re-reading 500 response bodies out of the blob table on every
    # stats call — 19-27 MB of scattered reads, measured at 5.1s cold against 0.001s for the
    # same rows without bodies. The response is already in memory when the row is finalised.
    "ALTER TABLE requests ADD COLUMN tool_calls TEXT",
    # Bench suites: a matrix submission creates one parent row plus a child row per cell.
    # env_json snapshots the machine/engine state at run time so a result stays interpretable
    # weeks later (which quant, which context length, how much VRAM was free).
    "ALTER TABLE bench_runs ADD COLUMN parent_id TEXT",
    "ALTER TABLE bench_runs ADD COLUMN axes_json TEXT",
    "ALTER TABLE bench_runs ADD COLUMN env_json TEXT",
    "ALTER TABLE bench_runs ADD COLUMN label TEXT",
    # What the run is doing when it is not measuring. Stopping a backend, loading 90 GB of
    # weights and waiting for a server to answer take minutes each, and all of them looked
    # identical to a hang: a counter that does not move and nothing saying why.
    "ALTER TABLE bench_runs ADD COLUMN phase TEXT",
    # The scratchboard shipped without replies. Databases created since then already have the
    # column from SCHEMA; this is for the ones that do not.
    "ALTER TABLE scratchboard ADD COLUMN parent_id TEXT",
    # Thinking, recorded at write time. It was only ever knowable by parsing the stored body
    # in the browser, so nothing could filter or count by it — and the stored body is the
    # client's original, taken before the model quirks run, so it cannot show what the proxy
    # itself decided. Three separate values because they answer three different questions:
    # what the client asked for, what actually happened, and who settled it.
    "ALTER TABLE requests ADD COLUMN reasoning_effort TEXT",
    "ALTER TABLE requests ADD COLUMN thinking TEXT",
    "ALTER TABLE requests ADD COLUMN thinking_src TEXT",
    # NOTE: images_data (full-fidelity image payloads) now lives in the request_blobs side
    # table, not `requests`. Do NOT re-add it here — the blob-split migration DROPs it from
    # `requests`, and re-adding would collide with request_blobs.images_data in the requests_v
    # view. Fresh installs get it via the request_blobs CREATE TABLE.
]


# Cold request bodies live in a second file. It is ALWAYS attached and the requests_v view
# always spans both, whether or not anything has been moved yet — a view that references
# `arch` only sometimes would fail on every connection that happened not to attach it, and
# the failure would be a query error rather than a missing row. An empty archive costs ~12 KB.
#
# The trade this makes: `requests_v` is unusable from a plain `sqlite3 proxy.db` shell, since
# nothing is attached there. Query `requests` and `request_blobs` directly for that.
ARCHIVE_DB_PATH = os.environ.get("PROXY_ARCHIVE_DB") or (str(DB_PATH) + ".archive")
_ARCHIVE_READY = False
# Attaching the archive and building the spanning view costs ~1.3ms per connection (measured),
# and db() is called on every save and every poll. Nobody who has never archived anything
# should pay it, so connections only do the work once there is something to read: set when a
# sweep moves rows, and re-checked at startup so a restart doesn't forget.
_ARCHIVE_ACTIVE = False


def _refresh_archive_active() -> bool:
    """True when the archive holds rows, or archiving is on and will soon."""
    global _ARCHIVE_ACTIVE
    active = False
    try:
        if _archive_settings().get("enabled"):
            active = True
        elif Path(ARCHIVE_DB_PATH).exists():
            # Through an attached connection, not a direct one — see _ensure_archive_file.
            _ARCHIVE_ACTIVE = True
            conn = db()
            try:
                active = bool(conn.execute(
                    "SELECT 1 FROM arch.request_blobs LIMIT 1").fetchone())
            except sqlite3.Error:
                active = False
            finally:
                conn.close()
    except Exception:
        active = False
    _ARCHIVE_ACTIVE = active
    return active


def _ensure_archive_file():
    """Create the archive with the same blob schema, once.

    Deliberately the ONLY place that opens the archive directly. Everything else reaches it
    through ATTACH on a main connection: mixing the two access paths meant a direct writer and
    an attached writer contending for the same file, which showed up as `database is locked`
    and ten-second timeouts rather than as anything obviously wrong.
    """
    global _ARCHIVE_READY
    if _ARCHIVE_READY and Path(ARCHIVE_DB_PATH).exists():
        return
    a = sqlite3.connect(ARCHIVE_DB_PATH, timeout=10.0)
    try:
        a.execute("PRAGMA journal_mode=WAL")
        a.execute("""CREATE TABLE IF NOT EXISTS request_blobs (
                        id TEXT PRIMARY KEY, request_body TEXT, response_body TEXT,
                        stream_chunks TEXT, images_data TEXT)""")
        a.commit()
        _ARCHIVE_READY = True
    finally:
        a.close()


def db():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    attached = False
    if _ARCHIVE_ACTIVE:
        try:
            _ensure_archive_file()
            conn.execute("ATTACH DATABASE ? AS arch", (ARCHIVE_DB_PATH,))
            attached = True
        except sqlite3.Error:
            pass  # Main-only view rather than failing to open the database at all.
    # requests_v has to be a TEMP view: SQLite refuses to let a stored view reference an
    # attached database ("cannot reference objects in database arch"). Temp views are
    # per-connection, which is the same scope the attach has, so the two agree by construction
    # — and every existing caller keeps querying requests_v without knowing any of this.
    try:
        conn.execute(
            "CREATE TEMP VIEW requests_v AS SELECT r.*, "
            + ("COALESCE(b.request_body,  ab.request_body)  AS request_body, "
               "COALESCE(b.response_body, ab.response_body) AS response_body, "
               "COALESCE(b.stream_chunks, ab.stream_chunks) AS stream_chunks, "
               "COALESCE(b.images_data,   ab.images_data)   AS images_data "
               "FROM requests r LEFT JOIN request_blobs b ON b.id = r.id "
               "LEFT JOIN arch.request_blobs ab ON ab.id = r.id"
               if attached else
               "b.request_body, b.response_body, b.stream_chunks, b.images_data "
               "FROM requests r LEFT JOIN request_blobs b ON b.id = r.id"))
    except sqlite3.Error:
        pass
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
    # (Historical client_app reclassification after adding new system-prompt fingerprints is done
    # via a one-off script, not a migration — the request bodies live in request_blobs which isn't
    # cleanly queryable this early in init_db, and a failing backfill here crash-loops startup.
    # See scripts/relabel_clients.py; the forward detector labels all new traffic correctly.)
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
    # requests_v is now created per connection as a TEMP view (see db()), because it has to
    # span the attached archive and SQLite won't let a stored view do that. Drop any stored
    # copy left by an older build so the two can never disagree about what a body is.
    try:
        conn.execute("DROP VIEW IF EXISTS requests_v")
    except sqlite3.OperationalError:
        pass
    # Benches live in an in-memory asyncio task, so a restart (or a crash) strands whatever was
    # queued: the rows stay 'pending'/'running' forever with nothing left to advance them, and
    # the UI shows a sweep that will never move. Nothing can resume them, so say so plainly.
    try:
        conn.execute(
            "UPDATE bench_runs SET status='failed', finished_ts=?, "
            "error=COALESCE(error, 'interrupted — the proxy restarted while this was queued') "
            "WHERE status IN ('pending','running')",
            (time.time(),),
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
    # aiter_raw() yields whatever came off the socket, which splits SSE lines mid-event as
    # often as not. Parsing each chunk in isolation dropped every fragment that didn't happen
    # to begin with "data: ", so live text silently lost pieces of the reply. Carry the
    # trailing partial line into the next chunk and only parse what is complete.
    chunk_text = (state.get("buf") or "") + chunk_text
    lines = chunk_text.split("\n")
    # A chunk ending exactly on "\n" leaves "" here, which is the correct empty carry-over.
    state["buf"] = lines.pop() if lines else ""
    # A pathological upstream that never emits a newline must not grow this without bound.
    if len(state["buf"]) > 65536:
        state["buf"] = ""
    for line in lines:
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


# Ollama-native endpoints that are *about* models rather than inference with one. Only Ollama
# implements them, so sending one anywhere else is a guaranteed 404 — and rewriting the model
# name in a metadata query answers a question the client did not ask.
_OLLAMA_NATIVE_ONLY = ("/api/show", "/api/tags", "/api/ps", "/api/pull", "/api/push",
                       "/api/create", "/api/copy", "/api/delete", "/api/blobs", "/api/version")


def _is_ollama_native_only(path: str) -> bool:
    p = "/" + (path or "").lstrip("/")
    return any(p == n or p.startswith(n + "/") for n in _OLLAMA_NATIVE_ONLY)


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
    # Decide once whether connections need to attach the archive, so a restart doesn't forget
    # that there is something out there to read.
    _refresh_archive_active()
    _load_panic_mode()
    app.state.started_at = time.time()
    # Read/write stay unbounded — a 200k-token prefill legitimately takes minutes and cutting
    # generation off is worse than waiting. The POOL wait must NOT be unbounded, though: with
    # httpx's default 100-connection cap and no pool timeout, exhaustion presents as every
    # proxied request hanging forever, with no error, no log, and no failing health check —
    # the management routes don't share this client, so /api/health answered in 3ms while
    # nothing else was served at all. A bounded pool wait turns that silence into a 504.
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(None, pool=15.0),
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=32),
    )
    app.state.metrics_client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
    app.state.metrics_task = asyncio.create_task(_metrics_loop(app))
    app.state.task_worker = asyncio.create_task(_task_worker_loop(app))
    app.state.zombie_killer = asyncio.create_task(_inflight_zombie_killer(app))
    # A bench that stopped the daily driver and then died would leave it stopped. Restoring
    # takes minutes (vLLM reloads its weights), so it runs as a task rather than blocking boot.
    app.state.residency_restore = asyncio.create_task(_restore_pending_residency())
    # One-time, throttled, and in a thread: it reads bodies, which is the cost the tool_calls
    # column exists to keep off the request path. Daemon so it can never hold up a shutdown.
    threading.Thread(target=_backfill_tool_calls, daemon=True, name="tool-calls-backfill").start()
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


# Latest released Ollama, from GitHub, cached for an hour: the System tab polls its snapshot
# every few seconds and the answer changes a few times a month.
_OLLAMA_LATEST: dict = {"ts": 0.0, "version": None}


async def _ollama_latest_version(client: httpx.AsyncClient) -> str | None:
    now = time.time()
    if now - _OLLAMA_LATEST["ts"] < 3600:
        return _OLLAMA_LATEST["version"]
    try:
        r = await client.get("https://api.github.com/repos/ollama/ollama/releases/latest",
                             timeout=8.0)
        if r.status_code == 200:
            tag = str((r.json() or {}).get("tag_name") or "").lstrip("v") or None
            _OLLAMA_LATEST.update(ts=now, version=tag)
            return tag
    except (httpx.RequestError, ValueError):
        pass
    _OLLAMA_LATEST["ts"] = now      # a failed probe also waits an hour before retrying
    return _OLLAMA_LATEST["version"]


def _ollama_update_cmd() -> list | None:
    """The command that updates Ollama, if this deployment has set one up.

    The proxy's user cannot run the installer directly — it needs root — so the contract is a
    root-owned script the sudoers file allows without a password, e.g.:
        /usr/local/sbin/ollama-update  containing  `curl -fsSL https://ollama.com/install.sh | sh`
        sudoers: crimson ALL=(root) NOPASSWD: /usr/local/sbin/ollama-update
    Configured via model_control.ollama_update_cmd (a list); absent means the button hides.
    """
    cmd = (load_rules_config().get("model_control") or {}).get("ollama_update_cmd")
    if isinstance(cmd, list) and cmd and all(isinstance(a, str) for a in cmd):
        return cmd
    return None


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
    latest = await _ollama_latest_version(client)
    out["latest_version"] = latest
    out["update_available"] = bool(
        out["version"] and latest
        and _semver_tuple(latest) > _semver_tuple(out["version"]))
    out["update_configured"] = bool(_ollama_update_cmd())
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


def _display_model_id(mid: str | None) -> str | None:
    """llama-server can report a model's id as the full GGUF path (often under the user's home dir, sometimes
    with a `-NNNNN-of-NNNNN` shard suffix). Surface just the model NAME: the filename without the directory,
    the `.gguf` extension, and any shard suffix. The full path is kept in `model_path`, and llama.cpp routing
    (one model per process) doesn't depend on this name, so cleaning it is display-only. An already-clean id
    (no path/extension) is returned unchanged."""
    if not mid:
        return mid
    base = os.path.basename(str(mid).replace("\\", "/"))
    base = re.sub(r"-\d{4,5}-of-\d{4,5}\.gguf$", "", base, flags=re.IGNORECASE)
    if base.lower().endswith(".gguf"):
        base = base[:-5]
    return base or mid


async def _llamacpp_snapshot(client: httpx.AsyncClient) -> dict:
    """llama-server: /v1/models lists what it was launched with (one model per process), and
    /props carries the loaded context size and the model path — which is the only place the
    quantisation is visible, since the filename carries it and the API does not."""
    out = {"reachable": False, "loaded": [], "available": [], "url": LLAMACPP_URL}
    try:
        r = await client.get(f"{LLAMACPP_URL}/v1/models")
        if r.status_code != 200:
            return out
        out["reachable"] = True
        for m in (r.json().get("data") or []):
            raw_id = m.get("id")
            rec = {"id": _display_model_id(raw_id), "state": "loaded", "arch": "llama.cpp",
                   "quant": _infer_quant(raw_id)}
            out["available"].append(rec)
            out["loaded"].append(rec)
    except (httpx.RequestError, ValueError):
        return out
    try:
        r = await client.get(f"{LLAMACPP_URL}/props")
        if r.status_code == 200:
            p = r.json()
            path = (p.get("model_path") or p.get("default_generation_settings", {})
                    .get("model") or "")
            out["model_path"] = path
            out["n_ctx"] = (p.get("default_generation_settings") or {}).get("n_ctx") or p.get("n_ctx")
            out["parallel"] = p.get("total_slots")
            if path:
                q = _infer_quant(path)
                for rec in out["loaded"]:
                    rec["quant"] = rec.get("quant") or q
                    rec["max_context_length"] = out.get("n_ctx")
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


# Quantisation is the first question asked of any benchmark number, and no engine reports it as
# a field: vLLM has no model_config metric, and LM Studio only labels GGUFs. It is however almost
# always written into the checkpoint name, because that is how these builds are distributed.
_QUANT_TOKENS = (
    "NVFP4", "MXFP4", "FP8", "FP16", "BF16", "AWQ", "GPTQ", "EXL2", "W8A8", "W4A16", "INT8", "INT4",
    "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K",
    "IQ4_XS", "IQ3_XXS", "IQ2_XXS",
)


def _infer_quant(*sources) -> str | None:
    """Recover a quantisation label from a checkpoint path or launch arguments.

    Inferred, not authoritative — a checkpoint named for one quant could contain another. It is
    reported as-is rather than guessed at, so a wrong name is the publisher's, not ours.
    """
    for src in sources:
        if not src:
            continue
        text = " ".join(src) if isinstance(src, (list, tuple)) else str(src)
        upper = text.upper()
        # Longest first so Q4_K_M wins over Q4_0's prefix and NVFP4 over FP4.
        for token in sorted(_QUANT_TOKENS, key=len, reverse=True):
            if token in upper:
                return token
    return None


# Phases vLLM announces on its way up, newest-looking first. Weight loading is the long one and
# the only one that reports a fraction, which is what makes the wait bearable to watch.
_VLLM_BOOT_PHASES = (
    (re.compile(r"Loading safetensors checkpoint shards:\s*(\d+)%[^|]*\|\s*(\d+)/(\d+)"),
     lambda m: f"loading weights — shard {m.group(2)} of {m.group(3)} ({m.group(1)}%)"),
    (re.compile(r"Capturing CUDA graph"), lambda m: "capturing CUDA graphs"),
    (re.compile(r"torch\.compile|Compiling|compilation"), lambda m: "compiling the model"),
    (re.compile(r"Available KV cache memory"), lambda m: "allocating the KV cache"),
    (re.compile(r"Starting to load model"), lambda m: "starting to load the model"),
    (re.compile(r"Initializing a V1 LLM engine"), lambda m: "initialising the engine"),
)


async def _vllm_boot_state() -> dict:
    """Why vLLM isn't answering: still coming up, stopped, or not there at all.

    Only called when /v1/models has already failed, so the docker calls never touch the healthy
    path. Without this the dashboard shows the same "unreachable" for a container nine minutes
    into loading 42 GiB of weights as for one that isn't running — and the first is worth
    waiting for.
    """
    docker = _docker_bin()
    name = await _vllm_container() if docker else None
    if not name:
        return {"state": "absent"}
    code, out = await _run_cmd(
        [docker, "inspect", name, "--format", "{{.State.Running}}\t{{.State.StartedAt}}"], 15.0)
    if code != 0:
        return {"state": "unknown", "container": name}
    parts = (out.strip().split("\t") + ["", ""])[:2]
    if parts[0].strip().lower() != "true":
        stopped = {"state": "stopped", "container": name}
        try:
            for c in await _vllm_configs():
                if c.get("container") == name and c.get("model"):
                    stopped["model"] = c["model"]
                    break
        except Exception:
            pass
        return stopped
    info = {"state": "loading", "container": name, "started_at": parts[1].strip() or None}
    # "loading" without a name is half an answer: the container's own config knows which
    # model it will serve long before /v1/models can say so.
    try:
        for c in await _vllm_configs():
            if c.get("container") == name and c.get("model"):
                info["model"] = c["model"]
                break
    except Exception:
        pass
    # The tail is enough: these lines are emitted continuously while it comes up.
    code, logs = await _run_cmd([docker, "logs", "--tail", "40", name], 15.0,
                                max_chars=20000, keep_tail=True)
    if code == 0 and logs:
        for line in reversed(logs.splitlines()):
            for pat, describe in _VLLM_BOOT_PHASES:
                m = pat.search(line)
                if m:
                    info["phase"] = describe(m)
                    return info
    return info


async def _vllm_snapshot(client: httpx.AsyncClient) -> dict:
    """vLLM OpenAI-compat: /v1/models lists the served model(s) (always 'loaded' — vLLM serves
    one config at a time); /metrics (Prometheus text) has live running/waiting counts + KV usage."""
    out = {"reachable": False, "loaded": [], "available": []}
    try:
        r = await client.get(f"{VLLM_URL}/v1/models")
        if r.status_code == 200:
            out["reachable"] = True
            out["state"] = "ready"
            for m in (r.json().get("data") or []):
                # `root` is the checkpoint vLLM was launched with (e.g.
                # "saricles/Qwen3-Coder-Next-NVFP4-GB10"); the served id is usually an alias.
                rec = {"id": m.get("id"), "state": "loaded",
                       "max_context_length": m.get("max_model_len"), "arch": "vllm",
                       "root": m.get("root"), "quant": _infer_quant(m.get("root"), m.get("id"))}
                out["available"].append(rec)
                out["loaded"].append(rec)
    except (httpx.RequestError, ValueError):
        out.update(await _vllm_boot_state())
        return out
    if not out["reachable"]:
        out.update(await _vllm_boot_state())
        return out
    try:
        r = await client.get(f"{VLLM_URL}/metrics")
        if r.status_code == 200:
            for key, metric in (("running", "vllm:num_requests_running"),
                                ("waiting", "vllm:num_requests_waiting"),
                                ("kv_cache_pct", "vllm:kv_cache_usage_perc")):
                m = re.search(r"^" + re.escape(metric) + r"[^ ]*\s+([0-9.eE+-]+)", r.text, re.M)
                if m:
                    try:
                        out[key] = float(m.group(1))
                    except ValueError:
                        pass
            # cache_config_info exposes the resolved engine config as Prometheus labels.
            cc = re.search(r"^vllm:cache_config_info\{([^}]*)\}", r.text, re.M)
            if cc:
                cfg = {}
                for k, v in re.findall(r'(\w+)="([^"]*)"', cc.group(1)):
                    cfg[k] = v
                if cfg:
                    out["config"] = cfg
    except (httpx.RequestError, ValueError):
        pass
    # Best-effort: the actual launch command line (tool-call-parser, max-model-len, etc.)
    # lives in the container args, not in any HTTP endpoint. Cheap, cached, non-fatal.
    args = await asyncio.to_thread(_vllm_launch_args)
    if args:
        out["launch_args"] = args
    return out


_VLLM_ARGS_CACHE: dict = {"ts": 0.0, "args": None}


def _vllm_launch_args() -> list | None:
    """Read the vLLM container's launch args via `docker inspect` (proxy runs on the same
    host, crimson is in the docker group). Cached 60s; returns None if docker/container absent."""
    now = time.monotonic()
    if _VLLM_ARGS_CACHE["args"] is not None and now - _VLLM_ARGS_CACHE["ts"] < 60:
        return _VLLM_ARGS_CACHE["args"]
    args = None
    try:
        if shutil.which("docker"):
            _pm = re.search(r":(\d+)", VLLM_URL.rsplit("/", 1)[0])
            port = _pm.group(1) if _pm else "8001"
            ps = subprocess.run(
                ["docker", "ps", "--format", "{{.ID}}\t{{.Image}}\t{{.Ports}}"],
                capture_output=True, text=True, timeout=4)
            cid = None
            for line in ps.stdout.splitlines():
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                _id, image, ports = parts[0], parts[1], parts[2]
                if ("vllm" in image.lower()) or (f":{port}->" in ports):
                    cid = _id
                    break
            if cid:
                ins = subprocess.run(
                    ["docker", "inspect", cid, "--format", "{{json .Args}}"],
                    capture_output=True, text=True, timeout=4)
                data = json.loads(ins.stdout.strip() or "null")
                if isinstance(data, list):
                    args = [str(a) for a in data]
    except (subprocess.SubprocessError, OSError, ValueError):
        args = None
    _VLLM_ARGS_CACHE["ts"] = now
    _VLLM_ARGS_CACHE["args"] = args
    return args


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


# ---- Backend registry -------------------------------------------------------------------
# Every backend used to be wired by hand into ten separate places: the metrics gather, the
# INSERT column list, the INSERT *values* (forgetting that one killed telemetry for an hour),
# system_now, GENERATION_RATE_UPSTREAMS, _BENCH_LOAD_MODES, the bench index, the router's
# upstream table, a System-tab provider tab and its panel. Every UI bug in this area has been a
# missed site rather than a logic error, and adding a fifth backend meant finding all ten.
#
# A SideService is something the proxy can start and stop but does not send inference to
# (ComfyUI). A Provider is a SideService that also serves models. The registry is the single
# list; the fan-out sites read from it instead of repeating themselves.
#
# Deliberately delegating to the existing functions rather than reimplementing them: the point
# of this step is to change where the knowledge lives, not what it does, so the behaviour is
# provably identical while the call sites move over one at a time.


class SideService:
    """Startable, stoppable, and able to say whether it is up. No models."""

    name: str = ""
    label: str = ""
    #  "unit"   -> systemd (see model_control.services)
    #  "docker" -> a container publishing the backend's port
    #  "none"   -> not managed by the proxy
    control: str = "none"

    async def probe(self, client) -> dict:
        """Reachability plus whatever the backend reports about itself."""
        return {"reachable": False}

    async def state(self) -> dict:
        """Everything needed to render a control: running, and why not if not."""
        return {"running": False, "control": self.control}

    async def start(self, container: str | None = None) -> dict:
        """Bring it up. Returns {ok, detail}; readiness is a separate question.

        `container` names the specific docker container to start, for backends where several
        are configured for the same port. Restore paths pass the one their snapshot recorded;
        without it, discovery falls back to the first configured match."""
        return await _control_backend(self, "start", container=container)

    async def stop(self) -> dict:
        return await _control_backend(self, "stop")

    async def died(self) -> str | None:
        """Why the process is gone, if it is. None means "still up, or can't tell".

        Only ever used to stop waiting early, so an uncertain answer must be None: claiming a
        healthy backend died would abort a load that was merely slow.
        """
        if self.control == "unit":
            svc = _service_def(self.name)
            if svc:
                st = await _service_state(svc)
                # "activating" and "reloading" are still on their way up. Only a terminal state
                # means no amount of further waiting will help.
                if st.get("active") in ("failed", "inactive"):
                    _code, log = await _run_cmd(
                        _systemctl_args(svc, "status", "--no-pager", "-n", "8"),
                        15.0, max_chars=600, keep_tail=True, env=_systemctl_env(svc))
                    return f"{svc['unit']} is {st['active']}: {log.strip()[-400:]}"
        if self.control == "docker":
            container = await _vllm_container()
            if container:
                code, out = await _run_cmd(
                    [_docker_bin(), "inspect", container, "--format", "{{.State.Running}}"],
                    15.0, max_chars=100)
                if code == 0 and out.strip().lower() == "false":
                    _c, log = await _run_cmd(
                        [_docker_bin(), "logs", "--tail", "12", container],
                        20.0, max_chars=600, keep_tail=True)
                    return f"{container} is not running: {log.strip()[-400:]}"
        return None


class Provider(SideService):
    """A SideService that serves models and can be benchmarked."""

    #  "on-demand" -> the backend loads whatever is asked for (Ollama)
    #  "jit"       -> loads on first use, evicts on its own schedule (LM Studio)
    #  "fixed"     -> one model per process, chosen at launch (vLLM, llama.cpp)
    load_mode: str = "fixed"
    # Whether decode-rate statistics can be attributed to this backend. A backend left out of
    # this is not merely missing from a chart — 100% of its traffic silently vanishes from the
    # throughput figures, which is what happened to vLLM once already.
    measures_decode: bool = True
    # Whether load() needs a model name. The fixed-model backends are launched with one, so the
    # operation is "start this server", not "load that model" — a blanket name check at the
    # endpoint made starting them impossible.
    requires_model_name: bool = True

    @property
    def base_url(self) -> str:
        raise NotImplementedError

    async def ready(self, timeout_s: float = 900.0) -> bool:
        """Serving, not merely started. Both fixed-model backends take minutes to load their
        weights, and every caller that reported success on the process starting was reporting
        a lie — this is the same poll _vllm_ready and _llamacpp_ready each had separately."""
        deadline = time.time() + timeout_s
        checks = 0
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as c:
            while time.time() < deadline:
                try:
                    if (await c.get(f"{self.base_url}/v1/models")).status_code == 200:
                        return True
                except httpx.RequestError:
                    pass
                checks += 1
                # A process that has exited will never answer, and polling a corpse for the
                # whole timeout turns a fast, explicable failure — "the KV pool did not fit" —
                # into a quarter-hour of silence. Skip the first pass: a unit that has just
                # been restarted is legitimately not up yet.
                if checks > 1 and await self.died():
                    return False
                await asyncio.sleep(5.0)
        return False

    def models(self, snapshot: dict) -> list:
        """Rows for the bench index, from a probe snapshot."""
        return [m for m in (snapshot.get("available") or []) if isinstance(m, dict)]

    async def load(self, payload: dict, name: str):
        """Make this backend serve `name`, and wait until it actually does.

        The mechanisms have nothing in common — a CLI, a docker start, a systemd drop-in
        rewrite, an HTTP warm-up — so this is the one method that genuinely differs per
        backend rather than being shared. Returns the endpoint's response body directly, or a
        JSONResponse to set a status code.
        """
        return JSONResponse({"error": f"{self.name} cannot load models on request"},
                            status_code=501)

    # Whether the served context window can be changed by relaunching. vLLM bakes max_model_len
    # into the container's arguments, so changing it means recreating the container rather than
    # restarting it — a different and much less safe operation.
    resizable_context: bool = False

    async def context_window(self) -> tuple[int, int]:
        """(tokens per slot, slot count) as currently served. (0, 1) when unknown."""
        return 0, 1

    async def resize_context(self, per_slot: int, concurrency: int = 1,
                             exact: bool = False) -> dict:
        """Relaunch serving at least `per_slot` tokens, with room for `concurrency` requests.

        Returns {ok, changed, detail, previous}. `previous` is whatever must be handed back to
        restore the old window, or None when nothing was changed.
        """
        return {"ok": False, "changed": False, "previous": None,
                "detail": f"{self.name} cannot change its context window"}


class _FnProvider(Provider):
    """Adapter over the existing module-level snapshot functions.

    Each backend keeps its current probe verbatim; only the ownership moves. Replacing these
    with real implementations is the next step and can be done one at a time.
    """

    def __init__(self, name, label, probe_fn, url_fn, load_mode="fixed",
                 control="none", measures_decode=True):
        self.name, self.label = name, label
        self._probe, self._url = probe_fn, url_fn
        self.load_mode, self.control = load_mode, control
        self.measures_decode = measures_decode

    @property
    def base_url(self) -> str:
        return self._url()

    async def probe(self, client) -> dict:
        return await self._probe(client)

    async def state(self) -> dict:
        if self.control == "unit":
            svc = _service_def(self.name)
            if svc:
                st = await _service_state(svc)
                return {**st, "control": "unit"}
        if self.control == "docker":
            container = await _vllm_container()
            return {"running": None, "container": container, "control": "docker"}
        return {"running": None, "control": self.control}


class _LmStudioProvider(_FnProvider):
    async def load(self, payload: dict, name: str):
        ok, why = _lms_available()
        if not ok:
            return JSONResponse({"error": why}, status_code=501)
        args = ["load", name, "--yes"]
        # These matter for real workloads: a model loaded at the wrong context length or without
        # parallelism benchmarks as a different thing entirely. The request wins over the
        # per-box defaults in model_control.
        mc = load_rules_config().get("model_control") or {}
        ctx = payload.get("context_length") or mc.get("lmstudio_context_length")
        par = payload.get("parallel") or mc.get("lmstudio_parallel")
        gpu = payload.get("gpu") or mc.get("lmstudio_gpu")
        if ctx:
            args += ["--context-length", str(int(ctx))]
        if par:
            args += ["--parallel", str(int(par))]
        if gpu:
            args += ["--gpu", str(gpu)]
        if payload.get("ttl_s"):
            args += ["--ttl", str(int(payload["ttl_s"]))]
        t0 = time.perf_counter()
        code, out = await _lms_run(args)
        if code != 0:
            return JSONResponse({"error": out or f"lms exited {code}"}, status_code=502)
        return {"ok": True, "model": name, "upstream": "lmstudio", "output": out,
                "load_ms": round((time.perf_counter() - t0) * 1000)}


# How long a bench-driven resize waits for the server to serve again. Generous enough for a
# 90 GB model to load from disk, short enough that an over-large pool is diagnosed and rolled
# back inside a bench rather than after it.
_BENCH_RESIZE_READY_S = 900.0
# How long a bench waits for a backend it started to begin serving. Deliberately generous: the
# run has hours to spend and a stalled start is now detected by died(), not by this expiring.
_BENCH_START_READY_S = 1800.0


def _load_result(res) -> tuple[bool, str]:
    """Read a load() outcome. It returns a response body on success and a JSONResponse to set a
    status code, and callers inside the process need both read the same way."""
    if isinstance(res, JSONResponse):
        try:
            body = json.loads(bytes(res.body).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            body = {}
        return False, str(body.get("error") or f"HTTP {res.status_code}")
    d = res or {}
    return bool(d.get("ok")), str(d.get("error") or d.get("note") or d.get("detail") or "")


class _LlamaCppProvider(_FnProvider):
    requires_model_name = False
    resizable_context = True

    async def context_window(self) -> tuple[int, int]:
        snap = await _llamacpp_snapshot(app.state.metrics_client)
        return int(snap.get("n_ctx") or 0), max(1, int(snap.get("parallel") or 1))

    async def resize_context(self, per_slot: int, concurrency: int = 1,
                             exact: bool = False) -> dict:
        """`--ctx-size` is the whole KV pool, divided across `--parallel` slots, so the window a
        single request actually gets is total/parallel. That makes lowering parallel the usual
        way to buy a bigger window: one slot of 36k costs less memory than four of 32k, not
        more. Sizing the pool to the concurrency being measured rather than to whatever the
        server happens to be running is the whole point.
        """
        per_slot, want_par = int(per_slot), max(1, int(concurrency))
        cur_slot, cur_par = await self.context_window()
        fits = (cur_slot == per_slot and cur_par == want_par) if exact else (
            cur_slot >= per_slot and cur_par >= want_par)
        if fits:
            return {"ok": True, "changed": False, "previous": None,
                    "detail": f"already serving {cur_slot:,} per slot across {cur_par}",
                    "per_slot": cur_slot, "parallel": cur_par}
        if not cur_slot:
            # Without a readable current window there is no way back, and a bench that cannot
            # undo its own change to the box must not make it. This is why a sweep once left
            # llama.cpp at 291,784/1: the probe came back empty, the resize went ahead anyway,
            # and the restore had nothing to restore to.
            return {"ok": False, "changed": False, "previous": None,
                    "detail": "could not read the current context window, so a resize could "
                              "not be undone — refusing to change it"}
        prev = {"context_length": cur_slot * cur_par, "parallel": cur_par}
        # Bounded, unlike load()'s 1800s default. The usual reason a resize never becomes ready
        # is that the pool did not fit in memory, and waiting half an hour to find that out —
        # before even starting the rollback — is the whole cost of the mistake.
        ok, detail = _load_result(
            await self.load({"context_length": per_slot * want_par, "parallel": want_par,
                             "ready_timeout_s": _BENCH_RESIZE_READY_S}, ""))
        if not ok:
            # Why it died beats "did not become ready in time" — the usual answer is that the
            # KV pool did not fit, and that is actionable where a timeout is not.
            why = await self.died()
            # A server that did not come back is worse than a bench that did not run: roll the
            # drop-in back rather than leaving the box with nothing serving on port 8080.
            rolled = None
            if prev:
                rolled = _load_result(
                    await self.load(dict(prev, ready_timeout_s=_BENCH_RESIZE_READY_S), ""))[0]
            return {"ok": False, "changed": False, "previous": None, "rolled_back": rolled,
                    "detail": why or detail
                    or f"llama.cpp did not come back at {per_slot * want_par:,} context"}
        return {"ok": True, "changed": True, "previous": prev, "detail": detail,
                "per_slot": per_slot, "parallel": want_par}

    async def load(self, payload: dict, name: str):
        cfg = _llamacpp_cfg()
        svc = _service_def("llamacpp") or {"unit": str(cfg.get("unit") or "llamacpp.service"),
                                           "scope": cfg.get("scope") or "user",
                                           "name": "llamacpp"}
        ctx = int(payload.get("context_length") or payload.get("ctx") or 0)
        par = int(payload.get("parallel") or 0)
        if ctx or par:
            # Carry over whatever was not specified so changing one does not silently reset
            # the other to a default the caller never asked for.
            cur = await _llamacpp_snapshot(app.state.metrics_client)
            par = par or int(cur.get("parallel") or 1)
            ctx = ctx or int((cur.get("n_ctx") or 0) * max(1, par)) or 65536
            ok, detail = _write_llamacpp_override(ctx, par)
            if not ok:
                return JSONResponse({"error": detail}, status_code=400)
            # daemon-reload takes no unit — _systemctl_args always appends one, which
            # systemctl rejects with "Too many arguments."
            reload_args = [a for a in _systemctl_args(svc, "daemon-reload")
                           if a != svc["unit"]]
            code, out = await _run_cmd(reload_args, 30.0, env=_systemctl_env(svc))
            if code != 0:
                return JSONResponse({"error": f"daemon-reload failed: {out[:200]}"},
                                    status_code=502)
        t0 = time.perf_counter()
        code, out = await _run_cmd(_systemctl_args(svc, "restart"), 120.0,
                                   env=_systemctl_env(svc))
        if code != 0:
            return JSONResponse({"error": f"restart failed: {out[:300]}"}, status_code=502)
        # Module-level rather than self.ready() on purpose: it is the seam the tests patch, and
        # it already delegates here.
        ready = await _llamacpp_ready(float(payload.get("ready_timeout_s") or 1800))
        return {"ok": ready, "upstream": "llamacpp", "ready": ready,
                "context_length": ctx or None, "parallel": par or None,
                "load_ms": round((time.perf_counter() - t0) * 1000),
                "note": None if ready else "restarted, but the server did not answer in time"}


class _VllmProvider(_FnProvider):
    requires_model_name = False

    async def load(self, payload: dict, name: str):
        wanted = (payload.get("container") or "").strip()
        configs = await _vllm_configs()
        if wanted:
            match = next((c for c in configs if c["container"] == wanted), None)
            if not match:
                return JSONResponse(
                    {"error": f"no vLLM container named {wanted!r}",
                     "available": [c["container"] for c in configs]}, status_code=404)
            if not match["serves_port"]:
                return JSONResponse(
                    {"error": f"{wanted!r} does not publish {VLLM_URL} — starting it would not "
                              f"make it reachable through this proxy"}, status_code=409)
            container = wanted
        else:
            container = await _vllm_container()
        if not container:
            return JSONResponse(
                {"error": "no local vLLM container publishing this port was found",
                 "available": [c["container"] for c in configs]}, status_code=501)
        t0 = time.perf_counter()
        # They contend for one port, so anything else running must come down first — otherwise
        # docker start fails on the binding and the error looks unrelated.
        stopped = []
        for c in configs:
            if c["running"] and c["container"] != container:
                await _run_cmd([_docker_bin(), "stop", c["container"]], 180.0)
                stopped.append(c["container"])
        # Last-line memory guard, on the one chokepoint every vLLM start passes through
        # (auto-load, bench, UI). Priced by the utilization claim — see _vllm_boot_claim_mb
        # for why checkpoint size is the wrong number. Callers that made room first pass
        # trivially; a blind start onto a full box is refused with the arithmetic.
        claim = await _vllm_boot_claim_mb(container)
        free = _free_mem_mb()
        if claim and free and free < claim and not payload.get("force"):
            return JSONResponse(
                {"error": f"starting {container!r} will claim ~{claim / 1024:.0f} GB "
                          f"(gpu-memory-utilization × total) but only {free / 1024:.0f} GB "
                          f"is free — that combination starves the box (it has wedged sshd "
                          f"and this proxy before). Free memory first, or pass force:true.",
                 "claim_mb": round(claim), "free_mb": round(free)}, status_code=409)
        code, out = await _run_cmd([_docker_bin(), "start", container], 60.0)
        if code != 0:
            return JSONResponse({"error": out or f"docker start exited {code}"}, status_code=502)
        # Started is not ready: vLLM takes minutes to load weights and build its KV cache.
        cfg_mc = load_rules_config().get("model_control") or {}
        ready = await _vllm_ready(float(payload.get("wait_s")
                                        or cfg_mc.get("vllm_ready_timeout_s") or 420))
        return {"ok": ready, "started_container": container, "stopped_containers": stopped,
                "ready": ready, "load_ms": round((time.perf_counter() - t0) * 1000),
                "error": None if ready else "container started but the server did not become "
                                            "ready in time — it may still be loading"}


class _OllamaProvider(_FnProvider):
    async def load(self, payload: dict, name: str):
        """A zero-token generate is how you ask Ollama to make a model resident. Loading a large
        model takes tens of seconds, so this waits for it rather than returning early — the
        caller wants to know it is actually resident, not that the request was accepted."""
        keep = payload.get("keep_alive") or "30m"
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as c:
                r = await c.post(f"{OLLAMA_URL}/api/generate",
                                 json={"model": name, "prompt": "", "keep_alive": keep})
                if r.status_code >= 400:
                    return JSONResponse({"error": f"ollama said {r.status_code}: {r.text[:200]}"},
                                        status_code=502)
        except httpx.RequestError as e:
            return JSONResponse({"error": f"ollama unreachable: {e}"}, status_code=502)
        return {"ok": True, "model": name, "keep_alive": keep,
                "load_ms": round((time.perf_counter() - t0) * 1000)}


class _FnSideService(SideService):
    def __init__(self, name, label, probe_fn, control="none"):
        self.name, self.label, self._probe, self.control = name, label, probe_fn, control

    async def probe(self, client) -> dict:
        return await self._probe(client)

    async def state(self) -> dict:
        if self.control == "unit":
            svc = _service_def(self.name)
            if svc:
                return {**(await _service_state(svc)), "control": "unit"}
        return {"running": None, "control": self.control}


async def _control_backend(b, action: str, container: str | None = None) -> dict:
    """The three start/stop mechanisms, in one place.

    They were previously re-derived from the backend's name at every call site — the residency
    handshake, the load endpoint and the UI each knew that vLLM means docker and ComfyUI means
    systemd. Adding llama.cpp meant teaching all of them again.
    """
    if b.control == "unit":
        svc = _service_def(b.name)
        if not svc:
            return {"ok": False, "detail": f"{b.name} has no unit configured"}
        code, out = await _run_cmd(_systemctl_args(svc, action), 120.0,
                                   env=_systemctl_env(svc))
        return {"ok": code == 0, "detail": out[:300], "via": "systemctl"}
    if b.control == "docker":
        container = container or await _vllm_container()
        if not container:
            return {"ok": False, "detail": "no container publishing this backend's port"}
        docker_action = "start" if action == "start" else "stop"
        if docker_action == "start":
            # Same guard as the load path: every docker start is a utilization-sized claim
            # on unified memory, and a start the box cannot absorb starves sshd and this
            # proxy first — refusal is the only failure mode anyone can still see.
            claim = await _vllm_boot_claim_mb(container)
            free = _free_mem_mb()
            if claim and free and free < claim:
                return {"ok": False, "via": "memory-guard",
                        "detail": f"refused to start {container}: it will claim "
                                  f"~{claim / 1024:.0f} GB and only {free / 1024:.0f} GB is "
                                  f"free — free memory first"}
        code, out = await _run_cmd([_docker_bin(), docker_action, container], 180.0)
        return {"ok": code == 0, "detail": out[:300], "via": "docker",
                "container": container}
    return {"ok": False, "detail": f"{b.name} is not managed by the proxy", "via": "none"}


# Order matters only for display: this is the order the System tab shows backends in.
PROVIDERS: dict = {}
SIDE_SERVICES: dict = {}


def _register_backends():
    """Built once at import. Kept as a function so tests can rebuild it after patching URLs."""
    PROVIDERS.clear()
    SIDE_SERVICES.clear()
    for p in (
        _VllmProvider("vllm", "vLLM", _vllm_snapshot, lambda: VLLM_URL,
                      load_mode="fixed", control="docker"),
        _LlamaCppProvider("llamacpp", "llama.cpp", _llamacpp_snapshot, lambda: LLAMACPP_URL,
                          load_mode="fixed", control="unit"),
        _LmStudioProvider("lmstudio", "LM Studio", _lmstudio_snapshot, lambda: LMSTUDIO_URL,
                          load_mode="jit"),
        _OllamaProvider("ollama", "Ollama", _ollama_snapshot, lambda: OLLAMA_URL,
                        load_mode="on-demand"),
    ):
        PROVIDERS[p.name] = p
    for s in (_FnSideService("comfyui", "ComfyUI", _comfyui_snapshot, control="unit"),):
        SIDE_SERVICES[s.name] = s


def backend(name: str):
    return PROVIDERS.get(name) or SIDE_SERVICES.get(name)


# Built at import, immediately after the snapshot functions it references.
_register_backends()


def _host_snapshot() -> tuple:
    """CPU, load, memory and GPU — gathered together so it can be run off the event loop.

    _gpu_snapshot shells out to nvidia-smi three times with a 3-second timeout each, so this is
    up to nine seconds of blocking work. Called straight from the async collector it stopped the
    proxy *accepting* connections for that long: the dashboard looked dead, requests piled up in
    the socket's accept queue, and it cleared on its own once the box quietened down — which
    made it look like load rather than a bug. nvidia-smi is slowest exactly when a benchmark is
    hammering the GPU, so it failed when it was most likely to be watched.
    """
    return _cpu_pct(), _load_avg(), _mem_snapshot(), _gpu_snapshot()


def _write_metrics_sample(ts: float, cpu, load, mem: dict, gpus: list, backends: dict) -> None:
    """The sample write plus its retention deletes. Also blocking, also off-thread."""
    conn = db()
    conn.execute(
        """INSERT OR REPLACE INTO system_metrics
           (ts, cpu_pct, load_1m, mem_total_mb, mem_used_mb, mem_avail_mb, gpu_json,
            backends_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ts,
            cpu,
            load,
            mem.get("total_mb"),
            mem.get("used_mb"),
            mem.get("avail_mb"),
            json.dumps(gpus) if gpus else None,
            json.dumps(backends),
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
        # Archived bodies outlive their rows otherwise: nothing else references them, and the
        # archive would grow forever holding blobs for requests that no longer exist.
        try:
            conn.execute("DELETE FROM arch.request_blobs WHERE id NOT IN "
                         "(SELECT id FROM requests)")
        except sqlite3.Error:
            pass
    conn.commit()
    conn.close()
    _maybe_sweep_archive(ts)


async def _collect_once(app: FastAPI):
    # Everything blocking goes to a thread; only the backend probes, which are genuinely async
    # HTTP, run on the loop.
    cpu, load, mem, gpus = await asyncio.to_thread(_host_snapshot)
    # Probe whatever is registered. return_exceptions=True because one unreachable backend
    # must not abort the whole sample — losing every metric because a single probe raised is
    # how the collector became a single point of failure before.
    entries = list(PROVIDERS.items()) + list(SIDE_SERVICES.items())
    results = await asyncio.gather(
        *(b.probe(app.state.metrics_client) for _n, b in entries),
        return_exceptions=True,
    )
    backends = {}
    for (name, _b), res in zip(entries, results):
        if isinstance(res, BaseException):
            backends[name] = {"reachable": False, "error": f"{type(res).__name__}: {res}"}
        else:
            backends[name] = res
    await asyncio.to_thread(_write_metrics_sample, time.time(), cpu, load, mem, gpus, backends)
    # Artifact sweep: extract file/url/image touches from recent requests (off-thread so the
    # metrics loop never blocks on DB work). Dedup makes it idempotent.
    try:
        await asyncio.to_thread(_artifact_sweep)
    except Exception:
        pass


_ARCHIVE_JOB: dict = {"state": "idle"}
_ARCHIVE_LOCK = threading.Lock()


def _archive_settings() -> dict:
    cfg = {}
    try:
        cfg = load_rules_config().get("archive") or {}
    except Exception:
        pass

    def num(key, default, lo, hi):
        try:
            v = float(cfg.get(key, default))
        except (TypeError, ValueError):
            return default
        return min(hi, max(lo, v))

    return {"enabled": bool(cfg.get("enabled")),
            "after_days": num("after_days", 7, 0.04, 3650),
            "chunk": int(num("chunk", 200, 1, 5000)),
            "pause_s": num("pause_s", 0.1, 0.0, 5.0)}


def _archive_status() -> dict:
    """Sizes and how much is waiting to move — the numbers worth seeing before pressing go."""
    s = _archive_settings()
    out = {**s, "main_db_bytes": 0, "archive_db_bytes": 0,
           "pending_rows": 0, "pending_bytes": 0, "archived_rows": 0,
           "job": dict(_ARCHIVE_JOB)}
    for key, path in (("main_db_bytes", DB_PATH), ("archive_db_bytes", ARCHIVE_DB_PATH)):
        try:
            out[key] = Path(path).stat().st_size
        except OSError:
            pass
    conn = db()
    try:
        cutoff = time.time() - s["after_days"] * 86400
        r = conn.execute(
            """SELECT COUNT(*) n, COALESCE(SUM(
                   LENGTH(COALESCE(b.request_body,'')) + LENGTH(COALESCE(b.response_body,'')) +
                   LENGTH(COALESCE(b.stream_chunks,'')) + LENGTH(COALESCE(b.images_data,''))), 0) sz
               FROM requests r JOIN request_blobs b ON b.id = r.id
               WHERE r.ts < ?""", (cutoff,)).fetchone()
        out["pending_rows"], out["pending_bytes"] = r["n"], r["sz"]
    except sqlite3.Error as e:
        out["error"] = str(e)
    finally:
        conn.close()
    # Via ATTACH like everything else; force it on for this connection so the count is
    # available even before the first sweep has made archiving active.
    global _ARCHIVE_ACTIVE
    if Path(ARCHIVE_DB_PATH).exists():
        was, _ARCHIVE_ACTIVE = _ARCHIVE_ACTIVE, True
        conn = db()
        try:
            out["archived_rows"] = conn.execute(
                "SELECT COUNT(*) FROM arch.request_blobs").fetchone()[0]
        except sqlite3.Error:
            pass
        finally:
            conn.close()
            _ARCHIVE_ACTIVE = was or bool(out["archived_rows"])
    return out


def _archive_sweep(force: bool = False) -> dict:
    """Move cold blobs into the archive file, a chunk per transaction.

    Insert-then-delete inside one transaction per chunk, so a crash mid-sweep can duplicate a
    row into the archive but can never drop one from both. Interrupted work is simply picked up
    on the next sweep.
    """
    s = _archive_settings()
    if not (s["enabled"] or force):
        return {"ok": False, "reason": "archiving is disabled"}
    if not _ARCHIVE_LOCK.acquire(blocking=False):
        return {"ok": False, "reason": "a sweep is already running", "job": dict(_ARCHIVE_JOB)}
    # From here on connections must attach the archive — this sweep is about to put rows in it,
    # and every later read has to be able to see them.
    global _ARCHIVE_ACTIVE
    _ensure_archive_file()
    _ARCHIVE_ACTIVE = True
    started, moved, freed = time.time(), 0, 0
    _ARCHIVE_JOB.update({"state": "running", "started": started, "moved": 0, "error": None})
    try:
        cutoff = started - s["after_days"] * 86400
        while True:
            conn = db()
            try:
                rows = conn.execute(
                    """SELECT b.id, LENGTH(COALESCE(b.request_body,'')) +
                              LENGTH(COALESCE(b.response_body,'')) +
                              LENGTH(COALESCE(b.stream_chunks,'')) +
                              LENGTH(COALESCE(b.images_data,'')) sz
                       FROM requests r JOIN request_blobs b ON b.id = r.id
                       WHERE r.ts < ? ORDER BY r.ts LIMIT ?""",
                    (cutoff, s["chunk"])).fetchall()
                if not rows:
                    break
                ids = [r["id"] for r in rows]
                marks = ",".join("?" for _ in ids)
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    f"""INSERT OR REPLACE INTO arch.request_blobs
                        (id, request_body, response_body, stream_chunks, images_data)
                        SELECT id, request_body, response_body, stream_chunks, images_data
                        FROM main.request_blobs WHERE id IN ({marks})""", ids)
                conn.execute(f"DELETE FROM main.request_blobs WHERE id IN ({marks})", ids)
                conn.commit()
                moved += len(ids)
                freed += sum(r["sz"] or 0 for r in rows)
                _ARCHIVE_JOB["moved"] = moved
            except sqlite3.Error as e:
                _ARCHIVE_JOB["error"] = str(e)
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                break
            finally:
                conn.close()
            if s["pause_s"]:
                time.sleep(s["pause_s"])
        _ARCHIVE_JOB.update({"state": "done", "finished": time.time()})
        if moved:
            try:
                print(f"[archive] moved {moved} blob row(s), {freed / 1e9:.2f} GB, "
                      f"in {time.time() - started:.0f}s")
            except Exception:
                pass
        return {"ok": True, "moved": moved, "freed_bytes": freed,
                "elapsed_s": round(time.time() - started, 1)}
    finally:
        _ARCHIVE_LOCK.release()


def _maybe_sweep_archive(ts: float) -> None:
    """Hourly at most, and only when switched on. In a thread: it moves gigabytes, and the
    metrics loop it is called from must not wait on that."""
    global _last_archive_sweep
    if (ts - _last_archive_sweep) <= 3600:
        return
    _last_archive_sweep = ts
    try:
        if _archive_settings().get("enabled"):
            threading.Thread(target=_archive_sweep, daemon=True, name="archive-sweep").start()
    except Exception:
        pass


def _backfill_tool_calls(cutoff_days: float = 7.0, chunk: int = 50, pause_s: float = 0.15):
    """Fill `tool_calls` for rows written before it existed, once, in the background.

    Reading those bodies is exactly the expensive thing this column exists to avoid, so it is
    done here rather than on the stats path: small chunks, a pause between them, and bounded to
    rows older than this process — so it converges and never touches anything the write path
    now handles. Rows with no tool calls are marked `[]` rather than left NULL, or they would be
    re-read on every pass forever.
    """
    started = time.time()
    since = started - cutoff_days * 86400
    done = found = 0
    while True:
        conn = db()
        try:
            rows = conn.execute(
                """SELECT r.id, b.response_body, b.stream_chunks
                   FROM requests r JOIN request_blobs b ON b.id = r.id
                   WHERE r.tool_calls IS NULL AND r.ts > ? AND r.ts < ?
                     AND (b.response_body IS NOT NULL OR b.stream_chunks IS NOT NULL)
                   ORDER BY r.ts DESC LIMIT ?""", (since, started, chunk)).fetchall()
            if not rows:
                break
            for r in rows:
                try:
                    names = _extract_tool_calls(r["response_body"], r["stream_chunks"])
                except Exception:
                    names = []
                conn.execute("UPDATE requests SET tool_calls=? WHERE id=?",
                             (json.dumps(names), r["id"]))
                found += len(names)
            conn.commit()
            done += len(rows)
        except sqlite3.Error:
            break
        finally:
            conn.close()
        time.sleep(pause_s)
    if done:
        try:
            print(f"[init] backfilled tool_calls for {done} request(s), {found} call(s), "
                  f"in {time.time() - started:.0f}s")
        except Exception:
            pass
    return done


async def _metrics_loop(app: FastAPI):
    # Prime CPU sample so the first real reading is meaningful.
    _metrics_fail = 0
    _cpu_pct()
    await asyncio.sleep(1)
    while True:
        try:
            await _collect_once(app)
            _metrics_fail = 0
        except Exception as e:
            # Telemetry must never crash the proxy — but silence here once cost an hour of
            # stale data: a column added to the INSERT without its value made every tick throw,
            # and the dashboard went on serving an hour-old sample as if it were current. Say
            # something, at a rate that cannot become a log flood.
            _metrics_fail += 1
            if _metrics_fail in (1, 10) or _metrics_fail % 100 == 0:
                try:
                    print(f"[metrics] collection failing ({_metrics_fail}x): "
                          f"{type(e).__name__}: {e}")
                except Exception:
                    pass
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
    "tool_aliases": {
        # Models call tools by the name they expect rather than the name the client declared —
        # "run" when the client offers "terminal" is the common one. Without this the call is
        # either rejected as hallucinated or passed through for the client to fail on, when the
        # intent was unambiguous and the fix is a rename.
        #
        # Each entry is either a plain target name, or an object that also renames arguments:
        #   "map": {
        #     "run": "terminal",
        #     "execute": {"to": "terminal", "args": {"command": "cmd", "script": "cmd"}}
        #   }
        "enabled": False,
        "action": "rewrite",      # "rewrite" applies it; "audit" records what it would do
        "map": {},
        # Only rename onto a tool the request actually declared. Off, a typo in the map would
        # invent a tool the client has never heard of, which is worse than the original problem.
        "require_declared": True,
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
    "context_compressor": {
        # Deterministic prompt compressor: squeezes bulky tool outputs (and JSON blobs) already
        # in the message history before forwarding upstream, reclaiming context / tokens. Lossy —
        # opt in per condition. Runs after tool_pruner, before the body is re-serialized.
        "enabled": False,
        # mode: "shadow" measures potential savings only (does NOT change the request);
        # "live" actually compresses the forwarded body. Start in shadow to see the numbers.
        "mode": "shadow",
        "tool_output_max_chars": 4000,   # head/tail-truncate tool results longer than this
        "json_min_chars": 2000,          # minify (strip whitespace) JSON content larger than this
        "min_saved_chars": 200,          # skip squeezes that don't save at least this much
        # Optional conditions (all provided must hold):
        "from_model_prefix": "",         # e.g. "qwen" — only for models whose name starts with this
        "client_app": "",                # e.g. "claude-code"
        "prompt_chars_gt": 0,            # only when the whole prompt exceeds this many chars
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
    "model_control": {
        # Loading and unloading models, and switching which vLLM server is up. Not a request
        # transform like the rest of this file, but this is the config surface that is editable
        # live and survives restarts, so it lives here rather than in a second mechanism.
        #
        # vllm_containers: the containers that represent your vLLM configurations. Leave empty to
        # auto-discover anything whose name or image mentions vllm — fine for the common case,
        # but list them explicitly when a container is named something else, or when you want the
        # proxy to leave the others alone. They must publish the port VLLM_URL points at, since
        # they contend for it and only one can be up.
        #   "vllm_containers": ["qwen-vllm", "ornith-vllm"]
        "vllm_containers": [],
        # Auto-load: Ollama-style on-demand loading for the fixed backends (vLLM containers,
        # llama.cpp). Off by default — turning it on means a request may stop the currently
        # serving twin and evict other models when the target does not fit in free memory.
        "auto_load": {"enabled": False, "ready_timeout_s": 900, "min_hold_s": 120,
                      "drain_s": 120},
        # vLLM takes minutes to load weights and build its KV cache. This is how long to wait for
        # /v1/models to answer before reporting that it started but is not ready.
        "vllm_ready_timeout_s": 420,
        # Applied when loading an LM Studio model, unless the request overrides them. A model
        # loaded at the wrong context length or without parallelism benchmarks as a different
        # thing, so these are worth pinning per box.
        "lmstudio_context_length": None,
        "lmstudio_parallel": None,
        "lmstudio_gpu": None,
        # Services the proxy can start and stop, by systemd unit. `scope: "user"` uses
        # `systemctl --user`, which needs no privileges when the proxy runs as the same user
        # that owns the unit — the alternative is a sudoers grant, and widening what this
        # process can do as root to press a button is a poor trade.
        #
        # Only units listed here can be touched, and only with start/stop/restart: the unit
        # name reaches systemctl as an argument, never through a shell.
        "services": {
            "comfyui": {"unit": "comfyui.service", "scope": "user"},
        },
        # llama.cpp serves one model per process with its context fixed at launch, so changing
        # the context means relaunching it. These are the pieces the proxy needs to rewrite the
        # unit's command line; everything not listed here is left exactly as the unit has it.
        "llamacpp": {
            "unit": "llamacpp.service", "scope": "user",
            "binary": "", "model": "",
            "host": "0.0.0.0", "port": 8080, "ngl": 999,
            "extra_args": [],
        },
    },
    "archive": {
        # Request bodies are ~340 KB each and arrive at ~3 GB/day, while the row describing
        # them is ~1.4 KB. Nothing analytical reads a body any more — stats, the reports and
        # the tool counts all run off metadata — so cold bodies are pure archival weight in a
        # file that has to be backed up, vacuumed and kept in page cache.
        #
        # Moving them to a second file keeps them readable (the requests_v view spans both, so
        # opening an old request works exactly as before) while the working database stays
        # small. Off by default: it relocates data, which is the caller's decision to make.
        "enabled": False,
        "after_days": 7,          # bodies older than this move on the next sweep
        "chunk": 200,             # rows per transaction — insert and delete together
        "pause_s": 0.1,           # breathing room between chunks so it never floods the disk
    },
    "pricing": {
        # Reference rates for the usage report's daily ledger. Local models produce no invoice,
        # so this is not a bill — it is what the same traffic would have cost had it gone
        # somewhere else, which is the only way to put a number on what the hardware returns.
        #
        # Two tiers by default, because one is misleading in either direction. A frontier rate
        # flatters a local 30B that is not frontier quality; an open-weights rate ignores that
        # some of this work would have gone to a better model if it hadn't run here. The pair
        # brackets the honest answer. Add or replace tiers freely — every one gets a column.
        #
        # Rates are US dollars per million tokens. `cached_input` matters more than either of
        # the others: agentic traffic re-sends the whole conversation every turn, so nearly all
        # prompt tokens are a prefix the server already holds, and every provider discounts
        # them roughly tenfold. Charging them at full rate overstates the total several times.
        "enabled": True,
        "tiers": [
            {"name": "Claude Sonnet 4.5", "input": 3.0, "cached_input": 0.30, "output": 15.0},
            {"name": "Hosted open-weights, 30B class",
             "input": 0.30, "cached_input": 0.03, "output": 0.60},
        ],
        # What the work actually cost to produce, for the other side of the comparison.
        #
        # Both wattages mean the WHOLE MACHINE at the wall, not the GPU: serving a token also
        # costs you CPU, RAM, NVMe, fans and PSU losses, and billing only the accelerator makes
        # local inference look cheaper than it is. A watt meter for ten seconds under load, and
        # ten more at rest, settles both numbers and beats every estimate here.
        #
        # usd_per_kwh defaults to the US residential average; set it to your own rate.
        # watts: left null, the report falls back to the GPU's own reported draw and labels the
        # figure a floor. That fallback is wrong twice over — it omits the rest of the box, and
        # on unified-memory parts (DGX Spark / GB10 and friends) nvidia-smi reports only one
        # rail of the module, with no power limit to check it against.
        # watts_idle: the machine draws power between requests too, and on bursty traffic the
        # idle hours are most of the day. Left null it is assumed at a fraction of load draw,
        # which the report says out loud.
        "electricity": {"usd_per_kwh": 0.17, "watts": None, "watts_idle": None},
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

# ---- Per-model quirks -------------------------------------------------------------------------
# Known model-specific workarounds the proxy applies automatically. Loaded from a JSON file
# (PROXY_MODEL_QUIRKS or the state dir), merged over the baked-in defaults below, hot-reloaded on
# file change. Each entry maps a model-name pattern to:
#   match:    "prefix" | "substring" | "exact"      (default "substring")
#   thinking: "force_off" | "default_off_optin" | "normal"
#             - force_off:          always inject chat_template_kwargs.enable_thinking=false
#             - default_off_optin:  off by default, but reasoning_effort high/medium turns it on
#             - normal:             don't touch thinking
#   notes:    human-readable description of the quirk (surfaced via /api/model-quirks)
MODEL_QUIRKS_FILE = _resolve_state_path(
    "PROXY_MODEL_QUIRKS", "model_quirks.json",
    Path(__file__).parent / "model_quirks.json", "model_quirks.json"
)
DEFAULT_MODEL_QUIRKS = {
    "ornith": {
        "match": "prefix",
        "thinking": "default_off_optin",
        "reasoning_control": "chat_template_kwargs.enable_thinking",
        "system_nudge": ("Be concise and direct. Start immediately with the answer — the code, "
                         "command, tool call, or the direct fact — with ZERO preamble (no \"I'll...\", "
                         "\"Let me...\", \"I need to...\", \"First,...\", \"I can see...\", \"The user "
                         "wants...\"). Give the minimal response that fully answers: do NOT add "
                         "examples, alternative approaches, step-by-step breakdowns, caveats, or a "
                         "closing summary unless explicitly asked. Prefer the shortest correct "
                         "answer. If acting, emit the tool call directly with no narration."),
        "notes": ("Qwen3.5-MoE / gated-delta-net. Ignores OpenAI reasoning_effort — thinking is "
                  "only controllable via chat_template_kwargs.enable_thinking, and the template "
                  "defaults to ON. On vLLM, requires --reasoning-parser qwen3 to split reasoning "
                  "into the `reasoning` field (else it bleeds into `content`). Over-reasons and can "
                  "return empty content when thinking is on, so default OFF (bench-optimal for "
                  "coding); reasoning_effort high/medium opts in. Even with thinking off it narrates "
                  "reasoning in the content, so system_nudge curbs that (only applied when no-think)."),
    },
}
_QUIRKS_CACHE: dict = {"mtime": None, "data": None}


def load_model_quirks() -> dict:
    """Baked-in DEFAULT_MODEL_QUIRKS merged with the JSON file (file wins per-key). Cached on mtime
    so an edit to the file takes effect without a restart; malformed JSON falls back to defaults."""
    try:
        mt = os.path.getmtime(MODEL_QUIRKS_FILE)
    except OSError:
        mt = None
    if _QUIRKS_CACHE["data"] is not None and _QUIRKS_CACHE["mtime"] == mt:
        return _QUIRKS_CACHE["data"]
    data = {k: dict(v) for k, v in DEFAULT_MODEL_QUIRKS.items()}
    if mt is not None:
        try:
            with open(MODEL_QUIRKS_FILE, encoding="utf-8") as f:
                file_q = json.load(f)
            if isinstance(file_q, dict):
                for k, v in file_q.items():
                    if isinstance(v, dict):
                        data[k] = {**data.get(k, {}), **v}
        except (OSError, ValueError):
            pass
    _QUIRKS_CACHE["mtime"] = mt
    _QUIRKS_CACHE["data"] = data
    return data


def _model_quirk(model_name: str) -> dict | None:
    """First quirk whose pattern matches the (post-routing) model name, else None."""
    if not model_name:
        return None
    ml = model_name.lower()
    for pat, q in load_model_quirks().items():
        if not isinstance(q, dict):
            continue
        mode = q.get("match", "substring")
        p = str(pat).lower()
        if ((mode == "prefix" and ml.startswith(p))
                or (mode == "exact" and ml == p)
                or (mode == "substring" and p in ml)):
            return q
    return None


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


# ---- Artifact extraction ---------------------------------------------------------------------
# Maps tool name -> (kind, op, path_arg, content_arg). path_arg may hold a list (web_extract 'urls')
# → one artifact per entry. content_arg (if present) is hashed for change-detection + size. Extend
# via the tool_artifact_map rule ({"tools": {"my_tool": ["file","write","path","content"]}}).
TOOL_ARTIFACT_MAP = {
    "write_file": ("file", "write", "path", "content"),
    "create_file": ("file", "create", "path", "content"),
    "read_file": ("file", "read", "path", None),
    "view_file": ("file", "read", "path", None),
    "edit_file": ("file", "edit", "path", "content"),
    "patch": ("file", "edit", "path", "new_string"),
    "str_replace": ("file", "edit", "path", "new_str"),
    "str_replace_editor": ("file", "edit", "path", "new_str"),
    "apply_patch": ("file", "edit", "path", "patch"),
    "delete_file": ("file", "delete", "path", None),
    "skill_view": ("skill", "read", "name", None),
    "skill_manage": ("skill", "edit", "name", "new_string"),
    "web_extract": ("url", "fetch", "urls", None),
    "web_fetch": ("url", "fetch", "url", None),
    "fetch": ("url", "fetch", "url", None),
    "browser_navigate": ("url", "fetch", "url", None),
    "web_search": ("url", "search", "query", None),
    "search_files": ("dir", "search", "pattern", None),
    "list_dir": ("dir", "list", "path", None),
    "glob": ("dir", "search", "pattern", None),
}
_TERMINAL_TOOLS = ("terminal", "bash", "shell", "run_command", "execute_command")
_PATH_IN_CMD_RE = re.compile(
    r'"([^"]+\.[A-Za-z0-9]{1,8})"|\'([^\']+\.[A-Za-z0-9]{1,8})\'|((?:[A-Za-z]:)?[\w.\\/-]*[\\/][\w.\\/-]*\.[A-Za-z0-9]{1,8})')


def _artifact_tool_map() -> dict:
    cfg = (load_rules_config().get("tool_artifact_map") or {})
    extra = cfg.get("tools") if isinstance(cfg, dict) else None
    if not isinstance(extra, dict):
        return TOOL_ARTIFACT_MAP
    merged = dict(TOOL_ARTIFACT_MAP)
    for k, v in extra.items():
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            merged[str(k).lower()] = (v[0], v[1], v[2], v[3] if len(v) > 3 else None)
    return merged


def _paths_from_command(cmd: str) -> list:
    """Best-effort file paths referenced in a shell command (for terminal/bash tools). Rejects
    grep/sed/regex fragments (which are often quoted and look path-ish) so they don't pollute the
    artifact list."""
    if not isinstance(cmd, str) or not cmd or len(cmd) > 4000:
        return []
    seen = []
    for m in _PATH_IN_CMD_RE.finditer(cmd):
        p = m.group(1) or m.group(2) or m.group(3)
        if not p or len(p) >= 300 or ("/" not in p and "\\" not in p):
            continue
        # Shell/regex noise: pipe-alternation, globs, redirects, var expansion, escaped-dot regex.
        if _RE_PATH_NOISE.search(p) or "\\|" in p or ".*" in p or p.startswith("\\."):
            continue
        if p not in seen:
            seen.append(p)
        if len(seen) >= 6:
            break
    return seen


_RE_PATH_NOISE = re.compile(r"[|*?<>$`;]")


def _artifact_events_from_call(name, args, tmap):
    if not name:
        return
    name_l = str(name).lower()
    spec = tmap.get(name_l)
    if spec:
        kind, op, path_arg, content_arg = spec
        if not isinstance(args, dict):
            return
        raw = args.get(path_arg)
        content = args.get(content_arg) if content_arg else None
        chash = csize = None
        if isinstance(content, str) and content:
            chash = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()[:16]
            csize = len(content)
        paths = raw if isinstance(raw, list) else ([raw] if raw is not None else [])
        for p in paths:
            if p is None:
                continue
            yield {"kind": kind, "op": op, "path": str(p)[:500], "tool": name_l,
                   "content_hash": chash, "size": csize}
        return
    if name_l in _TERMINAL_TOOLS and isinstance(args, dict):
        cmd = args.get("command") or args.get("cmd") or ""
        for p in _paths_from_command(cmd):
            yield {"kind": "file", "op": "access", "path": p, "tool": name_l,
                   "content_hash": None, "size": None}


def _extract_artifacts_into(conn, req_id, conv_id, client_ip, ts, body) -> int:
    """Extract artifacts from a request body's recent history; insert (dedup by sig). Returns #new.

    The kill switch is enforced here as well as at the sweep, so no path — sweep, backfill, or a
    future caller — can write artifact rows while capture is disabled."""
    if not ARTIFACTS_ENABLED:
        return 0
    if not isinstance(body, dict):
        return 0
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return 0
    tmap = _artifact_tool_map()
    events = []
    for m in msgs[-24:]:   # tail only — dedup catches earlier turns as they scroll through the tail
        if not isinstance(m, dict):
            continue
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            a = fn.get("arguments")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except (json.JSONDecodeError, ValueError):
                    a = None
            events.extend(_artifact_events_from_call(fn.get("name"), a, tmap))
        c = m.get("content")
        if isinstance(c, list):
            for blk in c:
                if not isinstance(blk, dict):
                    continue
                if blk.get("type") == "tool_use":
                    events.extend(_artifact_events_from_call(blk.get("name"), blk.get("input") or {}, tmap))
                elif blk.get("type") in ("image", "image_url"):
                    src = blk.get("image_url") or blk.get("source") or {}
                    url = src.get("url") if isinstance(src, dict) else None
                    if isinstance(url, str) and url.startswith("http"):
                        events.append({"kind": "image", "op": "vision", "path": url[:500],
                                       "tool": "image", "content_hash": None, "size": None})
                    else:
                        events.append({"kind": "image", "op": "vision", "path": "(inline image)",
                                       "tool": "image", "content_hash": None, "size": None})
    n = 0
    for ev in events:
        sig = hashlib.sha256(
            f"{conv_id}|{ev['tool']}|{ev['op']}|{ev['path']}|{ev.get('content_hash') or ''}"
            .encode("utf-8", "replace")).hexdigest()
        try:
            cur = conn.execute(
                "INSERT OR IGNORE INTO artifacts (sig, ts, request_id, conversation_id, client_ip, "
                "kind, op, path, tool_name, size_bytes, content_hash, meta_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (sig, ts, req_id, conv_id, client_ip, ev["kind"], ev["op"], ev["path"],
                 ev["tool"], ev.get("size"), ev.get("content_hash"), None))
            n += cur.rowcount
        except sqlite3.Error:
            pass
    return n


def _artifact_sweep() -> None:
    """Extract artifacts from requests completed since the last watermark. Dedup makes reprocessing
    safe, so a small ts lookback catches out-of-order stragglers. LIMIT bounds per-cycle work."""
    if not ARTIFACTS_ENABLED:
        return
    conn = db()
    s = get_setting("artifact_watermark")
    try:
        wm = float(s["value"]) if s and s.get("value") else 0.0
    except (ValueError, TypeError):
        wm = 0.0
    try:
        rows = conn.execute(
            "SELECT id, conversation_id, client_ip, ts, request_body FROM requests_v "
            "WHERE ts > ? AND request_body IS NOT NULL ORDER BY ts LIMIT 300", (wm - 30,)).fetchall()
    except sqlite3.Error:
        conn.close()
        return
    maxts = wm
    for r in rows:
        if r["ts"] and r["ts"] > maxts:
            maxts = r["ts"]
        try:
            body = json.loads(r["request_body"])
        except (json.JSONDecodeError, TypeError):
            continue
        _extract_artifacts_into(conn, r["id"], r["conversation_id"], r["client_ip"], r["ts"] or 0, body)
    conn.commit()
    conn.close()
    if maxts > wm:
        set_setting("artifact_watermark", str(maxts))


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
    """Block when the same tool call (name + normalized args) repeats too often SINCE THE LAST
    USER MESSAGE. A fresh user turn resets the loop window — so if the user tells the model to
    "start again", the request is no longer blocked. When a loop happened *before* the last user
    turn (i.e. the user just interrupted a loop), we allow the request but attach a soft-unblock
    note so the model knows it was stopped for looping and shouldn't repeat the same call."""
    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return None

    # Index of the most recent human turn. Tool calls before it belong to a loop the user has
    # since interrupted; tool calls after it are the current attempt.
    last_user_idx = -1
    for i, m in enumerate(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            last_user_idx = i

    window = max(1, int(cfg.get("window", 10)))
    max_repeats = max(2, int(cfg.get("max_repeats", 4)))
    tail_n = max(2, int(cfg.get("tail_consecutive", 3)))

    def _collect(msgs) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for m in msgs:
            if not (isinstance(m, dict) and m.get("role") == "assistant"):
                continue
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                out.append((fn.get("name") or "?", _normalize_args(fn.get("arguments"))))
        return out

    def _detect(sigs):
        """Return (tool_name, count, trigger, args_preview) or None."""
        if not sigs:
            return None
        counts: dict = {}
        for s in sigs:
            counts[s] = counts.get(s, 0) + 1
        top_sig, top_count = max(counts.items(), key=lambda kv: kv[1])
        if top_count >= max_repeats:
            return (top_sig[0], top_count, "count_in_window", top_sig[1])
        if len(sigs) >= tail_n and all(s == sigs[-1] for s in sigs[-tail_n:]):
            return (sigs[-1][0], tail_n, "tail_consecutive", sigs[-1][1])
        return None

    # Active loop: assistant tool calls made SINCE the last user message.
    active_msgs = messages[last_user_idx + 1:] if last_user_idx >= 0 else messages
    active = _detect(_collect(active_msgs)[-window:])
    if active:
        name, count, trigger, args_preview = active
        return {
            "reason": f"Tool call {name!r} repeated {count}× since the last user message",
            "details": {"trigger": trigger, "tool_name": name, "count": count,
                        "args_preview": args_preview[:300]},
        }

    # No active loop, but a loop before the last user turn means the user just interrupted one.
    # Allow through, but carry a note so the model learns why it was stopped.
    if last_user_idx >= 0:
        prior = _detect(_collect(messages[:last_user_idx + 1])[-window:])
        if prior:
            name, count, trigger, args_preview = prior
            note = (f"[AI Proxy loop-detector] Your earlier turn called the tool `{name}` with the "
                    f"same parameters {count}× in a row and was stopped to prevent an infinite loop. "
                    f"The user has asked you to continue — do NOT repeat that identical call; change "
                    f"the parameters or try a different approach.")
            return {
                "reason": f"loop on {name!r} ({count}×) cleared by user intervention — note injected",
                "details": {"trigger": trigger, "tool_name": name, "count": count,
                            "args_preview": args_preview[:300], "soft_unblock": True, "note": note},
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


def _alias_target(entry):
    """An alias is either "to", or {"to": ..., "args": {from: to}}."""
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict):
        return str(entry.get("to") or ""), (entry.get("args") or {})
    return "", {}


def _apply_tool_aliases(response_obj: dict, request_body) -> list:
    """Rename tool calls the model got wrong onto the tool the client actually declared.

    Runs before validation, so a rename-able name is never reported as a hallucination — the
    point is to fix the call, not to describe it. Returns what changed, for the audit log:
    silently rewriting a model's tool call is not something to do without a record.
    """
    cfg = (load_rules_config().get("tool_aliases") or {})
    if not cfg.get("enabled"):
        return []
    amap = cfg.get("map") or {}
    if not isinstance(amap, dict) or not amap:
        return []
    declared = _build_tool_schemas(request_body)
    require_declared = bool(cfg.get("require_declared", True))
    rewrite = (cfg.get("action") or "rewrite") == "rewrite"
    changes = []

    def fix(tc):
        if not isinstance(tc, dict):
            return
        fn = tc.get("function")
        if not isinstance(fn, dict):
            return
        name = fn.get("name")
        # Never rename a call that would already have worked.
        if not isinstance(name, str) or name in declared:
            return
        entry = amap.get(name)
        if entry is None:
            return
        target, arg_map = _alias_target(entry)
        if not target or (require_declared and declared and target not in declared):
            return
        change = {"from": name, "to": target, "renamed_args": {}}
        if rewrite:
            fn["name"] = target
        # Argument keys usually differ too: run(command=...) against terminal(cmd=...).
        if arg_map:
            raw = fn.get("arguments")
            parsed = raw
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    parsed = None
            if isinstance(parsed, dict):
                for src, dst in arg_map.items():
                    if src in parsed and dst not in parsed:
                        change["renamed_args"][src] = dst
                        if rewrite:
                            parsed[dst] = parsed.pop(src)
                if rewrite and change["renamed_args"]:
                    fn["arguments"] = json.dumps(parsed) if isinstance(raw, str) else parsed
        changes.append(change)

    if not isinstance(response_obj, dict):
        return changes
    for choice in (response_obj.get("choices") or []):
        msg = (choice.get("message") if isinstance(choice, dict) else None) or {}
        for tc in (msg.get("tool_calls") or []):
            fix(tc)
    msg = response_obj.get("message")          # Ollama native /api/chat
    if isinstance(msg, dict):
        for tc in (msg.get("tool_calls") or []):
            fix(tc)
    return changes



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
    # Renaming a tool call means rewriting the response, which means having the whole response —
    # so aliasing has to request buffering the same way the other post-flight rules do. Without
    # it the rule is silently inert for streaming clients, which is nearly all of them.
    ta_cfg = cfg.get("tool_aliases") or {}
    ta = bool(ta_cfg.get("enabled") and (ta_cfg.get("map") or {}))
    return bool(sv or ht or af or xa or ta)


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


# Magic bytes for the formats we claim to surface. Anything else isn't an image we can render.
_IMAGE_MAGIC = (
    b"\x89PNG\r\n\x1a\n",     # PNG
    b"\xff\xd8\xff",           # JPEG
    b"GIF87a", b"GIF89a",      # GIF
    b"RIFF",                   # WEBP (RIFF....WEBP)
    b"BM",                     # BMP
)


def _decode_b64_image(payload) -> bytes | None:
    """Decode a base64 image payload, returning the bytes only if they really are an image.

    Pattern-matching a base64 prefix is not enough to conclude a request contains an image: the
    JPEG marker '/9j/' is four base64 characters and turns up by chance inside unrelated blobs and
    text, and a truncated data URL yields a run that looks right but decodes to nothing usable.
    Both produce a request flagged as having an image whose image can never be displayed — so the
    test is 'does it decode to something with an image header', not 'does it look like one'.
    """
    if not isinstance(payload, str):
        return None
    b = "".join(payload.split())          # data URLs are sometimes wrapped across lines
    if len(b) < 64:
        return None
    try:
        # validate=True matters: with it off, base64 silently DISCARDS non-alphabet characters,
        # so "/9j/" followed by arbitrary prose still decodes to the three JPEG magic bytes and
        # would be accepted as an image.
        raw = base64.b64decode(b + "=" * (-len(b) % 4), validate=True)
    except (binascii.Error, ValueError):
        return None
    # Magic bytes alone aren't enough either — they're only 2-8 bytes, and no real image is that
    # small. Anything under a few hundred bytes is a coincidence, not a picture.
    if len(raw) < 128 or not raw.startswith(_IMAGE_MAGIC):
        return None
    return raw


def _image_is_complete(raw: bytes) -> bool:
    """Does the image actually finish, or does it just start convincingly?

    Magic bytes only prove the first few bytes. A screenshot cut off mid-transfer keeps a
    perfect header — right format, right dimensions — and loses the pixels, so it passes every
    check we had and then renders as a broken thumbnail with nothing to explain why. Clients cap
    tool output at a fixed character count and will happily slice a base64 payload in half; the
    only way to know is to look for the terminator each format is required to end with.
    """
    if not isinstance(raw, bytes) or len(raw) < 16:
        return False
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return raw.rstrip().endswith(b"IEND\xaeB`\x82")
    if raw.startswith(b"\xff\xd8\xff"):
        return raw.rstrip(b"\x00").endswith(b"\xff\xd9")
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return raw.endswith(b"\x3b")
    if raw.startswith(b"RIFF"):
        # The RIFF header declares its own payload size; a short file is a truncated one.
        try:
            return len(raw) >= 8 + struct.unpack("<I", raw[4:8])[0]
        except struct.error:
            return False
    # BMP and anything else: no reliable terminator, so don't claim it's broken.
    return True


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


# num_ctx cannot be delivered over Ollama's OpenAI-compatible endpoint. Measured directly:
# top-level `num_ctx` and nested `options.num_ctx` are both ignored there, and only the native
# /api/chat honours them. A preload does not survive either — the OpenAI handler sends the
# server default with every request, which reloads the model and silently discards the window
# that was asked for. That is what quietly undid a 65,536 preload in the middle of a benchmark.
#
# Ollama's own answer is a derived model carrying the parameter, which every endpoint respects
# because it lives in the model rather than the request. Deriving costs no disk (the blobs are
# shared) and the rule then works as its config always implied it did.
_CTX_MODELS_SEEN: set = set()


def _ctx_derived_name(model: str, num_ctx: int) -> str:
    """Stable, collision-free name for the pinned variant. Tag is dropped into the name so
    'qwen3.8:27b' at 512k becomes 'qwen3.8-27b-ctx512k', which reads in `ollama list`."""
    base = re.sub(r"[^A-Za-z0-9._-]", "-", (model or "").replace(":", "-")).strip("-")
    k = num_ctx // 1024
    suffix = f"{k // 1024}m" if k >= 1024 and k % 1024 == 0 else f"{k}k"
    return f"{base}-ctx{suffix}"


async def _ensure_ctx_model(base: str, num_ctx: int) -> str | None:
    """Create (once) and return the pinned variant of `base`, or None to leave the request be.

    Failure is never fatal: a request that cannot get its bigger window should still be served
    at the default rather than refused, so every error path returns None and the caller
    forwards the original model.
    """
    name = _ctx_derived_name(base, num_ctx)
    if name in _CTX_MODELS_SEEN:
        return name
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as c:
            tags = (await c.get(f"{OLLAMA_URL}/api/tags")).json()
            have = {(t.get("name") or "").split(":")[0] for t in (tags.get("models") or [])}
            if name in have:
                _CTX_MODELS_SEEN.add(name)
                return name
            # A derived vision model loses its projector and then fails to load at all:
            # deriving minicpm-v4.5 produced "Load failed" with the mmproj still being sized.
            # Serving the base model at the default window is a worse context but a working
            # one, which beats a variant that cannot answer.
            info = await c.post(f"{OLLAMA_URL}/api/show", json={"model": base})
            caps = (info.json() or {}).get("capabilities") or [] if info.status_code < 400 else []
            if "vision" in [str(x).lower() for x in caps]:
                print(f"[ollama_options] refusing to derive {name}: {base} is a vision model "
                      f"and loses its projector when derived — serving it unpinned instead")
                return None
            r = await c.post(f"{OLLAMA_URL}/api/create",
                             json={"model": name, "from": base,
                                   "parameters": {"num_ctx": int(num_ctx)}, "stream": False})
            if r.status_code >= 400:
                print(f"[ollama_options] could not derive {name} from {base}: "
                      f"HTTP {r.status_code} {r.text[:120]}")
                return None
    except (httpx.RequestError, ValueError, KeyError) as e:
        print(f"[ollama_options] could not derive {name}: {type(e).__name__}: {e}")
        return None
    _CTX_MODELS_SEEN.add(name)
    print(f"[ollama_options] derived {name} from {base} with num_ctx={num_ctx}")
    return name


async def apply_ollama_ctx_model(body: dict, ctx: dict) -> dict | None:
    """Swap the model for a context-pinned variant when the rule asks for a num_ctx the
    OpenAI path cannot carry. Native-path requests are left alone — there, num_ctx works."""
    if not isinstance(body, dict) or "/api/" in (ctx.get("path") or ""):
        return None
    cfg = (load_rules_config().get("ollama_options") or {})
    if not cfg.get("enabled", False):
        return None
    base = body.get("model")
    if not base or _ctx_derived_name(base, 1).rsplit("-ctx", 1)[0] in ("", None):
        return None
    want = None
    for src in ((cfg.get("per_model") or {}).get(base), cfg.get("defaults") or {}):
        if isinstance(src, dict) and src.get("num_ctx"):
            want = int(src["num_ctx"])
            break
    if not want:
        return None
    derived = await _ensure_ctx_model(base, want)
    if not derived or derived == base:
        return None
    body["model"] = derived
    return {"from": base, "to": derived, "num_ctx": want}


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
            if k == "num_ctx":
                # Measured: Ollama's OpenAI endpoint ignores num_ctx both at top level and
                # nested under options. Setting it here changed nothing while reporting itself
                # as applied — a rule claiming an effect it does not have. apply_ollama_ctx_model
                # delivers it for real, via a derived model, and records itself in `sources`.
                continue
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


# ---- Context compressor: deterministic squeeze of bulky tool outputs in the history ----
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Rolling in-memory savings tally (live + shadow), surfaced at /__proxy/api/compress-stats.
# by_tool maps tool name -> {saved, n} so the Stats page can show which outputs are the bloat.
_COMPRESS_STATS = {"live": {"reqs": 0, "orig": 0, "saved": 0, "by_tool": {}},
                   "shadow": {"reqs": 0, "orig": 0, "saved": 0, "by_tool": {}}}

def _squeeze_text(text, tool_max, json_min):
    """Shrink one bulky tool-output string deterministically. Returns (new_text, saved_chars).
    Strip ANSI → minify large JSON → head/tail-truncate anything still over the cap."""
    if not isinstance(text, str):
        return text, 0
    orig_len = len(text)
    if orig_len <= min(tool_max, json_min):
        return text, 0
    out = _ANSI_RE.sub("", text)
    stripped = out.strip()
    if orig_len >= json_min and stripped[:1] in "{[":
        try:
            out = json.dumps(json.loads(stripped), separators=(",", ":"), ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            pass
    if len(out) > tool_max:
        keep = max(400, tool_max)
        head = int(keep * 0.7); tail = keep - head
        elided = len(out) - head - tail
        if elided > 0:
            out = out[:head] + f"\n… [{elided} chars elided by proxy compressor] …\n" + out[-tail:]
    saved = orig_len - len(out)
    return (out, saved) if saved > 0 else (text, 0)


def evaluate_context_compressor(body, ctx) -> dict | None:
    """Squeeze bulky tool outputs in the message history. 'live' mode mutates body in place;
    'shadow' mode only measures. Returns audit info, or None if it did/would do nothing."""
    if not isinstance(body, dict):
        return None
    cfg = (load_rules_config().get("context_compressor") or {})
    if not cfg.get("enabled", False):
        return None
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None
    fmp = (cfg.get("from_model_prefix") or "").strip()
    if fmp and not str(body.get("model") or "").startswith(fmp):
        return None
    capp = (cfg.get("client_app") or "").strip()
    if capp and ctx.get("client_app") != capp:
        return None
    mode = "live" if (cfg.get("mode") or "shadow").lower() == "live" else "shadow"
    tool_max = max(200, int(cfg.get("tool_output_max_chars", 4000) or 4000))
    json_min = max(200, int(cfg.get("json_min_chars", 2000) or 2000))
    min_saved = max(0, int(cfg.get("min_saved_chars", 200) or 200))
    total_orig = sum(len(json.dumps(m)) for m in msgs if isinstance(m, dict))
    if int(cfg.get("prompt_chars_gt") or 0) and total_orig < int(cfg.get("prompt_chars_gt")):
        return None

    # Map tool_call_id / tool_use_id -> tool name from assistant turns, so savings can be
    # attributed to the tool that produced the bloated output.
    tcnames: dict = {}
    for m in msgs:
        if not (isinstance(m, dict) and m.get("role") == "assistant"):
            continue
        for tc in (m.get("tool_calls") or []):
            if isinstance(tc, dict) and tc.get("id") and (tc.get("function") or {}).get("name"):
                tcnames[tc["id"]] = tc["function"]["name"]
        if isinstance(m.get("content"), list):
            for blk in m["content"]:
                if isinstance(blk, dict) and blk.get("type") == "tool_use" and blk.get("id") and blk.get("name"):
                    tcnames[blk["id"]] = blk["name"]

    saved_total = 0
    n = 0
    by_tool: dict = {}
    def _sq(s, name="?"):
        nonlocal saved_total, n
        new, saved = _squeeze_text(s, tool_max, json_min)
        if saved >= min_saved:
            saved_total += saved; n += 1
            bt = by_tool.setdefault(name or "?", {"saved": 0, "n": 0})
            bt["saved"] += saved; bt["n"] += 1
            return new, True
        return s, False

    for m in msgs:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if m.get("role") == "tool" and isinstance(content, str):
            tname = m.get("name") or tcnames.get(m.get("tool_call_id")) or "?"
            new, changed = _sq(content, tname)
            if changed and mode == "live":
                m["content"] = new
        elif isinstance(content, list):  # Anthropic tool_result blocks ride in user messages
            for blk in content:
                if not (isinstance(blk, dict) and blk.get("type") == "tool_result"):
                    continue
                tname = tcnames.get(blk.get("tool_use_id")) or "?"
                c = blk.get("content")
                if isinstance(c, str):
                    new, changed = _sq(c, tname)
                    if changed and mode == "live":
                        blk["content"] = new
                elif isinstance(c, list):
                    for sub in c:
                        if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                            new, changed = _sq(sub["text"], tname)
                            if changed and mode == "live":
                                sub["text"] = new

    if saved_total <= 0:
        return None
    st = _COMPRESS_STATS[mode]
    st["reqs"] += 1; st["orig"] += total_orig; st["saved"] += saved_total
    for name, bt in by_tool.items():
        agg = st["by_tool"].setdefault(name, {"saved": 0, "n": 0})
        agg["saved"] += bt["saved"]; agg["n"] += bt["n"]
    return {
        "action": "compress" if mode == "live" else "measure",
        "mode": mode, "orig_chars": total_orig, "saved_chars": saved_total,
        "pct": round(100.0 * saved_total / total_orig, 1) if total_orig else 0.0, "n": n,
    }


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
            # A soft-unblock result (loop_detector after user intervention) is allowed through as
            # a 'warn' regardless of the rule's configured action, so it never blocks the request.
            soft = bool((result.get("details") or {}).get("soft_unblock"))
            return {
                "verdict": "warn" if soft else rcfg.get("action", "block"),
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
    # `ui` fingerprints the served dashboard so a long-lived tab can notice it is running code
    # older than the server. A dashboard is left open for days; without this, every deploy is
    # invisible to it and old behaviour looks like a bug in the new build.
    ui = None
    try:
        st = (STATIC_DIR / "index.html").stat()
        ui = f"{int(st.st_mtime)}-{st.st_size}"
    except OSError:
        pass
    return {"version": __version__, "ui": ui, "upstream": OLLAMA_URL, "anthropic": ANTHROPIC_URL,
            "lmstudio": LMSTUDIO_URL, "vllm": VLLM_URL, "llamacpp": LLAMACPP_URL,
            "port": PROXY_PORT}


@app.get("/__proxy/api/whoami")
async def whoami(request: Request):
    """What the current viewer is allowed to see. 'admin' → all requests on every subnet;
    'subnet' → only requests from the same /24 (or loopback); 'unrestricted' → redaction off."""
    ip = _client_ip(request)
    is_loopback = False
    try:
        is_loopback = bool(ip) and ipaddress.ip_address(ip).is_loopback
    except (ValueError, TypeError):
        pass
    is_admin = bool(ip and ip in ADMIN_IPS)
    if not REDACT_PII_ENABLED:
        level = "unrestricted"
    elif is_admin:
        level = "admin"
    else:
        level = "subnet"
    return {
        "ip": ip,
        "level": level,
        "is_admin": is_admin,
        "admin_configured": bool(ADMIN_IPS),
        "redact_pii": REDACT_PII_ENABLED,
        "loopback": is_loopback,
        "subnet_bits": REDACT_SUBNET_BITS_V4,
    }


@app.get("/__proxy/api/compress-stats")
async def compress_stats():
    """Rolling context-compressor savings (since last restart), for the Stats readout."""
    def _s(d):
        tools = sorted(
            ({"name": k, "saved_chars": v["saved"], "n": v["n"]} for k, v in (d.get("by_tool") or {}).items()),
            key=lambda t: t["saved_chars"], reverse=True)[:12]
        return {"reqs": d["reqs"], "orig_chars": d["orig"], "saved_chars": d["saved"],
                "pct": round(100.0 * d["saved"] / d["orig"], 1) if d["orig"] else 0.0,
                "by_tool": tools}
    cc = (load_rules_config().get("context_compressor") or {})
    return {"live": _s(_COMPRESS_STATS["live"]), "shadow": _s(_COMPRESS_STATS["shadow"]),
            "enabled": bool(cc.get("enabled")), "mode": (cc.get("mode") or "shadow"),
            "config": {k: cc.get(k) for k in ("tool_output_max_chars", "json_min_chars",
                       "min_saved_chars", "from_model_prefix", "client_app", "prompt_chars_gt")}}


@app.delete("/__proxy/api/compress-stats")
async def compress_stats_reset():
    for k in _COMPRESS_STATS:
        _COMPRESS_STATS[k] = {"reqs": 0, "orig": 0, "saved": 0, "by_tool": {}}
    return {"ok": True}


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


# dbstat walks every page of the database to attribute bytes to tables. On a multi-GB DB that
# is ~1s of saturated disk I/O, and health is polled — by the restart flow, by the shell, on
# every System tab load — so it ran constantly and starved every query running beside it.
# Sizes move slowly; cache them and only compute when someone actually asks.
_DBSTAT_CACHE: dict = {"ts": 0.0, "value": {}}
_DBSTAT_TTL_S = 300.0


def _scan_table_sizes() -> dict:
    """dbstat walks every page in the database. That is 100 seconds on a 6 GB file — measured,
    not estimated — so nothing may call this on the event loop or on a fixed short timer."""
    conn = db()
    started = time.time()
    try:
        sizes = {tbl: sz for tbl, sz in conn.execute(
            "SELECT name, SUM(pgsize) FROM dbstat GROUP BY name ORDER BY 2 DESC").fetchall()}
        _DBSTAT_CACHE["cost_s"] = time.time() - started
        return sizes
    except sqlite3.OperationalError:
        # dbstat virtual table not compiled in — row counts are the best we can do.
        return {}
    finally:
        conn.close()


def _dbstat_ttl() -> float:
    """Refresh no more often than 60x what the scan costs, so this never spends more than a
    couple of percent of wall-clock walking the database. A small database keeps the 5-minute
    floor; a 6 GB one that takes 100 seconds to scan drops to roughly one scan every 100
    minutes. Table sizes move slowly enough that nobody will notice the difference."""
    return max(_DBSTAT_TTL_S, (_DBSTAT_CACHE.get("cost_s") or 0.0) * 60)


def _table_sizes() -> dict:
    """Whatever is known right now, immediately — the scan always happens in the background.

    Never blocking, including on the first call. A cold cache used to mean the System tab hung
    for the length of a full scan (111 seconds on a 6 GB file, measured), and an informational
    panel folded inside a <details> has not earned that. It arrives on the next load instead.
    """
    now = time.time()
    cached = _DBSTAT_CACHE["value"]
    if cached and (now - _DBSTAT_CACHE["ts"]) < _dbstat_ttl():
        return cached
    if not _DBSTAT_CACHE.get("refreshing"):
        _DBSTAT_CACHE["refreshing"] = True

        def refresh():
            try:
                sizes = _scan_table_sizes()
                if sizes:
                    _DBSTAT_CACHE.update({"ts": time.time(), "value": sizes})
            finally:
                _DBSTAT_CACHE["refreshing"] = False

        threading.Thread(target=refresh, daemon=True, name="dbstat-refresh").start()
    return cached


@app.get("/__proxy/api/health")
# Sync handler: every line below blocks — five aggregate queries and, with ?tables=1, a scan of
# every page in the database. As `async def` all of that ran on the event loop, so opening the
# System tab stalled the proxy itself. Same reasoning as system_history.
def health(tables: bool = False):
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
    # Opt-in: only the System tab renders these, and it asks with ?tables=1. A liveness poll
    # has no use for them and shouldn't pay a full-database scan to get them.
    conn.close()
    table_sizes = _table_sizes() if tables else {}

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
            # The archive is a second file; a size that ignores it would understate what the
            # proxy is actually keeping on disk.
            "archive_path": ARCHIVE_DB_PATH,
            "archive_bytes": (Path(ARCHIVE_DB_PATH).stat().st_size
                              if Path(ARCHIVE_DB_PATH).exists() else 0),
            # Itemised, because the two fields above were computed and then never rendered:
            # the UI showed the main file alone while the proxy held four times that on disk.
            "disk": _db_files(),
            # So the UI can say "measuring" instead of silently showing nothing on the first
            # load after a restart.
            "table_sizes_pending": bool(tables and not table_sizes
                                        and _DBSTAT_CACHE.get("refreshing")),
        },
        # Pool occupancy is the one number that would have named the 2026-08-08 outage on
        # sight. Health is the right home for it: it is what a liveness poll already hits,
        # and this failure mode is invisible to every other signal — the service is active,
        # the upstreams are fast, and nothing is served.
        "upstream_pool": _pool_stats(),
        "inflight": len(_INFLIGHT_REQUESTS),
        "tls": _tls_status(),
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


# Deleting rows and rewriting a multi-gigabyte file takes minutes, and the caller deserves to
# see that rather than a button that has apparently hung. One job at a time: two VACUUMs on the
# same file contend for the same write lock, and a DELETE racing a VACUUM is worse still.
_DB_JOB: dict = {"state": "idle"}
_DB_JOB_LOCK = threading.Lock()


def _db_job_status() -> dict:
    job = dict(_DB_JOB)
    started = job.get("started")
    if started:
        end = job.get("finished") or time.time()
        job["elapsed_s"] = round(end - started, 1)
    step_started = job.pop("step_started", None)
    if step_started and job.get("state") == "running":
        job["step_elapsed_s"] = round(time.time() - step_started, 1)
    if job.get("state") == "running":
        # Disk grows while rows are being deleted — the removals land in the write-ahead log
        # before a checkpoint folds them in. Watching the total climb during a "clear" looks
        # like failure unless the log is shown as its own number.
        job["disk"] = _db_files()
    return job


def _db_size() -> int:
    """Everything the proxy keeps on disk, main plus archive — the reclaimed figure is
    meaningless if it ignores the second file."""
    total = 0
    for p in (DB_PATH, ARCHIVE_DB_PATH):
        try:
            total += Path(p).stat().st_size
        except OSError:
            pass
    return total


def _db_files() -> dict:
    """Every file the proxy keeps on disk, itemised.

    The System tab used to report only the main database. During a clear that read 7.7 GB
    while the proxy actually occupied 35 GB — the archive, the two write-ahead logs and a
    stale pre-backup made up the rest, and none of them appeared anywhere in the UI. The
    number you check before deciding to clear the database has to be the real one.
    """
    seen: list = []
    total = 0
    # -wal and -shm are not incidental: during a large delete the WAL can exceed the database
    # it belongs to, which is why disk climbs while rows are being removed.
    for label, path in (
        ("database", DB_PATH),
        ("write-ahead log", DB_PATH + "-wal"),
        ("shared index", DB_PATH + "-shm"),
        ("archive", ARCHIVE_DB_PATH),
        ("archive write-ahead log", ARCHIVE_DB_PATH + "-wal"),
        ("archive shared index", ARCHIVE_DB_PATH + "-shm"),
        ("pre-migration backup", DB_PATH + ".prebak"),
    ):
        try:
            size = Path(path).stat().st_size
        except OSError:
            continue                      # absent is the normal case for several of these
        seen.append({"label": label, "path": path, "bytes": size})
        total += size
    return {"files": seen, "total_bytes": total,
            "total_mb": round(total / (1024 * 1024), 2)}


# Batch size and pause for the reset's deletes. One `DELETE FROM requests` holds a write lock
# for the whole multi-minute operation, which is how clearing the database made the metrics
# collector fail ten times over and returned a 500 to a live /v1/chat/completions. Committing
# in batches with a breath between them lets other writers interleave. Same shape as the
# archive sweep, which already worked this way.
DB_RESET_CHUNK = int(os.environ.get("PROXY_DB_RESET_CHUNK", "2000") or 2000)
DB_RESET_PAUSE_S = float(os.environ.get("PROXY_DB_RESET_PAUSE_S", "0.05") or 0)


def _chunked_delete(conn, table: str, chunk: int, pause_s: float, on_progress) -> int:
    """Delete every row of `table`, committing as it goes. Returns the number removed.

    Chunked by rowid rather than a WHERE clause on the data: every one of these tables is
    being emptied completely, and rowid is the one column guaranteed present and indexed.
    """
    done = 0
    while True:
        cur = conn.execute(
            f"DELETE FROM {table} WHERE rowid IN (SELECT rowid FROM {table} LIMIT ?)",
            (int(chunk),))
        removed = cur.rowcount or 0
        conn.commit()                     # release the write lock between batches
        if removed <= 0:
            return done
        done += removed
        on_progress(done)
        if pause_s:
            time.sleep(pause_s)


def _db_reset_job(targets: list, vacuum: bool) -> dict:
    """Runs in a worker thread. Publishes each step so the UI has something true to show."""
    def step(name: str, total: int | None = None):
        _DB_JOB["step"] = name
        _DB_JOB["step_started"] = time.time()
        # Reset per-step progress, so a step with no countable work doesn't inherit the last
        # step's numbers and appear stuck at 100%.
        _DB_JOB["progress"] = {"done": 0, "total": total}

    def progress(done: int):
        p = _DB_JOB.get("progress") or {}
        p["done"] = done
        _DB_JOB["progress"] = p

    _DB_JOB.update({"state": "running", "started": time.time(), "finished": None,
                    "targets": list(targets), "vacuum": bool(vacuum), "deleted": {},
                    "error": None, "size_before": _db_size(), "size_after": None,
                    "reclaimed_bytes": None})
    # Force the archive attach for this job rather than trusting the lazy flag. Without it, a
    # proxy that has not archived anything yet skips `DELETE FROM arch.request_blobs` into a
    # swallowed except — the rows survive a "wipe requests", and the following VACUUM then has
    # nothing to reclaim while still reporting success.
    global _ARCHIVE_ACTIVE
    if Path(ARCHIVE_DB_PATH).exists():
        _ensure_archive_file()
        _ARCHIVE_ACTIVE = True
    counts: dict = {}
    try:
        step("counting rows")
        conn = db()
        try:
            for target, table in (("requests", "requests"), ("metrics", "system_metrics"),
                                  ("settings", "settings")):
                if target not in targets:
                    continue
                total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                counts[target] = total
                step(f"deleting {target}", total)
                _chunked_delete(conn, table, DB_RESET_CHUNK, DB_RESET_PAUSE_S, progress)
                _DB_JOB["deleted"] = dict(counts)
            # Blobs live in their own table now; wiping requests without them would strand
            # gigabytes of bodies with nothing pointing at them.
            if "requests" in targets:
                blob_total = conn.execute("SELECT COUNT(*) FROM request_blobs").fetchone()[0]
                arch_total = 0
                try:
                    arch_total = conn.execute(
                        "SELECT COUNT(*) FROM arch.request_blobs").fetchone()[0]
                except sqlite3.Error:
                    pass
                step("deleting request bodies", blob_total + arch_total)
                done_so_far = 0
                done_so_far += _chunked_delete(
                    conn, "request_blobs", DB_RESET_CHUNK, DB_RESET_PAUSE_S, progress)
                # Including the ones that were moved out — otherwise "wipe requests" leaves
                # gigabytes of bodies behind in the archive file.
                try:
                    base = done_so_far
                    _chunked_delete(conn, "arch.request_blobs", DB_RESET_CHUNK,
                                    DB_RESET_PAUSE_S, lambda n: progress(base + n))
                except sqlite3.Error:
                    pass
            conn.commit()
        finally:
            conn.close()
        if vacuum:
            # VACUUM must run outside any transaction, and rewrites the entire file. Both
            # files: emptying the rows leaves the archive's pages on its own free list, so
            # compacting only main would report space reclaimed while gigabytes stayed on disk.
            # One connection with the archive attached, rather than a second direct opener —
            # opening that file directly while other connections have it attached deadlocks.
            step("compacting the database")
            v = sqlite3.connect(DB_PATH, timeout=60.0)
            v.isolation_level = None
            try:
                v.execute("VACUUM main")
                if Path(ARCHIVE_DB_PATH).exists():
                    step("compacting the archive")
                    try:
                        v.execute("ATTACH DATABASE ? AS arch", (ARCHIVE_DB_PATH,))
                        v.execute("VACUUM arch")
                    except sqlite3.Error as e:
                        # Worth surfacing: the main file shrank and the archive did not.
                        _DB_JOB["archive_error"] = str(e)
                # In WAL mode the reclaimed pages sit in the write-ahead log until a
                # checkpoint, so the file on disk keeps its old size and the job reports
                # freeing nothing. TRUNCATE is what actually hands the space back.
                for schema in ("main", "arch"):
                    try:
                        v.execute(f"PRAGMA {schema}.wal_checkpoint(TRUNCATE)")
                    except sqlite3.Error:
                        pass
            finally:
                v.close()
        _overhead_samples.clear()
        _DBSTAT_CACHE.update({"ts": 0.0, "value": {}})
        _DB_JOB.update({"state": "done", "step": "finished"})
    except Exception as e:
        _DB_JOB.update({"state": "error", "step": "failed", "error": str(e)})
    finally:
        _DB_JOB["finished"] = time.time()
        _DB_JOB["size_after"] = _db_size()
        before, after = _DB_JOB.get("size_before") or 0, _DB_JOB["size_after"]
        _DB_JOB["reclaimed_bytes"] = max(0, before - after)
    return {"ok": _DB_JOB["state"] == "done", "deleted": counts, "vacuumed": bool(vacuum),
            "status": _db_job_status()}


@app.get("/__proxy/api/db/reset/status")
def db_reset_status():
    """Poll target while a wipe runs — the work happens in a thread, so the POST is still open."""
    return _db_job_status()


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
    if not _DB_JOB_LOCK.acquire(blocking=False):
        return JSONResponse({"error": "A database maintenance job is already running.",
                             "status": _db_job_status()}, status_code=409)
    try:
        # In a thread: DELETE and VACUUM on a multi-gigabyte file would otherwise hold the event
        # loop for minutes, which stops the proxy serving anything at all.
        return await asyncio.to_thread(_db_reset_job, targets, bool(payload.get("vacuum", True)))
    finally:
        _DB_JOB_LOCK.release()


def _backends_from_row(d: dict) -> dict:
    """Per-backend snapshots from a metrics row, newest storage first.

    Rows written before the registry have one column per backend; rows written since have a
    single blob. Reading both keeps history intact — and keeps every consumer's keys the same,
    which is why none of them had to change.
    """
    out = {name: {} for name in list(PROVIDERS) + list(SIDE_SERVICES)}
    for name in list(out):
        col = f"{name}_json"
        try:
            raw = d[col] if col in d.keys() else None
        except (TypeError, IndexError):
            raw = None
        if raw:
            try:
                out[name] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass
    raw = d["backends_json"] if "backends_json" in d.keys() else None
    if raw:
        try:
            out.update({k: v for k, v in (json.loads(raw) or {}).items()
                        if isinstance(v, dict)})
        except (json.JSONDecodeError, TypeError):
            pass
    return out


@app.get("/__proxy/api/system/now")
# Sync for the same reason as health and system_history: it reads the database, and the Live
# view polls it every few seconds.
def system_now():
    conn = db()
    row = conn.execute(
        """SELECT ts, cpu_pct, load_1m, mem_total_mb, mem_used_mb, mem_avail_mb,
                  gpu_json, ollama_json, lmstudio_json, comfyui_json, vllm_json,
                  llamacpp_json, backends_json
           FROM system_metrics ORDER BY ts DESC LIMIT 1"""
    ).fetchone()
    conn.close()
    if not row:
        return {"ts": None, "cpu_pct": None, "mem": None, "gpus": [], "ollama": {}, "lmstudio": {}, "comfyui": {}, "vllm": {},
                "llamacpp": {}}
    d = dict(row)
    return {
        "ts": d["ts"],
        "cpu_pct": d["cpu_pct"],
        "load_1m": d["load_1m"],
        "mem": {"total_mb": d["mem_total_mb"], "used_mb": d["mem_used_mb"], "avail_mb": d["mem_avail_mb"]},
        "gpus": json.loads(d["gpu_json"]) if d["gpu_json"] else [],
        **_backends_from_row(d),
        # So the UI can build its tab bar from the registry instead of a list it has to
        # be told to update — the omission that hid llama.cpp twice.
        "backends_meta": [{"name": n, "label": b.label, "kind": kind}
                          for kind, group in (("provider", PROVIDERS),
                                              ("side", SIDE_SERVICES))
                          for n, b in group.items()],
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
        f"""SELECT id, ts, conversation_id, turn_index, client_app, surface, client_ip, model, upstream, path,
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
            "surface": r["surface"],
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


# The catalogue asks five backends (one of them via `docker inspect`) what they have. Model
# pickers poll this endpoint, and answering a poll with a fan-out of subprocess calls is how
# a listing turns into load. Two seconds is short enough that starting a container still
# shows up promptly and long enough that a polling client costs one sweep.
_MODELS_CACHE: dict = {"ts": 0.0, "data": None}
_MODELS_CACHE_TTL_S = 2.0


@app.get("/v1/models")
async def list_models_enriched():
    """OpenAI /v1/models, but the union of EVERY backend the proxy fronts.

    A client pointed at the proxy previously saw only Ollama's catalogue — the vLLM models
    (the ones the router actually prefers for qwen traffic) were invisible, so a model
    picker in any OpenAI-style client couldn't select them. Now: Ollama's tags, vLLM's
    served models — including a stopped container's configured model, which the auto-loader
    boots on first request — plus whatever llama-server and LM Studio are serving. Deduped
    by id, first writer wins in that order; `owned_by` names the backend so the entries
    stay tellable apart.

    Context windows are filled in where a backend reports them (vLLM's max_model_len, LM
    Studio's loaded context, the cached snapshot for Ollama-served names), because clients
    auto-discover the window and fall back to a conservative default when the field is
    absent — often 128k, then they compact early against a model loaded much larger.
    All of it best-effort: an unreachable backend contributes nothing, never an error.
    """
    now = time.time()
    if _MODELS_CACHE["data"] is not None and now - _MODELS_CACHE["ts"] < _MODELS_CACHE_TTL_S:
        return JSONResponse(_MODELS_CACHE["data"])
    entries: dict[str, dict] = {}

    def add(mid, backend, ctx=None, state=None, created=None, loaded=None):
        if not mid or mid in entries:
            return
        e = {"id": mid, "object": "model", "created": created or 0, "owned_by": backend}
        if isinstance(ctx, int) and ctx > 0:
            e["context_length"] = e["max_context_length"] = e["max_model_len"] = ctx
        if state:
            e["proxy_state"] = state
        if loaded is not None:
            # Resident-in-memory vs on-disk, as a machine-readable boolean beside the
            # human-readable proxy_state — LM Studio's API makes the same distinction,
            # and "answers now" vs "costs a load first" is real routing information.
            e["loaded"] = bool(loaded)
        entries[mid] = e

    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        async def _ollama():
            r = await client.get(OLLAMA_URL + "/v1/models")
            return r.json() if r.status_code == 200 else {}

        async def _ollama_ps():
            r = await client.get(OLLAMA_URL + "/api/ps")
            return r.json() if r.status_code == 200 else {}

        results = await asyncio.gather(
            _ollama(), _ollama_ps(), _vllm_snapshot(client), _vllm_configs(),
            _llamacpp_snapshot(client), _lmstudio_snapshot(client),
            return_exceptions=True)
    odata, ops, vllm, vcfgs, lcpp, lms = (
        r if not isinstance(r, BaseException) else {} for r in results)

    resident = {str(m.get("name") or m.get("model") or "")
                for m in ((ops or {}).get("models") or []) if isinstance(m, dict)}
    for m in ((odata or {}).get("data") or []):
        if isinstance(m, dict):
            is_res = m.get("id") in resident
            add(m.get("id"), "ollama", created=m.get("created"), loaded=is_res,
                state="loaded" if is_res else "available — loads on request")
    for m in ((vllm or {}).get("available") or []):
        add(m.get("id"), "vllm", ctx=m.get("max_context_length"),
            state="loaded", loaded=True)
    # A twin that is configured but not up is still one request away: the auto-loader (or a
    # bench) boots it. Listing it is what makes it selectable at all.
    for c in (vcfgs if isinstance(vcfgs, list) else []):
        add(c.get("model"), "vllm", loaded=False,
            state=("stopped — loads on first request"
                   if str(c.get("state", "")).lower() != "running" else None))
    for m in ((lcpp or {}).get("available") or []):
        add(m.get("id"), "llamacpp", ctx=(lcpp or {}).get("n_ctx"),
            state="loaded", loaded=True)
    for m in ((lms or {}).get("available") or []):
        add(m.get("id"), "lmstudio", loaded=(m.get("state") == "loaded") or None,
            ctx=(m.get("loaded_context_length") or m.get("max_context_length")))

    # Ollama's own entries omit any context field; the cached LM Studio snapshot historically
    # filled it for names served through there. Best-effort, unchanged.
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
    for e in entries.values():
        if "context_length" not in e:
            ctx = ctxmap.get(_norm_model_id(e["id"]))
            if ctx:
                e["context_length"] = e["max_context_length"] = e["max_model_len"] = ctx

    payload = {"object": "list", "data": list(entries.values())}
    _MODELS_CACHE.update(ts=now, data=payload)
    return JSONResponse(payload)


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
def list_requests(request: Request, limit: int = 200, offset: int = 0, include_shadows: bool = False, client: str = "", ip: str = ""):  # sync → threadpool
    viewer = _client_ip(request)
    conn = db()
    conds, params = [], []
    if not include_shadows:
        conds.append("shadow_of IS NULL")
    if client:
        # Match either the detected app (e.g. "claude-code") or the raw client IP.
        conds.append("(client_app = ? OR client_ip = ?)")
        params += [client, client]
    if ip:
        # Separate from `client` and ANDed with it: one box can run several agents and one
        # agent name can come from several boxes, so "hermes" and "from 192.168.15.10" are
        # different questions and you often want both at once.
        conds.append("client_ip = ?")
        params.append(ip)
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    rows = conn.execute(
        f"""SELECT id, ts, method, path, model, is_stream, status, duration_ms, error,
                  prompt_tokens, completion_tokens, total_tokens, est_prompt_tokens, ttft_ms,
                  upstream, client_ip, client_app, surface, api_key_fp, gate_verdict, gate_rule, gate_reason,
                  shadow_of, has_images, reasoning_effort, thinking, thinking_src
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
    # Distinct source addresses for the filter dropdown. Deliberately just the address and a
    # count: a GROUP_CONCAT of the apps behind each one reads nicely but scans client_app across
    # every row of a multi-gigabyte table to decorate a select box. Not gated by the PII rules —
    # client_ip is not in the redaction set and the list already renders an address on every row.
    ip_rows = conn.execute(
        "SELECT client_ip AS ip, COUNT(*) AS count "
        "FROM requests WHERE client_ip IS NOT NULL"
        + ("" if include_shadows else " AND shadow_of IS NULL")
        + " GROUP BY client_ip ORDER BY count DESC"
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
            "clients": [dict(r) for r in client_rows],
            "ips": [dict(r) for r in ip_rows], "redacted": REDACT_PII_ENABLED}


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
                  GROUP_CONCAT(DISTINCT surface) AS surfaces,
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
    # Metadata only — NO bodies. The old version sent every turn's full request/response
    # blobs (multi-MB responses, ~2s to transfer). The dedicated conversation page lazy-loads
    # each turn's full content via /requests/{id} (fast, blob-split) on expand. This query
    # hits the lean `requests` table (no blob read at all).
    viewer = _client_ip(request)
    conn = db()
    rows = conn.execute(
        """SELECT id, ts, model, turn_index, prompt_tokens, completion_tokens, total_tokens,
                  est_prompt_tokens, duration_ms, ttft_ms, status, error, gate_verdict, gate_rule,
                  gate_reason, client_ip, client_app, surface, api_key_fp, is_stream, has_images, path, upstream
           FROM requests
           WHERE conversation_id = ?
           ORDER BY ts ASC""",
        (conv_id,),
    ).fetchall()
    conn.close()
    turns = []
    for r in rows:
        d = dict(r)
        _cpct, _cverdict = _cache_verdict(d.get("prompt_tokens"), d.get("est_prompt_tokens"),
                                          d.get("ttft_ms"), d.get("prompt_tokens"), d.get("upstream"))
        d["cache_pct"] = _cpct
        d["cache_verdict"] = _cverdict
        turns.append(_redact_row(d, viewer))
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
    known_extras = ["tool_aliases", "model_control", "pricing", "archive", "model_router", "ollama_options", "context_overflow_guard", "tool_pruner", "context_compressor", "protocol_bridge", "shadow_router", "tool_injector", "compaction_nudge", "request_priority", "request_dedup", "schema_validator", "hallucinated_tool", "tool_args_autofix", "xml_autofix", "tool_call_xml_retry"]
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


@app.get("/__proxy/api/archive")
def archive_status():
    """Sizes, and how much would move on the next sweep."""
    return _archive_status()


@app.post("/__proxy/api/archive/run")
async def archive_run(request: Request):
    """Move cold bodies now. `force` runs a sweep even while the rule is disabled, so the first
    one can be a deliberate act rather than something that starts on its own."""
    try:
        payload = await request.json() if (await request.body()) else {}
    except Exception:
        payload = {}
    res = await asyncio.to_thread(_archive_sweep, bool(payload.get("force")))
    return {**res, "status": _archive_status()}


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

    # Tool usage, from the names extracted when each row was written. This used to re-read 500
    # response bodies out of the blob table on every call — 19-27 MB of scattered reads,
    # measured at 5.1s against cold pages while the same rows without bodies took 0.001s. It
    # was the entire cost of this endpoint. Reading a small text column instead lifts the cap
    # too: every row in the window counts, not the most recent 500.
    _tool_cutoff = time.time() - 7 * 86400
    tool_rows = conn.execute(
        """SELECT model, tool_calls FROM requests
           WHERE ts > ? AND tool_calls IS NOT NULL""",
        (_tool_cutoff,),
    ).fetchall()
    by_tool: dict[str, dict] = {}
    for r in tool_rows:
        try:
            names = json.loads(r["tool_calls"]) or []
        except (json.JSONDecodeError, TypeError):
            continue
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
    # Anthropic batches its SSE output, so (duration − ttft) is transfer time, not decoding
    # time. Exclude it from the generation-rate distribution to keep that metric meaningful.
    # Processing-rate (prefill) and combined-throughput are still valid.
    # vLLM streams token-by-token like the other local engines and belongs here — it was
    # missing until 2026-07, which silently emptied the decode metric on any box that had
    # moved its daily driver to vLLM.
    GENERATION_RATE_UPSTREAMS = {n for n, p in PROVIDERS.items() if p.measures_decode}
    for r in tps_rows:
        ct = r["completion_tokens"] or 0
        pt = r["prompt_tokens"] or 0
        dur = r["duration_ms"] or 0
        ttft = r["ttft_ms"]
        ups = r["upstream"]
        if dur > 0 and ct > 0:
            combined_values.append(ct / (dur / 1000.0))
        # A decode rate needs a decode window worth dividing by. When a reply arrives in
        # essentially one chunk, (dur - ttft) can be a fraction of a millisecond and the rate
        # explodes into six figures — real rows, meaningless numbers, and they wreck p90/max
        # even though the median shrugs them off.
        if (ttft is not None and ttft > 0 and dur - ttft >= 100 and ct >= 8
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
    # Decode rate by prompt depth. A single median blends 80-token prompts with 67k-token ones,
    # and decode slows markedly as the KV cache grows — so the blended figure describes no real
    # request, and can't be compared against a number someone quotes from a short prompt.
    depth_buckets = [(0, 4000, "< 4K"), (4000, 16000, "4–16K"), (16000, 64000, "16–64K"),
                     (64000, 1 << 30, "> 64K")]
    by_depth = {label: [] for _lo, _hi, label in depth_buckets}
    for r in tps_rows:
        ct, dur, ttft, pt = (r["completion_tokens"] or 0), (r["duration_ms"] or 0), r["ttft_ms"], (r["prompt_tokens"] or 0)
        ups = r["upstream"]
        # Same guards as the overall generation rate, so the buckets can't disagree with it.
        if not (ttft and ttft > 0 and dur - ttft >= 100 and ct >= 8 and pt > 0):
            continue
        if ups not in GENERATION_RATE_UPSTREAMS and ups is not None:
            continue
        for lo, hi, label in depth_buckets:
            if lo <= pt < hi:
                by_depth[label].append(ct / ((dur - ttft) / 1000.0))
                break
    overall["throughput_by_depth"] = [
        {"bucket": label, **_dist(sorted(v))} for _lo, _hi, label in depth_buckets
        for v in [by_depth[label]] if v
    ]

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
            raw = _decode_b64_image(b64)
            d["images"].append({"index": i, "media_type": im.get("media_type") or "image/png",
                                "kind": "data", "size_bytes": len(b64) * 3 // 4,
                                # A header-only image renders as a broken thumbnail and looks
                                # like our fault. Say what actually happened to it.
                                "complete": _image_is_complete(raw) if raw else False})
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
    """Reconstruct and serve the idx-th image, from the full-fidelity images_data column.

    Two-step fetch: images_data and the PII gate first; the request_body — multi-megabyte
    for screenshot-heavy agent turns, and sitting behind the blob/archive join — is read
    ONLY for pre-capture rows that have no images_data. Selecting it unconditionally made
    every image click pay the full body's disk read and page cache just to throw it away.
    """
    conn = db()
    row = conn.execute("SELECT images_data, client_ip FROM requests_v WHERE id = ?",
                       (req_id,)).fetchone()
    if not row:
        conn.close()
        return JSONResponse({"error": "not found"}, status_code=404)
    # Same visibility gate as the request detail: only a viewer who may see this client's content.
    if not _can_view_pii(_client_ip(request), row["client_ip"]):
        conn.close()
        return JSONResponse({"error": "forbidden"}, status_code=403)
    if _load_images_data(row["images_data"]):
        conn.close()
        refs = _request_image_refs(row)
    else:
        row = conn.execute(
            "SELECT request_body, images_data, client_ip FROM requests_v WHERE id = ?",
            (req_id,)).fetchone()
        conn.close()
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
        # Served either way — a partial screenshot is still worth looking at — but the header
        # says so, so the viewer isn't left guessing why it renders half-drawn.
        return Response(content=raw, media_type=mt or "image/png",
                        headers={"Cache-Control": "private, max-age=300",
                                 "X-Image-Complete": "1" if _image_is_complete(raw) else "0"})
    return JSONResponse({"error": "image index out of range"}, status_code=404)


@app.get("/__proxy/api/requests/{req_id}/live")
async def request_live(req_id: str, request: Request):
    """Live output for an in-flight streaming request: the rolling reconstructed tail from the
    in-memory mirror (raw SSE chunks aren't retained; only the assembled text is). The detail
    view polls this while a request streams, then falls back to the persisted stream_chunks once
    it completes. Same PII gate as the detail: only a viewer allowed to see this client's data."""
    s = _LIVE_STREAMS.get(req_id)
    conn = db()
    row = conn.execute("SELECT client_ip, status FROM requests WHERE id = ?", (req_id,)).fetchone()
    conn.close()
    active = s is not None and (row is None or not row["status"])
    if row is not None and not _can_view_pii(_client_ip(request), row["client_ip"]):
        return {"active": active, "text": None, "redacted": True}
    if not s:
        return {"active": False, "text": None}
    return {
        "active": active,
        # False means no incremental output is coming — not that none has arrived yet. The
        # UI needs the difference to say so instead of showing an empty pane that looks broken.
        "streaming": bool(s.get("stream")),
        "text": s.get("text") or "",
        "prompt_tokens": s.get("prompt"),
        # Falls back to the chars/3.5 estimate so a non-streaming request still has something
        # truthful to show while it runs.
        "est_prompt_tokens": s.get("est_prompt"),
        "completion_tokens": s.get("completion"),
    }


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


@app.get("/__proxy/api/artifacts")
async def list_artifacts(request: Request, kind: str = "", q: str = "", conversation: str = "", limit: int = 300):
    """Artifacts (files/urls/images touched via tool calls), aggregated by path. PII-gated per the
    originator's subnet. Each row carries its most-recent event + request link for the content view."""
    viewer = _client_ip(request)
    where = ["1=1"]
    params: list = []
    if kind:
        where.append("kind = ?"); params.append(kind)
    if conversation:
        where.append("conversation_id = ?"); params.append(conversation)
    if q:
        where.append("path LIKE ?"); params.append("%" + q + "%")
    lim = max(1, min(2000, limit))
    conn = db()
    # 1) Aggregates per path — plain GROUP BY on the indexed `path`, no correlated subqueries
    #    (those were O(paths x rows) and got slow as the table grew).
    agg = conn.execute(
        f"""SELECT path, kind, COUNT(*) n, MIN(ts) first_ts, MAX(ts) last_ts,
                   GROUP_CONCAT(DISTINCT op) ops, COUNT(DISTINCT content_hash) versions
            FROM artifacts WHERE {' AND '.join(where)}
            GROUP BY path, kind ORDER BY last_ts DESC LIMIT ?""",
        (*params, lim)).fetchall()
    # 2) Latest-event fields (op / request / client_ip) for just those paths, via one windowed pass.
    paths = [r["path"] for r in agg]
    last: dict = {}
    if paths:
        ph = ",".join("?" for _ in paths)
        for r in conn.execute(
                f"""SELECT path, last_op, last_request, client_ip FROM (
                      SELECT path, op AS last_op, request_id AS last_request, client_ip,
                             ROW_NUMBER() OVER (PARTITION BY path ORDER BY ts DESC) rn
                      FROM artifacts WHERE path IN ({ph})
                    ) WHERE rn = 1""", paths).fetchall():
            last[r["path"]] = r
    conn.close()
    items = []
    for r in agg:
        lr = last.get(r["path"])
        if not _can_view_pii(viewer, lr["client_ip"] if lr else None):
            continue
        items.append({
            "path": r["path"], "kind": r["kind"], "n": r["n"],
            "ops": [o for o in (r["ops"] or "").split(",") if o],
            "versions": r["versions"], "first_ts": r["first_ts"], "last_ts": r["last_ts"],
            "last_op": lr["last_op"] if lr else None,
            "last_request": lr["last_request"] if lr else None,
        })
    # `enabled` lets the UI tell "nothing captured yet" apart from "capture is switched
    # off", which otherwise both render as an empty list.
    return {"items": items, "total": len(items), "enabled": ARTIFACTS_ENABLED}


@app.get("/__proxy/api/artifacts/conversations")
async def artifact_conversations(request: Request, limit: int = 150):
    """Conversations that touched files/urls, newest first — the top level of the Artifacts view.
    Each carries file/url counts + a preview (first user message) so you can recognize the session."""
    viewer = _client_ip(request)
    conn = db()
    rows = conn.execute(
        """SELECT conversation_id, COUNT(*) events, COUNT(DISTINCT path) paths,
                  COUNT(DISTINCT request_id) turns,
                  SUM(CASE WHEN kind='file' THEN 1 ELSE 0 END) files,
                  SUM(CASE WHEN kind='url' THEN 1 ELSE 0 END) urls,
                  SUM(CASE WHEN kind='image' THEN 1 ELSE 0 END) images,
                  SUM(CASE WHEN kind='skill' THEN 1 ELSE 0 END) skills,
                  MIN(ts) first_ts, MAX(ts) last_ts,
                  GROUP_CONCAT(DISTINCT client_ip) clients
           FROM artifacts WHERE conversation_id IS NOT NULL
           GROUP BY conversation_id ORDER BY last_ts DESC LIMIT ?""",
        (max(1, min(500, limit)),)).fetchall()
    labels = {r["conversation_id"]: r["label"] for r in
              conn.execute("SELECT conversation_id, label FROM conversation_labels").fetchall()}
    items = []
    for r in rows:
        ips = [ip.strip() for ip in (r["clients"] or "").split(",") if ip.strip()]
        if not any(_can_view_pii(viewer, ip) for ip in (ips or [None])):
            continue
        preview = None
        pv = conn.execute("SELECT request_body FROM requests_v WHERE conversation_id=? ORDER BY ts ASC LIMIT 1",
                          (r["conversation_id"],)).fetchone()
        if pv and pv["request_body"]:
            try:
                j = json.loads(pv["request_body"])
                fu = next((m for m in (j.get("messages") or [])
                           if isinstance(m, dict) and m.get("role") == "user"), None)
                if fu:
                    preview = _msg_text(fu)[:160]
            except (json.JSONDecodeError, TypeError):
                pass
        items.append({"conversation_id": r["conversation_id"], "label": labels.get(r["conversation_id"]),
                      "preview": preview, "events": r["events"], "paths": r["paths"], "turns": r["turns"],
                      "files": r["files"], "urls": r["urls"], "images": r["images"], "skills": r["skills"],
                      "first_ts": r["first_ts"], "last_ts": r["last_ts"]})
    conn.close()
    # `enabled` lets the UI tell "nothing captured yet" apart from "capture is switched
    # off", which otherwise both render as an empty list.
    return {"items": items, "total": len(items), "enabled": ARTIFACTS_ENABLED}


@app.get("/__proxy/api/artifacts/top")
async def top_artifacts(request: Request, kind: str = "file", limit: int = 30):
    """Most-active artifacts across all conversations — ranked by total touches, with edit/view
    breakdowns, so hot files surface without drilling. PII-gated per originator subnet."""
    viewer = _client_ip(request)
    where = ["1=1"]
    params: list = []
    if kind:
        where.append("kind = ?"); params.append(kind)
    lim = max(1, min(100, limit))
    conn = db()
    agg = conn.execute(
        f"""SELECT path, kind, COUNT(*) touches,
                   SUM(CASE WHEN op IN ('write','edit','create') THEN 1 ELSE 0 END) edits,
                   SUM(CASE WHEN op IN ('read','fetch','access','list','search') THEN 1 ELSE 0 END) views,
                   COUNT(DISTINCT content_hash) versions,
                   COUNT(DISTINCT conversation_id) convs, MAX(ts) last_ts
            FROM artifacts WHERE {' AND '.join(where)}
            GROUP BY path, kind ORDER BY touches DESC, last_ts DESC LIMIT ?""",
        (*params, lim)).fetchall()
    paths = [r["path"] for r in agg]
    ip_of: dict = {}
    if paths:
        ph = ",".join("?" for _ in paths)
        for r in conn.execute(
                f"""SELECT path, client_ip FROM (
                      SELECT path, client_ip, ROW_NUMBER() OVER (PARTITION BY path ORDER BY ts DESC) rn
                      FROM artifacts WHERE path IN ({ph})
                    ) WHERE rn = 1""", paths).fetchall():
            ip_of[r["path"]] = r["client_ip"]
    conn.close()
    items = []
    for r in agg:
        if not _can_view_pii(viewer, ip_of.get(r["path"])):
            continue
        items.append({"path": r["path"], "kind": r["kind"], "touches": r["touches"],
                      "edits": r["edits"], "views": r["views"], "versions": r["versions"],
                      "convs": r["convs"], "last_ts": r["last_ts"]})
    # `enabled` lets the UI tell "nothing captured yet" apart from "capture is switched
    # off", which otherwise both render as an empty list.
    return {"items": items, "total": len(items), "enabled": ARTIFACTS_ENABLED}


@app.get("/__proxy/api/artifacts/timeline")
async def artifact_timeline(request: Request, path: str, conversation: str = ""):
    """Event timeline for one file/url — each read/edit/fetch, newest first. Optionally scoped to one
    conversation (the turns within that session where it was touched)."""
    viewer = _client_ip(request)
    conn = db()
    if conversation:
        rows = conn.execute(
            "SELECT ts, op, tool_name, request_id, conversation_id, size_bytes, content_hash, client_ip, kind "
            "FROM artifacts WHERE path = ? AND conversation_id = ? ORDER BY ts DESC LIMIT 500",
            (path, conversation)).fetchall()
    else:
        rows = conn.execute(
            "SELECT ts, op, tool_name, request_id, conversation_id, size_bytes, content_hash, client_ip, kind "
            "FROM artifacts WHERE path = ? ORDER BY ts DESC LIMIT 500", (path,)).fetchall()
    conn.close()
    events = [{"ts": r["ts"], "op": r["op"], "tool": r["tool_name"], "request_id": r["request_id"],
               "conversation_id": r["conversation_id"], "size": r["size_bytes"],
               "content_hash": r["content_hash"], "kind": r["kind"]}
              for r in rows if _can_view_pii(viewer, r["client_ip"])]
    return {"path": path, "events": events}


@app.get("/__proxy/api/artifacts/content")
async def artifact_content(request: Request, request_id: str, path: str):
    """The content of one artifact, extracted inline from its request blob — file contents for a
    write/edit (from the tool-call args), or the returned text for a read/fetch (from the matching
    tool result). Lets the Artifacts view show content without jumping to the requests page."""
    conn = db()
    row = conn.execute("SELECT client_ip, request_body FROM requests_v WHERE id = ?", (request_id,)).fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "request not found"}, status_code=404)
    if not _can_view_pii(_client_ip(request), row["client_ip"]):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    try:
        body = json.loads(row["request_body"] or "{}")
    except (json.JSONDecodeError, TypeError):
        return {"content": None}
    tmap = _artifact_tool_map()
    tcid_to_path: dict = {}      # tool_call_id -> path, so read/fetch RESULTS can be matched back
    written = None               # content from a write/edit call (path is in the args)

    def _args(a):
        if isinstance(a, str):
            try:
                return json.loads(a)
            except (json.JSONDecodeError, ValueError):
                return {}
        return a if isinstance(a, dict) else {}

    for m in (body.get("messages") or []):
        if not isinstance(m, dict):
            continue
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            a = _args(fn.get("arguments"))
            spec = tmap.get(str(fn.get("name") or "").lower())
            if spec:
                _, _, path_arg, content_arg = spec
                if a.get(path_arg) == path:
                    if tc.get("id"):
                        tcid_to_path[tc["id"]] = path
                    if written is None and content_arg and isinstance(a.get(content_arg), str):
                        written = a[content_arg]
        if isinstance(m.get("content"), list):
            for blk in m["content"]:
                if isinstance(blk, dict) and blk.get("type") == "tool_use":
                    spec = tmap.get(str(blk.get("name") or "").lower())
                    a = blk.get("input") or {}
                    if spec and isinstance(a, dict):
                        _, _, path_arg, content_arg = spec
                        if a.get(path_arg) == path:
                            if blk.get("id"):
                                tcid_to_path[blk["id"]] = path
                            if written is None and content_arg and isinstance(a.get(content_arg), str):
                                written = a[content_arg]
    if written is not None:
        return {"content": written[:200000], "source": "write", "truncated": len(written) > 200000}
    # Else look for the tool RESULT of a read/fetch of this path.
    for m in (body.get("messages") or []):
        if not isinstance(m, dict):
            continue
        if m.get("role") == "tool" and m.get("tool_call_id") in tcid_to_path and isinstance(m.get("content"), str):
            c = m["content"]
            return {"content": c[:200000], "source": "result", "truncated": len(c) > 200000}
        if isinstance(m.get("content"), list):
            for blk in m["content"]:
                if isinstance(blk, dict) and blk.get("type") == "tool_result" and blk.get("tool_use_id") in tcid_to_path:
                    inner = blk.get("content")
                    if isinstance(inner, list):
                        inner = "\n".join(p.get("text", "") for p in inner if isinstance(p, dict) and p.get("text"))
                    if isinstance(inner, str):
                        return {"content": inner[:200000], "source": "result", "truncated": len(inner) > 200000}
    return {"content": None, "source": None}


@app.get("/__proxy/api/model-quirks")
async def get_model_quirks_endpoint():
    """Effective per-model quirks (baked-in defaults merged with model_quirks.json). Edit the file
    to override; changes hot-reload without a restart."""
    return {"quirks": load_model_quirks(), "path": MODEL_QUIRKS_FILE,
            "file_exists": Path(MODEL_QUIRKS_FILE).exists()}


@app.get("/__proxy/api/control/redact-pii")
async def control_redact_status():
    return {"redact_pii": REDACT_PII_ENABLED,
            "env_default": os.environ.get("PROXY_REDACT_PII", "1")}


@app.post("/__proxy/api/control/redact-pii")
async def control_redact_set(request: Request):
    """Toggle PII redaction at runtime, no restart. Body: {"on": bool} — `on` = redaction ENABLED.
    Admin/loopback only: turning redaction OFF exposes every client's request/response bodies and
    headers to any viewer, so it must not be flippable by arbitrary subnet users. This is an
    in-memory override; it reverts to the PROXY_REDACT_PII env default on the next restart."""
    ip = _client_ip(request)
    is_loopback = False
    try:
        is_loopback = bool(ip) and ipaddress.ip_address(ip).is_loopback
    except (ValueError, TypeError):
        pass
    if not (is_loopback or (ip and ip in ADMIN_IPS)):
        return JSONResponse({"error": "forbidden — admin IP or loopback only"}, status_code=403)
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    global REDACT_PII_ENABLED
    REDACT_PII_ENABLED = bool(payload.get("on", not REDACT_PII_ENABLED))
    return {"ok": True, "redact_pii": REDACT_PII_ENABLED}


# vLLM holds its weights for the life of the server process, so "unload" means "stop the
# server". When it runs as a container that is doable, and without it a box whose card is full
# of vLLM cannot benchmark anything else at all — which is most of the point of the bench.
#
# The container is found by the port it publishes rather than by name, so this keeps working when
# the container is renamed or recreated. Only a local VLLM_URL is eligible: a remote vLLM would
# be served by a container on another machine, and stopping a local one of the same name would be
# both wrong and destructive.
def _docker_bin() -> str | None:
    return os.environ.get("PROXY_DOCKER_BIN") or shutil.which("docker")


def _upstream_is_local(url: str) -> bool:
    try:
        host = (httpx.URL(url).host or "").lower()
    except Exception:
        return False
    return host in ("localhost", "127.0.0.1", "::1", "")


async def _run_cmd(args: list, timeout: float = 120.0, max_chars: int = 800,
                   keep_tail: bool = False, env: dict | None = None) -> tuple[int, str]:
    """No shell: arguments are passed as a list so a container or model name can never become
    part of a command.

    keep_tail decides which end of an oversized output survives. It matters for `docker logs`,
    where everything interesting is at the end — taking the head returned the oldest lines of
    the tail and cut off the progress output entirely.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            # Merged, not replaced: a bare env would drop PATH and the command would vanish.
            env=({**os.environ, **env} if env else None))
    except (OSError, ValueError) as e:
        return 127, str(e)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return 124, f"timed out after {timeout:.0f}s"
    text = (out or b"").decode("utf-8", "replace").strip()
    return proc.returncode or 0, (text[-max_chars:] if keep_tail else text[:max_chars])


async def _vllm_container() -> str | None:
    """The container publishing VLLM_URL's port, running or not."""
    docker = _docker_bin()
    if not docker or not _upstream_is_local(VLLM_URL):
        return None
    try:
        port = str(httpx.URL(VLLM_URL).port or 8000)
    except Exception:
        return None
    # {{.Ports}} is the LIVE port map and is empty once a container stops, so matching on it
    # only ever found vLLM while it was already running: the proxy could stop it and then could
    # not find it again to start it, which is precisely when you need it. PortBindings is the
    # container's configuration and survives being stopped — that is what "running or not" has
    # to mean.
    code, names = await _run_cmd([docker, "ps", "-a", "--format", "{{.Names}}"], 20.0,
                                 max_chars=4000)
    if code != 0 or not names.split():
        return None
    code, out = await _run_cmd(
        [docker, "inspect", "--format",
         "{{.Name}}	{{.State.Running}}	{{json .HostConfig.PortBindings}}"]
        + names.split(), 25.0, max_chars=8000)
    if code != 0:
        return None
    # Several containers can be *configured* for the same port — spark keeps qwen-vllm and
    # ornith-vllm side by side, only one running at a time. Returning the first match made
    # "stop vLLM" target whichever happened to sort first: the bench once stopped the
    # already-exited qwen-vllm, reported success, and left ornith-vllm's ~45 GB resident
    # while a 123B model tried to load beside it. The running one is the one that holds
    # memory, so it always wins; the first configured match is the fallback for "start it".
    first = None
    for line in out.splitlines():
        name, _, rest = line.partition("	")
        running, _, bindings = rest.partition("	")
        if f'"HostPort":"{port}"' in bindings.replace(" ", ""):
            clean = name.strip().lstrip("/")
            if running.strip() == "true":
                return clean
            first = first or clean
    return first


def _llamacpp_cfg() -> dict:
    return ((load_rules_config().get("model_control") or {}).get("llamacpp") or {})


def _llamacpp_override_path(unit: str) -> Path:
    return Path.home() / ".config" / "systemd" / "user" / f"{unit}.d" / "override.conf"


def _write_llamacpp_override(ctx: int, parallel: int) -> tuple[bool, str]:
    """Rewrite the unit's command line via a systemd drop-in.

    A drop-in rather than editing the unit: the original file stays the source of truth for
    everything else, and deleting one file restores it. The empty `ExecStart=` is required —
    without it systemd appends rather than replaces, and the server is launched twice.
    """
    cfg = _llamacpp_cfg()
    binary, model = (cfg.get("binary") or "").strip(), (cfg.get("model") or "").strip()
    if not binary or not model:
        return False, ("model_control.llamacpp needs `binary` and `model` set before the "
                       "context can be changed from here")
    args = [binary, "-m", model, "--ctx-size", str(int(ctx)), "--parallel", str(int(parallel)),
            "-ngl", str(int(cfg.get("ngl") or 999)),
            "--host", str(cfg.get("host") or "0.0.0.0"),
            "--port", str(int(cfg.get("port") or 8080))]
    args += [str(a) for a in (cfg.get("extra_args") or [])]
    path = _llamacpp_override_path(str(cfg.get("unit") or "llamacpp.service"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[Service]\nExecStart=\nExecStart=" + " ".join(args) + "\n")
    except OSError as e:
        return False, str(e)
    return True, " ".join(args)


async def _llamacpp_ready(timeout_s: float) -> bool:
    """Kept as a name for existing callers; the poll itself lives on the provider now."""
    return await PROVIDERS["llamacpp"].ready(timeout_s)


async def _vllm_ready(timeout_s: float) -> bool:
    """Kept as a name for existing callers; the poll itself lives on the provider now."""
    return await PROVIDERS["vllm"].ready(timeout_s)


def _lms_bin() -> str | None:
    for cand in (os.environ.get("PROXY_LMS_BIN"),
                 str(Path.home() / ".lmstudio" / "bin" / "lms"),
                 shutil.which("lms")):
        if cand and Path(cand).exists():
            return cand
    return None


def _lms_is_local() -> bool:
    """Only drive the CLI when it would act on the same LM Studio the proxy is proxying.
    A remote LMSTUDIO_URL with a local CLI would load models on the wrong machine."""
    try:
        host = (httpx.URL(LMSTUDIO_URL).host or "").lower()
    except Exception:
        return False
    return host in ("localhost", "127.0.0.1", "::1", "")


def _lms_available() -> tuple[bool, str]:
    if not _lms_is_local():
        return False, f"LM Studio is at {LMSTUDIO_URL}; the lms CLI only controls a local instance"
    if not _lms_bin():
        return False, "the lms CLI was not found (set PROXY_LMS_BIN to its path)"
    return True, ""


async def _lms_run(args: list, timeout: float = 600.0) -> tuple[int, str]:
    """Run the lms CLI. No shell, so a model name can't become a command; stdin is closed so a
    missing argument fails instead of dropping into the interactive picker and hanging."""
    bin_path = _lms_bin()
    if not bin_path:
        return 127, "lms CLI not found"
    proc = await asyncio.create_subprocess_exec(
        bin_path, *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return 124, f"timed out after {timeout:.0f}s"
    return proc.returncode or 0, (out or b"").decode("utf-8", "replace").strip()[:800]


VLLM_ARG_SUMMARY = ("--served-model-name", "--model", "--max-model-len",
                    "--gpu-memory-utilization", "--kv-cache-dtype", "--quantization")


async def _vllm_configs() -> list:
    """Every vLLM container on this host, running or not, with what it would serve.

    Configuration is expressed as containers rather than as launch arguments the proxy templates
    into `docker run`. Defining how a server starts is Docker's job and people already do it that
    way — spark had a stopped `ornith-vllm` beside the running `qwen-vllm` before any of this
    existed. The proxy only decides which one is up, which is the part a benchmark needs.
    """
    docker = _docker_bin()
    if not docker or not _upstream_is_local(VLLM_URL):
        return []
    code, out = await _run_cmd(
        [docker, "ps", "-a", "--format", "{{.Names}}	{{.Image}}	{{.State}}	{{.Ports}}"], 20.0)
    if code != 0:
        return []
    try:
        want_port = str(httpx.URL(VLLM_URL).port or 8000)
    except Exception:
        want_port = "8000"
    named = [str(n) for n in ((load_rules_config().get("model_control") or {})
                              .get("vllm_containers") or []) if n]
    out_list = []
    for line in out.splitlines():
        parts = line.split("	")
        if len(parts) < 3:
            continue
        name, image, state = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if named:
            # An explicit list means exactly these, so a container the proxy should not touch
            # cannot be swept up by a name that happens to contain "vllm".
            if name not in named:
                continue
        elif "vllm" not in image.lower() and "vllm" not in name.lower():
            continue
        rc, cmd_json = await _run_cmd([docker, "inspect", name, "--format", "{{json .Config.Cmd}}"], 15.0)
        args = []
        if rc == 0:
            try:
                args = json.loads(cmd_json) or []
            except (json.JSONDecodeError, TypeError):
                args = []
        flags = {}
        for i, a in enumerate(args):
            if a in VLLM_ARG_SUMMARY and i + 1 < len(args):
                flags[a.lstrip("-")] = args[i + 1]
        # Does it publish the port this proxy talks to? One that doesn't would start without
        # ever becoming reachable, which is worse than refusing.
        rc, ports_json = await _run_cmd(
            [docker, "inspect", name, "--format", "{{json .HostConfig.PortBindings}}"], 15.0)
        serves_port = False
        try:
            for binds in (json.loads(ports_json) or {}).values():
                for b in (binds or []):
                    if str(b.get("HostPort")) == want_port:
                        serves_port = True
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        mounts = []
        rc, mounts_json = await _run_cmd(
            [docker, "inspect", name, "--format", "{{json .Mounts}}"], 15.0)
        if rc == 0:
            try:
                mounts = [m.get("Source") for m in (json.loads(mounts_json) or []) if m.get("Source")]
            except (json.JSONDecodeError, TypeError):
                mounts = []
        out_list.append({
            "container": name, "image": image, "running": state.lower() == "running",
            "serves_port": serves_port,
            "model": flags.get("served-model-name") or flags.get("model"),
            "checkpoint": flags.get("model"),
            "mounts": mounts,
            # A bind mount makes --model "/model", which names nothing — fall back to the served
            # name and then to the host path that was mounted.
            "quant": _infer_quant(flags.get("model"), flags.get("served-model-name"),
                                  name, mounts),
            "max_model_len": flags.get("max-model-len"),
            "kv_cache_dtype": flags.get("kv-cache-dtype"),
            "prefix_caching": "--enable-prefix-caching" in args,
            # What a boot will actually claim: vLLM pre-allocates this fraction of total
            # memory at startup, which on unified memory is system RAM. The memory guards
            # price a start by this, NOT by checkpoint size — 43 GB of weights booting with
            # util 0.80 grabs ~97 GB, and the difference wedged the box once.
            "gpu_memory_utilization": flags.get("gpu-memory-utilization"),
            "args": " ".join(args),
        })
    return out_list


_SERVICE_ACTIONS = ("start", "stop", "restart")


def _service_def(name: str) -> dict | None:
    """Only units named in the config are reachable, so a request can never point systemctl at
    something that was never offered."""
    svcs = ((load_rules_config().get("model_control") or {}).get("services") or {})
    d = svcs.get(name)
    if not isinstance(d, dict) or not d.get("unit"):
        return None
    return {"name": name, "unit": str(d["unit"]),
            "scope": "user" if (d.get("scope") or "user") == "user" else "system"}


def _systemctl_args(svc: dict, *rest: str) -> list:
    args = ["systemctl"]
    if svc["scope"] == "user":
        args.append("--user")
    return args + list(rest) + [svc["unit"]]


def _systemctl_env(svc: dict) -> dict | None:
    """`systemctl --user` needs to find the user manager's bus, and this process is a *system*
    service — it inherits no XDG_RUNTIME_DIR, so systemctl reports "Failed to connect to bus:
    No medium found" and every call looks like the unit is broken. Point it at the runtime dir
    for our own uid. The manager still has to exist, which is what `loginctl enable-linger`
    guarantees across logouts and reboots.
    """
    if svc["scope"] != "user":
        return None
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime:
        uid = os.getuid() if hasattr(os, "getuid") else None
        if uid is None:
            return None
        runtime = f"/run/user/{uid}"
    return {"XDG_RUNTIME_DIR": runtime,
            "DBUS_SESSION_BUS_ADDRESS": f"unix:path={runtime}/bus"}


async def _service_state(svc: dict) -> dict:
    env = _systemctl_env(svc)
    code, out = await _run_cmd(_systemctl_args(svc, "is-active"), 10.0, max_chars=200, env=env)
    active = out.strip().splitlines()[0] if out.strip() else "unknown"
    code2, out2 = await _run_cmd(_systemctl_args(svc, "is-enabled"), 10.0, max_chars=200, env=env)
    enabled = out2.strip().splitlines()[0] if out2.strip() else "unknown"
    return {"name": svc["name"], "unit": svc["unit"], "scope": svc["scope"],
            "active": active, "enabled": enabled, "running": active == "active"}


@app.get("/__proxy/api/control/services")
async def list_services():
    svcs = ((load_rules_config().get("model_control") or {}).get("services") or {})
    out = []
    for name in svcs:
        svc = _service_def(name)
        if svc:
            out.append(await _service_state(svc))
    return {"services": out}


@app.post("/__proxy/api/control/services/{name}")
async def control_service(name: str, request: Request):
    """start / stop / restart a configured unit."""
    svc = _service_def(name)
    if not svc:
        return JSONResponse({"error": f"no service named {name!r} is configured"},
                            status_code=404)
    try:
        payload = await request.json() if (await request.body()) else {}
    except Exception:
        payload = {}
    action = (payload.get("action") or "").strip().lower()
    if action not in _SERVICE_ACTIONS:
        return JSONResponse({"error": f"action must be one of {', '.join(_SERVICE_ACTIONS)}"},
                            status_code=400)
    code, out = await _run_cmd(_systemctl_args(svc, action), 90.0, max_chars=1200,
                               keep_tail=True, env=_systemctl_env(svc))
    if code != 0:
        hint = ""
        if "No medium found" in out or "Failed to connect to bus" in out:
            hint = (" — the per-user systemd manager isn't running. "
                    "`sudo loginctl enable-linger " + (os.environ.get("USER") or "the proxy user")
                    + "` makes it start at boot and survive logout.")
        return JSONResponse({"error": f"systemctl {action} failed: {out[:400]}{hint}",
                             "state": await _service_state(svc)}, status_code=502)
    # A unit that reports active has been forked, not necessarily made ready — ComfyUI takes
    # a while to load. The caller polls the backend's own probe for readiness.
    return {"ok": True, "action": action, "state": await _service_state(svc)}


@app.post("/__proxy/api/control/ollama/update")
async def control_ollama_update():
    """Update Ollama in place, from the proxy.

    Refuses while a bench is running — the installer restarts the service, which would turn
    every in-flight cell into a page of 502s attributed to whatever model was being measured.
    Requires a root-owned updater script this deployment's sudoers allows without a password
    (see _ollama_update_cmd); the proxy's own user has no business running installers as root.
    """
    cmd = _ollama_update_cmd()
    if not cmd:
        return JSONResponse(
            {"error": "no updater configured — set model_control.ollama_update_cmd to a "
                      "command the proxy may run without a password, e.g. "
                      '["sudo", "-n", "/usr/local/sbin/ollama-update"]'},
            status_code=501)
    conn = db()
    busy = conn.execute(
        "SELECT COUNT(*) FROM bench_runs WHERE status IN ('running', 'pending')").fetchone()[0]
    conn.close()
    if busy:
        return JSONResponse(
            {"error": "a benchmark is running — updating restarts Ollama and would corrupt "
                      "it. Try again when it finishes."}, status_code=409)

    async def _version() -> str | None:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as c:
                r = await c.get(f"{OLLAMA_URL}/api/version")
                return (r.json() or {}).get("version") if r.status_code == 200 else None
        except (httpx.RequestError, ValueError):
            return None

    before = await _version()
    code, out = await _run_cmd(cmd, 900.0, max_chars=2000, keep_tail=True)
    if code != 0:
        return JSONResponse({"error": f"updater exited {code}: {out[:500]}",
                             "version_before": before}, status_code=502)
    # The installer restarts the service; give it a moment to come back before reporting.
    # If Ollama wasn't answering before the update there is nothing to wait out.
    after = None
    for i in range(30 if before else 1):
        after = await _version()
        if after:
            break
        await asyncio.sleep(2)
    _OLLAMA_LATEST["ts"] = 0.0          # the next snapshot re-checks what "latest" means now
    return {"ok": True, "version_before": before, "version_after": after,
            "changed": bool(before and after and before != after),
            "note": None if after else "updater succeeded but Ollama has not answered yet",
            "output": out[-800:]}


# Parameters a model's own defaults may sensibly carry. num_ctx is the headline: it is the
# ONLY per-model context lever that reaches /v1 traffic — Ollama's OpenAI endpoint ignores
# request-level num_ctx (verified live), and the env var is global and needs sudo.
_OLLAMA_PARAM_ALLOWED = {"num_ctx", "temperature", "top_p", "top_k", "min_p",
                         "repeat_penalty", "num_predict", "seed"}


async def _ollama_apply_params(model: str, params: dict) -> tuple:
    """Re-create the tag from itself with new default parameters. Blobs are shared, the
    change is instant, and a later edit simply overrides again."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as c:
        r = await c.post(f"{OLLAMA_URL}/api/create",
                         json={"model": model, "from": model,
                               "parameters": params, "stream": False})
        if r.status_code >= 400:
            return False, r.text[:300]
    return True, "ok"


@app.post("/__proxy/api/control/models/ollama-params")
async def ollama_model_params(request: Request):
    """Set a model's DEFAULT parameters (num_ctx etc.) by re-creating its tag in place.

    Guarded by the same KV arithmetic that protects benchmarks: a num_ctx whose cache
    cannot fit beside the weights is refused with the math, because that exact mistake
    OOM-killed the Ollama service in a loop once already. `force: true` overrides — with
    the numbers in hand, it is the operator's call."""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "JSON body required"}, status_code=400)
    model = str(payload.get("model") or "").strip()
    if not model:
        return JSONResponse({"error": "model is required"}, status_code=400)
    params = dict(payload.get("params") or {})
    if payload.get("num_ctx") is not None:
        params["num_ctx"] = payload["num_ctx"]
    bad = sorted(set(params) - _OLLAMA_PARAM_ALLOWED)
    if bad:
        return JSONResponse({"error": f"unsupported parameter(s): {', '.join(bad)} — "
                                      f"allowed: {', '.join(sorted(_OLLAMA_PARAM_ALLOWED))}"},
                            status_code=400)
    if not params:
        return JSONResponse({"error": "nothing to set"}, status_code=400)
    try:
        if "num_ctx" in params:
            params["num_ctx"] = int(params["num_ctx"])
            if not (256 <= params["num_ctx"] <= 4194304):
                raise ValueError
    except (TypeError, ValueError):
        return JSONResponse({"error": "num_ctx must be an integer >= 256"}, status_code=400)

    kv_note = None
    if "num_ctx" in params:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as c:
                show = (await c.post(f"{OLLAMA_URL}/api/show", json={"model": model})).json()
            info = show.get("model_info") or {}
            _slot, parallel = _bench_ollama_server_ctx()
            tokens = params["num_ctx"] * parallel
            kv_mb = _bench_ollama_kv_mb(info, tokens)
            size_mb = None
            async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as c:
                tags = (await c.get(f"{OLLAMA_URL}/api/tags")).json()
            for m in (tags.get("models") or []):
                if m.get("name") == model or m.get("model") == model:
                    size_mb = round((m.get("size") or 0) / 1048576) or None
            total_mb = (_mem_snapshot() or {}).get("total_mb")
            if kv_mb is not None and size_mb and total_mb:
                need = size_mb * _BENCH_FIT_OVERHEAD + kv_mb
                cap = total_mb - _BENCH_FIT_RESERVE_MB
                kv_note = (f"~{size_mb / 1024:.0f} GB weights + ~{kv_mb / 1024:.0f} GB KV at "
                           f"{params['num_ctx']:,} × {parallel} slots, "
                           f"of {cap / 1024:.0f} GB usable")
                if need > cap and not payload.get("force"):
                    return JSONResponse(
                        {"error": f"that context would OOM this box: {kv_note}. "
                                  f"Pass force:true to set it anyway.",
                         "kv_estimate_mb": round(kv_mb)}, status_code=400)
        except (httpx.RequestError, ValueError, KeyError):
            kv_note = "KV estimate unavailable — set with care"

    ok, detail = await _ollama_apply_params(model, params)
    if not ok:
        return JSONResponse({"error": f"ollama create failed: {detail}"}, status_code=502)
    return {"ok": True, "model": model, "applied": params, "kv_note": kv_note,
            "note": "defaults apply on the model's next load; a resident copy keeps its "
                    "old settings until it unloads"}


@app.get("/__proxy/api/control/models/vllm")
async def control_vllm_configs():
    """The vLLM configurations available to switch between. They contend for one port, so at
    most one can be up — which is exactly what makes them a menu."""
    return {"configs": await _vllm_configs(), "url": VLLM_URL}


# Pulling a model is tens of gigabytes over minutes to hours, so it cannot be a request that
# blocks until done. One at a time: two concurrent pulls of that size just thrash the disk and
# make both progress bars meaningless.
_PULL_JOB: dict = {"state": "idle"}
_PULL_LOCK = threading.Lock()
_PULL_TASK: asyncio.Task | None = None


# How long a finished pull keeps reporting itself. Long enough to read the outcome, short
# enough that a failure from an hour ago isn't still on the System tab.
_PULL_RESULT_TTL_S = 120.0


def _pull_status() -> dict:
    job = dict(_PULL_JOB)
    if job.get("started"):
        job["elapsed_s"] = round((job.get("finished") or time.time()) - job["started"], 1)
    total, done = job.get("total_bytes") or 0, job.get("completed_bytes") or 0
    job["percent"] = round(100.0 * done / total, 1) if total else None
    # A terminal state is news for a minute or two, then it is clutter. Without this a failed
    # pull sat on the System tab indefinitely, redrawn on every refresh, with no way to dismiss.
    fin = job.get("finished")
    if job.get("state") in ("done", "error", "cancelled") and fin:
        job["stale"] = (time.time() - fin) > _PULL_RESULT_TTL_S
    else:
        job["stale"] = False
    return job


def _pull_clear() -> bool:
    """Forget a finished pull. Refuses while one is running so a click can't hide live work."""
    if _PULL_JOB.get("state") == "running":
        return False
    _PULL_JOB.clear()
    _PULL_JOB.update({"state": "idle"})
    return True


async def _ollama_pull(model: str):
    """Stream Ollama's pull and keep a running total across layers.

    Ollama reports progress per layer digest, not for the download as a whole, so a naive
    reading restarts from zero on every layer. Summing the per-digest totals is the only way to
    get a figure that moves in one direction.
    """
    layers: dict = {}
    _PULL_JOB.update({"state": "running", "model": model, "started": time.time(),
                      "finished": None, "status": "starting", "error": None,
                      "total_bytes": 0, "completed_bytes": 0})
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as c:
            async with c.stream("POST", f"{OLLAMA_URL}/api/pull",
                                json={"model": model, "stream": True}) as r:
                if r.status_code >= 400:
                    body = (await r.aread()).decode("utf-8", "replace")[:300]
                    raise RuntimeError(f"ollama said {r.status_code}: {body}")
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("error"):
                        raise RuntimeError(str(ev["error"]))
                    _PULL_JOB["status"] = ev.get("status") or _PULL_JOB.get("status")
                    dg = ev.get("digest")
                    if dg and ev.get("total"):
                        layers[dg] = (ev.get("total") or 0, ev.get("completed") or 0)
                        _PULL_JOB["total_bytes"] = sum(t for t, _ in layers.values())
                        _PULL_JOB["completed_bytes"] = sum(c for _, c in layers.values())
        _PULL_JOB.update({"state": "done", "status": "success"})
    except asyncio.CancelledError:
        _PULL_JOB.update({"state": "cancelled", "status": "cancelled"})
        raise
    except Exception as e:
        _PULL_JOB.update({"state": "error", "error": f"{type(e).__name__}: {e}"})
    finally:
        _PULL_JOB["finished"] = time.time()
        if _PULL_LOCK.locked():
            _PULL_LOCK.release()


@app.get("/__proxy/api/control/models/pull")
def pull_status():
    return _pull_status()


@app.post("/__proxy/api/control/models/pull")
async def pull_model(request: Request):
    """Start downloading a model into Ollama. Returns immediately — poll for progress."""
    global _PULL_TASK
    try:
        payload = await request.json() if (await request.body()) else {}
    except Exception:
        payload = {}
    model = (payload.get("model") or "").strip()
    if not model:
        return JSONResponse({"error": "'model' is required"}, status_code=400)
    if not _PULL_LOCK.acquire(blocking=False):
        return JSONResponse({"error": f"already pulling {_PULL_JOB.get('model')!r}",
                             "status": _pull_status()}, status_code=409)
    # The lock is released by the task itself, in its finally.
    _PULL_TASK = asyncio.create_task(_ollama_pull(model))
    return {"ok": True, "model": model, "status": _pull_status()}


@app.post("/__proxy/api/control/models/pull/clear")
async def pull_clear():
    if not _pull_clear():
        return JSONResponse({"error": "a pull is running — cancel it first",
                             "status": _pull_status()}, status_code=409)
    return {"ok": True, "status": _pull_status()}


@app.post("/__proxy/api/control/models/pull/cancel")
async def pull_cancel():
    t = _PULL_TASK
    if t is None or t.done():
        return JSONResponse({"error": "no pull is running", "status": _pull_status()},
                            status_code=409)
    t.cancel()
    return {"ok": True, "status": _pull_status()}


@app.get("/__proxy/api/control/models/capabilities")
async def control_model_capabilities():
    """What can actually be loaded or unloaded, per backend, and why not when not. The UI needs
    this to avoid offering buttons that cannot work."""
    lms_ok, lms_why = _lms_available()
    container = await _vllm_container()
    return {
        "ollama": {"load": True, "unload": True, "how": "HTTP keep_alive"},
        "lmstudio": {"load": lms_ok, "unload": lms_ok,
                     "how": "lms CLI" if lms_ok else None, "reason": lms_why or None},
        "vllm": {"load": bool(container), "unload": bool(container),
                 "how": f"docker ({container})" if container else None,
                 "reason": None if container else (
                     "no container publishing this port was found"
                     if _upstream_is_local(VLLM_URL)
                     else f"vLLM is at {VLLM_URL}; only a local container can be controlled"),
                 "note": "vLLM holds one model for the life of the server, so this stops and "
                         "starts the container rather than swapping models"},
    }


@app.post("/__proxy/api/control/models/unload")
async def control_model_unload(request: Request):
    """Unload a model from Ollama, or all of them. Body: {"model": name} or {"all": true}.

    Ollama is the only backend that can be told this over HTTP: LM Studio exposes no unload
    endpoint (only its `lms` CLI, which requires being on the same host) and vLLM serves the one
    model its server was launched with. So this is deliberately Ollama-specific rather than a
    general abstraction that would be mostly holes.
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    upstream = (payload.get("upstream") or "ollama").strip().lower()
    if upstream == "lmstudio":
        ok, why = _lms_available()
        if not ok:
            return JSONResponse({"error": why}, status_code=501)
        args = ["unload", "-a"] if payload.get("all") else ["unload", (payload.get("model") or "").strip()]
        if not payload.get("all") and not args[1]:
            return JSONResponse({"error": "'model' is required (or 'all': true)"}, status_code=400)
        code, out = await _lms_run(args, timeout=120.0)
        if code != 0:
            return JSONResponse({"error": out or f"lms exited {code}"}, status_code=502)
        return {"ok": True, "unloaded": ["all"] if payload.get("all") else [args[1]], "output": out}
    if upstream == "vllm":
        container = await _vllm_container()
        if not container:
            return JSONResponse(
                {"error": "no local vLLM container publishing this port was found"},
                status_code=501)
        code, out = await _run_cmd([_docker_bin(), "stop", container], 180.0)
        if code != 0:
            return JSONResponse({"error": out or f"docker stop exited {code}"}, status_code=502)
        return {"ok": True, "stopped_container": container, "output": out,
                "note": "the vLLM server is stopped; start it again to serve that model"}
    if payload.get("all"):
        unloaded = await _bench_evict_ollama(keep="")
        return {"ok": True, "unloaded": unloaded}
    name = (payload.get("model") or "").strip()
    if not name:
        return JSONResponse({"error": "'model' is required (or 'all': true)"}, status_code=400)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as c:
            # keep_alive=0 with an empty prompt is Ollama's unload; there is no stop endpoint.
            r = await c.post(f"{OLLAMA_URL}/api/generate",
                             json={"model": name, "keep_alive": 0, "prompt": ""})
            if r.status_code >= 400:
                return JSONResponse({"error": f"ollama said {r.status_code}: {r.text[:200]}"},
                                    status_code=502)
    except httpx.RequestError as e:
        return JSONResponse({"error": f"ollama unreachable: {e}"}, status_code=502)
    return {"ok": True, "unloaded": [name]}


@app.post("/__proxy/api/control/models/load")
async def control_model_load(request: Request):
    """Make a backend serve a model, and wait until it does.
    Body: {"upstream": "ollama", "model": name, ...} — the rest is per-backend.

    Every mechanism lives on its Provider, so this handler only resolves the name and checks the
    one thing common to all of them. It used to branch four ways here, which is how llama.cpp
    ended up needing an exemption written into a condition rather than stated on the backend.
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    upstream = (payload.get("upstream") or "ollama").strip().lower()
    name = (payload.get("model") or "").strip()
    prov = PROVIDERS.get(upstream)
    if prov is None:
        return JSONResponse({"error": f"unknown upstream {upstream!r}",
                             "known": sorted(PROVIDERS)}, status_code=404)
    if not name and prov.requires_model_name:
        return JSONResponse({"error": "'model' is required"}, status_code=400)
    return await prov.load(payload, name)


@app.get("/__proxy/api/control/artifacts")
async def control_artifacts_status():
    """Whether artifact capture is on, plus how many rows already exist (so the UI can offer to
    purge what was collected before it was switched off)."""
    conn = db()
    try:
        n = conn.execute("SELECT COUNT(*) c FROM artifacts").fetchone()["c"]
    except sqlite3.Error:
        n = 0
    conn.close()
    return {"artifacts": ARTIFACTS_ENABLED,
            "env_default": os.environ.get("PROXY_ARTIFACTS", "1"),
            "stored_rows": n}


@app.post("/__proxy/api/control/artifacts")
async def control_artifacts_set(request: Request):
    """Turn artifact capture on or off at runtime. Body: {"on": bool, "purge": bool}.

    Admin/loopback only, same reasoning as the PII toggle: artifacts are a record of the paths
    in someone's source tree, so who may flip capture back on is not an arbitrary-viewer
    decision. `purge` additionally deletes everything already captured — irreversible, and only
    honored when turning capture OFF, so it can't be used as a drive-by wipe. This is an
    in-memory override; it reverts to the PROXY_ARTIFACTS env default on the next restart.
    """
    ip = _client_ip(request)
    is_loopback = False
    try:
        is_loopback = bool(ip) and ipaddress.ip_address(ip).is_loopback
    except (ValueError, TypeError):
        pass
    if not (is_loopback or (ip and ip in ADMIN_IPS)):
        return JSONResponse({"error": "forbidden — admin IP or loopback only"}, status_code=403)
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    global ARTIFACTS_ENABLED
    ARTIFACTS_ENABLED = bool(payload.get("on", not ARTIFACTS_ENABLED))
    purged = None
    if payload.get("purge") and not ARTIFACTS_ENABLED:
        conn = db()
        try:
            purged = conn.execute("SELECT COUNT(*) c FROM artifacts").fetchone()["c"]
            conn.execute("DELETE FROM artifacts")
            # Reset the watermark too, otherwise re-enabling capture would resume from the old
            # position and silently skip everything in between.
            set_setting("artifact_watermark", "0")
            conn.commit()
        except sqlite3.Error as e:
            conn.close()
            return JSONResponse({"error": f"purge failed: {e}"}, status_code=500)
        conn.close()
    return {"ok": True, "artifacts": ARTIFACTS_ENABLED, "purged_rows": purged}


# -------- Task queue REST API --------

_VALID_TASK_MODES = {"chat"}


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
    """Create a queued task. Body: {prompt, model?, schedule?}. mode is 'chat' (runs against
    OLLAMA_URL directly). schedule null = one-shot, else cron expr or 'every Nm/Nh/Nd'."""
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
    target_endpoint = None
    model = (payload.get("model") or "").strip() or None
    schedule = (payload.get("schedule") or "").strip() or None
    approval = None
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


@app.get("/__proxy/api/bench/suites")
async def bench_suites():
    """The graded task suites available to bench runs, with their task ids and case counts."""
    return {
        "suites": [
            {
                "name": name,
                "tasks": [{"id": t["id"], "entry": t["entry"], "cases": len(t["cases"]),
                           "lang": t.get("lang") or "python",
                           "desc": _BENCH_TASK_DESC.get(t["id"]) or "",
                           "available": _bench_lang_available(t.get("lang") or "python")}
                          for t in tasks],
                "task_count": len(tasks),
                "case_count": sum(len(t["cases"]) for t in tasks),
            }
            for name, tasks in _BENCH_SUITES.items()
        ],
        "thinking_modes": list(_BENCH_THINK_MODES),
    }


# How each backend behaves when you ask for a model that isn't currently resident. This is the
# difference the bench has to expose, because it decides whether picking a model is enough:
#   ollama    — pulls it into VRAM on demand; the warm-up absorbs the load
#   lmstudio  — JIT-loads a downloaded model on demand (when JIT loading is enabled); the
#               catalogue lists everything downloaded, with per-model state
#   vllm      — serves exactly the model(s) the server process was launched with. Nothing else
#               is reachable without restarting that server, so an unlisted model is unbenchable
# How each backend gets a model into memory, which decides what a benchmark can ask of it.
# "fixed" means the server serves exactly what it was launched with: the bench cannot swap
# models there, only measure the one that is up. llama.cpp is fixed for the same reason vLLM
# is — one model per process, chosen on the command line.
# Derived, not repeated: a backend added to the registry appears in the bench without
# anyone remembering to update a second list.
def _bench_load_modes() -> dict:
    return {name: p.load_mode for name, p in PROVIDERS.items()}


async def _bench_model_index() -> dict:
    """Map "<upstream>:<model>" → {model, upstream, loaded, ...}.

    Keyed by the PAIR, not the model name: the same name can be served by more than one backend
    (a GGUF in LM Studio and the same weights in vLLM), and keying by name alone let one silently
    overwrite the other, making it unbenchable. The pair is also what the UI selects, so there is
    no separate "which upstream" question to keep in sync.

    Carries the metadata a benchmark actually needs — quantization and context window — because
    'which quant was this?' is the first question asked of any result, and a prompt larger than
    the model's window silently produces an empty completion.
    """
    index: dict[str, dict] = {}

    def put(name, upstream, loaded, **extra):
        if not name:
            return
        key = f"{upstream}:{name}"
        rec = {"model": name, "upstream": upstream, "loaded": bool(loaded),
               "load_mode": _bench_load_modes().get(upstream, "unknown")}
        rec.update({k: v for k, v in extra.items() if v is not None})
        # A loaded entry always wins over a catalogue entry for the same pair.
        if key in index and index[key].get("loaded") and not loaded:
            return
        index[key] = rec

    try:
        # system_now is a sync handler (it reads the DB and must stay off the event loop).
        # `await`ing it raised TypeError into the blanket except below, silently skipping
        # every backend's enrichment — the bench showed 0 models for all of them but Ollama.
        sysinfo = await asyncio.to_thread(system_now)
        ollama = sysinfo.get("ollama") or {}
        for t in (ollama.get("tags") or []):
            if isinstance(t, dict):
                put(t.get("name"), "ollama", False,
                    size_mb=t.get("size_mb"), params=t.get("parameter_size"))
        for m in (ollama.get("ps") or []):
            if isinstance(m, dict):
                put(m.get("name") or m.get("model"), "ollama", True,
                    size_mb=round((m.get("size_vram") or m.get("size") or 0) / 1048576) or None)
        # NOTE: both snapshots expose `loaded` / `available`, NOT `models`. Reading the wrong
        # key here is why LM Studio and vLLM models never showed up in the bench picker.
        lms = sysinfo.get("lmstudio") or {}
        for m in (lms.get("available") or []):
            if isinstance(m, dict):
                put(m.get("id"), "lmstudio",
                    (m.get("state") or "").lower() in ("loaded", "ready"),
                    quant=m.get("quant"), arch=m.get("arch"),
                    max_context=m.get("max_context_length"),
                    loaded_context=m.get("loaded_context_length"),
                    size_mb=round((m.get("size_bytes") or 0) / 1048576) or None,
                    parallel=m.get("parallel"))
        vll = sysinfo.get("vllm") or {}
        vcfg = vll.get("config") or {}
        # A launch flag is authoritative where the checkpoint name is only a convention.
        v_quant_from_args = _infer_quant(vll.get("launch_args"))
        for m in (vll.get("available") or []):
            if isinstance(m, dict):
                put(m.get("id"), "vllm", (m.get("state") or "loaded").lower() == "loaded",
                    max_context=m.get("max_context_length"), arch=m.get("arch"),
                    quant=m.get("quant") or v_quant_from_args,
                    checkpoint=m.get("root"),
                    prefix_caching=(str(vcfg.get("enable_prefix_caching", "")).lower() == "true"
                                    if vcfg.get("enable_prefix_caching") is not None else None),
                    kv_cache_dtype=vcfg.get("cache_dtype"))
        # Containers that are configured but not running. vLLM serves exactly what its process
        # was launched with, so a stopped container looked identical to a model that does not
        # exist — the only way to benchmark it was to go to the System tab, start it by hand,
        # come back, and hope you picked the right one. The proxy already knows how to start it
        # and what it would serve, so offer it here and let the bench do it.
        try:
            # A container may be RUNNING but not yet serving — vLLM spends minutes loading
            # shards. Skipping every running container made a booting model vanish from the
            # index entirely: not loaded, not startable, so its cells failed preflight in one
            # second ("does not serve — it has <the other twin>") instead of using the same
            # start-and-wait path a stopped container gets. Only a container whose model the
            # live probe actually answered for is excluded here.
            _live_ready = {m.get("id") for m in (vll.get("available") or [])}
            for c in await _vllm_configs():
                if not c.get("serves_port") or not c.get("model"):
                    continue
                if c.get("running") and c["model"] in _live_ready:
                    continue
                put(c["model"], "vllm", False, startable=True, container=c["container"],
                    quant=c.get("quant"), checkpoint=c.get("checkpoint"),
                    max_context=int(c["max_model_len"]) if c.get("max_model_len") else None,
                    kv_cache_dtype=c.get("kv_cache_dtype"),
                    prefix_caching=c.get("prefix_caching"))
        except Exception as e:
            # Not silent: a failure here makes stopped containers vanish from the picker, which
            # looks exactly like "that model does not exist".
            try:
                print(f"[bench] vLLM container scan failed: {type(e).__name__}: {e}")
            except Exception:
                pass
        lcp = sysinfo.get("llamacpp") or {}
        for m in (lcp.get("available") or []):
            if isinstance(m, dict):
                # Identity is the checkpoint PATH, not the snapshot's display id. The stopped
                # fallback below keys on the configured path, and sweeps submit that path — so
                # when the display id was cleaned for the System tab, a running llama.cpp
                # stopped matching its own cells: preflight said "does not serve <path> — it
                # has <clean name>" and eleven server-context cells died in zero seconds each.
                put(lcp.get("model_path") or m.get("id"), "llamacpp", True,
                    arch=m.get("arch"), quant=m.get("quant"),
                    max_context=m.get("max_context_length") or lcp.get("n_ctx"),
                    loaded_context=lcp.get("n_ctx"),
                    checkpoint=lcp.get("model_path"),
                    parallel=lcp.get("parallel"))
        # vLLM reports no weight size — its OpenAI /v1/models says nothing about bytes on disk,
        # so every vLLM row rendered a blank in the report's size column and the memory chart
        # could not describe the one backend where memory is tightest. Size the checkpoint in
        # the Hugging Face cache instead, which is where vLLM put it.
        for rec in index.values():
            if rec.get("upstream") != "vllm" or rec.get("size_mb"):
                continue
            ckpt = rec.get("checkpoint")
            if not ckpt:
                continue
            try:
                total = _hf_cache_bytes(ckpt)
                if total:
                    rec["size_mb"] = round(total / 1048576)
            except OSError:
                pass          # a missing cache is a missing metric, not a broken index
        # A stopped llama.cpp offers no models at all, so once the bench stopped it to make room
        # for vLLM, its own cells could not resolve a model and failed preflight. The unit is
        # configured with exactly one model; offer that, the same way a stopped container is.
        lc_path = (_llamacpp_cfg() or {}).get("model")
        if lc_path:
            try:
                # Split GGUFs: the first shard names the set, so size the whole set.
                pat = re.sub(r"-\d{4,5}-of-\d{4,5}\.gguf$", "-*.gguf", lc_path)
                shards = sorted(Path(lc_path).parent.glob(Path(pat).name)) or [Path(lc_path)]
                total = sum(f.stat().st_size for f in shards if f.exists())
                for rec in index.values():
                    if rec.get("upstream") == "llamacpp" and total:
                        rec.setdefault("size_mb", round(total / 1048576))
            except OSError:
                pass
        if not (lcp.get("reachable")):
            lc = _llamacpp_cfg()
            if lc.get("model"):
                put(lc["model"], "llamacpp", False, startable=True,
                    quant=_infer_quant(lc.get("model")), checkpoint=lc.get("model"))
    except Exception as e:
        # This once swallowed an UnboundLocalError and returned an empty picker: every backend
        # showed zero models and the bench looked like it had lost the box. Falling back is
        # right; doing it silently is not.
        try:
            print(f"[bench] model index build failed: {type(e).__name__}: {e}")
        except Exception:
            pass
    if index:
        _bench_annotate_fit(index)
        _bench_annotate_engines(index)
        try:
            await _bench_annotate_caps(index, app.state.metrics_client)
        except Exception as e:
            print(f"[bench] capability probe failed: {type(e).__name__}: {e}")
        return index
    # The snapshot comes from the background metrics collector, which hasn't necessarily run
    # yet on a freshly-started proxy — and a bench submitted in that window would silently get
    # no upstream inference. Probe the upstreams directly rather than depend on that timing.
    async def _probe(client, url, parse):
        try:
            r = await client.get(url)
            if r.status_code == 200:
                parse(r.json())
        except Exception:
            pass

    def _ollama_tags(j):
        for m in (j.get("models") or []):
            if isinstance(m, dict):
                put(m.get("name"), "ollama", False)

    def _ollama_ps(j):
        for m in (j.get("models") or []):
            if isinstance(m, dict):
                put(m.get("name") or m.get("model"), "ollama", True)

    def _lms_native(j):
        # LM Studio's own API reports every downloaded model plus its state; the OpenAI-compat
        # /v1/models can't distinguish loaded from merely present.
        entries = j.get("data") if isinstance(j, dict) else j
        for m in (entries or []):
            if isinstance(m, dict):
                put(m.get("id"), "lmstudio",
                    (m.get("state") or "").lower() in ("loaded", "ready"),
                    quant=m.get("quantization"), arch=m.get("arch"),
                    max_context=m.get("max_context_length"),
                    loaded_context=m.get("loaded_context_length"))

    def _openai_models(label):
        def parse(j):
            for m in (j.get("data") or []):
                if isinstance(m, dict):
                    put(m.get("id"), label, True, max_context=m.get("max_model_len"))
        return parse

    try:
        # Short: this path runs when the metrics snapshot is empty, and it gates the bench model
        # picker. An unconfigured upstream is usually an instant connection refusal, but a
        # firewalled host would hang — and the picker sitting on "loading…" reads as broken.
        async with httpx.AsyncClient(timeout=httpx.Timeout(1.5)) as c:
            await asyncio.gather(
                _probe(c, f"{OLLAMA_URL}/api/tags", _ollama_tags),
                _probe(c, f"{LMSTUDIO_URL}/api/v0/models", _lms_native),
                _probe(c, f"{VLLM_URL}/v1/models", _openai_models("vllm")),
            )
            # After tags, so a loaded model's entry wins over its catalogue entry.
            await _probe(c, f"{OLLAMA_URL}/api/ps", _ollama_ps)
            if not any(v["upstream"] == "lmstudio" for v in index.values()):
                # Older LM Studio builds predate /api/v0 — fall back, losing load state.
                await _probe(c, f"{LMSTUDIO_URL}/v1/models", _openai_models("lmstudio"))
    except Exception:
        pass
    # Same annotation on the cold-start path, or a model would be selectable on a freshly
    # restarted proxy and refused a minute later.
    _bench_annotate_fit(index)
    _bench_annotate_engines(index)
    return index


# Weights are the floor, not the total: a runtime also needs a KV cache, activations and its
# own footprint, and the OS needs room left to not fall over. Deliberately generous — refusing
# a model that would have fit is a worse failure than letting a borderline one try and fail
# with a real error.
_BENCH_FIT_OVERHEAD = 1.15
_BENCH_FIT_RESERVE_MB = 4096


# Capabilities are a property of the model tag, so they never change once read. Cached for the
# process: /api/show is a call per model, and the picker would otherwise make fifteen of them
# every time someone opened the bench tab.
_OLLAMA_CAPS: dict = {}


async def _ollama_capabilities(client, name: str):
    """What Ollama says a model can do, or None if it would not say."""
    if name in _OLLAMA_CAPS:
        return _OLLAMA_CAPS[name]
    caps = None
    try:
        r = await client.post(f"{OLLAMA_URL}/api/show", json={"model": name})
        if r.status_code == 200:
            got = (r.json() or {}).get("capabilities")
            caps = [str(c) for c in got] if isinstance(got, list) else None
    except (httpx.RequestError, ValueError):
        caps = None
    _OLLAMA_CAPS[name] = caps
    return caps


def _hf_cache_bytes(checkpoint: str) -> int:
    """Bytes on disk for a Hugging Face repo id, or a local checkpoint path.

    vLLM serves whatever `--model` named: either a path, or a repo id it downloaded into the
    cache as `models--org--name`. Only weight files are counted — a repo also carries configs,
    tokenizers and often the original unquantised safetensors under a different revision, and
    summing the directory would report a number nobody is holding in memory.
    """
    p = Path(checkpoint)
    if p.is_dir():
        roots = [p]
    else:
        cache = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
        hub = cache / "hub" if (cache / "hub").is_dir() else cache
        slug = "models--" + checkpoint.strip("/").replace("/", "--")
        root = hub / slug / "snapshots"
        if not root.is_dir():
            return 0
        # Newest revision only: an updated repo leaves the previous snapshot in place, and
        # adding them together would double the model's apparent size.
        revs = sorted((d for d in root.iterdir() if d.is_dir()),
                      key=lambda d: d.stat().st_mtime, reverse=True)
        roots = revs[:1]
    total = 0
    for r in roots:
        for f in r.rglob("*"):
            if f.suffix.lower() in (".safetensors", ".bin", ".gguf", ".pt") and f.is_file():
                try:
                    total += f.stat().st_size      # follows the symlink into blobs/
                except OSError:
                    pass
    return total


def _bench_fits(size_mb: float | None, total_mb: float | None) -> bool | None:
    """Can this model be resident on this box at all? None when the size is unknown.

    Unknown must not mean "no". vLLM checkpoints live inside a container and cannot be sized
    from here, and silently hiding a model because it could not be measured would be
    indistinguishable from the model not existing.
    """
    if not size_mb or not total_mb:
        return None
    return (size_mb * _BENCH_FIT_OVERHEAD) <= (total_mb - _BENCH_FIT_RESERVE_MB)


async def _bench_annotate_caps(index: dict, client) -> None:
    """Mark what cannot answer a chat request at all, and what merely will not answer it well.

    A model with no `completion` capability — an embedding model — cannot produce a measurement
    under any configuration, so it is refused the same way a model that does not fit is.

    `vision` is recorded and never acted on. It means the model also accepts images, not that
    it is weak at text: gemma3, gemma4, llama4 and qwen3.6 all report it on this box and all
    code perfectly well. There is no reliable signal separating "vision-first" from "multimodal
    and strong at text", so that judgement stays with the person choosing the models.
    """
    names = [r.get("model") for r in index.values()
             if r.get("upstream") == "ollama" and r.get("model")]
    if not names:
        return
    caps = await asyncio.gather(*(_ollama_capabilities(client, n) for n in names),
                                return_exceptions=True)
    by_name = {n: (c if isinstance(c, list) else None) for n, c in zip(names, caps)}
    for rec in index.values():
        got = by_name.get(rec.get("model"))
        if got is None:
            continue          # Ollama would not say; an unknown is not a refusal
        rec["capabilities"] = got
        if "vision" in got:
            rec["vision"] = True
        if "completion" not in got:
            rec["benchable"] = False
            rec["bench_detail"] = ("cannot answer a chat request — Ollama reports "
                                   f"{', '.join(got) or 'no capabilities'}")


def _bench_annotate_engines(index: dict) -> None:
    """Mark models the box can serve through more than one engine.

    That pairing is the only controlled way to ask "is vLLM faster than Ollama here" — same
    weights, everything else held — and until it is pointed out it is findable only by noticing
    that two rows of the picker happen to be the same checkpoint.
    """
    by_id: dict = {}
    for rec in index.values():
        ident = _bench_model_identity(rec.get("model") or "")
        if ident:
            by_id.setdefault(ident, set()).add(rec.get("upstream"))
    for rec in index.values():
        ups = by_id.get(_bench_model_identity(rec.get("model") or "")) or set()
        if len(ups) > 1:
            rec["also_on"] = sorted(u for u in ups if u and u != rec.get("upstream"))


def _bench_annotate_fit(index: dict) -> None:
    """Mark entries the box cannot hold. Checked against total memory, not free memory: the
    bench evicts everything before it measures, so what matters is capacity, not what happens
    to be resident while someone is reading the picker."""
    total_mb = (_mem_snapshot() or {}).get("total_mb")
    for rec in index.values():
        fits = _bench_fits(rec.get("size_mb"), total_mb)
        if fits is None:
            continue
        rec["fits"] = fits
        if not fits:
            rec["fit_detail"] = (
                f"{rec['size_mb'] / 1024:.1f} GB of weights needs about "
                f"{rec['size_mb'] * _BENCH_FIT_OVERHEAD / 1024:.0f} GB resident; "
                f"this machine has {total_mb / 1024:.0f} GB")


def _bench_resolve_model(index: dict, model: str, upstream: str = "") -> dict:
    """Find a model's entry. With an upstream, matches that exact pair; without one, falls back
    to the first backend serving that name (preferring a loaded copy) so an old-style submission
    that names only a model still resolves."""
    if upstream:
        return index.get(f"{upstream}:{model}") or {}
    matches = [v for v in index.values() if v.get("model") == model]
    if not matches:
        return {}
    return next((m for m in matches if m.get("loaded")), matches[0])


@app.get("/__proxy/api/bench/models")
async def bench_models():
    """Every model the proxy can reach, with the upstream that serves it and whether it's
    currently loaded. `loaded` matters for benchmarking: an unloaded Ollama model is pulled into
    VRAM by the first request, so without a warm-up that load time lands inside the first
    measurement."""
    index = await _bench_model_index()
    # What each model has historically cost per cell, so a sweep can be priced before it runs
    # rather than discovered to be a three-hour job an hour in.
    conn = db()
    try:
        cost = _bench_history_cost(conn)
        measured = _bench_completed_cells(conn)
    finally:
        conn.close()
    typical = (sorted(cost.values())[len(cost) // 2] if cost else None)
    for meta in index.values():
        meta["est_cell_s"] = cost.get(meta.get("model")) or typical
        meta["measured_before"] = meta.get("model") in {
            json.loads(k).get("model") for k in measured}
    items = [dict(meta, key=key) for key, meta in index.items()]
    items.sort(key=lambda i: (i["upstream"], not i["loaded"], i["model"]))
    # Per-backend rollup: what's reachable, how much of it is resident, and whether unloaded
    # models there can be used at all. This is what the upstream picker is built from.
    upstreams = []
    for name, mode in _bench_load_modes().items():
        mine = [i for i in items if i["upstream"] == name]
        upstreams.append({
            "upstream": name,
            "load_mode": mode,
            "total": len(mine),
            "loaded": sum(1 for i in mine if i["loaded"]),
            "reachable": bool(mine),
            # Lets the UI say "will be resized" instead of "expect empty completions" for the
            # backends where that is now true.
            "resizable_context": bool(
                getattr(PROVIDERS.get(name), "resizable_context", False)),
        })
    out: dict = {
        "items": items,
        "upstreams": upstreams,
        "load_modes": _bench_load_modes(),
        # Kept for older clients / convenience.
        "ollama": {
            "loaded": [i["model"] for i in items if i["upstream"] == "ollama" and i["loaded"]],
            "available": [i["model"] for i in items if i["upstream"] == "ollama" and not i["loaded"]],
        },
        "lmstudio": [i["model"] for i in items if i["upstream"] == "lmstudio"],
        "vllm": [i["model"] for i in items if i["upstream"] == "vllm"],
    }
    return out


@app.post("/__proxy/api/bench/run")
async def bench_run(request: Request):
    """Queue a benchmark. Body: {model | models[], runs?, max_tokens?, prompt_tokens?,
    concurrency?, randomize?, exclusive?, drain_seconds?, thinking?, temperature?, top_p?,
    top_k?, min_p?, seed?, extra_body?, upstream?, bypass_router?, no_nudge?, suite?,
    grade_timeout?, quiesce?}.

    models/prompt_tokens/thinking/temperature accept lists, which expands the submission into a
    matrix: one child run per combination, executed serially. Returns the bench id; poll
    /api/bench/runs/{id} for progress and results.
    """
    try:
        payload = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"invalid JSON: {e}"}, status_code=400)

    # One bench at a time, refused rather than queued. _BENCH_SEM already serialises execution,
    # so a second submission would run eventually — but "eventually" is the problem: a bench
    # takes the box exclusively, resizes context windows, evicts residents and gates other
    # traffic. A run that starts an hour later measures a machine configured by whatever ran
    # before it, and nothing in its results would say so. Stale rows cannot wedge this: a
    # restart marks everything pending/running as interrupted on the way up.
    conn = db()
    busy = conn.execute(
        "SELECT COALESCE(parent_id, id) root FROM bench_runs "
        "WHERE status IN ('pending','running') ORDER BY ts LIMIT 1").fetchone()
    active = conn.execute(
        "SELECT id, model, label, status, ts, progress, progress_total FROM bench_runs "
        "WHERE id=?", (busy["root"],)).fetchone() if busy else None
    conn.close()
    if busy:
        d = dict(active) if active else {"id": busy["root"]}
        return JSONResponse(
            {"error": f"a bench is already running ({d.get('label') or d.get('model') or d['id']})"
                      f" — wait for it to finish",
             "running": {k: d.get(k) for k in
                         ("id", "model", "label", "status", "progress", "progress_total")},
             "started_ago_s": round(time.time() - (d.get("ts") or time.time()))},
            status_code=409)

    # A model is identified by (name, upstream). Accepted forms, in order of preference:
    #   [{"model": "x", "upstream": "vllm"}]   explicit — what the UI sends
    #   ["vllm:x"]                              the index key
    #   ["x"]                                   name only; the upstream is inferred later
    def _parse_model(entry) -> dict | None:
        if isinstance(entry, dict):
            name = str(entry.get("model") or "").strip()
            up = str(entry.get("upstream") or "").strip().lower()
        else:
            raw = str(entry or "").strip()
            name, up = raw, ""
            head = raw.split(":", 1)[0].lower()
            if head in PROVIDERS and ":" in raw:
                up, name = head, raw.split(":", 1)[1]
        if not name:
            return None
        if up and up not in PROVIDERS:
            return None
        return {"model": name, "upstream": up}

    raw_models = payload.get("models")
    models = []
    if isinstance(raw_models, list):
        models = [m for m in (_parse_model(e) for e in raw_models) if m]
    if not models and payload.get("model"):
        one = _parse_model(payload.get("model"))
        if one:
            models = [one]
    if not models:
        return JSONResponse({"error": "'model' or 'models' is required"}, status_code=400)

    thinking = payload.get("thinking")
    for t in (thinking if isinstance(thinking, list) else [thinking]):
        if t is not None and str(t) not in _BENCH_THINK_MODES:
            return JSONResponse(
                {"error": f"thinking must be one of {list(_BENCH_THINK_MODES)}"}, status_code=400)
    upstream = str(payload.get("upstream") or "").strip().lower()
    if upstream and upstream not in ("ollama", "lmstudio", "vllm"):
        return JSONResponse({"error": "upstream must be ollama, lmstudio or vllm"}, status_code=400)
    cache = payload.get("cache")
    for c in (cache if isinstance(cache, list) else [cache]):
        if c is not None and str(c) not in ("cold", "cached"):
            return JSONResponse({"error": "cache must be 'cold' or 'cached'"}, status_code=400)
    suite = str(payload.get("suite") or "").strip()
    if suite and suite not in _BENCH_SUITES:
        return JSONResponse(
            {"error": f"unknown suite {suite!r}; available: {list(_BENCH_SUITES)}"}, status_code=400)

    def keep(key, cast=None):
        v = payload.get(key)
        if v is None:
            return None
        if isinstance(v, list):
            return v
        return cast(v) if cast else v

    # Graded runs default to a budget the long tasks can actually finish in. 512 cut verbose
    # models off mid-function on semver_cmp-class tasks — the failure examples then read as
    # broken code when the truth was an empty tank. An explicit max_tokens still wins.
    _mt_default = 1024 if str(payload.get("suite") or "").strip() else 256
    config = {
        "runs": int(payload.get("runs", 5) or 5),
        "max_tokens": int(payload.get("max_tokens", _mt_default) or _mt_default),
        "prompt_tokens": keep("prompt_tokens") if isinstance(payload.get("prompt_tokens"), list)
                         else int(payload.get("prompt_tokens", 0) or 0),
        "concurrency": (payload["concurrency"] if isinstance(payload.get("concurrency"), list)
                        else int(payload.get("concurrency", 1) or 1)),
        "randomize": bool(payload.get("randomize", False)),
        # Exclusive by default: a benchmark sharing the box with live traffic measures
        # contention, disturbs the humans, and is defenseless against their restarts —
        # all three happened. Callers must opt OUT, knowingly.
        "exclusive": bool(payload.get("exclusive", True)),
        "drain_seconds": float(payload.get("drain_seconds", 5.0) or 0.0),
        "thinking": thinking if thinking is not None else "auto",
        "bypass_router": bool(payload.get("bypass_router", False)),
        "no_nudge": bool(payload.get("no_nudge", False)),
        "quiesce": bool(payload.get("quiesce", False)),
        # Unload other Ollama models before each cell, without gating traffic. Only Ollama can
        # be controlled this way; see _bench_evict_ollama.
        "evict_others": bool(payload.get("evict_others", False)),
        # On by default: an unloaded model's load time would otherwise land in the first
        # measured request. Turn it off only when cold-start IS what you're measuring.
        "warmup": bool(payload.get("warmup", True)),
    }
    if cache is not None:
        config["cache"] = cache
        # A single cache setting still has to drive randomize, since only the matrix path
        # translates the axis into the flag.
        if not isinstance(cache, list):
            config["randomize"] = (str(cache) == "cold")
    if upstream:
        # Explicit pin. Left unset, each cell resolves its own backend from the model, which is
        # what makes a mixed-model sweep (Ollama + vLLM in one submission) work.
        config["upstream"] = upstream
    if suite:
        config["suite"] = suite
        config["grade_timeout"] = float(payload.get("grade_timeout", 10.0) or 10.0)
    for key, cast in (("temperature", float), ("top_p", float), ("top_k", int),
                      ("min_p", float), ("seed", int), ("presence_penalty", float),
                      ("frequency_penalty", float), ("repetition_penalty", float)):
        v = keep(key, cast)
        if v is not None:
            config[key] = v
    if isinstance(payload.get("extra_body"), dict):
        config["extra_body"] = payload["extra_body"]
    # The window the backend is launched with. Absent means "leave it as found", so it is only
    # carried when actually asked for — an explicit None here would read as an axis of one.
    sctx = keep("server_context", int)
    if sctx:
        config["server_context"] = sctx
    # Reuse cells an earlier run already measured with identical settings. On unless told
    # otherwise: the signature covers every setting that changes what a cell measures, so a
    # reused cell is one that would have produced the same numbers, and the copy records where
    # they came from. The narrow risk of a stale number is worth less than the repeated cost of
    # forgetting — three multi-hour runs went out unprotected because this had to be asked for.
    config["resume"] = bool(payload.get("resume", True))

    # A matrix is any submission with more than one cell across the sweepable axes.
    cells = _bench_expand_matrix(models, config)
    is_matrix = len(cells) > 1
    bench_id = "b_" + uuid.uuid4().hex[:12]
    creator_ip = _client_ip(request)
    conn = db()
    if is_matrix:
        config["models"] = models
        label = f"{len(cells)} cells · {len(models)} model(s)"
        conn.execute(
            """INSERT INTO bench_runs (id, ts, model, config_json, status, creator_ip, label)
               VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
            (bench_id, time.time(), models[0]["model"], json.dumps(config), creator_ip, label),
        )
    else:
        # Flatten the single cell so the child config carries scalars, not one-element lists.
        only = cells[0]
        config["prompt_tokens"] = only["prompt_tokens"]
        config["thinking"] = only["thinking"]
        if only.get("temperature") is not None:
            config["temperature"] = only["temperature"]
        if only.get("upstream"):
            config["upstream"] = only["upstream"]
        if only.get("cache"):
            config["cache"] = only["cache"]
            config["randomize"] = (only["cache"] == "cold")
        if only.get("concurrency"):
            config["concurrency"] = only["concurrency"]
        config.pop("models", None)
        conn.execute(
            """INSERT INTO bench_runs (id, ts, model, config_json, status, creator_ip, axes_json, label)
               VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)""",
            (bench_id, time.time(), only["model"], json.dumps(config), creator_ip,
             json.dumps(only), _bench_cell_label(only)),
        )
    conn.commit()
    conn.close()
    if is_matrix:
        asyncio.create_task(_bench_execute_suite(bench_id, request.app))
    else:
        asyncio.create_task(_bench_execute(bench_id, request.app))
    return {"id": bench_id, "model": models[0]["model"], "config": config,
            "matrix": is_matrix, "cells": len(cells)}


# Everything that changes what a cell measures. Two cells with the same signature would
# produce the same numbers, which is what makes reuse safe — and what makes changing any
# setting correctly force a re-measurement.
_BENCH_SIG_KEYS = ("upstream", "suite", "runs", "max_tokens", "prompt_tokens", "server_context",
                   "thinking", "temperature", "cache", "concurrency", "randomize", "warmup")


def _bench_phase(bench_id: str, text: str | None) -> None:
    """Record what a run is doing right now, visible immediately.

    Its own connection and commit on purpose: a phase that only lands when the step finishes
    describes the past, and the whole point is the minutes in between.
    """
    try:
        conn = db()
        conn.execute("UPDATE bench_runs SET phase=? WHERE id=?", (text, bench_id))
        conn.commit()
        conn.close()
    except sqlite3.Error:
        pass


def _bench_cell_sig(model: str, cfg: dict) -> str:
    parts = {k: cfg.get(k) for k in _BENCH_SIG_KEYS}
    parts["model"] = model
    return json.dumps(parts, sort_keys=True, default=str)


def _bench_completed_cells(conn, days: int = 30) -> dict:
    """Signature -> the most recent completed run that measured it."""
    out: dict = {}
    rows = conn.execute(
        "SELECT id, model, config_json, results_json, env_json, axes_json, label, "
        "       started_ts, finished_ts, progress, progress_total "
        "FROM bench_runs WHERE status='done' AND ts > ? ORDER BY ts ASC",
        (time.time() - days * 86400,)).fetchall()
    for r in rows:
        try:
            cfg = json.loads(r["config_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if cfg.get("models"):
            continue          # a sweep parent measures nothing itself
        try:
            summary = (json.loads(r["results_json"] or "{}") or {}).get("summary") or {}
        except (json.JSONDecodeError, TypeError):
            summary = {}
        if not summary.get("n_success"):
            # "Done" is not "measured". A cell whose backend was OOM-cycling once finished
            # with 87 identical 502 rows and status 'done' — and a resume then copied that
            # garbage forward as if it were data. Zero successes means there is nothing here
            # worth reusing; let the new sweep measure it for real.
            continue
        out[_bench_cell_sig(r["model"], cfg)] = r      # later rows win: most recent
    return out


def _bench_history_cost(conn, days: int = 30) -> dict:
    """Mean completed-cell seconds per model, so a sweep can be costed before it runs."""
    rows = conn.execute(
        "SELECT model, started_ts, finished_ts FROM bench_runs "
        "WHERE status='done' AND started_ts IS NOT NULL AND finished_ts IS NOT NULL "
        "AND ts > ? ORDER BY ts DESC LIMIT 500", (time.time() - days * 86400,)).fetchall()
    agg: dict = {}
    for r in rows:
        d = r["finished_ts"] - r["started_ts"]
        if d > 0:
            agg.setdefault(r["model"], []).append(d)
    return {m: round(sum(v) / len(v)) for m, v in agg.items()}


def _bench_eta_s(cells: list, now: float) -> float | None:
    """Seconds left in a sweep, from what it has already done.

    Two costs, and the second is the one a naive estimate misses: a cell's own runtime, and the
    gap before it starts. On a server_context sweep that gap is a backend restart and a full
    model reload — minutes, sometimes longer than the cell itself — so an ETA built from cell
    durations alone reads far too optimistic. Both are measured from this run rather than
    assumed, so a slow box gets a slow estimate.
    """
    # `is not None`, not truthiness: a timestamp of 0 is a real time, and dropping those cells
    # silently degrades the estimate to "no basis" rather than to a wrong number.
    done = [c for c in cells if c["status"] == "done"
            and c["started_ts"] is not None and c["finished_ts"] is not None]
    if not done:
        return None
    durs = [c["finished_ts"] - c["started_ts"] for c in done]
    mean_dur = sum(durs) / len(durs)
    # Gap = time between one cell finishing and the next starting.
    gaps = []
    for prev, nxt in zip(cells, cells[1:]):
        if (prev["finished_ts"] is not None and nxt["started_ts"] is not None
                and nxt["started_ts"] > prev["finished_ts"]):
            gaps.append(nxt["started_ts"] - prev["finished_ts"])
    mean_gap = (sum(gaps) / len(gaps)) if gaps else 0.0
    total = 0.0
    for c in cells:
        if c["status"] in ("done", "skipped", "failed"):
            continue
        if c["status"] == "running":
            elapsed = now - (now if c["started_ts"] is None else c["started_ts"])
            prog, want = (c["progress"] or 0), (c["progress_total"] or 0)
            # Project from this cell's own pace once it has produced anything; before that its
            # own history is all we have.
            total += max(0.0, (elapsed * want / prog) - elapsed) if (prog and want) \
                else max(0.0, mean_dur - elapsed)
        else:
            total += mean_dur + mean_gap
    return round(total, 1) if total > 0 else None


@app.get("/__proxy/api/bench/runs")
async def bench_runs_list(request: Request, limit: int = 50, include_children: bool = False):
    """History, one entry per submission.

    A sweep's cells are rows in this table too, so listing everything showed one click as 25
    entries — and, because the limit counts them, two sweeps pushed every earlier run off the
    end of the history. The cells are already rendered inside their parent's detail view, so
    listing them again at the top level was duplication as well as noise.
    """
    limit = max(1, min(int(limit or 50), 200))
    conn = db()
    rows = conn.execute(
        "SELECT id, ts, model, config_json, status, progress, progress_total, "
        "started_ts, finished_ts, error, creator_ip, parent_id, axes_json, label, phase, "
        "env_json "
        "FROM bench_runs "
        + ("" if include_children else "WHERE parent_id IS NULL ")
        + "ORDER BY ts DESC LIMIT ?",
        (limit,),
    ).fetchall()
    # How each sweep's cells actually landed. The parent's own progress counter only tracks
    # cells that were queued, so without this a sweep that skipped or failed half its matrix
    # still reads as a clean "done".
    cells: dict = {}
    ids = [r["id"] for r in rows if not r["parent_id"]]
    if ids:
        qs = ",".join("?" * len(ids))
        for c in conn.execute(
            f"SELECT parent_id, status, COUNT(*) n FROM bench_runs "
            f"WHERE parent_id IN ({qs}) GROUP BY parent_id, status", ids,
        ).fetchall():
            entry = cells.setdefault(c["parent_id"], {"total": 0})
            entry[c["status"]] = c["n"]
            entry["total"] += c["n"]
        # Which cell is in flight, and how far into it. The parent's own counter only ticks
        # when a whole cell finishes, so on a graded sweep with large prompts it can sit at
        # 0/6 for many minutes. That detail used to be visible because every cell had its own
        # history row; now that they are grouped, it has to come along with the group.
        for c in conn.execute(
            f"SELECT parent_id, label, progress, progress_total, phase FROM bench_runs "
            f"WHERE parent_id IN ({qs}) AND status='running' ORDER BY ts, rowid", ids,
        ).fetchall():
            entry = cells.setdefault(c["parent_id"], {"total": 0})
            entry.setdefault("now", {"label": c["label"], "progress": c["progress"],
                                     "progress_total": c["progress_total"],
                                     "phase": c["phase"]})
        # How much longer, measured from this run's own pace.
        now_ts = time.time()
        by_parent: dict = {}
        for c in conn.execute(
            f"SELECT parent_id, status, started_ts, finished_ts, progress, progress_total "
            f"FROM bench_runs WHERE parent_id IN ({qs}) ORDER BY ts, rowid", ids,
        ).fetchall():
            by_parent.setdefault(c["parent_id"], []).append(dict(c))
        for pid, kids in by_parent.items():
            eta = _bench_eta_s(kids, now_ts)
            if eta is not None and cells.get(pid):
                cells[pid]["eta_s"] = eta
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
        try:
            d["axes"] = json.loads(d.pop("axes_json") or "null")
        except (json.JSONDecodeError, TypeError):
            d["axes"] = None
        if cells.get(d["id"]):
            d["cells"] = cells[d["id"]]
        try:
            d["memory_warning"] = (json.loads(r["env_json"] or "{}") or {}).get("memory_warning")
        except (json.JSONDecodeError, TypeError):
            d["memory_warning"] = None
        items.append(d)
    return {"items": items}


@app.get("/__proxy/api/bench/runs/{bench_id}/grades")
async def bench_grades(bench_id: str):
    """The grading browser: a cell renders every request and case statically; a sweep parent
    renders an index of its cells. This is where "89% — what failed in the other 11% and
    WHY?" gets answered, one request at a time."""
    conn = db()
    row = conn.execute("SELECT * FROM bench_runs WHERE id=?", (bench_id,)).fetchone()
    if not row:
        conn.close()
        return JSONResponse({"error": "not found"}, status_code=404)
    run = dict(row)
    for k, col in (("config", "config_json"), ("results", "results_json")):
        try:
            run[k] = json.loads(run.get(col) or "{}")
        except (json.JSONDecodeError, TypeError):
            run[k] = {}
    kids = conn.execute(
        "SELECT id, label, status, results_json FROM bench_runs WHERE parent_id=? ORDER BY ts",
        (bench_id,)).fetchall()
    conn.close()
    if kids:
        children = []
        for k in kids:
            c = dict(k)
            try:
                c["results"] = json.loads(c.get("results_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                c["results"] = {}
            children.append(c)
        html = _bench_report_mod._bench_grades_index_html(run, children)
    else:
        html = _bench_report_mod._bench_grades_html(run)
    return Response(content=html, media_type="text/html")


@app.get("/__proxy/api/bench/runs/{bench_id}")
async def bench_run_get(bench_id: str, request: Request):
    conn = db()
    row = conn.execute("SELECT * FROM bench_runs WHERE id=?", (bench_id,)).fetchone()
    conn.close()
    if not row or not _bench_visible_to(_client_ip(request), row["creator_ip"]):
        return JSONResponse({"error": "not found"}, status_code=404)
    d = dict(row)
    for src, dst, default in (("config_json", "config", {}), ("results_json", "results", None),
                              ("axes_json", "axes", None), ("env_json", "env", None)):
        try:
            d[dst] = json.loads(d.pop(src) or "null")
            if d[dst] is None and default is not None:
                d[dst] = default
        except (json.JSONDecodeError, TypeError):
            d[dst] = default
    # A suite parent carries its children inline so the UI renders the whole sweep in one fetch.
    if not d.get("parent_id"):
        conn = db()
        kids = conn.execute(
            "SELECT id, model, label, status, axes_json, results_json, error "
            "FROM bench_runs WHERE parent_id=? ORDER BY ts, rowid", (bench_id,),
        ).fetchall()
        conn.close()
        if kids:
            children = []
            for k in kids:
                kd = dict(k)
                try:
                    kd["axes"] = json.loads(kd.pop("axes_json") or "null")
                except (json.JSONDecodeError, TypeError):
                    kd["axes"] = None
                try:
                    res = json.loads(kd.pop("results_json") or "null")
                except (json.JSONDecodeError, TypeError):
                    res = None
                # Summary only — the per-request rows across a whole sweep are far too much
                # to ship into a list view.
                kd["summary"] = (res or {}).get("summary")
                children.append(kd)
            d["children"] = children
    return d


@app.delete("/__proxy/api/bench/runs/{bench_id}")
async def bench_run_delete(bench_id: str, request: Request):
    conn = db()
    row = conn.execute("SELECT creator_ip FROM bench_runs WHERE id=?", (bench_id,)).fetchone()
    if not row or not _bench_visible_to(_client_ip(request), row["creator_ip"]):
        conn.close()
        return JSONResponse({"error": "not found"}, status_code=404)
    # Cascade: deleting a suite parent must take its cells with it, or they become orphans
    # that no view lists and nothing can clean up.
    conn.execute("DELETE FROM bench_runs WHERE id=? OR parent_id=?", (bench_id, bench_id))
    conn.commit()
    conn.close()
    return {"ok": True}


# The whitepaper renderer lives in bench_report — pure functions, importable
# without the app. Re-bound under the old names for callers and tests.
_host_hw_facts = _bench_report_mod._host_hw_facts
_fmt_n = _bench_report_mod._fmt_n
_d3_source = _bench_report_mod._d3_source
_bench_model_display = _bench_report_mod._bench_model_display
_bench_model_identity = _bench_report_mod._bench_model_identity
_bench_label_display = _bench_report_mod._bench_label_display
_bench_fmt = _bench_report_mod._bench_fmt
_bench_report_row = _bench_report_mod._bench_report_row
_REPORT_TOKENS_LIGHT = _bench_report_mod._REPORT_TOKENS_LIGHT
_REPORT_TOKENS_DARK = _bench_report_mod._REPORT_TOKENS_DARK
_REPORT_CSS = _bench_report_mod._REPORT_CSS
_report_head = _bench_report_mod._report_head
_REPORT_BUILDING = _bench_report_mod._REPORT_BUILDING
_REPORT_BUILT = _bench_report_mod._REPORT_BUILT
_report_foot = _bench_report_mod._report_foot
_report_page = _bench_report_mod._report_page
_SVG_HALO = _bench_report_mod._SVG_HALO
_BENCH_WEIGHT_DEFAULT = _bench_report_mod._BENCH_WEIGHT_DEFAULT
_bench_weighted_data = _bench_report_mod._bench_weighted_data
_bench_weighted_rows = _bench_report_mod._bench_weighted_rows
_bench_weighted_html = _bench_report_mod._bench_weighted_html
_bench_parallel_groups = _bench_report_mod._bench_parallel_groups
_bench_category_winners_html = _bench_report_mod._bench_category_winners_html
_bench_coldstart_split_html = _bench_report_mod._bench_coldstart_split_html
_bench_language_profile_html = _bench_report_mod._bench_language_profile_html
_bench_category_html = _bench_report_mod._bench_category_html
_bench_efficiency_html = _bench_report_mod._bench_efficiency_html
_bench_variance_html = _bench_report_mod._bench_variance_html
_bench_place_labels = _bench_report_mod._bench_place_labels
_bench_size_by_model = _bench_report_mod._bench_size_by_model
_bench_best_per_model = _bench_report_mod._bench_best_per_model
_bench_bubbles_svg = _bench_report_mod._bench_bubbles_svg
_bench_answer_time_svg = _bench_report_mod._bench_answer_time_svg
_bench_engine_pair_data = _bench_report_mod._bench_engine_pair_data
_bench_engine_pairs_svg = _bench_report_mod._bench_engine_pairs_svg
_bench_coldstart_svg = _bench_report_mod._bench_coldstart_svg
_bench_scorecards = _bench_report_mod._bench_scorecards
_bench_pareto = _bench_report_mod._bench_pareto
_bench_scatter_svg = _bench_report_mod._bench_scatter_svg
_bench_bar_svg = _bench_report_mod._bench_bar_svg
_h = _bench_report_mod._h
_BENCH_AXIS_LABELS = _bench_report_mod._BENCH_AXIS_LABELS
_bench_axis_values = _bench_report_mod._bench_axis_values
_bench_axis_split = _bench_report_mod._bench_axis_split
_bench_cell_name = _bench_report_mod._bench_cell_name
_bench_case_text = _bench_report_mod._bench_case_text
_bench_failure_examples = _bench_report_mod._bench_failure_examples
_bench_report_html = _bench_report_mod._bench_report_html






def _fmt_tokens(v):
    """Token counts run to the billions here; full digits are unreadable at a glance."""
    if not v:
        return "0"
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if v >= div:
            return f"{v / div:.1f}{unit}"
    return f"{int(v):,}"


def _fmt_dur(seconds):
    if not seconds:
        return "—"
    d, rem = divmod(int(seconds), 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m}m"
    return f"{m}m"


def _status_label(st):
    """0 is what a request that never completed records — it is not an HTTP status, and showing
    it as one sends people looking for a status-0 response that doesn't exist."""
    if st is None or st == 0:
        return "no response (aborted or still in flight)"
    return str(st)


def _bar_cell(value, top):
    pct = 0 if not top else max(1, round((value or 0) / top * 100))
    return f'<td style="width:120px"><span class="bar"><i style="width:{pct}%"></i></span></td>'


# These run only when a report is generated, never on the dashboard's poll: the tool-hygiene
# pass parses request bodies, which is far too expensive to sit behind a 3s refresh.
_TURN_BUCKETS = ((0, 5, "1–4"), (5, 15, "5–14"), (15, 40, "15–39"),
                 (40, 100, "40–99"), (100, 1 << 30, "100+"))


# Idle draw as a share of load draw, used only when watts_idle isn't configured. Measured
# systems land anywhere from ~15% (a big discrete GPU that clocks right down) to ~35% (an
# always-on SoC), so this is a middle the report names rather than hides.
_IDLE_DRAW_FRACTION = 0.30


def _gpu_watts():
    """Current total GPU draw, or None. A spot reading, not a per-request measurement.

    Worth distrusting on unified-memory parts: a GB10 at 96% utilisation reports ~34W, which is
    a fraction of what the module pulls at the wall, because the field covers only one domain.
    That's why the config can override it — a measured number beats a reported one here."""
    if not _NVIDIA_SMI:
        return None
    try:
        r = subprocess.run([_NVIDIA_SMI, "--query-gpu=power.draw",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=3)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    total = 0.0
    for line in r.stdout.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("["):
            continue
        try:
            total += float(line)
        except ValueError:
            pass
    return total or None


def _busy_seconds_by_day(rows) -> dict:
    """Wall-clock seconds the upstreams were working, per local day.

    Summing durations would double-count everything served concurrently — on this traffic that
    overstates busy time severalfold, and energy is billed by the clock, not by the request. So
    overlapping requests are merged into intervals first, then split at midnight so a generation
    running over the boundary is charged to the days it actually spanned."""
    spans = sorted((r["ts"], r["ts"] + (r["duration_ms"] or 0) / 1000.0)
                   for r in rows if r["ts"] and (r["duration_ms"] or 0) > 0)
    merged: list = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    out: dict = {}
    for start, end in merged:
        cursor = start
        while cursor < end:
            day = datetime.datetime.fromtimestamp(cursor)
            midnight = datetime.datetime.combine(
                day.date() + datetime.timedelta(days=1), datetime.time.min).timestamp()
            stop = min(end, midnight)
            out[day.date().isoformat()] = out.get(day.date().isoformat(), 0.0) + (stop - cursor)
            cursor = stop
    return out


def _blank_day(date: str, n_tiers: int) -> dict:
    return {"date": date, "n": 0, "input": 0, "cached_input": 0, "output": 0,
            "costs": [0.0] * n_tiers, "busy_s": 0.0, "idle_s": 0.0, "kwh": 0.0,
            "power_cost": 0.0}


def _usage_by_day(days: int = 30) -> dict:
    """A daily ledger of tokens, what they would have cost elsewhere, and what they cost here.

    Local models produce no invoice, so a cost column has to be honest about what it is: the
    price the same traffic would have carried elsewhere, which is the only way to put a figure
    on what the hardware returns. Two tiers by default rather than one, because a frontier rate
    flatters a local 30B and an open-weights rate undersells it; the pair brackets the answer.
    Against them sits the electricity actually spent producing the same tokens.

    The number lives or dies on the cached share, and that is worked out per conversation rather
    than per request. An agentic client re-sends the whole conversation every turn, so turn N's
    prompt is turn N-1's prompt plus whatever is new; only the difference is a token any API
    would charge at the full input rate, and the shared prefix is a cache read at roughly a
    tenth of it. That is how the commercial APIs actually bill, and it needs nothing but the
    conversation id and the prompt sizes.

    The earlier approach — reading the proxy's own cache verdict — was worse for this: it is a
    prefill-speed heuristic that can say nothing at all about a request that was not streamed,
    and a fifth of this traffic isn't, which left the largest term in the total resting on
    "unknown, assume uncached".
    """
    cfg = load_rules_config().get("pricing") or {}
    tiers = [t for t in (cfg.get("tiers") or []) if isinstance(t, dict)]
    priced = bool(cfg.get("enabled", True)) and bool(tiers)
    if not priced:
        tiers = []
    out: dict = {"days": days, "priced": priced,
                 "tiers": [t.get("name") or "unnamed" for t in tiers]}
    conn = db()
    try:
        rows = conn.execute(
            """SELECT ts, date(ts, 'unixepoch', 'localtime') d, conversation_id, duration_ms,
                      prompt_tokens, est_prompt_tokens, completion_tokens
               FROM requests
               WHERE ts > ? AND shadow_of IS NULL
               ORDER BY conversation_id, ts""",
            (time.time() - days * 86400,)).fetchall()
    except sqlite3.Error as e:
        conn.close()
        return {**out, "error": str(e), "by_day": []}
    conn.close()

    per_day: dict = {}
    conv, prev_prompt = object(), 0
    for r in rows:
        # Rows arrive grouped by conversation, so one running total is enough. A request with no
        # conversation is a one-shot: nothing preceded it, so all of it is new.
        cid = r["conversation_id"]
        if cid is None or cid != conv:
            conv, prev_prompt = (cid if cid is not None else object()), 0
        full = r["prompt_tokens"] or r["est_prompt_tokens"] or 0
        # Compaction and context trimming can shrink a prompt; the reusable prefix is then at
        # most what is left, never what the longer previous turn held.
        cached = min(prev_prompt, full)
        uncached = full - cached
        prev_prompt = full
        completion = r["completion_tokens"] or 0

        day = per_day.setdefault(r["d"], _blank_day(r["d"], len(tiers)))
        day["n"] += 1
        day["input"] += uncached
        day["cached_input"] += cached
        day["output"] += completion
        for i, t in enumerate(tiers):
            day["costs"][i] += ((uncached / 1e6) * (t.get("input") or 0)
                                + (cached / 1e6) * (t.get("cached_input") or 0)
                                + (completion / 1e6) * (t.get("output") or 0))

    # --- what producing it actually drew ---------------------------------------------------
    elec = cfg.get("electricity") or {}
    # Both figures mean the whole machine at the wall. Serving a token also costs CPU, RAM,
    # disk, fans and PSU losses, and charging only the accelerator makes local inference look
    # cheaper than it is.
    watts, source = elec.get("watts"), "whole system, configured"
    if not watts:
        # The fallback is wrong twice: it omits everything that is not the GPU, and on
        # unified-memory parts nvidia-smi reports one rail of the module — a GB10 at 96%
        # utilisation reads ~34 W with no power limit to check it against. Named for what it
        # is, so nobody reads it as the machine's draw.
        watts, source = _gpu_watts(), "GPU rail only, well under the true figure"
    idle_w, idle_source = elec.get("watts_idle"), "configured"
    if idle_w is None and watts:
        idle_w, idle_source = watts * _IDLE_DRAW_FRACTION, "assumed"
    rate = elec.get("usd_per_kwh")
    if watts and rate:
        out["power"] = {"watts": watts, "watts_idle": idle_w, "usd_per_kwh": rate,
                        "source": source, "idle_source": idle_source}
        busy = _busy_seconds_by_day(rows)
        now, start = time.time(), time.time() - days * 86400
        # A generation running across midnight belongs to both days. The second of those may
        # have no request of its own — the row that started it is filed under the previous day —
        # and looking days up instead of creating them dropped that power on the floor.
        for date in busy:
            per_day.setdefault(date, _blank_day(date, len(tiers)))
        for date, day in per_day.items():
            day["busy_s"] = busy.get(date, 0.0)
            # The machine draws power between requests, and on traffic this bursty that is most
            # of the day. Clipped to the window at both ends so the first and last days aren't
            # charged for hours outside it.
            midnight = datetime.datetime.combine(
                datetime.date.fromisoformat(date), datetime.time.min).timestamp()
            covered = min(now, midnight + 86400) - max(start, midnight)
            day["idle_s"] = max(0.0, covered - day["busy_s"])
            day["kwh"] = ((day["busy_s"] / 3600.0) * (watts / 1000.0)
                          + (day["idle_s"] / 3600.0) * ((idle_w or 0) / 1000.0))
            day["power_cost"] = day["kwh"] * rate

    by_day = sorted(per_day.values(), key=lambda d: d["date"])
    for d in by_day:
        d["total"] = d["input"] + d["cached_input"] + d["output"]
    out["by_day"] = by_day
    if by_day:
        out["total"] = {
            k: sum(d[k] for d in by_day)
            for k in ("n", "input", "cached_input", "output", "total",
                      "busy_s", "idle_s", "kwh", "power_cost")
        }
        out["total"]["costs"] = [sum(d["costs"][i] for d in by_day) for i in range(len(tiers))]
    else:
        out["total"] = None
    # Days with no traffic at all are absent rather than zero — worth saying, since a reader
    # counting rows would otherwise take the ledger for a continuous record.
    if by_day:
        first = datetime.date.fromisoformat(by_day[0]["date"])
        last = datetime.date.fromisoformat(by_day[-1]["date"])
        out["missing_days"] = (last - first).days + 1 - len(by_day)
    return out


def _usage_extras(window_s: float = 86400.0, hygiene_limit: int = 800) -> dict:
    """Views the stats endpoint doesn't compute, because they only make sense over a window and
    the last one is too costly to poll."""
    out: dict = {"window_s": window_s}
    conn = db()
    since = time.time() - window_s
    try:
        # --- how conversations grow, and what it costs -----------------------------------
        case = " ".join(
            f"WHEN turn_index >= {lo} AND turn_index < {hi} THEN '{label}'"
            for lo, hi, label in _TURN_BUCKETS)
        rows = conn.execute(
            f"""SELECT CASE {case} END AS bucket, COUNT(*) n,
                       AVG(COALESCE(prompt_tokens, est_prompt_tokens)) prompt,
                       AVG(duration_ms) dur, AVG(ttft_ms) ttft
                FROM requests
                WHERE turn_index IS NOT NULL AND ts > ? AND (status IS NULL OR status < 400)
                GROUP BY bucket""", (since,)).fetchall()
        order = {label: i for i, (_lo, _hi, label) in enumerate(_TURN_BUCKETS)}
        out["by_turn"] = sorted(
            [{"bucket": r["bucket"], "n": r["n"], "prompt": r["prompt"],
              "duration_ms": r["dur"], "ttft_ms": r["ttft"]} for r in rows if r["bucket"]],
            key=lambda r: order.get(r["bucket"], 99))

        # --- where the upstream time actually went ----------------------------------------
        r = conn.execute(
            """SELECT SUM(ttft_ms) prefill, SUM(duration_ms - ttft_ms) decode, COUNT(*) n
               FROM requests WHERE ttft_ms > 0 AND duration_ms > ttft_ms AND ts > ?""",
            (since,)).fetchone()
        prefill, decode = (r["prefill"] or 0), (r["decode"] or 0)
        total = prefill + decode
        out["time_split"] = {
            "prefill_ms": prefill, "decode_ms": decode, "n": r["n"] or 0,
            "prefill_pct": (100 * prefill / total) if total else None,
            "decode_pct": (100 * decode / total) if total else None,
            "hours": total / 3600000.0,
        }

        # 499 is the client hanging up mid-generation: work done, nothing delivered.
        r = conn.execute(
            """SELECT COUNT(*) n, SUM(duration_ms) ms FROM requests
               WHERE status = 499 AND ts > ?""", (since,)).fetchone()
        out["abandoned"] = {"n": r["n"] or 0, "wasted_ms": r["ms"] or 0}

        # --- tool calls the client never offered -------------------------------------------
        # The mismatch that had Hermes calling run() ~93k times against a client that declared
        # terminal. What a client declares is a property of the client, not of the request — it
        # sends the same tool list every turn — so sample one body per client rather than
        # parsing thousands of half-megabyte agentic prompts to re-read the same array.
        declared_by_app: dict = {}
        apps = conn.execute(
            """SELECT client_app FROM requests WHERE ts > ?
               GROUP BY client_app ORDER BY COUNT(*) DESC LIMIT 12""", (since,)).fetchall()
        for a in apps:
            app = a["client_app"]
            row = conn.execute(
                """SELECT b.request_body FROM requests r JOIN request_blobs b ON b.id = r.id
                   WHERE r.ts > ? AND r.client_app IS ? AND b.request_body LIKE '%"tools"%'
                   ORDER BY r.ts DESC LIMIT 1""", (since, app)).fetchone()
            try:
                tools = (json.loads(row["request_body"]).get("tools") or []) if row else []
            except (json.JSONDecodeError, TypeError):
                tools = []
            names = {(t.get("function") or {}).get("name")
                     for t in tools if isinstance(t, dict)}
            names.discard(None)
            declared_by_app[app] = names

        # Replies are small next to the prompts, so this side can cover far more requests.
        # Most tool calls arrive streamed, assembled across SSE deltas rather than sitting in
        # response_body — _extract_tool_calls is what already knows how to read both.
        called_undeclared: dict = {}
        rows = conn.execute(
            """SELECT r.client_app, b.response_body, b.stream_chunks FROM requests r
               JOIN request_blobs b ON b.id = r.id
               WHERE r.ts > ? AND (b.response_body IS NOT NULL OR b.stream_chunks IS NOT NULL)
               ORDER BY r.ts DESC LIMIT ?""", (since, hygiene_limit)).fetchall()
        for row in rows:
            declared = declared_by_app.get(row["client_app"])
            # No sampled declaration means every name would look invented. Say nothing instead.
            if not declared:
                continue
            for name in _extract_tool_calls(row["response_body"], row["stream_chunks"]):
                if name in declared:
                    continue
                slot = called_undeclared.setdefault(
                    name, {"tool": name, "calls": 0, "clients": set()})
                slot["calls"] += 1
                slot["clients"].add(row["client_app"] or "unknown")
        # Logging captures what the model emitted, not what the client was handed, so a name an
        # alias already rewrites still shows up here. That is the useful reading — it says the
        # model is still getting it wrong — but the row has to say the fix is in place.
        ta = load_rules_config().get("tool_aliases") or {}
        amap = (ta.get("map") or {}) if ta.get("enabled") else {}
        out["undeclared_tools"] = sorted(
            [{"tool": v["tool"], "calls": v["calls"], "clients": sorted(v["clients"]),
              "aliased_to": (_alias_target(amap[v["tool"]])[0] or None) if v["tool"] in amap else None}
             for v in called_undeclared.values()],
            key=lambda v: -v["calls"])[:10]
        out["hygiene_sampled"] = len(rows)
    except sqlite3.Error as e:
        out["error"] = str(e)
    finally:
        conn.close()
    # Its own window: a day-by-day ledger over 24 hours would be one row.
    out["daily"] = _usage_by_day()
    return out


def _stats_report_html(d: dict, env: dict, extras: dict | None = None) -> str:
    """Usage over a window. Deliberately not the same shape as the benchmark report: that one
    compares configurations you chose, this one describes traffic you didn't — so it leads with
    volume and composition, and treats latency as a property of the mix rather than a score."""
    o = d.get("overall") or {}
    gen = o.get("throughput_generation") or {}
    depth = o.get("throughput_by_depth") or []
    count = o.get("count") or 0
    errors = o.get("errors") or 0
    err_rate = (errors / count) if count else 0
    span = ((o.get("last_ts") or 0) - (o.get("first_ts") or 0)) or 0

    # The characteristic fact about this traffic is its asymmetry: enormous prompts, small
    # replies. It explains the rest of the page — why aggregate throughput looks low, why decode
    # is reported by depth, why prefill dominates latency — so it leads.
    pt = o.get("prompt_tokens") or 0
    ct = o.get("completion_tokens") or 0
    ratio = (pt / ct) if ct else None
    # Short sentences, and no number the meta strip above already carries. The strip states the
    # period, the request count and the token total; repeating them here spends the most
    # valuable space on the page saying nothing new.
    per_hour = (count / (span / 3600)) if span > 3600 else None
    why = ("That is the shape of agentic traffic: long context, short answers. It is why one "
           "tokens-per-second figure across every request reads low, and why decode rate is "
           "broken out by prompt depth further down."
           if ratio and ratio > 2 else
           "Prompt and completion volumes are close, so aggregate throughput and decode rate "
           "should track each other.")
    hero = f"""<div class="hero">
      <p class="lede">These models read <b>{_fmt_tokens(pt)}</b> tokens and wrote
        <b>{_fmt_tokens(ct)}</b>{f" — <b>{ratio:,.0f}</b> read for every one written" if ratio and ratio > 2 else ""}.</p>
      <p class="why">{why}</p>
    </div>
    <div class="cards">
      <div class="card hi"><p class="k">Decode rate</p><p class="v">{_fmt_n(gen.get("p50"), 1)}<small> tok/s</small></p>
        <p class="d">median, measured from the first token on</p></div>
      <div class="card"><p class="k">Reply length</p><p class="v">{_fmt_n(ct / count if count else 0, 0)}<small> tok</small></p>
        <p class="d">mean across every request</p></div>
      <div class="card"><p class="k">Throughput</p><p class="v">{_fmt_n(per_hour, 1) if per_hour else _fmt_n(count)}<small>{" /h" if per_hour else " reqs"}</small></p>
        <p class="d">{"requests per hour over the period" if per_hour else "requests recorded"}</p></div>
      <div class="card"><p class="k">Errors</p><p class="v" style="color:{'var(--bad)' if err_rate > 0.02 else 'var(--ink)'}">{err_rate * 100:.1f}<small>%</small></p>
        <p class="d">{_fmt_n(errors)} of {_fmt_n(count)}</p></div>
    </div>"""
    cards = hero

    def table(title, note, headers, rows_html):
        if not rows_html:
            return ""
        head = "".join(f'<th{" class=\"n\"" if h.startswith("#") else ""}>{_h(h.lstrip("#"))}</th>'
                       for h in headers)
        return (f"<section><h2>{_h(title)}</h2>"
                + (f'<p class="note">{note}</p>' if note else "")
                + f'<div class="tbl"><table><thead><tr>{head}</tr></thead>'
                + f"<tbody>{rows_html}</tbody></table></div></section>")

    # --- what ran -------------------------------------------------------------------------
    models = sorted(d.get("by_model") or [], key=lambda m: -(m.get("count") or 0))[:12]
    top = max((m.get("count") or 0) for m in models) if models else 0
    model_rows = "".join(
        f'<tr><th scope="row"><code>{_h(m.get("model") or "—")}</code></th>'
        f'<td class="n">{_fmt_n(m.get("count"))}</td>{_bar_cell(m.get("count"), top)}'
        f'<td class="n">{_fmt_tokens(m.get("prompt_tokens"))}</td>'
        f'<td class="n">{_fmt_tokens(m.get("completion_tokens"))}</td>'
        f'<td class="n">{_fmt_n(m.get("avg_ms"), 0, " ms")}</td>'
        f'<td class="n {"bad" if (m.get("errors") or 0) else "ok"}">{_fmt_n(m.get("errors"))}</td></tr>'
        for m in models)

    apps = sorted(d.get("by_app") or [], key=lambda a: -(a.get("count") or 0))[:12]
    atop = max((a.get("count") or 0) for a in apps) if apps else 0
    app_rows = "".join(
        f'<tr><th scope="row">{_h(a.get("client_app") or "unknown")}</th>'
        f'<td class="n">{_fmt_n(a.get("count"))}</td>{_bar_cell(a.get("count"), atop)}'
        f'<td class="n">{_fmt_n(a.get("conversations"))}</td>'
        f'<td class="n">{_fmt_tokens(a.get("total_tokens"))}</td>'
        f'<td class="n">{_fmt_n(a.get("avg_ms"), 0, " ms")}</td></tr>'
        for a in apps)

    tools = sorted(d.get("by_tool") or [], key=lambda t: -(t.get("calls") or 0))[:12]
    ttop = max((t.get("calls") or 0) for t in tools) if tools else 0
    tool_rows = "".join(
        f'<tr><th scope="row"><code>{_h(t.get("name") or "—")}</code></th>'
        f'<td class="n">{_fmt_n(t.get("calls"))}</td>{_bar_cell(t.get("calls"), ttop)}</tr>'
        for t in tools)

    depth_rows = "".join(
        f'<tr><th scope="row">{_h(b.get("bucket"))}</th>'
        f'<td class="n">{_fmt_n(b.get("samples"))}</td>'
        f'<td class="n">{_fmt_n(b.get("p50"), 1)}</td>'
        f'<td class="n">{_fmt_n(b.get("p90"), 1)}</td>'
        f'<td class="n">{_fmt_n(b.get("max"), 1)}</td></tr>'
        for b in depth)

    ups = sorted(d.get("by_upstream") or [], key=lambda u: -(u.get("count") or 0))
    utop = max((u.get("count") or 0) for u in ups) if ups else 0
    up_rows = "".join(
        f'<tr><th scope="row">{_h(u.get("upstream") or "—")}</th>'
        f'<td class="n">{_fmt_n(u.get("count"))}</td>{_bar_cell(u.get("count"), utop)}'
        f'<td class="n">{_fmt_n(u.get("avg_ms"), 0, " ms")}</td></tr>'
        for u in ups)

    # None and 0 both mean "never completed"; keeping them apart produced two rows with the
    # same label and no way to tell why.
    merged: dict = {}
    for st in (d.get("by_status") or []):
        merged[_status_label(st.get("status"))] = merged.get(_status_label(st.get("status")), 0) + (st.get("count") or 0)
    statuses = sorted(merged.items(), key=lambda kv: -kv[1])[:8]
    status_rows = "".join(
        f'<tr><th scope="row">{_h(label)}</th><td class="n">{_fmt_n(n)}</td></tr>'
        for label, n in statuses)

    # Minor tables sit side by side; full width would give them an authority they have not
    # earned next to Models and Clients.
    minor = "".join([
        table("Upstreams", "", ["Upstream", "#Requests", "", "#Mean latency"], up_rows),
        table("Response status", "", ["Status", "#Requests"], status_rows),
        table("Tool calls",
              "Tools the models actually invoked — useful for spotting definitions that are "
              "sent every turn and never used.",
              ["Tool", "#Calls", ""], tool_rows),
    ])
    # The order is the argument: what ran, who asked for it, how fast it went, then small print.
    # --- sections computed only for the report ------------------------------------------
    ex = extras or {}
    turn_rows = ""
    for b in (ex.get("by_turn") or []):
        turn_rows += (
            f'<tr><th scope="row">{_h(b["bucket"])} turns</th>'
            f'<td class="n">{_fmt_n(b["n"])}</td>'
            f'<td class="n">{_fmt_n(b["prompt"], 0)}</td>'
            f'<td class="n">{_fmt_n(b["ttft_ms"], 0, " ms")}</td>'
            f'<td class="n">{_fmt_n(b["duration_ms"], 0, " ms")}</td></tr>')
    turn_html = ""
    if turn_rows:
        turn_html = (
            "<h2>Conversation depth</h2>"
            '<p class="note">How much context a request carries as a conversation goes on, and '
            "what it costs. Latency rising with depth means every turn re-reads the "
            "conversation; latency staying flat, or falling, means the prefix cache is holding "
            "the shared history and only the new turn is being read.</p>"
            "<div class=\"tbl\"><table><thead><tr><th>Depth</th><th class=\"n\">Requests</th>"
            "<th class=\"n\">Prompt tokens</th><th class=\"n\">TTFT</th>"
            "<th class=\"n\">Total</th></tr></thead><tbody>" + turn_rows + "</tbody></table></div>")

    ts = ex.get("time_split") or {}
    ab = ex.get("abandoned") or {}
    time_html = ""
    if ts.get("prefill_pct") is not None:
        time_html = (
            "<h2>Where the time goes</h2>"
            '<p class="note">Upstream time split between reading the prompt and writing the '
            "reply. With prompts this much larger than replies, prefill should dominate — that "
            "it does not is the prompt cache doing its job, so a jump in the prefill share is "
            "how a caching regression would first show up.</p>"
            '<div class="cards">'
            f'<div class="card"><p class="k">Prefill</p><p class="v">{ts["prefill_pct"]:.0f}<small>%</small></p>'
            f'<p class="d">reading the prompt</p></div>'
            f'<div class="card hi"><p class="k">Decode</p><p class="v">{ts["decode_pct"]:.0f}<small>%</small></p>'
            f'<p class="d">writing the reply</p></div>'
            f'<div class="card"><p class="k">Upstream time</p><p class="v">{ts["hours"]:.1f}<small> h</small></p>'
            f'<p class="d">across {_fmt_n(ts["n"])} requests</p></div>'
            + (f'<div class="card"><p class="k">Discarded</p><p class="v">{_fmt_dur((ab.get("wasted_ms") or 0)/1000)}</p>'
               f'<p class="d">{_fmt_n(ab.get("n"))} generations the client abandoned</p></div>'
               if ab.get("n") else "")
            + "</div>")

    # --- the daily ledger, last thing on the page --------------------------------------------
    daily = ex.get("daily") or {}
    by_day = daily.get("by_day") or []
    tot = daily.get("total") or {}
    tiers = daily.get("tiers") or []
    priced = bool(daily.get("priced") and tiers)
    power = daily.get("power") or {}
    day_html = ""
    if by_day:
        def day_row(d, is_total=False):
            label = ("Total" if is_total else
                     datetime.date.fromisoformat(d["date"]).strftime("%b %d"))
            # The row is an argument in three parts: what moved, what it drew, what it saved.
            cells = "".join(f'<td class="n{" seam" if i == 0 else ""}">${c:,.2f}</td>'
                            for i, c in enumerate(d.get("costs") or []))
            if power:
                cells = (f'<td class="n seam">{d["busy_s"] / 3600:,.1f}</td>'
                         f'<td class="n">${d["power_cost"]:,.2f}</td>' + cells)
            tr = '<tr class="sum">' if is_total else "<tr>"
            return (tr + f'<th scope="row">{_h(label)}</th>'
                    f'<td class="n">{_fmt_tokens(d["input"])}</td>'
                    f'<td class="n">{_fmt_tokens(d["cached_input"])}</td>'
                    f'<td class="n">{_fmt_tokens(d["output"])}</td>{cells}</tr>')

        cached_share = (100 * tot["cached_input"] / tot["total"]) if tot.get("total") else 0
        gap = daily.get("missing_days") or 0
        costs = tot.get("costs") or []
        # One number per card, in the page's own vocabulary. A second hero here competed with
        # the one at the top of the report, and three mono figures set inline in a 21px
        # sentence read as jumbled rather than emphatic. Every other section states its
        # headline numbers as cards; this one has no reason to be different.
        lede = ""
        if priced and costs:
            # Tokens first: they're what the section is a ledger of, and the money only means
            # anything once you know how much moved. The table's column groups run in the same
            # order, so the two read the same way.
            # Deliberately NOT named `cards`: that holds the page's opening hero, and assigning
            # to it here replaced the top of the report with this section's money and printed it
            # twice. Cost belongs at the bottom, which is where this section already sits.
            ledger_cards = (f'<div class="card"><p class="k">Tokens</p>'
                            f'<p class="v">{_fmt_tokens(tot["total"])}</p>'
                            f'<p class="d">{cached_share:.0f}% of them a cache read, '
                            f'{_fmt_tokens(tot["output"])} written</p></div>')
            if power and tot.get("power_cost"):
                ledger_cards += (f'<div class="card hi"><p class="k">Cost to produce</p>'
                                 f'<p class="v">${tot["power_cost"]:,.2f}</p>'
                                 f'<p class="d">electricity, over {tot["busy_s"] / 3600:,.0f} '
                                 f'GPU hours and the idle time around them</p></div>')
            for name, c in sorted(zip(tiers, costs), key=lambda p: p[1]):
                ledger_cards += (f'<div class="card"><p class="k">{_h(name)}</p>'
                                 f'<p class="v">${c:,.0f}</p>'
                                 f'<p class="d">what the same tokens would have cost</p></div>')
            lede = f'<div class="cards">{ledger_cards}</div>'

        # Everything qualifying the table goes below it. Stacked above, three paragraphs of
        # method were a wall between the reader and the only thing they came for.
        notes = [f"Tokens are split per conversation. Each turn re-sends everything before it, "
                 f"so only the growth counts as input and the rest is a cache read: "
                 f"{cached_share:.0f}% of all tokens here. Two things push that low. A "
                 f"conversation already running when the range opened has its first turn "
                 f"counted in full, and a real provider's cache expires between turns where "
                 f"this one never does."]
        if priced and len(costs) > 1:
            notes.insert(0, "Two reference rates, not one. A frontier price flatters a local "
                            "model that is not frontier quality; an open-weights price ignores "
                            "that some of this work would have gone to a frontier model if it "
                            "had not run here. The honest answer is between them.")
        if power:
            idle_w = power.get("watts_idle") or 0
            how = ("configured" if power.get("idle_source") == "configured"
                   else f"assumed at {_IDLE_DRAW_FRACTION:.0%} of load")
            notes.append(
                f"GPU hours are wall-clock, with concurrent requests merged. Energy is billed "
                f"by the clock, not by the request. Cost covers the whole day: "
                f"{power['watts']:,.0f} W under load ({_h(power.get('source') or '')}), "
                f"{idle_w:,.0f} W idle ({how}), at {power['usd_per_kwh']:.3f} $/kWh.")
            # The wattage is meant to be the whole machine at the wall. When it came from the
            # GPU instead, the reader has to be told the figure is a floor, not a measurement.
            if power.get("source", "").startswith("GPU rail"):
                notes.append(
                    "That wattage is the GPU's own reported draw, which is a floor rather than "
                    "a measurement: it leaves out the CPU, memory, disks, fans and power-supply "
                    "losses that serving a token also costs, and on unified-memory boards it "
                    "covers only one rail of the module. Put a meter on the wall under load and "
                    "at rest, and set <code>watts</code> and <code>watts_idle</code> from that.")
        if gap:
            notes.append(f"No traffic on {gap} day{'s' if gap != 1 else ''} in this range. "
                         f"Those have no row, and no idle draw counted.")
        if priced:
            notes.append("Rates, tiers and wattage live under <code>pricing</code> in the rules "
                         "config.")

        groups = [("", 1, "blank"), ("Tokens", 3, "")]
        if power:
            groups.append(("Produced here", 2, "seam"))
        if tiers:
            groups.append(("Bought elsewhere", len(tiers), "seam"))
        day_html = (
            "<h2>Day by day</h2>"
            + '<p class="note">What this traffic would have cost bought elsewhere, against the '
              'electricity spent making it here. Nothing on this page was billed — the models '
              'run on your own hardware.</p>'
            + lede
            + '<div class="tbl"><table><thead>'
            + '<tr class="grp">'
            + "".join(f'<th colspan="{n}" class="{cls}">{_h(name)}</th>' for name, n, cls in groups)
            + "</tr><tr><th>Date</th>"
            '<th class="n">Input</th><th class="n">Cached</th><th class="n">Output</th>'
            + ('<th class="n seam">GPU hrs</th><th class="n">Electricity</th>' if power else "")
            + "".join(f'<th class="n wrap{" seam" if i == 0 else ""}">{_h(name)}</th>'
                      for i, name in enumerate(tiers))
            + "</tr></thead><tbody>"
            + "".join(day_row(d) for d in by_day)
            + (day_row({**tot, "date": by_day[-1]["date"]}, True) if tot else "")
            + "</tbody></table></div>"
            + '<ul class="fn">' + "".join(f"<li>{n}</li>" for n in notes) + "</ul>")

    und = ex.get("undeclared_tools") or []
    tool_html = ""
    if und:
        rows_html = "".join(
            f'<tr><th scope="row"><code>{_h(u["tool"])}</code></th>'
            f'<td class="n">{_fmt_n(u["calls"])}</td>'
            f'<td>{_h(", ".join(u["clients"]))}</td>'
            + (f'<td>renamed to <code>{_h(u["aliased_to"])}</code></td>'
               if u.get("aliased_to") else '<td>reaches the client as-is</td>')
            + "</tr>"
            for u in und)
        tool_html = (
            "<h2>Tool calls the client never offered</h2>"
            '<p class="note">Names the model invented, against the tools its client actually '
            "declared. These are counted as the model emitted them, so a name an alias already "
            "rewrites still appears — that is the point, it says the model has not stopped "
            "getting it wrong. Anything unhandled is a call the client will reject, and where "
            "the intended target is obvious the <code>tool_aliases</code> rule can rename it. "
            f"Read from the {_fmt_n(ex.get('hygiene_sampled'))} most recent replies, against "
            "each client's most recently declared tool list.</p>"
            "<div class=\"tbl\"><table><thead><tr><th>Called</th><th class=\"n\">Times</th>"
            "<th>Clients</th><th>Status</th></tr></thead><tbody>"
            + rows_html + "</tbody></table></div>")

    body = "".join([
        cards,
        table("Models",
              "Ranked by request volume. Latency here is a mean over whole requests, so a few "
              "very long ones drag it upward — read it as a rough scale and use decode rate "
              "by depth for how fast the model actually generates.",
              ["Model", "#Requests", "", "#Prompt", "#Completion", "#Mean latency", "#Errors"],
              model_rows),
        table("Clients", "Which applications the traffic came from, by their own fingerprint.",
              ["Application", "#Requests", "", "#Conversations", "#Tokens", "#Mean latency"],
              app_rows),
        table("Decode rate by prompt depth",
              "Decode slows as the KV cache grows, so a single median across every prompt size "
              "describes no real request. Compare a quoted tok/s figure against the bucket that "
              "matches its prompt size.",
              ["Prompt size", "#Samples", "#p50 tok/s", "#p90", "#max"], depth_rows),
        turn_html,
        time_html,
        tool_html,
        f'<div class="band">{minor}</div>',
        day_html,
    ])

    gpus = env.get("gpus") or []
    gpu_txt = ", ".join(g.get("name") for g in gpus if isinstance(g, dict) and g.get("name")) or None
    when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return _report_page(
        title="Usage report",
        eyebrow="AI Proxy · usage",
        sub=f"{when} · everything the proxy has recorded",
        meta=[("Period", _fmt_dur(span)),
              ("First seen", datetime.datetime.fromtimestamp(o["first_ts"]).strftime("%Y-%m-%d")
                             if o.get("first_ts") else None),
              ("Requests", f"{count:,}"),
              ("Tokens", _fmt_tokens((o.get("prompt_tokens") or 0) + (o.get("completion_tokens") or 0))),
              ("GPU", gpu_txt),
              ("Proxy", env.get("proxy_version"))],
        body=body,
    )


@app.get("/__proxy/api/stats/report")
async def stats_report(format: str = "html"):
    """Everything the proxy has recorded, as a standalone page — the long-term counterpart to
    the benchmark report. The benchmark report compares configurations you chose; this one
    describes traffic you didn't, so it leads with volume and composition.

    Streamed. Assembling it means scanning the whole request table, walking recent bodies and
    pricing 30 days of conversations, which is seconds of work — and a browser given nothing to
    render for those seconds shows a blank tab that looks like a hang. Sending the head and a
    "building" notice first costs nothing and turns the wait into something legible; the notice
    hides itself via a rule at the end of the stream, so a saved copy has no leftovers.
    """
    if format != "html":
        return await asyncio.to_thread(stats)

    async def render():
        yield (_report_head("Usage report", "AI Proxy · usage") + _REPORT_BUILDING).encode()
        try:
            data = await asyncio.to_thread(stats)
            env = await _bench_env_snapshot()
            extras = await asyncio.to_thread(_usage_extras)
            html = _stats_report_html(data, env, extras)
        except Exception as e:
            # The head is already on the wire, so an error has to be rendered into the page
            # rather than returned as a status code.
            yield (f'<div class="hero"><p class="lede">The report could not be built.</p>'
                   f'<p class="why">{_h(f"{type(e).__name__}: {e}")}</p></div>'
                   + _report_foot()).encode()
            return
        # The shared builder emits the whole document; send only what follows the head.
        head_end = html.find("</h1>")
        yield (html[head_end + 5:] if head_end >= 0 else html).encode()

    return StreamingResponse(render(), media_type="text/html; charset=utf-8",
                             headers={"cache-control": "no-store",
                                      # nginx and friends will happily buffer a stream and
                                      # defeat the whole point.
                                      "x-accel-buffering": "no"})


@app.get("/__proxy/api/bench/report")
async def bench_report(request: Request, ids: str = "", format: str = "json"):
    """Comparison report over any number of runs. `ids` is comma-separated; a suite parent
    expands to its cells. format=json (default), markdown (pasteable table), or html (a
    standalone, printable page — no external requests, so it survives being saved or mailed)."""
    wanted = [i.strip() for i in (ids or "").split(",") if i.strip()]
    if not wanted:
        return JSONResponse({"error": "'ids' is required (comma-separated bench ids)"},
                            status_code=400)
    viewer = _client_ip(request)
    conn = db()
    runs: list[dict] = []
    for bid in wanted:
        row = conn.execute("SELECT * FROM bench_runs WHERE id=?", (bid,)).fetchone()
        if not row or not _bench_visible_to(viewer, row["creator_ip"]):
            continue
        kids = conn.execute("SELECT * FROM bench_runs WHERE parent_id=? ORDER BY ts, rowid",
                            (bid,)).fetchall()
        for r in (kids or [row]):
            d = dict(r)
            for src, dst in (("config_json", "config"), ("results_json", "results"),
                             ("axes_json", "axes"), ("env_json", "env")):
                try:
                    d[dst] = json.loads(d.pop(src) or "null")
                except (json.JSONDecodeError, TypeError):
                    d[dst] = None
            runs.append(d)
    conn.close()
    if not runs:
        return JSONResponse({"error": "no visible runs for those ids"}, status_code=404)
    rows = [_bench_report_row(r) for r in runs]
    if format == "html":
        return Response(content=_bench_report_html(runs, rows),
                        media_type="text/html; charset=utf-8")
    if format != "markdown":
        return {"rows": rows, "env": [r.get("env") for r in runs]}

    graded = any(r.get("perfect_rate") is not None for r in rows)
    head = ["Run", "Model", "Think", "Prompt", "Server ctx", "TTFT p50", "Decode p50", "Total p50"]
    if graded:
        head += ["Fully correct", "Cases"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        cells = [
            _bench_label_display(r["label"] or "—"),
            _bench_model_display(r["served"] or r["model"] or "—"),
            str(r["thinking"] or "auto"),
            _bench_fmt(r["prompt_tokens"], 0),
            _bench_fmt(r.get("server_context"), 0),
            _bench_fmt(r["ttft_p50"], 0, " ms"),
            _bench_fmt(r["decode_p50"], 1, " tok/s"),
            _bench_fmt(r["total_p50"], 0, " ms"),
        ]
        if graded:
            pr = r["perfect_rate"]
            cr = r["case_pass_rate"]
            cells += [f"{pr * 100:.0f}%" if pr is not None else "—",
                      f"{cr * 100:.0f}%" if cr is not None else "—"]
        lines.append("| " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    return Response(content=md, media_type="text/markdown; charset=utf-8")


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


# -------- Scratchboard: shared working notes, visible from every subnet --------
#
# Everything else the dashboard shows is somebody's captured traffic, so it goes through the
# PII gate: a viewer on another subnet sees placeholders. These notes are written BY the
# viewers rather than captured from them, and a shared board that each machine sees a
# different version of is not a shared board. So this endpoint deliberately does not redact,
# and that is the whole feature rather than an oversight.
_SCRATCH_MAX_CHARS = 8000
_SCRATCH_MAX_ROWS = 500
_SCRATCH_MAX_FILE_BYTES = 8 * 1024 * 1024


@app.get("/__proxy/api/scratchboard")
def scratchboard_list():          # sync → threadpool, same as the other small reads
    conn = db()
    rows = [dict(r) for r in conn.execute(
        "SELECT id, ts, author, text, client_ip, pinned, parent_id FROM scratchboard "
        "ORDER BY ts DESC LIMIT ?", (_SCRATCH_MAX_ROWS * 4,)).fetchall()]
    conn.close()
    # Roots newest-first with pinned on top; replies oldest-first beneath their parent, because
    # a thread reads forwards even though a board reads backwards.
    by_parent: dict = {}
    for r in rows:
        if r.get("parent_id"):
            by_parent.setdefault(r["parent_id"], []).append(r)
    roots = [r for r in rows if not r.get("parent_id")]
    roots.sort(key=lambda r: (-(r.get("pinned") or 0), -(r.get("ts") or 0)))
    # Attachment metadata only — never the bytes. A board with a few images on it would
    # otherwise ship megabytes of base64 on every poll of a page that refreshes constantly.
    files: dict = {}
    conn2 = db()
    for f in conn2.execute("SELECT id, note_id, name, mime, size, ts FROM scratchboard_files "
                           "ORDER BY ts").fetchall():
        files.setdefault(f["note_id"], []).append(dict(f))
    conn2.close()
    for r in rows:
        r["files"] = files.get(r["id"], [])
    for r in roots:
        r["replies"] = sorted(by_parent.get(r["id"], []), key=lambda x: x.get("ts") or 0)
    # A reply whose parent was deleted would otherwise vanish silently; surface it as a root
    # rather than losing someone's answer to a question that no longer exists.
    known = {r["id"] for r in roots}
    orphans = [r for pid, kids in by_parent.items() if pid not in known for r in kids]
    for o in orphans:
        o["replies"] = []
        o["orphaned"] = True
    return {"items": roots[:_SCRATCH_MAX_ROWS] + orphans}


@app.post("/__proxy/api/scratchboard")
async def scratchboard_add(request: Request):
    """Add a note. Body: {text, author?, pinned?}."""
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    text = (payload.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)
    if len(text) > _SCRATCH_MAX_CHARS:
        return JSONResponse(
            {"error": f"text too long ({len(text)} chars, limit {_SCRATCH_MAX_CHARS})"},
            status_code=413)
    author = (payload.get("author") or "").strip()[:64] or None
    entry_id = uuid.uuid4().hex[:16]
    conn = db()
    # One level of nesting: replying to a reply attaches to the same root, so a thread stays a
    # thread instead of becoming a tree nobody can render.
    parent = (payload.get("parent_id") or "").strip() or None
    if parent:
        row = conn.execute("SELECT id, parent_id FROM scratchboard WHERE id = ?",
                           (parent,)).fetchone()
        if not row:
            conn.close()
            return JSONResponse({"error": f"no note {parent!r} to reply to"}, status_code=404)
        parent = row["parent_id"] or row["id"]
    conn.execute(
        "INSERT INTO scratchboard (id, ts, author, text, client_ip, pinned, parent_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (entry_id, time.time(), author, text, _client_ip(request),
         1 if (payload.get("pinned") and not parent) else 0, parent))
    # Bound the table rather than letting a board become a log nobody reads. Pinned notes are
    # exempt: pinning is the way to say "this one is not scratch". Replies are exempt too —
    # pruning them by age would strip answers off notes that are still on the board.
    conn.execute(
        "DELETE FROM scratchboard WHERE pinned = 0 AND parent_id IS NULL AND id NOT IN "
        "(SELECT id FROM scratchboard WHERE pinned = 0 AND parent_id IS NULL "
        " ORDER BY ts DESC LIMIT ?)",
        (_SCRATCH_MAX_ROWS,))
    # The prune above took only the root. Its replies came back as orphan roots — which is the
    # right answer when a human deletes a note by accident, and the wrong one for a bound whose
    # whole job is to stop the board growing — and its attachments stayed in the table with
    # nothing referencing them, at up to 8 MB each. Written as a sweep rather than a targeted
    # delete so it also clears whatever the earlier version already stranded.
    conn.execute("DELETE FROM scratchboard WHERE parent_id IS NOT NULL "
                 "AND parent_id NOT IN (SELECT id FROM scratchboard WHERE parent_id IS NULL)")
    conn.execute("DELETE FROM scratchboard_files "
                 "WHERE note_id NOT IN (SELECT id FROM scratchboard)")
    conn.commit()
    conn.close()
    return {"ok": True, "id": entry_id}


@app.delete("/__proxy/api/scratchboard/{entry_id}")
def scratchboard_delete(entry_id: str):
    conn = db()
    # Replies go with their parent. Leaving them behind would surface answers as roots, which
    # is right for a parent that vanished by accident and wrong for one deleted on purpose.
    # Attachments go too — orphaned blobs would sit in the database with nothing referencing
    # them, which is how a bounded table stops being bounded.
    conn.execute("DELETE FROM scratchboard_files WHERE note_id = ? OR note_id IN "
                 "(SELECT id FROM scratchboard WHERE parent_id = ?)", (entry_id, entry_id))
    cur = conn.execute("DELETE FROM scratchboard WHERE id = ? OR parent_id = ?",
                       (entry_id, entry_id))
    conn.commit()
    conn.close()
    return {"ok": True, "removed": cur.rowcount}


@app.post("/__proxy/api/scratchboard/{entry_id}/files")
async def scratchboard_attach(entry_id: str, request: Request):
    """Attach a file to a note. Body: {name, mime?, data_b64}.

    base64 in JSON rather than multipart: multipart needs python-multipart, and this package
    ships three dependencies on purpose. The 33% encoding overhead is irrelevant at an 8 MB cap.
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    name = (payload.get("name") or "").strip()[:200]
    b64 = payload.get("data_b64") or ""
    if not name or not b64:
        return JSONResponse({"error": "name and data_b64 are required"}, status_code=400)
    try:
        blob = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        return JSONResponse({"error": "data_b64 is not valid base64"}, status_code=400)
    if len(blob) > _SCRATCH_MAX_FILE_BYTES:
        return JSONResponse(
            {"error": f"file is {len(blob):,} bytes; limit {_SCRATCH_MAX_FILE_BYTES:,}"},
            status_code=413)
    conn = db()
    if not conn.execute("SELECT 1 FROM scratchboard WHERE id = ?", (entry_id,)).fetchone():
        conn.close()
        return JSONResponse({"error": f"no note {entry_id!r}"}, status_code=404)
    fid = uuid.uuid4().hex[:16]
    conn.execute("INSERT INTO scratchboard_files (id, note_id, ts, name, mime, size, data) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (fid, entry_id, time.time(), name,
                  (payload.get("mime") or "").strip()[:100] or None, len(blob), blob))
    conn.commit()
    conn.close()
    return {"ok": True, "id": fid, "size": len(blob)}


_SCRATCH_ZIP_MAX_BYTES = 512 * 1024 * 1024


def _zip_safe_name(name: str, fallback: str = "file") -> str:
    """A filename a zip can carry and an extractor will not write outside the target
    directory. Path separators and .. are stripped rather than escaped: the name is a label,
    and no attachment here legitimately contains a directory."""
    base = (name or "").replace("\\", "/").split("/")[-1]
    base = re.sub(r'[^A-Za-z0-9._ ()\[\]-]', "_", base).strip(" .")
    return base[:120] or fallback


def _scratch_note_text(note: dict, replies: list) -> str:
    """The note as it reads on the board. Whoever opens the zip a month later needs the author,
    the time and the thread, not just the body — the attachments are only half of a note."""
    def block(n, indent=""):
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(n.get("ts") or 0))
        head = f"{n.get('author') or 'anonymous'} · {when}"
        if n.get("client_ip"):
            head += f" · from {n['client_ip']}"
        files = [f["name"] for f in (n.get("files") or [])]
        out = [indent + head, indent + "-" * len(head)]
        out += [indent + line for line in (n.get("text") or "").splitlines()]
        if files:
            out += ["", indent + "attached: " + ", ".join(files)]
        return "\n".join(out)

    parts = [block(note)]
    for r in replies:
        parts.append("\n" + block(r, indent="    "))
    return "\n".join(parts) + "\n"


def _scratch_zip(notes: list[dict], filename: str):
    """One download: every attachment plus the text that goes with it.

    Built in memory. The per-file cap is 8 MB and a board is 500 notes, so the ceiling is
    theoretically large — hence the explicit total check, which refuses with the arithmetic
    rather than having the process killed while assembling a 4 GB buffer.
    """
    import io
    import zipfile

    total = sum(f.get("size") or 0 for n in notes for f in (n.get("files") or []))
    if total > _SCRATCH_ZIP_MAX_BYTES:
        return JSONResponse(
            {"error": f"attachments total {total / 1048576:.0f} MB; the zip is built in memory "
                      f"and refuses above {_SCRATCH_ZIP_MAX_BYTES // 1048576} MB. Download the "
                      f"files individually, or delete some first.",
             "total_bytes": total}, status_code=413)

    conn = db()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        board = []
        for note in notes:
            replies = note.get("replies") or []
            text = _scratch_note_text(note, replies)
            board.append(text)
            # One folder per note only when there is more than one, so the common case —
            # downloading a single note — extracts flat instead of into a stray directory.
            stem = _zip_safe_name(
                f"{time.strftime('%Y%m%d-%H%M', time.localtime(note.get('ts') or 0))}-"
                f"{note.get('author') or 'anon'}", "note")
            prefix = f"{stem}/" if len(notes) > 1 else ""
            if len(notes) > 1:
                z.writestr(prefix + "note.txt", text)
            seen: dict = {}
            for n in [note] + list(replies):
                for f in (n.get("files") or []):
                    row = conn.execute("SELECT data FROM scratchboard_files WHERE id = ?",
                                       (f["id"],)).fetchone()
                    if not row:
                        continue          # deleted between listing the board and reading it
                    name = _zip_safe_name(f.get("name") or "", "file")
                    # Two attachments called screenshot.png are normal; silently keeping one
                    # would be a download that quietly lost a file.
                    if name in seen:
                        seen[name] += 1
                        stem_, dot, ext = name.rpartition(".")
                        name = (f"{stem_} ({seen[name] - 1}).{ext}" if dot
                                else f"{name} ({seen[name] - 1})")
                    else:
                        seen[name] = 1
                    z.writestr(prefix + name, row["data"])
        z.writestr("notes.txt" if len(notes) > 1 else "note.txt",
                   ("\n\n" + "=" * 78 + "\n\n").join(board))
    conn.close()
    data = buf.getvalue()
    return Response(
        content=data, media_type="application/zip",
        headers={"content-disposition": f'attachment; filename="{filename}"',
                 "content-length": str(len(data))})


@app.get("/__proxy/api/scratchboard/zip")
def scratchboard_zip_all():
    """The whole board as one file. Registered before the {entry_id} route below so that
    'zip' is read as the literal path it is."""
    board = scratchboard_list()
    notes = board.get("items") or []
    if not notes:
        return JSONResponse({"error": "the board is empty"}, status_code=404)
    return _scratch_zip(notes, f"scratchboard-{time.strftime('%Y%m%d-%H%M')}.zip")


@app.get("/__proxy/api/scratchboard/{entry_id}/zip")
def scratchboard_zip_note(entry_id: str):
    notes = [n for n in (scratchboard_list().get("items") or []) if n["id"] == entry_id]
    if not notes:
        return JSONResponse({"error": f"no note {entry_id!r}"}, status_code=404)
    stem = _zip_safe_name(f"{notes[0].get('author') or 'note'}-"
                          f"{time.strftime('%Y%m%d-%H%M', time.localtime(notes[0].get('ts') or 0))}",
                          "note")
    return _scratch_zip(notes, f"{stem}.zip")


@app.get("/__proxy/api/scratchboard/files/{file_id}")
def scratchboard_file(file_id: str):
    conn = db()
    row = conn.execute("SELECT name, mime, data FROM scratchboard_files WHERE id = ?",
                       (file_id,)).fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "not found"}, status_code=404)
    # Always as an attachment, always octet-stream, never sniffed. These bytes are uploaded by
    # whoever can reach the proxy and are served from the same origin as the dashboard — an
    # HTML or SVG file rendered inline would be script execution against this page.
    safe = re.sub(r'[^A-Za-z0-9._ -]', "_", row["name"] or "file")[:120]
    return Response(
        content=row["data"], media_type="application/octet-stream",
        headers={"content-disposition": f'attachment; filename="{safe}"',
                 "x-content-type-options": "nosniff"})


@app.delete("/__proxy/api/scratchboard/files/{file_id}")
def scratchboard_file_delete(file_id: str):
    conn = db()
    cur = conn.execute("DELETE FROM scratchboard_files WHERE id = ?", (file_id,))
    conn.commit()
    conn.close()
    return {"ok": True, "removed": cur.rowcount}


@app.post("/__proxy/api/scratchboard/{entry_id}/pin")
async def scratchboard_pin(entry_id: str, request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    conn = db()
    cur = conn.execute("UPDATE scratchboard SET pinned = ? WHERE id = ?",
                       (0 if payload.get("pinned") is False else 1, entry_id))
    conn.commit()
    conn.close()
    return {"ok": True, "updated": cur.rowcount}


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


@app.get("/__proxy")
@app.get("/__proxy/")
async def ui_index():
    # no-cache means "revalidate before reusing", not "don't store": the browser still gets a
    # 304 when nothing changed, but a tab can never come back holding a stale dashboard after a
    # deploy. Debugging an old UI against a new server is a bad use of anyone's evening.
    return FileResponse(STATIC_DIR / "index.html",
                        headers={"cache-control": "no-cache, must-revalidate"})


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

# ---- Graded task suite -------------------------------------------------------------------------
# Perf without correctness is a speedometer, not a decision tool: a model that streams 40% faster
# but gets the code wrong is not the better model. Each task asks for one named Python function
# and carries deterministic (input → expected) cases. The model's code block is extracted and run
# in a subprocess, and the score is the fraction of cases that produced the expected value.
#
# SECURITY: this executes model-generated code. It runs in a separate process with a hard timeout,
# in a scratch cwd, with the parent's env stripped down. That contains accidents and runaway loops
# — it is NOT a sandbox against deliberately hostile code. Grading is opt-in per bench run.
# Suite task data lives in its own module — pure data, no logic. Re-bound here under the
# old names so every existing reference (and every test that monkeypatches through this
# module) keeps working unchanged.
_BENCH_SUITES = _bench_suites_mod.SUITES
_BENCH_TASK_DESC = _bench_suites_mod.TASK_DESC
_BENCH_TASK_NOTES = _bench_suites_mod.TASK_NOTES

# agent-v1: multi-tool episodes. Registered into the same suite registry so the picker, the
# reports, the grades browser and reuse signatures all treat it as just another suite.
try:
    from . import bench_agent as _bench_agent_mod
except ImportError:          # flat-script launch
    import bench_agent as _bench_agent_mod
_BENCH_SUITES.setdefault("agent-v1", _bench_agent_mod.AGENT_TASKS)
# agent-v2: the hard set — each episode targets one specific agentic failure mode
# (error-message protocols, joins, budgeted search, dedup, authority, migration, recall).
_BENCH_SUITES.setdefault("agent-v2", _bench_agent_mod.AGENT2_TASKS
                         + _bench_agent_mod.CONSTRAINT_AGENT_TASKS)
_BENCH_TASK_DESC.update(_bench_agent_mod.AGENT_TASK_DESC)
_BENCH_TASK_DESC.update(_bench_agent_mod.CONSTRAINT_AGENT_DESC)
_BENCH_TASK_NOTES.update(_bench_agent_mod.AGENT_TASK_NOTES)
_BENCH_TASK_NOTES.update(_bench_agent_mod.CONSTRAINT_AGENT_NOTES)

try:
    from . import bench_security as _bench_security_mod
except ImportError:          # flat-script launch
    import bench_security as _bench_security_mod
_BENCH_SUITES.setdefault("security-v1", _bench_security_mod.SECURITY_TASKS
                         + _bench_agent_mod.SECURITY_AGENT_TASKS)

try:
    from . import bench_instruct as _bench_instruct_mod
except ImportError:          # flat-script launch
    import bench_instruct as _bench_instruct_mod
# instruct-v1: obedience and output shape — the properties a parser depends on.
# refusal-v1: two-sided calibration — engages with security work, declines the harmful end.
try:
    from . import bench_memory as _bench_memory_mod
except ImportError:          # flat-script launch
    import bench_memory as _bench_memory_mod
# memory-v1: what the model chooses to write down, look up, and revise. Graded on the
# STORE the episode leaves behind, not only on the answer it produced.
_BENCH_SUITES.setdefault("memory-v1", _bench_memory_mod.MEMORY_TASKS)
_BENCH_TASK_DESC.update(_bench_memory_mod.MEMORY_TASK_DESC)
_BENCH_TASK_NOTES.update(_bench_memory_mod.MEMORY_TASK_NOTES)

try:
    from . import bench_langpref as _bench_langpref_mod
except ImportError:          # flat-script launch
    import bench_langpref as _bench_langpref_mod
# langpref-v1: not "is it right" but "what did it reach for". The score only catches an
# outright strange pick; the profile in the report is the point.
_BENCH_SUITES.setdefault("langpref-v1", _bench_langpref_mod.LANGPREF_TASKS)
_BENCH_TASK_DESC.update(_bench_langpref_mod.LANGPREF_TASK_DESC)
_BENCH_TASK_NOTES.update(_bench_langpref_mod.LANGPREF_TASK_NOTES)

try:
    from . import bench_longctx as _bench_longctx_mod
except ImportError:          # flat-script launch
    import bench_longctx as _bench_longctx_mod
# longctx-v1: whether a big window holds anything. Deliberately NOT folded into full-v1 or
# full-v2 — a 300k prefill is minutes of exclusive GPU, and quietly adding that to the suite
# everyone runs would turn a 20-minute sweep into an afternoon. Run it on purpose.
_BENCH_SUITES.setdefault("longctx-v1", _bench_longctx_mod.LONGCTX_TASKS)
_BENCH_TASK_DESC.update(_bench_longctx_mod.LONGCTX_TASK_DESC)
_BENCH_TASK_NOTES.update(_bench_longctx_mod.LONGCTX_TASK_NOTES)

_BENCH_SUITES.setdefault("instruct-v1", _bench_instruct_mod.INSTRUCT_TASKS)
_BENCH_SUITES.setdefault("refusal-v1", _bench_instruct_mod.REFUSAL_TASKS)
_BENCH_TASK_DESC.update(_bench_instruct_mod.INSTRUCT_TASK_DESC)
_BENCH_TASK_DESC.update(_bench_instruct_mod.REFUSAL_TASK_DESC)
_BENCH_TASK_NOTES.update(_bench_instruct_mod.INSTRUCT_TASK_NOTES)
_BENCH_TASK_NOTES.update(_bench_instruct_mod.REFUSAL_TASK_NOTES)
_BENCH_TASK_DESC.update(_bench_security_mod.SECURITY_TASK_DESC)
_BENCH_TASK_DESC.update(_bench_agent_mod.SECURITY_AGENT_DESC)
_BENCH_TASK_NOTES.update(_bench_security_mod.SECURITY_TASK_NOTES)
_BENCH_TASK_NOTES.update(_bench_agent_mod.SECURITY_AGENT_NOTES)

# full-v1: one suite, three categories. Splitting them was an artefact of when each was
# written, not of how models are used — the same request stream wants code that works, an
# agent that finishes, and neither of them handing an attacker the keys. Running them
# together is also the only way a single number means anything: a model that writes clean
# code and follows injected instructions is not "87% good".
_BENCH_SUITES.setdefault("full-v1", _BENCH_SUITES["coding-v3"]
                         + _BENCH_SUITES["agent-v2"]
                         + _BENCH_SUITES["security-v1"])
# full-v2 adds the behavioural halves: doing as it is told, and engaging with the work.
_BENCH_SUITES.setdefault("full-v2", _BENCH_SUITES["full-v1"]
                         + _BENCH_SUITES["instruct-v1"]
                         + _BENCH_SUITES["refusal-v1"]
                         + _BENCH_SUITES["memory-v1"])

# Category per task id, defaulted by which suite a task came from so the older suites did
# not need editing task-by-task. Read by the report to break results out by category.
_BENCH_TASK_CATEGORY: dict = {}
for _suite_name, _default_cat in (("coding-v1", "coding"), ("coding-v2", "coding"),
                                  ("coding-v3", "coding"), ("agent-v1", "agentic"),
                                  ("agent-v2", "agentic"), ("security-v1", "security"),
                                  ("instruct-v1", "instruct"), ("refusal-v1", "refusal"),
                                  ("memory-v1", "memory"),
                                  ("langpref-v1", "preference"),
                                  ("longctx-v1", "longcontext")):
    for _t in _BENCH_SUITES.get(_suite_name) or []:
        _BENCH_TASK_CATEGORY.setdefault(_t["id"], _t.get("category") or _default_cat)
# The security suite's own tasks additionally carry a side; agentic security episodes are
# blue-team by construction (the model is the thing under attack).
_BENCH_TASK_SIDE: dict = {t["id"]: t.get("side")
                          for t in (_BENCH_SUITES.get("security-v1") or [])
                          + (_BENCH_SUITES.get("refusal-v1") or []) if t.get("side")}
# The report renders the category breakdown but must not re-derive the mapping — one
# definition, pushed to it here, so the two cannot drift.
_bench_report_mod.TASK_CATEGORY.update(_BENCH_TASK_CATEGORY)
_bench_report_mod.TASK_SIDE.update(_BENCH_TASK_SIDE)
_bench_report_mod.TASK_REQUESTED_LANG.update(_bench_langpref_mod.LANGPREF_REQUESTED)

# Grading machinery lives in bench_graders — synchronous, importable without the
# app. Re-bound under the old names so references and monkeypatching tests keep
# working; note that graders-internal calls resolve inside that module, so tests
# that patch grader INTERNALS patch ai_proxy.bench_graders, not this facade.
_BENCH_CODE_BLOCK_RE = _bench_graders_mod._BENCH_CODE_BLOCK_RE
_BENCH_LANGS = _bench_graders_mod._BENCH_LANGS
_BENCH_TOOL_DIRS = _bench_graders_mod._BENCH_TOOL_DIRS
_bench_tool = _bench_graders_mod._bench_tool
_bench_lit = _bench_graders_mod._bench_lit
_BENCH_TOOLCHAIN_PROBES = _bench_graders_mod._BENCH_TOOLCHAIN_PROBES
_BENCH_TOOLCHAIN_OK = _bench_graders_mod._BENCH_TOOLCHAIN_OK
_bench_lang_available = _bench_graders_mod._bench_lang_available
_bench_suite_tasks = _bench_graders_mod._bench_suite_tasks
_BenchHTML = _bench_graders_mod._BenchHTML
_bench_html_select = _bench_graders_mod._bench_html_select
_bench_html_text = _bench_graders_mod._bench_html_text
_bench_check_html = _bench_graders_mod._bench_check_html
_bench_check_css = _bench_graders_mod._bench_check_css
_html_attr_eq = _bench_graders_mod._html_attr_eq
toolchain_versions = _bench_graders_mod.toolchain_versions
_bench_extract_code = _bench_graders_mod._bench_extract_code
_BENCH_GRADER_SRC = _bench_graders_mod._BENCH_GRADER_SRC
_BENCH_JS_GRADER_SRC = _bench_graders_mod._BENCH_JS_GRADER_SRC
_bench_grade_c = _bench_graders_mod._bench_grade_c
_bench_grade_compiled = _bench_graders_mod._bench_grade_compiled
_bench_grade_php = _bench_graders_mod._bench_grade_php
_bench_grade_sql = _bench_graders_mod._bench_grade_sql
_bench_grade_bash = _bench_graders_mod._bench_grade_bash
_bench_grade_sync = _bench_graders_mod._bench_grade_sync
_bench_grade_needles = _bench_graders_mod._bench_grade_needles



async def _bench_grade(text: str, task: dict, timeout_s: float,
                       reasoning: str = "", finish_reason: str | None = None) -> dict:
    """Grade one response against one task definition."""
    lang = task.get("lang") or "python"
    # Analysis, format and refusal tasks are graded on the reply itself: extracting a code
    # block from an answer whose deliverable IS the answer would throw away the thing being
    # graded — and for format tasks the wrapper is precisely what is under test.
    code = (text if lang in ("text", "answer", "format", "refusal", "langpick", "needles")
            else _bench_extract_code(text, lang))
    # A model that answered in its reasoning field has still answered. Measured: at 700k this
    # model returned empty content with all five codes in `reasoning`, which would score 0
    # against a suite whose whole point is whether the context was readable. Scoped to needles
    # rather than applied generally — for a format task the content IS the deliverable, and
    # grading its scratchpad instead would be measuring the wrong thing.
    if lang == "needles" and not code.strip():
        code = reasoning or ""
    if not code.strip():
        return {"task": task["id"], "passed": 0, "total": len(task["cases"]),
                "score": 0.0, "error": "no code in response"}
    res = await asyncio.to_thread(_bench_grade_sync, code, task["entry"], task["cases"],
                                  timeout_s, lang, task.get("suffix") or "")
    total = res.get("total") or len(task["cases"])
    res["task"] = task["id"]
    res["score"] = (res.get("passed", 0) / total) if total else 0.0
    # A reply cut off at max_tokens did not fail the task, it never finished answering it.
    # Measured during the first longctx run: one 262k unit spent its whole 2,048-token budget
    # on 6,134 characters of reasoning and emitted no content at all, which would have been
    # read as recall collapsing at 256k. Flagged rather than silently scored.
    if finish_reason == "length" and res.get("passed", 0) < total:
        res["truncated"] = True
    return res


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


def _bench_task_at_depth(task_prompt: str, prompt_tokens: int, randomize: bool,
                         seq: int) -> str:
    """The same graded task, asked after `prompt_tokens` of irrelevant context.

    Every suite task is a short prompt, so the suites measure a model at its best and say
    nothing about the state it is actually used in — hermes sits at 50k tokens by turn 30.
    Sweeping prompt_tokens as an axis turns any graded suite into a long-context test: the
    task is identical at every depth, so a score that falls is the context doing it, not
    the task getting harder.

    The task goes LAST, after the filler, because that is where the real ask sits in a long
    agent transcript, and it is delimited so a model that reads only the tail can still see
    the whole instruction. Nothing in the filler is needed to answer — a model that ignores
    the padding entirely SHOULD score exactly as it does at zero depth, which is what makes
    a drop meaningful.
    """
    if prompt_tokens <= 0:
        return task_prompt
    target_chars = int(prompt_tokens * 3.5)
    salt = f"// nonce: {uuid.uuid4().hex}\n" if randomize else ""
    head = ("Reference material from the current session follows. It is background only "
            "and is NOT needed to answer; the task is stated after it.\n\n"
            f"{salt}<CONTEXT>\n")
    tail = ("\n</CONTEXT>\n\nIgnore the reference material above unless it is relevant. "
            f"Your task:\n\n{task_prompt}")
    body = ""
    while len(head) + len(body) + len(tail) < target_chars:
        body += _BENCH_FILLER
    return head + body + tail


# Thinking control. Three different engines expose three different switches, and each was
# established empirically (see docs/benchmarking.md):
#   - chat_template_kwargs.enable_thinking  → Qwen3-lineage on vLLM (ignores reasoning_effort)
#   - reasoning_effort: "none"              → ds4 / DwarfStar (its only off switch; low/med/high
#                                             all map to the same "think" mode)
#   - empty <think></think> assistant prefill → LM Studio / llama.cpp, which drops
#                                             chat_template_kwargs entirely, so the prefill is
#                                             the ONLY thing that suppresses reasoning there
_BENCH_THINK_MODES = ("auto", "on", "off", "off_prefill")
_BENCH_THINK_PREFILL = "<think>\n\n</think>\n\n"


def _bench_apply_thinking(body: dict, mode: str) -> None:
    """Mutate a chat-completions body to force reasoning on or off. 'auto' leaves the body
    alone so the proxy's own per-model quirks apply (which is what production traffic sees)."""
    if mode == "auto":
        return
    want_on = mode == "on"
    ctk = body.get("chat_template_kwargs")
    if not isinstance(ctk, dict):
        ctk = {}
    ctk["enable_thinking"] = want_on
    body["chat_template_kwargs"] = ctk
    body["reasoning_effort"] = "high" if want_on else "none"
    if mode == "off_prefill":
        # Pre-close the think block by putting a spent one in the model's mouth. Must be the
        # final message so the model continues from it.
        body.setdefault("messages", []).append(
            {"role": "assistant", "content": _BENCH_THINK_PREFILL})


def _bench_apply_sampling(body: dict, cfg: dict) -> None:
    """Copy sampling knobs from the bench config onto the request body, skipping unset ones
    so the engine's own defaults stand."""
    for key in ("temperature", "top_p", "top_k", "min_p", "seed", "presence_penalty",
                "frequency_penalty", "repetition_penalty"):
        v = cfg.get(key)
        if v is not None:
            body[key] = v
    extra = cfg.get("extra_body")
    if isinstance(extra, dict):
        for k, v in extra.items():
            body[k] = v


def _bench_build_body(model: str, prompt: str, max_tokens: int, cfg: dict) -> dict:
    """Assemble the full request body for one bench request."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
    }
    _bench_apply_thinking(body, str(cfg.get("thinking") or "auto"))
    _bench_apply_sampling(body, cfg)
    return body


def _bench_headers(cfg: dict) -> dict:
    """Headers for a bench request. x-client-name is what panic mode and the exclusive-mode
    gate whitelist; the x-proxy-* headers pin routing so we measure the model we named."""
    h = {
        "content-type": "application/json",
        "x-client-name": "ai-proxy-bench",
        "x-priority": "high",
    }
    if cfg.get("bypass_router"):
        h["x-proxy-no-router"] = "1"
    if cfg.get("no_nudge"):
        h["x-proxy-no-nudge"] = "1"
    up = str(cfg.get("upstream") or "").strip().lower()
    if up:
        h["x-proxy-upstream"] = up
    return h


async def _bench_run_agent(client: httpx.AsyncClient, base: str, model: str,
                           task: dict, cfg: dict) -> dict:
    """One agent EPISODE: the model drives, the bench plays the tools deterministically.

    Non-streaming on purpose — an episode's metric is turns and wall time, not token
    cadence — and the whole conversation transcript is stored so the grading browser can
    show exactly where an episode went off the rails."""
    cfg = cfg or {}
    world = _bench_agent_mod.AgentWorld(task)
    messages = [{"role": "user", "content": task["prompt"]}]
    headers = _bench_headers(cfg)
    transcript = [f"USER: {task['prompt']}"]
    t0 = time.perf_counter()
    steps = 0
    final = None
    err = None
    ttft = None
    ctok = 0
    for _ in range(int(task.get("max_steps") or 10)):
        body = {"model": model, "messages": messages, "tools": task.get("tools") or [],
                "stream": False, "max_tokens": int(cfg.get("max_tokens") or 1024)}
        _bench_apply_thinking(body, str(cfg.get("thinking") or "auto"))
        _bench_apply_sampling(body, cfg)
        try:
            resp = await client.post(base + "/v1/chat/completions", headers=headers,
                                     json=body, timeout=httpx.Timeout(600.0))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            break
        if resp.status_code != 200:
            err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            break
        try:
            j = resp.json()
        except ValueError:
            err = "non-JSON response"
            break
        steps += 1
        if ttft is None:
            ttft = (time.perf_counter() - t0) * 1000
        ctok += ((j.get("usage") or {}).get("completion_tokens") or 0)
        msg = ((j.get("choices") or [{}])[0].get("message")) or {}
        tcs = msg.get("tool_calls") or []
        if tcs:
            messages.append({"role": "assistant", "content": msg.get("content") or None,
                             "tool_calls": tcs})
            for k, tc in enumerate(tcs):
                fn = (tc.get("function") or {})
                nm = fn.get("name") or "?"
                out = world.execute(nm, fn.get("arguments"))
                transcript.append(f"TOOL CALL: {nm}({str(fn.get('arguments'))[:200]})")
                transcript.append(f"TOOL RESULT: {out[:200]}")
                messages.append({"role": "tool",
                                 "tool_call_id": tc.get("id") or f"call_{steps}_{k}",
                                 "content": out})
            continue
        final = msg.get("content") or ""
        transcript.append(f"ASSISTANT: {final[:400]}")
        break
    exhausted = final is None and err is None
    if exhausted:
        transcript.append(f"[step budget of {task.get('max_steps')} exhausted]")
    grade = _bench_agent_mod.grade_episode(task, world, final, steps, exhausted)
    total_ms = (time.perf_counter() - t0) * 1000
    return {"seq": 0, "task": task["id"], "error": err, "grade": grade,
            "ttft_ms": ttft, "ttfc_ms": ttft, "total_ms": total_ms,
            "completion_tokens": ctok or None, "reasoning_tokens": None,
            "decode_tps": (ctok / (total_ms / 1000)) if ctok and total_ms else None,
            "steps": steps, "text": "\n".join(transcript)[:6000]}


async def _bench_run_one(client: httpx.AsyncClient, base: str, model: str,
                         max_tokens: int, prompt: str, run_seq: int,
                         cfg: dict | None = None, capture_text: bool = False) -> dict:
    """Issue one streaming chat-completion request via the proxy and collect timings.

    Reasoning-aware: a thinking model emits its reasoning as `reasoning_content` deltas before
    any `content` arrives. Timing only `content` (as this did originally) reports time-to-END-of-
    reasoning as TTFT and drops every reasoning token from the decode rate — which understates
    throughput and overstates latency by whatever the model spent thinking. We record both
    boundaries: `ttft_ms` (first token of either kind) and `ttfc_ms` (first content token).
    """
    cfg = cfg or {}
    body = json.dumps(_bench_build_body(model, prompt, max_tokens, cfg)).encode("utf-8")
    headers = _bench_headers(cfg)
    t0 = time.perf_counter()
    ttft_ms: float | None = None       # first token of ANY kind (reasoning or content)
    ttfc_ms: float | None = None       # first content token
    upstream_ct: int | None = None
    upstream_pt: int | None = None
    upstream_rt: int | None = None
    content_words = 0
    reasoning_words = 0
    served_model: str | None = None
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: str | None = None
    err: str | None = None
    status_code: int | None = None
    try:
        async with client.stream("POST", base + "/v1/chat/completions",
                                 headers=headers, content=body, timeout=httpx.Timeout(600.0)) as resp:
            status_code = resp.status_code
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
                    if served_model is None and j.get("model"):
                        served_model = str(j["model"])
                    for ch in (j.get("choices") or []):
                        if ch.get("finish_reason"):
                            finish_reason = str(ch["finish_reason"])
                        delta = ch.get("delta") or {}
                        # Engines disagree on the field name for reasoning deltas.
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        if reasoning:
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - t0) * 1000
                            reasoning_words += len(re.findall(r"\S+", reasoning))
                            if capture_text:
                                reasoning_parts.append(reasoning)
                        if delta.get("content"):
                            now_ms = (time.perf_counter() - t0) * 1000
                            if ttft_ms is None:
                                ttft_ms = now_ms
                            if ttfc_ms is None:
                                ttfc_ms = now_ms
                            content_words += len(re.findall(r"\S+", delta["content"]))
                            if capture_text:
                                text_parts.append(delta["content"])
                    u = j.get("usage")
                    if isinstance(u, dict):
                        if u.get("completion_tokens"):
                            upstream_ct = u["completion_tokens"]
                        if u.get("prompt_tokens"):
                            upstream_pt = u["prompt_tokens"]
                        details = u.get("completion_tokens_details")
                        if isinstance(details, dict) and details.get("reasoning_tokens"):
                            upstream_rt = details["reasoning_tokens"]
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    total_ms = (time.perf_counter() - t0) * 1000
    # Prefer the upstream's own token accounting; fall back to a word count only when absent.
    completion_tokens = upstream_ct if upstream_ct is not None else (content_words + reasoning_words)
    # Decode rate spans first token → end, so reasoning tokens count as generated work.
    decode_tps: float | None = None
    if ttft_ms is not None and total_ms > ttft_ms and completion_tokens > 0:
        decode_tps = completion_tokens / ((total_ms - ttft_ms) / 1000.0)
    # An empty completion is a failure, not a very fast success. This is how a prompt that
    # overflows the model's context window presents (see the tokenizer-density note in
    # docs/benchmarking.md) — the request 200s and returns nothing.
    if not err and completion_tokens == 0:
        err = "empty completion (0 tokens) — possible context overflow or refused generation"
    row = {
        "seq": run_seq,
        "ttft_ms": ttft_ms,
        "ttfc_ms": ttfc_ms,
        "total_ms": total_ms,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": upstream_rt if upstream_rt is not None else (reasoning_words or None),
        "prompt_tokens": upstream_pt,
        "decode_tps": decode_tps,
        "served_model": served_model,
        "status": status_code,
        "error": err,
    }
    if capture_text:
        row["text"] = "".join(text_parts)
        # Kept apart from `text` rather than concatenated: for every other suite the content
        # field is the deliverable and folding a scratchpad into it would change what they
        # grade. Only the long-context grader reads this, and only when content came back empty.
        row["reasoning_text"] = "".join(reasoning_parts)
    row["finish_reason"] = finish_reason
    return row


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


def _bench_stat(values: list) -> dict:
    """min / p50 / p90 / max / mean for one metric, all None when there's nothing to report."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"min": None, "p50": None, "p90": None, "max": None, "mean": None}
    return {
        "min": min(vals),
        "p50": _bench_pct(vals, 50),
        "p90": _bench_pct(vals, 90),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
    }


def _bench_summarize(rows: list[dict]) -> dict:
    """Roll up per-request timings. Reports TTFT (first token of any kind) and TTFC (first
    *content* token) separately — for a thinking model the gap between them is the reasoning
    phase, and conflating the two is what made the original bench misreport reasoning models."""
    successes = [r for r in rows if not r.get("error")]
    reasoning_ms = [
        r["ttfc_ms"] - r["ttft_ms"]
        for r in successes
        if r.get("ttfc_ms") is not None and r.get("ttft_ms") is not None
    ]
    reasoning_tok = [r.get("reasoning_tokens") for r in successes]
    served = sorted({r["served_model"] for r in successes if r.get("served_model")})
    out = {
        "n_total": len(rows),
        "n_success": len(successes),
        "ttft_ms": _bench_stat([r.get("ttft_ms") for r in successes]),
        "ttfc_ms": _bench_stat([r.get("ttfc_ms") for r in successes]),
        "decode_tps": _bench_stat([r.get("decode_tps") for r in successes]),
        "total_ms": _bench_stat([r.get("total_ms") for r in successes]),
        "completion_tokens": _bench_stat([r.get("completion_tokens") for r in successes]),
        "prompt_tokens": _bench_stat([r.get("prompt_tokens") for r in successes]),
        "served_models": served,
    }
    # Only surface the reasoning block when the model actually reasoned, so non-thinking
    # runs don't carry a wall of nulls.
    if any(v for v in reasoning_tok if v):
        out["reasoning_tokens"] = _bench_stat(reasoning_tok)
    if reasoning_ms:
        out["reasoning_ms"] = _bench_stat(reasoning_ms)
    errors = [r["error"] for r in rows if r.get("error")]
    if errors:
        # Collapse to distinct messages so a systematic failure reads as one line, not N.
        seen: dict = {}
        for e in errors:
            seen[e] = seen.get(e, 0) + 1
        out["errors"] = [{"message": k, "count": v} for k, v in
                         sorted(seen.items(), key=lambda kv: -kv[1])]
    return out


def _bench_quality_summary(rows: list[dict], suite: list[dict]) -> dict:
    """Aggregate graded results: overall pass rate plus a per-task breakdown, so a model that
    is strong everywhere except one task is distinguishable from one that is mediocre
    throughout — the two look identical in a single average."""
    per_task: dict[str, dict] = {}
    for task in suite:
        per_task[task["id"]] = {"task": task["id"], "runs": 0, "passed_cases": 0,
                                "total_cases": 0, "perfect_runs": 0, "errors": 0}
    for r in rows:
        tid = r.get("task")
        if not tid or tid not in per_task:
            continue
        slot = per_task[tid]
        slot["runs"] += 1
        g = r.get("grade")
        if not g:
            slot["errors"] += 1
            continue
        if g.get("truncated"):
            slot["truncated"] = slot.get("truncated", 0) + 1
        slot["passed_cases"] += g.get("passed", 0)
        slot["total_cases"] += g.get("total", 0)
        if g.get("total") and g.get("passed") == g.get("total"):
            slot["perfect_runs"] += 1
    for slot in per_task.values():
        slot["case_pass_rate"] = (slot["passed_cases"] / slot["total_cases"]
                                  if slot["total_cases"] else None)
        slot["perfect_rate"] = (slot["perfect_runs"] / slot["runs"] if slot["runs"] else None)
    tier_of = {t["id"]: t.get("tier", "core") for t in suite}
    for tid, slot in per_task.items():
        slot["tier"] = tier_of.get(tid, "core")
    tiers = {}
    for tier in sorted({t.get("tier", "core") for t in suite}):
        mine = [v for v in per_task.values() if v["tier"] == tier]
        runs = sum(v["runs"] for v in mine)
        perfect = sum(v["perfect_runs"] for v in mine)
        cases = sum(v["total_cases"] for v in mine)
        passed = sum(v["passed_cases"] for v in mine)
        tiers[tier] = {
            "tasks": len(mine), "runs": runs, "perfect_runs": perfect,
            "perfect_rate": (perfect / runs) if runs else None,
            "case_pass_rate": (passed / cases) if cases else None,
        }
    total_cases = sum(s["total_cases"] for s in per_task.values())
    passed_cases = sum(s["passed_cases"] for s in per_task.values())
    total_runs = sum(s["runs"] for s in per_task.values())
    perfect_runs = sum(s["perfect_runs"] for s in per_task.values())
    return {
        "suite_size": len(suite),
        "case_pass_rate": (passed_cases / total_cases) if total_cases else None,
        "passed_cases": passed_cases,
        "total_cases": total_cases,
        # The headline number: fraction of responses that were fully correct. Partial credit
        # is useful for diagnosis but "mostly right" code is still broken code.
        "perfect_rate": (perfect_runs / total_runs) if total_runs else None,
        "perfect_runs": perfect_runs,
        "total_runs": total_runs,
        "tiers": tiers,
        "tasks": list(per_task.values()),
    }


# The reply needs room alongside the prompt, and the char-per-token estimate drifts by ~15%
# between tokenisers, so a run sized right up to the limit returns empty completions.
_BENCH_CTX_HEADROOM = 0.9


def _bench_ctx_budget(window: int, max_tokens: int) -> float:
    """The largest prompt that fits in `window`. Inverse of _bench_ctx_needed."""
    return (window - int(max_tokens)) * _BENCH_CTX_HEADROOM


def _bench_ctx_needed(prompt_tokens: int, max_tokens: int) -> int:
    """The window a request of this shape needs end to end."""
    return int(math.ceil(int(prompt_tokens) / _BENCH_CTX_HEADROOM)) + int(max_tokens)


def _bench_preflight(model: str, meta: dict, upstream: str, index: dict) -> str | None:
    """Why this run cannot produce a measurement, checked before a single request is sent.

    A doomed run is not free: it still costs the full request count — 54 empty completions
    against a backend serving a different model — and with `exclusive` set every other proxy
    client queues behind it for the duration. The matrix path already skipped cells that could
    not fit; a single run had no equivalent and just sent them.
    """
    if not upstream:
        known = sorted({v.get("model") for v in index.values() if v.get("model")})
        return (f"no reachable backend serves {model!r}"
                + (f" — reachable models: {', '.join(known[:8])}" if known else
                   " — no backend is reachable"))
    if meta.get("benchable") is False:
        return f"{model!r} cannot be benchmarked — {meta.get('bench_detail') or 'it does not answer chat requests'}"
    if meta.get("fits") is False:
        return f"{model!r} does not fit on this machine — {meta.get('fit_detail') or 'too large'}"
    if meta.get("startable") and not meta.get("loaded"):
        return None          # not running yet, but the bench knows how to start it
    if not meta:
        served = sorted(v.get("model") for v in index.values()
                        if v.get("upstream") == upstream and v.get("model"))
        if not served:
            return (f"{upstream} is not serving anything — it is unreachable, or it has no "
                    f"model loaded")
        return (f"{upstream} does not serve {model!r} — it has "
                f"{', '.join(served[:6])}{' …' if len(served) > 6 else ''}")
    return None


# Ollama sizes its default context from TOTAL vram — 262,144 tokens on this box — and
# multiplies it by OLLAMA_NUM_PARALLEL when it starts llama-server. Neither value is
# queryable from outside, so they are mirrored here (override via model_control:
# `ollama_server_ctx` / `ollama_num_parallel`, matching OLLAMA_CONTEXT_LENGTH /
# OLLAMA_NUM_PARALLEL on the service). The KV bytes-per-element matches q8_0 (34/32).
_BENCH_OLLAMA_DEFAULT_CTX = 262144
_BENCH_OLLAMA_PARALLEL = 4
_BENCH_KV_BYTES_PER_ELT = 1.0625


def _bench_ollama_server_ctx(cfg: dict | None = None) -> tuple[int, int]:
    """The per-slot context Ollama will actually load with, and the parallel slot count."""
    mc = load_rules_config().get("model_control") or {}
    per_slot = int((cfg or {}).get("server_context") or 0) \
        or int(mc.get("ollama_server_ctx") or 0) or _BENCH_OLLAMA_DEFAULT_CTX
    parallel = int(mc.get("ollama_num_parallel") or 0) or _BENCH_OLLAMA_PARALLEL
    return per_slot, parallel


def _bench_ollama_kv_mb(model_info: dict, tokens: int) -> float | None:
    """Estimate the KV cache for `tokens` total context, in MB, from /api/show model_info.

    Deliberately conservative: sliding-window layers cost less than this assumes (devstral-2
    measures ~1.7× below the estimate), so an over-estimate can only block a model that was
    going to be tight anyway — the failure mode this exists to prevent is the opposite one,
    where a 108 GB allocation OOM-kills the whole Ollama service.
    """
    arch = model_info.get("general.architecture")
    if not arch:
        return None
    def g(key):
        v = model_info.get(f"{arch}.{key}")
        # Per-layer lists appear on some archs; the max is the conservative read.
        if isinstance(v, list):
            v = max((x for x in v if isinstance(x, (int, float))), default=None)
        return v
    blocks = g("block_count")
    if not blocks:
        return None
    lora = g("attention.kv_lora_rank")
    if lora:
        # MLA (deepseek-style) stores one compressed vector per layer per token, plus the
        # decoupled rope key.
        per_layer_elts = float(lora) + float(g("attention.key_length") or 64)
    else:
        heads_kv = g("attention.head_count_kv")
        if not heads_kv:
            return None
        head = g("attention.head_count")
        k_len = g("attention.key_length") or (
            (g("embedding_length") or 0) / head if head else None)
        v_len = g("attention.value_length") or k_len
        if not k_len:
            return None
        per_layer_elts = float(heads_kv) * (float(k_len) + float(v_len))
    return blocks * per_layer_elts * _BENCH_KV_BYTES_PER_ELT * tokens / (1024 * 1024)


async def _bench_ollama_kv_preflight(model: str, meta: dict, cfg: dict) -> str | None:
    """Why loading this model under Ollama's real context settings would OOM the box.

    The weights-only fit check missed this entirely: devstral-2:123b fits its 65 GB of
    weights easily, but Ollama's vram-based default context (262k × 4 slots) put a 108 GB
    KV cache on top, and the OOM killer took down ollama.service in a load/kill/restart
    loop while the cell burned all 87 of its requests on 502s. Checked before a single
    request is sent, with the arithmetic in the message so the remedy is obvious.
    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as c:
            r = await c.post(f"{OLLAMA_URL}/api/show", json={"model": model})
            if r.status_code >= 400:
                return None                      # unknown must not mean "no"
            info = (r.json() or {}).get("model_info") or {}
    except (httpx.RequestError, ValueError):
        return None
    per_slot, parallel = _bench_ollama_server_ctx(cfg)
    model_ctx = info.get(f"{info.get('general.architecture')}.context_length")
    if isinstance(model_ctx, (int, float)) and model_ctx:
        per_slot = min(per_slot, int(model_ctx))  # ollama clamps to the training context
    tokens = per_slot * parallel
    kv_mb = _bench_ollama_kv_mb(info, tokens)
    total_mb = (_mem_snapshot() or {}).get("total_mb")
    size_mb = meta.get("size_mb")
    if kv_mb is None or not total_mb or not size_mb:
        return None
    need = size_mb * _BENCH_FIT_OVERHEAD + kv_mb
    cap = total_mb - _BENCH_FIT_RESERVE_MB
    if need <= cap:
        return None
    return (f"{model!r} would OOM this box under Ollama's context settings: "
            f"~{size_mb / 1024:.0f} GB weights + ~{kv_mb / 1024:.0f} GB KV cache at "
            f"{per_slot:,} tokens × {parallel} parallel slots > {cap / 1024:.0f} GB usable. "
            f"Set OLLAMA_CONTEXT_LENGTH on the ollama service (and mirror it in "
            f"model_control.ollama_server_ctx) to bench this model.")


# ---- Auto-load: Ollama-style on-demand loading for the fixed backends -----------------------
#
# Opt-in via model_control.auto_load.enabled. A request for a model that lives on a stopped
# backend (or the other vLLM twin) starts what it needs and waits, instead of 404ing until a
# human swaps containers. Other models are unloaded ONLY when the target does not fit in free
# memory — a request must not strip the box out of habit the way a benchmark deliberately does.
_AUTO_LOAD_LOCK = asyncio.Lock()
_AUTO_LOAD_LAST = {"ts": 0.0, "target": ""}


def _auto_load_cfg() -> dict:
    return (load_rules_config().get("model_control") or {}).get("auto_load") or {}


async def _ensure_model_served(model: str, upstream: str) -> dict | None:
    """Make `model` servable on `upstream`, Ollama-style. None = proceed with the request;
    a dict = refuse it with this error/status instead.

    Single-flight: concurrent requests during a load coalesce on one lock, re-check, and
    proceed together. min_hold_s stops two clients ping-ponging the port between the twins."""
    cfg = _auto_load_cfg()
    if not cfg.get("enabled"):
        return None
    if upstream not in ("vllm", "llamacpp"):
        return None                    # ollama and lmstudio already load on demand
    index = await _bench_model_index()
    meta = _bench_resolve_model(index, model, upstream)
    if meta.get("loaded"):
        return None
    if not meta or not (meta.get("startable") or upstream == "llamacpp"):
        return None                    # unknown model: let the upstream 404 honestly
    conn = db()
    busy = conn.execute("SELECT COUNT(*) FROM bench_runs "
                        "WHERE status IN ('running', 'pending')").fetchone()[0]
    conn.close()
    if busy:
        return {"status": 503,
                "error": f"a benchmark owns the box right now; auto-load will not fight it. "
                         f"{model!r} loads once the bench finishes."}
    async with _AUTO_LOAD_LOCK:
        index = await _bench_model_index()
        meta = _bench_resolve_model(index, model, upstream)
        if meta.get("loaded"):
            return None                # someone else loaded it while we queued
        min_hold = float(cfg.get("min_hold_s", 120))
        tgt = f"{upstream}:{model}"
        # HELD, not refused: a request that arrives inside another target's hold window
        # sleeps the window out and then takes its turn. The hold still prevents thrash —
        # two clients on different twins get slow alternation instead of a coin-flip 503.
        while (_AUTO_LOAD_LAST["target"] and _AUTO_LOAD_LAST["target"] != tgt
                and time.time() - _AUTO_LOAD_LAST["ts"] < min_hold):
            await asyncio.sleep(min(2.0, min_hold - (time.time() - _AUTO_LOAD_LAST["ts"])))
            index = await _bench_model_index()
            meta = _bench_resolve_model(index, model, upstream)
            if meta.get("loaded"):
                return None            # the hold ended in our favour after all
        prov = PROVIDERS.get(upstream)
        if prov is None:
            return {"status": 503, "error": f"no provider for {upstream!r}"}
        # The vLLM twins contend for one port: the running one must stop before the target
        # starts, and stopping it is also the cheapest way to free memory. But someone may
        # be mid-stream on it — drain in-flight requests (bounded) before pulling the plug.
        if upstream == "vllm":
            running = await _vllm_container()
            if running and running != meta.get("container"):
                drain_deadline = time.time() + float(cfg.get("drain_s", 120))
                def _busy():
                    now2 = time.time()
                    return any((not v.get("cancelled")) and v.get("upstream") == upstream
                               and now2 - v.get("ts", now2) < 900
                               for v in dict(_INFLIGHT_REQUESTS).values())
                while _busy() and time.time() < drain_deadline:
                    await asyncio.sleep(2)
                await prov.stop()
        # Evict others ONLY if the target does not fit in what is free now. A vLLM boot is
        # priced by its utilization claim, never by checkpoint size — size_mb is usually
        # unknown for containers, and want=0 skipped this check entirely, which is exactly
        # how a 97 GB boot landed beside a 22 GB resident and wedged the box.
        want = (meta.get("size_mb") or 0) * _BENCH_FIT_OVERHEAD
        if upstream == "vllm":
            claim = await _vllm_boot_claim_mb(meta.get("container"))
            if claim:
                want = max(want, claim)
        if want and _free_mem_mb() < want:
            snap = await _bench_residency_snapshot()
            await _bench_free_gpu(snap, keep=model, spare=upstream, want_free_mb=want)
            free_now = _free_mem_mb()
            if free_now < want:
                # Refusing IS the guard: starting anyway starves sshd and the proxy first,
                # so nobody can even reach the box to fix it.
                return {"status": 503,
                        "error": f"not enough memory to bring up {model!r} on {upstream}: "
                                 f"it will claim ~{want / 1024:.0f} GB and only "
                                 f"{free_now / 1024:.0f} GB is free even after eviction. "
                                 f"Refusing to start it rather than wedge the box."}
        timeout = float(cfg.get("ready_timeout_s", 900))
        payload: dict = {"ready_timeout_s": timeout, "wait_s": timeout}
        if meta.get("container"):
            payload["container"] = meta["container"]
        res = await prov.load(payload, model)
        ok, detail = _load_result(res)
        if not ok:
            return {"status": 503,
                    "error": f"auto-load could not bring up {model!r} on {upstream}: "
                             f"{detail or 'it did not become ready in time'}"}
        _AUTO_LOAD_LAST.update(ts=time.time(), target=tgt)
        return None


async def _bench_start_backend(meta: dict, persist: bool = True,
                               bench_id: str = "") -> dict:
    """Make room, bring up the stopped backend the run needs, and say how to undo both.

    Freeing first is a precondition, not a courtesy. A fixed backend that cannot allocate never
    becomes ready, and the symptom is indistinguishable from a slow load — vLLM wants ~99 GB on
    a 121 GB box, so llama.cpp holding 90 GB of it means the start can only ever time out. It
    did, for seven minutes, and read as "vLLM is broken".
    """
    upstream, container = meta.get("upstream") or "", meta.get("container")
    prov = PROVIDERS.get(upstream)
    if prov is None:
        return {"ok": False, "started": False, "restore": None,
                "detail": f"unknown upstream {upstream!r}"}
    if bench_id:
        _bench_phase(bench_id, f"freeing memory for {upstream}")
    snap = await _bench_residency_snapshot()
    want = meta.get("size_mb")
    want_free = (want * _BENCH_FIT_OVERHEAD) if want else 0
    if upstream == "vllm":
        # The utilization claim, not the checkpoint: without this, an unsized vLLM meta
        # meant want_free=0 and the bench started a ~97 GB boot without waiting for the
        # evicted memory to actually come back.
        claim = await _vllm_boot_claim_mb(meta.get("container"))
        if claim:
            want_free = max(want_free, claim)
    freed = await _bench_free_gpu(snap, keep=meta.get("model") or "", bench_id=bench_id,
                                  want_free_mb=want_free)
    # Persisted before anything is started: a proxy restart between here and the restore would
    # otherwise leave the box stripped, with nothing recording what had been running. A cell
    # inside a sweep does not persist: the sweep already recorded the state before its first
    # cell, and this snapshot is of the box mid-run, which would restore the wrong thing.
    if persist:
        _save_pending_residency(snap)
    if bench_id:
        _bench_phase(bench_id, f"starting {upstream} and waiting for it to serve "
                               f"(a large model takes minutes)")
    # A bench is already committed to waiting, and 420s is not enough for a large vLLM
    # checkpoint on this box — qwen3-coder-next was stopped mid-load at the deadline and
    # recorded as a failure. Waiting longer is only safe because died() now covers docker as
    # well as systemd: a container that actually exits is caught in seconds, so the ceiling
    # only ever applies to something genuinely still loading.
    payload = {"container": container} if container else {}
    payload["wait_s"] = _BENCH_START_READY_S
    payload["ready_timeout_s"] = _BENCH_START_READY_S
    # Timed here because this is where the expensive part happens. For an on-demand backend
    # the warm-up request IS the load, and env["load_ms"] measures it correctly — but a
    # container backend loads its weights BEFORE the first request, so the warm-up saw an
    # already-hot server and reported 16s against a boot that really took ~6 minutes
    # (qwen3-coder-next: 4.7 min of weight loading alone, per the container's own log).
    # Twenty-fold under-reporting, in the column a reader uses to compare engines.
    _t_start = time.perf_counter()
    res = await prov.load(payload, meta.get("model") or "")
    start_ms = round((time.perf_counter() - _t_start) * 1000)
    ok, detail = _load_result(res)
    # `ok` is "it is serving"; this is "the box was changed". vLLM's load starts the container
    # and *then* waits for readiness, so a timeout leaves it running. Reporting nothing to undo
    # left one crash-looping under restart=unless-stopped for thirteen minutes, competing for
    # the memory it had just failed to get.
    started = ok or bool(isinstance(res, dict) and res.get("started_container"))
    return {"ok": ok, "started": started, "detail": detail, "freed": freed,
            "backend_start_ms": start_ms,
            "restore": {"upstream": upstream, "stop": started,
                        "residency": snap if persist else None}}


async def _bench_restore_backend(token: dict | None) -> dict | None:
    """Undo both halves, in order: stop what the bench started, then put back what it stopped.

    Stopping first is deliberate — vLLM holds ~99 GB whether or not anything is asking it, and
    llama.cpp cannot reload its weights until that memory comes back.
    """
    if not token:
        return None
    out: dict = {}
    if token.get("stop"):
        prov = PROVIDERS.get(token.get("upstream"))
        if prov is not None:
            try:
                out["stopped"] = await prov.stop()
            except Exception as e:
                out["stopped"] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}
    snap = token.get("residency")
    if snap:
        try:
            out["residency"] = await _bench_restore_residency(snap)
            _save_pending_residency(None)
        except Exception as e:
            out["residency"] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}
    return out


async def _bench_fit_context(upstream: str, needed: int, concurrency: int,
                             exact: bool = False, bench_id: str = "") -> dict:
    """Make `upstream` serve at least `needed` tokens per slot, or say why it cannot.

    Returns {ok, detail, changed, restore}. `restore` is the token for _bench_restore_context,
    and is None when nothing was changed and so nothing needs putting back.
    """
    prov = PROVIDERS.get(upstream)
    if prov is None or not prov.resizable_context:
        return {"ok": False, "changed": False, "restore": None,
                "detail": f"{upstream} cannot change its context window"}
    if bench_id:
        _bench_phase(bench_id, f"restarting {upstream} at a {needed:,}-token window "
                               f"and reloading its weights")
    res = await prov.resize_context(needed, concurrency, exact=exact)
    prev = res.get("previous")
    return {"ok": bool(res.get("ok")), "changed": bool(res.get("changed")),
            "detail": res.get("detail") or "",
            "restore": dict(prev, upstream=upstream) if prev else None}


async def _bench_restore_context(token: dict | None) -> dict | None:
    """Put the window back where the bench found it.

    Same reasoning as the residency handshake: a bench that permanently reconfigures the box
    silently changes what the *next* bench measures, and the difference would be read as a
    property of the model.
    """
    if not token:
        return None
    prov = PROVIDERS.get(token.get("upstream"))
    if prov is None:
        return None
    ok, detail = _load_result(await prov.load(
        {"context_length": token["context_length"], "parallel": token["parallel"],
         "ready_timeout_s": _BENCH_RESIZE_READY_S}, ""))
    return {"ok": ok, "detail": detail, "context_length": token["context_length"],
            "parallel": token["parallel"]}




async def _bench_env_snapshot() -> dict:
    """Machine + engine state at run time. Without this a number from three weeks ago is
    uninterpretable: you can't tell which quant was loaded, how much context it was serving,
    or whether the GPU was already half-full when the run started."""
    env: dict = {"ts": time.time(), "proxy_version": __version__}
    try:
        env["hw"] = _host_hw_facts()
    except Exception:
        pass
    try:
        env["toolchains"] = _bench_graders_mod.toolchain_versions()
    except Exception:
        pass
    try:
        # system_now is sync so Starlette runs it in the threadpool; awaiting it raises
        # TypeError into the blanket except below, and every run since has recorded an error
        # string instead of the GPU, memory and engine state it exists to capture. Same
        # mistake, same shape, second location — _bench_model_index had it too.
        sysinfo = await asyncio.to_thread(system_now)
        gpus = sysinfo.get("gpus") or []
        env["gpus"] = [
            {"name": g.get("name"), "mem_used_mb": g.get("mem_used_mb"),
             "mem_total_mb": g.get("mem_total_mb"), "util_pct": g.get("util_pct")}
            for g in gpus if isinstance(g, dict)
        ]
        env["mem"] = sysinfo.get("mem")
        ollama = sysinfo.get("ollama") or {}
        env["ollama_loaded"] = [
            {"name": m.get("name") or m.get("model"), "size_vram": m.get("size_vram")}
            for m in (ollama.get("ps") or []) if isinstance(m, dict)
        ]
        env["ollama_config"] = ollama.get("config")
        env["ollama_version"] = ollama.get("version")
        lms = sysinfo.get("lmstudio") or {}
        env["lmstudio_loaded"] = [m.get("id") for m in (lms.get("models") or [])
                                  if isinstance(m, dict)]
        vllm = sysinfo.get("vllm") or {}
        env["vllm_models"] = [m.get("id") for m in (vllm.get("models") or [])
                              if isinstance(m, dict)]
    except Exception as e:
        env["error"] = f"{type(e).__name__}: {e}"
        # Not silent: this is the only record of what the machine looked like, and a run whose
        # environment is missing is a number nobody can interpret later.
        try:
            print(f"[bench] environment snapshot failed: {type(e).__name__}: {e}")
        except Exception:
            pass
    return env


async def _bench_evict_ollama(keep: str = "") -> list:
    """Unload every Ollama model except `keep`. Returns the names unloaded.

    This is the only per-model residency control any of the three backends offers over HTTP.
    LM Studio exposes no unload endpoint (its JIT TTL handles eviction on its own schedule), and
    vLLM serves exactly the model its server was launched with — so a benchmark cannot arrange
    for its target to be the only resident model, only for Ollama not to be holding VRAM it
    doesn't need to.
    """
    unloaded: list = []
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as c:
            ps = (await c.get(f"{OLLAMA_URL}/api/ps")).json()
            for m in (ps.get("models") or []):
                name = m.get("name") or m.get("model")
                if not name or name == keep:
                    continue
                # keep_alive=0 is Ollama's "unload now"; there is no separate stop endpoint.
                await c.post(f"{OLLAMA_URL}/api/generate",
                             json={"model": name, "keep_alive": 0, "prompt": ""})
                unloaded.append(name)
    except Exception as e:
        # Not silent: the caller takes an empty list to mean the GPU is clear, and proceeds to
        # measure a contended box as if it were quiet. Three outages this codebase has had were
        # a live failure swallowed exactly here.
        try:
            print(f"[bench] Ollama eviction failed: {type(e).__name__}: {e}")
        except Exception:
            pass
    return unloaded


async def _bench_loaded_context(model: str, upstream: str) -> int | None:
    """The context window the backend ACTUALLY loaded, read after the model is up.

    Not the same as what was asked for. A preload at num_ctx=65536 is discarded the moment a
    request arrives carrying no context hint — Ollama reloads at OLLAMA_CONTEXT_LENGTH, which
    here is 262144. That happened silently in the middle of a benchmark and nothing in the
    report could show it, because the only context the report knew was the catalogue's.
    """
    try:
        if upstream == "ollama":
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as c:
                ps = (await c.get(f"{OLLAMA_URL}/api/ps")).json()
            for m in (ps.get("models") or []):
                if (m.get("name") or m.get("model")) == model:
                    return m.get("context_length") or None
            return None
        if upstream == "vllm":
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as c:
                data = (await c.get(f"{VLLM_URL}/v1/models")).json()
            for m in (data.get("data") or []):
                if m.get("id") == model or len(data.get("data") or []) == 1:
                    return m.get("max_model_len") or m.get("context_length") or None
    except (httpx.RequestError, ValueError, KeyError, TypeError):
        return None
    return None


async def _bench_resident_mb(model: str, upstream: str) -> float | None:
    """How much the model is actually holding, as opposed to how big it is on disk.

    A 75 GB checkpoint with its KV cache allocated is a different number, and it is the one
    that decides what else fits beside it. Ollama reports it per model; the others are inferred
    from what the machine lost, which is why this is read after the model is up and before
    anything else moves.
    """
    try:
        if upstream == "ollama":
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as c:
                ps = (await c.get(f"{OLLAMA_URL}/api/ps")).json()
            for m in (ps.get("models") or []):
                if (m.get("name") or m.get("model")) == model:
                    v = m.get("size_vram") or m.get("size")
                    return round(v / 1048576) if v else None
            return None
    except (httpx.RequestError, ValueError, KeyError):
        return None
    # vLLM holds its weights inside a container that was already running, so the caller's
    # "what did the machine lose while this loaded" fallback measures nothing — it reported
    # 712 MB for a 20 GB model. Ask the container what it is holding instead.
    if upstream == "vllm":
        try:
            name = await _vllm_container()
            docker = _docker_bin()
            if name and docker:
                rc, out = await _run_cmd(
                    [docker, "stats", "--no-stream", "--format", "{{.MemUsage}}", name],
                    timeout=25.0)
                if rc == 0:
                    return _docker_mem_mb(out)
        except Exception:
            return None          # an unreadable container costs the metric, not the run
    return None


def _docker_mem_mb(usage: str) -> float | None:
    """Megabytes from a `docker stats` MemUsage cell, e.g. '12.34GiB / 121GiB' -> 12636."""
    m = re.match(r"\s*([\d.]+)\s*([KMGT]?i?B)", usage or "", re.I)
    if not m:
        return None
    scale = {"b": 1 / 1048576, "kib": 1 / 1024, "kb": 1 / 1024, "mib": 1.0, "mb": 1.0,
             "gib": 1024.0, "gb": 1024.0, "tib": 1048576.0, "tb": 1048576.0}
    factor = scale.get(m.group(2).lower())
    return round(float(m.group(1)) * factor) if factor else None


async def _bench_reload_ollama(names: list) -> list:
    """Make previously-resident Ollama models resident again.

    Eviction used to be opt-in, so losing them was the point. Now that every run evicts, the
    models have to come back — otherwise a three-minute benchmark quietly leaves the daily
    driver cold and the next request pays a full load with nothing explaining it.
    """
    back = []
    for name in names or []:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as c:
                r = await c.post(f"{OLLAMA_URL}/api/generate",
                                 json={"model": name, "prompt": "", "keep_alive": "30m"})
                if r.status_code < 400:
                    back.append(name)
        except httpx.RequestError:
            pass
    return back


async def _bench_residency_snapshot() -> dict:
    """Everything currently holding the GPU, so it can be put back exactly as found.

    Discovered rather than declared: the caller shouldn't have to know that vLLM is a container
    and ComfyUI is a unit and Ollama is an HTTP keep_alive. Whatever is up when the bench starts
    is what comes back when it ends.
    """
    snap: dict = {"taken_at": time.time(), "backends": [], "ollama": []}
    for name, b in list(PROVIDERS.items()) + list(SIDE_SERVICES.items()):
        if b.control == "none":
            continue          # nothing to stop or restore
        try:
            st = await b.state()
            running = st.get("running")
            container = st.get("container") if b.control == "docker" else None
            if b.control == "docker" and not container:
                container = await _vllm_container()
            if running is None and b.control == "docker":
                # docker's state() reports the container, not whether it is up.
                if container:
                    code, out = await _run_cmd(
                        [_docker_bin(), "inspect", container, "--format", "{{.State.Running}}"],
                        15.0, max_chars=100)
                    running = code == 0 and out.strip().lower() == "true"
            entry = {"name": name, "was_running": bool(running), "control": b.control}
            if container:
                # WHICH container matters, not just that "vllm was up": several can be
                # configured for the same port (qwen-vllm / ornith-vllm), and a restore
                # that rediscovers by port starts whichever sorts first — the sweep that
                # measured Ornith all afternoon handed the box back running qwen instead.
                entry["container"] = container
            snap["backends"].append(entry)
        except Exception as e:
            snap.setdefault("errors", []).append(f"{name}: {type(e).__name__}: {e}")
    # Ollama holds models rather than a process the proxy starts, so it is recorded separately.
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as c:
            ps = (await c.get(f"{OLLAMA_URL}/api/ps")).json()
            snap["ollama"] = [m.get("name") or m.get("model")
                              for m in (ps.get("models") or []) if (m.get("name") or m.get("model"))]
    except Exception:
        pass
    return snap


def _free_mem_mb() -> float:
    m = _mem_snapshot() or {}
    return float(m.get("avail_mb") or 0)


async def _vllm_boot_claim_mb(container: str | None) -> float | None:
    """What starting this vLLM container will actually take from the box.

    NOT the checkpoint size: vLLM pre-allocates gpu_memory_utilization × total memory at
    boot (weights + KV pool + CUDA graphs), and on a unified-memory box that claim IS
    system RAM. Sizing a boot by its 43 GB checkpoint while it grabbed 97 GB is how
    qwen-vllm landed on top of a resident gemma4 and wedged the machine — sshd and the
    proxy starved while the resident backends kept answering. Utilization is read from
    the container's own launch args; vLLM's default is 0.9 when the flag is absent.
    None when there is nothing to compute with (no total, no such container)."""
    total = (_mem_snapshot() or {}).get("total_mb")
    if not total:
        return None
    util = 0.90
    try:
        for c in await _vllm_configs():
            if not container or c.get("container") == container:
                if c.get("gpu_memory_utilization") is not None:
                    util = float(c["gpu_memory_utilization"])
                break
    except (ValueError, TypeError):
        pass
    return float(total) * util


async def _bench_free_gpu(snap: dict, keep: str = "", want_free_mb: float = 0,
                          timeout_s: float = 240.0, spare: str = "",
                          bench_id: str = "") -> dict:
    """Stop everything in the snapshot, then wait for the memory to actually come back.

    The waiting is the part that matters. `docker stop` returns as soon as the process is
    signalled, but unmapping ~99 GB takes seconds longer — starting the run at that moment
    measures a machine that is still tidying up, which is exactly the kind of contaminated
    number a benchmark exists to avoid.
    """
    out: dict = {"stopped": [], "evicted_ollama": []}
    for entry in (snap.get("backends") or []):
        if not entry.get("was_running"):
            continue          # restoring must not start what was already down
        if spare and entry["name"] == spare:
            continue          # the backend about to be measured
        b = backend(entry["name"])
        if not b:
            continue
        res = await b.stop()
        out["stopped"].append({"name": entry["name"], **res})
    out["evicted_ollama"] = await _bench_evict_ollama(keep)

    before = _free_mem_mb()
    if want_free_mb:
        # `docker stop` and `systemctl stop` return when the process is signalled, not when its
        # memory is back. Unmapping ~90 GB takes appreciably longer, and starting the next model
        # into a machine that is still handing memory back is how a 36 GB model ended up with 28
        # GB resident and took 24 minutes a request. Wait for the memory, not for the signal.
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            now_free = _free_mem_mb()
            if now_free >= want_free_mb:
                break
            if bench_id:
                _bench_phase(bench_id,
                             f"waiting for memory to come back — {now_free / 1024:.0f} GB free, "
                             f"need {want_free_mb / 1024:.0f} GB")
            await asyncio.sleep(2.0)
    else:
        # Even without a target, give the kernel a moment to finish reclaiming.
        await asyncio.sleep(3.0)
    out["free_mb_before"] = before
    out["free_mb_after"] = _free_mem_mb()
    out["reached_target"] = (not want_free_mb) or out["free_mb_after"] >= want_free_mb
    out["wanted_mb"] = want_free_mb or None
    return out


async def _bench_restore_residency(snap: dict) -> dict:
    """Put back exactly what the snapshot found running, and wait for it to be usable.

    vLLM measured ~9 minutes to reload its weights, so this returns only once it answers —
    reporting "restored" while the daily driver is still refusing connections would be worse
    than saying nothing.
    """
    res: dict = {"started": [], "reloaded_ollama": []}
    for entry in (snap.get("backends") or []):
        if not entry.get("was_running"):
            continue
        b = backend(entry["name"])
        if not b:
            continue
        started = await b.start(container=entry.get("container")) \
            if entry.get("container") else await b.start()
        # Started is not serving. vLLM measured ~9 minutes to reload its weights, and reporting
        # "restored" while the daily driver still refuses connections is worse than silence.
        # Started is not serving; ask the backend itself rather than mapping names to waiters.
        ready = await b.ready(_vllm_ready_timeout()) if isinstance(b, Provider) else None
        res["started"].append({"name": entry["name"], **started, "ready": ready})
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as c:
            for name in (snap.get("ollama") or []):
                await c.post(f"{OLLAMA_URL}/api/generate",
                             json={"model": name, "prompt": "", "keep_alive": "30m"})
                res["reloaded_ollama"].append(name)
    except Exception:
        pass
    return res


def _vllm_ready_timeout() -> float:
    try:
        return float((load_rules_config().get("model_control") or {}).get(
            "vllm_ready_timeout_s") or 420)
    except Exception:
        return 420.0


# A bench that stops the daily driver and then dies would leave it stopped indefinitely. The
# snapshot is persisted the moment anything is stopped and cleared only once it is restored, so
# a restart can finish the job — the same shape as failing orphaned bench rows at startup.
_RESIDENCY_SETTING = "bench_residency_pending"


def _save_pending_residency(snap: dict | None):
    try:
        if snap:
            set_setting(_RESIDENCY_SETTING, json.dumps(snap))
        else:
            delete_setting(_RESIDENCY_SETTING)
    except Exception:
        pass


async def _restore_pending_residency() -> dict | None:
    """Called at startup: if a previous run stopped things and never put them back, do it now."""
    try:
        row = get_setting(_RESIDENCY_SETTING)
    except Exception:
        return None
    # get_setting hands back the whole row ({"value", "updated_ts"}), not the stored string —
    # treating the wrapper as the snapshot silently "restored" nothing at all.
    raw = row.get("value") if isinstance(row, dict) else row
    if not raw:
        return None
    try:
        snap = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        _save_pending_residency(None)
        return None
    if not isinstance(snap, dict):
        _save_pending_residency(None)
        return None
    try:
        print("[init] a previous benchmark left services stopped — restoring")
    except Exception:
        pass
    res = await _bench_restore_residency(snap)
    _save_pending_residency(None)
    return res


async def _bench_quiesce(enable: bool, prior: dict | None = None, keep: str = "",
                         want_free_mb: float = 0) -> dict:
    """Quiet the box for a clean run, and put it back exactly as found.

    Gating proxy traffic is not enough on its own: vLLM holds ~99 GB whether or not anything is
    asking it, and an idle Ollama model still occupies VRAM. On this box that left ~19 GB free —
    less than the smallest model worth benchmarking — so a run either failed to load or measured
    a contended GPU.

    So quiescing now takes a snapshot of everything running, stops all of it, waits for the
    memory to actually come back, and restores the same set afterwards. `keep` spares the model
    the next cell needs so it isn't evicted and immediately reloaded.
    """
    state: dict = {"panic_was": _PANIC_MODE, "ollama_was": []}
    if enable:
        try:
            _set_panic_mode(True)
        except Exception as e:
            state["panic_error"] = f"{type(e).__name__}: {e}"
        # Discover before stopping anything: whatever is up now is what gets put back.
        snap = await _bench_residency_snapshot()
        state["residency"] = snap
        # Persist first. If this process dies between here and the restore, startup finds it.
        _save_pending_residency(snap)
        state["freed"] = await _bench_free_gpu(snap, keep=keep, want_free_mb=want_free_mb)
        # Ollama models the snapshot already recorded; don't double-count them on restore.
        state["ollama_was"] = []
        return state
    # Restore.
    prior = prior or {}
    try:
        if not prior.get("panic_was"):
            _set_panic_mode(False)
    except Exception:
        pass
    snap = prior.get("residency")
    if snap:
        prior["restored"] = await _bench_restore_residency(snap)
        _save_pending_residency(None)
        return prior
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as c:
            for name in (prior.get("ollama_was") or []):
                # Zero-token generate warms the model back into VRAM without producing output.
                await c.post(f"{OLLAMA_URL}/api/generate",
                             json={"model": name, "prompt": "", "keep_alive": "30m"})
    except Exception:
        pass
    return prior


def _bench_expand_matrix(model_axis: list, cfg: dict) -> list[dict]:
    """Expand axis lists into one cell per combination. Axes are models × prompt_tokens ×
    thinking × temperature; anything given as a scalar is treated as a single-value axis."""
    def as_list(v, default):
        if v is None:
            return [default]
        if isinstance(v, list):
            return v or [default]
        return [v]

    prompts = as_list(cfg.get("prompt_tokens"), 0)
    thinks = as_list(cfg.get("thinking"), "auto")
    temps = as_list(cfg.get("temperature"), None)
    # "cold" salts every prompt so nothing can be served from the prefix cache; "cached" repeats
    # an identical prompt so everything after the warm-up should hit it. Comparing the pair is
    # the only way to see whether a backend's caching is working at all — a backend with caching
    # silently disabled looks merely slow, not misconfigured.
    caches = as_list(cfg.get("cache"), None)
    concs = as_list(cfg.get("concurrency"), 1)
    # The window the *server* is launched with, as opposed to prompt_tokens, which is how much
    # is sent. They are different questions — "is a model slower when it has a 262k KV pool
    # allocated, even for a short prompt?" is this axis, and there was no way to ask it before.
    # It is also the only axis whose value the bench must change on the box between cells.
    servers = as_list(cfg.get("server_context"), None)
    cells = []
    for m in model_axis:
        # Entries may be bare names (older callers) or {model, upstream} pairs, so a sweep can
        # span backends — the same weights on LM Studio and on vLLM are two different cells.
        name = m.get("model") if isinstance(m, dict) else m
        up = (m.get("upstream") or "") if isinstance(m, dict) else ""
        # server_context only means anything where the window is a launch argument. Expanding
        # it across every model turned one preset into three identical cells per Ollama model,
        # all of which failed at the gate with "ollama cannot change its context window" — 84
        # failures out of 102 from a single tick box. An axis a backend cannot honour is not a
        # failure for that backend, it is simply not an axis.
        _prov = PROVIDERS.get(up) if up else None
        my_servers = servers if (_prov is not None and _prov.resizable_context) else [None]
        for pt in prompts:
            for th in thinks:
                for tp in temps:
                    for ca in caches:
                        for cc in concs:
                            for sv in my_servers:
                                axes = {"model": name, "prompt_tokens": int(pt or 0),
                                        "thinking": str(th or "auto")}
                                if up:
                                    axes["upstream"] = up
                                if tp is not None:
                                    axes["temperature"] = float(tp)
                                if ca:
                                    axes["cache"] = str(ca)
                                if int(cc or 1) != 1:
                                    axes["concurrency"] = int(cc)
                                if sv:
                                    axes["server_context"] = int(sv)
                                cells.append(axes)
    # Each distinct server_context costs a restart and a full model reload, so run the cells
    # that share one together. A stable sort keeps the order within each group, so the cache
    # axis still sees cold before cached.
    if any(c.get("server_context") for c in cells):
        cells.sort(key=lambda c: c.get("server_context") or 0)
    # Backend last, so it is the outermost grouping. Two backends rarely fit on one box at
    # once, so every switch means stopping one and loading the other — interleaving them would
    # pay that repeatedly, and a mixed sweep once ran llama.cpp's cell while it was stopped for
    # vLLM's.
    if len({c.get("upstream") for c in cells}) > 1:
        cells.sort(key=lambda c: str(c.get("upstream") or ""))
    return cells


def _bench_cell_label(axes: dict) -> str:
    bits = [_bench_model_display(axes.get("model") or "?")]
    if axes.get("upstream"):
        bits.append(f"@{axes['upstream']}")
    pt = axes.get("prompt_tokens") or 0
    # "32k ctx" for a prompt size read as a context setting, which is exactly the confusion
    # this axis pair caused. Prompts say "prompt", windows say "ctx".
    bits.append(f"{int(pt) // 1000}k prompt" if pt >= 1000
                else (f"{pt} prompt" if pt else "short"))
    if axes.get("server_context"):
        bits.append(f"{int(axes['server_context']) // 1024}k ctx")
    if axes.get("thinking") and axes["thinking"] != "auto":
        bits.append(f"think={axes['thinking']}")
    if axes.get("cache"):
        bits.append(str(axes["cache"]))
    if axes.get("concurrency"):
        bits.append(f"{axes['concurrency']}×parallel")
    if axes.get("temperature") is not None:
        bits.append(f"t={axes['temperature']}")
    return " · ".join(bits)


async def _bench_execute_suite(parent_id: str, app: FastAPI):
    """Run a matrix submission: expand the axes, then run each cell serially as its own child
    bench so every cell gets a full set of percentiles. Serial by design — running cells
    concurrently would have them contend for the same GPU and corrupt every number."""
    conn = db()
    row = conn.execute("SELECT * FROM bench_runs WHERE id=?", (parent_id,)).fetchone()
    if not row or row["status"] != "pending":
        conn.close()
        return
    index = await _bench_model_index()
    try:
        cfg = json.loads(row["config_json"])
    except (json.JSONDecodeError, TypeError):
        conn.execute("UPDATE bench_runs SET status='failed', error='invalid config JSON', finished_ts=? WHERE id=?",
                     (time.time(), parent_id))
        conn.commit()
        conn.close()
        return
    creator_ip = row["creator_ip"]
    models = cfg.get("models") or [row["model"]]
    cells = _bench_expand_matrix(models, cfg)
    child_ids = []
    skipped = 0
    now = time.time()
    # Largest window any cell needs, and the widest concurrency it needs it at. Collected while
    # planning so the backend is sized once, before the first cell runs.
    prior = _bench_completed_cells(conn) if cfg.get("resume") else {}
    reused = 0
    need_ctx, need_par, need_up = 0, 1, ""
    ctx_axis_up = ""      # set when the sweep varies the window itself, per cell
    start_meta = None     # a configured-but-stopped backend this sweep needs
    for axes in cells:
        cid = "b_" + uuid.uuid4().hex[:12]
        child_cfg = dict(cfg)
        child_cfg.pop("models", None)
        child_cfg["prompt_tokens"] = axes["prompt_tokens"]
        child_cfg["thinking"] = axes["thinking"]
        if axes.get("temperature") is not None:
            child_cfg["temperature"] = axes["temperature"]
        # Per-cell upstream — this is what lets one sweep compare the same model across two
        # backends without them collapsing into each other.
        if axes.get("upstream"):
            child_cfg["upstream"] = axes["upstream"]
        if axes.get("cache"):
            # The cache axis IS the randomize flag, named for what it measures rather than for
            # the mechanism, so a result column reads "cold / cached" not "randomize true/false".
            child_cfg["cache"] = axes["cache"]
            child_cfg["randomize"] = (axes["cache"] == "cold")
        # The one axis a cell has to apply to the box itself, so the child owns it.
        child_cfg["server_context"] = axes.get("server_context") or None
        # Always write the scalar, including the default. Copying it only when it differed left
        # every concurrency-1 cell holding the parent's [1, 4] list, and int() on a list raises.
        child_cfg["concurrency"] = axes.get("concurrency") or 1
        # The suite owns quiescing for the whole sweep; children must not toggle it per cell.
        child_cfg["quiesce"] = False
        meta = _bench_resolve_model(index, axes["model"], axes.get("upstream", ""))
        window = meta.get("loaded_context") or meta.get("max_context")
        budget = (_bench_ctx_budget(window, child_cfg.get("max_tokens", 256))
                  if window else None)
        cell_up = axes.get("upstream") or meta.get("upstream") or ""
        cell_prov = PROVIDERS.get(cell_up)
        if meta.get("startable") and not meta.get("loaded") and start_meta is None:
            start_meta = meta
        if axes.get("server_context"):
            # The sweep is varying the window on purpose. Sizing it once here would flatten the
            # axis into a constant — which is what it did to a prompt sweep that meant this.
            ctx_axis_up = ctx_axis_up or cell_up
        elif budget and axes["prompt_tokens"] > budget and cell_prov is not None \
                and cell_prov.resizable_context:
            # Resizable: record what the sweep needs and let the cell run. Sizing once to the
            # largest cell rather than per-cell is both far cheaper (one restart, not 24) and
            # more comparable — every cell then measures against the same KV pool, so a
            # difference between them is the axis rather than the window.
            need_ctx = max(need_ctx,
                           _bench_ctx_needed(axes["prompt_tokens"],
                                             child_cfg.get("max_tokens", 256)))
            need_par = max(need_par, int(child_cfg["concurrency"]))
            need_up = need_up or cell_up
        elif budget and axes["prompt_tokens"] > budget:
            conn.execute(
                """INSERT INTO bench_runs (id, ts, model, config_json, status, creator_ip,
                                           parent_id, axes_json, label, error, finished_ts)
                   VALUES (?, ?, ?, ?, 'skipped', ?, ?, ?, ?, ?, ?)""",
                (cid, now, axes["model"], json.dumps(child_cfg), creator_ip, parent_id,
                 json.dumps(axes), _bench_cell_label(axes),
                 f"skipped — {axes['prompt_tokens']:,} prompt tokens will not fit "
                 f"{axes['model']}'s {window:,}-token window", now),
            )
            skipped += 1
            continue
        old = prior.get(_bench_cell_sig(axes["model"], child_cfg))
        if old is not None:
            # Copied rather than re-parented, so the run it came from keeps its own report
            # intact. The copy records where the numbers came from — a measurement whose
            # provenance is invisible is worse than no measurement.
            try:
                oenv = json.loads(old["env_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                oenv = {}
            oenv["reused_from"] = old["id"]
            oenv["reused_measured_at"] = old["finished_ts"]
            conn.execute(
                """INSERT INTO bench_runs (id, ts, model, config_json, status, creator_ip,
                                           parent_id, axes_json, label, results_json, env_json,
                                           started_ts, finished_ts, progress, progress_total)
                   VALUES (?, ?, ?, ?, 'done', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (cid, now, axes["model"], json.dumps(child_cfg), creator_ip, parent_id,
                 json.dumps(axes), _bench_cell_label(axes), old["results_json"],
                 json.dumps(oenv), old["started_ts"], old["finished_ts"],
                 old["progress"], old["progress_total"]),
            )
            reused += 1
            continue
        conn.execute(
            """INSERT INTO bench_runs (id, ts, model, config_json, status, creator_ip,
                                       parent_id, axes_json, label)
               VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?)""",
            (cid, now, axes["model"], json.dumps(child_cfg), creator_ip, parent_id,
             json.dumps(axes), _bench_cell_label(axes)),
        )
        child_ids.append(cid)
    conn.execute(
        "UPDATE bench_runs SET status='running', started_ts=?, progress=0, progress_total=?, "
        "results_json=? WHERE id=?",
        (time.time(), len(child_ids),
         json.dumps({"children": child_ids, "skipped": skipped, "reused": reused}), parent_id),
    )
    conn.commit()
    conn.close()

    # Deliberately not evicting here: each child evicts for itself, sparing its own model, so a
    # multi-model sweep measures each model without the previous one still holding VRAM. A single
    # eviction up front would only help the first cell.
    quiesce_state = None
    if cfg.get("quiesce"):
        quiesce_state = await _bench_quiesce(True, keep="")
    # Record the box as it was before the first cell. Cells start and stop backends between
    # themselves — a mixed sweep switches backends because two rarely fit at once — so the only
    # snapshot worth restoring is the one taken before any of that happened.
    # Unconditional: every cell evicts, and only this snapshot knows what was resident before
    # the first one did. Taken here whether or not a backend switch is coming.
    orig_residency = await _bench_residency_snapshot()
    _save_pending_residency(orig_residency)
    ctx_restore = None
    if need_ctx:
        fit = await _bench_fit_context(need_up, need_ctx, need_par, bench_id=parent_id)
        if not fit["ok"]:
            conn = db()
            conn.execute(
                "UPDATE bench_runs SET status='failed', error=?, finished_ts=? WHERE id=?",
                (f"could not size {need_up} to the {need_ctx:,}-token window this sweep "
                 f"needs: {fit['detail']}", time.time(), parent_id))
            conn.commit()
            conn.close()
            if quiesce_state is not None:
                await _bench_quiesce(False, quiesce_state)
            return
        ctx_restore = fit["restore"]
    elif ctx_axis_up:
        # Cells will each set their own window, so capture where to put it back *before* the
        # first one changes it — there is no single "previous" to recover afterwards.
        prov = PROVIDERS.get(ctx_axis_up)
        if prov is not None and prov.resizable_context:
            slot, par = await prov.context_window()
            if slot:
                ctx_restore = {"upstream": ctx_axis_up,
                               "context_length": slot * par, "parallel": par}
    try:
      try:
        for i, cid in enumerate(child_ids, start=1):
            # One bad cell must not abandon the other 23. Before this, an exception here
            # propagated out of a fire-and-forget task, which dies silently — leaving the
            # parent 'running' and every child 'pending' with nothing to explain why.
            try:
                await _bench_execute(cid, app)
            except Exception as e:
                conn = db()
                conn.execute(
                    "UPDATE bench_runs SET status='failed', error=?, finished_ts=? WHERE id=?",
                    (f"{type(e).__name__}: {e}", time.time(), cid))
                conn.commit()
                conn.close()
            conn = db()
            conn.execute("UPDATE bench_runs SET progress=? WHERE id=?", (i, parent_id))
            conn.commit()
            conn.close()
      except Exception as e:
        # A fire-and-forget task that raises disappears without trace. Record it on the parent,
        # or the sweep looks like it is still running for the rest of the process's life.
        conn = db()
        conn.execute(
            "UPDATE bench_runs SET status='failed', error=?, finished_ts=? WHERE id=?",
            (f"{type(e).__name__}: {e}", time.time(), parent_id))
        conn.commit()
        conn.close()
        return
    finally:
        if quiesce_state is not None:
            await _bench_quiesce(False, quiesce_state)
        await _bench_restore_context(ctx_restore)
        if orig_residency is not None:
            _bench_phase(parent_id, "putting the backends and models back as they were")
            # Stop whatever the sweep left running, then put back exactly what it found.
            await _bench_free_gpu(await _bench_residency_snapshot())
            await _bench_restore_residency(orig_residency)
            _save_pending_residency(None)
            _bench_phase(parent_id, None)
    conn = db()
    conn.execute(
        "UPDATE bench_runs SET status='done', finished_ts=?, results_json=? WHERE id=?",
        (time.time(), json.dumps({"children": child_ids, "cells": len(cells)}), parent_id),
    )
    conn.commit()
    conn.close()


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
        def _scalar(key, default):
            # Defence in depth for the bug above: a sweep axis that reaches a child unflattened
            # would otherwise crash it, and a crashed child explains nothing to the user.
            v = cfg.get(key, default)
            return (v[0] if v else default) if isinstance(v, list) else v

        runs = max(1, min(int(_scalar("runs", 5)), 50))
        max_tokens = max(16, min(int(_scalar("max_tokens", 256)), 8192))
        prompt_tokens = max(0, min(int(_scalar("prompt_tokens", 0)), 262144))
        concurrency = max(1, min(int(_scalar("concurrency", 1)), 8))
        randomize = bool(cfg.get("randomize", False))
        exclusive = bool(cfg.get("exclusive", False))
        drain_seconds = max(0.0, min(float(cfg.get("drain_seconds", 5.0)), 30.0))
        model = row["model"]
        # Graded mode replaces the synthetic prompt with a suite of real tasks, one request
        # each, and scores the returned code. runs is then per-task rather than total.
        suite_name = str(cfg.get("suite") or "").strip()
        suite, _sk = _bench_suite_tasks(suite_name) if suite_name else (None, {})
        suite = suite or None
        grade_timeout = max(1.0, min(float(cfg.get("grade_timeout", 10.0)), 60.0))
        warmup = bool(cfg.get("warmup", True))
        total_units = (len(suite) * runs) if suite else runs
        conn.execute(
            "UPDATE bench_runs SET status='running', started_ts=?, progress=0, progress_total=? WHERE id=?",
            (time.time(), total_units, bench_id),
        )
        conn.commit()
        conn.close()
        env = await _bench_env_snapshot()
        if _sk:
            # Portability: a task whose tooling is absent on this machine is dropped from the
            # run and recorded — never scored as zero, which would punish the model for the
            # box. The report's method section reads this back. This must sit AFTER the env
            # snapshot exists: it originally ran ten lines earlier and crashed every graded
            # cell on any machine actually missing a toolchain — the exact machines the
            # feature exists for — while passing everywhere the suite ran complete.
            env["skipped_languages"] = {k: v for k, v in sorted(_sk.items())}
        # Resolve the backend from the model unless the user pinned one explicitly. Asking the
        # user to state both invites contradictions (an Ollama model aimed at vLLM just errors),
        # and the proxy already knows the answer.
        index = await _bench_model_index()
        model_meta = _bench_resolve_model(index, model, str(cfg.get("upstream") or ""))
        if not cfg.get("upstream") and model_meta.get("upstream"):
            cfg["upstream"] = model_meta["upstream"]
            cfg["upstream_inferred"] = True
        env["model_loaded_at_start"] = model_meta.get("loaded")
        env["resolved_upstream"] = cfg.get("upstream")
        # Same contract as skipped_languages, applied to context rather than toolchains: a task
        # whose prompt cannot fit this model's window is DROPPED and recorded, never scored as a
        # zero. Until longctx-v1 no graded task had a prompt worth checking, so a task simply
        # too big for the model would have been sent, front-truncated by the backend, and
        # reported as the model failing to recall something it was never shown.
        if suite:
            _window = model_meta.get("loaded_context") or model_meta.get("max_context")
            if _window:
                _budget = _bench_ctx_budget(int(_window), int(cfg.get("max_tokens") or 256))
                _too_big = {t["id"]: int(t["target_tokens"]) for t in suite
                            if t.get("target_tokens")
                            and int(t["target_tokens"]) > _budget}
                if _too_big:
                    suite = [t for t in suite if t["id"] not in _too_big]
                    env["skipped_context"] = {
                        "window": int(_window),
                        "budget": int(_budget),
                        "tasks": {k: v for k, v in sorted(_too_big.items(), key=lambda kv: kv[1])},
                    }
                    total_units = (len(suite) * runs) if suite else runs
                    _c = db()
                    _c.execute("UPDATE bench_runs SET progress_total=? WHERE id=?",
                               (total_units, bench_id))
                    _c.commit()
                    _c.close()
                if not suite:
                    _c = db()
                    _c.execute("UPDATE bench_runs SET status='failed', error=?, "
                               "finished_ts=? WHERE id=?",
                               (f"every task in this suite needs more context than {model!r} "
                                f"has ({int(_window):,} tokens)", time.time(), bench_id))
                    _c.commit()
                    _c.close()
                    return
        for k in ("quant", "size_mb", "max_context", "loaded_context", "checkpoint",
                  "prefix_caching", "kv_cache_dtype"):
            if model_meta.get(k) is not None:
                env[k] = model_meta[k]
        conn = db()
        conn.execute("UPDATE bench_runs SET env_json=?, config_json=? WHERE id=?",
                     (json.dumps(env), json.dumps(cfg), bench_id))
        conn.commit()
        conn.close()

        def _abort(reason: str):
            conn = db()
            conn.execute(
                "UPDATE bench_runs SET status='failed', error=?, finished_ts=? WHERE id=?",
                (reason, time.time(), bench_id))
            conn.commit()
            conn.close()

        # Everything below this point costs real time and, under `exclusive`, blocks every other
        # proxy client. Refuse here rather than discovering it 54 empty completions later.
        why = _bench_preflight(model, model_meta, str(cfg.get("upstream") or ""), index)
        if not why and str(cfg.get("upstream") or "") == "ollama":
            why = await _bench_ollama_kv_preflight(model, model_meta, cfg)
        if why:
            _abort(why)
            return
        # Free the box for whatever this cell is about to measure — every cell, not only the
        # ones that need a backend started. Wiring this to "am I starting something" was wrong:
        # an Ollama cell starts nothing, so llama.cpp kept its ~90 GB and Ollama could fit only
        # 28 of codellama:70b's 38, spilling the rest. One request took 24 minutes instead of
        # ten seconds, and the number that came out measured memory pressure, not the model.
        this_up = str(cfg.get("upstream") or "")
        resid_snap = None
        try:
            snap_now = await _bench_residency_snapshot()
            if any(e.get("was_running") and e.get("name") != this_up
                   for e in (snap_now.get("backends") or [])):
                _bench_phase(bench_id, f"stopping other backends to free memory for {this_up}")
                # What this cell's model actually needs resident, so the wait has a target
                # rather than a guess. Unknown size falls back to no target, which is the old
                # behaviour and no worse.
                # A model already resident needs no room made for it — its weights are inside
                # the pool its backend allocated at boot. Asking for size_mb of FREE memory on
                # top double-counts, and it stamped every vLLM cell "measured under memory
                # pressure — only 22 GB free; 23 GB wanted" while the 20 GB in question was
                # already loaded and serving requests.
                _want = 0 if model_meta.get("loaded") else (model_meta.get("size_mb") or 0)
                _want = _want * _BENCH_FIT_OVERHEAD if _want else 0
                # A standalone run owns the restore; inside a sweep the suite already recorded
                # the box before its first cell and puts everything back at the end.
                if row["parent_id"] is None:
                    resid_snap = snap_now
                    _save_pending_residency(snap_now)
                freed = await _bench_free_gpu(snap_now, keep=model, spare=this_up,
                                              want_free_mb=_want, bench_id=bench_id)
                env["freed_before_run"] = freed
                if _want and not freed.get("reached_target"):
                    # Not fatal — a model can run partly offloaded — but it is the difference
                    # between a measurement and a measurement of memory pressure, so it has to
                    # appear beside the numbers rather than be inferred from them later.
                    env["memory_warning"] = (
                        f"only {freed.get('free_mb_after', 0) / 1024:.0f} GB free after "
                        f"stopping others; {_want / 1024:.0f} GB wanted")
        except Exception as e:
            print(f"[bench] freeing memory failed: {type(e).__name__}: {e}")

        # Bring the backend up if the run needs a container that is configured but stopped.
        # Before the ready() poll and the window fit, both of which assume something is serving.
        backend_restore = None
        if model_meta.get("startable") and not model_meta.get("loaded"):
            got = await _bench_start_backend(model_meta, persist=row["parent_id"] is None,
                                             bench_id=bench_id)
            if not got["ok"]:
                await _bench_restore_backend(got["restore"])
                _abort(f"could not start {cfg.get('upstream')} for {model!r}: {got['detail']}")
                return
            env["started_backend"] = model_meta.get("container") or cfg.get("upstream")
            # The boot cost, kept separate from the warm-up so the report can show the split
            # and so neither number silently absorbs the other.
            if got.get("backend_start_ms"):
                env["backend_start_ms"] = got["backend_start_ms"]
            # Only a standalone run puts it back. In a sweep the suite started it once for the
            # whole matrix; stopping it here would make every cell pay a full vLLM boot.
            if row["parent_id"] is None:
                backend_restore = got["restore"]
            # Re-resolve: the index entry was the container's configuration, and what the server
            # actually reports once up is the authority for context size and quantisation.
            model_meta = _bench_resolve_model(await _bench_model_index(), model,
                                              str(cfg.get("upstream") or "")) or model_meta

        # Fit the window. Children of a sweep skip this: the suite sized the backend once for
        # the whole matrix, and the metrics snapshot this reads from may not have caught up yet.
        ctx_restore = None
        window = model_meta.get("loaded_context") or model_meta.get("max_context")
        want_ctx = int(cfg.get("server_context") or 0)
        if want_ctx:
            # The window is this cell's variable, so set it exactly — including downwards, which
            # the fit-to-prompt path never does. A parent sweep captured the original before its
            # first cell ran and restores it at the end, so nothing is restored here.
            fit = await _bench_fit_context(str(cfg.get("upstream") or ""), want_ctx,
                                           concurrency, exact=True, bench_id=bench_id)
            if not fit["ok"]:
                _abort(f"could not set {cfg.get('upstream')} to a {want_ctx:,}-token "
                       f"window: {fit['detail']}")
                return
            env["server_context"] = want_ctx
            if row["parent_id"] is None:
                ctx_restore = fit["restore"]
            conn = db()
            conn.execute("UPDATE bench_runs SET env_json=? WHERE id=?",
                         (json.dumps(env), bench_id))
            conn.commit()
            conn.close()
        elif row["parent_id"] is None and window:
            needed = _bench_ctx_needed(prompt_tokens, max_tokens)
            if needed > int(window):
                fit = await _bench_fit_context(str(cfg.get("upstream") or ""), needed,
                                               concurrency, bench_id=bench_id)
                if not fit["ok"]:
                    _abort(f"{prompt_tokens:,} prompt tokens need a {needed:,}-token window; "
                           f"{cfg.get('upstream')} serves {int(window):,} and {fit['detail']}")
                    return
                ctx_restore = fit["restore"]
                env["context_resized_to"] = needed
                env["context_was"] = int(window)
                conn = db()
                conn.execute("UPDATE bench_runs SET env_json=? WHERE id=?",
                             (json.dumps(env), bench_id))
                conn.commit()
                conn.close()
        # Exclusive mode: clear the gate, optionally drain in-flight requests, run, then re-set.
        gate_held = False
        if exclusive:
            _BENCH_TRAFFIC_OK.clear()
            # Safety cap scales with the workload — a 256k-context graded sweep legitimately
            # runs past the old fixed 5 minutes, and expiring mid-run would let competing
            # traffic in and silently poison the second half of the results.
            _BENCH_EXCLUSIVE_DEADLINE = time.time() + max(300.0, 60.0 * total_units)
            gate_held = True
            if drain_seconds > 0:
                await asyncio.sleep(drain_seconds)
        # Spare the model about to run, or it would be evicted and immediately reloaded.
        quiesce_state = (await _bench_quiesce(True, keep=model)
                         if cfg.get("quiesce") else None)
        # Always, not on request. A benchmark that leaves other models resident is not measuring
        # the model, it is measuring whatever the box happened to be holding — and on a unified
        # memory box an idle model competes for the same bandwidth whichever backend is being
        # measured. That is not a preference, so it stopped being a checkbox.
        evicted = []
        if not cfg.get("quiesce"):        # quiesce already does this, and restores it
            _bench_phase(bench_id, "unloading other Ollama models")
            _t_ev = time.perf_counter()
            evicted = await _bench_evict_ollama(keep=model)
            if evicted:
                env["unload_ms"] = round((time.perf_counter() - _t_ev) * 1000)
                env["evicted_before_run"] = evicted
                conn = db()
                conn.execute("UPDATE bench_runs SET env_json=? WHERE id=?",
                             (json.dumps(env), bench_id))
                conn.commit()
                conn.close()
        try:
            base = f"http://127.0.0.1:{PROXY_PORT}"
            client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
            try:
                rows: list[dict] = []
                # Warm-up: Ollama pulls an unloaded model into VRAM on first use, so without
                # this the first measured request carries the load time (tens of seconds for a
                # large model) and the run's min/max are nonsense. The warm-up result is
                # discarded, never scored, and never counted in the summary.
                if warmup:
                    # The load itself. For Ollama this is where a 36 GB model is pulled into
                    # memory, and it was the one long step with no phase: the counter sits at
                    # 0 for minutes because the warm-up is deliberately not counted, so the
                    # run looks wedged exactly when it is doing the most work.
                    _sz = model_meta.get("size_mb")
                    env["free_mb_before_load"] = _free_mem_mb()
                    _bench_phase(bench_id, f"loading {_bench_model_display(model)} into memory"
                                 + (f" ({_sz / 1024:.0f} GB)" if _sz else "")
                                 + " — warm-up, not measured")
                    # In cached mode the warm-up must send the prompt the measured runs will
                    # send, otherwise it primes nothing and the first "cached" request is
                    # actually a cold prefill — which is exactly the confound this axis exists
                    # to expose. In cold mode a short throwaway is enough.
                    warm_prompt = (_bench_build_prompt(prompt_tokens, False, 1)
                                   if not randomize else _bench_build_prompt(0, True, 0))
                    warm_max = max_tokens if not randomize else 16
                    _t_warm = time.perf_counter()
                    warm = await _bench_run_one(client, base, model, warm_max,
                                                warm_prompt, 0, cfg=cfg)
                    # The warm-up IS the cold load for an on-demand backend, and it is already
                    # excluded from every measurement — so its duration is the load time, free.
                    # For a container backend the weights were loaded before this request, by
                    # the start above: add that in, or the column compares an Ollama cold load
                    # against a vLLM warm request and calls vLLM twenty times faster to start.
                    env["warmup_request_ms"] = round((time.perf_counter() - _t_warm) * 1000)
                    env["load_ms"] = (env["warmup_request_ms"]
                                      + int(env.get("backend_start_ms") or 0))
                    env["free_mb_after_load"] = _free_mem_mb()
                    _res = await _bench_resident_mb(model, str(cfg.get("upstream") or ""))
                    if _res is None and env.get("free_mb_before_load") is not None:
                        # Not reported per model: infer from what the machine lost while the
                        # only thing happening was this model loading.
                        _res = max(0, env["free_mb_before_load"] - env["free_mb_after_load"])
                    env["resident_mb"] = _res
                    # Read after the warm-up, so it reflects what is serving rather than what
                    # was requested. Overrides the catalogue value: the catalogue records what
                    # the backend offered, this records what it did.
                    _ctx = await _bench_loaded_context(model, str(cfg.get("upstream") or ""))
                    if _ctx:
                        env["loaded_context"] = _ctx
                    conn = db()
                    # env_json rides the write that already happens here. Without it, load_ms
                    # and resident_mb were captured into a dict that never touched the
                    # database again — and because the report only renders the Load and
                    # Resident columns when data exists, the loss was silent: a design meant
                    # to avoid columns of dashes made missing data look like a layout choice.
                    conn.execute("UPDATE bench_runs SET results_json=?, env_json=? WHERE id=?",
                                 (json.dumps({"rows": [], "warmup": warm}),
                                  json.dumps(env), bench_id))
                    conn.commit()
                    conn.close()
                    _bench_phase(bench_id, None)      # measuring from here; the counter moves
                else:
                    warm = None
                    _bench_phase(bench_id, f"loading {_bench_model_display(model)} into memory "
                                           f"— warm-up is off, so "
                                           f"the first measurement carries it")
                # Each unit is one request. Without a suite that's the synthetic prompt N times;
                # with one it's every task × N repeats, so each task gets its own percentiles
                # and a repeatable score.
                units: list[tuple[str, dict | None]] = []
                if suite:
                    for task in suite:
                        for _rep in range(runs):
                            # A task may vary itself per repeat. longctx does: repeating one
                            # identical prompt measures the prompt cache, because the backend
                            # serves repeats 2..N off a KV prefix it already holds and five
                            # matching answers look like consistency when they are one answer
                            # counted five times. The builder is stored UNCALLED — every unit
                            # of the run is assembled before the first request is sent, and a
                            # long-context ladder materialised up front is ~100 MB of filler
                            # held for the whole run.
                            _pf = getattr(task, "prompt_for", None)
                            units.append((_pf(_rep) if callable(_pf) else task["prompt"], task))
                else:
                    units = [(None, None)] * runs  # prompt built per-seq below (salting needs seq)

                seq_counter = 0
                consec_errors = 0
                if warm and warm.get("error"):
                    # A failed warm-up already counts toward the breaker: when the load
                    # itself died, the measured requests are not going to fare better.
                    consec_errors = 1
                while seq_counter < len(units):
                    wave_size = min(concurrency, len(units) - seq_counter)
                    coros = []
                    wave_tasks = []
                    for offset in range(wave_size):
                        idx = seq_counter + offset
                        seq = idx + 1
                        task_prompt, task = units[idx]
                        if callable(task_prompt):
                            task_prompt = task_prompt()   # built here so only one is resident
                        # A graded task at prompt_tokens > 0 is the same task asked from
                        # inside a long context — the axis that turns any suite into a
                        # long-context test. Episodes are excluded: their length comes from
                        # the transcript they build, and padding the first turn would
                        # measure something else.
                        prompt = (
                            _bench_task_at_depth(task_prompt, prompt_tokens, randomize, seq)
                            if (task_prompt is not None and prompt_tokens
                                and not (task or {}).get("tools"))
                            else task_prompt if task_prompt is not None
                            else _bench_build_prompt(prompt_tokens, randomize, seq))
                        wave_tasks.append(task)
                        if task and task.get("tools"):
                            # An agent task is an episode, not a completion: its runner owns
                            # the tool loop and returns the row already graded.
                            coros.append(_bench_run_agent(client, base, model, task, cfg))
                        else:
                            coros.append(_bench_run_one(client, base, model, max_tokens,
                                                        prompt, seq,
                                                        cfg=cfg, capture_text=bool(task)))
                    results = await asyncio.gather(*coros, return_exceptions=False)
                    # Grade after the wave so scoring never overlaps generation — a busy CPU
                    # during generation would show up as slower decode.
                    for res, task in zip(results, wave_tasks):
                        if task:
                            res["task"] = task["id"]
                            if "grade" in res:
                                pass          # agent episodes arrive graded by their runner
                            elif not res.get("error"):
                                res["grade"] = await _bench_grade(
                                    res.get("text") or "", task, grade_timeout,
                                    reasoning=res.get("reasoning_text") or "",
                                    finish_reason=res.get("finish_reason"))
                                # A failing grade on a response that used its whole token
                                # budget is a different diagnosis from wrong code, and the
                                # report must say which one it is.
                                g = res["grade"]
                                if (isinstance(g, dict)
                                        and (g.get("passed") or 0) < (g.get("total") or 1)
                                        and (res.get("completion_tokens") or 0) >= max_tokens):
                                    g["truncated"] = True
                            # The full response can be megabytes across a suite; the grade is
                            # the durable artifact, so keep only a readable excerpt.
                            if "text" in res:
                                res["text"] = res["text"][:4000]
                    rows.extend(results)
                    seq_counter += wave_size
                    # Circuit breaker: a cell whose backend is dead does not get to burn its
                    # whole request budget discovering that. The devstral OOM loop sent 87
                    # requests into a service the OOM killer was restarting, each one
                    # re-triggering the fatal load; every row said the same thing. A handful
                    # of consecutive failures with no success in between is that signature —
                    # a merely slow or flaky backend produces successes in the mix and never
                    # trips this.
                    for res in results:
                        consec_errors = consec_errors + 1 if res.get("error") else 0
                    if consec_errors >= max(6, 2 * concurrency):
                        raise RuntimeError(
                            f"aborted after {consec_errors} consecutive failed requests "
                            f"({seq_counter}/{len(units)} sent) — the backend is failing, "
                            f"not slow — last: {str(rows[-1].get('error'))[:200]}")
                    # Persist progress incrementally so the UI can poll.
                    conn = db()
                    conn.execute(
                        "UPDATE bench_runs SET progress=?, results_json=? WHERE id=?",
                        (seq_counter, json.dumps({"rows": rows}), bench_id),
                    )
                    conn.commit()
                    conn.close()
                summary = _bench_summarize(rows)
                if suite:
                    summary["quality"] = _bench_quality_summary(rows, suite)
                if warm:
                    # Surfaced, not hidden: if the warm-up took 40s and the measured runs took
                    # 2s, that gap IS the model-load cost and is worth seeing.
                    summary["warmup_ms"] = warm.get("total_ms")
                    summary["warmup_ttft_ms"] = warm.get("ttft_ms")
                    summary["warmup_error"] = warm.get("error")
                final = {"rows": rows, "summary": summary, "config_used": cfg, "env": env,
                         "warmup": warm}
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
            if quiesce_state is not None:
                await _bench_quiesce(False, quiesce_state)
            # Reload what was evicted. Opt-in, this was the user's explicit choice to clear the
            # GPU; unconditional, a three-minute bench would silently leave their daily driver
            # cold with nothing saying why.
            # Standalone runs put them back themselves. Inside a sweep the next cell is about
            # to evict them again, so reloading here would be minutes of thrash per cell —
            # the suite restores the whole set once, at the end.
            if evicted and row["parent_id"] is None:
                _bench_phase(bench_id, "reloading the models it unloaded")
                await _bench_reload_ollama(evicted)
            if backend_restore or ctx_restore or resid_snap:
                _bench_phase(bench_id, "putting the backends back as they were")
            await _bench_restore_backend(backend_restore)
            restored = await _bench_restore_context(ctx_restore)
            if resid_snap is not None:
                try:
                    await _bench_restore_residency(resid_snap)
                    _save_pending_residency(None)
                except Exception as e:
                    print(f"[bench] restoring residency failed: {type(e).__name__}: {e}")
            if restored is not None:
                conn = db()
                row2 = conn.execute("SELECT env_json FROM bench_runs WHERE id=?",
                                    (bench_id,)).fetchone()
                try:
                    e = json.loads(row2["env_json"]) if row2 and row2["env_json"] else {}
                except (json.JSONDecodeError, TypeError):
                    e = {}
                e["context_restored"] = restored
                conn.execute("UPDATE bench_runs SET env_json=? WHERE id=?",
                             (json.dumps(e), bench_id))
                conn.commit()
                conn.close()
            if gate_held:
                _BENCH_TRAFFIC_OK.set()
                _BENCH_EXCLUSIVE_DEADLINE = 0.0
            # A finished row must not keep describing a step it is no longer doing.
            _bench_phase(bench_id, None)


# -------- Task queue (one-shots + cron/interval recurring) --------
#
# Lets the user queue prompts and run them in the background. Recurring rows fire on a cron
# expression or "every Nm/Nh/Nd" interval and spawn a one-shot child each time. Chat tasks
# hit OLLAMA_URL directly with non-streaming completions.

# Chat tasks run up to 3 in parallel. Semaphore lives module-global so the worker tick can
# fire-and-forget.
_TASK_CHAT_SEM = asyncio.Semaphore(3)


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
    sem = _TASK_CHAT_SEM
    async with sem:
        # Re-check that we weren't cancelled while waiting on the semaphore.
        conn = db()
        cur_row = conn.execute("SELECT status FROM proxy_tasks WHERE id=?", (task_id,)).fetchone()
        conn.close()
        if not cur_row or cur_row["status"] != "running":
            return
        try:
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


_POOL_WARNED = {"at": 0.0}
_TLS_WARNED = {"at": 0.0}
_TLS_CACHE: dict = {"key": None, "value": None}
# Certificates are renewed on a schedule nobody watches. Two weeks is enough warning to act
# without being so early the message becomes background noise.
TLS_EXPIRY_WARN_DAYS = 14


async def _wait_for_http_start(http_server, http_task, timeout_s: float) -> None:
    """Block until the HTTP listener has finished starting, or refuse to continue.

    The HTTPS listener runs with lifespan="off" and borrows the HTTP server's state, so it
    must not open until that state exists. The original loop polled for 5 seconds and then
    opened the listener regardless — but init_db runs migrations and backfills, which on a
    multi-gigabyte database takes far longer than that, and every HTTPS request arriving in
    the gap fails on an app.state.client that has not been created yet.

    A slow start is a reason to keep waiting, not a reason to serve half-built state, so
    exhausting the deadline raises instead of proceeding.
    """
    deadline = time.monotonic() + timeout_s
    while not getattr(http_server, "started", False):
        if http_task.done():
            await http_task                  # surface whatever killed it, if anything
            raise RuntimeError("HTTP listener exited during startup; HTTPS was not started")
        if time.monotonic() > deadline:
            http_server.should_exit = True
            raise RuntimeError(
                f"HTTP listener did not finish starting within {timeout_s}s; refusing to open "
                f"HTTPS against an uninitialised app (raise PROXY_STARTUP_TIMEOUT_S if startup "
                f"is legitimately slow)")
        await asyncio.sleep(0.05)


def _cert_not_after(path: str) -> float | None:
    """Expiry of a PEM certificate as an epoch, or None if it can't be read.

    `_test_decode_cert` is private, but it is the only stdlib way to read a certificate from
    disk without opening a connection, and this package deliberately carries no crypto
    dependency. `cert_time_to_seconds` beside it is public. Both are wrapped: a certificate we
    cannot parse must cost the metric, never the health endpoint.
    """
    try:
        info = ssl._ssl._test_decode_cert(path)
        return ssl.cert_time_to_seconds((info or {})["notAfter"])
    except Exception:
        return None


def _tls_status() -> dict:
    """Expiry of the configured HTTPS certificate.

    Expiry is the same shape of failure as pool exhaustion: scheduled, silent, and total the
    moment it lands, with nothing reporting it beforehand. Health is where it belongs — the
    file is stat'd per call and only re-parsed when it changes, so a polling UI costs nothing.
    """
    cert = os.environ.get("PROXY_SSL_CERT", "").strip()
    try:
        port = int(os.environ.get("PROXY_HTTPS_PORT", "0") or "0")
    except ValueError:
        port = 0
    if not cert or port <= 0:
        return {"enabled": False}
    out = {"enabled": True, "port": port, "cert": cert, "exists": False}
    try:
        mtime = Path(cert).stat().st_mtime
    except OSError:
        # Configured but unreadable: HTTPS did not start, and silence would read as healthy.
        return out
    out["exists"] = True
    key = (cert, mtime)
    if _TLS_CACHE.get("key") != key:
        _TLS_CACHE["key"] = key
        _TLS_CACHE["value"] = _cert_not_after(cert)
    not_after = _TLS_CACHE.get("value")
    if not_after is None:
        return out
    days = (not_after - time.time()) / 86400.0
    out["not_after"] = not_after
    out["days_remaining"] = round(days, 2)
    out["expired"] = days <= 0
    out["expiring_soon"] = days <= TLS_EXPIRY_WARN_DAYS
    return out


def _tls_watch() -> None:
    """Warn while the certificate is still valid, rather than after it isn't."""
    s = _tls_status()
    if not s.get("enabled") or not s.get("expiring_soon"):
        return
    if time.time() - _TLS_WARNED["at"] < 3600:
        return          # hourly: this is a slow-moving condition, not an incident
    _TLS_WARNED["at"] = time.time()
    days = s.get("days_remaining")
    try:
        if s.get("expired"):
            print(f"[tls] certificate {s['cert']} EXPIRED {abs(days):.1f} days ago — "
                  f"HTTPS clients are failing")
        else:
            print(f"[tls] certificate {s['cert']} expires in {days:.1f} days")
    except Exception:
        pass


def _pool_stats() -> dict:
    """Occupancy of the shared upstream connection pool.

    httpx exposes no public accessor for this, so it reaches through the transport into
    httpcore. Wrapped whole: an observability hook must never be the thing that breaks
    proxying, so a version bump that moves these internals costs the metric, not the box.
    """
    try:
        pool = app.state.client._transport._pool
        conns = list(pool.connections)
        limit = pool._max_connections
    except Exception:
        return {}
    idle = 0
    for c in conns:
        try:
            idle += 1 if c.is_idle() else 0
        except Exception:
            pass
    return {"connections": len(conns), "idle": idle,
            "active": len(conns) - idle, "limit": limit}


def _pool_watch() -> None:
    """Say something while the pool is filling, rather than after it is full.

    Exhaustion has no natural symptom — requests just stop finishing — so the only warning
    anyone gets is the one printed here."""
    s = _pool_stats()
    limit = s.get("limit") or 0
    if not limit or s.get("active", 0) < limit * 0.8:
        return
    if time.time() - _POOL_WARNED["at"] < 300:
        return          # once per 5 min: a full pool would otherwise flood the journal
    _POOL_WARNED["at"] = time.time()
    try:
        print(f"[pool] {s['active']}/{limit} upstream connections checked out "
              f"({s['idle']} idle, {len(_INFLIGHT_REQUESTS)} requests in flight) — "
              f"at the cap, proxied requests will queue")
    except Exception:
        pass


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
            # Drain sockets orphaned by _save_finish first — these are already-finished
            # requests, so there is nothing to wait for and every one held is a pool slot gone.
            while _ORPHANED_RESPONSES:
                _resp = _ORPHANED_RESPONSES.pop()
                try:
                    await _resp.aclose()
                except Exception:
                    pass
            _pool_watch()
            _tls_watch()
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


# High-precision system-prompt fingerprints for agentic clients that ride on a generic SDK
# User-Agent (openai-sdk / anthropic-sdk / unknown). Each marker is a distinctive phrase from
# the tool's system prompt, kept deliberately specific so a request whose prompt merely
# *mentions* a tool isn't misattributed. Only consulted when header/UA detection is generic.
_SYS_PROMPT_FINGERPRINTS = (
    # Hermes Agent (Nous Research) often calls through the OpenAI Python SDK, which overwrites its
    # UA with "OpenAI/Python" — so the same app splits between the 'hermes' (real UA) and
    # 'openai-sdk' buckets. The system prompt is stable and unifies them. Its command-approval
    # sub-agent uses a distinct "security reviewer" prompt (same host, OpenAI SDK) → hermes-safety.
    ("you are hermes agent", "hermes"),
    ("security reviewer for an ai coding agent", "hermes-safety"),
    # Hindsight, Hermes's memory plugin. It runs its own pipeline through the stock
    # AsyncOpenAI SDK with no x-client-name, so it landed in the generic 'openai-sdk'
    # bucket and was invisible as a caller — while being the second-busiest thing on the
    # box. Two stages, labelled apart because they cost very differently: extraction runs
    # per batch of facts, consolidation carries the existing observations in its prompt
    # (~7k tokens against ~3.5k) and is the one that gets expensive as memory grows.
    ("extract significant facts from text", "hindsight-extract"),
    ("you are a memory consolidation system", "hindsight-consolidate"),
    # Not memory work: whatever generates conversation titles, also on the bare SDK.
    ("generate a short, descriptive title", "title-generator"),
    ("you are pair programming with a user", "cursor"),
    ("you are cline", "cline"),
    ("you are roo,", "roo-code"),
    ("you are kilo code", "kilo-code"),
    ("you are cascade", "windsurf"),
    ("codeium engineering team", "windsurf"),
    ("you are bolt", "bolt.new"),
    ("search/replace block", "aider"),
    ("act as an expert software developer", "aider"),
    ("you are openhands", "openhands"),
    ("ai agent called goose", "goose"),
    ("running in the codex cli", "codex-cli"),
    ("you are amazon q", "amazon-q"),
    ("you are github copilot", "github-copilot"),
)


# Headers whose value is a credential. Stored redacted, never verbatim.
_CREDENTIAL_HEADERS = ("authorization", "x-api-key", "api-key", "proxy-authorization",
                       "cookie", "x-auth-token")
_CRED_REDACTED = "[redacted]"


def _credential_fingerprint(headers: dict | None) -> str | None:
    """A stable, non-reversible id for the credential a caller presented.

    Callers here share an IP and often a client label — everything hermes runs arrives from
    one host through the same SDK — so the key is the one thing that reliably tells two
    callers apart. Storing the key itself to get that would be trading a real secret for a
    convenience: the database is copied, archived and handed around, and a bearer token in
    it is a bearer token in all of those. A hash plus the last four characters identifies
    the caller just as well and is worthless to whoever finds it.

    Placeholder credentials that local backends accept ('no-key-required' and friends) are
    not fingerprinted: they identify nobody, and a column full of one shared value invites
    the reader to think it means something.
    """
    if not isinstance(headers, dict):
        return None
    for k, v in headers.items():
        if k.lower() not in _CREDENTIAL_HEADERS:
            continue
        raw = str(v or "").strip()
        if raw.lower().startswith("bearer "):
            raw = raw[7:].strip()
        if len(raw) < 8:
            return None
        if re.fullmatch(r"(?i)[-_a-z0-9]*(not|no)[-_]?(key|auth|required|needed)[-_a-z0-9]*",
                        raw) or raw.lower() in ("none", "null", "dummy", "placeholder"):
            return None
        digest = hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:12]
        return f"{digest}…{raw[-4:]}"
    return None


def _redact_credential_headers(headers: dict) -> dict:
    """Headers as stored: same shape, credentials replaced. The fingerprint column carries
    whatever identification value the real value had."""
    out = {}
    for k, v in (headers or {}).items():
        out[k] = _CRED_REDACTED if k.lower() in _CREDENTIAL_HEADERS else v
    return out


def _fingerprint_client_from_system(sys_text: str) -> str | None:
    """Best-effort agentic-client identification from the system prompt. High-precision markers
    only; returns None when nothing distinctive matches so the caller keeps its generic verdict."""
    if not sys_text:
        return None
    t = sys_text.lower()
    for marker, label in _SYS_PROMPT_FINGERPRINTS:
        if marker in t:
            return label
    return None


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

    # No confident header/UA match. Before settling for a generic SDK label (openai-sdk /
    # anthropic-sdk / unknown), try to fingerprint the agentic client from its system prompt —
    # Cursor, Cline, Aider, Windsurf, etc. all ride on a stock SDK UA but embed a telltale
    # system prompt. A fingerprint hit is strictly more specific than the SDK bucket below.
    fp = _fingerprint_client_from_system(sys_text)
    if fp:
        return fp

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


def _body_system_text(body: dict | None) -> str:
    """The system prompt as one string, whatever shape the client sent it in."""
    if not isinstance(body, dict):
        return ""
    sys_field = body.get("system")
    if isinstance(sys_field, str):
        return sys_field
    if isinstance(sys_field, list):
        return " ".join(p.get("text", "") for p in sys_field
                        if isinstance(p, dict) and isinstance(p.get("text"), str))
    for m in (body.get("messages") or []):
        if isinstance(m, dict) and m.get("role") == "system":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return " ".join(p.get("text", "") for p in c
                                if isinstance(p, dict) and isinstance(p.get("text"), str))
            break
    return ""


def _detect_surface(headers: dict | None, body: dict | None) -> str | None:
    """Which surface an agentic client was driven from: 'discord', 'cron', or 'cli'.

    One client_app can front several execution contexts (hermes runs as a Discord bot, a
    scheduled cron job, AND an interactive local session — same IP, same UA, same
    x-client-name), so the client label alone can't answer "where did this session come
    from". The context leaks into the request anyway: cron prompts announce there is no
    user present, Discord sessions carry a guild/channel context block and discord tools,
    local sessions offer file/exec tools without any of that. NULL when nothing matches —
    an honest unknown beats a guessed label."""
    h = {k.lower(): str(v) for k, v in (headers or {}).items()} if isinstance(headers, dict) else {}
    v = (h.get("x-client-surface") or "").strip()
    if v:  # caller-supplied wins, same contract as x-client-name
        slug = re.sub(r"[^a-z0-9._-]+", "-", v.lower())[:24].strip("-")
        if slug:
            return slug
    sys_text = _body_system_text(body)
    tools = set()
    if isinstance(body, dict):
        for t in (body.get("tools") or []):
            if isinstance(t, dict):
                name = (t.get("function") or {}).get("name") or t.get("name")
                if isinstance(name, str):
                    tools.add(name)
    if not sys_text and not tools:
        return None
    if re.search(r"running as a scheduled (cron job|task)", sys_text, re.I):
        return "cron"
    # A guild/channel context block means a live Discord session; the discord tools alone
    # also imply one (DMs get the tools without the guild block).
    if re.search(r"^\s*-?\s*Guild:\s*`?\d", sys_text, re.M) or \
            not tools.isdisjoint({"discord", "discord_admin"}):
        return "discord"
    # Local exec toolset with no discord anywhere → an interactive local/CLI session.
    if {"execute_code", "read_file"} <= tools or {"read_file", "patch"} <= tools:
        return "cli"
    return None


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


def _upstream_error_message(status: int | None, body_text: str | None) -> str | None:
    """The upstream's own account of a 5xx, lifted out of its response body.

    When vLLM's engine died it returned a perfectly clear EngineDeadError in the body and the
    row recorded status 500 with a blank error column, so the detail view had nothing to show.
    Whatever the upstream took the trouble to say is the best explanation available.
    """
    if not status or status < 500 or not body_text:
        return None
    try:
        j = json.loads(body_text)
    except (json.JSONDecodeError, TypeError):
        # Not JSON: the first non-empty line beats nothing.
        line = next((l.strip() for l in body_text.splitlines() if l.strip()), "")
        return f"upstream returned {status}: {line[:300]}" if line else None
    msg = None
    if isinstance(j, dict):
        err = j.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("type")
        elif isinstance(err, str):
            msg = err
        msg = msg or j.get("detail") or j.get("message")
        if isinstance(msg, (dict, list)):
            msg = json.dumps(msg)
    if not msg:
        return None
    return f"upstream returned {status}: {str(msg)[:400]}"


def _explain_upstream_error(exc, upstream: str | None, url: str | None,
                            elapsed_ms: float | None = None, got_bytes: bool = False) -> str:
    """Turn a transport exception into something worth reading in the request detail.

    httpx's own text is frequently useless — a connection dropped mid-response raises
    ReadError('') with an empty message, which is what the log said for every request lost when
    vLLM's engine died. The exception type already carries the diagnosis; this spells it out,
    says which upstream and how far in, and keeps the raw repr on the end for forensics.
    """
    name = type(exc).__name__
    where = f"{upstream or 'the upstream'}"
    if url:
        try:                                   # host:port is the useful part, not the full path
            p = url.split("//", 1)[-1].split("/", 1)[0]
            where = f"{upstream or 'the upstream'} at {p}"
        except Exception:
            pass
    when = f" after {elapsed_ms:.0f} ms" if elapsed_ms is not None else ""
    phase = ("mid-response, after part of the answer had arrived" if got_bytes
             else "before any response bytes arrived")

    if name == "ConnectError":
        what = (f"Could not open a connection to {where}{when}. Nothing is listening — the "
                f"upstream is down, restarting, or on a different port.")
    elif name == "ConnectTimeout":
        what = (f"Timed out opening a connection to {where}{when} — it is reachable but not "
                f"completing the handshake.")
    elif name == "ReadError":
        what = (f"{where} accepted the request then closed the connection{when}, {phase}. "
                f"That is what a crashed or restarted upstream looks like from here — its own "
                f"log will say why.")
    elif name == "RemoteProtocolError":
        what = (f"{where} ended the response early{when}, {phase} — the connection closed in "
                f"the middle of a message rather than at a boundary.")
    elif name == "ReadTimeout":
        what = (f"{where} accepted the request but sent nothing back{when}. Reachable but not "
                f"answering — usually a model still loading, or a queue that is not draining.")
    elif name == "PoolTimeout":
        what = (f"No connection slot was free for {where}{when} — the proxy's own pool is "
                f"saturated, not the upstream.")
    elif name in ("WriteError", "WriteTimeout"):
        what = (f"{where} closed the connection while the request was still being sent{when} — "
                f"often an oversized body, or the upstream dying at the wrong moment.")
    else:
        what = f"{where} failed{when}, {phase}."
    detail = str(exc).strip()
    return f"{what} [{name}{': ' + detail if detail else ''}]"


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


# Every dialect a client has used here to say "think" or "don't", in the order they override
# each other: a body carrying both reasoning_effort and an explicit enable_thinking meant the
# latter, because it is the one the template actually reads.
_THINK_LEVELS = {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}


def _thinking_intent(body) -> tuple[str | None, str | None, str | None]:
    """What the client asked for: (effort, mode, source).

    effort is the level when one was named ('low'/'medium'/'high'), mode is 'on' or 'off', and
    source names the field that settled it — so a row can say *why* it was recorded that way
    rather than leaving you to guess which of six dialects the client speaks. All three are
    None when the request said nothing, which is the common case: most clients never mention
    thinking and take whatever the model defaults to.
    """
    if not isinstance(body, dict):
        return None, None, None
    effort = mode = src = None

    eff = body.get("reasoning_effort")
    if eff is None and isinstance(body.get("reasoning"), dict):
        eff = body["reasoning"].get("effort")
    if eff is not None:
        e = str(eff).strip().lower()
        src = "reasoning_effort"
        if e in ("none", "off"):
            mode = "off"
        elif e in _THINK_LEVELS:
            effort, mode = _THINK_LEVELS[e], "on"
        else:
            mode = "on"

    ctk = body.get("chat_template_kwargs")
    et = body.get("enable_thinking")
    if et is None and isinstance(ctk, dict):
        et = ctk.get("enable_thinking")
    if isinstance(et, bool):
        mode, src = ("on" if et else "off"), "enable_thinking"

    # Ollama's native field. Newer versions take a level here as well as a bool.
    think = body.get("think")
    if isinstance(think, bool):
        mode, src = ("on" if think else "off"), "think"
    elif isinstance(think, str) and think.strip().lower() in _THINK_LEVELS:
        effort, mode, src = _THINK_LEVELS[think.strip().lower()], "on", "think"

    t = body.get("thinking")
    if isinstance(t, dict):
        if t.get("type") == "disabled":
            mode, src = "off", "thinking"
        elif t.get("type") == "enabled":
            b = t.get("budget_tokens") or 0
            effort = "high" if b >= 8000 else "medium" if b >= 2000 else "low" if b > 0 else None
            mode, src = "on", "thinking_budget"

    # Qwen's in-band switches, which live in the prompt rather than the body.
    last_user = ""
    msgs = body.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    last_user = c
                elif isinstance(c, list):
                    last_user = " ".join(str(p.get("text") or "") for p in c if isinstance(p, dict))
                break
    if not last_user and isinstance(body.get("prompt"), str):
        last_user = body["prompt"]
    if last_user:
        if re.search(r"/no_think\b", last_user):
            mode, src = "off", "/no_think"
        elif re.search(r"/think\b", last_user):
            mode, src = "on", "/think"

    return effort, mode, src


def _save_pending(req_id: str, request: Request, full_path: str, upstream_url: str, body_text: str, body_json, model, is_stream: bool, upstream: str | None = None):
    conv_id = _conversation_id(body_json)
    turn = _turn_index(body_json)
    headers_dict = dict(request.headers)
    client_app = _detect_client_app(headers_dict, body_json if isinstance(body_json, dict) else None)
    surface = _detect_surface(headers_dict, body_json if isinstance(body_json, dict) else None)
    # Fingerprint before redacting, store only the fingerprint. The forwarded request is
    # untouched — headers_dict is our copy for the log, not the one going upstream.
    api_key_fp = _credential_fingerprint(headers_dict)
    headers_dict = _redact_credential_headers(headers_dict)
    # Chars-based prompt-token estimate, persisted so the requests list can derive a cache
    # verdict (evaluated vs estimated) without re-parsing the body on every load.
    est = _estimate_prompt_tokens(body_json, 3.5) if isinstance(body_json, dict) else 0
    # has_images flags standard vision parts (image_url / Anthropic image) AND base64 blobs
    # embedded in tool-output text (e.g. screenshot_png_b64). The router keys off _body_has_images()
    # directly (standard parts only), so flagging an embedded blob here surfaces it in the dashboard
    # viewer WITHOUT routing a text request to the vision model (which couldn't consume it anyway).
    # An embedded blob only counts once it decodes to something with an image header. A bare
    # substring test (the previous approach) flagged any body merely containing the 4-character
    # sequence "/9j/", so text-only agents were reported as vision requests en masse and every
    # such badge led to an image that couldn't be rendered.
    _embedded_imgs = any(_decode_b64_image(b64) for _mt, b64 in _embedded_b64_images(body_text or ""))
    # Pull inline base64 images into their own column at full size, then store the text body
    # with them stripped — so the body cap can't truncate (and corrupt the JSON of) a ~700KB
    # screenshot. The image bytes stay reconstructable from images_data.
    images_data = None
    store_body_text = body_text
    imgs: list = []
    if isinstance(body_json, dict):
        # Standard vision parts get the same validation: a truncated data URL is not an image.
        imgs = [{"media_type": mt, "data": payload}
                for (_i, mt, _k, payload) in _iter_request_images(body_json)
                if _k == "data" and _decode_b64_image(payload)]
        if imgs:
            images_data = json.dumps(imgs)
            body_copy = json.loads(json.dumps(body_json))   # don't mutate the body we forward
            _strip_image_data(body_copy)
            store_body_text = json.dumps(body_copy)
    # Flag only what a viewer could actually open: a decodable inline image, a decodable embedded
    # blob, or an image referenced by URL (nothing to decode, but it is genuinely an image part).
    _url_imgs = any(k == "url" for (_i, _mt, k, _p) in _iter_request_images(body_json)) \
        if isinstance(body_json, dict) else False
    has_imgs = 1 if (imgs or _embedded_imgs or _url_imgs) else None
    _effort, _think_mode, _think_src = _thinking_intent(body_json)
    conn = db()
    conn.execute(
        """INSERT INTO requests (id, ts, method, path, upstream_url, request_headers, model, is_stream, client_ip, conversation_id, turn_index, client_app, surface, api_key_fp, upstream, est_prompt_tokens, has_images, reasoning_effort, thinking, thinking_src)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            surface,
            api_key_fp,
            upstream,
            est or None,
            has_imgs,
            _effort,
            _think_mode,
            _think_src,
        ),
    )
    _blobs_upsert(conn, req_id,
                  request_body=_truncate_for_store(store_body_text),
                  images_data=images_data)
    conn.commit()
    conn.close()
    # Seed live state with the estimate so the UI shows something immediately, even before
    # the upstream confirms the real input_tokens.
    #
    # `stream` is recorded because a non-streaming request produces NO live text, ever: the
    # body arrives in one aread() and streamer() — the only caller of _live_update_from_chunk —
    # never runs. Without this flag the live pane cannot tell "streaming, nothing yet" from
    # "nothing is coming", so it claimed to be live and rendered blank for the full 30s of
    # every hindsight call. Absence of data and absence of streaming should not look alike.
    _LIVE_STREAMS[req_id] = {"prompt": None, "completion": None, "est_prompt": est or None,
                             "stream": bool(is_stream), "buf": ""}


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
    # Extract the tool names now, while the response is in memory. Doing it later means going
    # back to disk for the whole body, which is what made the stats endpoint slow.
    try:
        _tools = _extract_tool_calls(body_text, stream_text)
    except Exception:
        _tools = []
    conn = db()
    conn.execute(
        """UPDATE requests
           SET status=?, response_headers=?, duration_ms=?, error=?,
               prompt_tokens=?, completion_tokens=?, total_tokens=?, ttft_ms=?, tool_calls=?
           WHERE id=?""",
        (
            status,
            json.dumps(resp_headers) if resp_headers else None,
            elapsed_ms,
            error,
            pt, ct, tt, ttft_ms,
            json.dumps(_tools) if _tools else None,
            req_id,
        ),
    )
    _blobs_upsert(conn, req_id,
                  response_body=_truncate_for_store(body_text),
                  stream_chunks=_truncate_for_store(stream_text))
    conn.commit()
    conn.close()
    _LIVE_STREAMS.pop(req_id, None)
    _gone = _INFLIGHT_REQUESTS.pop(req_id, None)
    # Finalizing the row must not strand the socket: whichever path got here, if the response
    # is still open this is the last moment anything holds a reference to it.
    if _gone is not None:
        _resp = _gone.get("upstream_resp")
        if _resp is not None and not getattr(_resp, "is_closed", True):
            _ORPHANED_RESPONSES.append(_resp)


@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy(full_path: str, request: Request):
    # Defensive: never proxy our own UI/API namespace.
    if full_path.startswith("__proxy"):
        return JSONResponse({"error": "not found"}, status_code=404)

    # Panic kill-switch: refuse all upstream traffic until disabled. Proxy stays up so the
    # UI / phone PWA / control endpoints remain reachable to flip it off again. The benchmark
    # client is spared so a bench can run against a quiesced GPU (no competing traffic skewing
    # the timings) simply by flipping panic mode on for the duration of the run.
    if _PANIC_MODE and (request.headers.get("x-client-name") or "").lower() != "ai-proxy-bench":
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
    # x-proxy-no-router pins the request to the model it names. Benchmarks need this: with a
    # routing rule in play, asking for "qwen" can silently measure a different model on a
    # different backend, and the stored result would attribute the numbers to "qwen".
    _skip_router = (request.headers.get("x-proxy-no-router") or "").strip().lower() in ("1", "true", "yes")
    # Ollama's metadata endpoints are not inference. Routing POST /api/show to vLLM because the
    # model name matched a rule sent a native Ollama call to an OpenAI-compatible server, which
    # 404s every time — 21 of them in one second from vscode-copilot. Rewriting the model there
    # is wrong too: the client asked about a specific model, not whichever one the rule prefers.
    if _is_ollama_native_only(full_path):
        _skip_router = True
    rewrite = (evaluate_router(body_json, router_ctx)
               if isinstance(body_json, dict) and not _skip_router else None)

    # Per-rule upstream override: a model_router rule may carry {"upstream": "lmstudio"} to
    # send that (rewritten) model to a different backend than the path-based default. This is
    # how e.g. qwen traffic reaches LM Studio's OpenAI endpoint *through* the proxy so it's
    # inspected/logged like everything else. Only OpenAI-compat backends are valid targets —
    # both Ollama and LM Studio speak the same /v1 shape, so no body translation is needed.
    # "anthropic" is deliberately NOT a target: forwarding an OpenAI-shape body there without
    # the protocol bridge (which only goes anthropic→ollama) would produce 400s.
    _UPSTREAM_BASES = {n: p.base_url for n, p in PROVIDERS.items()}
    # x-proxy-upstream forces the backend directly, without needing a routing rule. Same
    # motivation as x-proxy-no-router: benchmark the engine you meant to benchmark.
    # Kept OUT of `rewrite`: that dict is the model_router verdict and downstream audit code
    # requires its from/to/via keys. Synthesizing a partial one here to carry the upstream
    # made every pinned request 500 on the audit path.
    _pin_upstream = (request.headers.get("x-proxy-upstream") or "").strip().lower()
    if _pin_upstream not in _UPSTREAM_BASES:
        _pin_upstream = ""
    _want_upstream = _pin_upstream or ((rewrite or {}).get("upstream") or "")
    # Even an explicit x-proxy-upstream cannot make another backend answer /api/show.
    if _is_ollama_native_only(full_path):
        _want_upstream = ""
    if _want_upstream and upstream_label != "anthropic":
        _new_label = str(_want_upstream).lower()
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

    # Auto-load (opt-in): a request for a model on a stopped backend or the other vLLM twin
    # starts what it needs and waits, Ollama-style, evicting only when it doesn't fit.
    if (isinstance(body_json, dict) and body_json.get("model")
            and upstream_label in ("vllm", "llamacpp")
            and full_path.startswith(("v1/chat", "v1/completions"))):
        _al_err = await _ensure_model_served(str(body_json["model"]), upstream_label)
        if _al_err:
            return JSONResponse({"error": _al_err["error"], "auto_load": True},
                                status_code=int(_al_err.get("status", 503)))

    # Per-model quirk thinking control (see DEFAULT_MODEL_QUIRKS / model_quirks.json). Some models
    # need thinking managed via chat_template_kwargs.enable_thinking because they ignore the OpenAI
    # reasoning_effort knob (e.g. Ornith). The quirk says whether to force thinking off, or default
    # it off but let reasoning_effort high/medium opt in. A client that set enable_thinking itself
    # is left untouched.
    ornith_think = None
    _qmodel = str(body_json.get("model") or "") if isinstance(body_json, dict) else ""
    _quirk = _model_quirk(_qmodel) or {}
    _tmode = _quirk.get("thinking")
    if _tmode in ("force_off", "default_off_optin"):
        ctk = body_json.get("chat_template_kwargs")
        if not isinstance(ctk, dict):
            ctk = {}
        if "enable_thinking" not in ctk:
            want = False
            if _tmode == "default_off_optin":
                want = str(body_json.get("reasoning_effort") or "").lower() in ("high", "medium")
            ctk["enable_thinking"] = want
            body_json["chat_template_kwargs"] = ctk
            ornith_think = want
            # The row already holds what the client asked for; this is the only place that
            # knows what it actually got. The stored body is the client's original, so
            # without this the audit would show "nothing requested" for a model the proxy
            # had just silently switched off.
            try:
                _conn = db()
                _conn.execute(
                    "UPDATE requests SET thinking=?, thinking_src=? WHERE id=?",
                    ("on" if want else "off", f"quirk:{_tmode}", req_id),
                )
                _conn.commit()
                _conn.close()
            except Exception:
                pass
    # system_nudge: some reasoning-trained models (Ornith) narrate their reasoning in the *content*
    # even with the <think> block disabled. A short system-prompt instruction is the only lever
    # that curbs that (the thinking toggle can't — it's not in a think block). Appended, not replaced.
    # x-proxy-no-nudge suppresses it: the nudge is an extra system message, which is a real
    # confound when the point of the request is to measure the model rather than use it.
    _skip_nudge = (request.headers.get("x-proxy-no-nudge") or "").strip().lower() in ("1", "true", "yes")
    _nudge = _quirk.get("system_nudge")
    if _nudge and isinstance(body_json, dict) and not ornith_think and not _skip_nudge:
        _inject_system_reminder(body_json, str(_nudge))

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
    # num_ctx is the one option the OpenAI-compat endpoint drops on the floor, so it is carried
    # by a derived model instead. Runs right after the injection whose promise it keeps, and
    # before the overflow guard so that guard sees the window the request will really get.
    ctx_swap = None
    if isinstance(body_json, dict) and upstream_label == "ollama":
        try:
            ctx_swap = await apply_ollama_ctx_model(body_json, router_ctx)
        except Exception as e:                       # never fail a request over a tuning knob
            print(f"[ollama_options] ctx model swap failed: {type(e).__name__}: {e}")
        if ctx_swap:
            options_inject = dict(options_inject or {})
            options_inject.setdefault("applied", {})["num_ctx"] = ctx_swap["num_ctx"]
            options_inject.setdefault("sources", []).append(
                f"ctx_model:{ctx_swap['from']}->{ctx_swap['to']}")
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
    # context_compressor: squeeze bulky tool outputs in the history. In shadow mode it only
    # tallies potential savings; in live mode it mutates body_json before re-serialize.
    compressed = (evaluate_context_compressor(body_json, {**router_ctx, "client_app": _ti_ctx.get("client_app")})
                  if isinstance(body_json, dict) else None)
    if compressed and compressed.get("action") == "compress":
        _save_gate(req_id, {
            "verdict": "rewrite", "rule": "context_compressor",
            "reason": f"compressed {compressed['n']} tool output(s): saved {compressed['saved_chars']} chars ({compressed['pct']}%)",
            "details": compressed,
        })
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
        or (compressed and compressed.get("action") == "compress")
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
    # Loop-detector soft-unblock: the model previously looped but the user has intervened, so we
    # let the request through AND inject a note (into the system prompt) explaining the earlier
    # block, so the model doesn't immediately repeat the same call. Re-encode the outgoing body
    # (already serialized above) after the injection.
    if (gate.get("rule") == "loop_detector" and (gate.get("details") or {}).get("soft_unblock")
            and isinstance(body_json, dict)):
        _note = (gate.get("details") or {}).get("note")
        if _note and _inject_system_reminder(body_json, _note):
            body = json.dumps(body_json).encode("utf-8")
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

    # Registered BEFORE the send, not after it. client.send(stream=True) does not return until
    # the upstream produces response headers, and for a long prefill — or a request queued
    # behind a busy single-slot model — that is minutes away. Registering afterwards meant the
    # Live view was blind to exactly the requests worth watching: a 300k-token prefill was
    # invisible for its whole six minutes, and a queued request never appeared at all. (Short
    # ones still showed, because the tile query also picks up rows from the last 32 seconds,
    # which is why this looked like an intermittent glitch rather than a rule.)
    _INFLIGHT_REQUESTS[req_id] = {"ts": time.time(), "upstream_resp": None,
                                  "upstream": upstream_label, "cancelled": False}
    try:
        upstream_req = client.build_request(
            request.method, upstream_url, headers=headers_out, content=body or None
        )
        upstream_resp = await client.send(upstream_req, stream=True)
    except Exception as e:
        _release_pri_slot()
        elapsed = (time.perf_counter() - start) * 1000
        explained = _explain_upstream_error(e, upstream_label, upstream_url, elapsed,
                                            got_bytes=False)
        _save_finish(req_id, 0, {}, None, None, elapsed, explained)
        return JSONResponse(
            {"error": "upstream unreachable", "upstream": upstream_base,
             "detail": str(e), "explanation": explained},
            status_code=502,
        )

    # Attach the socket to the entry registered above, so /api/control/cancel/{req_id} has
    # something to close.
    _entry = _INFLIGHT_REQUESTS.get(req_id)
    if _entry is None:            # zombie-killer reaped it while we waited on headers
        _entry = {"ts": time.time(), "upstream": upstream_label, "cancelled": False}
        _INFLIGHT_REQUESTS[req_id] = _entry
    _entry["upstream_resp"] = upstream_resp
    if _entry.get("cancelled"):
        # Killed while the prefill was still running. There was no socket to close at the time,
        # so honour it now rather than streaming out a response nobody is waiting for.
        try:
            await upstream_resp.aclose()
        except Exception:
            pass
        _release_pri_slot()
        _save_finish(req_id, 499, {}, None, None, (time.perf_counter() - start) * 1000,
                     "cancelled while waiting for the upstream to respond")
        return JSONResponse({"error": "cancelled"}, status_code=499)

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

        # Shared finalize-state so a client disconnect (which cancels streamer() mid-yield,
        # before its own _save_finish runs) can still be finalized by the outer wrapper. Without
        # this, an aborted stream leaves the row stuck 'pending' and leaks the inflight/live entry.
        _fin: dict = {"saved": False, "chunks": None}

        async def streamer():
            # REVERTED to simple async-for. The earlier asyncio.wait keepalive pattern
            # broke bridged claude-code requests starting ~2026-05-15 (success rate went
            # from ~97% to 0%). Keepalive bytes are now sent ONCE immediately and once
            # before generation starts; the simple async-for handles the actual upstream
            # reads without the create_task/wait dance that was wedging.
            chunks: list[bytes] = []
            err: str | None = None
            first_chunk_ms: float | None = None
            _fin["chunks"] = chunks   # expose accumulated bytes to the disconnect backstop
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
                # Rename before validating, so a fixable name is not reported as a hallucination.
                _alias_changes = _apply_tool_aliases(resp_obj, body_json)
                if _alias_changes:
                    _save_gate(req_id, {
                        "verdict": "rewrite", "rule": "tool_aliases",
                        "reason": ", ".join(c["from"] + " -> " + c["to"] for c in _alias_changes),
                        "details": {"changes": _alias_changes},
                    })
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
            # A streamed 5xx carries its reason in the body too — vLLM sends the whole
            # EngineDeadError before closing. Keep it unless something already said more.
            _save_finish(req_id, upstream_resp.status_code, resp_headers_full, None, full, elapsed,
                         err or _upstream_error_message(upstream_resp.status_code, full),
                         ttft_ms=first_chunk_ms)
            _fin["saved"] = True

        # Wrap streamer to (a) release the priority slot even on disconnect/error, (b)
        # optionally rewrite the `model` field in every emitted chunk when the user asked
        # for it (preserve_response_model_name), and (c) tee each chunk into the dedup
        # fanout so concurrent duplicates receive the same bytes.
        async def _gen_with_release():
            stream_err: str | None = None
            sent_any = False
            try:
                async for chunk in streamer():
                    sent_any = sent_any or bool(chunk)
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
                # Mid-stream failures are the ones worth explaining: the client already had
                # part of an answer, so "ReadError('')" tells whoever reads this row nothing
                # about why it stopped halfway.
                stream_err = _explain_upstream_error(
                    e, upstream_label, upstream_url, (time.perf_counter() - start) * 1000,
                    got_bytes=bool(sent_any))
                raise
            finally:
                _release_pri_slot()
                if _dedup_fanout is not None:
                    _dedup_fanout.finish(stream_err)
                # Backstop: if streamer() didn't reach its own _save_finish (client disconnected
                # mid-stream → GeneratorExit/CancelledError, or an error before the final save),
                # finalize the row here so it never stays 'pending' and the inflight/live entries
                # are freed. Status 499 = client closed request (nginx convention).
                if not _fin.get("saved"):
                    try:
                        _partial = b"".join(_fin.get("chunks") or []).decode("utf-8", errors="replace")
                    except Exception:
                        _partial = ""
                    try:
                        _save_finish(req_id, 499, resp_headers_full, None, _partial,
                                     (time.perf_counter() - start) * 1000,
                                     stream_err or "client disconnected before response completed")
                    except Exception:
                        pass
                    _fin["saved"] = True
                # Unconditional, and outside the `not saved` branch it used to live in: on the
                # happy path streamer() has already closed this and a second aclose() is a
                # no-op, but "streamer finished normally" and "the socket was released" are not
                # the same claim, and betting a pool slot on them being equal is what filled
                # the pool. Closing twice is free; closing never is a 25-minute outage.
                try:
                    await upstream_resp.aclose()   # free the upstream socket / GPU slot
                except Exception:
                    pass

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
            # Rename before validating, so a fixable name is not reported as a hallucination.
            _alias_changes = _apply_tool_aliases(resp_obj, body_json)
            if _alias_changes:
                _save_gate(req_id, {
                    "verdict": "rewrite", "rule": "tool_aliases",
                    "reason": ", ".join(c["from"] + " -> " + c["to"] for c in _alias_changes),
                    "details": {"changes": _alias_changes},
                })
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

    _save_finish(req_id, upstream_resp.status_code, resp_headers_full, body_text_resp, None,
                 elapsed, _upstream_error_message(upstream_resp.status_code, body_text_resp))

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
    lines.append("- `context_compressor` (transform): deterministically squeeze bulky tool outputs (strip ANSI, minify JSON, head/tail-truncate) already in the history to reclaim context/tokens. Lossy. `mode: shadow` measures potential savings only; `mode: live` compresses the forwarded body. Keys: `enabled`, `mode` (shadow|live), `tool_output_max_chars`, `json_min_chars`, `min_saved_chars`, `from_model_prefix`, `client_app`, `prompt_chars_gt`. Savings show at `/__proxy/api/compress-stats` and on the Stats page.")
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
    "read_scratchboard",
    "Read the shared scratchboard: working notes anyone on the network can post, newest "
    "first with pinned notes at the top. Use it to pick up context a human or another agent "
    "left behind — what is being worked on, what was already tried, what not to touch.",
    {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 200},
            "pinned_only": {"type": "boolean", "default": False,
                            "description": "Only the pinned notes — the standing context."},
        },
    },
)
async def _mcp_read_scratchboard(args: dict, request: Request | None = None):
    limit = max(1, min(int(args.get("limit", 50) or 50), 200))
    conn = db()
    where = "WHERE pinned = 1 " if args.get("pinned_only") else ""
    roots = conn.execute(
        "SELECT id, ts, author, text, pinned FROM scratchboard "
        + where + ("AND " if where else "WHERE ") + "parent_id IS NULL "
        "ORDER BY pinned DESC, ts DESC LIMIT ?", (limit,)).fetchall()
    replies: dict = {}
    for r in conn.execute("SELECT id, ts, author, text, parent_id FROM scratchboard "
                          "WHERE parent_id IS NOT NULL ORDER BY ts ASC").fetchall():
        replies.setdefault(r["parent_id"], []).append(r)
    conn.close()

    # Rendered rather than dumped: an agent reading this wants the notes, not a JSON envelope
    # around them, and timestamps are more useful as dates than as floats. Replies are indented
    # under their note so a question and its answer arrive together — a reply read apart from
    # what it answers is worse than not reading it.
    def fmt(r, indent=""):
        when = datetime.datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d %H:%M")
        who = f" — {r['author']}" if r["author"] else ""
        head = f"{indent}{'📌 ' if r.keys().__contains__('pinned') and r['pinned'] else ''}[{when}{who}]"
        body = "\n".join(indent + line for line in (r["text"] or "").splitlines())
        return f"{head}\n{body}"

    out = []
    for r in roots:
        block = [fmt(r)]
        for rep in replies.get(r["id"], []):
            block.append(fmt(rep, indent="    ↳ "))
        out.append("\n".join(block))
    return _mcp_text_result("\n\n".join(out) or "(the scratchboard is empty)")


@mcp_tool(
    "write_scratchboard",
    "Post a note to the shared scratchboard so humans and other agents can see it. "
    "WRITE TOOL — gated by MCP_ALLOW_WRITE.",
    {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The note."},
            "author": {"type": "string", "description": "Who is posting — an agent name is fine."},
            "pinned": {"type": "boolean", "default": False,
                       "description": "Pin as standing context rather than a scratch note."},
            "reply_to": {"type": "string",
                         "description": "Note id to reply to — answer a question on the board "
                                        "rather than starting a new thread."},
        },
        "required": ["text"],
    },
    write=True,
)
async def _mcp_write_scratchboard(args: dict, request: Request | None = None):
    text = (args.get("text") or "").strip()
    if not text:
        raise ValueError("text is required")
    if len(text) > _SCRATCH_MAX_CHARS:
        raise ValueError(f"text too long ({len(text)} chars, limit {_SCRATCH_MAX_CHARS})")
    entry_id = uuid.uuid4().hex[:16]
    conn = db()
    parent = (args.get("reply_to") or "").strip() or None
    if parent:
        row = conn.execute("SELECT id, parent_id FROM scratchboard WHERE id = ?",
                           (parent,)).fetchone()
        if not row:
            conn.close()
            raise ValueError(f"no note {parent!r} to reply to")
        parent = row["parent_id"] or row["id"]      # flatten to one level
    conn.execute("INSERT INTO scratchboard (id, ts, author, text, client_ip, pinned, parent_id) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (entry_id, time.time(), (args.get("author") or "mcp").strip()[:64],
                  text, _client_ip(request) if request else None,
                  1 if (args.get("pinned") and not parent) else 0, parent))
    conn.commit()
    conn.close()
    return _mcp_text_result({"ok": True, "id": entry_id})


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

                await _wait_for_http_start(http_server, http_task, STARTUP_TIMEOUT_S)
                https_task = asyncio.create_task(https_server.serve())
                try:
                    # Not gather(): the HTTPS server runs with lifespan="off" and borrows the
                    # HTTP server's client and database, so it must never outlive it answering
                    # requests against torn-down state. Whichever stops first takes the other
                    # down with it.
                    done, pending = await asyncio.wait(
                        {http_task, https_task}, return_when=asyncio.FIRST_COMPLETED)
                    http_server.should_exit = True
                    https_server.should_exit = True
                    for t in pending:
                        try:
                            await asyncio.wait_for(t, timeout=(GRACEFUL_SHUTDOWN_S or 10) + 5)
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            t.cancel()
                    for t in done:
                        exc = t.exception()
                        if exc and not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                            raise exc
                except (KeyboardInterrupt, asyncio.CancelledError):
                    pass

            try:
                asyncio.run(_serve_both())
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
