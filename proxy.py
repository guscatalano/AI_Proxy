import os
import json
import time
import sqlite3
import uuid
import hashlib
import asyncio
import shutil
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, Response

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
LMSTUDIO_URL = os.environ.get("LMSTUDIO_URL", "http://localhost:1234").rstrip("/")
PROXY_HOST = os.environ.get("PROXY_HOST", "0.0.0.0")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8000"))
DB_PATH = os.environ.get("PROXY_DB", "proxy.db")
STATIC_DIR = Path(__file__).parent / "static"
METRICS_INTERVAL_S = float(os.environ.get("PROXY_METRICS_INTERVAL", "5"))
METRICS_RETENTION_S = float(os.environ.get("PROXY_METRICS_RETENTION", str(24 * 3600)))

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
"""

SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts DESC);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model);
CREATE INDEX IF NOT EXISTS idx_requests_client ON requests(client_ip);
CREATE INDEX IF NOT EXISTS idx_requests_verdict ON requests(gate_verdict);
CREATE INDEX IF NOT EXISTS idx_requests_conversation ON requests(conversation_id, ts);
CREATE INDEX IF NOT EXISTS idx_metrics_ts ON system_metrics(ts DESC);
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
    # Backfill token counts for rows captured before usage tracking existed.
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
    # Backfill conversation IDs and turn indices for rows captured before conversation tracking existed.
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
    conn.commit()
    conn.close()


def _extract_usage(body_text, stream_text):
    """Pull (prompt, completion, total) tokens out of a JSON body or SSE stream chunks."""
    pt = ct = tt = None

    def from_obj(j):
        if not isinstance(j, dict):
            return None, None, None
        u = j.get("usage")
        if isinstance(u, dict):
            return u.get("prompt_tokens"), u.get("completion_tokens"), u.get("total_tokens")
        # Ollama native (/api/chat, /api/generate)
        if "prompt_eval_count" in j or "eval_count" in j:
            return j.get("prompt_eval_count"), j.get("eval_count"), None
        return None, None, None

    if body_text:
        try:
            pt, ct, tt = from_obj(json.loads(body_text))
        except json.JSONDecodeError:
            pass

    if stream_text and pt is None and ct is None and tt is None:
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
            if p is not None: pt = p
            if c is not None: ct = c
            if t is not None: tt = t

    if tt is None and (pt is not None or ct is not None):
        tt = (pt or 0) + (ct or 0)
    return pt, ct, tt


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(None))
    app.state.metrics_client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
    app.state.metrics_task = asyncio.create_task(_metrics_loop(app))
    try:
        yield
    finally:
        app.state.metrics_task.cancel()
        try:
            await app.state.metrics_task
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


def _gpu_snapshot() -> list:
    """Returns [{idx, name, util_pct, mem_total_mb, mem_used_mb, temp_c, processes:[{pid, name, mem_mb}]}]"""
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
            uuid_to_idx[parts[1]] = int(parts[0])

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
                "mem_mb": int(mem_s) if mem_s.isdigit() else 0,
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
            gpus.append({
                "idx": idx,
                "name": parts[1],
                "util_pct": int(parts[2]) if parts[2].isdigit() else None,
                "mem_total_mb": int(parts[3]) if parts[3].isdigit() else None,
                "mem_used_mb": int(parts[4]) if parts[4].isdigit() else None,
                "temp_c": int(parts[5]) if parts[5].isdigit() else None,
                "processes": procs_by_idx.get(idx, []),
            })
        except (ValueError, IndexError):
            pass
    return gpus


async def _ollama_snapshot(client: httpx.AsyncClient) -> dict:
    out = {"reachable": False, "ps": [], "tags": []}
    try:
        r = await client.get(f"{OLLAMA_URL}/api/ps")
        if r.status_code == 200:
            out["reachable"] = True
            data = r.json()
            for m in (data.get("models") or []):
                out["ps"].append({
                    "name": m.get("name"),
                    "model": m.get("model"),
                    "size_mb": (m.get("size") or 0) // (1024 * 1024) if m.get("size") else None,
                    "size_vram_mb": (m.get("size_vram") or 0) // (1024 * 1024) if m.get("size_vram") else None,
                    "expires_at": m.get("expires_at"),
                    "parameter_size": (m.get("details") or {}).get("parameter_size"),
                })
    except (httpx.RequestError, ValueError):
        return out
    try:
        r = await client.get(f"{OLLAMA_URL}/api/tags")
        if r.status_code == 200:
            data = r.json()
            for m in (data.get("models") or []):
                out["tags"].append({
                    "name": m.get("name"),
                    "size_mb": (m.get("size") or 0) // (1024 * 1024) if m.get("size") else None,
                    "modified_at": m.get("modified_at"),
                    "parameter_size": (m.get("details") or {}).get("parameter_size"),
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


async def _collect_once(app: FastAPI):
    cpu = _cpu_pct()
    mem = _mem_snapshot()
    gpus = _gpu_snapshot()
    ollama, lmstudio = await asyncio.gather(
        _ollama_snapshot(app.state.metrics_client),
        _lmstudio_snapshot(app.state.metrics_client),
        return_exceptions=False,
    )
    ts = time.time()
    conn = db()
    conn.execute(
        """INSERT OR REPLACE INTO system_metrics
           (ts, cpu_pct, load_1m, mem_total_mb, mem_used_mb, mem_avail_mb, gpu_json, ollama_json, lmstudio_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
        ),
    )
    # Retention
    conn.execute("DELETE FROM system_metrics WHERE ts < ?", (ts - METRICS_RETENTION_S,))
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
    "model_router": {
        "enabled": False,
        # Static aliases applied first: { "from_model": "to_model" }
        "aliases": {},
        # Conditional rules; first match wins (after aliases). Each rule:
        #   { "if": { ...conditions... }, "then": "target_model_name" }
        # Conditions (any combination, all must match):
        #   from_model: string | [strings]      match the model name (post-alias)
        #   from_client: string | [strings]     exact client IP
        #   from_client_prefix: string          IP prefix
        #   prompt_chars_lt: int                total chars across all message content
        #   prompt_chars_gt: int                total chars across all message content
        #   has_tools: bool                     request includes a non-empty tools[] array
        #   path_prefix: string                 match URL path prefix
        "rules": [],
    },
}

RULES_FILE = os.environ.get("PROXY_RULES_FILE", str(Path(__file__).parent / "rules.json"))


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


def _match_router_cond(cond: dict, body: dict, ctx: dict) -> bool:
    """Return True iff every key in `cond` matches the request."""
    if not isinstance(cond, dict):
        return False

    def as_list(v):
        return v if isinstance(v, list) else [v]

    if "from_model" in cond:
        if (body.get("model") or "") not in as_list(cond["from_model"]):
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
            break

    if not target or target == original:
        return None
    body["model"] = target
    return {"from": original, "to": target, "via": via, "condition": matched_cond}


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
    return {"upstream": OLLAMA_URL, "lmstudio": LMSTUDIO_URL, "port": PROXY_PORT}


@app.get("/__proxy/api/system/now")
async def system_now():
    conn = db()
    row = conn.execute(
        """SELECT ts, cpu_pct, load_1m, mem_total_mb, mem_used_mb, mem_avail_mb,
                  gpu_json, ollama_json, lmstudio_json
           FROM system_metrics ORDER BY ts DESC LIMIT 1"""
    ).fetchone()
    conn.close()
    if not row:
        return {"ts": None, "cpu_pct": None, "mem": None, "gpus": [], "ollama": {}, "lmstudio": {}}
    d = dict(row)
    return {
        "ts": d["ts"],
        "cpu_pct": d["cpu_pct"],
        "load_1m": d["load_1m"],
        "mem": {"total_mb": d["mem_total_mb"], "used_mb": d["mem_used_mb"], "avail_mb": d["mem_avail_mb"]},
        "gpus": json.loads(d["gpu_json"]) if d["gpu_json"] else [],
        "ollama": json.loads(d["ollama_json"]) if d["ollama_json"] else {},
        "lmstudio": json.loads(d["lmstudio_json"]) if d["lmstudio_json"] else {},
    }


@app.get("/__proxy/api/system/history")
async def system_history(minutes: int = 60):
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
    return {"minutes": minutes, "samples": out}


@app.get("/__proxy/api/requests")
async def list_requests(limit: int = 200, offset: int = 0):
    conn = db()
    rows = conn.execute(
        """SELECT id, ts, method, path, model, is_stream, status, duration_ms, error,
                  prompt_tokens, completion_tokens, total_tokens, client_ip,
                  gate_verdict, gate_rule, gate_reason
           FROM requests ORDER BY ts DESC LIMIT ? OFFSET ?""",
        (limit, offset),
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    conn.close()
    return {"total": total, "items": [dict(r) for r in rows]}


@app.get("/__proxy/api/audit")
async def audit(limit: int = 200, include_allow: bool = False):
    conn = db()
    if include_allow:
        rows = conn.execute(
            """SELECT id, ts, method, path, model, client_ip, gate_verdict, gate_rule, gate_reason, gate_details
               FROM requests WHERE gate_verdict IS NOT NULL
               ORDER BY ts DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT id, ts, method, path, model, client_ip, gate_verdict, gate_rule, gate_reason, gate_details
               FROM requests WHERE gate_verdict IN ('block', 'warn', 'rewrite')
               ORDER BY ts DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    counts = {}
    for v, n in conn.execute(
        "SELECT COALESCE(gate_verdict, '(none)'), COUNT(*) FROM requests GROUP BY gate_verdict"
    ).fetchall():
        counts[v] = n
    conn.close()
    return {"counts": counts, "items": [dict(r) for r in rows]}


@app.get("/__proxy/api/conversations")
async def list_conversations(limit: int = 100):
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
                  SUM(CASE WHEN gate_verdict = 'block' THEN 1 ELSE 0 END) AS blocks,
                  SUM(CASE WHEN gate_verdict = 'rewrite' THEN 1 ELSE 0 END) AS rewrites
           FROM requests
           WHERE conversation_id IS NOT NULL
           GROUP BY conversation_id
           ORDER BY last_ts DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    # Pull a preview of the first user message per conversation.
    items = []
    for r in rows:
        d = dict(r)
        preview = conn.execute(
            """SELECT request_body FROM requests
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
        items.append(d)
    conn.close()
    return {"items": items}


@app.get("/__proxy/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    conn = db()
    rows = conn.execute(
        """SELECT id, ts, model, turn_index, request_body, response_body, stream_chunks,
                  prompt_tokens, completion_tokens, total_tokens, duration_ms,
                  status, error, gate_verdict, gate_rule, gate_reason, gate_details,
                  client_ip, is_stream
           FROM requests
           WHERE conversation_id = ?
           ORDER BY ts ASC""",
        (conv_id,),
    ).fetchall()
    conn.close()
    return {"conversation_id": conv_id, "turns": [dict(r) for r in rows]}


@app.get("/__proxy/api/suggestions")
async def suggestions():
    """Scan recent traffic and surface config tuning recommendations."""
    conn = db()
    cutoff = time.time() - 30 * 86400
    rows = conn.execute(
        """SELECT model, request_body, response_body, stream_chunks, prompt_tokens, completion_tokens,
                  total_tokens, duration_ms, client_ip, gate_verdict, gate_rule
           FROM requests
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

    return {"sample_size": len(rows), "items": out}


@app.get("/__proxy/api/rules")
async def get_rules():
    cfg = load_rules_config()
    src, raw = _rules_source()
    setting = get_setting("rules")
    return {
        "registered": list(RULES_REGISTRY.keys()),
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


@app.get("/__proxy/api/stats")
async def stats():
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

    # Tool usage requires parsing response bodies — scan up to a bounded set.
    tool_rows = conn.execute(
        """SELECT model, response_body, stream_chunks FROM requests
           WHERE response_body IS NOT NULL OR stream_chunks IS NOT NULL
           ORDER BY ts DESC LIMIT 5000"""
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

    conn.close()
    return {
        "overall": overall,
        "by_model": by_model,
        "by_path": by_path,
        "by_status": by_status,
        "by_client": by_client,
        "by_tool": by_tool_list,
    }


@app.get("/__proxy/api/requests/{req_id}")
async def get_request(req_id: str):
    conn = db()
    row = conn.execute("SELECT * FROM requests WHERE id = ?", (req_id,)).fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "not found"}, status_code=404)
    return dict(row)


@app.post("/__proxy/api/clear")
async def clear_requests():
    conn = db()
    conn.execute("DELETE FROM requests")
    conn.commit()
    conn.close()
    return {"ok": True}


@app.get("/__proxy")
@app.get("/__proxy/")
async def ui_index():
    return FileResponse(STATIC_DIR / "index.html")


# -------- Transparent proxy --------

REQ_HOP_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}
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
        parts = []
        for p in c:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "\n".join(parts)
    if c is None:
        return ""
    try:
        return json.dumps(c, sort_keys=True)
    except (TypeError, ValueError):
        return str(c)


def _conversation_id(body) -> str | None:
    """Stable conversation hash: derived from system message + first user message.
    Survives appended turns, breaks if the client truncates earlier history."""
    if not isinstance(body, dict):
        return None
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        sys_msg = next((m for m in messages if isinstance(m, dict) and m.get("role") == "system"), None)
        first_user = next((m for m in messages if isinstance(m, dict) and m.get("role") == "user"), None)
        parts: list[str] = []
        if sys_msg:
            parts.append("system:" + _msg_text(sys_msg).strip())
        if first_user:
            parts.append("user:" + _msg_text(first_user).strip())
        if parts:
            return hashlib.sha256("\n---\n".join(parts).encode("utf-8")).hexdigest()[:16]
    # Ollama /api/generate has no messages — single-shot, no conversation.
    return None


def _turn_index(body) -> int | None:
    if not isinstance(body, dict):
        return None
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return None
    return sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "user")


def _client_ip(request: Request) -> str | None:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    xri = request.headers.get("x-real-ip")
    if xri:
        return xri.strip()
    return request.client.host if request.client else None


def _save_pending(req_id: str, request: Request, full_path: str, upstream_url: str, body_text: str, body_json, model, is_stream: bool):
    conv_id = _conversation_id(body_json)
    turn = _turn_index(body_json)
    conn = db()
    conn.execute(
        """INSERT INTO requests (id, ts, method, path, upstream_url, request_headers, request_body, model, is_stream, client_ip, conversation_id, turn_index)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            req_id,
            time.time(),
            request.method,
            "/" + full_path,
            upstream_url,
            json.dumps(dict(request.headers)),
            body_text,
            model,
            int(is_stream),
            _client_ip(request),
            conv_id,
            turn,
        ),
    )
    conn.commit()
    conn.close()


def _extract_tool_calls(response_body, stream_text):
    """Return list of tool function names actually invoked in a response."""
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
        names.extend(per_idx.values())

    return names


def _save_finish(req_id: str, status: int, resp_headers: dict, body_text: str | None, stream_text: str | None, elapsed_ms: float, error: str | None):
    pt, ct, tt = _extract_usage(body_text, stream_text)
    conn = db()
    conn.execute(
        """UPDATE requests
           SET status=?, response_headers=?, response_body=?, stream_chunks=?, duration_ms=?, error=?,
               prompt_tokens=?, completion_tokens=?, total_tokens=?
           WHERE id=?""",
        (
            status,
            json.dumps(resp_headers) if resp_headers else None,
            body_text,
            stream_text,
            elapsed_ms,
            error,
            pt, ct, tt,
            req_id,
        ),
    )
    conn.commit()
    conn.close()


@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy(full_path: str, request: Request):
    # Defensive: never proxy our own UI/API namespace.
    if full_path.startswith("__proxy"):
        return JSONResponse({"error": "not found"}, status_code=404)

    req_id = uuid.uuid4().hex
    start = time.perf_counter()
    body = await request.body()

    upstream_url = f"{OLLAMA_URL}/{full_path}"
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

    _save_pending(req_id, request, full_path, upstream_url, body_text, body_json, model, is_stream)

    # Phase 1: routing transform (mutates body_json["model"] when a rewrite fires).
    router_ctx = {"client_ip": _client_ip(request), "path": "/" + full_path}
    rewrite = evaluate_router(body_json, router_ctx) if isinstance(body_json, dict) else None
    if rewrite:
        # Re-serialize the body so the rewritten model is what we forward upstream.
        body = json.dumps(body_json).encode("utf-8")
        model = body_json.get("model")
        # Reflect the rewrite in the captured row: keep original request_body, update routed model.
        conn = db()
        conn.execute("UPDATE requests SET model=? WHERE id=?", (model, req_id))
        conn.commit()
        conn.close()

    # Phase 2: block/warn rules.
    gate = evaluate_rules(body_json)

    # Combine: rewrite gets folded in; block beats rewrite, rewrite beats warn beats allow.
    if gate["verdict"] == "block":
        if rewrite:
            details = gate.get("details") or {}
            details["rewrite"] = rewrite
            gate["details"] = details
        _save_gate(req_id, gate)
        elapsed = (time.perf_counter() - start) * 1000
        err_payload = {
            "error": {
                "message": f"Request blocked by AI Proxy rule {gate['rule']!r}: {gate['reason']}",
                "type": "proxy_block",
                "code": gate["rule"],
            }
        }
        err_text = json.dumps(err_payload)
        _save_finish(req_id, 400, {"content-type": "application/json", "x-proxy-block": gate["rule"]}, err_text, None, elapsed, f"blocked: {gate['reason']}")
        return JSONResponse(err_payload, status_code=400, headers={"X-Proxy-Block": gate["rule"], "X-Proxy-Reason": (gate["reason"] or "")[:200]})

    if rewrite:
        # A rewrite happened; record it as the dominant verdict (audit-visible).
        if gate["verdict"] == "warn":
            # Preserve the warn rule alongside the rewrite.
            combined_details = {"rewrite": rewrite, "warn": {"rule": gate["rule"], "reason": gate["reason"], "details": gate.get("details")}}
            gate = {"verdict": "rewrite", "rule": "model_router", "reason": f"{rewrite['from']} → {rewrite['to']} (via {rewrite['via']}); also: {gate['reason']}", "details": combined_details}
        else:
            gate = {"verdict": "rewrite", "rule": "model_router", "reason": f"{rewrite['from']} → {rewrite['to']} (via {rewrite['via']})", "details": {"rewrite": rewrite}}

    _save_gate(req_id, gate)

    headers_out = _filter(request.headers, REQ_HOP_HEADERS)
    client: httpx.AsyncClient = request.app.state.client

    try:
        upstream_req = client.build_request(
            request.method, upstream_url, headers=headers_out, content=body or None
        )
        upstream_resp = await client.send(upstream_req, stream=True)
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        _save_finish(req_id, 0, {}, None, None, elapsed, f"upstream error: {e!r}")
        return JSONResponse(
            {"error": "upstream unreachable", "upstream": OLLAMA_URL, "detail": str(e)},
            status_code=502,
        )

    resp_headers_full = dict(upstream_resp.headers)
    out_headers = dict(_filter(upstream_resp.headers, RESP_HOP_HEADERS))
    content_type = upstream_resp.headers.get("content-type", "")
    treat_as_stream = ("text/event-stream" in content_type) or ("application/x-ndjson" in content_type) or is_stream

    if treat_as_stream:
        async def streamer():
            chunks: list[bytes] = []
            err: str | None = None
            try:
                async for chunk in upstream_resp.aiter_raw():
                    chunks.append(chunk)
                    yield chunk
            except Exception as e:
                err = f"stream error: {e!r}"
            finally:
                try:
                    await upstream_resp.aclose()
                except Exception:
                    pass
                full = b"".join(chunks).decode("utf-8", errors="replace")
                elapsed = (time.perf_counter() - start) * 1000
                _save_finish(req_id, upstream_resp.status_code, resp_headers_full, None, full, elapsed, err)

        return StreamingResponse(
            streamer(),
            status_code=upstream_resp.status_code,
            headers=out_headers,
            media_type=content_type or "text/event-stream",
        )

    body_bytes = await upstream_resp.aread()
    await upstream_resp.aclose()
    elapsed = (time.perf_counter() - start) * 1000
    body_text_resp = body_bytes.decode("utf-8", errors="replace")
    _save_finish(req_id, upstream_resp.status_code, resp_headers_full, body_text_resp, None, elapsed, None)

    return Response(
        content=body_bytes,
        status_code=upstream_resp.status_code,
        headers=out_headers,
        media_type=content_type or "application/octet-stream",
    )


if __name__ == "__main__":
    import uvicorn

    print(f"AI Proxy → forwarding to {OLLAMA_URL}")
    print(f"Listening on http://{PROXY_HOST}:{PROXY_PORT}")
    print(f"UI: http://{PROXY_HOST}:{PROXY_PORT}/__proxy/")
    uvicorn.run("proxy:app", host=PROXY_HOST, port=PROXY_PORT, reload=False)
