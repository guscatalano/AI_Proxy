# AI Proxy

A transparent inspector and rule engine that sits between your AI clients (Claude Code, GitHub Copilot Chat, Cursor, raw SDKs) and their upstreams (Anthropic, Ollama, LM Studio, anything OpenAI-compatible). Logs every request, surfaces conversations and tool calls, applies configurable rules, and gives you a live web UI to see what's actually happening.

Built around two ideas:
1. **Observability without changing your client** — set `ANTHROPIC_BASE_URL` (or `OLLAMA_URL`) to the proxy and every request, response, tool call, token count, and conversation thread is captured automatically.
2. **Rules that improve model behavior** — pre/post-flight hooks catch loops, fix malformed tool calls, prune unused tools, prevent silent context-window truncation, route requests across models, and more.

![Live traffic flow with rule pipeline](docs/screenshots/tab-stats.png)

---

## What it does

- **Inspect every request** — full bodies, headers, streaming chunks, tool calls, token counts, latency. Searchable. PII-redacted by default for cross-network viewers.
- **Group requests into conversations** — automatic threading by `system + first_user` hash, with a turn-by-turn timeline view.
- **Identify the client app** — distinguishes Claude Code, VS Code Copilot Chat, Cursor, Anthropic SDK, OpenAI SDK, LangChain, browsers, etc. via header/User-Agent fingerprinting.
- **Translate Anthropic ↔ OpenAI** — point Claude Code at any OpenAI-compatible backend (Ollama, LM Studio, vLLM). The proxy translates request/response bodies and SSE streams in both directions.
- **Shadow runs** — send the same request to a primary upstream AND a local model in parallel, store both, and get an automatic side-by-side comparison page (latency delta, token delta, tool-call agreement, text similarity).
- **Rule pipeline** — pluggable pre-flight (block/warn/transform) and post-flight (intercept/autofix) rules with a JSON editor and quick-toggle UI.
- **Auditor suggestions** — the proxy analyzes recent traffic and recommends config changes (route slow-short requests off Opus, prune unused tools, bump `OLLAMA_NUM_PARALLEL`, etc.).
- **MCP server** — exposes the same data via Model Context Protocol so an LLM can query traffic patterns directly.
- **Restart from the UI** — one button to bounce the systemd-managed proxy, with health-check polling for confirmation.

---

## Screenshots

### Requests list — every API call, with client app badges
![Requests list](docs/screenshots/tab-requests.png)

### Request detail — body, response, headers, gate verdict, downsize heuristic
![Request detail](docs/screenshots/request-detail.png)

### Conversations — threaded turns with TOC navigation
![Conversation detail](docs/screenshots/conversation-detail.png)

### Shadow comparison — side-by-side primary vs local model
![Compare view](docs/screenshots/compare-view.png)

### Audit tab — routing toggle, rule editor, automatic suggestions
![Audit tab](docs/screenshots/tab-audit.png)

### System — CPU, memory, GPU, loaded models, restart button
![System tab](docs/screenshots/tab-system.png)

---

## Quick start

### Prerequisites
- Python 3.10+
- An upstream to forward to (Ollama on `localhost:11434`, Anthropic API, etc.)

### Install

```bash
git clone https://github.com/guscatalano/AI_Proxy.git
cd AI_Proxy
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
python -m ai_proxy
```

Or install it as a command (`ai-proxy`) with an isolated environment:

```bash
pipx install .    # or: pip install .
ai-proxy
```

UI: <http://127.0.0.1:8000/__proxy/>

### Point your client at the proxy

```bash
# Claude Code (and any Anthropic SDK client)
export ANTHROPIC_BASE_URL=http://localhost:8000
claude

# OpenAI SDK / VS Code Copilot Chat / Cursor / Continue / Cline
export OPENAI_BASE_URL=http://localhost:8000/v1
# or set the equivalent in the client's settings

# Ollama-native clients (set OLLAMA_HOST)
export OLLAMA_HOST=http://localhost:8000
```

The proxy routes by path: `/v1/messages*` and `/v1/complete*` go to Anthropic (`ANTHROPIC_URL` env var, default `https://api.anthropic.com`); everything else (`/v1/chat/completions`, `/api/*`) goes to Ollama (`OLLAMA_URL` env var, default `http://localhost:11434`).

---

## Configuration

### Environment variables

| Variable | Default | What it does |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | OpenAI-compat / Ollama upstream |
| `ANTHROPIC_URL` | `https://api.anthropic.com` | Anthropic upstream for `/v1/messages*` |
| `LMSTUDIO_URL` | `http://localhost:1234` | Optional LM Studio for the System tab |
| `PROXY_HOST` | `127.0.0.1` | Bind address |
| `PROXY_PORT` | `8000` | Bind port |
| `PROXY_DB` | `./proxy.db` | SQLite DB path |
| `PROXY_RULES_FILE` | `./rules.json` | One-time rules import (subsequently stored in DB) |
| `PROXY_REDACT_PII` | `1` | Redact bodies/headers from cross-subnet viewers |
| `PROXY_REDACT_SUBNET_BITS` | `24` | IPv4 subnet width for PII gating |
| `MCP_ALLOW_WRITE` | `false` | Allow `update_rules` MCP tool |
| `MCP_API_KEY` | (none) | Bearer token for MCP endpoint |

### Rule pipeline

All rules are configured via a single JSON object, edited in the **Audit tab → Rules Editor** (or POSTed to `/__proxy/api/rules`). Saves apply on the next request — no restart needed.

| Rule | Phase | What it does |
|---|---|---|
| `model_router` | transform | Rewrite `model` based on conditions (from_model, prompt size, has_tools, client IP). |
| `ollama_options` | transform | Inject Ollama options (`num_ctx`, `keep_alive`, `cache_prompt`, etc.) when client didn't set them. |
| `protocol_bridge` | transform | When an Anthropic-shape request gets routed to a non-Claude model, translate body to OpenAI shape and route to Ollama; translate the SSE stream back on the way out. |
| `tool_pruner` | transform | Drop tool definitions the model has been offered repeatedly but never invoked in this conversation. |
| `context_overflow_guard` | transform | Estimate prompt tokens; warn / bump `num_ctx` / trim oldest messages / block when prompt exceeds the effective context window. |
| `shadow_router` | transform | Fan out a parallel shadow request to a comparison model. Best-effort, never blocks the primary. |
| `loop_detector` | pre-flight | Block when the same tool call repeats too often. |
| `tool_failure_breaker` | pre-flight | Block when a tool has N consecutive error results. |
| `schema_validator` | post-flight | Validate tool-call args against the request's tool schemas; replace invalid calls with corrective assistant content. |
| `hallucinated_tool` | post-flight | Reject tool calls naming functions not declared in the request. |
| `tool_args_autofix` | post-flight | Fill in missing required tool-call fields from configured defaults. |

### Routing modes

The Audit tab includes a one-click toggle between three modes (with auto-detection of the active mode):

- **Passthrough** — Anthropic requests go straight to api.anthropic.com.
- **Shadow** — primary still goes to Anthropic; local model runs in parallel for comparison.
- **Redirect** — Anthropic requests are translated to OpenAI shape and sent to the local model entirely.

---

## Deployment

### Linux (systemd) — one command

The installer creates an isolated venv, installs the app, writes a systemd unit and a
config file, and enables the service.

```bash
# System service (boots at startup, runs as a dedicated 'ai-proxy' user)
sudo ./deploy/install-service.sh

# ...or a per-user service (no root)
./deploy/install-service.sh --user
loginctl enable-linger "$USER"   # optional: keep running after logout / across reboots
```

Then edit the generated config and restart:

- System: `/etc/ai-proxy/ai-proxy.env` → `sudo systemctl restart ai-proxy`
- User: `~/.config/ai-proxy/ai-proxy.env` → `systemctl --user restart ai-proxy`

Add `--pypi` to install the published release instead of this checkout. The service
self-restarts via the System tab's "↻ Restart proxy" button (it exits non-zero;
`Restart=on-failure` brings it back).

### Docker

```bash
docker compose up -d          # UI at http://localhost:8000/__proxy/
```

State (DB, rules, generated images) persists in the `ai-proxy-data` volume. The compose
file reaches an Ollama on the host via `host.docker.internal`; edit `OLLAMA_URL` /
`ANTHROPIC_URL` for your setup. Or without compose:

```bash
docker build -t ai-proxy .
docker run -d -p 8000:8000 -v ai-proxy-data:/data \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  --add-host host.docker.internal:host-gateway ai-proxy
```

### Windows

```powershell
# Run as Administrator
New-Service -Name "AIProxy" `
  -BinaryPathName "C:\path\to\python.exe -m ai_proxy" `
  -DisplayName "AI Proxy" -StartupType Automatic
```

---

## Development

```bash
# from a source checkout
pip install -e ".[dev]"
pytest
```

The suite (in `tests/`) covers the proxy's API endpoints, the Anthropic↔OpenAI
translation, the writable-state path resolution, and the version single-source
invariant — all offline, no upstream required. CI runs it on every push before the
per-platform binary build.

### Cutting a release

Bump the version on `main`, then promote `main` to the `release` branch:

```bash
python scripts/bump_version.py minor --commit   # or: major | patch | 1.2.3
git push origin main
git push origin main:release                    # fast-forward release -> triggers publish
```

Updating the `release` branch triggers the release workflow, which builds a
self-contained binary per platform, publishes to npm and PyPI, and creates a
`vX.Y.Z` GitHub Release with the binaries. It reads the version from the committed
files and skips anything already published, so re-triggering is safe. See
[CHANGELOG.md](CHANGELOG.md).

---

## Architecture

```
clients ──► [PROXY :8000] ──► upstreams
                │
                ├── pre-flight: block/warn rules (loop_detector, tool_failure_breaker)
                ├── transform:  model_router, ollama_options, protocol_bridge,
                │               tool_pruner, context_overflow_guard
                ├── fan-out:    shadow_router (concurrent comparison runs)
                ├── upstream:   stream chunks captured + live token tracking
                └── post-flight: schema_validator, hallucinated_tool, tool_args_autofix
                                 (intercept invalid tool calls, replace with corrective content)

                ▼
          SQLite (proxy.db)
                │
                ├── /__proxy/  ◄── web UI (single-page, no build step)
                └── /__proxy/mcp ◄── MCP server (Model Context Protocol)
```

- **Single Python file** (`proxy.py`) — FastAPI + httpx, no other heavy deps.
- **SQLite for storage** — WAL mode, schema migrations baked in, automatic backfills for parser improvements.
- **Single static file** (`static/index.html`) — vanilla JS, dark theme, no build step or framework.
- **Streaming-aware** — buffers SSE only when post-flight intercept is enabled or `protocol_bridge` is translating; otherwise passes through chunk-by-chunk.

---

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/__proxy/api/info` | GET | Upstream URLs, port |
| `/__proxy/api/health` | GET | DB stats, process info, pre-flight overhead |
| `/__proxy/api/requests` | GET | List requests (with live token state for pending rows) |
| `/__proxy/api/requests/{id}` | GET | Full detail incl. shadows |
| `/__proxy/api/conversations` | GET | List grouped conversations |
| `/__proxy/api/conversations/{id}` | GET | Turn-by-turn timeline |
| `/__proxy/api/stats` | GET | Per-model, per-client, per-app, per-tool aggregates |
| `/__proxy/api/audit` | GET | Gate verdict log |
| `/__proxy/api/suggestions` | GET | Auto-detected config recommendations |
| `/__proxy/api/rules` | GET / POST | Rules config (live-edit) |
| `/__proxy/api/system/now` | GET | CPU, memory, GPU, loaded models |
| `/__proxy/api/restart` | POST | Self-restart (requires `X-Confirm: restart-now`) |
| `/__proxy/api/export` | GET | Markdown/JSON digest for AI review |
| `/__proxy/mcp` | POST | MCP server (JSON-RPC) |

---

## MCP integration

Register the proxy as an MCP server in any MCP-aware client:

```bash
claude mcp add --transport http ai-proxy http://localhost:8000/__proxy/mcp
```

Available tools: `list_recent_requests`, `get_request_detail`, `list_conversations`, `get_conversation`, `get_stats`, `get_audit`, `get_suggestions`, `get_rules`, `get_system_metrics`, `export_digest` (and `update_rules` when `MCP_ALLOW_WRITE=true`). Now Claude can answer "what's been going through the proxy in the last hour?" or "show me the slowest requests today" by querying the data directly.

---

## Screenshots automation

Reproducible UI screenshots via Playwright headless Chromium:

```bash
pip install playwright
playwright install chromium
sudo playwright install-deps chromium  # Linux only

python scripts/screenshots.py --url http://localhost:8000/__proxy/ --out docs/screenshots/
```

Generates 9 PNGs covering every tab plus request detail, conversation detail, and the shadow compare view. See `scripts/screenshots.py` for options.

---

## Security notes

- No authentication built in — bind to `127.0.0.1` for local-only or front with a reverse proxy that handles auth.
- PII redaction is on by default: viewers from a different subnet than the request originator see body/header/preview fields replaced with a placeholder. Loopback viewers always see everything. Tunable via `PROXY_REDACT_PII` and `PROXY_REDACT_SUBNET_BITS`.
- Stored data includes full request/response bodies and headers (which means API keys, since they're in headers). The DB lives at `PROXY_DB` (default `./proxy.db`); protect it accordingly.
- Anthropic API keys are stripped from headers when the proxy bridges a request to a non-Anthropic upstream (so they don't leak into Ollama logs).

---

## Troubleshooting

- **Proxy not starting**: check Ollama is running (`curl http://localhost:11434`), check the port isn't in use (`netstat -ano | findstr :8000` on Windows, `ss -ltn | grep :8000` on Linux).
- **Database errors**: delete `proxy.db` and restart for a fresh DB. Schema migrations run automatically on startup.
- **Slow shadow runs on Ollama**: see the auditor's `OLLAMA_NUM_PARALLEL` suggestion in the Audit tab. Each parallel slot is a separate KV-cache; with too few slots, interleaved conversations evict each other and pay full prefill every turn.
- **Conversations not grouping**: the proxy hashes `system + first_user` to derive a conversation ID. Per-request volatile content (e.g., timestamps, billing headers) is normalized out automatically; if you find a client that breaks grouping, the normalizer in `_normalize_for_cid` is the place to extend.
- **Tokens missing on Anthropic streams**: requires `Accept-Encoding: identity` to upstream (already forced by the proxy) — older rows from before that fix get repaired automatically by `backfill_v5`/`backfill_v7` migrations on next startup.

---

## Status

Active development. Single-file design, SQLite-backed, no build step. Pull requests welcome.
