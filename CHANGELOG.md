# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The version is single-sourced in `ai_proxy/_version.py`. Bump it with
`python scripts/bump_version.py <version|major|minor|patch> --commit` on `main`, then
promote `main` to the `release` branch (`git push origin main:release`) to trigger the
release workflow (publishes to PyPI and npm, and creates a `vX.Y.Z` GitHub Release).

## [Unreleased]

### Added
- **Benchmark runner, rebuilt around model comparison** — see
  [docs/benchmarking.md](docs/benchmarking.md).
  - **Reasoning-aware timing.** TTFT is now the first token of *any* kind; TTFC is the first
    *content* token, and the gap between them is reported as the reasoning phase along with the
    reasoning token count. Token counts come from the upstream's own `usage` rather than a word
    count. A run that returns zero completion tokens is recorded as a failure instead of a very
    fast success (this is how a context overflow presents).
  - **Thinking control** per run: `auto` / `on` / `off` / `off_prefill`, mapping onto the three
    mechanisms different engines actually honor — `chat_template_kwargs.enable_thinking`,
    `reasoning_effort: "none"`, and an empty `<think></think>` assistant prefill for
    LM Studio/llama.cpp.
  - **Sampling knobs** (temperature, top_p, top_k, min_p, seed, penalties) and an `extra_body`
    passthrough.
  - **Routing pinned by default.** New `x-proxy-no-router`, `x-proxy-upstream` and
    `x-proxy-no-nudge` request headers let a bench measure the model it named rather than
    whatever `model_router` would have substituted. The summary reports the model the upstream
    echoed back.
  - **Matrix sweeps.** models × prompt sizes × thinking modes × temperatures expand into one
    child run per cell, executed serially so cells never contend for the GPU.
  - **Graded task suites.** `coding-v1` (6 tasks / 27 cases) asks for named Python functions and
    executes the returned code against deterministic cases in a subprocess with a hard timeout,
    reporting a fully-correct rate, a case pass rate, and a per-task breakdown. Opt-in.
  - **Environment capture** per run — proxy version, GPU/VRAM, system memory, loaded models and
    Ollama config — so a result stays interpretable weeks later.
  - **GPU quiesce.** One toggle enables panic mode and unloads loaded Ollama models for the
    duration, restoring them afterward; exclusive mode alone only gates traffic through the
    proxy. The exclusive-mode safety cap now scales with the workload instead of expiring after
    a fixed 5 minutes mid-sweep.
  - **N-way comparison** (was hard-limited to 2) with quality beside perf, plus Markdown export
    and `GET /api/bench/report`. New `GET /api/bench/suites` and `GET /api/bench/models` (the
    model picker now spans Ollama, LM Studio and vLLM instead of Ollama only).
- **Artifact-capture kill switch.** `PROXY_ARTIFACTS=0` disables recording of files/URLs/
  directories touched via tool calls entirely, with a runtime toggle and an optional purge of
  already-stored rows at `GET`/`POST /api/control/artifacts` (admin/loopback only, same rule as
  the PII toggle). Enforced at both the sweep and the extraction function, so no path can write
  artifact rows while capture is off.

## [0.2.0] - 2026-07-25

The dashboard is rebuilt around a console shell, and the proxy grows three new
observability surfaces — Live View, Artifacts, and Bench — plus per-model quirk
handling and a deterministic context compressor.

### Added

#### Dashboard
- **Console shell.** Grouped navigation rail, status strip, and a global theme reskin,
  now the only dashboard skin. On narrow screens the rail becomes a slide-over drawer
  behind a hamburger.
- **Live View tab** (`GET /api/live`) — one tile per active or recently-finished
  conversation, merging the in-flight registry (elapsed time, live token counts) with
  each request's stored row. Includes a full-window mode that fills the browser window.
- **Artifacts tab** — every file, URL, directory, and image a model touched through a
  tool call, extracted from request bodies via a tool→artifact map and aggregated by
  path. Conversation-first drill-down, a folder browser for the Files category with
  per-folder file counts, per-variation content sizes (empty writes flagged), a "most
  active" ranking, and inline content viewing. Endpoints: `/api/artifacts`,
  `/api/artifacts/conversations`, `/api/artifacts/top`, `/api/artifacts/timeline`,
  `/api/artifacts/content`. The tool map is extensible via a `tool_artifact_map` rules
  entry; bare terminal commands are scanned for path-shaped arguments.
- **Bench tab** (`/api/bench/run`, `/api/bench/runs`, `/api/bench/runs/{id}`) plus
  `scripts/bench.py` — queue a benchmark against a model with configurable runs, token
  counts, and concurrency, optionally taking exclusive use of the GPU with a drain
  period first. Poll for progress and results.
- **Tasks tab** (`/api/control/tasks*`) — queue a prompt to run against the upstream,
  either one-shot or on a schedule (cron expression or `every 10m` / `every 2h` / `every 1d`),
  with cancel, pause, resume, and run-now controls.
- **Stats reorganized** into a KPI strip with a tabbed breakdown, and charts moved into
  a Trends tab with readable axis labels, a compact 3-per-row layout, and a per-chart
  enlarge toggle that survives a refresh.
- **Conversation timeline**, substantially reworked: a dedicated conversation page fed
  by a single-fetch transcript endpoint, in-conversation search, message-type filters
  (isolate tool calls, results, responses), collapsible long messages, and a
  distinguishing snippet on each tool call/result.
- **Request detail**: a thinking badge, and live streaming output while a request is
  still in flight (`GET /api/requests/{id}/live`, which serves the rolling reconstructed
  tail from the in-memory mirror and falls back to persisted chunks once complete).
- **System tab**: runtime sub-tabs, an access badge showing whether the current viewer
  sees redacted or full data, and an Ollama update check
  (`/api/ollama/update-check`, floor set by `PROXY_OLLAMA_MIN_VERSION`).

#### Proxy behavior
- **`context_compressor` rule (v1)** — a deterministic prompt compressor that squeezes
  bulky tool outputs and JSON blobs already in the message history before forwarding
  upstream. Runs in `shadow` mode (measure only) or `live` mode (actually compress),
  with per-tool savings broken out in the Stats panel. Conditionable on model prefix,
  client app, and prompt size. See [docs/context-compressor.md](docs/context-compressor.md).
- **Per-model quirks** (`ai_proxy/model_quirks.json`, override via `PROXY_MODEL_QUIRKS`,
  inspect via `GET /api/model-quirks`). Each entry matches a model-name pattern and
  declares how to handle reasoning — `force_off`, `default_off_optin` (off unless the
  client asks for high/medium reasoning effort), or `normal` — plus an optional system
  nudge. Hot-reloaded on file change. Ships with an Ornith entry whose thinking is
  opt-in and whose nudge curbs inline reasoning narration and over-elaboration.
- **vLLM upstream** via `VLLM_URL`, and LM Studio as a `model_router` routing target,
  so a single rule can send e.g. `qwen` to LM Studio while everything else stays on
  Ollama — all still logged through the proxy.
- **`ollama_options.num_ctx_max`** — a hard ceiling applied even to client-set values,
  so a client can't force a context window that balloons the KV cache and kills
  parallelism.
- **Runtime PII-redaction toggle** (`GET`/`POST /api/control/redact-pii`) — flip
  redaction without restarting or editing the environment.
- **Panic mode** (`GET`/`POST /api/control/panic`) — every proxied request returns 503
  while the proxy itself stays up, so the endpoint remains reachable to turn it back off.
- **Kill an in-flight request** (`POST /api/control/cancel/{id}`) — closes the upstream
  connection so Ollama 0.21+ aborts generation and frees the GPU slot.
- **Client fingerprinting from the system prompt**, in addition to headers — identifies
  Hermes Agent and its safety sub-agent. `scripts/relabel_clients.py` backfills labels
  on existing rows.
- **Vision routing and an image viewer** — route requests carrying images to a
  vision-capable model via the `has_images` router condition.
- **Streaming keepalives** to stop idle proxied streams from being dropped by
  intermediaries.
- Requests view: filter by client (dropdown of detected apps) and jump to a specific
  page number (the jump input works on the Audit/Suggestions/Conversations pagers too).
  `GET /api/requests` gains a `client` param and returns the available `clients`.
- `PROXY_GRACEFUL_SHUTDOWN` (default 10 s) caps uvicorn's shutdown wait so a lingering
  connection can't hang a `systemctl restart`.
- Data-growth guards (env-tunable): `PROXY_MAX_STORED_BODY` caps persisted body size
  (default 4 MB), `PROXY_REQUEST_RETENTION_DAYS` prunes old requests (default 30),
  `PROXY_ANALYTICS_CACHE_TTL` memoizes stats/suggestions. `system_history` is
  downsampled to ≤800 points per response.

#### Packaging
- **Windows installers**: an MSI built with WiX (attached to each GitHub Release) and an
  unsigned MSIX for Microsoft Store submission (a workflow artifact), both built from
  the frozen `win32-x64` binary.
- A real SVG logo, with the icon and Store assets generated from it by
  `packaging/windows/msix/make-assets.mjs`.

### Changed
- Large request/response payloads moved out of the `requests` table into a
  `request_blobs` side table, so the analytics scans no longer drag megabytes of body
  text through every query. Migration is automatic on startup.
- Perf and Trends charts are more compact and flatter, 3 per row.

### Fixed
- **Proxy timeouts under a growing database.** The analytics endpoints (`stats`,
  `suggestions`, `system_history`, `audit`, conversation list/detail) ran blocking
  SQLite queries directly on the asyncio event loop; as the DB grew, a single slow
  query (hundreds of ms to ~1 s) stalled the loop and in-flight request proxying,
  causing timeouts. Those handlers now run in Starlette's threadpool.
- Conversations list took ~2 s to load; it now uses a metadata-only endpoint and a
  dedicated page for the full transcript.
- Streaming requests are finalized when the client disconnects, instead of being left
  as permanently-pending zombie rows.
- The loop detector resets on user intervention (so a new instruction isn't blocked by
  the previous loop) and now explains why it blocked.
- Token counts, context discovery, and large-body rendering in the request view.
- Request images are stored separately so large ones aren't truncated, and the
  post-split migration no longer re-adds an `images_data` column that would collide
  with the `requests_v` view.
- Artifacts: Windows paths are case-normalized and drive spellings canonicalized so the
  same file doesn't appear as several rows; regex-fragment noise from terminal commands
  is dropped; the list query no longer scans full bodies.
- Perf charts no longer emit an invalid SVG path when a series is entirely null.
- Diagnostics overlay, over-polling, and uptime reporting.

### Removed
- **The VS Code editor bridge** — live editor mirror, phone PWA, agent tasks, and
  tool-approval flow — along with its backend routes, globals, frontend UI, and
  `static/remote.html`.

## [0.1.0]

Initial packaged release.

### Added
- Installable Python package (`guscatalano-ai-proxy`) with an `ai-proxy` console
  script; `pip`/`pipx` install and `python -m ai_proxy`.
- npm distribution as a self-contained binary (no Python required), built with
  PyInstaller and shipped via platform-gated optional-dependency packages.
- Release CI: on a `vX.Y.Z` tag, builds one binary per platform (win/mac x64+arm64,
  linux x64+arm64), publishes all npm packages and the PyPI sdist+wheel, and attaches
  binaries to a GitHub Release.
- Per-user default location for writable state (DB, rules, generated images) when
  installed, overridable via `PROXY_STATE_DIR` / `PROXY_DB` / `PROXY_RULES_FILE`.
- `version` field in `GET /__proxy/api/info`.

[Unreleased]: https://github.com/guscatalano/AI_Proxy/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/guscatalano/AI_Proxy/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/guscatalano/AI_Proxy/releases/tag/v0.1.0
