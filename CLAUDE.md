# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[dev]"
pytest                                   # coverage is in addopts; runs offline, no upstream needed
pytest tests/test_bench_security.py -q -o addopts=""     # one file; drops the --cov flags, which
pytest tests/test_bench_security.py::test_name -q -o addopts=""   # fail outright without pytest-cov
shellcheck deploy/install-service.sh     # CI gates on this

ai-proxy                                 # run from a source checkout (or python -m ai_proxy)

# UI smoke test (headless Chromium) — needs a running proxy
cd tests/ui && npm install && npx playwright install --with-deps chromium
UI_URL="http://127.0.0.1:8000/__proxy/" node tests/ui/smoke.mjs

python scripts/bump_version.py minor --commit   # version is single-sourced; test_version.py guards it
```

`tests/conftest.py` sets `PROXY_STATE_DIR` / `PROXY_DB` to a temp dir **before importing the app**, because paths are resolved at import time. Any new module-level path resolution must respect those env vars or tests will write to the real DB.

The full suite runs in about two minutes. If it appears to hang for far longer, suspect the runner rather than the tests — backgrounded pytest runs in this repo have twice died silently mid-run, both times leaving a partial log that reads like a stuck test. Re-run the suspect file in the foreground before believing it.

There is a second, different hang, and the two look identical from the outside: **every test passes and then the process never exits.** Something in teardown keeps the interpreter alive, so the summary line is never printed — a backgrounded run reports exit code 0 with a log frozen partway through, and a foreground run appears to stall forever. Do not go hunting for the test that "stopped" it; there isn't one. Confirm with `-v` and count the results instead, which needs no summary line:

```bash
python -m pytest -o addopts="" -p no:randomly -v > /tmp/full.txt 2>&1   # will not exit; kill it once 100% is reached
grep -c PASSED /tmp/full.txt; grep -c FAILED /tmp/full.txt
```

Last measured this way: 1158 passed, 0 failed, 19 skipped.

## Deploying to production (spark)

Production is the GB10 box `spark`: systemd unit `ai_proxy` on `:11444`, package at `/home/crimson/ai_proxy/ai_proxy/` behind a launcher shim (`proxy.py`) that imports the package beside it.

**Deploy only via `bash scripts/deploy_spark.sh`.** It tar-pipes the whole package, clears `__pycache__`, and restarts. Copying individual files has silently split the deployment before. `--no-restart` stages without bouncing the service.

A restart kills in-flight agent sessions. When the box is busy, stage the code and arm `scripts/arm_deferred_deploy.sh`, which installs a cron running `scripts/deferred_deploy.sh` on the box: it waits for a quiet window (`scripts/idle_check.py`), backs up, rsyncs `--checksum`, health-checks `/__proxy/api/stats`, rolls back on failure, and disarms itself. Its comments record specific failures that guard clauses exist for — read them before editing.

`proxy.db` in production is multi-gigabyte. Any ad-hoc query must be bounded (time window + `LIMIT`), and `VACUUM` needs a maintenance window.

## Architecture

`README.md` documents the rule pipeline, every env var, the API surface and the routing modes — consult it rather than re-deriving from code. What follows is the structure that isn't visible from any single file.

### The core module

`ai_proxy/proxy.py` is ~17.7k lines: FastAPI + httpx + sqlite3, no heavy deps. Navigate it by its section banners (`grep -n '^# ---' ai_proxy/proxy.py`) — DB schema, system metrics collector, backend registry, rule engine, model quirks, artifact extraction, context compressor, web UI + management API, task queue, benchmark API, benchmark runner, graded task suite, auto-load, transparent proxy, protocol bridge, proxy-owned tools, export/digest, MCP server.

Management routes are registered **before** the catch-all proxy route, so ordering in the file is load-bearing.

`ai_proxy/static/index.html` is the entire dashboard: vanilla JS, no build step, no framework. Edit it directly.

### The bench subsystem and its facade

The benchmark code was split out of `proxy.py` into sibling modules, and `proxy.py` re-binds their names at module level:

- `bench_suites.py` — `SUITES` dict of graded coding tasks
- `bench_graders.py` — code extraction, per-language execution graders, and the non-executable grading modes (`text`, `format`, `refusal`, `answer`, `langpick`)
- `bench_agent.py` — multi-turn agent episodes, the tool dispatcher, and `grade_episode`
- `bench_security.py`, `bench_instruct.py`, `bench_memory.py`, `bench_langpref.py` — task banks
- `bench_report.py` — the HTML/SVG whitepaper renderer (pure functions, importable standalone)

Three seams matter when adding anything:

1. **The re-binding blocks** (`_bench_report_mod.X = ...` style assignments around lines 9900 and 11310). A new public function in a bench module does not exist under `proxy.` until it's re-bound there. Tests and older call sites reference the `proxy.`-level names.
2. **The suite registry** (`_BENCH_SUITES`, ~line 11209). Suites are registered with `setdefault`, and `full-v1`/`full-v2` are built by concatenating other suites — adding a suite to the aggregate is a separate edit from defining it.
3. **`_bench_lang_available` in `bench_graders.py`.** A task whose `lang` isn't listed there is *silently skipped*, not failed. Two whole suites once vanished from a full run this way. Any new grading mode must be added to that gate.

Report-side metadata flows the other direction: `proxy.py` populates `bench_report.TASK_CATEGORY` / `TASK_SIDE` / `TASK_REQUESTED_LANG` at import (~line 11305), because the renderer must stay importable without the runner.

Grading **executes model-generated code** in a separate process with `-I`, a scratch cwd, a stripped env and a hard timeout. That contains accidents, not hostile code.

### Storage

SQLite in WAL mode. `SCHEMA` holds `CREATE TABLE`/`CREATE INDEX ... IF NOT EXISTS`; `MIGRATIONS` is an **append-only** list of `ALTER TABLE ... ADD COLUMN` statements executed with failures ignored. Never reorder or delete entries.

Data repairs (re-parsing old rows after a parser improvement) go in `init_db` as a `backfill_vN` block guarded by a row in `settings`, so they run exactly once per database. Large payloads live in the `request_blobs` side table keyed by request id — analytics queries must not join it unless they need bodies.

Conversations are grouped by hashing `system + first_user`; volatile per-request content is stripped in `_normalize_for_cid`, which is where to extend when a client breaks grouping.

## Conventions

Comments in this codebase explain *why*, usually by naming the incident that motivated the code — the fail-open guard in `deferred_deploy.sh`, the `--checksum` flag, the standalone `__version__` fallback. Match that register: no comments restating what the line does, but keep the reason a non-obvious guard exists.
