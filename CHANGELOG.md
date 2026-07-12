# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The version is single-sourced in `ai_proxy/_version.py`. Bump it with
`python scripts/bump_version.py <version|major|minor|patch> --commit` on `main`, then
promote `main` to the `release` branch (`git push origin main:release`) to trigger the
release workflow (publishes to PyPI and npm, and creates a `vX.Y.Z` GitHub Release).

## [Unreleased]

### Fixed
- **Proxy timeouts under a growing database.** The analytics endpoints (`stats`,
  `suggestions`, `system_history`, `audit`, conversation list/detail) ran blocking
  SQLite queries directly on the asyncio event loop; as the DB grew, a single slow
  query (hundreds of ms to ~1 s) stalled the loop and in-flight request proxying,
  causing timeouts. Those handlers now run in Starlette's threadpool.

### Added
- Data-growth guards (env-tunable): `PROXY_MAX_STORED_BODY` caps persisted body size
  (default 256 KB), `PROXY_REQUEST_RETENTION_DAYS` prunes old requests (default 30),
  `PROXY_ANALYTICS_CACHE_TTL` memoizes stats/suggestions. `system_history` is
  downsampled to ≤800 points per response.

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

[Unreleased]: https://github.com/guscatalano/AI_Proxy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/guscatalano/AI_Proxy/releases/tag/v0.1.0
