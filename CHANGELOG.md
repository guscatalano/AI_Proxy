# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The version is single-sourced in `ai_proxy/_version.py`. Bump it with
`python scripts/bump_version.py <version|major|minor|patch> --commit --tag`, then push
the tag to trigger the release workflow (publishes to PyPI and npm).

## [Unreleased]

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
