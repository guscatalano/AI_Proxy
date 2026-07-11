# guscatalano-ai-proxy

A transparent inspector and rule engine that sits between your AI clients (Claude Code,
GitHub Copilot Chat, Cursor, raw SDKs) and their upstreams (Anthropic, Ollama, LM Studio,
anything OpenAI-compatible). Logs every request, surfaces conversations and tool calls,
applies configurable rules, and gives you a live web UI to see what's actually happening.

This npm package ships a **self-contained binary** — Python is **not** required.

## Install

```bash
npm install -g guscatalano-ai-proxy
ai-proxy
```

Or run without installing:

```bash
npx guscatalano-ai-proxy
```

Then open the UI at <http://127.0.0.1:8000/__proxy/>.

## Point your client at the proxy

```bash
# Claude Code (and any Anthropic SDK client)
export ANTHROPIC_BASE_URL=http://localhost:8000
claude

# OpenAI SDK / VS Code Copilot Chat / Cursor / Continue / Cline
export OPENAI_BASE_URL=http://localhost:8000/v1
```

## Configuration

Configured entirely via environment variables — for example `PROXY_PORT`, `PROXY_HOST`,
`OLLAMA_URL`, `ANTHROPIC_URL`, `PROXY_DB`, and `PROXY_STATE_DIR` (where the SQLite DB and
runtime state are written; defaults to a per-user directory). See the
[full documentation](https://github.com/guscatalano/AI_Proxy#configuration).

## How it's packaged

The heavy lifting is a Python app frozen into a single executable per platform with
PyInstaller. Those binaries are published as platform-gated optional dependencies
(`guscatalano-ai-proxy-<platform>-<arch>`); npm installs only the one matching your
machine, and this package's `ai-proxy` launcher execs it.

Supported platforms: `win32-x64`, `darwin-x64`, `darwin-arm64`, `linux-x64`, `linux-arm64`.

Prefer Python? `pipx install guscatalano-ai-proxy`.

## License

MIT © Gus Catalano
