# AI Proxy Remote Chat (VS Code companion)

A tiny VS Code extension that lets you drive the editor's chat panel (Copilot Chat, Continue, Cursor, etc.) via HTTP. POST a prompt to a port; VS Code opens chat with the prompt pre-filled and submitted.

Useful for:
- Triggering chat from your phone, another machine, a webhook, or a CI script
- Letting an external agent (the AI Proxy's `tool_injector`, Claude Code, etc.) drive your editor
- Routing a "send this to Copilot" action from anywhere on your network

No build step — pure JavaScript, sideload directly.

## Install

1. Drop this folder somewhere stable (e.g. `~/vscode-extensions/ai-proxy-remote-chat/`).
2. Open VS Code → press F1 → `Developer: Install Extension from Location...` → pick that folder.
3. Reload VS Code if prompted.

You should see a `📡 ProxyChat 127.0.0.1:13337` indicator in the status bar.

## Configure

Settings (User or Workspace):

| Setting | Default | Notes |
|---|---|---|
| `aiProxyRemoteChat.port` | `13337` | Port to listen on |
| `aiProxyRemoteChat.bind` | `127.0.0.1` | Use `0.0.0.0` to accept LAN connections |
| `aiProxyRemoteChat.token` | (empty) | Required when bind is non-loopback |
| `aiProxyRemoteChat.defaultCommand` | `workbench.action.chat.open` | The VS Code command that receives the prompt |

Common command alternatives:
- `workbench.action.chat.open` — generic chat (works for Copilot Chat and most others)
- `github.copilot-chat.openChat` — Copilot-specific
- `workbench.action.chat.openInSidebar` — pin chat to the sidebar
- `continue.continueGUIView.focus` — Continue extension

## Use

```bash
# Send a prompt to the default chat
curl -X POST http://localhost:13337/chat \
  -H 'content-type: application/json' \
  -d '{"prompt": "explain what this file does"}'

# With auth and a specific command
curl -X POST http://192.168.6.113:13337/chat \
  -H 'authorization: Bearer YOUR_TOKEN' \
  -H 'content-type: application/json' \
  -d '{"prompt": "find bugs", "command": "github.copilot-chat.openChat"}'

# Run any VS Code command (escape hatch)
curl -X POST http://localhost:13337/command \
  -H 'content-type: application/json' \
  -d '{"id": "workbench.action.files.save"}'

# Health check
curl http://localhost:13337/status
```

## Endpoints

### `POST /chat` (or `/prompt`)
Opens chat with `query` pre-filled and submitted.

Body:
```json
{
  "prompt": "string (required)",
  "command": "workbench.action.chat.open (optional override)",
  "location": "panel | editor | terminal (optional)",
  "isPartialQuery": false,
  "attachScreenshot": false
}
```

### `POST /command`
Runs an arbitrary VS Code command. Power-user escape hatch — use carefully if exposed beyond loopback.

Body:
```json
{ "id": "command.id", "args": [...] }
```

### `GET /status`
Returns the current configuration and VS Code version.

## Wire it up to the AI Proxy

Once the extension is running on your VS Code machine (say `192.168.6.113:13337`), add a proxy-side tool to `PROXY_TOOLS` in `proxy.py` so any model going through the proxy can drive your chat:

```python
{
    "name": "send_to_vscode_chat",
    "description": "Open VS Code's chat panel on the user's editor with the given prompt.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "The message to send to chat"},
        },
        "required": ["prompt"],
    },
}
```

…with a corresponding executor in `_exec_proxy_tool` that POSTs to your VS Code endpoint. Then enable `tool_injector` in the rules editor and the model can call `send_to_vscode_chat({prompt: "..."})` from any conversation.

## Security

- **Default bind is `127.0.0.1`** — only the machine running VS Code can hit it. This is the safe default.
- **For LAN access**, set `bind: "0.0.0.0"` AND set a non-empty `token`. The extension refuses to start a non-loopback listener without a token.
- The `/command` endpoint can run any VS Code command — keep your token strong if exposing this.
- Settings are per-machine; each install needs its own token if you have multiple.

## Troubleshooting

- **Status bar shows error**: another process is on the port, change `aiProxyRemoteChat.port` or kill the conflicting process.
- **`vscode command failed`**: the command id doesn't exist or the chat extension isn't installed. Try `github.copilot-chat.openChat` for Copilot, `continue.continueGUIView.focus` for Continue, or check `Developer: Show Running Extensions` to confirm your chat extension is loaded.
- **Prompt opens chat but doesn't auto-submit**: set `isPartialQuery: false` explicitly. Some chat extensions interpret this differently.
- **LAN request gets 403**: you set `bind: 0.0.0.0` without a token. Set `aiProxyRemoteChat.token`.
