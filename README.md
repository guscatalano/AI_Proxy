# AI Proxy - Setup Guide

## Quick Start

### Prerequisites
- Python 3.10+
- Ollama running on `http://localhost:11434`

### Installation

```bash
# Clone or navigate to the project directory
cd ai_proxy

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment example
cp .env.example .env

# Edit .env with your settings
```

### Running

```bash
# Start the proxy server
python proxy.py

# Or use uvicorn directly
uvicorn proxy:app --host 127.0.0.1 --port 8000
```

### Access
- **Proxy API**: `http://127.0.0.1:8000/`
- **Web UI**: `http://127.0.0.1:8000/__proxy/`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Upstream Ollama server |
| `PROXY_HOST` | `127.0.0.1` | Proxy bind address |
| `PROXY_PORT` | `8000` | Proxy port |
| `PROXY_DB` | `./proxy.db` | SQLite database path |
| `PROXY_RULES_FILE` | `./rules.json` | Rules configuration file |

### Rules Configuration

Create `rules.json` to define traffic rules:

```json
{
  "loop_detector": {
    "enabled": true,
    "action": "block",
    "max_repeats": 4,
    "window": 10,
    "tail_consecutive": 3
  },
  "model_router": {
    "enabled": false,
    "aliases": {},
    "rules": []
  }
}
```

## Deployment

### Linux (systemd)

```bash
# Edit the service file with your paths
sudo cp ai_proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai_proxy
```

### Windows (PowerShell)

```powershell
# Run as Administrator
New-Service -Name "AIProxy" `
  -BinaryPathName "C:\path\to\python.exe C:\path\to\proxy.py" `
  -DisplayName "AI Proxy" `
  -StartupType Automatic
```

## Web UI Features

- **Requests Tab**: View all intercepted API calls
- **Stats Tab**: Token usage, response times, model distribution
- **Audit Tab**: Block/warn/rewrite verdicts
- **Conversations Tab**: Grouped request history
- **Rules Editor**: Configure traffic rules
- **Suggestions Tab**: AI-powered optimization recommendations

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/__proxy/api/info` | GET | Proxy configuration |
| `/__proxy/api/requests` | GET | List requests |
| `/__proxy/api/requests/{id}` | GET | Get request details |
| `/__proxy/api/stats` | GET | Statistics |
| `/__proxy/api/audit` | GET | Audit log |
| `/__proxy/api/conversations` | GET | List conversations |
| `/__proxy/api/conversations/{id}` | GET | Conversation details |
| `/__proxy/api/rules` | GET/POST | Rules configuration |
| `/__proxy/api/suggestions` | GET | Optimization suggestions |
| `/__proxy/api/clear` | POST | Clear request history |

## Security Notes

- The proxy stores full request/response data in SQLite
- No authentication is built-in (add reverse proxy if needed)
- Bind to `127.0.0.1` for local-only access
- Consider rate limiting for production use

## Troubleshooting

### Proxy not starting
- Check Ollama is running: `curl http://localhost:11434`
- Verify port is not in use: `netstat -ano | findstr :8000`

### Database errors
- Delete `proxy.db` and restart (creates fresh database)

### Rules not applying
- Check `PROXY_RULES_FILE` environment variable
- Verify JSON syntax in rules file
