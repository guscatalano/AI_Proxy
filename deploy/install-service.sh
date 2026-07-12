#!/usr/bin/env bash
# Install AI Proxy as a systemd service.
#
#   System (default, boots at startup, runs as a dedicated 'ai-proxy' user):
#       sudo ./deploy/install-service.sh
#
#   Per-user (no root; add linger for boot persistence):
#       ./deploy/install-service.sh --user
#
# By default it installs the package from this checkout. Use --pypi to install the
# published release instead. Re-running is safe (it reinstalls and restarts).
set -euo pipefail

MODE=system
SOURCE=auto   # auto | source | pypi

usage() { sed -n '2,12p' "$0"; exit "${1:-0}"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --user) MODE=user ;;
    --system) MODE=system ;;
    --pypi) SOURCE=pypi ;;
    --source) SOURCE=source ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PKG_NAME="guscatalano-ai-proxy"

command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }

# Decide what to install: this checkout, or the published package.
if [ "$SOURCE" = pypi ]; then
  TARGET="$PKG_NAME"
elif [ "$SOURCE" = source ] || [ -f "$REPO_ROOT/pyproject.toml" ]; then
  TARGET="$REPO_ROOT"
else
  TARGET="$PKG_NAME"
fi
echo "Installing from: $TARGET"

render_unit() {  # <template> <exec> <envfile> [user]
  sed -e "s|@EXECSTART@|$2|g" -e "s|@ENVFILE@|$3|g" -e "s|@USER@|${4:-}|g" "$1"
}

if [ "$MODE" = system ]; then
  [ "$(id -u)" = 0 ] || { echo "system mode needs root; re-run with: sudo $0" >&2; exit 1; }

  PREFIX=/opt/ai-proxy
  ENVFILE=/etc/ai-proxy/ai-proxy.env
  SERVICE_USER=ai-proxy
  STATE_DIR=/var/lib/ai-proxy

  echo "Creating venv at $PREFIX"
  python3 -m venv "$PREFIX"
  "$PREFIX/bin/pip" install --quiet --upgrade pip
  "$PREFIX/bin/pip" install --quiet "$TARGET"
  EXEC="$PREFIX/bin/ai-proxy"

  id "$SERVICE_USER" >/dev/null 2>&1 || \
    useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"

  mkdir -p /etc/ai-proxy
  if [ ! -f "$ENVFILE" ]; then
    install -m 640 "$SCRIPT_DIR/ai-proxy.env.example" "$ENVFILE"
  fi
  grep -q '^PROXY_STATE_DIR=' "$ENVFILE" || echo "PROXY_STATE_DIR=$STATE_DIR" >> "$ENVFILE"

  render_unit "$SCRIPT_DIR/ai-proxy.service.in" "$EXEC" "$ENVFILE" "$SERVICE_USER" \
    > /etc/systemd/system/ai-proxy.service

  systemctl daemon-reload
  systemctl enable --now ai-proxy
  echo
  echo "Installed. Config: $ENVFILE   (edit, then: sudo systemctl restart ai-proxy)"
  systemctl --no-pager --lines=0 status ai-proxy || true

else
  PREFIX="${XDG_DATA_HOME:-$HOME/.local/share}/ai-proxy/venv"
  ENVFILE="${XDG_CONFIG_HOME:-$HOME/.config}/ai-proxy/ai-proxy.env"
  UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

  echo "Creating venv at $PREFIX"
  python3 -m venv "$PREFIX"
  "$PREFIX/bin/pip" install --quiet --upgrade pip
  "$PREFIX/bin/pip" install --quiet "$TARGET"
  EXEC="$PREFIX/bin/ai-proxy"

  mkdir -p "$(dirname "$ENVFILE")" "$UNIT_DIR"
  [ -f "$ENVFILE" ] || cp "$SCRIPT_DIR/ai-proxy.env.example" "$ENVFILE"

  render_unit "$SCRIPT_DIR/ai-proxy.user.service.in" "$EXEC" "$ENVFILE" \
    > "$UNIT_DIR/ai-proxy.service"

  systemctl --user daemon-reload
  systemctl --user enable --now ai-proxy
  echo
  echo "Installed (user service). Config: $ENVFILE"
  echo "For boot persistence (survive logout/reboot):  loginctl enable-linger \"$USER\""
  systemctl --user --no-pager --lines=0 status ai-proxy || true
fi
