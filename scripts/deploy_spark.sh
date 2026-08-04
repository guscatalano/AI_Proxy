#!/usr/bin/env bash
# Deploy the ai_proxy package to spark and restart the service.
#
# The only deploy path. Production launches through a tiny shim at
# /home/crimson/ai_proxy/proxy.py that imports the package beside it, so the package
# directory is the single source of truth — the era of "scp one file to two places and
# hope" ended with the evening a whole night of fixes silently never ran.
set -euo pipefail
cd "$(dirname "$0")/.."

tar -cf - --exclude='__pycache__' --exclude='*.pyc' ai_proxy \
  | ssh spark 'tar -C /home/crimson/ai_proxy --overwrite -xf - \
      && find /home/crimson/ai_proxy/ai_proxy -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; true'

if [ "${1:-}" != "--no-restart" ]; then
  ssh spark 'sudo systemctl restart ai_proxy && sleep 5 && systemctl is-active ai_proxy'
fi
echo "deployed$( [ "${1:-}" = "--no-restart" ] && echo ' (no restart)' )"
