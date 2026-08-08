#!/usr/bin/env bash
# Deploy staged code the next time the proxy is quiet — runs ON spark, unattended.
#
# The normal path is scripts/deploy_spark.sh from a workstation, which restarts the service
# immediately. That is fine in a gap between requests and wrong while an agent is mid-session:
# a restart kills whatever is in flight. When the box is busy for hours there is no gap to
# wait for interactively, so the wait moves to the box: this checks every two minutes, and
# the first time nothing has called the proxy for QUIET_S it swaps the staged package in.
#
# Safety, because nobody is watching it:
#   - the current package is copied aside before anything moves
#   - the service must come back AND answer HTTP, or the backup goes straight back
#   - it deploys at most once, then removes its own cron entry
#   - a bench in progress counts as busy, however quiet the request log looks
#
# Installed by scripts/arm_deferred_deploy.sh; log at /home/crimson/deferred_deploy.log
set -uo pipefail

# Overridable so the swap/rollback path can be rehearsed against throwaway directories
# before it is trusted to run unattended against production. Defaults are production.
ROOT=${ROOT:-/home/crimson/ai_proxy}
STAGED=${STAGED:-/home/crimson/ai_proxy_staged/ai_proxy}
BACKUP=${BACKUP:-/home/crimson/ai_proxy_backup}
LOG=${LOG:-/home/crimson/deferred_deploy.log}
DEADLINE_FILE=${DEADLINE_FILE:-/home/crimson/deferred_deploy.deadline}
QUIET_S=${QUIET_S:-1800}
RESTART_CMD=${RESTART_CMD:-"sudo systemctl restart ai_proxy"}
HEALTH_URL=${HEALTH_URL:-http://localhost:11444/__proxy/api/version}
IS_ACTIVE_CMD=${IS_ACTIVE_CMD:-"systemctl is-active --quiet ai_proxy"}
IDLE_CMD=${IDLE_CMD:-"python3 /home/crimson/idle_check.py"}
SELF_DISARM=${SELF_DISARM:-1}

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

[ -d "$STAGED" ] || { log "no staged package at $STAGED — nothing to do"; exit 0; }

# Give up after the deadline rather than deploying into a week-old window.
if [ -f "$DEADLINE_FILE" ]; then
  deadline=$(cat "$DEADLINE_FILE" 2>/dev/null || echo 0)
  now=$(date +%s)
  if [ "$now" -gt "$deadline" ]; then
    log "deadline passed with no quiet window — disarming, nothing deployed"
    crontab -l 2>/dev/null | grep -v deferred_deploy.sh | crontab - || true
    exit 0
  fi
fi

read -r idle bench < <($IDLE_CMD 2>/dev/null || echo "0 1")
if [ "${bench:-1}" != "0" ]; then log "bench running — waiting"; exit 0; fi
if [ "${idle:-0}" -lt "$QUIET_S" ]; then exit 0; fi   # silent: this is the common case

log "quiet for ${idle}s — deploying"
rm -rf "$BACKUP"; mkdir -p "$BACKUP"
cp -a "$ROOT/ai_proxy" "$BACKUP/" || { log "backup failed — aborting, nothing changed"; exit 1; }

# --checksum, not rsync's default size+mtime quick-check: a staged file that happens to
# match production's size and timestamp would be skipped silently, and a deploy that copies
# most of the package is the failure mode that cost an evening once already. The package is
# small; hashing it is free next to being wrong.
rsync -a --checksum --delete "$STAGED/" "$ROOT/ai_proxy/" \
  || { log "rsync failed — restoring"; rm -rf "$ROOT/ai_proxy"; cp -a "$BACKUP/ai_proxy" "$ROOT/"; exit 1; }
find "$ROOT/ai_proxy" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null

$RESTART_CMD
sleep 8
ok=0
for _ in 1 2 3 4 5 6; do
  if $IS_ACTIVE_CMD && curl -fsS -m 5 -o /dev/null "$HEALTH_URL"; then
    ok=1; break
  fi
  sleep 5
done

if [ "$ok" = "1" ]; then
  log "deployed and healthy"
  rm -rf "$STAGED"
else
  log "service did not come back — ROLLING BACK"
  rm -rf "$ROOT/ai_proxy"
  cp -a "$BACKUP/ai_proxy" "$ROOT/"
  $RESTART_CMD
  sleep 5
  $IS_ACTIVE_CMD && log "rollback restored the service" \
    || log "ROLLBACK FAILED — service is down, needs a human"
fi
# One shot either way: a script that keeps trying after a rollback just keeps breaking things.
if [ "$SELF_DISARM" = "1" ]; then
  crontab -l 2>/dev/null | grep -v deferred_deploy.sh | crontab - || true
  log "disarmed"
fi
