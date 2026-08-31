#!/usr/bin/env bash
# Stage the current package on spark and arm the deferred deploy.
#
# Run from a workstation like scripts/deploy_spark.sh, but instead of restarting now it
# leaves the code staged and a cron entry that deploys during the first quiet window inside
# the deadline. Nothing about the running service changes until that happens.
set -euo pipefail
cd "$(dirname "$0")/.."

HOURS=${1:-24}
QUIET_S=${2:-1800}

tar -cf - --exclude='__pycache__' --exclude='*.pyc' ai_proxy \
  | ssh spark 'rm -rf /home/crimson/ai_proxy_staged && mkdir -p /home/crimson/ai_proxy_staged \
      && tar -C /home/crimson/ai_proxy_staged -xf -'

scp -q scripts/deferred_deploy.sh spark:/home/crimson/deferred_deploy.sh
scp -q scripts/idle_check.py spark:/home/crimson/idle_check.py

ssh spark "chmod +x /home/crimson/deferred_deploy.sh \
  && date -d '+${HOURS} hours' +%s > /home/crimson/deferred_deploy.deadline \
  && ( crontab -l 2>/dev/null | grep -v deferred_deploy.sh; \
       echo '*/2 * * * * QUIET_S=${QUIET_S} /home/crimson/deferred_deploy.sh' ) | crontab - \
  && echo armed: checking every 2 min for a ${QUIET_S}s quiet window, giving up in ${HOURS}h"
