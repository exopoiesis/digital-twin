#!/bin/bash
# vast_watchdog.sh -- restart wrapper for Vast.ai scripts
# Restarts on signal kills (SIGTERM=143, SIGKILL=137).
# Stops on: exit 0, DONE file, or script errors (exit 1/2).
#
# Usage:
#   nohup bash /workspace/vast_watchdog.sh /workspace/vast_launch.sh script.py [args...] &
#   nohup bash /workspace/vast_watchdog.sh /workspace/launch_q075.sh &
#
# Environment:
#   MAX_RETRIES  -- max restart attempts (default 20)
#   RETRY_DELAY  -- seconds between retries (default 30)
#   DONE_FILE    -- path to completion marker (default /workspace/results/DONE)

set -uo pipefail
export TZ=Europe/Kyiv

MAX_RETRIES=${MAX_RETRIES:-20}
RETRY_DELAY=${RETRY_DELAY:-30}
DONE_FILE=${DONE_FILE:-/workspace/results/DONE}
WATCHDOG_LOG="/workspace/watchdog.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [watchdog] $1" | tee -a "$WATCHDOG_LOG"
}

log "Starting: $*"
log "MAX_RETRIES=$MAX_RETRIES, RETRY_DELAY=${RETRY_DELAY}s"

for attempt in $(seq 1 $MAX_RETRIES); do
    if [ -f "$DONE_FILE" ]; then
        log "DONE file found, exiting"
        exit 0
    fi

    log "Attempt $attempt/$MAX_RETRIES"

    bash "$@"
    EXIT_CODE=$?

    log "Exit code: $EXIT_CODE"

    if [ $EXIT_CODE -eq 0 ]; then
        log "Clean exit"
        exit 0
    fi

    if [ -f "$DONE_FILE" ]; then
        log "DONE file created, exiting"
        exit 0
    fi

    # Signal kills = transient (Vast.ai migration/preemption), retry
    if [ $EXIT_CODE -eq 137 ] || [ $EXIT_CODE -eq 143 ] || [ $EXIT_CODE -eq 130 ]; then
        log "Signal kill (exit $EXIT_CODE), retrying in ${RETRY_DELAY}s..."
        sleep "$RETRY_DELAY"
        continue
    fi

    # Script errors = bug, don't retry
    log "Script error (exit $EXIT_CODE), NOT retrying"
    exit $EXIT_CODE
done

log "Max retries ($MAX_RETRIES) reached"
exit 1
