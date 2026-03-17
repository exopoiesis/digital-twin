#!/bin/bash
# onstart_template.sh — onstart bootstrap for Vast.ai instances
#
# This script is meant to be pasted as --onstart-cmd when creating an instance.
# It clones the tools repo from GitHub, deploys scripts, and starts monitoring.
#
# TWO WAYS TO USE:
#
# 1) Inline (recommended) — paste the bootstrap one-liner:
#    vastai create instance <offer_id> --image exopoiesis/infra-gpaw-gpu --disk 20 --ssh \
#      --label "my-worker" \
#      --onstart-cmd 'export TZ=Europe/Kyiv; git clone --depth 1 https://github.com/exopoiesis/digital-twin.git /workspace/digital-twin && cp /workspace/digital-twin/tools/*.sh /workspace/digital-twin/tools/*.py /workspace/ 2>/dev/null; chmod +x /workspace/*.sh; mkdir -p /workspace/results; nohup bash /workspace/vast_monitor.sh >/dev/null 2>&1 & disown'
#
# 2) Full script — from local machine with the repo cloned:
#    vastai create instance <offer_id> ... \
#      --onstart-cmd "$(cat tools/onstart_template.sh)"
#
# Environment variables for auto-start (set via --env):
#   AUTOSTART=yes  — auto-launch GPAW (default: no)
#   WORKER_ID=0    — worker ID for parallel runs
#   NUM_WORKERS=4  — total workers
#   MINERAL=pentlandite — mineral name
#   MAIN_SCRIPT=generate_sulfide_dft_data_v2.py — python script to run
#   OUTPUT=/workspace/results/sulfide_train.xyz — output path

set -eo pipefail
export TZ=Europe/Kyiv

RESULTS_DIR="/workspace/results"
REPO_URL="https://github.com/exopoiesis/digital-twin.git"
REPO_DIR="/workspace/digital-twin"
TOOLS_DIR="$REPO_DIR/tools"
LOG="/var/log/onstart.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

mkdir -p "$RESULTS_DIR"

# === 1. Clone tools from GitHub ===
log "Fetching tools from $REPO_URL..."

if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR" && git pull 2>&1 | tee -a "$LOG"
    cd /workspace
else
    git clone --depth 1 "$REPO_URL" "$REPO_DIR" 2>&1 | tee -a "$LOG"
fi

if [ ! -d "$TOOLS_DIR" ]; then
    log "ERROR: $TOOLS_DIR not found. Waiting for manual upload."
else
    cp "$TOOLS_DIR"/*.sh /workspace/ 2>/dev/null || true
    cp "$TOOLS_DIR"/*.py /workspace/ 2>/dev/null || true
    chmod +x /workspace/*.sh
    log "Deployed $(ls /workspace/*.sh /workspace/*.py 2>/dev/null | wc -l) files"
fi

# === 2. Start monitor daemon ===
if [ -f /workspace/vast_monitor.sh ]; then
    nohup bash /workspace/vast_monitor.sh > /dev/null 2>&1 & disown
    log "Monitor started (PID $!)"
fi

# === 3. Auto-start GPAW if requested ===
AUTOSTART="${AUTOSTART:-no}"
if [ "$AUTOSTART" = "yes" ]; then
    WORKER_ID="${WORKER_ID:-0}"
    NUM_WORKERS="${NUM_WORKERS:-1}"
    MINERAL="${MINERAL:-pentlandite}"
    OUTPUT="${OUTPUT:-/workspace/results/sulfide_train.xyz}"
    MAIN_SCRIPT="${MAIN_SCRIPT:-generate_sulfide_dft_data_v2.py}"

    if [ ! -f "/workspace/$MAIN_SCRIPT" ]; then
        log "ERROR: /workspace/$MAIN_SCRIPT not found."
    else
        log "Auto-starting: $MAIN_SCRIPT (w${WORKER_ID}/${NUM_WORKERS}, $MINERAL)"
        nohup bash /workspace/vast_launch.sh "$MAIN_SCRIPT" \
            --output "$OUTPUT" --resume \
            --mineral "$MINERAL" \
            --worker-id "$WORKER_ID" --num-workers "$NUM_WORKERS" \
            > /dev/null 2>&1 & disown
        log "GPAW launched (PID $!)"
    fi
else
    log "AUTOSTART=no. Run: bash /workspace/vast_launch.sh <script.py> [args...]"
fi

log "Onstart complete."
