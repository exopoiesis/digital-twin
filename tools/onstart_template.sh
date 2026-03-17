#!/bin/bash
# onstart_template.sh — onstart script for Vast.ai template
# This runs automatically when the instance starts.
# Deploys monitoring, downloads scripts, optionally auto-starts GPAW.
#
# To use as --onstart-cmd: paste contents or reference this file.
# Environment variables (set in template or --env):
#   SCRIPT_URL  — URL to download the main python script (optional)
#   AUTOSTART   — "yes" to auto-launch GPAW (default: no, wait for SSH)
#   WORKER_ID   — worker ID for parallel runs
#   NUM_WORKERS — total workers
#   MINERAL     — mineral name (pentlandite, mackinawite, pyrite)
#   OUTPUT      — output XYZ path (default: /workspace/results/sulfide_train.xyz)

set -eo pipefail
export TZ=Europe/Kyiv

RESULTS_DIR="/workspace/results"
MONITOR_SCRIPT="/workspace/vast_monitor.sh"
LAUNCH_SCRIPT="/workspace/vast_launch.sh"
LOG="/var/log/onstart.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

mkdir -p "$RESULTS_DIR"

# === 1. Write monitoring scripts inline (no external download needed) ===
log "Writing monitoring scripts..."

cat > "$MONITOR_SCRIPT" << 'MONITOR_EOF'
#!/bin/bash
# Inline copy of vast_monitor.sh
set -uo pipefail
# no -e: daemon must not exit on grep/dmesg returning non-zero
export TZ=Europe/Kyiv
LOGFILE="/workspace/results/monitor.log"
PIDFILE="/workspace/results/monitor.pid"
HEARTBEAT="/workspace/results/heartbeat"
INTERVAL=60
MAX_LOG_SIZE=10485760

mkdir -p /workspace/results
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then exit 0; fi
echo $$ > "$PIDFILE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOGFILE"; }

rotate_log() {
    if [ -f "$LOGFILE" ] && [ "$(stat -c%s "$LOGFILE" 2>/dev/null || echo 0)" -gt "$MAX_LOG_SIZE" ]; then
        mv "$LOGFILE" "${LOGFILE}.old"
        log "=== Log rotated ==="
    fi
}

check_oom() {
    local oom_lines
    oom_lines=$(dmesg 2>/dev/null | grep -i "oom\|killed process\|out of memory" | tail -5)
    if [ -n "$oom_lines" ]; then
        log "!!! OOM DETECTED !!!"
        log "$oom_lines"
    fi
}

check_python() {
    local py_pid
    py_pid=$(pgrep -f "python3" 2>/dev/null | head -1 || true)
    if [ -z "$py_pid" ]; then
        log "!!! PYTHON3 NOT RUNNING !!!"
        if ls /workspace/results/DONE* 2>/dev/null | head -1 > /dev/null; then
            log "DONE file exists — completed normally."
        else
            log "NO DONE file — crashed or killed!"
            log "Last run.log: $(tail -3 /workspace/run.log 2>/dev/null || echo 'none')"
            log "Last stderr: $(tail -3 /workspace/run_stderr.log 2>/dev/null || echo 'none')"
        fi
    else
        local info
        info=$(ps -p "$py_pid" -o pid,pcpu,rss,etime --no-headers 2>/dev/null || echo "N/A")
        log "python3 alive: $info"
    fi
}

log_metrics() {
    log "$(free -m | awk '/Mem:/{printf "RAM: %dM/%dM (%.0f%%), avail %dM", $3, $2, $3/$2*100, $7}')"
    log "$(free -m | awk '/Swap:/{if($2>0) printf "Swap: %dM/%dM", $3, $2; else print "Swap: none"}')"
    log "$(df -h /workspace 2>/dev/null | awk 'NR==2{printf "Disk: %s/%s (%s)", $3, $2, $5}')"
    log "Load: $(cat /proc/loadavg 2>/dev/null | awk '{print $1, $2, $3}')"
    if command -v nvidia-smi &>/dev/null; then
        log "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | head -1)"
    fi
    local configs
    configs=$(grep -c Properties /workspace/results/*.xyz 2>/dev/null || echo 0)
    log "XYZ configs: $configs"
}

cleanup() { log "Monitor stopping"; rm -f "$PIDFILE"; }
trap cleanup EXIT

log "======= Monitor started (PID $$) ======="

while true; do
    rotate_log
    log "--- tick ---"
    log_metrics
    check_python
    check_oom
    date '+%Y-%m-%d %H:%M:%S' > "$HEARTBEAT"
    sleep "$INTERVAL"
done
MONITOR_EOF

chmod +x "$MONITOR_SCRIPT"
log "Monitor script written."

# === 2. Start monitor daemon ===
log "Starting monitor daemon..."
nohup bash "$MONITOR_SCRIPT" > /dev/null 2>&1 &
log "Monitor started (PID $!)"

# === 3. Auto-start GPAW if requested ===
AUTOSTART="${AUTOSTART:-no}"
if [ "$AUTOSTART" = "yes" ]; then
    WORKER_ID="${WORKER_ID:-0}"
    NUM_WORKERS="${NUM_WORKERS:-1}"
    MINERAL="${MINERAL:-pentlandite}"
    OUTPUT="${OUTPUT:-/workspace/results/sulfide_train.xyz}"
    MAIN_SCRIPT="${MAIN_SCRIPT:-generate_sulfide_dft_data_v2.py}"

    if [ ! -f "/workspace/$MAIN_SCRIPT" ]; then
        log "ERROR: /workspace/$MAIN_SCRIPT not found. Waiting for manual upload."
    else
        log "Auto-starting GPAW: $MAIN_SCRIPT (worker $WORKER_ID/$NUM_WORKERS, mineral=$MINERAL)"
        nohup bash /workspace/vast_launch.sh "$MAIN_SCRIPT" \
            --output "$OUTPUT" --resume \
            --mineral "$MINERAL" \
            --worker-id "$WORKER_ID" --num-workers "$NUM_WORKERS" \
            > /dev/null 2>&1 &
        log "GPAW launched (PID $!)"
    fi
else
    log "AUTOSTART=no. Monitor running. Waiting for manual launch via SSH."
    log "  Usage: bash /workspace/vast_launch.sh <script.py> [args...]"
fi

log "Onstart complete."
