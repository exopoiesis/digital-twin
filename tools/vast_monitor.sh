#!/bin/bash
# vast_monitor.sh — monitoring daemon for Vast.ai instances
# Runs alongside GPAW/python3, logs system metrics every INTERVAL seconds.
# Detects: OOM kills, process death, disk full, swap exhaustion.
#
# Usage (on the instance):
#   nohup bash /workspace/vast_monitor.sh > /dev/null 2>&1 &
#
# Logs: /workspace/results/monitor.log (rotates at 10MB)
# PID:  /workspace/results/monitor.pid

set -uo pipefail
# NOTE: no -e ! Monitor is a long-running daemon, must not exit on grep/dmesg failures
export TZ=Europe/Kyiv

LOGFILE="/workspace/results/monitor.log"
PIDFILE="/workspace/results/monitor.pid"
HEARTBEAT="/workspace/results/heartbeat"
INTERVAL=60  # seconds between checks
MAX_LOG_SIZE=10485760  # 10MB

mkdir -p /workspace/results

# Prevent duplicates
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Monitor already running (PID $(cat "$PIDFILE")). Exiting."
    exit 0
fi
echo $$ > "$PIDFILE"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOGFILE"
}

# Rotate log if too large
rotate_log() {
    if [ -f "$LOGFILE" ] && [ "$(stat -c%s "$LOGFILE" 2>/dev/null || echo 0)" -gt "$MAX_LOG_SIZE" ]; then
        mv "$LOGFILE" "${LOGFILE}.old"
        log "=== Log rotated ==="
    fi
}

# Check for OOM kills in kernel log
check_oom() {
    local oom_lines
    oom_lines=$(dmesg 2>/dev/null | grep -i "oom\|killed process\|out of memory" | tail -5)
    if [ -n "$oom_lines" ]; then
        log "!!! OOM DETECTED !!!"
        log "$oom_lines"
        return 1
    fi
    return 0
}

# Check if main python3 process is alive
check_python() {
    local py_pid py_info
    py_pid=$(pgrep -f "python3.*generate_" 2>/dev/null | head -1 || true)
    if [ -z "$py_pid" ]; then
        # Try broader match
        py_pid=$(pgrep -f "python3" 2>/dev/null | head -1 || true)
    fi
    if [ -z "$py_pid" ]; then
        log "!!! PYTHON3 NOT RUNNING !!! No python3 process found."
        # Check if DONE file exists
        if ls /workspace/results/DONE* 2>/dev/null | head -1 > /dev/null; then
            log "DONE file exists — script completed normally."
        else
            log "NO DONE file — script crashed or was killed!"
            # Try to find exit info
            local last_log
            last_log=$(tail -3 /workspace/run.log 2>/dev/null || echo "no run.log")
            log "Last run.log lines: $last_log"
        fi
        return 1
    else
        py_info=$(ps -p "$py_pid" -o pid,pcpu,rss,vsz,etime --no-headers 2>/dev/null || echo "N/A")
        log "python3 alive: PID=$py_pid $py_info"
        return 0
    fi
}

# System metrics
log_metrics() {
    # Memory
    local mem_info
    mem_info=$(free -m | awk '/Mem:/{printf "RAM: %dM/%dM used (%.0f%%), avail %dM", $3, $2, $3/$2*100, $7}')
    log "$mem_info"

    # Swap
    local swap_info
    swap_info=$(free -m | awk '/Swap:/{if($2>0) printf "Swap: %dM/%dM (%.0f%%)", $3, $2, $3/$2*100; else print "Swap: none"}')
    log "$swap_info"

    # Disk
    local disk_info
    disk_info=$(df -h /workspace 2>/dev/null | awk 'NR==2{printf "Disk: %s/%s (%s used)", $3, $2, $5}')
    log "$disk_info"

    # Load average
    local load
    load=$(cat /proc/loadavg 2>/dev/null | awk '{printf "Load: %s %s %s (procs: %s)", $1, $2, $3, $4}')
    log "$load"

    # GPU (if nvidia-smi available)
    if command -v nvidia-smi &>/dev/null; then
        local gpu_info
        gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$gpu_info" ]; then
            log "GPU: util=${gpu_info%,*,*,*}%, $(echo "$gpu_info" | awk -F', ' '{printf "VRAM %sMiB/%sMiB, temp %s°C", $2, $3, $4}')"
        fi
    fi

    # XYZ progress
    local xyz_lines
    xyz_lines=$(wc -l /workspace/results/*.xyz 2>/dev/null | tail -1 | awk '{print $1}' || echo 0)
    local xyz_configs
    xyz_configs=$(grep -c "Properties" /workspace/results/*.xyz 2>/dev/null | awk -F: '{s+=$2}END{print s+0}' || echo 0)
    log "XYZ: ${xyz_lines} lines, ${xyz_configs} configs"
}

# Update heartbeat file
update_heartbeat() {
    date '+%Y-%m-%d %H:%M:%S' > "$HEARTBEAT"
}

# Cleanup on exit
cleanup() {
    log "Monitor stopping (PID $$)"
    rm -f "$PIDFILE"
}
trap cleanup EXIT

# === MAIN LOOP ===
log "========================================="
log "Monitor started (PID $$, interval ${INTERVAL}s)"
log "Host: $(hostname), Uptime: $(uptime -p 2>/dev/null || uptime)"
log "========================================="

while true; do
    rotate_log
    log "--- tick ---"
    log_metrics

    if ! check_python; then
        # Python died — log full diagnostics and keep monitoring
        # (user may restart the script)
        check_oom
        log "Waiting for python3 to restart..."
    fi

    check_oom || true
    update_heartbeat

    sleep "$INTERVAL"
done
