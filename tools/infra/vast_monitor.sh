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

# Check if main compute process is alive (python3, jdftx, abacus, mpirun)
check_process() {
    local pid info name
    # Try each known compute binary
    for pat in "python3" "jdftx" "abacus_2p" "mpirun"; do
        pid=$(pgrep -x "$pat" 2>/dev/null | head -1 || true)
        if [ -n "$pid" ]; then
            # Skip zombies
            local state
            state=$(ps -p "$pid" -o stat --no-headers 2>/dev/null | head -c1)
            [ "$state" = "Z" ] && continue
            info=$(ps -p "$pid" -o pid,pcpu,rss,vsz,etime --no-headers 2>/dev/null || echo "N/A")
            log "$pat alive: PID=$pid $info"
            return 0
        fi
    done
    # Nothing running — check DONE
    log "!!! NO COMPUTE PROCESS RUNNING !!!"
    # Only report DONE if it's fresh (matches current script)
    local done_file
    done_file=$(ls -t /workspace/results/DONE* 2>/dev/null | head -1)
    if [ -n "$done_file" ]; then
        local done_age done_content
        done_age=$(( $(date +%s) - $(stat -c%Y "$done_file" 2>/dev/null || echo 0) ))
        done_content=$(head -1 "$done_file" 2>/dev/null)
        if [ "$done_age" -lt 86400 ]; then
            log "DONE (<24h old): $done_content"
        else
            log "DONE file STALE (${done_age}s old, from previous run): $done_content"
        fi
    else
        log "NO DONE file — script crashed or was killed!"
        local last_log
        last_log=$(tail -3 /workspace/run.log 2>/dev/null || echo "no run.log")
        log "Last run.log lines: $last_log"
    fi
    return 1
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

    # ETA estimation from FIRE/NEB/LCAOMinimize logs
    estimate_eta
}

# ETA estimation — parse FIRE/LCAOMinimize logs for step timing and convergence
estimate_eta() {
    local logfiles fire_log lcao_log neb_log

    # Find active log files
    fire_log=$(ls -t /workspace/neb_work/neb.log /workspace/neb_work/relax_start.log /workspace/neb_work/relax_end.log 2>/dev/null | head -1)
    lcao_log=$(find /workspace/results -name "*.out" -newer /workspace/results/heartbeat 2>/dev/null | head -1)
    [ -z "$lcao_log" ] && lcao_log=$(ls -t /workspace/results/*candle*/*.out /workspace/results/*.out 2>/dev/null | head -1)

    # FIRE-based ETA (ABACUS NEB / endpoint relax)
    if [ -n "$fire_log" ] && [ -f "$fire_log" ]; then
        local lines last_two n_steps fmax
        lines=$(grep "^FIRE:" "$fire_log" 2>/dev/null | tail -5)
        if [ -n "$lines" ]; then
            n_steps=$(echo "$lines" | wc -l)
            local total_steps
            total_steps=$(grep -c "^FIRE:" "$fire_log" 2>/dev/null || echo 0)
            # Get last fmax
            fmax=$(echo "$lines" | tail -1 | awk '{print $NF}')
            # Get time per step from last 2 steps
            local t1 t2 dt_sec
            last_two=$(echo "$lines" | tail -2)
            t1=$(echo "$last_two" | head -1 | awk '{print $3}')
            t2=$(echo "$last_two" | tail -1 | awk '{print $3}')
            if [ -n "$t1" ] && [ -n "$t2" ] && [ "$t1" != "$t2" ]; then
                # Convert HH:MM:SS to seconds
                local s1 s2
                s1=$(echo "$t1" | awk -F: '{print $1*3600+$2*60+$3}')
                s2=$(echo "$t2" | awk -F: '{print $1*3600+$2*60+$3}')
                dt_sec=$(( s2 - s1 ))
                [ "$dt_sec" -lt 0 ] && dt_sec=$(( dt_sec + 86400 ))  # midnight wrap
                # Estimate remaining steps: fmax decays ~exponentially
                # Rough: if fmax halves every ~5 steps, steps_left ~ 5*log2(fmax/0.05)
                local steps_left eta_hours
                if command -v python3 &>/dev/null; then
                    steps_left=$(python3 -c "
import math
fmax=${fmax}; target=0.05
if fmax <= target: print(0)
else: print(int(math.log(fmax/target)/math.log(2)*5))
" 2>/dev/null || echo "?")
                else
                    steps_left="?"
                fi
                if [ "$steps_left" != "?" ] && [ "$steps_left" -gt 0 ] 2>/dev/null; then
                    eta_hours=$(python3 -c "print(f'{${steps_left}*${dt_sec}/3600:.1f}')" 2>/dev/null || echo "?")
                    log "ETA: FIRE step ${total_steps}, fmax=${fmax}, ~${dt_sec}s/step, ~${steps_left} steps left, ~${eta_hours}h"
                else
                    log "ETA: FIRE step ${total_steps}, fmax=${fmax}, ~${dt_sec}s/step"
                fi
            else
                log "ETA: FIRE step ${total_steps}, fmax=${fmax} (need >=2 steps for timing)"
            fi
        fi
    fi

    # LCAOMinimize-based ETA (JDFTx)
    if [ -n "$lcao_log" ] && [ -f "$lcao_log" ]; then
        local lcao_lines
        lcao_lines=$(grep "LCAOMinimize: Iter:" "$lcao_log" 2>/dev/null | tail -5)
        if [ -n "$lcao_lines" ]; then
            local iter grad
            iter=$(echo "$lcao_lines" | tail -1 | awk -F'Iter:' '{print $2}' | awk '{print $1}')
            grad=$(echo "$lcao_lines" | tail -1 | sed 's/.*|grad|_K:\s*//' | awk '{print $1}')
            # Time per iter from t[s] field
            local last_two_lcao t_s1 t_s2 dt_lcao
            last_two_lcao=$(echo "$lcao_lines" | tail -2)
            t_s1=$(echo "$last_two_lcao" | head -1 | awk -F't\\[s\\]:' '{print $2}' | awk '{printf "%.0f", $1}')
            t_s2=$(echo "$last_two_lcao" | tail -1 | awk -F't\\[s\\]:' '{print $2}' | awk '{printf "%.0f", $1}')
            if [ -n "$t_s1" ] && [ -n "$t_s2" ]; then
                dt_lcao=$(( t_s2 - t_s1 ))
                # LCAOMinimize typically converges in 30-50 iters, then SCF follows
                local iters_left eta_h
                iters_left=$(( 40 - iter ))
                [ "$iters_left" -lt 5 ] && iters_left=5
                eta_h=$(python3 -c "print(f'{${iters_left}*${dt_lcao}/3600:.1f}')" 2>/dev/null || echo "?")
                log "ETA: LCAOMinimize iter ${iter}, |grad|=${grad}, ~${dt_lcao}s/iter, ~${iters_left} iters left, ~${eta_h}h (LCAO only, SCF follows)"
            else
                log "ETA: LCAOMinimize iter ${iter}, |grad|=${grad}"
            fi
        fi
    fi
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

    if ! check_process; then
        # Compute process died — log full diagnostics and keep monitoring
        check_oom
        log "Waiting for compute process to restart..."
    fi

    check_oom || true
    update_heartbeat

    sleep "$INTERVAL"
done
