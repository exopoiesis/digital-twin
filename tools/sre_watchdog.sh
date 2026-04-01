#!/bin/bash
# sre_watchdog.sh -- ядро SRE Watchdog (библиотека)
# SOURCE этот файл из tool-specific скриптов, НЕ запускать напрямую.
#
# Предоставляет:
#   - Telegram alerts (send_telegram)
#   - State management (should_alert, save_state)
#   - Common checks (process alive, disk, log staleness)
#   - fmax trend detection (check_fmax_trend)
#   - Main loop (sre_main)
#
# Tool-specific скрипт должен определить sre_tool_check() и вызвать sre_main.
#
# .sre_env format:
#   SRE_BOT_TOKEN="..."
#   SRE_CHAT_ID="..."
#   SRE_LABEL="W3-troilite"

set -uo pipefail
# NOTE: no -e ! Watchdog is a daemon, must not exit on grep/awk failures
export TZ=Europe/Kyiv

#=== CONFIG ===
SRE_ENV="/workspace/.sre_env"
[[ -f "$SRE_ENV" ]] && source "$SRE_ENV"

BOT_TOKEN="${SRE_BOT_TOKEN:-}"
CHAT_ID="${SRE_CHAT_ID:-}"
LABEL="${SRE_LABEL:-$(hostname)}"
TOOL_NAME="${SRE_TOOL:-unknown}"
CHECK_INTERVAL="${SRE_INTERVAL:-300}"  # 5 min

STATE_FILE="/tmp/sre_state"
LOGFILE="/workspace/results/sre_watchdog.log"

# Thresholds (can be overridden before sre_main)
FMAX_WARN_CONSEC=${FMAX_WARN_CONSEC:-3}
FMAX_CRIT_CONSEC=${FMAX_CRIT_CONSEC:-5}
DISK_WARN_PCT=${DISK_WARN_PCT:-85}
DISK_CRIT_PCT=${DISK_CRIT_PCT:-95}
LOG_STALE_SEC=${LOG_STALE_SEC:-1800}

mkdir -p /workspace/results

#=== HELPERS ===
sre_log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOGFILE"
}

# Float: |$1| > $2 ? (combined, fixes QA BASH-3 precision bug)
float_abs_gt() {
    awk "BEGIN {v=$1+0; if(v<0) v=-v; exit !(v > $2)}" 2>/dev/null
}

float_gt() { awk "BEGIN {exit !($1 > $2)}" 2>/dev/null; }

# Safe integer check (fixes QA BASH-12: non-numeric -gt crash)
is_integer() { [[ "$1" =~ ^[0-9]+$ ]]; }

#=== TELEGRAM ===
send_telegram() {
    local level="$1" msg="$2"
    [[ -z "$BOT_TOKEN" || -z "$CHAT_ID" ]] && { sre_log "SKIP telegram: no credentials"; return; }

    local emoji
    case "$level" in
        CRITICAL) emoji="🔴" ;;
        WARNING)  emoji="⚠️" ;;
        DONE)     emoji="🏁" ;;
        OK)       emoji="✅" ;;
        *)        emoji="ℹ️" ;;
    esac

    local text
    text=$(printf "%s %s: %s\n--------------\n%b\n--------------\n%s" \
        "$emoji" "$level" "$LABEL" "$msg" "$(date -u '+%Y-%m-%d %H:%M UTC')")

    curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        -d chat_id="${CHAT_ID}" \
        -d text="${text}" \
        --max-time 10 \
        >/dev/null 2>&1 || sre_log "WARN: telegram send failed"

    sre_log "TELEGRAM [${level}]: $(echo -e "$msg" | head -1)"
}

#=== STATE ===
get_prev_state() { cat "$STATE_FILE" 2>/dev/null || echo "INIT"; }
save_state() { echo "$1" > "$STATE_FILE"; }
should_alert() { [[ "$1" != "$(get_prev_state)" ]]; }

#=== STATUS GLOBALS ===
WD_STATUS="OK"
WD_MSG=""

wd_escalate() {
    local new="$1"
    case "$WD_STATUS" in
        CRITICAL) ;;
        WARNING)  [[ "$new" == "CRITICAL" ]] && WD_STATUS="CRITICAL" ;;
        *)        WD_STATUS="$new" ;;
    esac
}

wd_append() { WD_MSG="${WD_MSG}${WD_MSG:+\n}$1"; }

#=== COMMON CHECKS ===
sre_common_checks() {
    # --- Process alive ---
    local alive=false
    for pat in python3 jdftx abacus_2p mpirun; do
        if ps aux 2>/dev/null | grep -w "$pat" | grep -v grep | grep -v sre_ >/dev/null 2>&1; then
            alive=true
            break
        fi
    done

    if ! $alive; then
        local done_file
        done_file=$(ls -t /workspace/results/DONE* 2>/dev/null | head -1)
        if [[ -n "$done_file" ]]; then
            local done_age
            done_age=$(( $(date +%s) - $(stat -c%Y "$done_file" 2>/dev/null || echo 0) ))
            if [[ $done_age -lt 600 ]]; then
                WD_STATUS="DONE"
                wd_append "Task completed: $(head -1 "$done_file" 2>/dev/null)"
                return
            fi
        fi
        wd_escalate "CRITICAL"
        wd_append "No compute process running, no fresh DONE -- crashed?"
    fi

    # --- Disk ---
    local disk_pct
    disk_pct=$(df /workspace 2>/dev/null | awk 'NR==2{gsub(/%/,""); print $5}')
    if [[ -n "$disk_pct" ]] && is_integer "$disk_pct"; then
        if [[ "$disk_pct" -gt "$DISK_CRIT_PCT" ]]; then
            wd_escalate "CRITICAL"
            wd_append "Disk ${disk_pct}% full!"
        elif [[ "$disk_pct" -gt "$DISK_WARN_PCT" ]]; then
            wd_escalate "WARNING"
            wd_append "Disk ${disk_pct}%"
        fi
    fi

    # --- Log staleness ---
    if $alive; then
        local newest_log
        newest_log=$(ls -t /workspace/results/*.log /workspace/results/*.txt \
                     /workspace/neb_work/*.log /workspace/*.out 2>/dev/null | head -1)
        if [[ -n "$newest_log" ]]; then
            local age
            age=$(( $(date +%s) - $(stat -c%Y "$newest_log" 2>/dev/null || echo 0) ))
            if [[ $age -gt $LOG_STALE_SEC ]]; then
                wd_escalate "WARNING"
                wd_append "Output stale: ${newest_log##*/} not updated for $((age/60)) min"
            fi
        fi
    fi
}

#=== SHARED: fmax trend detection (used by ABACUS and GPAW) ===
# Usage: check_fmax_trend "/path/to/fire_or_neb.log"
check_fmax_trend() {
    local log_file="$1"
    [[ -z "$log_file" || ! -f "$log_file" ]] && return

    local fmax_vals
    fmax_vals=$(grep -E "^(FIRE|BFGS):" "$log_file" 2>/dev/null | awk '{print $NF}')
    [[ -z "$fmax_vals" ]] && return

    local n_steps last_fmax consec=0 prev=""
    n_steps=$(echo "$fmax_vals" | wc -l)
    last_fmax=$(echo "$fmax_vals" | tail -1)

    while IFS= read -r val; do
        if [[ -n "$prev" ]] && float_gt "$val" "$prev"; then
            ((consec++))
        else
            consec=0
        fi
        prev="$val"
    done <<< "$fmax_vals"

    if [[ $consec -ge $FMAX_CRIT_CONSEC ]]; then
        local first_fmax
        first_fmax=$(echo "$fmax_vals" | tail -$((consec+1)) | head -1)
        wd_escalate "CRITICAL"
        wd_append "fmax DIVERGING: ${first_fmax} -> ${last_fmax} (${consec} steps up)"
    elif [[ $consec -ge $FMAX_WARN_CONSEC ]]; then
        wd_escalate "WARNING"
        wd_append "fmax growing: ${consec} steps up, current=${last_fmax}"
    else
        wd_append "FIRE: step ${n_steps}, fmax=${last_fmax}"
    fi
}

#=== MAIN LOOP ===
sre_main() {
    # PID lock with flock (fixes QA BASH-15 race condition)
    local PIDFILE="/tmp/sre_watchdog.pid"
    exec 200>"$PIDFILE"
    flock -n 200 || { echo "SRE Watchdog already running. Exiting."; exit 0; }
    echo $$ >&200

    trap 'rm -f "$PIDFILE"; sre_log "Watchdog stopped (PID $$)"' EXIT

    sre_log "========================================="
    sre_log "SRE Watchdog started (PID $$, label=${LABEL}, tool=${TOOL_NAME}, interval=${CHECK_INTERVAL}s)"
    sre_log "========================================="

    send_telegram "OK" "SRE Watchdog started\nTool: ${TOOL_NAME}\nInterval: ${CHECK_INTERVAL}s"

    while true; do
        WD_STATUS="OK"
        WD_MSG=""

        # Common checks
        sre_common_checks

        # Tool-specific checks (defined by the sourcing script)
        if declare -f sre_tool_check >/dev/null 2>&1; then
            sre_tool_check
        fi

        WD_MSG="Tool: ${TOOL_NAME}\n${WD_MSG}"

        # Log
        local summary
        summary=$(echo -e "$WD_MSG" | tr '\n' ' ' | head -c 200)
        sre_log "[${WD_STATUS}] ${summary}"

        # Alert on state transition
        if should_alert "$WD_STATUS"; then
            send_telegram "$WD_STATUS" "$(echo -e "$WD_MSG")"
            save_state "$WD_STATUS"
        fi

        # Rotate log
        local log_size
        log_size=$(stat -c%s "$LOGFILE" 2>/dev/null || echo 0)
        if is_integer "$log_size" && [[ "$log_size" -gt 5242880 ]]; then
            mv "$LOGFILE" "${LOGFILE}.old"
            sre_log "=== Log rotated ==="
        fi

        sleep "$CHECK_INTERVAL"
    done
}
