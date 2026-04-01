#!/bin/bash
# sre_gpaw_watchdog.sh -- GPAW DFT watchdog
# Checks: KohnShamConvergenceError, density stall, fmax trend
#
# Deploy: copy sre_watchdog.sh + this file to /workspace/, run:
#   nohup bash /workspace/sre_gpaw_watchdog.sh > /dev/null 2>&1 & disown

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/sre_watchdog.sh"

sre_tool_check() {
    # Find active log
    local gpaw_log=""
    for pattern in "/workspace/results/*.txt" "/workspace/run.log" "/workspace/*stdout*.log"; do
        local found
        found=$(ls -t $pattern 2>/dev/null | head -1)
        if [[ -n "$found" ]] && [[ -f "$found" ]]; then
            gpaw_log="$found"
            break
        fi
    done
    [[ -z "$gpaw_log" ]] && return

    # --- CHECK G1: KohnShamConvergenceError ---
    local n_errors
    n_errors=$(grep -c "KohnShamConvergenceError\|Did not converge" "$gpaw_log" 2>/dev/null || echo 0)
    if [[ "$n_errors" -gt 0 ]]; then
        local last_error_line=0 total_lines=0 lines_after=0
        last_error_line=$(grep -n "KohnShamConvergenceError" "$gpaw_log" 2>/dev/null | tail -1 | cut -d: -f1)
        total_lines=$(wc -l < "$gpaw_log" 2>/dev/null || echo 0)
        lines_after=$(( total_lines - ${last_error_line:-0} ))

        if [[ $lines_after -lt 5 ]]; then
            wd_escalate "WARNING"
            wd_append "KohnShamConvergenceError (${n_errors} total, last near EOF)"
        else
            wd_append "SCF errors: ${n_errors} (script continues, OK)"
        fi
    fi

    # --- CHECK G2: SCF density stall ---
    # GPAW iter line: iter:  57 17:31:13  -443.370c  -3.72  -2.14  +23.38
    # $6 = log10(d_density)
    local density_exps
    density_exps=$(grep "iter:" "$gpaw_log" 2>/dev/null | tail -10 | awk '{print $6}')
    if [[ -n "$density_exps" ]]; then
        local last_exp
        last_exp=$(echo "$density_exps" | tail -1)
        local uniq_int
        uniq_int=$(echo "$density_exps" | awk '{printf "%d\n", $1}' | sort -u | wc -l)
        local n_vals
        n_vals=$(echo "$density_exps" | wc -l)

        if [[ "$uniq_int" -le 1 ]] && [[ "$n_vals" -ge 8 ]]; then
            wd_escalate "WARNING"
            wd_append "SCF density stalled at 10^(${last_exp}) for ${n_vals}+ iters"
        else
            wd_append "SCF density: 10^(${last_exp})"
        fi
    fi

    # --- CHECK G3: fmax trend (shared helper) ---
    local fire_log
    fire_log=$(ls -t /workspace/neb_work/neb.log /workspace/results/neb.log \
               /workspace/neb_work/relax*.log 2>/dev/null | head -1)
    check_fmax_trend "$fire_log"
}

sre_main
