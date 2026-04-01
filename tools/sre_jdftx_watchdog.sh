#!/bin/bash
# sre_jdftx_watchdog.sh -- JDFTx DFT watchdog
# Checks: NaN/Inf, LCAO stall, ElecMinimize, electron drift, checkpoints
#
# Deploy: copy sre_watchdog.sh + this file to /workspace/, run:
#   nohup bash /workspace/sre_jdftx_watchdog.sh > /dev/null 2>&1 & disown

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/sre_watchdog.sh"

sre_tool_check() {
    # Find active .out file
    local jdftx_log=""
    for pattern in "/workspace/*.out" "/workspace/results/*.out" "/workspace/*/*.out"; do
        local found
        found=$(ls -t $pattern 2>/dev/null | head -1)
        if [[ -n "$found" ]] && [[ -f "$found" ]]; then
            jdftx_log="$found"
            break
        fi
    done
    [[ -z "$jdftx_log" ]] && return

    # --- CHECK J1: NaN / Inf (instant CRITICAL) ---
    if tail -100 "$jdftx_log" 2>/dev/null | grep -qiE "\bnan\b|\binf\b|-inf"; then
        wd_escalate "CRITICAL"
        wd_append "NaN/Inf in JDFTx output!"
        return
    fi

    # --- CHECK J2: LCAO stall ---
    local lcao_iters=0 step_increased=0
    lcao_iters=$(grep -c "LCAOMinimize: Iter:" "$jdftx_log" 2>/dev/null || echo 0)
    step_increased=$(grep -c "Step increased F" "$jdftx_log" 2>/dev/null || echo 0)

    if [[ $lcao_iters -gt 5 ]] && [[ $step_increased -gt $(( lcao_iters * 2 / 3 )) ]]; then
        wd_escalate "WARNING"
        wd_append "LCAO struggling: ${step_increased}/${lcao_iters} iters had energy increase"
    fi

    if [[ $lcao_iters -gt 0 ]]; then
        local last_grad
        last_grad=$(grep "LCAOMinimize: Iter:" "$jdftx_log" 2>/dev/null | tail -1 \
                    | sed 's/.*|grad|_K:\s*//' | awk '{print $1}')
        wd_append "LCAO: iter ${lcao_iters}, |grad|=${last_grad:-?}"
    fi

    # --- CHECK J3: ElecMinimize ---
    local elec_iters
    elec_iters=$(grep -c "ElecMinimize: Iter:" "$jdftx_log" 2>/dev/null || echo 0)
    if [[ $elec_iters -gt 0 ]]; then
        local last_elec_grad
        last_elec_grad=$(grep "ElecMinimize: Iter:" "$jdftx_log" 2>/dev/null | tail -1 \
                        | sed 's/.*|grad|_K:\s*//' | awk '{print $1}')
        wd_append "SCF: iter ${elec_iters}, |grad|=${last_elec_grad:-?}"
    fi

    # --- CHECK J4: Electron drift ---
    local first_ne="" last_ne=""
    first_ne=$(grep "nElectrons:" "$jdftx_log" 2>/dev/null | head -1 \
              | sed 's/.*nElectrons:\s*//' | awk '{printf "%.1f", $1}')
    last_ne=$(grep "nElectrons:" "$jdftx_log" 2>/dev/null | tail -1 \
             | sed 's/.*nElectrons:\s*//' | awk '{printf "%.1f", $1}')
    if [[ -n "$first_ne" ]] && [[ -n "$last_ne" ]] && [[ "$first_ne" != "$last_ne" ]]; then
        wd_escalate "CRITICAL"
        wd_append "Electron drift: ${first_ne} -> ${last_ne}"
    fi

    # --- CHECK J5: Checkpoint freshness ---
    local wfns
    wfns=$(ls -t /workspace/*.wfns /workspace/*/*.wfns 2>/dev/null | head -1)
    if [[ -n "$wfns" ]]; then
        local wfns_age
        wfns_age=$(( $(date +%s) - $(stat -c%Y "$wfns" 2>/dev/null || echo 0) ))
        if [[ $wfns_age -gt 7200 ]]; then
            wd_append "Checkpoint stale: $((wfns_age/60)) min old"
        fi
    fi
}

sre_main
