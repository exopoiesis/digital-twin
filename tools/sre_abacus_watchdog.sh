#!/bin/bash
# sre_abacus_watchdog.sh -- ABACUS DFT watchdog
# Checks: AFM collapse, fmax trend, SCF convergence, drho, zombie orted
#
# Deploy: copy sre_watchdog.sh + this file to /workspace/, run:
#   nohup bash /workspace/sre_abacus_watchdog.sh > /dev/null 2>&1 & disown

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/sre_watchdog.sh"

AFM_MAG_THRESHOLD=${AFM_MAG_THRESHOLD:-1}

sre_tool_check() {
    local scf_log="" fire_log="" neb_log=""

    # Find logs
    scf_log=$(find /workspace -path "*/OUT.ABACUS/running_scf.log" -print -quit 2>/dev/null || true)
    for f in /workspace/neb_work/relax_start.log /workspace/neb_work/relax_end.log \
             /workspace/relax_start.log /workspace/relax_end.log; do
        [[ -f "$f" ]] && { fire_log="$f"; break; }
    done
    for f in /workspace/neb_work/neb.log /workspace/results/neb.log /workspace/neb.log; do
        [[ -f "$f" ]] && { neb_log="$f"; break; }
    done

    # Prefer freshest FIRE-type log
    local active_log="$fire_log"
    if [[ -n "$neb_log" ]] && [[ -n "$fire_log" ]]; then
        local neb_mt=0 fire_mt=0
        neb_mt=$(stat -c%Y "$neb_log" 2>/dev/null || echo 0)
        fire_mt=$(stat -c%Y "$fire_log" 2>/dev/null || echo 0)
        [[ "$neb_mt" -gt "$fire_mt" ]] && active_log="$neb_log"
    fi
    [[ -z "$active_log" ]] && active_log="$neb_log"

    # --- CHECK A1: AFM collapse (instant CRITICAL) ---
    if [[ -n "$scf_log" ]] && [[ -f "$scf_log" ]]; then
        local last_mag
        last_mag=$(grep "Total magnetism" "$scf_log" 2>/dev/null | tail -1 | awk '{print $NF}')
        if [[ -n "$last_mag" ]]; then
            if float_abs_gt "$last_mag" "$AFM_MAG_THRESHOLD"; then
                wd_escalate "CRITICAL"
                wd_append "AFM COLLAPSE! Total mag = ${last_mag} (expected ~0)"
                wd_append "Fix: nupdown=0 + mixing_beta_mag=0.2"
            else
                wd_append "AFM: OK (mag=${last_mag})"
            fi
        fi
    fi

    # --- CHECK A2: fmax trend (shared helper) ---
    check_fmax_trend "$active_log"

    # --- CHECK A3: SCF convergence ---
    if [[ -n "$scf_log" ]] && [[ -f "$scf_log" ]]; then
        if grep -q "SCF IS NOT CONVERGED" "$scf_log" 2>/dev/null; then
            wd_escalate "WARNING"
            local scf_iters
            scf_iters=$(grep "#ELEC ITER#" "$scf_log" 2>/dev/null | tail -1 | awk '{print $NF}')
            local last_drho
            last_drho=$(grep "Electron density deviation" "$scf_log" 2>/dev/null | tail -1 | awk '{print $NF}')
            wd_append "SCF NOT CONVERGED (${scf_iters:-?} iters, drho=${last_drho:-?})"
        fi

        # SCF near ceiling (QA BASH-12 fix: is_integer guard)
        local current_iter
        current_iter=$(grep "#ELEC ITER#" "$scf_log" 2>/dev/null | tail -1 | awk '{print $NF}')
        if [[ -n "$current_iter" ]] && is_integer "$current_iter" && [[ "$current_iter" -gt 350 ]]; then
            wd_append "SCF: ${current_iter} iters (near ceiling)"
        fi
    fi

    # --- CHECK A4: zombie orted ---
    local n_orted
    n_orted=$(ps aux 2>/dev/null | grep -c "[o]rted" || echo 0)
    if is_integer "$n_orted" && [[ "$n_orted" -gt 3 ]]; then
        wd_escalate "WARNING"
        wd_append "Zombie orted: ${n_orted} processes (memory leak)"
    fi
}

sre_main
