#!/bin/bash
# vast_launch.sh — launch wrapper for GPAW scripts on Vast.ai
# Handles: logging, signal traps, monitor daemon, lock files.
#
# Usage (on the instance):
#   bash /workspace/vast_launch.sh <script.py> [args...]
#
# Example:
#   bash /workspace/vast_launch.sh generate_sulfide_dft_data_v2.py \
#     --output /workspace/results/sulfide_train.xyz --resume \
#     --mineral pentlandite --worker-id 2 --num-workers 4
#
# Creates:
#   /workspace/run.log        — stdout
#   /workspace/run_stderr.log — stderr (separate!)
#   /workspace/run.lock       — lock file (prevents duplicates)
#   /workspace/results/monitor.log — system metrics (from monitor daemon)
#   /workspace/results/exit_code   — python3 exit code
#   /workspace/results/crash_info  — crash details if non-zero exit

set -uo pipefail
export TZ=Europe/Kyiv

# === GPU FFT (GPAW new API + CuPy) ===
# GPAW_NEW=1 switches to new API with CuPyFFTPlans (GPU FFT).
# WARNING: incompatible with Blackwell (RTX 5070 Ti, sm_120).
# Works on: RTX 3060/3080/3090/4000 Ada (sm_86/89).
# Default OFF — enable per-instance:
#   export GPAW_NEW=1 GPAW_USE_GPUS=1
export GPAW_NEW="${GPAW_NEW:-0}"
export GPAW_USE_GPUS="${GPAW_USE_GPUS:-0}"

# === CPU THREADING ===
# PW + GPU (GPAW_NEW=1): OMP_NUM_THREADS=1 mandatory — GPU FFT replaces OMP.
#   Source: gpaw/new/pw/builder.py:42 warns "OMP>1 in PW-mode is not useful!"
# PW without GPU: OMP=nproc — FFTW + LAPACK benefit from threads.
# FD mode (SolvationGPAW): OMP=nproc — grid operations are CPU-bound.
if [ -z "${OMP_NUM_THREADS:-}" ]; then
    if [ "${GPAW_NEW}" = "1" ]; then
        export OMP_NUM_THREADS=1
    else
        export OMP_NUM_THREADS=$(nproc)
    fi
fi
# Propagate to BLAS/LAPACK backends
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

SCRIPT="$1"
shift
ARGS="$@"

LOCKFILE="/workspace/run.lock"
LOGFILE="/workspace/run.log"
ERRLOG="/workspace/run_stderr.log"
EXIT_CODE_FILE="/workspace/results/exit_code"
CRASH_INFO="/workspace/results/crash_info"

mkdir -p /workspace/results

# === LOCK FILE (prevent duplicates) ===
if [ -f "$LOCKFILE" ]; then
    OLD_PID=$(cat "$LOCKFILE" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: Another instance running (PID $OLD_PID). Exiting."
        echo "  To force: rm $LOCKFILE"
        exit 1
    else
        echo "WARNING: Stale lock file (PID $OLD_PID not running). Removing."
        rm -f "$LOCKFILE"
    fi
fi

# === START MONITOR DAEMON ===
echo "Starting monitor daemon..."
if [ -f /workspace/vast_monitor.sh ]; then
    nohup bash /workspace/vast_monitor.sh > /dev/null 2>&1 &
    echo "Monitor started (PID $!)"
else
    echo "WARNING: vast_monitor.sh not found. No system monitoring."
fi

# === SIGNAL HANDLING ===
cleanup() {
    local exit_code=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Script terminated with exit code: $exit_code" | tee -a "$LOGFILE"
    echo "$exit_code" > "$EXIT_CODE_FILE"

    if [ $exit_code -ne 0 ]; then
        {
            echo "=== CRASH REPORT ==="
            echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "Exit code: $exit_code"
            echo "Signal: $(kill -l $exit_code 2>/dev/null || echo 'N/A')"
            echo ""
            echo "--- Last 20 lines of stdout ---"
            tail -20 "$LOGFILE" 2>/dev/null
            echo ""
            echo "--- Last 20 lines of stderr ---"
            tail -20 "$ERRLOG" 2>/dev/null
            echo ""
            echo "--- Memory at crash ---"
            free -m 2>/dev/null
            echo ""
            echo "--- OOM in dmesg ---"
            dmesg 2>/dev/null | grep -i "oom\|killed process\|out of memory" | tail -10
        } > "$CRASH_INFO"
        echo "Crash info saved to $CRASH_INFO"
    fi

    rm -f "$LOCKFILE"
}

trap cleanup EXIT
trap 'echo "[$(date)] Received SIGTERM" >> "$LOGFILE"; exit 143' TERM
trap 'echo "[$(date)] Received SIGINT" >> "$LOGFILE"; exit 130' INT

# === LOG ROTATION ===
# Preserve previous logs for crash forensics (don't overwrite!)
for f in "$LOGFILE" "$ERRLOG" "$CRASH_INFO"; do
    [ -s "$f" ] && cp "$f" "${f}.prev" 2>/dev/null
done

# === BACKUP RESULTS ===
# Preserve trajectory, BFGS logs, JSON results before launching.
# Script may overwrite .traj on start — losing hours/days of optimization.
RESULTS_DIR="/workspace/results"
BACKUP_TS=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${RESULTS_DIR}/_backup_${BACKUP_TS}"
_backed_up=0
for ext in traj json xyz; do
    for f in $(find "$RESULTS_DIR" -maxdepth 2 -name "*.${ext}" -size +0c 2>/dev/null); do
        if [ $_backed_up -eq 0 ]; then
            mkdir -p "$BACKUP_DIR"
            echo "Backing up results to $BACKUP_DIR ..."
            _backed_up=1
        fi
        cp "$f" "$BACKUP_DIR/" 2>/dev/null
        echo "  backed up: $(basename $f) ($(stat -c%s "$f" 2>/dev/null || echo '?') bytes)"
    done
done
# Also backup BFGS/FIRE logs
for f in $(find "$RESULTS_DIR" -maxdepth 2 -name "*bfgs*.log" -size +0c 2>/dev/null); do
    [ $_backed_up -eq 0 ] && mkdir -p "$BACKUP_DIR" && _backed_up=1
    cp "$f" "$BACKUP_DIR/" 2>/dev/null
done
[ $_backed_up -eq 1 ] && echo "Backup complete: $(du -sh $BACKUP_DIR | cut -f1)" || echo "No previous results to back up."

# === LAUNCH ===
echo $$ > "$LOCKFILE"

echo "=========================================" | tee "$LOGFILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching: python3 -u $SCRIPT $ARGS" | tee -a "$LOGFILE"
echo "Host: $(hostname)" | tee -a "$LOGFILE"
echo "RAM: $(free -m | awk '/Mem:/{print $2}')M total" | tee -a "$LOGFILE"
echo "CPU: $(nproc) cores, OMP_NUM_THREADS=$OMP_NUM_THREADS" | tee -a "$LOGFILE"
echo "GPU: GPAW_NEW=$GPAW_NEW, GPAW_USE_GPUS=$GPAW_USE_GPUS" | tee -a "$LOGFILE"
echo "Lock: $LOCKFILE (PID $$)" | tee -a "$LOGFILE"
echo "=========================================" | tee -a "$LOGFILE"

# Run python3 with SEPARATE stdout and stderr
python3 -u "/workspace/$SCRIPT" $ARGS >> "$LOGFILE" 2>> "$ERRLOG"
PY_EXIT=$?

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Python exited with code: $PY_EXIT" | tee -a "$LOGFILE"
echo "$PY_EXIT" > "$EXIT_CODE_FILE"

exit $PY_EXIT
