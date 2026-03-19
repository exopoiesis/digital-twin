#!/bin/bash
# batch_runner.sh — параллельный запуск GPU-скриптов на одном GPU
# Использование: bash batch_runner.sh MAX_PARALLEL script1.py [script2.py ...]
# Пример: bash batch_runner.sh 8 neb_pentlandite_gpu.py neb_pyrite_gpu.py neb_greigite_gpu.py
#
# MAX_PARALLEL = floor((GPU_VRAM_GB * 0.85) / VRAM_per_task)
# См. vram_budgets.md для VRAM по типам задач

set -euo pipefail

MAX_PARALLEL="${1:?Usage: batch_runner.sh MAX_PARALLEL script1.py [script2.py ...]}"
shift
SCRIPTS=("$@")

if [ ${#SCRIPTS[@]} -eq 0 ]; then
    echo "[BATCH] No scripts specified"
    exit 1
fi

LOGDIR="/workspace/results/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "[BATCH] $(date +%H:%M:%S) Starting ${#SCRIPTS[@]} tasks, max parallel: $MAX_PARALLEL"
echo "[BATCH] Log dir: $LOGDIR"

# GPU info
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "[BATCH] GPU: $GPU_NAME, VRAM: ${GPU_VRAM} MiB"
fi

PIDS=()
RUNNING=0
FAILED=0
COMPLETED=0
OOM=0

for SCRIPT in "${SCRIPTS[@]}"; do
    # Wait if at capacity
    while [ $RUNNING -ge $MAX_PARALLEL ]; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                wait "${PIDS[$i]}" && STATUS=$? || STATUS=$?
                RUNNING=$((RUNNING - 1))
                COMPLETED=$((COMPLETED + 1))
                if [ $STATUS -eq 137 ]; then
                    echo "[BATCH] WARNING: PID ${PIDS[$i]} killed (OOM?), exit=$STATUS"
                    OOM=$((OOM + 1))
                    FAILED=$((FAILED + 1))
                elif [ $STATUS -ne 0 ]; then
                    echo "[BATCH] FAIL: PID ${PIDS[$i]} exit=$STATUS"
                    FAILED=$((FAILED + 1))
                fi
                unset 'PIDS[$i]'
            fi
        done
        PIDS=("${PIDS[@]}")  # reindex
        sleep 1
    done

    LOGFILE="$LOGDIR/$(basename "$SCRIPT" .py).log"
    echo "[BATCH] $(date +%H:%M:%S) Starting: $SCRIPT (running: $RUNNING/$MAX_PARALLEL)"
    python -u "$SCRIPT" > "$LOGFILE" 2>&1 &
    PIDS+=($!)
    RUNNING=$((RUNNING + 1))
done

# Wait for remaining
for PID in "${PIDS[@]}"; do
    wait "$PID" && STATUS=$? || STATUS=$?
    COMPLETED=$((COMPLETED + 1))
    if [ $STATUS -eq 137 ]; then
        echo "[BATCH] WARNING: PID $PID killed (OOM?), exit=$STATUS"
        OOM=$((OOM + 1))
        FAILED=$((FAILED + 1))
    elif [ $STATUS -ne 0 ]; then
        echo "[BATCH] FAIL: PID $PID exit=$STATUS"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "[BATCH] ====== SUMMARY ======"
echo "[BATCH] Total: ${#SCRIPTS[@]}, Completed: $COMPLETED, Failed: $FAILED, OOM: $OOM"
echo "[BATCH] Logs: $LOGDIR"

if [ $OOM -gt 0 ]; then
    echo "[BATCH] WARNING: $OOM tasks killed with exit 137 (likely OOM). Reduce MAX_PARALLEL."
fi

exit $FAILED
