#!/bin/bash
# vast_check.sh — quick status check for Vast.ai instance
# Usage: vast_check.sh <ssh_node> <port> [label]
# Example: vast_check.sh ssh4 19224 W5

NODE="$1"
PORT="$2"
LABEL="${3:-$NODE:$PORT}"
HOST="${NODE}.vast.ai"

if [ -z "$NODE" ] || [ -z "$PORT" ]; then
  echo "Usage: vast_check.sh <ssh_node> <port> [label]"
  echo "  e.g. vast_check.sh ssh4 19224 W5"
  exit 1
fi

# Use heredoc to avoid escaping hell
REMOTE_SCRIPT=$(cat << 'REMOTEOF'
echo "--- PROCESS ---"
ps aux --sort=-pcpu | head -5
echo ""
echo "--- MEMORY ---"
free -m | grep Mem
echo ""
echo "--- HEARTBEAT ---"
cat /workspace/results/heartbeat 2>/dev/null || echo "No heartbeat"
echo ""
echo "--- DONE ---"
cat /workspace/results/DONE* 2>/dev/null || echo "NOT_DONE"
echo ""
echo "--- EXIT CODE ---"
cat /workspace/results/exit_code 2>/dev/null || echo "N/A"
echo ""
echo "--- CRASH INFO ---"
head -10 /workspace/results/crash_info 2>/dev/null || echo "No crash"
echo ""
echo "--- MONITOR (last 5) ---"
tail -5 /workspace/results/monitor.log 2>/dev/null || echo "No monitor.log"
echo ""
echo "--- RUN LOG (last 5) ---"
tail -5 /workspace/run.log 2>/dev/null
tail -5 /workspace/results/v2_log.txt 2>/dev/null
echo ""
echo "--- STDERR (last 3) ---"
tail -3 /workspace/run_stderr.log 2>/dev/null || echo "No stderr"
echo ""
echo "--- XYZ ---"
wc -l /workspace/results/*.xyz 2>/dev/null
grep -c Properties /workspace/results/*.xyz 2>/dev/null

echo ""
echo "--- ETA ---"
# FIRE logs (ABACUS NEB)
for f in /workspace/neb_work/neb.log /workspace/neb_work/relax_start.log /workspace/neb_work/relax_end.log; do
  if [ -f "$f" ] && grep -q "^FIRE:" "$f" 2>/dev/null; then
    n=$(grep -c "^FIRE:" "$f")
    last=$(grep "^FIRE:" "$f" | tail -1)
    fmax=$(echo "$last" | awk '{print $NF}')
    # Time per step from last 2
    t1=$(grep "^FIRE:" "$f" | tail -2 | head -1 | awk '{print $3}')
    t2=$(grep "^FIRE:" "$f" | tail -1 | awk '{print $3}')
    s1=$(echo "$t1" | awk -F: '{print $1*3600+$2*60+$3}')
    s2=$(echo "$t2" | awk -F: '{print $1*3600+$2*60+$3}')
    dt=$((s2-s1)); [ "$dt" -lt 0 ] && dt=$((dt+86400))
    fname=$(basename "$f")
    if command -v python3 &>/dev/null && [ "$dt" -gt 0 ]; then
      eta=$(python3 -c "
import math
fmax=${fmax}; target=0.05; dt=${dt}
if fmax<=target: print('CONVERGED')
else:
  steps=int(math.log(fmax/target)/math.log(2)*5)
  h=steps*dt/3600
  print(f'step {$n}, fmax={fmax}, ~{dt}s/step, ~{steps} steps, ~{h:.1f}h')
" 2>/dev/null)
      echo "  $fname: $eta"
    else
      echo "  $fname: step $n, fmax=$fmax, ~${dt}s/step"
    fi
  fi
done
# LCAOMinimize (JDFTx)
for f in $(find /workspace/results -name "*.out" -size +1k 2>/dev/null); do
  if grep -q "LCAOMinimize: Iter:" "$f" 2>/dev/null; then
    last=$(grep "LCAOMinimize: Iter:" "$f" | tail -1)
    iter=$(echo "$last" | awk -F'Iter:' '{print $2}' | awk '{print $1}')
    grad=$(echo "$last" | sed 's/.*|grad|_K:\s*//' | awk '{print $1}')
    t1=$(grep "LCAOMinimize: Iter:" "$f" | tail -2 | head -1 | sed 's/.*t\[s\]:\s*//' | awk '{printf "%.0f",$1}')
    t2=$(grep "LCAOMinimize: Iter:" "$f" | tail -1 | sed 's/.*t\[s\]:\s*//' | awk '{printf "%.0f",$1}')
    if [ -n "$t1" ] && [ -n "$t2" ]; then
      dt=$((t2-t1))
      left=$((40-iter)); [ "$left" -lt 5 ] && left=5
      if command -v python3 &>/dev/null; then
        eta_h=$(python3 -c "print(f'{${left}*${dt}/3600:.1f}')" 2>/dev/null)
        echo "  $(basename $f): LCAOMin iter $iter, |grad|=$grad, ~${dt}s/iter, ~${left} left, ~${eta_h}h (LCAO phase, SCF follows)"
      else
        echo "  $(basename $f): LCAOMin iter $iter, ~${dt}s/iter"
      fi
    else
      echo "  $(basename $f): LCAOMin iter $iter, |grad|=$grad"
    fi
  fi
done
true
REMOTEOF
)

echo "=== $LABEL ==="
docker exec skypilot ssh -T -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$PORT" "root@$HOST" "$REMOTE_SCRIPT" < /dev/null 2>&1
