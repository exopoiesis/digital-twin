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
true
REMOTEOF
)

echo "=== $LABEL ==="
docker exec skypilot ssh -T -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$PORT" "root@$HOST" "$REMOTE_SCRIPT" < /dev/null 2>&1
