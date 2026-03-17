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

docker exec skypilot bash -c "ssh -T -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $PORT root@$HOST '\
echo === $LABEL ===; \
echo; \
echo --- PROCESS ---; \
ps aux --sort=-pcpu | head -5; \
echo; \
echo --- MEMORY ---; \
free -m | awk \"/Mem:/{printf \\\"RAM: %dM/%dM (%.0f%% used), avail %dM\n\\\", \\\$3, \\\$2, \\\$3/\\\$2*100, \\\$7}\"; \
echo; \
echo --- HEARTBEAT ---; \
if [ -f /workspace/results/heartbeat ]; then \
  echo \"Last: \$(cat /workspace/results/heartbeat)\"; \
  echo \"Now:  \$(date \"+%Y-%m-%d %H:%M:%S\")\"; \
else \
  echo \"No heartbeat (monitor not running)\"; \
fi; \
echo; \
echo --- DONE ---; \
cat /workspace/results/DONE* 2>/dev/null || echo NOT_DONE; \
echo; \
echo --- EXIT CODE ---; \
cat /workspace/results/exit_code 2>/dev/null || echo \"N/A\"; \
echo; \
echo --- CRASH INFO ---; \
if [ -f /workspace/results/crash_info ]; then \
  head -15 /workspace/results/crash_info; \
else \
  echo \"No crash\"; \
fi; \
echo; \
echo --- MONITOR (last 5) ---; \
tail -5 /workspace/results/monitor.log 2>/dev/null || echo \"No monitor.log\"; \
echo; \
echo --- RUN LOG (last 5) ---; \
tail -5 /workspace/run.log 2>/dev/null; \
tail -5 /workspace/results/v2_log.txt 2>/dev/null; \
echo; \
echo --- STDERR (last 5) ---; \
tail -5 /workspace/run_stderr.log 2>/dev/null || echo \"No stderr\"; \
echo; \
echo --- XYZ ---; \
wc -l /workspace/results/*.xyz 2>/dev/null; \
grep -c Properties /workspace/results/*.xyz 2>/dev/null; \
true \
' < /dev/null" 2>&1
