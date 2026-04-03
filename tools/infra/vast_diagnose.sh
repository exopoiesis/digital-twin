#!/bin/bash
# vast_diagnose.sh — post-mortem diagnostics for Vast.ai instance
# Run from LOCAL machine (via loki/skypilot).
#
# Usage: bash infra/vast_diagnose.sh <ssh_node> <port> [label]
# Example: bash infra/vast_diagnose.sh ssh9 27144 v2-pent-w2

NODE="$1"
PORT="$2"
LABEL="${3:-$NODE:$PORT}"
HOST="${NODE}.vast.ai"

if [ -z "$NODE" ] || [ -z "$PORT" ]; then
    echo "Usage: vast_diagnose.sh <ssh_node> <port> [label]"
    exit 1
fi

echo "=== DIAGNOSTICS: $LABEL ($HOST:$PORT) ==="
echo ""

docker exec skypilot bash -c "ssh -T -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p $PORT root@$HOST '

echo \"=== 1. SYSTEM INFO ===\"
echo \"Host: \$(hostname)\"
echo \"Uptime: \$(uptime)\"
echo \"Kernel: \$(uname -r)\"
echo \"\"

echo \"=== 2. MEMORY ===\"
free -m
echo \"\"

echo \"=== 3. DISK ===\"
df -h /workspace /tmp / 2>/dev/null
echo \"\"

echo \"=== 4. OOM KILLER (dmesg) ===\"
dmesg 2>/dev/null | grep -i \"oom\|killed process\|out of memory\|invoked oom\" | tail -20
if [ \$? -ne 0 ]; then echo \"No OOM events found (or dmesg not available)\"; fi
echo \"\"

echo \"=== 5. ALL PROCESSES ===\"
ps aux --sort=-pcpu 2>/dev/null | head -15
echo \"\"

echo \"=== 6. PYTHON3 STATUS ===\"
py_pids=\$(pgrep python3 2>/dev/null)
if [ -n \"\$py_pids\" ]; then
    echo \"python3 RUNNING: PIDs=\$py_pids\"
    for p in \$py_pids; do
        echo \"  PID \$p: \$(ps -p \$p -o pcpu,rss,vsz,etime,args --no-headers 2>/dev/null)\"
    done
else
    echo \"!!! python3 NOT RUNNING !!!\"
fi
echo \"\"

echo \"=== 7. DONE FILE ===\"
ls -la /workspace/results/DONE* 2>/dev/null || echo \"No DONE file\"
cat /workspace/results/DONE* 2>/dev/null || true
echo \"\"

echo \"=== 8. MONITOR LOG (last 30 lines) ===\"
tail -30 /workspace/results/monitor.log 2>/dev/null || echo \"No monitor.log\"
echo \"\"

echo \"=== 9. HEARTBEAT ===\"
if [ -f /workspace/results/heartbeat ]; then
    echo \"Last heartbeat: \$(cat /workspace/results/heartbeat)\"
    echo \"Current time:   \$(date \"+%Y-%m-%d %H:%M:%S\")\"
else
    echo \"No heartbeat file (monitor not running?)\"
fi
echo \"\"

echo \"=== 10. RUN LOG (last 30 lines) ===\"
tail -30 /workspace/run.log 2>/dev/null || echo \"No run.log\"
echo \"\"

echo \"=== 11. STDERR LOG (last 20 lines) ===\"
tail -20 /workspace/run_stderr.log 2>/dev/null || echo \"No stderr log\"
echo \"\"

echo \"=== 12. XYZ PROGRESS ===\"
wc -l /workspace/results/*.xyz 2>/dev/null || echo \"No XYZ files\"
grep -c Properties /workspace/results/*.xyz 2>/dev/null || echo \"0 configs\"
echo \"\"

echo \"=== 13. LOCK FILES ===\"
ls -la /workspace/*.lock 2>/dev/null || echo \"No lock files\"
echo \"\"

echo \"=== 14. LAST MODIFIED FILES ===\"
ls -lt /workspace/results/ 2>/dev/null | head -10
echo \"\"

echo \"=== 15. NETWORK ===\"
ping -c 1 -W 2 8.8.8.8 > /dev/null 2>&1 && echo \"Internet: OK\" || echo \"Internet: FAIL\"
echo \"\"

echo \"=== END DIAGNOSTICS ===\"
' < /dev/null" 2>&1
