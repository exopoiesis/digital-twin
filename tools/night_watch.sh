#!/bin/bash
# night_watch.sh — hourly check of all Vast.ai instances
# If something dies, try to restart. If restart fails, stop (not destroy) the instance.
set -uo pipefail
SSH_OPTS="-T -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=5"
LOG="D:/home/ignat/project-third-matter/tmp/night_watch.log"

check_instance() {
  local label="$1" host="$2" port="$3" id="$4" task="$5"

  DATA=$(docker exec skypilot ssh $SSH_OPTS -p "$port" "root@$host" \
    'S=$(ps aux --sort=-pcpu|grep python3|grep -v grep|head -1); if [ -n "$S" ]; then echo "ALIVE"; else if [ -f /workspace/DONE ]; then echo "DONE"; else echo "DEAD"; fi; fi' \
    2>/dev/null </dev/null)

  local ts
  ts=$(date '+%H:%M:%S')

  if [ -z "$DATA" ]; then
    echo "[$ts] $label: SSH_ERR (id=$id)" | tee -a "$LOG"
    return
  fi

  local status
  status=$(echo "$DATA" | tr -d '[:space:]')

  case "$status" in
    ALIVE)
      echo "[$ts] $label: ALIVE" | tee -a "$LOG"
      ;;
    DONE)
      echo "[$ts] $label: DONE — task finished" | tee -a "$LOG"
      ;;
    DEAD)
      echo "[$ts] $label: DEAD — attempting restart..." | tee -a "$LOG"
      # Try restart: re-run the last command from run.log or known task
      local restarted=0
      case "$task" in
        bridge)
          docker exec skypilot ssh $SSH_OPTS -p "$port" "root@$host" \
            "cd /workspace && export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 && nohup mpirun --allow-run-as-root -np 12 python3 -u q075_solvation_dft.py --step site_bridge --output-dir /workspace/results/q075_solv > /workspace/results/q075_solv/bridge_stdout.log 2>&1 & disown" \
            2>/dev/null </dev/null && restarted=1
          ;;
        ontop)
          docker exec skypilot ssh $SSH_OPTS -p "$port" "root@$host" \
            "cd /workspace && export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 && nohup mpirun --allow-run-as-root -np 8 python3 -u q075_solvation_dft.py --step site_ontop --output-dir /workspace/results/q075_solv > /workspace/results/q075_solv/ontop_stdout.log 2>&1 & disown" \
            2>/dev/null </dev/null && restarted=1
          ;;
        hollow)
          docker exec skypilot ssh $SSH_OPTS -p "$port" "root@$host" \
            "cd /workspace && export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 && nohup mpirun --allow-run-as-root -np 8 python3 -u q075_solvation_dft.py --step site_hollow --output-dir /workspace/results/q075_solv > /workspace/results/q075_solv/hollow_stdout.log 2>&1 & disown" \
            2>/dev/null </dev/null && restarted=1
          ;;
        greigite|marcasite)
          # These have watchdog, skip manual restart
          echo "[$ts] $label: has watchdog, skipping manual restart" | tee -a "$LOG"
          return
          ;;
      esac

      if [ "$restarted" -eq 1 ]; then
        sleep 5
        # Verify it started
        local verify
        verify=$(docker exec skypilot ssh $SSH_OPTS -p "$port" "root@$host" \
          'ps aux|grep python3|grep -v grep|head -1|wc -l' 2>/dev/null </dev/null)
        if [ "${verify:-0}" -gt 0 ]; then
          echo "[$ts] $label: RESTARTED OK" | tee -a "$LOG"
        else
          echo "[$ts] $label: RESTART FAILED — stopping instance $id" | tee -a "$LOG"
          docker exec skypilot bash -c "vastai stop instance $id 2>/dev/null" </dev/null
          echo "[$ts] $label: instance $id STOPPED" | tee -a "$LOG"
        fi
      fi
      ;;
  esac
}

echo "" >> "$LOG"
echo "=== Night Watch started: $(date '+%Y-%m-%d %H:%M') ===" | tee -a "$LOG"

while true; do
  echo "--- $(date '+%Y-%m-%d %H:%M:%S') ---" | tee -a "$LOG"

  # Instance list: label host port id task
  check_instance "w0-greigite"  "ssh8.vast.ai" 10290 32970291 greigite
  check_instance "W1-ontop"     "ssh8.vast.ai" 10292 32970293 ontop
  check_instance "W3-greigite"  "ssh3.vast.ai" 13050 33003051 greigite
  check_instance "W2-marcasite" "ssh6.vast.ai" 38390 33038391 marcasite
  check_instance "Q075-bridge"  "ssh7.vast.ai" 11228 33051228 bridge
  check_instance "W5-hollow"    "ssh6.vast.ai" 11390 33051390 hollow

  echo "" | tee -a "$LOG"
  sleep 3600
done
