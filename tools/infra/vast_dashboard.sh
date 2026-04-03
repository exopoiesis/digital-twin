#!/bin/bash
# vast_dashboard.sh — Сводная таблица всех Vast.ai инстансов
# Использование: bash infra/vast_dashboard.sh
#
# Собирает данные по каждому running-инстансу за один SSH-вызов:
#   задача, статус, CPU/RAM/GPU, прогресс, скорость

set -uo pipefail
SSH_OPTS="-T -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=5"

# Colors (disable if piped)
if [ -t 1 ]; then
  G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; N='\033[0m'; B='\033[1m'
else
  G=''; R=''; Y=''; N=''; B=''
fi

echo -e "${B}=== Vast.ai Dashboard — $(date '+%Y-%m-%d %H:%M') ===${N}"
echo ""

# Get instance list
INSTANCES=$(docker exec skypilot bash -c "vastai show instances 2>/dev/null" 2>/dev/null)
if [ -z "$INSTANCES" ]; then
  echo "ERROR: cannot reach vastai CLI"
  exit 1
fi

# Parse running instances
RUNNING=$(echo "$INSTANCES" | awk 'NR>1 && $3=="running" {
  print $1"|"$10"|"$11"|"$17"|"$5"|"$7"|"$8"|"$12"|"$19
}')

if [ -z "$RUNNING" ]; then
  echo "Нет running инстансов."
  exit 0
fi

COUNT=$(echo "$RUNNING" | wc -l)
echo -e "Инстансов: ${B}$COUNT${N}"
echo ""

# Header
echo -e "  ${B}STATUS  LABEL          VRAM/RAM      vCPU  \$/hr  UPTIME  PROGR.       TASK${N}"
echo    "  ------  -------------- ------------- ----- -----  ------  -----------  ------------------------------------------"

# Remote probe script — single line to avoid heredoc issues
# Outputs: STATUS|CPU%|VRAM_used/total|RAM_used/total|XYZ_cfgs|SCF_iters|TASK_short|LAST_CFG
PROBE='
S=$(ps aux --sort=-pcpu|grep python3|grep -v grep|head -1);
if [ -n "$S" ]; then ST=ALIVE; else if [ -f /workspace/DONE ]; then ST=DONE; else ST=DEAD; fi; fi;
CPU=$(echo "$S"|awk "{printf \"%.0f\", \$3}");
TASK=$(echo "$S"|awk "{for(i=11;i<=NF;i++) printf \"%s \",\$i}"|tr -d "\n"|sed "s|/workspace/||g;s|python3 -u ||;s| --output[^ ]* [^ ]*||g;s| --step | |;s| --num-workers [0-9]*||;s| --worker-id | w|;s| --resume||;s|  *| |g"|head -c50);
RU=$(free -m|awk "NR==2{printf \"%.0f\",\$3/1024}");
RT=$(free -m|awk "NR==2{printf \"%.0f\",\$2/1024}");
GI=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null|head -1);
if [ -n "$GI" ]; then VU=$(echo "$GI"|awk -F", " "{printf \"%.0f\",\$1/1024}"); VT=$(echo "$GI"|awk -F", " "{printf \"%.0f\",\$2/1024}"); else VU="?"; VT="?"; fi;
XC=0; for f in /workspace/results/*.xyz /workspace/results/*/*.xyz; do [ -f "$f" ]&&XC=$((XC+$(grep -c "^[0-9]" "$f" 2>/dev/null||echo 0))); done;
SC=0; for f in /workspace/results/q075_solv/site_*.txt; do [ -f "$f" ]&&SC=$((SC+$(grep -c "^iter:" "$f" 2>/dev/null||echo 0))); done;
printf "%s|%s|%s/%s|%s/%s|%s|%s|%s" "$ST" "$CPU" "$VU" "$VT" "$RU" "$RT" "$XC" "$SC" "$TASK"
'

# Read from fd3, leaving stdin free for docker exec
while IFS='|' read -r id ssh_addr ssh_port label gpu vcpu ram cost uptime_min <&3; do
  DATA=$(docker exec skypilot ssh $SSH_OPTS -p "$ssh_port" "root@$ssh_addr" "$PROBE" 2>/dev/null </dev/null)

  if [ -z "$DATA" ]; then
    echo -e "  ${R}SSH_ERR${N}  $label ($gpu, ${vcpu}cpu, \$$cost/hr)"
    continue
  fi

  IFS='|' read -r status cpu vram_info ram_info xyz_n scf_n task <<< "$DATA"

  # Progress string (short: Nc = configs, Ns = SCF iters)
  prog=""
  [ "${xyz_n:-0}" -gt 0 ] 2>/dev/null && prog="${xyz_n}c"
  if [ "${scf_n:-0}" -gt 0 ] 2>/dev/null; then
    [ -n "$prog" ] && prog="${prog}+"
    prog="${prog}${scf_n}s"
  fi
  [ -z "$prog" ] && prog="-"

  # Color status with manual padding (ANSI codes break printf width)
  case "$status" in
    ALIVE) sc="${G}ALIVE${N} " ;;
    DONE)  sc="${Y}DONE${N}  " ;;
    DEAD)  sc="${R}DEAD${N}  " ;;
    *)     sc=$(printf "%-6s" "$status") ;;
  esac

  # Format price to 2 decimals
  price=$(awk "BEGIN{printf \"%.2f\", ${cost:-0}}")

  # Format uptime: minutes -> XhYm
  ut_int=$(awk "BEGIN{printf \"%.0f\", ${uptime_min:-0}}")
  ut_h=$((ut_int / 60))
  ut_m=$((ut_int % 60))
  if [ "$ut_h" -gt 0 ]; then
    uptime="${ut_h}h${ut_m}m"
  else
    uptime="${ut_m}m"
  fi

  # One-line output: status first, then fixed-width fields
  rest=$(printf "%-14s %-13s %5s  \$%-5s %-7s %-12s %s" \
    "$label" "${vram_info}|${ram_info}" "$vcpu" "$price" "$uptime" "$prog" "$task")
  echo -e "  ${sc}  ${rest}"

done 3<<< "$RUNNING"

echo ""
TOTAL=$(echo "$RUNNING" | awk -F'|' '{s+=$8}END{printf "%.2f",s}')
DAILY=$(awk "BEGIN{printf \"%.2f\", $TOTAL*24}")
echo -e "${B}Total: \$${TOTAL}/hr = \$${DAILY}/day${N}"
echo ""
