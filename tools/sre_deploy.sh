#!/bin/bash
# sre_deploy.sh -- Deploy SRE Watchdog (core + tool) to a remote host
#
# Usage: bash infra/sre_deploy.sh <ssh_node> <ssh_port> <label> <tool> [container]
#   tool: abacus|gpaw|jdftx|mattergen
#   container: only for ax102 (default: abacus-worker)
#
# Examples:
#   bash infra/sre_deploy.sh ssh5 34572 W3-troilite abacus
#   bash infra/sre_deploy.sh ssh8 10292 W1-vacancy gpaw
#   bash infra/sre_deploy.sh ax102 0 AX102-candle jdftx abacus-worker
#   bash infra/sre_deploy.sh gomer 0 gomer-mattergen mattergen mattergen-gen

set -euo pipefail

NODE="${1:?Usage: sre_deploy.sh <node> <port> <label> <tool> [container]}"
PORT="${2:?Missing port}"
LABEL="${3:?Missing label}"
TOOL="${4:?Missing tool (abacus|gpaw|jdftx|mattergen)}"
CONTAINER="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CRED_FILE="${SCRIPT_DIR}/sre_credentials.env"
CORE="${SCRIPT_DIR}/sre_watchdog.sh"
TOOL_SCRIPT="${SCRIPT_DIR}/sre_${TOOL}_watchdog.sh"

[[ ! -f "$CRED_FILE" ]] && { echo "ERROR: $CRED_FILE not found"; exit 1; }
[[ ! -f "$TOOL_SCRIPT" ]] && { echo "ERROR: $TOOL_SCRIPT not found (bad tool name?)"; exit 1; }
source "$CRED_FILE"

echo "=== Deploying SRE [${TOOL}] to ${LABEL} (${NODE}:${PORT}) ==="

# Write .sre_env content
make_env() {
    cat <<ENVEOF
SRE_BOT_TOKEN="${SRE_BOT_TOKEN}"
SRE_CHAT_ID="${SRE_CHAT_ID}"
SRE_LABEL="${LABEL}"
SRE_TOOL="${TOOL}"
ENVEOF
}

# --- gomer (local Docker context) ---
if [[ "$NODE" == "gomer" ]]; then
    CONTAINER="${CONTAINER:-mattergen-gen}"
    echo "[1/3] Copy files to gomer..."
    docker --context gomer cp "$CORE" "${CONTAINER}:/workspace/sre_watchdog.sh"
    docker --context gomer cp "$TOOL_SCRIPT" "${CONTAINER}:/workspace/sre_${TOOL}_watchdog.sh"

    echo "[2/3] Write .sre_env..."
    make_env | docker --context gomer exec -i "${CONTAINER}" bash -c "cat > /workspace/.sre_env"
    docker --context gomer exec "${CONTAINER}" bash -c "chmod +x /workspace/sre_*.sh"

    echo "[3/3] Start watchdog..."
    docker --context gomer exec "${CONTAINER}" bash -c "
        if [ -f /tmp/sre_watchdog.pid ]; then
            kill \$(cat /tmp/sre_watchdog.pid) 2>/dev/null || true
            rm -f /tmp/sre_watchdog.pid /tmp/sre_state
        fi
        nohup bash /workspace/sre_${TOOL}_watchdog.sh > /dev/null 2>&1 & disown
        sleep 2
        echo \"PID: \$(cat /tmp/sre_watchdog.pid 2>/dev/null || echo FAILED)\"
    "
    echo "=== DONE: ${LABEL} ==="
    exit 0
fi

# --- AX102 (Hetzner, via skypilot SSH) ---
if [[ "$NODE" == "ax102" ]]; then
    AX_IP="135.181.133.250"
    AX_PW="semuc49U__fFg4"
    CONTAINER="${CONTAINER:-abacus-worker}"

    echo "[1/3] Copy files to AX102..."
    docker cp "$CORE" skypilot:/tmp/sre_watchdog.sh
    docker cp "$TOOL_SCRIPT" skypilot:/tmp/sre_${TOOL}_watchdog.sh
    docker exec skypilot bash -c "sshpass -p '${AX_PW}' scp -o StrictHostKeyChecking=no \
        /tmp/sre_watchdog.sh /tmp/sre_${TOOL}_watchdog.sh root@${AX_IP}:/tmp/" < /dev/null

    echo "[2/3] Write .sre_env + copy into container..."
    docker exec skypilot bash -c "sshpass -p '${AX_PW}' ssh -T -o StrictHostKeyChecking=no root@${AX_IP} '
        docker cp /tmp/sre_watchdog.sh ${CONTAINER}:/workspace/
        docker cp /tmp/sre_${TOOL}_watchdog.sh ${CONTAINER}:/workspace/
        docker exec ${CONTAINER} bash -c \"chmod +x /workspace/sre_*.sh\"
        cat > /tmp/.sre_env << EEOF
SRE_BOT_TOKEN=\"${SRE_BOT_TOKEN}\"
SRE_CHAT_ID=\"${SRE_CHAT_ID}\"
SRE_LABEL=\"${LABEL}\"
SRE_TOOL=\"${TOOL}\"
EEOF
        docker cp /tmp/.sre_env ${CONTAINER}:/workspace/.sre_env
    ' < /dev/null"

    echo "[3/3] Start watchdog..."
    docker exec skypilot bash -c "sshpass -p '${AX_PW}' ssh -T -o StrictHostKeyChecking=no root@${AX_IP} '
        docker exec ${CONTAINER} bash -c \"
            if [ -f /tmp/sre_watchdog.pid ]; then
                kill \\\$(cat /tmp/sre_watchdog.pid) 2>/dev/null || true
                rm -f /tmp/sre_watchdog.pid /tmp/sre_state
            fi
            nohup bash /workspace/sre_${TOOL}_watchdog.sh > /dev/null 2>&1 & disown
            sleep 2
            echo PID: \\\$(cat /tmp/sre_watchdog.pid 2>/dev/null || echo FAILED)
        \"
    ' < /dev/null"
    echo "=== DONE: ${LABEL} ==="
    exit 0
fi

# --- Vast.ai instances ---
echo "[1/3] Copy files to host..."
docker cp "$CORE" skypilot:/tmp/sre_watchdog.sh
docker cp "$TOOL_SCRIPT" skypilot:/tmp/sre_${TOOL}_watchdog.sh
docker exec skypilot bash -c "scp -o StrictHostKeyChecking=no -P ${PORT} \
    /tmp/sre_watchdog.sh /tmp/sre_${TOOL}_watchdog.sh root@${NODE}.vast.ai:/workspace/" < /dev/null

echo "[2/3] Write .sre_env..."
docker exec skypilot bash -c "ssh -T -o StrictHostKeyChecking=no -p ${PORT} root@${NODE}.vast.ai '
    cat > /workspace/.sre_env << EEOF
SRE_BOT_TOKEN=\"${SRE_BOT_TOKEN}\"
SRE_CHAT_ID=\"${SRE_CHAT_ID}\"
SRE_LABEL=\"${LABEL}\"
SRE_TOOL=\"${TOOL}\"
EEOF
    chmod +x /workspace/sre_*.sh
' < /dev/null"

echo "[3/3] Start watchdog..."
docker exec skypilot bash -c "ssh -T -o StrictHostKeyChecking=no -p ${PORT} root@${NODE}.vast.ai '
    if [ -f /tmp/sre_watchdog.pid ]; then
        kill \$(cat /tmp/sre_watchdog.pid) 2>/dev/null || true
        rm -f /tmp/sre_watchdog.pid /tmp/sre_state
    fi
    nohup bash /workspace/sre_${TOOL}_watchdog.sh > /dev/null 2>&1 & disown
    sleep 2
    echo \"PID: \$(cat /tmp/sre_watchdog.pid 2>/dev/null || echo FAILED)\"
' < /dev/null"

echo "=== DONE: ${LABEL} ==="
