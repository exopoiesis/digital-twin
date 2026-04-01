#!/bin/bash
# Проверка Vast.ai инстансов через loki
echo "=== SKYPILOT ==="
docker ps --filter name=skypilot --format "{{.Names}}\t{{.Status}}" 2>&1

echo "=== VAST INSTANCES ==="
docker exec skypilot bash -c "vastai show instances 2>/dev/null | head -30"
