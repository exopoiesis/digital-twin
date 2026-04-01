#!/bin/bash
# sre_mattergen_watchdog.sh -- MatterGen/ML pipeline watchdog
# Only common checks: process alive, disk, log staleness.
#
# Deploy: copy sre_watchdog.sh + this file to /workspace/, run:
#   nohup bash /workspace/sre_mattergen_watchdog.sh > /dev/null 2>&1 & disown

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/sre_watchdog.sh"

# No tool-specific checks -- common checks are sufficient for ML pipelines
# (process alive, disk space, log staleness)

sre_main
