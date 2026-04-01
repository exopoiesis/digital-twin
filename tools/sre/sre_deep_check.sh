#!/bin/bash
# W3 troilite U-Ramping deep SCF check
docker exec skypilot bash -c "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p 34572 root@ssh5.vast.ai '
echo === SCF LOG ===
tail -40 /workspace/neb_work/relax_start/OUT.ABACUS/running_scf.log 2>/dev/null
echo === MAGNETISM ===
grep -i \"total magnetism\|TOTAL MAGNETISM\|magnetism\" /workspace/neb_work/relax_start/OUT.ABACUS/running_scf.log 2>/dev/null | tail -10
echo === U-RAMPING ===
grep -i \"uramping\|U_Ramping\|Hubbard\" /workspace/neb_work/relax_start/OUT.ABACUS/running_scf.log 2>/dev/null | tail -5
echo === CONVERGENCE ===
grep -i \"drho\|DRHO\|density\|charge\" /workspace/neb_work/relax_start/OUT.ABACUS/running_scf.log 2>/dev/null | tail -10
echo === FIRE RELAX ===
cat /workspace/neb_work/relax_start/relax.log 2>/dev/null | tail -10
echo === PROCESS ===
ps aux | grep abacus | grep -v grep | head -3
echo === UPTIME ===
ps -o pid,etime,pcpu,rss -p $(pgrep -x abacus_2p 2>/dev/null || echo 1) 2>/dev/null
' < /dev/null" 2>&1
