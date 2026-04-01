#!/bin/bash
# SRE Quick Health Check - Session 60

echo "=== W3 AFM CHECK (troilite nspin=2) ==="
docker exec skypilot bash -c "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p 34572 root@ssh5.vast.ai '
echo --- RELAX_START SCF ---
ls -la /workspace/neb_work/relax_start/ 2>/dev/null | head -5
cat /workspace/neb_work/relax_start/running_scf.log 2>/dev/null | grep -i \"total magnetism\" | tail -5
echo --- RELAX_START FIRE ---
cat /workspace/neb_work/relax_start/relax.log 2>/dev/null | tail -10
echo --- STDOUT TAIL ---
tail -30 /workspace/results/troilite_neb_stdout.log 2>/dev/null
echo --- ABACUS LOG ---
find /workspace/neb_work -name "running_scf.log" -newer /workspace/neb_work -exec echo {} \; 2>/dev/null
ls -lt /workspace/neb_work/*/running_scf.log 2>/dev/null | head -5
' < /dev/null" 2>&1

echo ""
echo "=== W2 NEB CONVERGENCE TREND ==="
docker exec skypilot bash -c "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p 38390 root@ssh6.vast.ai '
echo --- FULL NEB LOG ---
cat /workspace/neb_work/neb.log 2>/dev/null
echo --- RELAX STATUS ---
tail -3 /workspace/neb_work/relax_start.log 2>/dev/null
tail -3 /workspace/neb_work/relax_end.log 2>/dev/null
' < /dev/null" 2>&1

echo ""
echo "=== W1 VACANCY PROGRESS ==="
docker exec skypilot bash -c "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p 10292 root@ssh8.vast.ai '
echo --- CURRENT CONFIG ---
tail -5 /workspace/results/vacancy_formation_stdout.log 2>/dev/null
echo --- RESUME COUNT ---
python3 -c \"
import json, os
f = \"/workspace/results/vacancy_formation_resume.json\"
if os.path.exists(f):
    d = json.load(open(f))
    print(f\"Completed: {len(d)} configs\")
    for k in sorted(d.keys())[-5:]:
        print(f\"  {k}: {d[k].get('status','?')}\")
else:
    print(\"No resume file\")
\" 2>/dev/null
' < /dev/null" 2>&1
