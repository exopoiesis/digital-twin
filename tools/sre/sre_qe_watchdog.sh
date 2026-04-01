#!/bin/bash
# SRE Watchdog for Quantum ESPRESSO on Vast.ai
# Usage: bash infra/sre/sre_qe_watchdog.sh <ssh_node> <ssh_port> <label>
# Example: bash infra/sre/sre_qe_watchdog.sh ssh5 34572 W3-QE

NODE=${1:?Usage: sre_qe_watchdog.sh <node> <port> <label>}
PORT=${2:?}
LABEL=${3:-QE}

SSH="docker exec skypilot ssh -T -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $PORT root@$NODE.vast.ai"

echo "=== SRE QE Watchdog: $LABEL ($NODE:$PORT) ==="
echo "Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

echo ""
echo "--- CHECK-Q0: Process alive ---"
$SSH 'ps aux | grep -E "pw\.x|neb\.x" | grep -v grep | head -3' < /dev/null 2>&1
ALIVE=$($SSH 'ps aux | grep -E "pw\.x|neb\.x" | grep -v grep | wc -l' < /dev/null 2>&1 | tr -d '[:space:]')
if [ "$ALIVE" = "0" ]; then
    echo "!!! NO QE PROCESS RUNNING !!!"
    echo "--- Check for DONE or crash ---"
    $SSH 'ls -la /workspace/results/DONE* 2>/dev/null; tail -5 /workspace/results/*stderr* 2>/dev/null' < /dev/null 2>&1
    echo "=== STATUS: DEAD ==="
    exit 1
fi

echo ""
echo "--- CHECK-Q1: SCF progress ---"
$SSH '
OUTFILE=$(find /workspace -name "pwscf.out" -o -name "*.out" 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
if [ -n "$OUTFILE" ]; then
    echo "File: $OUTFILE"
    echo "SCF iterations:"
    grep "iteration #" "$OUTFILE" | tail -3
    echo "SCF accuracy:"
    grep "estimated scf accuracy" "$OUTFILE" | tail -5
    echo "Converged SCF blocks:"
    grep -c "convergence has been achieved" "$OUTFILE" 2>/dev/null
else
    echo "No output file found"
fi
' < /dev/null 2>&1

echo ""
echo "--- CHECK-Q2: Magnetism (AFM check) ---"
$SSH '
OUTFILE=$(find /workspace -name "pwscf.out" -o -name "*.out" 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
if [ -n "$OUTFILE" ]; then
    echo "Total magnetization (last 5):"
    grep "total magnetization" "$OUTFILE" | tail -5
    echo "Absolute magnetization (last 3):"
    grep "absolute magnetization" "$OUTFILE" | tail -3
fi
' < /dev/null 2>&1

echo ""
echo "--- CHECK-Q3: Energy convergence ---"
$SSH '
OUTFILE=$(find /workspace -name "pwscf.out" -o -name "*.out" 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
if [ -n "$OUTFILE" ]; then
    echo "Total energies (last 5 converged):"
    grep "!" "$OUTFILE" | tail -5
    echo "Forces (last 3):"
    grep "Total force" "$OUTFILE" | tail -3
    echo "BFGS status:"
    grep -E "bfgs converged|number of bfgs steps" "$OUTFILE" | tail -3
fi
' < /dev/null 2>&1

echo ""
echo "--- CHECK-Q4: NEB progress (if neb.x) ---"
$SSH '
NEBFILE=$(find /workspace -name "neb.out" -o -name "*neb*.out" 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
if [ -n "$NEBFILE" ]; then
    echo "File: $NEBFILE"
    echo "NEB convergence:"
    grep -E "activation energy|neb: convergence|path_thr" "$NEBFILE" | tail -5
    echo "Image energies:"
    grep "image.*energy" "$NEBFILE" | tail -10
else
    echo "No NEB output (pw.x mode or not started yet)"
fi
' < /dev/null 2>&1

echo ""
echo "--- CHECK-Q5: Walltime ---"
$SSH '
OUTFILE=$(find /workspace -name "pwscf.out" -o -name "*.out" 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
if [ -n "$OUTFILE" ]; then
    grep "PWSCF.*WALL\|WALL$" "$OUTFILE" | tail -3
    echo "File size:"
    ls -lh "$OUTFILE" 2>/dev/null | awk "{print \$5}"
fi
' < /dev/null 2>&1

echo ""
echo "--- CHECK-Q6: Errors/Warnings ---"
$SSH '
OUTFILE=$(find /workspace -name "pwscf.out" -o -name "*.out" 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
if [ -n "$OUTFILE" ]; then
    grep -i "error\|not converged\|charge is wrong\|S matrix" "$OUTFILE" | tail -5
fi
STDERRFILE=$(find /workspace/results -name "*stderr*" 2>/dev/null | head -1)
if [ -n "$STDERRFILE" ]; then
    echo "Stderr:"
    tail -5 "$STDERRFILE"
fi
' < /dev/null 2>&1

echo ""
echo "--- CHECK-Q7: Disk & Memory ---"
$SSH '
df -h /workspace | tail -1
free -m | grep Mem
' < /dev/null 2>&1

echo ""
echo "--- VERDICT ---"

# Parse key metrics for verdict
ACCURACY=$($SSH 'OUTFILE=$(find /workspace -name "pwscf.out" -o -name "*.out" 2>/dev/null | xargs ls -t 2>/dev/null | head -1); grep "estimated scf accuracy" "$OUTFILE" 2>/dev/null | tail -1 | awk "{print \$NF}"' < /dev/null 2>&1 | tr -d '[:space:]')
TOTMAG=$($SSH 'OUTFILE=$(find /workspace -name "pwscf.out" -o -name "*.out" 2>/dev/null | xargs ls -t 2>/dev/null | head -1); grep "total magnetization" "$OUTFILE" 2>/dev/null | tail -1 | awk "{print \$4}"' < /dev/null 2>&1 | tr -d '[:space:]')

echo "  SCF accuracy: $ACCURACY Ry"
echo "  Total mag: $TOTMAG Bohr mag"

# Simple verdict logic
if [ -n "$TOTMAG" ]; then
    MAG_ABS=$(echo "$TOTMAG" | sed 's/-//')
    if echo "$MAG_ABS" | awk '{exit ($1 > 1.0) ? 0 : 1}' 2>/dev/null; then
        echo "  !!! CRITICAL: AFM COLLAPSED (|Total mag| > 1.0) !!!"
    fi
fi

echo "=== END SRE QE: $LABEL ==="
