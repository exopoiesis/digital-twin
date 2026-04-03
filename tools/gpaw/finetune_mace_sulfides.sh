#!/bin/bash
# MACE frozen transfer learning for iron sulfides
# Based on Radova et al. 2025 (L-556): freeze bottom 4/6 interaction layers
# Container: infra-mace-neb (mace-torch 0.3.15, PyTorch 2.5.1+cu121)
# Usage: docker exec -it <container> bash /workspace/scripts/finetune_mace_sulfides.sh

set -e

WORKDIR=/workspace
TRAIN_FILE=${WORKDIR}/results/sulfide_train.xyz
VALID_FILE=${WORKDIR}/results/sulfide_valid.xyz
OUTPUT_DIR=${WORKDIR}/results/mace_sulfide_ft
SEED=42

echo "========================================" | tee -a ${WORKDIR}/finetune.log
echo "MACE Sulfide Fine-tuning" | tee -a ${WORKDIR}/finetune.log
echo "Started: $(date)" | tee -a ${WORKDIR}/finetune.log
echo "========================================" | tee -a ${WORKDIR}/finetune.log
echo ""

# Check for training data
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "ERROR: Training file not found: ${TRAIN_FILE}" | tee -a ${WORKDIR}/finetune.log
    echo "Expected location: /workspace/results/sulfide_train.xyz" | tee -a ${WORKDIR}/finetune.log
    echo "Please copy DFT training data to container before running." | tee -a ${WORKDIR}/finetune.log
    exit 1
fi

mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/logs

# Count structures in training file
N_TRAIN=$(grep -c "Lattice=" ${TRAIN_FILE} || echo "0")
echo "Training structures: ${N_TRAIN}" | tee -a ${WORKDIR}/finetune.log

# Check if validation file exists, otherwise will use --valid_fraction
if [ -f "${VALID_FILE}" ]; then
    N_VALID=$(grep -c "Lattice=" ${VALID_FILE} || echo "0")
    echo "Validation structures: ${N_VALID}" | tee -a ${WORKDIR}/finetune.log
    VALID_ARG="--test_file=${VALID_FILE}"
else
    echo "No separate validation file, using --valid_fraction=0.1" | tee -a ${WORKDIR}/finetune.log
    VALID_ARG=""
fi

echo ""
echo "Configuration:" | tee -a ${WORKDIR}/finetune.log
echo "  Foundation model: small (MACE-MP-0)" | tee -a ${WORKDIR}/finetune.log
echo "  Frozen layers: 4/6 interaction layers" | tee -a ${WORKDIR}/finetune.log
echo "  Energy weight: 1.0" | tee -a ${WORKDIR}/finetune.log
echo "  Forces weight: 100.0" | tee -a ${WORKDIR}/finetune.log
echo "  Stress weight: 10.0" | tee -a ${WORKDIR}/finetune.log
echo "  Batch size: 4" | tee -a ${WORKDIR}/finetune.log
echo "  Max epochs: 200" | tee -a ${WORKDIR}/finetune.log
echo "  Learning rate: 0.001" | tee -a ${WORKDIR}/finetune.log
echo "  Patience: 20 epochs" | tee -a ${WORKDIR}/finetune.log
echo ""
echo "Starting training..." | tee -a ${WORKDIR}/finetune.log
echo ""

# Launch MACE training with frozen transfer learning
python -u -m mace.cli.run_train \
    --name="mace_sulfide_ft" \
    --foundation_model="small" \
    --train_file="${TRAIN_FILE}" \
    --valid_fraction=0.1 \
    ${VALID_ARG} \
    --energy_key="energy" \
    --forces_key="forces" \
    --stress_key="stress" \
    --E0s="average" \
    --freeze=4 \
    --loss="weighted" \
    --energy_weight=1.0 \
    --forces_weight=100.0 \
    --stress_weight=10.0 \
    --batch_size=4 \
    --max_num_epochs=200 \
    --lr=0.001 \
    --patience=20 \
    --scheduler="ReduceLROnPlateau" \
    --scheduler_patience=10 \
    --ema \
    --ema_decay=0.99 \
    --device=cuda \
    --seed=${SEED} \
    --save_cpu \
    --model_dir="${OUTPUT_DIR}" \
    --results_dir="${OUTPUT_DIR}" \
    --default_dtype="float64" \
    2>&1 | tee ${OUTPUT_DIR}/training.log

EXIT_CODE=$?

echo ""
echo "========================================" | tee -a ${WORKDIR}/finetune.log
echo "Training completed: $(date)" | tee -a ${WORKDIR}/finetune.log
echo "Exit code: ${EXIT_CODE}" | tee -a ${WORKDIR}/finetune.log
echo "========================================" | tee -a ${WORKDIR}/finetune.log

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "SUCCESS: Model saved to ${OUTPUT_DIR}/" | tee -a ${WORKDIR}/finetune.log
    echo ""
    echo "Files created:" | tee -a ${WORKDIR}/finetune.log
    ls -lh ${OUTPUT_DIR}/*.model 2>/dev/null | tee -a ${WORKDIR}/finetune.log || echo "  (no .model files found)" | tee -a ${WORKDIR}/finetune.log
    echo ""
    echo "Next steps:" | tee -a ${WORKDIR}/finetune.log
    echo "  1. Run validation: python -u /workspace/scripts/validate_mace_sulfide.py" | tee -a ${WORKDIR}/finetune.log
    echo "  2. Copy model to host: docker cp <container>:${OUTPUT_DIR} ./results/" | tee -a ${WORKDIR}/finetune.log
    echo ""
else
    echo ""
    echo "FAILED: Training exited with code ${EXIT_CODE}" | tee -a ${WORKDIR}/finetune.log
    echo "Check logs at ${OUTPUT_DIR}/training.log" | tee -a ${WORKDIR}/finetune.log
    echo ""
    exit ${EXIT_CODE}
fi
