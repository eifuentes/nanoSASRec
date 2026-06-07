#!/usr/bin/env bash
# train_v5.sh - Depth scaling experiment on ML-20M
#
# Runs train_v4.py at 2, 4, and 6 attention blocks to demonstrate
# that SASRec metrics plateau or degrade at 4+ blocks due to
# over-smoothing (Peng et al. 2022, Guo et al. 2024).
#
# Usage: bash train_v5.sh [device]
# Example: bash train_v5.sh mps
#          bash train_v5.sh cuda
set -euo pipefail

DEVICE="${1:-cpu}"
DATASET="ml-20m"
DIM=256
HEADS=8
BATCH_SIZE=128
EPOCHS=50
EVAL_EVERY=10
SEED=42
LOG_DIR="logs/v5"

mkdir -p "$LOG_DIR"

echo "============================================================="
echo "  v5: Depth scaling experiment on $DATASET"
echo "  Device: $DEVICE | Dim: $DIM | Epochs: $EPOCHS | Seed: $SEED"
echo "============================================================="

# Step 0: Download and preprocess
echo ""
echo "[Step 0/3] Preprocessing $DATASET..."
uv run python -u data.py --dataset "$DATASET"

for BLOCKS in 2 4 8 16; do
    LOG_FILE="$LOG_DIR/${DATASET}_blocks${BLOCKS}.log"
    echo ""
    echo "============================================================"
    echo "  [Step ${BLOCKS}/6] Training with --blocks $BLOCKS"
    echo "  Logging to: $LOG_FILE"
    echo "============================================================"

    uv run python -u train_v4.py \
        --dataset "$DATASET" \
        --device "$DEVICE" \
        --dim "$DIM" \
        --blocks "$BLOCKS" \
        --heads "$HEADS" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --eval-every "$EVAL_EVERY" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_FILE"
done

echo ""
echo "========================================================================="
echo "  Done! Logs saved to $LOG_DIR/"
echo "  Compare final metrics across block counts to see measure depth scaling."
echo "========================================================================="