#!/usr/bin/env bash
# train_v5.sh - Depth scaling experiment on ML-20M
#
# Runs train_v4.py at 2, 4, 8, and 16 attention blocks to measure
# depth scaling on the ML-20M benchmark setup used in the README.
#
# With "wiring" as the second argument, it instead runs the block-wiring
# comparison from CURRICULUM v5 and v6: the SASRec authors' block
# (--block-style original) at the same four depths, then 3-epoch runs at
# 8 blocks for each single change and for the original block with ReZero.
# The collapse shows from epoch 1, so 3 epochs are enough for those.
#
# Usage: bash train_v5.sh [device] [wiring]
# Example: bash train_v5.sh mps
#          bash train_v5.sh cuda
#          bash train_v5.sh cuda wiring
set -euo pipefail

DEVICE="${1:-cpu}"
MODE="${2:-depth}"
DATASET="ml-20m"
DIM=256
HEADS=8
BATCH_SIZE=128
EPOCHS=50
EVAL_EVERY=10
NUM_NEGATIVES=512
SEED=42
LOG_DIR="logs/v5"

mkdir -p "$LOG_DIR"

echo "============================================================="
echo "  v5: Depth scaling experiment on $DATASET"
echo "  Device: $DEVICE | Dim: $DIM | Epochs: $EPOCHS | Negatives: $NUM_NEGATIVES | Seed: $SEED"
echo "============================================================="

# Step 0: Download and preprocess
echo ""
echo "[Step 0/5] Preprocessing $DATASET..."
uv run python -u data.py --dataset "$DATASET"

# Run train_v4.py once. $1 names the log file; the remaining arguments are extra flags.
run() {
    local NAME="$1"; shift
    local LOG_FILE="$LOG_DIR/${DATASET}_${NAME}.log"
    echo ""
    echo "============================================================"
    echo "  Training $NAME: $*"
    echo "  Logging to: $LOG_FILE"
    echo "============================================================"

    uv run python -u train_v4.py \
        --dataset "$DATASET" \
        --device "$DEVICE" \
        --dim "$DIM" \
        --heads "$HEADS" \
        --batch-size "$BATCH_SIZE" \
        --num-negatives "$NUM_NEGATIVES" \
        --seed "$SEED" \
        "$@" \
        2>&1 | tee "$LOG_FILE"
}

if [[ "$MODE" == "wiring" ]]; then
    for BLOCKS in 2 4 8 16; do
        run "original_blocks${BLOCKS}" --blocks "$BLOCKS" --block-style original --epochs "$EPOCHS" --eval-every "$EVAL_EVERY"
    done
    # Which of the original block's two changes causes the collapse, and does ReZero prevent it?
    for STYLE in ln-residual raw-kv pre-ln; do
        run "${STYLE}_blocks8_3ep" --blocks 8 --block-style "$STYLE" --epochs 3 --eval-every 3 --checkpoint-dir checkpoints/v5-wiring
    done
    run "original_rezero_blocks8_3ep" --blocks 8 --block-style original --scale-res --epochs 3 --eval-every 3
else
    for BLOCKS in 2 4 8 16; do
        run "blocks${BLOCKS}" --blocks "$BLOCKS" --epochs "$EPOCHS" --eval-every "$EVAL_EVERY"
    done
fi

echo ""
echo "========================================================================="
echo "  Done! Logs saved to $LOG_DIR/"
echo "  Compare final metrics across block counts to measure depth scaling."
echo "========================================================================="
