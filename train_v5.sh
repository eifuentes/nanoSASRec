#!/usr/bin/env bash
# train_v5.sh - Depth scaling and block wiring on ML-20M
#
# Runs train_v4.py on the ML-20M benchmark setup used in the README, in one pass:
#   1. Which wire collapses the SASRec authors' block? 3 epochs each at 8 blocks:
#      pre-LN, then each of the authors' two changes on its own. The collapse shows
#      from epoch 1, so 3 epochs are enough, and these finish first (~15 min each).
#   2. Depth scaling at 2, 4, 8 and 16 blocks, for pre-LN (the default) and for the
#      authors' block (both wiring flags flipped), 50 epochs each. Shallow runs go
#      first, so the 2- and 4-block comparisons are done before the long runs start.
# See CURRICULUM v5 for the results.
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
NUM_NEGATIVES=512
SEED=42
LOG_DIR="logs/v5"

# The SASRec authors' block flips both of train_v4.py's wiring flags (CURRICULUM v4)
AUTHORS_CODE=(--attention-block-residual-from normed --attention-block-kv-from input)

mkdir -p "$LOG_DIR"

echo "============================================================="
echo "  v5: Depth scaling and block wiring on $DATASET"
echo "  Device: $DEVICE | Dim: $DIM | Epochs: $EPOCHS | Negatives: $NUM_NEGATIVES | Seed: $SEED"
echo "============================================================="

# Step 0: Download and preprocess
echo ""
echo "[Step 0] Preprocessing $DATASET..."
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

# Step 1: which wire collapses the authors' block. The run directory doesn't depend on
# --epochs, so these short runs keep their checkpoints apart from the 50-epoch runs below.
SHORT=(--blocks 8 --epochs 3 --eval-every 3 --checkpoint-dir checkpoints/v5-3-epochs)
run "pre-ln_blocks8_3-epochs" "${SHORT[@]}"
run "residual-from-normed_blocks8_3-epochs" "${SHORT[@]}" --attention-block-residual-from normed
run "kv-from-input_blocks8_3-epochs" "${SHORT[@]}" --attention-block-kv-from input

# Step 2: depth scaling, pre-LN and the authors' block at each depth
for BLOCKS in 2 4 8 16; do
    run "blocks${BLOCKS}" --blocks "$BLOCKS" --epochs "$EPOCHS" --eval-every "$EVAL_EVERY"
    run "authors-code_blocks${BLOCKS}" --blocks "$BLOCKS" --epochs "$EPOCHS" --eval-every "$EVAL_EVERY" "${AUTHORS_CODE[@]}"
done

echo ""
echo "========================================================================="
echo "  Done! Logs saved to $LOG_DIR/"
echo "  Compare final metrics across block counts and wirings."
echo "========================================================================="
