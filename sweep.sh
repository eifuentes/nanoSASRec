#!/usr/bin/env bash
# sweep.sh — Launch all benchmark runs on Modal
set -euo pipefail

LOG_DIR="logs/sweep"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sweep_$(date +%Y%m%d_%H%M%S).log"
echo "Sweep launched at $(date)" | tee "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

declare -a COMMANDS=(
    # v4: ML-1M baseline
    "python -u train_v4.py --dataset ml-1m"
    # v5: depth sweep on ML-20M
    "python -u train_v4.py --dataset ml-20m --dim 256 --heads 8 --blocks 2 --num-negatives 512 --epochs 50 --eval-every 10"
    "python -u train_v4.py --dataset ml-20m --dim 256 --heads 8 --blocks 4 --num-negatives 512 --epochs 50 --eval-every 10"
    "python -u train_v4.py --dataset ml-20m --dim 256 --heads 8 --blocks 8 --num-negatives 512 --epochs 50 --eval-every 10"
    "python -u train_v4.py --dataset ml-20m --dim 256 --heads 8 --blocks 16 --num-negatives 512 --epochs 50 --eval-every 10"
    # v6: ReZero
    "python -u train_v6.py --dataset ml-20m"
    # v7: relative bias
    "python -u train_v7.py --dataset ml-20m"
    # v8: HSTU
    "python -u train_v8.py --dataset ml-20m"
)

for cmd in "${COMMANDS[@]}"; do
    echo "Launching: $cmd" | tee -a "$LOG_FILE"
    uv run modal run --detach remote.py --cmd "$cmd" 2>&1 | tee -a "$LOG_FILE"
    echo "---" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "All runs launched. Check Modal dashboard for results." | tee -a "$LOG_FILE"
echo "Log saved to $LOG_FILE"