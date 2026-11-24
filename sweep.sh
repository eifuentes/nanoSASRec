#!/usr/bin/env bash
# sweep.sh — Launch all benchmark runs on Modal
#
# Usage: bash sweep.sh            # launch every run
#        bash sweep.sh --from 2   # start at run 2 (1-based), e.g. after launching run 1 alone as a warm-up
set -euo pipefail

LOG_DIR="logs/sweep"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sweep_$(date +%Y%m%d_%H%M%S).log"
echo "Sweep launched at $(date)" | tee "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

declare -a COMMANDS=(
    # v5: depth sweep on ML-20M. The longest run goes first, so it starts earliest and
    # warms the data volume (remote.py commits right after preprocessing).
    "python -u train_v4.py --dataset ml-20m --dim 256 --heads 8 --blocks 16 --num-negatives 512 --epochs 50 --eval-every 10"
    "python -u train_v4.py --dataset ml-20m --dim 256 --heads 8 --blocks 2 --num-negatives 512 --epochs 50 --eval-every 10"
    "python -u train_v4.py --dataset ml-20m --dim 256 --heads 8 --blocks 4 --num-negatives 512 --epochs 50 --eval-every 10"
    "python -u train_v4.py --dataset ml-20m --dim 256 --heads 8 --blocks 8 --num-negatives 512 --epochs 50 --eval-every 10"
    # v4: ML-1M baseline
    "python -u train_v4.py --dataset ml-1m"
    # v6: ReZero
    "python -u train_v6.py --dataset ml-20m"
    # v7: relative bias, plus two extra seeds to measure run-to-run variance
    "python -u train_v7.py --dataset ml-20m"
    "python -u train_v7.py --dataset ml-20m --seed 43"
    "python -u train_v7.py --dataset ml-20m --seed 44"
    # v8: HSTU
    "python -u train_v8.py --dataset ml-20m"
    # v9: HSTU + phi1 SwishLayerNorm, plus two extra seeds
    "python -u train_v9.py --dataset ml-20m"
    "python -u train_v9.py --dataset ml-20m --seed 43"
    "python -u train_v9.py --dataset ml-20m --seed 44"
    # v10: training alignment (temperature + L2 norm + collision masking)
    "python -u train_v10.py --dataset ml-20m"
)

START=0
if [[ "${1:-}" == "--from" ]]; then
    START=$(( ${2:?--from needs a run number} - 1 ))
fi

for cmd in "${COMMANDS[@]:START}"; do
    echo "Launching: $cmd" | tee -a "$LOG_FILE"
    uv run modal run --detach remote.py --cmd "$cmd" 2>&1 | tee -a "$LOG_FILE"
    echo "---" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "All runs launched. Check Modal dashboard for results." | tee -a "$LOG_FILE"
echo "Log saved to $LOG_FILE"