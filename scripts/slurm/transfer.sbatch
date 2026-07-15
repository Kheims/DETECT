#!/bin/bash
# Cross-dataset evaluation of the models of CONFIG on TARGET_JSON.
#
#   CONFIG=config/experiments/ts_baseline.yml TARGET_JSON=data/... TARGET_NAME=smellycode \
#     sbatch scripts/slurm/transfer.sbatch [--key=value ...]
#
#SBATCH --job-name=detect_transfer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
PYTHON="${DETECT_PYTHON:-$HOME/miniconda3/envs/detect-fe/bin/python}"
echo "host=$(hostname) config=$CONFIG target=$TARGET_JSON start=$(date -Is)"
"$PYTHON" scripts/analysis/eval_transfer.py --config "$CONFIG" --target-json "$TARGET_JSON" \
  --target-name "$TARGET_NAME" --training.device=cuda "$@"
echo "end=$(date -Is)"
