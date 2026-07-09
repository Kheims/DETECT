#!/bin/bash
# Build the cached stages (construction, dataset, token encoder) of a config
# once, before an array of training shards reuses them.
#
#   CONFIG=config/experiments/x.yml sbatch scripts/slurm/prepare.sbatch [--key=value ...]
#
#SBATCH --job-name=detect_prepare
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
PYTHON="${DETECT_PYTHON:-$HOME/miniconda3/envs/detect-fe/bin/python}"
echo "host=$(hostname) config=$CONFIG start=$(date -Is)"
"$PYTHON" main.py --config "$CONFIG" --ablation.enabled=false --training.enabled=false "$@"
echo "end=$(date -Is)"
