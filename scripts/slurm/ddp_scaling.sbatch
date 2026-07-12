#!/bin/bash
# Single-node DDP scaling benchmark of the pre-training (1, 2, 4 GPUs).
#
#   CONFIG=config/experiments/ssl_scaling.yml sbatch --qos=nvlink --partition=nvlink --gres=gpu:4 scripts/slurm/ddp_scaling.sbatch [--key=value ...]
#
#SBATCH --job-name=detect_ddp_scaling
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
PYTHON="${DETECT_PYTHON:-$HOME/miniconda3/envs/detect-fe/bin/python}"
echo "host=$(hostname) config=$CONFIG start=$(date -Is)"
nvidia-smi topo -m || true
"$PYTHON" scripts/pretraining/pretrain_scaling.py --config "$CONFIG" "$@"
echo "end=$(date -Is)"
