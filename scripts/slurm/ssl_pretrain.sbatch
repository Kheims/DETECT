#!/bin/bash
# Self-supervised pre-training on the GPUs of one node (DistributedDataParallel).
#   CONFIG=config/experiments/ssl_pretrain_methods.yml sbatch --qos=nvlink --partition=nvlink --gres=gpu:4 \
#     scripts/slurm/ssl_pretrain.sbatch --pretraining.corpus.shard.count=6
#
#SBATCH --job-name=ssl_pretrain
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
PYTHON="${DETECT_PYTHON:-$HOME/miniconda3/envs/detect-fe/bin/python}"
echo "host=$(hostname) config=$CONFIG start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# only the pretraining stage runs, the other stages are switched off
"$PYTHON" main.py --config "$CONFIG" \
  --construction.enabled=false --dataset.enabled=false --token_encoder.enabled=false \
  --training.enabled=false --reporting.enabled=false "$@"
echo "end=$(date -Is)"
