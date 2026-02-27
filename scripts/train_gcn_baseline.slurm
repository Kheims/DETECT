#!/bin/bash
#SBATCH --job-name=mlcq-gcn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

uv run python scripts/train_gcn_baseline.py \
  --dataset-path data/processed/dataset_v2.pt \
  --epochs 40 \
  --patience 10 \
  --hidden-dim 128 \
  --type-emb-dim 64 \
  --max-nodes-per-batch 45000 \
  --eval-max-nodes-per-batch 70000 \
  --grad-accum-steps 1 \
  --num-workers 4 \
  --pin-memory \
  --device cuda
