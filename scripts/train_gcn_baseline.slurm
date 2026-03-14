#!/bin/bash
#SBATCH --job-name=mlcq-gcn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

# Number of GPUs auto-detected from SLURM allocation
NGPUS=${SLURM_GPUS_ON_NODE:-1}

if [ "$NGPUS" -gt 1 ]; then
    # Multi-GPU: use torchrun with train_ddp.py
    uv run torchrun \
      --standalone --nnodes=1 \
      --nproc_per_node="$NGPUS" \
      scripts/train_ddp.py \
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
else
    # Single-GPU: use baseline script directly
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
fi
