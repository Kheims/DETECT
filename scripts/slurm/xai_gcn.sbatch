#!/bin/bash
# GCN interpretability study on one V100 (Integrated Gradients, deletion, embeddings, k-NN neighbours)
# with the reference GCN checkpoints of seed 42 (artifacts/runs/<run>/).
#   sbatch --array=0-4 scripts/slurm/xai_gcn.sbatch      # one fold per task, outputs in fold<f>/
#   python scripts/xai/xai_gcn.py --merge                       # afterwards, on the login node
#
#SBATCH --job-name=xai_gcn
#SBATCH --partition=all
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/xai_gcn_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/detect-fe/bin/python}"
FOLD_ARGS=()
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] && FOLD_ARGS=(--fold "$SLURM_ARRAY_TASK_ID")
"$PYTHON" scripts/xai/xai_gcn.py --runs_dir artifacts/runs --device cuda "${FOLD_ARGS[@]}" "$@"
