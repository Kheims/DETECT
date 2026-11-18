#!/bin/bash
# Supervised training on MLCQ with NGPU GPUs, one array task per seed (42 + task id).
#   CONFIG=config/experiments/ssl_finetune.yml NGPU=4 sbatch --array=0-4 --qos=nvlink --partition=nvlink \
#     --gres=gpu:4 scripts/slurm/ssl_finetune.sbatch [--key=value ...]
# The use of each GPU is sampled every second into logs/gpu_<job>_<task>.csv.
#
#SBATCH --job-name=ssl_finetune
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
PYTHON="${DETECT_PYTHON:-$HOME/miniconda3/envs/detect-fe/bin/python}"
NGPU="${NGPU:-1}"
SEED=$((42 + ${SLURM_ARRAY_TASK_ID:-0}))
echo "host=$(hostname) config=$CONFIG ngpu=$NGPU seed=$SEED start=$(date -Is)"
nvidia-smi topo -m || true

GPU_LOG="logs/gpu_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,nounits -l 1 > "$GPU_LOG" &
GPU_LOG_PID=$!
trap 'kill "$GPU_LOG_PID" 2>/dev/null || true' EXIT

"$PYTHON" main.py --config "$CONFIG" \
  --reporting.enabled=false \
  --training.seed="$SEED" \
  --training.distributed.nproc_per_node="$NGPU" "$@"
echo "end=$(date -Is)"
