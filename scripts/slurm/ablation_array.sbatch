#!/bin/bash
# Sharded ablation over a job array: each task runs PROCS shards spread round-robin over its GPUs.
#   CONFIG=config/experiments/x.yml PROCS=3 sbatch --array=0-2 scripts/slurm/ablation_array.sbatch [--key=value ...]
#   CONFIG=... PROCS=7 sbatch --qos=pcie --partition=pcie --gres=gpu:3 --array=0-0 scripts/slurm/ablation_array.sbatch
# PREPARE=1 builds the cached stages first inside the task (single-task arrays only).
#
#SBATCH --job-name=detect_ablation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
PYTHON="${DETECT_PYTHON:-$HOME/miniconda3/envs/detect-fe/bin/python}"
PROCS="${PROCS:-1}"
COUNT=$((SLURM_ARRAY_TASK_COUNT * PROCS))
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "host=$(hostname) config=$CONFIG task=$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT procs=$PROCS gpus=${GPUS[*]} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ "${PREPARE:-0}" == "1" ]]; then
  "$PYTHON" main.py --config "$CONFIG" --ablation.enabled=false --training.enabled=false "$@" \
    > "logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_prepare.out" 2>&1
  echo "prepared=$(date -Is)"
fi

pids=()
for ((j = 0; j < PROCS; j++)); do
  INDEX=$((SLURM_ARRAY_TASK_ID * PROCS + j))
  GPU=${GPUS[$((j % ${#GPUS[@]}))]}
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" main.py --config "$CONFIG" --training.device=cuda \
    --ablation.shard.index="$INDEX" --ablation.shard.count="$COUNT" "$@" \
    > "logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_shard${INDEX}.out" 2>&1 &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
echo "end=$(date -Is) status=$status"
exit $status
