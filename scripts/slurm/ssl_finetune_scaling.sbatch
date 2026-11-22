#!/bin/bash
# Runs of one seed (42 + array task id) on one node with 4 GPUs, one after the other:
# from random weights on 1 GPU, then from the pre-trained encoder on 1, 2 and 4 GPUs, then from
# the encoder of the first pre-training epoch on 1 GPU.
#   CONFIG=config/experiments/ssl_finetune.yml sbatch --array=0-4%1 --partition=all --qos=default -w n101 \
#     --gres=gpu:4 scripts/slurm/ssl_finetune_scaling.sbatch [--key=value ...]
# The 4 GPUs of the node are reserved for the whole job, so the timings of a run do not depend on
# other jobs. The use of each GPU is sampled every second into logs/gpu_<job>_<task>_<run>.csv.
#
#SBATCH --job-name=ssl_finetune_scaling
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
PYTHON="${DETECT_PYTHON:-$HOME/miniconda3/envs/detect-fe/bin/python}"
SEED=$((42 + ${SLURM_ARRAY_TASK_ID:-0}))
JOB="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
EXTRA=("$@")
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "host=$(hostname) config=$CONFIG seed=$SEED gpus=${GPUS[*]} start=$(date -Is)"
nvidia-smi topo -m || true

# run <label> <number of GPUs> [--key=value ...]
run() {
  local label=$1 ngpu=$2
  shift 2
  local visible
  visible=$(IFS=,; echo "${GPUS[*]:0:$ngpu}")
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,nounits -l 1 \
    > "logs/gpu_${JOB}_${label}.csv" &
  local logger=$!
  echo "run=$label ngpu=$ngpu visible=$visible start=$(date -Is)"
  CUDA_VISIBLE_DEVICES="$visible" "$PYTHON" main.py --config "$CONFIG" \
    --reporting.enabled=false \
    --training.seed="$SEED" \
    --training.distributed.nproc_per_node="$ngpu" "$@" ${EXTRA[@]+"${EXTRA[@]}"}
  kill "$logger" 2>/dev/null || true
  echo "run=$label end=$(date -Is)"
}

run scratch_1gpu 1 --pretraining.enabled=false --training.init_from_pretraining=false
run pretrained_1gpu 1
run pretrained_2gpu 2
run pretrained_4gpu 4
run pretrained_epoch1_1gpu 1 --training.pretrained_checkpoint=encoder_epoch1.pt
echo "end=$(date -Is)"
