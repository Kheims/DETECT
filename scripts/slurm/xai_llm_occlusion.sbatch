#!/bin/bash
# Line occlusion of the Qwen2.5-Coder-7B-Instruct judge on one V100, phase check or run (one shard per array task).
#   sbatch scripts/slurm/xai_llm_occlusion.sbatch check
#   sbatch --array=0-3 scripts/slurm/xai_llm_occlusion.sbatch run
#
#SBATCH --job-name=xai_llm_occl
#SBATCH --partition=all
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/xai_llm_occlusion_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

PHASE=${1:?usage: xai_llm_occlusion.sbatch check|run [args]}
shift
PYTHON=${PYTHON:-$HOME/miniconda3/envs/detect-fe/bin/python}
# transformers 4.57 and scikit-learn for the torch 2.5 env, which has V100 kernels
L=$HOME/code_smells_detection/fe_relational
export PYTHONPATH=$L/pylibs_hf312:$L/pylibs312${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none} phase=$PHASE args=$*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [ "$PHASE" = run ]; then
  SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}
  SHARD_COUNT=${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}
  "$PYTHON" scripts/xai/xai_llm_occlusion.py run --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" "$@"
else
  "$PYTHON" scripts/xai/xai_llm_occlusion.py "$PHASE" "$@"
fi
