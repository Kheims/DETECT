#!/bin/bash
# Zero-shot LLM smell judge with vLLM on one RTX A6000, then its evaluation when the run is not sharded.
# The first argument is a judge name of llm_judge.judges, the others are config overrides.
#   sbatch scripts/slurm/llm_judge.sbatch qwen25_coder_7b_instruct
#   sbatch --array=0-1 scripts/slurm/llm_judge.sbatch qwen36_27b_fp8   # 2 shards, run eval_llm_judge.py afterwards
#
#SBATCH --job-name=llm_judge
#SBATCH --partition=all
#SBATCH --constraint=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/llm_judge_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

JUDGE=${1:?usage: llm_judge.sbatch <judge> [--section.key=value ...]}
shift
CONFIG=${CONFIG:-config/experiments/llm_judge.yml}

B=/mnt/beegfs/projects/code_smells_detection/vllm
PYTHON=$B/venv/bin/python
export HF_HOME=$B/hf VLLM_CACHE_ROOT=$B/vllm-cache TRITON_CACHE_DIR=$B/triton-cache XDG_CACHE_HOME=$B/cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# FlashInfer sampling kernels fail to build with the node nvcc, and first-token scoring does not need them
export VLLM_USE_FLASHINFER_SAMPLER=0
# scikit-learn and scipy for mlcq_graphs.evaluation (not in the vLLM venv)
export PYTHONPATH=$HOME/code_smells_detection/fe_relational/pylibs312${PYTHONPATH:+:$PYTHONPATH}

SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}
# SHARD_COUNT from the environment wins, to resubmit one shard alone
SHARD_COUNT=${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} judge=$JUDGE shard=$SHARD_INDEX/$SHARD_COUNT" \
  "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset} overrides=$*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"$PYTHON" scripts/llm/llm_judge.py --config "$CONFIG" --judge "$JUDGE" \
  --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" "$@"

# evaluation in the graph env (sklearn and matplotlib, no GPU needed)
EVAL_PYTHON=${EVAL_PYTHON:-$HOME/miniconda3/envs/detect-fe/bin/python}
if [ "$SHARD_COUNT" -eq 1 ]; then
  PYTHONPATH= "$EVAL_PYTHON" scripts/llm/eval_llm_judge.py --config "$CONFIG" --judge "$JUDGE" "$@"
fi
