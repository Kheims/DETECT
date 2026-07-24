#!/bin/bash
# LLM context study with vLLM on one RTX A6000: the first argument is the condition, the others are config overrides.
# REPLAY reads the C4 transcripts, so submit it after C4 has finished.
#   sbatch scripts/slurm/context_judge.sbatch C1
#   JUDGE=qwen25_coder_7b_instruct sbatch scripts/slurm/context_judge.sbatch C3
#
#SBATCH --job-name=context_judge
#SBATCH --partition=all
#SBATCH --constraint=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/context_judge_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

CONDITION=${1:?usage: context_judge.sbatch <condition> [--section.key=value ...]}
shift
CONFIG=${CONFIG:-config/experiments/context_study.yml}
JUDGE_ARGS=()
if [ -n "${JUDGE:-}" ]; then
  JUDGE_ARGS=(--judge "$JUDGE")
fi

B=/mnt/beegfs/projects/code_smells_detection/vllm
PYTHON=$B/venv/bin/python
export HF_HOME=$B/hf VLLM_CACHE_ROOT=$B/vllm-cache TRITON_CACHE_DIR=$B/triton-cache XDG_CACHE_HOME=$B/cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# FlashInfer sampling kernels fail to build with the node nvcc, and greedy decoding does not need them
export VLLM_USE_FLASHINFER_SAMPLER=0
# scikit-learn and scipy for mlcq_graphs.evaluation (not in the vLLM venv)
export PYTHONPATH=$HOME/code_smells_detection/fe_relational/pylibs312${PYTHONPATH:+:$PYTHONPATH}
# link tree_sitter and tree_sitter_java of the graph env (missing from the vLLM venv) for the C4 tools
TS_SITE=${TS_SITE:-$HOME/miniconda3/envs/detect-fe/lib/python3.12/site-packages}
TS_DIR=$(mktemp -d "${TMPDIR:-/tmp}/context_judge_ts_XXXXXX")
trap 'rm -rf "$TS_DIR"' EXIT
ln -s "$TS_SITE/tree_sitter" "$TS_SITE/tree_sitter_java" "$TS_DIR/"
export PYTHONPATH=$TS_DIR:$PYTHONPATH

SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}
# SHARD_COUNT from the environment wins, to resubmit one shard alone
SHARD_COUNT=${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} condition=$CONDITION judge=${JUDGE:-default}" \
  "shard=$SHARD_INDEX/$SHARD_COUNT cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset} overrides=$*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"$PYTHON" scripts/llm/llm_context_judge.py --config "$CONFIG" --condition "$CONDITION" ${JUDGE_ARGS[@]+"${JUDGE_ARGS[@]}"} \
  --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" "$@"
