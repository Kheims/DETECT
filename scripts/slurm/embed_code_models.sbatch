#!/bin/bash
# Frozen code-model embeddings on the RTX A6000 nodes, one shard of the CONFIG model list per array task.
#   sbatch scripts/slurm/embed_code_models.sbatch
#   sbatch --array=0 scripts/slurm/embed_code_models.sbatch --code_embeddings.only=[qwen3_emb_06b]
#
#SBATCH --job-name=embed_code
#SBATCH --partition=all
#SBATCH --constraint=a6000
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"

CONFIG="${CONFIG:-config/experiments/code_embeddings.yml}"
VLLM_ROOT=/mnt/beegfs/projects/code_smells_detection/vllm
PYTHON="${EMBED_PYTHON:-$VLLM_ROOT/venv/bin/python}"
# scikit-learn and scipy for mlcq_graphs.evaluation.cv (not in the vLLM venv)
export PYTHONPATH="$HOME/code_smells_detection/fe_relational/pylibs312${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=$VLLM_ROOT/hf
export VLLM_CACHE_ROOT=$VLLM_ROOT/vllm-cache
export TRITON_CACHE_DIR=$VLLM_ROOT/triton-cache
export XDG_CACHE_HOME=$VLLM_ROOT/cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# export SHARD_COUNT to rerun one task of the original array alone
SHARD_INDEX=${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}
SHARD_COUNT=${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}
echo "host=$(hostname) config=$CONFIG shard=$SHARD_INDEX/$SHARD_COUNT start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"

"$PYTHON" scripts/cv/embed_code_models.py --config "$CONFIG" \
  --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" "$@" < /dev/null
echo "end=$(date -Is)"
