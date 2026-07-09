#!/bin/bash
# Fine-tuning of pretrained code encoders on one A6000 per array task, one shard of code_finetune.run_models each.
# Finished items are skipped, so the same command resumes a run.
#   sbatch --array=0-5%2 scripts/slurm/code_finetune.sbatch [--key=value ...]
#   COLLECT=1 bash scripts/slurm/code_finetune.sbatch    (login node, merges item records, no GPU)
#
#SBATCH --job-name=code_finetune
#SBATCH --partition=all
#SBATCH --constraint=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "${SLURM_SUBMIT_DIR:-.}"

VLLM_ROOT=/mnt/beegfs/projects/code_smells_detection/vllm
PYTHON="${DETECT_PYTHON:-$VLLM_ROOT/venv/bin/python}"
export PYTHONPATH="$HOME/code_smells_detection/fe_relational/pylibs312${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$VLLM_ROOT/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$VLLM_ROOT/cache" TRITON_CACHE_DIR="$VLLM_ROOT/triton-cache"

CONFIG="${CONFIG:-config/experiments/code_finetune.yml}"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
COUNT="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"

if [[ "${COLLECT:-0}" == "1" ]]; then
  "$PYTHON" scripts/cv/finetune_code_model.py --config "$CONFIG" --collect "$@"
  exit 0
fi

echo "host=$(hostname) config=$CONFIG shard=$INDEX/$COUNT gpus=${CUDA_VISIBLE_DEVICES:-unset} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"

status=0
"$PYTHON" scripts/cv/finetune_code_model.py --config "$CONFIG" --shard-index "$INDEX" --shard-count "$COUNT" \
  --code_finetune.device=cuda "$@" || status=$?
echo "end=$(date -Is) status=$status"
exit $status
