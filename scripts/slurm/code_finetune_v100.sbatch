#!/bin/bash
# code_finetune.sbatch on V100 nodes with the torch 2.5 env detect-fe, in float32 (no native bfloat16 on V100).
#   CONFIG=config/experiments/review_share_codebert.yml sbatch --array=0-5 scripts/slurm/code_finetune_v100.sbatch
#
#SBATCH --job-name=code_finetune_v100
#SBATCH --partition=all
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "${SLURM_SUBMIT_DIR:-.}"

PYTHON="${DETECT_PYTHON:-$HOME/miniconda3/envs/detect-fe/bin/python}"
L=$HOME/code_smells_detection/fe_relational
export PYTHONPATH=$L/pylibs_hf312:$L/pylibs312${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

CONFIG="${CONFIG:-config/experiments/code_finetune.yml}"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
COUNT="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"

echo "host=$(hostname) config=$CONFIG shard=$INDEX/$COUNT gpus=${CUDA_VISIBLE_DEVICES:-unset} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"

status=0
"$PYTHON" scripts/cv/finetune_code_model.py --config "$CONFIG" --shard-index "$INDEX" --shard-count "$COUNT" \
  --code_finetune.device=cuda --code_finetune.bf16=false "$@" || status=$?
echo "end=$(date -Is) status=$status"
exit $status
