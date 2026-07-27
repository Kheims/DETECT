#!/bin/bash
# BiLSTM with attention interpretability study: one array task per fold
# trains the model and computes the test attributions (scripts/xai/xai_bilstm.py run).
# Expects inputs.json.gz in $OUT (python scripts/xai/xai_bilstm.py prepare, run locally).
#
#   OUT=/abs/path/artifacts/xai/bilstm sbatch scripts/slurm/xai_bilstm.sbatch
#
#SBATCH --job-name=xai_bilstm
#SBATCH --array=0-4
#SBATCH --partition=all
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/xai_bilstm_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON="$HOME/miniconda3/envs/detect-fe/bin/python"
echo "host=$(hostname) fold=$SLURM_ARRAY_TASK_ID out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" scripts/xai/xai_bilstm.py run --fold "$SLURM_ARRAY_TASK_ID" --device cuda --out "$OUT"
