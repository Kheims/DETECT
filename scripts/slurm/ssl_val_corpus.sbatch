#!/bin/bash
# Parse the validation corpus of the pre-training (java-small validation project) on CPU.
#   CONFIG=config/experiments/ssl_pretrain_methods.yml sbatch scripts/slurm/ssl_val_corpus.sbatch
#
#SBATCH --job-name=ssl_val_corpus
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
PYTHON="${DETECT_PYTHON:-$HOME/miniconda3/envs/detect-fe/bin/python}"
echo "host=$(hostname) config=$CONFIG start=$(date -Is)"

# only the validation corpus of the pretraining stage is built, the other stages are switched off
"$PYTHON" main.py --config "$CONFIG" \
  --construction.enabled=false --dataset.enabled=false --token_encoder.enabled=false \
  --training.enabled=false --reporting.enabled=false \
  --pretraining.val_corpus.workers="$SLURM_CPUS_PER_TASK" \
  --pretraining.val_corpus.prepare_only=true "$@"
echo "end=$(date -Is)"
