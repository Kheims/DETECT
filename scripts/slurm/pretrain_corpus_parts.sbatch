#!/bin/bash
# Parse the pre-training corpus in parts on CPU, one array task per part (pretraining.corpus.shard).
#   CONFIG=config/experiments/ssl_pretrain_antlr.yml sbatch --array=0-5 scripts/slurm/pretrain_corpus_parts.sbatch
# When every part is done, the first run with '--pretraining.corpus.shard.count=6' and no index
# merges the parts into the corpus manifest and goes on with the pre-training.
#
#SBATCH --job-name=pretrain_corpus
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
PYTHON="${DETECT_PYTHON:-$HOME/miniconda3/envs/detect-fe/bin/python}"
echo "host=$(hostname) config=$CONFIG part=$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT start=$(date -Is)"

# only the corpus step of the pretraining stage runs, the other stages are switched off
"$PYTHON" main.py --config "$CONFIG" \
  --construction.enabled=false --dataset.enabled=false --token_encoder.enabled=false \
  --training.enabled=false --reporting.enabled=false \
  --pretraining.corpus.workers="$SLURM_CPUS_PER_TASK" \
  --pretraining.corpus.shard.count="$SLURM_ARRAY_TASK_COUNT" \
  --pretraining.corpus.shard.index="$SLURM_ARRAY_TASK_ID" "$@"
echo "end=$(date -Is)"
