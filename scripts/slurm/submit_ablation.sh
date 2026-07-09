#!/bin/bash
# Submit the cache-building job, then the sharded ablation once it succeeds.
#
#   PROCS=3 scripts/slurm/submit_ablation.sh <config.yml> <num_gpus> [--key=value ...]
set -euo pipefail
CONFIG="$1"; GPUS="$2"; shift 2
mkdir -p logs
PREP=$(CONFIG="$CONFIG" sbatch --parsable --export=ALL scripts/slurm/prepare.sbatch "$@")
ARR=$(CONFIG="$CONFIG" PROCS="${PROCS:-1}" sbatch --parsable --export=ALL --dependency=afterok:"$PREP" \
  --array=0-$((GPUS - 1)) scripts/slurm/ablation_array.sbatch "$@")
echo "prepare=$PREP ablation=$ARR config=$CONFIG gpus=$GPUS procs=${PROCS:-1}"
