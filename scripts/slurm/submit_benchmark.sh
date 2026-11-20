#!/bin/bash
# Submit the 17 models of one series: the cached stages first, then one job array per model.
#   scripts/slurm/submit_benchmark.sh <artifacts folder> [--key=value ...]
# GPU_OPTS: sbatch options of the GPU jobs, e.g. GPU_OPTS=--constraint=v100
# SEQUENCE_ENV: variables added to the environment of the sequence jobs, e.g. SEQUENCE_ENV=HF_HUB_OFFLINE=1
set -euo pipefail
ROOT="$1"; shift
ARGS=("--run.artifacts_root=$ROOT" "$@")
E=config/experiments
read -r -a GPU <<< "${GPU_OPTS:-}"
SEQUENCE="ALL${SEQUENCE_ENV:+,$SEQUENCE_ENV}"
mkdir -p logs

# labels, graphs, folds and tokens, then the metrics, then the two graph datasets
PREP=""
for config in sequence_lstm classical_xgboost gnn_gcn gnn_graphsage; do
  PREP=$(CONFIG=$E/${config}_pipeline.yml sbatch --parsable --export=ALL ${PREP:+--dependency=afterok:$PREP} \
    scripts/slurm/prepare.sbatch "${ARGS[@]}")
done

array() {  # array <config> <tasks> <processes per task> <environment> [sbatch options]
  local config=$1 tasks=$2 procs=$3 env=$4; shift 4
  CONFIG=$E/${config}_pipeline.yml PROCS=$procs sbatch --parsable --export="$env" --dependency=afterok:"$PREP" \
    --array=0-$((tasks - 1)) "$@" scripts/slurm/ablation_array.sbatch "${ARGS[@]}"
}
for model in rf svm xgboost dt knn j48 mlp nb; do array classical_$model 1 1 ALL --gres=gpu:0; done
for model in lstm bilstm bilstm_plain gru cnn; do array sequence_$model 5 1 "$SEQUENCE" ${GPU[@]+"${GPU[@]}"}; done
array sequence_codebert 10 1 "$SEQUENCE" ${GPU[@]+"${GPU[@]}"}
array gnn_gcn 5 2 ALL ${GPU[@]+"${GPU[@]}"}
array gnn_graphsage 5 2 ALL ${GPU[@]+"${GPU[@]}"}
array gnn_gat 10 1 ALL ${GPU[@]+"${GPU[@]}"}
