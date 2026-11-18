#!/bin/bash
# The 17 models of the benchmark on one series: the cached stages first, then one array for each model.
# The overrides go to every job, so the 17 models share the snippets, the labels and the folds.
#
#   scripts/slurm/submit_benchmark.sh <artifacts folder of the series> [--key=value ...]
#   scripts/slurm/submit_benchmark.sh artifacts/ds2 --normalization.rule=ds2
#   scripts/slurm/submit_benchmark.sh artifacts/seed42 '--ablation.seeds=[42]'
set -euo pipefail
ROOT="$1"; shift
ARGS=("--run.artifacts_root=$ROOT" "$@")
E=config/experiments
mkdir -p logs

# labels, AST graphs, folds and metrics, then the two graph datasets
PREP=$(CONFIG=$E/classical_xgboost_pipeline.yml sbatch --parsable --export=ALL scripts/slurm/prepare.sbatch "${ARGS[@]}")
for config in gnn_gcn gnn_graphsage; do
  PREP=$(CONFIG=$E/${config}_pipeline.yml sbatch --parsable --export=ALL --dependency=afterok:"$PREP" \
    scripts/slurm/prepare.sbatch "${ARGS[@]}")
done

# array <config> <tasks> <processes for each task> [sbatch option ...]
array() {
  local config=$1 tasks=$2 procs=$3; shift 3
  CONFIG=$E/${config}_pipeline.yml PROCS=$procs sbatch --parsable --export=ALL --dependency=afterok:"$PREP" \
    --array=0-$((tasks - 1)) "$@" scripts/slurm/ablation_array.sbatch "${ARGS[@]}"
}
for model in rf svm xgboost dt knn j48 mlp nb; do array classical_$model 1 1 --gres=gpu:0; done
for model in lstm bilstm bilstm_plain gru cnn; do array sequence_$model 5 1; done
array sequence_codebert 10 1
array gnn_gcn 5 2
array gnn_graphsage 5 2
array gnn_gat 10 1
