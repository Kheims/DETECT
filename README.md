# DETECT

**DE**sign defec**T** **E**valuation and **C**lassification **T**oolkit for code smell detection.

A unified pipeline that evaluates classical ML, sequence DL and graph neural network models under identical preprocessing, label definitions and evaluation protocols. Three model families, one entry point, YAML-driven configuration.

The experiments use the [MLCQ](https://zenodo.org/records/3666840) dataset, with three label rules: DS1 and DS2 of Madeyski and Lewowski, and a strict majority of the reviews.

![Pipeline overview](docs/diagrams/pipeline_overview.svg)

## Setup

```bash
git clone https://github.com/Kheims/DETECT.git && cd DETECT
uv sync
```

Requires Python 3.13+, Java 17+ (for DesigniteJava) and [uv](https://docs.astral.sh/uv/). Java 17 or later must be the first `java` in `PATH`: DesigniteJava stops with `UnsupportedClassVersionError` under Java 8. The CodeBERT runs download `microsoft/codebert-base` from the Hugging Face Hub at the first run. On a machine without network, point `HF_HOME` to a cache that already holds it.

## Reproducing the paper results

Each experiment is one command. The pipeline caches intermediate stages so only the first run per family pays the full cost.

The 17 models share one protocol: the same snippets (those with an AST), the same folds (10 seeds x 5 stratified folds, written once by the folds stage), a grid or an early stopping decided on the validation part, and one threshold per smell tuned on the validation part.

### Classical ML (8 models)

```bash
for model in rf svm xgboost dt knn j48 mlp nb; do
  uv run python main.py --config config/experiments/classical_${model}_pipeline.yml
done
```

Extracts 21 OO metrics via DesigniteJava. Each run fits every point of the grid on the training part and keeps the one with the best PR-AUC on the validation part.

### Sequence DL (6 models)

```bash
for model in lstm bilstm bilstm_plain gru cnn codebert; do
  uv run python main.py --config config/experiments/sequence_${model}_pipeline.yml
done
```

Reads each snippet as consecutive 512-token chunks (at most 3, 8 for CodeBERT), with a regex tokenizer (vocabulary of the training part, 10k) or the CodeBERT subword tokenizer. Each config runs 10 seeds x 5 folds on the shared folds of the folds stage, the same train, validation and test parts as the GNN runs. Focal loss and early stopping on the validation macro F1, BCE and early stopping on the validation average precision for CodeBERT. GPU recommended: CodeBERT takes about 3 h per seed on a V100.

### GNN (3 architectures)

```bash
for model in gcn gat graphsage; do
  uv run python main.py --config config/experiments/gnn_${model}_pipeline.yml
done
```

Parses each snippet into an AST with ANTLR Java 8 and builds PyTorch Geometric graphs. Each config holds the best setting of its architecture: Child and NextToken edges with focal loss for GCN and GAT, Child edges with weighted BCE for GraphSAGE.

### Scores of the 17 models

```bash
uv run python scripts/cv/eval_eligible.py --config config/experiments/benchmark.yml
uv run python scripts/cv/summarize_cv.py artifacts/benchmark \
  --folds artifacts/cache/folds/<fingerprint>/folds.json --baseline codebert
```

Each smell is scored on the snippets of its level: methods for Feature Envy and Long Method, classes for Blob and Data Class. The summary gives the mean MCC of each model over the 50 runs with its 95% interval, and the paired differences with the baseline (corrected resampled t-test).

### Other labels, other snippets

The labels come from the normalization stage, so one override changes them for the three families. Give each series its own artifacts folder. The thesis reports six series:

| Folder | Labels | Snippets | Overrides |
|---|---|---|---|
| `artifacts/ds1_4367` | DS1: median severity above none | 4,367 | none |
| `artifacts/ds1_4770` | DS1 | 4,770 | `--normalization.input_json=data/mlcq_complete/MLCQCodeSmellSamples.json` |
| `artifacts/majority_4367` | positive when more than half of the reviews see the smell | 4,367 | `--normalization.method=majority` |
| `artifacts/majority_4770` | majority | 4,770 | both overrides above |
| `artifacts/ds2_4367` | Madeyski DS2: major and critical only | 4,367 | `--normalization.rule=ds2` |
| `artifacts/ds2_4770` | DS2 | 4,770 | `--normalization.rule=ds2` and the input override |

`data/mlcq_complete` holds the complete dataset, rebuilt by `scripts/data/build_mlcq_complete.py`. One series is the 17 configs with the same folder and overrides, then the two scoring commands:

```bash
ROOT=artifacts/ds2_4770
OPTS="--normalization.rule=ds2 --normalization.input_json=data/mlcq_complete/MLCQCodeSmellSamples.json"
for config in config/experiments/{classical,sequence,gnn}_*_pipeline.yml; do
  uv run python main.py --config $config --run.artifacts_root=$ROOT $OPTS
done
uv run python scripts/cv/eval_eligible.py --config config/experiments/benchmark.yml --eligible_eval.root=$ROOT \
  --eligible_eval.input_json=$(ls $ROOT/cache/normalization/*/normalized*.json)
uv run python scripts/cv/summarize_cv.py $ROOT/benchmark \
  --folds $(ls $ROOT/cache/folds/*/folds.json) --baseline codebert
```

On a Slurm cluster, `scripts/slurm/submit_benchmark.sh $ROOT $OPTS` submits the same 17 models as job arrays.

### Ablation of the graph models

```bash
uv run python main.py --config config/experiments/gnn_ablation.yml
```

48 graph models (3 architectures x 2 losses x 2 edge sets x 2 node feature sets x Word2Vec tokens or not) on the five folds of seed 42. On Slurm: `PROCS=1 scripts/slurm/submit_ablation.sh config/experiments/gnn_ablation.yml 10`.

### Pre-training and several GPUs

The GATv2 encoder is pre-trained on the java-small corpus of code2seq, unpacked in `data/pretrain/raw/java-small`, then the classifier is trained on MLCQ from random weights and from the pre-trained encoder, on 1, 2 and 4 GPUs. Each job file is a wrapper around `main.py`:

```bash
P=config/experiments/ssl_pretrain_methods.yml
F=config/experiments/ssl_finetune.yml
R=--run.artifacts_root=artifacts/ssl
# training corpus in 6 parts and validation corpus, on CPU
CONFIG=config/experiments/ssl_pretrain_antlr.yml sbatch --array=0-5 scripts/slurm/pretrain_corpus_parts.sbatch $R
CONFIG=$P sbatch scripts/slurm/ssl_val_corpus.sbatch $R
# pre-training on 4 GPUs, then five seeds of the five training runs
CONFIG=$P sbatch --gres=gpu:4 scripts/slurm/ssl_pretrain.sbatch $R --pretraining.corpus.shard.count=6
CONFIG=$F sbatch --gres=gpu:4 --array=0-4%1 scripts/slurm/ssl_finetune_scaling.sbatch $R --pretraining.corpus.shard.count=6
uv run python scripts/pretraining/finetune_summary.py --artifacts artifacts/ssl
```

### What a run leaves

In its artifacts folder: `resolved_config.yml`, `cache/<stage>/<fingerprint>/` with the output of each stage and a `stage_meta.json` (config, date, commit), `runs/<fingerprint>/` for each training, and `reports/<config>_<timestamp>/` with `records.json` (one record per training), `summary.csv` and `predictions.jsonl`. `eval_eligible.py` adds `benchmark/<model>/records.json`.

### Results of the thesis

`results/` holds the records, the folds and the summaries of the runs behind the tables of the thesis, copied from the artifacts folders by `scripts/collect_results.sh`. `results/README.md` says which file gives which table.

### Generate figures

```bash
uv run python scripts/report_runs.py --artifacts-root artifacts
```

## Changing things

Any YAML key can be overridden from the command line. The pipeline re-runs only the affected stages.

```bash
# Larger hidden layer, fewer seeds
uv run python main.py --config config/experiments/sequence_lstm_pipeline.yml \
  '--training.hidden_dim=512' \
  '--ablation.seeds=[42,43]'

# Custom hyperparameter grid
uv run python main.py --config config/experiments/classical_xgboost_pipeline.yml \
  '--training.param_grid.n_estimators=[50,100,500]' \
  '--training.param_grid.max_depth=[3,5,10]'

# Force rebuild (ignore cache)
uv run python main.py --config config/experiments/classical_rf_pipeline.yml \
  '--run.force_rebuild=true'
```

Config files inherit from `config/defaults.yml`.

## Pipeline stages

**Normalization** -- Aggregates multi-reviewer annotations into binary labels. Default: median severity, positive if severity > none. Configurable aggregation (median, majority, max, min) and rules (default, ds1, ds2). When enabled, every later stage reads its labels.

**Metric extraction** -- DesigniteJava extracts 11 type-level metrics (NOF, NOM, WMC, DIT, LCOM, FANIN, FANOUT, ...) and 3 method-level metrics (LOC, CC, PC) aggregated by max/sum/avg plus method count. 21 features total.

**Token dataset** -- Regex tokenizer splits code into identifiers, keywords, operators, literals. Vocabulary built from training data, sequences padded/truncated to fixed length. Optional Word2Vec pre-training.

**AST construction** -- ANTLR Java 8 grammar parses each snippet into a syntax tree with 202 node types. Exports as DOT, converted to PyTorch Geometric Data objects.

**Folds** -- Train, validation and test parts of every (seed, fold), built once over the snippets that have an AST with the splitter of the GNN runs. 10 seeds x 5 stratified folds by default. The sequence models read them, so every family is tested on the same snippets.

**Training** -- Dispatches based on `training.used_method` (classical, sequence, chunked_sequence, gnn, codebert). Multi-seed evaluation, per-label threshold tuning, focal loss for DL models.

**Caching** -- SHA-256 fingerprint per stage. Unchanged stages are skipped across runs.

## Project structure

```
main.py                             # entry point
pyproject.toml                      # dependencies, locked in uv.lock
config/
  defaults.yml                      # all default parameters
  experiments/                      # one YAML per experiment
  ablations/                        # GCN model and feature ablation
mlcq_graphs/
  pipeline.py                       # stage orchestrator + caching
  config.py                         # YAML includes and CLI overrides
  constants.py                      # label order and edge types
  token_encoder.py                  # Word2Vec on AST token sequences
  metamorphic.py                    # transforms that keep or change a smell
  reporting.py                      # training and ablation figures
  visualization.py                  # PR, ROC, F1 and confusion plots
  models/                           # classical, sequence, GNN, genetic, fusion
  training/                         # focal loss, weighted BCE, plain BCE
  evaluation/                       # metrics, threshold tuning, shared CV
  stats/                            # paired t-test and Wilcoxon
  xai/                              # helpers of the interpretability scripts
scripts/
  NormalizeFromJson.py              # reviewer aggregation + binarization
  extract_designite_metrics.py      # batched DesigniteJava
  build_token_dataset.py            # tokenizer
  build_ast_dot_from_normalized.py  # ANTLR parsing
  build_pyg_dataset_from_dot.py     # DOT to PyG
  train_classical.py                # sklearn training
  train_sequence.py                 # PyTorch sequence training
  train_chunked_sequence.py         # sequence models and CodeBERT on 512-token chunks, shared folds
  train_codebert.py                 # CodeBERT fine-tuning
  train_gcn_baseline.py             # GNN training, run by the pipeline
  train_gcn_baseline.slurm          # cluster job for the GNN training
  train_genetic.py                  # genetic algorithm on OO metrics
  report_runs.py                    # tables + figures
  plot_confusion_matrices.py        # confusion matrices of every family
  data/                             # DACOS, SmellyCode++ conversion, features
  cv/                               # shared folds, model runs and tables
  pretraining/                      # self-supervised GNN pre-training, DDP
  llm/                              # LLM judges, context study, routing
  labels/                           # label analyses, reviewer agreement
  analysis/                         # size control, transfer, PMD, metamorphic
  xai/                              # attributions of the detectors
  slurm/                            # cluster job scripts
prompts/                            # prompts of the LLM judges
tests/                              # metamorphic and LLM context tests
tools/
  designite.jar                     # metric extraction
  antlr/                            # Java grammars + generated parser
docs/                               # pipeline diagram
data/                               # MLCQ, raw and normalized
artifacts/                          # outputs, not in the repository
results/                            # records, folds and summaries of the thesis runs
```

## Licence

The code of DETECT is under the MIT licence (`LICENSE`). The data and the tools below keep their own licence.

## Data and third-party material

- `data/MLCQCodeSmellSamples.json` and `data/mlcq_complete/` come from MLCQ (Madeyski and Lewowski, EASE 2020, doi:10.1145/3383219.3383264), distributed under CC BY 4.0. The code snippets belong to the open-source projects listed in the dataset and keep the licence of their project. `data/mlcq_complete/provenance.csv` says where each snippet was read.
- `tools/designite.jar` is DesigniteJava 1.1.2, built from https://github.com/tushartushar/DesigniteJava (Apache License 2.0).
- `tools/antlr/` holds the Java 8 grammar of the ANTLR grammars-v4 project (BSD licence, in the header of the grammar files) and the parser generated from it.

## Citation

Mesbah, D., El Madhoun, N., Al Agha, K. and Zouaoui, A. A survey on code smells detection using machine learning techniques. Information and Software Technology 198, 108242 (2026). doi:10.1016/j.infsof.2026.108242
