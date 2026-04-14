# DETECT

**DE**sign defec**T** **E**valuation and **C**lassification **T**oolkit for code smell detection.

A unified pipeline that evaluates classical ML, sequence DL and graph neural network models under identical preprocessing, label definitions and evaluation protocols. Three model families, one entry point, YAML-driven configuration.

Currently supports the [MLCQ](https://zenodo.org/records/3666840) dataset. Integration of [DaCoSX](https://zenodo.org/records/7570428) and support for alternative binarization protocols (Madeyski DS1/DS2) are ongoing.

![Pipeline overview](docs/diagrams/pipeline_overview.svg)

## Setup

```bash
git clone https://github.com/Kheims/DETECT.git && cd DETECT
uv sync
```

Requires Python 3.13+, Java 17+ (for DesigniteJava) and [uv](https://docs.astral.sh/uv/).

## Reproducing the paper results

Each experiment is one command. The pipeline caches intermediate stages so only the first run per family pays the full cost.

### Classical ML (8 models)

```bash
for model in rf svm xgboost dt knn j48 mlp nb; do
  uv run python main.py --config config/experiments/classical_${model}_pipeline.yml
done
```

Extracts 21 OO metrics via DesigniteJava, trains with 5-fold stratified CV, grid search and per-label threshold tuning.

### Sequence DL (6 models)

```bash
for model in lstm bilstm bilstm_plain gru cnn codebert; do
  uv run python main.py --config config/experiments/sequence_${model}_pipeline.yml
done
```

Tokenizes code with a regex splitter (vocab 10k, max length 512), trains with focal loss and early stopping. CodeBERT uses its own subword tokenizer. GPU recommended.

### GNN (3 architectures)

```bash
uv run python main.py --config config/experiments/gcn_baseline.yml
```

Parses each snippet into an AST with ANTLR Java 8, builds PyTorch Geometric graphs with Child edges and 69-dim node features, trains with focal loss. Supports GCN, GAT, GraphSAGE via ablation config.

### Generate figures

```bash
uv run python scripts/report_runs.py --artifacts-root artifacts
```

## Changing things

Any YAML key can be overridden from the command line. The pipeline re-runs only the affected stages.

```bash
# Larger hidden layer, fewer seeds
uv run python main.py --config config/experiments/sequence_lstm_pipeline.yml \
  '--training.model_params.hidden_dim=256' \
  '--training.seeds=[42,43]'

# Custom hyperparameter grid
uv run python main.py --config config/experiments/classical_xgboost_pipeline.yml \
  '--training.param_grid.n_estimators=[50,100,500]' \
  '--training.param_grid.max_depth=[3,5,10]'

# Switch binarization to match Madeyski DS1
uv run python main.py --config config/experiments/classical_rf_pipeline.yml \
  '--normalization.rule=madeyskiDS1'

# Force rebuild (ignore cache)
uv run python main.py --config config/experiments/classical_rf_pipeline.yml \
  '--run.force_rebuild=true'
```

Config files inherit from `config/defaults.yml`.

## Pipeline stages

**Normalization** -- Aggregates multi-reviewer annotations into binary labels. Default: median severity, positive if severity > none. Configurable rules (default, madeyskiDS1, madeyskiDS2).

**Metric extraction** -- DesigniteJava extracts 11 type-level metrics (NOF, NOM, WMC, DIT, LCOM, FANIN, FANOUT, ...) and 3 method-level metrics (LOC, CC, PC) aggregated by max/sum/avg plus method count. 21 features total.

**Token dataset** -- Regex tokenizer splits code into identifiers, keywords, operators, literals. Vocabulary built from training data, sequences padded/truncated to fixed length. Optional Word2Vec pre-training.

**AST construction** -- ANTLR Java 8 grammar parses each snippet into a syntax tree with 202 node types. Exports as DOT, converted to PyTorch Geometric Data objects.

**Training** -- Dispatches based on `training.used_method` (classical, sequence, gnn, codebert). Multi-seed evaluation, per-label threshold tuning, focal loss for DL models.

**Caching** -- SHA-256 fingerprint per stage. Unchanged stages are skipped across runs.

## Project structure

```
main.py                             # entry point
config/
  defaults.yml                      # all default parameters
  experiments/                      # one YAML per experiment
mlcq_graphs/
  pipeline.py                      # stage orchestrator + caching
  models/                          # classical.py, sequence.py, gcn.py, gat.py, graphsage.py
  training/                        # focal loss, weighted BCE
  evaluation/                      # F1, MCC, PR-AUC, threshold tuning
scripts/
  NormalizeFromJson.py              # reviewer aggregation + binarization
  extract_designite_metrics.py      # batched DesigniteJava
  build_token_dataset.py            # tokenizer
  build_ast_dot_from_normalized.py  # ANTLR parsing
  build_pyg_dataset_from_dot.py     # DOT to PyG
  train_classical.py                # sklearn training
  train_sequence.py                 # PyTorch sequence training
  train_codebert.py                 # CodeBERT fine-tuning
  report_runs.py                    # tables + figures
tools/
  designite.jar                     # metric extraction
  antlr/                            # Java 8 grammar
data/                               # datasets (included)
artifacts/                          # outputs (gitignored)
```


