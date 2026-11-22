# DETECT

**DE**sign defec**T** **E**valuation and **C**lassification **T**oolkit for code smell detection.

DETECT trains and evaluates classical ML, sequence and graph neural network detectors of four code smells (Feature Envy, Long Method, Blob and Data Class) on the MLCQ dataset [1]. Every model goes through the same preprocessing, labels, folds and metrics. One entry point, YAML configuration.

![Pipeline overview](docs/diagrams/pipeline_overview.svg)

## Setup

```bash
git clone https://github.com/Kheims/DETECT.git && cd DETECT
uv sync
```

Requires Python 3.13+, Java 17+ (for DesigniteJava) and [uv](https://docs.astral.sh/uv/). Java 17 or later must be the first `java` in `PATH`. The CodeBERT runs download `microsoft/codebert-base` from the Hugging Face Hub at the first run.

## Dataset

MLCQ [1] lists 14,739 reviews of 4,770 Java snippets. It gives the location of each snippet on GitHub (repository, commit, file and lines) and does not hold the code. `data/` holds the original table and two files with the code:

| File | Content | Source |
|---|---|---|
| `MLCQCodeSmellSamples.csv` | the reviews and the location of each snippet | Zenodo record [3666840](https://zenodo.org/records/3666840), unchanged |
| `MLCQCodeSmellSamples.json` | original dataset: 13,456 reviews of 4,367 snippets, with the code | the files that GitHub serves at the locations of the CSV |
| `mlcq_complete/MLCQCodeSmellSamples.json` | complete dataset: 14,739 reviews of 4,770 snippets, with the code | the original dataset and the 403 snippets recovered below |
| `mlcq_complete/provenance.csv` | where the code of each snippet was read | |
| `mlcq_complete/missing_samples.csv` | the 403 recovered snippets: original repository and repository where the code was read | |

GitHub no longer serves 403 snippets at the location of the CSV, because their repository was renamed, moved or deleted. Their commit hash still identifies them. We found 343 of them in another GitHub repository that holds the same commit (the renamed repository or a copy of it) and 60 in the [Software Heritage](https://www.softwareheritage.org) archive. `data/mlcq_complete/missing_samples.csv` holds the mapping between the original repositories and the new ones.

Three scripts rebuild the two files with code from the CSV:

```bash
# code of each snippet at the location of the CSV
uv run python scripts/data/extract_mlcq_snippets.py
# where the commit of each missing snippet is today
uv run python scripts/data/check_mlcq_availability.py
# code of the missing snippets, then the complete dataset
uv run python scripts/data/build_mlcq_complete.py
```

## Training

Each model is one command. The pipeline caches every stage, so only the first run of a family pays for the preprocessing.

```bash
# classical ML on 21 OO metrics from DesigniteJava
for model in rf svm xgboost dt knn j48 mlp nb; do
  uv run python main.py --config config/experiments/classical_${model}_pipeline.yml
done

# sequence models on 512-token chunks (GPU recommended)
for model in lstm bilstm bilstm_plain gru cnn codebert; do
  uv run python main.py --config config/experiments/sequence_${model}_pipeline.yml
done

# graph neural networks on the AST of each snippet
for model in gcn gat graphsage; do
  uv run python main.py --config config/experiments/gnn_${model}_pipeline.yml
done
```

The 17 models share one protocol: the same snippets, the same folds (10 seeds x 5 stratified folds, written once by the folds stage), a grid or an early stopping decided on the validation part and one threshold per smell tuned on the validation part.

### Label rules

The normalization stage turns the reviews of a snippet into one label per smell. DS1 and DS2 are the two rules of Madeyski and Lewowski [2]. One override changes the rule for every model:

| Rule | A smell is present when | Override |
|---|---|---|
| DS1 (default) | the median severity is minor, major or critical | none |
| DS2 | the median severity is major or critical | `--normalization.rule=ds2` |
| majority | more than half of the reviews give minor, major or critical | `--normalization.method=majority` |

### Original or complete dataset

The configs read the original dataset (4,367 snippets). This override selects the complete dataset (4,770 snippets):

```bash
--normalization.input_json=data/mlcq_complete/MLCQCodeSmellSamples.json
```

Give each combination of a rule and a dataset its own artifacts folder. For example, DS2 on the complete dataset:

```bash
ROOT=artifacts/ds2_4770
OPTS="--normalization.rule=ds2 --normalization.input_json=data/mlcq_complete/MLCQCodeSmellSamples.json"
for config in config/experiments/{classical,sequence,gnn}_*_pipeline.yml; do
  uv run python main.py --config $config --run.artifacts_root=$ROOT $OPTS
done
```

On a Slurm cluster, `scripts/slurm/submit_benchmark.sh $ROOT $OPTS` submits the same 17 models as job arrays.

### Other overrides

Any YAML key can be set on the command line. The pipeline runs again only the stages that the change affects. Config files inherit from `config/defaults.yml`.

```bash
uv run python main.py --config config/experiments/sequence_lstm_pipeline.yml \
  '--training.hidden_dim=512' '--ablation.seeds=[42,43]'
```

`config/experiments/gnn_ablation.yml` trains 48 graph models (3 architectures x 2 losses x 2 edge sets x 2 node feature sets x Word2Vec tokens or not) on the five folds of seed 42.

## Scores and significance tests

```bash
uv run python scripts/cv/eval_eligible.py --config config/experiments/benchmark.yml --eligible_eval.root=$ROOT \
  --eligible_eval.input_json=$(ls $ROOT/cache/normalization/*/normalized*.json)
uv run python scripts/cv/summarize_cv.py $ROOT/benchmark \
  --folds $(ls $ROOT/cache/folds/*/folds.json) --baseline codebert
```

The first script scores each smell on the snippets of its level: methods for Feature Envy and Long Method, classes for Blob and Data Class. The second script gives the mean MCC of each model over the 50 runs with its 95% interval. With `--baseline`, it adds the paired difference between each model and the baseline, tested with the corrected resampled t-test and adjusted with the Holm method.

The records in `results/` let you run the tests without training. For example, the 17 models against CodeBERT with DS1 labels on the original dataset:

```bash
uv run python scripts/cv/summarize_cv.py results/ds1_4367/benchmark \
  --folds results/ds1_4367/folds.json --baseline codebert
```

```
  gcn {"eligible_only": true, "split_strategy": "kfold"} macro_mcc         diff=-0.014 [-0.055, +0.027] wins=18/50 p=0.502 p_holm=0.502
  xgboost {"eligible_only": true, "split_strategy": "kfold"} macro_mcc         diff=-0.057 [-0.099, -0.016] wins=5/50 p=0.00801 p_holm=0.0401
```

## Results

`results/` holds the records of our runs for each label rule and each dataset: 50 runs for each of the 17 models, the folds and the summary. `results/README.md` describes the files.

A run writes into its artifacts folder: `resolved_config.yml`, `cache/<stage>/<fingerprint>/` with the output of each stage and a `stage_meta.json` (config, date, commit), `runs/<fingerprint>/` for each training and `reports/<config>_<timestamp>/` with `records.json`, `summary.csv` and `predictions.jsonl`. `scripts/collect_results.sh` copies the records of an artifacts folder to `results/`.

## Pipeline stages

**Normalization** -- Aggregates the reviews of each snippet into binary labels (aggregation: median, majority, max or min, rules: ds1, ds2).

**Metric extraction** -- DesigniteJava [3] extracts 11 type-level metrics (NOF, NOM, WMC, DIT, LCOM, FANIN, FANOUT, ...) and 3 method-level metrics (LOC, CC, PC) aggregated by max, sum and mean, plus the method count. 21 features in total.

**Token dataset** -- A regex tokenizer or the CodeBERT tokenizer reads each snippet as consecutive 512-token chunks. The vocabulary comes from the training part.

**AST construction** -- The ANTLR Java 8 grammar parses each snippet into a syntax tree with 202 node types, exported as DOT then converted to PyTorch Geometric graphs.

**Folds** -- Train, validation and test parts of every (seed, fold), built once over the snippets that have an AST.

**Training** -- `training.used_method` selects classical, sequence, chunked_sequence, gnn or codebert.

**Caching** -- SHA-256 fingerprint per stage. A stage whose inputs did not change is skipped.

## Project structure

```
main.py              # entry point
config/              # defaults.yml and one YAML per experiment
mlcq_graphs/         # pipeline, models, training, evaluation
scripts/             # stages run by the pipeline
  data/              # MLCQ extraction
  cv/                # scores on eligible snippets, summaries and tests
  slurm/             # cluster job scripts
tools/               # DesigniteJava and the ANTLR Java grammar
data/                # MLCQ: Zenodo table, original and complete datasets
results/             # records, folds and summaries of our runs
```

## Pre-training

We also ran self-supervised pre-training experiments with a GNN encoder. They can be found in the branch [`pretraining`](https://github.com/Kheims/DETECT/tree/pretraining).

## Licence

MIT (`LICENSE`) for the code of DETECT.

- `data/` comes from MLCQ [1] (CC BY 4.0).
- `tools/designite.jar` is DesigniteJava 1.1.2 [3] (Apache License 2.0).
- `tools/antlr/` holds the Java 8 grammar of [grammars-v4](https://github.com/antlr/grammars-v4) (BSD).

## References

[1] Madeyski, L. and Lewowski, T. MLCQ: Industry-Relevant Code Smell Data Set. Evaluation and Assessment in Software Engineering (EASE), 342-347 (2020). doi:10.1145/3383219.3383264. Data: doi:10.5281/zenodo.3666840

[2] Madeyski, L. and Lewowski, T. Detecting code smells using industry-relevant data. Information and Software Technology 155, 107112 (2023). doi:10.1016/j.infsof.2022.107112

[3] Sharma, T. DesigniteJava. Zenodo (2018). doi:10.5281/zenodo.2566861. Source: https://github.com/tushartushar/DesigniteJava

## Citation

Mesbah, D., El Madhoun, N., Al Agha, K. and Zouaoui, A. A survey on code smells detection using machine learning techniques. Information and Software Technology 198, 108242 (2026). doi:10.1016/j.infsof.2026.108242
