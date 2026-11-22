# Results

Outputs of the runs behind the tables of the thesis. Every file comes from the code of this repository (commit `fe190c2`), run on a Slurm cluster with V100 GPUs in November 2026. `scripts/collect_results.sh` copies them from the artifacts folders.

## The 17 models under one protocol

One folder per series, a series being one label rule on one set of snippets (see "Other labels, other snippets" in the main README).

| Folder | Labels | Snippets |
|---|---|---|
| `ds1_4367` | DS1 | 4,367 |
| `ds1_4770` | DS1 | 4,770 |
| `ds2_4367` | DS2 | 4,367 |
| `ds2_4770` | DS2 | 4,770 |

Each folder holds:

- `benchmark/<model>/records.json`: one record per run, 10 seeds x 5 folds = 50 runs. A record gives the thresholds tuned on the validation part and, for each smell, precision, recall, F1, MCC, PR-AUC and the confusion counts on the test snippets of the level of the smell.
- `folds.json`: the snippets of the training, validation and test parts of every (seed, fold).
- `folds_meta.json` and `labels_meta.json`: configuration, date and commit of the folds stage and of the normalization stage.
- `summary.txt`: mean and 95% interval of each model over the 50 runs, then the paired differences with CodeBERT (corrected resampled t-test).

The summary is rebuilt from the records with:

```bash
uv run python scripts/cv/summarize_cv.py results/ds1_4367/benchmark \
  --folds results/ds1_4367/folds.json --baseline codebert
```

In the thesis, `ds1_4367` gives the tables of the unified evaluation (chapter 7), the scores of the three graph models (chapter 3: `gcn`, `gat`, `graphsage`) and the scores of the sequence models (chapter 4: `lstm`, `bilstm`, `bilstm_attention`, `codebert`).

## Ablation of the graph models

`ablation/records.json` holds 240 runs: 48 configurations on the five folds of seed 42, with the folds in `ablation/folds.json`. These runs are scored on all the test snippets of a fold, not on the snippets of the level of each smell.

## Pre-training and several GPUs

`pretraining/runs/<fingerprint>/` holds the configuration, the metrics and the training curves of the 25 training runs on MLCQ: five settings (random weights, two pre-trained encoders on 1 GPU, the pre-trained encoder on 2 and 4 GPUs) with five seeds each. `pretraining/summary.txt` and `pretraining/summary.json` come from:

```bash
uv run python scripts/pretraining/finetune_summary.py --artifacts results/pretraining
```
