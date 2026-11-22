# Results

Records of our runs of the 17 models, one folder per label rule and dataset (see "Label rules" and "Original or complete dataset" in the main README). `scripts/collect_results.sh` copies them from the artifacts folders.

| Folder | Labels | Dataset |
|---|---|---|
| `ds1_4367` | DS1 | original, 4,367 snippets |
| `ds1_4770` | DS1 | complete, 4,770 snippets |
| `ds2_4367` | DS2 | original |
| `ds2_4770` | DS2 | complete |
| `majority_4367` | majority | original |

Each folder holds:

- `benchmark/<model>/records.json`: one record per run, 10 seeds x 5 folds = 50 runs. A record gives the thresholds tuned on the validation part and, for each smell, precision, recall, F1, MCC, PR-AUC and the confusion counts on the test snippets of the level of the smell.
- `folds.json`: the snippets of the training, validation and test parts of every (seed, fold).
- `folds_meta.json` and `labels_meta.json`: configuration, date and commit of the folds stage and of the normalization stage.
- `summary.txt`: mean and 95% interval of each model over the 50 runs, then the paired differences with CodeBERT (corrected resampled t-test, Holm adjustment).

The summary is rebuilt from the records with:

```bash
uv run python scripts/cv/summarize_cv.py results/ds1_4367/benchmark \
  --folds results/ds1_4367/folds.json --baseline codebert
```

## Ablation of the graph models

`ablation/records.json` holds 240 runs: 48 configurations of `config/experiments/gnn_ablation.yml` on the five folds of seed 42, with the folds in `ablation/folds.json`. These runs are scored on all the test snippets of a fold.
