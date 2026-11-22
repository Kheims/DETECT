#!/usr/bin/env bash
set -euo pipefail

PY=${PYTHON:-python}
OUT=${OUT:-results}

for root in "$@"; do
  dest=$OUT/$(basename "$root")
  mkdir -p "$dest"
  for model in "$root"/benchmark/*/; do
    mkdir -p "$dest/benchmark/$(basename "$model")"
    cp "$model/records.json" "$dest/benchmark/$(basename "$model")/"
  done
  cp "$root"/cache/folds/*/folds.json "$dest/folds.json"
  cp "$root"/cache/folds/*/stage_meta.json "$dest/folds_meta.json"
  cp "$root"/cache/normalization/*/stage_meta.json "$dest/labels_meta.json"
  $PY scripts/cv/summarize_cv.py "$dest/benchmark" --folds "$dest/folds.json" --baseline codebert > "$dest/summary.txt"
done
