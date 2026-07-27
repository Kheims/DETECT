"""Per-snippet contributions of the size model (logistic regression on 14 size features).

Refits the model per smell and seed-42 kfold fold. Writes artifacts/xai/size_contributions.jsonl
and artifacts/xai/size_importance.csv.

Usage:
    python scripts/xai/xai_size_model.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlcq_graphs.xai import common  # noqa: E402

COLS = ["loc_physical", "loc_nonblank", "chars", "tokens", "ast_nodes", "ast_depth", "n_methods", "n_fields",
        "n_statements", "n_params_first_method", "max_block_nesting", "n_distinct_identifiers", "n_classes",
        "n_comments"]


def main() -> None:
    with (ROOT / "data/derived/size_features.csv").open() as f:
        feats = {int(r["sample_idx"]): np.array([float(r[c] or 0) for c in COLS]) for r in csv.DictReader(f)}
    x_of = {i: np.sign(v) * np.log1p(np.abs(v)) for i, v in feats.items()}
    rows, imp = [], defaultdict(list)
    for f, parts in common.folds(42, "kfold").items():
        for k, smell in enumerate(common.LABELS):
            tr = [i for i in parts["train"] if i in x_of and common.eligible(i, smell)]
            xtr = np.stack([x_of[i] for i in tr])
            mean = xtr.mean(0)
            ytr = np.array([common.label(i, smell) for i in tr])
            m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(xtr, ytr)
            coef, b = m.coef_[0], float(m.intercept_[0])
            base = b + float(coef @ mean)
            for i in parts["test"]:
                if not common.eligible(i, smell) or i not in x_of:
                    continue
                c = coef * (x_of[i] - mean)
                logit = base + float(c.sum())
                rows.append({"json_index": i, "smell": smell, "label": common.label(i, smell), "fold": f,
                             "prob": float(1 / (1 + np.exp(-logit))), "base": base,
                             "contributions": {col: round(float(v), 4) for col, v in zip(COLS, c)},
                             "raw": {col: float(feats[i][j]) for j, col in enumerate(COLS)}})
                for col, v in zip(COLS, c):
                    imp[(smell, col)].append(abs(float(v)))
    out = ROOT / "artifacts/xai"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "size_contributions.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with (out / "size_importance.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smell", "feature", "mean_abs_contribution"])
        for (smell, col), v in sorted(imp.items(), key=lambda kv: (kv[0][0], -np.mean(kv[1]))):
            w.writerow([smell, col, round(float(np.mean(v)), 4)])
    for smell in common.LABELS:
        top = sorted(((col, np.mean(v)) for (s, col), v in imp.items() if s == smell), key=lambda x: -x[1])[:4]
        print(f"[size] {common.NAMES[smell]:13s} " + ", ".join(f"{c} {v:.2f}" for c, v in top))
    print(f"[size] {len(rows)} rows")


if __name__ == "__main__":
    main()
