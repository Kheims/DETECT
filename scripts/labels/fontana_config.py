"""Metric models on MLCQ with the Arcelli Fontana et al. 2016 configuration against natural prevalence.

- "Fontana 2016": per smell, every positive plus two random negatives per positive
- "natural": every sample of the level
Random and project-grouped 5-fold CV, threshold 0.5.

Usage:
    python scripts/labels/fontana_config.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlcq_graphs.xai import common  # noqa: E402

OUT = ROOT / "artifacts/analysis/fontana_config"
REPEATS = 5


def models():
    return {
        "decision tree": lambda s: DecisionTreeClassifier(max_depth=8, random_state=s),
        "random forest": lambda s: RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=s),
        "logistic regression": lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "data/derived/designite_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    cols = [c for c in rows[0] if c not in ("sample_idx",) and not c.startswith("y_")]
    feats = {int(r["sample_idx"]): np.array([float(r[c] or 0) for c in cols]) for r in rows}
    repo = {i: s["repo_url"] for i, s in enumerate(common.samples())}
    out = []
    for smell in common.LABELS:
        idx_all = np.array([i for i in feats if common.eligible(i, smell)])
        y_all = np.array([common.label(i, smell) for i in idx_all])
        for config in ("Fontana 2016", "natural"):
            for split in ("random", "project-grouped"):
                for mname, make in models().items():
                    res = []
                    for rep in range(REPEATS):
                        rng = np.random.default_rng(rep)
                        if config == "Fontana 2016":
                            pos = idx_all[y_all == 1]
                            neg = rng.choice(idx_all[y_all == 0], size=min(2 * len(pos), int((y_all == 0).sum())),
                                             replace=False)
                            idx = np.concatenate([pos, neg])
                        else:
                            idx = idx_all
                        x = np.sign(X := np.stack([feats[i] for i in idx])) * np.log1p(np.abs(X))
                        y = np.array([common.label(i, smell) for i in idx])
                        groups = np.array([repo[i] for i in idx])
                        cv = (StratifiedKFold(5, shuffle=True, random_state=rep) if split == "random"
                              else StratifiedGroupKFold(5, shuffle=True, random_state=rep))
                        pred = np.zeros(len(y), dtype=int)
                        for tr, te in cv.split(x, y, groups):
                            m = make(rep).fit(x[tr], y[tr])
                            pred[te] = (m.predict_proba(x[te])[:, 1] >= 0.5).astype(int)
                        res.append((accuracy_score(y, pred), f1_score(y, pred, zero_division=0),
                                    matthews_corrcoef(y, pred), y.mean(), len(y)))
                    r = np.array(res)
                    out.append({"smell": smell, "config": config, "split": split, "model": mname,
                                "n": int(r[0, 4]), "positive_share": round(float(r[0, 3]), 3),
                                "accuracy": r[:, 0].mean(), "accuracy_sd": r[:, 0].std(ddof=1),
                                "f1": r[:, 1].mean(), "f1_sd": r[:, 1].std(ddof=1),
                                "mcc": r[:, 2].mean(), "mcc_sd": r[:, 2].std(ddof=1)})
                    o = out[-1]
                    print(f"{common.NAMES[smell]:13s} {config:13s} {split:16s} {mname:20s} n={o['n']:5d} "
                          f"pos={o['positive_share']:.2f} acc={o['accuracy']:.3f} f1={o['f1']:.3f} mcc={o['mcc']:.3f}",
                          flush=True)
    with (OUT / "fontana_config.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"[fontana] wrote {OUT / 'fontana_config.csv'}")


if __name__ == "__main__":
    main()
