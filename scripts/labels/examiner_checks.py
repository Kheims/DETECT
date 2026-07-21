"""Three checks on the MLCQ model predictions (CPU only).

- severity: Spearman correlation between detector scores and MLCQ severity
- effects: each model against the size model per smell (MCC difference, Vargha-Delaney A)
- surrogates: depth-3 decision tree on metrics fitted to the model decisions

Usage:
    python scripts/labels/examiner_checks.py --config config/experiments/mlcq_eligible_all.yml
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlcq_graphs.config import load_config  # noqa: E402
from mlcq_graphs.evaluation.cv import tune_thresholds  # noqa: E402
from mlcq_graphs.xai import common  # noqa: E402

OUT = ROOT / "artifacts/analysis/examiner_checks"
SEV = {"none": 0, "minor": 1, "major": 2, "critical": 3}


def fold_predictions(pattern: str, strategy: str = "kfold"):
    """{(seed, fold): {"val": (idx, y, p), "test": (idx, y, p)}} of a predictions glob."""
    groups = defaultdict(lambda: {"val": [], "test": []})
    for f in sorted(glob.glob(str(ROOT / pattern))):
        for line in open(f):
            r = json.loads(line)
            if r.get("strategy") == strategy and r["part"] in ("val", "test"):
                groups[(r["seed"], r["fold"])][r["part"]].append(r)
    out = {}
    for key, g in groups.items():
        out[key] = {part: (np.array([r["json_index"] for r in g[part]]), np.array([r["y"] for r in g[part]]),
                           np.array([r["prob"] for r in g[part]])) for part in ("val", "test")}
    return out


def eligible_mask(idx: np.ndarray, smell: str) -> np.ndarray:
    return np.array([common.eligible(int(i), smell) for i in idx])


def per_fold(preds: dict, smell: str):
    """Per (seed, fold): test json_index, label, score, decision at the threshold tuned on eligible validation."""
    k = common.LABELS.index(smell)
    out = {}
    for key, g in preds.items():
        iv, yv, pv = g["val"]
        it, yt, pt = g["test"]
        mv, mt = eligible_mask(iv, smell), eligible_mask(it, smell)
        t = tune_thresholds(yv[mv][:, [k]], pv[mv][:, [k]], 101)[0]
        out[key] = (it[mt], yt[mt, k], pt[mt, k], (pt[mt, k] >= t).astype(int))
    return out


def severity_check(models: dict, out_rows: list) -> None:
    for name, preds in models.items():
        for smell in common.LABELS:
            rho_all, rho_pos = [], []
            for key, (idx, y, s, _) in per_fold(preds, smell).items():
                sev = np.array([SEV[common.severity(int(i), smell)] for i in idx])
                rho_all.append(stats.spearmanr(s, sev).statistic)
                if (y == 1).sum() >= 5 and len(set(sev[y == 1])) > 1:
                    rho_pos.append(stats.spearmanr(s[y == 1], sev[y == 1]).statistic)
            out_rows.append({"model": name, "smell": smell, "spearman_all": float(np.nanmean(rho_all)),
                             "spearman_positives": float(np.nanmean(rho_pos)) if rho_pos else float("nan"),
                             "folds": len(rho_all)})


def corrected_t(diffs: np.ndarray, test_frac: float = 0.2) -> float:
    """Two-sided p of the corrected resampled t-test (Nadeau and Bengio 2003)."""
    n = len(diffs)
    var = diffs.var(ddof=1)
    if var == 0:
        return 1.0
    t = diffs.mean() / np.sqrt((1 / n + test_frac / (1 - test_frac)) * var)
    return float(2 * stats.t.sf(abs(t), df=n - 1))


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj = np.zeros(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def effects_check(models: dict, baseline: str, out_rows: list) -> None:
    for smell in common.LABELS:
        base = {k: matthews_corrcoef(v[1], v[3]) for k, v in per_fold(models[baseline], smell).items()}
        rows = []
        for name, preds in models.items():
            if name == baseline:
                continue
            m = {k: matthews_corrcoef(v[1], v[3]) for k, v in per_fold(preds, smell).items()}
            keys = sorted(set(m) & set(base))
            d = np.array([m[k] - base[k] for k in keys])
            a12 = float(np.mean([(m[k] > base[k]) + 0.5 * (m[k] == base[k]) for k in keys]))
            rows.append({"smell": smell, "model": name, "baseline": baseline, "folds": len(keys),
                         "mean_mcc_diff": float(d.mean()), "a12": a12, "p": corrected_t(d)})
        for r, p in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = p
        out_rows.extend(rows)


def features():
    with (ROOT / "data/derived/size_features.csv").open() as f:
        size = {int(r["sample_idx"]): r for r in csv.DictReader(f)}
    with (ROOT / "data/derived/designite_metrics.csv").open() as f:
        des = {int(r["sample_idx"]): r for r in csv.DictReader(f)}
    scols = [c for c in next(iter(size.values())) if c not in ("sample_idx", "has_parse_error")]
    dcols = [c for c in next(iter(des.values())) if c != "sample_idx" and not c.startswith("y_")]
    names = scols + dcols
    feats = {i: np.array([float(size[i][c] or 0) for c in scols] + [float(des.get(i, {}).get(c) or 0) for c in dcols])
             for i in size}
    return names, feats


def surrogate_check(models: dict, out_rows: list, rules: list) -> None:
    names, feats = features()
    for smell in common.LABELS:
        targets = {}
        for name, preds in models.items():
            dec = {}
            for key, (idx, y, s, d) in per_fold(preds, smell).items():
                if key[0] != 42:
                    continue
                dec.update({int(i): int(v) for i, v in zip(idx, d)})
            targets[name] = dec
        idx = sorted(set.intersection(*[set(v) for v in targets.values()]))
        targets["human label"] = {i: common.label(i, smell) for i in idx}
        x = np.stack([feats[i] for i in idx])
        for name, dec in targets.items():
            y = np.array([dec[i] for i in idx])
            tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=0)
            cvp = cross_val_predict(tree, x, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
            tree.fit(x, y)
            out_rows.append({"smell": smell, "target": name, "n": len(y), "yes_rate": float(y.mean()),
                             "fidelity_agreement": float((cvp == y).mean()),
                             "fidelity_mcc": float(matthews_corrcoef(y, cvp)),
                             "tree_vs_label_mcc": float(matthews_corrcoef([common.label(i, smell) for i in idx], cvp))})
            rules.append(f"### {common.NAMES[smell]}: {name} (fidelity MCC {out_rows[-1]['fidelity_mcc']:.2f})\n\n```\n"
                         + export_text(tree, feature_names=names, decimals=1) + "```\n")


def write(path: Path, rows: list) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", action="append", default=[])
    ap.add_argument("--baseline", default="size_logreg")
    args = ap.parse_args()
    cfg = load_config(Path(args.config), {})["eligible_eval"]
    inputs = [(i["name"], i["predictions"]) for i in cfg["inputs"]] + [tuple(s.split("=", 1)) for s in args.input]
    models = {name: fold_predictions(pattern) for name, pattern in inputs}
    OUT.mkdir(parents=True, exist_ok=True)
    sev, eff, sur, rules = [], [], [], []
    severity_check(models, sev)
    write(OUT / "severity_ranking.csv", sev)
    effects_check(models, args.baseline, eff)
    write(OUT / "effects_vs_size.csv", eff)
    surrogate_check({k: v for k, v in models.items() if k in ("size_logreg", "gcn", "ft_modernbert_2048",
                                                                 "judge_qwen27b", "probe_qwen15b_l14", "bilstm_attention")},
                    sur, rules)
    write(OUT / "surrogates.csv", sur)
    (OUT / "surrogate_rules.md").write_text("# Depth-3 surrogate trees of the model decisions (seed 42 test parts)\n\n"
                                             + "\n".join(rules))
    for r in sev:
        print(f"[severity] {r['model']:20s} {r['smell']:16s} all {r['spearman_all']:.3f} positives {r['spearman_positives']:.3f}")
    for r in eff:
        print(f"[effects] {r['smell']:16s} {r['model']:20s} dMCC {r['mean_mcc_diff']:+.3f} A12 {r['a12']:.2f} "
              f"p_holm {r['p_holm']:.3f}")
    for r in sur:
        print(f"[surrogate] {r['smell']:16s} {r['target']:20s} yes {r['yes_rate']:.2f} fidelity MCC {r['fidelity_mcc']:.2f}")


if __name__ == "__main__":
    main()
