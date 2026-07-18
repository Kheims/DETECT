"""Measure what each detector knows beyond snippet size.

Reports per smell, averaged over seeds:
- roc_auc: plain ROC-AUC
- stratified_auc: ROC-AUC within size strata
- incremental_auc: gain over a size-only model

Usage:
    python scripts/analysis/size_controlled.py --config config/experiments/size_controlled.yml
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_test_scores(patterns: list[str], strategy: str, n_labels: int) -> dict[int, dict[int, np.ndarray]]:
    """{seed: {json_index: score vector}} from the test rows of prediction files."""
    out: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    files = sorted({f for p in patterns for f in glob.glob(str(resolve(p)))})
    if not files:
        raise SystemExit(f"no prediction file matches {patterns}")
    for path in files:
        with open(path) as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("part") != "test" or row.get("strategy") != strategy:
                    continue
                probs = row.get("prob", row.get("probs"))
                out[int(row["seed"])][int(row["json_index"])] = np.asarray(probs[:n_labels], dtype=float)
    return out


def load_parts(patterns: list[str], strategy: str, n_labels: int) -> dict[tuple[int, int], dict[str, dict[int, np.ndarray]]]:
    """{(seed, fold): {"val" | "test": {json_index: score vector}}} from prediction files."""
    out: dict[tuple[int, int], dict[str, dict[int, np.ndarray]]] = defaultdict(lambda: {"val": {}, "test": {}})
    files = sorted({f for p in patterns for f in glob.glob(str(resolve(p)))})
    for path in files:
        with open(path) as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("strategy") != strategy or row.get("part") not in ("val", "test"):
                    continue
                probs = row.get("prob", row.get("probs"))
                out[(int(row["seed"]), int(row["fold"]))][row["part"]][int(row["json_index"])] = \
                    np.asarray(probs[:n_labels], dtype=float)
    return out


def incremental_auc_by_seed(fam: dict, ref: dict, col: int, label_y: dict[int, int], keep) -> dict[int, float]:
    """AUC gain of the validation-fitted combiner over the size-only combiner, per seed."""
    pooled: dict[int, dict[str, list]] = defaultdict(lambda: {"y": [], "two": [], "one": []})
    for (seed, fold), parts in fam.items():
        rparts = ref.get((seed, fold))
        if rparts is None:
            continue
        fit_idx = [i for i in parts["val"] if i in rparts["val"] and keep(i)]
        test_idx = [i for i in parts["test"] if i in rparts["test"] and keep(i)]
        y_fit = np.array([label_y[i] for i in fit_idx])
        if len(test_idx) == 0 or y_fit.min() == y_fit.max():
            continue
        x2 = np.column_stack([logit(np.array([rparts["val"][i][col] for i in fit_idx])),
                              logit(np.array([parts["val"][i][col] for i in fit_idx]))])
        t2 = np.column_stack([logit(np.array([rparts["test"][i][col] for i in test_idx])),
                              logit(np.array([parts["test"][i][col] for i in test_idx]))])
        two = LogisticRegression(max_iter=1000).fit(x2, y_fit)
        one = LogisticRegression(max_iter=1000).fit(x2[:, :1], y_fit)
        pooled[seed]["y"] += [label_y[i] for i in test_idx]
        pooled[seed]["two"] += list(two.predict_proba(t2)[:, 1])
        pooled[seed]["one"] += list(one.predict_proba(t2[:, :1])[:, 1])
    out = {}
    for seed, d in pooled.items():
        y = np.array(d["y"])
        if y.min() != y.max():
            out[seed] = roc_auc_score(y, d["two"]) - roc_auc_score(y, d["one"])
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stratified_auc(y: np.ndarray, s: np.ndarray, strata: np.ndarray) -> float:
    """ROC-AUC over the (positive, negative) pairs of the same stratum."""
    wins = pairs = 0.0
    for k in np.unique(strata):
        m = strata == k
        pos, neg = s[m & (y == 1)], s[m & (y == 0)]
        if len(pos) == 0 or len(neg) == 0:
            continue
        # pairwise comparison by ranks: wins = sum over positives of negatives below (ties count half)
        order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
        ranks = np.empty(len(order))
        values = np.concatenate([neg, pos])[order]
        # average ranks for ties
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[j + 1] == values[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        r_pos = ranks[len(neg):].sum()
        wins += r_pos - len(pos) * (len(pos) + 1) / 2.0
        pairs += len(pos) * len(neg)
    return wins / pairs if pairs else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args, extra = parser.parse_known_args()
    cfg = load_config(args.config, parse_cli_overrides(extra))["size_controlled"]
    results: dict[str, Any] = {}
    for ds_name, ds in cfg["datasets"].items():
        folds = json.loads(resolve(ds["folds"]).read_text())
        labels = list(folds["meta"]["label_order"])
        entries = json.loads(resolve(ds["input_json"]).read_text())
        size = {int(r[ds["size_key"]]): float(r[ds["size_column"]])
                for r in csv.DictReader(resolve(ds["size_csv"]).open())}
        if ds.get("level_csv"):
            level = {int(r[ds["level_key"]]): r[ds["level_column"]] for r in csv.DictReader(resolve(ds["level_csv"]).open())}
        else:
            level = {i: str(e.get("level", "all")) for i, e in enumerate(entries)}
        scores = {name: load_test_scores(list(spec["predictions"]), cfg["strategy"], len(labels))
                  for name, spec in ds["families"].items()}
        parts = {name: load_parts(list(spec["predictions"]), cfg["strategy"], len(labels))
                 for name, spec in ds["families"].items()}
        ref = ds["reference"]
        ds_out: dict[str, Any] = {}
        for col, label in enumerate(labels):
            lv = ds["label_levels"][label]
            per_family: dict[str, Any] = {}
            label_y = {i: int(bool(e["y"][col])) for i, e in enumerate(entries)}
            for name, by_seed in scores.items():
                vals = defaultdict(list)
                inc = ({} if name == ref else incremental_auc_by_seed(
                    parts[name], parts[ref], col, label_y, lambda i: level.get(i) == lv and i in size))
                for seed, rows in sorted(by_seed.items()):
                    idx = sorted(i for i in rows if level.get(i) == lv and i in size
                                 and i in scores[ref].get(seed, {}))
                    if len(idx) < 50:
                        continue
                    y = np.array([int(bool(entries[i]["y"][col])) for i in idx])
                    s = np.array([rows[i][col] for i in idx])
                    loc = np.array([size[i] for i in idx])
                    if ds.get("strata_field"):
                        # strata given by a field of the entries (e.g. the DACOS sampling stratum)
                        names = [json.dumps(entries[i].get(ds["strata_field"])) for i in idx]
                        strata = np.unique(names, return_inverse=True)[1]
                    else:
                        edges = np.unique(np.quantile(loc, np.linspace(0, 1, int(cfg["n_strata"]) + 1)[1:-1]))
                        strata = np.searchsorted(edges, loc, side="right")
                    vals["roc_auc"].append(roc_auc_score(y, s))
                    vals["stratified_auc"].append(stratified_auc(y, s, strata))
                    if name == ref:
                        vals["incremental_auc"].append(0.0)
                    elif seed in inc:
                        vals["incremental_auc"].append(inc[seed])
                    vals["n"].append(len(idx))
                    vals["n_pos"].append(int(y.sum()))
                per_family[name] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                                        "seeds": len(v)} for k, v in vals.items()}
            ds_out[label] = per_family
        results[ds_name] = ds_out
    out = resolve(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    for ds_name, ds_out in results.items():
        dspec = cfg["datasets"][ds_name]
        how = (f"field {dspec['strata_field']}" if dspec.get("strata_field")
               else f"{dspec['size_column']}, {cfg['n_strata']} quantile strata")
        print(f"== {ds_name} (stratified on {how})")
        for label, fams in ds_out.items():
            print(f"  {label}")
            for name, m in fams.items():
                if not m:
                    continue
                print(f"    {name:<26} auc {m['roc_auc']['mean']:.3f}  size-stratified {m['stratified_auc']['mean']:.3f}"
                      f"  +auc over size model {m['incremental_auc']['mean']:+.3f}  (n {m['n']['mean']:.0f}, pos {m['n_pos']['mean']:.0f})")
    print(f"[size_controlled] wrote {out}")


if __name__ == "__main__":
    main()
