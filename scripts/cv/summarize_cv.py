"""Summarise cross-validation records of any model family and compare them.

Inputs are NAME=PATH pairs. Metrics are averaged over the folds of a seed, then over seeds.
--baseline adds paired comparisons with the corrected resampled t-test and Holm adjustment.

Usage:
    python scripts/cv/summarize_cv.py gnn=artifacts/reports/gnn_cv xgb=artifacts/tabular_cv/x \
        --baseline gnn --out artifacts/summaries/main
    python scripts/cv/summarize_cv.py designite__xgb=artifacts/dacos/tabular_cv/designite__xgb --strategy group_kfold
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

FOLD_KEYS = ("cv_fold", "training.cv_fold")
STRATEGY_KEYS = ("split_strategy", "training.split_strategy")
METRICS = ("mcc", "f1", "pr_auc", "roc_auc")


def _combo(record: dict[str, Any]) -> dict[str, Any]:
    combo = record.get("combo", {})
    if isinstance(combo, str):
        combo = ast.literal_eval(combo)
    return dict(combo)


def _per_label(record: dict[str, Any]) -> dict[str, Any]:
    per_label = record.get("per_label_tuned", {})
    return ast.literal_eval(per_label) if isinstance(per_label, str) else per_label


def load_groups(
    specs: list[str], strategy: str | None = None,
) -> tuple[dict[str, dict[tuple[str, int, int], dict[str, float]]], list[str]]:
    """{group: {(strategy, seed, fold): {metric: value}}} and the label names of the records."""
    groups: dict[str, dict[tuple[str, int, int], dict[str, float]]] = {}
    labels: list[str] = []
    for spec in specs:
        name, raw = spec.split("=", 1)
        path = Path(raw)
        files = [path] if path.is_file() else sorted(path.glob("**/records.json"))
        if not files:
            raise SystemExit(f"no records under {path}")
        for file in files:
            for record in json.loads(file.read_text()):
                combo = _combo(record)
                fold = next((int(combo.pop(k)) for k in FOLD_KEYS if k in combo), 0)
                split = next((str(combo[k]) for k in STRATEGY_KEYS if k in combo), "random")
                if strategy is not None and split != strategy:
                    continue
                for k in ("run", "model", "features"):
                    combo.pop(k, None)
                key = name if not combo else f"{name} {json.dumps(combo, sort_keys=True)}"
                values: dict[str, float] = {"macro_mcc": float(record["mcc_tuned"])}
                if record.get("pr_auc_tuned") is not None:
                    values["macro_pr_auc"] = float(record["pr_auc_tuned"])
                values["macro_f1"] = float(record["f1_macro_tuned"])
                if record.get("roc_auc_tuned") is not None:
                    values["macro_roc_auc"] = float(record["roc_auc_tuned"])
                for label, m in _per_label(record).items():
                    if label not in labels:
                        labels.append(label)
                    for metric in METRICS:
                        if m.get(metric) is not None:
                            values[f"{label.removeprefix('is_')}_{metric}"] = float(m[metric])
                cell = (split, int(record["seed"]), fold)
                if cell in groups.setdefault(key, {}):
                    raise SystemExit(f"duplicate record for {key} seed/fold {cell}")
                groups[key][cell] = values
    return groups, labels


def seed_means(cells: dict[tuple[str, int, int], dict[str, float]], metric: str) -> list[float]:
    by_seed: dict[int, list[float]] = {}
    for (_, seed, _), values in cells.items():
        if metric in values:
            by_seed.setdefault(seed, []).append(values[metric])
    return [statistics.mean(v) for _, v in sorted(by_seed.items())]


def corrected_ttest(diffs: np.ndarray, test_train_ratio: float) -> tuple[float, float, float]:
    """t statistic, two-sided p value and half-width of the corrected 95% interval."""
    n = len(diffs)
    var = float(np.var(diffs, ddof=1))
    if n < 2 or var == 0.0:
        return float("nan"), float("nan"), float("nan")
    se = math.sqrt((1.0 / n + test_train_ratio) * var)
    t = float(np.mean(diffs)) / se
    return t, float(2 * stats.t.sf(abs(t), df=n - 1)), float(stats.t.ppf(0.975, df=n - 1) * se)


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [float("nan")] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def ratio_from_folds(path: Path, strategies: list[str] | None = None, tolerance: float = 0.01) -> float:
    """Mean n_test / n_train over the splits of the given strategies, which must agree within tolerance."""
    splits = json.loads(path.read_text())["splits"]
    if strategies is None:
        strategies = sorted({str(s["strategy"]) for s in splits})
    by_strategy = {st: [len(s["test"]) / len(s["train"]) for s in splits if s["strategy"] == st] for st in strategies}
    absent = [st for st, v in by_strategy.items() if not v]
    if absent:
        raise SystemExit(f"{path} has no splits of strategies {absent}")
    means = {st: statistics.mean(v) for st, v in by_strategy.items()}
    if max(means.values()) > (1.0 + tolerance) * min(means.values()):
        shown = {st: round(m, 4) for st, m in means.items()}
        raise SystemExit(f"split strategies with different test/train ratios {shown} would share one ratio: "
                         "pass --strategy, or --test-train-ratio")
    return statistics.mean(r for v in by_strategy.values() for r in v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="NAME=PATH")
    parser.add_argument("--baseline", default=None,
                        help="group to compare against: full group key, or a NAME matching exactly one group")
    parser.add_argument("--strategy", default=None, help="keep only records of this split strategy")
    parser.add_argument("--folds", type=Path, default=None,
                        help="folds file used to compute n_test / n_train for the corrected t-test, over the "
                             "splits of the strategies of the compared records")
    parser.add_argument("--test-train-ratio", type=float, default=None,
                        help="n_test / n_train of one fold (overrides --folds)")
    parser.add_argument("--ratio-tolerance", type=float, default=0.01,
                        help="largest relative difference between the test/train ratios of the strategies of "
                             "the records for one pooled ratio (default 0.01)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    groups, labels = load_groups(args.inputs, args.strategy)
    if not groups:
        raise SystemExit(f"no records of strategy {args.strategy!r} in {args.inputs}")
    if args.baseline is not None and args.baseline not in groups:
        matches = [g for g in groups if g == args.baseline or g.startswith(args.baseline + " ")]
        if len(matches) != 1:
            raise SystemExit(f"--baseline {args.baseline!r} matches {len(matches)} groups: {matches}")
        args.baseline = matches[0]
    ratio = args.test_train_ratio
    if ratio is None and args.folds is not None and args.baseline is not None:
        used = sorted({st for cells in groups.values() for st, _, _ in cells})
        ratio = ratio_from_folds(args.folds, used, args.ratio_tolerance)
    if args.baseline is not None and ratio is None:
        raise SystemExit("--baseline needs --folds or --test-train-ratio")
    shorts = [label.removeprefix("is_") for label in labels]
    keys = ([f"{s}_mcc" for s in shorts] + ["macro_mcc", "macro_pr_auc"] + [f"{s}_pr_auc" for s in shorts]
            + ["macro_roc_auc"] + [f"{s}_roc_auc" for s in shorts])

    rows = []
    for name, cells in groups.items():
        row: dict[str, Any] = {"group": name, "runs": len(cells),
                               "seeds": len({s for _, s, _ in cells}), "folds": len({f for _, _, f in cells})}
        for key in keys:
            means = seed_means(cells, key)
            row[f"{key}_mean"] = statistics.mean(means) if means else float("nan")
            row[f"{key}_std"] = statistics.pstdev(means) if len(means) > 1 else 0.0
        rows.append(row)

    print("| group | runs | " + " | ".join(f"{s} MCC" for s in shorts) + " | macro MCC | macro PR-AUC |")
    print("|---|---|" + "---|" * (len(shorts) + 2))
    for row in rows:
        cells = [f"{row[f'{k}_mean']:.3f} ± {row[f'{k}_std']:.3f}" for k in keys[: len(shorts) + 2]]
        print(f"| {row['group']} | {row['runs']} | " + " | ".join(cells) + " |")

    comparisons = []
    if args.baseline is not None:
        base = groups[args.baseline]
        for name, cells in groups.items():
            if name == args.baseline:
                continue
            shared = sorted(set(base) & set(cells))
            for key in ("macro_mcc", *[f"{s}_mcc" for s in shorts]):
                pairs = [(cells[c][key], base[c][key]) for c in shared if key in cells[c] and key in base[c]]
                if len(pairs) < 2:
                    continue
                diffs = np.array([a - b for a, b in pairs])
                t, p, half = corrected_ttest(diffs, ratio)
                try:
                    w_p = float(stats.wilcoxon(diffs).pvalue)
                except ValueError:
                    w_p = float("nan")
                comparisons.append({"group": name, "baseline": args.baseline, "metric": key, "n": len(pairs),
                                    "mean_diff": float(np.mean(diffs)), "wins": int((diffs > 0).sum()),
                                    "ci95_low": float(np.mean(diffs)) - half, "ci95_high": float(np.mean(diffs)) + half,
                                    "corrected_t": t, "corrected_p": p, "wilcoxon_p_descriptive": w_p})
        for metric in {c["metric"] for c in comparisons}:
            subset = [c for c in comparisons if c["metric"] == metric]
            for c, adj in zip(subset, holm([c["corrected_p"] for c in subset])):
                c["corrected_p_holm"] = adj
        print(f"\npaired comparisons against {args.baseline} (corrected resampled t-test, test/train ratio {ratio:.3f}, "
              "Holm over groups per metric):")
        for c in comparisons:
            print(f"  {c['group']:<40} {c['metric']:<17} diff={c['mean_diff']:+.3f} "
                  f"[{c['ci95_low']:+.3f}, {c['ci95_high']:+.3f}] wins={c['wins']}/{c['n']} "
                  f"p={c['corrected_p']:.3g} p_holm={c['corrected_p_holm']:.3g}")

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "summary.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        (args.out / "summary.json").write_text(json.dumps({"rows": rows, "comparisons": comparisons,
                                                           "inputs": args.inputs}, indent=1))
        cells_out = {name: {f"{st}_{s}_{f}": v for (st, s, f), v in cells.items()} for name, cells in groups.items()}
        (args.out / "cells.json").write_text(json.dumps(cells_out))
        print(f"[summary] wrote {args.out}")


if __name__ == "__main__":
    main()
