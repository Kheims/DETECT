"""Effort-aware ranking metrics and precision under prevalence for MLCQ predictions.

Reads the fold predictions of every model family (no training) and reports
Recall@20%Effort, Popt, IFA, PCI@20% and Precision@10%, plus PPV over a grid
of prevalences. Writes CSV files, a LaTeX table and figures.

Usage:
    python scripts/analysis/effort_aware.py --config config/experiments/industrial.yml
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.constants import LABEL_ORDER  # noqa: E402
from mlcq_graphs.evaluation.cv import iter_folds, load_folds, read_jsonl, tune_thresholds  # noqa: E402

Key = tuple[str, int, int]  # strategy, seed, fold
RANK_METRICS = ["recall_at_budget", "popt", "ifa", "pci_at_budget", "precision_at_top"]
THRESHOLD_METRICS = ["roc_auc", "mcc", "precision", "recall", "specificity"]
DELTA_METRICS = ["recall_at_budget", "popt", "ifa", "pci_at_budget", "precision_at_top"]


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def rel_path(path: Path) -> str:
    """Path relative to the repository when it is inside it, absolute otherwise."""
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def expand(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        found = sorted(PROJECT_ROOT.glob(pattern)) if any(c in pattern for c in "*?[") else [resolve(pattern)]
        if not found or not all(f.exists() for f in found):
            raise SystemExit(f"no file matches {pattern}")
        files.extend(found)
    return files


def update_report(path: Path, sections: dict[str, Any]) -> None:
    """Replace the given top-level sections of the shared JSON report."""
    report = json.loads(path.read_text()) if path.exists() else {}
    report.update(sections)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=1))
    tmp.replace(path)


def load_predictions(patterns: list[str]) -> dict[Key, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]]:
    """{(strategy, seed, fold): {part: {json_index: (y, prob)}}}."""
    out: dict[Key, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]] = defaultdict(lambda: defaultdict(dict))
    for file in expand(patterns):
        for row in read_jsonl(file):
            key = (str(row["strategy"]), int(row["seed"]), int(row["fold"]))
            out[key][row["part"]][int(row["json_index"])] = (
                np.asarray(row["y"], dtype=np.int64), np.asarray(row["prob"], dtype=np.float64))
    return out


def average_members(members: list[dict]) -> dict[Key, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]]:
    """Probability average over the members, on the samples they all cover."""
    out: dict[Key, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]] = defaultdict(lambda: defaultdict(dict))
    for key in set.intersection(*(set(m) for m in members)):
        for part in ("val", "test"):
            common = set.intersection(*(set(m[key][part]) for m in members))
            for i in common:
                out[key][part][i] = (members[0][key][part][i][0],
                                     np.mean([m[key][part][i][1] for m in members], axis=0))
    return out


def load_records(spec: dict[str, Any]) -> dict[Key, dict[str, Any]]:
    """Records of a family keyed like the predictions (combo keys may carry a prefix)."""
    if not spec.get("records"):
        return {}
    wanted = spec.get("records_filter") or {}
    out = {}
    for file in expand([spec["records"]]):
        for rec in json.loads(file.read_text()):
            combo = rec.get("combo", {})
            if any(combo.get(k) != v for k, v in wanted.items()):
                continue
            strategy = next(v for k, v in combo.items() if k.endswith("split_strategy"))
            fold = next(v for k, v in combo.items() if k.endswith("cv_fold"))
            out[(str(strategy), int(rec["seed"]), int(fold))] = rec
    return out


def stack(cell: dict[int, tuple[np.ndarray, np.ndarray]], order: list[int]) -> tuple[np.ndarray, np.ndarray]:
    return np.stack([cell[i][0] for i in order]), np.stack([cell[i][1] for i in order])


def rank_orders(score: np.ndarray, rng: np.random.Generator, n_perm: int) -> list[np.ndarray]:
    """Descending orders of the score, ties broken at random n_perm times."""
    if len(np.unique(score)) == len(score):
        return [np.argsort(-score, kind="stable")]
    return [np.lexsort((rng.random(len(score)), -score)) for _ in range(n_perm)]


def cumulative(order: np.ndarray, y: np.ndarray, effort: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative share of effort and of positives after each inspected sample, from (0, 0)."""
    x = np.concatenate([[0.0], np.cumsum(effort[order]) / effort.sum()])
    p = np.concatenate([[0.0], np.cumsum(y[order]) / y.sum()])
    return x, p


def optimal_order(y: np.ndarray, effort: np.ndarray) -> np.ndarray:
    """Actual density descending: positives first, small ones first."""
    return np.lexsort((effort, -(y / effort)))


def extreme_areas(y: np.ndarray, effort: np.ndarray) -> tuple[float, float]:
    """Areas of the optimal and worst curves (ranking by actual density)."""
    worst = np.lexsort((-effort, y / effort))  # negatives first, then large positives first
    return (float(np.trapezoid(*cumulative(optimal_order(y, effort), y, effort)[::-1])),
            float(np.trapezoid(*cumulative(worst, y, effort)[::-1])))


def order_metrics(order: np.ndarray, y: np.ndarray, effort: np.ndarray, areas: tuple[float, float],
                  budget: float, top: int, grid: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    x, p = cumulative(order, y, effort)
    inspected = x[1:] <= budget + 1e-12
    a_opt, a_worst = areas
    ranked = y[order]
    metrics = {
        "recall_at_budget": float(ranked[inspected].sum() / y.sum()),
        "pci_at_budget": float(inspected.mean()),
        "popt": float((np.trapezoid(p, x) - a_worst) / (a_opt - a_worst)),
        "ifa": float(np.argmax(ranked > 0)),
        "precision_at_top": float(ranked[:top].mean()),
    }
    return metrics, np.interp(grid, x, p)


def ranking_metrics(score: np.ndarray, y: np.ndarray, effort: np.ndarray, rng: np.random.Generator,
                    ecfg: dict[str, Any], grid: np.ndarray, n_perm: int | None = None,
                    orders: list[np.ndarray] | None = None) -> tuple[dict[str, float], np.ndarray]:
    """Metrics and Alberg curve averaged over the tie-breaking (or given) orders."""
    if orders is None:
        orders = rank_orders(score, rng, int(n_perm or ecfg["tie_permutations"]))
    areas = extreme_areas(y, effort)
    top = int(math.ceil(float(ecfg["top_fraction"]) * len(y)))
    runs = [order_metrics(o, y, effort, areas, float(ecfg["effort_budget"]), top, grid) for o in orders]
    metrics = {k: float(np.mean([r[0][k] for r in runs])) for k in RANK_METRICS}
    return metrics, np.mean([r[1] for r in runs], axis=0)


def threshold_metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(np.int64)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    return {
        "roc_auc": float(roc_auc_score(y, prob)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def ppv(sens: float, spec: float, pi: np.ndarray) -> np.ndarray:
    num = sens * pi
    den = num + (1.0 - spec) * (1.0 - pi)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def seed_average(rows: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    """Mean over the folds of a seed, then mean and std over the seeds."""
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_seed[int(r["seed"])].append(r)
    out: dict[str, Any] = {"seeds": len(by_seed), "folds": len(rows)}
    for m in metrics:
        per_seed = [float(np.nanmean([float(r[m]) for r in rs])) for rs in by_seed.values()
                    if any(np.isfinite(float(r[m])) for r in rs)]
        out[m] = float(np.mean(per_seed)) if per_seed else None
        out[f"{m}_sd"] = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None
    return out


def seed_average_curves(curves: list[tuple[int, np.ndarray]]) -> np.ndarray:
    by_seed: dict[int, list[np.ndarray]] = defaultdict(list)
    for seed, c in curves:
        by_seed[seed].append(c)
    return np.nanmean([np.nanmean(cs, axis=0) for cs in by_seed.values()], axis=0)


def paired_delta(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Seed-level difference a - b (each seed averaged over its folds first)."""
    def per_seed(rows: list[dict[str, Any]]) -> dict[int, float]:
        acc: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            acc[int(r["seed"])].append(float(r[metric]))
        return {s: float(np.nanmean(v)) for s, v in acc.items()}
    a, b = per_seed(rows_a), per_seed(rows_b)
    diffs = [a[s] - b[s] for s in sorted(set(a) & set(b))]
    return {"mean": float(np.mean(diffs)), "sd": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None,
            "seeds_higher": int(sum(d > 0 for d in diffs)), "seeds": len(diffs)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args, extra = parser.parse_known_args()
    cfg = load_config(resolve(args.config), parse_cli_overrides(extra))
    ecfg = cfg["effort_aware"]
    label_levels = cfg["label_levels"]
    out_dir = resolve(ecfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = int(ecfg["threshold_steps"])
    grid = np.linspace(0.0, 1.0, int(ecfg["curve_points"]))
    rng = np.random.default_rng(int(ecfg["seed"]))
    effort_models = {str(k): float(v) for k, v in ecfg["effort_models"].items()}
    main_model = str(ecfg["main_effort_model"])

    folds = load_folds(resolve(ecfg["folds"]))
    if list(folds["meta"]["label_order"]) != list(LABEL_ORDER):
        raise SystemExit(f"fold label order {folds['meta']['label_order']} differs from {LABEL_ORDER}")
    entries = json.loads(resolve(ecfg.get("input_json", folds["meta"]["input_json"])).read_text())
    with resolve(ecfg["sample_meta"]).open() as fh:
        meta = {int(r["json_index"]): r for r in csv.DictReader(fh)}
    level = {i: r[ecfg["level_column"]] for i, r in meta.items()}
    with resolve(ecfg["size_features"]).open() as fh:
        loc_of = {int(r[ecfg["size_key"]]): max(float(r[ecfg["effort_column"]]), 1.0) for r in csv.DictReader(fh)}
    fold_list = list(iter_folds(folds, ecfg.get("strategies")))
    indexed = [int(i) for i in folds["meta"]["json_index"]]

    labels_of = {}
    for i in indexed:
        y = entries[i]["y"]
        y = json.loads(y.lower()) if isinstance(y, str) else y
        labels_of[i] = np.asarray([int(bool(v)) for v in y[:len(LABEL_ORDER)]], dtype=np.int64)

    prevalence = {}
    for j, label in enumerate(LABEL_ORDER):
        pool = [i for i in indexed if level[i] == label_levels[label]]
        review_col = ecfg["review_columns"][label]
        prevalence[label] = {"level": label_levels[label], "samples": len(pool),
                             "positives": int(sum(labels_of[i][j] for i in pool)),
                             "prevalence": float(np.mean([labels_of[i][j] for i in pool])),
                             "samples_without_review_of_this_smell": int(sum(
                                 int(float(meta[i][review_col] or 0)) == 0 for i in pool))}
    print(f"[effort] MLCQ prevalence at the level of each smell: "
          f"{ {k: round(v['prevalence'], 4) for k, v in prevalence.items()} }; samples of the level "
          f"without a review of the smell: "
          f"{ {k: v['samples_without_review_of_this_smell'] for k, v in prevalence.items()} }", flush=True)

    # family predictions (ensembles last, from their members)
    preds: dict[str, Any] = {}
    fams = ecfg["families"]
    for name, spec in fams.items():
        if "predictions" in spec:
            preds[name] = load_predictions(list(spec["predictions"]))
    for name, spec in fams.items():
        if "members" in spec:
            preds[name] = average_members([preds[m] for m in spec["members"]])

    fold_rows: list[dict[str, Any]] = []
    curves: dict[tuple[str, str, str, str], list[tuple[int, np.ndarray]]] = defaultdict(list)
    ppv_rows: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    pis = np.geomspace(float(ecfg["prevalence"]["low"]), float(ecfg["prevalence"]["high"]),
                       int(ecfg["prevalence"]["points"]))

    # per fold and smell: test samples of the level, labels and LOC
    cells: dict[tuple[Key, str], tuple[list[int], np.ndarray, np.ndarray]] = {}
    for fold in fold_list:
        key = (fold.strategy, fold.seed, fold.fold)
        for j, label in enumerate(LABEL_ORDER):
            idx = [int(i) for i in fold.test if level[int(i)] == label_levels[label]]
            y = np.asarray([labels_of[i][j] for i in idx], dtype=np.int64)
            loc = np.asarray([loc_of[i] for i in idx], dtype=np.float64)
            if 0 < int(y.sum()) < len(y):
                cells[(key, label)] = (idx, y, loc)
            else:
                print(f"[effort] {key} {label}: test part has one class only, left out", flush=True)

    # unsupervised baselines and the optimal curve
    for (key, label), (idx, y, loc) in cells.items():
        for em, overhead in effort_models.items():
            effort = loc + overhead
            base = {"family": None, "effort_model": em, "strategy": key[0], "seed": key[1], "fold": key[2],
                    "smell": label, "n": len(idx), "n_pos": int(y.sum())}
            for name, score in (("loc_desc", loc), ("loc_asc", -loc)):
                m, c = ranking_metrics(score, y, effort, rng, ecfg, grid)
                fold_rows.append({**base, "family": name, "ranking": "score", **m})
                curves[(name, "score", label, em)].append((key[1], c))
            orders = [rng.permutation(len(y)) for _ in range(int(ecfg["random_permutations"]))]
            m, c = ranking_metrics(np.zeros(len(y)), y, effort, rng, ecfg, grid, orders=orders)
            fold_rows.append({**base, "family": "random", "ranking": "score", **m})
            curves[("random", "score", label, em)].append((key[1], c))
            m, c = ranking_metrics(np.zeros(len(y)), y, effort, rng, ecfg, grid, orders=[optimal_order(y, effort)])
            fold_rows.append({**base, "family": "optimal", "ranking": "score", **m})
            curves[("optimal", "score", label, em)].append((key[1], c))

    for name, pred in preds.items():
        records = load_records(fams[name])
        mcc_diffs: list[float] = []
        missing = 0
        for fold in fold_list:
            key = (fold.strategy, fold.seed, fold.fold)
            if key not in pred:
                missing += 1
                continue
            val_idx = sorted(pred[key]["val"])
            test_idx = sorted(pred[key]["test"])
            if set(test_idx) != {int(i) for i in fold.test}:
                raise SystemExit(f"{name} {key}: test part differs from the fold file")
            y_val, p_val = stack(pred[key]["val"], val_idx)
            y_test, p_test = stack(pred[key]["test"], test_idx)
            if any((labels_of[i] != y_test[k]).any() for k, i in enumerate(test_idx)):
                raise SystemExit(f"{name} {key}: labels differ from the input JSON")
            thr = tune_thresholds(y_val, p_val, steps)
            if key in records:
                for j, label in enumerate(LABEL_ORDER):
                    ref = records[key]["per_label_tuned"][label]["mcc"]
                    if ref is not None:
                        mine = matthews_corrcoef(y_test[:, j], (p_test[:, j] >= thr[j]).astype(np.int64))
                        mcc_diffs.append(abs(float(mine) - float(ref)))
            pos = {i: k for k, i in enumerate(test_idx)}
            for j, label in enumerate(LABEL_ORDER):
                if (key, label) not in cells:
                    continue
                idx, y, loc = cells[(key, label)]
                prob = p_test[[pos[i] for i in idx], j]
                base = {"family": name, "strategy": key[0], "seed": key[1], "fold": key[2], "smell": label,
                        "n": len(idx), "n_pos": int(y.sum())}
                tm = threshold_metrics(y, prob, float(thr[j]))
                for em, overhead in effort_models.items():
                    effort = loc + overhead
                    for ranking, score in (("score", prob), ("density", prob / effort)):
                        m, c = ranking_metrics(score, y, effort, rng, ecfg, grid)
                        row = {**base, "effort_model": em, "ranking": ranking, **m}
                        if ranking == "score":
                            row.update({k: tm[k] for k in THRESHOLD_METRICS})
                        fold_rows.append(row)
                        curves[(name, ranking, label, em)].append((key[1], c))
                ppv_rows.append({**base, **tm})
        checks[name] = {"records": len(records), "folds_missing": missing,
                        "mcc_compared": len(mcc_diffs),
                        "max_abs_mcc_diff": max(mcc_diffs) if mcc_diffs else None}
        print(f"[effort] {name}: {len(fold_list) - missing} folds; per-label test MCC vs records: "
              f"{len(mcc_diffs)} compared, max abs diff "
              f"{checks[name]['max_abs_mcc_diff'] if mcc_diffs else 'n/a'}", flush=True)

    # per-fold CSV
    columns = ["family", "effort_model", "ranking", "smell", "strategy", "seed", "fold", "n", "n_pos",
               *RANK_METRICS, *THRESHOLD_METRICS]
    with (out_dir / "effort_aware_folds.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(fold_rows)

    # aggregation
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in fold_rows:
        groups[(r["family"], r["ranking"], r["smell"], r["effort_model"])].append(r)
    summary: dict[str, Any] = {}
    reference = str(ecfg["reference"])
    for (fam, ranking, label, em), rows in sorted(groups.items()):
        metrics = RANK_METRICS + ([m for m in THRESHOLD_METRICS if m in rows[0]] if ranking == "score" else [])
        agg = seed_average(rows, metrics)
        deltas = {}
        for other_fam, other_rank in ((reference, ranking), ("loc_desc", "score"), ("loc_asc", "score")):
            other = groups.get((other_fam, other_rank, label, em))
            if other and other_fam != fam:
                deltas[f"{other_fam}_{other_rank}"] = {m: paired_delta(rows, other, m) for m in DELTA_METRICS}
        agg["delta_vs"] = deltas
        summary.setdefault(em, {}).setdefault(label, {}).setdefault(fam, {})[ranking] = agg

    table_metrics = ["recall_at_budget", "popt", "ifa", "pci_at_budget", "precision_at_top", "roc_auc", "mcc"]
    with (out_dir / "effort_aware_table.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["effort_model", "smell", "family", "ranking", "recall_at_20pct_effort", "popt", "ifa",
                         "pci_at_20pct_effort", "precision_at_10pct", "roc_auc", "mcc_tuned"])
        for em in effort_models:
            for label in LABEL_ORDER:
                for fam, by_rank in summary[em][label].items():
                    for ranking, agg in by_rank.items():
                        writer.writerow([em, label, fam, ranking] + [
                            "" if agg.get(m) is None else f"{agg[m]:.3f}" for m in table_metrics])

    for em in effort_models:
        for label in LABEL_ORDER:
            print(f"[effort] effort {em}, {label} (level {label_levels[label]}): Recall@"
                  f"{ecfg['effort_budget']:.0%}Effort / Popt / IFA / PCI@{ecfg['effort_budget']:.0%} / "
                  f"P@{ecfg['top_fraction']:.0%}, score ranking | density ranking")
            for fam, by_rank in summary[em][label].items():
                s = by_rank["score"]
                line = (f"  {fam:24s} {s['recall_at_budget']:.3f} {s['popt']:.3f} {s['ifa']:6.2f} "
                        f"{s['pci_at_budget']:.3f} {s['precision_at_top']:.3f}")
                if "density" in by_rank:
                    d = by_rank["density"]
                    line += (f" | {d['recall_at_budget']:.3f} {d['popt']:.3f} {d['ifa']:6.2f} "
                             f"{d['pci_at_budget']:.3f} {d['precision_at_top']:.3f}")
                print(line, flush=True)

    curve_out = {f"{fam}|{ranking}|{label}|{em}": seed_average_curves(cs).round(5).tolist()
                 for (fam, ranking, label, em), cs in curves.items()}

    # precision under prevalence
    pcfg = ecfg["prevalence"]
    table_pis = np.asarray([float(p) for p in pcfg["table"]])
    ppv_summary: dict[str, Any] = {}
    ppv_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in ppv_rows:
        ppv_groups[(r["family"], r["smell"])].append(r)
    for (fam, label), rows in sorted(ppv_groups.items()):
        agg = seed_average(rows, ["recall", "specificity", "precision", "mcc"])
        pi_mlcq = prevalence[label]["prevalence"]
        at = {}
        for pi in [*table_pis, pi_mlcq]:
            vals = [(r["seed"], float(ppv(r["recall"], r["specificity"], np.asarray([pi]))[0])) for r in rows]
            agg_pi = seed_average([{"seed": s, "ppv": v} for s, v in vals], ["ppv"])
            # PPV of the seed-averaged rates, not the mean of the fold PPVs
            at[f"{pi:.4g}"] = {"ppv": float(ppv(agg["recall"], agg["specificity"], np.asarray([pi]))[0]),
                               "ppv_mean_of_folds": agg_pi["ppv"], "ppv_mean_of_folds_sd": agg_pi["ppv_sd"],
                               "folds_undefined": int(sum(not np.isfinite(v) for _, v in vals)),
                               "folds_without_false_positive": int(sum(r["fp"] == 0 for r in rows))}
        agg["ppv_at"] = at
        agg["mlcq_prevalence"] = pi_mlcq
        ppv_summary.setdefault(label, {})[fam] = agg
    ppv_curve_out = {f"{fam}|{label}": ppv(agg["recall"], agg["specificity"], pis).round(5).tolist()
                     for label, by_fam in ppv_summary.items() for fam, agg in by_fam.items()}
    for label in LABEL_ORDER:
        pi_mlcq = prevalence[label]["prevalence"]
        print(f"[ppv] {label}: sens / spec at the val-tuned threshold, PPV (plug-in) at "
              f"{', '.join(f'{p:.3g}' for p in table_pis)} and at the MLCQ prevalence {pi_mlcq:.4f}")
        for fam, agg in ppv_summary[label].items():
            vals = " ".join("  nan" if agg["ppv_at"][f"{p:.4g}"]["ppv"] is None
                            else f"{agg['ppv_at'][f'{p:.4g}']['ppv']:.3f}" for p in [*table_pis, pi_mlcq])
            print(f"  {fam:24s} {agg['recall']:.3f} {agg['specificity']:.3f} | {vals} "
                  f"| observed precision {agg['precision']:.3f}", flush=True)

    (out_dir / "curves.json").write_text(json.dumps({"grid": grid.round(5).tolist(), "alberg": curve_out,
                                                    "ppv_grid": pis.tolist(), "ppv": ppv_curve_out}))
    labels = {**{k: v["label"] for k, v in fams.items()}, **ecfg["baseline_labels"]}
    tex = write_latex_table(cfg, summary[main_model], labels, out_dir / "effort_aware_table.tex")
    figure_paths = make_figures(cfg, curves, ppv_curve_out, grid, pis, prevalence, effort_models)

    budget = float(ecfg["effort_budget"])
    update_report(resolve(cfg["report"]), {
        "effort_aware": {
            "definition": {
                "population": "test part of every kfold split, samples of the level of the smell only "
                              "(method and class LOC never share a budget)",
                "effort": f"{ecfg['effort_column']} (minimum 1) plus the per-item overhead of the effort model",
                "effort_models": effort_models, "main_effort_model": main_model,
                "recall_at_budget": f"Recall@{budget:.0%}Effort as in Kamei et al. TSE 2013 (called ACC by "
                                    "Yang et al. 2016): items inspected in ranked order, an item counts only "
                                    f"when the cumulative effort including it is at most {budget:.0%} of the "
                                    "total effort of the cell, no partial credit for the item crossing the budget",
                "popt": "normalized Popt = (A_model - A_worst) / (A_optimal - A_worst), as in Yang et al. FSE "
                        "2016 (origin Mende and Koschke PROMISE 2009); areas by the trapezoidal rule under "
                        "the cumulative share of positives against the cumulative share of effort, from (0, 0) "
                        "to (1, 1); optimal = actual density descending (ties: smaller effort first), worst = "
                        "actual density ascending (ties: larger effort first)",
                "ifa": "number of negatives ranked before the first positive (Huang, Xia and Lo ICSME 2017)",
                "pci_at_budget": f"share of the items inspected within the {budget:.0%} effort budget (PCI@20% "
                                 "of Huang, Xia and Lo ICSME 2017)",
                "precision_at_top": f"precision of the top ceil({float(ecfg['top_fraction']):.0%} n) items",
                "top_fraction": ecfg["top_fraction"], "budget": budget,
                "tie_permutations": ecfg["tie_permutations"], "random_permutations": ecfg["random_permutations"],
                "rankings": {"score": "model score descending",
                             "density": "score / effort descending (effort of the effort model)"},
                "baselines": {"loc_desc": "LOC descending", "loc_asc": "LOC ascending",
                              "random": "random order", "optimal": "actual density descending"},
                "averaging": "mean over the folds of a seed, then mean and sd over the seeds; delta_vs = "
                             "paired seed-level difference (family minus other)",
            },
            "family_labels": labels,
            "threshold_check": checks,
            "mlcq_prevalence": prevalence,
            "summary": summary,
            "files": {"folds_csv": rel_path(out_dir / "effort_aware_folds.csv"),
                      "table_csv": rel_path(out_dir / "effort_aware_table.csv"),
                      "table_tex": rel_path(tex), "curves": rel_path(out_dir / "curves.json")},
            "figures": figure_paths,
        },
        "prevalence_ppv": {
            "definition": "PPV(pi) = sens*pi / (sens*pi + (1-spec)*(1-pi)); sens and spec of every fold at the "
                          "val-tuned threshold on the level-restricted test part, averaged per seed then over seeds; "
                          "ppv uses the averaged rates (plug-in), ppv_mean_of_folds averages the fold PPVs",
            "assumptions": "sensitivity and specificity are assumed unchanged when only the prevalence changes; "
                           "the adjustment cannot correct the MLCQ sampling windows (four selection phases, the "
                           "more-than-4-lines filter of phase 3), project shift, reviewer shift or a different "
                           "mix of hard and easy negatives in a real code base",
            "summary": ppv_summary,
        },
        "mlcq_sampling": cfg.get("literature", {}).get("mlcq_sampling"),
        "citations": {k: v for k, v in cfg.get("literature", {}).items() if k != "mlcq_sampling"},
    })
    print(f"[effort] wrote {resolve(cfg['report'])} and {out_dir}", flush=True)


def write_latex_table(cfg: dict[str, Any], summary: dict[str, Any], labels: dict[str, str], path: Path) -> Path:
    """Main effort model: score ranking metrics and density ranking Recall and Popt per smell."""
    ecfg = cfg["effort_aware"]
    order = [*ecfg["table"]["families"], *ecfg["table"]["baselines"]]
    lines = ["% scripts/analysis/effort_aware.py, " + str(ecfg["main_effort_model"]) + " effort model",
             r"\begin{tabular}{l rrrrr rr}", r"\toprule",
             " & ".join(ecfg["table"]["header"]) + r" \\", r"\midrule"]
    for label in LABEL_ORDER:
        lines.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{cfg['smell_labels'][label]}}}}} \\")
        for fam in order:
            by_rank = summary[label].get(fam)
            if not by_rank:
                continue
            s, d = by_rank["score"], by_rank.get("density")
            cells = [f"{s['recall_at_budget']:.3f}", f"{s['popt']:.3f}", f"{s['ifa']:.1f}",
                     f"{s['pci_at_budget']:.3f}", f"{s['precision_at_top']:.3f}",
                     f"{d['recall_at_budget']:.3f}" if d else "", f"{d['popt']:.3f}" if d else ""]
            lines.append(f"{labels[fam]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if label != LABEL_ORDER[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n")
    return path


def make_figures(cfg: dict[str, Any], curves: dict, ppv_curves: dict[str, list[float]], grid: np.ndarray,
                 pis: np.ndarray, prevalence: dict[str, Any], effort_models: dict[str, float]) -> dict[str, list[str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.cv.make_figures import PALETTE, style

    fcfg, ecfg = cfg["figures"], cfg["effort_aware"]
    style(fcfg)
    fig_dir = resolve(fcfg["dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = {**{k: v["label"] for k, v in ecfg["families"].items()}, **ecfg["baseline_labels"]}
    smell_labels = cfg["smell_labels"]
    level_labels = fcfg["level_labels"]
    written: dict[str, list[str]] = {}

    def save(fig: Any, name: str) -> None:
        paths = []
        for ext in ("pdf", "png"):
            p = fig_dir / f"{name}.{ext}"
            fig.savefig(p, dpi=int(fcfg["dpi"]))
            paths.append(rel_path(p))
        plt.close(fig)
        written[name] = paths
        print(f"[figures] {name}", flush=True)

    acfg = ecfg["alberg"]
    for em in effort_models:
        for ranking in ("score", "density"):
            fig, axes = plt.subplots(1, len(LABEL_ORDER), figsize=tuple(acfg["size"]), sharey=True)
            for ax, label in zip(axes, LABEL_ORDER):
                ax.plot(grid, seed_average_curves(curves[("optimal", "score", label, em)]), color="black",
                        linewidth=1.2, label=labels["optimal"])
                ax.plot(grid, seed_average_curves(curves[("random", "score", label, em)]), color="grey",
                        linestyle=":", linewidth=1.0, label=labels["random"])
                for k, base in enumerate(acfg["baselines"]):
                    ax.plot(grid, seed_average_curves(curves[(base, "score", label, em)]), color="grey",
                            linestyle=["--", "-."][k % 2], linewidth=1.0, label=labels[base])
                for k, fam in enumerate(acfg["families"]):
                    ax.plot(grid, seed_average_curves(curves[(fam, ranking, label, em)]),
                            color=PALETTE[k % len(PALETTE)], linewidth=1.1, label=labels[fam])
                ax.axvline(float(ecfg["effort_budget"]), color="grey", linewidth=0.6)
                ax.set_title(f"{smell_labels[label]} ({level_labels[prevalence[label]['level']]})")
                ax.set_xlabel(acfg["xlabel"][em])
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1.01)
            axes[0].set_ylabel(acfg["ylabel"])
            handles, names = axes[0].get_legend_handles_labels()
            fig.tight_layout(rect=(0, 0.16, 1, 1))
            fig.legend(handles, names, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), frameon=False)
            save(fig, f"industrial_alberg_{ranking}_{em}")

    pcfg = ecfg["prevalence"]
    fig, axes = plt.subplots(1, len(LABEL_ORDER), figsize=tuple(pcfg["size"]), sharey=True)
    for ax, label in zip(axes, LABEL_ORDER):
        for k, fam in enumerate(pcfg["families"]):
            ax.plot(pis, ppv_curves[f"{fam}|{label}"], color=PALETTE[k % len(PALETTE)],
                    linewidth=1.1, label=labels[fam])
        ax.plot(pis, pis, color="grey", linestyle=":", linewidth=1.0, label=labels["random"])
        ax.axvline(prevalence[label]["prevalence"], color="black", linestyle="--", linewidth=0.8)
        ax.text(prevalence[label]["prevalence"], 0.98, f"MLCQ {prevalence[label]['prevalence']:.1%} ", ha="right",
                fontsize=7, va="top")
        ax.set_xscale("log")
        ax.set_title(f"{smell_labels[label]} ({level_labels[prevalence[label]['level']]})")
        ax.set_xlabel(pcfg["xlabel"])
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel(pcfg["ylabel"])
    handles, names = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.legend(handles, names, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), frameon=False)
    save(fig, "industrial_ppv_prevalence")
    return written


if __name__ == "__main__":
    main()
