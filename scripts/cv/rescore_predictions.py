"""Re-score saved cross-validation predictions under other labels and threshold rules.

A variant sets the label set, eligible-only scoring and the threshold rule (max_f1, max_mcc or fixed).
A validation bootstrap measures threshold stability. Writes records.json per variant, summary.json and table.csv.

Usage:
    python scripts/cv/rescore_predictions.py --config config/experiments/rescoring.yml
    python scripts/cv/rescore_predictions.py --config ... --rescoring.bootstrap.n_boot=200
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import math
import os
import statistics
import sys
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cv.eval_eligible import FOLD_KEYS, STRATEGY_KEYS, compose, label_metrics  # noqa: E402
from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.constants import LABEL_ORDER  # noqa: E402
from mlcq_graphs.evaluation.cv import iter_folds, label_matrix_from_json, load_folds, read_jsonl, tune_thresholds  # noqa: E402
from mlcq_graphs.evaluation.metrics import compute_full_metrics  # noqa: E402
from scripts.cv.summarize_cv import corrected_ttest, holm, load_groups, ratio_from_folds, seed_means  # noqa: E402

LABELS = list(LABEL_ORDER)
SHORTS = [label.removeprefix("is_") for label in LABELS]
RULES = ("max_f1", "max_mcc", "fixed")
TABLE_METRICS = [f"{s}_mcc" for s in SHORTS] + ["macro_mcc", "macro_f1", "macro_pr_auc"]


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # per-process temporary name for concurrent writers
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def finite(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


# inputs

@dataclass
class Cell:
    """Validation and test predictions of one (strategy, seed, fold), sorted by json_index."""
    strategy: str
    seed: int
    fold: int
    model: str
    source: str
    idx: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    p: dict[str, np.ndarray]
    source_record: dict[str, Any] | None = None

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.strategy, self.seed, self.fold)


@dataclass
class Input:
    name: str
    base: str
    family: str
    training_labels: str
    variants: list[str]
    cells: list[Cell] = field(default_factory=list)
    files_found: list[str] = field(default_factory=list)
    files_missing: list[str] = field(default_factory=list)


def record_key(record: dict[str, Any]) -> tuple[str, int, int] | None:
    combo = record.get("combo", {})
    fold = next((int(combo[k]) for k in FOLD_KEYS if k in combo), None)
    strategy = next((str(combo[k]) for k in STRATEGY_KEYS if k in combo), None)
    if fold is None or strategy is None:
        return None
    return (strategy, int(record["seed"]), fold)


def prediction_files(spec: dict[str, Any]) -> tuple[list[tuple[Path, dict[str, Any] | None]], list[str]]:
    """(predictions file, source record or None) pairs, and the missing files."""
    if spec.get("from_records") is not None:
        fr = spec["from_records"]
        records = json.loads(resolve(fr["records"]).read_text())
        found, missing = [], []
        for rec in records:
            run = Path(str(rec[str(fr.get("field", "run_dir"))]).rstrip("/")).name
            path = resolve(fr["root"]) / run / str(fr.get("file", "predictions.jsonl"))
            (found if path.exists() else missing).append((path, rec))
        return found, [rel(p) for p, _ in missing]
    patterns = spec["predictions"] if isinstance(spec["predictions"], list) else [spec["predictions"]]
    files, missing = [], []
    for pattern in patterns:
        hits = sorted(glob.glob(str(resolve(pattern))))
        if not hits:
            missing.append(str(pattern))
        files.extend(Path(h) for h in hits)
    return [(f, None) for f in sorted(set(files))], missing


def side_records(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    rec_path = path.parent / "records.json"
    if not rec_path.exists():
        return {}
    out = {}
    for record in json.loads(rec_path.read_text()):
        key = record_key(record)
        if key is not None:
            out[key] = record
    return out


def load_input(spec: dict[str, Any], variant_names: list[str], strategies: list[str] | None) -> Input:
    name = str(spec["name"])
    inp = Input(name=name, base=str(spec.get("base", name)), family=str(spec.get("family", "")),
                training_labels=str(spec.get("training_labels", "stored")),
                variants=[str(v) for v in spec.get("variants", variant_names)])
    unknown = [v for v in inp.variants if v not in variant_names]
    if unknown:
        raise SystemExit(f"input {name}: unknown variants {unknown}")
    files, inp.files_missing = prediction_files(spec)
    seen: dict[tuple[str, int, int], str] = {}
    for path, source_record in files:
        inp.files_found.append(rel(path))
        side = side_records(path) if source_record is None else {}
        groups: dict[tuple[str, int, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        models = set()
        for row in read_jsonl(path):
            groups[(str(row["strategy"]), int(row["seed"]), int(row["fold"]))][str(row["part"])].append(row)
            models.add(str(row["model"]))
        for key, parts in sorted(groups.items()):
            if strategies is not None and key[0] not in strategies:
                continue
            if key in seen:
                raise SystemExit(f"input {name}: cell {key} in {seen[key]} and {rel(path)}")
            seen[key] = rel(path)
            if "val" not in parts or "test" not in parts:
                raise SystemExit(f"{rel(path)}: cell {key} lacks a val or test part")
            idx, ys, ps = {}, {}, {}
            for part in ("val", "test"):
                rows = sorted(parts[part], key=lambda r: int(r["json_index"]))
                idx[part] = np.array([int(r["json_index"]) for r in rows], dtype=np.int64)
                if len(np.unique(idx[part])) != len(idx[part]):
                    raise SystemExit(f"{rel(path)}: cell {key} has duplicate json_index in {part}")
                ys[part] = np.array([r["y"] for r in rows], dtype=np.int64)
                ps[part] = np.array([r["prob"] for r in rows], dtype=np.float64)
            src = source_record if source_record is not None else side.get(key)
            if source_record is not None and record_key(source_record) != key:
                raise SystemExit(f"{rel(path)}: rows of {key} but the source record is {record_key(source_record)}")
            inp.cells.append(Cell(strategy=key[0], seed=key[1], fold=key[2], model="|".join(sorted(models)),
                                  source=rel(path), idx=idx, y=ys, p=ps, source_record=src))
    inp.cells.sort(key=lambda c: c.key)
    return inp


# thresholds

def threshold_grid(steps: int) -> np.ndarray:
    """The grid of cv.tune_thresholds (float32 linspace), as float64 values."""
    return np.linspace(0.0, 1.0, max(int(steps), 3), dtype=np.float32).astype(np.float64)


def grid_counts(y: np.ndarray, p: np.ndarray, grid: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, ...]:
    """Weighted tp, fp, fn, tn of p >= t per grid threshold (rows) and weight column."""
    order = np.argsort(p, kind="stable")
    ps, ws, ys = p[order], w[order], y[order].astype(np.float64)
    n = len(ps)
    zero = np.zeros((1, w.shape[1]))
    suffix_w = np.vstack([zero, np.cumsum(ws[::-1], axis=0)])
    suffix_wy = np.vstack([zero, np.cumsum((ws * ys[:, None])[::-1], axis=0)])
    above = n - np.searchsorted(ps, grid, side="left")
    tp = suffix_wy[above]
    pred = suffix_w[above]
    pos = suffix_wy[n][None, :]
    tot = suffix_w[n][None, :]
    fp = pred - tp
    fn = pos - tp
    tn = tot - tp - fp - fn
    return tp, fp, fn, tn


def f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: np.ndarray) -> np.ndarray:
    """Binary F1 with zero_division=0, as sklearn.metrics.f1_score."""
    den = 2 * tp + fp + fn
    return np.divide(2 * tp, den, out=np.zeros_like(den), where=den > 0)


def mcc_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: np.ndarray) -> np.ndarray:
    """MCC, 0 when a margin of the confusion matrix is empty (as sklearn)."""
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(tp * tn - fp * fn, den, out=np.zeros_like(den), where=den > 0)


SCORES = {"max_f1": f1_from_counts, "max_mcc": mcc_from_counts}


def grid_choice(y: np.ndarray, p: np.ndarray, grid: np.ndarray, w: np.ndarray, rule: str) -> tuple[np.ndarray, np.ndarray]:
    """Best grid threshold index per weight column, and the columns without a positive."""
    counts = grid_counts(y, p, grid, w)
    choice = np.argmax(SCORES[rule](*counts), axis=0)
    no_positive = (w * y[:, None]).sum(axis=0) == 0
    return choice, no_positive


def tune(rule: str, y: np.ndarray, p: np.ndarray, steps: int, fixed_value: float) -> np.ndarray:
    """Per-label thresholds of a rule, as float32 values."""
    if rule == "max_f1":
        return tune_thresholds(y, p, steps)
    if rule == "fixed":
        return np.full(y.shape[1], fixed_value, dtype=np.float32)
    if rule != "max_mcc":
        raise SystemExit(f"unknown threshold rule {rule}")
    grid32 = np.linspace(0.0, 1.0, max(int(steps), 3), dtype=np.float32)
    out = np.full(y.shape[1], 0.5, dtype=np.float32)
    for i in range(y.shape[1]):
        if int(y[:, i].sum()) == 0:
            continue
        choice, _ = grid_choice(y[:, i], p[:, i], grid32.astype(np.float64), np.ones((len(y), 1)), rule)
        out[i] = grid32[int(choice[0])]
    return out


# scoring

@dataclass
class Context:
    label_sets: dict[str, np.ndarray]
    eligible: np.ndarray  # (n_samples, L) bool
    steps: int
    folds: dict[tuple[str, int, int], Any]


def variant_arrays(cell: Cell, variant: dict[str, Any], ctx: Context) -> dict[str, Any]:
    labels = ctx.label_sets[str(variant["labels"])]
    out: dict[str, Any] = {}
    for part in ("val", "test"):
        out[f"y_{part}"] = labels[cell.idx[part]]
        out[f"p_{part}"] = cell.p[part]
        out[f"m_{part}"] = (ctx.eligible[cell.idx[part]] if bool(variant.get("eligible_only", False))
                            else np.ones_like(out[f"y_{part}"], dtype=bool))
        out[f"changed_{part}"] = [int(v) for v in (out[f"y_{part}"] != cell.y[part]).sum(axis=0)]
    return out


def score_cell(inp: Input, cell: Cell, vname: str, variant: dict[str, Any], ctx: Context) -> dict[str, Any]:
    """One record of a cell, scored with the labels and threshold rule of the variant."""
    a = variant_arrays(cell, variant, ctx)
    rule = str(variant["threshold"])
    fixed_value = float(variant.get("fixed_value", 0.5))
    eligible_only = bool(variant.get("eligible_only", False))
    combo = {"run": inp.name, "model": cell.model, "split_strategy": cell.strategy, "cv_fold": cell.fold,
             "rescoring_variant": vname, "training_labels": inp.training_labels}
    meta = {"variant": vname, "labels": str(variant["labels"]), "threshold_rule": rule,
            "fixed_value": fixed_value if rule == "fixed" else None, "eligible_only": eligible_only,
            "training_labels": inp.training_labels, "base": inp.base,
            "labels_changed_vs_rows": {"val": a["changed_val"], "test": a["changed_test"]}}
    if not eligible_only:
        thresholds = tune(rule, a["y_val"], a["p_val"], ctx.steps, fixed_value)
        names = LABELS[: a["y_test"].shape[1]]

        def metrics(t: np.ndarray) -> dict[str, Any]:
            pred = (a["p_test"] >= t.reshape(1, -1)).astype(np.int64)
            return compute_full_metrics(a["y_test"].astype(np.int64), pred, a["p_test"].astype(np.float64), names)

        tuned, fixed = metrics(thresholds), metrics(np.full(len(names), 0.5, dtype=np.float32))
        return {
            "seed": cell.seed, "combo": combo, "thresholds": [float(t) for t in thresholds],
            "f1_micro_fixed": fixed["f1_micro"], "f1_macro_fixed": fixed["f1_macro"], "mcc_fixed": fixed["mcc_macro"],
            "per_label_fixed": fixed["per_label"],
            "f1_micro_tuned": tuned["f1_micro"], "f1_macro_tuned": tuned["f1_macro"],
            "pr_auc_tuned": tuned["pr_auc_macro"], "mcc_tuned": tuned["mcc_macro"],
            "roc_auc_tuned": tuned["roc_auc_macro"], "balanced_acc_tuned": tuned["balanced_acc_macro"],
            "hamming_loss_tuned": tuned["hamming_loss"], "subset_acc_tuned": tuned["subset_accuracy"],
            "per_label_tuned": tuned["per_label"], "rescoring": meta, "source_predictions": cell.source,
        }
    thresholds, tuned, fixed, support = [], {}, {}, {}
    for i, label in enumerate(LABELS):
        mv, mt = a["m_val"][:, i], a["m_test"][:, i]
        yv, pv = a["y_val"][mv, i], a["p_val"][mv, i]
        yt, pt = a["y_test"][mt, i], a["p_test"][mt, i]
        t = float(tune(rule, yv.reshape(-1, 1), pv.reshape(-1, 1), ctx.steps, fixed_value)[0])
        thresholds.append(t)
        tuned[label] = label_metrics(yt, pt, t, label)
        fixed[label] = label_metrics(yt, pt, 0.5, label)
        support[label] = {"val": int(mv.sum()), "test": int(mt.sum()),
                          "val_positives": int(yv.sum()), "test_positives": int(yt.sum())}
    t_sum, f_sum = compose(tuned), compose(fixed)
    return {
        "seed": cell.seed, "combo": combo, "thresholds": thresholds,
        "f1_micro_fixed": f_sum["f1_micro"], "f1_macro_fixed": f_sum["f1_macro"], "mcc_fixed": f_sum["mcc_macro"],
        "per_label_fixed": fixed,
        "f1_micro_tuned": t_sum["f1_micro"], "f1_macro_tuned": t_sum["f1_macro"], "pr_auc_tuned": t_sum["pr_auc_macro"],
        "mcc_tuned": t_sum["mcc_macro"], "roc_auc_tuned": t_sum["roc_auc_macro"],
        "balanced_acc_tuned": t_sum["balanced_acc_macro"], "hamming_loss_tuned": t_sum["hamming_loss"],
        "subset_acc_tuned": None, "per_label_tuned": tuned, "eligible_support": support,
        "rescoring": meta, "source_predictions": cell.source,
    }


def compare_records(mine: list[dict[str, Any]], theirs: dict[tuple[str, int, int], dict[str, Any]],
                    tolerance: float) -> dict[str, Any]:
    """Largest absolute difference of the per-label and macro tuned MCC and of the thresholds."""
    worst_mcc, worst_macro, worst_t, compared, missing = 0.0, 0.0, None, 0, []
    for record in mine:
        key = record_key(record)
        other = theirs.get(key) if key is not None else None
        if other is None:
            missing.append(list(key) if key else None)
            continue
        compared += 1
        worst_macro = max(worst_macro, abs(float(record["mcc_tuned"]) - float(other["mcc_tuned"])))
        per_label = other.get("per_label_tuned")
        if isinstance(per_label, str):
            per_label = ast.literal_eval(per_label)
        for label in LABELS:
            a = record["per_label_tuned"][label]["mcc"]
            b = per_label.get(label, {}).get("mcc") if per_label else a
            if a is None and b is None:
                continue
            if a is None or b is None:
                worst_mcc = float("inf")
                continue
            worst_mcc = max(worst_mcc, abs(float(a) - float(b)))
        if other.get("thresholds") is not None:
            gap = float(np.max(np.abs(np.asarray(record["thresholds"]) - np.asarray(other["thresholds"]))))
            worst_t = gap if worst_t is None else max(worst_t, gap)
    return {"compared": compared, "missing_source": missing, "max_abs_diff_label_mcc": worst_mcc,
            "max_abs_diff_macro_mcc": worst_macro, "max_abs_diff_threshold": worst_t,
            "ok": compared > 0 and worst_mcc <= tolerance and worst_macro <= tolerance}


def source_variant(inp: Input, variants: dict[str, dict[str, Any]]) -> str | None:
    """The variant that repeats the source protocol of the input."""
    for vname in inp.variants:
        v = variants[vname]
        if (str(v["labels"]) == inp.training_labels and str(v["threshold"]) == "max_f1"
                and not bool(v.get("eligible_only", False))):
            return vname
    return None


# bootstrap

def resample_weights(n: int, n_boot: int, seed: int, cell_key: tuple[str, int, int]) -> np.ndarray:
    """(n, n_boot) bootstrap multiplicities, seeded by the seed and the cell only."""
    strategy, cv_seed, fold = cell_key
    rng = np.random.default_rng([int(seed), zlib.crc32(strategy.encode()), int(cv_seed), int(fold)])
    draws = rng.integers(0, n, size=(n_boot, n))
    counts = np.bincount((draws + (np.arange(n_boot) * n)[:, None]).ravel(), minlength=n_boot * n)
    return counts.reshape(n_boot, n).T.astype(np.float64)


def bootstrap_cell(cell: Cell, variant: dict[str, Any], record: dict[str, Any], ctx: Context,
                   bcfg: dict[str, Any], weight_cache: dict[tuple[Any, ...], np.ndarray]) -> dict[str, Any]:
    a = variant_arrays(cell, variant, ctx)
    rule = str(variant["threshold"])
    grid = threshold_grid(ctx.steps)
    n_boot = int(bcfg["n_boot"])
    shift_steps = round(float(bcfg["max_shift"]) * (len(grid) - 1))
    cache_key = (cell.key, tuple(cell.idx["val"].tolist()))
    if cache_key not in weight_cache:
        weight_cache[cache_key] = resample_weights(len(cell.idx["val"]), n_boot, int(bcfg["seed"]), cell.key)
    W = weight_cache[cache_key]
    mcc = np.full((n_boot, len(LABELS)), np.nan)
    thr_idx = np.full((n_boot, len(LABELS)), -1, dtype=np.int16)
    per_label: dict[str, Any] = {}
    checks = {"threshold_mismatch": 0, "max_abs_diff_mcc": 0.0}
    for i, label in enumerate(LABELS):
        mv, mt = a["m_val"][:, i], a["m_test"][:, i]
        yv, pv, wv = a["y_val"][mv, i], a["p_val"][mv, i], W[mv]
        yt, pt = a["y_test"][mt, i], a["p_test"][mt, i]
        test_counts = grid_counts(yt, pt, grid, np.ones((len(yt), 1)))
        test_mcc = mcc_from_counts(*test_counts)[:, 0]
        half = grid_counts(yt, pt, np.array([0.5]), np.ones((len(yt), 1)))
        mcc_half = float(mcc_from_counts(*half)[0, 0])
        if int(yt.sum()) == 0:
            # compute_full_metrics leaves the MCC undefined without a test positive
            test_mcc[:], mcc_half = np.nan, float("nan")
        choice, no_pos = grid_choice(yv, pv, grid, wv, rule)
        boot_mcc = np.where(no_pos, mcc_half, test_mcc[choice])
        orig_choice, orig_no_pos = grid_choice(yv, pv, grid, np.ones((len(yv), 1)), rule)
        orig_t = 0.5 if orig_no_pos[0] else float(np.float32(grid[int(orig_choice[0])]))
        orig_mcc = mcc_half if orig_no_pos[0] else float(test_mcc[int(orig_choice[0])])
        # the vectorised tuning must give the thresholds and test MCC of the record
        if abs(orig_t - float(record["thresholds"][i])) > 1e-6:
            checks["threshold_mismatch"] += 1
        rec_mcc = record["per_label_tuned"][label]["mcc"]
        if rec_mcc is not None and math.isfinite(orig_mcc):
            checks["max_abs_diff_mcc"] = max(checks["max_abs_diff_mcc"], abs(orig_mcc - float(rec_mcc)))
        # compare thresholds through their grid positions
        half_pos = 0.5 * (len(grid) - 1)
        pos_0 = half_pos if orig_no_pos[0] else float(orig_choice[0])
        pos_b = np.where(no_pos, half_pos, choice.astype(np.float64))
        moved = np.abs(pos_b - pos_0) > shift_steps + 1e-6
        t_b = np.where(no_pos, 0.5, grid[choice])
        mcc[:, i], thr_idx[:, i] = boot_mcc, np.where(no_pos, -1, choice)
        q = [float(v) for v in np.nanpercentile(boot_mcc, bcfg["percentiles"])] if np.isfinite(boot_mcc).any() \
            else [None] * len(bcfg["percentiles"])
        tq = [float(v) for v in np.percentile(t_b, bcfg["percentiles"])]
        per_label[label] = {
            "threshold_original": orig_t, "mcc_original": finite(orig_mcc),
            "mcc_percentiles": dict(zip([str(p) for p in bcfg["percentiles"]], q)),
            "mcc_sd": finite(float(np.nanstd(boot_mcc, ddof=1))) if np.isfinite(boot_mcc).sum() > 1 else None,
            "threshold_percentiles": dict(zip([str(p) for p in bcfg["percentiles"]], tq)),
            "share_threshold_moved": float(moved.mean()),
            "n_boot_without_val_positive": int(no_pos.sum()),
            "val_samples": int(mv.sum()), "val_positives": int(yv.sum()),
            "test_samples": int(mt.sum()), "test_positives": int(yt.sum()),
        }
    macro = np.nanmean(mcc, axis=1) if np.isfinite(mcc).any() else np.full(n_boot, np.nan)
    per_label["macro"] = {
        "mcc_original": finite(float(record["mcc_tuned"])),
        "mcc_percentiles": dict(zip([str(p) for p in bcfg["percentiles"]],
                                    [float(v) for v in np.nanpercentile(macro, bcfg["percentiles"])])),
        "mcc_sd": finite(float(np.nanstd(macro, ddof=1))),
    }
    return {"cell": list(cell.key), "per_label": per_label, "checks": checks, "_mcc": mcc, "_thr_idx": thr_idx,
            "_macro": macro}


def spread_summary(cells: list[dict[str, Any]], bcfg: dict[str, Any]) -> dict[str, Any]:
    """Per label: test MCC spread across cells against the spread due to the threshold."""
    lo, hi = min(bcfg["percentiles"]), max(bcfg["percentiles"])
    out: dict[str, Any] = {"n_cells": len(cells)}
    for k, label in enumerate([*LABELS, "macro"]):
        orig = np.array([c["per_label"][label]["mcc_original"] if c["per_label"][label]["mcc_original"] is not None
                         else np.nan for c in cells], dtype=np.float64)
        boot = np.stack([c["_macro"] if label == "macro" else c["_mcc"][:, k] for c in cells])  # (cells, B)
        by_seed: dict[int, list[float]] = defaultdict(list)
        for c, v in zip(cells, orig):
            if np.isfinite(v):
                by_seed[int(c["cell"][1])].append(float(v))
        seed_avg = [statistics.mean(v) for _, v in sorted(by_seed.items())]
        within_sd = np.array([np.nanstd(b, ddof=1) for b in boot])
        within_width = np.array([np.nanpercentile(b, hi) - np.nanpercentile(b, lo) for b in boot])
        grand = np.nanmean(boot, axis=0)
        fold_sd = float(np.nanstd(orig, ddof=1)) if np.isfinite(orig).sum() > 1 else float("nan")
        # threshold spread of the per-seed fold mean, same unit as the std over seeds
        rows_of_seed: dict[int, list[int]] = defaultdict(list)
        for j, c in enumerate(cells):
            rows_of_seed[int(c["cell"][1])].append(j)
        seed_boot_sd = [float(np.nanstd(np.nanmean(boot[rows], axis=0), ddof=1))
                        for _, rows in sorted(rows_of_seed.items())]
        seed_sd = statistics.pstdev(seed_avg) if len(seed_avg) > 1 else None
        seed_thr_sd = float(np.nanmean(seed_boot_sd)) if seed_boot_sd else float("nan")
        entry = {
            "mean_original": finite(float(np.nanmean(orig))),
            "across_cells_sd": finite(fold_sd),
            "across_cells_width": finite(float(np.nanpercentile(orig, hi) - np.nanpercentile(orig, lo))),
            "across_seeds_sd_of_fold_means": finite(float(seed_sd)) if seed_sd is not None else None,
            "seed_fold_mean_threshold_sd": finite(seed_thr_sd),
            "ratio_seed_fold_mean_threshold_sd_to_across_seeds_sd": (finite(seed_thr_sd / seed_sd)
                                                                     if seed_sd else None),
            "within_cell_sd_mean": finite(float(np.nanmean(within_sd))),
            "within_cell_width_mean": finite(float(np.nanmean(within_width))),
            "within_cell_width_max": finite(float(np.nanmax(within_width))),
            "ratio_within_sd_to_across_sd": finite(float(np.nanmean(within_sd)) / fold_sd) if fold_sd > 0 else None,
            "mean_of_cell_medians": finite(float(np.nanmean(np.nanmedian(boot, axis=1)))),
            "grand_mean_percentiles": dict(zip([str(p) for p in bcfg["percentiles"]],
                                               [float(v) for v in np.nanpercentile(grand, bcfg["percentiles"])])),
            "grand_mean_sd": finite(float(np.nanstd(grand, ddof=1))),
            "grand_mean_width": finite(float(np.nanpercentile(grand, hi) - np.nanpercentile(grand, lo))),
        }
        if label != "macro":
            entry["share_threshold_moved_mean"] = float(np.mean([c["per_label"][label]["share_threshold_moved"]
                                                                 for c in cells]))
            entry["val_positives_median"] = float(np.median([c["per_label"][label]["val_positives"] for c in cells]))
        out[label] = entry
    return out


# summaries

def group_cells(path: Path, name: str, strategy: str) -> dict[tuple[str, int, int], dict[str, float]]:
    groups, _ = load_groups([f"{name}={path}"], strategy)
    if len(groups) > 1:
        raise SystemExit(f"{path}: {len(groups)} groups for strategy {strategy}")
    return next(iter(groups.values())) if groups else {}


def paired(a: dict[tuple[str, int, int], dict[str, float]], b: dict[tuple[str, int, int], dict[str, float]],
           metric: str, ratio: float) -> dict[str, Any] | None:
    shared = sorted(set(a) & set(b))
    pairs = [(a[c][metric], b[c][metric]) for c in shared if metric in a[c] and metric in b[c]]
    if len(pairs) < 2:
        return None
    diffs = np.array([x - y for x, y in pairs])
    t, p, half = corrected_ttest(diffs, ratio)
    return {"n": len(pairs), "mean_diff": float(np.mean(diffs)), "min_diff": float(np.min(diffs)),
            "max_diff": float(np.max(diffs)), "wins": int((diffs > 0).sum()), "losses": int((diffs < 0).sum()),
            "ci95_low": finite(float(np.mean(diffs)) - half), "ci95_high": finite(float(np.mean(diffs)) + half),
            "corrected_t": finite(t), "corrected_p": finite(p)}


def rank_changes(ref: dict[str, float], var: dict[str, float]) -> dict[str, Any]:
    names = sorted(set(ref) & set(var), key=lambda n: -ref[n])
    order_var = sorted(names, key=lambda n: -var[n])
    swaps = []
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            if ref[x] > ref[y] and var[x] < var[y]:
                swaps.append({"higher_in_reference": x, "lower_in_reference": y,
                              "reference_gap": ref[x] - ref[y], "variant_gap": var[x] - var[y]})
    tau = stats.kendalltau([ref[n] for n in names], [var[n] for n in names]) if len(names) > 1 else None
    return {"n_models": len(names), "order_reference": names, "order_variant": order_var,
            "rank_moves": {n: names.index(n) - order_var.index(n) for n in names if names.index(n) != order_var.index(n)},
            "kendall_tau": finite(float(tau.statistic)) if tau is not None else None,
            "swapped_pairs": swaps}


# figures

class Figures:
    def __init__(self, fcfg: dict[str, Any]) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt, self.cfg = plt, fcfg
        self.dir, self.prefix = resolve(fcfg["dir"]), str(fcfg["prefix"])
        self.written: list[str] = []
        self.dir.mkdir(parents=True, exist_ok=True)
        ink, ink2 = str(fcfg["ink"]), str(fcfg["ink_secondary"])
        plt.rcParams.update({
            "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
            "axes.edgecolor": ink2, "axes.labelcolor": ink, "text.color": ink,
            "xtick.color": ink2, "ytick.color": ink2, "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.color": str(fcfg["grid"]), "grid.linewidth": 0.6, "axes.axisbelow": True,
            "legend.frameon": False, "figure.dpi": 100, "savefig.bbox": "tight",
        })

    def save(self, fig: Any, name: str) -> None:
        for ext in ("pdf", "png"):
            path = self.dir / f"{self.prefix}{name}.{ext}"
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            fig.savefig(tmp, format=ext, dpi=200)
            tmp.replace(path)
            self.written.append(rel(path))
        self.plt.close(fig)

    def label(self, name: str) -> str:
        return str(self.cfg.get("model_labels", {}).get(name, name))


def figure_variants(figs: Figures, table: list[dict[str, Any]], ref_variant: str, ref_labels: str) -> dict[str, Any]:
    fcfg = figs.cfg
    strategy = str(fcfg["strategy"])
    series = list(fcfg["series"])
    palette = list(fcfg["palette"])
    if len(series) > len(palette):
        raise SystemExit("more series than palette colours")
    rows = {(r["training_labels"], r["variant"], r["base"]): r for r in table if r["strategy"] == strategy}
    bases = sorted({r["base"] for r in table if r["strategy"] == strategy and r["training_labels"] == ref_labels
                    and r["variant"] == ref_variant},
                   key=lambda b: rows[(ref_labels, ref_variant, b)]["macro_mcc_mean"])
    metrics = list(fcfg["metrics"])
    plt = figs.plt
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 0.32 * len(bases) + 1.6), sharey=True)
    axes = np.atleast_1d(axes)
    ys = np.arange(len(bases))
    for ax, metric in zip(axes, metrics):
        key = f"{metric['key']}_mean"
        for yi, b in zip(ys, bases):
            vals = [rows[(s["training_labels"], s["variant"], b)][key] for s in series
                    if (s["training_labels"], s["variant"], b) in rows]
            if vals:
                ax.plot([min(vals), max(vals)], [yi, yi], color=str(fcfg["grid"]), linewidth=2.4, zorder=1,
                        solid_capstyle="round")
        for k, s in enumerate(series):
            pts = [(rows[(s["training_labels"], s["variant"], b)][key], yi) for yi, b in zip(ys, bases)
                   if (s["training_labels"], s["variant"], b) in rows]
            if pts:
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=palette[k], marker=str(s["marker"]),
                           s=22, zorder=3, label=str(s["label"]), linewidths=0.4, edgecolors="white")
        ax.axvline(0, color=str(fcfg["ink_secondary"]), linewidth=0.8)
        ax.set_title(str(metric["title"]))
        ax.set_xlabel(str(fcfg["xlabel"]))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, [figs.label(b) for b in bases])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(str(fcfg["title"]))
    fig.tight_layout()
    # legend below the axes, kept in the file by bbox_inches tight
    fig.legend(handles, labels, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 0.0))
    figs.save(fig, str(fcfg["variants_name"]))
    return {"bases": bases, "series": [f"{s['training_labels']}/{s['variant']}" for s in series]}


def figure_bootstrap(figs: Figures, boot_summary: dict[str, Any], inputs: dict[str, Input]) -> dict[str, Any]:
    fcfg = figs.cfg
    strategy, label, variant = str(fcfg["strategy"]), str(fcfg["bootstrap_label"]), str(fcfg["bootstrap_variant"])
    entries = [(name, s[variant][strategy][label]) for name, s in boot_summary.items()
               if variant in s and strategy in s[variant] and label in s[variant][strategy]]
    if not entries:
        return {"skipped": "no bootstrap summary for the configured label, variant and strategy"}
    entries.sort(key=lambda e: e[1]["mean_original"] if e[1]["mean_original"] is not None else -1)
    names = [figs.label(inputs[n].base) for n, _ in entries]
    palette = list(fcfg["palette"])
    plt = figs.plt
    # each panel compares two standard deviations of the same quantity
    panels = list(fcfg["bootstrap_panels"])
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.0 + 2.6 * len(panels), 0.32 * len(entries) + 1.8),
                             sharey=True, gridspec_kw={"width_ratios": [1.0] * len(panels) + [0.8]})
    ys = np.arange(len(entries))
    handles, labels, k = [], [], 0
    for ax, panel in zip(axes, panels):
        series = list(panel["series"])
        for yi, (_, e) in zip(ys, entries):
            vals = [e[s["key"]] for s in series if e.get(s["key"]) is not None]
            if vals:
                ax.plot([min(vals), max(vals)], [yi, yi], color=str(fcfg["grid"]), linewidth=2.4, zorder=1)
        for s in series:
            pts = [(e[s["key"]], yi) for yi, (_, e) in zip(ys, entries) if e.get(s["key"]) is not None]
            h = ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=palette[k], s=22, zorder=3,
                           marker=str(s["marker"]), linewidths=0.4, edgecolors="white")
            handles.append(h)
            labels.append(str(s["label"]))
            k += 1
        ax.set_title(str(panel["title"]))
        ax.set_xlabel(str(fcfg["bootstrap_xlabel"]))
        ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, names)
    ax2 = axes[-1]
    shares = [e[1]["share_threshold_moved_mean"] for e in entries]
    ax2.barh(ys, shares, color=palette[k], height=0.6)
    ax2.set_xlim(0, max(0.5, math.ceil(max(shares) * 10) / 10))
    ax2.set_title(str(fcfg["bootstrap_share_title"]))
    ax2.set_xlabel(str(fcfg["bootstrap_share_xlabel"]))
    ax2.grid(axis="y", visible=False)
    fig.suptitle(str(fcfg["bootstrap_title"]))
    fig.tight_layout()
    fig.legend(handles, labels, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 0.0))
    figs.save(fig, str(fcfg["bootstrap_name"]))
    return {"inputs": [n for n, _ in entries], "label": label, "variant": variant, "strategy": strategy}


# main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args, extra = parser.parse_known_args()
    cfg = load_config(resolve(args.config), parse_cli_overrides(extra))
    rcfg = cfg["rescoring"]
    started = time.perf_counter()
    out_root = resolve(rcfg["out_dir"])
    steps = int(rcfg.get("threshold_steps", 101))
    strategies = rcfg.get("strategies")
    variants: dict[str, dict[str, Any]] = {str(k): dict(v) for k, v in rcfg["variants"].items()}
    for vname, v in variants.items():
        if str(v["threshold"]) not in RULES:
            raise SystemExit(f"variant {vname}: threshold must be one of {RULES}")
        if str(v["labels"]) not in rcfg["label_sets"]:
            raise SystemExit(f"variant {vname}: unknown label set {v['labels']}")
    ref_variant = str(rcfg["reference_variant"])
    ref_labels = str(rcfg["reference_training_labels"])
    if ref_labels not in rcfg["label_sets"]:
        raise SystemExit(f"reference_training_labels {ref_labels} is not a label set")
    tolerance = float(rcfg.get("reproduction_tolerance", 1e-9))

    # label sets and eligibility
    label_sets, label_info = {}, {}
    for lname, lpath in rcfg["label_sets"].items():
        entries = json.loads(resolve(lpath).read_text())
        label_sets[str(lname)] = label_matrix_from_json(entries, np.arange(len(entries)), len(LABELS))
        label_info[str(lname)] = {"path": str(lpath), "samples": len(entries),
                                  "positives": dict(zip(SHORTS, label_sets[str(lname)].sum(axis=0).tolist()))}
    sizes = {k: len(v) for k, v in label_sets.items()}
    if len(set(sizes.values())) != 1:
        raise SystemExit(f"label sets of different sizes: {sizes}")
    n_samples = next(iter(sizes.values()))
    ecfg = rcfg["eligibility"]
    with resolve(ecfg["sample_meta"]).open() as fh:
        level_of = {int(r["json_index"]): r[str(ecfg.get("level_column", "level"))] for r in csv.DictReader(fh)}
    levels = {str(k): str(v) for k, v in ecfg["label_levels"].items()}
    if set(levels) != set(LABELS):
        raise SystemExit(f"eligibility.label_levels must cover {LABELS}")
    eligible = np.zeros((n_samples, len(LABELS)), dtype=bool)
    for j in range(n_samples):
        if j in level_of:
            eligible[j] = [level_of[j] == levels[label] for label in LABELS]

    folds_file = resolve(rcfg["folds"])
    folds = {(f.strategy, f.seed, f.fold): f for f in iter_folds(load_folds(folds_file))}
    ratio = ratio_from_folds(folds_file)
    ctx = Context(label_sets=label_sets, eligible=eligible, steps=steps, folds=folds)

    # label differences between label sets (descriptive)
    label_diff = {}
    names = list(label_sets)
    for i, a_name in enumerate(names):
        for b_name in names[i + 1:]:
            diff = label_sets[a_name] != label_sets[b_name]
            rows = np.where(diff.any(axis=1))[0]
            label_diff[f"{a_name}_vs_{b_name}"] = {
                "samples_differing": len(rows),
                "json_index_min": int(rows.min()) if len(rows) else None,
                "json_index_max": int(rows.max()) if len(rows) else None,
                "per_label_differing": dict(zip(SHORTS, diff.sum(axis=0).tolist())),
                "json_indices": rows.tolist(),
            }

    # records
    bcfg = dict(rcfg.get("bootstrap", {}))
    boot_on = bool(bcfg.get("enabled", False))
    summary: dict[str, Any] = {
        "config": rel(resolve(args.config)), "label_sets": label_info, "label_differences": label_diff,
        "variants": variants, "reference_variant": ref_variant, "reference_training_labels": ref_labels,
        "threshold_steps": steps,
        "test_train_ratio": ratio, "inputs": [], "variant_checks": {}, "bootstrap": {},
    }
    written: dict[tuple[str, str, str], Path] = {}
    inputs: dict[str, Input] = {}
    weight_cache: dict[tuple[Any, ...], np.ndarray] = {}
    for spec in rcfg["inputs"]:
        inp = load_input(spec, list(variants), strategies)
        if inp.name in inputs:
            raise SystemExit(f"duplicate input name {inp.name}")
        inputs[inp.name] = inp
        if inp.training_labels not in label_sets:
            raise SystemExit(f"input {inp.name}: unknown training_labels {inp.training_labels}")
        info: dict[str, Any] = {"name": inp.name, "base": inp.base, "family": inp.family,
                                "training_labels": inp.training_labels, "variants": inp.variants,
                                "files_found": len(inp.files_found), "files_missing": inp.files_missing,
                                "cells": dict(sorted((s, sum(c.strategy == s for c in inp.cells))
                                                     for s in {c.strategy for c in inp.cells})),
                                "models": sorted({c.model for c in inp.cells})}
        if not inp.cells:
            print(f"[rescore] skip {inp.name}: no predictions found ({inp.files_missing[:3]})", flush=True)
            info["skipped"] = "no predictions"
            summary["inputs"].append(info)
            continue
        # the rows must carry the training labels, and the parts must be the shared folds
        mism = sum(int((label_sets[inp.training_labels][c.idx[p]] != c.y[p]).any(axis=1).sum())
                   for c in inp.cells for p in ("val", "test"))
        info["rows_with_labels_differing_from_training_labels"] = mism
        if mism:
            raise SystemExit(f"input {inp.name}: {mism} prediction rows whose y differs from label set "
                             f"{inp.training_labels}")
        if bool(rcfg.get("check_folds", True)):
            bad = [c.key for c in inp.cells if c.key not in folds
                   or set(c.idx["val"].tolist()) != set(folds[c.key].val.tolist())
                   or set(c.idx["test"].tolist()) != set(folds[c.key].test.tolist())]
            info["cells_not_matching_folds"] = [list(k) for k in bad]
            if bad:
                raise SystemExit(f"input {inp.name}: cells {bad[:3]} do not match the folds file")

        src_variant = source_variant(inp, variants)
        for vname in inp.variants:
            variant = variants[vname]
            records = [score_cell(inp, c, vname, variant, ctx) for c in inp.cells]
            path = out_root / f"train_{inp.training_labels}" / vname / inp.name / "records.json"
            write_json(path, records)
            written[(inp.training_labels, vname, inp.name)] = path
            fe = [r["per_label_tuned"]["is_feature_envy"]["mcc"] for r in records]
            fe = [v for v in fe if v is not None]
            print(f"[rescore] {inp.name} {vname}: {len(records)} records, mean macro MCC "
                  f"{np.mean([r['mcc_tuned'] for r in records]):.3f}, FE {np.mean(fe):.3f}", flush=True)
            if vname == src_variant:
                sources = {c.key: c.source_record for c in inp.cells if c.source_record is not None}
                check = compare_records(records, sources, tolerance)
                info["reproduction_check"] = {"variant": vname, **check}
                print(f"[rescore] {inp.name}: reproduction of the source records on {check['compared']} cells, "
                      f"max |diff| label MCC {check['max_abs_diff_label_mcc']:.2e}, "
                      f"macro MCC {check['max_abs_diff_macro_mcc']:.2e}, "
                      f"threshold {check['max_abs_diff_threshold']}", flush=True)
                if sources and not check["ok"]:
                    raise SystemExit(f"input {inp.name}: variant {vname} does not reproduce the source records")
            vcheck = rcfg.get("variant_checks", {}).get(vname)
            if vcheck is not None:
                other = resolve(vcheck["root"]) / inp.name / "records.json"
                if other.exists():
                    theirs = {record_key(r): r for r in json.loads(other.read_text())}
                    check = compare_records(records, theirs, tolerance)
                    summary["variant_checks"].setdefault(vname, {})[inp.name] = {"against": rel(other), **check}
                    if not check["ok"]:
                        raise SystemExit(f"input {inp.name}: variant {vname} differs from {rel(other)}")

            if (boot_on and vname in bcfg.get("variants", []) and inp.training_labels in bcfg.get("training_labels", [])
                    and str(variant["threshold"]) != "fixed"):
                cells_out = [bootstrap_cell(c, variant, r, ctx, bcfg, weight_cache) for c, r in zip(inp.cells, records)]
                checks = {"threshold_mismatch": sum(c["checks"]["threshold_mismatch"] for c in cells_out),
                          "max_abs_diff_mcc": max(c["checks"]["max_abs_diff_mcc"] for c in cells_out)}
                if checks["threshold_mismatch"] or checks["max_abs_diff_mcc"] > 1e-9:
                    raise SystemExit(f"input {inp.name} {vname}: the vectorised tuning does not reproduce the "
                                     f"records ({checks})")
                by_strategy = {}
                for strategy in sorted({c.strategy for c in inp.cells}):
                    sel = [co for c, co in zip(inp.cells, cells_out) if c.strategy == strategy]
                    by_strategy[strategy] = spread_summary(sel, bcfg)
                summary["bootstrap"].setdefault(inp.name, {})[vname] = by_strategy
                bdir = out_root / "bootstrap" / vname
                write_json(bdir / f"{inp.name}.json", {
                    "input": inp.name, "variant": vname, "n_boot": int(bcfg["n_boot"]), "seed": int(bcfg["seed"]),
                    "max_shift": float(bcfg["max_shift"]), "checks": checks, "summary": by_strategy,
                    "cells": [{k: v for k, v in c.items() if not k.startswith("_")} for c in cells_out]})
                tmp = bdir / f"{inp.name}.{os.getpid()}.tmp.npz"
                np.savez_compressed(tmp, cells=np.array([c["cell"] for c in cells_out], dtype=object).astype(str),
                                    labels=np.array(LABELS), test_mcc=np.stack([c["_mcc"] for c in cells_out]).astype(np.float32),
                                    threshold_index=np.stack([c["_thr_idx"] for c in cells_out]),
                                    macro_mcc=np.stack([c["_macro"] for c in cells_out]).astype(np.float32))
                tmp.replace(bdir / f"{inp.name}.npz")
                for strategy, spread in by_strategy.items():
                    fe = {k: ("nan" if v is None else f"{v:.3f}") for k, v in spread[LABELS[0]].items()
                          if not isinstance(v, dict)}
                    print(f"[rescore] {inp.name} {vname} {strategy} bootstrap: {SHORTS[0]} across-cell sd "
                          f"{fe['across_cells_sd']}, within-cell sd {fe['within_cell_sd_mean']}, threshold moved > "
                          f"{bcfg['max_shift']} in a share {fe['share_threshold_moved_mean']} of resamples; "
                          f"fold-mean sd over seeds {fe['across_seeds_sd_of_fold_means']}, due to the threshold "
                          f"{fe['seed_fold_mean_threshold_sd']}", flush=True)
        summary["inputs"].append(info)

    # tables
    scfg = rcfg["summary"]
    table: list[dict[str, Any]] = []
    cells_of: dict[tuple[str, str, str, str], dict[tuple[str, int, int], dict[str, float]]] = {}
    for (tl, vname, name), path in sorted(written.items()):
        for strategy in scfg["strategies"]:
            cells = group_cells(path, name, strategy)
            if not cells:
                continue
            cells_of[(tl, vname, name, strategy)] = cells
            row: dict[str, Any] = {"training_labels": tl, "variant": vname, "input": name,
                                   "base": inputs[name].base, "family": inputs[name].family, "strategy": strategy,
                                   "cells": len(cells), "seeds": len({s for _, s, _ in cells})}
            for metric in TABLE_METRICS:
                means = seed_means(cells, metric)
                row[f"{metric}_mean"] = statistics.mean(means) if means else None
                row[f"{metric}_std"] = statistics.pstdev(means) if len(means) > 1 else None
            table.append(row)
    summary["table"] = table

    # deltas of every setting against the reference variant of the base model
    deltas = []
    for (tl, vname, name, strategy), cells in cells_of.items():
        ref_key = (ref_labels, ref_variant, inputs[name].base, strategy)
        if (tl, vname, name, strategy) == ref_key or ref_key not in cells_of:
            continue
        same_labels = str(variants[vname]["labels"]) == str(variants[ref_variant]["labels"]) and \
            bool(variants[vname].get("eligible_only", False)) == bool(variants[ref_variant].get("eligible_only", False))
        entry = {"training_labels": tl, "variant": vname, "input": name, "base": inputs[name].base,
                 "strategy": strategy, "reference": f"{ref_labels}/{ref_variant}/{inputs[name].base}",
                 "same_test_labels_and_samples": same_labels}
        for metric in ("macro_mcc", "feature_envy_mcc"):
            entry[metric] = paired(cells, cells_of[ref_key], metric, ratio)
        deltas.append(entry)
    summary["deltas_vs_reference"] = deltas

    # rankings of the models trained with the reference labels, per variant
    min_cells = int(scfg.get("min_cells", 25))
    rankings: dict[str, Any] = {}
    for strategy in scfg["strategies"]:
        complete = {(r["variant"], r["input"]): r for r in table if r["strategy"] == strategy
                    and r["training_labels"] == ref_labels and r["cells"] >= min_cells}
        excluded = sorted({r["input"] for r in table if r["strategy"] == strategy and r["training_labels"] == ref_labels
                           and r["cells"] < min_cells})
        for metric in scfg["rank_metrics"]:
            ref = {n: r[f"{metric}_mean"] for (v, n), r in complete.items() if v == ref_variant}
            if len(ref) < 2:
                continue
            for vname in variants:
                if vname == ref_variant:
                    continue
                var = {n: r[f"{metric}_mean"] for (v, n), r in complete.items() if v == vname}
                if len(var) >= 2:
                    rankings.setdefault(strategy, {}).setdefault(metric, {})[vname] = rank_changes(ref, var)
        if strategy in rankings:
            rankings[strategy]["excluded_incomplete_inputs"] = excluded
    summary["rankings"] = rankings

    # label settings side by side, for the bases that have a retrained input
    lccfg = scfg["label_comparison"]
    lc_settings = [(str(s["training_labels"]), str(s["variant"])) for s in lccfg["settings"]]
    paired_variants = [str(v) for v in lccfg.get("paired_eval_variants", [])]
    retrained = [i for i in inputs.values() if i.training_labels != ref_labels and i.cells]
    retrained_bases = sorted({i.base for i in retrained})
    label_comparison: dict[str, Any] = {
        "settings": [f"train_{tl}/{v}" for tl, v in lc_settings],
        "paired_eval_variants": paired_variants,
        "paired_note": ("retrained_minus_eval_only/<variant>: input trained with other labels minus the model "
                        f"trained with {ref_labels} labels, both scored by the same variant (same test labels "
                        "and samples). corrected_p is unadjusted; corrected_p_holm is Holm-adjusted over the "
                        "retrained bases for one (strategy, variant, metric)."),
        "models": {}}
    for strategy in scfg["strategies"]:
        for base in retrained_bases:
            entry = {}
            for tl, vname in lc_settings:
                name = next((i.name for i in inputs.values() if i.base == base and i.training_labels == tl), None)
                row = next((r for r in table if r["strategy"] == strategy and r["training_labels"] == tl
                            and r["variant"] == vname and r["input"] == name), None)
                if row is not None:
                    entry[f"train_{tl}/{vname}"] = {m: {"mean": row[f"{m}_mean"], "std": row[f"{m}_std"]}
                                                    for m in ("macro_mcc", "feature_envy_mcc")}
            # retrained against evaluation-only relabelling: same test labels and samples
            for eval_variant in paired_variants:
                for inp in (i for i in retrained if i.base == base):
                    if str(variants[eval_variant]["labels"]) != inp.training_labels:
                        continue
                    a = cells_of.get((inp.training_labels, eval_variant, inp.name, strategy))
                    b = cells_of.get((ref_labels, eval_variant, base, strategy))
                    if a and b:
                        entry[f"retrained_minus_eval_only/{eval_variant}"] = {
                            m: paired(a, b, m, ratio) for m in ("macro_mcc", "feature_envy_mcc")}
            if entry:
                label_comparison["models"].setdefault(strategy, {})[base] = entry
        for eval_variant in paired_variants:
            for metric in ("macro_mcc", "feature_envy_mcc"):
                tests = [e[f"retrained_minus_eval_only/{eval_variant}"][metric]
                         for e in label_comparison["models"].get(strategy, {}).values()
                         if e.get(f"retrained_minus_eval_only/{eval_variant}", {}).get(metric) is not None]
                tests = [t for t in tests if t["corrected_p"] is not None]
                for t, adj in zip(tests, holm([t["corrected_p"] for t in tests])):
                    t["corrected_p_holm"] = adj
        for tl, vname in lc_settings:
            for metric in ("macro_mcc", "feature_envy_mcc"):
                vals = {}
                for base in retrained_bases:
                    e = label_comparison["models"].get(strategy, {}).get(base, {}).get(f"train_{tl}/{vname}")
                    if e is not None:
                        vals[base] = e[metric]["mean"]
                if vals:
                    label_comparison.setdefault("order", {}).setdefault(strategy, {}).setdefault(
                        f"train_{tl}/{vname}", {})[metric] = sorted(vals, key=lambda b: -vals[b])
    summary["label_comparison"] = label_comparison

    summary["notes"] = [
        ("Every number is the mean over the folds of a seed, then the mean and population standard deviation "
         "over seeds (scripts/cv/summarize_cv.py); paired differences use the corrected resampled t-test on shared "
         "(strategy, seed, fold) cells, with n_test / n_train of the folds file."),
        ("Variants with labels rule_default are scored against labels that differ from the stored ones only for "
         "the samples listed in label_differences (all in the first part of the normalized JSON); thresholds are "
         "re-tuned on the validation part with the new labels. Deltas with same_test_labels_and_samples false "
         "compare numbers measured against different targets and are descriptive."),
        ("Eligible-only variants score each smell on the samples of its level only (a subset selected by the "
         "sample level, not by the labels); validation and test sizes per label are in eligible_support of each "
         "record."),
        "max_mcc is a declared sensitivity analysis: every reported number of the thesis uses max_f1.",
        ("Threshold bootstrap: the validation part is resampled, the test part and the trained model are fixed, "
         "so the spread reflects the threshold choice only. All inputs share the resamples of a cell (same json "
         "indices); cells use independent resamples. Compare quantities of the same unit only: "
         "within_cell_sd_mean (threshold) against across_cells_sd (one cell, original thresholds; it also holds "
         "the sampling variation of the test part, the training and the fold partition), and "
         "seed_fold_mean_threshold_sd (threshold, mean over the folds of one seed) against "
         "across_seeds_sd_of_fold_means (the +- of the tables). grand_mean_* is the distribution of the mean "
         "over all cells of the strategy when every cell re-tunes on its own resample; it has no counterpart "
         "in the tables."),
        ("For most inputs the mean bootstrap test MCC (grand_mean_percentiles, 50) is below mean_original: "
         "thresholds tuned on a resample, which holds about 63% of the distinct validation samples, are worse "
         "on average than the original one. The naive bootstrap is not consistent for an estimator defined as "
         "the maximiser of a non-smooth criterion (Abrevaya and Huang 2005, for the maximum score estimator), "
         "which a grid threshold maximising F1 or MCC is. When seed_fold_mean_threshold_sd exceeds "
         "across_seeds_sd_of_fold_means, the threshold part alone exceeds the total spread over seeds, which "
         "points to an overstatement by the bootstrap or to too few seeds to estimate that spread. The spreads "
         "are a sensitivity measure, not a confidence interval."),
        ("Rankings only use inputs trained with the reference labels and with at least min_cells cells of the "
         "strategy; GNN cells exist for kfold and group_kfold, tabular cells for kfold only. A swap between two "
         "models whose difference the corrected t-test does not separate is not evidence of an order."),
    ]

    # outputs
    fcfg = rcfg.get("figures")
    if fcfg:
        figs = Figures(fcfg)
        summary["figures"] = {"variants": figure_variants(figs, table, ref_variant, ref_labels),
                              "bootstrap": figure_bootstrap(figs, summary["bootstrap"], inputs),
                              "files": figs.written}
    summary["records"] = {f"train_{tl}/{v}/{n}": rel(p) for (tl, v, n), p in sorted(written.items())}
    summary["elapsed_sec"] = time.perf_counter() - started
    write_json(out_root / "summary.json", summary)
    if table:
        tmp = out_root / f"table.csv.{os.getpid()}.tmp"
        with tmp.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
            writer.writeheader()
            writer.writerows(table)
        tmp.replace(out_root / "table.csv")
    print(f"[rescore] wrote {rel(out_root / 'summary.json')} and {rel(out_root / 'table.csv')} "
          f"in {summary['elapsed_sec']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
