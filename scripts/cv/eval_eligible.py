"""Re-score saved predictions on the samples eligible for each smell.

Eligibility comes from the sample level (MLCQ) or from per-label 0/1 columns (DACOS).
Thresholds are re-tuned on the eligible validation samples. Records keep the cv.build_record schema.

Usage:
    python scripts/cv/eval_eligible.py --config config/experiments/shortcut_baselines.yml
    python scripts/cv/eval_eligible.py --config ... --input name=path/to/predictions.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.constants import LABEL_ORDER  # noqa: E402
from mlcq_graphs.evaluation.cv import read_jsonl, tune_thresholds  # noqa: E402
from mlcq_graphs.evaluation.metrics import compute_full_metrics  # noqa: E402

FOLD_KEYS = ("cv_fold", "training.cv_fold")
STRATEGY_KEYS = ("split_strategy", "training.split_strategy")
# label level that keeps every sample (sanity check against the source records)
ANY_LEVEL = "any"


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def source_combos(pred_path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Combos of the records.json next to the predictions, keyed by (strategy, seed, fold)."""
    rec_path = pred_path.parent / "records.json"
    if not rec_path.exists():
        return {}
    out = {}
    for record in json.loads(rec_path.read_text()):
        combo = dict(record.get("combo", {}))
        fold = next((int(combo[k]) for k in FOLD_KEYS if k in combo), None)
        strategy = next((str(combo[k]) for k in STRATEGY_KEYS if k in combo), None)
        if fold is not None and strategy is not None:
            out[(strategy, int(record["seed"]), fold)] = combo
    return out


def source_macro_mcc(pred_path: Path) -> list[float]:
    rec_path = pred_path.parent / "records.json"
    if not rec_path.exists():
        return []
    return [float(r["mcc_tuned"]) for r in json.loads(rec_path.read_text())]


def label_metrics(y: np.ndarray, p: np.ndarray, threshold: float, name: str) -> dict[str, Any]:
    """Metrics of one label, padded with a zero column to keep the multilabel reading."""
    pred = (p >= threshold).astype(np.int64)
    pad = np.zeros(len(y), dtype=np.int64)
    m = compute_full_metrics(np.column_stack([y.astype(np.int64), pad]), np.column_stack([pred, pad]),
                             np.column_stack([p.astype(np.float64), pad.astype(np.float64)]), [name, "_pad"])
    entry = dict(m["per_label"][name])
    entry["support"] = int(len(y))
    return entry


def compose(per_label: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Macro and pooled summaries over labels evaluated on different samples."""
    values = list(per_label.values())

    def mean_valid(key: str) -> float:
        valid = [v[key] for v in values if v[key] is not None]
        return float(np.mean(valid)) if valid else 0.0

    tp, fp, fn, tn = (sum(v[k] for v in values) for k in ("tp", "fp", "fn", "tn"))
    return {
        "f1_micro": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "f1_macro": float(np.mean([v["f1"] for v in values])),
        "mcc_macro": mean_valid("mcc"),
        "pr_auc_macro": float(np.mean([v["pr_auc"] for v in values])),
        "roc_auc_macro": mean_valid("roc_auc"),
        "balanced_acc_macro": mean_valid("balanced_acc"),
        "hamming_loss": (fp + fn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0,
    }


def evaluate(name: str, pred_path: Path, level_of: dict[int, str], label_levels: dict[str, str],
             steps: int, labels: list[str] | None = None,
             eligible_of: dict[str, dict[int, bool]] | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(pred_path)
    combos = source_combos(pred_path)
    groups: dict[tuple[str, int, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[(str(row["strategy"]), int(row["seed"]), int(row["fold"]))][str(row["part"])].append(row)

    labels = list(labels or LABEL_ORDER)
    records = []
    for (strategy, seed, fold), parts in sorted(groups.items()):
        if "val" not in parts or "test" not in parts:
            raise SystemExit(f"{pred_path}: {strategy} seed {seed} fold {fold} lacks a val or test part")
        arrays = {}
        for part in ("val", "test"):
            idx = np.array([int(r["json_index"]) for r in parts[part]])
            missing = [int(i) for i in idx if int(i) not in level_of]
            if missing:
                raise SystemExit(f"{pred_path}: json_index {missing[:5]} not in the sample metadata")
            arrays[part] = (idx, np.array([r["y"] for r in parts[part]], dtype=np.int64),
                            np.array([r["prob"] for r in parts[part]], dtype=np.float64))
        thresholds, tuned, fixed, support = [], {}, {}, {}
        for i, label in enumerate(labels):
            if eligible_of is not None:
                masks = {part: np.array([eligible_of[label].get(int(j), False) for j in arrays[part][0]])
                         for part in arrays}
            else:
                want = label_levels[label]
                masks = {part: np.array([want == ANY_LEVEL or level_of[int(j)] == want for j in arrays[part][0]])
                         for part in arrays}
            yv, pv = arrays["val"][1][masks["val"], i], arrays["val"][2][masks["val"], i]
            yt, pt = arrays["test"][1][masks["test"], i], arrays["test"][2][masks["test"], i]
            t = float(tune_thresholds(yv.reshape(-1, 1), pv.reshape(-1, 1), steps)[0])
            thresholds.append(t)
            tuned[label] = label_metrics(yt, pt, t, label)
            fixed[label] = label_metrics(yt, pt, 0.5, label)
            support[label] = {"val": int(masks["val"].sum()), "test": int(masks["test"].sum()),
                              "val_positives": int(yv.sum()), "test_positives": int(yt.sum())}
        combo = dict(combos.get((strategy, seed, fold), {"run": name, "model": str(rows[0]["model"]),
                                                          "split_strategy": strategy, "cv_fold": fold}))
        combo["eligible_only"] = True
        t_sum, f_sum = compose(tuned), compose(fixed)
        records.append({
            "seed": seed,
            "combo": combo,
            "thresholds": thresholds,
            "f1_micro_fixed": f_sum["f1_micro"],
            "f1_macro_fixed": f_sum["f1_macro"],
            "mcc_fixed": f_sum["mcc_macro"],
            "per_label_fixed": fixed,
            "f1_micro_tuned": t_sum["f1_micro"],
            "f1_macro_tuned": t_sum["f1_macro"],
            "pr_auc_tuned": t_sum["pr_auc_macro"],
            "mcc_tuned": t_sum["mcc_macro"],
            "roc_auc_tuned": t_sum["roc_auc_macro"],
            "balanced_acc_tuned": t_sum["balanced_acc_macro"],
            "hamming_loss_tuned": t_sum["hamming_loss"],
            "subset_acc_tuned": None,
            "per_label_tuned": tuned,
            "eligible_support": support,
            "source_predictions": str(pred_path.relative_to(PROJECT_ROOT) if pred_path.is_relative_to(PROJECT_ROOT)
                                      else pred_path),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", action="append", default=[], help="NAME=PATH to a predictions.jsonl (repeatable)")
    args, extra = parser.parse_known_args()
    cfg = load_config(resolve(args.config), parse_cli_overrides(extra))
    ecfg = cfg["eligible_eval"]

    labels = list(ecfg.get("label_order") or LABEL_ORDER)
    with resolve(ecfg["sample_meta"]).open() as fh:
        meta_rows = list(csv.DictReader(fh))
    eligible_of = None
    label_levels: dict[str, str] = {}
    if ecfg.get("eligibility_columns"):
        cols = dict(ecfg["eligibility_columns"])
        if set(cols) != set(labels):
            raise SystemExit(f"eligible_eval.eligibility_columns must cover {labels}")
        eligible_of = {lab: {int(r["json_index"]): r[col] == "1" for r in meta_rows} for lab, col in cols.items()}
        level_of = {int(r["json_index"]): "" for r in meta_rows}
    else:
        level_col = str(ecfg.get("level_column", "level"))
        level_of = {int(r["json_index"]): r[level_col] for r in meta_rows}
        label_levels = {str(k): str(v) for k, v in ecfg["label_levels"].items()}
        if set(label_levels) != set(labels):
            raise SystemExit(f"eligible_eval.label_levels must cover {labels}")
    steps = int(ecfg.get("threshold_steps", 101))
    out_root = resolve(ecfg["out_dir"])

    inputs = [(str(i["name"]), resolve(i["predictions"])) for i in ecfg.get("inputs", [])]
    inputs += [(spec.split("=", 1)[0], resolve(spec.split("=", 1)[1])) for spec in args.input]
    for name, pred_path in inputs:
        if any(ch in str(pred_path) for ch in "*?["):
            # concatenate the matching files into the output folder
            parts = sorted(PROJECT_ROOT.glob(str(pred_path.relative_to(PROJECT_ROOT))))
            if not parts:
                print(f"[eligible] skip {name}: no file matches {pred_path}", flush=True)
                continue
            merged = out_root / name / "merged_predictions.jsonl"
            merged.parent.mkdir(parents=True, exist_ok=True)
            with merged.open("w") as fh:
                for part in parts:
                    fh.write(part.read_text())
            pred_path = merged
        if not pred_path.exists():
            print(f"[eligible] skip {name}: {pred_path} not found", flush=True)
            continue
        records = evaluate(name, pred_path, level_of, label_levels, steps, labels, eligible_of)
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        # per-process temporary name, so that concurrent writers never share one
        tmp = out / f"records.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(records, indent=1))
        tmp.replace(out / "records.json")
        before = source_macro_mcc(pred_path)
        after = [r["mcc_tuned"] for r in records]
        per_label = {lab.removeprefix("is_"): np.mean([r["per_label_tuned"][lab]["mcc"] for r in records
                                                       if r["per_label_tuned"][lab]["mcc"] is not None])
                     for lab in labels}
        print(f"[eligible] {name}: {len(records)} records, mean macro MCC "
              f"{np.mean(before) if before else float('nan'):.3f} (all samples) -> {np.mean(after):.3f} (eligible); "
              + " ".join(f"{k}={v:.3f}" for k, v in per_label.items()) + f"; wrote {out / 'records.json'}",
              flush=True)


if __name__ == "__main__":
    main()
