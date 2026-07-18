"""Cross-model analysis of per-sample predictions on the shared folds.

Writes under analysis.out_dir:
- ensembles/<name>/records.json: probability-averaging ensembles
- agreement.json: pairwise Cohen kappa of test errors and oracle recall
- sample_difficulty.csv: share of models that misclassify each sample

Usage:
    python scripts/analysis/analyze_predictions.py --config config/experiments/prediction_analysis.yml
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.constants import LABEL_ORDER  # noqa: E402
from mlcq_graphs.evaluation.cv import build_record, read_jsonl, tune_thresholds  # noqa: E402

Key = tuple[str, int, int]  # strategy, seed, fold


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_model(paths: list[str]) -> dict[Key, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]]:
    """{(strategy, seed, fold): {part: {json_index: (y, prob)}}} from one or more files."""
    out: dict[Key, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]] = defaultdict(lambda: defaultdict(dict))
    for pattern in paths:
        files = sorted(PROJECT_ROOT.glob(pattern)) if any(c in pattern for c in "*?[") else [resolve(pattern)]
        if not files:
            raise SystemExit(f"no prediction file matches {pattern}")
        for file in files:
            for row in read_jsonl(file):
                key = (str(row["strategy"]), int(row["seed"]), int(row["fold"]))
                out[key][row["part"]][int(row["json_index"])] = (
                    np.asarray(row["y"], dtype=np.int64), np.asarray(row["prob"], dtype=np.float64))
    return out


def stack(cell: dict[int, tuple[np.ndarray, np.ndarray]], order: list[int]) -> tuple[np.ndarray, np.ndarray]:
    return (np.stack([cell[i][0] for i in order]), np.stack([cell[i][1] for i in order]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args, extra = parser.parse_known_args()
    cfg = load_config(args.config, parse_cli_overrides(extra))["analysis"]
    out_dir = resolve(cfg["out_dir"])
    steps = int(cfg.get("threshold_steps", 101))
    names = [s.removeprefix("is_") for s in LABEL_ORDER]

    models = {name: load_model(spec["predictions"]) for name, spec in cfg["models"].items()}
    shared_keys = sorted(set.intersection(*(set(m) for m in models.values())))
    if cfg.get("strategies"):
        shared_keys = [k for k in shared_keys if k[0] in cfg["strategies"]]
    if not shared_keys:
        raise SystemExit("models share no (strategy, seed, fold)")
    print(f"[analysis] {len(models)} models, {len(shared_keys)} shared folds")

    # per model and fold: aligned val/test labels, probabilities, tuned predictions
    aligned: dict[str, dict[Key, dict[str, Any]]] = defaultdict(dict)
    for key in shared_keys:
        parts = {}
        for part in ("val", "test"):
            common = set.intersection(*(set(models[m][key][part]) for m in models))
            parts[part] = sorted(common)
            sizes = {m: len(models[m][key][part]) for m in models}
            if any(n != len(common) for n in sizes.values()):
                raise SystemExit(f"{key} {part}: models cover different samples {sizes}")
        for name, model in models.items():
            y_val, p_val = stack(model[key]["val"], parts["val"])
            y_test, p_test = stack(model[key]["test"], parts["test"])
            thr = tune_thresholds(y_val, p_val, steps)
            aligned[name][key] = {"y_val": y_val, "p_val": p_val, "y_test": y_test, "p_test": p_test,
                                  "pred_test": (p_test >= thr).astype(np.int64), "test_index": parts["test"]}

    for ens_name, members in (cfg.get("ensembles") or {}).items():
        records = []
        for key in shared_keys:
            first = aligned[members[0]][key]
            p_val = np.mean([aligned[m][key]["p_val"] for m in members], axis=0)
            p_test = np.mean([aligned[m][key]["p_test"] for m in members], axis=0)
            records.append(build_record(
                combo={"run": ens_name, "members": "+".join(members), "split_strategy": key[0], "cv_fold": key[2]},
                seed=key[1], y_val=first["y_val"], p_val=p_val, y_test=first["y_test"], p_test=p_test,
                threshold_steps=steps))
        dest = out_dir / "ensembles" / ens_name
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "records.json").write_text(json.dumps(records, indent=1))
        macro = np.mean([r["mcc_tuned"] for r in records])
        print(f"[analysis] ensemble {ens_name} ({'+'.join(members)}): mean macro MCC over folds {macro:.3f}")

    agreement: dict[str, Any] = {"pairwise_error_kappa": {}, "oracle_recall": {}, "folds": len(shared_keys)}
    for j, smell in enumerate(names):
        for a, b in itertools.combinations(models, 2):
            kappas = []
            for key in shared_keys:
                y = aligned[a][key]["y_test"][:, j]
                err_a = (aligned[a][key]["pred_test"][:, j] != y).astype(int)
                err_b = (aligned[b][key]["pred_test"][:, j] != y).astype(int)
                if err_a.any() or err_b.any():
                    kappas.append(float(cohen_kappa_score(err_a, err_b)))
            agreement["pairwise_error_kappa"][f"{smell}|{a}|{b}"] = float(np.mean(kappas)) if kappas else None
        found_any, total, per_model = 0, 0, defaultdict(int)
        for key in shared_keys:
            y = aligned[next(iter(models))][key]["y_test"][:, j]
            hits = np.stack([aligned[m][key]["pred_test"][:, j] for m in models]) & y
            found_any += int((hits.any(axis=0) & (y == 1)).sum())
            total += int(y.sum())
            for m in models:
                per_model[m] += int((aligned[m][key]["pred_test"][:, j] & y).sum())
        agreement["oracle_recall"][smell] = {"any_model": found_any / max(total, 1),
                                             **{m: per_model[m] / max(total, 1) for m in models},
                                             "positives_over_folds": total}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agreement.json").write_text(json.dumps(agreement, indent=1))

    # difficulty: share of (model, seed) predictions that are wrong for each test sample and smell
    wrong: dict[tuple[int, int], list[int]] = defaultdict(list)
    truth: dict[tuple[int, int], int] = {}
    for name in models:
        for key in shared_keys:
            cell = aligned[name][key]
            for k, idx in enumerate(cell["test_index"]):
                for j in range(len(names)):
                    wrong[(idx, j)].append(int(cell["pred_test"][k, j] != cell["y_test"][k, j]))
                    truth[(idx, j)] = int(cell["y_test"][k, j])
    with (out_dir / "sample_difficulty.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["json_index", "smell", "label", "n_predictions", "error_rate"])
        for (idx, j), errs in sorted(wrong.items()):
            writer.writerow([idx, names[j], truth[(idx, j)], len(errs), round(float(np.mean(errs)), 4)])
    print(f"[analysis] wrote {out_dir}")


if __name__ == "__main__":
    main()
