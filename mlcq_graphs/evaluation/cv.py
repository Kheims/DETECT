"""Shared cross-validation protocol for every model family.

Folds are stored as json_index lists. Every model tunes thresholds the same way
and writes records in the schema of the pipeline ablation records.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from mlcq_graphs.constants import LABEL_ORDER
from mlcq_graphs.evaluation.metrics import compute_full_metrics


@dataclass(frozen=True)
class Fold:
    strategy: str
    seed: int
    fold: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def load_folds(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def folds_label_order(folds: dict[str, Any], configured: list[str] | None = None) -> list[str]:
    """Label order of a folds file, checked against the configured order if given."""
    order = [str(s) for s in folds["meta"]["label_order"]]
    if configured is not None and [str(s) for s in configured] != order:
        raise SystemExit(f"config label_order {list(configured)} differs from the folds label order {order}")
    return order


def variant_predictions_path(out_dir: Path, name: str) -> Path:
    """Test predictions of the base-fold model on the metamorphic variant `name`."""
    return Path(out_dir) / f"predictions_variant_{name}.jsonl"


# fields a variant entry may change
VARIANT_FIELDS = ("code_snippet", "metamorphic")


def is_variant(entries: list[dict[str, Any]]) -> bool:
    """Whether a normalized JSON is a metamorphic variant (its entries carry `metamorphic`)."""
    return any("metamorphic" in e for e in entries)


def check_variant_entries(base: list[dict[str, Any]], variant: list[dict[str, Any]], name: str) -> int:
    """Stop unless `variant` matches `base` entry by entry except on VARIANT_FIELDS.

    Returns the number of entries whose code_snippet differs from the base."""
    if len(variant) != len(base):
        raise SystemExit(f"variant {name}: {len(variant)} entries, the base JSON has {len(base)}")
    no_field = [j for j, v in enumerate(variant) if "metamorphic" not in v]
    if no_field:
        raise SystemExit(f"variant {name}: {len(no_field)} entries without a metamorphic field (first {no_field[:5]})")
    differ = [j for j, (b, v) in enumerate(zip(base, variant))
              if {k: x for k, x in b.items() if k not in VARIANT_FIELDS}
              != {k: x for k, x in v.items() if k not in VARIANT_FIELDS}]
    if differ:
        raise SystemExit(f"variant {name}: {len(differ)} entries differ from the base entry at the same position "
                         f"on fields other than {list(VARIANT_FIELDS)} (first {differ[:5]}); a variant keeps the "
                         "length and order of its base JSON")
    return sum(b.get("code_snippet") != v.get("code_snippet") for b, v in zip(base, variant))


def iter_folds(
    folds: dict[str, Any],
    strategies: list[str] | None = None,
    seeds: list[int] | None = None,
    fold_ids: list[int] | None = None,
) -> Iterator[Fold]:
    for split in folds["splits"]:
        if strategies is not None and split["strategy"] not in strategies:
            continue
        if seeds is not None and int(split["seed"]) not in seeds:
            continue
        if fold_ids is not None and int(split["fold"]) not in fold_ids:
            continue
        yield Fold(
            strategy=str(split["strategy"]),
            seed=int(split["seed"]),
            fold=int(split["fold"]),
            train=np.asarray(split["train"], dtype=np.int64),
            val=np.asarray(split["val"], dtype=np.int64),
            test=np.asarray(split["test"], dtype=np.int64),
        )


def fold_digest(fold: Fold) -> str:
    """SHA-256 of the train, val and test indices of a fold."""
    parts = {p: [int(i) for i in getattr(fold, p)] for p in ("train", "val", "test")}
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def shard(items: list[Any], index: int, count: int) -> list[Any]:
    """Round-robin shard, the same convention as the ablation arrays."""
    if count <= 1:
        return list(items)
    if not 0 <= index < count:
        raise ValueError(f"shard index {index} outside [0, {count})")
    return [item for i, item in enumerate(items) if i % count == index]


def label_matrix_from_json(entries: list[dict[str, Any]], indices: np.ndarray, num_labels: int) -> np.ndarray:
    rows = []
    for i in indices:
        y = entries[int(i)]["y"]
        if isinstance(y, str):
            y = json.loads(y.lower())
        rows.append([int(bool(v)) for v in y[:num_labels]])
    return np.asarray(rows, dtype=np.int64)


def tune_thresholds(y_true: np.ndarray, probs: np.ndarray, steps: int) -> np.ndarray:
    """Per-label threshold maximising F1 on validation, as in train_gcn_baseline."""
    grid = np.linspace(0.0, 1.0, max(int(steps), 3), dtype=np.float32)
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    for i in range(y_true.shape[1]):
        truth = y_true[:, i].astype(np.int64)
        if int(truth.sum()) == 0:
            continue
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (probs[:, i] >= t).astype(np.int64)
            score = float(f1_score(truth, pred, zero_division=0))
            if score > best_f1:
                best_f1, best_t = score, float(t)
        thresholds[i] = best_t
    return thresholds


def _metrics(y_true: np.ndarray, probs: np.ndarray, thresholds: np.ndarray, label_names: list[str]) -> dict[str, Any]:
    pred = (probs >= thresholds.reshape(1, -1)).astype(np.int64)
    return compute_full_metrics(y_true.astype(np.int64), pred, probs.astype(np.float64), label_names)


def build_record(
    *,
    combo: dict[str, Any],
    seed: int,
    y_val: np.ndarray,
    p_val: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    threshold_steps: int,
    label_names: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One record in the schema of the pipeline ablation records.json.

    label_names defaults to the MLCQ LABEL_ORDER."""
    names = list(label_names or LABEL_ORDER[: y_test.shape[1]])
    if len(names) != y_test.shape[1]:
        raise ValueError(f"{len(names)} label names for {y_test.shape[1]} label columns")
    thresholds = tune_thresholds(y_val, p_val, threshold_steps)
    tuned = _metrics(y_test, p_test, thresholds, names)
    fixed = _metrics(y_test, p_test, np.full(len(names), 0.5, dtype=np.float32), names)
    record: dict[str, Any] = {
        "seed": int(seed),
        "combo": combo,
        "thresholds": [float(t) for t in thresholds],
        "f1_micro_fixed": fixed["f1_micro"],
        "f1_macro_fixed": fixed["f1_macro"],
        "mcc_fixed": fixed["mcc_macro"],
        "per_label_fixed": fixed["per_label"],
        "f1_micro_tuned": tuned["f1_micro"],
        "f1_macro_tuned": tuned["f1_macro"],
        "pr_auc_tuned": tuned["pr_auc_macro"],
        "mcc_tuned": tuned["mcc_macro"],
        "roc_auc_tuned": tuned["roc_auc_macro"],
        "balanced_acc_tuned": tuned["balanced_acc_macro"],
        "hamming_loss_tuned": tuned["hamming_loss"],
        "subset_acc_tuned": tuned["subset_accuracy"],
        "per_label_tuned": tuned["per_label"],
    }
    if extra:
        record.update(extra)
    return record


def prediction_rows(
    *,
    model: str,
    strategy: str,
    seed: int,
    fold: int,
    part: str,
    indices: np.ndarray,
    y: np.ndarray,
    probs: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "model": model,
            "strategy": strategy,
            "seed": int(seed),
            "fold": int(fold),
            "part": part,
            "json_index": int(idx),
            "y": [int(v) for v in y[k]],
            "prob": [round(float(v), 6) for v in probs[k]],
        }
        for k, idx in enumerate(indices)
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]
