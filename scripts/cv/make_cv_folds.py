"""Build shared cross-validation folds for a dataset without graphs.

Same procedure and output format as scripts/cv/export_cv_folds.py.
An optional `folds.subset` restricts the folds to the samples flagged in a CSV.

Usage:
    python scripts/cv/make_cv_folds.py --config config/experiments/dacos.yml --section dacos
    python scripts/cv/make_cv_folds.py --config config/experiments/dacos_offered_cv.yml --section offered_complex_method
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch_geometric.data import Data  # noqa: E402

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from scripts.train_gcn_baseline import split_kfold  # noqa: E402


def leave_one_group_out(dataset: list[Data], groups: list[str], fcfg: dict) -> list[dict]:
    """Leave-one-group-out folds, with a seeded stratified validation part from the other groups."""
    from sklearn.model_selection import StratifiedKFold

    y = np.array([[int(v) for v in d.y.tolist()] for d in dataset])
    strat = np.array([hash(tuple(row)) % (2 ** 31) for row in y])
    group_arr = np.array(groups)
    out = []
    for seed in fcfg["seeds"]:
        for fold, name in enumerate(sorted(set(groups))):
            test = np.where(group_arr == name)[0]
            rest = np.where(group_arr != name)[0]
            n_inner = max(2, int(round(len(rest) / max(1.0, float(fcfg["val_ratio"]) * len(dataset)))))
            inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=int(seed))
            tr, va = next(inner.split(rest, strat[rest]))
            out.append({"strategy": "logo", "seed": int(seed), "fold": fold, "group": name,
                        "train": sorted(int(i) for i in rest[tr]), "val": sorted(int(i) for i in rest[va]),
                        "test": sorted(int(i) for i in test)})
    return out


def subset_indices(scfg: dict, num_entries: int) -> list[int]:
    """Sorted json_index of the CSV rows whose 0/1 column is 1."""
    key, column = str(scfg.get("key", "json_index")), str(scfg["column"])
    with (PROJECT_ROOT / scfg["csv"]).open() as fh:
        rows = list(csv.DictReader(fh))
    flags: dict[int, bool] = {}
    for row in rows:
        idx, value = int(row[key]), str(row[column]).strip()
        if value not in ("0", "1"):
            raise SystemExit(f"{scfg['csv']}: column {column} holds {value!r} for {key} {idx}, expected 0 or 1")
        if idx in flags:
            raise SystemExit(f"{scfg['csv']}: {key} {idx} appears twice")
        flags[idx] = value == "1"
    if sorted(flags) != list(range(num_entries)):
        raise SystemExit(f"{scfg['csv']}: {key} must cover the {num_entries} JSON entries exactly once")
    return [i for i in range(num_entries) if flags[i]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--section", default="dacos", help="top-level config section with label_order and folds")
    args, extra = parser.parse_known_args()
    cfg = load_config(args.config, parse_cli_overrides(extra))[args.section]
    fcfg = cfg["folds"]
    label_order = list(cfg["label_order"])
    input_json = PROJECT_ROOT / fcfg["input_json"]
    raw = input_json.read_bytes()
    entries = json.loads(raw)
    subset = fcfg.get("subset")
    # json_index of the samples in the folds, indexed by split positions
    index = list(range(len(entries))) if subset is None else subset_indices(subset, len(entries))
    dataset = []
    for i in index:
        d = Data(y=torch.tensor([int(bool(v)) for v in entries[i]["y"]], dtype=torch.int32), num_nodes=1)
        d.json_index = i
        dataset.append(d)
    groups = [str(entries[i].get(fcfg["group_key"], "")) for i in index]

    splits = []
    for strategy in fcfg["strategies"]:
        if strategy == "logo":
            splits += leave_one_group_out(dataset, groups, fcfg)
            continue
        for seed in fcfg["seeds"]:
            for fold in range(int(fcfg["n_folds"])):
                *_, positions = split_kfold(
                    dataset=dataset,
                    groups=groups if strategy == "group_kfold" else None,
                    n_folds=int(fcfg["n_folds"]),
                    fold=fold,
                    val_ratio=float(fcfg["val_ratio"]),
                    seed=int(seed),
                    num_labels=len(label_order),
                )
                splits.append({"strategy": strategy, "seed": int(seed), "fold": fold,
                               **{p: sorted(positions[p]) for p in ("train", "val", "test")}})
    # dataset positions to json_index (index is increasing, lists stay sorted)
    splits = [{**s, **{p: [index[k] for k in s[p]] for p in ("train", "val", "test")}} for s in splits]
    payload = {
        "meta": {
            "config": str(args.config),
            "dataset_fingerprint": hashlib.sha256(raw).hexdigest(),
            "num_samples": len(index),
            "label_order": label_order,
            "n_folds": int(fcfg["n_folds"]),
            "val_ratio": float(fcfg["val_ratio"]),
            "seeds": [int(s) for s in fcfg["seeds"]],
            "strategies": list(fcfg["strategies"]),
            "group_key": fcfg["group_key"],
            "input_json": fcfg["input_json"],
            "json_index": index,
        },
        "splits": splits,
    }
    if subset is not None:
        payload["meta"]["subset"] = {"csv": str(subset["csv"]), "key": str(subset.get("key", "json_index")),
                                     "column": str(subset["column"]), "num_json_entries": len(entries)}
    out = PROJECT_ROOT / fcfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    sizes = {s["strategy"]: (len(s["train"]), len(s["val"]), len(s["test"])) for s in splits if s["fold"] == 0}
    print(f"[folds] wrote {out}: {len(splits)} splits, sizes train/val/test of fold 0 {sizes}")


if __name__ == "__main__":
    main()
