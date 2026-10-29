"""Export the GNN cross-validation folds as indices into the normalized JSON.

Lets models without graphs use the same train, validation and test parts as the GNN.

Usage:
    python scripts/cv/export_cv_folds.py --config config/experiments/gnn_cv.yml \
        --out artifacts/folds/mlcq_cv.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch_geometric.data import Data  # noqa: E402

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.constants import LABEL_ORDER  # noqa: E402
from mlcq_graphs.pipeline import PipelineRunner, to_path  # noqa: E402
from scripts.train_gcn_baseline import split_kfold  # noqa: E402

KFOLD_STRATEGIES = ("kfold", "group_kfold")


def build_cv_folds(
    json_index: list[int],
    labels: list[list[int]],
    groups: list[str],
    strategies: list[str],
    seeds: list[int],
    n_folds: int,
    val_ratio: float,
    num_labels: int,
) -> list[dict[str, Any]]:
    dataset = [Data(y=torch.as_tensor(y).view(-1).int(), num_nodes=1) for y in labels]
    splits = []
    for strategy in strategies:
        for seed in seeds:
            for fold in range(n_folds):
                *_, positions = split_kfold(
                    dataset=dataset,
                    groups=groups if strategy == "group_kfold" else None,
                    n_folds=n_folds,
                    fold=fold,
                    val_ratio=val_ratio,
                    seed=seed,
                    num_labels=num_labels,
                )
                splits.append(
                    {
                        "strategy": strategy,
                        "seed": seed,
                        "fold": fold,
                        **{p: sorted(json_index[i] for i in positions[p]) for p in ("train", "val", "test")},
                    }
                )
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args, extra = parser.parse_known_args()

    # remaining --section.key=value tokens are config overrides
    cfg = load_config(args.config, parse_cli_overrides(extra))
    train_cfg = cfg.get("training", {})
    ablation = cfg.get("ablation", {})
    seeds = [int(s) for s in ablation.get("seeds", [train_cfg.get("seed", 42)])]
    strategies = list(ablation.get("grid", {}).get("training.split_strategy", [train_cfg.get("split_strategy")]))
    strategies = [s for s in strategies if s in KFOLD_STRATEGIES]
    if not strategies:
        raise SystemExit("config has no kfold/group_kfold split strategy")
    n_folds = int(train_cfg.get("cv_folds", 5))
    val_ratio = float(train_cfg.get("val_ratio", 0.1))
    num_labels = int(train_cfg.get("num_labels", 4))
    group_key = str(train_cfg.get("split_group_key", "repo_url"))
    group_json = to_path(PROJECT_ROOT, train_cfg.get("split_group_json") or cfg["construction"]["input_json"])

    runner = PipelineRunner(cfg, PROJECT_ROOT)
    artifacts_root = to_path(PROJECT_ROOT, cfg.get("run", {}).get("artifacts_root", "artifacts"))
    construction = runner._run_construction(cfg, artifacts_root, force_stage=False)
    dataset_state = runner._run_dataset(cfg, artifacts_root, construction, force_stage=False)
    graphs = torch.load(dataset_state.outputs["dataset_path"], weights_only=False)
    json_index = [int(g.json_index) for g in graphs]
    labels = [g.y.view(-1).int().tolist() for g in graphs]
    entries = json.loads(group_json.read_text())
    groups = [str(entries[i].get(group_key, "")) for i in json_index]
    split_dir = artifacts_root / "cache" / "splits" / dataset_state.fingerprint

    splits = build_cv_folds(json_index, labels, groups, strategies, seeds, n_folds, val_ratio, num_labels)

    # the GNN runs cache their parts as positions in the dataset
    position = {j: i for i, j in enumerate(json_index)}
    checked = 0
    for split in splits:
        cached = split_dir / f"split_{split['strategy']}_{group_key}_k{n_folds}_f{split['fold']}_seed_{split['seed']}.json"
        if cached.exists():
            stored = json.loads(cached.read_text())["split_indices"]
            if any(sorted(stored[p]) != sorted(position[j] for j in split[p]) for p in ("train", "val", "test")):
                raise SystemExit(f"rebuilt fold differs from {cached}")
            checked += 1

    payload = {
        "meta": {
            "config": str(args.config),
            "dataset_fingerprint": dataset_state.fingerprint,
            "num_samples": len(json_index),
            "label_order": LABEL_ORDER[:num_labels],
            "n_folds": n_folds,
            "val_ratio": val_ratio,
            "seeds": seeds,
            "strategies": strategies,
            "group_key": group_key,
            "input_json": str(group_json.relative_to(PROJECT_ROOT) if group_json.is_relative_to(PROJECT_ROOT) else group_json),
            "json_index": json_index,
            "checked_against_cache": checked,
        },
        "splits": splits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))
    print(f"[folds] {len(splits)} folds ({checked} checked against cached splits) -> {args.out}")


if __name__ == "__main__":
    main()
