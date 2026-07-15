"""Apply the models trained by an experiment config to another labelled dataset.

Thresholds come from the source validation split. Metrics at 0.5 are reported too.

Usage:
    python scripts/analysis/eval_transfer.py --config config/experiments/ts_baseline.yml \\
        --target-json data/transfer/smellycode_pp/smellycode.normalized.json \\
        --target-name smellycode [--key=value ...]
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides, set_nested_value
from mlcq_graphs.constants import LABEL_ORDER
from mlcq_graphs.evaluation import evaluate_full
from mlcq_graphs.models import get_model
from mlcq_graphs.pipeline import PipelineRunner, to_path
from scripts.train_gcn_baseline import attach_token_vectors, build_loader, collect_logits_and_targets, load_dataset


def trial_configs(config: dict) -> list[tuple[dict, dict, int]]:
    ablation = config.get("ablation", {}) or {}
    if not ablation.get("enabled", False):
        return [(copy.deepcopy(config), {}, int(config.get("training", {}).get("seed", 42)))]
    grid = ablation.get("grid", {}) or {}
    keys = list(grid.keys())
    values = [v if isinstance(v, list) else [v] for v in grid.values()]
    out = []
    for combo_values in itertools.product(*values):
        combo = dict(zip(keys, combo_values))
        for seed in ablation.get("seeds", [42]):
            cfg = copy.deepcopy(config)
            cfg["ablation"]["enabled"] = False
            cfg.setdefault("training", {})["seed"] = int(seed)
            for key, value in combo.items():
                set_nested_value(cfg, key, value)
            out.append((cfg, combo, int(seed)))
    return out


def load_model(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    snap = json.loads((run_dir / "config.json").read_text())
    state = ckpt["model_state_dict"]
    kwargs = {
        "pooling": snap.get("pooling", "mean"),
        "set2set_steps": snap.get("set2set_steps", 3),
        "subset_pooling": bool(snap.get("pool_subset_types")),
        "num_graph_feats": len(snap.get("graph_features") or []),
    }
    if snap.get("graph_features") or snap.get("pool_subset_types"):
        raise ValueError("transfer of models with graph features or subset pooling is not supported")
    if snap["architecture"] == "gat":
        kwargs["num_heads"] = snap.get("num_heads", 4)
    if snap["architecture"] == "graphsage":
        kwargs["aggregation"] = snap.get("aggregation", "mean")
    if snap["architecture"] == "rgcn":
        kwargs["num_relations"] = snap.get("num_relations", 1)
    model = get_model(
        snap["architecture"],
        num_node_types=int(snap["num_node_types"]),
        type_emb_dim=int(state["type_emb.weight"].shape[1]) if "type_emb.weight" in state else 128,
        num_numeric_feats=int(snap["num_numeric_feats"]),
        num_token_feats=int(snap["num_token_feats"]),
        hidden_dim=int(state["lin1.weight"].shape[0]),
        num_labels=int(state["lin2.weight"].shape[0]),
        dropout=0.0,
        num_layers=int(snap["num_layers"]),
        use_type_features=bool(snap["use_type_features"]),
        use_numeric_features=bool(snap["use_numeric_features"]),
        use_token_features=bool(snap["use_token_features"]),
        **kwargs,
    )
    model.load_state_dict(state)
    return model.to(device).eval(), snap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-json", type=Path, required=True)
    parser.add_argument("--target-name", default="target")
    parser.add_argument("--device", default="auto")
    args, unknown = parser.parse_known_args()

    config = load_config(args.config, parse_cli_overrides(unknown))
    runner = PipelineRunner(config=config, project_root=PROJECT_ROOT)
    artifacts_root = to_path(PROJECT_ROOT, config.get("run", {}).get("artifacts_root", "artifacts"))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))

    # target graphs with the same construction and dataset settings
    target_cfg = copy.deepcopy(config)
    target_cfg["construction"]["input_json"] = str(args.target_json)
    if (target_cfg["construction"].get("relational") or {}).get("enabled"):
        raise ValueError("relational graphs need enclosing files, not available for the target")
    t_construction = runner._run_construction(target_cfg, artifacts_root, force_stage=False)
    t_dataset = runner._run_dataset(target_cfg, artifacts_root, t_construction, force_stage=False)
    target_vocab = json.loads(Path(t_dataset.outputs["node_type_vocab_path"]).read_text())
    target_entries = json.loads(args.target_json.read_text())

    rows = []
    cached_target = None
    for trial_cfg, combo, seed in trial_configs(config):
        result = runner._run_single(trial_cfg)
        stages = result["stages"]
        run_dir = Path(stages["training"]["outputs"]["run_dir"])
        source_vocab = json.loads(Path(stages["dataset"]["outputs"]["node_type_vocab_path"]).read_text())
        model, snap = load_model(run_dir, device)

        if cached_target is None:
            data = load_dataset(
                Path(t_dataset.outputs["dataset_path"]),
                num_labels=len(LABEL_ORDER),
                max_graphs=None,
                to_undirected_edges=bool(trial_cfg["training"].get("to_undirected", True)),
            )
            remap = torch.zeros(max(target_vocab.values()) + 1, dtype=torch.long)
            unknown = [name for name in target_vocab if name not in source_vocab]
            for name, idx in target_vocab.items():
                remap[idx] = source_vocab.get(name, 0)
            unknown_ids = torch.tensor(sorted(target_vocab[n] for n in unknown), dtype=torch.long)
            unknown_nodes = 0
            for g in data:
                unknown_nodes += int(torch.isin(g.type_id, unknown_ids).sum())
                g.type_id = remap[g.type_id]
            token_stats = None
            if snap["use_token_features"]:
                _, token_stats = attach_token_vectors(
                    data,
                    Path(t_construction.outputs["manifest_path"]),
                    Path(stages["token_encoder"]["outputs"]["vectors_path"]),
                    progress_every=1000,
                )
            cached_target = (data, {"unknown_types": unknown, "unknown_nodes": unknown_nodes, "token_stats": token_stats})
        data, target_info = cached_target

        loader = build_loader(data, max_nodes_per_batch=80000, max_edges_per_batch=None, shuffle=False,
                              seed=0, epoch=0, num_workers=0, pin_memory=False)
        logits, targets = collect_logits_and_targets(model=model, loader=loader, device=device, num_labels=len(LABEL_ORDER))
        metrics = json.loads((run_dir / "metrics.json").read_text())
        thresholds = torch.tensor([metrics["thresholds"][label] for label in LABEL_ORDER])
        tuned = evaluate_full(logits, targets, thresholds, LABEL_ORDER)
        fixed = evaluate_full(logits, targets, None, LABEL_ORDER)

        levels = {}
        json_ids = [int(g.json_index) for g in data]
        for level in ("class", "method"):
            idx = [i for i, j in enumerate(json_ids) if target_entries[j].get("level") == level]
            if idx:
                levels[level] = evaluate_full(logits[idx], targets[idx], thresholds, LABEL_ORDER)["per_label"]
        rows.append({
            "combo": combo,
            "seed": seed,
            "run_dir": str(run_dir),
            "source_test_per_label": metrics["test_tuned"]["per_label"],
            "target_tuned": tuned,
            "target_fixed_0_5": fixed,
            "target_by_level_tuned": levels,
        })
        print(f"[transfer] seed={seed} combo={combo} target MCC "
              + " ".join(f"{lab.removeprefix('is_')}={tuned['per_label'][lab]['mcc']}" for lab in LABEL_ORDER), flush=True)

    summary = {}
    for label in LABEL_ORDER:
        for split, key in (("source", "source_test_per_label"), ("target", "target_tuned")):
            vals = []
            for r in rows:
                per_label = r[key] if split == "source" else r[key]["per_label"]
                v = per_label.get(label, {}).get("mcc")
                if v is not None:
                    vals.append(float(v))
            if vals:
                summary[f"{label}_{split}_mcc"] = [round(statistics.mean(vals), 3), round(statistics.pstdev(vals), 3)]
    report = {
        "config": str(args.config),
        "target": str(args.target_json),
        "target_graphs": len(cached_target[0]) if cached_target else 0,
        "target_info": cached_target[1] if cached_target else None,
        "summary_mcc_mean_std": summary,
        "runs": rows,
    }
    out = artifacts_root / "transfer" / f"{config.get('run', {}).get('name', 'run')}_to_{args.target_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(summary, indent=2))
    print(f"[transfer] wrote {out}")


if __name__ == "__main__":
    main()
