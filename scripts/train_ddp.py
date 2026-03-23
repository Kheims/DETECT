"""Distributed Data Parallel training wrapper.

Thin script that imports all training logic from train_gcn_baseline.py and
mlcq_graphs.*, adding only DDP orchestration (process group init/cleanup,
model wrapping, rank-gated I/O).

Launch via torchrun:
    torchrun --standalone --nproc_per_node=4 scripts/train_ddp.py \
        --dataset-path artifacts/cache/dataset/latest/dataset.pt \
        --architecture gcn --loss focal --epochs 40

Single-GPU fallback (no torchrun):
    python scripts/train_ddp.py --dataset-path ... --epochs 40
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import precision_recall_curve, roc_curve
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.constants import LABEL_ORDER
from mlcq_graphs.evaluation import evaluate_full
from mlcq_graphs.models import get_model
from mlcq_graphs.training import Trainer, TrainResult, build_loss_fn
from scripts.train_gcn_baseline import (
    SMELL_LABELS,
    DynamicBudgetBatchSampler,
    apply_split_indices,
    attach_token_vectors,
    build_loader,
    collect_logits_and_targets,
    evaluate,
    load_dataset,
    load_split_indices,
    parse_args,
    run_name,
    save_split_indices,
    select_device,
    set_seed,
    split_sample_level,
    summarize_dataset,
    tune_thresholds,
)

class DistributedDynamicBudgetBatchSampler(DynamicBudgetBatchSampler):
    """DDP-aware variant that rank-partitions samples before budget batching.

    Each rank sees a disjoint subset of graph indices (stride-based slicing)
    with the same per-GPU OOM protection from the parent sampler.
    """

    def __init__(
        self,
        node_counts: list[int],
        edge_counts: list[int],
        max_nodes: int,
        max_edges: int | None,
        shuffle: bool,
        seed: int,
        rank: int,
        world_size: int,
    ) -> None:
        super().__init__(
            node_counts=node_counts,
            edge_counts=edge_counts,
            max_nodes=max_nodes,
            max_edges=max_edges,
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
        )
        self.rank = rank
        self.world_size = world_size

    def _ordered_indices(self) -> list[int]:
        all_indices = super()._ordered_indices()
        remainder = len(all_indices) % self.world_size
        if remainder:
            all_indices = all_indices + all_indices[: self.world_size - remainder]
        return all_indices[self.rank :: self.world_size]

def build_ddp_loader(
    dataset: list,
    max_nodes_per_batch: int,
    max_edges_per_batch: int | None,
    shuffle: bool,
    seed: int,
    epoch: int,
    num_workers: int,
    pin_memory: bool,
    rank: int,
    world_size: int,
) -> DataLoader:
    node_counts = [int(d.num_nodes) for d in dataset]
    edge_counts = [int(d.edge_index.size(1)) for d in dataset]
    batch_sampler = DistributedDynamicBudgetBatchSampler(
        node_counts=node_counts,
        edge_counts=edge_counts,
        max_nodes=max_nodes_per_batch,
        max_edges=max_edges_per_batch,
        shuffle=shuffle,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )
    batch_sampler.set_epoch(epoch)
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

def init_distributed(backend: str = "nccl") -> tuple[int, int, int, bool]:
    """Initialize DDP if launched via torchrun, otherwise return single-GPU defaults."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 0, 1, False

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    return rank, local_rank, world_size, True

def main() -> None:
    args = parse_args()

    rank, local_rank, world_size, is_distributed = init_distributed(
        backend=getattr(args, "distributed_backend", "nccl"),
    )
    is_main = rank == 0

    seed = args.seed
    set_seed(seed + rank)

    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = select_device(args.device)

    cfg: dict[str, Any] = vars(args)

    max_edges = int(cfg.get("max_edges_per_batch", 0))
    eval_max_edges = int(cfg.get("eval_max_edges_per_batch", 0))
    edge_budget = max_edges if max_edges > 0 else None
    eval_edge_budget = eval_max_edges if eval_max_edges > 0 else None

    dataset_path = Path(str(cfg.get("dataset_path", "artifacts/cache/dataset/latest/dataset.pt")))
    num_labels = int(cfg.get("num_labels", 4))
    max_graphs = cfg.get("max_graphs")
    if max_graphs is not None:
        max_graphs = int(max_graphs)
    to_undirected_edges = bool(cfg.get("to_undirected", True))

    dataset = load_dataset(
        dataset_path=dataset_path,
        num_labels=num_labels,
        max_graphs=max_graphs,
        to_undirected_edges=to_undirected_edges,
    )

    token_vectors_path = cfg.get("token_vectors_path")
    manifest_path = cfg.get("manifest_path")
    use_token_features = token_vectors_path is not None
    token_feature_dim = 0
    if use_token_features:
        if manifest_path is None:
            raise ValueError("manifest_path is required when token_vectors_path is provided.")
        token_feature_dim, _ = attach_token_vectors(
            dataset=dataset,
            manifest_path=Path(str(manifest_path)),
            vectors_path=Path(str(token_vectors_path)),
        )

    train_ratio = float(cfg.get("train_ratio", 0.8))
    val_ratio = float(cfg.get("val_ratio", 0.1))
    split_path_raw = cfg.get("split_path")
    split_path = Path(str(split_path_raw)) if split_path_raw is not None else None

    if split_path is not None and split_path.exists():
        split_indices = load_split_indices(split_path)
        train_ds, val_ds, test_ds = apply_split_indices(dataset, split_indices)
    else:
        train_ds, val_ds, test_ds, split_indices = split_sample_level(
            dataset=dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            num_labels=num_labels,
        )
        if is_main and split_path is not None:
            save_split_indices(
                split_path=split_path,
                split_indices=split_indices,
                seed=seed,
                dataset_size=len(dataset),
            )

    if not train_ds or not val_ds or not test_ds:
        raise ValueError("One split is empty; adjust split ratios or dataset size.")

    feature_mode = str(cfg.get("feature_mode", "type_numeric"))
    use_type_features = feature_mode in {"type_numeric", "type_only"}
    use_numeric_features = feature_mode in {"type_numeric", "numeric_only"}

    num_node_types = max(int(d.type_id.max().item()) for d in dataset) + 1
    num_numeric_feats = int(train_ds[0].x.size(-1)) if use_numeric_features else 0

    architecture = str(cfg.get("architecture", "gcn"))
    num_layers = int(cfg.get("num_layers", 2))
    hidden_dim = int(cfg.get("hidden_dim", 256))
    type_emb_dim = int(cfg.get("type_emb_dim", 128))
    dropout = float(cfg.get("dropout", 0.2))
    num_heads = int(cfg.get("num_heads", 4))
    aggregation = str(cfg.get("aggregation", "mean"))
    loss_name = str(cfg.get("loss", "weighted_bce"))

    arch_kwargs: dict[str, Any] = {}
    if architecture == "gat":
        arch_kwargs["num_heads"] = num_heads
    elif architecture == "graphsage":
        arch_kwargs["aggregation"] = aggregation

    model = get_model(
        architecture,
        num_node_types=num_node_types,
        type_emb_dim=type_emb_dim,
        num_numeric_feats=num_numeric_feats,
        num_token_feats=token_feature_dim,
        hidden_dim=hidden_dim,
        num_labels=num_labels,
        dropout=dropout,
        num_layers=num_layers,
        use_type_features=use_type_features,
        use_numeric_features=use_numeric_features,
        use_token_features=use_token_features,
        **arch_kwargs,
    ).to(device)

    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    lr = float(cfg.get("lr", 0.0005))
    weight_decay = float(cfg.get("weight_decay", 5e-5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    focal_gamma = float(cfg.get("focal_gamma", 2.0))
    focal_alpha = float(cfg.get("focal_alpha", -1.0))
    loss_fn = build_loss_fn(
        loss_name=loss_name,
        train_ds=train_ds,
        num_labels=num_labels,
        device=device,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )

    output_dir = Path(str(cfg.get("output_dir", "artifacts/runs")))
    run_dir_name = run_name(architecture, cfg.get("run_name"))
    run_dir = output_dir / run_dir_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
    if is_distributed:
        dist.barrier()

    epochs = int(cfg.get("epochs", 40))
    patience = int(cfg.get("patience", 10))
    early_stopping_metric = str(cfg.get("early_stopping_metric", "macro_f1"))
    early_stopping_min_delta = float(cfg.get("early_stopping_min_delta", 0.0))
    grad_accum_steps = int(cfg.get("grad_accum_steps", 1))
    grad_clip_norm = float(cfg.get("grad_clip_norm", 1.0))
    threshold_steps = int(cfg.get("threshold_steps", 101))

    if is_main:
        train_summary = summarize_dataset(train_ds, SMELL_LABELS)
        val_summary = summarize_dataset(val_ds, SMELL_LABELS)
        test_summary = summarize_dataset(test_ds, SMELL_LABELS)

        config_snapshot = {
            "device": str(device),
            "architecture": architecture,
            "distributed": is_distributed,
            "world_size": world_size,
            "num_layers": num_layers,
            "feature_mode": feature_mode,
            "loss": loss_name,
            "focal_gamma": focal_gamma,
            "focal_alpha": focal_alpha,
            "early_stopping_metric": early_stopping_metric,
            "train_summary": asdict(train_summary),
            "val_summary": asdict(val_summary),
            "test_summary": asdict(test_summary),
            "split_sizes": {
                "train": len(train_ds),
                "val": len(val_ds),
                "test": len(test_ds),
            },
            "num_node_types": num_node_types,
        }
        (run_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2, default=str))

    raw_model = model.module if is_distributed else model

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=epochs,
        patience=patience,
        min_delta=early_stopping_min_delta,
        early_stopping_metric=early_stopping_metric,
        checkpoint_dir=run_dir,
        config_snapshot={} if not is_main else config_snapshot,
        seed=seed,
        grad_accum_steps=grad_accum_steps,
        grad_clip_norm=grad_clip_norm,
        architecture=architecture,
        loss_name=loss_name,
        wandb_run=None,
        is_main=is_main,
    )

    max_nodes_per_batch = int(cfg.get("max_nodes_per_batch", 20000))
    eval_max_nodes = int(cfg.get("eval_max_nodes_per_batch", 40000))
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", False))

    def train_loader_fn(epoch: int) -> DataLoader:
        return build_ddp_loader(
            dataset=train_ds,
            max_nodes_per_batch=max_nodes_per_batch,
            max_edges_per_batch=edge_budget,
            shuffle=True,
            seed=seed,
            epoch=epoch,
            num_workers=num_workers,
            pin_memory=pin_memory,
            rank=rank,
            world_size=world_size,
        )

    val_loader = build_loader(
        dataset=val_ds,
        max_nodes_per_batch=eval_max_nodes,
        max_edges_per_batch=eval_edge_budget,
        shuffle=False,
        seed=seed,
        epoch=0,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    train_start = time.perf_counter()
    result: TrainResult = trainer.fit(
        train_loader_fn=train_loader_fn,
        val_loader=val_loader,
        num_labels=num_labels,
        evaluate_fn=evaluate,
    )
    train_duration_sec = time.perf_counter() - train_start
    peak_gpu_bytes = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0

    # Synchronize before loading best weights
    if is_distributed:
        dist.barrier()

    # Only rank 0 does evaluation and writes outputs
    if not is_main:
        dist.destroy_process_group()
        return

    best_epoch = result.best_epoch

    # Use raw model (unwrapped) for evaluation
    eval_model = raw_model

    val_loader = build_loader(
        dataset=val_ds,
        max_nodes_per_batch=eval_max_nodes,
        max_edges_per_batch=eval_edge_budget,
        shuffle=False,
        seed=seed,
        epoch=0,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = build_loader(
        dataset=test_ds,
        max_nodes_per_batch=eval_max_nodes,
        max_edges_per_batch=eval_edge_budget,
        shuffle=False,
        seed=seed,
        epoch=0,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_logits, val_targets = collect_logits_and_targets(
        model=eval_model,
        loader=val_loader,
        device=device,
        num_labels=num_labels,
    )
    val_probs = torch.sigmoid(val_logits)
    thresholds = tune_thresholds(
        y_true=val_targets.int(),
        probs=val_probs,
        steps=threshold_steps,
    )

    smell_labels = SMELL_LABELS[:num_labels]

    eval_start = time.perf_counter()
    test_logits, test_targets = collect_logits_and_targets(
        model=eval_model,
        loader=test_loader,
        device=device,
        num_labels=num_labels,
    )
    eval_duration_sec = time.perf_counter() - eval_start
    eval_throughput = len(test_ds) / max(eval_duration_sec, 1e-9)
    total_train_graphs = len(train_ds) * max(result.best_epoch, 1)
    train_throughput = total_train_graphs / max(train_duration_sec, 1e-9)

    test_metrics_fixed = evaluate_full(
        logits=test_logits,
        y_true=test_targets,
        thresholds=None,
        label_names=smell_labels,
    )
    test_metrics_tuned = evaluate_full(
        logits=test_logits,
        y_true=test_targets,
        thresholds=thresholds,
        label_names=smell_labels,
    )

    print(f"\nbest epoch: {best_epoch}\n")
    print("thresholds (tuned):")
    for i, label in enumerate(smell_labels):
        short = label.removeprefix("is_")
        print(f"  {short:<14} {float(thresholds[i]):.2f}")

    print()
    print(f"{'':>13} {'f1_micro':>9} {'f1_macro':>9} {'pr_auc':>7} {'hamming':>8} {'subset_acc':>11}")
    for tag, m in [("fixed 0.5", test_metrics_fixed), ("tuned", test_metrics_tuned)]:
        print(
            f"{tag:<13} {float(m['f1_micro']):>9.4f} {float(m['f1_macro']):>9.4f} "
            f"{float(m['pr_auc_macro']):>7.4f} {float(m['hamming_loss']):>8.4f} "
            f"{float(m['subset_accuracy']):>11.4f}"
        )

    # Curve data for publication plots
    test_probs_np = torch.sigmoid(test_logits).numpy()
    test_y_np = test_targets.int().numpy()

    pr_curves: dict[str, dict] = {}
    roc_curves_data: dict[str, dict] = {}
    f1_vs_threshold: dict[str, dict] = {}

    for i, name in enumerate(smell_labels):
        y_t = test_y_np[:, i]
        p = test_probs_np[:, i]

        if y_t.sum() == 0:
            pr_curves[name] = {}
            roc_curves_data[name] = {}
            f1_vs_threshold[name] = {}
            continue

        prec, rec, pr_thresh = precision_recall_curve(y_t, p)
        fpr, tpr, roc_thresh = roc_curve(y_t, p)

        thresh_grid = np.linspace(0.0, 1.0, threshold_steps).tolist()
        f1_vals = [
            float(sklearn_f1_score(y_t, (p >= t).astype(int), zero_division=0))
            for t in thresh_grid
        ]

        pr_curves[name] = {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "thresholds": pr_thresh.tolist(),
        }
        roc_curves_data[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresh.tolist(),
        }
        f1_vs_threshold[name] = {
            "thresholds": thresh_grid,
            "f1": f1_vals,
            "tuned_threshold": float(thresholds[i]),
        }

    curves_payload = {
        "pr_curves": pr_curves,
        "roc_curves": roc_curves_data,
        "f1_vs_threshold": f1_vs_threshold,
    }
    (run_dir / "curves.json").write_text(json.dumps(curves_payload, indent=2))

    artifacts = {
        "best_epoch": best_epoch,
        "stopped_early": result.stopped_early,
        "best_val_metric": result.best_val_metric,
        "train_duration_sec": train_duration_sec,
        "history": result.history,
        "distributed": {
            "enabled": is_distributed,
            "world_size": world_size,
        },
        "threshold_tuning": {
            "strategy": "per_label_val_f1_grid",
            "threshold_steps": threshold_steps,
        },
        "thresholds": {
            smell_labels[i] if i < len(smell_labels) else f"label_{i}": float(thresholds[i])
            for i in range(num_labels)
        },
        "test_fixed_0_5": test_metrics_fixed,
        "test_tuned": test_metrics_tuned,
        "profiling": {
            "train_duration_sec": round(train_duration_sec, 3),
            "avg_epoch_duration_sec": round(train_duration_sec / max(result.best_epoch, 1), 3),
            "peak_gpu_memory_bytes": peak_gpu_bytes,
            "peak_gpu_memory_mb": round(peak_gpu_bytes / 1024 ** 2, 2),
            "train_throughput_graphs_per_sec": round(train_throughput, 2),
            "eval_throughput_graphs_per_sec": round(eval_throughput, 2),
        },
    }

    (run_dir / "metrics.json").write_text(json.dumps(artifacts, indent=2))
    print(json.dumps(artifacts, indent=2))

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
