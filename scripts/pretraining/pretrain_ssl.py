"""Self-supervised pre-training of a DETECT GNN encoder on unlabelled AST graphs.

Objectives: masked node type prediction and graph contrastive learning (NT-Xent).
Runs on one device or with DistributedDataParallel under torchrun.

Usage:
    python scripts/pretraining/pretrain_ssl.py --config-json run_cfg.json
    torchrun --nproc_per_node 4 scripts/pretraining/pretrain_ssl.py --config-json run_cfg.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_undirected

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.models import get_model


# data

def subtree_sizes(num_nodes: int, child_edges: torch.Tensor) -> torch.Tensor:
    """Size of the subtree rooted at each node, computed level by level."""
    parent = torch.full((num_nodes,), -1, dtype=torch.long)
    if child_edges.numel():
        parent[child_edges[1]] = child_edges[0]
    has_parent = parent >= 0
    depth = torch.zeros(num_nodes, dtype=torch.long)
    while True:
        new_depth = torch.where(has_parent, depth[parent.clamp(min=0)] + 1, depth)
        if torch.equal(new_depth, depth):
            break
        depth = new_depth
    size = torch.ones(num_nodes, dtype=torch.long)
    for level in range(int(depth.max()) if num_nodes else 0, 0, -1):
        nodes = torch.nonzero(depth == level).view(-1)
        size.index_add_(0, parent[nodes], size[nodes])
    return size


def cut_units(
    data: Data,
    child_type_ids: set[int],
    unit_type_ids: set[int],
    keep_file: bool,
    min_nodes: int,
    max_nodes: int,
) -> list[Data]:
    """Return the graph itself and/or its method and class subtrees as separate graphs."""
    n = data.num_nodes
    edge_type = data.edge_type
    child_mask = torch.isin(edge_type, torch.tensor(sorted(child_type_ids)))
    size = subtree_sizes(n, data.edge_index[:, child_mask])
    out: list[Data] = []
    if keep_file and min_nodes <= n <= max_nodes:
        out.append(Data(type_id=data.type_id, edge_index=data.edge_index, sub_size=size, num_nodes=n))
    if unit_type_ids:
        roots = torch.nonzero(torch.isin(data.type_id, torch.tensor(sorted(unit_type_ids)))).view(-1)
        src, dst = data.edge_index
        for root in roots.tolist():
            length = int(size[root])
            if not min_nodes <= length <= max_nodes:
                continue
            lo, hi = root, root + length
            keep = (src >= lo) & (src < hi) & (dst >= lo) & (dst < hi)
            out.append(
                Data(
                    type_id=data.type_id[lo:hi],
                    edge_index=data.edge_index[:, keep] - lo,
                    sub_size=size[lo:hi],
                    num_nodes=length,
                )
            )
    return out


def load_units(cfg: dict[str, Any], vocab: dict[str, int], log) -> list[Data]:
    raw = torch.load(cfg["dataset_path"], weights_only=False)
    edge_order = list(cfg["edge_type_order"])
    child_ids = {edge_order.index("Child")}
    units_cfg = cfg.get("units", {}) or {}
    unit_names = [str(t) for t in units_cfg.get("node_types", [])]
    unit_ids = {vocab[t] for t in unit_names if t in vocab}
    keep_file = bool(units_cfg.get("keep_file", not unit_ids))
    min_nodes = int(units_cfg.get("min_nodes", 10))
    max_nodes = int(units_cfg.get("max_nodes", 15000))

    units: list[Data] = []
    for data in raw:
        data.type_id = data.type_id.view(-1).long()
        for unit in cut_units(data, child_ids, unit_ids, keep_file, min_nodes, max_nodes):
            unit.edge_index = to_undirected(unit.edge_index, num_nodes=unit.num_nodes)
            units.append(unit)
    log(f"[pretrain] files={len(raw)} units={len(units)} unit_types={unit_names} keep_file={keep_file}")
    return units


def budget_batches(indices: list[int], sizes: list[int], max_nodes: int) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    used = 0
    for idx in indices:
        n = sizes[idx]
        if current and used + n > max_nodes:
            batches.append(current)
            current, used = [], 0
        current.append(idx)
        used += n
    if current:
        batches.append(current)
    return batches


# model

class PretrainModel(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int, num_types: int, proj_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.type_head = nn.Linear(hidden_dim, num_types)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, proj_dim)
        )

    def forward(
        self,
        masked: Batch | None,
        view1: Batch | None,
        view2: Batch | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        # one forward call per step, as DistributedDataParallel expects
        logits = self.type_head(self.encoder.encode_nodes(masked)) if masked is not None else None
        z1 = z2 = None
        if view1 is not None and view2 is not None:
            z1 = self.proj_head(global_mean_pool(self.encoder.encode_nodes(view1), view1.batch))
            z2 = self.proj_head(global_mean_pool(self.encoder.encode_nodes(view2), view2.batch))
        return logits, z1, z2


def mask_nodes(batch: Batch, ratio: float, mode: str, max_subtree: int, gen: torch.Generator) -> torch.Tensor:
    n = batch.num_nodes
    device = batch.type_id.device
    if mode == "node":
        return torch.rand(n, generator=gen, device=device) < ratio
    # subtree masking: pick roots so that the expected masked fraction is `ratio`
    size = batch.sub_size.clamp(min=1)
    eligible = size <= max_subtree
    mean_size = size[eligible].float().mean().clamp(min=1.0)
    p_root = (ratio / mean_size).clamp(max=1.0)
    roots = eligible & (torch.rand(n, generator=gen, device=device) < p_root)
    idx = torch.nonzero(roots).view(-1)
    diff = torch.zeros(n + 1, device=device, dtype=torch.long)
    diff.index_add_(0, idx, torch.ones_like(idx))
    diff.index_add_(0, (idx + size[idx]).clamp(max=n), -torch.ones_like(idx))
    return diff[:n].cumsum(0) > 0


def augment(batch: Batch, mask_token: int, feat_p: float, edge_p: float, gen: torch.Generator) -> Batch:
    view = batch.clone()
    device = view.type_id.device
    hide = torch.rand(view.num_nodes, generator=gen, device=device) < feat_p
    view.type_id = torch.where(hide, torch.full_like(view.type_id, mask_token), view.type_id)
    keep = torch.rand(view.edge_index.size(1), generator=gen, device=device) >= edge_p
    view.edge_index = view.edge_index[:, keep]
    return view


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z = F.normalize(torch.cat([z1, z2]), dim=-1)
    sim = z @ z.t() / temperature
    b = z1.size(0)
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, target)


# baselines

def parent_lookup_baseline(units: list[Data], train_idx: list[int], val_idx: list[int]) -> dict[str, float]:
    """Accuracy of the parent-type lookup and of the majority type."""
    pair = defaultdict(Counter)
    overall = Counter()

    def parents(unit: Data) -> torch.Tensor:
        # in a pre-order walk the parent is the closest earlier node whose subtree covers this one
        n = unit.num_nodes
        par = torch.full((n,), -1, dtype=torch.long)
        stack: list[int] = []
        for i in range(n):
            while stack and stack[-1] + int(unit.sub_size[stack[-1]]) <= i:
                stack.pop()
            par[i] = stack[-1] if stack else -1
            stack.append(i)
        return par

    for i in train_idx[:2000]:
        unit = units[i]
        par = parents(unit)
        t = unit.type_id
        for node in range(1, unit.num_nodes):
            pair[int(t[par[node]])][int(t[node])] += 1
            overall[int(t[node])] += 1
    best = {p: c.most_common(1)[0][0] for p, c in pair.items()}
    majority = overall.most_common(1)[0][0] if overall else -1

    hit_parent = hit_major = total = 0
    for i in val_idx[:500]:
        unit = units[i]
        par = parents(unit)
        t = unit.type_id
        for node in range(1, unit.num_nodes):
            total += 1
            hit_parent += int(best.get(int(t[par[node]]), majority) == int(t[node]))
            hit_major += int(majority == int(t[node]))
    return {
        "parent_type_lookup_acc": hit_parent / max(total, 1),
        "majority_type_acc": hit_major / max(total, 1),
    }


# main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-json", type=Path, required=True)
    args = parser.parse_args()
    cfg: dict[str, Any] = json.loads(args.config_json.read_text())

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world > 1
    if distributed:
        dist.init_process_group(backend=str(cfg.get("dist_backend", "nccl")))
    if torch.cuda.is_available() and str(cfg.get("device", "auto")) != "cpu":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)

    out_dir = Path(cfg["output_dir"])
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    vocab: dict[str, int] = json.loads(Path(cfg["node_type_vocab_path"]).read_text())
    num_types = len(vocab)
    mask_token = num_types          # MASK id
    unk_token = num_types + 1       # id for MLCQ types unseen in the corpus
    units = load_units(cfg, vocab, log)
    sizes = [u.num_nodes for u in units]

    order = list(range(len(units)))
    random.Random(seed).shuffle(order)
    n_val = max(1, int(float(cfg.get("val_fraction", 0.02)) * len(order)))
    val_idx, train_idx = order[:n_val], order[n_val:]

    objectives = cfg.get("objectives", {}) or {}
    mt_cfg = objectives.get("masked_type", {}) or {}
    ct_cfg = objectives.get("contrastive", {}) or {}
    w_mt = float(mt_cfg.get("weight", 0.0))
    w_ct = float(ct_cfg.get("weight", 0.0))
    if w_mt <= 0 and w_ct <= 0:
        raise ValueError("pretraining.objectives needs a positive weight")

    baselines = parent_lookup_baseline(units, train_idx, val_idx) if rank == 0 else {}
    log(f"[pretrain] baselines {baselines}")

    hidden_dim = int(cfg.get("hidden_dim", 512))
    encoder = get_model(
        str(cfg.get("architecture", "gcn")),
        num_node_types=num_types + 2,
        type_emb_dim=int(cfg.get("type_emb_dim", 128)),
        num_numeric_feats=0,
        num_token_feats=0,
        hidden_dim=hidden_dim,
        num_labels=1,
        dropout=float(cfg.get("dropout", 0.2)),
        num_layers=int(cfg.get("num_layers", 3)),
        use_type_features=True,
        use_numeric_features=False,
        use_token_features=False,
    )
    model = PretrainModel(encoder, hidden_dim, num_types, int(ct_cfg.get("proj_dim", 128))).to(device)
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    core = model.module if distributed else model

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.get("lr", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-5))
    )
    epochs = int(cfg.get("epochs", 10))
    max_steps = cfg.get("max_steps_per_epoch")
    # weak scaling keeps the per-GPU node budget, strong scaling splits a global one
    budget = int(cfg.get("max_nodes_per_batch", 20000))
    if str(cfg.get("batch_scaling", "weak")) == "strong":
        budget = max(1, budget // world)
    gen = torch.Generator(device=device).manual_seed(seed + rank)

    def run_epoch(epoch: int, indices: list[int], train: bool) -> dict[str, float]:
        rng = random.Random(seed * 1000 + epoch)
        idx = list(indices)
        if train:
            rng.shuffle(idx)
        batches = budget_batches(idx[rank::world], sizes, budget)
        steps = torch.tensor(len(batches), device=device)
        if distributed:
            dist.all_reduce(steps, op=dist.ReduceOp.MIN)
        n_steps = int(steps)
        if train and max_steps is not None:
            n_steps = min(n_steps, int(max_steps))
        model.train(train)
        sums = torch.zeros(7, device=device)  # loss, mt_loss, ct_loss, masked, correct, graphs, nodes
        for step in range(n_steps):
            batch = Batch.from_data_list([units[i] for i in batches[step]]).to(device, non_blocking=True)
            loss = torch.zeros((), device=device)
            with torch.set_grad_enabled(train):
                masked = mask = None
                if w_mt > 0:
                    mask = mask_nodes(
                        batch,
                        float(mt_cfg.get("mask_ratio", 0.15)),
                        str(mt_cfg.get("mask_mode", "node")),
                        int(mt_cfg.get("max_subtree_nodes", 64)),
                        gen,
                    )
                    if not mask.any():
                        mask[0] = True
                    masked = batch.clone()
                    masked.type_id = torch.where(mask, torch.full_like(batch.type_id, mask_token), batch.type_id)
                v1 = v2 = None
                if w_ct > 0 and batch.num_graphs > 1:
                    fm, ed = float(ct_cfg.get("feature_mask", 0.2)), float(ct_cfg.get("edge_drop", 0.1))
                    v1 = augment(batch, mask_token, fm, ed, gen)
                    v2 = augment(batch, mask_token, fm, ed, gen)
                logits, z1, z2 = model(masked, v1, v2)
                if logits is not None:
                    target = batch.type_id[mask]
                    logits = logits[mask]
                    mt_loss = F.cross_entropy(logits, target)
                    loss = loss + w_mt * mt_loss
                    sums[1] += mt_loss.detach() * mask.sum()
                    sums[3] += mask.sum()
                    sums[4] += (logits.argmax(-1) == target).sum()
                if z1 is not None:
                    ct_loss = nt_xent(z1, z2, float(ct_cfg.get("temperature", 0.2)))
                    loss = loss + w_ct * ct_loss
                    sums[2] += ct_loss.detach() * batch.num_graphs
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip_norm", 1.0)))
                    optimizer.step()
            sums[0] += loss.detach() * batch.num_graphs
            sums[5] += batch.num_graphs
            sums[6] += batch.num_nodes
        if distributed:
            dist.all_reduce(sums)
        graphs = max(float(sums[5]), 1.0)
        return {
            "loss": float(sums[0]) / graphs,
            "masked_type_loss": float(sums[1]) / max(float(sums[3]), 1.0),
            "masked_type_acc": float(sums[4]) / max(float(sums[3]), 1.0),
            "contrastive_loss": float(sums[2]) / graphs,
            "graphs": int(sums[5]),
            "nodes": int(sums[6]),
            "steps_per_rank": n_steps,
        }

    history: list[dict[str, Any]] = []
    best_val = math.inf
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log(f"[pretrain] world={world} device={device} train_units={len(train_idx)} val_units={len(val_idx)} "
        f"budget_per_gpu={budget} objectives={objectives}")
    total_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        train_stats = run_epoch(epoch, train_idx, train=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_sec = time.perf_counter() - start
        val_stats = run_epoch(0, val_idx, train=False)
        row = {
            "epoch": epoch,
            "train_sec": round(train_sec, 3),
            "train_graphs_per_sec": round(train_stats["graphs"] / train_sec, 2),
            "train_nodes_per_sec": round(train_stats["nodes"] / train_sec, 1),
            "train": train_stats,
            "val": val_stats,
        }
        history.append(row)
        log(json.dumps(row))
        if rank == 0 and val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(out_dir / "encoder_best.pt", core, cfg, vocab, epoch, val_stats, mask_token, unk_token)

    total_sec = time.perf_counter() - total_start
    if rank == 0:
        save_checkpoint(out_dir / "encoder_last.pt", core, cfg, vocab, epochs, history[-1]["val"], mask_token, unk_token)
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        summary = {
            "world_size": world,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "budget_per_gpu": budget,
            "batch_scaling": str(cfg.get("batch_scaling", "weak")),
            "units_train": len(train_idx),
            "units_val": len(val_idx),
            "baselines": baselines,
            "best_val_loss": best_val,
            "total_sec": round(total_sec, 2),
            "peak_gpu_memory_mb_rank0": round(peak / 1024 ** 2, 1),
            "history": history,
        }
        (out_dir / "pretrain_metrics.json").write_text(json.dumps(summary, indent=2))
        log(f"[pretrain] done in {total_sec:.1f}s, best val loss {best_val:.4f}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def save_checkpoint(path: Path, core: PretrainModel, cfg: dict, vocab: dict, epoch: int, val: dict,
                    mask_token: int, unk_token: int) -> None:
    encoder_state = {
        k: v.detach().cpu() for k, v in core.encoder.state_dict().items()
        if k.startswith("type_emb.") or k.startswith("convs.")
    }
    torch.save(
        {
            "encoder_state": encoder_state,
            "architecture": str(cfg.get("architecture", "gcn")),
            "hidden_dim": int(cfg.get("hidden_dim", 512)),
            "num_layers": int(cfg.get("num_layers", 3)),
            "type_emb_dim": int(cfg.get("type_emb_dim", 128)),
            "node_type_vocab": vocab,
            "mask_token": mask_token,
            "unk_token": unk_token,
            "num_node_types": len(vocab) + 2,
            "epoch": epoch,
            "val": val,
        },
        path,
    )


if __name__ == "__main__":
    main()
