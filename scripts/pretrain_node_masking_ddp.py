"""DDP-enabled self-supervised pre-training via node type masking.

Thin wrapper around the pre-training logic that adds Distributed Data Parallel
orchestration: process group init/cleanup, model wrapping, rank-gated I/O.

Launch via torchrun:
    torchrun --standalone --nproc_per_node=4 scripts/pretrain_node_masking_ddp.py \
        --dataset-path artifacts/pretrain/dataset/dataset.pt \
        --vocab-path artifacts/pretrain/dataset/node_type_vocab.json \
        --output-dir artifacts/pretrain/checkpoints \
        --architecture gcn --hidden-dim 512 --num-layers 3

Single-GPU fallback (no torchrun):
    python scripts/pretrain_node_masking_ddp.py --dataset-path ... --epochs 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.models import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DDP pre-train GNN encoder with node type masking.",
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pretrain/checkpoints"))
    parser.add_argument("--architecture", type=str, default="gcn", choices=["gcn", "gat", "graphsage"])
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--type-emb-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--aggregation", type=str, default="mean")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


class PretrainWrapper(nn.Module):
    """Wraps a GNN encoder with a node-level type prediction head."""

    def __init__(self, encoder: nn.Module, hidden_dim: int, num_node_types: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.predict_head = nn.Linear(hidden_dim, num_node_types)

    def forward(self, data: Data) -> torch.Tensor:
        x = self.encoder.compose_features(data)

        for i, (conv, bn) in enumerate(zip(self.encoder.convs, self.encoder.bns)):
            residual = (
                self.encoder.residual_proj(x)
                if i == 0 and self.encoder.residual_proj is not None
                else x
            )
            x = conv(x, data.edge_index)
            x = bn(x)
            x = x + residual
            x = F.relu(x)
            x = F.dropout(x, p=self.encoder.dropout, training=self.training)

        # Node-level prediction (no pooling)
        return self.predict_head(x)


def init_distributed(backend: str = "nccl") -> tuple[int, int, int, bool]:
    """Initialize DDP if launched via torchrun, otherwise single-GPU."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 0, 1, False

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    return rank, local_rank, world_size, True


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()

    rank, local_rank, world_size, is_distributed = init_distributed()
    is_main = rank == 0
    seed = 42
    set_seed(seed + rank)

    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main:
        print(f"[pretrain-ddp] distributed={is_distributed} world_size={world_size} device={device}")

    # ── Load dataset ──────────────────────────────────────────────────
    if is_main:
        print(f"[pretrain-ddp] loading dataset from {args.dataset_path}")
    dataset = torch.load(args.dataset_path, weights_only=False)
    if is_main:
        print(f"[pretrain-ddp] {len(dataset)} graphs loaded")

    vocab = json.loads(args.vocab_path.read_text())
    num_node_types = len(vocab)
    mask_token_id = num_node_types
    total_types = num_node_types + 1
    if is_main:
        print(f"[pretrain-ddp] {num_node_types} node types + 1 MASK = {total_types}")

    # Convert edges to undirected
    for data in dataset:
        data.edge_index = to_undirected(data.edge_index)

    num_numeric_feats = dataset[0].x.shape[1] if dataset[0].x is not None else 0

    # ── Build model ───────────────────────────────────────────────────
    arch_kwargs: dict = {}
    if args.architecture == "gat":
        arch_kwargs["num_heads"] = args.num_heads
    elif args.architecture == "graphsage":
        arch_kwargs["aggregation"] = args.aggregation

    encoder = get_model(
        args.architecture,
        num_node_types=total_types,
        type_emb_dim=args.type_emb_dim,
        num_numeric_feats=num_numeric_feats,
        num_token_feats=0,
        hidden_dim=args.hidden_dim,
        num_labels=4,  # placeholder, not used in pre-training
        dropout=args.dropout,
        num_layers=args.num_layers,
        num_graph_features=0,
        use_type_features=True,
        use_numeric_features=num_numeric_feats > 0,
        use_token_features=False,
        **arch_kwargs,
    )

    model = PretrainWrapper(encoder, args.hidden_dim, total_types).to(device)

    if is_distributed:
        # find_unused_parameters=True because the encoder's graph-level pooling
        # and classifier head are not used in pre-training (node-level only).
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    raw_model = model.module if is_distributed else model

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # ── DataLoader with DistributedSampler ────────────────────────────
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
    ) if is_distributed else None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[pretrain-ddp] arch={args.architecture} hidden={args.hidden_dim} "
              f"layers={args.num_layers} mask_ratio={args.mask_ratio}")
        print(f"[pretrain-ddp] epochs={args.epochs} lr={args.lr} "
              f"batch_size={args.batch_size} (per-GPU)")
        print(f"[pretrain-ddp] global_batch_size={args.batch_size * world_size}")
        print(f"[pretrain-ddp] starting training")

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        total_masked = 0
        total_correct = 0
        num_batches = 0
        epoch_start = time.perf_counter()

        for batch in loader:
            batch = batch.to(device)

            # Mask random node types
            mask = torch.rand(batch.type_id.shape, device=device) < args.mask_ratio
            original_type_ids = batch.type_id.clone()
            batch.type_id = batch.type_id.clone()
            batch.type_id[mask] = mask_token_id

            logits = model(batch)

            # Loss only on masked nodes
            if mask.sum() == 0:
                continue
            loss = F.cross_entropy(logits[mask], original_type_ids[mask])

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch_loss = loss.item()
            if not (batch_loss != batch_loss):  # skip NaN
                total_loss += batch_loss * mask.sum().item()
            else:
                total_loss += 0.0  # count masked tokens but don't add NaN loss
            total_masked += mask.sum().item()
            preds = logits[mask].argmax(dim=-1)
            total_correct += (preds == original_type_ids[mask]).sum().item()
            num_batches += 1

        # Aggregate metrics across ranks
        if is_distributed:
            metrics = torch.tensor(
                [total_loss, total_masked, total_correct, num_batches],
                device=device, dtype=torch.float64,
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss = metrics[0].item()
            total_masked = int(metrics[1].item())
            total_correct = int(metrics[2].item())
            num_batches = int(metrics[3].item())

        avg_loss = total_loss / max(total_masked, 1)
        accuracy = total_correct / max(total_masked, 1)
        epoch_time = time.perf_counter() - epoch_start

        if is_main:
            print(
                f"Epoch {epoch:03d} | loss={avg_loss:.4f} | "
                f"acc={accuracy:.4f} | time={epoch_time:.1f}s | "
                f"batches={num_batches}"
            )

        # Checkpoint (rank 0 only)
        if is_main and (epoch % args.save_every == 0 or epoch == args.epochs):
            ckpt_path = args.output_dir / f"pretrained_{args.architecture}_ep{epoch}.pt"
            torch.save({
                "encoder_state_dict": raw_model.encoder.state_dict(),
                "architecture": args.architecture,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "type_emb_dim": args.type_emb_dim,
                "num_node_types": num_node_types,
                "total_types_with_mask": total_types,
                "node_type_vocab": vocab,
                "epoch": epoch,
                "loss": avg_loss,
                "accuracy": accuracy,
                "distributed": {
                    "world_size": world_size,
                    "global_batch_size": args.batch_size * world_size,
                },
            }, ckpt_path)
            print(f"  saved: {ckpt_path}")

    if is_main:
        print("[pretrain-ddp] done")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
