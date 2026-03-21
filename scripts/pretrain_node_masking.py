"""Self-supervised pre-training via node type masking on AST graphs.

Masks a fraction of node type_ids and trains the GNN encoder to predict the
original types from the graph context. The learned encoder weights transfer
to the downstream MLCQ multi-label classification task.

Usage:
    python scripts/pretrain_node_masking.py \
        --dataset-path artifacts/pretrain/dataset/dataset.pt \
        --vocab-path artifacts/pretrain/dataset/node_type_vocab.json \
        --output-dir artifacts/pretrain/checkpoints \
        --architecture gcn --hidden-dim 512 --num-layers 3 \
        --epochs 50 --lr 0.001 --mask-ratio 0.15 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.models import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-train GNN encoder with node type masking.",
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
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--aggregation", type=str, default="mean")
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


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    print(f"[pretrain] loading dataset from {args.dataset_path}")
    dataset = torch.load(args.dataset_path, weights_only=False)
    print(f"[pretrain] {len(dataset)} graphs loaded")

    vocab = json.loads(args.vocab_path.read_text())
    num_node_types = len(vocab)
    # Reserve one extra ID for the MASK token
    mask_token_id = num_node_types
    total_types = num_node_types + 1
    print(f"[pretrain] {num_node_types} node types + 1 MASK token = {total_types}")

    # Convert edges to undirected
    for data in dataset:
        data.edge_index = to_undirected(data.edge_index)

    # Determine numeric feature count from first graph
    num_numeric_feats = dataset[0].x.shape[1] if dataset[0].x is not None else 0

    # Build encoder (same architecture as downstream, but with expanded type vocab for MASK token)
    arch_kwargs: dict = {}
    if args.architecture == "gat":
        arch_kwargs["num_heads"] = args.num_heads
    elif args.architecture == "graphsage":
        arch_kwargs["aggregation"] = args.aggregation

    encoder = get_model(
        args.architecture,
        num_node_types=total_types,  # +1 for MASK token
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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pretrain] architecture={args.architecture} hidden={args.hidden_dim} "
          f"layers={args.num_layers} mask_ratio={args.mask_ratio}")
    print(f"[pretrain] device={device} epochs={args.epochs} lr={args.lr} "
          f"batch_size={args.batch_size}")
    print(f"[pretrain] starting training")

    for epoch in range(1, args.epochs + 1):
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
            loss = F.cross_entropy(logits[mask], original_type_ids[mask])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * mask.sum().item()
            total_masked += mask.sum().item()
            preds = logits[mask].argmax(dim=-1)
            total_correct += (preds == original_type_ids[mask]).sum().item()
            num_batches += 1

        avg_loss = total_loss / max(total_masked, 1)
        accuracy = total_correct / max(total_masked, 1)
        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch:03d} | loss={avg_loss:.4f} | "
            f"acc={accuracy:.4f} | time={epoch_time:.1f}s"
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = args.output_dir / f"pretrained_{args.architecture}_ep{epoch}.pt"
            # Save only encoder weights (not the prediction head)
            torch.save({
                "encoder_state_dict": encoder.state_dict(),
                "architecture": args.architecture,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "type_emb_dim": args.type_emb_dim,
                "num_node_types": num_node_types,  # original count without MASK
                "total_types_with_mask": total_types,
                "node_type_vocab": vocab,
                "epoch": epoch,
                "loss": avg_loss,
                "accuracy": accuracy,
            }, ckpt_path)
            print(f"  saved: {ckpt_path}")

    print("[pretrain] done")


if __name__ == "__main__":
    main()
