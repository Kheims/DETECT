"""Build a PyG dataset from pre-training corpus DOT files (no labels).

Reads DOT files produced by `build_ast_dot_from_normalized.py from-dir`,
constructs PyG Data objects with node features, edge index and type IDs.
No labels (y) or graph-level features (graph_x) — these are only needed
for the downstream MLCQ task.

Usage:
    python scripts/build_pretrain_dataset.py \
        --manifest-path artifacts/pretrain/construction/manifest.jsonl \
        --dataset-out artifacts/pretrain/dataset/dataset.pt \
        --node-type-vocab-out artifacts/pretrain/dataset/node_type_vocab.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse parsing utilities from the MLCQ dataset builder
from build_pyg_dataset_from_dot import (
    NodeRecord,
    build_features,
    parse_dot_file,
    _compute_tree_features,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PyG dataset from pre-training DOT files (no labels).",
    )
    parser.add_argument(
        "--manifest-path", type=Path, required=True,
        help="Path to from-dir manifest JSONL.",
    )
    parser.add_argument(
        "--dataset-out", type=Path,
        default=Path("artifacts/pretrain/dataset/dataset.pt"),
    )
    parser.add_argument(
        "--node-type-vocab-out", type=Path,
        default=Path("artifacts/pretrain/dataset/node_type_vocab.json"),
    )
    parser.add_argument(
        "--edge-types", type=str, default="Child,NextToken",
        help="Comma-separated edge types to keep.",
    )
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--max-nodes", type=int, default=15000,
                        help="Skip graphs with more nodes than this (avoid OOM).")
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def safe_int(value: object, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def main() -> None:
    args = parse_args()

    keep_edge_types = {
        t.strip() for t in args.edge_types.split(",") if t.strip()
    }

    rows: list[dict] = []
    with open(args.manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") != "success":
                continue
            rows.append(row)

    if args.max_graphs is not None:
        rows = rows[:args.max_graphs]

    print(f"[pretrain-dataset] {len(rows)} successful DOT files to process")

    dataset: list[Data] = []
    node_type_vocab: dict[str, int] = {}
    skipped_large = 0
    failed = 0
    start = time.perf_counter()

    pbar = tqdm(total=len(rows), desc="pretrain_dataset", unit="graph") if tqdm else None

    for processed, row in enumerate(rows, start=1):
        dot_path = Path(str(row["output_dot"]))

        if not dot_path.exists():
            failed += 1
            if pbar:
                pbar.update(1)
            continue

        try:
            nodes_raw, edges_raw = parse_dot_file(dot_path)
        except Exception:
            failed += 1
            if pbar:
                pbar.update(1)
            continue

        ordered_nodes = sorted(
            nodes_raw,
            key=lambda rec: safe_int(
                rec.attrs.get("node_index"),
                safe_int(rec.dot_id[1:], 0),
            ),
        )

        if len(ordered_nodes) > args.max_nodes:
            skipped_large += 1
            if pbar:
                pbar.update(1)
            continue

        x, node_types = build_features(ordered_nodes, edges_raw)

        type_ids = []
        for node_type in node_types:
            if node_type not in node_type_vocab:
                node_type_vocab[node_type] = len(node_type_vocab)
            type_ids.append(node_type_vocab[node_type])
        type_id_tensor = torch.tensor(type_ids, dtype=torch.int64)

        id_to_idx = {node.dot_id: idx for idx, node in enumerate(ordered_nodes)}
        edge_pairs: list[tuple[int, int]] = []
        for src_dot, dst_dot, edge_type in edges_raw:
            if edge_type not in keep_edge_types:
                continue
            if src_dot not in id_to_idx or dst_dot not in id_to_idx:
                continue
            edge_pairs.append((id_to_idx[src_dot], id_to_idx[dst_dot]))

        if edge_pairs:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        graph = Data(x=x, edge_index=edge_index, type_id=type_id_tensor)
        dataset.append(graph)

        if pbar:
            pbar.update(1)

        if processed % args.progress_every == 0 or processed == len(rows):
            elapsed = time.perf_counter() - start
            rate = processed / max(elapsed, 1e-9)
            print(
                f"[pretrain-dataset] {processed}/{len(rows)} "
                f"built={len(dataset)} skipped_large={skipped_large} "
                f"failed={failed} rate={rate:.1f} g/s"
            )

    if pbar:
        pbar.close()

    elapsed = time.perf_counter() - start

    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, args.dataset_out)

    args.node_type_vocab_out.parent.mkdir(parents=True, exist_ok=True)
    args.node_type_vocab_out.write_text(json.dumps(node_type_vocab, indent=2))

    summary = {
        "num_graphs": len(dataset),
        "skipped_large": skipped_large,
        "failed": failed,
        "num_node_types": len(node_type_vocab),
        "edge_types": sorted(keep_edge_types),
        "max_nodes_cap": args.max_nodes,
        "elapsed_sec": round(elapsed, 1),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
