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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_pyg_dataset_from_dot import (
    build_features,
    parse_dot_file,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PyG dataset from pre-training DOT files (no labels).",
    )
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument(
        "--dataset-out", type=Path,
        default=Path("artifacts/pretrain/dataset/dataset.pt"),
    )
    parser.add_argument(
        "--node-type-vocab-out", type=Path,
        default=Path("artifacts/pretrain/dataset/node_type_vocab.json"),
    )
    parser.add_argument(
        "--canonical-vocab", type=Path, default=None,
        help="Path to canonical vocab JSON for static type_id mapping.",
    )
    parser.add_argument("--edge-types", type=str, default="Child,NextToken")
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=8)
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


def _process_one_dot(task: dict[str, Any]) -> dict[str, Any]:
    """Worker function: parse DOT, build features, return serializable result."""
    dot_path = Path(task["dot_path"])
    max_nodes = task["max_nodes"]
    keep_edge_types = set(task["keep_edge_types"])

    if not dot_path.exists():
        return {"status": "failed", "reason": "missing"}

    try:
        nodes_raw, edges_raw = parse_dot_file(dot_path)
    except Exception as e:
        return {"status": "failed", "reason": str(e)}

    ordered_nodes = sorted(
        nodes_raw,
        key=lambda rec: safe_int(
            rec.attrs.get("node_index"),
            safe_int(rec.dot_id[1:], 0),
        ),
    )

    if len(ordered_nodes) > max_nodes:
        return {"status": "skipped_large", "num_nodes": len(ordered_nodes)}

    x, node_types = build_features(ordered_nodes, edges_raw)

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

    return {
        "status": "success",
        "x": x,
        "edge_index": edge_index,
        "node_types": node_types,
    }


def main() -> None:
    args = parse_args()
    keep_edge_types = [t.strip() for t in args.edge_types.split(",") if t.strip()]

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

    print(f"[pretrain-dataset] {len(rows)} DOT files, {args.workers} workers, max_nodes={args.max_nodes}")

    tasks = [
        {"dot_path": row["output_dot"], "max_nodes": args.max_nodes, "keep_edge_types": keep_edge_types}
        for row in rows
    ]

    dataset: list[Data] = []
    if args.canonical_vocab is not None and args.canonical_vocab.exists():
        node_type_vocab: dict[str, int] = json.loads(args.canonical_vocab.read_text())
        print(f"[pretrain-dataset] loaded canonical vocab: {len(node_type_vocab)} types")
    else:
        node_type_vocab: dict[str, int] = {}
    success = 0
    skipped_large = 0
    failed = 0
    start = time.perf_counter()

    pbar = tqdm(total=len(tasks), desc="pretrain_dataset", unit="graph") if tqdm else None

    if args.workers <= 1:
        results_iter = (_process_one_dot(t) for t in tasks)
        for result in results_iter:
            _handle_result(result, dataset, node_type_vocab)
            st = result["status"]
            if st == "success":
                success += 1
            elif st == "skipped_large":
                skipped_large += 1
            else:
                failed += 1
            if pbar:
                pbar.set_postfix(ok=success, fail=failed, skip=skipped_large)
                pbar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process_one_dot, t): t for t in tasks}
            for future in as_completed(futures):
                result = future.result()
                st = result["status"]
                if st == "success":
                    _handle_result(result, dataset, node_type_vocab)
                    success += 1
                elif st == "skipped_large":
                    skipped_large += 1
                else:
                    failed += 1
                if pbar:
                    pbar.set_postfix(ok=success, fail=failed, skip=skipped_large)
                    pbar.update(1)

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
        "edge_types": keep_edge_types,
        "max_nodes_cap": args.max_nodes,
        "elapsed_sec": round(elapsed, 1),
    }
    print(json.dumps(summary, indent=2))


def _handle_result(
    result: dict[str, Any],
    dataset: list[Data],
    node_type_vocab: dict[str, int],
) -> None:
    """Build type_id tensor and append graph to dataset (parent process only)."""
    node_types = result["node_types"]
    type_ids = []
    for nt in node_types:
        if nt not in node_type_vocab:
            node_type_vocab[nt] = len(node_type_vocab)
        type_ids.append(node_type_vocab[nt])

    graph = Data(
        x=result["x"],
        edge_index=result["edge_index"],
        type_id=torch.tensor(type_ids, dtype=torch.int64),
    )
    dataset.append(graph)


if __name__ == "__main__":
    main()
