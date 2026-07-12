"""Measure the per-snippet latency of DETECT (parse, graph, tokens, inference).

Also reports the cold start and the batched inference throughput. --parse-only
skips the model.

Usage:
    python scripts/analysis/bench_inference.py --run-dir artifacts/runs/<fp> \\
        --node-type-vocab .../node_type_vocab.json --token-vectors .../word2vec.kv \\
        --normalized data/MLCQCodeSmellSamples.normalized.json --parser antlr \\
        --samples 300 --device cpu --out-json artifacts/latency/antlr_cpu.json
"""
from __future__ import annotations

import time

_T0 = time.perf_counter()

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from torch_geometric.data import Batch, Data  # noqa: E402
from torch_geometric.utils import to_undirected  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_ast_dot_from_normalized as builder  # noqa: E402
from scripts import build_pyg_dataset_from_dot as pyg_builder  # noqa: E402
from mlcq_graphs.models import get_model  # noqa: E402

IMPORT_SEC = time.perf_counter() - _T0


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(p: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))], 3)

    return {"n": len(ordered), "mean": round(statistics.mean(ordered), 3), "p50": at(0.5),
            "p90": at(0.9), "p95": at(0.95), "p99": at(0.99), "max": round(ordered[-1], 3)}


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_graph(dot_text: str, vocab: dict[str, int], edge_types: set[str], tmp_dir: Path) -> tuple[Data, list, int]:
    path = tmp_dir / "snippet.dot"
    path.write_text(dot_text)
    nodes_raw, edges_raw = pyg_builder.parse_dot_file(path)
    ordered = sorted(
        nodes_raw,
        key=lambda rec: pyg_builder.safe_int(rec.attrs.get("node_index"), pyg_builder.safe_int(rec.dot_id[1:], 0)),
    )
    x, node_types = pyg_builder.build_features(ordered)
    unknown = sum(1 for t in node_types if t not in vocab)
    # unseen node types fall back to id 0 at inference time (counted and reported)
    type_id = torch.tensor([vocab.get(t, 0) for t in node_types], dtype=torch.long)
    id_to_idx = {node.dot_id: i for i, node in enumerate(ordered)}
    pairs = [(id_to_idx[s], id_to_idx[d]) for s, d, k in edges_raw if k in edge_types and s in id_to_idx and d in id_to_idx]
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs else torch.empty((2, 0), dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=len(ordered))
    return Data(x=x, edge_index=edge_index, type_id=type_id, num_nodes=len(ordered)), ordered, unknown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--node-type-vocab", type=Path, default=None)
    parser.add_argument("--token-vectors", type=Path, default=None)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--parser", choices=["antlr", "tree_sitter"], default="antlr")
    parser.add_argument("--edge-types", default="Child,NextToken")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    entries = json.loads(args.normalized.read_text())
    rng = random.Random(args.seed)
    picked = rng.sample(range(len(entries)), min(args.samples, len(entries)))
    edge_types = {e.strip() for e in args.edge_types.split(",") if e.strip()}

    cold: dict[str, float] = {"imports_sec": round(IMPORT_SEC, 3)}
    model = vocab = vectors = None
    if not args.parse_only:
        t = time.perf_counter()
        ckpt = torch.load(args.run_dir / "best_model.pt", map_location="cpu", weights_only=False)
        snap = json.loads((args.run_dir / "config.json").read_text())
        vocab = json.loads(args.node_type_vocab.read_text())
        kwargs = {"pooling": snap.get("pooling", "mean"), "set2set_steps": snap.get("set2set_steps", 3)}
        if snap["architecture"] == "gat":
            kwargs["num_heads"] = snap.get("num_heads", 4)
        if snap["architecture"] == "graphsage":
            kwargs["aggregation"] = snap.get("aggregation", "mean")
        state = ckpt["model_state_dict"]
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
        model.to(device).eval()
        sync(device)
        cold["model_load_sec"] = round(time.perf_counter() - t, 3)
        if model.use_token_features:
            from gensim.models import KeyedVectors

            t = time.perf_counter()
            vectors = KeyedVectors.load(str(args.token_vectors), mmap="r")
            cold["token_vectors_load_sec"] = round(time.perf_counter() - t, 3)

    timings: dict[str, list[float]] = {k: [] for k in ("parse", "graph", "tokens", "inference", "total")}
    per_sample: list[dict] = []
    graphs: list[Data] = []
    failures = 0
    unknown_types = 0
    task = {"parser": args.parser, "allow_errors": False}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with torch.inference_mode():
            for idx in picked:
                entry = entries[idx]
                source = builder._wrap_snippet_in_class(str(entry.get("code_snippet", "")))
                t0 = time.perf_counter()
                dot_text, stats, error = builder._build_graph(source, "bench", task)
                t1 = time.perf_counter()
                row = {"json_index": idx, "lines": source.count("\n"), "parse_ms": (t1 - t0) * 1000}
                if dot_text is None:
                    failures += 1
                    per_sample.append({**row, "failed": True})
                    continue
                row["nodes"] = stats.node_count
                row["tokens"] = stats.syntax_token_count
                timings["parse"].append(row["parse_ms"])
                if args.parse_only:
                    timings["total"].append(row["parse_ms"])
                    per_sample.append(row)
                    continue
                data, ordered, unknown = build_graph(dot_text, vocab, edge_types, tmp_dir)
                unknown_types += unknown
                t2 = time.perf_counter()
                if vectors is not None:
                    token_x = torch.zeros((data.num_nodes, vectors.vector_size), dtype=torch.float32)
                    for i, node in enumerate(ordered):
                        if str(node.attrs.get("node_kind")) == "SyntaxToken":
                            text = str(node.attrs.get("token_text", ""))
                            if text in vectors.key_to_index:
                                token_x[i] = torch.from_numpy(vectors[text].copy())
                    data.token_x = token_x
                t3 = time.perf_counter()
                batch = Batch.from_data_list([data]).to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits).cpu()
                sync(device)
                t4 = time.perf_counter()
                row.update({"graph_ms": (t2 - t1) * 1000, "tokens_ms": (t3 - t2) * 1000,
                            "inference_ms": (t4 - t3) * 1000, "total_ms": (t4 - t0) * 1000,
                            "probs": [round(float(p), 4) for p in probs.view(-1)]})
                for key in ("graph", "tokens", "inference", "total"):
                    timings[key].append(row[f"{key}_ms"])
                per_sample.append(row)
                graphs.append(data)

            throughput = None
            if graphs and model is not None:
                # inference only, batches of --batch-size, after one warm-up batch
                chunks = [graphs[i:i + args.batch_size] for i in range(0, len(graphs), args.batch_size)]
                model(Batch.from_data_list(chunks[0]).to(device))
                sync(device)
                t = time.perf_counter()
                for chunk in chunks:
                    model(Batch.from_data_list(chunk).to(device))
                sync(device)
                throughput = round(len(graphs) / (time.perf_counter() - t), 1)

    by_size: dict[str, dict] = {}
    buckets = [(0, 50), (50, 200), (200, 1000), (1000, 10 ** 9)]
    for lo, hi in buckets:
        rows = [r for r in per_sample if not r.get("failed") and lo <= r["tokens"] < hi]
        by_size[f"tokens_{lo}_{hi if hi < 10 ** 9 else 'inf'}"] = {
            "n": len(rows),
            "parse_ms": quantiles([r["parse_ms"] for r in rows]),
            "total_ms": quantiles([r.get("total_ms", r["parse_ms"]) for r in rows]),
        }

    report = {
        "parser": args.parser,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "samples": len(picked),
        "parse_failures": failures,
        "unknown_node_types": unknown_types,
        "cold_start": cold,
        "per_snippet_ms": {k: quantiles(v) for k, v in timings.items() if v},
        "by_token_count": by_size,
        "batched_inference_graphs_per_sec": throughput,
        "batch_size": args.batch_size,
        "per_sample": per_sample,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_sample"}, indent=2))


if __name__ == "__main__":
    main()
