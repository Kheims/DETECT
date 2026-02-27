from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import resource

import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.constants import LABEL_ORDER

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


@dataclass
class NodeRecord:
    dot_id: str
    attrs: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PyG dataset.pt from generated AST DOT files and manifest.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("artifacts/cache/construction/latest/manifest.jsonl"),
        help="Path to DOT generation manifest.",
    )
    parser.add_argument(
        "--dot-dir",
        type=Path,
        default=Path("artifacts/cache/construction/latest/dot"),
        help="DOT directory used only when fallback is explicitly allowed.",
    )
    parser.add_argument(
        "--dataset-out",
        type=Path,
        default=Path("artifacts/cache/dataset/latest/dataset.pt"),
        help="Output path for list[Data] file.",
    )
    parser.add_argument(
        "--node-type-vocab-out",
        type=Path,
        default=Path("artifacts/cache/dataset/latest/node_type_vocab.json"),
        help="Output path for node-type vocabulary JSON.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=Path("artifacts/cache/dataset/latest/dataset.meta.json"),
        help="Output path for dataset metadata.",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="Optional cap on number of success manifest rows.",
    )
    parser.add_argument(
        "--edge-types",
        type=str,
        default="Child,NextToken",
        help="Comma-separated edge types to keep.",
    )
    parser.add_argument(
        "--progress-style",
        choices=["auto", "print", "tqdm", "none"],
        default="auto",
        help="Progress display style.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Emit progress every N processed rows.",
    )
    parser.add_argument(
        "--memory-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include process memory in progress logs.",
    )
    parser.add_argument(
        "--allow-dot-dir-fallback",
        action="store_true",
        help="Allow dot-dir scan when manifest is missing or empty.",
    )
    return parser.parse_args()


def choose_progress_style(style: str) -> str:
    if style == "none":
        return "none"
    if style == "print":
        return "print"
    if style == "tqdm":
        if tqdm is None:
            print("[build_pyg] tqdm unavailable; falling back to print progress.")
            return "print"
        return "tqdm"

    # auto
    if tqdm is not None and sys.stderr.isatty():
        return "tqdm"
    return "print"


def get_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = float(usage.ru_maxrss)
    # Linux reports KB; macOS reports bytes.
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def format_eta(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def progress_payload(
    processed: int,
    total: int,
    built: int,
    missing: int,
    start_time: float,
    memory_stats: bool,
) -> tuple[dict[str, str], float, float]:
    elapsed = max(1e-9, time.perf_counter() - start_time)
    rate = processed / elapsed
    remaining = max(0, total - processed)
    eta = remaining / rate if rate > 0 else 0.0

    payload = {
        "processed": f"{processed}/{total}",
        "built": str(built),
        "missing": str(missing),
        "rate": f"{rate:.2f} g/s",
        "eta": format_eta(eta),
        "elapsed": format_eta(elapsed),
    }
    if memory_stats:
        payload["rss_mb"] = f"{get_rss_mb():.1f}"
    return payload, elapsed, rate


def parse_dot_value(raw: str) -> object:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(raw)
    except ValueError:
        pass

    try:
        return float(raw)
    except ValueError:
        pass

    return raw


def parse_attr_tokens(attr_str: str) -> dict[str, object]:
    attrs: dict[str, object] = {}
    for token in shlex.split(attr_str, posix=True):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        attrs[key] = parse_dot_value(value)
    return attrs


def parse_dot_file(dot_path: Path) -> tuple[list[NodeRecord], list[tuple[str, str, str]]]:
    nodes: list[NodeRecord] = []
    edges: list[tuple[str, str, str]] = []

    for raw_line in dot_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith('"n') and '" -> "' not in line and "[" in line and "]" in line:
            left = line.index("[")
            right = line.rindex("]")
            dot_id = line[1 : line.index('"', 1)]
            attrs = parse_attr_tokens(line[left + 1 : right])
            nodes.append(NodeRecord(dot_id=dot_id, attrs=attrs))
            continue

        if line.startswith('"n') and '" -> "' in line and "[" in line and "]" in line:
            arrow = line.index("->")
            src = line[1 : line.index('"', 1)]

            dst_start = line.index('"', arrow)
            dst_end = line.index('"', dst_start + 1)
            dst = line[dst_start + 1 : dst_end]

            left = line.index("[")
            right = line.rindex("]")
            attrs = parse_attr_tokens(line[left + 1 : right])
            edge_type = str(attrs.get("edge_type", ""))
            edges.append((src, dst, edge_type))

    if not nodes:
        raise ValueError(f"No nodes parsed from {dot_path}")
    return nodes, edges


def safe_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
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


def build_features(
    ordered_nodes: list[NodeRecord],
) -> tuple[torch.Tensor, list[str]]:
    node_types: list[str] = []

    end_lines = [safe_int(node.attrs.get("end_line"), 0) for node in ordered_nodes]
    end_cols = [safe_int(node.attrs.get("end_col"), 0) for node in ordered_nodes]
    max_line = max(1, max(end_lines) if end_lines else 1)
    max_col = max(1, max(end_cols) if end_cols else 1)

    token_lengths = []
    for node in ordered_nodes:
        if str(node.attrs.get("node_kind", "")) == "SyntaxToken":
            token_text = str(node.attrs.get("token_text", ""))
            token_lengths.append(len(token_text))
    max_token_len = max(1, max(token_lengths) if token_lengths else 1)

    rows: list[list[float]] = []
    for node in ordered_nodes:
        node_kind = str(node.attrs.get("node_kind", "SyntaxNode"))
        node_type = str(node.attrs.get("node_type", "unknown"))
        node_types.append(node_type)

        is_token = 1.0 if node_kind == "SyntaxToken" else 0.0

        start_line = float(safe_int(node.attrs.get("start_line"), 0))
        start_col = float(safe_int(node.attrs.get("start_col"), 0))
        end_line = float(safe_int(node.attrs.get("end_line"), int(start_line)))

        span_lines = max(0.0, end_line - start_line) + 1.0
        token_len = 0.0
        if is_token > 0:
            token_len = float(len(str(node.attrs.get("token_text", ""))))

        rows.append(
            [
                is_token,
                start_line / max_line,
                start_col / max_col,
                span_lines / max_line,
                token_len / max_token_len,
            ]
        )

    x = torch.tensor(rows, dtype=torch.float32)
    return x, node_types


def parse_bool_token(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def parse_y_from_filename(filename: str) -> list[bool] | None:
    match = re.search(r"_\[(true|false),(true|false),(true|false),(true|false)\]_", filename)
    if match is None:
        return None

    values = [parse_bool_token(group) for group in match.groups()]
    if any(value is None for value in values):
        return None
    return [bool(value) for value in values]


def parse_json_index_from_filename(filename: str) -> int | None:
    match = re.match(r"^(\d+)_", filename)
    if match is None:
        return None
    return int(match.group(1))


def parse_y_from_dot_comment(dot_path: Path) -> list[bool] | None:
    for line in dot_path.read_text().splitlines():
        if "graph [comment=" not in line or " y=[" not in line:
            continue
        start = line.find(" y=[")
        if start < 0:
            continue
        end = line.find("]", start)
        if end < 0:
            continue
        payload = line[start + len(" y=[") : end]
        parts = [part.strip() for part in payload.split(",")]
        if len(parts) != 4:
            continue

        bools: list[bool] = []
        for part in parts:
            parsed = parse_bool_token(part)
            if parsed is None:
                bools = []
                break
            bools.append(parsed)
        if len(bools) == 4:
            return bools
    return None


def load_rows_from_manifest(manifest_path: Path) -> list[dict[str, object]]:
    if not manifest_path.exists():
        return []

    rows: list[dict[str, object]] = []
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") in {"success", "skipped"}:
            rows.append(row)

    rows.sort(key=lambda item: safe_int(item.get("json_index"), -1))
    return rows


def load_rows_from_dot_dir(dot_dir: Path) -> list[dict[str, object]]:
    if not dot_dir.exists():
        raise FileNotFoundError(f"DOT directory not found: {dot_dir}")

    rows: list[dict[str, object]] = []
    for dot_path in sorted(dot_dir.glob("*.dot")):
        json_index = parse_json_index_from_filename(dot_path.name)
        y = parse_y_from_filename(dot_path.name)
        if y is None:
            y = parse_y_from_dot_comment(dot_path)
        if json_index is None or y is None:
            continue

        rows.append(
            {
                "json_index": json_index,
                "output_dot": str(dot_path),
                "y": y,
                "status": "success",
            }
        )

    rows.sort(key=lambda item: safe_int(item.get("json_index"), -1))
    return rows


def main() -> None:
    args = parse_args()
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be > 0")

    progress_style = choose_progress_style(args.progress_style)
    keep_edge_types = {
        edge_type.strip() for edge_type in args.edge_types.split(",") if edge_type.strip()
    }
    if not keep_edge_types:
        raise ValueError("--edge-types must include at least one edge type.")

    rows = load_rows_from_manifest(args.manifest_path)
    source_mode = "manifest"
    if not rows:
        if not args.allow_dot_dir_fallback:
            raise ValueError(
                "Manifest is missing or empty. "
                "Re-run construction or pass --allow-dot-dir-fallback explicitly."
            )
        rows = load_rows_from_dot_dir(args.dot_dir)
        source_mode = "dot_dir"

    if args.max_graphs is not None:
        rows = rows[: args.max_graphs]

    if not rows:
        raise ValueError(
            "No successful rows were found from manifest or dot directory."
        )

    total_rows = len(rows)
    start_time = time.perf_counter()
    dataset: list[Data] = []
    node_type_vocab: dict[str, int] = {}
    missing_dot = 0

    pbar = None
    if progress_style == "tqdm" and tqdm is not None:
        pbar = tqdm(total=total_rows, desc="build_pyg_dataset", unit="graph")

    for processed, row in enumerate(rows, start=1):
        dot_path = Path(str(row["output_dot"]))
        if not dot_path.is_absolute():
            dot_path = Path.cwd() / dot_path

        if not dot_path.exists():
            missing_dot += 1
        else:
            nodes_raw, edges_raw = parse_dot_file(dot_path)
            ordered_nodes = sorted(
                nodes_raw,
                key=lambda rec: safe_int(rec.attrs.get("node_index"), safe_int(rec.dot_id[1:], 0)),
            )

            x, node_types = build_features(ordered_nodes)

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

            y_raw_obj = row.get("y", [])
            y_raw = y_raw_obj if isinstance(y_raw_obj, list) else []
            if len(y_raw) != len(LABEL_ORDER):
                raise ValueError(
                    f"Unexpected y length for json_index={row.get('json_index')}: {len(y_raw)}"
                )
            y = torch.tensor([1 if bool(v) else 0 for v in y_raw], dtype=torch.int64)

            graph = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                type_id=type_id_tensor,
            )
            graph.json_index = safe_int(row.get("json_index"), -1)
            dataset.append(graph)

        if pbar is not None:
            pbar.update(1)

        should_log = (
            processed % args.progress_every == 0
            or processed == total_rows
            or processed == 1
        )
        if should_log:
            payload, _, _ = progress_payload(
                processed=processed,
                total=total_rows,
                built=len(dataset),
                missing=missing_dot,
                start_time=start_time,
                memory_stats=args.memory_stats,
            )
            if pbar is not None:
                pbar.set_postfix(payload)
                pbar.refresh()
            elif progress_style == "print":
                print(
                    "[build_pyg] "
                    + " | ".join(f"{key}={value}" for key, value in payload.items())
                )

    if pbar is not None:
        pbar.close()

    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, args.dataset_out)

    args.node_type_vocab_out.parent.mkdir(parents=True, exist_ok=True)
    args.node_type_vocab_out.write_text(json.dumps(node_type_vocab, indent=2))

    counts = [0, 0, 0, 0]
    for graph in dataset:
        y_tensor = torch.as_tensor(graph.y).view(-1)
        for i in range(len(LABEL_ORDER)):
            counts[i] += safe_int(y_tensor[i].item(), 0)

    elapsed_total = max(1e-9, time.perf_counter() - start_time)
    throughput = len(dataset) / elapsed_total
    metadata = {
        "source_mode": source_mode,
        "source_manifest": str(args.manifest_path),
        "source_dot_dir": str(args.dot_dir),
        "allow_dot_dir_fallback": bool(args.allow_dot_dir_fallback),
        "dataset_path": str(args.dataset_out),
        "node_type_vocab_path": str(args.node_type_vocab_out),
        "label_order": LABEL_ORDER,
        "kept_edge_types": sorted(keep_edge_types),
        "num_graphs": len(dataset),
        "missing_dot_files": missing_dot,
        "elapsed_sec": round(elapsed_total, 3),
        "throughput_graphs_per_sec": round(throughput, 3),
        "label_counts": {LABEL_ORDER[i]: int(counts[i]) for i in range(len(LABEL_ORDER))},
    }
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.write_text(json.dumps(metadata, indent=2))

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
