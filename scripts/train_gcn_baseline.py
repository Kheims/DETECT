from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import precision_recall_curve, roc_curve
from torch import nn
from torch.utils.data import Sampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.constants import LABEL_ORDER
from mlcq_graphs.evaluation import evaluate_full
from mlcq_graphs.models import get_model
from mlcq_graphs.training import Trainer, TrainResult, build_loss_fn


SMELL_LABELS = LABEL_ORDER


@dataclass
class DatasetSummary:
    num_graphs: int
    label_counts: dict[str, int]
    zero_label_graphs: int
    node_stats: dict[str, float | int]
    edge_stats: dict[str, float | int]


class DynamicBudgetBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        node_counts: list[int],
        edge_counts: list[int],
        max_nodes: int,
        max_edges: int | None,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        if max_nodes <= 0:
            raise ValueError("max_nodes must be > 0")

        self.node_counts = node_counts
        self.edge_counts = edge_counts
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _ordered_indices(self) -> list[int]:
        indices = list(range(len(self.node_counts)))
        if not self.shuffle:
            return indices
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(len(indices), generator=generator).tolist()
        return [indices[i] for i in perm]

    def __iter__(self):
        batch: list[int] = []
        batch_nodes = 0
        batch_edges = 0

        for idx in self._ordered_indices():
            n_nodes = int(self.node_counts[idx])
            n_edges = int(self.edge_counts[idx])

            if n_nodes <= 0:
                continue

            exceeds_nodes = n_nodes > self.max_nodes
            exceeds_edges = self.max_edges is not None and n_edges > self.max_edges

            if exceeds_nodes or exceeds_edges:
                if batch and not self.drop_last:
                    yield batch
                if not self.drop_last:
                    yield [idx]
                batch = []
                batch_nodes = 0
                batch_edges = 0
                continue

            would_exceed_nodes = batch_nodes + n_nodes > self.max_nodes
            would_exceed_edges = (
                self.max_edges is not None and batch_edges + n_edges > self.max_edges
            )

            if batch and (would_exceed_nodes or would_exceed_edges):
                if not self.drop_last:
                    yield batch
                batch = [idx]
                batch_nodes = n_nodes
                batch_edges = n_edges
                continue

            batch.append(idx)
            batch_nodes += n_nodes
            batch_edges += n_edges

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if len(self.node_counts) == 0:
            return 0

        total_nodes = sum(max(1, n) for n in self.node_counts)
        estimate = math.ceil(total_nodes / self.max_nodes)
        return max(1, estimate)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quantile_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {
            "min": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            "mean": 0.0,
        }

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def at(p: float) -> int:
        idx = min(n - 1, max(0, int(round((n - 1) * p))))
        return sorted_vals[idx]

    return {
        "min": sorted_vals[0],
        "p50": at(0.50),
        "p90": at(0.90),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": sorted_vals[-1],
        "mean": round(sum(sorted_vals) / n, 2),
    }


def summarize_dataset(dataset: list[Data], smell_labels: list[str]) -> DatasetSummary:
    y = torch.stack([d.y.view(-1) for d in dataset]).int()
    counts = y.sum(dim=0).tolist()
    label_counts = {
        smell_labels[i] if i < len(smell_labels) else f"label_{i}": int(count)
        for i, count in enumerate(counts)
    }
    zero_label_graphs = int((y.sum(dim=1) == 0).sum().item())
    node_stats = quantile_stats([int(d.num_nodes) for d in dataset])
    edge_stats = quantile_stats([int(d.edge_index.size(1)) for d in dataset])
    return DatasetSummary(
        num_graphs=len(dataset),
        label_counts=label_counts,
        zero_label_graphs=zero_label_graphs,
        node_stats=node_stats,
        edge_stats=edge_stats,
    )


def validate_graph(data: Data, num_labels: int) -> None:
    if not hasattr(data, "x") or data.x is None:
        raise ValueError("Each graph must contain x features.")
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError("Each graph must contain edge_index.")
    if not hasattr(data, "y") or data.y is None:
        raise ValueError("Each graph must contain y labels.")
    if not hasattr(data, "type_id") or data.type_id is None:
        raise ValueError("Each graph must contain type_id.")

    if data.y.view(-1).numel() != num_labels:
        raise ValueError(
            f"Expected y to have {num_labels} labels, got {data.y.view(-1).numel()}"
        )


def load_dataset(
    dataset_path: Path,
    num_labels: int,
    max_graphs: int | None,
    to_undirected_edges: bool,
) -> list[Data]:
    dataset_obj = torch.load(dataset_path, weights_only=False)
    if not isinstance(dataset_obj, list):
        raise TypeError("Expected dataset file to contain a list[Data].")

    if max_graphs is not None:
        dataset_obj = dataset_obj[:max_graphs]

    dataset: list[Data] = []
    for data in dataset_obj:
        validate_graph(data, num_labels)
        data.y = data.y.view(-1).int()
        data.type_id = data.type_id.view(-1).long()
        data.x = data.x.float()

        if to_undirected_edges:
            data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

        dataset.append(data)

    if not dataset:
        raise ValueError("Loaded dataset is empty.")
    return dataset


def _safe_int(value: object, default: int = 0) -> int:
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


def _parse_dot_attrs(attr_str: str) -> dict[str, object]:
    attrs: dict[str, object] = {}
    for token in shlex.split(attr_str, posix=True):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        lowered = value.lower()
        if lowered == "true":
            attrs[key] = True
            continue
        if lowered == "false":
            attrs[key] = False
            continue
        try:
            attrs[key] = int(value)
            continue
        except ValueError:
            attrs[key] = value
    return attrs


def _load_token_rows_from_dot(dot_path: Path) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    for raw_line in dot_path.read_text().splitlines():
        line = raw_line.strip()
        if not line.startswith('"n') or '" -> "' in line:
            continue
        if "[" not in line or "]" not in line:
            continue

        left = line.index("[")
        right = line.rindex("]")
        attrs = _parse_dot_attrs(line[left + 1 : right])

        node_index = _safe_int(attrs.get("node_index", -1), -1)
        if node_index < 0:
            continue

        node_kind = str(attrs.get("node_kind", ""))
        token_text = str(attrs.get("token_text", "")) if node_kind == "SyntaxToken" else ""
        rows.append((node_index, node_kind, token_text))

    rows.sort(key=lambda item: item[0])
    return rows


def _load_manifest_dot_mapping(manifest_path: Path) -> dict[int, Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    mapping: dict[int, Path] = {}
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        status = str(row.get("status", ""))
        if status not in {"success", "skipped"}:
            continue

        json_index = _safe_int(row.get("json_index", -1), -1)
        if json_index < 0:
            continue

        raw_dot = str(row.get("output_dot", "")).strip()
        if raw_dot == "":
            continue
        dot_path = Path(raw_dot)
        if not dot_path.is_absolute():
            dot_path = (manifest_path.parent / dot_path).resolve()
        mapping[json_index] = dot_path

    if not mapping:
        raise ValueError(f"No success/skipped rows with DOT paths found in manifest: {manifest_path}")
    return mapping


def attach_token_vectors(
    dataset: list[Data],
    manifest_path: Path,
    vectors_path: Path,
    progress_every: int,
) -> tuple[int, dict[str, int | float]]:
    try:
        from gensim.models import KeyedVectors
    except ImportError as exc:  # pragma: no cover - dependency failure
        raise ImportError(
            "gensim is required to consume token vectors during training. Install dependencies with `uv sync`."
        ) from exc

    if progress_every <= 0:
        raise ValueError("token progress_every must be > 0")

    if not vectors_path.exists():
        raise FileNotFoundError(f"Token vectors file not found: {vectors_path}")

    dot_mapping = _load_manifest_dot_mapping(manifest_path)
    vectors = KeyedVectors.load(str(vectors_path), mmap="r")
    token_dim = int(vectors.vector_size)

    start_time = time.perf_counter()
    token_nodes = 0
    hit_tokens = 0
    oov_tokens = 0

    print(
        f"[token_features] attach start graphs={len(dataset)} token_dim={token_dim} "
        f"manifest={manifest_path}"
    )

    for processed, data in enumerate(dataset, start=1):
        json_index = _safe_int(getattr(data, "json_index", -1), -1)
        if json_index not in dot_mapping:
            raise ValueError(
                f"Missing DOT mapping for json_index={json_index}. "
                f"Manifest may not match dataset."
            )

        dot_path = dot_mapping[json_index]
        if not dot_path.exists():
            raise FileNotFoundError(f"DOT file missing for json_index={json_index}: {dot_path}")

        token_rows = _load_token_rows_from_dot(dot_path)
        if len(token_rows) != int(data.num_nodes):
            raise ValueError(
                f"Node count mismatch for json_index={json_index}: "
                f"dot_nodes={len(token_rows)} dataset_nodes={int(data.num_nodes)}"
            )

        token_x = torch.zeros((int(data.num_nodes), token_dim), dtype=torch.float32)
        for node_index, node_kind, token_text in token_rows:
            if node_kind != "SyntaxToken":
                continue
            token_nodes += 1
            if token_text in vectors:
                token_x[node_index] = torch.tensor(vectors[token_text], dtype=torch.float32)
                hit_tokens += 1
            else:
                oov_tokens += 1

        data.token_x = token_x

        if (
            processed % progress_every == 0
            or processed == len(dataset)
            or processed == 1
        ):
            elapsed = time.perf_counter() - start_time
            rate = processed / max(1e-9, elapsed)
            print(
                f"[token_features] attach processed={processed}/{len(dataset)} "
                f"token_nodes={token_nodes} hits={hit_tokens} oov={oov_tokens} "
                f"rate={rate:.2f} graph/s"
            )

    elapsed = time.perf_counter() - start_time
    coverage = hit_tokens / max(1, token_nodes)
    stats = {
        "token_dim": token_dim,
        "token_nodes": token_nodes,
        "token_hits": hit_tokens,
        "token_oov": oov_tokens,
        "token_coverage": round(coverage, 6),
        "elapsed_sec": round(elapsed, 3),
    }
    print(
        f"[token_features] attach done elapsed_sec={elapsed:.2f} coverage={coverage:.4f}"
    )
    return token_dim, stats


def label_matrix(dataset: list[Data], num_labels: int) -> torch.Tensor:
    return torch.stack([d.y.view(-1)[:num_labels] for d in dataset]).int()


def split_sample_level(
    dataset: list[Data],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    num_labels: int,
) -> tuple[list[Data], list[Data], list[Data], dict[str, list[int]]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1).")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in (0, 1).")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.")

    y = label_matrix(dataset, num_labels)
    n = y.size(0)

    if n < 3:
        raise ValueError("Need at least 3 graphs to produce non-empty train/val/test splits.")

    n_train = max(1, int(train_ratio * n))
    n_val = max(1, int(val_ratio * n))
    n_test = n - n_train - n_val

    if n_test < 1:
        overflow = 1 - n_test
        if n_train >= n_val and n_train - overflow >= 1:
            n_train -= overflow
        elif n_val - overflow >= 1:
            n_val -= overflow
        else:
            n_train = max(1, n_train - overflow)
        n_test = n - n_train - n_val

    if min(n_train, n_val, n_test) < 1:
        raise ValueError(
            "Split ratios produce an empty split for this dataset size; adjust ratios or max_graphs."
        )

    total_pos = y.sum(dim=0).float()
    target = {
        "train": total_pos * (n_train / n),
        "val": total_pos * (n_val / n),
        "test": total_pos * (n_test / n),
    }

    scores = y.sum(dim=1)
    generator = torch.Generator().manual_seed(seed)
    tiebreak = torch.rand(n, generator=generator)
    order = torch.argsort(scores.float() + tiebreak * 1e-3, descending=True)

    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    split_counts = {
        "train": torch.zeros(num_labels, dtype=torch.float32),
        "val": torch.zeros(num_labels, dtype=torch.float32),
        "test": torch.zeros(num_labels, dtype=torch.float32),
    }
    split_limits = {"train": n_train, "val": n_val, "test": n_test}

    for idx in order.tolist():
        label_vec = y[idx].float()
        available = [
            split
            for split in ("train", "val", "test")
            if len(split_indices[split]) < split_limits[split]
        ]
        if not available:
            break

        if int(label_vec.sum().item()) == 0:
            selected = max(
                available,
                key=lambda split: split_limits[split] - len(split_indices[split]),
            )
        else:
            selected = max(
                available,
                key=lambda split: (
                    (target[split] - split_counts[split])
                    .clamp(min=0)
                    .mul(label_vec)
                    .sum()
                    .item()
                ),
            )

        split_indices[selected].append(idx)
        split_counts[selected] += label_vec

    train_ds = [dataset[i] for i in split_indices["train"]]
    val_ds = [dataset[i] for i in split_indices["val"]]
    test_ds = [dataset[i] for i in split_indices["test"]]
    return train_ds, val_ds, test_ds, split_indices


def validate_split_indices(split_indices: dict[str, list[int]], dataset_size: int) -> None:
    required = {"train", "val", "test"}
    if set(split_indices.keys()) != required:
        raise ValueError("split_indices must contain train/val/test keys.")

    all_indices = []
    for split in ("train", "val", "test"):
        all_indices.extend(split_indices[split])

    if len(all_indices) != dataset_size:
        raise ValueError("Split indices do not cover full dataset.")
    if len(set(all_indices)) != dataset_size:
        raise ValueError("Split indices contain duplicates.")
    if min(all_indices) < 0 or max(all_indices) >= dataset_size:
        raise ValueError("Split indices out of range.")


def apply_split_indices(
    dataset: list[Data],
    split_indices: dict[str, list[int]],
) -> tuple[list[Data], list[Data], list[Data]]:
    validate_split_indices(split_indices=split_indices, dataset_size=len(dataset))
    train_ds = [dataset[i] for i in split_indices["train"]]
    val_ds = [dataset[i] for i in split_indices["val"]]
    test_ds = [dataset[i] for i in split_indices["test"]]
    return train_ds, val_ds, test_ds


def load_split_indices(split_path: Path) -> dict[str, list[int]]:
    payload = json.loads(split_path.read_text())
    split_indices = payload.get("split_indices")
    if not isinstance(split_indices, dict):
        raise ValueError(f"Invalid split file: {split_path}")
    return {
        "train": [int(item) for item in split_indices.get("train", [])],
        "val": [int(item) for item in split_indices.get("val", [])],
        "test": [int(item) for item in split_indices.get("test", [])],
    }


def save_split_indices(
    split_path: Path,
    split_indices: dict[str, list[int]],
    seed: int,
    dataset_size: int,
) -> None:
    validate_split_indices(split_indices=split_indices, dataset_size=dataset_size)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": int(seed),
        "dataset_size": int(dataset_size),
        "split_indices": split_indices,
    }
    split_path.write_text(json.dumps(payload, indent=2))


def build_loader(
    dataset: list[Data],
    max_nodes_per_batch: int,
    max_edges_per_batch: int | None,
    shuffle: bool,
    seed: int,
    epoch: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    node_counts = [int(d.num_nodes) for d in dataset]
    edge_counts = [int(d.edge_index.size(1)) for d in dataset]
    batch_sampler = DynamicBudgetBatchSampler(
        node_counts=node_counts,
        edge_counts=edge_counts,
        max_nodes=max_nodes_per_batch,
        max_edges=max_edges_per_batch,
        shuffle=shuffle,
        seed=seed,
    )
    batch_sampler.set_epoch(epoch)
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def compute_f1_scores(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[float, float, list[float]]:
    y_true = y_true.int()
    y_pred = y_pred.int()

    tp = ((y_true == 1) & (y_pred == 1)).sum().float()
    fp = ((y_true == 0) & (y_pred == 1)).sum().float()
    fn = ((y_true == 1) & (y_pred == 0)).sum().float()
    f1_micro = (2 * tp) / (2 * tp + fp + fn + eps)

    per_label: list[float] = []
    for i in range(y_true.size(1)):
        tp_i = ((y_true[:, i] == 1) & (y_pred[:, i] == 1)).sum().float()
        fp_i = ((y_true[:, i] == 0) & (y_pred[:, i] == 1)).sum().float()
        fn_i = ((y_true[:, i] == 1) & (y_pred[:, i] == 0)).sum().float()
        f1_i = (2 * tp_i) / (2 * tp_i + fp_i + fn_i + eps)
        per_label.append(float(f1_i.item()))

    f1_macro = float(sum(per_label) / len(per_label))
    return float(f1_micro.item()), f1_macro, per_label


def compute_pr_auc(y_true: torch.Tensor, probs: torch.Tensor) -> tuple[float, list[float]]:
    y_true = y_true.int().cpu()
    probs = probs.float().cpu()

    per_label: list[float] = []
    for i in range(y_true.size(1)):
        y = y_true[:, i]
        p = probs[:, i]
        positives = int(y.sum().item())
        if positives == 0:
            per_label.append(0.0)
            continue

        order = torch.argsort(p, descending=True)
        y_sorted = y[order]

        tp = torch.cumsum(y_sorted, dim=0).float()
        fp = torch.cumsum(1 - y_sorted, dim=0).float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (positives + 1e-8)

        recall = torch.cat([torch.tensor([0.0]), recall])
        precision = torch.cat([torch.tensor([1.0]), precision])
        area = torch.trapezoid(precision, recall).item()
        per_label.append(float(area))

    macro = float(sum(per_label) / len(per_label))
    return macro, per_label


@torch.no_grad()
def collect_logits_and_targets(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_labels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        targets = batch.y.view(-1, num_labels).float()
        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    if not all_logits:
        return torch.empty((0, num_labels)), torch.empty((0, num_labels))

    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_labels: int,
    thresholds: torch.Tensor | None,
) -> dict[str, float | list[float]]:
    logits, y_true = collect_logits_and_targets(model, loader, device, num_labels)
    if logits.numel() == 0:
        return {
            "f1_micro": 0.0,
            "f1_macro": 0.0,
            "f1_per_label": [0.0] * num_labels,
            "pr_auc_macro": 0.0,
            "pr_auc_per_label": [0.0] * num_labels,
        }

    probs = torch.sigmoid(logits)
    if thresholds is None:
        thresholds = torch.full((num_labels,), 0.5)

    y_pred = (probs >= thresholds.view(1, -1)).int()
    y_true_int = y_true.int()

    f1_micro, f1_macro, f1_per_label = compute_f1_scores(y_true_int, y_pred)
    pr_auc_macro, pr_auc_per_label = compute_pr_auc(y_true_int, probs)
    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_per_label": f1_per_label,
        "pr_auc_macro": pr_auc_macro,
        "pr_auc_per_label": pr_auc_per_label,
    }


def tune_thresholds(
    y_true: torch.Tensor,
    probs: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    thresholds = torch.full((y_true.size(1),), 0.5)
    grid = torch.linspace(0.0, 1.0, max(steps, 3))

    for i in range(y_true.size(1)):
        label_true = y_true[:, i].int()
        if int(label_true.sum().item()) == 0:
            thresholds[i] = 0.5
            continue

        best_f1 = -1.0
        best_t = 0.5
        for threshold in grid:
            label_pred = (probs[:, i] >= threshold).int()
            f1_micro, _, _ = compute_f1_scores(
                label_true.view(-1, 1),
                label_pred.view(-1, 1),
            )
            if f1_micro > best_f1:
                best_f1 = f1_micro
                best_t = float(threshold.item())
        thresholds[i] = best_t
    return thresholds


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def run_name(default_prefix: str, provided: str | None) -> str:
    if provided:
        return provided
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{default_prefix}_{now}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a sample-level multi-label GCN baseline with dynamic node-budget batching.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("artifacts/cache/dataset/latest/dataset.pt"),
        help="Path to list[torch_geometric.data.Data] dataset.",
    )
    parser.add_argument("--num-labels", type=int, default=4)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--type-emb-dim", type=int, default=128)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    parser.add_argument("--max-nodes-per-batch", type=int, default=20000)
    parser.add_argument(
        "--max-edges-per-batch",
        type=int,
        default=0,
        help="0 disables edge budget constraint.",
    )
    parser.add_argument("--eval-max-nodes-per-batch", type=int, default=40000)
    parser.add_argument(
        "--eval-max-edges-per-batch",
        type=int,
        default=0,
        help="0 disables edge budget constraint.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--to-undirected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert edge_index to undirected before split.",
    )

    parser.add_argument("--threshold-steps", type=int, default=101)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--feature-mode",
        choices=["type_numeric", "type_only", "numeric_only"],
        default="type_numeric",
    )
    parser.add_argument(
        "--architecture",
        choices=["gcn", "gat", "graphsage"],
        default="gcn",
        help="GNN architecture to use",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of GNN message-passing layers (default 2, research recommends 2-3 max for AST graphs)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="GAT: number of attention heads (ignored for other architectures)",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        help="GraphSAGE: aggregation method (ignored for other architectures)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="weighted_bce",
        choices=["weighted_bce", "focal"],
        help="Loss function to use (default: weighted_bce)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss focusing parameter gamma (ignored when --loss=weighted_bce)",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=-1.0,
        help="Focal loss alpha parameter; -1.0 disables scalar alpha (ignored when --loss=weighted_bce)",
    )
    parser.add_argument(
        "--early-stopping-metric",
        type=str,
        default="macro_f1",
        choices=["macro_f1", "pr_auc"],
        help="Validation metric to monitor for early stopping",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum improvement in early stopping metric to count as improvement",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--token-vectors-path",
        type=Path,
        default=None,
        help="Optional path to gensim KeyedVectors (.kv) for token-node lexical features.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Manifest path used to map dataset json_index to DOT files for token features.",
    )
    parser.add_argument(
        "--token-progress-every",
        type=int,
        default=200,
        help="Progress logging frequency while attaching token vectors.",
    )
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument(
        "--save-split",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reuse-split",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def run_training(cfg: dict[str, Any], wandb_run: Any | None = None) -> dict[str, Any]:
    """Run a complete training trial from a config dict.

    This is the core training function used by both the CLI and the pipeline.
    It handles dataset loading, splitting, model creation, training, evaluation,
    threshold tuning, curve generation and metrics writing.

    Args:
        cfg: Flat config dict with keys matching argparse defaults (e.g.
            'dataset_path', 'seed', 'epochs', 'architecture', etc.). The
            pipeline constructs this from YAML; the CLI builds it from argparse.
        wandb_run: Optional W&B run object for live per-epoch logging. When
            provided, epoch metrics are streamed to W&B during training.

    Returns:
        The metrics dict that is also written to metrics.json.
    """
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = select_device(str(cfg.get("device", "auto")))
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
    token_feature_stats: dict[str, int | float] | None = None
    if use_token_features:
        if manifest_path is None:
            raise ValueError(
                "manifest_path is required when token_vectors_path is provided."
            )
        token_feature_dim, token_feature_stats = attach_token_vectors(
            dataset=dataset,
            manifest_path=Path(str(manifest_path)),
            vectors_path=Path(str(token_vectors_path)),
            progress_every=int(cfg.get("token_progress_every", 200)),
        )

    train_ratio = float(cfg.get("train_ratio", 0.8))
    val_ratio = float(cfg.get("val_ratio", 0.1))
    split_path_raw = cfg.get("split_path")
    split_path = Path(str(split_path_raw)) if split_path_raw is not None else None
    reuse_split = bool(cfg.get("reuse_split", True))
    save_split = bool(cfg.get("save_split", True))

    split_indices: dict[str, list[int]]
    if split_path is not None and reuse_split and split_path.exists():
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
        if split_path is not None and save_split:
            save_split_indices(
                split_path=split_path,
                split_indices=split_indices,
                seed=seed,
                dataset_size=len(dataset),
            )

    if not train_ds or not val_ds or not test_ds:
        raise ValueError("One split is empty; adjust split ratios or dataset size.")

    train_summary = summarize_dataset(train_ds, SMELL_LABELS)
    val_summary = summarize_dataset(val_ds, SMELL_LABELS)
    test_summary = summarize_dataset(test_ds, SMELL_LABELS)

    feature_mode = str(cfg.get("feature_mode", "type_numeric"))
    use_type_features = feature_mode in {"type_numeric", "type_only"}
    use_numeric_features = feature_mode in {"type_numeric", "numeric_only"}

    if not use_type_features and not use_numeric_features and not use_token_features:
        raise ValueError("Invalid feature_mode: both feature sources disabled.")

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

    # Detect graph-level features if present in dataset
    num_graph_features = 0
    sample_graph_x = getattr(dataset[0], "graph_x", None)
    if sample_graph_x is not None:
        num_graph_features = sample_graph_x.shape[-1]

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
        num_graph_features=num_graph_features,
        use_type_features=use_type_features,
        use_numeric_features=use_numeric_features,
        use_token_features=use_token_features,
        **arch_kwargs,
    ).to(device)

    lr = float(cfg.get("lr", 0.0005))
    weight_decay = float(cfg.get("weight_decay", 5e-5))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

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
    run_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(cfg.get("epochs", 40))
    patience = int(cfg.get("patience", 10))
    early_stopping_metric = str(cfg.get("early_stopping_metric", "macro_f1"))
    early_stopping_min_delta = float(cfg.get("early_stopping_min_delta", 0.0))
    grad_accum_steps = int(cfg.get("grad_accum_steps", 1))
    grad_clip_norm = float(cfg.get("grad_clip_norm", 1.0))
    threshold_steps = int(cfg.get("threshold_steps", 101))

    config_snapshot = {
        "device": str(device),
        "architecture": architecture,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "aggregation": aggregation,
        "feature_mode": feature_mode,
        "use_type_features": use_type_features,
        "use_numeric_features": use_numeric_features,
        "use_token_features": use_token_features,
        "loss": loss_name,
        "focal_gamma": focal_gamma,
        "focal_alpha": focal_alpha,
        "early_stopping_metric": early_stopping_metric,
        "early_stopping_min_delta": early_stopping_min_delta,
        "train_summary": asdict(train_summary),
        "val_summary": asdict(val_summary),
        "test_summary": asdict(test_summary),
        "split_sizes": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "split_indices": split_indices,
        "num_node_types": num_node_types,
        "num_numeric_feats": num_numeric_feats,
        "num_token_feats": token_feature_dim,
        "token_feature_stats": token_feature_stats,
        "split_path": str(split_path) if split_path is not None else None,
    }
    (run_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2, default=str))

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
        config_snapshot=config_snapshot,
        seed=seed,
        grad_accum_steps=grad_accum_steps,
        grad_clip_norm=grad_clip_norm,
        architecture=architecture,
        loss_name=loss_name,
        wandb_run=wandb_run,
    )

    max_nodes_per_batch = int(cfg.get("max_nodes_per_batch", 20000))
    eval_max_nodes = int(cfg.get("eval_max_nodes_per_batch", 40000))
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", False))

    def train_loader_fn(epoch: int) -> DataLoader:
        return build_loader(
            dataset=train_ds,
            max_nodes_per_batch=max_nodes_per_batch,
            max_edges_per_batch=edge_budget,
            shuffle=True,
            seed=seed,
            epoch=epoch,
            num_workers=num_workers,
            pin_memory=pin_memory,
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

    best_epoch = result.best_epoch

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
        model=model,
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
        model=model,
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
        "threshold_tuning": {
            "strategy": "per_label_val_f1_grid",
            "threshold_steps": threshold_steps,
            "fixed_threshold": 0.5,
            "pr_auc_threshold_independent": True,
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
            "eval_duration_sec": round(eval_duration_sec, 3),
        },
    }

    (run_dir / "metrics.json").write_text(json.dumps(artifacts, indent=2))
    return artifacts


def main() -> None:
    args = parse_args()

    cfg: dict[str, Any] = {
        "dataset_path": str(args.dataset_path),
        "num_labels": args.num_labels,
        "max_graphs": args.max_graphs,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "hidden_dim": args.hidden_dim,
        "type_emb_dim": args.type_emb_dim,
        "grad_accum_steps": args.grad_accum_steps,
        "grad_clip_norm": args.grad_clip_norm,
        "max_nodes_per_batch": args.max_nodes_per_batch,
        "max_edges_per_batch": args.max_edges_per_batch,
        "eval_max_nodes_per_batch": args.eval_max_nodes_per_batch,
        "eval_max_edges_per_batch": args.eval_max_edges_per_batch,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "to_undirected": args.to_undirected,
        "threshold_steps": args.threshold_steps,
        "device": args.device,
        "feature_mode": args.feature_mode,
        "architecture": args.architecture,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "aggregation": args.aggregation,
        "loss": args.loss,
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "early_stopping_metric": args.early_stopping_metric,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "output_dir": str(args.output_dir),
        "run_name": args.run_name,
        "token_vectors_path": str(args.token_vectors_path) if args.token_vectors_path else None,
        "manifest_path": str(args.manifest_path) if args.manifest_path else None,
        "token_progress_every": args.token_progress_every,
        "split_path": str(args.split_path) if args.split_path else None,
        "save_split": args.save_split,
        "reuse_split": args.reuse_split,
    }

    run_training(cfg)


if __name__ == "__main__":
    main()
