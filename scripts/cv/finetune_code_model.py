"""Fine-tune pretrained code encoders on the shared cross-validation folds.

One work item is (model, split strategy, seed, fold). Each item writes record.json,
predictions.jsonl and dynamics.npz. --collect merges the items of each model.

Usage:
    python scripts/cv/finetune_code_model.py --config config/experiments/code_finetune.yml
    python scripts/cv/finetune_code_model.py --config ... --shard-index 0 --shard-count 6
    python scripts/cv/finetune_code_model.py --config ... --collect
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.evaluation.cv import (  # noqa: E402
    Fold,
    build_record,
    check_variant_entries,
    fold_digest,
    folds_label_order,
    iter_folds,
    label_matrix_from_json,
    load_folds,
    prediction_rows,
    read_jsonl,
    shard,
    variant_predictions_path,
    write_jsonl,
)

SECTION = "code_finetune"
HPARAM_KEYS = (
    "path", "max_length", "truncation", "head_tokens", "prefix_tokens", "pooling", "dropout", "lr",
    "batch_size", "grad_accum", "epochs", "warmup_ratio", "weight_decay", "grad_clip_norm", "loss",
    "focal_gamma", "focal_alpha", "gradient_checkpointing", "attn_implementation", "eval_batch_size",
)
# every setting comes from the config, a missing key is an error
REQUIRED_KEYS = (
    "folds", "input_json", "models_dir", "out_dir", "strategies", "seeds", "fold_ids", "threshold_steps", "device",
    "bf16", "tf32", "dynamics", "early_stopping", "defaults", "models", "run_models",
)
EARLY_STOPPING_KEYS = ("metric", "patience", "min_delta")


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_losses_module() -> Any:
    """Load mlcq_graphs/training/losses.py by path, without importing torch_geometric."""
    path = PROJECT_ROOT / "mlcq_graphs" / "training" / "losses.py"
    spec = importlib.util.spec_from_file_location("detect_training_losses", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# Configuration and work items


def model_spec(fcfg: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in fcfg["models"]:
        raise SystemExit(f"unknown model variant {name!r}; known: {sorted(fcfg['models'])}")
    spec = {**(fcfg["defaults"] or {}), **(fcfg["models"][name] or {})}
    missing = [k for k in HPARAM_KEYS if k not in spec]
    if missing:
        raise SystemExit(f"model variant {name!r} lacks {missing} (set them in defaults or in the variant)")
    return spec


def check_config(fcfg: dict[str, Any]) -> None:
    """Fail before any training when a key is missing."""
    missing = [k for k in REQUIRED_KEYS if k not in fcfg]
    missing += [f"early_stopping.{k}" for k in EARLY_STOPPING_KEYS if k not in (fcfg.get("early_stopping") or {})]
    if missing:
        raise SystemExit(f"config section {SECTION} lacks {missing}")
    if str(fcfg["early_stopping"]["metric"]) not in ("macro_f1", "ap_macro"):
        raise SystemExit("early_stopping.metric must be macro_f1 (validation, threshold 0.5) or ap_macro "
                         "(validation average precision, no threshold)")
    for name in fcfg["run_models"]:
        spec = model_spec(fcfg, name)
        target = str(spec.get("target", "consensus"))
        if target not in ("consensus", "review_share"):
            raise SystemExit(f"model variant {name!r}: unknown target {target!r} (consensus | review_share)")
        if target == "review_share" and not fcfg.get("review_shares"):
            raise SystemExit(f"model variant {name!r} has target review_share but the section has no review_shares file")


def build_items(fcfg: dict[str, Any], folds: dict[str, Any]) -> list[tuple[str, Fold]]:
    fold_list = list(iter_folds(folds, fcfg["strategies"], fcfg["seeds"], fcfg["fold_ids"]))
    return [(name, fold) for name in fcfg["run_models"] for fold in fold_list]


def item_dir(out_root: Path, name: str, fold: Fold) -> Path:
    return out_root / name / f"{fold.strategy}_s{fold.seed}_f{fold.fold}"


def stop_on_stale(out_root: Path, items: list[tuple[str, Fold]]) -> None:
    """Stop when a finished item was trained on other folds than the current folds file."""
    stale = []
    for name, fold in items:
        path = item_dir(out_root, name, fold) / "record.json"
        if path.exists():
            recorded = json.loads(path.read_text()).get("fold_sha256")
            if recorded is not None and recorded != fold_digest(fold):
                stale.append(f"{name}/{path.parent.name}")
    if stale:
        raise SystemExit(f"{len(stale)} finished items were trained on other folds than the current folds file "
                         f"(first {stale[:5]}); remove their folders to retrain them")


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Tokenization


class Encoded:
    """Token ids of every snippet for one tokenizer, before truncation."""

    def __init__(self, tokenizer: Any, codes: dict[int, str]) -> None:
        self.tokenizer = tokenizer
        keys = sorted(codes)
        ids = tokenizer([codes[k] for k in keys], add_special_tokens=False)["input_ids"]
        self.ids = dict(zip(keys, ids))
        self.prefix, self.suffix = special_template(tokenizer)
        self.pad_id = int(tokenizer.pad_token_id)


def special_template(tokenizer: Any) -> tuple[list[int], list[int]]:
    """Special tokens the tokenizer puts before and after a single sequence."""
    bare = tokenizer("x = 1", add_special_tokens=False)["input_ids"]
    full = tokenizer("x = 1", add_special_tokens=True)["input_ids"]
    for k in range(len(full) - len(bare) + 1):
        if full[k:k + len(bare)] == bare:
            return list(full[:k]), list(full[k + len(bare):])
    raise ValueError("cannot locate the sequence inside the tokenizer template")


def truncate(ids: list[int], budget: int, mode: str, head_tokens: int) -> list[int]:
    if len(ids) <= budget:
        return list(ids)
    if mode == "head":
        return list(ids[:budget])
    if mode == "tail":
        return list(ids[len(ids) - budget:])
    if mode == "head_tail":
        head = min(int(head_tokens), budget)
        tail = budget - head
        return list(ids[:head]) + (list(ids[len(ids) - tail:]) if tail > 0 else [])
    raise ValueError(f"unknown truncation mode {mode!r} (head | tail | head_tail)")


class SequenceSet:
    """Model inputs (special tokens and truncation applied) for one model variant."""

    def __init__(self, encoded: Encoded, spec: dict[str, Any]) -> None:
        tok = encoded.tokenizer
        extra = list(spec["prefix_tokens"] or [])
        extra_ids = tok.convert_tokens_to_ids(extra) if extra else []
        for token, tid in zip(extra, extra_ids):
            if tid is None or (tid == tok.unk_token_id and token != tok.unk_token):
                raise SystemExit(f"prefix token {token!r} is not in the vocabulary of {spec['path']}")
        head = encoded.prefix + list(extra_ids)
        self.budget = int(spec["max_length"]) - len(head) - len(encoded.suffix)
        if self.budget <= 0:
            raise SystemExit(f"max_length {spec['max_length']} leaves no room for code tokens")
        mode = str(spec["truncation"])
        head_tokens = int(spec["head_tokens"])
        self.pad_id = encoded.pad_id
        self.num_tokens = {k: len(v) for k, v in encoded.ids.items()}
        self.inputs = {
            k: head + truncate(v, self.budget, mode, head_tokens) + encoded.suffix for k, v in encoded.ids.items()
        }

    def truncated(self, indices: np.ndarray) -> np.ndarray:
        return np.array([self.num_tokens[int(i)] > self.budget for i in indices], dtype=bool)

    def batch(self, indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [self.inputs[int(i)] for i in indices]
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for k, r in enumerate(rows):
            ids[k, : len(r)] = torch.tensor(r, dtype=torch.long)
            mask[k, : len(r)] = 1
        return ids.to(device), mask.to(device)


# Model


class EncoderClassifier(nn.Module):
    """Pretrained encoder, pooling (first token or masked mean), dropout, linear head."""

    def __init__(self, encoder: nn.Module, num_labels: int, dropout: float, pooling: str, init_std: float) -> None:
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError(f"unknown pooling {pooling!r} (cls | mean)")
        self.encoder = encoder
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(int(encoder.config.hidden_size), num_labels)
        nn.init.normal_(self.head.weight, std=init_std)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            m = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.head(self.dropout(pooled))


def position_limit(config: Any) -> int | None:
    limit = getattr(config, "max_position_embeddings", None)
    if limit is None:
        return None
    if getattr(config, "model_type", "") in ("roberta", "xlm-roberta", "camembert"):
        # RoBERTa positions start after the padding index
        limit -= int(config.pad_token_id) + 1
    return int(limit)


def build_model(spec: dict[str, Any], model_path: Path, num_labels: int) -> EncoderClassifier:
    from transformers import AutoModel

    kwargs: dict[str, Any] = {"dtype": torch.float32}
    if spec["attn_implementation"]:
        kwargs["attn_implementation"] = str(spec["attn_implementation"])
    encoder = AutoModel.from_pretrained(str(model_path), **kwargs)
    limit = position_limit(encoder.config)
    if limit is not None and int(spec["max_length"]) > limit:
        raise SystemExit(f"max_length {spec['max_length']} exceeds the {limit} positions of {model_path.name}")
    if bool(spec["gradient_checkpointing"]):
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # head initialised like the encoder's own layers (initializer_range of its config)
    init_std = float(encoder.config.initializer_range)
    return EncoderClassifier(encoder, num_labels, float(spec["dropout"]), str(spec["pooling"]), init_std)


def build_loss(spec: dict[str, Any], losses: Any, y_train: np.ndarray, device: torch.device) -> nn.Module:
    name = str(spec["loss"])
    if name == "bce":
        return nn.BCEWithLogitsLoss().to(device)
    # build_loss_fn reads `.y` of each sample for the class weights, as for the GNN
    train_ds = [SimpleNamespace(y=torch.tensor(row, dtype=torch.float32)) for row in y_train]
    return losses.build_loss_fn(
        loss_name=name,
        train_ds=train_ds,
        num_labels=y_train.shape[1],
        device=device,
        focal_gamma=float(spec["focal_gamma"]),
        focal_alpha=float(spec["focal_alpha"]),
    )


def build_optimizer(model: nn.Module, spec: dict[str, Any]) -> torch.optim.Optimizer:
    """AdamW without weight decay on biases and normalisation weights."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim < 2 or "norm" in name.lower() else decay).append(param)
    groups = [
        {"params": decay, "weight_decay": float(spec["weight_decay"])},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=float(spec["lr"]))


# Evaluation helpers


def gnn_f1_scores(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> tuple[float, float, list[float]]:
    """Same formulas as train_gcn_baseline.compute_f1_scores."""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    micro = 2 * tp / (2 * tp + fp + fn + eps)
    per_label = []
    for i in range(y_true.shape[1]):
        t, p = y_true[:, i], y_pred[:, i]
        tp_i = float(((t == 1) & (p == 1)).sum())
        fp_i = float(((t == 0) & (p == 1)).sum())
        fn_i = float(((t == 1) & (p == 0)).sum())
        per_label.append(2 * tp_i / (2 * tp_i + fp_i + fn_i + eps))
    return micro, float(sum(per_label) / len(per_label)), per_label


def ap_macro(y_true: np.ndarray, probs: np.ndarray) -> float:
    values = [float(average_precision_score(y_true[:, i], probs[:, i])) if y_true[:, i].sum() > 0 else 0.0
              for i in range(y_true.shape[1])]
    return float(np.mean(values))


@torch.no_grad()
def predict(model: nn.Module, seqs: SequenceSet, indices: np.ndarray, batch_size: int, device: torch.device,
            amp_dtype: torch.dtype | None) -> np.ndarray:
    """Sigmoid probabilities in the order of `indices`, with batches sorted by length."""
    model.eval()
    order = np.argsort([len(seqs.inputs[int(i)]) for i in indices], kind="stable")
    out = np.zeros((len(indices), model.head.out_features), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        pos = order[start:start + batch_size]
        ids, mask = seqs.batch(indices[pos], device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(ids, mask)
        out[pos] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


# Atomic writes


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(buffer.getvalue())
    tmp.replace(path)


# One work item


def run_item(
    *,
    name: str,
    spec: dict[str, Any],
    fold: Fold,
    fcfg: dict[str, Any],
    entries: list[dict[str, Any]],
    seqs: SequenceSet,
    variant_seqs: dict[str, SequenceSet],
    losses: Any,
    device: torch.device,
    out: Path,
    shares: np.ndarray | None,
) -> dict[str, Any]:
    num_labels = len(fcfg["label_order"])
    subset = fcfg.get("subset") or {}
    parts: dict[str, np.ndarray] = {}
    for part in ("train", "val", "test"):
        idx = getattr(fold, part)
        n = subset.get(part)
        parts[part] = idx[: int(n)] if n else idx
    Y = {p: label_matrix_from_json(entries, idx, num_labels) for p, idx in parts.items()}
    target = str(spec.get("target", "consensus"))
    if target == "review_share":
        assert shares is not None
        T_train = shares[parts["train"], :num_labels].astype(np.float32)
    else:
        T_train = Y["train"].astype(np.float32)

    use_bf16 = bool(fcfg["bf16"]) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    set_seed(fold.seed)
    model = build_model(spec, resolve(fcfg["models_dir"]) / str(spec["path"]), num_labels).to(device)
    loss_fn = build_loss(spec, losses, Y["train"], device)
    optimizer = build_optimizer(model, spec)

    from transformers import get_linear_schedule_with_warmup

    batch_size = int(spec["batch_size"])
    grad_accum = max(1, int(spec["grad_accum"]))
    epochs = int(spec["epochs"])
    eval_bs = int(spec["eval_batch_size"])
    clip = float(spec["grad_clip_norm"])
    n_train = len(parts["train"])
    batches_per_epoch = math.ceil(n_train / batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup = int(math.ceil(float(spec["warmup_ratio"]) * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    # validation macro F1 at 0.5 or validation macro average precision
    es = fcfg["early_stopping"]
    es_metric = "val_ap_macro" if str(es["metric"]) == "ap_macro" else "val_f1_macro"
    patience = int(es["patience"])
    min_delta = float(es["min_delta"])
    keep_dynamics = bool(fcfg["dynamics"])

    trunc = {p: seqs.truncated(idx) for p, idx in parts.items()}
    all_trunc = np.concatenate(list(trunc.values()))
    print(f"[finetune] {name} {fold.strategy} seed={fold.seed} fold={fold.fold} device={device} bf16={use_bf16} "
          f"train={n_train} val={len(parts['val'])} test={len(parts['test'])} "
          f"truncated={all_trunc.mean():.3f} steps={total_steps} warmup={warmup}", flush=True)

    best_metric, best_epoch, no_improve, stopped_early = -float("inf"), -1, 0, False
    best_state: dict[str, torch.Tensor] | None = None
    best_val: np.ndarray | None = None
    history: list[dict[str, Any]] = []
    train_probs: list[np.ndarray] = []
    val_probs: list[np.ndarray] = []
    epoch_seconds: list[float] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        rng = np.random.default_rng([fold.seed, epoch])
        perm = rng.permutation(n_train)
        total_loss, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for b in range(batches_per_epoch):
            pos = perm[b * batch_size:(b + 1) * batch_size]
            idx = parts["train"][pos]
            ids, mask = seqs.batch(idx, device)
            y = torch.as_tensor(T_train[pos], dtype=torch.float32, device=device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(ids, mask)
            loss = loss_fn(logits.float(), y)
            (loss / grad_accum).backward()
            total_loss += float(loss.item()) * len(idx)
            seen += len(idx)
            if (b + 1) % grad_accum == 0 or b + 1 == batches_per_epoch:
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        train_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        if keep_dynamics:
            train_probs.append(predict(model, seqs, parts["train"], eval_bs, device, amp_dtype))
        p_val = predict(model, seqs, parts["val"], eval_bs, device, amp_dtype)
        val_probs.append(p_val)
        eval_sec = time.perf_counter() - t1
        epoch_seconds.append(time.perf_counter() - t0)

        f1_micro, f1_macro, f1_labels = gnn_f1_scores(Y["val"], (p_val >= 0.5).astype(np.int64))
        entry = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "val_f1_micro": f1_micro,
            "val_f1_macro": f1_macro,
            "val_f1_per_label": f1_labels,
            "val_ap_macro": ap_macro(Y["val"], p_val),
            "lr": float(scheduler.get_last_lr()[0]),
            "train_sec": train_sec,
            "eval_sec": eval_sec,
        }
        history.append(entry)
        print(f"Epoch {epoch:03d} | loss={entry['train_loss']:.4f} | val_f1_micro={f1_micro:.4f} | "
              f"val_f1_macro={f1_macro:.4f} | val_ap={entry['val_ap_macro']:.4f} | "
              f"train {train_sec:.1f}s eval {eval_sec:.1f}s", flush=True)

        if (entry[es_metric] - best_metric) > min_delta:
            best_metric, best_epoch, no_improve = entry[es_metric], epoch, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            best_val = p_val
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)", flush=True)
            stopped_early = True
            break

    train_duration = time.perf_counter() - started
    assert best_state is not None and best_val is not None
    model.load_state_dict(best_state)
    t2 = time.perf_counter()
    p_val = predict(model, seqs, parts["val"], eval_bs, device, amp_dtype)
    p_test = predict(model, seqs, parts["test"], eval_bs, device, amp_dtype)
    test_sec = time.perf_counter() - t2
    # the same model on the test indices of every metamorphic variant
    p_variants: dict[str, np.ndarray] = {}
    variant_info: dict[str, Any] = {}
    for variant, vseqs in variant_seqs.items():
        t3 = time.perf_counter()
        p_variants[variant] = predict(model, vseqs, parts["test"], eval_bs, device, amp_dtype)
        variant_info[variant] = {"eval_sec": time.perf_counter() - t3,
                                 "truncation_rate_test": float(vseqs.truncated(parts["test"]).mean())}
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None

    record = build_record(
        combo={"model": name, "split_strategy": fold.strategy, "cv_fold": fold.fold},
        seed=fold.seed,
        y_val=Y["val"], p_val=p_val, y_test=Y["test"], p_test=p_test,
        threshold_steps=int(fcfg["threshold_steps"]),
        label_names=list(fcfg["label_order"]),
        extra={
            "family": "code_finetune",
            "hparams": {k: spec[k] for k in HPARAM_KEYS if k in spec},
            "train_target": target,
            "review_shares": str(fcfg.get("review_shares")) if target == "review_share" else None,
            "best_epoch": best_epoch,
            "best_val_f1_macro_at_0_5": history[best_epoch - 1]["val_f1_macro"],
            "early_stopping_metric": es_metric,
            "best_val_early_stopping_metric": best_metric,
            "epochs_run": len(history),
            "stopped_early": stopped_early,
            "train_duration_sec": train_duration,
            "epoch_seconds": epoch_seconds,
            "test_eval_sec": test_sec,
            "peak_gpu_mem_bytes": peak,
            "peak_gpu_reserved_bytes": peak_reserved,
            "truncation_rate": float(all_trunc.mean()),
            "truncation_rate_by_part": {p: float(t.mean()) for p, t in trunc.items()},
            "token_budget": seqs.budget,
            "bf16": use_bf16,
            "device": str(device),
            "torch": torch.__version__,
            "split_sizes": {p: int(len(idx)) for p, idx in parts.items()},
            "subset": {k: v for k, v in subset.items() if v} or None,
            "val_probs_recomputed_max_abs_diff": float(np.abs(p_val - best_val).max()),
            "fold_sha256": fold_digest(fold),
            "history": history,
            **({"eval_variants": variant_info} if variant_info else {}),
        },
    )

    rows = []
    for part in ("val", "test"):
        rows.extend(prediction_rows(model=name, strategy=fold.strategy, seed=fold.seed, fold=fold.fold, part=part,
                                    indices=parts[part], y=Y[part], probs=p_val if part == "val" else p_test))
    write_jsonl(out / "predictions.jsonl", rows)
    for variant, probs in p_variants.items():
        write_jsonl(variant_predictions_path(out, variant), prediction_rows(
            model=name, strategy=fold.strategy, seed=fold.seed, fold=fold.fold, part="test",
            indices=parts["test"], y=Y["test"], probs=probs))
    dyn: dict[str, np.ndarray] = {
        "epochs": np.arange(1, len(history) + 1, dtype=np.int32),
        "best_epoch": np.int32(best_epoch),
        "label_order": np.array(fcfg["label_order"]),
        "train_index": parts["train"].astype(np.int64),
        "train_y": Y["train"].astype(np.int8),
        "train_num_tokens": np.array([seqs.num_tokens[int(i)] for i in parts["train"]], dtype=np.int32),
        "val_index": parts["val"].astype(np.int64),
        "val_y": Y["val"].astype(np.int8),
        "val_prob": np.stack(val_probs).astype(np.float32),
        "train_loss": np.array([h["train_loss"] for h in history], dtype=np.float32),
        "val_f1_macro": np.array([h["val_f1_macro"] for h in history], dtype=np.float32),
    }
    if keep_dynamics:
        dyn["train_prob"] = np.stack(train_probs).astype(np.float32)
    write_npz(out / "dynamics.npz", **dyn)
    write_json(out / "record.json", record)

    per = record["per_label_tuned"]
    shown = " ".join(f"{s.removeprefix('is_')}={'NA' if per[s]['mcc'] is None else round(per[s]['mcc'], 3)}"
                     for s in fcfg["label_order"])
    print(f"[finetune] done {name} {fold.strategy} seed={fold.seed} fold={fold.fold} best_epoch={best_epoch} "
          f"macroMCC={record['mcc_tuned']:.3f} {shown} "
          f"train={train_duration:.0f}s peak_mem={'n/a' if peak is None else f'{peak / 2**30:.2f}GiB'}"
          + (f" variants={sorted(p_variants)}" if p_variants else ""), flush=True)

    del model, optimizer, scheduler, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


# Collect


def collect(fcfg: dict[str, Any], folds: dict[str, Any], out_root: Path) -> None:
    """Merge every finished item of each model and list the configured items still missing."""
    every = list(iter_folds(folds))
    wanted = {(f.strategy, f.seed, f.fold)
              for f in iter_folds(folds, fcfg["strategies"], fcfg["seeds"], fcfg["fold_ids"])}
    stop_on_stale(out_root, [(name, fold) for name in fcfg["run_models"] for fold in every])
    variants = list(fcfg.get("eval_variants") or {})
    for name in fcfg["run_models"]:
        records, preds, missing = [], [], []
        variant_preds: dict[str, list[dict[str, Any]]] = {v: [] for v in variants}
        variant_missing: dict[str, list[str]] = {v: [] for v in variants}
        for fold in every:
            d = item_dir(out_root, name, fold)
            if not (d / "record.json").exists():
                if (fold.strategy, fold.seed, fold.fold) in wanted:
                    missing.append(d.name)
                continue
            records.append(json.loads((d / "record.json").read_text()))
            preds.extend(read_jsonl(d / "predictions.jsonl"))
            for v in variants:
                path = variant_predictions_path(d, v)
                if path.exists():
                    variant_preds[v].extend(read_jsonl(path))
                else:
                    variant_missing[v].append(d.name)
        extra = len(records) - (len(wanted) - len(missing))
        print(f"[collect] {name}: {len(wanted) - len(missing)}/{len(wanted)} configured items" +
              (f" (+{extra} outside the configured strategies, seeds or folds)" if extra else "") +
              (f", missing {missing}" if missing else ""))
        for v, absent in variant_missing.items():
            if absent:
                print(f"[collect] {name}: variant {v} missing in {len(absent)} finished items {absent}")
        if not records:
            continue
        write_json(out_root / name / "records.json", records)
        write_jsonl(out_root / name / "predictions.jsonl", preds)
        for v, rows in variant_preds.items():
            if rows:
                write_jsonl(variant_predictions_path(out_root / name, v), rows)
        mcc = [r["mcc_tuned"] for r in records]
        print(f"[collect] wrote {out_root / name / 'records.json'} (mean macro MCC over items {np.mean(mcc):.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/experiments/code_finetune.yml")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--collect", action="store_true", help="merge item records per model and exit")
    parser.add_argument("--list", action="store_true", help="print the items of this shard and their status")
    args, extra = parser.parse_known_args()
    cfg = load_config(args.config, parse_cli_overrides(extra))
    fcfg = cfg[SECTION]
    check_config(fcfg)
    run_name = str(cfg.get("run", {}).get("name", args.config.stem))
    out_root = resolve(fcfg["out_dir"]) / run_name

    folds = load_folds(resolve(fcfg["folds"]))
    fcfg["label_order"] = folds_label_order(folds, fcfg.get("label_order"))
    variant_paths = {str(k): resolve(v) for k, v in (fcfg.get("eval_variants") or {}).items()}
    absent = sorted(str(p) for p in variant_paths.values() if not p.exists())
    if absent and not args.collect:
        raise SystemExit(f"eval_variants files not found: {absent}")
    items = build_items(fcfg, folds)

    if args.collect:
        collect(fcfg, folds, out_root)
        return

    mine = shard(items, args.shard_index, args.shard_count)
    stop_on_stale(out_root, mine)
    todo = [(n, f) for n, f in mine if not (item_dir(out_root, n, f) / "record.json").exists()]
    print(f"[finetune] run={run_name} out={out_root} items={len(items)} shard={args.shard_index}/{args.shard_count} "
          f"mine={len(mine)} todo={len(todo)}", flush=True)
    # finished items are never retrained, missing variants are only reported
    lacking = {}
    for n, f in mine:
        d = item_dir(out_root, n, f)
        absent_variants = [v for v in variant_paths if not variant_predictions_path(d, v).exists()]
        if (d / "record.json").exists() and absent_variants:
            lacking[f"{n}/{d.name}"] = absent_variants
    if lacking:
        print(f"[finetune] {len(lacking)} finished items lack variant predictions (remove their record.json to "
              f"retrain them): {sorted(lacking.items())[:5]}{' ...' if len(lacking) > 5 else ''}", flush=True)
    if args.list:
        for n, f in mine:
            done = (item_dir(out_root, n, f) / "record.json").exists()
            print(f"  {'done' if done else 'todo'} {n} {f.strategy}_s{f.seed}_f{f.fold}")
        return
    if not todo:
        return
    if fcfg.get("subset") and any((fcfg.get("subset") or {}).values()):
        print(f"[finetune] WARNING subset active {fcfg['subset']}: smoke test only", flush=True)

    from transformers import AutoTokenizer

    if bool(fcfg["tf32"]) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = select_device(str(fcfg["device"]))
    entries = json.loads(resolve(fcfg["input_json"]).read_text())
    codes = {int(i): str(entries[int(i)]["code_snippet"]) for i in folds["meta"]["json_index"]}
    variant_codes = {}
    for variant, path in variant_paths.items():
        variant_entries = json.loads(path.read_text())
        # same length and order as input_json, only code_snippet and metamorphic changed
        check_variant_entries(entries, variant_entries, variant)
        variant_codes[variant] = {i: str(variant_entries[i]["code_snippet"]) for i in codes}
        print(f"[finetune] variant {variant}: {sum(variant_codes[variant][i] != codes[i] for i in codes)} of "
              f"{len(codes)} snippets differ from input_json", flush=True)
        del variant_entries
    losses = load_losses_module()
    shares = None
    if fcfg.get("review_shares"):
        shares_obj = json.loads(resolve(fcfg["review_shares"]).read_text())
        if list(shares_obj["meta"]["label_order"]) != list(fcfg["label_order"]):
            raise SystemExit(f"review_shares label order {shares_obj['meta']['label_order']} differs from "
                             f"{fcfg['label_order']}")
        shares = np.asarray(shares_obj["share"], dtype=np.float64)
        if len(shares) != len(entries):
            raise SystemExit(f"review_shares has {len(shares)} rows, input_json {len(entries)}")
        n_rev = np.asarray(shares_obj["n_reviewers"])
        y_all = label_matrix_from_json(entries, np.arange(len(entries)), shares.shape[1])
        clash = (n_rev > 0) & (((shares == 1.0) & (y_all == 0)) | ((shares == 0.0) & (y_all == 1)))
        if clash.any():
            raise SystemExit(f"{int(clash.sum())} unanimous review shares contradict the labels of input_json "
                             f"(first json_index {int(np.argwhere(clash)[0][0])})")

    # (tokenizer path, None for the base snippets or a variant name) -> token ids
    encoded_cache: dict[tuple[str, str | None], Encoded] = {}
    seq_cache: dict[tuple[str, str | None], SequenceSet] = {}

    def sequences(name: str, spec: dict[str, Any], variant: str | None) -> SequenceSet:
        path = str(spec["path"])
        if (path, variant) not in encoded_cache:
            tokenizer = AutoTokenizer.from_pretrained(str(resolve(fcfg["models_dir"]) / path))
            encoded_cache[(path, variant)] = Encoded(tokenizer, codes if variant is None else variant_codes[variant])
        if (name, variant) not in seq_cache:
            seq_cache[(name, variant)] = SequenceSet(encoded_cache[(path, variant)], spec)
        return seq_cache[(name, variant)]

    for name, fold in todo:
        out = item_dir(out_root, name, fold)
        if (out / "record.json").exists():
            continue
        spec = model_spec(fcfg, name)
        run_item(name=name, spec=spec, fold=fold, fcfg=fcfg, entries=entries, seqs=sequences(name, spec, None),
                 variant_seqs={v: sequences(name, spec, v) for v in variant_codes},
                 losses=losses, device=device, out=out, shares=shares)


if __name__ == "__main__":
    main()
