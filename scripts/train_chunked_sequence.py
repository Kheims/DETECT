"""Sequence models that read every snippet as consecutive 512-token chunks, on the shared folds.

Called by the pipeline for training.used_method=chunked_sequence. One call trains one model on one
(seed, fold) of the folds file built by the folds stage and writes record.json, predictions.jsonl
(validation and test probabilities) and metrics.json.

Models: lstm, bilstm, bilstm_attention, gru and cnn (regex tokens, embeddings learned on the task,
vocabulary of the training part) and codebert (subword tokens, fine-tuned end to end). Each chunk
goes through the same encoder and the chunk vectors are max-pooled before the head
(mlcq_graphs/models/chunked_sequence.py). max_chunks: 1 reads the first chunk only.

Scoring in metrics.json: each smell on its eligible snippets (methods for feature envy and long
method, classes for blob and data class), threshold tuned on the eligible validation snippets.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlcq_graphs.evaluation.cv import (  # noqa: E402
    fold_digest,
    iter_folds,
    label_matrix_from_json,
    load_folds,
    prediction_rows,
    write_jsonl,
)
from scripts.build_token_dataset import build_vocab, tokenize_code  # noqa: E402

RNN_MODELS = ("lstm", "bilstm", "bilstm_attention", "gru", "cnn")
SMELL_NAMES = ("feature_envy", "long_method", "blob", "data_class")
# keys of the normalized JSON labels, in the order of y
LABEL_KEYS = ("feature envy", "long method", "blob", "data class")
METHOD_SMELLS = {"feature envy", "long method"}
CLASS_SMELLS = {"blob", "data class"}


def _load_by_path(name: str, rel: str) -> Any:
    """Import one module file without running the package __init__ (it needs torch_geometric)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_chunked = _load_by_path("detect_chunked_sequence", "mlcq_graphs/models/chunked_sequence.py")
ChunkedRNN, ChunkedCNN = _chunked.ChunkedRNN, _chunked.ChunkedCNN
ChunkedEncoderClassifier = _chunked.ChunkedEncoderClassifier


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def native_bf16(cfg: dict[str, Any], device: torch.device) -> bool:
    """bf16 autocast only with allow_bf16 and a GPU with native bf16 (compute capability 8.0 or more), fp32 otherwise."""
    return bool(cfg.get("allow_bf16", False)) and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8


def versions() -> dict[str, str]:
    import sklearn
    out = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
           "sklearn": sklearn.__version__, "cuda": str(torch.version.cuda)}
    try:
        import transformers
        out["transformers"] = transformers.__version__
    except ImportError:
        pass
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ap_macro(y: np.ndarray, p: np.ndarray) -> float:
    vals = [average_precision_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(vals)) if vals else 0.0


def f1_macro_05(y: np.ndarray, p: np.ndarray) -> float:
    return float(f1_score(y, (p >= 0.5).astype(int), average="macro", zero_division=0))


def snippet_levels(entries: list[dict[str, Any]]) -> dict[int, str]:
    """method or class from the smells a snippet was reviewed for (mixed or none otherwise)."""
    out = {}
    for i, e in enumerate(entries):
        reviewed = {s for s in LABEL_KEYS if int(e.get("labels", {}).get(s, {}).get("vote_count", 0)) > 0}
        if reviewed and reviewed <= METHOD_SMELLS:
            out[i] = "method"
        elif reviewed and reviewed <= CLASS_SMELLS:
            out[i] = "class"
        else:
            out[i] = "mixed" if reviewed else "none"
    return out


def eligible_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """F1-max threshold on a 0..1 grid (step 0.01), lower threshold on ties, 0.5 without positives."""
    grid = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    if not y.sum():
        return 0.5
    pred = p.astype(np.float32)[:, None] >= grid
    tp = (pred * y[:, None]).sum(0)
    den = pred.sum(0) + y.sum()
    f = np.divide(2 * tp, den, out=np.zeros(len(grid)), where=den > 0)
    return float(grid[np.argmax(f)])


def eligible_metrics(parts: dict[str, np.ndarray], Y: dict[str, np.ndarray], probs: dict[str, np.ndarray],
                     level: dict[int, str]) -> dict[str, Any]:
    """Per smell: threshold tuned on the eligible validation snippets, scores on the eligible test snippets."""
    out = {}
    for j, smell in enumerate(SMELL_NAMES):
        lev = "method" if j < 2 else "class"
        mv = np.array([level[int(i)] == lev for i in parts["val"]])
        mt = np.array([level[int(i)] == lev for i in parts["test"]])
        thr = eligible_threshold(Y["val"][mv, j], probs["val"][mv, j])
        yt = Y["test"][mt, j]
        pt = probs["test"][mt, j]
        dec = (pt.astype(np.float32) >= np.float32(thr)).astype(int)
        out[smell] = {"threshold": thr,
                      "mcc": float(matthews_corrcoef(yt, dec)) if len(set(yt)) > 1 or dec.any() else 0.0,
                      "f1": float(f1_score(yt, dec, zero_division=0)),
                      "precision": float(precision_score(yt, dec, zero_division=0)),
                      "recall": float(recall_score(yt, dec, zero_division=0)),
                      "pr_auc": float(average_precision_score(yt, pt)) if 0 < yt.sum() < len(yt) else 0.0,
                      "n_test": int(mt.sum()), "positives_test": int(yt.sum())}
    return out


# ---------------------------------------------------------------- inputs


class RegexChunks:
    """Regex tokens of every snippet, chunked with the vocabulary of one training part."""

    def __init__(self, tokens: dict[int, list[str]], train_idx: np.ndarray, cap: int, chunk_len: int,
                 min_freq: int, max_vocab: int) -> None:
        self.vocab = build_vocab([t for i in train_idx for t in tokens[int(i)]], min_freq, max_vocab)
        unk = self.vocab["<UNK>"]
        self.cap = cap
        self.chunk_len = chunk_len
        self.n_chunks_full: dict[int, int] = {}
        self.chunks: dict[int, np.ndarray] = {}
        for i, toks in tokens.items():
            ids = [self.vocab.get(t, unk) for t in toks]
            n_full = max(1, math.ceil(len(ids) / chunk_len))
            n = min(n_full, cap)
            arr = np.zeros((n, chunk_len), dtype=np.int64)
            for c in range(n):
                part = ids[c * chunk_len:(c + 1) * chunk_len]
                arr[c, :len(part)] = part
            self.n_chunks_full[i] = n_full
            self.chunks[i] = arr

    def batch(self, idx: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
        c = max(self.chunks[int(i)].shape[0] for i in idx)
        ids = np.zeros((len(idx), c, self.chunk_len), dtype=np.int64)
        mask = np.zeros((len(idx), c), dtype=bool)
        for k, i in enumerate(idx):
            a = self.chunks[int(i)]
            ids[k, :a.shape[0]] = a
            mask[k, :a.shape[0]] = True
        return torch.as_tensor(ids, device=device), torch.as_tensor(mask, device=device)


class BertChunks:
    """Subword ids of every snippet, as <s> + (chunk_len - 2) tokens + </s> chunks."""

    def __init__(self, ids: dict[int, list[int]], tok: Any, cap: int, chunk_len: int) -> None:
        self.cap = cap
        self.chunk_len = chunk_len
        content = chunk_len - 2
        self.n_chunks_full: dict[int, int] = {}
        self.chunks: dict[int, np.ndarray] = {}
        self.masks: dict[int, np.ndarray] = {}
        for i, seq in ids.items():
            n_full = max(1, math.ceil(len(seq) / content))
            n = min(n_full, cap)
            arr = np.full((n, chunk_len), tok.pad_token_id, dtype=np.int64)
            att = np.zeros((n, chunk_len), dtype=np.int64)
            for c in range(n):
                part = [tok.cls_token_id] + seq[c * content:(c + 1) * content] + [tok.sep_token_id]
                arr[c, :len(part)] = part
                att[c, :len(part)] = 1
            self.n_chunks_full[i] = n_full
            self.chunks[i] = arr
            self.masks[i] = att

    def size(self, i: int) -> int:
        return self.chunks[int(i)].shape[0]

    def batch(self, idx: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
        c = max(self.size(i) for i in idx)
        ids = np.ones((len(idx), c, self.chunk_len), dtype=np.int64)
        att = np.zeros((len(idx), c, self.chunk_len), dtype=np.int64)
        mask = np.zeros((len(idx), c), dtype=bool)
        for k, i in enumerate(idx):
            a = self.chunks[int(i)]
            ids[k, :a.shape[0]] = a
            att[k, :a.shape[0]] = self.masks[int(i)]
            mask[k, :a.shape[0]] = True
        return (torch.as_tensor(ids, device=device), torch.as_tensor(att, device=device),
                torch.as_tensor(mask, device=device))


def groups_by_budget(idx: np.ndarray, size: Any, budget: int) -> list[np.ndarray]:
    """Consecutive groups of snippets whose total chunk count stays within the budget."""
    out, cur, tot = [], [], 0
    for i in idx:
        s = size(i)
        if cur and tot + s > budget:
            out.append(np.array(cur))
            cur, tot = [], 0
        cur.append(i)
        tot += s
    if cur:
        out.append(np.array(cur))
    return out


def chunk_stats(data: Any, parts: dict[str, np.ndarray]) -> dict[str, Any]:
    allidx = np.concatenate(list(parts.values()))
    full = np.array([data.n_chunks_full[int(i)] for i in allidx])
    used = np.minimum(full, data.cap)
    return {"cap": int(data.cap), "mean_chunks_used": float(used.mean()), "share_multi_chunk": float((full > 1).mean()),
            "share_beyond_cap": float((full > data.cap).mean()), "max_chunks_full": int(full.max())}


# ---------------------------------------------------------------- training


def train_rnn(cfg: dict[str, Any], kind: str, data: RegexChunks, parts: dict[str, np.ndarray],
              Y: dict[str, np.ndarray], seed: int, epochs: int, device: torch.device,
              losses: Any) -> tuple[nn.Module, list[dict[str, Any]], int]:
    """Focal loss with inverse-frequency class weights (or BCE), early stopping on the validation macro F1 at 0.5."""
    num_labels = int(cfg["num_labels"])
    set_seed(seed)
    if kind == "cnn":
        model = ChunkedCNN(len(data.vocab), int(cfg["embed_dim"]), num_labels, int(cfg["num_filters"]),
                           tuple(int(s) for s in cfg["filter_sizes"]), float(cfg["dropout"])).to(device)
    else:
        model = ChunkedRNN(kind, len(data.vocab), int(cfg["embed_dim"]), int(cfg["hidden_dim"]), num_labels,
                           int(cfg["num_layers"]), float(cfg["dropout"])).to(device)
    y_tr = Y["train"].astype(np.float32)
    if cfg["loss"] == "focal":
        counts = torch.as_tensor(Y["train"].astype(np.float32).sum(0)).clamp(min=1)
        pos_weight = len(y_tr) / (num_labels * counts)
        loss_fn = losses.FocalLoss(gamma=float(cfg["focal_gamma"]), class_weight=pos_weight.float()).to(device)
    elif cfg["loss"] == "bce":
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:
        raise ValueError(f"training.loss must be focal or bce for {kind}, got {cfg['loss']!r}")
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    best, best_state, best_epoch, no_improve, hist = -1.0, None, -1, 0, []
    rng = np.random.default_rng(seed)
    bs = int(cfg["batch_size"])
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = rng.permutation(len(parts["train"]))
        tot, seen = 0.0, 0
        for b in range(0, len(perm), bs):
            pos = perm[b:b + bs]
            ids, mask = data.batch(parts["train"][pos], device)
            y = torch.as_tensor(y_tr[pos], device=device)
            opt.zero_grad()
            loss = loss_fn(model(ids, mask), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()
            tot += float(loss.item()) * len(pos)
            seen += len(pos)
        pv = predict_rnn(model, data, parts["val"], device, int(cfg["eval_batch_size"]))
        hist.append({"epoch": epoch, "train_loss": tot / seen, "val_f1_macro_05": f1_macro_05(Y["val"], pv),
                     "val_ap_macro": ap_macro(Y["val"], pv), "seconds": time.perf_counter() - t0})
        print(f"  epoch {epoch} loss={tot / seen:.4f} val_f1@0.5={hist[-1]['val_f1_macro_05']:.4f} "
              f"val_ap={hist[-1]['val_ap_macro']:.4f}", flush=True)
        metric = hist[-1]["val_f1_macro_05"]
        if metric > best:
            best, best_epoch, no_improve = metric, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= int(cfg["patience"]):
                break
    model.load_state_dict(best_state)
    return model, hist, best_epoch


@torch.no_grad()
def predict_rnn(model: nn.Module, data: RegexChunks, idx: np.ndarray, device: torch.device, bs: int) -> np.ndarray:
    model.eval()
    out = []
    for b in range(0, len(idx), bs):
        ids, mask = data.batch(idx[b:b + bs], device)
        out.append(torch.sigmoid(model(ids, mask)).float().cpu().numpy())
    return np.concatenate(out)


def train_bert(cfg: dict[str, Any], data: BertChunks, parts: dict[str, np.ndarray], Y: dict[str, np.ndarray],
               seed: int, epochs: int, device: torch.device, source: str,
               amp: Any) -> tuple[nn.Module, list[dict[str, Any]], int]:
    """AdamW with linear warmup, BCE, early stopping on the validation macro average precision."""
    from transformers import AutoModel, get_linear_schedule_with_warmup

    set_seed(seed)
    encoder = AutoModel.from_pretrained(source, dtype=torch.float32)
    if data.cap > 1 and bool(cfg["gradient_checkpointing"]):
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = ChunkedEncoderClassifier(encoder, int(cfg["num_labels"]), float(cfg["dropout"]),
                                     float(encoder.config.initializer_range)).to(device)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "norm" in name.lower() else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": float(cfg["weight_decay"])},
                             {"params": no_decay, "weight_decay": 0.0}], lr=float(cfg["lr"]))
    bs = int(cfg["batch_size"])
    n = len(parts["train"])
    steps = math.ceil(n / bs) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(math.ceil(float(cfg["warmup_ratio"]) * steps)), steps)
    if cfg["loss"] != "bce":
        raise ValueError(f"training.loss must be bce for codebert, got {cfg['loss']!r}")
    loss_fn = nn.BCEWithLogitsLoss()
    y_tr = Y["train"].astype(np.float32)
    best, best_state, best_epoch, no_improve, hist = -float("inf"), None, -1, 0, []
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = np.random.default_rng([seed, epoch]).permutation(n)
        tot, seen = 0.0, 0
        for b in range(0, n, bs):
            pos = perm[b:b + bs]
            idx = parts["train"][pos]
            opt.zero_grad(set_to_none=True)
            # micro-batches bound the number of chunks in one forward pass, the gradient is the batch mean
            for g in groups_by_budget(np.arange(len(idx)), lambda k: data.size(idx[k]), int(cfg["max_chunks_per_forward"])):
                ids, att, mask = data.batch(idx[g], device)
                y = torch.as_tensor(y_tr[pos[g]], device=device)
                with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None):
                    logits = model(ids, att, mask)
                loss = loss_fn(logits.float(), y)
                (loss * len(g) / len(idx)).backward()
                tot += float(loss.item()) * len(g)
                seen += len(g)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()
            sched.step()
        pv = predict_bert(model, data, parts["val"], device, amp, int(cfg["eval_batch_size"]))
        metric = ap_macro(Y["val"], pv)
        hist.append({"epoch": epoch, "train_loss": tot / seen, "val_ap_macro": metric, "val_f1_macro_05": f1_macro_05(Y["val"], pv),
                     "seconds": time.perf_counter() - t0})
        print(f"  epoch {epoch} loss={tot / seen:.4f} val_ap={metric:.4f}", flush=True)
        if metric > best + float(cfg["min_delta"]):
            best, best_epoch, no_improve = metric, epoch, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= int(cfg["patience"]):
                break
    model.load_state_dict(best_state)
    return model, hist, best_epoch


@torch.no_grad()
def predict_bert(model: nn.Module, data: BertChunks, idx: np.ndarray, device: torch.device, amp: Any,
                 budget: int) -> np.ndarray:
    model.eval()
    out = []
    for g in groups_by_budget(idx, data.size, budget):
        ids, att, mask = data.batch(g, device)
        with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None):
            out.append(torch.sigmoid(model(ids, att, mask).float()).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- driver


def run_chunked_sequence(training_cfg: dict[str, Any], input_json: str | Path, folds_json: str | Path,
                         output_dir: str | Path) -> dict[str, Any]:
    """Train training_cfg.model on fold training_cfg.cv_fold of seed training_cfg.seed and write the outputs."""
    cfg = dict(training_cfg)
    model_name = str(cfg["model"])
    if model_name not in RNN_MODELS + ("codebert",):
        raise ValueError(f"training.model must be one of {RNN_MODELS + ('codebert',)}, got {model_name!r}")
    seed, fold_id = int(cfg["seed"]), int(cfg["cv_fold"])
    num_labels = int(cfg["num_labels"])
    chunk_len, cap = int(cfg["chunk_len"]), int(cfg["max_chunks"])
    variant = "chunked" if cap > 1 else "truncated"
    out = Path(output_dir)

    device_name = str(cfg.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    input_path = resolve(input_json)
    entries = json.loads(input_path.read_text())
    folds = load_folds(resolve(folds_json))
    expected_md5 = folds["meta"].get("input_json_md5")
    if expected_md5 and hashlib.md5(input_path.read_bytes()).hexdigest() != expected_md5:
        raise SystemExit(f"{input_path} is not the JSON the folds were built on ({folds_json})")
    found = list(iter_folds(folds, strategies=["kfold"], seeds=[seed], fold_ids=[fold_id]))
    if len(found) != 1:
        raise SystemExit(f"{folds_json} has no kfold split for seed {seed} and fold {fold_id}")
    fold = found[0]
    index = [int(i) for i in folds["meta"]["json_index"]]
    level = snippet_levels(entries)
    losses = _load_by_path("detect_losses", "mlcq_graphs/training/losses.py")

    parts = {"train": fold.train, "val": fold.val, "test": fold.test}
    if cfg.get("subset"):
        parts = {p: idx[:int(cfg["subset"])] for p, idx in parts.items()}
    Y = {p: label_matrix_from_json(entries, idx, num_labels) for p, idx in parts.items()}
    epochs = int(cfg["epochs"])
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    amp = None
    if model_name in RNN_MODELS:
        tokens = {i: tokenize_code(entries[i]["code_snippet"], max_tokens=10**9) for i in index}
        data = RegexChunks(tokens, fold.train, cap, chunk_len, int(cfg["min_freq"]), int(cfg["max_vocab"]))
        stats = chunk_stats(data, parts)
        print(f"[{model_name}/{variant}] seed={seed} fold={fold_id} vocab={len(data.vocab)} {stats}", flush=True)
        model, hist, best_epoch = train_rnn(cfg, model_name, data, parts, Y, seed, epochs, device, losses)
        probs = {p: predict_rnn(model, data, parts[p], device, int(cfg["eval_batch_size"])) for p in ("val", "test")}
    else:
        from transformers import AutoTokenizer

        source = str(resolve(cfg["pretrained"])) if resolve(cfg["pretrained"]).exists() else str(cfg["pretrained"])
        tok = AutoTokenizer.from_pretrained(source)
        bert_ids = {i: tok(entries[i]["code_snippet"], add_special_tokens=False)["input_ids"] for i in index}
        data = BertChunks(bert_ids, tok, cap, chunk_len)
        stats = chunk_stats(data, parts)
        print(f"[{model_name}/{variant}] seed={seed} fold={fold_id} {stats}", flush=True)
        amp = torch.bfloat16 if native_bf16(cfg, device) else None
        model, hist, best_epoch = train_bert(cfg, data, parts, Y, seed, epochs, device, source, amp)
        probs = {p: predict_bert(model, data, parts[p], device, amp, int(cfg["eval_batch_size"])) for p in ("val", "test")}
    for p, arr in probs.items():
        if not np.isfinite(arr).all():
            raise SystemExit(f"non-finite probabilities in {out} ({p})")
    seconds = time.perf_counter() - t0
    peak_mb = (torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None

    rows = []
    for p in ("val", "test"):
        rows += prediction_rows(model=f"{model_name}_{variant}", strategy="kfold", seed=seed, fold=fold_id, part=p,
                                indices=parts[p], y=Y[p], probs=probs[p])
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "predictions.jsonl", rows)
    eligible = eligible_metrics(parts, Y, probs, level)
    record = {"model": model_name, "variant": variant, "train_target": "consensus", "strategy": "kfold",
              "seed": seed, "fold": fold_id, "fold_digest": fold_digest(fold), "epochs_budget": epochs,
              "epochs_run": len(hist), "best_epoch": best_epoch, "history": hist, "chunks": stats,
              "settings": cfg, "seconds": seconds, "peak_gpu_mb": peak_mb,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "bf16": amp is not None, "input_json": str(input_json), "folds_json": str(folds_json),
              "versions": versions(), "test_ap_macro": ap_macro(Y["test"], probs["test"]), "eligible_test": eligible}
    (out / "record.json").write_text(json.dumps(record, indent=2))
    # the schema of the GNN metrics.json that the ablation records read
    metrics = {"best_epoch": best_epoch,
               "test_tuned": {"f1_macro": float(np.mean([v["f1"] for v in eligible.values()])),
                              "mcc_macro": float(np.mean([v["mcc"] for v in eligible.values()])),
                              "pr_auc_macro": float(np.mean([v["pr_auc"] for v in eligible.values()])),
                              "per_label": eligible},
               "profiling": {"train_duration_sec": seconds, "peak_gpu_memory_mb": peak_mb or 0.0}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"done {out} best_epoch={best_epoch} test_ap={record['test_ap_macro']:.4f} "
          f"mcc_macro={metrics['test_tuned']['mcc_macro']:.4f} seconds={seconds:.0f}", flush=True)
    return metrics
