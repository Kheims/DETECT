"""Embed the snippets of the shared folds with frozen code models (MLCQ or DACOS).

Writes <out_dir>/<name>/emb.npy, index.json and meta.json for every model of the config.

Usage:
    python scripts/cv/embed_code_models.py --config config/experiments/code_embeddings.yml
    python scripts/cv/embed_code_models.py --config ... --shard-index 0 --shard-count 4
    python scripts/cv/embed_code_models.py --config config/experiments/code_embeddings.yml \\
        --code_embeddings.input_json=data/derived/metamorphic/mlcq/<transform>.normalized.json \\
        --code_embeddings.base_json=data/MLCQCodeSmellSamples.normalized.json \\
        --code_embeddings.out_dir=artifacts/embeddings_variants/<transform>
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.evaluation.cv import check_variant_entries, is_variant, load_folds, shard  # noqa: E402

KINDS = {"encoder", "decoder", "embedding"}
POOLINGS = {"cls", "mean", "last_token", "model_output"}
LONG_INPUT = {"truncate", "chunk_mean"}
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
# Errors that custom model code written for an older transformers raises at forward time.
CUSTOM_CODE_ERRORS = (AttributeError, ImportError, NameError, TypeError)
# settings that change the embedding values, a change invalidates a finished model
SETTING_KEYS = ("path", "loader", "kind", "pooling", "layers", "max_length", "long_input", "stride", "max_chunks",
                "instruction", "prefix_tokens", "whitespace", "padding_side", "normalize", "dtype",
                "trust_remote_code", "forward_kwargs")


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.save(fh, array)
    os.replace(tmp, path)


def write_json_atomic(path: Path, obj: Any) -> None:
    write_atomic(path, (json.dumps(obj, indent=1) + "\n").encode())


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def model_spec(defaults: dict[str, Any], entry: dict[str, Any], max_length_cap: int | None) -> dict[str, Any]:
    spec = {**defaults, **entry}
    if max_length_cap:
        spec["max_length"] = min(int(spec["max_length"]), int(max_length_cap))
    spec["pooling"] = as_list(spec["pooling"])
    spec["layers"] = [int(k) for k in as_list(spec.get("layers", -1))]
    spec["prefix_tokens"] = list(spec.get("prefix_tokens") or [])
    spec["forward_kwargs"] = dict(spec.get("forward_kwargs") or {})
    if spec["kind"] not in KINDS:
        raise ValueError(f"{spec['name']}: kind must be one of {sorted(KINDS)}")
    if any(p not in POOLINGS for p in spec["pooling"]):
        raise ValueError(f"{spec['name']}: pooling must be in {sorted(POOLINGS)}")
    if "model_output" in spec["pooling"] and (len(spec["pooling"]) > 1 or spec["layers"] != [-1]):
        raise ValueError(f"{spec['name']}: model_output pooling takes no other pooling or layer")
    if spec["long_input"] not in LONG_INPUT:
        raise ValueError(f"{spec['name']}: long_input must be one of {sorted(LONG_INPUT)}")
    if spec["dtype"] not in DTYPES:
        raise ValueError(f"{spec['name']}: dtype must be one of {sorted(DTYPES)}")
    return spec


def fingerprint(spec: dict[str, Any], indices: list[int]) -> str:
    payload = {"settings": {k: spec.get(k) for k in SETTING_KEYS},
               "indices": hashlib.sha256(json.dumps(indices).encode()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def texts_digest(texts: list[str]) -> str:
    """Digest of the embedded snippets, kept out of the fingerprint."""
    return hashlib.sha256(json.dumps(texts).encode()).hexdigest()


def same_texts(meta: dict[str, Any], digest: str, variant: bool) -> bool:
    """Whether a finished folder holds the snippets of this run (no digest means base snippets)."""
    recorded = meta.get("texts_sha256")
    return not variant if recorded is None else recorded == digest


def is_done(out_dir: Path, fp: str, digest: str, variant: bool) -> bool:
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    return (meta.get("status") == "ok" and meta.get("fingerprint") == fp
            and same_texts(meta, digest, variant)
            and all((out_dir / f).exists() for f in [*meta.get("outputs", {}), "index.json"]))


def reduced_run_conflict(out_dir: Path, indices: list[int], max_length_cap: Any, digest: str,
                         variant: bool) -> str | None:
    """Reason a finished folder must not be replaced by this run, or None."""
    try:
        meta = json.loads((out_dir / "meta.json").read_text())
        done_index = json.loads((out_dir / "index.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if meta.get("status") != "ok":
        return None
    if len(indices) < len(done_index):
        return f"it holds {len(done_index)} snippets and this run has {len(indices)}"
    if max_length_cap and not meta.get("max_length_cap"):
        return "it was computed without max_length_cap"
    if indices == done_index and not same_texts(meta, digest, variant):
        return f"it holds embeddings of other snippets (input_json {meta.get('input_json', 'of a base run')})"
    return None


def special_frame(tokenizer: Any) -> tuple[list[int], list[int]]:
    """Special token ids the tokenizer puts before and after a single sequence."""
    bare = tokenizer("x", add_special_tokens=False)["input_ids"]
    full = tokenizer("x", add_special_tokens=True)["input_ids"]
    for k in range(len(full) - len(bare) + 1):
        if full[k:k + len(bare)] == bare:
            return list(full[:k]), list(full[k + len(bare):])
    raise ValueError("cannot locate the special tokens around a probe string")


def windows_of(ids: list[int], budget: int, mode: str, stride: int,
               max_chunks: int | None) -> tuple[list[list[int]], bool]:
    """Token windows of one snippet and whether some tokens were left out."""
    if len(ids) <= budget:
        return [ids], False
    if mode == "truncate":
        return [ids[:budget]], True
    step = budget - stride
    if step <= 0:
        raise ValueError(f"stride {stride} must be smaller than the token budget {budget}")
    starts = list(range(0, len(ids) - stride, step))
    if max_chunks and len(starts) > int(max_chunks):
        starts = starts[: int(max_chunks)]
    return [ids[s:s + budget] for s in starts], starts[-1] + budget < len(ids)


def make_batches(lengths: list[int], batch_size: int, max_batch_tokens: int | None) -> list[list[int]]:
    """Batches of sequence positions, longest first, bounded in rows and padded tokens."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches: list[list[int]] = []
    current: list[int] = []
    width = 0
    for i in order:
        new_width = max(width, lengths[i])
        too_many = len(current) + 1 > batch_size
        too_wide = bool(max_batch_tokens) and new_width * (len(current) + 1) > int(max_batch_tokens)
        if current and (too_many or too_wide):
            batches.append(current)
            current, new_width = [], lengths[i]
        current.append(i)
        width = new_width
    if current:
        batches.append(current)
    return batches


def pool(hidden: torch.Tensor, mask: torch.Tensor, how: str) -> torch.Tensor:
    """Pool token states with an attention mask, for left or right padding."""
    hidden = hidden.float()
    if how == "mean":
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    # argmax on float32: integer argmax is not implemented on every backend
    if how == "cls":
        return hidden[rows, mask.float().argmax(dim=1)]
    if how == "last_token":
        positions = torch.arange(mask.shape[1], device=mask.device, dtype=torch.float32).unsqueeze(0)
        return hidden[rows, (mask.float() * positions).argmax(dim=1)]
    raise ValueError(f"unknown pooling: {how}")


class EncoderWithProjection(torch.nn.Module):
    """First-token state of an encoder through a linear projection, L2 normalised."""

    def __init__(self, encoder: Any, proj: torch.nn.Linear) -> None:
        super().__init__()
        self.encoder, self.proj, self.config = encoder, proj, encoder.config

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs).last_hidden_state
        return torch.nn.functional.normalize(self.proj(hidden[:, 0, :]), dim=-1)


def read_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if (path / "model.safetensors").exists():
        from safetensors.torch import load_file

        return load_file(str(path / "model.safetensors"))
    return torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)


def load_t5_encoder_proj(path: Path, dtype: torch.dtype) -> EncoderWithProjection:
    """CodeT5+ embedding model rebuilt from library classes."""
    from transformers import T5Config, T5EncoderModel

    raw = json.loads((path / "config.json").read_text())
    drop = {"auto_map", "architectures", "model_type", "_name_or_path", "transformers_version", "torch_dtype"}
    config = T5Config(**{k: v for k, v in raw.items() if k not in drop})
    state = read_state_dict(path)
    encoder = T5EncoderModel(config)
    missing, unexpected = encoder.load_state_dict(
        {k: v for k, v in state.items() if not k.startswith("proj.")}, strict=False)
    if missing or unexpected:
        raise ValueError(f"T5 encoder weights do not match: missing={missing} unexpected={unexpected}")
    proj = torch.nn.Linear(config.d_model, int(raw["embed_dim"]))
    proj.load_state_dict({"weight": state["proj.weight"], "bias": state["proj.bias"]})
    return EncoderWithProjection(encoder, proj).to(dtype)


def load_model(spec: dict[str, Any], models_dir: Path, device: torch.device) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer

    path = models_dir / str(spec["path"])
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"no config.json under {path}")
    trust = bool(spec["trust_remote_code"])
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust)
    loader = str(spec.get("loader", "auto"))
    if loader == "auto":
        model = AutoModel.from_pretrained(path, trust_remote_code=trust, dtype=DTYPES[spec["dtype"]])
    elif loader == "t5_encoder_proj":
        model = load_t5_encoder_proj(path, DTYPES[spec["dtype"]])
    else:
        raise ValueError(f"unknown loader: {loader}")
    model.to(device).eval()
    return tokenizer, model


def num_layers(model: Any) -> int | None:
    cfg = model.config
    for attr in ("num_hidden_layers", "num_layers", "n_layer"):
        value = getattr(cfg, attr, None)
        if isinstance(value, int):
            return value
    return None


def embed_model(spec: dict[str, Any], tokenizer: Any, model: Any, texts: list[str], device: torch.device,
                log_every: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    # long snippets are cut here, so silence the tokenizer length warning
    tokenizer.model_max_length = int(1e12)
    if spec.get("padding_side"):
        tokenizer.padding_side = str(spec["padding_side"])
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.unk_token_id
    if pad_id is None:
        raise ValueError("tokenizer has no pad, eos or unk token")
    left = tokenizer.padding_side == "left"

    head, tail = special_frame(tokenizer)
    prefix_ids = tokenizer.convert_tokens_to_ids(spec["prefix_tokens"]) if spec["prefix_tokens"] else []
    unknown = [t for t, i in zip(spec["prefix_tokens"], prefix_ids) if i is None or i == tokenizer.unk_token_id]
    if unknown:
        raise ValueError(f"prefix tokens not in the vocabulary: {unknown}")
    instr_ids = tokenizer(spec["instruction"], add_special_tokens=False)["input_ids"] if spec.get("instruction") else []
    frame = len(head) + len(prefix_ids) + len(instr_ids) + len(tail)
    budget = int(spec["max_length"]) - frame
    if budget <= 0:
        raise ValueError(f"max_length {spec['max_length']} leaves no room for code after {frame} framing tokens")

    t1 = time.perf_counter()
    if spec.get("whitespace", "keep") == "collapse":
        texts = [" ".join(t.split()) for t in texts]
    code_ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
    lengths = np.array([len(ids) for ids in code_ids])
    owner, seqs = [], []
    n_cut = 0
    for row, ids in enumerate(code_ids):
        wins, cut = windows_of(list(ids), budget, spec["long_input"], int(spec.get("stride", 0)),
                               spec.get("max_chunks"))
        n_cut += int(cut)
        for w in wins:
            seq = head + prefix_ids + instr_ids + w + tail
            seqs.append(seq if seq else [pad_id])
            owner.append(row)
    owner_arr = np.asarray(owner)
    windows_per_snippet = np.bincount(owner_arr, minlength=len(texts))
    t_tok = time.perf_counter() - t1

    n_layers = num_layers(model)
    outputs: list[tuple[str, int | None, str]] = []
    for layer in spec["layers"]:
        for how in spec["pooling"]:
            if how == "model_output":
                outputs.append(("emb.npy", None, how))
                continue
            if n_layers is None and layer != -1:
                raise ValueError("cannot resolve layer indices: the config has no layer count")
            k = None if n_layers is None else (layer if layer >= 0 else n_layers + 1 + layer)
            if k is not None and not 0 <= k <= n_layers:
                raise ValueError(f"layer {layer} outside [0, {n_layers}]")
            fname = "emb.npy" if not outputs else f"emb_{how}_layer{k}.npy"
            outputs.append((fname, k, how))
    need_hidden = any(k is not None and n_layers is not None and k != n_layers for _, k, _ in outputs)

    batches = make_batches([len(s) for s in seqs], int(spec["batch_size"]), spec.get("max_batch_tokens"))
    pooled: dict[str, np.ndarray] = {}
    oom_splits = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def run(batch: list[int]) -> None:
        nonlocal oom_splits
        width = max(len(seqs[i]) for i in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for r, i in enumerate(batch):
            s = torch.tensor(seqs[i], dtype=torch.long)
            sl = slice(width - len(s), width) if left else slice(0, len(s))
            ids[r, sl] = s
            mask[r, sl] = 1
        ids, mask = ids.to(device), mask.to(device)
        kwargs = dict(spec["forward_kwargs"])
        if need_hidden:
            kwargs["output_hidden_states"] = True
        try:
            out = model(input_ids=ids, attention_mask=mask, **kwargs)
        except torch.OutOfMemoryError:
            if len(batch) == 1:
                raise
            out = None
        if out is None:
            # retried outside the except block so that the failed activations are released
            del ids, mask
            free_memory(device)
            oom_splits += 1
            half = len(batch) // 2
            run(batch[:half])
            run(batch[half:])
            return
        for fname, k, how in outputs:
            if how == "model_output":
                if not torch.is_tensor(out):
                    raise TypeError(f"model_output pooling needs a tensor output, got {type(out).__name__}")
                vec = out.float()
            else:
                last = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                hidden = last if k is None or k == n_layers else out.hidden_states[k]
                vec = pool(hidden, mask, how)
            if spec.get("normalize"):
                vec = torch.nn.functional.normalize(vec, dim=-1)
            arr = vec.cpu().numpy().astype(np.float32)
            if fname not in pooled:
                pooled[fname] = np.zeros((len(seqs), arr.shape[1]), dtype=np.float32)
            pooled[fname][batch] = arr

    t2 = time.perf_counter()
    with torch.inference_mode():
        for b, batch in enumerate(batches):
            run(batch)
            if log_every and ((b + 1) % log_every == 0 or b + 1 == len(batches)):
                print(f"[embed] {spec['name']} batch {b + 1}/{len(batches)} "
                      f"width={max(len(seqs[i]) for i in batch)} {time.perf_counter() - t2:.0f}s", flush=True)
    t_embed = time.perf_counter() - t2

    results: dict[str, np.ndarray] = {}
    for fname, per_window in pooled.items():
        sums = np.zeros((len(texts), per_window.shape[1]), dtype=np.float64)
        np.add.at(sums, owner_arr, per_window.astype(np.float64))
        emb = sums / windows_per_snippet[:, None]
        if spec.get("normalize"):
            emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        results[fname] = emb.astype(np.float32)

    info = {
        "outputs": {f: {"layer": k, "pooling": how, "dim": int(results[f].shape[1])} for f, k, how in outputs},
        "num_layers": n_layers,
        "n_truncated": int(n_cut),
        "n_chunked": int((windows_per_snippet > 1).sum()),
        "n_windows": int(len(seqs)),
        "max_windows_per_snippet": int(windows_per_snippet.max()),
        "code_token_length": {
            "mean": float(lengths.mean()),
            **{f"p{q}": float(np.percentile(lengths, q)) for q in (50, 75, 90, 95, 99)},
            "max": int(lengths.max()),
        },
        "token_budget_per_window": budget,
        "framing": {"head": head, "prefix": prefix_ids, "instruction_tokens": len(instr_ids), "tail": tail},
        "pad_token_id": int(pad_id),
        "padding_side": tokenizer.padding_side,
        "n_batches": len(batches),
        "n_oom_splits": oom_splits,
        "peak_gpu_memory_mib": (round(torch.cuda.max_memory_allocated(device) / 2**20)
                                if device.type == "cuda" else None),
        "seconds": {"tokenize": round(t_tok, 2), "embed": round(t_embed, 2)},
    }
    return results, info


def free_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--force", action="store_true",
                        help="recompute models that are already done, even with fewer snippets than before")
    args, extra = parser.parse_known_args()
    cfg = load_config(args.config, parse_cli_overrides(extra))
    ecfg = cfg["code_embeddings"]

    folds = load_folds(resolve(ecfg["folds"]))
    indices = [int(i) for i in folds["meta"]["json_index"]]
    if ecfg.get("limit"):
        indices = indices[: int(ecfg["limit"])]
    entries = json.loads(resolve(ecfg["input_json"]).read_text())
    # a variant input must follow the variant interface against its base JSON
    variant = is_variant(entries)
    base_json = ecfg.get("base_json")
    if variant and (not base_json or resolve(base_json) == resolve(ecfg["input_json"])):
        raise SystemExit(f"{ecfg['input_json']} is a metamorphic variant (its entries carry a metamorphic field): "
                         "set code_embeddings.base_json to its base normalized JSON")
    if base_json and resolve(base_json) != resolve(ecfg["input_json"]):
        n_changed = check_variant_entries(json.loads(resolve(base_json).read_text()), entries,
                                          str(ecfg["input_json"]))
        print(f"[embed] variant of {base_json}: {n_changed} of {len(entries)} snippets differ", flush=True)
    texts = [str(entries[i]["code_snippet"]) for i in indices]
    digest = texts_digest(texts)
    models_dir = resolve(ecfg["models_dir"])
    out_root = resolve(ecfg["out_dir"])
    device = pick_device(str(ecfg.get("device", "auto")))

    only = set(ecfg.get("only") or [])
    specs = [model_spec(ecfg.get("defaults", {}), m, ecfg.get("max_length_cap"))
             for m in ecfg["models"] if not only or m["name"] in only]
    if only - {s["name"] for s in specs}:
        raise SystemExit(f"unknown model names in only: {sorted(only - {s['name'] for s in specs})}")
    specs = shard(specs, args.shard_index, args.shard_count)
    print(f"[embed] device={device} snippets={len(texts)} models={[s['name'] for s in specs]}", flush=True)
    # a smaller run or a run on other snippets must not replace finished embeddings
    conflicts = {s["name"]: why for s in specs
                 if (why := reduced_run_conflict(out_root / s["name"], indices, ecfg.get("max_length_cap"), digest,
                                                 variant))}
    if conflicts and not args.force:
        raise SystemExit(f"refusing to replace finished embeddings under {out_root}: {conflicts}; "
                         "set code_embeddings.out_dir to another folder, or pass --force")

    import transformers

    env = {"device": str(device), "torch": torch.__version__, "transformers": transformers.__version__,
           "python": platform.python_version(), "host": socket.gethostname(),
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    failed = []
    for spec in specs:
        out_dir = out_root / spec["name"]
        fp = fingerprint(spec, indices)
        if not args.force and is_done(out_dir, fp, digest, variant):
            print(f"[embed] {spec['name']} already done, skipping", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        settings = {k: spec.get(k) for k in SETTING_KEYS}
        base = {"name": spec["name"], "fingerprint": fp, "settings": settings,
                "batch_size": spec["batch_size"], "max_batch_tokens": spec.get("max_batch_tokens"),
                "limit": ecfg.get("limit"), "max_length_cap": ecfg.get("max_length_cap"),
                "n_snippets": len(indices), "input_json": str(ecfg["input_json"]), "texts_sha256": digest,
                **({"base_json": str(base_json)} if variant else {}), **env}
        started = time.perf_counter()
        print(f"[embed] {spec['name']} start ({spec['path']})", flush=True)
        model = tokenizer = None
        error_meta: dict[str, Any] | None = None
        try:
            tokenizer, model = load_model(spec, models_dir, device)
            t_load = time.perf_counter() - started
            results, info = embed_model(spec, tokenizer, model, texts, device, int(ecfg.get("log_every", 50)))
            bad = [f for f, emb in results.items() if not np.isfinite(emb).all()]
            if bad:
                raise FloatingPointError(f"non finite values in {bad}")
        except Exception as exc:  # recorded per model so that one broken checkpoint does not stop the others
            # load and custom code errors are skipped, other errors are failures
            stage = "load" if model is None else "embed"
            custom_code = bool(spec["trust_remote_code"]) and isinstance(exc, CUSTOM_CODE_ERRORS)
            status = "skipped" if stage == "load" or custom_code else "failed"
            error_meta = {**base, "status": status, "stage": stage, "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc().splitlines()[-25:],
                          "seconds_total": round(time.perf_counter() - started, 2)}
        # freed here, once the exception and the frames it holds are released
        del model, tokenizer
        free_memory(device)
        if error_meta is not None:
            write_json_atomic(out_dir / "meta.json", error_meta)
            print(f"[embed] {spec['name']} {error_meta['status']} at {error_meta['stage']}: {error_meta['error']}",
                  flush=True)
            if error_meta["status"] == "failed":
                failed.append(spec["name"])
            continue
        for fname, emb in results.items():
            save_npy_atomic(out_dir / fname, emb)
        for stale in out_dir.glob("emb*.npy"):
            if stale.name not in results:
                stale.unlink()
        write_json_atomic(out_dir / "index.json", indices)
        info["seconds"]["load"] = round(t_load, 2)
        meta = {**base, "status": "ok", **info, "seconds_total": round(time.perf_counter() - started, 2),
                "dim": int(results["emb.npy"].shape[1])}
        write_json_atomic(out_dir / "meta.json", meta)
        print(f"[embed] {spec['name']} done dim={meta['dim']} truncated={info['n_truncated']} "
              f"chunked={info['n_chunked']} p95_tokens={info['code_token_length']['p95']:.0f} "
              f"oom_splits={info['n_oom_splits']} peak_gpu_mib={info['peak_gpu_memory_mib']} "
              f"{meta['seconds_total']:.0f}s -> {out_dir}", flush=True)
        free_memory(device)
    if failed:
        raise SystemExit(f"failed models: {failed}")


if __name__ == "__main__":
    main()
