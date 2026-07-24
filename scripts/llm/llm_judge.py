"""Zero-shot LLM smell judge with vLLM (offline inference).

Asks one yes/no question per sample and applicable smell. The score is
P(yes) / (P(yes) + P(no)) from the first-token logprobs, written to
<out_root>/<judge><tag>/scores.jsonl. Scored pairs are skipped on restart.

Usage:
    python scripts/llm/llm_judge.py --config config/experiments/llm_judge.yml --judge qwen25_coder_7b_instruct
    python scripts/llm/llm_judge.py --config ... --judge qwen36_27b_fp8 --shard-index 0 --shard-count 2
    python scripts/llm/llm_judge.py --config config/experiments/dacos_llm_judge.yml --judge qwen25_coder_7b_instruct
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.evaluation.cv import folds_label_order, load_folds, shard  # noqa: E402

LEVELS = ("method", "class")
LEVEL_MODES = ("raw_reviews", "field")


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


# ---------------------------------------------------------------- data


def sample_key(row: dict[str, Any]) -> tuple:
    return (row["repo_url"], row["commit_hash"], row["file_path"], int(row["start_line"]), int(row["end_line"]))


def sample_levels(
    entries: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    smell_levels: dict[str, str],
    raw_smell_names: dict[str, str],
) -> tuple[dict[int, str], dict[str, int]]:
    """Level of every normalized entry from the smells its raw review rows cover.

    Returns {json_index: level} and consistency counters."""
    by_name = {raw_smell_names[s]: s for s in smell_levels}
    reviewed: dict[tuple, set[str]] = {}
    for row in raw_rows:
        reviewed.setdefault(sample_key(row), set()).add(by_name[row["smell"]])
    levels: dict[int, str] = {}
    stats = Counter()
    for i, entry in enumerate(entries):
        smells = reviewed.get(sample_key(entry))
        if not smells:
            raise ValueError(f"entry {i} has no raw review rows")
        found = {smell_levels[s] for s in smells}
        if len(found) != 1:
            raise ValueError(f"entry {i} reviewed for smells of both levels: {sorted(smells)}")
        level = found.pop()
        levels[i] = level
        if len(smells) < sum(1 for lv in smell_levels.values() if lv == level):
            stats["single_smell_reviewed"] += 1
        votes = {s for s in smell_levels if entry.get("labels", {}).get(raw_smell_names[s], {}).get("vote_count", 0) > 0}
        if votes and votes != smells:
            stats["vote_count_mismatch"] += 1
    return levels, dict(stats)


def field_levels(entries: list[dict[str, Any]], field: str) -> tuple[dict[int, str], dict[str, int]]:
    """Level of every normalized entry read from entry[field], and the count per level."""
    levels: dict[int, str] = {}
    for i, entry in enumerate(entries):
        level = entry.get(field)
        if level not in LEVELS:
            raise ValueError(f"entry {i}: {field} is {level!r}, not one of {LEVELS}")
        levels[i] = str(level)
    return levels, dict(Counter(levels.values()))


def resolve_levels(
    jcfg: dict[str, Any], prompts: dict[str, Any], label_order: list[str], entries: list[dict[str, Any]],
) -> tuple[str, dict[str, str], dict[int, str], dict[str, int]]:
    """Level mode, {smell: level}, {json_index: level} and level stats from the config."""
    lcfg = dict(jcfg.get("levels") or {})
    mode = str(lcfg.get("mode", "raw_reviews"))
    if mode not in LEVEL_MODES:
        raise SystemExit(f"levels.mode must be one of {LEVEL_MODES}, not {mode!r}")
    missing = [s for s in label_order if s not in prompts["smells"]]
    if missing:
        raise SystemExit(f"no question for {missing} in {jcfg['prompts']}")
    if mode == "raw_reviews":
        smell_levels = {s: str(prompts["smells"][s]["level"]) for s in label_order}
        raw_rows = json.loads(resolve(jcfg["raw_json"]).read_text())
        levels, stats = sample_levels(entries, raw_rows, smell_levels, dict(jcfg["raw_smell_names"]))
        return mode, smell_levels, levels, stats
    label_levels = {str(k): str(v) for k, v in (lcfg.get("label_levels") or {}).items()}
    if sorted(label_levels) != sorted(label_order):
        raise SystemExit(f"levels.label_levels must give the level of every label of {label_order}: {label_levels}")
    for s, level in label_levels.items():
        if level not in LEVELS:
            raise SystemExit(f"levels.label_levels.{s} is {level!r}, not one of {LEVELS}")
        asked = prompts["smells"][s].get("level")
        if asked is not None and str(asked) != level:
            raise SystemExit(f"{s}: levels.label_levels says {level}, the prompts file says {asked}")
    smell_levels = {s: label_levels[s] for s in label_order}
    levels, stats = field_levels(entries, str(lcfg.get("field", "level")))
    return mode, smell_levels, levels, stats


def label_vector(entry: dict[str, Any]) -> list[int]:
    y = entry["y"]
    if isinstance(y, str):
        y = json.loads(y.lower())
    return [int(bool(v)) for v in y]


def select_subset(
    indices: list[int],
    levels: dict[int, str],
    labels: dict[int, list[int]],
    label_order: list[str],
    smell_levels: dict[str, str],
    n_per_level: int,
    min_positives: int,
    seed: int,
) -> list[int]:
    """Small stratified subset for smoke tests: n_per_level samples per level,
    with at least min_positives positives of each smell of that level."""
    rng = np.random.RandomState(seed)
    chosen: list[int] = []
    for level in LEVELS:
        pool = [i for i in indices if levels[i] == level]
        picked: list[int] = []
        for col, smell in enumerate(label_order):
            if smell_levels[smell] != level:
                continue
            taken = set(picked)
            pos = [i for i in pool if labels[i][col] and i not in taken]
            take = min(min_positives, len(pos))
            picked.extend(int(i) for i in rng.choice(pos, size=take, replace=False))
        taken = set(picked)
        rest = [i for i in pool if i not in taken]
        fill = max(0, n_per_level - len(picked))
        picked.extend(int(i) for i in rng.choice(rest, size=min(fill, len(rest)), replace=False))
        chosen.extend(picked)
    return sorted(chosen)


# ---------------------------------------------------------------- prompts


def asked_prompts(prompts: dict[str, Any], label_order: list[str]) -> dict[str, Any]:
    """The prompts file restricted to the questions of the labels asked."""
    return {**prompts, "smells": {s: prompts["smells"][s] for s in label_order}}


def prompt_fingerprint(
    prompts: dict[str, Any],
    judge_cfg: dict[str, Any],
    answer_cfg: dict[str, Any],
    sampling_cfg: dict[str, Any],
) -> str:
    """Hash of everything that changes a score (prompts, budget, sampling, model)."""
    payload = {
        "prompts": prompts,
        "max_code_tokens": judge_cfg.get("max_code_tokens"),
        "chat_template_kwargs": judge_cfg.get("chat_template_kwargs", {}),
        "answer": answer_cfg,
        "sampling": sampling_cfg,
        "model": judge_cfg.get("model_path") or judge_cfg.get("hf_id"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def truncate_code(code: str, tokenizer: Any, max_tokens: int, marker: str) -> tuple[str, bool, int]:
    """Keep the first max_tokens tokens of the snippet, cut back to a line end,
    and append the marker line. Returns (code, truncated, original token count)."""
    ids = tokenizer.encode(code, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return code, False, len(ids)
    head = tokenizer.decode(ids[:max_tokens])
    cut = head.rfind("\n")
    if cut > 0:
        head = head[:cut]
    head = head.rstrip()
    omitted = code.count("\n") - head.count("\n")
    return head + "\n" + marker.format(omitted_lines=omitted), True, len(ids)


def build_messages(prompts: dict[str, Any], smell: str, code: str, truncated: bool) -> list[dict[str, str]]:
    note = prompts["truncation"]["note"] if truncated else ""
    question = prompts["smells"][smell]["question"].format(code=code, truncation_note=note)
    messages = []
    if prompts.get("system"):
        messages.append({"role": "system", "content": prompts["system"]})
    messages.append({"role": "user", "content": question})
    return messages


def template_text(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None) or ""
    if isinstance(template, dict):
        return "\n".join(str(v) for v in template.values())
    return str(template)


def supported_template_kwargs(tokenizer: Any, wanted: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Keep only the kwargs the chat template refers to (e.g. enable_thinking)."""
    text = template_text(tokenizer)
    used = {k: v for k, v in (wanted or {}).items() if k in text}
    return used, sorted(set(wanted or {}) - set(used))


def render_prompt(tokenizer: Any, messages: list[dict[str, str]], template_kwargs: dict[str, Any]) -> list[int]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
    return tokenizer.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------- answer scoring


def answer_token_ids(tokenizer: Any, variants: list[str]) -> dict[int, str]:
    """Token id -> string for the variants that are a single token."""
    out: dict[int, str] = {}
    for text in variants:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            out[int(ids[0])] = text
    return out


def top_logprobs(position: Any) -> list[tuple[int, float, str | None]]:
    """(token id, logprob, decoded token) for one position of vLLM logprobs
    (a dict of token id -> Logprob), or of a plain {id: (logprob, text)} dict."""
    rows = []
    for token_id, lp in position.items():
        if hasattr(lp, "logprob"):
            rows.append((int(token_id), float(lp.logprob), getattr(lp, "decoded_token", None)))
        else:
            value, text = lp
            rows.append((int(token_id), float(value), text))
    rows.sort(key=lambda r: -r[1])
    return rows


def score_answer(
    top: list[tuple[int, float, str | None]],
    yes_ids: dict[int, str],
    no_ids: dict[int, str],
) -> dict[str, Any]:
    """P(yes) / (P(yes) + P(no)) from the top-K first-token logprobs, NaN if neither appears."""
    p = {"yes": 0.0, "no": 0.0}
    matched = []
    for token_id, logprob, text in top:
        word = None
        if token_id in yes_ids:
            word = "yes"
        elif token_id in no_ids:
            word = "no"
        elif text is not None and text.strip().lower() in p:
            word = text.strip().lower()
        if word is None:
            continue
        p[word] += math.exp(logprob)
        shown = text if text is not None else (yes_ids.get(token_id) or no_ids.get(token_id))
        matched.append([token_id, shown, round(logprob, 6), word])
    total = p["yes"] + p["no"]
    score = p["yes"] / total if total > 0 else float("nan")
    first = top[0] if top else (None, float("nan"), None)
    return {
        "score": score,
        "p_yes": p["yes"],
        "p_no": p["no"],
        "matched": matched,
        "top_token": [first[2], round(first[1], 6)],
    }


# ---------------------------------------------------------------- io


def scores_path(out_dir: Path, shard_index: int, shard_count: int) -> Path:
    if shard_count > 1:
        return out_dir / f"scores.shard{shard_index}of{shard_count}.jsonl"
    return out_dir / "scores.jsonl"


def meta_path(out_dir: Path, shard_index: int, shard_count: int) -> Path:
    if shard_count > 1:
        return out_dir / f"meta.shard{shard_index}of{shard_count}.json"
    return out_dir / "meta.json"


def read_valid_rows(path: Path) -> list[dict[str, Any]]:
    """Rows of a scores file, ignoring a partial last line left by a kill."""
    rows = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def repair_scores_file(path: Path) -> list[dict[str, Any]]:
    rows = read_valid_rows(path)
    if path.exists():
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        tmp.replace(path)
    return rows


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, allow_nan=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def scored_pairs(out_dir: Path, fingerprint: str) -> set[tuple[int, str]]:
    done: set[tuple[int, str]] = set()
    for path in sorted(out_dir.glob("scores*.jsonl")):
        for row in read_valid_rows(path):
            if row.get("prompt_hash") != fingerprint:
                raise SystemExit(f"{path} holds scores of another prompt ({row.get('prompt_hash')} != {fingerprint}); "
                                 "use another tag or remove the directory")
            done.add((int(row["json_index"]), str(row["smell"])))
    return done


def file_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    applicable = [r for r in rows if r.get("applicable")]
    matched = Counter(m[1] for r in applicable for m in r.get("matched", []))
    return {
        "rows": len(rows),
        "applicable": len(applicable),
        "nan": sum(1 for r in applicable if r["score"] is None or math.isnan(r["score"])),
        "one_sided": sum(1 for r in applicable if (r["p_yes"] > 0) != (r["p_no"] > 0)),
        "truncated_prompts": sum(1 for r in applicable if r.get("truncated")),
        "truncated_samples": len({r["json_index"] for r in applicable if r.get("truncated")}),
        "matched_tokens": dict(matched.most_common()),
        "top_tokens": dict(Counter(str(r["top_token"][0]) for r in applicable).most_common(10)),
    }


# ---------------------------------------------------------------- main


def load_tokenizer(model_ref: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_ref)


def model_reference(judge_cfg: dict[str, Any]) -> tuple[str, str]:
    if judge_cfg.get("model_path"):
        path = resolve(judge_cfg["model_path"])
        if not path.exists():
            raise SystemExit(f"model path not found: {path}")
        return str(path), "path"
    if judge_cfg.get("hf_id"):
        return str(judge_cfg["hf_id"]), "hf_id"
    raise SystemExit("judge needs model_path or hf_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--judge", type=str, default=None, help="judge name under llm_judge.judges")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="build prompts and print stats without vLLM")
    args, extra = parser.parse_known_args()
    cfg = load_config(args.config, parse_cli_overrides(extra))
    jcfg = cfg["llm_judge"]
    judge = args.judge or str(jcfg["default_judge"])
    judge_cfg = jcfg["judges"][judge]
    name = f"{judge}{jcfg.get('tag') or ''}"
    out_dir = resolve(jcfg["out_root"]) / name

    prompts_text = resolve(jcfg["prompts"]).read_text()
    folds = load_folds(resolve(jcfg["folds"]))
    label_order = folds_label_order(folds, jcfg.get("label_order"))
    entries = json.loads(resolve(jcfg["input_json"]).read_text())
    all_prompts = yaml.safe_load(prompts_text)
    level_mode, smell_levels, levels, level_stats = resolve_levels(jcfg, all_prompts, label_order, entries)
    prompts = asked_prompts(all_prompts, label_order)
    labels = {i: label_vector(e) for i, e in enumerate(entries)}

    indices = [int(i) for i in folds["meta"]["json_index"]]
    subset = jcfg.get("subset") or {}
    if subset.get("n_per_level"):
        indices = select_subset(indices, levels, labels, label_order, smell_levels,
                                int(subset["n_per_level"]), int(subset.get("min_positives_per_smell", 0)),
                                int(subset.get("seed", 0)))
    indices = shard(indices, args.shard_index, args.shard_count)
    print(f"[judge] {name}: {len(indices)} samples in shard {args.shard_index}/{args.shard_count} "
          f"(method {sum(levels[i] == 'method' for i in indices)}, class {sum(levels[i] == 'class' for i in indices)}); "
          f"level mode {level_mode}, level stats {level_stats}", flush=True)

    model_ref, model_source = model_reference(judge_cfg)
    tokenizer = load_tokenizer(model_ref)
    template_kwargs, dropped = supported_template_kwargs(tokenizer, dict(judge_cfg.get("chat_template_kwargs") or {}))
    if dropped:
        print(f"[judge] chat template does not use {dropped}: not passed", flush=True)
    answer_cfg = dict(jcfg["answer"])
    yes_ids = answer_token_ids(tokenizer, list(answer_cfg["yes_variants"]))
    no_ids = answer_token_ids(tokenizer, list(answer_cfg["no_variants"]))
    if not yes_ids or not no_ids:
        raise SystemExit(f"no single-token yes/no variant in this tokenizer: yes={yes_ids} no={no_ids}")
    print(f"[judge] yes tokens {yes_ids} no tokens {no_ids}; template kwargs {template_kwargs}", flush=True)
    sampling_cfg = dict(jcfg["sampling"])
    fingerprint = prompt_fingerprint(prompts, judge_cfg, answer_cfg, sampling_cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    own_file = scores_path(out_dir, args.shard_index, args.shard_count)
    if not args.dry_run:
        repair_scores_file(own_file)
    done = scored_pairs(out_dir, fingerprint)

    engine_cfg = dict(judge_cfg.get("engine") or {})
    max_len = int(engine_cfg.get("max_model_len", 0)) or None
    marker = str(prompts["truncation"]["marker"])
    max_code_tokens = int(judge_cfg["max_code_tokens"])

    # non-applicable smells get score 0 without a query
    static_rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    code_cache: dict[int, tuple[str, bool, int]] = {}
    for idx in indices:
        for smell in label_order:
            if (idx, smell) in done:
                continue
            if smell_levels[smell] != levels[idx]:
                static_rows.append({"json_index": idx, "smell": smell, "level": levels[idx], "applicable": False,
                                    "score": 0.0, "p_yes": 0.0, "p_no": 0.0, "matched": [], "top_token": [None, None],
                                    "truncated": False, "prompt_tokens": 0, "prompt_hash": fingerprint})
                continue
            if idx not in code_cache:
                code_cache[idx] = truncate_code(entries[idx]["code_snippet"], tokenizer, max_code_tokens, marker)
            code, truncated, code_tokens = code_cache[idx]
            ids = render_prompt(tokenizer, build_messages(prompts, smell, code, truncated), template_kwargs)
            if max_len is not None and len(ids) + int(sampling_cfg["max_tokens"]) > max_len:
                raise SystemExit(f"prompt of json_index {idx} / {smell} has {len(ids)} tokens, over max_model_len "
                                 f"{max_len}: lower max_code_tokens")
            pending.append({"json_index": idx, "smell": smell, "level": levels[idx], "ids": ids,
                            "truncated": truncated, "code_tokens": code_tokens})

    lengths = np.array([len(p["ids"]) for p in pending]) if pending else np.zeros(1)
    print(f"[judge] {len(done)} pairs already scored, {len(static_rows)} non-applicable rows, {len(pending)} prompts; "
          f"prompt tokens mean {lengths.mean():.0f} p50 {np.percentile(lengths, 50):.0f} "
          f"p95 {np.percentile(lengths, 95):.0f} max {lengths.max():.0f}; "
          f"truncated prompts {sum(p['truncated'] for p in pending)} "
          f"(samples {len({p['json_index'] for p in pending if p['truncated']})})", flush=True)

    if args.dry_run:
        preview = out_dir / "prompt_preview.txt"
        shown = {}
        for p in pending:
            shown.setdefault(p["smell"], p)
        preview.write_text("\n\n".join(f"===== json_index {p['json_index']} {p['smell']} truncated={p['truncated']}\n"
                                       + tokenizer.decode(p["ids"]) for p in shown.values()))
        print(f"[judge] dry run: wrote {preview}", flush=True)
        return
    if not pending:
        if static_rows:
            append_rows(own_file, static_rows)
        print("[judge] nothing left to score", flush=True)
        return

    started = time.time()
    from vllm import LLM, SamplingParams
    import vllm

    llm = LLM(model=model_ref, **engine_cfg)
    load_seconds = time.time() - started
    params = SamplingParams(max_tokens=int(sampling_cfg["max_tokens"]), temperature=float(sampling_cfg["temperature"]),
                            logprobs=int(sampling_cfg["logprobs"]))
    gpu = None
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass

    if static_rows:
        append_rows(own_file, static_rows)
    chunk = int(jcfg["chunk_size"])
    gen_seconds, prompt_tokens, nan_run = 0.0, 0, 0
    for start in range(0, len(pending), chunk):
        batch = pending[start:start + chunk]
        t0 = time.time()
        outputs = llm.generate([{"prompt_token_ids": p["ids"]} for p in batch], params, use_tqdm=False)
        gen_seconds += time.time() - t0
        rows = []
        for p, out in zip(batch, outputs):
            position = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
            result = score_answer(top_logprobs(position), yes_ids, no_ids)
            nan_run += int(math.isnan(result["score"]))
            prompt_tokens += len(p["ids"])
            rows.append({"json_index": p["json_index"], "smell": p["smell"], "level": p["level"], "applicable": True,
                         **result, "truncated": p["truncated"], "code_tokens": p["code_tokens"],
                         "prompt_tokens": len(p["ids"]), "prompt_hash": fingerprint})
        append_rows(own_file, rows)
        print(f"[judge] {min(start + chunk, len(pending))}/{len(pending)} prompts, "
              f"{prompt_tokens / max(gen_seconds, 1e-9):.0f} prompt tokens/s, NaN so far {nan_run}", flush=True)

    mpath = meta_path(out_dir, args.shard_index, args.shard_count)
    meta = json.loads(mpath.read_text()) if mpath.exists() else {"runs": []}
    meta.update({
        "judge": judge, "name": name, "model": model_ref, "model_source": model_source,
        "prompts_file": str(jcfg["prompts"]), "prompt_hash": fingerprint,
        "chat_template_kwargs": template_kwargs, "chat_template_kwargs_dropped": dropped,
        "answer_tokens": {"yes": yes_ids, "no": no_ids}, "engine": engine_cfg, "sampling": sampling_cfg,
        "max_code_tokens": max_code_tokens, "vllm_version": getattr(vllm, "__version__", None),
        "label_order": label_order, "level_mode": level_mode, "smell_levels": smell_levels,
        "level_stats": level_stats, "subset": subset, "shard": [args.shard_index, args.shard_count],
    })
    meta["runs"].append({
        "host": socket.gethostname(), "gpu": gpu, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "load_seconds": round(load_seconds, 1), "generate_seconds": round(gen_seconds, 1),
        "seconds": round(time.time() - started, 1), "prompts": len(pending), "prompt_tokens": prompt_tokens,
        "prompts_per_second": round(len(pending) / max(gen_seconds, 1e-9), 2),
        "tokens_per_second": round(prompt_tokens / max(gen_seconds, 1e-9), 1), "nan": nan_run,
        "truncated_prompts": sum(p["truncated"] for p in pending),
    })
    meta["totals"] = file_totals(read_valid_rows(own_file))
    write_json_atomic(mpath, meta)
    print(f"[judge] done in {time.time() - started:.0f}s (load {load_seconds:.0f}s); totals {meta['totals']}", flush=True)
    print(f"[judge] wrote {own_file} and {mpath}", flush=True)


if __name__ == "__main__":
    main()
