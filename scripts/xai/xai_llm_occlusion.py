"""Line occlusion of the zero-shot LLM smell judge on MLCQ, snippet by snippet.

Runs the Qwen2.5-Coder 7B judge with transformers, then scores line occlusion and ERASER
metrics per (snippet, smell). Outputs go to artifacts/xai/llm_occlusion/.

Usage:
    python scripts/xai/xai_llm_occlusion.py select                     # write selection.json
    python scripts/xai/xai_llm_occlusion.py check --n 20               # compare with stored vLLM scores
    python scripts/xai/xai_llm_occlusion.py run --shard-index 0 --shard-count 4
    python scripts/xai/xai_llm_occlusion.py merge                      # merge the shard files
    python scripts/xai/xai_llm_occlusion.py report                     # write REPORT.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.xai import common  # noqa: E402

MODEL = "qwen25_coder_7b_instruct"
METHOD = "occlusion"
OUT = PROJECT_ROOT / "artifacts" / "xai" / "llm_occlusion"
ROWS_FILE = "qwen7b_line_attributions.jsonl"
CONFIG = PROJECT_ROOT / "config" / "experiments" / "llm_judge.yml"
SUBSET = PROJECT_ROOT / "artifacts" / "context_study" / "scores" / "qwen36_27b_fp8" / "C4" / "subset.json"
STORED = PROJECT_ROOT / "artifacts" / "llm_judge_cv" / MODEL / "predictions.jsonl"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "Qwen__Qwen2.5-Coder-7B-Instruct"
LABELS = common.LABELS
SMELLS_OF = {"method": ["is_feature_envy", "is_long_method"], "class": ["is_blob", "is_data_class"]}
FORCED = [289, 20, 1766, 334, 3047, 4030, 1049, 2298, 50]
N_PER_CLASS = 40
SEED = 0
MAX_UNITS = 60
TOP_FRAC = 0.20
N_RANDOM = 3
CLIP = 1e-6
TOP_K_LOGPROBS = 20


def logit(p: float) -> float:
    p = min(max(p, CLIP), 1 - CLIP)
    return math.log(p / (1 - p))


# ---------------------------------------------------------------- selection

def select(out: Path) -> list[dict[str, Any]]:
    strata = json.loads(SUBSET.read_text())["strata"]
    levels = common.levels()
    rng = np.random.default_rng(SEED)
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for smell in LABELS:
        st = strata[smell]
        if st["level"] != common.LEVEL_OF[smell]:
            raise SystemExit(f"{smell}: subset level {st['level']} != {common.LEVEL_OF[smell]}")
        for key, want in (("positives", 1), ("negatives", 0)):
            pool = sorted(int(i) for i in st[key])
            drawn = rng.choice(pool, size=min(N_PER_CLASS, len(pool)), replace=False)
            for i in sorted(int(i) for i in drawn):
                if common.label(i, smell) != want or levels[i] != st["level"]:
                    raise SystemExit(f"json_index {i} {smell}: label or level differs from the subset strata")
                pairs.append({"json_index": i, "smell": smell, "label": want, "forced": False})
                seen.add((i, smell))
    for i in FORCED:
        for smell in SMELLS_OF[levels[i]]:
            if (i, smell) in seen:
                next(p for p in pairs if (p["json_index"], p["smell"]) == (i, smell))["forced"] = True
                continue
            pairs.append({"json_index": i, "smell": smell, "label": common.label(i, smell), "forced": True})
            seen.add((i, smell))
    for p in pairs:
        p["n_lines"] = len(common.snippet_lines(p["json_index"]))
    out.mkdir(parents=True, exist_ok=True)
    passes = sum(min(p["n_lines"], MAX_UNITS) + 3 + 2 * N_RANDOM for p in pairs)
    meta = {"seed": SEED, "n_per_class": N_PER_CLASS, "subset": str(SUBSET.relative_to(PROJECT_ROOT)),
            "forced": FORCED, "n_pairs": len(pairs), "expected_forward_passes": passes, "pairs": pairs}
    (out / "selection.json").write_text(json.dumps(meta, indent=1))
    print(f"[select] {len(pairs)} pairs ({Counter(p['smell'] for p in pairs)}), "
          f"{sum(p['n_lines'] > MAX_UNITS for p in pairs)} chunked, about {passes} forward passes; "
          f"wrote {out / 'selection.json'}")
    return pairs


def stored_scores() -> dict[tuple[int, str], float]:
    out: dict[tuple[int, str], float] = {}
    with STORED.open() as fh:
        for line in fh:
            r = json.loads(line)
            for col, smell in enumerate(LABELS):
                out.setdefault((int(r["json_index"]), smell), float(r["prob"][col]))
    return out


# ---------------------------------------------------------------- judge

class Judge:
    """The zero-shot judge of scripts/llm/llm_judge.py with transformers (one prompt per forward pass)."""

    def __init__(self, model_path: Path, device: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from scripts.llm import llm_judge as lj

        self.lj = lj
        self.torch = torch
        cfg = yaml.safe_load(CONFIG.read_text())["llm_judge"]
        jcfg = cfg["judges"][MODEL]
        prompts = yaml.safe_load((PROJECT_ROOT / cfg["prompts"]).read_text())
        self.prompts = lj.asked_prompts(prompts, LABELS)
        self.max_code_tokens = int(jcfg["max_code_tokens"])
        self.marker = str(self.prompts["truncation"]["marker"])
        self.tok = AutoTokenizer.from_pretrained(str(model_path))
        self.template_kwargs, _ = lj.supported_template_kwargs(self.tok, dict(jcfg.get("chat_template_kwargs") or {}))
        self.yes_ids = lj.answer_token_ids(self.tok, list(cfg["answer"]["yes_variants"]))
        self.no_ids = lj.answer_token_ids(self.tok, list(cfg["answer"]["no_variants"]))
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.float16)
        self.model.to(self.device).eval()
        self.passes = 0
        self.fallbacks = 0
        self.max_prompt_tokens = 0
        print(f"[judge] yes {self.yes_ids} no {self.no_ids} template kwargs {self.template_kwargs} "
              f"dtype float16 device {self.device}", flush=True)

    def prompt_ids(self, smell: str, code: str) -> tuple[list[int], bool]:
        code_t, truncated, _ = self.lj.truncate_code(code, self.tok, self.max_code_tokens, self.marker)
        msgs = self.lj.build_messages(self.prompts, smell, code_t, truncated)
        return self.lj.render_prompt(self.tok, msgs, self.template_kwargs), truncated

    def score(self, smell: str, code: str) -> float:
        torch = self.torch
        ids, _ = self.prompt_ids(smell, code)
        self.max_prompt_tokens = max(self.max_prompt_tokens, len(ids))
        with torch.inference_mode():
            x = torch.tensor([ids], device=self.device)
            h = self.model.model(input_ids=x, use_cache=False).last_hidden_state[:, -1]
            lp = torch.log_softmax(self.model.lm_head(h).float()[0], dim=-1)
        self.passes += 1
        vals, idx = torch.topk(lp, TOP_K_LOGPROBS)
        top = [(int(i), float(v), self.tok.decode([int(i)])) for v, i in zip(vals.tolist(), idx.tolist())]
        s = self.lj.score_answer(top, self.yes_ids, self.no_ids)["score"]
        if math.isnan(s):
            self.fallbacks += 1
            p_yes = float(lp[list(self.yes_ids)].exp().sum())
            p_no = float(lp[list(self.no_ids)].exp().sum())
            s = p_yes / (p_yes + p_no) if p_yes + p_no > 0 else 0.5
        return s


# ---------------------------------------------------------------- occlusion and ERASER

def units_of(n: int) -> tuple[list[list[int]], bool]:
    if n <= MAX_UNITS:
        return [[i] for i in range(n)], False
    return [list(map(int, c)) for c in np.array_split(np.arange(n), MAX_UNITS)], True


def attribute_pair(judge: Judge, pair: dict[str, Any], stored: dict[tuple[int, str], float]) -> dict[str, Any]:
    idx, smell = pair["json_index"], pair["smell"]
    lines = common.snippet_lines(idx)
    n = len(lines)
    t0, p0, f0 = time.time(), judge.passes, judge.fallbacks

    def score_lines(keep: list[int]) -> float:
        return judge.score(smell, "\n".join(lines[i] for i in keep))

    s = score_lines(list(range(n)))
    units, chunked = units_of(n)
    attr = [0.0] * n
    attr_logit = [0.0] * n
    unit_scores = []
    for unit in units:
        drop = set(unit)
        s_wo = score_lines([i for i in range(n) if i not in drop])
        unit_scores.append(s_wo)
        for i in unit:
            attr[i] = (s - s_wo) / len(unit)
            attr_logit[i] = (logit(s) - logit(s_wo)) / len(unit)

    k = max(1, int(round(TOP_FRAC * n)))
    order = sorted(range(n), key=lambda i: (-attr[i], i))
    top = set(order[:k])
    s_without_top = score_lines([i for i in range(n) if i not in top])
    s_only_top = score_lines(sorted(top))
    rng = np.random.default_rng([SEED, idx, LABELS.index(smell)])
    r_without, r_only = [], []
    for _ in range(N_RANDOM):
        pick = set(int(i) for i in rng.choice(n, size=k, replace=False))
        r_without.append(score_lines([i for i in range(n) if i not in pick]))
        r_only.append(score_lines(sorted(pick)))
    _, truncated = judge.prompt_ids(smell, "\n".join(lines))
    row = {
        "model": MODEL, "method": METHOD, "json_index": idx, "smell": smell, "label": pair["label"],
        "score": s, "line_scores": attr, "line_scores_logit": attr_logit, "chunked": chunked,
        "units": [[u[0], u[-1]] for u in units] if chunked else None, "unit_scores_without": unit_scores,
        "base_score": s, "stored_score": stored.get((idx, smell)), "n_lines": n, "truncated": truncated,
        "top_k": k, "top_lines": sorted(top),
        "comprehensiveness": s - s_without_top, "sufficiency": s - s_only_top,
        "random_comprehensiveness": float(np.mean([s - v for v in r_without])),
        "random_sufficiency": float(np.mean([s - v for v in r_only])),
        "score_without_top": s_without_top, "score_only_top": s_only_top,
        "random_scores_without": r_without, "random_scores_only": r_only,
        "severity": common.severity(idx, smell) if pair["label"] == 1 else None,
        "forced": pair.get("forced", False),
        "forward_passes": judge.passes - p0, "fallbacks": judge.fallbacks - f0,
        "seconds": round(time.time() - t0, 2),
    }
    return row


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    common.write_rows(tmp, rows)
    tmp.replace(path)


def load_selection(out: Path) -> list[dict[str, Any]]:
    path = out / "selection.json"
    if not path.exists():
        raise SystemExit(f"{path} missing: run the select phase first")
    return json.loads(path.read_text())["pairs"]


def gpu_name() -> str | None:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


def check(out: Path, model_path: Path, device: str, n: int) -> None:
    """Base scores of n selected pairs (n / 4 per smell) against the stored vLLM scores."""
    pairs = load_selection(out)
    per = max(1, n // len(LABELS))
    chosen = [p for smell in LABELS for p in [q for q in pairs if q["smell"] == smell][:per]]
    stored = stored_scores()
    judge = Judge(model_path, device)
    rows = []
    t0 = time.time()
    for p in chosen:
        s = judge.score(p["smell"], "\n".join(common.snippet_lines(p["json_index"])))
        rows.append({"json_index": p["json_index"], "smell": p["smell"], "hf_score": s,
                     "stored_score": stored[(p["json_index"], p["smell"])]})
    diffs = np.array([abs(r["hf_score"] - r["stored_score"]) for r in rows])
    res = {"n": len(rows), "mean_abs_diff": float(diffs.mean()), "max_abs_diff": float(diffs.max()),
           "spearman": spearman([r["hf_score"] for r in rows], [r["stored_score"] for r in rows]),
           "seconds": round(time.time() - t0, 1), "gpu": gpu_name(), "host": socket.gethostname(),
           "fallbacks": judge.fallbacks, "rows": rows}
    (out / "check.json").write_text(json.dumps(res, indent=1))
    print(f"[check] n={res['n']} mean |hf - vllm| {res['mean_abs_diff']:.4f} max {res['max_abs_diff']:.4f} "
          f"spearman {res['spearman']:.3f}; wrote {out / 'check.json'}", flush=True)


def spearman(a: list[float], b: list[float]) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def run(out: Path, model_path: Path, device: str, shard_index: int, shard_count: int, limit: int | None) -> None:
    pairs = load_selection(out)
    mine = [p for k, p in enumerate(pairs) if k % shard_count == shard_index]
    if limit:
        mine = mine[:limit]
    path = out / (ROWS_FILE if shard_count == 1 else f"{Path(ROWS_FILE).stem}.shard{shard_index}of{shard_count}.jsonl")
    rows = common.read_rows(path) if path.exists() else []
    done = {(r["json_index"], r["smell"]) for r in rows}
    todo = [p for p in mine if (p["json_index"], p["smell"]) not in done]
    print(f"[run] shard {shard_index}/{shard_count}: {len(mine)} pairs, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return
    stored = stored_scores()
    t_load = time.time()
    judge = Judge(model_path, device)
    load_seconds = time.time() - t_load
    t0 = time.time()
    for k, p in enumerate(todo, 1):
        row = attribute_pair(judge, p, stored)
        rows.append(row)
        save_rows(path, rows)
        print(f"[run] {k}/{len(todo)} json_index {p['json_index']} {p['smell']} n_lines {row['n_lines']} "
              f"s {row['base_score']:.4f} (stored {row['stored_score']}) comp {row['comprehensiveness']:+.4f} "
              f"suff {row['sufficiency']:+.4f} passes {row['forward_passes']} {row['seconds']:.0f}s", flush=True)
    meta = {"host": socket.gethostname(), "gpu": gpu_name(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "shard": [shard_index, shard_count], "pairs": len(todo), "forward_passes": judge.passes,
            "fallbacks": judge.fallbacks, "max_prompt_tokens": judge.max_prompt_tokens,
            "load_seconds": round(load_seconds, 1), "seconds": round(time.time() - t0, 1),
            "transformers": __import__("transformers").__version__, "torch": judge.torch.__version__}
    mpath = out / f"run_meta.shard{shard_index}of{shard_count}.json"
    old = json.loads(mpath.read_text()) if mpath.exists() else {"runs": []}
    old["runs"].append(meta)
    mpath.write_text(json.dumps(old, indent=1))
    print(f"[run] done: {meta}", flush=True)


def merge(out: Path) -> None:
    pairs = load_selection(out)
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for path in sorted(out.glob(f"{Path(ROWS_FILE).stem}*.jsonl")):
        for r in common.read_rows(path):
            rows[(r["json_index"], r["smell"])] = r
    ordered = [rows[(p["json_index"], p["smell"])] for p in pairs if (p["json_index"], p["smell"]) in rows]
    missing = len(pairs) - len(ordered)
    save_rows(out / ROWS_FILE, ordered)
    print(f"[merge] {len(ordered)} rows, {missing} pairs missing; wrote {out / ROWS_FILE}")


# ---------------------------------------------------------------- report

KEYWORDS = {"return", "if", "else", "for", "while", "do", "try", "catch", "finally", "throw", "switch", "case",
            "default", "new", "super", "break", "continue", "synchronized", "assert", "import", "package"}
MODIFIERS = r"(?:public|private|protected|static|final|abstract|synchronized|native|default|transient|volatile)"
TYPE = r"[A-Za-z_][\w.]*(?:<[^()]*>)?(?:\[\])*"
SIG_RE = re.compile(rf"^(?:(?:{MODIFIERS}\s+)+(?:<[^>]+>\s+)?(?:{TYPE}\s+)?|(?:<[^>]+>\s+)?{TYPE}\s+)[A-Za-z_]\w*\s*\(")
CLASS_RE = re.compile(rf"^(?:{MODIFIERS}\s+)*(?:class|interface|enum|record)\s+\w+")
FIELD_RE = re.compile(rf"^(?:{MODIFIERS}\s+)+{TYPE}\s+\w+\s*(?:=|;)")
LOCAL_RE = re.compile(rf"^(?:final\s+)?{TYPE}\s+\w+\s*(?:=|;|:)")


def line_kind(line: str) -> str:
    """Coarse kind of a Java line: a structural category or the first token."""
    t = line.strip()
    if not t:
        return "<blank line>"
    if t.startswith(("//", "/*", "*")):
        return "<comment>"
    if re.fullmatch(r"[{}();,\s]+", t):
        return "<braces only>"
    if t.startswith("@"):
        return re.match(r"@\w+", t).group(0) if re.match(r"@\w+", t) else "@"
    if CLASS_RE.match(t):
        return "<class declaration>"
    first = re.match(r"[A-Za-z_]\w*", t)
    word = first.group(0) if first else None
    if t.startswith("this."):
        return "this."
    if t.startswith(('"', "'")):
        return "<string literal>"
    if t.startswith(("+", "-", "&&", "||", ".", "?", ":", "|", "&")):
        return "<expression continuation>"
    if t.startswith("}"):
        rest = t.lstrip("} ").split()
        return f"}} {rest[0]}" if rest else "<braces only>"
    if word in KEYWORDS:
        return word
    if FIELD_RE.match(t):
        return "<field declaration>"
    if SIG_RE.match(t) and not t.endswith(";") and "=" not in t.split("(")[0]:
        return "<method signature>"
    if LOCAL_RE.match(t):
        return "<local variable declaration>"
    if word and re.match(r"^(get|set|is|has|add|put|create|update)[A-Z]", word):
        return re.match(r"^(get|set|is|has|add|put|create|update)", word).group(0) + "...(" + ")"
    if word and re.match(rf"^{word}\s*\.", t):
        return "<call on a variable or class>"
    if word and re.match(rf"^{word}\s*(=|\+=|-=|\+\+|--)", t):
        return "<assignment>"
    if word and re.match(rf"^{word}\s*\(", t):
        return "<call of an own method>"
    return word or t[:12]


def signature_span(lines: list[str]) -> set[int]:
    """Lines of a method declaration, up to the first line holding "{" or ";"."""
    span: set[int] = set()
    for i, line in enumerate(lines):
        t = line.strip()
        if not span and (not t or t.startswith(("@", "//", "/*", "*"))):
            continue
        span.add(i)
        if "{" in t or t.endswith(";"):
            break
    return span


def line_kinds(json_index: int) -> list[str]:
    lines = common.snippet_lines(json_index)
    kinds = [line_kind(x) for x in lines]
    if common.levels()[json_index] == "method":
        for i in signature_span(lines):
            kinds[i] = "<method signature>"
    return kinds


def fmt(x: float) -> str:
    return f"{x:+.3f}"


def report(out: Path) -> None:
    rows = common.read_rows(out / ROWS_FILE)
    sel = json.loads((out / "selection.json").read_text())
    chk = json.loads((out / "check.json").read_text()) if (out / "check.json").exists() else None
    metas = [m for p in sorted(out.glob("run_meta.shard*.json")) for m in json.loads(p.read_text())["runs"]]
    L: list[str] = []
    w = L.append
    w("# Line occlusion of the Qwen2.5-Coder-7B-Instruct smell judge on MLCQ\n")
    w("Generated by `scripts/xai/xai_llm_occlusion.py report` from `qwen7b_line_attributions.jsonl`.\n")
    w("## Setup\n")
    w(f"The judge is the zero-shot judge of `scripts/llm/llm_judge.py` (same prompts, truncation and yes/no tokens), "
      f"run with Hugging Face transformers in float16 on one V100, one forward pass per prompt. "
      f"The score is P(yes) / (P(yes) + P(no)) on the first answer token. "
      f"The selection holds {sel['n_pairs']} snippet and smell pairs: {N_PER_CLASS} positives and {N_PER_CLASS} "
      f"negatives per smell drawn with numpy seed {SEED} from the strata of the agent subset, plus the json_index "
      f"{', '.join(map(str, FORCED))} for both smells of their level. "
      f"Line attribution is the score drop when the line is deleted (positive means the line pushes toward yes). "
      f"Snippets longer than {MAX_UNITS} lines are split into {MAX_UNITS} contiguous chunks and each line gets its "
      f"chunk attribution divided by the chunk length. "
      f"ERASER metrics use the top 20% lines by attribution (at least one line) and 3 draws of the same number of "
      f"random lines.\n")
    passes = sum(r["forward_passes"] for r in rows)
    fallbacks = sum(r["fallbacks"] for r in rows)
    w(f"Rows: {len(rows)} pairs, {sum(r['chunked'] for r in rows)} chunked, "
      f"{sum(r['truncated'] for r in rows)} truncated by the 7000 token budget. "
      f"Forward passes: {passes}. Scores that needed the full vocabulary fallback (neither yes nor no in the top "
      f"20): {fallbacks}.")
    if metas:
        gpus = sorted({str(m["gpu"]) for m in metas})
        secs = sum(m["seconds"] for m in metas)
        w(f" GPU time over all shards: {secs / 3600:.2f} hours on {', '.join(gpus)}.")
    w("\n")

    w("## Base score check\n")
    if chk:
        w(f"On {chk['n']} selected pairs ({chk['n'] // 4} per smell) the transformers float16 score differs from the "
          f"stored vLLM bfloat16 score of `artifacts/llm_judge_cv/{MODEL}/predictions.jsonl` by "
          f"{chk['mean_abs_diff']:.4f} on average (max {chk['max_abs_diff']:.4f}, Spearman {chk['spearman']:.3f}).")
    st = [(r["base_score"], r["stored_score"]) for r in rows if r.get("stored_score") is not None]
    if st:
        d = np.array([abs(a - b) for a, b in st])
        w(f" Over all {len(st)} pairs of the run: mean absolute difference {d.mean():.4f}, median "
          f"{np.median(d):.4f}, max {d.max():.4f}, Spearman {spearman([a for a, _ in st], [b for _, b in st]):.3f}, "
          f"and {int((d > 0.05).sum())} pairs differ by more than 0.05.\n")

    w("## ERASER faithfulness per smell\n")
    w("Comprehensiveness is s minus the score without the top lines (high means the top lines were needed for the "
      "yes score). Sufficiency is s minus the score with only the top lines (low means the top lines alone keep the "
      "score). Probability scale, means over pairs. The last column is the share of pairs where the top lines beat "
      "the random lines on comprehensiveness.\n")
    w("| smell | label | n | mean s | comp | random comp | suff | random suff | comp > random |")
    w("|---|---|---|---|---|---|---|---|---|")
    for smell in LABELS:
        for lab in (1, 0, None):
            sub = [r for r in rows if r["smell"] == smell and (lab is None or r["label"] == lab)]
            if not sub:
                continue
            g = lambda k: float(np.mean([r[k] for r in sub]))  # noqa: E731
            beat = np.mean([r["comprehensiveness"] > r["random_comprehensiveness"] for r in sub])
            name = {1: "positive", 0: "negative", None: "all"}[lab]
            w(f"| {common.NAMES[smell]} | {name} | {len(sub)} | {g('base_score'):.3f} | {fmt(g('comprehensiveness'))} | "
              f"{fmt(g('random_comprehensiveness'))} | {fmt(g('sufficiency'))} | {fmt(g('random_sufficiency'))} | "
              f"{beat:.0%} |")
    w("")
    w("Same table in logit space (differences of logit scores clipped to [1e-6, 1 - 1e-6]), all pairs:\n")
    w("| smell | comp | random comp | suff | random suff |")
    w("|---|---|---|---|---|")
    for smell in LABELS:
        sub = [r for r in rows if r["smell"] == smell]
        if not sub:
            continue
        c = np.mean([logit(r["base_score"]) - logit(r["score_without_top"]) for r in sub])
        rc = np.mean([np.mean([logit(r["base_score"]) - logit(v) for v in r["random_scores_without"]]) for r in sub])
        s_ = np.mean([logit(r["base_score"]) - logit(r["score_only_top"]) for r in sub])
        rs = np.mean([np.mean([logit(r["base_score"]) - logit(v) for v in r["random_scores_only"]]) for r in sub])
        w(f"| {common.NAMES[smell]} | {fmt(c)} | {fmt(rc)} | {fmt(s_)} | {fmt(rs)} |")
    w("")

    w("## Concentration of the attribution\n")
    w("Share of the absolute attribution mass carried by the 3 lines with the largest absolute attribution "
      "(probability scale). Chunked snippets spread one chunk value over several lines, so they are reported "
      "apart.\n")
    w("| smell | snippets up to 60 lines: n | median share | mean share | chunked: n | median share |")
    w("|---|---|---|---|---|---|")
    for smell in LABELS:
        shares = {False: [], True: []}
        for r in rows:
            if r["smell"] != smell:
                continue
            a = np.abs(np.array(r["line_scores"]))
            if a.sum() <= 0:
                continue
            shares[r["chunked"]].append(float(np.sort(a)[::-1][:3].sum() / a.sum()))
        nc, ch = shares[False], shares[True]
        w(f"| {common.NAMES[smell]} | {len(nc)} | {np.median(nc) if nc else float('nan'):.2f} | "
          f"{np.mean(nc) if nc else float('nan'):.2f} | {len(ch)} | {np.median(ch) if ch else float('nan'):.2f} |")
    w("")

    w("## What kind of line carries the attribution\n")
    w("For each pair, the top-1 line is the line with the largest positive attribution (the line whose deletion "
      "lowers the yes score most). Each top-1 line is mapped to a coarse kind: the first keyword or token, or a "
      "structural category in angle brackets. The tables give the 15 most frequent kinds per smell, split by label. "
      "Chunked snippets are left out because their top line is only known to the chunk. A line kind is also counted "
      "over all lines of the same snippets (base rate) to show whether the top line is over-represented.\n")
    for smell in LABELS:
        sub = [r for r in rows if r["smell"] == smell and not r["chunked"]]
        if not sub:
            continue
        w(f"### {common.NAMES[smell]}\n")
        w("| rank | kind (positives) | count | kind (negatives) | count | kind (all lines, base rate) | share |")
        w("|---|---|---|---|---|---|---|")
        cols = []
        for lab in (1, 0):
            c = Counter()
            for r in sub:
                if r["label"] != lab:
                    continue
                a = r["line_scores"]
                i = int(np.argmax(a))
                c[line_kinds(r["json_index"])[i]] += 1
            cols.append(c.most_common(15))
        base = Counter(k for r in sub for k in line_kinds(r["json_index"]))
        tot = sum(base.values())
        cols.append([(k, v / tot) for k, v in base.most_common(15)])
        for k in range(15):
            cells = []
            for j, col in enumerate(cols):
                if k < len(col):
                    cells += [f"`{col[k][0]}`", f"{col[k][1]:.0%}" if j == 2 else str(col[k][1])]
                else:
                    cells += ["", ""]
            w(f"| {k + 1} | " + " | ".join(cells) + " |")
        w("")

    w("## Examples\n")
    w("TP means a positive with a base score of at least 0.5 and FN a positive below 0.5 (a fixed cut for the "
      "examples, not the tuned threshold of the cross-validation). Among snippets of at most 40 lines, the TP with "
      "the highest score and the FN with the lowest score are shown. The number at the left of each line is its "
      "attribution in probability points (s minus the score without the line).\n")
    for smell in ("is_feature_envy", "is_long_method"):
        pos = [r for r in rows if r["smell"] == smell and r["label"] == 1 and r["n_lines"] <= 40]
        tp = sorted([r for r in pos if r["base_score"] >= 0.5], key=lambda r: -r["base_score"])
        fn = sorted([r for r in pos if r["base_score"] < 0.5], key=lambda r: r["base_score"])
        for kind, cand in (("TP", tp), ("FN", fn)):
            if not cand:
                w(f"### {common.NAMES[smell]}, {kind}\n\nNo {kind} among the positives of at most 40 lines.\n")
                continue
            r = cand[0]
            w(f"### {common.NAMES[smell]}, {kind}: json_index {r['json_index']}\n")
            w(f"Base score {r['base_score']:.3f}, severity {r['severity']}, {r['n_lines']} lines, comprehensiveness "
              f"{r['comprehensiveness']:+.3f}, sufficiency {r['sufficiency']:+.3f}.\n")
            w("```text")
            for a, line in zip(r["line_scores"], common.snippet_lines(r["json_index"])):
                w(f"{a:+.3f} | {line}")
            w("```\n")
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"[report] wrote {out / 'REPORT.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["select", "check", "run", "merge", "report"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=20, help="pairs of the check phase")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="at most this many pairs of the shard (smoke test)")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    if args.phase == "select":
        select(out)
    elif args.phase == "check":
        check(out, args.model_path, args.device, args.n)
    elif args.phase == "run":
        run(out, args.model_path, args.device, args.shard_index, args.shard_count, args.limit)
    elif args.phase == "merge":
        merge(out)
    else:
        report(out)


if __name__ == "__main__":
    main()
