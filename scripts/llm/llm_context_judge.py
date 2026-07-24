"""LLM smell judge with repository context (vLLM offline inference).

Asks the yes/no question of llm_judge.py under one context condition of
config/experiments/context_study.yml: C0 to C3, SHUF, FACTS_ONLY, PAD, C4
(agent with repository tools), REPLAY and REASON. Writes scores.jsonl and
meta.json (plus transcripts.jsonl for C4) under <out_root>/<judge><tag>/<condition>/.

Usage:
    python scripts/llm/llm_context_judge.py --config config/experiments/context_study.yml --condition C1
    python scripts/llm/llm_context_judge.py --config ... --condition C4 --shard-index 0 --shard-count 2
    python scripts/llm/llm_context_judge.py --config ... --condition C3 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import socket
import sys
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.llm.build_context_packages import (  # noqa: E402
    IDENTIFIER_RE,
    METHOD_DECLS,
    FileContext,
    SourceCache,
    body_members,
    declared_name,
    find_type_node,
    first_line,
    last_line,
    parser as java_parser,
    load_type_index,
    proximity_order,
    read_levels,
    repo_name,
    resolve_type,
    resolved_skeleton,
    usage_lines,
)
from scripts.llm.llm_judge import (  # noqa: E402
    answer_token_ids,
    append_rows,
    file_totals,
    label_vector,
    model_reference,
    read_valid_rows,
    render_prompt,
    repair_scores_file,
    score_answer,
    scored_pairs,
    select_subset,
    supported_template_kwargs,
    top_logprobs,
    truncate_code,
    write_json_atomic,
)
from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.evaluation.cv import load_folds, shard  # noqa: E402

CONDITIONS = ("C0", "C1", "C2", "C3", "SHUF", "FACTS_ONLY", "PAD", "C4", "REPLAY", "REASON")
# conditions run as multi-turn conversations (agent loop), with or without tools
MULTI_TURN = ("C4", "REASON")
SINGLE_TURN = ("C0", "C1", "C2", "C3", "SHUF", "FACTS_ONLY", "PAD", "REPLAY")
# conditions scored on every json_index of the folds meta (the others on the agent subset)
FULL_SCOPE = ("C0", "C1", "C2", "C3", "SHUF", "FACTS_ONLY", "PAD")
SIZE_SKIP = {"sample_idx", "has_parse_error"}
QUALIFIED_RE = re.compile(r"^[A-Za-z_$][\w$]*(\.[A-Za-z_$][\w$]*)*$")
# statuses of a call that ran a tool and returned repository content
TOOL_OK = ("ok",)
# statuses of a call whose symbol could not be found in the snapshot
TOOL_UNRESOLVED = ("unresolved", "external", "implicit", "ambiguous", "method_not_found")


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


# ---------------------------------------------------------------- data


def load_packages(path: Path) -> dict[int, dict[str, Any]]:
    with path.open() as fh:
        return {int(p["json_index"]): p for p in map(json.loads, fh)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def read_size_features(path: Path) -> dict[int, dict[str, int]]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    return {int(float(r["sample_idx"])): {k: int(float(v)) for k, v in r.items() if k not in SIZE_SKIP}
            for r in rows}


def agent_subset(
    indices: list[int],
    levels: dict[int, str],
    labels: dict[int, list[int]],
    label_order: list[str],
    smell_levels: dict[str, str],
    negatives_per_positive: int,
    seed: int,
    positives_per_smell: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-smell subset: positives of its level (optionally capped) plus
    negatives_per_positive times as many negatives from the same level."""
    rng = np.random.RandomState(seed)
    out: dict[str, dict[str, Any]] = {}
    for col, smell in enumerate(label_order):
        level = smell_levels[smell]
        pool = sorted(i for i in indices if levels[i] == level)
        pos = [i for i in pool if labels[i][col]]
        neg_pool = [i for i in pool if not labels[i][col]]
        if positives_per_smell is not None and len(pos) > positives_per_smell:
            pos = sorted(int(i) for i in rng.choice(pos, size=positives_per_smell, replace=False))
        take = min(len(neg_pool), negatives_per_positive * len(pos))
        neg = sorted(int(i) for i in rng.choice(neg_pool, size=take, replace=False)) if take else []
        out[smell] = {"level": level, "positives": pos, "negatives": neg}
    return out


def subset_pairs(subset: dict[str, dict[str, Any]], label_order: list[str]) -> list[tuple[int, str]]:
    pairs = {(i, s) for s in label_order for part in ("positives", "negatives") for i in subset[s][part]}
    return sorted(pairs, key=lambda p: (p[0], label_order.index(p[1])))


def config_subset(jcfg: dict[str, Any], indices: list[int], levels: dict[int, str], labels: dict[int, list[int]],
                  label_order: list[str], smell_levels: dict[str, str]) -> dict[str, dict[str, Any]]:
    scfg = dict(jcfg["agent_subset"])
    cap = scfg.get("positives_per_smell")
    return agent_subset(indices, levels, labels, label_order, smell_levels, int(scfg["negatives_per_positive"]),
                        int(scfg["seed"]), None if cap is None else int(cap))


def shuffle_donors(
    indices: list[int],
    levels: dict[int, str],
    packages: dict[int, dict[str, Any]],
    repos: dict[int, str],
    seed: int,
) -> dict[int, int | None]:
    """Donor sample for SHUF: same level and skeleton kind, another repository,
    drawn with a seed per json_index."""
    pools: dict[tuple[str, str], list[int]] = {}
    for i in sorted(indices):
        pkg = packages.get(i)
        if pkg and pkg.get("skeleton"):
            pools.setdefault((levels[i], pkg["skeleton_kind"]), []).append(i)
    donors: dict[int, int | None] = {}
    for i in indices:
        pkg = packages.get(i)
        if not pkg or not pkg.get("skeleton"):
            donors[i] = None
            continue
        pool = [j for j in pools[(levels[i], pkg["skeleton_kind"])] if repos[j] != repos[i]]
        if not pool:
            donors[i] = None
            continue
        rng = np.random.default_rng([int(seed), int(i)])
        donors[i] = int(pool[int(rng.integers(len(pool)))])
    return donors


def load_replays(c4_dir: Path, c4_hash: str,
                 scored: set[tuple[int, str]] | None = None) -> dict[tuple[int, str], list[dict[str, Any]]]:
    """Tool outputs of the C4 transcripts with the given prompt hash, keyed by
    (json_index, smell). With scored, only pairs that have a C4 score are kept."""
    out: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for path in sorted(c4_dir.glob("transcripts*.jsonl")):
        for row in read_valid_rows(path):
            if row.get("prompt_hash") != c4_hash:
                raise SystemExit(f"{path} holds C4 transcripts of another prompt ({row.get('prompt_hash')} != "
                                 f"{c4_hash}): REPLAY must use the C4 run of the same configuration")
            key = (int(row["json_index"]), str(row["smell"]))
            if scored is None or key in scored:
                out[key] = list(row["tool_outputs"])
    return out


# ---------------------------------------------------------------- prompts


def fit(text: str, tokenizer: Any, budget: int, marker: str) -> tuple[str, bool, int]:
    """Text cut to budget tokens at a line end, with a marker line."""
    return truncate_code(text, tokenizer, max(int(budget), 0), marker)


class PromptBuilder:
    """User messages of the single-turn conditions and of the agent."""

    def __init__(self, prompts: dict[str, Any], tokenizer: Any, budgets: dict[str, Any]) -> None:
        self.prompts = prompts
        self.tok = tokenizer
        self.budgets = budgets
        self.code_cache: dict[int, tuple[str, bool, int]] = {}
        self.pad_cache: dict[int, str] = {}
        self.marker = str(prompts["truncation"]["context_marker"])

    def ntok(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def code(self, idx: int, snippet: str) -> tuple[str, bool, int]:
        """The snippet cut to max_code_tokens, the same for every condition."""
        if idx not in self.code_cache:
            self.code_cache[idx] = truncate_code(snippet, self.tok, int(self.budgets["max_code_tokens"]),
                                                 str(self.prompts["truncation"]["marker"]))
        return self.code_cache[idx]

    def section(self, name: str, kind: str, text: str | None, budget: int,
                built: bool) -> tuple[str, dict[str, Any]]:
        """One rendered context section and its token stats."""
        spec = self.prompts["context"]["sections"][name][kind]
        title = spec["title"]
        if not text:
            if built and spec.get("empty"):
                return f"{title}: {spec['empty']}.\n\n", {"available": True, "empty": True, "tokens": 0,
                                                          "truncated": False}
            unavailable = str(self.prompts["context"]["unavailable"])
            return f"{title}: {unavailable}.\n\n", {"available": False, "empty": True, "tokens": 0,
                                                     "truncated": False}
        text, truncated, tokens = fit(text, self.tok, budget, self.marker)
        body = f"```java\n{text}\n```" if spec.get("code") else text
        return f"{title}:\n{body}\n\n", {"available": True, "empty": False, "tokens": tokens, "truncated": truncated}

    def referenced_text(self, pkg: dict[str, Any], scale: float) -> str | None:
        parts = []
        each = int(int(self.budgets["referenced_each"]) * scale)
        for item in pkg.get("referenced_skeletons") or []:
            text, _, _ = fit(item["skeleton"], self.tok, each, self.marker)
            parts.append(text)
        return "\n\n".join(parts) if parts else None

    def usages_text(self, pkg: dict[str, Any]) -> str | None:
        usages = pkg.get("usages") or {}
        lines = usages.get("lines") or []
        if not lines:
            return None
        head = self.prompts["context"]["sections"]["usages"]["class"]["head"].format(
            n_lines=usages.get("n_lines", 0), n_files=usages.get("n_files", 0), shown=len(lines))
        return head + "\n" + "\n".join(lines)

    def replay_text(self, outputs: list[dict[str, Any]]) -> str | None:
        """Tool outputs of a C4 conversation in call order, each under its call."""
        item = str(self.prompts["context"]["sections"]["replay"]["any"]["item"])
        parts = []
        for k, out in enumerate(outputs, start=1):
            args = ", ".join(f"{a}={v}" for a, v in (out.get("arguments") or {}).items())
            parts.append(item.format(k=k, name=out["name"], arguments=args) + "\n" + str(out["text"]))
        return "\n\n".join(parts) if parts else None

    def context(self, condition: str, pkg: dict[str, Any] | None, donor: dict[str, Any] | None,
                level: str, scale: float = 1.0,
                replay: list[dict[str, Any]] | None = None) -> tuple[str, dict[str, Any]]:
        """Rendered context sections of a condition and their token stats."""
        names = list(self.prompts["conditions"][condition])
        if not names:
            return "", {}
        pkg = pkg or {}
        b = {k: (int(int(v) * scale) if k != "fit_scales" else v) for k, v in self.budgets.items()}
        out, info = [], {}
        built = pkg.get("target") is not None
        skeleton_kind = pkg.get("skeleton_kind") or ("enclosing_type" if level == "method" else "file_context")
        for name in names:
            if name == "skeleton":
                text, meta = self.section("skeleton", skeleton_kind, pkg.get("skeleton"), b["skeleton"], False)
            elif name == "skeleton_shuffled":
                source = donor or {}
                text, meta = self.section("skeleton", skeleton_kind, source.get("skeleton"), b["skeleton"], False)
            elif name == "facts":
                text, meta = self.section("facts", level, pkg.get("facts_text"), b["facts"], False)
            elif name == "extra" and level == "method":
                text, meta = self.section("referenced", "method", self.referenced_text(pkg, scale),
                                          b["referenced_total"], built)
                meta["types"] = len(pkg.get("referenced_skeletons") or [])
            elif name == "extra":
                text, meta = self.section("usages", "class", self.usages_text(pkg), b["usages"],
                                          built and pkg.get("usages") is not None)
            elif name == "replay":
                text, meta = self.section("replay", "any", self.replay_text(replay or []), b["replay_total"],
                                          replay is not None)
                meta["outputs"] = len(replay or [])
            else:
                raise SystemExit(f"unknown context section {name!r} of condition {condition}")
            out.append(text)
            info[name] = meta
        return str(self.prompts["context"]["intro"]) + "".join(out), info

    def pad_block(self, target: int) -> str:
        """Filler block of about target tokens, laid out like a context section."""
        if target in self.pad_cache:
            return self.pad_cache[target]
        pcfg = self.prompts["pad"]
        head = f"{pcfg['title']}:\n"
        line = f"{pcfg['line']}\n"
        render = lambda k, tail="": f"{head}{line * k}{tail}\n"  # noqa: E731
        k = max(0, (target - self.ntok(render(0))) // max(self.ntok(line), 1))
        while k > 0 and self.ntok(render(k)) > target:
            k -= 1
        while self.ntok(render(k + 1)) <= target:
            k += 1
        tail = ""
        for word in str(pcfg["line"]).split():
            longer = f"{tail} {word}".strip()
            if self.ntok(render(k, longer + "\n")) > target:
                break
            tail = longer
        text = render(k, tail + "\n" if tail else "")
        self.pad_cache[target] = text
        return text

    def user_message(self, template: str, smell: str, code: str, truncated: bool, **fmt: Any) -> str:
        s = self.prompts["smells"][smell]
        note = self.prompts["truncation"]["note"] if truncated else ""
        return template.format(smell_name=s["name"], definition=s["definition"], unit=s["unit"],
                               unit_lower=s["unit"].lower(), question=s["question"], code=code,
                               truncation_note=note, **fmt)


def fingerprint(condition: str, prompts: dict[str, Any], jcfg: dict[str, Any], judge_cfg: dict[str, Any],
                packages_sha: str | None, backend: str, c4_hash: str | None = None,
                tool_inputs: dict[str, Any] | None = None) -> str:
    """Hash of everything that changes a score of this condition (tool inputs
    included for C4)."""
    payload: dict[str, Any] = {
        "condition": condition,
        "smells": prompts["smells"],
        "truncation": prompts["truncation"],
        "budgets": jcfg["budgets"],
        "chat_template_kwargs": judge_cfg.get("chat_template_kwargs", {}),
        "answer": jcfg["answer"],
        "sampling": jcfg["sampling"],
        "model": judge_cfg.get("model_path") or judge_cfg.get("hf_id"),
        "max_model_len": (judge_cfg.get("engine") or {}).get("max_model_len"),
        "backend": backend,
    }
    if condition in SINGLE_TURN:
        payload.update({"system": prompts["system"], "user_template": prompts["user_template"],
                        "context": prompts["context"], "sections": prompts["conditions"][condition]})
    if condition != "C0":
        payload["packages"] = packages_sha
    if condition == "SHUF":
        payload["shuffle_seed"] = jcfg["shuffle_seed"]
    if condition == "FACTS_ONLY":
        payload["facts_only_template"] = prompts["facts_only_template"]
    if condition == "PAD":
        payload.update({"pad": prompts["pad"], "c3_sections": prompts["conditions"]["C3"]})
    if condition in ("C4", "REPLAY", "REASON"):
        payload["agent_subset"] = jcfg["agent_subset"]
    if condition == "REASON":
        payload.update({"reason_prompts": prompts["reason"], "final_question": prompts["agent"]["final_question"],
                        "agent": jcfg["agent"], "stop_tokens": judge_cfg.get("stop_tokens")})
    if condition == "C4":
        if tool_inputs is None:
            raise ValueError("the C4 fingerprint needs tool_inputs")
        payload.update({"agent_prompts": prompts["agent"], "agent": jcfg["agent"],
                        "stop_tokens": judge_cfg.get("stop_tokens"), "tool_inputs": tool_inputs})
    if condition == "REPLAY":
        payload.update({"c4_prompt_hash": c4_hash, "replay_statuses": jcfg["replay"]["statuses"]})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def render_chat(tokenizer: Any, messages: list[dict[str, Any]], template_kwargs: dict[str, Any],
                tools: list[dict[str, Any]] | None) -> list[int]:
    text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True,
                                         **template_kwargs)
    return tokenizer.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------- tool calls


def _strip_value(value: str) -> str:
    """XML parameter value without the newlines added by the template."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def parse_call_body(body: str) -> dict[str, Any]:
    """{name, arguments, error} of the text between <tool_call> and </tool_call>."""
    text = body.strip()
    if text.startswith("{"):
        try:
            obj, _ = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError as exc:
            return {"name": None, "arguments": {}, "error": f"invalid JSON: {exc.msg}"}
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            return {"name": None, "arguments": {}, "error": "JSON without a name"}
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                return {"name": obj["name"], "arguments": {}, "error": "arguments are not a JSON object"}
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return {"name": obj["name"], "arguments": {}, "error": "arguments are not a JSON object"}
        return {"name": obj["name"], "arguments": args, "error": None}
    match = re.search(r"<function=([^>\n]+)>(.*?)(</function>|\Z)", text, re.S)
    if match:
        name = match.group(1).strip()
        if not match.group(3):
            return {"name": name, "arguments": {}, "error": "missing </function>"}
        args = {m.group(1).strip(): _strip_value(m.group(2))
                for m in re.finditer(r"<parameter=([^>\n]+)>(.*?)</parameter>", match.group(2), re.S)}
        return {"name": name, "arguments": args, "error": None}
    return {"name": None, "arguments": {}, "error": "expected a JSON object or a <function=...> block"}


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Every <tool_call> block of an assistant message, in order."""
    calls = []
    for part in text.split("<tool_call>")[1:]:
        body, sep, _ = part.partition("</tool_call>")
        call = parse_call_body(body)
        if not sep and call["error"] is None and not body.strip():
            call["error"] = "empty tool call"
        call["raw"] = body.strip()[:400]
        calls.append(call)
    return calls


# ---------------------------------------------------------------- tools


class SnapshotStore:
    """LRU cache of snapshot type indexes, parsed files and file texts."""

    def __init__(self, snapshots_dir: Path, index_dir: Path, capacity: int) -> None:
        self.snapshots_dir = snapshots_dir
        self.index_dir = index_dir
        self.capacity = max(1, int(capacity))
        self.items: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, snapshot: str) -> dict[str, Any] | None:
        if snapshot in self.items:
            self.items.move_to_end(snapshot)
            return self.items[snapshot]
        path = self.snapshots_dir / snapshot
        if not path.is_dir():
            return None
        item = {"dir": path, "index": load_type_index(path, self.index_dir), "cache": SourceCache(path),
                "texts": None}
        self.items[snapshot] = item
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
        return item

    @staticmethod
    def texts(item: dict[str, Any]) -> dict[str, str]:
        if item["texts"] is None:
            texts = {}
            for rel in item["index"]["files"]:
                try:
                    texts[rel] = (item["dir"] / rel).read_text(errors="replace")
                except OSError:
                    continue
            item["texts"] = texts
        return item["texts"]


class RepositoryTools:
    """The C4 repository tools for one sample."""

    def __init__(self, pkg: dict[str, Any] | None, store: SnapshotStore, sizes: dict[str, int] | None,
                 messages: dict[str, str], acfg: dict[str, Any], opts: dict[str, Any], implicit: set[str],
                 tool_names: list[str]) -> None:
        self.pkg = pkg or {}
        self.store = store
        self.sizes = sizes
        self.msg = messages
        self.acfg = acfg
        self.opts = opts
        self.implicit = implicit
        self.tool_names = tool_names
        self._ctx: FileContext | None = None
        self._item: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any] | None:
        if self._item is None and self.pkg.get("found_snapshot"):
            self._item = self.store.get(self.pkg["snapshot"])
        return self._item

    def file_context(self) -> FileContext | None:
        if self._ctx is None:
            item = self.snapshot()
            if item is None or not self.pkg.get("found_file"):
                return None
            hit = item["cache"].get(self.pkg["file"])
            if hit is None:
                return None
            self._ctx = FileContext(self.pkg["file"], hit[1], hit[0])
        return self._ctx

    def call(self, name: str | None, arguments: dict[str, Any]) -> tuple[str, str]:
        """Run one tool call and return (status, text)."""
        if name not in self.tool_names:
            return "unknown_tool", self.msg["unknown_tool"].format(name=name, tools=", ".join(self.tool_names))
        required = {"open_type": ["name"], "read_method": ["type_name", "method_name"], "find_usages": ["name"]}
        for arg in required.get(name, []):
            value = arguments.get(arg)
            if not isinstance(value, str) or not value.strip():
                return "missing_argument", self.msg["missing_argument"].format(argument=arg, name=name)
        try:
            return getattr(self, name)(**{k: str(arguments[k]).strip() for k in required.get(name, [])})
        except Exception as exc:  # a tool failure is reported to the model, never fatal
            return "exception", f"Error: {type(exc).__name__}: {exc}"

    def get_enclosing_class(self) -> tuple[str, str]:
        text = self.pkg.get("skeleton")
        return ("ok", text) if text else ("unavailable", self.msg["unavailable"])

    def get_metrics(self) -> tuple[str, str]:
        if not self.sizes:
            return "unavailable", self.msg["unavailable"]
        return "ok", "\n".join(f"{k}: {v}" for k, v in self.sizes.items())

    def _resolve(self, name: str) -> tuple[dict[str, Any] | None, str, str | None]:
        """(resolution, status, error message) of a type name."""
        name = name.strip().removesuffix(".java")
        if not QUALIFIED_RE.match(name):
            return None, "invalid_name", self.msg["type_unresolved"].format(name=name)
        ctx, item = self.file_context(), self.snapshot()
        if ctx is None or item is None:
            return None, "unavailable", self.msg["unavailable"]
        res = resolve_type(name, ctx, item["index"]["types"], self.implicit)
        status = res["status"]
        if status == "resolved":
            return res, "ok", None
        if status == "external":
            return None, status, self.msg["type_external"].format(name=name)
        if status == "implicit":
            return None, status, self.msg["type_implicit"].format(name=name)
        if status == "ambiguous":
            cands = res["candidates"]
            return None, status, self.msg["type_ambiguous"].format(name=name, count=len(cands),
                                                                   paths=", ".join(cands[:8]))
        return None, "unresolved", self.msg["type_unresolved"].format(name=name)

    def open_type(self, name: str) -> tuple[str, str]:
        res, status, error = self._resolve(name)
        if res is None:
            return status, error
        text = resolved_skeleton(res, self.snapshot()["cache"], self.opts, int(self.acfg["skeleton_max_members"]))
        return ("ok", text) if text else ("unresolved", self.msg["type_unresolved"].format(name=name))

    def read_method(self, type_name: str, method_name: str) -> tuple[str, str]:
        res, status, error = self._resolve(type_name)
        if res is None:
            return status, error
        hit = self.snapshot()["cache"].get(res["path"])
        node = find_type_node(hit[1], hit[0], res.get("inner") or "") if hit else None
        if node is None:
            return "unresolved", self.msg["type_unresolved"].format(name=type_name)
        src = hit[0]
        type_simple = declared_name(node, src)
        members = [m for m in body_members(node.child_by_field_name("body")) if m.type in METHOD_DECLS]
        wanted = method_name.strip().removesuffix("()")
        found = [m for m in members if declared_name(m, src) == wanted
                 or (m.type != "method_declaration" and wanted == type_simple)]
        if not found:
            names = sorted({declared_name(m, src) for m in members})
            return "method_not_found", self.msg["method_not_found"].format(method=wanted, type=type_simple,
                                                                           methods=", ".join(names[:60]) or "none")
        lines = src.decode("utf-8", errors="replace").splitlines()
        parts = [f"// {res['path']}:{first_line(m)}-{last_line(m)}\n" + "\n".join(lines[first_line(m) - 1:last_line(m)])
                 for m in found]
        return "ok", "\n\n".join(parts)

    def find_usages(self, name: str) -> tuple[str, str]:
        name = name.strip().removesuffix("()").split(".")[-1]
        if not IDENTIFIER_RE.match(name):
            return "invalid_name", self.msg["no_usages"].format(name=name)
        item = self.snapshot()
        if item is None:
            return "unavailable", self.msg["unavailable"]
        texts = self.store.texts(item)
        if "order" not in item or item.get("order_for") != self.pkg["file"]:
            item["order"] = proximity_order(item["index"]["files"], self.pkg["file"])
            item["order_for"] = self.pkg["file"]
        found = usage_lines(item["dir"], item["order"], name, set(), int(self.acfg["find_usages_max_lines"]),
                            int(self.acfg["find_usages_per_file"]), int(self.acfg["line_chars"]), texts)
        if not found["lines"]:
            return "empty", self.msg["no_usages"].format(name=name)
        head = self.msg["usages"].format(n_lines=found["n_lines"], n_files=found["n_files"], name=name,
                                         shown=len(found["lines"]))
        return "ok", head + "\n" + "\n".join(found["lines"])


# ---------------------------------------------------------------- backends


class VllmBackend:
    """vLLM offline engine (written against vLLM 0.30)."""

    name = "vllm"

    def __init__(self, model_ref: str, engine_cfg: dict[str, Any], temperature: float) -> None:
        from vllm import LLM, SamplingParams
        import vllm

        self.llm = LLM(model=model_ref, **engine_cfg)
        self.SamplingParams = SamplingParams
        self.temperature = temperature
        self.version = getattr(vllm, "__version__", None)

    def generate(self, prompts: list[list[int]], max_tokens: int, logprobs: int | None,
                 stop_token_ids: list[int] | None) -> list[dict[str, Any]]:
        params = self.SamplingParams(max_tokens=int(max_tokens), temperature=self.temperature,
                                     logprobs=logprobs, stop_token_ids=stop_token_ids or None,
                                     skip_special_tokens=False)
        outputs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
        results = []
        for out in outputs:
            gen = out.outputs[0]
            first = top_logprobs(gen.logprobs[0]) if logprobs and gen.logprobs else []
            results.append({"token_ids": list(gen.token_ids), "first_top": first,
                            "finish_reason": gen.finish_reason})
        return results


def default_stub_policy(prompt: str) -> str:
    """Stub completion: call get_enclosing_class then get_metrics when tools
    are offered, then answer no."""
    if "<tools>" not in prompt:
        return "no"
    done = prompt.count("<tool_response>")
    names = ["get_enclosing_class", "get_metrics"]
    if done >= len(names):
        return "no"
    name = names[done]
    if "<function=" in prompt:
        return f"<tool_call>\n<function={name}>\n</function>\n</tool_call>"
    return f'<tool_call>\n{{"name": "{name}", "arguments": {{}}}}\n</tool_call>'


class StubBackend:
    """Deterministic stand-in for vLLM in local tests."""

    name = "stub"
    version = None

    def __init__(self, tokenizer: Any, yes_ids: dict[int, str], no_ids: dict[int, str],
                 policy: Callable[[str], str] | None = None) -> None:
        self.tok = tokenizer
        self.yes = min(yes_ids)
        self.no = min(no_ids)
        self.yes_text, self.no_text = yes_ids[self.yes], no_ids[self.no]
        self.policy = policy or default_stub_policy
        self.calls: list[int] = []

    def generate(self, prompts: list[list[int]], max_tokens: int, logprobs: int | None,
                 stop_token_ids: list[int] | None) -> list[dict[str, Any]]:
        self.calls.append(len(prompts))
        results = []
        for ids in prompts:
            text = self.tok.decode(ids)
            completion = self.policy(text)
            out = self.tok.encode(completion, add_special_tokens=False)
            finish = "length" if len(out) > max_tokens else "stop"
            out = out[:max_tokens]
            first = []
            if logprobs:
                h = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                p = min(max(h, 0.01), 0.99)
                first = sorted([(self.yes, math.log(p), self.yes_text), (self.no, math.log(1 - p), self.no_text)],
                               key=lambda r: -r[1])
            results.append({"token_ids": out, "first_top": first, "finish_reason": finish})
        return results


def split_seconds(seconds: float, sizes: list[int]) -> list[float]:
    """Wall seconds of one generate call shared in proportion to tokens."""
    total = sum(sizes)
    if total <= 0:
        return [seconds / max(len(sizes), 1)] * len(sizes)
    return [seconds * s / total for s in sizes]


# ---------------------------------------------------------------- single-turn conditions


def output_paths(out_dir: Path, shard_index: int, shard_count: int) -> dict[str, Path]:
    suffix = f".shard{shard_index}of{shard_count}" if shard_count > 1 else ""
    return {"scores": out_dir / f"scores{suffix}.jsonl", "meta": out_dir / f"meta{suffix}.json",
            "transcripts": out_dir / f"transcripts{suffix}.jsonl"}


def static_row(idx: int, smell: str, level: str, condition: str, prompt_hash: str) -> dict[str, Any]:
    return {"json_index": idx, "smell": smell, "level": level, "condition": condition, "applicable": False,
            "score": 0.0, "p_yes": 0.0, "p_no": 0.0, "matched": [], "top_token": [None, None], "truncated": False,
            "prompt_tokens": 0, "generated_tokens": 0, "seconds_share": 0.0, "prompt_hash": prompt_hash}


def build_single_turn(
    condition: str,
    pairs: list[tuple[int, str]],
    levels: dict[int, str],
    entries: list[dict[str, Any]],
    packages: dict[int, dict[str, Any]],
    donors: dict[int, int | None],
    replays: dict[tuple[int, str], list[dict[str, Any]]],
    builder: PromptBuilder,
    tokenizer: Any,
    template_kwargs: dict[str, Any],
    max_len: int | None,
) -> list[dict[str, Any]]:
    """Pending prompts of a single-turn condition, with context budgets scaled
    down until each prompt fits in max_len."""
    prompts = builder.prompts
    scales = [float(s) for s in builder.budgets["fit_scales"]]
    template = prompts["facts_only_template"] if condition == "FACTS_ONLY" else prompts["user_template"]
    shows_code = condition != "FACTS_ONLY"
    context_cache: dict[tuple[int, float], tuple[str, dict[str, Any]]] = {}

    def context_at(idx: int, smell: str, scale: float) -> tuple[str, dict[str, Any]]:
        level = levels[idx]
        pkg = packages.get(idx)
        if condition == "REPLAY":
            return builder.context(condition, pkg, None, level, scale, replays[(idx, smell)])
        key = (idx, scale)
        if key not in context_cache:
            if condition == "PAD":
                c3, _ = builder.context("C3", pkg, None, level, scale)
                target = builder.ntok(c3)
                text = builder.pad_block(target)
                context_cache[key] = (text, {"pad": {"target_tokens": target, "tokens": builder.ntok(text)}})
            else:
                donor = donors.get(idx) if condition == "SHUF" else None
                context_cache[key] = builder.context(condition, pkg, packages.get(donor) if donor is not None
                                                     else None, level, scale)
        return context_cache[key]

    pending = []
    for idx, smell in pairs:
        code, truncated, code_tokens = builder.code(idx, entries[idx]["code_snippet"])
        ids, context, info, used_scale = None, "", {}, None
        for scale in scales:
            context, info = context_at(idx, smell, scale)
            user = builder.user_message(template, smell, code, truncated, context=context)
            messages = [{"role": "system", "content": prompts["system"]}, {"role": "user", "content": user}]
            ids = render_prompt(tokenizer, messages, template_kwargs)
            used_scale = scale
            if max_len is None or len(ids) + 1 <= max_len:
                break
        else:
            raise SystemExit(f"prompt of json_index {idx} / {smell} has {len(ids)} tokens at the smallest context "
                             f"scale, over max_model_len {max_len}: lower max_code_tokens")
        pending.append({"json_index": idx, "smell": smell, "level": levels[idx], "ids": ids,
                        "truncated": truncated and shows_code, "code_tokens": code_tokens if shows_code else 0,
                        "context_tokens": builder.ntok(context) if context else 0, "context_scale": used_scale,
                        "context": info, "donor": donors.get(idx) if condition == "SHUF" else None})
    return pending


def run_single_turn(pending: list[dict[str, Any]], backend: Any, own_file: Path, chunk: int, logprobs: int,
                    yes_ids: dict[int, str], no_ids: dict[int, str], condition: str,
                    prompt_hash: str) -> dict[str, Any]:
    gen_seconds, prompt_tokens, nan_run = 0.0, 0, 0
    for start in range(0, len(pending), chunk):
        batch = pending[start:start + chunk]
        t0 = time.time()
        outs = backend.generate([p["ids"] for p in batch], 1, logprobs, None)
        seconds = time.time() - t0
        gen_seconds += seconds
        shares = split_seconds(seconds, [len(p["ids"]) + 1 for p in batch])
        rows = []
        for p, out, share in zip(batch, outs, shares):
            result = score_answer(out["first_top"], yes_ids, no_ids)
            nan_run += int(math.isnan(result["score"]))
            prompt_tokens += len(p["ids"])
            rows.append({"json_index": p["json_index"], "smell": p["smell"], "level": p["level"],
                         "condition": condition, "applicable": True, **result, "truncated": p["truncated"],
                         "code_tokens": p["code_tokens"], "prompt_tokens": len(p["ids"]),
                         "context_tokens": p["context_tokens"], "context_scale": p["context_scale"],
                         "generated_tokens": len(out["token_ids"]), "seconds_share": round(share, 5),
                         "context": p["context"], "donor": p["donor"], "prompt_hash": prompt_hash})
        append_rows(own_file, rows)
        print(f"[context] {condition} {min(start + chunk, len(pending))}/{len(pending)} prompts, "
              f"{prompt_tokens / max(gen_seconds, 1e-9):.0f} prompt tokens/s, NaN so far {nan_run}", flush=True)
    return {"generate_seconds": round(gen_seconds, 1), "prompts": len(pending), "prompt_tokens": prompt_tokens,
            "generated_tokens": len(pending), "nan": nan_run}


def single_turn_summary(condition: str, pending: list[dict[str, Any]]) -> str:
    lengths = np.array([len(p["ids"]) for p in pending]) if pending else np.zeros(1)
    ctx = np.array([p["context_tokens"] for p in pending]) if pending else np.zeros(1)
    sections = lambda pred: dict(Counter(k for p in pending for k, v in p["context"].items()  # noqa: E731
                                         if "available" in v and pred(v)))
    pad = [p["context"]["pad"]["tokens"] - p["context"]["pad"]["target_tokens"] for p in pending
           if "pad" in p["context"]]
    text = (f"[context] {condition}: {len(pending)} prompts; prompt tokens mean {lengths.mean():.0f} "
            f"p50 {np.percentile(lengths, 50):.0f} p95 {np.percentile(lengths, 95):.0f} max {lengths.max():.0f}; "
            f"context tokens mean {ctx.mean():.0f} max {ctx.max():.0f}; truncated snippets "
            f"{sum(p['truncated'] for p in pending)}; context scaled down "
            f"{sum(1 for p in pending if p['context_scale'] < 1.0)}; truncated sections "
            f"{sections(lambda v: v['truncated'])}; empty sections {sections(lambda v: v['available'] and v['empty'])}; "
            f"unavailable sections {sections(lambda v: not v['available'])}")
    if pad:
        text += f"; PAD tokens minus target: min {min(pad)} max {max(pad)}"
    return text


# ---------------------------------------------------------------- agent condition


@dataclass
class Conversation:
    idx: int
    smell: str
    level: str
    messages: list[dict[str, Any]]
    tools: Any
    budget: int
    truncated: bool
    code_tokens: int
    calls: list[dict[str, Any]] = field(default_factory=list)
    # the tool message of each call (same order as calls)
    tool_messages: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    prompt_tokens_total: int = 0
    generated_tokens: int = 0
    seconds: float = 0.0
    stop_reason: str | None = None
    final_result: dict[str, Any] = field(default_factory=dict)
    final_prompt_tokens: int = 0
    final_fit: dict[str, int] = field(default_factory=dict)


def strip_stop(ids: list[int], stop_ids: list[int]) -> list[int]:
    ids = list(ids)
    while ids and ids[-1] in stop_ids:
        ids.pop()
    return ids


def fit_final(conv: Conversation, render: Callable[[list[dict[str, Any]]], list[int]], max_len: int | None,
              note: str) -> list[int]:
    """Final prompt ids, replacing long tool outputs then dropping messages
    until the prompt fits."""
    ids = render(conv.messages)
    replaced = dropped = 0
    while max_len is not None and len(ids) + 1 > max_len:
        tool_msgs = [m for m in conv.messages[2:-1] if m["role"] == "tool" and m["content"] != note]
        if tool_msgs:
            max(tool_msgs, key=lambda m: len(m["content"]))["content"] = note
            replaced += 1
        elif len(conv.messages) > 3:
            del conv.messages[-2]
            dropped += 1
        else:
            raise SystemExit(f"final prompt of json_index {conv.idx} / {conv.smell} has {len(ids)} tokens, "
                             f"over max_model_len {max_len}, without any tool output: lower max_code_tokens")
        ids = render(conv.messages)
    conv.final_fit = {"outputs_replaced": replaced, "messages_dropped": dropped}
    return ids


def run_agent_chunk(
    convs: list[Conversation],
    backend: Any,
    tokenizer: Any,
    tools_schema: list[dict[str, Any]],
    template_kwargs: dict[str, Any],
    acfg: dict[str, Any],
    agent_prompts: dict[str, Any],
    builder: PromptBuilder,
    max_len: int | None,
    stop_ids: list[int],
    logprobs: int,
    yes_ids: dict[int, str],
    no_ids: dict[int, str],
) -> dict[str, Any]:
    """Advance the conversations in lockstep until each stops, then score the
    final question. Returns cost counters of the chunk."""
    max_turn = int(acfg["max_turn_tokens"])
    max_out = int(acfg["max_tool_output_tokens"])
    max_calls = int(acfg["max_tool_calls"])
    overhead = int(acfg["message_overhead_tokens"])
    msgs = agent_prompts["messages"]
    marker = builder.marker
    render = lambda m: render_chat(tokenizer, m, template_kwargs, tools_schema)  # noqa: E731
    final_text = {c.smell: agent_prompts["final_question"].format(
        question=builder.prompts["smells"][c.smell]["question"]) for c in convs}
    # the reserve always holds the final question and its chat markup
    margin = int(acfg["final_question_margin_tokens"])
    reserve = max([int(acfg["final_reserve_tokens"])] + [builder.ntok(t) + margin for t in final_text.values()])
    stats = Counter()
    seconds_total = 0.0
    active = list(convs)
    while active:
        batch = []
        for conv in active:
            ids = render(conv.messages)
            if max_len is not None and len(ids) + max_turn + reserve > max_len:
                conv.stop_reason = "context_limit"
                continue
            batch.append((conv, ids))
        if not batch:
            break
        t0 = time.time()
        outs = backend.generate([ids for _, ids in batch], max_turn, None, stop_ids)
        seconds = time.time() - t0
        seconds_total += seconds
        stats["generate_calls"] += 1
        sizes = [len(ids) + len(out["token_ids"]) for (_, ids), out in zip(batch, outs)]
        for (conv, ids), out, share in zip(batch, outs, split_seconds(seconds, sizes)):
            conv.turns += 1
            conv.prompt_tokens_total += len(ids)
            conv.generated_tokens += len(out["token_ids"])
            conv.seconds += share
            gen_ids = strip_stop(out["token_ids"], stop_ids)
            text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            conv.messages.append({"role": "assistant", "content": text})
            calls = parse_tool_calls(text)
            if not calls:
                conv.stop_reason = "answered" if out["finish_reason"] != "length" else "turn_length"
                continue
            used = len(ids) + len(gen_ids)
            for call in calls:
                record = {"name": call["name"], "arguments": call["arguments"], "status": None, "output_tokens": 0}
                if conv.budget <= 0:
                    record["status"] = "over_budget"
                    output = msgs["budget_used"].format(max_tool_calls=max_calls)
                else:
                    conv.budget -= 1
                    if call["error"] is not None:
                        record["status"] = "malformed"
                        record["error"] = call["error"]
                        record["raw"] = call["raw"]
                        output = msgs["malformed"].format(error=call["error"])
                    else:
                        record["status"], output = conv.tools.call(call["name"], call["arguments"])
                room = (max_len - reserve - used - overhead) if max_len is not None else max_out
                output, cut_, n_full = fit(output, tokenizer, max(1, min(max_out, room)), marker)
                shown = builder.ntok(output)
                record["output_tokens"] = shown
                record["output_tokens_before_cut"] = n_full
                record["output_truncated"] = cut_
                used += shown + overhead
                message = {"role": "tool", "content": output}
                conv.calls.append(record)
                conv.tool_messages.append(message)
                conv.messages.append(message)
                stats[f"status_{record['status']}"] += 1
            if conv.budget <= 0:
                conv.stop_reason = "budget"
        active = [c for c in active if c.stop_reason is None]

    note = msgs["output_dropped"]
    final_ids = []
    for conv in convs:
        conv.messages.append({"role": "user", "content": final_text[conv.smell]})
        final_ids.append(fit_final(conv, render, max_len, note))
    t0 = time.time()
    outs = backend.generate(final_ids, 1, logprobs, None)
    seconds = time.time() - t0
    seconds_total += seconds
    stats["generate_calls"] += 1
    for conv, ids, out, share in zip(convs, final_ids, outs, split_seconds(seconds, [len(i) + 1 for i in final_ids])):
        conv.prompt_tokens_total += len(ids)
        conv.generated_tokens += 1
        conv.seconds += share
        conv.final_result = score_answer(out["first_top"], yes_ids, no_ids)
        conv.final_prompt_tokens = len(ids)
    stats["seconds"] = seconds_total
    return dict(stats)


def tool_outputs(conv: Conversation, note: str) -> list[dict[str, Any]]:
    """Every call with the output text the model saw at the final question."""
    live = {id(m) for m in conv.messages}
    out = []
    for call, message in zip(conv.calls, conv.tool_messages):
        text = message["content"] if id(message) in live else None
        out.append({"name": call["name"], "arguments": call["arguments"], "status": call["status"], "text": text,
                    "replaced": text == note})
    return out


def agent_rows(convs: list[Conversation], prompt_hash: str, note: str,
               condition: str = "C4") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, transcripts = [], []
    for conv in convs:
        status = Counter(c["status"] for c in conv.calls)
        rows.append({"json_index": conv.idx, "smell": conv.smell, "level": conv.level, "condition": condition,
                     "applicable": True, **conv.final_result, "truncated": conv.truncated,
                     "code_tokens": conv.code_tokens, "prompt_tokens": conv.final_prompt_tokens,
                     "prompt_tokens_total": conv.prompt_tokens_total, "generated_tokens": conv.generated_tokens,
                     "seconds_share": round(conv.seconds, 5), "turns": conv.turns, "n_tool_calls": len(conv.calls),
                     "n_tool_ok": sum(status[s] for s in TOOL_OK),
                     "n_tool_unresolved": sum(status[s] for s in TOOL_UNRESOLVED),
                     "tool_calls": [{k: v for k, v in c.items() if k != "raw"} for c in conv.calls],
                     "stop_reason": conv.stop_reason, "final_fit": conv.final_fit, "prompt_hash": prompt_hash})
        transcripts.append({"json_index": conv.idx, "smell": conv.smell, "stop_reason": conv.stop_reason,
                            "messages": conv.messages[2:], "tool_calls": conv.calls,
                            "tool_outputs": tool_outputs(conv, note), "prompt_hash": prompt_hash})
    return rows, transcripts


# ---------------------------------------------------------------- main


def load_backend(name: str, model_ref: str, engine_cfg: dict[str, Any], temperature: float, tokenizer: Any,
                 yes_ids: dict[int, str], no_ids: dict[int, str]) -> Any:
    if name == "stub":
        return StubBackend(tokenizer, yes_ids, no_ids)
    return VllmBackend(model_ref, engine_cfg, temperature)


def replay_items(outputs: list[dict[str, Any]], statuses: list[str]) -> list[dict[str, Any]]:
    """C4 outputs shown by REPLAY: listed statuses still in the final prompt."""
    return [o for o in outputs if o["status"] in statuses and o.get("text") is not None and not o.get("replaced")]


def main(argv: list[str] | None = None, backend_factory: Callable[..., Any] | None = None,
         tokenizer: Any = None) -> Path:
    """Run one condition (backend_factory and tokenizer are test hooks)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--judge", type=str, default=None, help="judge name under context_study.judge.judges")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--backend", choices=("vllm", "stub"), default="vllm")
    parser.add_argument("--tokenizer", type=str, default=None, help="tokenizer path (default: the judge model)")
    parser.add_argument("--dry-run", action="store_true", help="build prompts and print stats without generation")
    args, extra = parser.parse_known_args(argv)
    cfg = load_config(args.config, parse_cli_overrides(extra))
    ccfg = cfg["context_study"]
    jcfg = ccfg["judge"]
    judge = args.judge or str(jcfg["default_judge"])
    judge_cfg = jcfg["judges"][judge]
    condition = args.condition
    name = f"{judge}{jcfg.get('tag') or ''}"
    out_dir = resolve(jcfg["out_root"]) / name / condition

    prompts = yaml.safe_load(resolve(jcfg["prompts"]).read_text())
    folds = load_folds(resolve(ccfg["folds"]))
    label_order = list(folds["meta"]["label_order"])
    entries = json.loads(resolve(ccfg["input_json"]).read_text())
    levels = read_levels(resolve(ccfg["sample_meta"]))
    smell_levels = {s: str(prompts["smells"][s]["level"]) for s in label_order}
    labels = {i: label_vector(e) for i, e in enumerate(entries)}
    all_indices = [int(i) for i in folds["meta"]["json_index"]]
    packages_path = resolve(ccfg["packages"]["jsonl"])
    packages = load_packages(packages_path) if condition != "C0" else {}
    packages_sha = file_sha256(packages_path) if condition != "C0" else None

    model_ref, model_source = model_reference(judge_cfg)
    tokenizer_ref = "injected" if tokenizer is not None else (args.tokenizer or model_ref)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref)
    template_kwargs, dropped = supported_template_kwargs(tokenizer, dict(judge_cfg.get("chat_template_kwargs") or {}))
    answer_cfg = dict(jcfg["answer"])
    yes_ids = answer_token_ids(tokenizer, list(answer_cfg["yes_variants"]))
    no_ids = answer_token_ids(tokenizer, list(answer_cfg["no_variants"]))
    if not yes_ids or not no_ids:
        raise SystemExit(f"no single-token yes/no variant in this tokenizer: yes={yes_ids} no={no_ids}")
    sampling_cfg = dict(jcfg["sampling"])
    engine_cfg = dict(judge_cfg.get("engine") or {})
    max_len = int(engine_cfg.get("max_model_len", 0)) or None
    tool_inputs = None
    if condition in ("C4", "REPLAY"):
        tool_inputs = {"package_options": ccfg["packages"]["options"],
                       "size_features_sha": file_sha256(resolve(ccfg["size_features"]))}
    c4_hash = (fingerprint("C4", prompts, jcfg, judge_cfg, packages_sha, args.backend, tool_inputs=tool_inputs)
               if condition == "REPLAY" else None)
    prompt_hash = fingerprint(condition, prompts, jcfg, judge_cfg, packages_sha, args.backend, c4_hash, tool_inputs)
    builder = PromptBuilder(prompts, tokenizer, dict(jcfg["budgets"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(out_dir, args.shard_index, args.shard_count)
    if not args.dry_run:
        repair_scores_file(paths["scores"])
        if condition in MULTI_TURN:
            repair_scores_file(paths["transcripts"])
    done = scored_pairs(out_dir, prompt_hash)
    print(f"[context] {name} {condition}: prompt hash {prompt_hash}, {len(done)} pairs already scored; "
          f"template kwargs {template_kwargs} (dropped {dropped}); yes {yes_ids} no {no_ids}", flush=True)

    started = time.time()
    run_info: dict[str, Any] = {}
    subset_info = None
    replay_info = None
    if condition in ("C4", "REPLAY", "REASON"):
        subset = config_subset(jcfg, all_indices, levels, labels, label_order, smell_levels)
        subset_info = {s: {"level": v["level"], "positives": len(v["positives"]), "negatives": len(v["negatives"])}
                       for s, v in subset.items()}
        if condition == "C4":
            write_json_atomic(out_dir / "subset.json", {"config": jcfg["agent_subset"], "strata": subset})
        pairs = subset_pairs(subset, label_order)
        # conversations grouped by snapshot, so that the snapshot caches are reused
        pairs.sort(key=lambda p: ((packages.get(p[0]) or {}).get("snapshot") or "", p[0], label_order.index(p[1])))
        pairs = shard(pairs, args.shard_index, args.shard_count)

    if condition in SINGLE_TURN:
        replays: dict[tuple[int, str], list[dict[str, Any]]] = {}
        if condition == "REPLAY":
            statuses = list(jcfg["replay"]["statuses"])
            c4_dir = out_dir.parent / "C4"
            c4_scored = scored_pairs(c4_dir, c4_hash)
            loaded = load_replays(c4_dir, c4_hash, c4_scored)
            waiting = [p for p in pairs if p not in loaded]
            replays = {p: replay_items(loaded[p], statuses) for p in pairs if p in loaded}
            replay_info = {"pairs": len(pairs), "with_transcript": len(replays), "waiting_for_c4": len(waiting),
                           "c4_prompt_hash": c4_hash, "statuses": statuses,
                           "empty": sum(1 for v in replays.values() if not v)}
            print(f"[context] REPLAY: {replay_info}", flush=True)
            todo = [p for p in pairs if p in replays and p not in done]
            static_rows: list[dict[str, Any]] = []
        else:
            indices = list(all_indices)
            smoke = jcfg.get("smoke") or {}
            if smoke.get("n_per_level"):
                chosen = set(select_subset(indices, levels, labels, label_order, smell_levels,
                                           int(smoke["n_per_level"]), int(smoke.get("min_positives_per_smell", 0)),
                                           int(smoke.get("seed", 0))))
                if smoke.get("include_agent_subset"):
                    agent = config_subset(jcfg, all_indices, levels, labels, label_order, smell_levels)
                    chosen |= {i for i, _ in subset_pairs(agent, label_order)}
                indices = sorted(chosen)
            indices = shard(indices, args.shard_index, args.shard_count)
            todo = [(i, s) for i in indices for s in label_order if (i, s) not in done and smell_levels[s] == levels[i]]
            static_rows = [static_row(i, s, levels[i], condition, prompt_hash) for i in indices for s in label_order
                           if (i, s) not in done and smell_levels[s] != levels[i]]
            print(f"[context] {condition}: {len(indices)} samples in shard {args.shard_index}/{args.shard_count}, "
                  f"{len(static_rows)} non-applicable rows", flush=True)
        donors = {}
        if condition == "SHUF":
            repos = {i: repo_name(entries[i]["repo_url"]) for i in all_indices}
            donors = shuffle_donors(all_indices, levels, packages, repos, int(jcfg["shuffle_seed"]))
        pending = build_single_turn(condition, todo, levels, entries, packages, donors, replays, builder, tokenizer,
                                    template_kwargs, max_len)
        print(single_turn_summary(condition, pending), flush=True)
        if args.dry_run:
            preview = out_dir / "prompt_preview.txt"
            shown: dict[str, Any] = {}
            for p in pending:
                shown.setdefault(p["smell"], p)
            preview.write_text("\n\n".join(f"===== json_index {p['json_index']} {p['smell']}\n"
                                           + tokenizer.decode(p["ids"]) for p in shown.values()))
            print(f"[context] dry run: wrote {preview}", flush=True)
            return out_dir
        if static_rows:
            append_rows(paths["scores"], static_rows)
        if pending:
            t_load = time.time()
            backend = (backend_factory or load_backend)(args.backend, model_ref, engine_cfg,
                                                        float(sampling_cfg["temperature"]), tokenizer, yes_ids, no_ids)
            run_info["load_seconds"] = round(time.time() - t_load, 1)
            run_info.update(run_single_turn(pending, backend, paths["scores"], int(jcfg["chunk_size"]),
                                            int(sampling_cfg["logprobs"]), yes_ids, no_ids, condition, prompt_hash))
            run_info["backend_version"] = backend.version
    else:
        try:
            if condition == "C4":
                java_parser()
        except ImportError as exc:
            # open_type and read_method would fail on every call
            raise SystemExit(f"C4 needs tree_sitter and tree_sitter_java ({exc}); "
                             "scripts/slurm/context_judge.sbatch links them from the graph env") from exc
        acfg = dict(jcfg["agent"])
        pending_pairs = [p for p in pairs if p not in done]
        tools_schema = list(prompts["agent"]["tools"]) if condition == "C4" else None
        tool_names = [t["function"]["name"] for t in tools_schema or []]
        conv_prompts = prompts["agent"] if condition == "C4" else {**prompts["agent"], **prompts["reason"]}
        store = SnapshotStore(resolve(ccfg["snapshots_dir"]), resolve(ccfg["packages"]["type_index_dir"]),
                              int(acfg["snapshot_cache"]))
        sizes = read_size_features(resolve(ccfg["size_features"]))
        opts = dict(ccfg["packages"]["options"])
        implicit = set(opts["implicit_types"])
        stop_ids = [int(t) for t in (tokenizer.convert_tokens_to_ids(s) for s in judge_cfg.get("stop_tokens") or [])
                    if t is not None and t != tokenizer.unk_token_id]
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in stop_ids:
            stop_ids.append(int(tokenizer.eos_token_id))
        note = prompts["agent"]["messages"]["output_dropped"]

        def new_conversation(idx: int, smell: str) -> Conversation:
            code, truncated, code_tokens = builder.code(idx, entries[idx]["code_snippet"])
            user = builder.user_message(conv_prompts["user_template"], smell, code, truncated,
                                        max_tool_calls=int(acfg["max_tool_calls"]))
            tools = RepositoryTools(packages.get(idx), store, sizes.get(idx), prompts["agent"]["messages"], acfg,
                                    opts, implicit, tool_names)
            return Conversation(idx=idx, smell=smell, level=levels[idx],
                                messages=[{"role": "system", "content": conv_prompts["system"]},
                                          {"role": "user", "content": user}],
                                tools=tools, budget=int(acfg["max_tool_calls"]) if tools_schema else 0,
                                truncated=truncated,
                                code_tokens=code_tokens)

        print(f"[context] {condition}: subset {subset_info}; {len(pairs)} conversations in shard "
              f"{args.shard_index}/{args.shard_count}, {len(pending_pairs)} pending; stop ids {stop_ids}", flush=True)
        if args.dry_run:
            preview_lines = []
            lengths = []
            for idx, smell in pending_pairs:
                conv = new_conversation(idx, smell)
                lengths.append(len(render_chat(tokenizer, conv.messages, template_kwargs, tools_schema)))
            statuses: Counter = Counter()
            for k, (idx, smell) in enumerate(pending_pairs[: int(acfg["dry_run_tool_samples"])]):
                conv = new_conversation(idx, smell)
                # probe arguments for each tool
                pkg = packages.get(idx) or {}
                target = (pkg.get("target") or {}).get("name", "")
                owner = (pkg.get("enclosing") or {}).get("name", "") if levels[idx] == "method" else target
                first_ref = ((pkg.get("referenced") or {}).get("resolved") or [{}])[0].get("name") or owner
                calls = {"get_enclosing_class": {}, "get_metrics": {}, "open_type": {"name": first_ref},
                         "read_method": {"type_name": owner, "method_name": target},
                         "find_usages": {"name": target}}
                if k < 2:
                    preview_lines.append(f"===== json_index {idx} {smell}\n"
                                         + tokenizer.decode(render_chat(tokenizer, conv.messages, template_kwargs,
                                                                        tools_schema)))
                for tool in tool_names:
                    status, text = conv.tools.call(tool, calls.get(tool, {}))
                    statuses[(tool, status)] += 1
                    if k < 2:
                        preview_lines.append(f"----- {tool}({calls.get(tool, {})}) -> {status}\n{text[:3000]}")
            arr = np.array(lengths) if lengths else np.zeros(1)
            print(f"[context] C4 dry run: initial prompt tokens mean {arr.mean():.0f} p95 {np.percentile(arr, 95):.0f} "
                  f"max {arr.max():.0f}; tool statuses on {min(len(pending_pairs), int(acfg['dry_run_tool_samples']))} "
                  f"conversations {dict(sorted(statuses.items()))}", flush=True)
            preview = out_dir / "prompt_preview.txt"
            preview.write_text("\n\n".join(preview_lines))
            print(f"[context] dry run: wrote {preview}", flush=True)
            return out_dir
        if pending_pairs:
            t_load = time.time()
            backend = (backend_factory or load_backend)(args.backend, model_ref, engine_cfg,
                                                        float(sampling_cfg["temperature"]), tokenizer, yes_ids, no_ids)
            run_info["load_seconds"] = round(time.time() - t_load, 1)
            run_info["backend_version"] = backend.version
            totals = Counter()
            chunk = int(acfg["chunk_size"])
            for start in range(0, len(pending_pairs), chunk):
                convs = [new_conversation(i, s) for i, s in pending_pairs[start:start + chunk]]
                stats = run_agent_chunk(convs, backend, tokenizer, tools_schema, template_kwargs, acfg,
                                        conv_prompts, builder, max_len, stop_ids, int(sampling_cfg["logprobs"]),
                                        yes_ids, no_ids)
                rows, transcripts = agent_rows(convs, prompt_hash, note, condition)
                # transcripts first: a scored pair always has its transcript (REPLAY)
                append_rows(paths["transcripts"], transcripts)
                append_rows(paths["scores"], rows)
                totals.update(stats)
                totals["conversations"] += len(convs)
                totals["prompt_tokens"] += sum(c.prompt_tokens_total for c in convs)
                totals["generated_tokens"] += sum(c.generated_tokens for c in convs)
                totals["tool_calls"] += sum(len(c.calls) for c in convs)
                totals["nan"] += sum(1 for c in convs if math.isnan(c.final_result["score"]))
                for c in convs:
                    totals[f"stop_{c.stop_reason}"] += 1
                    for call in c.calls:
                        totals[f"tool_{call['name']}"] += 1
                print(f"[context] {condition} {min(start + chunk, len(pending_pairs))}/{len(pending_pairs)} conversations, "
                      f"{totals['tool_calls']} tool calls, {totals['generate_calls']} generate calls, "
                      f"{totals['seconds']:.0f}s generating, NaN so far {totals['nan']}", flush=True)
            run_info.update({k: (round(v, 1) if isinstance(v, float) else v) for k, v in totals.items()})
            run_info["generate_seconds"] = round(float(totals["seconds"]), 1)

    if not run_info:
        print(f"[context] {condition}: nothing left to score", flush=True)
    gpu = None
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    mpath = paths["meta"]
    meta = json.loads(mpath.read_text()) if mpath.exists() else {"runs": []}
    meta.update({
        "judge": judge, "name": name, "condition": condition, "model": model_ref, "model_source": model_source,
        "tokenizer": tokenizer_ref, "backend": args.backend, "prompts_file": str(jcfg["prompts"]),
        "prompt_hash": prompt_hash, "packages_file": str(ccfg["packages"]["jsonl"]), "packages_sha": packages_sha,
        "chat_template_kwargs": template_kwargs, "chat_template_kwargs_dropped": dropped,
        "answer_tokens": {"yes": yes_ids, "no": no_ids}, "engine": engine_cfg, "sampling": sampling_cfg,
        "budgets": jcfg["budgets"], "shard": [args.shard_index, args.shard_count],
    })
    if condition == "SHUF":
        meta["shuffle_seed"] = jcfg["shuffle_seed"]
    if condition in ("C4", "REPLAY", "REASON"):
        meta.update({"agent": jcfg["agent"], "agent_subset": jcfg["agent_subset"], "subset": subset_info,
                     "tool_inputs": tool_inputs})
    if condition == "REPLAY":
        meta["replay"] = replay_info
    if run_info:
        meta["runs"].append({"host": socket.gethostname(), "gpu": gpu, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                             "seconds": round(time.time() - started, 1), **run_info})
    rows = read_valid_rows(paths["scores"])
    meta["totals"] = file_totals(rows)
    applicable = [r for r in rows if r.get("applicable")]
    meta["totals"]["costs"] = {
        "queries": len(applicable),
        "prompt_tokens": int(sum(r.get("prompt_tokens_total", r["prompt_tokens"]) for r in applicable)),
        "context_tokens": int(sum(r.get("context_tokens", 0) for r in applicable)),
        "generated_tokens": int(sum(r.get("generated_tokens", 0) for r in applicable)),
        "seconds": round(sum(r.get("seconds_share", 0.0) for r in applicable), 1),
        "tool_calls": int(sum(r.get("n_tool_calls", 0) for r in applicable)),
        "context_scaled_down": int(sum(1 for r in applicable if r.get("context_scale", 1.0) < 1.0)),
    }
    write_json_atomic(mpath, meta)
    print(f"[context] done in {time.time() - started:.0f}s; totals {meta['totals']}", flush=True)
    print(f"[context] wrote {paths['scores']} and {mpath}", flush=True)
    return out_dir


if __name__ == "__main__":
    main()
