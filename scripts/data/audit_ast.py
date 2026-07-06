"""Audit the AST graphs produced by the construction stage.

- Per graph: parse status, Child tree, NextToken path, token coverage against tree-sitter
- Dataset: exact duplicates, label conflicts and train/test leakage per split seed
- Writes a JSON report and a markdown summary

Usage:
    python scripts/data/audit_ast.py --manifest <manifest.jsonl> --normalized <normalized.json> \
        --out-json <report.json> --out-md <report.md>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import tree_sitter as ts
import tree_sitter_java as tsj

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.constants import LABEL_ORDER

TS_PARSER = ts.Parser(ts.Language(tsj.language()))
TS_ATOMIC = {"string_literal", "character_literal", "text_block"}
TS_COMMENTS = {"line_comment", "block_comment", "comment"}

NODE_RE = re.compile(r'^"(n\d+)" \[(.*)\]$')
EDGE_RE = re.compile(r'^"(n\d+)" -> "(n\d+)" \[edge_type="(\w+)"\]$')
ATTR_RE = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|-?\d+|true|false)')


ESCAPES = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}


def unescape_dot(value: str) -> str:
    # inverse of _escape_dot in build_ast_dot_from_normalized.py
    return re.sub(r'\\(.)', lambda m: ESCAPES.get(m.group(1), m.group(0)), value)


def parse_dot(path: Path) -> tuple[dict[str, dict], list[tuple[str, str, str]]]:
    nodes: dict[str, dict] = {}
    edges: list[tuple[str, str, str]] = []
    for line in path.read_text().splitlines():
        m = EDGE_RE.match(line)
        if m:
            edges.append((m.group(1), m.group(2), m.group(3)))
            continue
        m = NODE_RE.match(line)
        if m:
            attrs: dict = {}
            for key, raw in ATTR_RE.findall(m.group(2)):
                if raw.startswith('"'):
                    attrs[key] = unescape_dot(raw[1:-1])
                elif raw in {"true", "false"}:
                    attrs[key] = raw == "true"
                else:
                    attrs[key] = int(raw)
            nodes[m.group(1)] = attrs
    return nodes, edges


def check_tree(nodes: dict[str, dict], edges: list[tuple[str, str, str]]) -> list[str]:
    problems: list[str] = []
    syntax = {nid for nid, a in nodes.items() if a.get("node_kind") in {"SyntaxNode", "SyntaxToken"}}
    parents: dict[str, list[str]] = defaultdict(list)
    children: dict[str, list[str]] = defaultdict(list)
    for src, dst, kind in edges:
        if kind == "Child":
            parents[dst].append(src)
            children[src].append(dst)
    roots = [nid for nid in syntax if not parents.get(nid)]
    if len(roots) != 1:
        problems.append(f"roots={len(roots)}")
    multi = [nid for nid in syntax if len(parents.get(nid, [])) > 1]
    if multi:
        problems.append(f"multi_parent={len(multi)}")
    if roots:
        seen = {roots[0]}
        stack = [roots[0]]
        while stack:
            for child in children.get(stack.pop(), []):
                if child in seen:
                    problems.append("cycle_or_shared_child")
                    stack.clear()
                    break
                seen.add(child)
                stack.append(child)
        unreachable = len(syntax - seen)
        if unreachable:
            problems.append(f"unreachable={unreachable}")
    return problems


def check_next_token(nodes: dict[str, dict], edges: list[tuple[str, str, str]]) -> list[str]:
    problems: list[str] = []
    tokens = sorted(
        (a["token_index"], nid) for nid, a in nodes.items() if a.get("node_kind") == "SyntaxToken"
    )
    expected = {(tokens[i][1], tokens[i + 1][1]) for i in range(len(tokens) - 1)}
    actual = {(s, d) for s, d, k in edges if k == "NextToken"}
    if actual != expected:
        problems.append(f"next_token_mismatch missing={len(expected - actual)} extra={len(actual - expected)}")
    positions = [(nodes[nid]["start_line"], nodes[nid]["start_col"]) for _, nid in tokens]
    if any(b < a for a, b in zip(positions, positions[1:])):
        problems.append("token_positions_not_increasing")
    if [idx for idx, _ in tokens] != list(range(len(tokens))):
        problems.append("token_index_not_contiguous")
    return problems


def ts_tokens(source: bytes) -> tuple[list[str], bool]:
    tree = TS_PARSER.parse(source)
    out: list[str] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in TS_COMMENTS:
            continue
        if node.child_count == 0 or node.type in TS_ATOMIC:
            if node.text:
                out.append(node.text.decode("utf-8", errors="replace"))
            continue
        stack.extend(reversed(node.children))
    return out, tree.root_node.has_error


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def wrap(snippet: str) -> str:
    # must mirror _wrap_snippet_in_class in build_ast_dot_from_normalized.py
    body = snippet.rstrip("\n")
    return f"public class SnippetWrapper {{\n{body}\n}}\n"


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(p: float) -> float:
        return float(ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1)))])

    return {
        "n": len(ordered),
        "mean": round(statistics.mean(ordered), 2),
        "p50": at(0.5),
        "p90": at(0.9),
        "p99": at(0.99),
        "max": float(ordered[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--dot-dir", type=Path, default=None, help="Resolve DOT files by name here (manifests may hold stale absolute paths)")
    parser.add_argument("--dataset", type=Path, default=None, help="dataset.pt, for split leakage checks")
    parser.add_argument("--split-seeds", default="", help="comma separated seeds")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-labels", type=int, default=4)
    parser.add_argument("--context-dir", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    entries = json.loads(args.normalized.read_text())
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: int(r["json_index"]))
    if args.limit is not None:
        rows = rows[: args.limit]
    labels = [label.removeprefix("is_") for label in LABEL_ORDER[: args.num_labels]]

    status = Counter()
    failed_by_label = Counter()
    positives_by_label = Counter()
    error_kinds = Counter()
    problems = Counter()
    examples: dict[str, list] = defaultdict(list)
    sizes = defaultdict(list)
    durations = []
    token_match = Counter()
    ts_error = 0
    provenance = Counter()

    file_key = None
    if args.context_dir is not None:
        from scripts.data.fetch_context_files import file_key  # noqa: F811

    def note(kind: str, json_index: int, detail: str = "") -> None:
        problems[kind] += 1
        if len(examples[kind]) < args.max_examples:
            examples[kind].append({"json_index": json_index, "detail": detail[:300]})

    for row in rows:
        idx = int(row["json_index"])
        entry = entries[idx]
        y = [bool(v) for v in row.get("y", [])][: args.num_labels]
        for i, flag in enumerate(y):
            if flag:
                positives_by_label[labels[i]] += 1
        status[row["status"]] += 1
        if row.get("duration_ms") is not None:
            durations.append(float(row["duration_ms"]))
        if row["status"] != "success":
            for i, flag in enumerate(y):
                if flag:
                    failed_by_label[labels[i]] += 1
            msg = str(row.get("error") or "")
            kind = re.sub(r"line \d+:\d+ ", "", msg.split(";")[0])[:80]
            error_kinds[kind] += 1
            note("parse_failed", idx, msg)
            continue

        dot_path = Path(row["output_dot"])
        if args.dot_dir is not None:
            dot_path = args.dot_dir / dot_path.name
        nodes, edges = parse_dot(dot_path)
        for p in check_tree(nodes, edges):
            note("tree:" + p.split("=")[0], idx, p)
        for p in check_next_token(nodes, edges):
            note("next_token:" + p.split(" ")[0], idx, p)

        stats = row.get("graph_stats") or {}
        sizes["nodes"].append(stats.get("node_count", len(nodes)))
        sizes["tokens"].append(stats.get("syntax_token_count", 0))
        for i, flag in enumerate(y):
            if flag:
                sizes[f"nodes_{labels[i]}"].append(stats.get("node_count", len(nodes)))

        antlr_text = "".join(
            a.get("token_text", "")
            for a in sorted(
                (a for a in nodes.values() if a.get("node_kind") == "SyntaxToken"),
                key=lambda a: a["token_index"],
            )
        )
        snippet = str(entry.get("code_snippet", ""))
        ts_toks, has_err = ts_tokens(wrap(snippet).encode("utf-8"))
        ts_error += int(has_err)
        if squash(antlr_text) == squash("".join(ts_toks)):
            token_match["exact"] += 1
        else:
            token_match["mismatch"] += 1
            a, b = squash(antlr_text), squash("".join(ts_toks))
            cut = next((i for i, (u, v) in enumerate(zip(a, b)) if u != v), min(len(a), len(b)))
            note("token_coverage_mismatch", idx, f"antlr[{cut}:]={a[cut:cut + 60]!r} ts[{cut}:]={b[cut:cut + 60]!r}")

        if file_key is not None:
            path = args.context_dir / f"{file_key(entry['repo_url'], entry['commit_hash'], entry['file_path'])}.java"
            if not path.exists():
                provenance["file_missing"] += 1
            else:
                lines = path.read_text(errors="replace").splitlines()
                start, end = int(entry["start_line"]), int(entry["end_line"])
                region = "\n".join(lines[start - 1 : end])
                if squash(region) == squash(snippet):
                    provenance["exact"] += 1
                elif squash(snippet) in squash("\n".join(lines)):
                    provenance["found_elsewhere_in_file"] += 1
                    note("provenance_shifted", idx, f"lines {start}-{end}")
                else:
                    provenance["not_found"] += 1
                    note("provenance_not_found", idx, f"lines {start}-{end}")

    # exact duplicates over the whole normalized file
    by_hash: dict[str, list[int]] = defaultdict(list)
    for i, entry in enumerate(entries):
        by_hash[hashlib.sha1(squash(str(entry.get("code_snippet", ""))).encode()).hexdigest()].append(i)
    dup_groups = [g for g in by_hash.values() if len(g) > 1]
    conflict_groups = [
        g for g in dup_groups
        if len({tuple(bool(v) for v in entries[i].get("y", [])) for i in g}) > 1
    ]
    hash_of = {i: h for h, g in by_hash.items() for i in g}

    leakage: dict[str, dict] = {}
    seeds = [int(s) for s in args.split_seeds.split(",") if s.strip()]
    if seeds and args.dataset is not None:
        from scripts.train_gcn_baseline import split_sample_level

        dataset = torch.load(args.dataset, weights_only=False)
        for d in dataset:
            d.y = d.y.view(-1).int()
        json_ids = [int(d.json_index) for d in dataset]
        for seed in seeds:
            _, _, _, split = split_sample_level(dataset, args.train_ratio, args.val_ratio, seed, args.num_labels)
            train_hashes = Counter(hash_of[json_ids[i]] for i in split["train"])
            test_ids = [json_ids[i] for i in split["test"]]
            leaked = [j for j in test_ids if train_hashes.get(hash_of[j], 0) > 0]
            leaked_pos = Counter()
            for j in leaked:
                for k, flag in enumerate(entries[j].get("y", [])[: args.num_labels]):
                    if flag:
                        leaked_pos[labels[k]] += 1
            test_pos = Counter()
            for j in test_ids:
                for k, flag in enumerate(entries[j].get("y", [])[: args.num_labels]):
                    if flag:
                        test_pos[labels[k]] += 1
            leakage[str(seed)] = {
                "test_size": len(test_ids),
                "test_with_exact_duplicate_in_train": len(leaked),
                "test_positives": dict(test_pos),
                "leaked_positives": dict(leaked_pos),
            }

    report = {
        "graphs": len(rows),
        "status": dict(status),
        "positives_by_label": dict(positives_by_label),
        "failed_positives_by_label": dict(failed_by_label),
        "error_kinds": dict(error_kinds.most_common(15)),
        "structural_problems": dict(problems),
        "token_coverage_vs_tree_sitter": dict(token_match),
        "tree_sitter_parse_errors": ts_error,
        "provenance_vs_fetched_file": dict(provenance),
        "sizes": {k: quantiles(v) for k, v in sizes.items()},
        "construction_ms_per_snippet": quantiles(durations),
        "exact_duplicates": {
            "groups": len(dup_groups),
            "samples_in_groups": sum(len(g) for g in dup_groups),
            "groups_with_label_conflict": len(conflict_groups),
            "largest_group": max((len(g) for g in dup_groups), default=0),
        },
        "split_leakage": leakage,
        "examples": dict(examples),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))

    md = ["# AST audit", ""]
    for key in (
        "graphs", "status", "positives_by_label", "failed_positives_by_label", "error_kinds",
        "structural_problems", "token_coverage_vs_tree_sitter", "tree_sitter_parse_errors",
        "provenance_vs_fetched_file", "construction_ms_per_snippet", "exact_duplicates", "split_leakage",
    ):
        md.append(f"- **{key}**: `{json.dumps(report[key])}`")
    md.append("")
    md.append("## Sizes")
    for key, val in report["sizes"].items():
        md.append(f"- {key}: `{json.dumps(val)}`")
    args.out_md.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
