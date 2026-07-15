"""Size and shape features of every sample of a normalized JSON, and review metadata for MLCQ.

Writes a size CSV and, if enabled, review metadata and entry metrics CSVs.

Usage:
    python scripts/data/extract_size_features.py --config config/experiments/mlcq_analysis.yml
    python scripts/data/extract_size_features.py --config config/experiments/dacos_size.yml
    python scripts/data/extract_size_features.py --config config/experiments/dacos_size.yml \\
        --inputs.normalized_json=data/derived/metamorphic/dacos/<transform>.normalized.json \\
        --size_features.out_csv=data/derived/metamorphic/dacos/<transform>.size_features.csv \\
        --entry_metrics.enabled=false
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.constants import LABEL_ORDER  # noqa: E402

# tree-sitter-java node types used by the feature definitions
COMMENT_TYPES = {"line_comment", "block_comment", "comment"}
ATOMIC_TYPES = {"string_literal", "character_literal", "text_block"}
METHOD_TYPES = {"method_declaration", "constructor_declaration", "compact_constructor_declaration"}
CLASS_TYPES = {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration",
               "annotation_type_declaration"}
FIELD_TYPES = {"field_declaration", "constant_declaration"}
BLOCK_TYPES = {"block", "constructor_body", "switch_block"}
IDENTIFIER_TYPES = {"identifier", "type_identifier"}
PARAMETER_TYPES = {"formal_parameter", "spread_parameter"}
STATEMENT_TYPES = {
    "expression_statement", "local_variable_declaration", "if_statement", "while_statement",
    "do_statement", "for_statement", "enhanced_for_statement", "return_statement", "throw_statement",
    "try_statement", "try_with_resources_statement", "switch_expression", "break_statement",
    "continue_statement", "synchronized_statement", "labeled_statement", "assert_statement",
    "yield_statement", "explicit_constructor_invocation", "local_class_declaration",
}

SIZE_COLUMNS = [
    "loc_physical", "loc_nonblank", "chars", "tokens", "ast_nodes", "ast_depth", "n_methods",
    "n_fields", "n_statements", "n_params_first_method", "max_block_nesting",
    "n_distinct_identifiers", "n_classes", "n_comments", "has_parse_error",
]

_PARSER: Any = None


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def _parser() -> Any:
    global _PARSER
    if _PARSER is None:
        import tree_sitter as ts
        import tree_sitter_java as tsj

        _PARSER = ts.Parser(ts.Language(tsj.language()))
    return _PARSER


def size_features(task: tuple[int, str, str, str]) -> dict[str, int]:
    """Features of one snippet, task is (json_index, snippet, wrapper prefix, suffix)."""
    json_index, snippet, prefix, suffix = task
    body = snippet.rstrip("\n")
    prefix_bytes = prefix.encode("utf-8")
    body_bytes = body.encode("utf-8")
    lo, hi = len(prefix_bytes), len(prefix_bytes) + len(body_bytes)
    tree = _parser().parse(prefix_bytes + body_bytes + suffix.encode("utf-8"))

    counts = defaultdict(int)
    identifiers: set[str] = set()
    first_method_params: int | None = None
    max_depth = max_nesting = 0
    # pre-order walk, depth and nesting count only the nodes inside the snippet
    stack: list[tuple[Any, int, int]] = [(tree.root_node, 0, 0)]
    while stack:
        node, depth, nesting = stack.pop()
        inside = node.start_byte >= lo and node.end_byte <= hi
        kind = node.type
        if inside and kind in COMMENT_TYPES:
            counts["n_comments"] += 1
            continue
        if node.is_missing:
            continue
        if inside:
            depth += 1
            counts["ast_nodes"] += 1
            max_depth = max(max_depth, depth)
            if kind in BLOCK_TYPES:
                nesting += 1
                max_nesting = max(max_nesting, nesting)
            if kind in METHOD_TYPES:
                counts["n_methods"] += 1
                if first_method_params is None:
                    params = node.child_by_field_name("parameters")
                    first_method_params = 0 if params is None else sum(
                        1 for c in params.children if c.type in PARAMETER_TYPES)
            elif kind in CLASS_TYPES:
                counts["n_classes"] += 1
            elif kind in FIELD_TYPES:
                counts["n_fields"] += sum(1 for c in node.children if c.type == "variable_declarator")
            if kind in STATEMENT_TYPES:
                counts["n_statements"] += 1
            if kind in IDENTIFIER_TYPES:
                identifiers.add(node.text.decode("utf-8", errors="replace"))
        is_token = node.child_count == 0 or kind in ATOMIC_TYPES
        if is_token:
            if inside:
                counts["tokens"] += 1
            continue
        for child in reversed(node.children):
            stack.append((child, depth, nesting))

    lines = body.split("\n") if body else []
    return {
        "sample_idx": json_index,
        "loc_physical": len(lines),
        "loc_nonblank": sum(1 for line in lines if line.strip()),
        "chars": len(body),
        "tokens": counts["tokens"],
        "ast_nodes": counts["ast_nodes"],
        "ast_depth": max_depth,
        "n_methods": counts["n_methods"],
        "n_fields": counts["n_fields"],
        "n_statements": counts["n_statements"],
        "n_params_first_method": first_method_params or 0,
        "max_block_nesting": max_nesting,
        "n_distinct_identifiers": len(identifiers),
        "n_classes": counts["n_classes"],
        "n_comments": counts["n_comments"],
        "has_parse_error": int(tree.root_node.has_error),
    }


def aggregate(severities: list[str], order: list[str], method: str) -> str:
    if method != "upper_median":
        raise ValueError(f"unsupported aggregation: {method}")
    scores = sorted(order.index(s) for s in severities)
    return order[scores[len(scores) // 2]]


def repo_name(url: str) -> str:
    """git@github.com:owner/name.git or https://github.com/owner/name -> owner/name."""
    tail = url.rstrip("/").removesuffix(".git")
    tail = tail.split(":", 1)[1] if tail.startswith("git@") else tail.split("://", 1)[-1].split("/", 1)[-1]
    return "/".join(tail.split("/")[-2:])


def tmp_path(path: Path) -> Path:
    """Per-process temporary name, so that concurrent writers never share one."""
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path(path)
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def sample_metadata(cfg: dict[str, Any], entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], dict]:
    inputs, lcfg = cfg["inputs"], cfg["labels"]
    order = list(lcfg["severity_order"])
    positive = set(lcfg["positive"])
    smells, shorts = list(lcfg["smell_names"]), list(lcfg["short_names"])
    if len(smells) != len(LABEL_ORDER):
        raise SystemExit("labels.smell_names must follow LABEL_ORDER")
    level_of = {s: level for level, names in lcfg["levels"].items() for s in names}

    raw = json.loads(resolve(inputs["raw_json"]).read_text())
    votes: dict[tuple, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in raw:
        key = (row["repo_url"], row["commit_hash"], row["file_path"], int(row["start_line"]), int(row["end_line"]))
        votes[key][row["smell"]].append(row["severity"])

    folds_path = str(inputs["folds"])
    folds = json.loads(resolve(folds_path).read_text())
    parsable = {int(i) for i in folds["meta"]["json_index"]}
    fold_columns: list[str] = []
    test_fold: dict[str, dict[int, int]] = {}
    for split in folds["splits"]:
        col = f"test_fold_{split['strategy']}_s{split['seed']}"
        if col not in test_fold:
            fold_columns.append(col)
            test_fold[col] = {}
        for i in split["test"]:
            test_fold[col][int(i)] = int(split["fold"])

    mismatches: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for j, entry in enumerate(entries):
        key = (entry["repo_url"], entry["commit_hash"], entry["file_path"], int(entry["start_line"]), int(entry["end_line"]))
        sample_votes = votes.get(key, {})
        reviewed = [s for s in smells if sample_votes.get(s)]
        levels = {level_of[s] for s in reviewed}
        level = levels.pop() if len(levels) == 1 else ("mixed" if levels else "none")
        y = entry["y"]
        if isinstance(y, str):
            y = json.loads(y.lower())
        row: dict[str, Any] = {
            "json_index": j,
            "repo": repo_name(entry["repo_url"]),
            "repo_url": entry["repo_url"],
            "commit_hash": entry["commit_hash"],
            "file_path": entry["file_path"],
            "start_line": entry["start_line"],
            "end_line": entry["end_line"],
            "level": level,
            "is_method_level": int(level == "method"),
            "parse_ok": int(j in parsable),
            "folds_file": folds_path if j in parsable else "",
        }
        agree = True
        for s, short, label in zip(smells, shorts, y):
            v = sample_votes.get(s, [])
            stored = entry["labels"][s]
            agg = aggregate(v, order, lcfg["aggregation"]) if v else ""
            raw_present = bool(v) and agg in positive
            row[f"n_reviews_{short}"] = len(v)
            row[f"vote_count_stored_{short}"] = int(stored["vote_count"])
            for sev in order:
                row[f"votes_{short}_{sev}"] = sum(1 for x in v if x == sev)
            row[f"severity_{short}"] = stored["severity"] if stored["vote_count"] else ""
            row[f"severity_raw_{short}"] = agg
            row[f"y_{short}"] = int(bool(label))
            row[f"y_raw_{short}"] = int(raw_present)
            present = [x in positive for x in v]
            # empty for a single review
            row[f"unanimous_{short}"] = int(len(set(present)) == 1) if len(v) >= 2 else ""
            row[f"unanimous_severity_{short}"] = int(len(set(v)) == 1) if len(v) >= 2 else ""
            diff = {"vote_count": stored["vote_count"] != len(v),
                    "severity": bool(v) and stored["severity"] != agg,
                    "label": bool(label) != raw_present}
            if any(diff.values()):
                agree = agree and not diff["label"]
                mismatches.append({"json_index": j, "smell": s, "raw_votes": v, "stored": stored,
                                   "y": bool(label), "y_raw": raw_present,
                                   "differs": [k for k, d in diff.items() if d]})
        row["labels_match_raw"] = int(agree)
        for col in fold_columns:
            row[col] = test_fold[col].get(j, "")
        rows.append(row)

    columns = list(rows[0].keys())
    by_kind: dict[str, int] = defaultdict(int)
    label_diffs: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for m in mismatches:
        for kind in m["differs"]:
            by_kind[kind] += 1
        if "label" in m["differs"]:
            label_diffs[m["smell"]]["stored_positive" if m["y"] else "stored_negative"] += 1
    indices = sorted({m["json_index"] for m in mismatches})
    check = {
        "rule": {"aggregation": lcfg["aggregation"], "positive": sorted(positive)},
        "samples": len(entries), "raw_rows": len(raw), "raw_samples": len(votes),
        "mismatched_pairs": len(mismatches), "mismatched_samples": len(indices),
        "mismatch_index_range": [indices[0], indices[-1]] if indices else None,
        "by_kind": dict(by_kind),
        "label_differences": {k: dict(v) for k, v in label_diffs.items()},
        "vote_count_excess": {str(k): v for k, v in sorted(_excess(mismatches).items())},
        "positives_stored": {short: sum(r[f"y_{short}"] for r in rows) for short in shorts},
        "positives_raw_rule": {short: sum(r[f"y_raw_{short}"] for r in rows) for short in shorts},
        "mismatches": mismatches,
    }
    return rows, columns, check


def _excess(mismatches: list[dict[str, Any]]) -> dict[int, int]:
    """Histogram of stored vote_count minus the number of raw reviews."""
    out: dict[int, int] = defaultdict(int)
    for m in mismatches:
        out[int(m["stored"]["vote_count"]) - len(m["raw_votes"])] += 1
    return out


def entry_metrics(ecfg: dict[str, Any], entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Rows of sample_idx and the configured numeric columns of entry[field]."""
    field, columns = str(ecfg["field"]), [str(c) for c in ecfg["columns"]]
    rows = []
    for j, entry in enumerate(entries):
        values = entry.get(field)
        if not isinstance(values, dict) or any(values.get(c) is None for c in columns):
            raise SystemExit(f"entry {j} has no {field} values for {columns}")
        rows.append({"sample_idx": j, **{c: float(values[c]) for c in columns}})
    return rows, ["sample_idx", *columns]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config/experiments/mlcq_analysis.yml"))
    args, extra = parser.parse_known_args()
    cfg = load_config(resolve(args.config), parse_cli_overrides(extra))
    scfg, mcfg = cfg["size_features"], cfg["sample_meta"]

    entries = json.loads(resolve(cfg["inputs"]["normalized_json"]).read_text())
    print(f"[size] {len(entries)} entries from {cfg['inputs']['normalized_json']}")
    # imported here to keep torch out of the feature workers
    from mlcq_graphs.evaluation.cv import check_variant_entries, is_variant

    base_json = cfg["inputs"].get("base_json")
    same = base_json is not None and resolve(base_json) == resolve(cfg["inputs"]["normalized_json"])
    if is_variant(entries) and (base_json is None or same):
        raise SystemExit(f"{cfg['inputs']['normalized_json']} is a metamorphic variant (its entries carry a "
                         "metamorphic field): set inputs.base_json to its base normalized JSON")
    if base_json is not None and not same:
        n_changed = check_variant_entries(json.loads(resolve(base_json).read_text()), entries,
                                          str(cfg["inputs"]["normalized_json"]))
        print(f"[size] variant of {base_json}: {n_changed} of {len(entries)} snippets differ")

    if bool(mcfg.get("enabled", True)):
        meta_rows, meta_columns, check = sample_metadata(cfg, entries)
        check_path = resolve(mcfg["check_out"])
        check_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = tmp_path(check_path)
        tmp.write_text(json.dumps(check, indent=1))
        tmp.replace(check_path)
        print(f"[meta] label rule check: {check['mismatched_pairs']} (sample, smell) pairs differ over "
              f"{check['mismatched_samples']} samples, json_index range {check['mismatch_index_range']}, "
              f"by kind {check['by_kind']}")
        print(f"[meta] label differences {check['label_differences']}; positives stored {check['positives_stored']}"
              f" vs raw rule {check['positives_raw_rule']}")
        if check["mismatched_pairs"] and cfg["labels"].get("fail_on_mismatch", False):
            raise SystemExit(f"the label rule does not reproduce {cfg['inputs']['normalized_json']}, see {check_path}")
        write_csv(resolve(mcfg["out_csv"]), meta_rows, meta_columns)
        print(f"[meta] wrote {resolve(mcfg['out_csv'])} ({len(meta_rows)} rows, {len(meta_columns)} columns)")

    ecfg = cfg.get("entry_metrics") or {}
    if ecfg and bool(ecfg.get("enabled", True)):
        rows, columns = entry_metrics(ecfg, entries)
        write_csv(resolve(ecfg["out_csv"]), rows, columns)
        print(f"[metrics] wrote {resolve(ecfg['out_csv'])} ({len(rows)} rows, columns {columns})")

    tasks = [(j, str(e.get("code_snippet", "")), str(scfg["wrapper_prefix"]), str(scfg["wrapper_suffix"]))
             for j, e in enumerate(entries)]
    workers = int(scfg.get("workers", 1))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            feats = list(pool.map(size_features, tasks, chunksize=64))
    else:
        feats = [size_features(t) for t in tasks]
    write_csv(resolve(scfg["out_csv"]), feats, ["sample_idx", *SIZE_COLUMNS])
    errors = sum(f["has_parse_error"] for f in feats)
    print(f"[size] wrote {resolve(scfg['out_csv'])} ({len(feats)} rows, {errors} with tree-sitter errors)")


if __name__ == "__main__":
    main()
