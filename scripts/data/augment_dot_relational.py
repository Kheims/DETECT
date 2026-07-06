"""Add relational nodes and edges to the AST graphs built by DETECT.

- Links each accessed member token to a node for the owning class (own, project, external, unresolved)
- New edge types: RelAccess and RelOwner
- Level 0 uses the snippet only, level 1 also the enclosing file

Usage:
    uv run --with tree-sitter --with tree-sitter-java python scripts/data/augment_dot_relational.py \
        --level 1 --dot-dir <construction>/dot --out-dir data/relational_dot_l1
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data import extract_relational_features as rf
from scripts.data.fetch_context_files import file_key

NODE_RE = re.compile(r'^"(n\d+)" \[(.*)\]$')
TOKEN_ATTR = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|[^,\s]+)')


def snippet_events(sample, ctx_dir: Path, level: int):
    """Access events in DOT coordinates: (line 1-based, col, member, kind, type)."""
    events = []
    if level >= 1:
        path = ctx_dir / f"{file_key(sample['repo_url'], sample['commit_hash'], sample['file_path'])}.java"
        if path.exists() and path.stat().st_size > 0:
            tree = rf.PARSER.parse(path.read_bytes())
            target = rf.locate(tree.root_node, int(sample["start_line"]), int(sample["end_line"]))
            if target is not None:
                ctx = rf.FileContext(tree.root_node)
                own_decl = target if target.type in rf.TYPE_DECL_TYPES else rf.enclosing_type(target)
                own_fields, own_methods = rf.class_members(own_decl) if own_decl is not None else ({}, set())
                own_types = {rf.text(n.child_by_field_name("name")) for n in rf.walk(tree.root_node)
                             if n.type in rf.TYPE_DECL_TYPES and n.child_by_field_name("name") is not None}
                raw = []
                rf.analyse(target, ctx, own_fields, own_methods, own_types, use_file=True, events=raw)
                offset = int(sample["start_line"]) - 1
                # snippet row r (0-based) is DOT line r + 2
                return [(row - offset + 2, col, m, k, t) for row, col, m, k, t in raw], "file"
    src = sample["code_snippet"].encode("utf8")
    tree = rf.PARSER.parse(src)
    target = next((n for n in rf.walk(tree.root_node) if n.type in rf.DECL_TYPES), None)
    shift = 2
    if target is None or tree.root_node.has_error:
        tree = rf.PARSER.parse(b"class __Wrapper__ {\n" + src + b"\n}")
        decls = [n for n in rf.walk(tree.root_node) if n.type in rf.DECL_TYPES]
        target = next((n for n in decls if n.type != "class_declaration"
                       or rf.text(n.child_by_field_name("name")) != "__Wrapper__"), None)
        shift = 1
        if target is None:
            return [], "none"
    own_type = rf.enclosing_type(target)
    own_name = rf.text(own_type.child_by_field_name("name")) if own_type is not None else ""
    own_types = {own_name} if own_name and own_name != "__Wrapper__" else set()
    raw = []
    rf.analyse(target, None, {}, set(), own_types, use_file=False, events=raw)
    return [(row + shift, col, m, k, t) for row, col, m, k, t in raw], "snippet"


def parse_nodes(dot_text: str):
    nodes = {}
    for line in dot_text.splitlines():
        m = NODE_RE.match(line.strip())
        if not m:
            continue
        attrs = {k: v.strip('"') for k, v in TOKEN_ATTR.findall(m.group(2))}
        nodes[m.group(1)] = attrs
    return nodes


def augment(dot_text: str, events):
    nodes = parse_nodes(dot_text)
    by_pos = {}
    for nid, a in nodes.items():
        if a.get("node_kind") == "SyntaxToken":
            by_pos[(int(a.get("start_line", -1)), int(a.get("start_col", -1)))] = (nid, a.get("token_text", ""))
    next_index = max(int(a.get("node_index", 0)) for a in nodes.values()) + 1
    new_nodes, new_edges, targets = [], [], {}
    stats = Counter()

    def target_node(kind, tname):
        nonlocal next_index
        key = (kind, tname if kind == "project" else "")
        if key not in targets:
            node_type = {"own": "RelOwnClass", "project": "RelProjectType",
                         "external": "RelExternalType"}.get(kind, "RelUnresolved")
            # ids must start with "n" for build_pyg_dataset_from_dot.parse_dot_file
            nid = f"n{next_index}"
            new_nodes.append(f'"{nid}" [node_index={next_index} node_kind="RelNode" node_type="{node_type}" '
                             f'start_line=0 start_col=0 end_line=0 end_col=0]')
            next_index += 1
            targets[key] = nid
            if kind == "own":
                new_edges.append(f'"n0" -> "{nid}" [edge_type="RelOwner"]')
        return targets[key]

    for line, col, member, kind, tname in events:
        hit = by_pos.get((line, col))
        if hit is None or hit[1] != member:
            stats["unmatched"] += 1
            continue
        stats["matched"] += 1
        new_edges.append(f'"{hit[0]}" -> "{target_node(kind, tname)}" [edge_type="RelAccess"]')
    body = dot_text.rstrip().rstrip("}")
    return body + "\n".join(new_nodes + new_edges) + "\n}\n", stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized", default="data/MLCQCodeSmellSamples.normalized.json")
    parser.add_argument("--context-dir", default="data/context_files")
    parser.add_argument("--dot-dir", required=True)
    parser.add_argument("--manifest", required=True, help="construction manifest.jsonl of the base graphs")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--level", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()

    samples = json.loads(Path(args.normalized).read_text())
    out_dir = Path(args.out_dir)
    (out_dir / "dot").mkdir(parents=True, exist_ok=True)
    totals, sources = Counter(), Counter()
    out_manifest = []
    for line in Path(args.manifest).read_text().splitlines():
        row = json.loads(line)
        if not row.get("output_dot"):
            continue
        dot_path = Path(args.dot_dir) / Path(row["output_dot"]).name
        if not dot_path.exists():
            continue
        events, source = snippet_events(samples[int(row["json_index"])], Path(args.context_dir), args.level)
        sources[source] += 1
        new_text, stats = augment(dot_path.read_text(), events)
        totals.update(stats)
        new_path = out_dir / "dot" / dot_path.name
        new_path.write_text(new_text)
        row["output_dot"] = str(new_path.resolve())
        out_manifest.append(json.dumps(row))
    (out_dir / "manifest.jsonl").write_text("\n".join(out_manifest) + "\n")
    report = {"graphs": len(out_manifest), "event_sources": dict(sources), **dict(totals)}
    (out_dir / "augment_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
