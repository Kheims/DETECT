"""Lines of each MLCQ method snippet that access another class or the own class.

Parses the source file from the repository snapshot of the sample. Writes the
own, foreign and most accessed type lines to artifacts/xai/access_lines.jsonl.

Usage:
    python scripts/xai/xai_access_lines.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llm import build_context_packages as bcp  # noqa: E402
from mlcq_graphs.xai import common  # noqa: E402

OUT = ROOT / "artifacts/xai/access_lines.jsonl"


def walk(node):
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(reversed(cur.children))


def access_lines(target, src: bytes, owner, outer, anonymous: bool) -> dict:
    mf = bcp.MethodFacts(target, src, owner, outer, anonymous)
    body = target.child_by_field_name("body")
    foreign, own = defaultdict(list), set()
    if body is None:
        return {"foreign": {}, "own": []}
    for n in walk(body):
        line = n.start_point[0] + 1
        if n.type in ("method_invocation", "field_access"):
            obj = n.child_by_field_name("object")
            if obj is None:
                name = n.child_by_field_name("name")
                if name is not None and bcp.text_of(name, src) in mf.methods:
                    own.add(line)
                continue
            if obj.type in ("this", "super"):
                own.add(line)
                continue
            kind, tname, _ = mf.receiver(obj)
            if kind == "typed" and tname and tname != mf.owner_name:
                foreign[tname].append(line)
            elif kind == "own":
                own.add(line)
        elif n.type == "identifier" and n.parent is not None and n.parent.type not in (
                "method_invocation", "field_access", "method_declaration", "formal_parameter",
                "variable_declarator", "class_declaration"):
            name = bcp.text_of(n, src)
            if name in mf.fields and name not in mf.locals:
                own.add(line)
    return {"foreign": {t: sorted(set(v)) for t, v in foreign.items()}, "own": sorted(own),
            "foreign_counts": {t: len(v) for t, v in foreign.items()}}


def main() -> None:
    pkgs = {}
    for line in (ROOT / "data/derived/context_packages/mlcq_context.jsonl").open():
        r = json.loads(line)
        if r["level"] == "method" and r.get("snippet_match") and r.get("target"):
            pkgs[r["json_index"]] = r
    by_snap = defaultdict(list)
    for j, r in pkgs.items():
        by_snap[r["snapshot"]].append(j)
    rows, failed = [], Counter()
    for snap, idxs in sorted(by_snap.items()):
        sdir = ROOT / "data/repo_snapshots" / snap
        if not sdir.is_dir():
            failed["no_snapshot"] += len(idxs)
            continue
        cache = bcp.SourceCache(sdir)
        for j in idxs:
            r = pkgs[j]
            hit = cache.get(r["file"])
            if hit is None:
                failed["no_file"] += 1
                continue
            src, root = hit
            start = int(common.samples()[j]["start_line"])
            target, _ = bcp.locate_target(root, "method", start, int(common.samples()[j]["end_line"]))
            if target is None:
                failed["no_target"] += 1
                continue
            owners, anonymous = bcp.enclosing_types(target)
            owner = owners[0] if owners else None
            acc = access_lines(target, src, owner, owners[1:], anonymous)
            n_lines = len(common.snippet_lines(j))
            to_snip = lambda ls: sorted({x - start for x in ls if 0 <= x - start < n_lines})  # noqa: E731
            foreign = {t: to_snip(v) for t, v in acc["foreign"].items()}
            envied = max(acc["foreign_counts"], key=acc["foreign_counts"].get) if acc["foreign_counts"] else None
            rows.append({"json_index": j, "n_lines": n_lines, "own_lines": to_snip(acc["own"]),
                         "foreign_lines": sorted({x for v in foreign.values() for x in v}),
                         "foreign_by_type": foreign, "foreign_counts": acc["foreign_counts"],
                         "envied_type": envied, "envied_lines": foreign.get(envied, []) if envied else []})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with_foreign = sum(1 for r in rows if r["foreign_lines"])
    print(f"[access] {len(rows)} method samples, {with_foreign} with a foreign access line; failed {dict(failed)}")


if __name__ == "__main__":
    main()
