"""Extract own versus foreign member accesses for every MLCQ snippet.

- Level 0 types receivers from the snippet only, level 1 also uses the enclosing file
- Writes one CSV row per snippet, keyed by sample_idx

Usage:
    uv run --with tree-sitter --with tree-sitter-java python \
        scripts/data/extract_relational_features.py \
        --normalized data/MLCQCodeSmellSamples.normalized.json \
        --context-dir data/context_files \
        --out artifacts/relational_features.csv
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import tree_sitter as ts
import tree_sitter_java as tsj

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.fetch_context_files import file_key

LANG = ts.Language(tsj.language())
PARSER = ts.Parser(LANG)

DECL_TYPES = {
    "method_declaration", "constructor_declaration", "class_declaration",
    "interface_declaration", "enum_declaration", "record_declaration",
}
TYPE_DECL_TYPES = {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"}
CALLABLE_TYPES = {"method_declaration", "constructor_declaration", "lambda_expression"}

# java.lang and very common JDK types that are never a project class.
JDK_SIMPLE = {
    "String", "Object", "Integer", "Long", "Short", "Byte", "Double", "Float", "Boolean",
    "Character", "Number", "Math", "System", "StringBuilder", "StringBuffer", "Thread",
    "Class", "Enum", "Exception", "RuntimeException", "Throwable", "Error", "Iterable",
    "List", "ArrayList", "LinkedList", "Map", "HashMap", "TreeMap", "LinkedHashMap",
    "Set", "HashSet", "TreeSet", "LinkedHashSet", "Collection", "Collections", "Arrays",
    "Optional", "Iterator", "Objects", "Stream", "Collectors", "Date", "File", "Path",
    "Paths", "Files", "UUID", "BigDecimal", "BigInteger", "Pattern", "Matcher",
    "Queue", "Deque", "ArrayDeque", "Locale", "TimeUnit", "Duration", "Instant",
}
EXTERNAL_PREFIXES = (
    "java.", "javax.", "jakarta.", "sun.", "com.sun.", "org.slf4j", "org.apache.commons",
    "org.apache.log4j", "org.apache.logging", "com.google.common", "com.google.gson",
    "org.junit", "junit.", "org.mockito", "org.hamcrest", "android.", "androidx.",
    "kotlin.", "scala.", "org.springframework", "com.fasterxml", "io.netty", "lombok.",
)

FEATURES = [
    "own_field", "own_call", "own_total", "foreign_project", "foreign_external",
    "foreign_unresolved", "distinct_project_types", "max_single_project_type",
    "envy_ratio", "foreign_ratio", "envy_rule",
]


def text(node) -> str:
    return node.text.decode("utf8", errors="replace")


def strip_type(type_text: str) -> str:
    """'Map<String, Foo>[]' -> 'Map', 'a.b.Foo' -> 'Foo'."""
    base = type_text.split("<", 1)[0].replace("[]", "").replace("...", "").strip()
    return base.rsplit(".", 1)[-1]


def walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def declarators(decl_node):
    """(name, type) pairs of a field, local variable or parameter declaration."""
    type_node = decl_node.child_by_field_name("type")
    type_name = strip_type(text(type_node)) if type_node is not None else None
    if decl_node.type in ("formal_parameter", "spread_parameter", "catch_formal_parameter"):
        name = decl_node.child_by_field_name("name")
        if name is None:
            ids = [c for c in decl_node.children if c.type == "identifier"]
            name = ids[-1] if ids else None
        return [(text(name), type_name)] if name is not None else []
    out = []
    for c in decl_node.children:
        if c.type == "variable_declarator":
            name = c.child_by_field_name("name")
            if name is not None:
                out.append((text(name), type_name))
    return out


class FileContext:
    """Package, imports and per-class members of one compilation unit."""

    def __init__(self, root):
        self.package = ""
        self.imports = {}
        for c in root.children:
            if c.type == "package_declaration":
                self.package = text(c).replace("package", "").replace(";", "").strip()
            elif c.type == "import_declaration":
                path = text(c).replace("import", "").replace("static", "").replace(";", "").strip()
                if not path.endswith("*"):
                    self.imports[path.rsplit(".", 1)[-1]] = path

    def classify(self, type_name, own_types):
        """'own', 'project' or 'external' for a simple type name."""
        if not type_name:
            return "unresolved"
        if type_name in own_types:
            return "own"
        fq = self.imports.get(type_name)
        if fq is not None:
            if fq.startswith(EXTERNAL_PREFIXES):
                return "external"
            own_root = ".".join(self.package.split(".")[:2])
            return "project" if own_root and fq.startswith(own_root) else "external"
        if type_name in JDK_SIMPLE or not type_name[:1].isupper():
            return "external"
        # not imported and not java.lang: same package, hence a project class
        return "project"


def class_members(type_decl):
    """Fields (name -> type) and method names declared directly in a type."""
    fields, methods = {}, set()
    body = type_decl.child_by_field_name("body")
    if body is None:
        return fields, methods
    for c in body.children:
        if c.type in ("field_declaration", "constant_declaration"):
            for name, tname in declarators(c):
                fields[name] = tname
        elif c.type == "method_declaration":
            name = c.child_by_field_name("name")
            if name is not None:
                methods.add(text(name))
        elif c.type == "enum_body_declarations":
            for cc in c.children:
                if cc.type == "field_declaration":
                    for name, tname in declarators(cc):
                        fields[name] = tname
    return fields, methods


def enclosing_type(node):
    n = node
    while n is not None:
        if n.type in TYPE_DECL_TYPES:
            return n
        n = n.parent
    return None


def local_scope(callable_node):
    """Parameters and local variables declared anywhere in a method body."""
    scope = {}
    for n in walk(callable_node):
        if n.type in ("formal_parameter", "spread_parameter", "catch_formal_parameter",
                      "local_variable_declaration"):
            for name, tname in declarators(n):
                scope[name] = tname
        elif n.type == "enhanced_for_statement":
            t = n.child_by_field_name("type")
            name = n.child_by_field_name("name")
            if name is not None:
                scope[text(name)] = strip_type(text(t)) if t is not None else None
        elif n.type == "lambda_expression":
            params = n.child_by_field_name("parameters")
            if params is not None and params.type == "identifier":
                scope[text(params)] = None
    return scope


def root_receiver(expr):
    """Follow `a.b.c()` down to `a`, `this`, `super` or an opaque expression."""
    n = expr
    while n is not None and n.type in ("field_access", "method_invocation", "parenthesized_expression"):
        if n.type == "parenthesized_expression":
            inner = [c for c in n.children if c.is_named]
            n = inner[0] if inner else None
            continue
        obj = n.child_by_field_name("object")
        if obj is None:
            return n  # unqualified call such as getFoo().bar()
        n = obj
    return n


def analyse(target, ctx, own_fields, own_methods, own_types, use_file, events=None):
    """Count accesses inside `target` and optionally append (row, col, member, kind, type) to `events`."""

    def record(member_node, kind, tname=None):
        if events is not None and member_node is not None:
            events.append((member_node.start_point[0], member_node.start_point[1],
                           text(member_node), kind, tname or ""))

    counts = Counter()
    per_type = Counter()
    callables = [n for n in walk(target) if n.type in ("method_declaration", "constructor_declaration")]
    if not callables:
        callables = [target]
    seen = set()
    for callable_node in callables:
        scope = local_scope(callable_node)
        for n in walk(callable_node):
            if n.id in seen:
                continue
            if n.type not in ("field_access", "method_invocation", "identifier"):
                continue
            seen.add(n.id)
            if n.type == "identifier":
                if not use_file:
                    continue
                parent = n.parent
                if parent is None or parent.type in ("field_access", "method_invocation",
                                                     "variable_declarator", "formal_parameter",
                                                     "method_declaration", "class_declaration"):
                    continue
                name = text(n)
                if name in own_fields and name not in scope:
                    counts["own_field"] += 1
                    record(n, "own")
                continue
            # only count the outermost access of a chain once
            parent = n.parent
            if parent is not None and parent.type in ("field_access", "method_invocation") \
                    and parent.child_by_field_name("object") == n:
                continue
            is_call = n.type == "method_invocation"
            member = n.child_by_field_name("name" if is_call else "field")
            obj = n.child_by_field_name("object")
            if obj is None:
                if is_call:
                    counts["own_call"] += 1
                    record(member, "own")
                continue
            root = root_receiver(obj)
            if root is None:
                counts["foreign_unresolved"] += 1
                record(member, "unresolved")
                continue
            if root.type in ("this", "super"):
                counts["own_call" if is_call else "own_field"] += 1
                record(member, "own")
                continue
            if root.type == "method_invocation" and root.child_by_field_name("object") is None:
                counts["foreign_unresolved"] += 1
                record(member, "unresolved")
                continue
            if root.type != "identifier":
                counts["foreign_unresolved"] += 1
                record(member, "unresolved")
                continue
            name = text(root)
            if name in scope:
                tname = scope[name]
            elif use_file and name in own_fields:
                tname = own_fields[name]
            elif name[:1].isupper():
                tname = name  # static reference to a class
            else:
                tname = None
            kind = ctx.classify(tname, own_types) if ctx is not None else (
                "unresolved" if not tname else "own" if tname in own_types
                else "external" if tname in JDK_SIMPLE or not tname[:1].isupper() else "project")
            if kind == "own":
                counts["own_call" if is_call else "own_field"] += 1
            elif kind == "project":
                counts["foreign_project"] += 1
                per_type[tname] += 1
            elif kind == "external":
                counts["foreign_external"] += 1
            else:
                counts["foreign_unresolved"] += 1
            record(member, kind, tname)
    own_total = counts["own_field"] + counts["own_call"]
    max_single = max(per_type.values()) if per_type else 0
    return {
        "own_field": counts["own_field"],
        "own_call": counts["own_call"],
        "own_total": own_total,
        "foreign_project": counts["foreign_project"],
        "foreign_external": counts["foreign_external"],
        "foreign_unresolved": counts["foreign_unresolved"],
        "distinct_project_types": len(per_type),
        "max_single_project_type": max_single,
        "envy_ratio": max_single / (own_total + 1),
        "foreign_ratio": counts["foreign_project"] / (own_total + counts["foreign_project"] + 1),
        "envy_rule": int(max_single > own_total),
    }


def locate(root, start_line, end_line):
    """Smallest declaration that starts within 3 lines of the snippet start and covers it."""
    best = None
    for n in walk(root):
        if n.type not in DECL_TYPES:
            continue
        s, e = n.start_point[0] + 1, n.end_point[0] + 1
        if abs(s - start_line) <= 3 and e >= end_line - 1:
            if best is None or (e - s) < (best.end_point[0] - best.start_point[0]):
                best = n
    return best


def snippet_level0(code):
    """Level 0: parse the snippet alone, wrapped in a synthetic class if needed."""
    src = code.encode("utf8")
    tree = PARSER.parse(src)
    target = next((n for n in walk(tree.root_node) if n.type in DECL_TYPES), None)
    if target is None or tree.root_node.has_error:
        wrapped = b"class __Wrapper__ {\n" + src + b"\n}"
        tree = PARSER.parse(wrapped)
        decls = [n for n in walk(tree.root_node) if n.type in DECL_TYPES]
        target = next((n for n in decls if n.type != "class_declaration"
                       or text(n.child_by_field_name("name")) != "__Wrapper__"), None)
        if target is None:
            return None, False
    own_type = enclosing_type(target)
    own_name = text(own_type.child_by_field_name("name")) if own_type is not None else ""
    own_types = {own_name} if own_name and own_name != "__Wrapper__" else set()
    is_method = target.type in ("method_declaration", "constructor_declaration")
    return analyse(target, None, {}, set(), own_types, use_file=False), is_method


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized", default="data/MLCQCodeSmellSamples.normalized.json")
    parser.add_argument("--context-dir", default="data/context_files")
    parser.add_argument("--out", default="artifacts/relational_features.csv")
    args = parser.parse_args()

    samples = json.loads(Path(args.normalized).read_text())
    ctx_dir = Path(args.context_dir)
    rows, stats = [], Counter()
    for idx, s in enumerate(samples):
        row = {"sample_idx": idx}
        lvl0, is_method = snippet_level0(s["code_snippet"])
        stats["level0_ok" if lvl0 else "level0_fail"] += 1
        for k in FEATURES:
            row[f"l0_{k}"] = (lvl0 or {}).get(k, 0)

        lvl1, has_ctx = None, 0
        path = ctx_dir / f"{file_key(s['repo_url'], s['commit_hash'], s['file_path'])}.java"
        if path.exists() and path.stat().st_size > 0:
            src = path.read_bytes()
            tree = PARSER.parse(src)
            target = locate(tree.root_node, int(s["start_line"]), int(s["end_line"]))
            if target is not None:
                ctx = FileContext(tree.root_node)
                own_decl = target if target.type in TYPE_DECL_TYPES else enclosing_type(target)
                own_fields, own_methods = class_members(own_decl) if own_decl is not None else ({}, set())
                own_types = {text(n.child_by_field_name("name")) for n in walk(tree.root_node)
                             if n.type in TYPE_DECL_TYPES and n.child_by_field_name("name") is not None}
                lvl1 = analyse(target, ctx, own_fields, own_methods, own_types, use_file=True)
                is_method = target.type in ("method_declaration", "constructor_declaration")
                has_ctx = 1
        stats["level1_ok" if has_ctx else "level1_missing"] += 1
        for k in FEATURES:
            row[f"l1_{k}"] = (lvl1 if lvl1 is not None else (lvl0 or {})).get(k, 0)
        row["has_context"] = has_ctx
        row["is_method"] = int(bool(is_method))
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
