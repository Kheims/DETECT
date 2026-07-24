"""Repository context packages of the MLCQ samples for the LLM context study.

Each sample is located in its repository snapshot and parsed with
tree-sitter-java to build four context parts: enclosing type skeleton, static
facts, referenced type skeletons and usage lines. Writes <packages.jsonl>
(one object per json_index) and <packages.stats>.

Usage:
    python scripts/llm/build_context_packages.py --config config/experiments/context_study.yml
    python scripts/llm/build_context_packages.py --config ... --limit 40
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402

TYPE_DECLS = {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration",
              "annotation_type_declaration"}
METHOD_DECLS = {"method_declaration", "constructor_declaration", "compact_constructor_declaration"}
SIGNATURE_DECLS = METHOD_DECLS | {"annotation_type_element_declaration"}
FIELD_DECLS = {"field_declaration", "constant_declaration"}
COMMENT_TYPES = {"line_comment", "block_comment", "comment"}
KIND_WORD = {"class_declaration": "class", "interface_declaration": "interface", "enum_declaration": "enum",
             "record_declaration": "record", "annotation_type_declaration": "@interface"}
LEVEL_KINDS = {"method": METHOD_DECLS, "class": TYPE_DECLS}
WORD_RE = re.compile(r"\w+")
IDENTIFIER_RE = re.compile(r"^[A-Za-z_$][\w$]*$")
# capitalised name with a lower-case letter (LOG or MAX are constants)
TYPE_NAME_RE = re.compile(r"^[A-Z][\w$]*[a-z][\w$]*$")
GETTER_RE = re.compile(r"^(get|is|has)[A-Z0-9_]")
SETTER_RE = re.compile(r"^set[A-Z0-9_]")

_PARSER: Any = None


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def parser() -> Any:
    global _PARSER
    if _PARSER is None:
        import tree_sitter as ts
        import tree_sitter_java as tsj

        _PARSER = ts.Parser(ts.Language(tsj.language()))
    return _PARSER


def parse(src: bytes) -> Any:
    return parser().parse(src)


# ---------------------------------------------------------------- text helpers


def text_of(node: Any, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def collapse(text: str) -> str:
    return " ".join(text.split())


def cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(limit - 3, 1)].rstrip() + "..."


def strip_ws(text: str) -> str:
    return "".join(text.split())


def same(a: Any | None, b: Any | None) -> bool:
    """Node identity by position and type (portable across tree-sitter versions)."""
    if a is None or b is None:
        return a is b
    return (a.start_byte, a.end_byte, a.type) == (b.start_byte, b.end_byte, b.type)


def first_line(node: Any) -> int:
    return node.start_point[0] + 1


def last_line(node: Any) -> int:
    return node.end_point[0] + 1


def snapshot_name(repo: str, commit: str) -> str:
    return f"{repo.replace('/', '__')}@{commit}"


def repo_name(url: str) -> str:
    """git@github.com:owner/name.git or https://github.com/owner/name -> owner/name."""
    tail = url.rstrip("/").removesuffix(".git")
    tail = tail.split(":", 1)[1] if tail.startswith("git@") else tail.split("://", 1)[-1].split("/", 1)[-1]
    return "/".join(tail.split("/")[-2:])


# ---------------------------------------------------------------- declarations


def walk(node: Any):
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(reversed(cur.children))


def type_text(node: Any | None, src: bytes) -> str | None:
    """Declared type without generics and annotations (List<Order> -> List,
    Order[] -> Order[], var -> None)."""
    if node is None:
        return None
    kind = node.type
    if kind == "generic_type":
        for child in node.named_children:
            if child.type in ("type_identifier", "scoped_type_identifier"):
                return type_text(child, src)
    if kind == "array_type":
        inner = type_text(node.child_by_field_name("element"), src)
        return None if inner is None else inner + "[]"
    if kind == "annotated_type":
        for child in node.named_children:
            if child.type not in ("annotation", "marker_annotation"):
                return type_text(child, src)
    text = text_of(node, src)
    while "<" in text:
        new = re.sub(r"<[^<>]*>", "", text)
        if new == text:
            break
        text = new
    text = collapse(text).replace(" ", "")
    return None if text == "var" else text


def package_of(root: Any, src: bytes) -> str:
    for child in root.named_children:
        if child.type == "package_declaration":
            for sub in child.named_children:
                if sub.type in ("scoped_identifier", "identifier"):
                    return collapse(text_of(sub, src)).replace(" ", "")
    return ""


def imports_of(root: Any, src: bytes) -> tuple[dict[str, str], list[str], int]:
    """Single-type imports {simple name: qualified name}, wildcard packages
    (non-static) and the number of import declarations."""
    single: dict[str, str] = {}
    wildcard: list[str] = []
    count = 0
    for child in root.named_children:
        if child.type != "import_declaration":
            continue
        count += 1
        is_static = any(c.type == "static" for c in child.children)
        name_node = next((c for c in child.named_children if c.type in ("scoped_identifier", "identifier")), None)
        if name_node is None:
            continue
        name = collapse(text_of(name_node, src)).replace(" ", "")
        if any(c.type == "asterisk" for c in child.children):
            if not is_static:
                wildcard.append(name)
        elif not is_static:
            single[name.rsplit(".", 1)[-1]] = name
    return single, wildcard, count


def top_level_types(root: Any, src: bytes) -> list[tuple[str, Any]]:
    out = []
    for child in root.named_children:
        if child.type in TYPE_DECLS:
            name = child.child_by_field_name("name")
            if name is not None:
                out.append((text_of(name, src), child))
    return out


def declared_name(node: Any, src: bytes) -> str:
    name = node.child_by_field_name("name")
    return text_of(name, src) if name is not None else ""


def body_members(body: Any | None) -> list[Any]:
    """Direct members of a type body, enum body declarations flattened."""
    if body is None:
        return []
    out = []
    for child in body.named_children:
        if child.type == "enum_body_declarations":
            out.extend(child.named_children)
        else:
            out.append(child)
    return out


def all_type_nodes(root: Any, src: bytes) -> list[tuple[str, Any]]:
    """(name, node) of every named type declared in a file, at any depth."""
    return [(declared_name(n, src), n) for n in walk(root) if n.type in TYPE_DECLS]


def find_type_node(root: Any, src: bytes, dotted: str) -> Any | None:
    """Type declaration of a file by simple or Outer.Inner name (top-level first)."""
    parts = dotted.split(".")
    candidates = [n for name, n in top_level_types(root, src) if name == parts[0]]
    if not candidates:
        candidates = [n for name, n in all_type_nodes(root, src) if name == parts[0]]
    if not candidates:
        return None
    node = candidates[0]
    for part in parts[1:]:
        nxt = None
        for member in body_members(node.child_by_field_name("body")):
            if member.type in TYPE_DECLS and declared_name(member, src) == part:
                nxt = member
                break
        if nxt is None:
            return None
        node = nxt
    return node


def enclosing_types(node: Any) -> tuple[list[Any], bool]:
    """Named type declarations around a node, innermost first, and whether an
    anonymous class body lies between the node and the innermost named type."""
    out: list[Any] = []
    anonymous = False
    cur = node.parent
    while cur is not None:
        if cur.type in TYPE_DECLS:
            out.append(cur)
        elif cur.type == "class_body" and cur.parent is not None and cur.parent.type == "object_creation_expression":
            if not out:
                anonymous = True
        cur = cur.parent
    return out, anonymous


# ---------------------------------------------------------------- skeletons


def header_text(node: Any, src: bytes, limit: int) -> str:
    """Declaration text up to its body (or its semicolon), comments removed,
    on one line."""
    body = node.child_by_field_name("body")
    end = body.start_byte if body is not None else node.end_byte
    comments = []
    stack = [c for c in node.children if c.start_byte < end]
    while stack:
        cur = stack.pop()
        if cur.type in COMMENT_TYPES:
            comments.append((cur.start_byte, cur.end_byte))
        else:
            stack.extend(cur.children)
    pieces, pos = [], node.start_byte
    for lo, hi in sorted(comments):
        pieces.append(src[pos:lo])
        pos = max(pos, hi)
    pieces.append(src[pos:end])
    text = collapse(b" ".join(pieces).decode("utf-8", errors="replace")).rstrip(";").rstrip()
    return cut(text, limit)


def field_line(node: Any, src: bytes, opts: dict[str, Any]) -> str:
    type_limit = int(opts["field_type_chars"])
    init_limit = int(opts["field_initializer_chars"])
    modifiers, type_part, declarators = "", "", []
    for child in node.children:
        if child.type == "modifiers":
            modifiers = collapse(text_of(child, src))
        elif child.type == "variable_declarator":
            name = child.child_by_field_name("name")
            dims = child.child_by_field_name("dimensions")
            value = child.child_by_field_name("value")
            text = text_of(name, src) if name is not None else "?"
            if dims is not None:
                text += collapse(text_of(dims, src))
            if value is not None:
                text += " = " + cut(collapse(text_of(value, src)), init_limit)
            declarators.append(text)
        elif same(child, node.child_by_field_name("type")):
            type_part = cut(collapse(text_of(child, src)), type_limit)
    head = " ".join(p for p in (cut(modifiers, int(opts["modifiers_chars"])), type_part) if p)
    return f"{head} {', '.join(declarators)};"


def member_lines(node: Any, src: bytes, opts: dict[str, Any], indent: str) -> tuple[list[str], Counter]:
    """One line per member of a type declaration (bodies omitted)."""
    lines: list[str] = []
    counts: Counter = Counter()
    body = node.child_by_field_name("body")
    if body is None:
        return lines, counts
    limit = int(opts["signature_chars"])
    constants = [declared_name(c, src) for c in body.named_children if c.type == "enum_constant"]
    if constants:
        shown = int(opts["max_enum_constants"])
        text = ", ".join(constants[:shown])
        if len(constants) > shown:
            text += f", ... ({len(constants) - shown} more)"
        lines.append(f"{indent}{text};")
        counts["enum_constants"] += len(constants)
    for member in body_members(body):
        kind = member.type
        if kind in FIELD_DECLS:
            lines.append(indent + field_line(member, src, opts))
            counts["fields"] += sum(1 for c in member.children if c.type == "variable_declarator")
        elif kind in SIGNATURE_DECLS:
            lines.append(f"{indent}{header_text(member, src, limit)};")
            counts["constructors" if kind in ("constructor_declaration", "compact_constructor_declaration")
                   else "methods"] += 1
        elif kind in TYPE_DECLS:
            lines.append(f"{indent}{header_text(member, src, limit)} {{ ... }}")
            counts["nested_types"] += 1
        elif kind == "static_initializer":
            lines.append(f"{indent}static {{ ... }}")
        elif kind == "block":
            lines.append(f"{indent}{{ ... }}")
    return lines, counts


def type_skeleton(node: Any, src: bytes, package: str, opts: dict[str, Any], max_lines: int) -> tuple[str, dict[str, Any]]:
    """Skeleton of one type declaration, members past max_lines counted in a
    closing comment."""
    lines = [f"package {package};", ""] if package else []
    lines.append(header_text(node, src, int(opts["header_chars"])) + " {")
    members, counts = member_lines(node, src, opts, "    ")
    n_members = len(members)
    if n_members > max_lines:
        members = members[:max_lines] + [f"    // ... {n_members - max_lines} more members not shown"]
    lines.extend(members)
    lines.append("}")
    return "\n".join(lines), {"member_lines": n_members, "cut": max(0, n_members - max_lines), **counts}


def file_context(root: Any, src: bytes, target: Any, opts: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Package, import count and other top-level types of the file of a
    top-level class sample."""
    package = package_of(root, src)
    _, _, n_imports = imports_of(root, src)
    lines = [f"package {package};"] if package else ["// default package"]
    lines.append(f"// {n_imports} import declarations")
    others = [(name, n) for name, n in top_level_types(root, src) if n.start_byte != target.start_byte]
    if others:
        lines.append("// other top-level types in this file:")
        for _, n in others:
            lines.append(f"{header_text(n, src, int(opts['header_chars']))} {{ ... }}")
    else:
        lines.append("// no other top-level type in this file")
    return "\n".join(lines), {"imports": n_imports, "other_top_level_types": len(others)}


# ---------------------------------------------------------------- method facts


def type_members(node: Any, src: bytes) -> tuple[dict[str, str | None], set[str]]:
    """Fields {name: declared type} and method names declared directly in a type."""
    fields: dict[str, str | None] = {}
    methods: set[str] = set()
    for member in body_members(node.child_by_field_name("body")):
        if member.type in FIELD_DECLS:
            ftype = type_text(member.child_by_field_name("type"), src)
            for child in member.children:
                if child.type == "variable_declarator":
                    name = child.child_by_field_name("name")
                    if name is not None:
                        fields[text_of(name, src)] = ftype
        elif member.type == "method_declaration":
            methods.add(declared_name(member, src))
    if node.type == "record_declaration":
        params = node.child_by_field_name("parameters")
        for p in params.named_children if params is not None else []:
            if p.type == "formal_parameter":
                fields[declared_name(p, src)] = type_text(p.child_by_field_name("type"), src)
                methods.add(declared_name(p, src))
    if node.type == "enum_declaration":
        body = node.child_by_field_name("body")
        for c in body.named_children if body is not None else []:
            if c.type == "enum_constant":
                fields[declared_name(c, src)] = declared_name(node, src)
    return fields, methods


def anonymous_members(method: Any, src: bytes) -> tuple[dict[str, str | None], set[str]]:
    """Fields and methods of the anonymous class bodies around a method."""
    fields: dict[str, str | None] = {}
    methods: set[str] = set()
    cur = method.parent
    while cur is not None and cur.type not in TYPE_DECLS:
        if cur.type == "class_body" and cur.parent is not None and cur.parent.type == "object_creation_expression":
            for member in cur.named_children:
                if member.type == "field_declaration":
                    ftype = type_text(member.child_by_field_name("type"), src)
                    for child in member.children:
                        if child.type == "variable_declarator":
                            fields[declared_name(child, src)] = ftype
                elif member.type == "method_declaration":
                    methods.add(declared_name(member, src))
        cur = cur.parent
    return fields, methods


def local_declarations(method: Any, src: bytes) -> tuple[dict[str, tuple[str | None, str]], set[tuple[int, int]]]:
    """Parameters and local variables of a method (scopes merged) and the byte
    ranges of their names."""
    decls: dict[str, tuple[str | None, str]] = {}
    spans: set[tuple[int, int]] = set()

    def add(name_node: Any | None, tnode: Any | None, kind: str) -> None:
        if name_node is None or name_node.type != "identifier":
            return
        decls[text_of(name_node, src)] = (type_text(tnode, src), kind)
        spans.add((name_node.start_byte, name_node.end_byte))

    for node in walk(method):
        kind = node.type
        if kind in ("formal_parameter", "catch_formal_parameter"):
            tnode = node.child_by_field_name("type")
            if tnode is None:
                tnode = next((c for c in node.named_children if c.type == "catch_type"), None)
                if tnode is not None and tnode.named_child_count == 1:
                    tnode = tnode.named_children[0]
            add(node.child_by_field_name("name"), tnode, "parameter" if kind == "formal_parameter" else "local")
        elif kind == "spread_parameter":
            tnode = next((c for c in node.named_children if c.type not in ("modifiers", "variable_declarator")), None)
            for c in node.named_children:
                if c.type == "variable_declarator":
                    add(c.child_by_field_name("name"), tnode, "parameter")
        elif kind == "local_variable_declaration":
            tnode = node.child_by_field_name("type")
            for c in node.children:
                if c.type == "variable_declarator":
                    add(c.child_by_field_name("name"), tnode, "local")
        elif kind == "enhanced_for_statement":
            add(node.child_by_field_name("name"), node.child_by_field_name("type"), "local")
        elif kind == "resource":
            add(node.child_by_field_name("name"), node.child_by_field_name("type"), "local")
        elif kind == "lambda_expression":
            params = node.child_by_field_name("parameters")
            if params is None:
                continue
            if params.type == "identifier":
                add(params, None, "lambda parameter")
            elif params.type == "inferred_parameters":
                for c in params.named_children:
                    add(c, None, "lambda parameter")
        elif kind == "type_pattern":
            tnode = next((c for c in node.named_children if c.type != "identifier"), None)
            ident = next((c for c in node.named_children if c.type == "identifier"), None)
            add(ident, tnode, "local")
        elif kind == "instanceof_expression":
            name = node.child_by_field_name("name")
            if name is not None:
                add(name, node.child_by_field_name("right"), "local")
    return decls, spans


SKIP_IDENTIFIER_PARENTS = {"labeled_statement", "break_statement", "continue_statement", "switch_label",
                           "marker_annotation", "annotation", "scoped_identifier", "method_reference",
                           "enum_constant", "inferred_parameters"}


class MethodFacts:
    """Member accesses of one method body, by receiver."""

    def __init__(self, method: Any, src: bytes, owner: Any | None, outer: list[Any], anonymous: bool) -> None:
        self.src = src
        self.owner_name = declared_name(owner, src) if owner is not None else ""
        self.fields, self.methods = type_members(owner, src) if owner is not None else ({}, set())
        if anonymous:
            afields, amethods = anonymous_members(method, src)
            self.fields.update(afields)
            self.methods |= amethods
        self.outer_fields: dict[str, str | None] = {}
        for node in outer:
            f, _ = type_members(node, src)
            for k, v in f.items():
                self.outer_fields.setdefault(k, v)
        self.locals, self.decl_spans = local_declarations(method, src)
        params = method.child_by_field_name("parameters")
        self.n_params = sum(1 for c in params.named_children if c.type in ("formal_parameter", "spread_parameter")) \
            if params is not None else 0
        self.own_fields: Counter = Counter()
        self.own_methods: Counter = Counter()
        self.other_unqualified_calls: Counter = Counter()
        self.outer_field_uses: Counter = Counter()
        self.other_unqualified_names: Counter = Counter()
        self.foreign: dict[str, dict[str, Any]] = {}
        self.unresolved: Counter = Counter()
        self.created: Counter = Counter()
        self.static_type_refs: Counter = Counter()

    def chain(self, node: Any) -> list[str] | None:
        """Identifiers of a pure a.b.c chain, else None."""
        parts = []
        cur = node
        while cur is not None and cur.type == "field_access":
            field = cur.child_by_field_name("field")
            if field is None:
                return None
            parts.append(text_of(field, self.src))
            cur = cur.child_by_field_name("object")
        if cur is None or cur.type != "identifier":
            return None
        parts.append(text_of(cur, self.src))
        return parts[::-1]

    def package_chain(self, node: Any) -> list[str] | None:
        """Identifiers of a package-qualified chain (org.apache.Foo), else None."""
        parts = self.chain(node)
        if parts is None or len(parts) < 2 or not parts[0][:1].islower():
            return None
        if parts[0] in self.locals or parts[0] in self.fields or parts[0] in self.outer_fields:
            return None
        return parts

    def receiver(self, obj: Any) -> tuple[str, str | None, str]:
        """(kind, declared type, label) of a receiver expression."""
        src = self.src
        while obj.type == "parenthesized_expression" and obj.named_child_count == 1:
            obj = obj.named_children[0]
        if obj.type == "this" or obj.type == "super":
            return "own", None, obj.type
        if obj.type == "cast_expression":
            tname = type_text(obj.child_by_field_name("type"), src)
            if tname:
                return "typed", tname, "cast"
        qualified = self.package_chain(obj)
        if qualified is not None:
            k = next((j for j, part in enumerate(qualified) if TYPE_NAME_RE.match(part)), None)
            if k is None:
                return "unresolved", None, cut(".".join(qualified), 50)
            self.static_type_refs[qualified[k]] += 1
            if k == len(qualified) - 1:
                return "typed", qualified[k], "static"
            self.foreign_access(qualified[k], "static", qualified[k + 1])
            return "unresolved", None, cut(".".join(qualified[k:]), 50)
        if obj.type == "identifier":
            name = text_of(obj, src)
            if name in self.locals:
                tname, kind = self.locals[name]
                if tname is None:
                    return "unresolved", None, name
                return "typed", tname, f"{kind} {name}"
            if name in self.fields:
                self.own_fields[name] += 1
                tname = self.fields[name]
                return ("typed", tname, f"field {name}") if tname else ("unresolved", None, name)
            if name in self.outer_fields:
                self.outer_field_uses[name] += 1
                tname = self.outer_fields[name]
                return ("typed", tname, f"outer field {name}") if tname else ("unresolved", None, name)
            if TYPE_NAME_RE.match(name):
                self.static_type_refs[name] += 1
                return "typed", name, "static"
            self.other_unqualified_names[name] += 1
            return "unresolved", None, name
        if obj.type == "field_access":
            inner = obj.child_by_field_name("object")
            field = obj.child_by_field_name("field")
            if inner is not None and inner.type == "this" and field is not None:
                name = text_of(field, src)
                if name in self.fields:
                    self.own_fields[name] += 1
                    tname = self.fields[name]
                    return ("typed", tname, f"field {name}") if tname else ("unresolved", None, f"this.{name}")
                self.own_fields[name] += 1
                return "unresolved", None, f"this.{name}"
        return "unresolved", None, cut(collapse(text_of(obj, src)), 50)

    def foreign_access(self, tname: str, label: str, member: str) -> None:
        group = self.foreign.setdefault(tname, {"members": Counter(), "via": Counter()})
        group["members"][member] += 1
        group["via"][label] += 1

    def member_access(self, obj: Any | None, member: str, is_call: bool) -> None:
        if obj is None:
            if is_call:
                if member in self.methods:
                    self.own_methods[member] += 1
                else:
                    self.other_unqualified_calls[member] += 1
            return
        kind, tname, label = self.receiver(obj)
        if kind == "own":
            (self.own_methods if is_call else self.own_fields)[member] += 1
        elif kind == "typed":
            self.foreign_access(tname, label, member + ("()" if is_call else ""))
        else:
            self.unresolved[f"{label}.{member}{'()' if is_call else ''}"] += 1

    def visit(self, node: Any) -> None:
        stack = [node]
        src = self.src
        while stack:
            cur = stack.pop()
            kind = cur.type
            if kind in ("marker_annotation", "annotation") or kind in COMMENT_TYPES:
                continue
            if kind == "method_invocation":
                obj = cur.child_by_field_name("object")
                name = cur.child_by_field_name("name")
                if name is not None:
                    self.member_access(obj, text_of(name, src), True)
                # only a compound receiver is visited
                simple = obj is not None and (obj.type in ("identifier", "this", "super") or self._is_this_field(obj)
                                              or self.package_chain(obj) is not None)
                for child in reversed(cur.children):
                    if same(child, name) or (simple and same(child, obj)):
                        continue
                    stack.append(child)
                continue
            if kind == "field_access":
                obj = cur.child_by_field_name("object")
                field = cur.child_by_field_name("field")
                qualified = self.package_chain(cur)
                if qualified is not None:
                    # org.apache.Foo or org.apache.Foo.BAR
                    k = next((j for j, part in enumerate(qualified) if TYPE_NAME_RE.match(part)), None)
                    if k is not None:
                        self.static_type_refs[qualified[k]] += 1
                        if k < len(qualified) - 1:
                            self.foreign_access(qualified[k], "static", qualified[k + 1])
                    continue
                if obj is not None and field is not None:
                    if obj.type == "this" or obj.type == "super":
                        self.own_fields[text_of(field, src)] += 1
                    else:
                        self.member_access(obj, text_of(field, src), False)
                        if obj.type != "identifier" and not self._is_this_field(obj):
                            stack.append(obj)
                continue
            if kind == "method_reference":
                named = cur.named_children
                if named:
                    head, tail = named[0], named[-1]
                    if tail.type == "identifier" and head is not tail:
                        self.member_access(head, text_of(tail, src), True)
                continue
            if kind == "object_creation_expression":
                tnode = cur.child_by_field_name("type")
                tname = type_text(tnode, src)
                if tname:
                    self.created[tname] += 1
                stack.extend(reversed([c for c in cur.children if c is not tnode]))
                continue
            if kind == "identifier":
                if (cur.start_byte, cur.end_byte) in self.decl_spans:
                    continue
                parent = cur.parent
                if parent is not None and parent.type in SKIP_IDENTIFIER_PARENTS:
                    continue
                name = text_of(cur, src)
                if name in self.locals:
                    continue
                if name in self.fields:
                    self.own_fields[name] += 1
                elif name in self.outer_fields:
                    self.outer_field_uses[name] += 1
                elif not TYPE_NAME_RE.match(name):
                    self.other_unqualified_names[name] += 1
                continue
            stack.extend(reversed(cur.children))

    @staticmethod
    def _is_this_field(node: Any) -> bool:
        if node.type != "field_access":
            return False
        inner = node.child_by_field_name("object")
        return inner is not None and inner.type == "this"

    def as_dict(self) -> dict[str, Any]:
        foreign = {}
        for tname, group in sorted(self.foreign.items(), key=lambda kv: (-sum(kv[1]["members"].values()), kv[0])):
            foreign[tname] = {"count": int(sum(group["members"].values())),
                              "same_type": tname == self.owner_name,
                              "members": dict(group["members"].most_common()),
                              "via": dict(group["via"].most_common())}
        return {
            "owner": self.owner_name,
            "owner_fields": len(self.fields),
            "owner_methods": len(self.methods),
            "parameters": self.n_params,
            "own_field_accesses": dict(self.own_fields.most_common()),
            "own_method_calls": dict(self.own_methods.most_common()),
            "other_unqualified_calls": dict(self.other_unqualified_calls.most_common()),
            "outer_field_accesses": dict(self.outer_field_uses.most_common()),
            "other_unqualified_names": dict(self.other_unqualified_names.most_common()),
            "foreign": foreign,
            "unresolved_receivers": dict(self.unresolved.most_common()),
            "created_types": dict(self.created.most_common()),
        }


def counted(items: dict[str, int], limit: int) -> str:
    shown = list(items.items())[:limit]
    text = ", ".join(f"{k} x{v}" for k, v in shown)
    if len(items) > limit:
        text += f", ... ({len(items) - limit} more)"
    return text


def method_facts_text(facts: dict[str, Any], opts: dict[str, Any]) -> str:
    k = int(opts["facts_items"])
    groups = int(opts["facts_groups"])
    total = lambda d: sum(d.values())  # noqa: E731
    lines = [f"Declared in {facts['owner'] or 'an unknown type'} ({facts['owner_fields']} fields, "
             f"{facts['owner_methods']} methods); {facts['parameters']} parameters."]
    own_f, own_m = facts["own_field_accesses"], facts["own_method_calls"]
    lines.append(f"Accesses to fields of its own class: {total(own_f)}" + (f" ({counted(own_f, k)})" if own_f else ""))
    lines.append(f"Calls to methods of its own class: {total(own_m)}" + (f" ({counted(own_m, k)})" if own_m else ""))
    other = facts["other_unqualified_calls"]
    if other:
        lines.append(f"Unqualified calls to methods not declared in its class (inherited or statically imported): "
                     f"{total(other)} ({counted(other, k)})")
    outer = facts["outer_field_accesses"]
    if outer:
        lines.append(f"Accesses to fields of enclosing classes: {total(outer)} ({counted(outer, k)})")
    foreign = facts["foreign"]
    n_foreign = sum(g["count"] for g in foreign.values())
    lines.append(f"Accesses to members of other objects or types: {n_foreign}, grouped by declared type of the receiver:"
                 if foreign else "Accesses to members of other objects or types: 0")
    for i, (tname, group) in enumerate(foreign.items()):
        if i >= groups:
            lines.append(f"  ... ({len(foreign) - groups} more types)")
            break
        via = ", ".join(group["via"].keys())
        note = ", same type as its own class" if group["same_type"] else ""
        lines.append(f"  {tname} (via {cut(via, 80)}{note}): {group['count']} ({counted(group['members'], k)})")
    unresolved = facts["unresolved_receivers"]
    if unresolved:
        lines.append(f"Accesses on receivers of unknown declared type: {total(unresolved)} ({counted(unresolved, k)})")
    names = facts["other_unqualified_names"]
    if names:
        lines.append(f"Other unqualified names (inherited fields, constants or enum values): {total(names)} "
                     f"({counted(names, k)})")
    created = facts["created_types"]
    if created:
        lines.append(f"Objects created: {total(created)} ({counted(created, k)})")
    return "\n".join(lines)


# ---------------------------------------------------------------- class facts


def statement_list(body: Any | None) -> list[Any]:
    if body is None:
        return []
    return [c for c in body.named_children if c.type not in COMMENT_TYPES]


def accessor_kind(method: Any, src: bytes) -> str | None:
    """Return getter or setter when the method has the shape of one, else None."""
    name = declared_name(method, src)
    params = method.child_by_field_name("parameters")
    n_params = sum(1 for c in params.named_children if c.type in ("formal_parameter", "spread_parameter")) \
        if params is not None else 0
    stmts = statement_list(method.child_by_field_name("body"))
    if GETTER_RE.match(name) and n_params == 0 and len(stmts) == 1 and stmts[0].type == "return_statement":
        return "getter"
    if SETTER_RE.match(name) and n_params == 1 and 1 <= len(stmts) <= 2:
        first = stmts[0]
        assign = first.type == "expression_statement" and first.named_child_count == 1 \
            and first.named_children[0].type == "assignment_expression"
        tail_ok = len(stmts) == 1 or (stmts[1].type == "return_statement"
                                      and strip_ws(text_of(stmts[1], src)) == "returnthis;")
        if assign and tail_ok:
            return "setter"
    return None


def class_facts(node: Any, src: bytes, start: int, end: int) -> dict[str, Any]:
    counts: Counter = Counter()
    for member in body_members(node.child_by_field_name("body")):
        kind = member.type
        if kind in FIELD_DECLS:
            n = sum(1 for c in member.children if c.type == "variable_declarator")
            counts["fields"] += n
            mods = next((c for c in member.children if c.type == "modifiers"), None)
            words = set(text_of(mods, src).split()) if mods is not None else set()
            if "static" in words or kind == "constant_declaration":
                counts["static_fields"] += n
            if "public" in words and "static" not in words:
                counts["public_instance_fields"] += n
        elif kind == "method_declaration":
            acc = accessor_kind(member, src)
            counts[acc + "s" if acc else "other_methods"] += 1
        elif kind in ("constructor_declaration", "compact_constructor_declaration"):
            counts["constructors"] += 1
        elif kind in TYPE_DECLS:
            counts["nested_types"] += 1
    if node.type == "record_declaration":
        params = node.child_by_field_name("parameters")
        counts["fields"] += sum(1 for c in params.named_children if c.type == "formal_parameter") if params else 0
    if node.type == "enum_declaration":
        body = node.child_by_field_name("body")
        counts["enum_constants"] += sum(1 for c in body.named_children if c.type == "enum_constant") if body else 0
    keys = ("fields", "static_fields", "public_instance_fields", "getters", "setters", "other_methods",
            "constructors", "nested_types", "enum_constants")
    out = {k: int(counts.get(k, 0)) for k in keys}
    out["kind"] = KIND_WORD.get(node.type, node.type)
    out["name"] = declared_name(node, src)
    out["lines"] = end - start + 1
    return out


def class_facts_text(facts: dict[str, Any]) -> str:
    lines = [f"Kind: {facts['kind']} {facts['name']}, {facts['lines']} lines."]
    lines.append(f"Fields: {facts['fields']} (static: {facts['static_fields']}, "
                 f"public non-static: {facts['public_instance_fields']}).")
    accessors = facts["getters"] + facts["setters"]
    n_methods = accessors + facts["other_methods"]
    lines.append(f"Methods: {n_methods} (getter or setter shaped: {accessors}, of which {facts['getters']} getters "
                 f"and {facts['setters']} setters; other methods: {facts['other_methods']}); "
                 f"constructors: {facts['constructors']}.")
    if facts["nested_types"] or facts["enum_constants"]:
        lines.append(f"Nested types: {facts['nested_types']}; enum constants: {facts['enum_constants']}.")
    if facts.get("mentioning_files") is not None:
        lines.append(f"Other files of the repository that mention {facts['name']}: {facts['mentioning_files']} "
                     f"(of {facts['repository_files']} Java files).")
    return "\n".join(lines)


# ---------------------------------------------------------------- type index and resolution


def index_path(index_dir: Path, snapshot: str) -> Path:
    return index_dir / f"{snapshot}.json"


def java_files(snapshot_dir: Path) -> list[str]:
    return sorted(str(p.relative_to(snapshot_dir)) for p in snapshot_dir.rglob("*.java") if p.is_file())


def build_type_index(snapshot_dir: Path) -> dict[str, Any]:
    """{name: [[path, package], ...]} of the top-level types of every file."""
    files = java_files(snapshot_dir)
    types: dict[str, list[list[str]]] = defaultdict(list)
    errors = 0
    for rel in files:
        try:
            src = (snapshot_dir / rel).read_bytes()
        except OSError:
            errors += 1
            continue
        root = parse(src).root_node
        package = package_of(root, src)
        for name, _ in top_level_types(root, src):
            types[name].append([rel, package])
    return {"snapshot": snapshot_dir.name, "n_files": len(files), "read_errors": errors, "files": files,
            "types": dict(sorted(types.items()))}


def load_type_index(snapshot_dir: Path, index_dir: Path, rebuild: bool = False) -> dict[str, Any]:
    path = index_path(index_dir, snapshot_dir.name)
    if path.exists() and not rebuild:
        return json.loads(path.read_text())
    index = build_type_index(snapshot_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(index))
    tmp.replace(path)
    return index


class FileContext:
    """What type resolution needs to know about the file of a sample."""

    def __init__(self, rel: str, root: Any, src: bytes) -> None:
        self.rel = rel
        self.package = package_of(root, src)
        self.single, self.wildcard, self.n_imports = imports_of(root, src)
        self.local_types = {name for name, _ in all_type_nodes(root, src)}


def resolve_type(name: str, ctx: FileContext, types: dict[str, list[list[str]]], implicit: set[str]) -> dict[str, Any]:
    """Resolve a type name used in a file to a file of the snapshot.

    Returns {status, how, path, package, inner}.
    """
    parts = name.split(".")
    if len(parts) > 1:
        if parts[0][:1].islower():
            pkg_parts = []
            while parts and parts[0][:1].islower():
                pkg_parts.append(parts.pop(0))
            if not parts:
                return {"status": "unresolved", "how": None}
            pkg = ".".join(pkg_parts)
            hits = [e for e in types.get(parts[0], []) if e[1] == pkg]
            if hits:
                return {"status": "resolved", "how": "qualified", "path": hits[0][0], "package": pkg,
                        "inner": ".".join(parts)}
            return {"status": "external", "how": "qualified"}
        head = resolve_type(parts[0], ctx, types, implicit)
        if head["status"] == "resolved":
            head = dict(head)
            head["inner"] = ".".join([head.get("inner") or parts[0]] + parts[1:])
        return head
    if name in ctx.local_types:
        return {"status": "resolved", "how": "same_file", "path": ctx.rel, "package": ctx.package, "inner": name}
    entries = types.get(name, [])
    if name in ctx.single:
        # a.b.C, or a.b.Outer.C for an imported nested type
        segs = ctx.single[name].split(".")
        for j in range(len(segs) - 1, -1, -1):
            pkg, outer = ".".join(segs[:j]), segs[j]
            hits = [e for e in types.get(outer, []) if e[1] == pkg]
            if hits:
                return {"status": "resolved", "how": "import", "path": hits[0][0], "package": pkg,
                        "inner": ".".join(segs[j:])}
        return {"status": "external", "how": "import"}
    same = [e for e in entries if e[1] == ctx.package]
    if same:
        here = os.path.dirname(ctx.rel)
        same.sort(key=lambda e: (os.path.dirname(e[0]) != here, e[0]))
        return {"status": "resolved", "how": "same_package", "path": same[0][0], "package": same[0][1], "inner": name}
    wild = sorted(e for e in entries if e[1] in ctx.wildcard)
    if wild:
        return {"status": "resolved", "how": "wildcard_import", "path": wild[0][0], "package": wild[0][1],
                "inner": name}
    if name in implicit:
        return {"status": "implicit", "how": None}
    if len(entries) == 1:
        return {"status": "resolved", "how": "unique_name", "path": entries[0][0], "package": entries[0][1],
                "inner": name}
    if len(entries) > 1:
        return {"status": "ambiguous", "how": None, "candidates": sorted(e[0] for e in entries)}
    return {"status": "unresolved", "how": None}


def type_references(node: Any, src: bytes, exclude: set[str]) -> Counter:
    """Type names used under a node: type identifiers (Outer.Inner kept
    whole) and capitalised identifiers used as static receivers."""
    refs: Counter = Counter()
    declared_params = set()
    for n in walk(node):
        if n.type == "type_parameter":
            ident = next((c for c in n.named_children if c.type in ("type_identifier", "identifier")), None)
            if ident is not None:
                declared_params.add(text_of(ident, src))
    for n in walk(node):
        kind = n.type
        parent = n.parent
        if kind == "type_identifier" and (parent is None or parent.type != "scoped_type_identifier"):
            if parent is not None and parent.type == "type_parameter":
                continue
            refs[text_of(n, src)] += 1
        elif kind == "scoped_type_identifier" and (parent is None or parent.type != "scoped_type_identifier"):
            refs[collapse(text_of(n, src)).replace(" ", "")] += 1
        elif kind == "identifier" and parent is not None and parent.type in ("method_invocation", "field_access") \
                and same(parent.child_by_field_name("object"), n) and TYPE_NAME_RE.match(text_of(n, src)):
            refs[text_of(n, src)] += 1
    for name in list(refs):
        if name in exclude or name in declared_params or name.split(".")[0] in declared_params:
            del refs[name]
    return refs


class SourceCache:
    """Parsed files of one snapshot (bytes and tree), most recent first."""

    def __init__(self, snapshot_dir: Path, capacity: int = 256) -> None:
        self.dir = snapshot_dir
        self.capacity = capacity
        self.items: dict[str, tuple[bytes, Any]] = {}

    def get(self, rel: str) -> tuple[bytes, Any] | None:
        if rel in self.items:
            value = self.items.pop(rel)
            self.items[rel] = value
            return value
        try:
            src = (self.dir / rel).read_bytes()
        except OSError:
            return None
        value = (src, parse(src).root_node)
        self.items[rel] = value
        if len(self.items) > self.capacity:
            self.items.pop(next(iter(self.items)))
        return value


def resolved_skeleton(res: dict[str, Any], cache: SourceCache, opts: dict[str, Any], max_lines: int) -> str | None:
    hit = cache.get(res["path"])
    if hit is None:
        return None
    src, root = hit
    node = find_type_node(root, src, res.get("inner") or "")
    if node is None:
        return None
    text, _ = type_skeleton(node, src, res.get("package") or "", opts, max_lines)
    return f"// {res['path']}\n{text}"


def proximity_order(files: list[str], rel: str) -> list[str]:
    """Files ordered by closeness to rel: rel itself, its directory, its
    top-level directory, then the rest, each group in path order."""
    here = os.path.dirname(rel)
    top = rel.split("/", 1)[0]

    def key(path: str) -> tuple[int, str]:
        if path == rel:
            return 0, path
        if os.path.dirname(path) == here:
            return 1, path
        if path.split("/", 1)[0] == top:
            return 2, path
        return 3, path

    return sorted(files, key=key)


def usage_lines(snapshot_dir: Path, files: list[str], name: str, exclude: set[str], max_lines: int,
                per_file: int, line_chars: int, texts: dict[str, str] | None = None) -> dict[str, Any]:
    """Lines that mention a name (word boundary), import and package lines
    skipped, files in the given order, at most per_file lines per file."""
    pattern = re.compile(rf"(?<![\w$]){re.escape(name)}(?![\w$])")
    out: list[str] = []
    n_files = n_lines = 0
    for rel in files:
        if rel in exclude:
            continue
        text = texts.get(rel) if texts is not None else None
        if text is None:
            try:
                text = (snapshot_dir / rel).read_text(errors="replace")
            except OSError:
                continue
        if name not in text or not pattern.search(text):
            continue
        n_files += 1
        taken = 0
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith(("import ", "package ")) or not pattern.search(line):
                continue
            n_lines += 1
            if taken < per_file and len(out) < max_lines:
                out.append(f"{rel}:{lineno}: {cut(stripped, line_chars)}")
                taken += 1
    return {"lines": out, "n_files": n_files, "n_lines": n_lines}


# ---------------------------------------------------------------- one sample


def locate_target(root: Any, level: str, start: int, end: int) -> tuple[Any | None, float]:
    """Declaration of the sample level whose line range best overlaps the
    snippet range (intersection over union)."""
    best, best_iou, best_size = None, 0.0, 0
    for node in walk(root):
        if node.type not in LEVEL_KINDS[level]:
            continue
        lo, hi = first_line(node), last_line(node)
        inter = min(hi, end) - max(lo, start) + 1
        if inter <= 0:
            continue
        iou = inter / (max(hi, end) - min(lo, start) + 1)
        size = hi - lo
        if iou > best_iou or (iou == best_iou and size < best_size):
            best, best_iou, best_size = node, iou, size
    return best, best_iou


def innermost_type_at(root: Any, line: int) -> Any | None:
    best = None
    for node in walk(root):
        if node.type in TYPE_DECLS and first_line(node) <= line <= last_line(node):
            if best is None or (last_line(node) - first_line(node)) < (last_line(best) - first_line(best)):
                best = node
    return best


def empty_package(json_index: int, level: str, snapshot: str, rel: str) -> dict[str, Any]:
    return {"json_index": json_index, "level": level, "snapshot": snapshot, "file": rel,
            "found_snapshot": False, "found_file": False, "range_ok": False, "snippet_match": False,
            "target": None, "enclosing": None, "skeleton": None, "skeleton_kind": None, "skeleton_stats": None,
            "facts": None, "facts_text": None, "referenced": None, "referenced_skeletons": [],
            "usages": None, "resolution": None}


def build_package(task: dict[str, Any], index: dict[str, Any], cache: SourceCache, opts: dict[str, Any],
                  implicit: set[str]) -> dict[str, Any]:
    """Context package of one sample whose snapshot exists."""
    out = empty_package(task["json_index"], task["level"], task["snapshot"], task["file"])
    out["found_snapshot"] = True
    hit = cache.get(task["file"])
    if hit is None:
        return out
    src, root = hit
    out["found_file"] = True
    start, end = int(task["start_line"]), int(task["end_line"])
    lines = src.decode("utf-8", errors="replace").splitlines()
    out["range_ok"] = 1 <= start <= end <= len(lines)
    if out["range_ok"]:
        out["snippet_match"] = strip_ws("\n".join(lines[start - 1:end])) == strip_ws(task["code_snippet"])
    level = task["level"]
    target, iou = locate_target(root, level, start, end)
    ctx = FileContext(task["file"], root, src)
    max_lines = int(opts["skeleton_max_members"])

    if target is not None:
        owners, anonymous = enclosing_types(target)
        out["target"] = {"kind": target.type, "name": declared_name(target, src), "start_line": first_line(target),
                         "end_line": last_line(target), "line_iou": round(iou, 4)}
    else:
        owner = innermost_type_at(root, start)
        owners, anonymous = ([owner] if owner is not None else []), False
    owner = owners[0] if owners else None
    if owner is not None:
        out["enclosing"] = {"kind": owner.type, "name": declared_name(owner, src), "start_line": first_line(owner),
                            "end_line": last_line(owner), "depth": len(owners), "anonymous": anonymous}

    # (a) skeleton
    if level == "method" and owner is not None:
        out["skeleton"], out["skeleton_stats"] = type_skeleton(owner, src, ctx.package, opts, max_lines)
        out["skeleton_kind"] = "enclosing_type"
    elif level == "class" and target is not None:
        if owner is not None:
            out["skeleton"], out["skeleton_stats"] = type_skeleton(owner, src, ctx.package, opts, max_lines)
            out["skeleton_kind"] = "outer_type"
        else:
            out["skeleton"], out["skeleton_stats"] = file_context(root, src, target, opts)
            out["skeleton_kind"] = "file_context"

    # (b) facts
    if level == "method" and target is not None:
        mf = MethodFacts(target, src, owner, owners[1:], anonymous)
        body = target.child_by_field_name("body")
        if body is not None:
            mf.visit(body)
        out["facts"] = mf.as_dict()
        out["facts_text"] = method_facts_text(out["facts"], opts)
    elif level == "class" and target is not None:
        out["facts"] = class_facts(target, src, start, end)

    # (c) referenced types
    if target is not None:
        exclude = {declared_name(n, src) for n in owners}
        if level == "class":
            exclude.add(declared_name(target, src))
        refs = type_references(target, src, exclude)
        resolved, external, ambiguous, unresolved, implicit_hits = [], [], [], [], []
        by_how: Counter = Counter()
        for name, count in sorted(refs.items(), key=lambda kv: (-kv[1], kv[0])):
            res = resolve_type(name, ctx, index["types"], implicit)
            if res["status"] == "resolved":
                # a nested type of the sample itself is part of the snippet
                if res["how"] == "same_file" and level == "class":
                    node = find_type_node(root, src, res["inner"])
                    if node is not None and target.start_byte <= node.start_byte < target.end_byte:
                        continue
                by_how[res["how"]] += 1
                resolved.append({"name": name, "count": count, "how": res["how"], "path": res["path"],
                                 "package": res.get("package"), "inner": res.get("inner")})
            elif res["status"] == "external":
                external.append(name)
            elif res["status"] == "ambiguous":
                ambiguous.append({"name": name, "candidates": len(res["candidates"])})
            elif res["status"] == "implicit":
                implicit_hits.append(name)
            else:
                unresolved.append(name)
        out["referenced"] = {"names": dict(sorted(refs.items(), key=lambda kv: (-kv[1], kv[0]))),
                             "resolved": resolved, "external": external, "implicit": implicit_hits,
                             "ambiguous": ambiguous, "unresolved": unresolved}
        out["resolution"] = {"n_type_names": len(refs), "n_resolved": len(resolved), "n_external": len(external),
                             "n_implicit": len(implicit_hits), "n_ambiguous": len(ambiguous),
                             "n_unresolved": len(unresolved), "by_how": dict(by_how)}
        skeletons = []
        for item in resolved[: int(opts["top_referenced_types"])]:
            text = resolved_skeleton(item, cache, opts, int(opts["referenced_max_members"]))
            if text is not None:
                skeletons.append({"name": item["name"], "path": item["path"], "how": item["how"],
                                  "count": item["count"], "skeleton": text})
        out["referenced_skeletons"] = skeletons
    return out


def process_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    """Packages of every sample of one snapshot (worker process)."""
    started = time.time()
    snapshot_dir = Path(job["snapshot_dir"])
    opts, tasks = job["opts"], job["tasks"]
    implicit = set(opts["implicit_types"])
    if not snapshot_dir.is_dir():
        return {"packages": [empty_package(t["json_index"], t["level"], t["snapshot"], t["file"]) for t in tasks],
                "snapshot": snapshot_dir.name, "seconds": 0.0, "n_files": 0}
    index = load_type_index(snapshot_dir, Path(job["index_dir"]), bool(job.get("rebuild_index")))
    cache = SourceCache(snapshot_dir)
    packages = [build_package(t, index, cache, opts, implicit) for t in tasks]

    # mentions and usage lines of the class samples, one pass over the files
    wanted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pkg in packages:
        if pkg["level"] == "class" and pkg["facts"] is not None:
            wanted[pkg["facts"]["name"]].append(pkg)
    if wanted:
        names = set(wanted)
        mentions: dict[str, set[str]] = defaultdict(set)
        texts: dict[str, str] = {}
        for rel in index["files"]:
            try:
                text = (snapshot_dir / rel).read_text(errors="replace")
            except OSError:
                continue
            hits = names.intersection(WORD_RE.findall(text))
            for name in hits:
                mentions[name].add(rel)
            if hits:
                texts[rel] = text
        for name, pkgs in wanted.items():
            files = sorted(mentions.get(name, set()))
            for pkg in pkgs:
                own = {pkg["file"]}
                pkg["facts"]["mentioning_files"] = len([f for f in files if f not in own])
                pkg["facts"]["repository_files"] = int(index["n_files"])
                pkg["facts_text"] = class_facts_text(pkg["facts"])
                pkg["usages"] = usage_lines(snapshot_dir, proximity_order(files, pkg["file"]), name, own,
                                            int(opts["max_usage_lines"]),
                                            int(opts["usage_lines_per_file"]), int(opts["usage_line_chars"]), texts)
    return {"packages": packages, "snapshot": snapshot_dir.name, "seconds": round(time.time() - started, 2),
            "n_files": int(index["n_files"])}


# ---------------------------------------------------------------- main


def read_levels(path: Path) -> dict[int, str]:
    with path.open() as fh:
        return {int(r["json_index"]): r["level"] for r in csv.DictReader(fh)}


def summarize(packages: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {"samples": len(packages)}
    for key in ("found_snapshot", "found_file", "range_ok", "snippet_match"):
        stats[key] = sum(1 for p in packages if p[key])
    stats["snippet_mismatch"] = sum(1 for p in packages if p["range_ok"] and not p["snippet_match"])
    stats["target_found"] = sum(1 for p in packages if p["target"] is not None)
    ious = [p["target"]["line_iou"] for p in packages if p["target"] is not None]
    stats["target_exact_range"] = sum(1 for v in ious if v == 1.0)
    stats["target_iou_below_0.9"] = sum(1 for v in ious if v < 0.9)
    by_level: dict[str, Any] = {}
    for level in ("method", "class"):
        part = [p for p in packages if p["level"] == level]
        res = [p["resolution"] for p in part if p["resolution"] is not None]
        how: Counter = Counter()
        for r in res:
            how.update(r["by_how"])
        n_names = sum(r["n_type_names"] for r in res)
        n_resolved = sum(r["n_resolved"] for r in res)
        by_level[level] = {
            "samples": len(part),
            "target_found": sum(1 for p in part if p["target"] is not None),
            "skeleton": dict(Counter(p["skeleton_kind"] for p in part if p["skeleton_kind"])),
            "facts": sum(1 for p in part if p["facts_text"]),
            "type_names": n_names,
            "resolved": n_resolved,
            "resolved_share": round(n_resolved / n_names, 4) if n_names else None,
            "external": sum(r["n_external"] for r in res),
            "implicit": sum(r["n_implicit"] for r in res),
            "ambiguous": sum(r["n_ambiguous"] for r in res),
            "unresolved": sum(r["n_unresolved"] for r in res),
            "resolved_by": dict(how.most_common()),
            "samples_with_resolved_type": sum(1 for r in res if r["n_resolved"] > 0),
            "samples_with_referenced_skeleton": sum(1 for p in part if p["referenced_skeletons"]),
            "referenced_skeletons": sum(len(p["referenced_skeletons"]) for p in part),
        }
        if level == "class":
            by_level[level]["samples_with_usage_lines"] = sum(1 for p in part if p["usages"] and p["usages"]["lines"])
            mentions = sorted(p["facts"]["mentioning_files"] for p in part
                              if p["facts"] and "mentioning_files" in p["facts"])
            by_level[level]["mentioning_files_median"] = mentions[len(mentions) // 2] if mentions else None
            by_level[level]["mentioning_files_zero"] = sum(1 for v in mentions if v == 0)
        else:
            by_level[level]["in_anonymous_class"] = sum(1 for p in part if p["enclosing"] and p["enclosing"]["anonymous"])
        by_level[level]["enclosing_depth_2_or_more"] = sum(1 for p in part if p["enclosing"] and p["enclosing"]["depth"] > 1)
    stats["by_level"] = by_level
    return stats


def main() -> None:
    parser_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser_.add_argument("--config", type=Path, required=True)
    parser_.add_argument("--limit", type=int, default=None, help="first N json_index only (quick check)")
    parser_.add_argument("--rebuild-index", action="store_true", help="rebuild the cached type indexes")
    args, extra = parser_.parse_known_args()
    cfg = load_config(args.config, parse_cli_overrides(extra))
    ccfg = cfg["context_study"]
    pcfg = ccfg["packages"]
    opts = dict(pcfg["options"])
    snapshots_dir = resolve(ccfg["snapshots_dir"])
    index_dir = resolve(pcfg["type_index_dir"])

    folds = json.loads(resolve(ccfg["folds"]).read_text())
    indices = [int(i) for i in folds["meta"]["json_index"]]
    if args.limit:
        indices = indices[: args.limit]
    entries = json.loads(resolve(ccfg["input_json"]).read_text())
    levels = read_levels(resolve(ccfg["sample_meta"]))

    jobs: dict[str, dict[str, Any]] = {}
    for idx in indices:
        entry = entries[idx]
        snap = snapshot_name(repo_name(entry["repo_url"]), entry["commit_hash"])
        job = jobs.setdefault(snap, {"snapshot_dir": str(snapshots_dir / snap), "index_dir": str(index_dir),
                                     "opts": opts, "tasks": [], "rebuild_index": args.rebuild_index})
        job["tasks"].append({"json_index": idx, "level": levels[idx], "snapshot": snap,
                             "file": entry["file_path"].lstrip("/"), "start_line": entry["start_line"],
                             "end_line": entry["end_line"], "code_snippet": entry["code_snippet"]})
    # largest snapshots first, for load balance
    order = sorted(jobs.values(), key=lambda j: -len(j["tasks"]))
    print(f"[packages] {len(indices)} samples in {len(jobs)} snapshots, {int(pcfg['workers'])} workers", flush=True)

    started = time.time()
    packages: list[dict[str, Any]] = []
    seconds: list[tuple[float, str]] = []
    workers = int(pcfg["workers"])
    if workers <= 1:
        results = map(process_snapshot, order)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(process_snapshot, order, chunksize=1)
    for k, result in enumerate(results, start=1):
        packages.extend(result["packages"])
        seconds.append((result["seconds"], result["snapshot"]))
        if k % 50 == 0 or k == len(order):
            print(f"[packages] {k}/{len(order)} snapshots, {len(packages)} samples, "
                  f"{time.time() - started:.0f}s", flush=True)
    if workers > 1:
        pool.shutdown()
    packages.sort(key=lambda p: p["json_index"])

    out_path = resolve(pcfg["jsonl"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with tmp.open("w") as fh:
        for pkg in packages:
            fh.write(json.dumps(pkg) + "\n")
    tmp.replace(out_path)

    stats = summarize(packages)
    stats["snapshots"] = len(jobs)
    stats["missing_snapshots"] = sorted({p["snapshot"] for p in packages if not p["found_snapshot"]})
    stats["seconds"] = round(time.time() - started, 1)
    stats["slowest_snapshots"] = [[s, n] for s, n in sorted(seconds, reverse=True)[:5]]
    stats["options"] = opts
    stats_path = resolve(pcfg["stats"])
    stats_path.write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("options", "by_level")}, indent=1), flush=True)
    print(json.dumps(stats["by_level"], indent=1), flush=True)
    print(f"[packages] wrote {out_path} and {stats_path}", flush=True)


if __name__ == "__main__":
    main()
