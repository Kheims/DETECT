"""Source transformations of Java snippets for metamorphic tests of smell detectors.

Snippets are parsed with tree-sitter-java inside a wrapper class and edited
through byte spans. Transforms are grouped by their expected effect on the
label (presentation_invariant, presentation_ambiguous, evidence_ablation,
suggestion, construct_control). Each variant is checked against its base
(parse errors, tokens, node types or name bindings) and a sample that fails
keeps its base code with applied=False.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Iterator

COMMENT_TYPES = frozenset({"line_comment", "block_comment"})
LITERAL_TYPES = frozenset({"string_literal", "character_literal"})
TYPE_DECLS = frozenset({"class_declaration", "interface_declaration", "enum_declaration", "record_declaration",
                        "annotation_type_declaration"})
BODY_TYPES = frozenset({"class_body", "interface_body", "enum_body", "annotation_type_body"})
METHOD_TYPES = frozenset({"method_declaration", "constructor_declaration", "compact_constructor_declaration"})
# members whose code runs (a local belongs to the outermost one around it)
UNIT_TYPES = METHOD_TYPES | {"static_initializer", "field_declaration", "constant_declaration", "enum_constant"}
SCOPE_BLOCKS = frozenset({"block", "constructor_body", "switch_block", "lambda_expression", "field_declaration",
                          "constant_declaration"})
# parents whose identifier children are neither variables nor names declared in the snippet
NOT_NAMES = frozenset({"labeled_statement", "break_statement", "continue_statement", "scoped_identifier",
                       "package_declaration", "import_declaration", "annotation_type_element_declaration",
                       "receiver_parameter", "module_declaration", "requires_module_directive",
                       "exports_module_directive", "opens_module_directive", "uses_module_directive",
                       "provides_module_directive"})
PRIMITIVE_TYPES = frozenset({"byte", "short", "char", "int", "long", "float", "double", "boolean"})
# declared types of a variable that can be a constant variable (JLS 4.12.4), hence a case constant
CONSTANT_TYPES = PRIMITIVE_TYPES | {"String"}
# selector types of a switch whose case labels are constant expressions (JLS 14.11.1)
CONSTANT_SELECTORS = CONSTANT_TYPES | {"Byte", "Short", "Character", "Integer", "Long", "Float", "Double", "Boolean"}
# expressions whose type is primitive or String: they make the switch a constant switch
CONSTANT_EXPRESSIONS = frozenset({"binary_expression", "unary_expression", "update_expression", "string_literal",
                                  "character_literal", "decimal_integer_literal", "hex_integer_literal",
                                  "octal_integer_literal", "binary_integer_literal", "decimal_floating_point_literal",
                                  "hex_floating_point_literal", "true", "false"})

CLASS_NAMES = ("presentation_invariant", "presentation_ambiguous", "evidence_ablation", "suggestion",
               "construct_control")
SUGGESTIONS = ("comment_claims_smell", "comment_denies_smell")
LEVELS = ("method", "class")

_PARSER: Any = None


def _parser() -> Any:
    global _PARSER
    if _PARSER is None:
        import tree_sitter as ts
        import tree_sitter_java as tsj

        _PARSER = ts.Parser(ts.Language(tsj.language()))
    return _PARSER


@dataclass(frozen=True)
class Options:
    wrapper_prefix: str
    wrapper_suffix: str
    local_prefix: str
    field_prefix: str
    method_prefix: str
    type_prefix: str
    copy_prefix: str
    param_prefix: str
    accessor_prefix: str
    keep_method_names: tuple[str, ...]
    tab_spaces: int
    space_divisor: int
    param_types: tuple[str, ...]
    accessor_types: tuple[str, ...]
    neutral_comment: str
    neutral_every: int
    # (transform, level, comment) for comment_claims_smell and comment_denies_smell
    suggestion_texts: tuple[tuple[str, str, str], ...]

    def suggestion(self, name: str, level: str) -> str:
        for t, lv, text in self.suggestion_texts:
            if t == name and lv == level:
                return text
        raise KeyError(f"no {name} comment for {level} snippets")

    @classmethod
    def from_config(cls, cfg: dict[str, Any], dataset: str | None = None) -> "Options":
        """Build options from the config, with the dataset's transform_options merged in."""
        wrapper, topts = cfg["wrapper"], cfg["transform_options"]
        if dataset is not None:
            from mlcq_graphs.config import deep_merge_dicts

            topts = deep_merge_dicts(topts, cfg["datasets"][dataset].get("transform_options") or {})
        names = topts["names"]
        texts = tuple((str(t), str(lv), str(text)) for t, by_level in topts["suggestions"].items()
                      for lv, text in by_level.items())
        neutral = topts["neutral_comments"]
        opts = cls(
            wrapper_prefix=str(wrapper["prefix"]),
            wrapper_suffix=str(wrapper["suffix"]),
            local_prefix=str(names["local"]),
            field_prefix=str(names["field"]),
            method_prefix=str(names["method"]),
            type_prefix=str(names["type"]),
            copy_prefix=str(names["copy_local"]),
            param_prefix=str(names["param"]),
            accessor_prefix=str(names["accessor"]),
            keep_method_names=tuple(str(n) for n in topts["keep_method_names"]),
            tab_spaces=int(topts["reindent"]["tab_spaces"]),
            space_divisor=int(topts["reindent"]["space_divisor"]),
            param_types=tuple(str(t) for t in topts["add_params"]["types"]),
            accessor_types=tuple(str(t) for t in topts["add_accessors"]["types"]),
            neutral_comment=str(neutral["text"]),
            neutral_every=int(neutral["every"]),
            suggestion_texts=texts,
        )
        for text in [opts.neutral_comment] + [t for _, _, t in texts]:
            # a line comment that cannot end early or turn into code (javac reads unicode escapes in comments)
            if not text.startswith("//") or any(c in text for c in "\r\n\x0b\x0c\x85  ") or "\\u" in text:
                raise ValueError(f"inserted comments must be single line comments without unicode escapes: {text!r}")
        if opts.neutral_every < 1:
            raise ValueError("neutral_comments.every must be at least 1")
        for t, lv, _ in texts:
            if t not in SUGGESTIONS or lv not in LEVELS:
                raise ValueError(f"unknown suggestion {t} for level {lv}")
        prefixes = [opts.local_prefix, opts.field_prefix, opts.method_prefix, opts.type_prefix, opts.copy_prefix,
                    opts.param_prefix]
        for a in prefixes:
            if not a or any(a != b and b.startswith(a) for b in prefixes):
                raise ValueError(f"name prefixes must be non-empty and none a prefix of another: {prefixes}")
        if len(set(opts.param_types)) != len(opts.param_types):
            raise ValueError("add_params.types must be distinct")
        return opts


@dataclass
class Outcome:
    code: str
    applied: bool
    n_edits: int
    status: str  # applied, no_change, not_applicable, verification_failed, error
    reason: str = ""
    info: dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> tuple[str, bool, int]:
        return self.code, self.applied, self.n_edits


@dataclass
class _Draft:
    code: str | None  # None: the transform does not apply to this snippet
    n_edits: int = 0
    reason: str = ""
    info: dict[str, Any] = field(default_factory=dict)
    check: Callable[["Snippet", "Snippet"], str] | None = None


class _Rejected(Exception):
    """The transform produced code it cannot vouch for."""


class Snippet:
    """A snippet parsed inside the wrapper, with node offsets in the wrapped source."""

    def __init__(self, code: str, opts: Options) -> None:
        self.code = code
        self.opts = opts
        body = code.rstrip("\n")
        self.tail = code[len(body):]
        prefix = opts.wrapper_prefix.encode("utf-8")
        self.body = body.encode("utf-8")
        self.src = prefix + self.body + opts.wrapper_suffix.encode("utf-8")
        self.lo = len(prefix)
        self.hi = self.lo + len(self.body)
        self.root = _parser().parse(self.src).root_node

    def inside(self, node: Any) -> bool:
        return node.start_byte >= self.lo and node.end_byte <= self.hi

    def text(self, node: Any) -> str:
        return self.src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def nodes(self) -> Iterator[Any]:
        """Pre-order nodes inside the snippet."""
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.end_byte < self.lo or node.start_byte > self.hi:
                continue
            if self.inside(node):
                yield node
            stack.extend(reversed(node.children))

    def edited(self, edits: list[tuple[int, int, bytes]]) -> str:
        """New snippet code after (start, end, replacement) edits in wrapped-source offsets."""
        out = bytearray(self.body)
        bound = None
        for start, end, repl in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
            if start < self.lo or end > self.hi or start > end:
                raise ValueError(f"edit [{start}, {end}) outside the snippet")
            if bound is not None and end > bound:
                raise ValueError("overlapping edits")
            out[start - self.lo:end - self.lo] = repl
            bound = start
        return out.decode("utf-8") + self.tail


def _walk(root: Any) -> Iterator[tuple[Any, list[Any]]]:
    """Pre-order (node, ancestors) pairs. The ancestor list is reused, do not keep it."""
    path: list[Any] = []
    stack: list[tuple[Any, bool]] = [(root, False)]
    while stack:
        node, leaving = stack.pop()
        if leaving:
            path.pop()
            continue
        yield node, path
        if node.child_count:
            path.append(node)
            stack.append((node, True))
            stack.extend((c, False) for c in reversed(node.children))


def _same(a: Any, b: Any) -> bool:
    return (a is not None and b is not None and a.start_byte == b.start_byte and a.end_byte == b.end_byte
            and a.type == b.type)


def _key(node: Any) -> tuple[int, int]:
    return node.start_byte, node.end_byte


# ---------------------------------------------------------------- checks


def _error_count(sn: Snippet) -> int:
    count = 0
    stack = [sn.root]
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            count += 1
        if node.has_error:
            stack.extend(node.children)
    return count


def _node_types(sn: Snippet) -> list[str]:
    return [n.type for n in sn.nodes()]


def _token_nodes(sn: Snippet, comments: bool) -> list[Any]:
    """Leaf tokens inside the snippet (string and character literals whole), in order."""
    out = []
    stack = [sn.root]
    while stack:
        node = stack.pop()
        if node.end_byte <= sn.lo or node.start_byte >= sn.hi:
            continue
        if node.type in COMMENT_TYPES:
            if comments and sn.inside(node):
                out.append(node)
            continue
        if node.child_count == 0 or node.type in LITERAL_TYPES:
            if sn.inside(node) and node.end_byte > node.start_byte:
                out.append(node)
            continue
        stack.extend(reversed(node.children))
    return out


def _tokens(sn: Snippet, comments: bool) -> list[tuple[str, str]]:
    """Token types and texts, with whitespace runs collapsed in comments."""
    return [(n.type, " ".join(sn.text(n).split()) if n.type in COMMENT_TYPES else sn.text(n))
            for n in _token_nodes(sn, comments)]


def _names(sn: Snippet) -> set[str]:
    return {sn.text(n) for n in sn.nodes() if n.type in ("identifier", "type_identifier")}


# ---------------------------------------------------------------- structure


def _wrapper_body(sn: Snippet) -> Any:
    for child in sn.root.children:
        if child.type == "class_declaration" and child.start_byte < sn.lo and child.end_byte > sn.hi:
            return child.child_by_field_name("body")
    return None


def _top_level(sn: Snippet) -> list[Any]:
    body = _wrapper_body(sn)
    if body is None:
        return []
    return [c for c in body.named_children if sn.inside(c) and c.type not in COMMENT_TYPES]


def _top_type(sn: Snippet) -> Any:
    """The type declaration of a class snippet, None for any other snippet."""
    top = _top_level(sn)
    return top[0] if len(top) == 1 and top[0].type in TYPE_DECLS else None


def _member_container(body: Any) -> list[Any]:
    """Named children of a type body, those of enum_body_declarations inlined (comments kept)."""
    out = []
    for c in body.named_children:
        if c.type == "enum_body_declarations":
            out.extend(c.named_children)
        else:
            out.append(c)
    return out


def _members(body: Any) -> list[Any]:
    return [c for c in _member_container(body) if c.type not in COMMENT_TYPES]


def _first_method(sn: Snippet) -> Any:
    """First method or constructor in pre-order, as n_params_first_method of the size features."""
    return next((n for n in sn.nodes() if n.type in METHOD_TYPES), None)


def _param_count(method: Any) -> int:
    params = method.child_by_field_name("parameters")
    return 0 if params is None else sum(1 for c in params.children if c.type in ("formal_parameter", "spread_parameter"))


def _line_indent(src: bytes, pos: int) -> bytes:
    start = src.rfind(b"\n", 0, pos) + 1
    end = start
    while end < len(src) and src[end:end + 1] in (b" ", b"\t"):
        end += 1
    return src[start:min(end, pos)]


def _protected_newlines(sn: Snippet, comments: bool = False) -> set[int]:
    """Snippet-relative offsets of the newlines inside text blocks (and inside comments if asked)."""
    out: set[int] = set()
    for n in sn.nodes():
        if (n.type == "string_literal" and sn.src.startswith(b'"""', n.start_byte)) or (
                comments and n.type in COMMENT_TYPES):
            seg = sn.src[n.start_byte:n.end_byte]
            out.update(n.start_byte - sn.lo + i for i, ch in enumerate(seg) if ch == 10)
    return out


def _eol(sn: Snippet) -> bytes:
    return b"\r\n" if b"\r\n" in sn.body else b"\n"


class _Fresh:
    """Names prefix0, prefix1, ... skipping the names of the snippet."""

    def __init__(self, prefix: str, used: set[str]) -> None:
        self.prefix, self.used, self.k = prefix, used, 0

    def next(self) -> str:
        while f"{self.prefix}{self.k}" in self.used:
            self.k += 1
        name = f"{self.prefix}{self.k}"
        self.k += 1
        return name


# ---------------------------------------------------------------- name resolution


@dataclass(eq=False)
class _Var:
    name: str
    ident: Any
    start: int  # scope
    end: int
    kind: str  # local, param, field, enum_constant, component
    owner: "_Class | None" = None
    type_name: str | None = None
    unit: int = -1
    # instanceof pattern variable: spans where it is definitely matched, None otherwise
    regions: tuple[tuple[int, int], ...] | None = None


@dataclass(eq=False)
class _Type:
    name: str
    ident: Any
    start: int
    end: int
    kind: str  # type or type_param
    cls: "_Class | None" = None


@dataclass(eq=False)
class _Class:
    body: Any
    name: str | None
    decl: Any
    anonymous: bool
    super_name: str | None
    fields: dict[str, list[_Var]] = field(default_factory=dict)
    methods: dict[str, list[Any]] = field(default_factory=dict)
    types: dict[str, _Type] = field(default_factory=dict)


def _type_name(sn: Snippet, node: Any) -> str | None:
    """Simple name of a declared type (last part, type arguments dropped)."""
    if node is None:
        return None
    t = node.type
    if t in ("type_identifier", "identifier", "integral_type", "floating_point_type", "boolean_type"):
        return sn.text(node)
    if t in ("generic_type", "annotated_type") and node.named_child_count:
        parts = [c for c in node.named_children if c.type not in ("type_arguments", "marker_annotation", "annotation")]
        return _type_name(sn, parts[0]) if parts else None
    if t == "scoped_type_identifier":
        parts = [c for c in node.named_children if c.type == "type_identifier"]
        return sn.text(parts[-1]) if parts else None
    return None


def _matched_regions(chain: list[Any]) -> tuple[tuple[int, int], ...]:
    """Spans where the pattern variables of the innermost instanceof in `chain` are definitely matched."""
    at = next((i for i in range(len(chain) - 1, -1, -1) if chain[i].type == "instanceof_expression"), None)
    if at is None:
        return ()
    regions: list[tuple[int, int]] = []
    cur, when_true = chain[at], True
    for anc in reversed(chain[:at]):
        t = anc.type
        if t == "parenthesized_expression":
            cur = anc
            continue
        if t == "unary_expression" and getattr(anc.child_by_field_name("operator"), "type", "") == "!":
            cur, when_true = anc, not when_true
            continue
        if t == "binary_expression":
            op = getattr(anc.child_by_field_name("operator"), "type", "")
            if (op, when_true) not in (("&&", True), ("||", False)):
                break
            right = anc.child_by_field_name("right")
            if _same(anc.child_by_field_name("left"), cur) and right is not None:
                regions.append(_key(right))
            cur = anc
            continue
        if t in ("ternary_expression", "if_statement") and _same(anc.child_by_field_name("condition"), cur):
            branch = anc.child_by_field_name("consequence" if when_true else "alternative")
            if branch is not None:
                regions.append(_key(branch))
        elif t in ("while_statement", "for_statement") and when_true and _same(
                anc.child_by_field_name("condition"), cur):
            body = anc.child_by_field_name("body")
            if body is not None:
                regions.append(_key(body))
        break
    return tuple(regions)


class _Index:
    """Declarations of the wrapped snippet with their lexical scopes."""

    def __init__(self, sn: Snippet) -> None:
        self.sn = sn
        self.vars: dict[str, list[_Var]] = defaultdict(list)
        self.types: dict[str, list[_Type]] = defaultdict(list)
        self.classes: list[_Class] = []
        self.by_body: dict[tuple[int, int], _Class] = {}
        self.decl_of: dict[int, _Var] = {}  # declaring identifier start byte
        self.type_decl: dict[int, _Type] = {}
        for node, path in _walk(sn.root):
            if path:
                self._visit(node, path)

    # -- building

    def _add(self, v: _Var) -> None:
        self.vars[v.name].append(v)
        self.decl_of[v.ident.start_byte] = v

    def _owner(self, path: list[Any]) -> _Class | None:
        parent = path[-1]
        if parent.type == "enum_body_declarations" and len(path) > 1:
            parent = path[-2]
        return self.by_body.get(_key(parent))

    @staticmethod
    def _unit(path: list[Any]) -> int:
        for i, anc in enumerate(path):
            if anc.type in UNIT_TYPES or (anc.type == "block" and i > 0 and path[i - 1].type in BODY_TYPES):
                return anc.start_byte
        return -1

    @staticmethod
    def _block_end(parent: Any, path: list[Any]) -> int:
        if parent.type == "switch_block_statement_group" and len(path) > 1:
            return path[-2].end_byte
        return parent.end_byte

    def _var(self, ident: Any, start: int, end: int, kind: str, path: list[Any], type_node: Any = None,
             owner: _Class | None = None, regions: tuple[tuple[int, int], ...] | None = None) -> None:
        if ident is None or ident.type != "identifier":
            return
        unit = self._unit(path) if kind in ("local", "param") else -1
        self._add(_Var(self.sn.text(ident), ident, start, end, kind, owner, _type_name(self.sn, type_node), unit,
                       regions))

    def _pattern_scope(self, ident: Any, decl: Any,
                       path: list[Any]) -> tuple[int, int, tuple[tuple[int, int], ...] | None]:
        """Lexical scope of a pattern variable and, outside a switch label, its matched regions."""
        for i in range(len(path) - 1, -1, -1):
            if path[i].type == "switch_label" and i > 0:
                return path[i - 1].start_byte, path[i - 1].end_byte, None
        end = next((anc.end_byte for anc in reversed(path) if anc.type in SCOPE_BLOCKS), self.sn.hi)
        return ident.start_byte, end, _matched_regions(path + [decl])

    def _visit(self, node: Any, path: list[Any]) -> None:
        sn, t, parent = self.sn, node.type, path[-1]
        if t in BODY_TYPES:
            self._add_class(node, path)
        elif t in TYPE_DECLS:
            name = node.child_by_field_name("name")
            if name is None:
                return
            if parent.type in BODY_TYPES or parent.type == "enum_body_declarations":
                body = parent if parent.type in BODY_TYPES else path[-2]
                start, end = body.start_byte, body.end_byte
            elif parent.type in ("block", "constructor_body", "switch_block_statement_group"):
                start, end = node.start_byte, self._block_end(parent, path)
            else:
                start, end = 0, len(sn.src)
            td = _Type(sn.text(name), name, start, end, "type")
            self.types[td.name].append(td)
            self.type_decl[name.start_byte] = td
            owner = self._owner(path)
            if owner is not None:
                owner.types.setdefault(td.name, td)
        elif t == "type_parameter":
            ident = next((c for c in node.named_children if c.type in ("type_identifier", "identifier")), None)
            if ident is not None and len(path) > 1:
                holder = path[-2]
                self.types[sn.text(ident)].append(
                    _Type(sn.text(ident), ident, holder.start_byte, holder.end_byte, "type_param"))
        elif t == "method_declaration":
            owner, name = self._owner(path), node.child_by_field_name("name")
            if owner is not None and name is not None:
                owner.methods.setdefault(sn.text(name), []).append(node)
        elif t in ("field_declaration", "constant_declaration"):
            owner = self._owner(path)
            if owner is None:
                return
            for d in node.children_by_field_name("declarator"):
                ident = d.child_by_field_name("name")
                self._var(ident, owner.body.start_byte, owner.body.end_byte, "field", path,
                          node.child_by_field_name("type"), owner)
                if ident is not None and ident.type == "identifier":
                    v = self.decl_of[ident.start_byte]
                    owner.fields.setdefault(v.name, []).append(v)
        elif t == "enum_constant":
            owner = self._owner(path)
            if owner is not None:
                self._var(node.child_by_field_name("name"), owner.body.start_byte, owner.body.end_byte,
                          "enum_constant", path, None, owner)
        elif t == "local_variable_declaration":
            end = self._block_end(parent, path)
            for d in node.children_by_field_name("declarator"):
                self._var(d.child_by_field_name("name"), d.start_byte, end, "local", path,
                          node.child_by_field_name("type"))
        elif t in ("formal_parameter", "spread_parameter"):
            if t == "formal_parameter":
                ident, type_node = node.child_by_field_name("name"), node.child_by_field_name("type")
            else:
                decl = next((c for c in node.named_children if c.type == "variable_declarator"), None)
                ident = decl.child_by_field_name("name") if decl is not None else None
                type_node = node.named_children[0] if node.named_child_count else None
            holder = path[-2] if len(path) > 1 else parent
            kind = "component" if holder.type == "record_declaration" else "param"
            self._var(ident, holder.start_byte, holder.end_byte, kind, path, type_node)
        elif t == "catch_formal_parameter":
            self._var(node.child_by_field_name("name"), parent.start_byte, parent.end_byte, "param", path)
        elif t == "resource":
            stmt = path[-2] if len(path) > 1 else parent
            body = stmt.child_by_field_name("body")
            self._var(node.child_by_field_name("name"), node.start_byte,
                      body.end_byte if body is not None else stmt.end_byte, "local", path,
                      node.child_by_field_name("type"))
        elif t == "enhanced_for_statement":
            body = node.child_by_field_name("body")
            if body is not None:
                self._var(node.child_by_field_name("name"), body.start_byte, body.end_byte, "local",
                          path + [node], node.child_by_field_name("type"))
        elif t == "lambda_expression":
            params = node.child_by_field_name("parameters")
            idents = []
            if params is not None and params.type == "identifier":
                idents = [params]
            elif params is not None and params.type == "inferred_parameters":
                idents = [c for c in params.named_children if c.type == "identifier"]
            for ident in idents:
                self._var(ident, node.start_byte, node.end_byte, "param", path + [node])
        elif t == "instanceof_expression":
            ident = node.child_by_field_name("name")
            if ident is not None:
                start, end, regions = self._pattern_scope(ident, node, path)
                self._var(ident, start, end, "local", path, node.child_by_field_name("right"), regions=regions)
        elif t in ("type_pattern", "record_pattern_component"):
            named = node.named_children
            if named and named[-1].type == "identifier":
                start, end, regions = self._pattern_scope(named[-1], node, path)
                self._var(named[-1], start, end, "local", path, named[0] if len(named) > 1 else None,
                          regions=regions)

    def _add_class(self, body: Any, path: list[Any]) -> None:
        sn, owner = self.sn, path[-1]
        if owner.type in TYPE_DECLS:
            name = owner.child_by_field_name("name")
            sup = None
            superclass = owner.child_by_field_name("superclass")
            if superclass is not None and superclass.named_child_count:
                sup = _type_name(sn, superclass.named_children[0])
            cls = _Class(body, sn.text(name) if name is not None else None, owner, False, sup)
            if name is not None and name.start_byte in self.type_decl:
                self.type_decl[name.start_byte].cls = cls
        elif owner.type == "object_creation_expression":
            cls = _Class(body, None, owner, True, _type_name(sn, owner.child_by_field_name("type")))
        elif owner.type == "enum_constant":
            enum = path[-3] if len(path) > 2 else None
            enum_name = enum.child_by_field_name("name") if enum is not None and enum.type == "enum_declaration" else None
            cls = _Class(body, None, owner, True, sn.text(enum_name) if enum_name is not None else None)
        else:
            return
        self.classes.append(cls)
        self.by_body[_key(body)] = cls

    # -- lookups

    @staticmethod
    def _innermost(decls: list[Any], pos: int) -> Any:
        best = None
        for d in decls:
            if d.start <= pos < d.end and (best is None or d.end - d.start < best.end - best.start):
                best = d
        return best

    def classes_at(self, pos: int) -> list[_Class]:
        found = [c for c in self.classes if c.body.start_byte <= pos < c.body.end_byte]
        found.sort(key=lambda c: c.body.end_byte - c.body.start_byte)
        return found

    def type_at(self, name: str, pos: int) -> _Type | None:
        return self._innermost(self.types.get(name, []), pos)

    def class_named(self, name: str | None, pos: int) -> _Class | None:
        if not name:
            return None
        td = self.type_at(name, pos)
        return td.cls if td is not None and td.kind == "type" else None

    def superclass(self, cls: _Class) -> _Class | None:
        return self.class_named(cls.super_name, cls.decl.start_byte) if cls.super_name else None

    def chain(self, cls: _Class | None) -> Iterator[_Class]:
        seen: set[int] = set()
        while cls is not None and id(cls) not in seen:
            seen.add(id(cls))
            yield cls
            cls = self.superclass(cls)

    def find_field(self, cls: _Class | None, name: str) -> _Var | None:
        for c in self.chain(cls):
            if name in c.fields:
                return c.fields[name][0]
        return None

    def method_owner(self, cls: _Class | None, name: str) -> _Class | None:
        for c in self.chain(cls):
            if name in c.methods:
                return c
        return None

    def resolve_var(self, name: str, pos: int) -> _Var | None:
        """Innermost variable named `name` at pos, or a field of a nested class that shadows it."""
        v = self._innermost(self.vars.get(name, []), pos)
        span = v.end - v.start if v is not None else None
        for cls in self.classes_at(pos):
            if span is not None and cls.body.end_byte - cls.body.start_byte >= span:
                break
            f = self.find_field(cls, name)
            if f is not None:
                return f
        return v

    def expr_type(self, node: Any) -> str | None:
        """Declared simple type of a variable, field or cast expression, None when unknown."""
        if node is None:
            return None
        sn, t = self.sn, node.type
        if t == "parenthesized_expression":
            return self.expr_type(node.named_children[0]) if node.named_child_count else None
        if t == "cast_expression":
            return _type_name(sn, node.child_by_field_name("type"))
        if t == "identifier":
            v = self.resolve_var(sn.text(node), node.start_byte)
            return v.type_name if v is not None else None
        if t == "field_access":
            fld = node.child_by_field_name("field")
            cls = self.object_class(node.child_by_field_name("object"))
            f = self.find_field(cls, sn.text(fld)) if cls is not None and fld is not None else None
            return f.type_name if f is not None else None
        return None

    def selector_kind(self, path: list[Any]) -> str:
        """Kind of the switch around a case label: "constant", "enum" or "unknown"."""
        sw = next((a for a in reversed(path) if a.type in ("switch_expression", "switch_statement")), None)
        if sw is None:
            return "unknown"
        selector = sw.child_by_field_name("condition")
        while selector is not None and selector.type == "parenthesized_expression" and selector.named_child_count:
            selector = selector.named_children[0]
        if selector is not None and selector.type in CONSTANT_EXPRESSIONS:
            return "constant"
        block = sw.child_by_field_name("body")
        for item in block.named_children if block is not None else []:
            for label in (c for c in item.named_children if c.type == "switch_label"):
                if any(c.type in CONSTANT_EXPRESSIONS or c.type == "parenthesized_expression"
                       for c in label.named_children):
                    return "constant"
        name = self.expr_type(selector)
        if name is None or name == "var":
            return "unknown"
        if name in CONSTANT_SELECTORS:
            return "constant"
        td = self.type_at(name, sw.start_byte)
        return "unknown" if td is not None and td.kind == "type_param" else "enum"

    def resolve_ref(self, node: Any, path: list[Any]) -> tuple[_Var | None, bool]:
        """The variable an unqualified name refers to and whether that binding is certain.

        A bare case label names an enum constant unless its switch is a constant switch."""
        v = self.resolve_var(self.sn.text(node), node.start_byte)
        if v is None or path[-1].type != "switch_label" or v.kind == "enum_constant":
            return v, True
        kind = self.selector_kind(path)
        if kind == "constant":
            return v, True
        if kind == "enum" or v.kind not in ("local", "field") or v.type_name not in CONSTANT_TYPES:
            return None, True
        return v, False

    def type_class(self, node: Any) -> _Class | None:
        t = node.type
        if t in ("type_identifier", "identifier"):
            return self.class_named(self.sn.text(node), node.start_byte)
        if t == "generic_type" and node.named_child_count:
            return self.type_class(node.named_children[0])
        if t == "scoped_type_identifier":
            parts = [c for c in node.named_children if c.type in ("type_identifier", "scoped_type_identifier",
                                                                   "generic_type")]
            if len(parts) < 2:
                return None
            outer = self.type_class(parts[-2])
            name = self.sn.text(parts[-1])
            if outer is not None:
                for c in self.chain(outer):
                    if name in c.types:
                        return c.types[name].cls
        return None

    def object_class(self, node: Any) -> _Class | None:
        """Class of the snippet that the receiver expression `node` designates or is an instance of."""
        if node is None:
            return None
        sn, t, pos = self.sn, node.type, node.start_byte
        if t == "this":
            inner = self.classes_at(pos)
            return inner[0] if inner else None
        if t == "super":
            inner = self.classes_at(pos)
            return self.superclass(inner[0]) if inner else None
        if t == "parenthesized_expression":
            return self.object_class(node.named_children[0]) if node.named_child_count else None
        if t == "cast_expression":
            return self.class_named(_type_name(sn, node.child_by_field_name("type")), pos)
        if t == "object_creation_expression":
            body = next((c for c in node.children if c.type == "class_body"), None)
            if body is not None:
                return self.by_body.get(_key(body))
            return self.class_named(_type_name(sn, node.child_by_field_name("type")), pos)
        if t == "identifier":
            name = sn.text(node)
            v = self.resolve_var(name, pos)
            if v is not None:
                return self.class_named(v.type_name, v.ident.start_byte) if v.type_name else None
            return self.class_named(name, pos)
        if t in ("type_identifier", "generic_type", "scoped_type_identifier"):
            return self.type_class(node)
        if t == "field_access":
            obj, fld = node.child_by_field_name("object"), node.child_by_field_name("field")
            if obj is None or fld is None:
                return None
            if fld.type == "this":
                return self.class_named(sn.text(obj), pos) if obj.type == "identifier" else self.type_class(obj)
            base = self.object_class(obj)
            if base is None:
                return None
            name = sn.text(fld)
            f = self.find_field(base, name)
            if f is not None:
                return self.class_named(f.type_name, f.ident.start_byte) if f.type_name else None
            for c in self.chain(base):
                if name in c.types:
                    return c.types[name].cls
        return None


def _role(node: Any, parent: Any) -> str:
    """What an identifier names, from its position under its parent."""
    pt = parent.type
    if pt == "method_invocation":
        if _same(parent.child_by_field_name("object"), node):
            return "ref"
        return "method_name" if _same(parent.child_by_field_name("name"), node) else "other"
    if pt == "field_access":
        return "ref" if _same(parent.child_by_field_name("object"), node) else "field"
    if pt == "method_reference":
        return "ref" if _same(parent.children[0], node) else "method_ref_name"
    is_name = _same(parent.child_by_field_name("name"), node)
    if pt == "method_declaration":
        return "decl_method" if is_name else "other"
    if pt in TYPE_DECLS:
        return "decl_type" if is_name else "other"
    if pt in ("constructor_declaration", "compact_constructor_declaration"):
        return "decl_ctor" if is_name else "other"
    if pt in ("marker_annotation", "annotation"):
        return "annotation" if is_name else "other"
    if pt == "element_value_pair" and _same(parent.child_by_field_name("key"), node):
        return "other"
    if pt in NOT_NAMES:
        return "other"
    return "ref"


def _bindings(sn: Snippet) -> list[int]:
    """Pre-order rank of the declaring identifier of every unqualified variable name, -1 when undeclared."""
    idx = _Index(sn)
    rank: dict[int, int] = {}
    refs: list[_Var | None] = []
    for node, path in _walk(sn.root):
        if node.type != "identifier" or not path or not sn.inside(node):
            continue
        rank[node.start_byte] = len(rank)
        if node.start_byte not in idx.decl_of and _role(node, path[-1]) == "ref":
            refs.append(idx.resolve_ref(node, path)[0])
    return [rank.get(v.ident.start_byte, -1) if v is not None else -1 for v in refs]


def _keep_reason(sn: Snippet, method: Any, cls: _Class, name: str) -> str:
    if name in sn.opts.keep_method_names:
        return "name"
    modifiers = next((c for c in method.children if c.type == "modifiers"), None)
    if modifiers is not None:
        for a in modifiers.children:
            if a.type in ("marker_annotation", "annotation"):
                ann = a.child_by_field_name("name")
                if ann is not None and sn.text(ann).split(".")[-1] == "Override":
                    return "override"
    if cls.anonymous:
        return "anonymous"
    return ""


def _external_supertype(idx: _Index, cls: _Class) -> bool:
    """Whether the class extends or implements a type not declared in the snippet."""
    if cls.anonymous:
        return True
    types: list[Any] = []
    for child in cls.decl.children:
        if child.type == "superclass":
            types += child.named_children[:1]
        elif child.type in ("super_interfaces", "extends_interfaces"):
            for tl in child.named_children:
                types += tl.named_children if tl.type == "type_list" else [tl]
    return any(idx.type_class(t) is None for t in types)


def _rename_edits(sn: Snippet, idx: _Index, *, locals_: bool, members: bool, types: bool, local_prefix: str,
                  local_filter: Callable[[_Var], bool] | None = None) -> tuple[list[tuple[int, int, bytes]], dict]:
    opts = sn.opts
    used = _names(sn)
    info: dict[str, Any] = {}

    local_new: dict[int, str] = {}
    if locals_:
        decls = sorted((v for vs in idx.vars.values() for v in vs
                        if v.kind in ("local", "param") and sn.inside(v.ident)
                        and (local_filter is None or local_filter(v))), key=lambda v: v.ident.start_byte)
        counters: dict[int, _Fresh] = {}
        for v in decls:
            gen = counters.setdefault(v.unit, _Fresh(local_prefix, used))
            local_new[id(v)] = gen.next()
        info["locals"] = len(local_new)

    field_new: dict[str, str] = {}
    method_new: dict[str, str] = {}
    if members:
        gen = _Fresh(opts.field_prefix, used)
        for v in sorted((v for vs in idx.vars.values() for v in vs if v.kind == "field" and sn.inside(v.ident)),
                        key=lambda v: v.ident.start_byte):
            if v.name not in field_new:
                field_new[v.name] = gen.next()
        by_name: dict[str, list[tuple[Any, _Class]]] = defaultdict(list)
        for cls in idx.classes:
            for name, nodes in cls.methods.items():
                by_name[name].extend((m, cls) for m in nodes if sn.inside(m))
        kept: Counter = Counter()
        risk: Counter = Counter()
        gen = _Fresh(opts.method_prefix, used)
        for name in sorted((n for n in by_name if by_name[n]), key=lambda n: min(m.start_byte for m, _ in by_name[n])):
            reasons = [_keep_reason(sn, m, cls, name) for m, cls in by_name[name]]
            if any(reasons):
                kept.update(r or "same_name" for r in reasons)
                continue
            method_new[name] = gen.next()
            for _, cls in by_name[name]:
                # renamed declarations that may override a method outside the snippet
                if not sn.inside(cls.body):
                    risk["renamed_class_unknown"] += 1
                elif _external_supertype(idx, cls):
                    risk["renamed_in_subtype"] += 1
        info.update(fields=len(field_new), methods=len(method_new),
                    renamed_methods=sum(len(by_name[n]) for n in method_new), methods_kept=sum(kept.values()),
                    **{f"kept_{r}": c for r, c in sorted(kept.items())}, **dict(risk))

    type_new: dict[str, str] = {}
    if types:
        type_params = {t.name for ts in idx.types.values() for t in ts if t.kind == "type_param"}
        gen = _Fresh(opts.type_prefix, used)
        for td in sorted((t for ts in idx.types.values() for t in ts if t.kind == "type" and sn.inside(t.ident)),
                         key=lambda t: t.ident.start_byte):
            if td.name not in type_params and td.name not in type_new:
                type_new[td.name] = gen.next()
        info["types"] = len(type_new)

    def snippet_type(name: str, pos: int) -> bool:
        td = idx.type_at(name, pos)
        return td is not None and td.kind == "type" and sn.inside(td.ident)

    edits: dict[int, tuple[int, str]] = {}
    for node, path in _walk(sn.root):
        if node.type not in ("identifier", "type_identifier") or not sn.inside(node) or not path:
            continue
        parent, name, pos = path[-1], sn.text(node), node.start_byte
        new = None
        if node.type == "type_identifier":
            if name not in type_new or parent.type == "type_parameter":
                continue
            if parent.type == "scoped_type_identifier" and not _same(parent.named_children[0], node):
                kids = parent.named_children
                k = next(i for i, c in enumerate(kids) if _same(c, node))
                outer = idx.type_class(kids[k - 1])
                if any(name in c.types and sn.inside(c.types[name].ident) for c in idx.chain(outer)):
                    new = type_new[name]
            elif snippet_type(name, pos):
                new = type_new[name]
        elif pos in idx.decl_of:
            v = idx.decl_of[pos]
            if id(v) in local_new:
                new = local_new[id(v)]
            elif v.kind == "field" and name in field_new:
                new = field_new[name]
        else:
            role = _role(node, parent)
            if role == "ref":
                v, sure = idx.resolve_ref(node, path)
                if v is not None:
                    if v.regions is not None and not any(s <= pos < e for s, e in v.regions) and (
                            id(v) in local_new or name in field_new):
                        # a use of a pattern variable's name where flow scoping may bind it elsewhere
                        raise _Rejected("pattern_scope")
                    if id(v) in local_new:
                        new = local_new[id(v)]
                    elif v.kind == "field" and name in field_new and sn.inside(v.ident):
                        new = field_new[name]
                elif name in type_new and parent.type != "switch_label" and snippet_type(name, pos):
                    new = type_new[name]
                if new is not None and not sure:
                    # a case label that names either this constant or an enum constant of an unknown selector type
                    raise _Rejected("case_label")
            elif role == "field":
                cls = idx.object_class(parent.child_by_field_name("object"))
                if cls is not None and name in field_new:
                    f = idx.find_field(cls, name)
                    if f is not None and sn.inside(f.ident):
                        new = field_new[name]
                elif cls is not None and name in type_new:
                    if any(name in c.types and sn.inside(c.types[name].ident) for c in idx.chain(cls)):
                        new = type_new[name]
            elif role in ("method_name", "method_ref_name") and name in method_new:
                if role == "method_name" and parent.child_by_field_name("object") is None:
                    target = next((o for o in (idx.method_owner(c, name) for c in idx.classes_at(pos))
                                   if o is not None), None)
                else:
                    obj = parent.child_by_field_name("object") if role == "method_name" else parent.children[0]
                    target = idx.method_owner(idx.object_class(obj), name)
                if target is not None and any(sn.inside(m) for m in target.methods[name]):
                    new = method_new[name]
            elif role == "decl_method" and name in method_new:
                new = method_new[name]
            elif role in ("decl_type", "decl_ctor") and name in type_new:
                new = type_new[name]
            elif role == "annotation" and name in type_new and snippet_type(name, pos):
                new = type_new[name]
        if new is not None and new != name:
            edits[pos] = (node.end_byte, new)
    return [(s, e, n.encode("utf-8")) for s, (e, n) in edits.items()], info


# ---------------------------------------------------------------- transforms


def _draft_rename(sn: Snippet, seed: int, *, locals_: bool, members: bool, types: bool) -> _Draft:
    edits, info = _rename_edits(sn, _Index(sn), locals_=locals_, members=members, types=types,
                                local_prefix=sn.opts.local_prefix)
    return _Draft(sn.edited(edits) if edits else sn.code, len(edits), "nothing_to_rename" if not edits else "", info)


def _kept_line(body: bytes, gone: bytearray, start: int, end: int) -> bytes:
    """Bytes of one line without the comment bytes, keeping the tokens around it apart."""
    runs: list[list[Any]] = []
    i = start
    while i < end:
        j = i
        while j < end and gone[j] == gone[i]:
            j += 1
        runs.append([bool(gone[i]), body[i:j]])
        i = j
    out = b""
    for k, (deleted, chunk) in enumerate(runs):
        if not deleted:
            out += chunk
            continue
        if k + 1 >= len(runs):
            out = out.rstrip(b" \t")
            continue
        nxt = runs[k + 1][1]
        if not out.strip():
            runs[k + 1][1] = nxt.lstrip(b" \t")
        elif out[-1:] not in (b" ", b"\t") and nxt[:1] not in (b" ", b"\t", b""):
            out += b" "
    return out


def _strip_comments(sn: Snippet) -> tuple[str, int]:
    body = sn.body
    spans = sorted((n.start_byte - sn.lo, n.end_byte - sn.lo) for n in sn.nodes() if n.type in COMMENT_TYPES)
    if not spans:
        return sn.code, 0
    gone = bytearray(len(body))
    for s, e in spans:
        gone[s:e] = b"\x01" * (e - s)
    lines, pos = [], 0
    for line in body.split(b"\n"):
        start, end = pos, pos + len(line)
        pos = end + 1
        if not any(gone[start:end]) and not (end < len(body) and gone[end]):
            lines.append(line)
            continue
        kept = _kept_line(body, gone, start, end)
        if kept.strip():
            lines.append(kept)
    return b"\n".join(lines).decode("utf-8") + sn.tail, len(spans)


def _draft_strip(sn: Snippet, seed: int) -> _Draft:
    code, n = _strip_comments(sn)
    return _Draft(code, n, "no_comments" if not n else "")


def _lines(sn: Snippet) -> list[tuple[int, bytes]]:
    out, pos = [], 0
    for line in sn.body.split(b"\n"):
        out.append((pos, line))
        pos += len(line) + 1
    return out


def _draft_blank_inflate(sn: Snippet, seed: int) -> _Draft:
    prot = _protected_newlines(sn)
    lines = _lines(sn)
    out, n = [], 0
    for i, (start, line) in enumerate(lines):
        out.append(line)
        if i + 1 < len(lines) and start + len(line) not in prot:
            out.append(b"")
            n += 1
    return _Draft(b"\n".join(out).decode("utf-8") + sn.tail, n, "single_line" if not n else "")


def _draft_blank_remove(sn: Snippet, seed: int) -> _Draft:
    prot = _protected_newlines(sn)
    out, n = [], 0
    for start, line in _lines(sn):
        if not line.strip() and not (start > 0 and start - 1 in prot):
            n += 1
            continue
        out.append(line)
    return _Draft(b"\n".join(out).decode("utf-8") + sn.tail, n, "no_blank_lines" if not n else "")


def _draft_reindent(sn: Snippet, seed: int) -> _Draft:
    opts = sn.opts
    prot = _protected_newlines(sn)
    lines = _lines(sn)

    def lead(line: bytes) -> bytes:
        return line[:len(line) - len(line.lstrip(b" \t"))]

    usable = [i for i, (start, _) in enumerate(lines) if not (start > 0 and start - 1 in prot)]
    tabs = any(b"\t" in lead(lines[i][1]) for i in usable)
    out = [line for _, line in lines]
    n = 0
    for i in usable:
        ws = lead(out[i])
        if not ws:
            continue
        new = ws.replace(b"\t", b" " * opts.tab_spaces) if tabs else b" " * -(-len(ws) // opts.space_divisor)
        if new != ws:
            out[i] = new + out[i][len(ws):]
            n += 1
    return _Draft(b"\n".join(out).decode("utf-8") + sn.tail, n, "no_indentation" if not n else "",
                  {"mode": "tabs" if tabs else "spaces"})


def _comments_added(added: list[str]) -> Callable[[Snippet, Snippet], str]:
    """Check: the same non-comment tokens and exactly the comments `added` on top of those of the base."""
    want = Counter(" ".join(t.split()) for t in added)

    def check(base: Snippet, var: Snippet) -> str:
        if _tokens(var, False) != _tokens(base, False):
            return "tokens_changed"
        before = Counter(t for k, t in _tokens(base, True) if k in COMMENT_TYPES)
        after = Counter(t for k, t in _tokens(var, True) if k in COMMENT_TYPES)
        return "" if after - before == want and not before - after else "comments_changed"

    return check


def _draft_neutral_comments(sn: Snippet, seed: int) -> _Draft:
    opts = sn.opts
    prot = _protected_newlines(sn, comments=True)
    lines = _lines(sn)
    text = opts.neutral_comment.encode("utf-8")
    cr = b"\r" if _eol(sn) == b"\r\n" else b""
    out, n = [], 0
    for i, (start, line) in enumerate(lines):
        out.append(line)
        if (i + 1) % opts.neutral_every or i + 1 >= len(lines) or start + len(line) in prot:
            continue
        # the deeper indentation of this line and the next non-blank one
        nxt = next((ln for _, ln in lines[i + 1:] if ln.strip()), line)
        indent = max((ln[:len(ln) - len(ln.lstrip(b" \t"))] for ln in (line, nxt)), key=len)
        out.append(indent + text + cr)
        n += 1
    if not n:
        return _Draft(None, reason="too_few_lines")
    return _Draft(b"\n".join(out).decode("utf-8") + sn.tail, n, check=_comments_added([opts.neutral_comment] * n))


def _draft_suggestion(sn: Snippet, seed: int, *, name: str) -> _Draft:
    top = _top_level(sn)
    if not top:
        return _Draft(None, reason="no_declaration")
    level = "class" if _top_type(sn) is not None else "method"
    comment = sn.opts.suggestion(name, level)
    text, eol = comment.encode("utf-8"), _eol(sn)
    node = top[0]
    line_start = sn.src.rfind(b"\n", 0, node.start_byte) + 1
    indent = _line_indent(sn.src, node.start_byte)
    if line_start >= sn.lo and not sn.src[line_start:node.start_byte].strip():
        edit = (line_start, line_start, indent + text + eol)
    else:
        edit = (node.start_byte, node.start_byte, text + eol + indent)
    return _Draft(sn.edited([edit]), 1, info={"level": level}, check=_comments_added([comment]))


def _member_breaks(sn: Snippet) -> set[int]:
    """Start bytes of the tokens that begin a line in the compact form."""
    top = _top_type(sn)
    if top is None:
        return {n.start_byte for n in _top_level(sn)[1:]}
    body = top.child_by_field_name("body")
    if body is None:
        return set()
    members = _members(body)
    constants = [m for m in members if m.type == "enum_constant"]
    breaks = {m.start_byte for m in members if m.type != "enum_constant"}
    if constants:
        breaks.add(constants[0].start_byte)
    if body.children and body.children[-1].type == "}" and not body.children[-1].is_missing:
        breaks.add(body.children[-1].start_byte)
    return breaks


def _draft_compact(sn: Snippet, seed: int) -> _Draft:
    code, n_comments = _strip_comments(sn)
    s2 = Snippet(code, sn.opts)
    if any(n.type == "string_literal" and s2.src.startswith(b'"""', n.start_byte) for n in s2.nodes()):
        return _Draft(None, reason="text_block")
    breaks = _member_breaks(s2)
    out: list[bytes] = []
    prev = None
    changed = n_comments
    for tok in _token_nodes(s2, comments=False):
        if prev is not None:
            gap = s2.src[prev:tok.start_byte]
            if gap.strip():
                raise _Rejected("text_between_tokens")
            sep = b"\n" if tok.start_byte in breaks else (b" " if gap else b"")
            changed += sep != gap
            out.append(sep)
        out.append(s2.src[tok.start_byte:tok.end_byte])
        prev = tok.end_byte
    if not out:
        return _Draft(None, reason="no_tokens")
    return _Draft(b"".join(out).decode("utf-8") + sn.tail, changed)


def _attached_span(kids: list[Any], i: int) -> tuple[int, int]:
    """Byte span of member i with its leading comments and a comment that ends its last line."""
    node = kids[i]
    start, end = node.start_byte, node.end_byte
    if i + 1 < len(kids) and kids[i + 1].type in COMMENT_TYPES and kids[i + 1].start_point[0] == node.end_point[0]:
        end = kids[i + 1].end_byte
    j = i - 1
    while j >= 0 and kids[j].type in COMMENT_TYPES:
        k = j - 1
        while k >= 0 and kids[k].type in COMMENT_TYPES:
            k -= 1
        if k >= 0 and kids[j].start_point[0] == kids[k].end_point[0]:
            break
        start = kids[j].start_byte
        j -= 1
    return start, end


def _draft_shuffle(sn: Snippet, seed: int) -> _Draft:
    top = _top_type(sn)
    body = top.child_by_field_name("body") if top is not None else None
    if body is None:
        return _Draft(None, reason="not_a_class_snippet")
    container = next((c for c in body.named_children if c.type == "enum_body_declarations"), body)
    kids = list(container.named_children)
    methods = [i for i, c in enumerate(kids) if c.type == "method_declaration"]
    if len(methods) < 2:
        return _Draft(None, reason="fewer_than_two_methods")
    spans = [_attached_span(kids, i) for i in methods]
    order = list(range(len(spans)))
    rng = random.Random(seed)
    while order == sorted(order):
        rng.shuffle(order)
    edits = [(spans[i][0], spans[i][1], sn.src[spans[order[i]][0]:spans[order[i]][1]])
             for i in range(len(spans)) if order[i] != i]
    return _Draft(sn.edited(edits), len(edits), info={"methods": len(spans)})


def _member_multiset(sn: Snippet) -> Counter | None:
    top = _top_type(sn)
    body = top.child_by_field_name("body") if top is not None else None
    if body is None:
        return None
    return Counter((m.type, sn.text(m)) for m in _members(body))


def _draft_add_params(sn: Snippet, seed: int) -> _Draft:
    opts = sn.opts
    method = _first_method(sn)
    params = method.child_by_field_name("parameters") if method is not None else None
    if params is None:
        return _Draft(None, reason="no_parameter_list")
    gen = _Fresh(opts.param_prefix, _names(sn))
    decls = ", ".join(f"{t} {gen.next()}" for t in opts.param_types).encode("utf-8")
    existing = [c for c in params.named_children if c.type in ("formal_parameter", "spread_parameter",
                                                                "receiver_parameter")]
    spread = next((c for c in existing if c.type == "spread_parameter"), None)
    if spread is not None:
        edit = (spread.start_byte, spread.start_byte, decls + b", ")
    elif existing:
        edit = (existing[-1].end_byte, existing[-1].end_byte, b", " + decls)
    elif params.children and params.children[0].type == "(":
        edit = (params.children[0].end_byte, params.children[0].end_byte, decls)
    else:
        return _Draft(None, reason="no_parameter_list")
    expected = _param_count(method) + len(opts.param_types)

    def check(base: Snippet, var: Snippet) -> str:
        m = _first_method(var)
        return "" if m is not None and _param_count(m) == expected else "parameter_count"

    return _Draft(sn.edited([edit]), len(opts.param_types), check=check)


def _draft_duplicate_body(sn: Snippet, seed: int) -> _Draft:
    opts = sn.opts
    method = _first_method(sn)
    body = method.child_by_field_name("body") if method is not None else None
    if body is None:
        return _Draft(None, reason="no_method_body")
    stmts = [c for c in body.named_children if c.type not in COMMENT_TYPES]
    copied = [c for c in stmts if c.type != "explicit_constructor_invocation"]
    if not copied:
        return _Draft(None, reason="empty_body")
    copy = sn.src[copied[0].start_byte:copied[-1].end_byte]
    last = stmts[-1]
    indent = _line_indent(sn.src, last.start_byte)
    block = b"if (false) {\n" + indent + copy + b"\n" + indent + b"}"
    if last.type in ("return_statement", "throw_statement"):
        at, text, if_start = last.start_byte, block + b"\n" + indent, last.start_byte
    else:
        at, text = last.end_byte, b"\n" + indent + block
        if_start = at + 1 + len(indent)
    s1 = Snippet(sn.edited([(at, at, text)]), opts)
    if_node = next((n for n in s1.nodes() if n.type == "if_statement" and n.start_byte == if_start), None)
    inner = if_node.child_by_field_name("consequence") if if_node is not None else None
    if inner is None:
        raise _Rejected("copy_not_parsed")
    lo, hi = inner.start_byte, inner.end_byte
    edits, _ = _rename_edits(s1, _Index(s1), locals_=True, members=False, types=False,
                             local_prefix=opts.copy_prefix, local_filter=lambda v: lo <= v.ident.start_byte < hi)
    code = s1.edited(edits) if edits else s1.code
    # the base nodes, those of the copied span once more and those of the if (false) { } wrapper
    copy_lo, copy_hi = copied[0].start_byte, copied[-1].end_byte
    expected = (Counter(_node_types(sn)) + Counter(n.type for n in sn.nodes()
                                                    if copy_lo <= n.start_byte and n.end_byte <= copy_hi)
                + _if_false_types(opts))

    def check(base: Snippet, var: Snippet) -> str:
        return "" if Counter(_node_types(var)) == expected else "node_counts"

    return _Draft(code, len(copied), info={"renamed_in_copy": len(edits)}, check=check)


def _if_false_types(opts: Options) -> Counter:
    """Node types of an empty `if (false) { }` statement."""
    sn = Snippet("    void f() {\n        if (false) {\n        }\n    }\n", opts)
    node = next(n for n in sn.nodes() if n.type == "if_statement")
    out: Counter = Counter()
    stack = [node]
    while stack:
        n = stack.pop()
        out[n.type] += 1
        stack.extend(n.children)
    return out


def _top_counts(sn: Snippet) -> tuple[int, int] | None:
    top = _top_type(sn)
    body = top.child_by_field_name("body") if top is not None else None
    if body is None:
        return None
    members = _members(body)
    fields = sum(sum(1 for c in m.children if c.type == "variable_declarator") for m in members
                 if m.type == "field_declaration")
    return fields, sum(1 for m in members if m.type == "method_declaration")


def _draft_add_accessors(sn: Snippet, seed: int) -> _Draft:
    opts = sn.opts
    top = _top_type(sn)
    if top is None or top.type != "class_declaration":
        return _Draft(None, reason="not_a_class")
    body = top.child_by_field_name("body")
    closing = body.children[-1] if body is not None and body.children else None
    if closing is None or closing.type != "}" or closing.is_missing:
        return _Draft(None, reason="no_closing_brace")
    used = _names(sn)
    names: list[str] = []
    k = 0
    while len(names) < len(opts.accessor_types):
        n = f"{opts.accessor_prefix}{k}"
        cap = n[:1].upper() + n[1:]
        if not {n, f"get{cap}", f"set{cap}"} & used:
            names.append(n)
        k += 1
    class_indent = _line_indent(sn.src, top.start_byte).decode("utf-8")
    members = _members(body)
    first = _line_indent(sn.src, members[0].start_byte).decode("utf-8") if members else ""
    if len(first) > len(class_indent) and first.startswith(class_indent):
        mi, unit = first, first[len(class_indent):]
    else:
        unit = "    "
        mi = class_indent + unit
    fields = [f"{mi}private {t} {n};" for t, n in zip(opts.accessor_types, names)]
    methods = []
    for t, n in zip(opts.accessor_types, names):
        cap = n[:1].upper() + n[1:]
        methods.append(f"{mi}public {t} get{cap}() {{\n{mi}{unit}return {n};\n{mi}}}")
        methods.append(f"{mi}public void set{cap}({t} {n}) {{\n{mi}{unit}this.{n} = {n};\n{mi}}}")
    block = ("\n".join(fields) + "\n\n" + "\n\n".join(methods) + "\n").encode("utf-8")
    line_start = sn.src.rfind(b"\n", 0, closing.start_byte) + 1
    if line_start >= sn.lo and not sn.src[line_start:closing.start_byte].strip():
        edit = (line_start, line_start, b"\n" + block)
    else:
        edit = (closing.start_byte, closing.start_byte, b"\n" + block + class_indent.encode("utf-8"))
    base = _top_counts(sn)
    expected = (base[0] + len(names), base[1] + 2 * len(names)) if base is not None else None

    def check(base_sn: Snippet, var: Snippet) -> str:
        return "" if expected is not None and _top_counts(var) == expected else "member_counts"

    return _Draft(sn.edited([edit]), 3 * len(names), check=check)


_DRAFTS: dict[str, Callable[[Snippet, int], _Draft]] = {
    "rename_locals": partial(_draft_rename, locals_=True, members=False, types=False),
    "rename_members": partial(_draft_rename, locals_=False, members=True, types=False),
    "rename_all": partial(_draft_rename, locals_=True, members=True, types=True),
    "strip_comments": _draft_strip,
    "blank_inflate": _draft_blank_inflate,
    "blank_remove": _draft_blank_remove,
    "reindent": _draft_reindent,
    "compact": _draft_compact,
    "shuffle_methods": _draft_shuffle,
    "neutral_comments": _draft_neutral_comments,
    "comment_claims_smell": partial(_draft_suggestion, name="comment_claims_smell"),
    "comment_denies_smell": partial(_draft_suggestion, name="comment_denies_smell"),
    "add_params": _draft_add_params,
    "duplicate_body": _draft_duplicate_body,
    "add_accessors": _draft_add_accessors,
}

# invariant checked for each transform ("draft": the check the draft carries)
VERIFICATION = {
    "rename_locals": "node_types", "rename_members": "node_types", "rename_all": "node_types",
    "blank_inflate": "tokens", "blank_remove": "tokens", "reindent": "tokens",
    "strip_comments": "code_tokens", "compact": "code_tokens",
    "shuffle_methods": "members",
    "neutral_comments": "draft", "comment_claims_smell": "draft", "comment_denies_smell": "draft",
    "add_params": "draft", "duplicate_body": "draft", "add_accessors": "draft",
}
ALL_TRANSFORMS = tuple(_DRAFTS)


def transform_classes(cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Read and validate the `transform_classes` section of the config."""
    classes = {str(c): [str(t) for t in ts] for c, ts in cfg["transform_classes"].items()}
    names = [t for ts in classes.values() for t in ts]
    unknown = [c for c in classes if c not in CLASS_NAMES] + [t for t in names if t not in _DRAFTS]
    if unknown:
        raise ValueError(f"unknown transform classes or transforms: {unknown}")
    if len(set(names)) != len(names):
        raise ValueError("a transform is listed in more than one class")
    return classes


def _verify(kind: str, base: Snippet, var: Snippet, draft: _Draft) -> str:
    """Empty string when the variant keeps the invariant, else the reason."""
    if _error_count(var) > _error_count(base):
        return "more_parse_errors"
    if kind == "node_types":
        if _node_types(var) != _node_types(base):
            return "node_types_changed"
        return "" if _bindings(var) == _bindings(base) else "bindings_changed"
    if kind == "tokens":
        return "" if _tokens(var, True) == _tokens(base, True) else "tokens_changed"
    if kind == "code_tokens":
        return "" if _tokens(var, False) == _tokens(base, False) else "tokens_changed"
    if kind == "members":
        before = _member_multiset(base)
        return "" if before is not None and _member_multiset(var) == before else "members_changed"
    if kind == "draft":
        return draft.check(base, var) if draft.check is not None else "no_check"
    raise ValueError(f"unknown verification {kind}")


def apply(name: str, code: str, opts: Options, seed: int | str = 0) -> Outcome:
    """Run one transform on one snippet. The base code is kept unless the variant passes verification."""
    if name not in _DRAFTS:
        raise KeyError(f"unknown transform {name}")
    try:
        base = Snippet(code, opts)
        draft = _DRAFTS[name](base, seed)
        if draft.code is None:
            return Outcome(code, False, 0, "not_applicable", draft.reason, draft.info)
        if draft.code == code or draft.n_edits == 0:
            return Outcome(code, False, 0, "no_change", draft.reason or "nothing_to_change", draft.info)
        why = _verify(VERIFICATION[name], base, Snippet(draft.code, opts), draft)
    except _Rejected as exc:
        return Outcome(code, False, 0, "verification_failed", str(exc))
    except Exception as exc:  # noqa: BLE001 - one bad snippet must not stop a dataset run; counted as error
        return Outcome(code, False, 0, "error", f"{type(exc).__name__}: {exc}"[:300])
    if why:
        return Outcome(code, False, 0, "verification_failed", why, draft.info)
    return Outcome(draft.code, True, draft.n_edits, "applied", "", draft.info)


def transform(name: str, code: str, opts: Options, seed: int | str = 0) -> tuple[str, bool, int]:
    return apply(name, code, opts, seed).as_tuple()


TRANSFORMS: dict[str, Callable[..., tuple[str, bool, int]]] = {name: partial(transform, name)
                                                               for name in ALL_TRANSFORMS}
