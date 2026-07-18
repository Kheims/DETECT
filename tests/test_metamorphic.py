"""Tests of the metamorphic transforms and of the scripts that use them.

Covers every transform of mlcq_graphs/metamorphic.py on handwritten Java
snippets (shadowing, lambdas, catch and resource variables, anonymous classes,
generics, annotations, text blocks, comments and comment markers inside
strings, case labels, pattern variable scopes), the verification that sends a
sample back to its base code,
scripts/analysis/make_metamorphic_variants.py on a toy dataset, and
scripts/analysis/analyze_metamorphic.py end to end on synthetic predictions, its
numbers checked against a direct computation with sklearn.

    .venv/bin/python -m unittest tests/test_metamorphic.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import matthews_corrcoef, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis import analyze_metamorphic  # noqa: E402
from scripts.analysis import make_metamorphic_variants  # noqa: E402
from mlcq_graphs import metamorphic as mm  # noqa: E402
from mlcq_graphs.config import load_config  # noqa: E402

CONFIG = PROJECT_ROOT / "config/experiments/metamorphic.yml"
CFG = load_config(CONFIG, {})
OPTS = mm.Options.from_config(CFG, "mlcq")


def run(name: str, code: str, seed: str = "0:1", opts: mm.Options = OPTS) -> mm.Outcome:
    return mm.apply(name, code, opts, seed)


def errors(code: str) -> int:
    return mm._error_count(mm.Snippet(code, OPTS))


SHADOW = """class Counter {
    private int count;
    void add(int count) {
        this.count += count;
        int total = this.count + count;
        log(total, count);
    }
}
"""

LAMBDAS = """    int run(List<String> items, int limit) {
        int n = 0;
        for (int i = 0; i < limit; i++) { n += i; }
        for (String s : items) { n += s.length(); }
        items.forEach(x -> System.out.println(x));
        BiFunction<Integer, Integer, Integer> f = (a, b) -> a + b + n;
        try (Reader r = open()) { r.read(); } catch (IOException | RuntimeException e) { throw e; }
        if (items instanceof ArrayList<String> al) { al.trimToSize(); }
        return f.apply(n, limit);
    }
"""

ANONYMOUS = """    void start(final int delay) {
        final String name = "worker";
        Runnable r = new Runnable() {
            int ticks;
            @Override
            public void run() {
                int delay2 = delay + ticks;
                System.out.println(name + delay2);
            }
        };
        r.run();
    }
"""

PERSON = """public class Person {
    private String name;
    private int age;
    public Person(String name) { this.name = name; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public int getAge() { return this.age; }
    @Override
    public String toString() { return getName() + age; }
    public static void main(String[] args) { new Person("x").setName("y"); }
}
"""

GENERIC = """@Deprecated
public class Box<T extends Comparable<T>> {
    @SuppressWarnings("unchecked")
    private T value;
    public Box(T value) { this.value = value; }
    public <U> Box<U> map(Function<T, U> fn) { return new Box<U>(fn.apply(value)); }
    static Box<String> empty() { Box<String> b = new Box<>(""); return b; }
}
"""

STRINGS = """    /**
     * Javadoc.
     */
    String describe() {
        // a line comment
        String a = "// not a comment"; /* trailing */
        String b = "/* not a comment either */";
        char c = '/';
        return a + b + c; // done
    }
"""

TEXT_BLOCK = '''    String sql() {
        int x = 1;
        String q = """
            SELECT *
            FROM t // kept
            """;
        return q + x;
    }
'''

SHAPES = """class Shapes {
    private int a;
    int first() { return 1; }
    // second comment
    int second() { return 2; }
    private int b;
    int third() { return 3; }
}
"""

SUM = """    int sum(int[] xs) {
        int total = 0;
        for (int x : xs) { total += x; }
        return total;
    }
"""


class RenameTests(unittest.TestCase):
    def test_shadowing_local_against_field(self) -> None:
        out = run("rename_locals", SHADOW)
        self.assertTrue(out.applied)
        self.assertEqual(out.code, """class Counter {
    private int count;
    void add(int v0) {
        this.count += v0;
        int v1 = this.count + v0;
        log(v1, v0);
    }
}
""")

    def test_rename_members_keeps_the_shadowing_parameter(self) -> None:
        out = run("rename_members", SHADOW)
        self.assertIn("private int f0;", out.code)
        self.assertIn("void m0(int count)", out.code)
        self.assertIn("this.f0 += count;", out.code)
        self.assertIn("int total = this.f0 + count;", out.code)
        # log is not declared in the snippet
        self.assertIn("log(total, count);", out.code)

    def test_lambda_catch_resource_loop_and_pattern_variables(self) -> None:
        out = run("rename_locals", LAMBDAS)
        self.assertTrue(out.applied)
        self.assertEqual(out.n_edits, 32)
        for text in ("int run(List<String> v0, int v1)", "for (int v3 = 0; v3 < v1; v3++) { v2 += v3; }",
                     "for (String v4 : v0) { v2 += v4.length(); }", "v0.forEach(v5 -> System.out.println(v5));",
                     "(v7, v8) -> v7 + v8 + v2", "try (Reader v9 = open()) { v9.read(); }",
                     "catch (IOException | RuntimeException v10) { throw v10; }",
                     "ArrayList<String> v11) { v11.trimToSize(); }", "return v6.apply(v2, v1);"):
            self.assertIn(text, out.code)
        # method names and names bound outside the method are kept
        for name in ("open()", ".read()", ".length()", "System.out.println", ".apply("):
            self.assertIn(name, out.code)

    def test_anonymous_class(self) -> None:
        out = run("rename_locals", ANONYMOUS)
        self.assertIn("void start(final int v0)", out.code)
        self.assertIn("int v3 = v0 + ticks;", out.code)  # the field of the anonymous class is not a local
        self.assertIn("System.out.println(v1 + v3);", out.code)
        self.assertIn("public void run()", out.code)
        self.assertIn("v2.run();", out.code)
        members = run("rename_members", ANONYMOUS)
        self.assertIn("public void run()", members.code)  # @Override and anonymous class: kept
        self.assertIn("int f0;", members.code)
        self.assertIn("int delay2 = delay + f0;", members.code)
        self.assertEqual(members.info["kept_override"], 1)

    def test_rename_members_removes_accessor_names(self) -> None:
        out = run("rename_members", PERSON)
        self.assertTrue(out.applied)
        self.assertEqual(out.code, """public class Person {
    private String f0;
    private int f1;
    public Person(String name) { this.f0 = name; }
    public String m0() { return f0; }
    public void m1(String name) { this.f0 = name; }
    public int m2() { return this.f1; }
    @Override
    public String toString() { return m0() + f1; }
    public static void main(String[] args) { new Person("x").m1("y"); }
}
""")
        # toString (@Override) and main (keep_method_names) are kept; the constructor is not a method
        self.assertEqual(out.info["methods_kept"], 2)

    def test_rename_all_with_generics_and_annotations(self) -> None:
        out = run("rename_all", GENERIC)
        self.assertTrue(out.applied)
        self.assertEqual(out.code, """@Deprecated
public class C0<T extends Comparable<T>> {
    @SuppressWarnings("unchecked")
    private T f0;
    public C0(T v0) { this.f0 = v0; }
    public <U> C0<U> m0(Function<T, U> v0) { return new C0<U>(v0.apply(f0)); }
    static C0<String> m1() { C0<String> v0 = new C0<>(""); return v0; }
}
""")

    def test_rename_keeps_node_types_and_bindings(self) -> None:
        for code in (SHADOW, LAMBDAS, ANONYMOUS, PERSON, GENERIC, STRINGS, TEXT_BLOCK):
            for name in ("rename_locals", "rename_members", "rename_all"):
                out = run(name, code)
                base, var = mm.Snippet(code, OPTS), mm.Snippet(out.code, OPTS)
                self.assertEqual(mm._node_types(var), mm._node_types(base), name)
                self.assertEqual(mm._bindings(var), mm._bindings(base), name)

    def test_strings_and_text_blocks_untouched(self) -> None:
        out = run("rename_locals", TEXT_BLOCK)
        self.assertIn('"""\n            SELECT *\n            FROM t // kept\n            """', out.code)
        self.assertIn("return v1 + v0;", out.code)

    def test_nothing_to_rename(self) -> None:
        out = run("rename_locals", "    int one() { return 1; }\n")
        self.assertFalse(out.applied)
        self.assertEqual(out.status, "no_change")


class CaseLabelTests(unittest.TestCase):
    """A bare identifier case label names an enum constant unless the selector has a constant type."""

    def test_label_named_like_a_parameter_is_an_enum_constant(self) -> None:
        code = ("    void notify(Message message) {\n        switch (message.type()) {\n"
                "            case message: handle(message); break;\n            default: break;\n        }\n    }\n")
        out = run("rename_locals", code)
        self.assertTrue(out.applied)
        self.assertIn("switch (v0.type())", out.code)
        self.assertIn("case message: handle(v0); break;", out.code)

    def test_label_named_like_a_local_of_an_enum_selector(self) -> None:
        code = ("    void f(Kind kind) {\n        Error error = next();\n        switch (kind) {\n"
                "            case error: log(error); break;\n        }\n    }\n")
        out = run("rename_locals", code)
        self.assertIn("case error: log(v1); break;", out.code)

    def test_constant_field_against_an_enum_selector(self) -> None:
        code = ("class Painter {\n    static final String RED = \"r\";\n    void paint(Color c) {\n"
                "        switch (c) {\n            case RED: draw(RED); break;\n        }\n    }\n}\n")
        out = run("rename_members", code)
        self.assertTrue(out.applied)
        self.assertIn('static final String f0 = "r";', out.code)
        self.assertIn("case RED: draw(f0); break;", out.code)

    def test_constant_field_against_an_int_selector(self) -> None:
        code = ("class Codes {\n    static final int OK = 0;\n    int f(int c) {\n        switch (c) {\n"
                "            case OK: return OK;\n            default: return 1;\n        }\n    }\n}\n")
        out = run("rename_members", code)
        self.assertTrue(out.applied)
        self.assertIn("case f0: return f0;", out.code)

    def test_constant_field_against_an_unknown_selector(self) -> None:
        code = ("class Codes {\n    static final int OK = 0;\n    int f(int x) {\n        switch (code()) {\n"
                "            case OK: return x;\n        }\n        return 0;\n    }\n}\n")
        out = run("rename_members", code)
        self.assertEqual((out.code, out.applied, out.status, out.reason), (code, False, "verification_failed",
                                                                          "case_label"))
        # the label is not renamed by rename_locals, so the sample is kept
        local = run("rename_locals", code)
        self.assertTrue(local.applied)
        self.assertIn("case OK: return v0;", local.code)

    def test_constant_switch_inferred_from_the_selector_or_another_label(self) -> None:
        for switch, other in (("switch (code() & MASK)", ""), ("switch (name())", 'case "x": return 2;\n')):
            code = (f"class Codes {{\n    static final String OK = \"ok\";\n    int f() {{\n        {switch} {{\n"
                    f"            {other}case OK: return 1;\n        }}\n        return 0;\n    }}\n}}\n")
            out = run("rename_members", code)
            self.assertTrue(out.applied, f"{switch}: {out.reason}")
            self.assertIn("case f0: return 1;", out.code)


class PatternScopeTests(unittest.TestCase):
    """Pattern variables are renamed only where they are definitely matched (JLS 6.3.1)."""

    def test_uses_outside_the_matched_regions_are_rejected(self) -> None:
        for body in ("if (o instanceof String s) { use(s); }\n        use(s);",
                     "if (o instanceof String s) { use(s); } else { use(s); }",
                     "if (!(o instanceof String s)) { return; }\n        use(s);"):
            code = f"class P {{\n    String s;\n    void f(Object o) {{\n        {body}\n    }}\n}}\n"
            for name in ("rename_locals", "rename_members"):
                out = run(name, code)
                self.assertEqual((out.code, out.status, out.reason), (code, "verification_failed", "pattern_scope"),
                                 f"{name}: {body}")

    def test_matched_regions(self) -> None:
        code = """    int f(Object o) {
        if (o instanceof String s && s.length() > 0) { return s.length(); }
        int n = o instanceof Integer i ? i : 0;
        while (o instanceof Number num) { o = num.intValue() > 0 ? null : o; }
        if (!(o instanceof Long l) || l > 0) { return 1; }
        return n;
    }
"""
        out = run("rename_locals", code)
        self.assertTrue(out.applied, out.reason)
        self.assertEqual(out.code, """    int f(Object v0) {
        if (v0 instanceof String v1 && v1.length() > 0) { return v1.length(); }
        int v2 = v0 instanceof Integer v3 ? v3 : 0;
        while (v0 instanceof Number v4) { v0 = v4.intValue() > 0 ? null : v0; }
        if (!(v0 instanceof Long v5) || v5 > 0) { return 1; }
        return v2;
    }
""")


class LayoutTests(unittest.TestCase):
    def test_strip_comments_keeps_markers_inside_strings(self) -> None:
        out = run("strip_comments", STRINGS)
        self.assertEqual(out.code, """    String describe() {
        String a = "// not a comment";
        String b = "/* not a comment either */";
        char c = '/';
        return a + b + c;
    }
""")
        self.assertEqual(out.n_edits, 4)

    def test_blank_inflate_and_remove(self) -> None:
        out = run("blank_inflate", SUM)
        self.assertEqual(out.code.count("\n"), 2 * SUM.count("\n") - 1)
        self.assertEqual(run("blank_remove", out.code).code, SUM)
        self.assertEqual(run("blank_remove", SUM).status, "no_change")

    def test_blank_inflate_skips_text_blocks(self) -> None:
        out = run("blank_inflate", TEXT_BLOCK)
        self.assertTrue(out.applied)
        self.assertIn('"""\n            SELECT *\n            FROM t // kept\n            """', out.code)

    def test_reindent(self) -> None:
        tabs = run("reindent", "\tint f(int a) {\n\t\treturn a;\n\t}\n")
        self.assertEqual(tabs.code, "        int f(int a) {\n                return a;\n        }\n")
        spaces = run("reindent", SUM)
        self.assertEqual(spaces.code.splitlines()[1], "    int total = 0;")
        block = run("reindent", TEXT_BLOCK)
        self.assertIn('  String sql() {\n    int x = 1;\n    String q = """\n            SELECT *\n', block.code)

    def test_compact(self) -> None:
        self.assertEqual(run("compact", SUM).code,
                         "int sum(int[] xs) { int total = 0; for (int x : xs) { total += x; } return total; }\n")
        self.assertEqual(run("compact", SHAPES).code, """class Shapes {
private int a;
int first() { return 1; }
int second() { return 2; }
private int b;
int third() { return 3; }
}
""")
        self.assertEqual(run("compact", STRINGS).code,
                         'String describe() { String a = "// not a comment"; String b = "/* not a comment either */"; '
                         "char c = '/'; return a + b + c; }\n")
        self.assertEqual(run("compact", TEXT_BLOCK).reason, "text_block")

    def test_shuffle_methods(self) -> None:
        out = run("shuffle_methods", SHAPES, seed="0:1")
        self.assertTrue(out.applied)
        lines = out.code.splitlines()
        # fields stay in their slots, the comment moves with its method
        self.assertEqual(lines[1], "    private int a;")
        self.assertEqual(lines[5], "    private int b;")
        self.assertEqual(lines[lines.index("    int second() { return 2; }") - 1], "    // second comment")
        self.assertNotEqual(out.code, SHAPES)
        self.assertEqual(run("shuffle_methods", SHAPES, seed="0:1").code, out.code)
        self.assertEqual(sorted(SHAPES.splitlines()), sorted(lines))
        self.assertEqual(run("shuffle_methods", SUM).status, "not_applicable")

    def test_neutral_comments(self) -> None:
        out = run("neutral_comments", PERSON)
        self.assertEqual(out.n_edits, 2)
        lines = out.code.splitlines()
        self.assertEqual(lines[5], "    // see issue tracker")
        self.assertEqual(lines[11], "    // see issue tracker")
        self.assertEqual(run("neutral_comments", TEXT_BLOCK).reason, "too_few_lines")

    def test_neutral_comments_never_inside_text_blocks_or_comments(self) -> None:
        code = ('    String f() {\n        /* one\n           two\n           three\n           four\n           five\n'
                '           six */\n        String q = """\n            a\n            b\n            c\n'
                '            d\n            e\n            """;\n        return q;\n    }\n')
        out = run("neutral_comments", code)
        self.assertTrue(out.applied)
        self.assertIn('"""\n            a\n            b\n            c\n            d\n            e\n            """',
                      out.code)
        self.assertIn("/* one\n           two\n           three\n           four\n           five\n           six */",
                      out.code)

    def test_suggestions_by_level_and_dataset(self) -> None:
        dacos = mm.Options.from_config(CFG, "dacos")
        method = run("comment_claims_smell", SUM)
        self.assertEqual(method.code.splitlines()[0],
                         "    " + CFG["datasets"]["mlcq"]["transform_options"]["suggestions"]
                         ["comment_claims_smell"]["method"])
        self.assertEqual(method.info["level"], "method")
        cls = run("comment_denies_smell", SHAPES)
        self.assertEqual(cls.code.splitlines()[0], OPTS.suggestion("comment_denies_smell", "class"))
        self.assertEqual(cls.info["level"], "class")
        claim = run("comment_claims_smell", SUM, opts=dacos)
        self.assertIn("too many parameters", claim.code.splitlines()[0])
        # a Javadoc stays above, the claim goes right above the declaration
        doc = run("comment_claims_smell", STRINGS)
        self.assertTrue(doc.code.splitlines()[3].lstrip().startswith("// TODO"))
        self.assertEqual(doc.code.splitlines()[4], "    String describe() {")


class ControlTests(unittest.TestCase):
    def test_add_params(self) -> None:
        out = run("add_params", SUM)
        self.assertIn("int sum(int[] xs, int p0, String p1, boolean p2, double p3) {", out.code)
        varargs = run("add_params", "    void log(String fmt, Object... args) { System.out.printf(fmt, args); }\n")
        self.assertIn("void log(String fmt, int p0, String p1, boolean p2, double p3, Object... args)", varargs.code)
        empty = run("add_params", "    int one() { return 1; }\n")
        self.assertIn("int one(int p0, String p1, boolean p2, double p3)", empty.code)

    def test_duplicate_body_renames_the_locals_of_the_copy(self) -> None:
        out = run("duplicate_body", SUM)
        self.assertEqual(out.code, """    int sum(int[] xs) {
        int total = 0;
        for (int x : xs) { total += x; }
        if (false) {
        int d0 = 0;
        for (int d1 : xs) { d0 += d1; }
        return d0;
        }
        return total;
    }
""")
        self.assertEqual(errors(out.code), 0)
        lambdas = run("duplicate_body", LAMBDAS)
        self.assertTrue(lambdas.applied)
        self.assertEqual(errors(lambdas.code), 0)
        self.assertEqual(run("duplicate_body", "    abstract int f(int a);\n").status, "not_applicable")

    def test_duplicate_body_check_counts_every_node(self) -> None:
        base = mm.Snippet(SUM, OPTS)
        draft = mm._draft_duplicate_body(base, 0)
        self.assertEqual(draft.check(base, mm.Snippet(draft.code, OPTS)), "")
        lost = draft.code.replace("        for (int d1 : xs) { d0 += d1; }\n", "", 1)
        extra = draft.code.replace("int d0 = 0;", "int d0 = 0; d0++;", 1)
        for code in (lost, extra):
            self.assertNotEqual(code, draft.code)
            self.assertEqual(draft.check(base, mm.Snippet(code, OPTS)), "node_counts")

    def test_add_accessors(self) -> None:
        out = run("add_accessors", PERSON)
        self.assertEqual(out.n_edits, 9)
        for text in ("private int extra0;", "private String extra1;", "private boolean extra2;",
                     "public int getExtra0() {", "public void setExtra2(boolean extra2) {",
                     "        this.extra2 = extra2;"):
            self.assertIn(text, out.code)
        self.assertEqual(errors(out.code), 0)
        self.assertEqual(run("add_accessors", SUM).status, "not_applicable")
        self.assertEqual(run("add_accessors", "interface I {\n    void f();\n}\n").status, "not_applicable")


class VerificationTests(unittest.TestCase):
    def swap(self, name: str, draft) -> None:
        old = mm._DRAFTS[name]
        mm._DRAFTS[name] = draft
        self.addCleanup(mm._DRAFTS.__setitem__, name, old)

    def test_changed_binding_falls_back_to_the_base(self) -> None:
        code = "    void f(int a) { int b = a; }\n"
        self.swap("rename_locals", lambda sn, seed: mm._Draft(code.replace("int b = a", "int b = c"), 1))
        out = run("rename_locals", code)
        self.assertEqual((out.code, out.applied, out.n_edits, out.status, out.reason),
                         (code, False, 0, "verification_failed", "bindings_changed"))

    def test_changed_node_types_fall_back_to_the_base(self) -> None:
        code = "    void f(int a) { int b = a; }\n"
        self.swap("rename_locals", lambda sn, seed: mm._Draft(code.replace("int b = a", "int b = a + 1"), 1))
        self.assertEqual(run("rename_locals", code).reason, "node_types_changed")

    def test_parse_errors_fall_back_to_the_base(self) -> None:
        code = "    void f(int a) { int b = a; }\n"
        self.swap("blank_inflate", lambda sn, seed: mm._Draft(code.replace("int b = a;", "int b = a"), 1))
        self.assertEqual(run("blank_inflate", code).reason, "more_parse_errors")

    def test_token_change_falls_back_to_the_base(self) -> None:
        code = "    void f(int a) {\n        int b = a;\n    }\n"
        self.swap("reindent", lambda sn, seed: mm._Draft(code.replace("int b", "long b"), 1))
        self.assertEqual(run("reindent", code).reason, "tokens_changed")

    def test_every_transform_keeps_the_parse(self) -> None:
        for code in (SHADOW, LAMBDAS, ANONYMOUS, PERSON, GENERIC, STRINGS, TEXT_BLOCK, SHAPES, SUM):
            for name in mm.ALL_TRANSFORMS:
                out = run(name, code)
                self.assertNotIn(out.status, ("error", "verification_failed"), f"{name}: {out.reason}")
                self.assertLessEqual(errors(out.code), errors(code), name)
                self.assertEqual(out.applied, out.code != code, name)

    def test_config_checks(self) -> None:
        bad = dict(CFG, transform_classes={"presentation_invariant": ["no_such_transform"]})
        with self.assertRaises(ValueError):
            mm.transform_classes(bad)
        twice = dict(CFG, transform_classes={"presentation_invariant": ["blank_inflate"],
                                             "suggestion": ["blank_inflate"]})
        with self.assertRaises(ValueError):
            mm.transform_classes(twice)
        names = [t for ts in mm.transform_classes(CFG).values() for t in ts]
        self.assertEqual(sorted(names), sorted(mm.ALL_TRANSFORMS))
        topts = dict(CFG["transform_options"], neutral_comments={"text": "// two\nlines", "every": 5})
        with self.assertRaises(ValueError):
            mm.Options.from_config(dict(CFG, transform_options=topts))
        # no default for the inserted comments: the sections are required
        for section in ("neutral_comments", "suggestions"):
            topts = {k: v for k, v in CFG["transform_options"].items() if k != section}
            with self.assertRaises(KeyError, msg=section):
                mm.Options.from_config(dict(CFG, transform_options=topts))


def write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


class VariantScriptTest(unittest.TestCase):
    def test_toy_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = [{"code_snippet": code, "level": level, "y": [1, 0]}
                       for code, level in ((SUM, "method"), (PERSON, "class"), ("    int f( {\n", "method"))]
            (root / "base.json").write_text(json.dumps(entries))
            names = ["blank_inflate", "rename_locals", "comment_claims_smell", "add_params", "add_accessors"]
            cfg = {
                "transform_classes": {"presentation_invariant": names[:2], "suggestion": names[2:3],
                                      "construct_control": names[3:]},
                "seeds": {}, "wrapper": CFG["wrapper"], "transform_options": CFG["transform_options"],
                "datasets": {"toy": {"base_json": str(root / "base.json"), "label_order": ["is_a", "is_b"],
                                     "level": {"source": "field", "field": "level"}, "out_dir": str(root / "out")}},
                "variants": {"workers": 1, "chunksize": 1, "size_sanity_csv": str(root / "sanity.csv"),
                             "applicability_csv": str(root / "applicability.csv"),
                             "size_expectations": CFG["variants"]["size_expectations"]},
            }
            with redirect_stdout(StringIO()) as printed:
                make_metamorphic_variants.main(["--config", str(write_yaml(root / "cfg.yml", cfg))])
            self.assertNotIn("unexpected", printed.getvalue())
            stats = json.loads((root / "out/stats.json").read_text())
            self.assertEqual(stats["transforms"]["add_accessors"]["transform_class"], "construct_control")
            self.assertEqual(stats["transforms"]["add_accessors"]["levels"]["class"]["applied"], 1)
            self.assertEqual(set(stats["transforms"]["add_accessors"]), {"transform_class", "sha256", "levels"})
            for name in names:
                text = (root / f"out/{name}.normalized.json").read_text()
                self.assertEqual(stats["transforms"][name]["sha256"], hashlib.sha256(text.encode()).hexdigest())
                variant = json.loads(text)
                self.assertEqual(len(variant), len(entries))
                for base, v in zip(entries, variant):
                    self.assertEqual(set(v["metamorphic"]), {"transform", "applied", "n_edits"})
                    self.assertEqual(v["metamorphic"]["applied"], v["code_snippet"] != base["code_snippet"])
                    self.assertEqual({k: x for k, x in v.items() if k not in ("code_snippet", "metamorphic")},
                                     {k: x for k, x in base.items() if k != "code_snippet"})
                with (root / f"out/{name}.size_features.csv").open() as fh:
                    rows = list(csv.DictReader(fh))
                self.assertEqual([int(r["sample_idx"]) for r in rows], [0, 1, 2])
            with (root / "applicability.csv").open() as fh:
                table = {(r["transform"], r["level"]): r for r in csv.DictReader(fh)}
            self.assertEqual(table[("add_accessors", "method")]["applied"], "0")
            self.assertEqual(table[("blank_inflate", "all")]["errors"], "0")


class AnalysisTest(unittest.TestCase):
    N, LABELS, SEEDS = 40, ["is_a", "is_b"], (42, 43)

    def folds(self) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
        idx = np.arange(self.N)
        out = []
        for seed in self.SEEDS:
            halves = [idx[:20], idx[20:]] if seed == 42 else [idx[1::2], idx[0::2]]
            out += [(seed, 0, halves[0], halves[1]), (seed, 1, halves[1], halves[0])]
        return out

    def test_end_to_end(self) -> None:
        rng = np.random.default_rng(0)
        y = np.stack([np.isin(np.arange(self.N) % 4, [0, 3]), np.arange(self.N) % 3 == 1], axis=1).astype(int)
        base_p = rng.uniform(0.05, 0.95, size=(len(self.SEEDS), 2, self.N, 2))
        applied = {"blank_inflate": np.arange(self.N) % 3 != 0, "add_params": np.ones(self.N, dtype=bool)}
        thr = {(seed, fold): np.array([0.5, 0.35 + 0.1 * fold]) for seed in self.SEEDS for fold in (0, 1)}

        def variant_p(t: str, p: np.ndarray, j: int) -> np.ndarray:
            if t == "add_params":
                return 1.0 - p
            return np.clip(p + (0.3 if j % 5 == 0 else -0.05), 0.0, 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for t, flags in applied.items():
                (root / "variants").mkdir(exist_ok=True)
                (root / f"variants/{t}.normalized.json").write_text(json.dumps(
                    [{"code_snippet": "", "metamorphic": {"transform": t, "applied": bool(a), "n_edits": int(a)}}
                     for a in flags]))
            base_rows, records = [], []
            variant_rows: dict[str, list[dict]] = {t: [] for t in applied}
            for seed, fold, test, val in self.folds():
                si = self.SEEDS.index(seed)
                for part, ids in (("test", test), ("val", val)):
                    for j in ids:
                        row = {"model": "toy", "strategy": "kfold", "seed": seed, "fold": fold, "part": part,
                               "json_index": int(j), "y": [int(v) for v in y[j]],
                               "prob": [float(v) for v in base_p[si, fold, j]]}
                        base_rows.append(row)
                        for t in applied:
                            # blank_inflate: rows of applied samples, plus rows of even unapplied samples with
                            # another score (scorer noise: ignored, counted); variant val rows must be ignored
                            if part == "test" and (t == "add_params" or applied[t][j]):
                                variant_rows[t].append(dict(row, prob=[float(v) for v in variant_p(
                                    t, base_p[si, fold, j], int(j))]))
                            elif part == "test" and j % 2 == 0:
                                variant_rows[t].append(dict(row, prob=[float(1.0 - v) for v in base_p[si, fold, j]]))
                            elif part == "val":
                                variant_rows[t].append(dict(row, prob=[0.999, 0.999]))
                records.append({"seed": seed, "combo": {"split_strategy": "kfold", "cv_fold": fold},
                                "thresholds": thr[(seed, fold)].tolist()})
            run_dir = root / "run"
            run_dir.mkdir()
            for name, rows in [("predictions.jsonl", base_rows)] + [
                    (f"predictions_variant_{t}.jsonl", r) for t, r in variant_rows.items()]:
                (run_dir / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
            (run_dir / "records.json").write_text(json.dumps(records))
            # a second family without records: thresholds tuned on the base val part
            other = root / "other"
            other.mkdir()
            for name in [f"predictions_variant_{t}.jsonl" for t in applied] + ["predictions.jsonl"]:
                (other / name).write_text((run_dir / name).read_text())

            cfg = {
                "transform_classes": {"presentation_invariant": ["blank_inflate"],
                                      "construct_control": ["add_params"]},
                "datasets": {"toy": {"label_order": self.LABELS, "out_dir": str(root / "variants")}},
                "analysis": {
                    "out_dir": str(root / "analysis"), "figures_dir": str(root / "analysis/figures"),
                    "strategies": ["kfold"], "threshold_steps": 101, "clip": 1.0e-6,
                    "bootstrap": {"n_boot": 50, "ci": 0.95, "seed": 0},
                    "strongest": {"k": 2, "min_rows": 1}, "class_labels": {},
                    "variant_pattern": "predictions_variant_{transform}.jsonl",
                    "families": {"toy": {"dataset": "toy", "predictions": [str(run_dir / "predictions.jsonl")]},
                                 "tuned": {"dataset": "toy", "model": "toy",
                                           "predictions": [str(other / "predictions.jsonl")]}},
                },
            }
            with redirect_stdout(StringIO()):
                analyze_metamorphic.main(["--config", str(write_yaml(root / "cfg.yml", cfg))])
            with (root / "analysis/summary.csv").open() as fh:
                rows = list(csv.DictReader(fh))
            summary = json.loads((root / "analysis/summary.json").read_text())
            self.assertEqual(len(rows), 2 * 2 * 2)
            self.assertEqual(summary["families"]["toy"]["threshold_sources"], {"records": 4})
            self.assertEqual(summary["families"]["tuned"]["threshold_sources"], {"tuned_on_val": 4})
            for name in ("flip_kfold_a.png", "flip_kfold_b.pdf", "score_shift_kfold.png"):
                self.assertTrue((root / "analysis/figures" / name).exists(), name)

            got = {(r["transform"], r["smell"]): r for r in rows if r["family"] == "toy"}
            for t in applied:
                for k, label in enumerate(self.LABELS):
                    want = self.expected(t, k, y, base_p, applied[t], thr, variant_p)
                    self.assertGreater(want["n_c2w"] * want["n_w2c"], 0)  # both transitions occur
                    row = got[(t, label.removeprefix("is_"))]
                    self.assertEqual(row["transform_class"],
                                     "construct_control" if t == "add_params" else "presentation_invariant")
                    # unapplied even samples, once per seed
                    noisy = len(self.SEEDS) * int(((~applied[t]) & (np.arange(self.N) % 2 == 0)).sum())
                    self.assertEqual(int(row["n_unapplied_differ"]), noisy)
                    self.assertEqual(noisy > 0, t == "blank_inflate")
                    for key, value in want.items():
                        self.assertAlmostEqual(float(row[key]), value, places=3, msg=f"{t} {label} {key}")
                    self.assertLessEqual(float(row["flip_rate_lo"]), float(row["flip_rate"]) + 1e-9)
                    self.assertGreaterEqual(float(row["flip_rate_hi"]), float(row["flip_rate"]) - 1e-9)

    def expected(self, t: str, k: int, y: np.ndarray, base_p: np.ndarray, flags: np.ndarray,
                 thr: dict, variant_p) -> dict[str, float]:
        flips, c2w, w2c, dl, n_test = [], 0, 0, [], 0
        mcc: dict[int, list[float]] = {s: [] for s in self.SEEDS}
        auc: dict[int, list[float]] = {s: [] for s in self.SEEDS}
        for seed, fold, test, _ in self.folds():
            si = self.SEEDS.index(seed)
            pb = base_p[si, fold, test, k]
            pv = np.array([variant_p(t, base_p[si, fold, j], int(j))[k] if flags[j] else base_p[si, fold, j, k]
                           for j in test])
            yt, th = y[test, k], thr[(seed, fold)][k]
            n_test += len(test)
            mcc[seed].append(matthews_corrcoef(yt, pv >= th) - matthews_corrcoef(yt, pb >= th))
            auc[seed].append(roc_auc_score(yt, pv) - roc_auc_score(yt, pb))
            for r, j in enumerate(test):
                if not flags[j]:
                    continue
                before, after = pb[r] >= th, pv[r] >= th
                flips.append(before != after)
                c2w += int(before != after and before == bool(yt[r]))
                w2c += int(before != after and after == bool(yt[r]))
                q = np.clip([pb[r], pv[r]], 1e-6, 1 - 1e-6)
                dl.append(np.log(q[1] / (1 - q[1])) - np.log(q[0] / (1 - q[0])))
        return {
            "n_rows": float(len(flips)), "applicability": len(flips) / n_test, "flip_rate": float(np.mean(flips)),
            "n_c2w": float(c2w), "n_w2c": float(w2c), "c2w_rate": c2w / len(flips), "w2c_rate": w2c / len(flips),
            "mean_dlogit": float(np.mean(dl)), "median_dlogit": float(np.median(dl)),
            "delta_mcc": float(np.mean([np.mean(v) for v in mcc.values()])),
            "delta_auc": float(np.mean([np.mean(v) for v in auc.values()])),
        }


if __name__ == "__main__":
    unittest.main()
