"""Descriptive analysis of the DACOS labels: agreement, sampling design and what drives a label.

Inputs: normalized DACOS files, DACOS folds, the DACOS MySQL dump and MLCQ metadata.
Outputs: a JSON summary, tree-sitter features of the method snippets and figures.

Usage:
    python scripts/labels/analyze_dacos.py --config config/experiments/dacos_analysis.yml
    python scripts/labels/analyze_dacos.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.evaluation.cv import iter_folds, load_folds  # noqa: E402
from scripts.labels.analyze_mlcq import alpha_with_ci, atomic_json, safe_auc, tmp_path  # noqa: E402
from scripts.labels.analyze_mlcq import self_test as alpha_self_test  # noqa: E402
from scripts.data.convert_dacos import load_table, to_bit  # noqa: E402
from scripts.data.extract_size_features import COMMENT_TYPES, METHOD_TYPES, STATEMENT_TYPES  # noqa: E402

COLORS = {"cm": "#2a78d6", "lpl": "#eb6834", "ma": "#1baf7a", "mlcq": "#52514e"}
NEUTRAL = "#c9c8c3"
INK, INK_SECONDARY = "#0b0b0b", "#52514e"

SQL_COLUMNS = {
    "annotation": ["id", "iscm", "isim", "islp", "isma", "is_smell", "sample_id"],
    "sample": ["id", "designite_id", "has_smell", "is_class", "path_to_file", "project_name",
               "sample_constraints", "smells"],
    "method_metrics": ["id", "cc", "loc", "pc", "method_name", "package_name", "project_name", "type_name"],
    "class_metrics": ["id", "dit", "fanin", "fanout", "lcom", "loc", "nc", "nof", "nom", "nopf", "nopm", "wmc",
                      "package_name", "project_name", "type_name"],
}
FLAGS = ["iscm", "isim", "islp", "isma", "is_smell"]

# tree-sitter-java node types of the method features
LOOP_TYPES = {"for_statement", "enhanced_for_statement", "while_statement", "do_statement"}
NESTING_TYPES = LOOP_TYPES | {"if_statement", "switch_expression", "switch_statement", "try_statement",
                              "try_with_resources_statement"}
PRIMITIVE_TYPES = {"integral_type", "floating_point_type", "boolean_type"}
PARAM_TYPES = {"formal_parameter", "spread_parameter"}


def resolve(path: Any) -> Path:
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


# ------------------------------------------------------------ statistics

def z_value(level: float) -> float:
    return float(stats.norm.ppf(1 - (1 - level) / 2))


def wilson(k: int, n: int, level: float) -> dict[str, Any]:
    """Wilson score interval of a binomial proportion."""
    if n == 0:
        return {"n": 0, "k": 0, "rate": None, "ci_low": None, "ci_high": None}
    z = z_value(level)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # clamp so that rounding never puts the rate outside its interval
    lo, hi = min(p, max(0.0, center - half)), max(p, min(1.0, center + half))
    return {"n": int(n), "k": int(k), "rate": p, "ci_low": lo, "ci_high": hi}


def chi_square(groups: list[str], y: np.ndarray) -> dict[str, Any]:
    """Chi-square test of independence of a group label and a binary outcome."""
    names = sorted(set(groups))
    g = np.asarray(groups)
    table = np.array([[int(((g == n) & (y == 1)).sum()), int(((g == n) & (y == 0)).sum())] for n in names])
    table = table[table.sum(axis=1) > 0]
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"chi2": None, "dof": None, "p_value": None, "cramers_v": None, "groups": len(names)}
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    v = math.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return {"chi2": float(chi2), "dof": int(dof), "p_value": float(p), "cramers_v": v, "groups": int(table.shape[0]),
            "min_expected": float(expected.min()), "share_expected_below_5": float((expected < 5).mean())}


def chi2_stat(onehot: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson chi-square of a group x binary-flag table, for one flag vector or a batch."""
    n_g = onehot.sum(axis=0)
    total = onehot.shape[0]
    pos_total = float(np.asarray(y).sum(axis=-1).reshape(-1)[0])
    e1, e0 = n_g * pos_total / total, n_g * (total - pos_total) / total
    pos = np.asarray(y, dtype=np.float64) @ onehot
    return ((pos - e1) ** 2 * (1 / e1 + 1 / e0)).sum(axis=-1)


def block_permutations(samples: np.ndarray, n_perm: int, rng: np.random.Generator,
                       chunk: int = 500) -> Any:
    """Yield (batch, rows) permuted positions that move the rows of each sample together."""
    order = np.argsort(samples, kind="stable")
    _, first, counts = np.unique(samples[order], return_index=True, return_counts=True)
    by_size: dict[int, list[np.ndarray]] = defaultdict(list)
    for f, c in zip(first, counts):
        by_size[int(c)].append(order[f:f + c])
    mats = [np.array(v) for v in by_size.values()]
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        src = np.empty((b, len(samples)), dtype=np.int64)
        for m in mats:
            pi = np.argsort(rng.random((b, len(m))), axis=1)
            for j in range(m.shape[1]):
                src[:, m[:, j]] = m[pi, j]
        yield src
        done += b


def chi_square_blocks(groups: list[str], y: np.ndarray, samples: np.ndarray, n_perm: int,
                      rng: np.random.Generator) -> dict[str, Any]:
    """chi_square plus a block permutation p-value (rows of a sample move together)."""
    out = chi_square(groups, y)
    if out["chi2"] is None or n_perm <= 0:
        return out
    g = np.asarray(groups)
    onehot = np.stack([(g == n).astype(np.float64) for n in sorted(set(groups))], axis=1)
    obs = float(chi2_stat(onehot, y))
    ge = 0
    for src in block_permutations(samples, n_perm, rng):
        ge += int((chi2_stat(onehot, y[src]) >= obs * (1 - 1e-9)).sum())
    out["block_permutations"] = int(n_perm)
    out["block_permutation_p_value"] = (1 + ge) / (1 + n_perm)
    return out


def log_likelihood(x: np.ndarray, y: np.ndarray, c: float, max_iter: int) -> tuple[float, bool]:
    """Log-likelihood of a near-unpenalized logistic fit (L2 with a large C)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = LogisticRegression(C=c, max_iter=max_iter).fit(x, y)
    converged = not any("converge" in str(w.message).lower() for w in caught)
    p = np.clip(model.predict_proba(x)[:, 1], 1e-12, 1 - 1e-12)
    return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum()), converged


def lrt(base: np.ndarray, extra: np.ndarray, y: np.ndarray, c: float, max_iter: int) -> dict[str, Any]:
    """Likelihood-ratio test of the columns `extra` added to `base` (intercept in both)."""
    extra = extra[:, extra.std(axis=0) > 0]
    full = np.hstack([base, extra])
    df = np.linalg.matrix_rank(np.hstack([np.ones((len(y), 1)), full])) - \
        np.linalg.matrix_rank(np.hstack([np.ones((len(y), 1)), base]))
    ll0, ok0 = log_likelihood(base, y, c, max_iter)
    ll1, ok1 = log_likelihood(full, y, c, max_iter)
    stat = max(0.0, 2 * (ll1 - ll0))
    return {"n": int(len(y)), "ll_base": ll0, "ll_full": ll1, "lr_stat": stat, "df": int(df),
            "p_value": float(stats.chi2.sf(stat, df)) if df > 0 else None, "converged": bool(ok0 and ok1)}


def summary(values: np.ndarray, quantiles: list[float]) -> dict[str, Any]:
    if len(values) == 0:
        return {"n": 0}
    qs = np.quantile(values, quantiles)
    return {"n": int(len(values)), "mean": float(values.mean()),
            **{f"q{int(round(q * 100)):02d}": float(v) for q, v in zip(quantiles, qs)}}


def dummies(labels: list[str], drop_first: bool = True) -> tuple[np.ndarray, list[str]]:
    names = sorted(set(labels))
    if drop_first:
        names = names[1:]
    return np.array([[float(lab == n) for n in names] for lab in labels]).reshape(len(labels), len(names)), names


# ------------------------------------------------------------ tree-sitter features

_PARSER: Any = None


def _parser() -> Any:
    global _PARSER
    if _PARSER is None:
        import tree_sitter as ts
        import tree_sitter_java as tsj

        _PARSER = ts.Parser(ts.Language(tsj.language()))
    return _PARSER


def _param_type(param: Any) -> Any:
    t = param.child_by_field_name("type")
    if t is not None:
        return t
    for c in param.named_children:
        if c.type not in ("modifiers", "variable_declarator", "identifier", "dimensions"):
            return c
    return None


def _is_else_if(node: Any, parent: Any) -> bool:
    if node.type != "if_statement" or parent is None or parent.type != "if_statement":
        return False
    alt = parent.child_by_field_name("alternative")
    return alt is not None and (alt.start_byte, alt.end_byte) == (node.start_byte, node.end_byte)


def method_features(snippet: str, prefix: str, suffix: str) -> dict[str, Any]:
    """Tree-sitter counts on a method snippet, as shown to the annotators.

    Parameter features describe the top-level method with the most parameters."""
    body = snippet.rstrip("\n")
    pb, bb = prefix.encode("utf-8"), body.encode("utf-8")
    src = pb + bb + suffix.encode("utf-8")
    lo, hi = len(pb), len(pb) + len(bb)
    tree = _parser().parse(src)
    out: dict[str, Any] = {"ts_loc_nonblank": sum(1 for line in body.splitlines() if line.strip()),
                           "ts_parse_error": int(tree.root_node.has_error)}
    counts: Counter = Counter()
    methods: list[Any] = []
    max_nesting = 0
    # (node, nesting depth, parent, inside a top-level method of the snippet)
    stack: list[tuple[Any, int, Any, bool]] = [(tree.root_node, 0, None, False)]
    while stack:
        node, nesting, parent, in_method = stack.pop()
        inside = node.start_byte >= lo and node.end_byte <= hi
        kind = node.type
        if kind in COMMENT_TYPES or node.is_missing:
            continue
        if inside:
            if kind in METHOD_TYPES and not in_method:
                methods.append(node)
                in_method = True
            if kind in NESTING_TYPES and not _is_else_if(node, parent):
                nesting += 1
                max_nesting = max(max_nesting, nesting)
            if kind in STATEMENT_TYPES:
                counts["ts_statements"] += 1
            if kind in LOOP_TYPES:
                counts["ts_loops"] += 1
            if kind == "if_statement":
                counts["ts_if"] += 1
                if node.child_by_field_name("alternative") is not None:
                    counts["ts_else"] += 1
            elif kind == "switch_label" and node.child_count and node.children[0].type == "case":
                counts["ts_case"] += 1
            elif kind == "catch_clause":
                counts["ts_catch"] += 1
            elif kind == "ternary_expression":
                counts["ts_ternary"] += 1
            elif kind == "return_statement":
                counts["ts_returns"] += 1
            elif kind == "binary_expression":
                op = node.child_by_field_name("operator")
                if op is not None and op.type in ("&&", "||"):
                    counts["ts_bool_ops"] += 1
        for child in reversed(node.children):
            stack.append((child, nesting, node, in_method))
    out["ts_max_nesting"] = max_nesting
    for key in ("ts_statements", "ts_loops", "ts_if", "ts_else", "ts_case", "ts_catch", "ts_ternary",
                "ts_returns", "ts_bool_ops"):
        out[key] = int(counts[key])
    per_method = []
    for m in methods:
        plist = m.child_by_field_name("parameters")
        params = [c for c in plist.named_children if c.type in PARAM_TYPES] if plist is not None else []
        types = []
        for prm in params:
            t = _param_type(prm)
            text = "" if t is None else " ".join(src[t.start_byte:t.end_byte].decode("utf-8", "replace").split())
            types.append((prm.type, None if t is None else t.type, text))
        per_method.append(types)
    out["ts_methods"] = len(methods)
    out["ts_param_counts"] = [len(t) for t in per_method]
    types = max(per_method, key=len) if per_method else []
    out["ts_params"] = len(types)
    out["ts_param_distinct_types"] = len({t[2] for t in types})
    out["ts_param_primitive"] = sum(1 for _, k, _ in types if k in PRIMITIVE_TYPES)
    out["ts_param_boolean"] = sum(1 for _, k, txt in types
                                  if k == "boolean_type" or txt in ("Boolean", "java.lang.Boolean"))
    out["ts_param_string"] = sum(1 for _, _, txt in types if txt in ("String", "java.lang.String"))
    out["ts_varargs"] = sum(1 for kind, _, _ in types if kind == "spread_parameter")
    out["ts_param_generic"] = sum(1 for _, _, txt in types if "<" in txt)
    return out


def wilson_self_test() -> None:
    for k, n in ((0, 20), (7, 20), (20, 20), (456, 456)):
        ref = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
        w = wilson(k, n, 0.95)
        assert abs(w["ci_low"] - ref.low) < 1e-9 and abs(w["ci_high"] - ref.high) < 1e-9, (k, n, w, ref)
    print("[self-test] Wilson intervals match scipy")


def permutation_self_test() -> None:
    rng = np.random.default_rng(0)
    groups = [str(v) for v in rng.integers(0, 4, size=60)]
    y = (rng.random(60) < 0.4).astype(int)
    g = np.asarray(groups)
    onehot = np.stack([(g == n).astype(np.float64) for n in sorted(set(groups))], axis=1)
    ref = chi_square(groups, y)["chi2"]
    assert abs(float(chi2_stat(onehot, y)) - ref) < 1e-9, (chi2_stat(onehot, y), ref)
    # permuted samples must keep their size and row order
    samples = np.array([0, 0, 1, 1, 2, 3, 3, 3, 4, 4, 5, 6, 6, 6, 7, 7])
    flags = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1])
    vectors = {s: tuple(flags[samples == s]) for s in set(samples.tolist())}
    for src in block_permutations(samples, 50, rng, chunk=20):
        for row in flags[src]:
            got = {s: tuple(row[samples == s]) for s in vectors}
            for k in (1, 2, 3):
                assert sorted(v for s, v in got.items() if len(v) == k) == \
                    sorted(v for s, v in vectors.items() if len(v) == k)
    # longest runs against a direct scan
    for _ in range(200):
        seq = (rng.random(int(rng.integers(1, 40))) < 0.6).astype(int)
        best, run = {0: 0, 1: 0}, 0
        for i, v in enumerate(seq):
            run = run + 1 if i and v == seq[i - 1] else 1
            best[int(v)] = max(best[int(v)], run)
        assert longest_runs(seq) == best, (seq, longest_runs(seq), best)
    print("[self-test] chi-square statistic matches scipy, block permutations keep sample vectors, "
          "longest runs match a direct scan")


def feature_self_test(prefix: str, suffix: str) -> None:
    snippet = ("public <T> Map<String, List<T>> foo(final int a, boolean b, Boolean c, String s, List<String> l,\n"
               "    int[] arr, Object... rest) {\n"
               "  if (a > 1 && b || !c) { return null; } else if (a < 0) { x = a > 2 ? 1 : 2; } else { y(); }\n"
               "\n"
               "  switch (a) { case 1: case 2: break; default: z(); }\n"
               "  try { for (int i = 0; i < a; i++) { while (true) { } } } catch (Exception e) { }\n"
               "  return null;\n"
               "}\n"
               "public void foo(int a) { Runnable r = new Runnable() { public void run() { } }; }")
    f = method_features(snippet, prefix, suffix)
    expected = {"ts_loc_nonblank": 8, "ts_params": 7, "ts_param_distinct_types": 7, "ts_param_primitive": 2,
                "ts_param_boolean": 2, "ts_param_string": 1, "ts_varargs": 1, "ts_param_generic": 1, "ts_if": 2,
                "ts_else": 2, "ts_case": 2, "ts_catch": 1, "ts_ternary": 1, "ts_bool_ops": 2, "ts_returns": 2,
                "ts_loops": 2, "ts_max_nesting": 3, "ts_methods": 2, "ts_param_counts": [7, 1], "ts_parse_error": 0}
    bad = {k: (f[k], v) for k, v in expected.items() if f[k] != v}
    print(f"[self-test] tree-sitter features {'ok' if not bad else bad}")
    assert not bad


# ------------------------------------------------------------ data

class Data:
    def __init__(self, cfg: dict[str, Any]) -> None:
        icfg = cfg["inputs"]
        self.cfg = cfg
        self.smells = [dict(s) for s in cfg["method_smells"]]
        if [s["key"] for s in self.smells] != list(cfg["label_order"]):
            raise SystemExit("method_smells must follow label_order")
        self.cls_smell = dict(cfg["class_smell"])
        self.methods = json.loads(resolve(icfg["methods_json"]).read_text())
        self.classes = json.loads(resolve(icfg["classes_json"]).read_text())
        self.folds = load_folds(resolve(icfg["folds"]))
        if list(self.folds["meta"]["label_order"]) != list(cfg["label_order"]):
            raise SystemExit("folds label_order differs from the config")
        if list(self.folds["meta"]["json_index"]) != list(range(len(self.methods))):
            raise SystemExit("folds json_index must be 0..n-1 of the method JSON")
        sql = resolve(icfg["sql_dump"])
        self.tables = {t: [dict(zip(cols, row)) for row in load_table(sql, t)] for t, cols in SQL_COLUMNS.items()}
        self.sample = {int(r["id"]): r for r in self.tables["sample"]}
        self.method_metrics = {int(r["id"]): r for r in self.tables["method_metrics"]}
        self.class_metrics = {int(r["id"]): r for r in self.tables["class_metrics"]}
        self.rows = sorted(({"id": int(r["id"]), "sample_id": int(r["sample_id"]),
                             **{f: to_bit(r[f]) for f in FLAGS}} for r in self.tables["annotation"]
                            if r["sample_id"] not in ("NULL", "")), key=lambda r: r["id"])
        self.rows_of: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in self.rows:
            self.rows_of[r["sample_id"]].append(r)
        n = len(self.methods)
        self.n = n
        self.y = np.array([[int(bool(v)) for v in e["y"]] for e in self.methods])
        self.codes = [set(e["designite_smells_at_sampling"]) for e in self.methods]
        self.project = [e["project"] for e in self.methods]
        self.offered = np.array([[s["code"] in c for s in self.smells] for c in self.codes])
        self.designite = {m: np.array([float(e["designite"][m]) for e in self.methods])
                          for m in cfg["features"]["designite"]}
        # annotation order: equal-count bins over every annotation row of the dump
        ids = np.array([r["id"] for r in self.rows])
        n_bins = int(cfg["batches"]["order_bins"])
        self.bin_edges = np.quantile(ids, np.linspace(0, 1, n_bins + 1))
        self.n_bins = n_bins
        for r in self.rows:
            r["bin"] = self.bin_of(r["id"])
        self.check = self._check_votes()

    def bin_of(self, row_id: int) -> int:
        return int(np.searchsorted(self.bin_edges[1:-1], row_id, side="right"))

    def _check_votes(self) -> dict[str, Any]:
        """The votes stored in the JSON files must equal the flags of the dump rows."""
        bad = 0
        for entries, smells in ((self.methods, self.smells), (self.classes, [self.cls_smell])):
            for e in entries:
                rows = self.rows_of.get(int(e["sample_id"]), [])
                for s in smells:
                    lab = e["labels"][s["name"]]
                    if lab["votes_total"] != len(rows) or lab["votes_pos"] != sum(r[s["flag"]] for r in rows):
                        bad += 1
        return {"method_entries": len(self.methods), "class_entries": len(self.classes),
                "vote_mismatches": bad}


# ------------------------------------------------------------ (a) agreement

def units_from_labels(entries: list[dict[str, Any]], name: str, mask: np.ndarray | None = None) -> list[list[int]]:
    out = []
    for i, e in enumerate(entries):
        if mask is not None and not mask[i]:
            continue
        lab = e["labels"][name]
        out.append([1] * int(lab["votes_pos"]) + [0] * int(lab["votes_total"] - lab["votes_pos"]))
    return out


def describe_units(units: list[list[int]], n_boot: int, level: float, rng: np.random.Generator) -> dict[str, Any]:
    multi = [u for u in units if len(u) >= 2]
    flat = [v for u in multi for v in u]
    out = alpha_with_ci(multi, 2, "nominal", n_boot, level, rng)
    out.update({
        "samples": len(units), "units_multi": len(multi),
        "unanimous_share": float(np.mean([len(set(u)) == 1 for u in multi])) if multi else None,
        "row_positive_rate": float(np.mean(flat)) if flat else None,
        "units_with_any_positive": float(np.mean([max(u) == 1 for u in multi])) if multi else None,
        "annotations_per_unit": {str(k): v for k, v in sorted(Counter(len(u) for u in multi).items())},
    })
    return out


def mlcq_units(cfg: dict[str, Any]) -> dict[str, list[list[int]]]:
    mcfg = cfg["mlcq"]
    order, positive = list(mcfg["severity_order"]), set(mcfg["positive"])
    with resolve(cfg["inputs"]["mlcq_sample_meta_csv"]).open() as fh:
        meta = list(csv.DictReader(fh))
    out = {}
    for s in mcfg["short_names"]:
        units = []
        for r in meta:
            if int(r[f"n_reviews_{s}"]) < 2:
                continue
            units.append([int(sev in positive) for sev in order for _ in range(int(r[f"votes_{s}_{sev}"]))])
        out[s] = units
    return out


SENSITIVITY = {
    "flag_and_is_smell": ("all samples of the main estimate; the smell's flag of a row is set to 0 when its "
                          "is_smell is 0 (trusts is_smell)"),
    "without_samples_in_affected_bins": ("whole samples dropped when one of their rows lies in an affected bin "
                                         "of the smell, whatever their flags (selection on the annotation "
                                         "period, not on the answer)"),
    "without_samples_with_inconsistent_rows": ("whole samples dropped when one of their rows is inconsistent "
                                               "for the smell; such a row mostly needs the flag set, so this "
                                               "selection depends on the answer and removes disagreeing pairs "
                                               "by construction"),
}


def sensitivity(d: Data, entries: list[dict[str, Any]], mask: np.ndarray, smell: dict[str, Any],
                checks: dict[str, Any], n_boot: int, level: float, rng: np.random.Generator) -> dict[str, Any]:
    """Alpha of a smell under the three treatments of the inconsistent rows."""
    sids = [int(e["sample_id"]) for e, m in zip(entries, mask) if m]
    flag, bins = smell["flag"], set(checks["affected_bins"][smell["short"]])
    bad = checks["samples"][smell["short"]]
    variants = {
        "flag_and_is_smell": [[r[flag] * r["is_smell"] for r in d.rows_of[s]] for s in sids],
        "without_samples_in_affected_bins": [[r[flag] for r in d.rows_of[s]] for s in sids
                                             if not any(r["bin"] in bins for r in d.rows_of[s])],
        "without_samples_with_inconsistent_rows": [[r[flag] for r in d.rows_of[s]] for s in sids if s not in bad],
    }
    out: dict[str, Any] = {"affected_bins": sorted(bins)}
    for name, units in variants.items():
        out[name] = {"selection": SENSITIVITY[name], **describe_units(units, n_boot, level, rng)}
    return out


def agreement(d: Data, rng: np.random.Generator, checks: dict[str, Any]) -> dict[str, Any]:
    acfg = d.cfg["agreement"]
    n_boot, level = int(acfg["bootstrap"]), float(acfg["ci"])
    out: dict[str, Any] = {
        "method": ("Krippendorff (2011) alpha, nominal, coincidence matrix over samples with >= 2 annotation "
                   f"rows; {int(level * 100)}% percentile bootstrap over samples, {n_boot} resamples; "
                   "implementation imported from scripts/labels/analyze_mlcq.py (self test passed in this run)"),
        "observed_agreement": "pairwise agreement of the coincidence matrix, trace(o) / n",
        "kappa_note": ("The annotation table of the dump has no user_id column (the Tagman model has one, the "
                       "released dump drops it), so the annotators of a sample are unknown and Cohen kappa per "
                       "rater pair is not defined; Fleiss kappa and Krippendorff alpha only need the values per "
                       "sample."),
        "all_method_snippets_note": ("includes the samples whose code did not offer the smell; "
                                     "positive_rows_not_offered counts the rows of those samples that set the "
                                     "smell's flag"),
        "sensitivity_note": SENSITIVITY,
        "dacos": {}, "mlcq": {},
    }
    for k, s in enumerate(d.smells):
        not_offered = [e for e, m in zip(d.methods, d.offered[:, k]) if not m]
        out["dacos"][s["short"]] = {
            "name": s["name"],
            "all_method_snippets": {
                **describe_units(units_from_labels(d.methods, s["name"]), n_boot, level, rng),
                "samples_not_offered": len(not_offered),
                "positive_rows_not_offered": int(sum(e["labels"][s["name"]]["votes_pos"] for e in not_offered)),
                "positive_samples_not_offered": int(sum(bool(e["labels"][s["name"]]["present"])
                                                        for e in not_offered))},
            "offered": describe_units(units_from_labels(d.methods, s["name"], d.offered[:, k]), n_boot, level, rng),
            "offered_sensitivity": sensitivity(d, d.methods, d.offered[:, k], s, checks, n_boot, level, rng),
        }
    cs = d.cls_smell
    out["dacos"][cs["short"]] = {
        "name": cs["name"],
        "all_class_snippets": describe_units(units_from_labels(d.classes, cs["name"]), n_boot, level, rng),
        "sensitivity": sensitivity(d, d.classes, np.ones(len(d.classes), dtype=bool), cs, checks, n_boot, level,
                                   rng),
    }
    mcfg = d.cfg["mlcq"]
    for s, name in zip(mcfg["short_names"], mcfg["smell_names"]):
        out["mlcq"][s] = {"name": name, **describe_units(mlcq_units(d.cfg)[s], n_boot, level, rng)}
    note = ("MLCQ units with >= 2 reviews are a selected subset: most re-reviews come from the last sampling "
            "phase, which re-reviewed samples already rated above none, so they are enriched in positives and "
            "in disputed samples; every DACOS sample was sent to two annotators by design.")
    rpath = resolve(d.cfg["inputs"].get("mlcq_reviewers_summary", ""))
    if rpath.is_file():
        phases = json.loads(rpath.read_text()).get("phases", {}).get("phases", [])
        out["mlcq_phases"] = [{k: p.get(k) for k in ("phase", "description", "reviews", "samples",
                                                      "later_session_reviews")} for p in phases]
    out["mlcq_selection_note"] = note
    return out


# ------------------------------------------------------------ (b) sampling design

def in_window(value: float, rule: dict[str, Any]) -> bool:
    lo, hi = float(rule["low"]), float(rule["high"])
    ok_lo = value >= lo if rule["low_inclusive"] else value > lo
    ok_hi = value <= hi if rule["high_inclusive"] else value < hi
    return ok_lo and ok_hi


def codes_by_rule(metrics: dict[str, float], rules: dict[str, Any], codes: list[str]) -> set[str]:
    return {c for c in codes if in_window(metrics[rules[c]["metric"]], rules[c])}


def sampling(d: Data) -> dict[str, Any]:
    scfg = d.cfg["sampling"]
    quantiles = [float(q) for q in scfg["quantiles"]]
    annotated = set(d.rows_of)
    out: dict[str, Any] = {"quotes": scfg["quotes"], "code_rule": scfg["code_rule"], "paper_rule": scfg["paper_rule"]}

    # population of the dump and the codes the rules would give
    pop: Counter = Counter()
    rule_check: dict[str, Counter] = {"code_rule": Counter(), "paper_rule": Counter()}
    # samples inside the window of a rule: (level, annotated) -> count
    rule_window: dict[str, Counter] = {"code_rule": Counter(), "paper_rule": Counter()}
    # mismatches of the code rule: (level, annotated, stored, rule) -> metric values
    mismatches: dict[tuple[str, bool, str, str], Counter] = defaultdict(Counter)
    class_lcom: dict[str, list[float]] = {"coded": [], "uncoded": []}
    for s in d.tables["sample"]:
        is_class = to_bit(s["is_class"])
        level = "class" if is_class else "method"
        stored = {c for c in s["smells"].split(",") if c}
        sid = int(s["id"])
        pop[(level, s["smells"], sid in annotated)] += 1
        if s["designite_id"] in ("NULL", ""):
            continue
        if is_class:
            m = d.class_metrics.get(int(s["designite_id"]))
            if m is None:
                continue
            metrics, codes = {"lcom": float(m["lcom"])}, ["3"]
            class_lcom["coded" if stored else "uncoded"].append(metrics["lcom"])
        else:
            m = d.method_metrics.get(int(s["designite_id"]))
            if m is None:
                continue
            metrics, codes = {"cc": float(m["cc"]), "pc": float(m["pc"])}, ["1", "2"]
        for name in rule_check:
            got = codes_by_rule(metrics, scfg[name], codes)
            key = "match" if got == stored else "mismatch"
            rule_check[name][(level, key, sid in annotated)] += 1
            if got:
                rule_window[name][(level, sid in annotated)] += 1
            if name == "code_rule" and got != stored:
                key2 = (level, sid in annotated, ",".join(sorted(stored)), ",".join(sorted(got)))
                mismatches[key2][" ".join(f"{k}={v:g}" for k, v in metrics.items())] += 1
    out["population"] = [{"level": lv, "codes": c, "annotated": a, "samples": n}
                         for (lv, c, a), n in sorted(pop.items())]
    out["rule_check"] = {}
    for name, cnt in rule_check.items():
        entry = {}
        for lv in ("method", "class"):
            for scope, flag in (("all", None), ("annotated", True)):
                m = sum(v for (l2, key, a), v in cnt.items() if l2 == lv and key == "match" and (flag is None or a))
                t = sum(v for (l2, key, a), v in cnt.items() if l2 == lv and (flag is None or a))
                entry[f"{lv}_{scope}"] = {"match": m, "total": t, "rate": m / t if t else None}
        out["rule_check"][name] = entry
    out["code_rule_mismatches"] = [
        {"level": lv, "annotated": a, "stored": st or "none", "rule": rl or "none", "samples": sum(c.values()),
         "metric_values": dict(c.most_common(12))}
        for (lv, a, st, rl), c in sorted(mismatches.items(), key=lambda kv: -sum(kv[1].values()))]

    # windowed snippets annotated (census or sample of the window)
    windowed: Counter = Counter()
    for (lv, c, a), n in pop.items():
        if c:
            windowed[(lv, a)] += n
    out["window_annotated_share"] = {lv: {"annotated": windowed[(lv, True)],
                                          "windowed": windowed[(lv, True)] + windowed[(lv, False)]}
                                     for lv in ("method", "class")}
    # the same with the window of each rule instead of the stored codes
    out["rule_window_annotated_share"] = {
        name: {lv: {"annotated": cnt[(lv, True)], "in_window": cnt[(lv, True)] + cnt[(lv, False)]}
               for lv in ("method", "class")}
        for name, cnt in rule_window.items()}
    out["class_lcom"] = {}
    for key, values in class_lcom.items():
        v = np.array(values)
        out["class_lcom"][key] = {
            **summary(v, quantiles),
            **{f"in_{name}_window": int(sum(in_window(x, scfg[name]["3"]) for x in v)) for name in rule_window},
            "most_common": {f"{x:g}": c for x, c in Counter(v.tolist()).most_common(5)}}

    # metric distributions per code and per sample_constraints value (method JSON)
    metrics = list(scfg["metrics"])
    code_key = [",".join(sorted(c)) for c in d.codes]
    con = [int(e["sample_constraints"]) for e in d.methods]
    per_code = {}
    for key in sorted(set(code_key)):
        idx = [i for i, c in enumerate(code_key) if c == key]
        per_code[key or "none"] = {
            "samples": len(idx),
            **{m: summary(d.designite[m][idx], quantiles) for m in metrics},
            "cc_values": {str(int(v)): c for v, c in sorted(Counter(d.designite["cc"][idx]).items())},
            "pc_values": {str(int(v)): c for v, c in sorted(Counter(d.designite["pc"][idx]).items())},
            "positives": {s["short"]: int(d.y[idx, k].sum()) for k, s in enumerate(d.smells)},
            "sample_constraints": {str(v): c for v, c in sorted(Counter(con[i] for i in idx).items())},
        }
    out["per_code"] = per_code
    per_con = {}
    for v in sorted(set(con)):
        idx = [i for i, c in enumerate(con) if c == v]
        row_ids = [r["id"] for i in idx for r in d.rows_of[int(d.methods[i]["sample_id"])]]
        per_con[str(v)] = {
            "samples": len(idx),
            "annotations_per_sample": {str(k): c for k, c in
                                       sorted(Counter(d.methods[i]["n_annotations"] for i in idx).items())},
            "codes": {k or "none": c for k, c in sorted(Counter(code_key[i] for i in idx).items())},
            **{m: summary(d.designite[m][idx], quantiles) for m in metrics},
            "annotation_id_range": [min(row_ids), max(row_ids)] if row_ids else None,
            "annotation_bins": {str(b): c for b, c in sorted(Counter(d.bin_of(r) for r in row_ids).items())},
        }
    out["per_sample_constraints"] = per_con
    cls_con = Counter((int(e["sample_constraints"]), int(e["n_annotations"])) for e in d.classes)
    out["class_constraints_vs_annotations"] = [{"sample_constraints": a, "annotations": b, "samples": n}
                                               for (a, b), n in sorted(cls_con.items())]
    out["method_constraints_vs_annotations"] = [
        {"sample_constraints": a, "annotations": b, "samples": n}
        for (a, b), n in sorted(Counter((int(e["sample_constraints"]), int(e["n_annotations"]))
                                        for e in d.methods).items())]

    # Table II of the paper next to the dump
    dump_counts: dict[str, dict[str, int]] = {}
    method_rows = [(set(d.sample[sid]["smells"].split(",")) - {""}, rows) for sid, rows in d.rows_of.items()
                   if not to_bit(d.sample[sid]["is_class"])]
    class_rows = [rows for sid, rows in d.rows_of.items() if to_bit(d.sample[sid]["is_class"])]
    for s in d.smells:
        sel = [rows for codes, rows in method_rows if s["code"] in codes]
        dump_counts[s["name"]] = {"samples_offered": len(sel), "annotations_offered": sum(len(r) for r in sel)}
    dump_counts["method without code"] = {"samples": sum(1 for c, _ in method_rows if not c),
                                          "annotations": sum(len(r) for c, r in method_rows if not c)}
    dump_counts[d.cls_smell["name"]] = {"samples": len(class_rows), "annotations": sum(len(r) for r in class_rows)}
    dump_counts["total"] = {"samples": len(d.rows_of), "annotations": len(d.rows)}
    out["paper_table_ii"] = scfg["paper_counts"]
    out["dump_counts"] = dump_counts
    ids = [r["id"] for r in d.rows]
    # Tagman serves rows in table order, so annotation order should follow it
    rho = stats.spearmanr(ids, [r["sample_id"] for r in d.rows])
    out["annotation_ids"] = {"min": min(ids), "max": max(ids), "rows": len(ids),
                             "missing_ids_in_range": max(ids) - min(ids) + 1 - len(ids),
                             "spearman_with_sample_id": {"rho": float(rho.statistic), "p_value": float(rho.pvalue)}}
    return out


def metric_attachment(d: Data, ts_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Whether the Designite metrics attached to a snippet describe the code shown (overloads)."""
    names = Counter((r["project_name"], r["package_name"], r["type_name"], r["method_name"])
                    for r in d.tables["method_metrics"])
    tab: Counter = Counter()
    for i, e in enumerate(d.methods):
        m = d.method_metrics[int(d.sample[int(e["sample_id"])]["designite_id"])]
        shared = names[(m["project_name"], m["package_name"], m["type_name"], m["method_name"])] > 1
        counts, pc = ts_rows[i]["ts_param_counts"], int(float(m["pc"]))
        match = "largest" if counts and pc == max(counts) else ("other_method" if pc in counts else "none")
        tab[(shared, len(counts) > 1, match)] += 1
    return {"snippets": d.n,
            "designite_name_shared_by_several_rows": sum(v for (s, _, _), v in tab.items() if s),
            "snippets_with_several_methods": sum(v for (_, many, _), v in tab.items() if many),
            "pc_equals_largest_param_count": sum(v for (_, _, k), v in tab.items() if k == "largest"),
            "pc_equals_param_count_of_another_method": sum(v for (_, _, k), v in tab.items() if k == "other_method"),
            "pc_equals_no_param_count": sum(v for (_, _, k), v in tab.items() if k == "none"),
            "table": [{"designite_name_shared": s, "several_methods": many, "pc_match": k, "snippets": v}
                      for (s, many, k), v in sorted(tab.items())]}


# ------------------------------------------------------------ consistency with the public Tagman code

def consistency(d: Data) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rows whose flags the public Tagman code cannot produce.

    Returns the report and, per smell, the samples with an inconsistent row and the affected bins."""
    ui = d.cfg["ui_labels"]
    share = float(d.cfg["agreement"]["affected_bin_share"])
    all_smells = d.smells + [d.cls_smell]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bad_samples: set[int] = set()
    smell_samples: dict[str, set[int]] = {sm["short"]: set() for sm in all_smells}
    # per smell and bin: rows of samples offered the smell, inconsistent ones among them
    offered_rows: dict[str, Counter] = {sm["short"]: Counter() for sm in all_smells}
    offered_bad: dict[str, Counter] = {sm["short"]: Counter() for sm in all_smells}
    for r in d.rows:
        s = d.sample[r["sample_id"]]
        codes = {c for c in s["smells"].split(",") if c}
        is_class = to_bit(s["is_class"])
        level_smells = [d.cls_smell] if is_class else d.smells
        any_flag = int(any(r[sm["flag"]] for sm in level_smells))
        if not is_class:
            for sm in d.smells:
                if r[sm["flag"]] and sm["code"] not in codes:
                    groups[f"{sm['short']}_flag_without_code"].append(r)
            if not codes:
                groups["method_sample_without_code"].append(r)
        if r["is_smell"] != any_flag:
            groups[f"{'class' if is_class else 'method'}_is_smell_differs_from_flags"].append(r)
        for sm in level_smells:
            offered = sm["code"] in codes
            bad = bool((r[sm["flag"]] and not r["is_smell"]) or (r[sm["flag"]] and not offered)
                       or (offered and r["is_smell"] and not any_flag))
            if bad:
                smell_samples[sm["short"]].add(r["sample_id"])
            if offered:
                offered_rows[sm["short"]][r["bin"]] += 1
                offered_bad[sm["short"]][r["bin"]] += int(bad)
    affected = {k: sorted(b for b, n in offered_rows[k].items() if n and offered_bad[k][b] / n >= share)
                for k in offered_rows}
    out: dict[str, Any] = {"ui_labels": ui}
    out["class_rows_with_isma"] = sum(r["isma"] for r in d.rows if to_bit(d.sample[r["sample_id"]]["is_class"]))
    out["by_smell"] = {
        k: {"samples_with_inconsistent_rows": len(smell_samples[k]),
            "inconsistent_rows_of_offered_samples": int(sum(offered_bad[k].values())),
            "affected_bins": affected[k], "affected_bin_share": share,
            "share_by_bin": {str(b): offered_bad[k][b] / n for b, n in sorted(offered_rows[k].items())}}
        for k in offered_rows}
    for name, rows in sorted(groups.items()):
        bad_samples |= {r["sample_id"] for r in rows}
        mm = [d.method_metrics.get(int(d.sample[r["sample_id"]]["designite_id"])) for r in rows
              if not to_bit(d.sample[r["sample_id"]]["is_class"])]
        out[name] = {
            "rows": len(rows), "samples": len({r["sample_id"] for r in rows}),
            "id_range": [rows[0]["id"], rows[-1]["id"]],
            "bins": {str(b): c for b, c in sorted(Counter(r["bin"] for r in rows).items())},
            "flag_patterns": {f"cm{r['iscm']}_lpl{r['islp']}_ma{r['isma']}_smell{r['is_smell']}": c
                              for r, c in _pattern_counts(rows)},
            "projects": dict(Counter(d.sample[r["sample_id"]]["project_name"] for r in rows).most_common()),
        }
        if mm:
            for metric in ("pc", "cc"):
                values = Counter(int(float(m[metric])) for m in mm if m)
                out[name][f"{metric}_values"] = {str(k): v for k, v in sorted(values.items())}
    # isma along the order for class rows
    cls_rows = [r for r in d.rows if to_bit(d.sample[r["sample_id"]]["is_class"])]
    out["class_rows_by_bin"] = {str(b): {"rows": len(rs), "isma": sum(r["isma"] for r in rs),
                                         "is_smell": sum(r["is_smell"] for r in rs)}
                                for b, rs in sorted(_by_bin(cls_rows).items())}
    # flag patterns of method rows per bin, grouped by stored code
    scfg = d.cfg["sampling"]
    by_bin: dict[int, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for r in d.rows:
        s = d.sample[r["sample_id"]]
        if to_bit(s["is_class"]):
            continue
        m = d.method_metrics[int(s["designite_id"])]
        stored = {c for c in s["smells"].split(",") if c}
        paper = codes_by_rule({"cc": float(m["cc"]), "pc": float(m["pc"])}, scfg["paper_rule"],
                              [sm["code"] for sm in d.smells])
        group = (s["smells"] or "none") + "".join(f"+paper{c}" for c in sorted(paper - stored))
        by_bin[r["bin"]][group][f"cm{r['iscm']}_lpl{r['islp']}_smell{r['is_smell']}"] += 1
    out["method_patterns_by_bin"] = {str(b): {g: dict(c.most_common()) for g, c in sorted(gs.items())}
                                     for b, gs in sorted(by_bin.items())}
    out["samples_with_inconsistent_rows"] = len(bad_samples)
    return out, {"samples": smell_samples, "affected_bins": affected}


def longest_runs(values: np.ndarray) -> dict[int, int]:
    """Longest run of consecutive equal values, per value (0 and 1)."""
    v = np.asarray(values)
    if len(v) == 0:
        return {0: 0, 1: 0}
    change = np.flatnonzero(np.diff(v)) + 1
    starts = np.r_[0, change]
    lengths = np.r_[change, len(v)] - starts
    kinds = v[starts]
    return {k: int(lengths[kinds == k].max()) if (kinds == k).any() else 0 for k in (0, 1)}


def runs_test(values: np.ndarray, samples: np.ndarray, n_perm: int, quantile: float,
              rng: np.random.Generator) -> dict[str, Any]:
    """Longest runs of 1 and of 0 in annotation order against block permutations
    of the same rows (the rows of a sample move together)."""
    obs = longest_runs(values)
    perm = [longest_runs(row) for src in block_permutations(samples, n_perm, rng) for row in values[src]]
    out = {"rows": int(len(values)), "positives": int(values.sum()), "permutations": n_perm,
           "null": "block permutations of whole samples", "quantile": quantile}
    for v in (0, 1):
        null = np.array([p[v] for p in perm])
        out[f"longest_run_{v}"] = obs[v]
        out[f"longest_run_{v}_perm_quantile"] = float(np.quantile(null, quantile))
        out[f"longest_run_{v}_perm_max"] = int(null.max())
        out[f"longest_run_{v}_p_value"] = float((1 + (null >= obs[v]).sum()) / (1 + n_perm))
    return out


def _pattern_counts(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], int]]:
    cnt = Counter((r["iscm"], r["islp"], r["isma"], r["is_smell"]) for r in rows)
    return [({"iscm": a, "islp": b, "isma": c, "is_smell": e}, n) for (a, b, c, e), n in cnt.most_common()]


def _by_bin(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[r["bin"]].append(r)
    return out


# ------------------------------------------------------------ (c) what explains the label

def features_matrix(d: Data, ts_rows: list[dict[str, int]]) -> dict[str, Any]:
    fcfg = d.cfg["features"]
    blocks = {"designite": (np.stack([d.designite[m] for m in fcfg["designite"]], axis=1), list(fcfg["designite"])),
              "tree_sitter": (np.array([[r[c] for c in fcfg["tree_sitter"]] for r in ts_rows], dtype=float),
                              list(fcfg["tree_sitter"]))}
    proj, pnames = dummies(d.project, drop_first=False)
    blocks["project"] = (proj, [f"project={p}" for p in pnames])
    period = np.zeros((d.n, d.n_bins))
    for i, e in enumerate(d.methods):
        for r in d.rows_of[int(e["sample_id"])]:
            period[i, r["bin"]] += 1
    blocks["period"] = (period, [f"period_bin={b}" for b in range(d.n_bins)])
    return blocks


def build(blocks: dict[str, Any], names: list[str], smell: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    cols, labels = [], []
    for name in names:
        if name == "metric":
            x, lab = blocks["designite"]
            j = lab.index(smell["metric"])
            cols.append(x[:, [j]])
            labels.append(lab[j])
        else:
            x, lab = blocks[name]
            cols.append(x)
            labels += lab
    return np.hstack(cols), labels


def rate_by_value(values: np.ndarray, y: np.ndarray, rows_pos: np.ndarray, rows_tot: np.ndarray,
                  level: float) -> list[dict[str, Any]]:
    out = []
    for v in sorted(set(values.tolist())):
        m = values == v
        out.append({"value": float(v), "samples": wilson(int(y[m].sum()), int(m.sum()), level),
                    "rows": wilson(int(rows_pos[m].sum()), int(rows_tot[m].sum()), level)})
    return out


def cv_logistic(x: np.ndarray, y: np.ndarray, subset: np.ndarray, d: Data, ecfg: dict[str, Any]) -> dict[str, Any]:
    c_grid = [float(c) for c in ecfg["c_grid"]]
    runs = []
    for f in iter_folds(d.folds, strategies=[str(ecfg["cv_strategy"])], seeds=[int(s) for s in ecfg["cv_seeds"]]):
        tr, va, te = (idx[subset[idx]] for idx in (f.train, f.val, f.test))
        scaler = StandardScaler().fit(x[tr])
        xtr, xva, xte = scaler.transform(x[tr]), scaler.transform(x[va]), scaler.transform(x[te])
        best = None
        for c in c_grid:
            # fixed state: liblinear shuffles its coordinates
            model = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=c, max_iter=int(ecfg["max_iter"]),
                                       random_state=int(d.cfg["seed"]))
            model.fit(xtr, y[tr])
            auc = safe_auc(y[va], model.decision_function(xva))
            if auc is not None and (best is None or auc > best[0]):
                best = (auc, c, model)
        test_auc = safe_auc(y[te], best[2].decision_function(xte))
        runs.append({"seed": f.seed, "fold": f.fold, "c": best[1], "val_auc": best[0], "test_auc": test_auc,
                     "n_test": int(len(te)), "nonzero": int((best[2].coef_ != 0).sum())})
    aucs = np.array([r["test_auc"] for r in runs], dtype=float)
    seeds = sorted({r["seed"] for r in runs})
    seed_means = np.array([np.mean([r["test_auc"] for r in runs if r["seed"] == s]) for s in seeds])
    return {"runs": len(runs), "auc_mean": float(aucs.mean()), "auc_sd": float(aucs.std(ddof=1)),
            "auc_seed_mean": float(seed_means.mean()), "auc_seed_sd": float(seed_means.std(ddof=1)),
            "c_chosen": {str(k): v for k, v in sorted(Counter(r["c"] for r in runs).items())},
            "nonzero_mean": float(np.mean([r["nonzero"] for r in runs])), "per_run": runs}


def explain(d: Data, blocks: dict[str, Any]) -> dict[str, Any]:
    ecfg = d.cfg["explain"]
    level = float(ecfg["ci"])
    out: dict[str, Any] = {"label": "y of the method JSON (at least half of the annotators; with two, at least one)"}
    for k, s in enumerate(d.smells):
        name = s["name"]
        y = d.y[:, k]
        rows_pos = np.array([e["labels"][name]["votes_pos"] for e in d.methods])
        rows_tot = np.array([e["labels"][name]["votes_total"] for e in d.methods])
        off = d.offered[:, k]
        metric = d.designite[s["metric"]]
        entry: dict[str, Any] = {
            "name": name, "metric": s["metric"], "offered_samples": int(off.sum()),
            "offered_positives": int(y[off].sum()), "not_offered_positives": int(y[~off].sum()),
            "rate_by_metric_offered": rate_by_value(metric[off], y[off], rows_pos[off], rows_tot[off], level),
            "rate_by_metric_not_offered": rate_by_value(metric[~off], y[~off], rows_pos[~off], rows_tot[~off], level),
        }
        # univariate ROC-AUC, offered snippets and all method snippets
        uni = {}
        for block in ("designite", "tree_sitter"):
            x, labels = blocks[block]
            for j, lab in enumerate(labels):
                uni[lab] = {"offered": safe_auc(y[off], x[off, j]), "all": safe_auc(y, x[:, j])}
        entry["univariate_auc"] = uni
        x, labels = blocks["tree_sitter"]
        entry["rate_by_tree_sitter_offered"] = {
            lab: [{"value": p["value"], **p["samples"]}
                  for p in rate_by_value(x[off, j], y[off], rows_pos[off], rows_tot[off], level)]
            for j, lab in enumerate(labels)}
        # L1 logistic regression on the kfold folds, offered snippets
        cv = {}
        for fs, names in ecfg["feature_sets"].items():
            x, labels = build(blocks, list(names), s)
            res = cv_logistic(x, y, off, d, ecfg)
            res["features"] = len(labels)
            cv[fs] = res
            print(f"  [{s['short']}] {fs:<26} ROC-AUC {res['auc_mean']:.3f} +- {res['auc_sd']:.3f} "
                  f"(seed means {res['auc_seed_mean']:.3f} +- {res['auc_seed_sd']:.3f}, {len(labels)} features)")
        entry["cv_l1_logistic"] = cv
        # coefficients of one fit on all offered samples
        fs = str(ecfg["coefficients_of"])
        x, labels = build(blocks, list(ecfg["feature_sets"][fs]), s)
        c_counts = Counter({float(c): n for c, n in cv[fs]["c_chosen"].items()})
        c = max(c_counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        xs = StandardScaler().fit_transform(x[off])
        model = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=c, max_iter=int(ecfg["max_iter"]),
                                   random_state=int(d.cfg["seed"]))
        model.fit(xs, y[off])
        coefs = sorted(zip(labels, model.coef_[0].tolist()), key=lambda kv: -abs(kv[1]))
        entry["coefficients"] = {"feature_set": fs, "c": c, "intercept": float(model.intercept_[0]),
                                 "standardized": [{"feature": f, "coef": v} for f, v in coefs],
                                 "nonzero": int(sum(1 for _, v in coefs if v != 0))}
        out[s["short"]] = entry
    return out


# ------------------------------------------------------------ (d) projects and annotation order

def batches(d: Data, blocks: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    bcfg, ecfg = d.cfg["batches"], d.cfg["explain"]
    level = float(ecfg["ci"])
    c, it = float(bcfg["lrt_c"]), int(bcfg["lrt_max_iter"])
    n_perm, run_q = int(bcfg["run_permutations"]), float(bcfg["run_null_quantile"])
    n_block = int(bcfg["chi_square_block_permutations"])
    mixed_lo, mixed_hi = (float(v) for v in bcfg["mixed_bin_rate"])
    raw_x, raw_names = build(blocks, ["designite", "tree_sitter"], d.smells[0])
    code_x = StandardScaler().fit_transform(raw_x)
    out: dict[str, Any] = {"bins": []}
    for b in range(d.n_bins):
        rows = [r for r in d.rows if r["bin"] == b]
        out["bins"].append({"bin": b, "id_low": rows[0]["id"], "id_high": rows[-1]["id"], "rows": len(rows),
                            "projects": dict(Counter(d.sample[r["sample_id"]]["project_name"]
                                                     for r in rows).most_common())})
    index_of = {int(e["sample_id"]): i for i, e in enumerate(d.methods)}
    projects = sorted(set(d.project))
    for k, s in enumerate(d.smells):
        off = d.offered[:, k]
        y = d.y[:, k]
        entry: dict[str, Any] = {"name": s["name"]}
        entry["project_prevalence"] = {p: wilson(int(y[off & (np.array(d.project) == p)].sum()),
                                                 int((off & (np.array(d.project) == p)).sum()), level)
                                       for p in projects}
        idx = np.where(off)[0]
        proj_labels = [d.project[i] for i in idx]
        # one value per sample: each sample is its own block
        entry["project_chi_square"] = chi_square_blocks(proj_labels, y[idx], np.arange(len(idx)), n_block, rng)
        pd_, _ = dummies(proj_labels)
        entry["project_lrt_given_code"] = lrt(code_x[idx], pd_, y[idx], c, it)
        # annotation rows of offered samples
        rows = [r for r in d.rows if r["sample_id"] in index_of and off[index_of[r["sample_id"]]]]
        ry = np.array([r[s["flag"]] for r in rows])
        rsid = np.array([r["sample_id"] for r in rows])
        rbin = [str(r["bin"]) for r in rows]
        rproj = [d.sample[r["sample_id"]]["project_name"] for r in rows]
        rx = code_x[[index_of[r["sample_id"]] for r in rows]]
        entry["order_prevalence"] = {str(b): wilson(int(ry[np.array(rbin) == str(b)].sum()),
                                                    int((np.array(rbin) == str(b)).sum()), level)
                                     for b in range(d.n_bins)}
        entry["order_chi_square"] = chi_square_blocks(rbin, ry, rsid, n_block, rng)
        entry["order_spearman"] = dict(zip(("rho", "p_value"), map(float, stats.spearmanr([r["id"] for r in rows],
                                                                                              ry))))
        bd, _ = dummies(rbin)
        pdr, _ = dummies(rproj)
        entry["row_lrt"] = {
            "period_given_code": lrt(rx, bd, ry, c, it),
            "project_given_code": lrt(rx, pdr, ry, c, it),
            "period_given_code_project": lrt(np.hstack([rx, pdr]), bd, ry, c, it),
            "project_given_code_period": lrt(np.hstack([rx, bd]), pdr, ry, c, it),
        }
        # period effect inside each project that spans several bins
        within = {}
        for p in projects:
            m = np.array(rproj) == p
            if len(set(np.array(rbin)[m])) >= 2:
                within[p] = {"bins": len(set(np.array(rbin)[m])), "samples": len(set(rsid[m].tolist())),
                             **chi_square_blocks(list(np.array(rbin)[m]), ry[m], rsid[m], n_block, rng)}
        entry["order_chi_square_within_project"] = within
        entry["order_runs"] = runs_test(ry, rsid, n_perm, run_q, rng)
        # row-level ROC-AUC of the code features in bins without a near-constant rate
        rates = {b: w["rate"] for b, w in entry["order_prevalence"].items() if w["n"]}
        mixed = sorted(b for b, r in rates.items() if mixed_lo <= r <= mixed_hi)
        mm = np.isin(np.array(rbin), mixed)
        rraw = raw_x[[index_of[r["sample_id"]] for r in rows]]
        auc_mixed = {}
        for j, name in enumerate(raw_names):
            per_bin = [(safe_auc(ry[np.array(rbin) == b], rraw[np.array(rbin) == b, j]),
                        int((np.array(rbin) == b).sum())) for b in mixed]
            per_bin = [(a, n) for a, n in per_bin if a is not None]
            auc_mixed[name] = {"pooled": safe_auc(ry[mm], rraw[mm, j]),
                               "within_bin_mean": (float(np.average([a for a, _ in per_bin],
                                                                    weights=[n for _, n in per_bin]))
                                                   if per_bin else None)}
        entry["row_auc_mixed_bins"] = {"bins": mixed, "rows": int(mm.sum()), "positives": int(ry[mm].sum()),
                                       "auc": auc_mixed}
        out[s["short"]] = entry
        print(f"  [{s['short']}] project chi2 p={entry['project_chi_square']['p_value']:.3g}, "
              f"order chi2 p={entry['order_chi_square']['p_value']:.3g} "
              f"(block permutation p={entry['order_chi_square']['block_permutation_p_value']:.3g}), "
              f"LRT period|code+project p={entry['row_lrt']['period_given_code_project']['p_value']:.3g}")
    # multifaceted abstraction: class snippets, no code features
    cs = d.cls_smell
    cy = np.array([int(bool(e["labels"][cs["name"]]["present"])) for e in d.classes])
    cproj = [e["project"] for e in d.classes]
    rows = [r for r in d.rows if to_bit(d.sample[r["sample_id"]]["is_class"])]
    ry = np.array([r[cs["flag"]] for r in rows])
    rsid = np.array([r["sample_id"] for r in rows])
    rbin = [str(r["bin"]) for r in rows]
    out[cs["short"]] = {
        "name": cs["name"],
        "project_prevalence": {p: wilson(int(cy[np.array(cproj) == p].sum()), int((np.array(cproj) == p).sum()), level)
                               for p in sorted(set(cproj))},
        "project_chi_square": chi_square_blocks(cproj, cy, np.arange(len(cy)), n_block, rng),
        "order_prevalence": {str(b): wilson(int(ry[np.array(rbin) == str(b)].sum()),
                                            int((np.array(rbin) == str(b)).sum()), level) for b in range(d.n_bins)},
        "order_chi_square": chi_square_blocks(rbin, ry, rsid, n_block, rng),
        "order_runs": runs_test(ry, rsid, n_perm, run_q, rng),
    }
    return out


# ------------------------------------------------------------ figures

class Figures:
    def __init__(self, directory: Path, prefix: str, dpi: int) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt, self.dir, self.prefix, self.dpi = plt, directory, prefix, dpi
        self.written: list[str] = []
        directory.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({
            "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
            "xtick.labelsize": 7, "ytick.labelsize": 7,
            "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK, "text.color": INK,
            "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
            "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
        })

    def save(self, fig: Any, name: str) -> None:
        for ext in ("pdf", "png"):
            path = self.dir / f"{self.prefix}{name}.{ext}"
            tmp = tmp_path(path)
            fig.savefig(tmp, format=ext, dpi=self.dpi)
            tmp.replace(path)
            self.written.append(rel(path))
        self.plt.close(fig)


def fig_rate_vs_metric(d: Data, ex: dict[str, Any], figs: Figures) -> None:
    ecfg, scfg = d.cfg["explain"], d.cfg["sampling"]
    min_n = int(ecfg["min_samples_per_value"])
    plt = figs.plt
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.7), sharey=True)
    xlabels = {"cc": "Designite cyclomatic complexity (paths)", "pc": "Designite parameter count (parameters)"}
    for ax, s in zip(axes, d.smells):
        e = ex[s["short"]]
        xmax = float(ecfg["figure_max_value"][s["metric"]])
        rule = scfg["code_rule"][s["code"]]
        lo = rule["low"] + (0 if rule["low_inclusive"] else 1)
        hi = rule["high"] - (0 if rule["high_inclusive"] else 1)
        ax.axvspan(lo - 0.5, min(hi, xmax) + 0.5, color="#f0efec", zorder=0, label="Tagman sampling window")
        prule = scfg["paper_rule"][s["code"]]
        for v in (prule["low"] - 0.5, prule["high"] + 0.5):
            ax.axvline(v, color=INK_SECONDARY, linestyle=":", linewidth=0.9)
        for key, color, label in (("rate_by_metric_offered", COLORS[s["short"]], "smell offered"),
                                  ("rate_by_metric_not_offered", NEUTRAL, "smell not offered")):
            pts = [p for p in e[key] if p["value"] <= xmax and p["samples"]["n"] > 0]
            for p in pts:
                r = p["samples"]
                faded = r["n"] < min_n
                ax.errorbar(p["value"], r["rate"], yerr=[[r["rate"] - r["ci_low"]], [r["ci_high"] - r["rate"]]],
                            fmt="o", color=color, alpha=0.35 if faded else 1.0, markersize=3.5, capsize=1.5,
                            linewidth=0.9)
            ax.plot([], [], "o", color=color, markersize=3.5, label=label)
        ax.plot([], [], linestyle=":", color=INK_SECONDARY, label="paper Table I window")
        ax.set_xlim(-0.5, xmax + 0.5)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(xlabels[s["metric"]])
        ax.set_title(s["name"], loc="left")
    axes[0].set_ylabel("positive samples (share)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.2), ncols=4)
    figs.save(fig, "rate_vs_metric")


def fig_agreement(ag: dict[str, Any], d: Data, figs: Figures) -> None:
    plt = figs.plt
    from matplotlib.lines import Line2D

    rows = []
    for s, name in zip(d.cfg["mlcq"]["short_names"], d.cfg["mlcq"]["smell_names"]):
        rows.append(("MLCQ", name, ag["mlcq"][s], None, COLORS["mlcq"]))
    for s in d.smells:
        e = ag["dacos"][s["short"]]
        rows.append(("DACOS", s["name"], e["offered"], e["offered_sensitivity"], COLORS[s["short"]]))
    e = ag["dacos"][d.cls_smell["short"]]
    rows.append(("DACOS", d.cls_smell["name"], e["all_class_snippets"], e["sensitivity"],
                 COLORS[d.cls_smell["short"]]))
    # main estimate, then the sensitivity variants not selected on the answer
    styles = [("flag_and_is_smell", {"marker": "x", "markersize": 4.5}, "flag set to 0 when is_smell is 0"),
              ("without_samples_in_affected_bins", {"marker": "o", "markerfacecolor": "white", "markersize": 4},
               "samples of the affected bins dropped")]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.1), sharey=True,
                             gridspec_kw={"width_ratios": [1.3, 1.0], "wspace": 0.12})
    ys = np.arange(len(rows))

    def bar(ax: Any, a: dict[str, Any], y: float, color: str, **kw: Any) -> None:
        if a["alpha"] is None or math.isnan(a["alpha"]):
            return
        ax.errorbar(a["alpha"], y, xerr=[[a["alpha"] - a["ci_low"]], [a["ci_high"] - a["alpha"]]],
                    fmt="none", ecolor=color, capsize=2, linewidth=1.0)
        ax.plot(a["alpha"], y, linestyle="", color=color, **({"marker": "o", "markersize": 4} | kw))

    for y, (ds, name, a, sens, color) in zip(ys, rows):
        offsets = [-0.22, 0.0, 0.22] if sens else [0.0]
        bar(axes[0], a, y + offsets[0], color)
        if sens:
            for (key, kw, _), dy in zip(styles, offsets[1:]):
                bar(axes[0], sens[key], y + dy, color, **kw)
        axes[1].plot(a["observed_agreement"], y, "o", color=color, markersize=4)
        axes[1].plot(a["row_positive_rate"], y, "|", color=color, markersize=8, markeredgewidth=1.4)
    ref = d.cfg["agreement"].get("alpha_reference")
    axes[0].axvline(0, color=INK_SECONDARY, linewidth=0.8)
    if ref:
        axes[0].axvline(float(ref["value"]), color=NEUTRAL, linestyle="--", linewidth=0.8)
    axes[0].set_yticks(ys, [f"{ds}: {name} ({a['units_multi']})" for ds, name, a, _, _ in rows])
    axes[0].invert_yaxis()
    axes[0].set_xlim(-0.25, 1.0)
    axes[0].set_xlabel("Krippendorff alpha (nominal, 95% CI)")
    axes[1].set_xlim(0, 1.0)
    axes[1].set_xlabel("share")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    handles = [Line2D([], [], marker="o", color=INK_SECONDARY, linestyle="", markersize=4, label="all rows"),
               *[Line2D([], [], color=INK_SECONDARY, linestyle="", label=label,
                        **({"marker": "o", "markersize": 4} | kw)) for _, kw, label in styles],
               Line2D([], [], marker="o", color=INK_SECONDARY, linestyle="", markersize=4,
                      label="agreeing pairs (right)"),
               Line2D([], [], marker="|", color=INK_SECONDARY, linestyle="", markersize=8, markeredgewidth=1.4,
                      label="positive annotations (right)")]
    if ref:
        handles.append(Line2D([], [], color=NEUTRAL, linestyle="--", linewidth=0.8,
                              label=f"alpha {float(ref['value']):.3f}"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncols=3)
    figs.save(fig, "agreement_vs_mlcq")


def fig_projects(bt: dict[str, Any], d: Data, figs: Figures) -> None:
    plt = figs.plt
    series = [(s["short"], s["name"]) for s in d.smells] + [(d.cls_smell["short"], d.cls_smell["name"])]
    projects = sorted(bt[d.smells[0]["short"]]["project_prevalence"],
                      key=lambda p: -sum(bt[s]["project_prevalence"].get(p, {"n": 0})["n"] for s, _ in series))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ys = np.arange(len(projects))
    offsets = np.linspace(-0.22, 0.22, len(series))
    for (short, name), off in zip(series, offsets):
        for y, p in zip(ys, projects):
            r = bt[short]["project_prevalence"].get(p)
            if not r or not r["n"]:
                continue
            ax.errorbar(r["rate"], y + off, xerr=[[r["rate"] - r["ci_low"]], [r["ci_high"] - r["rate"]]], fmt="o",
                        color=COLORS[short], markersize=3.5, capsize=1.5, linewidth=0.9)
        ax.plot([], [], "o", color=COLORS[short], markersize=3.5, label=name)
    labels = []
    for p in projects:
        ns = "/".join(str(bt[s]["project_prevalence"].get(p, {"n": 0})["n"]) for s, _ in series)
        labels.append(f"{p} ({ns})")
    ax.set_yticks(ys, labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("positive samples (share, 95% Wilson interval)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.18), ncols=3)
    figs.save(fig, "project_prevalence")


def fig_order(bt: dict[str, Any], d: Data, figs: Figures) -> None:
    plt = figs.plt
    series = [(s["short"], s["name"]) for s in d.smells] + [(d.cls_smell["short"], d.cls_smell["name"])]
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    width = 100.0 / d.n_bins
    offsets = np.linspace(-0.2, 0.2, len(series)) * width
    for (short, name), off in zip(series, offsets):
        xs, ys, lo, hi = [], [], [], []
        for b in range(d.n_bins):
            r = bt[short]["order_prevalence"][str(b)]
            xs.append((b + 0.5) * width + off)
            # empty bins break the line
            ys.append(r["rate"] if r["n"] else np.nan)
            lo.append(r["rate"] - r["ci_low"] if r["n"] else np.nan)
            hi.append(r["ci_high"] - r["rate"] if r["n"] else np.nan)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=COLORS[short], markersize=3, capsize=1.5,
                    linewidth=0.8, label=name)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("position in the annotation order (percentile of annotation ids)")
    ax.set_ylabel("positive annotations (share)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=3)
    figs.save(fig, "annotation_order")


# ------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--self-test", action="store_true", help="check alpha and feature code and exit")
    args, extra = parser.parse_known_args()
    if args.self_test or args.config is None:
        alpha_self_test()
        wilson_self_test()
        permutation_self_test()
        feature_self_test("public class SnippetWrapper {\n", "\n}\n")
        return
    cfg = load_config(args.config, parse_cli_overrides(extra))
    alpha_self_test()
    wilson_self_test()
    permutation_self_test()
    fcfg = cfg["features"]
    feature_self_test(fcfg["wrapper_prefix"], fcfg["wrapper_suffix"])
    # one generator per section
    rng_agreement, rng_batches = (np.random.default_rng(s) for s in np.random.SeedSequence(int(cfg["seed"])).spawn(2))
    out_dir = resolve(cfg["out"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    d = Data(cfg)
    print(f"[dacos] {d.n} method and {len(d.classes)} class snippets, {len(d.rows)} annotation rows, "
          f"vote check {d.check}")
    if d.check["vote_mismatches"]:
        raise SystemExit("votes of the JSON files differ from the dump")

    ts_rows = [method_features(e["code_snippet"], fcfg["wrapper_prefix"], fcfg["wrapper_suffix"]) for e in d.methods]
    csv_path = out_dir / cfg["out"]["features_csv"]
    feat_info = {"csv": rel(csv_path),
                 "no_method_parsed": int(sum(r["ts_methods"] == 0 for r in ts_rows)),
                 "methods_per_snippet": {str(k): v for k, v in
                                         sorted(Counter(r["ts_methods"] for r in ts_rows).items())},
                 "parse_error": int(sum(r["ts_parse_error"] for r in ts_rows))}
    print(f"[features] {feat_info}")
    blocks = features_matrix(d, ts_rows)
    attach = metric_attachment(d, ts_rows)
    print(f"[metric attachment] {attach}")

    print("[consistency]")
    cons, checks = consistency(d)
    for k, v in cons.items():
        if isinstance(v, dict) and "rows" in v and "id_range" in v:
            print(f"  {k}: {v['rows']} rows, {v['samples']} samples, ids {v['id_range']}, "
                  f"patterns {v['flag_patterns']}")
    for k, v in cons["by_smell"].items():
        print(f"  {k}: {v['samples_with_inconsistent_rows']} samples with an inconsistent row, "
              f"affected bins {v['affected_bins']}")
    print(f"  class rows with isma = 1 (impossible under the public code): {cons['class_rows_with_isma']}")
    print("[sampling]")
    smp = sampling(d)
    for name, entry in smp["rule_check"].items():
        print(f"  {name}: " + ", ".join(f"{k} {v['match']}/{v['total']}" for k, v in entry.items()))
    print(f"  dump counts {smp['dump_counts']}")
    print("[agreement]")
    ag = agreement(d, rng_agreement, checks)

    def show(label: str, a: dict[str, Any]) -> None:
        print(f"  {label}: alpha {a['alpha']:.4f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}], "
              f"observed {a['observed_agreement']:.3f}, unanimous {a['unanimous_share']:.3f}, "
              f"units {a['units_multi']}")

    for ds in ("dacos", "mlcq"):
        for s, e in ag[ds].items():
            if "alpha" in e:
                show(f"{ds} {s}", e)
            for sub, a in e.items():
                if isinstance(a, dict) and "alpha" in a:
                    show(f"{ds} {s} {sub}", a)
                elif isinstance(a, dict) and "affected_bins" in a:
                    for var, v in a.items():
                        if isinstance(v, dict):
                            show(f"{ds} {s} {sub} {var}", v)
    print("[explain]")
    ex = explain(d, blocks)
    print("[batches]")
    bt = batches(d, blocks, rng_batches)

    figs = Figures(resolve(cfg["out"]["figures_dir"]), str(cfg["out"]["figure_prefix"]), int(cfg["out"]["dpi"]))
    fig_rate_vs_metric(d, ex, figs)
    fig_agreement(ag, d, figs)
    fig_projects(bt, d, figs)
    fig_order(bt, d, figs)

    result = {"config": rel(resolve(args.config)), "inputs": cfg["inputs"], "data_check": d.check,
              "features": feat_info, "metric_attachment": attach, "agreement": ag, "sampling": smp,
              "consistency": cons, "explain": ex,
              "batches": bt, "figures": figs.written}
    # write the feature CSV and the JSON last, each through a temporary file
    cols = ["json_index", "sample_id"] + list(fcfg["tree_sitter"]) + ["ts_param_counts", "ts_parse_error"]
    tmp = tmp_path(csv_path)
    with tmp.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, (e, r) in enumerate(zip(d.methods, ts_rows)):
            w.writerow([i, e["sample_id"]] + [";".join(map(str, r[c])) if isinstance(r[c], list) else r[c]
                                              for c in cols[2:]])
    tmp.replace(csv_path)
    path = out_dir / cfg["out"]["json"]
    atomic_json(path, result)
    print(f"[dacos] wrote {rel(csv_path)}, {rel(path)} and {len(figs.written)} figure files")


if __name__ == "__main__":
    main()
