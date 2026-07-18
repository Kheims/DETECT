"""Test whether the Data Class gain of the code models survives structural matching.

Stages:
- features: tree-sitter structural features of every snippet
- gbm, gbm_summary: XGBoost on structure, embedding and both
- matching: matched AUC of every family with bootstrap intervals
- figure: AUC before and after matching

Usage:
    python scripts/analysis/dc_structural_matching.py --config config/experiments/dc_matching.yml
    python scripts/analysis/dc_structural_matching.py --config config/experiments/dc_matching.yml --stages matching,figure
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.extract_size_features import (  # noqa: E402
    CLASS_TYPES,
    COMMENT_TYPES,
    FIELD_TYPES,
    PARAMETER_TYPES,
    STATEMENT_TYPES,
)
from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.evaluation.cv import (  # noqa: E402
    iter_folds,
    load_folds,
    prediction_rows,
    read_jsonl,
    tune_thresholds,
    write_jsonl,
)

STAGES = ("features", "gbm", "gbm_summary", "matching", "figure")
DECISION_TYPES = {"if_statement", "for_statement", "enhanced_for_statement", "while_statement", "do_statement",
                  "catch_clause", "ternary_expression"}
NESTING_TYPES = {"if_statement", "for_statement", "enhanced_for_statement", "while_statement", "do_statement",
                 "try_statement", "try_with_resources_statement", "switch_expression", "synchronized_statement"}
CONSTRUCTOR_TYPES = {"constructor_declaration", "compact_constructor_declaration"}
STRUCT_COLUMNS = [
    "loc_nonblank", "n_fields", "n_instance_fields", "n_public_fields", "n_methods", "n_constructors",
    "n_getters", "n_setters", "n_accessors", "accessor_fraction", "n_non_accessor_methods",
    "mean_statements_non_accessor", "max_nesting", "n_decisions", "n_nested_types", "has_type_declaration",
    "has_parse_error",
]

_PARSER: Any = None


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


# ---------------------------------------------------------------- features


def _parser() -> Any:
    global _PARSER
    if _PARSER is None:
        import tree_sitter as ts
        import tree_sitter_java as tsj

        _PARSER = ts.Parser(ts.Language(tsj.language()))
    return _PARSER


def text(node: Any) -> str:
    return node.text.decode("utf-8", errors="replace") if node is not None else ""


def named(node: Any) -> list[Any]:
    """Named children of a node, comments left out."""
    return [c for c in node.named_children if c.type not in COMMENT_TYPES]


def walk(node: Any) -> Iterator[Any]:
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def modifiers(node: Any) -> set[str]:
    for c in node.children:
        if c.type == "modifiers":
            return {m.type for m in c.children}
    return set()


def members(body: Any) -> Iterator[Any]:
    for c in body.named_children:
        if c.type == "enum_body_declarations":
            yield from c.named_children
        else:
            yield c


def unwrap(node: Any) -> Any:
    while node is not None and node.type == "parenthesized_expression":
        kids = named(node)
        node = kids[0] if kids else None
    return node


def is_field_ref(node: Any, fields: set[str], exclude: set[str]) -> bool:
    """`this.x`, or a bare identifier naming a field of the type."""
    node = unwrap(node)
    if node is None:
        return False
    if node.type == "identifier":
        name = text(node)
        return name in fields and name not in exclude
    if node.type == "field_access":
        obj = node.child_by_field_name("object")
        return obj is not None and obj.type == "this"
    return False


def param_name(param: Any) -> str:
    """Name of a formal or varargs parameter."""
    name = param.child_by_field_name("name")
    if name is None:
        decl = next((c for c in param.named_children if c.type == "variable_declarator"), None)
        name = decl.child_by_field_name("name") if decl is not None else None
    return text(name)


def param_names(method: Any) -> list[str]:
    params = method.child_by_field_name("parameters")
    if params is None:
        return []
    return [param_name(c) for c in params.named_children if c.type in PARAMETER_TYPES]


def body_statements(method: Any) -> list[Any] | None:
    body = method.child_by_field_name("body")
    return None if body is None else named(body)


def is_getter(method: Any, fields: set[str]) -> bool:
    stmts = body_statements(method)
    if param_names(method) or stmts is None or len(stmts) != 1 or stmts[0].type != "return_statement":
        return False
    value = named(stmts[0])
    return len(value) == 1 and is_field_ref(value[0], fields, set())


def is_setter(method: Any, fields: set[str], fluent: bool) -> bool:
    params = param_names(method)
    stmts = body_statements(method)
    if len(params) != 1 or stmts is None:
        return False
    if fluent and len(stmts) == 2 and stmts[1].type == "return_statement":
        ret = named(stmts[1])
        if len(ret) == 1 and ret[0].type == "this":
            stmts = stmts[:1]
    if len(stmts) != 1 or stmts[0].type != "expression_statement":
        return False
    expr = named(stmts[0])
    if len(expr) != 1 or expr[0].type != "assignment_expression":
        return False
    assign = expr[0]
    if text(assign.child_by_field_name("operator")) != "=":
        return False
    right = unwrap(assign.child_by_field_name("right"))
    if right is None or right.type != "identifier" or text(right) != params[0]:
        return False
    return is_field_ref(assign.child_by_field_name("left"), fields, {params[0]})


def decision_count(body: Any) -> int:
    count = 0
    for n in walk(body):
        if n.type in DECISION_TYPES:
            count += 1
        elif n.type == "switch_label":
            count += int(not any(c.type == "default" for c in n.children))
        elif n.type == "binary_expression":
            count += int(text(n.child_by_field_name("operator")) in ("&&", "||"))
    return count


def max_nesting(body: Any) -> int:
    """Deepest chain of nested control statements. An `else if` stays at the depth of its `if`."""
    best = 0
    stack: list[tuple[Any, int, Any]] = [(body, 0, None)]
    while stack:
        node, depth, parent = stack.pop()
        if node.type in NESTING_TYPES:
            alt = parent.child_by_field_name("alternative") if parent is not None and parent.type == "if_statement" else None
            if not (node.type == "if_statement" and alt is not None and alt.id == node.id):
                depth += 1
            best = max(best, depth)
        for c in node.children:
            stack.append((c, depth, node))
    return best


def structural_features(task: tuple[int, str, str, str, bool]) -> dict[str, Any]:
    """Features of one snippet, task = (json_index, snippet, wrapper prefix, suffix, fluent setters)."""
    json_index, snippet, prefix, suffix, fluent = task
    body_text = snippet.rstrip("\n")
    lo = len(prefix.encode("utf-8"))
    hi = lo + len(body_text.encode("utf-8"))
    tree = _parser().parse((prefix + body_text + suffix).encode("utf-8"))
    root = tree.root_node

    primary = next((n for n in walk(root) if n.type in CLASS_TYPES and lo <= n.start_byte < hi), None)
    if primary is not None:
        body = primary.child_by_field_name("body")
    else:
        wrapper = next((n for n in root.children if n.type == "class_declaration"), None)
        body = wrapper.child_by_field_name("body") if wrapper is not None else None
    own = [] if body is None else [m for m in members(body) if lo <= m.start_byte < hi]
    in_interface = body is not None and body.type == "interface_body"

    fields: set[str] = set()
    n_fields = n_instance = n_public = n_nested = 0
    methods, constructors = [], []
    for m in own:
        if m.type in FIELD_TYPES:
            names = [text(d.child_by_field_name("name")) for d in m.named_children if d.type == "variable_declarator"]
            mods = modifiers(m)
            fields.update(names)
            n_fields += len(names)
            if not ("static" in mods or m.type == "constant_declaration" or in_interface):
                n_instance += len(names)
            if "public" in mods or in_interface:
                n_public += len(names)
        elif m.type == "method_declaration":
            methods.append(m)
        elif m.type in CONSTRUCTOR_TYPES:
            constructors.append(m)
        elif m.type in CLASS_TYPES:
            n_nested += 1

    n_getters = n_setters = 0
    statements = []
    for m in methods:
        if is_getter(m, fields):
            n_getters += 1
        elif is_setter(m, fields, fluent):
            n_setters += 1
        else:
            b = m.child_by_field_name("body")
            statements.append(0 if b is None else sum(1 for n in walk(b) if n.type in STATEMENT_TYPES))
    bodies = [b for m in methods + constructors if (b := m.child_by_field_name("body")) is not None]
    n_accessors = n_getters + n_setters
    lines = body_text.split("\n") if body_text else []
    return {
        "sample_idx": json_index,
        "loc_nonblank": sum(1 for line in lines if line.strip()),
        "n_fields": n_fields,
        "n_instance_fields": n_instance,
        "n_public_fields": n_public,
        "n_methods": len(methods),
        "n_constructors": len(constructors),
        "n_getters": n_getters,
        "n_setters": n_setters,
        "n_accessors": n_accessors,
        "accessor_fraction": round(n_accessors / len(methods), 6) if methods else 0.0,
        "n_non_accessor_methods": len(statements),
        "mean_statements_non_accessor": round(float(np.mean(statements)), 6) if statements else 0.0,
        "max_nesting": max((max_nesting(b) for b in bodies), default=0),
        "n_decisions": sum(decision_count(b) for b in bodies),
        "n_nested_types": n_nested,
        "has_type_declaration": int(primary is not None),
        "has_parse_error": int(root.has_error),
    }


def stage_features(cfg: dict[str, Any], ctx: dict[str, Any]) -> None:
    fcfg = cfg["features"]
    entries = ctx["entries"]
    tasks = [(j, str(e.get("code_snippet", "")), str(fcfg["wrapper_prefix"]), str(fcfg["wrapper_suffix"]),
              bool(fcfg.get("fluent_setters", True))) for j, e in enumerate(entries)]
    workers = int(fcfg.get("workers", 1))
    started = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(structural_features, tasks, chunksize=64))
    else:
        rows = [structural_features(t) for t in tasks]
    # check before writing, so a failed check leaves the previous file in place
    size = {int(r["sample_idx"]): int(r["loc_nonblank"]) for r in csv.DictReader(resolve(fcfg["size_csv"]).open())}
    loc_mismatch = [r["sample_idx"] for r in rows if size.get(r["sample_idx"]) != r["loc_nonblank"]]
    if loc_mismatch:
        raise SystemExit(f"loc_nonblank differs from {fcfg['size_csv']} for {len(loc_mismatch)} samples, "
                         f"e.g. {loc_mismatch[:5]}; {fcfg['out_csv']} not written")
    out = resolve(fcfg["out_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sample_idx", *STRUCT_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out)

    cls =[r for r in rows if r["sample_idx"] in ctx["pool_set"]]
    y = ctx["y"]
    summary: dict[str, Any] = {"rows": len(rows), "class_level_indexed": len(cls),
                               "seconds": round(time.perf_counter() - started, 1)}
    for name, part in (("positives", [r for r in cls if y[r["sample_idx"]]]),
                       ("negatives", [r for r in cls if not y[r["sample_idx"]]])):
        summary[name] = {
            "n": len(part),
            "no_type_declaration": sum(1 for r in part if not r["has_type_declaration"]),
            "parse_error": sum(1 for r in part if r["has_parse_error"]),
            "median": {c: float(np.median([r[c] for r in part])) for c in STRUCT_COLUMNS},
            "mean": {c: round(float(np.mean([r[c] for r in part])), 4) for c in STRUCT_COLUMNS},
        }
    write_json(resolve(cfg["out_dir"]) / "features_summary.json", summary)
    print(f"[features] wrote {out} ({len(rows)} rows, {time.perf_counter() - started:.1f}s); "
          f"loc_nonblank equal to {fcfg['size_csv']} on every row")
    for name in ("positives", "negatives"):
        s = summary[name]
        med = s["median"]
        print(f"[features] class level, {name} n={s['n']} (no type declaration {s['no_type_declaration']}, "
              f"parse error {s['parse_error']}): median loc {med['loc_nonblank']:.0f} fields {med['n_fields']:.0f} "
              f"methods {med['n_methods']:.0f} accessors {med['n_accessors']:.0f} accessor fraction "
              f"{med['accessor_fraction']:.3f}; mean accessor fraction {s['mean']['accessor_fraction']:.3f}")


def load_struct(cfg: dict[str, Any]) -> tuple[dict[int, dict[str, float]], list[str]]:
    path = resolve(cfg["features"]["out_csv"])
    if not path.exists():
        raise SystemExit(f"{path} not found: run the features stage first")
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    columns = [c for c in rows[0] if c != "sample_idx"]
    return {int(r["sample_idx"]): {c: float(r[c]) for c in columns} for r in rows}, columns


# --------------------------------------------------------------------- gbm


def gbm_features(cfg: dict[str, Any], pool: list[int]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Structure (structural + size columns) and embedding matrices, rows in pool order."""
    gcfg = cfg["gbm"]
    struct, columns = load_struct(cfg)
    cols = list(gcfg.get("structural_columns") or columns)
    size = {int(r["sample_idx"]): r for r in csv.DictReader(resolve(gcfg["size_csv"]).open())}
    size_cols = list(gcfg.get("size_columns") or [])
    Xs = np.array([[struct[j][c] for c in cols] + [float(size[j][c]) for c in size_cols] for j in pool])
    ecfg = gcfg["embedding"]
    matrix = np.load(resolve(ecfg["path"]))
    index = json.loads(resolve(ecfg["index"]).read_text())
    if len(index) != matrix.shape[0]:
        raise SystemExit(f"{ecfg['path']}: {matrix.shape[0]} rows but {len(index)} indices")
    row_of = {int(j): k for k, j in enumerate(index)}
    missing = [j for j in pool if j not in row_of]
    if missing:
        raise SystemExit(f"{len(missing)} class-level samples have no embedding, e.g. {missing[:5]}")
    E = matrix[[row_of[j] for j in pool]].astype(np.float64)
    return Xs, cols + size_cols, E


def selection_value(y: np.ndarray, p: np.ndarray, metric: str) -> float:
    if y.min() == y.max():
        return float("nan")
    return float(average_precision_score(y, p) if metric == "pr_auc" else roc_auc_score(y, p))


def binary_metrics(y_val: np.ndarray, p_val: np.ndarray, y_test: np.ndarray, p_test: np.ndarray,
                   steps: int) -> dict[str, float]:
    """Test metrics of one label with the threshold tuned on validation."""
    t = float(tune_thresholds(y_val[:, None], p_val[:, None], steps)[0])
    pred = (p_test >= t).astype(np.int64)
    return {"threshold": t, "roc_auc": float(roc_auc_score(y_test, p_test)),
            "pr_auc": float(average_precision_score(y_test, p_test)),
            "mcc": float(matthews_corrcoef(y_test, pred)), "f1": float(f1_score(y_test, pred, zero_division=0)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0))}


def class_level_parts(fold: Any, pos_of: dict[int, int]) -> dict[str, np.ndarray]:
    return {p: np.array([pos_of[j] for j in getattr(fold, p) if j in pos_of], dtype=np.int64)
            for p in ("train", "val", "test")}


def stage_gbm(cfg: dict[str, Any], ctx: dict[str, Any]) -> None:
    """Fit every arm on every fold and write its predictions and fit records."""
    from xgboost import XGBClassifier

    gcfg = cfg["gbm"]
    pool, y_of = ctx["pool"], ctx["y"]
    pos_of = {j: k for k, j in enumerate(pool)}
    Xs, names, E = gbm_features(cfg, pool)
    Y = np.array([y_of[j] for j in pool], dtype=np.int64)
    pca_candidates = sorted(int(n) for n in gcfg["embedding"]["pca_candidates"])
    metric = str(gcfg.get("selection_metric", "pr_auc"))
    arms = dict(gcfg["arms"])
    out_root = resolve(cfg["out_dir"]) / "gbm"
    print(f"[gbm] {len(pool)} class-level samples, {Y.sum()} positives; structure {Xs.shape[1]} columns, "
          f"embedding {gcfg['embedding']['name']} {E.shape[1]} dims, PCA candidates {pca_candidates}", flush=True)

    records: dict[str, list] = defaultdict(list)
    preds: dict[str, list] = defaultdict(list)
    for fold in iter_folds(ctx["folds"], list(gcfg["strategies"]), gcfg.get("seeds")):
        part = class_level_parts(fold, pos_of)
        ytr, yva, yte = Y[part["train"]], Y[part["val"]], Y[part["test"]]
        # PCA fit on the training part, full SVD so components agree across n
        scaler = StandardScaler().fit(E[part["train"]])
        pca = PCA(n_components=max(pca_candidates), svd_solver="full").fit(scaler.transform(E[part["train"]]))
        Z = {p: pca.transform(scaler.transform(E[idx])) for p, idx in part.items()}
        spw = float((ytr == 0).sum()) / max(int(ytr.sum()), 1)
        for arm, spec in arms.items():
            started = time.perf_counter()
            best: tuple[float, Any, str, dict[str, np.ndarray]] | None = None
            scores: dict[str, float] = {}
            for n_comp in (pca_candidates if spec.get("embedding") else [None]):
                X = {}
                for p, idx in part.items():
                    blocks = ([Xs[idx]] if spec.get("structure") else []) + ([Z[p][:, :n_comp]] if n_comp else [])
                    X[p] = np.hstack(blocks)
                for key, params in gcfg["models"].items():
                    model = XGBClassifier(**{**gcfg.get("common", {}), **params}, scale_pos_weight=spw,
                                          random_state=fold.seed, eval_metric="logloss")
                    model.fit(X["train"], ytr)
                    name = key if n_comp is None else f"{key}_pca{n_comp}"
                    score = selection_value(yva, model.predict_proba(X["val"])[:, 1], metric)
                    scores[name] = score
                    rank = score if np.isfinite(score) else -np.inf
                    if best is None or rank > best[0]:
                        best = (rank, model, name, X)
            assert best is not None
            _, model, chosen, X = best
            p_val, p_test = model.predict_proba(X["val"])[:, 1], model.predict_proba(X["test"])[:, 1]
            records[arm].append({
                "arm": arm, "strategy": fold.strategy, "seed": fold.seed, "fold": fold.fold,
                "selected_model": chosen, "selection_metric": metric, "selection_val_scores": scores,
                "num_features": int(X["train"].shape[1]), "train_duration_sec": time.perf_counter() - started,
                **{f"n_{p}": int(len(idx)) for p, idx in part.items()}, "n_test_positives": int(yte.sum())})
            for p_name, idx, yy, pp in (("val", part["val"], yva, p_val), ("test", part["test"], yte, p_test)):
                preds[arm].extend(prediction_rows(model=arm, strategy=fold.strategy, seed=fold.seed, fold=fold.fold,
                                                  part=p_name, indices=np.array([pool[k] for k in idx]),
                                                  y=yy[:, None], probs=pp[:, None]))
            print(f"[gbm] {arm} {fold.strategy} seed={fold.seed} fold={fold.fold} "
                  f"auc={roc_auc_score(yte, p_test):.3f} selected={chosen}", flush=True)

    for arm in arms:
        out = out_root / arm
        write_jsonl(out / "predictions.jsonl", preds[arm])
        write_json(out / "records.json", records[arm])
        print(f"[gbm] wrote {out}", flush=True)
    write_json(out_root / "structure_columns.json", names)


def stage_gbm_summary(cfg: dict[str, Any], ctx: dict[str, Any]) -> None:
    """Per-fold test metrics of the arms, paired comparisons and family metrics."""
    gcfg = cfg["gbm"]
    steps = int(gcfg.get("threshold_steps", 101))
    out_root = resolve(cfg["out_dir"]) / "gbm"
    per_fold: dict[str, dict[tuple[str, int, int], dict[str, float]]] = {}
    for arm in gcfg["arms"]:
        path = out_root / arm / "predictions.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} not found: run the gbm stage first")
        parts: dict[tuple[str, int, int], dict[str, tuple[list, list]]] = defaultdict(lambda: defaultdict(lambda: ([], [])))
        for row in read_jsonl(path):
            ys, ps = parts[(row["strategy"], int(row["seed"]), int(row["fold"]))][row["part"]]
            ys.append(int(row["y"][0]))
            ps.append(float(row["prob"][0]))
        per_fold[arm] = {k: binary_metrics(np.array(v["val"][0]), np.array(v["val"][1]),
                                           np.array(v["test"][0]), np.array(v["test"][1]), steps)
                         for k, v in parts.items()}
    pos_of = {j: k for k, j in enumerate(ctx["pool"])}
    ratios: dict[str, list[float]] = defaultdict(list)
    for fold in iter_folds(ctx["folds"], list(gcfg["strategies"]), gcfg.get("seeds")):
        part = class_level_parts(fold, pos_of)
        ratios[fold.strategy].append(len(part["test"]) / len(part["train"]))
    summary = gbm_summary(cfg, per_fold, ratios)
    names_path = out_root / "structure_columns.json"
    summary["structure_columns"] = json.loads(names_path.read_text()) if names_path.exists() else None
    summary["per_fold"] = {arm: [{"strategy": k[0], "seed": k[1], "fold": k[2], **v} for k, v in sorted(cells.items())]
                           for arm, cells in per_fold.items()}
    summary["families_class_level"] = family_class_metrics(cfg, ctx)
    write_json(resolve(cfg["out_dir"]) / "gbm_summary.json", summary)
    print_gbm(summary)


def seed_stats(cells: dict[tuple[str, int, int], float]) -> dict[str, Any]:
    """Mean over the folds of a seed, then mean and sd over seeds."""
    by_seed: dict[int, list[float]] = defaultdict(list)
    for (_, seed, _), v in cells.items():
        by_seed[seed].append(v)
    seeds = [statistics.mean(v) for _, v in sorted(by_seed.items())]
    return {"mean": statistics.mean(seeds), "sd": statistics.stdev(seeds) if len(seeds) > 1 else 0.0,
            "per_seed": seeds, "folds": len(cells)}


def gbm_summary(cfg: dict[str, Any], per_fold: dict, ratios: dict[str, list[float]]) -> dict[str, Any]:
    from scripts.cv.summarize_cv import corrected_ttest

    gcfg = cfg["gbm"]
    metrics = ("roc_auc", "pr_auc", "mcc", "f1", "precision", "recall")
    out: dict[str, Any] = {"arms": {}, "comparisons": {}, "test_train_ratio": {}}
    for strategy in gcfg["strategies"]:
        ratio = statistics.mean(ratios[strategy])
        out["test_train_ratio"][strategy] = ratio
        for arm, cells in per_fold.items():
            sub = {k: v for k, v in cells.items() if k[0] == strategy}
            out["arms"].setdefault(arm, {})[strategy] = {
                m: seed_stats({k: v[m] for k, v in sub.items()}) for m in metrics}
        for a, b in gcfg["comparisons"]:
            keys = sorted(k for k in per_fold[a] if k[0] == strategy and k in per_fold[b])
            res = {}
            for m in metrics:
                diffs = {k: per_fold[a][k][m] - per_fold[b][k][m] for k in keys}
                t, p, half = corrected_ttest(np.array(list(diffs.values())), ratio)
                st = seed_stats(diffs)
                res[m] = {**st, "corrected_t": t, "corrected_p": p, "corrected_ci95": [st["mean"] - half, st["mean"] + half],
                          "folds_a_better": int(sum(1 for v in diffs.values() if v > 0))}
            out["comparisons"].setdefault(f"{a} - {b}", {})[strategy] = res
    return out


def print_gbm(summary: dict[str, Any]) -> None:
    for arm, by_strategy in summary["arms"].items():
        for strategy, ms in by_strategy.items():
            print(f"[gbm] {arm:<20} {strategy:<12} " + "  ".join(
                f"{m} {v['mean']:.3f} (sd {v['sd']:.3f})" for m, v in ms.items()))
    for name, by_strategy in summary["comparisons"].items():
        for strategy, ms in by_strategy.items():
            print(f"[gbm] {name} {strategy}: " + "  ".join(
                f"d{m} {v['mean']:+.3f} [{v['corrected_ci95'][0]:+.3f}, {v['corrected_ci95'][1]:+.3f}] "
                f"p={v['corrected_p']:.3g} ({v['folds_a_better']}/{v['folds']} folds)" for m, v in ms.items()))
    for strategy, fams in summary["families_class_level"].items():
        for fam, ms in fams.items():
            print(f"[gbm] reference {fam:<20} {strategy:<12} class-level roc_auc {ms['roc_auc']['mean']:.3f} "
                  f"pr_auc {ms['pr_auc']['mean']:.3f}")


# ----------------------------------------------------------- family scores


def load_scores(patterns: list[str], col: int) -> dict[str, dict[int, dict[int, tuple[int, float]]]]:
    """{strategy: {seed: {json_index: (fold, score of column col)}}} from the test rows."""
    files = sorted({f for p in patterns for f in glob.glob(str(resolve(p)))})
    if not files:
        raise SystemExit(f"no prediction file matches {patterns}")
    out: dict[str, dict[int, dict[int, tuple[int, float]]]] = defaultdict(lambda: defaultdict(dict))
    for path in files:
        for row in read_jsonl(Path(path)):
            if row.get("part") != "test":
                continue
            probs = row.get("prob", row.get("probs"))
            out[str(row["strategy"])][int(row["seed"])][int(row["json_index"])] = (int(row["fold"]), float(probs[col]))
    return out


def family_class_metrics(cfg: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Class-level ROC-AUC and PR-AUC of the families per test fold, for reference."""
    out: dict[str, Any] = {}
    pool_set, y = ctx["pool_set"], ctx["y"]
    for fam, spec in cfg["families"].items():
        for strategy, by_seed in load_scores(list(spec["predictions"]), ctx["label_col"]).items():
            if strategy not in cfg["gbm"]["strategies"]:
                continue
            cells: dict[str, dict] = {"roc_auc": {}, "pr_auc": {}}
            for seed, rows in by_seed.items():
                by_fold: dict[int, list[int]] = defaultdict(list)
                for j, (f, _) in rows.items():
                    if j in pool_set:
                        by_fold[f].append(j)
                for f, idx in by_fold.items():
                    yy = np.array([y[j] for j in idx])
                    ss = np.array([rows[j][1] for j in idx])
                    cells["roc_auc"][(strategy, seed, f)] = float(roc_auc_score(yy, ss))
                    cells["pr_auc"][(strategy, seed, f)] = float(average_precision_score(yy, ss))
            out.setdefault(strategy, {})[fam] = {m: seed_stats(c) for m, c in cells.items()}
    return out


# ---------------------------------------------------------------- matching


def accessor_bins(struct: dict[int, dict[str, float]], pool: list[int], edges: list[float]) -> np.ndarray:
    frac = np.array([struct[j]["accessor_fraction"] for j in pool])
    methods = np.array([struct[j]["n_methods"] for j in pool])
    return np.where(methods == 0, -1, np.searchsorted(np.asarray(edges, dtype=float), frac, side="right"))


def match(scheme: dict[str, Any], y: np.ndarray, abin: np.ndarray, raw: np.ndarray, covariates: list[str]) -> dict[int, np.ndarray]:
    """{position of a positive: positions of its matched negatives} (positives without a match left out)."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    matched: dict[int, np.ndarray] = {}
    if scheme["type"] == "nearest":
        Z = np.log1p(raw)
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0, ddof=1)
        VI = np.linalg.inv(np.cov(Z, rowvar=False))
        k, caliper = int(scheme["k"]), float(scheme["caliper_sd"])
        for p in pos:
            cand = neg[(abin[neg] == abin[p]) & (np.abs(Z[neg] - Z[p]).max(axis=1) <= caliper)]
            if cand.size == 0:
                continue
            d = Z[cand] - Z[p]
            dist = np.einsum("ij,jk,ik->i", d, VI, d)
            matched[int(p)] = cand[np.argsort(dist, kind="stable")[:k]]
    elif scheme["type"] == "exact":
        cells = [abin] + [np.searchsorted(np.asarray(scheme["edges"][c], dtype=float), raw[:, i], side="right")
                          for i, c in enumerate(covariates)]
        key = np.stack(cells, axis=1)
        for p in pos:
            cand = neg[(key[neg] == key[p]).all(axis=1)]
            if cand.size:
                matched[int(p)] = cand
    else:
        raise SystemExit(f"unknown matching type {scheme['type']}")
    return matched


def pair_wins(s: np.ndarray, matched: dict[int, np.ndarray]) -> np.ndarray:
    """Per matched positive: share of its matched negatives scored lower (ties half)."""
    return np.array([float(np.mean((s[p] > s[n]) + 0.5 * (s[p] == s[n]))) for p, n in matched.items()])


def twoway_draws(seed_scores: list[np.ndarray], matched: dict[int, np.ndarray], weights_pos: np.ndarray,
                 weights_neg: np.ndarray, negs: np.ndarray) -> np.ndarray:
    """Matched AUC of every bootstrap draw with positives and negatives resampled independently."""
    col = {int(n): k for k, n in enumerate(negs)}
    mask = np.zeros((len(matched), len(negs)))
    for a, n in enumerate(matched.values()):
        for m in n:
            mask[a, col[int(m)]] += 1.0
    pos = np.array(list(matched), dtype=int)
    # wins on every (matched positive, matched negative) pair, mean over seeds
    wins = np.mean([(s[pos][:, None] > s[negs][None, :]) + 0.5 * (s[pos][:, None] == s[negs][None, :])
                    for s in seed_scores], axis=0)
    den = mask @ weights_neg.T
    num = (wins * mask) @ weights_neg.T
    ok = den > 0
    per = np.divide(num, den, out=np.zeros_like(num), where=ok)
    w = weights_pos.T * ok
    return (w * per).sum(axis=0) / w.sum(axis=0)


def draw_counts(idx: np.ndarray, n: int) -> np.ndarray:
    """(draws, n) counts of the indices in every row of idx (draws, size), indices in [0, n)."""
    flat = (idx + n * np.arange(idx.shape[0])[:, None]).ravel()
    return np.bincount(flat, minlength=idx.shape[0] * n).reshape(idx.shape[0], n).astype(float)


def percentile_ci(draws: np.ndarray) -> list[float]:
    return [float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))]


def smd(a: np.ndarray, wa: np.ndarray, b: np.ndarray, wb: np.ndarray, denom: float) -> float:
    if denom == 0:
        return 0.0
    return float((np.average(a, weights=wa) - np.average(b, weights=wb)) / denom)


def balance(struct: dict, pool: list[int], y: np.ndarray, matched: dict[int, np.ndarray], columns: list[str]) -> dict:
    """Standardised mean differences (positives minus negatives, pooled sd before matching)."""
    out = {}
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    mp = np.array(list(matched))
    mn = np.concatenate(list(matched.values())) if matched else np.array([], dtype=int)
    wn = np.concatenate([np.full(len(v), 1.0 / len(v)) for v in matched.values()]) if matched else np.array([])
    for c in columns:
        v = np.array([struct[j][c] for j in pool])
        denom = float(np.sqrt((v[pos].var(ddof=1) + v[neg].var(ddof=1)) / 2))
        out[c] = {"before": smd(v[pos], np.ones(len(pos)), v[neg], np.ones(len(neg)), denom),
                  "after": smd(v[mp], np.ones(len(mp)), v[mn], wn, denom) if len(mp) else None}
    return out


def matching_table(cfg: dict[str, Any]) -> dict[int, dict[str, float]]:
    """Structural features plus the size counts of features.size_csv as size_<column>."""
    struct, _ = load_struct(cfg)
    with resolve(cfg["features"]["size_csv"]).open() as fh:
        for r in csv.DictReader(fh):
            j = int(r["sample_idx"])
            if j in struct:
                struct[j].update({f"size_{c}": float(v) for c, v in r.items() if c != "sample_idx"})
    return struct


def stage_matching(cfg: dict[str, Any], ctx: dict[str, Any]) -> None:
    mcfg = cfg["matching"]
    strategy = str(mcfg["strategy"])
    struct = matching_table(cfg)
    col = ctx["label_col"]
    labels = {f: spec.get("label", f) for f, spec in cfg["families"].items()}
    scores = {f: load_scores(list(spec["predictions"]), col).get(strategy, {}) for f, spec in cfg["families"].items()}
    gbm_root = resolve(cfg["out_dir"]) / "gbm"
    for arm in cfg["gbm"].get("matched_arms") or []:
        path = gbm_root / arm / "predictions.jsonl"
        if not path.exists():
            print(f"[matching] {arm}: {path} not found, run the gbm stage first; left out", flush=True)
            continue
        scores[arm] = load_scores([str(path)], 0).get(strategy, {})
        labels[arm] = cfg["gbm"]["arms"][arm].get("label", arm)

    seeds = sorted(set.intersection(*[set(s) for s in scores.values()]))
    pool = [j for j in ctx["pool"] if all(j in scores[f][s] for f in scores for s in seeds)]
    dropped = len(ctx["pool"]) - len(pool)
    y = np.array([ctx["y"][j] for j in pool], dtype=np.int64)
    S = {f: {s: np.array([scores[f][s][j][1] for j in pool]) for s in seeds} for f in scores}
    abin = accessor_bins(struct, pool, list(mcfg["accessor_fraction_edges"]))
    print(f"[matching] {strategy} test scores, seeds {seeds}; pool {len(pool)} class-level samples "
          f"({int(y.sum())} positives, {dropped} left out for missing scores); families {list(scores)}", flush=True)

    bins_table = {}
    for b in sorted(set(abin.tolist())):
        m = abin == b
        bins_table[str(b)] = {"positives": int(y[m].sum()), "negatives": int((y[m] == 0).sum())}
    before = {}
    for f in scores:
        aucs = [float(roc_auc_score(y, S[f][s])) for s in seeds]
        before[f] = {"mean": statistics.mean(aucs), "sd": statistics.stdev(aucs) if len(aucs) > 1 else 0.0}

    refs = [r for r in mcfg["references"] if r in scores]
    # unmatched AUC differences with the references, to compare with the matched ones
    for f in scores:
        before[f]["minus"] = {ref: before[f]["mean"] - before[ref]["mean"] for ref in refs if ref != f}
        print(f"[matching] all class-level samples: {f:<20} auc {before[f]['mean']:.3f} " + "  ".join(
            f"minus {ref} {d:+.3f}" for ref, d in before[f]["minus"].items()), flush=True)
    n_boot = int(mcfg["bootstrap"])
    results: dict[str, Any] = {"strategy": strategy, "seeds": seeds, "pool": len(pool), "positives": int(y.sum()),
                               "left_out_missing_scores": dropped, "covariates": list(mcfg["covariates"]),
                               "accessor_fraction_edges": list(mcfg["accessor_fraction_edges"]),
                               "accessor_bins": bins_table, "auc_before": before, "labels": labels,
                               "primary": mcfg["primary"], "references": refs, "schemes": {}}
    for name, scheme in mcfg["schemes"].items():
        covs = list(scheme.get("covariates") or mcfg["covariates"])
        raw = np.array([[struct[j][c] for c in covs] for j in pool], dtype=float)
        matched = match(scheme, y, abin, raw, covs)
        pos_all = np.flatnonzero(y == 1)
        neg_all = np.flatnonzero(y == 0)
        mp = np.array(list(matched), dtype=int)
        n_negs = [len(v) for v in matched.values()]
        distinct = len(set(np.concatenate(list(matched.values())).tolist())) if matched else 0
        for b in bins_table:
            bins_table[b].setdefault("matched_positives", {})[name] = int(sum(1 for p in mp if str(abin[p]) == b))
        res: dict[str, Any] = {
            "scheme": scheme,
            "covariates": covs,
            "matched_positives": len(matched),
            "unmatched_positives": int(len(pos_all) - len(matched)),
            "distinct_negatives": distinct,
            "negatives_per_positive": {"mean": float(np.mean(n_negs)) if n_negs else 0.0,
                                       "median": float(np.median(n_negs)) if n_negs else 0.0},
            "unmatched_positive_medians": {c: float(np.median([struct[pool[p]][c] for p in pos_all if p not in matched]))
                                           for c in ("loc_nonblank", "n_fields", "n_methods", "accessor_fraction")}
            if len(matched) < len(pos_all) else None,
            "balance_smd": balance(struct, pool, y, matched, list(mcfg["balance_columns"])),
            "families": {},
        }
        if matched:
            reuse = Counter(int(n) for v in matched.values() for n in v)
            res["negative_reuse"] = {"max_positives_per_negative": max(reuse.values()),
                                     "negatives_used_by_5_or_more": sum(1 for v in reuse.values() if v >= 5)}
            negs = np.array(sorted(reuse), dtype=int)
            rng = np.random.default_rng(int(mcfg["bootstrap_seed"]))
            boot_idx = rng.integers(0, len(matched), size=(n_boot, len(matched)))
            # two-way draws: the same positive draws, negatives drawn independently
            w_pos = draw_counts(boot_idx, len(matched))
            w_neg = draw_counts(rng.integers(0, len(negs), size=(n_boot, len(negs))), len(negs))
            wins = {f: np.stack([pair_wins(S[f][s], matched) for s in seeds]) for f in scores}
            mean_w = {f: w.mean(axis=0) for f, w in wins.items()}
            twoway = {f: twoway_draws([S[f][s] for s in seeds], matched, w_pos, w_neg, negs) for f in scores}
            for f in scores:
                per_seed = wins[f].mean(axis=1)
                sub = [float(roc_auc_score(np.r_[np.ones(len(mp)), np.zeros(len(neg_all))],
                                           np.r_[S[f][s][mp], S[f][s][neg_all]])) for s in seeds]
                entry = {
                    "matched_auc": float(per_seed.mean()),
                    "matched_auc_sd_seeds": float(per_seed.std(ddof=1)) if len(seeds) > 1 else 0.0,
                    "matched_auc_ci95": percentile_ci(mean_w[f][boot_idx].mean(axis=1)),
                    "matched_auc_ci95_twoway": percentile_ci(twoway[f]),
                    "auc_matched_positives_all_negatives": statistics.mean(sub),
                }
                # paired: the same bootstrap draws for every family
                entry["minus"] = {}
                for ref in refs:
                    if f != ref:
                        diff = mean_w[f] - mean_w[ref]
                        entry["minus"][ref] = {"mean": float(diff.mean()),
                                               "ci95": percentile_ci(diff[boot_idx].mean(axis=1)),
                                               "ci95_twoway": percentile_ci(twoway[f] - twoway[ref])}
                res["families"][f] = entry
        results["schemes"][name] = res
        reuse_text = (f"; a negative serves up to {res['negative_reuse']['max_positives_per_negative']} positives, "
                      f"{res['negative_reuse']['negatives_used_by_5_or_more']} negatives serve 5 or more"
                      if matched else "")
        print(f"[matching] {name}: {len(matched)}/{len(pos_all)} positives matched, {distinct} distinct negatives, "
              f"{res['negatives_per_positive']['mean']:.2f} negatives per positive{reuse_text}", flush=True)
        worst = max(res["balance_smd"].items(), key=lambda kv: abs(kv[1]["after"] or 0.0))
        print(f"[matching]   largest |SMD| after matching: {worst[0]} {worst[1]['after']:+.3f} "
              f"(before {worst[1]['before']:+.3f})", flush=True)
        for f, e in res["families"].items():
            extra = "".join(f"  minus {ref} {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] two-way "
                            f"[{d['ci95_twoway'][0]:+.3f}, {d['ci95_twoway'][1]:+.3f}]"
                            for ref, d in e["minus"].items())
            print(f"[matching]   {f:<20} auc all {before[f]['mean']:.3f}  matched positives vs all negatives "
                  f"{e['auc_matched_positives_all_negatives']:.3f}  matched {e['matched_auc']:.3f} "
                  f"[{e['matched_auc_ci95'][0]:.3f}, {e['matched_auc_ci95'][1]:.3f}] two-way "
                  f"[{e['matched_auc_ci95_twoway'][0]:.3f}, {e['matched_auc_ci95_twoway'][1]:.3f}] (sd seeds "
                  f"{e['matched_auc_sd_seeds']:.3f}){extra}", flush=True)
    print(f"[matching] accessor-fraction bins (-1 = no method): {bins_table}", flush=True)
    out = resolve(cfg["out_dir"]) / "matching.json"
    write_json(out, results)
    print(f"[matching] wrote {out}", flush=True)


# ------------------------------------------------------------------ figure


def stage_figure(cfg: dict[str, Any], ctx: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.cv.make_figures import PALETTE, style

    fcfg = cfg["figure"]
    style(fcfg)
    out_dir = resolve(cfg["out_dir"])
    mres = json.loads((out_dir / "matching.json").read_text())
    gres = json.loads((out_dir / "gbm_summary.json").read_text())
    primary = mres["schemes"][mres["primary"]]
    second_name = fcfg.get("secondary_scheme")
    second = mres["schemes"].get(second_name) if second_name else None
    fams = list(primary["families"])
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), gridspec_kw={"width_ratios": [1.3, 1.0], "wspace": 0.95})
    ax = axes[0]
    ys = np.arange(len(fams))[::-1].astype(float)
    shown = [(primary, "o", PALETTE[0], -0.12 if second else 0.0)] + ([(second, "s", PALETTE[1], 0.12)] if second else [])
    for yv, f in zip(ys, fams):
        b = mres["auc_before"][f]["mean"]
        ax.scatter([b], [yv], s=36, facecolor="white", edgecolor="#555555", linewidth=1.3, zorder=3)
        for scheme, marker, color, dy in shown:
            e = scheme["families"].get(f)
            if e is None:
                continue
            ax.plot(e["matched_auc_ci95_twoway"], [yv + dy] * 2, color=color, linewidth=0.8, alpha=0.5, zorder=3)
            ax.errorbar([e["matched_auc"]], [yv + dy], xerr=[[e["matched_auc"] - e["matched_auc_ci95"][0]],
                                                            [e["matched_auc_ci95"][1] - e["matched_auc"]]],
                        fmt=marker, color=color, markersize=5, capsize=2, linewidth=1.6, zorder=4)
    ax.axvline(0.5, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([mres["labels"][f] for f in fams])
    ax.set_xlabel("Data Class ROC-AUC (class level, kfold test scores)")
    ax.set_title("(a) before and inside matched pairs", fontsize=9, loc="left")
    handles = [Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="#555555",
                      label=f"all class-level samples ({mres['positives']} positives)")]
    for scheme, marker, color, _ in shown:
        what = scheme["scheme"].get("label") or f"accessor bin and {', '.join(scheme['covariates'])}"
        handles.append(Line2D([], [], marker=marker, linestyle="-", color=color,
                              label=f"matched pairs, {what} ({scheme['matched_positives']} positives)"))
    handles.append(Line2D([], [], linestyle="-", linewidth=1.6, color="#555555",
                          label="95% bootstrap interval over matched positives"))
    handles.append(Line2D([], [], linestyle="-", linewidth=0.8, alpha=0.5, color="#555555",
                          label="95% bootstrap interval, positives and negatives resampled"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.35, -0.16), fontsize=7.5)

    ax = axes[1]
    a, b = cfg["gbm"]["comparisons"][0]
    comp = gres["comparisons"][f"{a} - {b}"]
    metric_labels = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "mcc": "MCC"}
    strategies = list(cfg["gbm"]["strategies"])
    rows = [(m, s) for m in metric_labels for s in strategies]
    ys = np.arange(len(rows))[::-1]
    for yv, (m, s) in zip(ys, rows):
        v = comp[s][m]
        k = strategies.index(s)
        color, marker = PALETTE[k % len(PALETTE)], "Ds^v"[k % 4]
        ax.scatter(v["per_seed"], [yv] * len(v["per_seed"]), s=12, color=color, alpha=0.45, zorder=2)
        ax.errorbar([v["mean"]], [yv], xerr=[[v["mean"] - v["corrected_ci95"][0]], [v["corrected_ci95"][1] - v["mean"]]],
                    fmt=marker, color=color, markersize=5, capsize=2, linewidth=1.2, zorder=3)
    ax.axvline(0.0, color="#888888", linewidth=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{metric_labels[m]}, {s}" for m, s in rows])
    arms = cfg["gbm"]["arms"]
    ax.set_xlabel(f"{arms[a].get('label', a)} minus {arms[b].get('label', b)}")
    ax.set_title("(b) paired gain of the embedding in the boosting model", fontsize=9, loc="left")
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", color="#888888", alpha=0.45, label="mean over the folds of a seed"),
                       Line2D([], [], marker="D", linestyle="-", color="#888888",
                              label="mean over seeds, corrected 95% interval")],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=7.5)
    name = str(fcfg.get("name", "dc_matching"))
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=int(fcfg.get("dpi", 200)))
    plt.close(fig)
    print(f"[figure] wrote {out_dir / name}.pdf and .png", flush=True)


# -------------------------------------------------------------------- main


def context(cfg: dict[str, Any]) -> dict[str, Any]:
    folds = load_folds(resolve(cfg["folds"]))
    label_order = list(folds["meta"]["label_order"])
    col = label_order.index(cfg["label"])
    entries = json.loads(resolve(cfg["input_json"]).read_text())
    level = {int(r[cfg["level_key"]]): r[cfg["level_column"]] for r in csv.DictReader(resolve(cfg["level_csv"]).open())}
    pool = [int(j) for j in folds["meta"]["json_index"] if level.get(int(j)) == cfg["level"]]
    y = {j: int(bool(entries[j]["y"][col])) for j in pool}
    return {"folds": folds, "entries": entries, "label_col": col, "pool": pool, "pool_set": set(pool), "y": y,
            "level_counts": dict(Counter(level.values()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated subset of {STAGES}")
    args, extra = parser.parse_known_args()
    cfg = load_config(resolve(args.config), parse_cli_overrides(extra))["dc_matching"]
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages {unknown}, choose among {STAGES}")
    ctx = context(cfg)
    print(f"[dc] {cfg['label']} at level {cfg['level']}: {len(ctx['pool'])} indexed samples, "
          f"{sum(ctx['y'].values())} positives", flush=True)
    for stage in STAGES:
        if stage in stages:
            {"features": stage_features, "gbm": stage_gbm, "gbm_summary": stage_gbm_summary,
             "matching": stage_matching, "figure": stage_figure}[stage](cfg, ctx)


if __name__ == "__main__":
    main()
