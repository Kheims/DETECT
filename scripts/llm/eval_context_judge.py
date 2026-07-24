"""Evaluate the LLM context study (scripts/llm/llm_context_judge.py).

Writes cross-validation records per condition, then ranking (ROC-AUC, PR-AUC),
MCC with frozen and retuned thresholds, contrasts, coverage, costs and figures,
with repository-clustered bootstrap intervals, under <eval.report_dir>/<judge><tag>/.

Usage:
    python scripts/llm/eval_context_judge.py --config config/experiments/context_study.yml
    python scripts/llm/eval_context_judge.py --config ... --judge qwen25_coder_7b_instruct --context_study.eval.missing=drop
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.llm.build_context_packages import read_levels, repo_name  # noqa: E402
from scripts.llm.eval_llm_judge import evaluate_run, load_scores, run_name  # noqa: E402
from scripts.llm.llm_context_judge import TOOL_OK, TOOL_UNRESOLVED, config_subset  # noqa: E402
from scripts.llm.llm_judge import label_vector  # noqa: E402
from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.constants import LABEL_ORDER  # noqa: E402
from mlcq_graphs.evaluation.cv import iter_folds, load_folds  # noqa: E402

METRICS = ("roc_auc", "pr_auc")
EARLY_STOPS = ("budget", "context_limit", "turn_length")


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def fmt(value: Any) -> str:
    return "NA" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{float(value):.3f}"


# ---------------------------------------------------------------- metrics


def fast_roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC-AUC from average ranks (Mann-Whitney U), equal to roc_auc_score."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fast_average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision with one threshold per distinct score, equal to
    average_precision_score."""
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    last = np.r_[np.where(np.diff(ss))[0], len(ys) - 1]
    tps = np.cumsum(ys)[last]
    precision = tps / (last + 1)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def metric(name: str, y: np.ndarray, s: np.ndarray, fast: bool) -> float:
    if name == "roc_auc":
        return fast_roc_auc(y, s) if fast else float(roc_auc_score(y, s))
    return fast_average_precision(y, s) if fast else float(average_precision_score(y, s))


def fast_tune(y: np.ndarray, p: np.ndarray, steps: int) -> np.float32:
    """Fast equivalent of cv.tune_thresholds for one label (max F1 on the grid)."""
    grid = np.linspace(0.0, 1.0, max(int(steps), 3), dtype=np.float32)
    n_pos = int(y.sum())
    if n_pos == 0:
        return np.float32(0.5)
    pred = p[None, :] >= grid[:, None]
    tp = (pred & (y[None, :] == 1)).sum(axis=1)
    fp = pred.sum(axis=1) - tp
    f1 = np.where(tp > 0, (2.0 * tp) / (n_pos + (tp + fp)), 0.0)
    return grid[int(np.argmax(f1))]


def mcc_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: np.ndarray) -> np.ndarray:
    """Matthews correlation from confusion counts (0 when a margin is empty,
    as sklearn.metrics.matthews_corrcoef)."""
    tp, fp, fn, tn = (np.asarray(v, dtype=np.float64) for v in (tp, fp, fn, tn))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (tp * tn - fp * fn) / den
    return np.where(den > 0, out, 0.0)


def mcc_binary(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int(np.sum((y == 1) & pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    tn = int(np.sum((y == 0) & ~pred))
    return float(mcc_from_counts(tp, fp, fn, tn))


def cluster_draws(groups: np.ndarray, draws: int, rng: np.random.Generator) -> tuple[list[np.ndarray], np.ndarray]:
    """Bootstrap draws that resample groups with replacement.

    Returns the sample positions of each draw and the (draws, groups) multiplicities."""
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="mergesort")
    bounds = np.r_[0, np.cumsum(np.bincount(inv, minlength=len(uniq)))]
    members = [order[bounds[k]:bounds[k + 1]] for k in range(len(uniq))]
    idx, weights = [], np.zeros((int(draws), len(uniq)), dtype=np.int64)
    for d in range(int(draws)):
        pick = rng.integers(0, len(uniq), len(uniq))
        weights[d] = np.bincount(pick, minlength=len(uniq))
        idx.append(np.concatenate([members[k] for k in pick]))
    return idx, weights


def interval(values: np.ndarray, level: float) -> list[float | None]:
    if len(values) == 0:
        return [None, None]
    lo_q, hi_q = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    return [float(np.percentile(values, lo_q)), float(np.percentile(values, hi_q))]


def paired_bootstrap(y: np.ndarray, scores: dict[str, np.ndarray], baseline: str, draws: list[np.ndarray],
                     level: float, contrasts: list[tuple[str, str]] = ()
                     ) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    """ROC-AUC and PR-AUC per condition with intervals and paired differences.

    Also returns the contrasts A minus B, keyed (A, B, metric), on the same draws."""
    names = list(scores)
    out: dict[str, dict[str, Any]] = {c: {} for c in names}
    for c in names:
        for m in METRICS:
            out[c][m] = metric(m, y, scores[c], fast=False)
    samples = {(c, m): [] for c in names for m in METRICS}
    skipped = 0
    for idx in draws:
        yb = y[idx]
        if yb.min() == yb.max():
            skipped += 1
            continue
        for c in names:
            sb = scores[c][idx]
            for m in METRICS:
                samples[(c, m)].append(metric(m, yb, sb, fast=True))
    used = len(draws) - skipped
    for c in names:
        for m in METRICS:
            values = np.array(samples[(c, m)])
            out[c][f"{m}_ci"] = interval(values, level)
            if baseline in scores and c != baseline:
                diff = values - np.array(samples[(baseline, m)])
                out[c][f"{m}_diff"] = out[c][m] - out[baseline][m]
                out[c][f"{m}_diff_ci"] = interval(diff, level)
                out[c][f"{m}_diff_share_le_0"] = float(np.mean(diff <= 0)) if len(diff) else None
        out[c]["bootstrap_draws_used"] = used
    paired: dict[tuple[str, str, str], dict[str, Any]] = {}
    for a, b in contrasts:
        if a not in scores or b not in scores:
            continue
        for m in METRICS:
            diff = np.array(samples[(a, m)]) - np.array(samples[(b, m)])
            paired[(a, b, m)] = {"value_a": out[a][m], "value_b": out[b][m], "diff": out[a][m] - out[b][m],
                                 "diff_ci": interval(diff, level),
                                 "diff_share_le_0": float(np.mean(diff <= 0)) if len(diff) else None,
                                 "bootstrap_draws_used": used}
    return out, paired


# ---------------------------------------------------------------- data


def condition_rows(score_dir: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], list[str]] | None:
    if not score_dir.is_dir() or not list(score_dir.glob("scores*.jsonl")):
        return None
    return load_scores(score_dir)


def score_value(row: dict[str, Any], nan_fill: float) -> tuple[float, bool]:
    value = row["score"]
    if value is None or math.isnan(float(value)):
        return nan_fill, True
    return float(value), False


def rescale(scores: np.ndarray, applicable: np.ndarray, temperature: float) -> np.ndarray:
    """sigmoid(logit(score) / T) on applicable cells, as eval_llm_judge.score_matrix."""
    if temperature == 1.0:
        return scores
    clipped = np.clip(scores, 1e-15, 1 - 1e-15)
    rescaled = 1.0 / (1.0 + np.exp(-np.log(clipped / (1 - clipped)) / temperature))
    return np.where(applicable, rescaled, scores)


def aligned_pairs(scope: str, requested: dict[str, list[int]],
                  conditions: dict[str, dict[tuple[int, str], dict[str, Any]]], missing: str,
                  need_applicable: bool) -> dict[str, list[int]]:
    """Per smell, the requested json_index scored in every condition.

    With missing=error, a missing score stops the evaluation."""
    out = {}
    for smell, wanted in requested.items():
        keep = []
        for i in wanted:
            ok = True
            for rows in conditions.values():
                row = rows.get((i, smell))
                if row is None or (need_applicable and not row.get("applicable")):
                    ok = False
                    break
            if ok:
                keep.append(i)
        if len(keep) < len(wanted) and missing == "error":
            lacking = {c: sum(1 for i in wanted if (i, smell) not in rows) for c, rows in conditions.items()}
            raise SystemExit(f"scope {scope}, {smell}: {len(wanted) - len(keep)} of {len(wanted)} requested pairs "
                             f"are not scored in every condition (missing per condition {lacking}); run the "
                             "missing conditions or set eval.missing=drop")
        out[smell] = keep
    return out


# ---------------------------------------------------------------- ranking


def ranking_scope(
    scope: str,
    pairs: dict[str, list[int]],
    requested: dict[str, list[int]],
    conditions: dict[str, dict[tuple[int, str], dict[str, Any]]],
    labels: dict[int, list[int]],
    repos: dict[int, str],
    label_order: list[str],
    baseline: str,
    bcfg: dict[str, Any],
    nan_fill: float,
    scope_id: int,
    contrasts: list[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rows of the ranking table of one scope (one per smell and condition)
    and rows of the contrast table (one per smell, contrast and metric)."""
    table, paired_rows = [], []
    for col, smell in enumerate(label_order):
        common = pairs.get(smell) or []
        if not common or not conditions:
            continue
        y = np.array([labels[i][col] for i in common], dtype=np.int64)
        if y.min() == y.max():
            continue
        scores, nan = {}, {}
        for c, rows in conditions.items():
            values = [score_value(rows[(i, smell)], nan_fill) for i in common]
            scores[c] = np.array([v for v, _ in values], dtype=np.float64)
            nan[c] = sum(1 for _, filled in values if filled)
        rng = np.random.default_rng([int(bcfg["seed"]), scope_id, col])
        groups = np.array([repos[i] for i in common])
        draws, _ = cluster_draws(groups, int(bcfg["draws"]), rng)
        stats, paired = paired_bootstrap(y, scores, baseline, draws, float(bcfg["level"]), contrasts)
        common_cols = {"n": len(common), "positives": int(y.sum()), "repositories": int(len(set(groups)))}
        for c in scores:
            table.append({"scope": scope, "smell": smell, "condition": c, **common_cols,
                          "requested": len(requested[smell]), "nan_filled": nan[c], **stats[c]})
        for (a, b, m), values in paired.items():
            paired_rows.append({"scope": scope, "smell": smell, "contrast": f"{a} - {b}", "a": a, "b": b,
                                "metric": m, **common_cols, **values})
    return table, paired_rows


# ---------------------------------------------------------------- MCC


def mcc_scope(
    scope: str,
    sets: dict[str, np.ndarray],
    scores: dict[str, dict[str, np.ndarray]],
    ys: dict[str, np.ndarray],
    groups: dict[str, np.ndarray],
    folds: dict[str, Any],
    ecfg: dict[str, Any],
    label_order: list[str],
    baseline: str,
    temperature: float,
    scope_id: int,
    contrasts: list[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """MCC rows of one scope and temperature per smell, strategy, condition and threshold mode.

    Also returns the contrast rows A minus B on the same bootstrap draws."""
    steps = int(ecfg["threshold_steps"])
    modes = list(ecfg["mcc_threshold_modes"])
    bcfg = dict(ecfg["bootstrap"])
    level = float(bcfg["level"])
    names = list(scores)
    table, paired_rows = [], []
    for col, smell in enumerate(label_order):
        idx = sets.get(smell)
        if idx is None or len(idx) == 0 or baseline not in scores:
            continue
        y = ys[smell]
        pos = {int(i): k for k, i in enumerate(idx)}
        uniq, rid = np.unique(groups[smell], return_inverse=True)
        n_rep = len(uniq)
        fold_mcc: dict[tuple, list[float]] = {}
        pooled: dict[tuple, np.ndarray] = {}
        seen: dict[tuple, np.ndarray] = {}
        for fold in iter_folds(folds, ecfg["strategies"], ecfg["seeds"], ecfg["folds_subset"]):
            val = np.array([pos[int(i)] for i in fold.val if int(i) in pos], dtype=np.int64)
            test = np.array([pos[int(i)] for i in fold.test if int(i) in pos], dtype=np.int64)
            if len(val) == 0 or len(test) == 0:
                continue
            thr = {c: fast_tune(y[val], scores[c][smell][val], steps) for c in names}
            key_seed = (fold.strategy, fold.seed)
            if key_seed not in seen:
                seen[key_seed] = np.zeros(len(idx), dtype=np.int64)
            seen[key_seed][test] += 1
            for c in names:
                for mode in modes:
                    t = thr[baseline] if mode == "frozen" else thr[c]
                    pred = scores[c][smell][test] >= t
                    key = (c, mode, fold.strategy, fold.seed)
                    if int(y[test].sum()) > 0:
                        fold_mcc.setdefault(key, []).append(mcc_binary(y[test], pred))
                    if key not in pooled:
                        pooled[key] = np.zeros(len(idx), dtype=bool)
                    pooled[key][test] = pred
        for key_seed, count in seen.items():
            if count.max() > 1:
                raise SystemExit(f"{scope} {smell}: a sample is in the test part of two folds of {key_seed}")
        rng = np.random.default_rng([int(bcfg["seed"]), scope_id, col, 7])
        weights = np.stack([np.bincount(rng.integers(0, n_rep, n_rep), minlength=n_rep)
                            for _ in range(int(bcfg["draws"]))])
        strategies = sorted({k[0] for k in seen})
        for strategy in strategies:
            seeds = sorted(s for st, s in seen if st == strategy)
            boot: dict[tuple[str, str], np.ndarray] = {}
            point_pooled: dict[tuple[str, str], float] = {}
            for c in names:
                for mode in modes:
                    per_seed_draws, per_seed_point = [], []
                    for seed in seeds:
                        mask = seen[(strategy, seed)] > 0
                        pred = pooled[(c, mode, strategy, seed)]
                        cells = np.zeros((n_rep, 4))
                        for j, sel in enumerate(((y == 1) & pred, (y == 0) & pred, (y == 1) & ~pred,
                                                 (y == 0) & ~pred)):
                            cells[:, j] = np.bincount(rid[mask & sel], minlength=n_rep)
                        per_seed_point.append(float(mcc_from_counts(*cells.sum(axis=0))))
                        drawn = weights @ cells
                        per_seed_draws.append(mcc_from_counts(drawn[:, 0], drawn[:, 1], drawn[:, 2], drawn[:, 3]))
                    boot[(c, mode)] = np.mean(per_seed_draws, axis=0)
                    point_pooled[(c, mode)] = float(np.mean(per_seed_point))
            # fold mean then seed mean
            seed_means = {(c, mode): [float(np.mean(fold_mcc[(c, mode, strategy, s)])) for s in seeds
                                      if fold_mcc.get((c, mode, strategy, s))]
                          for c in names for mode in modes}
            mean_mcc = {k: (float(np.mean(v)) if v else None) for k, v in seed_means.items()}
            common_cols = {"scope": scope, "smell": smell, "strategy": strategy, "score_temperature": temperature}
            size_cols = {"n": int(len(idx)), "positives": int(y.sum()), "repositories": int(n_rep)}
            # the baseline threshold is the same in both modes
            base_key = (baseline, "retuned" if "retuned" in modes else modes[0])
            for c in names:
                for mode in modes:
                    means = seed_means[(c, mode)]
                    row = {**common_cols, "condition": c, "threshold": mode, **size_cols, "seeds": len(means),
                           "mcc": mean_mcc[(c, mode)],
                           "mcc_seed_sd": float(np.std(means, ddof=1)) if len(means) > 1 else None,
                           "mcc_pooled": point_pooled[(c, mode)],
                           "mcc_pooled_ci": interval(boot[(c, mode)], level)}
                    if (c, mode) != base_key and means and seed_means[base_key]:
                        diff = boot[(c, mode)] - boot[base_key]
                        row.update({"mcc_diff": row["mcc"] - mean_mcc[base_key],
                                    "mcc_pooled_diff": point_pooled[(c, mode)] - point_pooled[base_key],
                                    "mcc_pooled_diff_ci": interval(diff, level),
                                    "mcc_pooled_diff_share_le_0": float(np.mean(diff <= 0))})
                    table.append(row)
            for a, b in contrasts:
                if a not in names or b not in names:
                    continue
                for mode in modes:
                    ka, kb = (a, mode), (b, mode)
                    diff = boot[ka] - boot[kb]
                    both = mean_mcc[ka] is not None and mean_mcc[kb] is not None
                    paired_rows.append({**common_cols, "contrast": f"{a} - {b}", "a": a, "b": b, "metric": "mcc",
                                        "threshold": mode, **size_cols, "value_a": mean_mcc[ka],
                                        "value_b": mean_mcc[kb], "diff": mean_mcc[ka] - mean_mcc[kb] if both else None,
                                        "pooled_a": point_pooled[ka], "pooled_b": point_pooled[kb],
                                        "pooled_diff": point_pooled[ka] - point_pooled[kb],
                                        "pooled_diff_ci": interval(diff, level),
                                        "pooled_diff_share_le_0": float(np.mean(diff <= 0))})
    return table, paired_rows


def mcc_inputs(
    scope: str,
    pairs: dict[str, list[int]],
    conditions: dict[str, dict[tuple[int, str], dict[str, Any]]],
    labels: dict[int, list[int]],
    repos: dict[int, str],
    label_order: list[str],
    nan_fill: float,
    temperature: float,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    sets, scores, ys, groups = {}, {c: {} for c in conditions}, {}, {}
    for col, smell in enumerate(label_order):
        idx = np.array(sorted(pairs.get(smell) or []), dtype=np.int64)
        if len(idx) == 0:
            continue
        sets[smell] = idx
        ys[smell] = np.array([labels[int(i)][col] for i in idx], dtype=np.int64)
        groups[smell] = np.array([repos[int(i)] for i in idx])
        for c, rows in conditions.items():
            cells = [rows[(int(i), smell)] for i in idx]
            raw = np.array([score_value(r, nan_fill)[0] for r in cells], dtype=np.float64)
            applicable = np.array([bool(r.get("applicable")) for r in cells])
            scores[c][smell] = rescale(raw, applicable, temperature)
    return sets, scores, ys, groups


# ---------------------------------------------------------------- coverage and costs


def top_is_answer(row: dict[str, Any]) -> bool:
    """True when the most probable first token is a yes or no token (the
    first matched token of llm_judge.score_answer has the top logprob)."""
    matched = row.get("matched") or []
    top = row.get("top_token") or [None, None]
    return bool(matched) and top[1] is not None and matched[0][2] == top[1]


def coverage_scope(scope: str, pairs: dict[str, list[int]], requested: dict[str, list[int]],
                   conditions: dict[str, dict[tuple[int, str], dict[str, Any]]]) -> list[dict[str, Any]]:
    table = []
    wanted = [(i, s) for s, ids in requested.items() for i in ids]
    kept = {(i, s) for s, ids in pairs.items() for i in ids}
    for c, rows in conditions.items():
        part = [rows[k] for k in wanted if k in rows and rows[k].get("applicable")]
        row: dict[str, Any] = {"scope": scope, "condition": c, "requested": len(wanted), "scored": len(part),
                               "compared": sum(1 for k in wanted if k in kept),
                               "nan_scores": sum(1 for r in part if r["score"] is None or math.isnan(float(r["score"])))}
        sections = [r.get("context") or {} for r in part]
        with_sections = [s for s in sections if any("available" in v for v in s.values())]
        if with_sections:
            n = len(with_sections)
            row.update({
                "context_all_available": sum(1 for s in with_sections
                                             if all(v.get("available", True) for v in s.values())) / n,
                "context_any_unavailable": sum(1 for s in with_sections
                                               if any(not v.get("available", True) for v in s.values())) / n,
                "context_any_empty": sum(1 for s in with_sections
                                         if any(v.get("available", True) and v.get("empty") for v in s.values())) / n,
                "context_any_cut": sum(1 for s in with_sections if any(v.get("truncated") for v in s.values())) / n,
                "unavailable_by_section": dict(Counter(k for s in with_sections for k, v in s.items()
                                                       if not v.get("available", True))),
            })
        if part:
            row["context_scaled_down"] = sum(1 for r in part if r.get("context_scale", 1.0) < 1.0)
            row["snippet_truncated"] = sum(1 for r in part if r.get("truncated"))
            # first-token diagnostics
            row["top_token_not_answer"] = sum(1 for r in part if not top_is_answer(r))
            row["answer_mass_mean"] = float(np.mean([float(r.get("p_yes", 0.0)) + float(r.get("p_no", 0.0))
                                                     for r in part]))
            row["top_tokens"] = dict(Counter(str((r.get("top_token") or [None])[0]) for r in part).most_common(5))
        if c == "C4" and part:
            calls = [t for r in part for t in r.get("tool_calls", [])]
            status = Counter(t["status"] for t in calls)
            stops = Counter(r.get("stop_reason") for r in part)
            row.update({
                "tool_calls": len(calls),
                "tool_ok_share": sum(status[s] for s in TOOL_OK) / len(calls) if calls else None,
                "tool_failed": len(calls) - sum(status[s] for s in TOOL_OK),
                "tool_unresolved_symbol": sum(status[s] for s in TOOL_UNRESOLVED),
                "tool_status": dict(status.most_common()),
                "conversations_without_call": sum(1 for r in part if not r.get("tool_calls")),
                "conversations_without_ok_call": sum(1 for r in part
                                                     if not any(t["status"] in TOOL_OK for t in r.get("tool_calls", []))),
                "stop_reasons": dict(stops.most_common()),
                "stopped_early": sum(stops[s] for s in EARLY_STOPS if s != "budget"),
                "budget_used_up": stops["budget"],
                "final_outputs_replaced": sum((r.get("final_fit") or {}).get("outputs_replaced", 0) for r in part),
                "final_messages_dropped": sum((r.get("final_fit") or {}).get("messages_dropped", 0) for r in part),
            })
        if c == "REPLAY" and part:
            row["queries_without_tool_output"] = sum(1 for r in part
                                                     if not ((r.get("context") or {}).get("replay") or {}).get("outputs"))
        table.append(row)
    return table


def costs_scope(scope: str, pairs: set[tuple[int, str]], conditions: dict[str, dict[tuple[int, str], dict[str, Any]]],
                metas: dict[str, dict[str, Any]], label_order: list[str]) -> list[dict[str, Any]]:
    """Cost rows of one scope: per condition over all smells (smell "all"),
    then per condition and smell."""
    table = []
    for smell in ["all"] + list(label_order):
        for c, rows in conditions.items():
            part = [r for key, r in rows.items()
                    if key in pairs and r.get("applicable") and smell in ("all", key[1])]
            if part:
                table.append(cost_row(scope, smell, c, part, metas))
    return table


def cost_row(scope: str, smell: str, c: str, part: list[dict[str, Any]],
             metas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Mean tokens and seconds per query of the rows of one condition (C4:
    summed over turns), tool calls and the run seconds of meta.json."""
    prompt = np.array([r.get("prompt_tokens_total", r["prompt_tokens"]) for r in part], dtype=np.float64)
    gen = np.array([r.get("generated_tokens", 1) for r in part], dtype=np.float64)
    secs = np.array([r.get("seconds_share", 0.0) for r in part], dtype=np.float64)
    row = {"scope": scope, "smell": smell, "condition": c, "queries": len(part),
           "prompt_tokens_mean": float(prompt.mean()), "prompt_tokens_p95": float(np.percentile(prompt, 95)),
           "context_tokens_mean": float(np.mean([r.get("context_tokens", 0) for r in part])),
           "generated_tokens_mean": float(gen.mean()), "tokens_per_query": float((prompt + gen).mean()),
           "seconds_per_query": float(secs.mean()), "seconds_total": float(secs.sum()),
           "final_prompt_tokens_mean": float(np.mean([r["prompt_tokens"] for r in part])),
           "truncated_snippets": sum(1 for r in part if r.get("truncated"))}
    if c == "C4":
        calls = [r.get("n_tool_calls", 0) for r in part]
        row.update({
            "tool_calls_mean": float(np.mean(calls)),
            "tool_calls_max": int(max(calls)),
            "turns_mean": float(np.mean([r.get("turns", 0) for r in part])),
            "tools": dict(Counter(t["name"] for r in part for t in r.get("tool_calls", [])).most_common()),
        })
    runs = metas.get(c, {}).get("runs", [])
    row["meta_generate_seconds"] = float(sum(r.get("generate_seconds", 0.0) for r in runs))
    row["meta_load_seconds"] = float(sum(r.get("load_seconds", 0.0) for r in runs))
    row["gpus"] = sorted({str(r.get("gpu")) for r in runs})
    return row


# ---------------------------------------------------------------- figure


def plot_scope(ranking: list[dict[str, Any]], costs: list[dict[str, Any]], scope: str, metric_name: str,
               order: list[str], baseline: str, fig_cfg: dict[str, Any], title: str, out_stem: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[eval] matplotlib not available: figure skipped", flush=True)
        return False
    rows = [r for r in ranking if r["scope"] == scope]
    if not rows:
        return False
    cost = {(r["condition"], r["smell"]): r for r in costs if r["scope"] == scope}
    smells = [s for s in LABEL_ORDER if any(r["smell"] == s for r in rows)]
    conds = [c for c in order if any(r["condition"] == c for r in rows)]
    mark, ref, ink, muted = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e"
    fig, axes = plt.subplots(3, len(smells), figsize=tuple(fig_cfg["summary_size"]), squeeze=False,
                             gridspec_kw={"height_ratios": [2.2, 1, 1]})
    x = np.arange(len(conds))
    cost_rows = ((1, "tokens_per_query", "tokens per query"), (2, "seconds_per_query", "seconds per query"))
    # one scale per cost row, log when the agent costs dominate
    log_rows = set()
    for row_i, key, _ in cost_rows:
        values = [cost[(c, s)][key] for s in smells for c in conds if (c, s) in cost]
        if values and min(values) > 0 and max(values) / min(values) > 20:
            log_rows.add(row_i)
    for k, smell in enumerate(smells):
        by_c = {r["condition"]: r for r in rows if r["smell"] == smell}
        ax = axes[0][k]
        for j, c in enumerate(conds):
            r = by_c.get(c)
            if r is None:
                continue
            lo, hi = r[f"{metric_name}_ci"]
            v = r[metric_name]
            if lo is not None:
                ax.plot([j, j], [lo, hi], color=mark, linewidth=2, solid_capstyle="round")
            ax.plot([j], [v], marker="o", markersize=6, color=mark, markeredgecolor="white", markeredgewidth=1.2)
        if baseline in by_c:
            ax.axhline(by_c[baseline][metric_name], color=ref, linewidth=1, linestyle="--")
        any_row = next(iter(by_c.values()))
        ax.set_title(f"{smell.removeprefix('is_').replace('_', ' ')}\n(n={any_row['n']}, "
                     f"positives={any_row['positives']})", fontsize=9, color=ink)
        ax.set_xticks(x, conds, fontsize=6, rotation=45, ha="right")
        ax.tick_params(labelsize=7, colors=muted)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        if k == 0:
            ax.set_ylabel(f"{metric_name.replace('_', '-').upper()}\n(95% interval, repositories resampled)",
                          fontsize=8, color=ink)
        for row_i, key, label in cost_rows:
            axc = axes[row_i][k]
            values = [cost[(c, smell)][key] if (c, smell) in cost else 0.0 for c in conds]
            axc.bar(x, values, width=0.6, color=mark)
            if row_i in log_rows:
                axc.set_yscale("log")
            axc.set_xticks(x, conds, fontsize=6, rotation=45, ha="right")
            axc.tick_params(labelsize=7, colors=muted)
            axc.grid(axis="y", color="#e6e5e0", linewidth=0.6)
            axc.set_axisbelow(True)
            if k == 0:
                axc.set_ylabel(label, fontsize=8, color=ink)
        for a in axes[:, k]:
            a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=10, color=ink)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- main


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for r in rows:
        columns.extend(k for k in r if k not in columns)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})


def evaluate(cfg: dict[str, Any], judge: str) -> Path:
    ccfg = cfg["context_study"]
    jcfg, ecfg = ccfg["judge"], ccfg["eval"]
    name = f"{judge}{jcfg.get('tag') or ''}"
    judge_cfg = jcfg["judges"][judge]
    model = str(judge_cfg.get("hf_id") or judge_cfg.get("model_path"))
    score_root = resolve(jcfg["out_root"]) / name
    report = resolve(ecfg["report_dir"]) / name
    report.mkdir(parents=True, exist_ok=True)
    missing = str(ecfg["missing"])
    if missing not in ("error", "drop"):
        raise SystemExit(f"eval.missing must be error or drop, not {missing!r}")
    if str(ecfg["bootstrap"].get("cluster", "repository")) != "repository":
        raise SystemExit("eval.bootstrap.cluster: only repository is implemented")

    folds = load_folds(resolve(ccfg["folds"]))
    label_order = list(folds["meta"]["label_order"])
    if label_order != list(LABEL_ORDER):
        raise SystemExit(f"folds label order {label_order} differs from LABEL_ORDER {LABEL_ORDER}")
    entries = json.loads(resolve(ccfg["input_json"]).read_text())
    labels = {i: label_vector(e) for i, e in enumerate(entries)}
    repos = {i: repo_name(e["repo_url"]) for i, e in enumerate(entries)}
    levels = read_levels(resolve(ccfg["sample_meta"]))
    prompts = yaml.safe_load(resolve(jcfg["prompts"]).read_text())
    smell_levels = {s: str(prompts["smells"][s]["level"]) for s in label_order}
    indices = [int(i) for i in folds["meta"]["json_index"]]
    baseline = str(ecfg["baseline"])
    nan_fill = float(ecfg["nan_fill"])

    wanted = list(dict.fromkeys(list(ecfg["cv_conditions"]) + list(ecfg["subset_conditions"])))
    contrasts = []
    for pair in ecfg.get("contrasts") or []:
        if len(pair) != 2 or pair[0] == pair[1] or any(c not in wanted for c in pair):
            raise SystemExit(f"eval.contrasts: {pair} is not a pair of two conditions of cv_conditions or "
                             f"subset_conditions")
        contrasts.append((str(pair[0]), str(pair[1])))
    conditions: dict[str, dict[tuple[int, str], dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    metas: dict[str, dict[str, Any]] = {}
    for c in wanted:
        loaded = condition_rows(score_root / c)
        if loaded is None:
            print(f"[eval] {name} {c}: no scores under {score_root / c}, skipped", flush=True)
            continue
        rows, hs = loaded
        if len(hs) != 1:
            raise SystemExit(f"{score_root / c} mixes prompt hashes {hs}")
        conditions[c], hashes[c] = rows, hs[0]
        metas[c] = {"runs": []}
        for mpath in sorted((score_root / c).glob("meta*.json")):
            meta = json.loads(mpath.read_text())
            metas[c]["runs"].extend(meta.get("runs", []))
    if baseline not in conditions:
        raise SystemExit(f"no scores of the baseline {baseline} under {score_root}")

    # 1. cross-validation records
    temperatures = [float(t) for t in ecfg["score_temperatures"]]
    for c in ecfg["cv_conditions"]:
        if c not in conditions:
            continue
        scored = {i for i in indices if all((i, s) in conditions[c] for s in label_order)}
        empty = [(f.strategy, f.seed, f.fold) for f in iter_folds(folds, ecfg["strategies"], ecfg["seeds"],
                                                                  ecfg["folds_subset"])
                 if not scored.intersection(map(int, f.val)) or not scored.intersection(map(int, f.test))]
        if empty:
            # only possible with missing=drop on a small smoke run
            print(f"[eval] {c}: no records, {len(empty)} folds have an empty scored validation or test part "
                  f"(first {empty[0]})", flush=True)
            continue
        for t in temperatures:
            evaluate_run(ecfg, judge, model, run_name(f"{name}_{c}", "", t), t, folds, label_order, entries,
                         conditions[c], hashes[c])

    # scopes: every applicable pair and the strata of the agent subset
    saved = score_root / "C4" / "subset.json"
    if ecfg.get("subset_from_saved"):
        # score the pairs that were run, under the current labels
        subset = json.loads(saved.read_text())["strata"]
    else:
        subset = config_subset(jcfg, indices, levels, labels, label_order, smell_levels)
        if saved.exists():
            strata = json.loads(saved.read_text())["strata"]
            if strata != json.loads(json.dumps(subset)):
                raise SystemExit(f"{saved} differs from the subset of the config: rerun C4 or fix agent_subset")
    requested = {
        "all": {s: [i for i in indices if levels[i] == smell_levels[s]] for s in label_order},
        "subset": {s: sorted(subset[s]["positives"] + subset[s]["negatives"]) for s in label_order},
    }
    scoped = {"all": {c: conditions[c] for c in ecfg["cv_conditions"] if c in conditions},
              "subset": {c: conditions[c] for c in ecfg["subset_conditions"] if c in conditions}}
    pairs = {scope: aligned_pairs(scope, requested[scope], scoped[scope], missing, True) for scope in scoped}
    for scope in scoped:
        print(f"[eval] scope {scope}: conditions {list(scoped[scope])}; compared pairs per smell "
              f"{ {s: len(v) for s, v in pairs[scope].items()} } of requested "
              f"{ {s: len(v) for s, v in requested[scope].items()} }", flush=True)

    # 2. ranking
    bcfg = dict(ecfg["bootstrap"])
    ranking, paired_rows = [], []
    for scope_id, scope in enumerate(("all", "subset")):
        rows, paired = ranking_scope(scope, pairs[scope], requested[scope], scoped[scope], labels, repos,
                                     label_order, baseline, bcfg, nan_fill, scope_id, contrasts)
        ranking += rows
        paired_rows += paired
    for r in ranking:
        diff = "" if "roc_auc_diff" not in r else (
            f" dROC={fmt(r['roc_auc_diff'])} [{fmt(r['roc_auc_diff_ci'][0])}, {fmt(r['roc_auc_diff_ci'][1])}]"
            f" dPR={fmt(r['pr_auc_diff'])} [{fmt(r['pr_auc_diff_ci'][0])}, {fmt(r['pr_auc_diff_ci'][1])}]")
        print(f"[eval] {r['scope']} {r['smell']} {r['condition']} n={r['n']} pos={r['positives']} "
              f"repos={r['repositories']} ROC={fmt(r['roc_auc'])} PR={fmt(r['pr_auc'])}{diff}", flush=True)

    # 3. MCC with frozen and retuned thresholds
    full_pairs = aligned_pairs("all (MCC)", {s: indices for s in label_order}, scoped["all"], missing, False)
    mcc_rows = []
    for t in temperatures:
        for scope_id, (scope, scope_pairs) in enumerate((("all", full_pairs), ("subset", pairs["subset"]))):
            if not scoped[scope] or baseline not in scoped[scope]:
                continue
            sets, scores, ys, groups = mcc_inputs(scope, scope_pairs, scoped[scope], labels, repos, label_order,
                                                  nan_fill, t)
            rows, paired = mcc_scope(scope, sets, scores, ys, groups, folds, ecfg, label_order, baseline, t,
                                     scope_id, contrasts)
            mcc_rows += rows
            paired_rows += paired
    for r in mcc_rows:
        if r["condition"] == baseline and r["threshold"] != "retuned":
            continue
        diff = "" if "mcc_diff" not in r else (
            f" dMCC={fmt(r['mcc_diff'])} pooled d={fmt(r['mcc_pooled_diff'])} "
            f"[{fmt(r['mcc_pooled_diff_ci'][0])}, {fmt(r['mcc_pooled_diff_ci'][1])}]")
        print(f"[eval] MCC {r['scope']} {r['smell']} {r['strategy']} T={r['score_temperature']:g} {r['condition']} "
              f"{r['threshold']}: {fmt(r['mcc'])} (pooled {fmt(r['mcc_pooled'])}){diff}", flush=True)
    for r in paired_rows:
        if r["metric"] == "mcc":
            print(f"[eval] contrast {r['scope']} {r['smell']} {r['contrast']} MCC {r['strategy']} "
                  f"T={r['score_temperature']:g} {r['threshold']}: d={fmt(r['diff'])} pooled d={fmt(r['pooled_diff'])} "
                  f"[{fmt(r['pooled_diff_ci'][0])}, {fmt(r['pooled_diff_ci'][1])}]", flush=True)
        else:
            print(f"[eval] contrast {r['scope']} {r['smell']} {r['contrast']} {r['metric']}: {fmt(r['value_a'])} - "
                  f"{fmt(r['value_b'])} = {fmt(r['diff'])} [{fmt(r['diff_ci'][0])}, {fmt(r['diff_ci'][1])}]",
                  flush=True)

    # 4. coverage and 5. costs
    coverage = []
    costs = []
    for scope in ("all", "subset"):
        coverage += coverage_scope(scope, pairs[scope], requested[scope], scoped[scope])
        costs += costs_scope(scope, {(i, s) for s, ids in requested[scope].items() for i in ids}, scoped[scope], metas,
                             label_order)
    for r in coverage:
        extra = ""
        if r["condition"] == "C4":
            extra = (f", tool calls {r.get('tool_calls')} (ok share {fmt(r.get('tool_ok_share'))}, unresolved symbol "
                     f"{r.get('tool_unresolved_symbol')}), stops {r.get('stop_reasons')}")
        print(f"[eval] coverage {r['scope']} {r['condition']}: scored {r['scored']}/{r['requested']}, NaN "
              f"{r['nan_scores']}, top token not yes or no {r.get('top_token_not_answer')}, mean P(yes)+P(no) "
              f"{fmt(r.get('answer_mass_mean'))}, all sections available {fmt(r.get('context_all_available'))}{extra}",
              flush=True)
    for r in costs:
        if r["smell"] != "all":
            continue
        print(f"[eval] cost {r['scope']} {r['condition']}: {r['queries']} queries, "
              f"{r['tokens_per_query']:.0f} tokens/query, {r['seconds_per_query']:.4f} s/query"
              + (f", {r['tool_calls_mean']:.2f} tool calls/query" if r["condition"] == "C4" else ""), flush=True)

    summary = {"judge": judge, "name": name, "model": model, "prompt_hashes": hashes, "baseline": baseline,
               "nan_fill": nan_fill, "missing": missing, "bootstrap": bcfg, "contrasts": contrasts,
               "subset": {s: {"positives": len(v["positives"]), "negatives": len(v["negatives"])}
                          for s, v in subset.items()},
               "note": "group_kfold holds projects out of threshold tuning but does not remove the pretraining "
                       "exposure of the judge to these open-source repositories"}
    for stem, rows in (("ranking", ranking), ("mcc", mcc_rows), ("contrasts", paired_rows), ("coverage", coverage),
                       ("costs", costs)):
        (report / f"{stem}.json").write_text(json.dumps({**summary, "rows": rows}, indent=1))
        write_csv(report / f"{stem}.csv", rows)

    # 6. figures
    if ecfg["figure"]["enabled"]:
        order = list(dict.fromkeys(list(ecfg["cv_conditions"]) + list(ecfg["subset_conditions"])))
        for scope in ("all", "subset"):
            for m in METRICS:
                stem = report / f"context_{scope}_{m}"
                if plot_scope(ranking, costs, scope, m, order, baseline, ecfg["figure"],
                              f"{name}: {scope} scope ({model})", stem):
                    print(f"[eval] wrote {stem}.pdf and .png", flush=True)
    print(f"[eval] wrote {report}", flush=True)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--judge", action="append", default=None, help="judge name (repeatable)")
    args, extra = parser.parse_known_args(argv)
    cfg = load_config(args.config, parse_cli_overrides(extra))
    for judge in args.judge or [str(cfg["context_study"]["judge"]["default_judge"])]:
        evaluate(cfg, judge)


if __name__ == "__main__":
    main()
