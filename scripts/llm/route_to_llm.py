"""Route a share of the MLCQ samples of a smell's level from a cheap model to the LLM judge.

Uses existing out-of-fold predictions only. For every smell, a routing policy
(random, uncertainty, learned or oracle) sends a budget of test samples to the
LLM judge, whose score replaces the cheap score. Writes fold_results.csv,
summary.csv, references.csv, contrasts.csv, summary.json and a figure.

Usage:
    python scripts/llm/route_to_llm.py --config config/experiments/routing.yml
    python scripts/llm/route_to_llm.py --config ... --routing.routed_scores='[llm]' --routing.out_dir=...
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.evaluation.cv import (  # noqa: E402
    folds_label_order,
    iter_folds,
    load_folds,
    read_jsonl,
    tune_thresholds,
)
from scripts.cv.summarize_cv import corrected_ttest, ratio_from_folds  # noqa: E402

Key = tuple[str, int, int]  # strategy, seed, fold
Cell = dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]  # part -> json_index -> (y, prob)

ROUTED_SCORES = ("llm", "mean")
REFERENCES = ("cheap_alone", "llm_alone", "always_on_mean")
DELTA_FIELD = {"cheap_alone": "delta_vs_cheap", "always_on_mean": "delta_vs_always_on", "llm_alone": "delta_vs_llm"}
COUNT_FIELDS = ("n_routed", "routed_cheap_errors", "fixed", "broken", "cheap_errors_level")
FOLD_FIELDS = ["cheap", "routed_score", "policy", "budget", "smell", "level", "strategy", "seed", "fold", "val_mcc",
               "mcc", "f1", "n_level_val", "n_level_test", "n_budget", "n_routed", "cheap_errors_level",
               "routed_cheap_errors", "fixed", "broken", "router_C", "router_fallback", "oracle_running_max",
               "delta_vs_cheap", "delta_vs_always_on", "delta_vs_llm"]
SUMMARY_FIELDS = ["cheap", "routed_score", "policy", "budget", "smell", "level", "mcc_mean", "mcc_std_seeds",
                  "delta_vs_cheap_mean", "delta_vs_cheap_std_seeds", "delta_vs_always_on_mean",
                  "delta_vs_always_on_std_seeds", "delta_vs_llm_mean", "delta_vs_llm_std_seeds", "f1_mean",
                  "val_mcc_mean", "llm_calls_per_fold", "llm_calls_per_1000_level", "llm_seconds_per_1000_level",
                  "cheap_seconds_per_1000_level", "total_seconds_per_1000_level", "n_level_test_per_fold",
                  "cheap_errors_level_per_fold", "routed_cheap_errors_per_fold", "fixed_per_fold", "broken_per_fold"]
REFERENCE_FIELDS = ["reference", "cheap", "smell", "level", "mcc_mean", "mcc_std_seeds", "f1_mean",
                    "llm_calls_per_1000_level", "llm_seconds_per_1000_level", "cheap_seconds_per_1000_level",
                    "total_seconds_per_1000_level"]
CONTRAST_FIELDS = ["cheap", "routed_score", "policy", "budget", "against", "smell", "level", "n_pairs", "mean_diff",
                   "std_seeds", "ci95_low", "ci95_high", "corrected_t", "corrected_p", "wins", "losses",
                   "llm_calls_per_1000_level"]


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def prediction_files(spec: dict[str, Any]) -> list[Path]:
    """Files listed under `predictions` (paths or globs), or the run folders of `run_records`."""
    if "run_records" in spec:
        rr = spec["run_records"]
        records = json.loads(resolve(rr["path"]).read_text())
        # run_dir may be a cluster path: only its last component names the local run folder
        return [resolve(rr["runs_dir"]) / Path(str(r["run_dir"])).name / str(rr["file"]) for r in records]
    files: list[Path] = []
    for pattern in spec["predictions"]:
        found = sorted(PROJECT_ROOT.glob(pattern)) if any(c in pattern for c in "*?[") else [resolve(pattern)]
        if not found:
            raise SystemExit(f"no prediction file matches {pattern}")
        files.extend(found)
    return files


def load_predictions(spec: dict[str, Any]) -> dict[Key, Cell]:
    """{(strategy, seed, fold): {part: {json_index: (y, prob)}}}. Stops on another model or a duplicate row."""
    out: dict[Key, Cell] = defaultdict(lambda: defaultdict(dict))
    model = str(spec["model"])
    for file in prediction_files(spec):
        if not file.exists():
            raise SystemExit(f"missing prediction file {file}")
        for row in read_jsonl(file):
            if str(row["model"]) != model:
                raise SystemExit(f"{file}: model {row['model']!r}, expected {model!r}")
            key = (str(row["strategy"]), int(row["seed"]), int(row["fold"]))
            idx = int(row["json_index"])
            if idx in out[key][row["part"]]:
                raise SystemExit(f"{file}: duplicate prediction for {key} {row['part']} json_index {idx}")
            out[key][row["part"]][idx] = (np.asarray(row["y"], dtype=np.int64),
                                          np.asarray(row["prob"], dtype=np.float64))
    return out


def stack(cell: dict[int, tuple[np.ndarray, np.ndarray]], order: list[int]) -> tuple[np.ndarray, np.ndarray]:
    missing = [i for i in order if i not in cell]
    if missing:
        raise SystemExit(f"{len(missing)} fold samples have no prediction, e.g. json_index {missing[:5]}")
    return np.stack([cell[i][0] for i in order]), np.stack([cell[i][1] for i in order])


def load_levels(spec: dict[str, Any]) -> dict[int, str]:
    with resolve(spec["path"]).open() as fh:
        return {int(r[spec["key"]]): str(r[spec["column"]]) for r in csv.DictReader(fh)}


def load_size(spec: dict[str, Any]) -> dict[int, np.ndarray]:
    """log1p of the size counts, keyed by json index."""
    with resolve(spec["path"]).open() as fh:
        rows = list(csv.DictReader(fh))
    key = str(spec["key"])
    exclude = {key, *spec["exclude_columns"]}
    columns = [c for c in rows[0] if c not in exclude]
    return {int(float(r[key])): np.log1p(np.maximum([float(r[c]) for c in columns], 0.0)) for r in rows}


def confusion(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return tp, fp, fn, len(y) - tp - fp - fn


def mcc_counts(tp: int, fp: int, fn: int, tn: int) -> float:
    """Binary MCC, 0 when a margin is empty (as sklearn.metrics.matthews_corrcoef)."""
    den = math.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    return 0.0 if den == 0 else (tp * tn - fp * fn) / den


def mcc(y: np.ndarray, pred: np.ndarray) -> float:
    return mcc_counts(*confusion(y, pred))


def f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp, fp, fn, _ = confusion(y, pred)
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def threshold(y: np.ndarray, p: np.ndarray, steps: int) -> float:
    return float(tune_thresholds(y.reshape(-1, 1), p.reshape(-1, 1), steps)[0])


def budget_size(n: int, share: float) -> int:
    return int(round(share * n))


def route(priority: np.ndarray, share: float, tiebreak: np.ndarray) -> np.ndarray:
    """Mask of the round(share * n) samples with the highest priority (ties broken by `tiebreak`)."""
    n = budget_size(len(priority), share)
    mask = np.zeros(len(priority), dtype=bool)
    if n > 0:
        mask[np.lexsort((tiebreak, -priority))[:n]] = True
    return mask


def oracle_route(y: np.ndarray, base: np.ndarray, routed: np.ndarray, n: int,
                 dist: np.ndarray) -> tuple[np.ndarray, float]:
    """Best routing of at most n samples given the labels, and its MCC.

    Routes only cheap errors that the routed prediction fixes."""
    fix = (base != y) & (routed == y)
    fn_idx, fp_idx = np.flatnonzero(fix & (y == 1)), np.flatnonzero(fix & (y == 0))
    fn_idx, fp_idx = fn_idx[np.argsort(dist[fn_idx], kind="stable")], fp_idx[np.argsort(dist[fp_idx], kind="stable")]
    k = min(n, len(fn_idx) + len(fp_idx))
    tp, fp, fn, tn = confusion(y, base)
    best_score, best_a = -2.0, 0
    for a in range(max(0, k - len(fp_idx)), min(k, len(fn_idx)) + 1):
        score = mcc_counts(tp + a, fp - (k - a), fn - a, tn + (k - a))
        if score > best_score:
            best_score, best_a = score, a
    mask = np.zeros(len(y), dtype=bool)
    mask[fn_idx[:best_a]] = True
    mask[fp_idx[:k - best_a]] = True
    return mask, best_score


def router_features(p: np.ndarray, thr: float, size: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.column_stack([np.log(q / (1 - q)), np.abs(p - thr), size])


def fit_router(x: np.ndarray, t: np.ndarray, cfg: dict[str, Any],
               seed: int) -> tuple[Any, float | None, np.ndarray | None, bool]:
    """Logistic router fit on validation rows, C chosen by inner CV.

    Returns the model (None for a single class), C, validation priorities and an in-sample flag."""
    if t.min() == t.max():
        return None, None, None, False

    def make(c: float) -> Any:
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight=cfg["class_weight"], max_iter=int(cfg["max_iter"])))

    grid = [float(c) for c in cfg["C_grid"]]
    folds = int(cfg["inner_folds"])
    chosen = float(cfg["C_default"])
    enough = min(int(t.sum()), int(len(t) - t.sum())) >= folds
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    if len(grid) > 1 and enough:
        best = -1.0
        for c in grid:
            aucs = [roc_auc_score(t[te], make(c).fit(x[tr], t[tr]).predict_proba(x[te])[:, 1])
                    for tr, te in splitter.split(x, t)]
            if float(np.mean(aucs)) > best:
                best, chosen = float(np.mean(aucs)), c
    model = make(chosen).fit(x, t)
    if enough:
        val_priority = cross_val_predict(make(chosen), x, t, cv=splitter, method="predict_proba")[:, 1]
    else:
        val_priority = model.predict_proba(x)[:, 1]
    return model, chosen, val_priority, not enough


def seed_stats(rows: list[dict[str, Any]], field: str) -> tuple[float, float]:
    """Mean over the folds of each seed, then mean and population std over the seeds."""
    per_seed: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        per_seed[int(r["seed"])].append(float(r[field]))
    means = [float(np.mean(v)) for v in per_seed.values()]
    return float(np.mean(means)), (statistics.pstdev(means) if len(means) > 1 else 0.0)


def aggregate_folds(rows: list[dict[str, Any]], smells: list[str], mean_fields: tuple[str, ...],
                    sum_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    """One row per (seed, fold): mean of `mean_fields` and sum of `sum_fields` over the rows of `smells`,
    plus calls_per_1000_level (sum over the smells of 1000 * n_routed / n_level_test)."""
    by_fold: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["smell"] in smells:
            by_fold[(r["seed"], r["fold"])].append(r)
    out = []
    for (seed, fold), rs in sorted(by_fold.items()):
        if len(rs) != len(smells):
            raise SystemExit(f"seed {seed} fold {fold}: {len(rs)} rows for smells {smells}")
        agg: dict[str, Any] = {"seed": seed, "fold": fold}
        agg.update({k: float(np.mean([r[k] for r in rs])) for k in mean_fields})
        agg.update({k: sum(r[k] for r in rs) for k in sum_fields})
        if "n_routed" in rs[0]:
            agg["calls_per_1000_level"] = sum(1000.0 * r["n_routed"] / r["n_level_test"] for r in rs)
            agg["n_level_test"] = rs[0]["n_level_test"]
        out.append(agg)
    return out


def scopes(names: list[str], smell_level: dict[str, str]) -> dict[str, tuple[list[str], str]]:
    """Rows of the summaries: each smell, all_<level> (the smells of one level) and all (every smell)."""
    out = {s: ([s], smell_level[s]) for s in names}
    for lev in dict.fromkeys(smell_level[s] for s in names):
        out[f"all_{lev}"] = ([s for s in names if smell_level[s] == lev], lev)
    out["all"] = (list(names), "both")
    return out


def plot(summary: list[dict[str, Any]], references: list[dict[str, Any]], cfg: dict[str, Any], names: list[str],
         smell_level: dict[str, str], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_cfg = cfg["figure"]
    plt.rcParams.update({"font.size": float(fig_cfg["font_size"]), "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight"})
    panel_rows = [(c, rs) for c in cfg["cheap_models"] for rs in cfg["routed_scores"]]
    budgets = [float(b) for b in cfg["budgets"]]
    width, height = (float(v) for v in fig_cfg["panel_size"])
    fig, axes = plt.subplots(len(panel_rows), len(names), figsize=(width * len(names), height * len(panel_rows)),
                             squeeze=False)
    table = {(r["cheap"], r["routed_score"], r["policy"], float(r["budget"]), r["smell"]): r for r in summary}
    ref = {(r["reference"], r["cheap"], r["smell"]): r["mcc_mean"] for r in references}
    handles: dict[str, Any] = {}
    for row, (cname, rs) in enumerate(panel_rows):
        for col, smell in enumerate(names):
            ax = axes[row][col]
            for pname, pcfg in cfg["policies"].items():
                ys = [table[(cname, rs, pname, b, smell)]["mcc_mean"] for b in budgets]
                (line,) = ax.plot(budgets, ys, color=fig_cfg["colors"][pname], marker=fig_cfg["markers"][pname],
                                  markersize=3.5, linewidth=1.2,
                                  linestyle="--" if pname in fig_cfg["dashed"] else "-",
                                  label=pcfg["label"])
                handles.setdefault(pname, line)
            handles.setdefault("llm_alone", ax.axhline(
                ref[("llm_alone", "", smell)], color="grey", linestyle=":", linewidth=1.0,
                label=f"{cfg['llm']['label']} alone"))
            handles.setdefault("always_on_mean", ax.axhline(
                ref[("always_on_mean", cname, smell)], color="black", linestyle="-.", linewidth=0.9,
                label="always-on mean of cheap and LLM scores"))
            ax.set_xscale("symlog", linthresh=float(fig_cfg["linthresh"]), linscale=float(fig_cfg["linscale"]))
            ax.set_xlim(*(float(v) for v in fig_cfg["xlim"]))
            ax.set_xticks(budgets)
            ax.set_xticklabels([f"{100 * b:g}" for b in budgets])
            ax.minorticks_off()
            if row == 0:
                ax.set_title(f"{smell.replace('_', ' ')} ({smell_level[smell]} level)")
            if row == len(panel_rows) - 1:
                ax.set_xlabel("% of the level routed to the LLM")
            if col == 0:
                ax.set_ylabel(f"{cfg['cheap_models'][cname]['label']}\nrouted: {fig_cfg['routed_labels'][rs]}"
                              "\ntest MCC (level only)")
    fig.legend(list(handles.values()), [h.get_label() for h in handles.values()], loc="lower center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in fig_cfg["formats"]:
        fig.savefig(out_dir / f"{fig_cfg['name']}.{ext}", dpi=int(fig_cfg["dpi"]))
    plt.close(fig)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args, extra = parser.parse_known_args()
    cfg = load_config(args.config, parse_cli_overrides(extra))["routing"]
    out_dir = resolve(cfg["out_dir"])
    steps = int(cfg["threshold_steps"])
    budgets = sorted(float(b) for b in cfg["budgets"])
    routed_scores = [str(c) for c in cfg["routed_scores"]]
    policies = cfg["policies"]
    seed0 = int(cfg["seed"])
    if not budgets or any(not 0.0 <= b <= 1.0 for b in budgets):
        raise SystemExit(f"budgets must be shares in [0, 1], got {budgets}")
    if not routed_scores or any(c not in ROUTED_SCORES for c in routed_scores):
        raise SystemExit(f"routed_scores must be a non-empty subset of {list(ROUTED_SCORES)}, got {routed_scores}")
    # references, deltas and seed means are keyed by (seed, fold): two strategies would collide
    if not cfg.get("strategies") or len(cfg["strategies"]) != 1:
        raise SystemExit(f"routing.strategies must name exactly one split strategy, got {cfg.get('strategies')}")
    contrast_cfg = cfg["contrasts"]
    if any(a not in REFERENCES for a in contrast_cfg["against"]):
        raise SystemExit(f"contrasts.against must be in {list(REFERENCES)}, got {contrast_cfg['against']}")

    folds = load_folds(resolve(cfg["folds"]))
    label_order = folds_label_order(folds, cfg.get("label_order"))
    names = [s.removeprefix("is_") for s in label_order]
    smell_level = {s.removeprefix("is_"): str(cfg["smell_levels"][s]) for s in label_order}
    levels = load_levels(cfg["level"])
    size = load_size(cfg["learned"]["size_features"])
    scope = scopes(names, smell_level)
    ratio = ratio_from_folds(resolve(cfg["folds"]), cfg.get("strategies"))

    costs = json.loads(resolve(cfg["costs"]).read_text())
    llm_ms_per_call = float(costs[cfg["llm"]["cost_key"]]["inference_ms_per_sample"]) / float(
        cfg["llm"]["prompts_per_sample"])
    cheap_ms = {c: float(costs[spec["cost_key"]]["inference_ms_per_sample"]) for c, spec in cfg["cheap_models"].items()}

    llm = load_predictions(cfg["llm"])
    cheap = {c: load_predictions(spec) for c, spec in cfg["cheap_models"].items()}
    fold_list = list(iter_folds(folds, strategies=cfg.get("strategies")))
    print(f"[routing] {len(fold_list)} folds, cheap models {list(cheap)}, LLM {cfg['llm']['label']}, "
          f"{llm_ms_per_call:.1f} ms per LLM call, routed scores {routed_scores}, test/train ratio {ratio:.4f}")

    rows: list[dict[str, Any]] = []
    ref_rows: list[dict[str, Any]] = []  # reference, cheap, smell, seed, fold, mcc, f1
    overlap: dict[tuple[str, str], Counter] = defaultdict(Counter)
    kappas: dict[tuple[str, str], list[float]] = defaultdict(list)
    router_info: dict[tuple[str, str, str, str], Counter] = defaultdict(Counter)
    running_max_changes: Counter = Counter()

    for fold in fold_list:
        key = (fold.strategy, fold.seed, fold.fold)
        val_idx, test_idx = sorted(int(i) for i in fold.val), sorted(int(i) for i in fold.test)
        y_val, l_val = stack(llm[key]["val"], val_idx)
        y_test, l_test = stack(llm[key]["test"], test_idx)
        lev_val = np.array([levels[i] for i in val_idx])
        lev_test = np.array([levels[i] for i in test_idx])
        size_val = np.stack([size[i] for i in val_idx])
        size_test = np.stack([size[i] for i in test_idx])
        cheap_parts = {}
        for cname, cpred in cheap.items():
            cy_val, c_val = stack(cpred[key]["val"], val_idx)
            cy_test, c_test = stack(cpred[key]["test"], test_idx)
            if not (np.array_equal(cy_val, y_val) and np.array_equal(cy_test, y_test)):
                raise SystemExit(f"{cname} {key}: labels differ from the LLM prediction file")
            cheap_parts[cname] = (c_val, c_test)

        def add_ref(name: str, cname: str, smell: str, yt: np.ndarray, pred: np.ndarray) -> None:
            ref_rows.append({"reference": name, "cheap": cname, "smell": smell, "seed": fold.seed,
                             "fold": fold.fold, "mcc": mcc(yt, pred), "f1": f1(yt, pred)})

        for j, smell in enumerate(names):
            # eligible samples only: from here on every array holds the samples of the smell's level
            mv, mt = lev_val == smell_level[smell], lev_test == smell_level[smell]
            if not mv.any() or not mt.any():
                raise SystemExit(f"{key} {smell}: no validation or test sample of level {smell_level[smell]}")
            yv, yt = y_val[mv, j], y_test[mt, j]
            lv, lt = l_val[mv, j], l_test[mt, j]
            n_v, n_t = len(yv), len(yt)
            thr_l = threshold(yv, lv, steps)
            llm_v, llm_t = (lv >= thr_l).astype(np.int64), (lt >= thr_l).astype(np.int64)
            add_ref("llm_alone", "", smell, yt, llm_t)
            x_size_v, x_size_t = size_val[mv], size_test[mt]

            for ci, (cname, (c_val, c_test)) in enumerate(cheap_parts.items()):
                cv, ct = c_val[mv, j], c_test[mt, j]
                thr_c = threshold(yv, cv, steps)
                base_v, base_t = (cv >= thr_c).astype(np.int64), (ct >= thr_c).astype(np.int64)
                err_v, err_t = base_v != yv, base_t != yt
                dist_v, dist_t = np.abs(cv - thr_c), np.abs(ct - thr_c)
                add_ref("cheap_alone", cname, smell, yt, base_t)
                mean_v, mean_t = (cv + lv) / 2, (ct + lt) / 2
                thr_m = threshold(yv, mean_v, steps)
                routed = {"llm": (llm_v, llm_t),
                          "mean": ((mean_v >= thr_m).astype(np.int64), (mean_t >= thr_m).astype(np.int64))}
                add_ref("always_on_mean", cname, smell, yt, routed["mean"][1])

                le = llm_t != yt
                cnt = overlap[(cname, smell)]
                cnt["n_level_test"] += n_t
                cnt["cheap_errors"] += int(err_t.sum())
                cnt["llm_errors"] += int(le.sum())
                cnt["both_wrong"] += int((err_t & le).sum())
                cnt["cheap_wrong_llm_right"] += int((err_t & ~le).sum())
                cnt["cheap_right_llm_wrong"] += int((~err_t & le).sum())
                if err_t.any() or le.any():
                    kappas[(cname, smell)].append(float(cohen_kappa_score(err_t.astype(int), le.astype(int))))

                rng = np.random.default_rng([seed0, fold.seed, fold.fold, j, ci])
                tb_v, tb_t = rng.random(n_v), rng.random(n_t)
                # routing priorities, shared by the routed scores unless the router target depends on them
                draw_cache: dict[tuple[str, str], tuple[list[tuple[np.ndarray, np.ndarray]], float | None, bool]] = {}

                def draws_for(pname: str, pcfg: dict[str, Any],
                              rs: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], float | None, bool]:
                    kind = str(pcfg["kind"])
                    target_name = str(pcfg.get("target", ""))
                    cache_key = (pname, rs if kind == "learned" and target_name == "routed_fixes" else "")
                    if cache_key in draw_cache:
                        return draw_cache[cache_key]
                    router_c: float | None = None
                    fallback = False
                    if kind == "random":
                        draws = [(rng.random(n_v), rng.random(n_t)) for _ in range(int(pcfg["repeats"]))]
                    elif kind == "uncertainty":
                        draws = [(-dist_v, -dist_t)]
                    elif kind == "learned":
                        if target_name == "cheap_error":
                            target = err_v
                        elif target_name == "routed_fixes":
                            target = err_v & (routed[rs][0] == yv)
                        else:
                            raise SystemExit(f"unknown router target {target_name}")
                        info = router_info[(cname, cache_key[1] or "any", pname, smell)]
                        model, router_c, val_priority, in_sample = fit_router(
                            router_features(cv, thr_c, x_size_v), target.astype(np.int64), cfg["learned"],
                            seed0 + fold.seed)
                        if model is None:
                            fallback = True
                            draws = [(-dist_v, -dist_t)]
                            info["fallback"] += 1
                        else:
                            draws = [(val_priority,
                                      model.predict_proba(router_features(ct, thr_c, x_size_t))[:, 1])]
                            info[f"C={router_c:g}"] += 1
                            if in_sample:
                                info["in_sample_val"] += 1
                    else:
                        raise SystemExit(f"unknown policy kind {kind}")
                    draw_cache[cache_key] = (draws, router_c, fallback)
                    return draw_cache[cache_key]

                for rs in routed_scores:
                    rv, rt = routed[rs]
                    for pname, pcfg in policies.items():
                        kind = str(pcfg["kind"])
                        results: list[tuple[float, dict[str, list[float]], float | None, bool, bool]] = []
                        if kind == "oracle":
                            # upper bound with the labels of each part, monotone in the budget per fold
                            best_v, best_t = -2.0, (-2.0, np.zeros(n_t, dtype=bool))
                            for b in budgets:
                                val_score = oracle_route(yv, base_v, rv, budget_size(n_v, b), dist_v)[1]
                                mask_t, score_t = oracle_route(yt, base_t, rt, budget_size(n_t, b), dist_t)
                                changed = score_t < best_t[0] - 1e-12
                                if changed:
                                    running_max_changes[(cname, rs, smell)] += 1
                                else:
                                    best_t = (score_t, mask_t)
                                best_v = max(best_v, val_score)
                                results.append((best_v, {"masks": [best_t[1]]}, None, False, changed))
                        else:
                            draws, router_c, fallback = draws_for(pname, pcfg, rs)
                            for b in budgets:
                                val_scores, masks_t = [], []
                                for pv, pt in draws:
                                    m_v = route(pv, b, tb_v)
                                    h = base_v.copy()
                                    h[m_v] = rv[m_v]
                                    val_scores.append(mcc(yv, h))
                                    masks_t.append(route(pt, b, tb_t))
                                results.append((float(np.mean(val_scores)), {"masks": masks_t}, router_c,
                                                fallback, False))
                        for b, (val_score, res, router_c, fallback, changed) in zip(budgets, results):
                            stats: dict[str, list[float]] = defaultdict(list)
                            for m_t in res["masks"]:
                                h = base_t.copy()
                                h[m_t] = rt[m_t]
                                stats["mcc"].append(mcc(yt, h))
                                stats["f1"].append(f1(yt, h))
                                stats["n_routed"].append(int(m_t.sum()))
                                stats["routed_cheap_errors"].append(int((m_t & err_t).sum()))
                                stats["fixed"].append(int((m_t & err_t & (h == yt)).sum()))
                                stats["broken"].append(int((m_t & ~err_t & (h != yt)).sum()))
                            rows.append({
                                "cheap": cname, "routed_score": rs, "policy": pname, "budget": b, "smell": smell,
                                "level": smell_level[smell], "strategy": fold.strategy, "seed": fold.seed,
                                "fold": fold.fold, "val_mcc": round(val_score, 6),
                                **{k: round(float(np.mean(v)), 6) for k, v in stats.items()},
                                "n_level_val": n_v, "n_level_test": n_t, "n_budget": budget_size(n_t, b),
                                "cheap_errors_level": int(err_t.sum()), "router_C": router_c,
                                "router_fallback": fallback, "oracle_running_max": changed,
                            })
        print(f"[routing] {key} done")

    # paired differences with the references on the same fold and samples
    ref_fold = {(r["reference"], r["cheap"], r["smell"], r["seed"], r["fold"]): r["mcc"] for r in ref_rows}
    for r in rows:
        for name, field in DELTA_FIELD.items():
            cname = "" if name == "llm_alone" else r["cheap"]
            r[field] = round(r["mcc"] - ref_fold[(name, cname, r["smell"], r["seed"], r["fold"])], 6)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "fold_results.csv", FOLD_FIELDS, rows)

    def cost_columns(calls_per_1000: float | None, cname: str | None) -> dict[str, Any]:
        if calls_per_1000 is None:
            return {"llm_calls_per_1000_level": None, "llm_seconds_per_1000_level": None,
                    "cheap_seconds_per_1000_level": None, "total_seconds_per_1000_level": None}
        llm_s = calls_per_1000 * llm_ms_per_call / 1000.0
        cheap_s = cheap_ms[cname] if cname else 0.0
        return {"llm_calls_per_1000_level": round(calls_per_1000, 1),
                "llm_seconds_per_1000_level": round(llm_s, 1),
                "cheap_seconds_per_1000_level": round(cheap_s, 2),
                "total_seconds_per_1000_level": round(llm_s + cheap_s, 1)}

    # summary per (cheap, routed score, policy, budget, scope)
    mean_fields = ("mcc", "f1", "val_mcc", *DELTA_FIELD.values())
    groups: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["cheap"], r["routed_score"], r["policy"], r["budget"])].append(r)
    summary: list[dict[str, Any]] = []
    fold_scope: dict[tuple[str, str, str, float, str], list[dict[str, Any]]] = {}
    for (cname, rs, pname, b), group in groups.items():
        for sname, (smells, lev) in scope.items():
            sel = aggregate_folds(group, smells, mean_fields, COUNT_FIELDS)
            fold_scope[(cname, rs, pname, b, sname)] = sel
            calls = float(np.mean([r["calls_per_1000_level"] for r in sel])) if lev != "both" else None
            summary.append({
                "cheap": cname, "routed_score": rs, "policy": pname, "budget": b, "smell": sname, "level": lev,
                "mcc_mean": round(seed_stats(sel, "mcc")[0], 4), "mcc_std_seeds": round(seed_stats(sel, "mcc")[1], 4),
                **{f"{k}_{stat}": round(v, 4) for k in DELTA_FIELD.values()
                   for stat, v in zip(("mean", "std_seeds"), seed_stats(sel, k))},
                "f1_mean": round(seed_stats(sel, "f1")[0], 4), "val_mcc_mean": round(seed_stats(sel, "val_mcc")[0], 4),
                "llm_calls_per_fold": round(float(np.mean([r["n_routed"] for r in sel])), 2),
                **cost_columns(calls, cname),
                "n_level_test_per_fold": (round(float(np.mean([r["n_level_test"] for r in sel])), 2)
                                          if lev != "both" else None),
                **{f"{k}_per_fold": round(float(np.mean([r[k] for r in sel])), 2)
                   for k in COUNT_FIELDS if k != "n_routed"},
            })
    summary.sort(key=lambda r: (list(cheap).index(r["cheap"]), routed_scores.index(r["routed_score"]),
                                list(policies).index(r["policy"]), list(scope).index(r["smell"]), r["budget"]))
    write_csv(out_dir / "summary.csv", SUMMARY_FIELDS, summary)

    # references: cheap alone, LLM alone, always-on mean
    references: list[dict[str, Any]] = []
    for name in REFERENCES:
        for cname in ([""] if name == "llm_alone" else list(cheap)):
            sub = [r for r in ref_rows if r["reference"] == name and r["cheap"] == cname]
            for sname, (smells, lev) in scope.items():
                sel = aggregate_folds(sub, smells, ("mcc", "f1"), ())
                calls = None if lev == "both" else (0.0 if name == "cheap_alone" else 1000.0 * len(smells))
                references.append({
                    "reference": name, "cheap": cname, "smell": sname, "level": lev,
                    "mcc_mean": round(seed_stats(sel, "mcc")[0], 4),
                    "mcc_std_seeds": round(seed_stats(sel, "mcc")[1], 4),
                    "f1_mean": round(seed_stats(sel, "f1")[0], 4),
                    **cost_columns(calls, cname or None),
                })
    write_csv(out_dir / "references.csv", REFERENCE_FIELDS, references)

    # paired contrasts with the corrected resampled t-test
    contrasts: list[dict[str, Any]] = []
    for cname in cheap:
        for rs in routed_scores:
            for pname in contrast_cfg["policies"]:
                for b in (float(v) for v in contrast_cfg["budgets"]):
                    if (cname, rs, pname, b, names[0]) not in fold_scope:
                        raise SystemExit(f"contrast on policy {pname} budget {b}: not among the routed results")
                    for against in contrast_cfg["against"]:
                        field = DELTA_FIELD[against]
                        for sname, (_, lev) in scope.items():
                            sel = fold_scope[(cname, rs, pname, b, sname)]
                            diffs = np.array([r[field] for r in sel])
                            mean_d, std_d = seed_stats(sel, field)
                            t, p, half = corrected_ttest(diffs, ratio)
                            contrasts.append({
                                "cheap": cname, "routed_score": rs, "policy": pname, "budget": b, "against": against,
                                "smell": sname, "level": lev, "n_pairs": len(diffs), "mean_diff": round(mean_d, 4),
                                "std_seeds": round(std_d, 4), "ci95_low": round(float(np.mean(diffs)) - half, 4),
                                "ci95_high": round(float(np.mean(diffs)) + half, 4), "corrected_t": round(t, 3),
                                "corrected_p": round(p, 4), "wins": int((diffs > 0).sum()),
                                "losses": int((diffs < 0).sum()),
                                "llm_calls_per_1000_level": (
                                    round(float(np.mean([r["calls_per_1000_level"] for r in sel])), 1)
                                    if lev != "both" else None),
                            })
    write_csv(out_dir / "contrasts.csv", CONTRAST_FIELDS, contrasts)

    overlap_out = {}
    for (cname, smell), cnt in overlap.items():
        overlap_out[f"{cname}|{smell}"] = {
            **cnt,
            "share_of_cheap_errors_llm_right": round(cnt["cheap_wrong_llm_right"] / max(cnt["cheap_errors"], 1), 4),
            "error_kappa_mean_over_folds": round(float(np.mean(kappas[(cname, smell)])), 4),
        }
    (out_dir / "summary.json").write_text(json.dumps({
        "folds": len(fold_list), "strategies": cfg.get("strategies"), "budgets": budgets,
        "routed_scores": routed_scores, "evaluation": "eligible samples only (level of the smell), thresholds on "
        "the eligible validation samples", "test_train_ratio": ratio, "llm": cfg["llm"]["label"],
        "llm_ms_per_call": llm_ms_per_call, "cheap_ms_per_sample": cheap_ms,
        "oracle_running_max_changes": {f"{c}|{rs}|{s}": n for (c, rs, s), n in running_max_changes.items()},
        "references": references, "contrasts": contrasts, "error_overlap_level_test_pooled": overlap_out,
        "router_choices": {"|".join(k): dict(v) for k, v in router_info.items()},
        "summary": summary,
    }, indent=1))
    plot(summary, references, cfg, names, smell_level, out_dir)

    # console report
    table = {(r["cheap"], r["routed_score"], r["policy"], r["budget"], r["smell"]): r for r in summary}
    ref = {(r["reference"], r["cheap"], r["smell"]): r for r in references}
    head = " ".join(f"{100 * b:>6g}%" for b in budgets)
    for cname in cheap:
        for rs in routed_scores:
            print(f"\n[routing] cheap {cname}, routed score {rs}: eligible test MCC (mean over folds then seeds)")
            for sname in scope:
                print(f"  {sname:<14} {head}   cheap {ref[('cheap_alone', cname, sname)]['mcc_mean']:.3f}"
                      f"  LLM {ref[('llm_alone', '', sname)]['mcc_mean']:.3f}"
                      f"  always-on mean {ref[('always_on_mean', cname, sname)]['mcc_mean']:.3f}")
                for pname in policies:
                    vals = " ".join(f"{table[(cname, rs, pname, b, sname)]['mcc_mean']:7.3f}" for b in budgets)
                    print(f"    {pname:<14}{vals}")
    print(f"\n[routing] contrasts (corrected resampled t-test, test/train ratio {ratio:.4f})")
    for c in contrasts:
        print(f"  {c['cheap']:<11} {c['routed_score']:<4} {c['policy']:<13} {100 * c['budget']:g}% - "
              f"{c['against']:<14} {c['smell']:<12} {c['mean_diff']:+.4f} sd {c['std_seeds']:.4f} "
              f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}] p={c['corrected_p']:.3g} "
              f"wins {c['wins']}/{c['n_pairs']}")
    print("\n[routing] error overlap on the level test samples (pooled over folds)")
    for k, v in overlap_out.items():
        print(f"  {k:<30} cheap errors {v['cheap_errors']:>5}  LLM right on them {v['cheap_wrong_llm_right']:>5} "
              f"({v['share_of_cheap_errors_llm_right']:.3f})  LLM wrong where cheap right "
              f"{v['cheap_right_llm_wrong']:>5}  error kappa {v['error_kappa_mean_over_folds']:.3f}")
    print(f"[routing] oracle running max changed {sum(running_max_changes.values())} fold values")
    print(f"[routing] wrote {out_dir}")


if __name__ == "__main__":
    main()
