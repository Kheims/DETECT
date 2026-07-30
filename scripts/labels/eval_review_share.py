"""Compare models trained on the consensus label and on the share of positive reviews.

Metrics: consensus MCC, review-level scores, Manhattan distance and Brier score
on the shares, with repository bootstrap intervals on the differences.

Usage:
    python scripts/labels/eval_review_share.py \\
        --pipeline 'bilstm_attention=artifacts/reports/review_share_bilstm_*/records.json' \\
        --pipeline 'gcn=artifacts/reports/review_share_gcn_*/records.json' \\
        --finetune codebert=artifacts/code_finetune/review_share_codebert \\
        --tabular 'xgboost_designite=artifacts/tabular_cv/review_share/designite__xgb__{arm}' \\
        --tabular 'logreg_designite=artifacts/tabular_cv/review_share/designite__logreg__{arm}'
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlcq_graphs.constants import LABEL_ORDER  # noqa: E402
from mlcq_graphs.evaluation.cv import read_jsonl, tune_thresholds  # noqa: E402

ARMS = ("consensus", "review_share")
EPS = 1e-6


def resolve(path: str | Path) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------- loading


def item_arrays(rows: list[dict[str, Any]], steps: int) -> dict[str, np.ndarray]:
    """Test json_index, probabilities and decisions of one (seed, fold) item."""
    val = [r for r in rows if r["part"] == "val"]
    test = [r for r in rows if r["part"] == "test"]
    if not val or not test:
        raise SystemExit("an item has no validation or no test predictions")
    yv = np.array([r["y"] for r in val], dtype=np.int64)
    pv = np.array([r["prob"] for r in val], dtype=np.float64)
    thr = tune_thresholds(yv, pv, steps)
    pt = np.array([r["prob"] for r in test], dtype=np.float64)
    return {"index": np.array([r["json_index"] for r in test], dtype=np.int64), "prob": pt,
            "decision": (pt >= thr).astype(np.int64), "threshold": thr}


def load_pipeline(pattern: str, steps: int) -> dict[str, dict[int, list[dict[str, np.ndarray]]]]:
    """{arm: {seed: [item, ...]}} from the pipeline ablation records."""
    out: dict[str, dict[int, list[dict[str, np.ndarray]]]] = {a: defaultdict(list) for a in ARMS}
    seen = set()
    for rec_path in sorted(glob.glob(str(resolve(pattern)))):
        for rec in json.loads(Path(rec_path).read_text()):
            run_dir = Path(rec["run_dir"])
            if run_dir in seen:
                continue
            seen.add(run_dir)
            cfg = json.loads((run_dir / "config.json").read_text())
            arm = str(cfg.get("train_target", "consensus"))
            rows = read_jsonl(run_dir / "predictions.jsonl")
            out[arm][int(rec["seed"])].append(item_arrays(rows, steps))
    return out


def load_finetune(root: Path, steps: int) -> dict[str, dict[int, list[dict[str, np.ndarray]]]]:
    out: dict[str, dict[int, list[dict[str, np.ndarray]]]] = {a: defaultdict(list) for a in ARMS}
    for rec_path in sorted(root.glob("*/kfold_s*_f*/record.json")):
        rec = json.loads(rec_path.read_text())
        arm = str(rec.get("train_target", "consensus"))
        rows = read_jsonl(rec_path.parent / "predictions.jsonl")
        out[arm][int(rec["seed"])].append(item_arrays(rows, steps))
    return out


def load_tabular(pattern: str, steps: int) -> dict[str, dict[int, list[dict[str, np.ndarray]]]]:
    """One output directory per arm ({arm} in the pattern), possibly split in shards."""
    out: dict[str, dict[int, list[dict[str, np.ndarray]]]] = {a: defaultdict(list) for a in ARMS}
    for arm in ARMS:
        by_item: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for d in sorted(glob.glob(str(resolve(pattern.format(arm=arm))) + "*")):
            for row in read_jsonl(Path(d) / "predictions.jsonl"):
                by_item[(int(row["seed"]), int(row["fold"]))].append(row)
        for (seed, _), rows in sorted(by_item.items()):
            out[arm][seed].append(item_arrays(rows, steps))
    return out


def pool(items: list[dict[str, np.ndarray]], n: int) -> dict[str, np.ndarray]:
    """Pool the test parts of the folds of one seed (every snippet must appear once)."""
    prob = np.full((n, len(LABEL_ORDER)), np.nan)
    dec = np.full((n, len(LABEL_ORDER)), -1, dtype=np.int64)
    for it in items:
        if not np.isnan(prob[it["index"], 0]).all():
            raise SystemExit("a snippet is in the test part of two folds of the same seed")
        prob[it["index"]] = it["prob"]
        dec[it["index"]] = it["decision"]
    covered = ~np.isnan(prob[:, 0])
    return {"prob": prob, "decision": dec, "covered": covered}


# ---------------------------------------------------------------- metrics


def weighted_counts(dec: np.ndarray, pos: np.ndarray, neg: np.ndarray, w: np.ndarray) -> tuple[float, ...]:
    """TP, FP, FN, TN where a snippet contributes `pos` positive and `neg` negative examples."""
    tp = float((w * dec * pos).sum())
    fp = float((w * dec * neg).sum())
    fn = float((w * (1 - dec) * pos).sum())
    tn = float((w * (1 - dec) * neg).sum())
    return tp, fp, fn, tn


def mcc_f1(tp: float, fp: float, fn: float, tn: float) -> tuple[float, float, float, float]:
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    return float(mcc), float(f1), float(prec), float(rec)


def metrics_one(prob: np.ndarray, dec: np.ndarray, y: np.ndarray, share: np.ndarray, n_rev: np.ndarray,
                n_pos: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """Metrics of one smell for one pooled seed, w are snippet weights (bootstrap counts)."""
    m: dict[str, float] = {}
    tp, fp, fn, tn = weighted_counts(dec, y, 1 - y, w)
    m["consensus_mcc"], m["consensus_f1"], _, _ = mcc_f1(tp, fp, fn, tn)
    reviewed = n_rev > 0
    wr = w * reviewed
    tp, fp, fn, tn = weighted_counts(dec, n_pos, n_rev - n_pos, wr)
    m["reviews_mcc"], m["reviews_f1"], m["reviews_precision"], m["reviews_recall"] = mcc_f1(tp, fp, fn, tn)
    multi = n_rev >= 2
    wm = w * multi
    if wm.sum() > 0:
        m["manhattan_multi"] = float((wm * 2 * np.abs(prob - share)).sum() / wm.sum())
        p = np.clip(prob, EPS, 1 - EPS)
        m["brier_multi"] = float((wm * (p - share) ** 2).sum() / wm.sum())
        m["soft_ce_multi"] = float((wm * -(share * np.log(p) + (1 - share) * np.log(1 - p))).sum() / wm.sum())
    return m


def reference_reviews(y: np.ndarray, n_rev: np.ndarray, n_pos: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """The consensus label used as the decision, scored against every review."""
    tp, fp, fn, tn = weighted_counts(y, n_pos, n_rev - n_pos, w * (n_rev > 0))
    mcc, f1, prec, rec = mcc_f1(tp, fp, fn, tn)
    return {"reviews_mcc": mcc, "reviews_f1": f1, "reviews_precision": prec, "reviews_recall": rec}


def calibration(prob: np.ndarray, share: np.ndarray, bins: np.ndarray) -> list[dict[str, float]]:
    rows = []
    which = np.clip(np.digitize(prob, bins[1:-1]), 0, len(bins) - 2)
    for b in range(len(bins) - 1):
        sel = which == b
        if sel.sum() == 0:
            continue
        rows.append({"bin_low": float(bins[b]), "bin_high": float(bins[b + 1]), "n": int(sel.sum()),
                     "mean_prob": float(prob[sel].mean()), "mean_share": float(share[sel].mean())})
    return rows


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shares", default="data/derived/mlcq_review_shares.json")
    ap.add_argument("--labels-json", default="data/derived/MLCQCodeSmellSamples.normalized.rule_default.json")
    ap.add_argument("--pipeline", action="append", default=[], help="name=glob of pipeline records.json")
    ap.add_argument("--finetune", action="append", default=[], help="name=finetune run directory")
    ap.add_argument("--tabular", action="append", default=[], help="name=tabular output dir with {arm}")
    ap.add_argument("--threshold-steps", type=int, default=101)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="artifacts/analysis/review_share")
    args = ap.parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shares = json.loads(resolve(args.shares).read_text())
    if list(shares["meta"]["label_order"]) != LABEL_ORDER:
        raise SystemExit("label order of the shares file differs from LABEL_ORDER")
    S = np.asarray(shares["share"], dtype=np.float64)
    NR = np.asarray(shares["n_reviewers"], dtype=np.float64)
    NP = np.asarray(shares["n_positive"], dtype=np.float64)
    entries = json.loads(resolve(args.labels_json).read_text())
    Y = np.array([[int(bool(v)) for v in e["y"][:4]] for e in entries], dtype=np.int64)
    repos = np.array([e["repo_url"] for e in entries])
    n = len(entries)

    models = {}
    for spec in args.pipeline:
        name, path = spec.split("=", 1)
        models[name] = load_pipeline(path, args.threshold_steps)
    for spec in args.finetune:
        name, path = spec.split("=", 1)
        models[name] = load_finetune(resolve(path), args.threshold_steps)
    for spec in args.tabular:
        name, path = spec.split("=", 1)
        models[name] = load_tabular(path, args.threshold_steps)
    if not models:
        raise SystemExit("give at least one --pipeline, --finetune or --tabular input")
    pooled: dict[str, dict[str, dict[int, dict[str, np.ndarray]]]] = {}
    for model, arms in models.items():
        pooled[model] = {}
        for arm in ARMS:
            pooled[model][arm] = {}
            for seed, items in sorted(arms[arm].items()):
                if len(items) != 5:
                    print(f"[eval] {model} {arm} seed {seed}: {len(items)} folds, skipped")
                    continue
                pooled[model][arm][seed] = pool(items, n)
        common = sorted(set(pooled[model]["consensus"]) & set(pooled[model]["review_share"]))
        print(f"[eval] {model}: seeds with both arms complete {common}")
        for arm in ARMS:
            pooled[model][arm] = {s: pooled[model][arm][s] for s in common}

    # snippets covered by the folds (the 4,361 with an AST), the same for every model
    covered = None
    for model in pooled:
        for arm in ARMS:
            for p in pooled[model][arm].values():
                covered = p["covered"] if covered is None else covered & p["covered"]
    if covered is None:
        raise SystemExit("no complete seed for any model")
    idx = np.flatnonzero(covered)
    repo_codes, repo_of = np.unique(repos[idx], return_inverse=True)
    print(f"[eval] {len(idx)} snippets, {len(repo_codes)} repositories")

    def seed_mean(model: str, arm: str, s: int, w: np.ndarray) -> dict[str, float]:
        vals: dict[str, list[float]] = defaultdict(list)
        for p in pooled[model][arm].values():
            m = metrics_one(p["prob"][idx, s], p["decision"][idx, s], Y[idx, s], S[idx, s], NR[idx, s],
                            NP[idx, s], w)
            for k, v in m.items():
                vals[k].append(v)
        return {k: float(np.mean(v)) for k, v in vals.items()}

    rng = np.random.default_rng(args.seed)
    draws = [np.bincount(rng.integers(0, len(repo_codes), len(repo_codes)), minlength=len(repo_codes))[repo_of]
             .astype(np.float64) for _ in range(args.bootstrap)]
    ones = np.ones(len(idx))

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"snippets": int(len(idx)), "repositories": int(len(repo_codes)),
                               "bootstrap": args.bootstrap, "models": {}}
    for s, smell in enumerate(LABEL_ORDER):
        multi = NR[idx, s] >= 2
        ref = reference_reviews(Y[idx, s], NR[idx, s], NP[idx, s], ones)
        rows.append({"model": "reference (consensus label as decision)", "smell": smell, "arm": "", **ref})
        summary.setdefault("reference", {})[smell] = {**ref, "pairs_reviewed": int((NR[idx, s] > 0).sum()),
                                                      "pairs_multi": int(multi.sum()),
                                                      "pairs_mixed": int((multi & (S[idx, s] > 0)
                                                                          & (S[idx, s] < 1)).sum())}
    for model in pooled:
        if not pooled[model]["consensus"]:
            continue
        summary["models"][model] = {"seeds": sorted(pooled[model]["consensus"]), "smells": {}}
        for s, smell in enumerate(LABEL_ORDER):
            point = {arm: seed_mean(model, arm, s, ones) for arm in ARMS}
            keys = sorted(point["consensus"])
            boot = {k: [] for k in keys}
            for w in draws:
                a = seed_mean(model, "review_share", s, w)
                b = seed_mean(model, "consensus", s, w)
                for k in keys:
                    boot[k].append(a[k] - b[k])
            diff = {k: {"diff": point["review_share"][k] - point["consensus"][k],
                        "ci95": [float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))]}
                    for k in keys}
            multi = NR[idx, s] >= 2
            extra = {}
            for arm in ARMS:
                rhos, cal_prob = [], []
                for p in pooled[model][arm].values():
                    rhos.append(float(spearmanr(p["prob"][idx[multi], s], S[idx[multi], s]).statistic))
                    cal_prob.append(p["prob"][idx[multi], s])
                extra[arm] = {
                    "spearman_multi": float(np.mean(rhos)),
                    "calibration_multi": calibration(np.mean(cal_prob, axis=0), S[idx[multi], s],
                                                     np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])),
                }
            summary["models"][model]["smells"][smell] = {"point": point, "difference": diff, "extra": extra}
            for arm in ARMS:
                rows.append({"model": model, "smell": smell, "arm": arm, **point[arm],
                             "spearman_multi": extra[arm]["spearman_multi"]})
            print(f"[eval] {model:16s} {smell:16s} " + " | ".join(
                f"{k} {point['consensus'][k]:.3f} -> {point['review_share'][k]:.3f} "
                f"[{diff[k]['ci95'][0]:+.3f}, {diff[k]['ci95'][1]:+.3f}]"
                for k in ("consensus_mcc", "manhattan_multi", "brier_multi") if k in diff))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("model", "smell", "arm"), k))
    with (out_dir / "metrics.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(f"[eval] wrote {out_dir / 'summary.json'} and {out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
