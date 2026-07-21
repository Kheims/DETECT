"""Reviewer and sampling-phase effects on the MLCQ labels.

Reads the Zenodo MLCQ release (one row per review). Covers sample linkage, phases,
reviewer heterogeneity, cross-check selection, model errors by reviewer and the
reviewer survey. Writes summary.json, CSV tables and figures.

Usage:
    python scripts/labels/analyze_reviewers.py --config config/experiments/mlcq_reviewers.yml
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import itertools
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize, special, stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.constants import LABEL_ORDER  # noqa: E402
from mlcq_graphs.evaluation.cv import read_jsonl, tune_thresholds  # noqa: E402

# Okabe and Ito (2008) colour-blind safe palette
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
SMELL_COLORS = [OI["blue"], OI["vermillion"], OI["green"], OI["purple"]]
SERIES_COLORS = [OI["blue"], OI["orange"], OI["green"], OI["purple"], OI["sky"], OI["vermillion"], OI["black"],
                 OI["yellow"]]
GREY = "#999999"
SUBSETS = ("all", "first", "later")


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


# ------------------------------------------------------------ output helpers

def tmp_path(path: Path) -> Path:
    """Per-process temporary name, so that concurrent writers never share one."""
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def jsonable(obj: Any) -> Any:
    """Convert NaN and infinities to null and NumPy scalars to Python numbers."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    return obj


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path(path)
    tmp.write_text(json.dumps(jsonable(obj), indent=1, ensure_ascii=False))
    tmp.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            fields.extend(k for k in row if k not in fields)
    tmp = tmp_path(path)
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt_cell(row.get(k)) for k in fields})
    tmp.replace(path)
    return rel(path)


def fmt_cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "" if not math.isfinite(float(v)) else round(float(v), 6)
    return v


class Figures:
    def __init__(self, directory: Path, prefix: str) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt = plt
        self.dir, self.prefix = directory, prefix
        self.written: list[str] = []
        directory.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({
            "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
            "legend.frameon": False, "figure.dpi": 100, "savefig.bbox": "tight",
        })

    def save(self, fig: Any, name: str) -> None:
        for ext in ("pdf", "png"):
            path = self.dir / f"{self.prefix}{name}.{ext}"
            tmp = tmp_path(path)
            fig.savefig(tmp, format=ext, dpi=200)
            tmp.replace(path)
            self.written.append(rel(path))
        self.plt.close(fig)


# ------------------------------------------------------------ statistics

def wilson(k: float, n: float, level: float) -> tuple[float, float]:
    """Wilson score interval (k may be a non-integer mean count)."""
    if n <= 0:
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(0.5 + level / 2))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / den
    # min and max only absorb rounding at p = 0 or 1
    return min(max(0.0, centre - half), p), max(min(1.0, centre + half), p)


def rate(k: float, n: float, level: float) -> dict[str, Any]:
    lo, hi = wilson(k, n, level)
    return {"n": n, "k": k, "rate": (k / n) if n > 0 else float("nan"), "ci_low": lo, "ci_high": hi}


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa of two binary raters on (..., units) arrays."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    po = (a == b).mean(axis=-1)
    pa, pb = a.mean(axis=-1), b.mean(axis=-1)
    pe = pa * pb + (1 - pa) * (1 - pb)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(pe < 1, (po - pe) / (1 - pe), np.nan)


def logistic_deviance(X: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
    """-2 log-likelihood of an unpenalised logistic regression (L-BFGS)."""
    def fun(beta: np.ndarray) -> tuple[float, np.ndarray]:
        z = X @ beta
        ll = y * z - np.logaddexp(0.0, z)
        grad = X.T @ (y - special.expit(z))
        return -float(ll.sum()), -grad
    res = optimize.minimize(fun, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                            options={"maxiter": 10000, "gtol": 1e-9})
    return 2.0 * float(res.fun), bool(res.success)


def conditional_logit_deviance(groups: np.ndarray, X: np.ndarray, y: np.ndarray,
                               max_sets: int = 10000) -> dict[str, Any]:
    """-2 log-likelihood of a conditional logistic regression with one intercept per group."""
    order = np.argsort(groups, kind="stable")
    g, Xs, ys = groups[order], X[order].astype(np.float64), y[order].astype(np.int64)
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], len(g)]
    obs, cand, cand_start = [], [], []
    const = 0.0
    for a, b in zip(starts, ends, strict=True):
        k, m = int(ys[a:b].sum()), b - a
        if k == 0 or k == m:
            continue
        if math.comb(m, k) > max_sets:
            raise SystemExit(f"conditional logit: group of {m} reviews has too many outcome sets")
        cand_start.append(len(cand))
        obs.append(Xs[a:b][ys[a:b] == 1].sum(axis=0))
        for subset in itertools.combinations(range(a, b), k):
            cand.append(Xs[list(subset)].sum(axis=0))
        const += math.log(math.comb(m, k))
    n_groups = len(obs)
    if n_groups == 0:
        return {"deviance": 0.0, "converged": True, "informative_groups": 0, "beta": np.zeros(X.shape[1])}
    O = np.asarray(obs).sum(axis=0)
    C = np.asarray(cand)
    cs = np.asarray(cand_start)
    gid = np.repeat(np.arange(n_groups), np.diff(np.r_[cs, len(C)]))

    def fun(beta: np.ndarray) -> tuple[float, np.ndarray]:
        sc = C @ beta
        mx = np.maximum.reduceat(sc, cs)
        ex = np.exp(sc - mx[gid])
        tot = np.add.reduceat(ex, cs)
        lse = mx + np.log(tot)
        w = ex / tot[gid]
        return -(float(O @ beta) - float(lse.sum())), -(O - (w[:, None] * C).sum(axis=0))

    if X.shape[1] == 0:
        return {"deviance": 2.0 * const, "converged": True, "informative_groups": n_groups, "beta": np.zeros(0)}
    res = optimize.minimize(fun, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                            options={"maxiter": 10000, "gtol": 1e-9})
    return {"deviance": 2.0 * float(res.fun), "converged": bool(res.success), "informative_groups": n_groups,
            "beta": res.x}


def rank_adjusted_reviewer_test(revs: list[dict[str, Any]], rank_cols: dict[str, np.ndarray],
                                min_n: int) -> dict[str, Any]:
    """Likelihood-ratio test of reviewer indicators in a conditional logistic regression per unit."""
    unit = np.array([rv["sample"] for rv in revs])
    y = np.array([rv["pos"] for rv in revs])
    k_unit = defaultdict(list)
    for u, v in zip(unit, y, strict=True):
        k_unit[int(u)].append(int(v))
    informative = np.array([0 < sum(k_unit[int(u)]) < len(k_unit[int(u)]) for u in unit])
    counts = Counter(rv["reviewer"] for rv, inf in zip(revs, informative, strict=True) if inf)
    keep = [r for r, c in counts.most_common() if c >= min_n]
    if len(keep) < 2:
        return {"reviewers": len(keep), "note": "fewer than two reviewers with enough informative reviews"}
    names = list(rank_cols)
    Xr = np.column_stack([rank_cols[n] for n in names]) if names else np.zeros((len(revs), 0))
    reviewer = np.array([rv["reviewer"] for rv in revs])
    cols = [(reviewer == r).astype(np.float64) for r in keep[1:]]
    pooled = ~np.isin(reviewer, keep)
    has_other = bool((pooled & informative).any())
    if has_other:
        cols.append(pooled.astype(np.float64))
    Xv = np.column_stack(cols)
    m0 = conditional_logit_deviance(unit, Xr, y)
    m1 = conditional_logit_deviance(unit, np.column_stack([Xr, Xv]), y)
    lr = max(m0["deviance"] - m1["deviance"], 0.0)
    dof = Xv.shape[1]
    return {
        "statistic": lr, "dof": dof, "p_value": float(stats.chi2.sf(lr, dof)),
        "reviewers": len(keep), "reference_reviewer": keep[0],
        "pooled_reviewers_indicator": has_other,
        "informative_units": m0["informative_groups"], "reviews_in_informative_units": int(informative.sum()),
        "converged": m0["converged"] and m1["converged"],
        "rank_log_odds_null_model": {n: float(b) for n, b in zip(names, m0["beta"], strict=True)},
    }


def within_unit_permutation(unit: np.ndarray, reviewer: np.ndarray, x: np.ndarray, n_perm: int,
                            rng: np.random.Generator) -> dict[str, Any]:
    """Permutation test of reviewer heterogeneity that keeps the reviewed units fixed."""
    order = np.argsort(unit, kind="stable")
    u, r, v = unit[order], reviewer[order], x[order].astype(np.float64)
    _, inv, counts = np.unique(u, return_inverse=True, return_counts=True)
    means = np.bincount(inv, weights=v) / counts
    size = int(r.max()) + 1
    n_r = np.bincount(r, minlength=size).astype(np.float64)
    expected = np.bincount(r, weights=means[inv], minlength=size)
    active = n_r > 0

    def statistic(values: np.ndarray) -> tuple[float, np.ndarray]:
        observed = np.bincount(r, weights=values, minlength=size)
        return float(((observed - expected)[active] ** 2 / n_r[active]).sum()), observed

    t_obs, o_obs = statistic(v)
    base = inv.astype(np.float64)
    exceed = 0
    null_excess = np.empty((n_perm, int(active.sum())))
    for b in range(n_perm):
        # shuffle positions inside each sorted group
        perm = np.argsort(base + rng.random(len(v)), kind="stable")
        t, o = statistic(v[perm])
        exceed += int(t >= t_obs - 1e-12)
        null_excess[b] = (o - expected)[active] / n_r[active]
    ids = np.flatnonzero(active)
    per_reviewer = []
    for j, rid in enumerate(ids):
        per_reviewer.append({
            "reviewer": int(rid), "reviews": int(n_r[rid]), "non_none": float(o_obs[rid]),
            "expected_non_none": float(expected[rid]),
            "excess_rate": float((o_obs[rid] - expected[rid]) / n_r[rid]),
            "null_excess_2.5": float(np.percentile(null_excess[:, j], 2.5)),
            "null_excess_97.5": float(np.percentile(null_excess[:, j], 97.5)),
        })
    return {"statistic": t_obs, "p_value": (1 + exceed) / (1 + n_perm), "permutations": n_perm,
            "units": int(len(counts)), "reviews": int(len(v)), "reviewers": int(active.sum()),
            "per_reviewer": per_reviewer}


# ------------------------------------------------------------ data

class MLCQ:
    """Zenodo reviews, normalized samples and their link (every array indexed by json_index)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        icfg, lcfg = cfg["inputs"], cfg["labels"]
        self.cfg = cfg
        self.level_ci = float(cfg["reviewers"]["ci_level"])
        self.order = list(lcfg["severity_order"])
        self.sev_index = {s: i for i, s in enumerate(self.order)}
        self.positive = {self.sev_index[s] for s in lcfg["positive"]}
        self.smells = list(lcfg["smell_names"])
        self.shorts = list(lcfg["short_names"])
        self.display = list(lcfg["display_names"])
        if len(self.smells) != len(LABEL_ORDER):
            raise SystemExit("labels.smell_names must have one entry per LABEL_ORDER label")
        self.smell_type = dict(lcfg["smell_type"])
        self.type_level = dict(lcfg["type_level"])
        self.phases = [dict(p) for p in cfg["phases"]]
        self.phase_names = [str(p["name"]) for p in self.phases]
        self.crosscheck_phase = str(cfg.get("crosscheck_phase", self.phase_names[-1]))

        with resolve(icfg["zenodo_reviews_csv"]).open(newline="") as fh:
            raw = list(csv.DictReader(fh, delimiter=str(icfg.get("zenodo_separator", ";"))))
        self.reviews: list[dict[str, Any]] = []
        for r in raw:
            sev, smell, ts = r["severity"].strip(), r["smell"].strip(), r["review_timestamp"].strip()
            if sev not in self.sev_index or smell not in self.smells:
                raise SystemExit(f"review {r['id']}: unknown severity {sev!r} or smell {smell!r}")
            self.reviews.append({
                "id": int(r["id"]), "reviewer": int(r["reviewer_id"]), "sample": int(r["sample_id"]),
                "smell": smell, "s": self.smells.index(smell), "sev": self.sev_index[sev],
                "pos": int(self.sev_index[sev] in self.positive), "ts": ts, "phase": self.phase_of(ts),
                "type": r["type"].strip(), "repository": r["repository"].strip(),
                "key": (r["repository"].strip(), r["commit_hash"].strip(), r["path"].strip(),
                        int(r["start_line"]), int(r["end_line"])),
            })
        self.reviews.sort(key=lambda rv: (rv["ts"], rv["id"]))

        # sessions: the reviews one reviewer gave to one sample
        session_time: dict[tuple[int, int], str] = {}
        for rv in self.reviews:
            k = (rv["sample"], rv["reviewer"])
            session_time[k] = min(session_time.get(k, rv["ts"]), rv["ts"])
        self.sessions: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for (sample, reviewer), t in session_time.items():
            self.sessions[sample].append({"time": t, "reviewer": reviewer, "phase": self.phase_of(t),
                                          "reviews": []})
        for sample in self.sessions:
            self.sessions[sample].sort(key=lambda x: (x["time"], x["reviewer"]))
        rank = {(sample, s["reviewer"]): j for sample, ss in self.sessions.items() for j, s in enumerate(ss)}
        seen: set[tuple[int, int, int]] = set()
        for rv in self.reviews:
            rv["session_rank"] = rank[(rv["sample"], rv["reviewer"])]
            rv["first"] = rv["session_rank"] == 0
            k = (rv["sample"], rv["s"], rv["reviewer"])
            # a second record of the same (sample, smell, reviewer) is a duplicate submission
            rv["dup"] = k in seen
            seen.add(k)
            self.sessions[rv["sample"]][rv["session_rank"]]["reviews"].append(rv)
        self.dedup = [rv for rv in self.reviews if not rv["dup"]]

        # (sample, smell index) -> reviews in time order
        self.units: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for rv in self.reviews:
            self.units[(rv["sample"], rv["s"])].append(rv)
        # rank of each review inside its unit and the first rating of the unit
        for revs in self.units.values():
            kept = [rv for rv in revs if not rv["dup"]]
            for k, rv in enumerate(kept):
                rv["unit_rank"] = k
                rv["unit_first_pos"] = kept[0]["pos"]

        # link the normalized samples
        sample_of_key: dict[tuple, int] = {}
        key_of_sample: dict[int, tuple] = {}
        self.sample_type: dict[int, str] = {}
        self.sample_repo: dict[int, str] = {}
        for rv in self.reviews:
            if sample_of_key.setdefault(rv["key"], rv["sample"]) != rv["sample"]:
                raise SystemExit(f"two Zenodo samples share the location {rv['key']}")
            if key_of_sample.setdefault(rv["sample"], rv["key"]) != rv["key"]:
                raise SystemExit(f"Zenodo sample {rv['sample']} has two locations")
            if self.sample_type.setdefault(rv["sample"], rv["type"]) != rv["type"]:
                raise SystemExit(f"Zenodo sample {rv['sample']} has two types")
            self.sample_repo[rv["sample"]] = rv["repository"]
        self.entries = json.loads(resolve(icfg["normalized_json"]).read_text())
        rule_entries = json.loads(resolve(icfg["rule_default_json"]).read_text())
        if len(rule_entries) != len(self.entries):
            raise SystemExit("normalized and rule_default JSON files differ in length")
        self.n = len(self.entries)
        self.sample_of = np.full(self.n, -1, dtype=np.int64)
        for i, (e, e2) in enumerate(zip(self.entries, rule_entries)):
            key = (e["repo_url"], e["commit_hash"], e["file_path"], int(e["start_line"]), int(e["end_line"]))
            key2 = (e2["repo_url"], e2["commit_hash"], e2["file_path"], int(e2["start_line"]), int(e2["end_line"]))
            if key != key2:
                raise SystemExit(f"json_index {i}: normalized and rule_default samples differ")
            self.sample_of[i] = sample_of_key.get(key, -1)
        self.y_stored = self._labels(self.entries)
        self.y_rule = self._labels(rule_entries)
        self.stored_votes = np.zeros((self.n, 4), dtype=np.int64)
        for i, e in enumerate(self.entries):
            labels = e["labels"] if isinstance(e["labels"], dict) else ast.literal_eval(e["labels"])
            for s, smell in enumerate(self.smells):
                self.stored_votes[i, s] = int(labels.get(smell, {}).get("vote_count", 0))

        self.linked = self.sample_of >= 0
        self.level = np.array([self.type_level[self.sample_type[int(x)]] if x >= 0 else "" for x in self.sample_of])
        smell_level = [self.type_level[self.smell_type[s]] for s in self.smells]
        self.eligible = np.stack([self.level == lv for lv in smell_level], axis=1)
        # per (json_index, smell): every record, and the deduplicated reviews
        self.unit_reviews = [[self.units.get((int(self.sample_of[i]), s), []) for s in range(4)]
                             for i in range(self.n)]
        self.unit_dedup = [[[rv for rv in self.unit_reviews[i][s] if not rv["dup"]] for s in range(4)]
                           for i in range(self.n)]
        self.n_records = np.array([[len(x) for x in row] for row in self.unit_reviews], dtype=np.int64)
        self.n_reviews = np.array([[len(x) for x in row] for row in self.unit_dedup], dtype=np.int64)
        self.y_zenodo = np.array([[self.upper_median_label(x) for x in row] for row in self.unit_reviews])
        self.y_zenodo_dedup = np.array([[self.upper_median_label(x) for x in row] for row in self.unit_dedup])

    @staticmethod
    def _labels(entries: list[dict[str, Any]]) -> np.ndarray:
        rows = []
        for e in entries:
            y = e["y"]
            if isinstance(y, str):
                y = json.loads(y.lower())
            rows.append([int(bool(v)) for v in y[:4]])
        return np.asarray(rows, dtype=np.int64)

    def phase_of(self, ts: str) -> str:
        day = ts[:10]
        for p in self.phases:
            if str(p["start"]) <= day <= str(p["end"]):
                return str(p["name"])
        raise SystemExit(f"timestamp {ts} is outside every phase")

    def upper_median_label(self, reviews: list[dict[str, Any]]) -> int:
        if not reviews:
            return 0
        sevs = sorted(rv["sev"] for rv in reviews)
        return int(sevs[len(sevs) // 2] in self.positive)


# ------------------------------------------------------------ (1) linkage and labels

def section_linkage(d: MLCQ, out_dir: Path) -> dict[str, Any]:
    zen_samples = {rv["sample"] for rv in d.reviews}
    linked = {int(x) for x in d.sample_of if x >= 0}
    absent = sorted(zen_samples - linked)
    repo_all = Counter(d.sample_repo[s] for s in zen_samples)
    repo_absent = Counter(d.sample_repo[s] for s in absent)
    full = sorted(r for r in repo_absent if repo_absent[r] == repo_all[r])
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rv in d.reviews:
        by_sample[rv["sample"]].append(rv)
    rows = []
    for s in absent:
        rvs = by_sample[s]
        rows.append({"sample_id": s, "repository": d.sample_repo[s], "type": d.sample_type[s],
                     "reviews": len(rvs), "phases": "|".join(sorted({rv["phase"] for rv in rvs})),
                     "non_none_reviews": sum(rv["pos"] for rv in rvs),
                     "repository_entirely_absent": int(d.sample_repo[s] in full)})
    files = [atomic_csv(out_dir / "unlinked_zenodo_samples.csv", rows)]

    # stored vote counts against the Zenodo review counts
    mism = np.argwhere(d.linked[:, None] & (d.stored_votes != d.n_records))
    start = int(d.cfg["labels"]["stored_votes_checked_from_index"])
    vote_check = {
        "pairs_differing": int(len(mism)),
        "samples_differing": int(len(set(mism[:, 0].tolist()))) if len(mism) else 0,
        "json_index_range": [int(mism[:, 0].min()), int(mism[:, 0].max())] if len(mism) else None,
        "excess_distribution": dict(Counter(int(d.stored_votes[i, s] - d.n_records[i, s]) for i, s in mism)),
        f"pairs_equal_from_index_{start}": bool(
            (d.stored_votes[start:] == d.n_records[start:]).all()) if d.n > start else None,
    }

    # labels recomputed from the Zenodo reviews
    labels: dict[str, Any] = {}
    dis_rows = []
    for s, short in enumerate(d.shorts):
        ys, yr, yz, yzd = d.y_stored[:, s], d.y_rule[:, s], d.y_zenodo[:, s], d.y_zenodo_dedup[:, s]
        labels[short] = {
            "positives_stored": int(ys.sum()), "positives_rule_default": int(yr.sum()),
            "positives_zenodo_upper_median": int(yz.sum()),
            "positives_zenodo_upper_median_without_duplicate_records": int(yzd.sum()),
            "stored_positive_zenodo_negative": int(((ys == 1) & (yz == 0)).sum()),
            "stored_negative_zenodo_positive": int(((ys == 0) & (yz == 1)).sum()),
            "rule_default_differs_from_zenodo": int((yr != yz).sum()),
            "zenodo_changes_without_duplicate_records": int((yz != yzd).sum()),
        }
        for i in np.flatnonzero((ys != yz) | (yr != yz)):
            revs = d.unit_reviews[i][s]
            dis_rows.append({"json_index": int(i), "sample_id": int(d.sample_of[i]), "smell": short,
                             "y_stored": int(ys[i]), "y_rule_default": int(yr[i]), "y_zenodo": int(yz[i]),
                             "stored_vote_count": int(d.stored_votes[i, s]), "zenodo_reviews": len(revs),
                             "severities": "|".join(d.order[rv["sev"]] for rv in revs),
                             "reviewers": "|".join(str(rv["reviewer"]) for rv in revs)})
    first_only = sum(1 for r in dis_rows if r["y_stored"] == 1 and r["y_zenodo"] == 0
                     and r["severities"].split("|")[0] != d.order[0]
                     and all(x == d.order[0] for x in r["severities"].split("|")[1:]))
    stored_only = sum(1 for r in dis_rows if r["y_stored"] == 1 and r["y_zenodo"] == 0)
    files.append(atomic_csv(out_dir / "label_disagreements.csv", dis_rows,
                            ["json_index", "sample_id", "smell", "y_stored", "y_rule_default", "y_zenodo",
                             "stored_vote_count", "zenodo_reviews", "severities", "reviewers"]))

    # one row per normalized sample with its Zenodo link
    link_rows = []
    for i in range(d.n):
        row: dict[str, Any] = {"json_index": i, "sample_id": int(d.sample_of[i]), "level": d.level[i]}
        for s, short in enumerate(d.shorts):
            revs = d.unit_dedup[i][s]
            row[f"n_reviews_{short}"] = len(revs)
            row[f"reviewers_{short}"] = "|".join(str(rv["reviewer"]) for rv in revs)
            row[f"first_phase_{short}"] = revs[0]["phase"] if revs else ""
            row[f"y_stored_{short}"] = int(d.y_stored[i, s])
            row[f"y_rule_{short}"] = int(d.y_rule[i, s])
            row[f"y_zenodo_{short}"] = int(d.y_zenodo[i, s])
        link_rows.append(row)
    files.append(atomic_csv(out_dir / "sample_link.csv", link_rows))

    dup = [rv for rv in d.reviews if rv["dup"]]
    return {
        "normalized_samples": d.n,
        "normalized_linked": int(d.linked.sum()),
        "normalized_unlinked": int((~d.linked).sum()),
        "distinct_zenodo_ids_linked": len(linked),
        "zenodo_reviews": len(d.reviews),
        "zenodo_samples": len(zen_samples),
        "zenodo_reviewers": len({rv["reviewer"] for rv in d.reviews}),
        "zenodo_samples_not_in_normalized": len(absent),
        "zenodo_reviews_not_in_normalized": sum(len(by_sample[s]) for s in absent),
        "absent_by_type": dict(Counter(d.sample_type[s] for s in absent)),
        "absent_repositories": len(repo_absent),
        "absent_repositories_with_no_linked_sample": len(full),
        "absent_samples_in_those_repositories": sum(repo_absent[r] for r in full),
        "absent_repositories_list": sorted(repo_absent.items(), key=lambda x: -x[1]),
        "absent_reason_note": (
            "every absent Zenodo sample belongs to a repository with no sample in the normalized data, "
            "so samples were lost per repository (for example code that could not be retrieved, a guess); the "
            "retrieval step is not in this repository and the cause is not verified" if len(full) == len(repo_absent) else
            "some absent samples come from repositories that also have linked samples; cause not verified"),
        "duplicate_records_same_reviewer_unit": [
            {"id": rv["id"], "sample_id": rv["sample"], "smell": rv["smell"], "reviewer": rv["reviewer"],
             "severity": d.order[rv["sev"]], "timestamp": rv["ts"]} for rv in dup],
        "duplicate_records_note": ("a second record of the same (sample, smell, reviewer); the stored and "
                                   "rule labels count every record, reviewer statistics keep the first"),
        "stored_vote_counts_vs_zenodo": vote_check,
        "labels": labels,
        "stored_positive_zenodo_negative_pairs": stored_only,
        "of_which_first_review_above_none_and_every_later_review_none": first_only,
        "files": files,
    }


# ------------------------------------------------------------ (2) phases

def section_phases(d: MLCQ, figs: Figures, out_dir: Path) -> dict[str, Any]:
    lv = d.level_ci
    linked_samples = {int(x) for x in d.sample_of if x >= 0}
    rows, smell_rows = [], []
    for p in d.phases:
        name = str(p["name"])
        recs = [rv for rv in d.reviews if rv["phase"] == name]
        dd = [rv for rv in d.dedup if rv["phase"] == name]
        rows.append({
            "phase": name, "start": p["start"], "end": p["end"], "description": p.get("description", ""),
            "reviews": len(recs), "paper_reviews": p.get("paper_reviews"),
            "difference_to_paper": len(recs) - int(p["paper_reviews"]) if p.get("paper_reviews") is not None else None,
            "reviews_of_linked_samples": sum(rv["sample"] in linked_samples for rv in recs),
            "samples": len({rv["sample"] for rv in recs}),
            "sessions": len({(rv["sample"], rv["reviewer"]) for rv in recs}),
            "reviewers": len({rv["reviewer"] for rv in recs}),
            "first_session_reviews": sum(rv["first"] for rv in recs),
            "later_session_reviews": sum(not rv["first"] for rv in recs),
        })
        for s, short in enumerate(d.shorts):
            for subset in SUBSETS:
                sel = [rv for rv in dd if rv["s"] == s and (subset == "all" or rv["first"] == (subset == "first"))]
                smell_rows.append({"phase": name, "smell": short, "subset": subset,
                                   **rate(sum(rv["pos"] for rv in sel), len(sel), lv)})
    files = [atomic_csv(out_dir / "phases.csv", rows), atomic_csv(out_dir / "phase_positivity.csv", smell_rows)]

    # where the paper's phase totals fall when the reviews are sorted by time
    boundaries = []
    cum = np.cumsum([int(p["paper_reviews"]) for p in d.phases if p.get("paper_reviews") is not None])
    ordered = d.reviews  # sorted by (timestamp, id)
    for j, c in enumerate(cum[:-1]):
        last, nxt = ordered[int(c) - 1], ordered[int(c)]
        boundaries.append({"after_paper_phase": d.phase_names[j], "cumulative_reviews": int(c),
                           "last_review_timestamp": last["ts"], "last_review_date_phase": last["phase"],
                           "next_review_timestamp": nxt["ts"], "next_review_date_phase": nxt["phase"]})

    # sessions by phase and by the ratings of the earlier sessions of the sample
    kinds: Counter = Counter()
    cc = d.crosscheck_phase
    cc_start = next(str(p["start"]) for p in d.phases if str(p["name"]) == cc)
    n_earlier: Counter = Counter()
    pre_cc_earlier: Counter = Counter()
    two_earlier = Counter()
    two_earlier_presence = Counter()
    rule: Counter = Counter()
    rule_gaps: list[float] = []
    own_rereview = 0
    last_first_session = ""
    for sample, sess in d.sessions.items():
        last_first_session = max(last_first_session, sess[0]["time"])
        for j, se in enumerate(sess):
            earlier = sess[:j]
            if not earlier:
                kind = "first"
            elif any(rv["pos"] for e in earlier for rv in e["reviews"]):
                kind = "later_after_non_none"
            else:
                kind = "later_after_all_none"
            kinds[(se["phase"], kind)] += 1
            if se["phase"] != cc:
                continue
            n_earlier[len(earlier)] += 1
            pre_cc_earlier[sum(e["time"][:10] < cc_start for e in earlier)] += 1
            if se["reviewer"] in {e["reviewer"] for e in earlier}:
                own_rereview += 1
            differ = False
            if len(earlier) == 2:
                differ = any(
                    sorted(rv["sev"] for rv in earlier[0]["reviews"] if rv["s"] == s)
                    != sorted(rv["sev"] for rv in earlier[1]["reviews"] if rv["s"] == s)
                    for s in range(4))
                two_earlier["differing" if differ else "identical"] += 1
                differ_presence = any(
                    sorted(rv["pos"] for rv in earlier[0]["reviews"] if rv["s"] == s)
                    != sorted(rv["pos"] for rv in earlier[1]["reviews"] if rv["s"] == s)
                    for s in range(4))
                two_earlier_presence["differing" if differ_presence else "identical"] += 1
            # the paper's rule as written: one earlier review, or two differing earlier reviews
            if len(earlier) == 1:
                rule["one_earlier_session"] += 1
            elif len(earlier) == 2 and differ:
                rule["two_differing_earlier_sessions"] += 1
            else:
                rule["two_identical_earlier_sessions" if len(earlier) == 2
                     else "three_or_more_earlier_sessions"] += 1
                rule_gaps.append((np.datetime64(se["time"]) - np.datetime64(earlier[-1]["time"]))
                                 / np.timedelta64(1, "s"))

    # per (sample, smell): cross-check reviews whose unit had only none ratings before
    unit_check: Counter = Counter()
    unit_fresh_pos = Counter()
    for (sample, s), revs in d.units.items():
        sess = d.sessions[sample]
        for rv in revs:
            if rv["dup"] or rv["phase"] != cc:
                continue
            t0 = sess[rv["session_rank"]]["time"]
            earlier = [x for x in revs if sess[x["session_rank"]]["time"] < t0]
            if not earlier:
                key = "no_earlier_review_of_this_smell"
            elif any(x["pos"] for x in earlier):
                key = "earlier_non_none_for_this_smell"
            else:
                key = "earlier_reviews_of_this_smell_all_none"
            unit_check[(d.shorts[s], key)] += 1
            unit_fresh_pos[(d.shorts[s], key)] += rv["pos"]
    crosscheck = {
        "phase": cc,
        "sessions": sum(v for (ph, _), v in kinds.items() if ph == cc),
        "sessions_by_earlier_ratings": {k: v for (ph, k), v in kinds.items() if ph == cc},
        "sessions_with_no_earlier_session": int(kinds[(cc, "first")]),
        "sessions_whose_earlier_sessions_were_all_none": int(kinds[(cc, "later_after_all_none")]),
        "number_of_earlier_sessions": dict(sorted(n_earlier.items())),
        "number_of_earlier_sessions_before_phase_start": dict(sorted(pre_cc_earlier.items())),
        "sessions_with_two_earlier_sessions": dict(two_earlier),
        "sessions_with_two_earlier_sessions_compared_on_presence": dict(two_earlier_presence),
        "sessions_by_paper_rule_as_written": dict(rule),
        "sessions_not_matching_paper_rule_as_written": len(rule_gaps),
        "gap_to_previous_session_seconds_for_those": {
            "quantiles_0_10_50_90_100": np.percentile(rule_gaps, [0, 10, 50, 90, 100]).tolist() if rule_gaps else None,
            "share_below_600_seconds": float(np.mean(np.array(rule_gaps) < 600)) if rule_gaps else None},
        "sessions_by_a_reviewer_who_reviewed_the_sample_before": own_rereview,
        "reviews_per_smell_by_earlier_ratings_of_that_smell": [
            {"smell": sm, "earlier": k, "reviews": v, "non_none": unit_fresh_pos[(sm, k)],
             "non_none_rate": unit_fresh_pos[(sm, k)] / v if v else None}
            for (sm, k), v in sorted(unit_check.items())],
        "note": ("the paper selects cross-checked samples on an earlier rating above none for the sample; "
                 "a smell can be cross-checked while its own earlier ratings were all none when the other "
                 "smell of the sample triggered the selection; 'earlier' means sessions with an earlier "
                 "start time, including sessions of the same phase. The paper's rule (samples rated above none "
                 "with 1 review or 2 differing reviews) holds for the 'above none' part in every session, but "
                 "sessions_not_matching_paper_rule_as_written counts sessions that came after 2 identical "
                 "earlier sessions (severities of every smell equal) or after 3 or more; "
                 "gap_to_previous_session_seconds_for_those tells whether they could be two reviewers served "
                 "the same sample at the same time (gaps of seconds or minutes) or later re-selections"),
    }
    later_gap = [se for ss in d.sessions.values() for j, se in enumerate(ss)
                 if j > 0 and last_first_session < se["time"] and se["time"][:10] < cc_start]
    sessions_by_phase = defaultdict(dict)
    for (ph, k), v in sorted(kinds.items()):
        sessions_by_phase[ph][k] = v

    # figure: reviews per phase and rate of non-none ratings per smell and phase
    plt = figs.plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.5]})
    x = np.arange(len(rows))
    first = np.array([r["first_session_reviews"] for r in rows])
    later = np.array([r["later_session_reviews"] for r in rows])
    axes[0].bar(x, first, color=OI["blue"], label="première revue de l'échantillon")
    axes[0].bar(x, later, bottom=first, color=OI["orange"], label="revue ultérieure de l'échantillon")
    for k, r in enumerate(rows):
        axes[0].annotate(f"{r['reviews']}\n(article : {r['paper_reviews']})", (k, r["reviews"]),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
    axes[0].set_xticks(x, [f"phase {r['phase']}" for r in rows])
    axes[0].set_ylabel("Nombre de revues")
    axes[0].set_ylim(0, max(r["reviews"] for r in rows) * 1.25)
    axes[0].set_title("Revues par phase d'échantillonnage")
    axes[0].legend(loc="upper left", fontsize=7)
    axes[0].grid(axis="x", visible=False)
    width = 0.18
    for s, short in enumerate(d.shorts):
        sel = [r for r in smell_rows if r["smell"] == short and r["subset"] == "all"]
        xs = x + (s - 1.5) * width
        ys = np.array([r["rate"] for r in sel])
        err = np.array([[r["rate"] - r["ci_low"], r["ci_high"] - r["rate"]] for r in sel]).T
        axes[1].errorbar(xs, ys, yerr=err, fmt="o", ms=4, capsize=2, color=SMELL_COLORS[s], label=d.display[s])
    axes[1].set_xticks(x, [f"phase {r['phase']}" for r in rows])
    axes[1].set_ylabel("Part des revues au-dessus de « none »")
    axes[1].set_title("Taux de notes non nulles par phase (IC de Wilson à 95 %)")
    axes[1].legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    axes[1].grid(axis="x", visible=False)
    figs.save(fig, "phases")

    return {"phases": rows, "positivity": smell_rows, "paper_count_boundaries": boundaries,
            "sessions_by_phase": dict(sessions_by_phase),
            "last_first_session_time": last_first_session,
            "later_sessions_after_last_first_session_before_crosscheck_dates": {
                "sessions": len(later_gap),
                "reviews": sum(len(se["reviews"]) for se in later_gap),
                "first": min((se["time"] for se in later_gap), default=None),
                "reviewers": sorted({se["reviewer"] for se in later_gap})},
            "crosscheck_check": crosscheck,
            "phase_rule": "date part of review_timestamp in the inclusive date ranges of the config",
            "files": files}


# ------------------------------------------------------------ (3) reviewers

def subset_reviews(d: MLCQ, subset: str, s: int | None = None) -> list[dict[str, Any]]:
    return [rv for rv in d.dedup if (s is None or rv["s"] == s)
            and (subset == "all" or rv["first"] == (subset == "first"))]


def rank_category(rv: dict[str, Any], subset: str, max_rank: int) -> int:
    """Rank of the review in its unit, ranks above max_rank pooled."""
    low = 1 if subset == "later" else 0
    return int(min(max(rv["unit_rank"], low), max_rank))


def rank_columns(revs: list[dict[str, Any]], subset: str, max_rank: int) -> dict[str, np.ndarray]:
    """Rank indicators (lowest rank as reference), and for later reviews their product with the first rating."""
    cat = np.array([rank_category(rv, subset, max_rank) for rv in revs])
    first = np.array([rv["unit_first_pos"] for rv in revs], dtype=np.float64)
    cols: dict[str, np.ndarray] = {}
    levels = sorted(set(cat.tolist()))
    for c in levels[1:]:
        name = f"rank_{c}" + ("_or_more" if c == max_rank else "")
        cols[name] = (cat == c).astype(np.float64)
        if subset == "later":
            cols[f"{name}_x_first_rating_non_none"] = cols[name] * first
    return cols


def rank_table(revs: list[dict[str, Any]], max_rank: int, level: float) -> list[dict[str, Any]]:
    """Rate of non-none ratings per rank of the review in its unit, and the reviewers who
    wrote most of the reviews of that rank."""
    rows = []
    for c in sorted({min(rv["unit_rank"], max_rank) for rv in revs}):
        sel = [rv for rv in revs if min(rv["unit_rank"], max_rank) == c]
        who = Counter(rv["reviewer"] for rv in sel)
        rows.append({"rank": f"{c}+" if c == max_rank else str(c),
                     **rate(sum(rv["pos"] for rv in sel), len(sel), level),
                     "top_reviewers_share": {str(r): k / len(sel) for r, k in who.most_common(3)}})
    return rows


def section_reviewers(d: MLCQ, figs: Figures, out_dir: Path, rng: np.random.Generator) -> dict[str, Any]:
    rcfg = d.cfg["reviewers"]
    lv = d.level_ci
    reviewers = sorted({rv["reviewer"] for rv in d.reviews})
    counts = Counter(rv["reviewer"] for rv in d.reviews)
    total = len(d.reviews)
    top_k = int(rcfg["top_k_share"])
    top = [r for r, _ in counts.most_common(top_k)]
    summary_rows = []
    for r in sorted(reviewers, key=lambda x: -counts[x]):
        recs = [rv for rv in d.reviews if rv["reviewer"] == r]
        row = {"reviewer": r, "reviews": counts[r], "share_of_reviews": counts[r] / total,
               "samples": len({rv["sample"] for rv in recs}),
               "first_session_reviews": sum(rv["first"] for rv in recs),
               "later_session_reviews": sum(not rv["first"] for rv in recs),
               "first_review": recs[0]["ts"], "last_review": recs[-1]["ts"]}
        for ph in d.phase_names:
            row[f"reviews_phase_{ph}"] = sum(rv["phase"] == ph for rv in recs)
        summary_rows.append(row)
    files = [atomic_csv(out_dir / "reviewer_summary.csv", summary_rows)]

    # phase-specific pooled rates on first-session reviews, for the expected counts
    pooled_first = {}
    for s in range(4):
        for ph in d.phase_names:
            sel = [rv for rv in subset_reviews(d, "first", s) if rv["phase"] == ph]
            pooled_first[(s, ph)] = (sum(rv["pos"] for rv in sel) / len(sel)) if sel else float("nan")

    pos_rows = []
    for s, short in enumerate(d.shorts):
        for subset in SUBSETS:
            sel = subset_reviews(d, subset, s)
            pooled = rate(sum(rv["pos"] for rv in sel), len(sel), lv)
            pos_rows.append({"reviewer": "all", "smell": short, "subset": subset, **pooled})
            by_r: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for rv in sel:
                by_r[rv["reviewer"]].append(rv)
            for r in sorted(by_r, key=lambda x: -len(by_r[x])):
                k = sum(rv["pos"] for rv in by_r[r])
                row = {"reviewer": r, "smell": short, "subset": subset, **rate(k, len(by_r[r]), lv)}
                if subset == "first":
                    exp = sum(pooled_first[(s, rv["phase"])] for rv in by_r[r])
                    row["expected_given_phase"] = exp
                    row["observed_over_expected"] = k / exp if exp > 0 else float("nan")
                pos_rows.append(row)
    files.append(atomic_csv(out_dir / "reviewer_positivity.csv", pos_rows,
                            ["reviewer", "smell", "subset", "n", "k", "rate", "ci_low", "ci_high",
                             "expected_given_phase", "observed_over_expected"]))

    # heterogeneity tests
    min_n = int(rcfg["min_reviews_test"])
    n_perm = int(rcfg["permutations"])
    max_rank = int(rcfg["max_rank_category"])
    tests: dict[str, Any] = {}
    perm_rows = []
    for s, short in enumerate(d.shorts):
        t: dict[str, Any] = {}
        for subset in ("first", "all"):
            sel = subset_reviews(d, subset, s)
            by_r = Counter(rv["reviewer"] for rv in sel)
            keep = sorted(r for r, c in by_r.items() if c >= min_n)
            table = np.array([[sum(rv["pos"] for rv in sel if rv["reviewer"] == r),
                               sum(1 - rv["pos"] for rv in sel if rv["reviewer"] == r)] for r in keep])
            if len(keep) >= 2 and table.sum(axis=0).min() > 0:
                chi2, p, dof, expected = stats.chi2_contingency(table)
                t[f"chi2_{subset}"] = {"statistic": float(chi2), "dof": int(dof), "p_value": float(p),
                                       "reviewers": len(keep), "reviews": int(table.sum()),
                                       "min_expected_count": float(expected.min()),
                                       "reviewers_left_out": len(by_r) - len(keep)}
        # likelihood-ratio test of the reviewer on first-session reviews, adjusted for the phase
        sel = subset_reviews(d, "first", s)
        by_r = Counter(rv["reviewer"] for rv in sel)
        keep = [r for r, c in by_r.most_common() if c >= min_n]
        sel = [rv for rv in sel if rv["reviewer"] in set(keep)]
        phases_here = sorted({rv["phase"] for rv in sel})
        y = np.array([rv["pos"] for rv in sel], dtype=np.float64)
        Xp = np.column_stack([np.ones(len(sel))] + [[rv["phase"] == ph for rv in sel] for ph in phases_here[1:]])
        Xr = np.column_stack([[rv["reviewer"] == r for rv in sel] for r in keep[1:]]) if len(keep) > 1 else None
        if Xr is not None:
            dev0, ok0 = logistic_deviance(Xp.astype(np.float64), y)
            dev1, ok1 = logistic_deviance(np.column_stack([Xp, Xr]).astype(np.float64), y)
            lr = max(dev0 - dev1, 0.0)
            no_pos = [r for r in keep if not any(rv["pos"] for rv in sel if rv["reviewer"] == r)]
            t["lr_first_adjusted_for_phase"] = {
                "statistic": lr, "dof": len(keep) - 1, "p_value": float(stats.chi2.sf(lr, len(keep) - 1)),
                "reviewers": len(keep), "reference_reviewer": keep[0], "phases": phases_here,
                "reviews": len(sel), "converged": ok0 and ok1, "reviewers_with_no_non_none_rating": no_pos}
        # permutation tests with the units fixed, and a conditional logistic regression with the review rank
        for name, subset in (("all_reviews", "all"), ("later_reviews", "later")):
            sel = subset_reviews(d, subset, s)
            per_unit = Counter(rv["sample"] for rv in sel)
            sel = [rv for rv in sel if per_unit[rv["sample"]] >= 2]
            if len(sel) < 2:
                continue
            res = within_unit_permutation(np.array([rv["sample"] for rv in sel]),
                                          np.array([rv["reviewer"] for rv in sel]),
                                          np.array([rv["pos"] for rv in sel]), n_perm, rng)
            for row in res.pop("per_reviewer"):
                perm_rows.append({"smell": short, "test": f"permutation_{name}", **row})
            t[f"permutation_{name}"] = res
            t[f"rating_by_rank_{name}"] = rank_table(sel, max_rank, lv)
            t[f"clogit_rank_adjusted_{name}"] = {
                **rank_adjusted_reviewer_test(sel, rank_columns(sel, subset, max_rank), min_n),
                "rank_columns": list(rank_columns(sel, subset, max_rank)),
            }
            if subset == "later":
                t[f"clogit_rank_adjusted_{name}"]["units_whose_first_rating_is_in_the_subset"] = len(
                    {rv["sample"] for rv in sel if rv["unit_rank"] == 0})
        tests[short] = t
    files.append(atomic_csv(out_dir / "reviewer_excess_vs_coreviewers.csv", perm_rows))
    tests["note"] = (
        "chi2_first and lr_first use first-session reviews (random sampling phases), chi2_all mixes them with "
        "re-reviews of selected samples; the likelihood-ratio test adds phase indicators to both models; "
        "reviewers could skip samples and the paper does not say how samples were assigned to reviewers, so "
        "first-session rates also reflect which samples each reviewer chose to rate. "
        "The permutation tests shuffle the ratings of each (sample, smell) among its reviewers and use only "
        "units with at least two reviews in the subset, a subset selected by the cross-check (earlier rating "
        "above none), so they compare reviewers on the same units but on selected units; "
        "permutation_later_reviews leaves out the first rating, which triggered the selection. Their null "
        "hypothesis treats the reviews of a unit as exchangeable, which the sequential cross-check breaks: a "
        "unit got a further review only when its earlier ratings were above none or differed, so the rate of "
        "non-none ratings depends on the rank of the review in the unit (rating_by_rank_*), and reviewers are "
        "spread unevenly over ranks; the permutation p-values therefore mix reviewer and rank effects. "
        "clogit_rank_adjusted_* fits, on the same reviews, a conditional logistic regression (one free "
        "intercept per unit, so unit effects cancel) with review-rank indicators (ranks above "
        "max_rank_category pooled; for later reviews also rank x first rating of the smell) and tests the "
        "reviewer indicators by a likelihood ratio (chi-square); it removes the average rank effect only, "
        "not every dependence created by the selection, so it is a sensitivity analysis, not an exact test")

    # pairwise agreement between reviewers on shared (sample, smell) units
    min_overlap = int(rcfg["min_overlap"])
    n_boot = int(rcfg.get("pair_bootstrap", 1000))
    unit_ratings: dict[tuple[int, int], dict[int, tuple[str, int]]] = defaultdict(dict)
    for rv in d.dedup:
        unit_ratings[(rv["sample"], rv["s"])][rv["reviewer"]] = (rv["ts"], rv["pos"])
    pair_data: dict[tuple[int, int, str], list[tuple[int, int, int]]] = defaultdict(list)
    for (sample, s), ratings in unit_ratings.items():
        for a, b in itertools.combinations(sorted(ratings), 2):
            (ta, pa), (tb, pb) = ratings[a], ratings[b]
            earlier_pos = pa if ta <= tb else pb
            pair_data[(a, b, d.shorts[s])].append((pa, pb, earlier_pos))
            pair_data[(a, b, "all")].append((pa, pb, earlier_pos))
    pair_rows = []
    for (a, b, sm), vals in sorted(pair_data.items(), key=lambda x: (x[0][2], -len(x[1]))):
        if len(vals) < min_overlap:
            continue
        arr = np.array(vals)
        ka = float(cohen_kappa(arr[:, 0], arr[:, 1]))
        idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
        boot = cohen_kappa(arr[idx, 0], arr[idx, 1])
        boot = boot[np.isfinite(boot)]
        pa_pos, pb_pos = arr[:, 0] == 1, arr[:, 1] == 1
        pair_rows.append({
            "reviewer_a": a, "reviewer_b": b, "smell": sm, "units": len(arr),
            "agreement": float((arr[:, 0] == arr[:, 1]).mean()), "kappa": ka,
            "kappa_ci_low": float(np.percentile(boot, 2.5)) if len(boot) else float("nan"),
            "kappa_ci_high": float(np.percentile(boot, 97.5)) if len(boot) else float("nan"),
            "rate_a": float(pa_pos.mean()), "rate_b": float(pb_pos.mean()),
            "p_b_given_a": float(arr[pa_pos, 1].mean()) if pa_pos.any() else float("nan"),
            "p_a_given_b": float(arr[pb_pos, 0].mean()) if pb_pos.any() else float("nan"),
            "share_earlier_rating_non_none": float(arr[:, 2].mean()),
        })
    files.append(atomic_csv(out_dir / "pairwise_agreement.csv", pair_rows))

    # confirmation of a first rating by the second reviewer of the unit
    conf_first: dict[tuple[str, int], list[int]] = defaultdict(list)
    conf_second: dict[tuple[str, int], list[int]] = defaultdict(list)
    after_none: dict[str, list[int]] = defaultdict(list)
    for (sample, s), revs in d.units.items():
        revs = [rv for rv in revs if not rv["dup"]]
        if len(revs) < 2:
            continue
        first, second = revs[0], revs[1]
        if first["pos"]:
            for sm in (d.shorts[s], "all"):
                conf_first[(sm, first["reviewer"])].append(second["pos"])
                conf_second[(sm, second["reviewer"])].append(second["pos"])
        else:
            after_none[d.shorts[s]].append(second["pos"])
            after_none["all"].append(second["pos"])
    conf_rows = []
    conf_summary = {}
    for sm in [*d.shorts, "all"]:
        allv = [v for (m, _), vals in conf_first.items() if m == sm for v in vals]
        conf_summary[sm] = {"second_non_none_given_first_non_none": rate(sum(allv), len(allv), lv),
                            "second_non_none_given_first_none": rate(sum(after_none[sm]), len(after_none[sm]), lv)}
        for role, table in (("first_reviewer", conf_first), ("second_reviewer", conf_second)):
            for (m, r), vals in sorted(table.items(), key=lambda x: -len(x[1])):
                if m == sm:
                    conf_rows.append({"smell": sm, "role": role, "reviewer": r, **rate(sum(vals), len(vals), lv)})
    files.append(atomic_csv(out_dir / "confirmation.csv", conf_rows))

    # positive labels decided by a single review
    single_rows = []
    single_summary: dict[str, Any] = {}
    for source, Y in (("stored", d.y_stored), ("rule_default", d.y_rule), ("zenodo", d.y_zenodo)):
        single_summary[source] = {}
        for s, short in enumerate(d.shorts):
            pos = np.flatnonzero(Y[:, s] == 1)
            n_rev = d.n_reviews[pos, s]
            single = pos[n_rev == 1]
            by_r = Counter(d.unit_dedup[i][s][0]["reviewer"] for i in single)
            single_summary[source][short] = {
                "positives": int(len(pos)), "decided_by_one_review": int(len(single)),
                "share": len(single) / len(pos) if len(pos) else float("nan"),
                "reviews_distribution": dict(sorted(Counter(n_rev.tolist()).items())),
                "single_review_positives_by_reviewer": dict(by_r.most_common()),
                f"share_of_positives_decided_alone_by_top_{top_k}": (
                    sum(by_r[r] for r in top) / len(pos) if len(pos) else float("nan")),
            }
            for r, c in by_r.most_common():
                single_rows.append({"label_source": source, "smell": short, "reviewer": r,
                                    "positives_decided_alone": c, "share_of_positives": c / len(pos)})
    files.append(atomic_csv(out_dir / "single_review_positives.csv", single_rows))

    # reviewers behind the stored positive and negative labels
    source_rows = []
    sources: dict[str, Any] = {}
    for s, short in enumerate(d.shorts):
        pos = np.flatnonzero((d.y_stored[:, s] == 1) & d.eligible[:, s])
        neg = np.flatnonzero((d.y_stored[:, s] == 0) & d.eligible[:, s])
        first_r = Counter(d.unit_dedup[i][s][0]["reviewer"] for i in pos)
        neg_src = Counter()
        for i in neg:
            revs = d.unit_dedup[i][s]
            neg_src["not_reviewed" if not revs else ("several" if len(revs) > 1 else f"R{revs[0]['reviewer']}")] += 1
        sources[short] = {"positives": int(len(pos)), "positives_by_first_reviewer": dict(first_r.most_common()),
                          "negatives": int(len(neg)), "negatives_by_source": dict(neg_src.most_common()),
                          "negatives_decided_by_one_review": int(sum(v for k, v in neg_src.items()
                                                                     if k.startswith("R")))}
        for r, c in first_r.most_common():
            source_rows.append({"smell": short, "label": "positive", "source": f"first_review_R{r}", "units": c,
                                "share": c / len(pos)})
        for k, c in neg_src.most_common():
            source_rows.append({"smell": short, "label": "negative", "source": k, "units": c, "share": c / len(neg)})
    files.append(atomic_csv(out_dir / "label_sources.csv", source_rows))

    forest_figure(d, figs, pos_rows, int(rcfg["min_reviews_figure"]))
    pairwise_figure(d, figs, pair_rows, int(rcfg["heatmap_reviewers"]))
    confirmation_figure(d, figs, conf_rows, int(rcfg["min_confirmation_figure"]))
    label_sources_figure(d, figs, sources, top)

    return {
        "reviewers": len(reviewers),
        f"top_{top_k}_reviewers": top,
        f"top_{top_k}_share_of_reviews": sum(counts[r] for r in top) / total,
        "reviews_per_reviewer": dict(counts.most_common()),
        "tests": tests,
        "pairwise_note": ("pairs sharing at least min_overlap units; the shared units are mostly cross-check "
                          "units selected on an earlier rating above none (share_earlier_rating_non_none), so "
                          "kappa and agreement are conditional on that selection; kappa intervals are "
                          "percentile bootstrap over units"),
        "pairs_reported": len(pair_rows),
        "confirmation": conf_summary,
        "confirmation_note": ("first and second review of a (sample, smell) unit in time order; units with a "
                              "second review were selected on an earlier rating above none for the sample, so a "
                              "unit whose first rating of this smell was none got its second review because the "
                              "other smell of the sample was rated above none: the two rows compare differently "
                              "selected units, not the same units with different first ratings"),
        "single_review_positives": single_summary,
        "single_review_note": "number of reviews counted on the Zenodo release, duplicate records removed",
        "label_sources": sources,
        "label_sources_note": ("stored labels of the units each smell is reviewed for; positives by the reviewer "
                               "of their first review, negatives by the review that decided them: R<id> a single "
                               "review of that reviewer, several two or more reviews, not_reviewed no review of "
                               "the smell"),
        "files": files,
    }


def reviewer_label(r: Any) -> str:
    return f"R{r}"


def forest_figure(d: MLCQ, figs: Figures, pos_rows: list[dict[str, Any]], min_n: int) -> None:
    plt = figs.plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for s, short in enumerate(d.shorts):
        ax = axes.flat[s]
        rows = {sub: {r["reviewer"]: r for r in pos_rows if r["smell"] == short and r["subset"] == sub}
                for sub in ("first", "later")}
        shown = sorted({r for sub in rows for r, x in rows[sub].items() if r != "all" and x["n"] >= min_n},
                       key=lambda r: -sum(rows[sub][r]["n"] for sub in rows if r in rows[sub]))
        ypos = np.arange(len(shown))[::-1]
        for k, (sub, color, marker, name) in enumerate((
                ("first", OI["blue"], "o", "première revue de l'échantillon"),
                ("later", OI["orange"], "s", "revue ultérieure (contre-vérification)"))):
            pts = [(y, rows[sub][r]) for y, r in zip(ypos, shown) if r in rows[sub] and rows[sub][r]["n"] >= min_n]
            if pts:
                ys = np.array([y for y, _ in pts]) + (0.17 if k == 0 else -0.17)
                xs = np.array([p["rate"] for _, p in pts])
                err = np.array([[p["rate"] - p["ci_low"], p["ci_high"] - p["rate"]] for _, p in pts]).T
                ax.errorbar(xs, ys, xerr=err, fmt=marker, ms=3.5, capsize=1.5, lw=0.9, color=color,
                            label=name if s == 0 else None)
            if "all" in rows[sub]:
                ax.axvline(rows[sub]["all"]["rate"], color=color, ls="--", lw=0.8)
        labels = []
        for r in shown:
            nf = rows["first"][r]["n"] if r in rows["first"] else 0
            nl = rows["later"][r]["n"] if r in rows["later"] else 0
            labels.append(f"{reviewer_label(r)} ({nf} / {nl})")
        ax.set_yticks(ypos, labels, fontsize=7)
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(d.display[s])
        ax.set_xlabel("Part des revues au-dessus de « none »")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Taux de notes non nulles par relecteur (IC de Wilson à 95 %)\n"
                 f"entre parenthèses : revues en première revue / en revue ultérieure (au moins {min_n} affichées) ; "
                 "tirets : taux de l'ensemble", fontsize=9)
    fig.legend(loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    figs.save(fig, "forest")


def pairwise_figure(d: MLCQ, figs: Figures, pair_rows: list[dict[str, Any]], n_show: int) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    plt = figs.plt
    cmap = LinearSegmentedColormap.from_list("oi_diverging", [OI["vermillion"], "#ffffff", OI["blue"]])
    cmap.set_bad("#f2f2f2")
    totals: Counter = Counter()
    for r in pair_rows:
        if r["smell"] == "all":
            totals[r["reviewer_a"]] += r["units"]
            totals[r["reviewer_b"]] += r["units"]
    shown = sorted([r for r, _ in totals.most_common(n_show)])
    if len(shown) < 2:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    im = None
    for s, short in enumerate(d.shorts):
        ax = axes.flat[s]
        mat = np.full((len(shown), len(shown)), np.nan)
        num = np.zeros((len(shown), len(shown)), dtype=int)
        for r in pair_rows:
            if r["smell"] == short and r["reviewer_a"] in shown and r["reviewer_b"] in shown:
                i, j = shown.index(r["reviewer_a"]), shown.index(r["reviewer_b"])
                mat[i, j] = mat[j, i] = r["kappa"]
                num[i, j] = num[j, i] = r["units"]
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=-1, vmax=1)
        for i in range(len(shown)):
            for j in range(len(shown)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}\nn={num[i, j]}", ha="center", va="center", fontsize=6)
        ax.set_xticks(range(len(shown)), [reviewer_label(r) for r in shown], fontsize=7)
        ax.set_yticks(range(len(shown)), [reviewer_label(r) for r in shown], fontsize=7)
        ax.grid(False)
        ax.set_title(d.display[s])
    fig.colorbar(im, ax=axes, shrink=0.6, label="kappa de Cohen (présence de l'odeur)")
    fig.suptitle("Accord entre paires de relecteurs sur les unités communes\n"
                 "(cases vides : moins d'unités communes que le seuil ; unités surtout issues de la contre-vérification)",
                 fontsize=9)
    figs.save(fig, "pairwise_kappa")


def confirmation_figure(d: MLCQ, figs: Figures, conf_rows: list[dict[str, Any]], min_n: int) -> None:
    plt = figs.plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for ax, role, title in ((axes[0], "first_reviewer", "selon le premier relecteur"),
                            (axes[1], "second_reviewer", "selon le second relecteur")):
        rows = [r for r in conf_rows if r["smell"] == "all" and r["role"] == role and r["n"] >= min_n]
        rows.sort(key=lambda r: r["n"])
        y = np.arange(len(rows))
        if rows:
            xs = np.array([r["rate"] for r in rows])
            err = np.array([[r["rate"] - r["ci_low"], r["ci_high"] - r["rate"]] for r in rows]).T
            ax.errorbar(xs, y, xerr=err, fmt="o", ms=4, capsize=2, color=OI["blue"])
            ax.set_yticks(y, [f"{reviewer_label(r['reviewer'])} (n={r['n']})" for r in rows], fontsize=7)
        pooled = [r for r in conf_rows if r["smell"] == "all" and r["role"] == role]
        k, n = sum(r["k"] for r in pooled), sum(r["n"] for r in pooled)
        if n:
            ax.axvline(k / n, color=GREY, ls="--", lw=0.8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("Part des notes non nulles confirmées")
        ax.set_title(f"Confirmation {title}")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Confirmation d'une première note non nulle par la revue suivante, toutes odeurs "
                 f"(IC de Wilson à 95 %, au moins {min_n} unités ; tirets : ensemble)", fontsize=9)
    fig.tight_layout()
    figs.save(fig, "confirmation")


def label_sources_figure(d: MLCQ, figs: Figures, sources: dict[str, Any], top: list[int]) -> None:
    plt = figs.plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    y = np.arange(len(d.shorts))[::-1]
    colors = SERIES_COLORS[: len(top)] + [GREY, "#dddddd", "#f2f2f2"]
    panels = (
        ("positives", "Étiquettes positives, par premier relecteur",
         [(f"{reviewer_label(r)}", lambda st, r=r: st["positives_by_first_reviewer"].get(r, 0)) for r in top]
         + [("autres relecteurs", lambda st: st["positives"] - sum(st["positives_by_first_reviewer"].get(r, 0)
                                                                    for r in top))]),
        ("negatives", "Étiquettes négatives, par revue décisive",
         [(f"{reviewer_label(r)}, revue unique", lambda st, r=r: st["negatives_by_source"].get(f"R{r}", 0))
          for r in top]
         + [("autres relecteurs, revue unique",
             lambda st: st["negatives_decided_by_one_review"] - sum(st["negatives_by_source"].get(f"R{r}", 0)
                                                                      for r in top)),
            ("plusieurs revues", lambda st: st["negatives_by_source"].get("several", 0)),
            ("non relu", lambda st: st["negatives_by_source"].get("not_reviewed", 0))]),
    )
    for ax, (key, title, segments) in zip(axes, panels):
        left = np.zeros(len(d.shorts))
        for (name, fn), color in zip(segments, colors):
            shares = np.array([fn(sources[sh]) / sources[sh][key] for sh in d.shorts])
            ax.barh(y, shares, left=left, color=color, label=name, edgecolor="white", linewidth=0.5)
            left += shares
        ax.set_xlim(0, 1)
        ax.set_xlabel("Part des étiquettes stockées")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7)
    axes[0].set_yticks(y, [f"{d.display[s]}\n({sources[sh]['positives']} pos. / {sources[sh]['negatives']} nég.)"
                           for s, sh in enumerate(d.shorts)])
    fig.suptitle("Relecteurs derrière les étiquettes stockées (unités du niveau relu pour chaque odeur)", fontsize=9)
    fig.tight_layout()
    figs.save(fig, "label_sources")


# ------------------------------------------------------------ (4) selection

def section_selection(d: MLCQ, figs: Figures, out_dir: Path) -> dict[str, Any]:
    lv = d.level_ci
    max_bin = int(d.cfg["selection"]["max_reviews_bin"])
    rows = []
    out: dict[str, Any] = {}
    for s, short in enumerate(d.shorts):
        elig = np.flatnonzero(d.eligible[:, s] & d.linked)
        nrev = np.minimum(d.n_reviews[elig, s], max_bin)
        first_pos = np.array([d.unit_dedup[i][s][0]["pos"] if d.unit_dedup[i][s] else -1 for i in elig])
        out[short] = {}
        for b in range(0, max_bin + 1):
            m = nrev == b
            row = {"smell": short, "reviews": f"{b}+" if b == max_bin else str(b), "units": int(m.sum())}
            for src, Y in (("stored", d.y_stored), ("rule_default", d.y_rule)):
                r = rate(int(Y[elig[m], s].sum()), int(m.sum()), lv)
                row.update({f"{src}_positive": r["k"], f"{src}_rate": r["rate"],
                            f"{src}_ci_low": r["ci_low"], f"{src}_ci_high": r["ci_high"]})
            fm = m & (first_pos >= 0)
            r = rate(int(first_pos[fm].sum()), int(fm.sum()), lv)
            row.update({"first_review_non_none": r["k"], "first_review_non_none_rate": r["rate"]})
            rows.append(row)
        multi = d.n_reviews[elig, s] >= 2
        single = d.n_reviews[elig, s] == 1
        # positives whose non-none ratings are not a strict majority (upper median on a tie)
        pos_units = elig[d.y_stored[elig, s] == 1]
        nn = np.array([sum(rv["pos"] for rv in d.unit_reviews[i][s]) for i in pos_units])
        nr = np.array([len(d.unit_reviews[i][s]) for i in pos_units])
        out[short] = {
            "eligible_units": int(len(elig)),
            "units_with_two_or_more_reviews": int(multi.sum()),
            "stored_positive_rate_single_review": rate(int(d.y_stored[elig[single], s].sum()), int(single.sum()), lv),
            "stored_positive_rate_two_or_more_reviews": rate(int(d.y_stored[elig[multi], s].sum()), int(multi.sum()), lv),
            "share_of_stored_positives_with_two_or_more_reviews": (
                float(d.y_stored[elig[multi], s].sum() / max(d.y_stored[elig, s].sum(), 1))),
            "first_review_non_none_rate_single_review": rate(int(first_pos[single].sum()), int(single.sum()), lv),
            "first_review_non_none_rate_two_or_more_reviews": rate(int(first_pos[multi].sum()), int(multi.sum()), lv),
            "stored_positives_non_none_exactly_half_of_reviews": int((2 * nn == nr).sum()),
            "stored_positives_non_none_minority_of_reviews": int((2 * nn < nr).sum()),
            "stored_positives_two_reviews_one_non_none": int(((nr == 2) & (nn == 1)).sum()),
        }
    files = [atomic_csv(out_dir / "selection_by_reviews.csv", rows)]

    plt = figs.plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = [f"{b}+" if b == max_bin else str(b) for b in range(max_bin + 1)]
    x = np.arange(len(bins))
    for s, short in enumerate(d.shorts):
        sel = [r for r in rows if r["smell"] == short]
        ok = [k for k, r in enumerate(sel) if r["units"] > 0]
        xs = x[ok] + (s - 1.5) * 0.12
        ys = np.array([sel[k]["stored_rate"] for k in ok])
        err = np.array([[sel[k]["stored_rate"] - sel[k]["stored_ci_low"], sel[k]["stored_ci_high"] - sel[k]["stored_rate"]]
                        for k in ok]).T
        axes[0].errorbar(xs, ys, yerr=err, fmt="o-", ms=4, capsize=2, lw=0.9, color=SMELL_COLORS[s],
                         label=d.display[s])
        axes[1].plot(x[ok] + (s - 1.5) * 0.12, [sel[k]["units"] for k in ok], "o-", ms=4, lw=0.9,
                     color=SMELL_COLORS[s])
    axes[0].set_xticks(x, bins)
    axes[0].set_xlabel("Nombre de revues de l'unité (échantillon, odeur)")
    axes[0].set_ylabel("Part d'étiquettes positives stockées")
    axes[0].set_title("P(étiquette positive | nombre de revues), IC de Wilson à 95 %")
    axes[1].set_xticks(x, bins)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Nombre de revues de l'unité (échantillon, odeur)")
    axes[1].set_ylabel("Nombre d'unités (échelle log)")
    axes[1].set_title("Unités par nombre de revues")
    fig.legend(*axes[0].get_legend_handles_labels(), ncols=4, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    figs.save(fig, "selection")

    return {"by_smell": out, "table": rows, "files": files,
            "note": ("units of the level each smell is reviewed for (method for feature envy and long method, "
                     "class for blob and data class), numbers of reviews from the Zenodo release without "
                     "duplicate records; a unit receives a second review mainly through the cross-check, which "
                     "selected samples with an earlier rating above none, so P(positive) grows with the number "
                     "of reviews by design (with two reviews the upper median is the higher rating, so one non-none "
                     "rating out of two gives a positive) and every agreement statistic computed on units with two or more "
                     "reviews (Krippendorff alpha, kappa, unanimity) describes this selected subset, not the "
                     "single-review units that make up most of the data")}


# ------------------------------------------------------------ (5) model errors

def model_files(mcfg: dict[str, Any]) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    missing: list[str] = []
    if mcfg.get("records"):
        records = json.loads(resolve(mcfg["records"]).read_text())
        root = resolve(mcfg["local_root"])
        for r in records:
            p = root / Path(str(r[mcfg.get("run_dir_field", "run_dir")])).name / str(mcfg.get("file_name", "predictions.jsonl"))
            (files.append(p) if p.exists() else missing.append(rel(p)))
    for pattern in mcfg.get("paths", []) or []:
        matched = sorted(glob.glob(str(resolve(pattern))))
        if not matched:
            missing.append(str(pattern))
        files.extend(Path(m) for m in matched)
    return files, missing


def unit_groups(d: MLCQ) -> dict[str, dict[tuple[int, str], np.ndarray]]:
    """Grouping name -> {(smell index, group): json indices}."""
    max_bin = int(d.cfg["model_errors"]["max_reviews_bin"])
    groupings: dict[str, dict[tuple[int, str], list[int]]] = {
        name: defaultdict(list) for name in ("label_source", "first_reviewer", "first_rating", "first_phase",
                                             "reviews", "reviewer_involved", "stored_vs_rule")}
    for i in range(d.n):
        for s in range(4):
            revs = d.unit_dedup[i][s]
            if not d.eligible[i, s]:
                base = "other_level"
            elif not revs:
                base = "not_reviewed"
            else:
                base = ""
            if base:
                for g in ("label_source", "first_reviewer", "first_rating", "first_phase", "reviews"):
                    groupings[g][(s, base)].append(i)
            else:
                groupings["label_source"][(s, f"R{revs[0]['reviewer']}" if len(revs) == 1 else "several")].append(i)
                groupings["first_reviewer"][(s, f"R{revs[0]['reviewer']}")].append(i)
                groupings["first_rating"][(s, "first_non_none" if revs[0]["pos"] else "first_none")].append(i)
                groupings["first_phase"][(s, f"phase{revs[0]['phase']}")].append(i)
                groupings["reviews"][(s, str(len(revs)) if len(revs) < max_bin else f"{max_bin}+")].append(i)
                for r in {rv["reviewer"] for rv in revs}:
                    groupings["reviewer_involved"][(s, f"R{r}")].append(i)
            groupings["stored_vs_rule"][(s, "equal" if d.y_stored[i, s] == d.y_rule[i, s] else "differ")].append(i)
    return {g: {k: np.array(v, dtype=np.int64) for k, v in m.items()} for g, m in groupings.items()}


def seed_predictions(rows: list[dict[str, Any]], n: int, steps: int,
                     in_folds: np.ndarray) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Per complete seed: (thresholded test predictions, test labels), -1 outside the folds."""
    by_key: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_key[(int(r["seed"]), int(r["fold"]), str(r["part"]))].append(r)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    info: dict[str, Any] = {"incomplete_seeds": {}, "duplicate_rows": 0}
    for seed in sorted({k[0] for k in by_key}):
        folds = sorted({k[1] for k in by_key if k[0] == seed and k[2] == "test"})
        pred = np.full((n, 4), -1, dtype=np.int64)
        ytrue = np.full((n, 4), -1, dtype=np.int64)
        problem = None
        for f in folds:
            val, test = by_key.get((seed, f, "val"), []), by_key[(seed, f, "test")]
            if not val:
                problem = f"fold {f} has no validation rows"
                break
            thr = tune_thresholds(np.array([r["y"][:4] for r in val]), np.array([r["prob"][:4] for r in val]), steps)
            idx = np.array([int(r["json_index"]) for r in test])
            if (pred[idx, 0] >= 0).any() or len(set(idx.tolist())) != len(idx):
                info["duplicate_rows"] += 1
                problem = f"fold {f} repeats test samples"
                break
            pred[idx] = (np.array([r["prob"][:4] for r in test]) >= thr.reshape(1, -1)).astype(np.int64)
            ytrue[idx] = np.array([r["y"][:4] for r in test])
        if problem is None and ((pred[:, 0] >= 0) != in_folds).any():
            problem = (f"test parts of folds {folds} miss {int((in_folds & (pred[:, 0] < 0)).sum())} samples "
                       f"of the folds file and hold {int((~in_folds & (pred[:, 0] >= 0)).sum())} others")
        if problem:
            info["incomplete_seeds"][seed] = problem
            continue
        out[seed] = (pred, ytrue)
    return out, info


def section_model_errors(d: MLCQ, figs: Figures, out_dir: Path) -> dict[str, Any]:
    mcfg = d.cfg["model_errors"]
    lv = d.level_ci
    steps = int(mcfg["threshold_steps"])
    strategy = str(mcfg["strategy"])
    min_units = per_rate_type(mcfg["min_group_units"])
    if str(mcfg.get("label_source", "stored")) != "stored":
        raise SystemExit("model_errors.label_source: only 'stored' is supported (the labels the models were trained on)")
    folds = json.loads(resolve(d.cfg["inputs"]["folds"]).read_text())
    fold_index = folds["meta"]["json_index"]
    fold_index = json.loads(fold_index) if isinstance(fold_index, str) else fold_index
    in_folds = np.zeros(d.n, dtype=bool)
    in_folds[np.asarray(fold_index, dtype=np.int64)] = True
    # samples outside the folds are never scored, so they leave every group
    groups = {g: {k: idx[in_folds[idx]] for k, idx in m.items()} for g, m in unit_groups(d).items()}
    fig_min_units = per_rate_type(mcfg.get("figure_min_group_units", mcfg["min_group_units"]))
    rows_out: list[dict[str, Any]] = []
    fig_rows: list[dict[str, Any]] = []
    models_info: dict[str, Any] = {}
    for spec in mcfg["models"]:
        files, missing = model_files(spec)
        if not files:
            models_info[spec["name"]] = {"status": "no prediction files", "missing": missing}
            if not spec.get("optional"):
                print(f"[reviewers] {spec['name']}: no prediction files", flush=True)
            continue
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for f in files:
            for r in read_jsonl(f):
                if r.get("strategy") == strategy:
                    by_model[str(r.get("model", spec["name"]))].append(r)
        if not by_model:
            models_info[spec["name"]] = {"status": f"no prediction rows for strategy {strategy}",
                                         "files": len(files)}
            print(f"[reviewers] {spec['name']}: no rows for strategy {strategy}", flush=True)
            continue
        for row_model, rows in sorted(by_model.items()):
            name = spec["name"] if len(by_model) == 1 else f"{spec['name']}:{row_model}"
            seeds, info = seed_predictions(rows, d.n, steps, in_folds)
            label_mismatch = int(sum(((yt != d.y_stored) & (yt >= 0)).sum() for _, yt in seeds.values()))
            models_info[name] = {"label": spec.get("label", name), "files": len(files),
                                 "missing_files": len(missing), "missing_examples": missing[:5],
                                 "seeds_used": sorted(seeds), **info,
                                 "test_label_cells_differing_from_stored": label_mismatch}
            if not seeds:
                continue
            seed_ids = sorted(seeds)
            for grouping, members in groups.items():
                counts: dict[tuple[int, str], np.ndarray] = {}
                for (s, g), idx in members.items():
                    arr = np.zeros((len(seed_ids), 4))
                    for k, sd in enumerate(seed_ids):
                        pred, _ = seeds[sd]
                        yt = d.y_stored[idx, s]
                        pr = pred[idx, s]
                        arr[k] = [(yt == 1).sum(), ((yt == 1) & (pr == 0)).sum(),
                                  (yt == 0).sum(), ((yt == 0) & (pr == 1)).sum()]
                    counts[(s, g)] = arr
                rows_out.extend(error_rows(name, grouping, counts, min_units, lv, d.shorts))
                if grouping in FIGURE_GROUPINGS:
                    fig_rows.extend(error_rows(name, grouping, counts, fig_min_units, lv, d.shorts))
            print(f"[reviewers] model errors: {name} seeds={seed_ids}", flush=True)
    files = [atomic_csv(out_dir / "model_errors.csv", rows_out)]
    fig_models = [m for m in mcfg.get("figure_models", []) if m in models_info and models_info[m].get("seeds_used")]
    labels = {m: models_info[m]["label"] for m in fig_models}
    if fig_models:
        # figures: small reviewer groups are pooled, other small groups and other-level units are left out
        shown = [r for r in fig_rows if r["group"] != "other_level"
                 and (r["group"] == "other_reviewers" or r["units"] >= fig_min_units[r["rate_type"]])]
        model_error_figure(d, figs, shown, fig_models, labels, "first_reviewer", "fn", "model_fn_by_reviewer",
                           fig_min_units["fn"])
        model_error_figure(d, figs, shown, fig_models, labels, "label_source", "fp", "model_fp_by_reviewer",
                           fig_min_units["fp"])
        model_phase_figure(d, figs, shown, fig_models, labels, fig_min_units)
    return {"strategy": strategy, "models": models_info, "files": files,
            "samples_in_folds": int(in_folds.sum()),
            "samples_outside_folds": np.flatnonzero(~in_folds).tolist(),
            "method": ("per fold, thresholds tuned on the validation part with cv.tune_thresholds (binary F1, grid "
                       f"linspace(0, 1, {steps})), applied to the test part; each sample is in one test part per "
                       "seed, so counts are per seed; rate = mean errors over seeds / units, Wilson interval on "
                       "(mean errors, units), rate_seed_min/max give the spread over seeds; fn rate over stored "
                       "positives, fp rate over stored negatives"),
            "groupings": {
                "label_source": ("R<id>: unit rated by a single review of that reviewer; several: two or more "
                                 "reviews; not_reviewed: no review of the smell although the sample has the level; "
                                 "other_level: smell not reviewed for that level (label 0 by construction); "
                                 "other_reviewers: reviewers with fewer than min_group_units units, pooled per "
                                 "rate type (members column)"),
                "first_reviewer": ("reviewer of the first review of the unit, whatever the number of reviews; "
                                   "almost every stored positive has two or more reviews, so this is the grouping "
                                   "that splits the positives; pooled like label_source"),
                "first_rating": "first review of the unit above none or not",
                "first_phase": "phase of the first review of the unit",
                "reviews": "number of reviews of the unit (Zenodo, duplicate records removed)",
                "reviewer_involved": "every reviewer who rated the unit; a unit counts once for each of them",
                "stored_vs_rule": "stored label equal to or different from the rule_default label"}}


def per_rate_type(value: Any) -> dict[str, int]:
    """A threshold given once, or as {fn: ..., fp: ...}."""
    if isinstance(value, dict):
        return {"fn": int(value["fn"]), "fp": int(value["fp"])}
    return {"fn": int(value), "fp": int(value)}


def error_rows(model: str, grouping: str, counts: dict[tuple[int, str], np.ndarray], min_units: dict[str, int],
               level: float, shorts: list[str]) -> list[dict[str, Any]]:
    """Rows of false negative and false positive rates from per-seed counts
    [positives, false negatives, negatives, false positives] of every group."""
    rows = []
    n_seeds = len(next(iter(counts.values())))
    for rate_type, (n_col, e_col) in (("fn", (0, 1)), ("fp", (2, 3))):
        threshold = min_units[rate_type]
        entries = dict(counts)
        pooled_members: dict[int, list[str]] = {}
        if grouping in POOLED_GROUPINGS:
            # reviewers with few units of this rate type are pooled
            for s in range(len(shorts)):
                small = sorted(g for (ss, g), a in counts.items()
                               if ss == s and g.startswith("R") and 0 < a[0, n_col] < threshold)
                if small:
                    entries[(s, "other_reviewers")] = sum(counts[(s, g)] for g in small)
                    for g in small:
                        entries.pop((s, g))
                    pooled_members[s] = small
        for (s, g), a in sorted(entries.items(), key=lambda x: (x[0][0], x[0][1])):
            n_units = float(a[0, n_col])
            if n_units <= 0:
                continue
            per_seed = a[:, e_col] / a[:, n_col]
            mean_err = float(a[:, e_col].mean())
            lo, hi = wilson(mean_err, n_units, level)
            members = pooled_members.get(s, []) if g == "other_reviewers" else []
            rows.append({
                "model": model, "grouping": grouping, "group": g, "smell": shorts[s],
                "rate_type": rate_type, "units": int(n_units), "errors_mean": mean_err,
                "rate": mean_err / n_units, "ci_low": lo, "ci_high": hi,
                "rate_seed_min": float(per_seed.min()), "rate_seed_max": float(per_seed.max()),
                "seeds": n_seeds, "members": "|".join(members)})
    return rows


GROUP_FR = {"several": "plusieurs revues", "other_reviewers": "autres relecteurs", "not_reviewed": "non relu",
            "other_level": "autre niveau", "first_non_none": "1re note non nulle", "first_none": "1re note « none »"}
POOLED_GROUPINGS = ("label_source", "first_reviewer")
FIGURE_GROUPINGS = ("label_source", "first_reviewer", "first_phase")


def group_label(g: str) -> str:
    if g in GROUP_FR:
        return GROUP_FR[g]
    if g.startswith("phase"):
        return f"phase {g[5:]}"
    return g


def model_error_figure(d: MLCQ, figs: Figures, rows: list[dict[str, Any]], models: list[str],
                       labels: dict[str, str], grouping: str, rate_type: str, name: str, min_units: int) -> None:
    plt = figs.plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    width = 0.8 / len(models)
    for s, short in enumerate(d.shorts):
        ax = axes.flat[s]
        sel = [r for r in rows if r["grouping"] == grouping and r["rate_type"] == rate_type and r["smell"] == short]
        ref = [r for r in sel if r["model"] == models[0]]
        singles = sorted([r for r in ref if r["group"].startswith("R")], key=lambda r: -r["units"])
        order = [r["group"] for r in singles] + [g for g in ("other_reviewers", "several", "not_reviewed", "other_level")
                                                 if any(r["group"] == g for r in ref)]
        units = {r["group"]: r["units"] for r in ref}
        x = np.arange(len(order))
        for k, m in enumerate(models):
            pts = {r["group"]: r for r in sel if r["model"] == m}
            xs = [x[j] + (k - (len(models) - 1) / 2) * width for j, g in enumerate(order) if g in pts]
            ys = np.array([pts[g]["rate"] for g in order if g in pts])
            err = np.array([[pts[g]["rate"] - pts[g]["ci_low"], pts[g]["ci_high"] - pts[g]["rate"]]
                            for g in order if g in pts]).T
            if len(xs):
                ax.errorbar(xs, ys, yerr=err, fmt="o", ms=3.5, capsize=1.5, lw=0.9, color=SERIES_COLORS[k],
                            label=labels[m] if s == 0 else None)
        ax.set_xticks(x, [f"{group_label(g)}\n(n={units[g]})" for g in order], fontsize=7,
                      rotation=0 if len(order) <= 7 else 30, ha="center" if len(order) <= 7 else "right")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(d.display[s])
        ax.set_ylabel("Taux de faux négatifs" if rate_type == "fn" else "Taux de faux positifs")
        ax.grid(axis="x", visible=False)
    what = ("faux négatifs (sur les étiquettes positives)" if rate_type == "fn"
            else "faux positifs (sur les étiquettes négatives)")
    origin = ("selon le premier relecteur de l'unité" if grouping == "first_reviewer"
              else "selon l'origine de l'étiquette (revue unique d'un relecteur ou plusieurs revues)")
    fig.suptitle(f"Taux de {what} {origin}\n"
                 "seuils réglés sur la validation de chaque pli, moyenne sur les graines, IC de Wilson à 95 % ; "
                 f"relecteurs regroupés sous {min_units} unités",
                 fontsize=9)
    fig.legend(loc="lower center", ncols=min(len(models), 4), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    figs.save(fig, name)


def model_phase_figure(d: MLCQ, figs: Figures, rows: list[dict[str, Any]], models: list[str],
                       labels: dict[str, str], min_units: dict[str, int]) -> None:
    plt = figs.plt
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharey="row")
    width = 0.8 / len(models)
    for t, rate_type in enumerate(("fn", "fp")):
        for s, short in enumerate(d.shorts):
            ax = axes[t, s]
            sel = [r for r in rows if r["grouping"] == "first_phase" and r["rate_type"] == rate_type
                   and r["smell"] == short and r["group"].startswith("phase")]
            order = sorted({r["group"] for r in sel})
            units = {r["group"]: r["units"] for r in sel if r["model"] == models[0]}
            x = np.arange(len(order))
            for k, m in enumerate(models):
                pts = {r["group"]: r for r in sel if r["model"] == m}
                xs = [x[j] + (k - (len(models) - 1) / 2) * width for j, g in enumerate(order) if g in pts]
                ys = np.array([pts[g]["rate"] for g in order if g in pts])
                err = np.array([[pts[g]["rate"] - pts[g]["ci_low"], pts[g]["ci_high"] - pts[g]["rate"]]
                                for g in order if g in pts]).T
                if len(xs):
                    ax.errorbar(xs, ys, yerr=err, fmt="o", ms=3.5, capsize=1.5, lw=0.9, color=SERIES_COLORS[k],
                                label=labels[m] if (t == 0 and s == 0) else None)
            ax.set_xticks(x, [f"{group_label(g)}\n(n={units.get(g, 0)})" for g in order], fontsize=7)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(axis="x", visible=False)
            if t == 0:
                ax.set_title(d.display[s])
            if s == 0:
                ax.set_ylabel("Taux de faux négatifs" if rate_type == "fn" else "Taux de faux positifs")
    fig.suptitle("Erreurs des modèles selon la phase de la première revue de l'unité "
                 f"(moyenne sur les graines, IC de Wilson à 95 %, groupes d'au moins {min_units['fn']} positifs "
                 f"et {min_units['fp']} négatifs)", fontsize=9)
    fig.legend(loc="lower center", ncols=min(len(models), 4), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    figs.save(fig, "model_errors_by_phase")


# ------------------------------------------------------------ (6) survey

DURATION = re.compile(r"^\s*(?:(\d+(?:[.,]\d+)?)\s*Y)?\s*(?:(\d+)\s*M)?\s*$", re.IGNORECASE)


def parse_years(text: str) -> float | None:
    m = DURATION.match(text or "")
    if not m or (m.group(1) is None and m.group(2) is None):
        return None
    years = float(m.group(1).replace(",", ".")) if m.group(1) else 0.0
    return years + (int(m.group(2)) / 12 if m.group(2) else 0.0)


def section_survey(d: MLCQ, figs: Figures, out_dir: Path, rng: np.random.Generator) -> dict[str, Any]:
    scfg = d.cfg["survey"]
    lv = d.level_ci
    with resolve(d.cfg["inputs"]["survey_csv"]).open(newline="") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip() for h in rows[0]]
    id_col = str(scfg["reviewer_id_column"]).strip()
    if id_col not in header:
        return {"linked": False, "reason": f"no column {id_col!r} in the survey"}
    k_id = header.index(id_col)
    ids = [r[k_id].strip() for r in rows[1:]]
    zen = {rv["reviewer"] for rv in d.reviews}
    if any(not x.isdigit() for x in ids) or len(set(ids)) != len(ids):
        return {"linked": False, "reason": "reviewer_id values are not unique integers"}
    ids_int = [int(x) for x in ids]
    unknown = sorted(set(ids_int) - zen)
    if unknown:
        return {"linked": False, "reason": f"survey reviewer ids absent from the reviews: {unknown}"}

    def column(q: str) -> int:
        q = q.strip()
        if q not in header:
            raise SystemExit(f"survey column not found: {q!r}")
        return header.index(q)

    exp_cols = {k: column(v) for k, v in scfg["experience_columns"].items()}
    cat_cols = {k: column(v) for k, v in scfg["categorical_columns"].items()}
    unparsed = []
    people = []
    for r, rid in zip(rows[1:], ids_int):
        p: dict[str, Any] = {"reviewer": rid}
        for k, c in exp_cols.items():
            v = parse_years(r[c])
            p[k] = v
            if v is None and r[c].strip():
                unparsed.append({"reviewer": rid, "field": k, "value": r[c]})
        for k, c in cat_cols.items():
            p[k] = r[c].strip()
        for s, short in enumerate(d.shorts):
            for subset in ("first", "all"):
                sel = [rv for rv in subset_reviews(d, subset, s) if rv["reviewer"] == rid]
                rr = rate(sum(rv["pos"] for rv in sel), len(sel), lv)
                p[f"n_{subset}_{short}"] = rr["n"]
                p[f"rate_{subset}_{short}"] = rr["rate"]
        people.append(p)
    files = [atomic_csv(out_dir / "survey_reviewers.csv", people)]

    min_n = int(scfg["min_reviews_correlation"])
    min_people = int(scfg["min_reviewers_correlation"])
    n_perm = int(scfg["permutations"])

    def spearman_perm(xv: np.ndarray, yv: np.ndarray) -> tuple[float, float]:
        rho = float(stats.spearmanr(xv, yv).statistic)
        null = np.array([stats.spearmanr(xv, rng.permutation(yv)).statistic for _ in range(n_perm)])
        return rho, (1 + int((np.abs(null) >= abs(rho) - 1e-12).sum())) / (1 + n_perm)

    corr: dict[str, Any] = {}
    for key in exp_cols:
        corr[key] = {}
        for short in d.shorts:
            pts = [(p[key], p[f"rate_first_{short}"]) for p in people
                   if p[key] is not None and p[f"n_first_{short}"] >= min_n]
            if len(pts) < min_people:
                corr[key][short] = {"reviewers": len(pts), "spearman": None}
                continue
            xv, yv = np.array(pts).T
            rho, p_perm = spearman_perm(xv, yv)
            corr[key][short] = {"reviewers": len(pts), "spearman": rho, "p_value_permutation_two_sided": p_perm}
    # Holm adjustment over every (experience field, smell) correlation tested above
    tested = [(k, sm) for k in corr for sm in corr[k] if corr[k][sm].get("spearman") is not None]
    order = sorted(tested, key=lambda ks: corr[ks[0]][ks[1]]["p_value_permutation_two_sided"])
    running = 0.0
    for j, (k, sm) in enumerate(order):
        running = max(running, min(1.0, (len(order) - j) * corr[k][sm]["p_value_permutation_two_sided"]))
        corr[k][sm]["p_value_holm"] = running

    # self-reported experience against activity (number of reviews written)
    counts = Counter(rv["reviewer"] for rv in d.reviews)
    activity: dict[str, Any] = {}
    for key in exp_cols:
        pts = [(p[key], counts[p["reviewer"]]) for p in people if p[key] is not None]
        if len(pts) >= min_people:
            rho, p_perm = spearman_perm(*np.array(pts, dtype=np.float64).T)
            activity[key] = {"reviewers": len(pts), "spearman": rho, "p_value_permutation_two_sided": p_perm}

    fam: dict[str, Any] = {}
    for s, short in enumerate(d.shorts):
        groups: dict[str, list[int]] = defaultdict(list)
        for p in people:
            groups[p["smell_familiarity"] or "(no answer)"].extend(
                rv["pos"] for rv in subset_reviews(d, "first", s) if rv["reviewer"] == p["reviewer"])
        fam[short] = {g: {**rate(sum(v), len(v), lv), "reviewers": sum(
            1 for p in people if (p["smell_familiarity"] or "(no answer)") == g)} for g, v in groups.items()}

    no_survey = sorted(zen - set(ids_int))

    plt = figs.plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    for s, short in enumerate(d.shorts):
        ax = axes.flat[s]
        pts = [p for p in people if p["programming_years"] is not None and p[f"n_first_{short}"] >= min_n]
        for p in pts:
            rr = rate(p[f"rate_first_{short}"] * p[f"n_first_{short}"], p[f"n_first_{short}"], lv)
            ax.errorbar(p["programming_years"], rr["rate"], yerr=[[rr["rate"] - rr["ci_low"]], [rr["ci_high"] - rr["rate"]]],
                        fmt="o", ms=3 + 5 * math.sqrt(p[f"n_first_{short}"] / 1000), capsize=1.5, lw=0.8,
                        color=SMELL_COLORS[s])
            ax.annotate(reviewer_label(p["reviewer"]), (p["programming_years"], rr["rate"]), xytext=(3, 2),
                        textcoords="offset points", fontsize=6)
        c = corr["programming_years"][short]
        note = (f"Spearman = {c['spearman']:.2f}, p perm. = {c['p_value_permutation_two_sided']:.2f} "
                f"(Holm : {c['p_value_holm']:.2f}), {c['reviewers']} relecteurs"
                if c.get("spearman") is not None else f"{c['reviewers']} relecteurs")
        ax.set_title(f"{d.display[s]}\n{note}", fontsize=8.5)
        ax.set_ylim(-0.02, 1.02)
        if s % 2 == 0:
            ax.set_ylabel("Part des premières revues\nau-dessus de « none »")
        if s >= 2:
            ax.set_xlabel("Expérience professionnelle en programmation déclarée (années)")
    fig.suptitle("Expérience déclarée dans l'enquête et taux de notes non nulles en première revue\n"
                 f"(relecteurs ayant au moins {min_n} premières revues de l'odeur, IC de Wilson à 95 % ; "
                 f"p de Holm sur les {len(order)} corrélations testées)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figs.save(fig, "survey_experience")

    return {
        "linked": True,
        "link": (f"survey column {id_col!r} holds the reviewer_id of the review table; {len(ids_int)} survey rows, "
                 "all ids unique and present in the reviews"),
        "survey_reviewers": sorted(ids_int),
        "reviewers_without_survey": no_survey,
        "reviews_by_reviewers_without_survey": sum(counts[r] for r in no_survey),
        "share_of_reviews_without_survey": sum(counts[r] for r in no_survey) / len(d.reviews),
        "unparsed_durations": unparsed,
        "spearman_experience_vs_first_review_rate": corr,
        "correlations_tested": len(order),
        "spearman_experience_vs_reviews_written": activity,
        "first_review_rate_by_smell_familiarity": fam,
        "note": ("experience is self-reported; positivity uses first-session reviews (random sampling phases) to "
                 "avoid the cross-check selection, but reviewers were active in different phases with different "
                 "sample populations; with about a dozen reviewers per smell these correlations are descriptive; "
                 "p_value_holm adjusts the permutation p-values over every (experience field, smell) pair tested, "
                 "fields that are strongly correlated with each other, so Holm is conservative here; "
                 "spearman_experience_vs_reviews_written measures how far experience goes with activity (all "
                 "survey reviewers with the field answered); "
                 "the paper's senior, regular and junior grouping is not in the release and is not reconstructed"),
        "files": files,
    }


# ------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config/experiments/mlcq_reviewers.yml"))
    args, extra = parser.parse_known_args()
    cfg = load_config(resolve(args.config), parse_cli_overrides(extra))
    ocfg = cfg["output"]
    out_dir = resolve(ocfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = Figures(resolve(ocfg["figures_dir"]), str(ocfg["figure_prefix"]))
    rng = np.random.default_rng(int(cfg["reviewers"]["seed"]))

    d = MLCQ(cfg)
    print(f"[reviewers] {len(d.reviews)} Zenodo reviews, {d.n} normalized samples, "
          f"{int(d.linked.sum())} linked", flush=True)
    summary: dict[str, Any] = {"config": str(args.config), "label_order": LABEL_ORDER,
                               "ci": f"Wilson score interval at level {d.level_ci} unless stated"}
    summary["linkage"] = section_linkage(d, out_dir)
    print("[reviewers] (1) linkage done", flush=True)
    summary["phases"] = section_phases(d, figs, out_dir)
    print("[reviewers] (2) phases done", flush=True)
    summary["reviewers"] = section_reviewers(d, figs, out_dir, rng)
    print("[reviewers] (3) reviewer effects done", flush=True)
    summary["selection"] = section_selection(d, figs, out_dir)
    print("[reviewers] (4) selection done", flush=True)
    summary["model_errors"] = section_model_errors(d, figs, out_dir)
    print("[reviewers] (5) model errors done", flush=True)
    summary["survey"] = section_survey(d, figs, out_dir, rng)
    print("[reviewers] (6) survey done", flush=True)
    summary["figures"] = figs.written
    atomic_json(out_dir / "summary.json", summary)
    print(f"[reviewers] wrote {rel(out_dir / 'summary.json')} and {len(figs.written)} figure files", flush=True)


if __name__ == "__main__":
    main()
