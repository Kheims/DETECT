"""Descriptive analysis of the MLCQ labels and of the shortcuts that explain them.

Covers prevalence, severity, Krippendorff alpha, repository concentration,
size-feature ROC-AUC and duplicates. Writes summary.json and figures.

Usage:
    python scripts/labels/analyze_mlcq.py --config config/experiments/mlcq_analysis.yml
    python scripts/labels/analyze_mlcq.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402

# one colour per smell in LABEL_ORDER, then neutral and ordinal steps
SMELL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NEUTRAL = "#c9c8c3"
SEVERITY_COLORS = ["#d9d8d4", "#86b6ef", "#2a78d6", "#104281"]
DIVERGING = ["#b5302f", "#e88a89", "#f0efec", "#86b6ef", "#1c5cab"]
INK, INK_SECONDARY = "#0b0b0b", "#52514e"


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


# ------------------------------------------------------------ Krippendorff alpha

def coincidence_contributions(units: list[list[int]], n_values: int) -> np.ndarray:
    """Per-unit contribution to the coincidence matrix, shape (units, V, V) (Krippendorff 2011)."""
    out = np.zeros((len(units), n_values, n_values), dtype=np.float64)
    for u, values in enumerate(units):
        m = len(values)
        if m < 2:
            continue
        counts = np.bincount(np.asarray(values, dtype=np.int64), minlength=n_values).astype(np.float64)
        out[u] = (np.outer(counts, counts) - np.diag(counts)) / (m - 1)
    return out


def distance_matrix(marginals: np.ndarray, metric: str) -> np.ndarray:
    """Squared difference function delta^2_ck of Krippendorff (2011)."""
    v = len(marginals)
    if metric == "nominal":
        return 1.0 - np.eye(v)
    if metric == "ordinal":
        # delta^2_ck = (sum_{g=c..k} n_g - (n_c + n_k) / 2)^2 for c <= k
        d = np.zeros((v, v))
        for c in range(v):
            for k in range(c, v):
                d[c, k] = d[k, c] = (marginals[c:k + 1].sum() - (marginals[c] + marginals[k]) / 2.0) ** 2
        return d
    if metric == "interval":
        idx = np.arange(v, dtype=np.float64)
        return (idx[:, None] - idx[None, :]) ** 2
    raise ValueError(f"unknown metric: {metric}")


def alpha_from_coincidences(o: np.ndarray, metric: str) -> float:
    """alpha = 1 - D_o / D_e = 1 - (n - 1) sum o_ck d_ck / sum n_c n_k d_ck."""
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")
    d = distance_matrix(n_c, metric)
    expected = float((np.outer(n_c, n_c) * d).sum())
    if expected == 0.0:
        return float("nan")
    return 1.0 - (n - 1.0) * float((o * d).sum()) / expected


def krippendorff_alpha(units: list[list[int]], n_values: int, metric: str) -> float:
    return alpha_from_coincidences(coincidence_contributions(units, n_values).sum(axis=0), metric)


def alpha_with_ci(units: list[list[int]], n_values: int, metric: str, n_boot: int, level: float,
                  rng: np.random.Generator) -> dict[str, Any]:
    contrib = coincidence_contributions(units, n_values)
    o = contrib.sum(axis=0)
    alpha = alpha_from_coincidences(o, metric)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(units), size=len(units))
        boots.append(alpha_from_coincidences(contrib[idx].sum(axis=0), metric))
    boots = np.asarray([b for b in boots if not math.isnan(b)])
    lo, hi = (np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2]) if len(boots) else (np.nan, np.nan))
    n = o.sum()
    return {"alpha": alpha, "ci_low": float(lo), "ci_high": float(hi), "units": len(units),
            "pairable_values": float(n), "observed_agreement": float(np.trace(o) / n) if n else float("nan")}


def self_test() -> None:
    """Checks the alpha implementation on hand-computable data."""
    # two coders, four binary units: alpha = 8 / 15 by hand
    units = [[0, 0], [0, 1], [1, 1], [1, 1]]
    a = krippendorff_alpha(units, 2, "nominal")
    print(f"[self-test] hand example nominal alpha={a:.6f} expected={8 / 15:.6f}")
    assert abs(a - 8 / 15) < 1e-12
    # with two values the ordinal and nominal alphas coincide
    assert abs(krippendorff_alpha(units, 2, "ordinal") - a) < 1e-12
    # perfect agreement and a unit with a single value (not pairable)
    assert krippendorff_alpha([[0, 0], [1, 1, 1], [2]], 3, "nominal") == 1.0
    # Krippendorff (2011) section C example with missing data
    table = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
        [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
    ]
    units = [[row[u] - 1 for row in table if row[u] is not None] for u in range(12)]
    for metric, expected in (("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849)):
        value = krippendorff_alpha(units, 5, metric)
        print(f"[self-test] Krippendorff 2011 example {metric} alpha={value:.4f} published={expected}")
        assert abs(value - expected) < 5e-4
    print("[self-test] ok")


# ------------------------------------------------------------ helpers

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def gini(values: list[float] | np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=np.float64))
    if len(x) == 0 or x.sum() == 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((len(x) + 1 - 2 * cum.sum() / cum[-1]) / len(x))


def lorenz(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64))
    cum = np.concatenate([[0.0], np.cumsum(x)])
    return np.linspace(0, 1, len(x) + 1), cum / cum[-1]


def safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def tmp_path(path: Path) -> Path:
    """Per-process temporary name, so that concurrent writers never share one."""
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path(path)
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


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
            "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK, "text.color": INK,
            "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
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
            self.written.append(str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path))
        self.plt.close(fig)


# ------------------------------------------------------------ analyses

class Data:
    def __init__(self, cfg: dict[str, Any]) -> None:
        acfg, lcfg = cfg["analysis"], cfg["labels"]
        self.cfg, self.acfg, self.lcfg = cfg, acfg, lcfg
        self.meta = read_csv(resolve(acfg["sample_meta_csv"]))
        size_rows = {int(r["sample_idx"]): r for r in read_csv(resolve(acfg["size_features_csv"]))}
        self.n = len(self.meta)
        # every array below is indexed by json_index
        if [int(r["json_index"]) for r in self.meta] != list(range(self.n)) or set(size_rows) != set(range(self.n)):
            raise SystemExit("sample metadata and size features must hold json_index 0..n-1, in order")
        self.size_cols = list(acfg["size_columns"])
        self.size = np.array([[float(size_rows[i][c]) for c in self.size_cols] for i in range(self.n)])
        self.shorts = list(lcfg["short_names"])
        self.smells = list(lcfg["smell_names"])
        self.order = list(lcfg["severity_order"])
        self.positive = set(lcfg["positive"])
        self.level = np.array([r["level"] for r in self.meta])
        self.repo = np.array([r["repo"] for r in self.meta])
        self.parse_ok = np.array([int(r["parse_ok"]) for r in self.meta], dtype=bool)
        self.y = np.array([[int(r[f"y_{s}"]) for s in self.shorts] for r in self.meta])
        self.y_raw = np.array([[int(r[f"y_raw_{s}"]) for s in self.shorts] for r in self.meta])
        self.reviews = np.array([[int(r[f"n_reviews_{s}"]) for s in self.shorts] for r in self.meta])
        # raw severity votes, shape (n, smells, severities)
        self.votes = np.array([[[int(r[f"votes_{s}_{v}"]) for v in self.order] for s in self.shorts]
                               for r in self.meta])
        level_of = {s: lv for lv, names in lcfg["levels"].items() for s in names}
        self.smell_level = [level_of[s] for s in self.smells]
        self.eligible = np.stack([self.level == lv for lv in self.smell_level], axis=1)


def prevalence(d: Data, figs: Figures) -> dict[str, Any]:
    out: dict[str, Any] = {"samples": d.n, "by_level": {}, "parse_ok": int(d.parse_ok.sum())}
    for lv in sorted(set(d.level)):
        mask = d.level == lv
        entry = {"samples": int(mask.sum()), "parse_ok": int((mask & d.parse_ok).sum())}
        for k, s in enumerate(d.shorts):
            reviewed = mask & (d.reviews[:, k] > 0)
            pos = int(d.y[mask, k].sum())
            entry[s] = {"reviewed": int(reviewed.sum()), "positives": pos,
                        "prevalence_among_reviewed": pos / int(reviewed.sum()) if reviewed.any() else None,
                        "prevalence_among_level": pos / int(mask.sum())}
        entry["label_cardinality"] = {str(c): int(v) for c, v in sorted(Counter(d.y[mask].sum(axis=1)).items())}
        out["by_level"][lv] = entry
    out["positives"] = {s: int(d.y[:, k].sum()) for k, s in enumerate(d.shorts)}
    out["prevalence_all"] = {s: float(d.y[:, k].mean()) for k, s in enumerate(d.shorts)}
    out["reviewed_smells_per_sample"] = {
        "+".join(s for k, s in enumerate(d.shorts) if row[k] > 0): int(c)
        for row, c in Counter(map(tuple, (d.reviews > 0).astype(int))).items()
    }

    co = d.y.T @ d.y
    out["cooccurrence_counts"] = {"labels": d.shorts, "matrix": co.tolist()}
    pairs = {}
    for a in range(len(d.shorts)):
        for b in range(a + 1, len(d.shorts)):
            both = (d.reviews[:, a] > 0) & (d.reviews[:, b] > 0)
            ya, yb = d.y[both, a], d.y[both, b]
            n11 = int((ya & yb).sum())
            entry = {"reviewed_both": int(both.sum()), "both_positive": n11}
            if both.any() and ya.std() > 0 and yb.std() > 0:
                entry["phi"] = float(np.corrcoef(ya, yb)[0, 1])
                entry[f"p_{d.shorts[b]}_given_{d.shorts[a]}"] = n11 / int(ya.sum())
                entry[f"p_{d.shorts[a]}_given_{d.shorts[b]}"] = n11 / int(yb.sum())
            pairs[f"{d.shorts[a]}-{d.shorts[b]}"] = entry
    out["pairs"] = pairs

    plt = figs.plt
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    vals, labels, texts = [], [], []
    for k, s in enumerate(d.shorts):
        e = out["by_level"][d.smell_level[k]][s]
        vals.append(100 * e["prevalence_among_reviewed"])
        labels.append(f"{d.smells[k]}\n({d.smell_level[k]} level)")
        texts.append(f"{e['positives']} / {e['reviewed']}")
    bars = ax.bar(range(len(vals)), vals, color=SMELL_COLORS, width=0.6)
    for bar, t in zip(bars, texts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, t, ha="center", va="bottom",
                fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(range(len(vals)), labels)
    ax.set_ylabel("positives among reviewed samples (%)")
    ax.set_title("MLCQ label prevalence (normalized labels)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(vals) * 1.25)
    figs.save(fig, "prevalence")

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(co, cmap=_sequential(plt), vmin=0)
    for i in range(len(d.shorts)):
        for j in range(len(d.shorts)):
            ax.text(j, i, str(int(co[i, j])), ha="center", va="center", fontsize=8,
                    color="white" if co[i, j] > co.max() * 0.55 else INK)
    names = [s.upper() if len(s) == 2 else s.capitalize() for s in d.shorts]
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    ax.set_title("Samples positive for both labels")
    fig.colorbar(im, ax=ax, shrink=0.8)
    figs.save(fig, "cooccurrence")
    return out


def _sequential(plt: Any) -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])


def severity(d: Data, figs: Figures) -> dict[str, Any]:
    out: dict[str, Any] = {}
    order = d.order
    pos_idx = [i for i, v in enumerate(order) if v in d.positive]
    for k, s in enumerate(d.shorts):
        reviewed = d.reviews[:, k] > 0
        sev_stored = Counter(r[f"severity_{s}"] for r, ok in zip(d.meta, reviewed) if ok)
        v = d.votes[reviewed, k]
        m = v.sum(axis=1)
        present_votes = v[:, pos_idx].sum(axis=1)
        y = d.y[reviewed, k].astype(bool)
        y_raw = d.y_raw[reviewed, k].astype(bool)
        minor = order.index("minor") if "minor" in order else None
        above_minor = v[:, minor + 1:].sum(axis=1) if minor is not None else np.zeros(len(v))
        upper = np.array([_upper_median(row) for row in v])
        rules = {
            "stored_labels": int(y.sum()),
            "upper_median_default": int(np.isin(upper, pos_idx).sum()),
            "upper_median_major_or_critical": int((upper >= order.index("major")).sum()),
            "upper_median_critical": int((upper == order.index("critical")).sum()),
            "strict_majority_present": int((2 * present_votes > m).sum()),
            "any_reviewer_present": int((present_votes > 0).sum()),
        }
        out[s] = {
            "reviewed": int(reviewed.sum()),
            "aggregated_severity": {sev: int(sev_stored.get(sev, 0)) for sev in order},
            "raw_votes": {sev: int(v[:, i].sum()) for i, sev in enumerate(order)},
            "positives": int(y.sum()),
            "positives_aggregated_minor": int(sum(1 for r, ok in zip(d.meta, reviewed)
                                                  if ok and int(r[f"y_{s}"]) and r[f"severity_{s}"] == "minor")),
            "positives_minor_votes_only": int((y & (present_votes > 0) & (above_minor == 0)).sum()),
            "positives_without_strict_majority": int((y & (2 * present_votes <= m)).sum()),
            # positives without a strict majority: upper median ties or unsupported labels
            "positives_upper_median_tie": int((y & y_raw & (2 * present_votes == m)).sum()),
            "positives_not_supported_by_raw_rule": int((y & ~y_raw).sum()),
            "positives_by_rule": rules,
        }
        p = out[s]["positives"]
        for key in ("positives_aggregated_minor", "positives_minor_votes_only", "positives_without_strict_majority",
                    "positives_upper_median_tie", "positives_not_supported_by_raw_rule"):
            out[s][key.replace("positives_", "share_")] = out[s][key] / p if p else None

    plt = figs.plt
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.6), sharey=True)
    names = [f"{d.smells[k]}" for k in range(len(d.shorts))]
    for ax, key, title in ((axes[0], "aggregated_severity", "Aggregated severity per sample"),
                           (axes[1], "raw_votes", "Individual reviews")):
        left = np.zeros(len(d.shorts))
        for i, sev in enumerate(order):
            vals = np.array([out[s][key][sev] / sum(out[s][key].values()) * 100 for s in d.shorts])
            ax.barh(range(len(d.shorts)), vals, left=left, color=SEVERITY_COLORS[i], label=sev,
                    height=0.62, edgecolor="white", linewidth=1.0)
            for j, (lft, w) in enumerate(zip(left, vals)):
                if w >= 7:
                    ax.text(lft + w / 2, j, f"{w:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if i >= 2 else INK)
            left += vals
        ax.set_title(title)
        ax.set_xlim(0, 100)
        ax.set_xlabel("share (%)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(names)), names)
    axes[0].invert_yaxis()
    axes[1].legend(ncols=4, loc="upper center", bbox_to_anchor=(-0.05, -0.28))
    figs.save(fig, "severity")

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    rule_keys = ["any_reviewer_present", "stored_labels", "upper_median_default", "strict_majority_present",
                 "upper_median_major_or_critical", "upper_median_critical"]
    rule_names = ["any\nreviewer", "stored\nlabels", "upper\nmedian", "strict\nmajority", "median\n>= major",
                  "median\n= critical"]
    width = 0.8 / len(d.shorts)
    for k, s in enumerate(d.shorts):
        vals = [out[s]["positives_by_rule"][r] for r in rule_keys]
        ax.bar(np.arange(len(rule_keys)) + (k - (len(d.shorts) - 1) / 2) * width, vals, width=width * 0.9,
               color=SMELL_COLORS[k], label=d.smells[k])
    ax.set_xticks(range(len(rule_keys)), rule_names, fontsize=8)
    ax.set_ylabel("positive samples")
    ax.set_title("Positives per aggregation rule (raw reviews, except the stored labels)")
    ax.legend(ncols=2, loc="upper right")
    ax.grid(axis="x", visible=False)
    figs.save(fig, "label_rules")
    return out


def _upper_median(counts: np.ndarray) -> int:
    values = np.repeat(np.arange(len(counts)), counts)
    return int(values[len(values) // 2]) if len(values) else -1


def agreement(d: Data, figs: Figures, rng: np.random.Generator) -> dict[str, Any]:
    acfg = d.acfg
    n_boot, level = int(acfg["alpha_bootstrap"]), float(acfg["alpha_ci"])
    pos_idx = [i for i, v in enumerate(d.order) if v in d.positive]
    out: dict[str, Any] = {"reviews_per_pair": {}, "alpha": {}}
    pooled_presence: list[list[int]] = []
    pooled_severity: list[list[int]] = []
    for k, s in enumerate(d.shorts):
        reviewed = d.reviews[:, k] > 0
        out["reviews_per_pair"][s] = {str(c): int(v) for c, v in sorted(Counter(d.reviews[reviewed, k]).items())}
        multi = d.reviews[:, k] >= 2
        sev_units = [list(np.repeat(np.arange(len(d.order)), row)) for row in d.votes[multi, k]]
        pres_units = [[int(v in pos_idx) for v in u] for u in sev_units]
        pooled_presence += pres_units
        pooled_severity += sev_units
        unanimous = [len(set(u)) == 1 for u in pres_units]
        out["alpha"][s] = {
            "units_multi_review": len(sev_units),
            "share_of_reviewed_units": len(sev_units) / int(reviewed.sum()),
            "unanimous_presence_share": float(np.mean(unanimous)) if unanimous else None,
            "nominal_presence": alpha_with_ci(pres_units, 2, "nominal", n_boot, level, rng),
            "ordinal_severity": alpha_with_ci(sev_units, len(d.order), "ordinal", n_boot, level, rng),
        }
    out["alpha"]["all"] = {
        "units_multi_review": len(pooled_severity),
        "nominal_presence": alpha_with_ci(pooled_presence, 2, "nominal", n_boot, level, rng),
        "ordinal_severity": alpha_with_ci(pooled_severity, len(d.order), "ordinal", n_boot, level, rng),
    }
    out["method"] = ("Krippendorff (2011) coincidence-matrix alpha on (sample, smell) units with >= 2 raw "
                     f"reviews; {int(level * 100)}% percentile bootstrap over units, {n_boot} resamples")

    plt = figs.plt
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.55})
    ax = axes[0]
    max_r = max(int(c) for s in d.shorts for c in out["reviews_per_pair"][s])
    xs = np.arange(1, max_r + 1)
    width = 0.8 / len(d.shorts)
    for k, s in enumerate(d.shorts):
        counts = [out["reviews_per_pair"][s].get(str(x), 0) for x in xs]
        ax.bar(xs + (k - (len(d.shorts) - 1) / 2) * width, counts, width=width * 0.9, color=SMELL_COLORS[k],
               label=d.smells[k])
    ax.set_xticks(xs)
    ax.set_xlabel("reviews per (sample, smell)")
    ax.set_ylabel("pairs")
    ax.set_title("Reviews per labelled pair")
    ax.legend()
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    groups = d.shorts + ["all"]
    for j, (key, marker, name) in enumerate((("nominal_presence", "o", "nominal, presence"),
                                              ("ordinal_severity", "s", "ordinal, severity"))):
        for g_i, g in enumerate(groups):
            a = out["alpha"][g][key]
            color = SMELL_COLORS[g_i] if g_i < len(d.shorts) else INK_SECONDARY
            y = g_i + (j - 0.5) * 0.3
            ax.errorbar(a["alpha"], y, xerr=[[a["alpha"] - a["ci_low"]], [a["ci_high"] - a["alpha"]]],
                        fmt=marker, color=color, markersize=6, capsize=2, linewidth=1.2,
                        markerfacecolor=color if j == 0 else "white", markeredgewidth=1.4)
    ax.axvline(0, color=INK_SECONDARY, linewidth=0.8)
    reference = acfg.get("alpha_reference")
    if reference is not None:
        ax.axvline(float(reference["value"]), color=NEUTRAL, linewidth=0.8, linestyle="--")
    ax.set_xlim(-0.2, 1.0)
    ax.set_yticks(range(len(groups)), [d.smells[k] for k in range(len(d.shorts))] + ["all smells"])
    ax.invert_yaxis()
    ax.set_xlabel(f"Krippendorff alpha ({int(level * 100)}% CI)")
    ax.set_title("Inter-rater agreement")
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", color=INK_SECONDARY, linestyle="", label="nominal, presence"),
               Line2D([], [], marker="s", color=INK_SECONDARY, markerfacecolor="white", linestyle="",
                      label="ordinal, severity")]
    if reference is not None:
        handles.append(Line2D([], [], color=NEUTRAL, linewidth=0.8, linestyle="--",
                              label=f"{float(reference['value']):.3f}, {reference['label']}"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncols=len(handles), fontsize=7)
    ax.grid(axis="y", visible=False)
    figs.save(fig, "agreement")
    return out


def projects(d: Data, figs: Figures) -> dict[str, Any]:
    top_k, in_fig = int(d.acfg["top_k_repos"]), int(d.acfg["repos_in_figure"])
    repos = sorted(set(d.repo))
    samples = Counter(d.repo)
    pos = {s: Counter(d.repo[d.y[:, k] == 1]) for k, s in enumerate(d.shorts)}
    any_pos = Counter(d.repo[d.y.sum(axis=1) > 0])
    ranked = sorted(repos, key=lambda r: (-samples[r], r))
    out: dict[str, Any] = {"repositories": len(repos)}

    def concentration(counter: Counter) -> dict[str, Any]:
        vals = [counter.get(r, 0) for r in repos]
        total = sum(vals)
        top = sorted(vals, reverse=True)[:top_k]
        nonzero = sum(1 for v in vals if v > 0)
        top_repos = sorted(repos, key=lambda r: (-counter.get(r, 0), r))[:top_k]
        return {"total": total, "repos_with_any": nonzero, f"top{top_k}_share": sum(top) / total if total else None,
                f"top{top_k}": {r: counter.get(r, 0) for r in top_repos}, "gini": gini(vals)}

    out["samples"] = concentration(samples)
    out["positives_any"] = concentration(any_pos)
    out["positives"] = {s: concentration(pos[s]) for s in d.shorts}
    out["per_repo"] = {r: {"samples": samples[r], **{s: pos[s].get(r, 0) for s in d.shorts}} for r in ranked}

    plt = figs.plt
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.75})
    ax = axes[0]
    x, y = lorenz([samples[r] for r in repos])
    ax.plot(x, y, color=INK_SECONDARY, linewidth=2, label=f"samples (Gini {out['samples']['gini']:.2f})")
    for k, s in enumerate(d.shorts):
        x, y = lorenz([pos[s].get(r, 0) for r in repos])
        ax.plot(x, y, color=SMELL_COLORS[k], linewidth=2,
                label=f"{d.smells[k]} positives ({out['positives'][s]['gini']:.2f})")
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=0.8, linestyle="--")
    ax.set_xlabel("share of repositories (ascending)")
    ax.set_ylabel("cumulative share")
    ax.set_title(f"Lorenz curves over {len(repos)} repositories")
    ax.legend(loc="upper left", fontsize=7)
    ax = axes[1]
    shown = ranked[:in_fig]
    neg = [samples[r] - any_pos.get(r, 0) for r in shown]
    posv = [any_pos.get(r, 0) for r in shown]
    ys = np.arange(len(shown))
    ax.barh(ys, posv, color=SMELL_COLORS[0], height=0.7, label="positive for at least one smell")
    ax.barh(ys, neg, left=posv, color=NEUTRAL, height=0.7, label="no smell")
    ax.set_yticks(ys, shown, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("samples")
    ax.set_title(f"Top {in_fig} repositories by samples")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(axis="y", visible=False)
    figs.save(fig, "projects")
    return out


def size_shortcut(d: Data, figs: Figures) -> dict[str, Any]:
    out: dict[str, Any] = {"features": d.size_cols, "eligible": {}, "all": {}}
    auc_e = np.full((len(d.size_cols), len(d.shorts)), np.nan)
    for k, s in enumerate(d.shorts):
        for scope, mask in (("eligible", d.eligible[:, k]), ("all", np.ones(d.n, dtype=bool))):
            y = d.y[mask, k]
            entry = {"samples": int(mask.sum()), "positives": int(y.sum()), "auc": {}}
            for f, col in enumerate(d.size_cols):
                a = safe_auc(y, d.size[mask, f])
                entry["auc"][col] = a
                if scope == "eligible" and a is not None:
                    auc_e[f, k] = a
            valid = {c: a for c, a in entry["auc"].items() if a is not None}
            best = max(valid, key=lambda c: abs(valid[c] - 0.5)) if valid else None
            entry["best_feature"] = best
            entry["best_auc"] = valid.get(best) if best else None
            out[scope][s] = entry
    out["eligible_definition"] = {s: lv for s, lv in zip(d.shorts, d.smell_level)}
    # positives and negatives by size bucket of the eligible samples
    loc = d.size[:, d.size_cols.index("loc_nonblank")] if "loc_nonblank" in d.size_cols else None
    if loc is not None:
        out["median_loc_nonblank"] = {
            s: {"positive": float(np.median(loc[d.eligible[:, k] & (d.y[:, k] == 1)])),
                "negative": float(np.median(loc[d.eligible[:, k] & (d.y[:, k] == 0)]))}
            for k, s in enumerate(d.shorts)}

    plt = figs.plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    im = ax.imshow(auc_e, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    for f in range(len(d.size_cols)):
        for k in range(len(d.shorts)):
            v = auc_e[f, k]
            if not np.isnan(v):
                ax.text(k, f, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(v - 0.5) > 0.3 else INK)
    ax.set_xticks(range(len(d.shorts)), [d.smells[k].replace(" ", "\n") + f"\n({d.smell_level[k]})"
                                         for k in range(len(d.shorts))], fontsize=8)
    ax.set_yticks(range(len(d.size_cols)), d.size_cols, fontsize=8)
    ax.grid(False)
    ax.set_title("ROC-AUC of one size feature, eligible samples")
    fig.colorbar(im, ax=ax, shrink=0.7, label="ROC-AUC (0.5 = chance)")
    figs.save(fig, "size_auc")

    if loc is not None:
        fig, axes = plt.subplots(1, len(d.shorts), figsize=(8.4, 2.6), sharey=False,
                                 gridspec_kw={"wspace": 0.45})
        for k, ax in enumerate(axes):
            mask = d.eligible[:, k]
            data = [np.maximum(loc[mask & (d.y[:, k] == v)], 1) for v in (0, 1)]
            parts = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                               medianprops={"color": INK, "linewidth": 1.2})
            for patch, color in zip(parts["boxes"], (NEUTRAL, SMELL_COLORS[k])):
                patch.set_facecolor(color)
                patch.set_edgecolor(INK_SECONDARY)
            ax.set_yscale("log")
            ax.set_xticks([1, 2], ["absent", "present"])
            ax.set_title(f"{d.smells[k]}\nAUC {out['eligible'][d.shorts[k]]['auc']['loc_nonblank']:.2f}", fontsize=9)
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("non-blank lines (log)")
        fig.suptitle("Snippet length by label, eligible samples", fontsize=10, y=1.13)
        figs.save(fig, "size_by_label")
    return out


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def groups(self) -> list[list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return [g for g in out.values() if len(g) > 1]


def near_duplicate_pairs(token_sets: list[set[str]], threshold: float, block: int) -> list[tuple[int, int, float]]:
    """Exact all-pairs token set Jaccard via blocked sparse products."""
    vocab: dict[str, int] = {}
    rows, cols = [], []
    for i, toks in enumerate(token_sets):
        for t in toks:
            rows.append(i)
            cols.append(vocab.setdefault(t, len(vocab)))
    n = len(token_sets)
    x = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, len(vocab)))
    sizes = np.asarray(x.sum(axis=1)).ravel()
    xt = x.T.tocsc()
    pairs = []
    for a in range(0, n, block):
        b = min(n, a + block)
        inter = (x[a:b] @ xt).toarray()
        union = sizes[a:b, None] + sizes[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            jac = np.where(union > 0, inter / union, 0.0)
        ii, jj = np.nonzero(jac >= threshold)
        for i, j in zip(ii + a, jj):
            if j > i:
                pairs.append((int(i), int(j), float(jac[i - a, j])))
    return pairs


def duplicates(d: Data, figs: Figures, entries: list[dict[str, Any]], folds: dict[str, Any]) -> dict[str, Any]:
    dcfg = d.acfg["duplicates"]
    pattern = re.compile(str(dcfg["token_pattern"]))
    threshold = float(dcfg["jaccard_threshold"])
    texts = [str(e.get("code_snippet", "")) for e in entries]
    normalized = [" ".join(t.split()) for t in texts]

    exact_uf = UnionFind(d.n)
    first: dict[str, int] = {}
    for i, t in enumerate(normalized):
        if t in first:
            exact_uf.union(first[t], i)
        else:
            first[t] = i
    token_sets = [set(pattern.findall(t)) for t in texts]
    pairs = near_duplicate_pairs(token_sets, threshold, int(dcfg["block_rows"]))
    near_uf = UnionFind(d.n)
    neighbors: dict[int, set[int]] = defaultdict(set)
    for i, j, _ in pairs:
        near_uf.union(i, j)
        neighbors[i].add(j)
        neighbors[j].add(i)
    exact_neighbors: dict[int, set[int]] = defaultdict(set)
    for g in exact_uf.groups():
        for i in g:
            exact_neighbors[i].update(set(g) - {i})

    def describe(groups: list[list[int]]) -> dict[str, Any]:
        conflicts = {s: 0 for s in d.shorts}
        any_conflict = 0
        for g in groups:
            hit = False
            for k, s in enumerate(d.shorts):
                vals = {int(d.y[i, k]) for i in g if d.reviews[i, k] > 0}
                if len(vals) > 1:
                    conflicts[s] += 1
                    hit = True
            any_conflict += hit
        return {
            "groups": len(groups),
            "samples_in_groups": sum(len(g) for g in groups),
            "redundant_samples": sum(len(g) - 1 for g in groups),
            "group_sizes": {str(k): v for k, v in sorted(Counter(len(g) for g in groups).items())},
            "groups_mixing_levels": sum(1 for g in groups if len({d.level[i] for i in g}) > 1),
            "groups_spanning_repos": sum(1 for g in groups if len({d.repo[i] for i in g}) > 1),
            "groups_same_file_path": sum(1 for g in groups if len({entries[i]["file_path"] for i in g}) == 1),
            "groups_with_label_conflict": any_conflict,
            "conflicts_per_smell": conflicts,
            "positives_in_groups": {s: int(sum(d.y[i, k] for g in groups for i in g)) for k, s in enumerate(d.shorts)},
        }

    exact_groups, near_groups = exact_uf.groups(), near_uf.groups()
    out: dict[str, Any] = {
        "method": {
            "exact": "identical snippet text after collapsing every whitespace run to one space",
            "near": (f"token set Jaccard >= {threshold}, tokens from the regular expression in the config; "
                     "every pair is scored exactly (blocked sparse matrix products, no hashing), groups are "
                     "the connected components of the pairs above the threshold"),
        },
        "exact": describe(exact_groups),
        "near": {**describe(near_groups), "pairs": len(pairs),
                 "pairs_conflicting": {s: sum(1 for i, j, _ in pairs if d.reviews[i, k] > 0 and d.reviews[j, k] > 0
                                              and d.y[i, k] != d.y[j, k]) for k, s in enumerate(d.shorts)}},
    }

    leakage: dict[str, Any] = {}
    for strategy in dcfg["leakage_strategies"]:
        per_split = []
        for split in folds["splits"]:
            if split["strategy"] != strategy:
                continue
            train = set(split["train"])
            test = split["test"]
            near_hit = [i for i in test if neighbors.get(i, set()) & train]
            exact_hit = [i for i in test if exact_neighbors.get(i, set()) & train]
            same_label = [i for i in near_hit
                          if any((d.y[j] == d.y[i]).all() for j in neighbors[i] & train)]
            per_split.append({"test": len(test), "near": len(near_hit), "exact": len(exact_hit),
                              "near_same_labels": len(same_label),
                              "near_positive": int(sum(d.y[i].any() for i in near_hit))})
        if per_split:
            leakage[strategy] = {
                "splits": len(per_split),
                "mean_test_share_with_near_duplicate_in_train": float(np.mean([p["near"] / p["test"] for p in per_split])),
                "mean_test_share_with_exact_duplicate_in_train": float(np.mean([p["exact"] / p["test"] for p in per_split])),
                "mean_near_test_samples": float(np.mean([p["near"] for p in per_split])),
                "mean_near_with_identical_label_vector": float(np.mean([p["near_same_labels"] for p in per_split])),
                "mean_near_positive_any_smell": float(np.mean([p["near_positive"] for p in per_split])),
            }
    out["train_test_leakage"] = leakage

    plt = figs.plt
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6))
    ax = axes[0]
    for j, (name, key) in enumerate((("exact", "exact"), (f"Jaccard >= {threshold}", "near"))):
        sizes = out[key]["group_sizes"]
        xs = sorted(int(k) for k in sizes)
        ax.bar(np.array(xs) + (j - 0.5) * 0.38, [sizes[str(x)] for x in xs], width=0.36,
               color=[SMELL_COLORS[0], SMELL_COLORS[1]][j], label=name)
    ax.set_xlabel("group size (samples)")
    ax.set_ylabel("groups")
    ax.set_title("Duplicate groups")
    ax.legend()
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    strategies = list(leakage)
    vals_near = [100 * leakage[s]["mean_test_share_with_near_duplicate_in_train"] for s in strategies]
    vals_exact = [100 * leakage[s]["mean_test_share_with_exact_duplicate_in_train"] for s in strategies]
    xs = np.arange(len(strategies))
    ax.bar(xs - 0.19, vals_exact, width=0.36, color=SMELL_COLORS[0], label="exact")
    ax.bar(xs + 0.19, vals_near, width=0.36, color=SMELL_COLORS[1], label=f"Jaccard >= {threshold}")
    for x, v in zip(list(xs - 0.19) + list(xs + 0.19), vals_exact + vals_near):
        ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)
    ax.set_xticks(xs, strategies)
    ax.set_ylabel("test samples (%)")
    ax.set_title("Test samples with a duplicate in train")
    ax.legend()
    ax.grid(axis="x", visible=False)
    figs.save(fig, "duplicates")
    return out


MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def shortcut_figure(scfg: dict[str, Any], figs: Figures) -> dict[str, Any]:
    """Grouped bars of seed-mean MCC (std over seeds) read from summarize_cv outputs."""
    panels = []
    for panel in scfg["panels"]:
        path = resolve(panel["summary"])
        if not path.exists():
            print(f"[analysis] shortcut figure skipped: {path} not found", flush=True)
            return {"skipped": str(path)}
        rows = {r["group"]: r for r in json.loads(path.read_text())["rows"]}
        missing = [g["key"] for g in panel["groups"] if g["key"] not in rows]
        if missing:
            raise SystemExit(f"{path}: groups not found: {missing}")
        panels.append((panel, rows))
    metrics, names = list(scfg["metrics"]), list(scfg["metric_names"])
    labels: list[str] = []
    for panel, _ in panels:
        for g in panel["groups"]:
            if g["label"] not in labels:
                labels.append(g["label"])
    if len(labels) > len(MODEL_COLORS):
        raise SystemExit("too many models for the categorical palette")
    color = dict(zip(labels, MODEL_COLORS))

    plt = figs.plt
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.0, 2.7 * len(panels)), sharey=True)
    axes = np.atleast_1d(axes)
    out: dict[str, Any] = {}
    for ax, (panel, rows) in zip(axes, panels):
        groups = panel["groups"]
        width = 0.84 / len(groups)
        values: dict[str, Any] = {}
        for gi, g in enumerate(groups):
            row = rows[g["key"]]
            means = [row[f"{m}_mean"] for m in metrics]
            stds = [row[f"{m}_std"] for m in metrics]
            xs = np.arange(len(metrics)) + (gi - (len(groups) - 1) / 2) * width
            ax.bar(xs, means, width=width * 0.92, color=color[g["label"]], label=g["label"],
                   yerr=stds, error_kw={"elinewidth": 0.8, "capsize": 1.5, "ecolor": INK_SECONDARY})
            # a bar of height 0 is invisible: print its value instead
            for x, mean in zip(xs, means):
                if abs(mean) < 0.005:
                    ax.text(x, 0.01, f"{mean:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=90,
                            color=color[g["label"]])
            values[g["label"]] = {m: {"mean": a, "std": b} for m, a, b in zip(metrics, means, stds)}
        ax.set_xticks(range(len(metrics)), names)
        ax.set_ylabel("MCC, tuned thresholds")
        ax.set_title(panel["title"])
        ax.axhline(0, color=INK_SECONDARY, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        out[panel["title"]] = values
    handles = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(list(handles.values()), list(handles.keys()), ncols=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.02 - 0.03 * math.ceil(len(handles) / 2)), fontsize=8)
    fig.tight_layout()
    figs.save(fig, str(scfg["name"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config/experiments/mlcq_analysis.yml"))
    parser.add_argument("--self-test", action="store_true", help="check the Krippendorff alpha code and exit")
    args, extra = parser.parse_known_args()
    if args.self_test:
        self_test()
        return
    cfg = load_config(resolve(args.config), parse_cli_overrides(extra))
    acfg = cfg["analysis"]
    rng = np.random.default_rng(int(acfg["seed"]))
    d = Data(cfg)
    figs = Figures(resolve(acfg["figures_dir"]), str(acfg["figure_prefix"]))
    entries = json.loads(resolve(cfg["inputs"]["normalized_json"]).read_text())
    folds = json.loads(resolve(cfg["inputs"]["folds"]).read_text())
    check_path = resolve(cfg["sample_meta"]["check_out"])
    check = json.loads(check_path.read_text()) if check_path.exists() else {}

    summary: dict[str, Any] = {
        "config": str(args.config),
        "label_rule_check": {k: v for k, v in check.items() if k != "mismatches"},
    }
    summary["prevalence"] = prevalence(d, figs)
    print("[analysis] (a) prevalence done", flush=True)
    summary["severity"] = severity(d, figs)
    print("[analysis] (b) severity done", flush=True)
    summary["agreement"] = agreement(d, figs, rng)
    print("[analysis] (c) agreement done", flush=True)
    summary["projects"] = projects(d, figs)
    print("[analysis] (d) projects done", flush=True)
    summary["size_shortcut"] = size_shortcut(d, figs)
    print("[analysis] (e) size shortcut done", flush=True)
    summary["duplicates"] = duplicates(d, figs, entries, folds)
    print("[analysis] (f) duplicates done", flush=True)
    if acfg.get("shortcut_figure"):
        summary["shortcut_figure"] = shortcut_figure(acfg["shortcut_figure"], figs)
    summary["figures"] = figs.written

    out_dir = resolve(acfg["out_dir"])
    atomic_json(out_dir / "summary.json", summary)
    print(f"[analysis] wrote {out_dir / 'summary.json'} and {len(figs.written)} figure files")


if __name__ == "__main__":
    main()
