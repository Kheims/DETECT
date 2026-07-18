"""Compare base and metamorphic variant predictions of each model family.

Reports per family, smell and transform the logit shift, flip rate at the
frozen base threshold and change of MCC and ROC-AUC with bootstrap intervals.
Writes summary.csv, summary.json and flip and score shift figures.

Usage:
    python scripts/analysis/analyze_metamorphic.py --config config/experiments/metamorphic.yml
    python scripts/analysis/analyze_metamorphic.py --config ... --analysis.bootstrap.n_boot=200
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import textwrap
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cv.eval_eligible import FOLD_KEYS, STRATEGY_KEYS  # noqa: E402
from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.evaluation.cv import read_jsonl, tune_thresholds  # noqa: E402
from mlcq_graphs.metamorphic import transform_classes  # noqa: E402

Key = tuple[str, int, int]  # strategy, seed, fold

INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
SEQUENTIAL = ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"]
MISSING_CELL = "#f0efec"
POS_COLOR, NEG_COLOR = "#2a78d6", "#eb6834"

COLUMNS = [
    "family", "dataset", "strategy", "smell", "transform", "transform_class", "n_folds", "n_test_rows",
    "applicability", "n_rows", "n_samples", "n_missing", "n_unapplied_differ",
    "mean_dlogit", "mean_dlogit_lo", "mean_dlogit_hi", "median_dlogit", "mean_dlogit_pos", "mean_dlogit_neg",
    "flip_rate", "flip_rate_lo", "flip_rate_hi", "n_pos_rows", "flip_rate_pos", "flip_rate_pos_lo",
    "flip_rate_pos_hi", "n_neg_rows", "flip_rate_neg", "flip_rate_neg_lo", "flip_rate_neg_hi", "flip_up", "flip_down",
    "n_c2w", "c2w_rate", "c2w_rate_lo", "c2w_rate_hi", "n_w2c", "w2c_rate", "w2c_rate_lo", "w2c_rate_hi",
    "mcc_base", "mcc_variant", "delta_mcc", "delta_mcc_lo", "delta_mcc_hi",
    "auc_base", "auc_variant", "delta_auc", "delta_auc_lo", "delta_auc_hi",
]


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def expand(patterns: list[Any]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        text = str(resolve(pattern))
        files += [Path(f) for f in sorted(glob.glob(text))] if any(c in text for c in "*?[") else [Path(text)]
    return [f for f in files if f.exists()]


@dataclass
class FoldData:
    key: Key
    idx: np.ndarray  # test json indices, sorted
    y: np.ndarray  # (n, labels)
    p: np.ndarray  # base scores (n, labels)
    thr: np.ndarray  # (labels,)


def load_rows(files: list[Path], parts: tuple[str, ...],
              model: str | None = None) -> dict[Key, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]]:
    """Rows by (strategy, seed, fold), part and json index. A duplicate row is an error."""
    out: dict[Key, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]] = defaultdict(lambda: defaultdict(dict))
    for file in files:
        for row in read_jsonl(file):
            if row["part"] not in parts or (model is not None and str(row.get("model")) != model):
                continue
            key = (str(row["strategy"]), int(row["seed"]), int(row["fold"]))
            j = int(row["json_index"])
            if j in out[key][row["part"]]:
                raise SystemExit(f"{file}: json_index {j} twice in {key} {row['part']}; set the family's model")
            out[key][row["part"]][j] = (np.asarray(row["y"], dtype=np.int64),
                                        np.asarray(row["prob"], dtype=np.float64))
    return out


def load_thresholds(files: list[Path], labels: list[str]) -> dict[Key, np.ndarray]:
    """Per-fold thresholds of records.json (a list) or record.json (one record) files."""
    out: dict[Key, np.ndarray] = {}
    for file in files:
        data = json.loads(file.read_text())
        for rec in data if isinstance(data, list) else [data]:
            combo = rec.get("combo", {})
            fold = next((combo[k] for k in FOLD_KEYS if k in combo), None)
            strategy = next((combo[k] for k in STRATEGY_KEYS if k in combo), None)
            if fold is None or strategy is None or "seed" not in rec:
                continue
            thr = rec.get("thresholds")
            if thr is None:
                per = rec.get("per_label_tuned", {})
                thr = [per.get(label, {}).get("threshold") for label in labels]
            if thr is None or len(thr) != len(labels) or any(t is None for t in thr):
                continue
            out[(str(strategy), int(rec["seed"]), int(fold))] = np.asarray(thr, dtype=np.float64)
    return out


def family_folds(spec: dict[str, Any], labels: list[str], strategies: list[str],
                 steps: int) -> tuple[list[FoldData], list[Path], dict[str, int]]:
    files = expand(spec["predictions"])
    if not files:
        return [], [], {}
    rows = load_rows(files, ("val", "test"), spec.get("model"))
    dirs = sorted({f.parent for f in files})
    if spec.get("records"):
        rec_files = expand(spec["records"])
    else:
        rec_files = [d / n for d in dirs for n in ("records.json", "record.json") if (d / n).exists()]
    thresholds = load_thresholds(rec_files, labels)
    folds, sources = [], Counter()
    for key in sorted(rows):
        test = rows[key].get("test")
        if key[0] not in strategies or not test:
            continue
        idx = np.asarray(sorted(test), dtype=np.int64)
        y = np.stack([test[int(i)][0] for i in idx])
        p = np.stack([test[int(i)][1] for i in idx])
        if y.shape[1] != len(labels):
            raise SystemExit(f"{spec['predictions']}: {y.shape[1]} labels, the dataset has {len(labels)}")
        if key in thresholds:
            thr = thresholds[key]
            sources["records"] += 1
        elif rows[key].get("val"):
            val = rows[key]["val"]
            order = sorted(val)
            thr = tune_thresholds(np.stack([val[i][0] for i in order]), np.stack([val[i][1] for i in order]), steps)
            thr = thr.astype(np.float64)
            sources["tuned_on_val"] += 1
        else:
            sources["no_threshold"] += 1
            continue
        folds.append(FoldData(key, idx, y, p, thr))
    return folds, dirs, dict(sources)


def load_variant(dirs: list[Path], pattern: str, transform: str,
                 model: str | None) -> tuple[dict[Key, dict[int, tuple]], int] | None:
    files = sorted({f for d in dirs for f in d.glob(pattern.format(transform=transform))})
    if not files:
        return None
    rows = load_rows(files, ("test",), model)
    return {k: v["test"] for k, v in rows.items()}, len(files)


def applied_flags(path: Path) -> np.ndarray:
    entries = json.loads(path.read_text())
    return np.asarray([bool(e.get("metamorphic", {}).get("applied", False)) for e in entries], dtype=bool)


def logit(p: np.ndarray, clip: float) -> np.ndarray:
    q = np.clip(p, clip, 1.0 - clip)
    return np.log(q) - np.log1p(-q)


def weighted_mcc(w: np.ndarray, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """MCC per row of sample weights w (R, n), NaN without positives."""
    y, pred = y.astype(bool), pred.astype(bool)
    tp = w @ (y & pred).astype(np.float64)
    fp = w @ (~y & pred).astype(np.float64)
    fn = w @ (y & ~pred).astype(np.float64)
    tn = w @ (~y & ~pred).astype(np.float64)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1.0), 0.0)
    return np.where(tp + fn > 0, mcc, np.nan)


def weighted_auc(w: np.ndarray, y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """ROC-AUC (Mann-Whitney, ties count one half) per row of sample weights w (R, n), NaN with one class."""
    order = np.argsort(s, kind="mergesort")
    ss, ys = s[order], y[order].astype(bool)
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    wo = w[:, order]
    pos = np.add.reduceat(wo * ys, starts, axis=1)
    neg = np.add.reduceat(wo * ~ys, starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    num = (pos * (below + 0.5 * neg)).sum(axis=1)
    den = pos.sum(axis=1) * neg.sum(axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def protocol_mean(values: np.ndarray, seeds: list[int]) -> np.ndarray:
    """values (folds, R): mean over the folds of each seed, then over the seeds (NaN folds left out)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        per_seed = [np.nanmean(values[[i for i, s in enumerate(seeds) if s == seed]], axis=0)
                    for seed in sorted(set(seeds))]
        return np.nanmean(np.stack(per_seed), axis=0)


def interval(values: np.ndarray, ci: float) -> tuple[float, float]:
    v = values[np.isfinite(values)]
    if not len(v):
        return float("nan"), float("nan")
    a = (1.0 - ci) / 2.0
    return float(np.quantile(v, a)), float(np.quantile(v, 1.0 - a))


def ratio_replicates(W: np.ndarray, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    top, bottom = W @ num, W @ den
    return np.where(bottom > 0, top / np.where(bottom > 0, bottom, 1.0), np.nan)


def rounded(x: Any, digits: int = 4) -> Any:
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), digits)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def analyze_family(name: str, spec: dict[str, Any], cfg: dict[str, Any], class_of: dict[str, str],
                   flags: dict[tuple[str, str], np.ndarray | None]) -> tuple[list[dict], dict, dict]:
    acfg = cfg["analysis"]
    dataset = str(spec["dataset"])
    labels = list(cfg["datasets"][dataset]["label_order"])
    bcfg = acfg["bootstrap"]
    n_boot, ci, clip = int(bcfg["n_boot"]), float(bcfg["ci"]), float(acfg["clip"])
    strategies = list(acfg["strategies"])
    folds, dirs, sources = family_folds(spec, labels, strategies, int(acfg["threshold_steps"]))
    meta: dict[str, Any] = {"dataset": dataset, "dirs": [str(d) for d in dirs], "threshold_sources": sources,
                            "n_folds": len(folds), "variant_files": {}, "missing_transforms": []}
    if not folds:
        print(f"[{name}] no base predictions for strategies {strategies}: {spec['predictions']}")
        return [], meta, {}
    rows_out: list[dict[str, Any]] = []
    shifts: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    pattern = str(spec.get("variant_pattern", acfg["variant_pattern"]))
    for strategy in strategies:
        sfolds = [f for f in folds if f.key[0] == strategy]
        if not sfolds:
            continue
        universe = np.unique(np.concatenate([f.idx for f in sfolds]))
        rng = np.random.default_rng(int(bcfg["seed"]))
        W = rng.multinomial(len(universe), np.full(len(universe), 1.0 / len(universe)), size=n_boot).astype(np.float64)
        seeds = [f.key[1] for f in sfolds]
        n_test_rows = int(sum(len(f.idx) for f in sfolds))
        for transform, klass in class_of.items():
            applied_all = flags.get((dataset, transform))
            loaded = load_variant(dirs, pattern, transform, spec.get("model"))
            if applied_all is None or loaded is None:
                if strategy == strategies[0]:
                    meta["missing_transforms"].append(transform)
                continue
            variant, n_files = loaded
            meta["variant_files"][transform] = n_files
            L = len(labels)
            mcc = {part: np.full((len(sfolds), n_boot + 1, L), np.nan) for part in ("base", "variant")}
            auc = {part: np.full((len(sfolds), n_boot + 1, L), np.nan) for part in ("base", "variant")}
            parts: dict[str, list[np.ndarray]] = defaultdict(list)
            n_missing = n_differ = 0
            for fi, f in enumerate(sfolds):
                vrows = variant.get(f.key, {})
                app = applied_all[f.idx]
                pv = f.p.copy()
                present = np.ones(len(f.idx), dtype=bool)
                for r, j in enumerate(f.idx):
                    got = vrows.get(int(j))
                    if got is not None:
                        if not np.array_equal(got[0], f.y[r]):
                            raise SystemExit(f"[{name}] {transform} {f.key} json_index {j}: variant labels differ")
                        if app[r]:
                            pv[r] = got[1]
                        elif not np.array_equal(got[1], f.p[r]):
                            n_differ += 1
                    elif app[r]:
                        present[r] = False
                n_missing += int((~present).sum())
                cols = np.searchsorted(universe, f.idx[present])
                w = np.vstack([np.ones(len(cols)), W[:, cols]])
                y, pb, pvv, a = f.y[present], f.p[present], pv[present], app[present]
                for k in range(L):
                    mcc["base"][fi, :, k] = weighted_mcc(w, y[:, k], pb[:, k] >= f.thr[k])
                    mcc["variant"][fi, :, k] = weighted_mcc(w, y[:, k], pvv[:, k] >= f.thr[k])
                    auc["base"][fi, :, k] = weighted_auc(w, y[:, k], pb[:, k])
                    auc["variant"][fi, :, k] = weighted_auc(w, y[:, k], pvv[:, k])
                parts["col"].append(cols[a])
                parts["y"].append(y[a])
                parts["pb"].append(pb[a])
                parts["pv"].append(pvv[a])
                parts["thr"].append(np.repeat(f.thr[None, :], int(a.sum()), axis=0))
            col = np.concatenate(parts["col"])
            Y, PB, PV, THR = (np.concatenate(parts[k]) for k in ("y", "pb", "pv", "thr"))
            U = len(universe)
            n = len(col)
            for k, label in enumerate(labels):
                smell = label.removeprefix("is_")
                dl = logit(PV[:, k], clip) - logit(PB[:, k], clip)
                before, after = PB[:, k] >= THR[:, k], PV[:, k] >= THR[:, k]
                pos = Y[:, k] == 1
                flip = before != after
                c2w = flip & (before == pos)  # the base prediction was right
                w2c = flip & (after == pos)

                def sums(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
                    return np.bincount(col[mask], weights=values[mask].astype(np.float64), minlength=U)

                every = np.ones(n, dtype=bool)
                ones = np.ones(n)
                stats: dict[str, Any] = {}
                for tag, mask in (("", every), ("_pos", pos), ("_neg", ~pos)):
                    stats[f"flip_rate{tag}"] = float(flip[mask].mean()) if mask.any() else float("nan")
                    stats[f"flip_rate{tag}_lo"], stats[f"flip_rate{tag}_hi"] = interval(
                        ratio_replicates(W, sums(flip, mask), sums(ones, mask)), ci)
                for tag, hits in (("c2w", c2w), ("w2c", w2c)):
                    stats[f"n_{tag}"] = int(hits.sum())
                    stats[f"{tag}_rate"] = float(hits.mean()) if n else float("nan")
                    stats[f"{tag}_rate_lo"], stats[f"{tag}_rate_hi"] = interval(
                        ratio_replicates(W, sums(hits, every), sums(ones, every)), ci)
                lo, hi = interval(ratio_replicates(W, sums(dl, every), sums(ones, every)), ci)
                d_mcc = protocol_mean(mcc["variant"][:, :, k] - mcc["base"][:, :, k], seeds)
                d_auc = protocol_mean(auc["variant"][:, :, k] - auc["base"][:, :, k], seeds)
                row = {
                    "family": name, "dataset": dataset, "strategy": strategy, "smell": smell,
                    "transform": transform, "transform_class": klass, "n_folds": len(sfolds),
                    "n_test_rows": n_test_rows, "applicability": n / n_test_rows if n_test_rows else float("nan"),
                    "n_rows": n, "n_samples": int(len(np.unique(col))), "n_missing": n_missing,
                    "n_unapplied_differ": n_differ,
                    "mean_dlogit": float(dl.mean()) if n else float("nan"),
                    "mean_dlogit_lo": lo, "mean_dlogit_hi": hi,
                    "median_dlogit": float(np.median(dl)) if n else float("nan"),
                    "mean_dlogit_pos": float(dl[pos].mean()) if pos.any() else float("nan"),
                    "mean_dlogit_neg": float(dl[~pos].mean()) if (~pos).any() else float("nan"),
                    "n_pos_rows": int(pos.sum()), "n_neg_rows": int((~pos).sum()),
                    "flip_up": float((~before & after).mean()) if n else float("nan"),
                    "flip_down": float((before & ~after).mean()) if n else float("nan"),
                    "mcc_base": float(protocol_mean(mcc["base"][:, :1, k], seeds)[0]),
                    "mcc_variant": float(protocol_mean(mcc["variant"][:, :1, k], seeds)[0]),
                    "delta_mcc": float(d_mcc[0]),
                    "auc_base": float(protocol_mean(auc["base"][:, :1, k], seeds)[0]),
                    "auc_variant": float(protocol_mean(auc["variant"][:, :1, k], seeds)[0]),
                    "delta_auc": float(d_auc[0]),
                    **stats,
                }
                row["delta_mcc_lo"], row["delta_mcc_hi"] = interval(d_mcc[1:], ci)
                row["delta_auc_lo"], row["delta_auc_hi"] = interval(d_auc[1:], ci)
                rows_out.append({c: rounded(row.get(c)) for c in COLUMNS})
                shifts[(name, strategy, smell, transform)] = (dl, Y[:, k])
    return rows_out, meta, shifts


# ------------------------------------------------------------ figures


class Figures:
    def __init__(self, directory: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        self.plt = plt
        self.dir = directory
        self.cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
        self.written: list[str] = []
        directory.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({
            "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
            "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK, "text.color": INK,
            "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
            "axes.spines.top": False, "axes.spines.right": False,
            "legend.frameon": False, "figure.dpi": 100, "savefig.bbox": "tight",
        })

    def save(self, fig: Any, name: str) -> None:
        for ext in ("pdf", "png"):
            path = self.dir / f"{name}.{ext}"
            tmp = tmp_path(path)
            fig.savefig(tmp, format=ext, dpi=200)
            tmp.replace(path)
            self.written.append(str(path))
        self.plt.close(fig)


def heatmap(figs: Figures, rows: list[dict], families: list[str], labels: dict[str, str],
            classes: dict[str, list[str]], class_labels: dict[str, str], strategy: str, smell: str) -> None:
    plt = figs.plt
    fams = [f for f in families if any(r["family"] == f and r["smell"] == smell and r["strategy"] == strategy
                                       for r in rows)]
    if not fams:
        return
    columns: list[str | None] = []
    groups: list[tuple[str, int, int]] = []  # class, first column, last column
    for klass, names in classes.items():
        if columns:
            columns.append(None)
        groups.append((klass, len(columns), len(columns) + len(names) - 1))
        columns += names
    flip = np.full((len(fams), len(columns)), np.nan)
    shift = np.full_like(flip, np.nan)
    for r in rows:
        if r["smell"] != smell or r["strategy"] != strategy or r["family"] not in fams or r["transform"] not in columns:
            continue
        i, j = fams.index(r["family"]), columns.index(r["transform"])
        if r["flip_rate"] is not None:
            flip[i, j] = r["flip_rate"]
        if r["mean_dlogit"] is not None:
            shift[i, j] = r["mean_dlogit"]
    vmax = max(0.05, float(np.nanmax(flip)) if np.isfinite(flip).any() else 0.05)
    fig, ax = plt.subplots(figsize=(0.55 * len(columns) + 2.8, 0.42 * len(fams) + 1.9))
    cmap = figs.cmap.copy()
    cmap.set_bad(MISSING_CELL)
    im = ax.imshow(np.ma.masked_invalid(flip), cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
    for j, t in enumerate(columns):
        if t is None:
            ax.axvspan(j - 0.5, j + 0.5, color="white", zorder=2)
            continue
        for i in range(len(fams)):
            if np.isfinite(flip[i, j]):
                dark = flip[i, j] > 0.6 * vmax
                text = f"{shift[i, j]:+.2f}" if np.isfinite(shift[i, j]) else ""
                ax.text(j, i, text, ha="center", va="center", fontsize=6.5, color="white" if dark else INK)
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6, color=INK_MUTED)
    ax.set_xticks([j for j, t in enumerate(columns) if t is not None])
    ax.set_xticklabels([t for t in columns if t is not None], rotation=45, ha="right")
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([labels.get(f, f) for f in fams])
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    for klass, first, last in groups:
        title = textwrap.fill(class_labels.get(klass, klass), width=max(10, 9 * (last - first + 1)))
        ax.text((first + last) / 2, -0.7, title, ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("flip rate at the frozen threshold")
    cbar.outline.set_visible(False)
    ax.set_title(f"{smell.replace('_', ' ')} ({strategy}): flip rate, mean logit shift in the cells", pad=30)
    figs.save(fig, f"flip_{strategy}_{smell}")


def shift_figure(figs: Figures, rows: list[dict], shifts: dict, labels: dict[str, str], strategy: str, k: int,
                 min_rows: int, classes: list[str]) -> None:
    plt = figs.plt
    cands = [r for r in rows if r["strategy"] == strategy and r["n_rows"] >= min_rows and r["flip_rate"] is not None
             and r["transform_class"] in classes]
    cands.sort(key=lambda r: -r["flip_rate"])
    cands = cands[:k]
    if not cands:
        return
    ncols = min(4, len(cands))
    nrows = -(-len(cands) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.1 * nrows), squeeze=False)
    for ax, r in zip(axes.flat, cands):
        dl, y = shifts[(r["family"], strategy, r["smell"], r["transform"])]
        lo, hi = np.quantile(dl, [0.005, 0.995]) if len(dl) else (-1.0, 1.0)
        span = max(abs(lo), abs(hi), 0.5)
        bins = np.linspace(-span, span, 41)
        for mask, color, name in ((y == 0, NEG_COLOR, "negatives"), (y == 1, POS_COLOR, "positives")):
            if mask.any():
                ax.hist(np.clip(dl[mask], -span, span), bins=bins, density=True, histtype="step", linewidth=1.5,
                        color=color, label=name)
        ax.axvline(0.0, color=INK_MUTED, linewidth=0.8)
        ax.set_title(f"{labels.get(r['family'], r['family'])}\n{r['smell'].replace('_', ' ')}, {r['transform']}",
                     fontsize=8)
        # the note goes on the side away from the bulk of the shifts
        right = len(dl) > 0 and float(np.median(dl)) < 0
        ax.text(0.98 if right else 0.02, 0.97, f"flips {r['flip_rate']:.1%}\nn = {r['n_rows']}",
                transform=ax.transAxes, va="top", ha="right" if right else "left", fontsize=7, color=INK_SECONDARY)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("change of logit(score)")
    for ax in list(axes.flat)[len(cands):]:
        ax.axis("off")
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figs.save(fig, f"score_shift_{strategy}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path(path)
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows([{c: ("" if r.get(c) is None else r.get(c)) for c in COLUMNS} for r in rows])
    tmp.replace(path)


def fmt(x: Any, spec: str) -> str:
    return format(x if x is not None else float("nan"), spec)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config/experiments/metamorphic.yml"))
    args, extra = parser.parse_known_args(argv)
    cfg = load_config(resolve(args.config), parse_cli_overrides(extra))
    acfg = cfg["analysis"]
    classes = transform_classes(cfg)
    class_of = {t: c for c, ts in classes.items() for t in ts}
    out_dir = resolve(acfg["out_dir"])

    flags: dict[tuple[str, str], np.ndarray | None] = {}
    datasets = {str(spec["dataset"]) for spec in acfg["families"].values()}
    for dataset in sorted(datasets):
        for t in class_of:
            path = resolve(cfg["datasets"][dataset]["out_dir"]) / f"{t}.normalized.json"
            flags[(dataset, t)] = applied_flags(path) if path.exists() else None

    rows: list[dict[str, Any]] = []
    shifts: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    meta: dict[str, Any] = {}
    for name, spec in acfg["families"].items():
        r, m, s = analyze_family(name, spec, cfg, class_of, flags)
        rows += r
        shifts.update(s)
        meta[name] = m
        if r:
            print(f"[{name}] {m['n_folds']} folds, thresholds {m['threshold_sources']}, variant files "
                  f"{m['variant_files']}" + (f", no variant for {m['missing_transforms']}" if m["missing_transforms"]
                                             else ""))
            for row in r:
                print(f"  {row['strategy']:11} {row['smell']:22} {row['transform']:20} "
                      f"appl {fmt(row['applicability'], '.2f')} rows {row['n_rows']:>6} "
                      f"flip {fmt(row['flip_rate'], '.3f')} (pos {fmt(row['flip_rate_pos'], '.3f')}, "
                      f"neg {fmt(row['flip_rate_neg'], '.3f')}) c2w {row['n_c2w']} w2c {row['n_w2c']} "
                      f"dlogit {fmt(row['mean_dlogit'], '+.3f')} (median {fmt(row['median_dlogit'], '+.3f')}) "
                      f"dMCC {fmt(row['delta_mcc'], '+.4f')} [{row['delta_mcc_lo']}, {row['delta_mcc_hi']}] "
                      f"dAUC {fmt(row['delta_auc'], '+.4f')}"
                      + (f" n_unapplied_differ {row['n_unapplied_differ']}" if row["n_unapplied_differ"] else ""))
        elif m.get("n_folds"):
            print(f"[{name}] {m['n_folds']} base folds, no variant predictions in its {len(m['dirs'])} "
                  f"directories (first {m['dirs'][0]})")

    write_csv(out_dir / "summary.csv", rows)
    summary = {"config": str(args.config), "strategies": acfg["strategies"], "bootstrap": acfg["bootstrap"],
               "clip": acfg["clip"], "transform_classes": classes,
               "threshold_rule": "per fold, tuned on the base validation part (records or cv.tune_thresholds), "
                                 "never on variants",
               "families": meta, "rows": rows}
    tmp = tmp_path(out_dir / "summary.json")
    tmp.write_text(json.dumps(summary, indent=1))
    tmp.replace(out_dir / "summary.json")
    print(f"[out] {out_dir / 'summary.csv'} ({len(rows)} rows), {out_dir / 'summary.json'}")
    if not rows:
        return

    figs = Figures(resolve(acfg["figures_dir"]))
    labels = {name: str(spec.get("label", name)) for name, spec in acfg["families"].items()}
    families = list(acfg["families"])
    smells = list(dict.fromkeys(r["smell"] for r in rows))
    class_labels = {str(k): str(v) for k, v in (acfg.get("class_labels") or {}).items()}
    for strategy in acfg["strategies"]:
        for smell in smells:
            heatmap(figs, rows, families, labels, classes, class_labels, strategy, smell)
        shift_figure(figs, rows, shifts, labels, strategy, int(acfg["strongest"]["k"]),
                     int(acfg["strongest"]["min_rows"]), [str(c) for c in acfg["strongest"].get("classes", classes)])
    print(f"[figures] {len(figs.written)} files in {figs.dir}")


if __name__ == "__main__":
    main()
