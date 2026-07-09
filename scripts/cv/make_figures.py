"""Draw the figures declared in a YAML config.

Each entry of `figures` has a type (see FIGURES) and its inputs.
Every figure is written as PDF and PNG under figures_dir.

Usage:
    python scripts/cv/make_figures.py --config config/experiments/figures.yml [--only name1,name2]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from scripts.cv.summarize_cv import load_groups, seed_means  # noqa: E402

# Okabe and Ito colour-blind safe palette
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def style(cfg: dict[str, Any]) -> None:
    plt.rcParams.update({
        "font.size": cfg.get("font_size", 9),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 100,
        "savefig.bbox": "tight",
    })


def save(fig: plt.Figure, out_dir: Path, name: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=dpi)
    plt.close(fig)
    print(f"[figures] {name}")


def group_cells(spec: dict[str, Any]) -> dict:
    """Cells of one record group, read from one or several NAME=PATH inputs."""
    inputs = list(spec.get("inputs") or ([spec["input"]] if "input" in spec else []))
    if "input_glob" in spec:
        # NAME=pattern: one NAME=dir entry per directory matching the pattern
        name, pattern = spec["input_glob"].split("=", 1)
        inputs += [f"{name}={d}" for d in sorted(PROJECT_ROOT.glob(pattern))]
    groups, _ = load_groups(inputs, spec.get("strategy"))
    key = spec.get("group") or (next(iter(groups)) if len(groups) == 1 else None)
    if key is None or key not in groups:
        raise SystemExit(f"{inputs}: choose one of {list(groups)} with 'group'")
    return groups[key]


def fig_cv_bars(fig_cfg: dict[str, Any]) -> plt.Figure:
    metrics = fig_cfg.get("metrics", ["feature_envy_mcc", "long_method_mcc", "blob_mcc", "data_class_mcc", "macro_mcc"])
    labels = fig_cfg.get("metric_labels", metrics)
    groups = fig_cfg["groups"]
    fig, ax = plt.subplots(figsize=tuple(fig_cfg.get("size", [7.0, 3.0])))
    width = 0.8 / len(groups)
    x = np.arange(len(metrics))
    for i, spec in enumerate(groups):
        cells = group_cells(spec)
        means, stds = [], []
        for metric in metrics:
            values = seed_means(cells, metric)
            means.append(statistics.mean(values))
            stds.append(statistics.pstdev(values) if len(values) > 1 else 0.0)
        ax.bar(x + (i - (len(groups) - 1) / 2) * width, means, width, yerr=stds, capsize=2,
               color=spec.get("color", PALETTE[i % len(PALETTE)]), label=spec["label"])
    ax.set_xticks(x, labels)
    ax.set_ylabel(fig_cfg.get("ylabel", "MCC"))
    if "ylim" in fig_cfg:
        ax.set_ylim(*fig_cfg["ylim"])
    ax.legend(fontsize="small", ncol=fig_cfg.get("legend_ncol", 2), frameon=False)
    ax.set_title(fig_cfg.get("title", ""))
    return fig


def fig_forest(fig_cfg: dict[str, Any]) -> plt.Figure:
    summary = json.loads(resolve(fig_cfg["summary"]).read_text())
    metric = fig_cfg.get("metric", "macro_mcc")
    names = fig_cfg.get("group_labels", {})
    rows = [c for c in summary["comparisons"] if c["metric"] == metric and c["group"] in names]
    rows.sort(key=lambda c: list(names).index(c["group"]))
    fig, ax = plt.subplots(figsize=tuple(fig_cfg.get("size", [6.0, 0.45 * len(rows) + 1.0])))
    for i, c in enumerate(rows):
        significant = c.get("corrected_p_holm", c["corrected_p"]) < fig_cfg.get("alpha", 0.05)
        color = PALETTE[3] if significant else PALETTE[0]
        ax.errorbar(c["mean_diff"], i, xerr=[[c["mean_diff"] - c["ci95_low"]], [c["ci95_high"] - c["mean_diff"]]],
                    fmt="o", color=color, capsize=3)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(rows)), [names[c["group"]] for c in rows])
    ax.invert_yaxis()
    ax.set_xlabel(fig_cfg.get("xlabel", f"difference of {metric}"))
    ax.set_title(fig_cfg.get("title", ""))
    return fig


def fig_seed_strip(fig_cfg: dict[str, Any]) -> plt.Figure:
    metric = fig_cfg.get("metric", "feature_envy_mcc")
    groups = fig_cfg["groups"]
    fig, ax = plt.subplots(figsize=tuple(fig_cfg.get("size", [4.5, 3.0])))
    rng = np.random.default_rng(0)
    for i, spec in enumerate(groups):
        values = seed_means(group_cells(spec), metric)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(values)), values, s=18,
                   color=spec.get("color", PALETTE[i % len(PALETTE)]), alpha=0.85)
        ax.hlines(statistics.mean(values), i - 0.25, i + 0.25, color="black", linewidth=1.2)
    ax.set_xticks(range(len(groups)), [g["label"] for g in groups])
    ax.set_ylabel(fig_cfg.get("ylabel", metric))
    ax.set_title(fig_cfg.get("title", ""))
    return fig


def fig_scaling(fig_cfg: dict[str, Any]) -> plt.Figure:
    rows = json.loads(resolve(fig_cfg["summary"]).read_text())
    fig, ax = plt.subplots(figsize=tuple(fig_cfg.get("size", [4.5, 3.0])))
    labels = fig_cfg.get("mode_labels", {})
    for i, mode in enumerate(sorted({r["batch_scaling"] for r in rows})):
        pts = sorted((r["world_size"], r["graphs_per_sec"]) for r in rows if r["batch_scaling"] == mode)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", color=PALETTE[i], label=labels.get(mode, mode))
        ax.plot(xs, [ys[0] * x for x in xs], linestyle=":", color=PALETTE[i], alpha=0.6)
    modes = sorted({r["batch_scaling"] for r in rows})
    for extra in fig_cfg.get("extra_points", []):
        # same rule as scripts/pretraining/pretrain_scaling.py: throughput of the last epoch
        run = json.loads(resolve(extra["metrics"]).read_text())
        colour = PALETTE[modes.index(run["batch_scaling"])] if run["batch_scaling"] in modes else PALETTE[4]
        ax.scatter([run["world_size"]], [run["history"][-1]["train_graphs_per_sec"]], marker=extra.get("marker", "s"),
                   s=45, facecolors="none", edgecolors=colour, linewidths=1.5, label=extra["label"], zorder=3)
    ax.set_xlabel(fig_cfg.get("xlabel", "GPUs"))
    ax.set_ylabel(fig_cfg.get("ylabel", "graphs/s"))
    ax.set_xticks(sorted({r["world_size"] for r in rows}))
    ax.legend(fontsize="small", frameon=False)
    ax.set_title(fig_cfg.get("title", ""))
    return fig


def fig_latency_cdf(fig_cfg: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=tuple(fig_cfg.get("size", [4.5, 3.0])))
    for i, spec in enumerate(fig_cfg["inputs"]):
        data = json.loads(resolve(spec["path"]).read_text())
        field = f"{spec.get('step', 'total')}_ms"
        values = np.sort(np.asarray([r[field] for r in data["per_sample"] if r.get(field) is not None], dtype=float))
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=PALETTE[i], label=spec["label"])
    ax.set_xscale("log")
    for budget in fig_cfg.get("budgets_ms", []):
        ax.axvline(budget, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel(fig_cfg.get("xlabel", "ms per snippet"))
    ax.set_ylabel(fig_cfg.get("ylabel", "share of snippets"))
    ax.legend(fontsize="small", frameon=False)
    ax.set_title(fig_cfg.get("title", ""))
    return fig


def fig_learning_curve(fig_cfg: dict[str, Any]) -> plt.Figure:
    metrics = fig_cfg.get("metrics", ["feature_envy_mcc", "long_method_mcc", "blob_mcc", "data_class_mcc"])
    labels = fig_cfg.get("metric_labels", metrics)
    fig, ax = plt.subplots(figsize=tuple(fig_cfg.get("size", [4.5, 3.0])))
    for m, (metric, label) in enumerate(zip(metrics, labels)):
        xs, means, stds = [], [], []
        for point in fig_cfg["points"]:
            values = seed_means(group_cells(point), metric)
            xs.append(float(point["x"]))
            means.append(statistics.mean(values))
            stds.append(statistics.pstdev(values) if len(values) > 1 else 0.0)
        order = np.argsort(xs)
        xs, means, stds = (np.asarray(v)[order] for v in (xs, means, stds))
        ax.plot(xs, means, marker="o", color=PALETTE[m], label=label)
        ax.fill_between(xs, means - stds, means + stds, color=PALETTE[m], alpha=0.15)
    ax.set_xlabel(fig_cfg.get("xlabel", "training fraction"))
    ax.set_ylabel(fig_cfg.get("ylabel", "MCC"))
    # legend to the right of the axes, so it never covers a curve
    ax.legend(fontsize="small", frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title(fig_cfg.get("title", ""))
    return fig


def _family_colours(fig_cfg: dict[str, Any]) -> dict[str, str]:
    # the palette yellow is too light for small markers on white
    palette = [c for c in PALETTE if c != "#F0E442"]
    return {f: palette[i % len(palette)] for i, f in enumerate(fig_cfg["families"])}


def _family_legend(fig: plt.Figure, fig_cfg: dict[str, Any], colours: dict[str, str], used: set[str],
                   offset: float) -> None:
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=fig_cfg["families"][f])
               for f, c in colours.items() if f in used]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize="small",
               bbox_to_anchor=(0.5, offset))


def _smell_stats(spec: dict[str, Any], metric: str) -> tuple[float, float]:
    values = seed_means(group_cells(spec), metric)
    return statistics.mean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def fig_smell_panels(fig_cfg: dict[str, Any]) -> plt.Figure:
    """One panel per smell with the mean and std over seeds of every model."""
    from matplotlib.ticker import MaxNLocator

    smells = fig_cfg["smells"]
    models = fig_cfg["models"]
    colours = _family_colours(fig_cfg)
    fig, axes = plt.subplots(1, len(smells), figsize=tuple(fig_cfg.get("size", [12.0, 4.2])), sharey=True)
    for ax, (metric, title) in zip(np.atleast_1d(axes), smells.items()):
        stats_ = [_smell_stats(spec, metric) for spec in models]
        best = max(range(len(models)), key=lambda i: stats_[i][0])
        for y, (spec, (mean, std)) in enumerate(zip(models, stats_)):
            ax.errorbar(mean, y, xerr=std, fmt="o", color=colours[spec["family"]], capsize=2,
                        markersize=8 if y == best else 5)
        ax.set_title(title)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xlabel(fig_cfg.get("xlabel", "MCC"))
    first = np.atleast_1d(axes)[0]
    first.set_yticks(range(len(models)), [spec["label"] for spec in models], fontsize="small")
    first.invert_yaxis()
    _family_legend(fig, fig_cfg, colours, {spec["family"] for spec in models}, -0.04)
    fig.tight_layout()
    return fig


def fig_cost_vs_score(fig_cfg: dict[str, Any]) -> plt.Figure:
    """Per smell: score against measured inference time per sample (log scale)."""
    costs = json.loads(resolve(fig_cfg["costs"]).read_text())
    smells = fig_cfg["smells"]
    colours = _family_colours(fig_cfg)
    fig, axes = plt.subplots(1, len(smells), figsize=tuple(fig_cfg.get("size", [12.0, 3.4])), sharex=True)
    for ax, (metric, title) in zip(np.atleast_1d(axes), smells.items()):
        points = []
        for spec in fig_cfg["models"]:
            ms = costs.get(spec["cost_key"], {}).get("inference_ms_per_sample")
            if ms is None:
                continue
            mean, std = _smell_stats(spec, metric)
            points.append((ms, mean))
            ax.errorbar(ms, mean, yerr=std, fmt="o", color=colours[spec["family"]], capsize=2, markersize=5)
            ax.annotate(spec["short"], (ms, mean), textcoords="offset points", xytext=(4, 3), fontsize=7)
        if fig_cfg.get("frontier") and points:
            # cost frontier: a model is on it when every cheaper model scores lower
            best, front = -np.inf, []
            for ms, mean in sorted(points):
                if mean > best:
                    best = mean
                    front.append((ms, mean))
            xs, ys = zip(*front)
            ax.step(list(xs) + [max(p[0] for p in points) * 1.5], list(ys) + [ys[-1]], where="post",
                    color="grey", linewidth=1, linestyle="--", alpha=0.7, zorder=0)
        note = (fig_cfg.get("notes") or {}).get(metric)
        if note:
            ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left",
                    color="#333333", wrap=True)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel(fig_cfg.get("xlabel", "ms per sample"))
    np.atleast_1d(axes)[0].set_ylabel(fig_cfg.get("ylabel", "MCC"))
    used = {spec["family"] for spec in fig_cfg["models"]
            if costs.get(spec["cost_key"], {}).get("inference_ms_per_sample") is not None}
    _family_legend(fig, fig_cfg, colours, used, -0.12)
    fig.tight_layout()
    return fig


def fig_univariate_auc(fig_cfg: dict[str, Any]) -> plt.Figure:
    """ROC-AUC of single size features used directly as scores (no model)."""
    summary = json.loads(resolve(fig_cfg["summary"]).read_text())["size_shortcut"][fig_cfg.get("population", "eligible")]
    smells = fig_cfg["smells"]
    features = fig_cfg["features"]
    grid = np.array([[summary[s]["auc"][f] for s in smells] for f in features])
    fig, ax = plt.subplots(figsize=tuple(fig_cfg.get("size", [5.5, 5.0])))
    im = ax.imshow(grid, cmap="RdBu", vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(grid[i, j] - 0.5) > 0.3 else "black")
    ax.set_xticks(range(len(smells)), list(smells.values()))
    ax.set_yticks(range(len(features)), [features[f] for f in features])
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label=fig_cfg.get("colorbar_label", "ROC-AUC"))
    ax.set_title(fig_cfg.get("title", ""), fontsize="medium")
    return fig


def fig_size_controlled(fig_cfg: dict[str, Any]) -> plt.Figure:
    """ROC-AUC overall and inside size strata (top), and AUC added to the size model (bottom)."""
    data = json.loads(resolve(fig_cfg["input"]).read_text())[fig_cfg["dataset"]]
    smells = fig_cfg["smells"]
    fams = fig_cfg["models"]
    colours = _family_colours(fig_cfg)
    fig, axes = plt.subplots(2, len(smells), figsize=tuple(fig_cfg.get("size", [12.0, 5.6])),
                             sharey="row", gridspec_kw={"height_ratios": [1.6, 1.0]})
    for col, (label, title) in enumerate(smells.items()):
        top, bottom = axes[0, col], axes[1, col]
        for y, spec in enumerate(fams):
            m = data[label][spec["key"]]
            c = colours[spec["family"]]
            auc, strat = m["roc_auc"]["mean"], m["stratified_auc"]["mean"]
            top.plot([strat, auc], [y, y], color=c, linewidth=1.2, alpha=0.6)
            top.plot(auc, y, "o", markerfacecolor="white", markeredgecolor=c, markersize=6)
            top.plot(strat, y, "o", color=c, markersize=6)
            inc = m["incremental_auc"]
            bottom.barh(y, inc["mean"], xerr=inc["sd"], color=c, height=0.6, capsize=2)
        top.axvline(0.5, color="grey", linestyle=":", linewidth=1)
        top.set_title(title)
        top.set_xlim(fig_cfg.get("auc_xlim", [0.45, 1.0]))
        top.set_xlabel("ROC-AUC")
        bottom.axvline(0.0, color="black", linewidth=0.8)
        bottom.set_xlim(fig_cfg.get("inc_xlim", [-0.02, 0.08]))
        bottom.set_xlabel(fig_cfg.get("inc_xlabel", "AUC added to the size model"))
    for row in (0, 1):
        axes[row, 0].set_yticks(range(len(fams)), [spec["label"] for spec in fams], fontsize="small")
        axes[row, 0].invert_yaxis()
    marker_handles = [
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="black",
                   label="all samples of the level"),
        plt.Line2D([], [], marker="o", linestyle="", color="black",
                   label=f"pairs of the same size stratum ({fig_cfg.get('strata_label', 'LOC decile')})"),
    ]
    fig.legend(handles=marker_handles, loc="upper center", ncol=2, frameon=False, fontsize="small",
               bbox_to_anchor=(0.5, 1.03))
    _family_legend(fig, fig_cfg, colours, {spec["family"] for spec in fams}, -0.05)
    fig.tight_layout()
    return fig


def _seed_mean_label_mcc(path: Path, label: str, strategy: str) -> tuple[float, float]:
    """Per-label tuned MCC of a records.json: mean over folds per seed, then mean and std over seeds."""
    by_seed: dict[int, list[float]] = {}
    for r in json.loads(path.read_text()):
        combo = r["combo"]
        # pipeline records name the key training.split_strategy
        if combo.get("split_strategy", combo.get("training.split_strategy", strategy)) != strategy:
            continue
        v = r["per_label_tuned"][label]["mcc"]
        if v is not None:
            by_seed.setdefault(int(r["seed"]), []).append(float(v))
    means = [statistics.mean(v) for v in by_seed.values()]
    return statistics.mean(means), statistics.pstdev(means) if len(means) > 1 else 0.0


def fig_offered_collapse(fig_cfg: dict[str, Any]) -> plt.Figure:
    """Per smell: MCC on all samples and on the samples where the smell was offered."""
    smells = fig_cfg["smells"]
    models = fig_cfg["models"]
    colours = _family_colours(fig_cfg)
    fig, axes = plt.subplots(1, len(smells), figsize=tuple(fig_cfg.get("size", [10.0, 3.8])), sharey=True)
    for ax, (label, title) in zip(np.atleast_1d(axes), smells.items()):
        for y, spec in enumerate(models):
            c = colours[spec["family"]]
            a, _ = _seed_mean_label_mcc(resolve(spec["all"]), label, fig_cfg["strategy"])
            o, osd = _seed_mean_label_mcc(resolve(spec["offered"]), label, fig_cfg["strategy"])
            ax.annotate("", xy=(o, y), xytext=(a, y),
                        arrowprops={"arrowstyle": "->", "color": c, "alpha": 0.6, "linewidth": 1.2})
            ax.plot(a, y, "o", markerfacecolor="white", markeredgecolor=c, markersize=6)
            ax.errorbar(o, y, xerr=osd, fmt="o", color=c, markersize=6, capsize=2)
        ax.axvline(0.0, color="grey", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("MCC")
        ax.set_xlim(fig_cfg.get("xlim", [-0.1, 0.8]))
    first = np.atleast_1d(axes)[0]
    first.set_yticks(range(len(models)), [spec["label"] for spec in models], fontsize="small")
    first.invert_yaxis()
    marker_handles = [
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="black",
                   label=fig_cfg.get("all_label", "all method snippets")),
        plt.Line2D([], [], marker="o", linestyle="", color="black",
                   label=fig_cfg.get("offered_label", "snippets where the smell was offered")),
    ]
    fig.legend(handles=marker_handles, loc="upper center", ncol=2, frameon=False, fontsize="small",
               bbox_to_anchor=(0.5, 1.06))
    _family_legend(fig, fig_cfg, colours, {spec["family"] for spec in models}, -0.08)
    fig.tight_layout()
    return fig


def fig_scorer_auc(fig_cfg: dict[str, Any]) -> plt.Figure:
    """One panel per (evaluation file, smell) with the ROC-AUC of each scorer."""
    panels = fig_cfg["panels"]
    scorers = fig_cfg["scorers"]
    colours = _family_colours(fig_cfg)
    fig, axes = plt.subplots(1, len(panels), figsize=tuple(fig_cfg.get("size", [13.0, 3.4])), sharey=True)
    for ax, panel in zip(np.atleast_1d(axes), panels):
        res = json.loads(resolve(panel["input"]).read_text())["by_smell"][panel["smell"]]
        for y, spec in enumerate(scorers):
            m = res["scorers"].get(spec["key"])
            if m is None:
                continue
            ax.plot(m["roc_auc"], y, "o", color=colours[spec["family"]], markersize=6)
        ax.axvline(0.5, color="grey", linestyle=":", linewidth=1)
        ax.set_title(f"{panel['title']}\n{res['n_pos']} positive / {res['n']}", fontsize="medium")
        ax.set_xlim(panel.get("xlim", [0.45, 1.0]))
        ax.set_xlabel("ROC-AUC")
    first = np.atleast_1d(axes)[0]
    first.set_yticks(range(len(scorers)), [spec["label"] for spec in scorers], fontsize="small")
    first.invert_yaxis()
    _family_legend(fig, fig_cfg, colours, {spec["family"] for spec in scorers}, -0.1)
    fig.tight_layout()
    return fig


def fig_population_shift(fig_cfg: dict[str, Any]) -> plt.Figure:
    """One panel per (dataset, smell) with the MCC on the wide and the narrow population."""
    panels = fig_cfg["panels"]
    models = fig_cfg["models"]
    colours = _family_colours(fig_cfg)
    fig, axes = plt.subplots(1, len(panels), figsize=tuple(fig_cfg.get("size", [15.0, 3.8])), sharey=True)
    for ax, panel in zip(np.atleast_1d(axes), panels):
        for y, spec in enumerate(models):
            files = spec.get("records", {}).get(panel["dataset"])
            if not files:
                continue
            c = colours[spec["family"]]
            a, _ = _seed_mean_label_mcc(resolve(files["wide"]), panel["label"], fig_cfg["strategy"])
            o, osd = _seed_mean_label_mcc(resolve(files["narrow"]), panel["label"], fig_cfg["strategy"])
            ax.annotate("", xy=(o, y), xytext=(a, y),
                        arrowprops={"arrowstyle": "->", "color": c, "alpha": 0.6, "linewidth": 1.1})
            ax.plot(a, y, "o", markerfacecolor="white", markeredgecolor=c, markersize=5.5)
            ax.errorbar(o, y, xerr=osd, fmt="o", color=c, markersize=5.5, capsize=2)
        ax.axvline(0.0, color="grey", linestyle=":", linewidth=1)
        ax.set_title(panel["title"], fontsize="medium")
        ax.set_xlabel("MCC")
        ax.set_xlim(fig_cfg.get("xlim", [-0.1, 0.85]))
    first = np.atleast_1d(axes)[0]
    first.set_yticks(range(len(models)), [spec["label"] for spec in models], fontsize="small")
    first.invert_yaxis()
    marker_handles = [
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="black",
                   label=fig_cfg["wide_label"]),
        plt.Line2D([], [], marker="o", linestyle="", color="black", label=fig_cfg["narrow_label"]),
    ]
    fig.legend(handles=marker_handles, loc="upper center", ncol=2, frameon=False, fontsize="small",
               bbox_to_anchor=(0.5, 1.07))
    _family_legend(fig, fig_cfg, colours, {spec["family"] for spec in models}, -0.08)
    fig.tight_layout()
    return fig


FIGURES = {
    "population_shift": fig_population_shift,
    "scorer_auc": fig_scorer_auc,
    "offered_collapse": fig_offered_collapse,
    "size_controlled": fig_size_controlled,
    "smell_panels": fig_smell_panels,
    "cost_vs_score": fig_cost_vs_score,
    "univariate_auc": fig_univariate_auc,
    "cv_bars": fig_cv_bars,
    "forest": fig_forest,
    "seed_strip": fig_seed_strip,
    "scaling": fig_scaling,
    "latency_cdf": fig_latency_cdf,
    "learning_curve": fig_learning_curve,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", default=None, help="comma-separated figure names")
    args, extra = parser.parse_known_args()
    cfg = load_config(args.config, parse_cli_overrides(extra))["figures_config"]
    style(cfg)
    wanted = set(args.only.split(",")) if args.only else None
    out_dir = resolve(cfg["figures_dir"])
    for name, fig_cfg in cfg["figures"].items():
        if wanted is not None and name not in wanted:
            continue
        fig = FIGURES[fig_cfg["type"]](fig_cfg)
        save(fig, out_dir, name, int(cfg.get("dpi", 200)))


if __name__ == "__main__":
    main()
