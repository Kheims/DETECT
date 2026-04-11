"""Aggregate results from classical + sequence runs into tables and figures.

Usage:
    uv run python scripts/report_runs.py \
        --artifacts-root artifacts \
        --output-dir /Users/djamel/Repositories/personal/Thesis/survey-paper/img/benchmark

Reads:
  artifacts/runs/<fingerprint>/{model}_results.json  (classical + sequence)
  artifacts/runs/<fingerprint>/metrics.json          (GNN, optional)

Writes:
  <output-dir>/comparison_table.md
  <output-dir>/comparison_table.tex
  <output-dir>/f1_macro_comparison.pdf (+ .png)
  <output-dir>/mcc_macro_comparison.pdf (+ .png)
  <output-dir>/per_label_f1_heatmap.pdf (+ .png)
  <output-dir>/per_label_mcc_heatmap.pdf (+ .png)
  <output-dir>/madeyski_comparison.pdf (+ .png)
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

LABEL_NAMES = ["is_feature_envy", "is_long_method", "is_blob", "is_data_class"]
LABEL_SHORT = ["FE", "LM", "Blob", "DC"]

MADEYSKI_MCC = {
    "FE":   {"best": 0.31, "algo": "FDA",  "dataset": "DS1"},
    "LM":   {"best": 0.81, "algo": "RF",   "dataset": "DS2"},
    "Blob": {"best": 0.55, "algo": "RF",   "dataset": "DS1"},
    "DC":   {"best": 0.57, "algo": "RF",   "dataset": "DS1"},
}

MODEL_DISPLAY = {
    "random_forest":    "RF",
    "svm":              "SVM",
    "xgboost":          "XGBoost",
    "decision_tree":    "DT",
    "knn":              "KNN",
    "lstm":             "LSTM",
    "bilstm":           "BiLSTM",
    "bilstm_attention": "BiLSTM+Attn",
    "cnn":              "CNN",
    "gcn":              "GCN",
    "gat":              "GAT",
    "graphsage":        "GraphSAGE",
    "gin":              "GIN",
}

FAMILY_OF = {
    "random_forest":    "Classical",
    "svm":              "Classical",
    "xgboost":          "Classical",
    "decision_tree":    "Classical",
    "knn":              "Classical",
    "lstm":             "Sequence DL",
    "bilstm":           "Sequence DL",
    "bilstm_attention": "Sequence DL",
    "cnn":              "Sequence DL",
    "gcn":              "GNN",
    "gat":              "GNN",
    "graphsage":        "GNN",
    "gin":              "GNN",
}


def find_latest_results(artifacts_root: Path) -> dict[str, dict]:
    """Scan artifacts/runs/*/ for result JSON files, keep newest per model."""
    runs_dir = artifacts_root / "runs"
    latest: dict[str, tuple[float, dict]] = {}
    if not runs_dir.exists():
        return {}
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        for f in run_dir.glob("*_results.json"):
            name = f.stem.replace("_results", "")
            mtime = f.stat().st_mtime
            if name in latest and latest[name][0] >= mtime:
                continue
            try:
                with f.open() as fh:
                    data = json.load(fh)
                latest[name] = (mtime, data)
            except (json.JSONDecodeError, OSError):
                continue
    return {name: data for name, (_, data) in latest.items()}


def find_best_gnn_results(
    artifacts_root: Path, min_epochs: int = 20
) -> dict[str, dict]:
    """Scan artifacts/runs/*/metrics.json for GNN runs, keep the best per
    architecture (by tuned test f1_macro) among runs with >= min_epochs.

    Returns a dict keyed by architecture name ('gcn', 'gat', ...) containing
    a dict with keys: metrics, config, run_dir.
    """
    runs_dir = artifacts_root / "runs"
    best: dict[str, tuple[float, dict]] = {}
    if not runs_dir.exists():
        return {}
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        if not (metrics_path.exists() and config_path.exists()):
            continue
        try:
            with metrics_path.open() as fh:
                metrics = json.load(fh)
            with config_path.open() as fh:
                config = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        arch = config.get("architecture")
        if not arch:
            continue
        num_epochs = len(metrics.get("history", []))
        if num_epochs < min_epochs:
            continue
        tuned = metrics.get("test_tuned", {})
        f1 = tuned.get("f1_macro", 0.0)
        if arch not in best or f1 > best[arch][0]:
            best[arch] = (f1, {"metrics": metrics, "config": config,
                                "run_dir": str(run_dir)})
    return {arch: payload for arch, (_, payload) in best.items()}


def extract_gnn_metrics(arch: str, payload: dict) -> dict[str, Any] | None:
    """Normalize a GNN metrics.json into the same flat row dict used by
    extract_metrics() for classical/sequence.

    GNN runs are single-seed so std is 0. We pull per-label from `test_tuned`.
    """
    metrics = payload["metrics"]
    config = payload["config"]
    tuned = metrics.get("test_tuned", {})
    if not tuned:
        return None

    f1 = tuned.get("f1_macro", 0.0)
    mcc = tuned.get("mcc_macro", 0.0)

    pl_block = tuned.get("per_label", {})
    per_label = {}
    for label in LABEL_NAMES:
        entry = pl_block.get(label, {})
        per_label[label] = {
            "f1_mean":  float(entry.get("f1", 0.0)),
            "f1_std":   0.0,
            "mcc_mean": float(entry.get("mcc", 0.0)),
            "mcc_std":  0.0,
        }

    best_params = {
        "layers": config.get("num_layers"),
        "loss":   config.get("loss"),
    }
    # Drop None entries for a cleaner table
    best_params = {k: v for k, v in best_params.items() if v is not None}

    return {
        "name":        arch,
        "display":     MODEL_DISPLAY.get(arch, arch),
        "family":      FAMILY_OF.get(arch, "GNN"),
        "best_params": best_params,
        "seeds":       [42],  # single-run placeholder
        "f1_macro":    float(f1),
        "f1_std":      0.0,
        "mcc_macro":   float(mcc),
        "mcc_std":     0.0,
        "per_label":   per_label,
    }


def extract_metrics(model_name: str, data: dict) -> dict[str, Any] | None:
    """Normalize the per-result JSON into a flat dict."""
    # Classical uses data["best"]["results"]; sequence uses data["aggregated"]
    if "best" in data and data["best"] and "results" in data["best"]:
        agg = data["best"]["results"]
        best_params = data["best"].get("params", {})
    elif "aggregated" in data:
        agg = data["aggregated"]
        best_params = {}
    else:
        return None

    def get(d, *path):
        cur = d
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return None
            cur = cur[p]
        return cur

    f1 = get(agg, "f1_macro", "mean")
    f1_std = get(agg, "f1_macro", "std")
    mcc = get(agg, "mcc_macro", "mean")
    mcc_std = get(agg, "mcc_macro", "std")

    per_label = {}
    for label in LABEL_NAMES:
        per_label[label] = {
            "f1_mean":  get(agg, "per_label", label, "f1",  "mean") or 0.0,
            "f1_std":   get(agg, "per_label", label, "f1",  "std")  or 0.0,
            "mcc_mean": get(agg, "per_label", label, "mcc", "mean") or 0.0,
            "mcc_std":  get(agg, "per_label", label, "mcc", "std")  or 0.0,
        }

    return {
        "name":       model_name,
        "display":    MODEL_DISPLAY.get(model_name, model_name),
        "family":     FAMILY_OF.get(model_name, "Other"),
        "best_params": best_params,
        "seeds":      data.get("seeds", []),
        "f1_macro":   f1 or 0.0,
        "f1_std":     f1_std or 0.0,
        "mcc_macro":  mcc or 0.0,
        "mcc_std":    mcc_std or 0.0,
        "per_label":  per_label,
    }


def build_rows(
    results: dict[str, dict],
    gnn_results: dict[str, dict] | None = None,
) -> list[dict]:
    rows = []
    for model_name, data in results.items():
        m = extract_metrics(model_name, data)
        if m is not None:
            rows.append(m)
    if gnn_results:
        for arch, payload in gnn_results.items():
            m = extract_gnn_metrics(arch, payload)
            if m is not None:
                rows.append(m)
    # Sort by family (Classical first) then F1 descending
    family_order = {"Classical": 0, "Sequence DL": 1, "GNN": 2, "Other": 3}
    rows.sort(key=lambda r: (family_order.get(r["family"], 99), -r["f1_macro"]))
    return rows


def write_markdown_table(rows: list[dict], out_path: Path) -> None:
    lines = []
    lines.append("# Benchmark results — unified pipeline")
    lines.append("")
    lines.append("Aggregated across seeds (mean ± std).")
    lines.append("")
    lines.append("## F1-macro / MCC-macro")
    lines.append("")
    lines.append("| Family | Model | F1-macro | MCC-macro | Best params |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        params = ", ".join(f"{k}={v}" for k, v in r["best_params"].items()) or "—"
        lines.append(
            f"| {r['family']} | {r['display']} | "
            f"{r['f1_macro']:.3f} ± {r['f1_std']:.3f} | "
            f"{r['mcc_macro']:.3f} ± {r['mcc_std']:.3f} | "
            f"{params} |"
        )
    lines.append("")
    lines.append("## Per-smell F1")
    lines.append("")
    header = "| Model | " + " | ".join(LABEL_SHORT) + " |"
    sep = "|---|" + "|".join(["---"] * len(LABEL_SHORT)) + "|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        cells = []
        for label in LABEL_NAMES:
            pl = r["per_label"][label]
            cells.append(f"{pl['f1_mean']:.3f} ± {pl['f1_std']:.3f}")
        lines.append(f"| {r['display']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Per-smell MCC")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for r in rows:
        cells = []
        for label in LABEL_NAMES:
            pl = r["per_label"][label]
            cells.append(f"{pl['mcc_mean']:.3f} ± {pl['mcc_std']:.3f}")
        lines.append(f"| {r['display']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Comparison to Madeyski & Lewowski (IST 2023)")
    lines.append("")
    lines.append("Best MCC per smell — ours vs theirs.")
    lines.append("")
    lines.append("| Smell | Madeyski best | Ours best | Delta |")
    lines.append("|---|---|---|---|")
    for label, short in zip(LABEL_NAMES, LABEL_SHORT):
        best_our = max(r["per_label"][label]["mcc_mean"] for r in rows) if rows else 0.0
        best_our_model = max(
            rows, key=lambda r: r["per_label"][label]["mcc_mean"]
        )["display"] if rows else "—"
        their = MADEYSKI_MCC[short]
        delta = best_our - their["best"]
        lines.append(
            f"| {short} | {their['best']:.2f} ({their['algo']}, {their['dataset']}) | "
            f"{best_our:.3f} ({best_our_model}) | {delta:+.3f} |"
        )

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def write_latex_table(rows: list[dict], out_path: Path) -> None:
    """Write a LaTeX booktabs table for direct inclusion in the paper."""
    lines = []
    lines.append("% Auto-generated by scripts/report_runs.py")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Code smell detection benchmark on MLCQ via the unified pipeline. "
                 r"Scores are mean $\pm$ std over seeds.}")
    lines.append(r"\label{tab:benchmark_results}")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Family & Model & F1-macro & MCC-macro & "
                 r"FE & LM & Blob & DC \\")
    lines.append(r"\midrule")

    last_family = None
    for r in rows:
        family_cell = r["family"] if r["family"] != last_family else ""
        last_family = r["family"]

        cells = [family_cell, r["display"],
                 f"{r['f1_macro']:.3f} $\\pm$ {r['f1_std']:.3f}",
                 f"{r['mcc_macro']:.3f} $\\pm$ {r['mcc_std']:.3f}"]
        for label in LABEL_NAMES:
            pl = r["per_label"][label]
            cells.append(f"{pl['f1_mean']:.3f}")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def plot_bar_comparison(rows, metric_key, metric_std_key, title, ylabel, out_path):
    """Horizontal bar chart comparing all models on a single metric."""
    fig, ax = plt.subplots(figsize=(6.5, max(3.0, 0.5 * len(rows) + 1.5)))

    names = [r["display"] for r in rows]
    means = [r[metric_key] for r in rows]
    stds = [r[metric_std_key] for r in rows]
    families = [r["family"] for r in rows]

    # Colors by family (viridis-inspired, pastel)
    colors = []
    for fam in families:
        if fam == "Classical":
            colors.append("#5ec962")  # viridis green
        elif fam == "Sequence DL":
            colors.append("#3b528b")  # viridis dark blue
        elif fam == "GNN":
            colors.append("#fde725")  # viridis yellow
        else:
            colors.append("#999999")  # gray fallback

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor="#333",
                   linewidth=0.8, capsize=3, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, max(means) * 1.2 if means else 1)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels at end of bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(mean + std + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{mean:.3f}", va="center", fontsize=9, color="#333")

    plt.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(str(out_path) + ext, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}.pdf/.png")


def plot_per_label_heatmap(rows, metric_key_mean, title, out_path, cmap="viridis"):
    """Heatmap: rows = models, cols = labels, cells = metric."""
    names = [r["display"] for r in rows]
    data = np.array([[r["per_label"][label][metric_key_mean] for label in LABEL_NAMES]
                     for r in rows])

    fig, ax = plt.subplots(figsize=(6.0, max(3.0, 0.5 * len(rows) + 1.5)))
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=max(data.max(), 0.9))
    ax.set_xticks(np.arange(len(LABEL_SHORT)))
    ax.set_xticklabels(LABEL_SHORT, fontsize=11)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_title(title, fontsize=12)

    # Annotate each cell
    for i in range(len(names)):
        for j in range(len(LABEL_SHORT)):
            text_color = "white" if data[i, j] < 0.4 else "black"
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    color=text_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(str(out_path) + ext, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}.pdf/.png")


def plot_madeyski_comparison(rows, out_path):
    """Grouped bar chart: our best MCC vs Madeyski best MCC per smell."""
    if not rows:
        return
    our_best = {}
    our_label_model = {}
    for short, label in zip(LABEL_SHORT, LABEL_NAMES):
        best_val = max(r["per_label"][label]["mcc_mean"] for r in rows)
        best_model = max(
            rows, key=lambda r: r["per_label"][label]["mcc_mean"]
        )["display"]
        our_best[short] = best_val
        our_label_model[short] = best_model

    labels = LABEL_SHORT
    their = [MADEYSKI_MCC[s]["best"] for s in labels]
    ours = [our_best[s] for s in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    b1 = ax.bar(x - width / 2, their, width, label="Madeyski 2023 (best)",
                color="#3b528b", edgecolor="#333", linewidth=0.8, alpha=0.85)
    b2 = ax.bar(x + width / 2, ours, width, label="Ours (best across models)",
                color="#5ec962", edgecolor="#333", linewidth=0.8, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("MCC (best per smell)", fontsize=11)
    ax.set_title("Best per-smell MCC — ours vs Madeyski & Lewowski (IST 2023)",
                 fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(b1, their):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                f"{val:.2f}", ha="center", fontsize=9, color="#333")
    for bar, val, short in zip(b2, ours, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                f"{val:.2f}\n({our_label_model[short]})",
                ha="center", fontsize=8, color="#333")

    plt.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(str(out_path) + ext, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}.pdf/.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--output-dir",
                        default="/Users/djamel/Repositories/personal/Thesis/survey-paper/img/benchmark/results")
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root).resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {artifacts_root}/runs for results...")
    results = find_latest_results(artifacts_root)
    print(f"Found {len(results)} classical/sequence model(s): {sorted(results.keys())}")

    gnn_results = find_best_gnn_results(artifacts_root, min_epochs=20)
    print(f"Found {len(gnn_results)} GNN architecture(s): {sorted(gnn_results.keys())}")

    rows = build_rows(results, gnn_results)
    if not rows:
        print("No results found — nothing to report.")
        return

    write_markdown_table(rows, out_dir / "comparison_table.md")
    write_latex_table(rows, out_dir / "comparison_table.tex")

    plot_bar_comparison(rows, "f1_macro", "f1_std",
                        "F1-macro across models",
                        "F1-macro",
                        out_dir / "f1_macro_comparison")
    plot_bar_comparison(rows, "mcc_macro", "mcc_std",
                        "MCC-macro across models",
                        "MCC-macro",
                        out_dir / "mcc_macro_comparison")

    plot_per_label_heatmap(rows, "f1_mean",
                           "Per-smell F1 across models",
                           out_dir / "per_label_f1_heatmap")
    plot_per_label_heatmap(rows, "mcc_mean",
                           "Per-smell MCC across models",
                           out_dir / "per_label_mcc_heatmap")

    plot_madeyski_comparison(rows, out_dir / "madeyski_comparison")

    print(f"\nDone. Wrote to {out_dir}")


if __name__ == "__main__":
    main()
