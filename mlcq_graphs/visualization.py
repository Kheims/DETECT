from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42

import matplotlib.pyplot as plt

FIGURE_STYLE = ["seaborn-v0_8-paper", "tableau-colorblind10"]

FONT_SIZE_AXIS = 12
FONT_SIZE_TITLE = 13
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 10

LABEL_COLORS = {
    "is_feature_envy": "#006BA4",
    "is_long_method": "#FF800E",
    "is_blob": "#595959",
    "is_data_class": "#5F9ED1",
}


def _clean_label(name: str) -> str:
    return name.removeprefix("is_").replace("_", " ")


def _save_figure(fig: plt.Figure, output_dir: Path, name: str) -> dict[str, str]:
    pdf_path = output_dir / f"{name}.pdf"
    png_path = output_dir / f"{name}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {f"{name}_pdf": str(pdf_path), f"{name}_png": str(png_path)}


def plot_pr_curves(
    curves: dict[str, dict],
    label_names: list[str],
    output_dir: Path,
) -> dict[str, str]:
    with plt.style.context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(6, 4.5))

        for name in label_names:
            data = curves.get(name, {})
            if not data:
                continue
            precision = data.get("precision", [])
            recall = data.get("recall", [])
            if not precision or not recall:
                continue
            color = LABEL_COLORS.get(name)
            ax.plot(
                recall,
                precision,
                label=_clean_label(name),
                color=color,
                linewidth=1.5,
            )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Recall", fontsize=FONT_SIZE_AXIS)
        ax.set_ylabel("Precision", fontsize=FONT_SIZE_AXIS)
        ax.set_title("Precision-Recall Curves (Test Set)", fontsize=FONT_SIZE_TITLE)
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        ax.legend(loc="lower left", fontsize=FONT_SIZE_LEGEND)

    return _save_figure(fig, output_dir, "pr_curves")


def plot_roc_curves(
    curves: dict[str, dict],
    label_names: list[str],
    output_dir: Path,
) -> dict[str, str]:
    with plt.style.context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(6, 4.5))

        ax.plot([0, 1], [0, 1], linestyle="--", color="#ABABAB", linewidth=1.0, label="random")

        for name in label_names:
            data = curves.get(name, {})
            if not data:
                continue
            fpr = data.get("fpr", [])
            tpr = data.get("tpr", [])
            if not fpr or not tpr:
                continue
            color = LABEL_COLORS.get(name)
            ax.plot(
                fpr,
                tpr,
                label=_clean_label(name),
                color=color,
                linewidth=1.5,
            )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("False Positive Rate", fontsize=FONT_SIZE_AXIS)
        ax.set_ylabel("True Positive Rate", fontsize=FONT_SIZE_AXIS)
        ax.set_title("ROC Curves (Test Set)", fontsize=FONT_SIZE_TITLE)
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        ax.legend(loc="lower right", fontsize=FONT_SIZE_LEGEND)

    return _save_figure(fig, output_dir, "roc_curves")


def plot_f1_vs_threshold(
    f1_data: dict[str, dict],
    label_names: list[str],
    output_dir: Path,
) -> dict[str, str]:
    with plt.style.context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(6, 4.5))

        for name in label_names:
            data = f1_data.get(name, {})
            if not data:
                continue
            thresholds = data.get("thresholds", [])
            f1_vals = data.get("f1", [])
            tuned = data.get("tuned_threshold")
            if not thresholds or not f1_vals:
                continue
            color = LABEL_COLORS.get(name)
            ax.plot(
                thresholds,
                f1_vals,
                label=_clean_label(name),
                color=color,
                linewidth=1.5,
            )
            if tuned is not None:
                ax.axvline(x=tuned, color=color, linestyle="--", linewidth=0.8, alpha=0.7)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Threshold", fontsize=FONT_SIZE_AXIS)
        ax.set_ylabel("F1 Score", fontsize=FONT_SIZE_AXIS)
        ax.set_title("F1 Score vs Decision Threshold", fontsize=FONT_SIZE_TITLE)
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        ax.legend(loc="best", fontsize=FONT_SIZE_LEGEND)

    return _save_figure(fig, output_dir, "f1_vs_threshold")


def plot_confusion_matrices(
    per_label: dict[str, dict],
    label_names: list[str],
    output_dir: Path,
) -> dict[str, str]:
    n_labels = len(label_names)
    if n_labels == 0:
        return {}

    with plt.style.context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, n_labels, figsize=(3.5 * n_labels, 3.5))
        if n_labels == 1:
            axes = [axes]

        for ax, name in zip(axes, label_names):
            label_data = per_label.get(name, {})
            tp = int(label_data.get("tp", 0))
            fp = int(label_data.get("fp", 0))
            tn = int(label_data.get("tn", 0))
            fn = int(label_data.get("fn", 0))

            cm = [[tn, fp], [fn, tp]]

            im = ax.imshow(cm, cmap="Blues", vmin=0)
            ax.set_title(_clean_label(name), fontsize=FONT_SIZE_TITLE)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=FONT_SIZE_TICK)
            ax.set_yticklabels(["True 0", "True 1"], fontsize=FONT_SIZE_TICK)

            for row in range(2):
                for col in range(2):
                    val = cm[row][col]
                    cell_max = max(max(cm[0]), max(cm[1]))
                    text_color = "white" if val > cell_max * 0.6 else "black"
                    ax.text(
                        col,
                        row,
                        str(val),
                        ha="center",
                        va="center",
                        fontsize=FONT_SIZE_AXIS,
                        color=text_color,
                    )

        fig.tight_layout()

    return _save_figure(fig, output_dir, "confusion_matrices")


def generate_publication_figures(
    metrics_path: Path,
    curves_path: Path,
    output_dir: Path,
    label_names: list[str],
) -> dict[str, str]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    if not curves_path.exists():
        raise FileNotFoundError(f"Missing curves file: {curves_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(metrics_path.read_text())
    curves = json.loads(curves_path.read_text())

    per_label_tuned = metrics.get("test_tuned", {}).get("per_label", {})
    figure_paths: dict[str, str] = {}
    figure_paths.update(plot_pr_curves(curves.get("pr_curves", {}), label_names, output_dir))
    figure_paths.update(plot_roc_curves(curves.get("roc_curves", {}), label_names, output_dir))
    figure_paths.update(plot_f1_vs_threshold(curves.get("f1_vs_threshold", {}), label_names, output_dir))
    figure_paths.update(plot_confusion_matrices(per_label_tuned, label_names, output_dir))
    return figure_paths
