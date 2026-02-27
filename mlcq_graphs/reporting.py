from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def generate_training_figures(metrics_path: Path, output_dir: Path) -> dict[str, str]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(metrics_path.read_text())

    figure_paths: dict[str, str] = {}

    history = metrics.get("history", [])
    if isinstance(history, list) and history:
        epochs = [_as_float(item.get("epoch"), 0.0) for item in history]
        train_loss = [_as_float(item.get("train_loss"), 0.0) for item in history]
        val_f1_micro = [_as_float(item.get("val_f1_micro"), 0.0) for item in history]
        val_f1_macro = [_as_float(item.get("val_f1_macro"), 0.0) for item in history]

        fig, ax_left = plt.subplots(figsize=(9, 5))
        ax_left.plot(epochs, train_loss, label="train_loss", color="#1f77b4", linewidth=2)
        ax_left.set_xlabel("Epoch")
        ax_left.set_ylabel("Loss")

        ax_right = ax_left.twinx()
        ax_right.plot(
            epochs,
            val_f1_micro,
            label="val_f1_micro",
            color="#2ca02c",
            linewidth=2,
        )
        ax_right.plot(
            epochs,
            val_f1_macro,
            label="val_f1_macro",
            color="#d62728",
            linewidth=2,
        )
        ax_right.set_ylabel("F1")

        handles = ax_left.get_lines() + ax_right.get_lines()
        labels = [line.get_label() for line in handles]
        ax_left.legend(handles, labels, loc="best")
        ax_left.set_title("Training Curves")

        curves_path = output_dir / "training_curves.png"
        fig.tight_layout()
        fig.savefig(curves_path, dpi=180)
        plt.close(fig)
        figure_paths["training_curves"] = str(curves_path)

    fixed = metrics.get("test_fixed_0_5", {})
    tuned = metrics.get("test_tuned", {})
    if isinstance(fixed, dict) and isinstance(tuned, dict):
        labels = ["f1_micro", "f1_macro", "pr_auc_macro"]
        fixed_vals = [_as_float(fixed.get(name), 0.0) for name in labels]
        tuned_vals = [_as_float(tuned.get(name), 0.0) for name in labels]

        x = list(range(len(labels)))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar([idx - width / 2 for idx in x], fixed_vals, width, label="fixed_0_5")
        ax.bar([idx + width / 2 for idx in x], tuned_vals, width, label="tuned")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Test Metrics: Fixed vs Tuned Thresholds")
        ax.legend(loc="best")

        comparison_path = output_dir / "test_metrics_comparison.png"
        fig.tight_layout()
        fig.savefig(comparison_path, dpi=180)
        plt.close(fig)
        figure_paths["test_metrics_comparison"] = str(comparison_path)

    return figure_paths


def generate_ablation_figure(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not summary_rows:
        return

    labels = [str(row.get("combo", {})) for row in summary_rows]
    means = [_as_float(row.get("f1_micro_tuned_mean"), 0.0) for row in summary_rows]
    stds = [_as_float(row.get("f1_micro_tuned_std"), 0.0) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(max(9, len(summary_rows) * 1.5), 5))
    x = list(range(len(summary_rows)))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Ablation: Tuned F1 Micro (mean +/- std)")
    ax.set_ylabel("F1 Micro")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
