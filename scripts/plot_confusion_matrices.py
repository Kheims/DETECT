"""Generate per-label confusion matrices for all model families.

Reads:
  - Classical: artifacts/runs/<fp>/<model>_results.json (best combo, aggregated tp/fp/fn/tn across folds/seeds)
  - Sequence: artifacts/runs/<fp>/<model>_results.json (per_seed → per_label → tp/fp/fn/tn)
  - GNN: artifacts/reports/ase2026_ablation_*/records.json (per run → per_label_tuned → tp/fp/fn/tn)

Usage:
    uv run python scripts/plot_confusion_matrices.py \
        --artifacts-root artifacts \
        --output-dir results/confusion_matrices
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LABEL_NAMES = ["is_feature_envy", "is_long_method", "is_blob", "is_data_class"]
LABEL_SHORT = ["FE", "LM", "Blob", "DC"]


def sum_confusion_from_seeds(per_seed_list):
    """Sum tp/fp/fn/tn across seeds for each label."""
    totals = {}
    for label in LABEL_NAMES:
        tp = sum(s["per_label"][label]["tp"] for s in per_seed_list)
        fp = sum(s["per_label"][label]["fp"] for s in per_seed_list)
        fn = sum(s["per_label"][label]["fn"] for s in per_seed_list)
        tn = sum(s["per_label"][label]["tn"] for s in per_seed_list)
        totals[label] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return totals


def load_sequence_confusion(artifacts_root):
    """Load confusion data from sequence result JSONs."""
    models = {}
    runs_dir = artifacts_root / "runs"
    for f in runs_dir.glob("*/*_results.json"):
        with f.open() as fh:
            data = json.load(fh)
        model = data.get("model", "")
        if "per_seed" not in data:
            continue
        # Check first seed has tp
        first = data["per_seed"][0]
        if "per_label" not in first:
            continue
        first_label = list(first["per_label"].keys())[0]
        if "tp" not in first["per_label"][first_label]:
            continue
        # Keep latest by mtime
        mtime = f.stat().st_mtime
        if model in models and models[model][0] >= mtime:
            continue
        models[model] = (mtime, sum_confusion_from_seeds(data["per_seed"]))
    return {name: conf for name, (_, conf) in models.items()}


def load_classical_confusion(artifacts_root):
    """Load confusion data from classical result JSONs (per-fold tp/fp/fn/tn in best combo)."""
    models = {}
    runs_dir = artifacts_root / "runs"
    for f in runs_dir.glob("*/*_results.json"):
        with f.open() as fh:
            data = json.load(fh)
        model = data.get("model", "")
        if "best" not in data or not data["best"]:
            continue
        results = data["best"].get("results", {})
        pl = results.get("per_label", {})
        if not pl:
            continue
        first_label = list(pl.keys())[0]
        if "tp" not in pl[first_label]:
            continue
        # This is classical (has "best" with per_label tp)
        mtime = f.stat().st_mtime
        if model in models and models[model][0] >= mtime:
            continue
        totals = {}
        for label in LABEL_NAMES:
            entry = pl.get(label, {})
            totals[label] = {
                "tp": entry.get("tp", 0),
                "fp": entry.get("fp", 0),
                "fn": entry.get("fn", 0),
                "tn": entry.get("tn", 0),
            }
        models[model] = (mtime, totals)
    return {name: conf for name, (_, conf) in models.items()}


def load_gnn_confusion(artifacts_root):
    """Load confusion data from GNN ablation records.json."""
    models = {}
    for records_path in artifacts_root.glob("reports/*/records.json"):
        with records_path.open() as fh:
            records = json.load(fh)

        # Group by architecture (Child-only, no NextToken)
        from collections import defaultdict
        arch_runs = defaultdict(list)
        for r in records:
            combo = r.get("combo", {})
            edge_types = combo.get("dataset.edge_types", [])
            if "NextToken" in edge_types:
                continue
            arch = combo.get("training.architecture", "?")
            if "per_label_tuned" in r:
                arch_runs[arch].append(r["per_label_tuned"])

        for arch, runs in arch_runs.items():
            totals = {}
            for label in LABEL_NAMES:
                tp = sum(r[label]["tp"] for r in runs if label in r)
                fp = sum(r[label]["fp"] for r in runs if label in r)
                fn = sum(r[label]["fn"] for r in runs if label in r)
                tn = sum(r[label]["tn"] for r in runs if label in r)
                totals[label] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
            models[arch] = totals

    return models


MODEL_DISPLAY = {
    "random_forest": "RF", "svm": "SVM", "xgboost": "XGBoost",
    "decision_tree": "DT", "knn": "KNN",
    "lstm": "LSTM", "bilstm": "BiLSTM",
    "bilstm_attention": "BiLSTM+Attn", "cnn": "CNN",
    "gcn": "GCN", "gat": "GAT", "graphsage": "GraphSAGE",
}

FAMILY_ORDER = {
    "random_forest": 0, "svm": 0, "xgboost": 0, "decision_tree": 0, "knn": 0,
    "lstm": 1, "bilstm": 1, "bilstm_attention": 1, "cnn": 1,
    "gcn": 2, "gat": 2, "graphsage": 2,
}


def plot_confusion_grid(all_models, out_dir):
    """Plot a grid: rows = models, cols = labels. Each cell is a 2x2 confusion matrix."""
    # Sort by family then name
    sorted_models = sorted(all_models.keys(),
                           key=lambda m: (FAMILY_ORDER.get(m, 9), m))

    n_models = len(sorted_models)
    n_labels = len(LABEL_SHORT)

    fig, axes = plt.subplots(n_models, n_labels,
                             figsize=(n_labels * 2.5, n_models * 2.2))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for i, model in enumerate(sorted_models):
        conf = all_models[model]
        for j, (label, short) in enumerate(zip(LABEL_NAMES, LABEL_SHORT)):
            ax = axes[i, j]
            d = conf.get(label, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            matrix = np.array([[d["tn"], d["fp"]], [d["fn"], d["tp"]]])
            total = matrix.sum()
            normed = matrix / total if total > 0 else matrix

            ax.imshow(normed, cmap="Blues", vmin=0, vmax=1)
            for r in range(2):
                for c in range(2):
                    val = matrix[r, c]
                    color = "white" if normed[r, c] > 0.5 else "black"
                    ax.text(c, r, f"{val}", ha="center", va="center",
                            fontsize=10, color=color, fontweight="bold")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=7)
            ax.set_yticklabels(["True 0", "True 1"], fontsize=7)

            if i == 0:
                ax.set_title(short, fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(MODEL_DISPLAY.get(model, model),
                              fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(str(out_dir / f"confusion_matrices_grid{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir}/confusion_matrices_grid.pdf/.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--output-dir",
                        default="results")
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root).resolve()
    out_dir = Path(args.output_dir)

    print("Loading sequence results...")
    seq = load_sequence_confusion(artifacts_root)
    print(f"  Found: {sorted(seq.keys())}")

    print("Loading classical results...")
    classical = load_classical_confusion(artifacts_root)
    print(f"  Found: {sorted(classical.keys())}")

    print("Loading GNN results...")
    gnn = load_gnn_confusion(artifacts_root)
    print(f"  Found: {sorted(gnn.keys())}")

    all_models = {**classical, **seq, **gnn}
    print(f"\nTotal: {len(all_models)} models")

    if all_models:
        plot_confusion_grid(all_models, out_dir)


if __name__ == "__main__":
    main()
