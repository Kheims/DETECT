"""Train genetic algorithm search-based model on OO metrics for multi-label code smell detection.

Usage:
    python scripts/train_genetic.py --config config/experiments/genetic_search.yml
"""

import sys
import os
import json
import csv
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, matthews_corrcoef, precision_recall_curve, auc

from mlcq_graphs.config import load_config, parse_cli_overrides
from mlcq_graphs.models.genetic import GeneticSmellDetector

LABEL_NAMES = ["is_feature_envy", "is_long_method", "is_blob", "is_data_class"]


def load_metrics_dataset(csv_path):
    X_rows = []
    y_rows = []
    feature_names = None

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if feature_names is None:
                feature_names = [k for k in row.keys()
                                 if k not in ("sample_idx", "sample_id", "y_fe", "y_lm", "y_blob", "y_dc")]
            features = [float(row.get(k, 0)) for k in feature_names]
            labels = [int(float(row["y_fe"])), int(float(row["y_lm"])),
                      int(float(row["y_blob"])), int(float(row["y_dc"]))]
            X_rows.append(features)
            y_rows.append(labels)

    return np.array(X_rows), np.array(y_rows), feature_names


def evaluate_multilabel(y_true, y_pred):
    results = {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }
    per_label = {}
    for i, name in enumerate(LABEL_NAMES):
        per_label[name] = {
            "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "mcc": matthews_corrcoef(y_true[:, i], y_pred[:, i]) if y_true[:, i].sum() > 0 else 0.0,
        }
    results["per_label"] = per_label
    results["mcc_macro"] = np.mean([v["mcc"] for v in per_label.values()])
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, remaining = parser.parse_known_args()

    overrides = parse_cli_overrides(remaining)
    cfg = load_config(Path(args.config), overrides)
    training_cfg = cfg.get("training", {})
    metrics_csv = cfg.get("dataset", {}).get("metrics_csv", "artifacts/metrics_dataset.csv")
    seeds = training_cfg.get("seeds", [42, 43, 44, 45, 46])

    print(f"Loading metrics dataset from {metrics_csv}...")
    X, y, feature_names = load_metrics_dataset(metrics_csv)

    # Safety check for NaN/Inf
    nan_mask = np.isnan(X) | np.isinf(X)
    if nan_mask.any():
        print(f"Warning: {nan_mask.sum()} NaN/Inf values found, replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        print("Data clean: no NaN/Inf values")

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Label distribution: {dict(zip(LABEL_NAMES, y.sum(axis=0).astype(int).tolist()))}")

    params = training_cfg.get("model_params", {})
    train_ratio = training_cfg.get("train_ratio", 0.8)
    all_results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=(1 - train_ratio), random_state=seed
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        detector = GeneticSmellDetector(
            num_metrics=X_train_s.shape[1],
            num_labels=4,
            population_size=params.get("population_size", 100),
            generations=params.get("generations", 200),
            mutation_rate=params.get("mutation_rate", 0.1),
            crossover_rate=params.get("crossover_rate", 0.7),
            tournament_size=params.get("tournament_size", 5),
            elite_size=params.get("elite_size", 5),
            seed=seed,
        )

        detector.fit(X_train_s, y_train)
        y_pred_test = detector.predict(X_test_s)
        result = evaluate_multilabel(y_test, y_pred_test)
        all_results.append(result)

        print(f"  F1-macro: {result['f1_macro']:.4f}, MCC-macro: {result['mcc_macro']:.4f}")

    # Aggregate
    print(f"\n=== Aggregated ({len(seeds)} seeds) ===")
    for metric in ["f1_macro", "mcc_macro"]:
        values = [r[metric] for r in all_results]
        print(f"  {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    for name in LABEL_NAMES:
        f1s = [r["per_label"][name]["f1"] for r in all_results]
        mccs = [r["per_label"][name]["mcc"] for r in all_results]
        print(f"  {name}: F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}, MCC={np.mean(mccs):.4f}±{np.std(mccs):.4f}")

    # Save
    output_dir = cfg.get("run", {}).get("artifacts_root", "artifacts")
    os.makedirs(output_dir, exist_ok=True)
    agg = {
        "seeds": seeds,
        "per_seed": all_results,
        "aggregated": {
            "f1_macro": {"mean": float(np.mean([r["f1_macro"] for r in all_results])),
                         "std": float(np.std([r["f1_macro"] for r in all_results]))},
            "mcc_macro": {"mean": float(np.mean([r["mcc_macro"] for r in all_results])),
                          "std": float(np.std([r["mcc_macro"] for r in all_results]))},
        }
    }
    results_path = os.path.join(output_dir, "genetic_results.json")
    with open(results_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
