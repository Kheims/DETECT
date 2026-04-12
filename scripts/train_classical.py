"""Train classical ML models on OO metrics for multi-label code smell detection.

Supports:
- Multiple seeds for statistical robustness
- Hyperparameter grid search
- Stratified multi-label splitting
- Reports mean ± std across runs

Usage:
    python scripts/train_classical.py --config config/experiments/rf_metrics.yml
"""

import sys
import os
import json
import csv
import itertools
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, matthews_corrcoef, precision_recall_curve, auc

from mlcq_graphs.config import load_config, parse_cli_overrides
from mlcq_graphs.models.classical import build_classical_model

LABEL_NAMES = ["is_feature_envy", "is_long_method", "is_blob", "is_data_class"]


def load_metrics_dataset(csv_path):
    """Load metrics CSV into X (features) and y (multi-label)."""
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


def multilabel_stratify_column(y):
    """Create a single stratification column from multi-label y.
    Encodes each unique label combination as an integer."""
    return np.array([hash(tuple(row)) % (2**31) for row in y])


def evaluate_multilabel(y_true, y_pred, y_proba=None):
    """Compute multi-label metrics."""
    results = {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }

    per_label = {}
    for i, name in enumerate(LABEL_NAMES):
        tp = int(((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum())
        fp = int(((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum())
        fn = int(((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum())
        tn = int(((y_pred[:, i] == 0) & (y_true[:, i] == 0)).sum())
        label_metrics = {
            "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "mcc": matthews_corrcoef(y_true[:, i], y_pred[:, i]) if y_true[:, i].sum() > 0 else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
        if y_proba is not None and y_true[:, i].sum() > 0:
            try:
                precision_arr, recall_arr, _ = precision_recall_curve(y_true[:, i], y_proba[:, i])
                label_metrics["pr_auc"] = auc(recall_arr, precision_arr)
            except Exception:
                label_metrics["pr_auc"] = 0.0
        per_label[name] = label_metrics

    results["per_label"] = per_label
    results["mcc_macro"] = np.mean([v["mcc"] for v in per_label.values()])
    if y_proba is not None:
        pr_aucs = [v.get("pr_auc", 0) for v in per_label.values() if "pr_auc" in v]
        results["pr_auc_macro"] = np.mean(pr_aucs) if pr_aucs else 0.0

    return results


def tune_thresholds(y_true, y_proba, steps=101):
    """Tune per-label decision thresholds on validation set to maximize F1."""
    num_labels = y_true.shape[1]
    best_thresholds = np.full(num_labels, 0.5)
    for i in range(num_labels):
        best_f1 = -1
        for t in np.linspace(0, 1, steps):
            preds = (y_proba[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[i] = t
    return best_thresholds


def run_single_fold(X_train, y_train, X_test, y_test, model_name, model_params,
                    use_threshold_tuning=True):
    """Train and evaluate a single model on a single fold."""
    scaler = StandardScaler()

    if use_threshold_tuning:
        # Split train into train_inner + val for threshold tuning
        from sklearn.model_selection import train_test_split
        seed = model_params.get("seed", 42)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=seed
        )
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        model = build_classical_model(model_name, model_params)
        model.fit(X_tr_s, y_tr)

        # Get probabilities for threshold tuning
        try:
            y_proba_val = np.column_stack([est.predict_proba(X_val_s)[:, 1]
                                           for est in model.estimators_])
            thresholds = tune_thresholds(y_val, y_proba_val)
        except Exception:
            thresholds = np.full(y_train.shape[1], 0.5)

        # Apply tuned thresholds on test
        try:
            y_proba_test = np.column_stack([est.predict_proba(X_test_s)[:, 1]
                                            for est in model.estimators_])
            y_pred = (y_proba_test >= thresholds).astype(int)
        except Exception:
            y_pred = model.predict(X_test_s)
            y_proba_test = None
    else:
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = build_classical_model(model_name, model_params)
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        try:
            y_proba_test = np.column_stack([est.predict_proba(X_test_s)[:, 1]
                                            for est in model.estimators_])
        except Exception:
            y_proba_test = None
        thresholds = np.full(y_train.shape[1], 0.5)

    result = evaluate_multilabel(y_test, y_pred, y_proba_test)
    result["thresholds"] = thresholds.tolist()
    return result


def aggregate_results(all_results):
    """Aggregate results from multiple runs into mean ± std."""
    agg = {}
    for metric in ["f1_macro", "f1_micro", "mcc_macro", "pr_auc_macro"]:
        values = [r.get(metric, 0) for r in all_results]
        agg[metric] = {"mean": np.mean(values), "std": np.std(values)}

    agg["per_label"] = {}
    for name in LABEL_NAMES:
        label_agg = {}
        for metric in ["f1", "mcc", "pr_auc"]:
            values = [r["per_label"][name].get(metric, 0) for r in all_results
                      if name in r.get("per_label", {})]
            if values:
                label_agg[metric] = {"mean": np.mean(values), "std": np.std(values)}
        agg["per_label"][name] = label_agg

    return agg


def expand_param_grid(grid):
    """Expand a parameter grid dict into a list of param combinations."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def run_classical(
    training_cfg: dict,
    metrics_csv: str,
    output_dir: str,
    run_name: str | None = None,
) -> dict:
    """Run classical ML training from a training config and metrics CSV.

    Importable entry point for pipeline orchestration.

    Returns a dict with keys: model, seeds, best, all_combos, results_path.
    """
    model_name = training_cfg.get("model", "random_forest")

    seeds = training_cfg.get("seeds", [42, 43, 44, 45, 46])
    n_folds = training_cfg.get("n_folds", 5)
    use_cv = training_cfg.get("cross_validation", True)
    use_threshold_tuning = training_cfg.get("threshold_tuning", True)
    param_grid = training_cfg.get("param_grid", None)

    print(f"Loading metrics dataset from {metrics_csv}...")
    X, y, feature_names = load_metrics_dataset(metrics_csv)

    nan_mask = np.isnan(X) | np.isinf(X)
    if nan_mask.any():
        nan_count = nan_mask.sum()
        print(f"Warning: {nan_count} NaN/Inf values found, replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        print("Data clean: no NaN/Inf values")

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} labels")
    print(f"Label distribution: {dict(zip(LABEL_NAMES, y.sum(axis=0).astype(int).tolist()))}")

    if param_grid:
        param_combos = expand_param_grid(param_grid)
        print(f"Hyperparameter grid: {len(param_combos)} combinations")
    else:
        param_combos = [training_cfg.get("model_params", {})]

    best_combo = None
    best_score = -1
    all_combo_results = []

    for combo_idx, params in enumerate(param_combos):
        model_params = {**training_cfg.get("model_params", {}), **params}

        if len(param_combos) > 1:
            print(f"\n--- Combo {combo_idx+1}/{len(param_combos)}: {params} ---")

        combo_results = []

        for seed in seeds:
            model_params["seed"] = seed
            np.random.seed(seed)

            if use_cv:
                strat_col = multilabel_stratify_column(y)
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

                fold_results = []
                for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, strat_col)):
                    result = run_single_fold(
                        X[train_idx], y[train_idx],
                        X[test_idx], y[test_idx],
                        model_name, model_params,
                        use_threshold_tuning=use_threshold_tuning
                    )
                    fold_results.append(result)

                avg_result = {
                    "f1_macro": np.mean([r["f1_macro"] for r in fold_results]),
                    "f1_micro": np.mean([r["f1_micro"] for r in fold_results]),
                    "mcc_macro": np.mean([r["mcc_macro"] for r in fold_results]),
                    "per_label": {},
                }
                for name in LABEL_NAMES:
                    avg_result["per_label"][name] = {
                        "f1": np.mean([r["per_label"][name]["f1"] for r in fold_results]),
                        "mcc": np.mean([r["per_label"][name]["mcc"] for r in fold_results]),
                    }
                combo_results.append(avg_result)
            else:
                from sklearn.model_selection import train_test_split
                train_ratio = training_cfg.get("train_ratio", 0.8)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=(1 - train_ratio), random_state=seed
                )
                result = run_single_fold(X_train, y_train, X_test, y_test, model_name, model_params,
                                        use_threshold_tuning=use_threshold_tuning)
                combo_results.append(result)

        agg = aggregate_results(combo_results)
        score = agg["f1_macro"]["mean"]

        print(f"  F1-macro: {score:.4f} ± {agg['f1_macro']['std']:.4f}")
        print(f"  MCC-macro: {agg['mcc_macro']['mean']:.4f} ± {agg['mcc_macro']['std']:.4f}")
        for name in LABEL_NAMES:
            pl = agg["per_label"][name]
            print(f"  {name}: F1={pl['f1']['mean']:.4f}±{pl['f1']['std']:.4f}, "
                  f"MCC={pl['mcc']['mean']:.4f}±{pl['mcc']['std']:.4f}")

        all_combo_results.append({"params": params, "results": agg, "score": score})

        if score > best_score:
            best_score = score
            best_combo = {"params": params, "results": agg}

    if len(param_combos) > 1:
        print(f"\n=== Best configuration ===")
        print(f"Params: {best_combo['params']}")
        agg = best_combo["results"]
        print(f"F1-macro: {agg['f1_macro']['mean']:.4f} ± {agg['f1_macro']['std']:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    save_data = {
        "model": model_name,
        "seeds": seeds,
        "n_folds": n_folds if use_cv else "N/A",
        "cross_validation": use_cv,
        "best": best_combo,
        "all_combos": all_combo_results,
    }
    filename = f"{run_name}.json" if run_name else f"{model_name}_results.json"
    results_path = os.path.join(output_dir, filename)
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    save_data["results_path"] = results_path
    return save_data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, remaining = parser.parse_known_args()

    overrides = parse_cli_overrides(remaining)
    cfg = load_config(Path(args.config), overrides)
    training_cfg = cfg.get("training", {})
    metrics_csv = cfg.get("dataset", {}).get("metrics_csv", "artifacts/metrics_dataset.csv")
    output_dir = cfg.get("run", {}).get("artifacts_root", "artifacts")

    run_classical(
        training_cfg=training_cfg,
        metrics_csv=metrics_csv,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
