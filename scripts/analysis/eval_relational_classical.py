"""Re-run the classical benchmark with relational features added.

Same protocol as scripts/train_classical.py. Compares metrics only, metrics
plus level 0 (snippet), metrics plus level 1 (enclosing file) and both.

Usage:
    uv run python scripts/analysis/eval_relational_classical.py \
        --metrics artifacts_main/metrics_dataset.csv \
        --relational data/relational_features.csv \
        --out artifacts/relational_classical
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.train_classical import run_classical  # noqa: E402

BEST_PARAMS = {
    "xgboost": {"n_estimators": 100, "max_depth": 5, "lr": 0.05},
    "random_forest": {"n_estimators": 500, "max_depth": 10},
    "decision_tree": {"max_depth": 5},
}
LABELS = ["y_fe", "y_lm", "y_blob", "y_dc"]


def merged_csv(metrics_path, relational_path, prefixes, out_path):
    rel = {r["sample_idx"]: r for r in csv.DictReader(open(relational_path))}
    with open(metrics_path) as f:
        rows = list(csv.DictReader(f))
    metric_cols = [c for c in rows[0] if c not in LABELS and c != "sample_idx"]
    rel_cols = [c for c in next(iter(rel.values())) if any(c.startswith(p) for p in prefixes)]
    if prefixes:
        rel_cols += ["is_method"]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx"] + metric_cols + rel_cols + LABELS)
        for r in rows:
            extra = rel[r["sample_idx"]]
            writer.writerow([r["sample_idx"]] + [r[c] for c in metric_cols]
                            + [extra[c] for c in rel_cols] + [r[c] for c in LABELS])
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="artifacts_main/metrics_dataset.csv")
    parser.add_argument("--relational", default="data/relational_features.csv")
    parser.add_argument("--out", default="artifacts/relational_classical")
    parser.add_argument("--models", default="xgboost,random_forest,decision_tree")
    parser.add_argument("--feature-sets", default="metrics,metrics_l0,metrics_l1,metrics_l0_l1")
    parser.add_argument("--cv-group-json", default=None, help="normalized JSON for project-grouped folds")
    parser.add_argument("--cv-group-key", default="repo_url")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    feature_sets = {"metrics": [], "metrics_l0": ["l0_"], "metrics_l1": ["l1_"], "metrics_l0_l1": ["l0_", "l1_"]}
    summary = {}
    wanted = set(args.feature_sets.split(","))
    for fs_name, prefixes in feature_sets.items():
        if fs_name not in wanted:
            continue
        csv_path = merged_csv(args.metrics, args.relational, prefixes, out / f"{fs_name}.csv")
        for model in args.models.split(","):
            cfg = {"model": model, "seeds": [42, 43, 44, 45, 46], "n_folds": 5,
                   "cross_validation": True, "threshold_tuning": True,
                   "model_params": BEST_PARAMS[model],
                   "cv_group_json": args.cv_group_json, "cv_group_key": args.cv_group_key}
            res = run_classical(cfg, str(csv_path), str(out), run_name=f"{model}__{fs_name}")
            agg = res["best"]["results"]
            summary[f"{model}__{fs_name}"] = {
                "f1_macro": agg["f1_macro"], "mcc_macro": agg["mcc_macro"],
                "per_label": {k: {m: agg["per_label"][k][m] for m in ("f1", "mcc")}
                              for k in agg["per_label"]},
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("\n=== Feature Envy MCC (mean ± std over 5 seeds) ===")
    for key, v in summary.items():
        fe = v["per_label"]["is_feature_envy"]["mcc"]
        lm = v["per_label"]["is_long_method"]["mcc"]
        print(f"{key:36s} FE {fe['mean']:.3f} ± {fe['std']:.3f}   LM {lm['mean']:.3f}   "
              f"macro F1 {v['f1_macro']['mean']:.3f}")


if __name__ == "__main__":
    main()
