"""Check whether detector scores follow the severity given by MLCQ reviewers.

Among the positives of each smell: Spearman correlation between score and
severity, and ROC-AUC of major or critical against minor. Raw size is a
reference row.

Usage:
    python scripts/labels/severity_tracking.py --config config/experiments/size_controlled.yml
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from scripts.analysis.size_controlled import load_test_scores, resolve  # noqa: E402

RANK = {"minor": 1, "major": 2, "critical": 3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args, extra = parser.parse_known_args()
    cfg = load_config(args.config, parse_cli_overrides(extra))
    scfg, tcfg = cfg["size_controlled"], cfg["severity_tracking"]
    ds = scfg["datasets"][tcfg["dataset"]]
    folds = json.loads(resolve(ds["folds"]).read_text())
    labels = list(folds["meta"]["label_order"])
    meta = {int(r["json_index"]): r for r in csv.DictReader(resolve(ds["level_csv"]).open())}
    size = {int(r[ds["size_key"]]): float(r[ds["size_column"]]) for r in csv.DictReader(resolve(ds["size_csv"]).open())}
    scores = {name: load_test_scores(list(spec["predictions"]), scfg["strategy"], len(labels))
              for name, spec in ds["families"].items()}
    seeds = sorted(next(iter(scores.values())))
    # raw size as a reference "family": same score for every seed
    scores["raw_loc"] = {seed: {i: np.full(len(labels), size[i]) for i in size} for seed in seeds}
    results: dict[str, dict] = {}
    for col, label in enumerate(labels):
        short = tcfg["meta_short"][label]
        pos = {i for i, r in meta.items() if r[f"y_{short}"] == "1" and r[f"severity_{short}"] in RANK}
        out = {}
        for name, by_seed in scores.items():
            vals = defaultdict(list)
            for seed in seeds:
                idx = sorted(i for i in pos if i in by_seed.get(seed, {}))
                if len(idx) < 20:
                    continue
                sev = np.array([RANK[meta[i][f"severity_{short}"]] for i in idx])
                s = np.array([by_seed[seed][i][col] for i in idx])
                vals["spearman"].append(spearmanr(s, sev).statistic)
                vals["auc_major_vs_minor"].append(roc_auc_score((sev >= 2).astype(int), s))
                vals["n"].append(len(idx))
            out[name] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                         for k, v in vals.items()}
        results[label] = out
    path = resolve(tcfg["out"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1))
    for label, fams in results.items():
        print(label)
        for name, m in fams.items():
            if m:
                print(f"  {name:<22} spearman {m['spearman']['mean']:+.3f}  auc major+ vs minor "
                      f"{m['auc_major_vs_minor']['mean']:.3f}  (n {m['n']['mean']:.0f})")
    print(f"[severity] wrote {path}")


if __name__ == "__main__":
    main()
