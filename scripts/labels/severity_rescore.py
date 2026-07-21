"""MLCQ detection scores when the minor positives are left out.

Outputs in out_dir: severity_rescore.csv and severity_rescore.png.

Usage:
    python scripts/labels/severity_rescore.py --config config/experiments/mlcq_eligible_all.yml
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlcq_graphs.config import load_config  # noqa: E402
from mlcq_graphs.evaluation.cv import tune_thresholds  # noqa: E402

LABELS = ["is_feature_envy", "is_long_method", "is_blob", "is_data_class"]
KEYS = {"is_feature_envy": "feature envy", "is_long_method": "long method", "is_blob": "blob",
        "is_data_class": "data class"}
LEVEL = {"is_feature_envy": "method", "is_long_method": "method", "is_blob": "class", "is_data_class": "class"}
NAMES = {"is_feature_envy": "Feature Envy", "is_long_method": "Long Method", "is_blob": "Blob",
         "is_data_class": "Data Class"}


def score(rows: list[dict], keep: dict[str, np.ndarray], strategy: str) -> dict[tuple[str, str], dict]:
    groups = defaultdict(lambda: {"val": [], "test": []})
    for r in rows:
        if r["strategy"] == strategy and r["part"] in ("val", "test"):
            groups[(r["seed"], r["fold"])][r["part"]].append(r)
    per_seed = defaultdict(lambda: defaultdict(list))
    for (seed, _), g in groups.items():
        for part in ("val", "test"):
            g[part] = (np.array([r["json_index"] for r in g[part]]), np.array([r["y"] for r in g[part]]),
                       np.array([r["prob"] for r in g[part]]))
        for k, label in enumerate(LABELS):
            for setting, mask in keep.items():
                iv, yv, pv = g["val"]
                it, yt, pt = g["test"]
                mv, mt = mask[label][iv], mask[label][it]
                t = tune_thresholds(yv[mv][:, [k]], pv[mv][:, [k]], 101)[0]
                y_, pred = yt[mt, k], pt[mt, k] >= t
                per_seed[(label, setting)][seed].append(
                    (matthews_corrcoef(y_, pred), f1_score(y_, pred, zero_division=0),
                     precision_score(y_, pred, zero_division=0), recall_score(y_, pred, zero_division=0),
                     int(y_.sum()), len(y_)))
    out = {}
    for key, seeds in per_seed.items():
        means = np.array([np.mean(v, axis=0) for v in seeds.values()])
        out[key] = {"mcc": means[:, 0].mean(), "mcc_sd": means[:, 0].std(ddof=1) if len(means) > 1 else 0.0,
                    "f1": means[:, 1].mean(), "precision": means[:, 2].mean(), "recall": means[:, 3].mean(),
                    "test_positives": means[:, 4].mean(), "test_n": means[:, 5].mean(), "seeds": len(means)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", action="append", default=[], help="name=glob of predictions.jsonl")
    ap.add_argument("--strategy", default="kfold")
    ap.add_argument("--out", default="artifacts/analysis/severity_rescore")
    args = ap.parse_args()
    cfg = load_config(Path(args.config), {})["eligible_eval"]
    inputs = [(i["name"], i["predictions"]) for i in cfg["inputs"]] + [tuple(s.split("=", 1)) for s in args.input]
    data = json.loads((ROOT / "data/MLCQCodeSmellSamples.normalized.json").read_text())
    with (ROOT / "data/derived/mlcq_sample_meta.csv").open() as f:
        level = {int(r["json_index"]): r["level"] for r in csv.DictReader(f)}
    n = len(data)
    eligible = {lb: np.array([level.get(i) == LEVEL[lb] for i in range(n)]) for lb in LABELS}
    minor = {lb: np.array([data[i]["y"][LABELS.index(lb)] == 1 and data[i]["labels"][KEYS[lb]]["severity"] == "minor"
                           for i in range(n)]) for lb in LABELS}
    keep = {"all positives": eligible, "minor positives left out": {lb: eligible[lb] & ~minor[lb] for lb in LABELS}}
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    table = []
    for name, pattern in inputs:
        files = sorted(glob.glob(str(ROOT / pattern)))
        rows = [json.loads(line) for f in files for line in open(f)]
        res = score(rows, keep, args.strategy)
        for (label, setting), m in sorted(res.items()):
            table.append({"model": name, "smell": label, "setting": setting, **m})
        print(f"[severity] {name}: " + ", ".join(
            f"{NAMES[lb]} {res[(lb, 'all positives')]['mcc']:.3f}->{res[(lb, 'minor positives left out')]['mcc']:.3f}"
            for lb in LABELS))
    with (out / "severity_rescore.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(r["model"] for r in table))
    fig, axes = plt.subplots(1, 4, figsize=(17, 0.38 * len(models) + 1.8), sharey=True)
    for ax, label in zip(axes, LABELS):
        for i, mname in enumerate(models):
            a = next(r for r in table if r["model"] == mname and r["smell"] == label and r["setting"] == "all positives")
            b = next(r for r in table if r["model"] == mname and r["smell"] == label
                     and r["setting"] == "minor positives left out")
            ax.plot([a["mcc"], b["mcc"]], [i, i], color="#a0aec0", lw=1.5)
            ax.scatter([a["mcc"]], [i], color="#718096", s=25, zorder=3)
            ax.scatter([b["mcc"]], [i], color="#2b6cb0", s=25, zorder=3)
        ax.set_title(NAMES[label], fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_xlabel("MCC")
        ax.grid(axis="x", color="#edf2f7")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(range(len(models)), models, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].scatter([], [], color="#718096", label="all positives")
    axes[0].scatter([], [], color="#2b6cb0", label="minor positives left out")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("MLCQ: MCC per smell with all positives (grey) and with the positives rated minor left out (blue); "
                 "each smell on the samples of its level", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "severity_rescore.png", dpi=150)
    fig.savefig(out / "severity_rescore.pdf")
    plt.close(fig)
    print(f"[severity] wrote {out}")


if __name__ == "__main__":
    main()
