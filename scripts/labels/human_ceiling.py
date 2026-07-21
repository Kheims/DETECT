"""Human ceiling on MLCQ: how well one reviewer agrees with the other reviewers.

Pairwise MCC between reviews of the same (sample, smell), compared with the
model scores on the same samples.

Usage:
    python scripts/labels/human_ceiling.py --config config/experiments/mlcq_eligible_all.yml
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlcq_graphs.config import load_config  # noqa: E402
from mlcq_graphs.xai import common  # noqa: E402

from scripts.labels.examiner_checks import fold_predictions, per_fold  # noqa: E402

OUT = ROOT / "artifacts/analysis/human_ceiling"
SEV = {"none": 0, "minor": 1, "major": 2, "critical": 3}
SMELL_OF = {"feature envy": "is_feature_envy", "long method": "is_long_method", "blob": "is_blob",
            "data class": "is_data_class"}


def reviews() -> dict[tuple[int, str], list[tuple[int, int]]]:
    """{(json_index, smell): [(reviewer_id, severity), ...]}"""
    key_of = {}
    for j, s in enumerate(common.samples()):
        key_of[(s["repo_url"], s["commit_hash"], s["file_path"], int(s["start_line"]), int(s["end_line"]))] = j
    out = defaultdict(list)
    missing = 0
    with (ROOT / "data/mlcq_zenodo/MLCQCodeSmellSamples.csv").open() as f:
        for r in csv.DictReader(f, delimiter=";"):
            k = (r["repository"], r["commit_hash"], r["path"], int(r["start_line"]), int(r["end_line"]))
            j = key_of.get(k)
            if j is None:
                missing += 1
                continue
            out[(j, SMELL_OF[r["smell"]])].append((int(r["reviewer_id"]), SEV[r["severity"]]))
    print(f"[ceiling] {sum(len(v) for v in out.values())} reviews matched, {missing} not matched")
    return out


def consensus(sevs: list[int]) -> int:
    return int(np.median(sevs) > 0 or (len(sevs) % 2 == 0 and sorted(sevs)[len(sevs) // 2] > 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", action="append", default=[])
    ap.add_argument("--min_reviews", type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rev = reviews()
    cfg = load_config(Path(args.config), {})["eligible_eval"]
    inputs = [(i["name"], i["predictions"]) for i in cfg["inputs"]] + [tuple(s.split("=", 1)) for s in args.input]
    rows = []
    for smell in common.LABELS:
        pairs = {j: v for (j, s), v in rev.items() if s == smell and len(v) >= args.min_reviews}
        held, others = [], []
        for j, v in pairs.items():
            for a in range(len(v)):
                for b in range(len(v)):
                    if a != b:
                        others.append(int(v[a][1] > 0))
                        held.append(int(v[b][1] > 0))
        held, others = np.array(held), np.array(others)
        full = {j: common.label(j, smell) for j in pairs}
        agree_label = np.mean([consensus([s for _, s in v]) == common.label(j, smell) for j, v in pairs.items()])
        rows.append({"smell": smell, "who": "another reviewer", "samples": len(pairs), "comparisons": len(held),
                     "consensus_positive_share": float(others.mean()), "mcc": float(matthews_corrcoef(others, held)),
                     "f1": float(f1_score(others, held, zero_division=0)),
                     "consensus_equals_dataset_label": float(agree_label)})
        for name, pattern in inputs:
            preds = fold_predictions(pattern)
            ys, ds = [], []
            for key, (idx, y, s, d) in per_fold(preds, smell).items():
                for i, dd in zip(idx, d):
                    if int(i) in full:
                        ys.append(full[int(i)])
                        ds.append(int(dd))
            if len(set(ys)) < 2:
                continue
            rows.append({"smell": smell, "who": name, "samples": len(pairs), "comparisons": len(ys),
                         "consensus_positive_share": float(np.mean(ys)), "mcc": float(matthews_corrcoef(ys, ds)),
                         "f1": float(f1_score(ys, ds, zero_division=0)), "consensus_equals_dataset_label": float("nan")})
        for r in rows[-len(inputs) - 1:]:
            print(f"[ceiling] {common.NAMES[smell]:13s} {r['who']:24s} samples {r['samples']:4d} comparisons "
                  f"{r['comparisons']:5d} pos {r['consensus_positive_share']:.2f} MCC {r['mcc']:.3f} F1 {r['f1']:.3f}")
    with (OUT / "human_ceiling.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[ceiling] wrote {OUT / 'human_ceiling.csv'}")


if __name__ == "__main__":
    main()
