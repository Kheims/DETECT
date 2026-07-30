"""Share of positive reviews per (snippet, smell) on MLCQ, as a soft training target.

- One review per reviewer (the latest), positive when severity is minor or above
- Writes share, reviewer count and positive count per snippet and smell

Usage:
    python scripts/data/build_review_shares.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlcq_graphs.constants import LABEL_ORDER  # noqa: E402

SEVERITY = {"none": 0, "minor": 1, "major": 2, "critical": 3}
SMELL_NAMES = ["feature envy", "long method", "blob", "data class"]


def upper_median(sevs: list[int]) -> int:
    s = sorted(sevs)
    return s[len(s) // 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reviews-csv", type=Path, default=ROOT / "data/mlcq_zenodo/MLCQCodeSmellSamples.csv")
    ap.add_argument("--labels-json", type=Path,
                    default=ROOT / "data/derived/MLCQCodeSmellSamples.normalized.rule_default.json")
    ap.add_argument("--positive", default="minor,major,critical")
    ap.add_argument("--out", type=Path, default=ROOT / "data/derived/mlcq_review_shares.json")
    args = ap.parse_args()
    if len(SMELL_NAMES) != len(LABEL_ORDER):
        raise SystemExit("SMELL_NAMES must follow LABEL_ORDER")
    positive = {SEVERITY[s.strip()] for s in args.positive.split(",")}

    entries = json.loads(args.labels_json.read_text())
    index_of = {}
    for j, e in enumerate(entries):
        key = (e["repo_url"], e["commit_hash"], e["file_path"], int(e["start_line"]), int(e["end_line"]))
        if index_of.setdefault(key, j) != j:
            raise SystemExit(f"json_index {j} and {index_of[key]} share the location {key}")

    # (json_index, smell index) -> reviewer -> [(timestamp, id, severity)]
    records: dict[tuple[int, int], dict[int, list[tuple[str, int, int]]]] = defaultdict(lambda: defaultdict(list))
    unmatched = 0
    with args.reviews_csv.open(newline="") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            key = (r["repository"].strip(), r["commit_hash"].strip(), r["path"].strip(),
                   int(r["start_line"]), int(r["end_line"]))
            j = index_of.get(key)
            if j is None:
                unmatched += 1
                continue
            s = SMELL_NAMES.index(r["smell"].strip())
            records[(j, s)][int(r["reviewer_id"])].append(
                (r["review_timestamp"].strip(), int(r["id"]), SEVERITY[r["severity"].strip()]))

    n = len(entries)
    share = np.zeros((n, 4), dtype=np.float64)
    n_rev = np.zeros((n, 4), dtype=np.int64)
    n_pos = np.zeros((n, 4), dtype=np.int64)
    y = np.array([[int(bool(v)) for v in e["y"][:4]] for e in entries], dtype=np.int64)
    label_mismatch, dedup_mismatch, duplicates = 0, 0, 0
    for (j, s), by_reviewer in records.items():
        every = [sev for recs in by_reviewer.values() for _, _, sev in recs]
        latest = [max(recs)[2] for recs in by_reviewer.values()]
        duplicates += len(every) - len(latest)
        label_mismatch += int((upper_median(every) in positive) != bool(y[j, s]))
        dedup_mismatch += int((upper_median(latest) in positive) != bool(y[j, s]))
        n_rev[j, s] = len(latest)
        n_pos[j, s] = sum(sev in positive for sev in latest)
        share[j, s] = n_pos[j, s] / n_rev[j, s]

    print(f"[shares] {n} snippets, {len(records)} reviewed (snippet, smell) pairs, "
          f"{sum(len(v) for v in records.values())} reviews kept, "
          f"{unmatched} review rows not matched, {duplicates} repeated reviews dropped (latest kept)")
    print(f"[shares] upper median of all records differs from the labels for {label_mismatch} pairs; "
          f"upper median of one review per reviewer differs for {dedup_mismatch} pairs")
    for s, name in enumerate(LABEL_ORDER):
        rated = n_rev[:, s] > 0
        multi = n_rev[:, s] >= 2
        mixed = multi & (share[:, s] > 0) & (share[:, s] < 1)
        print(f"[shares] {name:16s} reviewed {int(rated.sum()):4d} | >=2 reviewers {int(multi.sum()):4d} | "
              f"mixed {int(mixed.sum()):4d} | label positives {int(y[:, s].sum()):3d} | "
              f"mean share among positives {share[y[:, s] == 1, s].mean():.3f}")

    out = {
        "meta": {
            "labels_json": str(args.labels_json.relative_to(ROOT)) if args.labels_json.is_relative_to(ROOT)
            else str(args.labels_json),
            "reviews_csv": str(args.reviews_csv.name),
            "positive": sorted(k for k, v in SEVERITY.items() if v in positive),
            "label_order": LABEL_ORDER,
            "one_review_per_reviewer": "latest",
            "unmatched_review_rows": unmatched,
            "repeated_reviews_dropped": duplicates,
            "label_mismatch_all_records": label_mismatch,
            "label_mismatch_one_per_reviewer": dedup_mismatch,
        },
        "share": share.round(6).tolist(),
        "n_reviewers": n_rev.tolist(),
        "n_positive": n_pos.tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"[shares] wrote {args.out}")


if __name__ == "__main__":
    main()
