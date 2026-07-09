"""Merge the records.json of several ablation report directories and print per-configuration means and standard deviations.

Usage:
    python scripts/cv/aggregate_ablation.py artifacts/reports/readout_* --out artifacts/reports/readout_merged
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.constants import LABEL_ORDER


def load_records(report_dirs: list[Path]) -> list[dict]:
    records: list[dict] = []
    for report_dir in report_dirs:
        path = report_dir / "records.json"
        if not path.exists():
            print(f"[aggregate] skip {report_dir}: no records.json", file=sys.stderr)
            continue
        records.extend(json.loads(path.read_text()))
    return records


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), statistics.pstdev(values)


def summarise(records: list[dict], metrics: list[str]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(json.dumps(record["combo"], sort_keys=True), []).append(record)

    rows: list[dict] = []
    for key, items in sorted(grouped.items()):
        row: dict = {"combo": key, "runs": len(items), "seeds": sorted(r["seed"] for r in items)}
        for label in LABEL_ORDER:
            short = label.removeprefix("is_")
            for metric in metrics:
                raw = [r["per_label_tuned"].get(label, {}).get(metric) for r in items]
                # undefined metrics (e.g. MCC without positives) are left out
                vals = [float(v) for v in raw if v is not None]
                row[f"{short}_{metric}_mean"], row[f"{short}_{metric}_std"] = mean_std(vals)
        for field in ("f1_macro_tuned", "mcc_tuned", "pr_auc_tuned", "train_duration_sec"):
            row[f"{field}_mean"], row[f"{field}_std"] = mean_std(
                [float(r[field]) for r in items if r.get(field) is not None]
            )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dirs", nargs="+", type=Path)
    parser.add_argument("--metrics", default="mcc,f1", help="Per-label metrics to report.")
    parser.add_argument("--out", type=Path, default=None, help="Directory for merged records and CSV.")
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    records = load_records(args.report_dirs)
    if not records:
        raise SystemExit("no records found")
    rows = summarise(records, metrics)

    shorts = [label.removeprefix("is_") for label in LABEL_ORDER]
    header = ["combo", "runs"] + [f"{s} {m}" for s in shorts for m in metrics] + ["macroF1", "MCC macro"]
    print(" | ".join(header))
    for row in rows:
        cells = [row["combo"], str(row["runs"])]
        for s in shorts:
            for m in metrics:
                cells.append(f"{row[f'{s}_{m}_mean']:.3f}±{row[f'{s}_{m}_std']:.3f}")
        cells.append(f"{row['f1_macro_tuned_mean']:.3f}±{row['f1_macro_tuned_std']:.3f}")
        cells.append(f"{row['mcc_tuned_mean']:.3f}±{row['mcc_tuned_std']:.3f}")
        print(" | ".join(cells))

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "records.json").write_text(json.dumps(records, indent=2))
        with (args.out / "summary.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[aggregate] wrote {args.out}")


if __name__ == "__main__":
    main()
