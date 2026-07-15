"""Convert SmellyCode++ to DETECT's normalized JSON format.

Drops projects and exact snippets shared with the reference dataset, then caps
positives per label and samples negatives.

Usage:
    python scripts/data/convert_smellycode.py --csv data/transfer/smellycode_pp/smellycode_pp.csv \\
        --reference data/MLCQCodeSmellSamples.normalized.json \\
        --max-positives-per-label 1000 --negatives 3000 --seed 0 \\
        --out data/transfer/smellycode_pp/smellycode.normalized.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.constants import LABEL_ORDER

# SmellyCode++ column for each DETECT label
COLUMNS = {
    "is_feature_envy": "Feature envy",
    "is_long_method": "Long method",
    "is_blob": "God class",
    "is_data_class": "Data class",
}


def repo_name(url: str) -> str:
    name = url.rstrip("/").split("/")[-1].lower()
    return name[:-4] if name.endswith(".gi" + "t") else name


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--max-positives-per-label", type=int, default=1000)
    parser.add_argument("--negatives", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    reference = json.loads(args.reference.read_text())
    ref_projects = {repo_name(str(e.get("repo_url", ""))) for e in reference}
    ref_code = {squash(str(e.get("code_snippet", ""))) for e in reference}

    stats = Counter()
    rows: list[dict] = []
    with args.csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            stats["rows"] += 1
            if row["Project"].lower() in ref_projects:
                stats["dropped_shared_project"] += 1
                continue
            code = row["Code"]
            if squash(code) in ref_code:
                stats["dropped_exact_duplicate"] += 1
                continue
            y = [row[COLUMNS[label]].strip() == "1" for label in LABEL_ORDER]
            rows.append({"project": row["Project"], "file": row["File"], "code": code, "y": y})

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    picked: list[dict] = []
    taken = set()
    for k, label in enumerate(LABEL_ORDER):
        positives = [i for i, r in enumerate(rows) if r["y"][k] and i not in taken]
        for i in positives[: args.max_positives_per_label]:
            taken.add(i)
            picked.append(rows[i])
    negatives = [i for i, r in enumerate(rows) if not any(r["y"]) and i not in taken]
    for i in negatives[: args.negatives]:
        picked.append(rows[i])
    rng.shuffle(picked)

    entries = []
    for r in picked:
        level = "class" if re.match(r"^\s*((public|private|protected|abstract|final|static)\s+)*(class|interface|enum)\b", r["code"]) else "method"
        stats[f"level_{level}"] += 1
        entries.append(
            {
                "repo_url": f"smellycode++/{r['project']}",
                "commit_hash": "",
                "file_path": r["file"],
                "start_line": -1,
                "end_line": -1,
                "code_snippet": r["code"],
                "y": r["y"],
                "level": level,
            }
        )
    label_counts = {label: sum(e["y"][k] for e in entries) for k, label in enumerate(LABEL_ORDER)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(entries))
    report = {
        "source_rows": stats["rows"],
        "dropped_shared_project": stats["dropped_shared_project"],
        "dropped_exact_duplicate": stats["dropped_exact_duplicate"],
        "kept_rows": len(rows),
        "written": len(entries),
        "label_counts": label_counts,
        "levels": {k: v for k, v in stats.items() if k.startswith("level_")},
        "shared_projects": len({r for r in ref_projects}),
    }
    (args.out.with_suffix(".stats.json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
