"""Normalize MLCQ raw reviews into per-sample aggregated labels.

Aggregates multiple reviewer severities per (sample, smell) pair into a single
median / max / majority severity, then applies a binarization rule to produce
the multi-label y vector in the canonical order [fe, lm, blob, dc].

The tie-breaking rule for median aggregation matches Madeyski and Lewowski
(IST 2023, section 3.3): when the number of reviews is even and the two
middle values differ, the higher severity is used.

Usage:
    python scripts/NormalizeFromJson.py \\
        --input data/MLCQCodeSmellSamples.json \\
        --method median \\
        --rule default

    # Madeyski DS1 preset
    python scripts/NormalizeFromJson.py --rule ds1

    # Custom positive class
    python scripts/NormalizeFromJson.py --rule major,critical
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SEVERITY_ORDER = ["none", "minor", "major", "critical"]
SEVERITY_TO_SCORE = {label: idx for idx, label in enumerate(SEVERITY_ORDER)}

# Canonical order of the y vector. Do not reorder — downstream CSVs and
# training scripts index y[0]=feature envy, y[1]=long method, y[2]=blob,
# y[3]=data class.
Y_ORDER = ["feature envy", "long method", "blob", "data class"]

# Binarization presets: which severity strings count as the positive class.
RULE_PRESETS: dict[str, set[str]] = {
    "default": {"minor", "major", "critical"},  # severity != none (legacy behaviour)
    "ds1":     {"major", "critical"},             # Madeyski DS1
    "ds2":     {"critical"},                       # Madeyski DS2
    "strict":  {"critical"},                       # alias of ds2
}


def aggregate_severity(severities: list[str], method: str = "median") -> str:
    """Reduce a list of severity strings to a single severity string."""
    scores = [SEVERITY_TO_SCORE[s] for s in severities if s in SEVERITY_TO_SCORE]
    if not scores:
        return "none"

    if method == "max":
        score = max(scores)
    elif method == "majority":
        present_votes = sum(s > 0 for s in scores)
        absent_votes = len(scores) - present_votes
        score = 1 if present_votes > absent_votes else 0
    elif method == "median":
        scores.sort()
        # Upper median: for even counts scores[len // 2] picks the higher of
        # the two middle values, matching Madeyski and Lewowski 2023.
        score = scores[len(scores) // 2]
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    return SEVERITY_ORDER[score]


def parse_rule(rule_spec: str) -> set[str]:
    """Resolve a rule spec (preset name or comma-separated severities) to a set."""
    if rule_spec in RULE_PRESETS:
        return set(RULE_PRESETS[rule_spec])
    items = {s.strip() for s in rule_spec.split(",") if s.strip()}
    valid = set(SEVERITY_ORDER)
    invalid = items - valid
    if invalid:
        raise ValueError(
            f"Invalid severity values in rule: {sorted(invalid)}. "
            f"Valid severities: {SEVERITY_ORDER}"
        )
    if not items:
        raise ValueError(f"Empty rule spec: {rule_spec!r}")
    return items


def normalize_json_to_json(
    input_json: Path,
    output_json: Path,
    method: str,
    rule: str,
) -> None:
    positive_set = parse_rule(rule)
    print(f"Aggregation method: {method}")
    print(f"Rule: {rule}  ->  positive severities = {sorted(positive_set)}")

    with open(input_json, "r") as f:
        raw = json.load(f)

    samples: dict[tuple, dict] = {}
    for row in raw:
        key = (
            row["repo_url"],
            row["commit_hash"],
            row["file_path"],
            int(row["start_line"]),
            int(row["end_line"]),
        )

        if key not in samples:
            samples[key] = {
                "repo_url": row["repo_url"],
                "commit_hash": row["commit_hash"],
                "file_path": row["file_path"],
                "start_line": int(row["start_line"]),
                "end_line": int(row["end_line"]),
                "code_snippet": row["code_snippet"],
                "votes": {smell: [] for smell in Y_ORDER},
            }

        samples[key]["votes"].setdefault(row["smell"], []).append(row["severity"])

    normalized: list[dict] = []
    for sample in samples.values():
        aggregated = {}
        for smell, votes in sample["votes"].items():
            agg_severity = aggregate_severity(votes, method=method)
            aggregated[smell] = {
                "severity": agg_severity,
                "present": agg_severity in positive_set,
                "vote_count": len(votes),
            }

        y = [aggregated[smell]["severity"] in positive_set for smell in Y_ORDER]

        normalized.append(
            {
                "repo_url": sample["repo_url"],
                "commit_hash": sample["commit_hash"],
                "file_path": sample["file_path"],
                "start_line": sample["start_line"],
                "end_line": sample["end_line"],
                "code_snippet": sample["code_snippet"],
                "labels": aggregated,
                "y": y,
            }
        )

    total = len(normalized)
    print(f"Normalized {total} samples")
    for i, smell in enumerate(Y_ORDER):
        pos = sum(1 for n in normalized if n["y"][i])
        pct = 100 * pos / total if total else 0.0
        print(f"  {smell:<14} positive={pos} ({pct:.1f}%)")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(normalized, f, indent=4)
    print(f"Wrote {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/MLCQCodeSmellSamples.json"),
        help="Raw MLCQ reviews JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output normalized JSON path. Default: "
            "data/MLCQCodeSmellSamples.normalized.<rule>.json"
        ),
    )
    parser.add_argument(
        "--method",
        choices=["median", "max", "majority"],
        default="median",
        help="Severity aggregation method across reviews of the same sample",
    )
    parser.add_argument(
        "--rule",
        default="default",
        help=(
            "Binarization rule for positive class. Presets: "
            f"{list(RULE_PRESETS.keys())}. Or comma-separated severities, "
            "e.g. 'major,critical'."
        ),
    )
    args = parser.parse_args()

    # Sanitize rule for filename when using custom spec
    rule_slug = args.rule.replace(",", "_")
    if args.output is None:
        args.output = Path(f"data/MLCQCodeSmellSamples.normalized.{rule_slug}.json")

    normalize_json_to_json(args.input, args.output, method=args.method, rule=args.rule)


if __name__ == "__main__":
    main()
