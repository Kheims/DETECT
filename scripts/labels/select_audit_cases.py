"""Select samples for a blind re-annotation of MLCQ labels.

Samples are grouped as hard_pos, easy_pos, hard_neg and easy_neg from the
cross-model error rates. Cases are written shuffled with opaque ids, and the
key is written to a separate file.

Usage:
    python scripts/labels/select_audit_cases.py --config config/experiments/label_audit.yml
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def truncate(code: str, max_lines: int, marker: str) -> tuple[str, bool]:
    lines = code.splitlines()
    if len(lines) <= max_lines:
        return code, False
    kept = lines[:max_lines]
    return "\n".join(kept + [marker.format(omitted_lines=len(lines) - max_lines)]), True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args, extra = parser.parse_known_args()
    cfg = load_config(args.config, parse_cli_overrides(extra))["audit"]
    rng = random.Random(int(cfg["seed"]))

    entries = json.loads(resolve(cfg["input_json"]).read_text())
    with resolve(cfg["sample_meta"]).open() as fh:
        level = {int(r["json_index"]): r["level"] for r in csv.DictReader(fh)}
    with resolve(cfg["difficulty_csv"]).open() as fh:
        rows = list(csv.DictReader(fh))

    cases, key = [], []
    for smell, smell_cfg in cfg["smells"].items():
        pool: dict[str, list[int]] = {"hard_pos": [], "easy_pos": [], "hard_neg": [], "easy_neg": []}
        for r in rows:
            if r["smell"] != smell:
                continue
            idx, label, err = int(r["json_index"]), int(r["label"]), float(r["error_rate"])
            if level.get(idx) != smell_cfg["level"]:
                continue
            if label == 1 and err >= float(cfg["hard_positive_error"]):
                pool["hard_pos"].append(idx)
            elif label == 1 and err <= float(cfg["easy_positive_error"]):
                pool["easy_pos"].append(idx)
            elif label == 0 and err >= float(cfg["hard_negative_error"]):
                pool["hard_neg"].append(idx)
            elif label == 0 and err <= float(cfg["easy_negative_error"]):
                pool["easy_neg"].append(idx)
        for group, members in pool.items():
            chosen = rng.sample(sorted(members), min(int(cfg["per_group"]), len(members)))
            print(f"[audit] {smell} {group}: pool {len(members)}, chosen {len(chosen)}")
            for idx in chosen:
                code, cut = truncate(entries[idx]["code_snippet"], int(cfg["max_lines"]), cfg["truncation_marker"])
                case_id = hashlib.sha256(f"{cfg['seed']}|{smell}|{idx}".encode()).hexdigest()[:10]
                cases.append({"case_id": case_id, "smell": smell, "code": code, "truncated": cut})
                key.append({"case_id": case_id, "smell": smell, "group": group, "json_index": idx,
                            "stored_label": int(group.endswith("pos")), "truncated": cut})
    rng.shuffle(cases)

    out = resolve(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps(cases, indent=1))
    (out / "key.json").write_text(json.dumps(key, indent=1))
    print(f"[audit] {len(cases)} cases -> {out}")


if __name__ == "__main__":
    main()
