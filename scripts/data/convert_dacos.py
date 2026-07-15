"""Convert the DACOS MySQL dump and its snippet archive to the normalized schema.

- Method and class snippets go to separate files
- y: at least half of the annotators flag the smell, y_unanimous: all of them
- Optional metadata CSV: whether each smell was offered to the annotators

Usage:
    python scripts/data/convert_dacos.py --config config/experiments/dacos.yml
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402

ESCAPES = {"0": "\x00", "n": "\n", "t": "\t", "r": "\r", "'": "'", '"': '"', "\\": "\\", "Z": "\x1a"}


def resolve(path: Any) -> Path:
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_tuples(line: str, start: int) -> list[list[str]]:
    """Rows of one mysqldump extended INSERT, starting after VALUES."""
    rows: list[list[str]] = []
    cur: list[str] = []
    field: list[str] = []
    in_str = False
    depth = 0
    i, n = start, len(line)
    while i < n:
        c = line[i]
        if in_str:
            if c == "\\":
                field.append(ESCAPES.get(line[i + 1], line[i + 1]))
                i += 2
                continue
            if c == "'":
                if i + 1 < n and line[i + 1] == "'":
                    field.append("'")
                    i += 2
                    continue
                in_str = False
            else:
                field.append(c)
            i += 1
            continue
        if c == "'":
            in_str = True
        elif c == "(":
            depth += 1
            if depth == 1:
                cur, field = [], []
        elif c == ")" and depth == 1:
            cur.append("".join(field).strip())
            rows.append(cur)
            depth = 0
        elif c == "," and depth == 1:
            cur.append("".join(field).strip())
            field = []
        elif depth == 1:
            field.append(c)
        i += 1
    return rows


def load_table(sql_path: Path, table: str) -> list[list[str]]:
    prefix = f"INSERT INTO `{table}`"
    rows: list[list[str]] = []
    with sql_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(prefix):
                rows.extend(parse_tuples(line, line.index("VALUES") + len("VALUES")))
    return rows


def to_bit(raw: str) -> int:
    # bit(1) columns are dumped as "_binary <byte>", the byte \x01 is true
    return int(raw.replace("_binary", "").strip() == "\x01")


def to_number(raw: str) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args, extra = parser.parse_known_args()
    cfg = load_config(args.config, parse_cli_overrides(extra))["dacos"]
    ccfg = cfg["convert"]
    sql = resolve(ccfg["sql_dump"])
    flags = dict(ccfg["flags"])  # annotation column -> smell name
    method_smells = list(ccfg["method_smells"])
    class_smells = list(ccfg["class_smells"])
    label_order = list(cfg["label_order"])
    if len(label_order) != len(method_smells):
        raise SystemExit("label_order and method_smells must have the same length")

    columns = {"annotation": ["id", "iscm", "isim", "islp", "isma", "is_smell", "sample_id"],
               "sample": ["id", "designite_id", "has_smell", "is_class", "path_to_file", "project_name",
                          "sample_constraints", "smells"],
               "method_metrics": ["id", "cc", "loc", "pc", "method_name", "package_name", "project_name",
                                  "type_name"]}
    tables = {t: [dict(zip(cols, row)) for row in load_table(sql, t)] for t, cols in columns.items()}
    samples = {int(r["id"]): r for r in tables["sample"]}
    metrics = {int(r["id"]): r for r in tables["method_metrics"]}
    votes: dict[int, list[dict[str, int]]] = defaultdict(list)
    for r in tables["annotation"]:
        if r["sample_id"] in ("NULL", ""):
            continue
        votes[int(r["sample_id"])].append({flag: to_bit(r[flag]) for flag in flags})

    archive = zipfile.ZipFile(resolve(ccfg["snippet_zip"]))
    names = set(archive.namelist())
    smell_to_flag = {smell: flag for flag, smell in flags.items()}
    out: dict[str, list[dict[str, Any]]] = {"method": [], "class": []}
    stats: Counter = Counter()
    for sid in sorted(votes):
        info = samples.get(sid)
        if info is None:
            stats["annotation_without_sample"] += 1
            continue
        member = info["path_to_file"].lstrip("/")
        if member not in names:
            stats["missing_snippet"] += 1
            continue
        level = "class" if to_bit(info["is_class"]) else "method"
        ann = votes[sid]
        smells = class_smells if level == "class" else method_smells
        labels = {}
        for smell in smells:
            pos = sum(a[smell_to_flag[smell]] for a in ann)
            labels[smell] = {"present": pos * 2 >= len(ann), "votes_pos": pos, "votes_total": len(ann)}
        entry: dict[str, Any] = {
            "dataset": "dacos",
            "sample_id": sid,
            "project": info["project_name"],
            "path_to_file": info["path_to_file"],
            "level": level,
            "code_snippet": archive.read(member).decode("utf-8", errors="replace"),
            "labels": labels,
            "n_annotations": len(ann),
            "sample_constraints": int(info["sample_constraints"]),
            "designite_smells_at_sampling": [s for s in info["smells"].split(",") if s],
        }
        if level == "method":
            entry["y"] = [bool(labels[s]["present"]) for s in method_smells]
            entry["y_unanimous"] = [labels[s]["votes_pos"] == labels[s]["votes_total"] for s in method_smells]
            m = metrics.get(int(info["designite_id"])) if info["designite_id"] not in ("NULL", "") else None
            entry["designite"] = None if m is None else {
                "cc": to_number(m["cc"]), "loc": to_number(m["loc"]), "pc": to_number(m["pc"]),
                "method_name": m["method_name"], "type_name": m["type_name"]}
            stats["method_without_metrics"] += int(m is None)
        out[level].append(entry)

    for level, key in (("method", "out_methods"), ("class", "out_classes")):
        path = resolve(ccfg[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out[level]))
        print(f"[dacos] wrote {path} ({len(out[level])} {level} snippets)")
    if ccfg.get("out_meta_csv"):
        offered = dict(ccfg["offered_codes"])  # smell -> sampling code that offers it
        path = resolve(ccfg["out_meta_csv"])
        cols = ["json_index", "project", "sampling_codes"] + [f"offered_{s.replace(' ', '_')}" for s in method_smells]
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for i, e in enumerate(out["method"]):
                codes = e["designite_smells_at_sampling"]
                w.writerow([i, e["project"], "+".join(codes) or "none"]
                           + [int(str(offered[s]) in codes) for s in method_smells])
        print(f"[dacos] wrote {path}")
    for level, smells in (("method", method_smells), ("class", class_smells)):
        for smell in smells:
            pos = sum(e["labels"][smell]["present"] for e in out[level])
            print(f"  {smell:<26} {pos}/{len(out[level])} positive ({100 * pos / max(1, len(out[level])):.1f}%)")
    print(f"  annotations per sample: {dict(Counter(e['n_annotations'] for lv in out.values() for e in lv))}")
    print(f"  {dict(stats)}")


if __name__ == "__main__":
    main()
