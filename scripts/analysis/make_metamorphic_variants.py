"""Build metamorphic variants of the MLCQ and DACOS snippets.

Writes per transform a variant JSON, size features and details, plus
stats.json, an applicability table and a size sanity table.

Usage:
    python scripts/analysis/make_metamorphic_variants.py --config config/experiments/metamorphic.yml
    python scripts/analysis/make_metamorphic_variants.py --config ... --variants.only_datasets=[dacos] --variants.only_transforms=[compact]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.extract_size_features import SIZE_COLUMNS, size_features  # noqa: E402
from mlcq_graphs.config import load_config, parse_cli_overrides  # noqa: E402
from mlcq_graphs.metamorphic import Options, Outcome, apply, transform_classes  # noqa: E402

SANITY_COLUMNS = ["dataset", "transform", "transform_class", "column", "n_applied", "n_changed", "share_changed",
                  "mean_delta", "min_delta", "max_delta", "expected_to_change", "n_other_changed"]
APPLICABILITY_COLUMNS = ["dataset", "level", "transform", "transform_class", "n", "applied", "applied_fraction",
                         "no_change", "not_applicable", "verification_failed", "errors"]


def resolve(path: Any) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path(path)
    tmp.write_text(text)
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path(path)
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _variant(task: tuple[str, str, str, Options]) -> Outcome:
    name, code, seed, opts = task
    return apply(name, code, opts, seed)


def run_pool(fn: Any, tasks: list[Any], workers: int, chunksize: int) -> list[Any]:
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, tasks, chunksize=chunksize))
    return [fn(t) for t in tasks]


def sizes(codes: list[str], opts: Options, workers: int, chunksize: int) -> list[dict[str, int]]:
    tasks = [(j, code, opts.wrapper_prefix, opts.wrapper_suffix) for j, code in enumerate(codes)]
    return run_pool(size_features, tasks, workers, chunksize)


def levels_of(spec: dict[str, Any], entries: list[dict[str, Any]]) -> list[str]:
    if spec["source"] == "field":
        return [str(e.get(spec["field"], "none")) for e in entries]
    if spec["source"] == "csv":
        table = {int(r[spec["key"]]): r[spec["column"]] for r in read_csv(resolve(spec["path"]))}
        return [table.get(j, "none") for j in range(len(entries))]
    raise ValueError(f"unknown level source {spec['source']}")


def check_reference(path_value: Any, base: list[dict[str, int]]) -> str:
    """Number of samples whose recomputed base size features differ from the reference CSV."""
    if not path_value or not resolve(path_value).exists():
        return f"reference {path_value} missing, not checked"
    ref = {int(r["sample_idx"]): r for r in read_csv(resolve(path_value))}
    differ = [f["sample_idx"] for f in base
              if f["sample_idx"] not in ref or any(int(ref[f["sample_idx"]][c]) != f[c] for c in SIZE_COLUMNS)]
    return (f"{len(differ)} of {len(base)} samples differ from {path_value}"
            + (f" (first json indices {differ[:5]})" if differ else ""))


def summarize(outcomes: list[Outcome], levels: list[str]) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for j, level in enumerate(levels):
        groups["all"].append(j)
        groups[level].append(j)
    out: dict[str, Any] = {}
    for group, idx in groups.items():
        status = Counter(outcomes[j].status for j in idx)
        reasons = Counter(f"{outcomes[j].status}:{outcomes[j].reason}" for j in idx
                          if outcomes[j].status != "applied")
        edits = [outcomes[j].n_edits for j in idx if outcomes[j].applied]
        info: Counter = Counter()
        for j in idx:
            info.update({k: v for k, v in outcomes[j].info.items() if isinstance(v, int) and not isinstance(v, bool)})
        out[group] = {
            "n": len(idx),
            "applied": status["applied"],
            "applied_fraction": round(status["applied"] / len(idx), 4) if idx else None,
            "verification_failed": status["verification_failed"],
            "errors": status["error"],
            "status": dict(status),
            "reasons": dict(reasons.most_common()),
            "mean_edits_applied": round(sum(edits) / len(edits), 2) if edits else None,
            # integer details of the transform (renamed names, kept methods, ...) summed over all samples
            "info_totals": dict(info),
        }
    return out


def size_sanity(dataset: str, name: str, klass: str, base: list[dict[str, int]], variant: list[dict[str, int]],
                outcomes: list[Outcome], allowed: list[str]) -> list[dict[str, Any]]:
    applied = [j for j, o in enumerate(outcomes) if o.applied]
    others = [j for j, o in enumerate(outcomes) if not o.applied]
    n_other = sum(any(variant[j][c] != base[j][c] for c in SIZE_COLUMNS) for j in others)
    rows = []
    for col in SIZE_COLUMNS:
        deltas = [variant[j][col] - base[j][col] for j in applied]
        changed = sum(d != 0 for d in deltas)
        rows.append({
            "dataset": dataset, "transform": name, "transform_class": klass, "column": col,
            "n_applied": len(applied), "n_changed": changed,
            "share_changed": round(changed / len(applied), 4) if applied else "",
            "mean_delta": round(sum(deltas) / len(deltas), 3) if deltas else "",
            "min_delta": min(deltas) if deltas else "", "max_delta": max(deltas) if deltas else "",
            "expected_to_change": int(col in allowed), "n_other_changed": n_other,
        })
    return rows


def print_sanity(rows: list[dict[str, Any]]) -> None:
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_pair[(r["dataset"], r["transform"])].append(r)
    print("\n[sanity] size features of the variants against the base (applied samples only)")
    print(f"{'dataset':8} {'transform':20} {'applied':>7}  changed columns: share of applied samples (mean change)")
    for (dataset, name), rs in by_pair.items():
        changed = [r for r in rs if int(r["n_changed"]) > 0]
        text = ", ".join(f"{r['column']} {float(r['share_changed']):.0%} ({float(r['mean_delta']):+.1f})"
                         for r in changed) or "none"
        unexpected = [r["column"] for r in changed if not int(r["expected_to_change"])]
        print(f"{dataset:8} {name:20} {rs[0]['n_applied']:>7}  {text}")
        if unexpected:
            print(f"{'':38}unexpected: {', '.join(unexpected)}")
        if int(rs[0]["n_other_changed"]):
            print(f"{'':38}samples not applied but changed: {rs[0]['n_other_changed']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config/experiments/metamorphic.yml"))
    args, extra = parser.parse_known_args(argv)
    cfg = load_config(resolve(args.config), parse_cli_overrides(extra))
    classes = transform_classes(cfg)
    class_of = {t: c for c, ts in classes.items() for t in ts}
    vcfg = cfg["variants"]
    workers, chunksize = int(vcfg["workers"]), int(vcfg["chunksize"])
    names = list(class_of)
    if vcfg.get("only_transforms"):
        names = [n for n in names if n in set(vcfg["only_transforms"])]
    expectations = vcfg.get("size_expectations", {})
    seeds = cfg.get("seeds", {}) or {}

    sanity_rows: list[dict[str, Any]] = []
    applicability: list[dict[str, Any]] = []
    done_pairs: set[tuple[str, str]] = set()
    for dataset, ds in cfg["datasets"].items():
        if vcfg.get("only_datasets") and dataset not in set(vcfg["only_datasets"]):
            continue
        t0 = time.time()
        opts = Options.from_config(cfg, dataset)
        entries = json.loads(resolve(ds["base_json"]).read_text())
        codes = [str(e.get("code_snippet", "")) for e in entries]
        levels = levels_of(ds["level"], entries)
        out_dir = resolve(ds["out_dir"])
        base = sizes(codes, opts, workers, chunksize)
        print(f"[{dataset}] {len(entries)} samples, levels {dict(Counter(levels))}; base size features: "
              f"{check_reference(ds.get('reference_size_csv'), base)}")

        stats_path = out_dir / "stats.json"
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        stats.update({"dataset": dataset, "base_json": str(ds["base_json"]), "n_samples": len(entries),
                      "levels": dict(Counter(levels)), "transform_classes": classes})
        # entries of earlier runs in another layout are dropped
        stats["transforms"] = {t: v for t, v in (stats.get("transforms") or {}).items()
                               if isinstance(v, dict) and "levels" in v}
        for name in names:
            t1 = time.time()
            seed = seeds.get(name, 0)
            outcomes = run_pool(_variant, [(name, code, f"{seed}:{j}", opts) for j, code in enumerate(codes)],
                                workers, chunksize)
            variant = []
            for entry, o in zip(entries, outcomes):
                v = dict(entry)
                v["code_snippet"] = o.code
                v["metamorphic"] = {"transform": name, "applied": o.applied, "n_edits": o.n_edits}
                variant.append(v)
            text = json.dumps(variant)
            write_text(out_dir / f"{name}.normalized.json", text)
            var_sizes = sizes([o.code for o in outcomes], opts, workers, chunksize)
            write_csv(out_dir / f"{name}.size_features.csv", var_sizes, ["sample_idx", *SIZE_COLUMNS])
            write_csv(out_dir / f"{name}.details.csv",
                      [{"json_index": j, "level": levels[j], "status": o.status, "reason": o.reason,
                        "applied": int(o.applied), "n_edits": o.n_edits, "info": json.dumps(o.info, sort_keys=True)}
                       for j, o in enumerate(outcomes)],
                      ["json_index", "level", "status", "reason", "applied", "n_edits", "info"])
            summary = summarize(outcomes, levels)
            stats["transforms"][name] = {"transform_class": class_of[name],
                                         "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                         "levels": summary}
            for level, v in summary.items():
                applicability.append({
                    "dataset": dataset, "level": level, "transform": name, "transform_class": class_of[name],
                    "n": v["n"], "applied": v["applied"], "applied_fraction": v["applied_fraction"],
                    "no_change": v["status"].get("no_change", 0), "not_applicable": v["status"].get("not_applicable", 0),
                    "verification_failed": v["verification_failed"], "errors": v["errors"]})
            sanity_rows += size_sanity(dataset, name, class_of[name], base, var_sizes, outcomes,
                                       list(expectations.get(name, [])))
            done_pairs.add((dataset, name))
            s = summary["all"]
            by_level = ", ".join(f"{lv} {v['applied']}/{v['n']}" for lv, v in summary.items() if lv != "all")
            print(f"[{dataset}] {name}: applied {s['applied']}/{s['n']} ({by_level}); verification failed "
                  f"{s['verification_failed']}, errors {s['errors']}; status {s['status']} ({time.time() - t1:.1f}s)")
        stats["transforms"] = {t: stats["transforms"][t] for t in class_of if t in stats["transforms"]}
        write_text(stats_path, json.dumps(stats, indent=1))
        print(f"[{dataset}] wrote {out_dir} ({time.time() - t0:.1f}s)")

    for key, columns, rows in (("size_sanity_csv", SANITY_COLUMNS, sanity_rows),
                               ("applicability_csv", APPLICABILITY_COLUMNS, applicability)):
        path = resolve(vcfg[key])
        kept = [r for r in (read_csv(path) if path.exists() else [])
                if (r["dataset"], r["transform"]) not in done_pairs and r["transform"] in class_of]
        write_csv(path, kept + rows, columns)
        print(f"[out] {path}")
    print_sanity(sanity_rows)


if __name__ == "__main__":
    main()
