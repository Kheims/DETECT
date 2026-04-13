"""Extract OO metrics from MLCQ code snippets using DesigniteJava.

Pipeline:
1. Read normalized JSON with multi-label y vectors
2. For each sample, prepare a standalone .java file (strip package/imports, rename or wrap class)
3. Write snippets into flat batch directories
4. Run DesigniteJava on each batch
5. Parse typeMetrics.csv and methodMetrics.csv keyed by Type Name
6. Aggregate method-level metrics per sample (max, sum, avg, count)
7. Join with y labels and write metrics_dataset.csv

DesigniteJava: https://github.com/tushartushar/DesigniteJava
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DESIGNITE_JAR = os.environ.get(
    "DESIGNITE_JAR",
    str(Path(__file__).resolve().parent.parent / "tools" / "designite.jar"),
)

# Type-level metrics produced by DesigniteJava
TYPE_METRICS = ["NOF", "NOPF", "NOM", "NOPM", "LOC", "WMC", "NC", "DIT", "LCOM", "FANIN", "FANOUT"]

# Method-level metrics aggregated per class
METHOD_METRICS = ["LOC", "CC", "PC"]

# Regex matching a top-level class/interface/enum declaration
CLASS_DECL_RE = re.compile(
    r"(?P<prefix>(?:^|\n)\s*(?:@\w+(?:\([^)]*\))?\s*)*"
    r"(?:public|private|protected|abstract|final|static|strictfp|\s)*)"
    r"(?P<keyword>class|interface|enum)"
    r"(?P<sep>\s+)"
    r"(?P<name>\w+)"
)

PACKAGE_RE = re.compile(r"^\s*package\s+[\w.]+\s*;\s*\n?", re.MULTILINE)
IMPORT_RE = re.compile(r"^\s*import\s+(?:static\s+)?[\w.*]+\s*;\s*\n?", re.MULTILINE)


def prepare_snippet(snippet: str, class_name: str) -> str:
    """Produce a standalone .java source renamed to class_name."""
    s = snippet.strip()
    # Strip package and imports (unresolvable names cause AST parse errors)
    s = PACKAGE_RE.sub("", s)
    s = IMPORT_RE.sub("", s)
    s = s.strip()

    m = CLASS_DECL_RE.search(s)
    if m:
        original = m.group("name")
        # Replace the class name in the declaration only
        start = m.start("name")
        end = m.end("name")
        s = s[:start] + class_name + s[end:]
        # Rename constructors: pattern "original(" at the start of a line with modifiers
        ctor_re = re.compile(
            r"(\n\s*(?:public|private|protected|\s)*)(" + re.escape(original) + r")(\s*\()"
        )
        s = ctor_re.sub(lambda mm: mm.group(1) + class_name + mm.group(3), s)
        return s
    # Fragment (method body or free statements): wrap in a dummy class
    return f"public class {class_name} {{\n{s}\n}}"


def run_designite(input_dir: Path, output_dir: Path, timeout: int = 300) -> bool:
    """Run DesigniteJava on input_dir writing to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["java", "-jar", DESIGNITE_JAR, "-i", str(input_dir), "-o", str(output_dir)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def _sanitize(v: float) -> float:
    """Replace Designite's sentinel values (NaN, Inf, -1) with 0."""
    import math

    if math.isnan(v) or math.isinf(v) or v == -1.0:
        return 0.0
    return v


def parse_type_metrics(csv_path: Path) -> dict[str, dict[str, float]]:
    """Parse typeMetrics.csv keyed by Type Name."""
    if not csv_path.is_file():
        return {}
    out: dict[str, dict[str, float]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            type_name = row.get("Type Name", "").strip()
            if not type_name:
                continue
            features: dict[str, float] = {}
            for m in TYPE_METRICS:
                v = row.get(m, "0")
                try:
                    features[f"type_{m}"] = _sanitize(float(v))
                except (TypeError, ValueError):
                    features[f"type_{m}"] = 0.0
            out[type_name] = features
    return out


def parse_method_metrics(csv_path: Path) -> dict[str, list[dict[str, float]]]:
    """Parse methodMetrics.csv and group method rows by Type Name."""
    if not csv_path.is_file():
        return {}
    out: dict[str, list[dict[str, float]]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            type_name = row.get("Type Name", "").strip()
            if not type_name:
                continue
            method: dict[str, float] = {}
            for m in METHOD_METRICS:
                v = row.get(m, "0")
                try:
                    method[m] = _sanitize(float(v))
                except (TypeError, ValueError):
                    method[m] = 0.0
            out.setdefault(type_name, []).append(method)
    return out


def aggregate_methods(methods: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate method-level metrics into max, sum, avg per metric + method_count."""
    agg: dict[str, float] = {}
    if not methods:
        for m in METHOD_METRICS:
            agg[f"method_{m}_max"] = 0.0
            agg[f"method_{m}_sum"] = 0.0
            agg[f"method_{m}_avg"] = 0.0
        agg["method_count"] = 0.0
        return agg
    for m in METHOD_METRICS:
        values = [row[m] for row in methods]
        agg[f"method_{m}_max"] = max(values)
        agg[f"method_{m}_sum"] = sum(values)
        agg[f"method_{m}_avg"] = sum(values) / len(values)
    agg["method_count"] = float(len(methods))
    return agg


def process_batch(batch_idx: int, snippets: list[tuple[int, str]], work_root: Path) -> dict[int, dict[str, float]]:
    """Prepare snippets, run Designite, parse output, return per-sample feature dict."""
    batch_dir = work_root / f"batch_{batch_idx:04d}"
    input_dir = batch_dir / "input"
    output_dir = batch_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)

    written: dict[int, str] = {}
    for global_idx, snippet in snippets:
        class_name = f"S{global_idx}"
        prepared = prepare_snippet(snippet, class_name)
        (input_dir / f"{class_name}.java").write_text(prepared)
        written[global_idx] = class_name

    ok = run_designite(input_dir, output_dir)
    if not ok:
        return {}

    type_metrics = parse_type_metrics(output_dir / "typeMetrics.csv")
    method_metrics = parse_method_metrics(output_dir / "methodMetrics.csv")

    per_sample: dict[int, dict[str, float]] = {}
    for global_idx, class_name in written.items():
        type_features = type_metrics.get(class_name)
        if type_features is None:
            continue
        method_rows = method_metrics.get(class_name, [])
        method_features = aggregate_methods(method_rows)
        per_sample[global_idx] = {**type_features, **method_features}
    return per_sample


def extract(input_json: Path, output_csv: Path, batch_size: int = 500, workers: int = 6, keep_workdir: bool = False) -> None:
    with open(input_json) as f:
        samples = json.load(f)
    total = len(samples)
    print(f"Loaded {total} samples from {input_json}")
    print(f"Using Designite jar: {DESIGNITE_JAR}")
    print(f"Batch size: {batch_size}, workers: {workers}")

    if not Path(DESIGNITE_JAR).is_file():
        print(f"ERROR: designite jar not found at {DESIGNITE_JAR}")
        sys.exit(1)

    work_root = Path(tempfile.mkdtemp(prefix="designite_extract_"))
    print(f"Working directory: {work_root}")

    batches: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for idx, sample in enumerate(samples):
        current.append((idx, sample["code_snippet"]))
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    print(f"Prepared {len(batches)} batches")

    all_features: dict[int, dict[str, float]] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_batch, i, batch, work_root): (i, len(batch))
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_idx, batch_size_actual = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"  batch {batch_idx}: failed with {type(exc).__name__}: {exc}")
                done += batch_size_actual
                continue
            all_features.update(result)
            done += batch_size_actual
            print(f"  batch {batch_idx}: {len(result)}/{batch_size_actual} extracted ({done}/{total} total)")

    print(f"\nExtracted features for {len(all_features)}/{total} samples")

    if not all_features:
        print("ERROR: no features extracted")
        if not keep_workdir:
            shutil.rmtree(work_root, ignore_errors=True)
        sys.exit(1)

    feature_keys = sorted({k for feats in all_features.values() for k in feats.keys()})
    header = ["sample_idx"] + feature_keys + ["y_fe", "y_lm", "y_blob", "y_dc"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx in sorted(all_features.keys()):
            feats = all_features[idx]
            y = samples[idx].get("y", [False, False, False, False])
            row = [idx] + [feats.get(k, 0.0) for k in feature_keys] + [int(bool(v)) for v in y]
            writer.writerow(row)
    print(f"Wrote {output_csv} with {len(feature_keys)} features")

    if keep_workdir:
        print(f"Kept working directory: {work_root}")
    else:
        shutil.rmtree(work_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/MLCQCodeSmellSamples.normalized.json")
    parser.add_argument("--output", default="artifacts/metrics_dataset.csv")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()
    extract(Path(args.input), Path(args.output), args.batch_size, args.workers, args.keep_workdir)


if __name__ == "__main__":
    main()
