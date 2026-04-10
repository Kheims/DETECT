"""Extract OO metrics from MLCQ code snippets using CK tool.

Pipeline:
1. Read normalized JSON
2. Write each snippet as a .java file (wrapped in a class if needed)
3. Run CK tool on each batch of files
4. Parse class.csv and method.csv outputs
5. Aggregate into a single metrics dataset aligned with multi-label y vectors

CK tool: https://github.com/mauricioaniche/ck
"""

import json
import os
import subprocess
import csv
import sys
import hashlib
import shutil
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

CK_JAR = os.environ.get(
    "CK_JAR",
    os.path.join(os.path.dirname(__file__), "..", "tools", "ck.jar")
)

# Class-level metrics from CK
CLASS_METRICS = [
    "cbo", "cboModified", "fanin", "fanout", "wmc", "dit", "noc", "rfc",
    "lcom", "tcc", "lcc",
    "totalMethodsQty", "staticMethodsQty", "publicMethodsQty", "privateMethodsQty",
    "totalFieldsQty", "staticFieldsQty", "publicFieldsQty", "privateFieldsQty",
    "loc", "returnQty", "loopQty", "comparisonsQty", "tryCatchQty",
    "stringLiteralsQty", "numbersQty", "assignmentsQty", "mathOperationsQty",
    "variablesQty", "maxNestedBlocksQty", "uniqueWordsQty",
]

# Method-level metrics from CK (aggregated per class)
METHOD_AGG_METRICS = [
    "loc", "returnsQty", "variablesQty", "parametersQty",
    "methodsInvokedQty", "loopQty", "comparisonsQty",
    "maxNestedBlocksQty",
]


def wrap_snippet(snippet, class_name="Sample"):
    """Wrap a code snippet in a class if it doesn't already have one."""
    stripped = snippet.strip()
    if stripped.startswith("package ") or "class " in stripped.split("\n")[0]:
        return stripped
    if stripped.startswith("public ") or stripped.startswith("private ") or stripped.startswith("protected "):
        # Likely a method — wrap in class
        return f"public class {class_name} {{\n{stripped}\n}}"
    # Assume it needs wrapping
    return f"public class {class_name} {{\n{stripped}\n}}"


def run_ck_on_batch(java_dir, output_dir):
    """Run CK tool on a directory of .java files."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "java", "-jar", CK_JAR,
        java_dir, "false", "0", "true", output_dir + "/"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.returncode == 0


def parse_class_csv(csv_path):
    """Parse CK class.csv output into dict keyed by filename."""
    if not os.path.isfile(csv_path):
        return {}
    results = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = os.path.basename(row.get("file", ""))
            metrics = {}
            for m in CLASS_METRICS:
                val = row.get(m, "0")
                try:
                    fval = float(val)
                    # CK returns -1 or NaN for undefined metrics (e.g. TCC/LCC with < 2 methods)
                    import math
                    if math.isnan(fval) or math.isinf(fval) or fval == -1.0:
                        metrics[f"class_{m}"] = 0.0
                    else:
                        metrics[f"class_{m}"] = fval
                except (ValueError, TypeError):
                    metrics[f"class_{m}"] = 0.0
            results[filename] = metrics
    return results


def parse_method_csv(csv_path):
    """Parse CK method.csv output, aggregate per file."""
    if not os.path.isfile(csv_path):
        return {}
    file_methods = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = os.path.basename(row.get("file", ""))
            if filename not in file_methods:
                file_methods[filename] = []
            method = {}
            for m in METHOD_AGG_METRICS:
                val = row.get(m, "0")
                try:
                    method[m] = float(val)
                except (ValueError, TypeError):
                    method[m] = 0.0
            # CC from wmc at method level
            try:
                method["cc"] = float(row.get("wmc", "0"))
            except (ValueError, TypeError):
                method["cc"] = 0.0
            file_methods[filename].append(method)

    # Aggregate: max, sum, avg per metric
    results = {}
    for filename, methods in file_methods.items():
        agg = {}
        all_metrics = list(methods[0].keys()) if methods else []
        for m in all_metrics:
            values = [meth[m] for meth in methods]
            agg[f"method_{m}_max"] = max(values) if values else 0.0
            agg[f"method_{m}_sum"] = sum(values) if values else 0.0
            agg[f"method_{m}_avg"] = (sum(values) / len(values)) if values else 0.0
        agg["method_count"] = len(methods)
        results[filename] = agg
    return results


def extract_metrics(input_json, output_csv, batch_size=200, workers=1):
    """Extract OO metrics for all MLCQ samples."""
    print(f"Loading {input_json}...")
    with open(input_json, "r") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples")

    # Process in batches
    all_metrics = {}
    num_batches = (len(samples) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(samples))
        batch = samples[start:end]

        with tempfile.TemporaryDirectory() as tmpdir:
            java_dir = os.path.join(tmpdir, "java")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(java_dir)

            # Write java files
            idx_to_file = {}
            for i, sample in enumerate(batch):
                global_idx = start + i
                sample_hash = hashlib.md5(
                    f"{sample.get('repo_url', '')}:{sample.get('commit_hash', '')}:{sample.get('file_path', '')}:{sample.get('start_line', '')}".encode()
                ).hexdigest()[:10]
                filename = f"S{global_idx}_{sample_hash}.java"
                filepath = os.path.join(java_dir, filename)
                wrapped = wrap_snippet(sample["code_snippet"], f"S{global_idx}_{sample_hash}")
                with open(filepath, "w") as f:
                    f.write(wrapped)
                idx_to_file[global_idx] = filename

            # Run CK
            success = run_ck_on_batch(java_dir, out_dir)
            if not success:
                print(f"  Batch {batch_idx+1}/{num_batches}: CK failed, skipping")
                continue

            # Parse results
            class_metrics = parse_class_csv(os.path.join(out_dir, "class.csv"))
            method_metrics = parse_method_csv(os.path.join(out_dir, "method.csv"))

            # Match back to samples
            for global_idx, filename in idx_to_file.items():
                cm = class_metrics.get(filename, {})
                mm = method_metrics.get(filename, {})
                if cm or mm:
                    combined = {**cm, **mm}
                    combined["y"] = samples[global_idx]["y"]
                    combined["sample_idx"] = global_idx
                    all_metrics[global_idx] = combined

        extracted = len([v for v in all_metrics.values() if v])
        print(f"  Batch {batch_idx+1}/{num_batches}: {extracted}/{end} samples with metrics")

    # Write CSV
    print(f"\nTotal: {len(all_metrics)}/{len(samples)} samples with metrics")
    if all_metrics:
        all_keys = set()
        for m in all_metrics.values():
            all_keys.update(k for k in m.keys() if k not in ("y", "sample_idx"))
        all_keys = sorted(all_keys)

        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["sample_idx"] + all_keys + ["y_fe", "y_lm", "y_blob", "y_dc"]
            writer.writerow(header)
            for idx in sorted(all_metrics.keys()):
                m = all_metrics[idx]
                y = m["y"]
                row = [m["sample_idx"]] + [m.get(k, 0) for k in all_keys]
                # y order: [fe, lm, blob, dc]
                if isinstance(y, list):
                    row += [int(v) for v in y]
                else:
                    row += [0, 0, 0, 0]
                writer.writerow(row)

        print(f"Saved to {output_csv} ({len(all_keys)} features)")


if __name__ == "__main__":
    input_json = sys.argv[1] if len(sys.argv) > 1 else "data/MLCQCodeSmellSamples.normalized.json"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "artifacts/metrics_dataset.csv"
    extract_metrics(input_json, output_csv)
