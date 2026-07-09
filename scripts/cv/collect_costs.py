"""Collect the measured training and inference costs of every model family.

Numbers are read from the files written by the runs. Costs that were never measured are left out.

Usage:
    python scripts/cv/collect_costs.py --config config/experiments/costs.yml
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
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


def files(pattern: str) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(str(resolve(pattern))))]


def median_of(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in records if r.get(key) is not None]
    return statistics.median(values) if values else None


def collect(spec: dict[str, Any]) -> dict[str, Any]:
    kind = spec["kind"]
    out: dict[str, Any] = {"kind": kind, "hardware": spec.get("hardware")}
    if kind == "llm_meta":
        runs = json.loads(resolve(spec["meta"]).read_text())["runs"]
        # throughput of the batched scoring runs (several when resumed)
        generate = sum(r["generate_seconds"] for r in runs)
        prompts = sum(r["prompts_per_second"] * r["generate_seconds"] for r in runs)
        tokens = sum(r["tokens_per_second"] * r["generate_seconds"] for r in runs)
        out["inference_ms_per_sample"] = 1000.0 * spec["prompts_per_sample"] * generate / prompts
        out["dataset_seconds"] = sum(r["seconds"] for r in runs)
        out["load_seconds"] = runs[0]["load_seconds"]
        out["prompt_tokens_per_second"] = tokens / generate
        out["train_seconds_per_fold"] = 0.0
    elif kind == "embedding_probe":
        meta = json.loads(resolve(spec["meta"]).read_text())
        n = len(json.loads(resolve(spec["index"]).read_text()))
        out["inference_ms_per_sample"] = 1000.0 * meta["seconds"]["embed"] / n
        out["peak_gpu_memory_gb"] = (meta.get("peak_gpu_memory_mib") or 0) / 1024 or None
        records = [r for f in files(spec["probe_records"]) for r in json.loads(f.read_text())]
        out["train_seconds_per_fold"] = median_of(records, "train_duration_sec")
    elif kind == "finetune":
        records = [json.loads(f.read_text()) for f in files(spec["records"])]
        out["items"] = len(records)
        out["train_seconds_per_fold"] = median_of(records, "train_duration_sec")
        n_test = [sum(m["tp"] + m["fp"] + m["tn"] + m["fn"] for m in [r["per_label_tuned"]["is_feature_envy"]])
                  for r in records]
        evals = [r["test_eval_sec"] / n for r, n in zip(records, n_test) if r.get("test_eval_sec")]
        out["inference_ms_per_sample"] = 1000.0 * statistics.median(evals) if evals else None
        peaks = [r["peak_gpu_mem_bytes"] for r in records if r.get("peak_gpu_mem_bytes")]
        out["peak_gpu_memory_gb"] = statistics.median(peaks) / 2 ** 30 if peaks else None
    elif kind == "pipeline_records":
        records = [r for f in files(spec["records"]) for r in json.loads(f.read_text())]
        out["train_seconds_per_fold"] = median_of(records, "train_duration_sec")
        mem = median_of(records, "peak_gpu_memory_mb")
        out["peak_gpu_memory_gb"] = mem / 1024 if mem else None
        out["note"] = spec.get("note")
    elif kind == "tabular_records":
        records = [r for f in files(spec["records"]) for r in json.loads(f.read_text())]
        out["train_seconds_per_fold"] = median_of(records, "train_duration_sec")
    else:
        raise ValueError(f"unknown cost kind {kind}")
    if "latency" in spec:
        lat = json.loads(resolve(spec["latency"]["path"]).read_text())
        steps = spec["latency"]["steps"]
        out["inference_ms_per_sample"] = sum(lat["per_snippet_ms"][s]["p50"] for s in steps)
        out["inference_measure"] = f"p50 of {'+'.join(steps)}, one snippet at a time"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args, extra = parser.parse_known_args()
    cfg = load_config(args.config, parse_cli_overrides(extra))["costs"]
    table = {name: collect(spec) for name, spec in cfg["models"].items()}
    out = resolve(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1))
    for name, row in table.items():
        inf = row.get("inference_ms_per_sample")
        train = row.get("train_seconds_per_fold")
        print(f"{name:<28} inference {inf if inf is None else f'{inf:9.2f} ms'}  "
              f"train/fold {train if train is None else f'{train:8.1f} s'}  {row.get('hardware')}")
    print(f"[costs] wrote {out}")


if __name__ == "__main__":
    main()
