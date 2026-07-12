"""Scaling benchmark of the self-supervised pre-training (DDP).

Runs pretrain_ssl.py with torchrun for every world size and batch scaling mode of `pretraining.scaling`.
--prepare-only writes the run JSON for a multi-node launch.

Usage:
    python scripts/pretraining/pretrain_scaling.py --config cfg.yml [--key=value ...]
    python scripts/pretraining/pretrain_scaling.py --config cfg.yml --prepare-only --mode weak --label 2x2 --out-json run.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides
from mlcq_graphs.pipeline import PipelineRunner, jsonable, to_path


def run_json_for(runner: PipelineRunner, config: dict, mode: str, label: str, out_root: Path) -> tuple[dict, Path]:
    artifacts_root = to_path(runner.project_root, config.get("run", {}).get("artifacts_root", "artifacts"))
    data_out, _ = runner.prepare_pretraining_data(config, artifacts_root)
    out_dir = out_root / f"{mode}_{label}"
    run_json = runner.pretraining_run_config(config, data_out, out_dir)
    scaling = config["pretraining"].get("scaling", {}) or {}
    run_json["epochs"] = int(scaling.get("epochs", 2))
    run_json["max_steps_per_epoch"] = scaling.get("max_steps_per_epoch", 200)
    run_json["batch_scaling"] = mode
    return jsonable(run_json), out_dir


def summarise(metrics_path: Path) -> dict:
    m = json.loads(metrics_path.read_text())
    last = m["history"][-1]
    return {
        "world_size": m["world_size"],
        "gpu": m["gpu_name"],
        "batch_scaling": m["batch_scaling"],
        "budget_per_gpu": m["budget_per_gpu"],
        "epoch_sec": last["train_sec"],
        "graphs_per_sec": last["train_graphs_per_sec"],
        "nodes_per_sec": last["train_nodes_per_sec"],
        "steps_per_rank": last["train"]["steps_per_rank"],
        "train_loss": last["train"]["loss"],
        "val_loss": last["val"]["loss"],
        "peak_gpu_memory_mb_rank0": m["peak_gpu_memory_mb_rank0"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--mode", default="weak")
    parser.add_argument("--label", default="multinode")
    parser.add_argument("--out-json", type=Path, default=None)
    args, unknown = parser.parse_known_args()

    config = load_config(args.config, parse_cli_overrides(unknown))
    runner = PipelineRunner(config=config, project_root=PROJECT_ROOT)
    artifacts_root = to_path(PROJECT_ROOT, config.get("run", {}).get("artifacts_root", "artifacts"))
    out_root = artifacts_root / "scaling" / str(config.get("run", {}).get("name", "pretrain"))

    if args.prepare_only:
        run_json, out_dir = run_json_for(runner, config, args.mode, args.label, out_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = args.out_json or (out_dir / "run_config.json")
        target.write_text(json.dumps(run_json, indent=2))
        print(target)
        return

    scaling = config["pretraining"].get("scaling", {}) or {}
    world_sizes = [int(w) for w in scaling.get("world_sizes", [1, 2, 4])]
    modes = [str(m) for m in scaling.get("modes", ["weak"])]
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d] or [
        str(i) for i in range(max(world_sizes))
    ]

    rows = []
    for mode in modes:
        for world in world_sizes:
            if world > len(visible):
                print(f"[scaling] skip world={world}: only {len(visible)} devices visible")
                continue
            run_json, out_dir = run_json_for(runner, config, mode, f"w{world}", out_root)
            out_dir.mkdir(parents=True, exist_ok=True)
            json_path = out_dir / "run_config.json"
            json_path.write_text(json.dumps(run_json, indent=2))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(visible[:world]))
            cmd = [
                sys.executable, "-m", "torch.distributed.run", "--standalone",
                f"--nproc_per_node={world}", str(PROJECT_ROOT / "scripts" / "pretraining" / "pretrain_ssl.py"),
                "--config-json", str(json_path),
            ]
            print(f"[scaling] mode={mode} world={world} -> {out_dir}", flush=True)
            with (out_dir / "stdout.log").open("w") as log:
                subprocess.run(cmd, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
            rows.append(summarise(out_dir / "pretrain_metrics.json"))
            print(json.dumps(rows[-1]), flush=True)

    for mode in modes:
        base = next((r for r in rows if r["batch_scaling"] == mode and r["world_size"] == 1), None)
        for r in rows:
            if r["batch_scaling"] == mode and base:
                r["speedup_graphs_per_sec"] = round(r["graphs_per_sec"] / base["graphs_per_sec"], 3)
                r["efficiency"] = round(r["speedup_graphs_per_sec"] / r["world_size"], 3)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "scaling_summary.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
