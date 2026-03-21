"""Pre-training pipeline for self-supervised GNN learning on Java AST graphs.

Orchestrates 3 stages:
1. construction: parse Java files → DOT graphs (reuses build_ast_dot_from_normalized.py)
2. dataset: DOT → PyG Data objects without labels (build_pretrain_dataset.py)
3. pretraining: node type masking loop (pretrain_node_masking.py)

Completely independent from the MLCQ pipeline. All artifacts go under
artifacts/pretrain/ to avoid mixing with MLCQ data.

Usage:
    python scripts/pretrain_pipeline.py --config config/pretrain.yml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_stage(name: str, cmd: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"[pretrain-pipeline] stage: {name}")
    print(f"[pretrain-pipeline] cmd: {' '.join(cmd[:5])}...")
    print(f"{'='*60}\n")

    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"\n[pretrain-pipeline] stage '{name}' FAILED (exit code {result.returncode})")
        sys.exit(1)

    print(f"\n[pretrain-pipeline] stage '{name}' done in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-training pipeline for GNN on Java ASTs.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "pretrain.yml")
    parser.add_argument("--skip-construction", action="store_true",
                        help="Skip DOT construction (reuse existing).")
    parser.add_argument("--skip-dataset", action="store_true",
                        help="Skip dataset building (reuse existing).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    corpus_cfg = cfg.get("corpus", {})
    dataset_cfg = cfg.get("dataset", {})
    pretrain_cfg = cfg.get("pretraining", {})
    artifacts_root = Path(cfg.get("artifacts_root", "artifacts/pretrain"))

    construction_dir = artifacts_root / "construction"
    dot_dir = construction_dir / "dot"
    manifest_path = construction_dir / "manifest.jsonl"
    dataset_dir = artifacts_root / "dataset"
    dataset_path = dataset_dir / "dataset.pt"
    vocab_path = dataset_dir / "node_type_vocab.json"
    checkpoint_dir = artifacts_root / "checkpoints"

    pipeline_start = time.perf_counter()

    # Stage 1: Construction (parse Java → DOT)
    if not args.skip_construction:
        input_dir = corpus_cfg.get("input_dir", "data/pretrain/raw/java-small/training")
        workers = corpus_cfg.get("workers", 8)
        chunksize = corpus_cfg.get("chunksize", 4)
        limit = corpus_cfg.get("limit")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_ast_dot_from_normalized.py"),
            "from-dir",
            "--input-dir", str(input_dir),
            "--output-dir", str(dot_dir),
            "--manifest-path", str(manifest_path),
            "--workers", str(workers),
            "--chunksize", str(chunksize),
            "--progress-every", str(corpus_cfg.get("progress_every", 500)),
        ]
        if limit is not None:
            cmd += ["--limit", str(limit)]

        run_stage("construction", cmd)
    else:
        print("[pretrain-pipeline] skipping construction (--skip-construction)")

    # Stage 2: Dataset (DOT → PyG Data, no labels)
    if not args.skip_dataset:
        edge_types = ",".join(dataset_cfg.get("edge_types", ["Child", "NextToken"]))
        max_nodes = corpus_cfg.get("max_nodes", 15000)
        max_graphs = corpus_cfg.get("max_graphs")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_pretrain_dataset.py"),
            "--manifest-path", str(manifest_path),
            "--dataset-out", str(dataset_path),
            "--node-type-vocab-out", str(vocab_path),
            "--edge-types", edge_types,
            "--max-nodes", str(max_nodes),
            "--progress-every", str(corpus_cfg.get("progress_every", 500)),
        ]
        if max_graphs is not None:
            cmd += ["--max-graphs", str(max_graphs)]

        run_stage("dataset", cmd)
    else:
        print("[pretrain-pipeline] skipping dataset (--skip-dataset)")

    # Stage 3: Pre-training (node type masking)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "pretrain_node_masking.py"),
        "--dataset-path", str(dataset_path),
        "--vocab-path", str(vocab_path),
        "--output-dir", str(checkpoint_dir),
        "--architecture", str(pretrain_cfg.get("architecture", "gcn")),
        "--hidden-dim", str(pretrain_cfg.get("hidden_dim", 512)),
        "--num-layers", str(pretrain_cfg.get("num_layers", 3)),
        "--type-emb-dim", str(pretrain_cfg.get("type_emb_dim", 128)),
        "--dropout", str(pretrain_cfg.get("dropout", 0.2)),
        "--epochs", str(pretrain_cfg.get("epochs", 50)),
        "--lr", str(pretrain_cfg.get("lr", 0.001)),
        "--mask-ratio", str(pretrain_cfg.get("mask_ratio", 0.15)),
        "--batch-size", str(pretrain_cfg.get("batch_size", 64)),
        "--device", str(pretrain_cfg.get("device", "auto")),
        "--save-every", str(pretrain_cfg.get("save_every", 10)),
    ]

    run_stage("pretraining", cmd)

    total = time.perf_counter() - pipeline_start
    print(f"\n[pretrain-pipeline] all stages complete in {total:.1f}s ({total/3600:.1f}h)")
    print(f"[pretrain-pipeline] checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    main()
