"""Write per-sample predictions (predictions.jsonl) of finished GNN runs.

Each run is rebuilt from the pipeline cache and its best checkpoint. Predictions are
written only when the rebuilt test MCC matches metrics.json within mcc_tolerance.

Usage:
    python scripts/cv/export_gnn_predictions.py --config config/experiments/gnn_cv_ablation.yml \\
        [--shard-index 0 --shard-count 2] [--gnn_export.out_dir=...]
"""
from __future__ import annotations

import argparse
import copy
import glob
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcq_graphs.config import load_config, parse_cli_overrides, set_nested_value  # noqa: E402
from mlcq_graphs.constants import LABEL_ORDER  # noqa: E402
from mlcq_graphs.evaluation import evaluate_full  # noqa: E402
from mlcq_graphs.evaluation.cv import shard  # noqa: E402
from mlcq_graphs.pipeline import PipelineRunner, StageState, to_path  # noqa: E402
from scripts.train_gcn_baseline import (  # noqa: E402
    LoadedGraphs,
    attach_graph_features,
    build_loader,
    build_model,
    collect_logits_and_targets,
    load_graphs,
    loader_json_indices,
    parse_feature_mode,
    prediction_model_name,
    select_device,
    split_graphs,
    standardise_graph_features,
    subsample_train,
    tune_thresholds,
    write_predictions,
)

SNAPSHOT_KEYS = (
    "architecture",
    "num_layers",
    "use_type_features",
    "use_numeric_features",
    "use_token_features",
    "num_node_types",
    "num_numeric_feats",
)


def trial_key(seed: int, combo: dict[str, Any]) -> str:
    return json.dumps({"seed": int(seed), "combo": combo}, sort_keys=True)


def job_trials(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Trials of one export job, in the order of the pipeline ablation."""
    experiment = to_path(PROJECT_ROOT, job["experiment"])
    overrides = {str(k): v for k, v in (job.get("overrides") or {}).items()}
    config = load_config(experiment, overrides)
    ablation = config.get("ablation", {}) or {}
    trials: list[dict[str, Any]] = []
    if ablation.get("enabled", False):
        grid = ablation.get("grid", {}) or {}
        keys = list(grid)
        for values in itertools.product(*[v if isinstance(v, list) else [v] for v in grid.values()]):
            combo = dict(zip(keys, values))
            for seed in ablation.get("seeds", [42]):
                trials.append({"experiment": str(experiment), "overrides": overrides, "combo": combo, "seed": int(seed)})
    else:
        seed = int(config.get("training", {}).get("seed", 42))
        trials.append({"experiment": str(experiment), "overrides": overrides, "combo": {}, "seed": seed})

    patterns = job.get("records")
    if not patterns:
        return trials
    records: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        for path in sorted(glob.glob(str(to_path(PROJECT_ROOT, pattern)))):
            for record in json.loads(Path(path).read_text()):
                records[trial_key(record["seed"], record["combo"])] = record
    if not records:
        raise SystemExit(f"no record found in {patterns}")
    known = {trial_key(t["seed"], t["combo"]) for t in trials}
    stray = sorted(set(records) - known)
    if stray:
        print(f"[export] {len(stray)} record(s) are not trials of {experiment.name}, skipped: {stray[:2]}")
    return [dict(t, record=records[k]) for t in trials if (k := trial_key(t["seed"], t["combo"])) in records]


def trial_config(trial: dict[str, Any]) -> dict[str, Any]:
    """Same construction as PipelineRunner._run_ablation."""
    config = load_config(Path(trial["experiment"]), trial["overrides"])
    config.setdefault("ablation", {})["enabled"] = False
    config.setdefault("training", {})["seed"] = int(trial["seed"])
    for key, value in trial["combo"].items():
        set_nested_value(config, key, value)
    return config


def resolve_run(trial: dict[str, Any]) -> tuple[str, dict[str, Any], Path]:
    """(fingerprint, run_training config, run_dir) of a trial, without training."""
    config = trial_config(trial)
    runner = PipelineRunner(config=config, project_root=PROJECT_ROOT)
    prepare = copy.deepcopy(config)
    prepare["training"]["enabled"] = False
    prepare.setdefault("run", {}).setdefault("wandb", {})["enabled"] = False
    stages = {name: StageState(**state) for name, state in runner._run_single(prepare)["stages"].items()}
    rebuilt = [name for name, state in stages.items() if not state.reused]
    if rebuilt:
        print(f"[export] stages missing from the cache were rebuilt: {rebuilt}")
    artifacts_root = to_path(PROJECT_ROOT, config.get("run", {}).get("artifacts_root", "artifacts"))
    fingerprint, _, train_cfg = runner.gnn_training_plan(
        config=config,
        artifacts_root=artifacts_root,
        construction_state=stages.get("construction"),
        dataset_state=stages.get("dataset"),
        token_state=stages.get("token_encoder"),
        pretraining_state=stages.get("pretraining"),
    )
    return fingerprint, train_cfg, artifacts_root / "runs" / fingerprint


def graph_key(train_cfg: dict[str, Any]) -> str:
    """Everything load_graphs depends on: runs with the same key share graphs."""
    keys = (
        "dataset_path",
        "num_labels",
        "max_graphs",
        "to_undirected",
        "edge_type_mode",
        "graph_features",
        "pool_subset_types",
        "node_type_vocab_path",
        "pretrained_encoder",
        "project_root",
    )
    _, _, allow_tokens = parse_feature_mode(str(train_cfg.get("feature_mode", "type_numeric")))
    tokens = [train_cfg.get("token_vectors_path"), train_cfg.get("manifest_path")] if allow_tokens else None
    return json.dumps({**{k: train_cfg.get(k) for k in keys}, "tokens": tokens}, sort_keys=True, default=str)


def mcc_gap(rebuilt: list[float | None], reference: list[float | None]) -> float:
    """Largest absolute per-label difference, where None only matches None."""
    gap = 0.0
    for a, b in zip(rebuilt, reference, strict=True):
        if a is None or b is None:
            gap = max(gap, 0.0 if a is None and b is None else float("inf"))
        else:
            gap = max(gap, abs(float(a) - float(b)))
    return gap


def export_run(
    train_cfg: dict[str, Any],
    run_dir: Path,
    out_path: Path,
    settings: dict[str, Any],
    cache: dict[str, Any],
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild one finished run, check its test MCC and write its predictions."""
    started = time.perf_counter()
    cfg = dict(train_cfg)
    cfg["save_split"] = False  # never touch the split cache from here
    num_labels = int(cfg.get("num_labels", 4))
    seed = int(cfg.get("seed", 42))
    labels = list(LABEL_ORDER[:num_labels])
    device = select_device(str(settings.get("device", "auto")))
    snapshot = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())

    key = graph_key(cfg)
    if cache.get("key") != key:
        cache.clear()
        cache["graphs"] = load_graphs(cfg)
        cache["key"] = key
    graphs: LoadedGraphs = cache["graphs"]
    dataset = graphs.dataset

    # same split as training: the cached split file when it exists
    train_ds, val_ds, test_ds, split_indices, _ = split_graphs(cfg, dataset, seed, num_labels)
    if split_indices != snapshot["split_indices"]:
        raise ValueError(f"{run_dir.name}: rebuilt split differs from config.json split_indices")
    train_fraction = float(cfg.get("train_fraction", 1.0))
    if train_fraction != 1.0:
        train_ds, subset = subsample_train(train_ds, split_indices["train"], train_fraction, seed, num_labels)
        if snapshot.get("train_subset_indices") not in (None, subset):
            raise ValueError(f"{run_dir.name}: rebuilt training subset differs from config.json")
    if graphs.graph_feature_names:
        # standardise_graph_features works in place: attach the raw values again first
        attach_graph_features(dataset, cfg["graph_features"], Path(str(cfg.get("project_root", PROJECT_ROOT))))
        standardise_graph_features(train_ds, dataset)

    model, info = build_model(cfg, graphs, train_ds, num_labels)
    for name in SNAPSHOT_KEYS:
        if name in snapshot and snapshot[name] != info[name]:
            raise ValueError(f"{run_dir.name}: {name}={info[name]} but config.json has {snapshot[name]}")
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    eval_max_nodes = int(cfg.get("eval_max_nodes_per_batch", 40000))
    eval_max_edges = int(cfg.get("eval_max_edges_per_batch", 0))

    def loader(part: list[Any]):
        return build_loader(
            dataset=part,
            max_nodes_per_batch=eval_max_nodes,
            max_edges_per_batch=eval_max_edges if eval_max_edges > 0 else None,
            shuffle=False,
            seed=seed,
            epoch=0,
            num_workers=0,
            pin_memory=False,
        )

    val_loader, test_loader = loader(val_ds), loader(test_ds)
    val_logits, val_targets = collect_logits_and_targets(model, val_loader, device, num_labels)
    val_probs = torch.sigmoid(val_logits)
    thresholds = tune_thresholds(y_true=val_targets.int(), probs=val_probs, steps=int(cfg.get("threshold_steps", 101)))
    test_logits, test_targets = collect_logits_and_targets(model, test_loader, device, num_labels)
    tuned = evaluate_full(logits=test_logits, y_true=test_targets, thresholds=thresholds, label_names=labels)

    rebuilt_mcc = tuned["mcc_per_label"]
    metrics_mcc = metrics["test_tuned"]["mcc_per_label"]
    gaps = {"metrics_json": mcc_gap(rebuilt_mcc, metrics_mcc)}
    if record is not None:
        record_mcc = [record["per_label_tuned"][label]["mcc"] for label in labels]
        gaps["records_json"] = mcc_gap(rebuilt_mcc, record_mcc)
    recorded_thresholds = [float(metrics["thresholds"][label]) for label in labels]
    threshold_gap = float(np.max(np.abs(thresholds.numpy() - np.asarray(recorded_thresholds))))
    tolerance = float(settings["mcc_tolerance"])
    ok = all(gap <= tolerance for gap in gaps.values())

    print(f"[export] run {run_dir.name[:12]} strategy={cfg.get('split_strategy')} seed={seed} fold={cfg.get('cv_fold')}")
    print(f"  {'label':<18} {'rebuilt_mcc':>12} {'metrics_mcc':>12} {'abs_diff':>10}")
    for label, a, b in zip(labels, rebuilt_mcc, metrics_mcc):
        diff = "n/a" if a is None or b is None else f"{abs(a - b):.2e}"
        a_txt = "None" if a is None else f"{a:.6f}"
        b_txt = "None" if b is None else f"{b:.6f}"
        print(f"  {label:<18} {a_txt:>12} {b_txt:>12} {diff:>10}")
    print(
        f"  max_gap={max(gaps.values()):.2e} (tolerance {tolerance:g}, vs {', '.join(gaps)}) "
        f"threshold_gap={threshold_gap:.2e} -> {'MATCH' if ok else 'MISMATCH'}"
    )

    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "fingerprint": run_dir.name,
        "split_strategy": cfg.get("split_strategy"),
        "seed": seed,
        "cv_fold": cfg.get("cv_fold"),
        "rebuilt_mcc": rebuilt_mcc,
        "metrics_mcc": metrics_mcc,
        "gaps": gaps,
        "threshold_gap": threshold_gap,
        "match": ok,
        "predictions_path": None,
    }
    if ok:
        model_name = settings.get("model_name") or prediction_model_name(cfg, info)
        n_rows = write_predictions(
            out_path,
            model_name=str(model_name),
            cfg=cfg,
            parts=[
                ("val", loader_json_indices(val_loader, val_ds), val_targets.int().numpy(), val_probs.numpy()),
                (
                    "test",
                    loader_json_indices(test_loader, test_ds),
                    test_targets.int().numpy(),
                    torch.sigmoid(test_logits).numpy(),
                ),
            ],
        )
        result["predictions_path"] = str(out_path)
        print(f"  wrote {n_rows} rows ({model_name}) -> {out_path}  [{time.perf_counter() - started:.1f}s]")
    return result


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config/experiments/gnn_cv_ablation.yml"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args, unknown = parser.parse_known_args()

    config = load_config(to_path(PROJECT_ROOT, args.config), parse_cli_overrides(unknown))
    settings = config.get("gnn_export")
    if not settings or not settings.get("jobs"):
        raise SystemExit(f"{args.config} has no gnn_export.jobs")
    out_dir = to_path(PROJECT_ROOT, settings["out_dir"]) if settings.get("out_dir") else None
    report_dir = to_path(PROJECT_ROOT, settings["report_dir"])

    trials = [trial for job in settings["jobs"] for trial in job_trials(job)]
    mine = shard(trials, args.shard_index, args.shard_count)
    print(f"[export] {len(trials)} trial(s), shard {args.shard_index}/{args.shard_count}: {len(mine)}")

    # resolve every run first, then group the runs that share graphs
    todo: list[tuple[str, dict[str, Any], Path, Path, dict[str, Any] | None]] = []
    results: list[dict[str, Any]] = []
    for trial in mine:
        fingerprint, train_cfg, run_dir = resolve_run(trial)
        record = trial.get("record")
        if record is not None and Path(record["run_dir"]).name != fingerprint:
            raise SystemExit(f"fingerprint {fingerprint} differs from the record run_dir {record['run_dir']}")
        if not (run_dir / "metrics.json").exists() or not (run_dir / "best_model.pt").exists():
            print(f"[export] run {fingerprint[:12]} not finished, skipped")
            results.append({"fingerprint": fingerprint, "skipped": "not finished"})
            continue
        out_path = (out_dir / fingerprint if out_dir else run_dir) / "predictions.jsonl"
        if out_path.exists() and not bool(settings.get("overwrite", False)):
            print(f"[export] {out_path} exists, skipped")
            results.append({"fingerprint": fingerprint, "skipped": "exists", "predictions_path": str(out_path)})
            continue
        todo.append((graph_key(train_cfg), train_cfg, run_dir, out_path, record))

    cache: dict[str, Any] = {}
    for _, train_cfg, run_dir, out_path, record in sorted(todo, key=lambda item: item[0]):
        results.append(export_run(train_cfg, run_dir, out_path, settings, cache, record))

    report_path = report_dir / f"export_check_shard{args.shard_index}of{args.shard_count}.json"
    write_json_atomic(report_path, results)
    failed = [r for r in results if r.get("match") is False]
    matched = sum(1 for r in results if r.get("match"))
    print(f"[export] matched={matched} mismatched={len(failed)} skipped={len(results) - matched - len(failed)} "
          f"report={report_path}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
