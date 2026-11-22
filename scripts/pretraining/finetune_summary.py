from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

LABELS = ["is_feature_envy", "is_long_method", "is_blob", "is_data_class"]
SHORT = ["FE", "LM", "Blob", "DC"]
# label of a run in ssl_finetune_scaling.sbatch -> (starting weights, number of GPUs)
RUN_LABELS = {
    "scratch_1gpu": ("random", 1),
    "pretrained_1gpu": ("encoder_last.pt", 1),
    "pretrained_2gpu": ("encoder_last.pt", 2),
    "pretrained_4gpu": ("encoder_last.pt", 4),
    "pretrained_epoch1_1gpu": ("encoder_epoch1.pt", 1),
}


def load_runs(artifacts: Path, epochs: int) -> list[dict]:
    runs = []
    for metrics_path in sorted((artifacts / "runs").glob("*/metrics.json")):
        meta_path = metrics_path.parent / "stage_meta.json"
        if not meta_path.exists():
            continue
        cfg = json.loads(meta_path.read_text())["payload"].get("config", {})
        if cfg.get("batch_graphs") is None or int(cfg.get("epochs", 0)) != epochs:
            continue
        m = json.loads(metrics_path.read_text())
        prof = m["profiling"]
        init = str(cfg.get("pretrained_checkpoint")) if cfg.get("init_from_pretraining") else "random"
        tuned = m["test_tuned"]
        runs.append({
            "run": metrics_path.parent.name[:8],
            "init": init,
            "gpus": int(prof.get("world_size", 1)),
            "seed": int(cfg.get("seed", -1)),
            "f1_macro": float(tuned["f1_macro"]),
            "f1": [float(tuned["per_label"][name]["f1"]) for name in LABELS],
            "mcc_macro": float(tuned["mcc_macro"]),
            "pr_auc_macro": float(tuned["pr_auc_macro"]),
            "f1_macro_fixed": float(m["test_fixed_0_5"]["f1_macro"]),
            "best_epoch": int(m["best_epoch"]),
            "epochs_run": int(prof["epochs_run"]),
            "epoch_sec": float(prof["epoch_train_sec_mean"]),
            "step_ms": float(prof.get("optimizer_step_ms_mean") or float("nan")),
            "graphs_per_sec": float(prof["train_graphs_per_sec_mean"]),
            "peak_gb": float(prof["peak_gpu_memory_mb_max_over_gpus"]) / 1024.0,
            "gpu_name": prof.get("gpu_name"),
        })
    return runs


def gpu_use(log_dir: Path) -> dict[tuple[str, int], list[float]]:
    """Mean use (%) of the GPUs of a run, over the samples where a GPU holds more than 1 GiB."""
    use: dict[tuple[str, int], list[float]] = defaultdict(list)
    for path in sorted(log_dir.glob("gpu_*.csv")):
        label = next((name for name in RUN_LABELS if path.stem.endswith("_" + name)), None)
        if label is None:
            continue
        samples = []
        with path.open() as handle:
            for row in csv.reader(handle):
                if len(row) < 4 or not row[1].strip().isdigit():
                    continue
                if float(row[3]) > 1024.0:
                    samples.append(float(row[2]))
        if samples:
            use[RUN_LABELS[label]].append(float(np.mean(samples)))
    return use


def mean_std(values: list[float], digits: int = 3) -> str:
    if not values:
        return "-"
    return f"{np.mean(values):.{digits}f} ± {np.std(values):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tables of the fine-tuning runs: effect of the pre-training and scaling on 1, 2 and 4 GPUs.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--gpu-logs", type=Path, default=Path("logs"))
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    runs = load_runs(args.artifacts, args.epochs)
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for run in runs:
        groups[(run["init"], run["gpus"])].append(run)
    use = gpu_use(args.gpu_logs)

    print(f"{len(runs)} runs, GPU: {sorted({str(r['gpu_name']) for r in runs})}")
    print("\nDetection on the test part (thresholds tuned on the validation part), mean ± std over the seeds")
    print(f"{'weights':<20} {'GPUs':>4} {'seeds':>5}  {'macro F1':<15} " + " ".join(f"{s:<15}" for s in SHORT)
          + f" {'macro MCC':<15} {'PR-AUC':<15}")
    for key in sorted(groups):
        rs = groups[key]
        per_label = [mean_std([r["f1"][i] for r in rs]) for i in range(len(LABELS))]
        print(f"{key[0]:<20} {key[1]:>4} {len(rs):>5}  {mean_std([r['f1_macro'] for r in rs]):<15} "
              + " ".join(f"{v:<15}" for v in per_label)
              + f" {mean_std([r['mcc_macro'] for r in rs]):<15} {mean_std([r['pr_auc_macro'] for r in rs]):<15}")

    # paired difference between the pre-trained and the random starting weights, seed by seed
    for init in sorted({k[0] for k in groups if k[0] != "random"}):
        base = {r["seed"]: r for r in groups.get(("random", 1), [])}
        other = {r["seed"]: r for r in groups.get((init, 1), [])}
        seeds = sorted(set(base) & set(other))
        if seeds:
            diff = [other[s]["f1_macro"] - base[s]["f1_macro"] for s in seeds]
            print(f"\nmacro F1 of {init} minus random weights, 1 GPU, seeds {seeds}: "
                  f"{mean_std(diff)}  per seed {[round(d, 3) for d in diff]}")

    print("\nTraining speed from the pre-trained encoder (first epoch left out), mean ± std over the seeds")
    print(f"{'GPUs':>4} {'seeds':>5}  {'graphs/s':<16} {'speedup':<14} {'efficiency':<14} {'epoch (s)':<14} "
          f"{'step of 32 graphs (ms)':<24} {'peak memory (GB)':<18} {'GPU use (%)':<14}")
    single = {r["seed"]: r for r in groups.get(("encoder_last.pt", 1), [])}
    table = {}
    for gpus in (1, 2, 4):
        rs = groups.get(("encoder_last.pt", gpus), [])
        if not rs:
            continue
        speedup = [r["graphs_per_sec"] / single[r["seed"]]["graphs_per_sec"] for r in rs if r["seed"] in single]
        table[gpus] = {
            "seeds": sorted(r["seed"] for r in rs),
            "graphs_per_sec": [r["graphs_per_sec"] for r in rs],
            "speedup": speedup,
            "epoch_sec": [r["epoch_sec"] for r in rs],
            "step_ms": [r["step_ms"] for r in rs],
            "peak_gb": [r["peak_gb"] for r in rs],
            "gpu_use": use.get(("encoder_last.pt", gpus), []),
        }
        t = table[gpus]
        print(f"{gpus:>4} {len(rs):>5}  {mean_std(t['graphs_per_sec'], 1):<16} {mean_std(speedup, 2):<14} "
              f"{mean_std([s / gpus for s in speedup], 2):<14} {mean_std(t['epoch_sec'], 1):<14} "
              f"{mean_std(t['step_ms'], 1):<24} {mean_std(t['peak_gb'], 1):<18} {mean_std(t['gpu_use'], 0):<14}")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({"runs": runs, "scaling": table}, indent=2))
        print(f"\nwritten {args.out_json}")


if __name__ == "__main__":
    main()
