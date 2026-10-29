from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

from .config import set_nested_value
from .constants import LABEL_ORDER
from .reporting import generate_ablation_figure, generate_training_figures
from .stats import run_baseline_tests, run_pairwise_tests
from .token_encoder import train_word2vec_from_dot
from .visualization import generate_publication_figures


def stable_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [jsonable(value) for value in obj]
    return obj


def to_path(project_root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


@dataclass
class StageState:
    name: str
    fingerprint: str
    stage_dir: str
    reused: bool
    outputs: dict[str, str]


class PipelineRunner:
    STAGE_ORDER: tuple[str, ...] = (
        "normalization",
        "construction",
        "folds",
        "metric_extraction",
        "dataset",
        "token_encoder",
        "token_dataset",
        "pretraining",
        "training",
        "reporting",
    )

    STAGE_RUNTIME_ONLY_KEYS: dict[str, set[str]] = {
        "construction": {"workers", "chunksize", "progress_every", "overwrite"},
        "dataset": {"progress_style", "progress_every", "memory_stats"},
        "metric_extraction": {"workers", "batch_size"},
        "token_dataset": {},
    }

    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        self.config = config
        self.project_root = project_root.resolve()

    def _config_for_fingerprint(self, stage_name: str, stage_cfg: dict[str, Any]) -> dict[str, Any]:
        excluded_keys = self.STAGE_RUNTIME_ONLY_KEYS.get(stage_name, set())
        if not excluded_keys:
            return dict(stage_cfg)
        return {
            key: value
            for key, value in stage_cfg.items()
            if key not in excluded_keys
        }

    def _stage_output_hint(self, stage_state: StageState) -> str:
        preferred = [
            "run_dir",
            "metrics_path",
            "metadata_path",
            "dataset_path",
            "vectors_path",
            "manifest_path",
            "dot_dir",
            "figure_index_path",
        ]
        for key in preferred:
            if key in stage_state.outputs:
                return stage_state.outputs[key]
        if stage_state.outputs:
            return next(iter(stage_state.outputs.values()))
        return stage_state.stage_dir

    def _run_stage_with_logs(
        self,
        stage_name: str,
        runner: Callable[[], StageState],
    ) -> StageState:
        print(f"[pipeline] {stage_name}", end="  ", flush=True)
        started = time.perf_counter()
        stage_state = runner()
        elapsed = time.perf_counter() - started
        mode = "reused" if stage_state.reused else "rebuilt"
        fp_short = stage_state.fingerprint[:8]
        print(f"{mode}  {elapsed:.1f}s  {fp_short}")
        return stage_state

    def _normalize_retry_from(self, value: Any) -> str | None:
        if value is None:
            return None

        retry_from = str(value).strip().lower()
        if retry_from in {"", "none", "null"}:
            return None

        aliases = {
            "normalization": "normalization",
            "normalize": "normalization",
            "construction": "construction",
            "folds": "folds",
            "metric_extraction": "metric_extraction",
            "metric-extraction": "metric_extraction",
            "metrics": "metric_extraction",
            "dataset": "dataset",
            "token_encoder": "token_encoder",
            "token-encoder": "token_encoder",
            "tokenencoder": "token_encoder",
            "token_dataset": "token_dataset",
            "token-dataset": "token_dataset",
            "pretraining": "pretraining",
            "training": "training",
            "reporting": "reporting",
        }
        if retry_from not in aliases:
            allowed = ", ".join(self.STAGE_ORDER)
            raise ValueError(f"run.retry_from must be one of: {allowed}")
        return aliases[retry_from]

    def _forced_stages(self, config: dict[str, Any]) -> set[str]:
        run_cfg = config.get("run", {})
        retry_from = self._normalize_retry_from(run_cfg.get("retry_from"))
        if retry_from is None:
            return set()

        stage_index = self.STAGE_ORDER.index(retry_from)
        forced: set[str] = {str(stage) for stage in self.STAGE_ORDER[stage_index:]}
        forced_display = ", ".join(self.STAGE_ORDER[stage_index:])
        print(f"[pipeline] retry_from={retry_from} -> forcing stages: {forced_display}")
        return forced

    def run(self) -> dict[str, Any]:
        ablation_cfg = self.config.get("ablation", {})
        if bool(ablation_cfg.get("enabled", False)):
            return self._run_ablation()
        return self._run_single(self.config)

    def _run_single(self, config: dict[str, Any]) -> dict[str, Any]:
        pipeline_started = time.perf_counter()
        run_cfg = config.get("run", {})
        artifacts_root = to_path(self.project_root, run_cfg.get("artifacts_root", "artifacts"))
        artifacts_root.mkdir(parents=True, exist_ok=True)
        forced_stages = self._forced_stages(config)

        stages: dict[str, StageState] = {}

        normalization_state = None
        if bool(config.get("normalization", {}).get("enabled", False)):
            normalization_state = self._run_stage_with_logs(
                "normalization",
                lambda: self._run_normalization(
                    config,
                    artifacts_root,
                    force_stage=("normalization" in forced_stages),
                ),
            )
            stages["normalization"] = normalization_state

        construction_state = None
        if bool(config.get("construction", {}).get("enabled", True)):
            construction_state = self._run_stage_with_logs(
                "construction",
                lambda: self._run_construction(
                    config,
                    artifacts_root,
                    force_stage=("construction" in forced_stages),
                ),
            )
            stages["construction"] = construction_state

        folds_state = None
        if bool(config.get("folds", {}).get("enabled", False)):
            if construction_state is None:
                raise ValueError("Folds stage requires construction stage output")
            folds_state = self._run_stage_with_logs(
                "folds",
                lambda: self._run_folds(
                    config,
                    artifacts_root,
                    construction_state,
                    force_stage=("folds" in forced_stages),
                ),
            )
            stages["folds"] = folds_state

        metric_extraction_state = None
        if bool(config.get("metric_extraction", {}).get("enabled", False)):
            if normalization_state is None:
                raise ValueError("Metric extraction stage requires normalization stage output")
            metric_extraction_state = self._run_stage_with_logs(
                "metric_extraction",
                lambda: self._run_metric_extraction(
                    config,
                    artifacts_root,
                    normalization_state,
                    force_stage=("metric_extraction" in forced_stages),
                ),
            )
            stages["metric_extraction"] = metric_extraction_state

        dataset_state = None
        if bool(config.get("dataset", {}).get("enabled", True)):
            if construction_state is None:
                raise ValueError("Dataset stage requires construction stage output")
            dataset_state = self._run_stage_with_logs(
                "dataset",
                lambda: self._run_dataset(
                    config,
                    artifacts_root,
                    construction_state,
                    force_stage=("dataset" in forced_stages),
                ),
            )
            stages["dataset"] = dataset_state

        token_state = None
        if bool(config.get("token_encoder", {}).get("enabled", False)):
            if construction_state is None:
                raise ValueError("Token encoder stage requires construction stage output")
            token_state = self._run_stage_with_logs(
                "token_encoder",
                lambda: self._run_token_encoder(
                    config,
                    artifacts_root,
                    construction_state,
                    force_stage=("token_encoder" in forced_stages),
                ),
            )
            stages["token_encoder"] = token_state

        token_dataset_state = None
        if bool(config.get("token_dataset", {}).get("enabled", False)):
            if normalization_state is None:
                raise ValueError("Token dataset stage requires normalization stage output")
            token_dataset_state = self._run_stage_with_logs(
                "token_dataset",
                lambda: self._run_token_dataset(
                    config,
                    artifacts_root,
                    normalization_state,
                    force_stage=("token_dataset" in forced_stages),
                ),
            )
            stages["token_dataset"] = token_dataset_state

        pretraining_state = None
        if bool(config.get("pretraining", {}).get("enabled", False)):
            pretraining_state = self._run_stage_with_logs(
                "pretraining",
                lambda: self._run_pretraining(
                    config,
                    artifacts_root,
                    force_stage=("pretraining" in forced_stages),
                ),
            )
            stages["pretraining"] = pretraining_state

        # Init W&B before training so the Trainer logs epochs live.
        # Skip if ablation already injected a run via _wandb_run.
        wandb_run = config.get("_wandb_run")
        wandb_enabled = bool(run_cfg.get("wandb", {}).get("enabled", False))
        training_enabled = bool(config.get("training", {}).get("enabled", True))
        if wandb_enabled and wandb_run is None and training_enabled:
            try:
                import wandb

                wandb_cfg = run_cfg.get("wandb", {})
                wandb_run = wandb.init(
                    project=str(wandb_cfg.get("project", "mlcq_graphs")),
                    entity=wandb_cfg.get("entity"),
                    name=str(run_cfg.get("name", "run")),
                    config=jsonable(config),
                    mode="online",
                    tags=wandb_cfg.get("tags", []),
                )
                config["_wandb_run"] = wandb_run
            except ImportError:
                print("[pipeline] wandb not installed; skipping live logging")
            except Exception as exc:
                print(f"[pipeline] W&B init failed: {exc!r}")

        training_state = None
        if training_enabled:
            used_method = str(config.get("training", {}).get("used_method", "gnn")).lower()
            if used_method == "gnn":
                if dataset_state is None or construction_state is None:
                    raise ValueError("GNN training requires construction and dataset stages")
            elif used_method == "classical":
                if metric_extraction_state is None:
                    raise ValueError("Classical training requires metric_extraction stage")
            elif used_method == "sequence":
                if token_dataset_state is None:
                    raise ValueError("Sequence training requires token_dataset stage")
            elif used_method == "chunked_sequence":
                if folds_state is None and not config.get("training", {}).get("folds_json"):
                    raise ValueError("Chunked sequence training requires the folds stage or training.folds_json")
            training_state = self._run_stage_with_logs(
                "training",
                lambda: self._run_training(
                    config=config,
                    artifacts_root=artifacts_root,
                    construction_state=construction_state,
                    dataset_state=dataset_state,
                    token_state=token_state,
                    metric_extraction_state=metric_extraction_state,
                    token_dataset_state=token_dataset_state,
                    normalization_state=normalization_state,
                    pretraining_state=pretraining_state,
                    force_stage=("training" in forced_stages),
                    folds_state=folds_state,
                ),
            )
            stages["training"] = training_state

        report_state = None
        if bool(config.get("reporting", {}).get("enabled", True)) and training_state is not None:
            # reporting currently only supports the GNN training output format
            # (metrics.json + curves.json). Skip gracefully for classical / sequence.
            _used_method = str(config.get("training", {}).get("used_method", "gnn")).lower()
            if _used_method == "gnn":
                report_state = self._run_stage_with_logs(
                    "reporting",
                    lambda: self._run_reporting(config, training_state),
                )
                stages["reporting"] = report_state
            else:
                print(f"[pipeline] reporting skipped (used_method={_used_method}; GNN-only for now)")

        # Log test metrics/figures to live W&B run, then finish it
        # Skip for non-GNN used_method because _log_wandb reads metrics.json format that
        # is specific to train_gcn_baseline output for now.
        #TODO : refactor _log_wandb to be more flexible and support classical/sequence outputs, then log for those as well.
        _used_method = str(config.get("training", {}).get("used_method", "gnn")).lower()
        if _used_method == "gnn":
            if wandb_run is not None and training_state is not None:
                self._log_wandb(
                    config=config,
                    training_state=training_state,
                    report_state=report_state,
                    wandb_run=wandb_run,
                )
                wandb_run.finish()
                config.pop("_wandb_run", None)
            elif wandb_enabled and training_state is not None:
                # Fallback: no live run (import failed etc.), log post-hoc
                self._log_wandb(config=config, training_state=training_state, report_state=report_state)
        else:
            if wandb_run is not None:
                try:
                    wandb_run.finish()
                except Exception:
                    pass
                config.pop("_wandb_run", None)

        elapsed_total = time.perf_counter() - pipeline_started
        if elapsed_total >= 60:
            mins = int(elapsed_total // 60)
            secs = elapsed_total % 60
            print(f"[pipeline] done  {mins}m {secs:.1f}s")
        else:
            print(f"[pipeline] done  {elapsed_total:.1f}s")

        return {
            "artifacts_root": str(artifacts_root),
            "elapsed_sec": round(elapsed_total, 3),
            "stages": {name: asdict(state) for name, state in stages.items()},
        }

    def _run_ablation(self) -> dict[str, Any]:
        base_cfg = copy.deepcopy(self.config)
        ablation_cfg = base_cfg.get("ablation", {})
        grid = ablation_cfg.get("grid", {})
        if not isinstance(grid, dict) or not grid:
            raise ValueError("ablation.grid must be a non-empty mapping")
        # a null value drops a key inherited from defaults.yml (the grid is deep-merged)
        grid = {key: value for key, value in grid.items() if value is not None}
        if not grid:
            raise ValueError("ablation.grid has no key left after dropping the null ones")

        seeds = ablation_cfg.get("seeds", [base_cfg.get("training", {}).get("seed", 42)])
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("ablation.seeds must be a non-empty list")

        # Optional sharding so that one ablation can be spread over a job array:
        # shard `index` runs every trial whose position modulo `count` equals it.
        shard_cfg = ablation_cfg.get("shard") or {}
        shard_count = int(shard_cfg.get("count", 1))
        shard_index = int(shard_cfg.get("index", 0))
        if shard_count < 1 or not 0 <= shard_index < shard_count:
            raise ValueError("ablation.shard needs count >= 1 and 0 <= index < count")

        run_cfg = base_cfg.get("run", {})
        artifacts_root = to_path(self.project_root, run_cfg.get("artifacts_root", "artifacts"))
        base_name = str(run_cfg.get("name", "ablation"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shard_suffix = f"_shard{shard_index}of{shard_count}" if shard_count > 1 else ""
        report_dir = artifacts_root / "reports" / f"{base_name}_{timestamp}{shard_suffix}"
        report_dir.mkdir(parents=True, exist_ok=True)

        grid_keys = list(grid.keys())
        value_lists = [value if isinstance(value, list) else [value] for value in grid.values()]
        combos = list(itertools.product(*value_lists))

        records: list[dict[str, Any]] = []
        failed_records: list[dict[str, Any]] = []
        total_trials = len(combos) * len(seeds)
        trial_num = 0
        for combo_idx, combo_values in enumerate(combos, start=1):
            combo = {grid_keys[i]: combo_values[i] for i in range(len(grid_keys))}
            for seed in seeds:
                trial_num += 1
                if (trial_num - 1) % shard_count != shard_index:
                    continue
                trial_cfg = copy.deepcopy(base_cfg)
                trial_cfg.setdefault("ablation", {})["enabled"] = False
                trial_cfg.setdefault("training", {})["seed"] = int(seed)
                for dotted_key, value in combo.items():
                    set_nested_value(trial_cfg, dotted_key, value)
                trial_cfg.setdefault("run", {})["name"] = (
                    f"{base_name}_c{combo_idx:03d}_s{seed}"
                )

                combo_display = " ".join(f"{k.split('.')[-1]}={v}" for k, v in sorted(combo.items()))
                print(f"\n[ablation] Trial {trial_num}/{total_trials}: {combo_display} seed={seed}")

                # Init W&B run BEFORE training so per-epoch metrics stream live
                wandb_run = None
                wandb_enabled = bool(run_cfg.get("wandb", {}).get("enabled", False))
                group_name = "_".join(f"{k.split('.')[-1]}={v}" for k, v in sorted(combo.items()))
                if wandb_enabled:
                    try:
                        import wandb  # type: ignore[import-not-found]
                        wandb_cfg = run_cfg.get("wandb", {})
                        combo_tags = [f"{k.split('.')[-1]}={v}" for k, v in sorted(combo.items())]
                        trial_tags = combo_tags + [f"seed={seed}", "ablation"] + list(wandb_cfg.get("tags", []))
                        wandb_run = wandb.init(
                            project=str(wandb_cfg.get("project", "mlcq_graphs")),
                            entity=wandb_cfg.get("entity"),
                            name=f"{base_name}_c{combo_idx:03d}_s{seed}",
                            config=jsonable(trial_cfg),
                            mode="online",
                            tags=trial_tags,
                            group=group_name,
                            reinit=True,
                        )
                    except ImportError:
                        print("[ablation] wandb not installed; skipping live logging")
                    except Exception as wandb_exc:
                        print(f"[ablation] W&B init failed: {wandb_exc!r}")

                # Pass wandb_run into pipeline so Trainer gets it
                if wandb_run is not None:
                    trial_cfg["_wandb_run"] = wandb_run

                try:
                    trial_result = self._run_single(trial_cfg)
                    stage_info = trial_result["stages"].get("training")
                    if stage_info is None:
                        raise ValueError("Training stage was not executed in ablation trial")

                    metrics_path = Path(stage_info["outputs"]["metrics_path"])
                    metrics = json.loads(metrics_path.read_text())

                    record = {
                        "seed": int(seed),
                        "combo": combo,
                        "run_dir": stage_info["outputs"]["run_dir"],
                        "best_epoch": int(metrics.get("best_epoch", -1)),
                        "f1_micro_fixed": float(metrics.get("test_fixed_0_5", {}).get("f1_micro", 0.0)),
                        "f1_macro_fixed": float(metrics.get("test_fixed_0_5", {}).get("f1_macro", 0.0)),
                        "f1_micro_tuned": float(metrics.get("test_tuned", {}).get("f1_micro", 0.0)),
                        "f1_macro_tuned": float(metrics.get("test_tuned", {}).get("f1_macro", 0.0)),
                        "pr_auc_tuned": float(metrics.get("test_tuned", {}).get("pr_auc_macro", 0.0)),
                        "hamming_loss_tuned": float(metrics.get("test_tuned", {}).get("hamming_loss", 0.0)),
                        "subset_acc_tuned": float(metrics.get("test_tuned", {}).get("subset_accuracy", 0.0)),
                        "pr_auc_fixed": float(metrics.get("test_fixed_0_5", {}).get("pr_auc_macro", 0.0)),
                        "hamming_loss_fixed": float(metrics.get("test_fixed_0_5", {}).get("hamming_loss", 0.0)),
                        "subset_acc_fixed": float(metrics.get("test_fixed_0_5", {}).get("subset_accuracy", 0.0)),
                        "per_label_tuned": metrics.get("test_tuned", {}).get("per_label", {}),
                        "balanced_acc_tuned": float(metrics.get("test_tuned", {}).get("balanced_acc_macro", 0.0)),
                        "mcc_tuned": float(metrics.get("test_tuned", {}).get("mcc_macro", 0.0)),
                        "roc_auc_tuned": float(metrics.get("test_tuned", {}).get("roc_auc_macro", 0.0)),
                        "train_duration_sec": float(metrics.get("profiling", {}).get("train_duration_sec", 0.0)),
                        "peak_gpu_memory_mb": float(metrics.get("profiling", {}).get("peak_gpu_memory_mb", 0.0)),
                        "train_throughput": float(metrics.get("profiling", {}).get("train_throughput_graphs_per_sec", 0.0)),
                    }
                    records.append(record)

                    # Log test metrics to W&B (per-epoch was already logged live via Trainer)
                    if wandb_run is not None:
                        training_state = StageState(
                            name="training",
                            fingerprint=stage_info["fingerprint"],
                            stage_dir=stage_info["stage_dir"],
                            reused=stage_info["reused"],
                            outputs=stage_info["outputs"],
                        )
                        try:
                            self._log_wandb(
                                config=trial_cfg,
                                training_state=training_state,
                                report_state=None,
                                group=group_name,
                                wandb_run=wandb_run,
                            )
                        except Exception as wandb_exc:
                            print(f"[ablation] W&B logging failed for trial {trial_num}: {wandb_exc!r}")

                except Exception as exc:
                    print(f"[ablation] trial FAILED: {exc!r} -- continuing")
                    failed_records.append({"seed": int(seed), "combo": combo, "error": str(exc)})
                finally:
                    trial_cfg.pop("_wandb_run", None)
                    if wandb_run is not None:
                        try:
                            wandb_run.finish()
                        except Exception:
                            pass
                        wandb_run = None

        if not records:
            raise RuntimeError("all ablation trials failed; no results to aggregate")

        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            key = json.dumps(record["combo"], sort_keys=True)
            grouped.setdefault(key, []).append(record)

        summary_rows: list[dict[str, Any]] = []
        summary_extended_rows: list[dict[str, Any]] = []
        for key, items in grouped.items():
            combo = json.loads(key)
            micro_vals = [row["f1_micro_tuned"] for row in items]
            macro_vals = [row["f1_macro_tuned"] for row in items]
            pr_vals = [row["pr_auc_tuned"] for row in items]

            summary_rows.append(
                {
                    "combo": combo,
                    "num_runs": len(items),
                    "f1_micro_tuned_mean": statistics.mean(micro_vals),
                    "f1_micro_tuned_std": statistics.pstdev(micro_vals),
                    "f1_macro_tuned_mean": statistics.mean(macro_vals),
                    "f1_macro_tuned_std": statistics.pstdev(macro_vals),
                    "pr_auc_tuned_mean": statistics.mean(pr_vals),
                    "pr_auc_tuned_std": statistics.pstdev(pr_vals),
                }
            )

            # per-label aggregation across seeds
            per_label_means: dict[str, dict[str, float]] = {}
            for label in LABEL_ORDER:
                label_f1 = [
                    float(row["per_label_tuned"].get(label, {}).get("f1", 0.0))
                    for row in items
                ]
                label_prec = [
                    float(row["per_label_tuned"].get(label, {}).get("precision", 0.0))
                    for row in items
                ]
                label_rec = [
                    float(row["per_label_tuned"].get(label, {}).get("recall", 0.0))
                    for row in items
                ]
                per_label_means[label] = {
                    "f1_mean": statistics.mean(label_f1),
                    "f1_std": statistics.pstdev(label_f1),
                    "precision_mean": statistics.mean(label_prec),
                    "precision_std": statistics.pstdev(label_prec),
                    "recall_mean": statistics.mean(label_rec),
                    "recall_std": statistics.pstdev(label_rec),
                }

            micro_fixed_vals = [row["f1_micro_fixed"] for row in items]
            macro_fixed_vals = [row["f1_macro_fixed"] for row in items]
            hamming_tuned_vals = [row["hamming_loss_tuned"] for row in items]
            subset_tuned_vals = [row["subset_acc_tuned"] for row in items]
            pr_fixed_vals = [row["pr_auc_fixed"] for row in items]
            train_dur_vals = [row["train_duration_sec"] for row in items]
            gpu_mem_vals = [row["peak_gpu_memory_mb"] for row in items]
            throughput_vals = [row["train_throughput"] for row in items]
            ba_vals = [row["balanced_acc_tuned"] for row in items]
            mcc_vals = [row["mcc_tuned"] for row in items]
            roc_auc_vals = [row["roc_auc_tuned"] for row in items]

            summary_extended_rows.append(
                {
                    "combo": combo,
                    "num_runs": len(items),
                    # existing fields
                    "f1_micro_tuned_mean": statistics.mean(micro_vals),
                    "f1_micro_tuned_std": statistics.pstdev(micro_vals),
                    "f1_macro_tuned_mean": statistics.mean(macro_vals),
                    "f1_macro_tuned_std": statistics.pstdev(macro_vals),
                    "pr_auc_tuned_mean": statistics.mean(pr_vals),
                    "pr_auc_tuned_std": statistics.pstdev(pr_vals),
                    # extended aggregate metrics
                    "f1_micro_fixed_mean": statistics.mean(micro_fixed_vals),
                    "f1_micro_fixed_std": statistics.pstdev(micro_fixed_vals),
                    "f1_macro_fixed_mean": statistics.mean(macro_fixed_vals),
                    "f1_macro_fixed_std": statistics.pstdev(macro_fixed_vals),
                    "hamming_loss_tuned_mean": statistics.mean(hamming_tuned_vals),
                    "hamming_loss_tuned_std": statistics.pstdev(hamming_tuned_vals),
                    "subset_acc_tuned_mean": statistics.mean(subset_tuned_vals),
                    "subset_acc_tuned_std": statistics.pstdev(subset_tuned_vals),
                    "pr_auc_fixed_mean": statistics.mean(pr_fixed_vals),
                    "pr_auc_fixed_std": statistics.pstdev(pr_fixed_vals),
                    # extended metrics means
                    "balanced_acc_tuned_mean": statistics.mean(ba_vals),
                    "balanced_acc_tuned_std": statistics.pstdev(ba_vals),
                    "mcc_tuned_mean": statistics.mean(mcc_vals),
                    "mcc_tuned_std": statistics.pstdev(mcc_vals),
                    "roc_auc_tuned_mean": statistics.mean(roc_auc_vals),
                    "roc_auc_tuned_std": statistics.pstdev(roc_auc_vals),
                    # per-label means
                    "per_label_means": per_label_means,
                    # profiling means
                    "avg_train_duration_sec": statistics.mean(train_dur_vals),
                    "avg_peak_gpu_memory_mb": statistics.mean(gpu_mem_vals),
                    "avg_train_throughput": statistics.mean(throughput_vals),
                }
            )

        # significance tests vs configurable baseline
        combo_scores: dict[str, list[float]] = {
            key: [row["f1_macro_tuned"] for row in items]
            for key, items in grouped.items()
        }
        baseline_cfg = ablation_cfg.get("baseline", {})
        baseline_key: str | None = None
        for key in grouped:
            combo_dict = json.loads(key)
            if baseline_cfg:
                # Match: combo is a superset of baseline config (all baseline keys match)
                match = all(
                    str(combo_dict.get(bk, "")) == str(bv)
                    if not isinstance(bv, list)
                    else combo_dict.get(bk) == bv
                    for bk, bv in baseline_cfg.items()
                )
                if match:
                    baseline_key = key
                    break
            else:
                # Legacy fallback: gcn+weighted_bce
                arch_val = combo_dict.get("training.architecture", combo_dict.get("architecture", ""))
                loss_val = combo_dict.get("training.loss", combo_dict.get("loss", ""))
                if str(arch_val).lower() == "gcn" and str(loss_val).lower() == "weighted_bce":
                    baseline_key = key
                    break

        significance_results: dict[str, Any] = {}
        if baseline_key is not None:
            significance_results["baseline_tests"] = run_baseline_tests(
                combo_scores, baseline_key, "f1_macro_tuned"
            )
            significance_results["pairwise_tests"] = run_pairwise_tests(
                combo_scores, "f1_macro_tuned"
            )
        else:
            significance_results["baseline_tests"] = {}
            significance_results["pairwise_tests"] = run_pairwise_tests(
                combo_scores, "f1_macro_tuned"
            )
            significance_results["note"] = "no baseline combo found; baseline_tests skipped"

        records_path = report_dir / "records.json"
        summary_path = report_dir / "summary.json"
        csv_path = report_dir / "summary.csv"
        summary_extended_path = report_dir / "summary_extended.json"

        records_path.write_text(json.dumps(jsonable(records), indent=2))
        summary_path.write_text(json.dumps(jsonable(summary_rows), indent=2))

        summary_extended_doc = {
            "summary": jsonable(summary_extended_rows),
            "significance": jsonable(significance_results),
            "metadata": {
                "baseline_key": baseline_key,
                "num_seeds": len(seeds),
                "metric_tested": "f1_macro_tuned",
                "note": f"n={len(seeds)} seeds; statistical power is limited" if len(seeds) <= 5 else None,
            },
        }
        summary_extended_path.write_text(json.dumps(summary_extended_doc, indent=2))

        with csv_path.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "combo",
                    "num_runs",
                    "f1_micro_tuned_mean",
                    "f1_micro_tuned_std",
                    "f1_macro_tuned_mean",
                    "f1_macro_tuned_std",
                    "pr_auc_tuned_mean",
                    "pr_auc_tuned_std",
                ]
            )
            for row in summary_rows:
                writer.writerow(
                    [
                        json.dumps(row["combo"], sort_keys=True),
                        row["num_runs"],
                        row["f1_micro_tuned_mean"],
                        row["f1_micro_tuned_std"],
                        row["f1_macro_tuned_mean"],
                        row["f1_macro_tuned_std"],
                        row["pr_auc_tuned_mean"],
                        row["pr_auc_tuned_std"],
                    ]
                )

        figure_path = report_dir / "ablation_f1_macro_tuned.png"
        generate_ablation_figure(summary_rows=summary_rows, output_path=figure_path, metric="f1_macro_tuned")

        # Console summary table: auto-detect columns from grid keys
        col_keys = grid_keys
        col_names = [k.split(".")[-1] for k in col_keys]
        col_widths = [max(len(name), 10) for name in col_names]

        header_sep_width = sum(col_widths) + len(col_widths) + 20 + 16
        print(f"\n{'=' * header_sep_width}")
        print(f" Ablation Summary (macro-F1 tuned) | {len(records)} trials")
        if failed_records:
            print(f" ({len(failed_records)} trial(s) failed)")
        print(f"{'=' * header_sep_width}")
        header_parts = [f"{name:<{w}}" for name, w in zip(col_names, col_widths)]
        print(f"{' '.join(header_parts)} {'Mean':>7} {'Std':>7}  {'vs baseline'}")
        print(f"{'-' * header_sep_width}")

        baseline_tests = significance_results.get("baseline_tests", {})

        for row in summary_extended_rows:
            combo = row["combo"]
            mean_f1 = row["f1_macro_tuned_mean"]
            std_f1 = row["f1_macro_tuned_std"]

            combo_key = json.dumps(combo, sort_keys=True)
            bt = baseline_tests.get(combo_key, {})
            p_val = bt.get("p_value")
            if combo_key == baseline_key:
                marker = "(baseline)"
            elif p_val is None:
                marker = "n/a"
            elif p_val < 0.05:
                marker = f"p={p_val:.3f} *"
            else:
                marker = f"p={p_val:.3f}"

            val_parts = []
            for gk, w in zip(col_keys, col_widths):
                val = combo.get(gk, "?")
                val_parts.append(f"{str(val):<{w}}")
            print(f"{' '.join(val_parts)} {mean_f1:>7.4f} {std_f1:>7.4f}  {marker}")

        note_seeds = len(seeds)
        if note_seeds <= 5:
            print(f"\n  Note: n={note_seeds} seeds; statistical power is limited")
        print()

        # W&B artifact upload
        if bool(run_cfg.get("wandb", {}).get("enabled", False)):
            try:
                import wandb  # type: ignore[import-not-found]
                wandb_cfg = run_cfg.get("wandb", {})
                run = wandb.init(
                    project=str(wandb_cfg.get("project", "mlcq_graphs")),
                    entity=wandb_cfg.get("entity"),
                    name=f"{base_name}_ablation_summary",
                    job_type="ablation-summary",
                    tags=["ablation", "summary"],
                )
                artifact = wandb.Artifact(f"{base_name}_ablation_outputs", type="results")
                artifact.add_file(str(figure_path))
                artifact.add_file(str(csv_path))
                artifact.add_file(str(summary_extended_path))
                run.log_artifact(artifact)
                run.finish()
            except ImportError:
                print("[pipeline] wandb not installed; skipping artifact upload")
            except Exception as exc:
                print(f"[ablation] W&B artifact upload failed: {exc!r}")

        return {
            "artifacts_root": str(artifacts_root),
            "ablation": {
                "num_combos": len(combos),
                "num_runs": len(records),
                "failed_trials": len(failed_records),
                "failed_records": failed_records,
                "records_path": str(records_path),
                "summary_path": str(summary_path),
                "summary_csv": str(csv_path),
                "summary_extended_path": str(summary_extended_path),
                "figure_path": str(figure_path),
            },
        }

    def _run_construction(
        self,
        config: dict[str, Any],
        artifacts_root: Path,
        force_stage: bool,
    ) -> StageState:
        stage_cfg = config.get("construction", {})
        run_cfg = config.get("run", {})

        fingerprint_cfg = self._config_for_fingerprint("construction", stage_cfg)

        payload = {
            "stage": "construction",
            "config": fingerprint_cfg,
        }
        fingerprint = stable_fingerprint(jsonable(payload))
        stage_dir = artifacts_root / "cache" / "construction" / fingerprint
        outputs = {
            "dot_dir": str(stage_dir / "dot"),
            "manifest_path": str(stage_dir / "manifest.jsonl"),
        }

        if self._can_reuse(
            stage_dir,
            outputs,
            force=bool(run_cfg.get("force_rebuild", False)) or force_stage,
        ):
            return StageState(
                name="construction",
                fingerprint=fingerprint,
                stage_dir=str(stage_dir),
                reused=True,
                outputs=outputs,
            )

        stage_dir.mkdir(parents=True, exist_ok=True)
        script_path = self.project_root / "scripts" / "build_ast_dot_from_normalized.py"

        # Optional relational augmentation: the plain AST graphs are built into
        # dot_base/ and the augmented graphs are written to the usual outputs.
        rel_cfg = stage_cfg.get("relational") or {}
        use_relational = bool(rel_cfg.get("enabled", False))
        base_dot_dir = str(stage_dir / "dot_base") if use_relational else outputs["dot_dir"]
        base_manifest = str(stage_dir / "manifest_base.jsonl") if use_relational else outputs["manifest_path"]

        mode = str(stage_cfg.get("mode", "from-json"))
        cmd = [sys.executable, str(script_path), mode]

        if mode == "from-json":
            cmd += [
                "--input-json",
                str(to_path(self.project_root, stage_cfg.get("input_json"))),
                "--output-dir",
                base_dot_dir,
                "--manifest-path",
                base_manifest,
                "--start-index",
                str(stage_cfg.get("start_index", 0)),
                "--workers",
                str(stage_cfg.get("workers", 1)),
                "--chunksize",
                str(stage_cfg.get("chunksize", 8)),
                "--progress-every",
                str(stage_cfg.get("progress_every", 100)),
            ]
            if stage_cfg.get("limit") is not None:
                cmd += ["--limit", str(stage_cfg.get("limit"))]
            if bool(stage_cfg.get("overwrite", False)):
                cmd.append("--overwrite")
        elif mode == "from-dir":
            cmd += [
                "--input-dir",
                str(to_path(self.project_root, stage_cfg.get("input_dir"))),
                "--output-dir",
                base_dot_dir,
                "--manifest-path",
                base_manifest,
                "--workers",
                str(stage_cfg.get("workers", 1)),
                "--chunksize",
                str(stage_cfg.get("chunksize", 8)),
                "--progress-every",
                str(stage_cfg.get("progress_every", 100)),
            ]
            if stage_cfg.get("limit") is not None:
                cmd += ["--limit", str(stage_cfg.get("limit"))]
            if bool(stage_cfg.get("overwrite", False)):
                cmd.append("--overwrite")
        else:
            raise ValueError(f"Unsupported construction mode: {mode}")

        # Parser backend (antlr by default); only passed when set so that the
        # fingerprints of existing ANTLR configurations do not change.
        if stage_cfg.get("parser") is not None:
            cmd += ["--parser", str(stage_cfg["parser"])]
        if bool(stage_cfg.get("allow_parse_errors", False)):
            cmd.append("--allow-parse-errors")

        self._run_command(cmd, print_commands=bool(run_cfg.get("print_commands", True)))

        if use_relational:
            if mode != "from-json":
                raise ValueError("construction.relational requires mode=from-json")
            rel_cmd = [
                sys.executable,
                str(self.project_root / "scripts" / "data" / "augment_dot_relational.py"),
                "--normalized",
                str(to_path(self.project_root, stage_cfg.get("input_json"))),
                "--context-dir",
                str(to_path(self.project_root, rel_cfg.get("context_dir", "data/context_files"))),
                "--dot-dir",
                base_dot_dir,
                "--manifest",
                base_manifest,
                "--out-dir",
                str(stage_dir),
                "--level",
                str(int(rel_cfg.get("level", 1))),
            ]
            self._run_command(rel_cmd, print_commands=bool(run_cfg.get("print_commands", True)))

        self._write_stage_meta(
            stage_dir=stage_dir,
            payload={
                "stage": "construction",
                "config": stage_cfg,
                "fingerprint_config": fingerprint_cfg,
            },
            outputs=outputs,
        )

        return StageState(
            name="construction",
            fingerprint=fingerprint,
            stage_dir=str(stage_dir),
            reused=False,
            outputs=outputs,
        )

    def _run_folds(
        self,
        config: dict[str, Any],
        artifacts_root: Path,
        construction_state: StageState,
        force_stage: bool,
    ) -> StageState:
        """Train, validation and test parts shared by every model, one per (strategy, seed, fold).

        The parts are those the GNN runs rebuild with split_kfold, over the snippets that have an AST,
        so that models without graphs read the same parts from folds.json.
        """
        stage_cfg = config.get("folds", {})
        run_cfg = config.get("run", {})
        num_labels = int(config.get("training", {}).get("num_labels", 4))

        payload = {
            "stage": "folds",
            "config": stage_cfg,
            "num_labels": num_labels,
            "construction_fingerprint": construction_state.fingerprint,
        }
        fingerprint = stable_fingerprint(jsonable(payload))
        stage_dir = artifacts_root / "cache" / "folds" / fingerprint
        outputs = {"folds_json": str(stage_dir / "folds.json")}

        if self._can_reuse(
            stage_dir,
            outputs,
            force=bool(run_cfg.get("force_rebuild", False)) or force_stage,
        ):
            return StageState(
                name="folds",
                fingerprint=fingerprint,
                stage_dir=str(stage_dir),
                reused=True,
                outputs=outputs,
            )

        strategies = [str(s) for s in stage_cfg.get("strategies", ["kfold"])]
        unknown = sorted(set(strategies) - {"kfold", "group_kfold"})
        if unknown:
            raise ValueError(f"folds.strategies supports kfold and group_kfold, got {unknown}")
        seeds = [int(s) for s in stage_cfg.get("seeds", [42])]
        n_folds = int(stage_cfg.get("n_folds", 5))
        val_ratio = float(stage_cfg.get("val_ratio", 0.1))
        group_key = str(stage_cfg.get("group_key", "repo_url"))

        # same rows and order as the dataset stage (scripts/build_pyg_dataset_from_dot.py):
        # success or skipped rows whose DOT file exists, sorted by json_index
        rows = []
        for line in Path(construction_state.outputs["manifest_path"]).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") not in {"success", "skipped"}:
                continue
            if not to_path(self.project_root, row["output_dot"]).exists():
                continue
            rows.append(row)
        rows.sort(key=lambda row: int(row["json_index"]))
        json_index = [int(row["json_index"]) for row in rows]
        labels = [[1 if bool(v) else 0 for v in row["y"]] for row in rows]
        input_json = to_path(self.project_root, config.get("construction", {}).get("input_json"))
        input_bytes = input_json.read_bytes()
        entries = json.loads(input_bytes)
        groups = [str(entries[i].get(group_key, "")) for i in json_index]

        from scripts.cv.export_cv_folds import build_cv_folds

        splits = build_cv_folds(json_index, labels, groups, strategies, seeds, n_folds, val_ratio, num_labels)
        folds = {
            "meta": {
                "construction_fingerprint": construction_state.fingerprint,
                "num_samples": len(json_index),
                "label_order": LABEL_ORDER[:num_labels],
                "n_folds": n_folds,
                "val_ratio": val_ratio,
                "seeds": seeds,
                "strategies": strategies,
                "group_key": group_key,
                "input_json": str(input_json),
                "input_json_md5": hashlib.md5(input_bytes).hexdigest(),
                "json_index": json_index,
            },
            "splits": splits,
        }
        stage_dir.mkdir(parents=True, exist_ok=True)
        Path(outputs["folds_json"]).write_text(json.dumps(folds))
        self._write_stage_meta(stage_dir=stage_dir, payload=payload, outputs=outputs)
        return StageState(
            name="folds",
            fingerprint=fingerprint,
            stage_dir=str(stage_dir),
            reused=False,
            outputs=outputs,
        )

    def _run_dataset(
        self,
        config: dict[str, Any],
        artifacts_root: Path,
        construction_state: StageState,
        force_stage: bool,
    ) -> StageState:
        stage_cfg = config.get("dataset", {})
        run_cfg = config.get("run", {})

        fingerprint_cfg = self._config_for_fingerprint("dataset", stage_cfg)

        payload = {
            "stage": "dataset",
            "config": fingerprint_cfg,
            "construction_fingerprint": construction_state.fingerprint,
        }
        fingerprint = stable_fingerprint(jsonable(payload))
        stage_dir = artifacts_root / "cache" / "dataset" / fingerprint
        outputs = {
            "dataset_path": str(stage_dir / "dataset.pt"),
            "node_type_vocab_path": str(stage_dir / "node_type_vocab.json"),
            "metadata_path": str(stage_dir / "dataset.meta.json"),
        }

        if self._can_reuse(
            stage_dir,
            outputs,
            force=bool(run_cfg.get("force_rebuild", False)) or force_stage,
        ):
            return StageState(
                name="dataset",
                fingerprint=fingerprint,
                stage_dir=str(stage_dir),
                reused=True,
                outputs=outputs,
            )

        stage_dir.mkdir(parents=True, exist_ok=True)
        script_path = self.project_root / "scripts" / "build_pyg_dataset_from_dot.py"

        edge_types = stage_cfg.get("edge_types", ["Child", "NextToken"])
        if isinstance(edge_types, list):
            edge_types_arg = ",".join(str(item) for item in edge_types)
        else:
            edge_types_arg = str(edge_types)

        cmd = [
            sys.executable,
            str(script_path),
            "--manifest-path",
            construction_state.outputs["manifest_path"],
            "--dot-dir",
            construction_state.outputs["dot_dir"],
            "--dataset-out",
            outputs["dataset_path"],
            "--node-type-vocab-out",
            outputs["node_type_vocab_path"],
            "--metadata-out",
            outputs["metadata_path"],
            "--edge-types",
            edge_types_arg,
            "--progress-style",
            str(stage_cfg.get("progress_style", "auto")),
            "--progress-every",
            str(stage_cfg.get("progress_every", 100)),
        ]

        if stage_cfg.get("max_graphs") is not None:
            cmd += ["--max-graphs", str(stage_cfg.get("max_graphs"))]
        if bool(stage_cfg.get("memory_stats", True)):
            cmd.append("--memory-stats")
        else:
            cmd.append("--no-memory-stats")
        if bool(stage_cfg.get("allow_dot_dir_fallback", False)):
            cmd.append("--allow-dot-dir-fallback")
        if bool(stage_cfg.get("store_edge_types", False)):
            cmd.append("--store-edge-types")

        self._run_command(cmd, print_commands=bool(run_cfg.get("print_commands", True)))
        self._write_stage_meta(
            stage_dir=stage_dir,
            payload={
                "stage": "dataset",
                "config": stage_cfg,
                "fingerprint_config": fingerprint_cfg,
                "construction_fingerprint": construction_state.fingerprint,
            },
            outputs=outputs,
        )

        return StageState(
            name="dataset",
            fingerprint=fingerprint,
            stage_dir=str(stage_dir),
            reused=False,
            outputs=outputs,
        )

    def _run_token_encoder(
        self,
        config: dict[str, Any],
        artifacts_root: Path,
        construction_state: StageState,
        force_stage: bool,
    ) -> StageState:
        stage_cfg = config.get("token_encoder", {})
        run_cfg = config.get("run", {})
        method = str(stage_cfg.get("method", "word2vec"))
        if method != "word2vec":
            raise ValueError(f"Unsupported token encoder method: {method}")

        payload = {
            "stage": "token_encoder",
            "config": stage_cfg,
            "construction_fingerprint": construction_state.fingerprint,
        }
        fingerprint = stable_fingerprint(jsonable(payload))
        stage_dir = artifacts_root / "cache" / "token_encoder" / fingerprint
        outputs = {
            "model_path": str(stage_dir / "word2vec.model"),
            "vectors_path": str(stage_dir / "word2vec.kv"),
            "vocab_path": str(stage_dir / "vocab.json"),
            "metadata_path": str(stage_dir / "meta.json"),
        }

        if self._can_reuse(
            stage_dir,
            outputs,
            force=bool(run_cfg.get("force_rebuild", False)) or force_stage,
        ):
            return StageState(
                name="token_encoder",
                fingerprint=fingerprint,
                stage_dir=str(stage_dir),
                reused=True,
                outputs=outputs,
            )

        meta = train_word2vec_from_dot(
            dot_dir=Path(construction_state.outputs["dot_dir"]),
            output_dir=stage_dir,
            vector_size=int(stage_cfg.get("vector_size", 128)),
            window=int(stage_cfg.get("window", 5)),
            min_count=int(stage_cfg.get("min_count", 1)),
            workers=int(stage_cfg.get("workers", 1)),
            epochs=int(stage_cfg.get("epochs", 10)),
            progress_every=int(stage_cfg.get("progress_every", 200)),
            max_graphs=(
                int(stage_cfg.get("max_graphs"))
                if stage_cfg.get("max_graphs") is not None
                else None
            ),
        )

        self._write_stage_meta(
            stage_dir=stage_dir,
            payload={**payload, "training_meta": meta},
            outputs=outputs,
        )
        return StageState(
            name="token_encoder",
            fingerprint=fingerprint,
            stage_dir=str(stage_dir),
            reused=False,
            outputs=outputs,
        )

    def _run_normalization(
        self,
        config: dict[str, Any],
        artifacts_root: Path,
        force_stage: bool,
    ) -> StageState:
        """Normalize raw MLCQ reviews into per-sample binarised labels."""
        stage_cfg = config.get("normalization", {})
        run_cfg = config.get("run", {})

        input_json_cfg = stage_cfg.get("input_json", "data/MLCQCodeSmellSamples.json")
        input_json = to_path(self.project_root, input_json_cfg)
        method = str(stage_cfg.get("method", "median"))
        rule = str(stage_cfg.get("rule", "default"))

        # Fingerprint includes input file hash so re-runs happen if raw data changes
        try:
            stat = input_json.stat()
            input_hint = f"{stat.st_size}:{int(stat.st_mtime)}"
        except FileNotFoundError:
            input_hint = "missing"

        payload = {
            "stage": "normalization",
            "method": method,
            "rule": rule,
            "input_hint": input_hint,
            "input_path": str(input_json),
        }
        fingerprint = stable_fingerprint(jsonable(payload))
        stage_dir = artifacts_root / "cache" / "normalization" / fingerprint
        rule_slug = rule.replace(",", "_")
        outputs = {
            "normalized_json_path": str(stage_dir / f"normalized.{rule_slug}.json"),
        }

        if self._can_reuse(
            stage_dir,
            outputs,
            force=bool(run_cfg.get("force_rebuild", False)) or force_stage,
        ):
            return StageState(
                name="normalization",
                fingerprint=fingerprint,
                stage_dir=str(stage_dir),
                reused=True,
                outputs=outputs,
            )

        stage_dir.mkdir(parents=True, exist_ok=True)
        from scripts.NormalizeFromJson import normalize_json_to_json

        normalize_json_to_json(
            input_json=input_json,
            output_json=Path(outputs["normalized_json_path"]),
            method=method,
            rule=rule,
        )

        self._write_stage_meta(
            stage_dir=stage_dir,
            payload=payload,
            outputs=outputs,
        )
        return StageState(
            name="normalization",
            fingerprint=fingerprint,
            stage_dir=str(stage_dir),
            reused=False,
            outputs=outputs,
        )

    def _run_metric_extraction(
        self,
        config: dict[str, Any],
        artifacts_root: Path,
        normalization_state: StageState,
        force_stage: bool,
    ) -> StageState:
        """Extract OO metrics from code snippets via DesigniteJava."""
        stage_cfg = config.get("metric_extraction", {})
        run_cfg = config.get("run", {})
        tool = str(stage_cfg.get("tool", "designite"))
        if tool != "designite":
            raise ValueError(f"Unsupported metric extraction tool: {tool}")

        fingerprint_cfg = self._config_for_fingerprint("metric_extraction", stage_cfg)
        payload = {
            "stage": "metric_extraction",
            "config": fingerprint_cfg,
            "normalization_fingerprint": normalization_state.fingerprint,
        }
        fingerprint = stable_fingerprint(jsonable(payload))
        stage_dir = artifacts_root / "cache" / "metric_extraction" / fingerprint
        outputs = {
            "metrics_csv_path": str(stage_dir / "metrics_dataset.csv"),
        }

        if self._can_reuse(
            stage_dir,
            outputs,
            force=bool(run_cfg.get("force_rebuild", False)) or force_stage,
        ):
            return StageState(
                name="metric_extraction",
                fingerprint=fingerprint,
                stage_dir=str(stage_dir),
                reused=True,
                outputs=outputs,
            )

        stage_dir.mkdir(parents=True, exist_ok=True)
        from scripts.extract_designite_metrics import extract as extract_designite

        extract_designite(
            input_json=Path(normalization_state.outputs["normalized_json_path"]),
            output_csv=Path(outputs["metrics_csv_path"]),
            batch_size=int(stage_cfg.get("batch_size", 500)),
            workers=int(stage_cfg.get("workers", 6)),
        )

        self._write_stage_meta(
            stage_dir=stage_dir,
            payload=payload,
            outputs=outputs,
        )
        return StageState(
            name="metric_extraction",
            fingerprint=fingerprint,
            stage_dir=str(stage_dir),
            reused=False,
            outputs=outputs,
        )

    def _run_token_dataset(
        self,
        config: dict[str, Any],
        artifacts_root: Path,
        normalization_state: StageState,
        force_stage: bool,
    ) -> StageState:
        """Build tokenised sequence dataset from normalised JSON for sequence DL."""
        stage_cfg = config.get("token_dataset", {})
        run_cfg = config.get("run", {})

        payload = {
            "stage": "token_dataset",
            "config": stage_cfg,
            "normalization_fingerprint": normalization_state.fingerprint,
        }
        fingerprint = stable_fingerprint(jsonable(payload))
        stage_dir = artifacts_root / "cache" / "token_dataset" / fingerprint
        outputs = {
            "token_dir": str(stage_dir),
            "dataset_path": str(stage_dir / "token_dataset.pt"),
            "vocab_path": str(stage_dir / "vocab.json"),
            "metadata_path": str(stage_dir / "meta.json"),
        }

        if self._can_reuse(
            stage_dir,
            outputs,
            force=bool(run_cfg.get("force_rebuild", False)) or force_stage,
        ):
            return StageState(
                name="token_dataset",
                fingerprint=fingerprint,
                stage_dir=str(stage_dir),
                reused=True,
                outputs=outputs,
            )

        stage_dir.mkdir(parents=True, exist_ok=True)
        from scripts.build_token_dataset import build_token_dataset

        build_token_dataset(
            input_json=str(Path(normalization_state.outputs["normalized_json_path"])),
            output_dir=str(stage_dir),
            max_tokens=int(stage_cfg.get("max_tokens", 512)),
            min_freq=int(stage_cfg.get("min_freq", 2)),
            max_vocab=int(stage_cfg.get("max_vocab", 10000)),
        )

        self._write_stage_meta(
            stage_dir=stage_dir,
            payload=payload,
            outputs=outputs,
        )
        return StageState(
            name="token_dataset",
            fingerprint=fingerprint,
            stage_dir=str(stage_dir),
            reused=False,
            outputs=outputs,
        )

    def _run_training(
        self,
        config: dict[str, Any],
        artifacts_root: Path,
        construction_state: StageState | None,
        dataset_state: StageState | None,
        token_state: StageState | None,
        metric_extraction_state: StageState | None,
        token_dataset_state: StageState | None,
        normalization_state: StageState | None = None,
        pretraining_state: StageState | None = None,
        force_stage: bool = False,
        folds_state: StageState | None = None,
    ) -> StageState:
        stage_cfg = config.get("training", {})
        used_method = str(stage_cfg.get("used_method", "gnn")).lower()

        if used_method == "chunked_sequence":
            return self._run_training_chunked_sequence(
                config=config,
                artifacts_root=artifacts_root,
                folds_state=folds_state,
                force_stage=force_stage,
            )
        if used_method == "classical":
            return self._run_training_classical(
                config=config,
                artifacts_root=artifacts_root,
                metric_extraction_state=metric_extraction_state,
                force_stage=force_stage,
            )
        if used_method == "sequence":
            return self._run_training_sequence(
                config=config,
                artifacts_root=artifacts_root,
                token_dataset_state=token_dataset_state,
                force_stage=force_stage,
            )
        if used_method == "codebert":
            return self._run_training_codebert(
                config=config,
                artifacts_root=artifacts_root,
                normalization_state=normalization_state,
                force_stage=force_stage,
            )
        if used_method != "gnn":
            raise ValueError(
                f"Unsupported training.used_method: {used_method!r}. "
                f"Expected one of: gnn, classical, sequence, codebert, chunked_sequence."
            )

        # GNN path (existing behaviour)
        fingerprint, payload, train_cfg = self.gnn_training_plan(
            config=config,
            artifacts_root=artifacts_root,
            construction_state=construction_state,
            dataset_state=dataset_state,
            token_state=token_state,
            pretraining_state=pretraining_state,
        )
        run_cfg = config.get("run", {})
        run_dir = artifacts_root / "runs" / fingerprint

        outputs = {
            "run_dir": str(run_dir),
            "metrics_path": str(run_dir / "metrics.json"),
            "config_path": str(run_dir / "config.json"),
            "best_model_path": str(run_dir / "best_model.pt"),
        }

        if self._can_reuse(
            run_dir,
            outputs,
            force=bool(run_cfg.get("force_rebuild", False)) or force_stage,
        ):
            if train_cfg.get("save_predictions") and not (run_dir / "predictions.jsonl").exists():
                print(f"[train] reused run {fingerprint[:8]} has no predictions.jsonl "
                      "(write it with scripts/cv/export_gnn_predictions.py)")
            return StageState(
                name="training",
                fingerprint=fingerprint,
                stage_dir=str(run_dir),
                reused=True,
                outputs=outputs,
            )

        run_dir.mkdir(parents=True, exist_ok=True)
        Path(train_cfg["split_path"]).parent.mkdir(parents=True, exist_ok=True)

        from scripts.train_gcn_baseline import run_training

        wandb_run = config.get("_wandb_run")
        if bool(run_cfg.get("print_commands", True)):
            print(
                f"[train] in-process  architecture={train_cfg['architecture']} "
                f"loss={train_cfg['loss']} seed={train_cfg['seed']}"
            )

        run_training(train_cfg, wandb_run=wandb_run)
        self._write_stage_meta(stage_dir=run_dir, payload=payload, outputs=outputs)

        return StageState(
            name="training",
            fingerprint=fingerprint,
            stage_dir=str(run_dir),
            reused=False,
            outputs=outputs,
        )

    def gnn_training_plan(
        self,
        config: dict[str, Any],
        artifacts_root: Path,
        construction_state: StageState | None,
        dataset_state: StageState | None,
        token_state: StageState | None,
        pretraining_state: StageState | None = None,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Fingerprint, fingerprint payload and run_training() config of the GNN
        training stage, without side effects. scripts/cv/export_gnn_predictions.py
        uses it to rebuild finished runs exactly as they were trained."""
        stage_cfg = config.get("training", {})
        if construction_state is None:
            raise ValueError("GNN training requires construction stage output")
        if dataset_state is None:
            raise ValueError("GNN training requires dataset stage output")

        # save_predictions leaves the trained model unchanged and train_fraction
        # 1.0 is the default: both stay out of the fingerprint, so runs trained
        # without these options keep their fingerprint and are reused.
        fingerprint_cfg = {key: value for key, value in stage_cfg.items() if key != "save_predictions"}
        train_fraction = float(stage_cfg.get("train_fraction", 1.0))
        if train_fraction == 1.0:
            fingerprint_cfg.pop("train_fraction", None)
        payload = {
            "stage": "training",
            "used_method": "gnn",
            "config": fingerprint_cfg,
            "dataset_fingerprint": dataset_state.fingerprint,
            "token_encoder_fingerprint": token_state.fingerprint if token_state else None,
        }
        init_from_pretraining = bool(stage_cfg.get("init_from_pretraining", False))
        if init_from_pretraining:
            if pretraining_state is None:
                raise ValueError("training.init_from_pretraining needs pretraining.enabled: true")
            payload["pretraining_fingerprint"] = pretraining_state.fingerprint
        fingerprint = stable_fingerprint(jsonable(payload))
        runs_root = artifacts_root / "runs"

        split_seed = int(stage_cfg.get("seed", 42))
        split_strategy = str(stage_cfg.get("split_strategy", "random"))
        # random splits keep their historical file name
        if split_strategy == "random":
            split_name = f"split_seed_{split_seed}.json"
        elif split_strategy in {"kfold", "group_kfold"}:
            split_name = (
                f"split_{split_strategy}_{stage_cfg.get('split_group_key', 'repo_url')}"
                f"_k{stage_cfg.get('cv_folds', 5)}_f{stage_cfg.get('cv_fold', 0)}_seed_{split_seed}.json"
            )
        else:
            split_name = f"split_{split_strategy}_{stage_cfg.get('split_group_key', 'repo_url')}_seed_{split_seed}.json"
        split_path = artifacts_root / "cache" / "splits" / dataset_state.fingerprint / split_name

        # Architecture-aware batch budget
        architecture = str(stage_cfg.get("architecture", "gcn"))
        if architecture == "gat":
            max_nodes = int(stage_cfg.get("gat_max_nodes_per_batch", stage_cfg.get("max_nodes_per_batch", 12000)))
        else:
            max_nodes = int(stage_cfg.get("max_nodes_per_batch", 20000))

        focal_cfg = stage_cfg.get("focal_loss", {})
        if not isinstance(focal_cfg, dict):
            focal_cfg = {}

        train_cfg: dict[str, Any] = {
            "dataset_path": dataset_state.outputs["dataset_path"],
            "output_dir": str(runs_root),
            "run_name": fingerprint,
            "split_path": str(split_path),
            "save_split": True,
            "reuse_split": True,
            "num_labels": int(stage_cfg.get("num_labels", 4)),
            "train_ratio": float(stage_cfg.get("train_ratio", 0.8)),
            "val_ratio": float(stage_cfg.get("val_ratio", 0.1)),
            "seed": split_seed,
            "epochs": int(stage_cfg.get("epochs", 40)),
            "patience": int(stage_cfg.get("patience", 10)),
            "lr": float(stage_cfg.get("lr", 1e-3)),
            "weight_decay": float(stage_cfg.get("weight_decay", 1e-4)),
            "dropout": float(stage_cfg.get("dropout", 0.2)),
            "hidden_dim": int(stage_cfg.get("hidden_dim", 256)),
            "type_emb_dim": int(stage_cfg.get("type_emb_dim", 128)),
            "grad_accum_steps": int(stage_cfg.get("grad_accum_steps", 1)),
            "grad_clip_norm": float(stage_cfg.get("grad_clip_norm", 1.0)),
            "max_nodes_per_batch": max_nodes,
            "max_edges_per_batch": int(stage_cfg.get("max_edges_per_batch", 0)),
            "eval_max_nodes_per_batch": int(stage_cfg.get("eval_max_nodes_per_batch", 40000)),
            "eval_max_edges_per_batch": int(stage_cfg.get("eval_max_edges_per_batch", 0)),
            "num_workers": int(stage_cfg.get("num_workers", 0)),
            "pin_memory": bool(stage_cfg.get("pin_memory", False)),
            "to_undirected": bool(stage_cfg.get("to_undirected", True)),
            "threshold_steps": int(stage_cfg.get("threshold_steps", 101)),
            "device": str(stage_cfg.get("device", "auto")),
            "feature_mode": str(stage_cfg.get("feature_mode", "type_numeric")),
            "architecture": architecture,
            "num_layers": int(stage_cfg.get("num_layers", 2)),
            "num_heads": int(stage_cfg.get("gat_num_heads", 4)),
            "aggregation": str(stage_cfg.get("graphsage_aggregation", "mean")),
            "pooling": str(stage_cfg.get("pooling", "mean")),
            "set2set_steps": int(stage_cfg.get("set2set_steps", 3)),
            "edge_type_mode": str(stage_cfg.get("edge_type_mode", "ignore")),
            "split_strategy": split_strategy,
            "cv_folds": int(stage_cfg.get("cv_folds", 5)),
            "cv_fold": int(stage_cfg.get("cv_fold", 0)),
            "split_group_key": str(stage_cfg.get("split_group_key", "repo_url")),
            "split_group_json": str(to_path(self.project_root, config.get("construction", {}).get("input_json"))),
            "rgcn_num_bases": stage_cfg.get("rgcn_num_bases"),
            "pool_subset_types": list(stage_cfg.get("pool_subset_types") or []),
            "graph_features": stage_cfg.get("graph_features"),
            "node_type_vocab_path": dataset_state.outputs["node_type_vocab_path"],
            "project_root": str(self.project_root),
            "loss": str(stage_cfg.get("loss", "weighted_bce")),
            "focal_gamma": float(focal_cfg.get("gamma", 2.0)),
            "focal_alpha": float(focal_cfg.get("alpha", -1.0)),
            "early_stopping_metric": str(stage_cfg.get("early_stopping_metric", "macro_f1")),
            "early_stopping_min_delta": float(stage_cfg.get("early_stopping_min_delta", 0.0)),
            "max_graphs": stage_cfg.get("max_graphs"),
        }
        # Optional keys, passed only when set so the fingerprints of older runs stay valid.
        for key in ("labels_json", "seq_source_json", "review_shares_json"):
            if stage_cfg.get(key):
                train_cfg[key] = str(to_path(self.project_root, stage_cfg[key]))
        for key in (
            "hybrid_branch", "seq_max_tokens", "seq_min_freq", "seq_max_vocab", "seq_embed_dim",
            "seq_hidden_dim", "seq_num_layers", "seq_dropout", "seq_mask_padding", "train_target",
        ):
            if stage_cfg.get(key) is not None:
                train_cfg[key] = stage_cfg[key]

        if train_fraction != 1.0:
            train_cfg["train_fraction"] = train_fraction
        if bool(stage_cfg.get("save_predictions", False)):
            train_cfg["save_predictions"] = True

        if init_from_pretraining:
            checkpoint_name = str(stage_cfg.get("pretrained_checkpoint", "encoder_best.pt"))
            train_cfg["pretrained_encoder"] = str(Path(pretraining_state.stage_dir) / checkpoint_name)
            train_cfg["freeze_encoder"] = bool(stage_cfg.get("freeze_encoder", False))

        if token_state is not None:
            train_cfg["token_vectors_path"] = token_state.outputs["vectors_path"]
            train_cfg["manifest_path"] = construction_state.outputs["manifest_path"]
        return fingerprint, payload, train_cfg

    def _run_pretraining(
        self,
        config: dict[str, Any],
        artifacts_root: Path,
        force_stage: bool,
    ) -> StageState:
        """Self-supervised pre-training on an unlabelled Java corpus.

        Three cached steps: AST graphs of the corpus (construction from-dir),
        the PyG dataset of those graphs, then scripts/pretraining/pretrain_ssl.py (plain
        python or torchrun, see pretraining.distributed).
        """
        stage_cfg = config.get("pretraining", {})
        run_cfg = config.get("run", {})
        print_cmds = bool(run_cfg.get("print_commands", True))
        force = bool(run_cfg.get("force_rebuild", False)) or force_stage
        data_out, data_fp = self.prepare_pretraining_data(config, artifacts_root, force)
        return self._run_pretraining_model(config, artifacts_root, data_out, data_fp, force, print_cmds)

    def prepare_pretraining_data(
        self,
        config: dict[str, Any],
        artifacts_root: Path,
        force: bool = False,
    ) -> tuple[dict[str, str], str]:
        """Build (or reuse) the corpus graphs and their PyG dataset."""
        stage_cfg = config.get("pretraining", {})
        run_cfg = config.get("run", {})
        corpus_cfg = dict(stage_cfg.get("corpus", {}) or {})
        print_cmds = bool(run_cfg.get("print_commands", True))

        input_dir = to_path(self.project_root, corpus_cfg["input_dir"])
        excluded = {str(d) for d in corpus_cfg.get("exclude_projects", []) or []}
        if bool(corpus_cfg.get("exclude_mlcq_projects", True)):
            # skip corpus projects that also appear in the labelled dataset
            labelled = json.loads(
                to_path(self.project_root, config.get("construction", {}).get("input_json")).read_text()
            )
            repo_names = {
                str(entry.get("repo_url", "")).rstrip("/").split("/")[-1].lower().removesuffix(".git")
                for entry in labelled
            }
            top_dirs = {path.name for path in input_dir.iterdir() if path.is_dir()}
            excluded |= {d for d in top_dirs if d.lower() in repo_names}
        excluded_list = sorted(excluded)

        runtime_keys = {"workers", "chunksize", "progress_every"}
        corpus_fp_cfg = {k: v for k, v in corpus_cfg.items() if k not in runtime_keys}
        corpus_fp = stable_fingerprint(
            jsonable({"stage": "pretrain_corpus", "config": corpus_fp_cfg, "excluded": excluded_list})
        )
        corpus_dir = artifacts_root / "cache" / "pretrain_corpus" / corpus_fp
        corpus_out = {"dot_dir": str(corpus_dir / "dot"), "manifest_path": str(corpus_dir / "manifest.jsonl")}
        if not self._can_reuse(corpus_dir, corpus_out, force=force):
            corpus_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(self.project_root / "scripts" / "build_ast_dot_from_normalized.py"),
                "from-dir",
                "--input-dir", str(input_dir),
                "--output-dir", corpus_out["dot_dir"],
                "--manifest-path", corpus_out["manifest_path"],
                "--workers", str(corpus_cfg.get("workers", 8)),
                "--chunksize", str(corpus_cfg.get("chunksize", 8)),
                "--progress-every", str(corpus_cfg.get("progress_every", 500)),
                "--exclude-top-dirs", ",".join(excluded_list),
            ]
            if corpus_cfg.get("limit") is not None:
                cmd += ["--limit", str(corpus_cfg["limit"])]
            if corpus_cfg.get("max_file_bytes") is not None:
                cmd += ["--max-file-bytes", str(corpus_cfg["max_file_bytes"])]
            if corpus_cfg.get("sample_seed") is not None:
                cmd += ["--sample-seed", str(corpus_cfg["sample_seed"])]
            if corpus_cfg.get("parser") is not None:
                cmd += ["--parser", str(corpus_cfg["parser"])]
            if bool(corpus_cfg.get("allow_parse_errors", False)):
                cmd.append("--allow-parse-errors")
            self._run_command(cmd, print_commands=print_cmds)
            self._write_stage_meta(
                stage_dir=corpus_dir,
                payload={"stage": "pretrain_corpus", "config": corpus_cfg, "excluded_projects": excluded_list},
                outputs=corpus_out,
            )

        edge_types = [str(e) for e in stage_cfg.get("edge_types", ["Child", "NextToken"])]
        data_fp = stable_fingerprint(
            jsonable({"stage": "pretrain_dataset", "corpus": corpus_fp, "edge_types": edge_types})
        )
        data_dir = artifacts_root / "cache" / "pretrain_dataset" / data_fp
        data_out = {
            "dataset_path": str(data_dir / "dataset.pt"),
            "node_type_vocab_path": str(data_dir / "node_type_vocab.json"),
            "metadata_path": str(data_dir / "dataset.meta.json"),
        }
        if not self._can_reuse(data_dir, data_out, force=force):
            data_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(self.project_root / "scripts" / "build_pyg_dataset_from_dot.py"),
                "--manifest-path", corpus_out["manifest_path"],
                "--dot-dir", corpus_out["dot_dir"],
                "--dataset-out", data_out["dataset_path"],
                "--node-type-vocab-out", data_out["node_type_vocab_path"],
                "--metadata-out", data_out["metadata_path"],
                "--edge-types", ",".join(edge_types),
                "--progress-style", "print",
                "--progress-every", "2000",
                "--no-memory-stats",
                "--allow-missing-labels",
                "--store-edge-types",
                "--store-source-path",
            ]
            self._run_command(cmd, print_commands=print_cmds)
            self._write_stage_meta(
                stage_dir=data_dir,
                payload={"stage": "pretrain_dataset", "corpus_fingerprint": corpus_fp, "edge_types": edge_types},
                outputs=data_out,
            )
        return data_out, data_fp

    def pretraining_run_config(
        self,
        config: dict[str, Any],
        data_out: dict[str, str],
        output_dir: Path,
    ) -> dict[str, Any]:
        """The JSON consumed by scripts/pretraining/pretrain_ssl.py."""
        stage_cfg = config.get("pretraining", {})
        metadata = json.loads(Path(data_out["metadata_path"]).read_text())
        return {
            **self._pretraining_model_cfg(stage_cfg),
            "device": stage_cfg.get("device", "auto"),
            "dataset_path": data_out["dataset_path"],
            "node_type_vocab_path": data_out["node_type_vocab_path"],
            "edge_type_order": metadata["edge_type_order"],
            "output_dir": str(output_dir),
        }

    @staticmethod
    def _pretraining_model_cfg(stage_cfg: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v for k, v in stage_cfg.items()
            if k not in {"enabled", "corpus", "edge_types", "distributed", "device"}
        }

    def _run_pretraining_model(
        self,
        config: dict[str, Any],
        artifacts_root: Path,
        data_out: dict[str, str],
        data_fp: str,
        force: bool,
        print_cmds: bool,
    ) -> StageState:
        stage_cfg = config.get("pretraining", {})
        dist_cfg = dict(stage_cfg.get("distributed", {}) or {})
        model_cfg = self._pretraining_model_cfg(stage_cfg)
        fingerprint = stable_fingerprint(
            jsonable({"stage": "pretraining", "dataset": data_fp, "config": model_cfg})
        )
        stage_dir = artifacts_root / "cache" / "pretraining" / fingerprint
        outputs = {
            "encoder_path": str(stage_dir / "encoder_best.pt"),
            "metrics_path": str(stage_dir / "pretrain_metrics.json"),
        }
        if self._can_reuse(stage_dir, outputs, force=force):
            return StageState(
                name="pretraining", fingerprint=fingerprint, stage_dir=str(stage_dir), reused=True, outputs=outputs
            )

        stage_dir.mkdir(parents=True, exist_ok=True)
        run_json = self.pretraining_run_config(config, data_out, stage_dir)
        run_json_path = stage_dir / "run_config.json"
        run_json_path.write_text(json.dumps(jsonable(run_json), indent=2))

        script = str(self.project_root / "scripts" / "pretraining" / "pretrain_ssl.py")
        nproc = int(dist_cfg.get("nproc_per_node", 1))
        if nproc > 1:
            cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", script]
        else:
            cmd = [sys.executable, script]
        cmd += ["--config-json", str(run_json_path)]
        self._run_command(cmd, print_commands=print_cmds)
        self._write_stage_meta(
            stage_dir=stage_dir,
            payload={"stage": "pretraining", "config": model_cfg, "dataset_fingerprint": data_fp, "distributed": dist_cfg},
            outputs=outputs,
        )
        return StageState(
            name="pretraining", fingerprint=fingerprint, stage_dir=str(stage_dir), reused=False, outputs=outputs
        )

    def _run_training_classical(
        self,
        config: dict[str, Any],
        artifacts_root: Path,
        metric_extraction_state: StageState | None,
        force_stage: bool,
    ) -> StageState:
        """Dispatch for training.used_method=classical."""
        if metric_extraction_state is None:
            raise ValueError("Classical training requires metric_extraction stage output")

        stage_cfg = config.get("training", {})
        run_cfg = config.get("run", {})

        payload = {
            "stage": "training",
            "used_method": "classical",
            "config": stage_cfg,
            "metric_extraction_fingerprint": metric_extraction_state.fingerprint,
        }
        fingerprint = stable_fingerprint(jsonable(payload))
        runs_root = artifacts_root / "runs"
        run_dir = runs_root / fingerprint
        model_name = str(stage_cfg.get("model", "random_forest"))
        outputs = {
            "run_dir": str(run_dir),
            "results_path": str(run_dir / f"{model_name}_results.json"),
        }

        if self._can_reuse(
            run_dir,
            outputs,
            force=bool(run_cfg.get("force_rebuild", False)) or force_stage,
        ):
            return StageState(
                name="training",
                fingerprint=fingerprint,
                stage_dir=str(run_dir),
                reused=True,
                outputs=outputs,
            )

        run_dir.mkdir(parents=True, exist_ok=True)
        from scripts.train_classical import run_classical

        if bool(run_cfg.get("print_commands", True)):
            seeds = stage_cfg.get("seeds", [])
            print(f"[train] classical in-process  model={model_name} seeds={len(seeds)}")

        run_classical(
            training_cfg=stage_cfg,
            metrics_csv=metric_extraction_state.outputs["metrics_csv_path"],
            output_dir=str(run_dir),
        )

        self._write_stage_meta(stage_dir=run_dir, payload=payload, outputs=outputs)
        return StageState(
            name="training",
            fingerprint=fingerprint,
            stage_dir=str(run_dir),
            reused=False,
            outputs=outputs,
        )

    def _run_training_sequence(
        self,
        config: dict[str, Any],
        artifacts_root: Path,
        token_dataset_state: StageState | None,
        force_stage: bool,
    ) -> StageState:
        """Dispatch for training.used_method=sequence."""
        if token_dataset_state is None:
            raise ValueError("Sequence training requires token_dataset stage output")

        stage_cfg = config.get("training", {})
        run_cfg = config.get("run", {})

        payload = {
            "stage": "training",
            "used_method": "sequence",
            "config": stage_cfg,
            "token_dataset_fingerprint": token_dataset_state.fingerprint,
        }
        fingerprint = stable_fingerprint(jsonable(payload))
        runs_root = artifacts_root / "runs"
        run_dir = runs_root / fingerprint
        model_name = str(stage_cfg.get("model", "bilstm_attention"))
        outputs = {
            "run_dir": str(run_dir),
            "results_path": str(run_dir / f"{model_name}_results.json"),
        }

        if self._can_reuse(
            run_dir,
            outputs,
            force=bool(run_cfg.get("force_rebuild", False)) or force_stage,
        ):
            return StageState(
                name="training",
                fingerprint=fingerprint,
                stage_dir=str(run_dir),
                reused=True,
                outputs=outputs,
            )

        run_dir.mkdir(parents=True, exist_ok=True)
        from scripts.train_sequence import run_sequence

        if bool(run_cfg.get("print_commands", True)):
            seeds = stage_cfg.get("seeds", [])
            print(f"[train] sequence in-process  model={model_name} seeds={len(seeds)}")

        run_sequence(
            training_cfg=stage_cfg,
            token_dir=token_dataset_state.outputs["token_dir"],
            output_dir=str(run_dir),
        )

        self._write_stage_meta(stage_dir=run_dir, payload=payload, outputs=outputs)
        return StageState(
            name="training",
            fingerprint=fingerprint,
            stage_dir=str(run_dir),
            reused=False,
            outputs=outputs,
        )

    def _run_training_chunked_sequence(
        self,
        config: dict[str, Any],
        artifacts_root: Path,
        folds_state: StageState | None,
        force_stage: bool,
    ) -> StageState:
        """Dispatch for training.used_method=chunked_sequence: one model, one (seed, fold) of the folds."""
        stage_cfg = config.get("training", {})
        run_cfg = config.get("run", {})

        if stage_cfg.get("folds_json"):
            folds_json = to_path(self.project_root, stage_cfg["folds_json"])
            folds_id = hashlib.md5(folds_json.read_bytes()).hexdigest()
        else:
            folds_json = Path(folds_state.outputs["folds_json"])
            folds_id = folds_state.fingerprint
        folds_meta = json.loads(folds_json.read_text())["meta"]
        input_json = to_path(self.project_root, stage_cfg.get("input_json") or folds_meta["input_json"])

        payload = {
            "stage": "training",
            "used_method": "chunked_sequence",
            "config": stage_cfg,
            "folds": folds_id,
            "input_json_md5": hashlib.md5(input_json.read_bytes()).hexdigest(),
        }
        fingerprint = stable_fingerprint(jsonable(payload))
        run_dir = artifacts_root / "runs" / fingerprint
        outputs = {
            "run_dir": str(run_dir),
            "metrics_path": str(run_dir / "metrics.json"),
            "record_path": str(run_dir / "record.json"),
            "predictions_path": str(run_dir / "predictions.jsonl"),
        }

        if self._can_reuse(
            run_dir,
            outputs,
            force=bool(run_cfg.get("force_rebuild", False)) or force_stage,
        ):
            return StageState(
                name="training",
                fingerprint=fingerprint,
                stage_dir=str(run_dir),
                reused=True,
                outputs=outputs,
            )

        run_dir.mkdir(parents=True, exist_ok=True)
        from scripts.train_chunked_sequence import run_chunked_sequence

        print(f"[train] chunked sequence in-process  model={stage_cfg.get('model')} "
              f"seed={stage_cfg.get('seed')} fold={stage_cfg.get('cv_fold')}")
        run_chunked_sequence(
            training_cfg=stage_cfg,
            input_json=input_json,
            folds_json=folds_json,
            output_dir=run_dir,
        )

        self._write_stage_meta(stage_dir=run_dir, payload=payload, outputs=outputs)
        return StageState(
            name="training",
            fingerprint=fingerprint,
            stage_dir=str(run_dir),
            reused=False,
            outputs=outputs,
        )

    def _run_training_codebert(
        self,
        config: dict[str, Any],
        artifacts_root: Path,
        normalization_state: StageState | None,
        force_stage: bool,
    ) -> StageState:
        """Dispatch for training.used_method=codebert."""
        if normalization_state is None:
            raise ValueError("CodeBERT training requires normalization stage output")

        stage_cfg = config.get("training", {})
        run_cfg = config.get("run", {})

        payload = {
            "stage": "training",
            "used_method": "codebert",
            "config": stage_cfg,
            "normalization_fingerprint": normalization_state.fingerprint,
        }
        fingerprint = stable_fingerprint(jsonable(payload))
        runs_root = artifacts_root / "runs"
        run_dir = runs_root / fingerprint
        outputs = {
            "run_dir": str(run_dir),
            "results_path": str(run_dir / "codebert_results.json"),
        }

        if self._can_reuse(
            run_dir,
            outputs,
            force=bool(run_cfg.get("force_rebuild", False)) or force_stage,
        ):
            return StageState(
                name="training",
                fingerprint=fingerprint,
                stage_dir=str(run_dir),
                reused=True,
                outputs=outputs,
            )

        run_dir.mkdir(parents=True, exist_ok=True)
        from scripts.train_codebert import run_codebert

        seeds = stage_cfg.get("seeds", [])
        print(f"[train] codebert in-process  seeds={len(seeds)}")

        normalized_json = normalization_state.outputs["normalized_json_path"]
        run_codebert(
            training_cfg=stage_cfg,
            normalized_json_path=normalized_json,
            output_dir=str(run_dir),
        )

        self._write_stage_meta(stage_dir=run_dir, payload=payload, outputs=outputs)
        return StageState(
            name="training",
            fingerprint=fingerprint,
            stage_dir=str(run_dir),
            reused=False,
            outputs=outputs,
        )

    def _run_reporting(self, config: dict[str, Any], training_state: StageState) -> StageState:
        report_cfg = config.get("reporting", {})
        run_dir = Path(training_state.outputs["run_dir"])
        fig_dir = run_dir / "figures"

        payload = {
            "stage": "reporting",
            "config": report_cfg,
            "training_fingerprint": training_state.fingerprint,
        }
        fingerprint = stable_fingerprint(jsonable(payload))
        metrics_path = Path(training_state.outputs["metrics_path"])
        output_map = generate_training_figures(
            metrics_path=metrics_path,
            output_dir=fig_dir,
        )

        curves_path = run_dir / "curves.json"
        if curves_path.exists():
            try:
                pub_map = generate_publication_figures(
                    metrics_path=metrics_path,
                    curves_path=curves_path,
                    output_dir=fig_dir / "publication",
                    label_names=LABEL_ORDER,
                )
                output_map.update(pub_map)
            except Exception as exc:
                print(f"[pipeline] publication figures failed: {exc!r}")

        outputs = {
            "figure_dir": str(fig_dir),
            "figure_index_path": str(run_dir / "figures.json"),
            **{key: str(value) for key, value in output_map.items()},
        }
        Path(outputs["figure_index_path"]).write_text(json.dumps(output_map, indent=2))

        return StageState(
            name="reporting",
            fingerprint=fingerprint,
            stage_dir=str(fig_dir),
            reused=False,
            outputs=outputs,
        )

    def _log_wandb(
        self,
        config: dict[str, Any],
        training_state: StageState,
        report_state: StageState | None,
        group: str | None = None,
        wandb_run: Any | None = None,
    ) -> None:
        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError:
            print("[pipeline] wandb is not installed; skipping W&B logging")
            return

        run_cfg = config.get("run", {})
        wandb_cfg = run_cfg.get("wandb", {})

        # Use an existing run (from in-process training) or create a new one
        run = wandb_run
        created_run = False
        if run is None:
            run = wandb.init(
                project=str(wandb_cfg.get("project", "mlcq_graphs")),
                entity=wandb_cfg.get("entity"),
                name=str(run_cfg.get("name", training_state.fingerprint[:12])),
                config=jsonable(config),
                mode="online",
                tags=wandb_cfg.get("tags", []),
                group=group,
            )
            created_run = True

        metrics = json.loads(Path(training_state.outputs["metrics_path"]).read_text())
        test_fixed = metrics.get("test_fixed_0_5", {})
        test_tuned = metrics.get("test_tuned", {})

        # Log per-epoch history as backup (in-process logging is primary)
        history = metrics.get("history", [])
        if history and created_run:
            for entry in history:
                epoch = int(entry.get("epoch", 0))
                run.log(
                    {
                        "epoch": epoch,
                        "train/loss": float(entry.get("train_loss", 0.0)),
                        "val/f1_micro": float(entry.get("val_f1_micro", 0.0)),
                        "val/f1_macro": float(entry.get("val_f1_macro", 0.0)),
                        "val/pr_auc_macro": float(entry.get("val_pr_auc_macro", 0.0)),
                    },
                    step=epoch,
                )

        # Test metrics: aggregate (both fixed and tuned thresholds)
        test_summary: dict[str, float | int] = {}
        for suffix, src in [("fixed", test_fixed), ("tuned", test_tuned)]:
            for key in ("f1_micro", "f1_macro", "pr_auc_macro", "balanced_acc_macro",
                        "mcc_macro", "roc_auc_macro", "hamming_loss", "subset_accuracy"):
                val = src.get(key)
                if val is not None:
                    test_summary[f"test/{key}_{suffix}"] = float(val)

        # Per-label test metrics (tuned thresholds)
        per_label = test_tuned.get("per_label", {})
        for label, label_metrics in per_label.items():
            if isinstance(label_metrics, dict):
                for metric_name in ("f1", "precision", "recall", "pr_auc",
                                    "balanced_acc", "mcc", "roc_auc",
                                    "tp", "fp", "tn", "fn"):
                    val = label_metrics.get(metric_name)
                    if val is not None:
                        test_summary[f"test_per_label/{label}/{metric_name}"] = float(val)

        # Training metadata
        best_epoch = metrics.get("best_epoch")
        if best_epoch is not None:
            test_summary["train/best_epoch"] = int(best_epoch)

        thresholds = metrics.get("thresholds", {})
        for label, thr in thresholds.items():
            test_summary[f"thresholds/{label}"] = float(thr)

        # Profiling
        profiling = metrics.get("profiling", {})
        if profiling:
            test_summary["profiling/train_duration_sec"] = float(profiling.get("train_duration_sec", 0.0))
            test_summary["profiling/peak_gpu_memory_mb"] = float(profiling.get("peak_gpu_memory_mb", 0.0))
            test_summary["profiling/train_throughput"] = float(profiling.get("train_throughput_graphs_per_sec", 0.0))

        run.summary.update(test_summary)

        if report_state is not None:
            for key, value in report_state.outputs.items():
                if key.endswith("_path"):
                    continue
                if key in {"figure_dir", "figure_index_path"}:
                    continue
                fig_path = Path(value)
                if fig_path.suffix.lower() in {".png", ".jpg", ".jpeg"} and fig_path.exists():
                    run.log({f"figures/{key}": wandb.Image(str(fig_path))})

        if created_run:
            run.finish()

    def _run_command(self, cmd: list[str], print_commands: bool) -> None:
        if print_commands:
            script_name = Path(cmd[1]).name if len(cmd) > 1 else "?"
            params = {}
            for flag in ("--architecture", "--loss", "--seed", "--epochs"):
                try:
                    idx = cmd.index(flag)
                    params[flag.lstrip("-")] = cmd[idx + 1]
                except (ValueError, IndexError):
                    pass
            if params:
                summary = " ".join(f"{k}={v}" for k, v in params.items())
                print(f"[train] {script_name}  {summary}")
            else:
                print(f"[train] {script_name}")
        subprocess.run(cmd, check=True, cwd=str(self.project_root))

    def _can_reuse(self, stage_dir: Path, outputs: dict[str, str], force: bool) -> bool:
        if force:
            return False
        meta_path = stage_dir / "stage_meta.json"
        if not meta_path.exists():
            return False
        return all(Path(path).exists() for path in outputs.values())

    def _write_stage_meta(
        self,
        stage_dir: Path,
        payload: dict[str, Any],
        outputs: dict[str, str],
    ) -> None:
        stage_dir.mkdir(parents=True, exist_ok=True)
        meta_path = stage_dir / "stage_meta.json"
        meta = {
            "created_at": datetime.now().isoformat(),
            "payload": jsonable(payload),
            "outputs": jsonable(outputs),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
