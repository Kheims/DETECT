from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import shlex
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
        "construction",
        "dataset",
        "token_encoder",
        "training",
        "reporting",
    )

    STAGE_RUNTIME_ONLY_KEYS: dict[str, set[str]] = {
        "construction": {"workers", "chunksize", "progress_every", "overwrite"},
        "dataset": {"progress_style", "progress_every", "memory_stats"},
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
        print(f"[pipeline] stage={stage_name} start")
        started = time.perf_counter()
        stage_state = runner()
        elapsed = time.perf_counter() - started
        mode = "reused" if stage_state.reused else "rebuilt"
        output_hint = self._stage_output_hint(stage_state)
        print(
            f"[pipeline] stage={stage_name} done mode={mode} elapsed_sec={elapsed:.2f} "
            f"fingerprint={stage_state.fingerprint[:12]} output={output_hint}"
        )
        return stage_state

    def _normalize_retry_from(self, value: Any) -> str | None:
        if value is None:
            return None

        retry_from = str(value).strip().lower()
        if retry_from in {"", "none", "null"}:
            return None

        aliases = {
            "construction": "construction",
            "dataset": "dataset",
            "token_encoder": "token_encoder",
            "token-encoder": "token_encoder",
            "tokenencoder": "token_encoder",
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

        training_state = None
        if bool(config.get("training", {}).get("enabled", True)):
            if dataset_state is None:
                raise ValueError("Training stage requires dataset stage output")
            if construction_state is None:
                raise ValueError("Training stage requires construction stage output")
            training_state = self._run_stage_with_logs(
                "training",
                lambda: self._run_training(
                    config=config,
                    artifacts_root=artifacts_root,
                    construction_state=construction_state,
                    dataset_state=dataset_state,
                    token_state=token_state,
                    force_stage=("training" in forced_stages),
                ),
            )
            stages["training"] = training_state

        report_state = None
        if bool(config.get("reporting", {}).get("enabled", True)) and training_state is not None:
            report_state = self._run_stage_with_logs(
                "reporting",
                lambda: self._run_reporting(config, training_state),
            )
            stages["reporting"] = report_state

        if bool(run_cfg.get("wandb", {}).get("enabled", False)) and training_state is not None:
            self._log_wandb(config=config, training_state=training_state, report_state=report_state)

        elapsed_total = time.perf_counter() - pipeline_started
        print(f"[pipeline] completed elapsed_sec={elapsed_total:.2f}")

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

        seeds = ablation_cfg.get("seeds", [base_cfg.get("training", {}).get("seed", 42)])
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("ablation.seeds must be a non-empty list")

        run_cfg = base_cfg.get("run", {})
        artifacts_root = to_path(self.project_root, run_cfg.get("artifacts_root", "artifacts"))
        base_name = str(run_cfg.get("name", "ablation"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = artifacts_root / "reports" / f"{base_name}_{timestamp}"
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
                trial_cfg = copy.deepcopy(base_cfg)
                trial_cfg.setdefault("ablation", {})["enabled"] = False
                trial_cfg.setdefault("training", {})["seed"] = int(seed)
                for dotted_key, value in combo.items():
                    set_nested_value(trial_cfg, dotted_key, value)
                trial_cfg.setdefault("run", {})["name"] = (
                    f"{base_name}_c{combo_idx:03d}_s{seed}"
                )

                trial_num += 1
                arch = combo.get("training.architecture", "?")
                loss = combo.get("training.loss", "?")
                print(f"[ablation] Trial {trial_num}/{total_trials}: {arch}+{loss} seed={seed}")

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
                        # extended fields from Plan 01's enriched metrics.json
                        "hamming_loss_tuned": float(metrics.get("test_tuned", {}).get("hamming_loss", 0.0)),
                        "subset_acc_tuned": float(metrics.get("test_tuned", {}).get("subset_accuracy", 0.0)),
                        "pr_auc_fixed": float(metrics.get("test_fixed_0_5", {}).get("pr_auc_macro", 0.0)),
                        "hamming_loss_fixed": float(metrics.get("test_fixed_0_5", {}).get("hamming_loss", 0.0)),
                        "subset_acc_fixed": float(metrics.get("test_fixed_0_5", {}).get("subset_accuracy", 0.0)),
                        "per_label_tuned": metrics.get("test_tuned", {}).get("per_label", {}),
                        # profiling
                        "train_duration_sec": float(metrics.get("profiling", {}).get("train_duration_sec", 0.0)),
                        "peak_gpu_memory_mb": float(metrics.get("profiling", {}).get("peak_gpu_memory_mb", 0.0)),
                        "train_throughput": float(metrics.get("profiling", {}).get("train_throughput_graphs_per_sec", 0.0)),
                    }
                    records.append(record)

                    if bool(run_cfg.get("wandb", {}).get("enabled", False)):
                        trial_wandb_cfg = copy.deepcopy(trial_cfg)
                        trial_tags = [str(arch), str(loss), f"seed={seed}", "ablation"]
                        trial_wandb_cfg.setdefault("run", {}).setdefault("wandb", {})["tags"] = trial_tags
                        training_state = StageState(
                            name="training",
                            fingerprint=stage_info["fingerprint"],
                            stage_dir=stage_info["stage_dir"],
                            reused=stage_info["reused"],
                            outputs=stage_info["outputs"],
                        )
                        try:
                            self._log_wandb(config=trial_wandb_cfg, training_state=training_state, report_state=None)
                        except Exception as wandb_exc:
                            print(f"[ablation] W&B logging failed for trial {trial_num}: {wandb_exc!r}")

                except Exception as exc:
                    print(f"[ablation] trial FAILED: {exc!r} -- continuing")
                    failed_records.append({"seed": int(seed), "combo": combo, "error": str(exc)})

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
                    # per-label means
                    "per_label_means": per_label_means,
                    # profiling means
                    "avg_train_duration_sec": statistics.mean(train_dur_vals),
                    "avg_peak_gpu_memory_mb": statistics.mean(gpu_mem_vals),
                    "avg_train_throughput": statistics.mean(throughput_vals),
                }
            )

        # significance tests vs GCN+weighted_bce baseline
        combo_scores: dict[str, list[float]] = {
            key: [row["f1_macro_tuned"] for row in items]
            for key, items in grouped.items()
        }
        baseline_key: str | None = None
        for key in grouped:
            combo_dict = json.loads(key)
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
            significance_results["note"] = "no gcn+weighted_bce baseline found; baseline_tests skipped"

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

        # Console summary table
        print(f"\n{'='*60}")
        print(f" Ablation Summary (macro-F1 tuned) | {len(records)} trials")
        if failed_records:
            print(f" ({len(failed_records)} trial(s) failed)")
        print(f"{'='*60}")
        print(f"{'Arch':<12} {'Loss':<14} {'Mean':>7} {'Std':>7}  {'vs baseline'}")
        print(f"{'-'*60}")

        baseline_tests = significance_results.get("baseline_tests", {})

        for row in summary_extended_rows:
            combo = row["combo"]
            r_arch = str(combo.get("training.architecture", combo.get("architecture", "?")))
            r_loss = str(combo.get("training.loss", combo.get("loss", "?")))
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

            print(f"{r_arch:<12} {r_loss:<14} {mean_f1:>7.4f} {std_f1:>7.4f}  {marker}")

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

        mode = str(stage_cfg.get("mode", "from-json"))
        cmd = [sys.executable, str(script_path), mode]

        if mode == "from-json":
            cmd += [
                "--input-json",
                str(to_path(self.project_root, stage_cfg.get("input_json"))),
                "--output-dir",
                outputs["dot_dir"],
                "--manifest-path",
                outputs["manifest_path"],
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
                outputs["dot_dir"],
                "--manifest-path",
                outputs["manifest_path"],
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

        self._run_command(cmd, print_commands=bool(run_cfg.get("print_commands", True)))
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

    def _run_training(
        self,
        config: dict[str, Any],
        artifacts_root: Path,
        construction_state: StageState,
        dataset_state: StageState,
        token_state: StageState | None,
        force_stage: bool,
    ) -> StageState:
        stage_cfg = config.get("training", {})
        run_cfg = config.get("run", {})

        payload = {
            "stage": "training",
            "config": stage_cfg,
            "dataset_fingerprint": dataset_state.fingerprint,
            "token_encoder_fingerprint": token_state.fingerprint if token_state else None,
        }
        fingerprint = stable_fingerprint(jsonable(payload))
        runs_root = artifacts_root / "runs"
        run_dir = runs_root / fingerprint

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
            return StageState(
                name="training",
                fingerprint=fingerprint,
                stage_dir=str(run_dir),
                reused=True,
                outputs=outputs,
            )

        run_dir.mkdir(parents=True, exist_ok=True)
        script_path = self.project_root / "scripts" / "train_gcn_baseline.py"
        split_seed = int(stage_cfg.get("seed", 42))
        split_path = (
            artifacts_root
            / "cache"
            / "splits"
            / dataset_state.fingerprint
            / f"split_seed_{split_seed}.json"
        )
        split_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(script_path),
            "--dataset-path",
            dataset_state.outputs["dataset_path"],
            "--output-dir",
            str(runs_root),
            "--run-name",
            fingerprint,
            "--split-path",
            str(split_path),
            "--save-split",
            "--reuse-split",
            "--num-labels",
            str(stage_cfg.get("num_labels", 4)),
            "--train-ratio",
            str(stage_cfg.get("train_ratio", 0.8)),
            "--val-ratio",
            str(stage_cfg.get("val_ratio", 0.1)),
            "--seed",
            str(split_seed),
            "--epochs",
            str(stage_cfg.get("epochs", 40)),
            "--patience",
            str(stage_cfg.get("patience", 10)),
            "--lr",
            str(stage_cfg.get("lr", 1e-3)),
            "--weight-decay",
            str(stage_cfg.get("weight_decay", 1e-4)),
            "--dropout",
            str(stage_cfg.get("dropout", 0.2)),
            "--hidden-dim",
            str(stage_cfg.get("hidden_dim", 256)),
            "--type-emb-dim",
            str(stage_cfg.get("type_emb_dim", 128)),
            "--grad-accum-steps",
            str(stage_cfg.get("grad_accum_steps", 1)),
            "--grad-clip-norm",
            str(stage_cfg.get("grad_clip_norm", 1.0)),
        ]

        # Architecture-aware batch budget
        architecture = str(stage_cfg.get("architecture", "gcn"))
        if architecture == "gat":
            max_nodes = int(stage_cfg.get("gat_max_nodes_per_batch", stage_cfg.get("max_nodes_per_batch", 12000)))
        else:
            max_nodes = int(stage_cfg.get("max_nodes_per_batch", 20000))

        cmd += [
            "--max-nodes-per-batch",
            str(max_nodes),
            "--max-edges-per-batch",
            str(stage_cfg.get("max_edges_per_batch", 0)),
            "--eval-max-nodes-per-batch",
            str(stage_cfg.get("eval_max_nodes_per_batch", 40000)),
            "--eval-max-edges-per-batch",
            str(stage_cfg.get("eval_max_edges_per_batch", 0)),
            "--num-workers",
            str(stage_cfg.get("num_workers", 0)),
            "--threshold-steps",
            str(stage_cfg.get("threshold_steps", 101)),
            "--device",
            str(stage_cfg.get("device", "auto")),
            "--feature-mode",
            str(stage_cfg.get("feature_mode", "type_numeric")),
            "--architecture",
            str(stage_cfg.get("architecture", "gcn")),
            "--num-layers",
            str(stage_cfg.get("num_layers", 2)),
            "--num-heads",
            str(stage_cfg.get("gat_num_heads", 4)),
            "--aggregation",
            str(stage_cfg.get("graphsage_aggregation", "mean")),
            "--loss",
            str(stage_cfg.get("loss", "weighted_bce")),
            "--early-stopping-metric",
            str(stage_cfg.get("early_stopping_metric", "macro_f1")),
            "--early-stopping-min-delta",
            str(stage_cfg.get("early_stopping_min_delta", 0.0)),
        ]

        focal_cfg = stage_cfg.get("focal_loss", {})
        if isinstance(focal_cfg, dict):
            cmd += [
                "--focal-gamma",
                str(focal_cfg.get("gamma", 2.0)),
                "--focal-alpha",
                str(focal_cfg.get("alpha", -1.0)),
            ]

        if stage_cfg.get("max_graphs") is not None:
            cmd += ["--max-graphs", str(stage_cfg.get("max_graphs"))]
        if bool(stage_cfg.get("pin_memory", False)):
            cmd.append("--pin-memory")
        if bool(stage_cfg.get("to_undirected", True)):
            cmd.append("--to-undirected")
        else:
            cmd.append("--no-to-undirected")

        if token_state is not None:
            cmd += [
                "--token-vectors-path",
                token_state.outputs["vectors_path"],
                "--manifest-path",
                construction_state.outputs["manifest_path"],
            ]

        self._run_command(cmd, print_commands=bool(run_cfg.get("print_commands", True)))
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
        output_map = generate_training_figures(
            metrics_path=Path(training_state.outputs["metrics_path"]),
            output_dir=fig_dir,
        )

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
    ) -> None:
        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError:
            print("[pipeline] wandb is not installed; skipping W&B logging")
            return

        run_cfg = config.get("run", {})
        wandb_cfg = run_cfg.get("wandb", {})

        run = wandb.init(
            project=str(wandb_cfg.get("project", "mlcq_graphs")),
            entity=wandb_cfg.get("entity"),
            name=str(run_cfg.get("name", training_state.fingerprint[:12])),
            config=jsonable(config),
            mode="online",
            tags=wandb_cfg.get("tags", []),
        )

        metrics = json.loads(Path(training_state.outputs["metrics_path"]).read_text())
        test_fixed = metrics.get("test_fixed_0_5", {})
        test_tuned = metrics.get("test_tuned", {})

        wandb.log(
            {
                "test/f1_micro_fixed": float(test_fixed.get("f1_micro", 0.0)),
                "test/f1_macro_fixed": float(test_fixed.get("f1_macro", 0.0)),
                "test/f1_micro_tuned": float(test_tuned.get("f1_micro", 0.0)),
                "test/f1_macro_tuned": float(test_tuned.get("f1_macro", 0.0)),
                "test/pr_auc_tuned": float(test_tuned.get("pr_auc_macro", 0.0)),
            }
        )

        if report_state is not None:
            for key, value in report_state.outputs.items():
                if key.endswith("_path"):
                    continue
                if key in {"figure_dir", "figure_index_path"}:
                    continue
                fig_path = Path(value)
                if fig_path.suffix.lower() in {".png", ".jpg", ".jpeg"} and fig_path.exists():
                    wandb.log({f"figures/{key}": wandb.Image(str(fig_path))})

        run.finish()

    def _run_command(self, cmd: list[str], print_commands: bool) -> None:
        if print_commands:
            print(f"$ {shlex.join(cmd)}")
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
