"""Trainer class with full training loop, early stopping and checkpoint saving.

Decouples the training loop from the monolithic training script into a
reusable class that works with any architecture and loss combination. The
training script becomes a thin configuration and setup layer that calls
trainer.fit().

Key design choices:
- Trainer owns: epoch loop, batch iteration, loss computation, optimizer step,
  gradient accumulation, gradient clipping, validation, early stopping,
  checkpointing and device management.
- The fit() method takes a train_loader_fn callable (epoch -> DataLoader) so
  the Trainer does not need to know about DynamicBudgetBatchSampler internals.
- evaluate_fn is injected to avoid circular imports between this package and
  the training script's evaluation code.
- Best model weights are always restored before fit() returns, even when early
  stopping was not triggered (last epoch is not guaranteed to be best epoch).

Checkpoint notes (pitfall 4 from research):
- Weights saved to disk only when validation metric improves.
- CPU copies kept in memory for fast restore at end of training.
- Checkpoint filename encodes architecture, loss and seed for self-documenting
  run directories (Pattern 3 from RESEARCH.md).
- A second copy is saved as best_model.pt for pipeline compatibility (the
  pipeline expects run_dir/best_model.pt in its outputs dict).
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader


# Maps user-facing config keys (training.early_stopping_metric) to the actual
# dict keys returned by the evaluate() function (pitfall 3 from research).
METRIC_KEY_MAP: dict[str, str] = {
    "macro_f1": "f1_macro",
    "pr_auc": "pr_auc_macro",
}


@dataclass
class TrainResult:
    """Structured result returned by Trainer.fit().

    Attributes:
        best_epoch: Epoch number (1-indexed) that achieved the best validation
            metric.
        best_val_metric: Value of the monitored validation metric at best_epoch.
        checkpoint_path: Absolute path to the saved best-model checkpoint file.
        history: Per-epoch training and validation metrics. Each entry contains
            epoch, train_loss, val_f1_micro, val_f1_macro and val_pr_auc_macro.
        stopped_early: True if early stopping triggered before reaching the
            configured maximum number of epochs.
    """

    best_epoch: int
    best_val_metric: float
    checkpoint_path: str
    history: list[dict[str, float]]
    stopped_early: bool


def _get_git_sha() -> tuple[str, bool]:
    """Return (sha, is_dirty) for the current git HEAD.

    Returns ('unknown', False) when not running inside a git working tree
    (e.g. SLURM jobs that copy files to scratch space -- pitfall 6 from
    research).
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty_output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha, bool(dirty_output)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    val_metric: float,
    config_snapshot: dict[str, Any],
    history: list[dict[str, float]],
    seed: int,
) -> None:
    """Save a full-reproducibility checkpoint to disk.

    The checkpoint dict contains the model state, training provenance (git SHA
    and dirty flag), the complete YAML config snapshot, per-epoch history,
    the random seed and environment information.

    Args:
        path: Destination file path (must end in .pt).
        model: Model whose state_dict to save.
        epoch: Epoch number at which this checkpoint was produced.
        val_metric: Validation metric value that triggered the save.
        config_snapshot: Full YAML config dict for reproducibility metadata.
        history: Per-epoch history list accumulated so far.
        seed: Random seed used for this run.
    """
    git_sha, git_dirty = _get_git_sha()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_metric": val_metric,
            "config": config_snapshot,
            "git_sha": git_sha,
            "git_dirty": git_dirty,
            "history": history,
            "seed": seed,
            "env": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda or "none",
                "platform": platform.platform(),
            },
        },
        path,
    )


class Trainer:
    """Reusable training loop for multi-label graph classification.

    Owns the complete training lifecycle: epoch loop, batch iteration, loss
    computation, optimizer step, gradient accumulation, gradient clipping,
    validation, early stopping, device placement and checkpoint saving.

    The training script configures the model, optimizer, loss function and
    data loaders, then delegates to trainer.fit(). This keeps the script as
    a thin setup layer and makes the training logic reusable across
    architectures and ablation experiments.

    Args:
        model: GNN model to train.
        optimizer: PyTorch optimizer wrapping model.parameters().
        loss_fn: Loss module with signature (logits, targets) -> scalar.
            Typically built via build_loss_fn() from mlcq_graphs.training.
        device: Device to move model and batches to.
        epochs: Maximum number of training epochs.
        patience: Early stopping patience (number of epochs without improvement
            before stopping). Set to a large value to effectively disable.
        min_delta: Minimum improvement in monitored metric to count as
            improvement. Default 0.0 means any improvement triggers.
        early_stopping_metric: Which validation metric to monitor. One of
            'macro_f1' or 'pr_auc'. The key must exist in METRIC_KEY_MAP.
        checkpoint_dir: Directory where checkpoint files are saved. Created if
            it does not exist.
        config_snapshot: Full YAML config dict, saved verbatim in checkpoint
            for reproducibility.
        seed: Random seed for this run, encoded in checkpoint filename and
            saved in checkpoint metadata.
        grad_accum_steps: Number of mini-batches to accumulate gradients over
            before taking an optimizer step. Default 1 (no accumulation).
        grad_clip_norm: Maximum gradient norm for gradient clipping. Default
            1.0. Set to 0.0 to disable clipping.
        architecture: Architecture name (e.g. 'gcn', 'gat', 'sage') encoded
            in the checkpoint filename.
        loss_name: Loss function name (e.g. 'weighted_bce', 'focal') encoded
            in the checkpoint filename.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        epochs: int,
        patience: int,
        min_delta: float,
        early_stopping_metric: str,
        checkpoint_dir: Path,
        config_snapshot: dict[str, Any],
        seed: int,
        grad_accum_steps: int = 1,
        grad_clip_norm: float = 1.0,
        architecture: str = "gcn",
        loss_name: str = "weighted_bce",
    ) -> None:
        if early_stopping_metric not in METRIC_KEY_MAP:
            raise ValueError(
                f"Unknown early_stopping_metric {early_stopping_metric!r}. "
                f"Valid options: {list(METRIC_KEY_MAP.keys())}"
            )
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta
        self.early_stopping_metric = early_stopping_metric
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config_snapshot = config_snapshot
        self.seed = seed
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.grad_clip_norm = grad_clip_norm
        self.architecture = architecture
        self.loss_name = loss_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader_fn: Callable[[int], DataLoader],
        val_loader: DataLoader,
        num_labels: int,
        evaluate_fn: Callable[..., dict[str, Any]],
    ) -> TrainResult:
        """Run the full training loop and return a structured result.

        Args:
            train_loader_fn: Callable that takes an epoch number (int, 1-indexed)
                and returns a DataLoader for that epoch. This abstraction lets the
                script control DynamicBudgetBatchSampler or any other sampler
                without exposing those internals to the Trainer.
            val_loader: DataLoader for the validation set. Used for evaluation
                after every epoch.
            num_labels: Number of label dimensions per sample. Used to reshape
                batch.y and passed to evaluate_fn.
            evaluate_fn: Evaluation function with signature::

                evaluate_fn(
                    model, loader, device, num_labels, thresholds
                ) -> dict[str, float | list[float]]

                Expected keys in the returned dict: 'f1_micro', 'f1_macro',
                'pr_auc_macro'. Injected to avoid circular imports with the
                training script.

        Returns:
            TrainResult with best epoch, best metric value, checkpoint path,
            per-epoch history and whether early stopping was triggered.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint paths: named file for self-documentation plus best_model.pt
        # for pipeline compatibility (open question 2 resolution in RESEARCH.md).
        named_ckpt = self.checkpoint_dir / (
            f"best_model_{self.architecture}_{self.loss_name}_s{self.seed}.pt"
        )
        compat_ckpt = self.checkpoint_dir / "best_model.pt"

        # Move model and loss to device.
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        metric_key = METRIC_KEY_MAP[self.early_stopping_metric]

        best_metric: float = -float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch: int = -1
        no_improve: int = 0
        stopped_early: bool = False
        history: list[dict[str, float]] = []

        for epoch in range(1, self.epochs + 1):
            train_loader = train_loader_fn(epoch)
            avg_loss = self._train_one_epoch(train_loader, num_labels)

            val_metrics = evaluate_fn(
                model=self.model,
                loader=val_loader,
                device=self.device,
                num_labels=num_labels,
                thresholds=None,
            )

            entry: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": avg_loss,
                "val_f1_micro": float(val_metrics["f1_micro"]),
                "val_f1_macro": float(val_metrics["f1_macro"]),
                "val_pr_auc_macro": float(val_metrics["pr_auc_macro"]),
            }
            history.append(entry)

            print(
                f"Epoch {epoch:03d} | loss={avg_loss:.4f} | "
                f"val_f1_micro={entry['val_f1_micro']:.4f} | "
                f"val_f1_macro={entry['val_f1_macro']:.4f} | "
                f"val_pr_auc={entry['val_pr_auc_macro']:.4f}"
            )

            if epoch % 10 == 0 or epoch == 1:
                f1_per_label = val_metrics.get("f1_per_label", [])
                if f1_per_label:
                    parts = [f"{v:.3f}" for v in f1_per_label]
                    print(f"  per-label f1: {' | '.join(parts)}")

            if entry["val_f1_macro"] < 0.05 and epoch > 5:
                print(f"  [warn] val_f1_macro={entry['val_f1_macro']:.4f} very low -- possible label collapse")

            val_metric = float(val_metrics[metric_key])
            improved = (val_metric - best_metric) > self.min_delta

            if improved:
                best_metric = val_metric
                best_epoch = epoch
                no_improve = 0
                # Keep CPU copy for fast restore at end of training.
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                _save_checkpoint(
                    named_ckpt,
                    self.model,
                    epoch,
                    val_metric,
                    self.config_snapshot,
                    history,
                    self.seed,
                )
                shutil.copy2(named_ckpt, compat_ckpt)
            else:
                no_improve += 1

            if no_improve >= self.patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {self.patience} epochs)"
                )
                stopped_early = True
                break

        # Always restore best weights before returning -- pitfall 4 from research.
        # The last epoch is not guaranteed to be the best epoch even when early
        # stopping was not triggered.
        if best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )

        return TrainResult(
            best_epoch=best_epoch,
            best_val_metric=best_metric,
            checkpoint_path=str(named_ckpt),
            history=history,
            stopped_early=stopped_early,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(
        self,
        loader: DataLoader,
        num_labels: int,
    ) -> float:
        """Run one training epoch and return the average loss per graph.

        Implements gradient accumulation and optional gradient clipping.
        Remaining gradients after the final batch are flushed even when the
        total number of batches is not divisible by grad_accum_steps.

        Args:
            loader: DataLoader for the training set.
            num_labels: Number of label dimensions. Used to reshape batch.y.

        Returns:
            Average loss per graph for this epoch.
        """
        self.model.train()
        total_loss: float = 0.0
        total_graphs: int = 0
        step_count: int = 0

        self.optimizer.zero_grad()

        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
            y = batch.y.view(-1, num_labels).float()

            loss = self.loss_fn(logits, y)
            scaled_loss = loss / self.grad_accum_steps
            scaled_loss.backward()

            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs
            step_count += 1

            if step_count % self.grad_accum_steps == 0:
                if self.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Flush remaining accumulated gradients after the final batch.
        if step_count % self.grad_accum_steps != 0:
            if self.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss / total_graphs if total_graphs > 0 else 0.0
