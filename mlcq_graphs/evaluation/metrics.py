from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)


def compute_full_metrics(
    y_true_np: np.ndarray,
    y_pred_np: np.ndarray,
    probs_np: np.ndarray,
    label_names: list[str],
) -> dict:
    """Compute comprehensive multi-label classification metrics.

    Args:
        y_true_np: Ground truth binary array of shape (N, num_labels).
        y_pred_np: Predicted binary array of shape (N, num_labels).
        probs_np: Predicted probabilities array of shape (N, num_labels).
        label_names: Ordered list of label names, length num_labels.

    Returns:
        dict with aggregate metrics at top level and nested per-label breakdown.
    """
    num_labels = y_true_np.shape[1]

    f1_micro = float(f1_score(y_true_np, y_pred_np, average="micro", zero_division=0))
    f1_macro = float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0))
    f1_per_label_arr = f1_score(y_true_np, y_pred_np, average=None, zero_division=0)
    f1_per_label: list[float] = [float(v) for v in f1_per_label_arr]

    precision_per_label = precision_score(y_true_np, y_pred_np, average=None, zero_division=0)
    recall_per_label = recall_score(y_true_np, y_pred_np, average=None, zero_division=0)

    h_loss = float(hamming_loss(y_true_np, y_pred_np))
    subset_acc = float(accuracy_score(y_true_np, y_pred_np))

    pr_auc_per_label: list[float] = []
    for i in range(num_labels):
        if y_true_np[:, i].sum() == 0:
            pr_auc_per_label.append(0.0)
        else:
            try:
                val = float(average_precision_score(y_true_np[:, i], probs_np[:, i]))
            except Exception:
                val = 0.0
            pr_auc_per_label.append(val)

    pr_auc_macro = float(np.mean(pr_auc_per_label))

    mcm = multilabel_confusion_matrix(y_true_np, y_pred_np)

    per_label: dict[str, dict] = {}
    for i, name in enumerate(label_names):
        tn = int(mcm[i, 0, 0])
        fp = int(mcm[i, 0, 1])
        fn = int(mcm[i, 1, 0])
        tp = int(mcm[i, 1, 1])
        per_label[name] = {
            "precision": float(precision_per_label[i]),
            "recall": float(recall_per_label[i]),
            "f1": float(f1_per_label[i]),
            "pr_auc": pr_auc_per_label[i],
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "pr_auc_macro": pr_auc_macro,
        "hamming_loss": h_loss,
        "subset_accuracy": subset_acc,
        # flat lists for backward compatibility with pipeline ablation code
        "f1_per_label": f1_per_label,
        "pr_auc_per_label": pr_auc_per_label,
        "per_label": per_label,
    }


def evaluate_full(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    thresholds: torch.Tensor | None,
    label_names: list[str],
) -> dict:
    """Compute comprehensive multi-label metrics from raw model outputs.

    Applies sigmoid to logits, applies per-label thresholds (default 0.5) and
    delegates to compute_full_metrics for the actual metric computation.

    Args:
        logits: Raw model output of shape (N, num_labels), not yet sigmoided.
        y_true: Ground truth binary tensor of shape (N, num_labels).
        thresholds: Per-label threshold tensor of shape (num_labels,). None uses 0.5.
        label_names: Ordered list of label names, length num_labels.

    Returns:
        dict with aggregate and per-label metrics — see compute_full_metrics for schema.
    """
    num_labels = len(label_names)

    if logits.numel() == 0:
        empty_per_label = {
            name: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "pr_auc": 0.0,
                   "tp": 0, "fp": 0, "tn": 0, "fn": 0}
            for name in label_names
        }
        return {
            "f1_micro": 0.0,
            "f1_macro": 0.0,
            "pr_auc_macro": 0.0,
            "hamming_loss": 0.0,
            "subset_accuracy": 0.0,
            "f1_per_label": [0.0] * num_labels,
            "pr_auc_per_label": [0.0] * num_labels,
            "per_label": empty_per_label,
        }

    logits = logits.cpu().float()
    y_true = y_true.cpu().float()

    probs = torch.sigmoid(logits)

    if thresholds is None:
        thresholds = torch.full((num_labels,), 0.5)
    thresholds = thresholds.cpu().float()

    y_pred = (probs >= thresholds.view(1, -1)).int()
    y_true_int = y_true.int()

    y_true_np = y_true_int.numpy()
    y_pred_np = y_pred.numpy()
    probs_np = probs.numpy()

    return compute_full_metrics(y_true_np, y_pred_np, probs_np, label_names)
