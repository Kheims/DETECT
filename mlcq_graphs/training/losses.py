"""Loss functions for multi-label code smell classification.

Provides WeightedBCELoss and FocalLoss implementations along with a factory
function ``build_loss_fn`` and an inverse-frequency weight calculator
``compute_class_weights``.

Both loss classes use ``F.binary_cross_entropy_with_logits`` as the numerically
stable BCE core (log-sum-exp trick prevents overflow for extreme logit values).
The focal loss implementation follows the torchvision reference pattern to
avoid the common p_t computation pitfall.

Pitfall notes (documented here to aid code review):
- weighted_bce puts inverse-frequency weights in ``pos_weight``
- focal puts them in ``class_weight`` — never mix the two
- Alpha guard uses ``>= 0`` not ``if alpha`` so that alpha=0.0 is not silently
  treated as disabled
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """BCEWithLogitsLoss with per-label pos_weight from inverse frequency.

    The pos_weight tensor is registered as a buffer so it moves with the
    module when ``.to(device)`` is called.

    Args:
        pos_weight: Per-label positive weights, shape ``[num_labels]``.
    """

    def __init__(self, pos_weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


class FocalLoss(nn.Module):
    """Binary focal loss for multi-label classification.

    Numerically stable implementation following the torchvision reference
    (Lin et al. 2017, RetinaNet). Class weights (inverse frequency) act as
    per-label alpha; the scalar ``alpha`` parameter is an additional global
    scaling knob disabled by default.

    Source pattern: torchvision.ops.sigmoid_focal_loss
    (https://docs.pytorch.org/vision/stable/_modules/torchvision/ops/focal_loss.html)

    Args:
        gamma: Focusing exponent. Default 2.0 (Lin et al. standard).
        alpha: Global positive/negative balance factor. ``-1.0`` disables it
            (sentinel convention matching torchvision). Use class_weight for
            per-label imbalance instead.
        class_weight: Per-label weights, shape ``[num_labels]``. Applied after
            focal modulation. Registered as a buffer when provided.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = -1.0,
        class_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight)
        else:
            self.class_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # p_t = probability of the true class (not just p)
        # This is the key computation: for positive labels p_t=p, for negatives p_t=1-p
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha guard: use >= 0 (not `if alpha`) so alpha=0.0 is not skipped
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            ce_loss = alpha_t * ce_loss

        loss = focal_weight * ce_loss  # shape: [batch, num_labels]

        if self.class_weight is not None:
            loss = loss * self.class_weight.view(1, -1)

        return loss.mean()


def compute_class_weights(
    train_ds: list,
    num_labels: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute inverse-frequency weights for each label.

    Formula: ``weight = N / (K * count_per_label)``

    where N is total samples, K is number of labels and count_per_label is
    the number of positive examples for that label. Labels with zero positives
    are clamped to 1 to avoid division by zero.

    Args:
        train_ds: List of PyG Data objects. Each must have a ``y`` attribute.
        num_labels: Number of label dimensions to use from each ``y`` tensor.
        device: Device to place the returned tensor on.

    Returns:
        Weight tensor of shape ``[num_labels]`` on ``device``.
    """
    y = torch.stack([d.y.view(-1)[:num_labels] for d in train_ds]).float()
    n = float(y.size(0))
    k = float(num_labels)
    counts = y.sum(dim=0).clamp(min=1.0)
    weights = n / (k * counts)
    return weights.to(device)


def build_loss_fn(
    loss_name: str,
    train_ds: list,
    num_labels: int,
    device: torch.device,
    focal_gamma: float = 2.0,
    focal_alpha: float = -1.0,
) -> nn.Module:
    """Factory: build and return the configured loss module moved to device.

    Computes class weights once from ``train_ds`` via ``compute_class_weights``
    and routes them to the appropriate slot for the chosen loss:

    - ``weighted_bce``: inverse-frequency weights go into ``pos_weight``
    - ``focal``: inverse-frequency weights go into ``class_weight``

    Never apply both simultaneously to the same labels.

    Args:
        loss_name: One of ``"weighted_bce"`` or ``"focal"``.
        train_ds: Training dataset (list of PyG Data objects).
        num_labels: Number of labels per sample.
        device: Device for the returned module and its buffers.
        focal_gamma: Focusing exponent for focal loss. Ignored for weighted_bce.
        focal_alpha: Global alpha for focal loss. ``-1.0`` disables it.
            Ignored for weighted_bce.

    Returns:
        A loss module with signature ``(logits, targets) -> scalar_loss``.

    Raises:
        ValueError: If ``loss_name`` is not a recognised option.
    """
    class_weights = compute_class_weights(train_ds, num_labels, device)

    if loss_name == "weighted_bce":
        return WeightedBCELoss(pos_weight=class_weights).to(device)

    if loss_name == "focal":
        return FocalLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
            class_weight=class_weights,
        ).to(device)

    raise ValueError(
        f"Unknown loss function {loss_name!r}. Valid options: 'weighted_bce', 'focal'."
    )
