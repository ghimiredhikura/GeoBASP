"""Losses that distinguish supervised pixels from unknown pixels."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_bce_with_logits(
    logits: torch.Tensor, target: torch.Tensor, supervision: torch.Tensor
) -> torch.Tensor:
    mask = supervision.to(dtype=logits.dtype)
    denominator = mask.sum().clamp_min(1.0)
    return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask).sum() / denominator


def masked_soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    supervision: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    mask = supervision.to(dtype=logits.dtype)
    probability = logits.sigmoid() * mask
    target = target * mask
    dims = tuple(range(1, logits.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()


def masked_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    supervision: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    return bce_weight * masked_bce_with_logits(logits, target, supervision) + dice_weight * masked_soft_dice_loss(
        logits, target, supervision
    )


def masked_class_balanced_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    supervision: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss with positive and background pixels normalized separately."""
    valid = supervision.bool()
    positive = valid & target.bool()
    negative = valid & ~target.bool()
    cross_entropy = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability_correct = torch.exp(-cross_entropy)
    focal = (1.0 - probability_correct).pow(gamma) * cross_entropy
    terms = []
    if positive.any():
        terms.append(alpha * focal[positive].mean())
    if negative.any():
        terms.append((1.0 - alpha) * focal[negative].mean())
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).sum()


def masked_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    supervision: torch.Tensor,
    false_positive_weight: float = 0.30,
    false_negative_weight: float = 0.70,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Tversky loss averaged only over samples containing supervised positives."""
    valid = supervision.to(dtype=logits.dtype)
    truth = target * valid
    probability = logits.sigmoid() * valid
    dims = tuple(range(1, logits.ndim))
    true_positive = (probability * truth).sum(dim=dims)
    false_positive = (probability * (1.0 - truth) * valid).sum(dim=dims)
    false_negative = ((1.0 - probability) * truth).sum(dim=dims)
    contains_positive = truth.sum(dim=dims) > 0
    score = (true_positive + smooth) / (
        true_positive
        + false_positive_weight * false_positive
        + false_negative_weight * false_negative
        + smooth
    )
    if contains_positive.any():
        return (1.0 - score[contains_positive]).mean()
    return logits.sum() * 0.0


def masked_balanced_focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    supervision: torch.Tensor,
    focal_weight: float = 0.40,
    tversky_weight: float = 0.60,
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    tversky_false_positive_weight: float = 0.30,
    tversky_false_negative_weight: float = 0.70,
    smooth: float = 1.0,
) -> torch.Tensor:
    focal = masked_class_balanced_focal_loss(
        logits, target, supervision, alpha=focal_alpha, gamma=focal_gamma
    )
    tversky = masked_tversky_loss(
        logits,
        target,
        supervision,
        false_positive_weight=tversky_false_positive_weight,
        false_negative_weight=tversky_false_negative_weight,
        smooth=smooth,
    )
    return focal_weight * focal + tversky_weight * tversky


def masked_boundary_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    supervision: torch.Tensor,
    kernel_size: int = 3,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Soft morphological-boundary Dice inside fully supervised neighborhoods."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("Boundary kernel size must be an odd integer of at least 3")
    padding = kernel_size // 2
    valid = supervision.float()
    truth = target.float() * valid
    probability = logits.float().sigmoid() * valid

    valid_neighborhood = -F.max_pool2d(
        -valid, kernel_size=kernel_size, stride=1, padding=padding
    )
    valid_neighborhood = (valid_neighborhood > 0.999).float()

    def boundary(values: torch.Tensor) -> torch.Tensor:
        dilation = F.max_pool2d(
            values, kernel_size=kernel_size, stride=1, padding=padding
        )
        erosion = -F.max_pool2d(
            -values, kernel_size=kernel_size, stride=1, padding=padding
        )
        return (dilation - erosion).clamp(0.0, 1.0) * valid_neighborhood

    predicted_boundary = boundary(probability)
    target_boundary = boundary(truth)
    dims = tuple(range(1, logits.ndim))
    intersection = (predicted_boundary * target_boundary).sum(dim=dims)
    denominator = predicted_boundary.sum(dim=dims) + target_boundary.sum(dim=dims)
    contains_boundary = target_boundary.sum(dim=dims) > 0
    if contains_boundary.any():
        score = (2.0 * intersection + smooth) / (denominator + smooth)
        return (1.0 - score[contains_boundary]).mean()
    return logits.sum() * 0.0


def masked_balanced_focal_tversky_boundary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    supervision: torch.Tensor,
    boundary_weight: float = 0.15,
    boundary_kernel_size: int = 3,
    **base_kwargs: float,
) -> torch.Tensor:
    if not 0.0 <= boundary_weight < 1.0:
        raise ValueError("boundary_weight must be in [0, 1)")
    base = masked_balanced_focal_tversky_loss(
        logits, target, supervision, **base_kwargs
    )
    boundary = masked_boundary_dice_loss(
        logits,
        target,
        supervision,
        kernel_size=boundary_kernel_size,
        smooth=float(base_kwargs.get("smooth", 1.0)),
    )
    return (1.0 - boundary_weight) * base + boundary_weight * boundary


def configured_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    supervision: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    name = config.get("name", "bce_dice")
    if name == "bce_dice":
        return masked_bce_dice_loss(
            logits,
            target,
            supervision,
            bce_weight=float(config.get("bce_weight", 0.5)),
            dice_weight=float(config.get("dice_weight", 0.5)),
        )
    if name == "balanced_focal_tversky":
        return masked_balanced_focal_tversky_loss(
            logits,
            target,
            supervision,
            focal_weight=float(config.get("focal_weight", 0.4)),
            tversky_weight=float(config.get("tversky_weight", 0.6)),
            focal_alpha=float(config.get("focal_alpha", 0.75)),
            focal_gamma=float(config.get("focal_gamma", 2.0)),
            tversky_false_positive_weight=float(
                config.get("tversky_false_positive_weight", 0.3)
            ),
            tversky_false_negative_weight=float(
                config.get("tversky_false_negative_weight", 0.7)
            ),
            smooth=float(config.get("smooth", 1.0)),
        )
    if name == "balanced_focal_tversky_boundary":
        return masked_balanced_focal_tversky_boundary_loss(
            logits,
            target,
            supervision,
            boundary_weight=float(config.get("boundary_weight", 0.15)),
            boundary_kernel_size=int(config.get("boundary_kernel_size", 3)),
            focal_weight=float(config.get("focal_weight", 0.4)),
            tversky_weight=float(config.get("tversky_weight", 0.6)),
            focal_alpha=float(config.get("focal_alpha", 0.75)),
            focal_gamma=float(config.get("focal_gamma", 2.0)),
            tversky_false_positive_weight=float(
                config.get("tversky_false_positive_weight", 0.3)
            ),
            tversky_false_negative_weight=float(
                config.get("tversky_false_negative_weight", 0.7)
            ),
            smooth=float(config.get("smooth", 1.0)),
        )
    raise ValueError(f"Unknown segmentation loss: {name}")
