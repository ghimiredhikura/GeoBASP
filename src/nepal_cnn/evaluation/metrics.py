"""Streaming binary segmentation metrics over explicitly supervised pixels."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BinarySegmentationMeter:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        supervision: torch.Tensor,
        threshold: float = 0.5,
    ) -> None:
        valid = supervision.bool()
        prediction = logits.sigmoid() >= threshold
        truth = target.bool()
        self.true_positive += int((prediction & truth & valid).sum().item())
        self.false_positive += int((prediction & ~truth & valid).sum().item())
        self.false_negative += int((~prediction & truth & valid).sum().item())
        self.true_negative += int((~prediction & ~truth & valid).sum().item())

    def compute(self) -> dict[str, float | int]:
        tp, fp, fn, tn = (
            self.true_positive,
            self.false_positive,
            self.false_negative,
            self.true_negative,
        )
        epsilon = 1e-12
        return {
            "dice": (2 * tp) / (2 * tp + fp + fn + epsilon),
            "iou": tp / (tp + fp + fn + epsilon),
            "precision": tp / (tp + fp + epsilon),
            "recall": tp / (tp + fn + epsilon),
            "supervised_pixels": tp + fp + fn + tn,
            "positive_pixels": tp + fn,
            "false_positive_pixels": fp,
            "predicted_positive_pixels": tp + fp,
            "false_positive_fraction": fp / (fp + tn + epsilon),
        }
