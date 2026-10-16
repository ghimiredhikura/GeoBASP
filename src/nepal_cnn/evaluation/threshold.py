"""Select a segmentation probability threshold using validation data only."""

from __future__ import annotations

from nepal_cnn.utils.paths import PROJECT_ROOT as ROOT

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nepal_cnn.data.dataset import KoshiPositiveDataset
from nepal_cnn.models.checkpoints import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/resnet50/dense.pt"))
    parser.add_argument("--step", type=float, default=0.025)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data_terrain.yaml"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated/exploratory/baseline/validation_threshold.json"),
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    dataset = KoshiPositiveDataset(ROOT / args.data_config, split="validation", normalize=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    checkpoint, model = load_checkpoint(args.checkpoint, device)
    model.eval()

    thresholds = torch.arange(args.step, 1.0, args.step, device=device)
    true_positive = torch.zeros_like(thresholds, dtype=torch.float64)
    false_positive = torch.zeros_like(thresholds, dtype=torch.float64)
    false_negative = torch.zeros_like(thresholds, dtype=torch.float64)
    negative_false_positive = torch.zeros_like(thresholds, dtype=torch.float64)
    negative_supervised = 0

    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True).bool()
            supervision = batch["supervision"].to(device, non_blocking=True).bool()
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                probability = model(image).sigmoid()
            prediction = probability.unsqueeze(0) >= thresholds[:, None, None, None, None]
            valid_target = target.unsqueeze(0)
            valid = supervision.unsqueeze(0)
            true_positive += (prediction & valid_target & valid).sum(dim=(1, 2, 3, 4))
            false_positive += (prediction & ~valid_target & valid).sum(dim=(1, 2, 3, 4))
            false_negative += (~prediction & valid_target & valid).sum(dim=(1, 2, 3, 4))

            negative_indices = [
                index for index, sample_type in enumerate(batch["sample_type"])
                if sample_type == "negative"
            ]
            if negative_indices:
                negative_probability = probability[negative_indices]
                negative_valid = supervision[negative_indices]
                negative_prediction = (
                    negative_probability.unsqueeze(0)
                    >= thresholds[:, None, None, None, None]
                )
                negative_false_positive += (
                    negative_prediction & negative_valid.unsqueeze(0)
                ).sum(dim=(1, 2, 3, 4))
                negative_supervised += int(negative_valid.sum())

    dice = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    iou = true_positive / (true_positive + false_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    best = int(torch.argmax(dice))

    def record(index: int) -> dict:
        return {
            "threshold": float(thresholds[index]),
            "dice": float(dice[index]),
            "iou": float(iou[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "negative_patch_false_positive_fraction": float(
                negative_false_positive[index] / negative_supervised
            ),
        }

    requested = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    indices = sorted(
        {best}
        | {
            int(torch.argmin(torch.abs(thresholds - threshold)))
            for threshold in requested
        }
    )
    report = {
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_epoch": int(checkpoint.get("epoch", checkpoint.get("validation_recovered", {}).get("epoch", 0))),
        "validation_patches": len(dataset),
        "selection_metric": "global pixel Dice over primary supervision",
        "device": str(device),
        "batch_size": args.batch_size,
        "best": record(best),
        "selected_thresholds": [record(index) for index in indices],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
