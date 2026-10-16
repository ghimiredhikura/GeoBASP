"""Evaluate a frozen segmentation checkpoint at a validation-selected threshold."""

from __future__ import annotations

from nepal_cnn.utils.paths import PROJECT_ROOT as ROOT

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from nepal_cnn.data.dataset import KoshiPositiveDataset
from nepal_cnn.models.checkpoints import load_checkpoint


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    return {
        "dice": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "iou": safe_ratio(tp, tp + fp + fn),
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold-report", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    checkpoint_path = ROOT / args.checkpoint
    threshold_path = ROOT / args.threshold_report
    output_path = ROOT / args.output
    threshold_report = json.loads(threshold_path.read_text(encoding="utf-8"))
    threshold = float(threshold_report["best"]["threshold"])

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    checkpoint, model = load_checkpoint(checkpoint_path, device)
    model.eval()

    dataset = KoshiPositiveDataset(ROOT / args.data_config, split=args.split, normalize=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    records: list[dict] = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True).bool()
            supervision = batch["supervision"].to(device, non_blocking=True).bool()
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prediction = model(image).sigmoid() >= threshold
            for index in range(len(batch["patch_id"])):
                valid = supervision[index]
                truth = target[index]
                estimate = prediction[index]
                tp = int((estimate & truth & valid).sum())
                fp = int((estimate & ~truth & valid).sum())
                fn = int((~estimate & truth & valid).sum())
                records.append(
                    {
                        "patch_id": batch["patch_id"][index],
                        "block_id": batch["block_id"][index],
                        "scene": batch["scene"][index],
                        "sample_type": batch["sample_type"][index],
                        "supervised_pixels": int(valid.sum()),
                        "positive_pixels": int((truth & valid).sum()),
                        "true_positive": tp,
                        "false_positive": fp,
                        "false_negative": fn,
                        **metrics(tp, fp, fn),
                    }
                )

    frame = pd.DataFrame(records)
    tp = int(frame.true_positive.sum())
    fp = int(frame.false_positive.sum())
    fn = int(frame.false_negative.sum())
    negative = frame[frame.sample_type == "negative"]
    summary = {
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_epoch": int(checkpoint.get("epoch", checkpoint.get("validation_recovered", {}).get("epoch", 0))),
        "threshold_report": args.threshold_report.as_posix(),
        "threshold_selected_on_validation": threshold,
        "split": args.split,
        "patches": len(frame),
        "spatial_blocks": int(frame.block_id.nunique()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        **metrics(tp, fp, fn),
        "negative_patches": len(negative),
        "negative_patch_supervised_pixels": int(negative.supervised_pixels.sum()),
        "negative_patch_false_positive_pixels": int(negative.false_positive.sum()),
        "negative_patch_false_positive_fraction": safe_ratio(
            int(negative.false_positive.sum()), int(negative.supervised_pixels.sum())
        ),
        "device": str(device),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    frame.to_csv(output_path.with_name(output_path.stem + "_per_patch.csv"), index=False)
    block = frame.groupby("block_id", as_index=False)[
        ["supervised_pixels", "positive_pixels", "true_positive", "false_positive", "false_negative"]
    ].sum()
    block.to_csv(output_path.with_name(output_path.stem + "_per_block.csv"), index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
