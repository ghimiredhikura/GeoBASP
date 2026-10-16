"""Manuscript-level validation analysis for dense and pruned checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from nepal_cnn.data.dataset import KoshiPositiveDataset
from nepal_cnn.pruning.audit import load_dense_model


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    return {
        "dice": _ratio(2 * tp, 2 * tp + fp + fn),
        "iou": _ratio(tp, tp + fp + fn),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
    }


def _boundary(mask: torch.Tensor, width: int) -> torch.Tensor:
    mask = mask.float()
    kernel = 2 * width + 1
    dilated = F.max_pool2d(mask, kernel, stride=1, padding=width)
    eroded = 1 - F.max_pool2d(1 - mask, kernel, stride=1, padding=width)
    return (dilated - eroded) > 0


def _size_bin(area: int, edges: list[int]) -> str:
    lower = 1
    for upper in edges:
        if area <= upper:
            return f"{lower}-{upper} px"
        lower = upper + 1
    return f">{edges[-1]} px"


def _loader(
    data_config: Path,
    batch_size: int,
    device: torch.device,
    split: str = "validation",
) -> DataLoader:
    dataset = KoshiPositiveDataset(
        data_config, split=split, normalize=True, augment=False
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


@torch.no_grad()
def _basic_records(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> pd.DataFrame:
    records = []
    model.eval()
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True).bool()
        supervision = batch["supervision"].to(device, non_blocking=True).bool()
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = model(image).sigmoid() >= threshold
        for index in range(len(batch["patch_id"])):
            valid = supervision[index]
            truth = target[index] & valid
            estimate = prediction[index] & valid
            tp = int((estimate & truth).sum())
            fp = int((estimate & ~truth & valid).sum())
            fn = int((~estimate & truth).sum())
            records.append(
                {
                    "patch_id": batch["patch_id"][index],
                    "block_id": batch["block_id"][index],
                    "sample_type": batch["sample_type"][index],
                    "supervised_pixels": int(valid.sum()),
                    "positive_pixels": int(truth.sum()),
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                }
            )
    return pd.DataFrame(records)


def ensure_dense_validation_reference(
    root: Path,
    model_name: str,
    checkpoint_path: Path,
    data_config: Path,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, float]:
    """Cache dense validation counts once for paired comparisons across branches."""
    checkpoint_path = checkpoint_path.resolve()
    root = root.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    patch_path = cache_dir / f"{model_name}_per_patch.csv"
    metadata_path = cache_dir / f"{model_name}.json"
    if patch_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return pd.read_csv(patch_path), float(metadata["threshold"])
    checkpoint, model = load_dense_model(checkpoint_path, device)
    threshold = float(checkpoint["metrics"]["validation_best_threshold"])
    records = _basic_records(
        model,
        _loader(data_config, batch_size, device),
        device,
        threshold,
    )
    records.to_csv(patch_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "checkpoint": str(checkpoint_path.relative_to(root)).replace("\\", "/"),
                "threshold": threshold,
                "split": "validation",
                "test_data_accessed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del checkpoint, model
    torch.cuda.empty_cache()
    return records, threshold


def _aggregate_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    block = frame.groupby("block_id", as_index=False)[
        [
            "supervised_pixels",
            "positive_pixels",
            "true_positive",
            "false_positive",
            "false_negative",
        ]
    ].sum()
    block["dice"] = [
        _metrics(int(row.true_positive), int(row.false_positive), int(row.false_negative))[
            "dice"
        ]
        for row in block.itertuples()
    ]
    return block


def _paired_bootstrap(
    pruned_blocks: pd.DataFrame,
    dense_blocks: pd.DataFrame,
    replicates: int,
    seed: int,
    margin: float,
) -> dict:
    columns = ["true_positive", "false_positive", "false_negative"]
    paired = pruned_blocks[["block_id", *columns]].merge(
        dense_blocks[["block_id", *columns]],
        on="block_id",
        how="outer",
        suffixes=("_pruned", "_dense"),
    ).fillna(0)
    pruned = paired[[f"{name}_pruned" for name in columns]].to_numpy(np.int64)
    dense = paired[[f"{name}_dense" for name in columns]].to_numpy(np.int64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(paired), size=(replicates, len(paired)))
    p = pruned[indices].sum(axis=1)
    d = dense[indices].sum(axis=1)
    p_dice = np.divide(
        2 * p[:, 0],
        2 * p[:, 0] + p[:, 1] + p[:, 2],
        out=np.zeros(replicates),
        where=(2 * p[:, 0] + p[:, 1] + p[:, 2]) != 0,
    )
    d_dice = np.divide(
        2 * d[:, 0],
        2 * d[:, 0] + d[:, 1] + d[:, 2],
        out=np.zeros(replicates),
        where=(2 * d[:, 0] + d[:, 1] + d[:, 2]) != 0,
    )
    delta = p_dice - d_dice
    lower, upper = np.quantile(delta, [0.025, 0.975])
    return {
        "unit": "20 km spatial block",
        "paired": True,
        "replicates": replicates,
        "seed": seed,
        "delta_dice_mean": float(delta.mean()),
        "delta_dice_ci95": [float(lower), float(upper)],
        "noninferiority_margin": margin,
        "passes_noninferiority": bool(lower > margin),
    }


@torch.no_grad()
def evaluate_manuscript_validation(
    model: torch.nn.Module,
    model_name: str,
    threshold: float,
    dense_checkpoint_path: Path,
    root: Path,
    data_config: Path,
    output_dir: Path,
    dense_cache_dir: Path,
    device: torch.device,
    batch_size: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    noninferiority_margin: float,
    boundary_width: int = 2,
    component_edges: list[int] | None = None,
    split: str = "validation",
    dense_records_path: Path | None = None,
    dense_threshold_override: float | None = None,
) -> dict:
    """Produce manuscript accuracy evidence for validation or a frozen final test."""
    component_edges = component_edges or [9, 25, 100]
    if split == "validation":
        dense_records, dense_threshold = ensure_dense_validation_reference(
            root,
            model_name,
            dense_checkpoint_path,
            data_config,
            dense_cache_dir,
            device,
            batch_size,
        )
    elif split == "test":
        if dense_records_path is None or dense_threshold_override is None:
            raise ValueError(
                "Frozen test evaluation requires dense_records_path and "
                "dense_threshold_override"
            )
        dense_records = pd.read_csv(dense_records_path)
        dense_threshold = float(dense_threshold_override)
    else:
        raise ValueError(f"Unsupported evaluation split: {split}")
    loader = _loader(data_config, batch_size, device, split=split)
    model.eval()
    records = []
    components = []
    boundary_predicted = boundary_matched_predicted = 0
    boundary_truth = boundary_matched_truth = 0
    structure = np.ones((3, 3), dtype=np.uint8)
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True).bool()
        supervision = batch["supervision"].to(device, non_blocking=True).bool()
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = model(image).sigmoid() >= threshold
        truth_valid = target & supervision
        prediction_valid = prediction & supervision
        truth_boundary = _boundary(truth_valid.float(), boundary_width) & supervision
        prediction_boundary = _boundary(prediction_valid.float(), boundary_width) & supervision
        tolerance_kernel = 2 * boundary_width + 1
        truth_tolerance = F.max_pool2d(
            truth_boundary.float(),
            tolerance_kernel,
            stride=1,
            padding=boundary_width,
        ).bool()
        prediction_tolerance = F.max_pool2d(
            prediction_boundary.float(),
            tolerance_kernel,
            stride=1,
            padding=boundary_width,
        ).bool()
        boundary_predicted += int(prediction_boundary.sum())
        boundary_truth += int(truth_boundary.sum())
        boundary_matched_predicted += int((prediction_boundary & truth_tolerance).sum())
        boundary_matched_truth += int((truth_boundary & prediction_tolerance).sum())
        pred_np = prediction_valid.cpu().numpy().astype(bool)
        truth_np = truth_valid.cpu().numpy().astype(bool)
        valid_np = supervision.cpu().numpy().astype(bool)
        for index in range(len(batch["patch_id"])):
            valid = valid_np[index, 0]
            truth = truth_np[index, 0]
            estimate = pred_np[index, 0]
            tp = int((estimate & truth).sum())
            fp = int((estimate & ~truth & valid).sum())
            fn = int((~estimate & truth).sum())
            records.append(
                {
                    "patch_id": batch["patch_id"][index],
                    "block_id": batch["block_id"][index],
                    "scene": batch["scene"][index],
                    "sample_type": batch["sample_type"][index],
                    "supervised_pixels": int(valid.sum()),
                    "positive_pixels": int(truth.sum()),
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                    **_metrics(tp, fp, fn),
                }
            )
            labels, count = label(truth, structure=structure)
            for component_id in range(1, count + 1):
                component = labels == component_id
                area = int(component.sum())
                recovered = int((estimate & component).sum())
                components.append(
                    {
                        "patch_id": batch["patch_id"][index],
                        "block_id": batch["block_id"][index],
                        "component_id": component_id,
                        "area_pixels": area,
                        "size_bin": _size_bin(area, component_edges),
                        "recovered_pixels": recovered,
                        "coverage": _ratio(recovered, area),
                    }
                )
    frame = pd.DataFrame(records)
    component_frame = pd.DataFrame(components)
    block = _aggregate_blocks(frame)
    dense_block = _aggregate_blocks(dense_records)
    tp = int(frame.true_positive.sum())
    fp = int(frame.false_positive.sum())
    fn = int(frame.false_negative.sum())
    negative = frame[frame.sample_type == "negative"]
    boundary_precision = _ratio(boundary_matched_predicted, boundary_predicted)
    boundary_recall = _ratio(boundary_matched_truth, boundary_truth)
    component_summary = {}
    ordered_bins = ["1-9 px", "10-25 px", "26-100 px", ">100 px"]
    for bin_name in ["all", *ordered_bins]:
        subset = (
            component_frame
            if bin_name == "all"
            else component_frame[component_frame.size_bin == bin_name]
        )
        component_summary[bin_name] = {
            "components": int(len(subset)),
            "mean_coverage": float(subset.coverage.mean()) if len(subset) else 0.0,
            "detected_10pct_fraction": (
                float((subset.coverage >= 0.10).mean()) if len(subset) else 0.0
            ),
            "detected_50pct_fraction": (
                float((subset.coverage >= 0.50).mean()) if len(subset) else 0.0
            ),
        }
    positive_blocks = block[block.positive_pixels > 0]
    summary = {
        "model": model_name,
        "split": split,
        "threshold": threshold,
        "dense_threshold": dense_threshold,
        "patches": len(frame),
        "spatial_blocks": int(frame.block_id.nunique()),
        **_metrics(tp, fp, fn),
        "boundary_width_pixels": boundary_width,
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "boundary_f1": _ratio(
            2 * boundary_precision * boundary_recall,
            boundary_precision + boundary_recall,
        ),
        "negative_patch_false_positive_fraction": _ratio(
            int(negative.false_positive.sum()), int(negative.supervised_pixels.sum())
        ),
        "worst_positive_block_dice": (
            float(positive_blocks.dice.min()) if len(positive_blocks) else 0.0
        ),
        "component_recall": component_summary,
        "paired_spatial_bootstrap": _paired_bootstrap(
            block,
            dense_block,
            bootstrap_replicates,
            bootstrap_seed,
            noninferiority_margin,
        ),
        "test_data_accessed": split == "test",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{split}_per_patch.csv", index=False)
    block.to_csv(output_dir / f"{split}_per_block.csv", index=False)
    component_frame.to_csv(output_dir / f"{split}_per_component.csv", index=False)
    (output_dir / f"{split}_extended.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
