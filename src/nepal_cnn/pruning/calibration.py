"""Deterministic training-only calibration selection and gradient accumulation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from nepal_cnn.data.dataset import KoshiPositiveDataset
from nepal_cnn.training.losses import configured_segmentation_loss


@dataclass(frozen=True)
class CalibrationSelection:
    indices: list[int]
    manifest: pd.DataFrame
    positive_count: int
    negative_count: int


def _stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _add_geographic_stratum(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    x_mid = float(rows["block_x"].median())
    y_mid = float(rows["block_y"].median())
    rows["calibration_geo_stratum"] = (
        np.where(rows["block_x"] <= x_mid, "W", "E")
        + np.where(rows["block_y"] <= y_mid, "S", "N")
    )
    return rows


def _round_robin_pick(
    rows: pd.DataFrame,
    group_columns: list[str],
    count: int,
    seed: int,
    descending_column: str | None = None,
) -> pd.DataFrame:
    """Pick evenly across strata, preferring distinct 20 km blocks."""
    if count > len(rows):
        raise ValueError(f"Requested {count} calibration rows from only {len(rows)} candidates")
    work = rows.copy()
    work["_stable_key"] = [
        _stable_key(str(patch_id), seed) for patch_id in work["patch_id"]
    ]
    sort_columns = list(group_columns)
    ascending = [True] * len(group_columns)
    if descending_column is not None:
        sort_columns.append(descending_column)
        ascending.append(False)
    sort_columns.append("_stable_key")
    ascending.append(True)
    work = work.sort_values(sort_columns, ascending=ascending)
    groups = {
        key if isinstance(key, tuple) else (key,): group.index.tolist()
        for key, group in work.groupby(group_columns, sort=True, observed=True)
    }
    selected: list[int] = []
    used_blocks: set[str] = set()
    for require_new_block in (True, False):
        while len(selected) < count:
            added = False
            for key in sorted(groups):
                chosen = None
                for index in groups[key]:
                    if index in selected:
                        continue
                    block = str(work.loc[index, "block_id"])
                    if require_new_block and block in used_blocks:
                        continue
                    chosen = index
                    break
                if chosen is not None:
                    selected.append(chosen)
                    used_blocks.add(str(work.loc[chosen, "block_id"]))
                    added = True
                    if len(selected) == count:
                        break
            if not added:
                break
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Could select only {len(selected)} of {count} calibration rows")
    return work.loc[selected].drop(columns="_stable_key")


def select_calibration_rows(
    rows: pd.DataFrame,
    positive_count: int,
    negative_count: int,
    seed: int,
) -> CalibrationSelection:
    """Select positive-size/geography strata and high-scoring hard negatives."""
    if positive_count < 1 or negative_count < 1:
        raise ValueError("Calibration requires both positive and negative patches")
    work = _add_geographic_stratum(rows.reset_index().rename(columns={"index": "dataset_index"}))
    positives = work[work["sample_type"] == "positive"].copy()
    negatives = work[work["sample_type"] == "negative"].copy()
    positives["calibration_size_stratum"] = pd.qcut(
        positives["landslide_pixels"].rank(method="first"),
        q=3,
        labels=["small", "medium", "large"],
    )
    selected_positive = _round_robin_pick(
        positives,
        ["calibration_size_stratum", "calibration_geo_stratum"],
        positive_count,
        seed,
    )
    negatives["calibration_size_stratum"] = "negative"
    selected_negative = _round_robin_pick(
        negatives,
        ["calibration_geo_stratum"],
        negative_count,
        seed,
        descending_column="hard_negative_score",
    )
    selected = pd.concat([selected_positive, selected_negative], ignore_index=True)
    selected["calibration_order"] = np.arange(len(selected))
    return CalibrationSelection(
        indices=selected["dataset_index"].astype(int).tolist(),
        manifest=selected,
        positive_count=len(selected_positive),
        negative_count=len(selected_negative),
    )


def build_calibration_loader(
    data_config: str | Path,
    positive_count: int,
    negative_count: int,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, CalibrationSelection]:
    """Create a deterministic full-patch loader; no validation/test rows are used."""
    dataset = KoshiPositiveDataset(
        data_config,
        split="train",
        normalize=True,
        augment=False,
        crop_size=None,
    )
    selection = select_calibration_rows(
        dataset.rows,
        positive_count=positive_count,
        negative_count=negative_count,
        seed=seed,
    )
    loader = DataLoader(
        Subset(dataset, selection.indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, selection


def accumulate_overall_taylor_gradients(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_config: dict,
) -> float:
    """Accumulate mean overall segmentation loss across calibration patches."""
    model.eval()
    model.zero_grad(set_to_none=True)
    loss_sum = 0.0
    sample_count = len(loader.dataset)
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        supervision = batch["supervision"].to(device, non_blocking=True)
        logits = model(image)
        loss = configured_segmentation_loss(logits, target, supervision, loss_config)
        batch_count = int(image.shape[0])
        (loss * (batch_count / sample_count)).backward()
        loss_sum += float(loss.detach()) * batch_count
    return loss_sum / sample_count
