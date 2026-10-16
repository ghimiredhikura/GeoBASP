"""Training-only calibration conditions; submitted selectors remain unchanged."""

import numpy as np
import pandas as pd

from nepal_cnn.pruning.calibration import (
    CalibrationSelection, _add_geographic_stratum, _round_robin_pick,
    select_calibration_rows,
)

CONDITIONS = {
    "balanced_16": "12 positives + 4 ranked hard negatives; size/geography balanced",
    "balanced_32": "24 positives + 8 ranked hard negatives; submitted selection rule",
    "balanced_64": "48 positives + 16 ranked hard negatives; size/geography balanced",
    "balanced_32_resampled_negatives": "Same positives as balanced_32; independently drawn hard negatives, two per quadrant",
    "unstratified_positive_32": "24 positives without size/geography quotas; same negatives as balanced_32_resampled_negatives",
}


def training_strata(rows):
    assert set(rows.split) == {"train"} and rows.patch_id.is_unique
    work = _add_geographic_stratum(rows.reset_index().rename(columns={"index": "dataset_index"}))
    positive = work[work.sample_type == "positive"].copy()
    positive["calibration_size_stratum"] = pd.qcut(
        positive.landslide_pixels.rank(method="first"), q=3, labels=["small", "medium", "large"])
    negative = work[work.sample_type == "negative"].copy()
    negative["calibration_size_stratum"] = "negative"
    return positive, negative


def hard_negative_pool(rows):
    """Hardest patch from each block; retain the eight hardest blocks per quadrant."""
    _, negatives = training_strata(rows)
    pool = negatives.sort_values(["hard_negative_score", "patch_id"], ascending=[False, True])
    pool = pool.drop_duplicates("block_id")
    pool = pool.groupby("calibration_geo_stratum", sort=True).head(8)
    assert pool.block_id.is_unique and len(pool) == 32
    assert pool.groupby("calibration_geo_stratum").size().to_dict() == {"EN": 8, "ES": 8, "WN": 8, "WS": 8}
    return pool


def select_condition(rows, condition, sampling_seed):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if condition in ("balanced_16", "balanced_32", "balanced_64"):
        size = int(condition.rsplit("_", 1)[1])
        return select_calibration_rows(rows, 3*size//4, size//4, sampling_seed)
    positive, _ = training_strata(rows)
    negative = _round_robin_pick(hard_negative_pool(rows), ["calibration_geo_stratum"],
                               8, sampling_seed + 100000)
    if condition == "balanced_32_resampled_negatives":
        selected = select_calibration_rows(rows, 24, 8, sampling_seed).manifest
        positive = selected[selected.sample_type == "positive"].copy()
    else:
        positive = _round_robin_pick(positive, ["sample_type"], 24, sampling_seed)
    selected = pd.concat([positive, negative], ignore_index=True)
    selected["calibration_order"] = np.arange(len(selected))
    assert len(selected) == 32 and selected.patch_id.is_unique
    return CalibrationSelection(selected.dataset_index.astype(int).tolist(), selected, 24, 8)
