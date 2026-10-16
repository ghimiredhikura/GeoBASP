"""Scientific readiness checks shared by training entry points."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def check_primary_training_readiness(
    root: Path,
    data_config: dict,
    train_config: dict,
) -> dict:
    summary_path = root / data_config.get(
        "mask_summary", "data/processed/manifests/positive_masks_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    reasons: list[str] = []
    policy = str(data_config.get("background_policy", "missing"))
    if train_config.get("require_confirmed_background", True) and (
        policy == "missing" or "unknown" in policy
    ):
        reasons.append(f"background policy is unresolved: {policy}")
    if int(summary.get("background_supervised_pixels", 0)) <= 0:
        reasons.append("materialized masks contain zero supervised background pixels")
    return {
        "ready": not reasons,
        "background_policy": policy,
        "background_supervised_pixels": int(summary.get("background_supervised_pixels", 0)),
        "reasons": reasons,
    }


def load_configs(
    data_config_path: str | Path,
    train_config_path: str | Path,
) -> tuple[Path, dict, dict]:
    data_path = Path(data_config_path).resolve()
    train_path = Path(train_config_path).resolve()
    data_config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    train_config = yaml.safe_load(train_path.read_text(encoding="utf-8"))
    root = (data_path.parent.parent / data_config.get("project_root", ".")).resolve()
    return root, data_config, train_config
