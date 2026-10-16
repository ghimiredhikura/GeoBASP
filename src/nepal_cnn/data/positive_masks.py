"""Materialize target, observation, and positive-only supervision masks."""

from __future__ import annotations

from nepal_cnn.utils.paths import PROJECT_ROOT as ROOT

import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.windows import Window, bounds as window_bounds, from_bounds


TILE_RE = re.compile(r"T(45R[A-Z]{2})")
DATE_RE = re.compile(r"(20\d{6})T")


def scene_key(value: str) -> tuple[str, str]:
    tile = TILE_RE.search(str(value))
    date = DATE_RE.search(str(value))
    if not tile or not date:
        raise ValueError(value)
    return tile.group(1), date.group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = (config_path.parent.parent / cfg.get("project_root", ".")).resolve()

    manifest_path = root / cfg["output_root"] / "manifests" / "positive_patches.csv"
    manifest = pd.read_csv(manifest_path)
    inventory = gpd.read_file(root / cfg["inventory"]).to_crs(cfg["target_crs"])
    inventory["_scene_key"] = [scene_key(v) for v in inventory["Image_ID"]]
    inventory = inventory.set_index(inventory["ID"].astype(int), drop=False)

    mask_dir = root / cfg["output_root"] / "positive_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    invalid_classes = np.asarray(cfg["invalid_scl_classes"], dtype=np.uint8)
    scl_positive_counts = np.zeros(12, dtype=np.int64)
    total_positive = total_supervised_positive = 0
    mismatched_pixel_counts: list[str] = []
    mask_paths: dict[str, str] = {}

    for scene, group in manifest.groupby("scene", sort=True):
        scene_dir = root / cfg["sentinel_root"] / scene
        with (
            rasterio.open(scene_dir / f"{cfg['reference_band']}.tif") as ref,
            rasterio.open(scene_dir / "SCL.tif") as scl_src,
        ):
            for row in group.itertuples(index=False):
                window = Window(row.col_off, row.row_off, row.width, row.height)
                transform = ref.window_transform(window)
                ids = [int(value) for value in str(row.polygon_ids).split(";") if value]
                shapes = [(inventory.loc[value].geometry, 1) for value in ids]
                target = rasterize(
                    shapes,
                    out_shape=(row.height, row.width),
                    transform=transform,
                    fill=0,
                    dtype="uint8",
                    all_touched=False,
                )
                if int(target.sum()) != int(row.landslide_pixels):
                    mismatched_pixel_counts.append(row.patch_id)

                bounds = window_bounds(window, ref.transform)
                scl_window = from_bounds(*bounds, transform=scl_src.transform)
                scl = scl_src.read(
                    1,
                    window=scl_window,
                    out_shape=(row.height, row.width),
                    boundless=True,
                    fill_value=0,
                    resampling=Resampling.nearest,
                )
                reference = ref.read(1, window=window, boundless=True, fill_value=0)
                observation = (~np.isin(scl, invalid_classes) & (reference != 0)).astype("uint8")
                supervision = (target.astype(bool) & observation.astype(bool)).astype("uint8")

                scl_positive_counts += np.bincount(scl[target == 1], minlength=12)[:12]
                total_positive += int(target.sum())
                total_supervised_positive += int(supervision.sum())

                relative = Path(cfg["output_root"]) / "positive_masks" / f"{row.patch_id}.tif"
                output = root / relative
                profile = {
                    "driver": "GTiff",
                    "width": row.width,
                    "height": row.height,
                    "count": 3,
                    "dtype": "uint8",
                    "crs": ref.crs,
                    "transform": transform,
                    "compress": "DEFLATE",
                    "predictor": 2,
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                    "nodata": None,
                }
                with rasterio.open(output, "w", **profile) as dst:
                    dst.write(target, 1)
                    dst.write(observation, 2)
                    dst.write(supervision, 3)
                    dst.set_band_description(1, "landslide_target")
                    dst.set_band_description(2, "valid_observation")
                    dst.set_band_description(3, "supervised_pixels_positive_only")
                    dst.update_tags(
                        background_policy=cfg["background_policy"],
                        scene=scene,
                        patch_id=row.patch_id,
                    )
                mask_paths[row.patch_id] = relative.as_posix()

    manifest["mask_path"] = manifest["patch_id"].map(mask_paths)
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "mask_files": len(mask_paths),
        "landslide_pixels": total_positive,
        "supervised_positive_pixels": total_supervised_positive,
        "positive_observation_fraction": total_supervised_positive / total_positive,
        "positive_pixels_by_scl_class": {
            str(i): int(value) for i, value in enumerate(scl_positive_counts) if value
        },
        "pixel_count_mismatches": mismatched_pixel_counts,
        "background_supervised_pixels": 0,
        "background_policy": cfg["background_policy"],
        "passed": len(mask_paths) == len(manifest) and not mismatched_pixel_counts,
    }
    summary_path = root / cfg["output_root"] / "manifests" / "positive_masks_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
