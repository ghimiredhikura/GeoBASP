"""Build conservative background supervision without altering positive-only masks."""

from __future__ import annotations

from nepal_cnn.utils.paths import PROJECT_ROOT as ROOT

import json
from contextlib import ExitStack
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, transform as window_transform


PRIMARY_COLUMN = "reconstructed_mask_path"


def intersects(left: tuple[float, float, float, float], right: rasterio.coords.BoundingBox) -> bool:
    return not (
        left[2] <= right.left
        or left[0] >= right.right
        or left[3] <= right.bottom
        or left[1] >= right.top
    )


def read_dem(
    sources: list[rasterio.DatasetReader],
    destination_transform: rasterio.Affine,
    destination_crs: rasterio.CRS,
    height: int,
    width: int,
) -> np.ndarray:
    destination_bounds = rasterio.transform.array_bounds(
        height, width, destination_transform
    )
    geographic_bounds = transform_bounds(
        destination_crs, "EPSG:4326", *destination_bounds, densify_pts=21
    )
    dem = np.full((height, width), np.nan, dtype=np.float32)
    matched = 0
    for source in sources:
        if not intersects(geographic_bounds, source.bounds):
            continue
        matched += 1
        tile = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(source, 1),
            destination=tile,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=destination_transform,
            dst_crs=destination_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            init_dest_nodata=True,
            num_threads=2,
        )
        valid = np.isfinite(tile)
        dem[valid] = tile[valid]
    if matched == 0:
        raise RuntimeError(f"No DEM tile intersects patch bounds {geographic_bounds}")
    return dem


def update_manifest(path: Path) -> None:
    frame = pd.read_csv(path)
    frame[PRIMARY_COLUMN] = frame["patch_id"].map(
        lambda patch_id: f"data/processed/reconstructed_masks/{patch_id}.tif"
    )
    frame.to_csv(path, index=False)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    manifest_path = ROOT / cfg["output_root"] / "manifests" / "positive_patches.csv"
    manifest = pd.read_csv(manifest_path)
    basin = gpd.read_file(ROOT / cfg["candidate_koshi_boundary"]).to_crs(
        cfg["target_crs"]
    )
    basin_geometry = basin.geometry.union_all()
    dem_paths = sorted((ROOT / cfg["dem_root"]).glob("Copernicus_DSM*_DEM.tif"))
    if not dem_paths:
        raise RuntimeError("No Copernicus DEM tiles found")

    output_dir = ROOT / cfg["output_root"] / "reconstructed_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    totals = {
        "mask_files": 0,
        "landslide_pixels": 0,
        "primary_supervised_pixels": 0,
        "primary_supervised_positive_pixels": 0,
        "primary_supervised_background_pixels": 0,
        "sensitivity_supervised_pixels": 0,
        "sensitivity_supervised_background_pixels": 0,
        "positive_pixels_outside_full_basin": 0,
        "positive_pixels_at_or_above_4500m": 0,
        "positive_pixels_without_dem": 0,
    }

    with ExitStack() as stack:
        dem_sources = [stack.enter_context(rasterio.open(path)) for path in dem_paths]
        for row in manifest.itertuples(index=False):
            scene_dir = ROOT / cfg["sentinel_root"] / row.scene
            window = Window(int(row.col_off), int(row.row_off), int(row.width), int(row.height))
            with rasterio.open(scene_dir / f"{cfg['reference_band']}.tif") as reference:
                transform = window_transform(window, reference.transform)
                crs = reference.crs
            with rasterio.open(ROOT / row.mask_path) as source_mask:
                original = source_mask.read()
            target = original[0].astype(bool)
            observation = original[1].astype(bool)
            height, width = target.shape

            inside_basin = rasterize(
                [(basin_geometry, 1)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                all_touched=False,
                dtype="uint8",
            ).astype(bool)
            dem = read_dem(dem_sources, transform, crs, height, width)
            dem_valid = np.isfinite(dem)
            below_4500 = dem_valid & (dem < 4500.0)

            primary_domain = (inside_basin & below_4500) | target
            sensitivity_domain = (inside_basin & dem_valid) | target
            primary_supervision = observation & primary_domain
            sensitivity_supervision = observation & sensitivity_domain
            output = np.stack(
                [target, observation, primary_supervision, sensitivity_supervision]
            ).astype(np.uint8)

            profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 4,
                "dtype": "uint8",
                "crs": crs,
                "transform": transform,
                "compress": "DEFLATE",
                "predictor": 2,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "nodata": None,
            }
            output_path = output_dir / f"{row.patch_id}.tif"
            with rasterio.open(output_path, "w", **profile) as destination:
                destination.write(output)
                destination.set_band_description(1, "landslide_target")
                destination.set_band_description(2, "valid_observation")
                destination.set_band_description(3, "reconstructed_primary_supervision")
                destination.set_band_description(4, "no_elevation_threshold_sensitivity")
                destination.update_tags(
                    background_policy=(
                        "scene_union_full_koshi_below_4500_positive_override"
                    ),
                    scene=row.scene,
                    patch_id=row.patch_id,
                )

            totals["mask_files"] += 1
            totals["landslide_pixels"] += int(target.sum())
            totals["primary_supervised_pixels"] += int(primary_supervision.sum())
            totals["primary_supervised_positive_pixels"] += int(
                (primary_supervision & target).sum()
            )
            totals["primary_supervised_background_pixels"] += int(
                (primary_supervision & ~target).sum()
            )
            totals["sensitivity_supervised_pixels"] += int(sensitivity_supervision.sum())
            totals["sensitivity_supervised_background_pixels"] += int(
                (sensitivity_supervision & ~target).sum()
            )
            totals["positive_pixels_outside_full_basin"] += int(
                (target & ~inside_basin).sum()
            )
            totals["positive_pixels_at_or_above_4500m"] += int(
                (target & dem_valid & (dem >= 4500.0)).sum()
            )
            totals["positive_pixels_without_dem"] += int((target & ~dem_valid).sum())

    if totals["mask_files"] != len(manifest):
        raise RuntimeError("Not every manifest row produced a reconstructed mask")
    if totals["landslide_pixels"] <= 0 or totals["primary_supervised_background_pixels"] <= 0:
        raise RuntimeError("Reconstructed masks do not contain both target and background")

    update_manifest(manifest_path)
    update_manifest(ROOT / cfg["output_root"] / "splits" / "primary_patch_split.csv")
    summary = {
        **totals,
        "background_supervised_pixels": totals["primary_supervised_background_pixels"],
        "background_policy": (
            "reconstructed_scene_union_full_koshi_below_4500_positive_override"
        ),
        "primary_domain": (
            "valid observation AND ((full ICIMOD Koshi basin AND Copernicus DEM <4500m) "
            "OR released landslide target), evaluated only in referenced scene patches"
        ),
        "sensitivity_domain": (
            "same as primary but without the elevation threshold"
        ),
        "passed": True,
    }
    summary_path = ROOT / cfg["output_root"] / "manifests" / "reconstructed_masks_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
