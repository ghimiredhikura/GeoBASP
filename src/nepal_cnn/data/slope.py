"""Materialize Copernicus-DEM slope patches aligned to Sentinel-2 inputs."""

from __future__ import annotations

from nepal_cnn.utils.paths import PROJECT_ROOT as ROOT

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, transform as window_transform
from scipy import ndimage


def intersects(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return not (
        left[2] <= right[0]
        or left[0] >= right[2]
        or left[3] <= right[1]
        or left[1] >= right[3]
    )


def fill_missing(array: np.ndarray) -> np.ndarray:
    missing = ~np.isfinite(array)
    if not missing.any():
        return array
    if missing.all():
        raise RuntimeError("DEM reprojection produced no valid elevation samples")
    indices = ndimage.distance_transform_edt(
        missing, return_distances=False, return_indices=True
    )
    return array[tuple(indices)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/processed/splits/model_patch_split_terrain.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/terrain/slope"),
    )
    parser.add_argument("--halo-pixels", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project_root = (config_path.parent.parent / config.get("project_root", ".")).resolve()
    source_manifest = (
        project_root / args.source_manifest
        if args.source_manifest is not None
        else project_root / config["training_manifest"]
    )
    output_manifest = project_root / args.output_manifest
    output_dir = project_root / args.output_dir
    rows = pd.read_csv(source_manifest)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows.iloc[: args.limit].copy()
    else:
        rows = rows.copy()

    dem_paths = sorted((project_root / config["dem_root"]).glob("*.tif"))
    if not dem_paths:
        raise FileNotFoundError("No Copernicus DEM tiles were found")
    target_crs = str(config["target_crs"])
    halo = int(args.halo_pixels)
    if halo < 1:
        raise ValueError("At least one halo pixel is required for slope gradients")
    output_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    reused = 0
    min_slope = np.inf
    max_slope = -np.inf
    sum_slope = 0.0
    slope_pixels = 0

    with ExitStack() as stack:
        dem_sources = [stack.enter_context(rasterio.open(path)) for path in dem_paths]
        dem_bounds = [
            transform_bounds(
                source.crs,
                target_crs,
                *source.bounds,
                densify_pts=21,
            )
            for source in dem_sources
        ]
        reference_sources: dict[Path, rasterio.io.DatasetReader] = {}

        for row_index, row in rows.iterrows():
            scene_dir = project_root / config["sentinel_root"] / str(row.scene)
            reference_path = scene_dir / f"{config['reference_band']}.tif"
            if reference_path not in reference_sources:
                reference_sources[reference_path] = stack.enter_context(
                    rasterio.open(reference_path)
                )
            reference = reference_sources[reference_path]
            if str(reference.crs) != target_crs:
                raise RuntimeError(
                    f"Reference CRS {reference.crs} does not match {target_crs}"
                )
            height = int(row.height)
            width = int(row.width)
            window = Window(int(row.col_off), int(row.row_off), width, height)
            transform = window_transform(window, reference.transform)
            padded_transform = transform * Affine.translation(-halo, -halo)
            padded_height = height + 2 * halo
            padded_width = width + 2 * halo
            bounds = array_bounds(padded_height, padded_width, padded_transform)

            relative_path = args.output_dir / f"{row.patch_id}.npy"
            output_path = project_root / relative_path
            if output_path.exists() and not args.overwrite:
                slope = np.load(output_path, allow_pickle=False)
                if slope.shape != (height, width):
                    raise RuntimeError(
                        f"Cached slope shape mismatch for {output_path}: {slope.shape}"
                    )
                slope = slope.astype(np.float32)
                reused += 1
            else:
                elevation = np.full(
                    (padded_height, padded_width), np.nan, dtype=np.float32
                )
                intersecting = 0
                for source, source_bounds in zip(dem_sources, dem_bounds):
                    if not intersects(bounds, source_bounds):
                        continue
                    intersecting += 1
                    projected = np.full_like(elevation, np.nan)
                    reproject(
                        source=rasterio.band(source, 1),
                        destination=projected,
                        src_transform=source.transform,
                        src_crs=source.crs,
                        src_nodata=source.nodata,
                        dst_transform=padded_transform,
                        dst_crs=reference.crs,
                        dst_nodata=np.nan,
                        resampling=Resampling.bilinear,
                        num_threads=2,
                        init_dest_nodata=True,
                    )
                    valid = np.isfinite(projected)
                    elevation[valid] = projected[valid]
                if intersecting == 0:
                    raise RuntimeError(f"No DEM tile intersects patch {row.patch_id}")
                elevation = fill_missing(elevation)
                resolution_x = abs(float(padded_transform.a))
                resolution_y = abs(float(padded_transform.e))
                gradient_y, gradient_x = np.gradient(
                    elevation, resolution_y, resolution_x
                )
                slope = np.degrees(
                    np.arctan(np.hypot(gradient_x, gradient_y))
                )[halo:-halo, halo:-halo]
                slope = np.clip(slope, 0.0, 90.0).astype(np.float32)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, slope.astype(np.float16), allow_pickle=False)
                created += 1

            rows.loc[row_index, "slope_path"] = relative_path.as_posix()
            min_slope = min(min_slope, float(slope.min()))
            max_slope = max(max_slope, float(slope.max()))
            sum_slope += float(slope.sum(dtype=np.float64))
            slope_pixels += int(slope.size)
            if (row_index + 1) % 100 == 0 or row_index + 1 == len(rows):
                print(
                    json.dumps(
                        {
                            "processed": row_index + 1,
                            "total": len(rows),
                            "created": created,
                            "reused": reused,
                        }
                    ),
                    flush=True,
                )

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_manifest, index=False)
    summary = {
        "source_manifest": source_manifest.relative_to(project_root).as_posix(),
        "output_manifest": output_manifest.relative_to(project_root).as_posix(),
        "terrain_feature": "slope_degrees",
        "dem_sources": [path.relative_to(project_root).as_posix() for path in dem_paths],
        "patches": len(rows),
        "created": created,
        "reused": reused,
        "dtype": "float16",
        "halo_pixels": halo,
        "slope_minimum": float(min_slope),
        "slope_maximum": float(max_slope),
        "slope_mean_all_pixels": float(sum_slope / max(slope_pixels, 1)),
    }
    summary_path = output_manifest.with_name(output_manifest.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
