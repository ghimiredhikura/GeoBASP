"""Materialize previously selected negative windows without sampling new rows."""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.windows import Window, bounds as window_bounds, from_bounds

from .reconstructed_masks import read_dem


def read_resampled(
    source: rasterio.DatasetReader,
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> np.ndarray:
    source_window = from_bounds(*bounds, transform=source.transform)
    return source.read(
        1,
        window=source_window,
        out_shape=shape,
        boundless=True,
        fill_value=0,
        resampling=Resampling.bilinear,
    ).astype(np.float32)


def patch_arrays(
    row: dict,
    reference: rasterio.DatasetReader,
    scl_source: rasterio.DatasetReader,
    b08_source: rasterio.DatasetReader,
    b11_source: rasterio.DatasetReader,
    dem_sources: list[rasterio.DatasetReader],
    basin_geometry,
    invalid_scl: np.ndarray,
) -> tuple[np.ndarray, dict]:
    window = Window(row["col_off"], row["row_off"], row["width"], row["height"])
    transform = reference.window_transform(window)
    bounds = window_bounds(window, reference.transform)
    shape = (row["height"], row["width"])
    b04 = reference.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
    b08 = read_resampled(b08_source, bounds, shape)
    b11 = read_resampled(b11_source, bounds, shape)
    scl_window = from_bounds(*bounds, transform=scl_source.transform)
    scl = scl_source.read(
        1,
        window=scl_window,
        out_shape=shape,
        boundless=True,
        fill_value=0,
        resampling=Resampling.nearest,
    )
    observation = (~np.isin(scl, invalid_scl) & (b04 != 0))
    inside_basin = rasterize(
        [(basin_geometry, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)
    dem = read_dem(dem_sources, transform, reference.crs, *shape)
    dem_valid = np.isfinite(dem)
    primary = observation & inside_basin & dem_valid & (dem < 4500.0)
    sensitivity = observation & inside_basin & dem_valid

    denominator = max(int(primary.sum()), 1)
    ndvi = (b08 - b04) / np.maximum(b08 + b04, 1.0)
    bare_proxy = primary & (ndvi < 0.25) & (b11 > b08)
    water_proxy = primary & (scl == 6)
    details = {
        "valid_scl_fraction": float((~np.isin(scl, invalid_scl)).mean()),
        "reference_nodata_fraction": float((b04 == 0).mean()),
        "primary_fraction": float(primary.mean()),
        "sensitivity_fraction": float(sensitivity.mean()),
        "bare_proxy_fraction": float(bare_proxy.sum() / denominator),
        "water_proxy_fraction": float(water_proxy.sum() / denominator),
        "hard_negative_score": float(
            bare_proxy.sum() / denominator + 0.5 * water_proxy.sum() / denominator
        ),
    }
    output = np.stack(
        [np.zeros(shape, dtype=bool), observation, primary, sensitivity]
    ).astype(np.uint8)
    return output, details
