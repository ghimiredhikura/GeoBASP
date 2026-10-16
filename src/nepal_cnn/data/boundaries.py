"""Download reproducible Koshi basin boundaries from ICIMOD's ArcGIS service."""

from __future__ import annotations

from nepal_cnn.utils.paths import PROJECT_ROOT as ROOT

import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import geopandas as gpd


SERVICE = (
    "https://geoapps.icimod.org/icimodarcgis/rest/services/"
    "RiverBasin/Koshi/MapServer"
)
LAYERS = {
    "icimod_koshi_basin": 9,
    "icimod_koshi_country_parts": 11,
}


def download_layer(name: str, layer_id: int, output_dir: Path) -> dict:
    query = urlencode(
        {
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }
    )
    url = f"{SERVICE}/{layer_id}/query?{query}"
    request = Request(url, headers={"User-Agent": "Nepal-CNN-Electronics/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    parsed = json.loads(payload)
    if parsed.get("type") != "FeatureCollection" or not parsed.get("features"):
        raise RuntimeError(f"ICIMOD layer {layer_id} did not return a feature collection")
    path = output_dir / f"{name}.geojson"
    path.write_bytes(payload)
    return {
        "name": name,
        "layer_id": layer_id,
        "url": url,
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "feature_count": len(parsed["features"]),
    }


def main() -> None:
    raw_dir = ROOT / "data" / "raw" / "boundaries"
    processed_dir = ROOT / "data" / "processed" / "boundaries"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    records = [
        download_layer(name, layer_id, raw_dir) for name, layer_id in LAYERS.items()
    ]
    basin = gpd.read_file(raw_dir / "icimod_koshi_basin.geojson").to_crs(32645)
    basin.geometry = basin.geometry.make_valid()
    countries = gpd.read_file(raw_dir / "icimod_koshi_country_parts.geojson").to_crs(32645)
    country_field = next(
        (column for column in countries.columns if column.lower() == "country"), None
    )
    if country_field is None:
        raise RuntimeError("ICIMOD country layer lacks its Country attribute")
    nepal = countries[countries[country_field].str.casefold() == "nepal"].copy()
    if len(nepal) != 1:
        raise RuntimeError(f"Expected one Nepal basin fragment, found {len(nepal)}")
    nepal.geometry = nepal.geometry.make_valid()
    nepal = gpd.overlay(nepal, basin[["geometry"]], how="intersection", keep_geom_type=True)
    basin_output = processed_dir / "icimod_koshi_basin.gpkg"
    nepal_output = processed_dir / "icimod_koshi_nepal.gpkg"
    basin.to_file(basin_output, layer="koshi_basin", driver="GPKG")
    nepal.to_file(nepal_output, layer="koshi_nepal", driver="GPKG")

    report = {
        "source_service": SERVICE,
        "license_note": "Separate boundary service; the inventory license does not establish this layer's redistribution terms.",
        "downloads": records,
        "processed_crs": str(nepal.crs),
        "processed": {
            "full_basin": {
                "path": basin_output.relative_to(ROOT).as_posix(),
                "features": len(basin),
                "area_km2": float(basin.geometry.area.sum() / 1e6),
                "bounds": [float(value) for value in basin.total_bounds],
            },
            "nepal_fragment": {
                "path": nepal_output.relative_to(ROOT).as_posix(),
                "features": len(nepal),
                "area_km2": float(nepal.geometry.area.sum() / 1e6),
                "bounds": [float(value) for value in nepal.total_bounds],
            },
        },
    }
    report_path = processed_dir / "icimod_boundary_manifest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
