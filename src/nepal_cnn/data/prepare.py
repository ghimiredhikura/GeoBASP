"""Regenerate arrays using the exact released patch definitions."""

import argparse
import hashlib
import json
from pathlib import Path

from nepal_cnn.utils.configuration import run_workflow
from nepal_cnn.utils.paths import PROJECT_ROOT as ROOT


def sources():
    return json.loads((ROOT / "metadata/data_sources.json").read_text(encoding="utf-8"))


def restore_definitions():
    paths = []
    for item in sources()["fixed_inputs"]:
        contents = (ROOT / item["source"]).read_bytes()
        if hashlib.sha256(contents).hexdigest() != item["sha256"]:
            raise ValueError("Released definition changed: " + item["source"])
        target = ROOT / item["output"]
        if target.exists() and target.read_bytes() != contents:
            raise ValueError("Existing definition differs; preserve it before preparing: " + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
        paths.append(target)
    return paths


def check_inventory():
    inventory = sources()["inventory"]
    directory = ROOT / inventory["target_directory"]
    for item in inventory["required_files"]:
        path = directory / item["name"]
        if not path.is_file():
            raise FileNotFoundError("Run python main.py data download --group inventory, or download " + inventory["doi"] + " and extract the inventory into " + str(directory))
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Inventory differs from the study input: " + item["name"])


def materialize_fixed_negatives():
    from contextlib import ExitStack
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import rasterio
    from rasterio.windows import Window
    import yaml
    from .negative_masks import patch_arrays
    cfg = yaml.safe_load((ROOT/'configs/data.yaml').read_text())
    rows = pd.read_csv(ROOT/'data/processed/manifests/negative_patches.csv')
    basin = gpd.read_file(ROOT/cfg['candidate_koshi_boundary']).to_crs(cfg['target_crs']).geometry.union_all()
    invalid = np.asarray(cfg['invalid_scl_classes'],dtype=np.uint8)
    with ExitStack() as stack:
        dem = [stack.enter_context(rasterio.open(p)) for p in sorted((ROOT/cfg['dem_root']).glob('*.tif'))]
        if not dem: raise FileNotFoundError('Download DEM tiles first')
        for row in rows.to_dict(orient='records'):
            target = ROOT/row['reconstructed_mask_path']
            target.parent.mkdir(parents=True,exist_ok=True)
            scene = ROOT/cfg['sentinel_root']/row['scene']
            with ExitStack() as scenes:
                sources = [scenes.enter_context(rasterio.open(scene/name)) for name in ['B04.tif','SCL.tif','B08.tif','B11.tif']]
                arrays,_ = patch_arrays(row,*sources,dem,basin,invalid)
                transform = sources[0].window_transform(Window(row['col_off'],row['row_off'],row['width'],row['height']))
                with rasterio.open(target,'w',driver='GTiff',height=row['height'],width=row['width'],count=4,dtype='uint8',crs=sources[0].crs,transform=transform,compress='DEFLATE',predictor=2,tiled=True,blockxsize=256,blockysize=256,nodata=None) as output:
                    output.write(arrays)
                    for i,name in enumerate(['landslide_target','valid_observation','reconstructed_primary_supervision','no_elevation_threshold_sensitivity'],1):output.set_band_description(i,name)
    print(f'Materialized {len(rows)} fixed negative patches; no negative samples were reselected.',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="Describe processing without downloads or writes")
    parser.add_argument("--definitions-only", action="store_true", help="Restore split/patch/normalization definitions without processing raw data")
    args = parser.parse_args()
    if args.plan:
        print("Download the original ICIMOD inventory and the recorded Sentinel-2/DEM inputs with data download.")
        print("Restore seven fixed patch/split/normalization definitions from metadata/.")
        print("Fetch source boundaries; rasterize positive and reconstructed masks.")
        print("Materialize the recorded negative windows and slope arrays; restore exact definition bytes.")
        print("No new negative selection, split selection, normalization fit, training, or inference.")
        return
    if args.definitions_only:
        print(f"Restored {len(restore_definitions())} fixed input definitions; no raw data accessed.")
        return
    check_inventory()
    paths = restore_definitions()
    frozen = {path: path.read_bytes() for path in paths}
    try:
        run_workflow("boundaries", [])
        for item in sources()["boundaries"]["downloads"]:
            if hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Boundary source changed since the study: " + item["name"])
        run_workflow("positive_masks", [])
        run_workflow("reconstructed_masks", [])
        materialize_fixed_negatives()
        run_workflow("slope", ["--source-manifest", "data/processed/splits/model_patch_split.csv",
                               "--output-manifest", "data/processed/splits/model_patch_split_terrain.csv"])
    finally:
        for path, contents in frozen.items():
            path.write_bytes(contents)
    print("Masks and slope arrays regenerated with the released inputs unchanged.")


if __name__ == "__main__":
    main()
