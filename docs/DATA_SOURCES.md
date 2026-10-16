# Data sources and attribution

The repository contains source references and fixed study definitions. The
original ICIMOD inventory and its metadata are individual GitHub Release assets.
Satellite imagery, DEM, basin geometry and processed arrays are not bundled.

## ICIMOD inventory

The unmodified *Landslide Inventory of Koshi Basin 2024*, its shapefile components
and original XML metadata are provided in the
[v1.0.0 release](https://github.com/ghimiredhikura/GeoBASP/releases/tag/v1.0.0).
`python main.py data download --group inventory` restores all seven files under
`data/raw/icimod/landslides_koshi_2024/` and verifies their SHA-256 hashes.
The original source is
[DOI 10.26066/rds.1973522](https://doi.org/10.26066/rds.1973522).
To restore the inventory manually from ICIMOD, download and unpack
`Landslides_Koshi_2024.shp`, `.shx`, `.dbf`, `.prj`, and `.cpg` into:

```text
data/raw/icimod/landslides_koshi_2024/data/
```

The [official record](https://rds.icimod.org/metadata/af73da0a-885b-459d-95ba-2ea0662a7e7c)
specifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), which permits
sharing and adaptation with attribution and identification of changes.
The supplied inventory is redistributed unchanged under these terms.
Cite ICIMOD (2026), the dataset title, and its DOI; retain the source metadata
and license link when sharing it.
The patch/split definitions in `metadata/` are study derivatives, not official
ICIMOD data products. Regenerated supervision is also study derived; the
reconstructed mapping footprint is not an official coverage layer.

## Sentinel-2 and Copernicus DEM

Use `python main.py data download` to retrieve the recorded public imagery/DEM
assets. `metadata/data_sources.json` retains exact scene IDs, band URLs,
byte counts, and SHA-256 hashes, including late-2024 and early-2025 imagery.
The helper verifies downloads and defaults to B04, B08, B11, SCL, scene metadata,
and DEM inputs. `--all-bands` additionally retrieves the recorded display bands.

Official access and terms: [Copernicus Data Space](https://dataspace.copernicus.eu/terms-and-conditions)
and [Copernicus DEM](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM).
The manifest downloader uses the recorded public mirrors; those URLs preserve
the exact study assets. If a URL or digest changes, stop and inspect the source.
Retain Sentinel acquisition-year and modification credits. DEM and slope
products require the source/modified-data attribution specified by Copernicus,
including DLR, Airbus, the European Union, and ESA. Slope is derived from GLO-30;
resampling does not turn it into an independently measured 10 m elevation layer.

## Basin boundaries

Preprocessing retrieves the recorded ICIMOD ArcGIS basin/country layers and
checks their original hashes. References are in `metadata/data_sources.json`.
These services are separate from the inventory; the inventory license does not
establish boundary redistribution rights. Geometry is not included here.

## Exact study definitions

Seven small CSV/JSON settings in `metadata/` retain selected positive/negative
windows, guard-aware spatial membership, and training normalization. These are
reproducibility inputs. `python main.py data prepare` restores them locally and
regenerates masks/slope without new negative sampling or split selection.
The spectral and terrain manifests have 991 train, 182 validation, 183 test,
and 169 excluded-guard rows. Frozen definitions remain unchanged after processing.
