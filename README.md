# GeoBASP

Code for **GeoBASP: Geography-Aware Structured Pruning for Landslide-Scar
Segmentation in the Nepal Himalaya**, by Deepak Ghimire, accepted in *Electronics*.
The article DOI will be added when assigned.

GeoBASP combines task sensitivity and training-block aggregation for
dependency-aware channel pruning. The study compares ResNet-50 U-Net,
ConvNeXt-V2-Tiny U-Net and SegFormer-B2 at 25%, 50% and 70% MAC reduction.
Magnitude and Taylor cover all three architectures; the post-training SIRFP
adaptation covers the two U-Nets.

## Quick Start

Run from the repository root in your existing Python environment. Dependencies
are listed in `requirements.txt`; training and pruning require CUDA.

**1. Prepare the data.** Download the original ICIMOD inventory from the
[v1.0.0 release](https://github.com/ghimiredhikura/GeoBASP/releases/tag/v1.0.0)
and the recorded Sentinel-2/DEM inputs from their public sources. The downloader
checks source hashes before masks and terrain inputs are regenerated:

```bash
python main.py data download
python main.py data prepare
```

**2. Use the main checkpoints, or train a dense model.** Six checkpoints are
distributed as individual GitHub Release assets:
three dense architectures and the ConvNeXt 25% GeoBASP/Magnitude/Taylor models.
Training is optional when using those checkpoints.
All six files omit optimizer, scheduler and scaler state;
they retain learned weights and the metadata needed to reconstruct each network.

```bash
python main.py models --tag v1.0.0
python main.py train --profile convnextv2_tiny
```

The other training profiles are `resnet50` and `segformer_b2`. New checkpoints
are written to `generated/dense_models/MODEL/best.pt`.

**3. Reproduce pruning experiments.** Preview a study before launching it:

```bash
python main.py experiments main --plan
python main.py experiments main
python main.py experiments sirfp
```

`main` runs progressive pruning and 30 recovery epochs per target across three
seeds. To use freshly trained checkpoints, add
`--dense-root generated/dense_models`. The compact studies are `components`,
`calibration` and `feature_guard`; their settings share one experiment catalog.

**4. Validate, then test.** Select the probability threshold using validation
data and apply it unchanged to the test split. Dense and pruned checkpoints
are both supported:

```bash
python main.py threshold --checkpoint models/convnextv2_tiny/geobasp_mac25_seed2028.pt --step 0.005 --output generated/threshold.json
python main.py evaluate --checkpoint models/convnextv2_tiny/geobasp_mac25_seed2028.pt --threshold-report generated/threshold.json --split test --output generated/test.json
```

For full evaluation, aggregation and latency commands, see
[Reproducing the study](docs/REPRODUCING.md).

## Repository contents

| Path | Contents |
| --- | --- |
| `main.py` | Common entry point for the commands above |
| `src/nepal_cnn/` | Code organized by purpose, as outlined below |
| `configs/` | Dense profiles and compact experiment protocols |
| `metadata/` | Source references, checkpoint hashes and fixed spatial definitions |
| `docs/` | Reproduction details, attribution and licensing |

### Code organization

Start with `main.py`, which dispatches commands through `cli/`. Within
`src/nepal_cnn/`, each folder follows one part of the workflow:

| Folder | Purpose and main files |
| --- | --- |
| [data/](src/nepal_cnn/data/) | Download and prepare inputs; load patches with `dataset.py` |
| [models/](src/nepal_cnn/models/) | Networks in `architectures.py`, checkpoint loading and release assets |
| [training/](src/nepal_cnn/training/) | Dense training in `train.py`, recovery training and losses |
| [pruning/](src/nepal_cnn/pruning/) | GeoBASP in `geobasp.py`, Magnitude/Taylor in `baseline.py`, and SIRFP |
| [evaluation/](src/nepal_cnn/evaluation/) | Metrics, threshold selection, test evaluation and latency |
| [experiments/](src/nepal_cnn/experiments/) | Experiment runners, ablations and summary generation |
| [cli/](src/nepal_cnn/cli/) | Main command dispatch and pruning commands |
| [utils/](src/nepal_cnn/utils/) | Shared paths, configuration and logging |

Processed arrays, figures, predictions, logs and result tables are regenerated
locally and excluded from Git. Raw data and model weights are separate release
assets. Fixed split and normalization definitions are retained to reproduce the
study's inputs; they are not generated experiment results.

The [v1.0.0 release](https://github.com/ghimiredhikura/GeoBASP/releases/tag/v1.0.0)
provides the seven original inventory files, six main checkpoints, catalogs,
checksums and upstream license notices. `data download` restores the raw inventory
into `data/raw/icimod/`; `models` installs the checkpoints into `models/`.
Other pruning targets, seeds and SIRFP checkpoints can be regenerated with the
supplied experiment commands. No ZIP extraction is required.

Original code is [MIT licensed](LICENSE); the inventory is CC BY 4.0. Pretrained
model terms remain separate; see [licensing and attribution](docs/LICENSES.md).
Please cite the article and the [ICIMOD inventory](https://doi.org/10.26066/rds.1973522).
