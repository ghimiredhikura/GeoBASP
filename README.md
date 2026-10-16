# GeoBASP

Code for **GeoBASP: Geography-Aware Structured Pruning for Landslide-Scar
Segmentation in the Nepal Himalaya**, by Deepak Ghimire, accepted in *Electronics*.
The article DOI will be added when assigned.

GeoBASP combines task sensitivity and training-block aggregation for
dependency-aware channel pruning. The study compares ResNet-50 U-Net,
ConvNeXt-V2-Tiny U-Net and SegFormer-B2 at 25%, 50% and 70% MAC reduction.
Magnitude and Taylor cover all three architectures; the post-training SIRFP
adaptation covers the two U-Nets.

## Sample results

![Validation Dice versus MAC reduction and GeoBASP's mean differences](docs/assets/pruning_validation_frontier.png)

**Pruning comparison (Figure 4).** Panel (a) compares the two U-Nets using
the same six runs per method; (b) shows ConvNeXt-V2-Tiny; (c) shows
GeoBASP-minus-method Dice over shared architectures: three for
Magnitude/Taylor and two for SIRFP. Error bars show sample SD across runs
(a,b), dense architecture references (a), and architecture differences (c).

**Selected operating point (Table 4, selected columns).** ConvNeXt-V2-Tiny
U-Net near 25% MAC reduction:

| Method | MAC reduction (%) | Params (M) | Validation Dice | Test Dice | NI | Latency (ms) |
| --- | ---: | ---: | ---: | ---: | :---: | ---: |
| Dense | 0 | 34.957 | 0.4873 | 0.5159 | -- | 31.31 ± 0.13 |
| Magnitude | 24.09 | 29.390 | 0.3058 | 0.3479 | No | **24.58 ± 0.04** |
| Taylor | 25.80 | 28.409 | 0.4620 | 0.4740 | No | 24.85 ± 0.02 |
| GeoBASP (Ours) | 24.22 | 30.648 | **0.4848** | 0.5054 | **Yes** | 25.58 ± 0.04 |
| SIRFP | 24.31 | **25.499** | 0.4580 | **0.5087** | No | 26.43 ± 0.18 |

Segmentation values are single selected checkpoints; bold marks the best
pruned value. NI denotes non-inferiority to the dense model using paired
20 km-block 95% intervals and a −0.02 Dice margin. Latency is CUDA Graph
FP32 mean ± SD of three session medians on an RTX 5060 Laptop GPU. GeoBASP
reduces MACs by 24.2% and latency by 18.3% while passing this criterion;
the higher global test Dice of SIRFP alone does not establish non-inferiority.

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

Apart from the sample plot and table above, experiment outputs are regenerated
locally: processed arrays, predictions, figures, logs and machine-readable result
tables are excluded from Git. Raw data and model weights are separate release
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
