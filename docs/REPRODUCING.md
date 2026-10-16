# Reproducing the study

Run commands from the repository root using the dependencies recorded in
`requirements.txt`. `main.py --help` lists workflows. Outputs are written to
ignored `generated/` and `data/processed/` directories.

## Data and dense training

```bash
python main.py data sources
python main.py data download --list
python main.py data download
python main.py data prepare --plan
python main.py data prepare
python main.py train --profile resnet50
python main.py train --profile convnextv2_tiny
python main.py train --profile segformer_b2
```

The original inventory is downloaded unchanged from the GitHub Release assets.
Imagery, DEM and boundary geometry are retrieved from recorded sources.
Processing restores the fixed patch membership,
20 km spatial splits, guard exclusions and training normalization from
`metadata/`; it regenerates masks and slope without selecting new patches.
Definitions alone can be restored with `data prepare --definitions-only`.
See [data sources](DATA_SOURCES.md) for attribution and exact scene references.

Dense settings are consolidated in `configs/dense.yaml`. `--seed`, `--epochs`
and `--output` are optional training overrides. Three main profiles use four
channels (B11, B08, B04 and slope). Dense training starts from the upstream
pretrained encoder; the selected dense checkpoint is shared by pruning runs.

## Supplied models and evaluation

Six inference checkpoints are distributed individually as GitHub Release
assets, outside Git. Dense checkpoints omit optimizer/scaler/scheduler state
but preserve every learned tensor. Pruned checkpoints retain structural history
and are replayed on their dense dependency graph before loading weights.
`metadata/models.json` records their paths, provenance and SHA-256 hashes.

```bash
python main.py models --list
python main.py models --tag v1.0.0
```

Local individual assets can also be installed using
`python main.py models --assets-dir PATH_TO_ASSETS`. No ZIP is needed.
For any checkpoint, select a threshold on validation and evaluate it unchanged:

```bash
python main.py threshold --checkpoint models/convnextv2_tiny/dense.pt --step 0.005 --output generated/dense_models/convnextv2_tiny/validation_threshold.json
python main.py evaluate --checkpoint models/convnextv2_tiny/dense.pt --threshold-report generated/dense_models/convnextv2_tiny/validation_threshold.json --split test --output generated/evaluation/dense/convnextv2_tiny_test.json
python main.py pruning final-test
```

The last command regenerates the frozen 25% GeoBASP/Magnitude/Taylor comparison,
including boundary/component metrics and paired block-bootstrap intervals.
It checks checkpoint hashes and requires the preceding dense test records.
Its completed-test guard prevents overwriting a finished evaluation.
Replace checkpoint/output paths to evaluate a newly trained model.

## Pruning and sensitivity experiments

Each training command below can be previewed with `--plan`. `--models` and
`--seeds` select a smaller subset where applicable. New dense checkpoints can
be used with `--dense-root generated/dense_models`; the default is the supplied
dense models. Use `--output generated/NAME` for a separate new execution.

| Paper evidence | Command | Scope |
| --- | --- | --- |
| Dense adaptation | `python main.py experiments dense_ablation` | ConvNeXt spectral/terrain and loss controls |
| Main pruning comparison | `python main.py experiments main` | Three architectures, three methods, three seeds |
| Main GeoBASP repeat | `python main.py experiments repeated` | GeoBASP at the same seeds and targets |
| Segmentation-oriented comparison | `python main.py experiments sirfp` | Two U-Nets, three seeds, 25/50/70% |
| Component table | `python main.py experiments components` | Seven ResNet variants, three recovery seeds, 25/50% |
| Calibration table | `python main.py experiments calibration` | Five ResNet conditions, three sampling seeds, 25/50% |
| Cross-architecture safeguard | `python main.py experiments feature_guard` | ConvNeXt with/without final-feature protection, three seeds |

Progressive targets are measured relative to the original dense MAC count;
each pruning stage is followed by 30 recovery epochs. Main and SIRFP calibration
use 24 positive and 8 hard-negative full training patches. SIRFP collects
unpooled spatial redundancy graphs after dense training and between recovery
stages; this is the declared adaptation, not the published training-time pipeline.
SegFormer is excluded from SIRFP because its spatial dependency mapping was not
validated. The component study fixes the same 32 calibration patches and protects
all 32 final decoder outputs. Calibration sensitivity varies membership with
recovery fixed at seed 2026; its sampling seeds are 9026, 9027 and 9028.

Study results contain validation-selected thresholds, every completed outcome,
checkpoint hashes, mean/sample SD and descriptive minima/maxima. Test access is
a separate command, using the saved thresholds without test tuning:

```bash
python main.py experiments components --test
python main.py experiments calibration --test
python main.py experiments feature_guard --test
python main.py experiments sirfp --models convnextv2_tiny --seeds 2028 --test --targets 0.25
python main.py summary components
python main.py summary calibration
python main.py summary feature_guard
python main.py summary sirfp
```

The SIRFP operating point refines its threshold on validation at a 0.005 step,
matching the frozen comparison. For other studies, saved recovery thresholds
are applied unchanged. All requested checkpoints are frozen before test access.

## Aggregation and selection

```bash
python main.py summary dense
python main.py summary pruning
```

Before `summary dense`, run `python main.py experiments dense_ablation --test`
to evaluate all nine dense adaptation checkpoints at their validation thresholds.

The manuscript's main GeoBASP summary retains the higher validation-Dice recovery
independently for each seed from the archived and repeated executions. Magnitude
and Taylor use the archived three-seed suite; SIRFP uses one completed trajectory
per seed. The pruning summary retains all candidates and selected provenance.
The revised component and calibration summaries use all three matched runs,
including weaker outcomes. Maxima are descriptive and may come from different
seeds for validation and test; they do not replace the reported means.
The MAC-utility flag is retained as implemented, without inferring an independent
utility benefit from a difference in Dice alone.

Aggregate architecture summaries must use equal architecture coverage.
SIRFP covers two U-Nets: report its available-model mean explicitly and compare
pooled curves on those same two architectures for every method. SegFormer's
SIRFP cells remain unreported. Fresh training may differ numerically from the
paper; matching settings does not guarantee bitwise reproducibility across GPUs
or software builds.

## Latency on another computer

No dataset is needed. Install the six model assets, ensure the GPU is idle,
then run the paper's fixed-shape CUDA Graph FP32 protocol:

```bash
python main.py benchmark --sessions 3 --warmup 50 --iterations 200
```

Each session uses a fresh process, batch 1, four channels and 512 x 512 inputs,
with TF32 disabled. Graph replay is checked against eager outputs before timing.
The report records hardware/software, raw forward times, allocator memory and
mean/sample SD of session medians, plus paired latency reductions against dense.
For the regenerated SIRFP model, append
`--checkpoint SIRFP=generated/sirfp/convnextv2_tiny/sirfp/seed_2028/mac_25/checkpoint.pt`.
`--backend eager` provides a separate reference measurement. Latency depends on
hardware; measurements from another computer do not reproduce the laptop's
absolute timings. TensorRT is not required for the paper's reported protocol.
