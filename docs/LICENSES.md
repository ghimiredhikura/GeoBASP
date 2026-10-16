# Licensing and attribution

The root [MIT license](../LICENSE) covers original GeoBASP code and documentation.
It does not relicense third-party datasets or pretrained weights.

## Data

The unmodified ICIMOD *Landslide Inventory of Koshi Basin 2024* is redistributed as release assets
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Credit ICIMOD (2026), the dataset title and
[DOI 10.26066/rds.1973522](https://doi.org/10.26066/rds.1973522).
Original source metadata accompanies the shapefile. Study patch/split definitions and
regenerated masks are modified study products, not official ICIMOD products;
retain attribution and identify modifications. Imagery, DEM and boundary
geometry are downloaded rather than bundled; their source terms are linked in
[DATA_SOURCES.md](DATA_SOURCES.md).

## Model assets

The six checkpoints are fine-tuned/pruned derivatives of pretrained encoders.
Their source identifiers, transformations and hashes are in
[metadata/models.json](../metadata/models.json).

| Encoder | Upstream terms / source |
| --- | --- |
| ConvNeXt-V2-Tiny | [CC BY-NC 4.0 model card](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in1k); retain noncommercial use and attribution to the ConvNeXt-V2 authors |
| ResNet-50 | TorchVision `ResNet50_Weights.IMAGENET1K_V2`; [TorchVision BSD notices](https://github.com/pytorch/vision/blob/main/LICENSE) are retained; upstream pretrained-data terms may also apply |
| SegFormer-B2 | The [NVIDIA/Hugging Face model card](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512) identifies [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer) as its original source; the NVIDIA Source Code License for SegFormer limits use to noncommercial research/evaluation |

ConvNeXt checkpoints retain CC BY-NC 4.0 terms. SegFormer's model card uses the
label `other` and points to the original NVIDIA release; the corresponding
NVIDIA research/evaluation terms and complete license text are retained here.
TorchVision's code license alone is not an assertion about all rights in
ImageNet-derived weights. This package does not grant a blanket MIT license
to pretrained weights or imply upstream endorsement.

Complete upstream texts are in [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt)
and accompany the release assets. The model catalog and release notices identify
the original encoders and the fine-tuning, four-channel adaptation and pruning.

## Dependencies and related methods

Torch-Pruning is an external MIT dependency pinned in `requirements.txt` to
the study commit; its source is not vendored. TorchVision, timm, Transformers
and geospatial dependencies retain their upstream licenses.
SIRFP is cited as Wu et al., AAAI 2025,
[DOI 10.1609/aaai.v39i8.32903](https://doi.org/10.1609/aaai.v39i8.32903), with the
[authors' reference repository](https://github.com/dywu98/SIRFP).
The released adaptation implements the paper's spatial redundancy and greedy
selection with this study's post-training collection and dependency grouping;
the authors' repository is not bundled or relicensed.
