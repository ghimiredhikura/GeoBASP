"""Conventional dense segmentation baseline used before structured pruning."""

from __future__ import annotations

import json
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    ResNet34_Weights,
    ResNet50_Weights,
    resnet34,
    resnet50,
)

from nepal_cnn.utils.paths import PROJECT_ROOT


def adapt_input_convolution(convolution: nn.Conv2d, input_channels: int) -> nn.Conv2d:
    """Expand a pretrained input convolution while preserving its RGB function."""
    if input_channels < 1:
        raise ValueError("input_channels must be positive")
    if convolution.in_channels == input_channels:
        return convolution
    if convolution.groups != 1:
        raise ValueError("Input adaptation currently requires an ungrouped convolution")
    adapted = nn.Conv2d(
        input_channels,
        convolution.out_channels,
        kernel_size=convolution.kernel_size,
        stride=convolution.stride,
        padding=convolution.padding,
        dilation=convolution.dilation,
        groups=convolution.groups,
        bias=convolution.bias is not None,
        padding_mode=convolution.padding_mode,
    )
    with torch.no_grad():
        adapted.weight.zero_()
        copied = min(convolution.in_channels, input_channels)
        adapted.weight[:, :copied].copy_(convolution.weight[:, :copied])
        if adapted.bias is not None and convolution.bias is not None:
            adapted.bias.copy_(convolution.bias)
    return adapted


def segformer_input_projection(network: nn.Module) -> nn.Conv2d:
    """Return the first SegFormer projection across Transformers layouts."""
    backbone = network.segformer
    if hasattr(backbone, "encoder"):
        return backbone.encoder.patch_embeddings[0].proj
    if hasattr(backbone, "stages"):
        return backbone.stages[0].patch_embeddings.proj
    raise RuntimeError("Unsupported Transformers SegFormer module layout")


def adapt_segformer_input(network: nn.Module, input_channels: int) -> None:
    """Replace the first SegFormer projection across Transformers layouts."""
    adapted = adapt_input_convolution(
        segformer_input_projection(network), input_channels
    )
    backbone = network.segformer
    if hasattr(backbone, "encoder"):
        backbone.encoder.patch_embeddings[0].proj = adapted
    else:
        backbone.stages[0].patch_embeddings.proj = adapted


def _dynamic_segformer_sequence_reduction_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Equivalent dense forward that remains valid after structural pruning."""
    batch_size, _, input_channels = hidden_states.shape
    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size, input_channels, height, width
    )
    hidden_states = module.sequence_reduction(hidden_states)
    output_channels = hidden_states.shape[1]
    hidden_states = hidden_states.reshape(batch_size, output_channels, -1).transpose(1, 2)
    return module.layer_norm(hidden_states)


def make_segformer_pruning_compatible(network: nn.Module) -> None:
    """Remove cached-shape assumptions without changing dense computations."""
    for module in network.modules():
        if type(module).__name__ == "SegformerSequenceReduction":
            module.forward = MethodType(
                _dynamic_segformer_sequence_reduction_forward, module
            )


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is None:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.activation(self.bn1(self.conv1(x)))
        return self.dropout(self.activation(self.bn2(self.conv2(x))))


class UNetResNet34(nn.Module):
    """Standard ResNet-34 encoder U-Net; no proposed architecture components."""

    def __init__(self, pretrained: bool = True, input_channels: int = 3):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        encoder = resnet34(weights=weights)
        encoder.conv1 = adapt_input_convolution(encoder.conv1, input_channels)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.encoder1 = encoder.layer1
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4

        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 32)
        self.decoder0 = DecoderBlock(32, 0, 16)
        self.segmentation_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        stem = self.stem(x)
        enc1 = self.encoder1(self.pool(stem))
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        x = self.decoder4(enc4, enc3)
        x = self.decoder3(x, enc2)
        x = self.decoder2(x, enc1)
        x = self.decoder1(x, stem)
        x = self.decoder0(x)
        x = self.segmentation_head(x)
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x

    def encoder_modules(self) -> list[nn.Module]:
        return [self.stem, self.encoder1, self.encoder2, self.encoder3, self.encoder4]


class UNetResNet50(nn.Module):
    """Higher-capacity conventional ResNet-50 U-Net dense baseline."""

    def __init__(
        self,
        pretrained: bool = True,
        decoder_dropout: float = 0.10,
        input_channels: int = 3,
    ):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        encoder = resnet50(weights=weights)
        encoder.conv1 = adapt_input_convolution(encoder.conv1, input_channels)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.encoder1 = encoder.layer1
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4

        self.decoder4 = DecoderBlock(2048, 1024, 512, decoder_dropout)
        self.decoder3 = DecoderBlock(512, 512, 256, decoder_dropout)
        self.decoder2 = DecoderBlock(256, 256, 128, decoder_dropout)
        self.decoder1 = DecoderBlock(128, 64, 64, decoder_dropout)
        self.decoder0 = DecoderBlock(64, 0, 32, decoder_dropout)
        self.segmentation_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        stem = self.stem(x)
        enc1 = self.encoder1(self.pool(stem))
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        x = self.decoder4(enc4, enc3)
        x = self.decoder3(x, enc2)
        x = self.decoder2(x, enc1)
        x = self.decoder1(x, stem)
        x = self.decoder0(x)
        x = self.segmentation_head(x)
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x

    def encoder_modules(self) -> list[nn.Module]:
        return [self.stem, self.encoder1, self.encoder2, self.encoder3, self.encoder4]


class UNetConvNeXtV2Tiny(nn.Module):
    """ConvNeXt-V2 Tiny multiscale encoder with a U-Net-style decoder."""

    backbone_name = "convnextv2_tiny.fcmae_ft_in1k"

    def __init__(
        self,
        pretrained: bool = True,
        decoder_dropout: float = 0.10,
        input_channels: int = 3,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise RuntimeError("UNetConvNeXtV2Tiny requires timm") from error
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=pretrained,
            in_chans=3,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        if input_channels != 3:
            self.backbone.stem_0 = adapt_input_convolution(
                self.backbone.stem_0, input_channels
            )
        channels = self.backbone.feature_info.channels()
        if list(channels) != [96, 192, 384, 768]:
            raise RuntimeError(f"Unexpected ConvNeXt-V2 feature channels: {channels}")
        self.decoder3 = DecoderBlock(768, 384, 384, decoder_dropout)
        self.decoder2 = DecoderBlock(384, 192, 192, decoder_dropout)
        self.decoder1 = DecoderBlock(192, 96, 96, decoder_dropout)
        self.decoder0 = DecoderBlock(96, 0, 64, decoder_dropout)
        self.decoder_full = DecoderBlock(64, 0, 32, decoder_dropout)
        self.segmentation_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        feature1, feature2, feature3, feature4 = self.backbone(x)
        x = self.decoder3(feature4, feature3)
        x = self.decoder2(x, feature2)
        x = self.decoder1(x, feature1)
        x = self.decoder0(x)
        x = self.decoder_full(x)
        x = self.segmentation_head(x)
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x

    def encoder_modules(self) -> list[nn.Module]:
        return [self.backbone]


class SegFormerB2Binary(nn.Module):
    """SegFormer-B2 with its native multiscale decoder and a binary head."""

    pretrained_model_name = "nvidia/segformer-b2-finetuned-ade-512-512"

    def __init__(self, pretrained: bool = True, input_channels: int = 3):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as error:
            raise RuntimeError("SegFormerB2Binary requires transformers") from error
        if pretrained:
            self.network = SegformerForSemanticSegmentation.from_pretrained(
                self.pretrained_model_name,
                num_labels=1,
                id2label={0: "landslide"},
                label2id={"landslide": 0},
                ignore_mismatched_sizes=True,
            )
            adapt_segformer_input(self.network, input_channels)
        else:
            config_path = PROJECT_ROOT / "configs/backbones/segformer_b2.json"
            config = SegformerConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
            config.num_channels = input_channels
            config.num_labels = 1
            config.id2label = {0: "landslide"}
            config.label2id = {"landslide": 0}
            self.network = SegformerForSemanticSegmentation(config)
        make_segformer_pruning_compatible(self.network)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        logits = self.network(pixel_values=x, return_dict=True).logits
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

    def encoder_modules(self) -> list[nn.Module]:
        return [self.network.segformer]


def build_model(
    architecture: str = "unet_resnet34",
    pretrained: bool = True,
    decoder_dropout: float = 0.0,
    input_channels: int = 3,
) -> nn.Module:
    if architecture == "unet_resnet34":
        return UNetResNet34(pretrained=pretrained, input_channels=input_channels)
    if architecture == "unet_resnet50":
        return UNetResNet50(
            pretrained=pretrained,
            decoder_dropout=decoder_dropout,
            input_channels=input_channels,
        )
    if architecture == "unet_convnextv2_tiny":
        return UNetConvNeXtV2Tiny(
            pretrained=pretrained,
            decoder_dropout=decoder_dropout,
            input_channels=input_channels,
        )
    if architecture == "segformer_b2":
        return SegFormerB2Binary(
            pretrained=pretrained, input_channels=input_channels
        )
    raise ValueError(f"Unknown architecture: {architecture}")
