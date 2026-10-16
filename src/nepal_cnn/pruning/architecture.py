"""Architecture constraints shared by all structured-pruning methods."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch_pruning as tp

from nepal_cnn.models.architectures import segformer_input_projection


@dataclass
class PruningSpec:
    ignored_layers: list[nn.Module]
    ignored_layer_names: list[str]
    num_heads: dict[nn.Module, int]
    attention_modules: list[nn.Module]
    unwrapped_parameters: list[tuple[nn.Parameter, int]]
    round_to: int | None


def architecture_pruning_spec(model: nn.Module, architecture: str) -> PruningSpec:
    """Declare protected layers and attention constraints before pruning."""
    named_modules = dict(model.named_modules())
    num_heads: dict[nn.Module, int] = {}
    attentions: list[nn.Module] = []
    unwrapped_parameters: list[tuple[nn.Parameter, int]] = []

    if architecture in {"unet_resnet34", "unet_resnet50"}:
        ignored_names = ["stem.0", "segmentation_head"]
        ignored = [named_modules[name] for name in ignored_names]
        round_to = 8
    elif architecture == "unet_convnextv2_tiny":
        ignored_names = ["backbone.stem_0", "segmentation_head"]
        ignored = [named_modules[name] for name in ignored_names]
        for module in model.modules():
            if type(module).__name__ == "GlobalResponseNorm":
                for parameter in module.parameters(recurse=False):
                    unwrapped_parameters.append((parameter, parameter.ndim - 1))
        round_to = 8
    elif architecture == "segformer_b2":
        input_projection = segformer_input_projection(model.network)
        classifier = model.network.decode_head.classifier
        ignored = [input_projection, classifier]
        module_names = {id(module): name for name, module in model.named_modules()}
        ignored_names = [module_names[id(module)] for module in ignored]
        for module in model.modules():
            if type(module).__name__ != "SegformerAttention":
                continue
            attentions.append(module)
            heads = int(module.num_attention_heads)
            for projection_name in ("q_proj", "k_proj", "v_proj"):
                num_heads[getattr(module, projection_name)] = heads
        round_to = 8
    else:
        raise ValueError(f"No pruning specification for {architecture}")

    return PruningSpec(
        ignored_layers=ignored,
        ignored_layer_names=ignored_names,
        num_heads=num_heads,
        attention_modules=attentions,
        unwrapped_parameters=unwrapped_parameters,
        round_to=round_to,
    )


def refresh_architecture_after_pruning(spec: PruningSpec) -> None:
    """Refresh non-parameter attention metadata after physical channel removal."""
    for attention in spec.attention_modules:
        heads = int(attention.num_attention_heads)
        width = int(attention.q_proj.out_features)
        if width % heads:
            raise RuntimeError(
                f"Pruned attention width {width} is not divisible by {heads} heads"
            )
        if attention.k_proj.out_features != width or attention.v_proj.out_features != width:
            raise RuntimeError("Query, key, and value projections were pruned inconsistently")
        attention.head_dim = width // heads
        attention.scaling = attention.head_dim**-0.5


def build_meta_pruner(
    model: nn.Module,
    example: torch.Tensor,
    architecture: str,
    importance: tp.importance.Importance,
    pruning_ratio: float,
    iterative_steps: int = 1,
) -> tuple[tp.pruner.MetaPruner, PruningSpec]:
    """Build the common dependency-aware global/isomorphic pruner."""
    spec = architecture_pruning_spec(model, architecture)
    if getattr(model, "protect_final_features", False):
        final = model.decoder0.conv2 if architecture == "unet_resnet50" else model.decoder_full.conv2
        spec.ignored_layers.append(final)
        spec.ignored_layer_names.append("final_decoder_features")
    pruner = tp.pruner.MetaPruner(
        model=model,
        example_inputs=example,
        importance=importance,
        global_pruning=True,
        isomorphic=True,
        pruning_ratio=pruning_ratio,
        iterative_steps=iterative_steps,
        ignored_layers=spec.ignored_layers,
        round_to=spec.round_to,
        num_heads=spec.num_heads,
        unwrapped_parameters=spec.unwrapped_parameters,
        prune_num_heads=False,
        prune_head_dims=True,
    )
    return pruner, spec


def importance_for_method(method: str) -> tp.importance.Importance:
    """Return a baseline importance estimator with preregistered settings."""
    if method == "group_magnitude":
        return tp.importance.GroupMagnitudeImportance(
            p=2, group_reduction="mean", normalizer="mean"
        )
    if method == "group_taylor":
        return tp.importance.GroupTaylorImportance(
            group_reduction="mean", normalizer="mean", multivariable=False
        )
    raise ValueError(f"Unknown pruning method: {method}")
