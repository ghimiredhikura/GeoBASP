"""Explicit spatial-layout and GRN invariants for the ConvNeXt SIRFP extension."""
import torch
from torch import nn


def spatial_linear_modules(model):
    modules = []
    for block in model.modules():
        if type(block).__name__ != 'GlobalResponseNormMlp':
            continue
        if isinstance(block.fc1, nn.Linear):
            if not isinstance(block.fc2, nn.Linear) or block.grn.channel_dim != -1:
                raise RuntimeError('Expected NHWC Linear ConvNeXt channel MLP')
            modules.extend((block.fc1, block.fc2))
    if not modules:
        raise RuntimeError('No NHWC ConvNeXt channel MLPs found')
    return modules


def output_width(module):
    if isinstance(module, nn.Conv2d):
        return module.out_channels
    if isinstance(module, nn.Linear):
        return module.out_features
    raise TypeError(type(module).__name__)


def protected_shapes(model):
    return {name: tuple(parameter.shape)
            for name, parameter in model.named_parameters()
            if name.startswith(('backbone.stem_0.', 'segmentation_head.'))}


def validate_structure(model, graphs=None, protected=None):
    if model.backbone.stem_0.in_channels != 4 or model.segmentation_head.out_channels != 1:
        raise RuntimeError('Protected four-channel projection / binary classifier changed')
    if protected is not None and protected_shapes(model) != protected:
        raise RuntimeError('Protected input/classifier parameter shapes changed')
    blocks = 0
    for name, block in model.named_modules():
        if type(block).__name__ != 'GlobalResponseNormMlp':
            continue
        expansion = output_width(block.fc1)
        contraction_input = block.fc2.in_features if isinstance(block.fc2, nn.Linear) else block.fc2.in_channels
        if expansion != contraction_input:
            raise RuntimeError(f'MLP hidden input/output coupling failed: {name}')
        if block.grn.weight.numel() != expansion or block.grn.bias.numel() != expansion:
            raise RuntimeError(f'GRN affine channels do not follow MLP pruning: {name}')
        blocks += 1
    if blocks != 18:
        raise RuntimeError(f'Expected 18 ConvNeXt-V2-Tiny channel MLPs, found {blocks}')
    if graphs is not None:
        names = {module: name for name, module in model.named_modules()}
        for module, graph in graphs.items():
            width = output_width(module)
            if graph.shape != (width, width) or not torch.isfinite(graph).all():
                raise RuntimeError(f'Graph/output mismatch at {names.get(module)}: {tuple(graph.shape)} versus {width}')
    return {'grn_mlp_blocks_checked': blocks, 'protected_endpoints_preserved': True,
            'coupled_grn_channels_consistent': True, 'graphs_follow_physical_removal': graphs is not None}
