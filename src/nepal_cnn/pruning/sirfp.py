"""Paper-based SIRFP redundancy and greedy selection for a shared DepGraph.

Reference: Wu et al., AAAI 2025, doi:10.1609/aaai.v39i8.32903,
Equations (3)-(4) and Algorithm 1. This is an explicitly adapted implementation,
not a reproduction of the authors' different default dot-product/Sobel code.
Probability conversion, collection phase and dependency grouping must be
disclosed in the experiment protocol. No spatial pooling is used here.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn
import torch_pruning as tp
from torch_pruning.pruner import function


@torch.no_grad()
def spatial_probabilities(feature: torch.Tensor) -> torch.Tensor:
    """Convert each unpooled channel to a spatial probability distribution.

    Use absolute activation magnitude, with spatial L1 normalization. This
    declared conversion handles signed pre-activation convolution outputs.
    A zero channel becomes uniform; no pixel is subsampled or pooled.
    """
    if feature.ndim != 4 or not torch.isfinite(feature).all():
        raise ValueError("Expected finite NCHW features")
    flat = feature.detach().float().flatten(2).abs()
    mass = flat.sum(-1, keepdim=True)
    return torch.where(mass > 0, flat / mass.clamp_min(1e-30),
                       torch.full_like(flat, 1 / flat.shape[-1]))


@torch.no_grad()
def pairwise_redundancy(probability: torch.Tensor, channel_tile=64, spatial_tile=2048):
    """Equation (3): ln(2) minus Jensen-Shannon divergence, averaged over N.

    Tile only the computation, preserving every H*W element. Output is C*C.
    Calculate off-diagonal pairs once and mirror them. Diagonal entries are
    zero because Algorithm 1 excludes self edges.
    """
    if probability.ndim != 3 or channel_tile < 1 or spatial_tile < 1:
        raise ValueError("Expected N*C*S probabilities and positive tile sizes")
    if not torch.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    if not torch.allclose(probability.sum(-1), torch.ones_like(probability.sum(-1)), atol=2e-5):
        raise ValueError("Each spatial distribution must sum to one")
    _, channels, pixels = probability.shape
    matrix = torch.zeros(channels, channels, device=probability.device, dtype=torch.float32)
    for sample in probability.float():
        entropy_term = torch.special.xlogy(sample, sample).sum(-1)
        for left in range(0, channels, channel_tile):
            p = sample[left:left + channel_tile]
            for right in range(left, channels, channel_tile):
                q = sample[right:right + channel_tile]
                mixed_entropy = torch.zeros(p.shape[0], q.shape[0], device=sample.device)
                for start in range(0, pixels, spatial_tile):
                    mix = (p[:, None, start:start + spatial_tile] +
                           q[None, :, start:start + spatial_tile]) * 0.5
                    mixed_entropy += torch.special.xlogy(mix, mix).sum(-1)
                divergence = ((entropy_term[left:left + p.shape[0], None] +
                               entropy_term[None, right:right + q.shape[0]]) * 0.5 - mixed_entropy)
                redundancy = (math.log(2) - divergence).clamp(0, math.log(2))
                matrix[left:left + p.shape[0], right:right + q.shape[0]] += redundancy
                if left != right:
                    matrix[right:right + q.shape[0], left:left + p.shape[0]] += redundancy.T
    matrix /= probability.shape[0]
    matrix.fill_diagonal_(0)
    return matrix.cpu()


def greedy_order(redundancy: torch.Tensor):
    """Delete greatest incident redundancy, updating the retained graph each time.

    For a fixed remaining cardinality, this is equivalent to Algorithm 1's
    least-edge-weight deletion with a_ij=1-r_ij. Return monotone negative
    deletion degrees so thresholding preserves the greedy removal prefix.
    Stable lowest-index tie breaking makes the declared adaptation repeatable.
    """
    matrix = redundancy.double().cpu().numpy().copy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected square redundancy matrix")
    if not np.isfinite(matrix).all() or (matrix < -1e-7).any():
        raise ValueError("Redundancy must be finite and nonnegative")
    if not np.allclose(matrix, matrix.T, atol=1e-6):
        raise ValueError("Redundancy must be symmetric")
    np.fill_diagonal(matrix, 0)
    alive = np.ones(len(matrix), dtype=bool)
    degrees = matrix.sum(1)
    scores = np.zeros(len(matrix), dtype=np.float64)
    order = []
    for _ in range(len(matrix)):
        index = int(np.argmax(np.where(alive, degrees, -np.inf)))
        scores[index] = -max(float(degrees[index]), 0.0)
        order.append(index)
        alive[index] = False
        degrees -= matrix[:, index]
        degrees[~alive] = 0
    return torch.from_numpy(scores).float(), order


class GraphCollector:
    """Capture full Conv2d and explicitly declared NHWC Linear spatial maps.

    Linear modules are opt-in: their layout must be established by the model's
    implementation. No token grid is inferred and no spatial pooling is used.
    The default remains the original ResNet Conv2d-only collector.
    """

    def __init__(self, model: nn.Module, ignored_layers=(), alpha=0.99,
                 channel_tile=64, spatial_tile=2048, linear_modules=()):
        if not 0 <= alpha < 1:
            raise ValueError("EMA alpha must lie in [0, 1)")
        self.alpha = alpha
        self.graphs = {}
        self.updates = {}
        self.seconds = {}
        self.handles = []
        ignored = set(ignored_layers)
        self.names = {module: name for name, module in model.named_modules()}
        self.channel_tile, self.spatial_tile = channel_tile, spatial_tile
        self.linear_modules = set(linear_modules)
        if any(not isinstance(module, nn.Linear) for module in self.linear_modules):
            raise TypeError("Declared NHWC feature modules must be Linear")
        for module in model.modules():
            if isinstance(module, nn.Conv2d) and module not in ignored and module.out_channels > 1:
                self.handles.append(module.register_forward_hook(self._capture))
            elif module in self.linear_modules and module not in ignored and module.out_features > 1:
                self.handles.append(module.register_forward_hook(self._capture))

    @torch.no_grad()
    def _capture(self, module, inputs, output):
        del inputs
        start = time.perf_counter()
        if module in self.linear_modules:
            if output.ndim != 4 or output.shape[-1] != module.out_features:
                raise ValueError("Declared Linear feature must be NHWC with matching output width")
            output = output.permute(0, 3, 1, 2)
        redundancy = pairwise_redundancy(spatial_probabilities(output),
                                         self.channel_tile, self.spatial_tile)
        previous = self.graphs.get(module)
        if previous is None:
            previous = torch.zeros_like(redundancy)
        if previous.shape != redundancy.shape:
            raise RuntimeError("Reset collector after a structural change")
        self.graphs[module] = previous * self.alpha + redundancy * (1 - self.alpha)
        self.updates[module] = self.updates.get(module, 0) + output.shape[0]
        self.seconds[module] = self.seconds.get(module, 0.0) + time.perf_counter() - start

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def diagnostics(self):
        return {self.names[module]: {"channels": graph.shape[0],
                                    "feature_layout": "NHWC -> NCHW" if module in self.linear_modules else "NCHW",
                                    "patch_updates": self.updates[module],
                                    "seconds": self.seconds[module]}
                for module, graph in self.graphs.items()}


class SIRFPImportance(tp.importance.Importance):
    """Mean coupled-output redundancy graph followed by set-dependent greedy order.

    Mean graph aggregation and common global/isomorphic, width-aligned removal
    are disclosed dependency adaptations. Graphs are sliced after each physical
    removal; they are not reduced to independent fixed per-channel saliency.
    """

    def __init__(self, graphs, *, include_linear=False, include_depthwise=False):
        self.graphs = graphs
        self.output_handlers = {function.prune_conv_out_channels}
        if include_linear:
            self.output_handlers.add(function.prune_linear_out_channels)
        if include_depthwise:
            self.output_handlers.add(function.prune_depthwise_conv_out_channels)

    @torch.no_grad()
    def __call__(self, group):
        channels = len(group[0].idxs)
        total = torch.zeros(channels, channels)
        counts = torch.zeros_like(total)
        seen = set()
        for item in group:
            dependency, indices = item
            module = dependency.target.module
            if module not in self.graphs or dependency.handler not in self.output_handlers:
                continue
            roots = item.root_idxs
            if any(root is None for root in roots):
                continue
            signature = (id(module), tuple(indices), tuple(roots))
            if signature in seen:
                continue
            seen.add(signature)
            roots = torch.tensor(roots)
            indices = torch.tensor(indices)
            graph = self.graphs[module][indices[:, None], indices[None, :]]
            if len(roots.unique()) != len(roots):
                raise RuntimeError("Repeated root indices require explicit graph mapping")
            total[roots[:, None], roots[None, :]] += graph
            counts[roots[:, None], roots[None, :]] += 1
        if not seen:
            return None
        if (counts == 0).any():
            raise RuntimeError("Incomplete dependency-group graph coverage")
        joint = total / counts
        return greedy_order(joint)[0]

    def apply_removal(self, group):
        removals = {}
        for dependency, indices in group:
            module = dependency.target.module
            if module in self.graphs and dependency.handler in self.output_handlers:
                removals.setdefault(module, set()).update(int(index) for index in indices)
        for module, indices in removals.items():
            graph = self.graphs[module]
            keep = [index for index in range(len(graph)) if index not in indices]
            if not keep:
                raise RuntimeError("Refusing to remove all graph channels")
            keep = torch.tensor(keep)
            self.graphs[module] = graph[keep[:, None], keep[None, :]]
