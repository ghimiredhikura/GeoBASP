"""GeoBASP gated saliency and dependency-group cost-aware importance."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
import torch_pruning as tp
from torch_pruning.pruner import function

from nepal_cnn.training.losses import configured_segmentation_loss


@dataclass
class GeoBASPDiagnostics:
    patches: int
    blocks: int
    regimes: list[str]
    gated_modules: int
    beta: float
    geographic_aggregation: bool
    cost_aware: bool


class ChannelGateBank:
    """Insert temporary differentiable unit gates after Conv2d/Linear outputs."""

    def __init__(self, model: nn.Module):
        self.gates: dict[nn.Module, torch.Tensor] = {}
        self.output_elements_per_channel: dict[nn.Module, float] = {}
        self.handles = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.handles.append(module.register_forward_hook(self._hook))

    def _hook(self, module: nn.Module, inputs, output):
        del inputs
        if not isinstance(output, torch.Tensor):
            return output
        channel_dim = 1 if isinstance(module, nn.Conv2d) else output.ndim - 1
        channels = int(output.shape[channel_dim])
        gate = torch.ones(channels, device=output.device, dtype=output.dtype, requires_grad=True)
        shape = [1] * output.ndim
        shape[channel_dim] = channels
        self.gates[module] = gate
        self.output_elements_per_channel[module] = float(
            output[0].numel() / channels
        )
        return output * gate.view(shape)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class GeoBASPImportance(tp.importance.GroupMagnitudeImportance):
    """Map robust gated channel scores to cost-aware dependency-group scores."""

    def __init__(
        self,
        channel_scores: dict[nn.Module, torch.Tensor],
        output_elements_per_channel: dict[nn.Module, float],
        cost_aware: bool,
    ):
        super().__init__(p=1, group_reduction="max", normalizer=None, bias=False)
        self.channel_scores = channel_scores
        self.output_elements_per_channel = output_elements_per_channel
        self.cost_aware = cost_aware

    def _module_channel_cost(
        self,
        layer: nn.Module,
        prune_fn,
        idxs: list[int],
    ) -> torch.Tensor | None:
        if layer not in self.output_elements_per_channel:
            return None
        output_elements = self.output_elements_per_channel[layer]
        if isinstance(layer, nn.Conv2d):
            total_weight_macs = output_elements * layer.weight.numel()
            if prune_fn == function.prune_conv_out_channels:
                unit = total_weight_macs / layer.out_channels
            elif prune_fn == function.prune_conv_in_channels:
                unit = total_weight_macs / layer.in_channels
            else:
                return None
        elif isinstance(layer, nn.Linear):
            total_weight_macs = output_elements * layer.weight.numel()
            if prune_fn == function.prune_linear_out_channels:
                unit = total_weight_macs / layer.out_features
            elif prune_fn == function.prune_linear_in_channels:
                unit = total_weight_macs / layer.in_features
            else:
                return None
        else:
            return None
        return torch.full((len(idxs),), float(unit), dtype=torch.float32)

    @torch.no_grad()
    def __call__(self, group):
        importance_parts = []
        importance_root_idxs = []
        cost_parts = []
        cost_root_idxs = []
        for index, (dependency, idxs) in enumerate(group):
            layer = dependency.target.module
            prune_fn = dependency.handler
            root_idxs = group[index].root_idxs
            if any(root_idx is None for root_idx in root_idxs):
                continue
            if (
                layer in self.channel_scores
                and prune_fn
                in (function.prune_conv_out_channels, function.prune_linear_out_channels)
            ):
                score = self.channel_scores[layer][idxs].float().cpu()
                importance_parts.append(score)
                importance_root_idxs.append(root_idxs)
            if self.cost_aware:
                cost = self._module_channel_cost(layer, prune_fn, idxs)
                if cost is not None:
                    cost_parts.append(cost)
                    cost_root_idxs.append(root_idxs)
        if not importance_parts:
            return None
        importance = self._reduce(importance_parts, importance_root_idxs)
        if not self.cost_aware or not cost_parts:
            return importance / importance.mean().clamp_min(1e-12)
        original_reduction = self.group_reduction
        self.group_reduction = "sum"
        cost = self._reduce(cost_parts, cost_root_idxs)
        self.group_reduction = original_reduction
        normalized_cost = cost / cost.mean().clamp_min(1e-12)
        utility = importance / normalized_cost.clamp_min(1e-6)
        return utility / utility.mean().clamp_min(1e-12)


def _boundary_supervision(target: torch.Tensor, supervision: torch.Tensor, width: int):
    kernel = 2 * width + 1
    foreground = target.float()
    dilated = F.max_pool2d(foreground, kernel, stride=1, padding=width)
    eroded = 1.0 - F.max_pool2d(1.0 - foreground, kernel, stride=1, padding=width)
    boundary = (dilated - eroded) > 0
    return supervision.bool() & boundary


def _hard_negative_loss(
    logits: torch.Tensor,
    supervision: torch.Tensor,
) -> torch.Tensor:
    valid_logits = logits[supervision.bool()]
    if valid_logits.numel() == 0:
        return logits.sum() * 0
    return F.binary_cross_entropy_with_logits(
        valid_logits, torch.zeros_like(valid_logits), reduction="mean"
    )


def calibrate_geobasp(
    model: nn.Module,
    loader,
    selection_manifest,
    device: torch.device,
    loss_config: dict,
    variant: dict,
) -> tuple[GeoBASPImportance, GeoBASPDiagnostics]:
    """Estimate robust per-channel saliency from training blocks and regimes."""
    regimes = list(variant["regimes"])
    beta = float(variant.get("beta", 1.0))
    use_geography = bool(variant.get("geographic_aggregation", True))
    cost_aware = bool(variant.get("cost_aware", True))
    boundary_width = int(variant.get("boundary_width_pixels", 2))
    regime_weights = {
        regime: float(variant.get("regime_weights", {}).get(regime, 1.0))
        for regime in regimes
    }
    metadata = selection_manifest.set_index("patch_id").to_dict("index")
    original_requires_grad = {
        parameter: parameter.requires_grad for parameter in model.parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    gate_bank = ChannelGateBank(model)
    raw: dict[nn.Module, dict[str, dict[str, list[torch.Tensor]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    patch_count = 0
    try:
        for batch in loader:
            if int(batch["image"].shape[0]) != 1:
                raise RuntimeError("GeoBASP calibration requires batch size one")
            patch_id = str(batch["patch_id"][0])
            row = metadata[patch_id]
            block_id = str(row["block_id"])
            size_stratum = str(row["calibration_size_stratum"])
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            supervision = batch["supervision"].to(device, non_blocking=True)
            logits = model(image)
            losses: dict[str, torch.Tensor] = {}
            if "overall" in regimes:
                losses["overall"] = configured_segmentation_loss(
                    logits, target, supervision, loss_config
                )
            if "boundary" in regimes and size_stratum != "negative":
                boundary_supervision = _boundary_supervision(
                    target, supervision, boundary_width
                )
                losses["boundary"] = configured_segmentation_loss(
                    logits, target, boundary_supervision, loss_config
                )
            if "small" in regimes and size_stratum == "small":
                losses["small"] = configured_segmentation_loss(
                    logits, target, supervision, loss_config
                )
            if "hard_negative" in regimes and size_stratum == "negative":
                losses["hard_negative"] = _hard_negative_loss(logits, supervision)
            gates = list(gate_bank.gates.items())
            for loss_index, (regime, loss) in enumerate(losses.items()):
                gradients = torch.autograd.grad(
                    loss,
                    [gate for _, gate in gates],
                    retain_graph=loss_index + 1 < len(losses),
                    allow_unused=True,
                )
                for (module, gate), gradient in zip(gates, gradients):
                    if gradient is None:
                        continue
                    sensitivity = (gate * gradient).detach().abs().float().cpu()
                    sensitivity /= sensitivity.mean().clamp_min(1e-12)
                    raw[module][regime][block_id].append(sensitivity)
            patch_count += 1
            del logits, losses, gates
    finally:
        gate_bank.close()
        for parameter, requires_grad in original_requires_grad.items():
            parameter.requires_grad_(requires_grad)

    channel_scores: dict[nn.Module, torch.Tensor] = {}
    for module, module_regimes in raw.items():
        protected_regimes = []
        for regime in regimes:
            block_values = module_regimes.get(regime, {})
            if not block_values:
                continue
            per_block = torch.stack(
                [torch.stack(values).mean(0) for values in block_values.values()]
            )
            if use_geography:
                median = per_block.median(0).values
                mad = (per_block - median).abs().median(0).values
                aggregated = median + beta * mad
            else:
                aggregated = per_block.mean(0)
            protected_regimes.append(regime_weights[regime] * aggregated)
        if protected_regimes:
            channel_scores[module] = torch.stack(protected_regimes).amax(0)
    if not channel_scores:
        raise RuntimeError("GeoBASP calibration produced no channel scores")
    importance = GeoBASPImportance(
        channel_scores,
        gate_bank.output_elements_per_channel,
        cost_aware=cost_aware,
    )
    diagnostics = GeoBASPDiagnostics(
        patches=patch_count,
        blocks=int(selection_manifest["block_id"].nunique()),
        regimes=regimes,
        gated_modules=len(channel_scores),
        beta=beta,
        geographic_aggregation=use_geography,
        cost_aware=cost_aware,
    )
    return importance, diagnostics
