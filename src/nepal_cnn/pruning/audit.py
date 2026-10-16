"""One-step compatibility audit for frozen models and baseline criteria."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch_pruning as tp
import yaml

from nepal_cnn.models.architectures import build_model
from nepal_cnn.pruning.architecture import (
    architecture_pruning_spec,
    build_meta_pruner,
    importance_for_method,
    refresh_architecture_after_pruning,
)
from nepal_cnn.pruning.calibration import (
    accumulate_overall_taylor_gradients,
    build_calibration_loader,
)


def load_dense_model(checkpoint_path: Path, device: torch.device):
    """Reconstruct a dense architecture and load its selected weights."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model = build_model(
        architecture=config["architecture"],
        pretrained=False,
        decoder_dropout=float(config.get("decoder_dropout", 0.0)),
        input_channels=int(config.get("input_channels", 3)),
    )
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model.to(device)


def run_audit(
    root: Path,
    config_path: Path,
    methods: list[str] | None,
    models: list[str] | None,
    pruning_ratio: float,
    image_size: int,
    output_path: Path,
) -> dict:
    """Physically prune once and verify outputs and protected graph endpoints."""
    if not 0 < pruning_ratio < 1:
        raise ValueError("pruning_ratio must be between zero and one")
    if not torch.cuda.is_available():
        raise RuntimeError("The pruning audit requires CUDA")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected = config["selected_dense_checkpoints"]
    methods = methods or list(config["baseline_methods"])
    models = models or list(selected)
    unknown_models = set(models) - set(selected)
    if unknown_models:
        raise ValueError(f"Unknown models: {sorted(unknown_models)}")
    unknown_methods = set(methods) - set(config["baseline_methods"])
    if unknown_methods:
        raise ValueError(f"Unknown baseline methods: {sorted(unknown_methods)}")

    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    calibration_config = config["calibration"]
    loader = selection = None
    if "group_taylor" in methods:
        loader, selection = build_calibration_loader(
            root / config["data_config"],
            positive_count=int(calibration_config["positive_patches"]),
            negative_count=int(calibration_config["hard_negative_patches"]),
            batch_size=int(calibration_config["batch_size"]),
            seed=seed,
        )

    reports: list[dict] = []
    for model_name in models:
        for method in methods:
            checkpoint_path = root / selected[model_name]
            checkpoint, model = load_dense_model(checkpoint_path, device)
            architecture = checkpoint["model_config"]["architecture"]
            model.eval()
            example = torch.randn(
                1,
                int(checkpoint["model_config"].get("input_channels", 3)),
                image_size,
                image_size,
                device=device,
            )
            with torch.inference_mode():
                dense_output = model(example)
            base_macs, base_parameters = tp.utils.count_ops_and_params(model, example)
            before = architecture_pruning_spec(model, architecture)
            protected_shapes = {
                name: tuple(module.weight.shape)
                for name, module in zip(before.ignored_layer_names, before.ignored_layers)
                if hasattr(module, "weight")
            }
            pruner, spec = build_meta_pruner(
                model,
                example,
                architecture,
                importance_for_method(method),
                pruning_ratio,
            )
            calibration_loss = None
            if method == "group_taylor":
                assert loader is not None
                calibration_loss = accumulate_overall_taylor_gradients(
                    model,
                    loader,
                    device,
                    checkpoint["train_config"]["loss"],
                )
            pruner.step()
            refresh_architecture_after_pruning(spec)
            model.zero_grad(set_to_none=True)
            model.eval()
            with torch.inference_mode():
                pruned_output = model(example)
            macs, parameters = tp.utils.count_ops_and_params(model, example)
            after = architecture_pruning_spec(model, architecture)
            post_shapes = {
                name: tuple(module.weight.shape)
                for name, module in zip(after.ignored_layer_names, after.ignored_layers)
                if hasattr(module, "weight")
            }
            if pruned_output.shape != dense_output.shape:
                raise RuntimeError(
                    f"{model_name}/{method}: output changed from "
                    f"{dense_output.shape} to {pruned_output.shape}"
                )
            input_name, classifier_name = spec.ignored_layer_names
            if not (
                protected_shapes[input_name] == post_shapes[input_name]
                and protected_shapes[classifier_name][0]
                == post_shapes[classifier_name][0]
                == 1
            ):
                raise RuntimeError(f"{model_name}/{method}: protected-layer invariant failed")
            if not torch.isfinite(pruned_output).all():
                raise RuntimeError(f"{model_name}/{method}: non-finite output")
            report = {
                "model": model_name,
                "architecture": architecture,
                "method": method,
                "checkpoint": selected[model_name],
                "checkpoint_epoch": checkpoint.get("epoch"),
                "requested_channel_pruning_ratio": pruning_ratio,
                "input_shape": list(example.shape),
                "output_shape": list(pruned_output.shape),
                "protected_layers": spec.ignored_layer_names,
                "protected_shapes_before": protected_shapes,
                "protected_shapes_after": post_shapes,
                "attention_modules": len(spec.attention_modules),
                "calibration_loss": calibration_loss,
                "parameters_before": int(base_parameters),
                "parameters_after": int(parameters),
                "parameter_reduction": float(1 - parameters / base_parameters),
                "macs_before": int(base_macs),
                "macs_after": int(macs),
                "mac_reduction": float(1 - macs / base_macs),
                "status": "passed",
            }
            reports.append(report)
            print(json.dumps(report), flush=True)
            del checkpoint, model, pruner, spec, example, dense_output, pruned_output
            torch.cuda.empty_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_summary = None
    if selection is not None:
        calibration_manifest = output_path.with_name("calibration_manifest.csv")
        selection.manifest.to_csv(calibration_manifest, index=False)
        calibration_summary = {
            "split": "train",
            "positive_patches": selection.positive_count,
            "hard_negative_patches": selection.negative_count,
            "unique_blocks": int(selection.manifest["block_id"].nunique()),
            "manifest": calibration_manifest.relative_to(root).as_posix(),
        }
    summary = {
        "torch": torch.__version__,
        "torch_pruning": tp.__version__,
        "torch_pruning_commit": config["torch_pruning_commit"],
        "data_split": "train",
        "test_data_accessed": False,
        "calibration": calibration_summary,
        "reports": reports,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"output": output_path.relative_to(root).as_posix(), "status": "all_passed"}
        )
    )
    return summary
