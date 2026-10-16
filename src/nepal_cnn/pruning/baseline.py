"""Matched-protocol Group Magnitude and Group Taylor pruning experiments."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from datetime import datetime

import torch
import torch_pruning as tp
import yaml

from nepal_cnn.pruning.architecture import (
    build_meta_pruner,
    importance_for_method,
    refresh_architecture_after_pruning,
)
from nepal_cnn.pruning.audit import load_dense_model
from nepal_cnn.pruning.calibration import (
    accumulate_overall_taylor_gradients,
    build_calibration_loader,
)
from nepal_cnn.pruning.geobasp import calibrate_geobasp
from nepal_cnn.evaluation.spatial import evaluate_manuscript_validation
from nepal_cnn.evaluation.benchmark import benchmark_gpu
from nepal_cnn.utils.logging import get_logger
from nepal_cnn.training.recovery import (
    build_recovery_loaders,
    evaluate_validation,
    recover_model,
    seed_everything,
)


LOGGER = get_logger()


def _append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prune_to_nearest_mac_target(
    model: torch.nn.Module,
    architecture: str,
    method: str,
    example: torch.Tensor,
    base_macs: int,
    target: float,
    micro_ratio: float,
    maximum_microsteps: int,
    calibration_loader,
    geobasp_loader,
    calibration_selection,
    calibration_loss_config: dict,
    geobasp_variant: dict | None,
    device: torch.device,
    cumulative_history: list,
) -> tuple[torch.nn.Module, float, list, list[dict]]:
    """Choose the closer aligned structure immediately below/above a MAC target."""
    steps: list[dict] = []
    current_macs, _ = tp.utils.count_ops_and_params(model, example)
    current_reduction = float(1 - current_macs / base_macs)
    for step_index in range(1, maximum_microsteps + 1):
        if current_reduction >= target:
            break
        model = model.cpu()
        previous_model = copy.deepcopy(model)
        model = model.to(device)
        previous_macs = int(current_macs)
        previous_reduction = current_reduction
        diagnostics = None
        if method in {"group_magnitude", "group_taylor"}:
            importance = importance_for_method(method)
        else:
            if geobasp_variant is None:
                raise RuntimeError(f"Missing GeoBASP variant configuration for {method}")
            importance, geo_diagnostics = calibrate_geobasp(
                model,
                geobasp_loader,
                calibration_selection.manifest,
                device,
                calibration_loss_config,
                geobasp_variant,
            )
            diagnostics = vars(geo_diagnostics)
        pruner, spec = build_meta_pruner(
            model,
            example,
            architecture,
            importance,
            micro_ratio,
        )
        calibration_loss = None
        if method == "group_taylor":
            calibration_loss = accumulate_overall_taylor_gradients(
                model,
                calibration_loader,
                device,
                calibration_loss_config,
            )
        pruner.step()
        step_history = copy.deepcopy(pruner.pruning_history())
        refresh_architecture_after_pruning(spec)
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.inference_mode():
            output = model(example)
        if not torch.isfinite(output).all():
            raise RuntimeError(f"Non-finite output after {method} micro-step {step_index}")
        current_macs, current_parameters = tp.utils.count_ops_and_params(model, example)
        current_macs = int(current_macs)
        current_reduction = float(1 - current_macs / base_macs)
        if current_macs >= previous_macs:
            raise RuntimeError(
                f"{method} micro-step {step_index} did not reduce MACs "
                f"({previous_macs} -> {current_macs})"
            )
        accept_new = True
        if current_reduction >= target:
            previous_distance = abs(previous_reduction - target)
            new_distance = abs(current_reduction - target)
            accept_new = new_distance < previous_distance
        if accept_new:
            cumulative_history.extend(step_history)
            del previous_model
            chosen_macs = current_macs
            chosen_reduction = current_reduction
        else:
            del model
            torch.cuda.empty_cache()
            model = previous_model.to(device)
            current_macs = previous_macs
            current_reduction = previous_reduction
            chosen_macs = previous_macs
            chosen_reduction = previous_reduction
        record = {
            "micro_step": step_index,
            "requested_channel_ratio": micro_ratio,
            "calibration_loss": calibration_loss,
            "candidate_macs": int(tp.utils.count_ops_and_params(model, example)[0]),
            "candidate_parameters": int(tp.utils.count_ops_and_params(model, example)[1]),
            "chosen_macs": chosen_macs,
            "chosen_mac_reduction": chosen_reduction,
            "accepted": accept_new,
            "criterion_diagnostics": diagnostics,
        }
        steps.append(record)
        LOGGER.info(
            "PRUNE | %s | target %.1f%% | step %02d | achieved %.2f%% | %s",
            method,
            100 * target,
            step_index,
            100 * chosen_reduction,
            "accepted" if accept_new else "kept previous structure",
        )
        del pruner, spec, output
        torch.cuda.empty_cache()
        if not accept_new or current_reduction >= target:
            break
    else:
        raise RuntimeError(
            f"Failed to reach nominal MAC target {target:.3f} in "
            f"{maximum_microsteps} micro-steps"
        )
    return model, current_reduction, cumulative_history, steps


def run_baselines(
    root: Path,
    config_path: Path,
    models: list[str] | None,
    methods: list[str] | None,
    targets: list[float] | None,
    output_root: Path,
    recovery_epochs: int | None,
    dry_run: bool,
    resume: bool = False,
    seed_override: int | None = None,
) -> list[dict]:
    """Run validation-only baseline pruning trajectories; never access test rows."""
    if not torch.cuda.is_available():
        raise RuntimeError("Baseline pruning requires CUDA")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected = config["selected_dense_checkpoints"]
    models = models or list(selected)
    methods = methods or list(config["baseline_methods"])
    targets = sorted(targets or [float(value) for value in config["target_mac_reductions"]])
    if set(models) - set(selected):
        raise ValueError(f"Unknown models: {sorted(set(models) - set(selected))}")
    available_methods = {
        *config["baseline_methods"], *config.get("geobasp_variants", {})
    }
    if set(methods) - available_methods:
        raise ValueError(
            f"Unknown methods: {sorted(set(methods) - available_methods)}"
        )
    if any(target <= 0 or target >= 1 for target in targets):
        raise ValueError("MAC targets must lie between zero and one")
    if any(left >= right for left, right in zip(targets, targets[1:])):
        raise ValueError("MAC targets must be strictly increasing")

    seed = int(config["seed"] if seed_override is None else seed_override)
    seed_everything(seed)
    device = torch.device("cuda")
    calibration_config = config["calibration"]
    calibration_loader = geobasp_loader = calibration_selection = None
    gradient_methods = set(methods) - {"group_magnitude"}
    if gradient_methods:
        calibration_loader, calibration_selection = build_calibration_loader(
            root / config["data_config"],
            int(calibration_config["positive_patches"]),
            int(calibration_config["hard_negative_patches"]),
            int(calibration_config["batch_size"]),
            seed,
        )
        output_root.mkdir(parents=True, exist_ok=True)
        calibration_selection.manifest.to_csv(
            output_root / "calibration_manifest.csv", index=False
        )
    if set(methods) & set(config.get("geobasp_variants", {})):
        geobasp_loader, geobasp_selection = build_calibration_loader(
            root / config["data_config"],
            int(calibration_config["positive_patches"]),
            int(calibration_config["hard_negative_patches"]),
            1,
            seed,
        )
        if geobasp_selection.indices != calibration_selection.indices:
            raise RuntimeError("Baseline and GeoBASP calibration selections differ")

    recovery_config = config["recovery"]
    configured_epochs = int(recovery_config["epochs_per_stage"])
    epochs = configured_epochs if recovery_epochs is None else int(recovery_epochs)
    if dry_run:
        epochs = 0
    example_size = int(config.get("mac_measurement_image_size", 512))
    micro_ratio = float(config.get("micro_channel_pruning_ratio", 0.01))
    maximum_microsteps = int(config.get("maximum_microsteps_per_target", 32))
    tolerance = float(config["maximum_target_overshoot"])
    all_summaries: list[dict] = []

    for model_name in models:
        for method in methods:
            branch_dir = output_root / model_name / method
            summary_path = branch_dir / "summary.json"
            if summary_path.exists():
                if resume:
                    LOGGER.info("SKIP | completed branch %s/%s", model_name, method)
                    all_summaries.append(
                        json.loads(summary_path.read_text(encoding="utf-8"))
                    )
                    continue
                raise FileExistsError(f"Refusing to overwrite completed branch: {summary_path}")
            if branch_dir.exists() and resume and any(branch_dir.iterdir()):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                preserved = branch_dir.with_name(f"{branch_dir.name}_incomplete_{timestamp}")
                branch_dir.rename(preserved)
                LOGGER.warning("PRESERVE | moved incomplete branch to %s", preserved)
            branch_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info(
                "START | model=%s | method=%s | seed=%d | targets=%s | recovery=%d epochs",
                model_name,
                method,
                seed,
                ",".join(f"{100 * value:.0f}%" for value in targets),
                epochs,
            )
            checkpoint, model = load_dense_model(root / selected[model_name], device)
            architecture = checkpoint["model_config"]["architecture"]
            dense_train_config = checkpoint["train_config"]
            example = torch.randn(
                1,
                int(checkpoint["model_config"].get("input_channels", 3)),
                example_size,
                example_size,
                device=device,
            )
            model.eval()
            base_macs, base_parameters = tp.utils.count_ops_and_params(model, example)
            base_macs, base_parameters = int(base_macs), int(base_parameters)
            train_loader, validation_loader = build_recovery_loaders(
                root / config["data_config"],
                dense_train_config,
                recovery_config,
                seed,
                device,
            )
            cumulative_history: list = []
            branch_stages: list[dict] = []
            for target in targets:
                model, achieved, cumulative_history, steps = _prune_to_nearest_mac_target(
                    model=model,
                    architecture=architecture,
                    method=method,
                    example=example,
                    base_macs=base_macs,
                    target=target,
                    micro_ratio=micro_ratio,
                    maximum_microsteps=maximum_microsteps,
                    calibration_loader=calibration_loader,
                    geobasp_loader=geobasp_loader,
                    calibration_selection=calibration_selection,
                    calibration_loss_config=dense_train_config["loss"],
                    geobasp_variant=config.get("geobasp_variants", {}).get(method),
                    device=device,
                    cumulative_history=cumulative_history,
                )
                current_macs, current_parameters = tp.utils.count_ops_and_params(model, example)
                no_recovery = evaluate_validation(
                    model,
                    validation_loader,
                    device,
                    dense_train_config["loss"],
                    [float(value) for value in dense_train_config["thresholds"]],
                    bool(recovery_config.get("amp", True)),
                )
                recovery_history: list[dict] = []
                if epochs > 0:
                    model, recovered, recovery_history = recover_model(
                        model,
                        train_loader,
                        validation_loader,
                        device,
                        dense_train_config,
                        recovery_config,
                        epochs,
                        {
                            "model": model_name,
                            "method": method,
                            "nominal_mac_reduction": target,
                            "achieved_mac_reduction": achieved,
                        },
                    )
                else:
                    recovered = {**no_recovery, "epoch": 0}
                label = f"mac_{int(round(target * 100)):02d}"
                stage_dir = branch_dir / label
                stage_dir.mkdir(parents=True, exist_ok=False)
                extended = None
                extended_config = config.get("extended_validation", {})
                if bool(extended_config.get("enabled", True)):
                    statistics = config["statistics"]
                    extended = evaluate_manuscript_validation(
                        model=model,
                        model_name=model_name,
                        threshold=float(recovered["best_threshold"]),
                        dense_checkpoint_path=root / selected[model_name],
                        root=root,
                        data_config=root / config["data_config"],
                        output_dir=stage_dir,
                        dense_cache_dir=root / "generated/pruning/dense_validation_reference/repeated",
                        device=device,
                        batch_size=int(extended_config.get("batch_size", 2)),
                        bootstrap_replicates=int(
                            statistics["spatial_block_bootstrap_replicates"]
                        ),
                        bootstrap_seed=int(statistics["spatial_block_bootstrap_seed"]),
                        noninferiority_margin=float(
                            statistics["validation_noninferiority_margin_dice"]
                        ),
                        boundary_width=int(
                            extended_config.get("boundary_width_pixels", 2)
                        ),
                        component_edges=[
                            int(value)
                            for value in extended_config.get(
                                "component_size_edges_pixels", [9, 25, 100]
                            )
                        ],
                    )
                benchmark = None
                benchmark_config = config.get("benchmark", {})
                if bool(benchmark_config.get("enabled", True)):
                    benchmark = benchmark_gpu(
                        model,
                        example,
                        warmup=int(benchmark_config.get("warmup_iterations", 30)),
                        repetitions=int(benchmark_config.get("timed_iterations", 100)),
                        pixel_resolution_m=float(
                            benchmark_config.get("pixel_resolution_m", 10.0)
                        ),
                    )
                model = model.cpu()
                checkpoint_payload = {
                    "model": model.state_dict(),
                    "model_config": checkpoint["model_config"],
                    "dense_checkpoint": selected[model_name],
                    "method": method,
                    "nominal_mac_reduction": target,
                    "achieved_mac_reduction": achieved,
                    "pruning_history": cumulative_history,
                    "validation_no_recovery": no_recovery,
                    "validation_recovered": recovered,
                    "validation_extended": extended,
                    "gpu_benchmark": benchmark,
                    "test_data_accessed": False,
                }
                torch.save(checkpoint_payload, stage_dir / "checkpoint.pt")
                checkpoint_size_mb = (stage_dir / "checkpoint.pt").stat().st_size / 1024**2
                (stage_dir / "pruning_steps.json").write_text(
                    json.dumps(steps, indent=2), encoding="utf-8"
                )
                _append_csv(stage_dir / "recovery_history.csv", recovery_history)
                stage_summary = {
                    "model": model_name,
                    "architecture": architecture,
                    "method": method,
                    "nominal_mac_reduction": target,
                    "achieved_mac_reduction": achieved,
                    "target_absolute_error": abs(achieved - target),
                    "within_target_tolerance": abs(achieved - target) <= tolerance,
                    "macs": int(current_macs),
                    "parameters": int(current_parameters),
                    "validation_no_recovery": no_recovery,
                    "validation_recovered": recovered,
                    "validation_extended": extended,
                    "gpu_benchmark": benchmark,
                    "checkpoint_size_mb": checkpoint_size_mb,
                    "recovery_epochs": epochs,
                    "test_data_accessed": False,
                }
                (stage_dir / "summary.json").write_text(
                    json.dumps(stage_summary, indent=2), encoding="utf-8"
                )
                branch_stages.append(stage_summary)
                LOGGER.info(
                    "DONE  | %s | %s | nominal %.0f%% / achieved %.2f%% | "
                    "val Dice %.4f | params %.3fM | MACs %.3fG",
                    model_name,
                    method,
                    100 * target,
                    100 * achieved,
                    recovered["best_dice"],
                    current_parameters / 1e6,
                    current_macs / 1e9,
                )
                model = model.to(device)
            branch_summary = {
                "model": model_name,
                "architecture": architecture,
                "method": method,
                "dense_checkpoint": selected[model_name],
                "dense_macs": base_macs,
                "dense_parameters": base_parameters,
                "targets": branch_stages,
                "calibration_split": "train",
                "test_data_accessed": False,
                "dry_run": dry_run,
                "seed": seed,
            }
            summary_path.write_text(json.dumps(branch_summary, indent=2), encoding="utf-8")
            all_summaries.append(branch_summary)
            del checkpoint, model, train_loader, validation_loader
            torch.cuda.empty_cache()
    return all_summaries
