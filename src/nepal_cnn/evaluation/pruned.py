"""One-pass held-out evaluation for configurations frozen on validation evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch_pruning as tp
import yaml
from torch.utils.data import DataLoader

from nepal_cnn.data.dataset import KoshiPositiveDataset
from nepal_cnn.pruning.architecture import (
    architecture_pruning_spec,
    refresh_architecture_after_pruning,
)
from nepal_cnn.utils.paths import resolve_artifact
from nepal_cnn.pruning.audit import load_dense_model
from nepal_cnn.evaluation.spatial import evaluate_manuscript_validation
from nepal_cnn.utils.logging import get_logger


LOGGER = get_logger()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pruned_model(root: Path, checkpoint_path: Path, device: torch.device):
    """Replay structural history on the dense graph, then load recovered weights."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dense_path = resolve_artifact(root, payload["dense_checkpoint"])
    dense_checkpoint, model = load_dense_model(dense_path, torch.device("cpu"))
    architecture = dense_checkpoint["model_config"]["architecture"]
    input_channels = int(dense_checkpoint["model_config"].get("input_channels", 3))
    example = torch.zeros(1, input_channels, 512, 512)
    spec = architecture_pruning_spec(model, architecture)
    graph = tp.DependencyGraph().build_dependency(
        model,
        example,
        unwrapped_parameters=spec.unwrapped_parameters,
        verbose=False,
    )
    graph.load_pruning_history(payload["pruning_history"])
    refresh_architecture_after_pruning(spec)
    model.load_state_dict(payload["model"])
    model = model.to(device).eval()
    with torch.inference_mode():
        output = model(example.to(device))
    if output.shape != (1, 1, 512, 512) or not torch.isfinite(output).all():
        raise RuntimeError(f"Invalid reconstructed output for {checkpoint_path}: {output.shape}")
    return payload, dense_checkpoint, model


@torch.no_grad()
def select_validation_threshold(
    model: torch.nn.Module,
    data_config: Path,
    device: torch.device,
    batch_size: int,
    step: float,
) -> dict:
    """Select the probability threshold by global validation Dice only."""
    model.eval()
    dataset = KoshiPositiveDataset(
        data_config, split="validation", normalize=True, augment=False
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    thresholds = torch.arange(step, 1.0, step, device=device)
    tp_count = torch.zeros_like(thresholds, dtype=torch.float64)
    fp_count = torch.zeros_like(thresholds, dtype=torch.float64)
    fn_count = torch.zeros_like(thresholds, dtype=torch.float64)
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True).bool().unsqueeze(0)
        supervision = (
            batch["supervision"].to(device, non_blocking=True).bool().unsqueeze(0)
        )
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            probability = model(image).sigmoid()
        prediction = probability.unsqueeze(0) >= thresholds[:, None, None, None, None]
        tp_count += (prediction & target & supervision).sum(dim=(1, 2, 3, 4))
        fp_count += (prediction & ~target & supervision).sum(dim=(1, 2, 3, 4))
        fn_count += (~prediction & target & supervision).sum(dim=(1, 2, 3, 4))
    dice = 2 * tp_count / (2 * tp_count + fp_count + fn_count).clamp_min(1)
    iou = tp_count / (tp_count + fp_count + fn_count).clamp_min(1)
    precision = tp_count / (tp_count + fp_count).clamp_min(1)
    recall = tp_count / (tp_count + fn_count).clamp_min(1)
    best = int(torch.argmax(dice))
    return {
        "split": "validation",
        "selection_metric": "global landslide-class Dice",
        "patches": len(dataset),
        "step": step,
        "threshold": float(thresholds[best]),
        "dice": float(dice[best]),
        "iou": float(iou[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "test_data_accessed": False,
    }


def run_final_test(root: Path, manifest_path: Path) -> dict:
    """Validate the freeze record, select thresholds, then access test once."""
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    output_root = root / manifest["output_root"]
    status_path = output_root / "final_test_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") == "completed":
            raise FileExistsError("Frozen final test has already completed; refusing to rerun")
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen final test requires CUDA")
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    data_config = root / manifest["data_config"]
    results = []
    status = {
        "state": "running",
        "manifest": str(manifest_path.relative_to(root)).replace("\\", "/"),
        "configurations": [],
        "test_data_accessed": False,
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    try:
        for entry in manifest["configurations"]:
            checkpoint_path = root / entry["checkpoint"]
            actual_hash = _sha256(checkpoint_path)
            if actual_hash.lower() != str(entry["sha256"]).lower():
                raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint_path}")
            LOGGER.info("FINAL TEST | reconstructing %s", entry["id"])
            payload, dense_checkpoint, model = load_pruned_model(
                root, checkpoint_path, device
            )
            threshold_report = select_validation_threshold(
                model,
                data_config,
                device,
                int(manifest.get("batch_size", 2)),
                float(manifest.get("threshold_step", 0.005)),
            )
            run_dir = output_root / entry["id"]
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "validation_threshold.json").write_text(
                json.dumps(threshold_report, indent=2), encoding="utf-8"
            )
            dense_summary = json.loads(
                (root / entry["dense_test_summary"]).read_text(encoding="utf-8")
            )
            summary = evaluate_manuscript_validation(
                model=model,
                model_name=entry["model"],
                threshold=float(threshold_report["threshold"]),
                dense_checkpoint_path=resolve_artifact(root, payload["dense_checkpoint"]),
                root=root,
                data_config=data_config,
                output_dir=run_dir,
                dense_cache_dir=output_root / "unused_dense_cache",
                device=device,
                batch_size=int(manifest.get("batch_size", 2)),
                bootstrap_replicates=int(manifest["bootstrap_replicates"]),
                bootstrap_seed=int(manifest["bootstrap_seed"]),
                noninferiority_margin=float(manifest["noninferiority_margin"]),
                boundary_width=int(manifest.get("boundary_width_pixels", 2)),
                component_edges=[int(v) for v in manifest["component_size_edges_pixels"]],
                split="test",
                dense_records_path=root / entry["dense_test_per_patch"],
                dense_threshold_override=float(
                    dense_summary["threshold_selected_on_validation"]
                ),
            )
            record = {
                "id": entry["id"],
                "model": entry["model"],
                "method": entry["method"],
                "nominal_mac_reduction": entry["nominal_mac_reduction"],
                "achieved_mac_reduction": payload["achieved_mac_reduction"],
                "checkpoint": entry["checkpoint"],
                "sha256": actual_hash,
                "validation_threshold": threshold_report,
                "test": summary,
                "test_data_accessed": True,
            }
            (run_dir / "final_result.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
            results.append(record)
            status["configurations"].append(entry["id"])
            status["test_data_accessed"] = True
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            LOGGER.info(
                "FINAL TEST | %s | threshold %.3f | Dice %.4f | IoU %.4f",
                entry["id"],
                threshold_report["threshold"],
                summary["dice"],
                summary["iou"],
            )
            del payload, dense_checkpoint, model
            torch.cuda.empty_cache()
    except Exception as error:
        status["state"] = "failed"
        status["error"] = repr(error)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        raise
    status["state"] = "completed"
    status["test_data_accessed"] = True
    status["results"] = [record["id"] for record in results]
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    (output_root / "final_test_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return status
