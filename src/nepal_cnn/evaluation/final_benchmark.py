"""Uniform FP32 benchmark for the frozen dense and pruned operating points."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch_pruning as tp
import yaml

from nepal_cnn.utils.paths import resolve_artifact
from nepal_cnn.pruning.audit import load_dense_model
from nepal_cnn.evaluation.benchmark import benchmark_gpu
from nepal_cnn.evaluation.pruned import load_pruned_model


def _measure(model: torch.nn.Module, input_channels: int, device: torch.device) -> dict:
    generator = torch.Generator(device=device).manual_seed(2026)
    example = torch.randn(
        1, input_channels, 512, 512, generator=generator, device=device
    )
    macs, parameters = tp.utils.count_ops_and_params(model, example)
    timing = benchmark_gpu(model, example, warmup=50, repetitions=200)
    return {"parameters": int(parameters), "macs": int(macs), **timing}


def run_final_benchmark(root: Path, manifest_path: Path) -> dict:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    output_root = root / manifest["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "final_benchmark.json"
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    first_payload = torch.load(
        root / manifest["configurations"][0]["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    dense_path = resolve_artifact(root, first_payload["dense_checkpoint"])
    dense_checkpoint, dense_model = load_dense_model(dense_path, device)
    input_channels = int(dense_checkpoint["model_config"].get("input_channels", 3))
    records = [
        {
            "id": "convnextv2_tiny_dense",
            "model": "convnextv2_tiny",
            "method": "dense",
            **_measure(dense_model, input_channels, device),
        }
    ]
    del dense_model, dense_checkpoint, first_payload
    torch.cuda.empty_cache()
    for entry in manifest["configurations"]:
        payload, dense_checkpoint, model = load_pruned_model(
            root, root / entry["checkpoint"], device
        )
        records.append(
            {
                "id": entry["id"],
                "model": entry["model"],
                "method": entry["method"],
                "achieved_mac_reduction": payload["achieved_mac_reduction"],
                **_measure(model, input_channels, device),
            }
        )
        del payload, dense_checkpoint, model
        torch.cuda.empty_cache()
    report = {
        "manifest": str(manifest_path.relative_to(root)).replace("\\", "/"),
        "policy": {
            "device": torch.cuda.get_device_name(device),
            "precision": "strict FP32",
            "tf32": False,
            "warmup_iterations": 50,
            "timed_iterations": 200,
            "seed": 2026,
        },
        "results": records,
        "test_data_accessed": False,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report
