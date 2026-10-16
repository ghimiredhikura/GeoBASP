"""Fixed-policy deployment measurements for pruned segmentation stages."""

from __future__ import annotations

import time

import numpy as np
import torch


@torch.no_grad()
def benchmark_gpu(
    model: torch.nn.Module,
    example: torch.Tensor,
    warmup: int,
    repetitions: int,
    pixel_resolution_m: float = 10.0,
) -> dict:
    if example.device.type != "cuda":
        raise ValueError("GPU benchmark requires a CUDA example tensor")
    model.eval()
    torch.cuda.empty_cache()
    for _ in range(warmup):
        model(example)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(example.device)
    timings = []
    for _ in range(repetitions):
        start = time.perf_counter()
        model(example)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    median = float(np.median(timings))
    p95 = float(np.quantile(timings, 0.95))
    area_km2 = (
        example.shape[0]
        * example.shape[-2]
        * example.shape[-1]
        * pixel_resolution_m**2
        / 1e6
    )
    return {
        "device": torch.cuda.get_device_name(example.device),
        "dtype": "float32",
        "batch_size": int(example.shape[0]),
        "input_shape": list(example.shape),
        "warmup_iterations": warmup,
        "timed_iterations": repetitions,
        "latency_median_ms": median,
        "latency_p95_ms": p95,
        "throughput_images_per_second": 1000 * example.shape[0] / median,
        "mapped_area_km2_per_second": 1000 * area_km2 / median,
        "peak_allocated_memory_mb": torch.cuda.max_memory_allocated(example.device)
        / 1024**2,
    }
