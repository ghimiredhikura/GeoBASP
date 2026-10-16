"""Matched CUDA Graph FP32 latency on the current computer; no dataset needed."""
import argparse
import gc
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import psutil
import timm
import torch
import torchvision
from nepal_cnn.utils.paths import PROJECT_ROOT as ROOT
from nepal_cnn.models.checkpoints import load_checkpoint
MIB = 1024 ** 2

def command_output(arguments):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=15)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def system_information(device):
    cpu = platform.processor()
    if sys.platform == "win32":
        cpu = command_output(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                              "(Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name) -join ', '"]) or cpu
    elif sys.platform.startswith("linux"):
        try:
            cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                       if line.startswith("model name"))
        except (OSError, StopIteration):
            pass
    info = {
        "measured_at_local": datetime.now().astimezone().isoformat(),
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(), "system_architecture": platform.machine(),
        "cpu": cpu, "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "system_ram_gib": psutil.virtual_memory().total / 1024 ** 3,
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "timm": timm.__version__,
        "pytorch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "execution_device": str(device), "torch_cpu_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info["gpu"] = {"name": properties.name, "total_vram_gib": properties.total_memory / 1024 ** 3,
                       "compute_capability": list(torch.cuda.get_device_capability(device)),
                       "multiprocessors": properties.multi_processor_count}
        info["nvidia_driver_inventory"] = command_output([
            "nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"])
    return info


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@torch.no_grad()
def measure(model, example, warmup, iterations):
    device = example.device
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    output = model(example)
    if tuple(output.shape) != (1, 1, 512, 512) or not bool(torch.isfinite(output).all()):
        raise RuntimeError("Invalid inference output")
    del output
    for _ in range(warmup):
        model(example)
    synchronize()
    rss_before = psutil.Process().memory_info().rss / MIB
    if device.type == "cuda":
        resident_allocated = torch.cuda.memory_allocated(device) / MIB
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        model(example)
        synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    median = statistics.median(timings)
    result = {
        "latency_median_ms": median, "latency_mean_ms": statistics.mean(timings),
        "latency_sd_ms": statistics.stdev(timings), "latency_p95_ms": percentile(timings, .95),
        "throughput_images_per_second": 1000 / median,
        "process_rss_before_timing_mib": rss_before,
        "process_rss_after_timing_mib": psutil.Process().memory_info().rss / MIB,
    }
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / MIB
        result.update({"gpu_resident_allocated_mib": resident_allocated,
                       "gpu_peak_allocated_mib": peak,
                       "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
                       "gpu_peak_extra_allocated_mib": peak - resident_allocated})
    return result, timings

@torch.no_grad()
def capture_graph(model, example):
    """Capture fixed-shape inference and verify two different inputs."""
    static_input = example.clone()
    current = torch.cuda.current_stream(example.device)
    side = torch.cuda.Stream(device=example.device)
    side.wait_stream(current)
    with torch.cuda.stream(side):
        for _ in range(5):
            model(static_input)
    current.wait_stream(side)
    torch.cuda.synchronize(example.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        static_output = model(static_input)
    current.wait_stream(side)
    checks = []
    generator = torch.Generator(device=example.device).manual_seed(2027)
    second_input = torch.randn(example.shape, dtype=example.dtype, device=example.device, generator=generator)
    for tensor in (example, second_input):
        expected = model(tensor)
        static_input.copy_(tensor)
        graph.replay()
        torch.cuda.synchronize(example.device)
        torch.testing.assert_close(static_output, expected, rtol=1e-5, atol=1e-5)
        checks.append({"max_absolute_difference": float((static_output - expected).abs().max()),
                       "exact_match": bool(torch.equal(static_output, expected)),
                       "rtol": 1e-5, "atol": 1e-5})
        del expected
    del tensor, second_input
    static_input.copy_(example)
    graph.replay()
    torch.cuda.synchronize(example.device)
    return graph, static_input, static_output, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=["cuda_graph", "eager"], default="cuda_graph")
    parser.add_argument("--checkpoint", action="append", metavar="LABEL=PATH", help="Add an explicitly chosen checkpoint, e.g. SIRFP=generated/sirfp/...")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-session", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.sessions < 1 or args.warmup < 1 or args.iterations < 2:
        parser.error("Sessions/warmup must be positive and iterations at least two")
    output = ROOT / (args.output or Path("generated/latency") / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if args.worker_session is None:
        output.mkdir(parents=True, exist_ok=False)
        reports = []
        for session in range(1, args.sessions + 1):
            command = [sys.executable, "-B", "-m", "nepal_cnn.evaluation.latency", "--worker-session", str(session),
                       "--output", str(output), "--backend", args.backend, "--device", args.device,
                       "--warmup", str(args.warmup), "--iterations", str(args.iterations)]
            for checkpoint in args.checkpoint or []:
                command += ["--checkpoint", checkpoint]
            subprocess.run(command, cwd=ROOT, check=True)
            reports.append(json.loads((output / f"session_{session:02d}.json").read_text()))
        summary = []
        for label in reports[0]["results"]:
            medians = [r["results"][label]["latency_median_ms"] for r in reports]
            gains = [100*(1-r["results"][label]["latency_median_ms"]/r["results"]["Dense"]["latency_median_ms"]) for r in reports]
            summary.append({"method": label, "latency_mean_ms": statistics.mean(medians),
                            "latency_sample_sd_ms": statistics.stdev(medians) if len(medians)>1 else None,
                            "mean_paired_latency_reduction_percent": statistics.mean(gains)})
        (output / "summary.json").write_text(json.dumps({"backend": args.backend, "sessions": len(reports), "results": summary}, indent=2)+"\n")
        print(json.dumps(summary, indent=2))
        print(f"Measurements: {output}")
        return
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("Use a CUDA-enabled PyTorch environment for this GPU protocol")
    torch.cuda.set_device(0 if device.index is None else device.index)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(2026)
    generator = torch.Generator(device=device).manual_seed(2026)
    example = torch.randn(1,4,512,512,device=device,generator=generator)
    catalog = json.loads((ROOT / "metadata/models.json").read_text())
    labels = {"dense_reference": "Dense", "geobasp": "GeoBASP", "group_magnitude": "Magnitude", "group_taylor": "Taylor"}
    entries = {}
    for item in catalog["models"]:
        if item["model"] == "convnextv2_tiny":
            path = ROOT / item["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Checkpoint checksum differs: {path}")
            entries[labels[item["kind"] if item["kind"]=="dense_reference" else item["method"]]] = path
    for specification in args.checkpoint or []:
        label, separator, path = specification.partition("=")
        if not separator or not label or label in entries:
            parser.error("Extra checkpoints require a unique LABEL=PATH")
        entries[label] = ROOT / path
    report = {"system": system_information(device), "session": args.worker_session,
              "protocol": {"backend": args.backend, "precision": "FP32", "tf32": False,
                           "input_shape": [1,4,512,512], "warmup": args.warmup, "iterations": args.iterations,
                           "timing": "synchronized forward wall time; resident input; loading and capture excluded",
                           "memory": "PyTorch allocator counters; graph-pool intermediates also occupy reserved memory"}, "results": {}}
    order = sorted(entries, key=lambda label: (label!="Dense", label))
    if args.worker_session % 2 == 0:
        order.reverse()
    for label in order:
        print(f"Session {args.worker_session}: {label} / {args.backend}", flush=True)
        with torch.no_grad():
            _, model = load_checkpoint(entries[label], device)
            model.eval()
            graph = static_input = static_output = checks = None
            if args.backend == "cuda_graph":
                graph, static_input, static_output, checks = capture_graph(model, example)
                def runner(_):
                    graph.replay()
                    return static_output
            else:
                runner = model
            measured, samples = measure(runner, example, args.warmup, args.iterations)
            measured.update(raw_latency_ms=samples, graph_output_checks=checks,
                            checkpoint_sha256=hashlib.sha256(entries[label].read_bytes()).hexdigest())
            report["results"][label] = measured
            del runner, graph, static_input, static_output, model
        gc.collect()
        torch.cuda.empty_cache()
    (output / f"session_{args.worker_session:02d}.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
