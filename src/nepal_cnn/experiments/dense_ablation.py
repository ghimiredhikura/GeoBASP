"""Run or smoke-test the resumable dense-model adaptation ablation."""

from __future__ import annotations

from nepal_cnn.utils.paths import PROJECT_ROOT as ROOT

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

from nepal_cnn.data.dataset import KoshiPositiveDataset
from nepal_cnn.training.losses import configured_segmentation_loss
from nepal_cnn.models.architectures import build_model
from nepal_cnn.training.readiness import check_primary_training_readiness, load_configs


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_suite(path: Path) -> dict:
    suite = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"name", "output_root", "status_path", "suite_log", "experiments"}
    missing = required - set(suite)
    if missing:
        raise RuntimeError(f"Dense ablation configuration lacks {sorted(missing)}")
    ids = [str(item["id"]) for item in suite["experiments"]]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Dense ablation experiment IDs must be unique")
    return suite


def resolve(relative: str) -> Path:
    return (ROOT / relative).resolve()


def validate_suite(suite: dict) -> None:
    expected = {
        ("standard_spectral", 2026),
        ("standard_spectral", 2027),
        ("standard_spectral", 2028),
        ("imbalance_spectral", 2027),
        ("imbalance_spectral", 2028),
        ("imbalance_spectral", 2026),
        ("terrain_full", 2026),
        ("terrain_full", 2027),
        ("terrain_full", 2028),
    }
    observed = {(str(item["variant"]), int(item["seed"])) for item in suite["experiments"]}
    if not observed or not observed <= expected:
        raise RuntimeError(
            f"Unexpected experiment matrix. Missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    for item in suite["experiments"]:
        for field in ("data_config", "model_config", "train_config"):
            path = resolve(str(item[field]))
            if not path.is_file():
                raise FileNotFoundError(f"{item['id']}: missing {field}: {path}")
    for item in suite.get("existing_runs", []):
        checkpoint = resolve(str(item["output_dir"])) / "best.pt"
        threshold = checkpoint.with_name("validation_threshold.json")
        if not checkpoint.is_file() or not threshold.is_file():
            raise FileNotFoundError(
                f"Existing run {item['id']} is incomplete: {checkpoint.parent}"
            )


def smoke_test(suite: dict) -> dict:
    """Exercise data, model, loss, and backward paths for every unique variant."""
    validate_suite(suite)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records: list[dict] = []
    seen: set[str] = set()
    for item in suite["experiments"]:
        variant = str(item["variant"])
        if variant in seen:
            continue
        seen.add(variant)
        root, data_config, train_config = load_configs(
            resolve(str(item["data_config"])), resolve(str(item["train_config"]))
        )
        del root
        train_config["seed"] = int(item["seed"])
        readiness = check_primary_training_readiness(ROOT, data_config, train_config)
        if not readiness["ready"]:
            raise RuntimeError(f"{variant} failed readiness: {readiness}")
        model_config = yaml.safe_load(
            resolve(str(item["model_config"])).read_text(encoding="utf-8")
        )
        dataset = KoshiPositiveDataset(
            resolve(str(item["data_config"])),
            split="train",
            normalize=True,
            augment=False,
            crop_size=128,
            positive_crop_probability=float(
                train_config.get("positive_crop_probability", 0.0)
            ),
        )
        sample = dataset[0]
        image = sample["image"].unsqueeze(0).to(device)
        target = sample["target"].unsqueeze(0).to(device)
        supervision = sample["supervision"].unsqueeze(0).to(device)
        model = build_model(
            architecture=str(model_config["architecture"]),
            pretrained=False,
            decoder_dropout=float(model_config.get("decoder_dropout", 0.0)),
            input_channels=int(model_config["input_channels"]),
        ).to(device)
        model.train()
        logits = model(image)
        loss = configured_segmentation_loss(
            logits, target, supervision, train_config["loss"]
        )
        if logits.shape != target.shape or not torch.isfinite(loss):
            raise RuntimeError(
                f"{variant}: invalid logits/loss: {tuple(logits.shape)}, {loss}"
            )
        loss.backward()
        finite_gradients = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if not finite_gradients:
            raise RuntimeError(f"{variant}: non-finite gradient detected")
        records.append(
            {
                "variant": variant,
                "device": str(device),
                "input_shape": list(image.shape),
                "output_shape": list(logits.shape),
                "loss_name": str(train_config["loss"]["name"]),
                "loss": float(loss.detach()),
                "finite_gradients": finite_gradients,
            }
        )
        del model, logits, loss, image, target, supervision, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[smoke] {variant}: PASS")
    report = {
        "checked_at": stamp(),
        "status": "passed",
        "planned_experiments": len(suite["experiments"]),
        "existing_runs": len(suite.get("existing_runs", [])),
        "variant_checks": records,
    }
    output = resolve(str(suite["output_root"])) / "smoke_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Smoke-test report: {output}")
    return report


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def log_line(path: Path, message: str) -> None:
    line = f"[{stamp()}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def run_process(command: list[str], raw_log: Path, prefix: str) -> None:
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    with raw_log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                stripped = line.strip()
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    if stripped:
                        print(f"[{prefix}] {stripped}", flush=True)
                    continue
                if "epoch" in record:
                    print(
                        f"[{prefix}] epoch {int(record['epoch']):03d} | "
                        f"train loss {float(record['train_loss']):.4f} | "
                        f"val Dice {float(record['validation_best_dice']):.4f} | "
                        f"threshold {float(record['validation_best_threshold']):.3f}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            process.terminate()
            process.wait(timeout=30)
            raise
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


def run_suite(suite: dict) -> None:
    validate_suite(suite)
    output_root = resolve(str(suite["output_root"]))
    status_path = resolve(str(suite["status_path"]))
    suite_log = resolve(str(suite["suite_log"]))
    output_root.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    total = len(suite["experiments"])
    log_line(suite_log, f"Dense ablation started with {total} planned trainings.")

    for index, item in enumerate(suite["experiments"], start=1):
        experiment_id = str(item["id"])
        output_dir = output_root / experiment_id
        history = output_dir / "history.csv"
        checkpoint = output_dir / "best.pt"
        threshold = output_dir / "validation_threshold.json"
        if history.is_file() and checkpoint.is_file() and threshold.is_file():
            completed.append(experiment_id)
            log_line(suite_log, f"[{index}/{total}] SKIP completed {experiment_id}.")
            continue
        if history.exists() and not checkpoint.exists():
            raise RuntimeError(
                f"Partial run found at {output_dir}. It was preserved; move or rename "
                "that directory before restarting this experiment."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        status = {
            "updated_at": stamp(),
            "state": "running",
            "current_experiment": experiment_id,
            "position": index,
            "total": total,
            "completed": completed,
        }
        write_status(status_path, status)
        log_line(
            suite_log,
            f"[{index}/{total}] START {experiment_id} ({item['variant']}, seed {item['seed']}).",
        )

        if not checkpoint.is_file():
            train_command = [
                sys.executable,
                "-u",
                "-m",
                "nepal_cnn.training.train",
                "--data-config",
                str(resolve(str(item["data_config"]))),
                "--model-config",
                str(resolve(str(item["model_config"]))),
                "--train-config",
                str(resolve(str(item["train_config"]))),
                "--output-dir",
                str(output_dir),
                "--seed",
                str(int(item["seed"])),
            ]
            run_process(train_command, output_dir / "stdout.log", experiment_id)

        threshold_config = suite.get("threshold", {})
        threshold_command = [
            sys.executable,
            "-u",
            "-m",
            "nepal_cnn.evaluation.threshold",
            "--data-config",
            str(resolve(str(item["data_config"]))),
            "--checkpoint",
            str(checkpoint),
            "--step",
            str(float(threshold_config.get("step", 0.005))),
            "--batch-size",
            str(int(threshold_config.get("batch_size", 2))),
            "--device",
            str(threshold_config.get("device", "cuda")),
            "--output",
            str(threshold),
        ]
        run_process(threshold_command, output_dir / "threshold.log", experiment_id)
        completed.append(experiment_id)
        log_line(suite_log, f"[{index}/{total}] COMPLETE {experiment_id}.")

    write_status(
        status_path,
        {
            "updated_at": stamp(),
            "state": "completed",
            "current_experiment": None,
            "position": total,
            "total": total,
            "completed": completed,
        },
    )
    log_line(suite_log, "Dense ablation completed successfully.")


def run_tests(suite: dict) -> None:
    """Evaluate every selected dense run after freezing validation thresholds."""
    from nepal_cnn.utils.configuration import run_workflow
    output = resolve(suite["output_root"])
    entries = []
    for item in suite["experiments"]:
        run_dir = output / item["id"]
        checkpoint = run_dir / "best.pt"
        threshold = run_dir / "validation_threshold.json"
        entries.append({"id": item["id"], "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                       "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       "threshold_report": threshold.relative_to(ROOT).as_posix(),
                       "threshold_sha256": hashlib.sha256(threshold.read_bytes()).hexdigest(),
                       "data_config": item["data_config"]})
    path = output / "frozen_test_manifest.json"
    if path.exists() and json.loads(path.read_text()) != entries:
        raise ValueError("Dense test inputs changed from the frozen manifest")
    path.write_text(json.dumps(entries, indent=2)+"\n")
    for entry in entries:
        target = output / entry["id"] / "test_metrics_ablation.json"
        if target.exists():
            continue
        run_workflow("evaluate", ["--checkpoint", entry["checkpoint"], "--threshold-report", entry["threshold_report"],
                                 "--data-config", entry["data_config"], "--split", "test", "--output", target])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--test", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    suite = load_suite(config_path.resolve())
    if args.smoke_test:
        smoke_test(suite)
    elif args.test:
        run_tests(suite)
    else:
        run_suite(suite)


if __name__ == "__main__":
    main()
