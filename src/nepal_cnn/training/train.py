"""Train a conventional dense U-Net after the background-domain gate passes."""

from __future__ import annotations

from nepal_cnn.utils.paths import PROJECT_ROOT as ROOT

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from nepal_cnn.data.dataset import KoshiPositiveDataset
from nepal_cnn.training.losses import configured_segmentation_loss
from nepal_cnn.evaluation.metrics import BinarySegmentationMeter
from nepal_cnn.models.architectures import build_model
from nepal_cnn.training.readiness import check_primary_training_readiness, load_configs


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_train_sampler(
    dataset: KoshiPositiveDataset,
    positive_probability: float | None,
    generator: torch.Generator,
) -> WeightedRandomSampler | None:
    if positive_probability is None:
        return None
    sample_types = dataset.rows["sample_type"].to_numpy()
    positive = sample_types == "positive"
    negative = sample_types == "negative"
    if not positive.any() or not negative.any():
        raise RuntimeError("Balanced sampling requires positive and negative patches")
    weights = np.empty(len(dataset), dtype=np.float64)
    weights[positive] = positive_probability / positive.sum()
    weights[negative] = (1.0 - positive_probability) / negative.sum()
    return WeightedRandomSampler(
        torch.from_numpy(weights),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def optimizer_groups(
    model: torch.nn.Module,
    encoder_learning_rate: float,
    decoder_learning_rate: float,
) -> list[dict]:
    if not hasattr(model, "encoder_modules"):
        raise TypeError(f"{type(model).__name__} does not declare encoder_modules()")
    encoder_modules = model.encoder_modules()
    encoder_parameters = {
        id(parameter)
        for module in encoder_modules
        for parameter in module.parameters()
    }
    return [
        {
            "params": [
                parameter for parameter in model.parameters()
                if id(parameter) in encoder_parameters
            ],
            "lr": encoder_learning_rate,
            "initial_lr": encoder_learning_rate,
            "name": "encoder",
        },
        {
            "params": [
                parameter for parameter in model.parameters()
                if id(parameter) not in encoder_parameters
            ],
            "lr": decoder_learning_rate,
            "initial_lr": decoder_learning_rate,
            "name": "decoder",
        },
    ]


def scheduler_multiplier(
    epoch_zero_based: int,
    total_epochs: int,
    warmup_epochs: int,
    minimum_ratio: float,
) -> float:
    if warmup_epochs > 0 and epoch_zero_based < warmup_epochs:
        return (epoch_zero_based + 1) / warmup_epochs
    progress = (epoch_zero_based - warmup_epochs) / max(total_epochs - warmup_epochs - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def update_threshold_counts(
    logits: torch.Tensor,
    target: torch.Tensor,
    supervision: torch.Tensor,
    thresholds: torch.Tensor,
    true_positive: torch.Tensor,
    false_positive: torch.Tensor,
    false_negative: torch.Tensor,
) -> None:
    probability = logits.sigmoid().unsqueeze(0)
    prediction = probability >= thresholds[:, None, None, None, None]
    valid = supervision.bool().unsqueeze(0)
    truth = target.bool().unsqueeze(0)
    true_positive += (prediction & truth & valid).sum(dim=(1, 2, 3, 4))
    false_positive += (prediction & ~truth & valid).sum(dim=(1, 2, 3, 4))
    false_negative += (~prediction & truth & valid).sum(dim=(1, 2, 3, 4))


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    loss_config: dict,
    thresholds: list[float] | None = None,
    gradient_clip_norm: float | None = None,
    gradient_accumulation_steps: int = 1,
    ema_model: AveragedModel | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    meters = {
        "": BinarySegmentationMeter(),
        "positive_patch_": BinarySegmentationMeter(),
        "negative_patch_": BinarySegmentationMeter(),
    }
    threshold_tensor = (
        torch.tensor(thresholds, device=device, dtype=torch.float32)
        if thresholds is not None else None
    )
    if threshold_tensor is not None:
        threshold_tp = torch.zeros(len(thresholds), device=device, dtype=torch.float64)
        threshold_fp = torch.zeros_like(threshold_tp)
        threshold_fn = torch.zeros_like(threshold_tp)
    loss_sum = 0.0
    batches = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        supervision = batch["supervision"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type, enabled=amp_enabled
        ):
            logits = model(image)
            loss = configured_segmentation_loss(logits, target, supervision, loss_config)
        if training:
            scaler.scale(loss / gradient_accumulation_steps).backward()
            update_now = (
                (batch_index + 1) % gradient_accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if update_now:
                if gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema_model is not None:
                    ema_model.update_parameters(model)

        detached = logits.detach()
        meters[""].update(detached, target, supervision)
        sample_types = list(batch["sample_type"])
        for sample_type, prefix in (
            ("positive", "positive_patch_"),
            ("negative", "negative_patch_"),
        ):
            indices = [index for index, value in enumerate(sample_types) if value == sample_type]
            if indices:
                meters[prefix].update(
                    detached[indices], target[indices], supervision[indices]
                )
        if threshold_tensor is not None:
            update_threshold_counts(
                detached,
                target,
                supervision,
                threshold_tensor,
                threshold_tp,
                threshold_fp,
                threshold_fn,
            )
        loss_sum += float(loss.detach())
        batches += 1

    metrics = meters[""].compute()
    for prefix in ("positive_patch_", "negative_patch_"):
        metrics.update(
            {f"{prefix}{key}": value for key, value in meters[prefix].compute().items()}
        )
    metrics["loss"] = loss_sum / max(batches, 1)
    if threshold_tensor is not None:
        dice = 2 * threshold_tp / (2 * threshold_tp + threshold_fp + threshold_fn).clamp_min(1)
        best = int(torch.argmax(dice))
        metrics.update(
            {
                "best_threshold": float(threshold_tensor[best]),
                "best_dice": float(dice[best]),
                "best_iou": float(
                    threshold_tp[best]
                    / (threshold_tp[best] + threshold_fp[best] + threshold_fn[best]).clamp_min(1)
                ),
                "best_precision": float(
                    threshold_tp[best] / (threshold_tp[best] + threshold_fp[best]).clamp_min(1)
                ),
                "best_recall": float(
                    threshold_tp[best] / (threshold_tp[best] + threshold_fn[best]).clamp_min(1)
                ),
            }
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--output-dir", default="generated/experiments/dense_models/baseline")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Optional pilot override of configured epochs"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional reproducible-run seed override"
    )
    args = parser.parse_args()

    root, data_config, train_config = load_configs(args.data_config, args.train_config)
    if args.seed is not None:
        train_config["seed"] = int(args.seed)
    model_config = yaml.safe_load(Path(args.model_config).resolve().read_text(encoding="utf-8"))
    readiness = check_primary_training_readiness(root, data_config, train_config)
    if not readiness["ready"]:
        raise RuntimeError(
            "Baseline training blocked by scientific readiness gate:\n"
            + json.dumps(readiness, indent=2)
        )

    seed = int(train_config["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(train_config["amp"] and device.type == "cuda")
    generator = torch.Generator().manual_seed(seed)
    crop_size = train_config.get("crop_size")
    train_dataset = KoshiPositiveDataset(
        args.data_config,
        split="train",
        normalize=True,
        augment=True,
        crop_size=int(crop_size) if crop_size is not None else None,
        positive_crop_probability=float(
            train_config.get("positive_crop_probability", 0.0)
        ),
        positive_center_sampling=str(
            train_config.get("positive_center_sampling", "pixel")
        ),
        minimum_component_pixels=int(
            train_config.get("minimum_component_pixels", 1)
        ),
        component_area_power=float(
            train_config.get("component_area_power", -0.5)
        ),
        radiometric_gain=float(train_config.get("radiometric_gain", 0.0)),
        radiometric_offset=float(train_config.get("radiometric_offset", 0.0)),
        radiometric_noise_std=float(train_config.get("radiometric_noise_std", 0.0)),
    )
    validation_dataset = KoshiPositiveDataset(
        args.data_config, split="validation", normalize=True
    )
    sampler = make_train_sampler(
        train_dataset,
        (
            float(train_config["positive_sample_probability"])
            if "positive_sample_probability" in train_config else None
        ),
        generator,
    )
    loader_options = {
        "num_workers": int(train_config["num_workers"]),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": int(train_config["num_workers"]) > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(train_config.get("validation_batch_size", train_config["batch_size"])),
        shuffle=False,
        **loader_options,
    )

    model = build_model(
        architecture=model_config["architecture"],
        pretrained=bool(
            model_config.get(
                "pretrained", model_config.get("encoder_weights") == "imagenet"
            )
        ),
        decoder_dropout=float(model_config.get("decoder_dropout", 0.0)),
        input_channels=int(model_config.get("input_channels", 3)),
    ).to(device)
    encoder_lr = float(
        train_config.get("encoder_learning_rate", train_config.get("learning_rate", 1e-4))
    )
    decoder_lr = float(
        train_config.get("decoder_learning_rate", train_config.get("learning_rate", 1e-4))
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, encoder_lr, decoder_lr),
        weight_decay=float(train_config["weight_decay"]),
    )
    ema_decay = float(train_config.get("ema_decay", 0.0))
    ema_model = (
        AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True)
        if ema_decay > 0
        else None
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    if history_path.exists():
        raise FileExistsError(f"Refusing to append to existing history: {history_path}")
    epochs = int(args.epochs if args.epochs is not None else train_config["epochs"])
    scheduler_config = train_config.get("scheduler", {"name": "none"})
    best_dice = -1.0
    stale_epochs = 0
    thresholds = [float(value) for value in train_config.get("thresholds", [0.5])]

    run_manifest = {
        "data_config": data_config,
        "model_config": model_config,
        "train_config": train_config,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "train_patches": len(train_dataset),
        "validation_patches": len(validation_dataset),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "effective_batch_size": int(train_config["batch_size"])
        * int(train_config.get("gradient_accumulation_steps", 1)),
        "checkpoint_weights": "exponential_moving_average" if ema_model is not None else "raw",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )

    for epoch in range(1, epochs + 1):
        if scheduler_config["name"] == "cosine_with_warmup":
            multiplier = scheduler_multiplier(
                epoch - 1,
                epochs,
                int(scheduler_config.get("warmup_epochs", 0)),
                float(scheduler_config.get("minimum_learning_rate_ratio", 0.01)),
            )
            optimizer.param_groups[0]["lr"] = encoder_lr * multiplier
            optimizer.param_groups[1]["lr"] = decoder_lr * multiplier
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            amp_enabled,
            train_config["loss"],
            gradient_clip_norm=(
                float(train_config["gradient_clip_norm"])
                if "gradient_clip_norm" in train_config else None
            ),
            gradient_accumulation_steps=int(
                train_config.get("gradient_accumulation_steps", 1)
            ),
            ema_model=ema_model,
        )
        validation_model = ema_model.module if ema_model is not None else model
        validation_metrics = run_epoch(
            validation_model,
            validation_loader,
            device,
            None,
            scaler,
            amp_enabled,
            train_config["loss"],
            thresholds=thresholds,
        )
        row = {
            "epoch": epoch,
            "encoder_learning_rate": optimizer.param_groups[0]["lr"],
            "decoder_learning_rate": optimizer.param_groups[1]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        checkpoint = {
            "epoch": epoch,
            "model": validation_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "metrics": row,
            "data_config": data_config,
            "model_config": model_config,
            "train_config": train_config,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        selection_dice = float(validation_metrics["best_dice"])
        if selection_dice > best_dice:
            best_dice = selection_dice
            stale_epochs = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            stale_epochs += 1
        print(json.dumps(row), flush=True)
        if (
            epoch >= int(train_config.get("minimum_epochs", 0))
            and stale_epochs >= int(train_config["early_stopping_patience"])
        ):
            break


if __name__ == "__main__":
    main()
