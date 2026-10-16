"""Shared recovery training and validation for physically pruned models."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from nepal_cnn.data.dataset import KoshiPositiveDataset
from nepal_cnn.training.losses import configured_segmentation_loss
from nepal_cnn.utils.logging import get_logger


LOGGER = get_logger()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _balanced_sampler(
    dataset: KoshiPositiveDataset,
    positive_probability: float,
    generator: torch.Generator,
) -> WeightedRandomSampler:
    sample_types = dataset.rows["sample_type"].to_numpy()
    positive = sample_types == "positive"
    negative = sample_types == "negative"
    if not positive.any() or not negative.any():
        raise RuntimeError("Recovery sampling requires positive and negative patches")
    weights = np.empty(len(dataset), dtype=np.float64)
    weights[positive] = positive_probability / positive.sum()
    weights[negative] = (1.0 - positive_probability) / negative.sum()
    return WeightedRandomSampler(
        torch.from_numpy(weights),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def _optimizer_groups(
    model: nn.Module,
    encoder_learning_rate: float,
    decoder_learning_rate: float,
) -> list[dict]:
    encoder_parameters = {
        id(parameter)
        for module in model.encoder_modules()
        for parameter in module.parameters()
    }
    return [
        {
            "params": [
                parameter
                for parameter in model.parameters()
                if id(parameter) in encoder_parameters
            ],
            "lr": encoder_learning_rate,
            "name": "encoder",
        },
        {
            "params": [
                parameter
                for parameter in model.parameters()
                if id(parameter) not in encoder_parameters
            ],
            "lr": decoder_learning_rate,
            "name": "decoder",
        },
    ]


def build_recovery_loaders(
    data_config: Path,
    dense_train_config: dict,
    recovery_config: dict,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Build the same class-balanced/augmented training and fixed validation data."""
    train_dataset = KoshiPositiveDataset(
        data_config,
        split="train",
        normalize=True,
        augment=True,
        crop_size=int(recovery_config["crop_size"]),
        positive_crop_probability=float(
            dense_train_config.get("positive_crop_probability", 0.9)
        ),
        positive_center_sampling=str(
            dense_train_config.get("positive_center_sampling", "pixel")
        ),
        minimum_component_pixels=int(
            dense_train_config.get("minimum_component_pixels", 1)
        ),
        component_area_power=float(
            dense_train_config.get("component_area_power", -0.5)
        ),
        radiometric_gain=float(dense_train_config.get("radiometric_gain", 0.0)),
        radiometric_offset=float(dense_train_config.get("radiometric_offset", 0.0)),
        radiometric_noise_std=float(
            dense_train_config.get("radiometric_noise_std", 0.0)
        ),
    )
    validation_dataset = KoshiPositiveDataset(
        data_config,
        split="validation",
        normalize=True,
        augment=False,
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = _balanced_sampler(
        train_dataset,
        float(dense_train_config.get("positive_sample_probability", 0.8)),
        generator,
    )
    workers = int(recovery_config.get("num_workers", 4))
    common = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _seed_worker,
        "generator": generator,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(recovery_config["batch_size"]),
        sampler=sampler,
        shuffle=False,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(recovery_config["validation_batch_size"]),
        shuffle=False,
        **common,
    )
    return train_loader, validation_loader


def _threshold_metrics(
    true_positive: torch.Tensor,
    false_positive: torch.Tensor,
    false_negative: torch.Tensor,
    thresholds: torch.Tensor,
) -> dict:
    dice = 2 * true_positive / (2 * true_positive + false_positive + false_negative).clamp_min(1)
    best = int(torch.argmax(dice))
    tp = true_positive[best]
    fp = false_positive[best]
    fn = false_negative[best]
    return {
        "best_threshold": float(thresholds[best]),
        "best_dice": float(dice[best]),
        "best_iou": float(tp / (tp + fp + fn).clamp_min(1)),
        "best_precision": float(tp / (tp + fp).clamp_min(1)),
        "best_recall": float(tp / (tp + fn).clamp_min(1)),
    }


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_config: dict,
    thresholds: list[float],
    amp_enabled: bool,
) -> dict:
    """Evaluate only the validation split and select a validation threshold."""
    model.eval()
    threshold_tensor = torch.tensor(thresholds, device=device, dtype=torch.float32)
    tp = torch.zeros(len(thresholds), device=device, dtype=torch.float64)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)
    loss_sum = 0.0
    sample_count = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        supervision = batch["supervision"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(image)
            loss = configured_segmentation_loss(logits, target, supervision, loss_config)
        probability = logits.sigmoid().unsqueeze(0)
        prediction = probability >= threshold_tensor[:, None, None, None, None]
        valid = supervision.bool().unsqueeze(0)
        truth = target.bool().unsqueeze(0)
        tp += (prediction & truth & valid).sum(dim=(1, 2, 3, 4))
        fp += (prediction & ~truth & valid).sum(dim=(1, 2, 3, 4))
        fn += (~prediction & truth & valid).sum(dim=(1, 2, 3, 4))
        count = int(image.shape[0])
        loss_sum += float(loss) * count
        sample_count += count
    return {
        "loss": loss_sum / max(sample_count, 1),
        **_threshold_metrics(tp, fp, fn, threshold_tensor),
    }


def recover_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    dense_train_config: dict,
    recovery_config: dict,
    epochs: int,
    context: dict,
) -> tuple[nn.Module, dict, list[dict]]:
    """Run a fixed recovery budget and restore validation-best raw weights."""
    if epochs < 1:
        raise ValueError("Recovery epochs must be positive")
    encoder_lr = float(recovery_config["encoder_learning_rate"])
    decoder_lr = float(recovery_config["decoder_learning_rate"])
    optimizer = torch.optim.AdamW(
        _optimizer_groups(model, encoder_lr, decoder_lr),
        weight_decay=float(recovery_config["weight_decay"]),
    )
    amp_enabled = bool(recovery_config.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    thresholds = [float(value) for value in dense_train_config["thresholds"]]
    loss_config = dense_train_config["loss"]
    warmup = int(recovery_config.get("warmup_epochs", 2))
    minimum_ratio = float(recovery_config.get("minimum_learning_rate_ratio", 0.01))
    clip_norm = float(recovery_config.get("gradient_clip_norm", 1.0))
    history: list[dict] = []
    best_metrics: dict | None = None
    best_state: dict | None = None
    best_dice = -1.0

    for epoch in range(1, epochs + 1):
        if epoch <= warmup:
            multiplier = epoch / max(warmup, 1)
        else:
            progress = (epoch - warmup - 1) / max(epochs - warmup - 1, 1)
            multiplier = minimum_ratio + (1 - minimum_ratio) * 0.5 * (
                1 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
            )
        optimizer.param_groups[0]["lr"] = encoder_lr * multiplier
        optimizer.param_groups[1]["lr"] = decoder_lr * multiplier
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_samples = 0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            supervision = batch["supervision"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(image)
                loss = configured_segmentation_loss(
                    logits, target, supervision, loss_config
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            count = int(image.shape[0])
            train_loss += float(loss.detach()) * count
            train_samples += count
        validation = evaluate_validation(
            model,
            validation_loader,
            device,
            loss_config,
            thresholds,
            amp_enabled,
        )
        record = {
            **context,
            "event": "recovery_epoch",
            "epoch": epoch,
            "encoder_learning_rate": optimizer.param_groups[0]["lr"],
            "decoder_learning_rate": optimizer.param_groups[1]["lr"],
            "train_loss": train_loss / max(train_samples, 1),
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(record)
        best_dice = max(best_dice, float(validation["best_dice"]))
        LOGGER.info(
            "%s | %s | target %d%% | recovery %02d/%02d | "
            "train loss %.4f | val Dice %.4f | best %.4f | threshold %.3f",
            context["model"],
            context["method"],
            round(100 * context["nominal_mac_reduction"]),
            epoch,
            epochs,
            record["train_loss"],
            validation["best_dice"],
            best_dice,
            validation["best_threshold"],
        )
        if best_metrics is None or validation["best_dice"] > best_metrics["best_dice"]:
            best_metrics = {**validation, "epoch": epoch}
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
    assert best_metrics is not None and best_state is not None
    model.load_state_dict(best_state)
    return model, best_metrics, history
