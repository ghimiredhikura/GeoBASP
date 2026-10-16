"""Manifest-driven access to exact Sentinel-2 windows and cached masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from rasterio.enums import Resampling
from rasterio.windows import Window, bounds as window_bounds, from_bounds
from scipy import ndimage
from torch.utils.data import Dataset


class KoshiPositiveDataset(Dataset):
    """Positive patch reader; background pixels deliberately remain unsupervised."""

    def __init__(
        self,
        config_path: str | Path = "configs/data.yaml",
        split: str | None = None,
        normalize: bool = False,
        augment: bool = False,
        crop_size: int | None = None,
        positive_crop_probability: float = 0.0,
        positive_center_sampling: str = "pixel",
        minimum_component_pixels: int = 1,
        component_area_power: float = -0.5,
        radiometric_gain: float = 0.0,
        radiometric_offset: float = 0.0,
        radiometric_noise_std: float = 0.0,
    ):
        config_path = Path(config_path).resolve()
        self.cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.root = (config_path.parent.parent / self.cfg.get("project_root", ".")).resolve()
        if split is None:
            manifest = self.root / self.cfg["output_root"] / "manifests" / "positive_patches.csv"
        else:
            manifest = self.root / self.cfg.get(
                "training_manifest", "data/processed/splits/primary_patch_split.csv"
            )
        self.rows = pd.read_csv(manifest)
        if split is not None:
            if split not in {"train", "validation", "test", "excluded_guard"}:
                raise ValueError(f"Unknown split: {split}")
            self.rows = self.rows[self.rows["split"] == split].reset_index(drop=True)
        self.mask_column = self.cfg.get("training_mask_column", "mask_path")
        required = {"patch_id", "scene", "row_off", "col_off", self.mask_column}
        missing = required - set(self.rows.columns)
        if missing:
            raise RuntimeError(f"Manifest lacks fields: {sorted(missing)}")

        self.normalize = normalize
        self.augment = augment
        self.crop_size = crop_size
        self.positive_crop_probability = positive_crop_probability
        self.positive_center_sampling = positive_center_sampling
        self.minimum_component_pixels = int(minimum_component_pixels)
        self.component_area_power = float(component_area_power)
        self.radiometric_gain = float(radiometric_gain)
        self.radiometric_offset = float(radiometric_offset)
        self.radiometric_noise_std = float(radiometric_noise_std)
        if crop_size is not None and (crop_size <= 0 or crop_size > int(self.cfg["patch_size_px"])):
            raise ValueError(f"Invalid crop size: {crop_size}")
        if not 0.0 <= positive_crop_probability <= 1.0:
            raise ValueError("positive_crop_probability must be in [0, 1]")
        if positive_center_sampling not in {"pixel", "component_balanced"}:
            raise ValueError(
                "positive_center_sampling must be 'pixel' or 'component_balanced'"
            )
        if self.minimum_component_pixels < 1:
            raise ValueError("minimum_component_pixels must be positive")
        self.terrain_features = list(self.cfg.get("terrain_features", []))
        for feature in self.terrain_features:
            required_feature_fields = {"name", "manifest_column"}
            if not required_feature_fields.issubset(feature):
                raise RuntimeError(
                    "Terrain features require name and manifest_column: "
                    f"{feature}"
                )
            if feature["manifest_column"] not in self.rows.columns:
                raise RuntimeError(
                    f"Manifest lacks terrain column {feature['manifest_column']}"
                )
        self.channel_names = list(self.cfg["input_bands"]) + [
            str(feature["name"]) for feature in self.terrain_features
        ]
        self.normalization_mean: np.ndarray | None = None
        self.normalization_std: np.ndarray | None = None
        if normalize:
            stats_path = self.root / self.cfg["normalization_stats"]
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            channels = stats["channels"]
            stat_bands = [channel["band"] for channel in channels]
            if stat_bands != self.channel_names:
                raise RuntimeError(
                    f"Normalization bands {stat_bands} do not match input bands "
                    f"{self.channel_names}"
                )
            if float(stats["reflectance_scale"]) != float(self.cfg["reflectance_scale"]) or float(
                stats["reflectance_offset"]
            ) != float(self.cfg["reflectance_offset"]):
                raise RuntimeError("Normalization radiometry does not match data configuration")
            self.normalization_mean = np.asarray(
                [channel["mean"] for channel in channels], dtype=np.float32
            )[:, None, None]
            self.normalization_std = np.asarray(
                [channel["std"] for channel in channels], dtype=np.float32
            )[:, None, None]
            if not np.all(self.normalization_std > 0):
                raise RuntimeError("Normalization standard deviations must be positive")

    def __len__(self) -> int:
        return len(self.rows)

    def _read_band(self, scene_dir: Path, band: str, window: Window,
                   bounds: tuple[float, float, float, float]) -> np.ndarray:
        with rasterio.open(scene_dir / f"{band}.tif") as src:
            source_window = from_bounds(*bounds, transform=src.transform)
            array = src.read(
                1,
                window=source_window,
                out_shape=(int(window.height), int(window.width)),
                boundless=True,
                fill_value=0,
                resampling=Resampling.bilinear,
            ).astype(np.float32)
        nodata = array == 0
        array = array * float(self.cfg["reflectance_scale"]) + float(
            self.cfg["reflectance_offset"]
        )
        array[nodata] = 0.0
        return array

    def _read_terrain(self, row: pd.Series, shape: tuple[int, int]) -> list[np.ndarray]:
        arrays: list[np.ndarray] = []
        for feature in self.terrain_features:
            path = self.root / row[feature["manifest_column"]]
            array = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
            if array.shape != shape:
                raise RuntimeError(
                    f"Terrain feature {path} has shape {array.shape}; expected {shape}"
                )
            arrays.append(array)
        return arrays

    def _select_positive_coordinate(
        self,
        target: torch.Tensor,
        supervision: torch.Tensor,
    ) -> tuple[int, int] | None:
        mask = (target[0].bool() & supervision[0]).numpy()
        if not mask.any():
            return None
        if self.positive_center_sampling == "pixel":
            coordinates = np.argwhere(mask)
            coordinate = coordinates[int(np.random.randint(len(coordinates)))]
            return int(coordinate[0]), int(coordinate[1])

        labels, component_count = ndimage.label(
            mask, structure=np.ones((3, 3), dtype=np.uint8)
        )
        component_ids = np.arange(1, component_count + 1, dtype=np.int32)
        areas = np.bincount(labels.ravel(), minlength=component_count + 1)[1:]
        eligible = areas >= self.minimum_component_pixels
        if not eligible.any():
            coordinates = np.argwhere(mask)
            coordinate = coordinates[int(np.random.randint(len(coordinates)))]
            return int(coordinate[0]), int(coordinate[1])
        component_ids = component_ids[eligible]
        areas = areas[eligible].astype(np.float64)
        weights = np.power(areas, self.component_area_power)
        weights /= weights.sum()
        selected = int(np.random.choice(component_ids, p=weights))
        coordinates = np.argwhere(labels == selected)
        coordinate = coordinates[int(np.random.randint(len(coordinates)))]
        return int(coordinate[0]), int(coordinate[1])

    def __getitem__(self, index: int) -> dict:
        row = self.rows.iloc[index]
        scene_dir = self.root / self.cfg["sentinel_root"] / row.scene
        window = Window(int(row.col_off), int(row.row_off), int(row.width), int(row.height))
        with rasterio.open(scene_dir / f"{self.cfg['reference_band']}.tif") as ref:
            bounds = window_bounds(window, ref.transform)
        image = np.stack(
            [self._read_band(scene_dir, band, window, bounds) for band in self.cfg["input_bands"]]
        )
        with rasterio.open(self.root / row[self.mask_column]) as src:
            masks = src.read()
        if self.augment and (
            self.radiometric_gain > 0
            or self.radiometric_offset > 0
            or self.radiometric_noise_std > 0
        ):
            observed = masks[1].astype(bool)
            gain = np.random.uniform(
                1.0 - self.radiometric_gain,
                1.0 + self.radiometric_gain,
                size=(image.shape[0], 1, 1),
            ).astype(np.float32)
            offset = np.random.uniform(
                -self.radiometric_offset,
                self.radiometric_offset,
                size=(image.shape[0], 1, 1),
            ).astype(np.float32)
            image = image * gain + offset
            if self.radiometric_noise_std > 0:
                image += np.random.normal(
                    0.0, self.radiometric_noise_std, size=image.shape
                ).astype(np.float32)
            image = np.clip(image, 0.0, None)
            image[:, ~observed] = 0.0
        terrain = self._read_terrain(row, image.shape[-2:])
        if terrain:
            image = np.concatenate([image, np.stack(terrain)], axis=0)
        if self.normalize:
            image = (image - self.normalization_mean) / self.normalization_std
            image[:, masks[1] == 0] = 0.0
        image_tensor = torch.from_numpy(image)
        target_tensor = torch.from_numpy(masks[0:1].astype(np.float32))
        observation_tensor = torch.from_numpy(masks[1:2].astype(bool))
        supervision_tensor = torch.from_numpy(masks[2:3].astype(bool))
        sensitivity_tensor = torch.from_numpy(
            (masks[3:4] if masks.shape[0] >= 4 else masks[2:3]).astype(bool)
        )
        if self.crop_size is not None:
            height, width = target_tensor.shape[-2:]
            crop = self.crop_size
            positive_coordinate = self._select_positive_coordinate(
                target_tensor, supervision_tensor
            )
            use_positive_center = (
                positive_coordinate is not None
                and bool(torch.rand(()) < self.positive_crop_probability)
            )
            if use_positive_center:
                jitter = max(crop // 6, 1)
                center_y = int(positive_coordinate[0]) + int(
                    torch.randint(-jitter, jitter + 1, ()).item()
                )
                center_x = int(positive_coordinate[1]) + int(
                    torch.randint(-jitter, jitter + 1, ()).item()
                )
                top = min(max(center_y - crop // 2, 0), height - crop)
                left = min(max(center_x - crop // 2, 0), width - crop)
            else:
                top = int(torch.randint(0, height - crop + 1, ()).item())
                left = int(torch.randint(0, width - crop + 1, ()).item())
            tensors = [
                image_tensor,
                target_tensor,
                observation_tensor,
                supervision_tensor,
                sensitivity_tensor,
            ]
            tensors = [
                tensor[..., top : top + crop, left : left + crop] for tensor in tensors
            ]
            (
                image_tensor,
                target_tensor,
                observation_tensor,
                supervision_tensor,
                sensitivity_tensor,
            ) = tensors
        if self.augment:
            rotations = int(torch.randint(0, 4, ()).item())
            tensors = [
                image_tensor,
                target_tensor,
                observation_tensor,
                supervision_tensor,
                sensitivity_tensor,
            ]
            tensors = [torch.rot90(tensor, rotations, dims=(-2, -1)) for tensor in tensors]
            if bool(torch.rand(()) < 0.5):
                tensors = [torch.flip(tensor, dims=(-1,)) for tensor in tensors]
            (
                image_tensor,
                target_tensor,
                observation_tensor,
                supervision_tensor,
                sensitivity_tensor,
            ) = tensors
        return {
            "image": image_tensor,
            "target": target_tensor,
            "observation": observation_tensor,
            "supervision": supervision_tensor,
            "sensitivity_supervision": sensitivity_tensor,
            "patch_id": row.patch_id,
            "block_id": row.block_id,
            "scene": row.scene,
            "sample_type": row.get("sample_type", "positive"),
        }
