"""Aggregate the completed three-seed dense-model adaptation ablation."""

from __future__ import annotations

from nepal_cnn.utils.paths import PROJECT_ROOT as ROOT

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


OUTPUT_ROOT = ROOT / "generated/dense_ablation"

RUNS = (
    ("standard_spectral", 2026, OUTPUT_ROOT / "standard_spectral_seed2026"),
    ("standard_spectral", 2027, OUTPUT_ROOT / "standard_spectral_seed2027"),
    ("standard_spectral", 2028, OUTPUT_ROOT / "standard_spectral_seed2028"),
    ("imbalance_spectral", 2026, ROOT / "generated/dense_ablation/imbalance_spectral_seed2026"),
    ("imbalance_spectral", 2027, OUTPUT_ROOT / "imbalance_spectral_seed2027"),
    ("imbalance_spectral", 2028, OUTPUT_ROOT / "imbalance_spectral_seed2028"),
    ("terrain_full", 2026, OUTPUT_ROOT / "terrain_full_seed2026"),
    ("terrain_full", 2027, OUTPUT_ROOT / "terrain_full_seed2027"),
    ("terrain_full", 2028, OUTPUT_ROOT / "terrain_full_seed2028"),
)

METRICS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "negative_patch_false_positive_fraction",
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required ablation result is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    per_seed: list[dict] = []
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for variant, seed, run_dir in RUNS:
        validation = load_json(run_dir / "validation_threshold.json")
        test = load_json(run_dir / "test_metrics_ablation.json")
        row = {
            "variant": variant,
            "seed": seed,
            "run_dir": run_dir.relative_to(ROOT).as_posix(),
            "checkpoint_epoch": int(test["checkpoint_epoch"]),
            "threshold": float(test["threshold_selected_on_validation"]),
        }
        for metric in METRICS:
            row[f"validation_{metric}"] = float(validation["best"][metric])
            row[f"test_{metric}"] = float(test[metric])
        per_seed.append(row)
        by_variant[variant].append(row)

    aggregate: list[dict] = []
    for variant in ("standard_spectral", "imbalance_spectral", "terrain_full"):
        rows = sorted(by_variant[variant], key=lambda item: int(item["seed"]))
        if [int(row["seed"]) for row in rows] != [2026, 2027, 2028]:
            raise RuntimeError(f"Incomplete seed matrix for {variant}")
        result: dict[str, object] = {"variant": variant, "runs": len(rows)}
        for split in ("validation", "test"):
            for metric in METRICS:
                values = [float(row[f"{split}_{metric}"]) for row in rows]
                result[f"{split}_{metric}_mean"] = mean(values)
                result[f"{split}_{metric}_sd"] = stdev(values)
        thresholds = [float(row["threshold"]) for row in rows]
        result["threshold_mean"] = mean(thresholds)
        result["threshold_sd"] = stdev(thresholds)
        aggregate.append(result)

    comparisons = []
    pairs = (
        ("imbalance_spectral", "standard_spectral"),
        ("terrain_full", "imbalance_spectral"),
        ("terrain_full", "standard_spectral"),
    )
    seed_rows = {(str(row["variant"]), int(row["seed"])): row for row in per_seed}
    for treatment, reference in pairs:
        result: dict[str, object] = {
            "treatment": treatment,
            "reference": reference,
        }
        for split in ("validation", "test"):
            differences = [
                float(seed_rows[(treatment, seed)][f"{split}_dice"])
                - float(seed_rows[(reference, seed)][f"{split}_dice"])
                for seed in (2026, 2027, 2028)
            ]
            result[f"{split}_dice_difference_by_seed"] = differences
            result[f"{split}_dice_difference_mean"] = mean(differences)
            result[f"{split}_dice_difference_sd"] = stdev(differences)
        comparisons.append(result)

    per_seed_fields = list(per_seed[0])
    aggregate_fields = list(aggregate[0])
    write_csv(OUTPUT_ROOT / "summary_per_seed.csv", per_seed, per_seed_fields)
    write_csv(OUTPUT_ROOT / "summary_by_variant.csv", aggregate, aggregate_fields)
    report = {
        "design": {
            "architecture": "ConvNeXt-V2-Tiny U-Net",
            "seeds": [2026, 2027, 2028],
            "threshold_selection": "independent validation selection for each seed",
            "test_use": "one frozen-threshold evaluation per checkpoint",
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
        "paired_dice_comparisons": comparisons,
    }
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"aggregate": aggregate, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
