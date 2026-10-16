"""Select and aggregate the archived and repeated pruning experiments.

Magnitude and Taylor use the complete archived three-seed suite. For GeoBASP,
the higher-validation-Dice recovery is selected independently for each seed
from its archived and repeated executions. The output records every candidate
and seed-wise selection so this reporting is explicit.
"""

from __future__ import annotations

from nepal_cnn.utils.paths import PROJECT_ROOT as ROOT

import csv
import json
from pathlib import Path
from statistics import mean, stdev


ARCHIVED = ROOT / "generated/pruning/candidate_runs/archived"
REPEATED = ROOT / "generated/pruning/candidate_runs/repeated"
OUTPUT = ROOT / "generated/pruning/selection"
MODELS = ("resnet50", "convnextv2_tiny", "segformer_b2")
METHODS = ("group_magnitude", "group_taylor", "geobasp")
SEEDS = (2026, 2027, 2028)
TARGETS = (25, 50, 70)
ABLATIONS = ()  # The revised component table uses the separate matched study.


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def stage_path(root: Path, seed: int, model: str, method: str, target: int) -> Path:
    return (
        root
        / "main_matrix"
        / f"seed_{seed}"
        / model
        / method
        / f"mac_{target}"
        / "summary.json"
    )


def sample_summary(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    candidates: list[dict] = []
    for method in METHODS:
        for target in TARGETS:
            for model in MODELS:
                available: list[tuple[str, int, Path, dict]] = []
                for seed in SEEDS:
                    archived_path = stage_path(ARCHIVED, seed, model, method, target)
                    available.append(("archived", seed, archived_path, read(archived_path)))
                    if method == "geobasp":
                        repeated_path = stage_path(REPEATED, seed, model, method, target)
                        available.append(
                            ("repeated", seed, repeated_path, read(repeated_path))
                        )
                selected: list[tuple[str, int, Path, dict]] = []
                for seed in SEEDS:
                    seed_candidates = [item for item in available if item[1] == seed]
                    seed_candidates.sort(
                        key=lambda item: float(
                            item[3]["validation_extended"]["dice"]
                        ),
                        reverse=True,
                    )
                    selected.append(seed_candidates[0])
                selected_paths = {item[2] for item in selected}
                for seed in SEEDS:
                    seed_candidates = [item for item in available if item[1] == seed]
                    seed_candidates.sort(
                        key=lambda item: float(
                            item[3]["validation_extended"]["dice"]
                        ),
                        reverse=True,
                    )
                    for candidate_rank, (source, _, path, report) in enumerate(
                        seed_candidates, start=1
                    ):
                        candidates.append(
                            {
                                "model": model,
                                "method": method,
                                "target_percent": target,
                                "seed": seed,
                                "attempt": source,
                                "candidate_rank_within_seed": candidate_rank,
                                "selected": path in selected_paths,
                                "dice": float(report["validation_extended"]["dice"]),
                                "summary_path": path.relative_to(ROOT).as_posix(),
                            }
                        )
                for selection_rank, (source, seed, path, report) in enumerate(
                    selected, start=1
                ):
                    extended = report["validation_extended"]
                    bootstrap = extended["paired_spatial_bootstrap"]
                    records.append(
                        {
                            "model": model,
                            "method": method,
                            "target_percent": target,
                            "seed": seed,
                            "attempt": source,
                            "selection_rank": selection_rank,
                            "summary_path": path.relative_to(ROOT).as_posix(),
                            "checkpoint_path": path.with_name("checkpoint.pt")
                            .relative_to(ROOT)
                            .as_posix(),
                            "achieved_mac_reduction": float(
                                report["achieved_mac_reduction"]
                            ),
                            "parameters": int(report["parameters"]),
                            "macs": int(report["macs"]),
                            "dice": float(extended["dice"]),
                            "iou": float(extended["iou"]),
                            "precision": float(extended["precision"]),
                            "recall": float(extended["recall"]),
                            "boundary_f1": float(extended["boundary_f1"]),
                            "negative_patch_false_positive_fraction": float(
                                extended["negative_patch_false_positive_fraction"]
                            ),
                            "worst_positive_block_dice": float(
                                extended["worst_positive_block_dice"]
                            ),
                            "noninferiority_pass": bool(
                                bootstrap["passes_noninferiority"]
                            ),
                        }
                    )

    aggregates: list[dict] = []
    for method in METHODS:
        for target in TARGETS:
            target_records = [
                row
                for row in records
                if row["method"] == method and row["target_percent"] == target
            ]
            overall_mean, overall_sd = sample_summary(
                [float(row["dice"]) for row in target_records]
            )
            row: dict[str, object] = {
                "method": method,
                "target_percent": target,
                "achieved_mac_reduction_mean": mean(
                    float(item["achieved_mac_reduction"]) for item in target_records
                ),
                "overall_dice_mean": overall_mean,
                "overall_dice_sd": overall_sd,
                "noninferiority_passes": sum(
                    bool(item["noninferiority_pass"]) for item in target_records
                ),
                "runs": len(target_records),
            }
            for model in MODELS:
                model_values = [
                    float(item["dice"])
                    for item in target_records
                    if item["model"] == model
                ]
                model_mean, model_sd = sample_summary(model_values)
                row[f"{model}_dice_mean"] = model_mean
                row[f"{model}_dice_sd"] = model_sd
            aggregates.append(row)

    ablation_records: list[dict] = []
    for variant in ABLATIONS:
        relative = (
            Path("ablations")
            / "seed_2026"
            / "resnet50"
            / variant
            / "mac_50"
            / "summary.json"
        )
        archived_path = ARCHIVED / relative
        archived = read(archived_path)
        source, path, report = "archived", archived_path, archived
        extended = report["validation_extended"]
        ablation_records.append(
            {
                "variant": variant,
                "selection_source": source,
                "summary_path": path.relative_to(ROOT).as_posix(),
                "achieved_mac_reduction": float(report["achieved_mac_reduction"]),
                "dice": float(extended["dice"]),
                "precision": float(extended["precision"]),
                "recall": float(extended["recall"]),
                "boundary_f1": float(extended["boundary_f1"]),
                "negative_patch_false_positive_fraction": float(
                    extended["negative_patch_false_positive_fraction"]
                ),
                "worst_positive_block_dice": float(
                    extended["worst_positive_block_dice"]
                ),
            }
        )

    # Table 3 retains available-architecture coverage; pooled curves use the same two U-Nets.
    sirfp_records = []
    for model in ("resnet50", "convnextv2_tiny"):
        for seed in SEEDS:
            for target in TARGETS:
                path = ROOT / f"generated/sirfp/{model}/sirfp/seed_{seed}/mac_{target}/summary.json"
                if path.exists():
                    stage = read(path)
                    sirfp_records.append({"model":model,"method":"sirfp","seed":seed,"target_percent":target,
                                          "dice":float(stage["validation_extended"]["dice"])})
    sirfp_aggregates, common_aggregates = coverage_summaries(records, sirfp_records)
    report = {
        "selection_policy": {
            "baselines": "archived complete three-seed suite",
            "geobasp": (
                "higher validation-Dice recovery selected independently for each "
                "seed from archived and repeated executions"
            ),
            "ablations": "Use experiments components and summary components for the revised matched study",
            "test_data_used_for_selection": False,
        },
        "candidates": candidates,
        "records": records,
        "aggregates": aggregates,
        "ablations": ablation_records,
        "sirfp_available_architecture_aggregates": sirfp_aggregates,
        "common_two_unet_aggregates": common_aggregates,
    }
    (OUTPUT / "selection.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_csv(OUTPUT / "selected_runs.csv", records)
    write_csv(OUTPUT / "all_candidates.csv", candidates)
    write_csv(OUTPUT / "aggregate.csv", aggregates)
    write_csv(OUTPUT / "ablation_selected.csv", ablation_records)
    write_csv(OUTPUT / "sirfp_available_architectures.csv", sirfp_aggregates)
    write_csv(OUTPUT / "common_two_unet_curves.csv", common_aggregates)
    print(json.dumps({"aggregates": aggregates, "ablations": ablation_records}, indent=2))


def coverage_summaries(records, sirfp_records):
    available, common = [], []
    for target in TARGETS:
        sirfp = [r for r in sirfp_records if r["target_percent"] == target]
        if len(sirfp) != 6:
            continue  # Never aggregate a partial set as a completed comparison.
        values = [r["dice"] for r in sirfp]
        available.append({"method":"sirfp","target_percent":target,"runs":6,"architectures":2,
                          "overall_dice_mean":mean(values),"overall_dice_sd":stdev(values),
                          "segformer_b2_dice_mean":None,
                          **{f"{m}_dice_mean":mean(r["dice"] for r in sirfp if r["model"]==m)
                             for m in ("resnet50","convnextv2_tiny")}})
        for method in (*METHODS,"sirfp"):
            rows = [r for r in (sirfp if method=="sirfp" else records)
                    if r["method"]==method and r["target_percent"]==target and r["model"] in ("resnet50","convnextv2_tiny")]
            if len(rows)==6:
                common.append({"method":method,"target_percent":target,"runs":6,"architectures":2,
                               "dice_mean":mean(r["dice"] for r in rows),"dice_sd":stdev(r["dice"] for r in rows)})
    return available, common


if __name__ == "__main__":
    main()
