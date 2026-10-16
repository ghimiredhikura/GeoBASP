"""Compact runners for the manuscript's SIRFP and sensitivity studies."""
from itertools import product
import hashlib
import json
from pathlib import Path
import statistics
from nepal_cnn.utils.paths import PROJECT_ROOT as ROOT


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def cases(name, config):
    if name == "sirfp":
        for model, seed in product(config["models"], config["seeds"]):
            yield dict(model=model, condition="sirfp", seed=seed, recovery_seed=seed,
                       calibration_seed=seed, protect_final_features=False)
    elif name == "calibration":
        for condition, seed in product(config["conditions"], config["sampling_seeds"]):
            yield dict(model=config["model_name"], condition=condition, seed=seed,
                       calibration_seed=seed, recovery_seed=config["recovery_seed"],
                       variant=config["geobasp_variant"], protect_final_features=False)
    elif name == "components":
        for variant, seed in product(config["variants"], config["recovery_seeds"]):
            yield dict(model=config["model_name"], condition=variant["id"], seed=seed,
                       recovery_seed=seed, calibration_seed=config["calibration_seed"],
                       variant=variant, protect_final_features=True)
    elif name == "feature_guard":
        for protected, seed in product(config["protect_final_features"], config["recovery_seeds"]):
            yield dict(model=config["model_name"], condition="guarded" if protected else "unguarded",
                       seed=seed, recovery_seed=seed, calibration_seed=config["calibration_seed"],
                       variant=config["geobasp_variant"], protect_final_features=protected)
    else:
        raise ValueError(name)


def branch(output, case):
    return output / case["model"] / case["condition"] / f'seed_{case["seed"]}'


def run(name, config, selected, dense_paths):
    import torch
    import torch_pruning as tp
    from torch.utils.data import DataLoader, Subset
    from nepal_cnn.data.dataset import KoshiPositiveDataset
    from nepal_cnn.pruning.audit import load_dense_model
    from nepal_cnn.pruning.calibration import select_calibration_rows
    from nepal_cnn.pruning.calibration_conditions import select_condition
    from nepal_cnn.pruning.architecture import architecture_pruning_spec
    from nepal_cnn.pruning.baseline import _prune_to_nearest_mac_target
    from nepal_cnn.evaluation.spatial import evaluate_manuscript_validation
    from nepal_cnn.training.recovery import seed_everything, build_recovery_loaders, evaluate_validation, recover_model
    from nepal_cnn.utils.logging import configure_logging
    from nepal_cnn.pruning.sirfp import GraphCollector, SIRFPImportance
    from nepal_cnn.pruning.sirfp_convnext import spatial_linear_modules
    from nepal_cnn.pruning.sirfp_pruning import prune_to_target
    from nepal_cnn.models.checkpoints import load_checkpoint
    if not torch.cuda.is_available():
        raise RuntimeError("These pruning/recovery experiments require CUDA")
    output = ROOT / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output / "run.log")
    device = torch.device("cuda")
    dataset = KoshiPositiveDataset(ROOT / config["data_config"], split="train", normalize=True,
                                   augment=False, crop_size=None)
    for case in selected:
        dense = ROOT / dense_paths[case["model"]]
        run_dir = branch(output, case)
        run_dir.mkdir(parents=True, exist_ok=True)
        frozen = {"study": name, "case": case, "config": config, "dense_checkpoint": str(dense.relative_to(ROOT)),
                  "dense_sha256": sha(dense), "metadata_sha256": {p.name: sha(p) for p in (ROOT / "metadata").glob("*.csv")}}
        freeze_path = run_dir / "protocol.json"
        if freeze_path.exists() and json.loads(freeze_path.read_text()) != frozen:
            raise ValueError(f"Existing trajectory uses different settings: {run_dir}")
        write_json(freeze_path, frozen)
        protocol_hash = sha(freeze_path)
        seed_everything(case["recovery_seed"])
        checkpoint, model = load_dense_model(dense, device)
        model.eval()
        architecture = checkpoint["model_config"]["architecture"]
        model.protect_final_features = case["protect_final_features"]
        channels = int(checkpoint["model_config"].get("input_channels", 4))
        example = torch.randn(1, channels, 512, 512, device=device) if name == "sirfp" else torch.zeros(1, channels, 512, 512, device=device)
        base_macs, _ = tp.utils.count_ops_and_params(model, example)
        selection = (select_condition(dataset.rows, case["condition"], case["calibration_seed"])
                     if name == "calibration" else select_calibration_rows(dataset.rows, 24, 8, case["calibration_seed"]))
        selection.manifest.to_csv(run_dir / "calibration.csv", index=False)
        loader = DataLoader(Subset(dataset, selection.indices), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        train_config = checkpoint["train_config"]
        train_loader = val_loader = None
        # SIRFP's recovery RNG/loaders continue between stages, as in its retained runs.
        if name == "sirfp":
            train_loader, val_loader = build_recovery_loaders(ROOT / config["data_config"], train_config,
                                                            config["recovery"], case["recovery_seed"], device)
        history = []
        for target in config["targets"]:
            stage = run_dir / f"mac_{round(target * 100):02d}"
            complete = stage / "summary.json"
            if complete.exists():
                saved = json.loads(complete.read_text())
                if saved["protocol_sha256"] != protocol_hash or sha(stage / "checkpoint.pt") != saved["checkpoint_sha256"]:
                    raise ValueError(f"Completed endpoint changed: {stage}")
                # SIRFP must continue with the original RNG: skip only an entire completed trajectory.
                if name == "sirfp" and target != config["targets"][-1]:
                    if not (run_dir / f'mac_{round(config["targets"][-1]*100):02d}/summary.json').exists():
                        raise RuntimeError("Incomplete SIRFP trajectory: preserve its output and restart into a new --output directory")
                _, model = load_checkpoint(stage / "checkpoint.pt", device)
                model.protect_final_features = case["protect_final_features"]
                history = torch.load(stage / "checkpoint.pt", map_location="cpu", weights_only=False)["pruning_history"]
                continue
            stage.mkdir(parents=True, exist_ok=False)
            logger.info("%s | %s | seed %d | target %.0f%%", name, case["condition"], case["seed"], target * 100)
            if name == "sirfp":
                spec = architecture_pruning_spec(model, architecture)
                linear = spatial_linear_modules(model) if architecture == "unet_convnextv2_tiny" else ()
                collector = GraphCollector(model, spec.ignored_layers, alpha=config["graph_collection"]["alpha"], linear_modules=linear)
                try:
                    for batch in loader:
                        with torch.inference_mode(), torch.amp.autocast("cuda"):
                            prediction = model(batch["image"].to(device))
                        if not torch.isfinite(prediction).all():
                            raise RuntimeError("Non-finite graph-collection output")
                finally:
                    collector.close()
                importance = SIRFPImportance(collector.graphs, include_linear=bool(linear), include_depthwise=bool(linear))
                model, achieved, history, steps = prune_to_target(model, importance, example, int(base_macs), target,
                                                                 {**config, "architecture": architecture}, history, logger)
                del importance, collector
            else:
                model, achieved, history, steps = _prune_to_nearest_mac_target(model=model, architecture=architecture,
                    method="geobasp", example=example, base_macs=int(base_macs), target=target,
                    micro_ratio=config["micro_channel_pruning_ratio"], maximum_microsteps=config["maximum_microsteps_per_target"],
                    calibration_loader=loader, geobasp_loader=loader, calibration_selection=selection,
                    calibration_loss_config=train_config["loss"], geobasp_variant=case["variant"], device=device,
                    cumulative_history=history)
                seed_everything(case["recovery_seed"])
                train_loader, val_loader = build_recovery_loaders(ROOT / config["data_config"], train_config,
                                                                config["recovery"], case["recovery_seed"], device)
            if abs(achieved - target) > config["maximum_target_absolute_error"]:
                raise RuntimeError("MAC target outside tolerance")
            if case["protect_final_features"] and model.segmentation_head.in_channels != 32:
                raise RuntimeError("Final-feature safeguard changed width")
            write_json(stage / "pruning_steps.json", steps)
            before = evaluate_validation(model, val_loader, device, train_config["loss"], train_config["thresholds"], config["recovery"]["amp"])
            if name != "sirfp":
                seed_everything(case["recovery_seed"])
                train_loader, val_loader = build_recovery_loaders(ROOT / config["data_config"], train_config,
                                                                config["recovery"], case["recovery_seed"], device)
            model, recovered, recovery = recover_model(model, train_loader, val_loader, device, train_config, config["recovery"],
                config["recovery"]["epochs_per_stage"], {"model": case["model"], "method": case["condition"], "nominal_mac_reduction": target})
            ext, stats = config["extended_validation"], config["statistics"]
            extended = evaluate_manuscript_validation(model, case["model"], recovered["best_threshold"], dense, ROOT,
                ROOT / config["data_config"], stage, output / "dense_validation_reference", device, ext["batch_size"],
                stats["spatial_block_bootstrap_replicates"], stats["spatial_block_bootstrap_seed"], stats["validation_noninferiority_margin_dice"],
                boundary_width=ext["boundary_width_pixels"], component_edges=ext["component_size_edges_pixels"])
            macs, parameters = tp.utils.count_ops_and_params(model, example)
            summary = {**case, "nominal_mac_reduction": target, "achieved_mac_reduction": achieved,
                       "macs": int(macs), "parameters": int(parameters), "validation_no_recovery": before,
                       "validation_recovered": recovered, "validation_extended": extended,
                       "protocol_sha256": protocol_hash, "test_data_accessed": False}
            payload = {**summary, "model_name": case["model"], "model": model.cpu().state_dict(),
                       "model_config": checkpoint["model_config"], "train_config": train_config,
                       "dense_checkpoint": str(dense.relative_to(ROOT)), "pruning_history": history}
            torch.save(payload, stage / "checkpoint.pt")
            _, replay = load_checkpoint(stage / "checkpoint.pt", device)
            with torch.inference_mode():
                torch.testing.assert_close(replay(example), model.to(device)(example), rtol=0, atol=0)
            del replay
            summary["checkpoint_sha256"] = sha(stage / "checkpoint.pt")
            write_json(stage / "summary.json", summary)
            write_json(stage / "recovery_history.json", recovery)
        del model, checkpoint, loader, train_loader, val_loader
        torch.cuda.empty_cache()


def distribution(values):
    return dict(n=len(values), mean=statistics.mean(values), sample_sd=statistics.stdev(values) if len(values)>1 else None,
                minimum=min(values), maximum=max(values))


def test(name, config, selected, dense_paths, targets=None):
    """Freeze all requested checkpoints and validation thresholds before test access."""
    import torch
    from nepal_cnn.models.checkpoints import load_checkpoint
    from nepal_cnn.evaluation.pruned import select_validation_threshold
    from nepal_cnn.evaluation.spatial import _loader, _basic_records, evaluate_manuscript_validation
    output = ROOT / config["output_root"]
    entries = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for case in selected:
        for target in targets or [config["targets"][-1]]:
            stage = branch(output, case) / f"mac_{round(target * 100):02d}"
            summary = json.loads((stage / "summary.json").read_text())
            if sha(stage / "checkpoint.pt") != summary["checkpoint_sha256"]:
                raise ValueError(f"Changed checkpoint: {stage}")
            threshold = summary["validation_extended"]["threshold"]
            if name == "sirfp":
                _, network = load_checkpoint(stage / "checkpoint.pt", device)
                threshold = select_validation_threshold(network, ROOT / config["data_config"], device,
                    config["extended_validation"]["batch_size"], .005)["threshold"]
                del network
            entries.append({"checkpoint": str((stage / "checkpoint.pt").relative_to(ROOT)),
                            "sha256": summary["checkpoint_sha256"], "model": case["model"],
                            "threshold": threshold})
    frozen = {"entries": entries, "config": config, "dense_sha256": {model: sha(ROOT / path) for model, path in dense_paths.items()}}
    signature = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:16]
    manifest = output / f"test_manifest_{signature}.json"
    if manifest.exists() and json.loads(manifest.read_text()) != frozen:
        raise ValueError("Test manifest changed")
    write_json(manifest, frozen)
    if all((ROOT / e["checkpoint"]).parent.joinpath("test/test_manifest_sha256.json").is_file() for e in entries):
        for entry in entries:
            marker = (ROOT / entry["checkpoint"]).parent / "test/test_manifest_sha256.json"
            if json.loads(marker.read_text())["manifest"] != sha(manifest):
                raise ValueError("Test endpoint belongs to another manifest")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config["extended_validation"]["batch_size"]
    dense_references = {}
    # Dense thresholds are selected only on validation; the pruned ones remain unchanged.
    for model in sorted({entry["model"] for entry in entries}):
        _, network = load_checkpoint(dense_paths[model], device)
        threshold = select_validation_threshold(network, ROOT / config["data_config"], device, batch_size, .005)["threshold"]
        loader = _loader(ROOT / config["data_config"], batch_size, device, split="test")
        import pandas as pd
        records = pd.DataFrame(_basic_records(network, loader, device, threshold))
        path = output / "dense_test_reference" / f"{model}_per_patch.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        records.to_csv(path, index=False)
        dense_references[model] = path, threshold
        del network, loader
    stats, ext = config["statistics"], config["extended_validation"]
    for entry in entries:
        stage = ROOT / entry["checkpoint"]
        folder = stage.parent / "test"
        if (folder / "test_manifest_sha256.json").exists():
            if json.loads((folder / "test_manifest_sha256.json").read_text())["manifest"] != sha(manifest):
                raise ValueError("Previously evaluated endpoint belongs to another test manifest")
            continue
        _, network = load_checkpoint(stage, device)
        records, dense_threshold = dense_references[entry["model"]]
        evaluate_manuscript_validation(network, entry["model"], entry["threshold"], ROOT / dense_paths[entry["model"]], ROOT,
            ROOT / config["data_config"], folder, output / "unused_dense_cache", device, batch_size,
            stats["spatial_block_bootstrap_replicates"], stats["spatial_block_bootstrap_seed"], stats["validation_noninferiority_margin_dice"],
            boundary_width=ext["boundary_width_pixels"], component_edges=ext["component_size_edges_pixels"], split="test",
            dense_records_path=records, dense_threshold_override=dense_threshold)
        write_json(folder / "test_manifest_sha256.json", {"manifest": sha(manifest)})
        del network


def summarize(config):
    import pandas as pd
    output = ROOT / config["output_root"]
    rows = []
    for path in output.glob("*/*/seed_*/mac_*/summary.json"):
        item = json.loads(path.read_text())
        if sha(path.with_name("checkpoint.pt")) != item["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed: {path}")
        test = path.parent / "test/test_extended.json"
        rows.append({"model": item["model"], "condition": item["condition"], "seed": item["seed"],
            "target": item["nominal_mac_reduction"], "achieved_mac_reduction": item["achieved_mac_reduction"],
            "validation_dice": item["validation_extended"]["dice"],
            "test_dice": json.loads(test.read_text())["dice"] if test.exists() else None})
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "per_seed.csv", index=False)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["model"], row["condition"], row["target"]), []).append(row)
    groups = [{"model": model, "condition": condition, "target": target,
               **{metric: distribution([r[metric] for r in records if r[metric] is not None])
                  for metric in ("validation_dice", "test_dice") if any(r[metric] is not None for r in records)}}
              for (model, condition, target), records in grouped.items()]
    write_json(output / "summary.json", {"groups": groups, "all_completed_outcomes_included": True})
    print(f"Summaries: {output}")
