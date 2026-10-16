"""One entry point for the compact public repository."""

import argparse
import json
from pathlib import Path
import sys

from nepal_cnn.utils.configuration import dense_catalog, experiment_config, run_workflow, write_dense_configs
from nepal_cnn.utils.paths import PROJECT_ROOT as ROOT


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    delegated = {
        ("data", "download"): "download", ("data", "prepare"): "prepare",
        ("models",): "models", ("threshold",): "threshold", ("evaluate",): "evaluate",
    }
    for prefix, module in delegated.items():
        if tuple(argv[:len(prefix)]) == prefix:
            arguments = argv[len(prefix):]
            if module in {"threshold", "evaluate"} and not any(value == "--data-config" or value.startswith("--data-config=") for value in arguments):
                arguments += ["--data-config", "configs/data_terrain.yaml"]
            run_workflow(module, arguments)
            return
    if argv[:1] == ["pruning"]:
        from nepal_cnn.cli.pruning import main as pruning_main
        pruning_main(argv)
        return
    if argv[:1] == ["benchmark"]:
        run_workflow("latency", argv[1:])
        return

    parser = argparse.ArgumentParser(description="GeoBASP data preparation, training, pruning, and evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="Source guidance, downloads, and preprocessing")
    data.add_argument("action", choices=["sources", "download", "prepare"])
    train = commands.add_parser("train", help="Train a dense model using a recorded profile")
    train.add_argument("--profile", choices=list(dense_catalog()["profiles"]), default="convnextv2_tiny")
    train.add_argument("--output", type=Path)
    train.add_argument("--epochs", type=int)
    train.add_argument("--seed", type=int)
    experiments = commands.add_parser("experiments", help="Reproduce a manuscript experiment")
    experiments.add_argument("name", choices=["main", "repeated", "dense_ablation", "sirfp", "components", "calibration", "feature_guard"])
    experiments.add_argument("--plan", action="store_true", help="Show trajectories without training or data access")
    experiments.add_argument("--models", nargs="+")
    experiments.add_argument("--seeds", nargs="+", type=int, help="Recovery seeds; sampling seeds for calibration")
    experiments.add_argument("--dense-root", type=Path, help="Use freshly trained MODEL/best.pt files instead of supplied models")
    experiments.add_argument("--output", type=Path, help="Separate output directory for a new execution")
    experiments.add_argument("--test", action="store_true", help="Evaluate completed study endpoints at saved validation thresholds")
    experiments.add_argument("--targets", nargs="+", type=float, help="Test endpoints only; training always uses the full progressive schedule")
    summary = commands.add_parser("summary", help="Aggregate regenerated experiment results")
    summary.add_argument("name", choices=["pruning", "dense", "sirfp", "components", "calibration", "feature_guard"])
    commands.add_parser("benchmark", help="CUDA Graph FP32 latency; no dataset needed")
    commands.add_parser("pruning", help="GeoBASP, baselines, and the frozen comparison")
    commands.add_parser("models", help="Retrieve published core checkpoint assets")
    commands.add_parser("threshold", help="Select a threshold using validation data")
    commands.add_parser("evaluate", help="Evaluate a checkpoint with its validation threshold")
    args = parser.parse_args(argv)

    if args.command == "data":
        sources = json.loads((ROOT / "metadata/data_sources.json").read_text(encoding="utf-8"))
        inventory = sources["inventory"]
        print(inventory["title"] + ": " + inventory["doi"])
        print("Original inventory: unchanged GitHub Release assets (CC BY 4.0), restored to " + inventory["target_directory"])
        print("Exact Sentinel-2/DEM scene references, download URLs, and checksums: metadata/data_sources.json")
        print("Run python main.py data download, then python main.py data prepare.")
    elif args.command == "train":
        profile = dense_catalog()["profiles"][args.profile]
        output = write_dense_configs(args.profile)
        arguments = ["--data-config", ROOT / "configs" / profile["data"],
                     "--model-config", output / profile["model"], "--train-config", output / profile["train"],
                     "--output-dir", args.output or "generated/dense_models/" + args.profile]
        for option in ["epochs", "seed"]:
            if getattr(args, option) is not None:
                arguments += ["--" + option, getattr(args, option)]
        run_workflow("train", arguments)
    elif args.command == "experiments":
        import yaml
        from nepal_cnn.experiments.runner import cases, run, test, summarize
        catalog = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"))
        settings = catalog[args.name]
        if args.targets and not args.test:
            parser.error("--targets filters test endpoints; training follows the full progressive schedule")
        if args.name in {"main", "repeated", "dense_ablation"}:
            if args.test and args.name != "dense_ablation":
                parser.error("Use pruning final-test for supplied main-comparison checkpoints; --test supports sensitivity/SIRFP studies")
            if args.plan:
                print(json.dumps(settings, indent=2))
                return
        else:
            selected = list(cases(args.name, settings))
            if args.models:
                selected = [c for c in selected if c["model"] in args.models]
            if args.seeds:
                selected = [c for c in selected if c["seed"] in args.seeds]
            if not selected:
                parser.error("No trajectories match the requested models/seeds")
            if args.targets and not set(args.targets) <= set(settings["targets"]):
                parser.error("Requested test target is absent from this study")
            if args.output:
                settings["output_root"] = str(args.output)
            if args.plan:
                print(json.dumps({"trajectories": len(selected), "targets": settings["targets"], "cases": selected}, indent=2))
                return
            pruning = yaml.safe_load((ROOT / "configs/pruning.yaml").read_text())
            dense_paths = {c["model"]: (str(args.dense_root / c["model"] / "best.pt") if args.dense_root else
                                       pruning["selected_dense_checkpoints"][c["model"]]) for c in selected}
            if args.test:
                test(args.name, settings, selected, dense_paths, args.targets)
            else:
                run(args.name, settings, selected, dense_paths)
            summarize(settings)
            return
        config = experiment_config(args.name)
        if args.name == "dense_ablation":
            if args.models or args.dense_root:
                parser.error("Dense ablation trains ConvNeXt variants from pretrained encoders; omit --models and --dense-root")
            materialized = yaml.safe_load(config.read_text())
            if args.seeds:
                materialized["experiments"] = [c for c in materialized["experiments"] if c["seed"] in args.seeds]
                if not materialized["experiments"]:
                    parser.error("No dense-ablation seeds match")
            if args.output:
                materialized["output_root"] = str(args.output)
                materialized["status_path"] = str(args.output / "status.json")
                materialized["suite_log"] = str(args.output / "suite.log")
            config.write_text(yaml.safe_dump(materialized, sort_keys=False))
            run_workflow("dense_ablation", ["--config", config, "--test" if args.test else "--run"])
        else:
            from nepal_cnn.experiments.pruning_suite import run_suite
            materialized = yaml.safe_load(config.read_text())
            if args.models:
                if not set(args.models) <= set(materialized["main_matrix"]["models"]):
                    parser.error("Unknown model for this matrix")
                materialized["main_matrix"]["models"] = args.models
            if args.seeds:
                materialized["main_matrix"]["seeds"] = args.seeds
            if args.output:
                materialized["output_root"] = str(args.output)
            if args.dense_root:
                pruning = yaml.safe_load((ROOT / "configs/pruning.yaml").read_text())
                for model in materialized["main_matrix"]["models"]:
                    pruning["selected_dense_checkpoints"][model] = str(args.dense_root / model / "best.pt")
                fresh = ROOT / "generated/configs/pruning_trained.yaml"
                fresh.write_text(yaml.safe_dump(pruning, sort_keys=False))
                materialized["pruning_config"] = str(fresh.relative_to(ROOT))
            config.write_text(yaml.safe_dump(materialized, sort_keys=False))
            run_suite(ROOT, config)
    elif args.command == "summary":
        if args.name in {"pruning", "dense"}:
            run_workflow("summarize_" + args.name, [])
        else:
            import yaml
            from nepal_cnn.experiments.runner import summarize
            settings = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())[args.name]
            summarize(settings)
