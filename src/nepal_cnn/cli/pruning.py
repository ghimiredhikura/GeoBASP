"""Command-line interface for reproducible project workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from nepal_cnn.utils.paths import PROJECT_ROOT

from nepal_cnn.pruning.baseline import run_baselines
from nepal_cnn.pruning.audit import run_audit
from nepal_cnn.utils.logging import configure_logging
from nepal_cnn.experiments.pruning_suite import run_suite
from nepal_cnn.evaluation.pruned import run_final_test
from nepal_cnn.evaluation.final_benchmark import run_final_benchmark


def project_root() -> Path:
    return PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nepal-cnn")
    commands = parser.add_subparsers(dest="command", required=True)
    pruning = commands.add_parser("pruning", help="Structured-pruning experiments")
    pruning_commands = pruning.add_subparsers(dest="pruning_command", required=True)

    audit = pruning_commands.add_parser(
        "audit", help="Run one-step architecture and criterion compatibility checks"
    )
    audit.add_argument("--config", type=Path, default=Path("configs/pruning.yaml"))
    audit.add_argument(
        "--methods",
        nargs="+",
        choices=("group_magnitude", "group_taylor"),
        default=None,
    )
    audit.add_argument("--models", nargs="+", default=None)
    audit.add_argument("--pruning-ratio", type=float, default=0.05)
    audit.add_argument("--image-size", type=int, default=256)
    audit.add_argument(
        "--output",
        type=Path,
        default=Path("generated/experiments/pruning/compatibility_audit.json"),
    )

    baseline = pruning_commands.add_parser(
        "baseline", help="Run matched Group Magnitude and Group Taylor trajectories"
    )
    baseline.add_argument("--config", type=Path, default=Path("configs/pruning.yaml"))
    baseline.add_argument("--models", nargs="+", default=None)
    baseline.add_argument(
        "--methods",
        nargs="+",
        choices=("group_magnitude", "group_taylor"),
        default=None,
    )
    baseline.add_argument("--resume", action="store_true")
    baseline.add_argument("--targets", nargs="+", type=float, default=None)
    baseline.add_argument(
        "--output-root",
        type=Path,
        default=Path("generated/experiments/pruning/baselines"),
    )
    baseline.add_argument("--recovery-epochs", type=int, default=None)
    baseline.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip recovery; use with a small explicit target for a fast pilot",
    )

    geobasp = pruning_commands.add_parser(
        "geobasp", help="Run GeoBASP or a preregistered component ablation"
    )
    geobasp.add_argument("--config", type=Path, default=Path("configs/pruning.yaml"))
    geobasp.add_argument("--models", nargs="+", default=None)
    geobasp.add_argument(
        "--variants",
        nargs="+",
        default=["geobasp"],
        choices=(
            "geobasp_overall",
            "geobasp_boundary",
            "geobasp_small",
            "geobasp_all_nogeo",
            "geobasp_geo_nocost",
            "geobasp",
        ),
    )
    geobasp.add_argument("--targets", nargs="+", type=float, default=None)
    geobasp.add_argument(
        "--output-root",
        type=Path,
        default=Path("generated/experiments/pruning/geobasp"),
    )
    geobasp.add_argument("--recovery-epochs", type=int, default=None)
    geobasp.add_argument("--dry-run", action="store_true")
    geobasp.add_argument("--resume", action="store_true")

    suite = pruning_commands.add_parser(
        "suite", help="Launch the resumable manuscript experiment matrix"
    )
    suite.add_argument(
        "--config",
        type=Path,
        default=None,
    )
    suite.add_argument("--experiment", choices=["main", "repeated"], default="main")

    final_test = pruning_commands.add_parser(
        "final-test", help="Evaluate a frozen pruning manifest once on held-out test data"
    )
    final_test.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pruning_final_test_selected_v2.yaml"),
    )

    final_benchmark = pruning_commands.add_parser(
        "benchmark-final", help="Eager diagnostic; use main.py benchmark for the reported CUDA Graph protocol"
    )
    final_benchmark.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pruning_final_test_selected_v2.yaml"),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = build_parser().parse_args(argv)
    root = project_root()
    if args.command == "pruning" and args.pruning_command == "audit":
        run_audit(
            root=root,
            config_path=(root / args.config).resolve(),
            methods=args.methods,
            models=args.models,
            pruning_ratio=args.pruning_ratio,
            image_size=args.image_size,
            output_path=(root / args.output).resolve(),
        )
        return
    if args.command == "pruning" and args.pruning_command == "baseline":
        run_baselines(
            root=root,
            config_path=(root / args.config).resolve(),
            models=args.models,
            methods=args.methods,
            targets=args.targets,
            output_root=(root / args.output_root).resolve(),
            recovery_epochs=args.recovery_epochs,
            dry_run=args.dry_run,
            resume=args.resume,
        )
        return
    if args.command == "pruning" and args.pruning_command == "geobasp":
        run_baselines(
            root=root,
            config_path=(root / args.config).resolve(),
            models=args.models,
            methods=args.variants,
            targets=args.targets,
            output_root=(root / args.output_root).resolve(),
            recovery_epochs=args.recovery_epochs,
            dry_run=args.dry_run,
            resume=args.resume,
        )
        return
    if args.command == "pruning" and args.pruning_command == "suite":
        from nepal_cnn.utils.configuration import experiment_config
        config = (root / args.config).resolve() if args.config else experiment_config(args.experiment)
        run_suite(root=root, suite_config_path=config)
        return
    if args.command == "pruning" and args.pruning_command == "final-test":
        run_final_test(root=root, manifest_path=(root / args.config).resolve())
        return
    if args.command == "pruning" and args.pruning_command == "benchmark-final":
        run_final_benchmark(root=root, manifest_path=(root / args.config).resolve())
        return
    raise RuntimeError("Unreachable command dispatch")
