"""Read consolidated settings and materialize configurations for existing workflows."""

import os
from pathlib import Path
import subprocess
import sys

import yaml

from nepal_cnn.utils.paths import PROJECT_ROOT as ROOT


def dense_catalog():
    return yaml.safe_load((ROOT / "configs/dense.yaml").read_text(encoding="utf-8"))


def write_dense_configs(profile=None):
    catalog = dense_catalog()
    names = catalog["definitions"] if profile is None else [
        catalog["profiles"][profile]["model"], catalog["profiles"][profile]["train"]
    ]
    output = ROOT / "generated/configs"
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        (output / name).write_text(
            yaml.safe_dump(catalog["definitions"][name], sort_keys=False), encoding="utf-8"
        )
    return output


def experiment_config(name):
    catalog = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"))
    config = catalog[name]
    if name == "dense_ablation":
        write_dense_configs()
        for experiment in config["experiments"]:
            for key in ["model_config", "train_config"]:
                experiment[key] = "generated/configs/" + Path(experiment[key]).name
    output = ROOT / "generated/configs"
    output.mkdir(parents=True, exist_ok=True)
    target = output / (name + ".yaml")
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


WORKFLOW_MODULES = {
    "boundaries": "nepal_cnn.data.boundaries",
    "dense_ablation": "nepal_cnn.experiments.dense_ablation",
    "download": "nepal_cnn.data.download",
    "evaluate": "nepal_cnn.evaluation.evaluate",
    "latency": "nepal_cnn.evaluation.latency",
    "models": "nepal_cnn.models.assets",
    "negative_masks": "nepal_cnn.data.negative_masks",
    "positive_masks": "nepal_cnn.data.positive_masks",
    "prepare": "nepal_cnn.data.prepare",
    "reconstructed_masks": "nepal_cnn.data.reconstructed_masks",
    "slope": "nepal_cnn.data.slope",
    "summarize_dense": "nepal_cnn.experiments.summarize_dense",
    "summarize_pruning": "nepal_cnn.experiments.summarize_pruning",
    "threshold": "nepal_cnn.evaluation.threshold",
    "train": "nepal_cnn.training.train",
}


def run_workflow(name, arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, "-B", "-m", WORKFLOW_MODULES[name], *map(str, arguments)],
        cwd=ROOT, env=environment, check=True,
    )
