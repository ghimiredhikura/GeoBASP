"""Resumable unattended orchestration for the manuscript pruning matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
import traceback

import torch
import yaml

from nepal_cnn.pruning.baseline import run_baselines
from nepal_cnn.utils.logging import configure_logging


@dataclass(frozen=True)
class SuiteTask:
    section: str
    seed: int
    model: str
    method: str
    targets: tuple[float, ...]
    recovery_epochs: int
    output_root: str

    @property
    def task_id(self) -> str:
        return f"{self.section}/seed_{self.seed}/{self.model}/{self.method}"


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _tasks_from_config(config: dict) -> list[SuiteTask]:
    tasks: list[SuiteTask] = []
    for section_name in ("main_matrix", "ablations"):
        section = config[section_name]
        for seed in section["seeds"]:
            for model in section["models"]:
                for method in section["methods"]:
                    tasks.append(
                        SuiteTask(
                            section=section_name,
                            seed=int(seed),
                            model=str(model),
                            method=str(method),
                            targets=tuple(float(value) for value in section["targets"]),
                            recovery_epochs=int(section["recovery_epochs"]),
                            output_root=str(section_name),
                        )
                    )
    return tasks


def run_suite(root: Path, suite_config_path: Path) -> dict:
    """Run every branch sequentially, preserving failures and skipping completions."""
    suite_config = yaml.safe_load(suite_config_path.read_text(encoding="utf-8"))
    pruning_config_path = (root / suite_config["pruning_config"]).resolve()
    output_root = (root / suite_config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_root / "suite.log")
    if not torch.cuda.is_available():
        raise RuntimeError("The extended pruning suite requires CUDA")
    free_gb = shutil.disk_usage(output_root).free / 1024**3
    required_gb = float(suite_config.get("minimum_free_disk_gb", 20))
    if free_gb < required_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GiB free; suite requires at least {required_gb:.1f} GiB"
        )
    pruning_config = yaml.safe_load(pruning_config_path.read_text(encoding="utf-8"))
    for checkpoint in pruning_config["selected_dense_checkpoints"].values():
        if not (root / checkpoint).is_file():
            raise FileNotFoundError(root / checkpoint)
    tasks = _tasks_from_config(suite_config)
    task_methods = {task.method for task in tasks}
    skip_methods = {str(method) for method in suite_config.get("skip_methods", [])}
    unknown_skip_methods = skip_methods - task_methods
    if unknown_skip_methods:
        raise ValueError(
            "skip_methods contains methods absent from the suite: "
            f"{sorted(unknown_skip_methods)}"
        )
    status_path = output_root / "suite_status.json"
    previous = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.exists()
        else {}
    )
    completed_ids = set(previous.get("completed_task_ids", []))
    skipped_ids = set(previous.get("skipped_task_ids", []))
    failures = dict(previous.get("failures", {}))
    status = {
        "state": "running",
        "started_at": previous.get("started_at", datetime.now().isoformat()),
        "updated_at": datetime.now().isoformat(),
        "suite_config": str(suite_config_path.relative_to(root)).replace("\\", "/"),
        "pruning_config": str(pruning_config_path.relative_to(root)).replace("\\", "/"),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "total_tasks": len(tasks),
        "completed_task_ids": sorted(completed_ids),
        "skipped_task_ids": sorted(skipped_ids),
        "skip_methods": sorted(skip_methods),
        "failures": failures,
        "test_data_accessed": False,
    }
    _write_json_atomic(status_path, status)
    logger.info("=" * 78)
    logger.info(
        "EXTENDED SUITE | %d tasks | GPU %s | free disk %.1f GiB",
        len(tasks),
        torch.cuda.get_device_name(0),
        free_gb,
    )
    logger.info("Held-out test data are disabled for this suite")
    logger.info("=" * 78)

    for task_index, task in enumerate(tasks, start=1):
        task_output = output_root / task.output_root / f"seed_{task.seed}"
        branch_summary = task_output / task.model / task.method / "summary.json"
        if task.task_id in completed_ids and branch_summary.exists():
            logger.info(
                "TASK %02d/%02d | SKIP completed | %s",
                task_index,
                len(tasks),
                task.task_id,
            )
            continue
        if task.method in skip_methods:
            skipped_ids.add(task.task_id)
            status["skipped_task_ids"] = sorted(skipped_ids)
            status["updated_at"] = datetime.now().isoformat()
            _write_json_atomic(status_path, status)
            logger.info(
                "TASK %02d/%02d | SKIP configured method | %s",
                task_index,
                len(tasks),
                task.task_id,
            )
            continue
        logger.info("-" * 78)
        logger.info(
            "TASK %02d/%02d | %s | targets %s",
            task_index,
            len(tasks),
            task.task_id,
            ", ".join(f"{100 * value:.0f}%" for value in task.targets),
        )
        status["active_task"] = asdict(task)
        status["updated_at"] = datetime.now().isoformat()
        _write_json_atomic(status_path, status)
        try:
            run_baselines(
                root=root,
                config_path=pruning_config_path,
                models=[task.model],
                methods=[task.method],
                targets=list(task.targets),
                output_root=task_output,
                recovery_epochs=task.recovery_epochs,
                dry_run=False,
                resume=True,
                seed_override=task.seed,
            )
            completed_ids.add(task.task_id)
            failures.pop(task.task_id, None)
        except Exception as error:
            failure = {
                "task": asdict(task),
                "failed_at": datetime.now().isoformat(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            failures[task.task_id] = failure
            failure_path = output_root / "failures" / Path(task.task_id).with_suffix(".json")
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
            logger.exception("TASK FAILED | %s", task.task_id)
            if not bool(suite_config.get("continue_on_error", True)):
                raise
        finally:
            status["completed_task_ids"] = sorted(completed_ids)
            status["skipped_task_ids"] = sorted(skipped_ids)
            status["failures"] = failures
            status["updated_at"] = datetime.now().isoformat()
            status.pop("active_task", None)
            _write_json_atomic(status_path, status)

    status["state"] = "completed" if not failures else "completed_with_failures"
    status["finished_at"] = datetime.now().isoformat()
    status["updated_at"] = status["finished_at"]
    _write_json_atomic(status_path, status)
    logger.info("=" * 78)
    logger.info(
        "SUITE FINISHED | completed %d | skipped %d | total %d | "
        "failures %d | status %s",
        len(completed_ids),
        len(skipped_ids),
        len(tasks),
        len(failures),
        status_path,
    )
    logger.info("=" * 78)
    if failures:
        raise RuntimeError(
            f"Extended suite completed with {len(failures)} failed task(s); "
            f"see {status_path}"
        )
    return status
