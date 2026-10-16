"""Resolve the six released checkpoints' original dependency paths."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_artifact(root: Path, recorded_path: str | Path) -> Path:
    root = Path(root).resolve()
    path = Path(recorded_path)
    if path.is_absolute():
        return path
    relative = path.as_posix().removeprefix("./")
    if relative.startswith("data/Landslide Inventory of Koshi Basin 2024"):
        relative = relative.replace("data/Landslide Inventory of Koshi Basin 2024",
                                    "data/raw/icimod/landslides_koshi_2024", 1)
    catalog = json.loads((root / "metadata/models.json").read_text(encoding="utf-8"))
    for model in catalog["models"]:
        if relative in (model["original_path"], model["recorded_path"]):
            return root / model["path"]
    return root / relative
