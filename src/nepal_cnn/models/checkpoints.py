"""Load dense or physically pruned checkpoints using their recorded graph."""
from pathlib import Path
import torch
from nepal_cnn.utils.paths import PROJECT_ROOT as ROOT
from nepal_cnn.pruning.audit import load_dense_model
from nepal_cnn.evaluation.pruned import load_pruned_model


def load_checkpoint(path, device):
    path = ROOT / Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("pruning_history"):
        with torch.enable_grad():
            payload, _, model = load_pruned_model(ROOT, path, device)
        return payload, model
    payload, model = load_dense_model(path, device)
    return payload, model.eval()
