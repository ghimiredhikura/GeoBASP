"""Dependency-aware structured-pruning research components."""

from .architecture import (
    PruningSpec,
    architecture_pruning_spec,
    build_meta_pruner,
    importance_for_method,
    refresh_architecture_after_pruning,
)

__all__ = [
    "PruningSpec",
    "architecture_pruning_spec",
    "build_meta_pruner",
    "importance_for_method",
    "refresh_architecture_after_pruning",
]
