"""Load the self-contained semantic builder source shipped in memory_module."""

from .builder_loader import (  # noqa: F401
    Sam3LocalClient,
    TrajectoryPipeline,
    extract_feature,
    load_feature_extractor,
)

__all__ = [
    "Sam3LocalClient",
    "TrajectoryPipeline",
    "extract_feature",
    "load_feature_extractor",
]
