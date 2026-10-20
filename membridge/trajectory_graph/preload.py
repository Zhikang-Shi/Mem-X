"""Locate and validate a trajectory graph inside an M0 session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .graph import TrajectoryGraph
from .persistence import load_trajectory


@dataclass(frozen=True)
class TrajectoryPreload:
    graph: TrajectoryGraph
    source_path: Path
    source_format: str = "memory_module_trajectory_graph_v1"


def load_session_trajectory(directory: str | Path) -> TrajectoryPreload:
    path = Path(directory) / "trajectory_graph.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing trajectory preload: {path}")
    return TrajectoryPreload(load_trajectory(path), path.resolve())
