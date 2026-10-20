"""Compatibility facade backed exclusively by memory_module capabilities."""

from membridge.occupancy_map import (
    SparseWorldMetricMap,
    WorldGridSpec,
    WorldPlanningWindow,
)
from membridge.occupancy_map.persistence import (
    COORDINATE_FRAME,
    WORLD_MAP_SCHEMA_VERSION,
)
from membridge.trajectory_graph import (
    TrajectoryGraph,
    TrajectoryNode,
    WorldMemoryRecorder,
    habitat_camera_to_world,
    habitat_depth_to_meters,
)
from membridge.trajectory_graph.persistence import TRAJECTORY_SCHEMA_VERSION

__all__ = [
    "COORDINATE_FRAME",
    "WORLD_MAP_SCHEMA_VERSION",
    "TRAJECTORY_SCHEMA_VERSION",
    "SparseWorldMetricMap",
    "TrajectoryGraph",
    "TrajectoryNode",
    "WorldGridSpec",
    "WorldPlanningWindow",
    "WorldMemoryRecorder",
    "habitat_camera_to_world",
    "habitat_depth_to_meters",
]
