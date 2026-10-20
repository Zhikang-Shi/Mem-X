"""Record navigation memory before semantic keyframe filtering."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from membridge.occupancy_map import SparseWorldMetricMap

from .graph import TrajectoryGraph


def habitat_camera_to_world(
    agent_position_xyz: Sequence[float], heading_rad: float, *,
    sensor_offset_xyz: Sequence[float] = (0.0, 1.2, 0.0),
) -> np.ndarray:
    position = np.asarray(agent_position_xyz, dtype=float).reshape(3)
    offset = np.asarray(sensor_offset_xyz, dtype=float).reshape(3)
    if not np.isfinite(position).all() or not np.isfinite(offset).all():
        raise ValueError("agent position and sensor offset must be finite")
    cosine, sine = math.cos(float(heading_rad)), math.sin(float(heading_rad))
    rotation = np.asarray([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]], dtype=float)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, position + rotation @ offset
    return pose


def habitat_depth_to_meters(
    depth: np.ndarray, *, normalized: bool, min_depth_m: float, max_depth_m: float
) -> np.ndarray:
    result = np.asarray(depth, dtype=np.float32)
    if normalized:
        result = float(min_depth_m) + result * (float(max_depth_m) - float(min_depth_m))
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


class WorldMemoryRecorder:
    def __init__(
        self, metric_map: SparseWorldMetricMap | None = None,
        trajectory: TrajectoryGraph | None = None,
    ) -> None:
        self.metric_map = metric_map or SparseWorldMetricMap()
        self.trajectory = trajectory or TrajectoryGraph()

    def record_observation(
        self, *, depth: np.ndarray, agent_position_xyz: Sequence[float],
        heading_rad: float, intrinsics: Sequence[float], frame_idx: int, step: int,
        episode_id: str, floor_id: int = 0,
        sensor_offset_xyz: Sequence[float] = (0.0, 1.2, 0.0),
        depth_normalized: bool = True, min_depth_m: float = 0.0,
        max_depth_m: float = 10.0, pixel_stride: int = 8, max_rays: int = 2048,
    ) -> dict[str, int]:
        position = np.asarray(agent_position_xyz, dtype=float).reshape(3)
        depth_m = habitat_depth_to_meters(
            depth, normalized=depth_normalized,
            min_depth_m=min_depth_m, max_depth_m=max_depth_m,
        )
        pose = habitat_camera_to_world(
            position, heading_rad, sensor_offset_xyz=sensor_offset_xyz
        )
        stats = self.metric_map.integrate_depth(
            depth_m, pose, intrinsics, floor_height_m=float(position[1]),
            floor_id=floor_id, min_depth_m=max(0.01, min_depth_m),
            max_depth_m=max_depth_m, pixel_stride=pixel_stride, max_rays=max_rays,
        )
        node = self.trajectory.observe(
            position, heading_rad, frame_idx=frame_idx, step=step,
            floor_id=floor_id, episode_id=episode_id,
        )
        return {**stats, "trajectory_node_id": node}

    def record_traversed_pose(
        self, *, agent_position_xyz: Sequence[float], heading_rad: float,
        frame_idx: int, step: int, episode_id: str, floor_id: int = 0,
    ) -> dict[str, int]:
        position = np.asarray(agent_position_xyz, dtype=float).reshape(3)
        if not np.isfinite(position).all() or not math.isfinite(float(heading_rad)):
            raise ValueError("traversed pose must be finite")
        cell = self.metric_map.spec.world_to_cell(position[0], position[2])
        updated = self.metric_map.update_cells([cell], -1.0, floor_id=floor_id)
        node = self.trajectory.observe(
            position, heading_rad, frame_idx=frame_idx, step=step,
            floor_id=floor_id, episode_id=episode_id,
        )
        return {"traversed_free_cells": updated, "trajectory_node_id": node}
