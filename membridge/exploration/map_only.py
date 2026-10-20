"""Query-time map updates that prove semantic SG and trajectory remain frozen."""

from __future__ import annotations

import hashlib
import math
import pickle
from typing import Any, Sequence

import numpy as np

from membridge.occupancy_map import SparseWorldMetricMap
from membridge.trajectory_graph import habitat_camera_to_world, habitat_depth_to_meters


def semantic_digest(value: Any) -> str:
    try:
        payload = pickle.dumps(value, protocol=4)
    except Exception:
        payload = repr(value).encode("utf-8", errors="backslashreplace")
    return hashlib.sha256(payload).hexdigest()


class QueryMapUpdater:
    """Owns only the query navigation map; no SG builder or trajectory API exists here."""

    def __init__(
        self, metric_map: SparseWorldMetricMap, *, frozen_semantic_state: Any,
        frozen_trajectory_state: Any, agent_radius_m: float = 0.10,
        maximum_certified_step_m: float = 0.40,
    ) -> None:
        self.metric_map = metric_map
        self._semantic_state, self._trajectory_state = frozen_semantic_state, frozen_trajectory_state
        self._semantic_digest = semantic_digest(frozen_semantic_state)
        self._trajectory_digest = semantic_digest(frozen_trajectory_state)
        self.agent_radius_m = max(0.01, float(agent_radius_m))
        self.maximum_certified_step_m = max(0.25, float(maximum_certified_step_m))
        self.last_position_xz: np.ndarray | None = None
        self.trace: list[dict[str, Any]] = []

    def assert_frozen(self) -> None:
        if semantic_digest(self._semantic_state) != self._semantic_digest:
            raise RuntimeError("query map update mutated frozen semantic SG")
        if semantic_digest(self._trajectory_state) != self._trajectory_digest:
            raise RuntimeError("query map update mutated frozen acquisition trajectory")

    def certify_footprint(self, position_xz: Sequence[float], *, floor_id: int = 0) -> int:
        current, spec = np.asarray(position_xz, dtype=float).reshape(2), self.metric_map.spec
        samples = [current]
        if self.last_position_xz is not None:
            distance = float(np.linalg.norm(current - self.last_position_xz))
            if math.isfinite(distance) and distance <= self.maximum_certified_step_m:
                count = max(1, math.ceil(distance / (0.5 * spec.resolution_m)))
                samples = [self.last_position_xz + (current - self.last_position_xz) * index / count for index in range(count + 1)]
        radius = self.agent_radius_m / spec.resolution_m + math.sqrt(0.5)
        cells = set()
        for sample in samples:
            center = spec.world_to_cell(*sample)
            for dz in range(-math.ceil(radius), math.ceil(radius) + 1):
                for dx in range(-math.ceil(radius), math.ceil(radius) + 1):
                    if dx*dx + dz*dz <= radius*radius + 1e-9:
                        cells.add((center[0] + dx, center[1] + dz))
        self.last_position_xz = current.copy()
        return self.metric_map.set_classified_cells(free_cells=cells, occupied_cells=(), floor_id=floor_id)["free_cells"]

    def record_observation(
        self, *, depth: np.ndarray, agent_position_xyz: Sequence[float], heading_rad: float,
        intrinsics: Sequence[float], frame_idx: int, step: int, sensor_offset_xyz: Sequence[float] = (0, 1.2, 0),
        depth_normalized: bool = True, min_depth_m: float = 0.0, max_depth_m: float = 10.0,
        pixel_stride: int = 4, max_rays: int = 4096, floor_id: int = 0,
    ) -> dict[str, int]:
        position = np.asarray(agent_position_xyz, dtype=float).reshape(3)
        depth_m = habitat_depth_to_meters(depth, normalized=depth_normalized, min_depth_m=min_depth_m, max_depth_m=max_depth_m)
        pose = habitat_camera_to_world(position, heading_rad, sensor_offset_xyz=sensor_offset_xyz)
        stats = self.metric_map.integrate_depth(
            depth_m, pose, intrinsics, floor_height_m=position[1], floor_id=floor_id,
            min_depth_m=max(0.01, min_depth_m), max_depth_m=max_depth_m,
            pixel_stride=pixel_stride, max_rays=max_rays,
        )
        traversed = self.certify_footprint(position[[0, 2]], floor_id=floor_id)
        self.trace.append({"frame_idx": frame_idx, "step": step, "position": position.tolist(), "heading_rad": heading_rad, "map_stats": stats, "traversed_free_cells": traversed})
        self.assert_frozen()
        return {**stats, "traversed_free_cells": traversed}
