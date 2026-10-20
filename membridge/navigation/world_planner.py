"""Compose occupancy, trajectory certificates and Theta* in world coordinates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from membridge.occupancy_map import SparseWorldMetricMap
from membridge.trajectory_graph import TrajectoryGraph, certified_masks

from .theta_star import theta_star


@dataclass(frozen=True)
class WorldPathPlan:
    status: str
    start_world_grid: tuple[int, int]
    requested_target_world_grid: tuple[int, int]
    target_world_grid: tuple[int, int] | None = None
    path_world_grids: tuple[tuple[int, int], ...] = ()
    path_world_xz: tuple[tuple[float, float], ...] = ()
    path_length_m: float = 0.0
    expanded_states: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _reachable(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    result = np.zeros(np.asarray(mask).shape, dtype=bool)
    if not (0 <= start[0] < result.shape[1] and 0 <= start[1] < result.shape[0] and mask[start[1], start[0]]):
        return result
    queue = deque([start])
    result[start[1], start[0]] = True
    while queue:
        x, z = queue.popleft()
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            neighbor = x + dx, z + dz
            if not (0 <= neighbor[0] < result.shape[1] and 0 <= neighbor[1] < result.shape[0]):
                continue
            if result[neighbor[1], neighbor[0]] or not mask[neighbor[1], neighbor[0]]:
                continue
            if dx and dz and (not mask[z, neighbor[0]] or not mask[neighbor[1], x]):
                continue
            result[neighbor[1], neighbor[0]] = True
            queue.append(neighbor)
    return result


def _nearest_reachable_target(
    reachable: np.ndarray, requested_local: tuple[int, int], maximum_radius_cells: float,
    *, exact: bool,
) -> tuple[int, int] | None:
    x, z = requested_local
    if 0 <= x < reachable.shape[1] and 0 <= z < reachable.shape[0] and reachable[z, x]:
        return requested_local
    if exact:
        return None
    cells_zx = np.argwhere(reachable)
    if not len(cells_zx):
        return None
    distance_squared = (cells_zx[:, 1] - x) ** 2 + (cells_zx[:, 0] - z) ** 2
    eligible = np.flatnonzero(distance_squared <= float(maximum_radius_cells) ** 2 + 1e-9)
    if not len(eligible):
        return None
    best = min(eligible, key=lambda index: (distance_squared[index], cells_zx[index, 0], cells_zx[index, 1]))
    return int(cells_zx[best, 1]), int(cells_zx[best, 0])


class WorldMemoryPlanner:
    def __init__(
        self, metric_map: SparseWorldMetricMap, trajectory: TrajectoryGraph | None = None,
        *, clearance_cells: float = 2.5, trajectory_execution_slack_cells: float = 0.0,
        target_snap_radius_m: float = 3.0, window_padding_cells: int = 40,
        maximum_window_cells: int = 4_000_000,
    ) -> None:
        self.metric_map, self.trajectory = metric_map, trajectory
        self.clearance_cells = max(0.0, float(clearance_cells))
        self.trajectory_execution_slack_cells = max(0.0, float(trajectory_execution_slack_cells))
        self.target_snap_radius_m = max(0.0, float(target_snap_radius_m))
        self.window_padding_cells = max(0, int(window_padding_cells))
        self.maximum_window_cells = int(maximum_window_cells)

    def plan(
        self, start_world_xz: Sequence[float], target_world_xz: Sequence[float], *,
        floor_id: int = 0, exact_target: bool = False,
        runtime_blocked_world_cells: Iterable[Sequence[int]] = (),
        connector_validator=None,
    ) -> WorldPathPlan:
        spec = self.metric_map.spec
        start = spec.world_to_cell(*np.asarray(start_world_xz, dtype=float).reshape(2))
        requested = spec.world_to_cell(*np.asarray(target_world_xz, dtype=float).reshape(2))
        window = self.metric_map.planning_window(
            start, requested, floor_id=floor_id, padding_cells=self.window_padding_cells,
            max_window_cells=self.maximum_window_cells,
        )
        certificate_metadata: dict[str, Any] = {"trajectory_certificate": "not_available"}
        corridor = centers = None
        if self.trajectory is not None:
            corridor, centers, certificate_metadata = certified_masks(
                self.trajectory, window, spec, clearance_cells=self.clearance_cells,
                floor_id=floor_id, warm_start_world_xz=start_world_xz,
                connector_validator=connector_validator,
            )
        safe = window.safe_mask(
            self.clearance_cells, certified_free=corridor, certified_centers=centers,
            certified_center_slack_cells=self.trajectory_execution_slack_cells,
        )
        blocked_count = 0
        for raw_cell in runtime_blocked_world_cells:
            grid_x, grid_z = map(int, raw_cell)
            local_x, local_z = window.global_to_local(grid_x, grid_z)
            if 0 <= local_z < safe.shape[0] and 0 <= local_x < safe.shape[1]:
                safe[local_z, local_x] = False
                blocked_count += 1
        start_local, requested_local = window.global_to_local(*start), window.global_to_local(*requested)
        reachable = _reachable(safe, start_local)
        metadata = {
            **certificate_metadata,
            "reachable_safe_cells": int(np.count_nonzero(reachable)),
            "runtime_blocked_cells": blocked_count,
            "clearance_cells": self.clearance_cells,
        }
        if not reachable.any():
            return WorldPathPlan("start_not_connected_to_safe_space", start, requested, metadata=metadata)
        radius_cells = self.target_snap_radius_m / spec.resolution_m
        target_local = _nearest_reachable_target(reachable, requested_local, radius_cells, exact=exact_target)
        if target_local is None:
            return WorldPathPlan("no_reachable_target_within_snap_radius", start, requested, metadata=metadata)
        result = theta_star(safe, start_local, target_local)
        if not result.path:
            metadata["theta_reason"] = result.reason
            return WorldPathPlan("world_theta_star_failed", start, requested, expanded_states=result.expanded_states, metadata=metadata)
        path_grids = tuple(window.local_to_global(*point) for point in result.path)
        path_world = tuple(spec.cell_center_world(*point) for point in path_grids)
        length = sum(float(np.linalg.norm(np.asarray(right) - left)) for left, right in zip(path_world[:-1], path_world[1:]))
        target_global = window.local_to_global(*target_local)
        metadata["target_resnapped"] = target_global != requested
        return WorldPathPlan(
            "planned", start, requested, target_global, path_grids, path_world,
            length, result.expanded_states, metadata,
        )
