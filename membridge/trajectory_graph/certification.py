"""Rasterize persisted trajectory edges into local planning certificates."""

from __future__ import annotations

import math
from typing import Callable, Sequence, Tuple, Union

import numpy as np

from membridge.occupancy_map import WorldGridSpec, WorldPlanningWindow
from membridge.occupancy_map.grid import dilate_disk

from .graph import TrajectoryGraph

ConnectorValidator = Callable[
    [np.ndarray, np.ndarray], Union[bool, Tuple[bool, object]]
]


def _world_line(
    start_world_xz: Sequence[float], end_world_xz: Sequence[float], spec: WorldGridSpec
) -> list[tuple[int, int]]:
    start, end = np.asarray(start_world_xz, dtype=float), np.asarray(end_world_xz, dtype=float)
    samples = max(1, int(math.ceil(np.linalg.norm(end - start) / spec.resolution_m * 2.0)))
    cells: list[tuple[int, int]] = []
    for index in range(samples + 1):
        point = start + index / samples * (end - start)
        cell = spec.world_to_cell(*point)
        if not cells or cell != cells[-1]:
            cells.append(cell)
    return cells


def _nearest_target(
    graph: TrajectoryGraph, start: np.ndarray, eligible_ids: set[int]
) -> tuple[float, str, int | str, np.ndarray] | None:
    candidates: list[tuple[float, str, int | str, np.ndarray]] = []
    for node_id in eligible_ids:
        point = np.asarray(graph.nodes[node_id].position_xyz)[[0, 2]]
        candidates.append((float(np.linalg.norm(start - point)), "node", node_id, point))
    for left, right, _ in graph.edges:
        if left not in eligible_ids or right not in eligible_ids:
            continue
        left_xz = np.asarray(graph.nodes[left].position_xyz)[[0, 2]]
        right_xz = np.asarray(graph.nodes[right].position_xyz)[[0, 2]]
        delta = right_xz - left_xz
        denominator = float(np.dot(delta, delta))
        if denominator <= 1e-12:
            continue
        fraction = float(np.clip(np.dot(start - left_xz, delta) / denominator, 0.0, 1.0))
        point = left_xz + fraction * delta
        candidates.append((float(np.linalg.norm(start - point)), "edge", f"{left}:{right}", point))
    return min(candidates, key=lambda item: (item[0], item[1] != "node", str(item[2]))) if candidates else None


def certified_masks(
    graph: TrajectoryGraph,
    window: WorldPlanningWindow,
    spec: WorldGridSpec,
    *,
    clearance_cells: float,
    floor_id: int = 0,
    warm_start_world_xz: Sequence[float] | None = None,
    max_terminal_connector_m: float = 0.35,
    connector_validator: ConnectorValidator | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return swept corridor, edge centerline and auditable connector metadata."""

    centerline = np.zeros(np.asarray(window.free).shape, dtype=bool)
    eligible_ids = {node.node_id for node in graph.nodes if node.floor_id == int(floor_id)}

    def mark(cell: tuple[int, int]) -> None:
        local_x, local_z = window.global_to_local(*cell)
        if 0 <= local_z < centerline.shape[0] and 0 <= local_x < centerline.shape[1]:
            centerline[local_z, local_x] = True

    for node_id in eligible_ids:
        node = graph.nodes[node_id]
        mark(spec.world_to_cell(node.position_xyz[0], node.position_xyz[2]))
    certified_edges = 0
    for left, right, _ in graph.edges:
        if left not in eligible_ids or right not in eligible_ids:
            continue
        certified_edges += 1
        start = np.asarray(graph.nodes[left].position_xyz)[[0, 2]]
        end = np.asarray(graph.nodes[right].position_xyz)[[0, 2]]
        for cell in _world_line(start, end, spec):
            mark(cell)

    status, distance, cell_count = "not_requested", None, 0
    target_kind = target_id = validation = None
    if warm_start_world_xz is not None:
        start = np.asarray(warm_start_world_xz, dtype=float).reshape(-1)
        if start.size != 2 or not np.isfinite(start).all():
            raise ValueError("warm_start_world_xz must contain two finite values")
        target = _nearest_target(graph, start, eligible_ids)
        if target is None:
            status = "missing_floor_trajectory_nodes"
        else:
            distance, target_kind, target_id, target_xz = target
            if distance > float(max_terminal_connector_m):
                status = "nearest_certified_trajectory_too_far"
            else:
                valid = True
                if connector_validator:
                    checked = connector_validator(start.copy(), target_xz.copy())
                    if isinstance(checked, tuple):
                        valid, validation = bool(checked[0]), checked[1]
                    else:
                        valid = bool(checked)
                if valid:
                    connector = _world_line(start, target_xz, spec)
                    for cell in connector:
                        mark(cell)
                    cell_count = len(connector)
                    status = f"connected_to_nearest_certified_trajectory_{target_kind}"
                else:
                    status = "local_connector_validation_failed"
    corridor = dilate_disk(centerline, max(0.0, float(clearance_cells)))
    metadata = {
        "trajectory_node_count": len(eligible_ids),
        "trajectory_edge_count": certified_edges,
        "trajectory_centerline_cells": int(np.count_nonzero(centerline)),
        "trajectory_certified_free_cells": int(np.count_nonzero(corridor)),
        "warm_start_connector_status": status,
        "warm_start_connector_distance_m": distance,
        "warm_start_connector_cells": cell_count,
        "warm_start_connector_target_kind": target_kind,
        "warm_start_connector_target_id": target_id,
        "warm_start_connector_validation": validation,
    }
    return corridor, centerline, metadata


def certified_free_mask(*args, **kwargs) -> tuple[np.ndarray, dict[str, object]]:
    corridor, _, metadata = certified_masks(*args, **kwargs)
    return corridor, metadata
