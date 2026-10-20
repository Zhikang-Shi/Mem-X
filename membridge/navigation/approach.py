"""Convert an SG object centre into same-island, standable terminal poses."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ProjectedApproachCandidate:
    source: str
    raw_world_xz: tuple[float, float]
    projected_world_xyz: tuple[float, float, float]
    projected_world_xz: tuple[float, float]
    object_distance_m: float
    route_anchor_distance_m: float
    navmesh_snap_distance_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def finite_xz(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    result = array[:2] if array.size == 2 else array[[0, 2]] if array.size >= 3 else None
    return result.copy() if result is not None and np.isfinite(result).all() else None


def object_center_xz(target: Mapping[str, Any]) -> np.ndarray:
    center = finite_xz(target.get("position"))
    if center is None:
        raise ValueError("SG target has no finite object centre")
    return center


def raw_approach_candidates(
    target: Mapping[str, Any], route_anchor_world_xz: Sequence[Sequence[float]] = (), *,
    max_object_distance_m: float = 3.0, annulus_step_m: float = 0.10,
    angular_samples: int = 64,
) -> list[tuple[str, np.ndarray]]:
    if max_object_distance_m <= 0 or annulus_step_m <= 0 or angular_samples < 4:
        raise ValueError("invalid approach sampling settings")
    center = object_center_xz(target)
    result = [("object_center_navmesh_projection", center.copy())]
    observation = target.get("best_observation") or {}
    observation_xz = finite_xz(observation.get("agent_position"))
    if observation_xz is None:
        observation_xz = finite_xz((target.get("waypoint") or {}).get("position"))
    if observation_xz is not None:
        result.append(("best_observation_pose", observation_xz))
        delta, distance = observation_xz - center, float(np.linalg.norm(observation_xz - center))
        if distance > 1e-9:
            for standoff in (0.75, 1.0, 1.25, 1.5, 2.0, 2.5):
                if standoff <= max_object_distance_m + 1e-9:
                    result.append((f"observation_ray_standoff_{standoff:.2f}m", center + delta / distance * standoff))
    anchors = [point for point in (finite_xz(value) for value in route_anchor_world_xz) if point is not None]
    anchors.sort(key=lambda point: (float(np.linalg.norm(point - center)), point[0], point[1]))
    for index, point in enumerate(anchors[:32]):
        if 0.25 <= np.linalg.norm(point - center) <= max_object_distance_m + 0.5:
            result.append((f"target_route_anchor_{index:02d}", point))
    for radius_index in range(1, int(math.floor(max_object_distance_m / annulus_step_m + 1e-9)) + 1):
        radius = radius_index * annulus_step_m
        for angle_index in range(angular_samples):
            angle = 2 * math.pi * angle_index / angular_samples
            result.append((f"object_annulus_{radius:.2f}m_{angle_index:02d}", center + radius * np.asarray([math.cos(angle), math.sin(angle)])))
    return result


def project_approach_candidates(
    raw_candidates: Sequence[tuple[str, Sequence[float]]], *, object_xz: Sequence[float],
    route_anchor_world_xz: Sequence[Sequence[float]], pathfinder: Any,
    current_world_xz: Sequence[float], floor_y: float,
    min_object_distance_m: float = 0.0, max_object_distance_m: float = 3.0,
    max_navmesh_snap_m: float = 0.75, dedup_resolution_m: float = 0.10,
    connectivity_check: Callable[[Any, np.ndarray, np.ndarray], bool | None] | None = None,
) -> list[ProjectedApproachCandidate]:
    center, current = finite_xz(object_xz), finite_xz(current_world_xz)
    if center is None or current is None or not math.isfinite(float(floor_y)):
        raise ValueError("invalid object/current/floor coordinates")
    if dedup_resolution_m <= 0:
        raise ValueError("dedup_resolution_m must be positive")
    route = [point for point in (finite_xz(value) for value in route_anchor_world_xz) if point is not None] or [center]
    route_array = np.stack(route)
    current_xyz = np.asarray([current[0], floor_y, current[1]], dtype=np.float32)
    try:
        if not pathfinder.is_navigable(current_xyz):
            snapped = np.asarray(pathfinder.snap_point(current_xyz), dtype=float).reshape(-1)
            if snapped.size < 3 or np.linalg.norm(snapped[[0, 2]] - current) > 0.25:
                return []
            current_xyz = snapped[:3]
        has_island = hasattr(pathfinder, "get_island")
        island = int(pathfinder.get_island(current_xyz)) if has_island else None
    except Exception:
        return []
    projected, by_cell = [], {}
    for source, raw_value in raw_candidates:
        raw = finite_xz(raw_value)
        if raw is None:
            continue
        raw_xyz = np.asarray([raw[0], floor_y, raw[1]], dtype=np.float32)
        try:
            snapped = np.asarray(pathfinder.snap_point(raw_xyz, island) if has_island else pathfinder.snap_point(raw_xyz), dtype=float).reshape(-1)
            if snapped.size < 3 or not np.isfinite(snapped[:3]).all() or not pathfinder.is_navigable(snapped[:3]):
                continue
            if has_island and int(pathfinder.get_island(snapped[:3])) != island:
                continue
            if not has_island and connectivity_check and connectivity_check(pathfinder, current_xyz, snapped[:3]) is not True:
                continue
        except Exception:
            continue
        snapped_xz = snapped[[0, 2]]
        snap_distance, object_distance = np.linalg.norm(snapped_xz - raw), np.linalg.norm(snapped_xz - center)
        if snap_distance > max_navmesh_snap_m + 1e-6 or not min_object_distance_m - 1e-6 <= object_distance <= max_object_distance_m + 1e-6:
            continue
        candidate = ProjectedApproachCandidate(
            str(source), tuple(map(float, raw)), tuple(map(float, snapped[:3])), tuple(map(float, snapped_xz)),
            float(object_distance), float(np.min(np.linalg.norm(route_array - snapped_xz, axis=1))), float(snap_distance),
        )
        key = tuple(np.round(snapped_xz / dedup_resolution_m).astype(int))
        rank = lambda item: (item.object_distance_m, item.navmesh_snap_distance_m, item.route_anchor_distance_m, item.source)
        if key not in by_cell:
            by_cell[key] = len(projected)
            projected.append(candidate)
        elif rank(candidate) < rank(projected[by_cell[key]]):
            projected[by_cell[key]] = candidate
    return sorted(projected, key=lambda item: (item.object_distance_m, item.navmesh_snap_distance_m, item.route_anchor_distance_m, item.source))


def build_projected_approach_candidates(
    target: Mapping[str, Any], route_anchor_world_xz: Sequence[Sequence[float]], **kwargs: Any
) -> tuple[list[ProjectedApproachCandidate], dict[str, Any]]:
    sampling_keys = {"max_object_distance_m", "annulus_step_m", "angular_samples"}
    sampling = {key: kwargs[key] for key in sampling_keys if key in kwargs}
    projection = {key: value for key, value in kwargs.items() if key not in {"annulus_step_m", "angular_samples"}}
    raw = raw_approach_candidates(target, route_anchor_world_xz, **sampling)
    projected = project_approach_candidates(
        raw, object_xz=object_center_xz(target), route_anchor_world_xz=route_anchor_world_xz, **projection
    )
    return projected, {
        "approach_policy": "v8_dense_same_island_closest_object",
        "approach_raw_candidate_count": len(raw), "approach_navmesh_candidate_count": len(projected),
    }
