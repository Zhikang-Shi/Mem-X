"""Score safe frontier viewpoints toward a known semantic goal."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

from membridge.occupancy_map.grid import dilate_disk

from .connected import bounded_start_connector, connected_components, neighbor_union, reachable_component


@dataclass(frozen=True)
class FrontierConfig:
    clearance_cells: float = 2.5
    window_padding_cells: int = 60
    maximum_window_cells: int = 1_500_000
    minimum_cluster_cells: int = 3
    minimum_information_gain_cells: int = 1
    viewpoint_extra_cells: int = 4
    information_radius_m: float = 2.5
    obstacle_risk_radius_m: float = 0.35
    blacklist_radius_m: float = 0.45
    maximum_candidates: int = 12
    progress_weight: float = 2.0
    information_weight: float = 0.40
    path_cost_weight: float = 1.0
    turn_cost_weight: float = 0.20
    revisit_weight: float = 2.0
    obstacle_risk_weight: float = 3.0
    start_connector_trigger_cells: int = 4
    start_connector_clearance_cells: float = 1.5
    start_connector_max_distance_m: float = 0.35


@dataclass(frozen=True)
class FrontierCandidate:
    world_grid_x: int
    world_grid_z: int
    world_x: float
    world_z: float
    scan_heading_rad: float
    score: float
    goal_progress_m: float
    path_lower_bound_m: float
    information_gain_cells: int
    turn_cost_rad: float
    revisit_penalty: float
    obstacle_risk: float
    frontier_cluster_cells: int
    frontier_boundary_world_grid: tuple[int, int]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FrontierSelection:
    status: str
    start_world_grid: tuple[int, int]
    goal_world_grids: tuple[tuple[int, int], ...]
    reachable_safe_cells: int
    raw_frontier_cells: int
    frontier_clusters: int
    candidates: tuple[FrontierCandidate, ...]
    start_connector_status: str = "not_needed"
    start_connector_clearance_cells: float = 0.0
    start_connector_world_grids: tuple[tuple[int, int], ...] = ()

    def to_dict(self) -> dict:
        result = asdict(self)
        result["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return result


def _count_near(mask: np.ndarray, x: int, z: int, radius: int) -> int:
    x0, x1 = max(0, x - radius), min(mask.shape[1], x + radius + 1)
    z0, z1 = max(0, z - radius), min(mask.shape[0], z + radius + 1)
    return int(np.count_nonzero(mask[z0:z1, x0:x1]))


def _goal_distance(point: np.ndarray, goals: np.ndarray, resolution: float) -> float:
    return float(np.linalg.norm(goals - point.reshape(1, 2), axis=1).min() * resolution)


class GoalDirectedFrontierSelector:
    def __init__(self, config: FrontierConfig = FrontierConfig()) -> None:
        self.config = config

    def select(
        self, metric_map, start_world_xz: Sequence[float],
        goal_world_xz: Iterable[Sequence[float]], *, current_heading_rad: float | None = None,
        blacklist_world_xz: Iterable[Sequence[float]] = (), floor_id: int = 0,
    ) -> FrontierSelection:
        config, spec = self.config, metric_map.spec
        start_world = np.asarray(start_world_xz, dtype=float).reshape(2)
        goals_world = np.asarray(list(goal_world_xz), dtype=float).reshape(-1, 2)
        if not np.isfinite(start_world).all() or not len(goals_world) or not np.isfinite(goals_world).all():
            raise ValueError("finite start and goal points are required")
        start_global = spec.world_to_cell(*start_world)
        goal_globals = np.asarray([spec.world_to_cell(*point) for point in goals_world], dtype=int)
        closest_goal = min(goal_globals, key=lambda point: np.linalg.norm(point - np.asarray(start_global)))
        window = metric_map.planning_window(
            start_global, closest_goal, floor_id=floor_id,
            padding_cells=config.window_padding_cells, max_window_cells=config.maximum_window_cells,
        )
        start_local = window.global_to_local(*start_global)
        safe = window.safe_mask(config.clearance_cells)
        reachable = reachable_component(safe, start_local)
        connector, connector_status = [], "not_needed"
        connector_clearance = min(config.clearance_cells, config.start_connector_clearance_cells)
        if 0 < np.count_nonzero(reachable) <= config.start_connector_trigger_cells and connector_clearance < config.clearance_cells and config.start_connector_max_distance_m > 0:
            connector = bounded_start_connector(
                safe, window.safe_mask(connector_clearance), start_local,
                int(math.floor(config.start_connector_max_distance_m / spec.resolution_m + 1e-9)),
            )
            connector_status = "bounded_known_free_egress" if connector else "no_bounded_known_free_egress"
            for x, z in connector:
                safe[z, x] = True
            reachable = reachable_component(safe, start_local)
        connector_global = tuple(window.local_to_global(x, z) for x, z in connector)

        def empty(status: str, raw: int = 0, clusters: int = 0) -> FrontierSelection:
            return FrontierSelection(
                status, tuple(start_global), tuple(map(tuple, goal_globals)), int(np.count_nonzero(reachable)),
                raw, clusters, (), connector_status, connector_clearance, connector_global,
            )

        if not reachable.any():
            return empty("start_not_clearance_safe")
        unknown = ~window.observed
        frontier = window.free & neighbor_union(unknown, outside_value=True)
        if not frontier.any():
            return empty("no_free_unknown_boundary")
        clusters = [cluster for cluster in connected_components(frontier) if len(cluster) >= config.minimum_cluster_cells]
        info_radius = max(1, round(config.information_radius_m / spec.resolution_m))
        risk_radius = max(1, round(config.obstacle_risk_radius_m / spec.resolution_m))
        viewpoint_radius = max(1, math.ceil(config.clearance_cells) + config.viewpoint_extra_cells)
        blacklist_values = list(blacklist_world_xz)
        blacklist = np.asarray(blacklist_values, dtype=float).reshape(-1, 2) if blacklist_values else np.empty((0, 2))
        if len(blacklist):
            blacklist = blacklist[np.isfinite(blacklist).all(axis=1)]
        start_goal_distance = _goal_distance(np.asarray(start_global), goal_globals, spec.resolution_m)
        candidates = []
        for cluster in clusters:
            cluster_mask = np.zeros_like(frontier)
            for x, z in cluster:
                cluster_mask[z, x] = True
            viewpoints = np.argwhere(dilate_disk(cluster_mask, viewpoint_radius) & reachable)
            if len(viewpoints) > 512:
                viewpoints = viewpoints[np.linspace(0, len(viewpoints) - 1, 512, dtype=int)]
            best = None
            boundary = np.asarray(cluster)
            for local_z, local_x in viewpoints:
                global_x, global_z = window.local_to_global(local_x, local_z)
                global_point = np.asarray([global_x, global_z])
                world_point = np.asarray(spec.cell_center_world(global_x, global_z))
                progress = start_goal_distance - _goal_distance(global_point, goal_globals, spec.resolution_m)
                path_cost = float(np.linalg.norm(global_point - start_global) * spec.resolution_m)
                information = _count_near(unknown, local_x, local_z, info_radius)
                if information < config.minimum_information_gain_cells:
                    continue
                risk = _count_near(window.occupied, local_x, local_z, risk_radius) / float((2 * risk_radius + 1) ** 2)
                scan_radius = max(1, viewpoint_radius + 1)
                x0, x1 = max(0, local_x-scan_radius), min(unknown.shape[1], local_x+scan_radius+1)
                z0, z1 = max(0, local_z-scan_radius), min(unknown.shape[0], local_z+scan_radius+1)
                unknown_points = np.argwhere(unknown[z0:z1, x0:x1])
                if len(unknown_points):
                    unknown_points += np.asarray([z0, x0])
                    nearest = unknown_points[np.argmin(np.sum((unknown_points - [local_z, local_x]) ** 2, axis=1))]
                    delta_x, delta_z = nearest[1] - local_x, nearest[0] - local_z
                else:
                    delta_x, delta_z = closest_goal[0] - global_x, closest_goal[1] - global_z
                scan_heading = float(math.atan2(-delta_x, -delta_z))
                turn = 0.0 if current_heading_rad is None else abs(math.atan2(math.sin(scan_heading-current_heading_rad), math.cos(scan_heading-current_heading_rad)))
                revisit = float(len(blacklist) and np.linalg.norm(blacklist - world_point, axis=1).min() <= config.blacklist_radius_m)
                score = config.progress_weight*progress + config.information_weight*math.sqrt(information) - config.path_cost_weight*path_cost - config.turn_cost_weight*turn - config.revisit_weight*revisit - config.obstacle_risk_weight*risk
                nearest_boundary = boundary[np.argmin(np.sum((boundary - [local_x, local_z]) ** 2, axis=1))]
                candidate = FrontierCandidate(
                    global_x, global_z, world_point[0], world_point[1], scan_heading, score,
                    progress, path_cost, information, turn, revisit, risk, len(cluster),
                    window.local_to_global(*nearest_boundary),
                )
                rank = (score, progress, information, -path_cost, -global_z, -global_x)
                if best is None or rank > best[0]:
                    best = rank, candidate
            if best and best[1].revisit_penalty < 1:
                candidates.append(best[1])
        candidates.sort(key=lambda item: (-item.score, -item.goal_progress_m, -item.information_gain_cells, item.path_lower_bound_m, item.world_grid_z, item.world_grid_x))
        return FrontierSelection(
            "ok" if candidates else "no_unvisited_safe_frontier", tuple(start_global),
            tuple(map(tuple, goal_globals)), int(np.count_nonzero(reachable)),
            int(np.count_nonzero(frontier)), len(clusters), tuple(candidates[:max(1, config.maximum_candidates)]),
            connector_status, connector_clearance, connector_global,
        )
