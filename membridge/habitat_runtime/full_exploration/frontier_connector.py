"""Goal-directed frontier selection on an absolute sparse navigation map."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


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

    @classmethod
    def from_environment(cls) -> "FrontierConfig":
        return cls(
            clearance_cells=_env_float(
                "LIFELONG_SG_EXPLORE_CLEARANCE_CELLS",
                _env_float("LIFELONG_SG_DISCRETE_CLEARANCE_CELLS", 2.5),
            ),
            window_padding_cells=_env_int(
                "LIFELONG_SG_EXPLORE_WINDOW_PADDING_CELLS", 60
            ),
            maximum_window_cells=_env_int(
                "LIFELONG_SG_EXPLORE_MAX_WINDOW_CELLS", 1_500_000
            ),
            minimum_cluster_cells=_env_int(
                "LIFELONG_SG_EXPLORE_MIN_FRONTIER_CELLS", 3
            ),
            minimum_information_gain_cells=max(
                1,
                _env_int("LIFELONG_SG_EXPLORE_MIN_INFORMATION_CELLS", 1),
            ),
            viewpoint_extra_cells=_env_int(
                "LIFELONG_SG_EXPLORE_VIEWPOINT_EXTRA_CELLS", 4
            ),
            information_radius_m=_env_float(
                "LIFELONG_SG_EXPLORE_INFORMATION_RADIUS_M", 2.5
            ),
            obstacle_risk_radius_m=_env_float(
                "LIFELONG_SG_EXPLORE_OBSTACLE_RISK_RADIUS_M", 0.35
            ),
            blacklist_radius_m=_env_float(
                "LIFELONG_SG_EXPLORE_BLACKLIST_RADIUS_M", 0.45
            ),
            maximum_candidates=_env_int(
                "LIFELONG_SG_EXPLORE_MAX_FRONTIER_CANDIDATES", 12
            ),
            progress_weight=_env_float(
                "LIFELONG_SG_EXPLORE_PROGRESS_WEIGHT", 4.0
            ),
            information_weight=_env_float(
                "LIFELONG_SG_EXPLORE_INFORMATION_WEIGHT", 0.25
            ),
            path_cost_weight=_env_float(
                "LIFELONG_SG_EXPLORE_PATH_COST_WEIGHT", 1.0
            ),
            turn_cost_weight=_env_float(
                "LIFELONG_SG_EXPLORE_TURN_COST_WEIGHT", 0.15
            ),
            revisit_weight=_env_float(
                "LIFELONG_SG_EXPLORE_REVISIT_WEIGHT", 2.0
            ),
            obstacle_risk_weight=_env_float(
                "LIFELONG_SG_EXPLORE_OBSTACLE_RISK_WEIGHT", 3.0
            ),
            start_connector_trigger_cells=max(
                1,
                _env_int(
                    "LIFELONG_SG_EXPLORE_START_CONNECTOR_TRIGGER_CELLS", 4
                ),
            ),
            start_connector_clearance_cells=max(
                0.0,
                _env_float(
                    "LIFELONG_SG_EXPLORE_START_CONNECTOR_CLEARANCE_CELLS",
                    1.5,
                ),
            ),
            start_connector_max_distance_m=max(
                0.0,
                _env_float(
                    "LIFELONG_SG_EXPLORE_START_CONNECTOR_MAX_DISTANCE_M",
                    0.35,
                ),
            ),
        )


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
    information_gain_extent_m: float
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
        result["candidates"] = [item.to_dict() for item in self.candidates]
        return result


def _shift_neighbor_union(mask: np.ndarray, *, outside_value: bool) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    padded = np.pad(source, 1, mode="constant", constant_values=outside_value)
    result = np.zeros_like(source)
    height, width = source.shape
    for dz in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dz == 0:
                continue
            result |= padded[
                1 + dz : 1 + dz + height,
                1 + dx : 1 + dx + width,
            ]
    return result


def _reachable_component(mask: np.ndarray, start_xy: tuple[int, int]):
    safe = np.asarray(mask, dtype=np.uint8)
    start_x, start_z = map(int, start_xy)
    if not (
        0 <= start_z < safe.shape[0]
        and 0 <= start_x < safe.shape[1]
        and bool(safe[start_z, start_x])
    ):
        return np.zeros_like(safe, dtype=bool)
    count, labels = cv2.connectedComponents(safe, connectivity=8)
    if count <= 1:
        return np.zeros_like(safe, dtype=bool)
    label = int(labels[start_z, start_x])
    return labels == label


def _bounded_start_connector(
    ordinary_safe: np.ndarray,
    connector_safe: np.ndarray,
    start_xy: tuple[int, int],
    maximum_steps: int,
) -> list[tuple[int, int]]:
    """Find the shortest known-free egress into another ordinary component.

    The relaxed mask is never used as a global planning layer.  It only admits
    a short centreline from the robot's physically valid current pose to a
    normal-clearance component.  Runtime's action-lattice gate subsequently
    validates every MOVE_FORWARD with Habitat ``try_step`` before this
    connector can be installed.
    """

    ordinary = np.asarray(ordinary_safe, dtype=bool)
    connector = np.asarray(connector_safe, dtype=bool)
    if ordinary.shape != connector.shape or ordinary.ndim != 2:
        return []
    start = tuple(map(int, start_xy))
    start_x, start_z = start
    height, width = ordinary.shape
    if not (
        0 <= start_x < width
        and 0 <= start_z < height
        and connector[start_z, start_x]
    ):
        return []
    initial_component = _reachable_component(ordinary, start)
    radius = max(0, int(maximum_steps))
    queue = [(start, 0)]
    parent = {start: None}
    cursor = 0
    endpoint = None
    while cursor < len(queue):
        current, depth = queue[cursor]
        cursor += 1
        if (
            current != start
            and ordinary[current[1], current[0]]
            and not initial_component[current[1], current[0]]
        ):
            endpoint = current
            break
        if depth >= radius:
            continue
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (current[0] + dx, current[1] + dz)
            x, z = neighbor
            if not (0 <= x < width and 0 <= z < height):
                continue
            if neighbor in parent or not connector[z, x]:
                continue
            parent[neighbor] = current
            queue.append((neighbor, depth + 1))
    if endpoint is None:
        return []
    path = []
    current = endpoint
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def _angle_distance(first: float, second: float) -> float:
    delta = float(first) - float(second)
    return abs(float(math.atan2(math.sin(delta), math.cos(delta))))


def _square_count(integral: np.ndarray, x: int, z: int, radius: int) -> int:
    height = integral.shape[0] - 1
    width = integral.shape[1] - 1
    x0 = max(0, int(x) - int(radius))
    x1 = min(width, int(x) + int(radius) + 1)
    z0 = max(0, int(z) - int(radius))
    z1 = min(height, int(z) + int(radius) + 1)
    return int(
        integral[z1, x1]
        - integral[z0, x1]
        - integral[z1, x0]
        + integral[z0, x0]
    )


def _integral(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=np.int64)
    result = np.zeros((source.shape[0] + 1, source.shape[1] + 1), dtype=np.int64)
    result[1:, 1:] = source.cumsum(axis=0).cumsum(axis=1)
    return result


def _nearest_goal_distance_m(
    point_global: np.ndarray,
    goal_global: np.ndarray,
    resolution_m: float,
) -> float:
    return float(
        np.linalg.norm(goal_global - point_global.reshape(1, 2), axis=1).min()
        * resolution_m
    )


def _subsample_rows(points: np.ndarray, maximum: int = 512) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices]


class GoalDirectedFrontierSelector:
    """Rank safe viewpoints that expose unknown space toward a known goal."""

    def __init__(self, config: FrontierConfig | None = None):
        self.config = config or FrontierConfig.from_environment()

    def select(
        self,
        metric_map,
        start_world_xz: Sequence[float],
        goal_world_xz: Iterable[Sequence[float]],
        *,
        current_heading_rad: float | None = None,
        blacklist_world_xz: Iterable[Sequence[float]] = (),
    ) -> FrontierSelection:
        config = self.config
        spec = metric_map.spec
        start_world = np.asarray(start_world_xz, dtype=np.float64).reshape(2)
        goals_world = np.asarray(list(goal_world_xz), dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(start_world).all() or not len(goals_world):
            raise ValueError("finite start and at least one goal point are required")
        goals_world = goals_world[np.isfinite(goals_world).all(axis=1)]
        if not len(goals_world):
            raise ValueError("goal points must be finite")

        start_global = spec.world_to_cell(*start_world)
        goal_globals = np.asarray(
            [spec.world_to_cell(*point) for point in goals_world], dtype=np.int64
        )
        closest_goal = goal_globals[
            int(
                np.argmin(
                    np.linalg.norm(
                        goal_globals - np.asarray(start_global).reshape(1, 2),
                        axis=1,
                    )
                )
            )
        ]
        window = metric_map.planning_window(
            start_global,
            closest_goal,
            floor_id=0,
            padding_cells=config.window_padding_cells,
            max_window_cells=config.maximum_window_cells,
        )
        start_local = window.global_to_local(*start_global)
        safe = window.safe_mask(config.clearance_cells)
        reachable = _reachable_component(safe, start_local)
        reachable_count = int(np.count_nonzero(reachable))
        start_connector_status = "not_needed"
        start_connector_local: list[tuple[int, int]] = []
        start_connector_clearance = min(
            float(config.clearance_cells),
            float(config.start_connector_clearance_cells),
        )
        if (
            0 < reachable_count <= int(config.start_connector_trigger_cells)
            and start_connector_clearance + 1e-9
            < float(config.clearance_cells)
            and float(config.start_connector_max_distance_m) > 0.0
        ):
            connector_safe = window.safe_mask(start_connector_clearance)
            maximum_steps = int(
                math.floor(
                    float(config.start_connector_max_distance_m)
                    / float(spec.resolution_m)
                    + 1e-9
                )
            )
            start_connector_local = _bounded_start_connector(
                safe,
                connector_safe,
                start_local,
                maximum_steps,
            )
            if start_connector_local:
                safe = safe.copy()
                for local_x, local_z in start_connector_local:
                    safe[local_z, local_x] = True
                reachable = _reachable_component(safe, start_local)
                reachable_count = int(np.count_nonzero(reachable))
                start_connector_status = "bounded_known_free_egress"
            else:
                start_connector_status = "no_bounded_known_free_egress"
        start_connector_global = tuple(
            tuple(map(int, window.local_to_global(local_x, local_z)))
            for local_x, local_z in start_connector_local
        )
        empty = lambda status: FrontierSelection(
            status=status,
            start_world_grid=tuple(map(int, start_global)),
            goal_world_grids=tuple(tuple(map(int, row)) for row in goal_globals),
            reachable_safe_cells=reachable_count,
            raw_frontier_cells=0,
            frontier_clusters=0,
            candidates=(),
            start_connector_status=start_connector_status,
            start_connector_clearance_cells=float(start_connector_clearance),
            start_connector_world_grids=start_connector_global,
        )
        if reachable_count == 0:
            return empty("start_not_clearance_safe")

        unknown = np.logical_not(window.observed)
        adjacent_unknown = _shift_neighbor_union(unknown, outside_value=True)
        raw_frontier = np.logical_and(window.free, adjacent_unknown)
        raw_frontier_count = int(np.count_nonzero(raw_frontier))
        if raw_frontier_count == 0:
            return empty("no_free_unknown_boundary")

        cluster_count, labels = cv2.connectedComponents(
            raw_frontier.astype(np.uint8), connectivity=8
        )
        unknown_integral = _integral(unknown)
        obstacle_integral = _integral(window.occupied)
        info_radius = max(1, int(round(config.information_radius_m / spec.resolution_m)))
        risk_radius = max(1, int(round(config.obstacle_risk_radius_m / spec.resolution_m)))
        viewpoint_radius = max(
            1,
            int(math.ceil(config.clearance_cells))
            + max(0, int(config.viewpoint_extra_cells)),
        )
        kernel_size = viewpoint_radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        blacklist = np.asarray(list(blacklist_world_xz), dtype=np.float64)
        if blacklist.size:
            blacklist = blacklist.reshape(-1, 2)
            blacklist = blacklist[np.isfinite(blacklist).all(axis=1)]
        else:
            blacklist = np.empty((0, 2), dtype=np.float64)

        start_goal_distance = _nearest_goal_distance_m(
            np.asarray(start_global, dtype=np.float64),
            goal_globals.astype(np.float64),
            spec.resolution_m,
        )
        candidates: list[FrontierCandidate] = []
        retained_clusters = 0
        for cluster_label in range(1, int(cluster_count)):
            cluster = labels == cluster_label
            cluster_cells = int(np.count_nonzero(cluster))
            if cluster_cells < config.minimum_cluster_cells:
                continue
            viewpoint_zone = cv2.dilate(
                cluster.astype(np.uint8), kernel, iterations=1
            ).astype(bool)
            viewpoint_zone &= reachable
            points_zx = np.argwhere(viewpoint_zone)
            if not len(points_zx):
                continue
            retained_clusters += 1
            boundary_zx = np.argwhere(cluster)
            boundary_centroid_zx = boundary_zx.mean(axis=0)
            points_zx = _subsample_rows(points_zx)
            best = None
            for local_z, local_x in points_zx:
                global_x, global_z = window.local_to_global(local_x, local_z)
                point_global = np.asarray([global_x, global_z], dtype=np.float64)
                point_world = np.asarray(
                    spec.cell_center_world(global_x, global_z), dtype=np.float64
                )
                goal_distance = _nearest_goal_distance_m(
                    point_global,
                    goal_globals.astype(np.float64),
                    spec.resolution_m,
                )
                progress = start_goal_distance - goal_distance
                path_lower_bound = float(
                    np.linalg.norm(point_global - np.asarray(start_global))
                    * spec.resolution_m
                )
                information = _square_count(
                    unknown_integral, local_x, local_z, info_radius
                )
                if information < config.minimum_information_gain_cells:
                    continue
                risk_count = _square_count(
                    obstacle_integral, local_x, local_z, risk_radius
                )
                risk_area = float((risk_radius * 2 + 1) ** 2)
                obstacle_risk = float(risk_count) / risk_area

                # Face the nearest unknown cell, not the centroid of a long
                # frontier cluster.  A room perimeter is often one connected
                # cluster; its centroid lies inside the room and would turn
                # the camera away from the unexplored side.  Goal distance is
                # only a deterministic tie-break between equally near unknown
                # neighbours.
                scan_radius = max(1, viewpoint_radius + 1)
                scan_x0 = max(0, int(local_x) - scan_radius)
                scan_x1 = min(unknown.shape[1], int(local_x) + scan_radius + 1)
                scan_z0 = max(0, int(local_z) - scan_radius)
                scan_z1 = min(unknown.shape[0], int(local_z) + scan_radius + 1)
                nearby_unknown = np.argwhere(
                    unknown[scan_z0:scan_z1, scan_x0:scan_x1]
                )
                if len(nearby_unknown):
                    nearby_unknown[:, 0] += scan_z0
                    nearby_unknown[:, 1] += scan_x0
                    deltas_zx = nearby_unknown - np.asarray(
                        [local_z, local_x], dtype=np.int64
                    )
                    distance_sq = np.sum(deltas_zx * deltas_zx, axis=1)
                    nearest_distance_sq = int(distance_sq.min())
                    nearest = nearby_unknown[distance_sq == nearest_distance_sq]
                    nearest_global = np.column_stack(
                        (
                            nearest[:, 1] + window.origin_grid_x,
                            nearest[:, 0] + window.origin_grid_z,
                        )
                    )
                    nearest_goal_distance = np.linalg.norm(
                        nearest_global[:, None, :] - goal_globals[None, :, :],
                        axis=2,
                    ).min(axis=1)
                    scan_unknown_zx = nearest[
                        int(np.argmin(nearest_goal_distance))
                    ]
                    boundary_delta = np.asarray(
                        [
                            float(scan_unknown_zx[1] - local_x),
                            float(scan_unknown_zx[0] - local_z),
                        ],
                        dtype=np.float64,
                    )
                else:
                    boundary_delta = np.asarray(
                        [
                            boundary_centroid_zx[1] - float(local_x),
                            boundary_centroid_zx[0] - float(local_z),
                        ],
                        dtype=np.float64,
                    )
                if float(np.linalg.norm(boundary_delta)) <= 1e-9:
                    goal_local = closest_goal - np.asarray(
                        [window.origin_grid_x, window.origin_grid_z]
                    )
                    boundary_delta = goal_local - np.asarray(
                        [local_x, local_z], dtype=np.float64
                    )
                scan_heading = float(
                    math.atan2(-boundary_delta[0], -boundary_delta[1])
                )
                turn_cost = (
                    0.0
                    if current_heading_rad is None
                    else _angle_distance(scan_heading, current_heading_rad)
                )
                revisit_penalty = 0.0
                if len(blacklist):
                    nearest_blacklist = float(
                        np.linalg.norm(blacklist - point_world, axis=1).min()
                    )
                    if nearest_blacklist <= config.blacklist_radius_m:
                        revisit_penalty = 1.0

                # Progress and path cost are measured in metres.  The old
                # score mixed those quantities with sqrt(cell_count), making
                # information gain 20x larger when the map resolution changed
                # from 1 m to 5 cm.  Convert the square-root area proxy back to
                # a physical linear extent before combining the terms.
                information_extent_m = (
                    math.sqrt(max(0, information)) * float(spec.resolution_m)
                )
                score = (
                    config.progress_weight * progress
                    + config.information_weight * information_extent_m
                    - config.path_cost_weight * path_lower_bound
                    - config.turn_cost_weight * turn_cost
                    - config.revisit_weight * revisit_penalty
                    - config.obstacle_risk_weight * obstacle_risk
                )
                key = (
                    float(score),
                    float(progress),
                    int(information),
                    -float(path_lower_bound),
                    -int(global_z),
                    -int(global_x),
                )
                if best is None or key > best[0]:
                    nearest_boundary = boundary_zx[
                        int(
                            np.argmin(
                                np.sum(
                                    (
                                        boundary_zx
                                        - np.asarray([local_z, local_x])
                                    )
                                    ** 2,
                                    axis=1,
                                )
                            )
                        )
                    ]
                    boundary_global = window.local_to_global(
                        int(nearest_boundary[1]), int(nearest_boundary[0])
                    )
                    best = (
                        key,
                        FrontierCandidate(
                            world_grid_x=int(global_x),
                            world_grid_z=int(global_z),
                            world_x=float(point_world[0]),
                            world_z=float(point_world[1]),
                            scan_heading_rad=scan_heading,
                            score=float(score),
                            goal_progress_m=float(progress),
                            path_lower_bound_m=path_lower_bound,
                            information_gain_cells=int(information),
                            information_gain_extent_m=float(
                                information_extent_m
                            ),
                            turn_cost_rad=float(turn_cost),
                            revisit_penalty=float(revisit_penalty),
                            obstacle_risk=float(obstacle_risk),
                            frontier_cluster_cells=cluster_cells,
                            frontier_boundary_world_grid=tuple(
                                map(int, boundary_global)
                            ),
                        ),
                    )
            if best is not None and best[1].revisit_penalty < 1.0:
                candidates.append(best[1])

        candidates.sort(
            key=lambda item: (
                -item.score,
                -item.goal_progress_m,
                -item.information_gain_cells,
                item.path_lower_bound_m,
                item.world_grid_z,
                item.world_grid_x,
            )
        )
        candidates = candidates[: max(1, config.maximum_candidates)]
        status = "ok" if candidates else "no_unvisited_safe_frontier"
        return FrontierSelection(
            status=status,
            start_world_grid=tuple(map(int, start_global)),
            goal_world_grids=tuple(tuple(map(int, row)) for row in goal_globals),
            reachable_safe_cells=reachable_count,
            raw_frontier_cells=raw_frontier_count,
            frontier_clusters=retained_clusters,
            candidates=tuple(candidates),
            start_connector_status=start_connector_status,
            start_connector_clearance_cells=float(start_connector_clearance),
            start_connector_world_grids=start_connector_global,
        )


__all__ = [
    "FrontierCandidate",
    "FrontierConfig",
    "FrontierSelection",
    "GoalDirectedFrontierSelector",
    "_bounded_start_connector",
]
