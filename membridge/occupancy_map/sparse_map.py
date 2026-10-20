"""Sparse log-odds occupancy memory in absolute Habitat world coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .grid import WorldGridSpec, WorldPlanningWindow


@dataclass
class MetricChunk:
    log_odds: np.ndarray
    observed_count: np.ndarray


class SparseWorldMetricMap:
    """Unbounded XZ map split into chunks; negative cell indices are supported."""

    def __init__(
        self,
        spec: WorldGridSpec | None = None,
        *,
        log_odds_occupied: float = 0.85,
        log_odds_free: float = -0.25,
        log_odds_min: float = -4.0,
        log_odds_max: float = 4.0,
        occupied_threshold: float = 0.85,
        free_threshold: float = -0.40,
    ) -> None:
        self.spec = spec or WorldGridSpec()
        self.log_odds_occupied = float(log_odds_occupied)
        self.log_odds_free = float(log_odds_free)
        self.log_odds_min = float(log_odds_min)
        self.log_odds_max = float(log_odds_max)
        self.occupied_threshold = float(occupied_threshold)
        self.free_threshold = float(free_threshold)
        if self.log_odds_min >= self.log_odds_max:
            raise ValueError("log_odds_min must be smaller than log_odds_max")
        if self.free_threshold >= self.occupied_threshold:
            raise ValueError("free_threshold must be smaller than occupied_threshold")
        self._chunks: dict[tuple[int, int, int], MetricChunk] = {}

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def _chunk(
        self, floor_id: int, chunk_x: int, chunk_z: int, *, create: bool
    ) -> MetricChunk | None:
        key = (int(floor_id), int(chunk_x), int(chunk_z))
        chunk = self._chunks.get(key)
        if chunk is None and create:
            shape = (self.spec.chunk_size, self.spec.chunk_size)
            chunk = MetricChunk(
                np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=np.uint16),
            )
            self._chunks[key] = chunk
        return chunk

    @staticmethod
    def _cells(cells: Iterable[Sequence[int]]) -> np.ndarray:
        raw = np.asarray(cells if isinstance(cells, np.ndarray) else list(cells))
        if raw.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        return np.unique(raw.astype(np.int64).reshape(-1, 2), axis=0)

    def update_cells(
        self, cells: Iterable[Sequence[int]], delta: float, *, floor_id: int = 0
    ) -> int:
        unique = self._cells(cells)
        grouped: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for grid_x, grid_z in unique:
            chunk_x, chunk_z, local_x, local_z = self.spec.split_cell(grid_x, grid_z)
            grouped.setdefault((chunk_x, chunk_z), []).append((local_x, local_z))
        for (chunk_x, chunk_z), local_cells in grouped.items():
            chunk = self._chunk(floor_id, chunk_x, chunk_z, create=True)
            assert chunk is not None
            local = np.asarray(local_cells, dtype=np.int64)
            x, z = local[:, 0], local[:, 1]
            chunk.log_odds[z, x] = np.clip(
                chunk.log_odds[z, x] + float(delta),
                self.log_odds_min,
                self.log_odds_max,
            )
            counts = chunk.observed_count[z, x].astype(np.uint32) + 1
            chunk.observed_count[z, x] = np.minimum(
                counts, np.iinfo(np.uint16).max
            ).astype(np.uint16)
        return int(unique.shape[0])

    def set_classified_cells(
        self,
        *,
        free_cells: Iterable[Sequence[int]],
        occupied_cells: Iterable[Sequence[int]],
        floor_id: int = 0,
    ) -> dict[str, int]:
        """Fuse a finalized navigation raster; obstacle wins over free."""

        free = self._cells(free_cells)
        occupied = self._cells(occupied_cells)
        occupied_set = {tuple(map(int, cell)) for cell in occupied}
        if occupied_set and free.size:
            free = np.asarray(
                [cell for cell in free if tuple(map(int, cell)) not in occupied_set],
                dtype=np.int64,
            ).reshape(-1, 2)

        def assign(cells: np.ndarray, value: float) -> int:
            for grid_x, grid_z in cells:
                chunk_x, chunk_z, local_x, local_z = self.spec.split_cell(
                    int(grid_x), int(grid_z)
                )
                chunk = self._chunk(floor_id, chunk_x, chunk_z, create=True)
                assert chunk is not None
                chunk.log_odds[local_z, local_x] = value
                chunk.observed_count[local_z, local_x] = max(
                    int(chunk.observed_count[local_z, local_x]), 1
                )
            return int(cells.shape[0])

        free_count = assign(free, self.log_odds_min)
        occupied_count = assign(occupied, self.log_odds_max)
        return {
            "classified_cells": free_count + occupied_count,
            "free_cells": free_count,
            "occupied_cells": occupied_count,
        }

    def fuse_navigation_raster(
        self,
        *,
        observed_free: np.ndarray,
        observed_obstacle: np.ndarray,
        world_to_map: Sequence[float],
        source_resolution_m: float,
        floor_id: int = 0,
    ) -> dict[str, int]:
        """Inverse-sample a rollout-local LaViRA raster into world cells."""

        free = np.asarray(observed_free, dtype=bool)
        obstacle = np.asarray(observed_obstacle, dtype=bool)
        transform = np.asarray(world_to_map, dtype=np.float64).reshape(-1)
        source_resolution = float(source_resolution_m)
        if free.ndim != 2 or obstacle.shape != free.shape:
            raise ValueError("free and obstacle rasters must share a 2D shape")
        if transform.size != 6 or not np.isfinite(transform).all():
            raise ValueError("world_to_map must contain six finite values")
        if source_resolution <= 0.0 or not math.isfinite(source_resolution):
            raise ValueError("source_resolution_m must be finite and positive")
        height, width = free.shape
        ax, ay, sx, sy, cosine, sine = map(float, transform)
        if not math.isclose(cosine * cosine + sine * sine, 1.0, abs_tol=1e-3):
            raise ValueError("world_to_map rotation is not normalized")

        corner_x = np.asarray([0, width - 1, 0, width - 1], dtype=float)
        corner_y = np.asarray([0, 0, height - 1, height - 1], dtype=float)
        delta_x = corner_x * source_resolution - ax
        delta_y = corner_y * source_resolution - ay
        corner_dsx = cosine * delta_x + sine * delta_y
        corner_dsy = -sine * delta_x + cosine * delta_y
        world_x, world_z = -(corner_dsy + sy), -(corner_dsx + sx)
        resolution = self.spec.resolution_m
        min_x, max_x = math.floor(world_x.min() / resolution) - 1, math.floor(world_x.max() / resolution) + 1
        min_z, max_z = math.floor(world_z.min() / resolution) - 1, math.floor(world_z.max() / resolution) + 1
        grid_x, grid_z = np.meshgrid(
            np.arange(min_x, max_x + 1), np.arange(min_z, max_z + 1)
        )
        sample_world_x = (grid_x + 0.5) * resolution
        sample_world_z = (grid_z + 0.5) * resolution
        dsx, dsy = -sample_world_z - sx, -sample_world_x - sy
        source_x = np.rint((ax + cosine * dsx - sine * dsy) / source_resolution).astype(int)
        source_y = np.rint((ay + sine * dsx + cosine * dsy) / source_resolution).astype(int)
        valid = (source_x >= 0) & (source_x < width) & (source_y >= 0) & (source_y < height)
        sampled_free = np.zeros(valid.shape, dtype=bool)
        sampled_obstacle = np.zeros(valid.shape, dtype=bool)
        sampled_free[valid] = free[source_y[valid], source_x[valid]]
        sampled_obstacle[valid] = obstacle[source_y[valid], source_x[valid]]
        sampled_free &= ~sampled_obstacle
        result = self.set_classified_cells(
            free_cells=np.column_stack((grid_x[sampled_free], grid_z[sampled_free])),
            occupied_cells=np.column_stack((grid_x[sampled_obstacle], grid_z[sampled_obstacle])),
            floor_id=floor_id,
        )
        result.update(
            source_free_cells=int(np.count_nonzero(free)),
            source_obstacle_cells=int(np.count_nonzero(obstacle)),
        )
        return result

    # Compatibility name used by the original runtime adapter.
    fuse_lavira_navigation_raster = fuse_navigation_raster

    @staticmethod
    def _line(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        x0, z0 = map(int, start)
        x1, z1 = map(int, end)
        dx, dz = abs(x1 - x0), abs(z1 - z0)
        step_x, step_z = (1 if x0 < x1 else -1), (1 if z0 < z1 else -1)
        error, result = dx - dz, []
        while True:
            result.append((x0, z0))
            if x0 == x1 and z0 == z1:
                return result
            twice = 2 * error
            if twice > -dz:
                error -= dz
                x0 += step_x
            if twice < dx:
                error += dx
                z0 += step_z

    def integrate_rays(
        self,
        origin_world_xz: Sequence[float],
        endpoint_world_xz: np.ndarray | Iterable[Sequence[float]],
        *,
        endpoint_occupied: bool | Sequence[bool] | np.ndarray = True,
        floor_id: int = 0,
        max_rays: int | None = None,
    ) -> dict[str, int]:
        origin = np.asarray(origin_world_xz, dtype=float).reshape(2)
        endpoints = np.asarray(endpoint_world_xz, dtype=float)
        if endpoints.size == 0:
            return {"rays": 0, "free_cells": 0, "occupied_cells": 0}
        endpoints = endpoints.reshape(-1, 2)
        if not np.isfinite(origin).all() or not np.isfinite(endpoints).all():
            raise ValueError("ray coordinates must be finite")
        occupied = np.asarray(endpoint_occupied, dtype=bool)
        if occupied.ndim == 0:
            occupied = np.full(endpoints.shape[0], bool(occupied), dtype=bool)
        if occupied.reshape(-1).shape[0] != endpoints.shape[0]:
            raise ValueError("endpoint_occupied must match endpoint count")
        by_cell: dict[tuple[int, int], bool] = {}
        for endpoint, is_occupied in zip(endpoints, occupied.reshape(-1)):
            cell = self.spec.world_to_cell(*endpoint)
            by_cell[cell] = by_cell.get(cell, False) or bool(is_occupied)
        items = sorted(by_cell.items())
        if max_rays and len(items) > int(max_rays):
            indices = np.linspace(0, len(items) - 1, int(max_rays), dtype=int)
            items = [items[index] for index in indices]
        origin_cell = self.spec.world_to_cell(*origin)
        free_cells: set[tuple[int, int]] = set()
        occupied_cells: set[tuple[int, int]] = set()
        for endpoint_cell, is_occupied in items:
            line = self._line(origin_cell, endpoint_cell)
            free_cells.update(line[:-1])
            if is_occupied:
                occupied_cells.add(endpoint_cell)
        free_cells -= occupied_cells
        return {
            "rays": len(items),
            "free_cells": self.update_cells(free_cells, self.log_odds_free, floor_id=floor_id),
            "occupied_cells": self.update_cells(occupied_cells, self.log_odds_occupied, floor_id=floor_id),
        }

    def integrate_depth(
        self,
        depth_m: np.ndarray,
        camera_to_world: np.ndarray,
        intrinsics: Sequence[float],
        *,
        floor_height_m: float,
        floor_id: int = 0,
        min_depth_m: float = 0.3,
        max_depth_m: float = 8.0,
        pixel_stride: int = 8,
        max_rays: int = 2048,
        occupied_height_min_m: float = 0.05,
        occupied_height_max_m: float = 1.35,
    ) -> dict[str, int]:
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        pose = np.asarray(camera_to_world, dtype=float)
        if depth.ndim != 2 or pose.shape != (4, 4):
            raise ValueError("depth must be HxW and camera_to_world must be 4x4")
        fx, fy, cx, cy = map(float, intrinsics)
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("focal lengths must be positive")
        stride = max(1, int(pixel_stride))
        rows, cols = np.meshgrid(
            np.arange(0, depth.shape[0], stride),
            np.arange(0, depth.shape[1], stride),
            indexing="ij",
        )
        distances = depth[rows, cols]
        valid = np.isfinite(distances) & (distances >= min_depth_m) & (distances <= max_depth_m)
        if not valid.any():
            return {"samples": 0, "rays": 0, "free_cells": 0, "occupied_cells": 0}
        rows, cols, distances = rows[valid], cols[valid], distances[valid].astype(float)
        camera_points = np.column_stack(
            ((cols - cx) * distances / fx, -(rows - cy) * distances / fy, -distances, np.ones_like(distances))
        )
        world = (pose @ camera_points.T).T[:, :3]
        height = world[:, 1] - float(floor_height_m)
        occupied = (height >= occupied_height_min_m) & (height <= occupied_height_max_m)
        result = self.integrate_rays(
            pose[[0, 2], 3], world[:, [0, 2]], endpoint_occupied=occupied,
            floor_id=floor_id, max_rays=max_rays,
        )
        return {"samples": int(distances.size), **result}

    def cell_evidence(self, grid_x: int, grid_z: int, *, floor_id: int = 0) -> tuple[float, int]:
        chunk_x, chunk_z, local_x, local_z = self.spec.split_cell(grid_x, grid_z)
        chunk = self._chunk(floor_id, chunk_x, chunk_z, create=False)
        return (0.0, 0) if chunk is None else (
            float(chunk.log_odds[local_z, local_x]),
            int(chunk.observed_count[local_z, local_x]),
        )

    def cell_state(self, grid_x: int, grid_z: int, *, floor_id: int = 0) -> str:
        evidence, count = self.cell_evidence(grid_x, grid_z, floor_id=floor_id)
        if count == 0:
            return "unknown"
        if evidence <= self.free_threshold:
            return "free"
        if evidence >= self.occupied_threshold:
            return "occupied"
        return "observed_uncertain"

    def world_state(self, world_x: float, world_z: float, *, floor_id: int = 0) -> str:
        return self.cell_state(*self.spec.world_to_cell(world_x, world_z), floor_id=floor_id)

    def planning_window(
        self, start_cell: Sequence[int], target_cell: Sequence[int], *,
        floor_id: int = 0, padding_cells: int = 40, max_window_cells: int = 4_000_000,
    ) -> WorldPlanningWindow:
        start_x, start_z = map(int, start_cell)
        target_x, target_z = map(int, target_cell)
        padding = max(0, int(padding_cells))
        min_x, max_x = min(start_x, target_x) - padding, max(start_x, target_x) + padding
        min_z, max_z = min(start_z, target_z) - padding, max(start_z, target_z) + padding
        width, height = max_x - min_x + 1, max_z - min_z + 1
        if width * height > int(max_window_cells):
            raise ValueError(f"planning window {width}x{height} is too large")
        log_odds = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.uint16)
        size = self.spec.chunk_size
        for chunk_z in range(min_z // size, max_z // size + 1):
            for chunk_x in range(min_x // size, max_x // size + 1):
                chunk = self._chunk(floor_id, chunk_x, chunk_z, create=False)
                if chunk is None:
                    continue
                gx0, gz0 = max(min_x, chunk_x * size), max(min_z, chunk_z * size)
                gx1, gz1 = min(max_x + 1, (chunk_x + 1) * size), min(max_z + 1, (chunk_z + 1) * size)
                source = np.s_[gz0 - chunk_z * size:gz1 - chunk_z * size, gx0 - chunk_x * size:gx1 - chunk_x * size]
                target = np.s_[gz0 - min_z:gz1 - min_z, gx0 - min_x:gx1 - min_x]
                log_odds[target], counts[target] = chunk.log_odds[source], chunk.observed_count[source]
        observed = counts > 0
        return WorldPlanningWindow(
            min_x, min_z, self.spec.resolution_m,
            observed & (log_odds <= self.free_threshold),
            observed & (log_odds >= self.occupied_threshold), observed,
        )

    def stats(self) -> dict[str, int | float]:
        observed = free = occupied = 0
        for chunk in self._chunks.values():
            known = chunk.observed_count > 0
            observed += int(np.count_nonzero(known))
            free += int(np.count_nonzero(known & (chunk.log_odds <= self.free_threshold)))
            occupied += int(np.count_nonzero(known & (chunk.log_odds >= self.occupied_threshold)))
        return {
            "chunks": self.chunk_count, "observed_cells": observed,
            "free_cells": free, "occupied_cells": occupied,
            "resolution_m": self.spec.resolution_m,
        }

    def save_npz(self, path, *, scene_id: str, source_episode_id: str):
        """Compatibility wrapper around the separated persistence module."""

        from .persistence import save_world_map

        return save_world_map(
            self, path, scene_id=scene_id, source_episode_id=source_episode_id
        )

    @classmethod
    def load_npz(cls, path) -> "SparseWorldMetricMap":
        from .persistence import load_world_map

        return load_world_map(path)
