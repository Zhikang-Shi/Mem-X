"""Coordinate and finite-window primitives for occupancy planning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def dilate_disk(mask: np.ndarray, distance: float) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    radius = int(math.ceil(max(0.0, float(distance))))
    if radius == 0:
        return source.copy()
    result = np.zeros_like(source)
    rows, cols = np.nonzero(source)
    for dz in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dz * dz > float(distance) ** 2:
                continue
            shifted_cols = cols + dx
            shifted_rows = rows + dz
            valid = (
                (shifted_cols >= 0)
                & (shifted_cols < source.shape[1])
                & (shifted_rows >= 0)
                & (shifted_rows < source.shape[0])
            )
            result[shifted_rows[valid], shifted_cols[valid]] = True
    return result


def erode_disk(mask: np.ndarray, distance: float) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    radius = int(math.ceil(max(0.0, float(distance))))
    if radius == 0:
        return source.copy()
    result = source.copy()
    padded = np.pad(source, radius, mode="constant", constant_values=False)
    for dz in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dz * dz > float(distance) ** 2:
                continue
            row = radius + dz
            col = radius + dx
            result &= padded[row : row + source.shape[0], col : col + source.shape[1]]
    return result


@dataclass(frozen=True)
class WorldGridSpec:
    resolution_m: float = 0.05
    chunk_size: int = 128

    def __post_init__(self) -> None:
        if not math.isfinite(self.resolution_m) or self.resolution_m <= 0.0:
            raise ValueError("resolution_m must be finite and positive")
        if int(self.chunk_size) != self.chunk_size or self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

    def world_to_cell(self, world_x: float, world_z: float) -> tuple[int, int]:
        return (
            math.floor(float(world_x) / self.resolution_m),
            math.floor(float(world_z) / self.resolution_m),
        )

    def cell_center_world(self, grid_x: int, grid_z: int) -> tuple[float, float]:
        return (
            (int(grid_x) + 0.5) * self.resolution_m,
            (int(grid_z) + 0.5) * self.resolution_m,
        )

    def split_cell(self, grid_x: int, grid_z: int) -> tuple[int, int, int, int]:
        chunk_x = int(grid_x) // self.chunk_size
        chunk_z = int(grid_z) // self.chunk_size
        return (
            chunk_x,
            chunk_z,
            int(grid_x) - chunk_x * self.chunk_size,
            int(grid_z) - chunk_z * self.chunk_size,
        )


@dataclass(frozen=True)
class WorldPlanningWindow:
    origin_grid_x: int
    origin_grid_z: int
    resolution_m: float
    free: np.ndarray
    occupied: np.ndarray
    observed: np.ndarray

    def global_to_local(self, grid_x: int, grid_z: int) -> tuple[int, int]:
        return int(grid_x) - self.origin_grid_x, int(grid_z) - self.origin_grid_z

    def local_to_global(self, local_x: int, local_z: int) -> tuple[int, int]:
        return self.origin_grid_x + int(local_x), self.origin_grid_z + int(local_z)

    def safe_mask(
        self,
        clearance_cells: float = 0.0,
        *,
        certified_free: np.ndarray | None = None,
        certified_centers: np.ndarray | None = None,
        certified_center_slack_cells: float = 0.0,
        certified_slack_obstacle_clearance_cells: float | None = None,
    ) -> np.ndarray:
        free = np.asarray(self.free, dtype=bool).copy()
        certified = None
        if certified_free is not None:
            certified = np.asarray(certified_free, dtype=bool)
            if certified.shape != free.shape:
                raise ValueError("certified_free shape mismatch")
            free |= certified
        safe = erode_disk(free, clearance_cells)
        slack = max(0.0, float(certified_center_slack_cells))
        if certified is None or slack <= 0.0:
            return safe
        centers = (
            erode_disk(certified, clearance_cells)
            if certified_centers is None
            else np.asarray(certified_centers, dtype=bool)
        )
        if centers.shape != free.shape:
            raise ValueError("certified_centers shape mismatch")
        obstacle_clearance = (
            float(clearance_cells)
            if certified_slack_obstacle_clearance_cells is None
            else max(0.0, float(certified_slack_obstacle_clearance_cells))
        )
        obstacle_safe = erode_disk(~np.asarray(self.occupied, dtype=bool), obstacle_clearance)
        safe |= dilate_disk(centers, slack) & obstacle_safe
        return safe
