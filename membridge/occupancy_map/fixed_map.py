"""Fixed-size log-odds map retained for semantic room segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .grid import dilate_disk


@dataclass(frozen=True)
class FixedMapConfig:
    resolution_m: float = 0.05
    width: int = 2000
    height: int = 2000
    min_x: float = -50.0
    min_z: float = -50.0
    log_odds_occupied: float = 0.85
    log_odds_free: float = -0.25
    log_odds_min: float = -4.0
    log_odds_max: float = 4.0
    occupied_threshold: float = 0.85
    free_threshold: float = -0.40
    minimum_range_m: float = 0.3
    maximum_range_m: float = 8.0
    pixel_stride: int = 6
    ray_stride: int = 3
    update_occupied_endpoints: bool = False
    robot_radius_m: float = 0.30
    inflation_m: float = 0.10


def _line(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    x0, z0 = start
    x1, z1 = end
    dx, dz = abs(x1 - x0), abs(z1 - z0)
    sx, sz = (1 if x0 < x1 else -1), (1 if z0 < z1 else -1)
    error = dx - dz
    cells = []
    while True:
        cells.append((x0, z0))
        if (x0, z0) == (x1, z1):
            return cells
        twice = 2 * error
        if twice > -dz:
            error -= dz
            x0 += sx
        if twice < dx:
            error += dx
            z0 += sz


class FixedSemanticMap:
    """Occupancy/free/door/room layers used by semantic mapping only."""

    def __init__(self, config: FixedMapConfig | None = None) -> None:
        self.config = config or FixedMapConfig()
        shape = (self.config.height, self.config.width)
        self.log_odds = np.zeros(shape, dtype=np.float32)
        self.door_hits = np.zeros(shape, dtype=np.float32)
        self.door = np.zeros(shape, dtype=bool)
        self.room_ids = np.full(shape, -1, dtype=np.int32)

    def world_to_grid(self, x: float, z: float) -> tuple[int, int]:
        return (
            int((float(x) - self.config.min_x) / self.config.resolution_m),
            int((float(z) - self.config.min_z) / self.config.resolution_m),
        )

    def grid_to_world(self, x: int, z: int) -> tuple[float, float]:
        return (
            int(x) * self.config.resolution_m + self.config.min_x,
            int(z) * self.config.resolution_m + self.config.min_z,
        )

    def _valid(self, x: int, z: int) -> bool:
        return 0 <= x < self.config.width and 0 <= z < self.config.height

    def _update(self, cells, delta: float) -> int:
        unique = {(int(x), int(z)) for x, z in cells if self._valid(int(x), int(z))}
        if not unique:
            return 0
        cols, rows = zip(*unique)
        values = self.log_odds[rows, cols] + float(delta)
        self.log_odds[rows, cols] = np.clip(
            values, self.config.log_odds_min, self.config.log_odds_max
        )
        return len(unique)

    @property
    def occupied(self) -> np.ndarray:
        return self.log_odds > self.config.occupied_threshold

    @property
    def free(self) -> np.ndarray:
        return self.log_odds < self.config.free_threshold

    def integrate_depth(
        self,
        depth: np.ndarray,
        camera_to_world: np.ndarray,
        intrinsics: Mapping[str, float],
    ) -> dict[str, int]:
        image = np.asarray(depth)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        pose = np.asarray(camera_to_world, dtype=np.float64).reshape(4, 4)
        origin = self.world_to_grid(float(pose[0, 3]), float(pose[2, 3]))
        stride = max(1, int(self.config.pixel_stride))
        rows, cols = np.meshgrid(
            np.arange(0, image.shape[0], stride),
            np.arange(0, image.shape[1], stride),
            indexing="ij",
        )
        raw = image[rows, cols].astype(np.float64)
        scale = float(intrinsics.get("depth_scale", 1000.0))
        distance = raw / scale
        valid = (
            np.isfinite(distance)
            & (distance > self.config.minimum_range_m)
            & (distance < self.config.maximum_range_m)
        )
        u, v, d = cols[valid], rows[valid], distance[valid]
        camera = np.stack(
            [
                (u - float(intrinsics["cx"])) * d / float(intrinsics["fx"]),
                -(v - float(intrinsics["cy"])) * d / float(intrinsics["fy"]),
                -d,
                np.ones_like(d),
            ],
            axis=1,
        )
        world = (pose @ camera.T).T[:, :3]
        free_cells: set[tuple[int, int]] = set()
        occupied_cells: set[tuple[int, int]] = set()
        for point in world[:: max(1, int(self.config.ray_stride))]:
            endpoint = self.world_to_grid(float(point[0]), float(point[2]))
            if not self._valid(*origin) or not self._valid(*endpoint):
                continue
            ray = _line(origin, endpoint)
            free_cells.update(ray[:-1])
            if self.config.update_occupied_endpoints:
                occupied_cells.add(endpoint)
        free_cells -= occupied_cells
        return {
            "rays": int(len(world)),
            "free_cells": self._update(free_cells, self.config.log_odds_free),
            "occupied_cells": self._update(
                occupied_cells, self.config.log_odds_occupied
            ),
        }

    def integrate_structure(self, points_xyz: np.ndarray, kind: str) -> int:
        points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        cells = [self.world_to_grid(point[0], point[2]) for point in points]
        if str(kind).lower() in {"door", "doorway", "complete door frame"}:
            valid = [(x, z) for x, z in cells if self._valid(x, z)]
            if valid:
                cols, rows = zip(*valid)
                self.door_hits *= 0.98
                self.door_hits[rows, cols] += 1.0
                np.clip(self.door_hits, 0.0, 20.0, out=self.door_hits)
                self.door = self.door_hits >= 6.0
            return len(valid)
        return self._update(cells, self.config.log_odds_occupied)

    def inflated_occupied(self) -> np.ndarray:
        radius = (self.config.robot_radius_m + self.config.inflation_m) / self.config.resolution_m
        return dilate_disk(self.occupied, radius)

    def snapshot(self) -> dict[str, np.ndarray | float]:
        return {
            "resolution": self.config.resolution_m,
            "min_x": self.config.min_x,
            "min_z": self.config.min_z,
            "room_mask": self.room_ids.copy(),
            "occ_bin": self.occupied.astype(np.uint8),
            "free_bin": self.free.astype(np.uint8),
            "door_map": self.door.astype(np.uint8),
        }

    def save_npz(self, path):
        from .persistence import save_fixed_map

        return save_fixed_map(self, path)

    @classmethod
    def load_npz(cls, path) -> "FixedSemanticMap":
        from .persistence import load_fixed_map

        return load_fixed_map(path)
