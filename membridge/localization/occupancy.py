"""Project episode-local RGB-D into a 2D XZ occupancy grid."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from frames import LocalFrame, camera_intrinsics

DEFAULT_RESOLUTION_M = 0.05
DEFAULT_MAX_DEPTH_M = 5.0
DEFAULT_MIN_DEPTH_M = 0.30
OCCUPIED_HEIGHT_MIN_M = 0.05
OCCUPIED_HEIGHT_MAX_M = 1.35
PIXEL_STRIDE = 8
RAY_FREE_SAMPLES = 6


@dataclass
class OccupancyGrid:
    origin_x: float
    origin_z: float
    resolution_m: float
    occupied: np.ndarray
    free: np.ndarray
    observed: np.ndarray

    @property
    def height(self) -> int:
        return int(self.occupied.shape[0])

    @property
    def width(self) -> int:
        return int(self.occupied.shape[1])

    def occupied_points(self) -> np.ndarray:
        rows, cols = np.nonzero(self.occupied)
        if rows.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        x = self.origin_x + (cols.astype(np.float64) + 0.5) * self.resolution_m
        z = self.origin_z + (rows.astype(np.float64) + 0.5) * self.resolution_m
        return np.stack([x, z], axis=1)

    def lookup(self, xz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts = np.asarray(xz, dtype=np.float64).reshape(-1, 2)
        cols = np.floor((pts[:, 0] - self.origin_x) / self.resolution_m).astype(np.int32)
        rows = np.floor((pts[:, 1] - self.origin_z) / self.resolution_m).astype(np.int32)
        inside = (
            (rows >= 0)
            & (rows < self.height)
            & (cols >= 0)
            & (cols < self.width)
        )
        occupied = np.zeros(len(pts), dtype=bool)
        free = np.zeros(len(pts), dtype=bool)
        observed = np.zeros(len(pts), dtype=bool)
        occupied[inside] = self.occupied[rows[inside], cols[inside]]
        free[inside] = self.free[rows[inside], cols[inside]]
        observed[inside] = self.observed[rows[inside], cols[inside]]
        return occupied, free, observed


def _unproject_frame(
    frame: LocalFrame,
    *,
    min_depth_m: float,
    max_depth_m: float,
    pixel_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(frame.depth_m, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    height, width = depth.shape
    fx, fy, cx, cy = camera_intrinsics(width, height)
    stride = max(1, int(pixel_stride))
    pixel_u = np.arange(0, width, stride, dtype=np.int32)
    pixel_v = np.arange(0, height, stride, dtype=np.int32)
    grid_u, grid_v = np.meshgrid(pixel_u, pixel_v)
    grid_u = grid_u.reshape(-1)
    grid_v = grid_v.reshape(-1)
    distances = depth[grid_v, grid_u]
    valid = (
        np.isfinite(distances)
        & (distances >= float(min_depth_m))
        & (distances <= float(max_depth_m))
    )
    if not np.any(valid):
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, empty, empty
    grid_u = grid_u[valid].astype(np.float64)
    grid_v = grid_v[valid].astype(np.float64)
    distances = distances[valid].astype(np.float64)
    camera_x = (grid_u - cx) * distances / fx
    camera_y = (grid_v - cy) * distances / fy
    points_h = np.stack(
        [camera_x, -camera_y, -distances, np.ones_like(distances)],
        axis=1,
    )
    points_world = (frame.camera_to_local @ points_h.T).T[:, :3]
    cam_xz = np.array(
        [frame.camera_to_local[0, 3], frame.camera_to_local[2, 3]],
        dtype=np.float64,
    )
    return cam_xz, points_world, distances


def _accumulate(
    frames: list[LocalFrame],
    *,
    resolution_m: float,
    min_depth_m: float,
    max_depth_m: float,
    pad_m: float,
) -> OccupancyGrid:
    hits = []
    frees = []
    floor_y = float(np.median([frame.y for frame in frames])) if frames else 0.0
    for frame in frames:
        cam_xz, points, distances = _unproject_frame(
            frame,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            pixel_stride=PIXEL_STRIDE,
        )
        if points.shape[0] == 0:
            continue
        relative_height = points[:, 1] - floor_y
        occupied_mask = (
            (relative_height >= OCCUPIED_HEIGHT_MIN_M)
            & (relative_height <= OCCUPIED_HEIGHT_MAX_M)
        )
        hit_xz = points[occupied_mask][:, [0, 2]]
        if hit_xz.size:
            hits.append(hit_xz)
        if RAY_FREE_SAMPLES > 0 and occupied_mask.any():
            occ_pts = points[occupied_mask][:, [0, 2]]
            occ_dist = distances[occupied_mask]
            fractions = np.linspace(0.15, 0.85, RAY_FREE_SAMPLES, dtype=np.float64)
            origin = cam_xz.reshape(1, 2)
            for frac in fractions:
                sample = origin + frac * (occ_pts - origin)
                near = occ_dist * frac > min_depth_m
                if np.any(near):
                    frees.append(sample[near])
    if not hits:
        size = 8
        return OccupancyGrid(
            origin_x=-size * resolution_m / 2.0,
            origin_z=-size * resolution_m / 2.0,
            resolution_m=resolution_m,
            occupied=np.zeros((size, size), dtype=bool),
            free=np.zeros((size, size), dtype=bool),
            observed=np.zeros((size, size), dtype=bool),
        )

    hit_pts = np.concatenate(hits, axis=0)
    free_pts = np.concatenate(frees, axis=0) if frees else np.zeros((0, 2), dtype=np.float64)
    all_pts = hit_pts if free_pts.size == 0 else np.concatenate([hit_pts, free_pts], axis=0)
    min_x = float(all_pts[:, 0].min()) - pad_m
    max_x = float(all_pts[:, 0].max()) + pad_m
    min_z = float(all_pts[:, 1].min()) - pad_m
    max_z = float(all_pts[:, 1].max()) + pad_m
    width = int(np.ceil((max_x - min_x) / resolution_m)) + 1
    height = int(np.ceil((max_z - min_z) / resolution_m)) + 1
    occupied = np.zeros((height, width), dtype=np.int32)
    free = np.zeros((height, width), dtype=np.int32)

    def splat(points: np.ndarray, canvas: np.ndarray) -> None:
        if points.size == 0:
            return
        cols = np.floor((points[:, 0] - min_x) / resolution_m).astype(np.int32)
        rows = np.floor((points[:, 1] - min_z) / resolution_m).astype(np.int32)
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        np.add.at(canvas, (rows[inside], cols[inside]), 1)

    splat(hit_pts, occupied)
    splat(free_pts, free)
    occ_bool = occupied > 0
    free_bool = (free > 0) & ~occ_bool
    observed = occ_bool | free_bool
    return OccupancyGrid(
        origin_x=min_x,
        origin_z=min_z,
        resolution_m=float(resolution_m),
        occupied=occ_bool,
        free=free_bool,
        observed=observed,
    )


def build_map_occupancy(frames: list[LocalFrame], resolution_m: float = DEFAULT_RESOLUTION_M) -> OccupancyGrid:
    return _accumulate(
        frames,
        resolution_m=resolution_m,
        min_depth_m=DEFAULT_MIN_DEPTH_M,
        max_depth_m=DEFAULT_MAX_DEPTH_M,
        pad_m=2.0,
    )


def build_query_occupancy(frames: list[LocalFrame], resolution_m: float = DEFAULT_RESOLUTION_M) -> OccupancyGrid:
    return _accumulate(
        frames,
        resolution_m=resolution_m,
        min_depth_m=DEFAULT_MIN_DEPTH_M,
        max_depth_m=DEFAULT_MAX_DEPTH_M,
        pad_m=0.5,
    )


class IncrementalOccupancy:
    """Add one RGB-D frame at a time without reprojecting the whole history."""

    def __init__(self, resolution_m: float = DEFAULT_RESOLUTION_M, half_extent_m: float = 16.0):
        self.resolution_m = float(resolution_m)
        size = int(math.ceil(2.0 * half_extent_m / self.resolution_m)) + 1
        self.origin_x = -float(half_extent_m)
        self.origin_z = -float(half_extent_m)
        self.occ_count = np.zeros((size, size), dtype=np.int32)
        self.free_count = np.zeros((size, size), dtype=np.int32)
        self.floor_ys: list[float] = []
        self.n_frames = 0

    def add_frame(self, frame: LocalFrame) -> None:
        self.floor_ys.append(float(frame.y))
        floor_y = float(np.median(self.floor_ys))
        cam_xz, points, distances = _unproject_frame(
            frame,
            min_depth_m=DEFAULT_MIN_DEPTH_M,
            max_depth_m=DEFAULT_MAX_DEPTH_M,
            pixel_stride=PIXEL_STRIDE,
        )
        if points.shape[0] == 0:
            self.n_frames += 1
            return
        relative_height = points[:, 1] - floor_y
        occupied_mask = (relative_height >= OCCUPIED_HEIGHT_MIN_M) & (
            relative_height <= OCCUPIED_HEIGHT_MAX_M
        )
        hits = points[occupied_mask][:, [0, 2]]
        frees = np.zeros((0, 2), dtype=np.float64)
        if occupied_mask.any():
            occ_pts = hits
            occ_dist = distances[occupied_mask]
            origin = cam_xz.reshape(1, 2)
            chunks = []
            for frac in np.linspace(0.15, 0.85, RAY_FREE_SAMPLES, dtype=np.float64):
                sample = origin + frac * (occ_pts - origin)
                near = occ_dist * frac > DEFAULT_MIN_DEPTH_M
                if np.any(near):
                    chunks.append(sample[near])
            if chunks:
                frees = np.concatenate(chunks, axis=0)
        all_pts = hits if frees.size == 0 else np.concatenate([hits, frees], axis=0)
        if all_pts.size:
            self._ensure_bounds(all_pts)
        self._splat(hits, self.occ_count)
        self._splat(frees, self.free_count)
        self.n_frames += 1

    def _ensure_bounds(self, points: np.ndarray) -> None:
        min_x, min_z = float(points[:, 0].min()), float(points[:, 1].min())
        max_x, max_z = float(points[:, 0].max()), float(points[:, 1].max())
        pad = 1.0
        need = (
            min_x < self.origin_x + pad
            or min_z < self.origin_z + pad
            or max_x > self.origin_x + self.occ_count.shape[1] * self.resolution_m - pad
            or max_z > self.origin_z + self.occ_count.shape[0] * self.resolution_m - pad
        )
        if not need:
            return
        new_ox = min(self.origin_x, min_x - 2.0)
        new_oz = min(self.origin_z, min_z - 2.0)
        new_max_x = max(
            self.origin_x + self.occ_count.shape[1] * self.resolution_m, max_x + 2.0
        )
        new_max_z = max(
            self.origin_z + self.occ_count.shape[0] * self.resolution_m, max_z + 2.0
        )
        width = int(np.ceil((new_max_x - new_ox) / self.resolution_m)) + 1
        height = int(np.ceil((new_max_z - new_oz) / self.resolution_m)) + 1
        occ = np.zeros((height, width), dtype=np.int32)
        free = np.zeros((height, width), dtype=np.int32)
        row0 = int(round((self.origin_z - new_oz) / self.resolution_m))
        col0 = int(round((self.origin_x - new_ox) / self.resolution_m))
        old_h, old_w = self.occ_count.shape
        occ[row0 : row0 + old_h, col0 : col0 + old_w] = self.occ_count
        free[row0 : row0 + old_h, col0 : col0 + old_w] = self.free_count
        self.occ_count, self.free_count = occ, free
        self.origin_x, self.origin_z = new_ox, new_oz

    def _splat(self, points: np.ndarray, canvas: np.ndarray) -> None:
        if points.size == 0:
            return
        cols = np.floor((points[:, 0] - self.origin_x) / self.resolution_m).astype(np.int32)
        rows = np.floor((points[:, 1] - self.origin_z) / self.resolution_m).astype(np.int32)
        inside = (rows >= 0) & (rows < canvas.shape[0]) & (cols >= 0) & (cols < canvas.shape[1])
        np.add.at(canvas, (rows[inside], cols[inside]), 1)

    def mark_blocked(self, x: float, z: float) -> None:
        self._ensure_bounds(np.array([[x, z]], dtype=np.float64))
        self._splat(np.array([[x, z]], dtype=np.float64), self.occ_count)

    def grid(self) -> OccupancyGrid:
        occ = self.occ_count > 0
        free = (self.free_count > 0) & ~occ
        return OccupancyGrid(
            origin_x=self.origin_x,
            origin_z=self.origin_z,
            resolution_m=self.resolution_m,
            occupied=occ,
            free=free,
            observed=occ | free,
        )
