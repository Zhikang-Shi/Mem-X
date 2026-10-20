"""Feasible occupancy frontiers in the current episode-local grid."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from occupancy import OccupancyGrid

MIN_COMPONENT = 2
MAX_FRONTIERS = 16


@dataclass
class Frontier:
    x: float
    z: float
    unknown_x: float
    unknown_z: float
    n_cells: int
    info: float
    risk: float
    path_m: float
    reachable: bool


def _cell(grid: OccupancyGrid, x: float, z: float) -> tuple[int, int]:
    col = int(np.floor((x - grid.origin_x) / grid.resolution_m))
    row = int(np.floor((z - grid.origin_z) / grid.resolution_m))
    return row, col


def _xz(grid: OccupancyGrid, row: int, col: int) -> tuple[float, float]:
    x = grid.origin_x + (col + 0.5) * grid.resolution_m
    z = grid.origin_z + (row + 0.5) * grid.resolution_m
    return x, z


def extract_frontiers(
    grid: OccupancyGrid,
    agent_xz: tuple[float, float],
) -> list[Frontier]:
    free = grid.free
    occ = grid.occupied
    unknown = ~grid.observed
    if free.shape[0] < 3 or free.shape[1] < 3:
        return []
    unknown_n = (
        unknown[:-2, 1:-1] | unknown[2:, 1:-1] | unknown[1:-1, :-2] | unknown[1:-1, 2:]
    )
    occ_n = occ[:-2, 1:-1] | occ[2:, 1:-1] | occ[1:-1, :-2] | occ[1:-1, 2:]
    core = np.zeros_like(free)
    core[1:-1, 1:-1] = free[1:-1, 1:-1] & unknown_n & ~occ_n
    labels, n_labels = _label_4(core)
    if n_labels == 0:
        return []
    agent_row, agent_col = _cell(grid, agent_xz[0], agent_xz[1])
    dist = _bfs_distance(free, agent_row, agent_col)
    unk_n = unknown[:-2, 1:-1]
    unk_s = unknown[2:, 1:-1]
    unk_w = unknown[1:-1, :-2]
    unk_e = unknown[1:-1, 2:]
    occ_n4 = occ[:-2, 1:-1]
    occ_s = occ[2:, 1:-1]
    occ_w = occ[1:-1, :-2]
    occ_e = occ[1:-1, 2:]
    clustered: list[Frontier] = []
    for label in range(1, n_labels + 1):
        cells = np.argwhere(labels == label)
        if cells.shape[0] < MIN_COMPONENT:
            continue
        rows, cols = cells[:, 0], cells[:, 1]
        path_vals = dist[rows, cols]
        path_vals = path_vals[path_vals < 10_000]
        if path_vals.size == 0:
            continue
        cx = float(grid.origin_x + (cols.astype(np.float64) + 0.5).mean() * grid.resolution_m)
        cz = float(grid.origin_z + (rows.astype(np.float64) + 0.5).mean() * grid.resolution_m)
        inner_r = np.clip(rows - 1, 0, unk_n.shape[0] - 1)
        inner_c = np.clip(cols - 1, 0, unk_n.shape[1] - 1)
        info = int(
            unk_n[inner_r, inner_c].sum()
            + unk_s[inner_r, inner_c].sum()
            + unk_w[inner_r, inner_c].sum()
            + unk_e[inner_r, inner_c].sum()
        )
        risk = float(
            (
                occ_n4[inner_r, inner_c].sum()
                + occ_s[inner_r, inner_c].sum()
                + occ_w[inner_r, inner_c].sum()
                + occ_e[inner_r, inner_c].sum()
            )
            / max(cells.shape[0], 1)
        )
        if info <= 0:
            continue
        ux = cx
        uz = cz + grid.resolution_m
        step = 0.15
        norm = math.hypot(ux - cx, uz - cz) + 1e-6
        # unknown centroid via neighbor offsets
        ys = []
        if np.any(unk_n[inner_r, inner_c]):
            ys.append((-1, 0))
        if np.any(unk_s[inner_r, inner_c]):
            ys.append((1, 0))
        if np.any(unk_w[inner_r, inner_c]):
            ys.append((0, -1))
        if np.any(unk_e[inner_r, inner_c]):
            ys.append((0, 1))
        if ys:
            dr = float(np.mean([p[0] for p in ys]))
            dc = float(np.mean([p[1] for p in ys]))
            ux = cx + dc * grid.resolution_m
            uz = cz + dr * grid.resolution_m
        norm = math.hypot(ux - cx, uz - cz) + 1e-6
        wx = cx + step * (ux - cx) / norm
        wz = cz + step * (uz - cz) / norm
        clustered.append(
            Frontier(
                x=wx,
                z=wz,
                unknown_x=ux,
                unknown_z=uz,
                n_cells=int(cells.shape[0]),
                info=float(info),
                risk=risk,
                path_m=float(int(path_vals.min()) * grid.resolution_m),
                reachable=True,
            )
        )
    clustered.sort(key=lambda item: item.info, reverse=True)
    return clustered[:MAX_FRONTIERS]


def _label_4(mask: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        from scipy.ndimage import generate_binary_structure, label

        structure = generate_binary_structure(2, 1)
        labels, count = label(mask, structure=structure)
        return labels.astype(np.int32), int(count)
    except Exception:
        return _label_4_python(mask)


def _label_4_python(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    rows, cols = np.nonzero(mask)
    for row, col in zip(rows.tolist(), cols.tolist()):
        if labels[row, col]:
            continue
        current += 1
        stack = [(row, col)]
        labels[row, col] = current
        while stack:
            cr, cc = stack.pop()
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and labels[nr, nc] == 0:
                    labels[nr, nc] = current
                    stack.append((nr, nc))
    return labels, current


def _bfs_distance(walkable: np.ndarray, start_row: int, start_col: int) -> np.ndarray:
    height, width = walkable.shape
    dist = np.full((height, width), 10_000, dtype=np.int32)
    if height == 0 or width == 0:
        return dist
    sr, sc = start_row, start_col
    if not (0 <= sr < height and 0 <= sc < width) or not walkable[sr, sc]:
        free_idx = np.argwhere(walkable)
        if free_idx.size == 0:
            return dist
        d = np.abs(free_idx[:, 0] - sr) + np.abs(free_idx[:, 1] - sc)
        sr, sc = (int(free_idx[int(d.argmin()), 0]), int(free_idx[int(d.argmin()), 1]))
    dist[sr, sc] = 0
    queue = [(sr, sc)]
    head = 0
    while head < len(queue):
        row, col = queue[head]
        head += 1
        nxt = dist[row, col] + 1
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= nr < height and 0 <= nc < width and walkable[nr, nc] and dist[nr, nc] > nxt:
                dist[nr, nc] = nxt
                queue.append((nr, nc))
    return dist
