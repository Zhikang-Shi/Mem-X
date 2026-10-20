"""A* on the agent's own occupancy. Unknown is walkable; occupied is not."""

from __future__ import annotations

import heapq

import numpy as np

from occupancy import OccupancyGrid


def _cell(grid: OccupancyGrid, x: float, z: float) -> tuple[int, int]:
    col = int(np.floor((x - grid.origin_x) / grid.resolution_m))
    row = int(np.floor((z - grid.origin_z) / grid.resolution_m))
    return row, col


def _xz(grid: OccupancyGrid, row: int, col: int) -> tuple[float, float]:
    return (
        grid.origin_x + (col + 0.5) * grid.resolution_m,
        grid.origin_z + (row + 0.5) * grid.resolution_m,
    )


def shortest_path_xz(
    grid: OccupancyGrid,
    start_xz: tuple[float, float],
    goal_xz: tuple[float, float],
    *,
    unknown_cost: int = 8,
) -> list[tuple[float, float]]:
    occupied = grid.occupied
    free = grid.free
    height, width = occupied.shape
    if height == 0 or width == 0:
        return []
    sr, sc = _cell(grid, start_xz[0], start_xz[1])
    gr, gc = _cell(grid, goal_xz[0], goal_xz[1])

    def step_cost(row: int, col: int) -> int | None:
        if occupied[row, col]:
            return None
        if free[row, col]:
            return 1
        return int(unknown_cost)

    def snap(row: int, col: int, prefer_free: bool) -> tuple[int, int] | None:
        if 0 <= row < height and 0 <= col < width and step_cost(row, col) is not None:
            if (not prefer_free) or free[row, col]:
                return row, col
        mask = free if prefer_free else ~occupied
        idx = np.argwhere(mask)
        if idx.size == 0:
            return None
        d = np.abs(idx[:, 0] - row) + np.abs(idx[:, 1] - col)
        i = int(d.argmin())
        return int(idx[i, 0]), int(idx[i, 1])

    start = snap(sr, sc, prefer_free=True) or snap(sr, sc, prefer_free=False)
    goal = snap(gr, gc, prefer_free=True) or snap(gr, gc, prefer_free=False)
    if start is None or goal is None:
        return []
    sr, sc = start
    gr, gc = goal
    inf = 10**9
    dist = np.full((height, width), inf, dtype=np.int32)
    parent = np.full((height, width, 2), -1, dtype=np.int16)
    dist[sr, sc] = 0
    heap = [(0 + abs(sr - gr) + abs(sc - gc), 0, sr, sc)]
    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    found = False
    while heap:
        _, cost, row, col = heapq.heappop(heap)
        if cost != dist[row, col]:
            continue
        if row == gr and col == gc:
            found = True
            break
        for dr, dc in dirs:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            step = step_cost(nr, nc)
            if step is None:
                continue
            ncost = cost + step
            if ncost < dist[nr, nc]:
                dist[nr, nc] = ncost
                parent[nr, nc] = (row, col)
                heapq.heappush(heap, (ncost + abs(nr - gr) + abs(nc - gc), ncost, nr, nc))
    if not found:
        return []
    path = []
    row, col = gr, gc
    while not (row == sr and col == sc):
        path.append(_xz(grid, row, col))
        pr, pc = int(parent[row, col, 0]), int(parent[row, col, 1])
        if pr < 0:
            break
        row, col = pr, pc
    path.append(_xz(grid, sr, sc))
    path.reverse()
    return path
