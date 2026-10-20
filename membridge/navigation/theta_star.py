"""Any-angle Theta* over a certified finite occupancy mask."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def grid_line_cells(start: Sequence[int], end: Sequence[int]) -> list[tuple[int, int]]:
    x0, y0 = map(int, start)
    x1, y1 = map(int, end)
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy, error, cells = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1), dx - dy, []
    while True:
        cells.append((x0, y0))
        if (x0, y0) == (x1, y1):
            return cells
        twice = 2 * error
        if twice > -dy:
            error -= dy
            x0 += sx
        if twice < dx:
            error += dx
            y0 += sy


def grid_line_is_safe(mask: np.ndarray, start: Sequence[int], end: Sequence[int]) -> bool:
    safe = np.asarray(mask, dtype=bool)
    previous = None
    for x, y in grid_line_cells(start, end):
        if not (0 <= y < safe.shape[0] and 0 <= x < safe.shape[1] and safe[y, x]):
            return False
        if previous and x != previous[0] and y != previous[1]:
            if not safe[previous[1], x] or not safe[y, previous[0]]:
                return False
        previous = (x, y)
    return True


@dataclass(frozen=True)
class ThetaStarResult:
    path: list[tuple[int, int]] | None
    expanded_states: int
    reason: str


def theta_star(
    safe_mask: np.ndarray, start: Sequence[int], goal: Sequence[int], *, max_states: int = 250_000
) -> ThetaStarResult:
    safe = np.asarray(safe_mask, dtype=bool)
    start, goal = tuple(map(int, start)), tuple(map(int, goal))
    for name, point in (("start", start), ("goal", goal)):
        if not (0 <= point[0] < safe.shape[1] and 0 <= point[1] < safe.shape[0] and safe[point[1], point[0]]):
            return ThetaStarResult(None, 0, f"{name}_not_safe")
    if start == goal:
        return ThetaStarResult([start], 0, "planned")
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    cost, parent, closed = {start: 0.0}, {start: start}, set()
    queue = [(math.dist(start, goal), 0.0, start[1], start[0])]
    expanded = 0
    while queue and expanded < int(max_states):
        _, queued_cost, y, x = heapq.heappop(queue)
        current = (x, y)
        if current in closed or queued_cost != cost.get(current):
            continue
        closed.add(current)
        expanded += 1
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return ThetaStarResult(list(reversed(path)), expanded, "planned")
        for dx, dy in neighbors:
            neighbor = x + dx, y + dy
            if not (0 <= neighbor[0] < safe.shape[1] and 0 <= neighbor[1] < safe.shape[0] and safe[neighbor[1], neighbor[0]]):
                continue
            if dx and dy and (not safe[y, neighbor[0]] or not safe[neighbor[1], x]):
                continue
            anchor = parent[current]
            candidate_parent = anchor if anchor != current and grid_line_is_safe(safe, anchor, neighbor) else current
            candidate_cost = cost[candidate_parent] + math.dist(candidate_parent, neighbor)
            if candidate_cost + 1e-9 < cost.get(neighbor, float("inf")):
                cost[neighbor], parent[neighbor] = candidate_cost, candidate_parent
                heapq.heappush(queue, (candidate_cost + math.dist(neighbor, goal), candidate_cost, neighbor[1], neighbor[0]))
    return ThetaStarResult(None, expanded, "max_states" if expanded >= max_states else "unreachable")
