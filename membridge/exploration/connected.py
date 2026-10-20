"""Small dependency-free connected-component and frontier helpers."""

from __future__ import annotations

from collections import deque

import numpy as np


def neighbor_union(mask: np.ndarray, *, outside_value: bool) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    padded = np.pad(source, 1, constant_values=outside_value)
    result = np.zeros_like(source)
    for dz in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dz:
                result |= padded[1 + dz:1 + dz + source.shape[0], 1 + dx:1 + dx + source.shape[1]]
    return result


def connected_components(mask: np.ndarray, *, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    source = np.asarray(mask, dtype=bool)
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if connectivity == 8:
        neighbors += ((1, 1), (1, -1), (-1, 1), (-1, -1))
    unseen = {(int(x), int(z)) for z, x in np.argwhere(source)}
    components = []
    while unseen:
        start = min(unseen, key=lambda point: (point[1], point[0]))
        unseen.remove(start)
        queue, component = deque([start]), []
        while queue:
            point = queue.popleft()
            component.append(point)
            for dx, dz in neighbors:
                neighbor = point[0] + dx, point[1] + dz
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def reachable_component(mask: np.ndarray, start_xy: tuple[int, int]) -> np.ndarray:
    result = np.zeros(np.asarray(mask).shape, dtype=bool)
    start = tuple(map(int, start_xy))
    for component in connected_components(mask):
        if start in component:
            for x, z in component:
                result[z, x] = True
            break
    return result


def bounded_start_connector(
    ordinary_safe: np.ndarray, connector_safe: np.ndarray,
    start_xy: tuple[int, int], maximum_steps: int,
) -> list[tuple[int, int]]:
    ordinary, connector = np.asarray(ordinary_safe, bool), np.asarray(connector_safe, bool)
    start = tuple(map(int, start_xy))
    if ordinary.shape != connector.shape or not (0 <= start[0] < ordinary.shape[1] and 0 <= start[1] < ordinary.shape[0] and connector[start[1], start[0]]):
        return []
    initial = reachable_component(ordinary, start)
    queue, parent, depth = deque([start]), {start: None}, {start: 0}
    endpoint = None
    while queue:
        current = queue.popleft()
        if current != start and ordinary[current[1], current[0]] and not initial[current[1], current[0]]:
            endpoint = current
            break
        if depth[current] >= int(maximum_steps):
            continue
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = current[0] + dx, current[1] + dz
            x, z = neighbor
            if 0 <= x < ordinary.shape[1] and 0 <= z < ordinary.shape[0] and connector[z, x] and neighbor not in parent:
                parent[neighbor], depth[neighbor] = current, depth[current] + 1
                queue.append(neighbor)
    if endpoint is None:
        return []
    path = []
    while endpoint is not None:
        path.append(endpoint)
        endpoint = parent[endpoint]
    return list(reversed(path))
