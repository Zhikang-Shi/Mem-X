"""World-frame topology extracted from the path actually traversed."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class TrajectoryNode:
    node_id: int
    position_xyz: tuple[float, float, float]
    heading_rad: float
    first_frame_idx: int
    last_frame_idx: int
    step: int
    floor_id: int
    episode_id: str


class TrajectoryGraph:
    def __init__(
        self, *, min_node_translation_m: float = 0.25, max_edge_length_m: float = 1.0
    ) -> None:
        if min_node_translation_m <= 0.0:
            raise ValueError("min_node_translation_m must be positive")
        if max_edge_length_m < min_node_translation_m:
            raise ValueError("max_edge_length_m must cover node spacing")
        self.min_node_translation_m = float(min_node_translation_m)
        self.max_edge_length_m = float(max_edge_length_m)
        self.nodes: list[TrajectoryNode] = []
        self.edges: list[tuple[int, int, float]] = []
        self.frame_to_node: dict[int, int] = {}

    def _add_node(
        self, position: np.ndarray, heading_rad: float, frame_idx: int,
        step: int, floor_id: int, episode_id: str,
    ) -> int:
        node_id = len(self.nodes)
        self.nodes.append(
            TrajectoryNode(
                node_id, tuple(map(float, position)), float(heading_rad),
                int(frame_idx), int(frame_idx), int(step), int(floor_id), str(episode_id),
            )
        )
        self.frame_to_node[int(frame_idx)] = node_id
        return node_id

    def observe(
        self, position_xyz: Sequence[float], heading_rad: float, *,
        frame_idx: int, step: int, floor_id: int = 0, episode_id: str = "",
    ) -> int:
        position = np.asarray(position_xyz, dtype=float).reshape(3)
        if not np.isfinite(position).all() or not math.isfinite(float(heading_rad)):
            raise ValueError("trajectory observations must be finite")
        if not self.nodes:
            return self._add_node(position, heading_rad, frame_idx, step, floor_id, episode_id)
        previous = self.nodes[-1]
        same_segment = previous.floor_id == int(floor_id) and previous.episode_id == str(episode_id)
        distance = float(np.linalg.norm(position[[0, 2]] - np.asarray(previous.position_xyz)[[0, 2]]))
        if same_segment and distance < self.min_node_translation_m:
            previous.last_frame_idx = int(frame_idx)
            self.frame_to_node[int(frame_idx)] = previous.node_id
            return previous.node_id
        node_id = self._add_node(position, heading_rad, frame_idx, step, floor_id, episode_id)
        if same_segment and distance <= self.max_edge_length_m:
            self.edges.append((previous.node_id, node_id, distance))
        return node_id

    def node_for_frame(self, frame_idx: int) -> int | None:
        return self.frame_to_node.get(int(frame_idx))

    def nearest_node(
        self, position_xyz: Sequence[float], *, floor_id: int = 0,
        max_distance_m: float | None = None,
    ) -> tuple[int, float] | None:
        query = np.asarray(position_xyz, dtype=float).reshape(3)
        candidates = [
            (node.node_id, float(np.linalg.norm(query[[0, 2]] - np.asarray(node.position_xyz)[[0, 2]])))
            for node in self.nodes if node.floor_id == int(floor_id)
        ]
        if not candidates:
            return None
        best = min(candidates, key=lambda item: (item[1], item[0]))
        return None if max_distance_m is not None and best[1] > float(max_distance_m) else best

    def shortest_path(self, start_node: int, target_node: int) -> list[int] | None:
        start, target = int(start_node), int(target_node)
        if not (0 <= start < len(self.nodes) and 0 <= target < len(self.nodes)):
            raise IndexError("trajectory node index out of range")
        if start == target:
            return [start]
        adjacency: dict[int, list[tuple[int, float]]] = {index: [] for index in range(len(self.nodes))}
        for left, right, length in self.edges:
            adjacency[left].append((right, length))
            adjacency[right].append((left, length))
        distance, parent, queue = {start: 0.0}, {}, [(0.0, start)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != distance.get(node):
                continue
            if node == target:
                break
            for neighbor, edge_cost in adjacency[node]:
                candidate = cost + edge_cost
                if candidate < distance.get(neighbor, float("inf")):
                    distance[neighbor], parent[neighbor] = candidate, node
                    heapq.heappush(queue, (candidate, neighbor))
        if target not in distance:
            return None
        path = [target]
        while path[-1] != start:
            path.append(parent[path[-1]])
        return list(reversed(path))

    def certified_masks(self, *args, **kwargs):
        from .certification import certified_masks

        return certified_masks(self, *args, **kwargs)

    def certified_free_mask(self, *args, **kwargs):
        from .certification import certified_free_mask

        return certified_free_mask(self, *args, **kwargs)

    def save_npz(self, path, *, scene_id: str, source_episode_id: str):
        from .persistence import save_trajectory

        return save_trajectory(
            self, path, scene_id=scene_id, source_episode_id=source_episode_id
        )

    @classmethod
    def load_npz(cls, path) -> "TrajectoryGraph":
        from .persistence import load_trajectory

        return load_trajectory(path)
