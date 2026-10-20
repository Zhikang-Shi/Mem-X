"""Deterministic Floor/Room/Area hierarchy construction."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

import numpy as np

from .models import ObjectNode, SceneGraphMemory


def _connected_components(nodes: list[ObjectNode], threshold_m: float) -> list[list[ObjectNode]]:
    positions = {
        node.node_id: np.asarray(node.position_xyz, dtype=np.float64)
        for node in nodes
        if node.position_xyz is not None
    }
    adjacency = {node.node_id: set() for node in nodes}
    for left_index, left in enumerate(nodes):
        if left.node_id not in positions:
            continue
        for right in nodes[left_index + 1 :]:
            if right.node_id not in positions:
                continue
            distance = float(
                np.linalg.norm(
                    positions[left.node_id][[0, 2]] - positions[right.node_id][[0, 2]]
                )
            )
            if distance < float(threshold_m):
                adjacency[left.node_id].add(right.node_id)
                adjacency[right.node_id].add(left.node_id)
    by_id = {node.node_id: node for node in nodes}
    components: list[list[ObjectNode]] = []
    unseen = set(by_id)
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = deque([start])
        component = []
        while queue:
            current = queue.popleft()
            component.append(by_id[current])
            for neighbor in sorted(adjacency[current]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def build_hierarchy(memory: SceneGraphMemory, area_distance_m: float = 0.8) -> None:
    """Rebuild hierarchy nodes and object parent-area assignments in place."""

    memory.rooms.clear()
    memory.areas.clear()
    memory.floors.clear()
    by_room: dict[tuple[int, str], list[ObjectNode]] = defaultdict(list)
    for node in memory.objects.values():
        room_key = str(node.parent_room) if node.parent_room is not None else "unknown"
        by_room[(int(node.floor_id), room_key)].append(node)

    floor_rooms: dict[int, list[str]] = defaultdict(list)
    for (floor_id, room_key), objects in sorted(by_room.items()):
        room_id = room_key if room_key.startswith("room_") else f"room_{room_key}"
        area_ids = []
        for index, component in enumerate(_connected_components(objects, area_distance_m)):
            area_id = f"area_{room_key}_{index}"
            area_ids.append(area_id)
            points = [node.position_xyz for node in component if node.position_xyz is not None]
            centroid = (
                np.mean(np.asarray(points, dtype=np.float64), axis=0).tolist()
                if points
                else [0.0, 0.0, 0.0]
            )
            labels = sorted({node.label for node in component})
            memory.areas[area_id] = {
                "node_id": area_id,
                "node_level": 2,
                "area_tag": " / ".join(labels),
                "contained_obj_nodes": [node.node_id for node in component],
                "parent_room_id": room_id,
                "centroid": centroid,
            }
            for node in component:
                node.parent_area = area_id
        room_points = [node.position_xyz for node in objects if node.position_xyz is not None]
        memory.rooms[room_id] = {
            "node_id": room_id,
            "node_level": 1,
            "room_label": room_key,
            "floor_id": floor_id,
            "contained_area_ids": area_ids,
            "centroid": (
                np.mean(np.asarray(room_points), axis=0).tolist()
                if room_points
                else [0.0, 0.0, 0.0]
            ),
        }
        floor_rooms[floor_id].append(room_id)

    for floor_id, rooms in sorted(floor_rooms.items()):
        floor_node_id = f"floor_{floor_id}"
        memory.floors[floor_node_id] = {
            "node_id": floor_node_id,
            "node_level": 0,
            "floor_id": floor_id,
            "contained_room_ids": sorted(rooms),
        }


def hierarchy_edges(memory: SceneGraphMemory) -> Iterable[tuple[str, str, int]]:
    for node in memory.objects.values():
        if node.parent_area:
            yield node.node_id, node.parent_area, 3
    for area in memory.areas.values():
        yield str(area["node_id"]), str(area["parent_room_id"]), 2
    for room in memory.rooms.values():
        yield str(room["node_id"]), f"floor_{int(room['floor_id'])}", 1
