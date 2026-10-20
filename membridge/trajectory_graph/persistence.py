"""Versioned NPZ persistence for trajectory topology."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .graph import TrajectoryGraph, TrajectoryNode

TRAJECTORY_SCHEMA_VERSION = 1
COORDINATE_FRAME = "habitat_world"


def save_trajectory(
    graph: TrajectoryGraph, path: str | os.PathLike[str], *, scene_id: str,
    source_episode_id: str,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    edges = np.asarray([[a, b] for a, b, _ in graph.edges], dtype=np.int32).reshape(-1, 2)
    frames = sorted(graph.frame_to_node)
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, schema_version=TRAJECTORY_SCHEMA_VERSION,
            coordinate_frame=COORDINATE_FRAME, scene_id=str(scene_id),
            source_episode_id=str(source_episode_id),
            min_node_translation_m=graph.min_node_translation_m,
            max_edge_length_m=graph.max_edge_length_m,
            node_positions_xyz=np.asarray([node.position_xyz for node in graph.nodes], dtype=float).reshape(-1, 3),
            node_headings_rad=np.asarray([node.heading_rad for node in graph.nodes]),
            node_first_frame_idx=np.asarray([node.first_frame_idx for node in graph.nodes]),
            node_last_frame_idx=np.asarray([node.last_frame_idx for node in graph.nodes]),
            node_steps=np.asarray([node.step for node in graph.nodes]),
            node_floor_ids=np.asarray([node.floor_id for node in graph.nodes]),
            node_episode_ids=np.asarray([node.episode_id for node in graph.nodes]),
            edge_indices=edges,
            edge_lengths_m=np.asarray([length for _, _, length in graph.edges]),
            frame_indices=np.asarray(frames),
            frame_node_ids=np.asarray([graph.frame_to_node[index] for index in frames]),
        )
    os.replace(temporary, destination)
    return destination


def load_trajectory(path: str | os.PathLike[str]) -> TrajectoryGraph:
    with np.load(path, allow_pickle=False) as data:
        if int(data["schema_version"]) != TRAJECTORY_SCHEMA_VERSION or str(data["coordinate_frame"]) != COORDINATE_FRAME:
            raise ValueError("unsupported trajectory graph schema")
        graph = TrajectoryGraph(
            min_node_translation_m=float(data["min_node_translation_m"]),
            max_edge_length_m=float(data["max_edge_length_m"]),
        )
        positions = np.asarray(data["node_positions_xyz"], dtype=float)
        fields = [np.asarray(data[name]) for name in (
            "node_headings_rad", "node_first_frame_idx", "node_last_frame_idx",
            "node_steps", "node_floor_ids", "node_episode_ids",
        )]
        edges, lengths = np.asarray(data["edge_indices"]), np.asarray(data["edge_lengths_m"])
        frames, frame_nodes = np.asarray(data["frame_indices"]), np.asarray(data["frame_node_ids"])
    count = positions.shape[0]
    if positions.shape != (count, 3) or any(field.shape != (count,) for field in fields):
        raise ValueError("trajectory node arrays have inconsistent shapes")
    headings, first, last, steps, floors, episodes = fields
    for index in range(count):
        graph.nodes.append(TrajectoryNode(
            index, tuple(map(float, positions[index])), float(headings[index]),
            int(first[index]), int(last[index]), int(steps[index]),
            int(floors[index]), str(episodes[index]),
        ))
    if edges.shape != (lengths.shape[0], 2):
        raise ValueError("trajectory edge arrays have inconsistent shapes")
    for pair, length in zip(edges, lengths):
        left, right = map(int, pair)
        if not (0 <= left < count and 0 <= right < count):
            raise ValueError("trajectory edge references an unknown node")
        graph.edges.append((left, right, float(length)))
    if frames.shape != frame_nodes.shape:
        raise ValueError("trajectory frame mapping arrays have inconsistent shapes")
    graph.frame_to_node = {int(frame): int(node) for frame, node in zip(frames, frame_nodes)}
    if any(not 0 <= node < count for node in graph.frame_to_node.values()):
        raise ValueError("trajectory frame mapping references an unknown node")
    return graph
