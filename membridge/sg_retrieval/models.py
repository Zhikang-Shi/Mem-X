"""Immutable retrieval snapshot built from the mutable semantic graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from membridge.semantic_scene_graph import SceneGraphMemory
from membridge.semantic_scene_graph.labels import retrieval_labels

from .entities import clean_phrase
from .rooms import ROOM_TYPE_RULES, infer_room


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str], **kwargs: Any) -> Any: ...


@dataclass
class RetrievalObject:
    node_id: str
    label: str
    retrieval_labels: list[str]
    position: list[float]
    floor_id: int
    parent_room: str | int | None
    room_label: str = ""
    embeddings: list[np.ndarray | None] = field(default_factory=list, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalRoom:
    room_id: str | int
    label: str
    confidence: float
    evidence: list[str]
    scores: dict[str, float]
    embedding: np.ndarray | None = field(default=None, repr=False)


@dataclass
class QuerySnapshot:
    objects: list[RetrievalObject]
    rooms: dict[str | int, RetrievalRoom]
    encoder: TextEncoder | None
    finalized_episode_count: int = 0


def encode_texts(encoder: TextEncoder | None, texts: list[str]) -> list[np.ndarray | None]:
    if encoder is None or not texts:
        return [None] * len(texts)
    try:
        try:
            values = encoder.encode(texts, normalize_embeddings=True, convert_to_tensor=False, show_progress_bar=False)
        except TypeError:
            values = encoder.encode(texts)
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[0] != len(texts):
            raise ValueError("text encoder returned an unexpected row count")
        array /= np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-8)
        return [array[index].copy() for index in range(array.shape[0])]
    except Exception:
        return [None] * len(texts)


def build_query_snapshot(
    memory: SceneGraphMemory, *, encoder: TextEncoder | None = None,
    finalized_episode_count: int = 0,
) -> QuerySnapshot:
    objects = []
    for node_id in sorted(memory.objects):
        node = memory.objects[node_id]
        if node.position_xyz is None:
            continue
        raw = node.to_dict()
        labels = [clean_phrase(label) for label in retrieval_labels(raw) if clean_phrase(label)]
        objects.append(RetrievalObject(
            node.node_id, clean_phrase(node.label), labels or [clean_phrase(node.label)],
            list(map(float, node.position_xyz)), int(node.floor_id), node.parent_room,
            metadata={key: raw.get(key) for key in (
                "navigation_waypoint", "best_observation", "best_observation_score",
                "observed_episode_ids", "observed_source_episode_ids",
                "best_observation_by_episode", "navigation_waypoint_by_episode",
            )},
        ))
    by_room: dict[str | int, list[RetrievalObject]] = {}
    for obj in objects:
        if obj.parent_room is not None:
            by_room.setdefault(obj.parent_room, []).append(obj)
    rooms = {}
    for room_id, room_objects in by_room.items():
        raw_room = memory.rooms.get(str(room_id), memory.rooms.get(f"room_{room_id}", {}))
        explicit = clean_phrase(raw_room.get("room_label") or raw_room.get("label"))
        inferred, confidence, evidence, scores = infer_room([obj.label for obj in room_objects])
        label = explicit if explicit in ROOM_TYPE_RULES else inferred
        rooms[room_id] = RetrievalRoom(room_id, label, confidence, evidence, scores)
        for obj in room_objects:
            obj.room_label = label
    texts = [label for obj in objects for label in obj.retrieval_labels]
    encoded = iter(encode_texts(encoder, texts))
    for obj in objects:
        obj.embeddings = [next(encoded) for _ in obj.retrieval_labels]
    room_ids = list(rooms)
    for room_id, embedding in zip(room_ids, encode_texts(encoder, [rooms[key].label for key in room_ids])):
        rooms[room_id].embedding = embedding
    return QuerySnapshot(objects, rooms, encoder, int(finalized_episode_count))
