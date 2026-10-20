"""Adapt the legacy online snapshot to the refactored retriever."""

from __future__ import annotations

from typing import Any

from .entities import TaskEntityExtractor
from .models import QuerySnapshot, RetrievalObject, RetrievalRoom
from .retriever import SceneGraphRetriever


def _snapshot(adapter: Any) -> QuerySnapshot | None:
    raw = getattr(adapter, "_query_snapshot", None)
    if not isinstance(raw, dict):
        return None
    objects = []
    for item in raw.get("objects") or []:
        labels = list(item.get("retrieval_labels") or [item.get("label", "")])
        embeddings = list(item.get("retrieval_embeddings") or [item.get("embedding")])
        objects.append(RetrievalObject(
            node_id=str(item.get("node_id") or ""),
            label=str(item.get("label") or ""),
            retrieval_labels=labels,
            position=list(item.get("global_pos_3d") or []),
            floor_id=int(item.get("floor_id") or 0),
            parent_room=item.get("parent_room"),
            room_label=str(item.get("room_label") or ""),
            embeddings=embeddings,
            metadata={key: item.get(key) for key in (
                "navigation_waypoint", "best_observation", "best_observation_score",
                "observed_episode_ids", "observed_source_episode_ids",
                "best_observation_by_episode", "navigation_waypoint_by_episode",
            )},
        ))
    rooms = {
        room_id: RetrievalRoom(
            room_id=room_id,
            label=str(room.get("label") or ""),
            confidence=float(room.get("confidence") or 0.0),
            evidence=list(room.get("evidence") or []),
            scores=dict(room.get("scores") or {}),
            embedding=room.get("embedding"),
        )
        for room_id, room in (raw.get("rooms") or {}).items()
    }
    return QuerySnapshot(
        objects, rooms, raw.get("encoder"),
        int(raw.get("finalized_episode_count") or getattr(adapter, "_finalized_episode_count", 0)),
    )


def install_online_adapter_retrieval(adapter_class: type) -> type:
    if getattr(adapter_class, "_memory_module_retrieval_installed", False):
        return adapter_class

    def query_two_stage(
        self, task_text=None, room_hint=None, target_object=None, top_k=5,
        required_observed_episode_ids=None,
    ):
        if not self.is_query_ready():
            return []
        snapshot = _snapshot(self)
        if snapshot is None:
            return []
        if target_object is None:
            room_hint, target_object = self.extract_task_entities(task_text)
        terms = [label for item in snapshot.objects for label in item.retrieval_labels]
        extractor = TaskEntityExtractor(
            terms,
            object_aliases=(self._object_aliases() if hasattr(self, "_object_aliases") else {}),
        )
        retriever = SceneGraphRetriever(
            snapshot,
            min_finalized_episodes=int(getattr(self, "_query_min_finalized_episodes", 1)),
            entity_extractor=extractor,
        )
        return retriever.query_two_stage(
            task_text, room_hint=room_hint, target_object=target_object,
            top_k=top_k,
            required_observed_episode_ids=required_observed_episode_ids,
        )

    adapter_class.query_two_stage = query_two_stage
    adapter_class._memory_module_retrieval_installed = True
    return adapter_class
