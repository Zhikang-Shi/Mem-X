"""Room-first, object-second SG retrieval with provenance filtering."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .entities import TaskEntityExtractor, clean_phrase, phrase_in_text
from .models import QuerySnapshot, encode_texts


def lexical_match_score(query: object, candidate: object, aliases: dict[str, str] | None = None) -> float:
    aliases = aliases or {}
    query_text, candidate_text = clean_phrase(query), clean_phrase(candidate)
    query_text, candidate_text = aliases.get(query_text, query_text), aliases.get(candidate_text, candidate_text)
    if not query_text or not candidate_text:
        return 0.0
    if query_text == candidate_text:
        return 1.0
    if phrase_in_text(query_text, candidate_text) or phrase_in_text(candidate_text, query_text):
        return 0.9
    singular_query = query_text[:-1] if query_text.endswith("s") else query_text
    singular_candidate = candidate_text[:-1] if candidate_text.endswith("s") else candidate_text
    return 0.9 if (
        singular_query == singular_candidate
        or phrase_in_text(singular_query, singular_candidate)
        or phrase_in_text(singular_candidate, singular_query)
    ) else 0.0


def embedding_score(query: np.ndarray | None, candidate: np.ndarray | None) -> float:
    if query is None or candidate is None:
        return 0.0
    try:
        return float(np.dot(np.asarray(query, dtype=np.float32), np.asarray(candidate, dtype=np.float32)))
    except (TypeError, ValueError):
        return 0.0


class SceneGraphRetriever:
    def __init__(
        self, snapshot: QuerySnapshot, *, min_finalized_episodes: int = 1,
        entity_extractor: TaskEntityExtractor | None = None,
    ) -> None:
        self.snapshot = snapshot
        terms = [label for obj in snapshot.objects for label in obj.retrieval_labels]
        self.entities = entity_extractor or TaskEntityExtractor(terms)
        self.min_finalized_episodes = int(min_finalized_episodes)

    @property
    def ready(self) -> bool:
        return bool(self.snapshot.objects) and self.snapshot.finalized_episode_count >= self.min_finalized_episodes

    def query_two_stage(
        self, task_text: str | None = None, *, room_hint: str | None = None,
        target_object: str | None = None, top_k: int = 5,
        required_observed_episode_ids: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.ready:
            return []
        if target_object is None:
            extracted_room, target_object = self.entities.extract(task_text or "")
            room_hint = room_hint or extracted_room
        room_hint = self.entities.canonical_room(room_hint)
        target_object = self.entities.canonical_object(target_object)
        if not target_object:
            return []
        required = frozenset(str(value) for value in (required_observed_episode_ids or ()))
        candidates = [
            obj for obj in self.snapshot.objects
            if required.issubset(frozenset(obj.metadata.get("observed_episode_ids") or []))
        ]
        room_match = self._best_room(room_hint) if room_hint else None
        query_embedding = encode_texts(self.snapshot.encoder, [target_object])[0]
        results = []
        for obj in candidates:
            lexical = max(lexical_match_score(target_object, label, self.entities.object_aliases) for label in obj.retrieval_labels)
            semantic = max((embedding_score(query_embedding, item) for item in obj.embeddings), default=0.0)
            score = max(lexical, semantic)
            room_consistent = bool(room_match and obj.parent_room == room_match["room_id"])
            results.append({
                "uid": obj.node_id, "node_id": obj.node_id, "label": obj.label,
                "label_aliases": [label for label in obj.retrieval_labels if label != obj.label],
                "position": obj.position, "floor_id": obj.floor_id,
                "parent_room": obj.parent_room, "room_label": obj.room_label,
                "waypoint": obj.metadata.get("navigation_waypoint"),
                "best_observation": obj.metadata.get("best_observation"),
                "best_observation_score": obj.metadata.get("best_observation_score"),
                "observed_episode_ids": obj.metadata.get("observed_episode_ids") or [],
                "observed_source_episode_ids": obj.metadata.get("observed_source_episode_ids") or [],
                "best_observation_by_episode": obj.metadata.get("best_observation_by_episode") or {},
                "navigation_waypoint_by_episode": obj.metadata.get("navigation_waypoint_by_episode") or {},
                "required_observed_episode_ids": sorted(required),
                "room_score": room_match["score"] if room_consistent else 0.0,
                "requested_room_evidence_score": room_match["evidence_score"] if room_consistent else 0.0,
                "room_consistent": room_consistent, "object_score": score, "score": score,
                "query_room": room_hint, "query_object": target_object,
                "source": "inherit_sg_two_stage",
            })
        results.sort(key=lambda item: (
            -item["score"], 0 if item["room_consistent"] else 1,
            item["best_observation_score"] if item["best_observation_score"] is not None else float("inf"),
            str(item["node_id"]),
        ))
        return results[:max(1, int(top_k))]

    def _best_room(self, room_hint: str) -> dict[str, Any] | None:
        query_embedding = encode_texts(self.snapshot.encoder, [room_hint])[0]
        candidates = []
        for room_id, room in self.snapshot.rooms.items():
            semantic = max(lexical_match_score(room_hint, room.label), embedding_score(query_embedding, room.embedding))
            evidence = float(room.scores.get(room_hint, 0.0))
            evidence_score = min(1.0, 0.6 + 0.08 * evidence) if evidence >= 2.0 else 0.0
            candidates.append((max(semantic, evidence_score), evidence, str(room_id), room_id, room))
        if not candidates:
            return None
        score, evidence, _, room_id, room = min(candidates, key=lambda item: (-item[0], -item[1], item[2]))
        if score < 0.6:
            return None
        return {"room_id": room_id, "label": room.label, "score": score, "evidence_score": evidence}
