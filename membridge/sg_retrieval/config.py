"""Validated defaults for building and querying an SG retriever."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .entities import TaskEntityExtractor
from .models import QuerySnapshot
from .retriever import SceneGraphRetriever


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    minimum_finalized_episodes: int = 1

    def __post_init__(self) -> None:
        if int(self.top_k) < 1 or int(self.minimum_finalized_episodes) < 0:
            raise ValueError("top_k must be positive and episode threshold non-negative")

    def build(
        self, snapshot: QuerySnapshot, *, entity_extractor: TaskEntityExtractor | None = None
    ) -> SceneGraphRetriever:
        return SceneGraphRetriever(
            snapshot, min_finalized_episodes=self.minimum_finalized_episodes,
            entity_extractor=entity_extractor,
        )

    def query(
        self, retriever: SceneGraphRetriever, task_text: str | None = None, *,
        room_hint: str | None = None, target_object: str | None = None,
        required_observed_episode_ids: Iterable[str] | None = None,
    ):
        return retriever.query_two_stage(
            task_text, room_hint=room_hint, target_object=target_object,
            top_k=self.top_k,
            required_observed_episode_ids=required_observed_episode_ids,
        )
