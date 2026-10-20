"""Dependency-injection protocols for model and semantic services."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .models import Detection, ObjectNode, SpatialRelation


class FrameFeatureExtractor(Protocol):
    def __call__(self, rgb: Any) -> Sequence[float]: ...


class DetectionProvider(Protocol):
    def __call__(
        self, rgb: Any, depth: Any, camera_to_world: Any, frame_index: int
    ) -> Sequence[Detection]: ...


class TextSimilarity(Protocol):
    def __call__(self, left: str, right: str) -> float: ...


class RoomLocator(Protocol):
    def __call__(self, position_xyz: Sequence[float], floor_id: int) -> int | str | None: ...


class RelationReasoner(Protocol):
    def __call__(
        self,
        object_nodes: Sequence[ObjectNode],
        frame_context: Mapping[str, Any],
    ) -> Sequence[SpatialRelation]: ...


class DescriptionProvider(Protocol):
    def __call__(self, detection: Detection) -> str: ...
