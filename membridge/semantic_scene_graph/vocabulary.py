"""Validated object-query vocabulary owned by semantic perception."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .labels import normalize_label


@dataclass(frozen=True)
class DetectionSizePolicy:
    default_min_area_px: int = 1500
    default_min_side_px: int = 30
    small_min_area_px: int = 196
    small_min_side_px: int = 8

    def __post_init__(self) -> None:
        values = (
            self.default_min_area_px, self.default_min_side_px,
            self.small_min_area_px, self.small_min_side_px,
        )
        if any(int(value) < 1 for value in values):
            raise ValueError("detection size thresholds must be positive")


@dataclass(frozen=True)
class RuntimeVocabulary:
    labels: tuple[str, ...]
    mapping_labels: tuple[str, ...] = ()
    aliases: Mapping[str, str] = field(default_factory=dict)
    room_labels: tuple[str, ...] = ()
    room_aliases: Mapping[str, str] = field(default_factory=dict)
    small_object_labels: frozenset[str] = frozenset()
    size_policy: DetectionSizePolicy = DetectionSizePolicy()
    schema_version: int = 1
    vocabulary_version: str = "memory_module_v1"
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        labels = tuple(normalize_label(label).replace("_", " ") for label in self.labels)
        if not labels or any(not label for label in labels):
            raise ValueError("vocabulary requires non-empty labels")
        if len(set(labels)) != len(labels):
            raise ValueError("vocabulary labels must be unique")
        aliases = {
            normalize_label(key).replace("_", " "): normalize_label(value).replace("_", " ")
            for key, value in self.aliases.items()
        }
        if any(not key or value not in labels for key, value in aliases.items()):
            raise ValueError("every alias target must be a vocabulary label")
        mapping_labels = tuple(
            normalize_label(value).replace("_", " ") for value in self.mapping_labels
        ) or labels
        if any(not value for value in mapping_labels):
            raise ValueError("mapping labels must be non-empty")
        rooms = tuple(
            normalize_label(value).replace("_", " ") for value in self.room_labels
        )
        room_aliases = {
            normalize_label(key).replace("_", " "): normalize_label(value).replace("_", " ")
            for key, value in self.room_aliases.items()
        }
        if any(not key or value not in rooms for key, value in room_aliases.items()):
            raise ValueError("every room alias target must be a room vocabulary label")
        small = frozenset(normalize_label(value).replace("_", " ") for value in self.small_object_labels)
        if not small.issubset(labels):
            raise ValueError("small-object labels must belong to the vocabulary")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "mapping_labels", mapping_labels)
        object.__setattr__(self, "room_labels", rooms)
        object.__setattr__(self, "room_aliases", room_aliases)
        object.__setattr__(self, "small_object_labels", small)

    def canonical(self, label: Any) -> str:
        normalized = normalize_label(label).replace("_", " ")
        return str(self.aliases.get(normalized, normalized))

    def minimum_size(self, label: Any) -> tuple[int, int]:
        if self.canonical(label) in self.small_object_labels:
            return self.size_policy.small_min_area_px, self.size_policy.small_min_side_px
        return self.size_policy.default_min_area_px, self.size_policy.default_min_side_px

    def canonical_room(self, label: Any) -> str:
        normalized = normalize_label(label).replace("_", " ")
        return str(self.room_aliases.get(normalized, normalized))

    def accepts_box(self, label: Any, bbox_xyxy: Sequence[float]) -> bool:
        x1, y1, x2, y2 = map(float, bbox_xyxy)
        width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        minimum_area, minimum_side = self.minimum_size(label)
        return width * height >= minimum_area and min(width, height) >= minimum_side


def load_runtime_vocabulary(path: str | Path) -> RuntimeVocabulary:
    source = Path(path)
    raw_bytes = source.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("vocabulary JSON must be an object")
    size_raw = payload.get("size_filter") or {}
    return RuntimeVocabulary(
        labels=tuple(payload.get("labels") or ()),
        mapping_labels=tuple(payload.get("mapping_labels") or ()),
        aliases=dict(payload.get("aliases") or {}),
        room_labels=tuple(payload.get("rooms") or ()),
        room_aliases=dict(payload.get("room_aliases") or {}),
        small_object_labels=frozenset(payload.get("small_object_labels") or ()),
        size_policy=DetectionSizePolicy(**{
            key: int(size_raw[key]) for key in DetectionSizePolicy.__dataclass_fields__ if key in size_raw
        }),
        schema_version=int(payload.get("schema_version", 1)),
        vocabulary_version=str(payload.get("vocabulary_version") or source.stem),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
