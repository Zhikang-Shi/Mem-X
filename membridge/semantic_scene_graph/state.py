"""Native continuation state for the refactored semantic builder."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .builder import SceneGraphBuilder
from .models import SceneGraphMemory


BUILDER_STATE_SCHEMA_VERSION = 1


def _memory_counters(memory: SceneGraphMemory) -> dict[str, int]:
    counters: dict[str, int] = {}
    for node in memory.objects.values():
        suffix = node.node_id.rsplit("_", 1)[-1]
        next_value = int(suffix) + 1 if suffix.isdigit() else 1
        counters[node.label] = max(counters.get(node.label, 0), next_value)
    return counters


@dataclass(frozen=True)
class BuilderContinuationState:
    label_counters: Mapping[str, int] = field(default_factory=dict)
    last_keyframe_feature: tuple[float, ...] | None = None
    frames_seen: int = 0
    confirmed_keyframes: int = 0
    schema_version: int = BUILDER_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        counters = {str(key): int(value) for key, value in self.label_counters.items()}
        if any(not key or value < 0 for key, value in counters.items()):
            raise ValueError("builder label counters must be non-negative")
        feature = self.last_keyframe_feature
        if feature is not None:
            feature = tuple(map(float, feature))
            if not feature or not all(math.isfinite(value) for value in feature):
                raise ValueError("last keyframe feature must be finite and non-empty")
        if self.frames_seen < 0 or not 0 <= self.confirmed_keyframes <= self.frames_seen:
            raise ValueError("invalid semantic frame counters")
        if int(self.schema_version) != BUILDER_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported semantic builder-state schema")
        object.__setattr__(self, "label_counters", counters)
        object.__setattr__(self, "last_keyframe_feature", feature)

    @classmethod
    def from_builder(
        cls, builder: SceneGraphBuilder, *, frames_seen: int = 0,
        confirmed_keyframes: int = 0,
    ) -> "BuilderContinuationState":
        feature = builder.keyframes.last_feature
        return cls(
            label_counters=dict(builder.label_counters),
            last_keyframe_feature=(
                tuple(map(float, np.asarray(feature).reshape(-1)))
                if feature is not None else None
            ),
            frames_seen=int(frames_seen), confirmed_keyframes=int(confirmed_keyframes),
        )

    @classmethod
    def from_memory(cls, memory: SceneGraphMemory) -> "BuilderContinuationState":
        return cls(label_counters=_memory_counters(memory))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "state_kind": "memory_module_semantic_continuation",
            "label_counters": dict(self.label_counters),
            "last_keyframe_feature": list(self.last_keyframe_feature) if self.last_keyframe_feature else None,
            "frames_seen": int(self.frames_seen),
            "confirmed_keyframes": int(self.confirmed_keyframes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BuilderContinuationState":
        if payload.get("state_kind") != "memory_module_semantic_continuation":
            raise ValueError("unexpected semantic builder-state kind")
        counters = payload.get("label_counters") or {}
        if not isinstance(counters, Mapping):
            raise ValueError("label_counters must be a mapping")
        feature = payload.get("last_keyframe_feature")
        if feature is not None and not isinstance(feature, (list, tuple)):
            raise ValueError("last_keyframe_feature must be an array or null")
        return cls(
            label_counters={str(key): int(value) for key, value in counters.items()},
            last_keyframe_feature=tuple(feature) if feature is not None else None,
            frames_seen=int(payload.get("frames_seen", 0)),
            confirmed_keyframes=int(payload.get("confirmed_keyframes", 0)),
            schema_version=int(payload.get("schema_version", -1)),
        )

    def restore(self, builder: SceneGraphBuilder) -> None:
        minimum = _memory_counters(builder.memory)
        builder.label_counters = {
            key: max(minimum.get(key, 0), int(value))
            for key, value in {**minimum, **dict(self.label_counters)}.items()
        }
        builder.keyframes.last_feature = (
            np.asarray(self.last_keyframe_feature, dtype=np.float64)
            if self.last_keyframe_feature is not None else None
        )


def save_builder_state(state: BuilderContinuationState, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(state.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return destination


def load_builder_state(path: str | Path) -> BuilderContinuationState:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("semantic builder state must be a JSON object")
    return BuilderContinuationState.from_dict(payload)
