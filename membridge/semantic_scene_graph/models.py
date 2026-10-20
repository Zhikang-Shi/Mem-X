"""Data contracts for semantic graph construction.

These records replace the legacy builder's loosely coupled dictionaries while
remaining JSON-compatible through ``to_dict``/``from_dict``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


def _float_list(value: Any, expected: int | None = None) -> list[float] | None:
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if expected is not None and result.size != expected:
        return None
    if not np.isfinite(result).all():
        return None
    return result.tolist()


@dataclass(frozen=True)
class Detection:
    """One model detection after mask/depth filtering and 3-D projection."""

    label: str
    score: float
    feature: Sequence[float]
    position_xyz: Sequence[float] | None
    bbox_xyxy: Sequence[float]
    rgb_path: str
    crop_path: str | None = None
    support: Mapping[str, Any] | None = None
    detector_id: str | None = None

    def __post_init__(self) -> None:
        feature = _float_list(self.feature)
        bbox = _float_list(self.bbox_xyxy, 4)
        position = _float_list(self.position_xyz, 3)
        if not str(self.label or "").strip():
            raise ValueError("detection label is empty")
        if feature is None or not feature:
            raise ValueError("detection feature must be a finite vector")
        if bbox is None:
            raise ValueError("bbox_xyxy must contain four finite values")
        score = float(self.score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("detection score must be in [0, 1]")
        object.__setattr__(self, "feature", tuple(feature))
        object.__setattr__(self, "bbox_xyxy", tuple(bbox))
        object.__setattr__(
            self, "position_xyz", tuple(position) if position is not None else None
        )


@dataclass
class ObjectNode:
    """Persistent object instance; ``node_id`` is immutable identity."""

    node_id: str
    label: str
    label_score: float
    feature: list[float]
    position_xyz: list[float] | None
    floor_id: int = 0
    parent_room: int | str | None = None
    parent_area: str | None = None
    description: str = ""
    bbox: list[float] | None = None
    full_image_path: str | None = None
    crop_path: str | None = None
    label_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = copy.deepcopy(self.metadata)
        payload.update(
            {
                "node_id": self.node_id,
                "node_level": 3,
                "label": self.label,
                "canonical_label": self.label,
                "label_score": float(self.label_score),
                "label_evidence": copy.deepcopy(self.label_evidence),
                "label_aliases": [],
                "dinov3_feature": list(map(float, self.feature)),
                "global_pos_3d": copy.deepcopy(self.position_xyz),
                "floor_id": int(self.floor_id),
                "parent_room": self.parent_room,
                "parent_area": self.parent_area,
                "llm_desc": self.description,
                "bbox": copy.deepcopy(self.bbox),
                "full_image_path": self.full_image_path,
                "cropped_image_path": self.crop_path,
            }
        )
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ObjectNode":
        known = {
            "node_id", "node_level", "label", "canonical_label", "label_score",
            "label_evidence", "label_aliases", "dinov3_feature", "global_pos_3d",
            "floor_id", "parent_room", "parent_area", "llm_desc", "bbox",
            "full_image_path", "cropped_image_path", "semantic_tag",
        }
        return cls(
            node_id=str(raw.get("node_id") or ""),
            label=str(
                raw.get("canonical_label") or raw.get("label")
                or raw.get("semantic_tag") or "object"
            ),
            label_score=float(raw.get("label_score", 0.0)),
            feature=list(map(float, raw.get("dinov3_feature") or raw.get("feature") or [])),
            position_xyz=_float_list(
                raw.get("global_pos_3d", raw.get("position_xyz")), 3
            ),
            floor_id=int(raw.get("floor_id", 0)),
            parent_room=raw.get("parent_room"),
            parent_area=raw.get("parent_area"),
            description=str(raw.get("llm_desc") or ""),
            bbox=_float_list(raw.get("bbox"), 4),
            full_image_path=raw.get("full_image_path"),
            crop_path=raw.get("cropped_image_path"),
            label_evidence=copy.deepcopy(dict(raw.get("label_evidence") or {})),
            metadata={key: copy.deepcopy(value) for key, value in raw.items() if key not in known},
        )


@dataclass(frozen=True)
class SpatialRelation:
    source: str
    target: str
    relation: str
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": float(self.weight),
        }


@dataclass
class SceneGraphMemory:
    objects: dict[str, ObjectNode] = field(default_factory=dict)
    rooms: dict[str, dict[str, Any]] = field(default_factory=dict)
    areas: dict[str, dict[str, Any]] = field(default_factory=dict)
    floors: dict[str, dict[str, Any]] = field(default_factory=dict)
    spatial_relations: list[SpatialRelation] = field(default_factory=list)

    def object_dicts(self) -> list[dict[str, Any]]:
        return [self.objects[key].to_dict() for key in sorted(self.objects)]

    def add_relation(self, relation: SpatialRelation) -> None:
        key = (relation.source, relation.target, relation.relation)
        for index, current in enumerate(self.spatial_relations):
            if (current.source, current.target, current.relation) == key:
                if relation.weight > current.weight:
                    self.spatial_relations[index] = relation
                return
        self.spatial_relations.append(relation)
