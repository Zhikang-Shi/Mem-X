"""Reviewed V7 two-stage object association, isolated from graph mutation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .keyframes import cosine_similarity
from .labels import normalize_label
from .models import Detection, ObjectNode


GENERIC_LABELS = frozenset({"object", "item", "thing", "stuff", "unknown", "entity"})


@dataclass(frozen=True)
class AssociationConfig:
    visual_similarity_threshold: float = 0.80
    spatial_distance_threshold_m: float = 1.50
    force_merge_distance_threshold_m: float = 1.00
    semantic_similarity_threshold: float = 0.60
    generic_visual_threshold: float = 0.65


@dataclass(frozen=True)
class AssociationMatch:
    detection_index: int
    node_id: str
    stage: int
    visual_similarity: float
    distance_m: float
    reason: str


def _position(value: Sequence[float] | None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result).all() else None


def _token_similarity(left: str, right: str) -> float:
    """Dependency-free fallback used when no text encoder is injected."""

    left_tokens = set(normalize_label(left).split("_"))
    right_tokens = set(normalize_label(right).split("_"))
    if not left_tokens or not right_tokens:
        return 0.0
    if left_tokens == right_tokens:
        return 1.0
    return 2.0 * len(left_tokens & right_tokens) / (
        len(left_tokens) + len(right_tokens)
    )


class V7ObjectAssociator:
    """Associate a frame one-to-one without mutating persistent objects.

    Stage 1 globally sorts strong DINO visual matches with a 3-D distance gate.
    Stage 2 handles remaining nearby nodes using exact labels, text semantics,
    or a lower visual threshold for generic detector labels.
    """

    def __init__(
        self,
        config: AssociationConfig | None = None,
        text_similarity: Callable[[str, str], float] | None = None,
    ) -> None:
        self.config = config or AssociationConfig()
        self.text_similarity = text_similarity or _token_similarity

    def associate(
        self,
        detections: Sequence[Detection],
        objects: Mapping[str, ObjectNode],
    ) -> list[AssociationMatch]:
        ordered_nodes = [objects[key] for key in sorted(objects)]
        used_nodes: set[str] = set()
        matched_detections: set[int] = set()
        matches: list[AssociationMatch] = []
        potential: list[tuple[float, float, int, ObjectNode]] = []

        for detection_index, detection in enumerate(detections):
            detection_position = _position(detection.position_xyz)
            if detection_position is None:
                continue
            for node in ordered_nodes:
                node_position = _position(node.position_xyz)
                if node_position is None or not node.feature:
                    continue
                visual = cosine_similarity(detection.feature, node.feature)
                distance = float(np.linalg.norm(detection_position - node_position))
                if (
                    visual > self.config.visual_similarity_threshold
                    and distance < self.config.spatial_distance_threshold_m
                ):
                    potential.append((-visual, distance, detection_index, node))

        for negative_visual, distance, detection_index, node in sorted(
            potential, key=lambda row: (row[0], row[1], row[2], row[3].node_id)
        ):
            if detection_index in matched_detections or node.node_id in used_nodes:
                continue
            visual = -negative_visual
            matches.append(
                AssociationMatch(
                    detection_index,
                    node.node_id,
                    1,
                    visual,
                    distance,
                    "strong_visual_and_spatial",
                )
            )
            matched_detections.add(detection_index)
            used_nodes.add(node.node_id)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            detection_position = _position(detection.position_xyz)
            if detection_position is None:
                continue
            candidates: list[tuple[float, ObjectNode]] = []
            for node in ordered_nodes:
                if node.node_id in used_nodes:
                    continue
                node_position = _position(node.position_xyz)
                if node_position is None:
                    continue
                distance = float(np.linalg.norm(detection_position - node_position))
                if distance < self.config.force_merge_distance_threshold_m:
                    candidates.append((distance, node))
            if not candidates:
                continue

            detection_label = normalize_label(detection.label)
            generic = detection_label in GENERIC_LABELS
            accepted: tuple[float, float, ObjectNode, str] | None = None
            if generic:
                for distance, node in candidates:
                    if not node.feature:
                        continue
                    visual = cosine_similarity(detection.feature, node.feature)
                    if visual <= self.config.generic_visual_threshold:
                        continue
                    row = (visual, -distance, node, "generic_visual")
                    if accepted is None or row[:2] > accepted[:2]:
                        accepted = row
            else:
                exact = [
                    (distance, node)
                    for distance, node in candidates
                    if detection_label == normalize_label(node.label)
                    or detection_label in normalize_label(node.label)
                    or normalize_label(node.label) in detection_label
                ]
                if exact:
                    distance, node = min(exact, key=lambda row: (row[0], row[1].node_id))
                    visual = (
                        cosine_similarity(detection.feature, node.feature)
                        if node.feature
                        else 0.0
                    )
                    accepted = (visual, -distance, node, "exact_label_nearby")
                else:
                    for distance, node in candidates:
                        semantic = float(self.text_similarity(detection_label, node.label))
                        if semantic <= self.config.semantic_similarity_threshold:
                            continue
                        row = (semantic, -distance, node, "semantic_label_nearby")
                        if accepted is None or row[:2] > accepted[:2]:
                            accepted = row
            if accepted is None:
                continue
            _, negative_distance, node, reason = accepted
            visual = (
                cosine_similarity(detection.feature, node.feature)
                if node.feature
                else 0.0
            )
            matches.append(
                AssociationMatch(
                    detection_index,
                    node.node_id,
                    2,
                    visual,
                    -negative_distance,
                    reason,
                )
            )
            matched_detections.add(detection_index)
            used_nodes.add(node.node_id)
        return sorted(matches, key=lambda match: match.detection_index)
