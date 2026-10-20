"""Canonical object-to-episode provenance contract for INHerit-SG.

This module deliberately has no mapper, Habitat, NumPy, or Torch dependency so
the base mapper, the online runtime, export tools, and artifact migrations all
use exactly the same schema semantics.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Iterable, Mapping, MutableMapping, Optional


OBJECT_PROVENANCE_SCHEMA_VERSION = 1

OBJECT_PROVENANCE_FIELDS = (
    "observed_episode_ids",
    "observed_source_episode_ids",
    "best_observation_by_episode",
    "navigation_waypoint_by_episode",
    "observation_provenance_schema_version",
)

OBJECT_OBSERVATION_FIELDS = (
    "best_observation",
    "best_observation_score",
    "navigation_waypoint",
    "waypoint_schema_version",
    "trajectory_node_id",
)


def normalize_episode_id(value: Any) -> Optional[str]:
    """Return the stable string representation used by every SG artifact."""

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def ordered_unique_episode_ids(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_episode_id(value)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def ensure_object_provenance(item: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Normalize an object node to the canonical, always-present schema."""

    item["observed_episode_ids"] = ordered_unique_episode_ids(
        item.get("observed_episode_ids") or []
    )
    item["observed_source_episode_ids"] = ordered_unique_episode_ids(
        item.get("observed_source_episode_ids") or []
    )
    for field in (
        "best_observation_by_episode",
        "navigation_waypoint_by_episode",
    ):
        value = item.get(field)
        item[field] = value if isinstance(value, dict) else {}
    item["observation_provenance_schema_version"] = (
        OBJECT_PROVENANCE_SCHEMA_VERSION
    )
    return item


def bbox_pixel_area(bbox: Any) -> Optional[float]:
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError, IndexError):
        return None
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def observation_selection_score(
    bbox: Any,
    image_width: int,
    image_height: int,
    *,
    center_weight: float = 0.4,
    area_weight: float = 0.6,
) -> float:
    """Score a view using the shared lower-is-better center/area policy."""

    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError, IndexError):
        return 1.0
    width = int(image_width)
    height = int(image_height)
    if width <= 0 or height <= 0:
        return 1.0
    center_weight = max(0.0, float(center_weight))
    area_weight = max(0.0, float(area_weight))
    weight_sum = center_weight + area_weight
    if weight_sum <= 0.0:
        raise ValueError("observation selection weights must have a positive sum")
    center_weight /= weight_sum
    area_weight /= weight_sum

    box_cx = (x1 + x2) / 2.0
    box_cy = (y1 + y2) / 2.0
    image_cx = float(width) / 2.0
    image_cy = float(height) / 2.0
    max_center_distance = max(math.hypot(image_cx, image_cy), 1.0e-8)
    center_offset = min(
        1.0,
        math.hypot(box_cx - image_cx, box_cy - image_cy)
        / max_center_distance,
    )
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_fraction = min(1.0, box_area / float(width * height))
    area_penalty = 1.0 - math.sqrt(area_fraction)
    return center_weight * center_offset + area_weight * area_penalty


def _score(payload: Optional[Mapping[str, Any]], explicit: Any = None) -> Optional[float]:
    value = explicit
    if value is None and isinstance(payload, Mapping):
        value = payload.get("selection_score")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_richer_tied_observation(
    current: Mapping[str, Any],
    previous: Optional[Mapping[str, Any]],
    current_score: Optional[float],
    previous_score: Optional[float],
) -> bool:
    if not isinstance(previous, Mapping):
        return False
    if current_score is None or previous_score is None:
        return False
    if abs(current_score - previous_score) > 1.0e-12:
        return False
    return (
        previous.get("trajectory_node_id") is None
        and current.get("trajectory_node_id") is not None
    )


def record_object_observation(
    item: MutableMapping[str, Any],
    *,
    runtime_episode_id: Any,
    source_episode_id: Any = None,
    observation: Optional[Mapping[str, Any]] = None,
    waypoint: Optional[Mapping[str, Any]] = None,
    selection_score: Any = None,
) -> dict[str, bool]:
    """Attach one observation to an object using the canonical contract.

    When no separate source ID is supplied, source identity equals runtime
    identity. This makes provenance complete for ordinary single-dataset runs
    while still supporting Level-3's explicit ep0/ep1-to-source mapping.
    """

    ensure_object_provenance(item)
    runtime_id = normalize_episode_id(runtime_episode_id)
    source_id = normalize_episode_id(source_episode_id) or runtime_id
    if runtime_id is None:
        return {"episode_updated": False, "global_updated": False}

    item["observed_episode_ids"] = ordered_unique_episode_ids(
        [*item["observed_episode_ids"], runtime_id]
    )
    if source_id is not None:
        item["observed_source_episode_ids"] = ordered_unique_episode_ids(
            [*item["observed_source_episode_ids"], source_id]
        )

    observation_payload = copy.deepcopy(dict(observation or {}))
    observation_payload["runtime_episode_id"] = runtime_id
    observation_payload["source_episode_id"] = source_id
    waypoint_payload = copy.deepcopy(dict(waypoint or {}))
    waypoint_payload["runtime_episode_id"] = runtime_id
    waypoint_payload["source_episode_id"] = source_id
    current_score = _score(observation_payload, selection_score)
    if current_score is not None:
        observation_payload["selection_score"] = current_score

    observations = item["best_observation_by_episode"]
    previous = observations.get(runtime_id)
    previous_score = _score(previous)
    episode_updated = bool(
        previous is None
        or (
            current_score is not None
            and (previous_score is None or current_score < previous_score - 1.0e-12)
        )
        or _is_richer_tied_observation(
            observation_payload,
            previous,
            current_score,
            previous_score,
        )
    )
    if episode_updated:
        observations[runtime_id] = observation_payload
        item["navigation_waypoint_by_episode"][runtime_id] = waypoint_payload

    previous_global = item.get("best_observation")
    previous_global_score = _score(
        previous_global,
        item.get("best_observation_score"),
    )
    global_updated = bool(
        previous_global is None
        or (
            current_score is not None
            and (
                previous_global_score is None
                or current_score < previous_global_score - 1.0e-12
            )
        )
        or _is_richer_tied_observation(
            observation_payload,
            previous_global,
            current_score,
            previous_global_score,
        )
    )
    if global_updated:
        item["best_observation"] = observation_payload
        item["best_observation_score"] = current_score
        item["navigation_waypoint"] = waypoint_payload
        item["waypoint_schema_version"] = OBJECT_PROVENANCE_SCHEMA_VERSION
        if "trajectory_node_id" in waypoint_payload:
            item["trajectory_node_id"] = waypoint_payload["trajectory_node_id"]

    return {
        "episode_updated": episode_updated,
        "global_updated": global_updated,
    }


def object_provenance_payload(
    item: MutableMapping[str, Any],
    *,
    include_global_observation: bool = True,
) -> dict[str, Any]:
    ensure_object_provenance(item)
    fields = list(OBJECT_PROVENANCE_FIELDS)
    if include_global_observation:
        fields.extend(OBJECT_OBSERVATION_FIELDS)
    return {
        field: copy.deepcopy(item[field])
        for field in fields
        if field in item
    }


def sync_object_provenance_to_graph(
    graph: Any,
    item: MutableMapping[str, Any],
) -> None:
    node_id = item.get("node_id")
    if graph is None or node_id is None:
        return
    if not graph.has_node(node_id):
        graph.add_node(node_id)
    graph.nodes[node_id].update(object_provenance_payload(item))


def merge_object_provenance_into_nodes(
    nodes: Iterable[Mapping[str, Any]],
    memory_kb: Iterable[MutableMapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy canonical provenance from memory objects into semantic JSON nodes."""

    by_node_id = {
        str(item.get("node_id")): item
        for item in memory_kb
        if item.get("node_id") is not None
    }
    merged: list[dict[str, Any]] = []
    for raw_node in nodes:
        node = copy.deepcopy(dict(raw_node))
        item = by_node_id.get(str(node.get("node_id")))
        if item is not None:
            node.update(object_provenance_payload(item))
        merged.append(node)
    return merged
