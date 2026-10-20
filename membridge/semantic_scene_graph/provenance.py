"""Canonical object-to-episode provenance contract."""

from __future__ import annotations

import copy
import math
from typing import Any, Iterable, Mapping, MutableMapping


SCHEMA_VERSION = 1
PROVENANCE_FIELDS = (
    "observed_episode_ids",
    "observed_source_episode_ids",
    "best_observation_by_episode",
    "navigation_waypoint_by_episode",
    "observation_provenance_schema_version",
)
GLOBAL_OBSERVATION_FIELDS = (
    "best_observation",
    "best_observation_score",
    "navigation_waypoint",
    "waypoint_schema_version",
    "trajectory_node_id",
)


def normalize_episode_id(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def ordered_unique_episode_ids(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_episode_id(value)
        if normalized is not None and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def ensure_provenance(
    item: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    item["observed_episode_ids"] = ordered_unique_episode_ids(
        item.get("observed_episode_ids") or []
    )
    item["observed_source_episode_ids"] = ordered_unique_episode_ids(
        item.get("observed_source_episode_ids") or []
    )
    for field in ("best_observation_by_episode", "navigation_waypoint_by_episode"):
        value = item.get(field)
        item[field] = value if isinstance(value, dict) else {}
    item["observation_provenance_schema_version"] = SCHEMA_VERSION
    return item


def bbox_pixel_area(bbox: Any) -> float | None:
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError, IndexError):
        return None
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def observation_selection_score(
    bbox: Any,
    image_width: int,
    image_height: int,
    center_weight: float = 0.4,
    area_weight: float = 0.6,
) -> float:
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError, IndexError):
        return 1.0
    width, height = int(image_width), int(image_height)
    if width <= 0 or height <= 0:
        return 1.0
    center_weight = max(0.0, float(center_weight))
    area_weight = max(0.0, float(area_weight))
    total = center_weight + area_weight
    if total <= 0.0:
        raise ValueError("observation weights must have a positive sum")
    center_weight /= total
    area_weight /= total
    center_distance = math.hypot(
        (x1 + x2) / 2.0 - width / 2.0,
        (y1 + y2) / 2.0 - height / 2.0,
    )
    center_penalty = min(
        1.0, center_distance / max(math.hypot(width / 2.0, height / 2.0), 1e-8)
    )
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_penalty = 1.0 - math.sqrt(min(1.0, area / float(width * height)))
    return center_weight * center_penalty + area_weight * area_penalty


def _score(payload: Mapping[str, Any] | None, explicit: Any = None) -> float | None:
    value = explicit
    if value is None and isinstance(payload, Mapping):
        value = payload.get("selection_score")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _richer_tie(
    current: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    current_score: float | None,
    previous_score: float | None,
) -> bool:
    return bool(
        isinstance(previous, Mapping)
        and current_score is not None
        and previous_score is not None
        and abs(current_score - previous_score) <= 1e-12
        and previous.get("trajectory_node_id") is None
        and current.get("trajectory_node_id") is not None
    )


def record_observation(
    item: MutableMapping[str, Any],
    runtime_episode_id: Any,
    source_episode_id: Any = None,
    observation: Mapping[str, Any] | None = None,
    waypoint: Mapping[str, Any] | None = None,
    selection_score: Any = None,
) -> dict[str, bool]:
    ensure_provenance(item)
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
    waypoint_payload = copy.deepcopy(dict(waypoint or {}))
    for payload in (observation_payload, waypoint_payload):
        payload["runtime_episode_id"] = runtime_id
        payload["source_episode_id"] = source_id
    current_score = _score(observation_payload, selection_score)
    if current_score is not None:
        observation_payload["selection_score"] = current_score

    previous = item["best_observation_by_episode"].get(runtime_id)
    previous_score = _score(previous)
    episode_updated = bool(
        previous is None
        or (
            current_score is not None
            and (previous_score is None or current_score < previous_score - 1e-12)
        )
        or _richer_tie(observation_payload, previous, current_score, previous_score)
    )
    if episode_updated:
        item["best_observation_by_episode"][runtime_id] = observation_payload
        item["navigation_waypoint_by_episode"][runtime_id] = waypoint_payload

    previous_global = item.get("best_observation")
    previous_global_score = _score(previous_global, item.get("best_observation_score"))
    global_updated = bool(
        previous_global is None
        or (
            current_score is not None
            and (
                previous_global_score is None
                or current_score < previous_global_score - 1e-12
            )
        )
        or _richer_tie(
            observation_payload, previous_global, current_score, previous_global_score
        )
    )
    if global_updated:
        item["best_observation"] = observation_payload
        item["best_observation_score"] = current_score
        item["navigation_waypoint"] = waypoint_payload
        item["waypoint_schema_version"] = SCHEMA_VERSION
        if "trajectory_node_id" in waypoint_payload:
            item["trajectory_node_id"] = waypoint_payload["trajectory_node_id"]
    return {"episode_updated": episode_updated, "global_updated": global_updated}


def provenance_payload(
    item: MutableMapping[str, Any], include_global: bool = True
) -> dict[str, Any]:
    ensure_provenance(item)
    fields = list(PROVENANCE_FIELDS)
    if include_global:
        fields.extend(GLOBAL_OBSERVATION_FIELDS)
    return {field: copy.deepcopy(item[field]) for field in fields if field in item}
