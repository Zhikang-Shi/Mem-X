"""Single winning-label state copied from the reviewed V8 mapper."""

from __future__ import annotations

import math
import re
from typing import Any, MutableMapping


LABEL_SCORE_SCHEMA_VERSION = 1
_SEPARATOR = re.compile(r"[^0-9a-zA-Z]+")


def normalize_label(value: Any) -> str:
    return _SEPARATOR.sub("_", str(value or "").strip().lower()).strip("_")


def _score(value: Any, default: float | None = None) -> float:
    if value is None and default is not None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("detector label score must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("detector label score must be finite and in [0, 1]")
    return result


def ensure_label_state(
    item: MutableMapping[str, Any], default_score: float = 0.0
) -> MutableMapping[str, Any]:
    original = str(item.get("label") or item.get("semantic_tag") or "").strip()
    canonical = normalize_label(item.get("canonical_label"))
    if not canonical:
        canonical = normalize_label(original.split()[0] if original else "")
    if not canonical:
        raise ValueError(f"node {item.get('node_id')!r} has no usable label")
    label_score = _score(item.get("label_score"), default_score)
    evidence = item.get("label_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    cleaned: dict[str, dict[str, Any]] = {}
    for raw_label, raw_stats in evidence.items():
        label = normalize_label(raw_label)
        if not label or not isinstance(raw_stats, dict):
            continue
        try:
            maximum = _score(raw_stats.get("max_score"), 0.0)
            total = float(raw_stats.get("score_sum", maximum))
            count = max(0, int(raw_stats.get("count", 0)))
        except (TypeError, ValueError):
            continue
        cleaned[label] = {
            "count": count,
            "score_sum": total,
            "max_score": maximum,
            "first_frame": raw_stats.get("first_frame"),
            "last_frame": raw_stats.get("last_frame"),
        }
    current = cleaned.setdefault(
        canonical,
        {
            "count": 0,
            "score_sum": 0.0,
            "max_score": label_score,
            "first_frame": None,
            "last_frame": None,
        },
    )
    current["max_score"] = max(float(current.get("max_score", 0.0)), label_score)
    item.update(
        {
            "label": canonical,
            "canonical_label": canonical,
            "label_score": label_score,
            "label_evidence": cleaned,
            "label_aliases": [],
            "label_score_schema_version": LABEL_SCORE_SCHEMA_VERSION,
        }
    )
    if original and original != canonical:
        item["legacy_mixed_label_import"] = original
    return item


def record_label(
    item: MutableMapping[str, Any],
    label: Any,
    score: Any,
    frame_index: int | None = None,
) -> dict[str, Any]:
    ensure_label_state(item)
    normalized = normalize_label(label)
    if not normalized:
        raise ValueError("observed label is empty")
    observed_score = _score(score)
    stats = item["label_evidence"].setdefault(
        normalized,
        {
            "count": 0,
            "score_sum": 0.0,
            "max_score": 0.0,
            "first_frame": frame_index,
            "last_frame": frame_index,
        },
    )
    stats["count"] = int(stats.get("count", 0)) + 1
    stats["score_sum"] = float(stats.get("score_sum", 0.0)) + observed_score
    stats["max_score"] = max(float(stats.get("max_score", 0.0)), observed_score)
    stats["first_frame"] = stats.get("first_frame", frame_index)
    stats["last_frame"] = frame_index
    previous_label = str(item["label"])
    previous_score = float(item["label_score"])
    changed = observed_score > previous_score
    if changed:
        item["label"] = normalized
        item["canonical_label"] = normalized
        item["label_score"] = observed_score
    item["label_aliases"] = []
    return {
        "changed": changed,
        "previous_label": previous_label,
        "previous_score": previous_score,
        "observed_label": normalized,
        "observed_score": observed_score,
        "final_label": item["label"],
        "final_score": float(item["label_score"]),
    }


def initialize_label(
    item: MutableMapping[str, Any], label: Any, score: Any, frame_index: int | None = None
) -> dict[str, Any]:
    normalized = normalize_label(label)
    item.update(
        {
            "label": normalized,
            "canonical_label": normalized,
            "label_score": 0.0,
            "label_evidence": {},
            "label_aliases": [],
        }
    )
    ensure_label_state(item)
    return record_label(item, normalized, score, frame_index)


def retrieval_labels(item: MutableMapping[str, Any]) -> list[str]:
    ensure_label_state(item)
    return [str(item["label"])]
