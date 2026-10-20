"""V8 canonical-label state layered on top of the unchanged V7 Re-ID policy.

This module deliberately does not decide whether two detections are the same
instance.  The V7 mapper owns that decision.  It is called only after V7 has
selected a node to merge and then keeps exactly one public label: the label
with the highest individual SAM3 detection score observed for that node.
"""

import math
import re
from typing import Any, Dict, List, Optional


LABEL_SCORE_SCHEMA_VERSION = 1
_LABEL_SEPARATOR_RE = re.compile(r"[^0-9a-zA-Z]+")


def normalize_label(value: Any) -> str:
    """Return the mapper's stable snake-case representation of one label."""
    label = _LABEL_SEPARATOR_RE.sub("_", str(value or "").strip().lower())
    return label.strip("_")


def _legacy_primary_label(value: Any) -> str:
    """Recover V7's first label from an old space-concatenated label string."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    # V7 normalized each detector label with underscores, then appended other
    # labels separated by literal spaces.  The first token is therefore the
    # original node label, not a multi-word phrase that needs joining.
    return normalize_label(raw.split()[0])


def _score(value: Any, *, default: Optional[float] = None) -> float:
    if value is None and default is not None:
        return float(default)
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("SAM3 label score must be a finite number") from exc
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError(
            "SAM3 label score must be finite and inside the closed interval [0, 1]"
        )
    return score


def ensure_label_score_state(
    item: Dict[str, Any],
    *,
    default_score: float = 0.0,
) -> Dict[str, Any]:
    """Migrate one new/V7 memory item to the V8 single-label schema in place."""
    original = str(item.get("label") or item.get("semantic_tag") or "").strip()
    canonical = normalize_label(item.get("canonical_label"))
    if not canonical:
        canonical = _legacy_primary_label(original)
    if not canonical:
        raise ValueError(
            f"memory node {item.get('node_id')!r} has no usable semantic label"
        )

    label_score = _score(item.get("label_score"), default=default_score)
    evidence = item.get("label_evidence")
    if not isinstance(evidence, dict):
        evidence = {}

    clean_evidence: Dict[str, Dict[str, Any]] = {}
    for raw_label, raw_stats in evidence.items():
        label = normalize_label(raw_label)
        if not label or not isinstance(raw_stats, dict):
            continue
        try:
            max_score = _score(raw_stats.get("max_score"), default=0.0)
            score_sum = float(raw_stats.get("score_sum", max_score))
            count = max(0, int(raw_stats.get("count", 0)))
        except (TypeError, ValueError):
            continue
        clean_evidence[label] = {
            "count": count,
            "score_sum": score_sum,
            "max_score": max_score,
            "first_frame": raw_stats.get("first_frame"),
            "last_frame": raw_stats.get("last_frame"),
        }

    current = clean_evidence.setdefault(
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

    item["label"] = canonical
    item["canonical_label"] = canonical
    item["label_score"] = label_score
    item["label_evidence"] = clean_evidence
    # Querying aliases reintroduces the exact semantic redundancy this V8 fixes.
    item["label_aliases"] = []
    item["label_score_schema_version"] = LABEL_SCORE_SCHEMA_VERSION
    if original and original != canonical:
        item["legacy_mixed_label_import"] = original
    return item


def record_label_score(
    item: Dict[str, Any],
    observed_label: Any,
    observed_score: Any,
    *,
    frame_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Record one post-Re-ID SAM3 observation and update the winning label."""
    ensure_label_score_state(item)
    label = normalize_label(observed_label)
    if not label:
        raise ValueError("SAM3 observation has an empty label")
    score = _score(observed_score)

    stats = item["label_evidence"].setdefault(
        label,
        {
            "count": 0,
            "score_sum": 0.0,
            "max_score": 0.0,
            "first_frame": frame_index,
            "last_frame": frame_index,
        },
    )
    stats["count"] = int(stats.get("count", 0)) + 1
    stats["score_sum"] = float(stats.get("score_sum", 0.0)) + score
    stats["max_score"] = max(float(stats.get("max_score", 0.0)), score)
    if stats.get("first_frame") is None:
        stats["first_frame"] = frame_index
    stats["last_frame"] = frame_index

    previous_label = item["label"]
    previous_score = float(item["label_score"])
    changed = score > previous_score
    if changed:
        item["label"] = label
        item["canonical_label"] = label
        item["label_score"] = score
    # A stronger observation of the same label also raises its canonical score.
    elif label == previous_label and score > float(item["label_score"]):
        item["label_score"] = score

    item["label_aliases"] = []
    return {
        "changed": changed,
        "previous_label": previous_label,
        "previous_score": previous_score,
        "observed_label": label,
        "observed_score": score,
        "final_label": item["label"],
        "final_score": float(item["label_score"]),
    }


def initialize_label_score(
    item: Dict[str, Any],
    observed_label: Any,
    observed_score: Any,
    *,
    frame_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Initialize a newly-created node from its founding SAM3 observation."""
    label = normalize_label(observed_label)
    score = _score(observed_score)
    item["label"] = label
    item["canonical_label"] = label
    item["label_score"] = 0.0
    item["label_evidence"] = {}
    item["label_aliases"] = []
    ensure_label_score_state(item)
    return record_label_score(
        item,
        label,
        score,
        frame_index=frame_index,
    )


def retrieval_labels(item: Dict[str, Any]) -> List[str]:
    """Expose only the current winner to retrieval/query code."""
    ensure_label_score_state(item)
    return [item["label"]]
