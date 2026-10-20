"""V8 object-instance identity helpers.

The V7 mapper used one mutable whitespace-concatenated ``label`` field for
three different jobs: instance identity, accumulated detector output and
retrieval text.  V8 keeps the node ID as the immutable instance identity,
keeps one canonical retrieval label, and stores detector outputs as explicit
evidence.  Re-ID is decided by measured visual and 3-D support evidence, never
by text similarity alone.

This module deliberately has no model dependencies so its policy can be unit
tested and replayed without Habitat, SAM3 or a GPU.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Iterable

import numpy as np


IDENTITY_SCHEMA_VERSION = 1
GENERIC_LABELS = {"object", "item", "thing", "stuff", "unknown", "entity"}

DEFAULT_POLICY = {
    # Every accepted match needs visual evidence.
    "same_label_visual_threshold": 0.78,
    "conflicting_label_visual_threshold": 0.90,
    # Masked RGB-D supports may be partial, so a small physical gap is allowed.
    "same_label_support_gap_m": 0.18,
    "conflicting_label_support_gap_m": 0.08,
    # The first physical support is immutable and prevents prototype chaining.
    "same_label_anchor_support_gap_m": 0.05,
    "conflicting_label_anchor_support_gap_m": 0.05,
    # Conservative fallback for V7/preloaded nodes without support geometry.
    "same_label_position_fallback_m": 0.55,
    "conflicting_label_position_fallback_m": 0.25,
    # An absolute sanity bound prevents a chain of local matches from drifting.
    "same_label_anchor_distance_m": 2.25,
    "conflicting_label_anchor_distance_m": 1.00,
    # If two instance candidates are effectively tied, create a new node.
    "ambiguous_score_margin": 0.025,
    "max_geometry_prototypes": 12,
    "max_position_samples": 31,
    "feature_ema_new_weight": 0.10,
}


def normalize_label(value: Any) -> str:
    """Return one stable snake-case detector label."""
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    return "_".join(text.split())


def _finite_xyz(value: Any) -> list[float] | None:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array.tolist()


def _legacy_labels(value: Any) -> list[str]:
    # V7 normalized multi-word labels with underscores and then joined distinct
    # labels with literal spaces.  Splitting on whitespace is therefore an
    # unambiguous V7 migration.
    labels = []
    for token in str(value or "").split():
        label = normalize_label(token)
        if label and label not in labels:
            labels.append(label)
    return labels


def _new_evidence(count: int = 1, frame_index: int | None = None) -> dict[str, Any]:
    return {
        "count": int(count),
        "first_frame": frame_index,
        "last_frame": frame_index,
    }


def ensure_identity_state(item: dict[str, Any]) -> dict[str, Any]:
    """Migrate an in-memory V7 object in place and return it."""
    labels = _legacy_labels(item.get("label"))
    canonical = normalize_label(item.get("canonical_label"))
    if not canonical:
        canonical = labels[0] if labels else "object"

    evidence = item.get("label_evidence")
    if not isinstance(evidence, dict):
        evidence = {
            label: _new_evidence()
            for label in (labels or [canonical])
        }
        if len(labels) > 1:
            item["legacy_mixed_label_import"] = True
    else:
        normalized_evidence: dict[str, dict[str, Any]] = {}
        for raw_label, raw_record in evidence.items():
            label = normalize_label(raw_label)
            if not label:
                continue
            record = raw_record if isinstance(raw_record, dict) else {}
            normalized_evidence[label] = {
                "count": max(1, int(record.get("count", 1))),
                "first_frame": record.get("first_frame"),
                "last_frame": record.get("last_frame"),
            }
        evidence = normalized_evidence or {canonical: _new_evidence()}

    if canonical not in evidence:
        evidence[canonical] = _new_evidence()

    item["identity_schema_version"] = IDENTITY_SCHEMA_VERSION
    item["canonical_label"] = canonical
    item["label"] = canonical
    item["label_evidence"] = evidence

    anchor = _finite_xyz(item.get("identity_anchor_pos_3d"))
    if anchor is None:
        anchor = _finite_xyz(item.get("global_pos_3d"))
    item["identity_anchor_pos_3d"] = anchor

    samples = [
        xyz
        for xyz in (
            _finite_xyz(value)
            for value in (item.get("position_samples") or [])
        )
        if xyz is not None
    ]
    if not samples and anchor is not None:
        samples = [anchor]
    item["position_samples"] = samples

    if item.get("identity_feature") is None and item.get("dinov3_feature") is not None:
        item["identity_feature"] = copy.deepcopy(item["dinov3_feature"])

    prototypes = []
    for prototype in item.get("geometry_prototypes") or []:
        normalized = normalize_support(prototype)
        if normalized is not None:
            prototypes.append(normalized)
    item["geometry_prototypes"] = prototypes
    anchor_support = normalize_support(item.get("identity_anchor_support"))
    if anchor_support is None and prototypes:
        anchor_support = copy.deepcopy(prototypes[0])
    item["identity_anchor_support"] = anchor_support
    item["observation_count"] = max(
        int(item.get("observation_count", 0)),
        sum(int(record["count"]) for record in evidence.values()),
        1,
    )
    _refresh_retrieval_aliases(item)
    return item


def _refresh_retrieval_aliases(item: dict[str, Any]) -> None:
    canonical = normalize_label(item.get("canonical_label"))
    evidence = item.get("label_evidence") or {}
    total = max(1, sum(int(record.get("count", 0)) for record in evidence.values()))
    aliases = []
    for label, record in evidence.items():
        label = normalize_label(label)
        count = int(record.get("count", 0))
        if (
            label
            and label != canonical
            and label not in GENERIC_LABELS
            and count >= 2
            and count / total >= 0.20
        ):
            aliases.append(label)
    aliases.sort(key=lambda label: (-int(evidence[label]["count"]), label))
    item["label_aliases"] = aliases


def retrieval_labels(item: dict[str, Any]) -> list[str]:
    ensure_identity_state(item)
    labels = [normalize_label(item.get("canonical_label"))]
    labels.extend(normalize_label(value) for value in item.get("label_aliases") or [])
    return [label for index, label in enumerate(labels) if label and label not in labels[:index]]


def record_label_observation(
    item: dict[str, Any],
    observed_label: Any,
    *,
    frame_index: int | None = None,
) -> str:
    """Add semantic evidence without changing instance identity.

    A challenger needs three observations and a two-vote lead before replacing
    the canonical label.  A non-generic label may immediately replace a generic
    bootstrap label.
    """
    ensure_identity_state(item)
    label = normalize_label(observed_label) or "object"
    evidence = item["label_evidence"]
    if label not in evidence:
        evidence[label] = _new_evidence(0, frame_index)
    record = evidence[label]
    record["count"] = int(record.get("count", 0)) + 1
    if record.get("first_frame") is None:
        record["first_frame"] = frame_index
    record["last_frame"] = frame_index

    incumbent = normalize_label(item["canonical_label"])
    incumbent_count = int(evidence[incumbent]["count"])
    challenger = min(
        evidence,
        key=lambda value: (-int(evidence[value]["count"]), value),
    )
    challenger_count = int(evidence[challenger]["count"])
    if incumbent in GENERIC_LABELS and challenger not in GENERIC_LABELS:
        incumbent = challenger
    elif (
        challenger != incumbent
        and challenger_count >= 3
        and challenger_count >= incumbent_count + 2
    ):
        incumbent = challenger

    item["canonical_label"] = incumbent
    item["label"] = incumbent
    item["observation_count"] = int(item.get("observation_count", 0)) + 1
    _refresh_retrieval_aliases(item)
    return incumbent


def normalize_support(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    minimum = _finite_xyz(value.get("min_xyz"))
    maximum = _finite_xyz(value.get("max_xyz"))
    center = _finite_xyz(value.get("center_xyz"))
    if minimum is None or maximum is None:
        return None
    low = np.minimum(minimum, maximum)
    high = np.maximum(minimum, maximum)
    if center is None:
        center = ((low + high) / 2.0).tolist()
    return {
        "min_xyz": low.tolist(),
        "max_xyz": high.tolist(),
        "center_xyz": center,
        "point_count": max(1, int(value.get("point_count", 1))),
    }


def build_depth_support(
    depth_crop: np.ndarray,
    mask_crop: np.ndarray | None,
    *,
    x_offset: int,
    y_offset: int,
    pose: np.ndarray,
    intrinsics: dict[str, float],
    max_samples: int = 4096,
) -> dict[str, Any] | None:
    """Build a robust world-space AABB from one masked RGB-D detection."""
    if depth_crop is None or pose is None or np.asarray(depth_crop).size == 0:
        return None
    depth = np.asarray(depth_crop)
    valid = depth > 0
    if mask_crop is not None and np.asarray(mask_crop).shape == depth.shape:
        valid &= np.asarray(mask_crop, dtype=bool)
    ys, xs = np.where(valid)
    if ys.size < 8:
        return None

    raw_depth = depth[ys, xs].astype(np.float64)
    depth_low, depth_high = np.percentile(raw_depth, [2.0, 98.0])
    inlier = (raw_depth >= depth_low) & (raw_depth <= depth_high)
    ys, xs, raw_depth = ys[inlier], xs[inlier], raw_depth[inlier]
    if ys.size < 8:
        return None
    if ys.size > max_samples:
        indices = np.linspace(0, ys.size - 1, max_samples, dtype=np.int64)
        ys, xs, raw_depth = ys[indices], xs[indices], raw_depth[indices]

    scale = float(intrinsics.get("depth_scale", 1000.0))
    z = raw_depth / scale
    u = xs.astype(np.float64) + float(x_offset)
    v = ys.astype(np.float64) + float(y_offset)
    x = (u - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (v - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    camera_habitat = np.stack(
        [x, -y, -z, np.ones_like(z)],
        axis=1,
    )
    world = (np.asarray(pose, dtype=np.float64) @ camera_habitat.T).T[:, :3]
    world = world[np.all(np.isfinite(world), axis=1)]
    if world.shape[0] < 8:
        return None

    low = np.percentile(world, 2.0, axis=0)
    high = np.percentile(world, 98.0, axis=0)
    center = np.median(world, axis=0)
    return normalize_support(
        {
            "min_xyz": low.tolist(),
            "max_xyz": high.tolist(),
            "center_xyz": center.tolist(),
            "point_count": int(world.shape[0]),
        }
    )


def support_gap_m(left: Any, right: Any) -> float | None:
    left_n = normalize_support(left)
    right_n = normalize_support(right)
    if left_n is None or right_n is None:
        return None
    left_low = np.asarray(left_n["min_xyz"], dtype=np.float64)
    left_high = np.asarray(left_n["max_xyz"], dtype=np.float64)
    right_low = np.asarray(right_n["min_xyz"], dtype=np.float64)
    right_high = np.asarray(right_n["max_xyz"], dtype=np.float64)
    separation = np.maximum(np.maximum(left_low - right_high, right_low - left_high), 0.0)
    return float(np.linalg.norm(separation))


def _distance(left: Any, right: Any) -> float | None:
    left_xyz = _finite_xyz(left)
    right_xyz = _finite_xyz(right)
    if left_xyz is None or right_xyz is None:
        return None
    return float(
        np.linalg.norm(
            np.asarray(left_xyz, dtype=np.float64)
            - np.asarray(right_xyz, dtype=np.float64)
        )
    )


def evaluate_match(
    item: dict[str, Any],
    *,
    observed_label: Any,
    visual_similarity: float,
    position_xyz: Any,
    support: Any = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a fail-closed V8 Re-ID decision for one observation/node pair."""
    ensure_identity_state(item)
    cfg = dict(DEFAULT_POLICY)
    cfg.update(policy or {})

    visual = float(visual_similarity)
    compatible = normalize_label(observed_label) in set(retrieval_labels(item))
    mode = "same_label" if compatible else "conflicting_label"
    visual_threshold = float(cfg[f"{mode}_visual_threshold"])
    if not math.isfinite(visual) or visual < visual_threshold:
        return {
            "accepted": False,
            "reason": "visual_below_threshold",
            "label_mode": mode,
            "visual_similarity": visual,
            "visual_threshold": visual_threshold,
        }

    anchor_distance = _distance(position_xyz, item.get("identity_anchor_pos_3d"))
    anchor_support_gap = support_gap_m(
        support,
        item.get("identity_anchor_support"),
    )
    if anchor_support_gap is not None:
        anchor_limit = float(cfg[f"{mode}_anchor_support_gap_m"])
        anchor_source = "masked_rgbd_anchor_support"
        anchor_value = anchor_support_gap
    else:
        anchor_limit = float(cfg[f"{mode}_anchor_distance_m"])
        anchor_source = "position_fallback"
        anchor_value = anchor_distance
    if anchor_value is not None and anchor_value > anchor_limit:
        return {
            "accepted": False,
            "reason": "outside_identity_anchor",
            "label_mode": mode,
            "visual_similarity": visual,
            "anchor_source": anchor_source,
            "anchor_support_gap_m": anchor_support_gap,
            "anchor_distance_m": anchor_distance,
            "anchor_limit_m": anchor_limit,
        }

    support_gaps = [
        gap
        for gap in (
            support_gap_m(support, prototype)
            for prototype in item.get("geometry_prototypes") or []
        )
        if gap is not None
    ]
    geometry_gap = min(support_gaps) if support_gaps else None
    if geometry_gap is not None:
        geometry_limit = float(cfg[f"{mode}_support_gap_m"])
        geometry_source = "masked_rgbd_support"
    else:
        geometry_gap = _distance(position_xyz, item.get("global_pos_3d"))
        geometry_limit = float(cfg[f"{mode}_position_fallback_m"])
        geometry_source = "position_fallback"

    if geometry_gap is None or geometry_gap > geometry_limit:
        return {
            "accepted": False,
            "reason": "geometry_outside_threshold",
            "label_mode": mode,
            "visual_similarity": visual,
            "geometry_source": geometry_source,
            "geometry_gap_m": geometry_gap,
            "geometry_limit_m": geometry_limit,
            "anchor_source": anchor_source,
            "anchor_support_gap_m": anchor_support_gap,
            "anchor_distance_m": anchor_distance,
        }

    geometry_quality = max(0.0, 1.0 - geometry_gap / max(geometry_limit, 1.0e-6))
    label_quality = 1.0 if compatible else 0.0
    score = 0.68 * visual + 0.27 * geometry_quality + 0.05 * label_quality
    return {
        "accepted": True,
        "reason": "visual_and_geometry",
        "label_mode": mode,
        "visual_similarity": visual,
        "visual_threshold": visual_threshold,
        "geometry_source": geometry_source,
        "geometry_gap_m": geometry_gap,
        "geometry_limit_m": geometry_limit,
        "anchor_source": anchor_source,
        "anchor_support_gap_m": anchor_support_gap,
        "anchor_distance_m": anchor_distance,
        "score": float(score),
    }


def candidate_is_ambiguous(
    ranked_decisions: Iterable[dict[str, Any]],
    *,
    policy: dict[str, Any] | None = None,
) -> bool:
    cfg = dict(DEFAULT_POLICY)
    cfg.update(policy or {})
    ranked = list(ranked_decisions)
    if len(ranked) < 2:
        return False
    return (
        float(ranked[0]["score"]) - float(ranked[1]["score"])
        < float(cfg["ambiguous_score_margin"])
    )


def record_geometry_observation(
    item: dict[str, Any],
    *,
    position_xyz: Any,
    support: Any = None,
    policy: dict[str, Any] | None = None,
) -> None:
    ensure_identity_state(item)
    cfg = dict(DEFAULT_POLICY)
    cfg.update(policy or {})
    position = _finite_xyz(position_xyz)
    if position is not None:
        samples = item["position_samples"]
        samples.append(position)
        del samples[: max(0, len(samples) - int(cfg["max_position_samples"]))]
        item["global_pos_3d"] = np.median(
            np.asarray(samples, dtype=np.float64),
            axis=0,
        ).tolist()
    normalized = normalize_support(support)
    if normalized is not None:
        if item.get("identity_anchor_support") is None:
            item["identity_anchor_support"] = copy.deepcopy(normalized)
        prototypes = item["geometry_prototypes"]
        prototypes.append(normalized)
        del prototypes[: max(0, len(prototypes) - int(cfg["max_geometry_prototypes"]))]


def update_running_feature(
    item: dict[str, Any],
    observed_feature: Any,
    *,
    policy: dict[str, Any] | None = None,
) -> None:
    ensure_identity_state(item)
    cfg = dict(DEFAULT_POLICY)
    cfg.update(policy or {})
    old = np.asarray(item.get("dinov3_feature"), dtype=np.float64).reshape(-1)
    new = np.asarray(observed_feature, dtype=np.float64).reshape(-1)
    if old.shape != new.shape or old.size == 0:
        item["dinov3_feature"] = new.tolist()
        return
    weight = float(cfg["feature_ema_new_weight"])
    fused = (1.0 - weight) * old + weight * new
    norm = float(np.linalg.norm(fused))
    if norm > 1.0e-12:
        fused /= norm
    item["dinov3_feature"] = fused.tolist()
