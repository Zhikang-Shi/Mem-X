"""RGB-D geometry helpers used before instance association."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def pixel_to_world(
    u: float,
    v: float,
    depth_m: float,
    camera_to_world: np.ndarray,
    intrinsics: Mapping[str, float],
) -> list[float] | None:
    if depth_m <= 0.0:
        return None
    x = (float(u) - float(intrinsics["cx"])) * depth_m / float(intrinsics["fx"])
    y = (float(v) - float(intrinsics["cy"])) * depth_m / float(intrinsics["fy"])
    point = np.asarray([x, -y, -depth_m, 1.0], dtype=np.float64)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("camera_to_world must be a finite 4x4 matrix")
    world = pose @ point
    return world[:3].tolist() if np.isfinite(world[:3]).all() else None


def masked_detection_position(
    depth: np.ndarray,
    bbox_xyxy: Any,
    camera_to_world: np.ndarray,
    intrinsics: Mapping[str, float],
    mask: np.ndarray | None = None,
) -> list[float] | None:
    """Project the median masked depth at the mask centroid into world space."""

    image = np.asarray(depth)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError("depth must be HxW")
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox_xyxy[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    valid = crop > 0
    local_mask = None
    if mask is not None:
        full_mask = np.asarray(mask, dtype=bool)
        if full_mask.shape == image.shape:
            local_mask = full_mask[y1:y2, x1:x2]
            valid &= local_mask
    values = crop[valid]
    if not values.size:
        return None
    if local_mask is not None and np.any(local_mask):
        rows, cols = np.where(local_mask)
        u = x1 + float(cols.mean())
        v = y1 + float(rows.mean())
    else:
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    depth_scale = float(intrinsics.get("depth_scale", 1000.0))
    return pixel_to_world(
        u, v, float(np.median(values)) / depth_scale, camera_to_world, intrinsics
    )


def finite_position(value: Any) -> np.ndarray | None:
    try:
        position = np.asarray(value, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    return position if np.isfinite(position).all() else None
