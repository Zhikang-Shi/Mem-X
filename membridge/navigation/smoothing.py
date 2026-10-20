"""Fail-closed rounding of a collision-certified polyline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass(frozen=True)
class PathSmoothingConfig:
    enabled: bool = True
    corner_radius_m: float = 0.60
    minimum_trim_m: float = 0.15
    trim_fraction: float = 0.35
    sample_spacing_m: float = 0.10
    minimum_turn_deg: float = 12.0
    maximum_turn_deg: float = 150.0
    maximum_sample_turn_deg: float = 30.0
    maximum_length_ratio: float = 1.01
    maximum_points: int = 4096


def path_length(path: Iterable[Point]) -> float:
    points = np.asarray(list(path), dtype=float)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) >= 2 else 0.0


def turn_angles_deg(path: Iterable[Point]) -> list[float]:
    points, result = np.asarray(list(path), dtype=float), []
    for first, middle, last in zip(points[:-2], points[1:-1], points[2:]):
        incoming, outgoing = middle - first, last - middle
        if np.linalg.norm(incoming) > 1e-9 and np.linalg.norm(outgoing) > 1e-9:
            result.append(abs(math.degrees(math.atan2(incoming[0] * outgoing[1] - incoming[1] * outgoing[0], np.dot(incoming, outgoing)))))
    return result


def smooth_safe_path(
    path: Iterable[Point], *, resolution_m: float,
    segment_is_safe: Callable[[Point, Point], bool], config: PathSmoothingConfig,
) -> dict:
    original = [tuple(map(float, point)) for point in path]
    original_length = path_length(original)
    metadata = {
        "enabled": config.enabled, "applied": False,
        "reason": "disabled" if not config.enabled else "not_attempted",
        "original_waypoints": len(original), "smoothed_waypoints": len(original),
        "smoothed_corners": 0, "original_length_cells": original_length,
        "smoothed_length_cells": original_length, "length_ratio": 1.0,
    }
    if not config.enabled or len(original) < 3 or not math.isfinite(resolution_m) or resolution_m <= 0:
        metadata["reason"] = "disabled" if not config.enabled else "invalid_or_short_path"
        return {"path": original, "metadata": metadata}
    points, output, corners = np.asarray(original, dtype=float), [original[0]], 0
    radius, minimum, spacing = config.corner_radius_m / resolution_m, config.minimum_trim_m / resolution_m, config.sample_spacing_m / resolution_m

    def all_safe(values) -> bool:
        try:
            return all(segment_is_safe(tuple(a), tuple(b)) for a, b in zip(values[:-1], values[1:]))
        except Exception:
            return False

    for index in range(1, len(points) - 1):
        previous, corner, following = points[index - 1:index + 2]
        incoming, outgoing = corner - previous, following - corner
        lengths = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        if min(lengths) <= 1e-9:
            output.append(tuple(corner)); continue
        incoming, outgoing = incoming / lengths[0], outgoing / lengths[1]
        turn = abs(math.degrees(math.atan2(incoming[0] * outgoing[1] - incoming[1] * outgoing[0], np.dot(incoming, outgoing))))
        trim = min(radius, config.trim_fraction * lengths[0], config.trim_fraction * lengths[1])
        if not config.minimum_turn_deg <= turn <= config.maximum_turn_deg or trim < minimum:
            output.append(tuple(corner)); continue
        accepted = None
        for scale in (1.0, 0.8, 0.6, 0.4, 0.25):
            amount = trim * scale
            if amount < minimum: continue
            entry, exit_point = corner - amount * incoming, corner + amount * outgoing
            samples = max(3, int(math.ceil((np.linalg.norm(corner-entry) + np.linalg.norm(exit_point-corner)) / spacing)) + 1)
            curve = [(1-t)**2 * entry + 2*(1-t)*t*corner + t*t*exit_point for t in np.linspace(0, 1, samples)]
            local = [np.asarray(output[-1]), *curve, following]
            if all_safe(local) and max(turn_angles_deg(map(tuple, local)), default=0) <= config.maximum_sample_turn_deg + 1e-6:
                accepted = curve; break
        if accepted is None:
            output.append(tuple(corner)); continue
        for point in accepted:
            value = tuple(map(float, point))
            if np.linalg.norm(np.asarray(output[-1]) - point) > 1e-8: output.append(value)
        corners += 1
        if len(output) > config.maximum_points:
            metadata["reason"] = "maximum_points_exceeded"
            return {"path": original, "metadata": metadata}
    if np.linalg.norm(np.asarray(output[-1]) - points[-1]) > 1e-8:
        output.append(original[-1])
    length = path_length(output)
    safe = all_safe(output)
    ratio = length / original_length if original_length > 1e-9 else 1.0
    metadata.update(smoothed_waypoints=len(output), smoothed_corners=corners, smoothed_length_cells=length, length_ratio=ratio)
    if corners and safe and output[0] == original[0] and output[-1] == original[-1] and ratio <= config.maximum_length_ratio + 1e-9:
        metadata.update(applied=True, reason="safe_quadratic_corner_rounding")
        return {"path": output, "metadata": metadata}
    metadata["reason"] = "no_safe_eligible_corner" if not corners else "final_audit_failed"
    return {"path": original, "metadata": metadata}
