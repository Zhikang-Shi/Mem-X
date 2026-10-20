"""Habitat-compatible pose conversion without importing Habitat packages."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def quaternion_xyzw(rotation: Any) -> tuple[float, float, float, float]:
    """Normalize common Habitat/Magnum/numpy-quaternion representations."""

    if all(hasattr(rotation, name) for name in ("x", "y", "z", "w")):
        return tuple(float(getattr(rotation, name)) for name in ("x", "y", "z", "w"))
    if hasattr(rotation, "vector") and hasattr(rotation, "scalar"):
        vector = np.asarray(rotation.vector, dtype=float).reshape(3)
        return float(vector[0]), float(vector[1]), float(vector[2]), float(rotation.scalar)
    if hasattr(rotation, "imag") and hasattr(rotation, "real"):
        vector = np.asarray(rotation.imag, dtype=float).reshape(3)
        return float(vector[0]), float(vector[1]), float(vector[2]), float(rotation.real)
    values = np.asarray(rotation, dtype=float).reshape(4)
    return tuple(map(float, values))


def heading_from_quaternion_xyzw(rotation_xyzw: Sequence[float]) -> float:
    x, y, z, w = quaternion_xyzw(rotation_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("rotation quaternion must be finite and non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))


@dataclass(frozen=True)
class HabitatPose:
    position_xyz: tuple[float, float, float]
    heading_rad: float

    @classmethod
    def from_values(
        cls, position_xyz: Sequence[float], *, heading_rad: float | None = None,
        rotation_xyzw: Sequence[float] | None = None,
    ) -> "HabitatPose":
        position = np.asarray(position_xyz, dtype=float).reshape(3)
        if not np.isfinite(position).all():
            raise ValueError("agent position must be finite")
        if heading_rad is None:
            if rotation_xyzw is None:
                raise ValueError("heading_rad or rotation_xyzw is required")
            heading_rad = heading_from_quaternion_xyzw(rotation_xyzw)
        if not math.isfinite(float(heading_rad)):
            raise ValueError("agent heading must be finite")
        return cls(tuple(map(float, position)), float(heading_rad))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HabitatPose":
        position = raw.get("position_xyz", raw.get("position"))
        return cls.from_values(
            position,
            heading_rad=raw.get("heading_rad"),
            rotation_xyzw=raw.get("rotation_xyzw", raw.get("rotation")),
        )


def record_terminal_pose(
    recorder: Any, pose: HabitatPose, *, frame_idx: int, step: int,
    episode_id: str, floor_id: int = 0,
) -> dict[str, int]:
    result = recorder.record_traversed_pose(
        agent_position_xyz=pose.position_xyz, heading_rad=pose.heading_rad,
        frame_idx=int(frame_idx), step=int(step), episode_id=str(episode_id),
        floor_id=int(floor_id),
    )
    return {**result, "terminal_pose_recorded": 1}
