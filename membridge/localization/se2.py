"""Ground-plane SE(2) in Habitat XZ, matching habitat_camera_to_world yaw."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SENSOR_OFFSET_XYZ = (0.0, 1.2, 0.0)


def wrap_angle(yaw: float) -> float:
    return float((yaw + math.pi) % (2.0 * math.pi) - math.pi)


def yaw_from_xyzw(xyzw: list[float] | tuple[float, ...] | np.ndarray) -> float:
    x, y, z, w = (float(v) for v in np.asarray(xyzw, dtype=np.float64).reshape(-1)[:4])
    return wrap_angle(2.0 * math.atan2(y, w))


def rot_xz(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)


def habitat_camera_to_world(
    agent_position_xyz,
    heading_rad: float,
    *,
    sensor_offset_xyz=SENSOR_OFFSET_XYZ,
) -> np.ndarray:
    position = np.asarray(agent_position_xyz, dtype=np.float64).reshape(3)
    offset = np.asarray(sensor_offset_xyz, dtype=np.float64).reshape(3)
    cosine, sine = math.cos(heading_rad), math.sin(heading_rad)
    rotation = np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = position + rotation @ offset
    return pose


def world_xz_to_local(
    world_x: float,
    world_z: float,
    start_x: float,
    start_z: float,
    start_yaw: float,
) -> tuple[float, float]:
    dx = float(world_x) - float(start_x)
    dz = float(world_z) - float(start_z)
    cosine, sine = math.cos(start_yaw), math.sin(start_yaw)
    return cosine * dx - sine * dz, sine * dx + cosine * dz


def world_agent_to_local(
    world_xyz,
    world_yaw: float,
    start_xyz,
    start_yaw: float,
) -> tuple[float, float, float, float]:
    world = np.asarray(world_xyz, dtype=np.float64).reshape(3)
    start = np.asarray(start_xyz, dtype=np.float64).reshape(3)
    x, z = world_xz_to_local(world[0], world[2], start[0], start[2], start_yaw)
    return x, float(world[1]), z, wrap_angle(float(world_yaw) - float(start_yaw))


@dataclass(frozen=True)
class SE2:
    """Maps source-local XZ into target-local XZ: p_t = R(yaw) @ p_s + t."""

    x: float
    z: float
    yaw: float

    def matrix(self) -> np.ndarray:
        cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
        return np.array(
            [
                [cosine, sine, self.x],
                [-sine, cosine, self.z],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def apply(self, xz: np.ndarray) -> np.ndarray:
        pts = np.asarray(xz, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
        x = cosine * pts[:, 0] + sine * pts[:, 1] + self.x
        z = -sine * pts[:, 0] + cosine * pts[:, 1] + self.z
        out = np.stack([x, z], axis=1)
        return out[0] if np.asarray(xz).ndim == 1 else out

    def inverse(self) -> "SE2":
        cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
        return SE2(
            x=-(cosine * self.x - sine * self.z),
            z=-(sine * self.x + cosine * self.z),
            yaw=wrap_angle(-self.yaw),
        )

    def compose(self, other: "SE2") -> "SE2":
        """self ⊕ other: apply other first, then self."""
        t = self.apply(np.array([other.x, other.z], dtype=np.float64))
        return SE2(x=float(t[0]), z=float(t[1]), yaw=wrap_angle(self.yaw + other.yaw))

    def translation_norm(self) -> float:
        return math.hypot(self.x, self.z)


def pose_se2(x: float, z: float, yaw: float) -> SE2:
    return SE2(x=float(x), z=float(z), yaw=wrap_angle(float(yaw)))


def minus(a: SE2, b: SE2) -> SE2:
    """a ⊖ b = a ⊕ inv(b)."""
    return a.compose(b.inverse())


def gt_ep1_to_ep0(ep0_xyz, ep0_yaw: float, ep1_xyz, ep1_yaw: float) -> SE2:
    """Ground-truth map from EP1-local into EP0-local, from world starts."""
    x, _, z, _ = world_agent_to_local(ep1_xyz, 0.0, ep0_xyz, ep0_yaw)
    return SE2(x=x, z=z, yaw=wrap_angle(float(ep1_yaw) - float(ep0_yaw)))


def se2_error(est: SE2, gt: SE2) -> tuple[float, float]:
    delta = minus(est, gt)
    return delta.translation_norm(), abs(wrap_angle(delta.yaw))
