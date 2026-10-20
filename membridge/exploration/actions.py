"""Turn/forward commands toward an episode-local XZ waypoint."""

from __future__ import annotations

import math

from se2 import wrap_angle

TURN_RAD = math.radians(30.0)
FORWARD_ALIGN_RAD = math.radians(20.0)


def heading_to_point(ax: float, az: float, tx: float, tz: float) -> float:
    """Habitat yaw whose forward axis (-sin, -cos) points at the target."""
    return wrap_angle(math.atan2(-(tx - ax), -(tz - az)))


def command_toward(
    ax: float,
    az: float,
    yaw: float,
    tx: float,
    tz: float,
    *,
    arrive_m: float = 0.30,
) -> str:
    if math.hypot(tx - ax, tz - az) <= arrive_m:
        return "stop"
    desired = heading_to_point(ax, az, tx, tz)
    delta = wrap_angle(desired - yaw)
    if abs(delta) > FORWARD_ALIGN_RAD:
        return "turn_left" if delta > 0 else "turn_right"
    return "move_forward"
