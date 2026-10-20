"""Pure configuration and action-history helpers for smooth SG navigation.

This module deliberately has no Habitat dependency so its safety policy can be
unit-tested on the host before the runtime overlay is staged in the container.
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path


LAST_TURN_KEY = "_sg_smooth_last_turn_action"
FORWARD_STEPS_KEY = "_sg_smooth_forward_steps_since_turn"
OVERRIDE_COUNT_KEY = "_sg_smooth_anti_flip_forward_count"
HEADING_HOLD_STREAK_KEY = "_sg_smooth_heading_hold_streak"
HEADING_HOLD_COUNT_KEY = "_sg_smooth_heading_hold_forward_count"
RECOVERY_CACHE_KEY = "_sg_smooth_recovery_plan_cache"


def _read_bool(environment, name, default):
    raw = environment.get(name)
    if raw is None:
        return bool(default)
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _read_float(environment, name, default, minimum):
    try:
        value = float(environment.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value) or value < minimum:
        return float(default)
    return value


def _read_int(environment, name, default, minimum, maximum):
    try:
        value = int(environment.get(name, default))
    except (TypeError, ValueError):
        return int(default)
    if value < minimum or value > maximum:
        return int(default)
    return value


def _read_environment_file(path):
    values = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, TypeError, ValueError):
        return values
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key.startswith("LIFELONG_SG_SMOOTH_"):
            values[key] = value.strip()
    return values


@dataclass(frozen=True)
class SmoothNavigationConfig:
    """Conservative defaults for the 30-degree Habitat action lattice.

    No turn angle is defined here. The overlay always reads TURN_ANGLE from
    the original Habitat configuration, so staging this version cannot
    silently turn a 30-degree experiment into a 15-degree experiment.
    """

    enabled: bool = True
    anti_flip_enabled: bool = True
    anti_flip_max_forward_steps: int = 4
    anti_flip_max_heading_error_deg: float = 25.0
    anti_flip_max_cross_track_m: float = 0.22
    heading_hold_enabled: bool = False
    heading_hold_max_consecutive_forwards: int = 3
    heading_hold_max_heading_error_deg: float = 27.0
    heading_hold_max_cross_track_m: float = 0.28
    heading_hold_lookahead_m: float = 1.50
    heading_hold_turn_cost_cells: float = 0.65
    recovery_plan_enabled: bool = True
    recovery_prefix_actions: int = 6
    recovery_pose_tolerance_m: float = 0.10
    recovery_heading_tolerance_deg: float = 4.0

    @classmethod
    def from_environment(cls, environment=None):
        runtime_environment = (
            dict(os.environ) if environment is None else dict(environment)
        )
        configured_path = runtime_environment.get(
            "LIFELONG_SG_SMOOTH_CONFIG_PATH"
        )
        if configured_path is None:
            configured_path = Path(__file__).with_name("smooth_navigation.env")
        file_environment = _read_environment_file(configured_path)
        file_environment.update(runtime_environment)
        environment = file_environment
        return cls(
            enabled=_read_bool(
                environment, "LIFELONG_SG_SMOOTH_NAV_ENABLE", True
            ),
            anti_flip_enabled=_read_bool(
                environment, "LIFELONG_SG_SMOOTH_ANTI_FLIP_ENABLE", True
            ),
            anti_flip_max_forward_steps=_read_int(
                environment,
                "LIFELONG_SG_SMOOTH_ANTI_FLIP_MAX_FORWARD_STEPS",
                4,
                1,
                32,
            ),
            anti_flip_max_heading_error_deg=_read_float(
                environment,
                "LIFELONG_SG_SMOOTH_ANTI_FLIP_MAX_HEADING_ERROR_DEG",
                25.0,
                0.0,
            ),
            anti_flip_max_cross_track_m=_read_float(
                environment,
                "LIFELONG_SG_SMOOTH_ANTI_FLIP_MAX_CROSS_TRACK_M",
                0.22,
                0.0,
            ),
            heading_hold_enabled=_read_bool(
                environment,
                "LIFELONG_SG_SMOOTH_HEADING_HOLD_ENABLE",
                False,
            ),
            heading_hold_max_consecutive_forwards=_read_int(
                environment,
                "LIFELONG_SG_SMOOTH_HEADING_HOLD_MAX_CONSECUTIVE_FORWARDS",
                3,
                1,
                16,
            ),
            heading_hold_max_heading_error_deg=_read_float(
                environment,
                "LIFELONG_SG_SMOOTH_HEADING_HOLD_MAX_HEADING_ERROR_DEG",
                27.0,
                0.0,
            ),
            heading_hold_max_cross_track_m=_read_float(
                environment,
                "LIFELONG_SG_SMOOTH_HEADING_HOLD_MAX_CROSS_TRACK_M",
                0.28,
                0.0,
            ),
            heading_hold_lookahead_m=_read_float(
                environment,
                "LIFELONG_SG_SMOOTH_HEADING_HOLD_LOOKAHEAD_M",
                1.50,
                0.25,
            ),
            heading_hold_turn_cost_cells=_read_float(
                environment,
                "LIFELONG_SG_SMOOTH_HEADING_HOLD_TURN_COST_CELLS",
                0.65,
                0.0,
            ),
            recovery_plan_enabled=_read_bool(
                environment,
                "LIFELONG_SG_SMOOTH_RECOVERY_PLAN_ENABLE",
                True,
            ),
            recovery_prefix_actions=_read_int(
                environment,
                "LIFELONG_SG_SMOOTH_RECOVERY_PREFIX_ACTIONS",
                6,
                1,
                32,
            ),
            recovery_pose_tolerance_m=_read_float(
                environment,
                "LIFELONG_SG_SMOOTH_RECOVERY_POSE_TOLERANCE_M",
                0.10,
                0.0,
            ),
            recovery_heading_tolerance_deg=_read_float(
                environment,
                "LIFELONG_SG_SMOOTH_RECOVERY_HEADING_TOLERANCE_DEG",
                4.0,
                0.0,
            ),
        )


def opposite_turn(action, turn_left, turn_right):
    action = int(action)
    if action == int(turn_left):
        return int(turn_right)
    if action == int(turn_right):
        return int(turn_left)
    return None


def anti_flip_is_eligible(
    target_info,
    proposed_action,
    move_forward,
    turn_left,
    turn_right,
    config,
):
    """Return true only for a short-gap reversal after real forward motion."""

    del move_forward
    if not config.enabled or not config.anti_flip_enabled:
        return False
    last_turn = target_info.get(LAST_TURN_KEY)
    if last_turn is None:
        return False
    expected_opposite = opposite_turn(last_turn, turn_left, turn_right)
    if expected_opposite is None or int(proposed_action) != expected_opposite:
        return False
    forward_steps = int(target_info.get(FORWARD_STEPS_KEY, 0))
    return 1 <= forward_steps <= config.anti_flip_max_forward_steps


def heading_hold_is_eligible(
    target_info,
    proposed_action,
    turn_left,
    turn_right,
    config,
):
    """Bound how long a safe current heading may beat a proposed turn.

    Geometric and progress safety are checked separately by the runtime patch;
    this pure helper only applies the temporal hysteresis budget.
    """

    if not config.enabled or not config.heading_hold_enabled:
        return False
    if int(proposed_action) not in {int(turn_left), int(turn_right)}:
        return False
    streak = int(target_info.get(HEADING_HOLD_STREAK_KEY, 0))
    return 0 <= streak < config.heading_hold_max_consecutive_forwards


def record_heading_hold(target_info):
    target_info[HEADING_HOLD_STREAK_KEY] = int(
        target_info.get(HEADING_HOLD_STREAK_KEY, 0)
    ) + 1
    target_info[HEADING_HOLD_COUNT_KEY] = int(
        target_info.get(HEADING_HOLD_COUNT_KEY, 0)
    ) + 1


def record_action(
    target_info,
    action,
    move_forward,
    turn_left,
    turn_right,
):
    """Update only the small action history needed by anti-flip hysteresis."""

    if action is None:
        return
    action = int(action)
    if action == int(move_forward):
        if target_info.get(LAST_TURN_KEY) is not None:
            target_info[FORWARD_STEPS_KEY] = int(
                target_info.get(FORWARD_STEPS_KEY, 0)
            ) + 1
        return
    if action in {int(turn_left), int(turn_right)}:
        target_info[LAST_TURN_KEY] = action
        target_info[FORWARD_STEPS_KEY] = 0
        target_info[HEADING_HOLD_STREAK_KEY] = 0


def clear_action_history(target_info):
    target_info.pop(LAST_TURN_KEY, None)
    target_info.pop(FORWARD_STEPS_KEY, None)
    target_info.pop(HEADING_HOLD_STREAK_KEY, None)
