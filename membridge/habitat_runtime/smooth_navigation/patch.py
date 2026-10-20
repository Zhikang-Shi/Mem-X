"""Install a conservative smoothing overlay on the reviewed SG evaluator.

The original controller remains the authority: it plans first on every
cycle. The overlay can replace only an immediate opposite correction with one
forward action that independently passes the same map, navmesh, progress and
anti-reversal checks. Local recovery retains a short prefix of an A* plan, but
validates measured pose and the next action before every execution.
"""

import math
import os

import numpy as np

from .smooth_navigation_controller import (
    HEADING_HOLD_STREAK_KEY,
    OVERRIDE_COUNT_KEY,
    RECOVERY_CACHE_KEY,
    SmoothNavigationConfig,
    anti_flip_is_eligible,
    clear_action_history,
    heading_hold_is_eligible,
    record_action,
    record_heading_hold,
)


_INSTALL_MARKER = "_sg_smooth_navigation_overlay_installed"
_REENTRY_MARKER = "_sg_smooth_navigation_get_action_active"


def _wrap_angle_radians(value):
    return float(math.atan2(math.sin(float(value)), math.cos(float(value))))


def _heading_distance(first, second):
    return abs(_wrap_angle_radians(float(first) - float(second)))


def _runtime_config(evaluator):
    config = getattr(evaluator, "_sg_smooth_navigation_config", None)
    if config is None:
        config = SmoothNavigationConfig.from_environment()
        evaluator._sg_smooth_navigation_config = config
    return config


def _current_heading(evaluator, kwargs):
    supplied = kwargs.get("current_world_heading")
    if supplied is None:
        supplied = getattr(evaluator, "_sg_world_heading_rad", None)
    if supplied is None:
        return None
    try:
        return float(np.asarray(supplied).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None


def _certified_probe_is_safe(
    evaluator,
    target_info,
    progress,
    candidate_cross_track,
    window,
    position,
    endpoint,
):
    trajectory_suffix_start = target_info.get(
        "_sg_trajectory_suffix_start_progress_cells"
    )
    certified_route_phase = bool(
        target_info.get("execution_path_source")
        == "certified_trajectory_then_world_theta_star"
        and trajectory_suffix_start is not None
        and float(progress) + 1e-6 < float(trajectory_suffix_start)
    )
    trajectory_probe_mask = getattr(
        evaluator, "_sg_world_trajectory_probe_mask", None
    )
    if not certified_route_phase or trajectory_probe_mask is None:
        return False
    if not evaluator._sg_float_segment_is_safe(
        trajectory_probe_mask,
        evaluator._sg_window_point(window, position),
        evaluator._sg_window_point(window, endpoint),
    ):
        return False
    probe_limit_cells = float(
        os.environ.get(
            "LIFELONG_SG_TRAJECTORY_PROBE_MAX_CROSS_TRACK_CELLS", "2.0"
        )
    )
    return bool(
        np.isfinite(candidate_cross_track)
        and float(candidate_cross_track) <= probe_limit_cells + 1e-6
    )


def _safe_forward_override(
    evaluator,
    kwargs,
    target_info,
    config,
    maximum_heading_error_deg=None,
    maximum_cross_track_m=None,
):
    """Revalidate the current forward action after the base chose a turn."""

    current_world_xz = evaluator._sg_current_world_xz(kwargs)
    metric_map = evaluator._sg_world_metric_map()
    window = getattr(evaluator, "_sg_world_plan_window", None)
    safe_mask = getattr(evaluator, "_sg_world_safe_mask", None)
    path = target_info.get("_sg_world_path_cells") or []
    current_heading = _current_heading(evaluator, kwargs)
    if (
        current_world_xz is None
        or metric_map is None
        or window is None
        or safe_mask is None
        or len(path) < 2
        or current_heading is None
    ):
        return None

    resolution_m = float(metric_map.spec.resolution_m)
    if not np.isfinite(resolution_m) or resolution_m <= 0:
        return None
    position = (
        np.asarray(current_world_xz, dtype=np.float64).reshape(2)
        / resolution_m
        - 0.5
    )
    reported_progress = float(target_info.get("_sg_path_progress_cells", 0.0))
    progress, _, lookahead, _ = evaluator._sg_world_path_progress_and_lookahead(
        path,
        position,
        reported_progress,
        window,
        safe_mask,
    )
    if lookahead is None:
        return None

    step_cells = (
        float(evaluator.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE)
        / resolution_m
    )
    endpoint, navmesh_predicted = evaluator._sg_predict_forward_endpoint(
        position,
        current_heading,
        step_cells,
        resolution_m,
        evaluator._sg_navmesh_context(),
    )
    actual_step = float(np.linalg.norm(endpoint - position))
    if navmesh_predicted and actual_step < step_cells * 0.5:
        return None

    clearance_safe = evaluator._sg_float_segment_is_safe(
        safe_mask,
        evaluator._sg_window_point(window, position),
        evaluator._sg_window_point(window, endpoint),
    )
    candidate_progress, candidate_cross_track = (
        evaluator._sg_project_position_to_path(path, endpoint, progress)
    )
    if candidate_progress is None or not np.isfinite(candidate_cross_track):
        return None
    if not clearance_safe and not _certified_probe_is_safe(
        evaluator,
        target_info,
        progress,
        candidate_cross_track,
        window,
        position,
        endpoint,
    ):
        return None

    target = np.asarray(
        [
            float(target_info["target_world_grid_x"]),
            float(target_info["target_world_grid_z"]),
        ],
        dtype=np.float64,
    )
    current_goal_distance = float(np.linalg.norm(position - target))
    candidate_goal_distance = float(np.linalg.norm(endpoint - target))
    progress_gain = float(candidate_progress) - float(progress)
    goal_gain = current_goal_distance - candidate_goal_distance
    minimum_advance_cells = max(
        0.25, float(evaluator.sg_nav_stall_eps_m) / resolution_m
    )
    maximum_progress_cross_track_cells = max(
        step_cells, float(evaluator.sg_path_deviation_m) / resolution_m
    )
    if not evaluator._sg_discrete_candidate_advances(
        progress_gain,
        goal_gain,
        candidate_cross_track,
        minimum_advance_cells,
        maximum_progress_cross_track_cells,
    ):
        return None

    previous_forward_origin = target_info.get(
        "_sg_last_forward_world_grid_origin"
    )
    if previous_forward_origin is not None and float(
        np.linalg.norm(
            endpoint - np.asarray(previous_forward_origin, dtype=np.float64)
        )
    ) <= max(0.5, step_cells * 0.25):
        return None

    delta = np.asarray(lookahead, dtype=np.float64) - position
    desired_heading = float(math.atan2(-delta[0], -delta[1]))
    heading_error = _heading_distance(desired_heading, current_heading)
    cross_track_m = float(candidate_cross_track) * resolution_m
    if maximum_heading_error_deg is None:
        maximum_heading_error_deg = config.anti_flip_max_heading_error_deg
    if maximum_cross_track_m is None:
        maximum_cross_track_m = config.anti_flip_max_cross_track_m
    if heading_error > math.radians(maximum_heading_error_deg) + 1e-9:
        return None
    if cross_track_m > maximum_cross_track_m + 1e-9:
        return None

    return {
        "position": tuple(map(float, position)),
        "endpoint": tuple(map(float, endpoint)),
        "heading_error_deg": math.degrees(heading_error),
        "cross_track_m": cross_track_m,
        "progress_gain_cells": progress_gain,
        "goal_gain_cells": goal_gain,
        "navmesh_predicted": bool(navmesh_predicted),
    }


def _expected_after_action(
    evaluator,
    actions,
    action,
    position,
    heading,
    window,
    safe_mask,
):
    position = np.asarray(position, dtype=np.float64).reshape(2)
    heading = float(heading)
    turn_rad = math.radians(
        float(evaluator.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE)
    )
    if int(action) == int(actions.TURN_LEFT):
        return position, _wrap_angle_radians(heading + turn_rad)
    if int(action) == int(actions.TURN_RIGHT):
        return position, _wrap_angle_radians(heading - turn_rad)
    if int(action) != int(actions.MOVE_FORWARD):
        return None

    metric_map = evaluator._sg_world_metric_map()
    if metric_map is None or window is None or safe_mask is None:
        return None
    resolution_m = float(metric_map.spec.resolution_m)
    step_cells = (
        float(evaluator.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE)
        / resolution_m
    )
    endpoint, navmesh_predicted = evaluator._sg_predict_forward_endpoint(
        position,
        heading,
        step_cells,
        resolution_m,
        evaluator._sg_navmesh_context(),
    )
    if navmesh_predicted and float(np.linalg.norm(endpoint - position)) < (
        step_cells * 0.5
    ):
        return None
    if not evaluator._sg_float_segment_is_safe(
        safe_mask,
        evaluator._sg_window_point(window, position),
        evaluator._sg_window_point(window, endpoint),
    ):
        return None
    return np.asarray(endpoint, dtype=np.float64), heading


def _target_key(target):
    return [float(value) for value in np.asarray(target).reshape(2)]


def _cache_fresh_recovery_plan(
    evaluator,
    actions,
    target_info,
    plan,
    position,
    current_heading,
    target,
    window,
    safe_mask,
    config,
):
    action_list = [int(action) for action in plan.get("actions", [])]
    prefix = action_list[: config.recovery_prefix_actions]
    if len(prefix) < 2:
        target_info.pop(RECOVERY_CACHE_KEY, None)
        return
    expected = _expected_after_action(
        evaluator,
        actions,
        prefix[0],
        position,
        current_heading,
        window,
        safe_mask,
    )
    if expected is None:
        target_info.pop(RECOVERY_CACHE_KEY, None)
        return
    expected_position, expected_heading = expected
    metadata = {
        key: value for key, value in plan.items() if key != "actions"
    }
    target_info[RECOVERY_CACHE_KEY] = {
        "actions": prefix,
        "next_index": 1,
        "target_world_grid": _target_key(target),
        "expected_position_grid": list(map(float, expected_position)),
        "expected_heading_rad": float(expected_heading),
        "plan_metadata": metadata,
    }


def _reuse_recovery_plan(
    evaluator,
    actions,
    logger,
    target_info,
    position,
    current_heading,
    target,
    window,
    safe_mask,
    config,
):
    cache = target_info.get(RECOVERY_CACHE_KEY)
    if not cache or not target_info.get("_sg_local_action_recovery_active"):
        return None
    try:
        cached_target = np.asarray(cache["target_world_grid"], dtype=np.float64)
        if not np.allclose(cached_target, np.asarray(target), atol=1e-4):
            raise ValueError("recovery target changed")
        metric_map = evaluator._sg_world_metric_map()
        if metric_map is None:
            raise ValueError("world metric map unavailable")
        resolution_m = float(metric_map.spec.resolution_m)
        position_error_m = float(
            np.linalg.norm(
                np.asarray(position, dtype=np.float64)
                - np.asarray(cache["expected_position_grid"], dtype=np.float64)
            )
            * resolution_m
        )
        heading_error_deg = math.degrees(
            _heading_distance(
                current_heading, float(cache["expected_heading_rad"])
            )
        )
        if position_error_m > config.recovery_pose_tolerance_m + 1e-9:
            raise ValueError(f"pose residual {position_error_m:.3f}m")
        if heading_error_deg > config.recovery_heading_tolerance_deg + 1e-9:
            raise ValueError(f"heading residual {heading_error_deg:.2f}deg")

        action_list = [int(action) for action in cache["actions"]]
        next_index = int(cache["next_index"])
        if next_index >= len(action_list):
            raise ValueError("cached prefix exhausted")
        expected = _expected_after_action(
            evaluator,
            actions,
            action_list[next_index],
            position,
            current_heading,
            window,
            safe_mask,
        )
        if expected is None:
            raise ValueError("next cached action failed live safety validation")
        expected_position, expected_heading = expected
        cache["next_index"] = next_index + 1
        cache["expected_position_grid"] = list(map(float, expected_position))
        cache["expected_heading_rad"] = float(expected_heading)
        reused = dict(cache.get("plan_metadata") or {})
        reused["actions"] = action_list[next_index:]
        target_info["_sg_smooth_recovery_reused_actions"] = int(
            target_info.get("_sg_smooth_recovery_reused_actions", 0)
        ) + 1
        logger.info(
            "[INHerit-SG-SMOOTH] reused validated recovery prefix: "
            f"action={action_list[next_index]}, "
            f"remaining_prefix={len(action_list) - next_index - 1}, "
            f"pose_residual={position_error_m:.3f}m, "
            f"heading_residual={heading_error_deg:.2f}deg"
        )
        return reused
    except Exception as error:  # Never let overlay state break base recovery.
        target_info.pop(RECOVERY_CACHE_KEY, None)
        logger.info(
            "[INHerit-SG-SMOOTH] discarded recovery prefix and returned "
            f"to measured-pose A*: {error}"
        )
        return None


def install_smooth_navigation(evaluator_class, actions, logger):
    """Patch one evaluator class in-place; safe to call more than once."""

    if getattr(evaluator_class, _INSTALL_MARKER, False):
        return evaluator_class

    original_init = evaluator_class.__init__
    original_get_action = evaluator_class._get_sg_world_path_action
    original_plan_recovery = evaluator_class._sg_plan_local_action_sequence

    def smooth_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        config = SmoothNavigationConfig.from_environment()
        self._sg_smooth_navigation_config = config
        original_lookahead = float(getattr(self, "sg_path_lookahead_m", 1.0))
        original_turn_cost = float(
            getattr(self, "sg_path_turn_cost_cells", 0.25)
        )
        if config.enabled and config.heading_hold_enabled:
            # A farther collision-checked lookahead reduces local heading
            # noise.  A larger turn cost adds hysteresis in the base lattice
            # scorer, while every selected forward candidate still has to be
            # safe and advancing according to the reviewed controller.
            self.sg_path_lookahead_m = max(
                original_lookahead, config.heading_hold_lookahead_m
            )
            self.sg_path_turn_cost_cells = max(
                original_turn_cost, config.heading_hold_turn_cost_cells
            )
        turn_angle = float(self.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE)
        logger.info(
            "[INHerit-SG-SMOOTH] overlay initialized: "
            f"enabled={config.enabled}, turn_angle={turn_angle:.1f}deg, "
            f"anti_flip={config.anti_flip_enabled}, "
            f"heading_hold={config.heading_hold_enabled}, "
            f"lookahead={original_lookahead:.2f}->"
            f"{float(getattr(self, 'sg_path_lookahead_m', original_lookahead)):.2f}m, "
            f"turn_cost={original_turn_cost:.2f}->"
            f"{float(getattr(self, 'sg_path_turn_cost_cells', original_turn_cost)):.2f}cells, "
            f"recovery_prefix={config.recovery_prefix_actions}"
        )

    def smooth_plan_recovery(
        self,
        position,
        current_heading,
        target,
        window,
        safe_mask,
        target_info,
    ):
        config = _runtime_config(self)
        if config.enabled and config.recovery_plan_enabled:
            reused = _reuse_recovery_plan(
                self,
                actions,
                logger,
                target_info,
                position,
                current_heading,
                target,
                window,
                safe_mask,
                config,
            )
            if reused is not None:
                return reused
        plan = original_plan_recovery(
            self,
            position,
            current_heading,
            target,
            window,
            safe_mask,
            target_info,
        )
        if config.enabled and config.recovery_plan_enabled and plan is not None:
            try:
                _cache_fresh_recovery_plan(
                    self,
                    actions,
                    target_info,
                    plan,
                    position,
                    current_heading,
                    target,
                    window,
                    safe_mask,
                    config,
                )
            except Exception as error:  # Retain the reviewed plan on failure.
                target_info.pop(RECOVERY_CACHE_KEY, None)
                logger.warning(
                    "[INHerit-SG-SMOOTH] recovery cache disabled for this "
                    f"plan; original plan retained: {error}"
                )
        return plan

    def smooth_get_action(self, **kwargs):
        if getattr(self, _REENTRY_MARKER, False):
            return original_get_action(self, **kwargs)
        setattr(self, _REENTRY_MARKER, True)
        try:
            original_action = original_get_action(self, **kwargs)
        finally:
            setattr(self, _REENTRY_MARKER, False)

        config = _runtime_config(self)
        target_info = kwargs.get("target_info")
        if target_info is None or not config.enabled:
            return original_action

        if target_info.get("_sg_local_action_recovery_active"):
            clear_action_history(target_info)
            return original_action
        target_info.pop(RECOVERY_CACHE_KEY, None)

        final_action = original_action
        anti_flip_eligible = bool(
            original_action is not None
            and anti_flip_is_eligible(
                target_info,
                original_action,
                actions.MOVE_FORWARD,
                actions.TURN_LEFT,
                actions.TURN_RIGHT,
                config,
            )
        )
        heading_hold_eligible = bool(
            original_action is not None
            and heading_hold_is_eligible(
                target_info,
                original_action,
                actions.TURN_LEFT,
                actions.TURN_RIGHT,
                config,
            )
        )
        forward = None
        override_kind = None
        if anti_flip_eligible or heading_hold_eligible:
            attempts = []
            if anti_flip_eligible:
                attempts.append(
                    (
                        "anti_flip",
                        config.anti_flip_max_heading_error_deg,
                        config.anti_flip_max_cross_track_m,
                    )
                )
            if heading_hold_eligible:
                attempts.append(
                    (
                        "heading_hold",
                        config.heading_hold_max_heading_error_deg,
                        config.heading_hold_max_cross_track_m,
                    )
                )
            for kind, heading_limit, cross_track_limit in attempts:
                try:
                    forward = _safe_forward_override(
                        self,
                        kwargs,
                        target_info,
                        config,
                        maximum_heading_error_deg=heading_limit,
                        maximum_cross_track_m=cross_track_limit,
                    )
                except Exception as error:  # Fail closed to reviewed action.
                    forward = None
                    logger.warning(
                        "[INHerit-SG-SMOOTH] forward hysteresis validation "
                        f"failed ({kind}); original action retained: {error}"
                    )
                if forward is not None:
                    override_kind = kind
                    break
            if forward is not None:
                target_info["_sg_pending_forward_world_grid_position"] = (
                    forward["position"]
                )
                target_info["_sg_pending_forward_world_grid_endpoint"] = (
                    forward["endpoint"]
                )
                final_action = int(actions.MOVE_FORWARD)
                if override_kind == "anti_flip":
                    target_info[OVERRIDE_COUNT_KEY] = int(
                        target_info.get(OVERRIDE_COUNT_KEY, 0)
                    ) + 1
                    logger.info(
                        "[INHerit-SG-SMOOTH] replaced opposite turn with "
                        "validated forward: "
                        f"original_action={int(original_action)}, "
                        f"heading_error={forward['heading_error_deg']:.2f}deg, "
                        f"cross_track={forward['cross_track_m']:.3f}m, "
                        f"progress_gain={forward['progress_gain_cells']:.2f}cells"
                    )
                else:
                    record_heading_hold(target_info)
                    logger.info(
                        "[INHerit-SG-SMOOTH-V2] held current heading with "
                        "validated forward: "
                        f"original_action={int(original_action)}, "
                        f"heading_error={forward['heading_error_deg']:.2f}deg, "
                        f"cross_track={forward['cross_track_m']:.3f}m, "
                        f"progress_gain={forward['progress_gain_cells']:.2f}cells, "
                        "hold_streak="
                        f"{target_info.get('_sg_smooth_heading_hold_streak', 0)}"
                    )

        if (
            final_action is not None
            and int(final_action) == int(actions.MOVE_FORWARD)
            and override_kind is None
        ):
            # The base controller agrees with the current heading again, so a
            # later disagreement starts a fresh, independently bounded hold.
            target_info[HEADING_HOLD_STREAK_KEY] = 0
        record_action(
            target_info,
            final_action,
            actions.MOVE_FORWARD,
            actions.TURN_LEFT,
            actions.TURN_RIGHT,
        )
        return final_action

    evaluator_class.__init__ = smooth_init
    evaluator_class._sg_plan_local_action_sequence = smooth_plan_recovery
    evaluator_class._get_sg_world_path_action = smooth_get_action
    setattr(evaluator_class, _INSTALL_MARKER, True)
    return evaluator_class
