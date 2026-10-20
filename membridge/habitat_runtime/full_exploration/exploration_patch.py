"""Install complete goal-directed exploration on memory_module's runtime.

This is the self-contained memory_module-owned adaptation of the reviewed
goal-directed exploration controller.  It deliberately has no runtime import
from the legacy ``SG`` tree.  In formal Level-2 runs, observations continue
through the mutable online builder so the navigation map, trajectory and
semantic scene graph are all updated before M1/M2 publication.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass

import numpy as np

from .frontier_connector import GoalDirectedFrontierSelector


_INSTALL_MARKER = "_sg_goal_directed_exploration_overlay_installed"
IMPLEMENTATION_VERSION = "memory_module_full_goal_directed_exploration_v1"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ExplorationConfig:
    enabled: bool
    action_lattice_recovery: bool
    maximum_actions: int
    maximum_distance_m: float
    connectivity_check_stride: int
    frontier_goal_tolerance_m: float
    scan_heading_tolerance_deg: float
    maximum_planner_candidates: int
    maximum_deferred_scan_turns: int
    connection_only: bool
    old_map_probe_nodes: int
    allow_world_only_gate: bool
    world_only_gate_after_scan_turns: int
    directional_probe: bool
    directional_probe_connectivity_stride: int
    directional_probe_max_actions: int
    directional_probe_max_forward_actions: int
    directional_probe_clearance_cells: float
    directional_probe_min_progress_m: float
    directional_probe_require_navmesh: bool
    no_action_recovery: bool = True
    old_map_fallback: bool = True
    old_map_fallback_max_actions: int = 48
    old_map_fallback_max_forwards: int = 36
    old_map_fallback_switch_actions: int = 8
    old_map_fallback_min_progress_m: float = 0.25

    @classmethod
    def from_environment(cls) -> "ExplorationConfig":
        return cls(
            enabled=_env_bool("LIFELONG_SG_EXPLORATION_ENABLE", False),
            action_lattice_recovery=_env_bool(
                "LIFELONG_SG_EXPLORE_ACTION_LATTICE_RECOVERY", True
            ),
            maximum_actions=max(
                1, int(os.environ.get("LIFELONG_SG_EXPLORE_MAX_ACTIONS", "100"))
            ),
            maximum_distance_m=max(
                0.25,
                float(os.environ.get("LIFELONG_SG_EXPLORE_MAX_DISTANCE_M", "20.0")),
            ),
            connectivity_check_stride=max(
                1,
                int(
                    os.environ.get(
                        "LIFELONG_SG_EXPLORE_CONNECTIVITY_CHECK_STRIDE", "2"
                    )
                ),
            ),
            frontier_goal_tolerance_m=max(
                0.05,
                float(
                    os.environ.get(
                        "LIFELONG_SG_EXPLORE_FRONTIER_TOLERANCE_M", "0.15"
                    )
                ),
            ),
            scan_heading_tolerance_deg=max(
                1.0,
                float(
                    os.environ.get(
                        "LIFELONG_SG_EXPLORE_SCAN_HEADING_TOLERANCE_DEG", "12"
                    )
                ),
            ),
            maximum_planner_candidates=max(
                1,
                int(
                    os.environ.get(
                        "LIFELONG_SG_EXPLORE_MAX_PLANNER_CANDIDATES", "8"
                    )
                ),
            ),
            maximum_deferred_scan_turns=max(
                1,
                int(
                    os.environ.get(
                        "LIFELONG_SG_EXPLORE_MAX_DEFERRED_SCAN_TURNS", "3"
                    )
                ),
            ),
            connection_only=_env_bool(
                "LIFELONG_SG_EXPLORE_CONNECTION_ONLY", False
            ),
            old_map_probe_nodes=max(
                1,
                int(
                    os.environ.get(
                        "LIFELONG_SG_EXPLORE_OLD_MAP_PROBE_NODES", "8"
                    )
                ),
            ),
            allow_world_only_gate=_env_bool(
                "LIFELONG_SG_EXPLORE_ALLOW_WORLD_ONLY_GATE", False
            ),
            world_only_gate_after_scan_turns=max(
                0,
                int(
                    os.environ.get(
                        "LIFELONG_SG_EXPLORE_WORLD_ONLY_GATE_AFTER_SCAN_TURNS",
                        "11",
                    )
                ),
            ),
            directional_probe=_env_bool(
                "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE", False
            ),
            directional_probe_connectivity_stride=max(
                1,
                int(
                    os.environ.get(
                        "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE_CONNECTIVITY_STRIDE",
                        "4",
                    )
                ),
            ),
            directional_probe_max_actions=max(
                1,
                int(
                    os.environ.get(
                        "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE_MAX_ACTIONS",
                        "24",
                    )
                ),
            ),
            directional_probe_max_forward_actions=max(
                1,
                int(
                    os.environ.get(
                        "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE_MAX_FORWARDS",
                        "8",
                    )
                ),
            ),
            directional_probe_clearance_cells=max(
                0.0,
                float(
                    os.environ.get(
                        "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE_CLEARANCE_CELLS",
                        "1.0",
                    )
                ),
            ),
            directional_probe_min_progress_m=float(
                os.environ.get(
                    "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE_MIN_PROGRESS_M",
                    "-0.26",
                )
            ),
            directional_probe_require_navmesh=_env_bool(
                "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE_REQUIRE_NAVMESH", True
            ),
            no_action_recovery=_env_bool(
                "LIFELONG_SG_EXPLORE_NO_ACTION_RECOVERY", True
            ),
            old_map_fallback=_env_bool('LIFELONG_SG_EXPLORE_OLD_MAP_FALLBACK', True),
            old_map_fallback_max_actions=max(1, int(os.environ.get('LIFELONG_SG_EXPLORE_OLD_MAP_MAX_ACTIONS','48'))),
            old_map_fallback_max_forwards=max(1, int(os.environ.get('LIFELONG_SG_EXPLORE_OLD_MAP_MAX_FORWARDS','36'))),
            old_map_fallback_switch_actions=max(1, int(os.environ.get('LIFELONG_SG_EXPLORE_OLD_MAP_SWITCH_ACTIONS','8'))),
            old_map_fallback_min_progress_m=max(0., float(os.environ.get('LIFELONG_SG_EXPLORE_OLD_MAP_MIN_PROGRESS_M','.25'))),
        )


def _runtime_config(evaluator) -> ExplorationConfig:
    config = getattr(evaluator, "_sg_exploration_config", None)
    if config is None:
        config = ExplorationConfig.from_environment()
        evaluator._sg_exploration_config = config
    return config


def _runtime_enabled(evaluator) -> bool:
    adapter = getattr(evaluator, "inherit_sg", None)
    return bool(
        _runtime_config(evaluator).enabled
        and adapter is not None
        and getattr(adapter, "enabled", False)
        and not getattr(adapter, "failed", False)
        and getattr(adapter, "_episode_active", False)
        and getattr(adapter, "world_memory", None) is not None
        and getattr(adapter, "pipeline", None) is not None
        and not (
            getattr(adapter, "preload_query_only", False)
            and getattr(adapter, "_preloaded_session", False)
            and not getattr(adapter, "preload_world_map_only", False)
        )
    )


def _failure_reasons(payload) -> set[str]:
    reasons: set[str] = set()
    if isinstance(payload, dict):
        reason = payload.get("reason")
        if reason:
            reasons.add(str(reason))
        for value in payload.values():
            if isinstance(value, (dict, list, tuple)):
                reasons.update(_failure_reasons(value))
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            reasons.update(_failure_reasons(value))
    return reasons


def _is_connectivity_failure(payload) -> bool:
    reasons = _failure_reasons(payload)
    direct = {
        "warm_start_not_free_in_world_memory",
        "warm_start_not_clearance_safe_in_world_memory",
        "no_world_reachable_target_within_snap_radius",
        "world_theta_star_failed",
        "start_not_connected_to_safe_space",
        "no_reachable_target_within_snap_radius",
        "theta_star_failed",
    }
    if reasons.intersection(direct):
        return True
    # The V8 wrapper summarizes its per-approach failures at the top level.
    # Treat that summary as geometric only when nested planner failures exist.
    return bool(
        "v8_post_hit_no_memory_reachable_approach" in reasons
        and len(reasons.intersection(direct)) > 0
    )


def _is_runtime_connectivity_abort(reason) -> bool:
    """Return true only when an online V1 abort proves map disconnection.

    Runtime reasons are flattened strings such as
    ``cross_track_0.370m_warm_start_not_free_in_world_memory``.  A plain
    cross-track, stall, or collision failure is not sufficient evidence for
    exploration; one of the map-connectivity certificates must be present.
    """

    text = str(reason or "")
    markers = (
        "warm_start_not_free_in_world_memory",
        "warm_start_not_clearance_safe_in_world_memory",
        "no_world_reachable_target_within_snap_radius",
        "world_theta_star_failed",
        "start_not_connected_to_safe_space",
        "no_reachable_target_within_snap_radius",
        "theta_star_failed",
    )
    return any(marker in text for marker in markers)


def _is_runtime_action_lattice_abort(reason) -> bool:
    """Identify V1's exhausted discrete-action reachability certificate.

    This is deliberately narrower than a generic stall, cross-track event, or
    collision.  V1 emits this reason only after rebuilding the world Theta*
    path and rejecting every bounded local recovery target in the 30-degree
    action lattice.  A caller must still find a safe, reachable frontier before
    treating it as recoverable exploration.
    """

    text = str(reason or "")
    return "local_action_lattice_unreachable_after_" in text


def _can_resume_original_target(target_info) -> bool:
    """Action-lattice recovery must first reach and scan its frontier."""

    return not bool(
        target_info.get("approach_exploration_require_frontier_scan_before_resume")
    )


def _candidate_goal_points(candidate, route_planning_anchors=()) -> list[list[float]]:
    points: list[list[float]] = []

    def append_xz(raw, indices=(0, 2)):
        try:
            values = np.asarray(raw, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return
        if len(values) <= max(indices):
            return
        point = values[list(indices)]
        if not np.isfinite(point).all():
            return
        item = [float(point[0]), float(point[1])]
        if not any(np.linalg.norm(np.asarray(item) - existing) <= 1e-6 for existing in points):
            points.append(item)

    for point in route_planning_anchors or ():
        append_xz(point, indices=(0, 1))
    append_xz((candidate.get("waypoint") or {}).get("position"))
    append_xz((candidate.get("best_observation") or {}).get("agent_position"))
    for waypoint in (candidate.get("navigation_waypoint_by_episode") or {}).values():
        append_xz((waypoint or {}).get("position"))
    for observation in (candidate.get("best_observation_by_episode") or {}).values():
        append_xz((observation or {}).get("agent_position"))
    append_xz(candidate.get("position"))
    return points


def _frozen_old_map_goal_points(evaluator) -> list[list[float]]:
    """Only the positions frozen before this episode, never online additions."""
    from membridge.navigation.map_connection import trajectory_points
    return trajectory_points(evaluator, frozen=True)


def _probe_frozen_old_map_connection(evaluator, current_world_xz):
    """Prove current-map connectivity to a frozen acquisition node.

    The probe uses V1's exact world planner on the merged navigation raster.
    It does not query semantic memory.  Testing the nearest few immutable
    trajectory nodes covers separate ep0/ep1 components without paying for a
    planner call to every historical pose on each RGB-D frame.
    """

    current = np.asarray(current_world_xz, dtype=np.float64).reshape(2)
    points = _frozen_old_map_goal_points(evaluator)
    if not points:
        return None
    ordered = sorted(
        points,
        key=lambda point: (
            float(np.linalg.norm(np.asarray(point, dtype=np.float64) - current)),
            point[1],
            point[0],
        ),
    )
    maximum = _runtime_config(evaluator).old_map_probe_nodes
    failures = []
    for rank, point in enumerate(ordered[:maximum], 1):
        plan = evaluator._sg_plan_world_path(
            point,
            current,
            exact=True,
            planning_anchor_world_xz=[point],
            project_target_to_navmesh=False,
            navmesh_terminal_snap_tolerance_m=0.0,
        )
        if plan is not None:
            return {
                "goal_world_xz": list(map(float, point)),
                "goal_rank_by_distance": int(rank),
                "tested_nodes": int(rank),
                "plan": plan,
            }
        failures.append(copy.deepcopy(getattr(evaluator, "_sg_last_theta_failure", None)))
    evaluator._sg_exploration_last_old_map_probe = {
        "status": "not_connected",
        "tested_nodes": min(len(ordered), maximum),
        "failures": failures,
    }
    return None


def _heading_error(desired: float, actual: float) -> float:
    delta = float(desired) - float(actual)
    return float(math.atan2(math.sin(delta), math.cos(delta)))


def _effective_scan_heading_tolerance_deg(
    evaluator, configured_tolerance_deg: float
) -> float:
    """Return a scan tolerance attainable on Habitat's turn lattice.

    The configured tolerance may be smaller than half of ``TURN_ANGLE``.  In
    that case a desired frontier heading between two discrete headings cannot
    ever satisfy the tolerance, so the controller alternates left/right until
    its exploration budget expires.  Half a turn is the tightest tolerance
    that every desired heading can satisfy on the fixed action lattice.
    """

    turn_angle_deg = 0.0
    try:
        turn_angle_deg = abs(
            float(evaluator.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE)
        )
    except (AttributeError, TypeError, ValueError):
        pass
    lattice_floor_deg = 0.5 * turn_angle_deg + 1e-6
    return float(max(float(configured_tolerance_deg), lattice_floor_deg))


def _compact_plan(plan) -> dict:
    if plan is None:
        return {}
    return {
        "target_grid": list(map(int, plan["target_grid"])),
        "target_world_xz": list(map(float, plan["target_world_xz"])),
        "path_waypoints": len(plan.get("path") or []),
        "path_length_cells": float(plan.get("path_length_cells", 0.0)),
        "expanded_states": int(plan.get("expanded_states", 0)),
        "reachable_cells": int(plan.get("reachable_cells", 0)),
        "start_connector": copy.deepcopy(
            plan.get("exploration_start_connector")
        ),
    }


def _start_connector_planning_pose(evaluator, selection, current_world_xz):
    connector = list(selection.start_connector_world_grids or ())
    if len(connector) < 2:
        return np.asarray(current_world_xz, dtype=np.float64).reshape(2)
    metric_map = evaluator._sg_world_metric_map()
    if metric_map is None:
        return np.asarray(current_world_xz, dtype=np.float64).reshape(2)
    return np.asarray(
        metric_map.spec.cell_center_world(*connector[-1]),
        dtype=np.float64,
    )


def _prepend_start_connector(evaluator, plan, selection):
    """Join the live pose to a normal-clearance V1 plan by a bounded egress.

    The suffix has already been planned by unchanged V1 Theta* from the first
    normal 2.5-cell-safe endpoint.  Only the selector's short, known-free
    connector is added to the plan mask.  The caller still runs V1's live
    navmesh-aware action-lattice gate from the real pose before accepting it.
    """

    connector = [
        tuple(map(int, point))
        for point in (selection.start_connector_world_grids or ())
    ]
    if len(connector) < 2:
        return plan
    window = plan.get("window")
    safe_mask = plan.get("safe_mask")
    if window is None or safe_mask is None:
        return None
    safe_mask = np.asarray(safe_mask, dtype=bool).copy()
    connector_local = []
    for global_x, global_z in connector:
        local_x, local_z = window.global_to_local(global_x, global_z)
        if not (
            0 <= local_x < safe_mask.shape[1]
            and 0 <= local_z < safe_mask.shape[0]
        ):
            return None
        safe_mask[local_z, local_x] = True
        connector_local.append((int(local_x), int(local_z)))

    suffix = [tuple(map(float, point)) for point in (plan.get("path") or [])]
    connector_path = [tuple(map(float, point)) for point in connector]
    if suffix and np.linalg.norm(
        np.asarray(connector_path[-1]) - np.asarray(suffix[0])
    ) <= 1e-6:
        combined_path = connector_path + suffix[1:]
    else:
        combined_path = connector_path + suffix
    if len(combined_path) < 2:
        return None
    cumulative = evaluator._sg_path_cumulative(combined_path)
    plan["safe_mask"] = safe_mask
    plan["path"] = combined_path
    raw_suffix = [
        tuple(map(int, point)) for point in (plan.get("raw_path") or [])
    ]
    if raw_suffix and raw_suffix[0] == connector[-1]:
        raw_suffix = raw_suffix[1:]
    plan["raw_path"] = list(connector) + raw_suffix
    plan["path_length_cells"] = float(cumulative[-1])
    plan["exploration_start_connector"] = {
        "status": str(selection.start_connector_status),
        "clearance_cells": float(selection.start_connector_clearance_cells),
        "world_grids": [list(point) for point in connector],
        "length_cells": float(len(connector) - 1),
        "length_m": float(len(connector) - 1)
        * float(evaluator._sg_world_metric_map().spec.resolution_m),
        "normal_clearance_endpoint": list(connector[-1]),
        "action_lattice_validation": "pending",
    }
    return plan


def _commit_v1_plan_buffers(evaluator, plan) -> None:
    """Atomically expose the selected plan's own V1 planning buffers.

    ``_sg_plan_world_path`` updates evaluator-level buffers on every call.  A
    frontier search evaluates several candidates, so those buffers otherwise
    belong to the last candidate tested rather than the candidate ultimately
    selected.  The exploration overlay owns this handoff; the V1 planner and
    controller remain unchanged.
    """

    evaluator._sg_world_plan_window = plan.get("window")
    evaluator._sg_world_safe_mask = plan.get("safe_mask")
    evaluator._sg_world_trajectory_probe_mask = plan.get(
        "_exploration_trajectory_probe_mask"
    )


def _frontier_action_lattice_gate(
    evaluator,
    candidate,
    plan,
    current_world_xz,
    current_heading_rad,
) -> dict | None:
    """Require a frontier path to enter V1's existing 30-degree lattice.

    Theta* proves continuous grid connectivity, but execution starts in
    Habitat's fixed TURN/TURN/FORWARD action space.  Reuse V1's own local-goal
    generator and action-lattice A* against a private target-info copy.  This
    is a read-only admission test: no V1 queue or controller state is changed.
    """

    recovery_targets_fn = getattr(
        evaluator, "_sg_local_action_recovery_targets", None
    )
    lattice_plan_fn = getattr(evaluator, "_sg_plan_local_action_sequence", None)
    if not callable(recovery_targets_fn) or not callable(lattice_plan_fn):
        return None
    window = plan.get("window")
    safe_mask = plan.get("safe_mask")
    if window is None or safe_mask is None or current_heading_rad is None:
        return None
    metric_map = evaluator._sg_world_metric_map()
    if metric_map is None:
        return None
    resolution_m = float(metric_map.spec.resolution_m)
    position_grid = (
        np.asarray(current_world_xz, dtype=np.float64).reshape(2) / resolution_m
        - 0.5
    )
    target_grid = np.asarray(plan["target_grid"], dtype=np.float64).reshape(2)
    try:
        gate_info = copy.deepcopy(candidate)
    except Exception:
        gate_info = dict(candidate or {})
    _clear_path_runtime(gate_info)
    gate_info.update(
        {
            "target_world_grid_x": int(plan["target_grid"][0]),
            "target_world_grid_z": int(plan["target_grid"][1]),
            "_sg_world_path_cells": list(plan.get("path") or []),
            "_sg_path_progress_cells": 0.0,
            "execution_path_source": "goal_directed_frontier_theta_star",
        }
    )
    recovery_targets = recovery_targets_fn(
        gate_info, position_grid, target_grid
    )
    rejected = []
    for recovery_target in recovery_targets:
        lattice_plan = lattice_plan_fn(
            position_grid,
            float(current_heading_rad),
            recovery_target["target"],
            window,
            safe_mask,
            gate_info,
        )
        if lattice_plan is not None:
            actions = list(lattice_plan.get("actions") or [])
            return {
                "status": "v1_action_lattice_reachable",
                "goal_kind": str(recovery_target.get("kind")),
                "goal_world_grid": list(
                    map(float, recovery_target["target"])
                ),
                "path_progress_cells": recovery_target.get(
                    "path_progress_cells"
                ),
                "action_count": len(actions),
                "first_action": int(actions[0]) if actions else None,
                "expanded_states": int(
                    lattice_plan.get("expanded_states", 0)
                ),
            }
        rejected.append(
            {
                "goal_kind": str(recovery_target.get("kind")),
                "goal_world_grid": list(
                    map(float, recovery_target["target"])
                ),
                "status": str(
                    gate_info.get("_sg_local_action_status", "unreachable")
                ),
            }
        )
    return None


def _select_frontier_plan(
    evaluator,
    candidate,
    current_world_xz,
    route_planning_anchors,
    logger,
):
    selector = getattr(evaluator, "_sg_frontier_selector", None)
    if selector is None:
        selector = GoalDirectedFrontierSelector()
        evaluator._sg_frontier_selector = selector
    config = _runtime_config(evaluator)
    goals = (
        _frozen_old_map_goal_points(evaluator)
        if config.connection_only
        else _candidate_goal_points(candidate, route_planning_anchors)
    )
    if not goals:
        goals = _candidate_goal_points(candidate, route_planning_anchors)
    if not goals:
        return None
    blacklist = list(getattr(evaluator, "_sg_exploration_blacklist", []))
    selection = selector.select(
        evaluator._sg_world_metric_map(),
        current_world_xz,
        goals,
        current_heading_rad=getattr(evaluator, "_sg_world_heading_rad", None),
        blacklist_world_xz=blacklist,
    )
    evaluator._sg_exploration_last_selection = selection.to_dict()
    if selection.status != "ok":
        logger.info(
            "[SG-EXPLORE] no usable frontier: "
            f"status={selection.status}, "
            f"reachable={selection.reachable_safe_cells}, "
            f"raw_frontier={selection.raw_frontier_cells}"
        )
        return None

    rejected = []
    planned = []
    world_only_planned = []
    resolution_m = float(evaluator._sg_world_metric_map().spec.resolution_m)
    current_heading = getattr(evaluator, "_sg_world_heading_rad", None)
    planning_current_world_xz = _start_connector_planning_pose(
        evaluator,
        selection,
        current_world_xz,
    )
    for frontier in selection.candidates[: config.maximum_planner_candidates]:
        plan = evaluator._sg_plan_world_path(
            (frontier.world_x, frontier.world_z),
            planning_current_world_xz,
            exact=True,
            planning_anchor_world_xz=None,
            project_target_to_navmesh=False,
            navmesh_terminal_snap_tolerance_m=float(
                getattr(evaluator, "sg_nav_goal_tolerance_m", 0.15)
            ),
        )
        if plan is not None:
            plan = _prepend_start_connector(evaluator, plan, selection)
            if plan is None:
                rejected.append(
                    {
                        "frontier": frontier.to_dict(),
                        "planner_failure": {
                            "reason": "invalid_bounded_start_connector"
                        },
                    }
                )
                continue
            # The reviewed planner exposes window/safe_mask in its return value
            # but keeps the optional trajectory probe only on the evaluator.
            # Capture it per candidate before the next candidate overwrites the
            # evaluator-level buffers.
            plan["_exploration_trajectory_probe_mask"] = getattr(
                evaluator, "_sg_world_trajectory_probe_mask", None
            )
            action_lattice_gate = _frontier_action_lattice_gate(
                evaluator,
                candidate,
                plan,
                current_world_xz,
                current_heading,
            )
            if action_lattice_gate is None:
                actual_path_m = float(plan["path_length_cells"]) * resolution_m
                corrected_score = float(frontier.score) - float(
                    selector.config.path_cost_weight
                ) * max(0.0, actual_path_m - frontier.path_lower_bound_m)
                world_only_planned.append(
                    (
                        corrected_score,
                        -actual_path_m,
                        frontier,
                        plan,
                        {
                            "status": (
                                "v1_world_theta_reachable_after_bounded_scan"
                            ),
                            "action_count": 0,
                            "preflight_action_lattice_reachable": False,
                        },
                    )
                )
                rejected.append(
                    {
                        "frontier": frontier.to_dict(),
                        "planner_failure": None,
                        "action_lattice_gate": {
                            "status": "v1_action_lattice_unreachable"
                        },
                    }
                )
                continue
            if plan.get("exploration_start_connector") is not None:
                plan["exploration_start_connector"][
                    "action_lattice_validation"
                ] = str(action_lattice_gate.get("status", "validated"))
            actual_path_m = float(plan["path_length_cells"]) * resolution_m
            # The pure selector uses straight-line distance only as a cheap
            # lower bound.  Once V1 Theta* has certified a candidate, charge
            # the additional detour so a frontier behind a long wall does not
            # win merely because its Euclidean coordinate is close to SG.
            corrected_score = float(frontier.score) - float(
                selector.config.path_cost_weight
            ) * max(0.0, actual_path_m - frontier.path_lower_bound_m)
            planned.append(
                (
                    corrected_score,
                    -actual_path_m,
                    frontier,
                    plan,
                    action_lattice_gate,
                )
            )
            continue
        rejected.append(
            {
                "frontier": frontier.to_dict(),
                "planner_failure": copy.deepcopy(
                    getattr(evaluator, "_sg_last_theta_failure", None)
                ),
            }
        )
    evaluator._sg_exploration_last_rejected_frontiers = copy.deepcopy(rejected)
    evaluator._sg_exploration_last_gate_audit = {
        "ranked_candidates": len(selection.candidates),
        "tested_candidates": len(rejected) + len(planned),
        "action_lattice_admitted": len(planned),
        "world_theta_only": len(world_only_planned),
        "deferred_scan_turns": int(
            candidate.get("approach_exploration_deferred_scan_turns", 0)
        ),
    }
    if not planned:
        scan_turns = int(
            candidate.get("approach_exploration_deferred_scan_turns", 0)
        )
        if (
            config.allow_world_only_gate
            and world_only_planned
            and scan_turns >= config.world_only_gate_after_scan_turns
        ):
            planned = world_only_planned
            logger.info(
                "[SG-EXPLORE] layered gate admitted a world-Theta frontier "
                "after bounded observation scan: scan_turns=%d, candidates=%d",
                scan_turns,
                len(world_only_planned),
            )
        else:
            theta_failures = sum(
                bool(item.get("planner_failure")) for item in rejected
            )
            lattice_failures = sum(
                bool(item.get("action_lattice_gate")) for item in rejected
            )
            logger.info(
                "[SG-EXPLORE] ranked frontiers exist but none passed both the V1 "
                "world planner and action-lattice admission gate: tested=%d, "
                "theta_rejected=%d, lattice_rejected=%d, scan_turns=%d",
                len(rejected),
                theta_failures,
                lattice_failures,
                scan_turns,
            )
            return None
    _, _, frontier, plan, action_lattice_gate = max(
        planned,
        key=lambda item: (
            item[0],
            item[1],
            -item[2].world_grid_z,
            -item[2].world_grid_x,
        ),
    )
    _commit_v1_plan_buffers(evaluator, plan)
    return {
        "frontier": frontier,
        "selection": selection,
        "plan": plan,
        "action_lattice_gate": action_lattice_gate,
        "goal_points": goals,
        "rejected": rejected,
        "planned_candidate_count": len(planned),
    }


def _frontier_navigation_reference(evaluator, selected, initial_failure) -> dict:
    frontier = selected["frontier"]
    return {
        "navigation_reference_policy": "goal_directed_safe_frontier_viewpoint",
        "navigation_target_source": "query_runtime_goal_directed_frontier",
        "approach_exploration_mode": True,
        "approach_exploration_version": IMPLEMENTATION_VERSION,
        "approach_exploration_frontier": frontier.to_dict(),
        "approach_exploration_goal_points_world_xz": copy.deepcopy(
            selected["goal_points"]
        ),
        "approach_exploration_initial_planner_failure": copy.deepcopy(
            initial_failure
        ),
        "approach_exploration_frontier_selection": (
            selected["selection"].to_dict()
        ),
        "approach_exploration_frontier_plan": _compact_plan(selected["plan"]),
        "approach_exploration_action_lattice_gate": copy.deepcopy(
            selected["action_lattice_gate"]
        ),
        "approach_exploration_start_step": int(
            getattr(evaluator, "current_step", 0) or 0
        ),
        "approach_exploration_action_count": 0,
        "approach_exploration_distance_m": 0.0,
    }


def _deferred_requery_approach(
    evaluator,
    candidate,
    current_world_xz,
    initial_failure,
    step=None,
) -> dict | None:
    """Create a bounded scan state when every SG waypoint is unreachable.

    A no-traversable result is evidence that the query component has not yet
    connected to the frozen acquisition map, not evidence that the semantic
    recall itself failed.  The synthetic zero-length plan keeps SG_NAV alive
    for one physical observation turn.  On the next fresh RGB-D map frame the
    normal frontier selector runs, and after connection the selected SG object
    is checked again across both episode waypoints.
    """

    metric_map = evaluator._sg_world_metric_map()
    current = np.asarray(current_world_xz, dtype=np.float64).reshape(-1)
    if metric_map is None or current.size != 2 or not np.isfinite(current).all():
        return None
    target_grid = tuple(
        map(int, metric_map.spec.world_to_cell(float(current[0]), float(current[1])))
    )
    target_world = tuple(
        map(float, metric_map.spec.cell_center_world(*target_grid))
    )
    map_frame = _query_map_frame(evaluator)
    event_step = int(
        step
        if step is not None
        else (getattr(evaluator, "current_step", 0) or 0)
    )
    evaluator._sg_exploration_initial_failures.append(
        copy.deepcopy(initial_failure)
    )
    evaluator._sg_path_plan_active = True
    return {
        "raw_world_xz": target_world,
        "navigation_reference": {
            "navigation_reference_policy": (
                "no_traversable_topk_scan_then_requery"
            ),
            "navigation_target_source": (
                "query_runtime_deferred_observation_scan"
            ),
            "approach_exploration_mode": True,
            "approach_exploration_version": IMPLEMENTATION_VERSION,
            "approach_exploration_trigger": "no_traversable_topk",
            "approach_exploration_requery_after_connection": True,
            "approach_exploration_goal_points_world_xz": (
                _candidate_goal_points(candidate)
            ),
            "approach_exploration_initial_planner_failure": copy.deepcopy(
                initial_failure
            ),
            "approach_exploration_start_step": event_step,
            "approach_exploration_action_count": 0,
            "approach_exploration_distance_m": 0.0,
            "approach_exploration_last_world_xz": list(map(float, current)),
            "approach_exploration_deferred_reselect": True,
            "approach_exploration_deferred_reason": (
                "no_traversable_topk_scan_before_frontier_selection"
            ),
            "approach_exploration_deferred_from_step": event_step,
            "approach_exploration_deferred_map_frame": map_frame,
            "approach_exploration_deferred_scan_turns": 0,
            "approach_exploration_require_frontier_scan_before_resume": True,
            "approach_exploration_history": [
                {
                    "event": "no_traversable_topk_scan_before_requery",
                    "step": event_step,
                    "world_xz": list(map(float, current)),
                    "map_frame": map_frame,
                }
            ],
        },
        "reachable": {
            "target_grid": target_grid,
            "target_world_xz": target_world,
            "raw_target_grid": target_grid,
            "path": [tuple(map(float, target_grid))],
            "raw_path": [target_grid],
            "path_length_cells": 0.0,
            "expanded_states": 0,
            "reachable_cells": 1,
            "clearance_cells": 0.0,
            "window_origin": target_grid,
            "planning_padding_cells": 0,
            "planning_route_anchor_count": 0,
            "shortcut_candidates": [],
            "trajectory_corridor": None,
            "navmesh_target_validation": {
                "status": "deferred_observation_scan",
            },
        },
    }


def _requery_approach_after_connection(
    evaluator,
    target_info,
    current_world_xz,
    original_approach,
):
    """Recheck both stored episode waypoints after the maps connect."""

    route_anchors = list(
        target_info.get("_sg_route_planning_anchors_world_xz") or []
    )
    if not bool(
        target_info.get("approach_exploration_requery_after_connection")
    ):
        approach = original_approach(
            evaluator,
            target_info,
            current_world_xz,
            {
                key: copy.deepcopy(value)
                for key, value in target_info.items()
                if key.startswith("trajectory_route_")
            },
            route_anchors,
        )
        return approach, route_anchors

    variants = evaluator._sg_dual_anchor_navigation_variants(
        target_info, current_world_xz
    )
    if not variants:
        variants = [copy.deepcopy(target_info)]
    for variant in variants:
        trajectory_route = evaluator._sg_trajectory_route_metadata(
            variant, current_world_xz
        )
        variant_route_anchors = evaluator._sg_trajectory_route_world_xz(
            trajectory_route
        )
        approach = original_approach(
            evaluator,
            variant,
            current_world_xz,
            trajectory_route,
            variant_route_anchors,
        )
        if approach is None:
            continue
        target_info.update(copy.deepcopy(variant))
        target_info.update(copy.deepcopy(trajectory_route))
        target_info["approach_exploration_requery_selected_episode"] = (
            variant.get("_sg_navigation_episode_id")
        )
        return approach, variant_route_anchors
    return None, route_anchors


def _clear_path_runtime(target_info):
    for key in (
        "_sg_local_action_queue",
        "_sg_local_action_goal_tolerance_cells",
        "_sg_local_action_goal_kind",
        "_sg_local_action_goal_world_grid",
        "_sg_local_action_goal_path_progress_cells",
        "_sg_local_action_recovery_active",
        "_sg_pending_abort_reason",
        "_sg_last_forward_world_grid_origin",
        "_sg_pending_forward_world_grid_position",
        "_sg_pending_forward_world_grid_endpoint",
        "_sg_exploration_directional_probe_pending_forward",
        "_sg_force_local_action_recovery",
        "_sg_trajectory_suffix_start_progress_cells",
    ):
        target_info.pop(key, None)


def _reset_world_path_handoff(evaluator, target_info):
    """Discard progress coordinates belonging to the previously active path."""

    reset = getattr(evaluator, "_sg_reset_world_path_stall_guard", None)
    if callable(reset):
        reset(target_info, clear_phase=True)
        return
    # Small test doubles and older reviewed bases may not expose the helper.
    # Keep the isolated overlay fail-safe by clearing the exact same fields.
    target_info.pop("_sg_world_stall_path_progress_cells", None)
    target_info.pop("_sg_previous_world_grid_cell", None)
    target_info.pop("_sg_world_stall_phase", None)
    target_info["_sg_non_improving_moves"] = 0
    pending = str(target_info.get("_sg_pending_abort_reason", ""))
    if pending.startswith("no_progress_before_exact_world_target_best_"):
        target_info.pop("_sg_pending_abort_reason", None)


def _apply_frontier_plan(
    evaluator,
    target_info,
    selected,
    step,
    logger,
    current_world_xz=None,
):
    frontier = selected["frontier"]
    plan = selected["plan"]
    resolution_m = float(target_info["world_map_resolution_m"])
    target_grid = tuple(map(int, plan["target_grid"]))
    target_world = tuple(map(float, plan["target_world_xz"]))
    position = np.asarray(target_info.get("position"), dtype=np.float64).reshape(-1)
    height = float(position[1]) if len(position) >= 2 else 0.0
    map_x, map_y = evaluator._sg_coords_to_map(
        (target_world[0], height, target_world[1])
    )
    handoff_generation = int(
        target_info.get("approach_exploration_handoff_generation", 0)
    ) + 1
    updates = {
            "target_map_x": int(map_x),
            "target_map_y": int(map_y),
            "target_world_xz": list(target_world),
            "raw_target_world_xz": [frontier.world_x, frontier.world_z],
            "target_world_grid_x": target_grid[0],
            "target_world_grid_z": target_grid[1],
            "raw_target_world_grid_x": int(plan["raw_target_grid"][0]),
            "raw_target_world_grid_z": int(plan["raw_target_grid"][1]),
            "navigation_target_source": "query_runtime_goal_directed_frontier",
            "navigation_reference_policy": "goal_directed_safe_frontier_viewpoint",
            "execution_path_source": "goal_directed_frontier_theta_star",
            "_sg_world_path_cells": list(plan["path"]),
            "_sg_path_progress_cells": 0.0,
            "theta_path_waypoints": len(plan["path"]),
            "theta_path_length_m": float(plan["path_length_cells"]) * resolution_m,
            "path_selection": {
                "execution_path_source": "goal_directed_frontier_theta_star",
                "selection_reason": "highest_ranked_safe_frontier_with_v1_plan",
                "path_length_cells": float(plan["path_length_cells"]),
            },
            "trajectory_guidance": {
                "trajectory_guidance_status": "disabled_during_frontier_connection"
            },
            "approach_exploration_mode": True,
            "approach_exploration_frontier": frontier.to_dict(),
            "approach_exploration_frontier_selection": (
                selected["selection"].to_dict()
            ),
            "approach_exploration_frontier_plan": _compact_plan(plan),
            "approach_exploration_action_lattice_gate": copy.deepcopy(
                selected["action_lattice_gate"]
            ),
            "approach_exploration_last_subgoal_step": int(step),
            "approach_exploration_handoff_generation": handoff_generation,
            "approach_exploration_handoff_state": "frontier_committed",
            "approach_exploration_handoff_step": int(step),
            "_sg_best_distance_cells": float("inf"),
            "_sg_non_improving_moves": 0,
            "_sg_correction_replan_count": 0,
            "_sg_collision_replans_without_progress": 0,
            "_sg_consecutive_failed_recoveries": 0,
        }
    # Commit target fields and the exact selected plan buffers as one overlay
    # transaction.  No stale abort/path state from the prior SG/frontier target
    # is allowed to survive this boundary.
    _clear_path_runtime(target_info)
    _reset_world_path_handoff(evaluator, target_info)
    target_info.update(updates)
    # A certified frontier handoff ends the local escape commitment.
    target_info.pop("_sg_no_action_recovery", None)
    target_info.pop("_sg_old_map_fallback", None)
    for key in (
        "approach_exploration_deferred_reselect",
        "approach_exploration_deferred_reason",
        "approach_exploration_deferred_from_step",
        "approach_exploration_deferred_map_frame",
        "approach_exploration_deferred_scan_turns",
        "_sg_exploration_forced_turn_action",
    ):
        target_info.pop(key, None)
    _commit_v1_plan_buffers(evaluator, plan)
    evaluator._sg_path_plan_active = True
    if current_world_xz is None:
        current_world_xz = evaluator._sg_current_world_xz()
    evaluator._sg_nav_begin_progress(
        step,
        float(
            np.linalg.norm(
                np.asarray(target_world)
                - np.asarray(current_world_xz, dtype=np.float64)
            )
            / resolution_m
        ),
    )
    logger.info(
        "[SG-EXPLORE] installed frontier subgoal: "
        f"world={target_world}, score={frontier.score:.3f}, "
        f"goal_progress={frontier.goal_progress_m:.3f}m, "
        f"information={frontier.information_gain_cells}, "
        f"path={float(plan['path_length_cells']) * resolution_m:.3f}m, "
        "start_connector="
        f"{(plan.get('exploration_start_connector') or {}).get('length_m', 0.0):.3f}m, "
        "v1_lattice_actions="
        f"{selected['action_lattice_gate'].get('action_count')}"
    )


def _apply_final_approach(
    evaluator,
    target_info,
    approach,
    current_world_xz,
    route_planning_anchors,
    step,
    logger,
):
    reachable = approach["reachable"]
    raw_world_xz = tuple(map(float, approach["raw_world_xz"]))
    navigation_reference = approach["navigation_reference"]
    target_world = tuple(map(float, reachable["target_world_xz"]))
    target_grid = tuple(map(int, reachable["target_grid"]))
    semantic = np.asarray(target_info.get("position"), dtype=np.float64).reshape(-1)
    semantic_xz = semantic[[0, 2]]
    trajectory_route = {
        key: copy.deepcopy(value)
        for key, value in target_info.items()
        if key.startswith("trajectory_route_")
        or key in {"trajectory_connector_kind", "trajectory_connector_world_xz"}
    }
    trajectory_guidance = evaluator._sg_compose_trajectory_guided_world_path(
        reachable, trajectory_route, current_world_xz
    )
    path_selection = evaluator._sg_select_execution_world_path(
        reachable, trajectory_guidance, allow_shortcut=True
    )
    path = list(path_selection["path"])
    resolution_m = float(target_info["world_map_resolution_m"])
    map_x, map_y = evaluator._sg_coords_to_map(
        (target_world[0], float(semantic[1]), target_world[1])
    )
    navigation_residual = float(
        np.linalg.norm(np.asarray(target_world) - np.asarray(raw_world_xz))
    )
    semantic_residual = float(
        np.linalg.norm(np.asarray(target_world) - semantic_xz)
    )
    audit = list(target_info.get("approach_exploration_history") or [])
    audit.append(
        {
            "event": "connected_to_frozen_target_component",
            "step": int(step),
            "action_count": int(
                target_info.get("approach_exploration_action_count", 0)
            ),
            "distance_m": float(
                target_info.get("approach_exploration_distance_m", 0.0)
            ),
        }
    )
    handoff_generation = int(
        target_info.get("approach_exploration_handoff_generation", 0)
    ) + 1
    updates = {
            "target_map_x": int(map_x),
            "target_map_y": int(map_y),
            "target_world_xz": list(target_world),
            "raw_target_world_xz": list(raw_world_xz),
            "navigation_reference_world_xz": list(raw_world_xz),
            "target_world_grid_x": target_grid[0],
            "target_world_grid_z": target_grid[1],
            "raw_target_world_grid_x": int(reachable["raw_target_grid"][0]),
            "raw_target_world_grid_z": int(reachable["raw_target_grid"][1]),
            "navigation_target_source": navigation_reference[
                "navigation_target_source"
            ],
            "navigation_reference_policy": navigation_reference[
                "navigation_reference_policy"
            ],
            "reachable_object_center_residual_m": navigation_residual,
            "reachable_navigation_reference_residual_m": navigation_residual,
            "reachable_semantic_object_center_residual_m": semantic_residual,
            "_sg_best_object_center_residual_m": navigation_residual,
            "_sg_best_object_center_target_world_xz": list(target_world),
            "_sg_route_planning_anchors_world_xz": [
                list(map(float, point)) for point in route_planning_anchors
            ],
            "_sg_world_path_cells": path,
            "_sg_path_progress_cells": 0.0,
            "execution_path_source": str(path_selection["execution_path_source"]),
            "path_selection": {
                key: copy.deepcopy(value)
                for key, value in path_selection.items()
                if key != "path"
            },
            "trajectory_guidance": copy.deepcopy(trajectory_guidance),
            "trajectory_corridor": copy.deepcopy(
                reachable.get("trajectory_corridor")
            ),
            "navmesh_target_validation": copy.deepcopy(
                reachable.get("navmesh_target_validation")
            ),
            "theta_path_waypoints": len(path),
            "theta_path_length_m": float(
                evaluator._sg_path_cumulative(path)[-1] * resolution_m
            ),
            "approach_exploration_mode": False,
            "approach_exploration_connected_step": int(step),
            "approach_exploration_handoff_generation": handoff_generation,
            "approach_exploration_handoff_state": "original_sg_committed",
            "approach_exploration_handoff_step": int(step),
            "approach_exploration_history": audit,
            "_sg_best_distance_cells": float("inf"),
            "_sg_non_improving_moves": 0,
            "_sg_correction_replan_count": 0,
            "_sg_collision_replans_without_progress": 0,
            "_sg_consecutive_failed_recoveries": 0,
        }
    _clear_path_runtime(target_info)
    _reset_world_path_handoff(evaluator, target_info)
    target_info.update(updates)
    # Original SG navigation is now connected; do not carry escape state into
    # a future disconnect or the second recalled object.
    target_info.pop("_sg_no_action_recovery", None)
    target_info.pop("_sg_old_map_fallback", None)
    for key in (
        "approach_exploration_deferred_reselect",
        "approach_exploration_deferred_reason",
        "approach_exploration_deferred_from_step",
        "approach_exploration_deferred_map_frame",
        "_sg_exploration_forced_turn_action",
    ):
        target_info.pop(key, None)
    for key, value in navigation_reference.items():
        if key.startswith("approach_") or key.startswith("sofa_"):
            target_info[key] = copy.deepcopy(value)
    if path_selection["execution_path_source"] == (
        "certified_trajectory_then_world_theta_star"
    ):
        target_info["_sg_trajectory_suffix_start_progress_cells"] = float(
            trajectory_guidance["trajectory_suffix_start_progress_cells"]
        )
    _commit_v1_plan_buffers(evaluator, reachable)
    evaluator._sg_path_plan_active = True
    evaluator._sg_nav_begin_progress(
        step,
        float(np.linalg.norm(np.asarray(target_world) - current_world_xz) / resolution_m),
    )
    evaluator._write_sg_decision(
        step,
        {
            "decision": "exploration_connected_resume_sg_nav",
            "resume_policy": (
                "requery_both_episode_waypoints_after_no_traversable_topk"
                if target_info.get(
                    "approach_exploration_requery_after_connection"
                )
                else "resume_frozen_sg_target"
            ),
            "selected_uid": target_info.get("uid") or target_info.get("node_id"),
            "selected_label": target_info.get("label"),
            "navigation_target_world_xz": list(target_world),
            "exploration_actions": int(
                target_info.get("approach_exploration_action_count", 0)
            ),
            "exploration_distance_m": float(
                target_info.get("approach_exploration_distance_m", 0.0)
            ),
            "ordinary_vln_executed": False
            if isinstance(target_info.get("_sg_dual_anchor_index"), int)
            else None,
        },
    )
    logger.info(
        "[SG-EXPLORE] runtime map connected; resumed the unchanged V1 SG goal: "
        f"target={target_world}, waypoints={len(path)}"
    )


def _mark_connection_only_complete(
    evaluator,
    target_info,
    connection,
    step,
    logger,
):
    """Record map-to-map connectivity and stop before any SG-goal motion."""

    goal = list(map(float, connection["goal_world_xz"]))
    plan = connection["plan"]
    history = list(target_info.get("approach_exploration_history") or [])
    history.append(
        {
            "event": "connected_to_frozen_acquisition_trajectory",
            "step": int(step),
            "action_count": int(
                target_info.get("approach_exploration_action_count", 0)
            ),
            "distance_m": float(
                target_info.get("approach_exploration_distance_m", 0.0)
            ),
            "old_map_goal_world_xz": goal,
            "old_map_goal_rank_by_distance": int(
                connection["goal_rank_by_distance"]
            ),
        }
    )
    _clear_path_runtime(target_info)
    _reset_world_path_handoff(evaluator, target_info)
    target_info.update(
        {
            "approach_exploration_mode": False,
            "approach_exploration_connected_step": int(step),
            "approach_exploration_handoff_state": (
                "frozen_old_map_connected_connection_only"
            ),
            "approach_exploration_history": history,
            "approach_exploration_connection_only": True,
            "approach_exploration_old_map_goal_world_xz": goal,
            "approach_exploration_old_map_goal_rank_by_distance": int(
                connection["goal_rank_by_distance"]
            ),
            "approach_exploration_old_map_connection_plan": _compact_plan(plan),
            "_sg_exploration_connection_only_complete": True,
        }
    )
    # Keep SG_NAV alive for exactly one controller call; its exploration
    # wrapper emits STOP without installing or following an SG target path.
    evaluator._sg_path_plan_active = True
    evaluator._write_sg_decision(
        step,
        {
            "decision": "exploration_connected_resume_sg_nav",
            "connection_only": True,
            "resume_policy": "stop_after_frozen_old_map_connection",
            "old_map_reference": "immutable_m1_acquisition_trajectory",
            "old_map_goal_world_xz": goal,
            "old_map_goal_rank_by_distance": int(
                connection["goal_rank_by_distance"]
            ),
            "old_map_nodes_tested": int(connection["tested_nodes"]),
            "exploration_actions": int(
                target_info.get("approach_exploration_action_count", 0)
            ),
            "exploration_distance_m": float(
                target_info.get("approach_exploration_distance_m", 0.0)
            ),
            "ordinary_vln_executed": False,
        },
    )
    logger.info(
        "[SG-EXPLORE] navigation map connected to frozen M1 trajectory; "
        "connection-only evaluation will stop: target=%s, actions=%d",
        goal,
        int(target_info.get("approach_exploration_action_count", 0)),
    )


def _update_motion_budget(target_info, current_world_xz):
    previous = target_info.get("approach_exploration_last_world_xz")
    if previous is not None:
        try:
            delta = float(
                np.linalg.norm(
                    np.asarray(current_world_xz, dtype=np.float64)
                    - np.asarray(previous, dtype=np.float64)
                )
            )
        except (TypeError, ValueError):
            delta = 0.0
        if np.isfinite(delta) and delta <= 1.0:
            target_info["approach_exploration_distance_m"] = float(
                target_info.get("approach_exploration_distance_m", 0.0)
            ) + delta
    target_info["approach_exploration_last_world_xz"] = list(
        map(float, current_world_xz)
    )


def _query_map_frame(evaluator) -> int:
    adapter = getattr(evaluator, "inherit_sg", None)
    return max(
        int(getattr(adapter, "_query_map_only_frames", 0)),
        int(getattr(adapter, "frame_idx", 0)),
        int(getattr(adapter, "_world_observation_count", 0)),
    )


def _directional_probe_cues(evaluator, target_info, current_world_xz):
    """Return directional hints without treating them as safe subgoals."""

    from .old_map_fallback import STATE_KEY
    state = target_info.get(STATE_KEY, {})
    if state.get('active') and state.get('cue') is not None:
        return [{'source':'frozen_old_map_fallback',
                 'world_xz':np.asarray(state['cue'],dtype=float)}]
    cues = []
    selection = getattr(evaluator, "_sg_exploration_last_selection", None)
    if isinstance(selection, dict):
        candidates = list(selection.get("candidates") or [])
        if candidates:
            best = candidates[0]
            try:
                point = np.asarray(
                    [best["world_x"], best["world_z"]], dtype=np.float64
                )
            except (KeyError, TypeError, ValueError):
                point = np.zeros(0, dtype=np.float64)
            if point.size == 2 and np.isfinite(point).all():
                cues.append(
                    {
                        "source": "best_policy_rejected_frontier",
                        "world_xz": point,
                    }
                )

    config = _runtime_config(evaluator)
    if config.connection_only:
        goal_points = _frozen_old_map_goal_points(evaluator)
        source = "nearest_frozen_old_trajectory"
    else:
        goal_points = list(
            target_info.get("approach_exploration_goal_points_world_xz") or []
        )
        if not goal_points:
            goal_points = _candidate_goal_points(target_info)
        source = "nearest_original_target_hint"
    current = np.asarray(current_world_xz, dtype=np.float64).reshape(2)
    valid_goals = []
    for point in goal_points:
        try:
            point = np.asarray(point, dtype=np.float64).reshape(2)
        except (TypeError, ValueError):
            continue
        if np.isfinite(point).all():
            valid_goals.append(point)
    valid_goals.sort(
        key=lambda point: (
            float(np.linalg.norm(point - current)),
            float(point[1]),
            float(point[0]),
        )
    )
    for point in valid_goals[:8]:
        if any(
            np.linalg.norm(point - cue["world_xz"]) <= 1e-6 for cue in cues
        ):
            continue
        cues.append({"source": source, "world_xz": point})
    return cues


def _known_obstacle_clear_mask(occupied, clearance_cells):
    """Veto known obstacles around a centreline while allowing unknown sides.

    A strict frontier requires the entire agent disk to be *known free*.  This
    fallback is intentionally less conservative: its centreline must still be
    observed free, while the surrounding disk only rejects known obstacles.
    Habitat ``try_step`` remains the final physical one-action certificate.
    """

    occupied = np.asarray(occupied, dtype=bool)
    result = np.ones_like(occupied, dtype=bool)
    radius = int(math.ceil(max(0.0, float(clearance_cells))))
    if radius <= 0 or not occupied.any():
        return result
    height, width = occupied.shape
    padded = np.pad(occupied, radius, mode="constant", constant_values=False)
    nearby = np.zeros_like(occupied, dtype=bool)
    for dz in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dz * dz > float(clearance_cells) ** 2:
                continue
            source_z = radius + dz
            source_x = radius + dx
            nearby |= padded[
                source_z : source_z + height,
                source_x : source_x + width,
            ]
    return np.logical_not(nearby)


def _settle_directional_probe_forward(
    evaluator, target_info, current_world_xz, step
):
    pending = target_info.pop(
        "_sg_exploration_directional_probe_pending_forward", None
    )
    if not isinstance(pending, dict):
        return None
    current = np.asarray(current_world_xz, dtype=np.float64).reshape(2)
    origin = np.asarray(pending.get("origin_world_xz"), dtype=np.float64).reshape(2)
    displacement_m = float(np.linalg.norm(current - origin))
    expected_m = float(pending.get("expected_displacement_m", 0.0))
    moved = bool(displacement_m + 1e-9 >= 0.5 * expected_m)
    if not moved:
        metric_map = evaluator._sg_world_metric_map()
        register = getattr(evaluator, "_sg_register_world_collision_block", None)
        endpoint = pending.get("endpoint_world_grid")
        if metric_map is not None and callable(register) and endpoint is not None:
            resolution_m = float(metric_map.spec.resolution_m)
            register(
                endpoint,
                actual_position_grid=current / resolution_m - 0.5,
            )
    result = {
        "decision": "exploration_directional_probe_forward_result",
        "moved": moved,
        "actual_displacement_m": displacement_m,
        "expected_displacement_m": expected_m,
        "cue_source": pending.get("cue_source"),
        "cue_world_xz": pending.get("cue_world_xz"),
        "ordinary_vln_executed": False,
    }
    evaluator._write_sg_decision(step, result)
    return result


def _directional_probe_connectivity_recheck_due(
    target_info,
    config,
    *,
    map_frame,
    action_map_frame,
    force=False,
    allow_without_actions=False,
):
    """Claim a periodic original-target recheck after fresh probe evidence.

    Directional-probe actions are counted rather than raw simulator frames: the
    requested four-step cadence therefore covers both observation turns and
    certified forward trials.  A check is delayed until the selected action has
    produced a new RGB-D map frame, and the same action count can never trigger
    the expensive SG approach query twice.
    """

    probe_actions = int(
        target_info.get("approach_exploration_directional_probe_actions", 0)
    )
    last_checked_actions = int(
        target_info.get(
            "approach_exploration_directional_probe_last_connectivity_actions",
            0,
        )
    )
    if force and allow_without_actions:
        # An observation-only scan (or newly fused dense depth) may connect the
        # target before the very first probe action. Deduplicate by evidence
        # frame and action count, not by a strictly positive action delta.
        identity = (int(map_frame), probe_actions)
        if int(map_frame) < 0 or target_info.get("_sg_probe_last_forced_evidence") == identity:
            return False
        target_info["_sg_probe_last_forced_evidence"] = identity
        target_info["approach_exploration_directional_probe_last_connectivity_actions"] = probe_actions
        target_info["approach_exploration_directional_probe_last_connectivity_frame"] = int(map_frame)
        return True
    if probe_actions <= last_checked_actions:
        return False
    if int(map_frame) <= int(action_map_frame):
        return False
    stride = max(1, int(config.directional_probe_connectivity_stride))
    if not force and probe_actions - last_checked_actions < stride:
        return False
    target_info[
        "approach_exploration_directional_probe_last_connectivity_actions"
    ] = probe_actions
    target_info[
        "approach_exploration_directional_probe_last_connectivity_frame"
    ] = int(map_frame)
    return True


def _recheck_original_target_after_directional_probe(
    evaluator,
    target_info,
    current_world_xz,
    original_approach,
    *,
    map_frame,
    action_map_frame,
    step,
    logger,
    force=False,
):
    """Requery the frozen SG target every four directional-probe actions."""

    from . import probe_recovery
    config = _runtime_config(evaluator)
    if not _directional_probe_connectivity_recheck_due(
        target_info,
        config,
        map_frame=map_frame,
        action_map_frame=action_map_frame,
        force=force,
        allow_without_actions=(probe_recovery.enabled(evaluator)
                               or getattr(config, 'old_map_fallback', False)),
    ):
        return False

    probe_actions = int(
        target_info.get("approach_exploration_directional_probe_actions", 0)
    )
    probe_forwards = int(
        target_info.get(
            "approach_exploration_directional_probe_forward_actions", 0
        )
    )
    connection = None
    final = None
    route_anchors = []
    if config.connection_only:
        connection = _probe_frozen_old_map_connection(
            evaluator, current_world_xz
        )
    else:
        final, route_anchors = _requery_approach_after_connection(
            evaluator,
            target_info,
            current_world_xz,
            original_approach,
        )
        if final is None:
            from .old_map_fallback import check_old_connection
            check_old_connection(evaluator, target_info, current_world_xz, step=step)

    connected = bool(connection is not None or final is not None)
    evaluator._write_sg_decision(
        step,
        {
            "decision": "exploration_directional_probe_connectivity_recheck",
            "status": "connected" if connected else "still_disconnected",
            "probe_actions": probe_actions,
            "probe_forwards": probe_forwards,
            "probe_stride": int(config.directional_probe_connectivity_stride),
            "map_frame": int(map_frame),
            "forced_before_probe_abort": bool(force),
            "ordinary_vln_executed": False,
        },
    )
    if connection is not None:
        _mark_connection_only_complete(
            evaluator,
            target_info,
            connection,
            step,
            logger,
        )
        return True
    if final is not None:
        _apply_final_approach(
            evaluator,
            target_info,
            final,
            current_world_xz,
            route_anchors,
            step,
            logger,
        )
        return True
    return False


def _certified_directional_probe_action(
    evaluator,
    target_info,
    current_world_xz,
    current_heading,
    actions,
    *,
    step,
):
    """Choose one certified action toward a rejected frontier or old map.

    The hint is never installed as a navigation target.  Every call replans
    from the measured pose; a forward action is returned only when its exact
    0.25-m centreline is observed free, clear of known obstacles, and accepted
    by Habitat's live navmesh ``try_step``.
    """

    from .old_map_fallback import config_for_target
    config = config_for_target(evaluator, target_info)
    if not config.directional_probe:
        return {"status": "disabled", "action": None}
    used_actions = int(
        target_info.get("approach_exploration_directional_probe_actions", 0)
    )
    used_forwards = int(
        target_info.get(
            "approach_exploration_directional_probe_forward_actions", 0
        )
    )
    if used_actions >= config.directional_probe_max_actions:
        return {"status": "action_budget_exhausted", "action": None}
    if used_forwards >= config.directional_probe_max_forward_actions:
        return {"status": "forward_budget_exhausted", "action": None}

    metric_map = evaluator._sg_world_metric_map()
    if metric_map is None or current_heading is None:
        return {"status": "missing_map_or_heading", "action": None}
    current = np.asarray(current_world_xz, dtype=np.float64).reshape(2)
    heading = float(np.asarray(current_heading).reshape(-1)[0])
    cues = _directional_probe_cues(evaluator, target_info, current)
    if not cues:
        return {"status": "missing_directional_cue", "action": None}

    resolution_m = float(metric_map.spec.resolution_m)
    try:
        forward_step_m = float(
            evaluator.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE
        )
        turn_angle_deg = abs(
            float(evaluator.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE)
        )
    except (AttributeError, TypeError, ValueError):
        return {"status": "missing_discrete_action_geometry", "action": None}
    if forward_step_m <= 0.0 or turn_angle_deg <= 0.0:
        return {"status": "invalid_discrete_action_geometry", "action": None}
    step_cells = forward_step_m / resolution_m
    position_grid = current / resolution_m - 0.5
    padding = int(
        math.ceil(step_cells + config.directional_probe_clearance_cells)
    ) + 2
    start_cell = metric_map.spec.world_to_cell(*current)
    try:
        window = metric_map.planning_window(
            start_cell,
            start_cell,
            floor_id=0,
            padding_cells=padding,
            max_window_cells=max(81, (2 * padding + 1) ** 2),
        )
    except Exception as exc:
        return {
            "status": "planning_window_failed",
            "detail": str(exc),
            "action": None,
        }
    probe_mask = np.logical_and(
        np.asarray(window.free, dtype=bool),
        _known_obstacle_clear_mask(
            window.occupied, config.directional_probe_clearance_cells
        ),
    )
    start_local = evaluator._sg_window_point(window, position_grid)
    start_x, start_z = map(int, np.rint(start_local))
    if 0 <= start_z < probe_mask.shape[0] and 0 <= start_x < probe_mask.shape[1]:
        # The simulator pose itself is a stronger certificate for this single
        # sample than a noisy occupancy classification at the same location.
        probe_mask[start_z, start_x] = True

    navmesh_context = evaluator._sg_navmesh_context()
    if config.directional_probe_require_navmesh and navmesh_context is None:
        return {"status": "missing_live_navmesh", "action": None}

    turn_count = max(1, int(round(360.0 / turn_angle_deg)))
    offsets = list(range(turn_count))
    offsets = [
        value if value <= turn_count // 2 else value - turn_count
        for value in offsets
    ]
    offsets.sort(key=lambda value: (abs(value), -value))
    rejection_counts = {
        "navmesh": 0,
        "displacement": 0,
        "known_free_or_obstacle_clearance": 0,
        "progress": 0,
    }
    selected = None
    for cue_rank, cue in enumerate(cues):
        cue_point = np.asarray(cue["world_xz"], dtype=np.float64).reshape(2)
        current_distance_m = float(np.linalg.norm(current - cue_point))
        candidates = []
        for offset in offsets:
            candidate_heading = heading + math.radians(turn_angle_deg * offset)
            endpoint_grid, navmesh_predicted = evaluator._sg_predict_forward_endpoint(
                position_grid,
                candidate_heading,
                step_cells,
                resolution_m,
                navmesh_context,
            )
            if config.directional_probe_require_navmesh and not navmesh_predicted:
                rejection_counts["navmesh"] += 1
                continue
            actual_step_cells = float(
                np.linalg.norm(
                    np.asarray(endpoint_grid, dtype=np.float64) - position_grid
                )
            )
            if actual_step_cells < 0.5 * step_cells:
                rejection_counts["displacement"] += 1
                continue
            endpoint_local = evaluator._sg_window_point(window, endpoint_grid)
            if not evaluator._sg_float_segment_is_safe(
                probe_mask, start_local, endpoint_local
            ):
                rejection_counts["known_free_or_obstacle_clearance"] += 1
                continue
            endpoint_world = (
                np.asarray(endpoint_grid, dtype=np.float64) + 0.5
            ) * resolution_m
            progress_m = current_distance_m - float(
                np.linalg.norm(endpoint_world - cue_point)
            )
            if progress_m + 1e-9 < config.directional_probe_min_progress_m:
                rejection_counts["progress"] += 1
                continue
            candidates.append(
                {
                    "cue_rank": int(cue_rank),
                    "cue_source": str(cue["source"]),
                    "cue_world_xz": list(map(float, cue_point)),
                    "heading_rad": float(candidate_heading),
                    "turn_offset": int(offset),
                    "endpoint_world_grid": list(map(float, endpoint_grid)),
                    "endpoint_world_xz": list(map(float, endpoint_world)),
                    "expected_displacement_m": actual_step_cells * resolution_m,
                    "progress_m": float(progress_m),
                    "navmesh_predicted": bool(navmesh_predicted),
                }
            )
        if candidates:
            selected = max(
                candidates,
                key=lambda item: (
                    item["progress_m"],
                    -abs(item["turn_offset"]),
                    item["expected_displacement_m"],
                ),
            )
            break
    if selected is None:
        return {
            "status": "no_certified_progressing_single_step",
            "action": None,
            "rejection_counts": rejection_counts,
        }

    error = _heading_error(selected["heading_rad"], heading)
    if abs(math.degrees(error)) > 0.5 * turn_angle_deg + 1e-6:
        action = int(actions.TURN_LEFT if error > 0 else actions.TURN_RIGHT)
        action_kind = "turn"
    else:
        action = int(actions.MOVE_FORWARD)
        action_kind = "forward"
        target_info[
            "_sg_exploration_directional_probe_pending_forward"
        ] = {
            "origin_world_xz": list(map(float, current)),
            "endpoint_world_grid": selected["endpoint_world_grid"],
            "expected_displacement_m": selected["expected_displacement_m"],
            "cue_source": selected["cue_source"],
            "cue_world_xz": selected["cue_world_xz"],
        }
        target_info[
            "approach_exploration_directional_probe_forward_actions"
        ] = used_forwards + 1
    target_info["approach_exploration_directional_probe_actions"] = (
        used_actions + 1
    )
    target_info["approach_exploration_directional_probe_active"] = True
    target_info["approach_exploration_deferred_map_frame"] = _query_map_frame(
        evaluator
    )
    target_info.pop("approach_exploration_directional_probe_requested", None)
    audit = {
        "status": "action_selected",
        "action": action,
        "action_kind": action_kind,
        "probe_action_index": used_actions + 1,
        "probe_forward_index": used_forwards + (1 if action_kind == "forward" else 0),
        **selected,
        "rejection_counts": rejection_counts,
    }
    return audit


def _primary_directional_probe_action(evaluator, target_info, current_world_xz,
                              current_heading, actions, *, step):
    """Use the configured probe gate first; recover only when it has no action."""
    from . import probe_recovery
    active = bool(target_info.get(probe_recovery.STATE_KEY, {}).get("active"))
    if (_runtime_config(evaluator).directional_probe
            and probe_recovery.enabled(evaluator) and active):
        if target_info[probe_recovery.STATE_KEY].get("segment_forwards", 0) >= 4:
            result = _certified_directional_probe_action(
                evaluator, target_info, current_world_xz, current_heading, actions, step=step)
            if result.get("action") is not None:
                target_info.pop(probe_recovery.STATE_KEY, None)
                return result
        return probe_recovery.recover(
            evaluator, target_info, current_world_xz, current_heading, actions,
            step=step, original_failure=target_info[probe_recovery.STATE_KEY]["trigger"])
    result = _certified_directional_probe_action(
        evaluator, target_info, current_world_xz, current_heading, actions, step=step)
    if (probe_recovery.enabled(evaluator)
            and result.get("status") == "no_certified_progressing_single_step"):
        return probe_recovery.recover(
            evaluator, target_info, current_world_xz, current_heading, actions,
            step=step, original_failure=result)
    return result


def _directional_probe_action(evaluator, target_info, current_world_xz,
                              current_heading, actions, *, step):
    from . import old_map_fallback
    state = target_info.get(old_map_fallback.STATE_KEY, {})
    if state.get('active') or state.get('connected'):
        return old_map_fallback.action(evaluator, target_info, current_world_xz,
                                       current_heading, actions, step=step)
    return _primary_directional_probe_action(evaluator,target_info,current_world_xz,
                                             current_heading,actions,step=step)


def _request_directional_probe(
    evaluator, target_info, *, reason, step, map_frame
):
    _clear_path_runtime(target_info)
    _reset_world_path_handoff(evaluator, target_info)
    target_info.update(
        {
            "approach_exploration_mode": True,
            "approach_exploration_deferred_reselect": True,
            "approach_exploration_deferred_reason": str(reason),
            "approach_exploration_deferred_map_frame": int(map_frame),
            "approach_exploration_directional_probe_requested": True,
            "_sg_world_path_cells": [],
            "_sg_path_progress_cells": 0.0,
        }
    )
    evaluator._sg_world_plan_window = None
    evaluator._sg_world_safe_mask = None
    evaluator._sg_world_trajectory_probe_mask = None
    evaluator._sg_path_plan_active = True
    evaluator._write_sg_decision(
        step,
        {
            "decision": "exploration_directional_probe_requested",
            "reason": str(reason),
            "map_frame": int(map_frame),
            "ordinary_vln_executed": False,
        },
    )


def _observation_turn_action(actions, target_info, current_heading) -> int:
    """Choose a collision-free turn owned by exploration, not V1 control."""

    frontier = target_info.get("approach_exploration_frontier") or {}
    desired = frontier.get("scan_heading_rad")
    if desired is not None and current_heading is not None:
        error = _heading_error(float(desired), float(current_heading))
        if abs(error) > math.radians(1.0):
            return int(actions.TURN_LEFT if error > 0 else actions.TURN_RIGHT)
    # Repeated scan-only retries rotate through distinct headings.
    return int(actions.TURN_LEFT)


def _defer_frontier_reselection(
    evaluator,
    target_info,
    *,
    reason,
    step,
    current_world_xz,
    actions,
    logger,
    frontier_was_active,
) -> None:
    """Consume the abort and require a physical observation before reselection."""

    current_heading = getattr(evaluator, "_sg_world_heading_rad", None)
    if current_heading is not None:
        current_heading = float(np.asarray(current_heading).reshape(-1)[0])
    map_frame = _query_map_frame(evaluator)
    failed_frontier = (
        target_info.get("target_world_xz") if frontier_was_active else None
    )
    attempts = list(
        target_info.get("approach_exploration_failed_frontier_attempts") or []
    )
    attempts.append(
        {
            "step": int(step),
            "map_frame": map_frame,
            "reason": str(reason),
            "frontier_world_xz": copy.deepcopy(failed_frontier),
            "agent_world_xz": list(map(float, current_world_xz)),
            "agent_heading_rad": current_heading,
        }
    )
    _clear_path_runtime(target_info)
    _reset_world_path_handoff(evaluator, target_info)
    target_info.update(
        {
            "approach_exploration_mode": True,
            "approach_exploration_deferred_reselect": True,
            "approach_exploration_deferred_reason": str(reason),
            "approach_exploration_deferred_from_step": int(step),
            "approach_exploration_deferred_map_frame": map_frame,
            "approach_exploration_failed_frontier_attempts": attempts,
            "_sg_world_path_cells": [],
            "_sg_path_progress_cells": 0.0,
            "_sg_exploration_forced_turn_action": _observation_turn_action(
                actions, target_info, current_heading
            ),
        }
    )
    evaluator._sg_world_plan_window = None
    evaluator._sg_world_safe_mask = None
    evaluator._sg_world_trajectory_probe_mask = None
    evaluator._sg_path_plan_active = True
    evaluator._write_sg_decision(
        step,
        {
            "decision": (
                "exploration_subgoal_abort_deferred_reselect"
                if frontier_was_active
                else "exploration_initial_abort_scan_before_reselect"
            ),
            "reason": str(reason),
            "selected_uid": target_info.get("uid")
            or target_info.get("node_id"),
            "failed_frontier_world_xz": copy.deepcopy(failed_frontier),
            "failed_agent_world_xz": list(map(float, current_world_xz)),
            "failed_agent_heading_rad": current_heading,
            "required_new_map_frame_after": map_frame,
            "ordinary_vln_executed": False
            if isinstance(target_info.get("_sg_dual_anchor_index"), int)
            else None,
        },
    )
    logger.info(
        "[SG-EXPLORE] consumed frontier abort; deferring reselection until "
        f"after a physical turn and a new RGB-D map frame: reason={reason}, "
        f"map_frame={map_frame}"
    )


def install_goal_directed_exploration(evaluator_class, actions, logger):
    """Patch geometric disconnects into safe frontier-connection episodes."""

    if getattr(evaluator_class, _INSTALL_MARKER, False):
        return evaluator_class

    original_init = evaluator_class.__init__
    original_reset = evaluator_class.reset
    original_approach = evaluator_class._sg_plan_v8_post_hit_approach
    original_reached = evaluator_class._sg_nav_reached_reason
    original_get_action = evaluator_class._get_sg_world_path_action
    original_aborted = evaluator_class._on_sg_nav_aborted

    def exploration_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._sg_exploration_config = ExplorationConfig.from_environment()
        self._sg_frontier_selector = GoalDirectedFrontierSelector()
        self._sg_exploration_blacklist = []
        self._sg_exploration_last_selection = None
        self._sg_exploration_initial_failures = []
        logger.info(
            "[MEMORY-EXPLORE] complete controller installed: version=%s, "
            "enabled=%s, connection_only=%s, directional_probe=%s, "
            "max_actions=%d, max_distance_m=%.2f",
            IMPLEMENTATION_VERSION,
            self._sg_exploration_config.enabled,
            self._sg_exploration_config.connection_only,
            self._sg_exploration_config.directional_probe,
            self._sg_exploration_config.maximum_actions,
            self._sg_exploration_config.maximum_distance_m,
        )

    def exploration_reset(self, *args, **kwargs):
        self._sg_exploration_blacklist = []
        self._sg_exploration_last_selection = None
        self._sg_exploration_initial_failures = []
        return original_reset(self, *args, **kwargs)

    def exploration_no_traversable_topk(
        self,
        *,
        candidates,
        rejected_targets,
        current_world_xz,
        task_text,
        full_instruction,
        query_source,
        todo_idx,
        dual_context,
        step,
    ):
        if not _runtime_enabled(self) or not candidates:
            return None
        candidate = copy.deepcopy(candidates[0])
        position = np.asarray(candidate.get("position"), dtype=np.float64).reshape(-1)
        if position.size < 3 or not np.isfinite(position[:3]).all():
            return None

        # Geometric failures are retryable after the query map grows.  Do not
        # let the normal failed-target blacklist hide the same VLM-selected UID
        # when reachability is recomputed after map connection.
        task_key = self._sg_task_key(task_text)
        blocked = self._sg_blocked_by_task.get(task_key)
        if blocked is not None:
            for item in candidates:
                blocked.discard(self._sg_target_key(item))
            if not blocked:
                self._sg_blocked_by_task.pop(task_key, None)

        trajectory_route = self._sg_trajectory_route_metadata(
            candidate, current_world_xz
        )
        route_planning_anchors = self._sg_trajectory_route_world_xz(
            trajectory_route
        )
        failure = {
            "reason": "no_traversable_topk",
            "rejected": copy.deepcopy(rejected_targets),
        }
        approach = _deferred_requery_approach(
            self,
            candidate,
            current_world_xz,
            failure,
            step=step,
        )
        if approach is None:
            return None
        logger.info(
            "[MEMORY-EXPLORE] no traversable SG waypoint; entering a fresh "
            "RGB-D scan/frontier cycle before rechecking both episode "
            "waypoints: uid={}".format(
                (dual_context or {}).get("selected_uid")
                or candidate.get("uid")
                or candidate.get("node_id")
            )
        )
        return (
            candidate,
            approach["raw_world_xz"],
            (float(position[0]), float(position[2])),
            approach["navigation_reference"],
            approach["reachable"],
            trajectory_route,
            route_planning_anchors,
        )

    def exploration_approach(
        self,
        candidate,
        current_world_xz,
        trajectory_route,
        route_planning_anchors,
    ):
        result = original_approach(
            self,
            candidate,
            current_world_xz,
            trajectory_route,
            route_planning_anchors,
        )
        if result is not None or not _runtime_enabled(self):
            return result
        failure = copy.deepcopy(getattr(self, "_sg_last_theta_failure", None))
        if not _is_connectivity_failure(failure):
            return None
        selected = _select_frontier_plan(
            self,
            candidate,
            current_world_xz,
            route_planning_anchors,
            logger,
        )
        if selected is None:
            return None
        self._sg_exploration_initial_failures.append(failure)
        return {
            "raw_world_xz": (
                selected["frontier"].world_x,
                selected["frontier"].world_z,
            ),
            "navigation_reference": _frontier_navigation_reference(
                self, selected, failure
            ),
            "reachable": selected["plan"],
        }

    def exploration_reached(self, **kwargs):
        target_info = kwargs.get("target_info") or {}
        if not bool(target_info.get("approach_exploration_mode")):
            return original_reached(self, **kwargs)
        current = self._sg_current_world_xz(kwargs)
        if current is None:
            return original_reached(self, **kwargs)
        current = np.asarray(current, dtype=np.float64).reshape(2)
        _update_motion_budget(target_info, current)
        config = _runtime_config(self)
        actions_used = int(target_info.get("approach_exploration_action_count", 0))
        distance_used = float(target_info.get("approach_exploration_distance_m", 0.0))
        if actions_used >= config.maximum_actions:
            target_info["_sg_pending_abort_reason"] = (
                f"exploration_budget_actions_{actions_used}"
            )
            self._sg_path_plan_active = False
            return None
        if distance_used >= config.maximum_distance_m:
            target_info["_sg_pending_abort_reason"] = (
                f"exploration_budget_distance_{distance_used:.2f}m"
            )
            self._sg_path_plan_active = False
            return None

        # A failed frontier is not replaced until an exploration-owned turn has
        # executed and its RGB-D observation has entered the query map.  Do not
        # let the stale frontier be interpreted as reached in the meantime.
        if bool(target_info.get("approach_exploration_deferred_reselect")):
            self._sg_path_plan_active = True
            return None

        adapter = getattr(self, "inherit_sg", None)
        map_frame = _query_map_frame(self)
        last_check = int(target_info.get("approach_exploration_last_connectivity_frame", -10**9))
        target_world = np.asarray(target_info["target_world_xz"], dtype=np.float64)
        frontier_distance = float(np.linalg.norm(current - target_world))
        at_frontier = frontier_distance <= config.frontier_goal_tolerance_m + 1e-6
        should_check = bool(
            (config.connection_only or _can_resume_original_target(target_info))
            and (
                at_frontier
                or map_frame - last_check >= config.connectivity_check_stride
            )
        )
        route_anchors = list(
            target_info.get("_sg_route_planning_anchors_world_xz") or []
        )
        if should_check:
            target_info["approach_exploration_last_connectivity_frame"] = map_frame
            if config.connection_only:
                connection = _probe_frozen_old_map_connection(self, current)
                if connection is not None:
                    _mark_connection_only_complete(
                        self,
                        target_info,
                        connection,
                        kwargs.get("step", 0),
                        logger,
                    )
                    return None
            else:
                final, route_anchors = _requery_approach_after_connection(
                    self,
                    target_info,
                    current,
                    original_approach,
                )
                if final is not None:
                    _apply_final_approach(
                        self,
                        target_info,
                        final,
                        current,
                        route_anchors,
                        kwargs.get("step", 0),
                        logger,
                    )
                    return None

        if not at_frontier:
            return original_reached(self, **kwargs)

        frontier = target_info.get("approach_exploration_frontier") or {}
        desired_heading = frontier.get("scan_heading_rad")
        current_heading = kwargs.get("current_world_heading")
        if current_heading is None:
            current_heading = getattr(self, "_sg_world_heading_rad", None)
        if desired_heading is not None and current_heading is not None:
            current_heading = float(np.asarray(current_heading).reshape(-1)[0])
            error = _heading_error(float(desired_heading), current_heading)
            effective_tolerance_deg = _effective_scan_heading_tolerance_deg(
                self, config.scan_heading_tolerance_deg
            )
            target_info[
                "_sg_exploration_scan_heading_effective_tolerance_deg"
            ] = effective_tolerance_deg
            if abs(math.degrees(error)) > effective_tolerance_deg:
                target_info["_sg_exploration_forced_turn_action"] = int(
                    actions.TURN_LEFT if error > 0 else actions.TURN_RIGHT
                )
                target_info["_sg_exploration_scan_heading_error_deg"] = float(
                    math.degrees(error)
                )
                self._sg_path_plan_active = True
                return None

        # The current observation was captured at the frontier-facing heading,
        # so its depth has already entered the runtime map through _SGEnvProxy.
        self._sg_exploration_blacklist.append(list(map(float, target_world)))
        history = list(target_info.get("approach_exploration_history") or [])
        history.append(
            {
                "event": "frontier_scanned",
                "step": int(kwargs.get("step", 0)),
                "world_xz": list(map(float, target_world)),
                "map_frame": map_frame,
            }
        )
        target_info["approach_exploration_history"] = history
        # For an action-lattice-triggered recovery, arriving at the safe
        # viewpoint and facing its unknown boundary is the evidence that makes
        # another attempt at the frozen SG goal meaningfully different.
        target_info[
            "approach_exploration_require_frontier_scan_before_resume"
        ] = False

        if config.connection_only:
            connection = _probe_frozen_old_map_connection(self, current)
            if connection is not None:
                _mark_connection_only_complete(
                    self,
                    target_info,
                    connection,
                    kwargs.get("step", 0),
                    logger,
                )
                return None
        else:
            final, route_anchors = _requery_approach_after_connection(
                self,
                target_info,
                current,
                original_approach,
            )
            if final is not None:
                _apply_final_approach(
                    self,
                    target_info,
                    final,
                    current,
                    route_anchors,
                    kwargs.get("step", 0),
                    logger,
                )
                return None

        selected = _select_frontier_plan(
            self, target_info, current, route_anchors, logger
        )
        if selected is None:
            if config.directional_probe:
                _request_directional_probe(
                    self,
                    target_info,
                    reason="no_safe_frontier_after_frontier_scan",
                    step=kwargs.get("step", 0),
                    map_frame=map_frame,
                )
                return None
            target_info["_sg_pending_abort_reason"] = (
                "exploration_no_safe_frontier_after_scan"
            )
            self._sg_path_plan_active = False
            return None
        _apply_frontier_plan(
            self, target_info, selected, kwargs.get("step", 0), logger
        )
        return None

    def exploration_get_action(self, **kwargs):
        target_info = kwargs.get("target_info") or {}
        if bool(target_info.pop("_sg_exploration_connection_only_complete", False)):
            self._sg_path_plan_active = False
            target_info["_sg_exploration_connection_only_stop_emitted"] = True
            return int(actions.STOP)
        if bool(target_info.get("approach_exploration_mode")):
            config = _runtime_config(self)
            current_budget_pose = self._sg_current_world_xz(kwargs)
            if current_budget_pose is not None:
                _update_motion_budget(target_info, current_budget_pose)
            actions_used = int(target_info.get('approach_exploration_action_count',0))
            distance_used = float(target_info.get('approach_exploration_distance_m',0))
            if actions_used >= config.maximum_actions or distance_used >= config.maximum_distance_m:
                target_info['_sg_pending_abort_reason'] = (
                    f'exploration_budget_actions_{actions_used}' if actions_used >= config.maximum_actions
                    else f'exploration_budget_distance_{distance_used:.2f}m')
                self._sg_path_plan_active = False
                return None
            forced = target_info.pop("_sg_exploration_forced_turn_action", None)
            if forced is not None:
                action = int(forced)
            elif bool(target_info.get("approach_exploration_deferred_reselect")):
                current = self._sg_current_world_xz(kwargs)
                if current is None:
                    target_info["_sg_pending_abort_reason"] = (
                        "exploration_deferred_reselect_missing_world_pose"
                    )
                    self._sg_path_plan_active = False
                    return None
                current = np.asarray(current, dtype=np.float64).reshape(2)
                failed_frame = int(
                    target_info.get(
                        "approach_exploration_deferred_map_frame", -1
                    )
                )
                map_frame = _query_map_frame(self)
                current_heading = kwargs.get("current_world_heading")
                if current_heading is None:
                    current_heading = getattr(
                        self, "_sg_world_heading_rad", None
                    )
                if current_heading is not None:
                    current_heading = float(
                        np.asarray(current_heading).reshape(-1)[0]
                    )
                config = _runtime_config(self)
                _settle_directional_probe_forward(
                    self, target_info, current, kwargs.get("step", 0)
                )
                if _recheck_original_target_after_directional_probe(
                    self,
                    target_info,
                    current,
                    original_approach,
                    map_frame=map_frame,
                    action_map_frame=failed_frame,
                    step=kwargs.get("step", 0),
                    logger=logger,
                ):
                    if config.connection_only:
                        target_info.pop(
                            "_sg_exploration_connection_only_complete", None
                        )
                        target_info[
                            "_sg_exploration_connection_only_stop_emitted"
                        ] = True
                        self._sg_path_plan_active = False
                        return int(actions.STOP)
                    return original_get_action(self, **kwargs)
                probe_requested = bool(
                    target_info.get(
                        "approach_exploration_directional_probe_requested"
                    ) or target_info.get('_sg_old_map_fallback', {}).get('active')
                    or target_info.get('_sg_old_map_fallback', {}).get('connected')
                )
                if probe_requested:
                    probe = _directional_probe_action(
                        self,
                        target_info,
                        current,
                        current_heading,
                        actions,
                        step=kwargs.get("step", 0),
                    )
                    if probe.get("action") is None:
                        from . import probe_recovery
                        if (str(probe.get("status", "")).endswith(
                            "budget_exhausted"
                        ) or probe_recovery.enabled(self) or config.old_map_fallback) and _recheck_original_target_after_directional_probe(
                            self,
                            target_info,
                            current,
                            original_approach,
                            map_frame=map_frame,
                            action_map_frame=failed_frame,
                            step=kwargs.get("step", 0),
                            logger=logger,
                            force=True,
                        ):
                            if config.connection_only:
                                target_info.pop(
                                    "_sg_exploration_connection_only_complete",
                                    None,
                                )
                                target_info[
                                    "_sg_exploration_connection_only_stop_emitted"
                                ] = True
                                self._sg_path_plan_active = False
                                return int(actions.STOP)
                            return original_get_action(self, **kwargs)
                        from .old_map_fallback import start_after_failure
                        probe = start_after_failure(
                            self, target_info, current, current_heading, actions,
                            step=kwargs.get('step',0), failure=probe) or probe
                        if probe.get('action') is None:
                            target_info["_sg_pending_abort_reason"] = (
                                "exploration_directional_probe_"
                                + str(probe.get("status", "failed"))
                            )
                            self._sg_path_plan_active = False
                            self._write_sg_decision(
                                kwargs.get("step"),
                                {
                                    "decision": "exploration_directional_probe_exhausted",
                                    **probe,
                                    "ordinary_vln_executed": False,
                                },
                            )
                            return None
                    action = int(probe["action"])
                    self._write_sg_decision(
                        kwargs.get("step"),
                        {
                            "decision": (
                                "exploration_directional_probe_"
                                + str(probe["action_kind"])
                            ),
                            **probe,
                            "ordinary_vln_executed": False,
                        },
                    )
                elif map_frame <= failed_frame:
                    action = _observation_turn_action(
                        actions, target_info, current_heading
                    )
                else:
                    if config.connection_only:
                        connection = _probe_frozen_old_map_connection(
                            self, current
                        )
                        if connection is not None:
                            _mark_connection_only_complete(
                                self,
                                target_info,
                                connection,
                                kwargs.get("step", 0),
                                logger,
                            )
                            target_info.pop(
                                "_sg_exploration_connection_only_complete",
                                None,
                            )
                            target_info[
                                "_sg_exploration_connection_only_stop_emitted"
                            ] = True
                            self._sg_path_plan_active = False
                            return int(actions.STOP)
                    route_anchors = list(
                        target_info.get(
                            "_sg_route_planning_anchors_world_xz"
                        )
                        or []
                    )
                    selected = _select_frontier_plan(
                        self,
                        target_info,
                        current,
                        route_anchors,
                        logger,
                    )
                    if selected is None:
                        scan_turns = int(
                            target_info.get(
                                "approach_exploration_deferred_scan_turns", 0
                            )
                        ) + 1
                        target_info[
                            "approach_exploration_deferred_scan_turns"
                        ] = scan_turns
                        if scan_turns < config.maximum_deferred_scan_turns:
                            target_info[
                                "approach_exploration_deferred_map_frame"
                            ] = map_frame
                            action = _observation_turn_action(
                                actions, target_info, current_heading
                            )
                            self._write_sg_decision(
                                kwargs.get("step"),
                                {
                                    "decision": (
                                        "exploration_deferred_reselect_"
                                        "continue_scan"
                                    ),
                                    "scan_turns": scan_turns,
                                    "map_frame": map_frame,
                                    "ordinary_vln_executed": False
                                    if isinstance(
                                        target_info.get(
                                            "_sg_dual_anchor_index"
                                        ),
                                        int,
                                    )
                                    else None,
                                },
                            )
                        else:
                            probe = _directional_probe_action(
                                self,
                                target_info,
                                current,
                                current_heading,
                                actions,
                                step=kwargs.get("step", 0),
                            )
                            if probe.get("action") is None:
                                from . import probe_recovery
                                if (str(probe.get("status", "")).endswith(
                                    "budget_exhausted"
                                ) or probe_recovery.enabled(self) or config.old_map_fallback) and _recheck_original_target_after_directional_probe(
                                    self,
                                    target_info,
                                    current,
                                    original_approach,
                                    map_frame=map_frame,
                                    action_map_frame=failed_frame,
                                    step=kwargs.get("step", 0),
                                    logger=logger,
                                    force=True,
                                ):
                                    if config.connection_only:
                                        target_info.pop(
                                            "_sg_exploration_connection_only_complete",
                                            None,
                                        )
                                        target_info[
                                            "_sg_exploration_connection_only_stop_emitted"
                                        ] = True
                                        self._sg_path_plan_active = False
                                        return int(actions.STOP)
                                    return original_get_action(self, **kwargs)
                                from .old_map_fallback import start_after_failure
                                probe = start_after_failure(
                                    self, target_info, current, current_heading, actions,
                                    step=kwargs.get('step',0), failure=probe) or probe
                                if probe.get('action') is None:
                                    if config.directional_probe:
                                        abort_reason = (
                                            "exploration_directional_probe_"
                                            + str(probe.get("status", "failed"))
                                        )
                                    else:
                                        abort_reason = (
                                            "exploration_no_action_lattice_reachable_"
                                            "frontier_after_deferred_scan"
                                        )
                                    target_info["_sg_pending_abort_reason"] = abort_reason
                                    self._sg_path_plan_active = False
                                    if config.directional_probe:
                                        self._write_sg_decision(
                                            kwargs.get("step"),
                                            {
                                                "decision": "exploration_directional_probe_exhausted",
                                                **probe,
                                                "ordinary_vln_executed": False,
                                            },
                                        )
                                    return None
                            action = int(probe["action"])
                            self._write_sg_decision(
                                kwargs.get("step"),
                                {
                                    "decision": (
                                        "exploration_directional_probe_"
                                        + str(probe["action_kind"])
                                    ),
                                    **probe,
                                    "ordinary_vln_executed": False,
                                },
                            )
                    else:
                        old_frontier = copy.deepcopy(
                            target_info.get("target_world_xz")
                        )
                        _apply_frontier_plan(
                            self,
                            target_info,
                            selected,
                            kwargs.get("step", 0),
                            logger,
                            current_world_xz=current,
                        )
                        self._write_sg_decision(
                            kwargs.get("step"),
                            {
                                "decision": (
                                    "exploration_deferred_frontier_committed"
                                ),
                                "old_frontier_world_xz": old_frontier,
                                "new_frontier_world_xz": target_info.get(
                                    "target_world_xz"
                                ),
                                "map_frame": map_frame,
                                "ordinary_vln_executed": False
                                if isinstance(
                                    target_info.get("_sg_dual_anchor_index"),
                                    int,
                                )
                                else None,
                            },
                        )
                        action = original_get_action(self, **kwargs)
            else:
                action = original_get_action(self, **kwargs)
            if action is not None:
                target_info["approach_exploration_action_count"] = int(
                    target_info.get("approach_exploration_action_count", 0)
                ) + 1
            return action
        return original_get_action(self, **kwargs)

    def exploration_aborted(self, **kwargs):
        target_info = kwargs.get("target_info") or {}
        reason = str(kwargs.get("reason", "unknown"))
        in_exploration = bool(target_info.get("approach_exploration_mode"))
        terminal_exploration = bool(
            reason.startswith("exploration_budget_")
            or reason == "exploration_no_safe_frontier_after_scan"
            or reason.startswith("exploration_directional_probe_")
            or reason.startswith(
                "exploration_no_action_lattice_reachable_frontier_"
            )
            or reason == "exploration_deferred_reselect_missing_world_pose"
        )
        config = _runtime_config(self)
        runtime_connectivity = _is_runtime_connectivity_abort(reason)
        runtime_action_lattice = bool(
            config.action_lattice_recovery
            and _is_runtime_action_lattice_abort(reason)
        )
        recoverable = bool(
            _runtime_enabled(self)
            and not terminal_exploration
            and (
                in_exploration
                or runtime_connectivity
                or runtime_action_lattice
            )
        )
        if not recoverable:
            return original_aborted(self, **kwargs)

        current = self._sg_current_world_xz(kwargs)
        if current is None:
            return original_aborted(self, **kwargs)
        current = np.asarray(current, dtype=np.float64).reshape(2)
        frontier = target_info.get("target_world_xz") if in_exploration else None
        if frontier is not None:
            self._sg_exploration_blacklist.append(list(map(float, frontier)))

        route_anchors = list(
            target_info.get("_sg_route_planning_anchors_world_xz") or []
        )
        # Once a frontier subgoal has failed, never select another one in this
        # same simulator step.  Consume the abort, execute a turn, wait for the
        # resulting RGB-D frame, and only then rank fresh candidates.
        if in_exploration:
            _defer_frontier_reselection(
                self,
                target_info,
                reason=reason,
                step=kwargs.get("step", 0),
                current_world_xz=current,
                actions=actions,
                logger=logger,
                frontier_was_active=True,
            )
            return True

        selected = _select_frontier_plan(
            self,
            target_info,
            current,
            route_anchors,
            logger,
        )
        if selected is None:
            trigger = (
                "runtime_v1_path_became_disconnected"
                if runtime_connectivity
                else "runtime_v1_action_lattice_unreachable"
            )
            target_info.update(
                {
                    "approach_exploration_mode": True,
                    "approach_exploration_version": IMPLEMENTATION_VERSION,
                    "approach_exploration_start_step": int(
                        kwargs.get("step", 0)
                    ),
                    "approach_exploration_action_count": 0,
                    "approach_exploration_distance_m": 0.0,
                    "approach_exploration_last_world_xz": list(
                        map(float, current)
                    ),
                    "approach_exploration_goal_points_world_xz": (
                        _candidate_goal_points(target_info, route_anchors)
                    ),
                    "approach_exploration_initial_planner_failure": {
                        "reason": trigger,
                        "runtime_abort_reason": reason,
                    },
                    "approach_exploration_history": [
                        {
                            "event": trigger + "_scan_before_reselect",
                            "step": int(kwargs.get("step", 0)),
                            "reason": reason,
                            "world_xz": list(map(float, current)),
                        }
                    ],
                    "approach_exploration_require_frontier_scan_before_resume": (
                        bool(runtime_action_lattice)
                    ),
                }
            )
            _defer_frontier_reselection(
                self,
                target_info,
                reason=reason,
                step=kwargs.get("step", 0),
                current_world_xz=current,
                actions=actions,
                logger=logger,
                frontier_was_active=False,
            )
            return True

        trigger = (
            "runtime_v1_path_became_disconnected"
            if runtime_connectivity
            else "runtime_v1_action_lattice_unreachable"
        )
        reference = _frontier_navigation_reference(
            self,
            selected,
            {
                "reason": trigger,
                "runtime_abort_reason": reason,
            },
        )
        for key, value in reference.items():
            target_info[key] = copy.deepcopy(value)
        target_info[
            "approach_exploration_require_frontier_scan_before_resume"
        ] = bool(runtime_action_lattice)
        target_info["approach_exploration_last_world_xz"] = list(
            map(float, current)
        )
        target_info["approach_exploration_history"] = [
            {
                "event": trigger,
                "step": int(kwargs.get("step", 0)),
                "reason": reason,
                "world_xz": list(map(float, current)),
            }
        ]

        self._sg_nav_reset_progress()
        self._sg_path_plan_active = False
        self._sg_world_plan_window = None
        self._sg_world_safe_mask = None
        self._sg_world_trajectory_probe_mask = None
        _apply_frontier_plan(
            self,
            target_info,
            selected,
            kwargs.get("step", 0),
            logger,
            current_world_xz=current,
        )
        self._write_sg_decision(
            kwargs.get("step"),
            {
                "decision": (
                    "exploration_subgoal_aborted_retry_frontier"
                    if in_exploration
                    else (
                        "runtime_disconnect_enter_exploration"
                        if runtime_connectivity
                        else "runtime_action_lattice_enter_exploration"
                    )
                ),
                "reason": reason,
                "selected_uid": target_info.get("uid") or target_info.get("node_id"),
                "frontier_world_xz": frontier,
                "new_frontier_world_xz": target_info.get("target_world_xz"),
                "exploration_actions": int(
                    target_info.get("approach_exploration_action_count", 0)
                ),
                "ordinary_vln_executed": False
                if isinstance(target_info.get("_sg_dual_anchor_index"), int)
                else None,
            },
        )
        logger.info(
            "[SG-EXPLORE] kept the SG UID and installed a frontier after "
            f"runtime geometric failure: reason={reason}, "
            f"retry={in_exploration}"
        )
        # The small parent-loop overlay treats True as "target recovered" and
        # keeps SG_NAV alive with target_info's newly installed frontier.
        return True

    evaluator_class.__init__ = exploration_init
    evaluator_class.reset = exploration_reset
    evaluator_class._sg_recover_no_traversable_topk = (
        exploration_no_traversable_topk
    )
    evaluator_class._sg_plan_v8_post_hit_approach = exploration_approach
    evaluator_class._sg_nav_reached_reason = exploration_reached
    evaluator_class._get_sg_world_path_action = exploration_get_action
    evaluator_class._on_sg_nav_aborted = exploration_aborted
    setattr(evaluator_class, _INSTALL_MARKER, True)
    return evaluator_class


__all__ = [
    "ExplorationConfig",
    "install_goal_directed_exploration",
    "_frontier_action_lattice_gate",
    "_commit_v1_plan_buffers",
    "_candidate_goal_points",
    "_failure_reasons",
    "_is_connectivity_failure",
    "_can_resume_original_target",
    "_is_runtime_action_lattice_abort",
    "_is_runtime_connectivity_abort",
    "_reset_world_path_handoff",
]
