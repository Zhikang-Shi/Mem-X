"""Navigation kernels shared by every model.

Theta*, world-planner audit, path smoothing, and the local action lattice
stay in MemBridge. Model-specific semantic mirroring lives with that model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SEMANTIC_MODE_ENV = "MEMORY_MODULE_SEMANTIC_RUNTIME_MODE"
NAVIGATION_MODE_ENV = "MEMORY_MODULE_NAVIGATION_RUNTIME_MODE"
FAIL_CLOSED_ENV = "MEMORY_MODULE_ADAPTER_FAIL_CLOSED"


def _enabled(value: str | None, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _mode(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in {"off", "shadow", "authoritative"}:
        raise ValueError(f"{name} must be off, shadow, or authoritative; got {value!r}")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _runtime(owner: Any) -> Any:
    return getattr(owner, "inherit_sg", owner)


def _audit_root(owner: Any) -> Path | None:
    runtime = _runtime(owner)
    pipeline = getattr(runtime, "pipeline", None)
    path = getattr(pipeline, "run_dir", None) or getattr(runtime, "output_root", None)
    return Path(path) if path else None


def _write_audit(owner: Any, section: str, values: Mapping[str, Any]) -> Path | None:
    root = _audit_root(owner)
    if root is None:
        return None
    root.mkdir(parents=True, exist_ok=True)
    path = root / "memory_module_runtime_audit.json"
    payload = {}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
    payload.update(
        schema_version="memory_module_unilavira_runtime_audit_v1",
        adapter_file=str(Path(__file__).resolve()),
    )
    payload.setdefault(section, {}).update(_plain(values))
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path



def _navigation_stats(evaluator: Any) -> dict[str, Any]:
    stats = getattr(evaluator, "_memory_module_navigation_stats", None)
    if stats is None:
        stats = {
            "mode": _mode(NAVIGATION_MODE_ENV, "authoritative"),
            "theta_star_calls": 0,
            "world_planner_audit_calls": 0,
            "smoothing_calls": 0,
            "local_action_lattice_calls": 0,
        }
        evaluator._memory_module_navigation_stats = stats
    return stats


def install_navigation_runtime_adapter(evaluator_class: type, habitat_actions: Any) -> dict[str, str]:
    """Delegate compatible navigation kernels while retaining Habitat state."""

    if getattr(evaluator_class, "_memory_module_navigation_adapter_installed", False):
        return {}
    legacy_theta = evaluator_class._sg_theta_star
    legacy_world_plan = evaluator_class._sg_plan_world_path
    legacy_select_path = evaluator_class._sg_select_execution_world_path
    legacy_local_plan = evaluator_class._sg_plan_local_action_sequence

    def theta_star_adapter(self, safe_mask, start, goal):
        from membridge.navigation import theta_star

        mode = _mode(NAVIGATION_MODE_ENV, "authoritative")
        if mode == "off":
            return legacy_theta(self, safe_mask, start, goal)
        legacy = legacy_theta(self, safe_mask, start, goal) if mode == "shadow" else None
        result = theta_star(
            safe_mask, start, goal,
            max_states=int(os.environ.get("LIFELONG_SG_THETA_MAX_STATES", "250000")),
        )
        stats = _navigation_stats(self)
        stats["theta_star_calls"] += 1
        stats["last_theta_status"] = result.reason
        if mode == "shadow":
            stats["last_theta_path_match"] = _plain(legacy[0]) == _plain(result.path)
        _write_audit(self, "navigation", stats)
        return legacy if mode == "shadow" else (result.path, int(result.expanded_states))

    def world_plan_adapter(self, raw_target_world_xz, current_world_xz, *args, **kwargs):
        result = legacy_world_plan(self, raw_target_world_xz, current_world_xz, *args, **kwargs)
        mode = _mode(NAVIGATION_MODE_ENV, "authoritative")
        if mode == "off":
            return result
        stats = _navigation_stats(self)
        try:
            from membridge.navigation import WorldMemoryPlanner

            metric_map = self._sg_world_metric_map()
            graph = self._sg_trajectory_graph()
            resolution = float(metric_map.spec.resolution_m)
            planner = WorldMemoryPlanner(
                metric_map, graph,
                clearance_cells=float(os.environ.get("LIFELONG_SG_DISCRETE_CLEARANCE_CELLS", "2.5")),
                trajectory_execution_slack_cells=self._sg_trajectory_execution_slack_cells(metric_map.spec),
                target_snap_radius_m=float(getattr(self, "sg_snap_radius_cells", 60)) * resolution,
                window_padding_cells=int(os.environ.get("LIFELONG_SG_WORLD_WINDOW_PADDING_CELLS", "40")),
                maximum_window_cells=int(os.environ.get("LIFELONG_SG_WORLD_MAX_WINDOW_CELLS", "4000000")),
            )
            audit = planner.plan(
                current_world_xz, raw_target_world_xz,
                exact_target=bool(kwargs.get("exact", False)),
                connector_validator=getattr(self, "_sg_validate_warm_start_connector", None),
            )
            stats["world_planner_audit_calls"] += 1
            stats["last_world_planner_status"] = audit.status
            stats["last_world_planner_path_length_m"] = audit.path_length_m
        except Exception as exc:
            stats["last_world_planner_status"] = f"error:{type(exc).__name__}"
            stats["last_world_planner_error"] = str(exc)
            if _enabled(os.environ.get(FAIL_CLOSED_ENV), True):
                raise
        _write_audit(self, "navigation", stats)
        return result

    def select_path_adapter(self, *args, **kwargs):
        selected = legacy_select_path(self, *args, **kwargs)
        mode = _mode(NAVIGATION_MODE_ENV, "authoritative")
        path = list((selected or {}).get("path") or [])
        if mode == "off" or len(path) < 3:
            return selected
        from membridge.navigation import PathSmoothingConfig, smooth_safe_path

        window, safe_mask = getattr(self, "_sg_world_plan_window", None), getattr(self, "_sg_world_safe_mask", None)
        metric_map = self._sg_world_metric_map()
        if window is None or safe_mask is None or metric_map is None:
            return selected
        smoothed = smooth_safe_path(
            path, resolution_m=float(metric_map.spec.resolution_m),
            segment_is_safe=lambda left, right: self._sg_float_segment_is_safe(
                safe_mask, self._sg_window_point(window, left), self._sg_window_point(window, right)
            ),
            config=PathSmoothingConfig(),
        )
        stats = _navigation_stats(self)
        stats["smoothing_calls"] += 1
        stats["last_smoothing"] = smoothed["metadata"]
        if mode == "authoritative" and smoothed["metadata"]["applied"]:
            selected["path"] = smoothed["path"]
            selected["path_length_cells"] = float(self._sg_path_cumulative(smoothed["path"])[-1])
            selected["memory_module_smoothing"] = smoothed["metadata"]
        _write_audit(self, "navigation", stats)
        return selected

    def local_plan_adapter(self, position, current_heading, target, window, safe_mask, target_info):
        legacy = legacy_local_plan(self, position, current_heading, target, window, safe_mask, target_info)
        mode = _mode(NAVIGATION_MODE_ENV, "authoritative")
        if mode == "off":
            return legacy
        from membridge.navigation import Action, ActionLatticeConfig, plan_action_lattice

        metric_map = self._sg_world_metric_map()
        resolution = float(metric_map.spec.resolution_m)
        config = ActionLatticeConfig(
            forward_step_cells=float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE) / resolution,
            turn_angle_deg=float(self.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE),
            goal_tolerance_cells=max(0.5, float(getattr(self, "sg_local_action_goal_tolerance_m", 0.15)) / resolution),
            radius_cells=max(1.0, float(getattr(self, "sg_local_action_radius_m", 2.5)) / resolution),
            max_states=int(getattr(self, "sg_local_action_max_states", 12000)),
            max_actions=int(os.environ.get("LIFELONG_SG_LOCAL_ACTION_MAX_ACTIONS", "96")),
        )
        planned = plan_action_lattice(
            position, current_heading, target,
            segment_is_safe=lambda left, right: self._sg_float_segment_is_safe(
                safe_mask, self._sg_window_point(window, left), self._sg_window_point(window, right)
            ),
            config=config,
        )
        stats = _navigation_stats(self)
        stats["local_action_lattice_calls"] += 1
        stats["last_local_action_status"] = planned.status
        _write_audit(self, "navigation", stats)
        if mode != "authoritative" or planned.status != "planned" or legacy is None:
            return legacy
        action_map = {
            Action.MOVE_FORWARD: int(habitat_actions.MOVE_FORWARD),
            Action.TURN_LEFT: int(habitat_actions.TURN_LEFT),
            Action.TURN_RIGHT: int(habitat_actions.TURN_RIGHT),
            Action.STOP: int(habitat_actions.STOP),
        }
        target_info["_sg_local_action_status"] = "planned_by_memory_module_validated_by_legacy"
        return {
            "actions": [action_map[action] for action in planned.actions],
            "expanded_states": int(planned.expanded_states),
            "goal_tolerance_cells": float(config.goal_tolerance_cells),
            "goal_distance_cells": float(planned.goal_distance_cells),
        }

    evaluator_class._sg_theta_star = theta_star_adapter
    evaluator_class._sg_plan_world_path = world_plan_adapter
    evaluator_class._sg_select_execution_world_path = select_path_adapter
    evaluator_class._sg_plan_local_action_sequence = local_plan_adapter
    evaluator_class._memory_module_navigation_adapter_installed = True
    return {
        "INHeritSGZeroShotVlnEvaluatorMP.navigation": (
            "memory_module theta/world-planner/smoothing/action-lattice adapter"
        )
    }


