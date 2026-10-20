"""Bounded exploration admission, budget and handoff state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np

from .planner import ExplorationPlan, GoalDirectedExplorationPlanner

CONNECTIVITY_FAILURE_MARKERS = (
    "warm_start_not_free_in_world_memory", "warm_start_not_clearance_safe_in_world_memory",
    "no_world_reachable_target_within_snap_radius", "world_theta_star_failed",
    "start_not_connected_to_safe_space", "no_reachable_target_within_snap_radius", "theta_star_failed",
)
ACTION_LATTICE_MARKER = "local_action_lattice_unreachable_after_"


class ExplorationState(str, Enum):
    IDLE = "idle"
    NAVIGATING = "navigating_to_frontier"
    SCANNING = "scanning_frontier"
    RETRY_TARGET = "retry_original_target"
    COMPLETE = "complete"
    ABORTED = "aborted"


@dataclass(frozen=True)
class ExplorationConfig:
    enabled: bool = False
    action_lattice_recovery: bool = True
    maximum_actions: int = 100
    maximum_distance_m: float = 20.0
    frontier_goal_tolerance_m: float = 0.15
    maximum_scan_turns: int = 12


def _failure_reasons(payload: Any) -> set[str]:
    if isinstance(payload, str):
        return {payload}
    if isinstance(payload, dict):
        result = {str(payload["reason"])} if payload.get("reason") else set()
        for value in payload.values():
            result |= _failure_reasons(value)
        return result
    if isinstance(payload, (list, tuple)):
        result = set()
        for value in payload:
            result |= _failure_reasons(value)
        return result
    return set()


class ExplorationRuntime:
    """Exploration is admitted only for proven geometry/lattice disconnection."""

    def __init__(self, planner: GoalDirectedExplorationPlanner, config: ExplorationConfig = ExplorationConfig()) -> None:
        self.planner, self.config = planner, config
        self.state, self.plan = ExplorationState.IDLE, None
        self.actions_used, self.distance_used_m, self.scan_turns = 0, 0.0, 0
        self._last_position: np.ndarray | None = None
        self.blacklist: list[tuple[float, float]] = []

    def admissible(self, failure: Any) -> bool:
        if not self.config.enabled:
            return False
        reasons = _failure_reasons(failure) | {str(failure or "")}
        return any(marker in reason for reason in reasons for marker in CONNECTIVITY_FAILURE_MARKERS) or (
            self.config.action_lattice_recovery and any(ACTION_LATTICE_MARKER in reason for reason in reasons)
        )

    def begin(
        self, failure: Any, start_world_xz: Sequence[float], goal_world_xz: Iterable[Sequence[float]], *,
        current_heading_rad: float | None = None,
    ) -> ExplorationPlan | None:
        if not self.admissible(failure):
            return None
        self.plan = self.planner.plan(
            start_world_xz, goal_world_xz, current_heading_rad=current_heading_rad,
            blacklist_world_xz=self.blacklist,
        )
        self.state = ExplorationState.NAVIGATING if self.plan.status == "planned" else ExplorationState.ABORTED
        self._last_position = np.asarray(start_world_xz, dtype=float).reshape(2)
        return self.plan

    def reset(self) -> None:
        self.state, self.plan = ExplorationState.IDLE, None
        self.actions_used, self.distance_used_m, self.scan_turns = 0, 0.0, 0
        self._last_position = None
        self.blacklist.clear()

    def record_action(self, measured_world_xz: Sequence[float]) -> ExplorationState:
        current = np.asarray(measured_world_xz, dtype=float).reshape(2)
        self.actions_used += 1
        if self._last_position is not None:
            self.distance_used_m += float(np.linalg.norm(current - self._last_position))
        self._last_position = current
        if self.actions_used > self.config.maximum_actions or self.distance_used_m > self.config.maximum_distance_m:
            self.state = ExplorationState.ABORTED
        elif self.state == ExplorationState.NAVIGATING and self.plan and self.plan.candidate:
            target = np.asarray([self.plan.candidate.world_x, self.plan.candidate.world_z])
            if np.linalg.norm(current - target) <= self.config.frontier_goal_tolerance_m:
                self.state, self.scan_turns = ExplorationState.SCANNING, 0
        return self.state

    def record_scan_turn(self, *, new_map_evidence: bool) -> ExplorationState:
        if self.state != ExplorationState.SCANNING:
            return self.state
        self.scan_turns += 1
        if new_map_evidence or self.scan_turns >= self.config.maximum_scan_turns:
            if self.plan and self.plan.candidate:
                self.blacklist.append((self.plan.candidate.world_x, self.plan.candidate.world_z))
            self.state = ExplorationState.RETRY_TARGET
        return self.state

    def target_retry_succeeded(self, succeeded: bool) -> ExplorationState:
        self.state = ExplorationState.COMPLETE if succeeded else ExplorationState.IDLE
        return self.state
