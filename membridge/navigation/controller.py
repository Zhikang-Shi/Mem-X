"""Measured-pose executor for action-lattice plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .action_lattice import Action, ActionPlan


@dataclass(frozen=True)
class ControllerConfig:
    goal_tolerance_m: float = 0.35
    collision_translation_m: float = 0.04
    pose_deviation_m: float = 0.20
    max_stalled_forwards: int = 2


@dataclass(frozen=True)
class NavigationDecision:
    action: Action | None
    status: str
    replan_required: bool = False


class NavigationController:
    """Never dead-reckons: every decision is checked against the measured pose."""

    def __init__(self, config: ControllerConfig = ControllerConfig()) -> None:
        self.config = config
        self._actions: list[Action] = []
        self._last_pose: np.ndarray | None = None
        self._last_action: Action | None = None
        self._stalled_forwards = 0

    def install_plan(self, plan: ActionPlan, measured_position_xz: Sequence[float]) -> None:
        if plan.status != "planned":
            raise ValueError("only a planned action sequence can be installed")
        self._actions = list(plan.actions)
        np.asarray(measured_position_xz, dtype=float).reshape(2)
        self._last_pose, self._last_action, self._stalled_forwards = None, None, 0

    def decide(
        self, measured_position_xz: Sequence[float], target_world_xz: Sequence[float], *,
        expected_position_xz: Sequence[float] | None = None,
    ) -> NavigationDecision:
        position = np.asarray(measured_position_xz, dtype=float).reshape(2)
        target = np.asarray(target_world_xz, dtype=float).reshape(2)
        if np.linalg.norm(position - target) <= self.config.goal_tolerance_m:
            self._actions.clear()
            return NavigationDecision(Action.STOP, "goal_reached")
        if self._last_action == Action.MOVE_FORWARD and self._last_pose is not None:
            moved = float(np.linalg.norm(position - self._last_pose))
            self._stalled_forwards = self._stalled_forwards + 1 if moved < self.config.collision_translation_m else 0
            if self._stalled_forwards >= self.config.max_stalled_forwards:
                self._actions.clear()
                return NavigationDecision(None, "forward_collision_or_stall", True)
        if expected_position_xz is not None:
            expected = np.asarray(expected_position_xz, dtype=float).reshape(2)
            if np.linalg.norm(position - expected) > self.config.pose_deviation_m:
                self._actions.clear()
                return NavigationDecision(None, "measured_pose_deviated", True)
        if not self._actions:
            return NavigationDecision(None, "plan_exhausted", True)
        action = self._actions.pop(0)
        self._last_pose, self._last_action = position.copy(), action
        return NavigationDecision(action, "executing")
