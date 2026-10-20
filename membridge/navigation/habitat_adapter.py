"""Translate module navigation decisions to injected Habitat action IDs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .action_lattice import Action, ActionPlan
from .controller import NavigationController, NavigationDecision
from .recovery import CollisionRecoveryController, RecoveryStatus


def wrap_heading(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


@dataclass(frozen=True)
class HabitatActionSet:
    stop: int
    move_forward: int
    turn_left: int
    turn_right: int

    def __post_init__(self) -> None:
        values = (self.stop, self.move_forward, self.turn_left, self.turn_right)
        if len(set(map(int, values))) != 4:
            raise ValueError("Habitat action IDs must be unique")

    @classmethod
    def from_enum(cls, actions: Any) -> "HabitatActionSet":
        return cls(
            int(actions.STOP), int(actions.MOVE_FORWARD),
            int(actions.TURN_LEFT), int(actions.TURN_RIGHT),
        )

    def encode(self, action: Action) -> int:
        return {
            Action.STOP: self.stop,
            Action.MOVE_FORWARD: self.move_forward,
            Action.TURN_LEFT: self.turn_left,
            Action.TURN_RIGHT: self.turn_right,
        }[Action(action)]

    def decode(self, action_id: int) -> Action:
        reverse = {
            self.stop: Action.STOP,
            self.move_forward: Action.MOVE_FORWARD,
            self.turn_left: Action.TURN_LEFT,
            self.turn_right: Action.TURN_RIGHT,
        }
        if int(action_id) not in reverse:
            raise ValueError(f"unknown Habitat action ID: {action_id}")
        return reverse[int(action_id)]


@dataclass(frozen=True)
class HeadingAlignmentConfig:
    tolerance_deg: float = 15.0
    turn_angle_deg: float = 30.0
    maximum_turns: int = 12

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.tolerance_deg))
            or not math.isfinite(float(self.turn_angle_deg))
            or self.tolerance_deg < 0 or self.turn_angle_deg <= 0
            or self.maximum_turns < 1
        ):
            raise ValueError("invalid heading-alignment configuration")


class HeadingAligner:
    def __init__(self, config: HeadingAlignmentConfig = HeadingAlignmentConfig()) -> None:
        self.config = config

    def plan(self, current_heading_rad: float, target_heading_rad: float) -> tuple[Action, ...]:
        delta = wrap_heading(target_heading_rad - current_heading_rad)
        tolerance = math.radians(self.config.tolerance_deg)
        if abs(delta) <= tolerance:
            return ()
        count = min(
            self.config.maximum_turns,
            int(math.ceil((abs(delta) - tolerance) / math.radians(self.config.turn_angle_deg))),
        )
        action = Action.TURN_LEFT if delta > 0 else Action.TURN_RIGHT
        return (action,) * count


@dataclass(frozen=True)
class StopPolicy:
    goal_tolerance_m: float = 0.35
    heading_tolerance_deg: float = 15.0
    require_heading_alignment: bool = True

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.goal_tolerance_m))
            or not math.isfinite(float(self.heading_tolerance_deg))
            or self.goal_tolerance_m < 0 or self.heading_tolerance_deg < 0
        ):
            raise ValueError("invalid stop-policy tolerance")

    def should_stop(
        self, distance_to_navigation_target_m: float, *, heading_error_rad: float | None = None,
        target_valid: bool = True,
    ) -> bool:
        if not target_valid or not math.isfinite(float(distance_to_navigation_target_m)):
            return False
        if float(distance_to_navigation_target_m) > self.goal_tolerance_m:
            return False
        return (
            not self.require_heading_alignment
            or (
                heading_error_rad is not None
                and abs(wrap_heading(heading_error_rad)) <= math.radians(self.heading_tolerance_deg)
            )
        )


@dataclass(frozen=True)
class HabitatNavigationDecision:
    action_id: int | None
    status: str
    replan_required: bool = False
    metadata: Mapping[str, Any] | None = None


class HabitatNavigationAdapter:
    """Keep Habitat IDs out of the measured-pose navigation controller."""

    def __init__(
        self, controller: NavigationController, actions: HabitatActionSet,
        *, heading_aligner: HeadingAligner | None = None,
        stop_policy: StopPolicy = StopPolicy(),
        recovery: CollisionRecoveryController | None = None,
    ) -> None:
        self.controller, self.actions = controller, actions
        self.heading_aligner = heading_aligner or HeadingAligner()
        self.stop_policy = stop_policy
        self.recovery = recovery or CollisionRecoveryController()
        self._alignment_actions: list[Action] = []
        self._recovering = False

    def install_plan(self, plan: ActionPlan, measured_position_xz) -> None:
        self._alignment_actions.clear()
        self._recovering = False
        self.controller.install_plan(plan, measured_position_xz)

    def reset_recovery(self) -> None:
        """Call after verified forward progress or a new navigation target."""

        self._recovering = False
        self.recovery.reset()

    def _recovery_decision(self, *, begin: bool = False) -> HabitatNavigationDecision:
        recovery = self.recovery.begin() if begin else self.recovery.next()
        if recovery.status == RecoveryStatus.EXECUTING and recovery.action is not None:
            self._recovering = True
            return HabitatNavigationDecision(
                self.actions.encode(recovery.action), "collision_recovery", False,
                {"recovery_attempt": recovery.attempt},
            )
        self._recovering = False
        return HabitatNavigationDecision(
            None,
            "collision_recovery_exhausted" if recovery.status == RecoveryStatus.EXHAUSTED
            else "collision_recovery_replan",
            True, {"recovery_attempt": recovery.attempt},
        )

    def decide(
        self, measured_position_xz, target_world_xz, *, expected_position_xz=None,
        current_heading_rad: float | None = None,
        target_heading_rad: float | None = None,
        target_valid: bool = True,
    ) -> HabitatNavigationDecision:
        if self._recovering:
            return self._recovery_decision()
        if self._alignment_actions:
            action = self._alignment_actions.pop(0)
            return HabitatNavigationDecision(self.actions.encode(action), "aligning_heading")
        decision: NavigationDecision = self.controller.decide(
            measured_position_xz, target_world_xz,
            expected_position_xz=expected_position_xz,
        )
        if decision.status == "forward_collision_or_stall":
            return self._recovery_decision(begin=True)
        if decision.action == Action.STOP:
            dx = float(measured_position_xz[0]) - float(target_world_xz[0])
            dz = float(measured_position_xz[1]) - float(target_world_xz[1])
            distance = math.hypot(dx, dz)
            heading_error = None
            if current_heading_rad is not None and target_heading_rad is not None:
                heading_error = wrap_heading(target_heading_rad - current_heading_rad)
                self._alignment_actions = list(
                    self.heading_aligner.plan(current_heading_rad, target_heading_rad)
                )
                if self._alignment_actions:
                    action = self._alignment_actions.pop(0)
                    return HabitatNavigationDecision(
                        self.actions.encode(action), "aligning_heading", metadata={
                            "distance_to_target_m": distance,
                            "heading_error_rad": heading_error,
                        },
                    )
            if not self.stop_policy.should_stop(
                distance, heading_error_rad=heading_error, target_valid=target_valid
            ):
                return HabitatNavigationDecision(
                    None, "stop_policy_not_satisfied", True,
                    {"distance_to_target_m": distance, "target_valid": bool(target_valid)},
                )
        return HabitatNavigationDecision(
            self.actions.encode(decision.action) if decision.action is not None else None,
            decision.status, decision.replan_required,
        )
