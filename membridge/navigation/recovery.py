"""Bounded collision/stall recovery independent of a trainer implementation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .action_lattice import Action


class RecoveryStatus(str, Enum):
    IDLE = "idle"
    EXECUTING = "executing"
    REPLAN = "replan"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class RecoveryConfig:
    maximum_attempts: int = 3
    turn_prefix_actions: int = 2
    forward_probe_actions: int = 1

    def __post_init__(self) -> None:
        if self.maximum_attempts < 1 or self.turn_prefix_actions < 1 or self.forward_probe_actions < 0:
            raise ValueError("invalid recovery configuration")


@dataclass(frozen=True)
class RecoveryDecision:
    action: Action | None
    status: RecoveryStatus
    attempt: int


class CollisionRecoveryController:
    """Alternate turn direction, probe, then request a measured-pose replan."""

    def __init__(self, config: RecoveryConfig = RecoveryConfig()) -> None:
        self.config = config
        self.attempts = 0
        self._actions: list[Action] = []

    @property
    def active(self) -> bool:
        return bool(self._actions)

    def begin(self, *, prefer_left: bool | None = None) -> RecoveryDecision:
        if self.attempts >= self.config.maximum_attempts:
            return RecoveryDecision(None, RecoveryStatus.EXHAUSTED, self.attempts)
        self.attempts += 1
        left = (self.attempts % 2 == 1) if prefer_left is None else bool(prefer_left)
        turn = Action.TURN_LEFT if left else Action.TURN_RIGHT
        self._actions = [turn] * self.config.turn_prefix_actions
        self._actions += [Action.MOVE_FORWARD] * self.config.forward_probe_actions
        return self.next()

    def next(self) -> RecoveryDecision:
        if self._actions:
            return RecoveryDecision(self._actions.pop(0), RecoveryStatus.EXECUTING, self.attempts)
        return RecoveryDecision(None, RecoveryStatus.REPLAN, self.attempts)

    def reset(self) -> None:
        self.attempts, self._actions = 0, []
