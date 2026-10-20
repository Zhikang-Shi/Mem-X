"""Translate exploration state into trainer-neutral navigation commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from membridge.navigation import Action, HabitatActionSet

from .runtime import ExplorationRuntime, ExplorationState


@dataclass(frozen=True)
class ExplorationCommand:
    kind: str
    target_world_xz: tuple[float, float] | None = None
    action_id: int | None = None


class ExplorationHabitatAdapter:
    def __init__(
        self, runtime: ExplorationRuntime, actions: HabitatActionSet,
        *, scan_left: bool = True,
    ) -> None:
        self.runtime, self.actions = runtime, actions
        self.scan_action = Action.TURN_LEFT if scan_left else Action.TURN_RIGHT

    def command(self) -> ExplorationCommand:
        state = self.runtime.state
        if state == ExplorationState.NAVIGATING and self.runtime.plan and self.runtime.plan.candidate:
            candidate = self.runtime.plan.candidate
            return ExplorationCommand(
                "navigate_to_frontier", (candidate.world_x, candidate.world_z)
            )
        if state == ExplorationState.SCANNING:
            return ExplorationCommand(
                "scan_frontier", action_id=self.actions.encode(self.scan_action)
            )
        if state == ExplorationState.RETRY_TARGET:
            return ExplorationCommand("retry_original_target")
        if state in {ExplorationState.COMPLETE, ExplorationState.ABORTED}:
            return ExplorationCommand(state.value, action_id=self.actions.stop)
        return ExplorationCommand("idle")

    def record_navigation_step(self, measured_world_xz: Sequence[float]) -> ExplorationCommand:
        self.runtime.record_action(measured_world_xz)
        return self.command()

    def record_scan_step(self, *, new_map_evidence: bool) -> ExplorationCommand:
        self.runtime.record_scan_turn(new_map_evidence=new_map_evidence)
        return self.command()
