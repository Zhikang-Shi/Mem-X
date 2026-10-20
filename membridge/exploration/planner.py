"""Plan to ranked frontiers through the same navigation safety contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from membridge.navigation import ThetaStarResult, theta_star

from .frontier import FrontierCandidate, FrontierSelection, GoalDirectedFrontierSelector


@dataclass(frozen=True)
class ExplorationPlan:
    status: str
    candidate: FrontierCandidate | None
    selection: FrontierSelection
    path_world_grids: tuple[tuple[int, int], ...] = ()
    expanded_states: int = 0


class GoalDirectedExplorationPlanner:
    def __init__(
        self, metric_map, selector: GoalDirectedFrontierSelector | None = None,
        *, floor_id: int = 0, maximum_planner_candidates: int = 8,
    ) -> None:
        self.metric_map, self.selector = metric_map, selector or GoalDirectedFrontierSelector()
        self.floor_id, self.maximum_planner_candidates = int(floor_id), int(maximum_planner_candidates)

    def plan(
        self, start_world_xz: Sequence[float], goal_world_xz: Iterable[Sequence[float]], *,
        current_heading_rad: float | None = None,
        blacklist_world_xz: Iterable[Sequence[float]] = (),
    ) -> ExplorationPlan:
        goals = list(goal_world_xz)
        selection = self.selector.select(
            self.metric_map, start_world_xz, goals, current_heading_rad=current_heading_rad,
            blacklist_world_xz=blacklist_world_xz, floor_id=self.floor_id,
        )
        if selection.status != "ok":
            return ExplorationPlan(selection.status, None, selection)
        start_cell = self.metric_map.spec.world_to_cell(*start_world_xz)
        expanded = 0
        for candidate in selection.candidates[:self.maximum_planner_candidates]:
            target = (candidate.world_grid_x, candidate.world_grid_z)
            window = self.metric_map.planning_window(
                start_cell, target, floor_id=self.floor_id,
                padding_cells=self.selector.config.window_padding_cells,
                max_window_cells=self.selector.config.maximum_window_cells,
            )
            safe = window.safe_mask(self.selector.config.clearance_cells)
            for connector_cell in selection.start_connector_world_grids:
                x, z = window.global_to_local(*connector_cell)
                if 0 <= z < safe.shape[0] and 0 <= x < safe.shape[1]:
                    safe[z, x] = True
            result: ThetaStarResult = theta_star(
                safe, window.global_to_local(*start_cell), window.global_to_local(*target)
            )
            expanded += result.expanded_states
            if result.path:
                path = tuple(window.local_to_global(*point) for point in result.path)
                return ExplorationPlan("planned", candidate, selection, path, expanded)
        return ExplorationPlan("no_reachable_frontier", None, selection, (), expanded)
