"""Habitat integration for goal-directed SG connectivity exploration.

This module deliberately lives outside :mod:`memory_module.exploration`.
The latter remains a trainer-neutral planning/state-machine package; this
adapter binds it to the mutable Level-2 Habitat memory and to the existing SG
world-path controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from membridge.exploration import (
    ExplorationConfig,
    ExplorationRuntime,
    ExplorationState,
    FrontierConfig,
    GoalDirectedExplorationPlanner,
    GoalDirectedFrontierSelector,
)


@dataclass(frozen=True)
class FrontierExecutionContext:
    """A core exploration plan plus the masks needed by Habitat control."""

    path_world_grids: tuple[tuple[int, int], ...]
    target_world_grid: tuple[int, int]
    target_world_xz: tuple[float, float]
    scan_heading_rad: float
    window: Any
    safe_mask: np.ndarray
    plan_summary: dict[str, Any]


class HabitatSGExplorationAdapter:
    """Run bounded exploration while the normal online memory keeps writing.

    The adapter never writes the map, trajectory graph, or semantic graph
    itself.  Every exploration action is executed by the ordinary evaluator,
    whose environment proxy sends its RGB-D observation to
    ``OnlineINHeritSGBuilder.process_observation``.  This preserves the formal
    EP0 -> M0 -> EP1 -> M1 -> EP2 memory contract, including asynchronous SG
    fusion and the final queue drain.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        maximum_actions: int = 100,
        maximum_distance_m: float = 20.0,
        maximum_scan_turns: int = 12,
        maximum_frontiers: int = 4,
        frontier_goal_tolerance_m: float = 0.15,
        frontier_config: FrontierConfig | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.runtime_config = ExplorationConfig(
            enabled=self.enabled,
            maximum_actions=max(1, int(maximum_actions)),
            maximum_distance_m=max(0.1, float(maximum_distance_m)),
            frontier_goal_tolerance_m=max(
                0.0, float(frontier_goal_tolerance_m)
            ),
            maximum_scan_turns=max(1, int(maximum_scan_turns)),
        )
        self.frontier_config = frontier_config or FrontierConfig()
        self.maximum_frontiers = max(1, int(maximum_frontiers))
        self.runtime: ExplorationRuntime | None = None
        self._metric_map = None
        self._episode_id: str | None = None
        self._task_key: str | None = None
        self._frontier_count = 0
        self._episode_start_step: int | None = None
        self._last_transition_step: int | None = None
        self._map_stats_before_scan: dict[str, Any] | None = None
        self._builder_stats_before_scan: dict[str, int] | None = None

    @property
    def active(self) -> bool:
        return bool(
            self.runtime is not None
            and self.runtime.state
            in {
                ExplorationState.NAVIGATING,
                ExplorationState.SCANNING,
                ExplorationState.RETRY_TARGET,
            }
        )

    @property
    def retry_pending(self) -> bool:
        return bool(
            self.runtime is not None
            and self.runtime.state == ExplorationState.RETRY_TARGET
        )

    def matches(self, episode_id: str, task_key: str) -> bool:
        return bool(
            self.active
            and self._episode_id == str(episode_id)
            and self._task_key == str(task_key)
        )

    def reset(self) -> None:
        if self.runtime is not None:
            self.runtime.reset()
        self.runtime = None
        self._metric_map = None
        self._episode_id = None
        self._task_key = None
        self._frontier_count = 0
        self._episode_start_step = None
        self._last_transition_step = None
        self._map_stats_before_scan = None
        self._builder_stats_before_scan = None

    @staticmethod
    def online_memory_is_mutable(builder: Any) -> bool:
        """Reject frozen replays: formal exploration must also update SG."""

        if builder is None:
            return False
        if bool(getattr(builder, "preload_query_only", False)) and bool(
            getattr(builder, "_preloaded_session", False)
        ) and not bool(getattr(builder, "preload_world_map_only", False)):
            return False
        return bool(
            getattr(builder, "enabled", False)
            and not getattr(builder, "failed", False)
            and getattr(builder, "_episode_active", False)
            and getattr(builder, "world_memory", None) is not None
            and getattr(builder, "pipeline", None) is not None
        )

    @staticmethod
    def _builder_stats(builder: Any) -> dict[str, int]:
        pipeline = getattr(builder, "pipeline", None)
        memory_kb = getattr(pipeline, "memory_kb", ()) if pipeline else ()
        return {
            "frame_idx": int(getattr(builder, "frame_idx", 0)),
            "world_observation_count": int(
                getattr(builder, "_world_observation_count", 0)
            ),
            "trajectory_nodes": int(
                len(getattr(getattr(builder, "world_memory", None), "trajectory", ()).nodes)
                if getattr(getattr(builder, "world_memory", None), "trajectory", None)
                is not None
                else 0
            ),
            "semantic_objects_current": int(len(memory_kb or ())),
            "confirmed_keyframes_current": int(
                getattr(builder, "_episode_confirmed_keyframes", 0)
            ),
        }

    @staticmethod
    def _delta(after: dict[str, Any], before: dict[str, Any] | None) -> dict[str, Any]:
        before = before or {}
        result = {}
        for key, value in after.items():
            old = before.get(key)
            if isinstance(value, (int, float)) and isinstance(old, (int, float)):
                result[key] = value - old
        return result

    def _new_runtime(self, metric_map: Any) -> ExplorationRuntime:
        selector = GoalDirectedFrontierSelector(self.frontier_config)
        planner = GoalDirectedExplorationPlanner(metric_map, selector)
        return ExplorationRuntime(planner, self.runtime_config)

    @staticmethod
    def _certify_live_agent_footprint(
        metric_map: Any,
        position_world_xz: Sequence[float],
        *,
        agent_radius_m: float = 0.10,
    ) -> int:
        """Seed planning from the physically occupied Habitat agent pose.

        A protocol EP1 start may fall in a raster hole just outside M0 even
        though Habitat has placed the capsule on its live navmesh.  Certifying
        only the capsule footprint supplies the bounded connector expected by
        the neutral frontier selector; subsequent free space still has to be
        observed by RGB-D exploration.
        """

        position = np.asarray(position_world_xz, dtype=float).reshape(2)
        if not np.isfinite(position).all():
            raise ValueError("live agent footprint requires a finite pose")
        spec = metric_map.spec
        center_x, center_z = spec.world_to_cell(*position)
        radius_cells = max(0.01, float(agent_radius_m)) / float(
            spec.resolution_m
        ) + math.sqrt(0.5)
        extent = int(math.ceil(radius_cells))
        cells = []
        for dz in range(-extent, extent + 1):
            for dx in range(-extent, extent + 1):
                if dx * dx + dz * dz <= radius_cells * radius_cells + 1e-9:
                    cells.append((center_x + dx, center_z + dz))
        result = metric_map.set_classified_cells(
            free_cells=cells, occupied_cells=(), floor_id=0
        )
        return int(result["free_cells"])

    def _certify_persisted_trajectory_corridor(
        self,
        metric_map: Any,
        trajectory_graph: Any,
        start_world_xz: Sequence[float],
        goal_world_xz: Sequence[Sequence[float]],
        connector_validator: Callable[..., Any] | None,
    ) -> dict[str, Any]:
        """Fuse the physically traversed trajectory certificate into the map.

        Persisted trajectory edges are stronger free-space evidence than a
        stale probabilistic obstacle raster: an agent capsule actually moved
        along every recorded edge.  The optional live-navmesh validator is
        still required for the bounded join from the current protocol start
        to that prior trajectory.  This is an adapter-side map update; the
        trainer-neutral exploration package remains unchanged.
        """

        status: dict[str, Any] = {
            "trajectory_corridor_status": "missing_trajectory_graph",
            "trajectory_corridor_fused_free_cells": 0,
        }
        if trajectory_graph is None:
            return status
        goals = np.asarray(goal_world_xz, dtype=float).reshape(-1, 2)
        start = np.asarray(start_world_xz, dtype=float).reshape(2)
        start_cell = metric_map.spec.world_to_cell(*start)
        closest_goal = min(
            goals,
            key=lambda point: float(np.linalg.norm(point - start)),
        )
        goal_cell = metric_map.spec.world_to_cell(*closest_goal)
        window = metric_map.planning_window(
            start_cell,
            goal_cell,
            floor_id=0,
            padding_cells=self.frontier_config.window_padding_cells,
            max_window_cells=self.frontier_config.maximum_window_cells,
        )
        # A continuous swept trajectory rasterized onto square cells can
        # otherwise collapse back to a diagonal-only centerline after the
        # planner applies its clearance erosion.  The selector's 8-connected
        # component accepts that corner contact, whereas collision-safe
        # Theta* correctly rejects it.  Half a raster cell is the bounded
        # digitization allowance needed to preserve a non-corner-cutting
        # route; it does not expose unknown space beyond the trajectory tube.
        raster_margin_cells = 0.5
        certificate_clearance_cells = (
            self.frontier_config.clearance_cells + raster_margin_cells
        )
        corridor, _, metadata = trajectory_graph.certified_masks(
            window,
            metric_map.spec,
            clearance_cells=certificate_clearance_cells,
            floor_id=0,
            warm_start_world_xz=start,
            max_terminal_connector_m=(
                self.frontier_config.start_connector_max_distance_m
            ),
            connector_validator=connector_validator,
        )
        world_cells = [
            window.local_to_global(int(local_x), int(local_z))
            for local_z, local_x in np.argwhere(corridor)
        ]
        fused = metric_map.set_classified_cells(
            free_cells=world_cells,
            occupied_cells=(),
            floor_id=0,
        )
        status.update(
            {
                "trajectory_corridor_status": "fused",
                "trajectory_corridor_fused_free_cells": int(
                    fused["free_cells"]
                ),
                **metadata,
                "trajectory_corridor_planner_clearance_cells": float(
                    self.frontier_config.clearance_cells
                ),
                "trajectory_corridor_raster_margin_cells": float(
                    raster_margin_cells
                ),
                "trajectory_corridor_certificate_clearance_cells": float(
                    certificate_clearance_cells
                ),
            }
        )
        return status

    def _budget_exhausted(self, step: int) -> bool:
        if self._episode_start_step is None:
            return False
        elapsed = max(0, int(step) - int(self._episode_start_step))
        return elapsed >= int(self.runtime_config.maximum_actions)

    def plan_frontier(
        self,
        *,
        episode_id: str,
        task_key: str,
        metric_map: Any,
        online_builder: Any,
        failure: Any,
        start_world_xz: Sequence[float],
        goal_world_xz: Iterable[Sequence[float]],
        trajectory_graph: Any | None = None,
        connector_validator: Callable[..., Any] | None = None,
        step: int,
        current_heading_rad: float | None = None,
    ) -> tuple[FrontierExecutionContext | None, dict[str, Any]]:
        """Plan the first/next frontier, preserving the core runtime budget."""

        status = {
            "enabled": self.enabled,
            "formal_memory_mutable": self.online_memory_is_mutable(
                online_builder
            ),
            "memory_update_mode": (
                "world_map_and_trajectory_sync_semantic_sg_async"
            ),
        }
        if not self.enabled:
            status["reason"] = "exploration_disabled"
            return None, status
        if not status["formal_memory_mutable"]:
            status["reason"] = "exploration_requires_mutable_online_sg"
            return None, status

        goals = [tuple(map(float, point)) for point in goal_world_xz]
        if not goals:
            status["reason"] = "missing_semantic_goal_points"
            return None, status
        try:
            status["trajectory_certificate"] = (
                self._certify_persisted_trajectory_corridor(
                    metric_map,
                    trajectory_graph,
                    start_world_xz,
                    goals,
                    connector_validator,
                )
            )
        except Exception as exc:
            status["trajectory_certificate"] = {
                "trajectory_corridor_status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        status["certified_live_agent_footprint_cells"] = (
            self._certify_live_agent_footprint(
                metric_map, start_world_xz
            )
        )

        identity_changed = bool(
            self.runtime is None
            or self._metric_map is not metric_map
            or self._episode_id != str(episode_id)
            or self._task_key != str(task_key)
        )
        if identity_changed:
            self.reset()
            self.runtime = self._new_runtime(metric_map)
            self._metric_map = metric_map
            self._episode_id = str(episode_id)
            self._task_key = str(task_key)
            self._episode_start_step = int(step)
        elif self.runtime.state == ExplorationState.RETRY_TARGET:
            status["scan_memory_update"] = self.scan_memory_update(
                metric_map, online_builder
            )
            self.runtime.target_retry_succeeded(False)
        elif self.runtime.state == ExplorationState.ABORTED:
            status.update(
                {
                    "reason": "exploration_runtime_budget_aborted",
                    "runtime_actions_used": int(self.runtime.actions_used),
                    "runtime_distance_used_m": float(
                        self.runtime.distance_used_m
                    ),
                }
            )
            return None, status

        if self._budget_exhausted(step):
            self.runtime.state = ExplorationState.ABORTED
            status["reason"] = "exploration_action_budget_exhausted"
            status["elapsed_episode_steps"] = int(step) - int(
                self._episode_start_step
            )
            return None, status
        if self._frontier_count >= self.maximum_frontiers:
            self.runtime.state = ExplorationState.ABORTED
            status["reason"] = "exploration_frontier_budget_exhausted"
            status["frontier_count"] = self._frontier_count
            return None, status

        plan = self.runtime.begin(
            failure,
            start_world_xz,
            goals,
            current_heading_rad=current_heading_rad,
        )
        if plan is None:
            status["reason"] = "exploration_failure_not_admissible"
            return None, status
        if plan.status != "planned" or plan.candidate is None:
            status.update(
                {
                    "reason": plan.status,
                    "selection": plan.selection.to_dict(),
                }
            )
            return None, status

        self._frontier_count += 1
        self._last_transition_step = int(step)
        context = self._execution_context(metric_map, start_world_xz, plan)
        status.update(context.plan_summary)
        status["reason"] = "planned"
        status["frontier_count"] = self._frontier_count
        return context, status

    def _execution_context(
        self, metric_map: Any, start_world_xz: Sequence[float], plan: Any
    ) -> FrontierExecutionContext:
        candidate = plan.candidate
        start_cell = metric_map.spec.world_to_cell(*start_world_xz)
        target_cell = (candidate.world_grid_x, candidate.world_grid_z)
        config = self.frontier_config
        window = metric_map.planning_window(
            start_cell,
            target_cell,
            floor_id=0,
            padding_cells=config.window_padding_cells,
            max_window_cells=config.maximum_window_cells,
        )
        safe = window.safe_mask(config.clearance_cells)
        for connector_cell in plan.selection.start_connector_world_grids:
            x, z = window.global_to_local(*connector_cell)
            if 0 <= z < safe.shape[0] and 0 <= x < safe.shape[1]:
                safe[z, x] = True
        summary = {
            "frontier": candidate.to_dict(),
            "selection": plan.selection.to_dict(),
            "path_waypoints": len(plan.path_world_grids),
            "expanded_states": int(plan.expanded_states),
        }
        return FrontierExecutionContext(
            path_world_grids=tuple(plan.path_world_grids),
            target_world_grid=target_cell,
            target_world_xz=(candidate.world_x, candidate.world_z),
            scan_heading_rad=float(candidate.scan_heading_rad),
            window=window,
            safe_mask=safe,
            plan_summary=summary,
        )

    def frontier_reached(
        self,
        *,
        measured_world_xz: Sequence[float],
        metric_map: Any,
        online_builder: Any,
        step: int,
    ) -> dict[str, Any]:
        if self.runtime is None:
            raise RuntimeError("exploration frontier reached without runtime")
        self.runtime.record_action(measured_world_xz)
        if self.runtime.state == ExplorationState.NAVIGATING:
            # The evaluator's physical tolerance can be a few centimetres
            # wider than the neutral runtime's tolerance.  Reaching the
            # controller terminal set is authoritative here.
            self.runtime.state = ExplorationState.SCANNING
            self.runtime.scan_turns = 0
        self._map_stats_before_scan = dict(metric_map.stats())
        self._builder_stats_before_scan = self._builder_stats(online_builder)
        for _ in range(self.runtime_config.maximum_scan_turns):
            self.runtime.record_scan_turn(new_map_evidence=False)
        self._last_transition_step = int(step)
        return {
            "state": self.runtime.state.value,
            "scan_turns": int(self.runtime_config.maximum_scan_turns),
            "map_stats_before_scan": dict(self._map_stats_before_scan),
            "builder_stats_before_scan": dict(
                self._builder_stats_before_scan
            ),
        }

    def frontier_navigation_aborted(self, *, reason: str, step: int) -> dict[str, Any]:
        if self.runtime is None:
            return {"state": "idle", "reason": reason}
        if self.runtime.plan is not None and self.runtime.plan.candidate is not None:
            candidate = self.runtime.plan.candidate
            self.runtime.blacklist.append((candidate.world_x, candidate.world_z))
        self.runtime.state = ExplorationState.RETRY_TARGET
        self._last_transition_step = int(step)
        return {
            "state": self.runtime.state.value,
            "reason": str(reason),
            "frontier_blacklist_size": len(self.runtime.blacklist),
        }

    def scan_memory_update(self, metric_map: Any, online_builder: Any) -> dict[str, Any]:
        map_after = dict(metric_map.stats())
        builder_after = self._builder_stats(online_builder)
        return {
            "map_stats_after_scan": map_after,
            "map_delta": self._delta(map_after, self._map_stats_before_scan),
            "builder_stats_after_scan": builder_after,
            "builder_delta": self._delta(
                builder_after, self._builder_stats_before_scan
            ),
            "semantic_finalize_contract": (
                "async_queue_drained_and_sg_persisted_at_episode_finish"
            ),
        }

    def complete_target_retry(self) -> dict[str, Any]:
        if self.runtime is None:
            return {"state": "idle"}
        self.runtime.target_retry_succeeded(True)
        result = {
            "state": self.runtime.state.value,
            "frontiers_visited": self._frontier_count,
            "blacklisted_frontiers": len(self.runtime.blacklist),
        }
        self.reset()
        return result
