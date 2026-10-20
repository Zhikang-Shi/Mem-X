"""Bounded A* in Habitat's turn/turn/forward action lattice."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Sequence

import numpy as np


class Action(IntEnum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


@dataclass(frozen=True)
class ActionLatticeConfig:
    forward_step_cells: float = 5.0
    turn_angle_deg: float = 30.0
    goal_tolerance_cells: float = 3.0
    radius_cells: float = 50.0
    max_states: int = 12_000
    max_actions: int = 96
    turn_cost: float = 0.20


@dataclass(frozen=True)
class ActionPlan:
    actions: tuple[Action, ...]
    expanded_states: int
    goal_distance_cells: float
    status: str


def _headings(current: float, turn: float) -> list[float]:
    count = int(round(2 * math.pi / turn))
    return [math.atan2(math.sin(current + index * turn), math.cos(current + index * turn)) for index in range(count)]


def _key(position: np.ndarray, heading: int) -> tuple[int, int, int]:
    return int(round(position[0] * 1000)), int(round(position[1] * 1000)), int(heading)


def plan_action_lattice(
    start_position: Sequence[float], current_heading_rad: float, target_position: Sequence[float], *,
    segment_is_safe: Callable[[tuple[float, float], tuple[float, float]], bool],
    config: ActionLatticeConfig = ActionLatticeConfig(),
) -> ActionPlan:
    start, target = np.asarray(start_position, dtype=float).reshape(2), np.asarray(target_position, dtype=float).reshape(2)
    turn = math.radians(config.turn_angle_deg)
    headings = _headings(current_heading_rad, turn)
    if not headings or config.forward_step_cells <= 0:
        return ActionPlan((), 0, float(np.linalg.norm(start - target)), "invalid_config")
    if np.linalg.norm(start - target) > config.radius_cells + config.forward_step_cells:
        return ActionPlan((), 0, float(np.linalg.norm(start - target)), "target_outside_local_radius")
    start_key = _key(start, 0)
    positions, parents, parent_actions = {start_key: start}, {start_key: None}, {}
    depths, costs, queue, tie = {start_key: 0}, {start_key: 0.0}, [], 0

    def heuristic(position: np.ndarray) -> float:
        return max(0.0, np.linalg.norm(position - target) - config.goal_tolerance_cells) / config.forward_step_cells

    heapq.heappush(queue, (heuristic(start), 0.0, tie, start_key))
    expanded, goal_key = 0, None
    while queue and expanded < config.max_states:
        _, queued_cost, _, state = heapq.heappop(queue)
        if queued_cost != costs.get(state):
            continue
        position, heading_index, depth = positions[state], state[2], depths[state]
        expanded += 1
        if np.linalg.norm(position - target) <= config.goal_tolerance_cells:
            goal_key = state
            break
        if depth >= config.max_actions:
            continue
        for action, next_heading in ((Action.TURN_LEFT, (heading_index + 1) % len(headings)), (Action.TURN_RIGHT, (heading_index - 1) % len(headings))):
            next_state, next_cost = _key(position, next_heading), queued_cost + config.turn_cost
            if next_cost + 1e-9 < costs.get(next_state, float("inf")):
                costs[next_state], positions[next_state] = next_cost, position
                parents[next_state], parent_actions[next_state], depths[next_state] = state, action, depth + 1
                tie += 1
                heapq.heappush(queue, (next_cost + heuristic(position), next_cost, tie, next_state))
        heading = headings[heading_index]
        # ``heading`` is Habitat's yaw extracted from the agent quaternion.
        # Habitat faces -Z at yaw zero and a positive yaw rotates the forward
        # vector toward -X.  Keeping that convention here is important: using
        # +sin mirrors every planned forward transition across the Z axis.
        endpoint = position + config.forward_step_cells * np.asarray([-math.sin(heading), -math.cos(heading)])
        if np.linalg.norm(endpoint - start) <= config.radius_cells and segment_is_safe(tuple(position), tuple(endpoint)):
            next_state, next_cost = _key(endpoint, heading_index), queued_cost + 1.0
            if next_cost + 1e-9 < costs.get(next_state, float("inf")):
                costs[next_state], positions[next_state] = next_cost, endpoint
                parents[next_state], parent_actions[next_state], depths[next_state] = state, Action.MOVE_FORWARD, depth + 1
                tie += 1
                heapq.heappush(queue, (next_cost + heuristic(endpoint), next_cost, tie, next_state))
    if goal_key is None:
        return ActionPlan((), expanded, float(np.linalg.norm(start - target)), "unreachable_within_bounded_action_lattice")
    actions, cursor = [], goal_key
    while parents[cursor] is not None:
        actions.append(parent_actions[cursor])
        cursor = parents[cursor]
    actions.reverse()
    # Re-simulate exact actions: a returned plan is safe by construction.
    position, heading_index = start.copy(), 0
    for action in actions:
        if action == Action.TURN_LEFT:
            heading_index = (heading_index + 1) % len(headings)
        elif action == Action.TURN_RIGHT:
            heading_index = (heading_index - 1) % len(headings)
        else:
            heading = headings[heading_index]
            endpoint = position + config.forward_step_cells * np.asarray([-math.sin(heading), -math.cos(heading)])
            if not segment_is_safe(tuple(position), tuple(endpoint)):
                return ActionPlan((), expanded, float(np.linalg.norm(position - target)), "exact_replay_failed")
            position = endpoint
    distance = float(np.linalg.norm(position - target))
    status = "planned" if distance <= config.goal_tolerance_cells + 1e-6 else "exact_replay_missed_goal"
    return ActionPlan(tuple(actions) if status == "planned" else (), expanded, distance, status)
