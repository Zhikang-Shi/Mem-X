"""Bounded second-stage escape toward immutable old-map regions.

Activated only after the ordinary probe and a final original-target recheck
fail. It borrows no SG/VLM calls, never replaces the recalled object, and uses
the existing directional probe / observation recovery for every action.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from membridge.navigation.map_connection import plan_to_any_old_node, trajectory_points

STATE_KEY = '_sg_old_map_fallback'
USED_KEY = 'approach_exploration_old_map_fallback_used'
ELIGIBLE_FAILURES = {
    'action_budget_exhausted', 'forward_budget_exhausted',
    'no_certified_progressing_single_step', 'recovery_no_observed_safe_step',
}


def config_for_target(evaluator, target):
    from .exploration_patch import _runtime_config
    config = _runtime_config(evaluator)
    state = target.get(STATE_KEY, {})
    if state.get('active'):
        return replace(config, directional_probe_max_actions=state['action_limit'],
                       directional_probe_max_forward_actions=state['forward_limit'])
    return config


def select_cue(goals, current, tried):
    current = np.asarray(current, dtype=float)
    def sector(point):
        delta = np.asarray(point)-current
        return int(math.floor((math.atan2(delta[1],delta[0])+math.pi)/(math.pi/4))) % 8
    used_sectors = {sector(p) for p in tried}
    candidates = []
    for point in goals:
        if any(np.linalg.norm(np.asarray(point)-p) < 1.0 for p in tried):
            continue
        candidates.append((sector(point) in used_sectors,
            float(np.linalg.norm(np.asarray(point)-current)), point[1],point[0],point))
    return list(min(candidates)[-1]) if candidates else None


def reset_segment(target):
    from .probe_recovery import STATE_KEY as RECOVERY_KEY
    state = target.get(RECOVERY_KEY, {})
    for key in ('active', 'locked_heading', 'segment_forwards', 'scan_turns'):
        state.pop(key, None)


def _log(evaluator, step, **payload):
    write = getattr(evaluator, '_write_sg_decision', None)
    if callable(write):
        write(step, dict(ordinary_vln_executed=False, **payload))


def start_after_failure(evaluator, target, current, heading, actions, *, step, failure):
    """Caller MUST recheck the original target before invoking this hook."""
    from . import exploration_patch as core
    config = core._runtime_config(evaluator)
    if (not config.enabled or not config.directional_probe
            or not config.old_map_fallback or config.connection_only
            or target.get(USED_KEY) or heading is None
            or failure.get('status') not in ELIGIBLE_FAILURES
            or int(target.get('approach_exploration_action_count',0)) >= config.maximum_actions
            or float(target.get('approach_exploration_distance_m',0)) >= config.maximum_distance_m):
        return None
    goals = trajectory_points(evaluator, frozen=True)
    if not goals:
        return None
    if plan_to_any_old_node(evaluator, current) is not None:
        # Already having an old-map path does not prove target reachability.
        # Leave this to ordinary SG failure handling, not another escape loop.
        return None
    used = int(target.get('approach_exploration_directional_probe_actions',0))
    forwards = int(target.get('approach_exploration_directional_probe_forward_actions',0))
    target[STATE_KEY] = dict(active=True, goals=goals, tried=[], cue=None,
        action_limit=used+config.old_map_fallback_max_actions,
        forward_limit=forwards+config.old_map_fallback_max_forwards,
        started_actions=used, started_forwards=forwards, start_step=int(step))
    target[USED_KEY] = True
    reset_segment(target)
    target['approach_exploration_directional_probe_active'] = True
    _log(evaluator, step, decision='exploration_old_map_fallback_started',
         trigger=failure['status'], frozen_nodes=len(goals),
         extra_actions=config.old_map_fallback_max_actions,
         extra_forwards=config.old_map_fallback_max_forwards)
    return action(evaluator, target, current, heading, actions, step=step)


def action(evaluator, target, current, heading, actions, *, step):
    from . import exploration_patch as core
    config = core._runtime_config(evaluator)
    state = target[STATE_KEY]
    if state.get('connected'):
        return dict(action=None, status='old_map_connected_target_unreachable')
    used = int(target.get('approach_exploration_directional_probe_actions',0))
    forwards = int(target.get('approach_exploration_directional_probe_forward_actions',0))
    if used >= state['action_limit']:
        return dict(action=None,status='action_budget_exhausted',old_map_fallback=True)
    if forwards >= state['forward_limit']:
        return dict(action=None,status='forward_budget_exhausted',old_map_fallback=True)
    current = np.asarray(current,dtype=float)
    cue = state.get('cue')
    if cue is not None and used-state['cue_start_actions'] >= config.old_map_fallback_switch_actions:
        distance = float(np.linalg.norm(current-np.asarray(cue)))
        if state['cue_start_distance']-distance < config.old_map_fallback_min_progress_m:
            state['tried'].append(cue)
            state['cue'] = None
            reset_segment(target)
        else:
            state['cue_start_actions'], state['cue_start_distance'] = used, distance
    # A failed directional cue may be replaced without advancing the simulator;
    # each replacement excludes a region, so this loop is finite.
    for _ in range(len(state['goals'])+1):
        if state['cue'] is None:
            cue = select_cue(state['goals'], current, state['tried'])
            if cue is None:
                return dict(action=None,status='old_map_cues_exhausted',old_map_fallback=True)
            state.update(cue=cue,cue_start_actions=used,
                         cue_start_distance=float(np.linalg.norm(current-np.asarray(cue))))
            _log(evaluator, step, decision='exploration_old_map_fallback_cue',
                 cue_world_xz=cue, distance_m=state['cue_start_distance'])
        result = core._primary_directional_probe_action(
            evaluator,target,current,heading,actions,step=step)
        if result.get('action') is not None or result.get('status') not in {
                'no_certified_progressing_single_step','recovery_no_observed_safe_step'}:
            return dict(result, old_map_fallback=True)
        state['tried'].append(state['cue'])
        state['cue'] = None
        reset_segment(target)
    return dict(action=None,status='old_map_cues_exhausted',old_map_fallback=True)


def check_old_connection(evaluator, target, current, *, step):
    """Do not declare the recalled object reached when only the map connects."""
    state = target.get(STATE_KEY, {})
    if not state.get('active'):
        return None
    connection = plan_to_any_old_node(evaluator, current)
    if connection is not None:
        state.update(active=False, connected=True)
        _log(evaluator, step, decision='exploration_old_map_connected_target_unreachable',
             goal_world_xz=connection['goal_world_xz'], target_completed=False)
    return connection
