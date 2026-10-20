"""Observed-map bounds and immutable episode-history connection certificates.

No semantic inference and no navmesh route generation. The exact world planner
remains the final witness; its ordinary safety/terminal checks are unchanged.
"""
from __future__ import annotations

import numpy as np

HISTORY_ATTR = '_episode_old_trajectory_positions'
GEOMETRIC_FAILURES = {
    'no_world_reachable_target_within_snap_radius', 'world_theta_star_failed',
    'warm_start_not_free_in_world_memory',
    'warm_start_not_clearance_safe_in_world_memory',
}


def freeze_episode_history(adapter):
    """Call after memory preload, before any observation of the new episode."""
    graph = getattr(getattr(adapter, 'world_memory', None), 'trajectory', None)
    positions = []
    for node in getattr(graph, 'nodes', ()):
        point = np.asarray(node.position_xyz, dtype=float).reshape(3)
        if np.isfinite(point).all() and getattr(node, 'floor_id', 0) == 0:
            positions.append(tuple(map(float, point)))
    setattr(adapter, HISTORY_ATTR, tuple(positions))


def trajectory_points(evaluator, *, frozen=False):
    adapter = getattr(evaluator, 'inherit_sg', None)
    if frozen:
        # Fail closed for adapters that did not mark the episode boundary.
        # Never infer "old" from a graph already extended by the current query.
        positions = getattr(adapter, HISTORY_ATTR, ())
    else:
        graph = getattr(getattr(adapter, 'world_memory', None), 'trajectory', None)
        positions = [n.position_xyz for n in getattr(graph, 'nodes', ())
                     if getattr(n, 'floor_id', 0) == 0]
    height = None
    context = getattr(evaluator, '_sg_live_navmesh_context', None)
    if callable(context):
        value = context()
        if value is not None:
            height = float(value[1])
    points = set()
    for raw in positions:
        point = np.asarray(raw, dtype=float).reshape(3)
        if np.isfinite(point).all() and (height is None or abs(point[1]-height) <= .5):
            points.add((float(point[0]), float(point[2])))
    return [list(p) for p in sorted(points)]


def expanded_trajectory_anchors(evaluator, anchors, current, target):
    original = []
    for raw in (() if anchors is None else anchors):
        point = np.asarray(raw, dtype=float).reshape(2)
        if np.isfinite(point).all():
            original.append(list(map(float, point)))
    points = trajectory_points(evaluator)
    if not points:
        return None
    bounds = np.asarray(original + [list(current), list(target)], dtype=float)
    lower, upper = bounds.min(axis=0), bounds.max(axis=0)
    extra = np.asarray(points)
    if np.all(extra >= lower-1e-6) and np.all(extra <= upper+1e-6):
        return None
    return [list(p) for p in sorted({tuple(p) for p in original+points})]


def reachable_component(mask, start):
    # With corner cutting forbidden, 8-neighbour reachability has exactly the
    # same components as 4-neighbour reachability. Avoid a Python BFS per goal.
    import cv2
    filled = np.asarray(mask, dtype=np.uint8).copy()
    if not filled[start[1], start[0]]:
        return np.zeros_like(filled, dtype=bool)
    cv2.floodFill(filled, None, tuple(map(int, start)), 2, flags=4)
    return filled == 2


class _MaskCaptured(Exception):
    pass


def plan_to_any_old_node(evaluator, current):
    """Screen every immutable old node, then return an exact-planner proof.

    Temporary method hooks inspect the production mask and are always restored.
    Memory and map evidence are never modified by this checker.
    """
    current = np.asarray(current, dtype=float).reshape(2)
    goals = sorted(trajectory_points(evaluator, frozen=True),
                   key=lambda p: (np.linalg.norm(np.asarray(p)-current), p[1], p[0]))
    if not goals:
        return None
    captured = {}
    factory = evaluator._sg_world_navmesh_target_validator
    names = ('_sg_world_navmesh_target_validator', '_sg_reachable_target')
    saved = {name:evaluator.__dict__.get(name) for name in names}
    present = {name:name in evaluator.__dict__ for name in names}

    def validator(window, *args, **kwargs):
        captured['window'] = window
        return factory(window, *args, **kwargs)

    def select(mask, start, raw, exact, target_validator=None):
        captured.update(mask=mask, start=start, validator=target_validator)
        raise _MaskCaptured()

    evaluator._sg_world_navmesh_target_validator = validator
    evaluator._sg_reachable_target = select
    try:
        evaluator._sg_plan_world_path(goals[0], current, exact=True,
            planning_anchor_world_xz=goals, project_target_to_navmesh=False)
    except _MaskCaptured:
        pass
    finally:
        for name in names:
            if present[name]:
                setattr(evaluator, name, saved[name])
            else:
                evaluator.__dict__.pop(name, None)
    if 'mask' not in captured:
        return None
    mask = reachable_component(captured['mask'], captured['start'])
    window = captured['window']
    spec = evaluator._sg_world_metric_map().spec
    for rank, point in enumerate(goals, 1):
        x, z = window.global_to_local(*spec.world_to_cell(*point))
        if not (0 <= z < mask.shape[0] and 0 <= x < mask.shape[1] and mask[z,x]):
            continue
        if captured['validator'] and not captured['validator']((x,z)):
            continue
        plan = evaluator._sg_plan_world_path(point, current, exact=True,
            planning_anchor_world_xz=goals, project_target_to_navmesh=False)
        if plan is not None:
            return dict(goal_world_xz=point, goal_rank_by_distance=rank,
                        tested_nodes=rank, total_old_nodes=len(goals), plan=plan)
    return None
