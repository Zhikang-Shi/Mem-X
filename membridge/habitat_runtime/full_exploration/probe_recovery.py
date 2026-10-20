"""Observation-backed fallback after the configured probe has no action.

No SG/VLM/model calls and no unseen navmesh route search. The live single-step
navmesh check is the same one used by the ordinary probe. Dense depth evidence
may fill raster holes, but never erases a known obstacle outside the actually
occupied agent footprint. All turns/forwards share the original probe budget.
Enabled by default within directional exploration; explicitly set the switch
to 0 to retain the non-recovery policy for controlled comparisons.
"""
from __future__ import annotations

import math
import os
from collections import deque

import numpy as np

from membridge.occupancy_map import SparseWorldMetricMap
from membridge.trajectory_graph import habitat_camera_to_world

ENABLE_ENV = "LIFELONG_SG_EXPLORE_NO_ACTION_RECOVERY"
VERSION = "depth_footprint_detour_v2"
STATE_KEY = "_sg_no_action_recovery"


def enabled(evaluator=None):
    override = getattr(evaluator, "_sg_no_action_recovery_enabled", None)
    if override is not None:
        return bool(override)
    config = getattr(evaluator, "_sg_exploration_config", None)
    configured = getattr(config, "no_action_recovery", None)
    if configured is not None:
        return bool(configured)
    return os.getenv(ENABLE_ENV, "1").strip().lower() in {"1", "true", "yes", "on"}


def observation_cache_enabled():
    """Do not allocate depth history when exploration/probing is disabled."""
    from .exploration_patch import _env_bool
    return (enabled()
            and _env_bool("LIFELONG_SG_EXPLORATION_ENABLE", False)
            and _env_bool("LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE", False))


def observe(recorder, *, depth_m, position, heading, intrinsics, sensor_offset,
            min_depth_m, max_depth_m, frame_idx, episode_id):
    """Cache only recent real observations; does not change the navigation map."""
    frames = getattr(recorder, "_probe_recovery_frames", None)
    identity = str(episode_id)
    if frames is None or getattr(recorder, "_probe_recovery_episode", None) != identity:
        frames = deque(maxlen=12)
        recorder._probe_recovery_frames = frames
        recorder._probe_recovery_episode = identity
    if frames and frames[-1]["frame"] == int(frame_idx):
        return
    depth = np.asarray(depth_m, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError("recovery depth must be HxW")
    frames.append(dict(depth=depth[::2, ::2].copy(), stride=2,
                       position=np.asarray(position, dtype=float).reshape(3).copy(),
                       heading=float(heading), intrinsics=tuple(intrinsics),
                       offset=tuple(sensor_offset), minimum=float(min_depth_m),
                       maximum=float(max_depth_m), frame=int(frame_idx)))


def _dense_frame_cells(frame, spec):
    if "cells" in frame:
        return frame["cells"]
    depth = frame["depth"]
    rows, cols = np.indices(depth.shape)
    valid = (np.isfinite(depth) & (depth > frame["minimum"] + 1e-4)
             & (depth < frame["maximum"] - 1e-4))
    distances = depth[valid].astype(float)
    fx, fy, cx, cy = frame["intrinsics"]
    points = np.column_stack(((cols[valid] * frame["stride"] - cx) * distances / fx,
                              -(rows[valid] * frame["stride"] - cy) * distances / fy,
                              -distances, np.ones_like(distances)))
    pose = habitat_camera_to_world(frame["position"], frame["heading"],
                                   sensor_offset_xyz=frame["offset"])
    world = (pose @ points.T).T[:, :3]
    origin = pose[[0, 2], 3]
    delta = world[:, [0, 2]] - origin
    lengths = np.linalg.norm(delta, axis=1)
    # Clip ray traversal, NOT the sensor measurement: a truncated ray is not
    # an obstacle. Invalid/no-return depth supplies no free-space evidence.
    reach = 0.80
    scale = np.minimum(1., reach / np.maximum(lengths, 1e-9))
    endpoints = origin + delta * scale[:, None]
    heights = world[:, 1] - frame["position"][1]
    occupied = (lengths <= reach) & (heights >= .05) & (heights <= 1.35)
    scratch = SparseWorldMetricMap(spec, log_odds_free=-1.)
    scratch.integrate_rays(origin, endpoints, endpoint_occupied=occupied)
    center = spec.world_to_cell(*origin)
    window = scratch.planning_window(center, center, padding_cells=18)
    def cells(mask):
        zz, xx = np.nonzero(mask)
        return set(zip((xx + window.origin_grid_x).tolist(),
                       (zz + window.origin_grid_z).tolist()))
    frame["cells"] = cells(window.free), cells(window.occupied)
    return frame["cells"]


def _occupied_footprint(spec, current, previous=None, radius=.10):
    """Only fully contained cells, never a dilated disc into an adjacent wall."""
    samples = [np.asarray(current, dtype=float)]
    if previous is not None:
        previous = np.asarray(previous, dtype=float)
        distance = np.linalg.norm(samples[0] - previous)
        if distance <= .40:
            samples = list(np.linspace(previous, samples[0],
                                       max(2, int(math.ceil(distance / .025)) + 1)))
    cells = set()
    half = spec.resolution_m / 2
    extent = int(math.ceil(radius / spec.resolution_m)) + 1
    for sample in samples:
        cx, cz = spec.world_to_cell(*sample)
        for z in range(cz - extent, cz + extent + 1):
            for x in range(cx - extent, cx + extent + 1):
                center = np.asarray(spec.cell_center_world(x, z))
                far_corner = np.abs(center - sample) + half
                if np.linalg.norm(far_corner) <= radius + 1e-9:
                    cells.add((x, z))
    return cells


def refresh_evidence(evaluator, target, current):
    metric = evaluator._sg_world_metric_map()
    recorder = getattr(getattr(evaluator, "inherit_sg", None), "world_memory", None)
    frames = getattr(recorder, "_probe_recovery_frames", ())
    state = target.setdefault(STATE_KEY, {})
    free_counts, occupied = {}, set()
    used = []
    height = float(frames[-1]["position"][1]) if frames else None
    for frame in frames:
        if (np.linalg.norm(frame["position"][[0, 2]] - current) > .75
                or abs(float(frame["position"][1]) - height) > .10):
            continue
        free, obstacles = _dense_frame_cells(frame, metric.spec)
        occupied.update(obstacles)
        for cell in free:
            free_counts[cell] = free_counts.get(cell, 0) + 1
        used.append(frame["frame"])
    previous = state.get("last_position")
    if state.get("last_height") != height:
        previous = None
    footprint = _occupied_footprint(metric.spec, current, previous)
    # Known obstacles outside the physical footprint always win. Two distinct
    # acquired frames are required to promote an unknown/uncertain raster cell.
    new_free = {cell for cell, count in free_counts.items()
                if count >= 2 and cell not in occupied
                and np.linalg.norm(np.asarray(metric.spec.cell_center_world(*cell)) - current) <= .65
                and metric.cell_state(*cell) != "occupied"}
    new_free |= footprint
    occupied = {cell for cell in occupied if cell not in footprint
                and np.linalg.norm(np.asarray(metric.spec.cell_center_world(*cell)) - current) <= .65}
    changed = sum(metric.cell_state(*c) != "free" for c in new_free)
    metric.set_classified_cells(free_cells=new_free, occupied_cells=occupied)
    state["last_position"], state["last_height"] = list(map(float, current)), height
    return dict(frames=used, footprint_cells=len(footprint),
                newly_free_cells=changed, observed_obstacle_cells=len(occupied))


def recover(evaluator, target, current, heading, actions, *, step, original_failure):
    from . import exploration_patch as core
    from .old_map_fallback import config_for_target
    config = config_for_target(evaluator, target)
    used = int(target.get("approach_exploration_directional_probe_actions", 0))
    forwards = int(target.get("approach_exploration_directional_probe_forward_actions", 0))
    if used >= config.directional_probe_max_actions:
        return dict(status="action_budget_exhausted", action=None)
    if forwards >= config.directional_probe_max_forward_actions:
        return dict(status="forward_budget_exhausted", action=None)
    if heading is None or evaluator._sg_world_metric_map() is None:
        return original_failure
    current = np.asarray(current, dtype=float).reshape(2)
    heading = float(np.asarray(heading).reshape(-1)[0])
    state = target.setdefault(STATE_KEY, {})
    state.setdefault("trigger", original_failure)
    state["active"] = True
    evidence = refresh_evidence(evaluator, target, current)
    metric = evaluator._sg_world_metric_map()
    resolution = metric.spec.resolution_m
    step_m = float(evaluator.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE)
    turn_rad = math.radians(float(evaluator.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE))
    if step_m <= 0 or turn_rad <= 0:
        return dict(status="invalid_discrete_action_geometry", action=None)
    visits = state.setdefault("visits", [])
    blocked = state.setdefault("blocked", [])
    pending = state.pop("pending", None)
    if pending:
        moved = float(np.linalg.norm(current - np.asarray(pending["origin"])))
        if moved < .5 * step_m:
            blocked.append(pending)
            state.pop("locked_heading", None)
        else:
            visits.append(list(map(float, current)))
    context = evaluator._sg_navmesh_context()
    if config.directional_probe_require_navmesh and context is None:
        return dict(status="missing_live_navmesh", action=None)
    grid = current / resolution - .5
    cell = metric.spec.world_to_cell(*current)
    padding = int(math.ceil(step_m / resolution + config.directional_probe_clearance_cells)) + 3
    window = metric.planning_window(cell, cell, padding_cells=padding)
    start = evaluator._sg_window_point(window, grid)
    mask = window.free & core._known_obstacle_clear_mask(window.occupied, config.directional_probe_clearance_cells)
    cues = core._directional_probe_cues(evaluator, target, current)
    if not cues:
        return dict(status="missing_directional_cue", action=None)
    counts = dict(navmesh=0, displacement=0, unknown_centerline=0,
                  known_obstacle_clearance=0, repeated_block=0)
    choices = []
    for k in range(max(1, round(2 * math.pi / turn_rad))):
        angle = heading + k * turn_rad
        endpoint, predicted = evaluator._sg_predict_forward_endpoint(grid, angle, step_m / resolution, resolution, context)
        if config.directional_probe_require_navmesh and not predicted:
            counts["navmesh"] += 1
            continue
        end_world = (np.asarray(endpoint) + .5) * resolution
        distance = float(np.linalg.norm(end_world - current))
        if distance < .5 * step_m:
            counts["displacement"] += 1
            continue
        end = evaluator._sg_window_point(window, endpoint)
        if not evaluator._sg_float_segment_is_safe(window.free, start, end):
            counts["unknown_centerline"] += 1
            continue
        if not evaluator._sg_float_segment_is_safe(mask, start, end):
            counts["known_obstacle_clearance"] += 1
            continue
        if any(np.linalg.norm(current - np.asarray(b["origin"])) < .20
               and abs(core._heading_error(angle, b["heading"])) < turn_rad / 2
               for b in blocked):
            counts["repeated_block"] += 1
            continue
        cue = cues[0]
        goal = np.asarray(cue["world_xz"])
        progress = float(np.linalg.norm(current - goal) - np.linalg.norm(end_world - goal))
        error = core._heading_error(angle, heading)
        repeat = sum(np.linalg.norm(end_world - np.asarray(p)) < .20 for p in visits)
        locked = ("locked_heading" in state and state.get("segment_forwards", 0) < 4
                  and abs(core._heading_error(angle, state["locked_heading"])) < turn_rad / 2)
        score = progress - .025 * abs(error / turn_rad) - .20 * repeat + (.40 if locked else 0)
        choices.append(dict(score=score, heading_rad=float(angle), progress_m=progress,
                            turn_offset=int(round(error / turn_rad)),
                            endpoint_world_grid=list(map(float, endpoint)),
                            endpoint_world_xz=end_world.tolist(), expected_displacement_m=distance,
                            cue_rank=0, cue_source=cue["source"], cue_world_xz=goal.tolist(),
                            navmesh_predicted=bool(predicted)))
    if not choices:
        state.pop("locked_heading", None)
        scans = int(state.get("scan_turns", 0))
        if scans >= 12:
            return dict(status="recovery_no_observed_safe_step", action=None,
                        recovery_version=VERSION, evidence=evidence, rejection_counts=counts)
        state["scan_turns"] = scans + 1
        selected = dict(action=int(actions.TURN_LEFT), action_kind="turn",
                        recovery_phase="dense_scan", rejection_counts=counts)
    else:
        candidate = max(choices, key=lambda c: c["score"])
        if ("locked_heading" not in state or state.get("segment_forwards", 0) >= 4
                or abs(core._heading_error(candidate["heading_rad"], state["locked_heading"])) >= turn_rad / 2):
            state["locked_heading"] = candidate["heading_rad"]
            state["segment_forwards"] = 0
        error = core._heading_error(candidate["heading_rad"], heading)
        if abs(error) > turn_rad / 2 + 1e-6:
            action = int(actions.TURN_LEFT if error > 0 else actions.TURN_RIGHT)
            kind = "turn"
        else:
            action, kind = int(actions.MOVE_FORWARD), "forward"
            state["segment_forwards"] += 1
            state["pending"] = dict(origin=current.tolist(), heading=candidate["heading_rad"])
            target["_sg_exploration_directional_probe_pending_forward"] = dict(
                origin_world_xz=current.tolist(), **{k: candidate[k] for k in (
                    "endpoint_world_grid", "expected_displacement_m", "cue_source", "cue_world_xz")})
        selected = dict(action=action, action_kind=kind, recovery_phase="safe_detour",
                        rejection_counts=counts, **candidate)
    target["approach_exploration_directional_probe_actions"] = used + 1
    target["approach_exploration_directional_probe_forward_actions"] = forwards + int(selected["action_kind"] == "forward")
    target["approach_exploration_directional_probe_active"] = True
    target["approach_exploration_deferred_map_frame"] = core._query_map_frame(evaluator)
    target.pop("approach_exploration_directional_probe_requested", None)
    return dict(status="action_selected", recovery_version=VERSION, evidence=evidence,
                probe_action_index=used + 1,
                probe_forward_index=target["approach_exploration_directional_probe_forward_actions"],
                **selected)
