"""Small helpers for the Level-2 EP1 +MemBridge runner. All imports stay in Mem-X."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from frames import EpisodeStart, LocalFrame
from run_ep1_memory_janus import Actions, pose_from_env
from se2 import (
    SE2,
    habitat_camera_to_world,
    world_agent_to_local,
    wrap_angle,
    yaw_from_xyzw,
)

SPIN_STEPS = 12
LOCK_EVERY_M = 1.0
MAP_STRIDE = 3
MAP_MAX_STEPS = 800


def LocalETPNav(args):
    from etpnav_policy import ETPNavSuffixPolicy

    class _LocalETPNav(ETPNavSuffixPolicy):
        """Same local-frame controller as the Level-2 EP1 MemBridge run.

        The graph is stored in the current episode's local frame. The waypoint
        is computed from that local position, not from the Habitat world position.
        """

        def __init__(self, args) -> None:
            super().__init__(args)
            self.origin_xyz = None
            self.origin_yaw = None

        def set_local_origin(self, xyz, yaw: float) -> None:
            self.origin_xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
            self.origin_yaw = float(yaw)

        def _agent_pose(self, env):
            position, orientation = super()._agent_pose(env)
            if self.origin_xyz is None:
                return position, orientation
            world_yaw = yaw_from_xyzw(orientation)
            x, y, z, yaw = world_agent_to_local(
                position, world_yaw, self.origin_xyz, self.origin_yaw
            )
            half = 0.5 * yaw
            local_orientation = np.array(
                [0.0, math.sin(half), 0.0, math.cos(half)], dtype=np.float32
            )
            return np.array([x, y, z], dtype=np.float32), local_orientation

        def observe_and_act(self, env, instruction, position, heading, *, act):
            """Copied from the previous local-loc policy, including the local waypoint."""
            torch = self.torch
            from etpnav_policy import MAX_HIGH_LEVEL, Actions, capture_twelve_views
            from vlnce_baselines.common.environments import calculate_vp_rel_pos

            views = capture_twelve_views(env)
            batch = self._batch_views(views)
            cur_pos, cur_ori = self._agent_pose(env)
            with torch.no_grad():
                wp_outputs = self.net(
                    mode="waypoint",
                    waypoint_predictor=self.waypoint,
                    observations=batch,
                    in_train=False,
                )
                vp_inputs = self._vp_feature_variable(wp_outputs)
                vp_inputs["mode"] = "panorama"
                pano_embeds, pano_masks = self.net(**vp_inputs)
                avg_pano = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / torch.sum(
                    pano_masks, 1, keepdim=True
                )
                cur_vp, cand_vp, cand_pos = self.gmap.identify_node(
                    cur_pos, cur_ori, wp_outputs["cand_angles"][0], wp_outputs["cand_distances"][0]
                )
                cand_embeds = pano_embeds[0][vp_inputs["nav_types"][0] == 1]
                step_id = self.actor_steps + 1 if act else max(self.actor_steps, 0) + 1
                self.gmap.update_graph(
                    self.prev_vp,
                    step_id,
                    cur_vp,
                    cur_pos,
                    avg_pano[0],
                    cand_vp,
                    cand_pos,
                    cand_embeds,
                    None,
                )
                self.prev_vp = cur_vp
                if not act:
                    return None, None, None
                self.actor_steps += 1
                self._ensure_language(instruction)
                nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
                no_vp_left = nav_inputs.pop("no_vp_left")
                nav_inputs.update(
                    {
                        "mode": "navigation",
                        "txt_embeds": self._txt_embeds,
                        "txt_masks": self._txt_masks,
                    }
                )
                nav_outs = self.net(**nav_inputs)
                logits = nav_outs["global_logits"]
                index = int(logits.argmax(dim=-1).item())
                if index == 0 or no_vp_left or self.actor_steps >= MAX_HIGH_LEVEL:
                    return Actions.STOP, None, None
                ghost_vp = nav_inputs["gmap_vp_ids"][0][index]
                ghost_pos = self.gmap.ghost_aug_pos[ghost_vp]
                ang, dis = calculate_vp_rel_pos(
                    cur_pos, ghost_pos, float(yaw_from_xyzw(cur_ori))
                )
                if self.cfg.MODEL.consume_ghost:
                    self.gmap.delete_ghost(ghost_vp)
                return None, float(ang), float(dis)

    return _LocalETPNav(args)


def parse_level2_instruction(instruction: str) -> dict[str, Any]:
    """One type-1 recall, then the concatenated type-2 suffix. Live VLM only."""
    from membridge.vlm_instruction_parser.runtime import InstructionRuntimeConfig

    try:
        parsed = InstructionRuntimeConfig.from_env().build().parse(instruction)
    except Exception as exc:
        return {"status": "fallback", "reason": f"parse_failed:{type(exc).__name__}", "error": str(exc)}
    types = [segment.type for segment in parsed.segments]
    recalls = [segment for segment in parsed.segments if segment.type == 1]
    suffixes = [segment for segment in parsed.segments if segment.type == 2]
    if len(recalls) != 1:
        return {
            "status": "fallback",
            "reason": "level2_parse_not_one_recall",
            "types": types,
        }
    recall = recalls[0].to_dict()
    suffix = " ".join(segment.text for segment in suffixes if segment.text).strip()
    return {
        "status": "ready",
        "types": types,
        "recall": recall,
        "merged_suffix": suffix,
        "suffix_parts": [segment.text for segment in suffixes],
    }


def ep0_local_pose(transform: SE2, world_xyz, world_yaw: float, ep1_start):
    """Map the live pose into the EP0 local frame. This is the aligned coordinate."""
    x, y, z, yaw = world_agent_to_local(world_xyz, world_yaw, ep1_start.xyz, ep1_start.yaw)
    aligned = transform.apply(np.array([x, z], dtype=np.float64))
    return [float(aligned[0]), float(y), float(aligned[1])], wrap_angle(yaw + transform.yaw)


def estimated_world_pose(transform: SE2, world_xyz, world_yaw: float, ep1_start, ep0_start):
    """Map the live EP1 pose into the world through the EP0 local frame."""
    x, y, z, yaw = world_agent_to_local(world_xyz, world_yaw, ep1_start.xyz, ep1_start.yaw)
    aligned = transform.apply(np.array([x, z], dtype=np.float64))
    yaw0 = wrap_angle(yaw + transform.yaw)
    cosine, sine = math.cos(ep0_start.yaw), math.sin(ep0_start.yaw)
    world_x = float(ep0_start.xyz[0] + cosine * aligned[0] + sine * aligned[1])
    world_z = float(ep0_start.xyz[2] - sine * aligned[0] + cosine * aligned[1])
    return [world_x, float(y), world_z], wrap_angle(yaw0 + ep0_start.yaw)


def _map_frame(observations, pose, start: EpisodeStart, idx: int) -> LocalFrame:
    rgb = np.asarray(observations["rgb"], dtype=np.uint8)[..., :3].copy()
    depth = np.asarray(observations["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    finite = depth[np.isfinite(depth)]
    if finite.size and float(np.nanmax(finite)) <= 1.5:
        depth = depth * 10.0
    world = np.asarray(pose["position"], dtype=np.float64)
    world_yaw = yaw_from_xyzw(pose["rotation_xyzw"])
    x, y, z, yaw = world_agent_to_local(world, world_yaw, start.xyz, start.yaw)
    return LocalFrame(
        frame_idx=idx,
        rgb=rgb,
        depth_m=depth,
        x=float(x),
        y=float(y),
        z=float(z),
        yaw=float(yaw),
        camera_to_local=habitat_camera_to_world([x, y, z], yaw),
        world_xyz=world,
        world_yaw=float(world_yaw),
        episode_name=start.episode_id,
    )


class _Repeat:
    def __init__(self, episode) -> None:
        self.episode = episode

    def __iter__(self):
        return self

    def __next__(self):
        return self.episode


def _walk_action(cur_xyz, cur_yaw: float, target_xyz) -> int:
    dx = float(target_xyz[0] - cur_xyz[0])
    dz = float(target_xyz[2] - cur_xyz[2])
    if math.hypot(dx, dz) < 0.20:
        return int(Actions.MOVE_FORWARD)
    want = math.atan2(-dx, -dz)
    err = wrap_angle(want - cur_yaw)
    if err > math.radians(20.0):
        return int(Actions.TURN_LEFT)
    if err < -math.radians(20.0):
        return int(Actions.TURN_RIGHT)
    return int(Actions.MOVE_FORWARD)


def _follow_poses(env, waypoints, start_obs, start_pose, max_steps: int, on_step=None):
    observations = start_obs
    pose = start_pose
    wp_i = 0
    step = 0
    while step < max_steps and wp_i < len(waypoints):
        if on_step is not None:
            on_step(observations, pose, step)
        cur = np.asarray(pose["position"], dtype=np.float64)
        while wp_i < len(waypoints) and math.hypot(
            waypoints[wp_i][0] - cur[0], waypoints[wp_i][2] - cur[2]
        ) < 0.30:
            wp_i += 1
        if wp_i >= len(waypoints):
            break
        action = _walk_action(cur, yaw_from_xyzw(pose["rotation_xyzw"]), waypoints[wp_i])
        observations = env.step(action)
        step += 1
        pose = pose_from_env(env, step, step)
        if env.episode_over:
            break
    if on_step is not None:
        on_step(observations, pose, step)
    return observations, pose, step


def replay_ep0_map(env, episode, session_id: str, rollout_root: Path):
    """Replay the saved EP0 path and keep one RGB-D frame every 3 steps."""
    rollout = Path(rollout_root) / "sessions" / session_id / "stages" / "M0" / "rollout.json"
    payload = json.loads(rollout.read_text(encoding="utf-8"))
    saved = payload.get("poses") or []
    if not saved:
        raise RuntimeError(f"{session_id}: empty EP0 rollout {rollout}")
    waypoints = np.asarray([item["position"] for item in saved], dtype=np.float64)
    inner = getattr(env, "_env", env)
    current = int(getattr(inner, "_max_episode_steps", MAP_MAX_STEPS) or MAP_MAX_STEPS)
    inner._max_episode_steps = max(current, MAP_MAX_STEPS)
    env.episode_iterator = _Repeat(episode)
    env.reset()
    first = saved[0]
    position = np.asarray(first["position"], dtype=np.float64)
    rotation = first.get("rotation_xyzw")
    if rotation is None:
        quat = env.sim.get_agent_state().rotation
    else:
        x, y, z, w = (float(v) for v in rotation)
        quat = np.quaternion(w, x, y, z)
    env.sim.set_agent_state(position, quat)
    observations = env.sim.get_sensor_observations()
    pose = pose_from_env(env, 0, 0)
    ep0_start = EpisodeStart(
        xyz=np.asarray(pose["position"], dtype=np.float64),
        yaw=float(yaw_from_xyzw(pose["rotation_xyzw"])),
        episode_id=str(episode.episode_id),
        role="ep0",
    )
    frames: list[LocalFrame] = []

    def keep(obs, cur_pose, step: int) -> None:
        if step % MAP_STRIDE == 0 or step == 0:
            frames.append(_map_frame(obs, cur_pose, ep0_start, len(frames)))

    _follow_poses(env, waypoints, observations, pose, MAP_MAX_STEPS, on_step=keep)
    if not frames:
        raise RuntimeError(f"{session_id}: EP0 replay produced no frames")
    return frames, ep0_start


def snap_relative(env, angle_rad: float) -> None:
    """Apply a relative heading the way the waypoint controller does."""
    state = env.sim.get_agent_state()
    rotation = state.rotation
    theta = np.arctan2(rotation.imag[1], rotation.real) + float(angle_rad) / 2.0
    snapped = np.quaternion(np.cos(theta), 0.0, np.sin(theta), 0.0)
    env.sim.set_agent_state(state.position, snapped)
