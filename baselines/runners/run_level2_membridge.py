#!/usr/bin/env python3
"""Level-2 EP1 and EP2 +MemBridge for ETPNav, Janus, CMA, and Uni-LaViRA.

Every episode is localized against an EP0 replay, one frame every 3 steps,
with a 12x30 degree start spin. Loc-SR uses all 238 episodes as the
denominator. The SE(2) takes the current episode's local frame into the EP0
local frame.

EP1 reads frozen M0. EP2 reads frozen M1. A retrieval hit plus a useful lock
walks the scene graph with the aligned pose, then the type-2 suffix goes to
ETPNav. ETPNav's graph stays in the current episode's local frame, and the
waypoint is computed from that same local position. No hit or a failed lock
gives ETPNav the full instruction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

from bootstrap import VOCABULARY, install

install()
_RUNNERS = Path(__file__).resolve().parent
HERE = _RUNNERS.parent

from dino import encode_rgbs  # noqa: E402
from frames import EpisodeStart  # noqa: E402
from geom_lock import (  # noqa: E402
    MIN_INLIERS,
    QUERY_WINDOW,
    extract_map_superpoint,
    models,
    try_lock_frame,
)
from level2_membridge_support import (  # noqa: E402
    LOCK_EVERY_M,
    SPIN_STEPS,
    ep0_local_pose,
    estimated_world_pose,
    parse_level2_instruction,
    replay_ep0_map,
    snap_relative,
)
from run_ep1_memory_janus import Actions  # noqa: E402

FORWARD_STEP_M = 0.25
from run_ep1_memory_janus import (  # noqa: E402
    MemoryNavigator,
    install_full_exploration_overlay,
    install_memory_environment,
    make_target_info,
    memory_observation,
    pose_from_env,
    prepare_recall_plan,
    sensor_memory_config,
)
from run_ep2_explore_loc_sg import make_env, make_janus_env  # noqa: E402
from se2 import gt_ep1_to_ep0, se2_error, world_agent_to_local, yaw_from_xyzw  # noqa: E402
from collections import deque

DEFAULT_M0 = Path(
    "/share/shared/shizk/level2_outputs/etpnav_memory_module/"
    "ep0_memory_from_poses_b15_20260910-ep0-sg-b15"
)
DEFAULT_M1 = Path(
    "/share/shared/shizk/level2_outputs/etpnav_memory_module/"
    "ep1_memory_from_poses_b15_20260910-ep1-sg-b15"
)
STAGE_DIR = {"ep1": "M1", "ep2": "M2"}
MEMORY_DIR = {"ep1": "M0", "ep2": "M1"}
DEFAULT_EP0 = Path(
    "/share/shared/shizk/level2_outputs/etpnav_nograph_local/ep0_20260912-l2-nograph"
)


def live_frame(observations, pose, start: EpisodeStart, idx: int):
    from frames import LocalFrame
    from se2 import habitat_camera_to_world

    rgb = np.asarray(observations["rgb"], dtype=np.uint8)[..., :3].copy()
    depth = np.asarray(observations["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    finite = depth[np.isfinite(depth)]
    if finite.size and float(np.nanmax(finite)) <= 1.5:
        depth = depth * 10.0
    world = np.asarray(pose["position"], dtype=np.float64)
    wyaw = float(yaw_from_xyzw(pose["rotation_xyzw"]))
    x, y, z, yaw = world_agent_to_local(world, wyaw, start.xyz, start.yaw)
    return LocalFrame(
        frame_idx=idx,
        rgb=rgb,
        depth_m=depth,
        x=x,
        y=y,
        z=z,
        yaw=yaw,
        camera_to_local=habitat_camera_to_world([x, y, z], yaw),
        world_xyz=world,
        world_yaw=wyaw,
        episode_name=start.episode_id,
    )


class _Repeat:
    def __init__(self, episode) -> None:
        self.episode = episode

    def __iter__(self):
        return self

    def __next__(self):
        return self.episode


def task_id_of(episode) -> str:
    episode_id = str(getattr(episode, "episode_id", "") or "")
    for role in ("ep0", "ep1", "ep2", "ep3", "ep4"):
        suffix = f"_{role}"
        if episode_id.endswith(suffix):
            return episode_id[: -len(suffix)]
    return episode_id


def instruction_of(episode) -> str:
    text = getattr(episode, "instruction", None)
    if text is None:
        return ""
    return str(getattr(text, "instruction_text", text))


SEMANTIC_FILES = (
    "memory_kb.json",
    "builder_state.pkl",
    "spatial_graph.pkl",
    "map_data.pkl",
    "world_metric_map.npz",
    "trajectory_graph.npz",
    "nodes.jsonl",
    "relations.csv",
    "config_snapshot.yaml",
    "episode_manifest.jsonl",
)
SEMANTIC_DIRS = ("frames", "objects")


def memory_is_local(memory: Path) -> bool:
    if (memory / "ep0_local_origin.json").is_file():
        return True
    manifest_path = memory / "stage_manifest.json"
    if not manifest_path.is_file():
        return False
    frame = json.loads(manifest_path.read_text(encoding="utf-8")).get("coordinate_frame")
    return frame in {"ep0_local", "ep0_local_after_lock"}


def enable_live_semantic_update() -> None:
    """SAM3 and DINOv3 stay on, and the navigation map keeps updating."""
    from run_ep2_memory_janus import apply_ep2_update_memory

    apply_ep2_update_memory("full")
    os.environ.update(
        {
            "INHERIT_SG_SAM3_CKPT": os.environ.get("MODELS_DIR", "") + "/facebook/sam3/sam3.pt",
            "INHERIT_SG_SAM3_SOURCE_ROOT": os.environ.get("MODELS_DIR", "") + "/source/sam3",
            "SAM3_SOURCE_ROOT": os.environ.get("MODELS_DIR", "") + "/source/sam3",
            "INHERIT_SG_DINOV3_MODEL": os.environ.get("MODELS_DIR", "") + "/facebook/dinov3",
            "INHERIT_SG_DINOV3_REPO": os.environ.get("MODELS_DIR", "") + "/source/dinov3",
            "INHERIT_SG_TEXT_ENCODER": os.environ.get("MODELS_DIR", "") + "/all-MiniLM-L6-v2",
            "INHERIT_SG_SAM3_CATEGORY_BATCH_SIZE": "15",
            "INHERIT_SG_KEYFRAME_FILTER": "1",
        }
    )


def publish_stage(adapter, memory: Path, stage_dir: Path, record: dict, locked: bool, role: str) -> None:
    """Write M1 or M2. A lock keeps the live semantic graph, navigation map, and trajectory."""
    stage_dir.parent.mkdir(parents=True, exist_ok=True)
    if stage_dir.exists():
        raise FileExistsError(stage_dir)
    read_stage = "M0" if role == "ep1" else "M1"
    write_stage = "M1" if role == "ep1" else "M2"
    if locked:
        adapter.finish_episode()
        if adapter.failed:
            raise RuntimeError(f"memory_module marked {role} increment failed")
        run_dir = Path(adapter.pipeline.run_dir).resolve()
        shutil.copytree(memory, stage_dir)
        for name in SEMANTIC_FILES:
            source = run_dir / name
            if source.is_file():
                shutil.copy2(source, stage_dir / name)
        for name in SEMANTIC_DIRS:
            source = run_dir / name
            if not source.is_dir():
                continue
            destination = stage_dir / name
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        missing = [
            name
            for name in ("memory_kb.json", "world_metric_map.npz", "trajectory_graph.npz")
            if not (stage_dir / name).is_file()
        ]
        if missing:
            raise RuntimeError(f"{role} increment did not publish {missing}")
        manifest_path = stage_dir / "stage_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        manifest.update(
            {
                "schema_version": "level2_memory_module_stage_v1",
                "session_id": record["session_id"],
                "episode_id": record["episode_id"],
                "role": role,
                "memory_read": read_stage,
                "memory_write": write_stage,
                "semantic_sg_policy": "live_sam3_dinov3_continuation",
                "navigation_memory_policy": "live_rgbd_world_map_and_trajectory_update",
                "coordinate_frame": (
                    "ep0_local_after_lock" if record.get("local_memory") else "habitat_world"
                ),
            }
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        try:
            adapter.abort_episode()
        except Exception:
            pass
        shutil.copytree(memory, stage_dir)
    (stage_dir / "rollout.json").write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8"
    )


def step_action(env, adapter, observations, action: int):
    observations = env.step(int(action))
    pose = pose_from_env(env, 0, 0)
    observe = getattr(adapter, "_l2_observe", None)
    if observe is not None:
        observe(observations, pose)
    else:
        adapter.process_observation(memory_observation(observations, pose), step=0)
    return observations, pose


def run_one(env, habitat_config, policy, episode, args) -> None:
    session_id = task_id_of(episode)
    stage = STAGE_DIR[args.role]
    memory_stage = MEMORY_DIR[args.role]
    memory_root = args.m0_root if args.role == "ep1" else args.m1_root
    stage_dir = args.output_root / "sessions" / session_id / "stages" / stage
    dest = stage_dir / "rollout.json"
    if dest.is_file():
        print(f"[l2-{args.role}-mb] skip {session_id}", flush=True)
        return
    memory = memory_root / "sessions" / session_id / "stages" / memory_stage
    env.episode_iterator = _Repeat(episode)
    observations = env.reset()
    pose = pose_from_env(env, 0, 0)
    ep1_start = EpisodeStart(
        xyz=np.asarray(pose["position"], dtype=np.float64),
        yaw=float(yaw_from_xyzw(pose["rotation_xyzw"])),
        episode_id=str(episode.episode_id),
        role=args.role,
    )
    work = args.output_root / ".working" / session_id
    work.mkdir(parents=True, exist_ok=True)
    local_memory = memory_is_local(memory)
    install_memory_environment(args, work, memory)
    enable_live_semantic_update()
    from membridge.habitat_runtime.online_adapter import OnlineINHeritSGBuilder

    adapter = OnlineINHeritSGBuilder(
        sensor_memory_config(habitat_config, args.sim_gpu_id),
        width=int(np.asarray(observations["rgb"]).shape[1]),
        height=int(np.asarray(observations["rgb"]).shape[0]),
        save_dir=str(work),
        device=args.device,
    )
    adapter.start_episode(
        str(episode.episode_id), scene_id=str(episode.scene_id), source_episode_id=str(episode.episode_id)
    )
    if hasattr(policy, "set_local_origin"):
        policy.set_local_origin(ep1_start.xyz, ep1_start.yaw)
    policy.begin_episode([0.0, float(pose["position"][1]), 0.0], 0.0)

    instruction = instruction_of(episode)
    route = parse_level2_instruction(instruction)
    recall = {"status": "fallback", "reason": route.get("reason", "parse_failed")}
    if route.get("status") == "ready":
        try:
            recall = prepare_recall_plan(adapter, route)
        except Exception as exc:
            recall = {"status": "fallback", "reason": f"recall_failed:{type(exc).__name__}", "error": str(exc)}
    sg_hit = recall.get("status") == "ready" and recall.get("selected_candidate")

    poses = [pose]
    actions: list[str] = []
    locked = False
    transform = None
    ok_useful = False
    err_t = err_y = None
    lock_reason = "not_attempted"
    best_inlier = 0
    mode = None
    nav = getattr(policy, "nav_name", "etpnav")
    phase = f"{nav}_native"
    active = instruction
    window: deque = deque(maxlen=QUERY_WINDOW)
    mem_feats = map_sp = None
    # Every episode is localized. Loc-SR uses all 238 as the denominator.
    map_frames, ep0_start = replay_ep0_map(env, episode, session_id, args.ep0_rollout_root)
    env.episode_iterator = _Repeat(episode)
    observations = env.reset()
    pose = pose_from_env(env, 0, 0)
    poses = [pose]
    ep1_start = EpisodeStart(
        xyz=np.asarray(pose["position"], dtype=np.float64),
        yaw=float(yaw_from_xyzw(pose["rotation_xyzw"])),
        episode_id=str(episode.episode_id),
        role=args.role,
    )
    if hasattr(policy, "set_local_origin"):
        policy.set_local_origin(ep1_start.xyz, ep1_start.yaw)
    policy.begin_episode([0.0, float(pose["position"][1]), 0.0], 0.0)
    pending: list = []

    def aligned_observation(observations, pose):
        pos, yaw = ep0_local_pose(
            transform, pose["position"], yaw_from_xyzw(pose["rotation_xyzw"]), ep1_start
        )
        observation = memory_observation(observations, pose)
        observation["position"] = np.asarray(pos, dtype=np.float32)
        observation["heading"] = np.asarray([yaw], dtype=np.float32)
        return observation

    def observe_memory(observations, pose) -> None:
        if local_memory and not (locked and transform is not None):
            rgb = np.asarray(observations["rgb"], dtype=np.uint8)[..., :3].copy()
            depth = np.asarray(observations["depth"], dtype=np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
            pending.append(({"rgb": rgb, "depth": depth.copy()}, pose))
            return
        if local_memory:
            adapter.process_observation(aligned_observation(observations, pose), step=0)
            return
        adapter.process_observation(memory_observation(observations, pose), step=0)

    def flush_pending() -> None:
        frames = list(pending)
        pending.clear()
        for stored_observations, stored_pose in frames:
            adapter.process_observation(
                aligned_observation(stored_observations, stored_pose), step=0
            )

    adapter._l2_observe = observe_memory
    observe_memory(observations, pose)
    models()
    mem_feats = encode_rgbs([frame.rgb for frame in map_frames])
    map_sp = extract_map_superpoint(map_frames)
    last_best = None

    def try_lock(frame) -> bool:
        nonlocal locked, transform, lock_reason, best_inlier, last_best
        ok, reason, best = try_lock_frame(
            frame, map_frames, mem_feats, map_sp, query_window=list(window)
        )
        last_best = best
        best_inlier = max(best_inlier, int(getattr(best, "n_inlier", 0) or 0))
        lock_reason = reason
        if ok and best.transform is not None:
            locked = True
            transform = best.transform
            if local_memory:
                flush_pending()
            return True
        return False

    frame = live_frame(observations, pose, ep1_start, 0)
    window.append(frame)
    mode = "start-spin"
    if try_lock(frame):
        mode = "start-spin+geom"
    else:
        for _ in range(SPIN_STEPS):
            if len(actions) >= args.max_steps:
                break
            observations, pose = step_action(env, adapter, observations, int(Actions.TURN_LEFT))
            actions.append("SPIN:TURN_LEFT")
            poses.append(pose)
            frame = live_frame(observations, pose, ep1_start, len(actions))
            window.append(frame)
            if try_lock(frame):
                mode = "start-spin+geom"
                break
    last_try = (frame.x, frame.z)
    while (not locked) and (not env.episode_over) and len(actions) < args.max_steps:
        frame = live_frame(observations, pose, ep1_start, len(actions))
        window.append(frame)
        moved = math.hypot(frame.x - last_try[0], frame.z - last_try[1]) >= LOCK_EVERY_M
        if moved and try_lock(frame):
            mode = "walk+geom"
            break
        if (
            moved
            and last_best is not None
            and int(getattr(last_best, "n_inlier", 0) or 0) >= MIN_INLIERS
            and "not unique" in (lock_reason or "")
        ):
            for _ in range(SPIN_STEPS):
                if locked or env.episode_over or len(actions) >= args.max_steps:
                    break
                observations, pose = step_action(env, adapter, observations, int(Actions.TURN_LEFT))
                actions.append("SPIN:TURN_LEFT")
                poses.append(pose)
                frame = live_frame(observations, pose, ep1_start, len(actions))
                window.append(frame)
                if try_lock(frame):
                    mode = "walk+spin+geom"
                    break
            if locked:
                break
            continue
        if moved:
            last_try = (frame.x, frame.z)
        observations, pose = step_action(env, adapter, observations, int(Actions.MOVE_FORWARD))
        actions.append("LOC:MOVE_FORWARD")
        poses.append(pose)
    if locked and transform is not None:
        gt = gt_ep1_to_ep0(ep0_start.xyz, ep0_start.yaw, ep1_start.xyz, ep1_start.yaw)
        err_t, err_yaw = se2_error(transform, gt)
        err_y = math.degrees(err_yaw)
        ok_useful = bool(err_t <= 0.25 and abs(err_y) <= 10.0)

    navigator = MemoryNavigator(
        adapter=adapter,
        env=env,
        habitat_config=habitat_config,
        decision_log=work / "sg_navigation_decisions.jsonl",
    )
    navigator.current_episode_id = str(episode.episode_id)
    target_info = None
    if sg_hit and ok_useful and transform is not None:
        est_pos, est_yaw = estimated_world_pose(
            transform, pose["position"], yaw_from_xyzw(pose["rotation_xyzw"]), ep1_start, ep0_start
        )
        navigator.update_pose(est_pos, est_yaw, len(actions))
        target_info = make_target_info(navigator, recall["selected_candidate"], recall["task_text"])
        if target_info is not None:
            phase = "sg_navigation"
            active = route.get("merged_suffix") or instruction
        else:
            phase = f"{nav}_fallback"
            lock_reason = (lock_reason or "") + "|sg_plan_failed"
    elif sg_hit:
        phase = f"{nav}_fallback"
    else:
        phase = f"{nav}_native"

    def planner_pose():
        if transform is None or not ok_useful:
            return list(pose["position"]), float(yaw_from_xyzw(pose["rotation_xyzw"]))
        yaw = yaw_from_xyzw(pose["rotation_xyzw"])
        if local_memory:
            return ep0_local_pose(transform, pose["position"], yaw, ep1_start)
        return estimated_world_pose(transform, pose["position"], yaw, ep1_start, ep0_start)

    anchor_reached = False
    while (not env.episode_over) and len(actions) < args.max_steps:
        plan_pos, plan_yaw = planner_pose()
        navigator.update_pose(plan_pos, plan_yaw, len(actions))
        if phase == "sg_navigation":
            reached = navigator._sg_nav_reached_reason(
                step=len(actions),
                current_world_position=plan_pos,
                current_world_heading=plan_yaw,
                target_info=target_info,
            )
            if reached is not None:
                anchor_reached = True
                phase = f"{nav}_suffix" if route.get("merged_suffix") else "stop"
                active = route.get("merged_suffix") or instruction
                if phase == "stop":
                    observations, pose = step_action(env, adapter, observations, int(Actions.STOP))
                    actions.append("SG:STOP")
                    poses.append(pose)
                    break
                continue
            action = navigator._get_sg_world_path_action(
                step=len(actions),
                current_world_position=plan_pos,
                current_world_heading=plan_yaw,
                target_info=target_info,
            )
            if action is None:
                phase = f"{nav}_fallback"
                active = instruction
                continue
            policy.observe_and_act(env, active, pose["position"], yaw_from_xyzw(pose["rotation_xyzw"]), act=False)
            observations, pose = step_action(env, adapter, observations, int(action))
            actions.append("SG:" + ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"][int(action)])
            poses.append(pose)
            continue
        decision = policy.observe_and_act(
            env, active, pose["position"], yaw_from_xyzw(pose["rotation_xyzw"]), act=True
        )
        action, angle, distance = decision
        if getattr(policy, "discrete_actions", False):
            if action is None:
                continue
            observations, pose = step_action(env, adapter, observations, int(action))
            label = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"][int(action)]
            actions.append(f"{nav.upper()}:{label}")
            poses.append(pose)
            if int(action) == int(Actions.STOP):
                break
            continue
        if action == Actions.STOP or (angle is None and distance is None):
            observations, pose = step_action(env, adapter, observations, int(Actions.STOP))
            actions.append("ETPNAV:STOP")
            poses.append(pose)
            break
        snap_relative(env, float(angle))
        observations = env.sim.get_sensor_observations()
        steps = int(float(distance) // FORWARD_STEP_M) if distance and distance >= FORWARD_STEP_M else 0
        actions.append(f"{phase.upper()}:WAYPOINT")
        for _ in range(steps):
            if env.episode_over or len(actions) >= args.max_steps:
                break
            observations, pose = step_action(env, adapter, observations, int(Actions.MOVE_FORWARD))
            actions.append("ETPNAV:MOVE_FORWARD")
            poses.append(pose)
        else:
            pose = pose_from_env(env, len(poses), len(actions))
            observations = env.sim.get_sensor_observations()

    record = {
        "session_id": session_id,
        "episode_id": ep1_start.episode_id,
        "role": args.role,
        "phase": phase,
        "anchor_reached": anchor_reached,
        "instruction": instruction,
        "route": {k: v for k, v in route.items() if k != "parsed"},
        "recall_plan": {k: v for k, v in recall.items() if k != "parsed"},
        "poses": poses,
        "executed_actions": actions,
        "steps": len(actions),
        "localization": {
            "locked": locked,
            "ok_useful": ok_useful,
            "lock_mode": mode,
            "lock_reason": lock_reason,
            "best_inlier": best_inlier,
            "trans_err_m": err_t,
            "yaw_err_deg": err_y,
            "n_map": len(map_frames),
            "map_source": "ep0_replay_stride3",
            "sg_hit": bool(sg_hit),
        },
        "local_memory": local_memory,
        "memory_update": "full",
    }
    publish_stage(adapter, memory, stage_dir, record, locked, args.role)
    print(
        f"[l2-{args.role}-mb] {session_id} phase={phase} locked={locked} "
        f"ok={ok_useful} sg={bool(sg_hit)} steps={len(actions)}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etpnav-root", type=Path, required=True)
    parser.add_argument("--exp-config", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--waypoint-ckpt", type=Path, required=True)
    parser.add_argument("--role", choices=("ep1", "ep2"), required=True)
    parser.add_argument("--m0-root", type=Path, default=DEFAULT_M0)
    parser.add_argument("--m1-root", type=Path, default=DEFAULT_M1)
    parser.add_argument("--ep0-rollout-root", type=Path, default=DEFAULT_EP0)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes-dir", type=Path, default=Path(os.environ.get("SCENES_DIR", "")))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-gpu-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--session-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bert-path", type=Path, default=Path(os.environ.get("MODELS_DIR", "")) / "bert-base-uncased")
    parser.add_argument(
        "--nav-model", choices=("etpnav", "janus", "cma", "unilavira"), default="etpnav"
    )
    parser.add_argument("--janus-root", type=Path, default=Path("/janus/src/JanusVLN"))
    parser.add_argument("--janus-model", type=Path, default=Path("/janus/models/JanusVLN_Extra"))
    parser.add_argument("--janus-config", type=Path)
    parser.add_argument("--cma-root", type=Path, default=Path("/cma"))
    parser.add_argument(
        "--r2r-train",
        type=Path,
        default=Path("/cma/data/datasets/HM3D_VLNCE_v5/level3/level3_ep2.json.gz"),
    )
    parser.add_argument("--unilavira-root", type=Path, default=Path("/unilavira"))
    args = parser.parse_args()
    args.update_memory = "full"
    args.turn_angle = 30.0
    args.bert_path = args.bert_path.resolve()
    args.m0_root = args.m0_root.resolve()
    args.m1_root = args.m1_root.resolve()
    args.output_root = args.output_root.resolve()
    args.query_vocabulary = VOCABULARY
    install_full_exploration_overlay()
    if args.nav_model == "etpnav":
        from level2_membridge_support import LocalETPNav

        env, habitat_config = make_env(args, split=args.role)
        policy = LocalETPNav(args)
    else:
        if args.nav_model == "janus":
            args.janus_dataset = args.dataset
            args.dataset_split = args.role
            if args.janus_config is None:
                args.janus_config = Path(args.janus_root) / "config/vln_level2_protocol.yaml"
            env, habitat_config = make_janus_env(args)
        else:
            env, habitat_config = make_env(args, split=args.role)
        from janus_policy import build_policy

        policy = build_policy(args, lambda _args: None)
    episodes = [
        episode
        for index, episode in enumerate(env.episodes)
        if index % args.num_shards == args.shard_index
        and (not args.session_id or task_id_of(episode) == args.session_id)
    ]
    if args.dry_run:
        print(json.dumps({"role": args.role, "selected": len(episodes), "turn_angle": 30}), flush=True)
        return 0
    failed = 0
    for episode in episodes:
        try:
            if args.nav_model == "janus":
                import torch

                torch.cuda.empty_cache()
            run_one(env, habitat_config, policy, episode, args)
        except Exception:
            failed += 1
            traceback.print_exc()
    if failed:
        raise SystemExit(f"[l2-{args.role}-mb] {failed} episodes failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
