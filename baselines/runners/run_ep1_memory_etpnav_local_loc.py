#!/usr/bin/env python3
"""ETPNav L3 EP1: locate while navigating, then increment the local M0.

L3 geom-lock protocol (same as Janus / Uni / CMA L3 loc):

* No opening spin, no god-view 1 m trigger.
* Try lock every ~1 m of EP1-local travel, current heading only.
* Extra 12x30 (24 x 15°) only on visual dual-peak (inlier>=80 and not unique).

Until lock, RGB-D is buffered in EP1-local. After lock, the buffer and all
later frames are warped by T into EP0-local and incrementally ingested into
the preloaded M0 world map / trajectory (no SAM3). ETPNav keeps navigating.
Unlocked episodes copy M0 to M1 and record the loc failure.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gzip
import json
import math
import os
import shutil
import sys
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bootstrap import VOCABULARY, install

install()
_RUNNERS = Path(__file__).resolve().parent
HERE = _RUNNERS.parent
ETP_L2 = HERE
ETP_ROOT = HERE / "models" / "ETPNav"
REPO_ROOT = HERE.parent

sys.path.insert(0, str(ETP_L2))
import etpnav_policy as etp  # noqa: E402

from dino import encode_rgbs  # noqa: E402
from frames import EpisodeStart, LocalFrame  # noqa: E402
from geom_lock import QUERY_WINDOW, extract_map_superpoint, models, try_lock_frame  # noqa: E402
from se2 import (  # noqa: E402
    SE2,
    gt_ep1_to_ep0,
    habitat_camera_to_world,
    se2_error,
    wrap_angle,
    world_agent_to_local,
    yaw_from_xyzw,
)

from run_ep1_memory_janus import (  # noqa: E402
    ACTION_NAMES,
    Actions,
    CORE_M0_FILES,
    NAVIGATION_CONTROLLER_VERSION,
    MemoryNavigator,
    append_jsonl,
    atomic_json,
    install_full_exploration_overlay,
    install_memory_environment,
    pose_from_env,
    read_json,
    read_jsonl,
    utc_now,
)

DEFAULT_DATASET = ETP_ROOT / "data/datasets/HM3D_VLNCE_v5/level3/level3_ep1.json.gz"
DEFAULT_TASK_CONFIG = ETP_ROOT / "run_r2r/hm3d_level3.yaml"
DEFAULT_EXP_CONFIG = ETP_ROOT / "run_r2r/iter_level3.yaml"
DEFAULT_M0 = HERE / "outputs/ep0_memory_from_poses_local_latest"
DEFAULT_OUTPUT = Path(
    "/share/shared/shizk/level3_outputs/etpnav_memory_module/ep1_local_loc"
)
EXPECTED_TASKS = 157
HIGH_INLIER = 80
LOCK_EVERY_M = 1.0
SPIN_STEPS = 12
FORWARD_STEP_M = etp.FORWARD_STEP_M


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etpnav-root", type=Path, default=ETP_ROOT)
    parser.add_argument("--exp-config", type=Path, default=DEFAULT_EXP_CONFIG)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ckpt", type=Path, default=etp.DEFAULT_CKPT)
    parser.add_argument("--waypoint-ckpt", type=Path, default=etp.DEFAULT_WAYPOINT)
    parser.add_argument("--m0-root", type=Path, default=DEFAULT_M0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-gpu-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--session-id", action="append", default=[])
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--worker-id", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expected-count", type=int, default=EXPECTED_TASKS)
    parser.add_argument(
        "--query-vocabulary",
        type=Path,
        default=VOCABULARY,
    )
    parser.add_argument("--bert-path", type=Path, default=Path(os.environ.get("MODELS_DIR", "")) / "bert-base-uncased")
    parser.add_argument("--scenes-dir", type=Path, default=Path(os.environ.get("SCENES_DIR", "")))
    parser.add_argument(
        "--nav-model",
        choices=("etpnav", "janus", "cma", "unilavira"),
        default="etpnav",
    )
    parser.add_argument(
        "--cma-root",
        type=Path,
        default=Path("/home/shizk/LifeLongHomeNav/baseline_experiment/CMA"),
    )
    parser.add_argument(
        "--r2r-train",
        type=Path,
        default=Path(
            "/home/shizk/LifeLongHomeNav/baseline_experiment/CMA/data/datasets/"
            "HM3D_VLNCE_v5/level3/level3_ep2.json.gz"
        ),
    )
    parser.add_argument("--janus-root", type=Path, default=Path("/janus/src/JanusVLN"))
    parser.add_argument("--janus-model", type=Path, default=Path("/janus/models/JanusVLN_Extra"))
    parser.add_argument("--num-history", type=int, default=8)
    args = parser.parse_args()
    args.split = "ep1"
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def read_any_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def instruction_text(episode: dict[str, Any]) -> str:
    raw = episode.get("instruction") or {}
    if isinstance(raw, dict):
        return str(raw.get("instruction_text") or "")
    return str(raw)


def stub_route(instruction: str) -> dict[str, Any]:
    return {
        "instruction": instruction,
        "recall": {},
        "merged_suffix": "",
        "suffix_parts": [],
    }


def _first_keyframe(m0: Path) -> dict[str, Any]:
    path = m0 / "frames" / "confirmed_keyframes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing M0 keyframes: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                return payload
    raise RuntimeError(f"M0 keyframes are empty: {path}")


def load_ep0_start(m0: Path, session_id: str) -> EpisodeStart:
    origin_path = m0 / "ep0_local_origin.json"
    if origin_path.is_file():
        payload = json.loads(origin_path.read_text(encoding="utf-8"))
        return EpisodeStart(
            xyz=np.asarray(payload["ep0_origin_world_xyz"], dtype=np.float64),
            yaw=float(payload["ep0_origin_world_yaw"]),
            episode_id=str(payload.get("episode_id") or session_id),
            role="ep0",
        )
    item = _first_keyframe(m0)
    return EpisodeStart(
        xyz=np.asarray(item["agent_position"], dtype=np.float64),
        yaw=float(item["agent_heading_rad"]),
        episode_id=str(item.get("episode_name") or session_id),
        role="ep0",
    )


def load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    payload = read_any_json(args.dataset)
    rows: list[dict[str, Any]] = []
    for episode in payload.get("episodes") or []:
        info = episode.get("info") or {}
        if str(info.get("lifelong_role") or info.get("level3_role") or "") != "ep1":
            continue
        episode_id = str(getattr(episode, "episode_id", "") or "")
        session_id = episode_id.rsplit("_", 1)[0] if episode_id.endswith(("_ep0", "_ep1", "_ep2", "_ep3", "_ep4")) else str(info.get("session_id") or "")
        if not session_id:
            raise RuntimeError(f"EP1 episode {episode.get('episode_id')} has no level3 task id")
        instruction = instruction_text(episode)
        m0 = args.m0_root / "sessions" / session_id / "stages/M0"
        missing = [name for name in CORE_M0_FILES if not (m0 / name).is_file()]
        if not (m0 / "frames" / "confirmed_keyframes.jsonl").is_file():
            missing.append("frames/confirmed_keyframes.jsonl")
        if not (m0 / "ep0_local_origin.json").is_file():
            frame = _first_keyframe(m0) if (m0 / "frames" / "confirmed_keyframes.jsonl").is_file() else {}
            if frame.get("coordinate_frame") not in (None, "habitat_world"):
                missing.append("ep0_local_origin.json")
        if missing:
            raise RuntimeError(f"M0 {session_id} lacks {missing}: {m0}")
        rows.append(
            {
                "session_id": session_id,
                "episode_id": str(episode["episode_id"]),
                "instruction": instruction,
                "route": stub_route(instruction),
                "m0": m0,
            }
        )
    rows.sort(key=lambda item: item["session_id"])
    if args.expected_count and not args.session_id and not args.episode_id:
        if len(rows) != args.expected_count:
            raise RuntimeError(f"Expected {args.expected_count} EP1 tasks, found {len(rows)}")
    requested_sessions = set(map(str, args.session_id))
    requested_episodes = set(map(str, args.episode_id))
    selected = [
        row
        for row in rows
        if (not requested_sessions or row["session_id"] in requested_sessions)
        and (not requested_episodes or row["episode_id"] in requested_episodes)
    ]
    selected = [
        row
        for index, row in enumerate(selected)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise RuntimeError("EP1 selection is empty")
    return selected


def _depth_m(path: Path) -> np.ndarray:
    raw = np.asarray(Image.open(path))
    depth = raw.astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if raw.dtype == np.uint16:
        return depth / 1000.0
    finite = depth[np.isfinite(depth)]
    if finite.size and float(np.nanmax(finite)) <= 1.5:
        return depth * 10.0
    return depth


def load_m0_local_frames(m0: Path, session_id: str) -> list[LocalFrame]:
    frames_dir = m0 / "frames"
    out: list[LocalFrame] = []
    for line in (frames_dir / "confirmed_keyframes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        stored = np.asarray(item["agent_position"], dtype=np.float64)
        stored_yaw = float(item["agent_heading_rad"])
        if item.get("coordinate_frame") == "ep0_local":
            x, y, z, yaw = float(stored[0]), float(stored[1]), float(stored[2]), stored_yaw
            world = np.asarray(item.get("ep0_origin_world_xyz") or stored, dtype=np.float64)
            wyaw = float(item.get("ep0_origin_world_yaw") or stored_yaw)
        elif item.get("coordinate_frame") in (None, "habitat_world"):
            origin = load_ep0_start(m0, session_id)
            x, y, z, yaw = world_agent_to_local(stored, stored_yaw, origin.xyz, origin.yaw)
            world = stored
            wyaw = stored_yaw
        else:
            raise RuntimeError(f"{session_id} M0 keyframe is not ep0_local")
        out.append(
            LocalFrame(
                frame_idx=int(item["frame_idx"]),
                rgb=np.asarray(Image.open(frames_dir / item["rgb"]).convert("RGB")),
                depth_m=_depth_m(frames_dir / item["depth"]),
                x=x,
                y=y,
                z=z,
                yaw=yaw,
                camera_to_local=habitat_camera_to_world([x, y, z], yaw),
                world_xyz=world,
                world_yaw=wyaw,
                episode_name=session_id,
                rgb_path=str(frames_dir / item["rgb"]),
            )
        )
    out.sort(key=lambda frame: frame.frame_idx)
    if not out:
        raise RuntimeError(f"{session_id}: empty M0 keyframes")
    return out


def copy_rgbd(observations) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(observations["rgb"], dtype=np.uint8)[..., :3].copy()
    depth = np.asarray(observations["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return rgb, depth.copy()


def live_frame(observations, pose, start: EpisodeStart, idx: int) -> LocalFrame:
    rgb, depth = copy_rgbd(observations)
    finite = depth[np.isfinite(depth)]
    if finite.size and float(np.nanmax(finite)) <= 1.5:
        depth = depth * 10.0
    wxyz = np.asarray(pose["position"], dtype=np.float64)
    wyaw = yaw_from_xyzw(pose["rotation_xyzw"])
    x, y, z, yaw = world_agent_to_local(wxyz, wyaw, start.xyz, start.yaw)
    return LocalFrame(
        frame_idx=idx,
        rgb=rgb,
        depth_m=depth,
        x=x,
        y=y,
        z=z,
        yaw=yaw,
        camera_to_local=habitat_camera_to_world([x, y, z], yaw),
        world_xyz=wxyz,
        world_yaw=wyaw,
        episode_name=start.episode_id,
    )


def ep1_to_ep0_local(transform: SE2, world_xyz, world_yaw: float, ep1_start: EpisodeStart):
    x, y, z, yaw = world_agent_to_local(world_xyz, world_yaw, ep1_start.xyz, ep1_start.yaw)
    xz0 = transform.apply(np.array([x, z], dtype=np.float64))
    return [float(xz0[0]), float(y), float(xz0[1])], wrap_angle(yaw + transform.yaw)


def aligned_memory_observation(rgb, depth, pose, transform: SE2, ep1_start: EpisodeStart):
    pos, yaw = ep1_to_ep0_local(
        transform, pose["position"], yaw_from_xyzw(pose["rotation_xyzw"]), ep1_start
    )
    return {
        "rgb": np.asarray(rgb, dtype=np.uint8)[..., :3],
        "depth": np.asarray(depth, dtype=np.float32),
        "position": np.asarray(pos, dtype=np.float32),
        "heading": np.asarray([yaw], dtype=np.float32),
    }


def run_one(*, env, habitat_config, episode, row, gridmm, args) -> dict[str, Any]:
    from membridge.habitat_runtime.online_adapter import OnlineINHeritSGBuilder

    session_id, episode_id = row["session_id"], row["episode_id"]
    destination = args.output_root / "sessions" / session_id / "stages/M1"
    if destination.exists() and not args.skip_existing:
        raise FileExistsError(f"M1 already exists: {destination}")
    work_root = args.output_root / ".working" / f"{session_id}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    work_root.mkdir(parents=True)
    install_memory_environment(args, work_root, row["m0"])

    env.episode_iterator = etp.RepeatEpisodeIterator(episode)
    observations = env.reset()
    initial_pose = pose_from_env(env, 0, 0)
    adapter = OnlineINHeritSGBuilder(
        etp.sensor_memory_config_from_protocol(args.sim_gpu_id),
        width=etp.PROTOCOL_WIDTH,
        height=etp.PROTOCOL_HEIGHT,
        save_dir=str(work_root),
        device=args.device,
    )
    adapter.start_episode(episode_id, scene_id=str(episode.scene_id), source_episode_id=episode_id)
    run_dir = Path(adapter.pipeline.run_dir).resolve()

    navigator = MemoryNavigator(
        adapter=adapter,
        env=env,
        habitat_config=habitat_config,
        decision_log=run_dir / "sg_navigation_decisions.jsonl",
    )
    navigator.current_episode_id = episode_id
    heading = yaw_from_xyzw(initial_pose["rotation_xyzw"])
    ep0_start = load_ep0_start(row["m0"], session_id)
    ep1_live = EpisodeStart(
        xyz=np.asarray(initial_pose["position"], dtype=np.float64),
        yaw=float(heading),
        episode_id=str(episode_id),
        role="ep1",
    )
    gridmm.begin_episode(initial_pose["position"], heading)

    poses = [initial_pose]
    requested_actions: list[str] = []
    executed_actions: list[str] = []
    etp_actions = n_spin = n_lock_try = n_spin_bouts = 0
    n_sg_frames = 0
    stopped = False
    sg_ingested = False
    started_at, wall_start = utc_now(), time.monotonic()
    transform = None
    locked = False
    ok_useful = False
    lock_mode = "walk"
    lock_reason = "not_attempted"
    lock_at = None
    best_inlier = 0
    trans_err = yaw_err = None
    last_best = None
    traveled = 0.0
    last_xz = None
    prev_xz = None
    spinning = False
    last_spin_travel = -1e9
    query_win: deque = deque(maxlen=QUERY_WINDOW)
    rgbd_buf: list[dict[str, Any]] = []
    obs_holder = {"obs": observations}

    map_frames = load_m0_local_frames(row["m0"], session_id)
    models()
    map_pack = (
        map_frames,
        encode_rgbs([frame.rgb for frame in map_frames]),
        extract_map_superpoint(map_frames),
    )

    def ingest_aligned(rgb, depth, pose, step: int) -> None:
        nonlocal n_sg_frames
        adapter.process_observation(
            aligned_memory_observation(rgb, depth, pose, transform, ep1_live),
            step=step,
        )
        n_sg_frames += 1

    def flush_sg_history() -> None:
        nonlocal sg_ingested
        if sg_ingested or transform is None:
            return
        for index, item in enumerate(rgbd_buf):
            ingest_aligned(item["rgb"], item["depth"], item["pose"], index)
        sg_ingested = True

    def maybe_lock(frame) -> bool:
        nonlocal locked, ok_useful, lock_reason, lock_at, transform, best_inlier
        nonlocal last_xz, last_best, trans_err, yaw_err, n_lock_try
        if locked:
            return False
        n_lock_try += 1
        last_xz = (frame.x, frame.z)
        ok, reason, best = try_lock_frame(
            frame, map_pack[0], map_pack[1], map_pack[2], query_window=list(query_win)
        )
        last_best = best
        best_inlier = max(best_inlier, int(getattr(best, "n_inlier", 0) or 0))
        lock_reason = reason
        if not ok or best.transform is None:
            return False
        locked = True
        transform = best.transform
        lock_at = len(executed_actions)
        gt = gt_ep1_to_ep0(ep0_start.xyz, ep0_start.yaw, ep1_live.xyz, ep1_live.yaw)
        trans_err, yaw_rad = se2_error(best.transform, gt)
        yaw_err = math.degrees(yaw_rad)
        ok_useful = bool(trans_err <= 0.25 and abs(yaw_err) <= 10.0)
        lock_reason = f"{reason} step={lock_at}"
        flush_sg_history()
        return True

    def should_extra_spin() -> bool:
        if traveled - last_spin_travel < 1.0 and n_spin_bouts:
            return False
        return (
            last_best is not None
            and int(getattr(last_best, "n_inlier", 0) or 0) >= HIGH_INLIER
            and "not unique" in (lock_reason or "")
        )

    def buffer_obs(pose) -> None:
        rgb, depth = copy_rgbd(obs_holder["obs"])
        rgbd_buf.append({"rgb": rgb, "depth": depth, "pose": copy.deepcopy(pose)})
        if locked and transform is not None:
            if not sg_ingested:
                flush_sg_history()
            else:
                ingest_aligned(rgb, depth, pose, len(executed_actions))

    def try_lock_here(mode: str) -> bool:
        nonlocal lock_mode
        frame = live_frame(obs_holder["obs"], poses[-1], ep1_live, len(executed_actions))
        query_win.append(frame)
        if maybe_lock(frame):
            lock_mode = mode
            return True
        return False

    def after_move() -> None:
        nonlocal traveled, prev_xz, lock_mode
        frame = live_frame(obs_holder["obs"], poses[-1], ep1_live, len(executed_actions))
        if prev_xz is not None:
            traveled += math.hypot(frame.x - prev_xz[0], frame.z - prev_xz[1])
        prev_xz = (frame.x, frame.z)
        if locked or spinning:
            return
        moved = last_xz is None or math.hypot(frame.x - last_xz[0], frame.z - last_xz[1]) >= LOCK_EVERY_M
        if not moved:
            return
        query_win.append(frame)
        if maybe_lock(frame):
            lock_mode = f"{getattr(gridmm, 'nav_name', 'etpnav')}-walk+geom"
        elif should_extra_spin():
            extra_spin()

    def extra_spin() -> None:
        """Dual-peak 12x30° for geom lock only. Restore heading; do not feed ETPNav."""
        nonlocal n_spin, n_spin_bouts, last_spin_travel, spinning, lock_mode
        n_spin_bouts += 1
        last_spin_travel = traveled
        spinning = True
        sim = env.sim
        state = sim.get_agent_state()
        orig_pos = np.asarray(state.position, dtype=np.float32)
        orig_rot = state.rotation
        base_yaw = yaw_from_xyzw(poses[-1]["rotation_xyzw"])
        try:
            for index in range(SPIN_STEPS):
                if locked:
                    break
                yaw = wrap_angle(base_yaw + (index + 1) * (2.0 * math.pi / SPIN_STEPS))
                sim.set_agent_state(
                    orig_pos,
                    np.quaternion(np.cos(yaw / 2.0), 0.0, np.sin(yaw / 2.0), 0.0),
                )
                obs = sim.get_sensor_observations()
                n_spin += 1
                spin_pose = {
                    "position": [float(v) for v in orig_pos],
                    "rotation_xyzw": [
                        0.0,
                        float(np.sin(yaw / 2.0)),
                        0.0,
                        float(np.cos(yaw / 2.0)),
                    ],
                }
                frame = live_frame(obs, spin_pose, ep1_live, len(executed_actions))
                query_win.append(frame)
                if maybe_lock(frame):
                    lock_mode = f"{getattr(gridmm, 'nav_name', 'etpnav')}-walk+spin+geom"
                    break
        finally:
            sim.set_agent_state(orig_pos, orig_rot)
            obs_holder["obs"] = sim.get_sensor_observations()
            spinning = False

    def step_low(action: int, requested: str) -> None:
        nonlocal observations
        requested_actions.append(requested)
        executed_actions.append(ACTION_NAMES[int(action)])
        observations = env.step(int(action))
        obs_holder["obs"] = observations
        pose = pose_from_env(env, len(poses), len(executed_actions))
        poses.append(pose)
        buffer_obs(pose)
        after_move()

    def execute_waypoint(angle_rad: float, distance_m: float) -> None:
        sim = env.sim
        state = sim.get_agent_state()
        rotation = state.rotation
        if hasattr(rotation, "imag") and hasattr(rotation, "real"):
            theta = np.arctan2(rotation.imag[1], rotation.real) + angle_rad / 2.0
        else:
            theta = angle_rad / 2.0
        snapped = np.quaternion(np.cos(theta), 0.0, np.sin(theta), 0.0)
        sim.set_agent_state(state.position, snapped)
        steps = max(1, int(distance_m // FORWARD_STEP_M)) if distance_m >= FORWARD_STEP_M else 0
        if steps == 0:
            pose = pose_from_env(env, len(poses), len(poses) - 1)
            poses[-1] = pose
            obs_holder["obs"] = env.sim.get_sensor_observations()
            rgb, depth = copy_rgbd(obs_holder["obs"])
            if rgbd_buf:
                rgbd_buf[-1] = {"rgb": rgb, "depth": depth, "pose": copy.deepcopy(pose)}
            elif locked and transform is not None:
                ingest_aligned(rgb, depth, pose, len(executed_actions))
            after_move()
            return
        for _ in range(steps):
            if env.episode_over or len(executed_actions) >= args.max_steps:
                return
            step_low(Actions.MOVE_FORWARD, "ETPNAV:MOVE_FORWARD")

    rgb0, depth0 = copy_rgbd(observations)
    rgbd_buf.append({"rgb": rgb0, "depth": depth0, "pose": copy.deepcopy(initial_pose)})
    frame0 = live_frame(observations, initial_pose, ep1_live, 0)
    last_xz = (frame0.x, frame0.z)
    prev_xz = (frame0.x, frame0.z)
    try_lock_here("start")

    def make_rollout_record() -> dict[str, Any]:
        metrics = env.get_metrics()
        ne = etp.metric_value(metrics, "distance_to_goal")
        oracle_ne = etp.metric_value(metrics, "oracle_navigation_error")
        return {
            "schema_version": "etpnav_level3_ep1_local_loc_rollout_v1",
            "phase": "level3_ep1",
            "episode_id": episode_id,
            "scene_id": str(episode.scene_id),
            "session_id": session_id,
            "role": "ep1",
            "coordinate_frame": "ep0_local_after_lock" if locked else "ep1_local",
            "instruction": row["instruction"],
            "route": row["route"],
            "localization": {
                "locked": locked,
                "ok_useful": ok_useful,
                "lock_mode": lock_mode,
                "lock_reason": lock_reason,
                "lock_at_step": lock_at,
                "n_lock_try": n_lock_try,
                "n_spin": n_spin,
                "n_spin_bouts": n_spin_bouts,
                "best_inlier": best_inlier,
                "trans_err_m": trans_err,
                "yaw_err_deg": yaw_err,
                "K": int(lock_at if lock_at is not None else len(executed_actions)),
                "sg_frames_ingested": n_sg_frames,
                "protocol": "no_start_spin_no_godview_1m_dualpeak_12x30",
            },
            "poses": poses,
            "positions": [pose["position"] for pose in poses],
            "rotations_xyzw": [pose["rotation_xyzw"] for pose in poses],
            "requested_actions": requested_actions,
            "executed_actions": executed_actions,
            "stopped": stopped,
            "timed_out": bool(not stopped and len(executed_actions) >= args.max_steps),
            "steps": len(executed_actions),
            "rgbd_persisted": False,
            "memory_module": {
                "read_stage": "M0",
                "write_stage": "M1",
                "m0": str(row["m0"].resolve()),
                "m1": str(destination.resolve()),
                "semantic_sg_updated": False,
                "world_navigation_map_updated_from_live_rgbd": bool(sg_ingested),
                "trajectory_graph_updated": bool(sg_ingested),
                "sam3_loaded": False,
                "dinov3_loaded": False,
                "rgbd_ingested_only_after_lock": True,
                "aligned_to_m0_after_lock": bool(sg_ingested),
                "navigation_controller_version": NAVIGATION_CONTROLLER_VERSION + "+l3_local_loc",
            },
            "metrics": {
                "ne": ne,
                "distance_to_goal": ne,
                "success": None if ne is None else float(ne <= 3.0),
                "habitat_success": etp.metric_value(metrics, "success"),
                "spl": etp.metric_value(metrics, "spl"),
                "oracle_navigation_error": oracle_ne,
                "oracle_success": None if oracle_ne is None else float(oracle_ne <= 3.0),
            },
            "timing": {
                "started_at": started_at,
                "finished_at": utc_now(),
                "wall_seconds": time.monotonic() - wall_start,
            },
        }

    try:
        while len(executed_actions) < args.max_steps and not env.episode_over:
            current_pose = poses[-1]
            heading = yaw_from_xyzw(current_pose["rotation_xyzw"])
            plan_pos, plan_yaw = current_pose["position"], heading
            if transform is not None:
                plan_pos, plan_yaw = ep1_to_ep0_local(
                    transform, current_pose["position"], yaw_from_xyzw(current_pose["rotation_xyzw"]), ep1_live
                )
            navigator.update_pose(plan_pos, plan_yaw, len(executed_actions))

            action, angle, distance = gridmm.observe_and_act(
                env, row["instruction"], current_pose["position"], heading, act=True
            )
            label = getattr(gridmm, "nav_name", "etpnav").upper()
            if getattr(gridmm, "discrete_actions", False):
                etp_actions += 1
                if int(action) == int(Actions.STOP):
                    requested_actions.append(f"{label}:STOP")
                    executed_actions.append("STOP")
                    observations = env.step(int(Actions.STOP))
                    obs_holder["obs"] = observations
                    pose = pose_from_env(env, len(poses), len(executed_actions))
                    poses.append(pose)
                    buffer_obs(pose)
                    stopped = True
                    break
                step_low(int(action), f"{label}:{ACTION_NAMES[int(action)]}")
                if env.episode_over:
                    stopped = True
                    break
                continue
            if action == Actions.STOP or (angle is None and distance is None):
                requested_actions.append("ETPNAV:STOP")
                executed_actions.append("STOP")
                observations = env.step(int(Actions.STOP))
                obs_holder["obs"] = observations
                pose = pose_from_env(env, len(poses), len(executed_actions))
                poses.append(pose)
                buffer_obs(pose)
                etp_actions += 1
                stopped = True
                break
            requested_actions.append(f"ETPNAV:HIGHTOLOW:{angle:.3f}:{distance:.3f}")
            executed_actions.append("HIGHTOLOW")
            etp_actions += 1
            execute_waypoint(float(angle), float(distance))
            if env.episode_over:
                stopped = True
                break

        if locked:
            adapter.finish_episode()
            if adapter.failed:
                raise RuntimeError("memory_module marked EP1 increment failed")
        else:
            try:
                adapter.abort_episode()
            except Exception:
                pass

        result = make_rollout_record()
        result["sg_handoff"] = {
            "etpnav_actions": etp_actions,
            "etpnav_actor_steps": gridmm.actor_steps,
            "loc_ok": ok_useful,
            "sg_incremented_after_lock": sg_ingested,
            "fallback_reason": None if locked else "not_locked",
            "navigation_controller_version": NAVIGATION_CONTROLLER_VERSION + "+l3_local_loc",
        }
        if not locked:
            atomic_json(work_root / "rollout.json", result)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copytree(row["m0"], destination)
            atomic_json(destination / "rollout.json", result)
        else:
            atomic_json(run_dir / "rollout.json", result)
            etp.publish_m1(
                row["m0"],
                run_dir,
                destination,
                {
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "route": row["route"],
                    "sg_handoff": result["sg_handoff"],
                    "coordinate_frame": "ep0_local_after_lock",
                },
            )
            manifest_path = destination / "stage_manifest.json"
            if manifest_path.is_file():
                manifest = read_json(manifest_path)
                manifest["schema_version"] = "etpnav_level3_memory_module_stage_v1"
                manifest["coordinate_frame"] = "ep0_local_after_lock"
                manifest["source_kind"] = (
                    f"{getattr(gridmm, 'nav_name', 'etpnav')}_ep1_live_local_loc_increment"
                )
                atomic_json(manifest_path, manifest)
        gridmm.save_graphmap(destination / "graphmap.pt")
    except Exception:
        try:
            adapter.abort_episode()
        except Exception:
            pass
        raise
    finally:
        if destination.exists():
            shutil.rmtree(work_root, ignore_errors=True)
    return result


def main() -> int:
    os.environ.setdefault("MEMBRIDGE_SKIP_PYDEPS", "1")
    args = parse_args()
    args.etpnav_root = args.etpnav_root.resolve()
    args.exp_config = args.exp_config.resolve()
    args.task_config = args.task_config.resolve()
    args.dataset = args.dataset.resolve()
    args.ckpt = args.ckpt.resolve()
    args.waypoint_ckpt = args.waypoint_ckpt.resolve()
    args.m0_root = args.m0_root.resolve()
    args.output_root = args.output_root.resolve()
    args.query_vocabulary = args.query_vocabulary.resolve()
    args.scenes_dir = args.scenes_dir.resolve()
    args.bert_path = args.bert_path.resolve()
    args.run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    selected = load_tasks(args)
    print(
        json.dumps(
            {
                "coordinate_frame": "ep1_local_until_lock_then_ep0_local",
                "m0_root": str(args.m0_root),
                "selected": len(selected),
                "shard": [args.shard_index, args.num_shards],
                "protocol": "no_start_spin_no_godview_1m_dualpeak_12x30",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0

    sys.path.insert(0, str(REPO_ROOT))
    install_full_exploration_overlay()
    np.random.seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    worker = args.worker_id.strip() or f"shard{args.shard_index:02d}"
    raw_log = args.output_root / "raw" / f"ep1_{worker}.jsonl"
    completed = {str(row["episode_id"]) for row in read_jsonl(raw_log)}

    env, habitat_config = etp.make_env(args, split="ep1")
    adapt = HERE / "adapt"
    if str(adapt) not in sys.path:
        sys.path.insert(0, str(adapt))
    from janus_policy import build_policy

    gridmm = build_policy(args, etp.ETPNavSuffixPolicy)
    episodes_by_id = {str(ep.episode_id): ep for ep in env.episodes}
    failures: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(selected, 1):
            episode_id = row["episode_id"]
            destination = args.output_root / "sessions" / row["session_id"] / "stages/M1"
            if args.skip_existing and destination.is_dir() and (destination / "rollout.json").is_file():
                recovered = read_json(destination / "rollout.json")
                append_jsonl(raw_log, recovered)
                completed.add(episode_id)
                continue
            if episode_id in completed:
                continue
            print(
                f"[ETPNAV-L3-EP1-LOC] {index}/{len(selected)} {row['session_id']} ep={episode_id}",
                flush=True,
            )
            episode = episodes_by_id.get(episode_id)
            if episode is None:
                raise RuntimeError(f"EP1 {episode_id} is not in {args.dataset}")
            try:
                result = run_one(
                    env=env,
                    habitat_config=habitat_config,
                    episode=episode,
                    row=row,
                    gridmm=gridmm,
                    args=args,
                )
                append_jsonl(raw_log, result)
                completed.add(episode_id)
                loc = result["localization"]
                print(
                    json.dumps(
                        {
                            "session": row["session_id"],
                            "locked": loc["locked"],
                            "ok_useful": loc["ok_useful"],
                            "lock_mode": loc["lock_mode"],
                            "trans_err_m": loc["trans_err_m"],
                            "yaw_err_deg": loc["yaw_err_deg"],
                            "ingested": loc["sg_frames_ingested"],
                            "metrics": result["metrics"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception as exc:
                import traceback

                failure = {
                    "session_id": row["session_id"],
                    "episode_id": episode_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "time": utc_now(),
                }
                failures.append(failure)
                append_jsonl(args.output_root / "raw" / f"failures_{worker}.jsonl", failure)
                print(f"[ETPNAV-L3-EP1-LOC] failed: {failure['error']}", file=sys.stderr, flush=True)
                if not args.continue_on_error:
                    raise
    finally:
        env.close()
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
