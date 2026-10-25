#!/usr/bin/env python3
"""Rebuild ETPNav Level-3 EP0 M0 in episode-local coordinates.

Native / nograph EP0 rollouts store Habitat-world XYZ+XYZW. This teleports
the same poses for protocol RGB-D, but feeds OnlineINHeritSGBuilder EP0-local
GPS (first pose = origin). Object nodes, occupancy, trajectory, and keyframes
are therefore in the local frame later EP1 lock / SG planning will use.

Habitat render still uses world poses. NPZ loaders keep the ``habitat_world``
label; ``stage_manifest`` and ``ep0_local_origin.json`` record the transform.

Default source is the empty-GraphMap nograph dump:

    /share/shared/shizk/level3_outputs/etpnav_nograph_local/ep0_20260912-l3-nograph
    == baseline_new/ETPNav/level3/outputs/ep0_latest
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bootstrap import install

install()
_RUNNERS = Path(__file__).resolve().parent
HERE = _RUNNERS.parent
REPO_ROOT = HERE.parent
JANUS_EXP = REPO_ROOT / "membridge" / "runners"
JANUS_BUILDER = JANUS_EXP / "build_m0_m1_from_poses.py"
if not JANUS_BUILDER.is_file():
    raise FileNotFoundError(f"Janus L3 pose builder missing: {JANUS_BUILDER}")

sys.path.insert(0, str(JANUS_EXP))
spec = importlib.util.spec_from_file_location(
    "janus_l3_build_m0_m1_from_poses", JANUS_BUILDER
)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load Janus L3 builder: {JANUS_BUILDER}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

LLHN = REPO_ROOT
loc = str(LLHN / "membridge" / "localization")
if loc not in sys.path:
    sys.path.insert(0, loc)

from se2 import world_agent_to_local, yaw_from_xyzw  # noqa: E402
from frames import rollout_world_poses  # noqa: E402

DEFAULT_EP0 = Path(
    "/share/shared/shizk/level3_outputs/etpnav_nograph_local/ep0_20260912-l3-nograph"
)
DEFAULT_OUTPUT = Path(
    "/share/shared/shizk/level3_outputs/etpnav_memory_module/ep0_from_poses_local"
)
DEFAULT_SENSOR = _RUNNERS / "sensor_config.yaml"
DEFAULT_DATASET = (
    HERE.parent / "ETPNav/data/datasets/HM3D_VLNCE_v5/level3/level3.json.gz"
)
EXPECTED_TASKS = 157


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep0-rollouts", type=Path, default=DEFAULT_EP0)
    parser.add_argument("--config", type=Path, default=DEFAULT_SENSOR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scenes-dir", type=Path, default=Path(os.environ.get("SCENES_DIR", "")))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--memory-module-root",
        type=Path,
        default=LLHN / "method" / "memory_module",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-gpu-id", type=int, default=0)
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--session-id", action="append", default=[])
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--worker-id", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--async-queue-size", type=int, default=256)
    parser.add_argument("--synchronous", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--sam3-ckpt",
        type=Path,
        default=Path(os.environ.get("MODELS_DIR", "") + "/facebook/sam3/sam3.pt"),
    )
    parser.add_argument(
        "--sam3-source-root",
        type=Path,
        default=Path(os.environ.get("MODELS_DIR", "") + "/source/sam3"),
    )
    parser.add_argument(
        "--dinov3-model",
        type=Path,
        default=Path(os.environ.get("MODELS_DIR", "") + "/facebook/dinov3"),
    )
    parser.add_argument(
        "--dinov3-repo",
        type=Path,
        default=Path(os.environ.get("MODELS_DIR", "") + "/source/dinov3"),
    )
    parser.add_argument(
        "--text-encoder",
        type=Path,
        default=Path(os.environ.get("MODELS_DIR", "") + "/all-MiniLM-L6-v2"),
    )
    parser.add_argument(
        "--query-vocabulary",
        type=Path,
        default=(
            LLHN
            / "membridge/semantic_scene_graph/config/"
            "level1_runtime_query_vocabulary_75.json"
        ),
    )
    parser.add_argument("--sam3-category-batch-size", type=int, default=15)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_TASKS)
    parser.add_argument(
        "--coordinate-frame",
        choices=("ep0_local", "habitat_world"),
        default="ep0_local",
        help="Level 2/3 use each episode's local frame. Level 1 stays in habitat world.",
    )
    args = parser.parse_args()
    args.stage = "m0"
    args.m0_root = args.output_root
    args.janus_results = args.ep0_rollouts
    args.janus_root = Path("/janus/src/JanusVLN")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.stride < 1 or args.async_queue_size < 1:
        parser.error("--stride and --async-queue-size must be positive")
    if args.sam3_category_batch_size < 1:
        parser.error("--sam3-category-batch-size must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    return args


def _row_from_native(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"native rollout is not an object: {path}")
    poses = payload.get("poses")
    if not isinstance(poses, list) or not poses:
        positions = payload.get("positions") or []
        rotations = payload.get("rotations_xyzw") or []
        if len(positions) != len(rotations) or not positions:
            raise RuntimeError(f"{path} has no xyz+xyzw poses")
        poses = [
            {
                "frame_index": index,
                "simulator_step": index,
                "position": position,
                "rotation_xyzw": rotation,
            }
            for index, (position, rotation) in enumerate(zip(positions, rotations))
        ]
    payload = dict(payload)
    # validate_role_row normalizes pose dictionaries and drops world_position.
    # Restore world xyz before passing through that normalization.
    payload["poses"] = rollout_world_poses(poses)
    payload["steps"] = max(len(poses) - 1, 0)
    payload["role"] = "ep0"
    payload["phase"] = "level3_ep0"
    payload["session_id"] = str(payload.get("session_id") or "")
    payload["episode_id"] = str(payload.get("episode_id") or "")
    payload["_source_jsonl"] = str(path.resolve())
    payload["_source_rollout"] = str(path.resolve())
    return base.validate_role_row(payload, "ep0")


def select_ep0_tasks(tasks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    session_ids = set(map(str, args.session_id))
    episode_ids = set(map(str, args.episode_id))
    known_sessions = {task["session_id"] for task in tasks}
    known_episodes = {task["ep0"]["episode_id"] for task in tasks}
    if session_ids - known_sessions:
        raise RuntimeError(f"Unknown Level-3 sessions: {sorted(session_ids - known_sessions)}")
    if episode_ids - known_episodes:
        raise RuntimeError(f"Unknown Level-3 episode ids: {sorted(episode_ids - known_episodes)}")
    selected = [
        task
        for task in tasks
        if (not session_ids or task["session_id"] in session_ids)
        and (not episode_ids or task["ep0"]["episode_id"] in episode_ids)
    ]
    selected = [
        task
        for index, task in enumerate(selected)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise RuntimeError("EP0 selection is empty")
    return selected


def load_ep0_tasks(root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    root = root.resolve()
    files = sorted(root.glob("sessions/*/stages/M0/rollout.json"))
    if not files:
        raise RuntimeError(f"No EP0 M0 rollouts under {root}")
    tasks: list[dict[str, Any]] = []
    for path in files:
        ep0 = _row_from_native(path)
        session_id = ep0["session_id"] or path.parents[2].name
        if not session_id:
            raise RuntimeError(f"{path} has empty session_id")
        ep0["session_id"] = session_id
        tasks.append(
            {
                "session_id": session_id,
                "scene_id": ep0["scene_id"],
                "ep0": ep0,
            }
        )
    tasks.sort(key=lambda item: item["session_id"])
    return tasks, files


def make_local_adapter_class(origin_xyz, origin_yaw: float, adapter_class):
    origin = np.asarray(origin_xyz, dtype=np.float64).reshape(3)
    yaw0 = float(origin_yaw)

    class LocalFrameSGBuilder(adapter_class):
        def process_observation(self, observation, step=0):
            obs = dict(observation)
            world = np.asarray(obs["position"], dtype=np.float64).reshape(3)
            heading = np.asarray(obs["heading"], dtype=np.float64).reshape(-1)
            x, y, z, yaw = world_agent_to_local(world, float(heading[0]), origin, yaw0)
            obs["position"] = np.asarray([x, y, z], dtype=np.float32)
            obs["heading"] = np.asarray([yaw], dtype=np.float32)
            return super().process_observation(obs, step=step)

    LocalFrameSGBuilder.__name__ = "LocalFrameSGBuilder"
    return LocalFrameSGBuilder


def stamp_local_m0(m0: Path, task: dict[str, Any], origin_xyz, origin_yaw: float) -> None:
    origin = [float(v) for v in np.asarray(origin_xyz, dtype=np.float64).reshape(3)]
    yaw0 = float(origin_yaw)
    row = task["ep0"]
    payload = {
        "schema_version": "etpnav_level3_ep0_local_origin_v1",
        "coordinate_frame": "ep0_local",
        "ep0_origin_world_xyz": origin,
        "ep0_origin_world_yaw": yaw0,
        "session_id": task["session_id"],
        "episode_id": row["episode_id"],
        "source_rollout": row.get("_source_rollout"),
    }
    base.atomic_json(m0 / "ep0_local_origin.json", payload)

    frames = m0 / "frames" / "confirmed_keyframes.jsonl"
    if frames.is_file():
        rewritten = []
        for line in frames.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            item["coordinate_frame"] = "ep0_local"
            item["source_world_coordinate_frame"] = "habitat_world"
            item["ep0_origin_world_xyz"] = origin
            item["ep0_origin_world_yaw"] = yaw0
            rewritten.append(json.dumps(item, ensure_ascii=False))
        frames.write_text("\n".join(rewritten) + ("\n" if rewritten else ""), encoding="utf-8")

    path = m0 / "stage_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "etpnav_level3_memory_module_stage_v1"
    manifest["source_kind"] = "etpnav_nograph_saved_pose_rgbd_rerender_ep0_local"
    manifest["coordinate_frame"] = "ep0_local"
    manifest["npz_coordinate_frame_label"] = "habitat_world"
    manifest["ep0_origin_world_xyz"] = origin
    manifest["ep0_origin_world_yaw"] = yaw0
    manifest["method"] = "ETPNav"
    manifest["protocol_rgbd_used_for_m0"] = [960, 720]
    manifest["protocol_depth_max_m"] = 4.0
    base.atomic_json(path, manifest)


def main() -> int:
    args = parse_args()
    args.ep0_rollouts = args.ep0_rollouts.resolve()
    args.config = args.config.resolve()
    args.dataset = args.dataset.resolve()
    args.scenes_dir = args.scenes_dir.resolve()
    args.output_root = args.output_root.resolve()
    args.m0_root = args.output_root
    args.memory_module_root = args.memory_module_root.resolve()
    args.query_vocabulary = args.query_vocabulary.resolve()
    args.sam3_ckpt = args.sam3_ckpt.resolve()
    args.sam3_source_root = args.sam3_source_root.resolve()
    args.dinov3_model = args.dinov3_model.resolve()
    args.dinov3_repo = args.dinov3_repo.resolve()
    args.text_encoder = args.text_encoder.resolve()

    tasks, source_files = load_ep0_tasks(args.ep0_rollouts)
    if args.expected_count and not args.session_id and not args.episode_id:
        if len(tasks) != args.expected_count:
            raise RuntimeError(
                f"Expected {args.expected_count} EP0 sessions, found {len(tasks)}"
            )
    selected = select_ep0_tasks(tasks, args)
    print(
        json.dumps(
            {
                "coordinate_frame": "ep0_local",
                "source_file_count": len(source_files),
                "completed_ep0": len(tasks),
                "selected_ep0": len(selected),
                "stage": "m0",
                "num_shards": int(args.num_shards),
                "shard_index": int(args.shard_index),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0

    base.require_paths(args, selected)
    sys.path.insert(0, str(LLHN))
    base.install_memory_environment(args, args.output_root / ".working/bootstrap")
    from membridge.habitat_runtime.bootstrap import install_environment_aliases
    from membridge.habitat_runtime.online_adapter import OnlineINHeritSGBuilder
    from membridge.trajectory_graph.habitat_adapter import (
        heading_from_quaternion_xyzw,
    )

    install_environment_aliases()
    sensor_config = base.load_sensor_config(args.config)
    renderer = base.HabitatPoseRenderer(sensor_config, args.scenes_dir, args.sim_gpu_id)

    args.output_root.mkdir(parents=True, exist_ok=True)
    worker_id = args.worker_id.strip()
    if not worker_id and args.num_shards > 1:
        worker_id = f"shard{args.shard_index:02d}-of-{args.num_shards:02d}"
    safe_worker_id = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in worker_id
    )
    suffix = f".{safe_worker_id}" if safe_worker_id else ""
    results_log = args.output_root / f"build_results{suffix}.jsonl"
    shared_models = None
    failures: list[dict[str, Any]] = []
    try:
        for index, task in enumerate(selected, 1):
            task["ep0"]["poses"] = rollout_world_poses(task["ep0"]["poses"])
            start = task["ep0"]["poses"][0]
            origin_xyz = start["position"]
            origin_yaw = yaw_from_xyzw(start["rotation_xyzw"])
            print(
                f"[ETPNAV-L3-EP0-LOCAL] session {index}/{len(selected)}: "
                f"{task['session_id']} episode={task['ep0']['episode_id']}",
                flush=True,
            )
            try:
                adapter_class = OnlineINHeritSGBuilder
                if args.coordinate_frame == "ep0_local":
                    adapter_class = make_local_adapter_class(
                        origin_xyz, origin_yaw, OnlineINHeritSGBuilder
                    )
                else:
                    for pose in task["ep0"]["poses"]:
                        if pose.get("world_position"):
                            pose["position"] = pose["world_position"]
                shared_models, result = base.build_one(
                    args=args,
                    renderer=renderer,
                    sensor_config=sensor_config,
                    task=task,
                    shared_models=shared_models,
                    adapter_class=adapter_class,
                    heading_from_quaternion=heading_from_quaternion_xyzw,
                )
                if result.get("status") == "complete" and result.get("m0"):
                    if args.coordinate_frame == "ep0_local":
                        stamp_local_m0(Path(result["m0"]), task, origin_xyz, origin_yaw)
                    result["coordinate_frame"] = args.coordinate_frame
            except Exception as exc:
                result = {
                    "status": "failed",
                    "session_id": task["session_id"],
                    "episode_id": task["ep0"]["episode_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(result)
                base.append_jsonl(results_log, result)
                if not args.continue_on_error:
                    raise
                print(f"[ETPNAV-L3-EP0-LOCAL] failed: {result}", file=sys.stderr, flush=True)
                continue
            base.append_jsonl(results_log, result)
            print(
                f"[ETPNAV-L3-EP0-LOCAL] {result['status']}: {result.get('m0')}",
                flush=True,
            )
    finally:
        renderer.close()

    summary = {
        "schema_version": "etpnav_level3_ep0_offline_memory_local_summary_v1",
        "method": "ETPNav",
        "coordinate_frame": args.coordinate_frame,
        "source_rollouts": str(args.ep0_rollouts),
        "output_root": str(args.output_root),
        "requested_count": len(selected),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "worker_id": worker_id or None,
        "sam3_category_batch_size": int(args.sam3_category_batch_size),
        "failure_count": len(failures),
        "failures": failures,
        "etpnav_completed_ep0_count": len(tasks),
    }
    base.atomic_json(args.output_root / f"build_summary{suffix}.json", summary)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
