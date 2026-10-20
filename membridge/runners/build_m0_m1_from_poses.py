#!/usr/bin/env python3
"""Rebuild Level-3 memory_module M0 then continue that checkpoint into M1.

Level-3 protocol is sequential, not independent query branches:

    M0 = Update(empty, actual EP0 observations)
    M1 = Update(M0, actual EP1 observations)

JanusVLN's audited rollouts store Habitat world poses (and not a
memory_module checkpoint).  This program re-renders RGB-D at every saved
pose and feeds the observations into ``OnlineINHeritSGBuilder``.

Default ``--stage both`` keeps one builder alive across EP0 and EP1 for
each task: after EP0 it snapshots a self-contained M0, then continues the
same adapter with EP1 and publishes M1.  ``--stage m1`` instead preloads a
previously published M0 and only runs EP1.  Existing complete stages are
never overwritten.

    OUTPUT/sessions/<task_id>/stages/M0/
    OUTPUT/sessions/<task_id>/stages/M1/
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
LEVEL2 = HERE
sys.path.insert(0, str(LEVEL2))
from build_ep0_memory_from_poses import (  # noqa: E402
    CORE_STAGE_FILES,
    HabitatPoseRenderer,
    OPTIONAL_STAGE_FILES,
    atomic_json,
    append_jsonl,
    device_index,
    finite_vector,
    load_sensor_config,
    memory_config,
    render_observation,
    sha256_file,
    sha256_json,
)


DEFAULT_EXP_ROOT = Path("/home/wangyj/projects/janusvln_hm3dsem_benchmark")
DEFAULT_JANUS_ROOT = DEFAULT_EXP_ROOT / "src/JanusVLN"
DEFAULT_RESULT_FILE = (
    DEFAULT_EXP_ROOT
    / "submissions/janusvln_level3_advisor_submission_20260907"
    / "trajectories/level3_full_rollouts.jsonl.gz"
)
DEFAULT_CONFIG = DEFAULT_JANUS_ROOT / "config/vln_level3_protocol.yaml"
DEFAULT_DATASET = (
    DEFAULT_EXP_ROOT / "data/level3_v34_20260906/habitat/level3/level3.json.gz"
)
DEFAULT_SCENES = DEFAULT_JANUS_ROOT / "data/scene_datasets"
DEFAULT_OUTPUT = Path(
    "/share/shared/shizk/level3_outputs/janus_memory_module/m0_m1_from_poses"
)
EXPECTED_TASKS = 157


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--janus-results",
        type=Path,
        default=DEFAULT_RESULT_FILE,
        help="Level-3 rollout JSONL(.gz) or a JanusVLN run/submission directory",
    )
    parser.add_argument("--janus-root", type=Path, default=DEFAULT_JANUS_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scenes-dir", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--m0-root",
        type=Path,
        help="Published M0 root for --stage m1; defaults to --output-root",
    )
    parser.add_argument("--memory-module-root", type=Path, default=REPO_ROOT / "memory_module")
    parser.add_argument(
        "--stage",
        choices=("m0", "m1", "both"),
        default="both",
        help="m0: EP0 only; m1: continue from published M0; both: snapshot M0 then continue",
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
            REPO_ROOT
            / "memory_module/semantic_scene_graph/config/"
            "level1_runtime_query_vocabulary_75.json"
        ),
    )
    parser.add_argument("--sam3-category-batch-size", type=int, default=15)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_TASKS)
    args = parser.parse_args()
    if args.m0_root is None:
        args.m0_root = args.output_root
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.stride < 1 or args.async_queue_size < 1:
        parser.error("--stride and --async-queue-size must be positive")
    if args.sam3_category_batch_size < 1:
        parser.error("--sam3-category-batch-size must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.expected_count < 0:
        parser.error("--expected-count must be non-negative")
    return args


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        lines = handle.readlines()
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines):
                break
            raise
        if not isinstance(row, dict):
            raise RuntimeError(f"JSONL row must be an object: {path}:{index}")
        row["_source_jsonl"] = str(path.resolve())
        yield row


def read_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def discover_rollout_files(source: Path) -> list[Path]:
    source = source.resolve()
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"JanusVLN result source does not exist: {source}")
    patterns = (
        "trajectories/level3_full_rollouts.jsonl.gz",
        "level3_full_rollouts.jsonl.gz",
        "**/level3_full_rollouts.jsonl.gz",
        "batches/batch_*/raw/level3_rank*.jsonl",
        "raw/level3_rank*.jsonl",
        "**/raw/level3_rank*.jsonl",
    )
    for pattern in patterns:
        files = sorted(set(source.glob(pattern)))
        if files:
            return files
    raise FileNotFoundError(f"No Level-3 rollout JSONL found below {source}")


def normalize_poses(raw_poses: Any, episode_id: str, steps: int) -> list[dict[str, Any]]:
    if not isinstance(raw_poses, list) or not raw_poses:
        raise RuntimeError(f"{episode_id} has no saved poses")
    if len(raw_poses) != steps + 1:
        raise RuntimeError(
            f"{episode_id} has {len(raw_poses)} poses for {steps} actions"
        )
    poses = []
    for index, pose in enumerate(raw_poses):
        if not isinstance(pose, dict):
            raise RuntimeError(f"{episode_id} pose {index} is not an object")
        position = finite_vector(
            pose.get("position"), 3, f"{episode_id} pose {index} position"
        )
        rotation = finite_vector(
            pose.get("rotation_xyzw"), 4, f"{episode_id} pose {index} rotation"
        )
        norm = float(np.linalg.norm(rotation))
        if norm <= 1.0e-12 or abs(norm - 1.0) > 1.0e-3:
            raise RuntimeError(
                f"{episode_id} pose {index} has non-unit quaternion norm {norm}"
            )
        poses.append(
            {
                "frame_index": index,
                "simulator_step": int(pose.get("simulator_step", index)),
                "position": position,
                "rotation_xyzw": [value / norm for value in rotation],
            }
        )
    return poses


def validate_role_row(row: dict[str, Any], role: str) -> dict[str, Any]:
    expected_phase = f"level3_{role}"
    if str(row.get("role")) != role or str(row.get("phase")) != expected_phase:
        raise RuntimeError(f"expected {expected_phase}, got {row.get('phase')}/{row.get('role')}")
    episode_id = str(row.get("episode_id") or "")
    session_id = str(row.get("session_id") or "")
    scene_id = str(row.get("scene_id") or "")
    if not episode_id or not session_id or not scene_id:
        raise RuntimeError(f"{role} row lacks identity fields: {row.get('_source_jsonl')}")
    result = dict(row)
    result["episode_id"] = episode_id
    result["session_id"] = session_id
    result["scene_id"] = scene_id
    result["poses"] = normalize_poses(row.get("poses"), episode_id, int(row.get("steps", -1)))
    return result


def load_tasks(source: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    files = discover_rollout_files(source)
    by_role: dict[str, dict[str, dict[str, Any]]] = {"ep0": {}, "ep1": {}}
    fingerprints: dict[tuple[str, str], str] = {}
    for path in files:
        for row in iter_jsonl(path):
            role = str(row.get("role") or "")
            if role not in by_role:
                continue
            row = validate_role_row(row, role)
            key = (row["session_id"], row["episode_id"])
            fingerprint = sha256_json(
                {
                    "episode_id": row["episode_id"],
                    "session_id": row["session_id"],
                    "scene_id": row["scene_id"],
                    "poses": row["poses"],
                }
            )
            if key in fingerprints and fingerprints[key] != fingerprint:
                raise RuntimeError(
                    f"Conflicting completed rollouts for {role} {row['episode_id']}"
                )
            fingerprints[key] = fingerprint
            by_role[role][row["session_id"]] = row
    missing = sorted(set(by_role["ep0"]) ^ set(by_role["ep1"]))
    if missing:
        preview = ", ".join(missing[:12])
        suffix = "" if len(missing) <= 12 else f"; ... and {len(missing)-12} more"
        raise RuntimeError(
            f"EP0/EP1 session mismatch for {len(missing)} tasks: {preview}{suffix}"
        )
    tasks = []
    for session_id in sorted(by_role["ep0"]):
        ep0 = by_role["ep0"][session_id]
        ep1 = by_role["ep1"][session_id]
        if ep0["scene_id"] != ep1["scene_id"]:
            raise RuntimeError(
                f"{session_id} changes scene between EP0 and EP1: "
                f"{ep0['scene_id']} vs {ep1['scene_id']}"
            )
        tasks.append({"session_id": session_id, "scene_id": ep0["scene_id"], "ep0": ep0, "ep1": ep1})
    if not tasks:
        raise RuntimeError("No completed Level-3 EP0/EP1 task pairs were found")
    return tasks, files


def validate_dataset(dataset_path: Path, tasks: list[dict[str, Any]]) -> dict[str, int]:
    payload = read_json(dataset_path)
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, list):
        raise RuntimeError(f"Level-3 dataset has no episodes list: {dataset_path}")
    dataset_tasks: dict[str, dict[str, dict[str, Any]]] = {}
    for episode in episodes:
        info = episode.get("info") or {}
        episode_id = str(episode.get("episode_id") or "")
        task_id = episode_id.rsplit("_", 1)[0] if episode_id.endswith(("_ep0", "_ep1", "_ep2", "_ep3", "_ep4")) else str(info.get("session_id") or "")
        role = str(info.get("role") or info.get("lifelong_role") or info.get("level3_role") or "")
        if not task_id or role not in {"ep0", "ep1", "ep2"}:
            raise RuntimeError(f"Invalid Level-3 dataset episode: {episode.get('episode_id')}")
        dataset_tasks.setdefault(task_id, {})
        if role in dataset_tasks[task_id]:
            raise RuntimeError(f"Duplicate {role} in dataset task {task_id}")
        dataset_tasks[task_id][role] = episode
    for task_id, roles in dataset_tasks.items():
        if set(roles) != {"ep0", "ep1", "ep2"}:
            raise RuntimeError(f"Dataset task {task_id} roles={sorted(roles)}")
    if len(dataset_tasks) != EXPECTED_TASKS:
        raise RuntimeError(
            f"Level-3 dataset has {len(dataset_tasks)} tasks, expected {EXPECTED_TASKS}"
        )
    for task in tasks:
        session_id = task["session_id"]
        dataset_task = dataset_tasks.get(session_id)
        if dataset_task is None:
            raise RuntimeError(f"Rollout task {session_id} is absent from {dataset_path}")
        for role in ("ep0", "ep1"):
            dataset_episode = dataset_task[role]
            rollout = task[role]
            expected_id = str(dataset_episode.get("episode_id") or "")
            if rollout["episode_id"] != expected_id:
                raise RuntimeError(
                    f"{session_id}/{role} episode mismatch: "
                    f"rollout={rollout['episode_id']} dataset={expected_id}"
                )
            dataset_scene = str(dataset_episode.get("scene_id") or "")
            rollout_scene = rollout["scene_id"]
            if not (
                rollout_scene == dataset_scene
                or rollout_scene.endswith("/" + dataset_scene)
                or dataset_scene.endswith("/" + rollout_scene)
            ):
                raise RuntimeError(
                    f"{session_id}/{role} scene mismatch: "
                    f"rollout={rollout_scene} dataset={dataset_scene}"
                )
    return {
        "dataset_episode_count": len(episodes),
        "dataset_task_count": len(dataset_tasks),
        "source_task_count": len(tasks),
    }


def select_tasks(tasks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    session_ids = set(map(str, args.session_id))
    episode_ids = set(map(str, args.episode_id))
    known_sessions = {task["session_id"] for task in tasks}
    known_episodes = {
        task[role]["episode_id"] for task in tasks for role in ("ep0", "ep1")
    }
    if session_ids - known_sessions:
        raise RuntimeError(f"Unknown Level-3 sessions: {sorted(session_ids-known_sessions)}")
    if episode_ids - known_episodes:
        raise RuntimeError(f"Unknown Level-3 episode ids: {sorted(episode_ids-known_episodes)}")
    selected = [
        task
        for task in tasks
        if (not session_ids or task["session_id"] in session_ids)
        and (
            not episode_ids
            or task["ep0"]["episode_id"] in episode_ids
            or task["ep1"]["episode_id"] in episode_ids
        )
    ]
    selected = [
        task
        for index, task in enumerate(selected)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise RuntimeError("Level-3 task selection is empty")
    return selected


def require_paths(args: argparse.Namespace, tasks: list[dict[str, Any]]) -> None:
    required = [
        args.config,
        args.dataset,
        args.scenes_dir,
        args.memory_module_root / "__init__.py",
        args.memory_module_root / "habitat_runtime/online_adapter.py",
        args.memory_module_root
        / "semantic_scene_graph/legacy_builder_source/config_sam3.yaml",
        args.query_vocabulary,
        args.sam3_ckpt,
        args.sam3_source_root,
        args.dinov3_model,
        args.dinov3_repo,
        args.text_encoder,
    ]
    if args.stage in {"m1"}:
        for task in tasks:
            m0 = args.m0_root / "sessions" / task["session_id"] / "stages/M0"
            required.extend(m0 / name for name in CORE_STAGE_FILES)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Level-3 offline memory dependencies:\n  " + "\n  ".join(missing[:100])
        )


def install_memory_environment(
    args: argparse.Namespace, work_root: Path, preload_m0: Path | None = None
) -> None:
    values = {
        "INHERIT_SG_ENABLE": "1",
        "INHERIT_SG_OUTPUT_DIR": str(work_root),
        "INHERIT_SG_SESSION_NAME": "working_memory",
        "INHERIT_SG_CONFIG": str(
            args.memory_module_root
            / "semantic_scene_graph/legacy_builder_source/config_sam3.yaml"
        ),
        "INHERIT_SG_PRELOAD_DIR": "" if preload_m0 is None else str(preload_m0),
        "INHERIT_SG_PRELOAD_QUERY_ONLY": "0",
        "INHERIT_SG_PRELOAD_WORLD_MAP_ONLY": "0",
        "INHERIT_SG_PRELOAD_BUILDER_STATE_REQUIRED": "0" if preload_m0 is None else "1",
        "INHERIT_SG_PRELOAD_MEMORY_STAGE": "" if preload_m0 is None else "M0",
        "INHERIT_SG_CAPTURE_ONLY": "0",
        "INHERIT_SG_STRIDE": str(args.stride),
        "INHERIT_SG_KEYFRAME_FILTER": "1",
        "INHERIT_SG_ASYNC": "0" if args.synchronous else "1",
        "INHERIT_SG_ASYNC_QUEUE_SIZE": str(args.async_queue_size),
        "INHERIT_SG_ASYNC_DROP_POLICY": "block",
        "INHERIT_SG_FAIL_CLOSED": "1",
        "INHERIT_SG_SHARE_MODELS": "1",
        "INHERIT_SG_SAVE_CONFIRMED_KEYFRAMES": "1",
        "INHERIT_SG_SAVE_OFFLINE_RGBD": "0",
        "INHERIT_SG_QUERY_MIN_FINALIZED_EPISODES": "2",
        "INHERIT_SG_QUERY_VOCABULARY": str(args.query_vocabulary),
        "INHERIT_SG_TEXT_ENCODER": str(args.text_encoder),
        "INHERIT_SG_DINOV3_MODEL": str(args.dinov3_model),
        "INHERIT_SG_DINOV3_REPO": str(args.dinov3_repo),
        "INHERIT_SG_SAM3_CKPT": str(args.sam3_ckpt),
        "INHERIT_SG_SAM3_SOURCE_ROOT": str(args.sam3_source_root),
        "SAM3_SOURCE_ROOT": str(args.sam3_source_root),
        "INHERIT_SG_SAM3_CATEGORY_BATCH_SIZE": str(args.sam3_category_batch_size),
        "MEMORY_MODULE_DEBUG": "0",
        "INHERIT_SG_DEBUG": "0",
        "LIFELONG_WORLD_MEMORY_ENABLE": "1",
        "LIFELONG_WORLD_MEMORY_REQUIRED": "1",
        "LIFELONG_WORLD_MAP_RESOLUTION_M": "0.05",
        "LIFELONG_WORLD_MAP_CHUNK_SIZE": "128",
        "LIFELONG_WORLD_MAP_PIXEL_STRIDE": "8",
        "LIFELONG_WORLD_MAP_MAX_RAYS": "2048",
        "LIFELONG_WORLD_TRAJECTORY_SPACING_M": "0.25",
        "LIFELONG_WORLD_TRAJECTORY_MAX_EDGE_M": "1.0",
    }
    os.environ.update(values)


def stage_dir(root: Path, session_id: str, stage: str) -> Path:
    return root / "sessions" / session_id / "stages" / stage


def stage_complete(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in CORE_STAGE_FILES) and (
        path / "stage_manifest.json"
    ).is_file()


def snapshot_copy(source: str, destination: str) -> str:
    # Copy rather than hardlink: EP1 keeps writing into the live run_dir, and a
    # shared inode would mutate the already published M0 checkpoint.
    return shutil.copy2(source, destination)


def publish_run_dir(run_dir: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    if temporary.exists():
        raise FileExistsError(f"Stale publication directory exists: {temporary}")
    os.replace(run_dir, temporary)
    os.replace(temporary, destination)


def snapshot_run_dir(run_dir: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    if temporary.exists():
        raise FileExistsError(f"Stale snapshot directory exists: {temporary}")
    shutil.copytree(run_dir, temporary, copy_function=snapshot_copy)
    os.replace(temporary, destination)


def write_stage_manifest(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    task: dict[str, Any],
    row: dict[str, Any],
    stage: str,
    source_m0: Path | None,
    continuation: str,
    sensor_config: Any,
) -> dict[str, Any]:
    missing = [name for name in CORE_STAGE_FILES if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"{task['session_id']}/{row['episode_id']} {stage} lacks artifacts: {missing}"
        )
    artifact_names = list(CORE_STAGE_FILES) + [
        name for name in OPTIONAL_STAGE_FILES if (run_dir / name).is_file()
    ]
    role = "ep0" if stage == "M0" else "ep1"
    manifest = {
        "schema_version": "janusvln_level3_memory_module_stage_v1",
        "session_id": task["session_id"],
        "episode_id": row["episode_id"],
        "role": role,
        "memory_read": None if stage == "M0" else "M0",
        "memory_write": stage,
        "continuation": continuation,
        "source_kind": "janusvln_saved_pose_rgbd_rerender",
        "source_rollout_jsonl": row["_source_jsonl"],
        "source_pose_sha256": sha256_json(row["poses"]),
        "source_scene_id": row["scene_id"],
        "source_m0": None if source_m0 is None else str(source_m0.resolve()),
        "frame_count": len(row["poses"]),
        "trajectory_semantics": "initial_pose_then_pose_after_each_executed_action",
        "rgbd_persisted": False,
        "rgbd_rendered_online_from_saved_poses": True,
        "camera": {
            "width": int(sensor_config.rgb_sensor.width),
            "height": int(sensor_config.rgb_sensor.height),
            "hfov_degrees": float(sensor_config.rgb_sensor.hfov),
            "sensor_offset_xyz": list(map(float, sensor_config.rgb_sensor.position)),
            "depth_min_m": float(sensor_config.depth_sensor.min_depth),
            "depth_max_m": float(sensor_config.depth_sensor.max_depth),
            "depth_unit": "meter",
        },
        "janus_metrics": row.get("metrics") or {},
        "semantic_models_loaded_once_per_process": True,
        "world_memory_updated": True,
        "sam3_category_batch_size": int(args.sam3_category_batch_size),
        "async_queue_drained_before_publish": True,
        "artifacts_sha256": {name: sha256_file(run_dir / name) for name in artifact_names},
    }
    atomic_json(run_dir / "stage_manifest.json", manifest)
    return manifest


def make_adapter(
    *,
    args: argparse.Namespace,
    sensor_config: Any,
    work_root: Path,
    shared_models: Any,
    adapter_class: Any,
    preload_m0: Path | None,
):
    install_memory_environment(args, work_root, preload_m0)
    adapter = adapter_class(
        memory_config(sensor_config, device_index(args.device)),
        width=int(sensor_config.rgb_sensor.width),
        height=int(sensor_config.rgb_sensor.height),
        save_dir=str(work_root),
        device=args.device,
    )
    adapter.shared_models = shared_models
    return adapter


def feed_episode(
    *,
    adapter: Any,
    renderer: HabitatPoseRenderer,
    row: dict[str, Any],
    heading_from_quaternion: Any,
    label: str,
) -> None:
    episode_id = row["episode_id"]
    adapter.start_episode(
        episode_id,
        scene_id=row["scene_id"],
        source_episode_id=episode_id,
    )
    for ordinal, pose in enumerate(row["poses"]):
        rendered = render_observation(renderer, pose, episode_id)
        observation = {
            **rendered,
            "position": np.asarray(pose["position"], dtype=np.float32),
            "heading": np.asarray(
                [heading_from_quaternion(pose["rotation_xyzw"])], dtype=np.float32
            ),
        }
        adapter.process_observation(observation, step=pose["simulator_step"])
        if ordinal % 10 == 0 or ordinal + 1 == len(row["poses"]):
            print(
                f"{label} {row['session_id']}/{episode_id}: "
                f"{ordinal + 1}/{len(row['poses'])} poses",
                flush=True,
            )
    adapter.finish_episode()
    if adapter.failed:
        raise RuntimeError(f"memory_module marked {episode_id} failed")


def build_one(
    *,
    args: argparse.Namespace,
    renderer: HabitatPoseRenderer,
    sensor_config: Any,
    task: dict[str, Any],
    shared_models: Any,
    adapter_class: Any,
    heading_from_quaternion: Any,
) -> tuple[Any, dict[str, Any]]:
    session_id = task["session_id"]
    m0_destination = stage_dir(args.output_root, session_id, "M0")
    m1_destination = stage_dir(args.output_root, session_id, "M1")
    published_m0 = stage_dir(args.m0_root, session_id, "M0")
    m0_ready = stage_complete(m0_destination) or (
        args.m0_root.resolve() != args.output_root.resolve() and stage_complete(published_m0)
    )
    m1_ready = stage_complete(m1_destination)

    skip_m0 = args.stage in {"m1"} or (args.skip_existing and m0_ready)
    skip_m1 = args.stage == "m0" or (args.skip_existing and m1_ready)
    if args.stage in {"m0", "both"} and m0_destination.exists() and not skip_m0:
        raise FileExistsError(f"Refusing to replace existing M0: {m0_destination}")
    if args.stage in {"m1", "both"} and m1_destination.exists() and not skip_m1:
        raise FileExistsError(f"Refusing to replace existing M1: {m1_destination}")
    if skip_m0 and skip_m1:
        return shared_models, {
            "status": "skipped_existing",
            "session_id": session_id,
            "m0": str((m0_destination if stage_complete(m0_destination) else published_m0).resolve()),
            "m1": None if args.stage == "m0" else str(m1_destination.resolve()),
            "built": [],
        }
    if args.stage == "m1" and not stage_complete(published_m0):
        raise FileNotFoundError(f"M1 continuation requires a complete M0: {published_m0}")

    work_root = (
        args.output_root / ".working" / f"{session_id}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    if work_root.exists():
        raise FileExistsError(f"Stale working directory exists: {work_root}")
    work_root.mkdir(parents=True)
    preload = None if not skip_m0 else published_m0
    adapter = make_adapter(
        args=args,
        sensor_config=sensor_config,
        work_root=work_root,
        shared_models=shared_models,
        adapter_class=adapter_class,
        preload_m0=preload,
    )
    built: list[str] = []
    result: dict[str, Any] = {
        "status": "complete",
        "session_id": session_id,
        "built": built,
    }
    try:
        renderer.open_scene(task["scene_id"])
        if not skip_m0:
            print(f"[JANUS-L3-M0] {session_id} episode={task['ep0']['episode_id']}", flush=True)
            feed_episode(
                adapter=adapter,
                renderer=renderer,
                row=task["ep0"],
                heading_from_quaternion=heading_from_quaternion,
                label="[JANUS-L3-M0]",
            )
            shared_models = adapter.shared_models
            run_dir = Path(adapter.pipeline.run_dir).resolve()
            write_stage_manifest(
                run_dir=run_dir,
                args=args,
                task=task,
                row=task["ep0"],
                stage="M0",
                source_m0=None,
                continuation="empty_to_m0",
                sensor_config=sensor_config,
            )
            if args.stage == "m0":
                publish_run_dir(run_dir, m0_destination)
            else:
                snapshot_run_dir(run_dir, m0_destination)
            built.append("M0")
            result["m0"] = str(m0_destination.resolve())
            print(f"[JANUS-L3-M0] complete: {m0_destination}", flush=True)
        else:
            result["m0"] = str(
                (m0_destination if stage_complete(m0_destination) else published_m0).resolve()
            )
            print(f"[JANUS-L3-M0] skipped_existing: {result['m0']}", flush=True)

        if not skip_m1:
            print(f"[JANUS-L3-M1] {session_id} episode={task['ep1']['episode_id']}", flush=True)
            feed_episode(
                adapter=adapter,
                renderer=renderer,
                row=task["ep1"],
                heading_from_quaternion=heading_from_quaternion,
                label="[JANUS-L3-M1]",
            )
            shared_models = adapter.shared_models
            run_dir = Path(adapter.pipeline.run_dir).resolve()
            continuation = (
                "same_adapter_m0_to_m1" if not skip_m0 else "preload_m0_to_m1"
            )
            write_stage_manifest(
                run_dir=run_dir,
                args=args,
                task=task,
                row=task["ep1"],
                stage="M1",
                source_m0=m0_destination if stage_complete(m0_destination) else published_m0,
                continuation=continuation,
                sensor_config=sensor_config,
            )
            publish_run_dir(run_dir, m1_destination)
            built.append("M1")
            result["m1"] = str(m1_destination.resolve())
            print(f"[JANUS-L3-M1] complete: {m1_destination}", flush=True)
        elif args.stage != "m0":
            result["m1"] = str(m1_destination.resolve())
            print(f"[JANUS-L3-M1] skipped_existing: {m1_destination}", flush=True)

        shutil.rmtree(work_root, ignore_errors=True)
        if not built:
            result["status"] = "skipped_existing"
        return shared_models, result
    except Exception:
        try:
            adapter.abort_episode()
        except Exception:
            pass
        raise


def role_success(tasks: list[dict[str, Any]], role: str) -> tuple[int, int, float]:
    values = []
    for task in tasks:
        value = (task[role].get("metrics") or {}).get("success")
        if value is not None:
            values.append(float(value))
    successes = int(sum(values))
    return successes, len(values), successes / len(values) if values else math.nan


def main() -> None:
    args = parse_args()
    args.janus_results = args.janus_results.resolve()
    args.janus_root = args.janus_root.resolve()
    args.config = args.config.resolve()
    args.dataset = args.dataset.resolve()
    args.scenes_dir = args.scenes_dir.resolve()
    args.output_root = args.output_root.resolve()
    args.m0_root = args.m0_root.resolve()
    args.memory_module_root = args.memory_module_root.resolve()
    args.query_vocabulary = args.query_vocabulary.resolve()
    args.sam3_ckpt = args.sam3_ckpt.resolve()
    args.sam3_source_root = args.sam3_source_root.resolve()
    args.dinov3_model = args.dinov3_model.resolve()
    args.dinov3_repo = args.dinov3_repo.resolve()
    args.text_encoder = args.text_encoder.resolve()

    tasks, source_files = load_tasks(args.janus_results)
    dataset_stats = validate_dataset(args.dataset, tasks)
    if args.expected_count and not args.session_id and not args.episode_id:
        if len(tasks) != args.expected_count:
            raise RuntimeError(
                f"Expected {args.expected_count} Level-3 EP0/EP1 tasks, found {len(tasks)}"
            )
    selected = select_tasks(tasks, args)
    ep0_ok, ep0_n, ep0_rate = role_success(tasks, "ep0")
    ep1_ok, ep1_n, ep1_rate = role_success(tasks, "ep1")
    print(
        json.dumps(
            {
                "source_jsonl_count": len(source_files),
                "completed_tasks": len(tasks),
                "selected_tasks": len(selected),
                "stage": args.stage,
                "ep0_successes": ep0_ok,
                "ep0_success_rate": ep0_rate,
                "ep1_successes": ep1_ok,
                "ep1_success_rate": ep1_rate,
                "dataset": dataset_stats,
                "memory_chain": "empty -> M0 -> M1",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    require_paths(args, selected)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(args.janus_root))
    install_memory_environment(args, args.output_root / ".working/bootstrap")
    from membridge.habitat_runtime.bootstrap import install_environment_aliases
    from membridge.habitat_runtime.online_adapter import OnlineINHeritSGBuilder
    from membridge.trajectory_graph.habitat_adapter import heading_from_quaternion_xyzw

    install_environment_aliases()
    sensor_config = load_sensor_config(args.config)
    renderer = HabitatPoseRenderer(sensor_config, args.scenes_dir, args.sim_gpu_id)
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
            print(
                f"[JANUS-L3] task {index}/{len(selected)}: {task['session_id']} stage={args.stage}",
                flush=True,
            )
            try:
                shared_models, result = build_one(
                    args=args,
                    renderer=renderer,
                    sensor_config=sensor_config,
                    task=task,
                    shared_models=shared_models,
                    adapter_class=OnlineINHeritSGBuilder,
                    heading_from_quaternion=heading_from_quaternion_xyzw,
                )
            except Exception as exc:
                result = {
                    "status": "failed",
                    "session_id": task["session_id"],
                    "episode_id_ep0": task["ep0"]["episode_id"],
                    "episode_id_ep1": task["ep1"]["episode_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(result)
                append_jsonl(results_log, result)
                if not args.continue_on_error:
                    raise
                print(f"[JANUS-L3] failed: {result}", file=sys.stderr, flush=True)
                continue
            append_jsonl(results_log, result)
            print(f"[JANUS-L3] {result['status']}: {task['session_id']} built={result.get('built')}", flush=True)
    finally:
        renderer.close()

    summary = {
        "schema_version": "janusvln_level3_offline_memory_summary_v1",
        "source_results": str(args.janus_results),
        "output_root": str(args.output_root),
        "m0_root": str(args.m0_root),
        "stage": args.stage,
        "requested_count": len(selected),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "worker_id": worker_id or None,
        "sam3_category_batch_size": int(args.sam3_category_batch_size),
        "failure_count": len(failures),
        "failures": failures,
        "janus_ep0_success_count": ep0_ok,
        "janus_ep0_success_rate": ep0_rate,
        "janus_ep0_metric_count": ep0_n,
        "janus_ep1_success_count": ep1_ok,
        "janus_ep1_success_rate": ep1_rate,
        "janus_ep1_metric_count": ep1_n,
    }
    atomic_json(args.output_root / f"build_summary{suffix}.json", summary)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
