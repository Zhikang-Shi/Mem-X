#!/usr/bin/env python3
"""Re-render JanusVLN Level-2 EP0 poses and build memory_module M0 stages.

JanusVLN's Level-2 rollout JSONL stores the initial Habitat world pose followed
by the pose after every executed action, but it does not persist RGB-D frames.
This program loads the same HM3D scene, renders RGB-D at every saved pose, and
feeds the observations directly into ``OnlineINHeritSGBuilder``.  The published
layout is intentionally suitable for a later EP1 runner::

    OUTPUT/sessions/<session_id>/stages/M0/
        memory_kb.json
        builder_state.pkl
        spatial_graph.pkl                 # when emitted by the SG builder
        world_metric_map.npz
        trajectory_graph.npz
        stage_manifest.json

No JanusVLN model or Janus native ``past_key_values_vggt`` checkpoint is loaded.
SAM3/DINOv3 are loaded once per process and shared across all selected sessions.
The source run and scenes are read-only; an existing M0 is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_EXP_ROOT = Path("/home/wangyj/projects/janusvln_hm3dsem_benchmark")
DEFAULT_JANUS_ROOT = DEFAULT_EXP_ROOT / "src/JanusVLN"
DEFAULT_RUN_ROOT = (
    DEFAULT_EXP_ROOT / "runs/janusvln_extra_level2_protocol_20260902_batched"
)
DEFAULT_CONFIG = DEFAULT_JANUS_ROOT / "config/vln_level2_protocol.yaml"
DEFAULT_DATASET = DEFAULT_EXP_ROOT / "data/level2_v8/habitat/level2/level2.json.gz"
DEFAULT_SCENES = DEFAULT_JANUS_ROOT / "data/scene_datasets"
DEFAULT_OUTPUT = HERE / "outputs/ep0_memory_from_poses"

CORE_STAGE_FILES = (
    "memory_kb.json",
    "builder_state.pkl",
    "world_metric_map.npz",
    "trajectory_graph.npz",
)
OPTIONAL_STAGE_FILES = (
    "spatial_graph.pkl",
    "map_data.pkl",
    "episode_manifest.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--janus-results",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="Level-2 run root or one level2_rank*.jsonl file",
    )
    parser.add_argument("--janus-root", type=Path, default=DEFAULT_JANUS_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scenes-dir", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--memory-module-root", type=Path, default=REPO_ROOT / "memory_module")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-gpu-id", type=int, default=0)
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--session-id", action="append", default=[])
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Deterministically split the sorted EP0 selection across workers",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based worker shard selected from --num-shards",
    )
    parser.add_argument(
        "--worker-id",
        default="",
        help="Suffix for this worker's result and summary files",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--async-queue-size", type=int, default=256)
    parser.add_argument(
        "--synchronous",
        action="store_true",
        help="Disable the builder worker thread (slower but useful for debugging)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip complete existing M0 stages; never replaces them",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep later sessions running after preserving a failed working directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/select rollout rows without importing Habitat or loading models",
    )
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
            / "memory_module/semantic_scene_graph/config/level1_runtime_query_vocabulary_75.json"
        ),
    )
    parser.add_argument("--sam3-category-batch-size", type=int, default=5)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.stride < 1:
        parser.error("--stride must be positive")
    if args.async_queue_size < 1:
        parser.error("--async-queue-size must be positive")
    if args.sam3_category_batch_size < 1:
        parser.error("--sam3-category-batch-size must be positive")
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A killed process may leave only its final JSONL line incomplete.
            if index == len(lines):
                break
            raise
        if not isinstance(row, dict):
            raise RuntimeError(f"JSONL row must be an object: {path}:{index}")
        row["_source_jsonl"] = str(path.resolve())
        yield row


def discover_rollout_files(source: Path) -> list[Path]:
    source = source.resolve()
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"JanusVLN result source does not exist: {source}")
    patterns = (
        "batches/batch_*/raw/level2_rank*.jsonl",
        "raw/level2_rank*.jsonl",
        "**/raw/level2_rank*.jsonl",
    )
    for pattern in patterns:
        files = sorted(set(source.glob(pattern)))
        if files:
            return files
    raise FileNotFoundError(f"No level2_rank*.jsonl found below {source}")


def finite_vector(value: Any, size: int, label: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size or not np.isfinite(array).all():
        raise RuntimeError(f"{label} must contain {size} finite numbers")
    return [float(item) for item in array]


def validate_ep0_row(row: dict[str, Any]) -> dict[str, Any]:
    if str(row.get("role")) != "ep0" or str(row.get("phase")) != "level2_ep0":
        raise RuntimeError("validate_ep0_row received a non-Level2-EP0 row")
    episode_id = str(row.get("episode_id") or "")
    session_id = str(row.get("session_id") or "")
    scene_id = str(row.get("scene_id") or "")
    if not episode_id or not session_id or not scene_id:
        raise RuntimeError(f"EP0 row lacks identity fields: {row.get('_source_jsonl')}")
    poses = row.get("poses")
    if not isinstance(poses, list) or not poses:
        raise RuntimeError(f"EP0 {episode_id} has no saved poses")
    steps = int(row.get("steps", -1))
    if len(poses) != steps + 1:
        raise RuntimeError(
            f"EP0 {episode_id} has {len(poses)} poses for {steps} actions"
        )
    normalized_poses = []
    for index, pose in enumerate(poses):
        if not isinstance(pose, dict):
            raise RuntimeError(f"EP0 {episode_id} pose {index} is not an object")
        position = finite_vector(pose.get("position"), 3, f"EP0 {episode_id} pose {index} position")
        rotation = finite_vector(
            pose.get("rotation_xyzw"), 4, f"EP0 {episode_id} pose {index} rotation"
        )
        norm = float(np.linalg.norm(rotation))
        if norm <= 1.0e-12 or abs(norm - 1.0) > 1.0e-3:
            raise RuntimeError(
                f"EP0 {episode_id} pose {index} has non-unit quaternion norm {norm}"
            )
        normalized_poses.append(
            {
                "frame_index": index,
                "simulator_step": int(pose.get("simulator_step", index)),
                "position": position,
                "rotation_xyzw": [value / norm for value in rotation],
            }
        )
    result = dict(row)
    result["episode_id"] = episode_id
    result["session_id"] = session_id
    result["scene_id"] = scene_id
    result["poses"] = normalized_poses
    return result


def load_ep0_rows(source: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    files = discover_rollout_files(source)
    by_episode: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, str] = {}
    for path in files:
        for row in iter_jsonl(path):
            if str(row.get("role")) != "ep0" or str(row.get("phase")) != "level2_ep0":
                continue
            row = validate_ep0_row(row)
            episode_id = row["episode_id"]
            fingerprint = sha256_json(
                {
                    "episode_id": episode_id,
                    "session_id": row["session_id"],
                    "scene_id": row["scene_id"],
                    "poses": row["poses"],
                }
            )
            if episode_id in by_episode and fingerprints[episode_id] != fingerprint:
                raise RuntimeError(f"Conflicting completed rollouts for EP0 {episode_id}")
            by_episode[episode_id] = row
            fingerprints[episode_id] = fingerprint
    rows = sorted(by_episode.values(), key=lambda item: (item["session_id"], item["episode_id"]))
    if not rows:
        raise RuntimeError("No completed Level-2 EP0 rollouts were found")
    return rows, files


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    episode_ids = set(map(str, args.episode_id))
    session_ids = set(map(str, args.session_id))
    known_episodes = {row["episode_id"] for row in rows}
    known_sessions = {row["session_id"] for row in rows}
    unknown_episodes = episode_ids - known_episodes
    unknown_sessions = session_ids - known_sessions
    if unknown_episodes:
        raise RuntimeError(f"Unknown EP0 episode ids: {sorted(unknown_episodes)}")
    if unknown_sessions:
        raise RuntimeError(f"Unknown Level-2 session ids: {sorted(unknown_sessions)}")
    selected = [
        row
        for row in rows
        if (not episode_ids or row["episode_id"] in episode_ids)
        and (not session_ids or row["session_id"] in session_ids)
    ]
    selected = [
        row
        for index, row in enumerate(selected)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise RuntimeError("EP0 selection is empty")
    return selected


def ep0_success(rows: list[dict[str, Any]]) -> tuple[int, int, float]:
    values = []
    for row in rows:
        value = (row.get("metrics") or {}).get("success")
        if value is not None:
            values.append(float(value))
    successes = int(sum(values))
    return successes, len(values), successes / len(values) if values else math.nan


def require_paths(args: argparse.Namespace) -> None:
    required = (
        args.config,
        args.dataset,
        args.scenes_dir,
        args.memory_module_root / "__init__.py",
        args.memory_module_root / "habitat_runtime/online_adapter.py",
        args.memory_module_root / "semantic_scene_graph/legacy_builder_source/config_sam3.yaml",
        args.query_vocabulary,
        args.sam3_ckpt,
        args.sam3_source_root,
        args.dinov3_model,
        args.dinov3_repo,
        args.text_encoder,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing offline M0 dependencies:\n  " + "\n  ".join(missing))


def install_memory_environment(args: argparse.Namespace, work_root: Path) -> None:
    values = {
        "INHERIT_SG_ENABLE": "1",
        "INHERIT_SG_OUTPUT_DIR": str(work_root),
        "INHERIT_SG_SESSION_NAME": "working_memory",
        "INHERIT_SG_CONFIG": str(
            args.memory_module_root
            / "semantic_scene_graph/legacy_builder_source/config_sam3.yaml"
        ),
        "INHERIT_SG_PRELOAD_DIR": "",
        "INHERIT_SG_PRELOAD_QUERY_ONLY": "0",
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
        "INHERIT_SG_QUERY_MIN_FINALIZED_EPISODES": "1",
        "INHERIT_SG_QUERY_VOCABULARY": str(args.query_vocabulary),
        "INHERIT_SG_TEXT_ENCODER": str(args.text_encoder),
        "INHERIT_SG_DINOV3_MODEL": str(args.dinov3_model),
        "INHERIT_SG_DINOV3_REPO": str(args.dinov3_repo),
        "INHERIT_SG_SAM3_CKPT": str(args.sam3_ckpt),
        "INHERIT_SG_SAM3_SOURCE_ROOT": str(args.sam3_source_root),
        "SAM3_SOURCE_ROOT": str(args.sam3_source_root),
        "INHERIT_SG_SAM3_CATEGORY_BATCH_SIZE": str(args.sam3_category_batch_size),
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


def memory_config(sensor_config: Any, logical_gpu: int) -> SimpleNamespace:
    rgb_source = sensor_config.rgb_sensor
    depth_source = sensor_config.depth_sensor
    rgb = SimpleNamespace(
        HFOV=float(rgb_source.hfov),
        POSITION=tuple(map(float, rgb_source.position)),
    )
    # HabitatSim returns metric depth here.  Marking it normalized would apply
    # the 0..10 m conversion a second time and corrupt the map.
    depth = SimpleNamespace(
        HFOV=float(depth_source.hfov),
        POSITION=tuple(map(float, depth_source.position)),
        NORMALIZE_DEPTH=False,
        MIN_DEPTH=float(depth_source.min_depth),
        MAX_DEPTH=float(depth_source.max_depth),
    )
    return SimpleNamespace(
        TORCH_GPU_ID=int(logical_gpu),
        TASK_CONFIG=SimpleNamespace(
            SIMULATOR=SimpleNamespace(RGB_SENSOR=rgb, DEPTH_SENSOR=depth)
        ),
    )


def device_index(device: str) -> int:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return 0


def quaternion_distance(first: Any, second: Any) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(4)
    b = np.asarray(second, dtype=np.float64).reshape(4)
    return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))


def state_rotation_xyzw(rotation: Any) -> list[float]:
    if all(hasattr(rotation, name) for name in ("x", "y", "z", "w")):
        return [float(getattr(rotation, name)) for name in ("x", "y", "z", "w")]
    if hasattr(rotation, "imag") and hasattr(rotation, "real"):
        vector = np.asarray(rotation.imag, dtype=np.float64).reshape(3)
        return [float(vector[0]), float(vector[1]), float(vector[2]), float(rotation.real)]
    return finite_vector(rotation, 4, "Habitat state rotation")


def load_sensor_config(path: Path) -> SimpleNamespace:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    try:
        sensors = payload["habitat"]["simulator"]["agents"]["main_agent"][
            "sim_sensors"
        ]
        rgb = sensors["rgb_sensor"]
        depth = sensors["depth_sensor"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Cannot read RGB-D sensor settings from {path}") from exc
    return SimpleNamespace(
        rgb_sensor=SimpleNamespace(
            width=int(rgb["width"]),
            height=int(rgb["height"]),
            hfov=float(rgb["hfov"]),
            position=tuple(map(float, rgb["position"])),
        ),
        depth_sensor=SimpleNamespace(
            width=int(depth["width"]),
            height=int(depth["height"]),
            hfov=float(depth["hfov"]),
            position=tuple(map(float, depth["position"])),
            min_depth=float(depth["min_depth"]),
            max_depth=float(depth["max_depth"]),
        ),
    )


def resolve_scene_path(scene_id: str, scenes_dir: Path) -> Path:
    raw = Path(scene_id)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    parts = raw.parts
    if "scene_datasets" in parts:
        index = parts.index("scene_datasets")
        candidates.append(scenes_dir.joinpath(*parts[index + 1 :]))
    candidates.extend((scenes_dir / raw, scenes_dir / raw.name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot resolve scene {scene_id!r} below {scenes_dir}; tried "
        + ", ".join(map(str, candidates))
    )


class HabitatPoseRenderer:
    """Minimal Habitat-Sim renderer, independent of Habitat-Lab config APIs."""

    def __init__(self, sensor_config: Any, scenes_dir: Path, gpu_id: int) -> None:
        import habitat_sim
        from habitat_sim.utils.common import quat_from_coeffs

        self.habitat_sim = habitat_sim
        self.quat_from_coeffs = quat_from_coeffs
        self.sensor_config = sensor_config
        self.scenes_dir = scenes_dir
        self.gpu_id = int(gpu_id)
        self.scene_path = None
        self.sim = None
        self.agent = None

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()
        self.scene_path = None
        self.sim = None
        self.agent = None

    def open_scene(self, scene_id: str) -> Path:
        path = resolve_scene_path(scene_id, self.scenes_dir)
        if self.scene_path == path and self.sim is not None:
            return path
        self.close()
        habitat_sim = self.habitat_sim
        simulator = habitat_sim.SimulatorConfiguration()
        simulator.scene_id = str(path)
        simulator.gpu_device_id = self.gpu_id
        simulator.enable_physics = False

        sensor_class = getattr(habitat_sim, "CameraSensorSpec", None)
        if sensor_class is None:
            sensor_class = habitat_sim.SensorSpec

        def sensor_spec(uuid: str, source: Any, sensor_type: Any):
            spec = sensor_class()
            spec.uuid = uuid
            spec.sensor_type = sensor_type
            if hasattr(habitat_sim, "SensorSubType"):
                spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            spec.resolution = [int(source.height), int(source.width)]
            spec.position = list(map(float, source.position))
            spec.hfov = float(source.hfov)
            return spec

        agent = habitat_sim.AgentConfiguration()
        agent.height = 1.2
        agent.sensor_specifications = [
            sensor_spec("rgb", self.sensor_config.rgb_sensor, habitat_sim.SensorType.COLOR),
            sensor_spec("depth", self.sensor_config.depth_sensor, habitat_sim.SensorType.DEPTH),
        ]
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(simulator, [agent]))
        self.agent = self.sim.initialize_agent(0)
        self.scene_path = path
        return path

    def render(self, pose: dict[str, Any], episode_id: str) -> tuple[dict[str, Any], Any]:
        habitat_sim = self.habitat_sim
        state = habitat_sim.AgentState()
        state.position = np.asarray(pose["position"], dtype=np.float32)
        state.rotation = self.quat_from_coeffs(pose["rotation_xyzw"])
        self.agent.set_state(state, infer_sensor_states=True)
        observations = self.sim.get_sensor_observations()
        if observations is None:
            raise RuntimeError(
                f"Habitat failed to render EP0 {episode_id} pose {pose['frame_index']}"
            )
        return observations, self.agent.get_state()


def render_observation(
    renderer: HabitatPoseRenderer,
    pose: dict[str, Any],
    episode_id: str,
) -> dict[str, Any]:
    position = pose["position"]
    rotation = pose["rotation_xyzw"]
    observations, state = renderer.render(pose, episode_id)
    position_error = float(
        np.linalg.norm(np.asarray(state.position, dtype=np.float64) - np.asarray(position))
    )
    rotation_error = quaternion_distance(state_rotation_xyzw(state.rotation), rotation)
    if position_error > 2.0e-5 or rotation_error > 2.0e-5:
        raise RuntimeError(
            f"EP0 {episode_id} render pose drift at frame {pose['frame_index']}: "
            f"position={position_error:.8f}, quaternion={rotation_error:.8f}"
        )
    rgb = np.asarray(observations["rgb"], dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
        raise RuntimeError(f"Unexpected RGB shape at EP0 {episode_id}: {rgb.shape}")
    depth = np.asarray(observations["depth"], dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise RuntimeError(f"Unexpected depth shape at EP0 {episode_id}: {depth.shape}")
    return {"rgb": rgb[..., :3].copy(), "depth": depth.copy()}


def build_one(
    *,
    args: argparse.Namespace,
    renderer: HabitatPoseRenderer,
    sensor_config: Any,
    row: dict[str, Any],
    shared_models: Any,
    adapter_class: Any,
    heading_from_quaternion: Any,
) -> tuple[Any, dict[str, Any]]:
    session_id = row["session_id"]
    episode_id = row["episode_id"]
    destination = args.output_root / "sessions" / session_id / "stages/M0"
    if destination.exists():
        complete = all((destination / name).is_file() for name in CORE_STAGE_FILES)
        if args.skip_existing and complete and (destination / "stage_manifest.json").is_file():
            return shared_models, {
                "status": "skipped_existing",
                "session_id": session_id,
                "episode_id": episode_id,
                "m0": str(destination.resolve()),
            }
        raise FileExistsError(f"Refusing to replace existing M0: {destination}")

    work_root = (
        args.output_root
        / ".working"
        / f"{session_id}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    if work_root.exists():
        raise FileExistsError(f"Stale working directory exists: {work_root}")
    work_root.mkdir(parents=True)
    install_memory_environment(args, work_root)
    adapter = adapter_class(
        memory_config(sensor_config, device_index(args.device)),
        width=int(sensor_config.rgb_sensor.width),
        height=int(sensor_config.rgb_sensor.height),
        save_dir=str(work_root),
        device=args.device,
    )
    adapter.shared_models = shared_models

    try:
        renderer.open_scene(row["scene_id"])
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
                    f"[JANUS-EP0-M0] {session_id}/{episode_id}: "
                    f"{ordinal + 1}/{len(row['poses'])} poses",
                    flush=True,
                )
        adapter.finish_episode()
        if adapter.failed:
            raise RuntimeError(f"memory_module marked EP0 {episode_id} failed")
        shared_models = adapter.shared_models
        run_dir = Path(adapter.pipeline.run_dir).resolve()
        missing = [name for name in CORE_STAGE_FILES if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"EP0 {episode_id} M0 lacks artifacts: {missing}")

        artifact_names = list(CORE_STAGE_FILES) + [
            name for name in OPTIONAL_STAGE_FILES if (run_dir / name).is_file()
        ]
        manifest = {
            "schema_version": "janusvln_level2_memory_module_stage_v1",
            "session_id": session_id,
            "episode_id": episode_id,
            "role": "ep0",
            "memory_read": None,
            "memory_write": "M0",
            "source_kind": "janusvln_saved_pose_rgbd_rerender",
            "source_rollout_jsonl": row["_source_jsonl"],
            "source_pose_sha256": sha256_json(row["poses"]),
            "source_scene_id": row["scene_id"],
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
            "janus_ep0_metrics": row.get("metrics") or {},
            "semantic_models_loaded_once_per_process": True,
            "async_queue_drained_before_publish": True,
            "artifacts_sha256": {name: sha256_file(run_dir / name) for name in artifact_names},
        }
        atomic_json(run_dir / "stage_manifest.json", manifest)

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".M0.tmp-{os.getpid()}"
        if temporary.exists():
            raise FileExistsError(f"Stale M0 publication directory exists: {temporary}")
        os.replace(run_dir, temporary)
        os.replace(temporary, destination)
        shutil.rmtree(work_root, ignore_errors=True)
        result = {
            "status": "complete",
            "session_id": session_id,
            "episode_id": episode_id,
            "m0": str(destination.resolve()),
            "frame_count": len(row["poses"]),
            "source_pose_sha256": manifest["source_pose_sha256"],
        }
        return shared_models, result
    except Exception:
        try:
            adapter.abort_episode()
        except Exception:
            pass
        raise


def main() -> None:
    args = parse_args()
    args.janus_results = args.janus_results.resolve()
    args.janus_root = args.janus_root.resolve()
    args.config = args.config.resolve()
    args.dataset = args.dataset.resolve()
    args.scenes_dir = args.scenes_dir.resolve()
    args.output_root = args.output_root.resolve()
    args.memory_module_root = args.memory_module_root.resolve()
    args.query_vocabulary = args.query_vocabulary.resolve()
    args.sam3_ckpt = args.sam3_ckpt.resolve()
    args.sam3_source_root = args.sam3_source_root.resolve()
    args.dinov3_model = args.dinov3_model.resolve()
    args.dinov3_repo = args.dinov3_repo.resolve()
    args.text_encoder = args.text_encoder.resolve()

    rows, source_files = load_ep0_rows(args.janus_results)
    selected = select_rows(rows, args)
    successes, count, rate = ep0_success(rows)
    print(
        json.dumps(
            {
                "source_jsonl_count": len(source_files),
                "completed_ep0": len(rows),
                "ep0_successes": successes,
                "ep0_success_rate": rate,
                "selected_ep0": len(selected),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    require_paths(args)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(args.janus_root))
    install_memory_environment(args, args.output_root / ".working/bootstrap")

    from membridge.habitat_runtime.bootstrap import install_environment_aliases
    from membridge.habitat_runtime.online_adapter import OnlineINHeritSGBuilder
    from membridge.trajectory_graph.habitat_adapter import (
        heading_from_quaternion_xyzw,
    )

    install_environment_aliases()
    sensor_config = load_sensor_config(args.config)
    renderer = HabitatPoseRenderer(sensor_config, args.scenes_dir, args.sim_gpu_id)

    args.output_root.mkdir(parents=True, exist_ok=True)
    worker_id = args.worker_id.strip()
    if not worker_id and args.num_shards > 1:
        worker_id = f"shard{args.shard_index:02d}-of-{args.num_shards:02d}"
    safe_worker_id = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in worker_id
    )
    result_suffix = f".{safe_worker_id}" if safe_worker_id else ""
    results_log = args.output_root / f"build_results{result_suffix}.jsonl"
    shared_models = None
    failures = []
    try:
        for index, row in enumerate(selected, 1):
            print(
                f"[JANUS-EP0-M0] session {index}/{len(selected)}: "
                f"{row['session_id']} episode={row['episode_id']}",
                flush=True,
            )
            try:
                shared_models, result = build_one(
                    args=args,
                    renderer=renderer,
                    sensor_config=sensor_config,
                    row=row,
                    shared_models=shared_models,
                    adapter_class=OnlineINHeritSGBuilder,
                    heading_from_quaternion=heading_from_quaternion_xyzw,
                )
            except Exception as exc:
                result = {
                    "status": "failed",
                    "session_id": row["session_id"],
                    "episode_id": row["episode_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(result)
                append_jsonl(results_log, result)
                if not args.continue_on_error:
                    raise
                print(f"[JANUS-EP0-M0] failed: {result}", file=sys.stderr, flush=True)
                continue
            append_jsonl(results_log, result)
            print(f"[JANUS-EP0-M0] {result['status']}: {result['m0']}", flush=True)
    finally:
        renderer.close()

    summary = {
        "schema_version": "janusvln_level2_ep0_offline_memory_summary_v1",
        "source_results": str(args.janus_results),
        "output_root": str(args.output_root),
        "requested_count": len(selected),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "worker_id": worker_id or None,
        "failure_count": len(failures),
        "failures": failures,
        "janus_completed_ep0_count": count,
        "janus_ep0_success_count": successes,
        "janus_ep0_success_rate": rate,
    }
    atomic_json(args.output_root / f"build_summary{result_suffix}.json", summary)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
