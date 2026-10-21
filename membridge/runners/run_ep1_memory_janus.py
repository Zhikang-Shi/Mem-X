#!/usr/bin/env python3
"""Run Level-2 EP1 with frozen M0 SG recall and JanusVLN continuation.

The instruction route is read from the already completed Uni-LaViRA route
JSON.  It is never parsed again.  A valid EP1 route has one leading recall
segment followed by zero or more ordinary-VLN segments.  All trailing type-2
segments are concatenated, in order, into the single instruction given to
JanusVLN after the SG anchor is reached.

Only ``memory_module`` owns memory retrieval, metric-map navigation and the
complete exploration overlay.  During SG navigation JanusVLN does not choose
actions, but every RGB observation is retained in its in-memory visual
history.  Thus its first continuation call sees the route to the anchor.

EP1 RGB-D is streamed to ``OnlineINHeritSGBuilder`` in model-free
``PRELOAD_WORLD_MAP_ONLY`` mode.  This updates the navigation map and
trajectory in real time without importing/loading SAM3 or DINOv3 and without
persisting a raw RGB-D capture.  M1 contains the frozen M0 semantic SG plus
the updated world map/trajectory.  Rollout output persists positions and
quaternions only (in Habitat's XYZW order), never RGB-D frames.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import gzip
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_JANUS_ROOT = Path("/home/wangyj/projects/janusvln_hm3dsem_benchmark/src/JanusVLN")
DEFAULT_EXP_ROOT = DEFAULT_JANUS_ROOT.parents[1]
DEFAULT_DATASET = DEFAULT_EXP_ROOT / "data/level2_v8/habitat/level2/level2.json.gz"
DEFAULT_CONFIG = DEFAULT_JANUS_ROOT / "config/vln_level2_protocol.yaml"
DEFAULT_MODEL = DEFAULT_EXP_ROOT / "models/JanusVLN_Extra"
DEFAULT_M0_ROOT = HERE / "outputs/ep0_memory_from_poses_full"
DEFAULT_ROUTE_SUMMARY = (
    REPO_ROOT
    / "baseline_experiment/Uni-LaViRA/experiment_level2/outputs/final_level2_sg_latest.json"
)
DEFAULT_EXTRA_ROUTE = Path(
    "/share/shared/shizk/level2_outputs/formal_runs/"
    "missing2-891003-gpu7-20260831-01/workers/gpu7_p0/"
    "sessions/level2_00891_003/routing/ep1_8910031.json"
)
DEFAULT_OUTPUT = Path(
    "/share/shared/shizk/level2_outputs/janus_memory_module/ep1_memory_janus"
)

CORE_M0_FILES = (
    "memory_kb.json",
    "builder_state.pkl",
    "world_metric_map.npz",
    "trajectory_graph.npz",
)

LOG = logging.getLogger("janus-memory-ep1")
NAVIGATION_CONTROLLER_VERSION = "memory_module_habitat_path_follower_v3_object_approach"


class Actions:
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


ACTION_NAMES = {
    Actions.STOP: "STOP",
    Actions.MOVE_FORWARD: "MOVE_FORWARD",
    Actions.TURN_LEFT: "TURN_LEFT",
    Actions.TURN_RIGHT: "TURN_RIGHT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--janus-root", type=Path, default=DEFAULT_JANUS_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--m0-root", type=Path, default=DEFAULT_M0_ROOT)
    parser.add_argument("--route-summary", type=Path, default=DEFAULT_ROUTE_SUMMARY)
    parser.add_argument("--extra-route", type=Path, action="append", default=[])
    parser.add_argument(
        "--recall-plan-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Reuse saved VLM Top3 selection from ROOT/sessions/<session>/stages/M1/"
            "recall_plan.json. May be repeated; the last root containing a session wins."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-gpu-id", type=int, default=0)
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
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
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="HOST_PREFIX=RUNTIME_PREFIX",
        help="Translate absolute paths recorded by an older container",
    )
    parser.add_argument(
        "--query-vocabulary",
        type=Path,
        default=(
            REPO_ROOT
            / "memory_module/semantic_scene_graph/config/level1_runtime_query_vocabulary_75.json"
        ),
    )
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.max_steps < 1 or args.num_history < 0:
        parser.error("--max-steps must be positive and --num-history non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
    return rows


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def parse_path_maps(values: Sequence[str]) -> list[tuple[str, str]]:
    defaults = [
        ("/home/shizk/LifeLongHomeNav", str(REPO_ROOT)),
        ("/home/wangyj/projects/janusvln_hm3dsem_benchmark", str(DEFAULT_EXP_ROOT)),
    ]
    result = list(defaults)
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --path-map {value!r}; expected OLD=NEW")
        old, new = value.split("=", 1)
        if not old.startswith("/") or not new.startswith("/"):
            raise ValueError("--path-map prefixes must be absolute")
        result.insert(0, (old.rstrip("/"), new.rstrip("/")))
    return result


def translated_path(value: str | Path, mappings: Sequence[tuple[str, str]]) -> Path:
    raw = str(value)
    if Path(raw).exists():
        return Path(raw)
    for old, new in mappings:
        if raw == old or raw.startswith(old + "/"):
            candidate = Path(new + raw[len(old) :])
            if candidate.exists():
                return candidate
    return Path(raw)


def source_run_root(stats_path: Path) -> Path:
    parts = stats_path.parts
    try:
        index = parts.index("run_metadata")
    except ValueError as exc:
        raise RuntimeError(f"source stats path has no run_metadata ancestor: {stats_path}") from exc
    return Path(*parts[:index])


def validate_route(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    if payload.get("schema_version") != "unilavira_level2_instruction_route_v1":
        raise RuntimeError(f"unsupported route schema: {path}")
    if str(payload.get("role")) != "ep1" or int(payload.get("instruction_parse_count", -1)) != 1:
        raise RuntimeError(f"route is not a single-parse EP1 record: {path}")
    parser = payload.get("parser")
    segments = parser.get("segments") if isinstance(parser, Mapping) else None
    if not isinstance(segments, list) or not segments:
        raise RuntimeError(f"route has no parser segments: {path}")
    types = [int(item.get("type", -1)) for item in segments]
    if types[0] != 1 or types.count(1) != 1 or any(value != 2 for value in types[1:]):
        raise RuntimeError(f"EP1 route must be [1, 2...], got {types}: {path}")
    recall = segments[0]
    if not normalize_text(recall.get("text")) or not normalize_text(recall.get("label")):
        raise RuntimeError(f"route recall lacks text/label: {path}")
    suffix_parts = [normalize_text(item.get("text")) for item in segments[1:]]
    if any(not value for value in suffix_parts):
        raise RuntimeError(f"route contains an empty type-2 segment: {path}")
    return {
        "episode_id": str(payload.get("episode_id")),
        "session_id": str(payload.get("session_id")),
        "instruction": normalize_text(parser.get("instruction")),
        "types": types,
        "recall": copy.deepcopy(recall),
        "suffix_parts": suffix_parts,
        "merged_suffix": " ".join(suffix_parts),
        "source": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "parser_model": parser.get("model"),
        "parser_request_id": parser.get("request_id"),
    }


def load_route_index(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    mappings = parse_path_maps(args.path_map)
    summary_path = translated_path(args.route_summary, mappings)
    summary = read_json(summary_path)
    sources = (summary.get("provenance") or {}).get("final_source_by_episode") or {}
    routes: dict[str, dict[str, Any]] = {}
    for episode_id, recorded_stats in sources.items():
        episode_id = str(episode_id)
        if not episode_id.endswith("1"):
            continue
        stats_path = translated_path(recorded_stats, mappings)
        run_root = source_run_root(stats_path)
        matches = list(run_root.glob(f"sessions/*/routing/ep1_{episode_id}.json"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one adopted route for EP1 {episode_id}, found {len(matches)} below {run_root}"
            )
        routes[episode_id] = validate_route(read_json(matches[0]), matches[0])

    extra_paths = list(args.extra_route)
    if not extra_paths:
        extra_paths.append(DEFAULT_EXTRA_ROUTE)
    for raw_path in extra_paths:
        path = translated_path(raw_path, mappings)
        route = validate_route(read_json(path), path)
        existing = routes.get(route["episode_id"])
        if existing and existing["source_sha256"] != route["source_sha256"]:
            if (
                existing["instruction"], existing["types"], existing["recall"],
                existing["suffix_parts"],
            ) != (
                route["instruction"], route["types"], route["recall"],
                route["suffix_parts"],
            ):
                raise RuntimeError(f"conflicting route for EP1 {route['episode_id']}")
        routes[route["episode_id"]] = route
    return routes


def dataset_sessions(dataset_path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(dataset_path)
    sessions: dict[str, dict[str, Any]] = {}
    for episode in payload.get("episodes") or []:
        info = episode.get("info") or {}
        if str(info.get("role") or info.get("lifelong_role") or info.get("level2_role")) != "ep1":
            continue
        episode_id = str(episode["episode_id"])
        session_id = episode_id.rsplit("_", 1)[0] if episode_id.endswith("_ep1") else episode_id
        if session_id in sessions:
            raise RuntimeError(f"duplicate EP1 session {session_id}")
        sessions[session_id] = {
            "episode_id": str(episode["episode_id"]),
            "session_id": session_id,
            "instruction": normalize_text((episode.get("instruction") or {}).get("instruction_text")),
        }
    return dict(sorted(sessions.items()))


def validate_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    routes = load_route_index(args)
    sessions = dataset_sessions(args.dataset)
    if len(sessions) != 238:
        raise RuntimeError(f"expected 238 Level-2 EP1 sessions, found {len(sessions)}")
    missing_routes = sorted(
        row["episode_id"] for row in sessions.values() if row["episode_id"] not in routes
    )
    if missing_routes:
        raise RuntimeError(f"missing Uni-LaViRA routes: {missing_routes}")
    rows = []
    for session_id, dataset_row in sessions.items():
        route = routes[dataset_row["episode_id"]]
        if route["session_id"] != session_id:
            raise RuntimeError(f"route/session mismatch for {dataset_row['episode_id']}")
        if route["instruction"] != dataset_row["instruction"]:
            raise RuntimeError(f"route instruction differs from dataset for {dataset_row['episode_id']}")
        m0 = args.m0_root / "sessions" / session_id / "stages/M0"
        missing = [name for name in CORE_M0_FILES if not (m0 / name).is_file()]
        if missing:
            raise RuntimeError(f"M0 {session_id} lacks {missing}: {m0}")
        rows.append({**dataset_row, "route": route, "m0": m0})

    requested_sessions = set(map(str, args.session_id))
    requested_episodes = set(map(str, args.episode_id))
    known_sessions = {row["session_id"] for row in rows}
    known_episodes = {row["episode_id"] for row in rows}
    if requested_sessions - known_sessions:
        raise RuntimeError(f"unknown sessions: {sorted(requested_sessions - known_sessions)}")
    if requested_episodes - known_episodes:
        raise RuntimeError(f"unknown EP1 episodes: {sorted(requested_episodes - known_episodes)}")
    rows = [
        row for row in rows
        if (not requested_sessions or row["session_id"] in requested_sessions)
        and (not requested_episodes or row["episode_id"] in requested_episodes)
    ]
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("EP1 selection is empty")
    return rows


def install_memory_environment(args: argparse.Namespace, work_root: Path, m0: Path) -> None:
    values = {
        "INHERIT_SG_ENABLE": "1",
        "INHERIT_SG_OUTPUT_DIR": str(work_root),
        "INHERIT_SG_SESSION_NAME": "working_memory",
        "INHERIT_SG_CONFIG": str(
            REPO_ROOT / "semantic_scene_graph/legacy_builder_source/config_sam3.yaml"
        ),
        "INHERIT_SG_PRELOAD_DIR": str(m0),
        "INHERIT_SG_PRELOAD_QUERY_ONLY": "1",
        "INHERIT_SG_PRELOAD_WORLD_MAP_ONLY": "1",
        "INHERIT_SG_CAPTURE_ONLY": "0",
        "INHERIT_SG_SAVE_OFFLINE_RGBD": "0",
        "INHERIT_SG_SAVE_CONFIRMED_KEYFRAMES": "0",
        "INHERIT_SG_FAIL_CLOSED": "1",
        "INHERIT_SG_ASYNC": "0",
        "INHERIT_SG_QUERY_MIN_FINALIZED_EPISODES": "1",
        "INHERIT_SG_QUERY_VOCABULARY": str(args.query_vocabulary),
        "LIFELONG_WORLD_MEMORY_ENABLE": "1",
        "LIFELONG_WORLD_MEMORY_REQUIRED": "1",
        "LIFELONG_WORLD_MAP_RESOLUTION_M": "0.05",
        "LIFELONG_WORLD_MAP_CHUNK_SIZE": "128",
        "LIFELONG_WORLD_MAP_PIXEL_STRIDE": "8",
        "LIFELONG_WORLD_MAP_MAX_RAYS": "2048",
        "LIFELONG_WORLD_TRAJECTORY_SPACING_M": "0.25",
        "LIFELONG_WORLD_TRAJECTORY_MAX_EDGE_M": "1.0",
        "LIFELONG_SG_EXPLORATION_ENABLE": "1",
        "MEMORY_MODULE_EXPLORATION_IMPLEMENTATION": "memory_module_full_goal_directed_exploration_v1",
        "LIFELONG_SG_EXPLORE_MAX_ACTIONS": "200",
        "LIFELONG_SG_EXPLORE_MAX_DISTANCE_M": "50.0",
        "LIFELONG_SG_EXPLORE_CONNECTIVITY_CHECK_STRIDE": "4",
        "LIFELONG_SG_EXPLORE_ACTION_LATTICE_RECOVERY": "1",
        "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE": "1",
        "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE_CONNECTIVITY_STRIDE": "4",
        "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE_MAX_ACTIONS": "48",
        "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE_MAX_FORWARDS": "36",
        "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE_MIN_PROGRESS_M": "-0.26",
        "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE_REQUIRE_NAVMESH": "1",
        "LIFELONG_SG_EXPLORE_NO_ACTION_RECOVERY": "1",
        "LIFELONG_SG_EXPLORE_OLD_MAP_FALLBACK": "1",
        "LIFELONG_SG_EXPLORE_OLD_MAP_MAX_ACTIONS": "48",
        "LIFELONG_SG_EXPLORE_OLD_MAP_MAX_FORWARDS": "36",
        "LIFELONG_SG_EXPLORE_OLD_MAP_SWITCH_ACTIONS": "8",
        "LIFELONG_SG_EXPLORE_OLD_MAP_MIN_PROGRESS_M": "0.25",
        "LIFELONG_SG_WORLD_FULL_WINDOW_RETRY": "1",
        "LIFELONG_SG_POST_HIT_MAX_OBJECT_DISTANCE_M": "3.0",
        "LIFELONG_SG_POST_HIT_MAX_NAVMESH_SNAP_M": "0.75",
        "LIFELONG_SG_POST_HIT_MAX_SAFE_SNAP_M": "1.0",
        "LIFELONG_SG_POST_HIT_ANNULUS_STEP_M": "0.10",
        "LIFELONG_SG_POST_HIT_ANGULAR_SAMPLES": "64",
        "LIFELONG_SG_POST_HIT_MAX_PLANNER_CANDIDATES": "24",
        "MEMORY_MODULE_NAVIGATION_RUNTIME_MODE": "authoritative",
        "MEMORY_MODULE_ADAPTER_FAIL_CLOSED": "1",
        "LIFELONG_SG_VLM_TOP3_FALLBACK_POLICY": "top1",
        "LIFELONG_SG_VLM_TOP3_SELECTION_MODE": "force_single",
        "LIFELONG_SG_VLM_TOP3_MAX_TOKENS": "2048",
    }
    os.environ.update(values)


def sensor_memory_config(habitat_config: Any, logical_gpu: int) -> SimpleNamespace:
    sensors = habitat_config.habitat.simulator.agents.main_agent.sim_sensors
    rgb_source, depth_source = sensors.rgb_sensor, sensors.depth_sensor
    return SimpleNamespace(
        TORCH_GPU_ID=int(logical_gpu),
        TASK_CONFIG=SimpleNamespace(
            SIMULATOR=SimpleNamespace(
                RGB_SENSOR=SimpleNamespace(
                    HFOV=float(rgb_source.hfov), POSITION=tuple(map(float, rgb_source.position))
                ),
                DEPTH_SENSOR=SimpleNamespace(
                    HFOV=float(depth_source.hfov),
                    POSITION=tuple(map(float, depth_source.position)),
                    NORMALIZE_DEPTH=bool(depth_source.normalize_depth),
                    MIN_DEPTH=float(depth_source.min_depth),
                    MAX_DEPTH=float(depth_source.max_depth),
                ),
            )
        ),
    )


def pose_from_env(env: Any, frame_index: int, simulator_step: int) -> dict[str, Any]:
    import quaternion

    state = env.sim.get_agent_state()
    w, x, y, z = quaternion.as_float_array(state.rotation)
    return {
        "frame_index": int(frame_index),
        "simulator_step": int(simulator_step),
        "position": [float(value) for value in state.position],
        "rotation_xyzw": [float(x), float(y), float(z), float(w)],
    }


def memory_observation(observations: Mapping[str, Any], pose: Mapping[str, Any]) -> dict[str, Any]:
    from membridge.trajectory_graph.habitat_adapter import heading_from_quaternion_xyzw

    rgb = np.asarray(observations["rgb"], dtype=np.uint8)
    depth = np.asarray(observations["depth"], dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    return {
        "rgb": rgb[..., :3].copy(),
        "depth": depth.copy(),
        "position": np.asarray(pose["position"], dtype=np.float32),
        "heading": np.asarray(
            [heading_from_quaternion_xyzw(pose["rotation_xyzw"])], dtype=np.float32
        ),
    }


def sample_history(history: Sequence[Image.Image], maximum_history: int) -> list[Image.Image]:
    last = len(history) - 1
    if last <= maximum_history:
        return list(history)
    indices = np.linspace(0, last, maximum_history + 1, dtype=int)
    return [history[int(index)] for index in indices]


class MemoryNavigator:
    """Small Janus-facing host for memory_module's complete exploration overlay."""

    def __init__(self, *, adapter: Any, env: Any, habitat_config: Any, decision_log: Path) -> None:
        self.inherit_sg = adapter
        self.env = env
        self.decision_log = decision_log
        self.current_episode_id = ""
        self.current_step = 0
        self._sg_world_position_xz = None
        self._sg_world_heading_rad = None
        self._sg_world_plan_window = None
        self._sg_world_safe_mask = None
        self._sg_world_trajectory_probe_mask = None
        self._sg_last_theta_failure = None
        self._sg_path_plan_active = False
        self._sg_blocked_by_task: dict[str, set[str]] = {}
        self._sg_world_collision_blocked_cells: set[tuple[int, int]] = set()
        self._sg_nav_progress = None
        self.sg_local_action_radius_m = 2.5
        self.sg_local_action_goal_tolerance_m = float(
            os.environ.get("LIFELONG_SG_LOCAL_ACTION_GOAL_TOLERANCE_M", "0.15")
        )
        self.sg_local_action_max_states = 12000
        self.sg_path_lookahead_m = 1.0
        self.sg_path_deviation_m = 0.35
        self.sg_nav_stall_steps = int(
            os.environ.get("LIFELONG_SG_STALL_STEPS", "10")
        )
        self.sg_nav_stall_eps_m = float(
            os.environ.get("LIFELONG_SG_STALL_EPS_M", "0.05")
        )
        self.sg_nav_goal_tolerance_m = float(
            os.environ.get(
                "LIFELONG_SG_NAV_GOAL_TOLERANCE_M",
                str(self.sg_local_action_goal_tolerance_m),
            )
        )
        self.sg_snap_radius_cells = 60
        # Keep Janus' SG handoff terminal selection identical to the formal
        # memory_module V8 runtime.  An SG object centre is a semantic
        # measurement, not an agent pose: sample its complete success
        # neighbourhood, project those samples to the live navmesh, and only
        # then ask the persisted M0 map for a route.
        self.sg_post_hit_max_object_distance_m = float(
            os.environ.get("LIFELONG_SG_POST_HIT_MAX_OBJECT_DISTANCE_M", "3.0")
        )
        self.sg_post_hit_max_navmesh_snap_m = float(
            os.environ.get("LIFELONG_SG_POST_HIT_MAX_NAVMESH_SNAP_M", "0.75")
        )
        self.sg_post_hit_max_safe_snap_m = float(
            os.environ.get("LIFELONG_SG_POST_HIT_MAX_SAFE_SNAP_M", "1.0")
        )
        self.sg_post_hit_annulus_step_m = float(
            os.environ.get("LIFELONG_SG_POST_HIT_ANNULUS_STEP_M", "0.10")
        )
        self.sg_post_hit_angular_samples = int(
            os.environ.get("LIFELONG_SG_POST_HIT_ANGULAR_SAMPLES", "64")
        )
        self.sg_post_hit_max_planner_candidates = int(
            os.environ.get("LIFELONG_SG_POST_HIT_MAX_PLANNER_CANDIDATES", "24")
        )
        simulator = habitat_config.habitat.simulator
        agent = simulator.agents.main_agent
        self.config = SimpleNamespace(
            TASK_CONFIG=SimpleNamespace(
                SIMULATOR=SimpleNamespace(
                    TURN_ANGLE=float(simulator.turn_angle),
                    FORWARD_STEP_SIZE=float(simulator.forward_step_size),
                    AGENT_0=SimpleNamespace(RADIUS=float(getattr(agent, "radius", 0.10))),
                )
            )
        )

    def reset(self) -> None:
        self._sg_world_plan_window = None
        self._sg_world_safe_mask = None
        self._sg_world_trajectory_probe_mask = None
        self._sg_last_theta_failure = None
        self._sg_path_plan_active = False
        self._sg_world_collision_blocked_cells.clear()

    def update_pose(self, position: Sequence[float], heading: float, step: int) -> None:
        values = np.asarray(position, dtype=np.float64).reshape(-1)
        self._sg_world_position_xz = values[[0, 2]]
        self._sg_world_heading_rad = float(heading)
        self.current_step = int(step)

    def _sg_world_metric_map(self):
        recorder = getattr(self.inherit_sg, "world_memory", None)
        return getattr(recorder, "metric_map", None)

    def _sg_trajectory_graph(self):
        recorder = getattr(self.inherit_sg, "world_memory", None)
        return getattr(recorder, "trajectory", None)

    def _sg_current_world_xz(self, kwargs=None):
        kwargs = kwargs or {}
        raw = kwargs.get("current_world_position")
        if raw is not None:
            values = np.asarray(raw, dtype=np.float64).reshape(-1)
            if values.size >= 3:
                self._sg_world_position_xz = values[[0, 2]]
        return None if self._sg_world_position_xz is None else self._sg_world_position_xz.copy()

    def _sg_live_navmesh_context(self):
        state = self.env.sim.get_agent_state()
        pathfinder = self.env.sim.pathfinder
        if pathfinder is None or not bool(pathfinder.is_loaded):
            return None
        return pathfinder, float(state.position[1])

    def _sg_navmesh_context(self):
        return self._sg_live_navmesh_context()

    @staticmethod
    def _sg_navmesh_points_connected(pathfinder, start_point, end_point):
        """Check connectivity without retaining or executing a navmesh route."""

        try:
            import habitat_sim

            query = habitat_sim.ShortestPath()
            query.requested_start = np.asarray(start_point, dtype=np.float32)
            query.requested_end = np.asarray(end_point, dtype=np.float32)
            return bool(pathfinder.find_path(query))
        except Exception:
            return None

    @staticmethod
    def _sg_predict_forward_endpoint(position, heading, step_cells, resolution_m, navmesh_context=None):
        position = np.asarray(position, dtype=np.float64).reshape(2)
        nominal = position + float(step_cells) * np.asarray(
            [-np.sin(float(heading)), -np.cos(float(heading))]
        )
        if navmesh_context is None:
            return nominal, False
        pathfinder, floor_y = navmesh_context
        start = np.asarray(
            [(position[0] + 0.5) * resolution_m, floor_y, (position[1] + 0.5) * resolution_m],
            dtype=np.float32,
        )
        end = np.asarray(
            [(nominal[0] + 0.5) * resolution_m, floor_y, (nominal[1] + 0.5) * resolution_m],
            dtype=np.float32,
        )
        predicted = np.asarray(pathfinder.try_step(start, end), dtype=np.float64).reshape(-1)
        if predicted.size < 3 or not np.isfinite(predicted[:3]).all():
            return nominal, False
        return predicted[[0, 2]] / resolution_m - 0.5, True

    def _sg_validate_warm_start_connector(self, start_world_xz, end_world_xz):
        context = self._sg_live_navmesh_context()
        if context is None:
            return False, {"status": "missing_navmesh"}
        pathfinder, floor_y = context
        start = np.asarray([start_world_xz[0], floor_y, start_world_xz[1]], dtype=np.float32)
        end = np.asarray([end_world_xz[0], floor_y, end_world_xz[1]], dtype=np.float32)
        reached = np.asarray(pathfinder.try_step(start, end), dtype=np.float64)
        residual = float(np.linalg.norm(reached[[0, 2]] - end[[0, 2]]))
        valid = bool(residual <= 0.05)
        return valid, {
            "status": "valid" if valid else "blocked",
            "try_step_residual_m": residual,
        }

    @staticmethod
    def _sg_window_point(window, global_point):
        point = np.asarray(global_point, dtype=np.float64).reshape(2)
        return point - np.asarray([window.origin_grid_x, window.origin_grid_z], dtype=np.float64)

    @staticmethod
    def _sg_float_segment_is_safe(mask, start, end):
        mask = np.asarray(mask, dtype=bool)
        start = np.asarray(start, dtype=np.float64).reshape(2)
        end = np.asarray(end, dtype=np.float64).reshape(2)
        samples = max(2, int(math.ceil(np.linalg.norm(end - start) * 4.0)) + 1)
        for point in np.linspace(start, end, samples):
            x, z = int(round(point[0])), int(round(point[1]))
            if not (0 <= z < mask.shape[0] and 0 <= x < mask.shape[1] and mask[z, x]):
                return False
        return True

    @staticmethod
    def _sg_path_cumulative(path):
        points = np.asarray(path, dtype=np.float64).reshape(-1, 2)
        values = [0.0]
        for left, right in zip(points[:-1], points[1:]):
            values.append(values[-1] + float(np.linalg.norm(right - left)))
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _sg_point_on_path(path, cumulative, progress):
        if not path:
            return None
        progress = min(max(float(progress), 0.0), float(cumulative[-1]))
        for index in range(len(path) - 1):
            if progress > float(cumulative[index + 1]):
                continue
            segment = float(cumulative[index + 1] - cumulative[index])
            if segment <= 1e-9:
                return np.asarray(path[index + 1], dtype=np.float64)
            fraction = (progress - float(cumulative[index])) / segment
            start = np.asarray(path[index], dtype=np.float64)
            end = np.asarray(path[index + 1], dtype=np.float64)
            return start + fraction * (end - start)
        return np.asarray(path[-1], dtype=np.float64)

    @staticmethod
    def _sg_project_position_to_path(path, position, minimum_progress, cumulative=None):
        if cumulative is None:
            cumulative = MemoryNavigator._sg_path_cumulative(path)
        position = np.asarray(position, dtype=np.float64).reshape(2)
        best = None
        for index, (start, end) in enumerate(zip(path[:-1], path[1:])):
            start = np.asarray(start, dtype=np.float64)
            end = np.asarray(end, dtype=np.float64)
            delta = end - start
            length_sq = float(np.dot(delta, delta))
            fraction = (
                0.0
                if length_sq <= 1e-9
                else float(np.clip(np.dot(position - start, delta) / length_sq, 0.0, 1.0))
            )
            projection = start + fraction * delta
            progress = float(cumulative[index]) + fraction * math.sqrt(length_sq)
            if progress + 1.0 < float(minimum_progress):
                continue
            distance = float(np.linalg.norm(position - projection))
            key = (distance, -progress)
            if best is None or key < best[0]:
                best = (key, progress, distance)
        if best is None:
            return None, float("inf")
        return float(best[1]), float(best[2])

    @staticmethod
    def _sg_theta_star(safe_mask, start, goal):
        from membridge.navigation import theta_star

        result = theta_star(safe_mask, start, goal)
        return list(result.path), int(result.expanded_states)

    def _sg_trajectory_execution_slack_cells(self, spec):
        configured = os.environ.get("LIFELONG_SG_TRAJECTORY_EXECUTION_SLACK_CELLS")
        if configured is not None:
            return max(0.0, float(configured))
        resolution = float(spec.resolution_m)
        forward = float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE)
        turn = math.radians(float(self.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE))
        return float(max(0, math.ceil((forward * abs(math.sin(turn / 2.0)) + resolution / 2.0) / resolution - 1e-9)))

    def _sg_world_navmesh_target_validator(self, window, current, terminal_snap_tolerance_m=0.15):
        context = self._sg_live_navmesh_context()
        metadata = {
            "status": "disabled" if context is None else "live_navmesh_same_island",
            "accepted_bounded_snap": 0,
            "rejected_other_island": 0,
            "rejected_snap_too_far": 0,
        }
        if context is None:
            return None, metadata
        pathfinder, floor_y = context
        metric_map = self._sg_world_metric_map()
        current_point = np.asarray([current[0], floor_y, current[1]], dtype=np.float32)
        current_point = np.asarray(pathfinder.snap_point(current_point), dtype=np.float64).reshape(-1)
        has_island_api = hasattr(pathfinder, "get_island")
        current_island = int(pathfinder.get_island(current_point[:3])) if has_island_api else None

        def validate(local_cell):
            global_cell = window.local_to_global(*local_cell)
            x, z = metric_map.spec.cell_center_world(*global_cell)
            point = np.asarray([x, floor_y, z], dtype=np.float32)
            validation = point
            if not bool(pathfinder.is_navigable(point, max_y_delta=0.5)):
                snapped = np.asarray(pathfinder.snap_point(point), dtype=np.float64).reshape(-1)
                if snapped.size < 3 or not np.isfinite(snapped[:3]).all():
                    return False
                distance = float(np.linalg.norm(snapped[[0, 2]] - point[[0, 2]]))
                if distance > float(terminal_snap_tolerance_m) + 1e-6:
                    metadata["rejected_snap_too_far"] += 1
                    return False
                validation = snapped[:3]
                metadata["accepted_bounded_snap"] += 1
            if has_island_api and int(pathfinder.get_island(validation)) != current_island:
                metadata["rejected_other_island"] += 1
                return False
            return True

        return validate, metadata

    def _sg_reachable_target(self, safe_mask, start, raw, exact, target_validator=None):
        from membridge.navigation.map_connection import reachable_component

        reachable = reachable_component(safe_mask, start)
        count = int(np.count_nonzero(reachable))
        raw = tuple(map(int, raw))
        if (
            0 <= raw[0] < reachable.shape[1]
            and 0 <= raw[1] < reachable.shape[0]
            and reachable[raw[1], raw[0]]
            and (target_validator is None or target_validator(raw))
        ):
            return raw, count, 0.0
        if exact:
            return None, count, math.inf
        cells = np.argwhere(reachable)
        ranked = sorted(
            ((float(np.linalg.norm(cell[[1, 0]] - raw)), (int(cell[1]), int(cell[0]))) for cell in cells),
            key=lambda item: (item[0], item[1][1], item[1][0]),
        )
        for distance, cell in ranked:
            if distance > self.sg_snap_radius_cells:
                break
            if target_validator is None or target_validator(cell):
                return cell, count, distance
        return None, count, ranked[0][0] if ranked else math.inf

    def _sg_plan_world_path(
        self,
        raw_target_world_xz,
        current_world_xz,
        *,
        exact=False,
        planning_anchor_world_xz=None,
        project_target_to_navmesh=True,
        navmesh_terminal_snap_tolerance_m=0.15,
        _expanded=False,
    ):
        from membridge.navigation import theta_star
        from membridge.navigation.map_connection import expanded_trajectory_anchors

        metric_map = self._sg_world_metric_map()
        graph = self._sg_trajectory_graph()
        if metric_map is None:
            self._sg_last_theta_failure = {"reason": "missing_world_metric_map"}
            return None
        spec = metric_map.spec
        current = np.asarray(current_world_xz, dtype=np.float64).reshape(2)
        target_world = np.asarray(raw_target_world_xz, dtype=np.float64).reshape(2)
        start_global = spec.world_to_cell(*current)
        target_global = spec.world_to_cell(*target_world)
        anchors = [np.asarray(value, dtype=np.float64).reshape(2) for value in (planning_anchor_world_xz or [])]
        padding = 60
        if anchors:
            cells = [spec.world_to_cell(*value) for value in anchors]
            low_x, high_x = sorted((start_global[0], target_global[0]))
            low_z, high_z = sorted((start_global[1], target_global[1]))
            overhang = max(
                0,
                low_x - min(value[0] for value in cells),
                max(value[0] for value in cells) - high_x,
                low_z - min(value[1] for value in cells),
                max(value[1] for value in cells) - high_z,
            )
            padding = max(padding, int(overhang) + 6)
        try:
            window = metric_map.planning_window(
                start_global, target_global, floor_id=0, padding_cells=padding,
                max_window_cells=4_000_000,
            )
        except Exception as exc:
            self._sg_last_theta_failure = {"reason": "world_planning_window_failed", "detail": str(exc)}
            return None
        corridor = centers = None
        corridor_meta = {"trajectory_corridor_status": "missing_trajectory_graph"}
        if graph is not None:
            try:
                corridor, centers, corridor_meta = graph.certified_masks(
                    window,
                    spec,
                    clearance_cells=2.5,
                    floor_id=0,
                    warm_start_world_xz=current,
                    max_terminal_connector_m=0.75,
                    connector_validator=self._sg_validate_warm_start_connector,
                )
            except Exception as exc:
                corridor_meta = {"trajectory_corridor_status": "failed", "error": str(exc)}
        safe = window.safe_mask(
            2.5,
            certified_free=corridor,
            certified_centers=centers,
            certified_center_slack_cells=2.0 if centers is not None else 0.0,
            certified_slack_obstacle_clearance_cells=2.0,
        )
        for blocked in self._sg_world_collision_blocked_cells:
            x, z = window.global_to_local(*blocked)
            if 0 <= z < safe.shape[0] and 0 <= x < safe.shape[1]:
                safe[z, x] = False
        start = window.global_to_local(*start_global)
        raw = window.global_to_local(*target_global)
        if not (0 <= start[0] < safe.shape[1] and 0 <= start[1] < safe.shape[0] and safe[start[1], start[0]]):
            self._sg_last_theta_failure = {
                "reason": "warm_start_not_clearance_safe_in_world_memory",
                **corridor_meta,
            }
            result = None
        else:
            validator, navmesh_meta = self._sg_world_navmesh_target_validator(
                window, current, navmesh_terminal_snap_tolerance_m
            )
            selected, reachable_count, raw_distance = self._sg_reachable_target(
                safe, start, raw, exact, target_validator=validator
            )
            if selected is None:
                self._sg_last_theta_failure = {
                    "reason": "no_world_reachable_target_within_snap_radius",
                    "reachable_cells": reachable_count,
                    "raw_distance_cells": raw_distance,
                    **corridor_meta,
                }
                result = None
            else:
                theta = theta_star(safe, start, selected)
                if not theta.path:
                    self._sg_last_theta_failure = {"reason": "world_theta_star_failed"}
                    result = None
                else:
                    raw_path = [window.local_to_global(*point) for point in theta.path]
                    path = [tuple(map(float, point)) for point in raw_path]
                    selected_global = window.local_to_global(*selected)
                    selected_world = spec.cell_center_world(*selected_global)
                    cumulative = self._sg_path_cumulative(path)
                    self._sg_world_plan_window = window
                    self._sg_world_safe_mask = safe
                    self._sg_world_trajectory_probe_mask = corridor
                    self._sg_last_theta_failure = None
                    result = {
                        "target_grid": selected_global,
                        "target_world_xz": selected_world,
                        "raw_target_grid": target_global,
                        "path": path,
                        "raw_path": raw_path,
                        "path_length_cells": float(cumulative[-1]),
                        "expanded_states": int(theta.expanded_states),
                        "reachable_cells": reachable_count,
                        "clearance_cells": 2.5,
                        "window_origin": (window.origin_grid_x, window.origin_grid_z),
                        "planning_padding_cells": padding,
                        "planning_route_anchor_count": len(anchors),
                        "shortcut_candidates": [],
                        "trajectory_corridor": corridor_meta,
                        "navmesh_target_validation": navmesh_meta,
                        "safe_mask": safe,
                        "window": window,
                        "_exploration_trajectory_probe_mask": corridor,
                    }
        if result is None and not _expanded and os.environ.get("LIFELONG_SG_WORLD_FULL_WINDOW_RETRY", "1") == "1":
            expanded = expanded_trajectory_anchors(self, anchors, current, target_world)
            if expanded:
                return self._sg_plan_world_path(
                    target_world,
                    current,
                    exact=exact,
                    planning_anchor_world_xz=expanded,
                    project_target_to_navmesh=project_target_to_navmesh,
                    navmesh_terminal_snap_tolerance_m=navmesh_terminal_snap_tolerance_m,
                    _expanded=True,
                )
        return result

    def _sg_trajectory_route_metadata(self, candidate, current_world_xz):
        return {"trajectory_route_status": "candidate_waypoints", "trajectory_route_world_xz": []}

    @staticmethod
    def _sg_trajectory_route_world_xz(route_metadata):
        return list(route_metadata.get("trajectory_route_world_xz") or [])

    @staticmethod
    def _sg_dual_anchor_navigation_variants(candidate, current_world_xz):
        variants = []
        for key, waypoint in sorted((candidate.get("navigation_waypoint_by_episode") or {}).items()):
            if not isinstance(waypoint, Mapping):
                continue
            variant = copy.deepcopy(candidate)
            variant["waypoint"] = copy.deepcopy(waypoint)
            observation = (candidate.get("best_observation_by_episode") or {}).get(key)
            if isinstance(observation, Mapping):
                variant["best_observation"] = copy.deepcopy(observation)
            variant["_sg_navigation_episode_id"] = str(key)
            variants.append(variant)
        return variants

    def _sg_plan_v8_post_hit_approach(
        self,
        candidate,
        current_world_xz,
        trajectory_route,
        route_planning_anchors,
    ):
        """Select the closest object-neighbourhood pose reachable in M0.

        The live navmesh is used only to validate/project possible terminal
        poses.  The path executed by the agent is still planned exclusively on
        the persisted memory_module world map.  A historical observation pose
        may provide an approach ray, but it can never become a terminal when
        it lies outside the semantic success radius.
        """

        from membridge.navigation.approach import (
            build_projected_approach_candidates,
        )

        context = self._sg_live_navmesh_context()
        if context is None:
            self._sg_last_theta_failure = {
                "reason": "v8_post_hit_live_navmesh_unavailable"
            }
            return None
        pathfinder, floor_y = context
        route_status = str(
            (trajectory_route or {}).get("trajectory_route_status", "missing")
        )
        try:
            approaches, approach_metadata = build_projected_approach_candidates(
                candidate,
                route_planning_anchors,
                pathfinder=pathfinder,
                current_world_xz=current_world_xz,
                floor_y=floor_y,
                min_object_distance_m=0.0,
                max_object_distance_m=self.sg_post_hit_max_object_distance_m,
                max_navmesh_snap_m=self.sg_post_hit_max_navmesh_snap_m,
                annulus_step_m=self.sg_post_hit_annulus_step_m,
                angular_samples=self.sg_post_hit_angular_samples,
                connectivity_check=self._sg_navmesh_points_connected,
            )
        except Exception as exc:
            self._sg_last_theta_failure = {
                "reason": "v8_post_hit_approach_construction_failed",
                "detail": str(exc),
            }
            return None
        if not approaches:
            self._sg_last_theta_failure = {
                "reason": "v8_post_hit_no_same_island_approach",
                **approach_metadata,
            }
            return None

        semantic_position = np.asarray(
            candidate.get("position"), dtype=np.float64
        ).reshape(-1)
        semantic_xz = semantic_position[[0, 2]]
        maximum_safe_snap = max(0.0, self.sg_post_hit_max_safe_snap_m)
        semantic_terminal_limit = max(
            0.25,
            self.sg_post_hit_max_object_distance_m - self.sg_nav_goal_tolerance_m,
        )
        tested = []
        for rank, approach in enumerate(
            approaches[: max(1, self.sg_post_hit_max_planner_candidates)]
        ):
            physical_xz = np.asarray(
                approach.projected_world_xz, dtype=np.float64
            )
            reachable = self._sg_plan_world_path(
                physical_xz,
                current_world_xz,
                planning_anchor_world_xz=route_planning_anchors,
                project_target_to_navmesh=False,
            )
            if reachable is None:
                tested.append(
                    {
                        "rank": rank,
                        "source": approach.source,
                        "status": "world_plan_failed",
                        "physical_object_distance_m": approach.object_distance_m,
                        "failure": copy.deepcopy(self._sg_last_theta_failure),
                    }
                )
                continue

            terminal_xz = np.asarray(
                reachable["target_world_xz"], dtype=np.float64
            )
            safe_snap_m = float(np.linalg.norm(terminal_xz - physical_xz))
            semantic_distance_m = float(np.linalg.norm(terminal_xz - semantic_xz))
            if safe_snap_m > maximum_safe_snap + 1e-6:
                tested.append(
                    {
                        "rank": rank,
                        "source": approach.source,
                        "status": "persisted_safe_snap_too_far",
                        "safe_snap_m": safe_snap_m,
                        "maximum_safe_snap_m": maximum_safe_snap,
                    }
                )
                continue
            if semantic_distance_m > semantic_terminal_limit + 1e-6:
                tested.append(
                    {
                        "rank": rank,
                        "source": approach.source,
                        "status": "terminal_ball_exceeds_semantic_radius",
                        "semantic_object_distance_m": semantic_distance_m,
                        "semantic_terminal_limit_m": semantic_terminal_limit,
                    }
                )
                continue

            navigation_reference = {
                "navigation_reference_policy": (
                    "v8_dense_same_island_closest_object_approach_set"
                ),
                "navigation_target_source": (
                    "v8_post_hit_closest_memory_reachable_approach"
                ),
                "semantic_object_center_world_xz": semantic_xz.tolist(),
                "navigation_reference_world_xz": list(map(float, physical_xz)),
                **approach_metadata,
                "approach_selected_rank": int(rank),
                "approach_selected_source": approach.source,
                "approach_selected_candidate": approach.to_dict(),
                "approach_selected_physical_world_xyz": list(
                    map(float, approach.projected_world_xyz)
                ),
                "approach_selected_physical_object_distance_m": float(
                    approach.object_distance_m
                ),
                "approach_selected_memory_terminal_world_xz": list(
                    map(float, terminal_xz)
                ),
                "approach_selected_memory_safe_snap_m": safe_snap_m,
                "approach_selected_memory_object_distance_m": semantic_distance_m,
                "approach_planner_candidates_tested": int(rank + 1),
                "approach_rejected_candidates": tested,
                "approach_execution_controller": (
                    "persisted_world_memory_discrete_action_lattice"
                ),
                "approach_navmesh_route_used": False,
                "approach_target_trajectory_route_status": route_status,
            }
            LOG.info(
                "selected V8 post-hit approach: source=%s rank=%d "
                "physical_object_distance=%.3fm memory_object_distance=%.3fm "
                "safe_snap=%.3fm",
                approach.source,
                rank,
                approach.object_distance_m,
                semantic_distance_m,
                safe_snap_m,
            )
            return {
                "raw_world_xz": tuple(map(float, physical_xz)),
                "navigation_reference": navigation_reference,
                "reachable": reachable,
            }

        self._sg_last_theta_failure = {
            "reason": "v8_post_hit_no_memory_reachable_approach",
            **approach_metadata,
            "approach_planner_candidate_limit": int(
                self.sg_post_hit_max_planner_candidates
            ),
            "tested": tested,
        }
        return None

    @staticmethod
    def _sg_compose_trajectory_guided_world_path(reachable, trajectory_route, current_world_xz):
        return {
            "trajectory_guidance_status": "world_theta_path",
            "path": list(reachable["path"]),
            "path_length_cells": float(reachable["path_length_cells"]),
        }

    def _sg_select_execution_world_path(
        self, reachable, trajectory_guidance, allow_shortcut=True
    ):
        return {
            "path": list(reachable["path"]),
            "execution_path_source": "memory_module_world_theta_star",
            "selection_reason": "closest_object_neighbourhood_reachable_in_m0",
            "path_length_cells": float(reachable["path_length_cells"]),
        }

    def _sg_coords_to_map(self, position_xyz):
        grid = self._sg_world_metric_map().spec.world_to_cell(position_xyz[0], position_xyz[2])
        return int(grid[0]), int(grid[1])

    def _sg_local_action_recovery_targets(self, target_info, position_grid, target_grid):
        metric_map = self._sg_world_metric_map()
        if metric_map is None:
            return []
        resolution = float(metric_map.spec.resolution_m)
        step_cells = float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE) / resolution
        radius_cells = max(step_cells * 2.0, self.sg_local_action_radius_m / resolution)
        position = np.asarray(position_grid, dtype=np.float64).reshape(2)
        final_target = np.asarray(target_grid, dtype=np.float64).reshape(2)
        if float(np.linalg.norm(position - final_target)) <= radius_cells + step_cells:
            return [{"kind": "final_target", "target": final_target, "path_progress_cells": None}]

        path = [
            np.asarray(point, dtype=np.float64)
            for point in target_info.get("_sg_world_path_cells") or []
        ]
        if len(path) < 2:
            target_info["_sg_local_action_status"] = "missing_world_path_for_local_subgoal"
            return []
        cumulative = self._sg_path_cumulative(path)
        reported_progress = float(target_info.get("_sg_path_progress_cells", 0.0))
        projected_progress, projected_cross_track = self._sg_project_position_to_path(
            path, position, reported_progress, cumulative=cumulative
        )
        maximum_cross_track = max(step_cells, self.sg_path_deviation_m / resolution)
        if projected_progress is not None and projected_cross_track <= maximum_cross_track + 1e-6:
            path_progress = max(reported_progress, projected_progress)
            target_info["_sg_path_progress_cells"] = float(path_progress)
        else:
            path_progress = reported_progress
        remaining = max(0.0, float(cumulative[-1]) - path_progress)
        if remaining <= 1e-6:
            target_info["_sg_local_action_status"] = "world_path_exhausted_before_final_target"
            return []

        lookahead = max(step_cells, self.sg_path_lookahead_m / resolution)
        maximum_advance = max(step_cells, radius_cells - 2.0 * step_cells)
        nominal = min(remaining, lookahead, maximum_advance)
        tolerance = max(0.5, self.sg_local_action_goal_tolerance_m / resolution)
        targets = []
        seen = set()
        for advance in (nominal, nominal * 0.75, nominal * 0.5, step_cells):
            advance = min(remaining, max(step_cells, float(advance)))
            candidate_progress = min(float(cumulative[-1]), path_progress + advance)
            candidate = self._sg_point_on_path(path, cumulative, candidate_progress)
            if candidate is None:
                continue
            distance = float(np.linalg.norm(candidate - position))
            if distance <= tolerance + 1e-6 or distance > radius_cells + step_cells + 1e-6:
                continue
            key = tuple(np.round(candidate, decimals=4))
            if key in seen:
                continue
            seen.add(key)
            targets.append(
                {
                    "kind": "world_path_subgoal",
                    "target": candidate,
                    "path_progress_cells": float(candidate_progress),
                }
            )
        if not targets:
            target_info["_sg_local_action_status"] = "no_bounded_world_path_subgoal"
        return targets

    def _sg_plan_local_action_sequence(self, position, heading, target, window, safe_mask, target_info):
        from membridge.navigation.action_lattice import ActionLatticeConfig, plan_action_lattice

        resolution = float(self._sg_world_metric_map().spec.resolution_m)

        navmesh_context = self._sg_navmesh_context()

        def segment_is_safe(left, right):
            if not self._sg_float_segment_is_safe(
                safe_mask, self._sg_window_point(window, left), self._sg_window_point(window, right)
            ):
                return False
            if navmesh_context is None:
                return True
            pathfinder, floor_y = navmesh_context
            start = np.asarray(
                [(float(left[0]) + 0.5) * resolution, floor_y, (float(left[1]) + 0.5) * resolution],
                dtype=np.float32,
            )
            requested = np.asarray(
                [(float(right[0]) + 0.5) * resolution, floor_y, (float(right[1]) + 0.5) * resolution],
                dtype=np.float32,
            )
            reached = np.asarray(pathfinder.try_step(start, requested), dtype=np.float64).reshape(-1)
            if reached.size < 3 or not np.isfinite(reached[:3]).all():
                return False
            requested_step = float(np.linalg.norm(requested[[0, 2]] - start[[0, 2]]))
            actual_step = float(np.linalg.norm(reached[[0, 2]] - start[[0, 2]]))
            endpoint_error = float(np.linalg.norm(reached[[0, 2]] - requested[[0, 2]]))
            return bool(actual_step >= requested_step * 0.5 and endpoint_error <= 0.05 + 1e-6)

        result = plan_action_lattice(
            position,
            heading,
            target,
            segment_is_safe=segment_is_safe,
            config=ActionLatticeConfig(
                forward_step_cells=0.25 / resolution,
                turn_angle_deg=30.0,
                goal_tolerance_cells=0.15 / resolution,
                radius_cells=2.5 / resolution,
                max_states=12000,
                max_actions=96,
                turn_cost=0.20,
            ),
        )
        target_info["_sg_local_action_status"] = result.status
        if result.status != "planned":
            return None
        return {
            "actions": [int(value) for value in result.actions],
            "expanded_states": result.expanded_states,
        }

    @staticmethod
    def _sg_discrete_heading_lattice(current_heading, turn_rad):
        if not np.isfinite(current_heading) or not np.isfinite(turn_rad) or turn_rad <= 0:
            return []
        count = int(round(2.0 * math.pi / float(turn_rad)))
        return [
            float(
                math.atan2(
                    math.sin(float(current_heading) + offset * float(turn_rad)),
                    math.cos(float(current_heading) + offset * float(turn_rad)),
                )
            )
            for offset in range(count)
        ]

    def _sg_world_path_progress_and_lookahead(
        self, path, position, previous_progress, window, safe_mask
    ):
        cumulative = self._sg_path_cumulative(path)
        projected, cross_track = self._sg_project_position_to_path(
            path, position, previous_progress, cumulative=cumulative
        )
        progress = (
            float(previous_progress)
            if projected is None
            else max(float(previous_progress), float(projected))
        )
        resolution = float(self._sg_world_metric_map().spec.resolution_m)
        step_cells = float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE) / resolution
        desired = min(float(cumulative[-1]), progress + self.sg_path_lookahead_m / resolution)
        minimum = min(float(cumulative[-1]), progress + step_cells)
        candidate_progress = desired
        lookahead = None
        position_local = self._sg_window_point(window, position)
        while candidate_progress >= minimum - 1e-6:
            candidate = self._sg_point_on_path(path, cumulative, candidate_progress)
            if candidate is not None and self._sg_float_segment_is_safe(
                safe_mask, position_local, self._sg_window_point(window, candidate)
            ):
                lookahead = candidate
                break
            candidate_progress -= step_cells
        if lookahead is None:
            lookahead = self._sg_point_on_path(path, cumulative, minimum)
        return progress, cross_track, lookahead, float(cumulative[-1])

    def _sg_replan_target_path(self, target_info, current_world_xz, reason):
        target_world = target_info.get("target_world_xz")
        if target_world is None:
            return None
        replanned = self._sg_plan_world_path(
            target_world,
            current_world_xz,
            exact=False,
            planning_anchor_world_xz=target_info.get(
                "_sg_route_planning_anchors_world_xz"
            ),
            project_target_to_navmesh=True,
            navmesh_terminal_snap_tolerance_m=0.75,
        )
        if replanned is None or len(replanned.get("path") or []) < 2:
            return None
        selected = self._sg_select_execution_world_path(
            replanned,
            self._sg_compose_trajectory_guided_world_path(
                replanned,
                target_info,
                np.asarray(current_world_xz, dtype=np.float64),
            ),
            allow_shortcut=True,
        )
        target_info["target_world_xz"] = list(map(float, replanned["target_world_xz"]))
        target_info["target_world_grid_x"] = int(replanned["target_grid"][0])
        target_info["target_world_grid_z"] = int(replanned["target_grid"][1])
        target_info["_sg_world_path_cells"] = list(selected["path"])
        target_info["_sg_path_progress_cells"] = 0.0
        target_info["execution_path_source"] = selected["execution_path_source"]
        self._sg_world_plan_window = replanned["window"]
        self._sg_world_safe_mask = replanned["safe_mask"]
        self._sg_world_trajectory_probe_mask = replanned.get(
            "_exploration_trajectory_probe_mask"
        )
        self._write_sg_decision(
            self.current_step,
            {
                "decision": "sg_world_path_replanned",
                "reason": reason,
                "waypoints": len(selected["path"]),
                "path_length_m": float(replanned["path_length_cells"])
                * float(self._sg_world_metric_map().spec.resolution_m),
            },
        )
        return replanned

    def _sg_register_world_collision_block(self, endpoint, actual_position_grid=None):
        values = np.asarray(endpoint, dtype=np.float64).reshape(2)
        self._sg_world_collision_blocked_cells.add((int(round(values[0])), int(round(values[1]))))

    def _get_sg_world_path_action(self, **kwargs):
        target_info = kwargs.get("target_info") or {}
        current = self._sg_current_world_xz(kwargs)
        metric_map = self._sg_world_metric_map()
        window, safe = self._sg_world_plan_window, self._sg_world_safe_mask
        if current is None or metric_map is None or window is None or safe is None:
            target_info["_sg_pending_abort_reason"] = "missing_world_path"
            return None
        resolution = float(metric_map.spec.resolution_m)
        position = current / resolution - 0.5
        pending = target_info.pop("_sg_pending_forward_world_grid_position", None)
        endpoint = target_info.pop("_sg_pending_forward_world_grid_endpoint", None)
        need_replan = False
        replan_reason = "missing_world_path"
        if pending is not None and np.linalg.norm(position - np.asarray(pending)) < (
            0.5 * float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE) / resolution
        ):
            if endpoint is not None:
                self._sg_register_world_collision_block(endpoint, position)
            need_replan = True
            replan_reason = "collision_displacement_too_small"
        path = [np.asarray(point, dtype=np.float64) for point in target_info.get("_sg_world_path_cells") or []]
        if len(path) < 2 or window is None or safe is None:
            need_replan = True
        target_grid = np.asarray(
            [target_info["target_world_grid_x"], target_info["target_world_grid_z"]], dtype=np.float64
        )
        heading = float(kwargs.get("current_world_heading", self._sg_world_heading_rad))
        if need_replan:
            replanned = self._sg_replan_target_path(target_info, current, replan_reason)
            if replanned is None:
                target_info["_sg_pending_abort_reason"] = (
                    f"{replan_reason}_world_theta_replan_failed"
                )
                return None
            path = [
                np.asarray(point, dtype=np.float64)
                for point in target_info.get("_sg_world_path_cells") or []
            ]
            window, safe = self._sg_world_plan_window, self._sg_world_safe_mask
            target_grid = np.asarray(
                [target_info["target_world_grid_x"], target_info["target_world_grid_z"]],
                dtype=np.float64,
            )

        previous_progress = float(target_info.get("_sg_path_progress_cells", 0.0))
        progress, cross_track, lookahead, total = self._sg_world_path_progress_and_lookahead(
            path, position, previous_progress, window, safe
        )
        if lookahead is None or cross_track * resolution > self.sg_path_deviation_m:
            replanned = self._sg_replan_target_path(
                target_info, current, f"cross_track_{cross_track * resolution:.3f}m"
            )
            if replanned is None:
                target_info["_sg_pending_abort_reason"] = "cross_track_world_theta_replan_failed"
                return None
            path = [
                np.asarray(point, dtype=np.float64)
                for point in target_info.get("_sg_world_path_cells") or []
            ]
            window, safe = self._sg_world_plan_window, self._sg_world_safe_mask
            progress, cross_track, lookahead, total = self._sg_world_path_progress_and_lookahead(
                path, position, 0.0, window, safe
            )

        target_info["_sg_path_progress_cells"] = float(progress)
        target_info["_sg_path_cross_track_m"] = float(cross_track * resolution)
        target_info["_sg_path_remaining_m"] = float((total - progress) * resolution)
        step_cells = float(self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE) / resolution
        turn_rad = math.radians(float(self.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE))
        headings = self._sg_discrete_heading_lattice(heading, turn_rad)
        navmesh_context = self._sg_navmesh_context()
        previous_forward = target_info.get("_sg_last_forward_world_grid_origin")
        current_goal_distance = float(np.linalg.norm(position - target_grid))
        alternatives = []
        for offset, candidate_heading in enumerate(headings):
            candidate_end, predicted = self._sg_predict_forward_endpoint(
                position, candidate_heading, step_cells, resolution, navmesh_context
            )
            displacement = float(np.linalg.norm(candidate_end - position))
            if predicted and displacement < step_cells * 0.5:
                continue
            if not self._sg_float_segment_is_safe(
                safe,
                self._sg_window_point(window, position),
                self._sg_window_point(window, candidate_end),
            ):
                continue
            candidate_progress, candidate_cross_track = self._sg_project_position_to_path(
                path, candidate_end, progress
            )
            if candidate_progress is None:
                continue
            progress_gain = float(candidate_progress - progress)
            goal_gain = current_goal_distance - float(np.linalg.norm(candidate_end - target_grid))
            advances = bool(progress_gain >= 0.25 or goal_gain >= 0.25)
            reverses = bool(
                previous_forward is not None
                and np.linalg.norm(candidate_end - np.asarray(previous_forward, dtype=np.float64))
                <= max(0.5, step_cells * 0.25)
            )
            if not advances or reverses:
                continue
            turns = min(offset, (-offset) % len(headings))
            score = (
                float(np.linalg.norm(candidate_end - np.asarray(lookahead, dtype=np.float64)))
                + 0.25 * turns
                + 0.10 * float(candidate_cross_track)
            )
            alternatives.append((score, turns, offset, candidate_end, predicted))

        if not alternatives:
            candidates = self._sg_local_action_recovery_targets(
                target_info, position, target_grid
            )
            for candidate in candidates:
                plan = self._sg_plan_local_action_sequence(
                    position, heading, candidate["target"], window, safe, target_info
                )
                if plan and plan["actions"]:
                    return int(plan["actions"][0])
            target_info["_sg_pending_abort_reason"] = (
                "local_action_lattice_unreachable_after_no_safe_path_forward"
            )
            return None

        _, turns, offset, candidate_end, _ = min(alternatives, key=lambda item: item[0])
        if turns:
            return Actions.TURN_LEFT if offset <= len(headings) // 2 else Actions.TURN_RIGHT

        selected = Actions.MOVE_FORWARD
        if selected == Actions.MOVE_FORWARD:
            predicted, _ = self._sg_predict_forward_endpoint(
                position, heading, 0.25 / resolution, resolution, self._sg_navmesh_context()
            )
            target_info["_sg_pending_forward_world_grid_position"] = position.tolist()
            target_info["_sg_pending_forward_world_grid_endpoint"] = predicted.tolist()
            target_info["_sg_last_forward_world_grid_origin"] = position.tolist()
        return selected

    @staticmethod
    def _sg_reset_world_path_stall_guard(target_info, clear_phase=False):
        """Reset progress state expressed in the active world-path frame."""

        target_info.pop("_sg_world_stall_path_progress_cells", None)
        target_info.pop("_sg_previous_world_grid_cell", None)
        target_info["_sg_non_improving_moves"] = 0
        pending = str(target_info.get("_sg_pending_abort_reason", ""))
        if pending.startswith("no_progress_before_exact_world_target_best_"):
            target_info.pop("_sg_pending_abort_reason", None)
        if clear_phase:
            target_info.pop("_sg_world_stall_phase", None)

    def _sg_world_nav_reached_reason(self, **kwargs):
        """Use memory_module's discrete world-target terminal semantics.

        In particular, the recalled semantic object's centre is never an
        arrival target.  The terminal is the reachable world-grid pose chosen
        by memory_module's post-hit planner.
        """

        target_info = kwargs.get("target_info") or {}
        current_world_xz = self._sg_current_world_xz(kwargs)
        metric_map = self._sg_world_metric_map()
        if current_world_xz is None or metric_map is None:
            return None
        resolution_m = float(metric_map.spec.resolution_m)
        current_grid_position = (
            np.asarray(current_world_xz, dtype=np.float64) / resolution_m - 0.5
        )
        current_grid_cell = metric_map.spec.world_to_cell(
            float(current_world_xz[0]), float(current_world_xz[1])
        )
        target_grid = (
            int(target_info["target_world_grid_x"]),
            int(target_info["target_world_grid_z"]),
        )
        dist_cells = float(
            np.linalg.norm(
                current_grid_position - np.asarray(target_grid, dtype=np.float64)
            )
        )
        if tuple(current_grid_cell) == target_grid:
            target_info["_sg_best_distance_cells"] = dist_cells
            target_info["_sg_best_cell"] = tuple(map(int, current_grid_cell))
            return "exact_world_grid_cell"

        goal_tolerance_m = max(
            0.0,
            float(
                getattr(
                    self,
                    "sg_nav_goal_tolerance_m",
                    getattr(self, "sg_local_action_goal_tolerance_m", 0.15),
                )
            ),
        )
        distance_m = dist_cells * resolution_m
        if distance_m <= goal_tolerance_m + 1e-6:
            target_info["_sg_best_distance_cells"] = dist_cells
            target_info["_sg_best_cell"] = tuple(map(int, current_grid_cell))
            return f"within_discrete_world_goal_tolerance_{distance_m:.2f}m"

        local_goal_tolerance = target_info.get(
            "_sg_local_action_goal_tolerance_cells"
        )
        if (
            local_goal_tolerance is not None
            and dist_cells <= float(local_goal_tolerance) + 1e-6
        ):
            target_info["_sg_best_distance_cells"] = dist_cells
            target_info["_sg_best_cell"] = tuple(map(int, current_grid_cell))
            return (
                "nearest_reachable_discrete_action_pose_"
                f"{distance_m:.2f}m"
            )

        best_dist = float(target_info.get("_sg_best_distance_cells", float("inf")))
        eps_cells = self.sg_nav_stall_eps_m / resolution_m
        distance_meaningfully_improved = dist_cells < best_dist - eps_cells
        closer = dist_cells < best_dist - 1e-6
        if closer:
            best_dist = dist_cells
            target_info["_sg_best_distance_cells"] = best_dist
            target_info["_sg_best_cell"] = tuple(map(int, current_grid_cell))

        if (
            target_info.get("_sg_local_action_queue")
            or target_info.get("_sg_local_action_recovery_active")
        ):
            self._sg_reset_world_path_stall_guard(target_info)
            target_info["_sg_world_stall_progress_source"] = (
                "bounded_local_action_recovery"
            )
            return None

        path = target_info.get("_sg_world_path_cells") or []
        meaningfully_progressed = distance_meaningfully_improved
        recovery_meaningfully_progressed = distance_meaningfully_improved
        if len(path) >= 2:
            cumulative = self._sg_path_cumulative(path)
            minimum_progress = max(
                float(target_info.get("_sg_path_progress_cells", 0.0)),
                float(
                    target_info.get("_sg_world_stall_path_progress_cells", 0.0)
                ),
            )
            projected_progress, _ = self._sg_project_position_to_path(
                path,
                current_grid_position,
                minimum_progress,
                cumulative=cumulative,
            )
            path_progress = (
                minimum_progress
                if projected_progress is None
                else max(minimum_progress, float(projected_progress))
            )

            suffix_start = target_info.get("_sg_trajectory_suffix_start_progress_cells")
            if suffix_start is None:
                phase = "world_theta"
            elif path_progress + 1e-6 >= float(suffix_start):
                phase = "trajectory_suffix"
            else:
                phase = "trajectory_route"
            if target_info.get("_sg_world_stall_phase") != phase:
                self._sg_reset_world_path_stall_guard(target_info)
                target_info["_sg_world_stall_phase"] = phase

            stall_progress = target_info.get("_sg_world_stall_path_progress_cells")
            if stall_progress is None:
                target_info["_sg_world_stall_path_progress_cells"] = float(
                    path_progress
                )
                meaningfully_progressed = True
            else:
                path_gain = path_progress - float(stall_progress)
                meaningfully_progressed = path_gain >= eps_cells - 1e-6
                if meaningfully_progressed:
                    recovery_meaningfully_progressed = True
                    target_info["_sg_world_stall_path_progress_cells"] = float(
                        path_progress
                    )
            target_info["_sg_world_stall_progress_source"] = "planned_path"
        else:
            target_info["_sg_world_stall_progress_source"] = (
                "target_distance_fallback"
            )

        if meaningfully_progressed:
            target_info["_sg_non_improving_moves"] = 0
        if recovery_meaningfully_progressed:
            target_info["_sg_collision_replans_without_progress"] = 0
            target_info["_sg_consecutive_failed_recoveries"] = 0

        previous_cell = target_info.get("_sg_previous_world_grid_cell")
        if previous_cell is not None and tuple(previous_cell) != tuple(current_grid_cell):
            if not meaningfully_progressed:
                target_info["_sg_non_improving_moves"] = int(
                    target_info.get("_sg_non_improving_moves", 0)
                ) + 1
        target_info["_sg_previous_world_grid_cell"] = tuple(
            map(int, current_grid_cell)
        )

        best_dist = float(target_info.get("_sg_best_distance_cells", dist_cells))
        at_best = dist_cells <= best_dist + 1e-6
        best_m = best_dist * resolution_m
        stalled = int(target_info.get("_sg_non_improving_moves", 0)) >= int(
            self.sg_nav_stall_steps
        )
        forward_step_m = float(
            self.config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE
        )
        path_remaining_m = float(
            target_info.get("_sg_path_remaining_m", float("inf"))
        )
        at_discrete_path_end = (
            path_remaining_m <= forward_step_m + self.sg_nav_stall_eps_m
        )
        if at_discrete_path_end and at_best:
            window = getattr(self, "_sg_world_plan_window", None)
            safe_mask = getattr(self, "_sg_world_safe_mask", None)
            has_safe_improving_forward = False
            if window is not None and safe_mask is not None:
                current_local = self._sg_window_point(window, current_grid_position)
                step_cells = forward_step_m / resolution_m
                supplied_heading = kwargs.get("current_world_heading")
                if supplied_heading is None:
                    supplied_heading = getattr(self, "_sg_world_heading_rad", None)
                if supplied_heading is None:
                    target_info["_sg_pending_abort_reason"] = (
                        "missing_current_world_heading_at_path_end"
                    )
                    return None
                current_heading = float(np.asarray(supplied_heading).reshape(-1)[0])
                turn_rad = np.deg2rad(
                    float(self.config.TASK_CONFIG.SIMULATOR.TURN_ANGLE)
                )
                headings = self._sg_discrete_heading_lattice(
                    current_heading, turn_rad
                )
                for heading in headings:
                    candidate = current_grid_position + step_cells * np.asarray(
                        [-np.sin(heading), -np.cos(heading)], dtype=np.float64
                    )
                    candidate_local = self._sg_window_point(window, candidate)
                    if not self._sg_float_segment_is_safe(
                        safe_mask, current_local, candidate_local
                    ):
                        continue
                    candidate_distance = float(
                        np.linalg.norm(
                            candidate - np.asarray(target_grid, dtype=np.float64)
                        )
                    )
                    if candidate_distance < dist_cells - 1e-6:
                        has_safe_improving_forward = True
                        break
            if not has_safe_improving_forward:
                return (
                    "no_safe_improving_discrete_forward_at_world_path_end_"
                    f"{distance_m:.2f}m"
                )
        if stalled:
            target_info["_sg_pending_abort_reason"] = (
                f"no_progress_before_exact_world_target_best_{best_m:.2f}m"
            )
        return None

    def _sg_nav_reached_reason(self, **kwargs):
        target_info = kwargs.get("target_info") or {}
        if target_info.get("target_world_xz") is not None:
            return self._sg_world_nav_reached_reason(**kwargs)
        return None

    def _on_sg_nav_aborted(self, **kwargs):
        self._write_sg_decision(
            kwargs.get("step"),
            {"decision": "sg_navigation_aborted_to_janus", "reason": kwargs.get("reason")},
        )
        return False

    def _sg_nav_begin_progress(self, step, distance):
        self._sg_nav_progress = {"step": int(step), "distance": float(distance)}

    def _sg_nav_reset_progress(self):
        self._sg_nav_progress = None

    @staticmethod
    def _sg_task_key(text):
        return normalize_text(text).lower()

    @staticmethod
    def _sg_target_key(candidate):
        return str(candidate.get("uid") or candidate.get("node_id") or "")

    def _write_sg_decision(self, step, payload):
        append_jsonl(
            self.decision_log,
            {"step": int(step or 0), "episode_id": self.current_episode_id, **copy.deepcopy(payload)},
        )


def install_full_exploration_overlay() -> None:
    from membridge.habitat_runtime.full_exploration import install_goal_directed_exploration
    from membridge.habitat_runtime.navigation_adapter import install_navigation_runtime_adapter

    # Keep memory_module's smoothing and bounded action-lattice checks in the
    # path executor.  Install exploration last so geometric failures enter its
    # frontier/scan/reconnect controller.
    install_navigation_runtime_adapter(MemoryNavigator, Actions)
    install_goal_directed_exploration(MemoryNavigator, Actions, LOG)


def prepare_recall_plan(adapter: Any, route: Mapping[str, Any]) -> dict[str, Any]:
    from membridge.vlm_instruction_parser import InstructionSegment, resolve_recall_query
    from membridge.vlm_top3_selection import (
        CandidateImagePreparer,
        VLMNoMatchError,
        VLMTop3Reranker,
    )
    from membridge.vlm_instruction_parser import OpenAICompatibleVLMClient

    raw = route["recall"]
    segment = InstructionSegment(
        type=1,
        text=normalize_text(raw.get("text")),
        label=normalize_text(raw.get("label")),
        room=normalize_text(raw.get("room")) or None,
        recall_scope=raw.get("recall_scope") or "standard",
    )
    vocabulary = adapter.runtime_query_vocabulary or {}
    resolved = resolve_recall_query(
        segment,
        label_aliases=vocabulary.get("aliases") or {},
        room_aliases=vocabulary.get("room_aliases") or {},
    )
    if resolved.label not in set(vocabulary.get("labels") or []):
        return {"status": "fallback", "reason": "recall_label_outside_runtime_vocabulary"}
    top3 = adapter.query_two_stage(
        task_text=segment.text,
        room_hint=resolved.room,
        target_object=resolved.label,
        top_k=3,
    )
    if len(top3) != 3:
        return {
            "status": "fallback",
            "reason": "sg_retrieval_did_not_return_exact_top3",
            "retrieved_count": len(top3),
            "raw_top3": top3,
        }
    reranker = VLMTop3Reranker(
        OpenAICompatibleVLMClient.from_env(),
        image_preparer=CandidateImagePreparer(
            session_dir=str(adapter.preload_dir), image_mode="frame_box"
        ),
        fallback_policy="top1",
        selection_mode="force_single",
        max_tokens=2048,
    )
    try:
        rerank = reranker.rerank(segment.text, top3)
    except VLMNoMatchError as exc:
        return {
            "status": "fallback",
            "reason": "vlm_top3_no_match",
            "selection_source": "vlm_explicit_no_match",
            "raw_top3": top3,
            "rejection": exc.to_dict(),
        }
    return {
        "status": "ready",
        "reason": "vlm_pipeline_failed_top1" if rerank.used_fallback else "vlm_selected_unique_candidate",
        "selection_source": "sg_raw_top1_after_vlm_pipeline_failed" if rerank.used_fallback else "vlm_unique_candidate",
        "task_text": segment.text,
        "room_hint": resolved.room,
        "target_object": resolved.label,
        "raw_top3": top3,
        "selected_candidate": copy.deepcopy(rerank.selected_candidate),
        "selected_uid": rerank.selected_uid,
        "rerank": rerank.to_dict(include_candidate=True, include_raw_response=True),
    }


def load_saved_recall_plan(
    roots: Sequence[Path], row: Mapping[str, Any], *, stage: str = "M1"
) -> dict[str, Any]:
    """Load an already completed VLM Top3 decision without calling the API."""

    session_id = str(row["session_id"])
    episode_id = str(row["episode_id"])
    candidates = [
        root / "sessions" / session_id / "stages" / stage / "recall_plan.json"
        for root in reversed(roots)
    ]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError(
            f"no saved recall plan for {stage} episode {episode_id} "
            f"({session_id}) in {list(roots)}"
        )
    plan = copy.deepcopy(read_json(source))
    if plan.get("status") != "ready" or not plan.get("selected_candidate"):
        raise RuntimeError(
            f"saved recall plan is not an SG hit for {stage} episode {episode_id}: "
            f"status={plan.get('status')} reason={plan.get('reason')} source={source}"
        )
    expected_task = normalize_text((row["route"].get("recall") or {}).get("text"))
    if normalize_text(plan.get("task_text")) != expected_task:
        raise RuntimeError(
            f"saved recall task differs from current route for {stage} episode "
            f"{episode_id}: {source}"
        )
    selected = plan.get("selected_candidate") or {}
    selected_uid = str(plan.get("selected_uid") or "")
    candidate_uid = str(selected.get("uid") or selected.get("node_id") or "")
    if not selected_uid or selected_uid != candidate_uid:
        raise RuntimeError(
            f"saved recall selected candidate mismatch for {stage} episode "
            f"{episode_id}: {source}"
        )
    plan["reused_saved_vlm_top3_selection"] = True
    plan["saved_recall_plan_source"] = str(source.resolve())
    plan["saved_recall_plan_sha256"] = sha256_file(source)
    return plan


def make_target_info(navigator: MemoryNavigator, candidate: Mapping[str, Any], task_text: str):
    current = navigator._sg_current_world_xz()
    route_meta = navigator._sg_trajectory_route_metadata(candidate, current)
    anchors = navigator._sg_trajectory_route_world_xz(route_meta)
    approach = navigator._sg_plan_v8_post_hit_approach(candidate, current, route_meta, anchors)
    if approach is None:
        return None
    reachable = approach["reachable"]
    selection = navigator._sg_select_execution_world_path(
        reachable,
        navigator._sg_compose_trajectory_guided_world_path(reachable, route_meta, current),
        allow_shortcut=True,
    )
    target = copy.deepcopy(dict(candidate))
    target.update(copy.deepcopy(route_meta))
    target.update(copy.deepcopy(approach["navigation_reference"]))
    target.update(
        {
            "todo_text": task_text,
            "query_source": "unilavira_saved_route",
            "target_world_xz": list(map(float, reachable["target_world_xz"])),
            "target_world_grid_x": int(reachable["target_grid"][0]),
            "target_world_grid_z": int(reachable["target_grid"][1]),
            "raw_target_world_grid_x": int(reachable["raw_target_grid"][0]),
            "raw_target_world_grid_z": int(reachable["raw_target_grid"][1]),
            "world_map_resolution_m": float(navigator._sg_world_metric_map().spec.resolution_m),
            "_sg_world_path_cells": list(selection["path"]),
            "_sg_path_progress_cells": 0.0,
            "_sg_route_planning_anchors_world_xz": anchors,
            "execution_path_source": selection["execution_path_source"],
            "theta_path_waypoints": len(selection["path"]),
            "theta_path_length_m": float(reachable["path_length_cells"])
            * float(navigator._sg_world_metric_map().spec.resolution_m),
        }
    )
    navigator._sg_world_plan_window = reachable.get("window")
    navigator._sg_world_safe_mask = reachable.get("safe_mask")
    navigator._sg_world_trajectory_probe_mask = reachable.get("_exploration_trajectory_probe_mask")
    navigator._sg_path_plan_active = True
    return target


def publish_m1(m0: Path, run_dir: Path, destination: Path, metadata: Mapping[str, Any]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".M1.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    if temporary.exists() or destination.exists():
        raise FileExistsError(f"refusing to replace M1: {destination}")
    shutil.copytree(m0, temporary)
    for name in ("world_metric_map.npz", "trajectory_graph.npz"):
        source = run_dir / name
        if not source.is_file():
            raise RuntimeError(f"world-map-only EP1 did not produce {source}")
        shutil.copy2(source, temporary / name)
    source_manifest = run_dir / "episode_manifest.jsonl"
    if source_manifest.is_file():
        with (temporary / "episode_manifest.jsonl").open("a", encoding="utf-8") as output:
            output.write(source_manifest.read_text(encoding="utf-8"))
    for name in ("recall_plan.json", "sg_navigation_decisions.jsonl", "rollout.json"):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, temporary / name)
    stage_manifest = {
        "schema_version": "janusvln_level2_memory_module_stage_v2",
        "session_id": metadata["session_id"],
        "episode_id": metadata["episode_id"],
        "role": "ep1",
        "memory_read": "M0",
        "memory_write": "M1",
        "semantic_sg_policy": "frozen_m0_no_sam3_no_dinov3",
        "navigation_memory_policy": "live_rgbd_world_map_and_trajectory_update",
        "raw_rgbd_persisted": False,
        "rollout_pose_persisted": "position_xyz_plus_rotation_xyzw",
        "route": copy.deepcopy(metadata["route"]),
        "sg_handoff": copy.deepcopy(metadata["sg_handoff"]),
        "artifacts_sha256": {
            name: sha256_file(temporary / name)
            for name in CORE_M0_FILES
            if (temporary / name).is_file()
        },
    }
    atomic_json(temporary / "stage_manifest.json", stage_manifest)
    os.replace(temporary, destination)
    return destination


def metric_value(metrics: Mapping[str, Any], name: str):
    value = metrics.get(name)
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def run_one(
    *, env: Any, habitat_config: Any, episode: Any, row: Mapping[str, Any], model: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from membridge.habitat_runtime.online_adapter import OnlineINHeritSGBuilder
    from membridge.trajectory_graph.habitat_adapter import heading_from_quaternion_xyzw

    session_id, episode_id = row["session_id"], row["episode_id"]
    destination = args.output_root / "sessions" / session_id / "stages/M1"
    if destination.exists():
        raise FileExistsError(f"M1 already exists without a completed rollout row: {destination}")
    work_root = args.output_root / ".working" / f"{session_id}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    work_root.mkdir(parents=True)
    install_memory_environment(args, work_root, row["m0"])

    env.current_episode = episode
    observations = env.reset()
    initial_pose = pose_from_env(env, 0, 0)
    adapter = OnlineINHeritSGBuilder(
        sensor_memory_config(habitat_config, args.sim_gpu_id),
        width=int(observations["rgb"].shape[1]),
        height=int(observations["rgb"].shape[0]),
        save_dir=str(work_root),
        device=args.device,
    )
    adapter.start_episode(episode_id, scene_id=str(episode.scene_id), source_episode_id=episode_id)
    adapter.process_observation(memory_observation(observations, initial_pose), step=0)
    run_dir = Path(adapter.pipeline.run_dir).resolve()

    navigator = MemoryNavigator(
        adapter=adapter,
        env=env,
        habitat_config=habitat_config,
        decision_log=run_dir / "sg_navigation_decisions.jsonl",
    )
    navigator.current_episode_id = episode_id
    heading = heading_from_quaternion_xyzw(initial_pose["rotation_xyzw"])
    navigator.update_pose(initial_pose["position"], heading, 0)
    model.reset_memory()

    route = row["route"]
    recall_plan = (
        load_saved_recall_plan(args.recall_plan_root, row)
        if args.recall_plan_root
        else prepare_recall_plan(adapter, route)
    )
    atomic_json(run_dir / "recall_plan.json", recall_plan)
    target_info = None
    phase = "janus_fallback"
    active_instruction = row["instruction"]
    if recall_plan["status"] == "ready":
        target_info = make_target_info(
            navigator, recall_plan["selected_candidate"], recall_plan["task_text"]
        )
        if target_info is not None:
            phase = "sg_navigation"

    poses = [initial_pose]
    rgb_history = [Image.fromarray(np.asarray(observations["rgb"], dtype=np.uint8)).convert("RGB")]
    requested_actions: list[str] = []
    executed_actions: list[str] = []
    model_step_seconds: list[float] = []
    sg_actions = 0
    janus_actions = 0
    first_janus_history_frames = None
    anchor_reached = False
    fallback_reason = None if target_info is not None else recall_plan.get("reason", "sg_plan_failed")
    stopped = False
    started_at, wall_start = utc_now(), time.monotonic()

    def make_rollout_record(sg_handoff: Mapping[str, Any]) -> dict[str, Any]:
        metrics = env.get_metrics()
        ne = metric_value(metrics, "distance_to_goal")
        oracle_ne = metric_value(metrics, "oracle_navigation_error")
        return {
            "schema_version": "janusvln_level2_ep1_memory_module_rollout_v1",
            "phase": "level2_ep1",
            "episode_id": episode_id,
            "scene_id": str(episode.scene_id),
            "session_id": session_id,
            "role": "ep1",
            "instruction": row["instruction"],
            "route": route,
            "sg_handoff": copy.deepcopy(dict(sg_handoff)),
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
                "world_navigation_map_updated_from_live_rgbd": True,
                "trajectory_graph_updated": True,
                "sam3_loaded": False,
                "dinov3_loaded": False,
                "navigation_controller_version": NAVIGATION_CONTROLLER_VERSION,
            },
            "metrics": {
                "ne": ne,
                "distance_to_goal": ne,
                "success": None if ne is None else float(ne <= 3.0),
                "habitat_success": metric_value(metrics, "success"),
                "spl": metric_value(metrics, "spl"),
                "oracle_navigation_error": oracle_ne,
                "oracle_success": None if oracle_ne is None else float(oracle_ne <= 3.0),
            },
            "timing": {
                "started_at": started_at,
                "finished_at": utc_now(),
                "wall_seconds": time.monotonic() - wall_start,
                "janus_model_step_seconds": model_step_seconds,
            },
        }

    try:
        while len(executed_actions) < args.max_steps and not env.episode_over:
            step = len(executed_actions)
            current_pose = poses[-1]
            heading = heading_from_quaternion_xyzw(current_pose["rotation_xyzw"])
            navigator.update_pose(current_pose["position"], heading, step)

            if phase == "sg_navigation":
                reached = navigator._sg_nav_reached_reason(
                    step=step,
                    current_world_position=current_pose["position"],
                    current_world_heading=heading,
                    target_info=target_info,
                )
                if reached is not None:
                    anchor_reached = True
                    navigator._write_sg_decision(
                        step,
                        {
                            "decision": "sg_anchor_reached_handoff_to_janus",
                            "reach_reason": reached,
                            "merged_suffix": route["merged_suffix"],
                        },
                    )
                    if route["merged_suffix"]:
                        phase = "janus_suffix"
                        active_instruction = route["merged_suffix"]
                        continue
                    action = Actions.STOP
                    requested = "SG:STOP_PURE_RECALL"
                else:
                    action = navigator._get_sg_world_path_action(
                        step=step,
                        current_world_position=current_pose["position"],
                        current_world_heading=heading,
                        target_info=target_info,
                    )
                    if action is None:
                        reason = target_info.pop("_sg_pending_abort_reason", "sg_controller_no_action")
                        recovered = navigator._on_sg_nav_aborted(
                            step=step,
                            reason=reason,
                            current_world_position=current_pose["position"],
                            current_world_heading=heading,
                            target_info=target_info,
                        )
                        if recovered:
                            continue
                        fallback_reason = reason
                        phase = "janus_fallback"
                        active_instruction = row["instruction"]
                        continue
                    requested = "SG:" + ACTION_NAMES[int(action)]
                    sg_actions += 1
            else:
                if first_janus_history_frames is None:
                    first_janus_history_frames = len(rgb_history)
                images = sample_history(rgb_history, args.num_history)
                inference_start = time.monotonic()
                answer = model.call_model(images, active_instruction, step)[0]
                model_step_seconds.append(time.monotonic() - inference_start)
                action = getattr(Actions, str(answer), Actions.STOP)
                requested = "JANUS:" + str(answer)
                janus_actions += 1

            requested_actions.append(requested)
            executed_actions.append(ACTION_NAMES[int(action)])
            observations = env.step(int(action))
            pose = pose_from_env(env, len(poses), len(executed_actions))
            poses.append(pose)
            rgb_history.append(
                Image.fromarray(np.asarray(observations["rgb"], dtype=np.uint8)).convert("RGB")
            )
            adapter.process_observation(
                memory_observation(observations, pose), step=len(executed_actions)
            )
            if int(action) == Actions.STOP:
                stopped = True

        adapter.finish_episode()
        if adapter.failed:
            raise RuntimeError("memory_module marked EP1 world-map update failed")
        sg_handoff = {
            "recall_plan_status": recall_plan["status"],
            "recall_plan_reason": recall_plan.get("reason"),
            "selected_uid": recall_plan.get("selected_uid"),
            "anchor_reached": anchor_reached,
            "fallback_reason": fallback_reason,
            "janus_instruction_mode": (
                "merged_type2_suffix"
                if anchor_reached and route["merged_suffix"]
                else "stop_after_pure_recall"
                if anchor_reached
                else "full_instruction_fallback"
            ),
            "merged_type2_count": len(route["suffix_parts"]),
            "merged_suffix": route["merged_suffix"],
            "sg_actions": sg_actions,
            "janus_actions": janus_actions,
            "sg_visual_history_frames_available_to_first_janus_call": first_janus_history_frames,
            "exploration_implementation": os.environ["MEMORY_MODULE_EXPLORATION_IMPLEMENTATION"],
            "navigation_controller_version": NAVIGATION_CONTROLLER_VERSION,
        }
        result = make_rollout_record(sg_handoff)
        atomic_json(run_dir / "rollout.json", result)
        publish_m1(
            row["m0"], run_dir, destination,
            {"session_id": session_id, "episode_id": episode_id, "route": route, "sg_handoff": sg_handoff},
        )
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


def write_summary(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> None:
    metrics = ("success", "spl", "oracle_success", "distance_to_goal")
    means = {}
    for name in metrics:
        values = [row["metrics"].get(name) for row in rows]
        values = [float(value) for value in values if value is not None]
        means[name] = sum(values) / len(values) if values else None
    atomic_json(
        args.output_root / "summary.json",
        {
            "schema_version": "janusvln_level2_ep1_memory_module_summary_v1",
            "episode_count": len(rows),
            "means": means,
            "anchor_reached_count": sum(bool(row["sg_handoff"]["anchor_reached"]) for row in rows),
            "sg_fallback_count": sum(row["sg_handoff"]["fallback_reason"] is not None for row in rows),
        },
    )


def consolidate_output(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Publish one race-free aggregate while independent shards are active."""

    lock_path = args.output_root / ".aggregate.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        all_rows = []
        for path in sorted((args.output_root / "raw").glob("ep1_*.jsonl")):
            all_rows.extend(read_jsonl(path))
        by_episode = {str(row["episode_id"]): row for row in all_rows}
        ordered = [by_episode[key] for key in sorted(by_episode, key=int)]
        atomic_jsonl(args.output_root / "episodes.jsonl", ordered)
        write_summary(args, ordered)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return ordered


def main() -> None:
    args = parse_args()
    args.janus_root = args.janus_root.resolve()
    args.model_path = args.model_path.resolve()
    args.config = args.config.resolve()
    args.dataset = args.dataset.resolve()
    args.m0_root = args.m0_root.resolve()
    args.route_summary = args.route_summary.resolve()
    args.recall_plan_root = [path.resolve() for path in args.recall_plan_root]
    args.output_root = args.output_root.resolve()
    args.query_vocabulary = args.query_vocabulary.resolve()
    args.run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    selected = validate_inputs(args)
    pattern_counts: dict[str, int] = {}
    for row in selected:
        key = str(row["route"]["types"])
        pattern_counts[key] = pattern_counts.get(key, 0) + 1
    print(json.dumps({"selected": len(selected), "route_patterns": pattern_counts}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    # Habitat resolves ``data/scene_datasets`` relative to the process CWD in
    # JanusVLN's protocol config.
    os.chdir(args.janus_root)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(args.janus_root / "src"))
    install_full_exploration_overlay()
    from evaluation import JanusVLN_Inference, set_seed
    from protocol_benchmark import group_level2_episodes, make_env

    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    worker = args.worker_id.strip() or f"shard{args.shard_index:02d}"
    raw_log = args.output_root / "raw" / f"ep1_{worker}.jsonl"
    completed = {str(row["episode_id"]) for row in read_jsonl(raw_log)}
    env, habitat_config = make_env(args.config, "level2", args.sim_gpu_id, args.dataset)
    grouped = group_level2_episodes(env.episodes)
    ep1_by_session = {
        session_id: next(
            episode for episode in episodes
            if str((getattr(episode, "info", {}) or {}).get("level2_role")) == "ep1"
        )
        for session_id, episodes in grouped.items()
    }
    model = JanusVLN_Inference(str(args.model_path), device=args.device)
    failures = []
    try:
        for index, row in enumerate(selected, 1):
            episode_id = row["episode_id"]
            if episode_id in completed:
                continue
            destination = args.output_root / "sessions" / row["session_id"] / "stages/M1"
            if destination.is_dir():
                embedded = destination / "rollout.json"
                if embedded.is_file():
                    recovered = read_json(embedded)
                    if str(recovered.get("episode_id")) != episode_id:
                        raise RuntimeError(f"embedded rollout/M1 mismatch: {embedded}")
                    append_jsonl(raw_log, recovered)
                    completed.add(episode_id)
                    LOG.info("recovered completed rollout from atomic M1: %s", embedded)
                    continue
                raise RuntimeError(
                    f"M1 exists without its atomic rollout record; refusing unsafe skip: {destination}"
                )
            LOG.info("EP1 %d/%d session=%s episode=%s route=%s", index, len(selected), row["session_id"], episode_id, row["route"]["types"])
            try:
                result = run_one(
                    env=env,
                    habitat_config=habitat_config,
                    episode=ep1_by_session[row["session_id"]],
                    row=row,
                    model=model,
                    args=args,
                )
                append_jsonl(raw_log, result)
                completed.add(episode_id)
                print(json.dumps({"episode": episode_id, "metrics": result["metrics"], "handoff": result["sg_handoff"]}, ensure_ascii=False), flush=True)
            except Exception as exc:
                failure = {
                    "session_id": row["session_id"],
                    "episode_id": episode_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "time": utc_now(),
                }
                failures.append(failure)
                append_jsonl(args.output_root / "raw" / f"failures_{worker}.jsonl", failure)
                LOG.exception("EP1 failed: %s", failure)
                if not args.continue_on_error:
                    raise
    finally:
        env.close()

    consolidate_output(args)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
