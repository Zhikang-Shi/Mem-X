"""Load Level-2 RGB-D keyframes and convert world poses into episode-local GPS."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from PIL import Image

from se2 import habitat_camera_to_world, world_agent_to_local, wrap_angle, yaw_from_xyzw

HFOV_DEG = 90.0
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 720
DEPTH_SCALE = 1000.0
START_RADIUS_M = 0.20


def rollout_world_poses(poses: list[dict]) -> list[dict]:
    """Recover Habitat poses from exports that also carry episode-local xyz.

    Local exports retain world rotation_xyzw and save the original xyz in
    world_position. Rendering and replay must use that original xyz.
    """
    return [{**pose, "position": pose.get("world_position", pose["position"])}
            for pose in poses]


def camera_intrinsics(width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT, hfov_deg: float = HFOV_DEG):
    fx = width / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))
    return float(fx), float(fx), width / 2.0, height / 2.0


@dataclass
class EpisodeStart:
    xyz: np.ndarray
    yaw: float
    episode_id: str
    role: str


@dataclass
class LocalFrame:
    frame_idx: int
    rgb: np.ndarray
    depth_m: np.ndarray
    x: float
    y: float
    z: float
    yaw: float
    camera_to_local: np.ndarray
    world_xyz: np.ndarray = field(repr=False)
    world_yaw: float = 0.0
    episode_name: str = ""
    rgb_path: str = ""


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_episode_start(routing_json: Path, role: str) -> EpisodeStart:
    payload = _read_json(routing_json)
    audit = payload.get("start_audit") or {}
    xyz = np.asarray(audit["protocol_start_position"], dtype=np.float64).reshape(3)
    yaw = yaw_from_xyzw(audit["protocol_start_rotation_xyzw"])
    return EpisodeStart(
        xyz=xyz,
        yaw=float(yaw),
        episode_id=str(payload.get("episode_id", "")),
        role=role,
    )


def load_dataset_starts(dataset_json: Path) -> dict[str, dict[str, EpisodeStart]]:
    payload = _read_json(dataset_json)
    starts: dict[str, dict[str, EpisodeStart]] = {}
    for episode in payload.get("episodes", []):
        info = episode.get("info") or {}
        session_id = str(info.get("lifelong_session_id") or info.get("level2_session_id") or "")
        role = str(info.get("lifelong_role") or info.get("level2_role") or "")
        if not session_id or role not in {"ep0", "ep1", "ep2"}:
            continue
        xyz = np.asarray(episode["start_position"], dtype=np.float64).reshape(3)
        yaw = yaw_from_xyzw(episode["start_rotation"])
        starts.setdefault(session_id, {})[role] = EpisodeStart(
            xyz=xyz,
            yaw=float(yaw),
            episode_id=str(episode.get("episode_id", "")),
            role=role,
        )
    return starts


def discover_session_dirs(workers_root: Path) -> list[Path]:
    return sorted(path for path in workers_root.glob("*/sessions/level2_*") if path.is_dir())


def discover_native_sessions(run_root: Path) -> list[Path]:
    """Prefer adopted_sessions, then worker sessions, unique by session id."""
    run_root = Path(run_root)
    found: dict[str, Path] = {}
    adopted = run_root / "adopted_sessions"
    if adopted.is_dir():
        for path in sorted(adopted.iterdir()):
            native = path / "native_frames"
            if path.is_dir() and path.name.startswith("level2_") and _native_has_ep0_ep1(native):
                found[path.name] = native
    workers = run_root / "workers"
    if workers.is_dir():
        for worker in sorted(workers.iterdir()):
            sessions = worker / "sessions"
            if not sessions.is_dir():
                continue
            for path in sorted(sessions.iterdir()):
                if path.name in found or not path.name.startswith("level2_"):
                    continue
                native = path / "native_frames"
                if _native_has_ep0_ep1(native):
                    found[path.name] = native
    return [found[key] for key in sorted(found)]


def _native_has_ep0_ep1(native_dir: Path) -> bool:
    manifest = native_dir / "capture_manifest.json"
    if not manifest.exists():
        return False
    spans = {item.get("role") for item in _read_json(manifest).get("stage_spans", [])}
    return "ep0" in spans and "ep1" in spans


def native_span(native_dir: Path, role: str) -> tuple[int, int]:
    spans = _read_json(native_dir / "capture_manifest.json").get("stage_spans", [])
    for item in spans:
        if item.get("role") == role:
            return int(item["start_frame_idx"]), int(item["end_frame_idx_exclusive"])
    raise KeyError(f"no span for {role} in {native_dir}")


def load_native_role_frames(
    native_dir: Path,
    role: str,
    start: EpisodeStart,
    *,
    stride: int = 1,
    load_rgb: bool = True,
) -> list[LocalFrame]:
    start_idx, end_idx = native_span(native_dir, role)
    frames_dir = native_dir
    frames: list[LocalFrame] = []
    stride = max(1, int(stride))
    for record in _iter_jsonl(native_dir / "frames.jsonl"):
        frame_idx = int(record.get("frame_index", record.get("frame_idx", -1)))
        if frame_idx < start_idx or frame_idx >= end_idx:
            continue
        if (frame_idx - start_idx) % stride != 0:
            continue
        world_xyz = np.asarray(record["agent_position"], dtype=np.float64).reshape(3)
        world_yaw = float(record["agent_heading_rad"])
        x, y, z, yaw = world_agent_to_local(world_xyz, world_yaw, start.xyz, start.yaw)
        rgb_rel = record["rgb"]
        depth_rel = record["depth"]
        rgb = _load_rgb(frames_dir / rgb_rel) if load_rgb else np.zeros((8, 8, 3), dtype=np.uint8)
        frames.append(
            LocalFrame(
                frame_idx=frame_idx,
                rgb=rgb,
                depth_m=_load_depth_m(frames_dir / depth_rel),
                x=x,
                y=y,
                z=z,
                yaw=yaw,
                camera_to_local=habitat_camera_to_world([x, y, z], yaw),
                world_xyz=world_xyz,
                world_yaw=world_yaw,
                episode_name=f"{start.episode_id}",
                rgb_path=str(frames_dir / rgb_rel),
            )
        )
    frames.sort(key=lambda frame: frame.frame_idx)
    return frames


def session_inputs(session_dir: Path) -> dict[str, Path | None]:
    routing = session_dir / "routing"
    ep0_routes = sorted(routing.glob("ep0_*.json"))
    ep1_routes = sorted(routing.glob("ep1_*.json"))
    m0_jsonl = session_dir / "stages" / "M0" / "frames" / "confirmed_keyframes.jsonl"
    live_jsonl = session_dir / "live" / "working_memory" / "frames" / "confirmed_keyframes.jsonl"
    m1_jsonl = session_dir / "stages" / "M1" / "frames" / "confirmed_keyframes.jsonl"
    query_jsonl = live_jsonl if live_jsonl.exists() else m1_jsonl
    return {
        "session_dir": session_dir,
        "ep0_route": ep0_routes[0] if ep0_routes else None,
        "ep1_route": ep1_routes[0] if ep1_routes else None,
        "m0_jsonl": m0_jsonl if m0_jsonl.exists() else None,
        "query_jsonl": query_jsonl if query_jsonl.exists() else None,
    }


def _load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _load_depth_m(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path))
    if image.ndim == 3:
        image = image[..., 0]
    if image.dtype == np.uint16:
        return image.astype(np.float32) / DEPTH_SCALE
    depth = image.astype(np.float32)
    if depth.max() <= 1.5:
        depth = depth * 10.0
    return depth


def _to_local_frame(record: dict, frames_dir: Path, start: EpisodeStart) -> LocalFrame:
    rgb_rel = record["rgb"]
    depth_rel = record["depth"]
    world_xyz = np.asarray(record["agent_position"], dtype=np.float64).reshape(3)
    world_yaw = float(record["agent_heading_rad"])
    x, y, z, yaw = world_agent_to_local(world_xyz, world_yaw, start.xyz, start.yaw)
    return LocalFrame(
        frame_idx=int(record["frame_idx"]),
        rgb=_load_rgb(frames_dir / rgb_rel),
        depth_m=_load_depth_m(frames_dir / depth_rel),
        x=x,
        y=y,
        z=z,
        yaw=yaw,
        camera_to_local=habitat_camera_to_world([x, y, z], yaw),
        world_xyz=world_xyz,
        world_yaw=world_yaw,
        episode_name=str(record.get("episode_name", "")),
        rgb_path=str(frames_dir / rgb_rel),
    )


def load_local_frames(jsonl_path: Path, start: EpisodeStart) -> list[LocalFrame]:
    frames_dir = jsonl_path.parent
    frames = []
    for record in _iter_jsonl(jsonl_path):
        if start.episode_id and start.episode_id not in str(record.get("episode_name", "")):
            continue
        frames.append(_to_local_frame(record, frames_dir, start))
    frames.sort(key=lambda frame: frame.frame_idx)
    return frames


def start_spin_frames(frames: Iterable[LocalFrame], radius_m: float = START_RADIUS_M) -> list[LocalFrame]:
    selected = [frame for frame in frames if math.hypot(frame.x, frame.z) <= radius_m]
    if not selected:
        return list(frames)[:1]
    by_heading: dict[int, LocalFrame] = {}
    for frame in selected:
        bucket = int(round(wrap_angle(frame.yaw) / (math.pi / 6.0)))
        by_heading.setdefault(bucket, frame)
    return [by_heading[key] for key in sorted(by_heading)]
