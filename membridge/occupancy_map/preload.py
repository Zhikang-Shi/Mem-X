"""Session-level occupancy preload and legacy raster conversion."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .grid import WorldGridSpec
from .fixed_map import FixedMapConfig, FixedSemanticMap
from .persistence import load_fixed_map, load_world_map
from .sparse_map import SparseWorldMetricMap


@dataclass(frozen=True)
class OccupancyPreload:
    memory: SparseWorldMetricMap
    source_path: Path
    source_format: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FixedOccupancyPreload:
    memory: FixedSemanticMap
    source_path: Path
    source_format: str


def _validated_legacy_layer(
    payload: Mapping[str, Any], name: str, shape: tuple[int, int], dtype: Any,
    *, required: bool = True,
) -> np.ndarray:
    value = payload.get(name)
    if value is None:
        if required:
            raise ValueError(f"legacy fixed map is missing {name}")
        return np.zeros(shape, dtype=dtype)
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"legacy fixed-map layer {name} has shape {result.shape}, expected {shape}")
    return result


def load_session_fixed_map(
    directory: str | Path, *, trusted_legacy_pickle: bool = False,
) -> FixedOccupancyPreload:
    """Load a native semantic map or explicitly trusted legacy ``map_data.pkl``."""

    root = Path(directory)
    native = root / "fixed_semantic_map.npz"
    if native.is_file():
        return FixedOccupancyPreload(
            load_fixed_map(native), native.resolve(), "memory_module_fixed_semantic_map_v1"
        )
    legacy = root / "map_data.pkl"
    if not legacy.is_file():
        raise FileNotFoundError(f"no fixed semantic-map preload found under {root}")
    if not trusted_legacy_pickle:
        raise ValueError("legacy map_data.pkl requires trusted_legacy_pickle=True")
    with legacy.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("legacy map_data.pkl must contain a mapping")
    room_ids = np.asarray(payload.get("room_mask"), dtype=np.int32)
    if room_ids.ndim != 2 or not room_ids.size:
        raise ValueError("legacy room_mask must be a non-empty 2D array")
    height, width = room_ids.shape
    resolution = float(payload.get("resolution", 0.05))
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("legacy fixed-map resolution must be finite and positive")
    memory = FixedSemanticMap(FixedMapConfig(
        resolution_m=resolution, width=width, height=height,
        min_x=float(payload.get("min_x", -width * resolution / 2.0)),
        min_z=float(payload.get("min_z", -height * resolution / 2.0)),
    ))
    occupied = _validated_legacy_layer(payload, "occ_bin", room_ids.shape, bool)
    free = _validated_legacy_layer(payload, "free_bin", room_ids.shape, bool)
    door = _validated_legacy_layer(payload, "door_map", room_ids.shape, bool, required=False)
    memory.log_odds[free & ~occupied] = memory.config.log_odds_min
    memory.log_odds[occupied] = memory.config.log_odds_max
    memory.door = door.copy()
    memory.door_hits[door] = 6.0
    memory.room_ids = room_ids.copy()
    return FixedOccupancyPreload(
        memory, legacy.resolve(), "legacy_inherit_sg_map_data"
    )


def load_session_occupancy(
    directory: str | Path, *, spec: WorldGridSpec | None = None
) -> OccupancyPreload:
    root = Path(directory)
    native = root / "world_metric_map.npz"
    if native.is_file():
        memory = load_world_map(native)
        return OccupancyPreload(
            memory, native.resolve(), "memory_module_sparse_world_map_v1",
            {"chunk_count": memory.chunk_count, "resolution_m": memory.spec.resolution_m},
        )
    legacy = root / "lavira_observed_free_memory.npz"
    if not legacy.is_file():
        raise FileNotFoundError(f"no occupancy preload found under {root}")
    with np.load(legacy, allow_pickle=False) as data:
        required = {"observed_free", "observed_obstacle", "world_to_map", "resolution_cm"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"legacy navigation raster is missing {sorted(missing)}")
        free = np.asarray(data["observed_free"], dtype=bool)
        occupied = np.asarray(data["observed_obstacle"], dtype=bool)
        transform = np.asarray(data["world_to_map"], dtype=np.float64)
        resolution_m = float(np.asarray(data["resolution_cm"]).item()) / 100.0
        source_episode = str(np.asarray(data["source_episode_id"]).item()) if "source_episode_id" in data else ""
    memory = SparseWorldMetricMap(spec or WorldGridSpec(resolution_m=resolution_m))
    stats = memory.fuse_navigation_raster(
        observed_free=free, observed_obstacle=occupied,
        world_to_map=transform, source_resolution_m=resolution_m,
    )
    return OccupancyPreload(
        memory, legacy.resolve(), "legacy_lavira_observed_free_raster",
        {**stats, "source_episode_id": source_episode, "source_resolution_m": resolution_m},
    )
