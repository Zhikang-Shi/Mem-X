"""Versioned persistence for sparse occupancy memory."""

from __future__ import annotations

import os
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .grid import WorldGridSpec
from .fixed_map import FixedMapConfig, FixedSemanticMap
from .sparse_map import MetricChunk, SparseWorldMetricMap

WORLD_MAP_SCHEMA_VERSION = 1
COORDINATE_FRAME = "habitat_world"
FIXED_MAP_SCHEMA_VERSION = 1


def save_fixed_map(memory: FixedSemanticMap, path: str | os.PathLike[str]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, schema_version=FIXED_MAP_SCHEMA_VERSION,
            config_json=json.dumps(asdict(memory.config), sort_keys=True),
            log_odds=memory.log_odds, door_hits=memory.door_hits,
            door=memory.door, room_ids=memory.room_ids,
        )
    os.replace(temporary, destination)
    return destination


def load_fixed_map(path: str | os.PathLike[str]) -> FixedSemanticMap:
    with np.load(path, allow_pickle=False) as data:
        if int(data["schema_version"]) != FIXED_MAP_SCHEMA_VERSION:
            raise ValueError("unsupported fixed semantic-map schema")
        result = FixedSemanticMap(FixedMapConfig(**json.loads(str(data["config_json"]))))
        arrays = {
            "log_odds": np.asarray(data["log_odds"], dtype=np.float32),
            "door_hits": np.asarray(data["door_hits"], dtype=np.float32),
            "door": np.asarray(data["door"], dtype=bool),
            "room_ids": np.asarray(data["room_ids"], dtype=np.int32),
        }
    if any(value.shape != result.log_odds.shape for value in arrays.values()):
        raise ValueError("fixed semantic-map arrays do not match configured shape")
    for name, value in arrays.items():
        setattr(result, name, value.copy())
    return result


def save_world_map(
    memory: SparseWorldMetricMap,
    path: str | os.PathLike[str],
    *,
    scene_id: str,
    source_episode_id: str,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(memory._chunks)
    size = memory.spec.chunk_size
    chunk_keys = np.asarray(keys, dtype=np.int32).reshape(-1, 3)
    log_odds = (
        np.stack([memory._chunks[key].log_odds for key in keys])
        if keys else np.zeros((0, size, size), dtype=np.float32)
    )
    counts = (
        np.stack([memory._chunks[key].observed_count for key in keys])
        if keys else np.zeros((0, size, size), dtype=np.uint16)
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(WORLD_MAP_SCHEMA_VERSION, dtype=np.int32),
            coordinate_frame=np.asarray(COORDINATE_FRAME),
            scene_id=np.asarray(str(scene_id)),
            source_episode_id=np.asarray(str(source_episode_id)),
            resolution_m=np.asarray(memory.spec.resolution_m),
            chunk_size=np.asarray(size, dtype=np.int32),
            chunk_keys=chunk_keys,
            log_odds=log_odds,
            observed_count=counts,
            log_odds_occupied=np.asarray(memory.log_odds_occupied),
            log_odds_free=np.asarray(memory.log_odds_free),
            log_odds_min=np.asarray(memory.log_odds_min),
            log_odds_max=np.asarray(memory.log_odds_max),
            occupied_threshold=np.asarray(memory.occupied_threshold),
            free_threshold=np.asarray(memory.free_threshold),
        )
    os.replace(temporary, destination)
    return destination


def load_world_map(path: str | os.PathLike[str]) -> SparseWorldMetricMap:
    with np.load(path, allow_pickle=False) as data:
        version = int(np.asarray(data["schema_version"]).item())
        frame = str(np.asarray(data["coordinate_frame"]).item())
        if version != WORLD_MAP_SCHEMA_VERSION or frame != COORDINATE_FRAME:
            raise ValueError(f"unsupported world map schema/frame: {version}/{frame}")
        spec = WorldGridSpec(float(data["resolution_m"]), int(data["chunk_size"]))
        memory = SparseWorldMetricMap(
            spec,
            log_odds_occupied=float(data["log_odds_occupied"]),
            log_odds_free=float(data["log_odds_free"]),
            log_odds_min=float(data["log_odds_min"]),
            log_odds_max=float(data["log_odds_max"]),
            occupied_threshold=float(data["occupied_threshold"]),
            free_threshold=float(data["free_threshold"]),
        )
        keys = np.asarray(data["chunk_keys"], dtype=np.int32)
        log_odds = np.asarray(data["log_odds"], dtype=np.float32)
        counts = np.asarray(data["observed_count"], dtype=np.uint16)
    expected = (keys.shape[0], spec.chunk_size, spec.chunk_size)
    if keys.ndim != 2 or keys.shape[1:] != (3,):
        raise ValueError(f"invalid chunk key shape: {keys.shape}")
    if log_odds.shape != expected or counts.shape != expected:
        raise ValueError("chunk arrays do not match the world-map schema")
    for index, raw_key in enumerate(keys):
        key = tuple(map(int, raw_key))
        if key in memory._chunks:
            raise ValueError(f"duplicate chunk key: {key}")
        memory._chunks[key] = MetricChunk(log_odds[index].copy(), counts[index].copy())
    return memory
