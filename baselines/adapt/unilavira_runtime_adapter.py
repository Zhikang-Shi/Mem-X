"""Uni-LaViRA semantic lifecycle adapter.

Confirmed detections are mirrored through SceneGraphBuilder. Navigation
kernels are the shared MemBridge adapter.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SEMANTIC_MODE_ENV = "MEMORY_MODULE_SEMANTIC_RUNTIME_MODE"
NAVIGATION_MODE_ENV = "MEMORY_MODULE_NAVIGATION_RUNTIME_MODE"
FAIL_CLOSED_ENV = "MEMORY_MODULE_ADAPTER_FAIL_CLOSED"


def _enabled(value: str | None, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _mode(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in {"off", "shadow", "authoritative"}:
        raise ValueError(f"{name} must be off, shadow, or authoritative; got {value!r}")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _runtime(owner: Any) -> Any:
    return getattr(owner, "inherit_sg", owner)


def _audit_root(owner: Any) -> Path | None:
    runtime = _runtime(owner)
    pipeline = getattr(runtime, "pipeline", None)
    path = getattr(pipeline, "run_dir", None) or getattr(runtime, "output_root", None)
    return Path(path) if path else None


def _write_audit(owner: Any, section: str, values: Mapping[str, Any]) -> Path | None:
    root = _audit_root(owner)
    if root is None:
        return None
    root.mkdir(parents=True, exist_ok=True)
    path = root / "memory_module_runtime_audit.json"
    payload = {}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
    payload.update(
        schema_version="memory_module_unilavira_runtime_audit_v1",
        adapter_file=str(Path(__file__).resolve()),
    )
    payload.setdefault(section, {}).update(_plain(values))
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path



def _legacy_memory(runtime: Any):
    from membridge.semantic_scene_graph import ObjectNode, SceneGraphMemory

    memory = SceneGraphMemory()
    skipped = 0
    pipeline = getattr(runtime, "pipeline", None)
    for raw_item in getattr(pipeline, "memory_kb", []) or []:
        try:
            raw = _plain(raw_item)
            raw.setdefault("canonical_label", raw.get("label") or raw.get("semantic_tag"))
            raw.setdefault("label_score", 0.0)
            raw.setdefault("floor_id", 0)
            node = ObjectNode.from_dict(raw)
            if not node.node_id or not node.label or not node.feature:
                skipped += 1
                continue
            memory.objects[node.node_id] = node
        except (TypeError, ValueError):
            skipped += 1
    return memory, skipped


def _semantic_stats(runtime: Any) -> dict[str, Any]:
    stats = getattr(runtime, "_memory_module_semantic_stats", None)
    if stats is None:
        stats = {
            "mode": _mode(SEMANTIC_MODE_ENV, "shadow"),
            "perception_backend": "legacy_sam3_dinov3",
            "semantic_memory": "memory_module.semantic_scene_graph",
            "episodes_started": 0,
            "confirmed_frames": 0,
            "mirrored_detections": 0,
            "mirror_errors": 0,
        }
        runtime._memory_module_semantic_stats = stats
    return stats


def _make_semantic_builder(runtime: Any) -> Any:
    from membridge.semantic_scene_graph import SceneGraphBuilder, SceneGraphBuilderConfig

    memory, skipped = _legacy_memory(runtime)
    runtime._memory_module_detections = []
    runtime._memory_module_frame_feature = [1.0, 0.0]
    runtime._memory_module_room_points = []

    def feature_extractor(_rgb):
        return runtime._memory_module_frame_feature

    def detection_provider(_rgb, _depth, _camera_to_world, _frame_index):
        return runtime._memory_module_detections

    def room_locator(position_xyz, _floor_id):
        point = np.asarray(position_xyz, dtype=np.float64).reshape(3)
        candidates = runtime._memory_module_room_points
        if not candidates:
            return None
        distance, room = min(
            (
                (float(np.linalg.norm(point - candidate)), room_id)
                for candidate, room_id in candidates
            ),
            key=lambda item: (item[0], str(item[1])),
        )
        return room if distance <= 1.0e-4 else None

    builder = SceneGraphBuilder(
        feature_extractor,
        detection_provider,
        config=SceneGraphBuilderConfig(keyframe_similarity_threshold=1.0),
        room_locator=room_locator,
        memory=memory,
    )
    runtime._memory_module_semantic_builder = builder
    stats = _semantic_stats(runtime)
    stats.update(imported_objects=len(memory.objects), skipped_legacy_objects=skipped)
    return builder


def _frame_detections(runtime: Any) -> tuple[list[Any], list[tuple[np.ndarray, Any]]]:
    from membridge.semantic_scene_graph import Detection

    pipeline = getattr(runtime, "pipeline", None)
    by_id = {
        str(item.get("node_id")): item
        for item in getattr(pipeline, "memory_kb", []) or []
        if item.get("node_id") is not None
    }
    observations = getattr(pipeline, "_inherit_sg_frame_observations", []) or []
    detections, room_points = [], []
    for observation in observations:
        item = by_id.get(str(observation.get("node_id")))
        if item is None:
            continue
        try:
            feature = _first_not_none(item.get("dinov3_feature"), item.get("feature"))
            position = _first_not_none(
                observation.get("object_position"), item.get("global_pos_3d")
            )
            bbox = _first_not_none(observation.get("bbox"), item.get("bbox"))
            score = min(1.0, max(0.0, float(item.get("label_score", 0.0))))
            detection = Detection(
                label=str(item.get("label") or item.get("semantic_tag") or "object"),
                score=score,
                feature=_plain(feature),
                position_xyz=_plain(position),
                bbox_xyxy=_plain(bbox),
                rgb_path=str(observation.get("rgb_path") or item.get("full_image_path") or ""),
                crop_path=item.get("cropped_image_path"),
                detector_id=str(item.get("node_id")),
            )
        except (TypeError, ValueError):
            continue
        detections.append(detection)
        room = item.get("parent_room")
        if detection.position_xyz is not None and room is not None:
            room_points.append((np.asarray(detection.position_xyz), room))
    return detections, room_points


def _rewrite_query_snapshot(runtime: Any, memory: Any) -> int:
    snapshot = getattr(runtime, "_query_snapshot", None)
    if not isinstance(snapshot, dict):
        return 0
    updated = 0
    for raw in snapshot.get("objects") or []:
        node = memory.objects.get(str(raw.get("node_id") or ""))
        if node is None:
            continue
        raw.update(
            label=node.label,
            global_pos_3d=_plain(node.position_xyz),
            floor_id=int(node.floor_id),
            parent_room=node.parent_room,
            memory_module_semantic_source="SceneGraphBuilder",
        )
        updated += 1
    snapshot["object_terms"] = sorted(
        {str(label) for item in snapshot.get("objects") or [] for label in item.get("retrieval_labels") or [item.get("label", "")] if label}
    )
    snapshot["memory_module_semantic_authoritative"] = True
    return updated


def install_semantic_runtime_adapter(adapter_class: type) -> dict[str, str]:
    """Mirror confirmed legacy detections through the new semantic graph core."""

    if getattr(adapter_class, "_memory_module_semantic_adapter_installed", False):
        return {}
    original_start = adapter_class.start_episode
    original_process = adapter_class._process_frame_payload
    original_refresh = adapter_class._refresh_query_snapshot

    def start_episode(self, *args, **kwargs):
        result = original_start(self, *args, **kwargs)
        if _mode(SEMANTIC_MODE_ENV, "shadow") != "off" and getattr(self, "pipeline", None) is not None:
            builder = _make_semantic_builder(self)
            stats = _semantic_stats(self)
            stats["episodes_started"] += 1
            if stats["mode"] == "authoritative" and isinstance(
                getattr(self, "_query_snapshot", None), dict
            ):
                stats["authoritative_preload_snapshot_objects"] = (
                    _rewrite_query_snapshot(self, builder.finalize())
                )
            _write_audit(self, "semantic_scene_graph", stats)
        return result

    def process_frame(self, frame):
        result = original_process(self, frame)
        builder = getattr(self, "_memory_module_semantic_builder", None)
        if builder is None or _mode(SEMANTIC_MODE_ENV, "shadow") == "off":
            return result
        stats = _semantic_stats(self)
        try:
            detections, room_points = _frame_detections(self)
            self._memory_module_detections = detections
            self._memory_module_room_points = room_points
            self._memory_module_frame_feature = [1.0, float(int(frame["idx"]) + 1)]
            floor_id = 0
            if detections:
                floor_id = int(getattr(getattr(self.pipeline, "floor_tracker", None), "current_floor_id", 0))
            builder.process_frame(
                frame["rgb_bgr"], frame["depth_mm"], frame["pose"], int(frame["idx"]),
                runtime_episode_id=str(getattr(self, "_current_runtime_episode_id", "")),
                source_episode_id=str(getattr(self, "_current_source_episode_id", "")),
                floor_id=floor_id,
            )
            stats["confirmed_frames"] += 1
            stats["mirrored_detections"] += len(detections)
        except Exception as exc:
            stats["mirror_errors"] += 1
            stats["last_error"] = f"{type(exc).__name__}: {exc}"
            if _enabled(os.environ.get(FAIL_CLOSED_ENV), True):
                raise
        return result

    def refresh_snapshot(self):
        result = original_refresh(self)
        builder = getattr(self, "_memory_module_semantic_builder", None)
        if builder is None or _mode(SEMANTIC_MODE_ENV, "shadow") == "off":
            return result
        memory = builder.finalize()
        from membridge.semantic_scene_graph import SceneGraphStore

        output = _audit_root(self) / "memory_module_semantic_graph"
        SceneGraphStore().save(memory, output)
        self._memory_module_semantic_memory = memory
        stats = _semantic_stats(self)
        stats.update(
            final_objects=len(memory.objects), final_rooms=len(memory.rooms),
            final_areas=len(memory.areas), artifact_dir=str(output),
        )
        if stats["mode"] == "authoritative":
            stats["authoritative_snapshot_objects"] = _rewrite_query_snapshot(self, memory)
        _write_audit(self, "semantic_scene_graph", stats)
        return result

    adapter_class.start_episode = start_episode
    adapter_class._process_frame_payload = process_frame
    adapter_class._refresh_query_snapshot = refresh_snapshot
    adapter_class._memory_module_semantic_adapter_installed = True
    return {"OnlineINHeritSGBuilder.semantic_lifecycle": "memory_module.unilavira_runtime_adapter"}



def install_unilavira_runtime_adapters(
    adapter_class: type, evaluator_class: type, habitat_actions: Any
) -> dict[str, str]:
    installed = {}
    installed.update(install_semantic_runtime_adapter(adapter_class))
    from membridge.habitat_runtime.navigation_adapter import install_navigation_runtime_adapter
    installed.update(install_navigation_runtime_adapter(evaluator_class, habitat_actions))
    return installed
