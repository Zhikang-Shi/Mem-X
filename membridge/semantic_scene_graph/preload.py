"""Load native or trusted legacy semantic memories without importing SG."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .hierarchy import build_hierarchy
from .models import ObjectNode, SceneGraphMemory
from .persistence import SceneGraphStore
from .state import BuilderContinuationState, load_builder_state, save_builder_state


@dataclass(frozen=True)
class SemanticPreload:
    memory: SceneGraphMemory
    source_directory: Path
    source_format: str
    legacy_builder_state: Mapping[str, Any] | None = field(default=None, repr=False)
    builder_state: BuilderContinuationState | None = None
    skipped_objects: int = 0

    def restore(self, builder: Any) -> None:
        """Install the preloaded memory and native continuation state."""

        builder.memory = self.memory
        if self.builder_state is not None:
            self.builder_state.restore(builder)
        else:
            builder.label_counters = builder._restore_label_counters()


def _legacy_memory(path: Path) -> tuple[SceneGraphMemory, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("legacy memory_kb.json must contain a list")
    memory, skipped = SceneGraphMemory(), 0
    for item in raw:
        try:
            node = ObjectNode.from_dict(item)
            if not node.node_id or not node.label or not node.feature:
                skipped += 1
                continue
            memory.objects[node.node_id] = node
        except (TypeError, ValueError):
            skipped += 1
    build_hierarchy(memory)
    return memory, skipped


def load_legacy_builder_state(path: str | Path, *, trusted: bool = False) -> Mapping[str, Any]:
    """Load old pickle state only when the caller explicitly marks it trusted."""

    source = Path(path)
    if not trusted:
        raise ValueError("legacy builder_state.pkl requires trusted=True")
    with source.open("rb") as handle:
        state = pickle.load(handle)
    if not isinstance(state, dict):
        raise ValueError("legacy builder state must be a mapping")
    if int(state.get("schema_version", -1)) != 1:
        raise ValueError("unsupported legacy builder-state schema")
    if state.get("state_kind") != "inherit_sg_streaming_continuation":
        raise ValueError("unexpected legacy builder-state kind")
    required = {"pipeline", "map_manager", "room_segmenter", "floor_tracker"}
    if not required.issubset(state):
        raise ValueError(f"legacy builder state is missing {sorted(required.difference(state))}")
    if any(not isinstance(state[key], dict) for key in required):
        raise ValueError("legacy builder-state sections must be mappings")
    return state


def load_semantic_preload(
    directory: str | Path, *, trusted_legacy_pickle: bool = False
) -> SemanticPreload:
    root = Path(directory)
    memory_path = root / "memory_kb.json"
    if not memory_path.is_file():
        raise FileNotFoundError(f"missing semantic memory: {memory_path}")
    native_manifest = root / "memory_module_semantic_manifest.json"
    if native_manifest.is_file():
        try:
            memory = SceneGraphStore().load(root)
            source_format, skipped = "memory_module_scene_graph_v1", 0
        except (AttributeError, KeyError, TypeError, ValueError):
            memory, skipped = _legacy_memory(memory_path)
            source_format = "legacy_inherit_sg_memory_kb"
    else:
        memory, skipped = _legacy_memory(memory_path)
        source_format = "legacy_inherit_sg_memory_kb"
    state_path = root / "builder_state.pkl"
    legacy_state = (
        load_legacy_builder_state(state_path, trusted=True)
        if state_path.is_file() and trusted_legacy_pickle else None
    )
    native_state_path = root / "semantic_builder_state.json"
    native_state = (
        load_builder_state(native_state_path)
        if native_state_path.is_file() else BuilderContinuationState.from_memory(memory)
    )
    return SemanticPreload(
        memory, root.resolve(), source_format, legacy_state, native_state, skipped
    )


def convert_legacy_semantic_archive(
    source_directory: str | Path, destination_directory: str | Path, *,
    trusted_legacy_pickle: bool = False,
) -> SemanticPreload:
    """Convert JSON memory and optional trusted state into native artifacts."""

    source, destination = Path(source_directory), Path(destination_directory)
    if source.resolve() == destination.resolve():
        raise ValueError("legacy conversion destination must differ from source")
    memory_path = source / "memory_kb.json"
    if not memory_path.is_file():
        raise FileNotFoundError(f"missing semantic memory: {memory_path}")
    memory, skipped = _legacy_memory(memory_path)
    legacy_path = source / "builder_state.pkl"
    legacy_state = None
    if legacy_path.is_file() and trusted_legacy_pickle:
        legacy_state = load_legacy_builder_state(legacy_path, trusted=True)
    SceneGraphStore().save(memory, destination)
    native_state = BuilderContinuationState.from_memory(memory)
    save_builder_state(native_state, destination / "semantic_builder_state.json")
    manifest = {
        "schema_version": 1,
        "conversion_kind": "inherit_sg_to_memory_module",
        "source_directory": str(source.resolve()),
        "object_count": len(memory.objects),
        "skipped_objects": skipped,
        "legacy_builder_state_validated": legacy_state is not None,
    }
    manifest_path = destination / "legacy_conversion_manifest.json"
    temporary = manifest_path.with_name(manifest_path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return SemanticPreload(
        memory, destination.resolve(), "converted_legacy_inherit_sg_v1",
        legacy_state, native_state, skipped,
    )
