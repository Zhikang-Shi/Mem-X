"""Module-local online lifecycle for semantic scene-graph construction."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .builder import SceneGraphBuilder
from .models import SceneGraphMemory
from .persistence import SceneGraphStore
from .state import BuilderContinuationState, save_builder_state


@dataclass(frozen=True)
class SemanticFrame:
    rgb: Any
    depth: Any
    camera_to_world: np.ndarray
    frame_index: int
    floor_id: int = 0
    trajectory_node_id: int | None = None


class OnlineSemanticSceneGraph:
    """Own only semantic episode state; the outer runtime owns Habitat/envs."""

    def __init__(self, builder: SceneGraphBuilder, *, output_directory: str | Path) -> None:
        self.builder, self.output_directory = builder, Path(output_directory)
        self.runtime_episode_id: str | None = None
        self.source_episode_id: str | None = None
        self.active, self.frames_seen, self.confirmed_keyframes = False, 0, 0
        self._rollback_memory: SceneGraphMemory | None = None
        self._rollback_state: BuilderContinuationState | None = None

    def start_episode(self, runtime_episode_id: str, source_episode_id: str | None = None) -> None:
        if self.active:
            raise RuntimeError("semantic episode is already active")
        if not str(runtime_episode_id).strip():
            raise ValueError("runtime_episode_id must be non-empty")
        self._rollback_memory = copy.deepcopy(self.builder.memory)
        self._rollback_state = BuilderContinuationState.from_builder(self.builder)
        self.runtime_episode_id = str(runtime_episode_id)
        self.source_episode_id = str(source_episode_id or runtime_episode_id)
        self.active, self.frames_seen, self.confirmed_keyframes = True, 0, 0

    def process_frame(self, frame: SemanticFrame) -> Mapping[str, Any]:
        if not self.active or self.runtime_episode_id is None:
            raise RuntimeError("start_episode must be called before process_frame")
        pose = np.asarray(frame.camera_to_world, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("camera_to_world must be a finite 4x4 matrix")
        result = self.builder.process_frame(
            frame.rgb, frame.depth, pose, int(frame.frame_index),
            runtime_episode_id=self.runtime_episode_id,
            source_episode_id=self.source_episode_id,
            floor_id=int(frame.floor_id),
            trajectory_node_id=frame.trajectory_node_id,
        )
        self.frames_seen += 1
        self.confirmed_keyframes += int(bool(result.get("keyframe")))
        return result

    def finish_episode(self, *, save: bool = True) -> SceneGraphMemory:
        if not self.active:
            raise RuntimeError("semantic episode is not active")
        memory = self.builder.finalize()
        if save:
            SceneGraphStore().save(memory, self.output_directory)
            save_builder_state(
                BuilderContinuationState.from_builder(
                    self.builder, frames_seen=self.frames_seen,
                    confirmed_keyframes=self.confirmed_keyframes,
                ),
                self.output_directory / "semantic_builder_state.json",
            )
        self.active = False
        self._rollback_memory, self._rollback_state = None, None
        return memory

    def abort_episode(self) -> None:
        if not self.active:
            raise RuntimeError("semantic episode is not active")
        if self._rollback_memory is not None:
            self.builder.memory = self._rollback_memory
        if self._rollback_state is not None:
            self._rollback_state.restore(self.builder)
        self.active = False
        self._rollback_memory, self._rollback_state = None, None
