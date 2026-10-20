"""Small orchestration layer for incremental semantic scene-graph mapping."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .association import AssociationConfig, V7ObjectAssociator
from .hierarchy import build_hierarchy
from .keyframes import KeyframeSelector
from .labels import initialize_label, normalize_label, record_label
from .models import Detection, ObjectNode, SceneGraphMemory, SpatialRelation
from .protocols import (
    DescriptionProvider,
    DetectionProvider,
    FrameFeatureExtractor,
    RelationReasoner,
    RoomLocator,
)
from .provenance import (
    ensure_provenance,
    observation_selection_score,
    provenance_payload,
    record_observation,
)


@dataclass(frozen=True)
class SceneGraphBuilderConfig:
    keyframe_similarity_threshold: float = 0.85
    position_old_weight: float = 0.70
    feature_new_weight: float = 0.80
    spatial_cluster_distance_m: float = 1.0
    area_cluster_distance_m: float = 0.8
    observation_center_weight: float = 0.4
    observation_area_weight: float = 0.6


class SceneGraphBuilder:
    """Compose injected perception with the reviewed memory update policy."""

    def __init__(
        self,
        frame_feature_extractor: FrameFeatureExtractor,
        detection_provider: DetectionProvider,
        *,
        config: SceneGraphBuilderConfig | None = None,
        association_config: AssociationConfig | None = None,
        room_locator: RoomLocator | None = None,
        relation_reasoner: RelationReasoner | None = None,
        description_provider: DescriptionProvider | None = None,
        memory: SceneGraphMemory | None = None,
        text_similarity=None,
    ) -> None:
        self.config = config or SceneGraphBuilderConfig()
        self.frame_feature_extractor = frame_feature_extractor
        self.detection_provider = detection_provider
        self.room_locator = room_locator
        self.relation_reasoner = relation_reasoner
        self.description_provider = description_provider
        self.memory = memory or SceneGraphMemory()
        self.keyframes = KeyframeSelector(self.config.keyframe_similarity_threshold)
        self.associator = V7ObjectAssociator(association_config, text_similarity)
        self.label_counters = self._restore_label_counters()

    def _restore_label_counters(self) -> dict[str, int]:
        counters: dict[str, int] = {}
        for node in self.memory.objects.values():
            label = normalize_label(node.label)
            suffix = node.node_id.rsplit("_", 1)[-1]
            next_value = int(suffix) + 1 if suffix.isdigit() else 1
            counters[label] = max(counters.get(label, 0), next_value)
        return counters

    def _new_node_id(self, label: str) -> str:
        count = self.label_counters.get(label, 0)
        self.label_counters[label] = count + 1
        return f"{label}_{count}"

    @staticmethod
    def _image_size(rgb: Any) -> tuple[int, int]:
        shape = getattr(rgb, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[1]), int(shape[0])
        size = getattr(rgb, "size", None)
        if isinstance(size, tuple) and len(size) == 2:
            return int(size[0]), int(size[1])
        return 1, 1

    @staticmethod
    def _agent_position(camera_to_world: Any) -> list[float] | None:
        try:
            pose = np.asarray(camera_to_world, dtype=np.float64).reshape(4, 4)
        except (TypeError, ValueError):
            return None
        return pose[:3, 3].tolist() if np.isfinite(pose).all() else None

    def process_frame(
        self,
        rgb: Any,
        depth: Any,
        camera_to_world: Any,
        frame_index: int,
        *,
        runtime_episode_id: str,
        source_episode_id: str | None = None,
        floor_id: int = 0,
        trajectory_node_id: int | None = None,
    ) -> dict[str, Any]:
        frame_feature = self.frame_feature_extractor(rgb)
        accepted, similarity = self.keyframes.accept(frame_feature)
        if not accepted:
            return {
                "frame_index": int(frame_index),
                "keyframe": False,
                "keyframe_similarity": similarity,
                "objects": [],
            }
        detections = list(
            self.detection_provider(rgb, depth, camera_to_world, int(frame_index))
        )
        matches = {
            item.detection_index: item
            for item in self.associator.associate(detections, self.memory.objects)
        }
        width, height = self._image_size(rgb)
        agent_position = self._agent_position(camera_to_world)
        touched: list[ObjectNode] = []
        audit = []
        for index, detection in enumerate(detections):
            match = matches.get(index)
            if match is None:
                node = self._create_node(detection, frame_index, floor_id)
                decision = "created"
            else:
                node = self.memory.objects[match.node_id]
                self._merge_node(node, detection, frame_index)
                decision = f"merged_stage_{match.stage}"
            if detection.position_xyz is not None and self.room_locator is not None:
                node.parent_room = self.room_locator(detection.position_xyz, floor_id)
            self._record_provenance(
                node,
                detection,
                frame_index,
                runtime_episode_id,
                source_episode_id,
                agent_position,
                trajectory_node_id,
                width,
                height,
            )
            touched.append(node)
            audit.append(
                {
                    "detector_id": detection.detector_id,
                    "node_id": node.node_id,
                    "decision": decision,
                    "association": copy.deepcopy(match.__dict__) if match else None,
                }
            )
        self._update_relations(touched, {"frame_index": frame_index, "rgb": rgb})
        return {
            "frame_index": int(frame_index),
            "keyframe": True,
            "keyframe_similarity": similarity,
            "objects": audit,
        }

    def _create_node(self, detection: Detection, frame_index: int, floor_id: int) -> ObjectNode:
        label = normalize_label(detection.label)
        payload = {"node_id": self._new_node_id(label), "label": label}
        initialize_label(payload, label, detection.score, frame_index)
        ensure_provenance(payload)
        node = ObjectNode(
            node_id=str(payload["node_id"]),
            label=str(payload["label"]),
            label_score=float(payload["label_score"]),
            feature=list(map(float, detection.feature)),
            position_xyz=(
                list(map(float, detection.position_xyz))
                if detection.position_xyz is not None
                else None
            ),
            floor_id=int(floor_id),
            description=(
                self.description_provider(detection)
                if self.description_provider is not None
                else ""
            ),
            bbox=list(map(float, detection.bbox_xyxy)),
            full_image_path=detection.rgb_path,
            crop_path=detection.crop_path,
            label_evidence=copy.deepcopy(payload["label_evidence"]),
            metadata=provenance_payload(payload),
        )
        self.memory.objects[node.node_id] = node
        return node

    def _merge_node(self, node: ObjectNode, detection: Detection, frame_index: int) -> None:
        if detection.position_xyz is not None:
            current = np.asarray(detection.position_xyz, dtype=np.float64)
            if node.position_xyz is None:
                node.position_xyz = current.tolist()
            else:
                old = np.asarray(node.position_xyz, dtype=np.float64)
                weight = float(self.config.position_old_weight)
                node.position_xyz = (weight * old + (1.0 - weight) * current).tolist()
        old_feature = np.asarray(node.feature, dtype=np.float64)
        new_feature = np.asarray(detection.feature, dtype=np.float64)
        if old_feature.shape == new_feature.shape:
            weight = float(self.config.feature_new_weight)
            node.feature = (weight * new_feature + (1.0 - weight) * old_feature).tolist()
        payload = node.to_dict()
        record_label(payload, detection.label, detection.score, frame_index)
        node.label = str(payload["label"])
        node.label_score = float(payload["label_score"])
        node.label_evidence = copy.deepcopy(payload["label_evidence"])

    def _record_provenance(
        self,
        node: ObjectNode,
        detection: Detection,
        frame_index: int,
        runtime_episode_id: str,
        source_episode_id: str | None,
        agent_position: list[float] | None,
        trajectory_node_id: int | None,
        width: int,
        height: int,
    ) -> None:
        score = observation_selection_score(
            detection.bbox_xyxy,
            width,
            height,
            center_weight=self.config.observation_center_weight,
            area_weight=self.config.observation_area_weight,
        )
        observation = {
            "frame_idx": int(frame_index),
            "rgb_path": detection.rgb_path,
            "bbox_xyxy": list(map(float, detection.bbox_xyxy)),
            "agent_position": copy.deepcopy(agent_position),
            "trajectory_node_id": trajectory_node_id,
            "crop_path": detection.crop_path,
        }
        waypoint = {
            "position": copy.deepcopy(agent_position),
            "frame_idx": int(frame_index),
            "trajectory_node_id": trajectory_node_id,
        }
        payload = node.to_dict()
        update = record_observation(
            payload,
            runtime_episode_id,
            source_episode_id,
            observation,
            waypoint,
            score,
        )
        node.metadata.update(provenance_payload(payload))
        if update["global_updated"]:
            node.bbox = list(map(float, detection.bbox_xyxy))
            node.full_image_path = detection.rgb_path
            node.crop_path = detection.crop_path

    def _update_relations(
        self, touched: Sequence[ObjectNode], context: Mapping[str, Any]
    ) -> None:
        if len(touched) < 2:
            return
        if self.relation_reasoner is not None:
            relations = self.relation_reasoner(touched, context)
        else:
            relations = self._near_relations(touched)
        for relation in relations:
            if relation.source in self.memory.objects and relation.target in self.memory.objects:
                self.memory.add_relation(relation)

    def _near_relations(self, nodes: Sequence[ObjectNode]) -> list[SpatialRelation]:
        result = []
        for index, left in enumerate(nodes):
            if left.position_xyz is None:
                continue
            for right in nodes[index + 1 :]:
                if right.position_xyz is None:
                    continue
                distance = float(
                    np.linalg.norm(
                        np.asarray(left.position_xyz)[[0, 2]]
                        - np.asarray(right.position_xyz)[[0, 2]]
                    )
                )
                if distance <= self.config.spatial_cluster_distance_m:
                    weight = max(0.0, 1.0 - distance / self.config.spatial_cluster_distance_m)
                    result.extend(
                        [
                            SpatialRelation(left.node_id, right.node_id, "near", weight),
                            SpatialRelation(right.node_id, left.node_id, "near", weight),
                        ]
                    )
        return result

    def finalize(self) -> SceneGraphMemory:
        build_hierarchy(self.memory, self.config.area_cluster_distance_m)
        return self.memory
