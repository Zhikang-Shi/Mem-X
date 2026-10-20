import copy
import csv
import datetime
import hashlib
import json
import os
import pickle
import queue
import shutil
import stat
import sys
import threading
import traceback
import types
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np

from .world_metric_memory import (
    SparseWorldMetricMap,
    TrajectoryGraph,
    WorldGridSpec,
    WorldMemoryRecorder,
    habitat_camera_to_world,
)

try:
    from membridge.semantic_scene_graph.legacy_builder_source.object_provenance import (
        bbox_pixel_area,
        ensure_object_provenance,
        observation_selection_score,
        ordered_unique_episode_ids,
        record_object_observation,
        sync_object_provenance_to_graph,
    )
except ImportError:
    _provenance_source_dir = os.environ.get("MEMORY_MODULE_SEMANTIC_SOURCE_DIR", "").strip()
    if _provenance_source_dir and _provenance_source_dir not in sys.path:
        sys.path.insert(0, _provenance_source_dir)
    from object_provenance import (
        bbox_pixel_area,
        ensure_object_provenance,
        observation_selection_score,
        ordered_unique_episode_ids,
        record_object_observation,
        sync_object_provenance_to_graph,
    )

V8_SAM3_CATEGORY_BATCH_SIZE = 10
V8_QUERY_LABEL_COUNT = 79
V8_QUERY_VOCABULARY_VERSION = "level1_combined_v3_sg79_v1"
V8_QUERY_LABELS = (
    "armchair",
    "bag",
    "basket",
    "bathtub",
    "bed",
    "bench",
    "bicycle",
    "blanket",
    "board",
    "bottle",
    "bowl",
    "box",
    "bucket",
    "cabinet",
    "calendar",
    "chair",
    "chest",
    "christmas tree",
    "clock",
    "coffee machine",
    "computer",
    "counter",
    "curtain",
    "desk",
    "dishwasher",
    "dresser",
    "dressing table",
    "easel",
    "exercise equipment",
    "fan",
    "fire extinguisher",
    "fireplace",
    "flower stand",
    "food stand",
    "globe stand",
    "grill",
    "guitar",
    "heater",
    "ironing board",
    "island",
    "jug",
    "kettle",
    "microwave",
    "mirror",
    "monitor",
    "nightstand",
    "ornament",
    "ottoman",
    "piano",
    "picture",
    "plant",
    "printer",
    "radio",
    "refrigerator",
    "rocking chair",
    "scale",
    "sculpture",
    "sewing machine",
    "shelf",
    "shower",
    "shower seat",
    "sink",
    "sofa",
    "speaker",
    "stove",
    "table",
    "table tennis table",
    "telephone",
    "toilet",
    "towel",
    "toy",
    "trash can",
    "tray",
    "tv",
    "tv stand",
    "vacuum cleaner",
    "vase",
    "wardrobe",
    "washing machine",
)
LEVEL3_DUAL_ANCHOR_QUERY_VOCABULARY_VERSION = (
    "level1_combined_v3_sg75_chair_storage_v1"
)
# Level-3 dual-anchor evaluation intentionally uses the reviewed 75-label
# vocabulary.  The four removed labels remain query aliases in the JSON
# contract and therefore do not create four additional SAM3 prompts.
LEVEL3_DUAL_ANCHOR_QUERY_LABELS = tuple(
    label
    for label in V8_QUERY_LABELS
    if label not in {"armchair", "chest", "dressing table", "rocking chair"}
)
LEGACY_V8_QUERY_VOCABULARY_VERSION = (
    "level1_v7_level3a_global_anchor_union_v3"
)
LEGACY_V8_QUERY_LABELS = (
    "armchair",
    "bag",
    "bathtub",
    "bed",
    "bench",
    "blanket",
    "board",
    "bowl of sweets",
    "box",
    "bucket",
    "cabinet",
    "clock",
    "coffee machine",
    "counter",
    "curtain",
    "desk",
    "dishwasher",
    "dresser",
    "exercise bike",
    "fan",
    "fire extinguisher",
    "fireplace",
    "flower stand",
    "island",
    "kettle",
    "microwave",
    "nightstand",
    "picture",
    "plant",
    "refrigerator",
    "sculpture",
    "shelf",
    "shower seat",
    "sink",
    "sofa",
    "speaker",
    "stool",
    "stove",
    "table tennis table",
    "telephone",
    "toilet",
    "towel",
    "toy",
    "trash can",
    "tray",
    "tv",
    "tv stand",
    "vacuum cleaner",
    "washing machine",
)
LEVEL3_QUERY_VOCABULARY_VERSION = "level3b_e012_release_union_v1"
LEVEL3_QUERY_LABELS = (
    "armchair",
    "bag",
    "bathtub",
    "bed",
    "bench",
    "blanket",
    "board",
    "bowl of sweets",
    "box",
    "bucket",
    "cabinet",
    "clock",
    "coffee machine",
    "counter",
    "curtain",
    "desk",
    "dishwasher",
    "dresser",
    "exercise bike",
    "fan",
    "fire extinguisher",
    "fireplace",
    "flower stand",
    "island",
    "kettle",
    "microwave",
    "nightstand",
    "picture",
    "plant",
    "refrigerator",
    "rod",
    "sculpture",
    "shelf",
    "shower seat",
    "sink",
    "sofa",
    "speaker",
    "stool",
    "stove",
    "table",
    "table tennis table",
    "telephone",
    "toilet",
    "towel",
    "toy",
    "trash can",
    "tray",
    "tv",
    "tv stand",
    "vacuum cleaner",
    "wardrobe",
    "washing machine",
)
SUPPORTED_QUERY_VOCABULARIES = {
    V8_QUERY_VOCABULARY_VERSION: V8_QUERY_LABELS,
    LEVEL3_DUAL_ANCHOR_QUERY_VOCABULARY_VERSION: (
        LEVEL3_DUAL_ANCHOR_QUERY_LABELS
    ),
    LEGACY_V8_QUERY_VOCABULARY_VERSION: LEGACY_V8_QUERY_LABELS,
    LEVEL3_QUERY_VOCABULARY_VERSION: LEVEL3_QUERY_LABELS,
}

try:
    from habitat import logger
except ModuleNotFoundError:
    import logging

    logger = logging.getLogger("inherit_sg")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


ROOM_TYPE_RULES = {
    "kitchen": {
        "refrigerator": 4.0,
        "fridge": 4.0,
        "microwave": 3.5,
        "stove": 4.0,
        "oven": 3.5,
        "dishwasher": 3.5,
        "kettle": 2.5,
        "toaster": 2.5,
        "coffee machine": 2.5,
        "coffee maker": 2.5,
        "sink": 1.0,
        "cabinet": 1.2,
        "shelf": 1.0,
        "counter": 2.0,
        "cup": 0.8,
        "mug": 0.8,
        "bowl": 0.8,
        "plate": 0.8,
        "knife": 0.8,
        "utensil": 0.8,
    },
    "bathroom": {
        "toilet": 5.0,
        "toilet paper": 4.0,
        "bathtub": 4.5,
        "shower": 4.5,
        "towel": 2.8,
        "soap dispenser": 3.0,
        "soap": 2.5,
        "mirror": 1.6,
        "sink": 2.0,
        "cabinet": 0.8,
    },
    "bedroom": {
        "bed": 5.0,
        "pillow": 3.5,
        "blanket": 2.5,
        "nightstand": 4.0,
        "wardrobe": 2.5,
        "dresser": 3.0,
        "clothes": 1.0,
        "shoe": 0.8,
        "lamp": 0.8,
    },
    "living room": {
        "sofa": 4.0,
        "couch": 4.0,
        "tv": 2.8,
        "television": 2.8,
        "coffee table": 3.0,
        "fireplace": 3.0,
        "armchair": 2.0,
        "speaker": 1.6,
        "ottoman": 2.0,
        "piano": 2.0,
        "lamp": 0.8,
        "table": 0.6,
    },
    "dining room": {
        "dining table": 4.5,
        "dining chair": 3.5,
        "table": 1.5,
        "chair": 0.8,
        "plate": 1.2,
        "bowl": 0.8,
        "cup": 0.6,
    },
    "office": {
        "desk": 4.0,
        "office chair": 3.5,
        "monitor": 3.0,
        "computer": 3.0,
        "laptop": 3.0,
        "keyboard": 3.0,
        "mouse": 3.0,
        "printer": 3.0,
        "book": 0.8,
        "shelf": 0.8,
    },
    "laundry room": {
        "washing machine": 5.0,
        "washer": 5.0,
        "dryer": 4.5,
        "laundry basket": 3.5,
        "iron": 2.5,
        "clothes": 1.0,
    },
    "closet": {
        "wardrobe": 3.0,
        "clothes": 3.0,
        "shoe": 2.0,
        "bag": 1.5,
        "backpack": 1.5,
        "luggage": 1.5,
        "box": 1.0,
    },
    "entryway": {
        "umbrella": 2.5,
        "shoe": 1.8,
        "bag": 1.5,
        "backpack": 1.5,
        "coat": 2.0,
        "mat": 1.2,
    },
    "garage": {
        "bicycle": 3.0,
        "tool": 3.0,
        "box": 1.5,
        "luggage": 1.0,
    },
}

ROOM_TYPE_ALIASES = {
    "fridge": "refrigerator",
    "television": "tv",
    "washing machine": "washing machine",
    "washer-dryer": "washing machine dryer",
    "clothes dryer": "dryer",
    "bottle of soap": "soap dispenser",
    "liquid soap": "soap dispenser",
    "bathroom shelf": "bathroom shelf",
    "kitchen shelf": "kitchen shelf",
    "kitchen cabinet": "kitchen cabinet",
    "kitchen counter": "counter",
    "kitchen countertop": "counter",
    "worktop": "counter",
    "oven and stove": "oven stove",
    "led tv": "tv",
}

BUILDER_STATE_NAME = "builder_state.pkl"
BUILDER_STATE_SCHEMA_VERSION = 1


def habitat_cam_pose_matrix(
    obs_dict,
    *,
    sensor_offset_xyz=(0.0, 1.2, 0.0),
) -> np.ndarray:
    """Build the camera pose from the Habitat agent state and sensor extrinsic.

    Habitat's ``position`` observation is the agent/base position, not the
    RGB-D sensor position.  The depth points therefore need a camera-to-world
    transform whose translation includes the configured sensor offset.
    """
    pos = np.asarray(obs_dict["position"], dtype=np.float32).reshape(3)
    heading = np.asarray(obs_dict["heading"], dtype=np.float32).reshape(-1)
    yaw = float(heading[0]) if heading.size else 0.0
    pose = habitat_camera_to_world(
        pos,
        yaw,
        sensor_offset_xyz=sensor_offset_xyz,
    )
    return np.asarray(pose, dtype=np.float32)


class OnlineINHeritSGBuilder:
    """Feed live Uni-LaViRA Habitat frames into INHerit-SG's online builder."""

    def __init__(self, lavira_config, width: int, height: int, save_dir: str, device: str = "cuda:0"):
        self.lavira_config = lavira_config
        self.width = int(width)
        self.height = int(height)
        self.save_dir = save_dir
        self.device = str(device)
        self.enabled = self._env_bool("INHERIT_SG_ENABLE", True)
        self.debug = self._env_bool("INHERIT_SG_DEBUG", False)
        self.stride = max(1, int(os.environ.get("INHERIT_SG_STRIDE", "1")))
        self.use_keyframe_filter = self._env_bool("INHERIT_SG_KEYFRAME_FILTER", True)
        self.share_models = self._env_bool("INHERIT_SG_SHARE_MODELS", True)
        self.async_enabled = self._env_bool("INHERIT_SG_ASYNC", True)
        self.fail_closed = self._env_bool("INHERIT_SG_FAIL_CLOSED", False)
        self.async_queue_size = max(1, int(os.environ.get("INHERIT_SG_ASYNC_QUEUE_SIZE", "256")))
        self.async_drop_policy = os.environ.get("INHERIT_SG_ASYNC_DROP_POLICY", "block").strip().lower()
        self.save_map_snapshots = self._env_bool("INHERIT_SG_SAVE_MAP_SNAPSHOTS", False)
        self.map_snapshot_stride = max(1, int(os.environ.get("INHERIT_SG_MAP_SNAPSHOT_STRIDE", "1")))
        self.save_map_snapshot_npz = self._env_bool("INHERIT_SG_SAVE_MAP_SNAPSHOT_NPZ", False)
        self.save_offline_rgbd = self._env_bool(
            "INHERIT_SG_SAVE_OFFLINE_RGBD", False
        )
        self.save_confirmed_keyframes = self._env_bool(
            "INHERIT_SG_SAVE_CONFIRMED_KEYFRAMES", False
        )
        self.capture_only = self._env_bool("INHERIT_SG_CAPTURE_ONLY", False)
        self.offline_rgbd_dir_template = os.environ.get(
            "INHERIT_SG_OFFLINE_RGBD_DIR", ""
        ).strip()
        self.offline_rgbd_dir = self.offline_rgbd_dir_template
        self.external_frame_root = os.environ.get(
            "INHERIT_SG_EXTERNAL_FRAME_ROOT", ""
        ).strip()
        if self.external_frame_root:
            self.external_frame_root = os.path.realpath(
                self.external_frame_root
            )
            if not os.path.isdir(self.external_frame_root):
                raise FileNotFoundError(
                    "INHERIT_SG_EXTERNAL_FRAME_ROOT does not exist: "
                    f"{self.external_frame_root}"
                )
        if self.save_offline_rgbd and not self.offline_rgbd_dir:
            raise ValueError(
                "INHERIT_SG_OFFLINE_RGBD_DIR is required when "
                "INHERIT_SG_SAVE_OFFLINE_RGBD=1"
            )
        if self.capture_only and not self.enabled:
            raise ValueError(
                "INHERIT_SG_CAPTURE_ONLY=1 requires INHERIT_SG_ENABLE=1"
            )
        if self.capture_only and not self.save_offline_rgbd:
            raise ValueError(
                "INHERIT_SG_CAPTURE_ONLY=1 requires "
                "INHERIT_SG_SAVE_OFFLINE_RGBD=1"
            )
        self._offline_rgbd_frame_indices = []
        self.observation_center_weight = max(
            0.0, float(os.environ.get("INHERIT_SG_OBSERVATION_CENTER_WEIGHT", "0.4"))
        )
        self.observation_area_weight = max(
            0.0, float(os.environ.get("INHERIT_SG_OBSERVATION_AREA_WEIGHT", "0.6"))
        )
        observation_weight_sum = self.observation_center_weight + self.observation_area_weight
        if observation_weight_sum <= 0.0:
            raise ValueError("INHERIT_SG observation-score weights must have a positive sum")
        self.observation_center_weight /= observation_weight_sum
        self.observation_area_weight /= observation_weight_sum
        self.preload_dir = os.environ.get("INHERIT_SG_PRELOAD_DIR", "").strip()
        self.preload_query_only = self._env_bool("INHERIT_SG_PRELOAD_QUERY_ONLY", False)
        self.preload_world_map_only = self._env_bool(
            "INHERIT_SG_PRELOAD_WORLD_MAP_ONLY", False
        )
        if self.preload_world_map_only and not self.preload_query_only:
            raise ValueError(
                "INHERIT_SG_PRELOAD_WORLD_MAP_ONLY=1 requires "
                "INHERIT_SG_PRELOAD_QUERY_ONLY=1"
            )
        self.preloaded_object_match_guard_m = max(
            0.1,
            float(
                os.environ.get(
                    "INHERIT_SG_PRELOADED_OBJECT_MATCH_GUARD_M",
                    "1.0",
                )
            ),
        )
        self.world_memory_enabled = self._env_bool("LIFELONG_WORLD_MEMORY_ENABLE", True)
        self.world_memory_required = self._env_bool("LIFELONG_WORLD_MEMORY_REQUIRED", True)
        self.world_map_resolution_m = float(
            os.environ.get("LIFELONG_WORLD_MAP_RESOLUTION_M", "0.05")
        )
        self.world_map_chunk_size = int(
            os.environ.get("LIFELONG_WORLD_MAP_CHUNK_SIZE", "128")
        )
        self.world_map_pixel_stride = int(
            os.environ.get("LIFELONG_WORLD_MAP_PIXEL_STRIDE", "8")
        )
        self.world_map_max_rays = int(
            os.environ.get("LIFELONG_WORLD_MAP_MAX_RAYS", "2048")
        )
        self.world_trajectory_spacing_m = float(
            os.environ.get("LIFELONG_WORLD_TRAJECTORY_SPACING_M", "0.25")
        )
        self.world_trajectory_max_edge_m = float(
            os.environ.get("LIFELONG_WORLD_TRAJECTORY_MAX_EDGE_M", "1.0")
        )
        self.world_memory = (
            self._new_world_memory()
            if self.world_memory_enabled and not self.capture_only
            else None
        )
        self._world_scene_id = ""
        self._world_source_episode_id = ""
        self._world_observation_count = 0
        self._world_integrated_ray_count = 0
        self._world_terminal_pose_count = 0
        self._world_lavira_occupancy_snapshot = None
        self._world_lavira_fusion_stats = None
        self._preloaded_session = False
        self._preloaded_object_baselines = {}
        self._preloaded_materialized_file_count = 0
        self._preloaded_object_guard_rejections = 0
        self._episode_preloaded_object_guard_rejections = 0
        self._episode_confirmed_keyframes = 0
        self.frame_idx = 0
        self.pipeline = None
        self.failed = False
        self.shared_models = None
        self.runtime_query_vocabulary = None
        self.output_root = os.environ.get("INHERIT_SG_OUTPUT_DIR", save_dir)
        self._frame_queue = None
        self._worker_thread = None
        self._worker_stop = object()
        self._dropped_frames = 0
        self._async_errors = []
        self._episode_active = False
        self._current_episode_name = None
        self._current_runtime_episode_id = None
        self._current_source_episode_id = None
        self._episode_start_frame_idx = 0
        self._session_episode_count = 0
        self._finalized_episode_count = 0
        self._query_lock = threading.RLock()
        # GroundingDINO keeps per-image ``features``/``poss`` as mutable model
        # attributes.  Running Uni-LaViRA semantic inference at the same time as
        # the memory worker's GPU vision stack can therefore leave the cached
        # feature pyramid inconsistent.  Keep the memory update asynchronous,
        # but serialize the GPU vision sections within this process.
        self.gpu_inference_lock = threading.RLock()
        self._query_snapshot = None
        self._query_min_finalized_episodes = max(
            1, int(os.environ.get("INHERIT_SG_QUERY_MIN_FINALIZED_EPISODES", "1"))
        )

    def _new_world_memory(self):
        spec = WorldGridSpec(
            resolution_m=self.world_map_resolution_m,
            chunk_size=self.world_map_chunk_size,
        )
        return WorldMemoryRecorder(
            metric_map=SparseWorldMetricMap(spec),
            trajectory=TrajectoryGraph(
                min_node_translation_m=self.world_trajectory_spacing_m,
                max_edge_length_m=self.world_trajectory_max_edge_m,
            ),
        )

    def _preloaded_query_is_frozen(self):
        return bool(
            self.preload_query_only
            and self._preloaded_session
            and not self.preload_world_map_only
        )

    def update_world_lavira_occupancy_snapshot(
        self,
        observed_free,
        observed_obstacle,
        world_to_map,
        resolution_cm,
    ):
        """Keep the latest LaViRA navigation classification for M0 fusion."""

        if (
            not self.enabled
            or self.world_memory is None
            or self._preloaded_query_is_frozen()
        ):
            return
        free = np.asarray(observed_free, dtype=bool)
        obstacle = np.asarray(observed_obstacle, dtype=bool)
        transform = np.asarray(world_to_map, dtype=np.float64).reshape(-1)
        if free.ndim != 2 or obstacle.shape != free.shape or transform.size != 6:
            raise ValueError("invalid LaViRA occupancy snapshot")
        self._world_lavira_occupancy_snapshot = {
            "observed_free": free.copy(),
            "observed_obstacle": obstacle.copy(),
            "world_to_map": transform.copy(),
            "source_resolution_m": float(resolution_cm) / 100.0,
        }

    def _fuse_world_lavira_occupancy_snapshot(self):
        snapshot = self._world_lavira_occupancy_snapshot
        if snapshot is None or self.world_memory is None:
            return None
        stats = self.world_memory.metric_map.fuse_lavira_navigation_raster(
            **snapshot,
            floor_id=0,
        )
        self._world_lavira_fusion_stats = dict(stats)
        logger.info(
            "[INHerit-SG-V7] fused LaViRA navigation occupancy into absolute "
            f"world map: {stats}"
        )
        return stats

    @staticmethod
    def _npz_scalar_string(path, key):
        with np.load(path, allow_pickle=False) as data:
            return str(np.asarray(data[key]).item())

    def _depth_intrinsics_tuple(self, depth):
        array = np.asarray(depth)
        height, width = array.shape[:2]
        sensor = self.lavira_config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR
        hfov_rad = np.deg2rad(float(sensor.HFOV))
        fx = float(width) / (2.0 * np.tan(hfov_rad / 2.0))
        fy = fx
        return fx, fy, float(width) / 2.0, float(height) / 2.0

    def _depth_sensor_offset(self):
        sensor = self.lavira_config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR
        position = getattr(sensor, "POSITION", (0.0, 1.2, 0.0))
        array = np.asarray(position, dtype=np.float64).reshape(-1)
        if array.size != 3 or not np.isfinite(array).all():
            raise ValueError(f"Invalid Habitat depth sensor position: {position!r}")
        return tuple(map(float, array))

    def _record_world_observation(self, obs_dict, frame_idx, step):
        if self.world_memory is None:
            return
        position = np.asarray(obs_dict["position"], dtype=np.float64).reshape(3)
        heading = np.asarray(obs_dict["heading"], dtype=np.float64).reshape(-1)
        if heading.size == 0:
            raise ValueError("Habitat heading observation is empty")
        sensor = self.lavira_config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR
        stats = self.world_memory.record_observation(
            depth=obs_dict["depth"],
            agent_position_xyz=position,
            heading_rad=float(heading[0]),
            intrinsics=self._depth_intrinsics_tuple(obs_dict["depth"]),
            frame_idx=int(frame_idx),
            step=int(step),
            episode_id=str(self._current_episode_name or ""),
            floor_id=0,
            sensor_offset_xyz=self._depth_sensor_offset(),
            depth_normalized=bool(getattr(sensor, "NORMALIZE_DEPTH", True)),
            min_depth_m=float(sensor.MIN_DEPTH),
            max_depth_m=float(sensor.MAX_DEPTH),
            pixel_stride=self.world_map_pixel_stride,
            max_rays=self.world_map_max_rays,
        )
        self._world_observation_count += 1
        self._world_integrated_ray_count += int(stats.get("rays", 0))
        # Cache observations only when directional exploration and its fallback
        # are enabled. Caching itself never modifies the navigation map.
        from .full_exploration import probe_recovery
        if probe_recovery.observation_cache_enabled():
            from membridge.trajectory_graph import habitat_depth_to_meters
            probe_recovery.observe(
                self.world_memory,
                depth_m=habitat_depth_to_meters(
                    obs_dict["depth"], normalized=bool(getattr(sensor, "NORMALIZE_DEPTH", True)),
                    min_depth_m=float(sensor.MIN_DEPTH), max_depth_m=float(sensor.MAX_DEPTH)),
                position=position, heading=float(heading[0]),
                intrinsics=self._depth_intrinsics_tuple(obs_dict["depth"]),
                sensor_offset=self._depth_sensor_offset(),
                min_depth_m=float(sensor.MIN_DEPTH), max_depth_m=float(sensor.MAX_DEPTH),
                frame_idx=int(frame_idx), episode_id=str(self._current_episode_name or ""))

    def record_terminal_pose(self, terminal_pose, step: int):
        """Commit the actual occupied end cell without inventing terminal depth."""
        if (
            self.capture_only
            and self.enabled
            and not self.failed
            and self._episode_active
        ):
            position = np.asarray(
                terminal_pose["position"], dtype=np.float64
            ).reshape(3)
            heading = float(terminal_pose["heading_rad"])
            payload = {
                "schema_version": "inherit_sg_offline_terminal_pose_v1",
                "step": int(step),
                "position": position.tolist(),
                "heading_rad": heading,
                "coordinate_frame": "habitat_world",
                "has_rgbd_observation": False,
            }
            path = os.path.join(
                self.offline_rgbd_dir, "terminal_pose.json"
            )
            with open(path, "x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=True)
                handle.write("\n")
            logger.info(
                "[INHerit-SG-CAPTURE] saved pose-only terminal state: "
                f"{path}"
            )
            return payload
        if (
            not self.enabled
            or self.failed
            or self.pipeline is None
            or self.world_memory is None
            or not self._episode_active
            or self._preloaded_query_is_frozen()
        ):
            return None
        position = np.asarray(terminal_pose["position"], dtype=np.float64).reshape(3)
        heading = float(terminal_pose["heading_rad"])
        idx = int(self.frame_idx)
        self.frame_idx += 1
        stats = self.world_memory.record_traversed_pose(
            agent_position_xyz=position,
            heading_rad=heading,
            frame_idx=idx,
            step=int(step),
            episode_id=str(self._current_episode_name or ""),
            floor_id=0,
        )
        self._world_terminal_pose_count += 1
        logger.info(
            "[INHerit-SG-V7] recorded actual terminal traversed pose: "
            f"position={position.tolist()}, heading={heading:.6f}, stats={stats}"
        )
        return stats

    def _save_world_memory(self):
        if self.world_memory is None or self.pipeline is None:
            return
        run_dir = getattr(self.pipeline, "run_dir", None)
        if not run_dir:
            raise RuntimeError("Cannot publish V7 world memory without pipeline.run_dir")
        scene_id = self._world_scene_id
        source_episode_id = self._world_source_episode_id
        map_path = self.world_memory.metric_map.save_npz(
            os.path.join(run_dir, "world_metric_map.npz"),
            scene_id=scene_id,
            source_episode_id=source_episode_id,
        )
        trajectory_path = self.world_memory.trajectory.save_npz(
            os.path.join(run_dir, "trajectory_graph.npz"),
            scene_id=scene_id,
            source_episode_id=source_episode_id,
        )
        logger.info(
            "[INHerit-SG-V7] published world memory: "
            f"map={map_path}, trajectory={trajectory_path}, "
            f"map_stats={self.world_memory.metric_map.stats()}, "
            f"trajectory_nodes={len(self.world_memory.trajectory.nodes)}, "
            f"trajectory_edges={len(self.world_memory.trajectory.edges)}"
        )

    def start_episode(
        self,
        episode_id,
        scene_id: Optional[str] = None,
        source_episode_id: Optional[str] = None,
    ):
        self._stop_async_worker(drain=True)
        self.failed = False
        self._dropped_frames = 0
        self._episode_confirmed_keyframes = 0
        self._episode_preloaded_object_guard_rejections = 0
        self._async_errors = []
        self._episode_active = False
        if not self.enabled:
            return

        try:
            episode_name = self._episode_dir_name(episode_id, scene_id)
            self._current_runtime_episode_id = str(episode_id)
            self._current_source_episode_id = (
                str(source_episode_id)
                if source_episode_id is not None
                else str(episode_id)
            )
            if self.capture_only:
                self._current_episode_name = episode_name
                self._prepare_offline_rgbd_capture()
                self._episode_start_frame_idx = self.frame_idx
                self._episode_active = True
                self._session_episode_count += 1
                logger.info(
                    "[INHerit-SG-CAPTURE] started model-free RGB-D capture "
                    f"for {episode_name} at {self.offline_rgbd_dir}; "
                    "SG pipeline and semantic models are not initialized"
                )
                return
            if self.pipeline is None:
                self._initialize_session_pipeline(scene_id=scene_id)
            else:
                self._patch_pipeline_compatibility()
            if hasattr(self.pipeline, "set_episode_context"):
                self.pipeline.set_episode_context(
                    self._current_runtime_episode_id,
                    self._current_source_episode_id,
                )

            self._current_episode_name = episode_name
            self._prepare_offline_rgbd_capture()
            # Freeze after session preload and before the current episode can
            # append observations/poses. EP2 therefore sees both EP0 and EP1.
            from membridge.navigation.map_connection import freeze_episode_history
            freeze_episode_history(self)
            self._world_scene_id = str(scene_id or self._world_scene_id)
            if not self._preloaded_session and not self._world_source_episode_id:
                self._world_source_episode_id = str(episode_id)
            self._episode_start_frame_idx = self.frame_idx
            self._episode_active = True
            self._session_episode_count += 1
            if self.async_enabled and not (self.preload_query_only and self._preloaded_session):
                self._start_async_worker()
            logger.info(
                f"[INHerit-SG] started episode {episode_name} in lifelong session at "
                f"{self.pipeline.run_dir} (async={self.async_enabled}, stride={self.stride}, "
                f"keyframe_filter={self.use_keyframe_filter}, "
                f"start_frame={self._episode_start_frame_idx})"
            )
        except Exception as exc:
            self.failed = True
            logger.warning(f"[INHerit-SG] disabled for this episode: {exc}\n{traceback.format_exc()}")
            if self.fail_closed:
                raise

    def process_observation(self, obs_dict, step: int):
        if self.fail_closed and self._async_errors:
            raise RuntimeError(
                "Asynchronous INHerit-SG processing failed: " + self._async_errors[0]
            )
        if not self.enabled or self.failed:
            return
        if self.capture_only:
            idx = self.frame_idx
            self.frame_idx += 1
            try:
                self._build_frame_payload(obs_dict, idx, step)
            except Exception as exc:
                logger.warning(
                    f"[INHerit-SG-CAPTURE] frame {idx} failed: "
                    f"{exc}\n{traceback.format_exc()}"
                )
                if self.fail_closed:
                    raise
            return
        if self.pipeline is None:
            return
        if self.preload_query_only and self._preloaded_session:
            if not self.preload_world_map_only and not self.save_offline_rgbd:
                return
            idx = self.frame_idx
            self.frame_idx += 1
            try:
                if self.preload_world_map_only:
                    self._record_world_observation(obs_dict, idx, step)
                if self.save_offline_rgbd:
                    # Preserve the replay observation stream without running
                    # semantic inference or mutating the imported scene graph.
                    self._build_frame_payload(obs_dict, idx, step)
            except Exception as exc:
                logger.warning(
                    f"[INHerit-SG-QUERY-ONLY] frame {idx} failed: "
                    f"{exc}\n{traceback.format_exc()}"
                )
                if self.fail_closed:
                    raise
            return
        idx = self.frame_idx
        self.frame_idx += 1
        try:
            self._record_world_observation(obs_dict, idx, step)
            frame = self._build_frame_payload(obs_dict, idx, step)
            if idx % self.stride != 0:
                return
            if self.use_keyframe_filter and not self._is_keyframe_payload(frame):
                return
            if self.async_enabled:
                self._enqueue_frame(frame)
            else:
                self._process_frame_payload(frame)
        except Exception as exc:
            logger.warning(f"[INHerit-SG] frame {idx} failed: {exc}\n{traceback.format_exc()}")
            if self.fail_closed:
                raise

    def finish_episode(self):
        if not self.enabled or not self._episode_active:
            return
        try:
            if self.capture_only:
                self._write_offline_rgbd_manifest()
                logger.info(
                    "[INHerit-SG-CAPTURE] model-free RGB-D capture "
                    "finalized; skipped all SG mapping"
                )
                return
            if self.pipeline is None:
                raise RuntimeError(
                    "INHerit-SG pipeline is missing during episode finalize"
                )
            if self.preload_query_only and self._preloaded_session:
                if self.preload_world_map_only:
                    self._fuse_world_lavira_occupancy_snapshot()
                    self._save_world_memory()
                self._write_episode_manifest_row()
                self._write_offline_rgbd_manifest()
                logger.info(
                    "[INHerit-SG] preload query-only episode finished; "
                    + (
                        "published RGB-D world map/trajectory updates without "
                        "semantic model inference"
                        if self.preload_world_map_only
                        else "skipped online finalize"
                    )
                )
                return
            self._stop_async_worker(drain=True)
            if self._async_errors:
                first_error = self._async_errors[0]
                raise RuntimeError(
                    f"{len(self._async_errors)} asynchronous SG frame(s) failed; "
                    f"first failure: {first_error}"
                )
            self.pipeline._finalize_algorithm()
            self.pipeline.save_all_data()
            self._fuse_world_lavira_occupancy_snapshot()
            self._save_world_memory()
            self._save_builder_state()
            self._prune_nonessential_output_artifacts()
            self._finalized_episode_count += 1
            self._refresh_query_snapshot()
            self._write_episode_manifest_row()
            self._write_offline_rgbd_manifest()
            if self._dropped_frames:
                logger.info(f"[INHerit-SG] episode SG finalized; async dropped {self._dropped_frames} frames")
            else:
                logger.info("[INHerit-SG] episode SG finalized")
        except Exception as exc:
            self.failed = True
            logger.warning(f"[INHerit-SG] finalize failed: {exc}\n{traceback.format_exc()}")
            if self.fail_closed:
                raise
        finally:
            self._episode_active = False
            self._current_episode_name = None
            self._current_runtime_episode_id = None
            self._current_source_episode_id = None
            self._frame_queue = None
            self._worker_thread = None

    def abort_episode(self):
        """Discard queued frames after a failed rollout without publishing partial M0."""
        if not self._episode_active:
            return
        try:
            self._stop_async_worker(drain=False)
        finally:
            self.failed = True
            self._episode_active = False
            self._current_episode_name = None
            self._current_runtime_episode_id = None
            self._current_source_episode_id = None
            self._frame_queue = None
            self._worker_thread = None

    def _initialize_session_pipeline(self, scene_id: Optional[str] = None):
        import yaml

        source_dir = os.environ.get(
            "MEMORY_MODULE_SEMANTIC_SOURCE_DIR",
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "semantic_scene_graph",
                "legacy_builder_source",
            ),
        )
        config_path = os.environ.get(
            "INHERIT_SG_CONFIG",
            os.path.join(source_dir, "config_sam3.yaml"),
        )
        with open(config_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)

        session_name = os.environ.get("INHERIT_SG_SESSION_NAME", "lifelong_session")
        session_dir = os.path.join(self.output_root, self._safe_name(session_name))
        data_dir = (
            self.external_frame_root
            if self.external_frame_root
            else os.path.join(session_dir, "frames")
        )
        intrinsics = self._camera_intrinsics()

        cfg = copy.deepcopy(base_cfg)
        cfg["main_output_directory"] = session_dir
        cfg["debug_mode"] = self.debug
        # The vendored builder historically wrote several visualizations even
        # when debug_mode was false.  Propagate the runtime policy into both
        # the pipeline and SAM3 component so production memories contain only
        # artifacts needed for retrieval or continuation.
        cfg["save_debug_artifacts"] = self.debug
        cfg["disable_llm"] = True
        cfg["episode_provenance"] = {
            "schema_version": 1,
            "runtime_episode_id": self._current_runtime_episode_id,
            "source_episode_id": self._current_source_episode_id,
            "selection_center_weight": self.observation_center_weight,
            "selection_area_weight": self.observation_area_weight,
        }
        cfg["use_predefined_object_classes"] = True
        self._apply_runtime_query_vocabulary(cfg)
        cfg["gpu_id"] = str(getattr(self.lavira_config, "TORCH_GPU_ID", 0))
        cfg["camera_intrinsics"] = {
            "w": self.width,
            "h": self.height,
            "fx": float(intrinsics[0, 0]),
            "fy": float(intrinsics[1, 1]),
            "cx": float(intrinsics[0, 2]),
            "cy": float(intrinsics[1, 2]),
            "depth_scale": 1000.0,
        }
        cfg.setdefault("multi_gpu", {})
        sam3_device = os.environ.get("INHERIT_SG_SAM3_DEVICE") or self.device
        dinov3_device = os.environ.get("INHERIT_SG_DINOV3_DEVICE") or self.device
        text_encoder_device = os.environ.get("INHERIT_SG_TEXT_ENCODER_DEVICE") or self.device
        cfg["multi_gpu"].update({
            "enabled": True,
            "sam3_device": sam3_device,
            "dinov3_device": dinov3_device,
            "text_encoder_device": text_encoder_device,
        })
        batch_size_text = os.environ.get(
            "INHERIT_SG_SAM3_CATEGORY_BATCH_SIZE",
            str(V8_SAM3_CATEGORY_BATCH_SIZE),
        )
        per_device_batch_sizes = os.environ.get(
            "INHERIT_SG_SAM3_CATEGORY_BATCH_SIZE_BY_DEVICE", ""
        ).strip()
        if per_device_batch_sizes:
            values = [
                item.strip() for item in per_device_batch_sizes.split(",")
            ]
            logical_device = int(getattr(self.lavira_config, "TORCH_GPU_ID", 0))
            if logical_device < 0 or logical_device >= len(values):
                raise ValueError(
                    "INHERIT_SG_SAM3_CATEGORY_BATCH_SIZE_BY_DEVICE has no "
                    f"entry for logical CUDA device {logical_device}"
                )
            batch_size_text = values[logical_device]
        batch_size = int(batch_size_text)
        if not 1 <= batch_size <= V8_QUERY_LABEL_COUNT:
            raise ValueError(
                "INHERIT_SG_SAM3_CATEGORY_BATCH_SIZE must be between 1 and "
                f"{V8_QUERY_LABEL_COUNT}, received {batch_size}"
            )
        cfg["category_batch_size"] = batch_size
        cfg.setdefault("segmentation_module", {})[
            "category_batch_size"
        ] = batch_size
        cfg["segmentation_module"]["save_debug_artifacts"] = self.debug
        cfg.setdefault("keyframe_detection", {})["enable_keyframe_detection"] = False
        if self.use_keyframe_filter:
            cfg["keyframe_detection"]["enable_keyframe_detection"] = True

        # Re-read public/internal env here.  The L1 warm binder historically
        # forced preload_query_only=False after construction; a query-only
        # replay must still skip SAM3/DINOv3 even if that happens again.
        if self._env_bool("INHERIT_SG_PRELOAD_QUERY_ONLY", False) or self._env_bool(
            "MEMORY_MODULE_PRELOAD_QUERY_ONLY", False
        ):
            self.preload_query_only = True
        if self._env_bool("INHERIT_SG_PRELOAD_WORLD_MAP_ONLY", False) or self._env_bool(
            "MEMORY_MODULE_PRELOAD_WORLD_MAP_ONLY", False
        ):
            self.preload_world_map_only = True
            self.preload_query_only = True

        if self.preload_query_only:
            if not self.preload_dir:
                raise RuntimeError(
                    "INHERIT_SG_PRELOAD_QUERY_ONLY=1 requires INHERIT_SG_PRELOAD_DIR"
                )
            # This branch must remain above the semantic builder import: even
            # importing that builder requires the SAM3 package.  A read-only
            # replay needs only persisted memory/world artifacts and lexical
            # query state.
            os.makedirs(session_dir, exist_ok=False)
            self.pipeline = types.SimpleNamespace(
                config=cfg,
                run_dir=session_dir,
                main_output_directory=session_dir,
                memory_kb=[],
                spatial_graph=None,
                label_counters={},
            )
            self._preload_existing_session_if_requested()
            self._write_session_metadata(scene_id)
            logger.info(
                "[INHerit-SG] initialized model-free query-only pipeline; "
                "SAM3 and DINOv3 were not imported or loaded"
            )
            return

        if os.environ.get("INHERIT_SG_SAM3_CKPT"):
            cfg.setdefault("segmentation_module", {})["checkpoint_path"] = os.environ["INHERIT_SG_SAM3_CKPT"]
        if os.environ.get("INHERIT_SG_DINOV3_MODEL"):
            feature_cfg = cfg.setdefault("feature_extractor", {})
            feature_cfg["weights_path"] = self._resolve_dinov3_weights_path(os.environ["INHERIT_SG_DINOV3_MODEL"])
        dinov3_repo = self._resolve_dinov3_repo_dir()
        if dinov3_repo:
            cfg.setdefault("feature_extractor", {})["repo_dir"] = dinov3_repo

        from . import builder as sg_builder

        if os.environ.get("INHERIT_SG_SAM3_SOURCE_ROOT"):
            sg_builder.SAM3_SOURCE_ROOT = os.environ[
                "INHERIT_SG_SAM3_SOURCE_ROOT"
            ]
        self._patch_builder_feature_extractor(sg_builder)
        self._patch_builder_no_api_llm(sg_builder)

        overrides = {
            "main_output_directory": session_dir,
            "debug_mode": self.debug,
            "trajectory_data": {
                "base_dir": data_dir,
                "rgb_subdir": "rgb",
                "depth_subdir": "depth",
                "pose_subdir": "pose",
                "filename_padding": 6,
                "filename_start_index": 0,
                "filename_prefix": "",
                "num_images_to_process": -1,
            },
        }

        shared_models = self._get_shared_models(sg_builder, cfg) if self.share_models else None
        self.pipeline = sg_builder.TrajectoryPipeline(cfg, override_settings=overrides, shared_models=shared_models)
        self._patch_pipeline_compatibility()
        self._preload_existing_session_if_requested()
        self._write_session_metadata(scene_id)

    def _apply_runtime_query_vocabulary(self, cfg):
        """Apply one exact, reviewed SAM3 query-vocabulary contract."""
        path = os.environ.get("INHERIT_SG_QUERY_VOCABULARY", "").strip()
        if not path:
            raise RuntimeError(
                "V8 requires INHERIT_SG_QUERY_VOCABULARY; refusing to use "
                "the builder's legacy prompt list"
            )
        with open(path, "rb") as handle:
            raw = handle.read()
        payload = json.loads(raw.decode("utf-8"))
        labels = []
        seen = set()
        for value in payload.get("labels") or []:
            label = " ".join(str(value or "").strip().lower().replace("_", " ").split())
            if not label:
                raise ValueError(f"Empty label in runtime query vocabulary: {path}")
            if "." in label:
                raise ValueError(
                    f"Runtime query label contains the mapper delimiter '.': {label!r}"
                )
            if label in seen:
                raise ValueError(f"Duplicate runtime query label {label!r}: {path}")
            seen.add(label)
            labels.append(label)
        vocabulary_version = payload.get("vocabulary_version")
        expected_labels = SUPPORTED_QUERY_VOCABULARIES.get(vocabulary_version)
        if expected_labels is None:
            raise ValueError(
                "Unsupported runtime query vocabulary version "
                f"{vocabulary_version!r}: {path}"
            )
        if len(labels) != len(expected_labels):
            raise ValueError(
                f"{vocabulary_version} requires exactly {len(expected_labels)} "
                f"runtime query labels, found {len(labels)}: {path}"
            )
        if tuple(labels) != expected_labels:
            raise ValueError(
                "Runtime query labels differ from the reviewed contract for "
                f"{vocabulary_version}: {path}"
            )
        raw_aliases = payload.get("aliases") or {}
        if not isinstance(raw_aliases, dict):
            raise ValueError(
                f"Runtime query aliases must be an object: {path}"
            )
        aliases = {}
        for raw_alias, raw_canonical in raw_aliases.items():
            alias = self._clean_phrase(raw_alias)
            canonical = self._clean_phrase(raw_canonical)
            if (
                not alias
                or not canonical
                or canonical not in seen
                or alias in aliases
            ):
                raise ValueError(
                    f"Invalid runtime query alias "
                    f"{raw_alias!r}->{raw_canonical!r}: {path}"
                )
            aliases[alias] = canonical
        raw_small_labels = payload.get("small_object_labels") or []
        small_labels = [self._clean_phrase(value) for value in raw_small_labels]
        if (
            any(not label or label not in seen for label in small_labels)
            or len(small_labels) != len(set(small_labels))
        ):
            raise ValueError(
                f"Invalid runtime small-object labels: {path}"
            )
        raw_size_filter = payload.get("size_filter")
        if not isinstance(raw_size_filter, dict):
            raise ValueError(f"Runtime vocabulary has no size_filter: {path}")
        size_filter = {}
        for key in (
            "default_min_area_px",
            "default_min_side_px",
            "small_min_area_px",
            "small_min_side_px",
        ):
            value = raw_size_filter.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"Invalid runtime size_filter {key}={value!r}: {path}"
                )
            size_filter[key] = int(value)
        if (
            size_filter["small_min_area_px"]
            > size_filter["default_min_area_px"]
            or size_filter["small_min_side_px"]
            > size_filter["default_min_side_px"]
        ):
            raise ValueError(
                f"Small-object thresholds exceed defaults: {path}"
            )
        cfg["use_predefined_object_classes"] = True
        cfg["predefined_object_classes"] = ".".join(labels)
        segmentation_cfg = cfg.get("segmentation_module")
        if not isinstance(segmentation_cfg, dict):
            raise ValueError(
                "Runtime SG config segmentation_module must be an object"
            )
        segmentation_cfg["size_filter"] = {
            **size_filter,
            "small_object_labels": small_labels,
        }
        self.runtime_query_vocabulary = {
            "path": os.path.abspath(path),
            "schema_version": payload.get("schema_version"),
            "vocabulary_version": payload.get("vocabulary_version"),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "label_count": len(labels),
            "labels": labels,
            "aliases": aliases,
            "small_object_labels": small_labels,
            "size_filter": size_filter,
        }
        logger.info(
            "[INHerit-SG-V8] applied fixed mapping/query vocabulary: "
            f"{len(labels)} labels, version="
            f"{payload.get('vocabulary_version')}, sha256="
            f"{self.runtime_query_vocabulary['sha256']}"
        )

    def _preload_existing_session_if_requested(self):
        if self._preloaded_session or not self.preload_dir or self.pipeline is None:
            return

        src_dir = self.preload_dir
        kb_path = os.path.join(src_dir, "memory_kb.json")
        graph_path = os.path.join(src_dir, "spatial_graph.pkl")
        map_path = os.path.join(src_dir, "map_data.pkl")
        world_map_path = os.path.join(src_dir, "world_metric_map.npz")
        trajectory_path = os.path.join(src_dir, "trajectory_graph.npz")
        builder_state_path = os.path.join(src_dir, BUILDER_STATE_NAME)
        if not os.path.isfile(kb_path):
            raise FileNotFoundError(f"INHERIT_SG_PRELOAD_DIR missing memory_kb.json: {kb_path}")
        builder_state_required = self._env_bool(
            "INHERIT_SG_PRELOAD_BUILDER_STATE_REQUIRED",
            False,
        )
        if builder_state_required and not os.path.isfile(builder_state_path):
            raise FileNotFoundError(
                "Read-write Level 3B continuation requires the complete builder "
                f"state: {builder_state_path}"
            )

        with open(kb_path, "r", encoding="utf-8") as f:
            self.pipeline.memory_kb = json.load(f)
        # V8 accepts V7 artifacts and migrates their old space-concatenated
        # labels to one canonical label. Instance association remains V7's.
        from membridge.semantic_scene_graph.legacy_builder_source.label_score import (
            ensure_label_score_state,
        )

        for item in self.pipeline.memory_kb:
            ensure_label_score_state(item)
            ensure_object_provenance(item)

        if os.path.isfile(graph_path):
            with open(graph_path, "rb") as f:
                self.pipeline.spatial_graph = pickle.load(f)
        for item in self.pipeline.memory_kb:
            sync_object_provenance_to_graph(
                self.pipeline.spatial_graph,
                item,
            )

        if os.path.isfile(map_path) and not self.preload_query_only:
            with open(map_path, "rb") as f:
                map_data = pickle.load(f)
            self._restore_map_manager_from_saved_data(map_data)
        self._copy_preloaded_runtime_artifacts(src_dir)
        self._rebase_preloaded_artifact_paths()
        if os.path.isfile(builder_state_path) and not self.preload_query_only:
            self._restore_builder_state(builder_state_path)

        if self.world_memory_enabled:
            missing_world = [
                path
                for path in (world_map_path, trajectory_path)
                if not os.path.isfile(path)
            ]
            if missing_world and self.world_memory_required:
                raise FileNotFoundError(
                    f"V7 preload is missing world-memory artifacts: {missing_world}"
                )
            if not missing_world:
                self.world_memory = WorldMemoryRecorder(
                    metric_map=SparseWorldMetricMap.load_npz(world_map_path),
                    trajectory=TrajectoryGraph.load_npz(trajectory_path),
                )
                self._world_scene_id = self._npz_scalar_string(
                    world_map_path, "scene_id"
                )
                self._world_source_episode_id = self._npz_scalar_string(
                    world_map_path, "source_episode_id"
                )

        self._restore_label_counters_from_memory()
        self._restore_frame_index_from_preload()
        self._preloaded_object_baselines = {
            str(item.get("node_id")): copy.deepcopy(item)
            for item in getattr(self.pipeline, "memory_kb", [])
            if item.get("node_id") is not None
        }
        self.pipeline._inherit_sg_final_room_ids = self._valid_room_ids_from_map(self.pipeline)
        self._finalized_episode_count = max(1, self._preload_finalized_episode_count(src_dir))
        self._preloaded_session = True
        self._refresh_query_snapshot()
        logger.info(
            f"[INHerit-SG] preloaded session from {src_dir}: "
            f"{len(getattr(self.pipeline, 'memory_kb', []))} objects, "
            f"finalized_eps={self._finalized_episode_count}, next_frame={self.frame_idx}, "
            f"world_memory={self.world_memory is not None}"
        )

    def _rebase_preloaded_artifact_paths(self):
        """Point copied M0/M1 observations at the current continuation workspace."""
        pipeline = self.pipeline
        if pipeline is None:
            return
        run_dir = os.path.normpath(str(pipeline.run_dir))
        markers = (
            "frames/",
            "objects/",
            "room_summary_images/",
            "trajectory_maps/",
        )

        def rebase(value):
            if not value:
                return value
            normalized = os.path.normpath(str(value))
            slash_path = normalized.replace(os.sep, "/")
            # EP0 SG memories can point directly at Uni-LaViRA's exported
            # source frames (``sg_source_frames/rgb``), rather than at a
            # ``frames`` directory inside the frozen M0 package.  Those
            # referenced images are materialized below into the continuation
            # workspace, so normalize both source layouts to the same local
            # destination.
            source_rgb_marker = "/sg_source_frames/rgb/"
            if source_rgb_marker in slash_path:
                name = slash_path.rsplit("/", 1)[-1]
                stem, extension = os.path.splitext(name)
                if stem.isdigit() and extension.lower() == ".png":
                    candidates = [os.path.join(run_dir, "frames", "rgb", name)]
                    if self.preload_query_only:
                        candidates.append(
                            os.path.join(self.preload_dir, "frames", "rgb", name)
                        )
                    for candidate in candidates:
                        if os.path.isfile(candidate):
                            return os.path.normpath(candidate)
            for marker in markers:
                token = f"/{marker}"
                position = slash_path.find(token)
                if position < 0 and slash_path.startswith(marker):
                    relative = slash_path
                elif position >= 0:
                    relative = slash_path[position + 1 :]
                else:
                    continue
                candidates = [
                    os.path.normpath(os.path.join(run_dir, *relative.split("/")))
                ]
                if self.preload_query_only:
                    candidates.append(
                        os.path.normpath(
                            os.path.join(self.preload_dir, *relative.split("/"))
                        )
                    )
                for candidate in candidates:
                    if os.path.isfile(candidate):
                        return candidate
            return value

        updated = 0
        graph = getattr(pipeline, "spatial_graph", None)
        for item in getattr(pipeline, "memory_kb", []):
            changed = False
            for field in ("full_image_path", "cropped_image_path"):
                old = item.get(field)
                new = rebase(old)
                if new != old:
                    item[field] = new
                    changed = True
            observation = item.get("best_observation")
            if isinstance(observation, dict):
                old = observation.get("rgb_path")
                new = rebase(old)
                if new != old:
                    observation["rgb_path"] = new
                    changed = True
            observations_by_episode = item.get("best_observation_by_episode")
            if isinstance(observations_by_episode, dict):
                for episode_observation in observations_by_episode.values():
                    if not isinstance(episode_observation, dict):
                        continue
                    old = episode_observation.get("rgb_path")
                    new = rebase(old)
                    if new != old:
                        episode_observation["rgb_path"] = new
                        changed = True
            if not changed:
                continue
            updated += 1
            node_id = item.get("node_id")
            if graph is not None and node_id is not None and graph.has_node(node_id):
                graph.nodes[node_id].update(
                    {
                        "full_image_path": item.get("full_image_path"),
                        "cropped_image_path": item.get("cropped_image_path"),
                        "best_observation": copy.deepcopy(
                            item.get("best_observation")
                        ),
                        "best_observation_by_episode": copy.deepcopy(
                            item.get("best_observation_by_episode")
                        ),
                    }
                )
        if updated:
            logger.info(
                f"[INHerit-SG] rebased copied observation paths for {updated} "
                f"preloaded object node(s) into {run_dir}"
            )

    @staticmethod
    def _make_working_copy_owner_writable(path):
        """Make a copied continuation artifact writable without touching M0."""
        if os.path.isdir(path):
            for root, directories, files in os.walk(path):
                os.chmod(
                    root,
                    os.stat(root).st_mode
                    | stat.S_IRUSR
                    | stat.S_IWUSR
                    | stat.S_IXUSR,
                )
                for name in directories:
                    directory = os.path.join(root, name)
                    os.chmod(
                        directory,
                        os.stat(directory).st_mode
                        | stat.S_IRUSR
                        | stat.S_IWUSR
                        | stat.S_IXUSR,
                    )
                for name in files:
                    file_path = os.path.join(root, name)
                    os.chmod(
                        file_path,
                        os.stat(file_path).st_mode
                        | stat.S_IRUSR
                        | stat.S_IWUSR,
                    )
        else:
            os.chmod(
                path,
                os.stat(path).st_mode | stat.S_IRUSR | stat.S_IWUSR,
            )

    @staticmethod
    def _copy_tree_contents(source, destination):
        if not os.path.isdir(source):
            return
        os.makedirs(destination, exist_ok=True)
        for name in os.listdir(source):
            src = os.path.join(source, name)
            dst = os.path.join(destination, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
            OnlineINHeritSGBuilder._make_working_copy_owner_writable(dst)

    def _copy_preloaded_runtime_artifacts(self, src_dir):
        """Seed the continuation run with E0 files needed by later finalization."""
        pipeline = self.pipeline
        if pipeline is None:
            return
        materialized_file_count = 0
        # A formal continuation does not need another full copy of the
        # immutable E0 RGB-D-pose package.  It does, however, need the RGB
        # frames referenced by the preloaded object nodes so that the
        # accumulated M1 is self-contained and can pass the frozen-memory
        # waypoint contract.  Query-only EP2 never publishes M2, so it does
        # not materialize even these prior RGB references.
        if not self.preload_query_only:
            source_rgb_dir = os.path.join(src_dir, "frames", "rgb")
            destination_rgb_dir = os.path.join(
                pipeline.run_dir, "frames", "rgb"
            )
            referenced_sources = defaultdict(list)

            def add_explicit_rgb(path, label):
                if not path:
                    return False
                path = os.path.normpath(str(path))
                name = os.path.basename(path)
                stem, extension = os.path.splitext(name)
                if not stem.isdigit() or extension.lower() != ".png":
                    raise RuntimeError(
                        "Preloaded object memory contains an invalid RGB "
                        f"reference for {label}: {path}"
                    )
                referenced_sources[name].append((path, label))
                return True

            def add_observation(observation, label):
                if not isinstance(observation, dict):
                    return
                # The persisted frame_idx is builder-local in older M0
                # exports and can be offset from the source capture filename.
                # An explicit rgb_path is therefore authoritative.  Only fall
                # back to frame_idx for packages that do not persist a path.
                if add_explicit_rgb(observation.get("rgb_path"), label):
                    return
                frame_idx = observation.get("frame_idx")
                if isinstance(frame_idx, int) and frame_idx >= 0:
                    name = f"{frame_idx:06d}.png"
                    referenced_sources[name].append(
                        (os.path.join(source_rgb_dir, name), label)
                    )

            for item in getattr(pipeline, "memory_kb", []):
                node_label = f"node {item.get('node_id', '<unknown>')}"
                add_explicit_rgb(
                    item.get("full_image_path"),
                    f"{node_label}.full_image_path",
                )
                add_observation(
                    item.get("best_observation"),
                    f"{node_label}.best_observation",
                )
                observations_by_episode = item.get(
                    "best_observation_by_episode"
                )
                if isinstance(observations_by_episode, dict):
                    for episode_id, episode_observation in (
                        observations_by_episode.items()
                    ):
                        add_observation(
                            episode_observation,
                            f"{node_label}.best_observation_by_episode"
                            f"[{episode_id}]",
                        )
            for name, references in sorted(referenced_sources.items()):
                # Prefer the explicit path stored in memory.  Also accept the
                # portable M0 layout when it is available.  Legacy Level-1 M0
                # archives can point at the immutable native EP0 capture via
                # its old container absolute path; resolve that data through
                # the explicit memory_module mount instead of requiring the
                # old workspace layout.
                candidates = [path for path, _ in references]
                candidates.append(os.path.join(source_rgb_dir, name))
                native_ep0_root = os.environ.get(
                    "MEMORY_MODULE_NATIVE_EP0_ROOT", ""
                ).strip()
                if native_ep0_root:
                    candidates.append(os.path.join(
                        native_ep0_root,
                        "sessions",
                        os.path.basename(os.path.normpath(src_dir)),
                        "sg_source_frames",
                        "rgb",
                        name,
                    ))
                existing = []
                seen = set()
                for candidate in candidates:
                    candidate = os.path.normpath(candidate)
                    if not os.path.isfile(candidate):
                        continue
                    real_path = os.path.realpath(candidate)
                    if real_path in seen:
                        continue
                    seen.add(real_path)
                    existing.append(candidate)
                if not existing:
                    described_references = ", ".join(
                        f"{label}={path}" for path, label in references
                    )
                    raise FileNotFoundError(
                        "Preloaded object memory references a non-portable RGB "
                        f"frame {name}: {described_references}"
                    )
                # A basename is the stable memory key.  Refuse to silently
                # collapse two different images onto the same key.
                if len(existing) > 1:
                    digests = set()
                    for candidate in existing:
                        digest = hashlib.sha256()
                        with open(candidate, "rb") as handle:
                            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                                digest.update(chunk)
                        digests.add(digest.hexdigest())
                    if len(digests) > 1:
                        raise RuntimeError(
                            "Conflicting preloaded RGB sources share filename "
                            f"{name}: {existing}"
                        )
                source = existing[0]
                os.makedirs(destination_rgb_dir, exist_ok=True)
                destination = os.path.join(destination_rgb_dir, name)
                shutil.copy2(source, destination)
                self._make_working_copy_owner_writable(destination)
                materialized_file_count += 1
        runtime_directories = ["objects", "room_summary_images"]
        if self.debug:
            runtime_directories.extend(("debug_merge", "trajectory_maps"))
        for directory in runtime_directories:
            source_directory = os.path.join(src_dir, directory)
            if os.path.isdir(source_directory):
                materialized_file_count += sum(
                    len(file_names)
                    for _, _, file_names in os.walk(source_directory)
                )
            self._copy_tree_contents(
                source_directory,
                os.path.join(pipeline.run_dir, directory),
            )
        for name in os.listdir(src_dir):
            if (
                name in {"nodes.jsonl", "relations.csv"}
                or name.startswith("mask_room_")
            ):
                source = os.path.join(src_dir, name)
                if os.path.isfile(source):
                    destination = os.path.join(pipeline.run_dir, name)
                    shutil.copy2(source, destination)
                    self._make_working_copy_owner_writable(destination)
                    materialized_file_count += 1
        # A read-write query branch is an auditable M0 -> M(query) update.  Seed
        # the immutable EP0 manifest row into the private branch before the
        # query row is appended at finalize time.  This copies metadata only;
        # the source M0 remains untouched.
        source_manifest = os.path.join(src_dir, "episode_manifest.jsonl")
        if os.path.isfile(source_manifest) and not self.preload_query_only:
            destination_manifest = os.path.join(
                pipeline.run_dir, "episode_manifest.jsonl"
            )
            shutil.copy2(source_manifest, destination_manifest)
            self._make_working_copy_owner_writable(destination_manifest)
            materialized_file_count += 1
        self._preloaded_materialized_file_count = materialized_file_count

    @staticmethod
    def _plain_floor_data(floors_data):
        result = {}
        for floor_id, value in dict(floors_data or {}).items():
            result[int(floor_id)] = {
                "heights": [float(item) for item in value.get("heights", [])],
                "rooms": sorted(str(item) for item in value.get("rooms", set())),
                "finalized_count": int(value.get("finalized_count", 0)),
            }
        return result

    def _builder_state_payload(self):
        pipeline = self.pipeline
        if pipeline is None:
            raise RuntimeError("Cannot serialize builder state without a pipeline")
        map_manager = pipeline.map_manager
        room_segmenter = pipeline.room_segmenter
        floor_tracker = pipeline.floor_tracker
        pipeline_state = {
            name: copy.deepcopy(getattr(pipeline, name, None))
            for name in (
                "robot_trajectory",
                "floor_trajectories",
                "room_frame_history",
                "pose_cache",
                "room_metadata_cache",
                "room_topology_cache",
                "room_exit_buffer",
                "global_update_history",
            )
        }
        pipeline_state.update(
            {
                "visited_room_ids": sorted(
                    int(item)
                    for item in getattr(pipeline, "visited_room_ids", set())
                ),
                "current_room_id": int(
                    getattr(pipeline, "current_room_id", 0)
                ),
                "last_room_id": int(getattr(pipeline, "last_room_id", 0)),
            }
        )
        map_state = {
            name: copy.deepcopy(getattr(map_manager, name, None))
            for name in (
                "logodds",
                "door_map",
                "occ_bin",
                "free_bin",
                "room_id_map",
                "door_hits",
                "floor_maps",
                "room_registry",
                "room_geometry",
            )
        }
        map_state.update(
            {
                "current_floor_idx": int(
                    getattr(map_manager, "current_floor_idx", 0)
                ),
                "current_room_id": int(
                    getattr(map_manager, "current_room_id", 0)
                ),
            }
        )
        room_state = {
            name: copy.deepcopy(getattr(room_segmenter, name, None))
            for name in (
                "global_id_grid",
                "global_conf_grid",
                "floor_states",
                "colors",
            )
        }
        room_state.update(
            {
                "current_floor_idx": int(
                    getattr(room_segmenter, "current_floor_idx", 0)
                ),
                "current_room_id": int(
                    getattr(room_segmenter, "current_room_id", 0)
                ),
                "next_global_room_id": int(
                    getattr(room_segmenter, "next_global_room_id", 1)
                ),
            }
        )
        floor_state = {
            "current_floor_id": int(
                getattr(floor_tracker, "current_floor_id", 0)
            ),
            "current_room_id": int(
                getattr(floor_tracker, "current_room_id", 0)
            ),
            "base_height": getattr(floor_tracker, "base_height", None),
            "height_buffer": list(
                getattr(floor_tracker, "height_buffer", [])
            ),
            "id_buffer": list(getattr(floor_tracker, "id_buffer", [])),
            "pending_data_cache": list(
                getattr(floor_tracker, "pending_data_cache", [])
            ),
            "floors_data": self._plain_floor_data(
                getattr(floor_tracker, "floors_data", {})
            ),
            "next_floor_id": getattr(floor_tracker, "next_floor_id", None),
        }
        return {
            "schema_version": BUILDER_STATE_SCHEMA_VERSION,
            "state_kind": "inherit_sg_streaming_continuation",
            "pipeline": pipeline_state,
            "map_manager": map_state,
            "room_segmenter": room_state,
            "floor_tracker": floor_state,
        }

    def _save_builder_state(self):
        if self.pipeline is None:
            return
        path = os.path.join(self.pipeline.run_dir, BUILDER_STATE_NAME)
        temporary = f"{path}.tmp-{os.getpid()}"
        with open(temporary, "wb") as handle:
            pickle.dump(
                self._builder_state_payload(),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        logger.info(f"[INHerit-SG] published complete builder state: {path}")

    @staticmethod
    def _restore_named_state(target, payload, names):
        for name in names:
            if name in payload:
                setattr(target, name, copy.deepcopy(payload[name]))

    def _restore_builder_state(self, path):
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        if not isinstance(state, dict) or int(state.get("schema_version", -1)) != (
            BUILDER_STATE_SCHEMA_VERSION
        ):
            raise RuntimeError(f"Unsupported INHerit-SG builder state: {path}")
        if state.get("state_kind") != "inherit_sg_streaming_continuation":
            raise RuntimeError(f"Unexpected INHerit-SG builder state kind: {path}")

        pipeline = self.pipeline
        map_manager = pipeline.map_manager
        room_segmenter = pipeline.room_segmenter
        floor_tracker = pipeline.floor_tracker
        pipeline_state = state.get("pipeline") or {}
        self._restore_named_state(
            pipeline,
            pipeline_state,
            (
                "robot_trajectory",
                "floor_trajectories",
                "room_frame_history",
                "pose_cache",
                "room_metadata_cache",
                "room_topology_cache",
                "room_exit_buffer",
                "global_update_history",
                "current_room_id",
                "last_room_id",
            ),
        )
        pipeline.visited_room_ids = set(
            int(item)
            for item in pipeline_state.get("visited_room_ids", [])
        )

        map_state = state.get("map_manager") or {}
        self._restore_named_state(
            map_manager,
            map_state,
            (
                "logodds",
                "door_map",
                "occ_bin",
                "free_bin",
                "room_id_map",
                "door_hits",
                "floor_maps",
                "room_registry",
                "room_geometry",
                "current_floor_idx",
                "current_room_id",
            ),
        )
        expected_shape = (int(map_manager.map_h), int(map_manager.map_w))
        for name in (
            "logodds",
            "door_map",
            "occ_bin",
            "free_bin",
            "room_id_map",
            "door_hits",
        ):
            value = np.asarray(getattr(map_manager, name))
            if value.shape != expected_shape:
                raise RuntimeError(
                    f"Restored {name} has shape {value.shape}, expected {expected_shape}"
                )

        room_state = state.get("room_segmenter") or {}
        self._restore_named_state(
            room_segmenter,
            room_state,
            (
                "global_id_grid",
                "global_conf_grid",
                "floor_states",
                "colors",
                "current_floor_idx",
                "current_room_id",
                "next_global_room_id",
            ),
        )
        if np.asarray(room_segmenter.global_id_grid).shape != expected_shape:
            raise RuntimeError("Restored room-segmenter ID grid has the wrong shape")
        if np.asarray(room_segmenter.global_conf_grid).shape != expected_shape:
            raise RuntimeError("Restored room-segmenter confidence grid has the wrong shape")

        floor_state = state.get("floor_tracker") or {}
        self._restore_named_state(
            floor_tracker,
            floor_state,
            (
                "current_floor_id",
                "current_room_id",
                "base_height",
                "pending_data_cache",
            ),
        )
        floor_tracker.height_buffer.clear()
        floor_tracker.height_buffer.extend(floor_state.get("height_buffer", []))
        floor_tracker.id_buffer.clear()
        floor_tracker.id_buffer.extend(floor_state.get("id_buffer", []))
        floor_tracker.floors_data.clear()
        for floor_id, value in (floor_state.get("floors_data") or {}).items():
            floor_tracker.floors_data[int(floor_id)] = {
                "heights": [float(item) for item in value.get("heights", [])],
                "rooms": set(str(item) for item in value.get("rooms", [])),
                "finalized_count": int(value.get("finalized_count", 0)),
            }
        if floor_state.get("next_floor_id") is None:
            if hasattr(floor_tracker, "next_floor_id"):
                delattr(floor_tracker, "next_floor_id")
        else:
            floor_tracker.next_floor_id = int(floor_state["next_floor_id"])
        logger.info(f"[INHerit-SG] restored complete builder state from {path}")

    def _restore_map_manager_from_saved_data(self, map_data):
        map_manager = getattr(self.pipeline, "map_manager", None)
        if map_manager is None or not isinstance(map_data, dict):
            return

        if "resolution" in map_data:
            map_manager.resolution = float(map_data["resolution"])
        if "min_x" in map_data:
            map_manager.min_x = float(map_data["min_x"])
        if "min_z" in map_data:
            map_manager.min_z = float(map_data["min_z"])

        room_mask = map_data.get("room_mask")
        if room_mask is not None:
            map_manager.room_id_map = np.asarray(room_mask, dtype=np.int32).copy()
            map_manager.map_h, map_manager.map_w = map_manager.room_id_map.shape[:2]
        for attr, dtype in (("occ_bin", np.uint8), ("free_bin", np.uint8), ("door_map", np.uint8)):
            value = map_data.get(attr)
            if value is not None:
                setattr(map_manager, attr, np.asarray(value, dtype=dtype).copy())
        if hasattr(map_manager, "logodds") and getattr(map_manager, "occ_bin", None) is not None:
            occ = np.asarray(map_manager.occ_bin) > 0
            free = np.asarray(map_manager.free_bin) > 0
            logodds = np.zeros_like(map_manager.occ_bin, dtype=np.float32)
            logodds[occ] = float(getattr(map_manager, "L_OCC", 0.85))
            logodds[free & ~occ] = float(getattr(map_manager, "L_FREE", -0.25))
            map_manager.logodds = logodds
        if hasattr(map_manager, "door_hits") and getattr(map_manager, "door_map", None) is not None:
            map_manager.door_hits = np.asarray(map_manager.door_map > 0, dtype=np.uint16)

    def _restore_label_counters_from_memory(self):
        counters = {}
        for item in getattr(self.pipeline, "memory_kb", []):
            node_id = str(item.get("node_id") or "")
            if "_" not in node_id:
                continue
            prefix, suffix = node_id.rsplit("_", 1)
            if prefix and suffix.isdigit():
                # The semantic label may be expanded during re-identification
                # (for example node ``sofa_0`` can carry label
                # ``sofa armchair``).  The ID allocator is keyed by the
                # original node prefix, not by that mutable label.
                counters[prefix] = max(
                    counters.get(prefix, 0),
                    int(suffix) + 1,
                )
        if counters:
            self.pipeline.label_counters = counters

    def _restore_frame_index_from_preload(self):
        manifest_path = os.path.join(self.preload_dir, "episode_manifest.jsonl")
        max_frame = 0
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    max_frame = max(max_frame, int(row.get("end_frame_idx_exclusive") or 0))
        if max_frame <= 0:
            frames_dir = os.path.join(self.preload_dir, "frames", "rgb")
            if os.path.isdir(frames_dir):
                for name in os.listdir(frames_dir):
                    stem, ext = os.path.splitext(name)
                    if ext.lower() == ".png" and stem.isdigit():
                        max_frame = max(max_frame, int(stem) + 1)
        self.frame_idx = max(self.frame_idx, max_frame)

    def _preload_finalized_episode_count(self, src_dir):
        raw = os.environ.get("INHERIT_SG_PRELOAD_FINALIZED_EPISODES")
        if raw is not None and raw.strip():
            return int(raw)
        manifest_path = os.path.join(src_dir, "episode_manifest.jsonl")
        if not os.path.isfile(manifest_path):
            return 1
        with open(manifest_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip()) or 1

    def _start_async_worker(self):
        self._frame_queue = queue.Queue(maxsize=self.async_queue_size)
        self._worker_thread = threading.Thread(
            target=self._async_worker_loop,
            name="inherit-sg-frame-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def _stop_async_worker(self, drain=True):
        worker = self._worker_thread
        frame_queue = self._frame_queue
        if worker is None or frame_queue is None:
            return

        if not drain:
            while True:
                try:
                    frame_queue.get_nowait()
                    frame_queue.task_done()
                except queue.Empty:
                    break

        frame_queue.put(self._worker_stop)
        worker.join()
        self._worker_thread = None
        self._frame_queue = None

    def _async_worker_loop(self):
        while True:
            item = self._frame_queue.get()
            try:
                if item is self._worker_stop:
                    return
                with self.gpu_inference_lock:
                    self._process_frame_payload(item)
            except Exception as exc:
                idx = item.get("idx", "unknown") if isinstance(item, dict) else "unknown"
                self._async_errors.append(f"frame {idx}: {type(exc).__name__}: {exc}")
                logger.warning(f"[INHerit-SG] async frame {idx} failed: {exc}\n{traceback.format_exc()}")
                if self.fail_closed:
                    return
            finally:
                self._frame_queue.task_done()

    def _enqueue_frame(self, frame):
        if self._frame_queue is None:
            return

        if self.async_drop_policy == "block":
            self._frame_queue.put(frame)
            return

        try:
            self._frame_queue.put_nowait(frame)
            return
        except queue.Full:
            pass

        if self.async_drop_policy == "drop_newest":
            self._dropped_frames += 1
            return

        try:
            self._frame_queue.get_nowait()
            self._frame_queue.task_done()
            self._dropped_frames += 1
        except queue.Empty:
            pass

        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            self._dropped_frames += 1

    def _build_frame_payload(self, obs_dict, idx: int, step: int):
        agent_position = np.asarray(obs_dict["position"], dtype=np.float32).reshape(3)
        heading = np.asarray(obs_dict["heading"], dtype=np.float32).reshape(-1)
        sensor_offset = self._depth_sensor_offset()
        frame = {
            "idx": int(idx),
            "timestamp": float(step),
            "rgb_bgr": self._rgb_bgr(obs_dict["rgb"]).copy(),
            "depth_mm": self._depth_uint16_mm(obs_dict["depth"]).copy(),
            "pose": habitat_cam_pose_matrix(
                obs_dict,
                sensor_offset_xyz=sensor_offset,
            ).copy(),
            "agent_position": agent_position.copy(),
            "agent_heading_rad": float(heading[0]) if heading.size else 0.0,
            "sensor_offset_xyz": np.asarray(
                sensor_offset, dtype=np.float32
            ).copy(),
        }
        self._save_offline_rgbd_frame(frame)
        return frame

    def _prepare_offline_rgbd_capture(self):
        self._offline_rgbd_frame_indices = []
        if not self.save_offline_rgbd:
            return
        try:
            configured = self.offline_rgbd_dir_template.format(
                episode_id=str(self._current_runtime_episode_id or "unknown"),
                episode_name=str(self._current_episode_name or "episode"),
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "INHERIT_SG_OFFLINE_RGBD_DIR supports only the "
                "{episode_id} and {episode_name} placeholders"
            ) from exc
        root = os.path.abspath(configured)
        if os.path.isdir(root) and os.listdir(root):
            raise FileExistsError(
                f"Refusing to replace existing offline RGB-D capture: {root}"
            )
        for name in ("rgb", "depth", "pose"):
            os.makedirs(os.path.join(root, name), exist_ok=True)
        self.offline_rgbd_dir = root

    def _save_offline_rgbd_frame(self, frame):
        if not self.save_offline_rgbd:
            return
        frame_idx = int(frame["idx"])
        stem = f"{frame_idx:06d}"
        rgb_path = os.path.join(self.offline_rgbd_dir, "rgb", f"{stem}.png")
        depth_path = os.path.join(
            self.offline_rgbd_dir, "depth", f"{stem}.png"
        )
        pose_path = os.path.join(
            self.offline_rgbd_dir, "pose", f"{stem}.json"
        )
        for path in (rgb_path, depth_path, pose_path):
            if os.path.exists(path):
                raise FileExistsError(
                    f"Offline RGB-D frame would overwrite {path}"
                )
        if not cv2.imwrite(rgb_path, np.asarray(frame["rgb_bgr"], dtype=np.uint8)):
            raise RuntimeError(f"Failed to save offline RGB frame: {rgb_path}")
        if not cv2.imwrite(
            depth_path, np.asarray(frame["depth_mm"], dtype=np.uint16)
        ):
            raise RuntimeError(
                f"Failed to save offline depth frame: {depth_path}"
            )
        pose = np.asarray(frame["pose"], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(
                f"Offline RGB-D frame {frame_idx} has an invalid camera pose"
            )
        pose_payload = {
            "schema_version": "inherit_sg_offline_rgbd_pose_v1",
            "frame_idx": frame_idx,
            "timestamp": float(frame["timestamp"]),
            "camera_to_world": pose.tolist(),
            "coordinate_frame": "habitat_world",
            "camera_pose_includes_sensor_extrinsic": True,
            "sensor_offset_xyz": np.asarray(
                frame.get("sensor_offset_xyz", self._depth_sensor_offset()),
                dtype=np.float64,
            ).reshape(3).tolist(),
            "agent_position": np.asarray(
                frame["agent_position"], dtype=np.float64
            ).reshape(3).tolist(),
            "agent_heading_rad": float(frame["agent_heading_rad"]),
        }
        with open(pose_path, "x", encoding="utf-8") as handle:
            json.dump(pose_payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        self._offline_rgbd_frame_indices.append(frame_idx)

    def _write_offline_rgbd_manifest(self):
        if not self.save_offline_rgbd:
            return
        indices = list(self._offline_rgbd_frame_indices)
        if not indices:
            raise RuntimeError("Offline RGB-D capture contains no frames")
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise RuntimeError(
                "Offline RGB-D capture frame indices are not contiguous"
            )
        intrinsics = self._camera_intrinsics()
        payload = {
            "schema_version": "inherit_sg_offline_rgbd_capture_v1",
            "episode_name": self._current_episode_name,
            "frame_count": len(indices),
            "start_frame_idx": indices[0],
            "end_frame_idx_exclusive": indices[-1] + 1,
            "frame_indices": indices,
            "modalities": {
                "rgb": {
                    "directory": "rgb",
                    "encoding": "png_srgb_uint8",
                },
                "depth": {
                    "directory": "depth",
                    "encoding": "png_uint16",
                    "unit": "millimeter",
                    "invalid_value": 0,
                },
                "pose": {
                    "directory": "pose",
                    "encoding": "json",
                    "transform": "camera_to_world",
                    "coordinate_frame": "habitat_world",
                },
            },
            "camera": {
                "width": self.width,
                "height": self.height,
                "fx": float(intrinsics[0, 0]),
                "fy": float(intrinsics[1, 1]),
                "cx": float(intrinsics[0, 2]),
                "cy": float(intrinsics[1, 2]),
                "depth_scale": 1000.0,
                "sensor_offset_xyz": list(self._depth_sensor_offset()),
                "camera_pose_includes_sensor_extrinsic": True,
            },
            "capture_scope": "every_rgbd_observation_before_keyframe_filter",
            "frame_index_semantics": "global_inherit_sg_frame_idx",
            "terminal_pose_without_rgbd_is_excluded": True,
        }
        path = os.path.join(self.offline_rgbd_dir, "capture_manifest.json")
        with open(path, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")

    def _is_keyframe_payload(self, frame):
        if self.pipeline is None:
            return False
        try:
            return bool(self.pipeline.detect_keyframe_only(frame["idx"], frame["rgb_bgr"]))
        except Exception as exc:
            logger.warning(
                f"[INHerit-SG] keyframe filter failed for frame {frame.get('idx')}; "
                f"processing frame anyway: {exc}\n{traceback.format_exc()}"
            )
            return True

    def _process_frame_payload(self, frame):
        if self.pipeline is None:
            return
        continuation_snapshot = self._snapshot_preloaded_object_state()
        protected_paths = self._external_frame_paths(frame)
        original_imwrite = cv2.imwrite

        if protected_paths:
            # The reviewed V7 builder expects filesystem paths and calls
            # cv2.imwrite before reading them back.  During offline rebuilds
            # those exact files already exist under the immutable input tree.
            # Suppress only those two redundant writes; all other writes
            # continue through the original OpenCV implementation.
            def external_frame_imwrite(path, image, *args):
                normalized = os.path.realpath(os.fspath(path))
                if normalized in protected_paths:
                    return True
                return original_imwrite(path, image, *args)

            cv2.imwrite = external_frame_imwrite
        try:
            processed = self.pipeline.process_confirmed_keyframe(
                frame["idx"],
                frame["rgb_bgr"],
                frame["depth_mm"],
                frame["pose"],
                timestamp=frame["timestamp"],
            )
        finally:
            if protected_paths:
                cv2.imwrite = original_imwrite
        if processed is False:
            raise RuntimeError(
                f"INHerit-SG rejected confirmed keyframe {frame['idx']}"
            )
        self._save_confirmed_keyframe_pose(frame)
        self._episode_confirmed_keyframes += 1
        if self.debug:
            self._centralize_debug_merge_frame(frame["idx"])
        else:
            self._discard_frame_diagnostics(frame["idx"])
        self._guard_preloaded_object_updates(
            continuation_snapshot,
            frame_idx=int(frame["idx"]),
        )
        self._attach_best_observation_waypoints(frame)
        self._save_map_snapshot(frame["idx"])

    def _save_confirmed_keyframe_pose(self, frame):
        """Complete V7's accepted keyframe RGB-D package with its exact pose.

        The native V7 builder writes RGB and depth only after a frame passes
        DINO keyframe selection. Persisting the pose at the same commit point
        makes offline replay exact without changing mapping or Re-ID state.
        """
        if not self.save_confirmed_keyframes or self.pipeline is None:
            return

        frame_idx = int(frame["idx"])
        stem = f"{frame_idx:06d}"
        rgb_path = os.path.join(self.pipeline.rgb_dir, f"{stem}.png")
        depth_path = os.path.join(self.pipeline.depth_dir, f"{stem}.png")
        pose_dir = self.pipeline.pose_dir
        pose_path = os.path.join(pose_dir, f"{stem}.txt")
        if not os.path.isfile(rgb_path) or not os.path.isfile(depth_path):
            raise FileNotFoundError(
                "V7 confirmed-keyframe RGB-D files are incomplete for "
                f"frame {frame_idx}: rgb={rgb_path}, depth={depth_path}"
            )

        pose = np.asarray(frame["pose"], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(
                f"Confirmed keyframe {frame_idx} has an invalid camera pose"
            )
        os.makedirs(pose_dir, exist_ok=True)
        if os.path.exists(pose_path):
            existing = np.loadtxt(pose_path, dtype=np.float64)
            if existing.shape != (4, 4) or not np.allclose(
                existing, pose, rtol=0.0, atol=1e-7
            ):
                raise RuntimeError(
                    f"Refusing conflicting pose for keyframe {frame_idx}: "
                    f"{pose_path}"
                )
        else:
            temporary = f"{pose_path}.tmp-{os.getpid()}"
            with open(temporary, "x", encoding="utf-8") as handle:
                np.savetxt(handle, pose, fmt="%.10f")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, pose_path)

        manifest_path = os.path.join(
            self.pipeline.trajectory_data_dir,
            "confirmed_keyframes.jsonl",
        )
        row = {
            "schema_version": "inherit_sg_v7_confirmed_keyframe_v1",
            "frame_idx": frame_idx,
            "timestamp": float(frame["timestamp"]),
            "rgb": f"rgb/{stem}.png",
            "depth": f"depth/{stem}.png",
            "pose": f"pose/{stem}.txt",
            "camera_to_world": pose.tolist(),
            "coordinate_frame": "habitat_world",
            "agent_position": np.asarray(
                frame["agent_position"], dtype=np.float64
            ).reshape(3).tolist(),
            "agent_heading_rad": float(frame["agent_heading_rad"]),
            "episode_name": self._current_episode_name,
            "commit_point": "after_process_confirmed_keyframe_success",
        }
        with open(manifest_path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(self._to_builtin(row), ensure_ascii=True) + "\n"
            )

    def _external_frame_paths(self, frame):
        if not self.external_frame_root:
            return set()
        source_paths = []
        for key in ("source_rgb_path", "source_depth_path"):
            raw = frame.get(key)
            if not raw:
                raise RuntimeError(
                    f"External-frame mode requires {key} for frame "
                    f"{frame.get('idx')}"
                )
            path = os.path.realpath(os.fspath(raw))
            try:
                inside_root = (
                    os.path.commonpath(
                        [self.external_frame_root, path]
                    )
                    == self.external_frame_root
                )
            except ValueError:
                inside_root = False
            if not inside_root or not os.path.isfile(path):
                raise RuntimeError(
                    f"External frame is outside the immutable input root: "
                    f"{path}"
                )
            source_paths.append(path)

        expected_rgb = os.path.realpath(
            os.path.join(
                self.pipeline.rgb_dir,
                f"{int(frame['idx']):06d}.png",
            )
        )
        expected_depth = os.path.realpath(
            os.path.join(
                self.pipeline.depth_dir,
                f"{int(frame['idx']):06d}.png",
            )
        )
        if source_paths != [expected_rgb, expected_depth]:
            raise RuntimeError(
                "External frame paths do not match the V7 builder filename "
                f"contract for frame {frame['idx']}: "
                f"{source_paths!r} != {[expected_rgb, expected_depth]!r}"
            )
        return set(source_paths)

    def _centralize_debug_merge_frame(self, frame_idx):
        """Move V7's per-frame merge visualization into one ordered folder."""
        if self.pipeline is None:
            return
        frame_idx = int(frame_idx)
        nested = os.path.join(
            self.pipeline.run_dir,
            "objects",
            f"frame_{frame_idx:04d}",
            "debug_merge",
        )
        if not os.path.isdir(nested):
            return
        images = sorted(
            os.path.join(nested, name)
            for name in os.listdir(nested)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if len(images) != 1:
            raise RuntimeError(
                f"Expected one V7 debug_merge image for frame {frame_idx}, "
                f"found {images}"
            )
        destination_dir = os.path.join(
            self.pipeline.run_dir, "debug_merge"
        )
        os.makedirs(destination_dir, exist_ok=True)
        destination = os.path.join(
            destination_dir, f"frame_{frame_idx:06d}.jpg"
        )
        if os.path.exists(destination):
            raise FileExistsError(
                f"Refusing to replace debug merge frame: {destination}"
            )
        os.replace(images[0], destination)
        if os.listdir(nested):
            raise RuntimeError(
                f"Unexpected files remain in nested debug directory: {nested}"
            )
        os.rmdir(nested)

    def _discard_frame_diagnostics(self, frame_idx):
        """Discard per-keyframe intermediates after the builder consumed them."""
        if self.pipeline is None:
            return
        frame_dir = os.path.join(
            self.pipeline.run_dir,
            "objects",
            f"frame_{int(frame_idx):04d}",
        )
        if not os.path.isdir(frame_dir):
            return

        for directory in (
            "debug_merge",
            "debug_overlaps",
            os.path.join("cropped_objects", "dropped"),
            os.path.join("cropped_objects", "containment_dropped"),
            os.path.join("cropped_objects", "small_dropped"),
        ):
            path = os.path.join(frame_dir, directory)
            if os.path.isdir(path):
                shutil.rmtree(path)

        for name in os.listdir(frame_dir):
            if name.startswith("pred_frame_") and name.endswith(".json"):
                os.unlink(os.path.join(frame_dir, name))

        cropped_root = os.path.join(frame_dir, "cropped_objects")
        kept_dir = os.path.join(cropped_root, "kept")
        if os.path.isdir(kept_dir) and not os.listdir(kept_dir):
            os.rmdir(kept_dir)
        if os.path.isdir(cropped_root) and not os.listdir(cropped_root):
            os.rmdir(cropped_root)
        if os.path.isdir(frame_dir) and not os.listdir(frame_dir):
            os.rmdir(frame_dir)

    def _prune_nonessential_output_artifacts(self):
        """Enforce a compact production checkpoint before it is published."""
        if self.pipeline is None or self.debug:
            return
        run_dir = os.path.realpath(self.pipeline.run_dir)
        if not os.path.isdir(run_dir):
            return

        for name in os.listdir(run_dir):
            path = os.path.join(run_dir, name)
            if os.path.isdir(path) and (
                (
                    name.startswith("debug_")
                    and not (
                        name == "debug_map_snapshots"
                        and self.save_map_snapshots
                    )
                )
                or name == "trajectory_maps"
            ):
                shutil.rmtree(path)

        objects_dir = os.path.join(run_dir, "objects")
        if os.path.isdir(objects_dir):
            for name in os.listdir(objects_dir):
                frame_dir = os.path.join(objects_dir, name)
                if os.path.isdir(frame_dir) and name.startswith("frame_"):
                    suffix = name[len("frame_") :]
                    if suffix.isdigit():
                        self._discard_frame_diagnostics(int(suffix))
            if not os.listdir(objects_dir):
                os.rmdir(objects_dir)

        run_log = os.path.join(run_dir, "run_log.log")
        if os.path.isfile(run_log) and os.path.getsize(run_log) == 0:
            os.unlink(run_log)

    def _snapshot_preloaded_object_state(self):
        """Capture M0 identities before a continuation frame mutates the graph."""
        pipeline = self.pipeline
        if (
            pipeline is None
            or not self._preloaded_session
            or self.preload_query_only
            or not self._preloaded_object_baselines
        ):
            return None
        node_ids = set(self._preloaded_object_baselines)
        objects = {
            str(item.get("node_id")): copy.deepcopy(item)
            for item in getattr(pipeline, "memory_kb", [])
            if str(item.get("node_id")) in node_ids
        }
        graph = getattr(pipeline, "spatial_graph", None)
        graph_nodes = {}
        graph_edges = {}
        if graph is not None:
            graph_nodes = {
                node_id: copy.deepcopy(graph.nodes[node_id])
                for node_id in node_ids
                if graph.has_node(node_id)
            }
            graph_edges = {
                (source, target): copy.deepcopy(data)
                for source, target, data in graph.edges(data=True)
                if source in node_ids or target in node_ids
            }
        return {
            "objects": objects,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
        }

    def _guard_preloaded_object_updates(self, snapshot, *, frame_idx: int):
        """Reject a frame-level merge that aliases an M0 identity to another object."""
        pipeline = self.pipeline
        if pipeline is None or not snapshot:
            return set()

        rejected = set()
        objects_before = snapshot["objects"]
        for index, item in enumerate(getattr(pipeline, "memory_kb", [])):
            node_id = str(item.get("node_id"))
            old = objects_before.get(node_id)
            baseline = self._preloaded_object_baselines.get(node_id)
            if old is None or baseline is None:
                continue
            try:
                old_position = np.asarray(
                    old.get("global_pos_3d"), dtype=np.float64
                ).reshape(3)
                new_position = np.asarray(
                    item.get("global_pos_3d"), dtype=np.float64
                ).reshape(3)
                baseline_position = np.asarray(
                    baseline.get("global_pos_3d"), dtype=np.float64
                ).reshape(3)
            except (TypeError, ValueError):
                continue
            if (
                not np.all(np.isfinite(old_position))
                or not np.all(np.isfinite(new_position))
                or not np.all(np.isfinite(baseline_position))
                or np.allclose(old_position, new_position, rtol=0.0, atol=1.0e-9)
            ):
                continue

            # The upstream builder updates matched centroids as
            # new = 0.7 * old + 0.3 * observation. Recover that observation
            # and compare it with the immutable M0 identity, rather than with
            # the already-moving online centroid.
            observation_position = old_position + (
                new_position - old_position
            ) / 0.3
            observation_distance = float(
                np.linalg.norm(observation_position - baseline_position)
            )
            if observation_distance <= self.preloaded_object_match_guard_m:
                continue

            pipeline.memory_kb[index] = copy.deepcopy(old)
            rejected.add(node_id)
            logger.warning(
                "[INHerit-SG] rejected continuation re-ID for preloaded "
                f"{node_id} at frame {frame_idx}: inferred observation is "
                f"{observation_distance:.3f}m from its immutable M0 position "
                f"(limit={self.preloaded_object_match_guard_m:.3f}m)"
            )

        if not rejected:
            return rejected

        self._preloaded_object_guard_rejections += len(rejected)
        self._episode_preloaded_object_guard_rejections += len(rejected)
        graph = getattr(pipeline, "spatial_graph", None)
        if graph is not None:
            old_edges = snapshot["graph_edges"]
            for node_id in rejected:
                current_edges = set(graph.in_edges(node_id)) | set(
                    graph.out_edges(node_id)
                )
                for edge in current_edges:
                    graph.remove_edge(*edge)
                for (source, target), data in old_edges.items():
                    if source == node_id or target == node_id:
                        graph.add_edge(source, target, **copy.deepcopy(data))
                if node_id in snapshot["graph_nodes"]:
                    if not graph.has_node(node_id):
                        graph.add_node(node_id)
                    graph.nodes[node_id].clear()
                    graph.nodes[node_id].update(
                        copy.deepcopy(snapshot["graph_nodes"][node_id])
                    )

        observations = getattr(
            pipeline,
            "_inherit_sg_frame_observations",
            None,
        )
        if observations is not None:
            pipeline._inherit_sg_frame_observations = [
                observation
                for observation in observations
                if str(observation.get("node_id")) not in rejected
            ]
        return rejected

    def _attach_best_observation_waypoints(self, frame):
        """Select and store a navigation observation without changing INHerit-SG's best frame."""
        pipeline = self.pipeline
        if pipeline is None:
            return

        rgb_dir = getattr(pipeline, "rgb_dir", None)
        filename_prefix = str(getattr(pipeline, "filename_prefix", ""))
        if not rgb_dir:
            return
        expected_rgb_path = os.path.normpath(
            os.path.join(rgb_dir, f"{filename_prefix}{int(frame['idx']):06d}.png")
        )

        pose = np.asarray(frame["pose"], dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            logger.warning(f"[INHerit-SG] invalid observation pose for frame {frame.get('idx')}")
            return

        position = np.asarray(frame.get("agent_position", pose[:3, 3]), dtype=np.float64).reshape(3)
        heading_rad = float(
            frame.get("agent_heading_rad", np.arctan2(pose[0, 2], pose[0, 0]))
        )
        rotation_xyzw = [0.0, float(np.sin(heading_rad / 2.0)), 0.0, float(np.cos(heading_rad / 2.0))]
        forward_world = (-pose[:3, 2]).tolist()
        runtime_episode_id = str(
            self._current_runtime_episode_id
            or self._current_episode_name
            or ""
        )
        source_episode_id = (
            str(self._current_source_episode_id)
            if self._current_source_episode_id is not None
            else None
        )
        updated = 0

        observations = getattr(pipeline, "_inherit_sg_frame_observations", None)
        if observations is None:
            observations = []
            for item in getattr(pipeline, "memory_kb", []):
                if os.path.normpath(str(item.get("full_image_path") or "")) == expected_rgb_path:
                    observations.append(
                        {
                            "node_id": item.get("node_id"),
                            "bbox": item.get("bbox"),
                            "rgb_path": item.get("full_image_path"),
                            "object_position": item.get("global_pos_3d"),
                        }
                    )

        memory_by_node = {
            item.get("node_id"): item
            for item in getattr(pipeline, "memory_kb", [])
            if item.get("node_id") is not None
        }
        for observation in observations:
            node_id = observation.get("node_id")
            item = memory_by_node.get(node_id)
            if item is None:
                continue

            bbox = observation.get("bbox")
            pixel_area = self._bbox_pixel_area(bbox)
            selection_score = self._observation_view_score(bbox)
            selected_rgb_path = observation.get("rgb_path") or expected_rgb_path
            object_position = observation.get("object_position")
            observation_distance_m = None
            if object_position is not None:
                try:
                    object_position = np.asarray(object_position, dtype=np.float64).reshape(-1)[:3]
                    if object_position.size == 3 and np.all(np.isfinite(object_position)):
                        observation_distance_m = float(np.linalg.norm(object_position - position))
                except (TypeError, ValueError):
                    pass

            trajectory_node_id = None
            if self.world_memory is not None:
                trajectory_node_id = self.world_memory.trajectory.node_for_frame(
                    int(frame["idx"])
                )

            best_observation = {
                "frame_idx": int(frame["idx"]),
                "timestamp": float(frame["timestamp"]),
                "rgb_path": str(selected_rgb_path),
                "bbox_xyxy": self._to_builtin(bbox),
                "pixel_area": pixel_area,
                "pixel_area_fraction": (
                    float(pixel_area) / float(self.width * self.height) if pixel_area is not None else None
                ),
                "observation_distance_m": observation_distance_m,
                "selection_score": selection_score,
                "agent_position": position.tolist(),
                "agent_heading_rad": heading_rad,
                "agent_rotation_xyzw": rotation_xyzw,
                "forward_world": forward_world,
                "camera_to_world": pose.tolist(),
                "coordinate_frame": "habitat_world",
                "trajectory_node_id": trajectory_node_id,
                "runtime_episode_id": runtime_episode_id,
                "source_episode_id": source_episode_id,
                "selection_policy": "minimum_weighted_center_area_observation_score",
                "selection_weights": {
                    "center_offset": float(self.observation_center_weight),
                    "bbox_area": float(self.observation_area_weight),
                },
            }
            waypoint = {
                "position": position.tolist(),
                "heading_rad": heading_rad,
                "rotation_xyzw": rotation_xyzw,
                "frame_idx": int(frame["idx"]),
                "source": "best_observation_pose",
                "coordinate_frame": "habitat_world",
                "trajectory_node_id": trajectory_node_id,
                "runtime_episode_id": runtime_episode_id,
                "source_episode_id": source_episode_id,
            }

            changes = record_object_observation(
                item,
                runtime_episode_id=runtime_episode_id,
                source_episode_id=source_episode_id,
                observation=best_observation,
                waypoint=waypoint,
                selection_score=selection_score,
            )
            sync_object_provenance_to_graph(
                getattr(pipeline, "spatial_graph", None),
                item,
            )
            if changes["episode_updated"] or changes["global_updated"]:
                updated += 1

        if updated:
            logger.info(
                f"[INHerit-SG] updated independent best observations for {updated} "
                f"object node(s) at frame {int(frame['idx'])}"
            )

    @staticmethod
    def _ordered_unique_strings(values):
        return ordered_unique_episode_ids(values)

    def _observation_view_score(self, bbox):
        return observation_selection_score(
            bbox,
            self.width,
            self.height,
            center_weight=self.observation_center_weight,
            area_weight=self.observation_area_weight,
        )

    @staticmethod
    def _bbox_pixel_area(bbox):
        return bbox_pixel_area(bbox)

    def _get_shared_models(self, sg_builder, cfg):
        if self.shared_models is not None:
            return self.shared_models

        from sentence_transformers import SentenceTransformer

        text_path = os.environ.get("INHERIT_SG_TEXT_ENCODER", "/data2/shizk/all-MiniLM-L6-v2")
        gpu_config = cfg.get("multi_gpu", {})
        sam3_device = gpu_config.get("sam3_device", self.device)
        dinov3_device = gpu_config.get("dinov3_device", self.device)
        text_encoder_device = gpu_config.get("text_encoder_device", self.device)
        try:
            text_encoder = SentenceTransformer(text_path, device=text_encoder_device)
        except Exception as exc:
            logger.warning(
                "[INHerit-SG] text encoder loader failed; using lightweight "
                f"fallback encoder instead: {exc}"
            )
            text_encoder = self._make_lightweight_text_encoder()
        self.shared_models = {
            "dinov3": sg_builder.load_feature_extractor(cfg, dinov3_device),
            "sam3": sg_builder.Sam3LocalClient(cfg, device=sam3_device),
            "text_encoder": text_encoder,
        }
        return self.shared_models

    def _patch_builder_feature_extractor(self, sg_builder):
        if getattr(sg_builder, "_inherit_sg_feature_loader_patched", False):
            return

        original_loader = sg_builder.load_feature_extractor
        original_extract = getattr(sg_builder, "extract_feature", None)

        def load_feature_extractor_with_fallback(config, device):
            cfg = config.get("feature_extractor", {}) if isinstance(config, dict) else {}
            if self._should_use_torchhub_dinov3(cfg):
                try:
                    import torch

                    repo_dir = cfg.get("repo_dir")
                    weights_path = self._resolve_dinov3_weights_path(cfg.get("weights_path"))
                    model_name = cfg.get("model_name", "dinov3_vitl16")
                    logger.info(
                        "[INHerit-SG] loading cache-free local DINOv3 via "
                        "torch.hub: "
                        f"repo={repo_dir}, model={model_name}, weights={weights_path}"
                    )
                    # DINOv3's hub wrapper converts a local checkpoint path to
                    # ``file://`` but still delegates to
                    # ``load_state_dict_from_url``.  That helper creates the
                    # global Torch hub cache before reading the local file,
                    # which fails for formal warm runs executed as an
                    # unprivileged UID (their implicit cache root is
                    # ``/.cache``).  Construct the reviewed local model
                    # without pretrained weights and load the immutable local
                    # checkpoint directly instead.  This is both offline and
                    # independent of HOME/TORCH_HOME permissions.
                    model = torch.hub.load(
                        repo_or_dir=repo_dir,
                        model=model_name,
                        source="local",
                        pretrained=False,
                    )
                    try:
                        state_dict = torch.load(
                            weights_path,
                            map_location="cpu",
                            weights_only=True,
                        )
                    except TypeError:  # torch versions before weights_only
                        state_dict = torch.load(weights_path, map_location="cpu")
                    if (
                        isinstance(state_dict, dict)
                        and isinstance(state_dict.get("state_dict"), dict)
                    ):
                        state_dict = state_dict["state_dict"]
                    model.load_state_dict(state_dict, strict=True)
                    model.to(device).eval()
                    model._inherit_sg_feature_backend = "torchhub_dinov3"
                    return model
                except Exception as exc:
                    logger.warning(
                        "[INHerit-SG] torch.hub DINOv3 loader failed; trying reference "
                        f"loader next: {exc}"
                    )
            try:
                return original_loader(config, device)
            except Exception as exc:
                if self.fail_closed:
                    raise RuntimeError(
                        "Full DINOv3 feature extractor is required when "
                        "INHERIT_SG_FAIL_CLOSED=1"
                    ) from exc
                logger.warning(
                    "[INHerit-SG] DINOv3 loader failed; using lightweight local "
                    f"feature extractor instead: {exc}"
                )
                return self._make_lightweight_feature_model(device)

        def extract_feature_with_dinov3(image_pil, model, device):
            if (
                getattr(model, "_inherit_sg_feature_backend", None) == "torchhub_dinov3"
                or (hasattr(model, "forward_features") and not hasattr(model, "hf_processor"))
            ):
                return self._extract_torchhub_dinov3_feature(image_pil, model, device)
            if original_extract is None:
                raise AttributeError("INHerit-SG builder has no extract_feature function.")
            return original_extract(image_pil, model, device)

        sg_builder.load_feature_extractor = load_feature_extractor_with_fallback
        sg_builder.extract_feature = extract_feature_with_dinov3
        external_module = getattr(sg_builder, "_module", None)
        if external_module is not None:
            external_module.load_feature_extractor = load_feature_extractor_with_fallback
            external_module.extract_feature = extract_feature_with_dinov3
        for obj_name in ("TrajectoryPipeline",):
            obj = getattr(sg_builder, obj_name, None)
            for method_name in (
                "_is_keyframe",
                "detect_keyframe_only",
                "process_confirmed_keyframe",
                "_process_objects_for_frame",
            ):
                method = getattr(obj, method_name, None)
                globals_dict = getattr(method, "__globals__", None)
                if isinstance(globals_dict, dict) and "extract_feature" in globals_dict:
                    globals_dict["extract_feature"] = extract_feature_with_dinov3
        sg_builder._inherit_sg_feature_loader_patched = True

    def _should_use_torchhub_dinov3(self, cfg):
        if not isinstance(cfg, dict):
            return False
        kind = str(cfg.get("type", "")).lower()
        model_name = str(cfg.get("model_name", "")).lower()
        if "dinov3" not in kind and not model_name.startswith("dinov3"):
            return False
        repo_dir = cfg.get("repo_dir") or self._resolve_dinov3_repo_dir()
        weights_path = self._resolve_dinov3_weights_path(cfg.get("weights_path"))
        return bool(
            repo_dir
            and weights_path
            and os.path.isfile(os.path.join(repo_dir, "hubconf.py"))
            and os.path.isfile(weights_path)
        )

    def _resolve_dinov3_repo_dir(self):
        candidates = [
            os.environ.get("INHERIT_SG_DINOV3_REPO"),
            "/workspace/inherit_sg_dinov3_repo",
            "/data2/shizk/INHerit-SG/seg-rag/dinov3",
        ]
        for path in candidates:
            if path and os.path.isfile(os.path.join(path, "hubconf.py")):
                return path
        return None

    def _resolve_dinov3_weights_path(self, path):
        if not path:
            return path
        if os.path.isfile(path):
            return path
        if not os.path.isdir(path):
            return path
        preferred = [
            name
            for name in sorted(os.listdir(path))
            if name.startswith("dinov3_") and name.endswith(".pth")
        ]
        if preferred:
            return os.path.join(path, preferred[0])
        for name in sorted(os.listdir(path)):
            if name.endswith(".pth"):
                return os.path.join(path, name)
        return path

    def _extract_torchhub_dinov3_feature(self, image_pil, model, device):
        import torch
        import torch.nn.functional as F

        image = image_pil.convert("RGB")
        width, height = image.size
        short_edge = min(width, height)
        if short_edge <= 0:
            raise ValueError("empty image for DINOv3 feature extraction")
        scale = 518.0 / float(short_edge)
        resized = (max(518, int(round(width * scale))), max(518, int(round(height * scale))))
        image = image.resize(resized)
        left = max(0, (image.size[0] - 518) // 2)
        top = max(0, (image.size[1] - 518) // 2)
        image = image.crop((left, top, left + 518, top + 518))

        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        try:
            model_device = next(model.parameters()).device
        except Exception:
            model_device = torch.device(device)
        tensor = tensor.to(model_device)

        with torch.no_grad():
            outputs = model.forward_features(tensor)
        if isinstance(outputs, dict):
            features = outputs.get("x_norm_clstoken")
            if features is None:
                features = outputs.get("cls_token")
            if features is None:
                features = outputs.get("pooled")
        else:
            features = outputs
        if isinstance(features, (list, tuple)):
            features = features[0]
        if features is None:
            raise RuntimeError("DINOv3 forward_features did not return a usable feature tensor")
        if features.ndim > 2:
            features = F.adaptive_avg_pool1d(features.flatten(1).unsqueeze(1), 1024).squeeze(1)
        return features.detach().cpu().numpy().flatten()

    def _patch_builder_no_api_llm(self, sg_builder):
        if getattr(sg_builder, "_inherit_sg_no_api_llm_patched", False):
            return

        class NoApiLLMHandler:
            def __init__(self, *args, **kwargs):
                print("[INHerit-SG] API LLM disabled; using local no-op LLM handler.")

            def filter_object_tags(self, raw_tags):
                return list(dict.fromkeys(str(tag).lower().strip() for tag in raw_tags if str(tag).strip()))

            def generate_leaf_node_semantics(self, image_path, tag, spatial_info=""):
                return f"{tag} observed locally. {spatial_info}".strip()

            def detect_spatial_relationships(
                self,
                image_path,
                object_list,
                existing_relations,
                object_positions=None,
                method="vlm",
            ):
                if object_positions is None:
                    return []
                object_ids = [oid for oid in object_list if oid in object_positions]
                relations = []
                for i, src in enumerate(object_ids):
                    p_src = np.asarray(object_positions[src], dtype=np.float32)
                    for dst in object_ids[i + 1:]:
                        p_dst = np.asarray(object_positions[dst], dtype=np.float32)
                        dist = float(np.linalg.norm(p_src[[0, 2]] - p_dst[[0, 2]]))
                        if dist <= 0.75:
                            rel, weight = "next_to", 0.8
                        elif dist <= 1.5:
                            rel, weight = "near", 0.6
                        else:
                            continue
                        relations.append([src, dst, rel, weight])
                return relations

            def generate_spatial_cluster_summary(self, object_list, region_image_path=None):
                tags = [str(item.get("tag", "")).strip() for item in object_list if item.get("tag")]
                unique_tags = list(dict.fromkeys(tags))
                area_tag = unique_tags[0] if unique_tags else "area"
                summary = "Cluster containing " + ", ".join(unique_tags[:8]) if unique_tags else "Cluster of objects."
                return {"area_tag": area_tag, "summary": summary}

            def generate_multimodal_room_summary(self, image_path, area_summaries):
                tags = []
                for area in area_summaries or []:
                    tag = str(area.get("tag", "")).strip()
                    if tag and tag != "area":
                        tags.append(tag)
                room_label, confidence, evidence, scores = OnlineINHeritSGBuilder._infer_room_type_from_labels(tags)
                desc = "Room summarized without API calls."
                if evidence:
                    desc += " Evidence: " + ", ".join(evidence[:8]) + "."
                return {
                    "room_label": room_label.title() if room_label else "Room",
                    "description": desc,
                    "confidence": confidence,
                    "scores": scores,
                }

            def detect_room_objects_vlm(self, image_path, existing_labels):
                return []

            def analyze_trajectory_map_for_actions(self, map_image_path, current_floor_id):
                return {"trigger_global_update": False, "reasoning": "API calls disabled."}

        sg_builder.LLMHandler = NoApiLLMHandler
        sg_builder._inherit_sg_no_api_handler_cls = NoApiLLMHandler
        for obj_name in ("TrajectoryPipeline", "Sam3LocalClient"):
            obj = getattr(sg_builder, obj_name, None)
            init_fn = getattr(obj, "__init__", None)
            globals_dict = getattr(init_fn, "__globals__", None)
            if isinstance(globals_dict, dict) and "LLMHandler" in globals_dict:
                globals_dict["LLMHandler"] = NoApiLLMHandler
        sg_builder._inherit_sg_no_api_llm_patched = True

    def _make_lightweight_feature_model(self, device):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _Processor:
            def __call__(self, images, return_tensors="pt", **kwargs):
                img = np.asarray(images.convert("RGB"), dtype=np.float32) / 255.0
                tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                return {"pixel_values": (tensor - mean) / std}

        class _FeatureModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
                self.hf_processor = _Processor()

            def forward(self, pixel_values):
                pooled = F.adaptive_avg_pool2d(pixel_values, (8, 8)).flatten(1)
                mean = pixel_values.mean(dim=(2, 3))
                std = pixel_values.std(dim=(2, 3))
                features = torch.cat([pooled, mean, std], dim=1)
                features = F.normalize(features, dim=1)
                return types.SimpleNamespace(pooler_output=features)

        model = _FeatureModel().to(device)
        model.eval()
        return model

    def _make_lightweight_text_encoder(self):
        class _TextEncoder:
            def to(self, *args, **kwargs):
                return self

            def eval(self):
                return self

            def encode(self, sentences, *args, **kwargs):
                if isinstance(sentences, str):
                    count = 1
                else:
                    count = len(sentences)
                embeddings = np.zeros((count, 16), dtype=np.float32)
                if kwargs.get("convert_to_numpy", True):
                    return embeddings[0] if isinstance(sentences, str) else embeddings
                try:
                    import torch

                    tensor = torch.from_numpy(embeddings)
                    return tensor[0] if isinstance(sentences, str) else tensor
                except Exception:
                    return embeddings[0] if isinstance(sentences, str) else embeddings

        return _TextEncoder()

    def _patch_pipeline_compatibility(self):
        """Patch only streaming compatibility and debug output; keep reference SG finalization."""
        tracker = getattr(self.pipeline, "floor_tracker", None)
        if tracker is not None and not getattr(tracker, "_inherit_sg_update_patched", False):
            original_update = tracker.update

            def update_with_default(pose, current_room_node_id=None):
                return original_update(pose, current_room_node_id)

            tracker.update = update_with_default
            tracker._inherit_sg_update_patched = True

        pipeline = self.pipeline
        if pipeline is not None:
            self._ensure_no_api_llm(pipeline)
            if self.debug:
                self._patch_cleanup_debug(pipeline)

    def _ensure_no_api_llm(self, pipeline):
        """Keep the original finalize path usable when API LLM calls are disabled."""
        if pipeline is None or not getattr(pipeline, "config", {}).get("disable_llm", False):
            return
        if getattr(pipeline, "llm", None) is not None:
            return

        try:
            from . import builder as sg_builder

            handler_cls = getattr(sg_builder, "_inherit_sg_no_api_handler_cls", None)
            if handler_cls is None:
                self._patch_builder_no_api_llm(sg_builder)
                handler_cls = getattr(sg_builder, "_inherit_sg_no_api_handler_cls", None)
            if handler_cls is None:
                return
            pipeline.llm = handler_cls(pipeline.config.get("llm_config", {}))
            pipeline._inherit_sg_no_api_llm_injected = True
            logger.info("[INHerit-SG] injected local no-op LLM for no-API finalization")
        except Exception as exc:
            logger.warning(f"[INHerit-SG] failed to inject no-op LLM: {exc}")

    def _patch_pipeline_lightweight_graph(self):
        pipeline = self.pipeline
        if pipeline is None or getattr(pipeline, "_inherit_sg_lightweight_graph_patched", False):
            return

        if hasattr(pipeline, "llm"):
            delattr(pipeline, "llm")

        if self.debug:
            self._patch_cleanup_debug(pipeline)

        def lightweight_finalize_room_semantics(room_id, subset_objects=None, force_vlm_expansion=False):
            rid = self._normalize_room_id(room_id)
            if rid is None:
                return None
            self._assign_objects_to_room(pipeline, rid)
            room_node_id = f"room_{rid}"
            floor_tracker = getattr(pipeline, "floor_tracker", None)
            if floor_tracker is not None:
                floor_tracker.add_room(room_node_id)
            return room_node_id

        def lightweight_global_update(current_pose, frame_index):
            try:
                final_ids = pipeline.cleanup_and_optimize_rooms()
            except Exception as exc:
                logger.warning(f"[INHerit-SG] global room cleanup failed, using current map ids: {exc}")
                final_ids = self._valid_room_ids_from_map(pipeline)
            self._reassign_all_objects_to_rooms(pipeline, final_ids)
            pipeline._inherit_sg_final_room_ids = final_ids
            return final_ids

        def lightweight_finalize_algorithm():
            print("\n" + "=" * 50)
            print(">>> Finalizing Lightweight House-Room-Object SG...")
            print("=" * 50)
            try:
                final_ids = pipeline.cleanup_and_optimize_rooms()
            except Exception as exc:
                logger.warning(f"[INHerit-SG] room cleanup failed, using current map ids: {exc}")
                final_ids = self._valid_room_ids_from_map(pipeline)
            self._reassign_all_objects_to_rooms(pipeline, final_ids)
            pipeline._inherit_sg_final_room_ids = final_ids
            self._write_lightweight_semantic_graph(pipeline, final_ids)
            print(">>> Lightweight SG Finalization Complete.")

        def lightweight_save_all_data():
            import yaml

            os.makedirs(pipeline.run_dir, exist_ok=True)
            final_ids = getattr(pipeline, "_inherit_sg_final_room_ids", None)
            if final_ids is None:
                final_ids = self._valid_room_ids_from_map(pipeline)
                self._reassign_all_objects_to_rooms(pipeline, final_ids)
            self._write_lightweight_semantic_graph(pipeline, final_ids)

            with open(os.path.join(pipeline.run_dir, "config_snapshot.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(pipeline.config, f, allow_unicode=True)

            kb_path = os.path.join(pipeline.run_dir, "memory_kb.json")
            with open(kb_path, "w", encoding="utf-8") as f:
                json.dump(
                    [self._lightweight_memory_item(item) for item in pipeline.memory_kb],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            graph_path = os.path.join(pipeline.run_dir, "spatial_graph.pkl")
            with open(graph_path, "wb") as f:
                pickle.dump(pipeline.spatial_graph, f)

            if hasattr(pipeline, "_save_map_data_pkl"):
                pipeline._save_map_data_pkl()
            print(f">>> Lightweight SG files saved to: {pipeline.run_dir}")

        pipeline.finalize_room_semantics = lightweight_finalize_room_semantics
        pipeline._perform_global_update = lightweight_global_update
        pipeline._finalize_algorithm = lightweight_finalize_algorithm
        pipeline.save_all_data = lightweight_save_all_data
        pipeline._inherit_sg_lightweight_graph_patched = True

    def _save_map_snapshot(self, frame_idx):
        if not self.save_map_snapshots or self.pipeline is None:
            return
        if int(frame_idx) % self.map_snapshot_stride != 0:
            return

        pipeline = self.pipeline
        map_manager = getattr(pipeline, "map_manager", None)
        if map_manager is None:
            return

        room_map = self._map_array(getattr(map_manager, "room_id_map", None))
        free_bin = self._map_array(getattr(map_manager, "free_bin", None))
        occ_bin = self._map_array(getattr(map_manager, "occ_bin", None))
        door_map = self._map_array(getattr(map_manager, "door_map", None))
        raw_room_map = None
        room_segmenter = getattr(pipeline, "room_segmenter", None)
        if room_segmenter is not None:
            raw_room_map = self._map_array(getattr(room_segmenter, "global_id_grid", None))

        arrays = [arr for arr in (room_map, free_bin, occ_bin, door_map, raw_room_map) if arr is not None]
        if not arrays:
            return

        active = np.zeros(arrays[0].shape, dtype=bool)
        for arr in arrays:
            active |= arr > 0
        if not np.any(active):
            return

        ys, xs = np.where(active)
        pad = 40
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(active.shape[0], int(ys.max()) + pad + 1)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(active.shape[1], int(xs.max()) + pad + 1)

        room_crop = room_map[y0:y1, x0:x1] if room_map is not None else None
        free_crop = free_bin[y0:y1, x0:x1] if free_bin is not None else None
        occ_crop = occ_bin[y0:y1, x0:x1] if occ_bin is not None else None
        door_crop = door_map[y0:y1, x0:x1] if door_map is not None else None
        raw_room_crop = raw_room_map[y0:y1, x0:x1] if raw_room_map is not None else None

        out_dir = os.path.join(pipeline.run_dir, "debug_map_snapshots")
        os.makedirs(out_dir, exist_ok=True)
        vis = self._render_map_snapshot(room_crop, free_crop, occ_crop, door_crop, raw_room_crop)
        cv2.imwrite(os.path.join(out_dir, f"map_{int(frame_idx):06d}.png"), vis)

        index_row = {
            "frame_idx": int(frame_idx),
            "bbox_yxxy": [y0, x0, y1, x1],
            "room_ids": self._unique_positive(room_crop),
            "raw_room_ids": self._unique_positive(raw_room_crop),
            "free_cells": int(np.count_nonzero(free_crop)) if free_crop is not None else 0,
            "occupied_cells": int(np.count_nonzero(occ_crop)) if occ_crop is not None else 0,
        }
        with open(os.path.join(out_dir, "snapshot_index.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(index_row, ensure_ascii=False) + "\n")

        if self.save_map_snapshot_npz:
            np.savez_compressed(
                os.path.join(out_dir, f"map_{int(frame_idx):06d}.npz"),
                room_map=room_crop if room_crop is not None else np.zeros((0, 0), dtype=np.int32),
                free_bin=free_crop if free_crop is not None else np.zeros((0, 0), dtype=np.uint8),
                occ_bin=occ_crop if occ_crop is not None else np.zeros((0, 0), dtype=np.uint8),
                door_map=door_crop if door_crop is not None else np.zeros((0, 0), dtype=np.uint8),
                raw_room_map=raw_room_crop if raw_room_crop is not None else np.zeros((0, 0), dtype=np.int32),
                crop_bbox_yxxy=np.asarray([y0, x0, y1, x1], dtype=np.int32),
                resolution=np.asarray([float(getattr(map_manager, "resolution", 0.0))], dtype=np.float32),
                min_x=np.asarray([float(getattr(map_manager, "min_x", 0.0))], dtype=np.float32),
                min_z=np.asarray([float(getattr(map_manager, "min_z", 0.0))], dtype=np.float32),
            )

    def _render_map_snapshot(self, room_map, free_bin, occ_bin, door_map, raw_room_map):
        base = room_map if room_map is not None else raw_room_map
        if base is None:
            base = free_bin if free_bin is not None else occ_bin
        h, w = base.shape[:2]
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        if free_bin is not None:
            vis[free_bin > 0] = (72, 72, 72)
        if raw_room_map is not None:
            self._paint_room_ids(vis, raw_room_map, alpha=0.35)
        if room_map is not None:
            self._paint_room_ids(vis, room_map, alpha=0.80)
        if occ_bin is not None:
            vis[occ_bin > 0] = (235, 235, 235)
        if door_map is not None:
            vis[door_map > 0] = (0, 220, 255)
        return vis

    def _paint_room_ids(self, vis, room_map, alpha):
        palette = np.asarray(
            [
                (31, 119, 180),
                (255, 127, 14),
                (44, 160, 44),
                (214, 39, 40),
                (148, 103, 189),
                (140, 86, 75),
                (227, 119, 194),
                (127, 127, 127),
                (188, 189, 34),
                (23, 190, 207),
            ],
            dtype=np.float32,
        )
        for room_id in np.unique(room_map):
            rid = int(room_id)
            if rid <= 0:
                continue
            mask = room_map == rid
            color = palette[rid % len(palette)]
            vis[mask] = (vis[mask].astype(np.float32) * (1.0 - alpha) + color * alpha).astype(np.uint8)

    def _patch_cleanup_debug(self, pipeline):
        if not self.debug:
            return
        if getattr(pipeline, "_inherit_sg_cleanup_debug_patched", False):
            return
        original_cleanup = pipeline.cleanup_and_optimize_rooms

        def cleanup_with_debug():
            self._write_cleanup_room_stats(pipeline, "before")
            final_ids = original_cleanup()
            self._write_cleanup_room_stats(pipeline, "after", final_ids=final_ids)
            return final_ids

        pipeline.cleanup_and_optimize_rooms = cleanup_with_debug
        pipeline._inherit_sg_cleanup_debug_patched = True

    def _write_cleanup_room_stats(self, pipeline, phase, final_ids=None):
        map_manager = getattr(pipeline, "map_manager", None)
        room_segmenter = getattr(pipeline, "room_segmenter", None)
        raw_room_map = None
        if phase == "before" and room_segmenter is not None:
            raw_room_map = self._map_array(getattr(room_segmenter, "global_id_grid", None))
        if raw_room_map is None and map_manager is not None:
            raw_room_map = self._map_array(getattr(map_manager, "room_id_map", None))

        room_ids = self._unique_positive(raw_room_map)
        object_counts = self._count_objects_by_room(pipeline, room_ids, raw_room_map)
        stats = []
        for room_id in room_ids:
            mask = raw_room_map == int(room_id)
            ys, xs = np.where(mask)
            stats.append(
                {
                    "room_id": int(room_id),
                    "area_px": int(np.count_nonzero(mask)),
                    "area_m2": self._room_area_m2(pipeline, int(np.count_nonzero(mask))),
                    "object_count": int(object_counts.get(int(room_id), 0)),
                    "bbox_yxxy": [int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1] if ys.size else None,
                }
            )

        payload = {
            "phase": phase,
            "final_ids_returned": [int(x) for x in final_ids] if final_ids is not None else None,
            "rooms": stats,
            "room_count": len(stats),
            "memory_objects": int(len(getattr(pipeline, "memory_kb", []))),
        }
        out_dir = os.path.join(pipeline.run_dir, "debug_map_snapshots")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"cleanup_room_stats_{phase}.json"), "w", encoding="utf-8") as f:
            json.dump(self._to_builtin(payload), f, indent=2, ensure_ascii=False)

    def _count_objects_by_room(self, pipeline, room_ids, room_map):
        counts = {int(room_id): 0 for room_id in room_ids}
        map_manager = getattr(pipeline, "map_manager", None)
        if map_manager is None or room_map is None:
            return counts
        for item in getattr(pipeline, "memory_kb", []):
            pos = item.get("global_pos_3d")
            if pos is None:
                continue
            try:
                u, v = map_manager.world_to_grid(pos[0], pos[2])
                if 0 <= int(v) < room_map.shape[0] and 0 <= int(u) < room_map.shape[1]:
                    rid = int(room_map[int(v), int(u)])
                    if rid in counts:
                        counts[rid] += 1
            except Exception:
                continue
        return counts

    def _room_area_m2(self, pipeline, area_px):
        map_manager = getattr(pipeline, "map_manager", None)
        resolution = float(getattr(map_manager, "resolution", 0.0)) if map_manager is not None else 0.0
        return round(float(area_px) * resolution * resolution, 4) if resolution > 0 else None

    def _map_array(self, value):
        if value is None:
            return None
        arr = np.asarray(value)
        return arr if arr.ndim == 2 else None

    def _unique_positive(self, value):
        if value is None:
            return []
        return [int(x) for x in np.unique(value) if int(x) > 0]

    def _write_lightweight_semantic_graph(self, pipeline, final_room_ids):
        timestamp = datetime.datetime.now().isoformat()
        room_ids = self._collect_room_ids(pipeline, final_room_ids)
        objects_by_room = defaultdict(list)

        for item in pipeline.memory_kb:
            parent_room = self._normalize_room_id(item.get("parent_room"))
            room_key = parent_room if parent_room is not None else "unknown"
            objects_by_room[room_key].append(item)
            if parent_room is not None and parent_room not in room_ids:
                room_ids.append(parent_room)

        room_ids = sorted(set(room_ids))
        room_keys = list(room_ids)
        if objects_by_room.get("unknown"):
            room_keys.append("unknown")

        room_nodes = []
        object_nodes = []
        edges = []

        for room_key in room_keys:
            room_node_id = f"room_{room_key}" if room_key != "unknown" else "room_unknown"
            room_objects = objects_by_room.get(room_key, [])
            room_type, room_confidence, room_evidence, room_scores = self._infer_room_type(room_objects)
            room_node = {
                "node_id": room_node_id,
                "node_level": 1,
                "semantic_tag": "room",
                "room_label": room_type,
                "room_type": room_type,
                "room_type_confidence": room_confidence,
                "room_type_evidence": room_evidence,
                "room_type_scores": room_scores,
                "contained_obj_nodes": [obj["node_id"] for obj in room_objects],
                "object_count": len(room_objects),
                "timestamp": timestamp,
            }

            if room_key != "unknown":
                mask_file = self._save_room_mask(pipeline, room_key)
                room_node.update({
                    "room_id": int(room_key),
                    "floor_id": self._room_floor_id(pipeline, room_objects),
                    "mask_file": mask_file,
                    "centroid": self._compute_room_centroid(room_objects),
                })
            else:
                room_node.update({"room_id": "unknown", "centroid": self._compute_room_centroid(room_objects)})

            room_nodes.append(room_node)
            edges.append([room_node_id, "house_0", 1])

            for item in room_objects:
                object_nodes.append(self._lightweight_object_node(item, room_node_id, timestamp))
                edges.append([item["node_id"], room_node_id, 2])

        house_node = {
            "node_id": "house_0",
            "node_level": 0,
            "semantic_tag": "house",
            "contained_room_nodes": [node["node_id"] for node in room_nodes],
            "room_count": len(room_nodes),
            "object_count": len(object_nodes),
            "timestamp": timestamp,
        }

        with open(pipeline.nodes_path, "w", encoding="utf-8") as f:
            for node in [house_node] + room_nodes + object_nodes:
                f.write(json.dumps(self._to_builtin(node), ensure_ascii=False) + "\n")

        with open(pipeline.edges_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["child_node_id", "parent_node_id", "node_level"])
            writer.writerows(edges)

    def _assign_objects_to_room(self, pipeline, room_id):
        for item in pipeline.memory_kb:
            pos = item.get("global_pos_3d")
            if pos is not None and pipeline.map_manager.is_point_in_room(pos, room_id):
                item["parent_room"] = int(room_id)
            elif item.get("parent_room") == room_id:
                item["parent_room"] = None

    def _reassign_all_objects_to_rooms(self, pipeline, final_room_ids=None):
        valid_room_ids = set(self._collect_room_ids(pipeline, final_room_ids))
        for item in pipeline.memory_kb:
            rid = self._room_id_for_object(pipeline, item, valid_room_ids)
            item["parent_room"] = rid

    def _room_id_for_object(self, pipeline, item, valid_room_ids):
        pos = item.get("global_pos_3d")
        if pos is not None:
            try:
                u, v = pipeline.map_manager.world_to_grid(pos[0], pos[2])
                if pipeline.map_manager._valid_uv(u, v):
                    candidate = int(pipeline.map_manager.room_id_map[v, u])
                    if candidate > 0 and (not valid_room_ids or candidate in valid_room_ids):
                        return candidate
            except Exception:
                pass

            for room_id in sorted(valid_room_ids):
                try:
                    if pipeline.map_manager.is_point_in_room(pos, room_id):
                        return int(room_id)
                except Exception:
                    continue

        previous = self._normalize_room_id(item.get("parent_room"))
        if previous is not None and (not valid_room_ids or previous in valid_room_ids):
            return previous
        return None

    def _collect_room_ids(self, pipeline, final_room_ids=None):
        room_ids = set()
        for room_id in final_room_ids or []:
            normalized = self._normalize_room_id(room_id)
            if normalized is not None:
                room_ids.add(normalized)
        room_ids.update(self._valid_room_ids_from_map(pipeline))
        return sorted(room_ids)

    def _valid_room_ids_from_map(self, pipeline):
        room_map = getattr(getattr(pipeline, "map_manager", None), "room_id_map", None)
        if room_map is None:
            return []
        return [int(room_id) for room_id in np.unique(room_map) if int(room_id) > 0]

    def _save_room_mask(self, pipeline, room_id):
        room_map = getattr(pipeline.map_manager, "room_id_map", None)
        if room_map is None:
            return None
        mask = (room_map == int(room_id)).astype(np.uint8)
        if not np.any(mask):
            return None
        mask_file = f"mask_room_{int(room_id)}.png"
        cv2.imwrite(os.path.join(pipeline.run_dir, mask_file), (mask * 255).astype(np.uint8))
        return mask_file

    def _compute_room_centroid(self, room_objects):
        coords = []
        for obj in room_objects:
            pos = obj.get("global_pos_3d")
            if pos is not None:
                coords.append([float(pos[0]), float(pos[2])])
        if not coords:
            return None
        return np.mean(np.asarray(coords, dtype=np.float32), axis=0).tolist()

    def _room_floor_id(self, pipeline, room_objects):
        for obj in room_objects:
            if obj.get("floor_id") is not None:
                return int(obj["floor_id"])
        floor_tracker = getattr(pipeline, "floor_tracker", None)
        return int(getattr(floor_tracker, "current_floor_id", 0))

    def _lightweight_object_node(self, item, room_node_id, timestamp):
        ensure_object_provenance(item)
        node = {
            "node_id": item["node_id"],
            "node_level": 2,
            "semantic_tag": item.get("label", item.get("semantic_tag", "object")),
            "label_aliases": item.get("label_aliases", []),
            "bbox": item.get("bbox"),
            "global_pos_3d": item.get("global_pos_3d"),
            "floor_id": item.get("floor_id"),
            "parent_room_id": room_node_id,
            "timestamp": timestamp,
        }
        for key in (
            "full_image_path",
            "cropped_image_path",
            "view_score",
            "best_observation",
            "best_observation_score",
            "navigation_waypoint",
            "waypoint_schema_version",
            "observed_episode_ids",
            "observed_source_episode_ids",
            "best_observation_by_episode",
            "navigation_waypoint_by_episode",
            "observation_provenance_schema_version",
        ):
            if key in item:
                node[key] = item[key]
        return self._to_builtin(node)

    def _lightweight_memory_item(self, item):
        ensure_object_provenance(item)
        keep_keys = (
            "node_id",
            "dinov3_feature",
            "label",
            "canonical_label",
            "label_score",
            "label_aliases",
            "label_evidence",
            "label_score_schema_version",
            "legacy_mixed_label_import",
            "full_image_path",
            "cropped_image_path",
            "bbox",
            "global_pos_3d",
            "floor_id",
            "parent_room",
            "view_score",
            "best_observation",
            "best_observation_score",
            "navigation_waypoint",
            "waypoint_schema_version",
            "observed_episode_ids",
            "observed_source_episode_ids",
            "best_observation_by_episode",
            "navigation_waypoint_by_episode",
            "observation_provenance_schema_version",
        )
        return self._to_builtin({key: item[key] for key in keep_keys if key in item})

    @staticmethod
    def _infer_room_type_from_labels(labels):
        if not labels:
            return "unknown", 0.0, [], {}

        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1

        scores = {}
        evidence = defaultdict(list)
        for room_type, rule in ROOM_TYPE_RULES.items():
            score = 0.0
            seen_terms = set()
            for label, count in label_counts.items():
                for term, weight in rule.items():
                    # Labels are already expanded into exact tokens plus the
                    # full normalized phrase. Substring matching makes
                    # ``washer`` fire on ``dishwasher`` and corrupts laundry
                    # room evidence, so semantic rules must match one of
                    # those exact normalized forms.
                    if term == label:
                        # Unique evidence matters more than repeated detections of the same kind.
                        contribution = float(weight) * (1.0 + min(count - 1, 3) * 0.15)
                        score += contribution
                        if term not in seen_terms:
                            evidence[room_type].append(term)
                            seen_terms.add(term)
            if score > 0:
                scores[room_type] = round(score, 3)

        if not scores:
            return "unknown", 0.0, [], {}

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_type, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        total_score = sum(scores.values())

        confidence = best_score / total_score if total_score > 0 else 0.0
        margin = best_score - second_score
        if best_score < 2.0 or (confidence < 0.38 and margin < 1.0):
            return "unknown", round(float(confidence), 3), evidence.get(best_type, []), scores

        return best_type, round(float(confidence), 3), evidence.get(best_type, []), scores

    @staticmethod
    def _normalize_label_tokens(label):
        if not label:
            return []
        raw_labels = []
        if isinstance(label, str):
            raw_labels = label.split()
            raw_labels.append(label)
        else:
            raw_labels = [str(label)]

        normalized = []
        for raw in raw_labels:
            text = str(raw).lower().replace("_", " ").replace("-", " ").strip()
            text = " ".join(text.split())
            if not text:
                continue
            normalized.append(ROOM_TYPE_ALIASES.get(text, text))
        return normalized

    def _infer_room_type(self, room_objects):
        labels = []
        for obj in room_objects:
            label = obj.get("label") or obj.get("semantic_tag")
            labels.extend(self._normalize_object_label(label))
        return self._infer_room_type_from_labels(labels)

    def _normalize_object_label(self, label):
        return self._normalize_label_tokens(label)

    def _normalize_room_id(self, room_id):
        if room_id is None or room_id == "unknown":
            return None
        if isinstance(room_id, str) and room_id.startswith("room_"):
            room_id = room_id.split("room_", 1)[1]
        try:
            room_id = int(room_id)
        except (TypeError, ValueError):
            return None
        return room_id if room_id > 0 else None

    def _to_builtin(self, value):
        if isinstance(value, dict):
            return {k: self._to_builtin(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_builtin(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            try:
                return value.tolist()
            except Exception:
                return value
        return value

    def is_query_ready(self):
        snap = self._query_snapshot
        if snap is None:
            return False
        return (
            self.enabled
            and not self.failed
            and self._finalized_episode_count >= self._query_min_finalized_episodes
            and len(snap.get("objects", [])) > 0
        )

    def query_two_stage(
        self,
        task_text=None,
        room_hint=None,
        target_object=None,
        top_k=5,
        required_observed_episode_ids=None,
    ):
        if not self.is_query_ready():
            return []
        if target_object is None:
            room_hint, target_object = self.extract_task_entities(task_text)
        room_hint = self._canonical_room(room_hint)
        target_object = self._clean_phrase(target_object)
        if not target_object:
            return []

        with self._query_lock:
            snap = self._query_snapshot

        room_match = None
        required_episode_ids = frozenset(
            self._ordered_unique_strings(required_observed_episode_ids or [])
        )
        candidates = snap["objects"]
        if required_episode_ids:
            candidates = [
                obj
                for obj in candidates
                if required_episode_ids.issubset(
                    frozenset(obj.get("observed_episode_ids") or [])
                )
            ]
        if room_hint:
            room_query_emb = self._encode_query_text(snap, room_hint)
            room_scores = []
            for room_id, room in snap["rooms"].items():
                semantic_score = max(
                    self._lexical_match_score(room_hint, room.get("label", "")),
                    self._embedding_score(room_query_emb, room.get("embedding")),
                )
                requested_evidence_score = float(
                    (room.get("scores") or {}).get(room_hint, 0.0)
                )
                # The room classifier retains scores for every supported room
                # type. Use query-specific evidence even when a mixed/open
                # region's winning label is another type (for example a
                # laundry alcove adjoining a kitchen).
                evidence_score = (
                    min(1.0, 0.6 + 0.08 * requested_evidence_score)
                    if requested_evidence_score >= 2.0
                    else 0.0
                )
                score = max(semantic_score, evidence_score)
                room_scores.append(
                    (
                        score,
                        requested_evidence_score,
                        room_id,
                        room,
                    )
                )
            room_scores.sort(
                key=lambda item: (
                    -float(item[0]),
                    -float(item[1]),
                    str(item[2]),
                )
            )
            if room_scores and room_scores[0][0] >= 0.6:
                room_score, evidence_score, room_id, room = room_scores[0]
                room_match = {
                    "room_id": room_id,
                    "label": room.get("label", ""),
                    "score": float(room_score),
                    "requested_room_evidence_score": float(evidence_score),
                }

        target_emb = self._encode_query_text(snap, target_object)
        results = []
        for obj in candidates:
            retrieval_labels = obj.get("retrieval_labels") or [obj.get("label", "")]
            object_score = max(
                max(
                    self._lexical_match_score(target_object, label)
                    for label in retrieval_labels
                ),
                max(
                    (
                        self._embedding_score(target_emb, embedding)
                        for embedding in (
                            obj.get("retrieval_embeddings")
                            or [obj.get("embedding")]
                        )
                    ),
                    default=0.0,
                ),
            )
            room_consistent = bool(room_match and obj.get("parent_room") == room_match["room_id"])
            results.append(
                {
                    "uid": obj.get("node_id"),
                    "node_id": obj.get("node_id"),
                    "label": obj.get("label", ""),
                    "label_aliases": obj.get("label_aliases", []),
                    "position": obj.get("global_pos_3d"),
                    "floor_id": obj.get("floor_id"),
                    "parent_room": obj.get("parent_room"),
                    "room_label": obj.get("room_label"),
                    "waypoint": self._to_builtin(obj.get("navigation_waypoint")),
                    "best_observation": self._to_builtin(obj.get("best_observation")),
                    "best_observation_score": obj.get("best_observation_score"),
                    "observed_episode_ids": self._to_builtin(
                        obj.get("observed_episode_ids") or []
                    ),
                    "observed_source_episode_ids": self._to_builtin(
                        obj.get("observed_source_episode_ids") or []
                    ),
                    "best_observation_by_episode": self._to_builtin(
                        obj.get("best_observation_by_episode") or {}
                    ),
                    "navigation_waypoint_by_episode": self._to_builtin(
                        obj.get("navigation_waypoint_by_episode") or {}
                    ),
                    "required_observed_episode_ids": sorted(
                        required_episode_ids
                    ),
                    "room_score": room_match.get("score") if room_consistent else 0.0,
                    "requested_room_evidence_score": (
                        room_match.get("requested_room_evidence_score")
                        if room_consistent
                        else 0.0
                    ),
                    "room_consistent": room_consistent,
                    "object_score": float(object_score),
                    "score": float(object_score),
                    "query_room": room_hint,
                    "query_object": target_object,
                    "source": "inherit_sg_two_stage",
                }
            )
        results.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                0 if item.get("room_consistent") else 1,
                float(item.get("best_observation_score"))
                if item.get("best_observation_score") is not None
                else float("inf"),
                str(item.get("node_id") or ""),
            )
        )
        return results[: max(1, int(top_k))]

    def extract_task_entities(self, task_text):
        text = self._clean_phrase(task_text)
        if not text:
            return None, None

        room_hint = None
        for room in sorted(ROOM_TYPE_RULES.keys(), key=len, reverse=True):
            if self._phrase_in_text(room, text):
                room_hint = room
                break
        if room_hint is None:
            room_aliases = {
                "lounge": "living room",
                "sitting room": "living room",
                "livingroom": "living room",
                "restroom": "bathroom",
                "washroom": "bathroom",
                "bed room": "bedroom",
                "dining area": "dining room",
                "kitchen area": "kitchen",
                "study": "office",
                "workspace": "office",
                "hall": "hallway",
                "corridor": "hallway",
            }
            for alias, canonical_room in room_aliases.items():
                if canonical_room and self._phrase_in_text(alias, text):
                    room_hint = canonical_room
                    break

        target_object = None
        for term in sorted(self._query_object_terms(), key=len, reverse=True):
            if self._phrase_in_text(term, text):
                target_object = term
                break
        return room_hint, target_object

    def _refresh_query_snapshot(self):
        if self.pipeline is None:
            return
        from membridge.semantic_scene_graph.legacy_builder_source.label_score import (
            ensure_label_score_state,
            retrieval_labels,
        )

        with self._query_lock:
            objects = []
            for item in getattr(self.pipeline, "memory_kb", []):
                ensure_label_score_state(item)
                position = item.get("global_pos_3d")
                label = self._clean_phrase(item.get("label") or item.get("semantic_tag"))
                if position is None or not label:
                    continue
                try:
                    if len(position) < 3:
                        continue
                except TypeError:
                    continue
                parent_room = self._normalize_room_id(item.get("parent_room"))
                canonical_labels = [
                    self._clean_phrase(value)
                    for value in retrieval_labels(item)
                    if self._clean_phrase(value)
                ]
                objects.append(
                    {
                        "node_id": item.get("node_id"),
                        "label": label,
                        "label_aliases": [
                            value for value in canonical_labels if value != label
                        ],
                        "retrieval_labels": canonical_labels or [label],
                        "global_pos_3d": self._to_builtin(position),
                        "floor_id": item.get("floor_id"),
                        "parent_room": parent_room,
                        "view_score": item.get("view_score"),
                        "navigation_waypoint": self._to_builtin(item.get("navigation_waypoint")),
                        "best_observation": self._to_builtin(item.get("best_observation")),
                        "best_observation_score": item.get("best_observation_score"),
                        "observed_episode_ids": self._to_builtin(
                            item.get("observed_episode_ids") or []
                        ),
                        "observed_source_episode_ids": self._to_builtin(
                            item.get("observed_source_episode_ids") or []
                        ),
                        "best_observation_by_episode": self._to_builtin(
                            item.get("best_observation_by_episode") or {}
                        ),
                        "navigation_waypoint_by_episode": self._to_builtin(
                            item.get("navigation_waypoint_by_episode") or {}
                        ),
                    }
                )

            rooms = {}
            objects_by_room = defaultdict(list)
            for obj in objects:
                objects_by_room[obj.get("parent_room")].append(obj)
            for room_id, room_objects in objects_by_room.items():
                if room_id is None:
                    continue
                room_type, confidence, evidence, scores = self._infer_room_type(room_objects)
                rooms[room_id] = {
                    "room_id": room_id,
                    "label": self._clean_phrase(room_type),
                    "confidence": confidence,
                    "evidence": evidence,
                    "scores": scores,
                    "object_count": len(room_objects),
                }
                for obj in room_objects:
                    obj["room_label"] = rooms[room_id]["label"]

            encoder = self._query_text_encoder()
            obj_texts = [
                text
                for obj in objects
                for text in obj["retrieval_labels"]
            ]
            room_ids = list(rooms.keys())
            room_texts = [rooms[room_id]["label"] for room_id in room_ids]
            obj_embs = self._encode_texts(encoder, obj_texts)
            room_embs = self._encode_texts(encoder, room_texts)
            embedding_offset = 0
            for obj in objects:
                label_count = len(obj["retrieval_labels"])
                embeddings = obj_embs[
                    embedding_offset : embedding_offset + label_count
                ]
                embedding_offset += label_count
                obj["retrieval_embeddings"] = embeddings
                obj["embedding"] = embeddings[0] if embeddings else None
            for room_id, emb in zip(room_ids, room_embs):
                rooms[room_id]["embedding"] = emb

            self._query_snapshot = {
                "objects": objects,
                "rooms": rooms,
                "encoder": encoder,
                "object_terms": sorted(self._build_object_terms(objects)),
                "finalized_episode_count": self._finalized_episode_count,
            }
            logger.info(
                f"[INHerit-SG] query snapshot refreshed: {len(objects)} objects, "
                f"{len(rooms)} rooms, finalized_eps={self._finalized_episode_count}"
            )

    def _query_object_terms(self):
        terms = set()
        snap = self._query_snapshot
        if snap is not None:
            terms.update(snap.get("object_terms", []))
        if self.pipeline is not None:
            cfg_terms = str(self.pipeline.config.get("predefined_object_classes", ""))
            for term in cfg_terms.split("."):
                cleaned = self._clean_phrase(term)
                if cleaned:
                    terms.add(cleaned)
        for alias, canonical in self._object_aliases().items():
            terms.add(alias)
            terms.add(canonical)
        return terms

    def _build_object_terms(self, objects):
        terms = set()
        for obj in objects:
            for candidate in obj.get("retrieval_labels") or [obj.get("label")]:
                label = self._clean_phrase(candidate)
                if not label:
                    continue
                terms.add(label)
                for part in label.split():
                    if len(part) > 2:
                        terms.add(part)
        if self.pipeline is not None:
            cfg_terms = str(self.pipeline.config.get("predefined_object_classes", ""))
            for term in cfg_terms.split("."):
                cleaned = self._clean_phrase(term)
                if cleaned:
                    terms.add(cleaned)
        for alias, canonical in self._object_aliases().items():
            terms.add(alias)
            terms.add(canonical)
        return terms

    def _query_text_encoder(self):
        if self.shared_models and self.shared_models.get("text_encoder") is not None:
            return self.shared_models["text_encoder"]
        if self.pipeline is not None and getattr(self.pipeline, "text_encoder", None) is not None:
            return self.pipeline.text_encoder
        return self._make_lightweight_text_encoder()

    def _encode_query_text(self, snap, text):
        encoder = snap.get("encoder") or self._query_text_encoder()
        embs = self._encode_texts(encoder, [text])
        return embs[0] if len(embs) else None

    def _encode_texts(self, encoder, texts):
        if not texts:
            return []
        try:
            arr = encoder.encode(texts, normalize_embeddings=True, convert_to_tensor=False, show_progress_bar=False)
        except TypeError:
            arr = encoder.encode(texts)
        except Exception as exc:
            logger.warning(f"[INHerit-SG] text embedding failed; lexical retrieval only: {exc}")
            return [None for _ in texts]
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = np.divide(arr, np.maximum(norms, 1e-8))
        return [arr[i].copy() for i in range(arr.shape[0])]

    def _embedding_score(self, query_emb, candidate_emb):
        if query_emb is None or candidate_emb is None:
            return 0.0
        try:
            return float(np.dot(np.asarray(query_emb, dtype=np.float32), np.asarray(candidate_emb, dtype=np.float32)))
        except Exception:
            return 0.0

    def _lexical_match_score(self, query, candidate):
        q = self._canonical_object(query)
        c = self._canonical_object(candidate)
        if not q or not c:
            return 0.0
        if q == c:
            return 1.0
        if self._phrase_in_text(q, c) or self._phrase_in_text(c, q):
            return 0.9
        q_s = q[:-1] if q.endswith("s") else q
        c_s = c[:-1] if c.endswith("s") else c
        if q_s and (q_s == c_s or self._phrase_in_text(q_s, c_s) or self._phrase_in_text(c_s, q_s)):
            return 0.9
        return 0.0

    def _canonical_room(self, text):
        text = self._clean_phrase(text)
        if not text:
            return None
        aliases = {
            "lounge": "living room",
            "sitting room": "living room",
            "livingroom": "living room",
            "restroom": "bathroom",
            "washroom": "bathroom",
            "bed room": "bedroom",
            "dining area": "dining room",
            "kitchen area": "kitchen",
            "study": "office",
            "workspace": "office",
            "hall": "hallway",
            "corridor": "hallway",
        }
        return aliases.get(text, text)

    def _canonical_object(self, text):
        text = self._clean_phrase(text)
        if not text:
            return ""
        aliases = self._object_aliases()
        return aliases.get(text, text)

    def _object_aliases(self):
        aliases = {
            # Historical Level-3 instructions use "bath" for the physical
            # bathtub object. Keep this built in so frozen pre-fix vocabulary
            # snapshots remain query-compatible.
            "bath": "bathtub",
            "bathroom cabinet": "cabinet",
            "couch": "sofa",
            "dining chair": "chair",
            "dining table": "table",
            "flowerpot": "pot",
            "kitchen cabinet": "cabinet",
            "kitchen sink": "sink",
            "kitchen table": "table",
            "television": "tv",
            "tv monitor": "tv",
            "monitor": "tv",
            "fridge": "refrigerator",
            "coffee maker": "coffee machine",
            "countertop": "counter",
            "worktop": "counter",
            "washbasin": "sink",
            "bookshelf": "shelf",
            "doormat": "rug",
            "door mat": "rug",
            "floor mat": "rug",
            "bath mat": "rug",
            "mat": "rug",
            "carpet": "rug",
        }
        runtime_aliases = (
            (getattr(self, "runtime_query_vocabulary", None) or {}).get(
                "aliases"
            )
            or {}
        )
        aliases.update(runtime_aliases)
        # The active vocabulary is the canonical namespace.  Historical
        # compatibility aliases must never rewrite one of its canonical
        # labels (for example, the old ``monitor -> tv`` fallback).
        runtime_labels = (
            (getattr(self, "runtime_query_vocabulary", None) or {}).get(
                "labels"
            )
            or []
        )
        canonical_labels = {
            self._clean_phrase(label)
            for label in runtime_labels
            if self._clean_phrase(label)
        }
        if canonical_labels:
            aliases = {
                alias: canonical
                for alias, canonical in aliases.items()
                if canonical in canonical_labels
            }
            aliases.update({label: label for label in canonical_labels})
        return aliases

    def _phrase_in_text(self, phrase, text):
        phrase = self._clean_phrase(phrase)
        text = self._clean_phrase(text)
        if not phrase or not text:
            return False
        return f" {phrase} " in f" {text} "

    def _clean_phrase(self, text):
        if text is None:
            return ""
        text = str(text).lower().replace("_", " ").replace("-", " ")
        return " ".join(text.split())

    def _camera_intrinsics(self):
        hfov = float(self.lavira_config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV)
        hfov_rad = np.deg2rad(hfov)
        fx = self.width / (2.0 * np.tan(hfov_rad / 2.0))
        fy = fx
        cx = self.width / 2.0
        cy = self.height / 2.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    def _write_session_metadata(self, scene_id):
        if self.pipeline is None:
            return
        preloaded_memory_stage = (
            os.environ.get("INHERIT_SG_PRELOAD_MEMORY_STAGE", "").strip()
            if self._preloaded_session
            else ""
        )
        payload = {
            "created_at": datetime.datetime.now().isoformat(),
            "mode": "lifelong_session",
            "object_waypoint_schema_version": 1,
            "object_waypoint_policy": "best_observation_pose",
            "inherit_sg_best_frame_policy": "unchanged_reference_minimum_center_offset",
            "best_observation_score_policy": "minimum_weighted_center_area_observation_score",
            "best_observation_score_weights": {
                "center_offset": float(self.observation_center_weight),
                "bbox_area": float(self.observation_area_weight),
            },
            "scene_id": str(scene_id or ""),
            "async_enabled": bool(self.async_enabled),
            "async_queue_size": int(self.async_queue_size),
            "async_drop_policy": self.async_drop_policy,
            "stride": int(self.stride),
            "runtime_query_vocabulary": self.runtime_query_vocabulary,
            "frame_storage": {
                "mode": (
                    "reuse_immutable_external_rgbd"
                    if self.external_frame_root
                    else "persist_under_session_frames"
                ),
                "external_frame_root": (
                    self.external_frame_root or None
                ),
                "duplicate_rgbd_written": False
                if self.external_frame_root
                else True,
            },
            "debug_artifact_storage": {
                "enabled": bool(self.debug),
                "directory": "debug_merge",
                "filename": "frame_{global_frame_idx:06d}.jpg",
                "nested_per_object_frame_directories": False,
            },
            "preloaded_memory": {
                "enabled": bool(self._preloaded_session),
                "source_dir": (
                    os.path.abspath(self.preload_dir)
                    if self._preloaded_session
                    else None
                ),
                "memory_stage": preloaded_memory_stage or None,
                "query_only": bool(self.preload_query_only),
                "world_map_only_updates": bool(
                    self.preload_world_map_only
                ),
                "builder_state_restored": bool(
                    self._preloaded_session
                    and os.path.isfile(
                        os.path.join(self.preload_dir, BUILDER_STATE_NAME)
                    )
                ),
                "materialized_file_count": int(
                    self._preloaded_materialized_file_count
                ),
                "finalized_episode_count": int(
                    self._finalized_episode_count
                ),
            },
            "preloaded_object_reid_guard": {
                "enabled": bool(self.preload_dir and not self.preload_query_only),
                "immutable_baseline_object_count": int(
                    len(self._preloaded_object_baselines)
                ),
                "maximum_inferred_observation_distance_m": float(
                    self.preloaded_object_match_guard_m
                ),
                "policy": "rollback_frame_merge_relative_to_immutable_preload",
            },
            "world_memory": {
                "enabled": bool(self.world_memory_enabled),
                "required": bool(self.world_memory_required),
                "schema_version": 1,
                "coordinate_frame": "habitat_world",
                "resolution_m": float(self.world_map_resolution_m),
                "chunk_size": int(self.world_map_chunk_size),
                "pixel_stride": int(self.world_map_pixel_stride),
                "max_rays_per_frame": int(self.world_map_max_rays),
                "trajectory_spacing_m": float(self.world_trajectory_spacing_m),
                "trajectory_max_edge_m": float(self.world_trajectory_max_edge_m),
                "occupancy_fusion": "lavira_floor_obstacle_inverse_sampled_to_habitat_world",
                "occupancy_fusion_obstacle_precedence": True,
            },
            "output_dir": self.pipeline.run_dir,
        }
        with open(os.path.join(self.pipeline.run_dir, "session_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self._to_builtin(payload), f, indent=2, ensure_ascii=False)

    def _write_episode_manifest_row(self):
        if self.pipeline is None:
            return
        row = {
            "finished_at": datetime.datetime.now().isoformat(),
            "episode_index": int(self._session_episode_count),
            "episode_name": self._current_episode_name,
            "start_frame_idx": int(self._episode_start_frame_idx),
            "end_frame_idx_exclusive": int(self.frame_idx),
            "num_streamed_frames": int(max(0, self.frame_idx - self._episode_start_frame_idx)),
            "memory_objects": int(len(getattr(self.pipeline, "memory_kb", []))),
            "memory_objects_with_waypoint": int(
                sum(
                    1
                    for item in getattr(self.pipeline, "memory_kb", [])
                    if item.get("navigation_waypoint") is not None
                )
            ),
            "dropped_frames": int(self._dropped_frames),
            "confirmed_keyframes": int(self._episode_confirmed_keyframes),
            "preloaded_object_reid_guard": {
                "immutable_baseline_object_count": int(
                    len(self._preloaded_object_baselines)
                ),
                "maximum_inferred_observation_distance_m": float(
                    self.preloaded_object_match_guard_m
                ),
                "episode_rejected_merges": int(
                    self._episode_preloaded_object_guard_rejections
                ),
                "session_rejected_merges": int(
                    self._preloaded_object_guard_rejections
                ),
            },
            "world_observations": int(self._world_observation_count),
            "world_integrated_rays": int(self._world_integrated_ray_count),
            "world_terminal_poses": int(self._world_terminal_pose_count),
            "world_lavira_occupancy_fusion": self._world_lavira_fusion_stats,
            "world_map_stats": (
                self.world_memory.metric_map.stats()
                if self.world_memory is not None
                else None
            ),
            "trajectory_nodes": (
                len(self.world_memory.trajectory.nodes)
                if self.world_memory is not None
                else 0
            ),
            "trajectory_edges": (
                len(self.world_memory.trajectory.edges)
                if self.world_memory is not None
                else 0
            ),
        }
        manifest_path = os.path.join(self.pipeline.run_dir, "episode_manifest.jsonl")
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._to_builtin(row), ensure_ascii=False) + "\n")

    def _depth_uint16_mm(self, depth):
        d = np.asarray(depth, dtype=np.float32)
        if d.ndim == 3:
            d = d.squeeze(-1)
        sensor = self.lavira_config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR
        if bool(getattr(sensor, "NORMALIZE_DEPTH", True)):
            min_d = float(sensor.MIN_DEPTH)
            max_d = float(sensor.MAX_DEPTH)
            d = min_d + d * (max_d - min_d)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        d = np.clip(d, 0.0, 65.535)
        return (d * 1000.0).astype(np.uint16)

    def _rgb_bgr(self, rgb):
        arr = np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def _episode_dir_name(self, episode_id, scene_id):
        scene = os.path.splitext(os.path.basename(str(scene_id or "scene")))[0]
        safe_scene = self._safe_name(scene)
        safe_ep = self._safe_name(str(episode_id))
        return f"{safe_scene}_ep_{safe_ep}"

    @staticmethod
    def _safe_name(value):
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))

    @staticmethod
    def _env_bool(name, default):
        raw = os.environ.get(name)
        if raw is None:
            return bool(default)
        return raw.strip().lower() not in {"0", "false", "no", "off"}
