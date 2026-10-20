#!/usr/bin/env python3
"""Run ETPNav Level-2 EP1 with frozen M0 SG recall and ETPNav continuation.

Mirrors JanusVLN† / GridMM† / Uni-LaViRA† Level-2 EP1:

* Uni-LaViRA routes are reused (one type-1 recall + type-2 suffix).  The
  instruction is not parsed again.
* ``memory_module`` owns retrieval, metric-map navigation and exploration.
* EP1 RGB-D is streamed to ``OnlineINHeritSGBuilder`` in
  ``PRELOAD_WORLD_MAP_ONLY`` mode so the navigation map and trajectory update
  live.  SAM3 and DINOv3 are never imported.  Raw RGB-D is never written.
* After the SG anchor, ETPNav runs the merged type-2 suffix.  Native GraphMap
  starts empty at EP1 (no EP0 topology import) and is updated live through SG
  and the suffix from 12-view RGB-D.  HIGHTOLOW only runs when ETPNav acts.
* Rollouts persist Habitat XYZ + XYZW quaternions only.

M1 = frozen M0 semantic SG + updated world map/trajectory.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import math
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_ETPNAV_ROOT = HERE.parent / "ETPNav"
METHOD = HERE.parents[1] / "method"
if not METHOD.is_dir():
    METHOD = Path("/workspace/LifeLongHomeNav/Mem-X/method")
REPO_ROOT = METHOD
JANUS_EP1_DIR = METHOD / "janus"
DEFAULT_DATASET = DEFAULT_ETPNAV_ROOT / "data/datasets/HM3D_VLNCE_v5/level2/level2_ep1.json.gz"
DEFAULT_TASK_CONFIG = DEFAULT_ETPNAV_ROOT / "run_r2r/hm3d_level2.yaml"
DEFAULT_EXP_CONFIG = DEFAULT_ETPNAV_ROOT / "run_r2r/iter_level2.yaml"
DEFAULT_CKPT = DEFAULT_ETPNAV_ROOT / "data/logs/checkpoints/release_r2r/ckpt.iter12000.pth"
DEFAULT_WAYPOINT = DEFAULT_ETPNAV_ROOT / "data/wp_pred/check_cwp_bestdist_hfov90"
DEFAULT_M0_ROOT = Path(
    "/share/shared/shizk/level2_outputs/etpnav_memory_module/ep0_memory_from_poses_b15_20260910-ep0-sg-b15"
)
DEFAULT_ROUTE_SUMMARY = (
    REPO_ROOT
    / "baseline_experiment/Uni-LaViRA/experiment_level2/outputs/final_level2_sg_latest.json"
)
DEFAULT_OUTPUT = Path(
    "/share/shared/shizk/level2_outputs/etpnav_memory_module/ep1_memory_etpnav"
)
PROTOCOL_WIDTH = 960
PROTOCOL_HEIGHT = 720
ETP_RGB = 224
ETP_DEPTH = 256
FORWARD_STEP_M = 0.25
TURN_ANGLE_DEG = 30.0
MAX_HIGH_LEVEL = 15

sys.path.insert(0, str(JANUS_EP1_DIR))
from run_ep1_memory_janus import (  # noqa: E402
    ACTION_NAMES,
    Actions,
    CORE_M0_FILES,
    NAVIGATION_CONTROLLER_VERSION,
    append_jsonl,
    atomic_json,
    atomic_jsonl,
    install_full_exploration_overlay,
    install_memory_environment,
    load_route_index,
    load_saved_recall_plan,
    make_target_info,
    MemoryNavigator,
    memory_observation,
    normalize_text,
    pose_from_env,
    prepare_recall_plan,
    publish_m1 as janus_publish_m1,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    validate_inputs as janus_validate_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etpnav-root", type=Path, default=DEFAULT_ETPNAV_ROOT)
    parser.add_argument("--exp-config", type=Path, default=DEFAULT_EXP_CONFIG)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--waypoint-ckpt", type=Path, default=DEFAULT_WAYPOINT)
    parser.add_argument("--m0-root", type=Path, default=DEFAULT_M0_ROOT)
    parser.add_argument("--route-summary", type=Path, default=DEFAULT_ROUTE_SUMMARY)
    parser.add_argument("--extra-route", type=Path, action="append", default=[])
    parser.add_argument("--recall-plan-root", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenes-dir", type=Path, default=Path(os.environ.get("SCENES_DIR", "")))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-gpu-id", type=int, default=0)
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
    parser.add_argument(
        "--force-etpnav-fallback",
        action="store_true",
        help="Skip SG recall/navigation and run the full instruction with ETPNav.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="HOST_PREFIX=RUNTIME_PREFIX",
    )
    parser.add_argument(
        "--query-vocabulary",
        type=Path,
        default=(
            REPO_ROOT
            / "memory_module/semantic_scene_graph/config/level1_runtime_query_vocabulary_75.json"
        ),
    )
    parser.add_argument("--bert-path", type=Path, default=Path(os.environ.get("MODELS_DIR", "")) / "bert-base-uncased")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


class VramTracker:
    """Track CUDA allocator + nvidia-smi used-memory peaks for this process."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.peak_alloc_bytes = 0
        self.peak_reserved_bytes = 0
        self.peak_nvidia_mib = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = None

    def _gpu_index(self) -> int:
        if self.device.startswith("cuda:"):
            return int(self.device.split(":", 1)[1])
        return 0

    def snapshot(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.peak_alloc_bytes = max(
                    self.peak_alloc_bytes, int(torch.cuda.max_memory_allocated())
                )
                self.peak_reserved_bytes = max(
                    self.peak_reserved_bytes, int(torch.cuda.max_memory_reserved())
                )
        except Exception:
            pass
        try:
            import subprocess

            raw = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self._gpu_index()}",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            mine = os.getpid()
            for line in raw.splitlines():
                parts = [item.strip() for item in line.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    pid = int(parts[0])
                    used = int(float(parts[1]))
                except ValueError:
                    continue
                if pid == mine:
                    self.peak_nvidia_mib = max(self.peak_nvidia_mib, used)
        except Exception:
            pass
        self.samples += 1

    def start(self, interval_s: float = 1.0) -> None:
        self.snapshot()

        def _loop() -> None:
            while not self._stop.wait(interval_s):
                self.snapshot()

        self._thread = threading.Thread(target=_loop, name="vram-tracker", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.snapshot()
        return {
            "torch_max_allocated_gib": round(self.peak_alloc_bytes / (1024**3), 3),
            "torch_max_reserved_gib": round(self.peak_reserved_bytes / (1024**3), 3),
            "nvidia_smi_this_pid_peak_mib": self.peak_nvidia_mib,
            "nvidia_smi_this_pid_peak_gib": round(self.peak_nvidia_mib / 1024.0, 3),
            "samples": self.samples,
            "device": self.device,
        }


def hydra_like_sensor_config(width: int, height: int, gpu: int) -> SimpleNamespace:
    rgb = SimpleNamespace(
        hfov=90.0, position=(0.0, 1.2, 0.0), width=int(width), height=int(height)
    )
    depth = SimpleNamespace(
        hfov=90.0,
        position=(0.0, 1.2, 0.0),
        normalize_depth=False,
        min_depth=0.0,
        max_depth=10.0,
        width=int(width),
        height=int(height),
    )
    agent = SimpleNamespace(
        radius=0.10,
        sim_sensors=SimpleNamespace(rgb_sensor=rgb, depth_sensor=depth),
    )
    simulator = SimpleNamespace(
        turn_angle=TURN_ANGLE_DEG,
        forward_step_size=FORWARD_STEP_M,
        agents=SimpleNamespace(main_agent=agent),
    )
    return SimpleNamespace(
        habitat=SimpleNamespace(simulator=simulator),
        TORCH_GPU_ID=int(gpu),
        TASK_CONFIG=SimpleNamespace(
            SIMULATOR=SimpleNamespace(
                RGB_SENSOR=SimpleNamespace(HFOV=90.0, POSITION=(0.0, 1.2, 0.0)),
                DEPTH_SENSOR=SimpleNamespace(
                    HFOV=90.0,
                    POSITION=(0.0, 1.2, 0.0),
                    NORMALIZE_DEPTH=False,
                    MIN_DEPTH=0.0,
                    MAX_DEPTH=10.0,
                ),
            )
        ),
    )


def sensor_memory_config_from_protocol(logical_gpu: int) -> SimpleNamespace:
    cfg = hydra_like_sensor_config(PROTOCOL_WIDTH, PROTOCOL_HEIGHT, logical_gpu)
    return SimpleNamespace(
        TORCH_GPU_ID=int(logical_gpu),
        TASK_CONFIG=cfg.TASK_CONFIG,
    )


def resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    import cv2

    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def resize_depth_metric(depth: np.ndarray, size: int) -> np.ndarray:
    import cv2

    image = np.asarray(depth, dtype=np.float32)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.float32)


def _pack_view(raw: Mapping[str, Any], rgb_key: str, depth_key: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(raw[rgb_key], dtype=np.uint8)
    depth = np.asarray(raw[depth_key], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return resize_rgb(rgb, ETP_RGB), resize_depth_metric(depth, ETP_DEPTH)[..., None]


def capture_twelve_views(env: Any) -> dict[str, Any]:
    """12 clockwise views. Prefer native multi-camera sensors; else rotate in place."""

    sim = env.sim
    raw = sim.get_sensor_observations()
    observations: dict[str, Any] = {}
    if "rgb_30" in raw and "depth_30" in raw:
        rgb, depth = _pack_view(raw, "rgb", "depth")
        observations["rgb"] = rgb
        observations["depth"] = depth
        for degrees in range(30, 360, 30):
            rgb, depth = _pack_view(raw, f"rgb_{degrees}", f"depth_{degrees}")
            observations[f"rgb_{degrees}"] = rgb
            observations[f"depth_{degrees}"] = depth
        return observations

    state = sim.get_agent_state()
    position = np.asarray(state.position, dtype=np.float32)
    rotation = state.rotation
    if hasattr(rotation, "imag") and hasattr(rotation, "real"):
        base = float(np.arctan2(rotation.imag[1], rotation.real))
    elif hasattr(rotation, "x"):
        base = 2.0 * math.atan2(float(rotation.y), float(rotation.w))
    else:
        base = 0.0
    try:
        for index in range(12):
            yaw = base + index * (math.pi / 6.0)
            quat = np.quaternion(np.cos(yaw / 2.0), 0.0, np.sin(yaw / 2.0), 0.0)
            sim.set_agent_state(position, quat)
            frame = sim.get_sensor_observations()
            rgb, depth = _pack_view(frame, "rgb", "depth")
            key = "" if index == 0 else f"_{index * 30}"
            observations[f"rgb{key}"] = rgb
            observations[f"depth{key}"] = depth
    finally:
        sim.set_agent_state(position, rotation)
    return observations


def panoramic_camera_orientations(num_views: int = 12) -> dict[str, list[float]]:
    """Match GridMM ``get_camera_orientations``: yaw-only extra Habitat cameras."""

    base_deg = 360.0 / num_views
    base_rad = math.pi / 6.0
    return {
        str(int(base_deg * index)): [0.0, base_rad * index, 0.0]
        for index in range(1, num_views)
    }


def register_gridmm_panoramic_cameras(cfg: Any) -> None:
    """Add rgb_30..rgb_330 and matching depth cameras (native GridMM 12-view)."""

    orientations = panoramic_camera_orientations(12)
    try:
        cfg.set_new_allowed(True)
    except Exception:
        pass
    sizes = {"RGB": (ETP_RGB, ETP_RGB), "DEPTH": (ETP_DEPTH, ETP_DEPTH)}
    sensors = list(cfg.SIMULATOR.AGENT_0.SENSORS)
    for sensor_type, (width, height) in sizes.items():
        sensor = getattr(cfg.SIMULATOR, f"{sensor_type}_SENSOR")
        for action, orient in orientations.items():
            camera_template = f"{sensor_type}_{action}"
            camera_config = copy.deepcopy(sensor)
            try:
                camera_config.set_new_allowed(True)
            except Exception:
                pass
            if hasattr(camera_config, "defrost"):
                camera_config.defrost()
            camera_config.ORIENTATION = list(orient)
            camera_config.UUID = camera_template.lower()
            camera_config.WIDTH = int(width)
            camera_config.HEIGHT = int(height)
            setattr(cfg.SIMULATOR, camera_template, camera_config)
            if camera_template not in sensors:
                sensors.append(camera_template)
    cfg.SIMULATOR.AGENT_0.SENSORS = sensors


def tokenize_instruction(text: str, bert_path: Path, max_length: int = 80) -> np.ndarray:
    from transformers import BertTokenizer

    tokenizer = tokenize_instruction._tokenizer  # type: ignore[attr-defined]
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(str(bert_path), local_files_only=True)
        tokenize_instruction._tokenizer = tokenizer  # type: ignore[attr-defined]
    encoded = tokenizer.encode(text, add_special_tokens=True)[:max_length]
    return np.asarray(encoded, dtype=np.int64)


tokenize_instruction._tokenizer = None  # type: ignore[attr-defined]


class ETPNavSuffixPolicy:
    """ETPNav GraphMap policy. EP1 starts empty; later episodes may load a saved map."""

    def __init__(self, args: argparse.Namespace) -> None:
        os.chdir(args.etpnav_root)
        habitat_lab = Path("/root/habitat-lab")
        if habitat_lab.is_dir():
            sys.path.insert(0, str(habitat_lab))
        sys.path.insert(1, str(args.etpnav_root))
        import types
        import gym.spaces as spaces
        import torch

        tf = types.ModuleType("tensorflow")
        tf.__spec__ = types.SimpleNamespace(
            name="tensorflow", loader=None, origin=None, submodule_search_locations=[]
        )
        sys.modules.setdefault("tensorflow", tf)

        boto3 = types.ModuleType("boto3")
        botocore = types.ModuleType("botocore")
        botocore_exceptions = types.ModuleType("botocore.exceptions")
        botocore_exceptions.ClientError = type("ClientError", (Exception,), {})
        sys.modules.setdefault("boto3", boto3)
        sys.modules.setdefault("botocore", botocore)
        sys.modules.setdefault("botocore.exceptions", botocore_exceptions)
        try:
            import transformers as _transformers

            sys.modules.setdefault("pytorch_transformers", _transformers)
        except Exception:
            pass

        from habitat_baselines.common.baseline_registry import baseline_registry
        from vlnce_baselines.config.default import get_config
        from vlnce_baselines.models.graph_utils import GraphMap
        from vlnce_baselines.models.utils import angle_feature_torch
        from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks
        from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
        import vlnce_baselines  # noqa: F401
        import habitat_extensions  # noqa: F401

        cfg = get_config(str(args.exp_config))
        cfg.defrost()
        cfg.TORCH_GPU_ID = args.sim_gpu_id
        cfg.TORCH_GPU_IDS = [args.sim_gpu_id]
        cfg.MODEL.TORCH_GPU_ID = args.sim_gpu_id
        cfg.MODEL.pretrained_path = None
        cfg.local_rank = 0
        cfg.GPU_NUMBERS = 1
        cfg.freeze()
        rgb = spaces.Box(low=0, high=255, shape=(ETP_RGB, ETP_RGB, 3), dtype=np.uint8)
        depth = spaces.Box(
            low=0.0, high=10.0, shape=(ETP_DEPTH, ETP_DEPTH, 1), dtype=np.float32
        )
        observation_space = spaces.Dict({"rgb": rgb, "depth": depth})
        action_space = spaces.Discrete(4)
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        policy_cls = baseline_registry.get_policy(cfg.MODEL.policy_name)
        self.policy = policy_cls.from_config(cfg, observation_space, action_space)
        ckpt = torch.load(str(args.ckpt), map_location="cpu")
        state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
        cleaned = {key.replace("module.", ""): value for key, value in state.items()}
        self.policy.load_state_dict(cleaned, strict=False)
        self.policy.to(self.device)
        self.policy.eval()
        self.waypoint = BinaryDistPredictor_TRM(device=self.device)
        packed = torch.load(str(args.waypoint_ckpt), map_location="cpu")
        self.waypoint.load_state_dict(packed["predictor"]["state_dict"])
        self.waypoint.to(self.device)
        self.waypoint.eval()
        for param in self.waypoint.parameters():
            param.requires_grad = False
        self.net = self.policy.net
        self.cfg = cfg
        self.GraphMap = GraphMap
        self.pad_tensors_wgrad = pad_tensors_wgrad
        self.gen_seq_masks = gen_seq_masks
        self.angle_feature_torch = angle_feature_torch
        self.actor_steps = 0
        self.bert_path = args.bert_path
        self.torch = torch
        self.gmap = None
        self.prev_vp = None
        self._txt_key = None
        self._txt_embeds = None
        self._txt_masks = None
        self.loc_noise = float(cfg.IL.loc_noise)
        self.merge_ghost = bool(cfg.MODEL.merge_ghost)

    def begin_episode(
        self,
        position: Sequence[float],
        heading: float,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        self.gmap = self.GraphMap(False, self.loc_noise, self.merge_ghost, 0)
        self.actor_steps = 0
        self._txt_key = None
        self._txt_embeds = None
        self._txt_masks = None
        if state:
            self.gmap.load_state(state, device=self.device)
            current = np.asarray(position, dtype=np.float32)
            self.prev_vp = self.gmap._localize(current, self.gmap.node_pos)
        else:
            self.prev_vp = None

    def reset_episode(self, position: Sequence[float], heading: float) -> None:
        self.begin_episode(position, heading)

    def save_graphmap(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        self.torch.save(self.gmap.export_state(), temporary)
        os.replace(temporary, path)

    def _batch_views(self, views: Mapping[str, Any]):
        torch = self.torch
        batch = {}
        order = ["rgb", "depth"]
        for degrees in range(30, 360, 30):
            order.extend([f"rgb_{degrees}", f"depth_{degrees}"])
        for key in order:
            value = views[key]
            tensor = torch.as_tensor(value, device=self.device).unsqueeze(0)
            if key.startswith("depth"):
                metric = np.asarray(value, dtype=np.float32)
                tensor = torch.as_tensor(metric, device=self.device).unsqueeze(0)
            batch[key] = tensor
        return batch

    def _agent_pose(self, env: Any):
        state = env.sim.get_agent_state()
        pos = np.asarray(state.position, dtype=np.float32)
        ori = np.array([*(state.rotation.imag), state.rotation.real], dtype=np.float32)
        return pos, ori

    def _vp_feature_variable(self, wp_outputs):
        torch = self.torch
        from torch.nn.utils.rnn import pad_sequence

        rgb_fts, dep_fts, loc_fts, nav_types = [], [], [], []
        cand_idxes = np.zeros(12, dtype=np.bool_)
        img_idxes = np.asarray(wp_outputs["cand_img_idxes"][0]).reshape(-1)
        if img_idxes.size:
            cand_idxes[img_idxes] = True
            rgb_fts.append(wp_outputs["cand_rgb"][0])
            dep_fts.append(wp_outputs["cand_depth"][0])
            loc_fts.append(wp_outputs["cand_angle_fts"][0])
            nav_types += [1] * len(wp_outputs["cand_angles"][0])
        rgb_fts.append(wp_outputs["pano_rgb"][0][~cand_idxes])
        dep_fts.append(wp_outputs["pano_depth"][0][~cand_idxes])
        loc_fts.append(wp_outputs["pano_angle_fts"][~cand_idxes])
        nav_types += [0] * (12 - int(np.sum(cand_idxes)))
        return {
            "rgb_fts": self.pad_tensors_wgrad([torch.cat(rgb_fts, dim=0)]),
            "dep_fts": self.pad_tensors_wgrad([torch.cat(dep_fts, dim=0)]),
            "loc_fts": self.pad_tensors_wgrad([torch.cat(loc_fts, dim=0)]).to(self.device),
            "nav_types": pad_sequence(
                [torch.LongTensor(nav_types)], batch_first=True
            ).to(self.device),
            "view_lens": torch.LongTensor([len(nav_types)]).to(self.device),
        }

    def _nav_gmap_variable(self, cur_vp, cur_pos, cur_ori):
        torch = self.torch
        from torch.nn.utils.rnn import pad_sequence
        from vlnce_baselines.models.graph_utils import MAX_DIST

        gmap = self.gmap
        node_vp_ids = list(gmap.node_pos.keys())
        ghost_vp_ids = list(gmap.ghost_pos.keys())
        reachable, unreachable = [], []
        for gvp in ghost_vp_ids:
            _front_dis, front_vp = gmap.front_to_ghost_dist(gvp)
            if gmap.shortest_dist.get(cur_vp, {}).get(front_vp) is not None or front_vp == cur_vp:
                reachable.append(gvp)
            else:
                unreachable.append(gvp)
        no_vp_left = len(reachable) == 0
        gmap_vp_ids = [None] + node_vp_ids + reachable + unreachable
        gmap_step_ids = [0] + [gmap.node_stepId.get(vp, 0) for vp in node_vp_ids] + [0] * (
            len(reachable) + len(unreachable)
        )
        gmap_visited_masks = (
            [0] + [1] * len(node_vp_ids) + [0] * len(reachable) + [1] * len(unreachable)
        )
        ordered_ghosts = reachable + unreachable
        img = [gmap.get_node_embeds(vp) for vp in node_vp_ids] + [
            gmap.get_node_embeds(vp) for vp in ordered_ghosts
        ]
        if not img:
            raise RuntimeError("GraphMap has no nodes to encode")
        gmap_img_fts = torch.stack([torch.zeros_like(img[0])] + img, dim=0)
        gmap_pos_fts = gmap.get_pos_fts(cur_vp, cur_pos, cur_ori, gmap_vp_ids)
        pair = np.zeros((len(gmap_vp_ids), len(gmap_vp_ids)), dtype=np.float32)
        for j in range(1, len(gmap_vp_ids)):
            for k in range(j + 1, len(gmap_vp_ids)):
                vp1, vp2 = gmap_vp_ids[j], gmap_vp_ids[k]
                if not vp1.startswith("g") and not vp2.startswith("g"):
                    dist = gmap.graph_hop(vp1, vp2)
                elif not vp1.startswith("g") and vp2.startswith("g"):
                    front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                    dist = gmap.graph_hop(vp1, front_vp2) + front_dis2
                else:
                    front_dis1, front_vp1 = gmap.front_to_ghost_dist(vp1)
                    front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                    dist = front_dis1 + gmap.graph_hop(front_vp1, front_vp2) + front_dis2
                pair[j, k] = pair[k, j] = dist / MAX_DIST
        gmap_lens = torch.LongTensor([len(gmap_vp_ids)])
        return {
            "gmap_vp_ids": [gmap_vp_ids],
            "gmap_step_ids": pad_sequence(
                [torch.LongTensor(gmap_step_ids)], batch_first=True
            ).to(self.device),
            "gmap_img_fts": self.pad_tensors_wgrad([gmap_img_fts]),
            "gmap_pos_fts": self.pad_tensors_wgrad(
                [torch.from_numpy(gmap_pos_fts)]
            ).to(self.device),
            "gmap_masks": self.gen_seq_masks(gmap_lens).to(self.device),
            "gmap_visited_masks": pad_sequence(
                [torch.BoolTensor(gmap_visited_masks)], batch_first=True
            ).to(self.device),
            "gmap_pair_dists": torch.from_numpy(pair).float().unsqueeze(0).to(self.device),
            "no_vp_left": no_vp_left,
        }

    def _ensure_language(self, instruction: str):
        torch = self.torch
        if instruction == self._txt_key and self._txt_embeds is not None:
            return
        tokens = tokenize_instruction(instruction, self.bert_path)
        txt_ids = torch.as_tensor(tokens, device=self.device).unsqueeze(0)
        txt_masks = txt_ids != 0
        self._txt_embeds = self.net(mode="language", txt_ids=txt_ids, txt_masks=txt_masks)
        self._txt_masks = txt_masks
        self._txt_key = instruction

    def observe_and_act(
        self,
        env: Any,
        instruction: str,
        position: Sequence[float],
        heading: float,
        *,
        act: bool,
    ) -> tuple[int | None, float | None, float | None]:
        torch = self.torch
        views = capture_twelve_views(env)
        batch = self._batch_views(views)
        cur_pos, cur_ori = self._agent_pose(env)
        with torch.no_grad():
            wp_outputs = self.net(
                mode="waypoint",
                waypoint_predictor=self.waypoint,
                observations=batch,
                in_train=False,
            )
            vp_inputs = self._vp_feature_variable(wp_outputs)
            vp_inputs["mode"] = "panorama"
            pano_embeds, pano_masks = self.net(**vp_inputs)
            avg_pano = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / torch.sum(
                pano_masks, 1, keepdim=True
            )
            cur_vp, cand_vp, cand_pos = self.gmap.identify_node(
                cur_pos, cur_ori, wp_outputs["cand_angles"][0], wp_outputs["cand_distances"][0]
            )
            cand_embeds = pano_embeds[0][vp_inputs["nav_types"][0] == 1]
            step_id = self.actor_steps + 1 if act else max(self.actor_steps, 0) + 1
            self.gmap.update_graph(
                self.prev_vp,
                step_id,
                cur_vp,
                cur_pos,
                avg_pano[0],
                cand_vp,
                cand_pos,
                cand_embeds,
                None,
            )
            self.prev_vp = cur_vp
            if not act:
                return None, None, None
            self.actor_steps += 1
            self._ensure_language(instruction)
            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
            no_vp_left = nav_inputs.pop("no_vp_left")
            nav_inputs.update(
                {
                    "mode": "navigation",
                    "txt_embeds": self._txt_embeds,
                    "txt_masks": self._txt_masks,
                }
            )
            nav_outs = self.net(**nav_inputs)
            logits = nav_outs["global_logits"]
            index = int(logits.argmax(dim=-1).item())
            if (
                index == 0
                or no_vp_left
                or self.actor_steps >= MAX_HIGH_LEVEL
            ):
                return Actions.STOP, None, None
            ghost_vp = nav_inputs["gmap_vp_ids"][0][index]
            ghost_pos = self.gmap.ghost_aug_pos[ghost_vp]
            from vlnce_baselines.common.environments import calculate_vp_rel_pos
            from habitat_extensions.utils import heading_from_quaternion

            state = env.sim.get_agent_state()
            ang, dis = calculate_vp_rel_pos(
                state.position, ghost_pos, heading_from_quaternion(state.rotation)
            )
            if self.cfg.MODEL.consume_ghost:
                self.gmap.delete_ghost(ghost_vp)
            return None, float(ang), float(dis)


def execute_waypoint(
    env: Any,
    adapter: Any,
    angle_rad: float,
    distance_m: float,
    poses: list[dict[str, Any]],
    observations_holder: dict[str, Any],
) -> None:
    """Match GridMM HIGHTOLOW: snap heading, then 0.25 m forwards."""

    sim = env.sim
    state = sim.get_agent_state()
    rotation = state.rotation
    if hasattr(rotation, "imag") and hasattr(rotation, "real"):
        theta = np.arctan2(rotation.imag[1], rotation.real) + angle_rad / 2.0
    else:
        theta = angle_rad / 2.0
    snapped = np.quaternion(np.cos(theta), 0.0, np.sin(theta), 0.0)
    sim.set_agent_state(state.position, snapped)
    steps = max(1, int(distance_m // FORWARD_STEP_M)) if distance_m >= FORWARD_STEP_M else 0
    if steps == 0:
        pose = pose_from_env(env, len(poses), len(poses) - 1)
        poses[-1] = pose
        observations_holder["obs"] = env.sim.get_sensor_observations()
        adapter.process_observation(
            memory_observation(observations_holder["obs"], pose), step=len(poses) - 1
        )
        return
    for _ in range(steps):
        if getattr(env, "episode_over", False):
            return
        observations_holder["obs"] = env.step(Actions.MOVE_FORWARD)
        pose = pose_from_env(env, len(poses), len(poses))
        poses.append(pose)
        adapter.process_observation(
            memory_observation(observations_holder["obs"], pose), step=len(poses) - 1
        )


def publish_m1(m0: Path, run_dir: Path, destination: Path, metadata: Mapping[str, Any]) -> Path:
    path = janus_publish_m1(m0, run_dir, destination, metadata)
    manifest_path = path / "stage_manifest.json"
    manifest = read_json(manifest_path)
    manifest["schema_version"] = "etpnav_level2_memory_module_stage_v1"
    atomic_json(manifest_path, manifest)
    return path


def quat_from_xyzw(xyzw: Sequence[float]):
    x, y, z, w = [float(value) for value in xyzw]
    return np.quaternion(w, x, y, z)


def pose_key(pose: Mapping[str, Any]) -> tuple[float, ...]:
    values = list(pose["position"]) + list(pose["rotation_xyzw"])
    return tuple(round(float(value), 4) for value in values)


def replay_graphmap_from_poses(env: Any, policy: ETPNavSuffixPolicy, poses: Sequence[Mapping[str, Any]]) -> int:
    """Rebuild GraphMap by teleporting through saved EP1 xyz+xyzw poses."""

    if not poses:
        raise RuntimeError("cannot rebuild GraphMap from an empty pose list")
    policy.begin_episode(poses[0]["position"], 0.0)
    simulator = env.sim
    ingested = 0
    seen: set[tuple[float, ...]] = set()
    for pose in poses:
        key = pose_key(pose)
        if key in seen:
            continue
        seen.add(key)
        simulator.set_agent_state(
            np.asarray(pose["position"], dtype=np.float32),
            quat_from_xyzw(pose["rotation_xyzw"]),
        )
        policy.observe_and_act(env, "", pose["position"], 0.0, act=False)
        ingested += 1
    return ingested


def make_env(args: argparse.Namespace, split: str | None = None):
    os.chdir(args.etpnav_root)
    habitat_lab = Path("/root/habitat-lab")
    if habitat_lab.is_dir():
        sys.path.insert(0, str(habitat_lab))
    sys.path.insert(1, str(args.etpnav_root))
    import habitat_extensions  # noqa: F401
    from habitat.core.env import Env
    from habitat_extensions.config.default import get_extended_config

    cfg = get_extended_config(str(args.task_config))
    cfg.defrost()
    cfg.DATASET.DATA_PATH = str(args.dataset)
    cfg.DATASET.SCENES_DIR = str(args.scenes_dir)
    cfg.DATASET.SPLIT = split or getattr(args, "split", None) or "ep1"
    cfg.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = int(args.sim_gpu_id)
    cfg.SIMULATOR.TURN_ANGLE = float(TURN_ANGLE_DEG)
    cfg.SIMULATOR.RGB_SENSOR.WIDTH = PROTOCOL_WIDTH
    cfg.SIMULATOR.RGB_SENSOR.HEIGHT = PROTOCOL_HEIGHT
    cfg.SIMULATOR.DEPTH_SENSOR.WIDTH = PROTOCOL_WIDTH
    cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT = PROTOCOL_HEIGHT
    if hasattr(cfg.SIMULATOR.DEPTH_SENSOR, "NORMALIZE_DEPTH"):
        cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    register_gridmm_panoramic_cameras(cfg)
    cfg.ENVIRONMENT.MAX_EPISODE_STEPS = int(args.max_steps)
    cfg.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    cfg.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE = False
    cfg.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    measurements = list(cfg.TASK.MEASUREMENTS or [])
    for name in ("DISTANCE_TO_GOAL", "SUCCESS", "SPL", "ORACLE_SUCCESS"):
        if name not in measurements:
            measurements.append(name)
    cfg.TASK.MEASUREMENTS = measurements
    cfg.freeze()
    return Env(config=cfg), hydra_like_sensor_config(
        PROTOCOL_WIDTH, PROTOCOL_HEIGHT, args.sim_gpu_id
    )


class RepeatEpisodeIterator:
    def __init__(self, episode: Any) -> None:
        self.episode = episode

    def __iter__(self):
        return self

    def __next__(self):
        return self.episode


def metric_value(metrics: Mapping[str, Any], name: str):
    value = metrics.get(name)
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def run_one(
    *,
    env: Any,
    habitat_config: Any,
    episode: Any,
    row: Mapping[str, Any],
    gridmm: ETPNavSuffixPolicy,
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

    env.episode_iterator = RepeatEpisodeIterator(episode)
    observations = env.reset()
    initial_pose = pose_from_env(env, 0, 0)
    adapter = OnlineINHeritSGBuilder(
        sensor_memory_config_from_protocol(args.sim_gpu_id),
        width=PROTOCOL_WIDTH,
        height=PROTOCOL_HEIGHT,
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
    gridmm.begin_episode(initial_pose["position"], heading)

    route = row["route"]
    if args.force_etpnav_fallback:
        recall_plan = {
            "status": "fallback",
            "reason": "forced_etpnav_fallback",
            "selected_uid": None,
            "selected_candidate": None,
            "task_text": route.get("recall", {}).get("text"),
        }
        target_info = None
        phase = "gridmm_fallback"
        active_instruction = row["instruction"]
    else:
        recall_plan = (
            load_saved_recall_plan(args.recall_plan_root, row)
            if args.recall_plan_root
            else prepare_recall_plan(adapter, route)
        )
        target_info = None
        phase = "gridmm_fallback"
        active_instruction = row["instruction"]
        if recall_plan["status"] == "ready":
            target_info = make_target_info(
                navigator, recall_plan["selected_candidate"], recall_plan["task_text"]
            )
            if target_info is not None:
                phase = "sg_navigation"
    atomic_json(run_dir / "recall_plan.json", recall_plan)

    poses = [initial_pose]
    requested_actions: list[str] = []
    executed_actions: list[str] = []
    sg_actions = 0
    gridmm_actions = 0
    anchor_reached = False
    fallback_reason = None if target_info is not None else recall_plan.get("reason", "sg_plan_failed")
    stopped = False
    started_at, wall_start = utc_now(), time.monotonic()
    obs_holder = {"obs": observations}

    def make_rollout_record(sg_handoff: Mapping[str, Any]) -> dict[str, Any]:
        metrics = env.get_metrics()
        ne = metric_value(metrics, "distance_to_goal")
        oracle_ne = metric_value(metrics, "oracle_navigation_error")
        return {
            "schema_version": "etpnav_level2_ep1_memory_module_rollout_v1",
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
                "native_graphmap_imported": False,
                "native_graphmap_built_from": "ep1_live",
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
                            "decision": "sg_anchor_reached_handoff_to_etpnav",
                            "reach_reason": reached,
                            "merged_suffix": route["merged_suffix"],
                        },
                    )
                    if route["merged_suffix"]:
                        phase = "gridmm_suffix"
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
                        reason = target_info.pop(
                            "_sg_pending_abort_reason", "sg_controller_no_action"
                        )
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
                        phase = "gridmm_fallback"
                        active_instruction = row["instruction"]
                        continue
                    requested = "SG:" + ACTION_NAMES[int(action)]
                    sg_actions += 1
                gridmm.observe_and_act(
                    env,
                    active_instruction,
                    current_pose["position"],
                    heading,
                    act=False,
                )
            else:
                stop_or_wp = gridmm.observe_and_act(
                    env, active_instruction, current_pose["position"], heading, act=True
                )
                action, angle, distance = stop_or_wp
                if action == Actions.STOP or (angle is None and distance is None):
                    requested = "ETPNAV:STOP"
                    action = Actions.STOP
                    gridmm_actions += 1
                else:
                    requested = f"ETPNAV:HIGHTOLOW:{angle:.3f}:{distance:.3f}"
                    requested_actions.append(requested)
                    executed_actions.append("HIGHTOLOW")
                    execute_waypoint(
                        env, adapter, float(angle), float(distance), poses, obs_holder
                    )
                    gridmm_actions += 1
                    if env.episode_over:
                        stopped = True
                    continue

            requested_actions.append(requested)
            executed_actions.append(ACTION_NAMES[int(action)])
            observations = env.step(int(action))
            obs_holder["obs"] = observations
            pose = pose_from_env(env, len(poses), len(executed_actions))
            poses.append(pose)
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
            "etpnav_instruction_mode": (
                "merged_type2_suffix"
                if anchor_reached and route["merged_suffix"]
                else "stop_after_pure_recall"
                if anchor_reached
                else "full_instruction_fallback"
            ),
            "merged_type2_count": len(route["suffix_parts"]),
            "merged_suffix": route["merged_suffix"],
            "sg_actions": sg_actions,
            "etpnav_actions": gridmm_actions,
            "etpnav_actor_steps": gridmm.actor_steps,
            "native_graphmap_built_from": "ep1_live",
            "navigation_controller_version": NAVIGATION_CONTROLLER_VERSION,
        }
        result = make_rollout_record(sg_handoff)
        atomic_json(run_dir / "rollout.json", result)
        publish_m1(
            row["m0"],
            run_dir,
            destination,
            {
                "session_id": session_id,
                "episode_id": episode_id,
                "route": route,
                "sg_handoff": sg_handoff,
            },
        )
        gridmm.save_graphmap(destination / "graphmap.pt")
        gridmm.save_graphmap(run_dir / "graphmap.pt")
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


def write_summary(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]], vram: Mapping[str, Any]) -> None:
    metrics = ("success", "spl", "oracle_success", "distance_to_goal")
    means = {}
    for name in metrics:
        values = [row["metrics"].get(name) for row in rows]
        values = [float(value) for value in values if value is not None]
        means[name] = sum(values) / len(values) if values else None
    atomic_json(
        args.output_root / "summary.json",
        {
            "schema_version": "etpnav_level2_ep1_memory_module_summary_v1",
            "episode_count": len(rows),
            "means": means,
            "anchor_reached_count": sum(bool(row["sg_handoff"]["anchor_reached"]) for row in rows),
            "sg_fallback_count": sum(
                row["sg_handoff"]["fallback_reason"] is not None for row in rows
            ),
            "vram_peak_single_process": dict(vram),
        },
    )


def consolidate_output(args: argparse.Namespace, vram: Mapping[str, Any]) -> list[dict[str, Any]]:
    lock_path = args.output_root / ".aggregate.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        all_rows = []
        for path in sorted((args.output_root / "raw").glob("ep1_*.jsonl")):
            all_rows.extend(read_jsonl(path))
        by_episode = {str(row["episode_id"]): row for row in all_rows}
        ordered = [by_episode[key] for key in sorted(by_episode, key=lambda item: str(item))]
        atomic_jsonl(args.output_root / "episodes.jsonl", ordered)
        write_summary(args, ordered, vram)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return ordered


def main() -> None:
    print(f"[ETPNAV-EP1] argv={sys.argv}", flush=True)
    args = parse_args()
    args.etpnav_root = args.etpnav_root.resolve()

    def _rebase(path: Path, *relative: str) -> Path:
        raw = Path(path)
        if raw.exists():
            return raw.resolve()
        candidate = args.etpnav_root.joinpath(*relative)
        if candidate.exists():
            return candidate.resolve()
        return raw

    args.exp_config = _rebase(args.exp_config, "run_r2r/iter_level2.yaml")
    args.task_config = _rebase(args.task_config, "run_r2r/hm3d_level2.yaml")
    args.dataset = _rebase(
        args.dataset, "data/datasets/HM3D_VLNCE_v5/level2/level2_ep1.json.gz"
    )
    for candidate in (
        Path("/etpnav/data/datasets/HM3D_VLNCE_v5/level2/level2_ep1.json.gz"),
        args.etpnav_root / "data/datasets/HM3D_VLNCE_v5/level2/level2_ep1.json.gz",
    ):
        if candidate.is_file():
            args.dataset = candidate.resolve()
            break
    args.waypoint_ckpt = _rebase(args.waypoint_ckpt, "data/wp_pred/check_cwp_bestdist_hfov90")
    args.ckpt = Path(args.ckpt)
    if not args.ckpt.exists():
        mapped = args.etpnav_root / "data/logs/checkpoints/release_r2r/ckpt.iter12000.pth"
        if mapped.exists():
            args.ckpt = mapped
    args.ckpt = args.ckpt.resolve()
    args.exp_config = args.exp_config.resolve()
    args.task_config = args.task_config.resolve()
    args.dataset = args.dataset.resolve()
    args.waypoint_ckpt = args.waypoint_ckpt.resolve()
    args.m0_root = args.m0_root.resolve()
    args.route_summary = args.route_summary.resolve()
    args.recall_plan_root = [path.resolve() for path in args.recall_plan_root]
    args.output_root = args.output_root.resolve()
    args.query_vocabulary = args.query_vocabulary.resolve()
    args.scenes_dir = args.scenes_dir.resolve()
    args.bert_path = args.bert_path.resolve()
    args.run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # janus_validate_inputs expects Janus dataset+config fields; reuse route/M0 checks
    # by constructing a Janus-like namespace.
    janus_args = copy.copy(args)
    janus_args.config = args.task_config
    janus_args.janus_root = JANUS_EP1_DIR
    print(
        f"[ETPNAV-EP1] dataset={args.dataset} exists={Path(args.dataset).is_file()}",
        flush=True,
    )
    janus_args.dataset = Path(args.dataset)
    routes = load_route_index(janus_args)
    payload = read_json(Path(args.dataset))
    sessions = {}
    for episode in payload.get("episodes") or []:
        info = episode.get("info") or {}
        if str(info.get("role") or info.get("lifelong_role") or info.get("level2_role")) != "ep1":
            continue
        episode_id = str(episode["episode_id"])
        session_id = episode_id.rsplit("_", 1)[0] if episode_id.endswith("_ep1") else episode_id
        sessions[session_id] = {
            "episode_id": str(episode["episode_id"]),
            "session_id": session_id,
            "instruction": normalize_text(
                (episode.get("instruction") or {}).get("instruction_text")
            ),
        }
    selected = []
    for session_id, dataset_row in sessions.items():
        if dataset_row["episode_id"] not in routes:
            raise RuntimeError(f"missing Uni-LaViRA route for {dataset_row['episode_id']}")
        route = routes[dataset_row["episode_id"]]
        m0 = args.m0_root / "sessions" / session_id / "stages/M0"
        missing = [name for name in CORE_M0_FILES if not (m0 / name).is_file()]
        if missing:
            raise RuntimeError(f"M0 {session_id} lacks {missing}: {m0}")
        selected.append({**dataset_row, "route": route, "m0": m0})
    requested_sessions = set(map(str, args.session_id))
    requested_episodes = set(map(str, args.episode_id))
    selected = [
        row
        for row in selected
        if (not requested_sessions or row["session_id"] in requested_sessions)
        and (not requested_episodes or row["episode_id"] in requested_episodes)
    ]
    selected = [
        row for index, row in enumerate(selected) if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    print(
        __import__("json").dumps(
            {"selected": len(selected), "shard": [args.shard_index, args.num_shards]},
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run or not selected:
        return

    sys.path.insert(0, str(REPO_ROOT))
    install_full_exploration_overlay()
    np.random.seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    worker = args.worker_id.strip() or f"shard{args.shard_index:02d}"
    raw_log = args.output_root / "raw" / f"ep1_{worker}.jsonl"
    completed = {str(row["episode_id"]) for row in read_jsonl(raw_log)}

    tracker = VramTracker(args.device)
    tracker.start(1.0)
    env, habitat_config = make_env(args)
    gridmm = ETPNavSuffixPolicy(args)
    episodes_by_id = {str(ep.episode_id): ep for ep in env.episodes}
    failures = []
    try:
        for index, row in enumerate(selected, 1):
            episode_id = row["episode_id"]
            if episode_id in completed:
                continue
            destination = args.output_root / "sessions" / row["session_id"] / "stages/M1"
            if destination.is_dir() and (destination / "rollout.json").is_file():
                recovered = read_json(destination / "rollout.json")
                append_jsonl(raw_log, recovered)
                completed.add(episode_id)
                continue
            print(
                f"[ETPNAV-EP1] {index}/{len(selected)} {row['session_id']} ep={episode_id}",
                flush=True,
            )
            episode = episodes_by_id.get(episode_id)
            if episode is None:
                raise RuntimeError(f"EP1 {episode_id} is not in {args.dataset}")
            try:
                result = run_one(
                    env=env,
                    habitat_config=habitat_config,
                    episode=episode,
                    row=row,
                    gridmm=gridmm,
                    args=args,
                )
                append_jsonl(raw_log, result)
                completed.add(episode_id)
                print(
                    __import__("json").dumps(
                        {
                            "episode": episode_id,
                            "metrics": result["metrics"],
                            "handoff": result["sg_handoff"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception as exc:
                import traceback

                failure = {
                    "session_id": row["session_id"],
                    "episode_id": episode_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "time": utc_now(),
                }
                failures.append(failure)
                append_jsonl(args.output_root / "raw" / f"failures_{worker}.jsonl", failure)
                print(f"[ETPNAV-EP1] failed: {failure['error']}\n{failure['traceback']}", file=sys.stderr, flush=True)
                if not args.continue_on_error:
                    raise
    finally:
        vram = tracker.stop()
        atomic_json(args.output_root / f"vram_peak.{worker}.json", vram)
        print(__import__("json").dumps({"vram_peak": vram}, ensure_ascii=False), flush=True)
        env.close()

    consolidate_output(args, vram)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
