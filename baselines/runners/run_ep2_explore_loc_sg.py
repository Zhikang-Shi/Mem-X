#!/usr/bin/env python3
"""Level-3 EP2: explore to lock, then SG-goal in EP0-local. Total 400.

``--nav-model etpnav`` uses ETPNav waypoints. ``--nav-model janus`` uses
JanusVLN discrete actions. Explore, geometric lock, and scene-graph control
stay the same. EP1 that did not lock skips explore and uses the nav model.

* Explore <= 100 (occupancy frontier + DINO heading, same skeleton as Janus EP2).
  Lock stops explore immediately.
* If still open at 100, ETPNav continues the instruction and keeps locking.
* Dual-peak 24x15° is loc-only: heading restored, not fed to ETPNav, not counted.
* After lock, T maps the current pose into EP0-local. If SG recall is ready,
  navigation uses the old global SG stack: SparseWorldMetricMap (every
  streamed RGB-D, not keyframes) + certified trajectory corridor + Theta*.
  Habitat navmesh is not used for routes or terminal-island checks because
  the map is EP0-local, not live world coordinates.
* Total budget 400, including explore.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import shutil
import sys
import time
import uuid
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from bootstrap import VOCABULARY, install

install()
_RUNNERS = Path(__file__).resolve().parent
HERE = _RUNNERS.parent
import run_ep1_memory_etpnav_local_loc as loc  # noqa: E402
import run_ep2_memory_janus as janus_l3  # noqa: E402

from actions import command_toward, heading_to_point  # noqa: E402
from dino import encode_rgbs  # noqa: E402
from frames import EpisodeStart  # noqa: E402
from frontiers import extract_frontiers  # noqa: E402
from geom_lock import QUERY_WINDOW, extract_map_superpoint, models, try_lock_frame  # noqa: E402
from local_plan import shortest_path_xz  # noqa: E402
from occupancy import IncrementalOccupancy  # noqa: E402
from se2 import gt_ep1_to_ep0, se2_error, wrap_angle, yaw_from_xyzw  # noqa: E402

from run_ep1_memory_janus import (  # noqa: E402
    ACTION_NAMES,
    Actions,
    CORE_M0_FILES,
    MemoryNavigator,
    append_jsonl,
    atomic_json,
    install_full_exploration_overlay,
    pose_from_env,
    read_json,
    read_jsonl,
    utc_now,
)


# ETPNav policy, env, and metric helpers used by this EP2 runner.
PROTOCOL_WIDTH = 960
PROTOCOL_HEIGHT = 720
ETP_RGB = 224
ETP_DEPTH = 256
FORWARD_STEP_M = 0.25
TURN_ANGLE_DEG = 30.0
MAX_HIGH_LEVEL = 15

def hydra_like_sensor_config(
    width: int, height: int, gpu: int, turn_angle: float = TURN_ANGLE_DEG
) -> SimpleNamespace:
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
        turn_angle=float(turn_angle),
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


def make_janus_env(args: argparse.Namespace):
    """Habitat env from the Janus Level-3 config. StreamVLN's Habitat, not ETPNav's."""
    root = Path(args.janus_root)
    if (root / "src" / "protocol_benchmark.py").is_file():
        code, tree = root / "src", root
    elif (root / "protocol_benchmark.py").is_file():
        code, tree = root, root.parent
    else:
        raise FileNotFoundError(f"protocol_benchmark.py missing under {root}")
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    os.chdir(tree)
    from habitat import Env
    import habitat
    from habitat_baselines.config.default import get_config as get_habitat_config

    config_path = Path(getattr(args, "janus_config", tree / "config/vln_level3_protocol.yaml"))
    dataset = Path(
        getattr(
            args,
            "janus_dataset",
            Path("/janus/data/level3_v34_20260906/habitat/level3/level3.json.gz"),
        )
    )
    import habitat_extensions.measures  # noqa: F401  registers OracleSuccess and the other Level-3 measures
    config = get_habitat_config(str(config_path))
    with habitat.config.read_write(config):
        config.habitat.environment.max_episode_steps = int(args.max_steps)
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = int(args.sim_gpu_id)
        config.habitat.dataset.data_path = str(dataset)
        config.habitat.dataset.split = str(getattr(args, "dataset_split", None) or "level3")
        if getattr(args, "apply_turn_angle", False):
            config.habitat.simulator.turn_angle = int(args.turn_angle)
    return Env(config=config), config


def episodes_for_nav(env, nav_model: str) -> dict[str, Any]:
    if nav_model != "janus":
        return {str(episode.episode_id): episode for episode in env.episodes}
    selected = {}
    for episode in env.episodes:
        info = dict(getattr(episode, "info", {}) or {})
        if str(info.get("lifelong_role") or info.get("level3_role")) != "ep2":
            continue
        episode_id = str(getattr(episode, "episode_id", "") or "")
        task_id = episode_id.rsplit("_", 1)[0] if episode_id.endswith(("_ep0", "_ep1", "_ep2", "_ep3", "_ep4")) else str(info.get("session_id") or "")
        if task_id:
            selected[task_id] = episode
    return selected


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
    turn_angle = float(getattr(args, "turn_angle", TURN_ANGLE_DEG))
    cfg.SIMULATOR.TURN_ANGLE = turn_angle
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
        PROTOCOL_WIDTH, PROTOCOL_HEIGHT, args.sim_gpu_id, turn_angle
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


HIGH_INLIER = loc.HIGH_INLIER
LOCK_EVERY_M = loc.LOCK_EVERY_M
SPIN_STEPS = loc.SPIN_STEPS
HEADING_SPIN = 12
HEADING_TURN = math.radians(30.0)
EXPLORE_BUDGET = 100
SG_PLANNER = "world_metric_theta_star_with_certified_trajectory_corridor"
DEFAULT_SESSION_FILE = HERE / "ep2_locked133.txt"
DEFAULT_M0 = HERE / "outputs/ep0_memory_from_poses_local_latest"
DEFAULT_M1 = HERE / "outputs/ep1_local_loc_latest"
DEFAULT_OUTPUT = Path(
    "/share/shared/shizk/level3_outputs/etpnav_memory_module/ep2_explore_loc_sg"
)
DEFAULT_DATASET = loc.ETP_ROOT / "data/datasets/HM3D_VLNCE_v5/level3/level3_ep2.json.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etpnav-root", type=Path, default=loc.ETP_ROOT)
    parser.add_argument("--exp-config", type=Path, default=loc.DEFAULT_EXP_CONFIG)
    parser.add_argument("--task-config", type=Path, default=loc.DEFAULT_TASK_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ckpt", type=Path, default=loc.etp.DEFAULT_CKPT)
    parser.add_argument("--waypoint-ckpt", type=Path, default=loc.etp.DEFAULT_WAYPOINT)
    parser.add_argument("--m0-root", type=Path, default=DEFAULT_M0)
    parser.add_argument("--m1-root", type=Path, default=DEFAULT_M1)
    parser.add_argument("--lavira-parse-root", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--session-id", action="append", default=[])
    parser.add_argument("--session-id-file", type=Path, default=DEFAULT_SESSION_FILE)
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--path-map", action="append", default=[])
    parser.add_argument("--run-id", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-gpu-id", type=int, default=0)
    parser.add_argument("--explore-steps", type=int, default=EXPLORE_BUDGET)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--worker-id", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expected-count", type=int, default=133)
    parser.add_argument(
        "--query-vocabulary",
        type=Path,
        default=VOCABULARY,
    )
    parser.add_argument(
        "--recall-policy",
        default="any_memory",
        choices=("any_memory", "both_prior_episodes"),
    )
    parser.add_argument("--bert-path", type=Path, default=Path(os.environ.get("MODELS_DIR", "")) / "bert-base-uncased")
    parser.add_argument("--scenes-dir", type=Path, default=Path(os.environ.get("SCENES_DIR", "")))
    parser.add_argument(
        "--nav-model",
        choices=("etpnav", "janus", "cma", "unilavira"),
        default="etpnav",
    )
    parser.add_argument(
        "--cma-root",
        type=Path,
        default=Path("/home/shizk/LifeLongHomeNav/baseline_experiment/CMA"),
    )
    parser.add_argument(
        "--r2r-train",
        type=Path,
        default=Path(
            "/home/shizk/LifeLongHomeNav/baseline_experiment/CMA/data/datasets/"
            "HM3D_VLNCE_v5/level3/level3_ep2.json.gz"
        ),
    )
    parser.add_argument(
        "--janus-root",
        type=Path,
        default=Path("/janus/src/JanusVLN"),
    )
    parser.add_argument(
        "--janus-model",
        type=Path,
        default=Path("/janus/models/JanusVLN_Extra"),
    )
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--ep1-lock-file", type=Path)
    parser.add_argument(
        "--update-memory",
        choices=("off", "nav", "full"),
        default="full",
        help="off freezes M1. nav updates the metric map and trajectory. full also updates the semantic scene graph.",
    )
    args = parser.parse_args()
    if not args.lavira_parse_root:
        args.lavira_parse_root = [
            Path(
                "/share/shared/shizk/LifeLongHomeNav/level3/outputs/"
                "final_level3_full_20260906/memories"
            ),
            Path("/share/shared/shizk/LifeLongHomeNav/level3/outputs/etpnav_ep2_instruction_parses"),
        ]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.max_steps < 1 or args.explore_steps < 1:
        parser.error("step budgets must be positive")
    if args.explore_steps > args.max_steps:
        parser.error("--explore-steps cannot exceed --max-steps")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def etpnav_continuation(row: Mapping[str, Any], plan: Any, completed_count: int) -> dict[str, Any]:
    """English ETPNav handoff. English suffix for the ETPNav handoff."""
    handoff = janus_l3.level3_continuation(row["route"], plan, completed_count)
    english = str(
        (row.get("route") or {}).get("dataset_instruction_en") or row.get("instruction") or ""
    ).strip()
    handoff = dict(handoff)
    if completed_count >= 2 and english:
        parts = [part.strip() for part in english.replace("!", ".").split(".") if part.strip()]
        handoff["continuation"] = (parts[-1] + ".") if parts else english
        handoff["etpnav_language"] = "en_suffix"
    else:
        handoff["continuation"] = english or handoff.get("continuation")
        handoff["etpnav_language"] = "en_full"
    return handoff


def read_session_ids(args: argparse.Namespace) -> list[str]:
    ids = [str(item) for item in args.session_id]
    if args.session_id_file and args.session_id_file.is_file() and not ids:
        for line in args.session_id_file.read_text(encoding="utf-8").splitlines():
            sid = line.strip()
            if sid and not sid.startswith("#"):
                ids.append(sid)
    return list(dict.fromkeys(ids))


def load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    mappings = janus_l3.common.parse_path_maps(args.path_map)
    tasks = janus_l3.dataset_tasks(args.dataset)
    requested = set(read_session_ids(args))
    requested_episodes = set(map(str, args.episode_id))
    if not requested and not requested_episodes:
        raise RuntimeError("EP2 explore-loc needs --session-id or --session-id-file")
    unknown = requested - set(tasks)
    if unknown:
        raise RuntimeError(f"unknown EP2 sessions: {sorted(unknown)[:8]}")
    if args.expected_count and not args.session_id and not args.episode_id:
        if len(requested) != args.expected_count:
            raise RuntimeError(
                f"Expected {args.expected_count} lock-correct sessions, found {len(requested)}"
            )
    rows = []
    missing = []
    for task_id, dataset_row in tasks.items():
        if requested and task_id not in requested:
            continue
        if requested_episodes and dataset_row["episode_id"] not in requested_episodes:
            continue
        m0 = args.m0_root / "sessions" / task_id / "stages/M0"
        m1 = args.m1_root / "sessions" / task_id / "stages/M1"
        lack = [name for name in CORE_M0_FILES if not (m0 / name).is_file()]
        if not (m0 / "frames" / "confirmed_keyframes.jsonl").is_file():
            lack.append("M0/frames/confirmed_keyframes.jsonl")
        map_frame = "ep0_local"
        if not (m0 / "ep0_local_origin.json").is_file():
            keyframe = {}
            keyframe_path = m0 / "frames" / "confirmed_keyframes.jsonl"
            if keyframe_path.is_file():
                for line in keyframe_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        keyframe = json.loads(line)
                        break
            if keyframe.get("coordinate_frame") in (None, "habitat_world"):
                map_frame = "habitat_world"
            else:
                lack.append("M0/ep0_local_origin.json")
        lack += [f"M1/{name}" for name in janus_l3.CORE_M1_FILES if not (m1 / name).is_file()]
        if lack:
            missing.append(f"{task_id}: {lack}")
            continue
        route = {
            "task_id": task_id,
            "instruction": dataset_row["instruction"],
            "dataset_instruction_en": dataset_row["instruction"],
            "types": None,
            "anchors": [],
            "suffix": "",
            "source": None,
            "source_sha256": None,
            "parsed": None,
        }
        rows.append({
            **dataset_row,
            "m0": m0,
            "m1": m1,
            "route": route,
            "map_coordinate_frame": map_frame,
        })
    if missing:
        preview = "; ".join(missing[:8])
        raise RuntimeError(f"EP2 missing memory for {len(missing)} tasks: {preview}")
    rows.sort(key=lambda item: item["session_id"])
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("EP2 explore-loc selection is empty")
    return rows


def _lookahead(ax: float, az: float, yaw: float, dist: float) -> tuple[float, float]:
    return ax + dist * (-math.sin(yaw)), az + dist * (-math.cos(yaw))


def explore_action(frame, occ: IncrementalOccupancy, pref_yaw: float, path_xz, blocked: set):
    grid = occ.grid()
    if path_xz:
        target = path_xz[-1]
        for point in path_xz[1:]:
            if math.hypot(point[0] - frame.x, point[1] - frame.z) >= 0.35:
                target = point
                break
        action = command_toward(frame.x, frame.z, frame.yaw, target[0], target[1], arrive_m=0.18)
        if action != "stop":
            return action, path_xz
        path_xz = []
    frontiers = extract_frontiers(grid, (frame.x, frame.z))
    best_f = None
    best_s = -1e9
    for frontier in frontiers:
        key = (round(frontier.x, 1), round(frontier.z, 1))
        if key in blocked:
            continue
        err = abs(wrap_angle(heading_to_point(frame.x, frame.z, frontier.x, frontier.z) - pref_yaw))
        if err > math.radians(80.0):
            continue
        score = math.sqrt(max(frontier.info, 0.0)) - 0.25 * frontier.path_m - 1.2 * err
        if score > best_s:
            best_s = score
            best_f = frontier
    path = []
    if best_f is not None:
        path = shortest_path_xz(grid, (frame.x, frame.z), (best_f.x, best_f.z), unknown_cost=8)
    if len(path) < 2:
        for dist_try in (2.0, 1.2):
            gx, gz = _lookahead(frame.x, frame.z, pref_yaw, dist_try)
            path = shortest_path_xz(grid, (frame.x, frame.z), (gx, gz), unknown_cost=8)
            if len(path) >= 2:
                break
    if len(path) < 2:
        return "turn_left", []
    target = path[1]
    action = command_toward(frame.x, frame.z, frame.yaw, target[0], target[1], arrive_m=0.18)
    if action == "stop":
        action = "move_forward"
    return action, path


def candidate_xz(candidate: Mapping[str, Any]) -> tuple[float, float] | None:
    pos = candidate.get("position") or candidate.get("global_pos_3d")
    if pos is None:
        pos = (candidate.get("waypoint") or {}).get("position")
    if pos is None or len(pos) < 3:
        return None
    return float(pos[0]), float(pos[2])


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def metric_map_stats(adapter) -> dict[str, Any]:
    memory = getattr(adapter, "world_memory", None)
    metric_map = getattr(memory, "metric_map", None)
    if metric_map is None:
        return {"source": "missing_world_metric_map"}
    stats = dict(metric_map.stats())
    stats["source"] = "memory_module_world_metric_map_all_rgbd"
    return stats


def disable_navmesh_for_ep0_local(navigator) -> None:
    """Keep Theta* on the EP0-local metric map; do not query Habitat navmesh."""

    navigator._sg_live_navmesh_context = lambda: None
    navigator._sg_navmesh_context = lambda: None
    navigator._sg_world_navmesh_target_validator = lambda *args, **kwargs: (
        None,
        {
            "status": "disabled_ep0_local_frame",
            "accepted_bounded_snap": 0,
            "rejected_other_island": 0,
            "rejected_snap_too_far": 0,
        },
    )
    navigator._sg_validate_warm_start_connector = lambda start, end: (
        True,
        {"status": "accepted_without_navmesh_ep0_local"},
    )


def begin_metric_sg_anchor(navigator, candidate: Mapping[str, Any], task_text: str):
    """Old global SG: corridor-certified Theta* on world_metric_map, no navmesh."""

    current = navigator._sg_current_world_xz()
    raw = candidate_xz(candidate)
    metric_map = navigator._sg_world_metric_map()
    if current is None or raw is None or metric_map is None:
        return None
    route_meta = navigator._sg_trajectory_route_metadata(candidate, current)
    anchors = navigator._sg_trajectory_route_world_xz(route_meta)
    reachable = navigator._sg_plan_world_path(
        raw,
        current,
        planning_anchor_world_xz=anchors,
        project_target_to_navmesh=False,
    )
    if reachable is None or len(reachable.get("path") or []) < 2:
        return None
    selection = navigator._sg_select_execution_world_path(
        reachable,
        navigator._sg_compose_trajectory_guided_world_path(reachable, route_meta, current),
        allow_shortcut=True,
    )
    if selection is None or len(selection.get("path") or []) < 2:
        return None
    target = copy.deepcopy(dict(candidate))
    if isinstance(route_meta, Mapping):
        target.update(copy.deepcopy(route_meta))
    target.update(
        {
            "todo_text": task_text,
            "query_source": "lavira_level3_saved_parse",
            "target_world_xz": list(map(float, reachable["target_world_xz"])),
            "raw_target_world_xz": [float(raw[0]), float(raw[1])],
            "target_world_grid_x": int(reachable["target_grid"][0]),
            "target_world_grid_z": int(reachable["target_grid"][1]),
            "raw_target_world_grid_x": int(reachable["raw_target_grid"][0]),
            "raw_target_world_grid_z": int(reachable["raw_target_grid"][1]),
            "world_map_resolution_m": float(metric_map.spec.resolution_m),
            "_sg_world_path_cells": list(selection["path"]),
            "_sg_path_progress_cells": 0.0,
            "_sg_route_planning_anchors_world_xz": anchors,
            "execution_path_source": selection.get("execution_path_source"),
            "theta_path_waypoints": len(selection["path"]),
            "theta_path_length_m": float(reachable["path_length_cells"])
            * float(metric_map.spec.resolution_m),
            "planner": SG_PLANNER,
            "navmesh_used": False,
            "coordinate_frame": "ep0_local",
            "trajectory_corridor": jsonable(reachable.get("trajectory_corridor")),
        }
    )
    navigator._sg_world_plan_window = reachable.get("window")
    navigator._sg_world_safe_mask = reachable.get("safe_mask")
    navigator._sg_world_trajectory_probe_mask = reachable.get(
        "_exploration_trajectory_probe_mask"
    )
    navigator._sg_path_plan_active = True
    return target


def nav_phase(policy: Any, name: str) -> str:
    return f"{getattr(policy, 'nav_name', 'etpnav')}_{name}"


def load_ep1_lock_table(path: Path | None) -> dict[str, bool]:
    if path is None or not path.is_file():
        return {}
    table: dict[str, bool] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = str(row.get("task_id") or "")
        if task_id:
            table[task_id] = bool(row.get("locked"))
    return table


def ep1_did_lock(m1: Path, lock_table: Mapping[str, bool] | None = None) -> bool:
    """EP1 geometric lock only. ok_useful needs ground truth and is not used."""
    rollout = m1 / "rollout.json"
    if rollout.is_file():
        try:
            payload = json.loads(rollout.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        loc = payload.get("localization")
        if isinstance(loc, dict) and "locked" in loc:
            return bool(loc.get("locked"))
    if lock_table:
        return bool(lock_table.get(m1.parents[1].name, False))
    return False


def run_one(*, env, habitat_config, episode, row, gridmm, args) -> dict[str, Any]:
    from membridge.habitat_runtime.online_adapter import OnlineINHeritSGBuilder

    session_id, episode_id = row["session_id"], row["episode_id"]
    destination = args.output_root / "sessions" / session_id / "stages/M2"
    if destination.exists() and not args.skip_existing:
        raise FileExistsError(f"M2 already exists: {destination}")
    work_root = args.output_root / ".working" / f"{session_id}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    work_root.mkdir(parents=True)
    janus_l3.install_ep2_memory_environment(args, work_root, row["m1"])

    env.episode_iterator = RepeatEpisodeIterator(episode)
    observations = env.reset()
    initial_pose = pose_from_env(env, 0, 0)
    adapter = OnlineINHeritSGBuilder(
        sensor_memory_config_from_protocol(args.sim_gpu_id),
        width=loc.etp.PROTOCOL_WIDTH,
        height=loc.etp.PROTOCOL_HEIGHT,
        save_dir=str(work_root),
        device=args.device,
    )
    adapter.start_episode(episode_id, scene_id=str(episode.scene_id), source_episode_id=episode_id)
    run_dir = Path(adapter.pipeline.run_dir).resolve()

    navigator = MemoryNavigator(
        adapter=adapter,
        env=env,
        habitat_config=habitat_config,
        decision_log=run_dir / "sg_navigation_decisions.jsonl",
    )
    navigator.current_episode_id = episode_id
    disable_navmesh_for_ep0_local(navigator)

    ep0_start = loc.load_ep0_start(row["m0"], session_id)
    heading0 = yaw_from_xyzw(initial_pose["rotation_xyzw"])
    ep2_live = EpisodeStart(
        xyz=np.asarray(initial_pose["position"], dtype=np.float64),
        yaw=float(heading0),
        episode_id=str(episode_id),
        role="ep2",
    )
    gridmm.begin_episode(initial_pose["position"], heading0)

    ep1_locked = ep1_did_lock(row["m1"])
    if ep1_locked:
        route = janus_l3.parse_level3_instruction_live(row["instruction"])
        route["task_id"] = row["session_id"]
        row["route"] = route
        recall_plan = janus_l3.prepare_dual_anchor_plan(adapter, route, args.recall_policy)
    else:
        route = row["route"]
        recall_plan = {"status": "fallback", "reason": "ep1_not_locked"}
    persistable = {key: value for key, value in recall_plan.items() if key not in {"state", "plan"}}
    atomic_json(run_dir / "recall_plan.json", persistable)
    if recall_plan.get("status") == "ready":
        atomic_json(run_dir / "level3_dual_anchor_recall_plan.json", persistable)
    dual_state = recall_plan.get("state")
    plan = recall_plan.get("plan")

    poses = [initial_pose]
    requested_actions: list[str] = []
    executed_actions: list[str] = []
    n_explore = n_etpnav = n_sg = n_spin = n_spin_bouts = n_lock_try = n_sg_frames = 0
    stopped = False
    sg_ingested = False
    started_at, wall_start = utc_now(), time.monotonic()
    transform = None
    locked = False
    ok_useful = False
    lock_mode = "explore"
    lock_reason = "not_attempted"
    lock_at = None
    best_inlier = 0
    trans_err = yaw_err = None
    last_best = None
    traveled = 0.0
    last_xz = None
    prev_xz = None
    spinning = False
    last_spin_travel = -1e9
    query_win: deque = deque(maxlen=QUERY_WINDOW)
    rgbd_buf: list[dict[str, Any]] = []
    obs_holder = {"obs": observations}
    occ = IncrementalOccupancy()
    map_frames = loc.load_m0_local_frames(row["m0"], session_id)
    models()
    map_pack = (
        map_frames,
        encode_rgbs([frame.rgb for frame in map_frames]),
        extract_map_superpoint(map_frames),
    )
    mem_feats = map_pack[1]
    views = []
    spin_base = None
    pref_yaw = None
    last_vis = 0.0
    path_xz: list = []
    blocked: set = set()
    since = 0
    if not ep1_locked:
        phase = nav_phase(gridmm, "native")
        fallback_reason = "ep1_not_locked"
    else:
        phase = "explore"
        fallback_reason = None
    target_info = None
    current_anchor_index = None
    anchors_reached: list[str] = []
    active_instruction = row["instruction"]
    occupancy_meta: dict[str, Any] = {}

    def habitat_heading(pose) -> float:
        return yaw_from_xyzw(pose["rotation_xyzw"])

    def aligned_pose(pose):
        if transform is None:
            return pose["position"], habitat_heading(pose)
        return loc.ep1_to_ep0_local(
            transform, pose["position"], habitat_heading(pose), ep2_live
        )

    def ingest_aligned(rgb, depth, pose, step: int) -> None:
        nonlocal n_sg_frames
        if args.update_memory == "off":
            return
        adapter.process_observation(
            loc.aligned_memory_observation(rgb, depth, pose, transform, ep2_live),
            step=step,
        )
        n_sg_frames += 1

    def flush_sg_history() -> None:
        nonlocal sg_ingested
        if sg_ingested or transform is None:
            return
        for index, item in enumerate(rgbd_buf):
            ingest_aligned(item["rgb"], item["depth"], item["pose"], index)
        sg_ingested = True

    def maybe_lock(frame) -> bool:
        nonlocal locked, ok_useful, lock_reason, lock_at, transform, best_inlier
        nonlocal last_xz, last_best, trans_err, yaw_err, n_lock_try
        if locked:
            return False
        n_lock_try += 1
        last_xz = (frame.x, frame.z)
        ok, reason, best = try_lock_frame(
            frame, map_pack[0], map_pack[1], map_pack[2], query_window=list(query_win)
        )
        last_best = best
        best_inlier = max(best_inlier, int(getattr(best, "n_inlier", 0) or 0))
        lock_reason = reason
        if not ok or best.transform is None:
            return False
        locked = True
        transform = best.transform
        lock_at = len(executed_actions)
        gt = gt_ep1_to_ep0(ep0_start.xyz, ep0_start.yaw, ep2_live.xyz, ep2_live.yaw)
        trans_err, yaw_rad = se2_error(best.transform, gt)
        yaw_err = math.degrees(yaw_rad)
        ok_useful = bool(trans_err <= 0.25 and abs(yaw_err) <= 10.0)
        lock_reason = f"{reason} step={lock_at}"
        flush_sg_history()
        return True

    def should_extra_spin() -> bool:
        if traveled - last_spin_travel < 1.0 and n_spin_bouts:
            return False
        return (
            last_best is not None
            and int(getattr(last_best, "n_inlier", 0) or 0) >= HIGH_INLIER
            and "not unique" in (lock_reason or "")
        )

    def extra_spin() -> None:
        nonlocal n_spin, n_spin_bouts, last_spin_travel, spinning, lock_mode
        n_spin_bouts += 1
        last_spin_travel = traveled
        spinning = True
        sim = env.sim
        state = sim.get_agent_state()
        orig_pos = np.asarray(state.position, dtype=np.float32)
        orig_rot = state.rotation
        base_yaw = habitat_heading(poses[-1])
        try:
            for index in range(SPIN_STEPS):
                if locked:
                    break
                yaw = wrap_angle(base_yaw + (index + 1) * (2.0 * math.pi / SPIN_STEPS))
                sim.set_agent_state(
                    orig_pos,
                    np.quaternion(np.cos(yaw / 2.0), 0.0, np.sin(yaw / 2.0), 0.0),
                )
                obs = sim.get_sensor_observations()
                n_spin += 1
                spin_pose = {
                    "position": [float(v) for v in orig_pos],
                    "rotation_xyzw": [
                        0.0,
                        float(np.sin(yaw / 2.0)),
                        0.0,
                        float(np.cos(yaw / 2.0)),
                    ],
                }
                frame = loc.live_frame(obs, spin_pose, ep2_live, len(executed_actions))
                query_win.append(frame)
                if maybe_lock(frame):
                    lock_mode = f"{phase}+spin+geom"
                    break
        finally:
            sim.set_agent_state(orig_pos, orig_rot)
            obs_holder["obs"] = sim.get_sensor_observations()
            spinning = False

    def buffer_obs(pose) -> None:
        rgb, depth = loc.copy_rgbd(obs_holder["obs"])
        rgbd_buf.append({"rgb": rgb, "depth": depth, "pose": copy.deepcopy(pose)})
        if locked and transform is not None:
            if not sg_ingested:
                flush_sg_history()
            else:
                ingest_aligned(rgb, depth, pose, len(executed_actions))

    def after_progress(force: bool = False) -> None:
        nonlocal traveled, prev_xz, lock_mode
        frame = loc.live_frame(obs_holder["obs"], poses[-1], ep2_live, len(executed_actions))
        occ.add_frame(frame)
        if prev_xz is not None:
            traveled += math.hypot(frame.x - prev_xz[0], frame.z - prev_xz[1])
        prev_xz = (frame.x, frame.z)
        if locked or spinning:
            return
        moved = last_xz is None or math.hypot(frame.x - last_xz[0], frame.z - last_xz[1]) >= LOCK_EVERY_M
        if not force and not moved:
            return
        query_win.append(frame)
        if maybe_lock(frame):
            lock_mode = f"{phase}+geom"
        elif should_extra_spin():
            extra_spin()

    def log_sg(payload: dict[str, Any]) -> None:
        append_jsonl(run_dir / "sg_navigation_decisions.jsonl", payload)

    def sync_navigator():
        plan_pos, plan_yaw = aligned_pose(poses[-1])
        navigator.update_pose(plan_pos, plan_yaw, len(executed_actions))
        return plan_pos, plan_yaw

    def plan_anchor(candidate, text):
        nonlocal occupancy_meta
        flush_sg_history()
        occupancy_meta = metric_map_stats(adapter)
        navigator.reset()
        sync_navigator()
        return begin_metric_sg_anchor(navigator, candidate, text)

    def start_sg_if_possible() -> None:
        nonlocal phase, target_info, current_anchor_index, fallback_reason, active_instruction
        if not locked:
            return
        if dual_state is None or plan is None:
            phase = nav_phase(gridmm, "fallback")
            fallback_reason = recall_plan.get("reason") or "recall_not_ready"
            active_instruction = row["instruction"]
            return
        current = dual_state.next_target()
        current_anchor_index = 0
        target_info = plan_anchor(current, plan.anchors[0].segment["text"])
        if target_info is None:
            failure = jsonable(getattr(navigator, "_sg_last_theta_failure", None)) or {}
            phase = nav_phase(gridmm, "fallback")
            fallback_reason = str(failure.get("reason") or "anchor1_world_theta_failed")
            active_instruction = etpnav_continuation(row, plan, 0)["continuation"]
            log_sg(
                {
                    "decision": "sg_world_theta_begin_failed",
                    "anchor_index": 0,
                    "uid": str(current.get("uid") or current.get("node_id") or ""),
                    "failure": failure,
                    "occupancy": occupancy_meta,
                    "planner": SG_PLANNER,
                    "navmesh_used": False,
                    "step": len(executed_actions),
                }
            )
            return
        phase = "sg_navigation"
        fallback_reason = None
        active_instruction = plan.anchors[0].segment["text"]
        log_sg(
            {
                "decision": "sg_world_theta_begin",
                "anchor_index": 0,
                "uid": str(target_info.get("uid") or target_info.get("node_id") or ""),
                "target_world_xz": target_info.get("target_world_xz"),
                "raw_target_world_xz": target_info.get("raw_target_world_xz"),
                "theta_path_waypoints": target_info.get("theta_path_waypoints"),
                "theta_path_length_m": target_info.get("theta_path_length_m"),
                "execution_path_source": target_info.get("execution_path_source"),
                "trajectory_corridor": target_info.get("trajectory_corridor"),
                "occupancy": occupancy_meta,
                "planner": SG_PLANNER,
                "navmesh_used": False,
                "step": len(executed_actions),
            }
        )

    def step_low(action: int, requested: str, kind: str) -> None:
        nonlocal observations, n_explore, n_sg
        requested_actions.append(requested)
        executed_actions.append(ACTION_NAMES[int(action)])
        observations = env.step(int(action))
        obs_holder["obs"] = observations
        pose = pose_from_env(env, len(poses), len(executed_actions))
        poses.append(pose)
        buffer_obs(pose)
        if kind == "explore":
            n_explore += 1
            after_progress(force=requested.endswith("TURN_LEFT") or requested.endswith("TURN_RIGHT"))
        elif kind == "sg":
            n_sg += 1
            after_progress(force=False)
        else:
            after_progress(force=False)

    def execute_waypoint(angle_rad: float, distance_m: float) -> None:
        nonlocal n_etpnav
        sim = env.sim
        state = sim.get_agent_state()
        rotation = state.rotation
        if hasattr(rotation, "imag") and hasattr(rotation, "real"):
            theta = np.arctan2(rotation.imag[1], rotation.real) + angle_rad / 2.0
        else:
            theta = angle_rad / 2.0
        snapped = np.quaternion(np.cos(theta), 0.0, np.sin(theta), 0.0)
        sim.set_agent_state(state.position, snapped)
        steps = max(1, int(distance_m // loc.FORWARD_STEP_M)) if distance_m >= loc.FORWARD_STEP_M else 0
        n_etpnav += 1
        if steps == 0:
            pose = pose_from_env(env, len(poses), len(poses) - 1)
            poses[-1] = pose
            obs_holder["obs"] = env.sim.get_sensor_observations()
            rgb, depth = loc.copy_rgbd(obs_holder["obs"])
            if rgbd_buf:
                rgbd_buf[-1] = {"rgb": rgb, "depth": depth, "pose": copy.deepcopy(pose)}
            elif locked and transform is not None:
                ingest_aligned(rgb, depth, pose, len(executed_actions))
            after_progress(force=False)
            return
        for _ in range(steps):
            if env.episode_over or len(executed_actions) >= args.max_steps:
                return
            step_low(Actions.MOVE_FORWARD, "ETPNAV:MOVE_FORWARD", kind="etpnav")

    def do_explore() -> None:
        nonlocal pref_yaw, last_vis, path_xz, since, spin_base
        frame = loc.live_frame(obs_holder["obs"], poses[-1], ep2_live, len(executed_actions))
        if n_explore < HEADING_SPIN:
            if spin_base is None:
                spin_base = float(frame.yaw)
            views.append(frame.rgb)
            step_low(Actions.TURN_LEFT, "EXPLORE:TURN_LEFT", kind="explore")
            return
        if pref_yaw is None and views:
            feats = encode_rgbs(views)
            vis = feats @ mem_feats.T
            per_view = vis.max(axis=1)
            best = int(np.argmax(per_view))
            pref_yaw = wrap_angle(float(spin_base) + best * HEADING_TURN)
            last_vis = float(per_view[best])
        if pref_yaw is None:
            pref_yaw = float(frame.yaw)
        if since >= 8:
            cur_vis = float(np.max(encode_rgbs([frame.rgb]) @ mem_feats.T))
            if cur_vis + 0.04 < last_vis:
                pref_yaw = float(frame.yaw)
                path_xz = []
            last_vis = max(last_vis, cur_vis)
            since = 0
        action, path_xz = explore_action(frame, occ, pref_yaw, path_xz, blocked)
        before = (frame.x, frame.z)
        step_low(
            int(getattr(Actions, action.upper(), Actions.TURN_LEFT)),
            f"EXPLORE:{action.upper()}",
            kind="explore",
        )
        now = loc.live_frame(obs_holder["obs"], poses[-1], ep2_live, len(executed_actions))
        if action == "move_forward" and math.hypot(now.x - before[0], now.z - before[1]) < 0.04:
            blocked.add(
                (
                    round(frame.x + 0.28 * (-math.sin(frame.yaw)), 1),
                    round(frame.z + 0.28 * (-math.cos(frame.yaw)), 1),
                )
            )
            path_xz = []
            pref_yaw = wrap_angle(pref_yaw + HEADING_TURN)
        since += 1
        gridmm.observe_and_act(
            env,
            row["instruction"],
            poses[-1]["position"],
            habitat_heading(poses[-1]),
            act=False,
        )

    def do_etpnav() -> None:
        nonlocal stopped, observations, n_etpnav
        current_pose = poses[-1]
        heading = habitat_heading(current_pose)
        if getattr(gridmm, "discrete_actions", False):
            action, _, _ = gridmm.observe_and_act(
                env, active_instruction, current_pose["position"], heading, act=True
            )
            label = getattr(gridmm, "nav_name", "etpnav").upper()
            n_etpnav += 1
            if int(action) == int(Actions.STOP):
                requested_actions.append(f"{label}:STOP")
                executed_actions.append("STOP")
                observations = env.step(int(Actions.STOP))
                obs_holder["obs"] = observations
                pose = pose_from_env(env, len(poses), len(executed_actions))
                poses.append(pose)
                buffer_obs(pose)
                after_progress(force=True)
                stopped = True
                return
            step_low(int(action), f"{label}:{ACTION_NAMES[int(action)]}", kind="etpnav")
            return
        stop_or_wp = gridmm.observe_and_act(
            env, active_instruction, current_pose["position"], heading, act=True
        )
        action, angle, distance = stop_or_wp
        if action == Actions.STOP or (angle is None and distance is None):
            requested_actions.append("ETPNAV:STOP")
            executed_actions.append("STOP")
            observations = env.step(int(Actions.STOP))
            obs_holder["obs"] = observations
            pose = pose_from_env(env, len(poses), len(executed_actions))
            poses.append(pose)
            buffer_obs(pose)
            after_progress(force=True)
            stopped = True
            return
        requested_actions.append(f"ETPNAV:HIGHTOLOW:{angle:.3f}:{distance:.3f}")
        executed_actions.append("HIGHTOLOW")
        execute_waypoint(float(angle), float(distance))

    def do_sg() -> None:
        nonlocal phase, target_info, current_anchor_index, fallback_reason, active_instruction
        step = len(executed_actions)
        plan_pos, plan_yaw = sync_navigator()
        reached = navigator._sg_nav_reached_reason(
            step=step,
            current_world_position=plan_pos,
            current_world_heading=plan_yaw,
            target_info=target_info,
        )
        if reached is not None:
            uid = str(target_info.get("uid") or target_info.get("node_id") or "")
            anchors_reached.append(uid)
            log_sg(
                {
                    "decision": "sg_anchor_reached",
                    "reach_reason": reached,
                    "anchor_index": current_anchor_index,
                    "uid": uid,
                    "step": step,
                    "planner": SG_PLANNER,
                    "navmesh_used": False,
                }
            )
            nxt = dual_state.mark_reached(uid)
            if nxt is None:
                handoff = etpnav_continuation(row, plan, 2)
                phase = nav_phase(gridmm, "suffix")
                active_instruction = handoff["continuation"]
                return
            current_anchor_index = 1
            target_info = plan_anchor(nxt, plan.anchors[1].segment["text"])
            if target_info is None:
                failure = jsonable(getattr(navigator, "_sg_last_theta_failure", None)) or {}
                fallback_reason = str(failure.get("reason") or "anchor2_world_theta_failed")
                handoff = etpnav_continuation(row, plan, 1)
                phase = nav_phase(gridmm, "fallback")
                active_instruction = handoff["continuation"]
                log_sg(
                    {
                        "decision": "sg_world_theta_anchor2_failed",
                        "failure": failure,
                        "occupancy": occupancy_meta,
                        "planner": SG_PLANNER,
                        "navmesh_used": False,
                        "step": step,
                    }
                )
            return
        action = navigator._get_sg_world_path_action(
            step=step,
            current_world_position=plan_pos,
            current_world_heading=plan_yaw,
            target_info=target_info,
        )
        if action is None:
            reason = target_info.pop("_sg_pending_abort_reason", "sg_controller_no_action")
            recovered = navigator._on_sg_nav_aborted(
                step=step,
                reason=reason,
                current_world_position=plan_pos,
                current_world_heading=plan_yaw,
                target_info=target_info,
            )
            if recovered:
                return
            fallback_reason = str(reason)
            handoff = etpnav_continuation(row, plan, len(anchors_reached))
            phase = nav_phase(gridmm, "fallback")
            active_instruction = handoff["continuation"]
            log_sg(
                {
                    "decision": "sg_world_theta_aborted",
                    "reason": fallback_reason,
                    "anchor_index": current_anchor_index,
                    "uid": target_info.get("uid"),
                    "step": step,
                    "planner": SG_PLANNER,
                    "navmesh_used": False,
                }
            )
            return
        step_low(
            int(action),
            "SG:" + ACTION_NAMES[int(action)],
            kind="sg",
        )

    rgb0, depth0 = loc.copy_rgbd(observations)
    rgbd_buf.append({"rgb": rgb0, "depth": depth0, "pose": copy.deepcopy(initial_pose)})
    frame0 = loc.live_frame(observations, initial_pose, ep2_live, 0)
    occ.add_frame(frame0)
    last_xz = (frame0.x, frame0.z)
    prev_xz = (frame0.x, frame0.z)
    query_win.append(frame0)
    if maybe_lock(frame0):
        lock_mode = "start+geom"
    elif should_extra_spin():
        extra_spin()
    if locked:
        start_sg_if_possible()

    def make_rollout_record() -> dict[str, Any]:
        metrics = env.get_metrics()
        ne = metric_value(metrics, "distance_to_goal")
        oracle_ne = metric_value(metrics, "oracle_navigation_error")
        handoff = etpnav_continuation(row, plan, len(anchors_reached))
        return {
            "schema_version": "etpnav_level3_ep2_explore_loc_sg_rollout_v1",
            "phase": "level3_ep2",
            "episode_id": episode_id,
            "scene_id": str(episode.scene_id),
            "session_id": session_id,
            "role": "ep2",
            "coordinate_frame": "ep0_local_after_lock" if locked else "ep2_local",
            "instruction": row["instruction"],
            "route": {
                "types": route.get("types"),
                "anchors": route.get("anchors"),
                "suffix": route.get("suffix"),
                "source": route.get("source"),
            },
            "localization": {
                "locked": locked,
                "ok_useful": ok_useful,
                "lock_mode": lock_mode,
                "lock_reason": lock_reason,
                "lock_at_step": lock_at,
                "n_lock_try": n_lock_try,
                "n_explore": n_explore,
                "n_spin": n_spin,
                "n_spin_bouts": n_spin_bouts,
                "best_inlier": best_inlier,
                "trans_err_m": trans_err,
                "yaw_err_deg": yaw_err,
                "K": int(lock_at if lock_at is not None else len(executed_actions)),
                "sg_frames_ingested": n_sg_frames,
                "protocol": "explore100_then_lock_then_world_metric_theta_sg_total400_no_navmesh",
            },
            "sg_handoff": {
                "recall_plan_status": recall_plan.get("status"),
                "recall_plan_reason": recall_plan.get("reason"),
                "selected_uids": [
                    str(item.get("uid") or item.get("node_id") or "")
                    for item in (plan.selected_targets if plan is not None else ())
                ],
                "anchors_reached": anchors_reached,
                "anchor_reached_count": len(anchors_reached),
                "both_anchors_reached": len(anchors_reached) == 2,
                "fallback_reason": fallback_reason if locked else (fallback_reason or "not_locked"),
                "etpnav_instruction_mode": handoff["mode"],
                "etpnav_continuation_instruction": handoff["continuation"],
                "completed_recall_todos": handoff["completed_recall_todos"],
                "sg_actions": n_sg,
                "etpnav_actions": n_etpnav,
                "explore_actions": n_explore,
                "final_phase": phase,
                "goal_transformed_to_ep0_local": bool(locked),
                "sg_planner": SG_PLANNER,
                "occupancy": occupancy_meta or metric_map_stats(adapter),
                "navmesh_used": False,
                "sam3_loaded": False,
                "nav_model": getattr(gridmm, "nav_name", "etpnav"),
                "navigation_controller_version": loc.NAVIGATION_CONTROLLER_VERSION + "+l3_ep2_explore_loc",
            },
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
                "read_stage": "M1",
                "write_stage": "M2",
                "m0": str(row["m0"].resolve()),
                "m1": str(row["m1"].resolve()),
                "m2": str(destination.resolve()),
                "semantic_sg_updated": bool(sg_ingested and args.update_memory == "full"),
                "world_navigation_map_updated_from_live_rgbd": bool(
                    sg_ingested and args.update_memory != "off"
                ),
                "trajectory_graph_updated": bool(sg_ingested and args.update_memory != "off"),
                "memory_update": args.update_memory,
                "sam3_loaded": bool(args.update_memory == "full"),
                "dinov3_loaded": False,
                "aligned_to_m0_after_lock": bool(sg_ingested),
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
        while len(executed_actions) < args.max_steps and not env.episode_over and not stopped:
            if not ep1_locked:
                phase = nav_phase(gridmm, "native")
                active_instruction = row["instruction"]
                do_etpnav()
                continue
            if not locked:
                if n_explore < int(args.explore_steps):
                    phase = "explore"
                    do_explore()
                else:
                    phase = nav_phase(gridmm, "locate")
                    active_instruction = row["instruction"]
                    do_etpnav()
                if locked:
                    start_sg_if_possible()
                continue
            if phase == "sg_navigation":
                do_sg()
            else:
                do_etpnav()

        if locked:
            adapter.finish_episode()
            if adapter.failed:
                raise RuntimeError("memory_module marked EP2 increment failed")
        else:
            try:
                adapter.abort_episode()
            except Exception:
                pass

        result = make_rollout_record()
        atomic_json(work_root / "rollout.json", result)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        if locked and sg_ingested:
            atomic_json(run_dir / "rollout.json", result)
            gridmm.save_graphmap(run_dir / "graphmap.pt")
            janus_l3.publish_m2(
                row["m1"],
                run_dir,
                destination,
                {
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "route": route,
                    "sg_handoff": result["sg_handoff"],
                },
            )
            gm = run_dir / "graphmap.pt"
            if gm.is_file():
                shutil.copy2(gm, destination / "graphmap.pt")
        else:
            shutil.copytree(row["m1"], destination)
            atomic_json(destination / "rollout.json", result)
            atomic_json(run_dir / "recall_plan.json", persistable)
            if (run_dir / "recall_plan.json").is_file():
                shutil.copy2(run_dir / "recall_plan.json", destination / "recall_plan.json")
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


def main() -> int:
    os.environ.setdefault("MEMBRIDGE_SKIP_PYDEPS", "1")
    args = parse_args()
    args.etpnav_root = args.etpnav_root.resolve()
    args.exp_config = args.exp_config.resolve()
    args.task_config = args.task_config.resolve()
    args.dataset = args.dataset.resolve()
    args.ckpt = args.ckpt.resolve()
    args.waypoint_ckpt = args.waypoint_ckpt.resolve()
    args.m0_root = args.m0_root.resolve()
    args.m1_root = args.m1_root.resolve()
    args.lavira_parse_root = [path.resolve() for path in args.lavira_parse_root]
    args.output_root = args.output_root.resolve()
    args.query_vocabulary = args.query_vocabulary.resolve()
    args.bert_path = args.bert_path.resolve()
    args.scenes_dir = args.scenes_dir.resolve()
    if args.session_id_file:
        args.session_id_file = args.session_id_file.resolve()
    args.run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    selected = load_tasks(args)
    print(
        json.dumps(
            {
                "protocol": "explore100_then_lock_then_world_metric_theta_sg_total400",
                "selected": len(selected),
                "expected_sessions": args.expected_count,
                "instruction_parse": "live_vlm",
                "vlm_top3": "live_api",
                "update_memory": args.update_memory,
                "m0_root": str(args.m0_root),
                "m1_root": str(args.m1_root),
                "sam3": args.update_memory == "full",
                "shard": [args.shard_index, args.num_shards],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0

    sys.path.insert(0, str(loc.REPO_ROOT))
    install_full_exploration_overlay()
    from membridge.habitat_runtime.level3_adaptation import (
        install_dual_anchor_plan_replay,
    )

    install_dual_anchor_plan_replay(MemoryNavigator, janus_l3.LOG)
    np.random.seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    worker = args.worker_id.strip() or f"shard{args.shard_index:02d}"
    raw_log = args.output_root / "raw" / f"ep2_{worker}.jsonl"
    completed = {str(row["episode_id"]) for row in read_jsonl(raw_log)}

    if args.nav_model == "janus":
        env, habitat_config = make_janus_env(args)
    else:
        env, habitat_config = make_env(args, split="ep2")
    adapt = HERE / "adapt"
    if str(adapt) not in sys.path:
        sys.path.insert(0, str(adapt))
    from janus_policy import build_policy

    gridmm = build_policy(args, ETPNavSuffixPolicy)
    episodes_by_id = episodes_for_nav(env, args.nav_model)
    failures: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(selected, 1):
            episode_id = row["episode_id"]
            destination = args.output_root / "sessions" / row["session_id"] / "stages/M2"
            if args.skip_existing and destination.is_dir() and (destination / "rollout.json").is_file():
                recovered = read_json(destination / "rollout.json")
                append_jsonl(raw_log, recovered)
                completed.add(episode_id)
                continue
            if episode_id in completed:
                continue
            episode_key = row["session_id"] if args.nav_model == "janus" else episode_id
            episode = episodes_by_id.get(episode_key)
            if episode is None:
                raise RuntimeError(f"EP2 {episode_key} is not in the Habitat dataset")
            print(
                f"[ETPNAV-L3-EP2-EXPLOC] {index}/{len(selected)} {row['session_id']} ep={episode_id}",
                flush=True,
            )
            if args.nav_model == "janus":
                import torch

                torch.cuda.empty_cache()
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
                loc_row = result["localization"]
                print(
                    json.dumps(
                        {
                            "session": row["session_id"],
                            "locked": loc_row["locked"],
                            "ok_useful": loc_row["ok_useful"],
                            "lock_mode": loc_row["lock_mode"],
                            "n_explore": loc_row["n_explore"],
                            "phase": result["sg_handoff"]["final_phase"],
                            "anchors": result["sg_handoff"]["anchor_reached_count"],
                            "metrics": result["metrics"],
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
                print(f"[ETPNAV-L3-EP2-EXPLOC] failed: {failure['error']}", file=sys.stderr, flush=True)
                if not args.continue_on_error:
                    raise
    finally:
        env.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
