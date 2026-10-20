"""Semantic scene-graph construction.

The caller supplies the API endpoint and key. This module does not store either value.
"""

import os
import sys
import datetime
import json
import yaml
import csv
import time
import pickle
import shutil
from typing import List, Dict, Any, Tuple
import cv2 
import gc
import copy
import argparse
import networkx as nx
import threading
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from sentence_transformers import util
from skimage.draw import line as sk_line
from scipy.ndimage import binary_dilation
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.patches import Wedge
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sentence_transformers import SentenceTransformer
try:
    from ram.models import ram_plus
    from ram import inference_ram as inference
    from ram import get_transform
except ImportError:
    ram_plus = None
    inference = None
    get_transform = None
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
SAM3_SOURCE_ROOT = os.environ.get("INHERIT_SG_SAM3_SOURCE_ROOT", "/data2/shizk/sam3")
DIST_THRESH = 1.2
if SAM3_SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SAM3_SOURCE_ROOT)

try:
    from sam3 import build_sam3_image_model
    from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
    from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
    from sam3.train.data.collator import collate_fn_api as collate
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.eval.postprocessors import PostProcessImage
except ImportError as e:
    print(f"\n[CRITICAL ERROR] 无法导入 SAM3 模块。请确认路径 {SAM3_SOURCE_ROOT} 正确。")
    print(f"详细错误: {e}\n")
    raise e


_GLOBAL_COUNTER = 1

def create_empty_datapoint():
    """Create an empty data point."""
    return Datapoint(find_queries=[], images=[])

def set_image(datapoint, pil_image):
    """Set the image."""
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]

def add_text_prompt(datapoint, text_query):
    """Add a text prompt."""
    global _GLOBAL_COUNTER
    
    if len(datapoint.images) != 1:
        raise ValueError("Please set the image first using set_image()")

    w, h = datapoint.images[0].size
    
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[], 
            is_exhaustive=True, 
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=_GLOBAL_COUNTER,
                original_image_id=_GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            )
        )
    )
    _GLOBAL_COUNTER += 1
    return _GLOBAL_COUNTER - 1

try:
    from .common import Logger, LLMHandler, read_pose_matrix, pixel_to_world, process_nodes_file, analyze_area_quality
except ImportError:
    from common import Logger, LLMHandler, read_pose_matrix, pixel_to_world, process_nodes_file, analyze_area_quality
try:
    from .label_score import (
        ensure_label_score_state,
        initialize_label_score,
        record_label_score,
    )
except ImportError:
    from label_score import (
        ensure_label_score_state,
        initialize_label_score,
        record_label_score,
    )
try:
    from .object_provenance import (
        bbox_pixel_area,
        ensure_object_provenance,
        merge_object_provenance_into_nodes,
        normalize_episode_id,
        observation_selection_score,
        record_object_observation,
        sync_object_provenance_to_graph,
    )
except ImportError:
    from object_provenance import (
        bbox_pixel_area,
        ensure_object_provenance,
        merge_object_provenance_into_nodes,
        normalize_episode_id,
        observation_selection_score,
        record_object_observation,
        sync_object_provenance_to_graph,
    )
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any
from transformers import AutoImageProcessor, AutoModel

def load_feature_extractor(config: Dict[str, Any], device: str) -> Any:
    cfg = config.get('feature_extractor', {})
    model_path = cfg.get('weights_path') 
    
    print(f"\n--- Loading Transformer Model from {model_path} ---")
    
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    
    model.to(device)
    model.eval()

    model.hf_processor = processor
    
    return model


def extract_feature(image_pil: Image, model: Any, device: str) -> np.ndarray:
    if not hasattr(model, 'hf_processor'):
        raise AttributeError("Model object missing 'hf_processor'.")
    
    processor = model.hf_processor
    
    inputs = processor(
        images=image_pil, 
        return_tensors="pt",
        
        do_resize=True,
        size={"shortest_edge": 518}, 
        
        do_center_crop=True,
        crop_size={"height": 518, "width": 518},
        
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )
    
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
        features = outputs.pooler_output
    else:
        features = outputs.last_hidden_state[:, 0, :]
        
    return features.cpu().numpy().flatten()


class MapManager:
    def __init__(self, config: Dict, intrinsics: Dict, run_dir: str):
        self.config = config
        self.intrinsics = intrinsics
        self.resolution = config.get('map_resolution', 0.05)
        self.run_dir = run_dir

        self.map_w = config.get('map_w', 2000)
        self.map_h = config.get('map_h', 2000)
        self.min_x = config.get('min_x', -50.0)
        self.min_z = config.get('min_z', -50.0)

        # ===== [NEW] log-odds occupancy mapping =====
        self.logodds = np.zeros((self.map_h, self.map_w), dtype=np.float32)
        self.L_OCC = config.get('L_OCC', 0.85)     # occupied update
        self.L_FREE = config.get('L_FREE', -0.25)  # free update
        self.L_MIN = config.get('L_MIN', -4.0)
        self.L_MAX = config.get('L_MAX', 4.0)

        # door as a separate semantic layer (binary)
        self.door_map = np.zeros((self.map_h, self.map_w), dtype=np.uint8)

        # cached binary maps (updated each keyframe for viz/seg)
        self.occ_bin = np.zeros((self.map_h, self.map_w), dtype=np.uint8)
        self.free_bin = np.zeros((self.map_h, self.map_w), dtype=np.uint8)

        # room tracking (pure geometry)
        self.room_id_map = -np.ones((self.map_h, self.map_w), dtype=np.int32)  # optional global label
        self.current_room_id = 0
        self.room_registry = {}  # room_id -> {"centroid": (x,z), "area": float}

        # [KEEP] optional room geometry overlay if you still want
        self.room_geometry = {}

        # mapping params
        self.max_range = config.get('max_range_m', 8.0)
        self.min_range = config.get('min_range_m', 0.3)
        self.raycast_stride = config.get('raycast_stride', 4)  # pixel subsample in depth image
        self.hit_stride = config.get('hit_stride', 2)          # subsample for hit points

        # occ inflation for planning/segmentation (robot radius)
        self.robot_radius_m = config.get('robot_radius_m', 0.30)
        self.inflation_m = config.get('inflation_m', 0.10)
        # [ADD in MapManager.__init__]
        self.raycast_update_occupied = bool(config.get("raycast_update_occupied", False))

        # Door evidence accumulator to avoid speckles
        self.door_hits = np.zeros((self.map_h, self.map_w), dtype=np.uint16)
        self.door_hit_inc = int(config.get("door_hit_inc", 1))
        self.door_hit_thresh = int(config.get("door_hit_thresh", 6))
        self.floor_maps = {} # floor_id -> dict of map arrays
        self.current_floor_idx = 0

    # ---------- [NEW] coordinate transforms ----------
    def world_to_grid(self, x: float, z: float) -> Tuple[int, int]:
        u = int((x - self.min_x) / self.resolution)
        v = int((z - self.min_z) / self.resolution)
        return u, v

    def grid_to_world(self, u: int, v: int) -> Tuple[float, float]:
        x = u * self.resolution + self.min_x
        z = v * self.resolution + self.min_z
        return x, z

    def _valid_uv(self, u: int, v: int) -> bool:
        return (0 <= u < self.map_w) and (0 <= v < self.map_h)

    # ---------- [NEW] log-odds update helpers ----------
    def _update_logodds(self, u: np.ndarray, v: np.ndarray, delta: float):
        valid = (u >= 0) & (u < self.map_w) & (v >= 0) & (v < self.map_h)
        u, v = u[valid], v[valid]
        self.logodds[v, u] = np.clip(self.logodds[v, u] + delta, self.L_MIN, self.L_MAX)

    def refresh_binary_maps(self):
        # P(occ)=sigmoid(L); threshold at 0.65-ish
        occ = self.logodds > 0.85
        free = self.logodds < -0.40
        self.occ_bin = occ.astype(np.uint8)
        self.free_bin = free.astype(np.uint8)

    # ---------- [NEW] depth raycasting mapping ----------
    def integrate_depth_raycast(self, depth_img: np.ndarray, pose: np.ndarray):
        """
        depth_img: uint16/float depth in raw units, convert using intrinsics['depth_scale']
        pose: 4x4 camera-to-world (Habitat world) like your pixel_to_world convention
        """
        if depth_img is None or pose is None:
            return

        fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
        cx, cy = self.intrinsics['cx'], self.intrinsics['cy']
        ds = self.intrinsics.get('depth_scale', 1000.0)

        h, w = depth_img.shape[:2]

        # camera origin in world
        cam_world = pose[:3, 3]
        u0, v0 = self.world_to_grid(cam_world[0], cam_world[2])

        # T_fix: OpenCV -> Habitat (same as your pixel_to_world)
        # p_hab = [x, -y, -z]
        # Here we do vectorized unprojection in OpenCV then apply sign flips.
        step = int(self.raycast_stride)
        us = np.arange(0, w, step)
        vs = np.arange(0, h, step)
        uu, vv = np.meshgrid(us, vs)
        uu = uu.reshape(-1)
        vv = vv.reshape(-1)

        d_raw = depth_img[vv, uu].astype(np.float32)
        z = d_raw / ds
        valid = (z > self.min_range) & (z < self.max_range)
        if np.count_nonzero(valid) == 0:
            return

        uu, vv, z = uu[valid], vv[valid], z[valid]

        x_cam = (uu - cx) * z / fx
        y_cam = (vv - cy) * z / fy
        # OpenCV -> Habitat
        p_hab = np.stack([x_cam, -y_cam, -z, np.ones_like(z)], axis=1)  # (N,4)
        p_world = (pose @ p_hab.T).T[:, :3]  # (N,3)

        # Raycast: mark free along ray (origin -> hit), mark occupied at hit
        # subsample hits to reduce work
        hit_stride = max(1, int(self.hit_stride))
        p_world = p_world[::hit_stride]

        for pw in p_world:
            uh, vh = self.world_to_grid(pw[0], pw[2])
            if not (self._valid_uv(u0, v0) and self._valid_uv(uh, vh)):
                continue
            rr, cc = sk_line(v0, u0, vh, uh)  # note: (row,col) = (v,u)

            if len(rr) < 2:
                continue

            # free cells: all but last
            rr_free, cc_free = rr[:-1], cc[:-1]
            self._update_logodds(cc_free, rr_free, self.L_FREE)

            # [REPLACE] occupied update is optional
            if self.raycast_update_occupied:
                self._update_logodds(np.array([uh]), np.array([vh]), self.L_OCC)

    # ---------- [NEW] integrate door / wall masks (optional but recommended) ----------
    def integrate_structure_points(self, points_3d: np.ndarray, label_type: str):
        """
        points_3d: (N,3) world coords
        label_type: 'wall' or 'door'
        """
        if points_3d is None or len(points_3d) == 0:
            return

        u = ((points_3d[:, 0] - self.min_x) / self.resolution).astype(int)
        v = ((points_3d[:, 2] - self.min_z) / self.resolution).astype(int)

        valid = (u >= 0) & (u < self.map_w) & (v >= 0) & (v < self.map_h)
        u, v = u[valid], v[valid]

        lt = (label_type or "").lower()

        if lt in ["door", "complete door frame", "doorway"]:
            # Ensure float hits (decay needs float). If your door_hits is uint16 from old code,
            # convert once here (cheap safeguard).
            if self.door_hits.dtype != np.float32 and self.door_hits.dtype != np.float64:
                self.door_hits = self.door_hits.astype(np.float32)

            # (1) decay old evidence each integration call
            decay = float(getattr(self, "door_hits_decay", 0.98))
            self.door_hits *= decay

            # (2) add new evidence
            inc = float(getattr(self, "door_hit_inc", 1.0))
            self.door_hits[v, u] += inc

            # (3) saturate / cap
            cap = float(getattr(self, "door_hits_cap", 20.0))
            np.clip(self.door_hits, 0.0, cap, out=self.door_hits)

            # (4) compute door_map dynamically from hits (NOT permanent)
            thr = float(getattr(self, "door_hit_thresh", 6.0))
            door_map = (self.door_hits >= thr)

            # (5) optional: dilation ONLY for visualization/cut convenience;
            # do NOT feed back into door_hits, and it will naturally shrink when hits decay.
            dil_iter = int(getattr(self, "door_map_dilate_iter", 1))
            if dil_iter > 0:
                door_map = binary_dilation(door_map, iterations=dil_iter)

            self.door_map = door_map.astype(np.uint8)
            return

        # ---- WALL / OTHER: treat as occupancy evidence ----
        if lt in ["wall"]:
            self._update_logodds(u, v, self.L_OCC)
            return

        # default: also occupancy
        self._update_logodds(u, v, self.L_OCC)

    # ---------- [NEW] inflated obstacle mask for segmentation ----------
    def get_inflated_occ(self) -> np.ndarray:
        r = int((self.robot_radius_m + self.inflation_m) / self.resolution)
        if r <= 0:
            return self.occ_bin.astype(bool)
        return binary_dilation(self.occ_bin.astype(bool), iterations=r)

    # ---------- [NEW] visualization per keyframe ----------
    def render_keyframe_viz(self, trajectory: List[np.ndarray], current_pose: np.ndarray,
                            local_window_m: float = 20.0,
                            cut_mask_local: np.ndarray = None,
                            room_mask_local: np.ndarray = None) -> np.ndarray:
        """
        Returns a BGR image. Shows occ/free/unknown + doors + cut + current room region + trajectory + arrow.
        Crops to a local window around robot for clarity.
        """
        self.refresh_binary_maps()

        vis = np.zeros((self.map_h, self.map_w, 3), dtype=np.uint8)

        # unknown = dark
        # free = light
        # occ = gray
        vis[self.free_bin == 1] = (240, 240, 240)
        vis[self.occ_bin == 1] = (120, 120, 120)

        # doors in cyan
        vis[self.door_map == 1] = (255, 255, 0)

        # overlay cut/room if provided in local coords
        if current_pose is not None and (cut_mask_local is not None or room_mask_local is not None):
            cx, cz = current_pose[0, 3], current_pose[2, 3]
            cu, cv = self.world_to_grid(cx, cz)
            half = int((local_window_m / self.resolution) / 2)
            x1, x2 = max(0, cu - half), min(self.map_w, cu + half)
            y1, y2 = max(0, cv - half), min(self.map_h, cv + half)

            overlay = vis.copy()
            if cut_mask_local is not None:
                # cut in magenta
                cm = cut_mask_local.astype(bool)
                overlay[y1:y2, x1:x2][cm] = (255, 0, 255)
            if room_mask_local is not None:
                # current room in green (semi)
                rm = room_mask_local.astype(bool)
                overlay[y1:y2, x1:x2][rm] = (0, 255, 0)
            cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

        # trajectory
        pts = []
        for p in trajectory:
            tx = p[0]
            tz = p[2] if len(p) > 2 else p[1]
            u, v = self.world_to_grid(tx, tz)
            if 0 <= u < self.map_w and 0 <= v < self.map_h:
                pts.append([u, v])
        if len(pts) > 1:
            cv2.polylines(vis, [np.array(pts, np.int32)], False, (0, 0, 255), 2)
        for u, v in pts:
            cv2.circle(vis, (u, v), 2, (0, 0, 255), -1)

        # robot arrow
        if current_pose is not None:
            cx, cz = current_pose[0, 3], current_pose[2, 3]
            cu, cv = self.world_to_grid(cx, cz)
            vec_fwd = current_pose[:3, 2] * (-1)
            ex = cx + vec_fwd[0] * 1.5
            ez = cz + vec_fwd[2] * 1.5
            eu, ev = self.world_to_grid(ex, ez)
            cv2.arrowedLine(vis, (cu, cv), (eu, ev), (0, 200, 0), 3, tipLength=0.3)

        # crop local
        if current_pose is not None:
            cx, cz = current_pose[0, 3], current_pose[2, 3]
            cu, cv = self.world_to_grid(cx, cz)
            half = int((local_window_m / self.resolution) / 2)
            x1, x2 = max(0, cu - half), min(self.map_w, cu + half)
            y1, y2 = max(0, cv - half), min(self.map_h, cv + half)
            vis = vis[y1:y2, x1:x2].copy()

        return vis
    def is_point_in_room(self, point_3d: list | np.ndarray, room_id: int | str) -> bool:
        """Test whether a world point lies inside a room."""
        if point_3d is None: 
            return False
            
        x = point_3d[0]
        z = point_3d[2]

        u, v = self.world_to_grid(x, z)
        
        if not self._valid_uv(u, v):
            return False

        current_val = self.room_id_map[v, u]

        try:
            return int(current_val) == int(room_id)
        except (ValueError, TypeError):
            return False
    def calculate_visible_room_area(self, pose: np.ndarray, room_id: int, fov_deg: float = 90.0, max_range_m: float = 5.0) -> int:
        """Measure the overlap between the view sector and the target room. Rays stop at walls."""
        if self.room_id_map is None: return 0
        if pose is None: return 0

        rx, rz = pose[0, 3], pose[2, 3]
        u0, v0 = self.world_to_grid(rx, rz)
        
        if not self._valid_uv(u0, v0): return 0

        # Habitat/OpenGL: Camera -Z is forward
        # World coords: X is Right, Z is Back. 
        # Grid coords: u aligns with X, v aligns with Z.
        vec_fwd = -pose[:3, 2] 
        # atan2(y, x) -> atan2(delta_z, delta_x)
        heading = np.arctan2(vec_fwd[2], vec_fwd[0])

        num_rays = 45
        max_dist_px = int(max_range_m / self.resolution)
        
        angles = np.linspace(heading - np.radians(fov_deg/2), heading + np.radians(fov_deg/2), num_rays)
        
        occ_map = self.occ_bin
        room_map = self.room_id_map
        target_rid = int(room_id)
        
        visible_room_pixels = set()
        
        h, w = self.map_h, self.map_w

        for ang in angles:
            ue = int(u0 + max_dist_px * np.cos(ang))
            ve = int(v0 + max_dist_px * np.sin(ang))
            
            rr, cc = sk_line(v0, u0, ve, ue) # sk_line returns (row, col) -> (v, u)
            
            for r, c in zip(rr, cc):
                if r < 0 or r >= h or c < 0 or c >= w:
                    break
                
                if occ_map[r, c] > 0:
                    break 
                
                if room_map[r, c] == target_rid:
                    visible_room_pixels.add((r, c))

        score = len(visible_room_pixels)

        current_pos_rid = room_map[v0, u0]
        if int(current_pos_rid) == target_rid:
            score *= 3.0
            
        return int(score)
    def regularize_mask_rectilinear(self, mask: np.ndarray) -> np.ndarray:
        if np.sum(mask) == 0: return mask

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        
        h, w = mask.shape
        scale_down = 0.1
        
        small = cv2.resize(mask, (int(w * scale_down), int(h * scale_down)), interpolation=cv2.INTER_NEAREST)
        
        small_kernel = np.ones((3,3), np.uint8)
        small = cv2.morphologyEx(small, cv2.MORPH_OPEN, small_kernel)
        small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, small_kernel)
        
        final_mask = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return final_mask
    def resolve_global_room_overlaps(self, room_masks: Dict[int, np.ndarray]) -> np.ndarray:
        """Assign each pixel to one room by distance-field competition."""
        if not room_masks:
            return np.zeros((self.map_h, self.map_w), dtype=np.int32)

        valid_ids = list(room_masks.keys())
        h, w = self.map_h, self.map_w
        
        distance_stack = np.zeros((len(valid_ids), h, w), dtype=np.float32)
        
        for idx, rid in enumerate(valid_ids):
            mask = room_masks[rid]
            if np.sum(mask) == 0: continue
            
            dist = distance_transform_edt(mask)
            distance_stack[idx] = dist

        max_dist_vals = np.max(distance_stack, axis=0)
        max_indices = np.argmax(distance_stack, axis=0)
        
        new_map = np.zeros((h, w), dtype=np.int32)
        
        mask_valid = max_dist_vals > 0
        
        id_array = np.array(valid_ids, dtype=np.int32)
        new_map[mask_valid] = id_array[max_indices[mask_valid]]
        
        
        self.room_id_map = new_map
        return new_map
    
    def switch_floor(self, new_floor_idx: int):
        if self.current_floor_idx == new_floor_idx:
            return

        print(f"[MapManager] Switching Context: Floor {self.current_floor_idx} -> {new_floor_idx}")

        self.floor_maps[self.current_floor_idx] = {
            "logodds": self.logodds.copy(),
            "door_map": self.door_map.copy(),
            "occ_bin": self.occ_bin.copy(),
            "free_bin": self.free_bin.copy(),
            "room_id_map": self.room_id_map.copy(),
            "door_hits": self.door_hits.copy()
        }

        if new_floor_idx in self.floor_maps:
            print(f"  -> Restoring existing map for Floor {new_floor_idx}")
            data = self.floor_maps[new_floor_idx]
            self.logodds = data["logodds"]
            self.door_map = data["door_map"]
            self.occ_bin = data["occ_bin"]
            self.free_bin = data["free_bin"]
            self.room_id_map = data["room_id_map"]
            self.door_hits = data["door_hits"]
        else:
            print(f"  -> Initializing NEW map for Floor {new_floor_idx}")
            self.logodds = np.zeros((self.map_h, self.map_w), dtype=np.float32)
            self.door_map = np.zeros((self.map_h, self.map_w), dtype=np.uint8)
            self.occ_bin = np.zeros((self.map_h, self.map_w), dtype=np.uint8)
            self.free_bin = np.zeros((self.map_h, self.map_w), dtype=np.uint8)
            self.room_id_map = -np.ones((self.map_h, self.map_w), dtype=np.int32)
            self.door_hits = np.zeros((self.map_h, self.map_w), dtype=np.uint16) # or float32 if updated

        self.current_floor_idx = new_floor_idx

class GeometryRoomTracker:
    def __init__(self, map_manager, config: dict):
        self.mm = map_manager
        self.config = config
        
        self.min_room_radius_px = int(1.5 / self.mm.resolution) 
        
        self.erosion_kernel = 3
        
        self.current_room_id = 0
        self.next_global_room_id = 1
        
        # Global Maps
        self.global_id_grid = np.zeros((self.mm.map_h, self.mm.map_w), dtype=np.int32)
        self.global_conf_grid = np.zeros((self.mm.map_h, self.mm.map_w), dtype=np.uint8)
        
        self.CONF_INC = 10
        self.CONF_DEC = 2
        self.CONF_MAX = 100 
        self.CONF_THRESH = 1

        self.colors = np.random.randint(50, 255, (100, 3), dtype=np.uint8)
        self.colors[0] = [0, 0, 0] 

        self.floor_states = {} 
        self.current_floor_idx = 0

    def switch_floor(self, new_floor_idx: int):
        self.floor_states[self.current_floor_idx] = {
            "id_grid": self.global_id_grid.copy(),
            "conf_grid": self.global_conf_grid.copy(),
            "next_id": self.next_global_room_id
        }
        if new_floor_idx in self.floor_states:
            st = self.floor_states[new_floor_idx]
            self.global_id_grid = st["id_grid"]
            self.global_conf_grid = st["conf_grid"]
            self.next_global_room_id = st["next_id"]
        else:
            self.global_id_grid = np.zeros((self.mm.map_h, self.mm.map_w), dtype=np.int32)
            self.global_conf_grid = np.zeros((self.mm.map_h, self.mm.map_w), dtype=np.uint8)
        self.current_floor_idx = new_floor_idx
        self.current_room_id = 0

    def step(self, pose: np.ndarray, floor_trajectory: List[np.ndarray] = None) -> dict:
        if pose is None: return {"room_id": 0, "viz": None}

        door_obstacle = (self.mm.door_map > 0).astype(np.uint8)
        grid_free = self.mm.free_bin.copy()
        
        if np.sum(grid_free) == 0:
            return {"room_id": 0, "viz": self._render_empty_viz(pose)}

        effective_free = cv2.bitwise_and(grid_free, grid_free, mask=cv2.bitwise_not(door_obstacle))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded_free = cv2.erode(effective_free, kernel, iterations=1)
        
        local_proposal = np.zeros_like(grid_free, dtype=np.int32)
        
        if np.sum(eroded_free) > 0:
            dt = distance_transform_edt(eroded_free)
            
            dt = cv2.GaussianBlur(dt, (7, 7), 2.0)
            
            local_maxi = peak_local_max(dt, min_distance=self.min_room_radius_px, labels=eroded_free, exclude_border=False)
            
            if len(local_maxi) > 0:
                markers = np.zeros_like(grid_free, dtype=np.int32)
                for idx, (r, c) in enumerate(local_maxi):
                    markers[r, c] = idx + 1
                
                ws_labels = watershed(-dt, markers, mask=effective_free)
                local_proposal = ws_labels.astype(np.int32)
        
        self._update_global_map(local_proposal)
        
        current_global_mask = np.zeros_like(self.global_id_grid)
        valid_mask = self.global_conf_grid >= self.CONF_THRESH
        current_global_mask[valid_mask] = self.global_id_grid[valid_mask]

        rx, rz = pose[0, 3], pose[2, 3]
        ru, rv = self.mm.world_to_grid(rx, rz)
        if 0 <= ru < self.mm.map_w and 0 <= rv < self.mm.map_h:
            rid = current_global_mask[rv, ru]
            if rid > 0: self.current_room_id = int(rid)

        viz_img = self._render_viz(current_global_mask, pose, trajectory=floor_trajectory)

        return {
            "room_id": self.current_room_id,
            "room_mask_local": current_global_mask,
            "viz": viz_img,
            "changed": False
        }

    def _update_global_map(self, local_proposal):
        unique_local = np.unique(local_proposal)
        unique_local = unique_local[unique_local > 0]
        
        local_to_global = {}
        used_global_ids = set()
        
        for lid in unique_local:
            local_mask = (local_proposal == lid)
            global_region = self.global_id_grid[local_mask]
            conf_region = self.global_conf_grid[local_mask]
            
            valid_globals = global_region[conf_region > 0]
            
            target_gid = -1
            if len(valid_globals) > 0:
                ids, counts = np.unique(valid_globals, return_counts=True)
                if ids[0] == 0 and len(ids) > 1:
                    ids = ids[1:]
                    counts = counts[1:]
                elif ids[0] == 0:
                    pass
                
                if len(ids) > 0 and ids[0] != 0:
                    best_idx = np.argmax(counts)
                    target_gid = ids[best_idx]
            
            if target_gid == -1 or target_gid in used_global_ids:
                target_gid = self.next_global_room_id
                self.next_global_room_id += 1
            
            local_to_global[lid] = target_gid
            used_global_ids.add(target_gid)
            
        for lid, gid in local_to_global.items():
            local_mask = (local_proposal == lid)
            
            new_area_mask = local_mask & (self.global_id_grid == 0)
            self.global_id_grid[new_area_mask] = gid
            self.global_conf_grid[new_area_mask] = self.CONF_THRESH + self.CONF_INC
            
            match_mask = local_mask & (self.global_id_grid == gid)
            self.global_conf_grid[match_mask] = np.clip(self.global_conf_grid[match_mask] + self.CONF_INC, 0, self.CONF_MAX)
            
            mismatch_mask = local_mask & (self.global_id_grid != gid) & (self.global_id_grid != 0)
            self.global_conf_grid[mismatch_mask] = np.clip(self.global_conf_grid[mismatch_mask] - self.CONF_DEC, 0, self.CONF_MAX)
            
            flip_mask = mismatch_mask & (self.global_conf_grid < self.CONF_INC)
            self.global_id_grid[flip_mask] = gid
            self.global_conf_grid[flip_mask] = self.CONF_THRESH + self.CONF_INC

    def _render_viz(self, room_mask, pose, trajectory: List[np.ndarray] = None):
        h, w = room_mask.shape
        viz = np.ones((h, w, 3), dtype=np.uint8) * 240 # Light Gray background
        
        valid_mask = room_mask > 0
        if np.sum(valid_mask) > 0:
            max_id = np.max(room_mask)
            if max_id >= len(self.colors):
                more_colors = np.random.randint(50, 255, (max_id - len(self.colors) + 50, 3), dtype=np.uint8)
                self.colors = np.vstack((self.colors, more_colors))
            viz[valid_mask] = self.colors[room_mask[valid_mask]]
        
        occ_mask = self.mm.occ_bin > 0
        viz[occ_mask] = [0, 0, 0]
        
        free_mask = (self.mm.free_bin > 0) & (room_mask == 0)
        viz[free_mask] = [255, 255, 255]

        cx, cz = pose[0, 3], pose[2, 3]
        u, v = self.mm.world_to_grid(cx, cz)
        cv2.circle(viz, (u, v), 5, (0, 0, 255), -1)
        
        vec_fwd = pose[:3, 2] * (-1)
        end_u = u + int(vec_fwd[0] * 20)
        end_v = v + int(vec_fwd[2] * 20)
        cv2.arrowedLine(viz, (u, v), (end_u, end_v), (0, 0, 255), 2, tipLength=0.3)
        
        text = f"F{self.current_floor_idx} | Room: {self.current_room_id}"
        cv2.putText(viz, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return viz

    def _render_empty_viz(self, pose):
        viz = np.ones((self.mm.map_h, self.mm.map_w, 3), dtype=np.uint8) * 240
        cx, cz = pose[0, 3], pose[2, 3]
        u, v = self.mm.world_to_grid(cx, cz)
        cv2.circle(viz, (u, v), 5, (0, 0, 255), -1)
        cv2.putText(viz, "Building Map...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return viz
     
class Sam3LocalClient:
    def __init__(self, config: dict, device='cuda'):
        self.device = device
        self.config = config.get('segmentation_module', {})
        self.save_debug_artifacts = bool(
            self.config.get('save_debug_artifacts', False)
        )
        size_filter = self.config.get('size_filter')
        if not isinstance(size_filter, dict):
            raise ValueError(
                "SAM3 requires the reviewed runtime vocabulary size_filter"
            )
        self.default_min_area_px = int(size_filter['default_min_area_px'])
        self.default_min_side_px = int(size_filter['default_min_side_px'])
        self.small_min_area_px = int(size_filter['small_min_area_px'])
        self.small_min_side_px = int(size_filter['small_min_side_px'])
        self.small_object_labels = {
            " ".join(str(value).strip().lower().replace("_", " ").split())
            for value in size_filter.get('small_object_labels', [])
        }
        if (
            min(
                self.default_min_area_px,
                self.default_min_side_px,
                self.small_min_area_px,
                self.small_min_side_px,
            ) <= 0
            or self.small_min_area_px > self.default_min_area_px
            or self.small_min_side_px > self.default_min_side_px
        ):
            raise ValueError(f"Invalid SAM3 size_filter: {size_filter!r}")
        
        self.sam3_root = SAM3_SOURCE_ROOT
        self.bpe_path = self.config.get(
            'bpe_path',
            os.path.join(self.sam3_root, "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        )
        self.checkpoint_path = self.config.get(
            'checkpoint_path',
            os.environ.get(
                "INHERIT_SG_SAM3_CKPT",
                "/data2/fyd/.cache/modelscope/hub/models/facebook/sam3/sam3.pt",
            ),
        )
        
        print(f"\n--- Loading SAM 3 Model (Batch Mode) ---")
        print(f"device: {self.device}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        try:
            self.model = build_sam3_image_model(
                bpe_path=self.bpe_path,
                checkpoint_path=self.checkpoint_path
            )
            self.model.to(self.device)
            self._move_model_tensor_caches(self.device)
            self.model.eval()
            
            self.user_threshold = self.config.get('box_threshold', 0.20)
            
            self.transform = ComposeAPI([
                RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            
            print(f"SAM 3 Loaded. Default Threshold: {self.user_threshold}")
        except Exception as e:
            print(f"Error loading SAM3: {e}")
            raise e
        # Keep category batching explicit and bounded.  It changes inference
        # throughput/memory use, not the canonical prompt vocabulary.
        self.category_batch_size = int(self.config.get('category_batch_size', 10))
        if not 1 <= self.category_batch_size <= 256:
            raise ValueError(
                "SAM3 category_batch_size must be between 1 and 256; "
                f"received {self.category_batch_size}"
            )
        print(f"[SAM3] Category batch size: {self.category_batch_size}")


        if not hasattr(self, 'text_encoder'):
            print("  [Init] Loading SentenceTransformer for semantic check...")
            try:
                self.text_encoder = SentenceTransformer(os.environ.get("INHERIT_SG_TEXT_ENCODER", ""))
            except Exception as e:
                print(f"  [Warning] Failed to load local model: {e}. Semantic check will be disabled.")
                self.text_encoder = None

        # Preserve the explicitly configured SAM3 device. Resetting this to
        # bare "cuda" moves inference inputs back to cuda:0 while the model
        # may live on another visible device.
        self.device = torch.device(self.device)
        
        ram_pretrained = '/home/szk/recognize-anything/pretrained/ram_plus_swin_large_14m.pth' 
        self.ram_image_size = 384
        
        if config["use_predefined_object_classes"]:
            print("[Info] Using predefined object classes for")
            return 
        
        
        try:
            self.ram_transform = get_transform(image_size=self.ram_image_size)
            self.ram_model = ram_plus(pretrained=ram_pretrained, image_size=self.ram_image_size, vit='swin_l')
            self.ram_model.eval()
            self.ram_model.to(self.device)
            print("[Init] RAM++ model loaded successfully.")
        except Exception as e:
            print(f"[Warning] RAM++ model failed to load: {e}")
            self.ram_model = None
        
        try:
            self.llm_handler = LLMHandler(config['llm_config'])
        except Exception as e:
            print(f"[Warning] LLM Handler failed to initialize: {e}")
            self.llm_handler = None

    def _move_model_tensor_caches(self, device):
        """Move SAM3's unregistered precomputed CUDA tensors with the model."""
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            if isinstance(value, list):
                return [move(item) for item in value]
            return value

        for module in self.model.modules():
            for name in ("cache", "coord_cache", "compilable_cord_cache"):
                if hasattr(module, name):
                    setattr(module, name, move(getattr(module, name)))
        
    
    def predict_masks(self, image_pil: Image.Image, prompts: list, threshold: float = None) -> Dict[str, List[np.ndarray]]:
        result = {p: [] for p in prompts}
        eff_threshold = threshold if threshold is not None else 0.10
        
        batch_results = self._batch_inference(image_pil, prompts, eff_threshold)
        
        for prompt in prompts:
            res = batch_results.get(prompt)
            if res is None: continue
            
            masks_np = res['masks']
            for i in range(masks_np.shape[0]):
                mask_binary = masks_np[i] > 0
                result[prompt].append(mask_binary)
                
        return result
    
    def _batch_inference(self, image_pil: Image.Image, prompts: list, threshold: float):
        """Run one batched inference call."""
        if self.device.type == "cuda":
            # SAM3 creates some prompt/geometry tensors using the current CUDA
            # device rather than the model parameter device.
            torch.cuda.set_device(self.device)
        datapoint = create_empty_datapoint()
        set_image(datapoint, image_pil)

        query_map = {} 
        for p in prompts:
            query_id = add_text_prompt(datapoint, p)
            query_map[p] = query_id

        datapoint = self.transform(datapoint)
        
        batch = collate([datapoint], dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, torch.device(self.device), non_blocking=True)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model(batch)

        postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=threshold,
            to_cpu=True 
        )
        
        processed_results = postprocessor.process_results(output, batch.find_metadatas)
        
        final_results = {}
        for prompt, q_id in query_map.items():
            if q_id in processed_results:
                res = processed_results[q_id]
                if res['boxes'] is not None and len(res['boxes']) > 0:
                    
                    
                    masks_np = res['masks'].float().numpy()
                    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                        masks_np = masks_np.squeeze(1)
                    
                    final_results[prompt] = {
                        'boxes': res['boxes'].float().numpy(),
                        'scores': res['scores'].float().numpy(),
                        'masks': masks_np
                    }
                    
                else:
                    final_results[prompt] = None
            else:
                final_results[prompt] = None
                
        return final_results


    def _mask_to_rle(self, binary_mask):
        rle = {'counts': [], 'size': list(binary_mask.shape)}
        counts = rle.get('counts')
        for i, (value, length) in enumerate(self._run_length_encoding(binary_mask.flatten())):
            if i == 0 and value == 1:
                counts.append(0)
            counts.append(length)
        return rle

    def _run_length_encoding(self, x):
        from itertools import groupby
        return ((x, len(list(y))) for x, y in groupby(x))

    def predict(self, image_pil: Image.Image, text_prompts: list, 
                    using_predefined_classes: bool = False,
                    output_dir: str = None, file_id: str = "temp") -> str:
            
            all_detections = []
            
            image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            h_img, w_img = image_pil.size[1], image_pil.size[0]

            # Kept crops are runtime memory. Rejected crops and overlap
            # visualizations are diagnostics and stay disabled in production.
            if output_dir: 
                os.makedirs(output_dir, exist_ok=True)
                cropped_kept_dir = os.path.join(output_dir, "cropped_objects", "kept")
                os.makedirs(cropped_kept_dir, exist_ok=True)
                if self.save_debug_artifacts:
                    cropped_dropped_dir = os.path.join(output_dir, "cropped_objects", "dropped")
                    cropped_contain_dir = os.path.join(output_dir, "cropped_objects", "containment_dropped")
                    os.makedirs(cropped_dropped_dir, exist_ok=True)
                    os.makedirs(cropped_contain_dir, exist_ok=True)
                    debug_vis_dir = os.path.join(output_dir, "debug_overlaps")
                    os.makedirs(debug_vis_dir, exist_ok=True)

           
            # =========================================================
            # =========================================================
            final_prompts = []
            current_batch_size = self.category_batch_size 

            if using_predefined_classes:
                print(f"  [Mode] Using Predefined Classes ({len(text_prompts)} tags)")
                final_prompts = text_prompts
            else:
                print(f"  [Mode] Using RAM++ for Image Tagging...")
                if not hasattr(self, 'ram_model') or self.ram_model is None:
                    raise RuntimeError("RAM++ model is not initialized.")
                
                with torch.no_grad():
                    ram_img_tensor = self.ram_transform(image_pil).unsqueeze(0).to(self.device)
                    
                    res = inference(ram_img_tensor, self.ram_model)
                    tags_str = res[0] 
                    print(f"  [RAM++] Raw Tags: {tags_str}")
                    generated_tags = [tag.strip() for tag in tags_str.split('|') if tag.strip()]
                    
                    if hasattr(self, 'llm_handler') and self.llm_handler:
                        print(f"  [Filter] Filtering {len(generated_tags)} raw tags via LLM...")
                        final_prompts = self.llm_handler.filter_object_tags(generated_tags)
                        print(f"  [Filter] {final_prompts} tags remain")
                    else:
                        print(f"  [Filter] No LLM handler found, skipping filtering.")
                        final_prompts = generated_tags

                    if len(final_prompts) > 0:
                        current_batch_size = len(final_prompts)
                        print(f"  [RAM++] Processed {len(final_prompts)} tags (Processing ALL at once).")
                    
            if not final_prompts:
                print("  [Warning] No prompts found to process.")
                return ""

            # =========================================================
            # =========================================================
            total_prompts = len(final_prompts)
            print(f"  [SAM3] Processing {total_prompts} categories with batch size {current_batch_size}...")
            
            for batch_idx in range(0, total_prompts, current_batch_size):
                batch_prompts = final_prompts[batch_idx:batch_idx + current_batch_size]
                batch_results = self._batch_inference(image_pil, batch_prompts, self.user_threshold)
                
                for prompt in batch_prompts:
                    res = batch_results.get(prompt)
                    if res is None: continue
                    
                    boxes = res['boxes']
                    scores = res['scores']
                    masks = res['masks']
                    
                    for i, score in enumerate(scores):
                        x1, y1, x2, y2 = boxes[i]
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))

                        if (x2 - x1) < 5 or (y2 - y1) < 5: continue

                        mask_binary = masks[i] > 0
                        object_id_str = f"{prompt}_{len(all_detections)}"

                        temp_crop_data = None
                        temp_filename = ""
                        
                        if output_dir:
                            cropped_object_cv2 = image_cv2[y1:y2, x1:x2]
                            mask_crop = mask_binary[y1:y2, x1:x2]
                            
                            cropped_alpha = np.zeros((cropped_object_cv2.shape[0], cropped_object_cv2.shape[1], 4), dtype=np.uint8)
                            cropped_alpha[:, :, :3] = cropped_object_cv2
                            if mask_crop.shape == cropped_object_cv2.shape[:2]:
                                cropped_alpha[:, :, 3] = mask_crop * 255
                            else:
                                cropped_alpha[:, :, 3] = 255 

                            safe_label = "".join(c for c in prompt if c.isalnum() or c in ('_', '-')).rstrip()
                            temp_filename = f"{safe_label}_{object_id_str}.png"
                            temp_crop_data = cropped_alpha 
                        
                        all_detections.append({
                            "object_id": object_id_str,
                            "label": prompt,
                            "score": float(score),
                            "bounding_box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                            "cropped_image_path": "",
                            "_temp_crop_data": temp_crop_data,
                            "_temp_filename": temp_filename,
                            "_temp_mask": mask_binary,
                            "_drop_reason": None
                        })
                
                torch.cuda.empty_cache()

            # =========================================================
            # [Post-Processing] NMS + Mask Containment + Semantic Check
            # =========================================================
            
            all_detections.sort(key=lambda x: x['score'], reverse=True)
            
            num_dets = len(all_detections)
            keep = [True] * num_dets 
            overlap_threshold = 0.85 
            
            SEMANTIC_THRESH = 0.6
            CONTAIN_THRESH = 0.90 

            print(f"\n  [NMS] Checking overlaps & containment across {num_dets} detections...")
            
            if num_dets > 0:
                for i in range(num_dets):
                    if not keep[i]: continue
                    
                    for j in range(i + 1, num_dets):
                        if not keep[j]: continue
                        
                        det_a = all_detections[i] # High Score
                        det_b = all_detections[j] # Low Score
                        
                        bbox_a = det_a['bounding_box_xyxy']
                        bbox_b = det_b['bounding_box_xyxy']
                        iou = self.calculate_iou(bbox_a, bbox_b)
                        
                        if iou > overlap_threshold:
                            keep[j] = False
                            det_b['_drop_reason'] = "iou_dropped"
                            continue

                        mask_a = det_a['_temp_mask']
                        mask_b = det_b['_temp_mask']
                        
                        intersection = np.logical_and(mask_a, mask_b).sum()
                        area_a = mask_a.sum()
                        area_b = mask_b.sum()
                        
                        if area_a == 0 or area_b == 0: continue
                        
                        ratio_b_in_a = intersection / area_b
                        ratio_a_in_b = intersection / area_a
                        
                        is_contained = False
                        container_is_a = True
                        
                        if ratio_b_in_a > CONTAIN_THRESH:
                            is_contained = True
                            container_is_a = True
                        elif ratio_a_in_b > CONTAIN_THRESH:
                            is_contained = True
                            container_is_a = False

                        if is_contained and self.text_encoder:
                            label_a = det_a['label']
                            label_b = det_b['label']
                            
                            emb = self.text_encoder.encode([label_a, label_b], convert_to_tensor=True)
                            sim_score = util.cos_sim(emb[0], emb[1]).item()
                            
                            if sim_score > SEMANTIC_THRESH:
                                
                                target_drop_idx = -1
                                drop_label = ""
                                keep_label = ""
                                
                                if container_is_a:
                                    target_drop_idx = j
                                    drop_label = label_b
                                    keep_label = label_a
                                    det_b['_drop_reason'] = "containment_dropped"
                                else:
                                    target_drop_idx = i
                                    drop_label = label_a
                                    keep_label = label_b
                                    det_a['_drop_reason'] = "containment_dropped"
                                
                                keep[target_drop_idx] = False
                                
                                print(f"    ❌ [Containment Drop] '{drop_label}' is part of '{keep_label}' "
                                      f"(Sim: {sim_score:.2f}). Dropping smaller part.")
                                
                                if target_drop_idx == i:
                                    break 

            if output_dir:
                print(f"  [Save] Saving crops...")
                for i, det in enumerate(all_detections):
                    img_data = det.pop('_temp_crop_data', None)
                    fname = det.pop('_temp_filename', None)
                    drop_reason = det.pop('_drop_reason', None)
                    
                    if img_data is not None and fname:
                        if keep[i]:
                            save_path = os.path.join(cropped_kept_dir, fname)
                            try:
                                cv2.imwrite(save_path, img_data)
                                det['cropped_image_path'] = os.path.join("cropped_objects", "kept", fname)
                            except: pass
                        elif self.save_debug_artifacts:
                            if drop_reason == "containment_dropped":
                                save_path = os.path.join(cropped_contain_dir, fname)
                            else:
                                save_path = os.path.join(cropped_dropped_dir, fname)
                            
                            try:
                                cv2.imwrite(save_path, img_data)
                            except: pass
            else:
                for det in all_detections:
                    det.pop('_temp_crop_data', None)
                    det.pop('_temp_filename', None)
                    det.pop('_drop_reason', None)

           
            
            cropped_small_dir = ""
            if output_dir and self.save_debug_artifacts:
                cropped_small_dir = os.path.join(output_dir, "cropped_objects", "small_dropped")
                os.makedirs(cropped_small_dir, exist_ok=True)

            final_detections = []
            
            
            
            import shutil

            count_small_removed = 0

            for k, det in zip(keep, all_detections):
                mask_bin = det.pop('_temp_mask', None) 
                
                if k:
                    x1, y1, x2, y2 = det['bounding_box_xyxy']
                    w_box = x2 - x1
                    h_box = y2 - y1
                    area = w_box * h_box
                    normalized_label = " ".join(
                        str(det.get('label', '')).strip().lower()
                        .replace("_", " ").split()
                    )
                    is_small_class = normalized_label in self.small_object_labels
                    min_area = (
                        self.small_min_area_px
                        if is_small_class
                        else self.default_min_area_px
                    )
                    min_side = (
                        self.small_min_side_px
                        if is_small_class
                        else self.default_min_side_px
                    )

                    if area < min_area or w_box < min_side or h_box < min_side:
                        count_small_removed += 1
                        
                        if output_dir and 'cropped_image_path' in det:
                            rel_path = det['cropped_image_path']
                            full_src_path = os.path.join(output_dir, rel_path)
                            
                            if os.path.exists(full_src_path):
                                try:
                                    if self.save_debug_artifacts:
                                        fname = os.path.basename(full_src_path)
                                        full_dst_path = os.path.join(cropped_small_dir, fname)
                                        shutil.move(full_src_path, full_dst_path)
                                    else:
                                        os.unlink(full_src_path)
                                except Exception as e:
                                    print(f"Warning: Failed to discard small object image: {e}")
                        
                        continue
                    
                    if mask_bin is not None:
                        det['segmentation'] = self._mask_to_rle(mask_bin)
                    final_detections.append(det)
            
            print(
                "  [Size Filter] Removed "
                f"{count_small_removed} detections; default="
                f"{self.default_min_area_px}px/{self.default_min_side_px}px, "
                f"reviewed-small={self.small_min_area_px}px/"
                f"{self.small_min_side_px}px."
            )
            print(f"  [NMS] Finished. {num_dets} -> {len(final_detections)} objects remaining.")

            
            json_path = ""
            if output_dir:
                json_path = os.path.join(output_dir, f"pred_{file_id}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(final_detections, f, indent=4, ensure_ascii=False)

            return json_path
    
    # def predict(self, image_pil: Image.Image, text_prompts: list, 
    #             using_predefined_classes: bool = False,
    #             output_dir: str = None, file_id: str = "temp") -> str:
        
    #     all_detections = []
        
    #     image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    #     h_img, w_img = image_pil.size[1], image_pil.size[0]

    #     if output_dir: 
    #         os.makedirs(output_dir, exist_ok=True)
    #         cropped_kept_dir = os.path.join(output_dir, "cropped_objects", "kept")
    #         cropped_dropped_dir = os.path.join(output_dir, "cropped_objects", "dropped")
    #         os.makedirs(cropped_kept_dir, exist_ok=True)
    #         os.makedirs(cropped_dropped_dir, exist_ok=True)
            
    #         debug_vis_dir = os.path.join(output_dir, "debug_overlaps")
    #         os.makedirs(debug_vis_dir, exist_ok=True)
    #     # =========================================================
    #     # =========================================================
    #     final_prompts = []

    #     if using_predefined_classes:
    #         print(f"  [Mode] Using Predefined Classes ({len(text_prompts)} tags)")
    #         final_prompts = text_prompts
    #     else:
    #         print(f"  [Mode] Using RAM++ for Image Tagging...")
            
    #         if not hasattr(self, 'ram_model') or self.ram_model is None:
    #             raise RuntimeError("RAM++ model is not initialized. Please check __init__.")
            
    #         with torch.no_grad():
    #             ram_img_tensor = self.ram_transform(image_pil).unsqueeze(0).to(self.device)
    #             res = inference(ram_img_tensor, self.ram_model)
                
    #             tags_str = res[0] # e.g., "chair | table | lamp"
    #             print(f"  [RAM++] Raw Tags: {tags_str}")
                
    #             generated_tags = [tag.strip() for tag in tags_str.split('|') if tag.strip()]
                
                
                
    #             if hasattr(self, 'llm_handler') and self.llm_handler:
    #                 print(f"  [Filter] Filtering {len(generated_tags)} raw tags via LLM...")
    #                 final_prompts = self.llm_handler.filter_object_tags(generated_tags)
    #                 print(f"  [Filter] {final_prompts} tags remain")
    #             else:
    #                 print(f"  [Filter] No LLM handler found, skipping filtering.")
    #                 final_prompts = generated_tags

    #             if len(final_prompts) > 0:
    #                 current_batch_size = len(final_prompts)
    #                 print(f"  [RAM++] Processed {len(final_prompts)} tags (Processing ALL at once).")
                
    #     if not final_prompts:
    #         print("  [Warning] No prompts found to process.")
    #         return ""

    #     # =========================================================
    #     # =========================================================
    #     total_prompts = len(final_prompts)
    #     print(f"  [SAM3] Processing {total_prompts} categories with batch size {current_batch_size}...")
        
    #     for batch_idx in range(0, total_prompts, current_batch_size):
    #         batch_prompts = final_prompts[batch_idx:batch_idx + current_batch_size]
            
    #         batch_results = self._batch_inference(image_pil, batch_prompts, self.user_threshold)
            
    #         for prompt in batch_prompts:
    #             res = batch_results.get(prompt)
    #             if res is None: continue
                
    #             boxes = res['boxes']
    #             scores = res['scores']
    #             masks = res['masks']
                
    #             for i, score in enumerate(scores):
    #                 x1, y1, x2, y2 = boxes[i]
    #                 x1, y1 = max(0, int(x1)), max(0, int(y1))
    #                 x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))

    #                 if (x2 - x1) < 5 or (y2 - y1) < 5: continue

    #                 mask_binary = masks[i] > 0
    #                 object_id_str = f"{prompt}_{len(all_detections)}"

    #                 temp_crop_data = None
    #                 temp_filename = ""
                    
    #                 if output_dir:
    #                     cropped_object_cv2 = image_cv2[y1:y2, x1:x2]
    #                     mask_crop = mask_binary[y1:y2, x1:x2]
                        
    #                     cropped_alpha = np.zeros((cropped_object_cv2.shape[0], cropped_object_cv2.shape[1], 4), dtype=np.uint8)
    #                     cropped_alpha[:, :, :3] = cropped_object_cv2
                        
    #                     if mask_crop.shape == cropped_object_cv2.shape[:2]:
    #                         cropped_alpha[:, :, 3] = mask_crop * 255
    #                     else:
    #                         cropped_alpha[:, :, 3] = 255 

    #                     safe_label = "".join(c for c in prompt if c.isalnum() or c in ('_', '-')).rstrip()
    #                     temp_filename = f"{safe_label}_{object_id_str}.png"
                    
    #                 all_detections.append({
    #                     "object_id": object_id_str,
    #                     "label": prompt,
    #                     "score": float(score),
    #                     "bounding_box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
    #                     "_temp_crop_data": temp_crop_data,
    #                     "_temp_filename": temp_filename
    #                 })
                    
            
    #         torch.cuda.empty_cache()
    #     # =========================================================
    #     # =========================================================
        
    #     all_detections.sort(key=lambda x: x['score'], reverse=True)
        
    #     num_dets = len(all_detections)
        
    #     print(f"\n  [NMS] Checking for overlaps across {num_dets} detections (Threshold: {overlap_threshold})...")
        
    #     if output_dir and num_dets > 0:
    #         for i in range(num_dets):
                
    #             for j in range(i + 1, num_dets):
                    
                    
    #                 bbox_a = det_a['bounding_box_xyxy']
    #                 bbox_b = det_b['bounding_box_xyxy']
                    
    #                 iou = self.calculate_iou(bbox_a, bbox_b)
                    
    #                 if iou > overlap_threshold:
    #                     keep[j] = False
                        
    #                     id_a = det_a['object_id']
    #                     id_b = det_b['object_id']
    #                     score_a = det_a['score']
    #                     score_b = det_b['score']
                        
    #                     print(f"    ❌ [Drop] {id_b}({score_b:.2f}) overlaps with {id_a}({score_a:.2f}) | IoU: {iou:.3f}")
                        
    #                     if output_dir:
    #                         img_vis = image_cv2.copy()
                            
    #                         cv2.rectangle(img_vis, (bbox_a[0], bbox_a[1]), (bbox_a[2], bbox_a[3]), (0, 255, 0), 3)
    #                         text_a = f"KEEP: {id_a} ({score_a:.2f})"
    #                         cv2.putText(img_vis, text_a, (bbox_a[0], max(20, bbox_a[1]-10)), 
    #                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
    #                         cv2.rectangle(img_vis, (bbox_b[0], bbox_b[1]), (bbox_b[2], bbox_b[3]), (0, 0, 255), 3)
    #                         text_b = f"DROP: {id_b} ({score_b:.2f})"
    #                         cv2.putText(img_vis, text_b, (bbox_b[0], bbox_b[3] + 20), 
    #                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
    #                         cv2.putText(img_vis, f"IoU Collision: {iou:.3f}", (20, 30), 
    #                                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                            
    #                         safe_a = "".join(c for c in id_a if c.isalnum())
    #                         safe_b = "".join(c for c in id_b if c.isalnum())
                           
                           
    #                         filename = f"nms_drop_{i}_{j}_{safe_b}_vs_{safe_a}.jpg"
    #                         save_path = os.path.join(debug_vis_dir, filename)
    #                         cv2.imwrite(save_path, img_vis)

    #     if output_dir:
    #         print(f"  [Save] Saving crops to kept/dropped folders...")
    #         for i, det in enumerate(all_detections):
    #             img_data = det.pop('_temp_crop_data', None)
    #             fname = det.pop('_temp_filename', None)
                
    #             if img_data is not None and fname:
    #                 if keep[i]:
    #                     save_path = os.path.join(cropped_kept_dir, fname)
    #                     try:
    #                         cv2.imwrite(save_path, img_data)
    #                         det['cropped_image_path'] = os.path.join("cropped_objects", "kept", fname)
    #                     except Exception: pass
    #                 else:
    #                     save_path = os.path.join(cropped_dropped_dir, fname)
    #                     try:
    #                         cv2.imwrite(save_path, img_data)
    #                     except Exception: pass
    #     else:
    #         for det in all_detections:
    #             det.pop('_temp_crop_data', None)
    #             det.pop('_temp_filename', None)

    #     final_detections = [d for k, d in zip(keep, all_detections) if k]
        
    #     print(f"  [NMS] Finished. {num_dets} -> {len(final_detections)} objects remaining.")
       
       
    #     json_path = ""
    #     if output_dir:
    #         json_path = os.path.join(output_dir, f"pred_{file_id}.json")
    #         with open(json_path, 'w', encoding='utf-8') as f:
    #             json.dump(final_detections, f, indent=4, ensure_ascii=False)

    #     return json_path
        
    
    def calculate_iou(self,box1, box2):
        """Compute the intersection over union of two boxes."""
        x1_a, y1_a, x2_a, y2_a = box1
        x1_b, y1_b, x2_b, y2_b = box2

        x1_inter = max(x1_a, x1_b)
        y1_inter = max(y1_a, y1_b)
        x2_inter = min(x2_a, x2_b)
        y2_inter = min(y2_a, y2_b)

        inter_w = max(0, x2_inter - x1_inter)
        inter_h = max(0, y2_inter - y1_inter)
        inter_area = inter_w * inter_h

        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)

        union_area = area_a + area_b - inter_area
        if union_area == 0: return 0
        return inter_area / union_area

    
       
class FloorTracker:
    def __init__(self, output_path: str, edges_path: str = None):
        self.nodes_path = output_path
        self.edges_path = edges_path
        
        self.floor_height_step = 3.5
        self.min_consistent_frames = 20
        
        self.current_floor_id = 0
        self.base_height = None
        
        self.height_buffer = deque(maxlen=self.min_consistent_frames)
        self.id_buffer = deque(maxlen=self.min_consistent_frames)
        
        self.floors_data = defaultdict(lambda: {'heights': [], 'rooms': set(), 'finalized_count': 0})
        
        self.pending_data_cache = [] 

    def _calculate_floor_id(self, y: float) -> int:
        """Quantize a floor id from height relative to the reference plane."""
        if self.base_height is None:
            return 0
        #       (5.8 - 0) / 3.0 = 1.93 -> 2
        return int(np.round((y - self.base_height) / self.floor_height_step))

    def add_room(self, room_node_id: str):
        """Add a room to the current floor."""
        if room_node_id:
            self.floors_data[self.current_floor_id]['rooms'].add(room_node_id)

    def update(self, pose: np.ndarray, current_room_node_id: str) -> bool:
        """Update state and return true when the floor changes."""
        if pose is None: return False
        
        y = float(pose[1, 3])
        
        if self.base_height is None:
            self.base_height = y
            self.floors_data[0]['heights'].append(y)
            if current_room_node_id:
                self.floors_data[0]['rooms'].add(current_room_node_id)
            print(f"[FloorTracker] Initialized Base Height: {self.base_height:.2f}m")
            return False

        suggested_id = self._calculate_floor_id(y)
        
        self.height_buffer.append(y)
        self.id_buffer.append(suggested_id)
        self.pending_data_cache.append(y)

        if len(self.id_buffer) == self.min_consistent_frames:
            unique_ids = set(self.id_buffer)
            
            if len(unique_ids) == 1:
                target_id = list(unique_ids)[0]
                
                if target_id != self.current_floor_id:
                    print(f"[FloorTracker] Floor Switch Detected: {self.current_floor_id} -> {target_id}")
                    
                    
                    
                    self.next_floor_id = target_id 
                    return True

        if len(self.pending_data_cache) > 0:
            self.floors_data[self.current_floor_id]['heights'].extend(self.pending_data_cache)
            self.pending_data_cache = []
            
            if current_room_node_id:
                self.floors_data[self.current_floor_id]['rooms'].add(current_room_node_id)

        return False

    def get_floor_id_by_object_y(self, obj_y: float) -> int:
        """Return the floor id for an object height."""
        calc_id = self._calculate_floor_id(obj_y)
        # if calc_id == 1 or calc_id == -1:
        #     print(f"[FloorTracker] Object Y: {obj_y:.2f}m -> Calculated Floor ID: {calc_id}")
        #     assert False, "Debug Breakpoint"
        
        return calc_id

    def finalize_floor(self):
        """Finalize and save the nodes of the current floor."""
        fid = self.current_floor_id
        data = self.floors_data[fid]
        
        heights = data['heights']
        if not heights:
            avg_h = self.base_height + (fid * self.floor_height_step)
        else:
            if len(heights) > 10:
                sorted_h = np.sort(heights)
                lower = int(len(heights) * 0.1)
                upper = int(len(heights) * 0.9)
                avg_h = np.mean(sorted_h[lower:upper])
            else:
                avg_h = np.mean(heights)

        rooms = list(data['rooms'])
        floor_node_id = f"floor_{fid:02d}" if fid >= 0 else f"floor_neg_{abs(fid):02d}"
        
        data['finalized_count'] += 1
        
        summary = f"Floor {fid} (Avg Height: {avg_h:.2f}m). Contains {len(rooms)} rooms."
        
        node = {
            "node_id": floor_node_id,
            "node_level": 0,
            "floor_index": fid,
            "height_info": {"avg": avg_h, "min": np.min(heights) if heights else avg_h, "max": np.max(heights) if heights else avg_h},
            "contained_room_nodes": rooms,
            "llm_summary": summary,
            "timestamp": datetime.datetime.now().isoformat(),
            "visit_count": data['finalized_count']
        }
        
        with open(self.nodes_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(node, ensure_ascii=False) + "\n")
        
        if self.edges_path and rooms:
            with open(self.edges_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for rid in rooms:
                    writer.writerow([rid, floor_node_id, 1])
        
        print(f"[FloorTracker] Finalized {floor_node_id}. Prepare to switch.")

        if hasattr(self, 'next_floor_id'):
            self.current_floor_id = self.next_floor_id
            del self.next_floor_id
            
            self.height_buffer.clear()
            self.id_buffer.clear()
            self.pending_data_cache = []
        

# ==============================================================================
# Pipeline Class
# ==============================================================================

class TrajectoryPipeline:
    def __init__(self, config_path: str, override_settings: Dict = None, shared_models: Dict = None):
        # =========================================================
        # =========================================================
        if isinstance(config_path, str):
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found at '{config_path}'")
            with open(config_path, 'r') as f: self.config = yaml.safe_load(f)
        else:
            self.config = config_path 

        if override_settings:
            if 'trajectory_data' in override_settings:
                self.config['trajectory_data'].update(override_settings['trajectory_data'])
            if 'main_output_directory' in override_settings:
                self.config['main_output_directory'] = override_settings['main_output_directory']
            if 'debug_mode' in override_settings:
                self.config['debug_mode'] = override_settings['debug_mode']

        # =========================================================
        # =========================================================
        self.run_dir = self.config['main_output_directory']
        self.main_output_directory = self.run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.external_context = None 
        traj_cfg = self.config.get('trajectory_data', {})
        self.trajectory_data_dir = traj_cfg.get('base_dir', './temp_data')
        self.filename_prefix = traj_cfg.get('filename_prefix', "")
        
        self.rgb_dir = os.path.join(self.trajectory_data_dir, traj_cfg.get('rgb_subdir', 'rgb'))
        self.depth_dir = os.path.join(self.trajectory_data_dir, traj_cfg.get('depth_subdir', 'depth'))
        self.pose_dir = os.path.join(self.trajectory_data_dir, traj_cfg.get('pose_subdir', 'pose'))
        
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        if not os.path.exists(self.pose_dir): os.makedirs(self.pose_dir, exist_ok=True)




        self.run_logger = Logger(os.path.join(self.run_dir, "run_log.log"))
        
        with open(os.path.join(self.run_dir, "config_snapshot.yaml"), 'w') as f:
            yaml.dump(self.config, f)

        # =========================================================
        # =========================================================
        self.debug_mode = self.config.get('debug_mode', False)
        
        self.sam3_output_root = os.path.join(self.run_dir, "sam3_outputs")
        if self.debug_mode:
            os.makedirs(self.sam3_output_root, exist_ok=True)

        self.debug_config = self.config.get('debug_config', {})
        
        self.online_map_viz_dir = os.path.join(self.run_dir, "debug_online_mapping")
        self.room_viz_dir = os.path.join(self.run_dir, "debug_room_segmentation")
        if self.debug_mode:
            os.makedirs(self.online_map_viz_dir, exist_ok=True)
            os.makedirs(self.room_viz_dir, exist_ok=True)

        self.enable_reid_viz = self.debug_mode and self.debug_config.get('save_reid_visualizations', False)
        self.reid_viz_dir = os.path.join(self.run_dir, "debug_reid_viz") if self.enable_reid_viz else None
        if self.reid_viz_dir: os.makedirs(self.reid_viz_dir, exist_ok=True)

        self.save_annotated_frames = self.debug_mode and self.debug_config.get('save_llm_annotated_frames', False)
        self.annotated_frames_dir = os.path.join(self.run_dir, "debug_annotated_frames") if self.save_annotated_frames else None
        if self.annotated_frames_dir: os.makedirs(self.annotated_frames_dir, exist_ok=True)

        self.room_summary_img_dir = os.path.join(self.run_dir, "room_summary_images")
        os.makedirs(self.room_summary_img_dir, exist_ok=True)
        # =========================================================
        # =========================================================
        self.spatial_graph = nx.DiGraph()
        self.robot_trajectory = []
        self.memory_kb = [] 
        self.label_counters = {}
        self.floor_trajectories = {} 
        self.room_frame_history = {} 
        self.pose_cache = {}
        self.room_metadata_cache = {} 
        self.room_image_buffer = {}
        self.current_room_id = 0
        self.room_topology_cache = {}
        self.room_exit_buffer = {}

        self.visited_room_ids = set() 
        self.global_update_history = []
        self.last_kf_feature = None 
        
        self.last_room_id = 0 

        self.nodes_path = os.path.join(self.run_dir, self.config['semantic_forest']['nodes_file'])
        self.edges_path = os.path.join(self.run_dir, self.config['semantic_forest']['edges_file'])
        
        open(self.nodes_path, 'w').close()
        with open(self.edges_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["child_node_id", "parent_node_id", "node_level"])

        reid_cfg = self.config.get('reidentification', {})
        self.reid_sim_threshold = reid_cfg.get('similarity_threshold', 0.75)
        self.reid_spatial_threshold = reid_cfg.get('spatial_threshold', 0.5) 
        
        spatial_cfg = self.config.get('spatial_reasoning', {})
        self.spatial_cluster_thresh = spatial_cfg.get('clustering_distance_threshold', 1.0) 
        
        self.finalize_delay_kf = int(self.config.get("room_segmentation", {}).get("finalize_delay_kf", 20))

        self.traj_base_dir = self.config['trajectory_data']['base_dir']
        self.pose_dir = os.path.join(self.traj_base_dir, "pose") 

        cam_cfg = self.config.get('camera_intrinsics', {})
        self.intrinsics = {
            'fx': cam_cfg.get('fx', 540.0),
            'fy': cam_cfg.get('fy', 540.0),
            'cx': cam_cfg.get('cx', 540.0),
            'cy': cam_cfg.get('cy', 360.0),
            'depth_scale': cam_cfg.get('depth_scale', 1000.0)
        }
        print(self.intrinsics)

        self.map_output_dir = os.path.join(self.run_dir, "trajectory_maps")
        if self.debug_mode:
            os.makedirs(self.map_output_dir, exist_ok=True)
        self.map_manager = MapManager(self.config.get('map_config', {}), self.intrinsics, self.run_dir)
        self.room_segmenter = GeometryRoomTracker(self.map_manager, self.config.get('room_segmentation', {}))
        self.floor_tracker = FloorTracker(self.nodes_path, self.edges_path)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config["gpu_id"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        # =========================================================
        # =========================================================
        self.gpu_config = self.config.get('multi_gpu', {
            'enabled': True,
            'sam3_device': 'cuda:0',
            'dinov3_device': 'cuda:0',
            'text_encoder_device': 'cuda:0'
        })
        
        if self.gpu_config['enabled']:
            print("[Multi-GPU] Enabled. Model distribution:")
            for k, v in self.gpu_config.items():
                if k != 'enabled':
                    print(f"  {k}: {v}")
        
        sam3_device = self.gpu_config.get('sam3_device', 'cuda:0') if self.gpu_config['enabled'] else self.device
        dinov3_device = self.gpu_config.get('dinov3_device', 'cuda:0') if self.gpu_config['enabled'] else self.device
        text_device = self.gpu_config.get('text_encoder_device', 'cuda:0') if self.gpu_config['enabled'] else self.device
        
        # =========================================================
        # =========================================================
        if shared_models and 'dinov3' in shared_models:
            print("[Pipeline] Using shared DINOv3 model.")
            self.dinov3_model = shared_models['dinov3']
            if self.gpu_config['enabled']:
                self.dinov3_model = self.dinov3_model.to(dinov3_device)
        else:
            print(f"[Pipeline] Loading DINOv3 model on {dinov3_device}...")
            self.dinov3_model = load_feature_extractor(self.config, dinov3_device)
        
        if shared_models and 'text_encoder' in shared_models:
            self.text_encoder = shared_models['text_encoder']
            if self.gpu_config['enabled']:
                self.text_encoder = self.text_encoder.to(text_device)
        else:
            from sentence_transformers import SentenceTransformer
            self.text_encoder = SentenceTransformer(os.environ.get("INHERIT_SG_TEXT_ENCODER", ""), device=text_device)
            
        if shared_models and 'sam3' in shared_models:
            print("[Pipeline] Using shared SAM 3 client.")
            self.sam3_client = shared_models['sam3']
        else:
            print(f"[Pipeline] Loading SAM 3 client on {sam3_device}...")
            seg_config = self.config.get("segmentation_module", {})
            self.sam3_client = Sam3LocalClient(self.config, device=sam3_device)
        
        self.sam3_device = sam3_device
        self.dinov3_device = dinov3_device
        self.text_device = text_device

        print(f"filter_edge_detections: {self.config['filter_edge_detections']}")
        
        if not self.config.get("disable_llm", False):
            try:
                self.llm = LLMHandler(self.config['llm_config'])
            except NameError:
                pass
        
        self.kf_config = self.config.get('keyframe_detection', {})
        self.enable_kf = self.kf_config.get('enable_keyframe_detection', False)
        self.kf_threshold = self.kf_config.get('keyframe_similarity_threshold', 0.98)
        self.last_kf_feature = None
        provenance_config = self.config.get('episode_provenance', {})
        self.current_runtime_episode_id = normalize_episode_id(
            provenance_config.get('runtime_episode_id')
        )
        self.current_source_episode_id = (
            normalize_episode_id(provenance_config.get('source_episode_id'))
            or self.current_runtime_episode_id
        )
        self.provenance_center_weight = float(
            provenance_config.get('selection_center_weight', 0.4)
        )
        self.provenance_area_weight = float(
            provenance_config.get('selection_area_weight', 0.6)
        )
        self._current_frame_timestamp = None

    def set_episode_context(self, runtime_episode_id, source_episode_id=None):
        """Set the provenance identity applied to subsequently observed objects."""
        self.current_runtime_episode_id = normalize_episode_id(runtime_episode_id)
        self.current_source_episode_id = (
            normalize_episode_id(source_episode_id)
            or self.current_runtime_episode_id
        )
        self.config['episode_provenance'] = {
            'schema_version': 1,
            'runtime_episode_id': self.current_runtime_episode_id,
            'source_episode_id': self.current_source_episode_id,
            'selection_center_weight': self.provenance_center_weight,
            'selection_area_weight': self.provenance_area_weight,
        }

    def _record_object_episode_provenance(
        self,
        item,
        *,
        frame_index,
        current_rgb_path,
        current_pose,
        bbox,
        object_position,
        image_width,
        image_height,
    ):
        """Record mapper-native provenance without requiring the runtime adapter."""
        ensure_object_provenance(item)
        pose = np.asarray(current_pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            sync_object_provenance_to_graph(self.spatial_graph, item)
            return

        position = pose[:3, 3]
        heading_rad = float(np.arctan2(pose[0, 2], pose[0, 0]))
        rotation_xyzw = [
            0.0,
            float(np.sin(heading_rad / 2.0)),
            0.0,
            float(np.cos(heading_rad / 2.0)),
        ]
        pixel_area = bbox_pixel_area(bbox)
        selection_score = observation_selection_score(
            bbox,
            image_width,
            image_height,
            center_weight=self.provenance_center_weight,
            area_weight=self.provenance_area_weight,
        )
        observation_distance_m = None
        try:
            object_xyz = np.asarray(object_position, dtype=np.float64).reshape(-1)[:3]
            if object_xyz.size == 3 and np.all(np.isfinite(object_xyz)):
                observation_distance_m = float(np.linalg.norm(object_xyz - position))
        except (TypeError, ValueError):
            pass

        observation = {
            'frame_idx': int(frame_index),
            'timestamp': float(
                self._current_frame_timestamp
                if self._current_frame_timestamp is not None
                else frame_index
            ),
            'rgb_path': str(current_rgb_path),
            'bbox_xyxy': list(bbox) if bbox is not None else None,
            'pixel_area': pixel_area,
            'pixel_area_fraction': (
                float(pixel_area) / float(image_width * image_height)
                if pixel_area is not None and image_width > 0 and image_height > 0
                else None
            ),
            'observation_distance_m': observation_distance_m,
            'selection_score': selection_score,
            'agent_position': position.tolist(),
            'agent_heading_rad': heading_rad,
            'agent_rotation_xyzw': rotation_xyzw,
            'forward_world': (-pose[:3, 2]).tolist(),
            'camera_to_world': pose.tolist(),
            'coordinate_frame': 'habitat_world',
            'trajectory_node_id': None,
            'selection_policy': 'minimum_weighted_center_area_observation_score',
            'selection_weights': {
                'center_offset': self.provenance_center_weight,
                'bbox_area': self.provenance_area_weight,
            },
        }
        waypoint = {
            'position': position.tolist(),
            'heading_rad': heading_rad,
            'rotation_xyzw': rotation_xyzw,
            'frame_idx': int(frame_index),
            'source': 'best_observation_pose',
            'coordinate_frame': 'habitat_world',
            'trajectory_node_id': None,
        }
        record_object_observation(
            item,
            runtime_episode_id=self.current_runtime_episode_id,
            source_episode_id=self.current_source_episode_id,
            observation=observation,
            waypoint=waypoint,
            selection_score=selection_score,
        )
        sync_object_provenance_to_graph(self.spatial_graph, item)
    
    def override_context(self, context_data: Dict):
        """Accept the room id and floor id supplied by the caller."""
        self.external_context = context_data


    def _annotate_image_with_clusters(self, image_path: str, nodes_in_frame: List[Dict], 
                                      clusters: List[List[str]], frame_index: int) -> str:
        """Draw the clustering result."""
        img = cv2.imread(image_path)
        if img is None: return image_path
        
        base_colors = [
            (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), 
            (255, 0, 255), (255, 255, 0), (128, 0, 128), (0, 128, 128)
        ]
        
        node_color_map = {}
        for i, cluster in enumerate(clusters):
            color = base_colors[i % len(base_colors)]
            if i >= len(base_colors):
                color = np.random.randint(50, 255, size=3).tolist()
            
            for node_id in cluster:
                node_color_map[node_id] = color

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for node in nodes_in_frame:
            node_id = node['node_id']
            color = node_color_map.get(node_id, (128, 128, 128))
            
            x1, y1, x2, y2 = map(int, node['bbox'])
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            label_text = node_id
            text_size, _ = cv2.getTextSize(label_text, font, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - text_size[1] - 8), (x1 + text_size[0] + 4, y1), color, -1)
            cv2.putText(img, label_text, (x1 + 2, y1 - 5), font, 0.6, (255, 255, 255), 1)

        if self.save_annotated_frames:
            filename = f"frame_{frame_index:04d}_clustered.jpg"
            save_path = os.path.join(self.annotated_frames_dir, filename)
        else:
            save_path = image_path.replace(".png", "_temp_spatial.jpg")
            
        cv2.imwrite(save_path, img)
        return save_path
    
    def _get_image_path(self, frame_index: int) -> Tuple[str, str]:
        traj_cfg = self.config['trajectory_data']
        file_id = traj_cfg['filename_start_index'] + frame_index
        image_filename = f"{traj_cfg['filename_prefix']}{file_id:0{traj_cfg['filename_padding']}}.png"
        return os.path.join(traj_cfg['base_dir'], traj_cfg['rgb_subdir'], image_filename), os.path.join(traj_cfg['base_dir'], traj_cfg['depth_subdir'], image_filename)
    
    def _get_pose_path(self, frame_index: int) -> str:
        """Resolve the pose-file path for a frame index."""
        traj_cfg = self.config['trajectory_data']
        file_id = traj_cfg['filename_start_index'] + frame_index
        image_filename = f"{traj_cfg['filename_prefix']}{file_id:0{traj_cfg['filename_padding']}}.txt"
        return os.path.join(traj_cfg['base_dir'], traj_cfg['pose_subdir'], image_filename)

    def _run_segmentation(self, frame_index: int, rgb_path: str, depth_path: str) -> str:
        """Run SAM3 and record the elapsed time."""
        image_pil = Image.open(rgb_path).convert("RGB")
        
        prompts_raw = self.config["predefined_object_classes"]
        if isinstance(prompts_raw, str):
            prompts = [p.strip() for p in prompts_raw.split('.') if p.strip()]
        else:
            prompts = prompts_raw

        if self.debug_mode:
            frame_output_dir = os.path.join(self.sam3_output_root, f"frame_{frame_index:04d}")
        else:
            frame_output_dir = os.path.join(self.run_dir, "objects", f"frame_{frame_index:04d}")
        
        os.makedirs(frame_output_dir, exist_ok=True)

        print(f"  [SAM3] Frame {frame_index} inference...", flush=True)
        sam_start_time = time.time()

        json_output_path = self.sam3_client.predict(
            image_pil=image_pil, 
            text_prompts=prompts, 
            using_predefined_classes = self.config['use_predefined_object_classes'],
            output_dir=frame_output_dir, 
            file_id=f"frame_{frame_index:04d}"
        )
        
        sam_duration = time.time() - sam_start_time
        print(f"  [SAM3] Inference Complete. Duration: {sam_duration:.4f}s. Output: {os.path.basename(frame_output_dir)}", flush=True)

        if not json_output_path or not os.path.exists(json_output_path):
            print("  [SAM3] No objects detected.", flush=True)
            empty_json_path = os.path.join(self.run_dir, f"frame_{frame_index:04d}_empty.json")
            with open(empty_json_path, 'w') as f:
                json.dump([], f)
            return empty_json_path
            
        return json_output_path

    def _is_keyframe(self, current_frame_pil: Image, frame_index: int) -> Tuple[bool, np.ndarray]:
        current_feature = extract_feature(current_frame_pil, self.dinov3_model, self.device)
        if self.last_kf_feature is None: return True, current_feature
        sim = cosine_similarity(current_feature.reshape(1, -1), self.last_kf_feature.reshape(1, -1))[0, 0]
        print(f"  - [Keyframe] Sim: {sim:.4f} (Thresh: {self.kf_threshold}) -> {'YES' if sim < self.kf_threshold else 'NO'}")
        return sim < self.kf_threshold, current_feature

    def _visualize_reid_match(self, old_image_path: str, old_bbox: List[int], 
                              new_image_path: str, new_bbox: List[int], 
                              old_node_id: str, new_obj_id: int, score: float, frame_index: int):
        if not self.enable_reid_viz or not self.reid_viz_dir: return
        try:
            img_old, img_new = cv2.imread(old_image_path), cv2.imread(new_image_path)
            if img_old is None or img_new is None: return
            
            if old_bbox:
                x1, y1, x2, y2 = map(int, old_bbox)
                cv2.rectangle(img_old, (x1, y1), (x2, y2), (0, 255, 0), 3) 
                cv2.putText(img_old, f"KB: {old_node_id}", (x1, max(y1-10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if new_bbox:
                x1, y1, x2, y2 = map(int, new_bbox)
                cv2.rectangle(img_new, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(img_new, f"New: {new_obj_id}", (x1, max(y1-10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            h_new, w_new = img_new.shape[:2]
            h_old, w_old = img_old.shape[:2]
            if h_old == 0: return
            scale = h_new / h_old
            img_old_resized = cv2.resize(img_old, (int(w_old * scale), h_new))
            viz_img = np.hstack((img_old_resized, img_new))
            
            info_h, info_bg = 80, np.ones((80, viz_img.shape[1], 3), dtype=np.uint8) * 255
            cv2.putText(info_bg, f"Frame {frame_index} Re-ID | Sim: {score:.4f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            cv2.putText(info_bg, f"Old: {old_node_id} | New: {new_obj_id}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
            final_img = np.vstack((info_bg, viz_img))
            
            cv2.imwrite(os.path.join(self.reid_viz_dir, f"frame_{frame_index:04d}_reid_{new_obj_id}_to_{old_node_id}.jpg"), final_img)
        except Exception as e: print(f"    [Viz Error] {e}")

    def visualize_spatial_graph(self):
        if not hasattr(self, 'spatial_graph') or self.spatial_graph.number_of_nodes() == 0: return
        plt.figure(figsize=(26, 26))
        pos = nx.spring_layout(self.spatial_graph, weight='weight', k=0.6, seed=42, iterations=100)
        nx.draw_networkx_edges(self.spatial_graph, pos, width=[d.get('weight', 0.5) * 3.0 for u, v, d in self.spatial_graph.edges(data=True)], edge_color='#888888', alpha=0.5, arrows=True, arrowsize=15)
        edge_labels = {(u, v): d.get('relation', '') for u, v, d in self.spatial_graph.edges(data=True) if d.get('relation', '') and d.get('relation') != 'connected'}
        nx.draw_networkx_edge_labels(self.spatial_graph, pos, edge_labels=edge_labels, font_size=7, font_color='darkblue')
        
        for node_id in self.spatial_graph.nodes():
            if node_id not in pos: continue
            nx.draw_networkx_nodes(self.spatial_graph, pos, nodelist=[node_id], node_color='red', node_size=50) 
            plt.text(pos[node_id][0], pos[node_id][1]+0.02, node_id, fontsize=8, ha='center')

        plt.axis('off')
        plt.savefig(os.path.join(self.run_dir, "semantic_topology_web.jpg"), bbox_inches='tight', dpi=150)
        plt.close()
        
    def _filter_detections_advanced(self, raw_detections: List[Dict], depth_img: np.ndarray, 
                                    img_w: int, img_h: int) -> List[Dict]:
        """Apply the combined acceptance filters."""
        if not raw_detections:
            return []

        BORDER_MARGIN = 5
        DEPTH_FLAT_THRESH = 0.05
        CONTAINMENT_RATIO = 0.90
        DEPTH_SCALE = self.intrinsics.get('depth_scale', 1000.0)

        valid_candidates = []
        for obj in raw_detections:
            x1, y1, x2, y2 = map(int, obj['bounding_box_xyxy'])
            
            box_w, box_h = x2 - x1, y2 - y1
            is_touching_edge = (x1 < BORDER_MARGIN) or (y1 < BORDER_MARGIN) or \
                               (x2 > img_w - BORDER_MARGIN) or (y2 > img_h - BORDER_MARGIN)
            
            is_huge = (box_w > img_w * 0.85) and (box_h > img_h * 0.85)
            
            if is_touching_edge and not is_huge and self.config['filter_edge_detections']:
                continue
            
            obj['_area'] = box_w * box_h
            obj['_bbox'] = (x1, y1, x2, y2)
            valid_candidates.append(obj)

        if depth_img is None:
            return valid_candidates

        valid_candidates.sort(key=lambda x: x['_area'], reverse=True)
        
        keep_indices = set(range(len(valid_candidates)))
        
        for i in range(len(valid_candidates)):
            if i not in keep_indices: continue
            parent = valid_candidates[i]
            px1, py1, px2, py2 = parent['_bbox']
            
            for j in range(len(valid_candidates)):
                if i == j: continue
                if j not in keep_indices: continue
                
                child = valid_candidates[j]
                
                if child['_area'] > parent['_area']: continue

                cx1, cy1, cx2, cy2 = child['_bbox']

                ix1 = max(px1, cx1); iy1 = max(py1, cy1)
                ix2 = min(px2, cx2); iy2 = min(py2, cy2)
                inter_w = max(0, ix2 - ix1)
                inter_h = max(0, iy2 - iy1)
                inter_area = inter_w * inter_h
                
                if child['_area'] > 0 and (inter_area / child['_area']) > CONTAINMENT_RATIO:
                    
                    
                    child_depth_crop = depth_img[cy1:cy2, cx1:cx2]
                    valid_c_d = child_depth_crop[child_depth_crop > 0]
                    if valid_c_d.size == 0: continue
                    median_child_depth = np.median(valid_c_d)
                    
                    margin = 15
                    bx1 = max(px1, cx1 - margin); by1 = max(py1, cy1 - margin)
                    bx2 = min(px2, cx2 + margin); by2 = min(py2, cy2 + margin)
                    
                    surround_crop = depth_img[by1:by2, bx1:bx2]
                    
                    mask = np.ones(surround_crop.shape, dtype=bool)
                    rel_x1 = cx1 - bx1; rel_y1 = cy1 - by1
                    rel_x2 = cx2 - bx1; rel_y2 = cy2 - by1
                    rel_x1, rel_y1 = max(0, rel_x1), max(0, rel_y1)
                    rel_x2, rel_y2 = min(surround_crop.shape[1], rel_x2), min(surround_crop.shape[0], rel_y2)
                    
                    mask[rel_y1:rel_y2, rel_x1:rel_x2] = False
                    
                    valid_surround_d = surround_crop[mask & (surround_crop > 0)]
                    
                    if valid_surround_d.size == 0:
                        continue
                        
                    median_surround_depth = np.median(valid_surround_d)
                    
                    diff_m = abs(median_child_depth - median_surround_depth) / DEPTH_SCALE
                    
                    if diff_m < DEPTH_FLAT_THRESH:
                        print(f"  [Filter] Removed '{child['label']}' inside '{parent['label']}' (Depth Diff: {diff_m:.3f}m < {DEPTH_FLAT_THRESH}m)")
                        keep_indices.remove(j)
                    else:
                        # print(f"  [Keep] Kept '{child['label']}' on '{parent['label']}' (Depth Diff: {diff_m:.3f}m)")
                        pass

        final_results = []
        for k in sorted(list(keep_indices)):
            obj = valid_candidates[k]
            if '_area' in obj: del obj['_area']
            if '_bbox' in obj: del obj['_bbox']
            final_results.append(obj)
            
        return final_results
    
    def _calculate_view_score(self, bbox, img_w, img_h):
        if img_w <= 0 or img_h <= 0: return 1.0
        bx = (bbox[0] + bbox[2]) / 2.0
        by = (bbox[1] + bbox[3]) / 2.0
        cx, cy = img_w / 2.0, img_h / 2.0
        
        max_dist = np.sqrt(cx**2 + cy**2)
        curr_dist = np.sqrt((bx - cx)**2 + (by - cy)**2)
        
        return curr_dist / max_dist
    
    def _decode_rle(self, rle):
        """Decode an RLE dictionary into a binary mask."""
        h, w = rle['size']
        counts = rle['counts']
        
        mask_flat = np.zeros(h * w, dtype=np.uint8)
        
        current_idx = 0
        current_val = 0 
        
        for count in counts:
            mask_flat[current_idx : current_idx + count] = current_val
            current_idx += count
            current_val = 1 - current_val
            
        return mask_flat.reshape((h, w)) > 0
    
    def _process_frame_results(self, frame_index: int, segmentation_json_path: str, current_rgb_path: str, current_depth_path: str, current_pose: np.ndarray):
        with open(segmentation_json_path, 'r') as f: 
            raw_results = json.load(f)
        
        depth_img = cv2.imread(current_depth_path, cv2.IMREAD_UNCHANGED)
        
        if depth_img is not None:
            h_img, w_img = depth_img.shape[:2]
        else:
            temp_img = Image.open(current_rgb_path)
            w_img, h_img = temp_img.size

        # ==========================================
        # ==========================================
        print(f"  [Filter] Raw objects: {len(raw_results)}", end="")
        
        filtered_results = self._filter_detections_advanced(raw_results, depth_img, w_img, h_img)
        
        print(f" -> Filtered: {len(filtered_results)}")
        # ==========================================
        seg_dir = os.path.dirname(segmentation_json_path)
        depth_img = cv2.imread(current_depth_path, cv2.IMREAD_UNCHANGED)
        
        re_id_config = self.config.get('re_identification', {})
        feat_start = time.time()
        current_objects_data = [] 
        for obj in filtered_results:
            obj_id = obj['object_id']
            try:
                crop_path = os.path.join(seg_dir, obj['cropped_image_path'])
                feat = extract_feature(Image.open(crop_path).convert("RGB"), self.dinov3_model, self.device)
                
                global_pos_3d = None
                
                if depth_img is not None and current_pose is not None:
                    x1, y1, x2, y2 = map(int, obj['bounding_box_xyxy'])
                    h, w = depth_img.shape[:2]
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    depth_crop = depth_img[y1:y2, x1:x2]
                    
                    mask_crop = None
                    if 'segmentation' in obj and obj['segmentation']:
                        try:
                            full_mask = self._decode_rle(obj['segmentation'])
                            mask_crop = full_mask[y1:y2, x1:x2]
                        except Exception as e:
                            print(f"Warning: Mask decode failed for {obj_id}: {e}")
                            mask_crop = None

                    valid_depths = np.array([])
                    cx, cy = (x1+x2)//2, (y1+y2)//2

                    if depth_crop.size > 0:
                        if mask_crop is not None and mask_crop.shape == depth_crop.shape:
                            valid_depths = depth_crop[(depth_crop > 0) & mask_crop]
                            
                            if valid_depths.size > 0:
                                ys, xs = np.where(mask_crop)
                                if len(ys) > 0:
                                    cy = y1 + int(np.mean(ys))
                                    cx = x1 + int(np.mean(xs))
                        else:
                            valid_depths = depth_crop[depth_crop > 0]

                        if valid_depths.size > 0:
                            median_d = np.median(valid_depths) / self.intrinsics['depth_scale']
                            
                            # if 0.1 < median_d < 10.0: 
                            global_pos_3d = pixel_to_world(cx, cy, median_d, current_pose, self.intrinsics)
       
                obj_floor_id = self.floor_tracker.current_floor_id
                if global_pos_3d:
                    obj_floor_id = self.floor_tracker.get_floor_id_by_object_y(global_pos_3d[1])

                current_objects_data.append({
                    "raw_obj": obj,
                    "feature": feat,
                    "pos_3d": global_pos_3d,
                    "floor_id": obj_floor_id,
                    "label": obj.get('label', 'object').lower().strip(),
                    "match_id": None,
                    "match_kb_idx": -1
                })
            except Exception as e:
                print(f"Error processing object {obj_id}: {e}")

        feat_duration = time.time() - feat_start
        print(f"  [Feature] Extracted features for {len(current_objects_data)} objects in {feat_duration:.4f}s")
        kb_candidates = []
        kb_indices_map = []
        kb_labels = []
        for idx, item in enumerate(self.memory_kb):
            # New nodes are initialized below; this also migrates V7 preload
            # artifacts before their labels participate in the V7 Re-ID gates.
            ensure_label_score_state(item)
            # if item.get('floor_id') == self.floor_tracker.current_floor_id:
            kb_candidates.append(item)
            kb_indices_map.append(idx)
            kb_labels.append(item.get('label', 'object').lower())
        
        kb_features = np.array([item['dinov3_feature'] for item in kb_candidates]) if kb_candidates else np.empty((0, 0))
        used_kb_local_indices = set()
        unmatched_curr_indices = list(range(len(current_objects_data)))
        GENERIC_LABELS = {'object', 'item', 'thing', 'stuff', 'unknown', 'entity'}

        # =================================================
        # =================================================
        VISUAL_SIM_THRESH = re_id_config.get('visual_similarity_threshold', 0.8)
        VISUAL_DIST_THRESH = re_id_config.get('spatial_distance_threshold', 1.5)
        FORCE_MERGE_DIST = re_id_config.get('force_merge_distance_threshold', 1.0)

        potential_matches = []
        
        reid_start = time.time()

        if len(kb_candidates) > 0 and len(unmatched_curr_indices) > 0:
            curr_feats_list = [current_objects_data[i]['feature'] for i in unmatched_curr_indices]
            curr_feats_matrix = np.stack(curr_feats_list)
            
            sim_matrix = cosine_similarity(curr_feats_matrix, kb_features)

            for i, real_curr_idx in enumerate(unmatched_curr_indices):
                curr_obj = current_objects_data[real_curr_idx]
                if curr_obj['pos_3d'] is None: continue
                
                curr_pos = np.array(curr_obj['pos_3d'])
                
                for k_local_idx, kb_item in enumerate(kb_candidates):
                    score = sim_matrix[i][k_local_idx]
                    
                    if score > VISUAL_SIM_THRESH:
                        k_pos = kb_item.get('global_pos_3d')
                        if k_pos:
                            dist = np.linalg.norm(curr_pos - np.array(k_pos))
                            if dist < VISUAL_DIST_THRESH:
                                potential_matches.append({
                                    'score': score,
                                    'dist': dist,
                                    'curr_idx': real_curr_idx,
                                    'kb_idx': k_local_idx,
                                    'kb_label': kb_item.get('label', ''),
                                    'curr_label': curr_obj['label']
                                })

        # sort by score (desc), then distance (asc)
        potential_matches.sort(key=lambda x: (-x['score'], x['dist']))

        matched_curr_set = set()
        
        for match in potential_matches:
            c_idx = match['curr_idx']
            k_idx = match['kb_idx']
            
            if c_idx in matched_curr_set or k_idx in used_kb_local_indices:
                continue
            kb_node_id = kb_candidates[k_idx]['node_id']
            kb_pos = kb_candidates[k_idx]['global_pos_3d']
            curr_pos = current_objects_data[c_idx]['pos_3d']

            raw_obj_id = current_objects_data[c_idx]['raw_obj']['object_id']

            k_pos_str = f"[{kb_pos[0]:.1f}, {kb_pos[1]:.1f}, {kb_pos[2]:.1f}]" if kb_pos else "None"
            c_pos_str = f"[{curr_pos[0]:.1f}, {curr_pos[1]:.1f}, {curr_pos[2]:.1f}]" if curr_pos else "None"

            print(f"  [Stage 1] MATCH: {raw_obj_id} ('{match['curr_label']}') -> Node '{kb_node_id}'")
            print(f"            Sim: {match['score']:.3f} | Dist: {match['dist']:.2f}m")
            print(f"            Pos: {c_pos_str} -> {k_pos_str}")
            # -------------------------------------------------------------------------
            
            current_objects_data[c_idx]['match_kb_idx'] = kb_indices_map[k_idx]
            current_objects_data[c_idx]['match_id'] = kb_candidates[k_idx]['node_id']
            
            used_kb_local_indices.add(k_idx)
            matched_curr_set.add(c_idx)
            if c_idx in unmatched_curr_indices:
                unmatched_curr_indices.remove(c_idx)
        
        reid1_duration = time.time() - reid_start
        print(f"  [Re-ID Stage 1] Completed in {reid1_duration:.4f}s. Matches found: {len(matched_curr_set)}")
        # =================================================
        # =================================================


        SEMANTIC_SIM_THRESH = re_id_config.get('semantic_similarity_threshold', 0.6)
        GENERIC_VISUAL_THRESH = re_id_config.get('generic_object_visual_threshold', 0.65)

        if len(kb_candidates) > 0 and len(unmatched_curr_indices) > 0:
            
            kb_label_embeddings = None
            if len(unmatched_curr_indices) > 0:
                 kb_label_embeddings = self.text_encoder.encode(kb_labels, convert_to_tensor=True, show_progress_bar=False)

            for curr_idx in list(unmatched_curr_indices):
                curr_obj = current_objects_data[curr_idx]
                if curr_obj['pos_3d'] is None: continue
                
                raw_obj_id = curr_obj['raw_obj']['object_id']

                curr_label = curr_obj['label']
                base_label = ''.join([i for i in curr_label if not i.isdigit()]).strip('_').strip()
                is_generic = base_label in GENERIC_LABELS
                
                candidates_in_range = []
                
                for k_idx, kb_item in enumerate(kb_candidates):
                    if k_idx in used_kb_local_indices: continue
                    
                    k_pos = kb_item.get('global_pos_3d')
                    if k_pos:
                        dist = np.linalg.norm(np.array(curr_obj['pos_3d']) - np.array(k_pos))
                        if dist < FORCE_MERGE_DIST:
                            candidates_in_range.append({
                                'idx': k_idx,
                                'dist': dist,
                                'kb_label': kb_labels[k_idx]
                            })
                
                if not candidates_in_range:
                    print(f"  [Stage 2] Fail: {raw_obj_id} ('{curr_label}') no neighbor within {FORCE_MERGE_DIST}m")
                    continue

                best_match_idx = -1
                match_reason = ""

                
                if is_generic:
                    
                    curr_vis_feat = curr_obj['feature'].reshape(1, -1)
                    cand_indices = [c['idx'] for c in candidates_in_range]
                    cand_vis_feats = kb_features[cand_indices]
                    
                    vis_sims = cosine_similarity(curr_vis_feat, cand_vis_feats)[0]
                    
                    best_local_idx = np.argmax(vis_sims)
                    best_score = vis_sims[best_local_idx]
                    
                    if best_score > GENERIC_VISUAL_THRESH:
                        best_match_idx = candidates_in_range[best_local_idx]['idx']
                        match_reason = f"Generic Visual Sim {best_score:.2f} (> {GENERIC_VISUAL_THRESH})"
                    else:
                        fail_kb_idx = candidates_in_range[best_local_idx]['idx']
                        fail_node = kb_candidates[fail_kb_idx]
                        print(f"  [Stage 2] Fail: {raw_obj_id} (Generic) best visual match {best_score:.2f} < {GENERIC_VISUAL_THRESH}")
                        print(f"            -> Closest: '{fail_node.get('label')}' (Node: {fail_node['node_id']})")
                        continue

                else:
                    
                    exact_matches = []
                    other_candidates = []
                    
                    for c in candidates_in_range:
                        if curr_label in c['kb_label'] or c['kb_label'] in curr_label:
                            exact_matches.append(c)
                        else:
                            other_candidates.append(c)
                    
                    if exact_matches:
                        exact_matches.sort(key=lambda x: x['dist'])
                        best_match = exact_matches[0]
                        best_match_idx = best_match['idx']
                        match_reason = f"Exact Label Match (Dist: {best_match['dist']:.2f}m)"
                    
                    elif other_candidates:
                        curr_text_emb = self.text_encoder.encode(curr_label, convert_to_tensor=True, show_progress_bar=False)
                        cand_indices = [c['idx'] for c in other_candidates]
                        target_kb_embs = kb_label_embeddings[cand_indices]
                        
                        semantic_scores = util.cos_sim(curr_text_emb, target_kb_embs)[0].cpu().numpy()
                        
                        valid_semantic_matches = []
                        for i, score in enumerate(semantic_scores):
                            if score > SEMANTIC_SIM_THRESH:
                                c = other_candidates[i]
                                c['sem_score'] = score
                                valid_semantic_matches.append(c)
                        
                        if valid_semantic_matches:
                            valid_semantic_matches.sort(key=lambda x: (-x['sem_score'], x['dist']))
                            best_match = valid_semantic_matches[0]
                            best_match_idx = best_match['idx']
                            match_reason = f"Semantic Match (Score: {best_match['sem_score']:.2f}, Dist: {best_match['dist']:.2f}m)"
                        else:
                            if len(semantic_scores) > 0:
                                best_fail_idx = np.argmax(semantic_scores)
                                best_fail_score = semantic_scores[best_fail_idx]
                                fail_cand = other_candidates[best_fail_idx]
                                fail_node = kb_candidates[fail_cand['idx']]
                                
                                print(f"  [Stage 2] Fail: {raw_obj_id} ('{curr_label}') low semantic score (< {SEMANTIC_SIM_THRESH})")
                                print(f"            -> Closest: '{fail_node.get('label')}' (Node: {fail_node['node_id']}) Score: {best_fail_score:.2f}")
                            else:
                                print(f"  [Stage 2] Fail: {raw_obj_id} ('{curr_label}') no semantic candidates.")
                            continue
                    else:
                        continue

                if best_match_idx != -1:
                    target_node = kb_candidates[best_match_idx]
                    target_id = target_node['node_id']
                    target_pos = target_node.get('global_pos_3d')
                    curr_pos = curr_obj['pos_3d']

                    t_pos_str = f"[{target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f}]" if target_pos else "None"
                    c_pos_str = f"[{curr_pos[0]:.1f}, {curr_pos[1]:.1f}, {curr_pos[2]:.1f}]" if curr_pos else "None"

                    print(f"  [Stage 2] MATCH ({match_reason}): {raw_obj_id} ('{curr_label}') -> Node '{target_id}'")
                    print(f"            Pos: {c_pos_str} -> {t_pos_str}")
                    # -------------------------------------------------------------------------
                    
                    current_objects_data[curr_idx]['match_kb_idx'] = kb_indices_map[best_match_idx]
                    current_objects_data[curr_idx]['match_id'] = kb_candidates[best_match_idx]['node_id']
                    used_kb_local_indices.add(best_match_idx)
                    unmatched_curr_indices.remove(curr_idx)

        reid2_duration = time.time() - reid1_duration - reid_start
        print(f"  [Re-ID Stage 2] Completed in {reid2_duration:.4f}s. New nodes created: {len(unmatched_curr_indices)}")
        
        current_frame_nodes = []
        for idx, curr_obj in enumerate(current_objects_data):
            clean_label = "".join([c if c.isalnum() else "_" for c in curr_obj['label']]).strip('_').lower()
            raw_obj_id = curr_obj['raw_obj']['object_id']

            current_view_score = self._calculate_view_score(curr_obj['raw_obj']['bounding_box_xyxy'], w_img, h_img)

            if curr_obj['match_kb_idx'] != -1:

                merge_start = time.time()

                real_kb_idx = curr_obj['match_kb_idx']
                target_node = self.memory_kb[real_kb_idx]
                final_node_id = target_node['node_id']
                
                if curr_obj['pos_3d']:
                    old_p = target_node['global_pos_3d']
                    if old_p:
                        self.memory_kb[real_kb_idx]['global_pos_3d'] = [(0.7*o + 0.3*n) for o, n in zip(old_p, curr_obj['pos_3d'])]
                    else:
                        self.memory_kb[real_kb_idx]['global_pos_3d'] = curr_obj['pos_3d']
                
                old_f = np.array(target_node['dinov3_feature'])
                self.memory_kb[real_kb_idx]['dinov3_feature'] = (0.8 * curr_obj['feature'] + 0.2 * old_f).tolist()
                
                # 3. V8 label update. Re-ID above remains exactly V7: only
                # after V7 has selected this instance do we compare the SAM3
                # per-detection score. Keep one public label, never append.
                ensure_label_score_state(target_node)
                label_decision = record_label_score(
                    target_node,
                    clean_label,
                    curr_obj['raw_obj'].get('score'),
                    frame_index=frame_index,
                )
                if label_decision['changed']:
                    print(
                        "  [V8 Label] "
                        f"{final_node_id}: "
                        f"{label_decision['previous_label']}"
                        f"({label_decision['previous_score']:.4f}) -> "
                        f"{label_decision['final_label']}"
                        f"({label_decision['final_score']:.4f})"
                    )
                    if self.spatial_graph.has_node(final_node_id):
                        self.spatial_graph.nodes[final_node_id][
                            'semantic_tag'
                        ] = label_decision['final_label']
                        self.spatial_graph.nodes[final_node_id][
                            'label'
                        ] = label_decision['final_label']
                
                old_view_score = target_node.get('view_score', 1.0)
                
                if current_view_score < old_view_score:
                    # print(f"  [Update View] Node {final_node_id} updated with better view (Score: {old_view_score:.2f} -> {current_view_score:.2f})")
                    self.memory_kb[real_kb_idx]['bbox'] = curr_obj['raw_obj']['bounding_box_xyxy']
                    self.memory_kb[real_kb_idx]['full_image_path'] = current_rgb_path
                    self.memory_kb[real_kb_idx]['cropped_image_path'] = curr_obj['raw_obj']['cropped_image_path']
                    self.memory_kb[real_kb_idx]['view_score'] = current_view_score
                print(f"merge time :{time.time()-merge_start:.4f}s")
            else:
                # NEW
                new_start = time.time()

                target_room = self.current_room_id if (curr_obj['floor_id'] == self.floor_tracker.current_floor_id and self.current_room_id != 0) else None
                final_node_id = self._create_leaf_node(
                    curr_obj['raw_obj'], curr_obj['feature'], frame_index, seg_dir, 
                    current_rgb_path, clean_label, curr_obj['pos_3d']
                )
                self.memory_kb[-1]['floor_id'] = curr_obj['floor_id']
                self.memory_kb[-1]['parent_room'] = target_room
                self.memory_kb[-1]['view_score'] = current_view_score 
                initialize_label_score(
                    self.memory_kb[-1],
                    clean_label,
                    curr_obj['raw_obj'].get('score'),
                    frame_index=frame_index,
                )
                
                current_objects_data[idx]['match_id'] = final_node_id

                curr_pos = curr_obj['pos_3d']
                c_pos_str = f"[{curr_pos[0]:.1f}, {curr_pos[1]:.1f}, {curr_pos[2]:.1f}]" if curr_pos else "None"
                
                print(f"  [Action] NEW NODE: {raw_obj_id} ('{clean_label}') -> ID '{final_node_id}'")
                print(f"           At Pos: {c_pos_str}")

                print(f"           Created in {time.time() - new_start:.4f}s")

            if final_node_id:
                current_frame_nodes.append({"node_id": final_node_id, "bbox": curr_obj['raw_obj']['bounding_box_xyxy']})
                if not self.spatial_graph.has_node(final_node_id): 
                    self.spatial_graph.add_node(final_node_id)
                kb_item = next(
                    item for item in self.memory_kb
                    if item['node_id'] == final_node_id
                )
                self._record_object_episode_provenance(
                    kb_item,
                    frame_index=frame_index,
                    current_rgb_path=current_rgb_path,
                    current_pose=current_pose,
                    bbox=curr_obj['raw_obj']['bounding_box_xyxy'],
                    object_position=curr_obj.get('pos_3d'),
                    image_width=w_img,
                    image_height=h_img,
                )

        # =========================================================
        # =========================================================
        
        valid_nodes_for_spatial = []
        node_positions_2d = {}
        node_positions_3d = {}
        
        for node_info in current_frame_nodes:
            nid = node_info['node_id']
            kb_item = next((item for item in self.memory_kb if item['node_id'] == nid), None)
            if kb_item and kb_item.get('global_pos_3d'):
                if kb_item.get('floor_id') == self.floor_tracker.current_floor_id:
                    pos = kb_item['global_pos_3d']
                    node_positions_2d[nid] = np.array([pos[0], pos[2]])
                    node_positions_3d[nid] = pos
                    valid_nodes_for_spatial.append(node_info)
        
        if len(valid_nodes_for_spatial) > 1:
            print(f"  [Topology] Clustering {len(valid_nodes_for_spatial)} nodes (Current Floor)...", end='', flush=True)
            
            cluster_graph = nx.Graph()
            cluster_graph.add_nodes_from([n['node_id'] for n in valid_nodes_for_spatial])
            node_ids = list(node_positions_2d.keys())
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    id_i, id_j = node_ids[i], node_ids[j]
                    dist = np.linalg.norm(node_positions_2d[id_i] - node_positions_2d[id_j])
                    if dist <= self.spatial_cluster_thresh:
                        cluster_graph.add_edge(id_i, id_j)
            clusters = [list(c) for c in nx.connected_components(cluster_graph)]
            print(f" Found {len(clusters)} clusters.", flush=True)
            
            spatial_method = self.config.get('spatial_reasoning', {}).get('method', 'vlm')

            annotated_img_path = current_rgb_path
            if spatial_method == 'vlm':
                 annotated_img_path = self._annotate_image_with_clusters(
                    current_rgb_path, current_frame_nodes, clusters, frame_index
                )
            

           


            topo_start = time.time()
            for cluster_nodes in clusters:
                if len(cluster_nodes) < 2: continue
                
                print(f"    -> Analyzing cluster ({spatial_method}): {cluster_nodes}...", end='', flush=True)
                existing_relations = []
                subgraph = self.spatial_graph.subgraph(cluster_nodes)
                for u, v, data in subgraph.edges(data=True):
                    existing_relations.append(f"{u}->{v}({data.get('relation', 'connected')})")
                
                if hasattr(self, 'llm'):
                    edge_gen_start = time.time()

                    spatial_quads = self.llm.detect_spatial_relationships(
                        image_path=annotated_img_path, 
                        object_list=cluster_nodes, 
                        existing_relations=existing_relations,
                        object_positions=node_positions_3d, 
                        method=spatial_method
                    )
                    
                    edge_gen_cost = time.time() - edge_gen_start
                    
                    print(f" [LLM/Algo Time: {edge_gen_cost:.4f}s] ", end='', flush=True)

                    for i, item in enumerate(spatial_quads):
                        u, v, rel, w = None, None, None, 0.0
                        if isinstance(item, dict):
                            u, v, rel, w = item.get('Source'), item.get('Target'), item.get('Relation'), item.get('Weight')
                            if not u: u, v, rel, w = item.get('source'), item.get('target'), item.get('relation'), item.get('weight')
                        elif isinstance(item, list) and len(item) >= 3:
                            u, v, rel = item[0], item[1], item[2]
                            w = item[3] if len(item) > 3 else 0.5

                        if u in cluster_nodes and v in cluster_nodes:
                            if self.spatial_graph.has_node(u) and self.spatial_graph.has_node(v):
                                try:
                                    weight_val = float(w)
                                    threshold = 0.6 if spatial_method == 'vlm' else 0.4 
                                    if weight_val > threshold:
                                        self.spatial_graph.add_edge(u, v, relation=rel, weight=weight_val)
                                        
                                        print(f"\n      + Edge [{u}->{v} ({rel})] added. (Batch Latency: {edge_gen_cost:.4f}s)", end='')
                                        
                                except (ValueError, TypeError): pass
                print(" Done.")

            if "_temp_spatial" in annotated_img_path and os.path.exists(annotated_img_path): 
                os.remove(annotated_img_path)
            
            total_topo_time = time.time() - topo_start
            print(f"  [Topology] Analysis finished ({total_topo_time:.2f}s).", flush=True)
            
            # topo_start = time.time()
            # for cluster_nodes in clusters:
            #     if len(cluster_nodes) < 2: continue
                
            #     print(f"    -> Analyzing cluster ({spatial_method}): {cluster_nodes}...", end='', flush=True)
            #     existing_relations = []
            #     subgraph = self.spatial_graph.subgraph(cluster_nodes)
            #     for u, v, data in subgraph.edges(data=True):
            #         existing_relations.append(f"{u}->{v}({data.get('relation', 'connected')})")
                
            #     if hasattr(self, 'llm'):
            #         spatial_quads = self.llm.detect_spatial_relationships(
            #             image_path=annotated_img_path, 
            #             object_list=cluster_nodes, 
            #             existing_relations=existing_relations,
            #             method=spatial_method
            #         )
                    
            #         for item in spatial_quads:
            #             u, v, rel, w = None, None, None, 0.0
            #             if isinstance(item, dict):
            #                 u, v, rel, w = item.get('Source'), item.get('Target'), item.get('Relation'), item.get('Weight')
            #                 if not u: u, v, rel, w = item.get('source'), item.get('target'), item.get('relation'), item.get('weight')
            #             elif isinstance(item, list) and len(item) >= 3:
            #                 u, v, rel = item[0], item[1], item[2]
            #                 w = item[3] if len(item) > 3 else 0.5

            #             if u in cluster_nodes and v in cluster_nodes:
            #                 if self.spatial_graph.has_node(u) and self.spatial_graph.has_node(v):
            #                     try:
            #                         weight_val = float(w)
            #                         threshold = 0.6 if spatial_method == 'vlm' else 0.4 
            #                         if weight_val > threshold:
            #                             self.spatial_graph.add_edge(u, v, relation=rel, weight=weight_val)
            #                     except (ValueError, TypeError): pass
            #     print(" Done.")

            # if "_temp_spatial" in annotated_img_path and os.path.exists(annotated_img_path): 
            #     os.remove(annotated_img_path)
            # print(f"  [Topology] Analysis finished ({time.time() - topo_start:.2f}s).", flush=True)

        self._update_and_save_bev_map(frame_index, current_pose)
        # ================= DEBUG VISUALIZATION =================
        
        debug_img = cv2.imread(current_rgb_path) if self.debug_mode else None
        if debug_img is not None:
            for idx, obj_data in enumerate(current_objects_data):
                x1, y1, x2, y2 = map(int, obj_data['raw_obj']['bounding_box_xyxy'])
                
                if obj_data['match_kb_idx'] != -1:
                    color = (0, 255, 0)
                    text = f"{obj_data['match_id']} (Old)"
                else:
                    color = (0, 0, 255)
                    text = f"{obj_data['match_id']} (New)" 

                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(debug_img, (x1, y1-15), (x1+tw, y1), color, -1)
                cv2.putText(debug_img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
            debug_dir = os.path.join(os.path.dirname(segmentation_json_path), "debug_merge")
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"debug_{frame_index:05d}.jpg"), debug_img)
        # =======================================================
    
    
    def _create_leaf_node(self, obj_data: Dict, feature: np.ndarray, frame_index: int, seg_dir: str, current_rgb_path: str, clean_label: str, global_pos_3d: List[float] = None) -> str:
        """Accept an optional global 3D position."""
        count = self.label_counters.get(clean_label, 0)
        node_id = f"{clean_label}_{count}"
        self.label_counters[clean_label] = count + 1
        
        crop_path = os.path.join(seg_dir, obj_data['cropped_image_path'])
        bbox = obj_data['bounding_box_xyxy']
        
        llm_desc = "Auto-generated description unavailable"
        if hasattr(self, 'llm'):
            llm_desc = self.llm.generate_leaf_node_semantics(crop_path, clean_label, f"bbox: {bbox}, frame: {frame_index}")

        leaf_node = {
            "node_id": node_id, 
            "node_level": 3, 
            "semantic_tag": clean_label, 
            "llm_desc": llm_desc, 
            "bbox": bbox, 
            "global_pos_3d": global_pos_3d, # Save 3D pos
            "timestamp": datetime.datetime.now().isoformat()
        }
        with open(self.nodes_path, 'a', encoding='utf-8') as f: f.write(json.dumps(leaf_node, ensure_ascii=False) + "\n")

        self.memory_kb.append({
            "node_id": node_id,
            "dinov3_feature": feature.tolist(),
            "label": clean_label,
            "llm_desc": llm_desc,
            "full_image_path": current_rgb_path,
            "bbox": bbox,
            "global_pos_3d": global_pos_3d # Save 3D pos in Memory
        })
        return node_id

    # -----------------------------------------------------------
    def _update_and_save_bev_map(self, frame_index: int, current_pose: np.ndarray, return_path_only=False) -> str:
        if current_pose is None: return None

        curr_pos = current_pose[:3, 3]
        curr_floor = self.floor_tracker.current_floor_id
        
        if curr_floor not in self.floor_trajectories:
            self.floor_trajectories[curr_floor] = []
        self.floor_trajectories[curr_floor].append(curr_pos)

        # The trajectory itself is persisted in builder_state.pkl. Rendering
        # one Matplotlib image per keyframe is diagnostic-only.
        if not self.debug_mode and not return_path_only:
            return None

        floor_map_dir = os.path.join(self.map_output_dir, f"floor_{curr_floor:02d}")
        os.makedirs(floor_map_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(14, 14)) 
        ax = plt.gca()
        
        def to_plot_coords(x, z): return x, -z

        room_map = self.map_manager.room_id_map
        if room_map is not None:
            min_x, min_z = self.map_manager.min_x, self.map_manager.min_z
            res = self.map_manager.resolution
            h, w = room_map.shape
            max_x = min_x + w * res
            max_z = min_z + h * res
            
            extent = [min_x, max_x, -max_z, -min_z]
            masked_data = np.ma.masked_where(room_map == 0, room_map)
            cmap = plt.cm.get_cmap('tab20b', np.max(room_map) + 1) if np.max(room_map) > 0 else plt.cm.gray
            plt.imshow(masked_data, extent=extent, cmap=cmap, alpha=0.8, origin='upper', interpolation='nearest')

        plotted_objects = {} 
        current_floor_objs = [item for item in self.memory_kb if item.get('floor_id') == curr_floor]
        
        cx, cz = to_plot_coords(curr_pos[0], curr_pos[2])

        for item in current_floor_objs:
            pos = item.get('global_pos_3d')
            if pos:
                px, pz = to_plot_coords(pos[0], pos[2])
                plotted_objects[item['node_id']] = (px, pz)
                plt.scatter(px, pz, c='grey', s=30, alpha=0.9, edgecolors='white', linewidth=0.8, zorder=3)

        if hasattr(self, 'spatial_graph'):
            for u, v in self.spatial_graph.edges():
                if u in plotted_objects and v in plotted_objects:
                    u_pos = plotted_objects[u]
                    v_pos = plotted_objects[v]
                    plt.plot([u_pos[0], v_pos[0]], [u_pos[1], v_pos[1]], color='green', linewidth=1.5, alpha=0.5, zorder=4)

        if hasattr(self, 'global_update_history'):
            FADE_OUT_FRAMES = 500
            MIN_ALPHA = 0.05

            for record in self.global_update_history:
                if record['floor'] != curr_floor:
                    continue

                hist_pose = record['pose']
                record_frame_idx = record.get('frame', -1) 
                
                age_frames = frame_index - record_frame_idx
                if age_frames < 0:
                    continue

                alpha_factor = 1.0 - (age_frames / FADE_OUT_FRAMES)
                alpha = max(alpha_factor * 0.4, MIN_ALPHA)
                if alpha <= MIN_ALPHA and alpha_factor <= 0:
                    continue

                vec_fwd = -hist_pose[:3, 2]
                angle = np.degrees(np.arctan2(vec_fwd[2] * -1, vec_fwd[0]))
                hx, hz = to_plot_coords(hist_pose[0, 3], hist_pose[2, 3])

                wedge = Wedge((hx, hz), r=5.0, theta1=angle-45, theta2=angle+45,
                              color='blue', alpha=alpha, edgecolor=None)
                ax.add_patch(wedge)

                point_alpha = max(alpha_factor * 0.8, MIN_ALPHA * 2)
                plt.scatter(hx, hz, c='blue', s=40, marker='D', zorder=4,
                            edgecolors='black', linewidth=0.8, alpha=point_alpha)

        traj_arr = np.array(self.floor_trajectories[curr_floor])
        if len(traj_arr) > 0:
            tx, tz = to_plot_coords(traj_arr[:, 0], traj_arr[:, 2])
            plt.plot(tx, tz, color='red', linestyle='-', linewidth=4.0, alpha=0.8, label='Path', zorder=3)

        curr_fwd = -current_pose[:3, 2]
        curr_angle = np.degrees(np.arctan2(curr_fwd[2] * -1, curr_fwd[0]))
        
        curr_wedge = Wedge((cx, cz), r=5.0, theta1=curr_angle-45, theta2=curr_angle+45, color='red', alpha=0.3)
        ax.add_patch(curr_wedge)
        plt.scatter(cx, cz, c='red', s=50, edgecolors='black', zorder=5)

        
        FOCUS_RADIUS_M = 4.0
        
        zoom_xs = [cx]
        zoom_zs = [cz]
        
        for (px, pz) in plotted_objects.values():
            dist = np.sqrt((px - cx)**2 + (pz - cz)**2)
            if dist < FOCUS_RADIUS_M:
                zoom_xs.append(px)
                zoom_zs.append(pz)
        
        if len(traj_arr) > 0:
            recent_traj = traj_arr[-20:] 
            tx_rec, tz_rec = to_plot_coords(recent_traj[:, 0], recent_traj[:, 2])
            zoom_xs.extend(tx_rec)
            zoom_zs.extend(tz_rec)

        min_x, max_x = min(zoom_xs), max(zoom_xs)
        min_z, max_z = min(zoom_zs), max(zoom_zs)
        
        center_x = (min_x + max_x) / 2
        center_z = (min_z + max_z) / 2
        
        MIN_WINDOW_SIZE = 2.0 
        span_x = max(max_x - min_x, MIN_WINDOW_SIZE)
        span_z = max(max_z - min_z, MIN_WINDOW_SIZE)
        
        max_span = max(span_x, span_z)
        
        pad = max_span * 0.1
        half_side = (max_span / 2) + pad
        half_side = half_side / 2
        ax.set_xlim(center_x - half_side, center_x + half_side)
        ax.set_ylim(center_z - half_side, center_z + half_side)

        plt.title(f"Map F{curr_floor} | Frame {frame_index} | {len(plotted_objects)} Objects", fontsize=12)
        plt.axis('equal') 
        plt.grid(True, linestyle=':', alpha=0.3)
        plt.xticks([]); plt.yticks([])

        if return_path_only:
            save_path = os.path.join(self.run_dir, "vlm_decision_map_temp.jpg")
        else:
            save_path = os.path.join(floor_map_dir, f"bev_map_{frame_index:04d}.jpg")
            
        plt.savefig(save_path, dpi=120, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        return save_path
    def _perform_global_update(self, current_pose, frame_index):
        """Run when the trigger condition is met."""
        print("\n" + "#"*60)
        print(f">>> [Life-Long Learning] TRIGGERING GLOBAL UPDATE at Frame {frame_index} <<<")
        print("#"*60)

        optimize_start = time.time()
        final_ids = self.cleanup_and_optimize_rooms()
        print(f"  [Global] Room optimization completed in {time.time() - optimize_start:.2f}s. Final room count: {len(final_ids)}")

        print("  [Global] Re-assigning objects to optimized rooms...")
        reassign_start = time.time()
        self._reassign_objects_to_rooms()
        print(f"  [Global] Object re-assignment completed in {time.time() - reassign_start:.2f}s.")

        print(f"  [Global] Running VLM Expansion & Summarization for {len(final_ids)} rooms...")
        room_summary_start = time.time()
        for rid in final_ids:
            r_node = self.finalize_room_semantics(rid, force_vlm_expansion=True)
            if r_node:
                self.floor_tracker.add_room(r_node)
        print(f"  [Global] Room summarization completed in {time.time() - room_summary_start:.2f}s.")
        print(">>> Global Update Cycle Completed.\n")
   
    def _reassign_objects_to_rooms(self):
        for item in self.memory_kb:
            if item.get('floor_id') == self.floor_tracker.current_floor_id:
                pos = item.get('global_pos_3d')
                if pos:
                    u, v = self.map_manager.world_to_grid(pos[0], pos[2])
                    if self.map_manager._valid_uv(u, v):
                        new_rid = self.map_manager.room_id_map[v, u]
                        if new_rid > 0: item['parent_room'] = int(new_rid)
    
    def _compute_centroid(self, objects: List[Dict]) -> np.ndarray:
        if not objects: return np.array([0.0, 0.0])
        coords = []
        for obj in objects:
            pos = obj.get('global_pos_3d')
            if pos:
                coords.append([pos[0], pos[2]])
        if not coords: return np.array([0.0, 0.0])
        return np.mean(coords, axis=0)
    
    def _merge_overfragmented_rooms(self, room_masks):
        ids = list(room_masks.keys())
        merged_map = {uid: uid for uid in ids}
        
        MERGE_INTERFACE_THRESH_PX = 60
        
        occ_map = (self.map_manager.occ_bin > 0).astype(np.uint8)
        
        print(f"  Checking {len(ids)} rooms for merge (Interface Width Check)...")
        
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                
                root_a = id_a
                while merged_map[root_a] != root_a: root_a = merged_map[root_a]
                root_b = id_b
                while merged_map[root_b] != root_b: root_b = merged_map[root_b]
                
                if root_a == root_b: continue
                
                mask_a, mask_b = room_masks[id_a], room_masks[id_b]
                
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                dilated_a = cv2.dilate(mask_a, kernel, iterations=2)
                
                intersection = cv2.bitwise_and(dilated_a, mask_b)
                
                contact_px = np.sum(intersection)
                
                if contact_px > 0:
                    wall_in_contact = cv2.bitwise_and(intersection, occ_map)
                    wall_ratio = np.sum(wall_in_contact) / contact_px
                    
                    if wall_ratio > 0.3:
                        # print(f"    {id_a} & {id_b}: Blocked by wall (ratio {wall_ratio:.2f}). Keep Separate.")
                        continue
                        
                    
                    estimated_interface_len = contact_px / 4.0
                    
                    if estimated_interface_len > MERGE_INTERFACE_THRESH_PX:
                        # print(f"    Merging {id_a} & {id_b}: Wide interface ({estimated_interface_len:.1f} > {MERGE_INTERFACE_THRESH_PX})")
                        merged_map[root_b] = root_a
                    else:
                        # print(f"    {id_a} & {id_b}: Narrow interface ({estimated_interface_len:.1f} <= {MERGE_INTERFACE_THRESH_PX}). Keep Separate.")
                        pass

        new_masks = defaultdict(lambda: np.zeros_like(list(room_masks.values())[0]))
        for old_id, mask in room_masks.items():
            root = old_id
            while merged_map[root] != root: root = merged_map[root]
            new_masks[root] = cv2.bitwise_or(new_masks[root], mask)
            
        return dict(new_masks), merged_map
    
    def cleanup_and_optimize_rooms(self):
        print("\n>>> Starting Global Room Optimization & Cleanup (NO MERGE) <<<")
        
        raw_masks = {}
        if hasattr(self, 'room_segmenter') and hasattr(self.room_segmenter, 'global_id_grid'):
            unique_ids = np.unique(self.room_segmenter.global_id_grid)
            for uid in unique_ids:
                if uid > 0: 
                    mask = (self.room_segmenter.global_id_grid == uid).astype(np.uint8)
                    if np.sum(mask) > 0: raw_masks[int(uid)] = mask
        elif hasattr(self.room_segmenter, 'persistent_masks') and self.room_segmenter.persistent_masks:
            raw_masks = self.room_segmenter.persistent_masks.copy()
        else:
            unique_ids = np.unique(self.map_manager.room_id_map)
            for uid in unique_ids:
                if uid > 0: raw_masks[int(uid)] = (self.map_manager.room_id_map == uid).astype(np.uint8)
        
        if not raw_masks: return []

        
        merged_masks = raw_masks
        

        room_obj_counts = defaultdict(int)
        current_floor_objs = [item for item in self.memory_kb if item.get('floor_id') == self.floor_tracker.current_floor_id]
        
        for item in current_floor_objs:
            pos = item.get('global_pos_3d')
            if pos:
                u, v = self.map_manager.world_to_grid(pos[0], pos[2])
                if self.map_manager._valid_uv(u, v):
                    for rid, mask in merged_masks.items():
                        if mask[v, u] > 0:
                            room_obj_counts[rid] += 1
                            break

        optimized_masks = {}
        valid_room_ids = []

        AREA_PRUNE_THRESH = 150
        
        for rid, mask in merged_masks.items():
            area_px = np.sum(mask)
            obj_count = room_obj_counts[rid]
            
            if area_px < 50: continue
            
            if obj_count == 0 and area_px < AREA_PRUNE_THRESH:
                print(f"  [Prune] Room {rid} removed (Area: {area_px}, Objs: 0)")
                continue
            
            if obj_count == 0 and area_px < 2000:
                print(f"  [Prune] Large Empty Room {rid} removed (Area: {area_px}, Objs: 0)")
                continue

            if hasattr(self.map_manager, 'regularize_mask_rectilinear'):
                reg_mask = self.map_manager.regularize_mask_rectilinear(mask)
            else:
                reg_mask = mask
            
            optimized_masks[rid] = reg_mask
            valid_room_ids.append(rid)

        if hasattr(self.map_manager, 'resolve_global_room_overlaps'):
            final_map = self.map_manager.resolve_global_room_overlaps(optimized_masks)
            self.map_manager.room_id_map = final_map
        else:
            final_map = np.zeros((self.map_manager.map_h, self.map_manager.map_w), dtype=np.int32)
            for rid, mask in optimized_masks.items():
                final_map[mask > 0] = rid
            self.map_manager.room_id_map = final_map
        
        final_valid_ids = []
        unique_final = np.unique(self.map_manager.room_id_map)
        for uid in unique_final:
            if uid > 0: final_valid_ids.append(int(uid))
            
        print(f"  Optimization Done. Final Valid Rooms: {final_valid_ids}")
        return final_valid_ids
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    def finalize_room_semantics(self, room_id, subset_objects=None, force_vlm_expansion=False):
        if room_id is None or room_id == 0: return None
        try: r_key = int(room_id)
        except: r_key = room_id

        print(f"\n>>> Finalizing Semantic Tree for Room ID: {r_key} (Force: {force_vlm_expansion}) <<<")
        current_time_str = datetime.datetime.now().isoformat()

        # 1. Mask
        mask = (self.map_manager.room_id_map == r_key).astype(np.uint8)
        if np.sum(mask) == 0:
            print(f"  [Skip] Room {r_key} has no area.")
            return None
        
        save_path = os.path.join(self.run_dir, f"mask_room_{r_key}.png")
        cv2.imwrite(save_path, (mask * 255).astype(np.uint8))

        # 2. Objects
        room_objects = []
        for item in self.memory_kb:
            if item.get('floor_id') != self.floor_tracker.current_floor_id: continue
            pos = item.get('global_pos_3d')
            if pos is None: continue
            if self.map_manager.is_point_in_room(pos, r_key):
                item['parent_room'] = r_key 
                room_objects.append(item)
            elif item.get('parent_room') == r_key: 
                item['parent_room'] = None

        if len(room_objects) == 0:
            print(f"  [Skip] Room {r_key} is empty.")
            return None

        best_frame_idx = -1
        max_overlap = -1
        candidate_frames = self.room_frame_history.get(r_key, [])
        
        if candidate_frames:
            step = max(1, len(candidate_frames) // 50) 
            for i in range(0, len(candidate_frames), step):
                f_idx = candidate_frames[i]
                pose = self.pose_cache.get(f_idx)
                if pose is not None:
                    overlap = self.map_manager.calculate_visible_room_area(pose, r_key)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_frame_idx = f_idx
            
            if best_frame_idx == -1 and len(candidate_frames) > 0:
                print(f"  [View Selection] No optimal view found (max_overlap={max_overlap}). Using middle frame as fallback.")
                best_frame_idx = candidate_frames[len(candidate_frames) // 2]
        
        best_image_path = None
        saved_summary_img_path = None 

        if best_frame_idx >= 0:
            best_image_path, _ = self._get_image_path(best_frame_idx)
            best_pose = self.pose_cache.get(best_frame_idx)

            if best_pose is not None:
                if not hasattr(self, 'global_update_history'): self.global_update_history = []
                self.global_update_history.append({
                    'pose': best_pose,
                    'floor': self.floor_tracker.current_floor_id,
                    'frame': best_frame_idx,
                    'room_id': r_key
                })

            if best_image_path and os.path.exists(best_image_path):
                if not os.path.exists(self.room_summary_img_dir):
                    os.makedirs(self.room_summary_img_dir, exist_ok=True)

                img_name = f"room_{r_key}_floor_{self.floor_tracker.current_floor_id}_summary.jpg"
                saved_summary_img_path = os.path.join(self.room_summary_img_dir, img_name)
                
                try:
                    shutil.copy(best_image_path, saved_summary_img_path)
                    print(f"  [IO] Saved Best View Image: {saved_summary_img_path}")
                except Exception as e:
                    print(f"  [Error] Failed to copy summary image: {e}")

                if self.config.get('debug_mode', True):
                    vlm_debug_dir = os.path.join(self.run_dir, "debug_vlm_images")
                    os.makedirs(vlm_debug_dir, exist_ok=True)
                    shutil.copy(best_image_path, os.path.join(vlm_debug_dir, img_name))
        else:
            print(f"  [Warning] No valid best view frame found for Room {r_key} (History Len: {len(candidate_frames)})")

        vlm_augmented_labels = []
        if r_key in self.room_metadata_cache and 'augmented_labels' in self.room_metadata_cache[r_key]:
            vlm_augmented_labels.extend(self.room_metadata_cache[r_key]['augmented_labels'])
        
        should_run_vlm = force_vlm_expansion or (not vlm_augmented_labels)
        if should_run_vlm and best_image_path and os.path.exists(best_image_path) and hasattr(self, 'llm'):
            print(f"  [VLM] Scanning room image (Force={force_vlm_expansion})...")
            existing_labels = list(set([obj['label'].split(' ')[0].split('_')[0] for obj in room_objects]))
            detected_new = self.llm.detect_room_objects_vlm(best_image_path, existing_labels)
            
            ignore_set = {'wall', 'walls', 'floor', 'floors', 'ceiling', 'ceilings', 'item', 'object', 'unknown'}
            for label in detected_new:
                clean_l = label.lower().strip()
                if clean_l not in existing_labels and clean_l not in ignore_set and clean_l not in vlm_augmented_labels:
                    vlm_augmented_labels.append(clean_l)
            
            if r_key not in self.room_metadata_cache: self.room_metadata_cache[r_key] = {}
            self.room_metadata_cache[r_key]['augmented_labels'] = vlm_augmented_labels

        cluster_graph = nx.Graph()
        for i in range(len(room_objects)): cluster_graph.add_node(i)
        
        for i in range(len(room_objects)):
            for j in range(i + 1, len(room_objects)):
                p1 = room_objects[i].get('global_pos_3d')
                p2 = room_objects[j].get('global_pos_3d')
                if p1 and p2:
                    dist = np.linalg.norm(np.array([p1[0], p1[2]]) - np.array([p2[0], p2[2]]))
                    if dist < DIST_THRESH: cluster_graph.add_edge(i, j)
        new_area_clusters = list(nx.connected_components(cluster_graph))
        
        new_area_ids = [] 
        csv_file = open(self.edges_path, 'a', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        room_node_id = f"room_{r_key}"
        all_area_summaries = [] 

        
        for idx, indices in enumerate(new_area_clusters):
            cluster_objs = [room_objects[i] for i in indices]
            centroid = self._compute_centroid(cluster_objs)
            llm_input = [{"tag": o['label'], "desc": o.get('llm_desc', '')} for o in cluster_objs]
            
            if len(cluster_objs) > 0:
                llm_area_summary_start = time.time()
                llm_res = self.llm.generate_spatial_cluster_summary(llm_input)
                print(f"  [LLM] Area summaries generated in {time.time() - llm_area_summary_start:.2f}s.")
        
                area_tag = llm_res.get('area_tag', 'area')
                area_summary = llm_res.get('summary', 'Cluster of objects.')
            else:
                area_tag = "empty_area"; area_summary = "Empty."

            uid = int(time.time() * 1000) % 100000
            area_id = f"area_{r_key}_{idx}_{uid}"
            new_area_ids.append(area_id)
            all_area_summaries.append({"tag": area_tag, "summary": area_summary, "id": area_id})

            area_node = {
                "node_id": area_id, "node_level": 2, "area_tag": area_tag, "llm_summary": area_summary,
                "contained_obj_nodes": [o['node_id'] for o in cluster_objs],
                "parent_room_id": room_node_id, "centroid": centroid.tolist(), "timestamp": current_time_str
            }
            with open(self.nodes_path, 'a', encoding='utf-8') as f: f.write(json.dumps(area_node, ensure_ascii=False) + "\n")
            
            for o in cluster_objs:
                csv_writer.writerow([o['node_id'], area_id, 3])
                o['parent_area'] = area_id 
            csv_writer.writerow([area_id, room_node_id, 2])

        llm_room_summary_start = time.time() 
        room_vlm_result = self.llm.generate_multimodal_room_summary(best_image_path, all_area_summaries)
        print(f"  [LLM] Room summary generated in {time.time() - llm_room_summary_start:.2f}s.")
        
        room_label = room_vlm_result.get("room_label", "Room")
        room_desc = room_vlm_result.get("description", "A generic room.")
        
        room_centroid = [0.0, 0.0]
        if room_objects: room_centroid = self._compute_centroid(room_objects).tolist()

        room_node = {
            "node_id": room_node_id, "node_level": 1, "room_label": room_label, "llm_summary": room_desc,
            "floor_id": self.floor_tracker.current_floor_id, "mask_file": f"mask_room_{r_key}.png",
            "centroid": room_centroid, "contained_area_ids": new_area_ids,
            "best_view_image": saved_summary_img_path,
            "vlm_augmented_labels": vlm_augmented_labels, 
            "timestamp": current_time_str, "room_type": room_label
        }

        with open(self.nodes_path, 'a', encoding='utf-8') as f: f.write(json.dumps(room_node, ensure_ascii=False) + "\n")
        csv_file.close()
        print(f"  Room Finalized: {room_node_id} | Label: {room_label} | New Areas: {len(new_area_ids)}")
        return room_node_id  
    
    
    def _generate_decision_map_for_vlm(self, prev_room_id: int, curr_room_id: int, frame_index: int) -> str:
        """Draw a simplified semantic map from the room id map."""
        room_map = self.map_manager.room_id_map
        occ = self.map_manager.occ_bin
        
        h, w = room_map.shape
        viz = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        viz[occ > 0] = (0, 0, 0)
        
        mask_all = (room_map > 0)
        viz[mask_all] = (220, 220, 220) # Light Gray
        
        if prev_room_id > 0:
            mask_prev = (room_map == prev_room_id)
            viz[mask_prev] = (0, 165, 255) # BGR for Orange
            
            ys, xs = np.where(mask_prev)
            if len(xs) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                cv2.putText(viz, f"PREV: {prev_room_id}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if curr_room_id > 0:
            mask_curr = (room_map == curr_room_id)
            viz[mask_curr] = (0, 255, 0) # BGR for Green
            
            ys, xs = np.where(mask_curr)
            if len(xs) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                cv2.putText(viz, f"CURR: {curr_room_id}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)

        curr_floor = self.floor_tracker.current_floor_id
        traj = self.floor_trajectories.get(curr_floor, [])
        if len(traj) > 1:
            pts = []
            for p in traj:
                u, v = self.map_manager.world_to_grid(p[0], p[2])
                if self.map_manager._valid_uv(u, v): pts.append([u, v])
            cv2.polylines(viz, [np.array(pts, np.int32)], False, (255, 0, 0), 1)

        mask_interest = (room_map == prev_room_id) | (room_map == curr_room_id)
        if np.sum(mask_interest) > 0:
            ys, xs = np.where(mask_interest)
            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)
            pad = 50
            x1, x2 = max(0, x_min - pad), min(w, x_max + pad)
            y1, y2 = max(0, y_min - pad), min(h, y_max + pad)
            viz = viz[y1:y2, x1:x2]

        debug_dir = os.path.join(self.run_dir, "debug_vlm_decisions")
        os.makedirs(debug_dir, exist_ok=True)
        save_path = os.path.join(debug_dir, f"map_decision_{frame_index:04d}_P{prev_room_id}_C{curr_room_id}.jpg")
        cv2.imwrite(save_path, viz)
        return save_path

    def _reassign_objects_to_rooms(self):
        """Reassign objects after the map shape changes."""
        for item in self.memory_kb:
            if item.get('floor_id') == self.floor_tracker.current_floor_id:
                pos = item.get('global_pos_3d')
                if pos:
                    u, v = self.map_manager.world_to_grid(pos[0], pos[2])
                    if self.map_manager._valid_uv(u, v):
                        new_rid = self.map_manager.room_id_map[v, u]
                        if new_rid > 0: item['parent_room'] = int(new_rid)

    def _perform_immediate_label_expansion(self, room_id: int, frame_index: int):
        """Expand tags on the room best view when requested."""
        image_path, _ = self._get_image_path(frame_index)
        
        room_objs = [obj for obj in self.memory_kb if obj.get('parent_room') == room_id]
        existing_labels = list(set([o['label'].split('_')[0] for o in room_objs]))
        
        if hasattr(self, 'llm'):
            new_labels = self.llm.detect_room_objects_vlm(image_path, existing_labels)
            if new_labels:
                print(f"  [VLM Expansion] Immediate scan found: {new_labels}")
                if room_id not in self.room_metadata_cache:
                    self.room_metadata_cache[room_id] = {}
                
                current_aug = self.room_metadata_cache[room_id].get('augmented_labels', [])
                current_aug.extend(new_labels)
                self.room_metadata_cache[room_id]['augmented_labels'] = list(set(current_aug))


    # [NEW]
    def _online_mapping_and_room_step(self, frame_index: int, rgb_path: str, depth_path: str, pose: np.ndarray):
        
        map_start = time.time()
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is not None and pose is not None:
            self.map_manager.integrate_depth_raycast(depth_img, pose)
            self.map_manager.refresh_binary_maps()

        print(f"  [Mapping] Frame {frame_index} integrated in {time.time() - map_start:.2f}s.")
        room_seg_start = time.time()
        
        curr_floor = self.floor_tracker.current_floor_id
        curr_traj = self.floor_trajectories.get(curr_floor, [])
        
        seg_result = self.room_segmenter.step(pose, floor_trajectory=curr_traj)
        
        if seg_result.get('room_mask_local') is not None:
            self.map_manager.room_id_map = seg_result['room_mask_local'].astype(np.int32)

        print(f"  [Room Segmentation] Frame {frame_index} processed in {time.time() - room_seg_start:.2f}s.")
        viz_img = seg_result.get('viz')
        if self.debug_mode and viz_img is not None:
            floor_viz_dir = os.path.join(self.room_viz_dir, f"floor_{curr_floor:02d}")
            os.makedirs(floor_viz_dir, exist_ok=True)
            save_path = os.path.join(floor_viz_dir, f"room_seg_{frame_index:04d}.jpg")
            cv2.imwrite(save_path, viz_img)
        
    def _save_map_data_pkl(self):
        map_data = {
            "resolution": self.map_manager.resolution,
            "min_x": self.map_manager.min_x,
            "min_z": self.map_manager.min_z,
            "room_mask": self.map_manager.room_id_map.astype(np.int32),

            "occ_bin": self.map_manager.occ_bin.astype(np.uint8),
            "free_bin": self.map_manager.free_bin.astype(np.uint8),
            "door_map": self.map_manager.door_map.astype(np.uint8),
        }
        out_path = os.path.join(self.run_dir, "map_data.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(map_data, f)
    def _renormalize_floor_indices(self):
        """Renumber floors so the lowest floor is 0 before saving."""
        print("[System] Renormalizing floor indices...")
        
        all_floor_ids = set()
        for item in self.memory_kb:
            if 'floor_id' in item: all_floor_ids.add(item['floor_id'])
        
        all_floor_ids.update(self.floor_tracker.floors_data.keys())
        
        if not all_floor_ids: return

        min_floor = min(all_floor_ids)
        if min_floor >= 0: 
            print("  Min floor is >= 0, no shift needed.")
            return

        offset = abs(min_floor)
        print(f"  Shifting all floor indices by +{offset} (Min was {min_floor})")

        for item in self.memory_kb:
            if 'floor_id' in item:
                item['floor_id'] += offset

        # This state is reused by the following episode. Shifting only the
        # dictionary keys leaves current_floor_id pointing at a missing key;
        # replacing defaultdict with dict also turns that mismatch into an
        # immediate KeyError in FloorTracker.update().
        tracker = self.floor_tracker
        old_floors_data = tracker.floors_data
        default_factory = getattr(old_floors_data, 'default_factory', None)
        if default_factory is None:
            default_factory = lambda: {
                'heights': [], 'rooms': set(), 'finalized_count': 0
            }
        new_floors_data = defaultdict(default_factory)
        for fid, data in old_floors_data.items():
            new_floors_data[int(fid) + offset] = data
        tracker.floors_data = new_floors_data
        tracker.current_floor_id = int(tracker.current_floor_id) + offset
        if hasattr(tracker, 'next_floor_id'):
            tracker.next_floor_id = int(tracker.next_floor_id) + offset
        tracker.id_buffer = deque(
            (int(fid) + offset for fid in tracker.id_buffer),
            maxlen=tracker.min_consistent_frames,
        )

        # Preserve new_id(y) == old_id(y) + offset for future observations.
        if tracker.base_height is not None:
            tracker.base_height = (
                float(tracker.base_height)
                - offset * float(tracker.floor_height_step)
            )

        # These objects are serialized into builder_state.pkl and therefore
        # must use the same shifted floor coordinate system for EP1/EP2.
        self.floor_trajectories = {
            int(fid) + offset: trajectory
            for fid, trajectory in self.floor_trajectories.items()
        }
        self.map_manager.floor_maps = {
            int(fid) + offset: state
            for fid, state in self.map_manager.floor_maps.items()
        }
        self.map_manager.current_floor_idx = (
            int(self.map_manager.current_floor_idx) + offset
        )
        self.room_segmenter.floor_states = {
            int(fid) + offset: state
            for fid, state in self.room_segmenter.floor_states.items()
        }
        self.room_segmenter.current_floor_idx = (
            int(self.room_segmenter.current_floor_idx) + offset
        )

        if tracker.current_floor_id not in tracker.floors_data:
            raise RuntimeError(
                "Floor renormalization lost the active floor state: "
                f"current={tracker.current_floor_id}, "
                f"known={sorted(tracker.floors_data)}"
            )
        
        self._rewrite_nodes_edges_with_offset(offset)

    def _rewrite_nodes_edges_with_offset(self, offset):
        nodes_data = []
        if os.path.exists(self.nodes_path):
            with open(self.nodes_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip(): nodes_data.append(json.loads(line))
        
        edges_data = []
        if os.path.exists(self.edges_path):
            with open(self.edges_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader: edges_data.append(row)

        
        id_map = {} # old_floor_node_id -> new_floor_node_id
        
        for node in nodes_data:
            if "floor_index" in node:
                old_idx = node["floor_index"]
                new_idx = old_idx + offset
                node["floor_index"] = new_idx
                
                old_id = node["node_id"]
                new_id = f"floor_{new_idx:02d}"
                node["node_id"] = new_id
                id_map[old_id] = new_id
                
                if "Floor" in node.get("llm_summary", ""):
                    node["llm_summary"] = node["llm_summary"].replace(f"Floor {old_idx}", f"Floor {new_idx}")

            if "floor_id" in node:
                node["floor_id"] += offset

        new_edges = []
        for row in edges_data:
            u, v, rel = row
            if u in id_map: u = id_map[u]
            if v in id_map: v = id_map[v]
            new_edges.append([u, v, rel])

        with open(self.nodes_path, 'w', encoding='utf-8') as f:
            for node in nodes_data:
                f.write(json.dumps(node, ensure_ascii=False) + "\n")
                
        with open(self.edges_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["child_node_id", "parent_node_id", "node_level"])
            writer.writerows(new_edges)
            
        print("  Nodes and Edges files updated with new floor indices.")
    # =========================================================
    # =========================================================
    def _finalize_algorithm(self):
        """Finalize optimization and published records after the trajectory."""
        print("\n" + "="*50)
        print(">>> Finalizing Trajectory Algorithm...")
        print("="*50)

        final_ids = self.cleanup_and_optimize_rooms()
        
        print("  [Algorithm] Re-assigning objects to optimized rooms...")
        count_reassigned = 0
        for item in self.memory_kb:
            if item.get('floor_id') == self.floor_tracker.current_floor_id:
                pos = item.get('global_pos_3d')
                if pos:
                    u, v = self.map_manager.world_to_grid(pos[0], pos[2])
                    if self.map_manager._valid_uv(u, v):
                        new_rid = self.map_manager.room_id_map[v, u]
                        if new_rid > 0 and item.get('parent_room') != int(new_rid):
                            item['parent_room'] = int(new_rid)
                            count_reassigned += 1
        print(f"  [Algorithm] {count_reassigned} objects reassigned.")

        print(f"  [Algorithm] Finalizing {len(final_ids)} valid rooms...")
        for rid in final_ids:
            r_node = self.finalize_room_semantics(rid)
            if r_node: 
                self.floor_tracker.add_room(r_node)
        
        self.floor_tracker.finalize_floor()

        self._renormalize_floor_indices()
        print(">>> Algorithm Finalization Complete.")

    # =========================================================
    # =========================================================
    def save_all_data(self):
        """Write the published data files."""
        print("\n" + "="*50)
        print(f">>> Saving All Data to: {self.run_dir}")
        print("="*50)
        
        os.makedirs(self.run_dir, exist_ok=True)

        # Normalize the same provenance schema into every persisted object
        # representation, including runs that did not supply an episode ID.
        for item in self.memory_kb:
            ensure_object_provenance(item)
            sync_object_provenance_to_graph(self.spatial_graph, item)

        config_path = os.path.join(self.run_dir, "config_snapshot.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f)
        print(f"  [Save] Config saved to: config_snapshot.yaml")

        kb_path = os.path.join(self.run_dir, "memory_kb.json")
        serializable_kb = []
        for item in self.memory_kb:
            s_item = item.copy()
            if 'dinov3_feature' in s_item:
                feat = s_item['dinov3_feature']
                if isinstance(feat, (np.ndarray, torch.Tensor)):
                    s_item['dinov3_feature'] = feat.tolist()
            if 'global_pos_3d' in s_item and s_item['global_pos_3d'] is not None:
                if isinstance(s_item['global_pos_3d'], (np.ndarray, torch.Tensor)):
                    s_item['global_pos_3d'] = s_item['global_pos_3d'].tolist()
            serializable_kb.append(s_item)

        with open(kb_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_kb, f, indent=2, ensure_ascii=False)
        print(f"  [Save] Memory KB ({len(serializable_kb)} items) saved to: memory_kb.json")

        graph_path = os.path.join(self.run_dir, "spatial_graph.pkl")
        with open(graph_path, 'wb') as f:
            pickle.dump(self.spatial_graph, f)
        print(f"  [Save] Spatial Graph ({self.spatial_graph.number_of_nodes()} nodes) saved to: spatial_graph.pkl")

        self._save_map_data_pkl()
        print(f"  [Save] Map Data saved to: map_data.pkl")

        deduplicated_nodes = process_nodes_file(self.nodes_path, self.nodes_path)
        final_nodes = merge_object_provenance_into_nodes(
            deduplicated_nodes.values(),
            self.memory_kb,
        )
        with open(self.nodes_path, 'w', encoding='utf-8') as f:
            for node in final_nodes:
                f.write(json.dumps(node, ensure_ascii=False) + "\n")
        
        analyze_area_quality(
            {str(node.get('node_id')): node for node in final_nodes}
        )
    
        
        print(">>> All Files Saved Successfully.")

    def set_realtime_map_provider(self, callback_fn):
        """Register a callback that returns the latest global map."""
        self.realtime_map_callback = callback_fn

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    def run(self):
        try:
            max_frames = self.config['trajectory_data']['num_images_to_process']
            total_start = time.time()
            
            self.current_room_id = 0
            self.room_segmenter.current_room_id = 0
            self.visited_room_ids = set()
            self.global_update_history = [] 
            per_start = time.time()
            i = 0
            while True:
                if max_frames != -1 and i >= max_frames:
                    print(f"Reached max frames limit: {max_frames}")
                    break

                rgb_path, depth_path = self._get_image_path(i)
                if not os.path.exists(rgb_path):
                    print(f"Image not found at {rgb_path}. End of trajectory.")
                    break

                print(f"\n===== Frame {i} =====")
                pose_path = self._get_pose_path(i)
                current_pose = read_pose_matrix(pose_path) if os.path.exists(pose_path) else None

                is_kf, feat = self._is_keyframe(Image.open(rgb_path).convert("RGB"), i)
                if is_kf:
                    self.last_kf_feature = feat

                    if current_pose is not None and os.path.exists(depth_path):
                        curr_room_node = self.room_topology_cache.get(self.current_room_id, {}).get("room_node_id")
                        is_floor_switch = self.floor_tracker.update(current_pose, curr_room_node)

                        if is_floor_switch:
                            print(f"\n>>> Floor Switch Detected! <<<")
                            global_update_start = time.time()

                            self._perform_global_update(current_pose, i)

                            print(f"  [Global Update] Completed in {time.time() - global_update_start:.2f}s")
                            
                            floor_start = time.time()
                            self.floor_tracker.finalize_floor()
                            print(f"  [Floor Tracker] Finalized Floor {self.floor_tracker.current_floor_id} in {time.time() - floor_start:.2f}s")
                            

                            switch_start = time.time()
                            new_floor_id = self.floor_tracker.current_floor_id
                            self.map_manager.switch_floor(new_floor_id)
                            self.room_segmenter.switch_floor(new_floor_id)
                            self.current_room_id = 0
                            self.room_segmenter.current_room_id = 0
                            print(f">>> Switched Context to Floor {new_floor_id}")
                            print(f"  [Floor Switch] Completed in {time.time() - switch_start:.2f}s")
                        
                        self._online_mapping_and_room_step(i, rgb_path, depth_path, current_pose)

                        seg_curr_id = self.room_segmenter.current_room_id
                        
                        if self.current_room_id != seg_curr_id:
                            print(f"\n>>> Room Switch Detected: {self.current_room_id} -> {seg_curr_id}")
                            self.current_room_id = seg_curr_id
                            
                            if seg_curr_id > 0:
                                self.cleanup_and_optimize_rooms()
                                
                                bev_path = self._update_and_save_bev_map(i, current_pose, return_path_only=True)
                                
                                if hasattr(self, 'llm'):
                                    print("  [Action] Asking VLM if Global Update is needed...")
                                    decision = self.llm.analyze_trajectory_map_for_actions(bev_path, self.floor_tracker.current_floor_id)
                                    
                                    if decision.get('trigger_global_update', False):
                                        global_update_start = time.time()
                                        self._perform_global_update(current_pose, i)
                                        print(f"  [Global Update] Completed in {time.time() - global_update_start:.2f}s")
                            
                                    else:
                                        print("  [VLM] Update skipped.")

                    seg_json_path = self._run_segmentation(i, rgb_path, depth_path)
                    self._process_frame_results(i, seg_json_path, rgb_path, depth_path, current_pose)

                if current_pose is not None: self.pose_cache[i] = current_pose
                
                if self.current_room_id > 0:
                    if self.current_room_id not in self.room_frame_history:
                        self.room_frame_history[self.current_room_id] = []
                    self.room_frame_history[self.current_room_id].append(i)
                
                if self.room_segmenter.current_room_id > 0:
                    self.visited_room_ids.add(self.room_segmenter.current_room_id)
                
                if self.current_room_id != self.room_segmenter.current_room_id:
                     self.current_room_id = self.room_segmenter.current_room_id

                print(f"Frame {i} Done. RoomID: {self.current_room_id}")
                i += 1
            
            per_end = time.time() - per_start
            print(f"\n=== Trajectory Processing Time: {per_end:.2f}s ===")
            print(f"{per_end / max(1, i):.2f}s per frame over {i} frames.")
            

            # End of Trajectory
            self._finalize_algorithm() 
            self.save_all_data()        
            
            print(f"\nTotal Time: {time.time() - total_start:.2f}s")
            
        except Exception as e:
            print(f"FATAL ERROR in run(): {e}")
            import traceback; traceback.print_exc()
            try: self.save_all_data()
            except: pass

    def detect_keyframe_only(self, i: int, rgb_img: np.ndarray) -> bool:
        """Detect keyframes with DINOv3 on the main thread."""
        try:
            pil_img = Image.fromarray(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
            
            is_kf, feat = self._is_keyframe(pil_img, i)
            
            if is_kf:
                self.last_kf_feature = feat
                return True
            return False
            
        except Exception as e:
            print(f"[KeyframeDetect] Error: {e}")
            return False

    def process_confirmed_keyframe(self, i: int, rgb_img: np.ndarray, depth_img: np.ndarray, current_pose: np.ndarray, timestamp: float):
        try:
            print(f"\n===== Processing Keyframe {i} (TS: {timestamp:.2f}) =====")
            self._current_frame_timestamp = float(timestamp)

            filename = f"{self.filename_prefix}{i:06d}.png" 
            rgb_path = os.path.join(self.rgb_dir, filename)
            depth_path = os.path.join(self.depth_dir, filename)
            cv2.imwrite(rgb_path, rgb_img)
            cv2.imwrite(depth_path, depth_img)

            use_external_map = False 
            
            # ==================================================================================
            # ==================================================================================
            if self.external_context is not None:
                use_external_map = True
                ext_room_id = self.external_context.get('room_id', 0)
                ext_floor_id = self.external_context.get('floor_id', 0)
                ext_room_mask = self.external_context.get('room_mask', None)

                if self.floor_tracker.current_floor_id != ext_floor_id:
                    if self.floor_tracker.current_floor_id != 0:
                        self.floor_tracker.finalize_floor()
                        self.map_manager.switch_floor(ext_floor_id)
                        self.room_segmenter.switch_floor(ext_floor_id)
                    self.floor_tracker.current_floor_id = ext_floor_id
                
                self.current_room_id = ext_room_id
                
                if ext_room_mask is not None:
                    target_h, target_w = self.map_manager.room_id_map.shape
                    if ext_room_mask.shape == (target_h, target_w):
                        self.map_manager.room_id_map = ext_room_mask.astype(np.int32)
                    else:
                        src_h, src_w = ext_room_mask.shape
                        final_mask = np.zeros((target_h, target_w), dtype=np.int32)
                        
                        dy, dx = (target_h - src_h) // 2, (target_w - src_w) // 2
                        
                        ty_start = max(0, dy)
                        ty_end = min(target_h, dy + src_h)
                        tx_start = max(0, dx)
                        tx_end = min(target_w, dx + src_w)
                        
                        sy_start = max(0, -dy)
                        sy_end = min(src_h, target_h - dy)
                        sx_start = max(0, -dx)
                        sx_end = min(src_w, target_w - dx)
                        
                        if ty_end > ty_start and tx_end > tx_start:
                            final_mask[ty_start:ty_end, tx_start:tx_end] = ext_room_mask[sy_start:sy_end, sx_start:sx_end]
                        
                        self.map_manager.room_id_map = final_mask
                        self.room_segmenter.global_id_grid = final_mask

                    mask_bool = (self.map_manager.room_id_map > 0)
                    self.room_segmenter.global_conf_grid[:] = 0
                    self.room_segmenter.global_conf_grid[mask_bool] = 2.0

                self.external_context = None 
            
            # ==================================================================================
            # ==================================================================================
            if not use_external_map and current_pose is not None:
                self.floor_tracker.update(current_pose)
                self._online_mapping_and_room_step(i, rgb_path, depth_path, current_pose)
                self.current_room_id = self.room_segmenter.current_room_id
            elif use_external_map and current_pose is not None:
                d_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                self.map_manager.integrate_depth_raycast(d_img, current_pose)

            # ==================================================================================
            # ==================================================================================
            if self.current_room_id != self.last_room_id:
                print(f">>> Room Change Detected: {self.last_room_id} -> {self.current_room_id}")
                
                if self.last_room_id > 0:
                    print(f"  [Trigger] Finalizing previous Room {self.last_room_id}...")
                    r_node = self.finalize_room_semantics(self.last_room_id)
                    if r_node:
                        self.floor_tracker.add_room(r_node)
                
                self.last_room_id = self.current_room_id
                self.floor_tracker.current_room_id = self.current_room_id

            if self.current_room_id > 0:
                if self.current_room_id not in self.visited_room_ids:
                    self.visited_room_ids.add(self.current_room_id)
                if self.current_room_id not in self.room_frame_history:
                    self.room_frame_history[self.current_room_id] = []
                self.room_frame_history[self.current_room_id].append(i)
                self.pose_cache[i] = current_pose

            # ==================================================================================
            # ==================================================================================
            seg_json_path = self._run_segmentation(i, rgb_path, depth_path)
            
            self._process_frame_results(i, seg_json_path, rgb_path, depth_path, current_pose)
            
            self._update_and_save_bev_map(i, current_pose)
            
            return True

        except Exception as e:
            print(f"ERROR in process_confirmed_keyframe({i}): {e}")
            import traceback
            traceback.print_exc()
            if str(os.environ.get("INHERIT_SG_FAIL_CLOSED", "0")).strip().lower() in {
                "1", "true", "yes", "on"
            }:
                raise
            return False
    

# export HF_ENDPOINT=https://hf-mirror.com
# export CUDA_VISIBLE_DEVICES=5,6,7
# ==============================================================================
# ==============================================================================




# ==============================================================================
# ==============================================================================
class GPUMonitor(threading.Thread):
    def __init__(self, save_dir, device, filename="gpu_memory_log.csv", interval=1.0, plot_name="gpu_memory_plot.png"):
        """save_dir is the output directory."""
        super().__init__()
        self.save_dir = save_dir
        self.csv_path = os.path.join(save_dir, filename)
        self.plot_path = os.path.join(save_dir, plot_name)
        self.interval = interval
        self.device = device
        self.running = False
        self.history = []
        self.start_time = 0

    def get_gpu_memory(self):
        """Read the process GPU memory in megabytes."""
        try:
            if torch.cuda.is_available():
                mem_bytes = torch.cuda.memory_reserved(self.device)
                return int(mem_bytes / 1024 / 1024)
            else:
                return 0
        except Exception:
            return 0

    def run(self):
        self.running = True
        self.start_time = time.time()
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time(s)", "Memory_Reserved(MB)"])

        while self.running:
            current_mem = self.get_gpu_memory()
            elapsed_time = time.time() - self.start_time
            
            self.history.append((elapsed_time, current_mem))
            
            try:
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f"{elapsed_time:.2f}", current_mem])
            except:
                pass
            
            time.sleep(self.interval)

    def stop(self):
        self.running = False

    def plot_and_save(self):
        """Plot and save the GPU memory curve."""
        if not self.history:
            return

        times = [x[0] for x in self.history]
        mems = [x[1] for x in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(times, mems, label=f'VRAM Reserved ({self.device})', color='orange', linewidth=1.5)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory (MB)')
        plt.title(f'Process GPU Memory Usage Over Time ({self.device})')
        plt.grid(True)
        plt.legend()
        
        try:
            plt.savefig(self.plot_path)
            plt.close()
            print(f"     [Monitor] Plot saved to {self.plot_path}")
        except Exception as e:
            print(f"     [Monitor] Failed to save plot: {e}")




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run Mapping Builder based on Config")
    parser.add_argument("--config", type=str, default="config_sam3.yaml", help="Path to config file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gpu_config = base_config.get('multi_gpu', {
            'enabled': True,
            'sam3_device': 'cuda:0',      
            'dinov3_device': 'cuda:0',    
            'text_encoder_device': 'cuda:0'
        })
        
    if gpu_config['enabled']:
        print("[Multi-GPU] Enabled. Model distribution:")
        for k, v in gpu_config.items():
            if k != 'enabled':
                print(f"  {k}: {v}")
        
    sam3_device = gpu_config.get('sam3_device', 'cuda:0') if gpu_config['enabled'] else device
    dinov3_device = gpu_config.get('dinov3_device', 'cuda:0') if gpu_config['enabled'] else device
    text_device = gpu_config.get('text_encoder_device', 'cuda:0') if gpu_config['enabled'] else device

    monitor_device = sam3_device 

    shared_models = {}
    print(">>> Pre-loading Models...")
    shared_models['dinov3'] = load_feature_extractor(base_config, dinov3_device)
    
    seg_config = base_config.get("segmentation_module", {})
    shared_models['sam3'] = Sam3LocalClient(base_config, device = sam3_device)
    
    shared_models['text_encoder'] = SentenceTransformer('/data2/shizk/all-MiniLM-L6-v2', device=text_device)
    print(">>> Models Loaded.")

    ds_cfg = base_config.get('dataset_config', {})
    root_dir = ds_cfg.get('root_dir', '/data2/lavira-team/SegRAG/hm3dsem_walks/val')
    max_frames = ds_cfg.get('max_frames', -1)
    
    target_scenes = []
    cfg_scenes = ds_cfg.get('selected_scenes', ['all'])
    if 'all' in cfg_scenes:
        if os.path.exists(root_dir):
            target_scenes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        else:
            print(f"Dataset root not found: {root_dir}")
            sys.exit(1)
    else:
        target_scenes = cfg_scenes

    cfg_episodes = ds_cfg.get('selected_episodes', ['all'])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_root = base_config.get('main_output_directory', 'output')
    batch_run_dir = os.path.join(main_output_root, f"run_{timestamp}")
    os.makedirs(batch_run_dir, exist_ok=True)
    print(f"[Output] Batch results will be saved to: {batch_run_dir}")

    for scene_id in target_scenes:
        scene_path = os.path.join(root_dir, scene_id)
        if not os.path.isdir(scene_path): continue
        
        try:
            scene_hash = scene_id.split('-')[1] if '-' in scene_id else ""
            file_prefix = f"{scene_hash}_" if scene_hash else f"{scene_id}_"
        except:
            file_prefix = f"{scene_id}_"

        current_episodes = []
        if 'all' in cfg_episodes:
            current_episodes = sorted([d for d in os.listdir(scene_path) if d.startswith('ep_') and os.path.isdir(os.path.join(scene_path, d))])
        else:
            current_episodes = cfg_episodes

        for ep_id in current_episodes:
            ep_path = os.path.join(scene_path, ep_id)
            if not os.path.exists(ep_path): continue

            print(f"\n{'='*40}")
            print(f"Processing: Scene {scene_id} | Episode {ep_id}")
            print(f"{'='*40}")

            output_dir = os.path.join(batch_run_dir, scene_id, ep_id)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # ==================================================================
            # ==================================================================
            gpu_monitor = GPUMonitor(save_dir=output_dir, device=monitor_device, interval=1.0)
            gpu_monitor.start()
            
            override_settings = {
                "main_output_directory": output_dir,
                "trajectory_data": {
                    "base_dir": ep_path,
                    "filename_prefix": file_prefix,
                    "num_images_to_process": max_frames,
                    "rgb_subdir": "rgb",
                    "depth_subdir": "depth",
                    "pose_subdir": "pose"
                }
            }

            try:
                pipeline = TrajectoryPipeline(
                    copy.deepcopy(base_config), 
                    override_settings=override_settings,
                    shared_models=shared_models 
                )
                
                pipeline.run()
                
            except Exception as e:
                print(f"[Error] {scene_id}/{ep_id}: {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                # ==============================================================
                # ==============================================================
                if 'gpu_monitor' in locals() and gpu_monitor.is_alive():
                    gpu_monitor.stop()
                    gpu_monitor.join()
                    gpu_monitor.plot_and_save()

                if 'pipeline' in locals():
                    del pipeline
                gc.collect()
                torch.cuda.empty_cache()

    print("\nBatch Builder Finished.")
