"""Object-landmark RANSAC + weighted SE(2) Procrustes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from frames import LocalFrame, camera_intrinsics
from se2 import SE2, wrap_angle

def _first_existing(paths: list[str]) -> Path:
    for path in paths:
        candidate = Path(path)
        if candidate.is_file():
            return candidate
    return Path(paths[-1])


GDINO_CONFIG = _first_existing(
    [
        "/workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "/home/shizk/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    ]
)
GDINO_WEIGHTS = _first_existing(
    [
        os.environ.get("MODELS_DIR", "") + "/groundingdino_swint_ogc.pth",
        "/workspace/GroundingDINO/weights/groundingdino_swint_ogc.pth",
    ]
)
BERT_PATH = Path(os.environ.get("MODELS_DIR", "")) / "bert-base-uncased"
CAPTION = (
    "chair . table . sofa . couch . plant . bed . toilet . tv . refrigerator . "
    "cabinet . desk . sink . lamp . shelf . microwave . oven . dresser . "
    "nightstand . bathtub . painting . window"
)
LABEL_ALIAS = {"couch": "sofa", "tv": "tv", "television": "tv"}
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
CLUSTER_M = 0.45
BASELINE_MIN_M = 0.50
PAIR_DIST_EPS_M = 0.40
INLIER_TAU_M = 0.40
MIN_INLIERS = 3
RANSAC_ITERS = 250
VIS_MIN = 0.35
P95_MAX_M = 0.50
INLIER_RATIO_MIN = 0.45

_GDINO = None


@dataclass
class Landmark:
    x: float
    z: float
    label: str
    conf: float
    vis: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(1, dtype=np.float32))


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _install_msda_fallback() -> None:
    import torch
    import torch.nn.functional as F
    import groundingdino.models.GroundingDINO.ms_deform_attn as msda

    if getattr(msda, "_llhn_fallback", False):
        return

    def _core(value, value_spatial_shapes, sampling_locations, attention_weights):
        n_batch, _, n_heads, depth = value.shape
        _, n_query, _, n_levels, n_points, _ = sampling_locations.shape
        split_sizes = [int(h * w) for h, w in value_spatial_shapes]
        value_list = value.split(split_sizes, dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampled = []
        for level, (height, width) in enumerate(value_spatial_shapes):
            value_l = (
                value_list[level]
                .flatten(2)
                .transpose(1, 2)
                .reshape(n_batch * n_heads, depth, int(height), int(width))
            )
            grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
            sampled.append(
                F.grid_sample(
                    value_l,
                    grid_l,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
            )
        weights = attention_weights.transpose(1, 2).reshape(
            n_batch * n_heads, 1, n_query, n_levels * n_points
        )
        stacked = torch.stack(sampled, dim=-2).flatten(-2)
        output = (stacked * weights).sum(-1).view(n_batch, n_heads * depth, n_query)
        return output.transpose(1, 2).contiguous()

    def _forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        return _core(value, value_spatial_shapes, sampling_locations, attention_weights)

    msda.MultiScaleDeformableAttnFunction.forward = staticmethod(_forward)
    msda._llhn_fallback = True


def _load_gdino():
    global _GDINO
    if _GDINO is not None:
        return _GDINO
    import torch
    _install_msda_fallback()
    from groundingdino.models import build_model
    from groundingdino.util.misc import clean_state_dict
    from groundingdino.util.slconfig import SLConfig

    args = SLConfig.fromfile(str(GDINO_CONFIG))
    args.device = _device()
    if BERT_PATH.is_dir():
        args.text_encoder_type = str(BERT_PATH)
    model = build_model(args)
    checkpoint = torch.load(str(GDINO_WEIGHTS), map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.to(_device()).eval()
    _GDINO = model
    return model


def _transform_image(rgb: np.ndarray):
    import groundingdino.datasets.transforms as T

    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensor, _ = transform(image, None)
    return tensor


def _detect_boxes(rgb: np.ndarray) -> list[tuple[str, float, np.ndarray]]:
    _install_msda_fallback()
    from groundingdino.util.inference import predict

    model = _load_gdino()
    image = _transform_image(rgb)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=CAPTION,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=_device(),
    )
    height, width = rgb.shape[:2]
    out = []
    for box, logit, phrase in zip(boxes, logits, phrases):
        cx, cy, bw, bh = [float(v) for v in box.tolist()]
        x0 = max(0.0, (cx - bw / 2.0) * width)
        y0 = max(0.0, (cy - bh / 2.0) * height)
        x1 = min(float(width), (cx + bw / 2.0) * width)
        y1 = min(float(height), (cy + bh / 2.0) * height)
        if (x1 - x0) * (y1 - y0) < 0.004 * width * height:
            continue
        label = LABEL_ALIAS.get(str(phrase).strip().lower(), str(phrase).strip().lower())
        label = label.split()[0] if label else "object"
        score = float(logit.max()) if hasattr(logit, "max") else float(logit)
        out.append((label, score, np.array([x0, y0, x1, y1], dtype=np.float64)))
    return out


def _backproject(frame: LocalFrame, u: float, v: float) -> np.ndarray | None:
    depth = np.asarray(frame.depth_m, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    height, width = depth.shape
    ui = int(round(u))
    vi = int(round(v))
    if ui < 0 or vi < 0 or ui >= width or vi >= height:
        return None
    patch = depth[max(0, vi - 2) : min(height, vi + 3), max(0, ui - 2) : min(width, ui + 3)]
    valid = patch[(np.isfinite(patch)) & (patch > 0.3) & (patch < 5.0)]
    if valid.size == 0:
        return None
    z_cam = float(np.median(valid))
    fx, fy, cx, cy = camera_intrinsics(width, height)
    camera_x = (u - cx) * z_cam / fx
    camera_y = (v - cy) * z_cam / fy
    point = frame.camera_to_local @ np.array([camera_x, -camera_y, -z_cam, 1.0], dtype=np.float64)
    return point[:3]


def _crop(rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(rgb.shape[1], x1), min(rgb.shape[0], y1)
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return rgb
    return rgb[y0:y1, x0:x1]


def extract_landmarks(frames: list[LocalFrame], *, stride: int = 1) -> list[Landmark]:
    from dino import encode_rgbs

    raw: list[tuple[str, float, float, float, np.ndarray]] = []
    crops = []
    chosen = frames[:: max(1, stride)]
    for frame in chosen:
        try:
            detections = _detect_boxes(frame.rgb)
        except Exception as exc:
            print(f"[landmarks] detect failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        for label, conf, box in detections:
            u = 0.5 * (box[0] + box[2])
            v = 0.5 * (box[1] + box[3])
            xyz = _backproject(frame, u, v)
            if xyz is None:
                continue
            raw.append((label, conf, float(xyz[0]), float(xyz[2]), box))
            crops.append(_crop(frame.rgb, box))
    if not raw:
        return []
    vis = encode_rgbs(crops)
    clustered: list[Landmark] = []
    used = np.zeros(len(raw), dtype=bool)
    for i, (label, conf, x, z, _) in enumerate(raw):
        if used[i]:
            continue
        members = [i]
        for j in range(i + 1, len(raw)):
            if used[j] or raw[j][0] != label:
                continue
            if math.hypot(raw[j][2] - x, raw[j][3] - z) <= CLUSTER_M:
                members.append(j)
        best = max(members, key=lambda k: raw[k][1])
        for k in members:
            used[k] = True
        feat = vis[best]
        norm = float(np.linalg.norm(feat)) + 1e-6
        clustered.append(
            Landmark(
                x=raw[best][2],
                z=raw[best][3],
                label=raw[best][0],
                conf=float(raw[best][1]),
                vis=(feat / norm).astype(np.float32),
            )
        )
    return clustered


def _two_point_se2(q1: np.ndarray, q2: np.ndarray, m1: np.ndarray, m2: np.ndarray) -> SE2 | None:
    hat = q2 - q1
    vec = m2 - m1
    n_hat = float(np.linalg.norm(hat))
    n_vec = float(np.linalg.norm(vec))
    if n_hat < BASELINE_MIN_M or n_vec < BASELINE_MIN_M:
        return None
    phi = math.atan2(vec[0] * hat[1] - vec[1] * hat[0], vec[0] * hat[0] + vec[1] * hat[1])
    mid_q = 0.5 * (q1 + q2)
    mid_m = 0.5 * (m1 + m2)
    rot = SE2(0.0, 0.0, wrap_angle(phi))
    mapped = rot.apply(mid_q)
    return SE2(x=float(mid_m[0] - mapped[0]), z=float(mid_m[1] - mapped[1]), yaw=wrap_angle(phi))


def _weighted_procrustes(query_pts: np.ndarray, map_pts: np.ndarray, weights: np.ndarray) -> SE2:
    w = np.asarray(weights, dtype=np.float64)
    w = np.clip(w, 1e-6, None)
    w = w / w.sum()
    q = np.asarray(query_pts, dtype=np.float64)
    m = np.asarray(map_pts, dtype=np.float64)
    q_bar = (w[:, None] * q).sum(axis=0)
    m_bar = (w[:, None] * m).sum(axis=0)
    x = q - q_bar
    y = m - m_bar
    covariance = np.zeros((2, 2), dtype=np.float64)
    for i in range(len(q)):
        covariance += w[i] * np.outer(y[i], x[i])
    u_mat, _, vt = np.linalg.svd(covariance)
    rotation = u_mat @ vt
    if np.linalg.det(rotation) < 0:
        u_mat[:, 1] *= -1
        rotation = u_mat @ vt
    cosine = float(rotation[0, 0])
    sine = float(rotation[0, 1])
    yaw = wrap_angle(math.atan2(sine, cosine))
    mapped = SE2(0.0, 0.0, yaw).apply(q_bar)
    return SE2(x=float(m_bar[0] - mapped[0]), z=float(m_bar[1] - mapped[1]), yaw=yaw)


def _residuals(transform: SE2, query_pts: np.ndarray, map_pts: np.ndarray) -> np.ndarray:
    mapped = transform.apply(query_pts)
    return np.linalg.norm(mapped - map_pts, axis=1)


def estimate_landmark_se2(map_lms: list[Landmark], query_lms: list[Landmark]) -> dict:
    empty = {
        "ok": False,
        "transform": SE2(0.0, 0.0, 0.0),
        "n_query": len(query_lms),
        "n_map": len(map_lms),
        "n_candidates": 0,
        "n_inliers": 0,
        "inlier_ratio": 0.0,
        "p95_m": 1e9,
        "c_loc": 0.0,
        "baseline_m": 0.0,
    }
    if len(query_lms) < 3 or len(map_lms) < 3:
        return empty
    candidates: list[tuple[int, int, float]] = []
    for qi, qlm in enumerate(query_lms):
        opts = []
        for mi, mlm in enumerate(map_lms):
            if qlm.label != mlm.label:
                continue
            vis = float(np.dot(qlm.vis, mlm.vis))
            if vis < VIS_MIN and sum(1 for item in map_lms if item.label == qlm.label) > 1:
                continue
            opts.append((mi, vis, qlm.conf * mlm.conf * max(vis, 0.0)))
        opts.sort(key=lambda item: item[2], reverse=True)
        for mi, vis, weight in opts[:3]:
            candidates.append((qi, mi, weight))
    if len(candidates) < 3:
        return empty

    query_pts = np.array([[lm.x, lm.z] for lm in query_lms], dtype=np.float64)
    map_pts = np.array([[lm.x, lm.z] for lm in map_lms], dtype=np.float64)
    rng = np.random.default_rng(0)
    best_inliers: np.ndarray | None = None
    best_count = -1
    best_transform = SE2(0.0, 0.0, 0.0)
    pairs = []
    for a in range(len(candidates)):
        for b in range(a + 1, len(candidates)):
            q1, m1, _ = candidates[a]
            q2, m2, _ = candidates[b]
            if q1 == q2 or m1 == m2:
                continue
            d_q = float(np.linalg.norm(query_pts[q1] - query_pts[q2]))
            d_m = float(np.linalg.norm(map_pts[m1] - map_pts[m2]))
            if d_q < BASELINE_MIN_M or abs(d_q - d_m) > PAIR_DIST_EPS_M:
                continue
            pairs.append((a, b))
    if not pairs:
        return empty
    sample = pairs if len(pairs) <= RANSAC_ITERS else [pairs[i] for i in rng.choice(len(pairs), RANSAC_ITERS, replace=False)]
    cand_q = np.array([query_pts[c[0]] for c in candidates])
    cand_m = np.array([map_pts[c[1]] for c in candidates])
    for ia, ib in sample:
        q1, m1, _ = candidates[ia]
        q2, m2, _ = candidates[ib]
        transform = _two_point_se2(query_pts[q1], query_pts[q2], map_pts[m1], map_pts[m2])
        if transform is None:
            continue
        resid = _residuals(transform, cand_q, cand_m)
        inliers = resid < INLIER_TAU_M
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_transform = transform
    if best_inliers is None or best_count < MIN_INLIERS:
        empty["n_candidates"] = len(candidates)
        return empty
    inlier_idx = np.nonzero(best_inliers)[0]
    q_in = cand_q[inlier_idx]
    m_in = cand_m[inlier_idx]
    weights = np.array([candidates[i][2] for i in inlier_idx], dtype=np.float64)
    transform = _weighted_procrustes(q_in, m_in, weights)
    resid = _residuals(transform, q_in, m_in)
    p95 = float(np.percentile(resid, 95)) if resid.size else 1e9
    baseline = 0.0
    if len(q_in) >= 2:
        diffs = q_in[:, None, :] - q_in[None, :, :]
        baseline = float(np.max(np.linalg.norm(diffs, axis=2)))
    inlier_ratio = best_count / max(len(candidates), 1)
    c_loc = (
        float(best_count >= MIN_INLIERS)
        * float(baseline > BASELINE_MIN_M)
        * float(inlier_ratio)
        * math.exp(-p95 / 0.30)
        * float(np.clip(np.mean(weights) / (np.mean(weights) + 1e-6), 0.0, 1.0))
    )
    ok = best_count >= MIN_INLIERS and p95 <= P95_MAX_M and inlier_ratio >= INLIER_RATIO_MIN and baseline >= BASELINE_MIN_M
    return {
        "ok": bool(ok),
        "transform": transform,
        "n_query": len(query_lms),
        "n_map": len(map_lms),
        "n_candidates": len(candidates),
        "n_inliers": int(best_count),
        "inlier_ratio": float(inlier_ratio),
        "p95_m": p95,
        "c_loc": float(c_loc),
        "baseline_m": baseline,
    }
