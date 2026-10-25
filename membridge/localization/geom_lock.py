"""Lock T_query→memory with DINO top-k + SuperPoint/LightGlue + depth SE(2) RANSAC.

Default gate from the Janus 157 eval: inliers >= 80, and if the second
candidate has >= 20 inliers it must lose by 1.5x (blocks the 9 m 84/63 alias).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from frames import camera_intrinsics
from se2 import SE2, se2_error, wrap_angle

LIGHTGLUE_ROOT = Path(os.environ.get("LIGHTGLUE_ROOT", ""))
PYDEPS = Path(os.environ.get("PYDEPS", ""))
TORCH_HOME = Path(os.environ.get("TORCH_HOME", str(Path.home() / ".cache/torch")))

TOP_K = 5
MIN_INLIERS = 80
SECOND_MIN = 20
SECOND_RATIO = 1.5
SAME_POSE_MAX_M = 0.25
SAME_POSE_MAX_RAD = math.radians(10.0)
INLIER_THRESH_M = 0.10
REFINE_MAX_M = 1.00
REFINE_MAX_DEG = 20.0
REFINE_MIN_SCORE = 0.12
REFINE_MIN_FRAMES = 1
YAW_ALIAS_GAP_MIN = 0.12
QUERY_WINDOW = 24
MIN_DEPTH_M = 0.30
MAX_DEPTH_M = 8.0
RANSAC_ITERS = 256
MAX_KPTS = 1024
GPU_MEM_FRACTION = 0.40

_SP = None
_LG = None
_DEVICE = None


def _bootstrap() -> None:
    os.environ["TORCH_HOME"] = str(TORCH_HOME)
    roots = [LIGHTGLUE_ROOT]
    if os.environ.get("MEMBRIDGE_SKIP_PYDEPS") != "1":
        roots.insert(0, PYDEPS)
    for path in roots:
        s = str(path)
        if path.exists() and s not in sys.path:
            sys.path.insert(0, s)


def limit_gpu_memory(fraction: float = GPU_MEM_FRACTION) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(fraction))
    except Exception as exc:
        print(f"gpu memory cap skipped: {exc}", flush=True)


def models():
    global _SP, _LG, _DEVICE
    if _SP is not None:
        return _SP, _LG
    _bootstrap()
    import torch
    from lightglue.lightglue import LightGlue
    from lightglue.superpoint import SuperPoint

    _DEVICE = torch.device(os.environ.get(
        "MEMBRIDGE_LOCALIZATION_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    ))
    _SP = SuperPoint(max_num_keypoints=MAX_KPTS).eval().to(_DEVICE)
    _LG = LightGlue(features="superpoint").eval().to(_DEVICE)
    return _SP, _LG


def _rgb_tensor(rgb: np.ndarray):
    import torch

    arr = np.asarray(rgb, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def _feat_to_cpu(feats: dict) -> dict:
    import torch

    out = {}
    for key, value in feats.items():
        out[key] = value.detach().cpu() if torch.is_tensor(value) else value
    return out


def _feat_to_device(feats: dict, device) -> dict:
    import torch

    out = {}
    for key, value in feats.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def extract_superpoint(rgb: np.ndarray) -> dict:
    import torch

    extractor, _ = models()
    image = _rgb_tensor(rgb).to(_DEVICE)
    with torch.no_grad():
        feats = extractor.extract(image, resize=None)
    kpts = feats["keypoints"][0].detach().cpu().numpy()
    return {"keypoints": kpts, "raw": _feat_to_cpu(feats)}


def match_pair(feat_q: dict, feat_m: dict) -> np.ndarray:
    import torch

    _, matcher = models()
    data = {
        "image0": _feat_to_device(feat_q["raw"], _DEVICE),
        "image1": _feat_to_device(feat_m["raw"], _DEVICE),
    }
    with torch.no_grad():
        out = matcher(data)
    matches = out["matches0"][0].detach().cpu().numpy()
    valid = matches >= 0
    q_idx = np.where(valid)[0]
    m_idx = matches[valid]
    if q_idx.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.stack([q_idx, m_idx], axis=1)


def unproject_keypoints(frame, kpts_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(frame.depth_m, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    height, width = depth.shape
    fx, fy, cx, cy = camera_intrinsics(width, height)
    n = kpts_xy.shape[0]
    xz = np.zeros((n, 2), dtype=np.float64)
    u = kpts_xy[:, 0]
    v = kpts_xy[:, 1]
    ui = np.clip(np.round(u).astype(np.int32), 0, width - 1)
    vi = np.clip(np.round(v).astype(np.int32), 0, height - 1)
    z = depth[vi, ui].astype(np.float64)
    valid = np.isfinite(z) & (z >= MIN_DEPTH_M) & (z <= MAX_DEPTH_M)
    if not np.any(valid):
        return xz, valid
    uu = u[valid].astype(np.float64)
    vv = v[valid].astype(np.float64)
    zz = z[valid]
    cam_x = (uu - cx) * zz / fx
    cam_y = (vv - cy) * zz / fy
    points_h = np.stack([cam_x, -cam_y, -zz, np.ones_like(zz)], axis=1)
    xyz = (frame.camera_to_local @ points_h.T).T[:, :3]
    xz[valid, 0] = xyz[:, 0]
    xz[valid, 1] = xyz[:, 2]
    return xz, valid


def _se2_from_two(src: np.ndarray, dst: np.ndarray) -> SE2 | None:
    d_src = src[1] - src[0]
    d_dst = dst[1] - dst[0]
    n_src = math.hypot(float(d_src[0]), float(d_src[1]))
    n_dst = math.hypot(float(d_dst[0]), float(d_dst[1]))
    if n_src < 0.05 or n_dst < 0.05:
        return None
    # Habitat yaw rotates XZ with [[cos, sin], [-sin, cos]], so the
    # polar-angle change has the opposite sign to yaw.
    yaw = wrap_angle(math.atan2(d_src[1], d_src[0]) - math.atan2(d_dst[1], d_dst[0]))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rx = cosine * src[0, 0] + sine * src[0, 1]
    rz = -sine * src[0, 0] + cosine * src[0, 1]
    return SE2(x=float(dst[0, 0] - rx), z=float(dst[0, 1] - rz), yaw=yaw)


def _se2_least_squares(src: np.ndarray, dst: np.ndarray) -> SE2 | None:
    c_s = src.mean(axis=0)
    c_d = dst.mean(axis=0)
    a = src - c_s
    b = dst - c_d
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    yaw = math.atan2(r[0, 1], r[0, 0])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rx = cosine * c_s[0] + sine * c_s[1]
    rz = -sine * c_s[0] + cosine * c_s[1]
    return SE2(x=float(c_d[0] - rx), z=float(c_d[1] - rz), yaw=wrap_angle(yaw))


def se2_ransac(src: np.ndarray, dst: np.ndarray) -> tuple[SE2 | None, int, float]:
    n = src.shape[0]
    if n < 2:
        return None, 0, 0.0
    rng = np.random.default_rng(0)
    best_T = None
    best_n = 0
    iters = min(RANSAC_ITERS, max(32, n * (n - 1)))
    for _ in range(iters):
        i, j = rng.choice(n, size=2, replace=False)
        T = _se2_from_two(src[[i, j]], dst[[i, j]])
        if T is None:
            continue
        mapped = T.apply(src)
        err = np.hypot(mapped[:, 0] - dst[:, 0], mapped[:, 1] - dst[:, 1])
        nin = int((err <= INLIER_THRESH_M).sum())
        if nin > best_n:
            best_n = nin
            best_T = T
    if best_T is None or best_n < 2:
        return None, 0, 0.0
    mapped = best_T.apply(src)
    err = np.hypot(mapped[:, 0] - dst[:, 0], mapped[:, 1] - dst[:, 1])
    src_i, dst_i = src[err <= INLIER_THRESH_M], dst[err <= INLIER_THRESH_M]
    if src_i.shape[0] >= 2:
        T_ref = _se2_from_two(src_i[:2], dst_i[:2])
        if T_ref is not None:
            mapped = T_ref.apply(src)
            err = np.hypot(mapped[:, 0] - dst[:, 0], mapped[:, 1] - dst[:, 1])
            nin = int((err <= INLIER_THRESH_M).sum())
            if nin >= best_n:
                best_T, best_n = T_ref, nin
                src_i, dst_i = src[err <= INLIER_THRESH_M], dst[err <= INLIER_THRESH_M]
        if src_i.shape[0] >= 3:
            T_ls = _se2_least_squares(src_i, dst_i)
            if T_ls is not None:
                mapped = T_ls.apply(src)
                err = np.hypot(mapped[:, 0] - dst[:, 0], mapped[:, 1] - dst[:, 1])
                nin = int((err <= INLIER_THRESH_M).sum())
                if nin >= best_n:
                    best_T, best_n = T_ls, nin
    return best_T, int(best_n), float(best_n / max(n, 1))


@dataclass
class GeomCandidate:
    n_match: int
    n_3d: int
    n_inlier: int
    ratio: float
    transform: SE2 | None
    k: int = -1
    dino: float = 0.0
    refined: bool = False
    occ_score: float | None = None
    geom_transform: SE2 | None = None


def score_candidate(query_frame, map_frame, feat_q: dict, feat_m: dict) -> GeomCandidate:
    pairs = match_pair(feat_q, feat_m)
    empty = GeomCandidate(int(pairs.shape[0]), 0, 0, 0.0, None)
    if pairs.shape[0] < 2:
        return empty
    q_xz, q_ok = unproject_keypoints(query_frame, feat_q["keypoints"][pairs[:, 0]])
    m_xz, m_ok = unproject_keypoints(map_frame, feat_m["keypoints"][pairs[:, 1]])
    both = q_ok & m_ok
    if int(both.sum()) < 2:
        return GeomCandidate(int(pairs.shape[0]), int(both.sum()), 0, 0.0, None)
    transform, nin, ratio = se2_ransac(q_xz[both], m_xz[both])
    return GeomCandidate(int(pairs.shape[0]), int(both.sum()), int(nin), float(ratio), transform)


def _competing_candidate(best: GeomCandidate, ranked: list[GeomCandidate]):
    """Multiple map views supporting one pose are agreement, not aliases."""
    if best.transform is None:
        return None
    for candidate in ranked:
        if candidate is best or candidate.transform is None:
            continue
        distance, angle = se2_error(candidate.transform, best.transform)
        if distance <= SAME_POSE_MAX_M and angle <= SAME_POSE_MAX_RAD:
            continue
        return candidate
    return None


def decide_geom_lock(cands: list[GeomCandidate]) -> tuple[bool, str, GeomCandidate]:
    if not cands:
        empty = GeomCandidate(0, 0, 0, 0.0, None)
        return False, "no candidates", empty
    ranked = sorted(cands, key=lambda c: c.n_inlier, reverse=True)
    best = ranked[0]
    competitor = _competing_candidate(best, ranked)
    second_n = competitor.n_inlier if competitor is not None else 0
    if best.transform is None or best.n_inlier < MIN_INLIERS:
        return False, f"inliers {best.n_inlier} < {MIN_INLIERS}", best
    if second_n >= SECOND_MIN and best.n_inlier < SECOND_RATIO * max(second_n, 1):
        return False, f"not unique inliers {best.n_inlier}/{second_n}", best
    return True, f"geom inliers {best.n_inlier}/{second_n}", best


def _yaw_alias_gap(map_frames, query_window, seed: SE2) -> tuple[bool, float, float]:
    """Reject 90/180/270 aliases: occupancy at T must beat the best alias."""
    from localize import _score_transform, build_map_occupancy, build_query_occupancy

    if seed is None or query_window is None or len(query_window) < 1:
        return True, 0.0, 0.0
    occupancy = build_map_occupancy(map_frames)
    query_occ = build_query_occupancy(list(query_window))
    pts = query_occ.occupied_points()
    # Empty occupancy carries no evidence for or against a yaw alias. Keep
    # the already gated geometric estimate instead of comparing sentinel scores.
    if pts.shape[0] == 0 or not np.any(occupancy.occupied):
        return True, 0.0, 0.0
    s0 = float(_score_transform(seed, pts, occupancy))
    alias = []
    for turn in (0.5, 1.0, 1.5):
        t_alias = SE2(seed.x, seed.z, wrap_angle(seed.yaw + turn * math.pi))
        alias.append(float(_score_transform(t_alias, pts, occupancy)))
    s_alias = max(alias) if alias else float("-inf")
    return (s0 - s_alias) >= YAW_ALIAS_GAP_MIN, s0, s_alias


def apply_occ_refine(map_frames, query_window, seed: SE2) -> tuple[SE2, bool, float]:
    """Local occupancy CSM around geom T. No 90/180 search."""
    from localize import refine_local
    from se2 import se2_error

    if seed is None or query_window is None or len(query_window) < REFINE_MIN_FRAMES:
        return seed, False, float("nan")
    occ = refine_local(
        map_frames,
        list(query_window),
        seed,
        trans_radius_m=REFINE_MAX_M,
        yaw_radius_deg=REFINE_MAX_DEG,
    )
    drift_m, drift_yaw = se2_error(occ.transform, seed)
    score = float(occ.score)
    if (
        score >= REFINE_MIN_SCORE
        and drift_m <= REFINE_MAX_M
        and math.degrees(drift_yaw) <= REFINE_MAX_DEG
    ):
        return occ.transform, True, score
    return seed, False, score


def try_lock_frame(
    query_frame,
    map_frames,
    mem_feats: np.ndarray,
    map_sp: list[dict],
    *,
    top_k: int = TOP_K,
    query_window=None,
) -> tuple[bool, str, GeomCandidate]:
    from dino import encode_rgbs

    qfeat = encode_rgbs([query_frame.rgb])[0]
    vis = mem_feats @ qfeat
    order = np.argsort(-vis)[: min(top_k, vis.size)]
    feat_q = extract_superpoint(query_frame.rgb)
    cands: list[GeomCandidate] = []
    for k in order:
        scored = score_candidate(query_frame, map_frames[int(k)], feat_q, map_sp[int(k)])
        scored.k = int(k)
        scored.dino = float(vis[int(k)])
        cands.append(scored)
    ok, reason, best = decide_geom_lock(cands)
    if best.transform is None:
        return ok, reason, best
    best.geom_transform = best.transform
    window = list(query_window) if query_window is not None else [query_frame]
    T, did, score = apply_occ_refine(map_frames, window, best.transform)
    best.occ_score = score if score == score else None
    if did:
        best.transform = T
        best.refined = True
        reason = reason + f" +occ {score:.3f}"
    ok_alias, s0, s_alias = _yaw_alias_gap(map_frames, window, best.transform)
    if ok:
        if not ok_alias:
            ok = False
            reason = f"yaw alias {s0:.3f}-{s_alias:.3f} < {YAW_ALIAS_GAP_MIN}"
            best.refined = False
        return ok, reason, best
    ranked = sorted(cands, key=lambda c: c.n_inlier, reverse=True)
    second = _competing_candidate(best, ranked)
    if (
        best.n_inlier >= MIN_INLIERS
        and second is not None
        and second.transform is not None
        and ok_alias
    ):
        _, _, s2 = apply_occ_refine(map_frames, window, second.transform)
        s2_use = s2 if s2 == s2 else float("-inf")
        if s0 - s2_use >= YAW_ALIAS_GAP_MIN:
            reason = (
                f"occ-disambig inliers {best.n_inlier}/{second.n_inlier} "
                f"occ {s0:.3f}-{s2_use:.3f}"
            )
            if did:
                reason += f" +occ {score:.3f}"
            return True, reason, best
    return ok, reason, best


def extract_map_superpoint(map_frames) -> list[dict]:
    return [extract_superpoint(frame.rgb) for frame in map_frames]
