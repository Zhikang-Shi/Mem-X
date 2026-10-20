"""Estimate SE(2) from EP1-local observations onto an EP0-local map."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from frames import LocalFrame
from occupancy import OccupancyGrid, build_map_occupancy, build_query_occupancy
from se2 import SE2, minus, pose_se2, wrap_angle

FEATURE_SIZE = 24
TOP_K_MATCHES = 3
MAX_SEEDS = 12
COARSE_TRANS_M = 2.0
COARSE_YAW_DEG = 45.0
COARSE_STEP_M = 0.10
COARSE_STEP_DEG = 5.0
FINE_TRANS_M = 0.30
FINE_YAW_DEG = 8.0
FINE_STEP_M = 0.05
FINE_STEP_DEG = 1.0
FREE_PENALTY = 0.6
UNKNOWN_PENALTY = 0.15
MANHATTAN_YAWS_DEG = (0, 90, 180, 270)
PEAK_GAP_MIN = 0.08
PEAK_RATIO_MIN = 1.15
YAW_MODE_SEP_DEG = 45.0

__all__ = [
    "LocalizationResult",
    "UNKNOWN_PENALTY",
    "estimate_ep1_to_ep0",
    "refine_around_seed",
    "refine_local",
]


@dataclass
class LocalizationResult:
    transform: SE2
    score: float
    n_query_points: int
    n_map_occupied: int
    n_seeds: int
    n_query_frames: int
    n_map_frames: int
    seed_source: str
    extras: dict = field(default_factory=dict)


def rgb_feature(rgb: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    small = np.asarray(image.resize((FEATURE_SIZE, FEATURE_SIZE), Image.BILINEAR), dtype=np.float32)
    vec = small.reshape(-1)
    norm = float(np.linalg.norm(vec)) + 1e-6
    return vec / norm


def _visual_seeds(
    map_frames: list[LocalFrame],
    query_frames: list[LocalFrame],
    *,
    use_dino: bool = False,
) -> list[SE2]:
    if not map_frames or not query_frames:
        return []
    if use_dino:
        from dino import encode_rgbs

        map_feats = encode_rgbs([frame.rgb for frame in map_frames])
        query_feats = encode_rgbs([frame.rgb for frame in query_frames])
    else:
        map_feats = np.stack([rgb_feature(frame.rgb) for frame in map_frames])
        query_feats = np.stack([rgb_feature(frame.rgb) for frame in query_frames])
    seeds: list[SE2] = []
    for query, feat in zip(query_frames, query_feats):
        scores = map_feats @ feat
        top = np.argsort(scores)[-TOP_K_MATCHES:][::-1]
        q_pose = pose_se2(query.x, query.z, query.yaw)
        for index in top:
            mapped = map_frames[int(index)]
            m_pose = pose_se2(mapped.x, mapped.z, mapped.yaw)
            seeds.append(minus(m_pose, q_pose))
    return _cluster_seeds(seeds)


def _cluster_seeds(seeds: list[SE2], trans_m: float = 0.30, yaw_deg: float = 15.0) -> list[SE2]:
    kept: list[SE2] = []
    yaw_tol = math.radians(yaw_deg)
    for seed in seeds:
        duplicate = False
        for other in kept:
            trans, yaw = (minus(seed, other).translation_norm(), abs(wrap_angle(seed.yaw - other.yaw)))
            if trans <= trans_m and yaw <= yaw_tol:
                duplicate = True
                break
        if not duplicate:
            kept.append(seed)
        if len(kept) >= MAX_SEEDS:
            break
    return kept


def _score_transform(transform: SE2, query_pts: np.ndarray, occupancy: OccupancyGrid) -> float:
    if query_pts.shape[0] == 0:
        return -1e9
    mapped = transform.apply(query_pts)
    occupied, free, observed = occupancy.lookup(mapped)
    hits = float(np.count_nonzero(occupied))
    free_hits = float(np.count_nonzero(free))
    unknown = float(np.count_nonzero(~observed))
    n = float(query_pts.shape[0])
    return (hits - FREE_PENALTY * free_hits - UNKNOWN_PENALTY * unknown) / n


def _packed_score_map(occupancy: OccupancyGrid) -> np.ndarray:
    packed = np.full(occupancy.occupied.shape, -UNKNOWN_PENALTY, dtype=np.float32)
    packed[occupancy.free] = -FREE_PENALTY
    packed[occupancy.occupied] = 1.0
    return packed


def _score_translations(
    x_rot: np.ndarray,
    z_rot: np.ndarray,
    txs: np.ndarray,
    tzs: np.ndarray,
    occupancy: OccupancyGrid,
    packed: np.ndarray | None = None,
) -> np.ndarray:
    if packed is None:
        packed = _packed_score_map(occupancy)
    height, width = packed.shape
    origin_x = occupancy.origin_x
    origin_z = occupancy.origin_z
    res = occupancy.resolution_m
    scores = np.empty((txs.size, tzs.size), dtype=np.float64)
    tzs = np.asarray(tzs, dtype=np.float64)
    for i, tx in enumerate(txs):
        cols = np.floor((x_rot + float(tx) - origin_x) / res).astype(np.int32)
        rows = np.floor((z_rot[None, :] + tzs[:, None] - origin_z) / res).astype(np.int32)
        cols_b = np.broadcast_to(cols[None, :], rows.shape)
        inside = (rows >= 0) & (rows < height) & (cols_b >= 0) & (cols_b < width)
        rows_c = np.clip(rows, 0, height - 1)
        cols_c = np.clip(cols_b, 0, width - 1)
        vals = np.where(inside, packed[rows_c, cols_c], -UNKNOWN_PENALTY)
        scores[i] = vals.mean(axis=1)
    return scores


def _search_window(
    seed: SE2,
    query_pts: np.ndarray,
    occupancy: OccupancyGrid,
    trans_radius_m: float,
    yaw_radius_deg: float,
    step_m: float,
    step_deg: float,
) -> tuple[SE2, float]:
    if query_pts.shape[0] == 0:
        return seed, -1e9
    yaw_radius = math.radians(yaw_radius_deg)
    step_yaw = math.radians(max(step_deg, 0.5))
    xs = seed.x + np.arange(-trans_radius_m, trans_radius_m + 1e-9, step_m)
    zs = seed.z + np.arange(-trans_radius_m, trans_radius_m + 1e-9, step_m)
    yaws = seed.yaw + np.arange(-yaw_radius, yaw_radius + 1e-9, step_yaw)
    best = seed
    best_score = _score_transform(seed, query_pts, occupancy)
    packed = _packed_score_map(occupancy)
    for yaw in yaws:
        alpha = wrap_angle(float(yaw))
        cosine, sine = math.cos(alpha), math.sin(alpha)
        x_rot = cosine * query_pts[:, 0] + sine * query_pts[:, 1]
        z_rot = -sine * query_pts[:, 0] + cosine * query_pts[:, 1]
        scores = _score_translations(x_rot, z_rot, xs, zs, occupancy, packed=packed)
        index = np.unravel_index(int(np.argmax(scores)), scores.shape)
        score = float(scores[index])
        if score > best_score:
            best_score = score
            best = SE2(x=float(xs[index[0]]), z=float(zs[index[1]]), yaw=alpha)
    return best, best_score


def _yaw_mode_key(yaw: float) -> int:
    deg = math.degrees(wrap_angle(yaw))
    return int(round(deg / 90.0)) % 4


def _peak_stats(candidates: list[tuple[SE2, float]]) -> dict:
    best_by_mode: dict[int, tuple[SE2, float]] = {}
    for transform, score in candidates:
        key = _yaw_mode_key(transform.yaw)
        if key not in best_by_mode or score > best_by_mode[key][1]:
            best_by_mode[key] = (transform, score)
    ranked = sorted(best_by_mode.values(), key=lambda item: item[1], reverse=True)
    best, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1e9
    gap = float(best_score - second_score)
    if second_score > 1e-6:
        ratio = float(best_score / second_score)
    else:
        ratio = 99.0
    ambiguous = gap < PEAK_GAP_MIN or (second_score > 0 and ratio < PEAK_RATIO_MIN)
    return {
        "best": best,
        "best_score": float(best_score),
        "second_score": float(second_score),
        "peak_gap": gap,
        "peak_ratio": ratio,
        "ambiguous": bool(ambiguous),
        "n_yaw_modes": len(ranked),
    }


def _search_seed(
    seed: SE2,
    query_pts: np.ndarray,
    occupancy: OccupancyGrid,
    *,
    trans_radius_m: float,
    yaw_radius_deg: float,
    step_m: float,
    step_deg: float,
) -> tuple[SE2, float]:
    coarse, _ = _search_window(
        seed,
        query_pts,
        occupancy,
        trans_radius_m=trans_radius_m,
        yaw_radius_deg=yaw_radius_deg,
        step_m=step_m,
        step_deg=step_deg,
    )
    return _search_window(
        coarse,
        query_pts,
        occupancy,
        trans_radius_m=FINE_TRANS_M,
        yaw_radius_deg=FINE_YAW_DEG,
        step_m=FINE_STEP_M,
        step_deg=FINE_STEP_DEG,
    )


def estimate_ep1_to_ep0(
    map_frames: list[LocalFrame],
    query_frames: list[LocalFrame],
    *,
    resolution_m: float = 0.05,
    manhattan: bool = False,
    use_dino: bool = False,
    use_landmarks: bool = False,
) -> LocalizationResult:
    occupancy = build_map_occupancy(map_frames, resolution_m=resolution_m)
    query_occ = build_query_occupancy(query_frames, resolution_m=resolution_m)
    query_pts = query_occ.occupied_points()
    seeds = _visual_seeds(map_frames, query_frames, use_dino=use_dino)
    source = "dino" if use_dino else "visual"
    if not seeds:
        seeds = [SE2(0.0, 0.0, 0.0)]
        source = "origin"

    candidates: list[tuple[SE2, float]] = []
    for seed in seeds:
        headings = MANHATTAN_YAWS_DEG if manhattan else (0,)
        for yaw_off in headings:
            rotated = SE2(
                x=seed.x,
                z=seed.z,
                yaw=wrap_angle(seed.yaw + math.radians(yaw_off)),
            )
            if yaw_off == 0:
                transform, score = _search_seed(
                    rotated,
                    query_pts,
                    occupancy,
                    trans_radius_m=COARSE_TRANS_M,
                    yaw_radius_deg=COARSE_YAW_DEG,
                    step_m=COARSE_STEP_M,
                    step_deg=COARSE_STEP_DEG,
                )
            else:
                transform, score = _search_seed(
                    rotated,
                    query_pts,
                    occupancy,
                    trans_radius_m=0.80,
                    yaw_radius_deg=12.0,
                    step_m=0.15,
                    step_deg=6.0,
                )
            candidates.append((transform, score))

    best = candidates[0][0]
    best_score = candidates[0][1]
    for transform, score in candidates:
        if score > best_score:
            best, best_score = transform, score

    if best_score < 0.12 and query_pts.shape[0] > 20:
        clustered = _cluster_map_positions(map_frames)[:8]
        source = "visual+place_grid"
        yaw_step = 90 if manhattan else 30
        for x, z in clustered:
            for yaw_deg in range(0, 360, yaw_step):
                seed = SE2(x=x, z=z, yaw=math.radians(yaw_deg))
                transform, score = _search_seed(
                    seed,
                    query_pts,
                    occupancy,
                    trans_radius_m=0.8,
                    yaw_radius_deg=8.0,
                    step_m=0.20,
                    step_deg=8.0,
                )
                candidates.append((transform, score))
                if score > best_score:
                    best, best_score = transform, score

    peaks = _peak_stats(candidates)
    best = peaks["best"]
    best_score = peaks["best_score"]
    extras = {
        "map_origin_x": occupancy.origin_x,
        "map_origin_z": occupancy.origin_z,
        "query_origin_x": query_occ.origin_x,
        "query_origin_z": query_occ.origin_z,
        "second_score": peaks["second_score"],
        "peak_gap": peaks["peak_gap"],
        "peak_ratio": peaks["peak_ratio"],
        "ambiguous": peaks["ambiguous"],
        "n_yaw_modes": peaks["n_yaw_modes"],
        "locked": not peaks["ambiguous"],
        "manhattan": bool(manhattan),
        "used_landmark": False,
    }
    if use_landmarks:
        from landmarks import estimate_landmark_se2, extract_landmarks
        from se2 import se2_error

        query_lms = extract_landmarks(query_frames, stride=1)
        map_lms = extract_landmarks(map_frames, stride=max(2, len(map_frames) // 24 or 1))
        landmark = estimate_landmark_se2(map_lms, query_lms)
        extras["landmark_ok"] = landmark["ok"]
        extras["landmark_n_inliers"] = landmark["n_inliers"]
        extras["landmark_p95_m"] = landmark["p95_m"]
        extras["landmark_c_loc"] = landmark["c_loc"]
        extras["landmark_n_query"] = landmark["n_query"]
        extras["landmark_n_map"] = landmark["n_map"]
        if landmark["ok"]:
            refined, refined_score = _search_seed(
                landmark["transform"],
                query_pts,
                occupancy,
                trans_radius_m=0.80,
                yaw_radius_deg=15.0,
                step_m=0.10,
                step_deg=3.0,
            )
            trans_err, yaw_err = se2_error(refined, best)
            prefer = (
                peaks["ambiguous"]
                or trans_err > 1.0
                or yaw_err > math.radians(30.0)
                or refined_score > best_score + 0.02
            )
            if prefer and (refined_score >= 0.20 or landmark["c_loc"] >= 0.20):
                best = refined
                best_score = refined_score
                extras["used_landmark"] = True
                extras["ambiguous"] = False
                extras["locked"] = True
                source = source + "+landmark"
    seed_trans = 1e9
    seed_yaw = math.pi
    for seed in seeds:
        dist = math.hypot(best.x - seed.x, best.z - seed.z)
        yaw = abs(wrap_angle(best.yaw - seed.yaw))
        if dist < seed_trans:
            seed_trans = dist
            seed_yaw = yaw
    extras["seed_trans_m"] = float(seed_trans)
    extras["seed_yaw_deg"] = float(math.degrees(seed_yaw))
    extras["locked"] = (not extras["ambiguous"]) and best_score >= 0.20
    return LocalizationResult(
        transform=best,
        score=float(best_score),
        n_query_points=int(query_pts.shape[0]),
        n_map_occupied=int(np.count_nonzero(occupancy.occupied)),
        n_seeds=len(seeds),
        n_query_frames=len(query_frames),
        n_map_frames=len(map_frames),
        seed_source=source + ("+manhattan" if manhattan else ""),
        extras=extras,
    )


def refine_around_seed(
    map_frames: list[LocalFrame],
    query_frames: list[LocalFrame],
    seed: SE2,
    *,
    resolution_m: float = 0.05,
    trans_radius_m: float = 0.50,
    yaw_radius_deg: float = 15.0,
) -> LocalizationResult:
    """CSM only in a small window around a DINO-implied T; score 90/180/270 aliases locally."""
    occupancy = build_map_occupancy(map_frames, resolution_m=resolution_m)
    query_occ = build_query_occupancy(query_frames, resolution_m=resolution_m)
    query_pts = query_occ.occupied_points()
    candidates: list[tuple[SE2, float]] = []
    for yaw_off in MANHATTAN_YAWS_DEG:
        rotated = SE2(
            x=seed.x,
            z=seed.z,
            yaw=wrap_angle(seed.yaw + math.radians(yaw_off)),
        )
        if yaw_off == 0:
            transform, score = _search_seed(
                rotated,
                query_pts,
                occupancy,
                trans_radius_m=trans_radius_m,
                yaw_radius_deg=yaw_radius_deg,
                step_m=0.05,
                step_deg=1.0,
            )
        else:
            transform, score = _search_seed(
                rotated,
                query_pts,
                occupancy,
                trans_radius_m=0.30,
                yaw_radius_deg=8.0,
                step_m=0.10,
                step_deg=4.0,
            )
        candidates.append((transform, score))
    peaks = _peak_stats(candidates)
    best = peaks["best"]
    best_score = peaks["best_score"]
    seed_t, seed_y = math.hypot(best.x - seed.x, best.z - seed.z), abs(wrap_angle(best.yaw - seed.yaw))
    extras = {
        "second_score": peaks["second_score"],
        "peak_gap": peaks["peak_gap"],
        "peak_ratio": peaks["peak_ratio"],
        "ambiguous": peaks["ambiguous"],
        "n_yaw_modes": peaks["n_yaw_modes"],
        "manhattan": False,
        "used_landmark": False,
        "anchored": True,
        "seed_trans_m": float(seed_t),
        "seed_yaw_deg": float(math.degrees(seed_y)),
    }
    return LocalizationResult(
        transform=best,
        score=float(best_score),
        n_query_points=int(query_pts.shape[0]),
        n_map_occupied=int(np.count_nonzero(occupancy.occupied)),
        n_seeds=1,
        n_query_frames=len(query_frames),
        n_map_frames=len(map_frames),
        seed_source="dino-anchor",
        extras=extras,
    )


def refine_local(
    map_frames: list[LocalFrame],
    query_frames: list[LocalFrame],
    seed: SE2,
    *,
    resolution_m: float = 0.05,
    trans_radius_m: float = 0.50,
    yaw_radius_deg: float = 15.0,
) -> LocalizationResult:
    """Occupancy CSM only around seed yaw; no 90/180 search."""
    occupancy = build_map_occupancy(map_frames, resolution_m=resolution_m)
    query_occ = build_query_occupancy(query_frames, resolution_m=resolution_m)
    query_pts = query_occ.occupied_points()
    transform, score = _search_seed(
        seed,
        query_pts,
        occupancy,
        trans_radius_m=trans_radius_m,
        yaw_radius_deg=yaw_radius_deg,
        step_m=0.05,
        step_deg=1.0,
    )
    seed_t = math.hypot(transform.x - seed.x, transform.z - seed.z)
    seed_y = abs(wrap_angle(transform.yaw - seed.yaw))
    extras = {
        "second_score": None,
        "peak_gap": 0.0,
        "peak_ratio": 1.0,
        "ambiguous": False,
        "n_yaw_modes": 1,
        "manhattan": False,
        "used_landmark": False,
        "anchored": True,
        "seed_trans_m": float(seed_t),
        "seed_yaw_deg": float(math.degrees(seed_y)),
    }
    return LocalizationResult(
        transform=transform,
        score=float(score),
        n_query_points=int(query_pts.shape[0]),
        n_map_occupied=int(np.count_nonzero(occupancy.occupied)),
        n_seeds=1,
        n_query_frames=len(query_frames),
        n_map_frames=len(map_frames),
        seed_source="geom-local",
        extras=extras,
    )


def _cluster_map_positions(map_frames: list[LocalFrame], radius_m: float = 0.75) -> list[tuple[float, float]]:
    points = []
    for frame in map_frames:
        candidate = (float(frame.x), float(frame.z))
        if all(math.hypot(candidate[0] - x, candidate[1] - z) > radius_m for x, z in points):
            points.append(candidate)
        if len(points) >= 24:
            break
    return points
