"""Online estimator of the constant episode-local SE(2) map-from-query transform."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from frames import LocalFrame, start_spin_frames
from localize import (
    UNKNOWN_PENALTY,
    _packed_score_map,
    _score_transform,
    _search_window,
    estimate_ep1_to_ep0,
)
from occupancy import build_map_occupancy, build_query_occupancy
from se2 import SE2, wrap_angle

N_PARTICLES = 360
LOCK_SCORE = 0.12
PEAK_RATIO = 1.15
TIGHT_TRANS_M = 0.10
TIGHT_YAW_RAD = math.radians(5.0)
COLLAPSE_SCORE = 0.04
MAX_QUERY_POINTS = 900
REFINE_TRANS_M = 0.40
REFINE_YAW_DEG = 10.0
REFINE_STEP_M = 0.05
REFINE_STEP_DEG = 1.0
MOVE_UPDATE_M = 0.35


@dataclass
class Snapshot:
    n_frames: int
    mode: str
    transform: SE2
    score: float
    trans_rms_m: float
    yaw_rms_deg: float
    n_query_points: int


class EpisodeLocalizer:
    """Particles live on g (query-local → memory-local). g does not walk."""

    def __init__(self, memory_frames: list[LocalFrame], *, resolution_m: float = 0.05):
        self.memory_frames = memory_frames
        self.memory_occ = build_map_occupancy(memory_frames, resolution_m=resolution_m)
        self.packed = _packed_score_map(self.memory_occ)
        self.resolution_m = resolution_m
        self.query_frames: list[LocalFrame] = []
        self.mode = "UNLOCKED"
        self.particles = np.zeros((N_PARTICLES, 3), dtype=np.float64)
        self.weights = np.ones(N_PARTICLES, dtype=np.float64) / N_PARTICLES
        self.transform = SE2(0.0, 0.0, 0.0)
        self.score = -1e9
        self.trans_rms_m = 1e9
        self.yaw_rms_deg = 1e9
        self._last_update_pose: tuple[float, float] | None = None
        self.n_updates = 0
        self.lock_frame_count = None

    def observe(self, frame: LocalFrame) -> Snapshot:
        self.query_frames.append(frame)
        moved = self._should_update(frame)
        if self.mode == "UNLOCKED":
            if moved:
                self._try_lock_from_recent()
                self.n_updates += 1
                self._last_update_pose = (frame.x, frame.z)
        elif moved:
            self._refine()
            self.n_updates += 1
            self._last_update_pose = (frame.x, frame.z)
        return self.snapshot()

    def ingest_spin(self, frames: list[LocalFrame]) -> Snapshot:
        for frame in frames:
            self.query_frames.append(frame)
        self._try_lock()
        if self.query_frames:
            last = self.query_frames[-1]
            self._last_update_pose = (last.x, last.z)
        return self.snapshot()

    def snapshot(self) -> Snapshot:
        return Snapshot(
            n_frames=len(self.query_frames),
            mode=self.mode,
            transform=self.transform,
            score=float(self.score),
            trans_rms_m=float(self.trans_rms_m),
            yaw_rms_deg=float(self.yaw_rms_deg),
            n_query_points=int(self._query_points().shape[0]),
        )

    def _enough_for_lock_attempt(self) -> bool:
        spin = start_spin_frames(self.query_frames)
        return len(spin) >= 4 or len(self.query_frames) >= 8

    def _should_update(self, frame: LocalFrame) -> bool:
        if self._last_update_pose is None:
            return True
        dx = frame.x - self._last_update_pose[0]
        dz = frame.z - self._last_update_pose[1]
        return math.hypot(dx, dz) >= MOVE_UPDATE_M

    def _query_points(self) -> np.ndarray:
        if not self.query_frames:
            return np.zeros((0, 2), dtype=np.float64)
        occ = build_query_occupancy(self.query_frames, resolution_m=self.resolution_m)
        pts = occ.occupied_points()
        if pts.shape[0] > MAX_QUERY_POINTS:
            idx = np.linspace(0, pts.shape[0] - 1, MAX_QUERY_POINTS).astype(np.int32)
            pts = pts[idx]
        return pts

    def _try_lock(self) -> None:
        spin = start_spin_frames(self.query_frames)
        use = spin if len(spin) >= 3 else self.query_frames
        result = estimate_ep1_to_ep0(self.memory_frames, use, resolution_m=self.resolution_m)
        pts = self._query_points()
        seed_score = _score_transform(result.transform, pts, self.memory_occ)
        if result.score < LOCK_SCORE and seed_score < LOCK_SCORE:
            return
        self._seed_particles(result.transform, spread_m=0.20, spread_yaw_deg=12.0)
        self._set_estimate(result.transform, max(result.score, seed_score))
        self.mode = "COARSE"
        self.lock_frame_count = len(self.query_frames)
        self._update_mode()

    def _refine(self) -> None:
        pts = self._query_points()
        if pts.shape[0] < 20:
            return
        scores = self._score_particles(pts)
        scores = scores - scores.max()
        weights = np.exp(scores / 0.08)
        weights_sum = float(weights.sum()) + 1e-12
        self.weights = weights / weights_sum
        self._resample_if_needed()
        mean = self._cluster_mean()
        refined, refined_score = _search_window(
            mean,
            pts,
            self.memory_occ,
            trans_radius_m=REFINE_TRANS_M,
            yaw_radius_deg=REFINE_YAW_DEG,
            step_m=REFINE_STEP_M,
            step_deg=REFINE_STEP_DEG,
        )
        if refined_score < COLLAPSE_SCORE:
            self.mode = "UNLOCKED"
            self.score = float(refined_score)
            self._try_lock_from_recent()
            return
        self._seed_particles(refined, spread_m=0.08, spread_yaw_deg=4.0)
        self._set_estimate(refined, refined_score)
        self._update_mode()

    def _try_lock_from_recent(self) -> None:
        recent = self.query_frames[-8:]
        result = estimate_ep1_to_ep0(
            self.memory_frames, recent, resolution_m=self.resolution_m
        )
        pts = self._query_points()
        score = _score_transform(result.transform, pts, self.memory_occ)
        if score >= LOCK_SCORE:
            self._seed_particles(result.transform, spread_m=0.20, spread_yaw_deg=12.0)
            self._set_estimate(result.transform, score)
            self.mode = "COARSE"
            if self.lock_frame_count is None:
                self.lock_frame_count = len(self.query_frames)

    def _score_particles(self, pts: np.ndarray) -> np.ndarray:
        occ = self.memory_occ
        packed = self.packed
        height, width = packed.shape
        origin_x, origin_z, res = occ.origin_x, occ.origin_z, occ.resolution_m
        n = self.particles.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            x, z, yaw = self.particles[i]
            cosine, sine = math.cos(yaw), math.sin(yaw)
            gx = cosine * pts[:, 0] + sine * pts[:, 1] + x
            gz = -sine * pts[:, 0] + cosine * pts[:, 1] + z
            cols = np.floor((gx - origin_x) / res).astype(np.int32)
            rows = np.floor((gz - origin_z) / res).astype(np.int32)
            inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            if not np.any(inside):
                out[i] = -UNKNOWN_PENALTY
                continue
            vals = packed[np.clip(rows, 0, height - 1), np.clip(cols, 0, width - 1)]
            vals = np.where(inside, vals, -UNKNOWN_PENALTY)
            out[i] = float(vals.mean())
        return out

    def _seed_particles(self, center: SE2, *, spread_m: float, spread_yaw_deg: float) -> None:
        rng = np.random.default_rng(0)
        trans = rng.normal(0.0, spread_m, size=(N_PARTICLES, 2))
        yaw = rng.normal(0.0, math.radians(spread_yaw_deg), size=N_PARTICLES)
        self.particles[:, 0] = center.x + trans[:, 0]
        self.particles[:, 1] = center.z + trans[:, 1]
        self.particles[:, 2] = np.array([wrap_angle(center.yaw + y) for y in yaw])
        self.weights[:] = 1.0 / N_PARTICLES

    def _resample_if_needed(self) -> None:
        ess = 1.0 / float(np.sum(self.weights ** 2) + 1e-12)
        if ess >= 0.5 * N_PARTICLES:
            return
        rng = np.random.default_rng(1)
        idx = rng.choice(N_PARTICLES, size=N_PARTICLES, p=self.weights)
        self.particles = self.particles[idx].copy()
        jitter = rng.normal(0.0, 0.02, size=(N_PARTICLES, 2))
        yaw_j = rng.normal(0.0, math.radians(1.0), size=N_PARTICLES)
        self.particles[:, 0] += jitter[:, 0]
        self.particles[:, 1] += jitter[:, 1]
        self.particles[:, 2] = np.array(
            [wrap_angle(self.particles[i, 2] + yaw_j[i]) for i in range(N_PARTICLES)]
        )
        self.weights[:] = 1.0 / N_PARTICLES

    def _cluster_mean(self) -> SE2:
        mode = int(np.argmax(self.weights))
        mx, mz, my = self.particles[mode]
        keep = []
        for i, (x, z, yaw) in enumerate(self.particles):
            if math.hypot(x - mx, z - mz) <= 0.6 and abs(wrap_angle(yaw - my)) <= math.radians(25.0):
                keep.append(i)
        if not keep:
            keep = [mode]
        w = self.weights[keep]
        w = w / (w.sum() + 1e-12)
        pts = self.particles[keep]
        x = float(np.dot(w, pts[:, 0]))
        z = float(np.dot(w, pts[:, 1]))
        cy = float(np.dot(w, np.cos(pts[:, 2])))
        sy = float(np.dot(w, np.sin(pts[:, 2])))
        yaw = math.atan2(sy, cy)
        trans = np.hypot(pts[:, 0] - x, pts[:, 1] - z)
        yaw_err = np.array([abs(wrap_angle(p - yaw)) for p in pts[:, 2]])
        self.trans_rms_m = float(np.sqrt(np.dot(w, trans ** 2)))
        self.yaw_rms_deg = float(math.degrees(np.sqrt(np.dot(w, yaw_err ** 2))))
        return SE2(x=x, z=z, yaw=wrap_angle(yaw))

    def _set_estimate(self, transform: SE2, score: float) -> None:
        self.transform = transform
        self.score = float(score)
        self._cluster_mean()

    def _update_mode(self) -> None:
        if self.mode == "UNLOCKED":
            return
        if self.trans_rms_m <= TIGHT_TRANS_M and math.radians(self.yaw_rms_deg) <= TIGHT_YAW_RAD:
            self.mode = "TIGHT"
        else:
            self.mode = "COARSE"
