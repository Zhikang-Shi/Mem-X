"""DINO-style accepted-keyframe filtering."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or not a.size:
        raise ValueError("feature vectors must have the same non-empty shape")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator <= 1.0e-12 else float(np.dot(a, b) / denominator)


class KeyframeSelector:
    """Compare each frame with the last *accepted* keyframe."""

    def __init__(self, threshold: float = 0.85) -> None:
        if not -1.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be in [-1, 1]")
        self.threshold = float(threshold)
        self.last_feature: np.ndarray | None = None

    def accept(self, feature: Sequence[float]) -> tuple[bool, float | None]:
        current = np.asarray(feature, dtype=np.float64).reshape(-1)
        if not current.size or not np.isfinite(current).all():
            raise ValueError("keyframe feature must be finite and non-empty")
        if self.last_feature is None:
            self.last_feature = current.copy()
            return True, None
        similarity = cosine_similarity(current, self.last_feature)
        accepted = similarity < self.threshold
        if accepted:
            self.last_feature = current.copy()
        return accepted, similarity

    def reset(self) -> None:
        self.last_feature = None
