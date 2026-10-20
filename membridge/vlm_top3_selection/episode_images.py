"""Two-prior-episode evidence panels for Level-3 recall."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PIL import Image

from .images import CandidateImageError, CandidateImagePreparer, candidate_uid
from .models import PreparedCandidateImage


class EpisodePairCandidateImagePreparer(CandidateImagePreparer):
    def __init__(
        self, *, session_dir: str | None = None,
        episode_ids: Sequence[str] = ("ep0", "ep1"), max_side: int = 1280,
        jpeg_quality: int = 90,
    ) -> None:
        super().__init__(session_dir=session_dir, max_side=max_side, jpeg_quality=jpeg_quality)
        self.episode_ids = tuple(str(value).strip() for value in episode_ids)
        if len(self.episode_ids) != 2 or len(set(self.episode_ids)) != 2 or not all(self.episode_ids):
            raise ValueError("episode_ids must contain two unique nonempty IDs")

    def _episode_path(self, observation: Mapping[str, object], uid: str, episode_id: str):
        raw = observation.get("rgb_path")
        if raw:
            for path in self._path_candidates(str(raw)):
                if path.is_file():
                    return path.resolve(), f"best_observation_by_episode.{episode_id}.rgb_path"
        if self.session_dir and observation.get("frame_idx") is not None:
            path = self.session_dir / "frames" / "rgb" / f"{int(observation['frame_idx']):06d}.png"
            if path.is_file():
                return path.resolve(), f"best_observation_by_episode.{episode_id}.frame_idx"
        raise CandidateImageError(f"no readable {episode_id} keyframe for candidate {uid}")

    def prepare(self, candidate: Mapping[str, object], candidate_index: int) -> PreparedCandidateImage:
        uid = candidate_uid(candidate)
        by_episode = candidate.get("best_observation_by_episode")
        if not isinstance(by_episode, Mapping):
            raise CandidateImageError(f"candidate {uid} has no best_observation_by_episode")
        panels, audit = [], {}
        for episode_id in self.episode_ids:
            observation = by_episode.get(episode_id)
            if not isinstance(observation, Mapping):
                raise CandidateImageError(f"candidate {uid} has no keyframe for {episode_id}")
            path, kind = self._episode_path(observation, uid, episode_id)
            with Image.open(path) as loaded:
                image = loaded.convert("RGB")
            values = observation.get("bbox_xyxy") or observation.get("bbox")
            bbox = tuple(map(float, values[:4])) if isinstance(values, (list, tuple)) and len(values) >= 4 else None
            bbox = self._clip(bbox, image.size) if bbox else None
            panels.append(self._render(image, bbox, candidate_index, uid, f" | {episode_id}"))
            audit[episode_id] = {
                "source_path": str(path), "source_kind": kind,
                "frame_idx": observation.get("frame_idx"),
                "bbox_xyxy": list(bbox) if bbox else None,
                "runtime_episode_id": observation.get("runtime_episode_id", episode_id),
                "source_episode_id": observation.get("source_episode_id"),
            }
        panel_width = max(128, (self.max_side - 8) // 2)
        for panel in panels:
            panel.thumbnail((panel_width, self.max_side), Image.Resampling.LANCZOS)
        composite = Image.new("RGB", (sum(panel.width for panel in panels) + 8, max(panel.height for panel in panels)), (24, 24, 24))
        cursor = 0
        for panel in panels:
            composite.paste(panel, (cursor, 0))
            cursor += panel.width + 8
        return self._encode(
            composite, index=candidate_index, uid=uid, label=str(candidate.get("label") or ""),
            room_label=str(candidate["room_label"]) if candidate.get("room_label") is not None else None,
            source_path=" | ".join(audit[episode]["source_path"] for episode in self.episode_ids),
            source_kind="best_observation_by_episode", bbox=None,
            image_mode="episode_pair_frame_box", episode_sources=audit,
        )
