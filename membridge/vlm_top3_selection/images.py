"""Resolve, mark and encode one image per SG candidate."""

from __future__ import annotations

import base64
import hashlib
import io
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .models import PreparedCandidateImage


class CandidateImageError(RuntimeError):
    pass


def candidate_uid(candidate: Mapping[str, object]) -> str:
    uid = str(candidate.get("uid") or candidate.get("node_id") or "").strip()
    if not uid:
        raise CandidateImageError("candidate has no uid or node_id")
    return uid[4:] if uid.startswith("uid:") else uid


def candidate_bbox(candidate: Mapping[str, object]) -> tuple[float, float, float, float] | None:
    best = candidate.get("best_observation")
    values = best.get("bbox_xyxy") if isinstance(best, Mapping) else None
    values = values or candidate.get("bbox_xyxy") or candidate.get("bbox")
    if not isinstance(values, (list, tuple)) or len(values) < 4:
        return None
    try:
        result = tuple(float(value) for value in values[:4])
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(value) for value in result) else None


class CandidateImagePreparer:
    def __init__(
        self, *, session_dir: str | None = None, image_mode: str = "frame_box",
        crop_padding: float = 0.18, max_side: int = 1280, jpeg_quality: int = 90,
    ) -> None:
        if image_mode not in {"frame_box", "crop"}:
            raise ValueError("image_mode must be frame_box or crop")
        if not 0 <= crop_padding <= 1 or max_side < 128 or not 40 <= jpeg_quality <= 100:
            raise ValueError("invalid image preparation settings")
        self.session_dir = Path(session_dir).resolve() if session_dir else None
        self.image_mode, self.crop_padding = image_mode, float(crop_padding)
        self.max_side, self.jpeg_quality = int(max_side), int(jpeg_quality)

    def _path_candidates(self, value: str) -> Iterable[Path]:
        path, emitted = Path(value), set()
        options = [path]
        if self.session_dir and not path.is_absolute():
            options.append(self.session_dir / path)
        marker = "/lifelong_session/"
        if self.session_dir and marker in value.replace("\\", "/"):
            options.append(self.session_dir / value.replace("\\", "/").split(marker, 1)[1])
        for option in options:
            if str(option) not in emitted:
                emitted.add(str(option))
                yield option

    def _resolve_image(self, candidate: Mapping[str, object]) -> tuple[Path, str]:
        best = candidate.get("best_observation")
        best = best if isinstance(best, Mapping) else {}
        options = [
            (best.get("rgb_path"), "best_observation.rgb_path"),
            (candidate.get("full_image_path"), "full_image_path"),
            (candidate.get("image_path"), "image_path"),
            (candidate.get("cropped_image_path"), "cropped_image_path"),
        ]
        attempted = []
        for raw, kind in options:
            if raw:
                for path in self._path_candidates(str(raw)):
                    attempted.append(str(path))
                    if path.is_file():
                        return path.resolve(), kind
        if self.session_dir and best.get("frame_idx") is not None:
            try:
                path = self.session_dir / "frames" / "rgb" / f"{int(best['frame_idx']):06d}.png"
                attempted.append(str(path))
                if path.is_file():
                    return path.resolve(), "session_dir_frame_idx"
            except (TypeError, ValueError):
                pass
        raise CandidateImageError(f"no readable image for {candidate_uid(candidate)}; attempted {attempted}")

    @staticmethod
    def _clip(bbox: tuple[float, ...], size: tuple[int, int]) -> tuple[float, float, float, float] | None:
        width, height = size
        x1, y1 = max(0, min(width - 1, bbox[0])), max(0, min(height - 1, bbox[1]))
        x2, y2 = max(0, min(width, bbox[2])), max(0, min(height, bbox[3]))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def _render(
        self, image: Image.Image, bbox: tuple[float, float, float, float] | None,
        index: int, uid: str, title_suffix: str = "",
    ) -> Image.Image:
        if self.image_mode == "crop" and bbox:
            x1, y1, x2, y2 = bbox
            px, py = max(8, (x2 - x1) * self.crop_padding), max(8, (y2 - y1) * self.crop_padding)
            image = image.crop((max(0, int(x1 - px)), max(0, int(y1 - py)), min(image.width, int(x2 + px)), min(image.height, int(y2 + py))))
            bbox = None
        result = image.copy()
        draw = ImageDraw.Draw(result)
        width = max(4, round(min(result.size) * 0.008))
        draw.rectangle(bbox or (1, 1, result.width - 2, result.height - 2), outline=(255, 0, 0), width=width)
        title_height = max(24, round(result.height * 0.05))
        draw.rectangle((0, 0, result.width, title_height), fill=(0, 0, 0))
        draw.text((8, 6), f"CANDIDATE {index} | {uid}{title_suffix}", fill="white", font=ImageFont.load_default())
        return result

    def _encode(
        self, image: Image.Image, *, index: int, uid: str, label: str,
        room_label: str | None, source_path: str, source_kind: str,
        bbox: tuple[float, float, float, float] | None, image_mode: str | None = None,
        episode_sources: Mapping[str, Mapping[str, object]] | None = None,
    ) -> PreparedCandidateImage:
        image.thumbnail((self.max_side, self.max_side), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=self.jpeg_quality, optimize=True)
        raw = output.getvalue()
        return PreparedCandidateImage(
            index, uid, label, room_label, source_path, source_kind,
            image_mode or self.image_mode, bbox, image.width, image.height,
            hashlib.sha256(raw).hexdigest(),
            "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii"),
            episode_sources or {},
        )

    def prepare(self, candidate: Mapping[str, object], candidate_index: int) -> PreparedCandidateImage:
        uid, (path, kind) = candidate_uid(candidate), self._resolve_image(candidate)
        with Image.open(path) as loaded:
            image = loaded.convert("RGB")
        bbox = candidate_bbox(candidate)
        bbox = self._clip(bbox, image.size) if bbox else None
        rendered = self._render(image, bbox, candidate_index, uid)
        return self._encode(
            rendered, index=candidate_index, uid=uid,
            label=str(candidate.get("label") or ""),
            room_label=str(candidate["room_label"]) if candidate.get("room_label") is not None else None,
            source_path=str(path), source_kind=kind, bbox=bbox,
        )
