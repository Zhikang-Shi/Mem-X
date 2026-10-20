"""Auditable image and selection data contracts."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PreparedCandidateImage:
    candidate_index: int
    uid: str
    label: str
    room_label: str | None
    source_path: str
    source_kind: str
    image_mode: str
    bbox_xyxy: tuple[float, float, float, float] | None
    encoded_width: int
    encoded_height: int
    sha256: str
    data_url: str = field(repr=False)
    episode_sources: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def audit_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index, "uid": self.uid,
            "label": self.label, "room_label": self.room_label,
            "source_path": self.source_path, "source_kind": self.source_kind,
            "image_mode": self.image_mode,
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy else None,
            "encoded_width": self.encoded_width, "encoded_height": self.encoded_height,
            "sha256": self.sha256,
            "episode_sources": copy.deepcopy(dict(self.episode_sources)),
        }


@dataclass(frozen=True)
class RerankResult:
    selected_index: int
    selected_uid: str
    selected_candidate: Mapping[str, Any]
    confidence: float
    reason: str
    model: str | None
    usage: Mapping[str, Any]
    request_id: str | None
    prepared_images: Sequence[PreparedCandidateImage]
    raw_response: str
    vlm_candidate_indices: Sequence[int] = field(default_factory=tuple)
    vlm_candidate_uids: Sequence[str] = field(default_factory=tuple)
    selection_policy: str = "vlm_unique"
    used_fallback: bool = False
    fallback_reason: str | None = None

    def to_dict(self, *, include_candidate: bool = True, include_raw_response: bool = False) -> dict[str, Any]:
        result = {
            "selected_index": self.selected_index, "selected_uid": self.selected_uid,
            "confidence": self.confidence, "reason": self.reason, "model": self.model,
            "usage": dict(self.usage), "request_id": self.request_id,
            "vlm_candidate_indices": list(self.vlm_candidate_indices),
            "vlm_candidate_uids": list(self.vlm_candidate_uids),
            "vlm_returned_multiple": len(self.vlm_candidate_indices) > 1,
            "selection_policy": self.selection_policy, "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
            "prepared_images": [image.audit_dict() for image in self.prepared_images],
        }
        if include_candidate:
            result["selected_candidate"] = copy.deepcopy(dict(self.selected_candidate))
        if include_raw_response:
            result["raw_response"] = self.raw_response
        return result
