"""Validated runtime settings for the three-candidate VLM selector."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .images import CandidateImagePreparer
from .reranker import VLMTop3Reranker


@dataclass(frozen=True)
class Top3RuntimeConfig:
    fallback_policy: str = "raise"
    selection_mode: str = "force_single"
    image_detail: str = "high"
    max_tokens: int = 2048

    def __post_init__(self) -> None:
        if self.fallback_policy not in {"raise", "top1"}:
            raise ValueError("fallback_policy must be raise or top1")
        if self.selection_mode not in {"allow_multiple", "force_single"}:
            raise ValueError("invalid VLM Top3 selection mode")
        if self.image_detail not in {"low", "high", "auto"} or int(self.max_tokens) < 1:
            raise ValueError("invalid VLM Top3 image detail or token limit")

    @classmethod
    def from_env(cls) -> "Top3RuntimeConfig":
        return cls(
            fallback_policy=os.environ.get("LIFELONG_SG_VLM_TOP3_FALLBACK_POLICY", "raise"),
            selection_mode=os.environ.get("LIFELONG_SG_VLM_TOP3_SELECTION_MODE", "force_single"),
            image_detail=os.environ.get("LIFELONG_SG_VLM_TOP3_IMAGE_DETAIL", "high"),
            max_tokens=int(os.environ.get("LIFELONG_SG_VLM_TOP3_MAX_TOKENS", "2048")),
        )

    def build(
        self, client: Any, *, image_preparer: CandidateImagePreparer | None = None
    ) -> VLMTop3Reranker:
        return VLMTop3Reranker(
            client, image_preparer=image_preparer,
            fallback_policy=self.fallback_policy,
            selection_mode=self.selection_mode,
            image_detail=self.image_detail, max_tokens=self.max_tokens,
        )
