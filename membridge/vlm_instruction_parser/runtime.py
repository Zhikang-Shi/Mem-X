"""Environment-backed construction for the instruction parsing module."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .client import OpenAICompatibleVLMClient
from .parser import VLMInstructionParser


@dataclass(frozen=True)
class InstructionRuntimeConfig:
    base_url: str
    model: str
    api_key: str = ""
    timeout_s: float = 90.0
    max_retries: int = 2
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if not str(self.base_url).strip() or not str(self.model).strip():
            raise ValueError("base_url and model are required")
        if float(self.timeout_s) <= 0 or int(self.max_retries) < 0 or int(self.max_tokens) < 1:
            raise ValueError("invalid instruction VLM timeout, retries, or token limit")

    @classmethod
    def from_env(cls) -> "InstructionRuntimeConfig":
        base_url = os.environ.get("SG_VLM_BASE_URL") or os.environ.get("VA_BASE_URL") or ""
        model = os.environ.get("SG_VLM_MODEL") or os.environ.get("VA_MODEL_NAME") or ""
        if not base_url or not model:
            raise ValueError("set SG_VLM_BASE_URL and SG_VLM_MODEL")
        return cls(
            base_url=base_url,
            model=model,
            api_key=os.environ.get("SG_VLM_API_KEY", os.environ.get("VA_API_KEY", "")),
            timeout_s=float(os.environ.get("SG_VLM_TIMEOUT_S", "90")),
            max_retries=int(os.environ.get("SG_VLM_MAX_RETRIES", "2")),
            max_tokens=int(os.environ.get("LIFELONG_SG_VLM_ROUTER_MAX_TOKENS", "1024")),
        )

    def build(self) -> VLMInstructionParser:
        client = OpenAICompatibleVLMClient(
            base_url=self.base_url, api_key=self.api_key, model=self.model,
            timeout_s=self.timeout_s, max_retries=self.max_retries,
            json_mode=True,
        )
        return VLMInstructionParser(client, max_tokens=self.max_tokens)
