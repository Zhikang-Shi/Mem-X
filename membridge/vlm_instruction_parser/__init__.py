"""VLM instruction routing into memory-recall and ordinary VLN spans."""

from .client import OpenAICompatibleVLMClient, VLMCompletion
from .models import (
    InstructionParseResult,
    InstructionSegment,
    RECALL_SCOPE_BOTH_PRIOR_EPISODES,
    RECALL_SCOPE_STANDARD,
    ResolvedRecallQuery,
)
from .parser import (
    InstructionParserError,
    InstructionVLMCallError,
    InstructionVLMResponseError,
    InvalidInstructionError,
    VLMInstructionParser,
)
from .resolver import resolve_recall_query
from .runtime import InstructionRuntimeConfig

__all__ = [
    "InstructionParseResult", "InstructionParserError", "InstructionSegment",
    "InstructionVLMCallError", "InstructionVLMResponseError",
    "InvalidInstructionError", "OpenAICompatibleVLMClient",
    "RECALL_SCOPE_BOTH_PRIOR_EPISODES", "RECALL_SCOPE_STANDARD",
    "ResolvedRecallQuery", "VLMCompletion", "VLMInstructionParser",
    "resolve_recall_query",
    "InstructionRuntimeConfig",
]
