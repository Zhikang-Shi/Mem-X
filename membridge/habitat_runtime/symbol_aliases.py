"""Names expected by the reviewed warm lifecycle, backed by new modules."""

from __future__ import annotations

import sys
import types
from typing import Any


def _module(name: str, values: dict[str, Any], *, package: bool = False):
    result = types.ModuleType(name)
    result.__dict__.update(values)
    if package:
        result.__path__ = []
    return result


def _instruction_module():
    from membridge.vlm_instruction_parser import (
        InstructionParseResult,
        InstructionParserError,
        InstructionSegment,
        InstructionVLMCallError,
        InstructionVLMResponseError,
        InvalidInstructionError,
        RECALL_SCOPE_BOTH_PRIOR_EPISODES,
        RECALL_SCOPE_STANDARD,
        ResolvedRecallQuery,
        VLMInstructionParser,
        resolve_recall_query,
    )
    from membridge.vlm_instruction_parser.models import VALID_RECALL_SCOPES
    from membridge.vlm_instruction_parser.parser import (
        coalesce_standard_terminal,
        extract_json_object,
        parse_segments,
    )
    from membridge.vlm_instruction_parser.prompts import (
        INSTRUCTION_ROUTER_SYSTEM_PROMPT,
    )

    return _module("vlm_top3_retrieval.instruction_parser", {
        "INSTRUCTION_ROUTER_SYSTEM_PROMPT": INSTRUCTION_ROUTER_SYSTEM_PROMPT,
        "RECALL_SCOPE_BOTH_PRIOR_EPISODES": RECALL_SCOPE_BOTH_PRIOR_EPISODES,
        "RECALL_SCOPE_STANDARD": RECALL_SCOPE_STANDARD,
        "VALID_RECALL_SCOPES": VALID_RECALL_SCOPES,
        "InstructionParseResult": InstructionParseResult,
        "InstructionParserError": InstructionParserError,
        "InstructionSegment": InstructionSegment,
        "InstructionVLMCallError": InstructionVLMCallError,
        "InstructionVLMResponseError": InstructionVLMResponseError,
        "InvalidInstructionError": InvalidInstructionError,
        "ResolvedRecallQuery": ResolvedRecallQuery,
        "VLMInstructionParser": VLMInstructionParser,
        "resolve_recall_query": resolve_recall_query,
        "_coalesce_standard_recall_terminals": coalesce_standard_terminal,
        "_extract_json_object": extract_json_object,
        "_parse_segments": parse_segments,
    })


def _reranker_module():
    from membridge.vlm_top3_selection import (
        CandidateImageError,
        CandidateImagePreparer,
        EpisodePairCandidateImagePreparer,
        InvalidCandidateError,
        PreparedCandidateImage,
        RerankResult,
        Top3RerankError,
        VLMCallError,
        VLMNoMatchError,
        VLMResponseError,
        VLMTop3Reranker,
        enrich_candidates_from_memory,
    )
    from membridge.vlm_top3_selection.prompts import (
        ALLOW_MULTIPLE_PROMPT,
        DUAL_EPISODE_FORCE_SINGLE_PROMPT,
        FORCE_SINGLE_PROMPT,
    )

    return _module("vlm_top3_retrieval.reranker", {
        "SYSTEM_PROMPT": ALLOW_MULTIPLE_PROMPT,
        "FORCE_SINGLE_SYSTEM_PROMPT": FORCE_SINGLE_PROMPT,
        "DUAL_EPISODE_FORCE_SINGLE_SYSTEM_PROMPT": (
            DUAL_EPISODE_FORCE_SINGLE_PROMPT
        ),
        "CandidateImageError": CandidateImageError,
        "CandidateImagePreparer": CandidateImagePreparer,
        "EpisodePairCandidateImagePreparer": EpisodePairCandidateImagePreparer,
        "InvalidCandidateError": InvalidCandidateError,
        "PreparedCandidateImage": PreparedCandidateImage,
        "RerankResult": RerankResult,
        "Top3RerankError": Top3RerankError,
        "VLMCallError": VLMCallError,
        "VLMNoMatchError": VLMNoMatchError,
        "VLMResponseError": VLMResponseError,
        "VLMTop3Reranker": VLMTop3Reranker,
        "enrich_candidates_from_memory": enrich_candidates_from_memory,
    })


def install_runtime_symbol_aliases() -> dict[str, str]:
    """Expose old neutral library names without importing the old runtime."""

    from membridge.vlm_instruction_parser import (
        OpenAICompatibleVLMClient,
        VLMCompletion,
    )
    from membridge.vlm_instruction_parser import client

    instruction = _instruction_module()
    reranker = _reranker_module()
    public = {
        name: value
        for source in (instruction, reranker)
        for name, value in source.__dict__.items()
        if not name.startswith("_")
    }
    public.update(
        OpenAICompatibleVLMClient=OpenAICompatibleVLMClient,
        VLMCompletion=VLMCompletion,
    )
    package = _module("vlm_top3_retrieval", public, package=True)
    aliases = {
        "vlm_top3_retrieval": package,
        "vlm_top3_retrieval.client": client,
        "vlm_top3_retrieval.instruction_parser": instruction,
        "vlm_top3_retrieval.reranker": reranker,
    }
    sys.modules.update(aliases)
    return {name: "memory_module" for name in aliases}
