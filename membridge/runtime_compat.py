"""Opt-in aliases that let the legacy warm trainer consume refactored modules."""

from __future__ import annotations

import sys
import types
import os
from typing import Any


def _module(name: str, values: dict[str, Any], *, package: bool = False) -> types.ModuleType:
    result = types.ModuleType(name)
    result.__dict__.update(values)
    if package:
        result.__path__ = []
    return result


def _world_memory_module() -> types.ModuleType:
    from membridge.occupancy_map import SparseWorldMetricMap, WorldGridSpec, WorldPlanningWindow
    from membridge.occupancy_map.persistence import COORDINATE_FRAME, WORLD_MAP_SCHEMA_VERSION
    from membridge.trajectory_graph import (
        TrajectoryGraph, TrajectoryNode, WorldMemoryRecorder,
        habitat_camera_to_world, habitat_depth_to_meters,
    )
    from membridge.trajectory_graph.persistence import TRAJECTORY_SCHEMA_VERSION

    return _module("vlnce_baselines.inherit_sg.world_metric_memory", {
        "COORDINATE_FRAME": COORDINATE_FRAME,
        "WORLD_MAP_SCHEMA_VERSION": WORLD_MAP_SCHEMA_VERSION,
        "TRAJECTORY_SCHEMA_VERSION": TRAJECTORY_SCHEMA_VERSION,
        "SparseWorldMetricMap": SparseWorldMetricMap,
        "TrajectoryGraph": TrajectoryGraph,
        "TrajectoryNode": TrajectoryNode,
        "WorldGridSpec": WorldGridSpec,
        "WorldPlanningWindow": WorldPlanningWindow,
        "WorldMemoryRecorder": WorldMemoryRecorder,
        "habitat_camera_to_world": habitat_camera_to_world,
        "habitat_depth_to_meters": habitat_depth_to_meters,
    })


def _instruction_module() -> types.ModuleType:
    from membridge.vlm_instruction_parser import (
        InstructionParseResult, InstructionParserError, InstructionSegment,
        InstructionVLMCallError, InstructionVLMResponseError,
        InvalidInstructionError, RECALL_SCOPE_BOTH_PRIOR_EPISODES,
        RECALL_SCOPE_STANDARD, ResolvedRecallQuery, VLMInstructionParser,
        resolve_recall_query,
    )
    from membridge.vlm_instruction_parser.models import VALID_RECALL_SCOPES
    from membridge.vlm_instruction_parser.parser import (
        coalesce_standard_terminal, extract_json_object, parse_segments,
    )
    from membridge.vlm_instruction_parser.prompts import INSTRUCTION_ROUTER_SYSTEM_PROMPT

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


def _reranker_module() -> types.ModuleType:
    from membridge.vlm_top3_selection import (
        CandidateImageError, CandidateImagePreparer,
        EpisodePairCandidateImagePreparer, InvalidCandidateError,
        PreparedCandidateImage, RerankResult, Top3RerankError,
        VLMCallError, VLMNoMatchError, VLMResponseError, VLMTop3Reranker,
        enrich_candidates_from_memory,
    )
    from membridge.vlm_top3_selection.prompts import (
        ALLOW_MULTIPLE_PROMPT, DUAL_EPISODE_FORCE_SINGLE_PROMPT,
        FORCE_SINGLE_PROMPT,
    )

    return _module("vlm_top3_retrieval.reranker", {
        "SYSTEM_PROMPT": ALLOW_MULTIPLE_PROMPT,
        "FORCE_SINGLE_SYSTEM_PROMPT": FORCE_SINGLE_PROMPT,
        "DUAL_EPISODE_FORCE_SINGLE_SYSTEM_PROMPT": DUAL_EPISODE_FORCE_SINGLE_PROMPT,
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


def _top3_package(
    instruction: types.ModuleType, reranker: types.ModuleType
) -> types.ModuleType:
    from membridge.vlm_instruction_parser import OpenAICompatibleVLMClient, VLMCompletion

    values = {
        name: value
        for source in (instruction, reranker)
        for name, value in source.__dict__.items()
        if not name.startswith("_")
    }
    values.update(
        OpenAICompatibleVLMClient=OpenAICompatibleVLMClient,
        VLMCompletion=VLMCompletion,
    )
    return _module("vlm_top3_retrieval", values, package=True)


def install_legacy_runtime_aliases() -> dict[str, str]:
    """Install aliases before the old trainer package is imported."""

    world = _world_memory_module()
    instruction = _instruction_module()
    reranker = _reranker_module()
    top3 = _top3_package(instruction, reranker)
    from membridge.navigation import approach
    from membridge.vlm_instruction_parser import client

    aliases = {
        "vlnce_baselines.inherit_sg.world_metric_memory": world,
        "vlnce_baselines.inherit_sg.post_hit_navigation": approach,
        "vlm_top3_retrieval": top3,
        "vlm_top3_retrieval.client": client,
        "vlm_top3_retrieval.instruction_parser": instruction,
        "vlm_top3_retrieval.reranker": reranker,
    }
    sys.modules.update(aliases)
    return {name: getattr(module, "__file__", "memory_module runtime alias") for name, module in aliases.items()}


def install_runtime_class_adapters() -> dict[str, str]:
    """Patch class-level contracts after the old trainer has been imported."""

    from membridge.sg_retrieval.runtime_adapter import install_online_adapter_retrieval
    from vlnce_baselines.inherit_sg.online_adapter import OnlineINHeritSGBuilder

    install_online_adapter_retrieval(OnlineINHeritSGBuilder)
    installed = {
        "OnlineINHeritSGBuilder.query_two_stage": "memory_module.sg_retrieval"
    }
    if os.environ.get(
        "MEMORY_MODULE_UNILAVIRA_ADAPTER_ENABLE", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}:
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
        from baselines.adapt.unilavira_runtime_adapter import (
            install_unilavira_runtime_adapters,
        )
        from vlnce_baselines.inherit_sg.trainer import (
            INHeritSGZeroShotVlnEvaluatorMP,
        )

        installed.update(
            install_unilavira_runtime_adapters(
                OnlineINHeritSGBuilder,
                INHeritSGZeroShotVlnEvaluatorMP,
                HabitatSimActions,
            )
        )
    return installed


def runtime_component_matrix() -> dict[str, str]:
    adapter_enabled = os.environ.get(
        "MEMORY_MODULE_UNILAVIRA_ADAPTER_ENABLE", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    return {
        "semantic_scene_graph": (
            "memory_module semantic memory adapter active; legacy perception backend retained"
            if adapter_enabled
            else "legacy TrajectoryPipeline retained (runtime adapter disabled)"
        ),
        "occupancy_map": "memory_module alias active",
        "trajectory_graph": "memory_module alias active",
        "vlm_instruction_parser": "memory_module alias active",
        "sg_retrieval": "memory_module class adapter active",
        "vlm_top3_selection": "memory_module alias active",
        "navigation": (
            "memory_module navigation kernel adapter active; Habitat lifecycle retained"
            if adapter_enabled
            else "memory_module post-hit approach; legacy trainer/controller retained"
        ),
        "exploration": "inactive in launch_warm.sh",
    }
