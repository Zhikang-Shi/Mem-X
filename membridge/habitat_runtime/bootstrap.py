"""Install the memory-only warm Habitat runtime and its module adapters."""

from __future__ import annotations

import os
from typing import Mapping


MEMORY_WARM_TRAINER = "ZS-Evaluator-mp-memory-module-warm"


_ENVIRONMENT_ALIASES: Mapping[str, str] = {
    "MEMORY_MODULE_ENABLE": "INHERIT_SG_ENABLE",
    "MEMORY_MODULE_OUTPUT_DIR": "INHERIT_SG_OUTPUT_DIR",
    "MEMORY_MODULE_SESSION_NAME": "INHERIT_SG_SESSION_NAME",
    "MEMORY_MODULE_PRELOAD_ROOT": "INHERIT_SG_PRELOAD_ROOT",
    "MEMORY_MODULE_PRELOAD_DIR": "INHERIT_SG_PRELOAD_DIR",
    "MEMORY_MODULE_EPISODE_MEMORY_MAP": "INHERIT_SG_EPISODE_MEMORY_MAP",
    "MEMORY_MODULE_PRELOAD_MEMORY_STAGE": "INHERIT_SG_PRELOAD_MEMORY_STAGE",
    "MEMORY_MODULE_PRELOAD_BUILDER_STATE_REQUIRED": (
        "INHERIT_SG_PRELOAD_BUILDER_STATE_REQUIRED"
    ),
    "MEMORY_MODULE_PRELOAD_FINALIZED_EPISODES": (
        "INHERIT_SG_PRELOAD_FINALIZED_EPISODES"
    ),
    "MEMORY_MODULE_PRELOAD_QUERY_ONLY": "INHERIT_SG_PRELOAD_QUERY_ONLY",
    "MEMORY_MODULE_PRELOAD_WORLD_MAP_ONLY": (
        "INHERIT_SG_PRELOAD_WORLD_MAP_ONLY"
    ),
    "MEMORY_MODULE_PRELOADED_OBJECT_MATCH_GUARD_M": (
        "INHERIT_SG_PRELOADED_OBJECT_MATCH_GUARD_M"
    ),
    "MEMORY_MODULE_TEXT_ENCODER": "INHERIT_SG_TEXT_ENCODER",
    "MEMORY_MODULE_DINOV3_MODEL": "INHERIT_SG_DINOV3_MODEL",
    "MEMORY_MODULE_DINOV3_REPO": "INHERIT_SG_DINOV3_REPO",
    "MEMORY_MODULE_SAM3_CKPT": "INHERIT_SG_SAM3_CKPT",
    "MEMORY_MODULE_SAM3_SOURCE_ROOT": "INHERIT_SG_SAM3_SOURCE_ROOT",
    "MEMORY_MODULE_QUERY_VOCABULARY": "INHERIT_SG_QUERY_VOCABULARY",
    "MEMORY_MODULE_SAM3_CATEGORY_BATCH_SIZE": (
        "INHERIT_SG_SAM3_CATEGORY_BATCH_SIZE"
    ),
    "MEMORY_MODULE_SAM3_CATEGORY_BATCH_SIZE_BY_DEVICE": (
        "INHERIT_SG_SAM3_CATEGORY_BATCH_SIZE_BY_DEVICE"
    ),
    "MEMORY_MODULE_STRIDE": "INHERIT_SG_STRIDE",
    "MEMORY_MODULE_KEYFRAME_FILTER": "INHERIT_SG_KEYFRAME_FILTER",
    "MEMORY_MODULE_ASYNC": "INHERIT_SG_ASYNC",
    "MEMORY_MODULE_FAIL_CLOSED": "INHERIT_SG_FAIL_CLOSED",
    "MEMORY_MODULE_SHARE_MODELS": "INHERIT_SG_SHARE_MODELS",
    "MEMORY_MODULE_SAVE_CONFIRMED_KEYFRAMES": (
        "INHERIT_SG_SAVE_CONFIRMED_KEYFRAMES"
    ),
    "MEMORY_MODULE_DEBUG": "INHERIT_SG_DEBUG",
    "MEMORY_MODULE_SAVE_OFFLINE_RGBD": "INHERIT_SG_SAVE_OFFLINE_RGBD",
    "MEMORY_MODULE_OFFLINE_RGBD_DIR": "INHERIT_SG_OFFLINE_RGBD_DIR",
}


def install_environment_aliases() -> dict[str, str]:
    """Translate public memory runtime settings for retained lifecycle code."""

    installed = {}
    for public_name, internal_name in _ENVIRONMENT_ALIASES.items():
        value = os.environ.get(public_name)
        if value is not None:
            os.environ[internal_name] = value
            installed[internal_name] = public_name
    source_root = os.environ.get("MEMORY_MODULE_SAM3_SOURCE_ROOT")
    if source_root:
        os.environ["SAM3_SOURCE_ROOT"] = source_root
    return installed


def install_runtime() -> dict[str, object]:
    """Register the memory-only trainer and all eight capability adapters."""

    aliases = install_environment_aliases()
    from .symbol_aliases import install_runtime_symbol_aliases

    module_aliases = install_runtime_symbol_aliases()
    from membridge.sg_retrieval.runtime_adapter import (
        install_online_adapter_retrieval,
    )
    from baselines.adapt.unilavira_runtime_adapter import (
        install_unilavira_runtime_adapters,
    )
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    from .online_adapter import OnlineINHeritSGBuilder
    from .trainer import INHeritSGZeroShotVlnEvaluatorMP

    install_online_adapter_retrieval(OnlineINHeritSGBuilder)
    adapters = install_unilavira_runtime_adapters(
        OnlineINHeritSGBuilder,
        INHeritSGZeroShotVlnEvaluatorMP,
        HabitatSimActions,
    )
    return {
        "trainer": MEMORY_WARM_TRAINER,
        "environment_aliases": aliases,
        "module_aliases": module_aliases,
        "class_adapters": adapters,
    }
