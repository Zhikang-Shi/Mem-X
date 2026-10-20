#!/usr/bin/env python3
"""Run JanusVLN Level-3 EP2 from frozen memory_module M1.

LaViRA's already saved Level-3 instruction parse is reused; the instruction
router is never called.  VLM Top3 still runs against this run's M1 scene
graph.  Dual-anchor SG navigation is executed in order; JanusVLN does not
choose actions during SG, but every RGB frame is kept in its visual history
so the first continuation call sees the path to the anchors.  Handoff matches
LaViRA Level-3: only unfinished clauses plus the type-2 suffix are given to
Janus.  If Anchor 1 is reached and Anchor 2 fails, Janus receives Anchor 2's
recall text plus the suffix, not the full original instruction.

M1 semantic SG stays query-only.  Live RGB-D updates only the metric map and
trajectory, published as M2.  SAM3/DINOv3 are not loaded.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import json
import logging
import math
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
LEVEL2 = HERE
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LEVEL2))
import run_ep1_memory_janus as common  # noqa: E402
DEFAULT_JANUS_ROOT = Path("/home/wangyj/projects/janusvln_hm3dsem_benchmark/src/JanusVLN")
DEFAULT_EXP_ROOT = DEFAULT_JANUS_ROOT.parents[1]
DEFAULT_DATASET = DEFAULT_EXP_ROOT / "data/level3_v34_20260906/habitat/level3/level3.json.gz"
DEFAULT_CONFIG = DEFAULT_JANUS_ROOT / "config/vln_level3_protocol.yaml"
DEFAULT_MODEL = DEFAULT_EXP_ROOT / "models/JanusVLN_Extra"
DEFAULT_M1_ROOT = HERE / "outputs/m0_m1_memory_from_poses_full"
DEFAULT_PARSE_ROOT = Path(
    "/share/shared/shizk/LifeLongHomeNav/level3/outputs/"
    "final_level3_full_20260906/memories"
)
DEFAULT_OUTPUT = Path(
    "/share/shared/shizk/level3_outputs/janus_memory_module/ep2_memory_janus"
)
EXPECTED_TASKS = 157
CORE_M1_FILES = common.CORE_M0_FILES
LOG = logging.getLogger("janus-level3-ep2")
NAVIGATION_CONTROLLER_VERSION = common.NAVIGATION_CONTROLLER_VERSION
PLAN_RELATIVE = (
    "memory_module/M2/memory_module/level3_dual_anchor_recall_plan.json",
    "stages/M2/memory_module/level3_dual_anchor_recall_plan.json",
    "stages/M2/level3_dual_anchor_recall_plan.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--janus-root", type=Path, default=DEFAULT_JANUS_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--m1-root", type=Path, default=DEFAULT_M1_ROOT)
    parser.add_argument(
        "--lavira-parse-root",
        type=Path,
        action="append",
        default=[],
        help="LaViRA Level-3 catalog or worker root containing dual-anchor plans",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-gpu-id", type=int, default=0)
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--session-id", action="append", default=[])
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--worker-id", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="HOST_PREFIX=RUNTIME_PREFIX",
    )
    parser.add_argument(
        "--query-vocabulary",
        type=Path,
        default=(
            REPO_ROOT
            / "memory_module/semantic_scene_graph/config/"
            "level1_runtime_query_vocabulary_75.json"
        ),
    )
    parser.add_argument(
        "--recall-policy",
        default="any_memory",
        choices=("any_memory", "both_prior_episodes"),
    )
    args = parser.parse_args()
    if not args.lavira_parse_root:
        args.lavira_parse_root = [DEFAULT_PARSE_ROOT]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.max_steps < 1 or args.num_history < 0:
        parser.error("--max-steps must be positive and --num-history non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def parse_from_plan(payload: Mapping[str, Any], path: Path):
    from membridge.vlm_instruction_parser import InstructionParseResult, InstructionSegment
    from membridge.habitat_runtime.level3_dual_anchor import validate_level3_parse

    if payload.get("schema_version") != "memory_module_level3_dual_anchor_plan_v1":
        raise RuntimeError(f"unsupported dual-anchor schema: {path}")
    raw = payload.get("instruction_parse")
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"dual-anchor plan has no instruction_parse: {path}")
    segments = []
    for item in raw.get("segments") or []:
        if int(item.get("type", -1)) == 1:
            segments.append(
                InstructionSegment(
                    type=1,
                    text=common.normalize_text(item.get("text")),
                    label=common.normalize_text(item.get("label")) or None,
                    room=common.normalize_text(item.get("room")) or None,
                    recall_scope=item.get("recall_scope"),
                )
            )
        else:
            segments.append(
                InstructionSegment(
                    type=2,
                    text=common.normalize_text(item.get("text")),
                )
            )
    parsed = InstructionParseResult(
        instruction=common.normalize_text(raw.get("instruction")),
        segments=tuple(segments),
        model=raw.get("model"),
        usage=raw.get("usage") or {},
        request_id=raw.get("request_id"),
        raw_response=str(raw.get("raw_response") or ""),
    )
    validate_level3_parse(parsed)
    return parsed


def discover_plan_file(root: Path, task_id: str) -> Path | None:
    root = root.resolve()
    direct = [
        root / task_id / relative
        for relative in PLAN_RELATIVE
    ]
    direct.append(root / "sessions" / task_id / "stages/M2/memory_module/level3_dual_anchor_recall_plan.json")
    for path in direct:
        if path.is_file():
            return path
    workers = root / "workers"
    if workers.is_dir():
        for worker in workers.iterdir():
            path = (
                worker
                / "sessions"
                / task_id
                / "stages/M2/memory_module/level3_dual_anchor_recall_plan.json"
            )
            if path.is_file():
                return path
    if (root / "arms").is_dir():
        for arm in (root / "arms").iterdir():
            found = discover_plan_file(arm, task_id)
            if found is not None:
                return found
    return None


def load_lavira_parse(roots: Sequence[Path], task_id: str, mappings) -> dict[str, Any] | None:
    for root in roots:
        path = discover_plan_file(common.translated_path(root, mappings), task_id)
        if path is None:
            continue
        parsed = parse_from_plan(common.read_json(path), path)
        return {
            "task_id": task_id,
            "instruction": parsed.instruction,
            "types": [segment.type for segment in parsed.segments],
            "anchors": [segment.to_dict() for segment in parsed.segments[:2]],
            "suffix": parsed.segments[2].text,
            "source": str(path.resolve()),
            "source_sha256": common.sha256_file(path),
            "parsed": parsed,
        }
    return None


def dataset_tasks(dataset_path: Path) -> dict[str, dict[str, Any]]:
    payload = common.read_json(dataset_path)
    tasks: dict[str, dict[str, Any]] = {}
    for episode in payload.get("episodes") or []:
        info = episode.get("info") or {}
        if str(info.get("lifelong_role") or info.get("level3_role")) != "ep2":
            continue
        episode_id = str(getattr(episode, "episode_id", "") or "")
        task_id = episode_id.rsplit("_", 1)[0] if episode_id.endswith(("_ep0", "_ep1", "_ep2", "_ep3", "_ep4")) else str(info.get("session_id") or "")
        if not task_id or task_id in tasks:
            raise RuntimeError(f"invalid Level-3 EP2 task: {task_id!r}")
        tasks[task_id] = {
            "episode_id": str(episode["episode_id"]),
            "session_id": task_id,
            "instruction": common.normalize_text(
                (episode.get("instruction") or {}).get("instruction_text")
            ),
            "scene_id": str(episode.get("scene_id") or ""),
        }
    return dict(sorted(tasks.items()))


def validate_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    mappings = common.parse_path_maps(args.path_map)
    tasks = dataset_tasks(args.dataset)
    if len(tasks) != EXPECTED_TASKS:
        raise RuntimeError(f"expected {EXPECTED_TASKS} Level-3 EP2 tasks, found {len(tasks)}")
    requested_sessions = set(map(str, args.session_id))
    requested_episodes = set(map(str, args.episode_id))
    known_episodes = {row["episode_id"] for row in tasks.values()}
    if requested_sessions - set(tasks):
        raise RuntimeError(f"unknown Level-3 sessions: {sorted(requested_sessions-set(tasks))}")
    if requested_episodes - known_episodes:
        raise RuntimeError(
            f"unknown Level-3 EP2 episodes: {sorted(requested_episodes-known_episodes)}"
        )

    rows = []
    missing_m1 = []
    parse_counts = {"saved_lavira_parse": 0, "missing_lavira_parse": 0}
    for task_id, dataset_row in tasks.items():
        if requested_sessions and task_id not in requested_sessions:
            continue
        if requested_episodes and dataset_row["episode_id"] not in requested_episodes:
            continue
        m1 = args.m1_root / "sessions" / task_id / "stages/M1"
        missing = [name for name in CORE_M1_FILES if not (m1 / name).is_file()]
        if missing:
            missing_m1.append(f"{task_id}: {missing}")
            continue
        route = load_lavira_parse(args.lavira_parse_root, task_id, mappings)
        if route is not None and route["instruction"] != dataset_row["instruction"]:
            raise RuntimeError(
                f"LaViRA parse instruction differs from dataset for {task_id}: "
                f"{route['source']}"
            )
        if route is None:
            parse_counts["missing_lavira_parse"] += 1
            route = {
                "task_id": task_id,
                "instruction": dataset_row["instruction"],
                "types": None,
                "anchors": [],
                "suffix": "",
                "source": None,
                "source_sha256": None,
                "parsed": None,
                "missing_reason": "lavira_dual_anchor_plan_not_persisted",
            }
        else:
            parse_counts["saved_lavira_parse"] += 1
        rows.append({**dataset_row, "m1": m1, "route": route})
    if missing_m1:
        preview = "; ".join(missing_m1[:12])
        suffix = "" if len(missing_m1) <= 12 else f"; ... and {len(missing_m1)-12} more"
        raise RuntimeError(
            f"EP2 needs complete M1 for {len(missing_m1)} tasks: {preview}{suffix}"
        )
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("Level-3 EP2 selection is empty")
    for row in rows:
        row["_parse_counts"] = parse_counts
    return rows


def install_ep2_memory_environment(args: argparse.Namespace, work_root: Path, m1: Path) -> None:
    common.install_memory_environment(args, work_root, m1)
    os.environ.update(
        {
            "INHERIT_SG_PRELOAD_MEMORY_STAGE": "M1",
            "INHERIT_SG_QUERY_MIN_FINALIZED_EPISODES": "2",
            "LIFELONG_SG_LEVEL3_RECALL_POLICY": args.recall_policy,
            "LIFELONG_SG_DUAL_ANCHOR_ENABLE": "1",
            "LIFELONG_SG_VLM_TOP3_FALLBACK_POLICY": "top1",
            "LIFELONG_SG_VLM_TOP3_SELECTION_MODE": "force_single",
            # Level-3 memory_module exploration knobs (formal.env + probe_defaults).
            "LIFELONG_SG_EXPLORE_CONNECTION_ONLY": "0",
            "LIFELONG_SG_EXPLORE_PROGRESS_WEIGHT": "4.0",
            "LIFELONG_SG_EXPLORE_INFORMATION_WEIGHT": "0.25",
            "LIFELONG_SG_EXPLORE_PATH_COST_WEIGHT": "1.0",
            "LIFELONG_SG_EXPLORE_TURN_COST_WEIGHT": "0.15",
            "LIFELONG_SG_EXPLORE_REVISIT_WEIGHT": "2.0",
            "LIFELONG_SG_EXPLORE_OBSTACLE_RISK_WEIGHT": "3.0",
            "LIFELONG_SG_EXPLORE_MAX_FRONTIER_CANDIDATES": "16",
            "LIFELONG_SG_EXPLORE_MAX_PLANNER_CANDIDATES": "16",
            "LIFELONG_SG_EXPLORE_MAX_DEFERRED_SCAN_TURNS": "12",
            "LIFELONG_SG_EXPLORE_ALLOW_WORLD_ONLY_GATE": "0",
            "LIFELONG_SG_EXPLORE_WORLD_ONLY_GATE_AFTER_SCAN_TURNS": "11",
            "LIFELONG_SG_EXPLORE_CLEARANCE_CELLS": "2.0",
            "LIFELONG_SG_EXPLORE_START_CONNECTOR_CLEARANCE_CELLS": "1.5",
            "LIFELONG_SG_EXPLORE_START_CONNECTOR_MAX_DISTANCE_M": "0.75",
            "LIFELONG_SG_EXPLORE_MAX_ACTIONS": "200",
            "LIFELONG_SG_EXPLORE_MAX_DISTANCE_M": "26.0",
            "LIFELONG_SG_EXPLORE_CONNECTIVITY_CHECK_STRIDE": "4",
            "LIFELONG_SG_EXPLORE_FRONTIER_TOLERANCE_M": "0.15",
            "LIFELONG_SG_EXPLORE_SCAN_HEADING_TOLERANCE_DEG": "15",
            "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE": "1",
            "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE_CONNECTIVITY_STRIDE": "4",
            "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE_MAX_ACTIONS": "48",
            "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE_MAX_FORWARDS": "36",
            "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE_CLEARANCE_CELLS": "1.0",
            "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE_MIN_PROGRESS_M": "-0.26",
            "LIFELONG_SG_EXPLORE_DIRECTIONAL_PROBE_REQUIRE_NAVMESH": "1",
            "LIFELONG_SG_EXPLORE_NO_ACTION_RECOVERY": "1",
            "LIFELONG_SG_EXPLORE_OLD_MAP_FALLBACK": "1",
            "LIFELONG_SG_EXPLORE_OLD_MAP_MAX_ACTIONS": "48",
            "LIFELONG_SG_EXPLORE_OLD_MAP_MAX_FORWARDS": "36",
            "LIFELONG_SG_EXPLORE_OLD_MAP_SWITCH_ACTIONS": "8",
            "LIFELONG_SG_EXPLORE_OLD_MAP_MIN_PROGRESS_M": "0.25",
            "LIFELONG_SG_WORLD_FULL_WINDOW_RETRY": "1",
        }
    )
    apply_ep2_update_memory(getattr(args, "update_memory", "full"))


def apply_ep2_update_memory(mode: str) -> None:
    """off freezes M1. nav updates the metric map and trajectory. full also updates the semantic scene graph."""
    mode = str(mode or "full").strip().lower()
    if mode == "off":
        flags = ("1", "0")
    elif mode == "nav":
        flags = ("1", "1")
    elif mode == "full":
        flags = ("0", "0")
    else:
        raise RuntimeError("EP2 update-memory must be off, nav, or full")
    os.environ["INHERIT_SG_PRELOAD_QUERY_ONLY"] = flags[0]
    os.environ["INHERIT_SG_PRELOAD_WORLD_MAP_ONLY"] = flags[1]
    os.environ["MEMORY_MODULE_PRELOAD_QUERY_ONLY"] = flags[0]
    os.environ["MEMORY_MODULE_PRELOAD_WORLD_MAP_ONLY"] = flags[1]


def parse_level3_instruction_live(instruction: str) -> dict[str, Any]:
    """Call the instruction VLM once. Saved LaViRA plans are not read."""
    from membridge.habitat_runtime.level3_dual_anchor import validate_level3_parse
    from membridge.vlm_instruction_parser.runtime import InstructionRuntimeConfig

    parsed = InstructionRuntimeConfig.from_env().build().parse(instruction)
    validate_level3_parse(parsed)
    return {
        "task_id": None,
        "instruction": common.normalize_text(instruction),
        "types": [segment.type for segment in parsed.segments],
        "anchors": [segment.to_dict() for segment in parsed.segments[:2]],
        "suffix": parsed.segments[2].text,
        "source": "live_vlm",
        "source_sha256": None,
        "parsed": parsed,
    }


def prepare_dual_anchor_plan(adapter: Any, route: Mapping[str, Any], recall_policy: str):
    from membridge.habitat_runtime.level3_adaptation import _top3_prompt_for_recall_scopes
    from membridge.habitat_runtime.level3_dual_anchor import (
        DualAnchorNavigationState,
        DualAnchorRecallPlanner,
        InsufficientProvenanceTop3,
    )
    from membridge.vlm_instruction_parser import OpenAICompatibleVLMClient
    from membridge.vlm_top3_selection import (
        CandidateImagePreparer,
        VLMNoMatchError,
        VLMTop3Reranker,
    )

    parsed = route.get("parsed")
    if parsed is None:
        raise RuntimeError("missing live Level-3 instruction parse")
    vocabulary = adapter.runtime_query_vocabulary or {}
    prompt, _ = _top3_prompt_for_recall_scopes(
        [segment.recall_scope for segment in parsed.segments[:2]],
        recall_policy=recall_policy,
    )
    image_root = str(getattr(adapter, "preload_dir", "") or adapter.pipeline.run_dir)
    reranker = VLMTop3Reranker(
        OpenAICompatibleVLMClient.from_env(),
        image_preparer=CandidateImagePreparer(session_dir=image_root, image_mode="frame_box"),
        fallback_policy="top1",
        selection_mode="force_single",
        system_prompt=prompt,
        max_tokens=2048,
    )
    planner = DualAnchorRecallPlanner(
        adapter,
        reranker,
        label_aliases=vocabulary.get("aliases") or {},
        room_aliases=vocabulary.get("room_aliases") or {},
        canonical_labels=vocabulary.get("labels") or (),
        recall_policy=recall_policy,
    )
    try:
        plan = planner.plan(parsed)
    except InsufficientProvenanceTop3 as exc:
        raise RuntimeError(
            "sg_retrieval_did_not_return_exact_top3: "
            f"anchor {exc.anchor_index} retrieved {len(exc.candidates)}, expected 3"
        ) from exc
    except VLMNoMatchError as exc:
        return {
            "status": "fallback",
            "reason": "vlm_top3_no_match",
            "selection_source": "live_vlm_explicit_no_match",
            "rejection": exc.to_dict() if hasattr(exc, "to_dict") else str(exc),
            "anchor_index": getattr(exc, "anchor_index", None),
        }
    payload = plan.to_dict()
    payload["status"] = "ready"
    payload["reason"] = "live_instruction_parse_plus_live_vlm_top3"
    payload["state"] = DualAnchorNavigationState(plan.selected_targets)
    payload["plan"] = plan
    return payload


def publish_m2(
    m1: Path, run_dir: Path, destination: Path, metadata: Mapping[str, Any]
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".M2.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    if temporary.exists() or destination.exists():
        raise FileExistsError(f"refusing to replace M2: {destination}")
    parent_manifest = m1 / "stage_manifest.json"
    shutil.copytree(m1, temporary)
    for name in ("world_metric_map.npz", "trajectory_graph.npz"):
        source = run_dir / name
        if not source.is_file():
            raise RuntimeError(f"world-map-only EP2 did not produce {source}")
        shutil.copy2(source, temporary / name)
    source_manifest = run_dir / "episode_manifest.jsonl"
    if source_manifest.is_file():
        with (temporary / "episode_manifest.jsonl").open("a", encoding="utf-8") as output:
            output.write(source_manifest.read_text(encoding="utf-8"))
    for name in (
        "recall_plan.json",
        "level3_dual_anchor_recall_plan.json",
        "sg_navigation_decisions.jsonl",
        "rollout.json",
    ):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, temporary / name)
    stage_manifest = {
        "schema_version": "janusvln_level3_memory_module_stage_v1",
        "session_id": metadata["session_id"],
        "episode_id": metadata["episode_id"],
        "role": "ep2",
        "memory_read": "M1",
        "memory_write": "M2",
        "parent_m1": str(m1.resolve()),
        "parent_stage_manifest_sha256": (
            common.sha256_file(parent_manifest) if parent_manifest.is_file() else None
        ),
        "semantic_sg_policy": "frozen_m1_no_sam3_no_dinov3",
        "navigation_memory_policy": "live_rgbd_world_map_and_trajectory_update",
        "instruction_parse_source": "live_vlm",
        "vlm_top3_source": "live_api",
        "raw_rgbd_persisted": False,
        "route": copy.deepcopy({k: metadata["route"].get(k) for k in (
            "instruction", "types", "anchors", "suffix", "source", "source_sha256"
        )}),
        "sg_handoff": copy.deepcopy(metadata["sg_handoff"]),
        "artifacts_sha256": {
            name: common.sha256_file(temporary / name)
            for name in CORE_M1_FILES
            if (temporary / name).is_file()
        },
    }
    common.atomic_json(temporary / "stage_manifest.json", stage_manifest)
    os.replace(temporary, destination)
    return destination


def begin_anchor(navigator, candidate, task_text):
    target_info = common.make_target_info(navigator, candidate, task_text)
    if target_info is None:
        return None
    target_info["query_source"] = "lavira_level3_saved_parse"
    return target_info


def level3_continuation(route: Mapping[str, Any], plan: Any, completed_count: int) -> dict[str, Any]:
    """Match LaViRA `_sg_activate_level3_lavira`: keep only unfinished work.

    completed_count is the number of anchors actually reached by SG, not the
    number of VLM selections.  Remaining recall clauses and the type-2 suffix
    are concatenated in source order.
    """

    full = common.normalize_text(route.get("instruction"))
    if completed_count not in (0, 1, 2):
        raise RuntimeError(f"invalid Level-3 completed-anchor count: {completed_count}")
    if completed_count == 0 or plan is None:
        return {
            "mode": "full_instruction_fallback",
            "completed_count": 0,
            "continuation": full,
            "completed_recall_todos": [],
        }
    pending = [
        common.normalize_text(anchor.segment.get("text"))
        for anchor in plan.anchors[completed_count:]
    ]
    pending.append(common.normalize_text(route.get("suffix")))
    continuation = " ".join(text for text in pending if text) or full
    completed = []
    for index, anchor in enumerate(plan.anchors[:completed_count], 1):
        completed.append(
            {
                "content": common.normalize_text(anchor.segment.get("text")),
                "status": "completed",
                "result": "Completed by SG navigation; reached Anchor {}: {}".format(
                    index,
                    (anchor.selected_candidate or {}).get("label", "object"),
                ),
            }
        )
    if completed_count == 2:
        mode = "level3_anchor2_to_ordinary_vln_suffix"
    else:
        mode = "level3_sg_failure_to_janus_remaining_clauses"
    return {
        "mode": mode,
        "completed_count": completed_count,
        "continuation": continuation,
        "completed_recall_todos": completed,
    }


def run_one(
    *,
    env: Any,
    habitat_config: Any,
    episode: Any,
    row: Mapping[str, Any],
    model: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from membridge.habitat_runtime.online_adapter import OnlineINHeritSGBuilder
    from membridge.trajectory_graph.habitat_adapter import heading_from_quaternion_xyzw

    session_id, episode_id = row["session_id"], row["episode_id"]
    destination = args.output_root / "sessions" / session_id / "stages/M2"
    if destination.exists():
        raise FileExistsError(f"M2 already exists without a completed rollout row: {destination}")
    work_root = (
        args.output_root / ".working" / f"{session_id}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    work_root.mkdir(parents=True)
    install_ep2_memory_environment(args, work_root, row["m1"])

    env.current_episode = episode
    observations = env.reset()
    initial_pose = common.pose_from_env(env, 0, 0)
    adapter = OnlineINHeritSGBuilder(
        common.sensor_memory_config(habitat_config, args.sim_gpu_id),
        width=int(observations["rgb"].shape[1]),
        height=int(observations["rgb"].shape[0]),
        save_dir=str(work_root),
        device=args.device,
    )
    adapter.start_episode(episode_id, scene_id=str(episode.scene_id), source_episode_id=episode_id)
    adapter.process_observation(common.memory_observation(observations, initial_pose), step=0)
    run_dir = Path(adapter.pipeline.run_dir).resolve()

    navigator = common.MemoryNavigator(
        adapter=adapter,
        env=env,
        habitat_config=habitat_config,
        decision_log=run_dir / "sg_navigation_decisions.jsonl",
    )
    navigator.current_episode_id = episode_id
    heading = heading_from_quaternion_xyzw(initial_pose["rotation_xyzw"])
    navigator.update_pose(initial_pose["position"], heading, 0)
    model.reset_memory()

    route = row["route"]
    recall_plan = prepare_dual_anchor_plan(adapter, route, args.recall_policy)
    persistable = {
        key: value
        for key, value in recall_plan.items()
        if key not in {"state", "plan"}
    }
    common.atomic_json(run_dir / "recall_plan.json", persistable)
    if recall_plan.get("status") == "ready":
        common.atomic_json(run_dir / "level3_dual_anchor_recall_plan.json", persistable)

    dual_state = recall_plan.get("state")
    plan = recall_plan.get("plan")
    target_info = None
    current_anchor_index = None
    phase = "janus_fallback"
    active_instruction = row["instruction"]
    if dual_state is not None and plan is not None:
        current = dual_state.next_target()
        current_anchor_index = 0
        target_info = begin_anchor(
            navigator, current, plan.anchors[0].segment["text"]
        )
        if target_info is not None:
            phase = "sg_navigation"
        else:
            recall_plan["status"] = "fallback"
            recall_plan["reason"] = "anchor1_not_navigable"

    poses = [initial_pose]
    rgb_history = [
        Image.fromarray(np.asarray(observations["rgb"], dtype=np.uint8)).convert("RGB")
    ]
    requested_actions: list[str] = []
    executed_actions: list[str] = []
    model_step_seconds: list[float] = []
    sg_actions = 0
    janus_actions = 0
    first_janus_history_frames = None
    anchors_reached: list[str] = []
    fallback_reason = None if phase == "sg_navigation" else recall_plan.get("reason")
    stopped = False
    started_at, wall_start = common.utc_now(), time.monotonic()

    def make_rollout_record(sg_handoff: Mapping[str, Any]) -> dict[str, Any]:
        metrics = env.get_metrics()
        ne = common.metric_value(metrics, "distance_to_goal")
        oracle_ne = common.metric_value(metrics, "oracle_navigation_error")
        return {
            "schema_version": "janusvln_level3_ep2_memory_module_rollout_v1",
            "phase": "level3_ep2",
            "episode_id": episode_id,
            "scene_id": str(episode.scene_id),
            "session_id": session_id,
            "role": "ep2",
            "instruction": row["instruction"],
            "route": {
                "types": route.get("types"),
                "anchors": route.get("anchors"),
                "suffix": route.get("suffix"),
                "source": route.get("source"),
            },
            "sg_handoff": copy.deepcopy(dict(sg_handoff)),
            "poses": poses,
            "positions": [pose["position"] for pose in poses],
            "rotations_xyzw": [pose["rotation_xyzw"] for pose in poses],
            "requested_actions": requested_actions,
            "executed_actions": executed_actions,
            "stopped": stopped,
            "timed_out": bool(not stopped and len(executed_actions) >= args.max_steps),
            "steps": len(executed_actions),
            "rgbd_persisted": False,
            "memory_module": {
                "read_stage": "M1",
                "write_stage": "M2",
                "m1": str(row["m1"].resolve()),
                "m2": str(destination.resolve()),
                "semantic_sg_updated": False,
                "world_navigation_map_updated_from_live_rgbd": True,
                "trajectory_graph_updated": True,
                "sam3_loaded": False,
                "dinov3_loaded": False,
                "navigation_controller_version": NAVIGATION_CONTROLLER_VERSION,
            },
            "metrics": {
                "ne": ne,
                "distance_to_goal": ne,
                "success": None if ne is None else float(ne <= 3.0),
                "habitat_success": common.metric_value(metrics, "success"),
                "spl": common.metric_value(metrics, "spl"),
                "oracle_navigation_error": oracle_ne,
                "oracle_success": None if oracle_ne is None else float(oracle_ne <= 3.0),
            },
            "timing": {
                "started_at": started_at,
                "finished_at": common.utc_now(),
                "wall_seconds": time.monotonic() - wall_start,
                "janus_model_step_seconds": model_step_seconds,
            },
        }

    try:
        while len(executed_actions) < args.max_steps and not env.episode_over:
            step = len(executed_actions)
            current_pose = poses[-1]
            heading = heading_from_quaternion_xyzw(current_pose["rotation_xyzw"])
            navigator.update_pose(current_pose["position"], heading, step)

            if phase == "sg_navigation":
                reached = navigator._sg_nav_reached_reason(
                    step=step,
                    current_world_position=current_pose["position"],
                    current_world_heading=heading,
                    target_info=target_info,
                )
                if reached is not None:
                    uid = str(target_info.get("uid") or target_info.get("node_id") or "")
                    anchors_reached.append(uid)
                    navigator._write_sg_decision(
                        step,
                        {
                            "decision": "sg_anchor_reached",
                            "reach_reason": reached,
                            "anchor_index": current_anchor_index,
                            "uid": uid,
                        },
                    )
                    nxt = dual_state.mark_reached(uid)
                    if nxt is None:
                        handoff = level3_continuation(route, plan, 2)
                        navigator._write_sg_decision(
                            step,
                            {
                                "decision": "sg_dual_anchor_complete_handoff",
                                "handoff": handoff,
                            },
                        )
                        phase = "janus_suffix"
                        active_instruction = handoff["continuation"]
                        continue
                    current_anchor_index = 1
                    target_info = begin_anchor(
                        navigator, nxt, plan.anchors[1].segment["text"]
                    )
                    if target_info is None:
                        fallback_reason = "anchor2_not_navigable"
                        handoff = level3_continuation(route, plan, 1)
                        phase = "janus_fallback"
                        active_instruction = handoff["continuation"]
                    continue
                action = navigator._get_sg_world_path_action(
                    step=step,
                    current_world_position=current_pose["position"],
                    current_world_heading=heading,
                    target_info=target_info,
                )
                if action is None:
                    reason = target_info.pop(
                        "_sg_pending_abort_reason", "sg_controller_no_action"
                    )
                    recovered = navigator._on_sg_nav_aborted(
                        step=step,
                        reason=reason,
                        current_world_position=current_pose["position"],
                        current_world_heading=heading,
                        target_info=target_info,
                    )
                    if recovered:
                        continue
                    fallback_reason = reason
                    handoff = level3_continuation(route, plan, len(anchors_reached))
                    phase = "janus_fallback"
                    active_instruction = handoff["continuation"]
                    continue
                requested = "SG:" + common.ACTION_NAMES[int(action)]
                sg_actions += 1
            else:
                if first_janus_history_frames is None:
                    first_janus_history_frames = len(rgb_history)
                images = common.sample_history(rgb_history, args.num_history)
                inference_start = time.monotonic()
                answer = model.call_model(images, active_instruction, step)[0]
                model_step_seconds.append(time.monotonic() - inference_start)
                action = getattr(common.Actions, str(answer), common.Actions.STOP)
                requested = "JANUS:" + str(answer)
                janus_actions += 1

            requested_actions.append(requested)
            executed_actions.append(common.ACTION_NAMES[int(action)])
            observations = env.step(int(action))
            pose = common.pose_from_env(env, len(poses), len(executed_actions))
            poses.append(pose)
            rgb_history.append(
                Image.fromarray(np.asarray(observations["rgb"], dtype=np.uint8)).convert("RGB")
            )
            adapter.process_observation(
                common.memory_observation(observations, pose), step=len(executed_actions)
            )
            if int(action) == common.Actions.STOP:
                stopped = True

        adapter.finish_episode()
        if adapter.failed:
            raise RuntimeError("memory_module marked EP2 world-map update failed")
        both_reached = len(anchors_reached) == 2
        janus_handoff = level3_continuation(route, plan, len(anchors_reached))
        sg_handoff = {
            "recall_plan_status": recall_plan.get("status"),
            "recall_plan_reason": recall_plan.get("reason"),
            "selected_uids": [
                str(item.get("uid") or item.get("node_id") or "")
                for item in (plan.selected_targets if plan is not None else ())
            ],
            "anchors_reached": anchors_reached,
            "anchor_reached_count": len(anchors_reached),
            "both_anchors_reached": both_reached,
            "fallback_reason": fallback_reason,
            "janus_instruction_mode": janus_handoff["mode"],
            "janus_continuation_instruction": janus_handoff["continuation"],
            "completed_recall_todos": janus_handoff["completed_recall_todos"],
            "suffix": route.get("suffix"),
            "sg_actions": sg_actions,
            "janus_actions": janus_actions,
            "sg_visual_history_frames_available_to_first_janus_call": first_janus_history_frames,
            "lavira_parse_source": route.get("source"),
            "vlm_top3_called": recall_plan.get("status") == "ready"
            or str(recall_plan.get("reason") or "").startswith("vlm_")
            or "top3" in str(recall_plan.get("reason") or ""),
            "exploration_implementation": os.environ[
                "MEMORY_MODULE_EXPLORATION_IMPLEMENTATION"
            ],
            "navigation_controller_version": NAVIGATION_CONTROLLER_VERSION,
            "recall_policy": args.recall_policy,
        }
        result = make_rollout_record(sg_handoff)
        common.atomic_json(run_dir / "rollout.json", result)
        publish_m2(
            row["m1"],
            run_dir,
            destination,
            {
                "session_id": session_id,
                "episode_id": episode_id,
                "route": route,
                "sg_handoff": sg_handoff,
            },
        )
    except Exception:
        try:
            adapter.abort_episode()
        except Exception:
            pass
        raise
    finally:
        if destination.exists():
            shutil.rmtree(work_root, ignore_errors=True)
    return result


def write_summary(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> None:
    metric_names = ("success", "spl", "oracle_success", "distance_to_goal")
    means: dict[str, float | None] = {}
    for name in metric_names:
        values = [row["metrics"].get(name) for row in rows]
        finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        means[name] = sum(finite) / len(finite) if finite else None
    common.atomic_json(
        args.output_root / "summary.json",
        {
            "schema_version": "janusvln_level3_ep2_memory_module_summary_v1",
            "episode_count": len(rows),
            "means": means,
            "both_anchors_reached_count": sum(
                bool(row["sg_handoff"].get("both_anchors_reached")) for row in rows
            ),
            "sg_fallback_count": sum(
                row["sg_handoff"].get("fallback_reason") is not None for row in rows
            ),
        },
    )


def consolidate_output(args: argparse.Namespace) -> list[dict[str, Any]]:
    lock_path = args.output_root / ".aggregate.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        all_rows = []
        for path in sorted((args.output_root / "raw").glob("ep2_*.jsonl")):
            all_rows.extend(common.read_jsonl(path))
        by_episode = {str(row["episode_id"]): row for row in all_rows}
        ordered = [by_episode[key] for key in sorted(by_episode, key=int)]
        common.atomic_jsonl(args.output_root / "episodes.jsonl", ordered)
        write_summary(args, ordered)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return ordered


def main() -> None:
    args = parse_args()
    args.janus_root = args.janus_root.resolve()
    args.model_path = args.model_path.resolve()
    args.config = args.config.resolve()
    args.dataset = args.dataset.resolve()
    args.m1_root = args.m1_root.resolve()
    args.lavira_parse_root = [path.resolve() for path in args.lavira_parse_root]
    args.output_root = args.output_root.resolve()
    args.query_vocabulary = args.query_vocabulary.resolve()
    args.run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    selected = validate_inputs(args)
    parse_counts = selected[0]["_parse_counts"]
    print(
        json.dumps(
            {
                "selected": len(selected),
                "expected_tasks": EXPECTED_TASKS,
                "parse_sources": parse_counts,
                "recall_policy": args.recall_policy,
                "memory_read": "M1",
                "sam3": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    os.chdir(args.janus_root)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(args.janus_root / "src"))
    common.install_full_exploration_overlay()
    from membridge.habitat_runtime.level3_adaptation import (
        install_dual_anchor_plan_replay,
    )

    # After the L2 exploration overlay, wrap post-hit approach so a
    # start_not_clearance_safe frontier still enters scan/directional probe.
    install_dual_anchor_plan_replay(common.MemoryNavigator, LOG)
    from evaluation import JanusVLN_Inference, set_seed
    from protocol_benchmark import group_level3_episodes, make_env

    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    worker = args.worker_id.strip() or f"shard{args.shard_index:02d}"
    raw_log = args.output_root / "raw" / f"ep2_{worker}.jsonl"
    completed = {str(row["episode_id"]) for row in common.read_jsonl(raw_log)}
    env, habitat_config = make_env(args.config, "level3", args.sim_gpu_id, args.dataset)
    grouped = group_level3_episodes(env.episodes)
    ep2_by_session = {
        session_id: next(
            episode
            for episode in episodes
            if str((getattr(episode, "info", {}) or {}).get("level3_role")) == "ep2"
        )
        for session_id, episodes in grouped.items()
    }
    model = JanusVLN_Inference(str(args.model_path), device=args.device)
    failures: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(selected, 1):
            episode_id = row["episode_id"]
            if episode_id in completed:
                continue
            destination = args.output_root / "sessions" / row["session_id"] / "stages/M2"
            if destination.is_dir():
                embedded = destination / "rollout.json"
                if embedded.is_file():
                    recovered = common.read_json(embedded)
                    if str(recovered.get("episode_id")) != episode_id:
                        raise RuntimeError(f"embedded rollout/M2 mismatch: {embedded}")
                    common.append_jsonl(raw_log, recovered)
                    completed.add(episode_id)
                    LOG.info("recovered completed rollout from atomic M2: %s", embedded)
                    continue
                if not args.skip_existing:
                    raise RuntimeError(
                        f"M2 exists without its atomic rollout record: {destination}"
                    )
            LOG.info(
                "EP2 %d/%d session=%s episode=%s parse=%s",
                index,
                len(selected),
                row["session_id"],
                episode_id,
                "saved" if row["route"].get("parsed") is not None else "missing",
            )
            try:
                result = run_one(
                    env=env,
                    habitat_config=habitat_config,
                    episode=ep2_by_session[row["session_id"]],
                    row=row,
                    model=model,
                    args=args,
                )
                common.append_jsonl(raw_log, result)
                completed.add(episode_id)
                print(
                    json.dumps(
                        {
                            "episode": episode_id,
                            "metrics": result["metrics"],
                            "handoff": result["sg_handoff"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception as exc:
                failure = {
                    "session_id": row["session_id"],
                    "episode_id": episode_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "time": common.utc_now(),
                }
                failures.append(failure)
                common.append_jsonl(
                    args.output_root / "raw" / f"failures_{worker}.jsonl", failure
                )
                LOG.exception("Level-3 EP2 failed: %s", failure)
                if not args.continue_on_error:
                    raise
    finally:
        env.close()

    consolidate_output(args)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
