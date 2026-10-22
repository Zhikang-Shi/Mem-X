#!/usr/bin/env python3
"""Level-1 query +MemBridge in habitat world coordinates.

Each query reads that session's frozen M0 and writes only its own branch.
There is no geometric localization and no SE(2) alignment. The instruction
VLM is called live. A pure type-1 recall hit walks the scene graph at 15
degrees and stops on the reachable cell. Otherwise the selected model follows
the full instruction.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from bootstrap import VOCABULARY, install

install()
_RUNNERS = Path(__file__).resolve().parent
HERE = _RUNNERS.parent

from run_ep1_memory_janus import (  # noqa: E402
    Actions,
    MemoryNavigator,
    install_full_exploration_overlay,
    install_memory_environment,
    make_target_info,
    memory_observation,
    pose_from_env,
    prepare_recall_plan,
    sensor_memory_config,
)
from se2 import yaw_from_xyzw  # noqa: E402

ROLES = ("ep1", "ep2", "ep3", "ep4")
# ETPNav and CMA follow the Level-1 paper walk at 15°. Janus and Uni-LaViRA
# follow their earlier Level-1 configs at 30°.
TURN_ANGLE_BY_MODEL = {"etpnav": 15.0, "cma": 15.0, "janus": 30.0, "unilavira": 30.0}
ARRIVAL_POLICY = "exact_memory_module_reachable_object_approach_no_semantic_3m_shortcut"
DEFAULT_LAVIRA_ROUTE_ROOT = Path(os.environ.get("LAVIRA_ROUTE_ROOT", ""))
DEFAULT_ROUTE_RECOVERY = Path(os.environ.get("LAVIRA_ROUTE_RECOVERY", ""))
SEMANTIC_FILES = (
    "memory_kb.json",
    "builder_state.pkl",
    "spatial_graph.pkl",
    "map_data.pkl",
    "world_metric_map.npz",
    "trajectory_graph.npz",
    "nodes.jsonl",
    "relations.csv",
    "config_snapshot.yaml",
)


def info_of(episode) -> dict:
    return dict(getattr(episode, "info", {}) or {})


def session_id_of(episode) -> str:
    episode_id = str(getattr(episode, "episode_id", "") or "")
    for role in ("ep0", "ep1", "ep2", "ep3", "ep4"):
        suffix = f"_{role}"
        if episode_id.endswith(suffix):
            return episode_id[: -len(suffix)]
    return str(info_of(episode).get("session_id") or episode_id)


def role_of(episode) -> str:
    info = info_of(episode)
    return str(info.get("role") or info.get("level1_role") or "")


def instruction_of(episode) -> str:
    text = getattr(episode, "instruction", None)
    if text is None:
        return ""
    return str(getattr(text, "instruction_text", None) or "")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def saved_route_path(root: Path, session_id: str, role: str, episode_id: str) -> Path:
    return root / session_id / f"{role}_{episode_id}" / "lifelong_session/level1_vlm_recall_plan.json"


def parse_level1_instruction(instruction: str) -> dict[str, Any]:
    """Call the instruction VLM. Level-1 accepts one type-1 recall and nothing else."""
    from membridge.vlm_instruction_parser.runtime import InstructionRuntimeConfig

    try:
        parsed = InstructionRuntimeConfig.from_env().build().parse(instruction)
    except Exception as exc:
        return {"status": "fallback", "reason": f"parse_failed:{type(exc).__name__}", "error": str(exc)}
    types = [segment.type for segment in parsed.segments]
    recalls = [segment for segment in parsed.segments if segment.type == 1]
    suffixes = [segment for segment in parsed.segments if segment.type == 2]
    if len(recalls) != 1 or suffixes:
        return {"status": "fallback", "reason": "level1_parse_not_pure_recall", "types": types}
    recall = recalls[0].to_dict()
    if not normalize_text(recall.get("text")) or not normalize_text(recall.get("label")):
        return {"status": "fallback", "reason": "level1_parse_missing_label", "types": types}
    return {
        "status": "ready",
        "types": types,
        "recall": recall,
        "merged_suffix": "",
        "suffix_parts": [],
        "source_kind": "live_instruction_vlm",
    }


def load_saved_route(session_id: str, role: str, episode_id: str, instruction: str, args) -> dict[str, Any]:
    """Use the already published LaViRA parse. Do not call the instruction VLM."""
    source = saved_route_path(args.lavira_route_root, session_id, role, episode_id)
    source_kind = "lavira_ready_plan"
    if source.is_file():
        payload = json.loads(source.read_text(encoding="utf-8"))
        router = payload.get("router") if isinstance(payload, dict) else None
    else:
        recovery = json.loads(args.route_recovery.read_text(encoding="utf-8"))
        payload = (recovery.get("episodes") or {}).get(str(episode_id))
        if not isinstance(payload, dict):
            return {
                "status": "fallback",
                "reason": "missing_saved_route",
                "error": str(source),
            }
        if str(payload.get("session_id")) != session_id or str(payload.get("role")) != role:
            return {
                "status": "fallback",
                "reason": "saved_route_identity_mismatch",
                "error": str(args.route_recovery),
            }
        router = payload
        source = args.route_recovery
        source_kind = "lavira_route_recovery"
    segments = (router or {}).get("segments") if isinstance(router, dict) else None
    if not isinstance(segments, list) or len(segments) != 1 or int(segments[0].get("type", -1)) != 1:
        return {"status": "fallback", "reason": "saved_route_not_pure_recall", "error": str(source)}
    segment = segments[0]
    saved_instruction = normalize_text((router or {}).get("instruction") or payload.get("instruction"))
    if saved_instruction != normalize_text(instruction):
        return {"status": "fallback", "reason": "saved_route_instruction_mismatch", "error": str(source)}
    if not normalize_text(segment.get("text")) or not normalize_text(segment.get("label")):
        return {"status": "fallback", "reason": "saved_route_missing_label", "error": str(source)}
    return {
        "status": "ready",
        "types": [1],
        "recall": copy.deepcopy(segment),
        "merged_suffix": "",
        "suffix_parts": [],
        "source": str(source),
        "source_kind": source_kind,
    }


def enable_live_semantic_update(nav_model: str = "") -> None:
    from run_ep2_memory_janus import apply_ep2_update_memory

    apply_ep2_update_memory("full")
    os.environ.update(
        {
            "INHERIT_SG_SAM3_CKPT": os.environ.get("MODELS_DIR", "") + "/facebook/sam3/sam3.pt",
            "INHERIT_SG_SAM3_SOURCE_ROOT": os.environ.get("MODELS_DIR", "") + "/source/sam3",
            "SAM3_SOURCE_ROOT": os.environ.get("MODELS_DIR", "") + "/source/sam3",
            "INHERIT_SG_DINOV3_MODEL": os.environ.get("MODELS_DIR", "") + "/facebook/dinov3",
            "INHERIT_SG_DINOV3_REPO": os.environ.get("MODELS_DIR", "") + "/source/dinov3",
            "INHERIT_SG_TEXT_ENCODER": os.environ.get("MODELS_DIR", "") + "/all-MiniLM-L6-v2",
            "INHERIT_SG_SAM3_CATEGORY_BATCH_SIZE": os.environ.get(
                "INHERIT_SG_SAM3_CATEGORY_BATCH_SIZE", "5" if nav_model == "janus" else "15"
            ),
            "INHERIT_SG_KEYFRAME_FILTER": "1",
        }
    )


def _target_info_from_approach(
    navigator: Any,
    candidate: Mapping[str, Any],
    task_text: str,
    approach: Mapping[str, Any],
    route_meta: Mapping[str, Any],
    anchors: Sequence[Any],
) -> dict[str, Any]:
    reachable = approach["reachable"]
    current = navigator._sg_current_world_xz()
    try:
        selection = navigator._sg_select_execution_world_path(
            reachable,
            navigator._sg_compose_trajectory_guided_world_path(reachable, route_meta, current),
            allow_shortcut=True,
        )
        path = list(selection["path"])
        source = str(selection["execution_path_source"])
        length_m = float(reachable.get("path_length_cells") or 0.0) * float(
            navigator._sg_world_metric_map().spec.resolution_m
        )
    except Exception:
        path = list(reachable.get("path") or [])
        source = "deferred_observation_scan"
        length_m = 0.0
    target = copy.deepcopy(dict(candidate))
    if isinstance(route_meta, Mapping):
        target.update(copy.deepcopy(dict(route_meta)))
    target.update(copy.deepcopy(approach["navigation_reference"]))
    target.update(
        {
            "todo_text": task_text,
            "query_source": "level1_live_route",
            "target_world_xz": list(map(float, reachable["target_world_xz"])),
            "target_world_grid_x": int(reachable["target_grid"][0]),
            "target_world_grid_z": int(reachable["target_grid"][1]),
            "raw_target_world_grid_x": int(reachable["raw_target_grid"][0]),
            "raw_target_world_grid_z": int(reachable["raw_target_grid"][1]),
            "world_map_resolution_m": float(navigator._sg_world_metric_map().spec.resolution_m),
            "_sg_world_path_cells": path,
            "_sg_path_progress_cells": 0.0,
            "_sg_route_planning_anchors_world_xz": [
                [float(point[0]), float(point[1])] for point in (anchors or [])
            ],
            "execution_path_source": source,
            "theta_path_waypoints": len(path),
            "theta_path_length_m": length_m,
        }
    )
    navigator._sg_world_plan_window = reachable.get("window")
    navigator._sg_world_safe_mask = reachable.get("safe_mask")
    navigator._sg_world_trajectory_probe_mask = reachable.get("_exploration_trajectory_probe_mask")
    navigator._sg_path_plan_active = True
    return target


def open_object_or_explore(navigator: Any, candidate: Mapping[str, Any], task_text: str, step: int = 0):
    """Use a theta path when one exists, otherwise scan and then explore."""
    target = make_target_info(navigator, candidate, task_text)
    if target is not None:
        return target, "theta_approach"
    recover = getattr(navigator, "_sg_recover_no_traversable_topk", None)
    packed = recover(
        candidates=[copy.deepcopy(dict(candidate))],
        rejected_targets=[],
        current_world_xz=navigator._sg_current_world_xz(),
        task_text=task_text,
        full_instruction=task_text,
        query_source="level1_live_route",
        todo_idx=0,
        dual_context=None,
        step=int(step),
    ) if callable(recover) else None
    if packed is not None:
        recovered_candidate, raw_world_xz, _semantic_xz, navigation_reference, reachable, route_meta, anchors = packed
        target = _target_info_from_approach(
            navigator,
            recovered_candidate,
            task_text,
            {
                "raw_world_xz": raw_world_xz,
                "navigation_reference": navigation_reference,
                "reachable": reachable,
            },
            route_meta,
            anchors,
        )
        return target, "no_traversable_scan"
    position = candidate.get("position") or [0.0, 0.0, 0.0]
    world_map = navigator._sg_world_metric_map()
    resolution_m = float(world_map.spec.resolution_m)
    target_xz = [float(position[0]), float(position[2])] if len(position) >= 3 else [0.0, 0.0]
    grid_x = int(round(target_xz[0] / resolution_m - 0.5)) if resolution_m else 0
    grid_z = int(round(target_xz[1] / resolution_m - 0.5)) if resolution_m else 0
    stub = copy.deepcopy(dict(candidate))
    stub.update(
        {
            "todo_text": task_text,
            "query_source": "level1_live_route",
            "uid": candidate.get("uid") or candidate.get("node_id"),
            "target_world_xz": target_xz,
            "target_world_grid_x": grid_x,
            "target_world_grid_z": grid_z,
            "raw_target_world_grid_x": grid_x,
            "raw_target_world_grid_z": grid_z,
            "world_map_resolution_m": resolution_m,
            "_sg_world_path_cells": [],
            "approach_exploration_mode": True,
            "_sg_pending_abort_reason": "no_initial_theta_path",
        }
    )
    recovered = navigator._on_sg_nav_aborted(
        step=step,
        reason="no_world_reachable_target_within_snap_radius",
        target_info=stub,
        current_world_position=None,
        current_world_heading=None,
    )
    if recovered or stub.get("approach_exploration_mode"):
        navigator._sg_path_plan_active = True
        return stub, "exploration_abort_overlay"
    return None, "no_exploration_entry"


def publish_branch(adapter, memory: Path, destination: Path, record: dict) -> None:
    """Copy frozen M0, then overlay this query's private scene-graph update."""
    if destination.exists():
        raise FileExistsError(destination)
    adapter.finish_episode()
    if adapter.failed:
        raise RuntimeError("memory_module marked the query branch failed")
    run_dir = Path(adapter.pipeline.run_dir).resolve()
    shutil.copytree(memory, destination)
    for name in SEMANTIC_FILES:
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    manifest_path = destination / "stage_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "schema_version": "level1_memory_module_query_branch_v1",
            "session_id": record["session_id"],
            "episode_id": record["episode_id"],
            "role": record["role"],
            "memory_read": "M0",
            "memory_write": "M0+query",
            "coordinate_frame": "habitat_world",
            "semantic_sg_policy": "live_sam3_dinov3_continuation",
            "navigation_memory_policy": "live_rgbd_world_map_and_trajectory_update",
            "cross_query_propagation": False,
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "rollout.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def step_discrete(env, adapter, action: int, poses: list) -> None:
    observations = env.step(int(action))
    pose = pose_from_env(env, len(poses), len(poses))
    poses.append(pose)
    adapter.process_observation(memory_observation(observations, pose), step=len(poses) - 1)


def run_one(env, habitat_config, policy, episode, args) -> None:
    session_id = session_id_of(episode)
    role = role_of(episode)
    memory = args.m0_root / "sessions" / session_id / "stages" / "M0"
    if not (memory / "memory_kb.json").is_file():
        raise FileNotFoundError(f"missing M0 for {session_id}: {memory}")
    destination = (
        args.output_root / "sessions" / session_id / "branches" / role
    )
    rollout_path = destination / "rollout.json"
    if rollout_path.is_file():
        print(f"[l1-mb] skip {session_id} {role}", flush=True)
        return
    work = args.output_root / ".working" / f"{session_id}-{role}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    install_memory_environment(args, work, memory)
    enable_live_semantic_update(args.nav_model)
    from membridge.habitat_runtime.online_adapter import OnlineINHeritSGBuilder

    class _Repeat:
        def __init__(self, item) -> None:
            self.item = item

        def __iter__(self):
            return self

        def __next__(self):
            return self.item

    env.episode_iterator = _Repeat(episode)
    observations = env.reset()
    pose = pose_from_env(env, 0, 0)
    heading = yaw_from_xyzw(pose["rotation_xyzw"])
    if hasattr(policy, "begin_episode"):
        policy.begin_episode(pose["position"], heading)
    else:
        policy.reset_episode(pose["position"], heading)
    adapter = OnlineINHeritSGBuilder(
        sensor_memory_config(habitat_config, args.sim_gpu_id),
        width=int(np.asarray(observations["rgb"]).shape[1]),
        height=int(np.asarray(observations["rgb"]).shape[0]),
        save_dir=str(work),
        device=args.device,
    )
    adapter.start_episode(str(episode.episode_id), scene_id=str(episode.scene_id))
    adapter.process_observation(memory_observation(observations, pose), step=0)
    navigator = MemoryNavigator(
        adapter=adapter,
        env=env,
        habitat_config=habitat_config,
        decision_log=work / "sg_navigation_decisions.jsonl",
    )
    navigator.current_episode_id = str(episode.episode_id)
    navigator.update_pose(pose["position"], heading, 0)
    instruction = instruction_of(episode)
    if args.route_source == "saved":
        route = load_saved_route(session_id, role, str(episode.episode_id), instruction, args)
    else:
        route = parse_level1_instruction(instruction)
    recall = {"status": "fallback", "reason": route.get("reason", "parse_failed")}
    if route.get("status") == "ready":
        try:
            recall = prepare_recall_plan(adapter, route)
        except Exception as exc:
            recall = {"status": "fallback", "reason": f"recall_failed:{type(exc).__name__}: {exc}"}
    target_info = None
    phase = f"{args.nav_model}_fallback"
    if recall.get("status") == "ready" and recall.get("selected_candidate"):
        target_info, _entry = open_object_or_explore(
            navigator,
            recall["selected_candidate"],
            recall.get("task_text") or instruction,
            step=0,
        )
        if target_info is not None:
            phase = "sg_navigation"
    poses = [pose]
    actions: list[str] = []
    names = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    anchor_reached = False
    fallback_reason = None
    while len(actions) < args.max_steps and not env.episode_over:
        pose = poses[-1]
        heading = yaw_from_xyzw(pose["rotation_xyzw"])
        navigator.update_pose(pose["position"], heading, len(actions))
        if phase == "sg_navigation":
            reached = navigator._sg_nav_reached_reason(
                step=len(actions),
                current_world_position=pose["position"],
                current_world_heading=heading,
                target_info=target_info,
            )
            if reached is not None:
                anchor_reached = True
                navigator._write_sg_decision(
                    len(actions),
                    {
                        "decision": "sg_anchor_reached_stop_pure_recall",
                        "reach_reason": reached,
                        "arrival_policy": "exact_memory_module_reachable_object_approach",
                    },
                )
                action = int(Actions.STOP)
                label = "SG:STOP"
            else:
                try:
                    action = navigator._get_sg_world_path_action(
                        step=len(actions),
                        current_world_position=pose["position"],
                        current_world_heading=heading,
                        target_info=target_info,
                    )
                except KeyError as exc:
                    action = None
                    target_info["_sg_pending_abort_reason"] = (
                        f"sg_controller_missing_key_{exc}"
                    )
                if action is None:
                    reason = target_info.pop(
                        "_sg_pending_abort_reason", "sg_controller_no_action"
                    )
                    if navigator._on_sg_nav_aborted(
                        step=len(actions),
                        reason=reason,
                        current_world_position=pose["position"],
                        current_world_heading=heading,
                        target_info=target_info,
                    ):
                        continue
                    fallback_reason = reason
                    phase = f"{args.nav_model}_fallback"
                    continue
                action = int(action)
                label = "SG:" + names[action]
            policy.observe_and_act(env, instruction, pose["position"], heading, act=False)
            actions.append(label)
            if action == int(Actions.STOP):
                env.step(action)
                break
            step_discrete(env, adapter, action, poses)
            continue
        decision = policy.observe_and_act(env, instruction, pose["position"], heading, act=True)
        if getattr(policy, "discrete_actions", False):
            action = int(decision[0] if isinstance(decision, tuple) else decision)
            actions.append(names[action])
            if action == int(Actions.STOP):
                env.step(action)
                break
            step_discrete(env, adapter, action, poses)
            continue
        from etpnav_policy import execute_waypoint

        action, angle, distance = decision
        if action == Actions.STOP or angle is None or distance is None:
            actions.append("STOP")
            env.step(int(Actions.STOP))
            break
        actions.append(f"WAYPOINT:{float(angle):.3f}:{float(distance):.3f}")
        execute_waypoint(env, adapter, float(angle), float(distance), poses, {"obs": observations})
    record = {
        "session_id": session_id,
        "episode_id": str(episode.episode_id),
        "role": role,
        "phase": "stop" if anchor_reached else phase,
        "anchor_reached": anchor_reached,
        "fallback_reason": fallback_reason,
        "arrival_policy": ARRIVAL_POLICY,
        "turn_angle_deg": float(args.turn_angle),
        "instruction": instruction,
        "coordinate_frame": "habitat_world",
        "localization": None,
        "route": {key: value for key, value in route.items() if key != "parsed"},
        "recall_status": recall.get("status"),
        "recall_reason": recall.get("reason"),
        "poses": poses,
        "executed_actions": actions,
        "steps": len(actions),
        "memory_read": "M0",
        "memory_write": "M0+query",
        "cross_query_propagation": False,
    }
    publish_branch(adapter, memory, destination, record)
    print(f"[l1-mb] {session_id} {role} phase={record['phase']} steps={len(actions)}", flush=True)


def make_runtime(args):
    from janus_policy import build_policy
    from run_ep2_explore_loc_sg import make_env, make_janus_env

    # Level-2/3 keep their own defaults. Level-1 sets the simulator turn from
    # the model: ETPNav and CMA 15°, Janus and Uni-LaViRA 30°.
    args.turn_angle = TURN_ANGLE_BY_MODEL[args.nav_model]
    args.apply_turn_angle = True
    args.update_memory = "full"
    args.query_vocabulary = str(VOCABULARY)
    if args.nav_model == "etpnav":
        from etpnav_policy import ETPNavSuffixPolicy

        env, habitat_config = make_env(args, split=args.role)
        policy = ETPNavSuffixPolicy(args)
    elif args.nav_model == "janus":
        args.janus_dataset = args.dataset
        args.dataset_split = args.role
        if args.janus_config is None:
            args.janus_config = Path(args.janus_root) / "config/vln_level1_protocol.yaml"
        env, habitat_config = make_janus_env(args)
        policy = build_policy(args, lambda _args: None)
    else:
        env, habitat_config = make_env(args, split=args.role)
        policy = build_policy(args, lambda _args: None)
    return env, habitat_config, policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nav-model", choices=("etpnav", "janus", "cma", "unilavira"), required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--etpnav-root", type=Path, default=Path("/etpnav"))
    parser.add_argument("--exp-config", type=Path, default=Path("/etpnav/run_r2r/iter_level1.yaml"))
    parser.add_argument("--task-config", type=Path, default=Path("/etpnav/run_r2r/hm3d_level1.yaml"))
    parser.add_argument("--ckpt", type=Path, default=Path("/etpnav/data/logs/checkpoints/release_r2r/ckpt.iter12000.pth"))
    parser.add_argument("--waypoint-ckpt", type=Path, default=Path("/etpnav/data/wp_pred/check_cwp_bestdist_hfov90"))
    parser.add_argument("--bert-path", type=Path, default=Path(os.environ.get("MODELS_DIR", "")) / "bert-base-uncased")
    parser.add_argument("--cma-root", type=Path, default=Path("/cma"))
    parser.add_argument(
        "--r2r-train",
        type=Path,
        default=Path("/cma/data/datasets/HM3D_VLNCE_v5/level3/level3_ep2.json.gz"),
    )
    parser.add_argument("--unilavira-root", type=Path, default=Path("/unilavira"))
    parser.add_argument("--janus-root", type=Path, default=Path("/janus/src/JanusVLN"))
    parser.add_argument("--janus-model", type=Path, default=Path("/janus/models/JanusVLN_Extra"))
    parser.add_argument("--janus-config", type=Path)
    parser.add_argument("--scenes-dir", type=Path, default=Path(os.environ.get("SCENES_DIR", "")))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-gpu-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--route-source", choices=("live", "saved"), default="live")
    parser.add_argument("--lavira-route-root", type=Path, default=DEFAULT_LAVIRA_ROUTE_ROOT)
    parser.add_argument("--route-recovery", type=Path, default=DEFAULT_ROUTE_RECOVERY)
    parser.add_argument("--session-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    env, habitat_config, policy = make_runtime(args)
    episodes = [
        episode
        for index, episode in enumerate(env.episodes)
        if role_of(episode) in {"", args.role}
        and index % args.num_shards == args.shard_index
        and (not args.session_id or session_id_of(episode) == args.session_id)
    ]
    if args.dry_run:
        print(json.dumps({"role": args.role, "selected": len(episodes), "coordinate_frame": "habitat_world"}))
        env.close()
        return 0
    install_full_exploration_overlay()
    failed = 0
    for episode in episodes:
        try:
            run_one(env, habitat_config, policy, episode, args)
        except Exception as exc:
            failed += 1
            print(f"[l1-mb] FAIL {session_id_of(episode)} {type(exc).__name__}: {exc}", flush=True)
    env.close()
    if failed:
        print(f"[l1-mb] {failed} episodes failed in {args.role}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
