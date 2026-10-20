#!/usr/bin/env python3
"""Level-2 native navigation, without MemBridge.

CMA and Uni-LaViRA clear their own state at every episode. Janus keeps the
VGGT cache inside a session: EP0 writes it, EP1 reads it, EP2 reads that.
Saved poses are relative to that episode's start.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bootstrap import install

install()
_RUNNERS = Path(__file__).resolve().parent
HERE = _RUNNERS.parent

from se2 import world_agent_to_local, yaw_from_xyzw  # noqa: E402

STAGE = {"ep0": "M0", "ep1": "M1", "ep2": "M2", "ep3": "M3", "ep4": "M4"}
PREVIOUS_ROLE = {"ep1": "ep0", "ep2": "ep1", "ep3": "ep2", "ep4": "ep3"}


def episode_info(episode) -> dict:
    return dict(getattr(episode, "info", {}) or {})


def session_id_of(episode) -> str:
    episode_id = str(getattr(episode, "episode_id", "") or "")
    for role in ("ep0", "ep1", "ep2", "ep3", "ep4"):
        suffix = f"_{role}"
        if episode_id.endswith(suffix):
            return episode_id[: -len(suffix)]
    info = episode_info(episode)
    return str(info.get("session_id") or episode_id)


def role_of(episode) -> str:
    info = episode_info(episode)
    return str(info.get("role") or info.get("lifelong_role") or info.get("level1_role") or info.get("level3_role") or info.get("level2_role") or "")


def instruction_of(episode) -> str:
    text = getattr(episode, "instruction", None)
    if text is None:
        return ""
    value = getattr(text, "instruction_text", None)
    return str(value or "")


def to_episode_local(poses: list[dict]) -> list[dict]:
    if not poses:
        return poses
    start = poses[0]["position"]
    start_yaw = yaw_from_xyzw(poses[0]["rotation_xyzw"])
    converted = []
    for pose in poses:
        x, y, z, yaw = world_agent_to_local(
            pose["position"], yaw_from_xyzw(pose["rotation_xyzw"]), start, start_yaw
        )
        converted.append(
            {
                **pose,
                "world_position": [float(v) for v in pose["position"]],
                "position": [float(x), float(y), float(z)],
                "yaw": float(yaw),
            }
        )
    return converted


def convert_tree(root: Path) -> int:
    count = 0
    for path in sorted(root.rglob("rollout.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("coordinate_frame") == "episode_local":
            continue
        poses = payload.get("poses") or []
        if not poses or "position" not in poses[0] or "rotation_xyzw" not in poses[0]:
            continue
        payload["poses"] = to_episode_local(poses)
        payload["positions"] = [pose["position"] for pose in payload["poses"]]
        payload["coordinate_frame"] = "episode_local"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        count += 1
    print(f"[l2-native] converted {count} rollouts under {root}", flush=True)
    return count


def write_rollout(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_episode(
    env, policy, episode, max_steps: int, reset_memory: bool, world_poses: bool = False
) -> dict:
    from run_ep1_memory_janus import Actions, pose_from_env

    class _Repeat:
        def __init__(self, item) -> None:
            self.item = item

        def __iter__(self):
            return self

        def __next__(self):
            return self.item

    env.episode_iterator = _Repeat(episode)
    env.reset()
    instruction = instruction_of(episode)
    pose = pose_from_env(env, 0, 0)
    if reset_memory:
        policy.begin_episode(pose["position"], 0.0)
    else:
        policy.history = []
        policy._key = None
        policy.actor_steps = 0
    poses = [pose]
    actions: list[str] = []
    while len(actions) < max_steps and not env.episode_over:
        action, _, _ = policy.observe_and_act(
            env, instruction, pose["position"], 0.0, act=True
        )
        if int(action) == int(Actions.STOP):
            env.step(int(Actions.STOP))
            actions.append("STOP")
            poses.append(pose_from_env(env, len(poses), len(actions)))
            break
        env.step(int(action))
        actions.append(["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"][int(action)])
        pose = pose_from_env(env, len(poses), len(actions))
        poses.append(pose)
    return {
        "episode_id": str(episode.episode_id),
        "scene_id": str(episode.scene_id),
        "session_id": session_id_of(episode),
        "role": role_of(episode),
        "instruction": instruction,
        "poses": poses if world_poses else to_episode_local(poses),
        "executed_actions": actions,
        "steps": len(actions),
        "coordinate_frame": "habitat_world" if world_poses else "episode_local",
        "nav_model": getattr(policy, "nav_name", ""),
        "memory_cleared": bool(reset_memory),
    }


def cache_path(root: Path, session_id: str, role: str) -> Path:
    return root / "janus_cache" / f"{session_id}_{STAGE[role]}.pt"


def _move_cache(value, device: str):
    import torch

    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, list):
        return [_move_cache(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_cache(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_cache(item, device) for key, item in value.items()}
    return value


def save_janus_cache(policy, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    memory = getattr(policy.model.model, "past_key_values_vggt", None)
    torch.save(_move_cache(memory, "cpu"), path)


def load_janus_cache(policy, path: Path) -> None:
    import torch

    memory = torch.load(path, map_location="cpu")
    policy.model.model.past_key_values_vggt = _move_cache(memory, str(policy.model.vggt_device))


def make_policy_and_env(args):
    from janus_policy import build_policy
    from run_ep2_explore_loc_sg import make_env, make_janus_env

    if args.nav_model == "janus":
        args.janus_dataset = args.dataset
        args.dataset_split = args.role
        if args.janus_config is None:
            args.janus_config = Path(args.janus_root) / "config/vln_level2_protocol.yaml"
        env, _ = make_janus_env(args)
    else:
        env, _ = make_env(args, split=args.role)
    policy = build_policy(args, lambda _args: None)
    return env, policy


def selected_episodes(env, args):
    episodes = [
        episode
        for episode in env.episodes
        if not args.role or role_of(episode) in {"", args.role}
    ]
    if args.nav_model == "janus":
        sessions = sorted({session_id_of(episode) for episode in episodes})
        keep = set(sessions[args.shard_index :: args.num_shards])
        episodes = [episode for episode in episodes if session_id_of(episode) in keep]
    else:
        episodes = [
            episode
            for index, episode in enumerate(episodes)
            if index % args.num_shards == args.shard_index
        ]
    if args.session_id:
        episodes = [episode for episode in episodes if session_id_of(episode) == args.session_id]
    return episodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convert-root", type=Path)
    parser.add_argument("--nav-model", choices=("janus", "cma", "unilavira"))
    parser.add_argument("--role", choices=("ep0", "ep1", "ep2", "ep3", "ep4"))
    parser.add_argument("--turn-angle", type=float)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--etpnav-root", type=Path, default=Path("/etpnav"))
    parser.add_argument("--exp-config", type=Path, default=Path("/etpnav/run_r2r/iter_level2.yaml"))
    parser.add_argument("--task-config", type=Path, default=Path("/etpnav/run_r2r/hm3d_level3.yaml"))
    parser.add_argument("--ckpt", type=Path, default=Path("/etpnav/data/logs/checkpoints/release_r2r/ckpt.iter12000.pth"))
    parser.add_argument("--waypoint-ckpt", type=Path, default=Path("/etpnav/data/wp_pred/check_cwp_bestdist_hfov90"))
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
    parser.add_argument("--session-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--world-poses",
        action="store_true",
        help="Keep habitat world poses. Level 1 uses this. Level 2/3 stay episode-local.",
    )
    args = parser.parse_args()
    if args.convert_root is not None:
        convert_tree(args.convert_root.resolve())
        return 0
    if not args.nav_model or not args.role or not args.dataset or not args.output_root:
        parser.error("--nav-model, --role, --dataset, and --output-root are required")
    if args.turn_angle is None:
        args.turn_angle = 30.0
    else:
        args.apply_turn_angle = True
    args.update_memory = "off"
    env, policy = make_policy_and_env(args)
    episodes = selected_episodes(env, args)
    if args.dry_run:
        print(json.dumps({"role": args.role, "nav_model": args.nav_model, "selected": len(episodes)}), flush=True)
        env.close()
        return 0
    cache_root = (args.cache_root or args.output_root.parent).resolve()
    stage = STAGE[args.role]
    for episode in episodes:
        session_id = session_id_of(episode)
        reset = True
        previous = PREVIOUS_ROLE.get(args.role)
        if args.nav_model == "janus" and previous:
            path = cache_path(cache_root, session_id, previous)
            if path.is_file():
                load_janus_cache(policy, path)
                reset = False
        payload = run_episode(env, policy, episode, args.max_steps, reset, args.world_poses)
        if args.nav_model != "janus":
            chain = "cleared_each_episode"
        elif args.role in {"ep3", "ep4"}:
            chain = "EP0 -> M0 -> EP1 -> M1 -> EP2 -> M2 -> EP3 -> M3 -> EP4 -> M4"
        else:
            chain = "EP0 -> M0 -> EP1 -> M1 -> EP2 -> M2"
        payload["memory_chain"] = chain
        dest = args.output_root / "sessions" / session_id / "stages" / stage / "rollout.json"
        write_rollout(dest, payload)
        if args.nav_model == "janus":
            save_janus_cache(policy, cache_path(cache_root, session_id, args.role))
        print(
            f"[l2-native] {args.nav_model} {session_id} {args.role} "
            f"steps={payload['steps']} reset={reset}",
            flush=True,
        )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
