#!/usr/bin/env python3
"""Write per-role Habitat gzip files from datasets/levelN/levelN.json."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

ROLES = {1: ("ep0", "ep1", "ep2", "ep3", "ep4"), 2: ("ep0", "ep1", "ep2"), 3: ("ep0", "ep1", "ep2")}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_gz(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


def role_of(episode: dict, level: int) -> str:
    info = episode.get("info") or {}
    return str(info.get("role") or info.get(f"level{level}_role") or info.get("lifelong_role") or "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--level", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    level = args.level
    folder = args.root / "datasets" / f"level{level}"
    data = load(folder / f"level{level}.json")
    gt = load(folder / f"level{level}_gt.json")
    by_role: dict[str, list] = {role: [] for role in ROLES[level]}
    for episode in data["episodes"]:
        role = role_of(episode, level)
        if role not in by_role:
            raise SystemExit(f"unexpected role {role!r} in episode {episode.get('episode_id')}")
        by_role[role].append(episode)
    out = folder / "habitat"
    for role, episodes in by_role.items():
        dump_gz(
            out / f"level{level}_{role}.json.gz",
            {"episodes": episodes, "instruction_vocab": data.get("instruction_vocab")},
        )
        ids = {str(episode["episode_id"]) for episode in episodes}
        dump_gz(out / f"level{level}_{role}_gt.json.gz", {key: gt[key] for key in ids if key in gt})
        print(f"level{level} {role} {len(episodes)}")


if __name__ == "__main__":
    main()
