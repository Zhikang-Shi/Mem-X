"""How the three levels differ.

Basic tasks navigate and build memory. They do not run the memory-task recall.
Native and +MemBridge share those basic tasks. Memory tasks then read the
built memory. Level 1 stays in one global frame and does not localize.
Level 2 and Level 3 use each episode's local frame and localize.
Level 3's memory task also explores before the query.
"""
from __future__ import annotations


def stage(level: int, role: str, pipeline: str = "membridge") -> dict:
    frame = "habitat_world" if int(level) == 1 else "episode_local"
    basic = {1: {"ep0"}, 2: {"ep0"}, 3: {"ep0", "ep1"}}[int(level)]
    if pipeline == "native" or role in basic:
        return {
            "kind": "basic",
            "coordinate_frame": frame,
            "localize": int(level) == 3 and role == "ep1",
            "explore_steps": 0,
            "build_memory": pipeline == "membridge",
        }
    return {
        "kind": "memory",
        "coordinate_frame": frame,
        "localize": int(level) >= 2,
        "explore_steps": 100 if int(level) == 3 else 0,
        "build_memory": True,
    }
