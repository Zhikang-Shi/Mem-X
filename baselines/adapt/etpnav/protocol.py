"""Session identity shared by ETPNav sharding, resume, and rollout export."""
from __future__ import annotations


def session_id_of(episode_id, info=None):
    info = info or {}
    for key in ("session_id", "level1_session_id", "level3_session_id",
                "level3_task_id", "level2_session_id"):
        if info.get(key):
            return str(info[key])
    episode_id = str(episode_id)
    # Public Mem-X episodes encode the session before the role suffix.
    for role in ("ep0", "ep1", "ep2", "ep3", "ep4"):
        if episode_id.endswith("_" + role):
            return episode_id[: -(len(role) + 1)]
    return "episode_" + episode_id
