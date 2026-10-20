"""Import paths shared by baselines/run.py and the level runners.

Shell scripts keep calling baselines/run.py. Runners then import flat module
names (se2, actions, run_ep1_memory_janus) from membridge.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

RUNNERS = Path(__file__).resolve().parent
BASELINES = RUNNERS.parent
REPO = BASELINES.parent
VOCABULARY = (
    REPO
    / "membridge/semantic_scene_graph/config/level1_runtime_query_vocabulary_75.json"
)


def install() -> Path:
    membridge = REPO / "membridge"
    ordered = (
        REPO,
        membridge / "localization",
        membridge / "exploration",
        membridge / "runners",
        BASELINES,
        BASELINES / "adapt",
        RUNNERS,
    )
    for path in ordered:
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    lightglue = Path(os.environ.get("LIGHTGLUE_ROOT", ""))
    if not lightglue.is_dir():
        lightglue = Path.home() / "LightGlue"
    if lightglue.is_dir():
        entry = str(lightglue)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return REPO
