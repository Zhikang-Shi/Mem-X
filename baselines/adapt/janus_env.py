"""Mem-X Janus RGB-D environment for the pinned Habitat 0.1.7 runtime.

The policy uses upstream Janus model code directly, without importing its
Hydra evaluation harness or requiring unpublished protocol_benchmark.py files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from habitat import Env
from habitat.config.default import get_config
from habitat.core.registry import registry
from habitat.datasets.utils import VocabDict
from habitat.datasets.vln.r2r_vln_dataset import VLNDatasetV1
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode


@registry.register_dataset(name="MemX-Janus-v1")
class MemXJanusDataset(VLNDatasetV1):
    """Load public string episode IDs, role metadata, and world scene paths."""

    def from_json(self, json_str, scenes_dir=None):
        data = json.loads(json_str)
        self.instruction_vocab = VocabDict(
            word_list=data.get("instruction_vocab", {}).get("word_list", [])
        )
        for raw in data["episodes"]:
            values = dict(raw)
            scene = values["scene_id"]
            prefix = "data/scene_datasets/"
            if scene.startswith(prefix):
                scene = scene[len(prefix):]
            if scenes_dir is not None and not os.path.isabs(scene):
                scene = os.path.join(scenes_dir, scene)
            values["scene_id"] = scene
            # Habitat 0.1.7 validates trajectory_id as non-null. Public Mem-X
            # episodes identify trajectories by their full string episode ID.
            if values.get("trajectory_id") is None:
                values["trajectory_id"] = str(values["episode_id"])
            values["instruction"] = InstructionData(**values["instruction"])
            values["goals"] = [NavigationGoal(**goal) for goal in values["goals"]]
            self.episodes.append(VLNEpisode(**values))


def make_legacy_config(args):
    config = get_config(str(Path(__file__).with_name("janus") / "hm3d_protocol.yaml"))
    config.defrost()
    config.DATASET.DATA_PATH = str(args.janus_dataset)
    config.DATASET.SPLIT = str(getattr(args, "dataset_split", None) or "ep0")
    config.DATASET.SCENES_DIR = str(args.scenes_dir)
    config.ENVIRONMENT.MAX_EPISODE_STEPS = int(args.max_steps)
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = int(args.sim_gpu_id)
    config.SIMULATOR.TURN_ANGLE = float(getattr(args, "turn_angle", 30.0))
    config.freeze()
    return config


def make_legacy_env(args):
    return Env(config=make_legacy_config(args))
