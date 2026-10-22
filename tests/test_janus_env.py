"""Regression coverage for public Mem-X data in the pinned Habitat runtime."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines/adapt"))
try:
    from janus_env import MemXJanusDataset, make_legacy_config
except ImportError:
    MemXJanusDataset = None


@unittest.skipIf(MemXJanusDataset is None, "requires the memx Habitat environment")
class JanusDatasetTest(unittest.TestCase):
    def test_public_ids_metadata_and_scene_resolution_survive_loading(self):
        raw = {
            "episode_id": "level1_00824_000_ep1",
            "scene_id": "data/scene_datasets/hm3dsem/val/scene.glb",
            "start_position": [0.0, 0.0, 0.0],
            "start_rotation": [0.0, 0.0, 0.0, 1.0],
            "goals": [{"position": [1.0, 0.0, 0.0], "radius": 3.0}],
            "instruction": {"instruction_text": "Return to the bench.", "instruction_tokens": []},
            "reference_path": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "info": {"role": "ep1", "task_type": "memory"},
        }
        absolute = dict(raw, episode_id="level1_00824_000_ep2", scene_id="/other/scene.glb")
        dataset = MemXJanusDataset()
        dataset.from_json(json.dumps({"episodes": [raw, absolute]}), scenes_dir="/scenes")
        episode, second = dataset.episodes
        self.assertEqual(episode.episode_id, raw["episode_id"])
        self.assertEqual(episode.info, raw["info"])
        self.assertEqual(episode.scene_id, "/scenes/hm3dsem/val/scene.glb")
        self.assertEqual(episode.instruction.instruction_text, "Return to the bench.")
        self.assertEqual(episode.goals[0].position, [1.0, 0.0, 0.0])
        self.assertEqual(second.scene_id, "/other/scene.glb")

    def test_runtime_overrides_select_query_split_and_step_budget(self):
        args = SimpleNamespace(
            janus_dataset=Path("/data/level1_ep1.json.gz"), dataset_split="ep1",
            scenes_dir=Path("/scenes"), max_steps=77, sim_gpu_id=1, turn_angle=30.0,
        )
        config = make_legacy_config(args)
        self.assertEqual(config.DATASET.SPLIT, "ep1")
        self.assertEqual(config.DATASET.DATA_PATH, "/data/level1_ep1.json.gz")
        self.assertEqual(config.ENVIRONMENT.MAX_EPISODE_STEPS, 77)
        self.assertEqual(config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID, 1)
        self.assertFalse(config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH)
        self.assertEqual(config.SIMULATOR.TURN_ANGLE, 30.0)


if __name__ == "__main__":
    unittest.main()
