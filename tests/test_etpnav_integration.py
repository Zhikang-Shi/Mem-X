"""Optional CPU integration tests against a real upstream checkout.

Run in the memx environment with ETPNAV_TEST_SOURCE set to an ETPNav checkout.
No GPU, scene rendering, VLM call, or model checkpoint loading is needed.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "baselines/adapt"))
from etpnav_runtime import prepare_runtime


@unittest.skipUnless(os.environ.get("ETPNAV_TEST_SOURCE"), "set ETPNAV_TEST_SOURCE for CPU integration")
class ETPNavIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.runtime = prepare_runtime(Path(os.environ["ETPNAV_TEST_SOURCE"]),
                                      Path(cls.temporary.name) / "runtime")
        cls.original_cwd = Path.cwd()
        cls.addClassCleanup(os.chdir, cls.original_cwd)
        os.chdir(cls.runtime)
        sys.path.insert(0, str(cls.runtime))
        sys.path.insert(0, str(cls.runtime / "third_party/CLIP"))
        # Import upstream via its adapted entrypoint, including compatibility imports.
        runpy.run_path(str(cls.runtime / "run.py"), run_name="memx_etpnav_test")
        from vlnce_baselines.config.default import get_config
        from habitat_extensions.task import VLNCEDatasetV1
        from vlnce_baselines.ss_trainer_ETP import RLTrainer
        cls.get_config = staticmethod(get_config)
        cls.dataset_class = VLNCEDatasetV1
        cls.trainer_class = RLTrainer

    def config(self, level=1):
        cfg = self.get_config(str(self.runtime / f"run_r2r/iter_level{level}.yaml"))
        cfg.defrost()
        cfg.local_rank = 0
        cfg.freeze()
        return cfg

    def test_level_configs_resolve_to_memx_datasets(self):
        for level in (1, 2, 3):
            cfg = self.config(level)
            self.assertEqual(cfg.EVAL.NUM_SHARDS, 1)
            self.assertFalse(cfg.EVAL.DUMP_MEMORY)
            self.assertEqual(cfg.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS, 300)
            self.assertEqual(cfg.TASK_CONFIG.DATASET.DATA_PATH,
                             str(REPO / f"datasets/level{level}/habitat/level{level}_{{split}}.json.gz"))

    def test_public_episode_filter_and_local_tokenization(self):
        cfg = self.config()
        path = cfg.TASK_CONFIG.DATASET.DATA_PATH.format(split="ep0")
        with gzip.open(path, "rt") as handle:
            episode_id = json.load(handle)["episodes"][0]["episode_id"]
        cfg.defrost()
        cfg.TASK_CONFIG.DATASET.EPISODES_ALLOWED = [episode_id]
        cfg.freeze()
        dataset = self.dataset_class(cfg.TASK_CONFIG.DATASET)
        self.assertEqual([str(ep.episode_id) for ep in dataset.episodes], [episode_id])
        tokens = dataset.episodes[0].instruction.instruction_tokens
        self.assertGreater(len(tokens), 2)
        self.assertEqual(tokens[0], 101)
        self.assertEqual(tokens[-1], 102)

    def test_shards_partition_dataset_and_resume_uses_public_session(self):
        partitions = []
        for shard in range(3):
            cfg = self.config()
            cfg.defrost()
            cfg.EVAL.SHARD_ID = shard
            cfg.EVAL.NUM_SHARDS = 3
            cfg.freeze()
            trainer = self.trainer_class.__new__(self.trainer_class)
            trainer.config = cfg
            partitions.append(set(trainer.collect_val_traj()))
        self.assertEqual(sum(map(len, partitions)), 315)
        self.assertEqual(len(set.union(*partitions)), 315)
        episode_id = next(iter(partitions[0]))
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config()
            cfg.defrost()
            cfg.EVAL.DUMP_MEMORY = True
            cfg.EVAL.MEMORY_ROOT = directory
            cfg.freeze()
            session = episode_id.rsplit("_", 1)[0]
            stage = Path(directory) / "sessions" / session / "stages/M0"
            stage.mkdir(parents=True)
            (stage / "graphmap.pt").touch()
            (stage / "rollout.json").write_text("{}")
            trainer = self.trainer_class.__new__(self.trainer_class)
            trainer.config = cfg
            remaining = trainer.collect_val_traj()
            self.assertEqual(len(remaining), 314)
            self.assertNotIn(episode_id, remaining)


if __name__ == "__main__":
    unittest.main()
