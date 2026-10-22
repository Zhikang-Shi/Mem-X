"""Runtime preparation must preserve upstream files and reject unknown edits."""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines/adapt"))
import etpnav_runtime as adapter

from etpnav.protocol import session_id_of


class SessionIdentityTest(unittest.TestCase):
    def test_public_ids_share_one_session_across_roles(self):
        for role in ("ep0", "ep1", "ep2", "ep3", "ep4"):
            self.assertEqual(session_id_of("level1_00800_000_" + role), "level1_00800_000")

    def test_legacy_metadata_takes_precedence(self):
        self.assertEqual(session_id_of("824970", {"level1_session_id": "legacy"}), "legacy")
        self.assertEqual(session_id_of("824970", {"session_id": "public"}), "public")
        self.assertEqual(session_id_of("824970"), "episode_824970")


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "upstream"
        self.assets = self.root / "adapter"
        self.cache = self.root / "runs"
        for folder in (self.source / "vlnce_baselines", self.source / "run_r2r",
                       self.source / "data/logs/checkpoints", self.assets / "patches",
                       self.assets / "configs"):
            folder.mkdir(parents=True)
        self.original = b"value = 1\n"
        self.adapted = b"value = 2\n"
        (self.source / "run.py").write_bytes(self.original)
        (self.source / "vlnce_baselines/__init__.py").write_text("")
        (self.source / "data/logs/checkpoints/model.pt").write_bytes(b"weights")
        delta = "".join(difflib.unified_diff(
            self.original.decode().splitlines(True), self.adapted.decode().splitlines(True),
            fromfile="a/run.py", tofile="b/run.py",
        ))
        (self.assets / "patches/run.patch").write_text(delta)
        (self.assets / "manifest.json").write_text(json.dumps({
            "upstream_commit": "test-commit", "files": {"run.py": {
                "upstream_sha256": hashlib.sha256(self.original).hexdigest(),
                "adapted_sha256": hashlib.sha256(self.adapted).hexdigest(),
                "patch": "run.patch",
            }},
        }))
        (self.assets / "configs/hm3d_level1.yaml").write_text(
            "SCENES_DIR: @SCENES_DIR@\nSCENE_DATASET: @SCENE_DATASET_CONFIG@\n"
        )
        override = patch.object(adapter, "ADAPTER", self.assets)
        override.start()
        self.addCleanup(override.stop)

    def prepare(self, **kwargs):
        return adapter.prepare_runtime(self.source, self.cache, repo=self.root, **kwargs)

    def test_official_source_stays_pristine_and_logs_are_separate(self):
        runtime = self.prepare()
        self.assertEqual((runtime / "run.py").read_bytes(), self.adapted)
        self.assertEqual((self.source / "run.py").read_bytes(), self.original)
        self.assertFalse((runtime / "run.py").is_symlink())
        self.assertEqual((runtime / "data/logs/checkpoints/model.pt").read_bytes(), b"weights")
        (runtime / "data/logs/running_log/eval.log").write_text("done")
        self.assertFalse((self.source / "data/logs/running_log").exists())
        self.assertEqual(self.prepare(), runtime)

    def test_recovered_adaptation_is_accepted_without_double_patching(self):
        (self.source / "run.py").write_bytes(self.adapted)
        self.assertEqual((self.prepare() / "run.py").read_bytes(), self.adapted)

    def test_unknown_source_fails_without_mutation(self):
        modified = b"value = 3\n"
        (self.source / "run.py").write_bytes(modified)
        with self.assertRaisesRegex(ValueError, "Unsupported ETPNav source"):
            self.prepare()
        self.assertEqual((self.source / "run.py").read_bytes(), modified)
        self.assertFalse(self.cache.exists())

    def test_scene_paths_are_quoted_and_select_a_distinct_runtime(self):
        scenes = self.root / "scenes with spaces"
        runtime = self.prepare(scenes_dir=scenes)
        config = (runtime / "run_r2r/hm3d_level1.yaml").read_text()
        self.assertIn(json.dumps(str(scenes)), config)
        self.assertNotIn("@SCENES_DIR@", config)
        self.assertNotEqual(self.prepare(), runtime)

    def test_adapter_changes_invalidate_cached_runtime(self):
        first = self.prepare()
        (self.assets / "configs/hm3d_level1.yaml").write_text("updated: true\n")
        second = self.prepare()
        self.assertNotEqual(first, second)
        self.assertEqual((second / "run_r2r/hm3d_level1.yaml").read_text(), "updated: true\n")

    def test_direct_arguments_rebase_configs_and_preserve_custom_settings(self):
        runtime = self.prepare()
        args = adapter.adapt_arguments([
            "--etpnav-root", str(runtime), "--exp-config=/custom/experiment.yaml",
            "--task-config", str(self.source / "run_r2r/hm3d_level1.yaml"),
            "--role", "ep1",
        ], 1)
        self.assertIn("--exp-config=/custom/experiment.yaml", args)
        self.assertIn(str(runtime / "run_r2r/hm3d_level1.yaml"), args)
        self.assertIn("--role", args)
        self.assertEqual(args.count("--etpnav-root"), 1)


if __name__ == "__main__":
    unittest.main()
