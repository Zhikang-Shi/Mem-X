"""Behavior of the level switch the three shell scripts share."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

RUNNERS = Path(__file__).resolve().parents[1] / "baselines" / "runners"
sys.path.insert(0, str(RUNNERS))

from stage import stage  # noqa: E402


class StageTest(unittest.TestCase):
    def test_level1_basic_builds_memory(self) -> None:
        row = stage(1, "ep0", "membridge")
        self.assertEqual(row["kind"], "basic")
        self.assertEqual(row["coordinate_frame"], "habitat_world")
        self.assertFalse(row["localize"])
        self.assertEqual(row["explore_steps"], 0)
        self.assertTrue(row["build_memory"])

    def test_native_basic_does_not_build_memory(self) -> None:
        row = stage(2, "ep0", "native")
        self.assertEqual(row["kind"], "basic")
        self.assertFalse(row["build_memory"])
        self.assertFalse(row["localize"])

    def test_level3_ep1_localizes_while_basic(self) -> None:
        row = stage(3, "ep1", "membridge")
        self.assertEqual(row["kind"], "basic")
        self.assertTrue(row["localize"])
        self.assertEqual(row["explore_steps"], 0)

    def test_level3_memory_explores(self) -> None:
        row = stage(3, "ep2", "membridge")
        self.assertEqual(row["kind"], "memory")
        self.assertEqual(row["coordinate_frame"], "episode_local")
        self.assertTrue(row["localize"])
        self.assertEqual(row["explore_steps"], 100)


if __name__ == "__main__":
    unittest.main()
