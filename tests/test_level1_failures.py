"""A failed episode must fail the shell job instead of reporting success."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines/runners"))
try:
    import run_level1_membridge as runner
except ImportError:
    runner = None


@unittest.skipIf(runner is None, "requires MemBridge runtime dependencies")
class Level1FailureTest(unittest.TestCase):
    def test_episode_exception_produces_nonzero_status(self):
        episode = SimpleNamespace(episode_id="level1_00824_000_ep1", info={"role": "ep1"})
        env = SimpleNamespace(episodes=[episode], close=Mock())
        argv = [
            "runner", "--nav-model", "janus", "--role", "ep1", "--dataset", "/data.json.gz",
            "--m0-root", "/m0", "--output-root", "/output",
        ]
        with patch.object(sys, "argv", argv), \
             patch.object(runner, "make_runtime", return_value=(env, None, None)), \
             patch.object(runner, "install_full_exploration_overlay"), \
             patch.object(runner, "run_one", side_effect=RuntimeError("episode failed")), \
             contextlib.redirect_stdout(io.StringIO()):
            code = runner.main()
        self.assertEqual(code, 1)
        env.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
