"""Turning in place must refresh the policy's visual observations."""
from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines/runners"))
from run_level2_native import run_episode


class NativeHeadingTest(unittest.TestCase):
    def test_turn_in_place_forwards_changed_heading_to_policy(self):
        def pose(yaw):
            return {
                "position": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)],
            }

        poses = iter([pose(math.pi / 6), pose(math.pi / 3), pose(math.pi / 3)])
        module = SimpleNamespace(
            Actions=SimpleNamespace(STOP=0),
            pose_from_env=lambda *_args: next(poses),
        )

        class Env:
            episode_over = False

            def reset(self):
                pass

            def step(self, action):
                self.episode_over = action == 0

        class Policy:
            nav_name = "janus"

            def __init__(self):
                self.headings = []

            def begin_episode(self, position, heading):
                self.start_heading = heading

            def observe_and_act(self, env, instruction, position, heading, *, act):
                self.headings.append(heading)
                return (3 if len(self.headings) == 1 else 0), None, None

        policy = Policy()
        episode = SimpleNamespace(
            episode_id="level1_00824_000_ep0", scene_id="scene.glb",
            instruction=SimpleNamespace(instruction_text="Turn toward the bench."),
            info={"role": "ep0"},
        )
        with patch.dict(sys.modules, {"run_ep1_memory_janus": module}):
            result = run_episode(Env(), policy, episode, 300, True, world_poses=True)
        self.assertEqual(result["executed_actions"], ["TURN_RIGHT", "STOP"])
        self.assertAlmostEqual(policy.start_heading, math.pi / 6)
        self.assertAlmostEqual(policy.headings[0], math.pi / 6)
        self.assertAlmostEqual(policy.headings[1], math.pi / 3)


if __name__ == "__main__":
    unittest.main()
