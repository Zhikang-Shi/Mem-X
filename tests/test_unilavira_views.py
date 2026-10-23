"""Uni-LaViRA must receive four real directional observations."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines/adapt"))
try:
    import numpy as np
    import quaternion
    from unilavira_policy import UniLaViRADiscretePolicy, _action_id
except ImportError:
    UniLaViRADiscretePolicy = None


@unittest.skipIf(UniLaViRADiscretePolicy is None, "requires numpy-quaternion")
class UniLaViRAViewsTest(unittest.TestCase):
    def test_false_stop_signal_and_reasoning_text_do_not_override_direction(self):
        self.assertEqual(_action_id({
            "action": "NAVIGATE", "direction": "left", "stop_signal": False,
            "reasoning": "Do not stop or turn right yet.",
        }), 2)
        self.assertEqual(_action_id({"action": "NAVIGATE", "direction": "right", "stop_signal": False}), 3)
        self.assertEqual(_action_id({"action": "NAVIGATE", "direction": "forward", "stop_signal": True}), 0)

    def test_each_view_renders_its_direction_without_moving_the_agent(self):
        state = SimpleNamespace(
            position=np.array([1.0, 0.0, 2.0]),
            rotation=quaternion.from_rotation_vector([0.0, np.pi / 6, 0.0]),
        )

        class Sim:
            def __init__(self):
                self.calls = []

            def get_agent_state(self):
                return state

            def get_observations_at(self, **kwargs):
                self.calls.append(kwargs)
                color = len(self.calls) * 40
                return {"rgb": np.full((2, 3, 4), color, dtype=np.uint8),
                        "depth": np.full((2, 3, 1), color, dtype=np.float32)}

        sim = Sim()
        policy = UniLaViRADiscretePolicy.__new__(UniLaViRADiscretePolicy)
        policy.actor_steps = 7
        frame, views = policy._views(SimpleNamespace(sim=sim))
        self.assertEqual([v["angle"] for v in views], [0, 90, 180, 270])
        self.assertEqual([int(v["rgb"][0, 0, 0]) for v in views], [40, 80, 120, 160])
        np.testing.assert_array_equal(np.asarray(frame), views[0]["rgb"])
        for index, call in enumerate(sim.calls):
            self.assertFalse(call["keep_agent_at_new_pose"])
            np.testing.assert_array_equal(call["position"], state.position)
            expected = quaternion.from_rotation_vector([0.0, index * np.pi / 2, 0.0]) * state.rotation
            np.testing.assert_allclose(
                quaternion.as_float_array(call["rotation"]), quaternion.as_float_array(expected)
            )
            self.assertEqual(views[index]["depth"].shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
