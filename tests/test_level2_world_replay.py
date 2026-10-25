"""Local exports must replay at their original Habitat positions."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'baselines/runners'))
from bootstrap import install
install()
from frames import rollout_world_poses
from run_level2_native import to_episode_local
from se2 import world_agent_to_local, yaw_from_xyzw


class Level2WorldReplayTest(unittest.TestCase):
    def test_local_export_roundtrip_preserves_render_pose_and_memory_origin(self):
        poses = [
            {'position': [8.0, 0.1, -4.0], 'rotation_xyzw': [0.0, 2**-0.5, 0.0, 2**-0.5]},
            {'position': [7.0, 0.1, -4.0], 'rotation_xyzw': [0.0, 0.0, 0.0, 1.0]},
        ]
        local = to_episode_local(poses)
        np.testing.assert_allclose(local[0]['position'], [0.0, 0.1, 0.0])
        restored = rollout_world_poses(local)
        origin = restored[0]['position']
        origin_yaw = yaw_from_xyzw(restored[0]['rotation_xyzw'])
        for source, exported, world in zip(poses, local, restored):
            self.assertEqual(world['position'], source['position'])
            self.assertEqual(world['rotation_xyzw'], source['rotation_xyzw'])
            x, y, z, _ = world_agent_to_local(world['position'], 0.0, origin, origin_yaw)
            np.testing.assert_allclose([x, y, z], exported['position'])
        self.assertNotEqual(local[0]['position'], poses[0]['position'])

    def test_world_exports_are_accepted_without_local_metadata(self):
        poses = [{'position': [4.0, 0.1, -2.0], 'rotation_xyzw': [0.0, 0.0, 0.0, 1.0]}]
        self.assertEqual(rollout_world_poses(poses), poses)

    def test_m0_loader_restores_world_xyz_before_normalizing_pose_fields(self):
        from build_ep0_memory_from_poses_local import _row_from_native
        world = [8.0, 0.1, -4.0]
        payload = {
            'episode_id': 'level2_test_ep0', 'session_id': 'level2_test',
            'scene_id': 'test.basis.glb', 'coordinate_frame': 'episode_local',
            'poses': [{'position': [0.0, 0.1, 0.0], 'world_position': world,
                       'rotation_xyzw': [0.0, 0.0, 0.0, 1.0]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rollout.json'
            path.write_text(json.dumps(payload))
            row = _row_from_native(path)
        self.assertEqual(row['poses'][0]['position'], world)
