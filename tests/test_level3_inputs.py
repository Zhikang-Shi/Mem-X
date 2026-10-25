"""Public L3 task selection and metric-depth localization inputs."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'baselines/runners'))
from bootstrap import install
install()
import run_ep1_memory_etpnav_local_loc as ep1
import run_ep2_explore_loc_sg as ep2
from run_ep2_explore_loc_sg import episodes_for_nav
from run_ep2_memory_janus import dataset_tasks
from run_ep2_memory_janus import publish_m2
from frames import EpisodeStart


def episode(sid, role):
    return {'episode_id': f'{sid}_{role}', 'scene_id': 'test.basis.glb',
            'info': {'role': role}, 'instruction': {'instruction_text': 'Walk ahead.'}}


class Level3InputsTest(unittest.TestCase):
    def _publish_fixture(self, root):
        parent, live, dest = root/'M1', root/'live', root/'M2'
        parent.mkdir()
        live.mkdir()
        for name in ['memory_kb.json','builder_state.pkl','world_metric_map.npz','trajectory_graph.npz']:
            (parent/name).write_text('old-'+name)
            (live/name).write_text('new-'+name)
        for name in ['frames','objects']:
            (parent/name).mkdir()
            (parent/name/'old.txt').write_text('old')
            (live/name).mkdir()
            (live/name/'new.txt').write_text('new')
        meta = {'session_id':'level3_000','episode_id':'level3_000_ep2','route':{},'sg_handoff':{}}
        return parent,live,dest,meta

    def test_full_m2_preserves_live_semantic_state_and_observation_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent,live,dest,meta = self._publish_fixture(Path(tmp))
            publish_m2(parent,live,dest,meta,update_memory='full')
            self.assertEqual((dest/'memory_kb.json').read_text(),'new-memory_kb.json')
            self.assertEqual((dest/'builder_state.pkl').read_text(),'new-builder_state.pkl')
            self.assertTrue((dest/'frames/new.txt').is_file())
            self.assertTrue((dest/'frames/old.txt').is_file())
            self.assertTrue((dest/'objects/new.txt').is_file())
            self.assertTrue((dest/'objects/old.txt').is_file())
            self.assertEqual((parent/'memory_kb.json').read_text(),'old-memory_kb.json')
            self.assertEqual(json.loads((dest/'stage_manifest.json').read_text())['memory_update'],'full')

    def test_nav_m2_keeps_parent_semantics_and_updates_navigation(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent,live,dest,meta = self._publish_fixture(Path(tmp))
            publish_m2(parent,live,dest,meta)
            self.assertEqual((dest/'memory_kb.json').read_text(),'old-memory_kb.json')
            self.assertEqual((dest/'world_metric_map.npz').read_text(),'new-world_metric_map.npz')
            self.assertTrue((dest/'frames/old.txt').is_file())
            self.assertFalse((dest/'frames/new.txt').exists())

    def test_full_m2_missing_semantic_state_does_not_publish_old_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent,live,dest,meta = self._publish_fixture(Path(tmp))
            (live/'builder_state.pkl').unlink()
            with self.assertRaisesRegex(RuntimeError,'full EP2 did not produce'):
                publish_m2(parent,live,dest,meta,update_memory='full')
            self.assertFalse(dest.exists())

    def test_ep1_selects_public_string_id_before_requiring_other_memories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / 'episodes.json'
            dataset.write_text(json.dumps({'episodes': [episode('level3_000', 'ep1'),
                                                       episode('level3_001', 'ep1')]}))
            memory = root / 'sessions/level3_000/stages/M0'
            memory.mkdir(parents=True)
            for name in [*ep1.CORE_M0_FILES, 'frames/confirmed_keyframes.jsonl', 'ep0_local_origin.json']:
                path = memory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            args = SimpleNamespace(dataset=dataset, m0_root=root, expected_count=157,
                                   session_id=['level3_000'], episode_id=[], num_shards=1,
                                   shard_index=0, limit=None)
            tasks = ep1.load_tasks(args)
        self.assertEqual([t['episode_id'] for t in tasks], ['level3_000_ep1'])
        self.assertEqual(tasks[0]['session_id'], 'level3_000')

    def test_ep2_reads_public_role_and_dictionary_episode_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'episodes.json'
            path.write_text(json.dumps({'episodes': [episode('level3_000', 'ep1'),
                                                     episode('level3_000', 'ep2')]}))
            tasks = dataset_tasks(path)
        self.assertEqual(list(tasks), ['level3_000'])
        self.assertEqual(tasks['level3_000']['episode_id'], 'level3_000_ep2')

    def test_janus_ep2_lookup_uses_public_role(self):
        values = [episode('level3_000', 'ep1'), episode('level3_000', 'ep2')]
        env = SimpleNamespace(episodes=[SimpleNamespace(**e) for e in values])
        self.assertEqual(episodes_for_nav(env, 'janus')['level3_000'].episode_id, 'level3_000_ep2')

    def test_ep2_default_selection_includes_all_public_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / 'episodes.json'
            dataset.write_text(json.dumps({'episodes': [episode('level3_000', 'ep2'),
                                                       episode('level3_001', 'ep2')]}))
            for sid in ['level3_000', 'level3_001']:
                for stage in ['M0', 'M1']:
                    memory = root / f'sessions/{sid}/stages/{stage}'
                    for name in [*ep1.CORE_M0_FILES, 'frames/confirmed_keyframes.jsonl', 'ep0_local_origin.json']:
                        path = memory / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.touch()
            args = SimpleNamespace(dataset=dataset, m0_root=root, m1_root=root, expected_count=2,
                                   session_id=[], session_id_file=None, episode_id=[], path_map=[],
                                   num_shards=1, shard_index=0, limit=None)
            tasks = ep2.load_tasks(args)
        self.assertEqual([t['session_id'] for t in tasks], ['level3_000','level3_001'])

    def test_float_depth_file_under_one_and_half_metres_keeps_metric_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'depth.tiff'
            Image.fromarray(np.full((2, 2), 1.2, dtype=np.float32)).save(path)
            depth = ep1._depth_m(path)
        np.testing.assert_allclose(depth, 1.2)

    def test_live_close_range_depth_keeps_metric_scale(self):
        obs = {'rgb': np.zeros((2, 2, 3), dtype=np.uint8),
               'depth': np.full((2, 2, 1), 1.2, dtype=np.float32)}
        pose = {'position': [0.0, 0.1, 0.0], 'rotation_xyzw': [0.0, 0.0, 0.0, 1.0]}
        start = EpisodeStart(xyz=np.asarray(pose['position']), yaw=0.0, episode_id='test', role='ep1')
        frame = ep1.live_frame(obs, pose, start, 0)
        np.testing.assert_allclose(frame.depth_m, 1.2)


if __name__ == '__main__':
    unittest.main()
