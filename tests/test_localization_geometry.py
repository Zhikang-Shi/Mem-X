"""Recover known Habitat XZ transforms, including rotated query frames."""
from pathlib import Path
import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'membridge/localization'))
from geom_lock import GeomCandidate, _se2_from_two, _yaw_alias_gap, decide_geom_lock, se2_ransac
from se2 import SE2, se2_error


class LocalizationGeometryTest(unittest.TestCase):
    def test_clockwise_quarter_turn_uses_habitat_xz_yaw(self):
        source = np.array([[0.0, 0.0], [1.0, 0.0]])
        # A positive Habitat yaw maps +X to -Z, then translates by (2,3).
        destination = np.array([[2.0, 3.0], [2.0, 2.0]])
        estimated = _se2_from_two(source, destination)
        self.assertAlmostEqual(estimated.yaw, math.pi/2)
        np.testing.assert_allclose(estimated.apply(source), destination, atol=1e-12)

    def test_ransac_recovers_nonzero_yaw_with_outlier_correspondences(self):
        rng = np.random.default_rng(8)
        source = rng.uniform(-4.0, 4.0, size=(64, 2))
        truth = SE2(1.2, -0.7, math.radians(75.0))
        destination = truth.apply(source)
        destination[:12] = rng.uniform(10.0, 20.0, size=(12, 2))
        estimated, inliers, _ = se2_ransac(source, destination)
        self.assertEqual(inliers, 52)
        translation, rotation = se2_error(estimated, truth)
        self.assertLess(translation, 1e-10)
        self.assertLess(rotation, 1e-10)

    def test_agreeing_map_views_are_one_pose_hypothesis(self):
        candidates = [
            GeomCandidate(200, 200, 174, 0.87, SE2(0.25, 0.95, -2.88)),
            GeomCandidate(190, 190, 157, 0.83, SE2(0.27, 0.98, -2.89)),
            GeomCandidate(150, 150, 126, 0.84, SE2(0.26, 0.96, -2.88)),
        ]
        locked, _, _ = decide_geom_lock(candidates)
        self.assertTrue(locked)

    def test_distinct_similar_score_pose_still_blocks_lock(self):
        candidates = [
            GeomCandidate(200, 200, 174, 0.87, SE2(0.25, 0.95, -2.88)),
            GeomCandidate(190, 190, 157, 0.83, SE2(4.0, 0.98, -2.89)),
        ]
        locked, reason, _ = decide_geom_lock(candidates)
        self.assertFalse(locked)
        self.assertIn('not unique', reason)

    def test_empty_occupancy_is_not_a_yaw_contradiction(self):
        empty = SimpleNamespace(occupied_points=lambda:np.zeros((0,2)))
        occupied_map = SimpleNamespace(occupied=np.ones((1,1),dtype=bool))
        with patch('localize.build_map_occupancy',return_value=occupied_map), \
             patch('localize.build_query_occupancy',return_value=empty), \
             patch('localize._score_transform') as score:
            valid, _, _ = _yaw_alias_gap([object()], [object()], SE2(0,0,0))
        self.assertTrue(valid)
        score.assert_not_called()

    def test_observed_occupancy_alias_is_still_rejected(self):
        query = SimpleNamespace(occupied_points=lambda:np.array([[0.0,0.0]]))
        occupied_map = SimpleNamespace(occupied=np.ones((1,1),dtype=bool))
        with patch('localize.build_map_occupancy',return_value=occupied_map), \
             patch('localize.build_query_occupancy',return_value=query), \
             patch('localize._score_transform',return_value=0.5):
            valid, _, _ = _yaw_alias_gap([object()], [object()], SE2(0,0,0))
        self.assertFalse(valid)
