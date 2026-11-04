import unittest
from src.data import synthetic_tracking
from src.ball import prepare_ball
from src.features import build_features
from src.algorithm import predict


class BallModuleTests(unittest.TestCase):
    def test_linear_fills_missing_and_changes_X_without_touching_raw(self):
        raw = synthetic_tracking()
        rebuilt, log = prepare_ball(raw, 'linear', .5)
        self.assertIsNone(raw.iloc[1].ball_xy)
        self.assertIsNone(rebuilt.iloc[1].ball_xy)
        self.assertEqual(rebuilt.iloc[1].ball_used_xy, (1.5, 0.))
        features = build_features(rebuilt, 'ball_used_xy')
        self.assertFalse(features.iloc[1].ball_observed)
        self.assertEqual(features.iloc[1].ball_source, 'linear')
        self.assertEqual(predict(features).iloc[1].label, 'H')
        self.assertEqual(log['estimated_frames'], 1)

    def test_none_matches_original_rule(self):
        raw = synthetic_tracking()
        rebuilt, log = prepare_ball(raw)
        self.assertEqual(predict(build_features(raw)).label.tolist(), predict(build_features(rebuilt, 'ball_used_xy')).label.tolist())
        self.assertEqual(log['estimated_frames'], 0)

    def test_real_half_boundary_and_outside_are_not_filled(self):
        raw = synthetic_tracking().iloc[:3].copy()
        raw['frame'] = [11183, 11184, 11185]
        rebuilt, log = prepare_ball(raw, 'linear', .5, 'real')
        self.assertIsNone(rebuilt.iloc[1].ball_used_xy)
        self.assertEqual(log['estimated_frames'], 0)
        raw['frame'] = [11185, 11186, 11187]
        rebuilt, log = prepare_ball(raw, 'linear', .5, 'real')
        self.assertEqual(log['estimated_frames'], 1)
