import unittest
import pandas as pd
from src.reconstruction import reconstruct_ball


class ReconstructionTests(unittest.TestCase):
    def table(self, points, frames=None, segments=None):
        data = pd.DataFrame({"frame": list(range(len(points))) if frames is None else frames, "ball_xy": points})
        if segments is not None:
            data["segment"] = segments
        return data

    def test_interpolation_preserves_raw_and_observed(self):
        data = self.table([(0., 0.), None, (2., 2.)])
        result = reconstruct_ball(data, method="linear", max_gap_s=1, fps=1)
        self.assertIsNone(data.iloc[1].ball_xy)
        self.assertIsNone(result.iloc[1].ball_xy)
        self.assertEqual(result.iloc[1].ball_estimated_xy, (1., 1.))
        self.assertIsNone(result.iloc[0].ball_estimated_xy)
        self.assertEqual(result.ball_source.tolist(), ["observed", "linear", "observed"])

    def test_long_gap_and_edges_stay_unknown(self):
        data = self.table([None, (0., 0.), None, None, (3., 3.), None])
        result = reconstruct_ball(data, method="linear", max_gap_s=1, fps=1)
        self.assertEqual(result.ball_source.tolist(), ["unknown", "observed", "unknown", "unknown", "observed", "unknown"])

    def test_velocity_uses_two_real_anchors(self):
        data = self.table([(0., 0.), (1., 2.), None, None])
        result = reconstruct_ball(data, method="velocity", max_gap_s=2, fps=1)
        self.assertEqual(result.iloc[3].ball_estimated_xy, (3., 6.))
        self.assertEqual(result.iloc[3].anchor_distance_s, 2)

    def test_discontinuities_and_period_boundaries_are_not_bridged(self):
        for data in [self.table([(0., 0.), None, (3., 3.)], frames=[0, 2, 3]),
                     self.table([(0., 0.), None, (2., 2.)], segments=[1, 2, 2])]:
            result = reconstruct_ball(data, method="linear", max_gap_s=10, fps=1)
            self.assertEqual(result.iloc[1].ball_source, "unknown")

    def test_no_extrapolation_from_estimates(self):
        data = self.table([(0., 0.), None, (2., 2.), None])
        result = reconstruct_ball(data, method="velocity", max_gap_s=1, fps=1)
        self.assertEqual(result.iloc[3].ball_source, "unknown")
