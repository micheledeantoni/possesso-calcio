import unittest
from src.data import synthetic_tracking
from src.quality import quality_timeline, missing_runs


class QualityTests(unittest.TestCase):
    def test_missing_duration_and_exact_pairs(self):
        tracking = synthetic_tracking()
        tracking.at[0, 'home_xy'] = [(1., 1.), (1., 1.)]
        timeline = quality_timeline(tracking)
        self.assertEqual(timeline.iloc[0].exact_same_team_pairs, 1)
        self.assertEqual(missing_runs(timeline), [1 / 25])

    def test_gaps_do_not_bridge_nonconsecutive_records(self):
        tracking = synthetic_tracking().iloc[[1, 3]].copy()
        tracking['ball_xy'] = None
        self.assertEqual(missing_runs(quality_timeline(tracking)), [1 / 25, 1 / 25])
