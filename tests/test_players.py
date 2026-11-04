import unittest
from src.data import synthetic_tracking
from src.players import filter_players, player_counts
from src.features import build_features
from src.algorithm import predict


class PlayerTests(unittest.TestCase):
    def fixture(self):
        data = synthetic_tracking().iloc[:1].copy()
        data.at[0, "home_xy"] = [(float(i), 0.) for i in range(12)]
        data.at[0, "away_xy"] = [(float(i), 1.) for i in range(12)]
        return data

    def test_filter_caps_without_changing_original(self):
        data = self.fixture()
        filtered, log = filter_players(data, "nearest22")
        self.assertEqual(player_counts(data).iloc[0].total, 24)
        self.assertEqual(player_counts(filtered).iloc[0].total, 22)
        self.assertEqual(log["observations_removed"], 2)
        self.assertEqual(predict(build_features(data)).label.tolist(), predict(build_features(filtered)).label.tolist())

    def test_missing_ball_skips_and_none_is_identity(self):
        data = self.fixture(); data.at[0, "ball_xy"] = None
        filtered, log = filter_players(data, "nearest22")
        self.assertEqual(player_counts(filtered).iloc[0].total, 24)
        self.assertEqual(log["overfull_frames_skipped_missing_ball"], 1)
        filtered, log = filter_players(data)
        self.assertEqual(log["frames_changed"], 0)
        self.assertEqual(filtered.iloc[0].home_xy, data.iloc[0].home_xy)
