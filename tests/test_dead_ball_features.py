import math
import unittest
import numpy as np
from src.arrays import TrackingArrays
from src.dead_ball_features import build_dead_ball_features, window_mean


def formation(n=60, segment=None, pitch=(105., 68.)):
    """Casa nella metà sinistra, ospiti nella destra, tutti fermi."""
    home = np.array([(20. + 2.5 * i, 10. + 4 * i) for i in range(11)])
    away = np.array([(58. + 2.5 * i, 12. + 4 * i) for i in range(11)])  # nessuno nelle aree
    players = np.tile(np.vstack([home, away]), (n, 1, 1)).astype(float)
    team = np.tile(np.array([0] * 11 + [1] * 11, dtype=np.int8), (n, 1))
    ball = np.full((n, 2), np.nan)
    return TrackingArrays(np.arange(n), players, ball, team,
                          np.ones(n, dtype=int) if segment is None else np.asarray(segment), pitch=pitch)


class WindowTests(unittest.TestCase):
    def test_window_stays_inside_continuous_block(self):
        values = np.array([1., 1., 1., 9., 9., 9.])
        continuous = np.array([False, True, True, False, True, True])
        np.testing.assert_allclose(window_mean(values, continuous, 2, 0), [1, 1, 1, 9, 9, 9])
        np.testing.assert_allclose(window_mean(values, continuous, 0, 5), [1, 1, 1, 9, 9, 9])
        with_nan = np.array([np.nan, 2., 4., np.nan, np.nan, np.nan])
        result = window_mean(with_nan, np.r_[False, np.ones(5, bool)], 1, 1)
        np.testing.assert_allclose(result, [2, 3, 3, 4, np.nan, np.nan])


class DeadBallFeatureTests(unittest.TestCase):
    def test_does_not_read_the_ball(self):
        data = formation()
        a = build_dead_ball_features(data)
        data.ball[:] = (50., 30.)
        b = build_dead_ball_features(data)
        np.testing.assert_array_equal(a.to_numpy(float), b.to_numpy(float))

    def test_standing_players_are_slow(self):
        f = build_dead_ball_features(formation())
        self.assertTrue(math.isnan(f.all_mean_speed[0]))  # nessun passo precedente
        self.assertEqual(f.all_mean_speed[1], 0.)
        self.assertEqual(f.share_slow[1], 1.)
        self.assertEqual(f.share_fast[1], 0.)
        self.assertEqual(f.players_present[0], 22)

    def test_speed_trend_detects_a_stop_without_crossing_segments(self):
        data = formation(n=200)
        data.players[:100, :, 0] += np.arange(100)[:, None] * .2  # 5 m/s poi fermi
        f = build_dead_ball_features(data)
        self.assertAlmostEqual(f.all_mean_speed[50], 5.)
        self.assertLess(f["all_mean_speed_trend_2s"][100], -4.)
        split = formation(n=200, segment=[1] * 100 + [2] * 100)
        split.players[:100, :, 0] += np.arange(100)[:, None] * .2
        g = build_dead_ball_features(split)
        self.assertEqual(g["all_mean_speed_past_2s"][110], 0.)  # non vede il tratto precedente

    def test_kickoff_formation(self):
        data = formation()
        data.players[:, 0] = (52.5, 34.)  # un giocatore di casa al centro
        f = build_dead_ball_features(data)
        self.assertEqual(f.teams_in_opposite_halves[0], 1.)
        self.assertEqual(f.min_dist_center_spot[0], 0.)
        mixed = formation()
        mixed.players[:, 1] = (80., 30.)  # un giocatore di casa nella metà avversaria
        self.assertLess(build_dead_ball_features(mixed).own_half_share_min[0], 1.)

    def test_throw_in_and_corner(self):
        data = formation()
        data.players[:, 0] = (30., -.5)
        f = build_dead_ball_features(data)
        self.assertEqual(f.players_beyond_sideline[0], 1)
        self.assertAlmostEqual(f.min_dist_sideline[0], -.5)
        data.players[:, 0] = (104., 67.)
        self.assertAlmostEqual(build_dead_ball_features(data).min_dist_corner[0], math.hypot(1, 1))

    def test_areas_and_clusters(self):
        data = formation()
        data.players[:, 12:17] = [(95., 34. + d) for d in (-2, -1, 0, 1, 2)]  # 5 ospiti in area destra
        data.players[:, 17] = (104., 30.)  # uno in area piccola, a più di 3 m dagli altri
        f = build_dead_ball_features(data)
        self.assertEqual(f.penalty_area_max[0], 6)
        self.assertEqual(f.penalty_area_min[0], 0)
        self.assertEqual(f.goal_area_max[0], 1)
        self.assertEqual(f.away_max_cluster_3m[0], 5)  # 34±2 entro 3 m dal centrale: 5 punti
        self.assertEqual(f.max_cluster_3m[0], 5)

    def test_team_swap_mirrors_team_columns(self):
        data = formation()
        swapped = TrackingArrays(data.frames, data.players, data.ball, 1 - data.team, data.segment, pitch=data.pitch)
        a, b = build_dead_ball_features(data), build_dead_ball_features(swapped)
        np.testing.assert_array_equal(a.home_max_cluster_3m, b.away_max_cluster_3m)
        np.testing.assert_array_equal(a.own_half_share_min, b.own_half_share_min)

    def test_requires_pitch(self):
        data = formation()
        data.pitch = None
        with self.assertRaises(ValueError):
            build_dead_ball_features(data)


if __name__ == "__main__":
    unittest.main()
