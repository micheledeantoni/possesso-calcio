import unittest
import numpy as np
from src.arrays import TrackingArrays
from src.cleaning import split_tracks, clean_players, clean_ball, clean_tracking
from src.team_features import continuity, build_team_features


def grid_positions(count, x0=10.):
    """Punti dentro il campo, a 6 m l'uno dall'altro."""
    return np.array([(x0 + 6. * (i % 8), 10. + 8. * (i // 8)) for i in range(count)])


def arrays(home=11, away=11, n=50, segment=None, pitch=(105., 68.), ball=(60., 60.)):
    players = np.full((n, home + away, 2), np.nan)
    players[:, :home] = grid_positions(home)
    players[:, home:] = grid_positions(away, x0=55.)
    team = np.tile(np.array([0] * home + [1] * away, dtype=np.int8), (n, 1))
    return TrackingArrays(np.arange(n), players, np.tile(ball, (n, 1)).astype(float), team,
                          np.ones(n, dtype=int) if segment is None else np.asarray(segment), pitch=pitch)


class TrackTests(unittest.TestCase):
    def test_tracks_break_on_jump_gap_and_segment(self):
        n = 14
        players = np.zeros((n, 1, 2))
        players[:, 0, 0] = np.arange(n) * .1      # 2,5 m/s
        players[5:, 0, 0] += 1.                    # salto di 1 m in un frame: 25 m/s
        players[10, 0] = np.nan                    # buco
        segment = np.array([1] * 8 + [2] * 6)
        tracks, jumps = split_tracks(players, np.zeros((n, 1), np.int8), continuity(np.arange(n), segment),
                                     speed_max_ms=12., fps=25)
        self.assertEqual(tracks, [(0, 0, 5), (0, 5, 3), (0, 8, 2), (0, 11, 3)])
        self.assertEqual(jumps, 1)


class PlayerCleaningTests(unittest.TestCase):
    def test_frames_with_eleven_are_untouched_and_input_kept(self):
        data = arrays()
        before = data.players.copy()
        cleaned, log = clean_players(data)
        np.testing.assert_array_equal(cleaned.players, before)
        np.testing.assert_array_equal(data.players, before)
        self.assertEqual(log["tracce_girate"] + log["doppioni_uniti"] + log["tolti_fuori_campo"], 0)

    def test_track_in_excess_changes_team_only_if_other_is_short(self):
        data = arrays(home=12, away=10)
        cleaned, log = clean_players(data)
        self.assertEqual(log["tracce_girate"], 1)
        self.assertEqual(int((cleaned.team[0] == 0).sum()), 11)
        self.assertEqual(int((data.team[0] == 0).sum()), 12)  # input invariato
        full = arrays(home=12, away=11)
        _, log = clean_players(full)
        self.assertEqual(log["tracce_girate"], 0)

    def test_duplicate_keeps_longer_track_without_averaging(self):
        data = arrays(home=12, away=11)
        data.players[:, 11] = np.nan
        data.players[5:, 11] = data.players[5:, 0] + (.5, 0.)  # doppione dello slot 0 dal frame 5
        cleaned, log = clean_players(data)
        self.assertEqual(log["doppioni_uniti"], 45)
        np.testing.assert_array_equal(cleaned.players[:, 0], data.players[:, 0])  # nessuna media
        self.assertTrue(np.isnan(cleaned.players[5:, 11]).all())
        speed = build_team_features(cleaned).home_avg_speed
        self.assertTrue((speed.dropna() == 0).all())

    def test_outside_object_removed_unless_nearest_to_ball(self):
        data = arrays(home=12, away=11)
        data.players[:, 11] = (-5., 30.)
        cleaned, log = clean_players(data)
        self.assertEqual(log["tolti_fuori_campo"], 50)
        self.assertTrue(np.isnan(cleaned.players[:, 11]).all())
        protected = arrays(home=12, away=11, ball=(-4., 30.))
        protected.players[:, 11] = (-5., 30.)
        cleaned, log = clean_players(protected)
        self.assertEqual(log["tolti_fuori_campo"], 0)
        self.assertEqual(log["frame_ancora_troppi"], 50)

    def test_pitch_size_is_a_parameter(self):
        for pitch, removed in (((105., 68.), 50), ((105.3, 68.), 0)):
            data = arrays(home=12, away=11, pitch=pitch)
            data.players[:, 11] = (106.2, 30.)  # 1,2 m fuori da 105; 0,9 m fuori da 105,3
            _, log = clean_players(data)
            self.assertEqual(log["tolti_fuori_campo"], removed, pitch)
        data = arrays()
        data.pitch = None
        with self.assertRaises(ValueError):
            clean_players(data)


class BallCleaningTests(unittest.TestCase):
    def line(self, n=60):
        ball = np.zeros((n, 2))
        ball[:, 0] = 10 + np.arange(n) * .4  # 10 m/s
        return ball

    def run_clean(self, ball, segment=None):
        n = len(ball)
        segment = np.ones(n, dtype=int) if segment is None else np.asarray(segment)
        return clean_ball(ball, continuity(np.arange(n), segment), segment > 0)

    def test_single_glitch_removed_real_kick_kept(self):
        ball = self.line()
        ball[10, 1] = 5.                    # punto che salta via e torna
        ball[30:, 0] += np.arange(30) * 1.  # calcio: arriva a 10 m/s, riparte a 35 m/s
        cleaned, interpolated, removed = self.run_clean(ball)
        self.assertEqual(np.flatnonzero(removed).tolist(), [10])
        self.assertTrue(interpolated[10])
        np.testing.assert_allclose(cleaned[10], ball[9] + (ball[11] - ball[9]) / 2)
        np.testing.assert_array_equal(cleaned[30:], ball[30:])

    def test_gap_rules(self):
        ball = self.line(120)
        ball[5:15] = np.nan    # 10 frame: riempito
        ball[30:60] = np.nan   # 30 frame (> 1 s): resta vuoto
        ball[70:75] = np.nan
        ball[75:, 1] += 50.    # riappare troppo lontano: resta vuoto
        cleaned, interpolated, _ = self.run_clean(ball)
        self.assertTrue(interpolated[5:15].all())
        self.assertFalse(interpolated[30:60].any())
        self.assertFalse(interpolated[70:75].any())

    def test_no_interpolation_across_segments_or_outside_evaluated_frames(self):
        ball = self.line(40)
        ball[18:22] = np.nan
        _, interpolated, _ = self.run_clean(ball, segment=[1] * 20 + [2] * 20)
        self.assertFalse(interpolated.any())
        _, interpolated, _ = self.run_clean(ball, segment=[0] * 40)
        self.assertFalse(interpolated.any())


class FullCleaningTests(unittest.TestCase):
    def test_clean_tracking_returns_arrays_mask_and_log(self):
        data = arrays(home=12, away=10)
        data.ball[20:23] = np.nan
        cleaned, interpolated, log = clean_tracking(data)
        self.assertEqual(int(interpolated.sum()), 3)
        self.assertTrue(np.isfinite(cleaned.ball).all())
        self.assertTrue(np.isnan(data.ball[20:23]).all())
        self.assertEqual(log["tracce_girate"], 1)
        self.assertEqual(log["palla_frame_interpolati"], 3)


class PlayerSpeedCapTests(unittest.TestCase):
    def test_slot_jump_does_not_create_a_speed(self):
        data = arrays(n=4)
        data.players[2:, 0] += (5., 0.)  # salto di 5 m: 125 m/s
        speed = build_team_features(data).home_avg_speed
        self.assertEqual(speed[2], 0.)  # il passo dello slot 0 è escluso, gli altri sono fermi
        self.assertEqual(build_team_features(data, player_speed_max_ms=200.).home_avg_speed[2], 125. / 11)


if __name__ == "__main__":
    unittest.main()
