import math
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.arrays import TrackingArrays, load_atd_arrays, load_metrica_npz, segment_from_halves
from src.data import load_tracking
from src.team_features import build_team_features, feature_columns


def tracking(n=6, ball=None, segment=None, frames=None):
    """Due tracce di casa ferme, una ospite che si muove di 0,2 m per frame (5 m/s) lungo x."""
    players = np.zeros((n, 3, 2))
    players[:, 0] = (0., 0.)
    players[:, 1] = (10., 0.)
    players[:, 2, 0] = 20. + .2 * np.arange(n)
    team = np.tile(np.array([0, 0, 1], dtype=np.int8), (n, 1))
    if ball is None:
        ball = np.tile([1., 0.], (n, 1))
    return TrackingArrays(frames=np.arange(n) if frames is None else np.asarray(frames),
                          players=players, ball=np.asarray(ball, dtype=float), team=team,
                          segment=np.ones(n, dtype=int) if segment is None else np.asarray(segment))


class TeamFeatureTests(unittest.TestCase):
    def test_one_row_per_frame_and_basic_geometry(self):
        data = tracking()
        features = build_team_features(data)
        self.assertEqual(features.frame.tolist(), list(range(6)))
        row = features.iloc[0]
        self.assertEqual(row.home_min_dist_ball, 1.)
        self.assertEqual(row.home_second_dist_ball, 9.)
        self.assertTrue(math.isnan(row.home_third_dist_ball))
        self.assertEqual(row.away_min_dist_ball, 19.)
        self.assertEqual(row.diff_min_dist_ball, 1. - 19.)
        self.assertEqual(row.home_players_2m, 1.)
        self.assertEqual(row.home_players_2m_pct, .5)
        self.assertEqual(row.home_x_extent, 10.)
        self.assertNotIn("frame", feature_columns(features))
        self.assertFalse(any(c.startswith("slot_") for c in features.columns))

    def test_inputs_are_not_modified(self):
        data = tracking()
        before = data.players.copy(), data.ball.copy(), data.team.copy()
        build_team_features(data)
        for original, after in zip(before, (data.players, data.ball, data.team)):
            np.testing.assert_array_equal(original, after)

    def test_team_swap_mirrors_columns(self):
        data = tracking()
        swapped = TrackingArrays(data.frames, data.players, data.ball, 1 - data.team, data.segment)
        a, b = build_team_features(data), build_team_features(swapped)
        np.testing.assert_allclose(a.home_min_dist_ball, b.away_min_dist_ball)
        np.testing.assert_allclose(a.home_avg_speed, b.away_avg_speed, equal_nan=True)
        np.testing.assert_allclose(a.diff_min_dist_ball, -b.diff_min_dist_ball)

    def test_missing_ball_gives_nan_not_zero(self):
        ball = np.tile([1., 0.], (6, 1))
        ball[2:4] = np.nan
        features = build_team_features(tracking(ball=ball))
        self.assertEqual(features.ball_available.tolist(), [1, 1, 0, 0, 1, 1])
        self.assertTrue(features.home_min_dist_ball[2:4].isna().all())
        self.assertTrue(features.home_players_5m[2:4].isna().all())
        self.assertEqual(features.ball_missing_run_length.tolist(), [0, 0, 1, 2, 0, 0])
        self.assertEqual(features.ball_gap_started.tolist(), [0, 0, 1, 0, 0, 0])
        self.assertTrue(features.ball_displacement_0_5s.isna().all())
        # Le tracce restano osservate anche senza palla.
        self.assertEqual(features.home_players_available[2], 2)

    def test_speeds_do_not_cross_segment_or_frame_gaps(self):
        features = build_team_features(tracking(segment=[1, 1, 1, 2, 2, 2]))
        self.assertTrue(math.isnan(features.away_avg_speed[0]))
        self.assertAlmostEqual(features.away_avg_speed[1], 5.)
        self.assertTrue(math.isnan(features.away_avg_speed[3]))
        self.assertAlmostEqual(features.away_avg_speed[4], 5.)
        gap = build_team_features(tracking(frames=[0, 1, 2, 10, 11, 12]))
        self.assertTrue(math.isnan(gap.away_avg_speed[3]))
        ball = np.tile([1., 0.], (6, 1))
        ball[2:4] = np.nan
        split = build_team_features(tracking(ball=ball, segment=[1, 1, 1, 2, 2, 2]))
        self.assertEqual(split.ball_missing_run_length.tolist(), [0, 0, 1, 1, 0, 0])
        self.assertEqual(split.ball_gap_started.tolist(), [0, 0, 1, 1, 0, 0])

    def test_ball_speed_threshold_is_explicit(self):
        ball = np.array([[0., 0.], [0.4, 0.], [3.4, 0.], [3.8, 0.]])  # 10, 75, 10 m/s
        features = build_team_features(tracking(n=4, ball=ball), ball_speed_max_ms=40.)
        self.assertAlmostEqual(features.ball_speed[1], 10.)
        self.assertTrue(math.isnan(features.ball_speed[2]))
        self.assertEqual(features.ball_tracking_anomaly.tolist(), [0, 0, 1, 0])
        self.assertTrue(math.isnan(features.ball_acceleration[2]))
        loose = build_team_features(tracking(n=4, ball=ball), ball_speed_max_ms=100.)
        self.assertAlmostEqual(loose.ball_speed[2], 75.)

    def test_team_is_read_per_frame(self):
        data = tracking(n=3)
        data.team[1, 1] = 1  # la seconda traccia passa agli ospiti solo nel frame 1
        features = build_team_features(data)
        self.assertEqual(features.home_players_available.tolist(), [2, 1, 2])
        self.assertEqual(features.away_players_available.tolist(), [1, 2, 1])
        self.assertEqual(features.away_players_newly_available.tolist(), [0, 1, 0])
        self.assertEqual(features.home_players_newly_missing.tolist(), [0, 1, 0])

    def test_nearest_player_speed_uses_the_nearest_track(self):
        data = tracking(n=3, ball=np.tile([19., 0.], (3, 1)))
        features = build_team_features(data)
        self.assertAlmostEqual(features.away_nearest_player_speed[1], 5.)
        self.assertAlmostEqual(features.home_nearest_player_speed[1], 0.)


class ArrayLoaderTests(unittest.TestCase):
    def test_atd_arrays_match_list_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "BAR-ATH_tactical_atd.xml").write_text(
                '<root><Team id="h"/><Team id="a"/><Player teamId="a"/><Player teamId="h"/></root>')
            (path / "BAR-ATH_tactical_atd.txt").write_text(
                '1:0.1,0.5,0;0.2,0.5,0:0.1,0,0\n2:NaN,NaN,0;0.3,0.5,0:NaN,NaN,NaN\n3:0.1,0.5,0;0.2,0.5,0:0.5,0.5,0\n')
            arrays = load_atd_arrays(path, pitch=(100, 60), start_frame=1, halves=((2, 5),))
            listed = load_tracking(path, pitch=(100, 60), start_frame=1)
        self.assertEqual(arrays.frames.tolist(), listed.frame.tolist())
        self.assertEqual(arrays.team[0].tolist(), [1, 0])  # casa = prima squadra dei metadati
        self.assertEqual(arrays.segment.tolist(), [0, 1])
        self.assertTrue(np.isnan(arrays.ball[0]).all())
        self.assertTrue(np.isnan(arrays.players[0, 0]).all())
        self.assertEqual(tuple(arrays.players[0, 1]), listed.home_xy[0][0])
        self.assertEqual(tuple(arrays.ball[1]), listed.ball_xy[1])

    def test_metrica_npz_clean_and_dirty(self):
        with tempfile.TemporaryDirectory() as directory:
            clean, dirty = Path(directory) / "pulita.npz", Path(directory) / "sporca.npz"
            gioc = np.zeros((4, 2, 2), dtype=np.float32)
            palla = np.zeros((4, 2), dtype=np.float32)
            labels = np.array(list("HHDA"))
            np.savez(clean, gioc=gioc, palla=palla, squadra_slot=np.array([0, 1], dtype=np.int8), etichette=labels)
            np.savez(dirty, gioc=gioc, palla=palla, etichette=labels,
                     squadra_fr=np.array([[0, 1], [0, 1], [1, 1], [0, 1]], dtype=np.int8),
                     periodo=np.array([1, 1, 2, 2], dtype=np.int8))
            a, b = load_metrica_npz(clean), load_metrica_npz(dirty)
        self.assertEqual(a.team.shape, (4, 2))
        self.assertEqual(a.segment.tolist(), [1, 1, 1, 1])
        self.assertEqual(b.team[2].tolist(), [1, 1])
        self.assertEqual(b.segment.tolist(), [1, 1, 2, 2])
        self.assertEqual(b.labels.tolist(), list("HHDA"))

    def test_invalid_shapes_and_teams_are_rejected(self):
        data = tracking(n=3)
        with self.assertRaises(ValueError):
            TrackingArrays(data.frames, data.players, data.ball[:2], data.team, data.segment)
        with self.assertRaises(ValueError):
            TrackingArrays(data.frames, data.players, data.ball, data.team + 1, data.segment)
        with self.assertRaises(ValueError):
            TrackingArrays(np.array([0, 2, 1]), data.players, data.ball, data.team, data.segment)

    def test_segment_from_halves(self):
        self.assertEqual(segment_from_halves([0, 5, 6, 9, 10], ((5, 6), (9, 9))).tolist(), [0, 1, 1, 2, 0])


class RunStageTests(unittest.TestCase):
    def config(self, **changes):
        config = {"source": "synthetic", "start_frame": 0, "end_frame": None, "synthetic_duration_s": 40,
                  "run_name": "squadre", "save_outputs": True}
        config.update(changes)
        return config

    def test_demo_arrays_follow_the_run_slice(self):
        from src.runs import load_run, load_run_arrays
        config = self.config(start_frame=800, end_frame=1000)  # 32–40 s
        listed, _ = load_run(config, Path.cwd())
        arrays = load_run_arrays(config, Path.cwd())
        self.assertEqual(arrays.frames.tolist(), listed.frame.tolist())
        # Nella demo due tracce di casa spariscono fra 35 e 43 s: restano NaN, non zero.
        self.assertEqual(int(np.isfinite(arrays.players[-1, :, 0]).sum()), 20)
        self.assertEqual(int((arrays.team[0] == 0).sum()), 11)

    def test_stage_uses_the_run_ball_and_saves(self):
        from src.runs import load_run, load_run_arrays
        from src.ball import prepare_ball
        from src.notebook_io import team_features_for_run, export_run
        from src.checks import check_tracking
        config = self.config()
        listed, _ = load_run(config, Path.cwd())
        tracking, _ = prepare_ball(listed, "linear", 0.5)
        features, log = team_features_for_run(load_run_arrays(config, Path.cwd()), tracking, 40.)
        estimated = tracking.ball_source.eq("linear").to_numpy()
        self.assertTrue(estimated.any())
        self.assertTrue((features.ball_available.to_numpy()[estimated] == 1).all())
        self.assertEqual(features.ball_source.tolist(), tracking.ball_source.tolist())
        self.assertEqual(log["frame_con_palla_stimata"], int(estimated.sum()))
        with self.assertRaises(ValueError):
            team_features_for_run(load_run_arrays(config, Path.cwd()), tracking.iloc[1:], 40.)
        predictions = pd.DataFrame({"frame": tracking.frame, "label": "X"})
        with tempfile.TemporaryDirectory() as directory:
            export_run(config, directory, tracking, features, predictions, check_tracking(tracking), {}, features)
            self.assertTrue((Path(directory) / "outputs/squadre/team_features.csv").exists())
            export_run(config, directory, tracking, features, predictions, check_tracking(tracking), {})
            self.assertFalse((Path(directory) / "outputs/squadre/team_features.csv").exists())

    def test_stage_with_cleaning_uses_cleaned_ball(self):
        from src.runs import load_run, load_run_arrays
        from src.ball import prepare_ball
        from src.cleaning import clean_tracking
        from src.notebook_io import team_features_for_run
        config = self.config()
        listed, _ = load_run(config, Path.cwd())
        tracking, _ = prepare_ball(listed)
        arrays, interpolated, log = clean_tracking(load_run_arrays(config, Path.cwd()))
        features, stage_log = team_features_for_run(arrays, tracking, 40., interpolated)
        self.assertTrue(interpolated.any())  # la demo ha buchi brevi della palla
        self.assertEqual(features.ball_source.eq("linear").sum(), interpolated.sum())
        self.assertEqual(features.ball_available.sum(), int(np.isfinite(arrays.ball).all(axis=1).sum()))
        self.assertIn("puliti", stage_log["giocatori_e_palla"])

    def test_real_arrays_use_published_halves(self):
        from src.runs import load_run_arrays
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "data"
            folder.mkdir()
            (folder / "BAR-ATH_tactical_atd.xml").write_text(
                '<root><Team id="h"/><Team id="a"/><Player teamId="h"/><Player teamId="a"/></root>')
            (folder / "BAR-ATH_tactical_atd.txt").write_text(
                ''.join(f'{i}:0,0,0;1,0,0:0.1,0,0\n' for i in range(11184, 11189)))
            arrays = load_run_arrays({"source": "real", "start_frame": 11183, "end_frame": 11187}, directory)
        self.assertEqual(arrays.frames.tolist(), [11183, 11184, 11185, 11186])
        self.assertEqual(arrays.segment.tolist(), [0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
