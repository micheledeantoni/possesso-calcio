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


class KeeperFeatureTests(unittest.TestCase):
    def match(self, flip=False, ball=True):
        """Casa a sinistra (portiere a 3 m dalla sua porta), ospiti a destra con il
        portiere alto, a 25 m dalla sua porta. `flip`: secondo tempo, campo ruotato."""
        n = 10
        home = [(3., 34.), (30., 20.), (35., 50.)]
        away = [(80., 34.), (60., 20.), (65., 50.)]
        players = np.tile(np.array(home + away, dtype=float), (n, 1, 1))
        b = np.tile([50., 34.], (n, 1)) if ball else np.full((n, 2), np.nan)
        if flip:
            players[..., 0], players[..., 1] = 105. - players[..., 0], 68. - players[..., 1]
            b = np.column_stack([105. - b[:, 0], 68. - b[:, 1]])
        team = np.tile(np.array([0, 0, 0, 1, 1, 1], dtype=np.int8), (n, 1))
        return TrackingArrays(np.arange(n), players, b, team, np.ones(n, dtype=int), pitch=(105., 68.))

    def test_keeper_depth_from_own_goal(self):
        f = build_team_features(self.match())
        self.assertAlmostEqual(f.home_keeper_depth[0], 3.)
        self.assertAlmostEqual(f.away_keeper_depth[0], 25.)
        self.assertAlmostEqual(f.diff_keeper_depth[0], -22.)

    def test_same_meaning_in_both_halves(self):
        a, b = build_team_features(self.match()), build_team_features(self.match(flip=True))
        for column in ("home_keeper_depth", "away_keeper_depth", "diff_keeper_depth"):
            np.testing.assert_allclose(a[column], b[column])

    def test_does_not_need_the_ball(self):
        f = build_team_features(self.match(ball=False))
        self.assertAlmostEqual(f.away_keeper_depth[0], 25.)


class PossessionExtrasTests(unittest.TestCase):
    """Casa a sinistra (porta a x = 0), ospiti a destra (porta a x = 105)."""
    def match(self, n=60, ball=(50., 34.), flip=False):
        home = [(3., 34.), (20., 10.), (20., 30.), (20., 50.), (40., 60.), (55., 34.)]   # linea a 20 m
        away = [(100., 34.), (70., 15.), (70., 30.), (70., 45.), (70., 60.), (45., 34.)]  # linea a 35 m
        players = np.tile(np.array(home + away, dtype=float), (n, 1, 1))
        b = np.tile(np.asarray(ball, dtype=float), (n, 1))
        if flip:
            players[..., 0], players[..., 1] = 105. - players[..., 0], 68. - players[..., 1]
            b = np.column_stack([105. - b[:, 0], 68. - b[:, 1]])
        team = np.tile(np.array([0] * 6 + [1] * 6, dtype=np.int8), (n, 1))
        return TrackingArrays(np.arange(n), players, b, team, np.ones(n, dtype=int), pitch=(105., 68.))

    def test_behind_level_ahead(self):
        f = build_team_features(self.match(), extra=["dietro"])
        # palla a x = 50: casa dietro = 3, 20, 20, 20, 40 → 5; davanti 55 → 1
        self.assertEqual(f.home_behind_ball[0], 5)
        self.assertEqual(f.home_ahead_ball[0], 1)
        # ospiti dietro (fra palla e x = 105): 100 e i quattro a 70 → 5; davanti: 45 → 1
        self.assertEqual(f.away_behind_ball[0], 5)
        self.assertEqual(f.away_ahead_ball[0], 1)
        f = build_team_features(self.match(ball=(55., 34.)), extra=["dietro"])
        self.assertEqual(f.home_level_ball[0], 1)  # il giocatore a 55 è in pari

    def test_missing_ball_is_nan(self):
        f = build_team_features(self.match(ball=(np.nan, np.nan)), extra=["dietro", "packing", "portieri"])
        self.assertTrue(f.home_behind_ball.isna().all())
        self.assertTrue(f.diff_behind_ball_trend_1s.isna().all())
        self.assertFalse(f.home_keeper_depth.isna().any())  # i portieri non usano la palla

    def test_packing_trend_when_ball_passes_the_opponents(self):
        data = self.match(n=100)
        data.ball[:] = np.column_stack([50. + .3 * np.arange(100), np.full(100, 34.)])  # la casa avanza: 50 → 80 m
        f = build_team_features(data, extra=["packing"])
        self.assertLess(f.away_behind_ball_trend_1s[50], 0)   # gli ospiti superati calano
        self.assertNotIn("home_behind_ball", f)               # senza "dietro" restano solo i trend

    def test_lines_and_lanes(self):
        f = build_team_features(self.match(), extra=["linee", "corsie"])
        # ospiti: quattro sulla linea a 70 (3 compagni ciascuno), due da soli → (4·3 + 0 + 0) / 6 = 2
        self.assertAlmostEqual(f.away_line_mates[0], 2.)
        self.assertAlmostEqual(f.home_line_mates[0], 1.)  # casa: tre a 20 m (2 compagni ciascuno) → 3·2 / 6 = 1
        self.assertGreater(f.diff_line_mates[0], -1.5)
        self.assertLess(f.diff_line_mates[0], 0)            # gli ospiti sono più "in linea"
        self.assertEqual(f.away_lanes[0], 4)

    def test_same_meaning_in_both_halves(self):
        groups = ["portieri", "dietro", "packing", "linee", "corsie"]
        a = build_team_features(self.match(), extra=groups)
        b = build_team_features(self.match(flip=True), extra=groups)
        for column in [c for c in a.columns if c.endswith(("keeper_depth", "behind_ball", "level_ball", "ahead_ball",
                                                            "trend_1s", "line_mates", "lanes"))]:
            np.testing.assert_allclose(a[column], b[column], err_msg=column)

    def pass_to_teammate(self, flip=False):
        """La palla parte da x = 40 e va dritta (10 m/s) verso un compagno di casa a x = 60;
        un ospite sta di lato, a 11 m dalla traiettoria."""
        n = 50
        players = np.tile(np.array([(3., 34.), (60., 34.), (100., 34.), (55., 45.)]), (n, 1, 1))
        ball = np.column_stack([40. + .4 * np.arange(n), np.full(n, 34.)])
        if flip:
            players[..., 0], players[..., 1] = 105. - players[..., 0], 68. - players[..., 1]
            ball = np.column_stack([105. - ball[:, 0], 68. - ball[:, 1]])
        team = np.tile(np.array([0, 0, 1, 1], dtype=np.int8), (n, 1))
        return TrackingArrays(np.arange(n), players, ball, team, np.ones(n, dtype=int), pitch=(105., 68.))

    def test_ball_goes_straight_to_a_teammate(self):
        f = build_team_features(self.pass_to_teammate(), extra=["direzione"])
        self.assertTrue(math.isnan(f.home_receiver_dist[0]))     # primi frame: direzione non ancora nota
        self.assertAlmostEqual(f.home_receiver_dist[10], 0.)     # il compagno è sulla traiettoria
        self.assertAlmostEqual(f.away_receiver_dist[10], 11.)    # l'ospite è di lato
        self.assertLess(f.diff_receiver_dist[10], 0)             # la palla va verso la casa
        self.assertEqual(f.home_in_cone[10], 1)
        self.assertEqual(f.away_in_cone[10], 0)                  # 11 m di lato a 7 m: fuori dal cono

    def test_direction_same_in_both_halves(self):
        a = build_team_features(self.pass_to_teammate(), extra=["direzione"])
        b = build_team_features(self.pass_to_teammate(flip=True), extra=["direzione"])
        for column in ("home_receiver_dist", "away_receiver_dist", "diff_receiver_dist", "home_in_cone"):
            np.testing.assert_allclose(a[column], b[column], err_msg=column)

    def test_direction_is_nan_when_ball_still_or_missing(self):
        data = self.pass_to_teammate()
        data.ball[:] = (40., 34.)  # ferma
        self.assertTrue(build_team_features(data, extra=["direzione"]).home_receiver_dist.isna().all())
        data.ball[:] = np.nan
        self.assertTrue(build_team_features(data, extra=["direzione"]).home_in_cone.isna().all())

    def test_groups_can_be_switched_off(self):
        f = build_team_features(self.match(), extra=[])
        self.assertNotIn("home_keeper_depth", f)
        with self.assertRaises(ValueError):
            build_team_features(self.match(), extra=["sconosciuta"])


class OrientedAndContextTests(unittest.TestCase):
    def test_oriented_features_do_not_depend_on_field_side(self):
        data = tracking(n=8, ball=np.column_stack([1. + np.arange(8) * .3, np.zeros(8)]))
        data = TrackingArrays(data.frames, data.players + (30., 20.), data.ball + (30., 20.), data.team,
                              data.segment, pitch=(105., 68.))
        rotated = TrackingArrays(data.frames, np.stack([105. - data.players[..., 0], 68. - data.players[..., 1]], axis=2),
                                 np.column_stack([105. - data.ball[:, 0], 68. - data.ball[:, 1]]), data.team,
                                 data.segment, pitch=(105., 68.))
        a = build_team_features(data, oriented=True)
        b = build_team_features(rotated, oriented=True)
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9)
        self.assertTrue((a.ball_x_oriented.diff().dropna() > 0).all())  # verso la porta degli ospiti
        plain = build_team_features(rotated, oriented=False)
        self.assertFalse(np.allclose(plain.ball_dx.dropna(), a.ball_dx.dropna()))

    def test_context_windows_and_streak(self):
        n = 60
        ball = np.tile([1., 0.], (n, 1))          # casa più vicina
        ball[30:] = (19.5, 0.)                     # poi ospiti
        ball[40:42] = np.nan                       # palla assente
        features = build_team_features(tracking(n=n, ball=ball), context=True)
        self.assertEqual(features.ctx_nearest_home_c12[10], 1.)
        self.assertAlmostEqual(features.ctx_nearest_home_c12[29], 13 / 23)  # frame 17-41, senza palla a 40-41
        self.assertAlmostEqual(features.ctx_nearest_home_past1s[35], 20 / 26)  # frame 10-35
        self.assertEqual(features.ctx_nearest_streak_s.tolist()[28:32], [29 / 25, 30 / 25, 1 / 25, 2 / 25])
        self.assertTrue(np.isnan(features.ctx_nearest_streak_s[40]))
        self.assertEqual(features.ctx_nearest_streak_s[42], 1 / 25)
        split = build_team_features(tracking(n=n, ball=ball, segment=[1] * 20 + [2] * 40), context=True)
        self.assertEqual(split.ctx_nearest_streak_s[20], 1 / 25)
        self.assertAlmostEqual(split.ctx_nearest_home_c12[20], 10 / 13)  # frame 20-32: niente del tempo precedente
        plain = build_team_features(tracking(n=n, ball=ball), context=False)
        self.assertFalse(any(c.startswith("ctx_") for c in plain.columns))
