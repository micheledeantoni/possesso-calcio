import math
import tempfile
from pathlib import Path
import unittest
import numpy as np
from src.arrays import TrackingArrays
from src.team_features import build_team_features, continuity
from src.model import (Game, viterbi, evaluate, switch_rate, decode, fit_possession_model,
                       predict_possession, cross_validate, load_metrica_games)


def synthetic_game(seed, n=1500, block=150, name="g"):
    """Possesso alternato a blocchi; la palla sta vicino a un giocatore della squadra
    in possesso, con rumore; ogni tanto un frame ingannevole e un tratto D senza palla."""
    rng = np.random.default_rng(seed)
    labels = np.where((np.arange(n) // block) % 2 == 0, "H", "A")
    labels[block * 3: block * 3 + 40] = "D"
    # Posizioni nuove per ogni frame: nessuna deriva temporale da cui imparare i blocchi.
    players = rng.uniform([10, 10], [95, 58], size=(n, 22, 2))
    team = np.tile(np.array([0] * 11 + [1] * 11, dtype=np.int8), (n, 1))
    ball = np.full((n, 2), np.nan)
    for i, label in enumerate(labels):
        if label == "D":
            continue
        owner = rng.integers(0, 11) + (0 if label == "H" else 11)
        if rng.random() < .15:  # frame ingannevole: palla vicino all'altra squadra
            owner = (owner + 11) % 22
        ball[i] = players[i, owner] + rng.normal(0, .5, 2)
    arrays = TrackingArrays(np.arange(n), players, ball, team, np.ones(n, dtype=int), labels)
    return Game(name, build_team_features(arrays), labels, continuity(arrays.frames, arrays.segment))


SMALL_GRID = {"temperature": (0.1, 1.0), "switch_scale": (1.0,), "neutral_without_ball": (True,)}


class ViterbiTests(unittest.TestCase):
    def test_removes_isolated_flicker(self):
        evidence = np.tile([1., -1.], (20, 1))
        evidence[7] = [-1., 1.]  # un solo frame contrario
        evidence[12:] = [-1., 1.]
        path = viterbi(evidence * .5, 0.01, np.r_[False, np.ones(19, bool)])
        self.assertEqual(path.tolist(), [0] * 12 + [1] * 8)

    def test_free_transition_equals_argmax(self):
        evidence = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
        self.assertEqual(viterbi(evidence, 0.01, np.zeros(4, bool)).tolist(), [0, 1, 0, 1])

    def test_neutral_frames_keep_previous_possession(self):
        evidence = np.zeros((10, 2))
        evidence[:3] = [0., 2.]
        self.assertEqual(viterbi(evidence, 0.01, np.r_[False, np.ones(9, bool)]).tolist(), [1] * 10)

    def test_rejects_invalid_probability_and_handles_empty(self):
        with self.assertRaises(ValueError):
            viterbi(np.zeros((2, 2)), 0., np.ones(2, bool))
        self.assertEqual(len(viterbi(np.zeros((0, 2)), .1, np.zeros(0, bool))), 0)


class MetricTests(unittest.TestCase):
    def test_changes_are_matched_by_team_and_tolerance(self):
        true = np.array(list("HHHHAAAAD" "DHHHH"))
        pred = np.array(list("HHHHHAAAA" "AAHHH"))
        result = evaluate(true, pred, tolerance_frames=2, fps=1)
        self.assertEqual(result["cambi_reali"], 2)  # H→A al frame 4, A→H al frame 10 (D saltati)
        self.assertEqual(result["cambi_predetti"], 2)  # frame 5 e 11: un frame di ritardo
        self.assertEqual(result["cambi_recall"], 1.)
        self.assertEqual(result["ritardo_mediano_s"], 1.)
        self.assertAlmostEqual(result["accuratezza_HA"], 10 / 12)
        strict = evaluate(true, pred, tolerance_frames=0, fps=1)
        self.assertEqual(strict["cambi_recall"], 0.)
        self.assertTrue(math.isnan(strict["ritardo_mediano_s"]))

    def test_switch_rate_ignores_dead_ball(self):
        game = Game("x", None, np.array(list("HHDDHHAA")), None)
        self.assertEqual(switch_rate([game]), 1 / 8)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.games = [synthetic_game(seed, name=f"g{seed}") for seed in (1, 2, 3)]

    def test_viterbi_beats_per_frame_and_keeps_order(self):
        model = fit_possession_model(self.games[:2], SMALL_GRID)
        test = self.games[2]
        result = predict_possession(model, test.features, test.continuous)
        self.assertEqual(result.frame.tolist(), test.features.frame.tolist())
        self.assertTrue(result.label.isin(["H", "A"]).all())
        metrics = evaluate(test.labels, result.label.to_numpy())
        self.assertGreater(metrics["accuratezza_HA"], .95)
        self.assertLessEqual(metrics["cambi_predetti"], 2 * metrics["cambi_reali"])
        self.assertEqual(model.training_games, ["g1", "g2"])

    def test_missing_feature_is_rejected(self):
        model = fit_possession_model(self.games[:2], SMALL_GRID)
        with self.assertRaises(ValueError):
            predict_possession(model, self.games[2].features.drop(columns="home_min_dist_ball"),
                               self.games[2].continuous)

    def test_nested_cross_validation_reports_both_methods(self):
        table = cross_validate(self.games, SMALL_GRID)
        self.assertEqual(sorted(table.metodo.unique()), ["per_frame", "viterbi"])
        self.assertEqual(len(table), 6)
        viterbi_changes = table[table.metodo == "viterbi"].cambi_predetti.sum()
        frame_changes = table[table.metodo == "per_frame"].cambi_predetti.sum()
        self.assertLess(viterbi_changes, frame_changes)

    def test_neutral_without_ball_only_changes_missing_frames(self):
        game = self.games[0]
        evidence = np.tile([3., -3.], (len(game.labels), 1))
        neutral = decode(evidence, game.features, np.zeros(len(game.labels), bool), .01,
                         {"temperature": 1., "switch_scale": 1., "neutral_without_ball": True})
        self.assertTrue((neutral[game.features.ball_available.to_numpy() == 1] == "H").all())

    def test_notebook_stage_trains_on_external_folder_only(self):
        from src.notebook_io import model_for_run
        from src.arrays import load_metrica_npz
        with tempfile.TemporaryDirectory() as directory:
            for seed in (4, 5):
                rng = np.random.default_rng(seed)
                n = 900
                labels = np.where((np.arange(n) // 150) % 2 == 0, "H", "A")
                players = rng.uniform([10, 10], [95, 58], size=(n, 22, 2))
                owner = rng.integers(0, 11, n) + np.where(labels == "H", 0, 11)
                ball = players[np.arange(n), owner] + rng.normal(0, .5, (n, 2))
                np.savez(Path(directory) / f"m{seed}.npz", gioc=players.astype(np.float32),
                         palla=ball.astype(np.float32), etichette=labels,
                         squadra_slot=np.array([0] * 11 + [1] * 11, np.int8))
            run = load_metrica_npz(Path(directory) / "m4.npz")
            features = build_team_features(run)
            predictions, log = model_for_run(run, features, directory, 40.)
            with self.assertRaises(ValueError):
                model_for_run(run, None, directory, 40.)
        self.assertEqual(predictions.frame.tolist(), list(range(900)))
        self.assertEqual(log["partite_training"], ["m4", "m5"])
        self.assertIn("viterbi_temperature", log)

    def test_notebook_stage_three_states(self):
        from src.notebook_io import model_dah_for_run
        from src.arrays import load_metrica_npz
        with tempfile.TemporaryDirectory() as directory:
            for seed in (6, 7):
                rng = np.random.default_rng(seed)
                n = 900
                labels = np.where((np.arange(n) // 150) % 2 == 0, "H", "A")
                labels[300:400] = "D"
                base = rng.uniform([10, 10], [95, 58], size=(22, 2))
                moving = rng.normal(0, .15, size=(n, 22, 2)).cumsum(axis=0)
                moving[300:400] = moving[299]  # gioco fermo: giocatori immobili
                players = base + moving
                owner = rng.integers(0, 11, n) + np.where(labels == "H", 0, 11)
                ball = players[np.arange(n), owner] + rng.normal(0, .5, (n, 2))
                ball[300:400] = np.nan
                np.savez(Path(directory) / f"d{seed}.npz", gioc=players.astype(np.float32),
                         palla=ball.astype(np.float32), etichette=labels,
                         squadra_slot=np.array([0] * 11 + [1] * 11, np.int8))
            run = load_metrica_npz(Path(directory) / "d6.npz")
            predictions, probabilities, log = model_dah_for_run(run, build_team_features(run), directory, 40.)
            with self.assertRaises(ValueError):
                model_dah_for_run(run, None, directory, 40.)
        self.assertEqual(predictions.columns.tolist(), ["frame", "label"])
        self.assertEqual(predictions.frame.tolist(), list(range(900)))
        self.assertTrue(predictions.label.isin(["H", "A", "D"]).all())
        np.testing.assert_allclose(probabilities[["p_H", "p_A", "p_D"]].sum(axis=1), 1.)
        self.assertEqual(log["partite_training"], ["d6", "d7"])

    def test_loader_reads_npz_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            n = 4
            np.savez(Path(directory) / "a.npz", gioc=np.zeros((n, 2, 2), np.float32),
                     palla=np.zeros((n, 2), np.float32), squadra_slot=np.array([0, 1], np.int8),
                     etichette=np.array(list("HHAD")))
            games = load_metrica_games(directory)
            self.assertEqual([g.name for g in games], ["a"])
            self.assertEqual(len(games[0].features), n)
            with self.assertRaises(ValueError):
                load_metrica_games(Path(directory) / "vuota")


if __name__ == "__main__":
    unittest.main()


class AugmentationTests(unittest.TestCase):
    def test_extra_versions_never_reach_their_own_test_game(self):
        from unittest import mock
        import src.model_dah as dah
        games = [dah.DahGame(name, None, None, np.array([]), None) for name in ("p1", "p2", "p3")]
        extra = [dah.DahGame(name, None, None, np.array([]), None, primary=False) for name in ("p1", "p2", "p3")]
        seen = []

        class Stop(Exception):
            pass

        def record(train, *args, **kwargs):
            seen.append(sorted((g.name, g.primary) for g in train))
            raise Stop

        with mock.patch.object(dah, "fit_dah_model", side_effect=record):
            for test in games:
                with self.assertRaises(Stop):
                    dah.cross_validate([test] + [g for g in games if g is not test], extra=extra)
        for names, test in zip(seen, ("p1", "p2", "p3")):
            self.assertNotIn(test, [n for n, _ in names])
            self.assertEqual(len(names), 4)  # due partite, ognuna nella versione principale e in quella in più


class ClassifierParamsTests(unittest.TestCase):
    def test_environment_overrides_only_the_given_keys(self):
        import json, os, subprocess, sys
        env = dict(os.environ, POSSESSO_CLASSIFIER_PARAMS='{"min_samples_leaf": 1000, "max_features": 0.5}')
        out = subprocess.run([sys.executable, "-c", "import json; from src.model import CLASSIFIER_PARAMS as p; print(json.dumps(p))"],
                             env=env, capture_output=True, text=True, check=True, cwd=str(Path(__file__).resolve().parents[1]))
        params = json.loads(out.stdout)
        self.assertEqual(params["min_samples_leaf"], 1000)
        self.assertEqual(params["max_features"], 0.5)
        self.assertEqual(params["max_iter"], 150)  # le altre chiavi restano quelle di default
