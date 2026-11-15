import math
import unittest
import warnings
from unittest import mock
import numpy as np
import pandas as pd
from src.arrays import TrackingArrays
from src.team_features import build_team_features, continuity
from src.dead_ball_features import build_dead_ball_features
from src.model import viterbi
from src.model_dah import (DahGame, transition_matrix, scaled_transitions, viterbi_k, posteriors,
                           evaluate_dah, fit_dah_model, predict_dah, cross_validate, emissions, select_params,
                           false_dead_rate, dead_ball_weight, DEAD_BALL_WEIGHT, STATES)


def synthetic_game(seed, ball_during_dead=False, name="g"):
    """Possesso a blocchi separati da interruzioni: a gioco fermo tutti immobili."""
    rng = np.random.default_rng(seed)
    pattern = ["H"] * 200 + ["D"] * 60 + ["A"] * 200 + ["D"] * 60
    labels = np.array((pattern * 4)[:1800])
    n = len(labels)
    base = rng.uniform([15, 10], [90, 58], size=(22, 2))
    players = base + rng.normal(0, .2, size=(n, 22, 2))  # ~7 m/s di agitazione a gioco vivo
    dead = labels == "D"
    for s in np.flatnonzero(dead & ~np.r_[False, dead[:-1]]):
        e = s
        while e < n and dead[e]:
            e += 1
        players[s:e] = players[s]  # fermi
    team = np.tile(np.array([0] * 11 + [1] * 11, dtype=np.int8), (n, 1))
    ball = np.full((n, 2), np.nan)
    for i, label in enumerate(labels):
        if label == "D" and not ball_during_dead:
            continue
        owner = rng.integers(0, 11) + (11 if label == "A" else 0)
        ball[i] = players[i, owner] + rng.normal(0, .5, 2)
    arrays = TrackingArrays(np.arange(n), players, ball, team, np.ones(n, dtype=int), labels, (105., 68.))
    return DahGame(name, build_team_features(arrays), build_dead_ball_features(arrays), labels,
                   continuity(arrays.frames, arrays.segment))


SMALL_GRID = {"temperature": (0.2,), "switch_scale": (1.0,), "neutral_without_ball": (False,), "w": (0.0,)}


class HmmTests(unittest.TestCase):
    def test_transition_matrix_counts_continuous_pairs(self):
        game = DahGame("x", None, None, np.array(list("HHHDDAA")), np.r_[False, np.ones(6, bool)])
        t = transition_matrix([game])
        np.testing.assert_allclose(t.sum(axis=1), 1)
        self.assertAlmostEqual(t[0, 0], 3 / 6)  # H→H: 2 coppie + 1 pseudo, su 3 + 3
        self.assertGreater(t[2, 1], t[2, 0])     # D→A visto, D→H no
        base = np.array([[.98, .01, .01], [.01, .98, .01], [.02, .02, .96]])
        scaled = scaled_transitions(base, 2.)
        np.testing.assert_allclose(scaled.sum(axis=1), 1)
        self.assertAlmostEqual(scaled[0, 1], .02)
        self.assertAlmostEqual(scaled[2, 2], .92)
        with self.assertRaises(ValueError):
            scaled_transitions(base, 60.)

    def test_viterbi_k_matches_two_state_version(self):
        rng = np.random.default_rng(0)
        e = rng.normal(0, 1, size=(300, 2))
        cont = np.r_[False, rng.random(299) > .05]
        s = .02
        log_t = np.log([[1 - s, s], [s, 1 - s]])
        np.testing.assert_array_equal(viterbi_k(e, log_t, cont), viterbi(e, s, cont))

    def test_posteriors_are_probabilities_and_follow_evidence(self):
        e = np.zeros((50, 3))
        e[:25, 2] = 3.
        e[25:, 0] = 3.
        post = posteriors(e, np.log(np.full((3, 3), .98) * np.eye(3) + .01 * (1 - np.eye(3))), np.r_[False, np.ones(49, bool)])
        np.testing.assert_allclose(post.sum(axis=1), 1)
        self.assertGreater(post[5, 2], .9)
        self.assertLess(post[45, 2], .1)
        self.assertGreater(post[45, 0], .9)

    def test_evaluate_counts_dead_ball_and_fills_changes(self):
        true = np.array(list("HHHDDAAA"))
        pred = np.array(list("HHDDDAAA"))
        r = evaluate_dah(true, pred, tolerance_frames=5, fps=1)
        self.assertAlmostEqual(r["accuratezza_tutti"], 7 / 8)
        self.assertAlmostEqual(r["D_precisione"], 2 / 3)
        self.assertEqual(r["D_recall"], 1.)
        self.assertAlmostEqual(r["accuratezza_HA"], 5 / 6)  # D su un frame H è un errore
        self.assertEqual(r["cambi_predetti"], 1)


class DahModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.games = [synthetic_game(seed, name=f"g{seed}") for seed in (1, 2, 3)]

    def test_dead_ball_found_even_when_ball_is_visible(self):
        model = fit_dah_model(self.games[:2], SMALL_GRID)
        test = synthetic_game(9, ball_during_dead=True)
        result = predict_dah(model, test.features, test.dead_features, test.continuous, probabilities=True)
        self.assertEqual(result.frame.tolist(), list(range(len(test.labels))))
        r = evaluate_dah(test.labels, result.label.to_numpy())
        self.assertGreater(r["D_recall"], .9)
        self.assertGreater(r["accuratezza_tutti"], .9)
        np.testing.assert_allclose(result[["p_H", "p_A", "p_D"]].sum(axis=1), 1)
        self.assertFalse(model.dead_ball_uses_ball)
        self.assertTrue(all(not c.startswith(("home_", "away_", "diff_", "ball_")) or "speed" in c or "cluster" in c
                            or "centroid_x" in c for c in model.dead_columns))

    def test_ball_classifier_is_used_only_when_weighted(self):
        model = fit_dah_model(self.games[:2], {**SMALL_GRID, "w": (0.5,)})
        self.assertTrue(model.dead_ball_uses_ball)
        self.assertEqual(model.dead_ball_weight, 0.5)
        self.assertIn("ball_available", model.dead_with[1])      # il classificatore con palla la vede
        self.assertNotIn("ball_available", model.dead_columns)   # quello senza palla no
        self.assertAlmostEqual(model.dead_with[2], model.dead_without[2])  # stesso prior di D

    def test_nested_cross_validation(self):
        table = cross_validate(self.games, SMALL_GRID)
        self.assertEqual(len(table), 3)
        self.assertTrue((table.accuratezza_tutti > .85).all())


class DeadBallMixTests(unittest.TestCase):
    PARTS = {"p_dead_without": np.array([.1, .8]), "p_dead_with": np.array([.9, .2]), "dead_prior": .4,
             "H": np.zeros(2), "A": np.zeros(2), "ball": np.array([True, False])}

    def emitted_p_dead(self, w):
        params = {"temperature": 1.0, "neutral_without_ball": False}
        if w is not None:
            params["w"] = w
        d = emissions(self.PARTS, params)[:, 2]
        return np.exp(d + math.log(self.PARTS["dead_prior"]))

    def test_weight_mixes_the_two_probabilities(self):
        np.testing.assert_allclose(self.emitted_p_dead(0.0), self.PARTS["p_dead_without"])
        np.testing.assert_allclose(self.emitted_p_dead(1.0), self.PARTS["p_dead_with"])
        np.testing.assert_allclose(self.emitted_p_dead(0.25), [.3, .65])
        np.testing.assert_allclose(self.emitted_p_dead(None), self.emitted_p_dead(DEAD_BALL_WEIGHT))

    def test_weight_must_be_a_probability(self):
        for w in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                dead_ball_weight({"w": w})


class FalseDeadLimitTests(unittest.TestCase):
    """select_params scarta le combinazioni che dicono D dove la palla manca a gioco vivo."""
    LABELS = np.array(list("HHHHHHAAAAAADDDD"))
    MISSING = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)  # palla assente
    PREDS = {
        "scorciatoia": np.where(MISSING, "D", LABELS),        # giusto, tranne D su tutti i buchi vivi
        "metà": np.where(MISSING & (np.arange(16) < 6), "D", LABELS),  # D su metà dei buchi vivi
        "tutto_H": np.full(16, "H"),                           # nessun falso D, ma sbagliato
    }

    def games(self):
        features = pd.DataFrame({"ball_available": (~self.MISSING).astype(int)})
        cont = np.r_[False, np.ones(15, bool)]
        return [DahGame(name, features, None, self.LABELS, cont) for name in ("a", "b")]

    def choose(self, names, max_false_d):
        with mock.patch("src.model_dah._parts_for", return_value=None),              mock.patch("src.model_dah.decode", side_effect=lambda parts, c, t, params: self.PREDS[params["p"]]):
            return select_params(self.games(), {"p": names}, max_false_d=max_false_d)["p"]

    def test_rate_counts_only_live_frames_without_ball(self):
        self.assertEqual(false_dead_rate(self.LABELS, self.PREDS["scorciatoia"], self.games()[0].features), 1.0)
        self.assertEqual(false_dead_rate(self.LABELS, self.PREDS["metà"], self.games()[0].features), 0.5)

    def test_limit_excludes_the_shortcut(self):
        self.assertEqual(self.choose(("scorciatoia", "tutto_H"), max_false_d=1.0), "scorciatoia")
        self.assertEqual(self.choose(("scorciatoia", "tutto_H"), max_false_d=0.3), "tutto_H")

    def test_falls_back_to_fewest_false_dead_with_a_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(self.choose(("scorciatoia", "metà"), max_false_d=0.3), "metà")
        self.assertTrue(any("falsi D" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
