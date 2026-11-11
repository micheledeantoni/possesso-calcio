import math
import unittest
import numpy as np
from src.arrays import TrackingArrays
from src.team_features import build_team_features, continuity
from src.dead_ball_features import build_dead_ball_features
from src.model import viterbi
from src.model_dah import (DahGame, transition_matrix, scaled_transitions, viterbi_k, posteriors,
                           evaluate_dah, fit_dah_model, predict_dah, cross_validate, STATES)


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


SMALL_GRID = {"temperature": (0.2,), "switch_scale": (1.0,), "neutral_without_ball": (False,)}


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

    def test_nested_cross_validation(self):
        table = cross_validate(self.games, SMALL_GRID)
        self.assertEqual(len(table), 3)
        self.assertTrue((table.accuratezza_tutti > .85).all())


if __name__ == "__main__":
    unittest.main()
