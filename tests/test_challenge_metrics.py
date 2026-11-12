import unittest
import numpy as np
from src.challenge_metrics import (valid_runs, fill_abstentions, possession_spells, spell_f1, macro_f1,
                                   challenge_scores, _codes)


def truth_sequence(seed=0, spells=60):
    """Possessi di 2–12 s alternati, con palle morte da 1–4 s fra alcuni di essi."""
    rng = np.random.default_rng(seed)
    labels, team = [], "H"
    for _ in range(spells):
        labels += [team] * int(rng.integers(50, 300))
        if rng.random() < .4:
            labels += ["D"] * int(rng.integers(25, 100))
        team = "A" if team == "H" else "H"
    return np.array(["X"] * 100 + labels + ["X"] * 100)


class SpellTests(unittest.TestCase):
    def test_perfect_and_one_frame_late(self):
        truth = truth_sequence()
        self.assertEqual(challenge_scores(truth, truth)["spellF1"], 1.)
        self.assertEqual(challenge_scores(truth, truth)["macroF1"], 1.)
        late = np.r_[truth[:1], truth[:-1]]
        self.assertEqual(spell_f1(truth, late)["spellF1"], 1.)

    def test_score_falls_with_lag_and_constant_guess_scores_zero(self):
        truth = truth_sequence()
        scores = [spell_f1(truth, np.r_[truth[:k], truth[:-k]])["spellF1"] for k in (3, 8, 25, 60)]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertLess(scores[-1], .8)
        for guess in "HAD":
            self.assertEqual(spell_f1(truth, np.full(len(truth), guess))["spellF1"], 0.)

    def test_dead_ball_breaks_a_spell_and_x_is_not_graded(self):
        codes = _codes(list("XHHHDDHHHAAX"))
        runs = valid_runs(codes)
        self.assertEqual(runs, [(1, 11)])
        self.assertEqual(possession_spells(codes, runs), [(1, 4, 0), (6, 9, 0), (9, 11, 1)])

    def test_breaks_split_runs_without_x(self):
        codes = _codes(list("HHHHAA"))
        breaks = np.array([False, False, False, True, False, False])
        self.assertEqual(valid_runs(codes, breaks), [(0, 3), (3, 6)])
        self.assertEqual(possession_spells(codes, valid_runs(codes, breaks)), [(0, 3, 0), (3, 4, 0), (4, 6, 1)])

    def test_abstentions_are_filled_inside_runs_only(self):
        codes = _codes(list("XXHHAXAXX"))
        filled = fill_abstentions(codes, [(0, 7), (7, 9)])
        self.assertEqual(filled.tolist(), _codes(list("HHHHAAAXX")).tolist())

    def test_iou_thresholds(self):
        truth = np.array(list("D" + "H" * 10 + "D"))
        pred = np.array(list("D" * 4 + "H" * 4 + "D" * 4))  # IoU 0,4
        result = spell_f1(truth, pred)
        self.assertEqual([round(v, 6) for v in result["spell_f1_per_iou"].values()], [1., 1., 0., 0., 0.])
        self.assertAlmostEqual(result["spellF1"], .4)

    def test_different_team_never_matches(self):
        truth = np.array(list("HHHHHH"))
        self.assertEqual(spell_f1(truth, np.array(list("AAAAAA")))["spellF1"], 0.)

    def test_rejects_unknown_labels_and_length_mismatch(self):
        with self.assertRaises(ValueError):
            spell_f1(np.array(list("HH")), np.array(list("HZ")))
        with self.assertRaises(ValueError):
            spell_f1(np.array(list("HH")), np.array(list("H")))


class MacroF1Tests(unittest.TestCase):
    def test_predicted_x_is_only_a_false_negative(self):
        truth = np.array(list("HHAADD"))
        pred = np.array(list("HXAADD"))
        result = macro_f1(truth, pred)
        self.assertAlmostEqual(result["F1_H"], 2 * 1 * .5 / 1.5)
        self.assertEqual(result["F1_A"], 1.)
        self.assertEqual(result["F1_D"], 1.)

    def test_truth_x_frames_are_ignored(self):
        self.assertEqual(macro_f1(np.array(list("XXHAD")), np.array(list("DDHAD")))["macroF1"], 1.)


if __name__ == "__main__":
    unittest.main()
