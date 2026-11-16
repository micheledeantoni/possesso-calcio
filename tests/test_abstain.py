"""Astensione (lavoro A8): X dove il modello è incerto."""
import unittest
import numpy as np
from src.challenge_metrics import macro_f1, spell_f1
from src.model_dah import abstain, cross_validate, fit_dah_model, predict_dah
from tests.test_model_dah import SMALL_GRID, synthetic_game


class AbstainRuleTests(unittest.TestCase):
    POST = np.array([[.9, .05, .05],   # sicuro
                     [.45, .40, .15],  # incerto
                     [.30, .30, .40],  # molto incerto
                     [.05, .90, .05]])
    LABELS = np.array(["H", "H", "D", "A"])

    def test_x_only_below_the_threshold(self):
        self.assertEqual(abstain(self.LABELS, self.POST, .5).tolist(), ["H", "X", "X", "A"])
        self.assertEqual(abstain(self.LABELS, self.POST, .42).tolist(), ["H", "H", "X", "A"])

    def test_no_threshold_changes_nothing(self):
        self.assertEqual(abstain(self.LABELS, None, None).tolist(), self.LABELS.tolist())

    def test_invalid_threshold(self):
        for bad in (0, -.1, 1.5):
            with self.assertRaises(ValueError):
                abstain(self.LABELS, self.POST, bad)


class AbstainMetricTests(unittest.TestCase):
    TRUE = np.array(list("HHHHHAAAAADDDDD"))

    def test_x_costs_less_than_a_wrong_label_in_macro_f1(self):
        wrong = self.TRUE.copy()
        wrong[4] = "A"                       # errore: penalizza H (recall) e A (precisione)
        unsure = self.TRUE.copy()
        unsure[4] = "X"                      # astensione: penalizza solo H (recall)
        self.assertGreater(macro_f1(self.TRUE, unsure)["macroF1"], macro_f1(self.TRUE, wrong)["macroF1"])

    def test_x_inside_a_spell_does_not_change_spell_f1(self):
        pred = self.TRUE.copy()
        pred[2] = "X"                        # riempito con l'ultima etichetta: H
        self.assertEqual(spell_f1(self.TRUE, pred)["spellF1"], 1.0)

    def test_x_on_the_first_frame_of_a_tempo_takes_the_next_label(self):
        breaks = np.zeros(len(self.TRUE), dtype=bool)
        breaks[5] = True                     # il secondo tempo inizia al frame 5 con A
        pred = self.TRUE.copy()
        pred[5] = "X"                        # niente prima nel tratto: vale l'etichetta dopo, A
        self.assertEqual(spell_f1(self.TRUE, pred, breaks)["spellF1"], 1.0)


class AbstainModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.games = [synthetic_game(seed, name=f"g{seed}") for seed in (1, 2, 3)]

    def test_predict_writes_x_where_the_top_probability_is_low(self):
        model = fit_dah_model(self.games[:2], SMALL_GRID)
        test = self.games[2]
        plain = predict_dah(model, test.features, test.dead_features, test.continuous, probabilities=True,
                            abstain_threshold=None)
        top = plain[["p_H", "p_A", "p_D"]].max(axis=1).to_numpy()
        threshold = float(np.quantile(top, .1))  # circa il 10% dei frame sotto soglia
        result = predict_dah(model, test.features, test.dead_features, test.continuous, abstain_threshold=threshold)
        x = result.label.to_numpy() == "X"
        np.testing.assert_array_equal(x, top < threshold)
        same = ~x
        np.testing.assert_array_equal(result.label.to_numpy()[same], plain.label.to_numpy()[same])

    def test_cross_validation_reports_each_threshold(self):
        table = cross_validate(self.games, SMALL_GRID, abstain_thresholds=(.5, .9))
        for column in ("spellF1_X0.5", "macroF1_X0.5", "quota_X0.5", "quota_X0.9"):
            self.assertIn(column, table)
        self.assertTrue((table["quota_X0.9"] >= table["quota_X0.5"]).all())  # soglia più alta, più X


if __name__ == "__main__":
    unittest.main()
