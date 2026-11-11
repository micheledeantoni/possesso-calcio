import tempfile
from dataclasses import replace
from pathlib import Path
import unittest
import numpy as np
from src.arrays import TrackingArrays, load_metrica_npz
from src.dirtying import DirtParams, dirty, dirt_statistics, make_training_game, save_training_game


def clean_game(n=3000, seed=0):
    """22 giocatori che si muovono lentamente; il 12° marca il 1° a 1 m; palla sul 1°."""
    rng = np.random.default_rng(seed)
    base = np.column_stack([rng.uniform(10, 95, 22), rng.uniform(8, 60, 22)])
    t = np.arange(n)[:, None] / 25
    players = np.stack([base[:, 0] + 4 * np.sin(t / 9 + np.arange(22)),
                        base[:, 1] + 3 * np.cos(t / 7 + np.arange(22))], axis=2)
    players[:, 11] = players[:, 0] + (1., 0.)
    ball = players[:, 0] + (.3, 0.)
    labels = np.where((np.arange(n) // 500) % 2 == 0, "H", "A")
    labels[1400:1500] = "D"
    team = np.tile(np.array([0] * 11 + [1] * 11, np.int8), (n, 1))
    return TrackingArrays(np.arange(n), players, ball, team, np.ones(n, int), labels, pitch=(105., 68.))


SMALL = dict(ball_gaps=3, outside_objects=1, outside_frame_share=.05)


class DirtyingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = clean_game()
        cls.dirty, cls.log = dirty(cls.clean, cls.clean.labels, DirtParams(**SMALL), seed=7)

    def test_reproducible_with_seed_and_input_untouched(self):
        before = self.clean.players.copy()
        again, _ = dirty(self.clean, self.clean.labels, DirtParams(**SMALL), seed=7)
        other, _ = dirty(self.clean, self.clean.labels, DirtParams(**SMALL), seed=8)
        np.testing.assert_array_equal(again.players, self.dirty.players)
        self.assertFalse(np.array_equal(other.players, self.dirty.players, equal_nan=True))
        np.testing.assert_array_equal(self.clean.players, before)
        np.testing.assert_array_equal(self.dirty.labels, self.clean.labels)

    def test_anonymous_slots_with_fixed_team(self):
        pool = DirtParams().slots_per_team
        self.assertEqual(self.dirty.players.shape[1], 2 * pool)
        self.assertTrue((self.dirty.team == self.dirty.team[0]).all())
        self.assertEqual(self.dirty.team[0].tolist(), [0] * pool + [1] * pool)

    def test_points_are_copies_jumps_or_outside_objects(self):
        D = self.dirty.players.reshape(-1, 2)
        frames = np.repeat(np.arange(len(self.clean.players)), self.dirty.players.shape[1])
        keep = np.isfinite(D[:, 0])
        distance = np.linalg.norm(D[keep, None] - self.clean.players[frames[keep]], axis=2).min(axis=1)
        y = D[keep, 1]
        outside = (y < 0) | (y > 68)
        jumped = (distance >= .6 - 1e-9) & (distance <= 3 + 1e-9)
        self.assertTrue(((distance < 1e-9) | jumped | outside).all())
        self.assertEqual(int((jumped & ~outside).sum()), self.log["salti"])
        self.assertTrue(outside.any())
        depth = np.maximum(-y[outside], y[outside] - 68)
        self.assertTrue(((depth >= 1.5) & (depth <= 6)).all())

    def test_absences_follow_the_budget(self):
        target = DirtParams().absent_per_frame * len(self.clean.players)
        self.assertGreaterEqual(self.log["assenze_frame_giocatore"], target)
        stats = dirt_statistics(self.clean, self.dirty, self.clean.labels)
        self.assertGreater(stats["assenti_per_frame_gioco_vivo"], .5)

    def test_team_errors_only_when_requested(self):
        none, _ = dirty(self.clean, self.clean.labels,
                        DirtParams(**SMALL, team_error_pieces_per_min=0, whole_track_error_probability=0), seed=1)
        many, _ = dirty(self.clean, self.clean.labels, DirtParams(**SMALL, team_error_pieces_per_min=3), seed=1)
        self.assertEqual(dirt_statistics(self.clean, none, self.clean.labels)["punti_squadra_sbagliata"], 0)
        self.assertGreater(dirt_statistics(self.clean, many, self.clean.labels)["punti_squadra_sbagliata"], .05)

    def test_ball_gaps_avoid_dead_ball(self):
        removed = np.isfinite(self.clean.ball[:, 0]) & ~np.isfinite(self.dirty.ball[:, 0])
        self.assertEqual(self.log["buchi_palla"], 3)
        margin = int(DirtParams().ball_gap_margin_s * 25)
        self.assertFalse(removed[1400 - margin:1500 + margin].any())
        lengths = np.diff(np.flatnonzero(np.diff(np.r_[0, removed.astype(int), 0])))[::2] / 25
        self.assertTrue(((lengths >= 5) & (lengths <= 10)).all())
        np.testing.assert_array_equal(self.dirty.ball[~removed], self.clean.ball[~removed])

    def test_needs_clean_input(self):
        mixed = replace(self.clean, team=self.clean.team.copy())
        mixed.team[5, 0] = 1
        with self.assertRaises(ValueError):
            dirty(mixed, mixed.labels)

    def test_training_game_round_trip(self):
        cleaned, interpolated, log = make_training_game(self.clean, self.clean.labels, DirtParams(**SMALL), seed=3)
        self.assertIn("tracce_girate", log["pulizia"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "g_sporca.npz"
            save_training_game(path, cleaned, interpolated, self.clean.labels)
            loaded = load_metrica_npz(path)
        np.testing.assert_array_equal(loaded.team, cleaned.team)
        np.testing.assert_array_equal(loaded.labels, self.clean.labels)
        self.assertEqual(loaded.pitch, (105., 68.))


if __name__ == "__main__":
    unittest.main()
