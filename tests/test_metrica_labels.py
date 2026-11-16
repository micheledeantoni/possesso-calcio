import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.arrays import load_metrica_npz
from src.metrica_labels import (make_labels, period_from_events, summary, read_csv_game, read_epts_game,
                                save_labelled_game, FPS)


def events(*rows):
    return pd.DataFrame(rows, columns=["frame", "type", "subtype", "team"])


class LabelRuleTests(unittest.TestCase):
    def ball_with_gap(self, n, start, end):
        ball = np.zeros((n, 2))
        ball[start:end] = np.nan
        return ball

    def test_gap_after_ball_out_is_dead(self):
        ball = self.ball_with_gap(400, 100, 200)
        labels, dead, live = make_labels(ball, events((0, "PASS", "", 0), (60, "BALL OUT", "", 0),
                                                      (250, "PASS", "", 1)))
        self.assertTrue((labels[100:200] == "D").all())
        self.assertEqual((dead, live), (1, 0))
        self.assertTrue((labels[:100] == "H").all())
        self.assertTrue((labels[250:] == "A").all())

    def test_stop_window_is_three_seconds_before_and_one_after(self):
        for stop, expected in ((100 - 3 * FPS, "D"), (100 - 3 * FPS - 1, "H"), (100 + FPS, "D"), (100 + FPS + 1, "H")):
            ball = self.ball_with_gap(400, 100, 200)
            labels, _, _ = make_labels(ball, events((0, "PASS", "", 0), (stop, "FAULT RECEIVED", "", 0)))
            self.assertEqual(labels[150], expected, stop)

    def test_gap_ending_with_a_restart_is_dead(self):
        ball = self.ball_with_gap(400, 100, 200)
        labels, _, _ = make_labels(ball, events((0, "PASS", "", 0), (200 + FPS, "SET PIECE", "THROW IN", 1)))
        self.assertTrue((labels[100:200] == "D").all())
        labels, _, _ = make_labels(ball, events((0, "PASS", "", 0), (200 + FPS + 1, "SET PIECE", "THROW IN", 1)))
        self.assertTrue((labels[100:200] == "H").all())

    def test_goal_kick_subtype_counts_as_restart(self):
        ball = self.ball_with_gap(400, 100, 200)
        labels, _, _ = make_labels(ball, events((0, "PASS", "", 0), (200, "PASS", "GOAL KICK", 1)))
        self.assertTrue((labels[100:200] == "D").all())
        self.assertTrue((labels[200:] == "A").all())

    def test_unexplained_gap_stays_live(self):
        ball = self.ball_with_gap(400, 100, 200)
        labels, dead, live = make_labels(ball, events((0, "PASS", "", 1)))
        self.assertTrue((labels == "A").all())
        self.assertEqual((dead, live), (0, 1))

    def test_x_before_first_possession_event_and_non_possession_events_ignored(self):
        labels, _, _ = make_labels(np.zeros((100, 2)), events((10, "CHALLENGE", "", 1), (30, "PASS", "", 0)))
        self.assertTrue((labels[:30] == "X").all())
        self.assertTrue((labels[30:] == "H").all())

    def test_periods_from_first_event_of_each_half(self):
        e = pd.DataFrame({"frame": [0, 50, 120, 130], "period": [1, 1, 2, 2]})
        period = period_from_events(200, e)
        self.assertTrue((period[:120] == 1).all() and (period[120:] == 2).all())

    def test_summary(self):
        labels = np.array(list("HHHHAAADDDHH"))
        spells, possessions, duration, short = summary(labels, fps=2)
        self.assertEqual((spells, possessions), (4, 3))
        self.assertAlmostEqual(duration, 3 / 2)
        self.assertAlmostEqual(short, 0.0)


class ReaderTests(unittest.TestCase):
    def test_csv_game_and_round_trip_through_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "Sample_Game_1"
            folder.mkdir()
            for side, x in (("Home", 0.25), ("Away", 0.75)):
                lines = [",,,{0},,{0},,,".format(side), ",,,1,,2,,,",
                         "Period,Frame,Time [s],Player1,,Player2,,Ball,"]
                for frame in range(1, 7):
                    period = 1 if frame <= 3 else 2
                    ball = "NaN,NaN" if frame in (3, 4) else "0.5,0.5"
                    lines.append(f"{period},{frame},{frame * .04:.2f},{x},0.5,{x},0.25,{ball}")
                (folder / f"Sample_Game_1_RawTrackingData_{side}_Team.csv").write_text("\n".join(lines) + "\n")
            (folder / "Sample_Game_1_RawEventsData.csv").write_text(
                "Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,"
                "Start X,Start Y,End X,End Y\n"
                "Home,PASS,,1,1,0.04,2,0.08,Player1,Player2,0.5,0.5,0.5,0.5\n"
                "Away,SET PIECE,KICK OFF,2,5,0.20,5,0.20,Player1,,NaN,NaN,NaN,NaN\n")
            players, team, ball, period, ev = read_csv_game(tmp, 1)
            self.assertEqual(players.shape, (6, 4, 2))
            np.testing.assert_array_equal(team, [0, 0, 1, 1])
            np.testing.assert_allclose(players[0, 0], [0.25 * 105, 0.5 * 68])
            np.testing.assert_array_equal(period, [1, 1, 1, 2, 2, 2])
            self.assertEqual(ev.frame.tolist(), [0, 4])  # zero-based
            self.assertEqual(ev.team.tolist(), [0, 1])
            labels, _, _ = make_labels(ball, ev)
            self.assertEqual("".join(labels), "HHDDAA")  # buco chiuso dal calcio d'inizio
            path = Path(tmp) / "metrica_partita_1.npz"
            save_labelled_game(path, players, team, ball, period, labels)
            arrays = load_metrica_npz(path)
            np.testing.assert_array_equal(arrays.segment, period)
            np.testing.assert_array_equal(arrays.labels, labels)
            self.assertEqual(arrays.pitch, (105.0, 68.0))

    def test_epts_game_follows_column_changes_and_periods(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "Sample_Game_3"
            folder.mkdir()
            players = "".join(f'<Player id="P{i}" teamId="{t}"/>' for i, t in ((1, "T1"), (2, "T2"), (3, "T1")))
            channels = "".join(f'<PlayerChannel id="c{i}_x" playerId="P{i}"/><PlayerChannel id="c{i}_y" playerId="P{i}"/>'
                               for i in (1, 2, 3))
            spec = '<DataFormatSpecification startFrame="{}" endFrame="{}">{}</DataFormatSpecification>'
            refs = lambda ids: "".join(f'<PlayerChannelRef playerChannelId="c{i}_x"/>'
                                       f'<PlayerChannelRef playerChannelId="c{i}_y"/>' for i in ids)
            (folder / "Sample_Game_3_metadata.xml").write_text(
                f'<Main><Teams><Team id="T1"/><Team id="T2"/></Teams><Players>{players}</Players>'
                f'<Channels>{channels}</Channels><DataFormatSpecifications>'
                f'{spec.format(1, 2, refs((1, 2)))}{spec.format(3, 4, refs((3, 2)))}'
                f'</DataFormatSpecifications></Main>')
            (folder / "Sample_Game_3_tracking.txt").write_text(
                "1:0.1,0.1;0.9,0.9:0.5,0.5\n2:0.1,0.1;NaN,NaN:NaN,NaN\n"
                "3:0.2,0.2;0.8,0.8:0.5,0.5\n4:0.2,0.2;0.8,0.8:0.5,0.5\n")
            team = lambda t: {"id": t, "name": t}
            (folder / "Sample_Game_3_events.json").write_text(json.dumps({"data": [
                {"start": {"frame": 1}, "type": {"name": "PASS"}, "subtypes": None, "team": team("T1"), "period": 1},
                {"start": {"frame": 3}, "type": {"name": "SET PIECE"}, "subtypes": {"name": "KICK OFF"},
                 "team": team("T2"), "period": 2}]}))
            players, team_slot, ball, period, ev = read_epts_game(tmp)
            np.testing.assert_array_equal(team_slot, [0, 1, 0])  # P1, P2, P3
            np.testing.assert_allclose(players[0, 0], [0.1 * 105, 0.1 * 68])
            self.assertTrue(np.isnan(players[0, 2]).all())               # P3 non ancora in campo
            np.testing.assert_allclose(players[2, 2], [0.2 * 105, 0.2 * 68])  # P3 sostituisce P1
            self.assertTrue(np.isnan(players[2, 0]).all())
            self.assertTrue(np.isnan(ball[1]).all())
            np.testing.assert_array_equal(period, [1, 1, 2, 2])
            self.assertNotIn("period", ev.columns)
            labels, _, _ = make_labels(ball, ev)
            self.assertEqual("".join(labels), "HDAA")


if __name__ == "__main__":
    unittest.main()
