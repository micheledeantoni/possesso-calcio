import tempfile
from pathlib import Path
import unittest
from src.data import synthetic_tracking, load_tracking
from src.features import build_features
from src.algorithm import predict
from src.checks import check_tracking, check_predictions, save_predictions


class PipelineTests(unittest.TestCase):
    def test_synthetic_pipeline_preserves_missing_information(self):
        tracking = synthetic_tracking()
        result = predict(build_features(tracking))
        self.assertEqual(result.label.tolist(), ["H", "X", "H", "A", "X"])
        self.assertTrue(check_predictions(result, 5)["expected_rows_ok"])
        self.assertEqual(check_tracking(tracking)["ball_missing_rows"], 1)

    def test_team_swap(self):
        tracking = synthetic_tracking()
        original = predict(build_features(tracking))
        swapped = tracking.copy()
        swapped["home_xy"], swapped["away_xy"] = tracking["away_xy"], tracking["home_xy"]
        self.assertEqual(predict(build_features(swapped)).label.tolist(),
                         original.label.map({"H": "A", "A": "H", "X": "X"}).tolist())

    def test_local_reader_with_invented_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "BAR-ATH_tactical_atd.xml").write_text(
                '<root><Team id="h"/><Team id="a"/><Player teamId="h"/><Player teamId="a"/></root>')
            (path / "BAR-ATH_tactical_atd.txt").write_text(
                '1:0,0,0;1,0,0:0.1,0,0\n2:0,0,0;1,0,0:NaN,NaN,NaN\n')
            tracking = load_tracking(path, pitch=(100, 60))
            self.assertEqual(tracking.iloc[0].ball_xy, (10, 0))
            self.assertEqual(predict(build_features(tracking)).label.tolist(), ["H", "X"])
            save_predictions(predict(build_features(tracking)), path / "out.csv")
            self.assertTrue((path / "out.csv").exists())

    def test_duplicate_frames_are_detected(self):
        tracking = synthetic_tracking()
        tracking.loc[1, "frame"] = 0
        self.assertEqual(check_tracking(tracking)["duplicates"], 1)
        self.assertFalse(check_tracking(tracking)["consecutive_in_file"])
