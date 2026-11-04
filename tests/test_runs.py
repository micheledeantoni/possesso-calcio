import tempfile
from pathlib import Path
import unittest
import pandas as pd
from src.data import synthetic_tracking
from src.runs import load_run, save_run
from src.preparation import prepare_tracking
from src.features import build_features
from src.algorithm import predict
from src.checks import check_tracking, check_predictions


class RunTests(unittest.TestCase):
    def config(self):
        return {"source": "synthetic", "start_frame": 5, "end_frame": 10,
                "synthetic_duration_s": 1, "run_name": "slice"}

    def test_slice_preserves_absolute_frames_and_exports(self):
        config = self.config()
        tracking, expected = load_run(config, Path.cwd())
        prepared, log = prepare_tracking(tracking)
        result = predict(build_features(prepared))
        checks = check_predictions(result, expected, start_frame=5)
        self.assertTrue(checks["full_index_ok"])
        self.assertEqual(result.frame.tolist(), [5, 6, 7, 8, 9])
        with tempfile.TemporaryDirectory() as directory:
            out = save_run(config, directory, result, check_tracking(tracking), log, checks)
            self.assertTrue((out / "predictions_slice.csv").exists())
            self.assertTrue((out / "run.json").exists())

    def test_preparation_preserves_original_and_rejects_duplicates(self):
        data = synthetic_tracking()
        data.at[0, "home_xy"] = [(float("nan"), 0), (1, 2), (1, 2)]
        prepared, log = prepare_tracking(data)
        self.assertEqual(len(data.iloc[0].home_xy), 3)
        self.assertEqual(prepared.iloc[0].home_xy, [(1, 2), (1, 2)])
        self.assertEqual(log["osservazioni_giocatore_non_finite_escluse"], 1)
        data.loc[1, "frame"] = 0
        with self.assertRaises(ValueError): prepare_tracking(data)

    def test_empty_or_oversized_slice_is_rejected(self):
        config = self.config()
        config["end_frame"] = 1000
        with self.assertRaises(ValueError): load_run(config, Path.cwd())

    def test_real_reader_selects_range_without_rebasing(self):
        from src.data import load_tracking
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "BAR-ATH_tactical_atd.xml").write_text('<root><Team id="h"/><Team id="a"/><Player teamId="h"/><Player teamId="a"/></root>')
            (folder / "BAR-ATH_tactical_atd.txt").write_text(''.join(f'{i}:0,0,0;1,0,0:0.1,0,0\n' for i in range(1, 7)))
            data = load_tracking(folder, start_frame=2, end_frame=5)
            self.assertEqual(data.frame.tolist(), [2, 3, 4])
