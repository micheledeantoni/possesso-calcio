import tempfile
from pathlib import Path
import unittest
import numpy as np
from src.events import load_events, synthetic_events
from src.official_baseline import smooth


class AdaptationTests(unittest.TestCase):
    def test_event_frames_and_unknown_team(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "BAR-ATH_tactical_smart_tagging.xml"
            path.write_text('<root><instance><start>1.2</start><end>2.0</end><code>SET PIECES</code><label><group>Team</group><text>N/A</text></label></instance></root>')
            row = load_events(directory).iloc[0]
            self.assertEqual((row.start_frame, row.end_frame), (30, 50))
            self.assertEqual(row.Team, 'N/A')

    def test_vote_ignores_missing_when_team_available(self):
        self.assertEqual(smooth(np.array([0, -1, 0]), 1).tolist(), ['H', 'H', 'H'])
        self.assertEqual(smooth(np.array([-1, -1]), 1).tolist(), ['D', 'D'])
        self.assertEqual(smooth(np.array([0, 1]), 1).tolist(), ['H', 'H'])
