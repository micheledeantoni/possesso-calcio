"""Lettura locale del tracking tattico; nessuna etichetta ufficiale caricata."""
from pathlib import Path
import xml.etree.ElementTree as ET
import math
import pandas as pd


def inventory(data_dir="data"):
    """Elenca presenza dei due input essenziali, senza aprirli."""
    folder = Path(data_dir)
    return pd.DataFrame([
        {"file": name, "present": (folder / name).is_file()}
        for name in ("BAR-ATH_tactical_atd.xml", "BAR-ATH_tactical_atd.txt",
                     "BAR-ATH_tactical_smart_tagging.xml")
    ])


def _xy(group, pitch):
    values = group.split(",")
    if len(values) < 2:
        raise ValueError("Osservazione priva delle coordinate x,y")
    x, y = float(values[0]), float(values[1])
    return (x * pitch[0], y * pitch[1]) if math.isfinite(x) and math.isfinite(y) else None


def load_tracking(data_dir="data", pitch=(105.3, 68.0), limit=None, start_frame=0, end_frame=None):
    """Legge ATD in ordine file. Coordinate 0–1 convertite in metri.

    pitch riprende games.py ufficiale: non è una soglia calibrata.
    frame è zero-based. Nessun riordino, riempimento o correzione automatica.
    limit limita il numero di record selezionati. start_frame incluso,
    end_frame escluso; gli indici originali non vengono rinumerati.
    """
    if start_frame < 0 or (end_frame is not None and end_frame <= start_frame):
        raise ValueError("Intervallo richiesto non valido")
    folder = Path(data_dir)
    tree = ET.parse(folder / "BAR-ATH_tactical_atd.xml").getroot()
    teams = [team.get("id") for team in tree.iter("Team")]
    if len(teams) != 2:
        raise ValueError("Attese due squadre nei metadati")
    slots = [player.get("teamId") for player in tree.iter("Player")]
    if not slots or any(team not in teams for team in slots):
        raise ValueError("Associazione tracce/squadre non valida")
    rows = []
    with (folder / "BAR-ATH_tactical_atd.txt").open() as handle:
        for number, line in enumerate(handle, start=1):
            if limit is not None and len(rows) >= limit:
                break
            try:
                frame_id, player_text, ball_text = line.strip().split(":")
                video_frame = int(frame_id) - 1
                if video_frame < start_frame:
                    continue
                if end_frame is not None and video_frame >= end_frame:
                    break
                groups = player_text.split(";")
                if len(groups) != len(slots):
                    raise ValueError("Numero tracce diverso dai metadati")
                home, away = [], []
                for team, group in zip(slots, groups):
                    point = _xy(group, pitch)
                    if point is not None:
                        (home if team == teams[0] else away).append(point)
                rows.append({"frame": int(frame_id) - 1,
                             "ball_xy": _xy(ball_text, pitch),
                             "home_xy": home, "away_xy": away})
            except (ValueError, IndexError) as exc:
                raise ValueError("Record ATD non valido alla riga {}".format(number)) from exc
    return pd.DataFrame(rows, columns=["frame", "ball_xy", "home_xy", "away_xy"])


def synthetic_tracking():
    """Piccola sequenza inventata: H, buco, H, A, parità. Nessun dato reale."""
    return pd.DataFrame([
        {"frame": frame, "ball_xy": ball,
         "home_xy": [(0.0, 0.0)], "away_xy": [(8.0, 0.0)]}
        for frame, ball in enumerate([(1., 0.), None, (2., 0.), (7., 0.), (4., 0.)])
    ])


def visualization_demo(fps=25, duration_s=120):
    """Scenario artificiale per grafici leggibili, non statistiche della partita.

    Due squadre da undici tracce, movimento periodico, buchi inventati nella
    palla e in alcune tracce. Separato dalla fixture minima dei test.
    """
    import numpy as np
    home_base = [(8., 34.), (27., 9.), (27., 25.), (27., 43.), (27., 59.),
                 (46., 14.), (46., 34.), (46., 54.), (66., 15.), (66., 34.), (66., 53.)]
    away_base = [(105.3 - x, 68. - y) for x, y in home_base]
    rows = []
    for frame in range(round(duration_s * fps)):
        t = frame / fps
        home = [(x + 3 * np.sin(t / 7 + i / 3), y + 2 * np.sin(t / 5 + i))
                for i, (x, y) in enumerate(home_base)]
        away = [(x + 3 * np.sin(t / 9 + i / 4), y + 2 * np.cos(t / 6 + i))
                for i, (x, y) in enumerate(away_base)]
        if 35 <= t < 43:
            home = home[:-2]
        if 75 <= t < 81:
            away = away[:-1]
        ball = (52.65 + 28 * np.sin(t / 11), 34 + 20 * np.sin(t / 8))
        if any(start <= t < end for start, end in [(8, 8.2), (22, 23), (48, 53), (86, 88), (110, 110.4)]):
            ball = None
        rows.append({"frame": frame, "ball_xy": ball, "home_xy": home, "away_xy": away})
    return pd.DataFrame(rows)
