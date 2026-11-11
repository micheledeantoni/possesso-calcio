"""Tracking in forma di array, con l'identità di traccia conservata.

Il lettore di `data.py` raggruppa le coordinate per squadra e perde lo slot:
basta per le distanze, non per le velocità. Qui ogni slot resta una colonna.

Formato comune a BAR-ATH e Metrica (`TrackingArrays`):
- frames  (n,)      indice assoluto, zero-based, crescente;
- players (n, S, 2) metri, NaN = osservazione assente;
- ball    (n, 2)    metri, NaN = palla assente;
- team    (n, S)    0 = casa, 1 = ospiti, per frame (nei dati sporcati cambia);
- segment (n,)      tratti continui: le derivate temporali non li attraversano;
- labels  (n,) o None, soltanto per dati esterni etichettati (Metrica);
- pitch   (lunghezza, larghezza) del campo in metri, o None se ignoto.
"""
from dataclasses import dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np


@dataclass
class TrackingArrays:
    frames: np.ndarray
    players: np.ndarray
    ball: np.ndarray
    team: np.ndarray
    segment: np.ndarray
    labels: np.ndarray = None
    pitch: tuple = None

    def __post_init__(self):
        n = len(self.frames)
        if self.players.ndim != 3 or self.players.shape[0] != n or self.players.shape[2] != 2:
            raise ValueError("players deve avere forma (n, S, 2)")
        if self.ball.shape != (n, 2):
            raise ValueError("ball deve avere forma (n, 2)")
        if self.team.shape != self.players.shape[:2]:
            raise ValueError("team deve avere forma (n, S)")
        if self.segment.shape != (n,):
            raise ValueError("segment deve avere forma (n,)")
        if self.labels is not None and len(self.labels) != n:
            raise ValueError("labels deve avere una voce per frame")
        if n and np.any(np.diff(self.frames) <= 0):
            raise ValueError("I frame devono essere crescenti e univoci")
        if not np.isin(self.team, (0, 1)).all():
            raise ValueError("team ammette soltanto 0 (casa) e 1 (ospiti)")


def _point(text, pitch):
    values = text.split(",")
    if len(values) < 2:
        raise ValueError("Osservazione priva delle coordinate x,y")
    x, y = float(values[0]), float(values[1])
    return (x * pitch[0], y * pitch[1]) if math.isfinite(x) and math.isfinite(y) else (math.nan, math.nan)


def load_atd_arrays(data_dir="data", pitch=(105.3, 68.0), start_frame=0, end_frame=None, halves=None):
    """ATD della challenge in array. Stesse convenzioni di `data.load_tracking`.

    Casa = prima squadra dei metadati. halves: intervalli (inizio, fine) inclusi;
    segment vale 1, 2, ... dentro un tempo e 0 fuori. None = un solo segmento.
    """
    if start_frame < 0 or (end_frame is not None and end_frame <= start_frame):
        raise ValueError("Intervallo richiesto non valido")
    folder = Path(data_dir)
    root = ET.parse(folder / "BAR-ATH_tactical_atd.xml").getroot()
    teams = [team.get("id") for team in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("Attese due squadre nei metadati")
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots or any(team not in teams for team in slots):
        raise ValueError("Associazione tracce/squadre non valida")
    frames, players, ball = [], [], []
    with (folder / "BAR-ATH_tactical_atd.txt").open() as handle:
        for number, line in enumerate(handle, start=1):
            try:
                frame_id, player_text, ball_text = line.strip().split(":")
                frame = int(frame_id) - 1
                if frame < start_frame:
                    continue
                if end_frame is not None and frame >= end_frame:
                    break
                groups = player_text.split(";")
                if len(groups) != len(slots):
                    raise ValueError("Numero tracce diverso dai metadati")
                frames.append(frame)
                players.append([_point(group, pitch) for group in groups])
                ball.append(_point(ball_text, pitch))
            except (ValueError, IndexError) as exc:
                raise ValueError("Record ATD non valido alla riga {}".format(number)) from exc
    frames = np.asarray(frames, dtype=int)
    team_row = np.array([0 if team == teams[0] else 1 for team in slots], dtype=np.int8)
    return TrackingArrays(
        frames=frames,
        players=np.asarray(players, dtype=float).reshape(len(frames), len(slots), 2),
        ball=np.asarray(ball, dtype=float).reshape(len(frames), 2),
        team=np.broadcast_to(team_row, (len(frames), len(slots))).copy(),
        segment=segment_from_halves(frames, halves), pitch=tuple(pitch))


def segment_from_halves(frames, halves=None):
    """1, 2, ... dentro gli intervalli inclusi di halves; 0 fuori."""
    frames = np.asarray(frames)
    if halves is None:
        return np.ones(len(frames), dtype=int)
    segment = np.zeros(len(frames), dtype=int)
    for number, (start, end) in enumerate(halves, start=1):
        segment[(frames >= start) & (frames <= end)] = number
    return segment


def arrays_from_demo(tracking):
    """Converte la demo inventata di `data.visualization_demo` in array.

    Vale solo per la demo: la posizione nella lista è la traccia e le assenze
    tolgono soltanto le ultime. Nei dati reali `data.load_tracking` scarta le
    osservazioni assenti e sposta le posizioni: usare `load_atd_arrays`.
    """
    home = max((len(points) for points in tracking.home_xy), default=0)
    away = max((len(points) for points in tracking.away_xy), default=0)
    n = len(tracking)
    players = np.full((n, home + away, 2), np.nan)
    ball = np.full((n, 2), np.nan)
    for i, row in enumerate(tracking.itertuples(index=False)):
        for offset, points in ((0, row.home_xy), (home, row.away_xy)):
            if points:
                players[i, offset:offset + len(points)] = points
        if row.ball_xy is not None:
            ball[i] = row.ball_xy
    team = np.broadcast_to(np.array([0] * home + [1] * away, dtype=np.int8), (n, home + away)).copy()
    return TrackingArrays(frames=tracking.frame.to_numpy(dtype=int), players=players, ball=ball,
                          team=team, segment=np.ones(n, dtype=int), pitch=(105.3, 68.0))


def load_metrica_npz(path):
    """npz Metrica preparati dal gruppo (dati esterni), puliti o sporcati.

    Puliti: `gioc`, `palla`, `squadra_slot`, `etichette`; nessun periodo, quindi
    un solo segmento. Sporcati: `squadra_fr` per frame e `periodo`.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        players = data["gioc"].astype(float)
        n, slots = players.shape[:2]
        if "squadra_fr" in data.files:
            team = data["squadra_fr"].astype(np.int8)
        elif "squadra_slot" in data.files:
            team = np.broadcast_to(data["squadra_slot"].astype(np.int8), (n, slots)).copy()
        else:
            raise ValueError("npz senza squadra_fr né squadra_slot")
        segment = data["periodo"].astype(int) if "periodo" in data.files else np.ones(n, dtype=int)
        labels = data["etichette"].astype(str) if "etichette" in data.files else None
        pitch = tuple(float(v) for v in data["campo"]) if "campo" in data.files else (105.0, 68.0)
        return TrackingArrays(frames=np.arange(n), players=players, ball=data["palla"].astype(float),
                              team=team, segment=segment, labels=labels, pitch=pitch)
