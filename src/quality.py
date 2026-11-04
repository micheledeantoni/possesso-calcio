"""Diagnostica descrittiva: nessuna correzione o calibrazione."""
import numpy as np
import pandas as pd

# Confini pubblicati in games.py/DATA.md della challenge; non ricavati dagli output.
HALVES = ((11185, 90651), (110808, 190197))


def quality_timeline(tracking, fps=25):
    """Una riga per record: assenze, conteggi, coordinate coincidenti e distanze.

    Coordinate identiche sono candidati da comprendere, non persone duplicate
    confermate. Non usa una soglia spaziale arbitraria per eliminare tracce.
    """
    rows = []
    for row in tracking.itertuples(index=False):
        same_team_pairs = 0
        min_distance = float("nan")
        for points in (row.home_xy, row.away_xy):
            if len(points) > 1:
                xy = np.asarray(points, dtype=float)
                delta = xy[:, None, :] - xy[None, :, :]
                distances = np.sqrt(np.sum(delta * delta, axis=2))[np.triu_indices(len(xy), 1)]
                same_team_pairs += int(np.count_nonzero(distances == 0))
                current = float(distances.min())
                min_distance = current if np.isnan(min_distance) else min(min_distance, current)
        points = row.home_xy + row.away_xy
        outside = sum(not (0 <= x <= 105.3 and 0 <= y <= 68.0) for x, y in points)
        rows.append({"frame": row.frame, "time_s": row.frame / fps,
                     "ball_observed": row.ball_xy is not None,
                     "home_tracks": len(row.home_xy), "away_tracks": len(row.away_xy),
                     "exact_same_team_pairs": same_team_pairs,
                     "min_same_team_distance_m": min_distance,
                     "player_observations_outside_pitch": outside})
    return pd.DataFrame(rows, columns=["frame", "time_s", "ball_observed", "home_tracks",
        "away_tracks", "exact_same_team_pairs", "min_same_team_distance_m",
        "player_observations_outside_pitch"])


def missing_runs(timeline, fps=25):
    """Durate dei buchi della palla; separa i record non consecutivi."""
    lengths, length, previous = [], 0, None
    for row in timeline.itertuples(index=False):
        if previous is not None and row.frame != previous + 1 and length:
            lengths.append(length / fps)
            length = 0
        if not row.ball_observed:
            length += 1
        elif length:
            lengths.append(length / fps)
            length = 0
        previous = row.frame
    if length:
        lengths.append(length / fps)
    return lengths


def quality_summary(timeline):
    """Separa intera registrazione e soli tempi di gioco pubblicati."""
    masks = {"intera_registrazione": np.ones(len(timeline), dtype=bool)}
    masks["tempi_di_gioco"] = np.logical_or.reduce([
        timeline.frame.between(start, end).to_numpy() for start, end in HALVES])
    result = {}
    for name, mask in masks.items():
        part = timeline.loc[mask]
        total = part.home_tracks + part.away_tracks
        gaps = missing_runs(part)
        result[name] = {
            "record": len(part),
            "palla_assente_percento": float(100 * (~part.ball_observed).mean()) if len(part) else None,
            "record_senza_tracce_casa": int(part.home_tracks.eq(0).sum()),
            "record_senza_tracce_ospiti": int(part.away_tracks.eq(0).sum()),
            "record_con_piu_di_22_tracce": int(total.gt(22).sum()),
            "record_con_coordinate_coincidenti_stessa_squadra": int(part.exact_same_team_pairs.gt(0).sum()),
            "osservazioni_giocatori_fuori_campo": int(part.player_observations_outside_pitch.sum()),
            "buco_palla_massimo_secondi": max(gaps, default=0),
        }
    return result
