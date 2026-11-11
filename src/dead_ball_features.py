"""Feature di gioco fermo che non usano la palla: solo posizioni e velocità dei giocatori.

Nei dati Metrica D coincide quasi sempre con "palla assente", per come sono
costruite le etichette; nella challenge non è garantito. Queste feature descrivono
invece cosa fanno i giocatori quando il gioco è fermo:
- movimento collettivo: velocità media e mediana, quota di giocatori lenti e veloci;
- contesto temporale: medie prima e dopo il frame (offline), entro lo stesso tratto;
- schieramenti delle riprese: calcio d'inizio (squadre nelle due metà, qualcuno al
  centro), rimessa (giocatore sulla linea laterale o oltre), angolo (giocatore vicino
  alla bandierina), rinvio e rigore (aree occupate o svuotate), barriere ed esultanze
  (compagni raggruppati);
- assetto: distanza fra le squadre, estensione complessiva.

Tutte simmetriche rispetto al verso di attacco (aree e linee considerate da entrambi
i lati). Soglie geometriche dal regolamento (aree, centro); soglie di velocità fissate
a priori, non calibrate sulla challenge. NaN = informazione assente.
"""
import warnings
import numpy as np
import pandas as pd
from src.team_features import continuity

PENALTY_DEPTH_M, PENALTY_HALF_WIDTH_M = 16.5, 20.16
GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_WIDTH_M = 5.5, 9.16
CLUSTER_RADIUS_M = 3.0


def _blocks(continuous):
    block = np.cumsum(~continuous)
    index = np.arange(len(block))
    first = np.zeros(len(block), dtype=int)
    last = np.zeros(len(block), dtype=int)
    starts = np.flatnonzero(~continuous)
    ends = np.r_[starts[1:] - 1, len(block) - 1]
    for s, e in zip(starts, ends):
        first[s:e + 1], last[s:e + 1] = s, e
    return index, first, last


def window_mean(values, continuous, before, after):
    """Media ignorando i NaN su [i − before, i + after], senza uscire dal tratto continuo."""
    values = np.asarray(values, dtype=float)
    seen = np.isfinite(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(seen, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(seen)))
    index, first, last = _blocks(np.asarray(continuous, dtype=bool))
    lo = np.maximum(index - before, first)
    hi = np.minimum(index + after, last) + 1
    n = count[hi] - count[lo]
    return np.where(n > 0, (total[hi] - total[lo]) / np.maximum(n, 1), np.nan)


def _max_cluster(points, radius):
    """Per ogni frame, il massimo numero di punti entro `radius` da uno di essi (incluso)."""
    n = len(points)
    result = np.zeros(n)
    for a in range(0, n, 5000):
        chunk = points[a:a + 5000]
        d = np.linalg.norm(chunk[:, :, None] - chunk[:, None], axis=3)
        near = (d <= radius).sum(axis=2)  # NaN → False; ogni punto presente conta se stesso
        result[a:a + 5000] = near.max(axis=1) if chunk.shape[1] else 0
    return result


def build_dead_ball_features(tracking, *, fps=25, player_speed_max_ms=12.0, slow_ms=2.0, fast_ms=4.0,
                             windows_s=(2.0, 5.0)):
    """Una riga per frame. Non legge `tracking.ball`."""
    if tracking.pitch is None:
        raise ValueError("Dimensioni del campo assenti")
    length, width = tracking.pitch
    players = np.asarray(tracking.players, dtype=float)
    team = np.asarray(tracking.team)
    n = len(players)
    continuous = continuity(tracking.frames, tracking.segment)
    present = np.isfinite(players).all(axis=2)
    step = np.full(present.shape, np.nan)
    step[1:] = np.linalg.norm(players[1:] - players[:-1], axis=2) * fps
    step[~continuous] = np.nan
    step[step > player_speed_max_ms] = np.nan
    x, y = players[..., 0], players[..., 1]

    out = {"frame": np.asarray(tracking.frames)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # righe interamente NaN
        moving = np.isfinite(step)
        count = moving.sum(axis=1)
        out["players_present"] = present.sum(axis=1)
        out["all_mean_speed"] = np.where(count > 0, np.nanmean(step, axis=1), np.nan)
        out["all_median_speed"] = np.where(count > 0, np.nanmedian(step, axis=1), np.nan)
        out["share_slow"] = np.where(count > 0, (step < slow_ms).sum(axis=1) / np.maximum(count, 1), np.nan)
        out["share_fast"] = np.where(count > 0, (step > fast_ms).sum(axis=1) / np.maximum(count, 1), np.nan)
        for name, code in (("home", 0), ("away", 1)):
            mine = team == code
            speed = np.where(mine, step, np.nan)
            out[f"{name}_mean_speed"] = np.nanmean(speed, axis=1)
            out[f"{name}_max_cluster_3m"] = _max_cluster(np.where((mine & present)[..., None], players, np.nan),
                                                          CLUSTER_RADIUS_M)
            out[f"{name}_centroid_x_from_mid"] = np.nanmean(np.where(mine & present, x, np.nan), axis=1) - length / 2

        for seconds in windows_s:
            frames = int(round(seconds * fps))
            label = f"{seconds:g}s".replace(".", "_")
            for series in ("all_mean_speed", "share_slow"):
                past = window_mean(out[series], continuous, frames, 0)
                future = window_mean(out[series], continuous, 0, frames)
                out[f"{series}_past_{label}"] = past
                out[f"{series}_future_{label}"] = future
                out[f"{series}_trend_{label}"] = future - past

        # Calcio d'inizio: squadre in metà opposte, ognuna tutta dalla parte del proprio baricentro.
        hx, ax = out["home_centroid_x_from_mid"], out["away_centroid_x_from_mid"]
        out["teams_in_opposite_halves"] = np.where(np.isfinite(hx) & np.isfinite(ax), (hx * ax < 0).astype(float), np.nan)
        shares = []
        for code, cx in ((0, hx), (1, ax)):
            mine = (team == code) & present
            side = np.sign(cx)[:, None]
            same = (np.sign(x - length / 2) == side) & mine
            shares.append(np.where(mine.sum(axis=1) > 0, same.sum(axis=1) / np.maximum(mine.sum(axis=1), 1), np.nan))
        out["own_half_share_min"] = np.fmin(shares[0], shares[1])
        out["min_dist_center_spot"] = np.nanmin(np.where(present, np.hypot(x - length / 2, y - width / 2), np.inf), axis=1)

        # Linee e bandierine: distanza con segno (negativa fuori dal campo).
        sideline = np.where(present, np.minimum(y, width - y), np.nan)
        goal_line = np.where(present, np.minimum(x, length - x), np.nan)
        out["min_dist_sideline"] = np.nanmin(np.where(present, sideline, np.inf), axis=1)
        out["players_beyond_sideline"] = (sideline < 0).sum(axis=1)
        out["min_dist_goal_line"] = np.nanmin(np.where(present, goal_line, np.inf), axis=1)
        out["players_beyond_goal_line"] = (goal_line < 0).sum(axis=1)
        corners = np.stack([np.hypot(x - cx_, y - cy_) for cx_ in (0, length) for cy_ in (0, width)])
        out["min_dist_corner"] = np.nanmin(np.where(present, corners.min(axis=0), np.inf), axis=1)

        # Aree: conteggi per lato, poi massimo e minimo (simmetrici).
        central = np.abs(y - width / 2)
        for label, depth, half in (("penalty_area", PENALTY_DEPTH_M, PENALTY_HALF_WIDTH_M),
                                   ("goal_area", GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_WIDTH_M)):
            left = (present & (x >= 0) & (x <= depth) & (central <= half)).sum(axis=1)
            right = (present & (x <= length) & (x >= length - depth) & (central <= half)).sum(axis=1)
            out[f"{label}_max"] = np.maximum(left, right)
            out[f"{label}_min"] = np.minimum(left, right)

        out["max_cluster_3m"] = np.fmax(out["home_max_cluster_3m"], out["away_max_cluster_3m"])
        out["team_centroid_gap_x"] = np.abs(hx - ax)
        out["all_x_extent"] = np.nanmax(np.where(present, x, np.nan), axis=1) - np.nanmin(np.where(present, x, np.nan), axis=1)
        out["all_y_extent"] = np.nanmax(np.where(present, y, np.nan), axis=1) - np.nanmin(np.where(present, y, np.nan), axis=1)

    for key in ("min_dist_center_spot", "min_dist_sideline", "min_dist_goal_line", "min_dist_corner"):
        out[key] = np.where(np.isfinite(out[key]), out[key], np.nan)
    for key in ("home_centroid_x_from_mid", "away_centroid_x_from_mid"):
        out[key] = np.abs(out[key])  # distanza dalla metà campo, senza verso
    return pd.DataFrame(out)


def dead_ball_columns(features):
    return [c for c in features.columns if c != "frame"]
