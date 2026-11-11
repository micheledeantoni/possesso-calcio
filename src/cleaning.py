"""Pulizia di giocatori e palla sul tracking ad array.

Riprende lo script del gruppo "Pulizia dati giocatori e palla" (stesse fasi,
stesse soglie fissate a priori), con alcune correzioni:
- campo, fps e soglie sono parametri, non costanti globali (Metrica 105 m, BAR-ATH 105,3 m);
- i doppioni non vengono mediati: resta il punto della traccia più lunga, invariato,
  così la sua velocità non contiene salti artificiali;
- tracce e interpolazione della palla non attraversano salti di frame o cambi di segmento;
- l'interpolazione della palla resta dentro i frame valutati (segmento > 0).

Fasi per i giocatori, in quest'ordine:
0. tracce: tratti continui di uno slot, spezzati anche dove supera `player_speed_max_ms`;
1. cambio di squadra traccia per traccia, solo se riduce gli oggetti oltre 11;
2. nei frame con più di 11 oggetti in una squadra: doppioni a meno di `duplicate_m` uniti;
3. se ancora troppi: via gli oggetti oltre `outside_margin_m` fuori dal campo, dal più
   lontano, mai il più vicino alla palla.
Non aggiunge giocatori e non tocca i frame con 11 oggetti o meno per squadra.

Palla: via i punti in cui arriva e riparte oltre `ball_speed_max_ms` (o sparisce subito
dopo); buchi fino a `ball_max_gap_s` riempiti linearmente solo se fisicamente possibili.
"""
from dataclasses import replace
import numpy as np
from src.team_features import continuity


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    return starts, np.flatnonzero(edges == -1) - starts


def split_tracks(players, team, continuous, *, speed_max_ms, fps):
    """Tracce (slot, inizio, lunghezza): tratti con slot presente, frame continui,
    squadra costante e spostamenti entro `speed_max_ms`."""
    present = np.isfinite(players).all(axis=2)
    step = np.full(present.shape, np.nan)
    step[1:] = np.linalg.norm(players[1:] - players[:-1], axis=2) * fps
    jump = np.nan_to_num(step, nan=0.0) > speed_max_ms
    previous = np.vstack([np.zeros((1, present.shape[1]), bool), present[:-1]])
    team_change = np.vstack([np.zeros((1, team.shape[1]), bool), team[1:] != team[:-1]])
    start = present & (~previous | jump | ~continuous[:, None] | team_change)
    tracks = []
    for slot in range(present.shape[1]):
        ids = np.cumsum(start[:, slot])
        where = np.flatnonzero(present[:, slot])
        if not len(where):
            continue
        _, first, length = np.unique(ids[where], return_index=True, return_counts=True)
        tracks += [(slot, int(a), int(l)) for a, l in zip(where[first], length)]
    return tracks, int((jump & present & previous).sum())


def reassign_teams(present, team, tracks, evaluated, *, max_team, min_frames, flip_quota):
    """Cambia squadra alle tracce che per più di `flip_quota` dei frame valutati
    stanno in una squadra oltre `max_team` mentre l'altra è sotto, se l'eccesso cala."""
    team = team.copy()
    counts = [(present & (team == k)).sum(axis=1).astype(np.int32) for k in (0, 1)]

    def excess(c0, c1):
        return np.maximum(c0 - max_team, 0) + np.maximum(c1 - max_team, 0)

    candidates = []
    for slot, start, length in tracks:
        inside = evaluated[start:start + length]
        if inside.sum() < min_frames:
            continue
        own = team[start, slot]
        mine = counts[own][start:start + length][inside]
        other = counts[1 - own][start:start + length][inside]
        quota = float(((mine > max_team) & (other < max_team)).mean())
        if quota > flip_quota:
            candidates.append((quota, slot, start, length))
    candidates.sort(reverse=True)
    flipped = frames_flipped = 0
    for quota, slot, start, length in candidates:
        own = team[start, slot]
        part = slice(start, start + length)
        inside = evaluated[part]
        new = [counts[0][part].copy(), counts[1][part].copy()]
        before = excess(new[0], new[1])[inside].sum()
        new[own] -= 1
        new[1 - own] += 1
        if excess(new[0], new[1])[inside].sum() < before:
            team[part, slot] = 1 - own
            counts[0][part], counts[1][part] = new
            flipped += 1
            frames_flipped += length
    return team, {"tracce_candidate": len(candidates), "tracce_girate": flipped,
                  "frame_girati": frames_flipped}


def merge_and_remove(players, team, ball, track_length, *, pitch, max_team, duplicate_m, outside_margin_m):
    """Fasi 2 e 3 sui soli frame con più di `max_team` oggetti in una squadra."""
    players = players.copy()
    length, width = pitch
    log = {"doppioni_uniti": 0, "tolti_fuori_campo": 0, "frame_ancora_troppi": 0}
    present = np.isfinite(players).all(axis=2)
    for code in (0, 1):
        crowded = np.flatnonzero((present & (team == code)).sum(axis=1) > max_team)
        for i in crowded:
            alive = list(np.flatnonzero((team[i] == code) & present[i]))
            while len(alive) > max_team:  # fase 2: doppioni
                xy = players[i, alive]
                distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
                np.fill_diagonal(distance, np.inf)
                a, b = np.unravel_index(np.argmin(distance), distance.shape)
                if distance[a, b] > duplicate_m:
                    break
                sa, sb = alive[a], alive[b]
                # Resta la traccia più lunga; a parità lo slot con indice minore.
                drop = sb if (track_length[i, sa], -sa) >= (track_length[i, sb], -sb) else sa
                players[i, drop] = np.nan
                alive.remove(drop)
                log["doppioni_uniti"] += 1
            while len(alive) > max_team:  # fase 3: fuori dal campo
                xy = players[i, alive]
                outside = np.maximum.reduce([-xy[:, 0], xy[:, 0] - length, -xy[:, 1], xy[:, 1] - width,
                                             np.zeros(len(alive))])
                if np.isfinite(ball[i]).all():
                    outside[np.argmin(np.hypot(*(xy - ball[i]).T))] = 0
                k = int(np.argmax(outside))
                if outside[k] <= outside_margin_m:
                    break
                players[i, alive[k]] = np.nan
                alive.pop(k)
                log["tolti_fuori_campo"] += 1
            if len(alive) > max_team:
                log["frame_ancora_troppi"] += 1
    return players, log


def clean_players(arrays, *, fps=25, max_team=11, player_speed_max_ms=12.0, min_track_s=1.0,
                  flip_quota=0.5, duplicate_m=1.0, outside_margin_m=1.0):
    """Restituisce (array con giocatori e squadre puliti, registro). Gli input restano intatti."""
    if arrays.pitch is None:
        raise ValueError("Dimensioni del campo assenti: servono per gli oggetti fuori campo")
    continuous = continuity(arrays.frames, arrays.segment)
    evaluated = np.asarray(arrays.segment) > 0
    players = np.asarray(arrays.players, dtype=float)
    team = np.asarray(arrays.team).astype(np.int8)
    tracks, jumps = split_tracks(players, team, continuous, speed_max_ms=player_speed_max_ms, fps=fps)
    present = np.isfinite(players).all(axis=2)
    team, log_teams = reassign_teams(present, team, tracks, evaluated, max_team=max_team,
                                     min_frames=round(min_track_s * fps), flip_quota=flip_quota)
    track_length = np.zeros(present.shape, dtype=np.int32)
    for slot, start, length in tracks:
        track_length[start:start + length, slot] = length
    players, log_crowd = merge_and_remove(players, team, np.asarray(arrays.ball, dtype=float), track_length,
                                          pitch=arrays.pitch, max_team=max_team, duplicate_m=duplicate_m,
                                          outside_margin_m=outside_margin_m)
    log = {"tracce": len(tracks), "salti_che_spezzano": jumps, **log_teams, **log_crowd}
    return replace(arrays, players=players, team=team), log


def clean_ball(ball, continuous, evaluated, *, fps=25, ball_speed_max_ms=40.0, ball_max_gap_s=1.0):
    """Restituisce (palla pulita, maschera interpolata, maschera dei punti cancellati)."""
    ball = np.asarray(ball, dtype=float).copy()
    n = len(ball)
    speed = np.full(n, np.nan)  # speed[i]: da i-1 a i
    speed[1:] = np.hypot(*(ball[1:] - ball[:-1]).T) * fps
    speed[~continuous] = np.nan
    arriving = speed
    leaving = np.r_[speed[1:], np.nan]
    removed = (arriving > ball_speed_max_ms) & ((leaving > ball_speed_max_ms) | np.isnan(leaving))
    ball[removed] = np.nan
    block = np.cumsum(~continuous)
    interpolated = np.zeros(n, dtype=bool)
    starts, lengths = _runs(~np.isfinite(ball).all(axis=1))
    for s, k in zip(starts, lengths):
        if k > ball_max_gap_s * fps or s == 0 or s + k >= n:
            continue
        if block[s - 1] != block[s + k] or not evaluated[s - 1:s + k + 1].all():
            continue
        if np.hypot(*(ball[s + k] - ball[s - 1])) / ((k + 1) / fps) > ball_speed_max_ms:
            continue
        fraction = np.arange(1, k + 1) / (k + 1)
        ball[s:s + k] = ball[s - 1] + (ball[s + k] - ball[s - 1]) * fraction[:, None]
        interpolated[s:s + k] = True
    return ball, interpolated, removed


def clean_tracking(arrays, *, fps=25, ball_speed_max_ms=40.0, ball_max_gap_s=1.0, **player_options):
    """Pulizia completa: (array puliti, maschera palla interpolata, registro)."""
    cleaned, log = clean_players(arrays, fps=fps, **player_options)
    ball, interpolated, removed = clean_ball(
        arrays.ball, continuity(arrays.frames, arrays.segment), np.asarray(arrays.segment) > 0,
        fps=fps, ball_speed_max_ms=ball_speed_max_ms, ball_max_gap_s=ball_max_gap_s)
    log.update({"palla_punti_cancellati": int(removed.sum()), "palla_frame_interpolati": int(interpolated.sum())})
    return replace(cleaned, ball=ball), interpolated, log
