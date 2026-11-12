"""Feature per squadra dal tracking ad array: una riga per frame, nessun fit.

Riprende le feature del gruppo (cartella Drive, `Modello2_DAH/features`), con le
correzioni dell'audit (`docs/audit_drive/AUDIT.md`):
- niente feature legate a uno slot o alla rosa nominale: uno slot non è la
  stessa persona fra partite, e nei dati sporcati o nella challenge è anonimo;
- la squadra è letta frame per frame (`TrackingArrays.team`), non dal primo frame;
- le derivate temporali non attraversano salti di frame né cambi di `segment`;
- soglie di velocità esplicite per palla e giocatori (salti di slot → NaN);
- `x_extent`/`y_extent` al posto di `width`/`length`, che avevano i nomi invertiti.

Convenzioni: metri, secondi, `fps` frame al secondo. NaN = informazione assente.
I conteggi sono NaN quando manca la palla. `diff_*` = casa − ospiti.
`mean_x` dipende dal verso di attacco, che cambia fra i tempi e fra le partite.
"""
import math
import warnings
import numpy as np
import pandas as pd

TEAMS = (("home", 0), ("away", 1))
RADII_M = (2, 5, 10, 15)
# Opzioni predefinite, uguali per training (Metrica) e run (challenge).
ORIENTED = True    # coordinate e direzioni orientate sul verso d'attacco (spellF1 +0,05 su Metrica)
CONTEXT = True     # finestre di contesto sui segnali di possesso (+0,01, insieme alle altre leve)
CONTEXT_WINDOWS = (12, 25, 50)  # frame: ±0,5 s, ±1 s, ±2 s


def continuity(frames, segment):
    """True dove il frame segue direttamente il precedente nello stesso segmento."""
    frames, segment = np.asarray(frames), np.asarray(segment)
    continuous = np.zeros(len(frames), dtype=bool)
    continuous[1:] = (frames[1:] == frames[:-1] + 1) & (segment[1:] == segment[:-1])
    return continuous


def _masked_mean(values, mask):
    count = mask.sum(axis=1)
    total = np.where(mask, values, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _kth_smallest(values, k):
    ordered = np.sort(np.where(np.isfinite(values), values, np.inf), axis=1)
    if ordered.shape[1] < k:
        return np.full(len(values), np.nan)
    column = ordered[:, k - 1]
    return np.where(np.isfinite(column), column, np.nan)


def _displacement(ball, block, steps):
    """Distanza fra la palla ora e `steps` frame prima, se osservata in tutto il tratto."""
    n = len(ball)
    result = np.full(n, np.nan)
    if steps <= 0 or n <= steps:
        return result
    seen = np.isfinite(ball).all(axis=1).astype(int)
    cumulative = np.concatenate(([0], np.cumsum(seen)))
    i = np.arange(steps, n)
    ok = (cumulative[i + 1] - cumulative[i - steps] == steps + 1) & (block[i] == block[i - steps])
    result[i[ok]] = np.linalg.norm(ball[i[ok]] - ball[i[ok] - steps], axis=1)
    return result


def build_team_features(tracking, *, fps=25, ball_speed_max_ms=40.0, player_speed_max_ms=12.0, radii_m=RADII_M,
                        oriented=None, context=None):
    """Feature per frame da un `TrackingArrays`. Gli input non vengono modificati.

    ball_speed_max_ms: velocità oltre la quale lo spostamento della palla fra due
    frame è trattato come errore del tracker (dx, dy, velocità → NaN, flag = 1).
    Il valore predefinito di 40 m/s è fisico, fissato a priori nelle regole di
    pulizia del gruppo; non è calibrato sulla partita della challenge.
    player_speed_max_ms: oltre questa velocità uno slot ha cambiato persona fra due
    frame; la velocità di quel passo è NaN (12 m/s, a priori, come nella pulizia).
    oriented: sostituisce le grandezze che dipendono dal lato del campo (baricentri
    lungo x, direzione e spostamenti della palla) con versioni orientate sul verso
    d'attacco, uguali nei due tempi e in tutte le partite (`orient_features`).
    context: aggiunge le finestre di contesto sul possesso (`add_context_features`).
    None = valori predefiniti del modulo (ORIENTED, CONTEXT).
    """
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError("fps deve essere positivo")
    if ball_speed_max_ms <= 0 or player_speed_max_ms <= 0:
        raise ValueError("Le soglie di velocità devono essere positive")
    players = np.asarray(tracking.players, dtype=float)
    ball = np.asarray(tracking.ball, dtype=float)
    team = np.asarray(tracking.team)
    n = len(ball)
    continuous = continuity(tracking.frames, tracking.segment)
    block = np.cumsum(~continuous)

    out = {"frame": np.asarray(tracking.frames)}
    ball_seen = np.isfinite(ball).all(axis=1)
    out["ball_available"] = ball_seen.astype(int)
    player_seen = np.isfinite(players).all(axis=2)
    distance = np.linalg.norm(players - ball[:, None, :], axis=2)
    distance[~(player_seen & ball_seen[:, None])] = np.nan

    step = np.full(players.shape[:2], np.nan)
    step[1:] = np.linalg.norm(players[1:] - players[:-1], axis=2) * fps
    step[~continuous] = np.nan
    step[step > player_speed_max_ms] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # righe interamente NaN
        for name, code in TEAMS:
            mine = team == code
            present = player_seen & mine
            available = present.sum(axis=1)
            near = np.where(mine, distance, np.nan)
            out[f"{name}_players_available"] = available
            out[f"{name}_min_dist_ball"] = _kth_smallest(near, 1)
            out[f"{name}_second_dist_ball"] = _kth_smallest(near, 2)
            out[f"{name}_third_dist_ball"] = _kth_smallest(near, 3)
            out[f"{name}_mean_dist_ball"] = _masked_mean(near, np.isfinite(near))
            for radius in radii_m:
                count = np.where(ball_seen, (near <= radius).sum(axis=1), np.nan)
                out[f"{name}_players_{radius}m"] = count
                out[f"{name}_players_{radius}m_pct"] = np.where(available > 0, count / np.maximum(available, 1), np.nan)

            speed = np.where(mine, step, np.nan)
            out[f"{name}_players_speed_available"] = np.isfinite(speed).sum(axis=1)
            out[f"{name}_avg_speed"] = _masked_mean(speed, np.isfinite(speed))
            ranked = np.where(np.isfinite(near), near, np.inf)
            nearest = np.argmin(ranked, axis=1) if ranked.shape[1] else np.zeros(n, dtype=int)
            has_nearest = np.isfinite(ranked).any(axis=1)
            nearest_speed = speed[np.arange(n), nearest] if ranked.shape[1] else np.full(n, np.nan)
            out[f"{name}_nearest_player_speed"] = np.where(has_nearest, nearest_speed, np.nan)

            previous = np.vstack([present[:1], present[:-1]])
            previous[~continuous] = present[~continuous]  # inizio tratto: nessun confronto
            out[f"{name}_players_newly_available"] = (present & ~previous).sum(axis=1)
            out[f"{name}_players_newly_missing"] = (~present & previous).sum(axis=1)

            x = np.where(present, players[..., 0], np.nan)
            y = np.where(present, players[..., 1], np.nan)
            cx, cy = _masked_mean(x, present), _masked_mean(y, present)
            has = available > 0
            out[f"{name}_mean_x"] = cx
            out[f"{name}_x_extent"] = np.where(has, np.nanmax(x, axis=1) - np.nanmin(x, axis=1), np.nan)
            out[f"{name}_y_extent"] = np.where(has, np.nanmax(y, axis=1) - np.nanmin(y, axis=1), np.nan)
            out[f"{name}_avg_dispersion"] = _masked_mean(np.hypot(x - cx[:, None], y - cy[:, None]), present)
            out[f"{name}_centroid_dist_ball"] = np.where(
                has & ball_seen, np.hypot(cx - ball[:, 0], cy - ball[:, 1]), np.nan)

    dx, dy = np.full(n, np.nan), np.full(n, np.nan)
    dx[1:], dy[1:] = ball[1:, 0] - ball[:-1, 0], ball[1:, 1] - ball[:-1, 1]
    dx[~continuous] = dy[~continuous] = np.nan
    ball_speed = np.hypot(dx, dy) * fps
    anomaly = np.isfinite(ball_speed) & (ball_speed > ball_speed_max_ms)
    dx[anomaly] = dy[anomaly] = ball_speed[anomaly] = np.nan
    out["ball_dx"], out["ball_dy"], out["ball_speed"] = dx, dy, ball_speed
    out["ball_tracking_anomaly"] = anomaly.astype(int)
    acceleration = np.full(n, np.nan)
    acceleration[1:] = (ball_speed[1:] - ball_speed[:-1]) * fps
    out["ball_acceleration"] = acceleration
    moving = np.isfinite(dx) & np.isfinite(dy) & ((dx != 0) | (dy != 0))
    direction = np.where(moving, np.arctan2(dy, dx), np.nan)
    out["ball_direction"] = direction
    change = np.full(n, np.nan)
    delta = direction[1:] - direction[:-1]
    change[1:] = np.arctan2(np.sin(delta), np.cos(delta))
    out["ball_direction_change"] = change
    for label, seconds in (("0_5s", 0.5), ("1s", 1.0)):
        out[f"ball_displacement_{label}"] = _displacement(ball, block, int(math.floor(seconds * fps + 0.5)))

    run = np.zeros(n, dtype=int)
    started = np.zeros(n, dtype=int)
    for i in range(n):
        if not ball_seen[i]:
            fresh = not continuous[i] or ball_seen[i - 1]
            started[i] = int(fresh)
            run[i] = 1 if fresh else run[i - 1] + 1
    out["ball_missing_run_length"] = run
    out["ball_gap_started"] = started

    for feature in (["min_dist_ball", "second_dist_ball", "mean_dist_ball", "centroid_dist_ball",
                     "avg_speed", "nearest_player_speed", "x_extent", "y_extent", "mean_x"]
                    + [f"players_{r}m" for r in radii_m] + [f"players_{r}m_pct" for r in radii_m]):
        out[f"diff_{feature}"] = out[f"home_{feature}"] - out[f"away_{feature}"]
    features = pd.DataFrame(out)
    if ORIENTED if oriented is None else oriented:
        features = orient_features(features, ball, block, tracking.pitch)
    if CONTEXT if context is None else context:
        features = add_context_features(features, continuous, fps)
    return features


def orient_features(features, ball, block, pitch):
    """Grandezze indipendenti dal lato del campo.

    In ogni tratto continuo (un tempo) il verso è il segno mediano di
    away_mean_x − home_mean_x: +1 se gli ospiti stanno a destra. Il cambio di
    campo ruota il campo di 180°, quindi si capovolgono sia x sia y. Dopo la
    trasformazione x positiva = verso la porta degli ospiti, in ogni tempo e partita.
    """
    f = features.copy()
    length, width = pitch if pitch is not None else (105.0, 68.0)
    gap = (f["away_mean_x"] - f["home_mean_x"]).to_numpy()
    orient = np.full(len(f), np.nan)
    for b in np.unique(block):
        part = block == b
        values = gap[part][np.isfinite(gap[part])]
        if len(values) and np.median(values) != 0:
            orient[part] = np.sign(np.median(values))
    for name in ("home", "away"):
        f[f"{name}_mean_x"] = (f[f"{name}_mean_x"] - length / 2) * orient
    f["diff_mean_x"] = f["home_mean_x"] - f["away_mean_x"]
    f["ball_dx"], f["ball_dy"] = f["ball_dx"] * orient, f["ball_dy"] * orient
    moving = np.isfinite(f["ball_dx"]) & np.isfinite(f["ball_dy"]) & ((f["ball_dx"] != 0) | (f["ball_dy"] != 0))
    f["ball_direction"] = np.where(moving, np.arctan2(f["ball_dy"], f["ball_dx"]), np.nan)
    f["ball_x_oriented"] = (ball[:, 0] - length / 2) * orient
    f["ball_y_oriented"] = (ball[:, 1] - width / 2) * orient
    return f


def add_context_features(features, continuous, fps=25, windows=CONTEXT_WINDOWS):
    """Finestre di contesto sui segnali di possesso, senza uscire dal tratto continuo.

    - quota di frame in cui la casa ha il giocatore più vicino alla palla, media
      di diff_min_dist_ball e di diff_players_5m su ±w frame (offline: anche il futuro);
    - la stessa quota solo nel secondo precedente e solo in quello successivo;
    - da quanti secondi la stessa squadra ha il giocatore più vicino.
    Le medie ignorano i NaN (palla assente): NaN solo se tutta la finestra è vuota.
    """
    from src.dead_ball_features import window_mean
    f = features.copy()
    continuous = np.asarray(continuous, dtype=bool)
    home, away = f["home_min_dist_ball"].to_numpy(), f["away_min_dist_ball"].to_numpy()
    known = np.isfinite(home) & np.isfinite(away)
    nearest_home = np.where(known, np.where(home < away, 1.0, np.where(home > away, 0.0, 0.5)), np.nan)
    signals = {"nearest_home": nearest_home, "diff_min_dist": f["diff_min_dist_ball"].to_numpy(),
               "diff_players_5m": f["diff_players_5m"].to_numpy()}
    for w in windows:
        for name, values in signals.items():
            f[f"ctx_{name}_c{w}"] = window_mean(values, continuous, w, w)
    second = int(round(fps))
    f["ctx_nearest_home_past1s"] = window_mean(nearest_home, continuous, second, 0)
    f["ctx_nearest_home_future1s"] = window_mean(nearest_home, continuous, 0, second)
    streak = np.zeros(len(f))
    for i in range(len(f)):
        if not known[i] or nearest_home[i] == 0.5:
            continue
        same = i > 0 and continuous[i] and known[i - 1] and nearest_home[i - 1] == nearest_home[i]
        streak[i] = streak[i - 1] + 1 if same else 1
    f["ctx_nearest_streak_s"] = np.where(known, streak / fps, np.nan)
    return f


def feature_columns(features):
    """Colonne da passare a un modello: tutto tranne l'indice dei frame."""
    return [column for column in features.columns if column != "frame"]
