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
import os
import warnings
import numpy as np
import pandas as pd

TEAMS = (("home", 0), ("away", 1))
RADII_M = (2, 5, 10, 15)
# Opzioni predefinite, uguali per training (Metrica) e run (challenge).
ORIENTED = True    # coordinate e direzioni orientate sul verso d'attacco (spellF1 +0,05 su Metrica)
CONTEXT = True     # finestre di contesto sui segnali di possesso (+0,01, insieme alle altre leve)
CONTEXT_WINDOWS = (12, 25, 50)  # frame: ±0,5 s, ±1 s, ±2 s
# Feature di possesso aggiuntive (A5), a gruppi. Per gli esperimenti si sceglie quali
# usare con la variabile d'ambiente POSSESSO_FEATURES, per esempio
#   set POSSESSO_FEATURES=portieri,dietro      (Windows)
# "nessuna" = nessun gruppo. Senza variabile valgono quelle qui sotto.
EXTRA_GROUPS = ("portieri", "dietro", "packing", "linee", "corsie", "direzione")
LEVEL_M = 1.0          # "in pari" con la palla: entro ±1 m di profondità
LINE_M = 1.5           # "sulla stessa linea": compagni entro ±1,5 m di profondità
LANES = 5              # corsie verticali: due fasce, due mezzi spazi, centro
DIRECTION_WINDOW_S = 0.2   # direzione della palla dallo spostamento in 0,2 s (un frame trema troppo)
DIRECTION_MIN_MS = 3.0     # sotto questa velocità la direzione è rumore: NaN
RECEIVER_MAX_M = 40.0      # ricevitori possibili: davanti alla palla, entro 40 m lungo la traiettoria
CONE_DEG = 15.0            # cono attorno alla direzione della palla


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
                        oriented=None, context=None, extra=None):
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
    extra: gruppi di feature di possesso aggiuntive (`add_possession_extras`).
    None = valori predefiniti del modulo (ORIENTED, CONTEXT, EXTRA_GROUPS o la
    variabile d'ambiente POSSESSO_FEATURES).
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
    features = add_possession_extras(features, players, ball, team, block, tracking.pitch,
                                     extra_groups(extra), fps)
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
    orient = attack_orientation(f, block)
    for name in ("home", "away"):
        f[f"{name}_mean_x"] = (f[f"{name}_mean_x"] - length / 2) * orient
    f["diff_mean_x"] = f["home_mean_x"] - f["away_mean_x"]
    f["ball_dx"], f["ball_dy"] = f["ball_dx"] * orient, f["ball_dy"] * orient
    moving = np.isfinite(f["ball_dx"]) & np.isfinite(f["ball_dy"]) & ((f["ball_dx"] != 0) | (f["ball_dy"] != 0))
    f["ball_direction"] = np.where(moving, np.arctan2(f["ball_dy"], f["ball_dx"]), np.nan)
    f["ball_x_oriented"] = (ball[:, 0] - length / 2) * orient
    f["ball_y_oriented"] = (ball[:, 1] - width / 2) * orient
    return f


def attack_orientation(features, block):
    """Verso per tratto continuo: +1 se gli ospiti stanno a destra (la casa difende
    la porta a x = 0), −1 al contrario, NaN se non si può dire. Usa i baricentri
    *non* orientati (`home_mean_x`, `away_mean_x` come escono dal calcolo)."""
    gap = (features["away_mean_x"] - features["home_mean_x"]).to_numpy()
    orient = np.full(len(features), np.nan)
    for b in np.unique(block):
        part = block == b
        values = gap[part][np.isfinite(gap[part])]
        if len(values) and np.median(values) != 0:
            orient[part] = np.sign(np.median(values))
    return orient


def extra_groups(extra=None):
    """Gruppi A5 da calcolare: `extra` se dato, altrimenti POSSESSO_FEATURES, altrimenti EXTRA_GROUPS."""
    if extra is None:
        text = os.environ.get("POSSESSO_FEATURES")
        if text is None:
            return tuple(EXTRA_GROUPS)
        extra = [] if text.strip().lower() in ("", "nessuna") else [g.strip().lower() for g in text.split(",")]
    unknown = set(extra) - set(EXTRA_GROUPS)
    if unknown:
        raise ValueError(f"Gruppi di feature sconosciuti: {sorted(unknown)}; ammessi: {EXTRA_GROUPS}")
    return tuple(extra)


def add_possession_extras(features, players, ball, team, block, pitch, groups, fps=25):
    """Feature di possesso "da allenatore" (lavoro A5), a gruppi.

    Tutte usano la *propria porta* di ogni squadra, ricavata dal verso di
    `attack_orientation`: significano la stessa cosa nei due tempi. Le profondità
    sono metri dalla propria linea di porta (0 = sulla linea, cresce verso l'avversario).

    - portieri: altezza del giocatore più arretrato di ogni squadra (quasi sempre il
      portiere). Chi attacca alza il portiere. Non usa la palla.
    - dietro: giocatori di ogni squadra dietro la palla (fra palla e propria porta),
      in pari (±1 m) e davanti. Chi difende scherma: tanti dietro. NaN senza palla.
    - packing: come cambia, nel secondo prima e in quello dopo, il numero di
      giocatori dietro la palla. Se la palla supera gli avversari, i loro "dietro"
      calano: chi li fa calare ha il possesso. NaN senza palla.
    - linee: quanti compagni ha in media ogni giocatore sulla sua stessa linea
      (±1,5 m di profondità). Chi difende si dispone in linee piatte; chi costruisce
      si sfalsa. Vale soprattutto in costruzione: il contesto (posizione e velocità
      della palla) lo danno le altre feature. Non usa la palla.
    - corsie: quante delle 5 corsie verticali occupa ogni squadra. Chi attacca
      allarga e le occupa tutte. Non usa la palla.
    - direzione: verso chi va la palla. Per ogni squadra, quanto è vicino alla
      traiettoria il suo giocatore meglio piazzato per riceverla (davanti alla
      palla, entro 40 m), e quanti giocatori stanno in un cono di ±15° attorno alla
      direzione. NaN se la palla non si vede o va sotto 3 m/s.
    """
    f = features.copy()
    if not groups:
        return f
    length, width = pitch if pitch is not None else (105.0, 68.0)
    orient = attack_orientation(f, block)
    players = np.asarray(players, dtype=float)
    x, y = players[..., 0], players[..., 1]
    present = np.isfinite(players).all(axis=2)
    team = np.asarray(team)
    ball = np.asarray(ball, dtype=float)
    ball_seen = np.isfinite(ball).all(axis=1)
    # Casa: porta a x = 0 se orient = +1, a x = length se orient = −1. Ospiti al contrario.
    home_goal = np.where(orient > 0, 0.0, np.where(orient < 0, length, np.nan))
    goals = {"home": home_goal, "away": length - home_goal}
    known = np.isfinite(home_goal)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # righe interamente NaN
        for name, code in TEAMS:
            mine = present & (team == code)
            goal = goals[name]
            depth = np.where(mine, np.abs(x - goal[:, None]), np.nan)  # giocatore → propria porta
            ball_depth = np.abs(ball[:, 0] - goal)                       # palla → stessa porta
            if "portieri" in groups:
                f[f"{name}_keeper_depth"] = np.nanmin(depth, axis=1)
            if "dietro" in groups or "packing" in groups:
                gap = depth - ball_depth[:, None]  # < 0: fra la palla e la propria porta
                ok = ball_seen & known
                behind = np.where(ok, (mine & (gap < -LEVEL_M)).sum(axis=1), np.nan)
                f[f"{name}_behind_ball"] = behind
                if "dietro" in groups:
                    f[f"{name}_level_ball"] = np.where(ok, (mine & (np.abs(gap) <= LEVEL_M)).sum(axis=1), np.nan)
                    f[f"{name}_ahead_ball"] = np.where(ok, (mine & (gap > LEVEL_M)).sum(axis=1), np.nan)
            if "linee" in groups:
                mates = np.zeros(depth.shape)
                for a in range(0, len(depth), 5000):  # a blocchi: tutte le coppie occupano memoria
                    d = depth[a:a + 5000]
                    dx = np.abs(d[:, :, None] - d[:, None, :])          # tutte le coppie di compagni
                    mates[a:a + 5000] = (dx <= LINE_M).sum(axis=2) - 1  # NaN → False; meno se stesso
                f[f"{name}_line_mates"] = np.where(mine.any(axis=1), _masked_mean(mates, mine), np.nan)
            if "corsie" in groups:
                lane = np.clip(np.floor(y / width * LANES), 0, LANES - 1)
                occupied = np.zeros(len(f))
                for k in range(LANES):
                    occupied += (mine & (lane == k)).any(axis=1)
                f[f"{name}_lanes"] = np.where(mine.any(axis=1), occupied, np.nan)

    if "direzione" in groups:
        f = _direction_features(f, players, present, team, ball, block, fps)

    if "packing" in groups:
        from src.dead_ball_features import window_mean
        continuous = np.r_[False, block[1:] == block[:-1]]
        second = int(round(fps))
        for name in ("home", "away"):
            values = f[f"{name}_behind_ball"].to_numpy(dtype=float)
            past = window_mean(values, continuous, second, 0)
            future = window_mean(values, continuous, 0, second)
            trend = future - past
            f[f"{name}_behind_ball_trend_1s"] = np.where(np.isfinite(values), trend, np.nan)
        if "dietro" not in groups:
            f = f.drop(columns=["home_behind_ball", "away_behind_ball"])
    for column in ("keeper_depth", "behind_ball", "level_ball", "ahead_ball", "behind_ball_trend_1s",
                   "line_mates", "lanes", "receiver_dist", "in_cone"):
        if f"home_{column}" in f and f"away_{column}" in f:
            f[f"diff_{column}"] = f[f"home_{column}"] - f[f"away_{column}"]
    return f


def _direction_features(f, players, present, team, ball, block, fps):
    """Gruppo "direzione": verso chi va la palla (vedi `add_possession_extras`)."""
    n = len(ball)
    steps = max(1, int(round(DIRECTION_WINDOW_S * fps)))
    # 1. Velocità della palla dallo spostamento in `steps` frame, solo se la palla si
    #    vede all'inizio e alla fine e il tratto è continuo.
    velocity = np.full((n, 2), np.nan)
    if n > steps:
        same_block = block[steps:] == block[:-steps]
        velocity[steps:] = np.where(same_block[:, None], (ball[steps:] - ball[:-steps]) * fps / steps, np.nan)
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    moving = np.isfinite(speed) & (speed >= DIRECTION_MIN_MS)
    unit = velocity / np.where(moving, speed, np.nan)[:, None]  # direzione, lunga 1
    # 2. Per ogni giocatore: quanto è avanti lungo la traiettoria (prodotto scalare)
    #    e quanto è lontano di lato (prodotto vettoriale), tutti i frame insieme.
    rel = players - ball[:, None, :]
    along = rel[..., 0] * unit[:, None, 0] + rel[..., 1] * unit[:, None, 1]
    side = np.abs(rel[..., 0] * unit[:, None, 1] - rel[..., 1] * unit[:, None, 0])
    ahead = present & (along > 0) & (along <= RECEIVER_MAX_M)
    in_cone = ahead & (np.degrees(np.arctan2(side, along)) <= CONE_DEG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, code in TEAMS:
            mine = team == code
            best = np.nanmin(np.where(ahead & mine, side, np.nan), axis=1)
            # Nessun compagno davanti: il ricevitore è "lontanissimo", non sconosciuto.
            f[f"{name}_receiver_dist"] = np.where(moving, np.where(np.isfinite(best), best, RECEIVER_MAX_M), np.nan)
            f[f"{name}_in_cone"] = np.where(moving, (in_cone & mine).sum(axis=1), np.nan)
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
