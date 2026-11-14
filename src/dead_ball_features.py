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

`ball_position_features` aggiunge la palla quando si vede (fuori dalle linee, ferma,
ferma su un punto di ripresa) senza mai segnalare la sua assenza;
`build_dead_features` mette insieme le due tabelle per il classificatore.
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


PENALTY_SPOT_M = 11.0
BALL_STILL_MS = 0.5        # sotto questa velocità la palla è "ferma"
BALL_SPEED_WINDOW_S = 0.4  # velocità media su questo intervallo: smorza il rumore del tracking
BALL_STILL_MIN_S = 1.0     # ferma da almeno questo tempo
BALL_STILL_CAP_S = 5.0
SPOT_RADIUS_M = 2.0        # dischetto
CENTER_RADIUS_M = 3.0      # centro del campo
CORNER_RADIUS_M = 3.0      # bandierina
HOLDER_RADIUS_M = 1.0      # un giocatore "sulla palla" (es. portiere che la tiene)
FREE_SPACE_RADIUS_M = 9.0  # nelle punizioni gli avversari stanno a 9,15 m
THROW_IN_LINE_M = 2.0
THROW_IN_PLAYER_M = 3.0
GOAL_HALF_WIDTH_M = 3.66
KEEPER_LINE_M = 1.5           # portiere "sulla linea": entro 1,5 m dalla linea di porta
PENALTY_SETUP_SPOT_M = 1.0    # palla entro 1 m dal dischetto
PENALTY_SETUP_MAX_IN_AREA = 3 # portiere + rigorista + un margine
BALL_OUT_MARGIN_M = 0.5       # "fuori" solo se oltre la linea di almeno mezzo metro (rumore)
OUT_MEMORY_CAP_S = 30.0
RESTART_SIDELINE_M = 1.5      # ricomparsa sulla linea laterale (rimessa)
RESTART_LOOKAHEAD_S = 20.0    # buco massimo prima di una ripresa


def _still_run_seconds(still, fps):
    """Per ogni frame, da quanti secondi `still` è vero senza interruzioni."""
    run = np.zeros(len(still))
    count = 0
    for i, s in enumerate(still):
        count = count + 1 if s else 0
        run[i] = count
    return run / fps


def ball_position_features(tracking, *, fps=25):
    """Feature della palla per il gioco fermo, una riga per frame.

    Ogni feature è un "indizio di palla morta": 0 = nessun indizio. Quando la palla
    non si vede valgono 0, mai NaN: così "palla assente" ha lo stesso aspetto di un
    normale frame di gioco vivo e il classificatore non può usarla come scorciatoia
    (nelle etichette Metrica D coincide quasi sempre con la palla assente).
    Nessun riempimento con l'ultimo valore visto: una palla che sparisce in un
    contrasto sembrerebbe ferma e creerebbe palle morte false.

    - ball_out_m: metri oltre la linea più vicina (laterale o di fondo);
    - ball_still_s: da quanti secondi la palla è ferma (fino a 5);
    - ball_still_center / _penalty_spot / _corner: ferma da almeno 1 s su un punto
      di ripresa (1 sul punto, scende a 0 al bordo del raggio);
    - ball_still_goal_area: ferma in area piccola (rinvio dal fondo);
    - ball_still_penalty_area: ferma in area di rigore, fuori dall'area piccola
      (può essere anche il portiere che la tiene a gioco vivo);
    - ball_still_open_field: ferma in campo, fuori dalle aree (punizione);
    - ball_still_with_holder: ferma con un giocatore a meno di 1 m (portiere che la
      tiene, giocatore che la sistema): da sola non decide, si combina con le altre;
    - ball_still_free_space: ferma con al massimo due giocatori entro 9 m (barriera
      a distanza, avversari lontani);
    - throw_in_setup: palla vicino alla linea laterale con un giocatore oltre la linea;
    - penalty_setup: palla ferma sul dischetto, un giocatore sulla linea di porta
      (il portiere) e al massimo tre giocatori in quell'area: il rigore schierato;
    - ball_out_memory_s: secondi da quando la palla è stata vista fuori dal campo,
      finché non ricompare in campo (anche nei frame senza palla);
    - ball_restart_ahead: la palla, dopo un buco, ricompare su un punto di ripresa
      (anche nei frame senza palla, prima della ricomparsa).
    Le ultime due sono le uniche che possono valere più di 0 senza palla: lo fanno
    solo se la palla vista prima (fuori) o dopo (su un punto di ripresa) dice "gioco
    fermo". Un buco in un contrasto, con la palla vista in campo prima e dopo, resta a 0.
    Misure dal campo di `tracking.pitch`.
    """
    if tracking.pitch is None:
        raise ValueError("Dimensioni del campo assenti")
    length, width = tracking.pitch
    ball = np.asarray(tracking.ball, dtype=float)
    players = np.asarray(tracking.players, dtype=float)
    n = len(ball)
    seen = np.isfinite(ball).all(axis=1)
    bx, by = ball[:, 0], ball[:, 1]
    continuous = continuity(tracking.frames, tracking.segment)

    # Velocità media su una finestra breve, solo se la palla si vede in tutta la finestra
    # e la finestra non attraversa salti di frame o cambi di tempo.
    steps = max(1, int(round(BALL_SPEED_WINDOW_S * fps)))
    block = np.cumsum(~continuous)
    speed = np.full(n, np.nan)
    if n > steps:
        seen_count = np.concatenate(([0], np.cumsum(seen)))
        i = np.arange(steps, n)
        ok = (seen_count[i + 1] - seen_count[i - steps] == steps + 1) & (block[i] == block[i - steps])
        moved = np.hypot(bx[i] - bx[i - steps], by[i] - by[i - steps])
        speed[i] = np.where(ok, moved / (steps / fps), np.nan)
    still = np.isfinite(speed) & (speed < BALL_STILL_MS)
    still_s = _still_run_seconds(still, fps)
    settled = still_s >= BALL_STILL_MIN_S

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        beyond = np.fmax(np.fmax(-bx, bx - length), np.fmax(-by, by - width))
        central = np.abs(by - width / 2)
        goal_line = np.fmin(bx, length - bx)
        in_pitch = (bx >= 0) & (bx <= length) & (by >= 0) & (by <= width)
        in_penalty = in_pitch & (goal_line <= PENALTY_DEPTH_M) & (central <= PENALTY_HALF_WIDTH_M)
        in_goal_area = in_pitch & (goal_line <= GOAL_AREA_DEPTH_M) & (central <= GOAL_AREA_HALF_WIDTH_M)
        d_center = np.hypot(bx - length / 2, by - width / 2)
        d_spot = np.hypot(goal_line - PENALTY_SPOT_M, by - width / 2)
        d_corner = np.hypot(np.fmin(bx, length - bx), np.fmin(by, width - by))
        to_ball = np.linalg.norm(players - ball[:, None], axis=2)
        to_ball = np.where(np.isfinite(to_ball), to_ball, np.inf)
        nearest = to_ball.min(axis=1) if players.shape[1] else np.full(n, np.inf)
        crowd = (to_ball <= FREE_SPACE_RADIUS_M).sum(axis=1)
        # Rigore: palla ferma sul dischetto, un giocatore sulla linea di porta (il portiere)
        # e area quasi vuota (portiere, rigorista, al massimo un altro; gli altri fuori).
        goal_x = np.where(bx < length / 2, 0.0, length)  # la porta dal lato della palla
        px, py = players[..., 0], players[..., 1]
        on_goal_line = (np.abs(px - goal_x[:, None]) <= KEEPER_LINE_M) & \
                       (np.abs(py - width / 2) <= GOAL_HALF_WIDTH_M + KEEPER_LINE_M)
        keeper_on_line = np.nan_to_num(on_goal_line, nan=False).any(axis=1)
        in_that_area = (np.abs(px - goal_x[:, None]) <= PENALTY_DEPTH_M) & \
                       (np.abs(py - width / 2) <= PENALTY_HALF_WIDTH_M)
        area_count = np.nan_to_num(in_that_area, nan=False).sum(axis=1)
        penalty_setup = settled & (d_spot <= PENALTY_SETUP_SPOT_M) & keeper_on_line & \
            (area_count <= PENALTY_SETUP_MAX_IN_AREA)
        player_y = players[..., 1]
        player_beyond_side = np.isfinite(player_y) & ((player_y < 0) | (player_y > width))
        thrower = (player_beyond_side & (to_ball <= THROW_IN_PLAYER_M)).any(axis=1)
        near_sideline = np.fmin(by, width - by) <= THROW_IN_LINE_M

    def closeness(distance, radius):
        return np.where(settled, np.clip(1 - distance / radius, 0, 1), 0.0)

    out = {"frame": np.asarray(tracking.frames),
           "ball_out_m": np.clip(beyond, 0, 10),
           "ball_still_s": np.minimum(still_s, BALL_STILL_CAP_S),
           "ball_still_center": closeness(d_center, CENTER_RADIUS_M),
           "ball_still_penalty_spot": closeness(d_spot, SPOT_RADIUS_M),
           "ball_still_corner": closeness(d_corner, CORNER_RADIUS_M),
           "ball_still_goal_area": (settled & in_goal_area).astype(float),
           "ball_still_penalty_area": (settled & in_penalty & ~in_goal_area).astype(float),
           "ball_still_open_field": (settled & in_pitch & ~in_penalty).astype(float),
           "ball_still_with_holder": (settled & (nearest <= HOLDER_RADIUS_M)).astype(float),
           "ball_still_free_space": (settled & (crowd <= 2)).astype(float),
           "throw_in_setup": (seen & near_sideline & thrower).astype(float),
           "penalty_setup": penalty_setup.astype(float)}
    frame = pd.DataFrame(out)
    columns = [c for c in frame.columns if c != "frame"]
    for column in columns:  # palla assente = nessun indizio
        frame[column] = np.where(seen & np.isfinite(frame[column]), frame[column], 0.0)

    # Bordi delle pause. Queste due valgono anche quando la palla non si vede, ma solo
    # se l'ultima o la prossima volta che si vede dice "gioco fermo": un buco di palla
    # in un contrasto (palla vista in campo prima e dopo) resta a 0.
    out_seen = seen & np.nan_to_num(beyond > BALL_OUT_MARGIN_M, nan=False)
    in_seen = seen & ~out_seen
    restart_point = seen & np.nan_to_num(
        (beyond > 0) | (np.fmin(by, width - by) <= RESTART_SIDELINE_M) | (d_corner <= CORNER_RADIUS_M)
        | in_goal_area | (d_center <= CENTER_RADIUS_M) | (d_spot <= SPOT_RADIUS_M), nan=False)
    not_moving = ~(np.isfinite(speed) & (speed >= BALL_STILL_MS))
    frame["ball_out_memory_s"] = _out_memory(out_seen, in_seen, block, fps, OUT_MEMORY_CAP_S)
    frame["ball_restart_ahead"] = _restart_ahead(seen, restart_point, not_moving, block,
                                                 int(RESTART_LOOKAHEAD_S * fps))
    return frame


def _out_memory(out_seen, in_seen, block, fps, cap_s):
    """Secondi da quando la palla è stata vista fuori, finché non torna in campo.

    Vale anche nei frame senza palla: se l'ultima volta che si vedeva era fuori,
    il gioco è ancora fermo. Si azzera quando la palla ricompare in campo o
    cambia il tratto continuo. 0 = nessun indizio.
    """
    result = np.zeros(len(out_seen))
    last_out = None
    for i in range(len(out_seen)):
        if i and block[i] != block[i - 1]:
            last_out = None
        if out_seen[i]:
            last_out = i
        elif in_seen[i]:
            last_out = None
        if last_out is not None:
            result[i] = min((i - last_out + 1) / fps, cap_s)
    return result


def _restart_ahead(seen, restart_point, not_moving, block, lookahead):
    """1 nei frame che precedono una ripresa: la palla, dopo un buco di al massimo
    `lookahead` frame, ricompare su un punto di ripresa (linea laterale, bandierina,
    area piccola, centro, dischetto, fuori). Vale 1 anche finché, ricomparsa lì,
    resta ferma. Guarda nel futuro: le feature sono offline, come le finestre
    `_future_` già presenti."""
    n = len(seen)
    result = np.zeros(n)
    i = 0
    while i < n:
        if seen[i]:
            i += 1
            continue
        start = i
        while i < n and not seen[i] and block[i] == block[start]:
            i += 1
        if i < n and seen[i] and block[i] == block[start] and i - start <= lookahead and restart_point[i]:
            result[start:i] = 1.0
            j = i
            while j < n and seen[j] and not_moving[j] and restart_point[j] and block[j] == block[i]:
                result[j] = 1.0
                j += 1
    return result


def build_dead_features(tracking, *, fps=25, use_ball=True):
    """Feature del classificatore del gioco fermo: giocatori, più la palla se `use_ball`."""
    players = build_dead_ball_features(tracking, fps=fps)
    if not use_ball:
        return players
    ball = ball_position_features(tracking, fps=fps).drop(columns="frame")
    return pd.concat([players, ball], axis=1)


def dead_ball_columns(features):
    return [c for c in features.columns if c != "frame"]
