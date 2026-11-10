"""Reimplementazione vettoriale delle 100 feature dei colleghi (Modello2_DAH/features).

Scopo: (1) verificare che riproduca i CSV, (2) applicarla ai dati sporcati.
team: array (n, S) con la squadra di ogni slot in ogni frame (0/1).
"""
import warnings
import numpy as np
import pandas as pd

FPS = 25
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _nanmin(a):
    out = np.full(a.shape[0], np.nan)
    ok = np.isfinite(a).any(1)
    out[ok] = np.nanmin(a[ok], 1)
    return out


def _kth(a, k):
    s = np.sort(np.where(np.isfinite(a), a, np.inf), axis=1)
    out = s[:, k - 1] if s.shape[1] >= k else np.full(a.shape[0], np.inf)
    return np.where(np.isfinite(out), out, np.nan)


def _speed(pos):
    sp = np.full(pos.shape[:-1], np.nan)
    d = np.linalg.norm(pos[1:] - pos[:-1], axis=-1) * FPS
    sp[1:] = d
    return sp


def _temporal(coord, secondi):
    passi = round(secondi * FPS)
    n = len(coord)
    out = np.full(n, np.nan)
    ok = np.isfinite(coord).all(1).astype(int)
    c = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(passi, n)
    full = (c[i + 1] - c[i - passi]) == passi + 1
    out[i[full]] = np.linalg.norm(coord[i[full]] - coord[i[full] - passi], axis=1)
    return out


def build(gioc, palla, team, n_slot_team=None, slot_cols=28):
    """gioc (n,S,2), palla (n,2), team (n,S). n_slot_team: slot nominali per squadra."""
    n, S, _ = gioc.shape
    f = {}
    pal_ok = np.isfinite(palla).all(1)
    f["ball_available"] = pal_ok.astype(int)
    pos_ok = np.isfinite(gioc).all(2)
    dist = np.linalg.norm(gioc - palla[:, None, :], axis=2)
    dist[~(pos_ok & pal_ok[:, None])] = np.nan
    sp = _speed(gioc)
    for k in (0, 1):
        m = team == k
        dk = np.where(m, dist, np.nan)
        avail = (pos_ok & m).sum(1)
        f[f"team_{k}_players_available"] = avail
        f[f"team_{k}_min_dist_ball"] = _nanmin(dk)
        cnt = np.isfinite(dk).sum(1)
        f[f"team_{k}_mean_dist_ball"] = np.where(cnt > 0, np.nansum(dk, 1) / np.maximum(cnt, 1), np.nan)
        for r in (5, 10):
            f[f"team_{k}_players_{r}m"] = np.where(pal_ok, (dk <= r).sum(1), np.nan)
        spk = np.where(m, sp, np.nan)
        sa = np.isfinite(spk).sum(1)
        f[f"team_{k}_players_speed_available"] = sa
        f[f"team_{k}_avg_speed"] = np.where(sa > 0, np.nansum(spk, 1) / np.maximum(sa, 1), np.nan)
        dd = np.where(np.isfinite(dk), dk, np.inf)
        idx = np.argmin(dd, 1)
        has = np.isfinite(dd).any(1)
        ns = spk[np.arange(n), idx]
        f[f"team_{k}_nearest_player_speed"] = np.where(has, ns, np.nan)
    # palla
    dx = np.full(n, np.nan); dy = np.full(n, np.nan)
    v = pal_ok[1:] & pal_ok[:-1]
    ddx = palla[1:, 0] - palla[:-1, 0]; ddy = palla[1:, 1] - palla[:-1, 1]
    dx[1:][v] = ddx[v]; dy[1:][v] = ddy[v]
    bs = np.sqrt(dx ** 2 + dy ** 2) * FPS
    an = np.isfinite(bs) & (bs > 35)
    dx[an] = dy[an] = bs[an] = np.nan
    f["ball_dx"], f["ball_dy"], f["ball_speed"] = dx, dy, bs
    f["ball_tracking_anomaly"] = an.astype(int)
    # differenze
    for a in ("min_dist_ball", "mean_dist_ball", "players_5m", "players_10m"):
        f[f"diff_{a}"] = f[f"team_0_{a}"] - f[f"team_1_{a}"]
    for k in (0, 1):
        av = f[f"team_{k}_players_available"]
        for r in (5, 10):
            f[f"team_{k}_players_{r}m_pct"] = np.where(av > 0, f[f"team_{k}_players_{r}m"] / np.maximum(av, 1), np.nan)
    for r in (5, 10):
        f[f"diff_players_{r}m_pct"] = f[f"team_0_players_{r}m_pct"] - f[f"team_1_players_{r}m_pct"]
    f["diff_avg_speed"] = f["team_0_avg_speed"] - f["team_1_avg_speed"]
    f["diff_nearest_player_speed"] = f["team_0_nearest_player_speed"] - f["team_1_nearest_player_speed"]
    # tracking incompleto
    run = np.zeros(n, int); started = np.zeros(n, int)
    for i in range(n):
        if not pal_ok[i]:
            started[i] = int(i == 0 or pal_ok[i - 1])
            run[i] = 1 if i == 0 else run[i - 1] + 1
    f["ball_missing_run_length"], f["ball_gap_started"] = run, started
    for s in range(slot_cols):
        f[f"slot_{s}_available"] = pos_ok[:, s].astype(int) if s < S else np.zeros(n, int)
    for k in (0, 1):
        m = team == k
        cur = pos_ok & m
        prev = np.vstack([cur[:1], cur[:-1]])
        nominal = n_slot_team[k] if n_slot_team else m.sum(1)
        av = f[f"team_{k}_players_available"]
        f[f"team_{k}_players_missing"] = nominal - av
        f[f"team_{k}_players_available_pct"] = av / nominal
        f[f"team_{k}_players_newly_available"] = (cur & ~prev).sum(1)
        f[f"team_{k}_players_newly_missing"] = (~cur & prev).sum(1)
    # nuove 29
    for k in (0, 1):
        dk = np.where(team == k, dist, np.nan)
        f[f"team_{k}_second_dist_ball"] = _kth(dk, 2)
        f[f"team_{k}_third_dist_ball"] = _kth(dk, 3)
        for r in (2, 15):
            f[f"team_{k}_players_{r}m"] = np.where(pal_ok, (dk <= r).sum(1), np.nan).astype(float)
    f["diff_second_dist_ball"] = f["team_0_second_dist_ball"] - f["team_1_second_dist_ball"]
    f["diff_players_2m"] = f["team_0_players_2m"] - f["team_1_players_2m"]
    acc = np.full(n, np.nan); acc[1:] = (bs[1:] - bs[:-1]) * FPS
    f["ball_acceleration"] = acc
    dirn = np.where(np.isfinite(dx) & np.isfinite(dy) & ((dx != 0) | (dy != 0)), np.arctan2(dy, dx), np.nan)
    f["ball_direction"] = dirn
    ch = np.full(n, np.nan); de = dirn[1:] - dirn[:-1]; ch[1:] = np.arctan2(np.sin(de), np.cos(de))
    f["ball_direction_change"] = ch
    f["ball_distance_0_5s"] = _temporal(palla, 0.5)
    f["ball_distance_1s"] = _temporal(palla, 1.0)
    for k in (0, 1):
        m = (team == k) & pos_ok
        x = np.where(m, gioc[..., 0], np.nan); y = np.where(m, gioc[..., 1], np.nan)
        c = m.sum(1); has = c > 0
        cx = np.where(has, np.nansum(x, 1) / np.maximum(c, 1), np.nan)
        cy = np.where(has, np.nansum(y, 1) / np.maximum(c, 1), np.nan)
        disp = np.sqrt((x - cx[:, None]) ** 2 + (y - cy[:, None]) ** 2)
        f[f"team_{k}_centroid_dist_ball"] = np.where(pal_ok & has, np.hypot(cx - palla[:, 0], cy - palla[:, 1]), np.nan)
        f[f"team_{k}_avg_dispersion"] = np.where(has, np.nansum(disp, 1) / np.maximum(c, 1), np.nan)
        f[f"team_{k}_width"] = np.where(has, np.nanmax(x, 1) - np.nanmin(x, 1), np.nan)
        f[f"team_{k}_length"] = np.where(has, np.nanmax(y, 1) - np.nanmin(y, 1), np.nan)
        f[f"team_{k}_mean_x"] = cx
    for a in ("centroid_dist_ball", "width", "length", "mean_x"):
        f[f"diff_{a if a != 'width' and a != 'length' and a != 'mean_x' else 'team_' + a}"] = f[f"team_0_{a}"] - f[f"team_1_{a}"]
    return pd.DataFrame(f)
