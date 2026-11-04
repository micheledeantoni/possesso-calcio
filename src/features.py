"""Caratteristiche geometriche; nessun fit né parametro da apprendere."""
import math
import pandas as pd


def build_features(tracking, ball_column="ball_xy"):
    """Distanze minime per squadra; NaN significa informazione assente."""
    def nearest(ball, players):
        if ball is None or not players:
            return float("nan")
        return min(math.hypot(p[0] - ball[0], p[1] - ball[1]) for p in players)
    if ball_column not in {"ball_xy", "ball_used_xy"}:
        raise ValueError("Colonna palla non supportata")
    if ball_column not in tracking:
        raise ValueError("Colonna palla assente")
    rows = []
    for row in tracking.itertuples(index=False):
        ball = getattr(row, ball_column)
        home = nearest(ball, row.home_xy)
        away = nearest(ball, row.away_xy)
        provenance = (getattr(row, "ball_source", "unknown") if ball_column == "ball_used_xy"
                      else ("observed" if row.ball_xy is not None else "unknown"))
        rows.append({"frame": row.frame, "ball_observed": row.ball_xy is not None, "ball_source": provenance,
                     "nearest_home_m": home, "nearest_away_m": away,
                     "away_minus_home_m": away - home})
    return pd.DataFrame(rows, columns=["frame", "ball_observed", "ball_source", "nearest_home_m",
                                      "nearest_away_m", "away_minus_home_m"])
