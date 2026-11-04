"""Audit del numero di tracce e filtro sperimentale, non deduplicazione."""
import math
import pandas as pd


def player_counts(tracking):
    """Numero di osservazioni disponibili, non persone identificate."""
    rows = []
    for row in tracking.itertuples(index=False):
        home, away = len(row.home_xy), len(row.away_xy)
        rows.append({"frame": row.frame, "home": home, "away": away,
                     "total": home + away, "over_22": home + away > 22,
                     "home_over_11": home > 11, "away_over_11": away > 11})
    return pd.DataFrame(rows, columns=["frame", "home", "away", "total", "over_22", "home_over_11", "away_over_11"])


def filter_players(tracking, policy="none"):
    """Solo nearest22: conserva al massimo 22 tracce più vicine alla palla reale.

    Nessun uso di palla interpolata, nessun riempimento se ci sono meno tracce.
    Se la palla è assente non modifica il frame. La selezione è globale, non
    garantisce 11 tracce per squadra e non riconosce duplicati di persone.
    Gli originali e l'ordine relativo delle tracce conservate restano intatti.
    """
    if policy not in {"none", "nearest22"}:
        raise ValueError("player_policy deve essere none oppure nearest22")
    result = tracking.copy(deep=True)
    log = {"policy": policy, "frames_changed": 0, "observations_removed": 0,
           "overfull_frames_skipped_missing_ball": 0}
    if policy == "none":
        return result, log
    for index, row in tracking.iterrows():
        total = len(row.home_xy) + len(row.away_xy)
        if total <= 22:
            continue
        ball = row.ball_xy
        if ball is None or any(not math.isfinite(value) for value in ball):
            log["overfull_frames_skipped_missing_ball"] += 1
            continue
        candidates = [(math.dist(point, ball), column, i)
                      for column in ("home_xy", "away_xy")
                      for i, point in enumerate(row[column])]
        kept = {(column, i) for _, column, i in sorted(candidates, key=lambda item: item[0])[:22]}
        for column in ("home_xy", "away_xy"):
            result.at[index, column] = [point for i, point in enumerate(row[column]) if (column, i) in kept]
        log["frames_changed"] += 1
        log["observations_removed"] += total - 22
    return result, log
