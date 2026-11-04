"""Modulo palla: coordinate originali, stime e coordinate operative separate."""
import pandas as pd
from src.reconstruction import reconstruct_ball
from src.quality import HALVES


def prepare_ball(tracking, method="none", max_gap_s=0.5, source="synthetic"):
    """Ricostruzione opzionale offline. Durata sperimentale, non calibrata.

    Nei dati reali non attraversa i confini dei tempi pubblicati e non ricostruisce
    prepartita/intervallo/dopopartita. Il filtro giocatori resta indipendente e usa
    soltanto la palla osservata. Nessuna stima diventa una nuova osservazione.
    """
    if method not in {"none", "linear", "velocity"}:
        raise ValueError("ball_method deve essere none, linear oppure velocity")
    working = tracking.copy(deep=True)
    if source == "real":
        working["segment"] = working.frame.map(lambda f: next(
            (i for i, (start, end) in enumerate(HALVES, 1) if start <= f <= end), 0))
    if method == "none":
        result = working.copy(deep=True)
        result["ball_estimated_xy"] = pd.Series([None] * len(result), index=result.index, dtype=object)
        result["ball_source"] = result.ball_xy.map(lambda p: "observed" if p is not None else "unknown")
        result["anchor_distance_s"] = result.ball_xy.map(lambda p: 0. if p is not None else float("nan"))
    else:
        result = reconstruct_ball(working, method=method, max_gap_s=max_gap_s)
        if source == "real":
            outside = result.segment.eq(0) & result.ball_source.ne("observed")
            result.loc[outside, "ball_estimated_xy"] = None
            result.loc[outside, "ball_source"] = "unknown"
            result.loc[outside, "anchor_distance_s"] = float("nan")
    result["ball_used_xy"] = pd.Series([
        observed if observed is not None else estimate
        for observed, estimate in zip(result.ball_xy, result.ball_estimated_xy)
    ], index=result.index, dtype=object)
    counts = result.ball_source.value_counts()
    log = {"method": method, "max_gap_s": max_gap_s if method != "none" else None,
           "observed_frames": int(counts.get("observed", 0)),
           "estimated_frames": int(result.ball_source.isin(["linear", "velocity"]).sum()),
           "unknown_frames": int(counts.get("unknown", 0))}
    return result, log
