"""Preparazione conservativa, senza soglie, interpolazioni o deduplicazione."""
import copy
import math


def prepare_tracking(tracking):
    """Valida l'indice e normalizza osservazioni non finite su una copia.

    Non elimina frame, non riordina, non modifica coordinate finite e non rimuove
    tracce coincidenti. Se l'indice è ambiguo, interrompe invece di correggerlo.
    Le osservazioni giocatore prive di coordinate finite non entrano nel calcolo
    delle distanze; il loro numero è registrato. Il lettore ATD le esclude già.
    """
    if tracking.empty:
        raise ValueError("Nessun frame selezionato")
    frames = tracking.frame
    if frames.duplicated().any() or not frames.is_monotonic_increasing:
        raise ValueError("Frame ripetuti o non ordinati: controllare l'audit, nessuna correzione automatica")
    if any(not isinstance(f, (int, float)) or not math.isfinite(f) or int(f) != f or f < 0 for f in frames.tolist()):
        raise ValueError("Indice dei frame non valido")
    result = tracking.copy(deep=True)
    notes = {"frame_rimossi": 0, "coordinate_finite_modificate": 0,
             "palle_non_finite_normalizzate": 0, "osservazioni_giocatore_non_finite_escluse": 0}
    def normalize(point):
        if point is None:
            return None
        if len(point) != 2:
            raise ValueError("Coordinate attese: coppia x,y")
        if any(v is None or not math.isfinite(v) for v in point):
            return None
        return tuple(point)
    for index, row in tracking.iterrows():
        ball = normalize(row.ball_xy)
        if row.ball_xy is not None and ball is None:
            notes["palle_non_finite_normalizzate"] += 1
        result.at[index, "ball_xy"] = ball
        for column in ("home_xy", "away_xy"):
            points = [normalize(p) for p in row[column]]
            notes["osservazioni_giocatore_non_finite_escluse"] += sum(p is None for p in points)
            result.at[index, column] = [p for p in points if p is not None]
    return result, notes
