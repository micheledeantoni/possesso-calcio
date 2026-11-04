"""Ricostruzione dimostrativa: mantiene osservazioni e stime separate."""
import math
import numpy as np
import pandas as pd


def reconstruct_ball(tracking, *, method, max_gap_s, fps=25):
    """Riempie soltanto buchi entro max_gap_s, senza cambiare ball_xy.

    method: linear (due ancore) oppure velocity (due osservazioni precedenti
    consecutive). Un buco lungo resta interamente sconosciuto. Nessuna stima
    viene riutilizzata come ancora. Nessun salto fra frame o segmenti.
    Se disponibile, segment deve distinguere tempi/interruzioni note.
    max_gap_s è obbligatorio: nella demo è illustrativo, mai calibrato sulla challenge.
    """
    if method not in {"linear", "velocity"}:
        raise ValueError("Metodo atteso: linear oppure velocity")
    if fps <= 0 or not math.isfinite(fps) or max_gap_s < 0 or not math.isfinite(max_gap_s):
        raise ValueError("fps positivo e durata massima finita non negativa richiesti")
    frames = tracking.frame.to_numpy()
    if len(frames) and (np.any(np.diff(frames) <= 0) or not np.all(frames == frames.astype(int))):
        raise ValueError("I frame devono essere interi, ordinati e univoci")
    balls = tracking.ball_xy.tolist()
    def valid(point):
        return point is not None and len(point) == 2 and all(
            v is not None and math.isfinite(v) for v in point)
    observed = [valid(p) for p in balls]
    segments = tracking.segment.tolist() if "segment" in tracking else [0] * len(tracking)
    output = tracking.copy(deep=True)
    estimated = [None] * len(tracking)
    source = ["observed" if seen else "unknown" for seen in observed]
    age = [0. if seen else float("nan") for seen in observed]
    block_start = 0
    while block_start < len(tracking):
        block_end = block_start + 1
        while (block_end < len(tracking)
               and frames[block_end] == frames[block_end - 1] + 1
               and segments[block_end] == segments[block_end - 1]):
            block_end += 1
        i = block_start
        while i < block_end:
            if observed[i]:
                i += 1
                continue
            start = i
            while i < block_end and not observed[i]:
                i += 1
            end = i  # esclusivo
            if (end - start) / fps > max_gap_s:
                continue
            left, right = start - 1, end
            if method == "linear" and left >= block_start and right < block_end:
                a, b = np.asarray(balls[left]), np.asarray(balls[right])
                for k in range(start, end):
                    weight = (frames[k] - frames[left]) / (frames[right] - frames[left])
                    estimated[k] = tuple(a + weight * (b - a))
                    age[k] = min(frames[k] - frames[left], frames[right] - frames[k]) / fps
                    source[k] = "linear"
            elif (method == "velocity" and left - 1 >= block_start
                  and observed[left] and observed[left - 1]):
                a, previous = np.asarray(balls[left]), np.asarray(balls[left - 1])
                step = (a - previous) / (frames[left] - frames[left - 1])
                for k in range(start, end):
                    estimated[k] = tuple(a + step * (frames[k] - frames[left]))
                    age[k] = (frames[k] - frames[left]) / fps
                    source[k] = "velocity"
        block_start = block_end
    # Assegnazione object: tuple e None, senza trasformare le assenze in coordinate zero.
    output["ball_estimated_xy"] = pd.Series(estimated, index=output.index, dtype=object)
    output["ball_source"] = source
    output["anchor_distance_s"] = age
    return output


def reconstruction_examples():
    """Traiettorie inventate, con un cambio di direzione nascosto nel secondo caso."""
    frames = np.arange(70)
    x = 15 + .7 * frames
    scenarios = {}
    for name, y in [("Moto rettilineo", 20 + .2 * frames),
                    ("Cambio di direzione nascosto", 20 + .6 * np.abs(frames - 27))]:
        truth = list(zip(x, y))
        hidden = ((frames >= 18) & (frames <= 35)) | (frames >= 65)
        data = pd.DataFrame({"frame": frames, "ball_xy": [None if hide else p for p, hide in zip(truth, hidden)],
                             "segment": 1})
        scenarios[name] = (data, truth)
    return scenarios


def synthetic_error(reconstruction, truth):
    """Errore solo sui punti nascosti e ricostruiti di un esempio sintetico."""
    errors = []
    missing = 0
    for row, target in zip(reconstruction.itertuples(index=False), truth):
        if row.ball_source != "observed":
            missing += 1
            if row.ball_estimated_xy is not None:
                errors.append(math.dist(row.ball_estimated_xy, target))
    return {"metodo": next((s for s in reconstruction.ball_source if s not in {"observed", "unknown"}), "nessuna_stima"),
            "punti_nascosti": missing, "punti_stimati": len(errors),
            "errore_medio_m": float(np.mean(errors)) if errors else None,
            "errore_massimo_m": max(errors, default=None)}
