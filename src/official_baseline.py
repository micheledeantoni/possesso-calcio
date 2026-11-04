"""Voto temporale della baseline ufficiale, non regolato qui.

Estratto da starter/baseline.py di More Than a Hack 2027.
Copyright (c) 2026-2027 Metrica Sports. Licenza in LICENSE-METRICA.txt.
"""
import numpy as np
HOME, AWAY = 0, 1


def smooth(codes, window):
    """Majority vote over a centred window of +/- `window` frames. D where nobody has the ball.

    Computed with cumulative sums, so the cost does not grow with the window -- a 125-frame window
    over 199319 frames is the same work as a 1-frame one.

    Ties go to the home side. With an odd-sized window over two classes a tie needs the two counts
    to be exactly equal, which is rare and never worth a coin flip.
    """
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))

    home_cumulative = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_cumulative = np.concatenate(([0], np.cumsum(codes == AWAY)))

    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_cumulative[high] - home_cumulative[low]
    away = away_cumulative[high] - away_cumulative[low]

    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels

