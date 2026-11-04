"""Prima regola locale, deliberatamente incompleta: nessun tuning."""
import math
import pandas as pd


def predict(features):
    """Squadra più vicina; X se una distanza manca o c'è parità.

    Non riconosce D, non distingue controllo e vicinanza, non colma buchi.
    È il riferimento leggibile su cui sviluppare la macchina a stati.
    Non è la baseline ufficiale: non assegna D alle osservazioni assenti.
    """
    rows = []
    for row in features.itertuples(index=False):
        home, away = row.nearest_home_m, row.nearest_away_m
        label = "X"
        if math.isfinite(home) and math.isfinite(away) and home != away:
            label = "H" if home < away else "A"
        rows.append({"frame": row.frame, "label": label})
    return pd.DataFrame(rows, columns=["frame", "label"])
