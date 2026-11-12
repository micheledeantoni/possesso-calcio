"""Baseline ufficiale della challenge (giocatore più vicino + voto di maggioranza) sulle partite Metrica.

Uso (dalla radice della repo):
    python docs/audit_drive/scripts/baseline_metrica.py <cartella npz Metrica etichettati>

Dà un riferimento confrontabile con i nostri modelli sugli stessi dati: la baseline
ufficiale ha spellF1 0,2433 sulla partita della challenge (finestra ±50 frame).
Qui la finestra non viene scelta: si riportano le stesse finestre della documentazione.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.model import load_metrica_games, _breaks
from src.official_baseline import smooth, HOME, AWAY
from src.challenge_metrics import challenge_scores

for game in load_metrica_games(sys.argv[1]):
    home, away = game.features.home_min_dist_ball.to_numpy(), game.features.away_min_dist_ball.to_numpy()
    codes = np.where(np.isfinite(home) & np.isfinite(away), np.where(home <= away, HOME, AWAY), 2)
    row = []
    for window in (0, 25, 50, 125):
        scores = challenge_scores(game.labels, smooth(codes, window), _breaks(game.continuous))
        row.append(f"±{window}: spellF1 {scores['spellF1']:.4f} macroF1 {scores['macroF1']:.4f}")
    print(game.name, " | ".join(row))
