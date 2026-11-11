"""Confronta src/cleaning.py con lo script originale del gruppo, sui soli dati Metrica.

Uso (dalla radice della repo):
    python docs/audit_drive/scripts/verify_cleaning.py "<Pulizia dati giocatori e palla.py>" <cartella Pragmata>

Dallo script originale si estraggono soltanto le funzioni (blocchi 3-6, `tratti`,
`pulisci_palla` del blocco 9b): le parti che leggono e salvano BAR-ATH non vengono eseguite.
Atteso: squadre, tracce, conteggi e palla identici; giocatori diversi solo nei frame
con doppioni, perché l'originale fa la media delle due posizioni e la repo no.
"""
import re
import sys
import types
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.arrays import TrackingArrays
from src.cleaning import clean_players, clean_ball
from src.team_features import continuity

original, folder = Path(sys.argv[1]), Path(sys.argv[2])
blocks = re.split(r"(?m)^# %% ", original.read_text(encoding="utf8"))
code = [b for b in blocks if b.startswith(("Blocco 3", "Blocco 4", "Blocco 5", "Blocco 6"))]
block9 = next(b for b in blocks if b.startswith("Blocco 9 "))
block9b = next(b for b in blocks if b.startswith("Blocco 9b")).split("def diagnosi_palla")[0]
source = ("import numpy as np\nL, W = 105.0, 68.0\nFPS = 25\nMAX_SQUADRA = 11\nV_MAX_GIOCATORE = 12.0\n"
          "DURATA_MIN = FPS\nQUOTA_GIRO = 0.5\nDIST_DOPPIONE = 1.0\nMARGINE_FUORI = 1.0\nV_MAX_PALLA = 40.0\n"
          "BUCO_MAX_S = 1.0\n" + "\n".join("# " + b for b in code)
          + "\ndef tratti" + block9.split("def tratti")[1].split("def pulisci_palla")[0] + "\n# " + block9b)
lorenzo = types.ModuleType("lorenzo")
exec(compile(source, "lorenzo", "exec"), lorenzo.__dict__)

for game in (1, 2, 3):
    data = np.load(folder / f"Train set/metrica_partita_{game}_sporca.npz")
    players, ball = data["gioc"], data["palla"]
    n, slots = players.shape[:2]
    present = np.isfinite(players[..., 0])
    # Squadra fissa per slot (la più frequente): stesso input per entrambe le versioni.
    team = np.array([np.bincount(data["squadra_fr"][present[:, s], s], minlength=2).argmax()
                     if present[:, s].any() else 0 for s in range(slots)], np.int8)
    theirs_players, theirs_team, theirs_log = lorenzo.pulisci_giocatori(players, team, ball, np.ones(n, bool))
    arrays = TrackingArrays(np.arange(n), players.astype(float), ball.astype(float), np.tile(team, (n, 1)),
                            np.ones(n, int), pitch=(105., 68.))
    ours, our_log = clean_players(arrays)
    different = ~np.isclose(theirs_players.astype(float), ours.players, atol=1e-4, equal_nan=True).all(axis=2)
    print(f"partita {game}: squadre identiche {np.array_equal(theirs_team, ours.team)} | "
          f"tracce {theirs_log['tracce']}/{our_log['tracce']} | girate {theirs_log['girate']}/{our_log['tracce_girate']} | "
          f"doppioni {theirs_log['uniti']}/{our_log['doppioni_uniti']} | fuori campo "
          f"{theirs_log['tolti_fuori']}/{our_log['tolti_fuori_campo']} | frame con giocatori diversi "
          f"{int(different.any(axis=1).sum())}")
    raw_ball = np.load(folder / f"metriche partite Lore non sporche/metrica_partita_{game}.npz")["palla"]
    m = len(raw_ball)
    tb, ti, tr = lorenzo.pulisci_palla(raw_ball)
    ob, oi, orm = clean_ball(raw_ball, continuity(np.arange(m), np.ones(m, int)), np.ones(m, bool))
    print(f"           palla identica {np.allclose(tb, ob, atol=1e-4, equal_nan=True)} | "
          f"interpolati {int(ti.sum())}/{int(oi.sum())} | cancellati {int(tr.sum())}/{int(orm.sum())}")
