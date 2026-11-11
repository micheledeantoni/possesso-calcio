"""Confronta la sporcatura della repo con i file `Train set` del gruppo (solo dati Metrica).

Uso (dalla radice della repo):
    python docs/audit_drive/scripts/verify_dirtying.py <cartella Pragmata> <cartella temporanea>

1. Sporca e pulisce le tre partite Metrica pulite (semi 1, 2, 3) e salva i file.
2. Confronta le quantità della sporcatura (`dirt_statistics`) con il `Train set`.
3. Verifica incrociata del modello: train sulle altre due partite (nostre o del
   gruppo), test sulla terza (nostra o del gruppo).
"""
import sys
from dataclasses import replace
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.arrays import load_metrica_npz
from src.dirtying import dirt_statistics, make_training_game, save_training_game
from src.model import load_metrica_games, fit_possession_model, predict_possession, evaluate

folder, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)
for game in (1, 2, 3):
    clean = load_metrica_npz(folder / f"metriche partite Lore non sporche/metrica_partita_{game}.npz")
    theirs = load_metrica_npz(folder / f"Train set/metrica_partita_{game}_sporca.npz")
    clean = replace(clean, segment=theirs.segment)
    ours, interpolated, _ = make_training_game(clean, clean.labels, seed=game)
    save_training_game(out / f"metrica_partita_{game}_sporca.npz", ours, interpolated, clean.labels)
    table = pd.DataFrame({"Train set": dirt_statistics(clean, theirs, clean.labels),
                          "repo": dirt_statistics(clean, ours, clean.labels)})
    print(f"== partita {game}\n{table.round(3).to_string()}\n")

ours, theirs = load_metrica_games(out), load_metrica_games(folder / "Train set")
rows = []
for i in range(3):
    for train_name, train in (("repo", ours), ("Train set", theirs)):
        model = fit_possession_model([g for j, g in enumerate(train) if j != i])
        for test_name, test in (("repo", ours[i]), ("Train set", theirs[i])):
            result = evaluate(test.labels, predict_possession(model, test.features, test.continuous).label.to_numpy())
            rows.append({"partita": i + 1, "train": train_name, "test": test_name,
                         "accuratezza_HA": round(result["accuratezza_HA"], 4),
                         "cambi_reali/predetti": f"{result['cambi_reali']}/{result['cambi_predetti']}",
                         "cambi_f1": round(result["cambi_f1"], 3)})
print(pd.DataFrame(rows).to_string(index=False))
