import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[2]); import features_lib as fl
B = sys.argv[1]
for g in (1, 2):
    df = pd.read_csv(f"{B}/Features 9_11/codice/dataset/dataset_features_completo_partita{g}.csv")
    z = np.load(f"{B}/metriche partite Lore non sporche/metrica_partita_{g}.npz")
    gi = z["gioc"].astype(float); pa = z["palla"].astype(float)
    team = np.broadcast_to(z["squadra_slot"], gi.shape[:2])
    mine = fl.build(gi, pa, team)
    bad = []
    for c in df.columns[2:]:
        a, b = df[c].to_numpy(float), mine[c].to_numpy(float)
        same = np.isclose(a, b, rtol=1e-4, atol=1e-4, equal_nan=True)
        if not same.all(): bad.append((c, round(1 - same.mean(), 5)))
    print(f"partita {g}: colonne {len(df.columns)-2}, non coincidenti:", bad, "| extra mie:", set(mine.columns) - set(df.columns))
