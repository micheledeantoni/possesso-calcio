import sys, numpy as np, pandas as pd
B = sys.argv[1]
for g in (1, 2):
    df = pd.read_csv(f"{B}/Features 9_11/codice/dataset/dataset_features_completo_partita{g}.csv")
    z = np.load(f"{B}/metriche partite Lore non sporche/metrica_partita_{g}.npz")
    zs = np.load(f"{B}/Train set/metrica_partita_{g}_sporca.npz")
    print(f"===== partita {g}: shape {df.shape}")
    print("frame 0..n-1:", df.frame.tolist() == list(range(len(df))))
    print("etichette = npz pulito:", (df.etichetta.to_numpy() == z["etichette"]).all())
    pal = np.isfinite(z["palla"]).all(1); pals = np.isfinite(zs["palla"]).all(1)
    print("ball_available = pulito:", (df.ball_available.to_numpy() == pal).mean(), " = sporco:", (df.ball_available.to_numpy() == pals).mean())
    print("squadra_slot pulito:", z["squadra_slot"].tolist())
    # team_0 player count vs pulito
    t0 = np.isfinite(z["gioc"][:, z["squadra_slot"] == 0, 0]).sum(1)
    print("team_0_players_available = pulito(squadra 0):", (df.team_0_players_available.to_numpy() == t0).mean())
    # label vs proximity: is H associated with team_0 closer?
    ha = df[df.etichetta.isin(["H", "A"]) & df.diff_min_dist_ball.notna()]
    closer0 = ha.diff_min_dist_ball < 0
    print("P(team_0 piu vicino | H) =", round(closer0[ha.etichetta == "H"].mean(), 3), " | A) =", round(closer0[ha.etichetta == "A"].mean(), 3))
    print("D: ball_available media", round(df.loc[df.etichetta == "D", "ball_available"].mean(), 3), " H/A:", round(df.loc[df.etichetta != "D", "ball_available"].mean(), 3))
    slots = [c for c in df.columns if c.startswith("slot_")]
    print("n slot cols:", len(slots), "| costanti:", [c for c in df.columns if df[c].nunique(dropna=False) <= 1][:50])
    print("ball_speed max", df.ball_speed.max(), "anomalie", df.ball_tracking_anomaly.sum())
    print("palla x range", np.nanmin(z["palla"][:,0]), np.nanmax(z["palla"][:,0]))
cols = list(df.columns)
print("\nCOLONNE:", cols)
