"""Confronta src/team_features.py con i CSV dei colleghi e ripete il test leave-one-game-out.

Uso (dalla radice della repo):
    python docs/audit_drive/scripts/verify_team_features.py <cartella Pragmata>
Legge solo dati Metrica (esterni). Richiede scikit-learn.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.arrays import load_metrica_npz
from src.team_features import build_team_features, feature_columns

B = Path(sys.argv[1])
RENAME = {"width": "x_extent", "length": "y_extent", "team_width": "x_extent", "team_length": "y_extent",
          "team_mean_x": "mean_x"}


def ours_name(col):
    for k, prefix in (("team_0_", "home_"), ("team_1_", "away_")):
        if col.startswith(k):
            rest = col[len(k):]
            return prefix + RENAME.get(rest, rest)
    if col.startswith("diff_"):
        rest = col[5:]
        return "diff_" + RENAME.get(rest, rest)
    # 0_5s: loro 12 frame (round(12.5)), noi 13: differenza voluta.
    return col.replace("ball_distance_", "ball_displacement_")


print("== Confronto con i CSV dei colleghi (soglia palla 35 m/s come loro)")
for g in (1, 2):
    theirs = pd.read_csv(B / f"Features 9_11/codice/dataset/dataset_features_completo_partita{g}.csv")
    ours = build_team_features(load_metrica_npz(B / f"metriche partite Lore non sporche/metrica_partita_{g}.npz"),
                               ball_speed_max_ms=35.)
    same, differ, missing = 0, [], []
    for col in theirs.columns[2:]:
        name = ours_name(col)
        if name not in ours:
            missing.append(col)
            continue
        a, b = theirs[col].to_numpy(float), ours[name].to_numpy(float)
        if col == "ball_direction_change":  # +pi e -pi sono lo stesso angolo
            a, b = np.abs(a), np.abs(b)
        if np.isclose(a, b, rtol=1e-3, atol=2e-2, equal_nan=True).all():
            same += 1
        else:
            differ.append(col)
    print(f"  partita {g}: uguali {same} | diverse {differ} | non riprese {len(missing)} "
          f"({sorted({m.split('_available')[0] if m.startswith('slot_') else m for m in missing})[:6]}...)")

print("\n== Leave-one-game-out su dati sporcati (soglia 40 m/s, segmenti = periodi)")
from sklearn.ensemble import HistGradientBoostingClassifier
data = {}
for g in (1, 2, 3):
    t = load_metrica_npz(B / f"Train set/metrica_partita_{g}_sporca.npz")
    f = build_team_features(t)
    keep = np.isin(t.labels, ["H", "A"])
    data[g] = (f[keep], t.labels[keep])
cols = feature_columns(data[1][0])
for test in (1, 2, 3):
    X = pd.concat([data[g][0] for g in (1, 2, 3) if g != test])
    y = np.concatenate([data[g][1] for g in (1, 2, 3) if g != test])
    model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=15,
                                           l2_regularization=1.0, random_state=42).fit(X[cols], y)
    Xt, yt = data[test]
    print(f"  test partita {test}: accuratezza H/A {np.mean(model.predict(Xt[cols]) == yt):.4f}")
