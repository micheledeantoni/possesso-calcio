import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[2]); import features_lib as fl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

B = sys.argv[1]
HP_HA = dict(max_iter=150, learning_rate=0.1, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
HP_3 = dict(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)

def load(g, dirty):
    if dirty:
        z = np.load(f"{B}/Train set/metrica_partita_{g}_sporca.npz"); team = z["squadra_fr"]
        nom = None
    else:
        z = np.load(f"{B}/metriche partite Lore non sporche/metrica_partita_{g}.npz")
        team = np.broadcast_to(z["squadra_slot"], z["gioc"].shape[:2])
        nom = [int((z["squadra_slot"] == k).sum()) for k in (0, 1)]
    X = fl.build(z["gioc"].astype(float), z["palla"].astype(float), team, n_slot_team=nom)
    X.insert(0, "frame", np.arange(len(X)))
    return X, pd.Series(z["etichette"])

D = {(g, d): load(g, d) for g in (1, 2, 3) for d in (False, True)}
ALL = [c for c in D[(1, False)][0].columns if c != "frame"]
NOSLOT = [c for c in ALL if not c.startswith("slot_")]
# feature "team-level" robuste: niente slot, niente conteggi assoluti di slot nominali
ROBUST = [c for c in NOSLOT if "players_missing" not in c and "available_pct" not in c]

def fit(X, y, cols, hp=HP_HA):
    keep = [c for c in cols if X[c].nunique(dropna=False) > 1]
    m = HistGradientBoostingClassifier(**hp).fit(X[keep], y)
    return m, keep

def ha(X, y):
    k = y.isin(["H", "A"]); return X[k], y[k]

def changes(lbl):
    lbl = np.asarray(lbl); return int((lbl[1:] != lbl[:-1]).sum())

def smooth_ha(pred, w):
    if w == 0: return pred
    h = (pred == "H").astype(float)
    k = np.ones(2 * w + 1) / (2 * w + 1)
    return np.where(np.convolve(h, k, mode="same") >= .5, "H", "A")

print("== 0. Regola 'squadra piu vicina' (nessun fit), frame H/A con palla")
for g in (1, 2, 3):
    for d in (False, True):
        X, y = D[(g, d)]; k = y.isin(["H", "A"]) & X.diff_min_dist_ball.notna()
        pred = np.where(X.diff_min_dist_ball[k] < 0, "H", "A")
        print(f"  partita {g} {'sporca' if d else 'pulita'}: acc {accuracy_score(y[k], pred):.4f} (copertura {k.sum()/y.isin(['H','A']).sum():.3f})")

print("\n== 1. Riproduzione esperimento colleghi (train p1 frame<100540 H/A, pulito)")
X1, y1 = D[(1, False)]; Xh, yh = ha(X1, y1)
tr = Xh.frame < 100540
m, keep = fit(Xh[tr], yh[tr], ALL)
print("  feature usate:", len(keep), "| slot usate:", [c for c in keep if c.startswith("slot_")])
print(f"  test temporale p1: {accuracy_score(yh[~tr], m.predict(Xh[~tr][keep])):.4f} (atteso 0.9179)")
X2, y2 = ha(*D[(2, False)])
p2 = m.predict(X2[keep])
print(f"  partita 2 H/A: {accuracy_score(y2, p2):.4f} (atteso 0.9192) | cambi reali {changes(y2)} predetti {changes(p2)}")
imp = pd.Series(dict(zip(keep, [0]*len(keep))))

print("\n== 2. Ablazione e leave-one-game-out (train su 2 partite H/A, test sulla terza)")
for name, cols in [("ALL", ALL), ("NOSLOT", NOSLOT), ("ROBUST", ROBUST)]:
    res = []
    for test in (1, 2, 3):
        trg = [g for g in (1, 2, 3) if g != test]
        Xtr = pd.concat([ha(*D[(g, False)])[0] for g in trg]); ytr = pd.concat([ha(*D[(g, False)])[1] for g in trg])
        m, keep = fit(Xtr, ytr, cols)
        Xt, yt = ha(*D[(test, False)]); Xs, ys = ha(*D[(test, True)])
        res.append(f"p{test}: pulito {accuracy_score(yt, m.predict(Xt[keep])):.4f} / sporco {accuracy_score(ys, m.predict(Xs[keep])):.4f}")
    print(f"  {name:7s}", " | ".join(res))

print("\n== 3. Train su dati SPORCATI (ROBUST), test su partita sporcata esclusa")
for test in (1, 2, 3):
    trg = [g for g in (1, 2, 3) if g != test]
    Xtr = pd.concat([ha(*D[(g, True)])[0] for g in trg]); ytr = pd.concat([ha(*D[(g, True)])[1] for g in trg])
    m, keep = fit(Xtr, ytr, ROBUST)
    Xs, ys = ha(*D[(test, True)]); ps = m.predict(Xs[keep])
    best = []
    for w in (0, 6, 12, 25, 50):
        q = smooth_ha(ps, w); best.append(f"w={w}: {accuracy_score(ys, q):.4f}/{changes(q)}")
    print(f"  p{test}: cambi reali {changes(ys)} | acc/cambi predetti ->", "  ".join(best))

print("\n== 4. D: 'palla assente => D' (accuratezza della regola su tutti i frame)")
for g in (1, 2, 3):
    for d in (False, True):
        X, y = D[(g, d)]; miss = X.ball_available == 0
        print(f"  p{g} {'sporca' if d else 'pulita'}: P(D|palla assente)={(y[miss]=='D').mean():.3f}  P(palla assente|D)={miss[y=='D'].mean():.3f}  frame assenti a gioco vivo={int((miss & (y!='D')).sum())}")

print("\n== 5. Modello 3 classi (Modello2_DAH): split 80% su p1 pulita, poi p2 pulita/sporca tutti i frame")
X1, y1 = D[(1, False)]; s = int(len(X1) * .8)
m3, keep3 = fit(X1.iloc[:s], y1.iloc[:s], ALL, HP_3)
print(f"  test temporale p1: {accuracy_score(y1.iloc[s:], m3.predict(X1.iloc[s:][keep3])):.4f}")
for d in (False, True):
    X, y = D[(2, d)]; p = m3.predict(X[keep3])
    print(f"  p2 {'sporca' if d else 'pulita'} tutti i frame: {accuracy_score(y, p):.4f} | cambi reali {changes(y)} predetti {changes(p)}")
