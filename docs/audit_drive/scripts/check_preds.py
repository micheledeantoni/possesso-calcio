import sys, numpy as np, pandas as pd, pickletools, io
B = sys.argv[1]
p = pd.read_csv(f"{B}/Features 9_11/codice/valutazione/predizioni_partita2.csv")
z = np.load(f"{B}/metriche partite Lore non sporche/metrica_partita_2.npz")
print("righe", len(p), "etichette = npz:", (p.etichetta_reale.to_numpy() == z["etichette"]).all())
ha = p[p.etichetta_reale != "D"]
print("acc H/A:", round((ha.etichetta_reale == ha.predizione_HA).mean(), 4))
# score contando anche D (la challenge valuta tutti i frame)
print("acc su tutti i frame (D sempre sbagliato):", round((p.etichetta_reale == p.predizione_HA).mean(), 4))
def runs(s):
    s = np.asarray(s); return 1 + int((s[1:] != s[:-1]).sum())
print("blocchi reali (H/A/D):", runs(p.etichetta_reale), " blocchi predetti:", runs(p.predizione_HA))
# cambi H<->A solo su frame H/A consecutivi
t, q = ha.etichetta_reale.to_numpy(), ha.predizione_HA.to_numpy()
print("cambi H/A reali (frame H/A):", int((t[1:] != t[:-1]).sum()), " predetti:", int((q[1:] != q[:-1]).sum()))
# lunghezza blocchi predetti
pr = p.predizione_HA.to_numpy(); b = np.flatnonzero(np.r_[True, pr[1:] != pr[:-1], True]); L = np.diff(b)
print("blocchi predetti < 1s:", int((L < 25).sum()), "su", len(L), "| mediana frame", int(np.median(L)))
# pickle: nomi feature senza eseguire
for path in [f"{B}/Features 9_11/codice/modello/modello_HA_100_partita1.pkl", f"{B}/Modello2_DAH/modello_unico_3_classi_partita1.pkl"]:
    data = open(path, "rb").read()
    strings = set()
    try:
        for op, arg, pos in pickletools.genops(io.BytesIO(data)):
            if isinstance(arg, str): strings.add(arg)
    except Exception as e: print("parse stop:", e)
    mods = sorted(s for s in strings if "." in s and s.split(".")[0] in {"sklearn","numpy","builtins","joblib","os","subprocess","posix","nt"})
    feats = sorted(s for s in strings if s.startswith(("slot_","team_","diff_","ball_")))
    print(path.split("/")[-1], "| moduli:", mods[:15], "| n feature-like:", len(feats), "| slot:", [s for s in feats if s.startswith("slot_")])
