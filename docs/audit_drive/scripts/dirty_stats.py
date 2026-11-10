import sys, numpy as np
B = sys.argv[1]
for g in (1, 2, 3):
    for kind in ("pulita", "sporca"):
        if kind == "pulita":
            z = np.load(f"{B}/metriche partite Lore non sporche/metrica_partita_{g}.npz"); team = np.broadcast_to(z["squadra_slot"], z["gioc"].shape[:2])
        else:
            z = np.load(f"{B}/Train set/metrica_partita_{g}_sporca.npz"); team = z["squadra_fr"]
        ok = np.isfinite(z["gioc"]).all(2)
        live = z["etichette"] != "D"
        c = [(ok & (team == k)).sum(1) for k in (0, 1)]
        over = ((c[0] > 11) | (c[1] > 11))[live].mean(); under = ((c[0] < 11) | (c[1] < 11))[live].mean()
        absent = (np.clip(11 - c[0], 0, None) + np.clip(11 - c[1], 0, None))[live].mean()
        # team switches per slot (dirty only)
        sw = int((team[1:] != team[:-1]).sum()) if kind == "sporca" else 0
        # doppioni: compagni a < 1 m
        g_ = z["gioc"][live][::25]; t_ = team[live][::25]
        d = np.linalg.norm(g_[:, :, None] - g_[:, None], axis=3); same = t_[:, :, None] == t_[:, None]
        iu = np.triu(np.ones(d.shape[1:], bool), 1)
        dup = ((d < 1) & same & iu).any((1, 2)).mean()
        extra = f" periodo-cambio frame={int(np.flatnonzero(np.diff(z['periodo']))[0])+1}" if "periodo" in z.files else ""
        print(f"p{g} {kind}: gioco vivo >11 {100*over:.1f}% | <11 {100*under:.1f}% | assenti medi {absent:.2f} | doppioni<1m {100*dup:.1f}% | cambi squadra slot {sw}{extra}")
