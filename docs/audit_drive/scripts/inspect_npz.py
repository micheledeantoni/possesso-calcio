import sys, numpy as np
for p in sys.argv[1:]:
    z = np.load(p, allow_pickle=False)
    print("==", p.split("Pragmata/")[-1])
    for k in z.files:
        a = z[k]
        info = f"{k}: {a.dtype} {a.shape}"
        if a.dtype.kind in "fi" and a.size:
            info += f" min={np.nanmin(a):.3f} max={np.nanmax(a):.3f} nan%={100*np.isnan(a).mean() if a.dtype.kind=='f' else 0:.1f}"
        elif a.size < 40:
            info += f" val={a.tolist()}"
        else:
            u, c = np.unique(a, return_counts=True)
            if len(u) < 10: info += f" uniq={dict(zip(u.tolist(), c.tolist()))}"
        print("  ", info)
