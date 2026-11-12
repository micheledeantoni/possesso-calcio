"""Metriche di classifica della challenge: spellF1 e macroF1.

Reimplementazione dalla specifica di `evaluation/possession_metrics.py` e
`EVALUATION.md` della repo pubblica More than a Hack 2027 (licenza MIT,
Metrica Sports). Serve a valutare e scegliere i modelli sulle partite Metrica
con la stessa metrica della classifica; non legge la soluzione della challenge.

- Frame valutati: quelli con etichetta vera diversa da X, divisi in tratti
  continui (`valid_runs`); spell e riempimenti non attraversano i tratti.
- spellF1: un possesso ("spell") è un tratto massimo di H o di A; D lo spezza.
  Le X della previsione sono riempite con l'ultima etichetta nota del tratto
  (all'inizio del tratto con la prima nota). Spell vere e previste della stessa
  squadra sono abbinate in modo greedy per IoU temporale decrescente; per ogni
  soglia in SPELL_IOUS: F1 = 2TP / (2TP + FP + FN). spellF1 = media sulle soglie.
- macroF1: media non pesata degli F1 per frame di H, A e D sui frame valutati,
  con le previsioni grezze: una X prevista è un falso negativo, mai un falso positivo.
"""
import numpy as np

CODES = {"H": 0, "A": 1, "D": 2, "X": 3}
H, A, D, X = 0, 1, 2, 3
SPELL_IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def _codes(labels):
    labels = np.asarray(labels).astype(str)
    unknown = ~np.isin(labels, list(CODES))
    if unknown.any():
        raise ValueError(f"Etichette non ammesse: {sorted(set(labels[unknown]))[:5]}")
    return np.vectorize(CODES.get, otypes=[np.int8])(labels) if len(labels) else np.zeros(0, np.int8)


def valid_runs(truth_codes, breaks=None):
    """Tratti [inizio, fine) con etichetta vera diversa da X.

    `breaks`: opzionale, True dove inizia un nuovo tratto anche senza X (per
    esempio il secondo tempo nei dati Metrica, che non hanno X all'intervallo).
    """
    graded = (np.asarray(truth_codes) != X).astype(np.int8)
    edges = np.diff(np.concatenate(([0], graded, [0])))
    starts, stops = list(np.flatnonzero(edges == 1)), list(np.flatnonzero(edges == -1))
    runs = []
    for start, stop in zip(starts, stops):
        cut = [start] + ([int(i) for i in np.flatnonzero(breaks[start + 1:stop]) + start + 1]
                         if breaks is not None else []) + [stop]
        runs += [(a, b) for a, b in zip(cut[:-1], cut[1:]) if b > a]
    return runs


def fill_abstentions(pred_codes, runs):
    """X → ultima etichetta nota nel tratto; in testa al tratto, la prima nota."""
    filled = np.array(pred_codes, copy=True)
    for start, stop in runs:
        window = filled[start:stop]
        known = np.flatnonzero(window != X)
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(window != X, np.arange(len(window)), -1))
        carry[carry < 0] = known[0]
        filled[start:stop] = window[carry]
    return filled


def possession_spells(codes, runs):
    """(inizio, fine, codice) dei tratti massimi di H o A dentro ogni tratto valutato."""
    spells = []
    for start, stop in runs:
        window = codes[start:stop]
        if not len(window):
            continue
        edges = np.flatnonzero(np.diff(window)) + 1
        bounds = np.concatenate(([0], edges, [len(window)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if window[a] in (H, A):
                spells.append((start + int(a), start + int(b), int(window[a])))
    return spells


def spell_f1(truth, pred, breaks=None, ious=SPELL_IOUS):
    """spellF1 e F1 per soglia. `truth`, `pred`: etichette H/A/D/X per frame."""
    t, p = _codes(truth), _codes(pred)
    if len(t) != len(p):
        raise ValueError("Verità e previsione hanno lunghezze diverse")
    runs = valid_runs(t, breaks)
    gt = possession_spells(t, runs)
    pr = possession_spells(fill_abstentions(p, runs), runs)
    pairs = []
    by_code = {}
    for j, (a, b, code) in enumerate(pr):
        by_code.setdefault(code, []).append((a, b, j))
    for i, (ga, gb, code) in enumerate(gt):
        for pa, pb, j in by_code.get(code, []):
            if pb <= ga or gb <= pa:
                continue
            pairs.append(((min(gb, pb) - max(ga, pa)) / float(max(gb, pb) - min(ga, pa)), i, j))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    per_threshold = {}
    for threshold in ious:
        gt_used, pred_used = set(), set()
        for iou, i, j in pairs:
            if iou < threshold:
                break
            if i in gt_used or j in pred_used:
                continue
            gt_used.add(i)
            pred_used.add(j)
        tp = len(gt_used)
        fp, fn = len(pr) - tp, len(gt) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_threshold[threshold] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"spellF1": float(np.mean(list(per_threshold.values()))), "spell_f1_per_iou": per_threshold,
            "spell_vere": len(gt), "spell_previste": len(pr)}


def macro_f1(truth, pred):
    """Media degli F1 per frame di H, A, D sui frame con verità diversa da X."""
    t, p = _codes(truth), _codes(pred)
    graded = t != X
    t, p = t[graded], p[graded]
    scores = {}
    for name, code in (("H", H), ("A", A), ("D", D)):
        tp = int(np.sum((t == code) & (p == code)))
        fp = int(np.sum((t != code) & (p == code)))
        fn = int(np.sum((t == code) & (p != code)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores[name] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"macroF1": float(np.mean(list(scores.values()))), **{f"F1_{k}": v for k, v in scores.items()}}


def challenge_scores(truth, pred, breaks=None):
    """spellF1 (classifica) e macroF1 (primo spareggio)."""
    spells = spell_f1(truth, pred, breaks)
    return {"spellF1": spells["spellF1"], **macro_f1(truth, pred),
            "spell_vere": spells["spell_vere"], "spell_previste": spells["spell_previste"]}
