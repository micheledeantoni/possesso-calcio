"""Modello a tre stati H/A/D: possesso dal modello H/A, gioco fermo da due classificatori.

Estende `src/model.py`:
- D vs gioco vivo: due classificatori addestrati su tutti i frame.
  - *senza palla generica*: feature di `src/dead_ball_features.py`, cioè giocatori
    più dove e come sta la palla quando si vede (`ball_position_features`, 0 quando
    manca: l'assenza della palla non è un indizio);
  - *con palla*: le stesse più tutte le feature per squadra, compresa la
    disponibilità della palla. Nelle etichette Metrica D coincide con la palla
    assente, quindi questo classificatore impara anche la scorciatoia "palla assente
    ⇒ D".
  P(D) = w · P_con_palla + (1 − w) · P_senza_palla, con w = `DEAD_BALL_WEIGHT`
  (0,34, fissato a mano, non scelto in validazione). w = 0 usa solo il primo.
- H vs A dato il gioco vivo: il classificatore di `src/model.py`.
- Evidenza per frame (log-probabilità meno log-prior):
    D:  log P(D)                    − log prior(D)
    H:  log(1 − P(D)) + log P(H|¬D) − log prior(H), e così A.
- Viterbi a tre stati per le etichette; forward-backward per le probabilità di ogni
  frame. Transizioni stimate dalle etichette Metrica, transizioni fra stati diversi
  moltiplicate per `switch_scale`; libere ai salti di frame e ai cambi di segmento.
- Temperatura, scala e emissioni H/A neutre senza palla scelte in leave-one-game-out
  sulle sole partite Metrica; obiettivo predefinito spellF1, la metrica di classifica.
  Sono escluse le combinazioni che dicono D in più di `MAX_FALSE_D` (30%) dei frame
  a gioco vivo con palla assente: è il freno alla scorciatoia del classificatore
  con palla.

Valutazione da terminale:
    python -m src.model_dah data/metrica --extra data/metrica_extra_1 --extra data/metrica_extra_2
    python -m src.model_dah data/metrica --peso-palla 0     (senza classificatore con palla)
"""
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

from src.arrays import load_metrica_npz
from src.team_features import build_team_features, feature_columns, continuity
from src.dead_ball_features import build_dead_features, dead_ball_columns
from src.model import CLASSIFIER_PARAMS, EPSILON, LABELS as HA, fit_classifier, log_evidence, evaluate, _breaks, _challenge

STATES = np.array(["H", "A", "D"])
DEAD_BALL_WEIGHT = 0.34  # w: peso del classificatore D con palla (vedi docstring)
MAX_FALSE_D = 0.30       # quota massima di D sui frame vivi con palla assente, in validazione
# Astensione (lavoro A8): dove la probabilità più alta (forward-backward) è sotto
# questa soglia, l'etichetta diventa X. None = mai X. Scelta su Metrica con
# `--astensione`: deve alzare il macroF1 (lo spareggio) senza abbassare lo spellF1.
ABSTAIN_THRESHOLD = None
DEFAULT_GRID = {"temperature": (0.03, 0.05, 0.1, 0.2),
                "switch_scale": (1.0, 2.0, 4.0, 8.0, 16.0),
                "neutral_without_ball": (False, True),
                "w": (DEAD_BALL_WEIGHT,)}


@dataclass
class DahGame:
    name: str
    features: pd.DataFrame
    dead_features: pd.DataFrame
    labels: np.ndarray
    continuous: np.ndarray
    primary: bool = True  # False: versione sporcata in più, solo per il training


@dataclass
class DahModel:
    ha_classifier: object
    ha_columns: list
    ha_prior: dict
    dead_without: tuple  # (classificatore, colonne, prior) senza palla generica
    dead_with: tuple     # (classificatore, colonne, prior) con palla
    transitions: np.ndarray
    params: dict
    training_games: list
    sklearn_version: str

    @property
    def dead_columns(self):
        """Colonne del classificatore D senza palla generica."""
        return self.dead_without[1]

    @property
    def dead_ball_weight(self):
        return dead_ball_weight(self.params)

    @property
    def dead_ball_uses_ball(self):
        """True se il classificatore D con palla pesa nelle previsioni (w > 0)."""
        return self.dead_ball_weight > 0


def dead_ball_weight(params):
    """w dei parametri; se manca, `DEAD_BALL_WEIGHT`."""
    w = float(params.get("w", DEAD_BALL_WEIGHT))
    if not 0 <= w <= 1:
        raise ValueError("w deve stare fra 0 e 1")
    return w


def load_dah_games(folder, ball_speed_max_ms=40.0, primary=True, ball_position=True, **feature_options):
    """Partite di una cartella. Il nome del file identifica la partita: versioni
    sporcate diverse della stessa partita (`primary=False`) hanno lo stesso nome e
    finiscono sempre dalla stessa parte della validazione incrociata.
    `ball_position=False`: gioco fermo con i soli giocatori (riferimento)."""
    games = []
    for path in sorted(Path(folder).glob("*.npz")):
        arrays = load_metrica_npz(path)
        if arrays.labels is None:
            raise ValueError(f"{path.name}: mancano le etichette")
        games.append(DahGame(path.stem, build_team_features(arrays, ball_speed_max_ms=ball_speed_max_ms,
                                                            **feature_options),
                             build_dead_features(arrays, use_ball=ball_position), np.asarray(arrays.labels),
                             continuity(arrays.frames, arrays.segment), primary))
    if not games:
        raise ValueError(f"Nessun file .npz in {folder}")
    return games


def transition_matrix(games):
    """P(stato al frame t | stato al frame t−1) contando le coppie di frame continui."""
    index = {s: i for i, s in enumerate(STATES)}
    counts = np.ones((3, 3))  # una pseudo-osservazione per cella
    for game in games:
        codes = np.array([index.get(s, -1) for s in game.labels])
        ok = game.continuous[1:] & (codes[1:] >= 0) & (codes[:-1] >= 0)
        np.add.at(counts, (codes[:-1][ok], codes[1:][ok]), 1)
    return counts / counts.sum(axis=1, keepdims=True)


def scaled_transitions(transitions, scale):
    """Transizioni fra stati diversi × scale; la diagonale completa la riga."""
    t = np.array(transitions, dtype=float)
    off = t * scale
    np.fill_diagonal(off, 0)
    if np.any(off.sum(axis=1) >= 1):
        raise ValueError("switch_scale troppo grande per queste transizioni")
    np.fill_diagonal(off, 1 - off.sum(axis=1))
    return off


def _dead_matrix(game, columns, uses_ball):
    if uses_ball:
        return pd.concat([game.dead_features, game.features.drop(columns=["frame", "ball_source"], errors="ignore")],
                         axis=1)[columns]
    return game.dead_features[columns]


def _fit_dead_one(games, uses_ball):
    from sklearn.ensemble import HistGradientBoostingClassifier
    columns = dead_ball_columns(games[0].dead_features)
    if uses_ball:
        columns += [c for c in feature_columns(games[0].features) if c != "ball_source"]
    X = pd.concat([_dead_matrix(g, columns, uses_ball) for g in games])
    y = np.concatenate([g.labels == "D" for g in games])
    columns = [c for c in columns if X[c].nunique(dropna=False) > 1]
    classifier = HistGradientBoostingClassifier(**CLASSIFIER_PARAMS).fit(X[columns], y)
    return classifier, columns, float(y.mean())


def fit_dead_classifier(games):
    """I due classificatori D, (senza palla generica, con palla): ognuno (classificatore, colonne, prior)."""
    return _fit_dead_one(games, False), _fit_dead_one(games, True)


def _p_dead(classifier, columns, game, uses_ball):
    p = classifier.predict_proba(_dead_matrix(game, columns, uses_ball))[:, list(classifier.classes_).index(True)]
    return np.clip(p, EPSILON, 1 - EPSILON)


def evidence_parts(model_parts, game):
    """Componenti dell'evidenza: P(D) dei due classificatori, prior di D, log-rapporti H e A."""
    ha_clf, ha_cols, ha_prior, (d0_clf, d0_cols, d_prior), (d1_clf, d1_cols, _) = model_parts
    ha = log_evidence(ha_clf, ha_cols, ha_prior, game.features)  # colonne H, A
    return {"p_dead_without": _p_dead(d0_clf, d0_cols, game, False),
            "p_dead_with": _p_dead(d1_clf, d1_cols, game, True),
            "dead_prior": d_prior,  # uguale per i due classificatori: stesse etichette
            "H": ha[:, 0], "A": ha[:, 1],
            "ball": game.features["ball_available"].to_numpy() == 1}


def emissions(parts, params):
    """Log-evidenze H, A, D per frame, con P(D) = w · P_con_palla + (1 − w) · P_senza_palla."""
    w = dead_ball_weight(params)
    p_dead = np.clip(w * parts["p_dead_with"] + (1 - w) * parts["p_dead_without"], EPSILON, 1 - EPSILON)
    d = np.log(p_dead) - math.log(parts["dead_prior"])
    live = np.log1p(-p_dead) - math.log1p(-parts["dead_prior"])
    h, a = parts["H"], parts["A"]
    if params["neutral_without_ball"]:
        h, a = np.where(parts["ball"], h, 0.0), np.where(parts["ball"], a, 0.0)
    return params["temperature"] * np.column_stack([live + h, live + a, d])


def viterbi_k(log_emission, log_transition, continuous):
    """Viterbi a K stati; dove `continuous` è False la transizione è uniforme."""
    e = np.asarray(log_emission, dtype=float)
    n, k = e.shape
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    lt = np.asarray(log_transition, dtype=float).tolist()
    free = math.log(1 / k)
    cont = np.asarray(continuous, dtype=bool).tolist()
    rows = e.tolist()
    back = np.zeros((n, k), dtype=np.int8)
    delta = rows[0][:]
    states = range(k)
    for i in range(1, n):
        new = [0.0] * k
        b = back[i]
        for j in states:
            if cont[i]:
                best, arg = -math.inf, 0
                for s in states:
                    v = delta[s] + lt[s][j]
                    if v > best:
                        best, arg = v, s
            else:
                arg = max(states, key=lambda s: delta[s])
                best = delta[arg] + free
            new[j] = best + rows[i][j]
            b[j] = arg
        top = max(new)
        delta = [v - top for v in new]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(delta))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i][path[i]]
    return path


def posteriors(log_emission, log_transition, continuous):
    """Forward-backward: P(stato | tutta la sequenza) per ogni frame, righe che sommano a 1."""
    e = np.asarray(log_emission, dtype=float)
    n, k = e.shape
    em = np.exp(e - e.max(axis=1, keepdims=True))
    t = np.exp(np.asarray(log_transition, dtype=float))
    uniform = np.full((k, k), 1 / k)
    cont = np.asarray(continuous, dtype=bool)
    alpha = np.zeros((n, k))
    alpha[0] = em[0] / em[0].sum()
    for i in range(1, n):
        a = (alpha[i - 1] @ (t if cont[i] else uniform)) * em[i]
        alpha[i] = a / a.sum()
    beta = np.ones((n, k))
    for i in range(n - 2, -1, -1):
        b = (t if cont[i + 1] else uniform) @ (em[i + 1] * beta[i + 1])
        beta[i] = b / b.sum()
    post = alpha * beta
    return post / post.sum(axis=1, keepdims=True)


def decode(parts, continuous, transitions, params):
    log_t = np.log(scaled_transitions(transitions, params["switch_scale"]))
    return STATES[viterbi_k(emissions(parts, params), log_t, continuous)]


def evaluate_dah(true, pred, tolerance_frames=125, fps=25, breaks=None):
    """Accuratezza su tutti i frame, D per frame, metriche H/A e metriche della challenge.

    Per i cambi di possesso i D predetti vengono sostituiti dall'ultimo H/A predetto;
    spellF1 e macroF1 usano invece le previsioni così come sono.
    """
    true, pred = np.asarray(true), np.asarray(pred)
    d_true, d_pred = true == "D", pred == "D"
    tp = int((d_true & d_pred).sum())
    precision = tp / d_pred.sum() if d_pred.any() else float("nan")
    recall = tp / d_true.sum() if d_true.any() else float("nan")
    filled = pd.Series(np.where(d_pred, None, pred)).ffill().bfill().to_numpy()
    result = {"accuratezza_tutti": float(np.mean(true == pred)),
              "D_precisione": precision, "D_recall": recall,
              "D_f1": 2 * precision * recall / (precision + recall) if tp else 0.0}
    result.update(evaluate(true, filled, tolerance_frames, fps, breaks))
    result.update(_challenge(true, pred, breaks))
    live = np.isin(true, HA)
    result["accuratezza_HA"] = float(np.mean(true[live] == pred[live])) if live.any() else float("nan")  # D = errore
    return result


def _parts_for(train, test):
    return evidence_parts((*fit_classifier(train), *fit_dead_classifier(train)), test)


def _grid(grid):
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]


def _live_missing(labels, features):
    """Frame a gioco vivo (etichetta non D) con la palla assente."""
    return (np.asarray(labels) != "D") & (features["ball_available"].to_numpy() == 0)


def false_dead_rate(labels, pred, features):
    """Quota dei frame a gioco vivo con palla assente predetti D; NaN se non ce ne sono."""
    missing = _live_missing(labels, features)
    return float(np.mean(np.asarray(pred)[missing] == "D")) if missing.any() else float("nan")


def select_params(games, grid=None, objective="spellF1", max_false_d=MAX_FALSE_D):
    """Parametri di decodifica in leave-one-game-out, fra quelli entro `max_false_d`.

    I falsi D si contano sulle partite lasciate fuori, tutte insieme. Se nessuna
    combinazione rispetta il limite, prende quella con meno falsi D e lo segnala
    con un warning invece di fermare la run.
    """
    grid = DEFAULT_GRID if grid is None else grid
    if len(games) < 2:
        raise ValueError("Servono almeno due partite per scegliere i parametri")
    names = list(dict.fromkeys(g.name for g in games))
    if len(names) < 2:
        raise ValueError("Servono almeno due partite diverse per scegliere i parametri")
    held_out = []
    for name in names:
        others = [g for g in games if g.name != name]
        game = next((g for g in games if g.name == name and g.primary), None)
        game = game or next(g for g in games if g.name == name)
        held_out.append((game, _parts_for(others, game), transition_matrix(others)))
    weights = [len(g.labels) for g, _, _ in held_out]
    missing = [_live_missing(g.labels, g.features) for g, _, _ in held_out]
    total_missing = sum(int(m.sum()) for m in missing)
    best, best_score = None, -math.inf
    fallback, fewest = None, math.inf
    for params in _grid(grid):
        preds = [decode(parts, g.continuous, t, params) for g, parts, t in held_out]
        false_d = sum(int(np.sum(p[m] == "D")) for p, m in zip(preds, missing))
        rate = false_d / total_missing if total_missing else 0.0
        if rate < fewest:
            fallback, fewest = params, rate
        if rate > max_false_d:
            continue
        results = [evaluate_dah(g.labels, p, breaks=_breaks(g.continuous)) for (g, _, _), p in zip(held_out, preds)]
        score = float(np.average([r[objective] for r in results], weights=weights))
        if score > best_score:
            best, best_score = params, score
    if best is None:
        warnings.warn(f"Nessuna combinazione entro il {max_false_d:.0%} di falsi D: "
                      f"scelta quella con meno falsi D ({fewest:.1%})")
        return fallback
    return best


def fit_dah_model(games, grid=None, objective="spellF1", max_false_d=MAX_FALSE_D):
    import sklearn
    params = select_params(games, grid, objective, max_false_d)
    return DahModel(*fit_classifier(games), *fit_dead_classifier(games), transition_matrix(games), params,
                    [g.name for g in games], sklearn.__version__)


def abstain(labels, post, threshold):
    """X dove la probabilità più alta del frame è sotto `threshold`; None = nessuna X.

    `labels`: etichette di Viterbi; `post`: probabilità (n, 3) da `posteriors`.
    Per lo spellF1 una X vale come l'ultima etichetta del tratto (all'inizio di un
    tratto, la prima che segue); per il macroF1 è un errore solo per la classe vera,
    mentre un'etichetta sbagliata penalizza due classi.
    """
    labels = np.asarray(labels).astype("<U1")
    if threshold is None:
        return labels
    if not 0 < threshold <= 1:
        raise ValueError("La soglia di astensione va in (0, 1]")
    return np.where(np.asarray(post).max(axis=1) < threshold, "X", labels)


def predict_dah(model, features, dead_features, continuous, probabilities=False, abstain_threshold="default"):
    """Etichette H/A/D per frame (X dove il modello è incerto, vedi `abstain`); con
    `probabilities` anche p_H, p_A, p_D (forward-backward).
    `abstain_threshold`: soglia di astensione; "default" = ABSTAIN_THRESHOLD."""
    threshold = ABSTAIN_THRESHOLD if abstain_threshold == "default" else abstain_threshold
    game = DahGame("run", features, dead_features, np.array([]), continuous)
    parts = evidence_parts((model.ha_classifier, model.ha_columns, model.ha_prior, model.dead_without,
                            model.dead_with), game)
    labels = decode(parts, continuous, model.transitions, model.params)
    post = None
    if probabilities or threshold is not None:
        log_t = np.log(scaled_transitions(model.transitions, model.params["switch_scale"]))
        post = posteriors(emissions(parts, model.params), log_t, continuous)
    result = pd.DataFrame({"frame": features["frame"].to_numpy(), "label": abstain(labels, post, threshold)})
    if probabilities:
        for j, state in enumerate(STATES):
            result[f"p_{state}"] = post[:, j]
    return result


def cross_validate(games, grid=None, objective="spellF1", extra=(), max_false_d=MAX_FALSE_D, abstain_thresholds=()):
    """Valutazione annidata: per ogni partita di test, parametri scelti sulle altre.

    `extra`: altre versioni sporcate delle stesse partite, usate solo per il
    training quando la loro partita non è quella di test.
    `abstain_thresholds`: soglie di astensione da provare. Il modello si allena una
    volta sola; per ogni soglia si aggiungono spellF1, macroF1 e quota di X ottenuti
    mettendo X sulle previsioni (colonne `spellF1_X0.5`, `macroF1_X0.5`, `quota_X0.5`).
    Le altre colonne restano quelle senza astensione.
    """
    if len(games) < 3:
        raise ValueError("La valutazione annidata richiede almeno tre partite")
    rows = []
    for test in games:
        train = [g for g in list(games) + list(extra) if g.name != test.name]
        model = fit_dah_model(train, grid, objective, max_false_d)
        result = predict_dah(model, test.features, test.dead_features, test.continuous,
                             probabilities=bool(abstain_thresholds), abstain_threshold=None)
        pred = result.label.to_numpy()
        breaks = _breaks(test.continuous)
        row = {"partita_test": test.name, **model.params, **evaluate_dah(test.labels, pred, breaks=breaks),
               "D_falsi_palla_assente_vivo": false_dead_rate(test.labels, pred, test.features)}
        if abstain_thresholds:
            post = result[[f"p_{s}" for s in STATES]].to_numpy()
            for threshold in abstain_thresholds:
                with_x = abstain(pred, post, threshold)
                scores = _challenge(test.labels, with_x, breaks)
                row[f"spellF1_X{threshold:g}"] = scores["spellF1"]
                row[f"macroF1_X{threshold:g}"] = scores["macroF1"]
                row[f"quota_X{threshold:g}"] = float(np.mean(with_x == "X"))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Valutazione leave-one-game-out H/A/D su partite Metrica")
    parser.add_argument("metrica_dir")
    parser.add_argument("--objective", default="spellF1",
                        choices=["spellF1", "macroF1", "accuratezza_tutti", "D_f1", "cambi_f1", "accuratezza_HA"])
    parser.add_argument("--peso-palla", type=float, metavar="W",
                        help=f"peso w del classificatore D con palla (predefinito {DEAD_BALL_WEIGHT}; 0 = non usarlo)")
    parser.add_argument("--max-falsi-d", type=float, default=MAX_FALSE_D,
                        help="quota massima di D sui frame vivi con palla assente (1 = nessun limite)")
    parser.add_argument("--senza-posizione-palla", action="store_true",
                        help="gioco fermo con i soli giocatori, senza ball_position_features (riferimento)")
    parser.add_argument("--extra", action="append", default=[],
                        help="cartella con altre versioni sporcate delle stesse partite (solo training)")
    parser.add_argument("--oriented", choices=["si", "no"], help="forza le feature orientate")
    parser.add_argument("--context", choices=["si", "no"], help="forza le feature di contesto")
    parser.add_argument("--astensione", type=lambda text: tuple(float(v) for v in text.split(",")), default=(),
                        metavar="S1,S2,...", help="soglie di astensione da provare, per esempio 0.4,0.5,0.6")
    parser.add_argument("--output", default="outputs/model_dah_cv.json")
    args = parser.parse_args(argv)
    options = {k: v == "si" for k, v in (("oriented", args.oriented), ("context", args.context)) if v}
    grid = DEFAULT_GRID if args.peso_palla is None else {**DEFAULT_GRID, "w": (args.peso_palla,)}
    position = not args.senza_posizione_palla
    extra = [g for folder in args.extra for g in load_dah_games(folder, primary=False, ball_position=position,
                                                                **options)]
    table = cross_validate(load_dah_games(args.metrica_dir, ball_position=position, **options), grid,
                           objective=args.objective, extra=extra, max_false_d=args.max_falsi_d,
                           abstain_thresholds=args.astensione)
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))
    if args.astensione:
        summary = pd.DataFrame([{"soglia": "nessuna", "spellF1": table.spellF1.mean(), "macroF1": table.macroF1.mean(),
                                 "quota_X": 0.0}]
                               + [{"soglia": f"{t:g}", "spellF1": table[f"spellF1_X{t:g}"].mean(),
                                   "macroF1": table[f"macroF1_X{t:g}"].mean(), "quota_X": table[f"quota_X{t:g}"].mean()}
                                  for t in args.astensione])
        print("\nAstensione, media sulle partite:")
        print(summary.round(4).to_string(index=False))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(table.to_dict(orient="records"), indent=2, default=str))
    print("Salvato in", args.output)


if __name__ == "__main__":
    main()
