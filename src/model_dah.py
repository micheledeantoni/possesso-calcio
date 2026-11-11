"""Modello a tre stati H/A/D: possesso dal modello H/A, gioco fermo senza palla.

Estende `src/model.py`:
- D vs gioco vivo: un secondo classificatore, addestrato su tutti i frame con le
  feature di `src/dead_ball_features.py`, che non usano la palla (opzione
  `dead_ball_uses_ball` per confronto). Nelle etichette Metrica D coincide con la
  palla assente: con la palla il modello imparerebbe quella scorciatoia.
- H vs A dato il gioco vivo: il classificatore di `src/model.py`.
- Evidenza per frame (log-probabilità meno log-prior):
    D:  log P(D)                    − log prior(D)
    H:  log(1 − P(D)) + log P(H|¬D) − log prior(H), e così A.
- Viterbi a tre stati per le etichette; forward-backward per le probabilità di ogni
  frame. Transizioni stimate dalle etichette Metrica, transizioni fra stati diversi
  moltiplicate per `switch_scale`; libere ai salti di frame e ai cambi di segmento.
- Temperatura, scala e emissioni H/A neutre senza palla scelte in leave-one-game-out
  sulle sole partite Metrica.

Valutazione da terminale:
    python -m src.model_dah data/metrica
"""
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

from src.arrays import load_metrica_npz
from src.team_features import build_team_features, feature_columns, continuity
from src.dead_ball_features import build_dead_ball_features, dead_ball_columns
from src.model import CLASSIFIER_PARAMS, EPSILON, LABELS as HA, fit_classifier, log_evidence, evaluate

STATES = np.array(["H", "A", "D"])
DEFAULT_GRID = {"temperature": (0.1, 0.2, 0.5),
                "switch_scale": (1.0, 2.0, 4.0, 8.0),
                "neutral_without_ball": (False, True)}


@dataclass
class DahGame:
    name: str
    features: pd.DataFrame
    dead_features: pd.DataFrame
    labels: np.ndarray
    continuous: np.ndarray


@dataclass
class DahModel:
    ha_classifier: object
    ha_columns: list
    ha_prior: dict
    dead_classifier: object
    dead_columns: list
    dead_prior: float
    transitions: np.ndarray
    params: dict
    dead_ball_uses_ball: bool = False
    training_games: list = field(default_factory=list)
    sklearn_version: str = ""


def load_dah_games(folder, ball_speed_max_ms=40.0):
    games = []
    for path in sorted(Path(folder).glob("*.npz")):
        arrays = load_metrica_npz(path)
        if arrays.labels is None:
            raise ValueError(f"{path.name}: mancano le etichette")
        games.append(DahGame(path.stem, build_team_features(arrays, ball_speed_max_ms=ball_speed_max_ms),
                             build_dead_ball_features(arrays), np.asarray(arrays.labels),
                             continuity(arrays.frames, arrays.segment)))
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


def fit_dead_classifier(games, uses_ball=False):
    from sklearn.ensemble import HistGradientBoostingClassifier
    columns = dead_ball_columns(games[0].dead_features)
    if uses_ball:
        columns += [c for c in feature_columns(games[0].features) if c != "ball_source"]
    X = pd.concat([_dead_matrix(g, columns, uses_ball) for g in games])
    y = np.concatenate([g.labels == "D" for g in games])
    columns = [c for c in columns if X[c].nunique(dropna=False) > 1]
    classifier = HistGradientBoostingClassifier(**CLASSIFIER_PARAMS).fit(X[columns], y)
    return classifier, columns, float(y.mean())


def evidence_parts(model_parts, game):
    """Componenti dell'evidenza: log-rapporti per D, ¬D, H|¬D, A|¬D."""
    ha_clf, ha_cols, ha_prior, d_clf, d_cols, d_prior, uses_ball = model_parts
    p_dead = d_clf.predict_proba(_dead_matrix(game, d_cols, uses_ball))[:, list(d_clf.classes_).index(True)]
    p_dead = np.clip(p_dead, EPSILON, 1 - EPSILON)
    ha = log_evidence(ha_clf, ha_cols, ha_prior, game.features)  # colonne H, A
    return {"D": np.log(p_dead) - math.log(d_prior),
            "live": np.log1p(-p_dead) - math.log1p(-d_prior),
            "H": ha[:, 0], "A": ha[:, 1],
            "ball": game.features["ball_available"].to_numpy() == 1}


def emissions(parts, params):
    h, a = parts["H"], parts["A"]
    if params["neutral_without_ball"]:
        h, a = np.where(parts["ball"], h, 0.0), np.where(parts["ball"], a, 0.0)
    return params["temperature"] * np.column_stack([parts["live"] + h, parts["live"] + a, parts["D"]])


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


def evaluate_dah(true, pred, tolerance_frames=125, fps=25):
    """Accuratezza su tutti i frame, D per frame, e metriche H/A di `src/model.evaluate`.

    Per i cambi di possesso i D predetti vengono sostituiti dall'ultimo H/A predetto.
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
    result.update(evaluate(true, filled, tolerance_frames, fps))
    live = np.isin(true, HA)
    result["accuratezza_HA"] = float(np.mean(true[live] == pred[live])) if live.any() else float("nan")  # D = errore
    return result


def _parts_for(train, test, uses_ball):
    ha = fit_classifier(train)
    dead = fit_dead_classifier(train, uses_ball)
    return evidence_parts((*ha, *dead, uses_ball), test)


def _grid(grid):
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]


def select_params(games, grid=None, objective="accuratezza_tutti", uses_ball=False):
    grid = DEFAULT_GRID if grid is None else grid
    if len(games) < 2:
        raise ValueError("Servono almeno due partite per scegliere i parametri")
    held_out = []
    for i, game in enumerate(games):
        others = games[:i] + games[i + 1:]
        held_out.append((game, _parts_for(others, game, uses_ball), transition_matrix(others)))
    best, best_score = None, -math.inf
    for params in _grid(grid):
        results = [evaluate_dah(g.labels, decode(parts, g.continuous, t, params)) for g, parts, t in held_out]
        weights = [len(g.labels) for g, _, _ in held_out]
        score = float(np.average([r[objective] for r in results], weights=weights))
        if score > best_score:
            best, best_score = params, score
    return best


def fit_dah_model(games, grid=None, objective="accuratezza_tutti", uses_ball=False):
    import sklearn
    params = select_params(games, grid, objective, uses_ball)
    ha_clf, ha_cols, ha_prior = fit_classifier(games)
    d_clf, d_cols, d_prior = fit_dead_classifier(games, uses_ball)
    return DahModel(ha_clf, ha_cols, ha_prior, d_clf, d_cols, d_prior, transition_matrix(games), params,
                    uses_ball, [g.name for g in games], sklearn.__version__)


def predict_dah(model, features, dead_features, continuous, probabilities=False):
    """Etichette H/A/D per frame; con `probabilities` anche p_H, p_A, p_D (forward-backward)."""
    game = DahGame("run", features, dead_features, np.array([]), continuous)
    parts = evidence_parts((model.ha_classifier, model.ha_columns, model.ha_prior, model.dead_classifier,
                            model.dead_columns, model.dead_prior, model.dead_ball_uses_ball), game)
    labels = decode(parts, continuous, model.transitions, model.params)
    result = pd.DataFrame({"frame": features["frame"].to_numpy(), "label": labels})
    if probabilities:
        log_t = np.log(scaled_transitions(model.transitions, model.params["switch_scale"]))
        post = posteriors(emissions(parts, model.params), log_t, continuous)
        for j, state in enumerate(STATES):
            result[f"p_{state}"] = post[:, j]
    return result


def cross_validate(games, grid=None, objective="accuratezza_tutti", uses_ball=False):
    """Valutazione annidata: per ogni partita di test, parametri scelti sulle altre."""
    if len(games) < 3:
        raise ValueError("La valutazione annidata richiede almeno tre partite")
    rows = []
    for i, test in enumerate(games):
        model = fit_dah_model(games[:i] + games[i + 1:], grid, objective, uses_ball)
        pred = predict_dah(model, test.features, test.dead_features, test.continuous).label.to_numpy()
        live_missing = (test.labels != "D") & (test.features.ball_available.to_numpy() == 0)
        rows.append({"partita_test": test.name, **model.params, **evaluate_dah(test.labels, pred),
                     "D_falsi_palla_assente_vivo": float(np.mean(pred[live_missing] == "D")) if live_missing.any() else float("nan")})
    return pd.DataFrame(rows)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Valutazione leave-one-game-out H/A/D su partite Metrica")
    parser.add_argument("metrica_dir")
    parser.add_argument("--objective", default="accuratezza_tutti",
                        choices=["accuratezza_tutti", "D_f1", "cambi_f1", "accuratezza_HA"])
    parser.add_argument("--dead-ball-uses-ball", action="store_true")
    parser.add_argument("--output", default="outputs/model_dah_cv.json")
    args = parser.parse_args(argv)
    table = cross_validate(load_dah_games(args.metrica_dir), objective=args.objective,
                           uses_ball=args.dead_ball_uses_ball)
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(table.to_dict(orient="records"), indent=2, default=str))
    print("Salvato in", args.output)


if __name__ == "__main__":
    main()
