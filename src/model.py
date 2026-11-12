"""Modello di possesso H/A: classificatore per frame + smoothing di Viterbi.

Addestramento e scelta dei parametri soltanto su dati esterni (partite Metrica
etichettate, tipicamente le versioni sporcate). Nessun uso della partita della
challenge per training o tuning: su quella il modello si applica e basta.

1. Classificatore: HistGradientBoosting sulle feature per squadra, addestrato
   sui soli frame H/A. Iperparametri ripresi dal lavoro del gruppo, non regolati.
2. Emissioni: log p(classe | frame) − log prior(classe), moltiplicate per una
   temperatura < 1 perché frame vicini sono fortemente correlati. Senza palla le
   emissioni possono diventare neutre (`neutral_without_ball`).
3. Viterbi a due stati (H, A): probabilità di cambio per frame = tasso di cambi
   nelle etichette di training × `switch_scale`. A ogni salto di frame o cambio
   di segmento la transizione è libera (0,5).
4. Temperatura, `switch_scale` e `neutral_without_ball` si scelgono con
   leave-one-game-out sulle sole partite di training.

D non è prodotto: il riconoscimento della palla morta è uno stadio separato.

Valutazione da terminale (cartella con gli npz Metrica):
    python -m src.model data/metrica
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

LABELS = np.array(["H", "A"])
CLASSIFIER_PARAMS = dict(max_iter=150, learning_rate=0.1, max_leaf_nodes=15,
                         l2_regularization=1.0, random_state=42)
DEFAULT_GRID = {"temperature": (0.05, 0.1, 0.2, 0.5, 1.0),
                "switch_scale": (0.25, 0.5, 1.0, 2.0, 4.0),
                "neutral_without_ball": (False, True)}
EPSILON = 1e-6


@dataclass
class Game:
    """Una partita etichettata: feature per frame, etichette H/A/D, continuità."""
    name: str
    features: pd.DataFrame
    labels: np.ndarray
    continuous: np.ndarray


@dataclass
class PossessionModel:
    classifier: object
    columns: list
    prior: dict
    switch_probability: float
    params: dict
    training_games: list = field(default_factory=list)
    sklearn_version: str = ""


def load_metrica_games(folder, ball_speed_max_ms=40.0, **feature_options):
    """Tutti gli npz Metrica della cartella, in ordine di nome."""
    games = []
    for path in sorted(Path(folder).glob("*.npz")):
        arrays = load_metrica_npz(path)
        if arrays.labels is None:
            raise ValueError(f"{path.name}: mancano le etichette")
        games.append(Game(name=path.stem,
                          features=build_team_features(arrays, ball_speed_max_ms=ball_speed_max_ms, **feature_options),
                          labels=np.asarray(arrays.labels),
                          continuous=continuity(arrays.frames, arrays.segment)))
    if not games:
        raise ValueError(f"Nessun file .npz in {folder}")
    return games


def switch_rate(games):
    """Cambi H↔A (ignorando D) per frame complessivo delle partite."""
    changes = frames = 0
    for game in games:
        possession = game.labels[np.isin(game.labels, LABELS)]
        changes += int(np.sum(possession[1:] != possession[:-1]))
        frames += len(game.labels)
    if frames == 0 or changes == 0:
        raise ValueError("Impossibile stimare il tasso di cambio: nessun cambio H/A")
    return changes / frames


def fit_classifier(games):
    from sklearn.ensemble import HistGradientBoostingClassifier
    columns = [c for c in feature_columns(games[0].features) if c != "ball_source"]
    X = pd.concat([g.features.loc[np.isin(g.labels, LABELS), columns] for g in games])
    y = np.concatenate([g.labels[np.isin(g.labels, LABELS)] for g in games])
    # Colonne costanti nel training: nessuna informazione, e alcune versioni di
    # scikit-learn falliscono nel calcolo dei bin.
    columns = [c for c in columns if X[c].nunique(dropna=False) > 1]
    X = X[columns]
    classifier = HistGradientBoostingClassifier(**CLASSIFIER_PARAMS).fit(X, y)
    prior = {label: float(np.mean(y == label)) for label in LABELS}
    return classifier, columns, prior


def log_evidence(classifier, columns, prior, features):
    """log p(classe | frame) − log prior, colonne nell'ordine H, A."""
    proba = classifier.predict_proba(features[columns])
    order = [list(classifier.classes_).index(label) for label in LABELS]
    proba = np.clip(proba[:, order], EPSILON, 1.0)
    return np.log(proba) - np.log([prior[label] for label in LABELS])


def viterbi(log_emission, switch_probability, continuous):
    """Sequenza più probabile di stati 0 (H) / 1 (A).

    switch_probability: probabilità di cambio fra frame consecutivi. Dove
    `continuous` è False la transizione è libera (0,5).
    """
    log_emission = np.asarray(log_emission, dtype=float)
    n = len(log_emission)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    if not 0 < switch_probability < 1:
        raise ValueError("switch_probability deve essere fra 0 e 1, esclusi")
    s = np.where(np.asarray(continuous, dtype=bool), switch_probability, 0.5)
    stay, move = np.log1p(-s).tolist(), np.log(s).tolist()
    e0, e1 = log_emission[:, 0].tolist(), log_emission[:, 1].tolist()
    from_other_0, from_other_1 = [False] * n, [False] * n
    d0, d1 = e0[0], e1[0]
    for i in range(1, n):
        keep, cross = d0 + stay[i], d1 + move[i]
        if cross > keep:
            n0, from_other_0[i] = cross, True
        else:
            n0 = keep
        keep, cross = d1 + stay[i], d0 + move[i]
        if cross > keep:
            n1, from_other_1[i] = cross, True
        else:
            n1 = keep
        top = max(n0, n1)  # evita derive numeriche su sequenze lunghe
        d0, d1 = n0 - top + e0[i], n1 - top + e1[i]
    state = 0 if d0 >= d1 else 1
    path = np.empty(n, dtype=np.int8)
    path[-1] = state
    for i in range(n - 1, 0, -1):
        if (from_other_0 if state == 0 else from_other_1)[i]:
            state = 1 - state
        path[i - 1] = state
    return path


def decode(evidence, features, continuous, switch_probability, params):
    emission = params["temperature"] * evidence
    if params["neutral_without_ball"]:
        emission = np.where(features["ball_available"].to_numpy()[:, None] == 1, emission, 0.0)
    return LABELS[viterbi(emission, switch_probability * params["switch_scale"], continuous)]


def evaluate(true, pred, tolerance_frames=125, fps=25, breaks=None):
    """Metriche sui soli frame con etichetta reale H/A, più le metriche della challenge.

    spellF1 e macroF1 (`src/challenge_metrics.py`) usano tutti i frame, D compresi:
    un modello che non predice D li sbaglia tutti. `breaks`: True dove inizia un
    nuovo tratto (per esempio il secondo tempo).

    Cambi: calcolati sulla sequenza dei frame H/A (i D sono saltati). Un cambio
    predetto è corretto se va verso la stessa squadra entro `tolerance_frames`.
    """
    true, pred = np.asarray(true), np.asarray(pred)
    mask = np.isin(true, LABELS)
    frames = np.flatnonzero(mask)
    t, p = true[mask], pred[mask]
    def changes(seq):
        at = np.flatnonzero(seq[1:] != seq[:-1]) + 1
        return frames[at], seq[at]
    true_at, true_to = changes(t)
    pred_at, pred_to = changes(p)
    used = np.zeros(len(pred_at), dtype=bool)
    delays = []
    for frame, team in zip(true_at, true_to):
        candidates = np.flatnonzero(~used & (pred_to == team) & (np.abs(pred_at - frame) <= tolerance_frames))
        if len(candidates):
            best = candidates[np.argmin(np.abs(pred_at[candidates] - frame))]
            used[best] = True
            delays.append((pred_at[best] - frame) / fps)
    matched = len(delays)
    recall = matched / len(true_at) if len(true_at) else float("nan")
    precision = matched / len(pred_at) if len(pred_at) else float("nan")
    return {"frame_HA": int(mask.sum()),
            "accuratezza_HA": float(np.mean(t == p)) if mask.any() else float("nan"),
            "cambi_reali": int(len(true_at)), "cambi_predetti": int(len(pred_at)),
            "cambi_recall": recall, "cambi_precisione": precision,
            "cambi_f1": 2 * recall * precision / (recall + precision) if matched else 0.0,
            "ritardo_mediano_s": float(np.median(delays)) if delays else float("nan"),
            "ritardo_assoluto_mediano_s": float(np.median(np.abs(delays))) if delays else float("nan"),
            **_challenge(true, pred, breaks)}


def _challenge(true, pred, breaks):
    from src.challenge_metrics import challenge_scores
    scores = challenge_scores(true, pred, breaks)
    return {"spellF1": scores["spellF1"], "macroF1": scores["macroF1"],
            "spell_vere": scores["spell_vere"], "spell_previste": scores["spell_previste"]}


def _grid(grid):
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]


def select_params(games, grid=None, objective="cambi_f1"):
    """Leave-one-game-out sulle partite date: sceglie i parametri di Viterbi.

    Ogni partita è decodificata con un classificatore addestrato sulle altre.
    L'obiettivo è calcolato sommando i frame (o i cambi) di tutte le partite.
    """
    if len(games) < 2:
        raise ValueError("Servono almeno due partite per scegliere i parametri")
    grid = DEFAULT_GRID if grid is None else grid
    held_out = []
    for i, game in enumerate(games):
        others = games[:i] + games[i + 1:]
        classifier, columns, prior = fit_classifier(others)
        held_out.append((game, log_evidence(classifier, columns, prior, game.features), switch_rate(others)))
    best, best_score, table = None, -math.inf, []
    for params in _grid(grid):
        results = [evaluate(game.labels, decode(evidence, game.features, game.continuous, rate, params),
                            breaks=_breaks(game.continuous)) for game, evidence, rate in held_out]
        score = _pooled(results, objective)
        table.append({**params, objective: score})
        if score > best_score:
            best, best_score = params, score
    return best, pd.DataFrame(table)


def _pooled(results, objective):
    if objective == "accuratezza_HA":
        return sum(r["accuratezza_HA"] * r["frame_HA"] for r in results) / sum(r["frame_HA"] for r in results)
    if objective == "cambi_f1":
        true = sum(r["cambi_reali"] for r in results)
        pred = sum(r["cambi_predetti"] for r in results)
        hit = sum(r["cambi_recall"] * r["cambi_reali"] for r in results)
        return 2 * hit / (true + pred) if true + pred else 0.0
    if objective in ("spellF1", "macroF1"):
        weight = "spell_vere" if objective == "spellF1" else "frame_HA"
        return sum(r[objective] * r[weight] for r in results) / max(1, sum(r[weight] for r in results))
    raise ValueError("objective: accuratezza_HA, cambi_f1, spellF1 oppure macroF1")


def _breaks(continuous):
    """Nuovo tratto dove i frame non sono continui (salti o cambio di tempo)."""
    breaks = ~np.asarray(continuous, dtype=bool)
    if len(breaks):
        breaks[0] = False
    return breaks


def fit_possession_model(games, grid=None, objective="cambi_f1"):
    """Parametri scelti in leave-one-game-out, poi classificatore su tutte le partite."""
    import sklearn
    params, _ = select_params(games, grid, objective)
    classifier, columns, prior = fit_classifier(games)
    return PossessionModel(classifier=classifier, columns=columns, prior=prior,
                           switch_probability=switch_rate(games), params=params,
                           training_games=[g.name for g in games], sklearn_version=sklearn.__version__)


def predict_possession(model, features, continuous):
    """Etichette H/A per ogni frame di `features`, nello stesso ordine."""
    missing = [c for c in model.columns if c not in features]
    if missing:
        raise ValueError(f"Feature mancanti: {missing[:5]}")
    evidence = log_evidence(model.classifier, model.columns, model.prior, features)
    labels = decode(evidence, features, continuous, model.switch_probability, model.params)
    return pd.DataFrame({"frame": features["frame"].to_numpy(), "label": labels})


def cross_validate(games, grid=None, objective="cambi_f1"):
    """Valutazione annidata: per ogni partita di test, parametri scelti sulle altre."""
    if len(games) < 3:
        raise ValueError("La valutazione annidata richiede almeno tre partite")
    rows = []
    for i, test in enumerate(games):
        model = fit_possession_model(games[:i] + games[i + 1:], grid, objective)
        evidence = log_evidence(model.classifier, model.columns, model.prior, test.features)
        raw = LABELS[np.argmax(evidence, axis=1)]
        smoothed = predict_possession(model, test.features, test.continuous).label.to_numpy()
        for method, pred in (("per_frame", raw), ("viterbi", smoothed)):
            rows.append({"partita_test": test.name, "metodo": method,
                         **{k: v for k, v in model.params.items()},
                         **evaluate(test.labels, pred, breaks=_breaks(test.continuous))})
    return pd.DataFrame(rows)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Valutazione leave-one-game-out su partite Metrica")
    parser.add_argument("metrica_dir")
    parser.add_argument("--objective", default="cambi_f1", choices=["accuratezza_HA", "cambi_f1", "spellF1", "macroF1"])
    parser.add_argument("--output", default="outputs/model_cv.json")
    args = parser.parse_args(argv)
    games = load_metrica_games(args.metrica_dir)
    table = cross_validate(games, objective=args.objective)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(table.to_dict(orient="records"), indent=2, default=str))
    print("Salvato in", args.output)


if __name__ == "__main__":
    main()
