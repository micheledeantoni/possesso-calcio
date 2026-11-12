"""Controlli e salvataggio per mantenere il notebook breve."""
from dataclasses import replace
import numpy as np
from src.checks import check_tracking, check_predictions
from src.runs import save_run
from src.team_features import build_team_features, feature_columns


def audit_run(tracking, expected_rows, start_frame):
    report = check_tracking(tracking, expected_rows)
    if not (report["expected_rows_ok"] and report["consecutive_in_file"]
            and report["duplicates"] == 0 and report["first_frame"] == start_frame):
        raise ValueError("Audit strutturale fallito: nessuna correzione automatica dell'indice")
    return report


def team_features_for_run(arrays, tracking, ball_speed_max_ms, ball_interpolated=None):
    """Feature per squadra sugli stessi frame della run.

    Senza pulizia (`ball_interpolated=None`) la palla è quella della tappa 3,
    `ball_used_xy`, e le tracce sono quelle originali: il filtro giocatori della
    tappa 4 lavora su liste senza slot e non si riporta sugli array.
    Con la pulizia (`src/cleaning.py`) giocatori e palla sono quelli puliti, come
    nei dati Metrica di training; `ball_interpolated` segna le stime.
    """
    frames = tracking.frame.to_numpy()
    if not np.array_equal(arrays.frames, frames):
        raise ValueError("Array e tracking della run non hanno gli stessi frame")
    if ball_interpolated is None:
        ball = np.array([point if point is not None else (np.nan, np.nan) for point in tracking.ball_used_xy],
                        dtype=float).reshape(len(frames), 2)
        source = tracking.ball_source.to_numpy()
        arrays = replace(arrays, ball=ball)
    else:
        seen = np.isfinite(arrays.ball).all(axis=1)
        source = np.where(ball_interpolated, "linear", np.where(seen, "observed", "unknown"))
    features = build_team_features(arrays, ball_speed_max_ms=ball_speed_max_ms)
    features.insert(1, "ball_source", source)
    columns = [c for c in feature_columns(features) if c != "ball_source"]
    log = {"frame": len(features), "tracce_negli_array": int(arrays.players.shape[1]),
           "feature": len(columns), "segmenti": sorted(int(s) for s in np.unique(arrays.segment)),
           "ball_speed_max_ms": ball_speed_max_ms,
           "anomalie_velocita_palla": int(features.ball_tracking_anomaly.sum()),
           "frame_con_palla_stimata": int(features.ball_source.isin(["linear", "velocity"]).sum()),
           "giocatori_e_palla": ("puliti (src/cleaning.py)" if ball_interpolated is not None
                                 else "palla della tappa 3, tracce originali")}
    return features, log


def model_for_run(arrays, team_features, metrica_dir, ball_speed_max_ms, objective="cambi_f1"):
    """Addestra sui soli npz Metrica di `metrica_dir` e predice H/A sulla run.

    Le feature di training usano la stessa soglia palla della run. La partita
    della challenge non entra né nel training né nella scelta dei parametri.
    """
    from src.model import load_metrica_games, fit_possession_model, predict_possession
    from src.team_features import continuity
    if team_features is None:
        raise ValueError("possession_method='model' richiede team_features=True")
    games = load_metrica_games(metrica_dir, ball_speed_max_ms=ball_speed_max_ms)
    model = fit_possession_model(games, objective=objective)
    predictions = predict_possession(model, team_features, continuity(arrays.frames, arrays.segment))
    labels = predictions.label.to_numpy()
    log = {"partite_training": model.training_games, "obiettivo": objective,
           **{f"viterbi_{k}": v for k, v in model.params.items()},
           "probabilita_cambio_base": round(model.switch_probability, 6),
           "feature_usate": len(model.columns), "scikit_learn": model.sklearn_version,
           "cambi_predetti": int((labels[1:] != labels[:-1]).sum()),
           "etichette": predictions.label.value_counts().to_dict(),
           "D": "non prodotto: stadio separato da sviluppare"}
    return predictions, log


def model_dah_for_run(arrays, team_features, metrica_dir, ball_speed_max_ms, objective="spellF1"):
    """Modello a tre stati H/A/D (`src/model_dah.py`), addestrato sui soli npz Metrica.

    Il gioco fermo è stimato da feature che non usano la palla
    (`src/dead_ball_features.py`), calcolate sugli stessi array delle feature per squadra.
    Restituisce (etichette frame,label; probabilità p_H, p_A, p_D; registro).
    """
    from src.model_dah import load_dah_games, fit_dah_model, predict_dah
    from src.dead_ball_features import build_dead_ball_features
    from src.team_features import continuity
    if team_features is None:
        raise ValueError("possession_method='model_dah' richiede team_features=True")
    games = load_dah_games(metrica_dir, ball_speed_max_ms=ball_speed_max_ms)
    model = fit_dah_model(games, objective=objective)
    result = predict_dah(model, team_features, build_dead_ball_features(arrays),
                         continuity(arrays.frames, arrays.segment), probabilities=True)
    labels = result.label.to_numpy()
    log = {"partite_training": model.training_games, "obiettivo": objective,
           **{f"viterbi_{k}": v for k, v in model.params.items()},
           "feature_HA": len(model.ha_columns), "feature_gioco_fermo": len(model.dead_columns),
           "gioco_fermo_usa_palla": model.dead_ball_uses_ball, "scikit_learn": model.sklearn_version,
           "cambi_etichetta": int((labels[1:] != labels[:-1]).sum()),
           "etichette": result.label.value_counts().to_dict()}
    return result[["frame", "label"]], result[["frame", "p_H", "p_A", "p_D"]], log


def export_run(config, root, tracking, features, predictions, audit, logs, team_features=None, probabilities=None):
    checks = check_predictions(predictions, len(tracking), config["start_frame"])
    if config["save_outputs"]:
        folder = save_run(config, root, predictions, audit, logs, checks)
        features.to_csv(folder / "features.csv", index=False)
        if team_features is not None:
            team_features.to_csv(folder / "team_features.csv", index=False)
        else:
            (folder / "team_features.csv").unlink(missing_ok=True)  # niente file di una run precedente
        if probabilities is not None:
            probabilities.to_csv(folder / "probabilities.csv", index=False)
        else:
            (folder / "probabilities.csv").unlink(missing_ok=True)
        tracking[["frame","ball_xy","ball_estimated_xy","ball_used_xy","ball_source","anchor_distance_s"]].to_csv(folder / "ball_provenance.csv", index=False)
        print("Salvato in:", folder.relative_to(root))
    else:
        print("Salvataggio disattivato in CONFIG.")
    return checks
