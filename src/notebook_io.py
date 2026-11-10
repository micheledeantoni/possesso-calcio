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


def team_features_for_run(arrays, tracking, ball_speed_max_ms):
    """Feature per squadra sugli stessi frame e con la stessa palla della tappa 5.

    La palla è `ball_used_xy`: osservata, altrimenti stima del modulo palla. Le
    tracce sono quelle originali: il filtro giocatori lavora su liste senza slot
    e non si può riportare sugli array, quindi qui non è applicato.
    """
    frames = tracking.frame.to_numpy()
    if not np.array_equal(arrays.frames, frames):
        raise ValueError("Array e tracking della run non hanno gli stessi frame")
    ball = np.array([point if point is not None else (np.nan, np.nan) for point in tracking.ball_used_xy],
                    dtype=float).reshape(len(frames), 2)
    features = build_team_features(replace(arrays, ball=ball), ball_speed_max_ms=ball_speed_max_ms)
    features.insert(1, "ball_source", tracking.ball_source.to_numpy())
    columns = [c for c in feature_columns(features) if c != "ball_source"]
    log = {"frame": len(features), "tracce_negli_array": int(arrays.players.shape[1]),
           "feature": len(columns), "segmenti": sorted(int(s) for s in np.unique(arrays.segment)),
           "ball_speed_max_ms": ball_speed_max_ms,
           "anomalie_velocita_palla": int(features.ball_tracking_anomaly.sum()),
           "frame_con_palla_stimata": int(features.ball_source.isin(["linear", "velocity"]).sum()),
           "filtro_giocatori": "non applicato alle feature per squadra"}
    return features, log


def export_run(config, root, tracking, features, predictions, audit, logs, team_features=None):
    checks = check_predictions(predictions, len(tracking), config["start_frame"])
    if config["save_outputs"]:
        folder = save_run(config, root, predictions, audit, logs, checks)
        features.to_csv(folder / "features.csv", index=False)
        if team_features is not None:
            team_features.to_csv(folder / "team_features.csv", index=False)
        else:
            (folder / "team_features.csv").unlink(missing_ok=True)  # niente file di una run precedente
        tracking[["frame","ball_xy","ball_estimated_xy","ball_used_xy","ball_source","anchor_distance_s"]].to_csv(folder / "ball_provenance.csv", index=False)
        print("Salvato in:", folder.relative_to(root))
    else:
        print("Salvataggio disattivato in CONFIG.")
    return checks
