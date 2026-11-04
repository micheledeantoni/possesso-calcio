"""Controlli e salvataggio per mantenere il notebook breve."""
from src.checks import check_tracking, check_predictions
from src.runs import save_run


def audit_run(tracking, expected_rows, start_frame):
    report = check_tracking(tracking, expected_rows)
    if not (report["expected_rows_ok"] and report["consecutive_in_file"]
            and report["duplicates"] == 0 and report["first_frame"] == start_frame):
        raise ValueError("Audit strutturale fallito: nessuna correzione automatica dell'indice")
    return report


def export_run(config, root, tracking, features, predictions, audit, logs):
    checks = check_predictions(predictions, len(tracking), config["start_frame"])
    if config["save_outputs"]:
        folder = save_run(config, root, predictions, audit, logs, checks)
        features.to_csv(folder / "features.csv", index=False)
        tracking[["frame","ball_xy","ball_estimated_xy","ball_used_xy","ball_source","anchor_distance_s"]].to_csv(folder / "ball_provenance.csv", index=False)
        print("Salvato in:", folder.relative_to(root))
    else:
        print("Salvataggio disattivato in CONFIG.")
    return checks
