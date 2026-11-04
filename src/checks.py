"""Controlli automatici strutturali, non misure di accuratezza."""
import pandas as pd


def check_tracking(tracking, expected_frames=None):
    """Riepiloga copertura e continuità senza confrontare output e video."""
    frames = tracking["frame"]
    count = len(tracking)
    return {
        "rows": count,
        "first_frame": int(frames.iloc[0]) if count else None,
        "last_frame": int(frames.iloc[-1]) if count else None,
        "duplicates": int(frames.duplicated().sum()),
        "consecutive_in_file": bool(frames.diff().iloc[1:].eq(1).all()) if count else False,
        "starts_at_zero": bool(count and frames.iloc[0] == 0),
        "expected_rows_ok": count == expected_frames if expected_frames is not None else None,
        "ball_missing_rows": int(tracking["ball_xy"].map(lambda p: p is None).sum()),
    }


def check_predictions(predictions, expected_frames=None, start_frame=0):
    """Controlla il contratto CSV, non la correttezza calcistica."""
    frames = predictions["frame"]
    count = len(predictions)
    return {"rows": count,
            "columns_ok": list(predictions.columns) == ["frame", "label"],
            "labels_ok": bool(predictions["label"].isin(["H", "A", "D", "X"]).all()),
            "full_index_ok": frames.tolist() == list(range(start_frame, start_frame + count)),
            "expected_rows_ok": count == expected_frames if expected_frames is not None else None,
            "label_counts": predictions["label"].value_counts().to_dict()}


def save_predictions(predictions, path, start_frame=0):
    """Salva solo output strutturalmente valido. Per la challenge attese 199319 righe."""
    from pathlib import Path
    result = check_predictions(predictions, start_frame=start_frame)
    if not (result["columns_ok"] and result["labels_ok"] and result["full_index_ok"]):
        raise ValueError("Output non valido: controllare colonne, etichette e indice")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
