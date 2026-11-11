"""Selezione delle run e salvataggio locale riproducibile."""
from pathlib import Path
import json
from src.data import load_tracking, visualization_demo
from src.checks import save_predictions


def load_run(config, root):
    """Seleziona intervallo assoluto [start_frame, end_frame), senza rinumerare."""
    start, end = config["start_frame"], config["end_frame"]
    if not isinstance(start, int) or start < 0 or (end is not None and (not isinstance(end, int) or end <= start)):
        raise ValueError("start_frame ed end_frame non validi")
    source = config["source"]
    if source == "real":
        tracking = load_tracking(Path(root) / "data", start_frame=start, end_frame=end)
        expected_end = 199319 if end is None else end
        if expected_end > 199319:
            raise ValueError("Intervallo oltre la registrazione prevista")
    elif source == "synthetic":
        tracking = visualization_demo(duration_s=config["synthetic_duration_s"])
        expected_end = len(tracking) if end is None else end
        if expected_end > len(tracking):
            raise ValueError("Intervallo oltre la demo: aumentare synthetic_duration_s")
        tracking = tracking[(tracking.frame >= start) & (tracking.frame < expected_end)].copy()
    else:
        raise ValueError("source deve essere synthetic oppure real")
    if tracking.empty:
        raise ValueError("Intervallo senza record")
    return tracking, expected_end - start


def load_run_arrays(config, root):
    """Stessa porzione di `load_run`, in array con le tracce separate.

    Reale: segmenti = tempi pubblicati (HALVES), 0 fuori. Demo: un solo segmento.
    """
    from src.arrays import load_atd_arrays, arrays_from_demo
    from src.quality import HALVES
    if config["source"] == "real":
        return load_atd_arrays(Path(root) / "data", start_frame=config["start_frame"],
                               end_frame=config["end_frame"], halves=HALVES)
    tracking, _ = load_run(config, root)
    return arrays_from_demo(tracking)


def save_run(config, root, predictions, audit, preparation, prediction_checks):
    """CSV parziale distinto da submission completa; salva anche configurazione e audit."""
    name = config["run_name"]
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("run_name deve essere un nome semplice, senza directory")
    if not (prediction_checks["columns_ok"] and prediction_checks["labels_ok"]
            and prediction_checks["full_index_ok"] and prediction_checks["expected_rows_ok"]):
        raise ValueError("Controlli delle predizioni non superati")
    out = Path(root) / "outputs" / name
    out.mkdir(parents=True, exist_ok=True)
    full = config["source"] == "real" and config["start_frame"] == 0 and len(predictions) == 199319
    save_predictions(predictions, out / ("submission_baseline.csv" if full else "predictions_slice.csv"),
                     start_frame=config["start_frame"])
    report = {"config": config, "audit_input": audit, "preparazione": preparation,
              "controlli_output": prediction_checks, "full_match_csv": full,
              "nota": ("Modello addestrato su Metrica + Viterbi, senza riconoscimento D"
                       if config.get("possession_method") == "model"
                       else "Regola di vicinanza, senza riconoscimento D o accuratezza validata")}
    (out / "run.json").write_text(json.dumps(report, indent=2))
    return out
