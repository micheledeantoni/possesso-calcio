"""Rapporto locale: non stampa dati della partita nel terminale."""
from pathlib import Path
import json
import html
import matplotlib
matplotlib.use("Agg")
from src.data import load_tracking
from src.checks import check_tracking
from src.quality import quality_timeline, quality_summary
from src.plots import plot_quality


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    tracking = load_tracking(root / "data")
    timeline = quality_timeline(tracking)
    report = {"struttura": check_tracking(tracking, 199319),
              "qualita": quality_summary(timeline)}
    (out / "sanity_check.json").write_text(json.dumps(report, indent=2))
    fig = plot_quality(timeline)
    fig.savefig(out / "sanity_check.png", dpi=130)
    content = """<!doctype html><html lang="it"><meta charset="utf-8">
    <title>Sanity check locale</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}pre{background:#f4f6f8;padding:20px;overflow:auto}img{max-width:100%}</style>
    <h1>Sanity check del tracking</h1><p>Rapporto locale sui materiali riservati: non condividere con servizi IA.</p>
    <p>Le coordinate coincidenti sono indizi, non duplicati di persone confermati. Più di 22 tracce può dipendere da duplicazioni o da altri problemi: non prova una causa specifica. Le osservazioni fuori dal campo non sono automaticamente errori. Nessuna osservazione è stata corretta o eliminata.</p>
    <p>Le squadre sono assegnate alle tracce dai metadati. Non possiamo confermare cambi di identità o di squadra di una persona con questo controllo. Per le assenze, confrontare i tempi di gioco con l'intera registrazione.</p><pre>""" + html.escape(json.dumps(report, indent=2)) + '</pre><img src="sanity_check.png" alt="Copertura e assenze del tracking"></html>'
    (out / "sanity_check.html").write_text(content)
    print("Rapporto locale creato in outputs/sanity_check.html; contenuti non stampati.")


if __name__ == "__main__":
    main()
