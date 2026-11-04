"""Audit locale separato dal notebook, senza stampare materiali riservati."""
from pathlib import Path
import html
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data import load_tracking
from src.players import player_counts, filter_players
from src.quality import HALVES


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs" / "player_audit"
    out.mkdir(parents=True, exist_ok=True)
    tracking = load_tracking(root / "data")
    before = player_counts(tracking)
    filtered, log = filter_players(tracking, "nearest22")
    after = player_counts(filtered)
    gameplay = np.logical_or.reduce([before.frame.between(a, b).to_numpy() for a, b in HALVES])
    report = {"filter_trial": log, "scopes": {}}
    for name, mask in [("intera_registrazione", np.ones(len(before), dtype=bool)), ("tempi_di_gioco", gameplay)]:
        part = before.loc[mask]
        report["scopes"][name] = {
            "frames": len(part), "min_tracks": int(part.total.min()), "max_tracks": int(part.total.max()),
            "median_tracks": float(part.total.median()), "frames_above_22": int(part.over_22.sum()),
            "frames_below_22": int(part.total.lt(22).sum()),
            "frames_home_above_11": int(part.home_over_11.sum()),
            "frames_away_above_11": int(part.away_over_11.sum()),
            "distribution_total": {str(k): int(v) for k, v in part.total.value_counts().sort_index().items()}}
    before.to_csv(out / "counts_before.csv", index=False)
    after.to_csv(out / "counts_after_nearest22.csv", index=False)
    (out / "audit.json").write_text(json.dumps(report, indent=2))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    distribution = before.loc[gameplay].total.value_counts().sort_index()
    axes[0].bar(distribution.index, distribution.values)
    axes[0].axvline(22, color="red", linestyle="--", label="22: riferimento")
    axes[0].set(xlabel="Tracce presenti", ylabel="Frame", title="Distribuzione nei tempi di gioco")
    axes[0].set_yscale("log"); axes[0].legend()
    seconds = before.frame // 25
    axes[1].plot(before.groupby(seconds).frame.first() / 1500, before.groupby(seconds).total.max(), label="Massimo originale al secondo")
    axes[1].plot(after.groupby(seconds).frame.first() / 1500, after.groupby(seconds).total.max(), label="Dopo filtro sperimentale", alpha=.7)
    axes[1].axhline(22, color="red", linestyle="--")
    axes[1].set(xlabel="Minuto di registrazione", ylabel="Tracce", title="Intera registrazione: prima e dopo")
    axes[1].legend()
    fig.savefig(out / "counts.png", dpi=120)
    intro = """<!doctype html><html lang="it"><meta charset="utf-8"><title>Audit delle tracce</title>
    <style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}pre{background:#f5f5f5;padding:20px;overflow:auto}img{max-width:100%}</style>
    <h1>Audit locale del numero di tracce</h1><p>Materiali riservati: non condividere questo rapporto con servizi IA.</p>
    <p>Si contano tracce, non giocatori identificati. Più di 22 è un segnale; meno di 22 può dipendere da copertura incompleta, espulsioni o altri fattori. Undici per squadra non è un vincolo garantito.</p>
    <p>La prova nearest22 conserva le 22 tracce più vicine alla palla osservata solo nei frame sovrappopolati. Non interviene quando la palla manca, non garantisce 11 tracce per squadra e non risolve le identità. Può conservare duplicati vicini e rimuovere persone reali lontane. Nessuna efficacia sul possesso è dimostrata; non si sceglie il metodo in base a questa partita.</p>
    <p>Con la sola regola del giocatore più vicino, conservare i 22 più vicini alla stessa palla non cambia l'etichetta quando entrambe le squadre rimangono rappresentate. Se una squadra sparisce dalla selezione, la regola prudente può invece astenersi.</p><pre>"""
    (out / "audit.html").write_text(intro + html.escape(json.dumps(report, indent=2)) + '</pre><img src="counts.png" alt="Numero di tracce"></html>')
    print("Audit locale creato: outputs/player_audit/audit.html. Nessun contenuto dei dati stampato.")


if __name__ == "__main__":
    main()
