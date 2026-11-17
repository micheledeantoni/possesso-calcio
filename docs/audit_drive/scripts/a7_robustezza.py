"""A7: quanto regge il modello consegnato a una sporcatura più cattiva di quella di training.

Idea e prima versione di Cesare (PR #8, script evaluate_a7.py e analisi_a7.py), qui
riscritta perché valuti il modello della consegna: per ogni partita di test il modello
si addestra sulle altre due esattamente come in `python -m src.model_dah` (versione
principale data/metrica + versioni in più), poi viene testato sulla stessa partita
sporcata in modo normale e in sei scenari più cattivi.

Gli scenari partono dalle stesse partite etichettate e dallo stesso seme della versione
principale (20): cambia solo la quantità indicata. Le intensità sono fissate a priori,
dalla scheda del lavoro A7: assenze × 1,5, errori di squadra × 2, buchi della palla × 3;
le intermedie a metà strada. Servono a capire dove il modello è fragile, non a regolarlo.

Uso (dalla radice della repo; prima i dati come in data/README.md):
    python docs/audit_drive/scripts/a7_robustezza.py --genera
    python docs/audit_drive/scripts/a7_robustezza.py            (scenari già generati)

Richiede circa un'ora: sei addestramenti (3 partite × con e senza classificatore D con
palla); il resto sono previsioni. Salva la tabella in outputs/a7_robustezza.csv.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import dirtying
from src.model import _breaks
from src.model_dah import (DEFAULT_GRID, DEAD_BALL_WEIGHT, load_dah_games, fit_dah_model, predict_dah,
                           evaluate_dah, false_dead_rate)

BASE = dirtying.DirtParams()
SEED = 20  # come la versione principale data/metrica
SCENARIOS = {
    "assenze_x1.25": {"absent_per_frame": BASE.absent_per_frame * 1.25},
    "assenze_x1.5": {"absent_per_frame": BASE.absent_per_frame * 1.5},
    "errori_squadra_x1.5": {"team_error_pieces_per_min": BASE.team_error_pieces_per_min * 1.5},
    "errori_squadra_x2": {"team_error_pieces_per_min": BASE.team_error_pieces_per_min * 2},
    "buchi_palla_x2": {"ball_gaps": BASE.ball_gaps * 2},
    "buchi_palla_x3": {"ball_gaps": BASE.ball_gaps * 3},
}
WEIGHTS = (DEAD_BALL_WEIGHT, 0.0)


def generate(labelled_dir, out_root):
    for name, change in SCENARIOS.items():
        items = [f"{key}={value:g}" for key, value in change.items()]
        print("Genero", name, items, flush=True)
        dirtying.main([str(labelled_dir), str(out_root / name), "--seed", str(SEED),
                       *[arg for item in items for arg in ("--param", item)]])


def evaluate(metrica_dir, extra_dirs, out_root):
    games = load_dah_games(metrica_dir)
    extra = [g for folder in extra_dirs for g in load_dah_games(folder, primary=False)]
    models = {}
    for test in games:
        train = [g for g in games + extra if g.name != test.name]
        for w in WEIGHTS:
            print(f"Addestro senza {test.name}, w = {w:g}", flush=True)
            models[test.name, w] = fit_dah_model(train, {**DEFAULT_GRID, "w": (w,)}, "spellF1")
    rows = []
    folders = {"normale": Path(metrica_dir), **{name: out_root / name for name in SCENARIOS}}
    for scenario, folder in folders.items():
        for test in load_dah_games(folder, primary=False):
            for w in WEIGHTS:
                pred = predict_dah(models[test.name, w], test.features, test.dead_features,
                                   test.continuous).label.to_numpy()
                r = evaluate_dah(test.labels, pred, breaks=_breaks(test.continuous))
                rows.append({"scenario": scenario, "partita": test.name.removeprefix("metrica_partita_")[0],
                             "w": w, "spellF1": r["spellF1"], "macroF1": r["macroF1"], "D_f1": r["D_f1"],
                             "falsi_D": false_dead_rate(test.labels, pred, test.features)})
        print("Valutato", scenario, flush=True)
    table = pd.DataFrame(rows)
    normal = table[table.scenario == "normale"].set_index(["partita", "w"]).spellF1
    table["delta_spellF1"] = table.spellF1 - [normal[p, w] for p, w in zip(table.partita, table.w)]
    return table


def summary(table):
    """Media sulle tre partite per scenario e peso, con la variazione rispetto al normale."""
    grouped = table.groupby(["scenario", "w"], sort=False)
    out = grouped[["spellF1", "delta_spellF1", "macroF1", "falsi_D"]].mean()
    out["partite_peggiorate"] = grouped.delta_spellF1.apply(lambda d: int((d < 0).sum()))
    return out.reset_index()


def main(argv=None):
    parser = argparse.ArgumentParser(description="A7: robustezza del modello consegnato a sporcature più cattive")
    parser.add_argument("--etichettati", default="data/metrica_etichettati")
    parser.add_argument("--metrica", default="data/metrica")
    parser.add_argument("--extra", action="append", default=None)
    parser.add_argument("--scenari", default="data/a7")
    parser.add_argument("--genera", action="store_true", help="genera prima le cartelle degli scenari")
    parser.add_argument("--output", default="outputs/a7_robustezza.csv")
    args = parser.parse_args(argv)
    out_root = Path(args.scenari)
    if args.genera:
        generate(args.etichettati, out_root)
    extra = args.extra or ["data/metrica_extra_1", "data/metrica_extra_2"]
    table = evaluate(args.metrica, extra, out_root)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    print("\nMedia sulle tre partite:")
    print(summary(table).round(4).to_string(index=False))
    print("Salvato in", args.output)


if __name__ == "__main__":
    main()
