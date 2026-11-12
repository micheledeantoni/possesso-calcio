"""Esegue `esplorazione.ipynb` senza interfaccia, cambiando solo CONFIG.

Uso (dalla radice della repo):
    python -m src.run_notebook --full --run-name consegna
    python -m src.run_notebook --source synthetic --set possession_method='"model"'

`--full` imposta tutta la registrazione (start_frame=0, end_frame=None), il
salvataggio e niente grafici: è la run che produce `outputs/<run_name>/submission.csv`.
Il notebook eseguito, con tutti i suoi output, viene salvato in
`outputs/<run_name>/notebook_eseguito.ipynb`: resta locale (cartella esclusa da Git).

Sul terminale stampa solo lo stato e i controlli strutturali del CSV (righe, colonne,
etichette ammesse, indice), non i contenuti: i materiali della challenge e i loro
output non vanno condivisi con servizi esterni.
"""
import argparse
import json
from pathlib import Path


def main(argv=None):
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError
    parser = argparse.ArgumentParser(description="Esegue il notebook con una CONFIG modificata")
    parser.add_argument("--notebook", default="esplorazione.ipynb")
    parser.add_argument("--full", action="store_true", help="tutta la registrazione, salvataggio attivo, niente grafici")
    parser.add_argument("--source", choices=["real", "synthetic"])
    parser.add_argument("--run-name")
    parser.add_argument("--set", action="append", default=[], metavar="CHIAVE=VALORE_JSON",
                        help="altra voce di CONFIG, con il valore in JSON (es. show_plots=false)")
    args = parser.parse_args(argv)

    changes = {}
    if args.full:
        changes.update({"start_frame": 0, "end_frame": None, "save_outputs": True, "show_plots": False})
    if args.source:
        changes["source"] = args.source
    if args.run_name:
        changes["run_name"] = args.run_name
    for item in args.set:
        key, _, value = item.partition("=")
        changes[key.strip()] = json.loads(value)

    root = Path.cwd()
    nb = nbformat.read(root / args.notebook, as_version=4)
    config_cell = next(c for c in nb.cells if c.cell_type == "code" and "CONFIG = {" in c.source)
    config_cell.source += "\nCONFIG.update(" + repr(changes) + ")  # aggiunto da src/run_notebook.py\n"
    run_name = changes.get("run_name", "run_notebook")
    out = root / "outputs" / run_name
    out.mkdir(parents=True, exist_ok=True)
    print("Modifiche a CONFIG:", changes)
    print("Esecuzione in corso: può richiedere molti minuti sulla partita intera.")
    try:
        NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
        status = 0
    except CellExecutionError as error:
        message = str(error).strip().splitlines()
        print("ERRORE durante l'esecuzione:", message[-1][:300] if message else "sconosciuto")
        status = 1
    finally:
        nbformat.write(nb, out / "notebook_eseguito.ipynb")
        print("Notebook eseguito salvato in:", (out / "notebook_eseguito.ipynb").relative_to(root))
    for name in ("submission.csv", "predictions_slice.csv"):
        path = out / name
        if path.exists():
            import pandas as pd
            from src.checks import check_predictions
            checks = check_predictions(pd.read_csv(path))
            print(f"{path.relative_to(root)}: righe {checks['rows']}, colonne ok {checks['columns_ok']}, "
                  f"etichette ammesse {checks['labels_ok']}, indice completo da {int(pd.read_csv(path).frame.iloc[0])} "
                  f"{checks['full_index_ok'] if name == 'submission.csv' else '(porzione)'}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
