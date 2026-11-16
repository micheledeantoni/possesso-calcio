
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "outputs" / "a7_risultati.csv"
EXCEL_FILE = BASE_DIR / "outputs" / "A7_tabella_completa.xlsx"

METRICHE = {
    "spellF1": "Spell F1",
    "D_f1": "D F1",
    "accuratezza_tutti": "Accuratezza complessiva",
    "falsi_D_palla_assente": "Falsi D con palla assente",
}

SCENARI = {
    "assenze_x1.5": ("Rilevamenti mancanti", "Alto"),
    "errori_squadra_x2": ("Errori di squadra", "Alto"),
    "buchi_palla_x3": ("Buchi della palla", "Alto"),
    "assenze_intermedio": ("Rilevamenti mancanti", "Intermedio"),
    "errori_squadra_intermedio": ("Errori di squadra", "Intermedio"),
    "buchi_palla_intermedio": ("Buchi della palla", "Intermedio"),
}


def main():
    print("INIZIO CREAZIONE EXCEL")

    if not CSV_FILE.exists():
        raise FileNotFoundError(f"File non trovato: {CSV_FILE}")

    print("Lettura dei risultati già calcolati...")
    dati = pd.read_csv(CSV_FILE)

    colonne_richieste = {
        "scenario", "partita", "usa_palla", *METRICHE.keys()
    }
    mancanti = colonne_richieste - set(dati.columns)

    if mancanti:
        raise ValueError(f"Colonne mancanti nel CSV: {mancanti}")

    dati["usa_palla"] = (
        dati["usa_palla"].astype(str).str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    if dati["usa_palla"].isna().any():
        raise ValueError("Valori non validi nella colonna usa_palla.")

    # Risultati dello scenario normale
    baseline = dati.loc[
        dati["scenario"] == "normale",
        ["partita", "usa_palla"] + list(METRICHE.keys())
    ].copy()

    if baseline.empty:
        raise ValueError("Scenario normale non trovato nel CSV.")

    chiavi = ["partita", "usa_palla"]

    if baseline.duplicated(chiavi).any():
        raise ValueError("Risultati normali duplicati.")

    baseline = baseline.rename(
        columns={m: f"riferimento_{m}" for m in METRICHE}
    )

    # Scenari alterati
    alterati = dati[dati["scenario"] != "normale"].copy()

    # Elimina eventuali vecchie colonne di confronto
    alterati = alterati.drop(
        columns=[
            c for c in alterati.columns
            if c.startswith("riferimento_") or c.endswith("_normale")
        ],
        errors="ignore"
    )

    risultati = alterati.merge(
        baseline,
        on=chiavi,
        how="left",
        validate="many_to_one"
    )

    colonne_riferimento = [
        f"riferimento_{m}" for m in METRICHE
    ]

    if risultati[colonne_riferimento].isna().any().any():
        raise ValueError(
            "Manca il riferimento normale per una o più partite."
        )

    righe = []

    for _, r in risultati.iterrows():
        scenario = r["scenario"]

        if scenario not in SCENARI:
            continue

        tipologia, intensita = SCENARI[scenario]
        modello = "Con palla" if r["usa_palla"] else "Senza palla"

        for colonna, nome_metrica in METRICHE.items():
            valore = r[colonna]
            riferimento = r[f"riferimento_{colonna}"]

            differenza = valore - riferimento

            percentuale = (
                differenza / abs(riferimento) * 100
                if riferimento != 0 else None
            )

            if colonna == "falsi_D_palla_assente":
                if differenza < 0:
                    valutazione = "Miglioramento"
                elif differenza > 0:
                    valutazione = "Peggioramento"
                else:
                    valutazione = "Invariato"
            else:
                if differenza > 0:
                    valutazione = "Miglioramento"
                elif differenza < 0:
                    valutazione = "Peggioramento"
                else:
                    valutazione = "Invariato"

            righe.append({
                "Scenario": scenario,
                "Tipologia di peggioramento": tipologia,
                "Intensità": intensita,
                "Modello": modello,
                "Metrica": nome_metrica,
                "Differenza": differenza,
                "Differenza %": percentuale,
                "Valutazione": valutazione,
            })

    tabella = pd.DataFrame(righe)

    if tabella.empty:
        raise ValueError("Nessun risultato da esportare.")

    tabella["Scenario"] = pd.Categorical(
        tabella["Scenario"],
        categories=list(SCENARI.keys()),
        ordered=True
    )

    tabella = tabella.sort_values(
        ["Scenario", "Modello", "Metrica"]
    ).reset_index(drop=True)

    EXCEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as writer:
        tabella.to_excel(
            writer,
            sheet_name="Confronto A7",
            index=False
        )

        foglio = writer.sheets["Confronto A7"]
        foglio.freeze_panes = "A2"
        foglio.auto_filter.ref = foglio.dimensions

        for colonna in foglio.columns:
            lettera = colonna[0].column_letter
            larghezza = max(
                len(str(c.value or "")) for c in colonna
            )
            foglio.column_dimensions[lettera].width = min(
                larghezza + 3, 38
            )

    print(f"\nExcel creato: {EXCEL_FILE}")
    print(f"Numero righe: {len(tabella)}")
    print(f"Numero colonne: {len(tabella.columns)}")
    print("Completato: nessun riaddestramento eseguito.")


if __name__ == "__main__":
    main()
