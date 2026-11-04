# Possesso calcio

Una pipeline esplorativa per stimare la squadra in possesso a partire dal tracking tattico della challenge **More Than a Hack 2027**.

**Un notebook, tre moduli su cui lavorare:** palla mancante, tracce dei giocatori, caratteristiche e regole di possesso. Ogni trasformazione conserva gli input originali e registra cosa modifica.

> Stato attuale: baseline geometrica, senza machine learning. Sceglie la squadra più vicina alla palla oppure si astiene. Non riconosce ancora il gioco fermo né il controllo effettivo della palla. Le visualizzazioni descrivono le trasformazioni, non dimostrano accuratezza.

## 1. Avvio

Servono Python 3.10 o successivo e i materiali della challenge ottenuti con accesso autorizzato. La versione minima è una scelta di supporto del progetto; non tutte le versioni Python sono state verificate.

Aprire un terminale **nella cartella della repository**, poi:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook esplorazione.ipynb
```

La repository contiene il codice, non i materiali della partita. Copiare o collegare in `data/`:

```text
data/
├── BAR-ATH_tactical_atd.xml
└── BAR-ATH_tactical_atd.txt
```

Sui computer che usano collegamenti simbolici, i file originali devono rimanere disponibili. Ogni collaboratore deve configurare la propria cartella: i collegamenti locali non vengono pubblicati.

Per una verifica del solo codice, senza leggere materiali riservati:

```sh
python -m unittest discover -s tests
```

## 2. Come usare il notebook

Aprire `esplorazione.ipynb`, modificare **soltanto CONFIG**, quindi scegliere **Restart Kernel & Run All**. Dopo una modifica ai file Python, riavviare il kernel evita di usare vecchie funzioni rimaste in memoria.

Il notebook segue queste tappe:

| Tappa | Operazione | Cosa leggere nell'output |
|---|---|---|
| 1. Caricamento | Legge l'intervallo scelto e converte coordinate in metri | Fonte, quantità e indici dei frame |
| 2. Audit e preparazione | Verifica indice e assenze, prepara una copia di lavoro | Copertura della palla, conteggi delle tracce, anomalie strutturali |
| 3. Palla | Lascia i buchi oppure ricostruisce quelli ammessi | Quantità di stime e grafico di un buco con osservazioni/stime separate |
| 4. Giocatori | Mantiene tutto oppure applica il filtro grezzo | Conteggi prima/dopo e osservazioni rimosse sul campo |
| 5. Possesso | Calcola distanze e assegna H/A/X | Distanze usate, etichette, provenienza osservata/stimata |
| 6. Output | Controlla ed eventualmente salva la run | Validità del formato e cartella di destinazione |

Non occorre modificare le celle delle tappe per cambiare intervallo o attivare un modulo.

## 3. CONFIG: i comandi della run

| Parametro | Significato |
|---|---|
| `source` | `real` per il tracking della challenge; `synthetic` per verificare il codice senza materiali della partita |
| `start_frame` | Primo frame incluso, indice assoluto della registrazione |
| `end_frame` | Primo frame escluso; `None` arriva alla fine della sorgente |
| `ball_method` | `none`, `linear` oppure `velocity` |
| `ball_max_gap_s` | Durata massima del buco da ricostruire, in secondi |
| `player_policy` | `none` oppure `nearest22` |
| `show_plots` | Mostra o nasconde i grafici della run |
| `preview_frames` | Ampiezza massima dell'anteprima di un buco; non cambia la ricostruzione |
| `save_outputs` | Attiva il salvataggio locale |
| `run_name` | Nome della sottocartella in `outputs/`; stesso nome = sovrascrittura |
| `synthetic_duration_s` | Durata dei dati inventati, usata solo con `source="synthetic"` |

### Prima run consigliata: dieci minuti dall'inizio del gioco

Queste sono impostazioni per eseguire la pipeline, non parametri selezionati per massimizzare un punteggio:

```python
CONFIG = {
    "source": "real",
    "start_frame": 11185,
    "end_frame": 26185,
    "ball_method": "none",
    "ball_max_gap_s": 0.5,
    "player_policy": "none",
    "show_plots": True,
    "preview_frames": 250,
    "save_outputs": True,
    "run_name": "baseline_15000",
    "synthetic_duration_s": 120,
}
```

Il tracking è a 25 frame al secondo: 15.000 frame corrispondono a dieci minuti. La registrazione include la prepartita: partire da zero non significa partire dal calcio d'inizio. I grafici conservano i minuti della registrazione.

### Attivare un modulo

- Palla: impostare `ball_method="linear"` oppure `"velocity"`.
- Giocatori: impostare `player_policy="nearest22"`.
- Tutta la registrazione: `start_frame=0`, `end_frame=None`.
- Senza materiali reali: `source="synthetic"`, `start_frame=0`, `end_frame=None`.

Usare nomi diversi per conservare run diverse. I confronti possono descrivere l'effetto di una trasformazione; non autorizzano a scegliere soglie sulla partita fornita. **0,5 secondi è un limite sperimentale non calibrato né validato**, non una raccomandazione di accuratezza.

## 4. I tre moduli: cosa fanno e dove modificarli

### A. Palla mancante

File: `src/ball.py` e `src/reconstruction.py`.

- `none`: lascia sconosciute le posizioni mancanti.
- `linear`: collega le osservazioni prima e dopo un buco breve; usa anche informazioni future.
- `velocity`: prolunga il movimento ricavato dalle due osservazioni reali precedenti. L'implementazione è offline: controlla la durata dell'intero buco.

I buchi troppo lunghi restano sconosciuti. Non si attraversano salti nei frame o confini dei tempi. Sui materiali reali non si ricostruiscono prepartita, intervallo e dopopartita. Ai margini della run possono mancare le ancore necessarie.

Le colonne restano distinte:

| Colonna | Contenuto |
|---|---|
| `ball_xy` | Posizione originale osservata oppure assenza |
| `ball_estimated_xy` | Soltanto la posizione stimata |
| `ball_used_xy` | Posizione osservata, altrimenti stima disponibile |
| `ball_source` | `observed`, `linear`, `velocity` o `unknown` |
| `anchor_distance_s` | Distanza temporale dall'ancora pertinente; non è una probabilità di correttezza |

Per aggiungere un metodo: implementarlo in `reconstruction.py`, farlo selezionare da `ball.py` e aggiornare le opzioni accettate. Conservare sempre le colonne e i reset ai confini. Aggiungere un test sintetico che mostri sia un caso riuscito sia un limite del metodo.

### B. Tracce dei giocatori

File: `src/players.py`.

`nearest22` conserva al massimo le 22 tracce più vicine alla **palla osservata**, soltanto nei frame con più di 22. Se manca la palla, non filtra, anche quando esiste una stima. Non riempie tracce mancanti e non garantisce 11 per squadra.

È una prova grezza, non una deduplicazione di persone. Può eliminare giocatori reali lontani e conservare doppie tracce vicine. Con la sola regola del più vicino, spesso l'etichetta rimane invariata.

Per aggiungere una politica: estendere `filter_players()` mantenendo il ritorno `(tracking_modificato, registro)`. Non modificare gli input; registrare rimozioni, frame modificati e casi saltati. Aggiungere un test che verifichi anche cosa succede quando manca la palla.

### C. Caratteristiche e regole di possesso

File: `src/features.py` e `src/algorithm.py`.

`build_features()` produce una riga per frame con distanze minime, differenza delle distanze e provenienza della palla. `predict()` decide:

```text
Distanza casa < distanza ospiti → H
Distanza ospiti < distanza casa → A
Distanza assente oppure parità  → X
```

Una posizione stimata entra attualmente nel calcolo geometrico come una osservata, ma rimane identificata come stima. Non esiste ancora un peso di fiducia. D (gioco fermo) non viene prodotto da questa regola.

Per aggiungere una caratteristica: calcolarla in `build_features()`, documentare unità e comportamento sui mancanti, includerla nelle colonne restituite. Non rinominare le colonne utilizzate dagli altri moduli.

Per aggiungere una regola: implementarla in `algorithm.py`, mantenendo l'output `frame,label`, stessi frame e stesso ordine. Se la regola richiede parametri, passarli esplicitamente dalla CONFIG attraverso la chiamata nel notebook: non nasconderli in costanti dentro una cella. Le politiche alternative di possesso non sono ancora selezionabili da CONFIG: quel collegamento va aggiunto insieme alla nuova regola.

## 5. Come proporre una modifica

1. Aprire un branch dedicato, per esempio `codex/palla-interpolazione`.
2. Modificare il modulo Python interessato; il notebook deve restare una sequenza di chiamate brevi.
3. Scrivere un test con input inventati e comportamento atteso. Controllare anche assenze, parità e confini temporali quando pertinenti.
4. Eseguire `python -m unittest discover -s tests`.
5. Eseguire il notebook dall'inizio. Spiegare la trasformazione nel Markdown della tappa e mostrare il suo effetto sui grafici.
6. Ripulire tutti gli output del notebook e proporre una pull request con modifica, motivo, verifiche e limiti.

Per una nuova opzione aggiornare insieme CONFIG, funzione, spiegazione e test. Nessuna assegnazione personale predefinita: scegliere un modulo e proporre il contributo.

## 6. Dove sono i risultati

Con `save_outputs=True`:

```text
outputs/<run_name>/
├── predictions_slice.csv     # oppure submission_baseline.csv per tutta la registrazione reale
├── run.json                  # configurazione, audit e registro dei moduli
├── features.csv              # caratteristiche effettivamente usate
└── ball_provenance.csv       # coordinate originali, stimate, usate e provenienza
```

Una porzione conserva gli indici originali e **non è una submission completa**. Per tutta la registrazione reale servono 199.319 righe da zero. La validità del formato non misura la correttezza calcistica.

Audit locali separati dal notebook:

```sh
python -m src.report
python -m src.audit_players
```

I rapporti HTML sono in `outputs/sanity_check.html` e `outputs/player_audit/audit.html`.

## 7. Regole della challenge e condivisione

La mail dell'organizzazione del **3 novembre 2026** prevale sulle indicazioni precedenti:

- training e tuning dei parametri soltanto su dati esterni, anche per metodi a regole;
- niente annotazioni manuali della partita fornita o confronto delle previsioni con il video;
- niente upload dei materiali forniti ai servizi IA;
- ispezione dei dati e controlli automatici consentiti nei limiti chiariti dall'organizzazione.

Il notebook non legge il ground truth e non seleziona parametri sul punteggio pubblico. Le coordinate e gli output riservati restano locali. Dati e cartella `outputs/` sono esclusi da Git, ma gli **output dentro un notebook non vengono esclusi automaticamente**: cancellarli prima di committare.

Il codice ufficiale riutilizzato conserva attribuzione e licenza in `LICENSE-METRICA.txt`; vedere `ADATTAMENTI.md`. Questa licenza non autorizza a redistribuire tracking, video o altri materiali della partita.

## 8. Pubblicazione su GitHub e fork

La cartella di lavoro è già una repository Git locale. Per pubblicarla:

1. Creare su GitHub una repository vuota, per esempio `possesso-calcio`. Non generare README, .gitignore o licenza dal sito: questi file sono già presenti localmente.
2. Verificare che il notebook non contenga output riservati.
3. Nel terminale della cartella del progetto, eseguire i comandi seguenti, sostituendo `TUO_ACCOUNT` con il proprio account:

```sh
git add .gitignore README.md ADATTAMENTI.md LICENSE-METRICA.txt requirements.txt esplorazione.ipynb src tests data/README.md outputs/.gitkeep
git diff --cached --stat
git commit -m "Aggiunge pipeline esplorativa del possesso"
git branch -M main
git remote add origin https://github.com/TUO_ACCOUNT/possesso-calcio.git
git push -u origin main
```

Questi comandi descrivono il primo invio; non rieseguirli alla cieca se esistono già commit o un remote origin. L'autenticazione dipende dal metodo GitHub configurato sul computer.

Dopo la pubblicazione, chi contribuisce può fare **Fork**, clonare il proprio fork, configurare i propri dati locali e proporre modifiche tramite pull request. Per una repository privata, accesso e possibilità di fork dipendono dai permessi e dalle impostazioni dell'account/organizzazione.

Riferimento per la creazione: [documentazione GitHub](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository).
