# Possesso calcio

Una pipeline esplorativa per stimare la squadra in possesso a partire dal tracking tattico della challenge **More Than a Hack 2027**.

**Un notebook, tre moduli su cui lavorare:** palla mancante, tracce dei giocatori, caratteristiche e regole di possesso. Ogni trasformazione conserva gli input originali e registra cosa modifica.

> Stato attuale: due metodi selezionabili. La baseline geometrica sceglie la squadra più vicina alla palla oppure si astiene; il modello (classificatore + Viterbi) è addestrato soltanto sulle partite Metrica e assegna H/A a ogni frame. Non riconosce ancora il gioco fermo né il controllo effettivo della palla. Le visualizzazioni descrivono le trasformazioni, non dimostrano accuratezza.

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
| 6. Pulizia e feature per squadra | Rilegge le tracce come array, pulisce giocatori e palla, calcola le feature per frame | Registro della pulizia, riepilogo, grafico di alcune feature |
| 7. Modello | Con `possession_method="model"` (H/A) o `"model_dah"` (H/A/D), addestra su Metrica e sostituisce le etichette | Parametri scelti, cambi predetti, confronto con la regola |
| 8. Output | Controlla ed eventualmente salva la run | Validità del formato e cartella di destinazione |

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
| `team_features` | Calcola le feature per squadra della tappa 6, usate dal modello |
| `clean_tracking` | Pulisce giocatori e palla prima delle feature, con le stesse regole dei dati Metrica di training |
| `ball_speed_max_ms` | Velocità oltre la quale lo spostamento della palla è un errore del tracker; 40 m/s fissati a priori |
| `possession_method` | `nearest` (regola della tappa 5), `model` (H/A) oppure `model_dah` (H/A/D); i modelli richiedono `team_features=True` |
| `metrica_dir` | Cartella degli npz Metrica etichettati usati per addestrare il modello |
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
    "team_features": True,
    "clean_tracking": True,
    "ball_speed_max_ms": 40.0,
    "possession_method": "nearest",
    "metrica_dir": "data/metrica",
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

### D. Pulizia e feature per squadra (tracking ad array)

File: `src/arrays.py`, `src/cleaning.py` e `src/team_features.py`; tappa 6 del notebook, attivata da `team_features` e `clean_tracking`.

`arrays.py` legge il tracking conservando l'identità di traccia, necessaria per le velocità: `load_atd_arrays()` per la challenge, `load_metrica_npz()` per gli npz Metrica preparati dal gruppo (dati esterni, puliti o sporcati). Il formato comune `TrackingArrays` contiene `players (n, S, 2)`, `ball (n, 2)`, `team (n, S)` letto per frame e `segment`.

`build_team_features()` riprende le feature del gruppo (distanze ordinate, conteggi entro 2/5/10/15 m, velocità, baricentri, estensione delle squadre, movimento della palla) con le correzioni dell'audit in `docs/audit_drive/AUDIT.md`: niente feature legate a uno slot o alla rosa, squadra letta per frame, derivate che non attraversano salti di frame o cambi di `segment`, soglia di velocità della palla esplicita (`ball_speed_max_ms`, 40 m/s fissati a priori). Colonne `home_*`, `away_*`, `diff_*` = casa − ospiti; NaN = informazione assente.

`cleaning.py` riprende la pulizia del gruppo (script "Pulizia dati giocatori e palla"), con le stesse soglie fissate a priori. Per i giocatori: tracce spezzate oltre 12 m/s; cambio di squadra di una traccia solo se riduce gli oggetti oltre 11; nei frame con più di 11 oggetti, doppioni a meno di 1 m uniti e oggetti oltre 1 m fuori dal campo tolti, mai il più vicino alla palla. Per la palla: via i punti impossibili oltre 40 m/s; buchi fino a 1 s interpolati solo se fisicamente possibili. Differenze dall'originale: campo e soglie sono parametri (Metrica 105 m, BAR-ATH 105,3 m); i doppioni non vengono mediati, resta il punto della traccia più lunga, così la velocità non ha salti artificiali; tracce e interpolazione non attraversano i confini dei tempi. Sui tre match Metrica i risultati coincidono con l'originale, tranne i frame con doppioni (`docs/audit_drive/scripts/verify_cleaning.py`).

Nel notebook, con `clean_tracking=True`, le feature usano giocatori e palla puliti: le tappe 3 e 4 non vi entrano. Senza pulizia la palla è quella della tappa 3 (`ball_used_xy`) e le tracce restano originali, perché il filtro della tappa 4 lavora su liste senza slot. `ball_source` indica sempre le posizioni stimate. Le feature escludono le velocità oltre 12 m/s (`player_speed_max_ms`): uno slot che cambia persona. Le tracce vengono rilette dal file: su tutta la registrazione la tappa richiede qualche secondo in più.

Verifica sui dati Metrica: `docs/audit_drive/scripts/verify_team_features.py` .

### E. Modello di possesso (classificatore + Viterbi)

File: `src/model.py`; tappa 7 del notebook con `possession_method="model"`.

1. Un HistGradientBoosting stima per ogni frame la probabilità di H e A dalle feature per squadra. Iperparametri ripresi dal lavoro del gruppo, non regolati.
2. Viterbi a due stati sceglie la sequenza H/A più probabile. Emissioni = log-probabilità meno log-prior, moltiplicate per una temperatura; probabilità di cambio = tasso di cambi nelle etichette Metrica × una scala; transizione libera ai confini dei tempi e ai salti di frame; senza palla le emissioni possono diventare neutre.
3. Temperatura, scala e emissioni neutre sono scelte in leave-one-game-out sulle sole partite Metrica. Obiettivo predefinito: F1 dei cambi di possesso (`cambi_f1`); in alternativa l'accuratezza per frame.

Addestramento e scelta dei parametri usano soltanto gli npz in `metrica_dir`, a ogni run: nessun modello salvato, nessun dato della challenge. D non è prodotto: ogni frame riceve H o A.

Valutazione annidata sui tre match Metrica sporcati (parametri scelti sulle altre due partite; frame H/A; un cambio è corretto se va alla stessa squadra entro 5 s):

| Partita di test | Accuratezza per frame | Accuratezza Viterbi | Cambi reali / predetti | F1 cambi |
|---|---|---|---|---|
| 1 | 88,7% | 89,5% | 328 / 223 | 0,65 |
| 2 | 89,4% | 90,3% | 293 / 320 | 0,67 |
| 3 | 88,4% | 89,3% | 363 / 299 | 0,66 |

Senza smoothing il classificatore predice 1.300–1.650 cambi. Viterbi tende a perdere i possessi brevi: recall dei cambi 55–70%, precisione 64–81%. I parametri scelti variano fra le partite e la scala del cambio è spesso al bordo della griglia (4,0): la griglia va allargata. Risultati completi: `docs/audit_drive/model_cv.txt`; per ripeterli: `python -m src.model data/metrica`.

### F. Sporcatura dei dati Metrica (training set)

File: `src/dirtying.py`. Non fa parte del notebook: prepara i dati di training.

Riscrive le "Linee guida sporcatura" del gruppo, perché lo script originale non è disponibile. Le operazioni, applicate ai dati Metrica puliti con un seme fisso, sono sei:
- assenze di durata log-normale, più frequenti vicino alle linee laterali e alla palla; spesso un gruppo di vicini sparisce e ricompare insieme;
- identità spezzate in tracce su slot anonimi, con squadra fissa per slot;
- pezzi brevi di traccia nella squadra sbagliata;
- pochi oggetti che camminano oltre una linea laterale;
- salti di 0,6–3 m per un frame;
- 12 buchi della palla da 5–10 s a gioco vivo, lontano dalle interruzioni e nei contrasti.

Non aggiunge doppioni e non cambia le etichette. Tutte le quantità sono parametri di `DirtParams`.

I valori predefiniti riproducono le quantità dei file `Train set` del gruppo, misurate con `dirt_statistics` confrontando le posizioni con i dati puliti. Le quantità del `Train set` erano state tarate dal gruppo sulle statistiche del tracking BAR-ATH, uso confermato come consentito. Non sono gli stessi file, ma dati con le stesse caratteristiche: i dettagli stanno in `docs/audit_drive/dirtying_check.txt`.

Verifica incrociata del modello:

| Test sulla partita | Train su dati repo | Train su `Train set` |
|---|---|---|
| 1 (dati `Train set`) | 88,7% | 89,5% |
| 2 (dati `Train set`) | 90,6% | 90,3% |
| 3 (dati `Train set`) | 90,2% | 89,3% |

Per il modello le due sporcature sono equivalenti. I dati della repo risultano circa 1 punto più facili come test, quindi il `Train set` resta la stima più prudente.

Per generare i file (stesso formato del `Train set`, già ripuliti con `src/cleaning.py`):

```sh
python -m src.dirtying "<cartella npz Metrica puliti>" data/metrica --periods-from "<cartella Train set>"
```

`--periods-from` prende il periodo di gioco dai file del gruppo: i dati puliti non lo contengono.

### G. Modello a tre stati H/A/D (gioco fermo senza palla)

File: `src/dead_ball_features.py` e `src/model_dah.py`; tappa 7 del notebook con `possession_method="model_dah"`.

Nelle etichette Metrica D coincide con la palla assente, perché le etichette sono costruite dai buchi della palla. Nella challenge questo non è garantito: la palla sparisce anche a gioco vivo. Il modello a tre stati riconosce quindi il gioco fermo **senza usare la palla**, con un secondo classificatore che guarda solo i giocatori:
- movimento collettivo: velocità media e mediana, quota di giocatori lenti e veloci, andamento prima e dopo il frame;
- schieramenti delle riprese: calcio d'inizio, rimessa, angolo, rinvio, rigore, barriere ed esultanze;
- assetto delle squadre.

Il possesso viene dal classificatore H/A del §4E. Le due evidenze si combinano così:
- D: log P(D) − log prior(D);
- H e A: log(1 − P(D)) + log P(H|non D) − log prior(H), e allo stesso modo per A.

Viterbi a tre stati sceglie la sequenza di etichette, con transizioni stimate dalle etichette Metrica; forward-backward dà le probabilità di ogni frame, salvate in `probabilities.csv`. Temperatura e scala delle transizioni sono scelte in leave-one-game-out sulle sole partite Metrica. Per confronto, `python -m src.model_dah data/metrica --dead-ball-uses-ball` lascia usare anche la palla al classificatore del gioco fermo.

Valutazione annidata sui tre match Metrica sporcati (parametri scelti sulle altre due partite; tutti i frame):

| Partita di test | Accuratezza H/A/D | F1 di D | Falsi D nei buchi della palla a gioco vivo | Accuratezza H/A con palla per D | Falsi D con palla |
|---|---|---|---|---|---|
| 1 | 82,4% | 0,84 | 32% | 92,3% | 74% |
| 2 | 85,9% | 0,89 | 22% | 93,2% | 65% |
| 3 | 86,4% | 0,88 | 5% | 91,8% | 47% |

Usare la palla fa guadagnare circa 7 punti su Metrica, ma è la scorciatoia "palla assente = D": nei buchi della palla a gioco vivo il modello dice D il 47–74% delle volte, contro il 5–32% senza palla. Quale versione convenga dipende da quanto, nella challenge, la palla manca anche a gioco vivo: va valutato in locale, non su Metrica. In entrambe le versioni i cambi di possesso predetti sono pochi (155–215 contro 293–363 reali) e i parametri scelti finiscono ai bordi della griglia (temperatura 0,1, scala 8): la griglia va allargata. Risultati completi: `docs/audit_drive/model_dah_cv.txt`.

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
├── team_features.csv         # feature per squadra, se team_features=True
├── probabilities.csv         # p_H, p_A, p_D per frame, con possession_method="model_dah"
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
