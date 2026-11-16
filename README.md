# Possesso calcio

Una pipeline esplorativa per stimare la squadra in possesso a partire dal tracking tattico della challenge **More Than a Hack 2027**.

**Un notebook, tre moduli su cui lavorare:** palla mancante, tracce dei giocatori, caratteristiche e regole di possesso. Ogni trasformazione conserva gli input originali e registra cosa modifica.

> Stato attuale (16 novembre 2026): tre metodi selezionabili. La baseline geometrica sceglie la squadra più vicina alla palla oppure si astiene; `model` (classificatore + Viterbi) assegna H/A a ogni frame; `model_dah`, il predefinito per la consegna, aggiunge il gioco fermo D. I modelli sono addestrati soltanto sulle partite Metrica: su Metrica `model_dah` fa 0,708 di spellF1 (§4H), che **non è una previsione dello score**. Tutti i dati di training si rigenerano dai dati pubblici Metrica con il codice della repo (§4F). Le visualizzazioni descrivono le trasformazioni, non dimostrano accuratezza.

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

### Consegna: tutta la partita, senza aprire il notebook

```sh
python -m src.run_notebook --full --run-name consegna
```

Esegue il notebook su tutta la registrazione, con il salvataggio attivo e senza grafici, usando il metodo predefinito: `model_dah`, con le leve di stabilità e le sporcature in più. Produce `outputs/consegna/submission.csv`, cioè 199.319 righe `frame,label`, e salva il notebook eseguito in `outputs/consegna/notebook_eseguito.ipynb`. Sul terminale stampa solo lo stato e i controlli di formato. Servono i file della partita in `data/` e gli npz Metrica in `data/metrica`, `data/metrica_extra_1` e `data/metrica_extra_2`, che si generano dai dati pubblici con tre comandi (vedi `data/README.md`). Sulla partita intera richiede circa 10 minuti, quasi tutti di addestramento su Metrica, e quasi tutta la RAM di un computer da 16 GB (0,3 GB liberi al minimo): chiudere gli altri programmi prima di lanciarla.

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

Due opzioni sono attive di default perché migliorano lo spellF1 su Metrica (`docs/audit_drive/stability_experiments.txt`):
- **feature orientate** (`ORIENTED`): baricentri lungo x, spostamenti e direzione della palla sono riferiti al verso d'attacco, ricavato in ogni tempo dalla posizione delle due squadre. In più ci sono `ball_x_oriented` e `ball_y_oriented`, dove x positivo vuol dire verso la porta degli ospiti. Così le feature hanno lo stesso significato nei due tempi e in tutte le partite;
- **feature di contesto** (`CONTEXT`, colonne `ctx_*`): quota di frame in cui la casa ha il giocatore più vicino alla palla, medie della differenza di distanza e dei giocatori entro 5 m su ±0,5, ±1 e ±2 s, la stessa quota nel secondo prima e in quello dopo, e da quanti secondi la stessa squadra ha il giocatore più vicino. Le finestre non escono dal tempo in corso.

Sei gruppi di **feature di possesso** (lavoro A5, `add_possession_extras`), tutti riferiti alla propria porta di ogni squadra:
- `portieri`: profondità del giocatore più arretrato;
- `dietro`: giocatori dietro la palla, in pari e davanti;
- `packing`: come cambia, nel secondo prima e in quello dopo, il numero di giocatori dietro la palla;
- `linee`: compagni sulla stessa linea di profondità;
- `corsie`: corsie verticali occupate, su 5;
- `direzione`: quanto è vicino alla traiettoria della palla il miglior ricevitore di ogni squadra, e quanti giocatori stanno nel cono di ±15°.

Sono attivi tutti per default. Per gli esperimenti la variabile d'ambiente `POSSESSO_FEATURES` sceglie quali usare, per esempio `portieri,dietro` oppure `nessuna`. Il registro della run (`gruppi_feature_possesso` in `run.json`) dice quali erano attivi: controllarlo, perché una variabile rimasta impostata cambia il modello senza avvisare.

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

### F. Etichette e sporcatura dei dati Metrica (training set)

File: `src/metrica_labels.py` e `src/dirtying.py`. Non fanno parte del notebook: preparano i dati di training.

Le etichette vengono dalle tre partite di esempio pubbliche di Metrica ([metrica-sports/sample-data](https://github.com/metrica-sports/sample-data), cartella `data/`, circa 180 MB). `src/metrica_labels.py` riprende lo script del gruppo con le stesse regole:
- **D** è un buco della palla nel tracking spiegato da un'interruzione (palla fuori, fallo, tiro o cartellino, da 3 s prima a 1 s dopo l'inizio del buco) oppure da una ripresa (calcio piazzato o rinvio dal fondo entro 1 s dalla fine del buco);
- **H/A** è la squadra dell'ultimo evento di possesso (passaggio, conduzione, recupero, calcio piazzato, tiro).

Rigenerate dai dati pubblici, le etichette e i tempi di gioco coincidono frame per frame con quelli dei file `Train set` del gruppo (`docs/audit_drive/b1_etichette.txt`).

```sh
python -m src.metrica_labels data/metrica_sample data/metrica_etichettati
```

Le partite etichettate sono poi sporcate e ripulite da `src/dirtying.py`.

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

Per il modello le due sporcature sono equivalenti.

Dal 16 novembre **tutti i dati di training sono generati dalla repo**, con tre semi fissati prima di guardare i risultati: 20 per la versione principale, 0 e 10 per le due in più. Partendo dalle partite etichettate qui sopra, che contengono già il tempo di gioco:

```sh
python -m src.dirtying data/metrica_etichettati data/metrica --seed 20
python -m src.dirtying data/metrica_etichettati data/metrica_extra_1 --seed 0
python -m src.dirtying data/metrica_etichettati data/metrica_extra_2 --seed 10
```

I file sono nello stesso formato del `Train set`, già ripuliti con `src/cleaning.py`. Il `Train set` del gruppo non serve più: con la versione principale generata così lo spellF1 medio passa da 0,713 a 0,708 (§4H, `docs/audit_drive/rigenerazione_dati.txt`).

`--periods-from "<cartella Train set>"` serve solo con gli npz etichettati dalla versione originale dello script, che non salvava il tempo di gioco.

### G. Modello a tre stati H/A/D (gioco fermo)

File: `src/dead_ball_features.py` e `src/model_dah.py`; tappa 7 del notebook con `possession_method="model_dah"`.

Nelle etichette Metrica D coincide con la palla assente, perché le etichette sono costruite dai buchi della palla. Nella challenge questo non è garantito: la palla sparisce anche a gioco vivo. Il gioco fermo viene quindi da due classificatori.

Il **classificatore senza palla generica** guarda i giocatori e, quando si vede, la palla:
- movimento collettivo: velocità media e mediana, quota di giocatori lenti e veloci, andamento prima e dopo il frame;
- schieramenti delle riprese: calcio d'inizio, rimessa, angolo, rinvio, rigore, barriere ed esultanze;
- assetto delle squadre;
- posizione della palla (lavoro A1, `ball_position_features`): fuori dalle linee, ferma, ferma su un punto di ripresa, e i bordi delle pause, cioè palla vista uscire e non ancora rientrata, oppure palla che dopo un buco ricompare su un punto di ripresa. A palla assente queste feature valgono 0: l'assenza della palla non è un indizio.

Il **classificatore con palla** usa le stesse feature più tutte quelle per squadra, compresa la disponibilità della palla. Su Metrica impara quindi anche la scorciatoia "palla assente ⇒ D".

I due si combinano con P(D) = w · P(con palla) + (1 − w) · P(senza palla), dove w = 0,34 (`DEAD_BALL_WEIGHT`) è fissato a mano, non scelto in validazione. Per frenare la scorciatoia, nella scelta dei parametri sono escluse le combinazioni che dicono D in più del 30% (`MAX_FALSE_D`) dei frame a gioco vivo con palla assente delle partite di validazione. Se nessuna rispetta il limite, la run non si ferma: un warning segnala la scelta della combinazione con meno falsi D.

Il possesso viene dal classificatore H/A del §4E. Le due evidenze si combinano così:
- D: log P(D) − log prior(D);
- H e A: log(1 − P(D)) + log P(H|non D) − log prior(H), e allo stesso modo per A.

Viterbi a tre stati sceglie la sequenza di etichette, con transizioni stimate dalle etichette Metrica; forward-backward dà le probabilità di ogni frame, salvate in `probabilities.csv`. Temperatura e scala delle transizioni sono scelte in leave-one-game-out sulle sole partite Metrica, con obiettivo predefinito lo spellF1, la metrica di classifica (§4H).

Per confronto:
- `python -m src.model_dah data/metrica --peso-palla 0` usa solo il classificatore senza palla;
- `--max-falsi-d 1` toglie il limite sui falsi D;
- `--senza-posizione-palla` toglie le feature di palla di A1.

La tabella seguente è **storica**: è la prima versione, con un solo classificatore D, valutata con obiettivo accuratezza prima dello spellF1. I numeri attuali sono nel §4H e in `docs/audit_drive/merge_a1_a5.txt`.

| Partita di test | Accuratezza H/A/D, D senza palla | F1 di D | Falsi D nei buchi a gioco vivo | Accuratezza H/A/D, D con palla | Falsi D, con palla |
|---|---|---|---|---|---|
| 1 | 82,4% | 0,84 | 32% | 92,3% | 74% |
| 2 | 85,9% | 0,89 | 22% | 93,2% | 65% |
| 3 | 86,4% | 0,88 | 5% | 91,8% | 47% |

Usare la palla fa guadagnare circa 7 punti su Metrica, ma è la scorciatoia "palla assente = D": nei buchi della palla a gioco vivo il modello dice D il 47–74% delle volte, contro il 5–32% senza palla. Quale versione convenga dipende da quanto, nella challenge, la palla manca anche a gioco vivo: va valutato in locale, non su Metrica. In entrambe le versioni i cambi di possesso predetti sono pochi (155–215 contro 293–363 reali) e i parametri scelti finiscono ai bordi della griglia (temperatura 0,1, scala 8): la griglia va allargata. Risultati completi: `docs/audit_drive/model_dah_cv.txt`.

### H. Metrica della challenge (spellF1, macroF1)

File: `src/challenge_metrics.py`, reimplementato dalla specifica di `evaluation/possession_metrics.py` della repo pubblica della challenge (licenza MIT).
- **spellF1** è la metrica di classifica. Un possesso è un tratto continuo di H o di A: D lo interrompe e non è mai un possesso. Possessi veri e previsti della stessa squadra vengono abbinati per sovrapposizione temporale (IoU); l'F1 si calcola alle soglie 0,3–0,7 e se ne fa la media. Le X previste sono riempite con l'ultima etichetta data.
- **macroF1**, il primo spareggio, è la media dell'F1 per frame di H, A e D.

`src/model.py` e `src/model_dah.py` riportano entrambe le metriche. Per il modello H/A/D l'obiettivo predefinito della scelta dei parametri è lo spellF1. Il calcolo usa solo le etichette Metrica: la soluzione della challenge non viene mai letta.

Sui tre match Metrica sporcati (valutazione annidata; baseline ufficiale con finestra ±50, non scelta qui):

| | Partita 1 | Partita 2 | Partita 3 |
|---|---|---|---|
| Baseline ufficiale, spellF1 | 0,371 | 0,352 | 0,326 |
| `model` (solo H/A), spellF1 | 0,387 | 0,449 | 0,394 |
| `model_dah`, D senza palla, spellF1, prima delle leve di stabilità | 0,553 | 0,558 | 0,509 |
| **`model_dah`, D senza palla, spellF1, con le leve (predefinito)** | **0,623** | **0,608** | **0,579** |
| `model_dah`, D con palla, spellF1, prima delle leve | 0,643 | 0,649 | 0,598 |
| `model_dah`, D senza palla, macroF1, prima delle leve | 0,819 | 0,856 | 0,863 |
| **`model_dah`, D senza palla, macroF1, con le leve** | **0,834** | **0,868** | **0,881** |
| `model_dah`, D con palla, macroF1, prima delle leve | 0,920 | 0,926 | 0,919 |

Il lavoro successivo, con la stessa valutazione, le leve e due sporcature in più (`docs/audit_drive/a3_regolarizzazione.txt` e `merge_a1_a5.txt`):

| `model_dah`, spellF1 | Partita 1 | Partita 2 | Partita 3 | Media | Falsi D nei buchi a gioco vivo |
|---|---|---|---|---|---|
| A3: classificatori regolarizzati | 0,637 | 0,611 | 0,601 | 0,616 | 20,0% |
| + A1: posizione della palla per D | 0,662 | 0,648 | 0,642 | 0,651 | 21,5% |
| + A5: feature di possesso | 0,687 | 0,678 | 0,672 | 0,679 | 21,8% |
| + classificatore D con palla, w = 0,34 | 0,722 | 0,732 | 0,685 | 0,713 | 24,8% |

Questi numeri usano come versione principale il `Train set` del gruppo. Con tutti i dati generati dalla repo (§4F), che è la configurazione attuale:

| `model_dah`, dati generati dalla repo | Partita 1 | Partita 2 | Partita 3 | Media | Falsi D nei buchi a gioco vivo |
|---|---|---|---|---|---|
| **predefinito, w = 0,34** | **0,748** | **0,736** | **0,640** | **0,708** | 22,7% |
| senza classificatore con palla, w = 0 | 0,708 | 0,680 | 0,670 | 0,686 | 21,9% |

macroF1 del predefinito: 0,920, 0,939, 0,926. Il classificatore con palla guadagna in media, ma in parte con la scorciatoia "palla assente ⇒ D", e non in tutte le partite: sulla partita 3 perde 0,03, perché lì i parametri scelti finiscono al bordo della griglia (scala 1). Per rifare il confronto: `--peso-palla 0`.

Le leve di stabilità sono tre: feature orientate, feature di contesto (§4D) e due versioni sporcate in più per partita nel training. L'effetto di ciascuna, misurata da sola:
- feature orientate: +0,051 di spellF1 medio;
- contesto: +0,006, cioè rumore;
- sporcature in più: +0,023.

Tutte e tre insieme danno +0,063, con un miglioramento in ogni partita. Nel notebook le sporcature in più si attivano con `metrica_extra_dirs` (vedi `data/README.md`). La versione con la palla per D non è stata rivalutata con le leve.

Sulla partita della challenge la baseline ufficiale fa 0,2433 di spellF1 e 0,67 di macroF1, molto meno che su Metrica (0,33–0,37 e 0,81–0,82). I dati Metrica sporcati sono quindi più facili del tracking vero, e i numeri qui sopra **non sono una previsione dello score**: servono a confrontare i metodi fra loro. Per la consegna va usato `model_dah`, perché senza D i possessi separati da una palla morta si fondono. Dettagli in `docs/audit_drive/challenge_metric_eval.txt`; per ripetere la baseline: `python docs/audit_drive/scripts/baseline_metrica.py data/metrica`.

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
├── predictions_slice.csv     # oppure submission.csv per tutta la registrazione reale
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
