# Audit della cartella Drive "Pragmata" — 10 novembre 2026

Oggetto: il lavoro dei colleghi nella cartella `Pragmata` (export Drive del 10/11/2026), da integrare in questa repository.

## 0. Perimetro e metodo

**Cosa ho letto e verificato**

- Tutti gli script Python (`Modello2_DAH/`, `Features 9_11/codice/`) e il testo dei 7 documenti Word.
- Dati Metrica pubblici, cioè dati esterni: `metriche partite Lore non sporche/*.npz`, `Train set/*_sporca.npz`, i CSV delle feature e `predizioni_partita2.csv`. Li ho analizzati con script locali; sono uscite soltanto statistiche aggregate.
- Ho riscritto in forma vettoriale le 100 feature dei colleghi (`scripts/features_lib.py`). I risultati coincidono con i loro CSV a meno dell'arrotondamento: lo scarto massimo è 0,01 sull'accelerazione della palla, che moltiplica per 25 l'arrotondamento dei CSV a 4 decimali. Questo ha permesso di ricalcolare le feature sui dati sporcati e sulla partita 3, che i colleghi non avevano elaborato.

**Cosa NON ho aperto**

La mail del 3/11 vieta di inviare i materiali forniti ai servizi IA, quindi non ho aperto:

- `BAR-ATH_pulito.npz`;
- `tracking_estratto.csv`: ho solo contato le righe, 199.320, cioè header + 199.319 frame, quindi è BAR-ATH;
- `etichette a mano Lore, Ce/*.xlsx`;
- `buchi_palla_check.xlsx` e `buchi_sospetti_gioco_vivo.xlsx`.

**File `.pkl`**

Non li ho deserializzati, perché caricare un pickle esegue codice. Ho ispezionato solo gli opcode: richiamano soltanto `sklearn`, `numpy` e `joblib`. Al posto dei pickle, ho riaddestrato i modelli dal codice.

## 1. Inventario: cosa c'è e cosa manca

| Filone | Documentazione | Codice nel Drive | Output nel Drive |
|---|---|---|---|
| Etichette H/A/D su Metrica (dagli eventi) | `Regole per etichette Metrica.docx` | **assente** | `etichette` dentro gli npz |
| Pulizia giocatori e palla | `ISTRUZIONI PULIZIA…docx` | **assente** | `BAR-ATH_pulito.npz`, `Train set/*` |
| Sporcatura di Metrica | `Linee guida sporcatura.docx` | **assente** | `Train set/*_sporca.npz` |
| Creazione di `dataset_partitaN.csv` dagli npz | — | **assente** | — |
| Feature (100) | `file spiegone…docx` | presente, **duplicato per partita** (16 file) | `dataset_features_completo_partita{1,2}.csv` |
| Modello H/A (HistGradientBoosting) | `Report-metriche.docx` | training e valutazione presenti; `applica_modello_partita2.py` e `analizza_errori_HA_100.py` **assenti** | `.pkl`, `predizioni_partita2.csv` |
| Modello a 3 classi H/A/D | — | presente | `.pkl` |
| Etichette manuali BAR-ATH | — | — | 2 xlsx (vedi C1) |

**La logica più delicata, cioè etichette, pulizia e sporcatura, esiste solo come output e come descrizione a parole.** Prima di integrarla servono gli script originali.

## 2. Problemi critici

### C1. Conformità al regolamento

- **Etichette manuali di BAR-ATH** (`etichette a mano Lore, Ce/BAR-ATH_manuale.xlsx` e `Etichette BAR - ATH 0-30min.xlsx`). La mail del 3/11 vieta le annotazioni manuali della partita fornita. Queste etichette vanno escluse da qualunque uso, compresa la "verifica a occhio" del modello. Bisogna poi controllare che nessuna soglia o scelta sia stata presa guardandole, e conviene toglierle dalla cartella condivisa.
- **`buchi_palla_check.xlsx` e `buchi_sospetti_gioco_vivo.xlsx`**: non sappiamo come sono stati prodotti. Se un controllo è stato fatto guardando il video, rientra nel divieto di confronto con il video. Va chiesto agli autori.
- **Sporcatura tarata sulle statistiche del tracking BAR-ATH.** Gli autori stessi la indicano come "zona grigia". Il README della repo, però, riporta la regola: "training e tuning dei parametri soltanto su dati esterni, anche per metodi a regole". Prima di basarci su questa taratura serve una conferma scritta dell'organizzazione.
- **La cartella Drive contiene materiali BAR-ATH.** Chi la condivide con strumenti IA (Drive connesso a un assistente, upload, ecc.) rischia di violare la regola sui servizi IA.

### C2. Il modello è addestrato su Metrica *pulito*, non sporcato

Le prove:

- i CSV hanno 28 slot fissi;
- `ball_available` coincide al 100% con l'npz pulito, mentre con quello sporco coincide al 98,3%;
- `team_0_players_available` coincide al 100% con l'npz pulito.

La roadmap (`Per le idee.docx`) e le linee guida prevedono invece l'addestramento sui dati sporcati e poi ripuliti, per assomigliare a BAR-ATH. Effetto misurato, con il modello addestrato su 2 partite pulite e testato sulla terza:

| Test su | Pulito | Sporcato |
|---|---|---|
| Partita 1 | 92,3% | 87,5% |
| Partita 2 | 92,5% | 87,0% |
| Partita 3 | 91,4% | 87,1% |

**Il 91,9% del report sovrastima di circa 5 punti la prestazione attesa su dati simili a BAR-ATH.** Addestrare direttamente sui dati sporcati dà l'88–89% (§6).

### C3. D è "palla assente", per costruzione

Nelle etichette Metrica, P(palla assente | D) = 1,000 in tutte e tre le partite pulite: la regola delle etichette definisce D a partire dai buchi della palla. Le conseguenze:

- **Modello H/A:** predice H o A anche nei frame D. Su tutti i frame della partita 2 l'accuratezza è del **55,4%**, perché il 40% dei frame è D e risulta sempre sbagliato. Il 91,9% è calcolato escludendo i D.
- **Modello a 3 classi:** il suo 94,5% sulla partita 2 pulita è gonfiato da un D banale (palla assente ⇒ D). Sui dati sporcati scende al 90,1% e P(D | palla assente) cala a 0,94–0,95.
- Su BAR-ATH il legame tra palla assente e D non è verificabile. La cartella stessa contiene un file sui buchi a gioco vivo. Il riconoscimento di D va quindi trattato come un problema a sé, non come un sottoprodotto del classificatore.

### C4. Previsioni instabili nel tempo

Sulla partita 2, nei soli frame H/A:

- cambi di possesso reali: **293**;
- cambi predetti dal modello: **1.167**;
- blocchi predetti più corti di 1 s: 835 su 1.299, con una mediana di 10 frame.

L'accuratezza per frame nasconde questo problema. Secondo `Change of possession_.docx`, la sfida è riconoscere *quando* il possesso cambia, quindi l'instabilità potrebbe pesare molto sul punteggio. Una maggioranza mobile centrata riduce i cambi predetti a 270–350, con un piccolo guadagno di accuratezza (§6). Questo risultato è solo indicativo: la finestra va scelta in cross-validation sulle partite di training, non sul test.

## 3. Problemi importanti (correttezza e generalizzazione)

- **I1. Feature legate allo slot** (`slot_0..27_available`). Uno slot non corrisponde alla stessa persona da una partita all'altra:
  - nella partita 2 la squadra 1 occupa gli slot 14–25, e gli slot 26–27 sono forzati a 0;
  - nei dati sporcati e in BAR-ATH gli slot sono tracce anonime.

  Il modello H/A usa `slot_1` e `slot_11`; quello a 3 classi ne usa 12. Toglierle non costa nulla: con le feature senza slot, il test leave-one-game-out dà 92,6 / 92,8 / 91,2% contro 92,3 / 92,5 / 91,4%. **Vanno rimosse.**
- **I2. `players_missing` e `players_available_pct` dipendono dalla rosa nominale** (14 slot contro 12), quindi hanno un significato diverso da partita a partita. Vanno rimosse oppure riferite a 11 giocatori.
- **I3. Assegnazione degli slot alle squadre scritta a mano o presa dal primo frame.**
  - `calcola_nuove_features*.py` usa intervalli fissi: slot 0–14 / 14–28 nella partita 1, 0–14 / 14–26 nella partita 2.
  - Gli altri script usano `giocatore_i_squadra.iloc[0]`.
  - Entrambi gli approcci si rompono sui dati sporcati, dove la squadra cambia frame per frame (`squadra_fr`, 216–250 cambi per partita), e sulla partita 3, che ha 35 slot.
- **I4. Le derivate temporali non rispettano i confini.** Velocità, accelerazione, direzione e `ball_missing_run_length` vengono calcolate anche a cavallo del cambio di tempo (frame 71.268 nella partita 1). L'effetto su Metrica è piccolo. Su BAR-ATH, però, la registrazione include prepartita e intervallo: va usato `HALVES`, come già fa `src/ball.py`.
- **I5. Soglie incoerenti per la palla:** 35 m/s nelle feature (`ball_tracking_anomaly`), 40 m/s nella pulizia.
- **I6. Validazione debole.**
  - C'è un solo split temporale e una sola partita di test.
  - **La partita 3 non è mai stata usata.**
  - Le metriche sono solo per frame.
- **I7. Riproducibilità.**
  - Mancano gli script elencati nel §1.
  - La versione di `scikit-learn` non è fissata. Con sklearn 1.9.1 la mia riproduzione dà 91,62% contro 91,92% sulla partita 2, e 91,44% contro 91,79% sul test temporale.
  - Ogni script esiste in una copia per partita, che differisce solo nei percorsi e nel numero di slot: è già una fonte di divergenza.
  - `addestra_modello_HA_100.py` è duplicato in due cartelle.
  - `"Feature escluse: 100 - …"` è scritto a mano nel codice.
  - I modelli sono condivisi come `.pkl`.
- **I8. Alcune affermazioni dei documenti non sono riproducibili.**
  - Le linee guida dichiarano che, dopo la pulizia, l'1–2% dei frame ha più di 11 giocatori per squadra. Misuro il 2,6–5,0% dei frame a gioco vivo in cui almeno una squadra ne ha più di 11.
  - Dichiarano "circa un frame su tre con meno di 11". Misuro il 47–60% di frame in cui almeno una squadra ne ha meno di 11.
  - Probabilmente le definizioni sono diverse (per squadra o per frame, su tutti i frame o solo a gioco vivo). Senza il codice non si può verificare.

## 4. Problemi minori

- `team_k_width` misura l'estensione lungo x, cioè la lunghezza del campo, e `team_k_length` quella lungo y: i nomi sono invertiti.
- `ball_distance_0_5s` usa 12 frame (0,48 s), perché `round(12.5)` dà 12.
- L'interpolazione della pulizia (buchi fino a 1 s) a volte riempie anche frame D: P(palla assente | D) scende da 1,000 a 0,998 sui dati sporcati.
- Il campo Metrica è 105 × 68 m, BAR-ATH nella repo è 105,3 × 68 m: va solo documentato.
- Gli script dipendono dalla cartella da cui vengono lanciati (percorsi relativi), tranne `calcola_nuove_features`, che usa `__file__`.

## 5. Cosa va tenuto

- **La regola delle etichette** (ultimo evento di possesso, D che prevale, ripresa assegnata a chi batte) è ben argomentata e coerente con la definizione di cambio di possesso della challenge. Ho verificato che H corrisponde a `team_0`: con H, la squadra 0 è la più vicina alla palla nell'82% dei frame.
- **Le regole di pulizia** sono fisicamente motivate, usano soglie fissate a priori e tengono separata la palla osservata da quella stimata (`palla_interpolata`). È lo stesso principio di `ball_source` nella repo.
- **Le feature per squadra** (distanze ordinate, conteggi entro un raggio, baricentri, velocità) danno un guadagno reale e stabile sulla regola della squadra più vicina:

  | Dati | Regola più vicina | Modello |
  |---|---|---|
  | Puliti | 79–82% | 91–93% |
  | Sporcati | 74–77% | 87–89% |

- **I controlli di allineamento** in `unisci_features*` (frame, etichette, colonne duplicate).
- **I documenti dichiarano onestamente i limiti.**

## 6. Numeri chiave (riprodotti con `scripts/experiments.py`)

Accuratezza sui frame H/A, salvo dove indicato.

| Esperimento | P1 | P2 | P3 |
|---|---|---|---|
| Squadra più vicina, puliti | 81,4 | 81,9 | 78,8 |
| Squadra più vicina, sporcati | 76,2 | 76,6 | 74,4 |
| Train su altre 2 partite pulite, tutte le feature → test pulito | 92,3 | 92,5 | 91,4 |
| idem → test sporcato | 87,5 | 87,0 | 87,1 |
| Senza slot e feature di rosa → test pulito | 92,7 | 92,8 | 91,0 |
| Train su 2 partite sporcate (senza slot) → test sporcato | 88,1 | 89,3 | 87,9 |
| idem + maggioranza mobile ±25 frame (indicativo) | 88,8 | 90,0 | 88,3 |
| Cambi H/A reali | 328 | 293 | 363 |
| Cambi predetti senza smoothing / con ±25 frame | 1.365 / 306 | 1.251 / 272 | 1.773 / 347 |

Modello a 3 classi (split 80% su P1 pulita), su tutti i frame della partita 2:

- 94,5% sui dati puliti;
- 90,1% sui dati sporcati;
- 1.714 cambi predetti contro 427 reali.

## 7. Proposta di integrazione nella repo

In ordine di priorità.

1. **Recuperare gli script mancanti** dai colleghi: etichette, pulizia, sporcatura, creazione dei dataset, applicazione del modello, analisi degli errori. Chiarire anche l'origine dei file `buchi_*.xlsx` e mettere da parte le etichette manuali (C1).
2. **Rappresentazione ad array nel lettore.** Oggi `src/data.py` raggruppa le coordinate per squadra e scarta l'identità di traccia, che però serve per le velocità. Va aggiunto un formato `gioc (n, S, 2)`, `palla (n, 2)`, `squadra (n, S)`, `segment (n,)`, comune a BAR-ATH e Metrica, insieme a un lettore degli npz Metrica (dati esterni, quindi ammessi per training e tuning).
3. **`src/team_features.py`:** le feature per squadra in forma vettoriale, partendo da `scripts/features_lib.py`, con queste modifiche:
   - niente feature `slot_*` né di rosa;
   - squadra letta per frame;
   - derivate azzerate a ogni confine di `segment`;
   - una sola soglia per la palla.

   Vanno accompagnate da test sintetici, nello stile di `tests/`: scambio delle squadre, palla assente, confine di tempo.

   **Stato (10/11):** fatto in `src/arrays.py`, `src/team_features.py` e `tests/test_team_features.py`. Il modulo si chiama `team_features.py`. Delle 100 feature dei colleghi, 67 coincidono con i loro CSV. Una è cambiata di proposito: `ball_displacement_0_5s` usa 13 frame, contro i loro 12. Le altre 32 sono state tolte: 28 di slot e 4 di rosa. Con il test leave-one-game-out sui dati sporcati l'accuratezza H/A è 88,4 / 89,5 / 88,1%. Lo script è `scripts/verify_team_features.py`. Nel notebook è la tappa 6, con le opzioni di CONFIG `team_features` e `ball_speed_max_ms`.
4. **`src/cleaning.py`:** le regole di pulizia, da riconciliare con `src/ball.py` e `src/players.py`. L'interpolazione lineare esiste già. Mancano il filtro a 40 m/s per la palla e la pulizia per squadra, al massimo 11 giocatori, che rispetto a `nearest22` è più fondata.
5. **`src/model.py`:**
   - training sui dati Metrica sporcati con cross-validation leave-one-game-out sulle 3 partite;
   - riconoscimento di D come stadio separato;
   - smoothing temporale con finestra scelta in cross-validation;
   - metriche anche sui cambi di possesso.

   Va aggiunto `scikit-learn` a `requirements.txt` con una versione fissata. I modelli si rigenerano dal codice: nessun `.pkl` in Git.
6. **Collegamento a CONFIG e al notebook**, secondo le regole del README (§4C): l'output resta `frame,label`, con gli stessi frame e lo stesso ordine.
