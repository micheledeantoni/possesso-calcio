# Aderenza ai notebook ufficiali

Riferimento: https://github.com/metrica-sports/more-than-a-hack-2027, notebook in `starter/`.

- **01 Smart Tagging:** struttura dei tag, conteggi, gestione N/A, conversione secondi/frame e limiti degli intervalli. Il lettore `src/events.py` adatta `starter/load_smart_tagging.py` e mantiene gli intervalli semiaperti. Nessun uso di ancore derivate dal ground truth privato.
- **02 Tracking:** indice ATD one-based, frame zero-based, coordinate normalizzate convertite in metri, tracce non identificative, visualizzazione del campo e assenza della palla. Manteniamo il lettore senza kloppy per ridurre le dipendenze. A differenza dell'originale, `src/data.py` raggruppa le coordinate per squadra; `src/arrays.py` conserva invece lo slot di ogni traccia per calcolare le velocità.
- **03 Prima submission:** struttura del percorso input → regola → CSV → controllo. La funzione `smooth()` in `src/official_baseline.py` è ripresa dall'implementazione ufficiale; la demo non applica tuning sulla challenge. La nostra regola iniziale conserva X dove l'originale grezzo usa D. Non presentiamo le due pipeline come equivalenti.

Le aggiunte locali sono sanity check, grafici delle assenze, rapporto HTML e ricostruzione sintetica della palla. Le stime entrano nelle feature soltanto attivando il modulo palla in CONFIG.

## Indicazioni non riprese

Niente ricerca della finestra migliore sul ground truth pubblico; niente confronto degli errori con il video; niente correzioni o annotazioni manuali della partita. Questi suggerimenti presenti nel materiale originale sono superati dalla mail dell'organizzazione del 3 novembre 2026.

## Licenza

Il codice ufficiale riutilizzato è attribuito a Metrica Sports; il testo della licenza è conservato in `LICENSE-METRICA.txt`. Questa licenza non consente la redistribuzione dei materiali riservati della partita. I dati locali non sono inclusi nella repository.
