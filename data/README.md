# Input locali, esclusi da Git

Input principali:

- `BAR-ATH_tactical_atd.xml`
- `BAR-ATH_tactical_atd.txt`

È possibile copiarli dalla propria estrazione o creare collegamenti simbolici. Non rinominarli. I file non verranno condivisi attraverso Git. Su questo computer sono collegati ai Download: spostare o cancellare gli originali interrompe i collegamenti.

Eseguire ogni lettura dei materiali riservati solo localmente. Non inviare contenuti o output ai servizi IA.

Facoltativo: `BAR-ATH_tactical_smart_tagging.xml`, per la sezione eventi.

Per i modelli di possesso servono gli npz delle tre partite Metrica etichettate, sporcate e ripulite. Sono dati esterni: si possono usare per training e scelta dei parametri. Si generano dai dati pubblici Metrica (vedi README, §4F):

1. Copiare in `data/metrica_sample/` le cartelle `Sample_Game_1`, `Sample_Game_2` e `Sample_Game_3` della cartella `data/` di https://github.com/metrica-sports/sample-data (circa 180 MB).
2. Generare etichette e sporcature, con i semi della consegna:

```sh
python -m src.metrica_labels data/metrica_sample data/metrica_etichettati
python -m src.dirtying data/metrica_etichettati data/metrica --seed 20
python -m src.dirtying data/metrica_etichettati data/metrica_extra_1 --seed 0
python -m src.dirtying data/metrica_etichettati data/metrica_extra_2 --seed 10
```

`data/metrica/` è la versione principale, usata anche per la scelta dei parametri; tutti i file `.npz` della cartella vengono usati. `data/metrica_extra_1` e `_2` sono versioni sporcate in più, solo per il training del modello H/A/D (CONFIG `metrica_extra_dirs`, già impostata). Con gli stessi semi i file escono identici, e con loro la submission.

Il `Train set` del gruppo (cartella Drive `Train set/`), usato fino al 15 novembre come versione principale, non serve più; su questo computer è in `data/metrica_trainset/` per confronto.
