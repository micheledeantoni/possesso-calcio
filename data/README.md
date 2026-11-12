# Input locali, esclusi da Git

Input principali:

- `BAR-ATH_tactical_atd.xml`
- `BAR-ATH_tactical_atd.txt`

È possibile copiarli dalla propria estrazione o creare collegamenti simbolici. Non rinominarli. I file non verranno condivisi attraverso Git. Su questo computer sono collegati ai Download: spostare o cancellare gli originali interrompe i collegamenti.

Eseguire ogni lettura dei materiali riservati solo localmente. Non inviare contenuti o output ai servizi IA.

Facoltativo: `BAR-ATH_tactical_smart_tagging.xml`, per la sezione eventi.

Per il modello di possesso (`possession_method="model"`): `data/metrica/` con gli npz delle partite Metrica etichettate, preparati dal gruppo (cartella Drive `Train set/`, versioni sporcate e ripulite). Sono dati esterni: si possono usare per training e scelta dei parametri. Tutti i file `.npz` della cartella vengono usati. In alternativa si possono rigenerare con `python -m src.dirtying` (vedi README, §4F).

Versioni sporcate in più per il training del modello H/A/D (CONFIG `metrica_extra_dirs`), da generare una volta:

```sh
python -m src.dirtying "<npz Metrica puliti>" data/metrica_extra_1 --periods-from data/metrica --seed 0
python -m src.dirtying "<npz Metrica puliti>" data/metrica_extra_2 --periods-from data/metrica --seed 10
```

e poi in CONFIG `"metrica_extra_dirs": ["data/metrica_extra_1", "data/metrica_extra_2"]`. Valgono solo come dati di training in più: la scelta dei parametri resta per partita.
