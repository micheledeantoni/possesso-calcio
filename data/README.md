# Input locali, esclusi da Git

Input principali:

- `BAR-ATH_tactical_atd.xml`
- `BAR-ATH_tactical_atd.txt`

È possibile copiarli dalla propria estrazione o creare collegamenti simbolici. Non rinominarli. I file non verranno condivisi attraverso Git. Su questo computer sono collegati ai Download: spostare o cancellare gli originali interrompe i collegamenti.

Eseguire ogni lettura dei materiali riservati solo localmente. Non inviare contenuti o output ai servizi IA.

Facoltativo: `BAR-ATH_tactical_smart_tagging.xml`, per la sezione eventi.

Per il modello di possesso (`possession_method="model"`): `data/metrica/` con gli npz delle partite Metrica etichettate, preparati dal gruppo (cartella Drive `Train set/`, versioni sporcate e ripulite). Sono dati esterni: si possono usare per training e scelta dei parametri. Tutti i file `.npz` della cartella vengono usati. In alternativa si possono rigenerare con `python -m src.dirtying` (vedi README, §4F).
