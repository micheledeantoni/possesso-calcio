# Pragmata: ball possession from tracking

Team Pragmata's submission to **More Than a Hack 2027**. The pipeline labels every frame of the Barcelona – Athletic tactical video as **H** (home possession), **A** (away possession) or **D** (dead ball), from the tracking data alone.

It combines per-frame gradient-boosting classifiers with a three-state Viterbi decoder, and is trained and tuned only on Metrica's three public sample matches. On those matches, each tested without being seen in training, it reaches a **spellF1 of 0.708**, against 0.350 for the official baseline.

## Reproduce the submission

The whole run takes about 25 minutes on a 16 GB Windows laptop. On 17 November 2026 it was repeated from a fresh clone and a fresh environment, and produced the identical file.

### 1. Install

The submission was produced with **Python 3.14.7** and the exact versions in `requirements.txt`. Other versions run the code, but the file may differ by a few frames.

```sh
python -m venv .venv
.venv\Scripts\activate             # Windows; on macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Add the data

Neither the match data nor the Metrica data is in this repository.

```text
data/
├── BAR-ATH_tactical_atd.xml        # challenge data bundle, tactical source
├── BAR-ATH_tactical_atd.txt
└── metrica_sample/                 # public Metrica sample data (about 180 MB)
    ├── Sample_Game_1/
    ├── Sample_Game_2/
    └── Sample_Game_3/
```

`metrica_sample/` holds the three `Sample_Game_*` folders from the `data/` folder of [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data).

### 3. Run

```sh
python -m src.metrica_labels data/metrica_sample data/metrica_etichettati
python -m src.dirtying data/metrica_etichettati data/metrica --seed 20
python -m src.dirtying data/metrica_etichettati data/metrica_extra_1 --seed 0
python -m src.dirtying data/metrica_etichettati data/metrica_extra_2 --seed 10
python -m src.run_notebook --full --run-name consegna
```

1. The first command labels the Metrica matches.
2. The next three degrade them to resemble the challenge tracking, with fixed seeds.
3. The last one runs the pipeline on the whole match. It takes 10–17 minutes and almost all of 16 GB of RAM.

The output is `outputs/consegna/submission.csv`, 199,319 rows of `frame,label`; *consegna* is Italian for submission. Its SHA256 is:

```text
fcce40e37e1d95efa07191d2f51552296440edb6ef122f2cf7278d57f9e4ebe0
```

The submitted copy is the same file renamed `Pragmata.csv`. It passes the challenge's `evaluation/validate_submission.py`.

To check the code without any data:

```sh
python -m unittest discover -s tests
```

## How it works

| Step | What it does | Code |
| --- | --- | --- |
| 1. Read | Player and ball positions per frame, in metres, keeping each track's identity | `src/arrays.py` |
| 2. Clean | Splits broken tracks, merges duplicates, drops objects off the pitch and impossible ball points | `src/cleaning.py` |
| 3. Features | Possession features per team (distances, players near the ball, lines, lanes, ball direction) and dead-ball features (collective speed, restart set-ups, the ball when visible) | `src/team_features.py`, `src/dead_ball_features.py` |
| 4. Classify | H against A, and two dead-ball classifiers mixed as P(D) = 0.34 · with ball + 0.66 · without generic ball | `src/model.py`, `src/model_dah.py` |
| 5. Decode | A three-state Viterbi picks the most likely H/A/D sequence, so a single frame cannot flip possession | `src/model_dah.py` |
| 6. Write | Checks the format and saves the CSV | `src/run_notebook.py`, `src/notebook_io.py` |

**Training data.** The three public Metrica sample matches have 430,023 frames, about 4.8 hours at 25 fps. Each is used in three degraded versions. The labels come from Metrica's published events, through the rule in `src/metrica_labels.py`:

- possession belongs to the team of the last pass, carry, recovery, set piece or shot;
- a ball gap explained by a stoppage (ball out, foul, shot, card) or followed by a restart is D;
- other ball gaps stay live play.

**The ball shortcut.** In these labels D coincides with a missing ball, but in the real match the ball also disappears during live play. The main dead-ball classifier therefore uses the ball only when it is visible. A second classifier that also sees the ball's absence gets a weight of 0.34. Parameter choices that call D on more than 30% of the live-play ball gaps are rejected.

**Model selection.** Every figure comes from a nested leave-one-game-out validation on the three Metrica matches, with spellF1 as the objective. Runs are deterministic (`random_state` 42).

| Validation on Metrica (spellF1) | Match 1 | Match 2 | Match 3 | Mean |
| --- | --- | --- | --- | --- |
| Submitted model | 0.748 | 0.736 | 0.640 | **0.708** |
| Without the ball dead-ball classifier | 0.708 | 0.680 | 0.670 | 0.686 |
| Official baseline | 0.371 | 0.352 | 0.326 | 0.350 |

The official baseline reaches 0.2433 on the real match, so Metrica is easier: these figures compare methods, they do not predict the score. The technical document submitted with the entry covers the method, the possession rules, the results and the limitations in full.

## Rules

- **Training and tuning data.** Training and parameter choices use only the public Metrica matches. No labels, manual annotation or video of the provided match are used, and no choice was made on the public 5-minute window.
- **Degradation calibration.** The degradation of the Metrica matches was calibrated on statistics of the provided match's tracking.
- **No match data here.** `data/` and `outputs/` are excluded from Git.
- **AI assistants.** We used Claude (Anthropic) to write and review the code and the documentation. The method itself uses no language model.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | The pipeline above, plus the modules of the exploration notebook (ball reconstruction, player filters, quality reports, plots) |
| `tests/` | 149 unit tests on synthetic data |
| `esplorazione.ipynb` | The notebook that `src/run_notebook.py` runs headless; it can also be opened to explore a stretch of the match |
| `docs/audit_drive/` | Working notes: every measurement behind the choices above, with the commands that produce it |
| `docs/guida_sviluppo.md` | The team's developer guide, in Italian |

Code comments and working notes are in Italian.

## Licence and attribution

The challenge's starter code reused here keeps its attribution and MIT licence in `LICENSE-METRICA.txt`; `ADATTAMENTI.md` lists what was adapted. Neither licence covers the match data, which is not redistributable.
