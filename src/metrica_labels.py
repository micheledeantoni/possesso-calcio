"""Etichette H/A/D sulle partite di esempio Metrica (dati pubblici, esterni alla challenge).

Riprende lo script del gruppo "Etichette sample games.py" (Lorenzo) con le stesse
regole e gli stessi parametri. Per ogni frame:
- D = palla morta: un buco della palla nel tracking spiegato da un'interruzione
  (palla fuori, fallo subito, tiro, cartellino, da 3 s prima a 1 s dopo l'inizio
  del buco) o da una ripresa (calcio piazzato o rinvio dal fondo entro 1 s dalla
  fine del buco). Gli altri buchi restano gioco vivo.
- H / A = squadra dell'ultimo evento di possesso (passaggio, conduzione, recupero,
  calcio piazzato, tiro). X prima del primo evento di possesso, se non è D.

Unica aggiunta rispetto all'originale: il tempo di gioco (`periodo`), che l'originale
non salvava e che la sporcatura prendeva dai file del `Train set` (`--periods-from`).

Dati: https://github.com/metrica-sports/sample-data, cartella `data/`, da copiare in
una cartella locale (per esempio `data/metrica_sample/`) con le sottocartelle
`Sample_Game_1`, `Sample_Game_2`, `Sample_Game_3`.

Uso:
    python -m src.metrica_labels data/metrica_sample data/metrica_etichettati
    python -m src.dirtying data/metrica_etichettati data/metrica --seed 0
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd

PITCH = (105.0, 68.0)  # campo dei dati di esempio Metrica, in metri
FPS = 25
POSSESSION_EVENTS = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT"}
STOP_EVENTS = {"BALL OUT", "FAULT RECEIVED", "SHOT", "CARD"}
STOP_WINDOW = (3 * FPS, 1 * FPS)  # interruzione da 3 s prima a 1 s dopo l'inizio del buco
RESTART_WINDOW = 1 * FPS          # ripresa entro 1 s dalla fine del buco


def _find(folder, name):
    """Un file per nome dentro `folder`, anche nelle sottocartelle."""
    found = sorted(Path(folder).rglob(name))
    if not found:
        raise FileNotFoundError(f"{name} non trovato in {folder}")
    return found[0]


def _read_tracking_csv(path, pitch=PITCH):
    """(giocatori n × S × 2, palla n × 2, periodo n) in metri, da un CSV Metrica."""
    df = pd.read_csv(path, skiprows=2)
    values = df.iloc[:, 3:].to_numpy(float)  # dopo Period, Frame, Time: coppie x, y
    players = values[:, :-2].reshape(len(df), -1, 2) * pitch
    ball = values[:, -2:] * pitch
    return players, ball, df["Period"].to_numpy(int)


def read_csv_game(folder, number, pitch=PITCH):
    """Partite 1 e 2 (formato CSV): giocatori, squadra per slot, palla, periodo, eventi."""
    home, ball, period = _read_tracking_csv(_find(folder, f"Sample_Game_{number}_RawTrackingData_Home_Team.csv"), pitch)
    away, _, _ = _read_tracking_csv(_find(folder, f"Sample_Game_{number}_RawTrackingData_Away_Team.csv"), pitch)
    players = np.concatenate([home, away], axis=1)
    team = np.array([0] * home.shape[1] + [1] * away.shape[1], dtype=np.int8)
    raw = pd.read_csv(_find(folder, f"Sample_Game_{number}_RawEventsData.csv"))
    events = pd.DataFrame({"frame": raw["Start Frame"].to_numpy() - 1,  # il tracking parte da 1, gli array da 0
                           "type": raw["Type"].to_numpy(),
                           "subtype": raw["Subtype"].fillna("").to_numpy(),
                           "team": (raw["Team"] == "Away").astype(int).to_numpy()})
    return players, team, ball, period, events


def read_epts_game(folder, pitch=PITCH):
    """Partita 3 (formato EPTS, come la partita della challenge)."""
    root = ET.parse(_find(folder, "Sample_Game_3_metadata.xml")).getroot()
    teams = [t.get("id") for t in root.iter("Team")]
    team_of = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    player_ids = sorted(set(channel.values()))
    column = {p: i for i, p in enumerate(player_ids)}
    team = np.array([0 if team_of[p] == teams[0] else 1 for p in player_ids], dtype=np.int8)
    # Le colonne del file cambiano a ogni sostituzione: una "specifica" per tratto di partita.
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")
                if r.get("playerChannelId").endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), [column[channel[r]] for r in refs]))

    lines = _find(folder, "Sample_Game_3_tracking.txt").read_text().splitlines()
    players = np.full((len(lines), len(player_ids), 2), np.nan)
    ball = np.full((len(lines), 2), np.nan)
    for line in lines:
        frame_id, player_part, ball_part = line.strip().split(":")
        frame_id = int(frame_id)
        mapping = next(c for a, b, c in specs if a <= frame_id <= b)
        for k, xy in enumerate(player_part.split(";")):
            if not xy.startswith("NaN"):
                x, y = xy.split(",")
                players[frame_id - 1, mapping[k]] = (float(x) * pitch[0], float(y) * pitch[1])
        if not ball_part.startswith("NaN"):
            x, y = ball_part.split(",")[:2]
            ball[frame_id - 1] = (float(x) * pitch[0], float(y) * pitch[1])

    data = json.loads(_find(folder, "Sample_Game_3_events.json").read_text())["data"]

    def subtype(event):
        s = event["subtypes"]
        if not s:
            return ""
        return s["name"] if isinstance(s, dict) else " ".join(x["name"] for x in s)

    events = pd.DataFrame({"frame": [e["start"]["frame"] - 1 for e in data],
                           "type": [e["type"]["name"] for e in data],
                           "subtype": [subtype(e) for e in data],
                           "team": [0 if e["team"] and e["team"]["id"] == teams[0] else 1 for e in data],
                           "period": [e["period"] for e in data]})
    period = period_from_events(len(lines), events)
    return players, team, ball, period, events.drop(columns="period")


def period_from_events(n, events):
    """Periodo per frame dal primo evento di ogni tempo (il file EPTS non lo riporta)."""
    period = np.ones(n, dtype=int)
    for number, start in events.groupby("period")["frame"].min().sort_index().items():
        period[max(int(start), 0):] = int(number)
    return period


def _runs(mask):
    padded = np.concatenate(([False], mask, [False])).astype(int)
    d = np.diff(padded)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return starts, ends - starts


def make_labels(ball, events, n=None):
    """Etichette H/A/D/X per frame e conteggio dei buchi (morti, a gioco vivo).

    `events`: colonne frame (zero-based), type, subtype, team (0 casa, 1 ospiti).
    """
    ball = np.asarray(ball, dtype=float)
    n = len(ball) if n is None else n
    restarts = events.loc[(events["type"] == "SET PIECE") | events["subtype"].str.contains("GOAL KICK"),
                          "frame"].to_numpy()
    stops = events.loc[events["type"].isin(STOP_EVENTS), "frame"].to_numpy()
    dead = np.zeros(n, bool)
    dead_gaps = live_gaps = 0
    for start, length in zip(*_runs(np.isnan(ball[:, 0]))):
        end = start + length
        near_restart = np.any(np.abs(restarts - end) <= RESTART_WINDOW)
        near_stop = np.any((stops >= start - STOP_WINDOW[0]) & (stops <= start + STOP_WINDOW[1]))
        if near_restart or near_stop:
            dead[start:end] = True
            dead_gaps += 1
        else:
            live_gaps += 1

    possession = events[events["type"].isin(POSSESSION_EVENTS)].sort_values("frame")
    last = np.searchsorted(possession["frame"].to_numpy(), np.arange(n), side="right") - 1
    team = np.where(last >= 0, possession["team"].to_numpy()[np.maximum(last, 0)], -1)
    labels = np.full(n, "X", dtype="<U1")
    labels[team == 0] = "H"
    labels[team == 1] = "A"
    labels[dead] = "D"
    return labels, dead_gaps, live_gaps


def summary(labels, fps=FPS):
    """Tratti, possessi H/A, durata media in secondi, quota di possessi sotto 1 s."""
    s = labels[labels != "X"]
    changes = np.flatnonzero(s[1:] != s[:-1]) + 1
    spells = np.split(s, changes)
    possessions = np.array([len(x) for x in spells if x[0] in "HA"])
    return len(spells), len(possessions), possessions.mean() / fps, float((possessions < fps).mean())


def label_game(folder, number):
    """Una partita: (giocatori, squadra per slot, palla, periodo, etichette, buchi morti, buchi vivi)."""
    players, team, ball, period, events = read_csv_game(folder, number) if number < 3 else read_epts_game(folder)
    labels, dead_gaps, live_gaps = make_labels(ball, events)
    return players, team, ball, period, labels, dead_gaps, live_gaps


def save_labelled_game(path, players, team, ball, period, labels, pitch=PITCH):
    """Formato degli npz "puliti" del gruppo, più `periodo` (letto da `load_metrica_npz`)."""
    np.savez_compressed(path, gioc=players.astype(np.float32), squadra_slot=team, palla=ball.astype(np.float32),
                        etichette=labels, periodo=np.asarray(period).astype(np.int8), campo=np.array(pitch))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Etichette H/A/D sulle partite di esempio Metrica")
    parser.add_argument("sample_dir", help="cartella con Sample_Game_1, Sample_Game_2, Sample_Game_3")
    parser.add_argument("out_dir")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for number in (1, 2, 3):
        players, team, ball, period, labels, dead_gaps, live_gaps = label_game(args.sample_dir, number)
        _, n_possessions, duration, short = summary(labels)
        print(f"Partita {number}: frame {len(ball):,} H {np.mean(labels == 'H'):.1%} A {np.mean(labels == 'A'):.1%} "
              f"D {np.mean(labels == 'D'):.1%} X {np.mean(labels == 'X'):.1%}")
        print(f"  buchi della palla: palla morta {dead_gaps}, gioco vivo {live_gaps}; "
              f"possessi {n_possessions}, durata media {duration:.1f} s, sotto 1 s {short:.0%}")
        save_labelled_game(out / f"metrica_partita_{number}.npz", players, team, ball, period, labels)
    print("Salvati in", out)


if __name__ == "__main__":
    main()
