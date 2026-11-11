"""Sporcatura dei dati Metrica puliti, per somigliare al tracking della challenge.

Riscrive le "Linee guida sporcatura" del gruppo: lo script originale non è
disponibile. Le quantità predefinite vengono dalle linee guida (tarate dal gruppo
sulle statistiche del tracking BAR-ATH, uso confermato come consentito) e dal
confronto con i file `Train set` già prodotti dal gruppo (`dirt_statistics`).
Non riproduce quei file uno a uno: ne riproduce le quantità. I valori predefiniti
di `DirtParams` sono stati scelti finché, dopo sporcatura e pulizia, le statistiche
di `dirt_statistics` stavano nella variabilità fra le tre partite del `Train set`
(assenti per frame, durata e posizione delle assenze, cambi di slot, punti in
squadra sbagliata, salti, oggetti fuori campo, frame con più o meno di 11, buchi
della palla). Solo dati Metrica.

Operazioni, in quest'ordine, con seme fisso:
1. assenze: giocatori tolti per tratti di durata log-normale (coda lunga), più
   spesso vicino alle linee laterali e alla palla, spesso insieme ai vicini;
2. identità spezzate: ogni presenza è divisa in tracce, distribuite su slot
   anonimi con squadra fissa;
3. errori di squadra: pezzi brevi (e raramente tracce intere) messi in uno slot
   dell'altra squadra;
4. oggetti fuori dal campo: pochi oggetti che camminano oltre una linea laterale;
5. salti: punti spostati di 0,6–3 m per un frame;
6. palla: buchi di 5–10 s a gioco vivo, lontano dalle interruzioni, nei contrasti.
Nessun doppione aggiunto: Metrica ne ha già. Le etichette non cambiano.

Uso: genera i file come quelli del `Train set` (sporcati e poi puliti):
    python -m src.dirtying <cartella npz Metrica puliti> <cartella di uscita> [--periods-from <Train set>]
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import math
import numpy as np
from src.arrays import TrackingArrays

FPS = 25


@dataclass
class DirtParams:
    # 1. assenze
    absent_per_frame: float = 1.2          # giocatori assenti in media per frame
    absence_median_s: float = 3.5
    absence_sigma: float = 1.6             # log-normale: coda lunga
    absence_max_s: float = 120.0
    line_margin_m: float = 5.0
    line_weight: float = 50.0              # peso relativo vicino alle linee laterali
    ball_radius_m: float = 5.0
    ball_weight: float = 1.5               # peso relativo vicino alla palla
    group_probability: float = 0.7         # il gruppo sparisce e ricompare insieme
    group_radius_m: float = 8.0
    # 2. identità spezzate
    track_median_s: float = 150.0
    slots_per_team: int = 15
    # 3. errori di squadra
    team_error_pieces_per_min: float = 0.09  # per giocatore
    team_error_median_s: float = 4.0
    whole_track_error_probability: float = 0.01
    # 4. oggetti fuori campo
    outside_objects: int = 3
    outside_frame_share: float = 0.015     # frame con un oggetto fuori campo, prima della pulizia
    outside_min_m: float = 1.5
    outside_max_m: float = 6.0
    outside_episode_s: float = 10.0
    # 5. salti
    jumps_per_min: float = 12.0
    jump_min_m: float = 0.6
    jump_max_m: float = 3.0
    # 6. palla
    ball_gaps: int = 12
    ball_gap_min_s: float = 5.0
    ball_gap_max_s: float = 10.0
    ball_gap_margin_s: float = 3.0         # distanza minima da frame D
    ball_gap_players: int = 2
    ball_gap_radius_m: float = 3.0


def _runs(mask):
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    return starts, np.flatnonzero(edges == -1) - starts


def _duration(rng, median_s, sigma, max_s, fps):
    return int(np.clip(round(rng.lognormal(math.log(median_s * fps), sigma)), 1, max_s * fps))


def _absences(players, ball, pitch, params, rng, fps):
    n, slots = players.shape[:2]
    present = np.isfinite(players).all(axis=2)
    y = players[..., 1]
    near_line = (y < params.line_margin_m) | (y > pitch[1] - params.line_margin_m)
    near_ball = np.linalg.norm(players - ball[:, None], axis=2) < params.ball_radius_m
    weight = present * (1 + (params.line_weight - 1) * near_line + (params.ball_weight - 1) * np.nan_to_num(near_ball))
    flat = weight.ravel() / weight.sum()
    absent = np.zeros((n, slots), dtype=bool)
    budget = params.absent_per_frame * n
    added = 0
    while added < budget:
        for cell in rng.choice(flat.size, size=200, p=flat):
            frame, player = divmod(int(cell), slots)
            if absent[frame, player] or added >= budget:
                continue
            group = [player]
            if rng.random() < params.group_probability:
                distance = np.linalg.norm(players[frame] - players[frame, player], axis=1)
                group += [int(p) for p in np.flatnonzero((distance < params.group_radius_m) & present[frame]
                                                         & ~absent[frame]) if p != player]
            # Il gruppo sparisce e ricompare insieme, come in un'occlusione.
            length = _duration(rng, params.absence_median_s, params.absence_sigma, params.absence_max_s, fps)
            for p in group:
                part = slice(frame, min(n, frame + length))
                new = present[part, p] & ~absent[part, p]
                absent[part, p] |= new
                added += int(new.sum())
    return absent


def _pieces(start, end, flipped, params, rng, fps):
    """Divide [start, end) in tracce; alcuni pezzi brevi passano all'altra squadra."""
    mean_track = params.track_median_s * fps / math.log(2)
    cuts, t = [start], start
    while True:
        t += max(1, int(rng.exponential(mean_track)))
        if t >= end:
            break
        cuts.append(t)
    cuts.append(end)
    pieces = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        whole = flipped ^ (rng.random() < params.whole_track_error_probability)
        errors = rng.poisson(params.team_error_pieces_per_min * (b - a) / (60 * fps))
        marks = []
        for _ in range(errors):
            length = _duration(rng, params.team_error_median_s, 0.8, 60, fps)
            if b - a <= length + 2:
                continue
            s = int(rng.integers(a + 1, b - length))
            marks.append((s, s + length))
        marks.sort()
        t = a
        for s, e in marks:
            if s < t:
                continue
            if s > t:
                pieces.append((t, s, whole))
            pieces.append((s, e, not whole))
            t = e
        if t < b:
            pieces.append((t, b, whole))
    return pieces


def _outside_objects(n, pitch, params, rng, fps):
    """Traiettorie (inizio, fine, squadra, xy) di oggetti che camminano oltre una linea laterale."""
    objects = []
    if params.outside_objects <= 0 or params.outside_frame_share <= 0:
        return objects
    per_object = params.outside_frame_share * n / params.outside_objects
    for _ in range(params.outside_objects):
        team = int(rng.integers(0, 2))
        side = int(rng.integers(0, 2))
        covered = 0
        while covered < per_object:
            length = max(fps, int(rng.exponential(params.outside_episode_s * fps)))
            length = int(min(length, max(fps, per_object - covered)))
            start = int(rng.integers(0, max(1, n - length)))
            steps = rng.normal(0, 0.02, size=length).cumsum()  # passo lento lungo la linea
            x = np.clip(rng.uniform(5, pitch[0] - 5) + steps, 0, pitch[0])
            offset = rng.uniform(params.outside_min_m, params.outside_max_m)
            y = np.full(length, -offset if side == 0 else pitch[1] + offset)
            objects.append((start, start + length, team, np.column_stack([x, y])))
            covered += length
    return objects


def _ball_gaps(ball, players, labels, params, rng, fps):
    n = len(ball)
    ball = ball.copy()
    live = np.isin(labels, ["H", "A"]) & np.isfinite(ball).all(axis=1)
    dead = ~np.isin(labels, ["H", "A"])
    margin = int(params.ball_gap_margin_s * fps)
    dead_near = np.convolve(dead.astype(int), np.ones(2 * margin + 1, int), mode="same") > 0
    distance = np.linalg.norm(players - ball[:, None], axis=2)
    contest = (np.nan_to_num(distance, nan=np.inf) < params.ball_gap_radius_m).sum(axis=1) >= params.ball_gap_players
    candidates = np.flatnonzero(live & ~dead_near & contest)
    taken = np.zeros(n, dtype=bool)
    gaps = []
    for start in rng.permutation(candidates):
        if len(gaps) >= params.ball_gaps:
            break
        span = params.ball_gap_max_s - params.ball_gap_min_s
        length = int(round((params.ball_gap_min_s + span * rng.beta(1.5, 2.5)) * fps))
        part = slice(max(0, start - margin), min(n, start + length + margin))
        if (start + length > n or taken[part].any() or dead_near[start:start + length].any()
                or not np.isfinite(ball[start:start + length]).all()):
            continue
        ball[start:start + length] = np.nan
        taken[start:start + length] = True
        gaps.append((int(start), int(length)))
    return ball, gaps


def dirty(clean, labels, params=None, seed=0, fps=FPS):
    """Restituisce (array sporcati, registro). `clean`: un `TrackingArrays` con una
    traccia per giocatore. Le etichette servono soltanto a scegliere i buchi della palla."""
    params = DirtParams() if params is None else params
    if clean.pitch is None:
        raise ValueError("Dimensioni del campo assenti")
    rng = np.random.default_rng(seed)
    players = np.asarray(clean.players, dtype=float)
    ball = np.asarray(clean.ball, dtype=float)
    n, slots = players.shape[:2]
    true_team = np.asarray(clean.team)[0]
    if not (np.asarray(clean.team) == true_team).all():
        raise ValueError("Servono dati puliti: squadra fissa per giocatore")

    absent = _absences(players, ball, clean.pitch, params, rng, fps)
    visible = np.isfinite(players).all(axis=2) & ~absent

    tracks = []  # (inizio, fine, squadra dello slot, sorgente: giocatore o oggetto)
    for p in range(slots):
        for start, length in zip(*_runs(visible[:, p])):
            for a, b, flipped in _pieces(int(start), int(start + length), False, params, rng, fps):
                tracks.append((a, b, int(true_team[p]) ^ int(flipped), ("player", p)))
    objects = _outside_objects(n, clean.pitch, params, rng, fps)
    for k, (a, b, team, _) in enumerate(objects):
        tracks.append((a, b, team, ("object", k)))
    tracks.sort(key=lambda t: t[0])

    pool = params.slots_per_team
    out = np.full((n, 2 * pool, 2), np.nan)
    last_end = np.zeros(2 * pool, dtype=int)
    without_slot = 0
    for a, b, team, (kind, index) in tracks:
        free = [s for s in range(team * pool, (team + 1) * pool) if last_end[s] <= a]
        if not free:  # come un tracker senza slot liberi: la traccia non viene seguita
            without_slot += 1
            continue
        slot = int(rng.choice(free))
        last_end[slot] = b
        out[a:b, slot] = players[a:b, index] if kind == "player" else objects[index][3][:b - a]

    present = np.flatnonzero(np.isfinite(out[..., 0]).ravel())
    jumps = rng.choice(present, size=min(len(present), int(params.jumps_per_min * n / (60 * fps))), replace=False)
    angle = rng.uniform(0, 2 * np.pi, len(jumps))
    size = rng.uniform(params.jump_min_m, params.jump_max_m, len(jumps))
    flat = out.reshape(-1, 2)
    flat[jumps] += np.column_stack([np.cos(angle), np.sin(angle)]) * size[:, None]

    ball, gaps = _ball_gaps(ball, players, np.asarray(labels), params, rng, fps)
    team = np.broadcast_to(np.repeat(np.array([0, 1], dtype=np.int8), pool), (n, 2 * pool)).copy()
    log = {"seme": seed, "assenze_frame_giocatore": int(absent.sum()), "tracce": len(tracks),
           "tracce_senza_slot": without_slot,
           "oggetti_fuori_campo_episodi": len(objects), "salti": int(len(jumps)), "buchi_palla": len(gaps),
           "parametri": asdict(params)}
    return replace(clean, players=out, ball=ball, team=team), log


def dirt_statistics(clean, dirty_arrays, labels, fps=FPS):
    """Quantità della sporcatura misurate confrontando posizioni identiche.

    `dirty_arrays` può essere sporcato e poi pulito. Serve a confrontare la nostra
    sporcatura con i file `Train set` del gruppo, e a verificarla nei test.
    """
    C, D = np.asarray(clean.players, dtype=float), np.asarray(dirty_arrays.players, dtype=float)
    n, Sc = C.shape[:2]
    labels = np.asarray(labels)
    match = np.full(D.shape[:2], -1)
    for a in range(0, n, 20000):
        part = np.linalg.norm(D[a:a + 20000, :, None] - C[a:a + 20000, None], axis=3)
        part = np.where(np.isnan(part), np.inf, part)
        best, value = part.argmin(axis=2), part.min(axis=2)
        match[a:a + 20000] = np.where(value < 1e-3, best, -1)
    exact = match >= 0
    seen = np.zeros((n, Sc), dtype=bool)
    rows, cols = np.nonzero(exact)
    seen[rows, match[rows, cols]] = True
    present = np.isfinite(C[..., 0])
    absent = present & ~seen
    live = np.isin(labels, ["H", "A"])
    durations, near_line, starts = [], 0, []
    for p in range(Sc):
        s, l = _runs(absent[:, p])
        keep = l >= 2  # un solo frame: salto o pulizia, non un'assenza
        durations += list(l[keep] / fps)
        starts += [(int(a), p) for a in s[keep]]
        near_line += int(np.sum([(C[a, p, 1] < 5) or (C[a, p, 1] > clean.pitch[1] - 5) for a in s[keep]]))
    start_frames = np.array([a for a, _ in starts])
    together = np.mean([np.sum(start_frames == a) > 1 for a in start_frames]) if len(starts) else float("nan")
    slot_of = np.full((n, Sc), -1)
    slot_of[rows, match[rows, cols]] = cols
    hops = int(((slot_of[1:] >= 0) & (slot_of[:-1] >= 0) & (slot_of[1:] != slot_of[:-1])).sum())
    follow = []  # tratti in cui uno slot segue lo stesso giocatore
    for s in range(D.shape[1]):
        who = match[:, s]
        edge = np.flatnonzero(np.r_[True, who[1:] != who[:-1]])
        lengths = np.diff(np.r_[edge, n])
        follow += list(lengths[who[edge] >= 0] / fps)
    true_team = np.asarray(clean.team)[0][np.maximum(match, 0)]
    wrong = exact & (np.asarray(dirty_arrays.team) != true_team)
    wrong_runs = np.concatenate([_runs(wrong[:, s])[1] for s in range(D.shape[1])]) / fps
    inexact = np.isfinite(D[..., 0]) & ~exact
    L, W = clean.pitch
    outside = inexact & ((D[..., 0] < 0) | (D[..., 0] > L) | (D[..., 1] < 0) | (D[..., 1] > W))
    team = np.asarray(dirty_arrays.team)
    counts = [(np.isfinite(D[..., 0]) & (team == k)).sum(axis=1) for k in (0, 1)]
    gone = np.isfinite(clean.ball[:, 0]) & ~np.isfinite(dirty_arrays.ball[:, 0])
    _, gap_lengths = _runs(gone)
    long_gaps = gap_lengths[gap_lengths >= 5 * fps] / fps
    durations = np.array(durations)
    return {
        "assenti_per_frame_gioco_vivo": float(absent[live].sum(axis=1).mean()),
        "assenze_episodi": len(durations),
        "assenze_mediana_s": float(np.median(durations)) if len(durations) else float("nan"),
        "assenze_p90_s": float(np.percentile(durations, 90)) if len(durations) else float("nan"),
        "assenze_vicino_linee": near_line / max(1, len(durations)),
        "assenze_insieme_ad_altre": float(together),
        "cambi_slot_senza_assenza_al_minuto": hops / (n / (60 * fps)),
        "slot_stesso_giocatore_mediana_s": float(np.median(follow)) if follow else float("nan"),
        "punti_squadra_sbagliata": float(wrong.sum() / max(1, exact.sum())),
        "squadra_sbagliata_mediana_s": float(np.median(wrong_runs)) if len(wrong_runs) else float("nan"),
        "salti_in_campo": int((inexact & ~outside).sum()),
        "frame_con_oggetti_fuori_campo": float(outside.any(axis=1).mean()),
        "frame_gioco_vivo_squadra_oltre_11": float(((counts[0] > 11) | (counts[1] > 11))[live].mean()),
        "frame_gioco_vivo_squadra_sotto_11": float(((counts[0] < 11) | (counts[1] < 11))[live].mean()),
        "buchi_palla_lunghi": int(len(long_gaps)),
        "buchi_palla_mediana_s": float(np.median(long_gaps)) if len(long_gaps) else float("nan"),
    }


def make_training_game(clean, labels, params=None, seed=0, fps=FPS):
    """Sporca e poi pulisce con `src/cleaning.py`, come i file del `Train set`."""
    from src.cleaning import clean_tracking
    dirty_arrays, log = dirty(clean, labels, params, seed, fps)
    cleaned, interpolated, cleaning_log = clean_tracking(dirty_arrays, fps=fps)
    return cleaned, interpolated, {"sporcatura": log, "pulizia": cleaning_log}


def save_training_game(path, cleaned, interpolated, labels):
    """Stesso formato dei file `Train set` del gruppo."""
    np.savez_compressed(path, gioc=cleaned.players.astype(np.float32), squadra_fr=cleaned.team.astype(np.int8),
                        palla=cleaned.ball.astype(np.float32), palla_interpolata=interpolated,
                        etichette=np.asarray(labels), periodo=np.asarray(cleaned.segment).astype(np.int8),
                        valutato=np.asarray(cleaned.segment) > 0, campo=np.asarray(cleaned.pitch, dtype=float))


def main(argv=None):
    import argparse, json
    from src.arrays import load_metrica_npz
    parser = argparse.ArgumentParser(description="Sporca e pulisce le partite Metrica pulite")
    parser.add_argument("clean_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--periods-from", help="cartella con npz che contengono `periodo` per le stesse partite")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for number, path in enumerate(sorted(Path(args.clean_dir).glob("*.npz")), start=1):
        clean = load_metrica_npz(path)
        if args.periods_from:
            match = sorted(Path(args.periods_from).glob(f"{path.stem}*.npz"))
            if match:
                with np.load(match[0]) as data:
                    clean = replace(clean, segment=data["periodo"].astype(int))
        cleaned, interpolated, log = make_training_game(clean, clean.labels, seed=args.seed + number)
        save_training_game(out / f"{path.stem}_sporca.npz", cleaned, interpolated, clean.labels)
        print(path.stem, json.dumps({k: v for k, v in log["sporcatura"].items() if k != "parametri"}),
              json.dumps(log["pulizia"]))


if __name__ == "__main__":
    main()
