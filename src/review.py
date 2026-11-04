"""Visualizzazioni della run corrente, senza demo o dati di riferimento."""
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.players import player_counts


def stage_summary(values):
    return pd.DataFrame({"voce": list(values), "valore": [str(v) for v in values.values()]})


def plot_input(tracking):
    """Copertura al secondo e tracce della porzione effettivamente selezionata."""
    counts = player_counts(tracking)
    seconds = tracking.frame // 25
    observed = tracking.ball_xy.map(lambda point: point is not None)
    coverage = pd.DataFrame({"second": seconds, "seen": observed}).groupby("second").seen.mean()
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), constrained_layout=True)
    axes[0].plot(coverage.index / 60, coverage * 100)
    axes[0].set(ylabel="Palla osservata (%)", ylim=(-2, 102), title="Input: copertura della palla per secondo")
    for key, label in [("home", "Casa"), ("away", "Ospiti")]:
        series = counts.groupby(counts.frame // 25)[key].mean()
        axes[1].plot(series.index / 60, series, label=label)
    axes[1].axhline(11, color="gray", linestyle="--", label="11: riferimento")
    axes[1].set(ylabel="Tracce medie al secondo", xlabel="Minuto della registrazione", title="Input: tracce per squadra")
    axes[1].legend()
    for ax in axes:
        left, right = tracking.frame.iloc[0] / 1500, tracking.frame.iloc[-1] / 1500
        ax.set_xlim(left, max(left + 1 / 1500, right))
    return fig


def plot_ball_gap(tracking, preview_frames=250):
    """Primo buco ricostruito, altrimenti primo buco non risolto; nessuna verità nota."""
    missing = tracking.ball_xy.map(lambda point: point is None).to_numpy()
    estimated = tracking.ball_source.isin(["linear", "velocity"]).to_numpy()
    choices = np.flatnonzero(estimated if estimated.any() else missing)
    if not len(choices):
        print("Nessun buco della palla nell'intervallo: nessuna ricostruzione da mostrare.")
        return None
    start = int(choices[0]); end = start
    while start > 0 and missing[start - 1]: start -= 1
    while end + 1 < len(tracking) and missing[end + 1]: end += 1
    # Finestra limitata e dichiarata: non espande l'intero buco se molto lungo.
    left = max(0, start - min(25, preview_frames // 4))
    part = tracking.iloc[left:min(len(tracking), left + preview_frames)]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True, constrained_layout=True)
    for component, ax, label in [(0, axes[0], "x (m)"), (1, axes[1], "y (m)")]:
        values = [p[component] if p is not None else np.nan for p in part.ball_xy]
        ax.plot(part.frame, values, '.-', color="black", label="Osservata", markersize=3)
        mask = part.ball_source.isin(["linear", "velocity"])
        selected = part.loc[mask]
        ax.scatter(selected.frame, [p[component] for p in selected.ball_estimated_xy], marker="x", color="#d98220", label="Stimata")
        ax.axvspan(tracking.frame.iloc[start], min(tracking.frame.iloc[end], part.frame.iloc[-1]), alpha=.1, color="gray", label="Buco originale")
        ax.set_ylabel(label); ax.legend(loc="best")
        ax.set_xlim(part.frame.iloc[0], max(part.frame.iloc[0] + 1, part.frame.iloc[-1]))
    axes[1].set_xlabel("Frame originale della registrazione")
    fig.suptitle("Palla: osservazioni e stime della run — la traiettoria reale nel buco è sconosciuta")
    return fig


def plot_players_change(before, after):
    """Conteggi prima/dopo e rimozioni nel primo frame modificato, se esiste."""
    a, b = player_counts(before), player_counts(after)
    changed = np.flatnonzero(a.total.to_numpy() != b.total.to_numpy())
    overfull = np.flatnonzero(a.total.to_numpy() > 22)
    index = int(changed[0]) if len(changed) else (int(overfull[0]) if len(overfull) else len(before) // 2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(a.frame / 1500, a.total, label="Prima", linewidth=1.5)
    axes[0].plot(b.frame / 1500, b.total, '--', label="Dopo", linewidth=1)
    axes[0].axhline(22, color="gray", linestyle=":")
    axes[0].set(xlabel="Minuto della registrazione", ylabel="Tracce totali", title="Giocatori: conteggio prima/dopo")
    axes[0].legend()
    old, new = before.iloc[index], after.iloc[index]
    for key, label, color in [("home_xy", "Casa", "#2878b5"), ("away_xy", "Ospiti", "#d55e43")]:
        keep = Counter(map(tuple, new[key])); removed = []
        for point in old[key]:
            if keep[tuple(point)]: keep[tuple(point)] -= 1
            else: removed.append(point)
        if new[key]:
            xy = np.asarray(new[key]); axes[1].scatter(xy[:, 0], xy[:, 1], color=color, label=label)
        if removed:
            xy = np.asarray(removed); axes[1].scatter(xy[:, 0], xy[:, 1], marker="x", s=90, color=color, label=label + " rimossa")
    if old.ball_xy is not None:
        axes[1].scatter(*old.ball_xy, marker="*", color="black", s=100, label="Palla osservata")
    axes[1].plot([0,105.3,105.3,0,0],[0,0,68,68,0],color="gray",linewidth=.8)
    axes[1].set(xlim=(-2,107.3),ylim=(70,-2),aspect="equal",title="Frame {} — {}".format(old.frame, "tracce rimosse: croci" if len(changed) else "nessun frame modificato"))
    axes[1].legend(fontsize=8)
    return fig


def plot_decisions(features, predictions):
    """Distanze effettivamente usate e output, mai confronto con il video."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True, constrained_layout=True)
    time = features.frame / 1500
    axes[0].plot(time, features.nearest_home_m, label="Distanza minima casa", linewidth=.8)
    axes[0].plot(time, features.nearest_away_m, label="Distanza minima ospiti", linewidth=.8)
    axes[0].set(ylabel="Metri",title="Regola: vince la squadra con distanza minima minore; parità o distanza assente → X")
    axes[0].legend()
    axes[1].step(time, predictions.label.map({"H":1,"A":-1,"X":0}), where="mid", linewidth=.8)
    estimated = features.ball_source.isin(["linear", "velocity"])
    axes[1].scatter(time.loc[estimated], np.full(int(estimated.sum()), 1.35), marker="|", s=20, color="#d98220", label="Palla stimata")
    axes[1].set(yticks=[-1,0,1],yticklabels=["A","X","H"],ylim=(-1.5,1.6),xlabel="Minuto della registrazione",title="Output della run; tacche arancioni = coordinate stimate")
    axes[1].legend()
    return fig
