"""Grafici descrittivi senza etichette reali di possesso."""
import numpy as np
import matplotlib.pyplot as plt
from src.quality import missing_runs


def plot_quality(timeline, fps=25):
    """Copertura per minuto, numero tracce, distribuzione durate dei buchi."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    if timeline.empty:
        axes[0].text(.5, .5, "Nessun record", ha="center")
        return fig
    minute = np.floor(timeline.time_s / 60).astype(int)
    coverage = timeline.groupby(minute).ball_observed.mean() * 100
    axes[0].bar(coverage.index + 0.5, coverage.values, width=0.9, color="#238b8e")
    axes[0].set_xlim(0, max(1, float(coverage.index.max() + 1)))
    axes[0].set(xlabel="Minuto della registrazione", ylabel="Palla osservata (%)", ylim=(0, 100),
                title="Copertura del tracking della palla")
    for column, label in [("home_tracks", "Casa"), ("away_tracks", "Ospiti")]:
        # Medie per secondo: riduce il rumore visivo, non modifica i dati.
        series = timeline.groupby(np.floor(timeline.time_s).astype(int))[column].mean()
        axes[1].plot(series.index / 60, series.values, label=label, linewidth=1, marker="o" if len(series) < 2 else None)
    axes[1].axhline(11, color="gray", linestyle="--", label="11: riferimento, non vincolo")
    axes[1].set(xlabel="Minuto della registrazione", ylabel="Tracce (media al secondo)",
                title="Tracce rilevate per squadra: non identità personali")
    axes[1].set_xlim(0, max(1 / 60, float(timeline.time_s.max() / 60)))
    axes[1].legend()
    gaps = missing_runs(timeline, fps)
    if gaps:
        axes[2].hist(gaps, bins=np.linspace(0, max(gaps) * 1.1, min(40, max(2, len(gaps))) + 1), color="#df9b32")
        axes[2].set_xlim(left=0)
        axes[2].set_yscale("log")
    else:
        axes[2].text(.5, .5, "Nessun buco della palla", ha="center", transform=axes[2].transAxes)
    axes[2].set(xlabel="Durata del buco (secondi)", ylabel="Numero di episodi (scala log)",
                title="Intervalli senza osservazioni della palla")
    return fig


def plot_frame(tracking, index=0):
    """Mappa statica delle coordinate: nessuna previsione sovrapposta a video."""
    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    ax.set_facecolor("#e7f1e7")
    ax.plot([0, 105.3, 105.3, 0, 0], [0, 0, 68, 68, 0], color="gray")
    ax.axvline(105.3 / 2, color="gray", linewidth=1)
    row = tracking.iloc[index]
    for key, label, color in [("home_xy", "Casa", "#2878b5"), ("away_xy", "Ospiti", "#d55e43")]:
        points = row[key]
        if points:
            x, y = zip(*points)
            ax.scatter(x, y, label=label, color=color, s=65)
    if row.ball_xy is not None:
        ax.scatter(*row.ball_xy, marker="*", color="black", s=140, label="Palla")
    ax.set(xlim=(-2, 107.3), ylim=(70, -2), aspect="equal", xlabel="Metri", ylabel="Metri",
           title="Coordinate del frame {} — osservazioni, non identità".format(row.frame))
    ax.legend()
    return fig


def plot_reconstruction(tracking, truth, reconstructions, title):
    """Confronto su verità sintetica: traiettoria e posizione verticale nel tempo."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    truth = np.asarray(truth)
    axes[0].plot(truth[:, 0], truth[:, 1], color="gray", label="Traiettoria sintetica completa")
    axes[1].plot(tracking.frame, truth[:, 1], color="gray", label="Traiettoria sintetica completa")
    mask = tracking.ball_xy.map(lambda p: p is not None).to_numpy()
    visible = truth[mask]
    axes[0].scatter(visible[:, 0], visible[:, 1], color="black", s=14, label="Osservazioni disponibili", zorder=4)
    axes[1].scatter(tracking.frame[mask], visible[:, 1], color="black", s=14, zorder=4)
    for label, table in reconstructions.items():
        selected = table[table.ball_estimated_xy.map(lambda p: p is not None)]
        if len(selected):
            xy = np.array(selected.ball_estimated_xy.tolist())
            axes[0].scatter(xy[:, 0], xy[:, 1], marker="x", label=label)
            axes[1].scatter(selected.frame, xy[:, 1], marker="x", label=label)
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Percorso sul campo", aspect="equal")
    axes[1].set(xlabel="Frame sintetico", ylabel="y (m)", title="Il cambio di direzione può essere invisibile")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    return fig
