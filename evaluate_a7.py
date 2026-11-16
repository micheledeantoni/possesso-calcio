
from pathlib import Path

import pandas as pd

from src.model import _breaks
from src.model_dah import (
    DEFAULT_GRID,
    fit_dah_model,
    load_dah_games,
    predict_dah,
    evaluate_dah,
    false_dead_rate,
)

CLEAN_DIR = Path("data/metriche partite Lore non sporche")
SCENARIOS = {
    "normale": Path("data/metrica"),
    "assenze_x1.5": Path("data/a7_assenze_x1_5"),
    "errori_squadra_x2": Path("data/a7_errori_squadra_x2"),
    "buchi_palla_x3": Path("data/a7_buchi_palla_x3"),
    "assenze_intermedio": Path("data/a7_assenze_x1_5_intermedio"),
    "errori_squadra_intermedio": Path("data/a7_errori_squadra_x1_5_intermedio"),
    "buchi_palla_intermedio": Path("data/a7_buchi_palla_x2_intermedio"),
}


def match_id(name):
    return name.removesuffix("_sporca")


def main():
    clean_games = load_dah_games(CLEAN_DIR)
    clean_by_id = {match_id(g.name): g for g in clean_games}

    rows = []

    for scenario, folder in SCENARIOS.items():
        dirty_games = load_dah_games(folder, primary=False)
        dirty_by_id = {match_id(g.name): g for g in dirty_games}

        if set(dirty_by_id) != set(clean_by_id):
            raise ValueError(
                f"Partite non corrispondenti nello scenario {scenario}"
            )

        for test_id, test in dirty_by_id.items():
            train = [
                game for name, game in clean_by_id.items()
                if name != test_id
            ]

            if len(train) != 2:
                raise ValueError(f"Training non valido per {test_id}")

            for w in (0.34, 0.0):
                grid = {**DEFAULT_GRID, "w": (w,)}

                model = fit_dah_model(
                    train,
                    grid=grid,
                    objective="spellF1",
                )

                pred = predict_dah(
                    model,
                    test.features,
                    test.dead_features,
                    test.continuous,
                ).label.to_numpy()

                metrics = evaluate_dah(
                    test.labels,
                    pred,
                    breaks=_breaks(test.continuous),
                )

                rows.append({
                    "scenario": scenario,
                    "partita": test_id,
                    "usa_palla": w > 0,
                    "spellF1": metrics["spellF1"],
                    "D_f1": metrics["D_f1"],
                    "accuratezza_tutti": metrics["accuratezza_tutti"],
                    "falsi_D_palla_assente": false_dead_rate(
                        test.labels, pred, test.features
                    ),
                })

    results = pd.DataFrame(rows)

    baseline = (
        results[results["scenario"] == "normale"]
        [["partita", "usa_palla", "spellF1"]]
        .rename(columns={"spellF1": "spellF1_normale"})
    )

    results = results.merge(
        baseline,
        on=["partita", "usa_palla"],
        how="left",
    )

    results["delta_spellF1"] = (
        results["spellF1"] - results["spellF1_normale"]
    )

    results = results.sort_values(
        ["scenario", "usa_palla", "partita"],
        ascending=[True, False, True],
    )

    Path("outputs").mkdir(exist_ok=True)
    results.to_csv("outputs/a7_risultati.csv", index=False)

    print(
        results.round(4).to_string(index=False)
    )
    print("\nSalvato in outputs/a7_risultati.csv")


if __name__ == "__main__":
    main()
