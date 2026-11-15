import pandas as pd

for name, path in [
    ("2026", "outputs/backtest_2026_weeks1_8.csv"),
    ("2021-2025", "outputs/validation_2021_2025_predictions.csv"),
]:
    d = pd.read_csv(path)
    d = d[d.home_score.notna() & d.away_score.notna()]
    eh = d.pred_home_score - d.home_score
    ea = d.pred_away_score - d.away_score
    print(name, "n", len(d))
    for label, e in [("home", eh), ("away", ea)]:
        print(
            label,
            "mean",
            round(e.mean(), 2),
            "std",
            round(e.std(), 2),
            "p10",
            round(e.quantile(0.10), 2),
            "p25",
            round(e.quantile(0.25), 2),
            "p75",
            round(e.quantile(0.75), 2),
            "p90",
            round(e.quantile(0.90), 2),
        )
    print()
