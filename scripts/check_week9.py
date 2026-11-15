import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 260)

f = pd.read_parquet("data/processed/game_features.parquet")
w9 = f[(f.season == 2026) & (f.week == 9) & f.home_score.isna()].copy()
cols = [
    "game_id",
    "home_team",
    "away_team",
    "home_qb_id",
    "home_qb_name",
    "home_last_qb_id",
    "home_last_qb_name",
    "home_qb_out",
    "home_qb_questionable",
    "home_qb_report_status",
    "home_qb_change",
    "away_qb_id",
    "away_qb_name",
    "away_last_qb_id",
    "away_last_qb_name",
    "away_qb_out",
    "away_qb_questionable",
    "away_qb_report_status",
    "away_qb_change",
]
print(w9[cols].to_string(index=False))

print("\nMarket win prob vs spread:")
print(
    w9[
        [
            "game_id",
            "spread_line",
            "market_margin",
            "market_total",
            "market_home_win_prob",
            "home_implied",
            "away_implied",
        ]
    ].to_string(index=False)
)

# Market baseline accuracy 2018-2025
h = f[(f.season >= 2018) & (f.season <= 2025) & f.home_score.notna()].copy()
h["market_pick"] = (h.market_margin > 0).astype(float)
h["actual_home_win"] = (h.margin > 0).astype(float)
valid = h.home_win.notna() & h.market_pick.notna()
print(
    "\nMarket favorite winner accuracy 2018-2025:",
    (h.loc[valid, "market_pick"] == h.loc[valid, "actual_home_win"]).mean(),
)
print("Home team winner accuracy:", (h.loc[valid, "actual_home_win"] == 1).mean())
print(
    "Market home win prob Brier:",
    (
        (h.loc[valid, "market_home_win_prob"] - h.loc[valid, "actual_home_win"]) ** 2
    ).mean(),
)

# Naive baseline winner accuracy
h["naive_pick"] = (h.naive_margin > 0).astype(float)
valid2 = h.home_win.notna()
print(
    "Naive winner accuracy:",
    (h.loc[valid2, "naive_pick"] == h.loc[valid2, "actual_home_win"]).mean(),
)

# Ties
print("Ties 2026:", ((f.season == 2026) & (f.margin == 0)).sum())
