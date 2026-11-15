import pandas as pd
import numpy as np

pd.set_option("display.width", 260)

f = pd.read_parquet("data/processed/game_features.parquet")
w = pd.read_csv("outputs/week09_predictions.csv")
train = f[(f.season >= 2018) & f.home_score.notna()]

nets = {
    "epa_net": "Overall EPA",
    "pass_epa_net": "Pass EPA",
    "rush_epa_net": "Rush EPA",
    "qb_vs_pass_net": "QB vs pass defense",
    "qb_cpoe_vs_coverage_net": "QB CPOE vs coverage",
    "wr1_vs_coverage_net": "WR1 vs coverage",
    "wr_corps_vs_coverage_net": "WR corps vs coverage",
    "rb1_vs_run_net": "RB1 vs run defense",
    "pass_rush_vs_protection_net": "Pass rush vs protection",
    "turnovers_net": "Turnover margin",
    "red_zone_net": "Red-zone efficiency",
    "points_drive_net": "Points per drive",
}

# z-score using training std
stats = {}
for c in nets:
    if c in train.columns:
        stats[c] = (train[c].mean(), train[c].std())

for _, r in w.iterrows():
    gid = r.game_id
    row = f[f.game_id == gid].iloc[0]
    items = []
    for c, label in nets.items():
        if c not in row.index or c not in stats:
            continue
        val = row[c]
        if pd.isna(val):
            continue
        mu, sd = stats[c]
        z = (val - mu) / sd if sd and sd > 0 else 0
        items.append((abs(z), label, val, z))
    items.sort(reverse=True)
    top = items[:3]
    txt = "; ".join(f"{lab} {z:+.2f}sd" for _, lab, val, z in top)
    print(f"{r.away_team} @ {r.home_team}: {txt}")
    print(
        "   QB:",
        row.get("away_qb_name"),
        "(away) vs",
        row.get("home_qb_name"),
        "(home)",
    )
    print(
        "   injuries: away off",
        row.get("away_inj_lost_off"),
        "def",
        row.get("away_inj_lost_def"),
        "| home off",
        row.get("home_inj_lost_off"),
        "def",
        row.get("home_inj_lost_def"),
    )
    print(
        "   WR out:",
        row.get("wr_out_a"),
        row.get("wr_out_h"),
        "RB out:",
        row.get("rb_out_a"),
        row.get("rb_out_h"),
        "TE out:",
        row.get("te_out_a"),
        row.get("te_out_h"),
    )
    print()
