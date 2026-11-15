import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 260)

f = pd.read_parquet("data/processed/game_features.parquet")
print("shape", f.shape)
print("completed by season:")
print(f[f.home_score.notna()].groupby("season").size().to_string())
print("\n2026 weeks:")
print(
    f[(f.season == 2026)]
    .groupby("week")
    .agg(n=("game_id", "size"), done=("home_score", lambda s: s.notna().sum()))
    .to_string()
)

print("\nWeek 9 rows:")
cols = [
    "game_id",
    "gameday",
    "away_team",
    "home_team",
    "home_score",
    "away_score",
    "spread_line",
    "total_line",
    "home_qb_out",
    "away_qb_out",
    "home_start_qb_epa_db_roll4",
    "away_start_qb_epa_db_roll4",
    "home_qb_change",
    "away_qb_change",
    "elo_diff",
    "home_inj_lost_off",
    "away_inj_lost_off",
]
print(f[(f.season == 2026) & (f.week == 9)][cols].to_string(index=False))

print("\nMissingness of key features for 2026 weeks 1-8:")
key = [
    c
    for c in f.columns
    if any(
        k in c
        for k in [
            "epa_net",
            "pass_epa_net",
            "qb_vs_pass_net",
            "wr1_vs_coverage_net",
            "rb1_vs_run_net",
            "pass_rush_vs_protection_net",
            "elo_diff",
            "spread_line",
            "total_line",
            "home_inj_lost_off",
            "home_qb_out",
            "start_qb_epa_diff",
            "naive_margin",
        ]
    )
]
sub = f[(f.season == 2026) & (f.week <= 8)]
print(sub[key].notna().mean().round(3).to_string())

print("\nCorrelations with margin/total (completed 2018-2025):")
train = f[(f.season >= 2018) & (f.season <= 2025) & f.margin.notna()].copy()
cand = [
    "elo_diff",
    "epa_net",
    "pass_epa_net",
    "rush_epa_net",
    "qb_vs_pass_net",
    "wr1_vs_coverage_net",
    "rb1_vs_run_net",
    "pass_rush_vs_protection_net",
    "turnovers_net",
    "red_zone_net",
    "points_drive_net",
    "qb_epa_db_diff",
    "coverage_yards_per_target_diff",
    "pressure_rate_diff",
    "naive_margin",
    "market_margin",
]
for c in cand:
    if c in train.columns:
        print(
            f"  {c:32s} margin corr {train[c].corr(train.margin):+.3f}   total corr {train[c].corr(train.total):+.3f}"
        )

print("\nSimple baseline errors (2018-2025):")
from sklearn.metrics import mean_absolute_error

for name, pred_m, pred_t in [
    ("market", train.market_margin, train.market_total),
    ("naive", train.naive_margin, train.naive_total),
]:
    ok = pred_m.notna() & pred_t.notna()
    print(
        f"  {name:8s} margin MAE {mean_absolute_error(train.loc[ok, 'margin'], pred_m[ok]):6.2f}  total MAE {mean_absolute_error(train.loc[ok, 'total'], pred_t[ok]):6.2f}  n={ok.sum()}"
    )

# Leakage spot check: recompute one team's prior 4-game average from team_games
tg = pd.read_parquet("data/processed/team_games.parquet")
team = "KC"
season = 2026
week = 8
hist = (
    tg[(tg.team == team) & (tg.season == season) & (tg.week < week)]
    .sort_values("week")
    .tail(4)
)
manual = hist.off_epa_play.mean()
feat = f[
    (f.season == season)
    & (f.week == week)
    & ((f.home_team == team) | (f.away_team == team))
]
val = (
    feat.iloc[0]["home_off_epa_play_roll4"]
    if feat.iloc[0]["home_team"] == team
    else feat.iloc[0]["away_off_epa_play_roll4"]
)
print(
    f"\nLeakage check {team} week {week}: manual prior-4 off EPA {manual:.4f}, feature {val:.4f}"
)

# Check 2026 week 1 values are prior-season based
w1 = f[(f.season == 2026) & (f.week == 1)].iloc[0]
print(
    "2026 W1 home off epa roll4:",
    w1["home_off_epa_play_roll4"],
    "prev:",
    w1.get("home_off_epa_play_prev"),
)
