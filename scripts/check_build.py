import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 250)

df = pd.read_parquet("data/processed/team_games.parquet")
print("shape", df.shape)
print("\nrows by season:")
print(df.groupby("season").size().to_string())
print("\n2026 weeks:")
print(df[df.season == 2026].groupby("week").size().to_string())
print("\n2026 week 9 rows:")
print(
    df[(df.season == 2026) & (df.week == 9)][
        [
            "game_id",
            "team",
            "opponent_team",
            "points_for",
            "points_allowed",
            "is_home",
            "elo",
        ]
    ].to_string(index=False)
)

cols = [
    "off_epa_play",
    "off_pass_epa_db",
    "off_rush_epa_carry",
    "off_success_rate",
    "def_epa_play",
    "def_pass_epa_db",
    "def_rush_epa_carry",
    "def_success_rate",
    "off_points_per_drive",
    "def_points_per_drive",
    "off_red_zone_td_rate",
    "def_red_zone_td_rate",
    "qb_epa_db",
    "qb_cpoe",
    "wr1_epa_target",
    "rb1_rush_epa_carry",
    "coverage_yards_per_target",
    "coverage_passer_rating",
    "adv_pressures",
    "pressure_rate",
    "team_off_snaps",
    "team_def_snaps",
    "elo",
    "off_pass_rate",
    "off_plays_per_drive",
]
print("\ncoverage by season (non-null %):")
print(
    (
        df.groupby("season")[cols].apply(lambda g: g.notna().mean() * 100).round(1)
    ).to_string()
)

print("\n2026 last 4 weeks sample stats:")
print(
    df[(df.season == 2026) & (df.week >= 5)]
    .groupby("week")[
        [
            "off_epa_play",
            "def_epa_play",
            "qb_epa_db",
            "coverage_yards_per_target",
            "adv_pressures",
        ]
    ]
    .mean()
    .round(3)
    .to_string()
)

print("\nAny duplicate team-game rows:", df.duplicated(["game_id", "team"]).sum())
print("Missing scores:", df.points_for.isna().sum())
print("Elo range:", df.elo.min(), df.elo.max())

# Check a few specific 2026 teams / QBs
print("\n2026 week 8 QB leaders:")
print(
    df[(df.season == 2026) & (df.week == 8)][
        ["team", "qb_name", "qb_attempts", "qb_epa_db", "qb_cpoe"]
    ]
    .sort_values("qb_epa_db", ascending=False)
    .head(10)
    .to_string(index=False)
)

# Check stats consistency
print(
    "\n2026 avg points:",
    df[df.season == 2026].points_for.mean().round(2),
    "epa/play:",
    df[df.season == 2026].off_epa_play.mean().round(3),
)
print(
    "2018-2025 avg points:",
    df[(df.season >= 2018) & (df.season <= 2025)].points_for.mean().round(2),
)
