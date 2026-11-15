import pandas as pd
from pathlib import Path

pd.set_option("display.width", 240)

# Coverage of each family
for family, pattern in [
    ("team", "stats_team_week_*.parquet"),
    ("player", "stats_player_week_*.parquet"),
    ("injuries", "injuries_*.parquet"),
    ("snaps", "snap_counts_*.parquet"),
    ("advdef", "advstats_week_def_*.parquet"),
    ("pbp", "play_by_play_*.parquet"),
]:
    rows = []
    for p in sorted(Path("data/raw").glob(pattern)):
        season = int(p.stem.split("_")[-1])
        try:
            df = pd.read_parquet(p, columns=None)
            rows.append((season, df.shape[0], df.shape[1]))
        except Exception as e:
            rows.append((season, f"ERR {e}", 0))
    print(f"\n{family}:")
    for r in rows:
        print("  ", r)

pbp = pd.read_parquet("data/raw/play_by_play_2024.parquet")
interesting = [
    c
    for c in pbp.columns
    if any(
        k in c
        for k in [
            "was_pressure",
            "pressure",
            "blitz",
            "qb_hit",
            "sack",
            "success",
            "epa",
            "explosive",
            "drive_",
            "series_success",
            "pass_oe",
            "xpass",
            "penalty",
            "turnover",
            "interception",
            "fumble",
            "first_down",
            "play_type",
            "posteam",
            "defteam",
            "home_team",
            "away_team",
            "game_id",
            "week",
            "season",
        ]
    )
]
print("\nPBP interesting columns:", len(interesting))
print(interesting)

# Sample team-week columns and missingness
team = pd.read_parquet("data/raw/stats_team_week_2024.parquet")
print("\nTEAM 2024 shape", team.shape)
print(
    team[
        [
            "game_id",
            "team",
            "opponent_team",
            "passing_epa",
            "rushing_epa",
            "def_sacks",
            "def_qb_hits",
        ]
    ].head()
)
print(
    "team seasons:",
    sorted(
        pd.read_parquet(p).season.iloc[0]
        for p in Path("data/raw").glob("stats_team_week_*.parquet")
    ),
)

# advdef teams
adv = pd.read_parquet("data/raw/advstats_week_def_2024.parquet")
print("\nADV DEF 2024 teams:", adv.team.nunique(), "rows", len(adv))
print(adv.groupby("team").size().head())

# snap counts 2026 teams
snap = pd.read_parquet("data/raw/snap_counts_2026.parquet")
print("\nSNAP 2026:", snap.shape, snap.week.min(), snap.week.max())
print(snap.groupby("week").size().to_string())
