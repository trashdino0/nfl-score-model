import pandas as pd

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 260)

pbp = pd.read_parquet("data/raw/play_by_play_2025.parquet")
print("PBP shape:", pbp.shape)
cols = [
    c
    for c in pbp.columns
    if any(
        k in c
        for k in [
            "game_id",
            "posteam",
            "defteam",
            "home_team",
            "away_team",
            "week",
            "season",
            "play_type",
            "epa",
            "success",
            "pass",
            "rush",
            "down",
            "ydstogo",
            "yardline",
            "drive",
            "series",
            "score",
            "quarter",
            "time",
            "half",
            "qtr",
            "wp",
            "wpa",
            "pressure",
            "blitz",
            "qb_hit",
            "sack",
            "hit",
            "penalty",
            "turnover",
            "fumble",
            "interception",
            "touchdown",
            "field_goal",
            "punt",
            "kickoff",
            "extra_point",
            "two_point",
            "no_play",
            "aborted",
            "lateral",
            "return",
            "sp",
            "td_team",
            "td_player",
            "passer",
            "rusher",
            "receiver",
            "tackle",
            "assist",
            "forced",
            "recovery",
            "safety",
            "drive_start",
            "drive_end",
            "drive_play",
            "drive_time",
            "drive_first",
            "drive_result",
            "start_time",
            "end_time",
            "game_seconds",
            "play_deleted",
            "desc",
            "play_id",
            "old_game_id",
        ]
    )
]
print("Relevant columns:", cols)
print("\nSample rows:")
print(pbp[cols].head(3).T.to_string())

snap = pd.read_parquet("data/raw/snap_counts_2025.parquet")
print("\nSNAP shape:", snap.shape)
print(snap.columns.tolist())
print(snap.head(3).to_string())

ftn = pd.read_parquet("data/raw/ftn_charting_2025.parquet")
print("\nFTN shape:", ftn.shape)
print(ftn.columns.tolist())
print(ftn.head(2).to_string())

adv = pd.read_parquet("data/raw/advstats_week_def_2025.parquet")
print("\nADV DEF shape:", adv.shape)
print(adv.columns.tolist())
print(adv.head(2).to_string())
