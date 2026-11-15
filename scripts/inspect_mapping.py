import pandas as pd

players = pd.read_parquet("data/raw/players.parquet")
print("players shape", players.shape)
print(
    players[
        [
            "gsis_id",
            "pfr_id",
            "display_name",
            "position",
            "birth_date",
            "rookie_season",
            "height",
            "weight",
        ]
    ]
    .head(8)
    .to_string()
)
print(
    "gsis unique:", players.gsis_id.nunique(), "pfr null:", players.pfr_id.isna().sum()
)

snap = pd.read_parquet("data/raw/snap_counts_2025.parquet")
print("\nSnap positions:", snap.position.value_counts().head(25).to_dict())
print("Snap null pfr ids:", snap.pfr_player_id.isna().sum(), "of", len(snap))

inj = pd.read_parquet("data/raw/injuries_2025.parquet")
print("\nInjury positions:", inj.position.value_counts().head(25).to_dict())
print("Injury gsis null:", inj.gsis_id.isna().sum(), "of", len(inj))
print("Injury report status:", inj.report_status.value_counts(dropna=False).to_dict())

pbp = pd.read_parquet(
    "data/raw/play_by_play_2024.parquet",
    columns=[
        "game_id",
        "posteam",
        "defteam",
        "fixed_drive",
        "fixed_drive_result",
        "drive_inside20",
        "drive_ended_with_score",
        "drive_end_transition",
        "drive_play_count",
        "drive_time_of_possession",
        "drive_start_yard_line",
        "td_team",
        "touchdown",
        "field_goal_result",
        "play_type",
        "season_type",
        "qb_kneel",
        "qb_spike",
        "yards_gained",
        "pass",
        "rush",
        "sack",
        "qb_hit",
        "interception",
        "fumble_lost",
        "first_down",
        "success",
        "epa",
        "pass_oe",
    ],
)
print(
    "\nPBP drive results:",
    pbp.fixed_drive_result.value_counts(dropna=False).head(15).to_dict(),
)
print(
    "drive end transitions:",
    pbp.drive_end_transition.value_counts(dropna=False).head(20).to_dict(),
)
print("play types:", pbp.play_type.value_counts(dropna=False).to_dict())
print("season types:", pbp.season_type.value_counts(dropna=False).to_dict())
print("kneel/spike:", pbp.qb_kneel.sum(), pbp.qb_spike.sum())
