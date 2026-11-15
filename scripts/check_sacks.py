import pandas as pd

pbp = pd.read_parquet(
    "data/raw/play_by_play_2025.parquet",
    columns=[
        "game_id",
        "posteam",
        "play_type",
        "pass",
        "rush",
        "sack",
        "qb_hit",
        "pass_attempt",
        "rush_attempt",
        "qb_dropback",
        "yards_gained",
        "season_type",
        "third_down_converted",
        "third_down_failed",
    ],
)
pbp = pbp[pbp.season_type == "REG"]
print("Rows:", len(pbp))
print("play_type x sack:")
print(pd.crosstab(pbp.play_type, pbp.sack).to_string())
print("\nplay_type x pass:")
print(pd.crosstab(pbp.play_type, pbp["pass"]).to_string())
print("\nplay_type x rush:")
print(pd.crosstab(pbp.play_type, pbp.rush).to_string())
print(
    "\nSack rows play_type:",
    pbp[pbp.sack == 1].play_type.value_counts(dropna=False).to_dict(),
)
print(
    "Pass rows sack:", pbp[pbp["pass"] == 1].sack.value_counts(dropna=False).to_dict()
)

team = pd.read_parquet("data/raw/stats_team_week_2025.parquet")
team = team[team.season_type == "REG"]
print("\nTeam attempts+sacks:", (team.attempts + team.sacks_suffered).sum())
print("PBP pass plays + sacks:", ((pbp.play_type == "pass").sum() + pbp.sack.sum()))
print("PBP pass plays:", (pbp.play_type == "pass").sum())
print(
    "PBP pass_attempt sum:",
    pbp.pass_attempt.sum(),
    "qb_dropback sum:",
    pbp.qb_dropback.sum(),
)
print("Team attempts:", team.attempts.sum(), "sacks:", team.sacks_suffered.sum())
print(
    "\nthird down flags:", pbp.third_down_converted.sum(), pbp.third_down_failed.sum()
)
