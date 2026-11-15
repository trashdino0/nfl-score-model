import pandas as pd

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 240)

games = pd.read_parquet("data/raw/games.parquet")
g = games[(games.season == 2026) & (games.week == 9)]
print("Week 9 QB info:")
print(
    g[
        [
            "game_id",
            "away_team",
            "home_team",
            "away_qb_name",
            "home_qb_name",
            "away_qb_id",
            "home_qb_id",
        ]
    ].to_string(index=False)
)

dc = pd.read_parquet("data/raw/depth_charts_2026.parquet")
print("\nDepth chart dt range:", dc.dt.min(), dc.dt.max(), "n unique:", dc.dt.nunique())
print(dc.dt.value_counts().sort_index().tail(20).to_string())
print("\nPos groups:", dc.pos_grp.unique())
print("\nQB starters (pos_rank==1) latest snapshot:")
latest = dc[dc.dt == dc.dt.max()]
print(
    latest[(latest.pos_abb == "QB") & (latest.pos_rank == 1)][
        ["team", "player_name", "pos_rank"]
    ].to_string(index=False)
)

inj = pd.read_parquet("data/raw/injuries_2026.parquet")
print("\nInjury weeks:", sorted(inj.week.unique()))
print("Report statuses:", inj.report_status.value_counts(dropna=False).to_dict())
print("Practice statuses:", inj.practice_status.value_counts(dropna=False).to_dict())
print("\nWeek 9 injuries sample:")
print(
    inj[inj.week == 9][
        ["team", "position", "full_name", "report_status", "practice_status"]
    ]
    .head(30)
    .to_string(index=False)
)

print("\nGames 2026 qb name availability:")
print(
    games[games.season == 2026]
    .groupby("week")
    .agg(
        qb_home=("home_qb_name", lambda s: s.notna().sum()),
        qb_away=("away_qb_name", lambda s: s.notna().sum()),
        n=("game_id", "size"),
    )
    .to_string()
)
