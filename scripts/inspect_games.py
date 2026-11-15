import pandas as pd

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 220)

df = pd.read_parquet("data/raw/games.parquet")
print("shape:", df.shape)
print("columns:", df.columns.tolist())

g = df[df.season == 2026].copy()
print("\n2026 games:", len(g))
print(
    g.groupby("week")
    .agg(
        n=("game_id", "size"),
        done=("home_score", lambda s: s.notna().sum()),
        first=("gameday", "min"),
        last=("gameday", "max"),
    )
    .to_string()
)

print("\nWeek 9 games:")
cols = [
    "game_id",
    "gameday",
    "weekday",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "spread_line",
    "total_line",
    "roof",
    "surface",
    "temp",
    "wind",
    "away_rest",
    "home_rest",
]
print(g[g.week == 9][cols].to_string(index=False))

print("\nLast 10 completed games:")
done = g[g.home_score.notna()].sort_values(["gameday", "game_id"]).tail(10)
print(done[cols].to_string(index=False))

print("\nSeasons:", sorted(df.season.unique()))
print("Max gameday:", df.gameday.max())
