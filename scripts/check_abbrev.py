import pandas as pd

for season in [2014, 2015, 2016, 2017, 2025, 2026]:
    team = pd.read_parquet(f"data/raw/stats_team_week_{season}.parquet")
    print(season, sorted(team.team.unique()))
    print("   non-REG rows:", (team.season_type != "REG").sum())
