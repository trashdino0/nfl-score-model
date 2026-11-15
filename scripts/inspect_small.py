import requests
from pathlib import Path

base = "https://github.com/nflverse/nflverse-data/releases/download"
files = {
    "stats_team_week_2026.parquet": f"{base}/stats_team/stats_team_week_2026.parquet",
    "stats_player_week_2026.parquet": f"{base}/stats_player/stats_player_week_2026.parquet",
    "injuries_2026.parquet": f"{base}/injuries/injuries_2026.parquet",
    "depth_charts_2026.parquet": f"{base}/depth_charts/depth_charts_2026.parquet",
    "roster_2026.parquet": f"{base}/rosters/roster_2026.parquet",
}
out = Path("data/raw")
for name, url in files.items():
    dest = out / name
    if dest.exists():
        print("exists", name)
        continue
    r = requests.get(url, timeout=120)
    print(name, r.status_code, len(r.content))
    if r.status_code == 200:
        dest.write_bytes(r.content)

import pandas as pd

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 250)
for name in files:
    p = out / name
    if not p.exists():
        continue
    df = pd.read_parquet(p)
    print("\n===", name, df.shape)
    print(df.columns.tolist())
    print(df.head(2).to_string())
