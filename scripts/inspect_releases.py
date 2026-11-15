import requests

releases = [
    "schedules",
    "stats_team",
    "stats_player",
    "pbp",
    "injuries",
    "depth_charts",
    "rosters",
    "nextgen_stats",
    "espn_data",
]
for rel in releases:
    try:
        r = requests.get(
            f"https://api.github.com/repos/nflverse/nflverse-data/releases/tags/{rel}",
            timeout=30,
        )
        if r.status_code != 200:
            print(f"\n=== {rel}: HTTP {r.status_code}")
            continue
        data = r.json()
        assets = data.get("assets", [])
        print(f"\n=== {rel}: {len(assets)} assets")
        for a in assets:
            name = a["name"]
            if "2026" in name or name.endswith(".parquet") and "week" not in name:
                print(f"  {name:60s} {a['size'] / 1e6:8.1f} MB")
    except Exception as e:
        print(f"\n=== {rel}: {e}")
