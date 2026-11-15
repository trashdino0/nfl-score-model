import requests
from pathlib import Path

base = "https://github.com/nflverse/nflverse-data/releases/download"
files = {
    "play_by_play_2025.parquet": f"{base}/pbp/play_by_play_2025.parquet",
    "snap_counts_2025.parquet": f"{base}/snap_counts/snap_counts_2025.parquet",
    "ftn_charting_2025.parquet": f"{base}/ftn_charting/ftn_charting_2025.parquet",
    "advstats_week_def_2025.parquet": f"{base}/pfr_advstats/advstats_week_def_2025.parquet",
}
out = Path("data/raw")
for name, url in files.items():
    dest = out / name
    if dest.exists():
        print("exists", name)
        continue
    r = requests.get(url, timeout=180)
    print(name, r.status_code, f"{len(r.content) / 1e6:.1f} MB")
    if r.status_code == 200:
        dest.write_bytes(r.content)
