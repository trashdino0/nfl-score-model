import requests

r = requests.get(
    "https://api.github.com/repos/nflverse/nflverse-data/releases/tags/snap_counts",
    timeout=30,
)
assets = [a["name"] for a in r.json()["assets"]]
print("snap_counts sample:", [a for a in assets if "2026" in a or "2025" in a][:20])

for rel in ["ftn_charting", "pfr_advstats", "officials"]:
    r = requests.get(
        f"https://api.github.com/repos/nflverse/nflverse-data/releases/tags/{rel}",
        timeout=30,
    )
    names = [a["name"] for a in r.json()["assets"]]
    print(rel, [n for n in names if "2026" in n or "2025" in n][:20])
