import requests

# List release tags to spot useful extras
r = requests.get(
    "https://api.github.com/repos/nflverse/nflverse-data/releases?per_page=100",
    timeout=30,
)
for rel in r.json():
    print(rel["tag_name"], len(rel.get("assets", [])))
