"""Download and cache nflverse release data.

Shared by ``scripts/download_all.py`` and the weekly workflow so there is one
definition of which files the model needs.
"""

from __future__ import annotations

from pathlib import Path

import requests

from .config import RAW

BASE = "https://github.com/nflverse/nflverse-data/releases/download"


def season_files(season: int) -> list[tuple[str, str]]:
    """Files that change every week during a season."""
    return [
        (
            f"stats_team_week_{season}.parquet",
            f"{BASE}/stats_team/stats_team_week_{season}.parquet",
        ),
        (
            f"stats_player_week_{season}.parquet",
            f"{BASE}/stats_player/stats_player_week_{season}.parquet",
        ),
        (f"injuries_{season}.parquet", f"{BASE}/injuries/injuries_{season}.parquet"),
        (
            f"snap_counts_{season}.parquet",
            f"{BASE}/snap_counts/snap_counts_{season}.parquet",
        ),
        (
            f"advstats_week_def_{season}.parquet",
            f"{BASE}/pfr_advstats/advstats_week_def_{season}.parquet",
        ),
        (
            f"play_by_play_{season}.parquet",
            f"{BASE}/pbp/play_by_play_{season}.parquet",
        ),
    ]


def context_files(season: int) -> list[tuple[str, str]]:
    """Schedule, depth-chart and roster files used for context features."""
    return [
        ("games.parquet", f"{BASE}/schedules/games.parquet"),
        (
            f"depth_charts_{season}.parquet",
            f"{BASE}/depth_charts/depth_charts_{season}.parquet",
        ),
        (f"roster_{season}.parquet", f"{BASE}/rosters/roster_{season}.parquet"),
    ]


def download_files(files: list[tuple[str, str]], force: bool = False) -> int:
    """Download files into ``data/raw`` and return the number of bytes written."""
    RAW.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for name, url in files:
        dest = RAW / name
        if dest.exists() and dest.stat().st_size > 0 and not force:
            continue
        try:
            with requests.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                size = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                            size += len(chunk)
                tmp.replace(dest)
            total_bytes += size
            print(f"downloaded {name:40s} {size / 1e6:7.1f} MB")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"FAILED {name}: {exc}")
    return total_bytes


def refresh_season(season: int) -> int:
    """Force-refresh the current-season files and the shared schedule."""
    return download_files(season_files(season) + context_files(season), force=True)


def refresh_schedule() -> int:
    """Refresh only the shared schedule/results file (fast grading path)."""
    return download_files(
        [("games.parquet", f"{BASE}/schedules/games.parquet")], force=True
    )
