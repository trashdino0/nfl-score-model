"""End-to-end weekly pipeline: refresh, rebuild, predict, write card."""

from __future__ import annotations

import time
import warnings

import pandas as pd
from pandas.errors import PerformanceWarning

# The feature builder inserts hundreds of columns one at a time. The resulting
# fragmentation warnings are noisy and harmless for this batch workload.
warnings.filterwarnings("ignore", category=PerformanceWarning)

from .build import build_team_games
from .config import PROCESSED, RAW
from .download import refresh_season
from .features import build_all
from .weekly import main as write_weekly


def latest_season() -> int:
    games = pd.read_parquet(RAW / "games.parquet")
    return int(pd.to_numeric(games["season"], errors="coerce").max())


def run_week(
    season: int | None = None,
    week: int | None = None,
    refresh: bool = True,
    build: bool = True,
) -> None:
    """Run the weekly job once.

    ``season``/``week`` default to the next unplayed week. ``refresh`` downloads
    the current season's files; ``build`` rebuilds the team-game and feature
    tables. Passing ``build=False`` reuses the existing feature table.
    """
    started = time.time()

    if refresh:
        season_to_refresh = season or latest_season()
        print(f"== Refreshing {season_to_refresh} nflverse files ==")
        downloaded = refresh_season(season_to_refresh)
        if downloaded == 0:
            print("Refresh downloaded nothing; using cached data.")

    if build:
        print("\n== Building team-game table ==")
        team_games = build_team_games()
        team_games.to_parquet(PROCESSED / "team_games.parquet", index=False)
        print(f"team_games: {team_games.shape}")

        print("\n== Building features ==")
        build_all()

    print("\n== Predicting ==")
    write_weekly(season, week)

    print(f"\nDone in {time.time() - started:.1f}s")
