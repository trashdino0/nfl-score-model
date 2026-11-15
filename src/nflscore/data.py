"""Load and cache nflverse data."""

from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from .config import RAW, SEASON_TYPE, SEASONS


def _read(path: Path, columns=None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def _concat(files: Iterable[Path], columns=None) -> pd.DataFrame:
    frames = [_read(p, columns=columns) for p in files if p.exists()]
    if not frames:
        raise FileNotFoundError("No cached parquet files found")
    return pd.concat(frames, ignore_index=True)


def load_games(seasons: Iterable[int] = SEASONS) -> pd.DataFrame:
    games = _read(RAW / "games.parquet")
    games = games[games.season.isin(list(seasons))].copy()
    games = games[games.game_type == SEASON_TYPE].copy()
    games["gameday"] = pd.to_datetime(games["gameday"], errors="coerce")
    return games.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def load_team_week(seasons: Iterable[int] = SEASONS) -> pd.DataFrame:
    files = [RAW / f"stats_team_week_{s}.parquet" for s in seasons]
    df = _concat(files)
    df = df[df.season_type == SEASON_TYPE].copy()
    return df


def load_player_week(seasons: Iterable[int] = SEASONS) -> pd.DataFrame:
    files = [RAW / f"stats_player_week_{s}.parquet" for s in seasons]
    df = _concat(files)
    df = df[df.season_type == SEASON_TYPE].copy()
    return df


def load_injuries(seasons: Iterable[int] = SEASONS) -> pd.DataFrame:
    files = [RAW / f"injuries_{s}.parquet" for s in seasons]
    df = _concat(files)
    if "season_type" in df.columns:
        df = df[df.season_type == SEASON_TYPE].copy()
    return df


def load_snaps(seasons: Iterable[int] = SEASONS) -> pd.DataFrame:
    files = [RAW / f"snap_counts_{s}.parquet" for s in seasons]
    df = _concat(files)
    if "game_type" in df.columns:
        df = df[df.game_type == SEASON_TYPE].copy()
    return df


def load_advdef(seasons: Iterable[int] = SEASONS) -> pd.DataFrame:
    files = [RAW / f"advstats_week_def_{s}.parquet" for s in seasons if s >= 2018]
    if not files:
        return pd.DataFrame()
    df = _concat(files)
    if "game_type" in df.columns:
        df = df[df.game_type == SEASON_TYPE].copy()
    return df


def iter_pbp(seasons: Iterable[int] = SEASONS) -> Iterator[pd.DataFrame]:
    """Yield one season of regular-season PBP at a time to keep memory bounded."""
    for season in seasons:
        path = RAW / f"play_by_play_{season}.parquet"
        if not path.exists():
            continue
        df = _read(path)
        df = df[df.season_type == SEASON_TYPE].copy()
        yield season, df


def load_players() -> pd.DataFrame:
    return _read(RAW / "players.parquet")


def load_depth_charts(season: int = 2026) -> pd.DataFrame:
    return _read(RAW / f"depth_charts_{season}.parquet")


def load_roster(season: int = 2026) -> pd.DataFrame:
    return _read(RAW / f"roster_{season}.parquet")
