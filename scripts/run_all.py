"""Rebuild the full pipeline from cached nflverse data.

Run with:
    $env:PYTHONPATH="src"; python scripts/run_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nflscore.backtest import run as run_backtest
from nflscore.build import build_team_games
from nflscore.config import PROCESSED
from nflscore.features import build_all
from nflscore.report import main as run_report
from nflscore.weekly import main as run_predict


def main() -> None:
    print("== Building team-game table ==")
    team_games = build_team_games()
    team_games.to_parquet(PROCESSED / "team_games.parquet", index=False)
    print(f"team_games: {team_games.shape}")

    print("\n== Building features ==")
    build_all()

    print("\n== Running backtest ==")
    run_backtest()

    print("\n== Predicting next unplayed week + card ==")
    run_predict()

    print("\n== Building report ==")
    run_report()


if __name__ == "__main__":
    main()
