"""One-command weekly workflow: refresh data, rebuild features, predict, write card.

Examples:
    # Normal weekly run (after the previous slate is final)
    python scripts/weekly.py --refresh

    # Predict a specific week without touching the network
    python scripts/weekly.py --season 2026 --week 10

    # Reuse the existing feature table (fast path; no rebuild)
    python scripts/weekly.py --week 10 --skip-build
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nflscore.pipeline import run_week


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one NFL week end to end.")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download current-season nflverse files before rebuilding.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Use the existing team_games/game_features tables.",
    )
    args = parser.parse_args()

    run_week(
        season=args.season,
        week=args.week,
        refresh=args.refresh,
        build=not args.skip_build,
    )


if __name__ == "__main__":
    main()
