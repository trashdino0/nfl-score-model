"""Download and cache nflverse data used by the game-score model.

Examples:
    python scripts/download_all.py                  # fill in anything missing
    python scripts/download_all.py --refresh        # refresh 2026 + schedule
    python scripts/download_all.py --refresh --season 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nflscore.config import SEASONS
from nflscore.download import context_files, download_files, season_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache nflverse parquet files.")
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Only download this season (default: all configured seasons).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Overwrite existing files. Use this weekly to pick up new results.",
    )
    args = parser.parse_args()

    seasons = [args.season] if args.season else SEASONS
    files = []
    for season in seasons:
        files += season_files(season)
    files += context_files(args.season or seasons[-1])

    total_bytes = download_files(files, force=args.refresh)
    print(f"\nDownloaded {total_bytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
