"""One-click grading for the most recent completed week.

Refreshes the nflverse schedule/results (a small download, not a full rebuild),
finds the newest fully completed week that has a saved prediction file, and
grades it.

Run it directly:
    python scripts/grade_last_week.py

Or double-click ``grade_last_week.cmd`` in the project root.

No arguments are needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from nflscore.config import OUTPUTS, RAW
from nflscore.download import refresh_schedule
from nflscore.grade import actuals_from_schedule, complete_weeks, grade_saved


def main() -> None:
    print("NFL last-week grader")
    print("Refreshing schedule/results...")
    downloaded = refresh_schedule()
    if downloaded == 0:
        print("Could not refresh the schedule; using the cached file.\n")

    games = pd.read_parquet(RAW / "games.parquet")
    season = int(pd.to_numeric(games["season"], errors="coerce").max())
    weeks = complete_weeks(season, games)

    if not weeks:
        print(f"No completed {season} regular-season weeks yet.")
        return

    chosen = None
    for week in weeks:
        path = OUTPUTS / f"week{week:02d}_predictions.csv"
        if path.exists():
            chosen = (week, path)
            break

    if chosen is None:
        print(f"No saved prediction file found for a completed {season} week.")
        print("Run predict_now.py before the games, then run this after they finish.")
        return

    week, path = chosen
    print(f"Grading {season} Week {week} from {path.name}.\n")
    grade_saved(
        season,
        week,
        path=str(path),
        actual=actuals_from_schedule(season, week),
    )


if __name__ == "__main__":
    main()
