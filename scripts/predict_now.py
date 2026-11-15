"""One-click weekly predictions.

Refreshes the current-season nflverse data, rebuilds the feature table, predicts
the next unplayed week, and writes both the CSV and the Markdown card.

Run it directly:
    python scripts/predict_now.py

Or double-click ``run_predictions.cmd`` in the project root.

No arguments are needed. The model resolves the next unplayed week automatically,
so the same script moves from Week 9 to Week 10 once Week 9 is final.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nflscore.pipeline import run_week


def main() -> None:
    print("NFL weekly predictions")
    print("This refreshes data, rebuilds features, and writes the next card.\n")
    try:
        run_week(refresh=True, build=True)
    except SystemExit as exc:
        print(f"\n{exc}")
        print("Nothing to predict right now (the regular season may be over).")
        raise


if __name__ == "__main__":
    main()
