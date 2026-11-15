"""Run a *legacy* copy of the package on the current walk-forward windows.

This script is executed as a child process by ``scripts/version_delta.py`` with
``LEGACY_SRC`` pointing at a git worktree of an older commit.  It deliberately
does not import the repository's own ``nflscore``; it imports whatever package
lives at ``LEGACY_SRC`` so the comparison is against real historical code, not
a re-implementation that could drift.

Training uses the *current* walk-forward rule (``gameday < slate date``) so the
delta isolates model changes from the backtest-alignment fix.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="game_features.parquet")
    parser.add_argument("--out", required=True, help="output parquet")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--weeks", required=True, help="comma separated, e.g. 1,2,3")
    args = parser.parse_args()

    legacy_src = os.environ.get("LEGACY_SRC")
    if not legacy_src:
        raise SystemExit("LEGACY_SRC must point at the legacy src directory")
    sys.path.insert(0, legacy_src)

    import pandas as pd  # noqa: PLC0415

    import nflscore  # noqa: PLC0415,F401
    from nflscore.backtest import _add_baselines  # noqa: PLC0415
    from nflscore.config import TRAIN_START  # noqa: PLC0415
    from nflscore.model import NFLModel  # noqa: PLC0415

    if "site-packages" not in nflscore.__file__ and "src" not in nflscore.__file__:
        raise SystemExit(f"unexpected nflscore location: {nflscore.__file__}")

    games = pd.read_parquet(args.data)
    weeks = [int(w) for w in args.weeks.split(",") if w.strip()]

    frames = []
    for week in weeks:
        test_week = games[(games.season == args.season) & (games.week == week)].copy()
        if test_week.empty:
            continue
        for gameday, slate in test_week.groupby("gameday", sort=True):
            train = games[
                (games.season >= TRAIN_START) & games.home_score.notna()
            ].copy()
            if pd.notna(gameday):
                train = train[train["gameday"] < gameday]

            model = NFLModel().fit(train)
            pred = _add_baselines(model.predict(slate))
            pred["test_week"] = week
            pred["train_rows"] = model.train_rows_
            frames.append(pred)

    if not frames:
        raise SystemExit("no predictions produced")

    out = pd.concat(frames, ignore_index=True)
    out["margin"] = out["home_score"] - out["away_score"]
    out["total"] = out["home_score"] + out["away_score"]
    out["home_win"] = (out["margin"] > 0).astype(float).where(out["margin"] != 0)

    keep = [
        "game_id",
        "season",
        "test_week",
        "gameday",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "margin",
        "total",
        "home_win",
        "market_margin",
        "market_total",
        "market_home_win_prob",
        "pred_margin",
        "pred_total",
        "pred_home_score",
        "pred_away_score",
        "pred_winner",
        "pred_home_win",
        "win_prob",
        "model_win_prob",
        "market_win_prob",
        "pure_margin",
        "pure_total",
        "pure_win_prob",
        "model_edge",
        "ats_pick",
        "train_rows",
    ]
    optional = [c for c in ["ou_pick", "total_edge", "ats_play", "naive_margin"] if c in out.columns]
    out = out[[c for c in keep if c in out.columns] + optional]

    # Old-style 80% intervals, reconstructed from the legacy constants, so
    # interval coverage can be compared even though the columns are gone.
    out["old_home_low"] = out["pred_home_score"] - 12.9
    out["old_home_high"] = out["pred_home_score"] + 11.0
    out["old_away_low"] = out["pred_away_score"] - 11.9
    out["old_away_high"] = out["pred_away_score"] + 11.3

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"legacy predictions: {len(out)} games -> {args.out}")


if __name__ == "__main__":
    main()
