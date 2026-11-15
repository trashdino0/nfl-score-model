"""Grade a saved weekly prediction file against final scores.

Run after the week's games are final:
    $env:PYTHONPATH="src"; python -m nflscore.grade --week 9
    $env:PYTHONPATH="src"; python -m nflscore.grade --season 2026 --week 10
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import OUTPUTS, PROCESSED, RAW
from .predict import prediction_path


def _rate(wins: int, losses: int) -> float:
    n = wins + losses
    return wins / n if n else float("nan")


def grade_frame(pred: pd.DataFrame, actual: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join predictions to finals and compute honest weekly metrics."""
    d = pred.merge(actual, on="game_id", how="left")
    graded = d[d["home_score"].notna()].copy()
    pending = len(d) - len(graded)

    if graded.empty:
        return graded, {"n": len(d), "graded": 0, "pending": pending}

    graded["actual_winner"] = np.select(
        [graded["margin"] > 0, graded["margin"] < 0],
        [graded["home_team"], graded["away_team"]],
        default="TIE",
    )
    graded["pick_correct"] = np.where(
        graded["actual_winner"] == "TIE",
        np.nan,
        (graded["pred_winner"] == graded["actual_winner"]).astype(float),
    )

    graded["home_score_err"] = (graded["pred_home_score"] - graded["home_score"]).abs()
    graded["away_score_err"] = (graded["pred_away_score"] - graded["away_score"]).abs()
    graded["margin_err"] = (graded["pred_margin"] - graded["margin"]).abs()
    graded["total_err"] = (graded["pred_total"] - graded["total"]).abs()

    cover = graded["margin"] - graded["market_margin"]
    pick_home = graded["ats_pick"] == graded["home_team"]
    graded["ats_result"] = np.select(
        [
            graded["market_margin"].isna(),
            cover == 0,
            pick_home & (cover > 0),
            (~pick_home) & (cover < 0),
        ],
        ["no_line", "push", "win", "win"],
        default="loss",
    )

    if "ou_diagnostic" in graded.columns:
        ou_pick = graded["ou_diagnostic"]
    elif "ou_pick" in graded.columns:
        ou_pick = graded["ou_pick"]
    else:
        ou_pick = pd.Series(np.nan, index=graded.index)

    over = graded["total"] > graded["market_total"]
    graded["ou_result"] = np.select(
        [
            ou_pick.isna(),
            graded["market_total"].isna(),
            graded["total"] == graded["market_total"],
            (ou_pick == "Over") & over,
            (ou_pick == "Under") & (~over),
        ],
        ["no_model", "no_line", "push", "win", "win"],
        default="loss",
    )

    wins = int((graded["pick_correct"] == 1).sum())
    losses = int((graded["pick_correct"] == 0).sum())
    ats = graded["ats_result"].value_counts()
    ou = graded["ou_result"].value_counts()

    summary = {
        "n": len(d),
        "graded": len(graded),
        "pending": pending,
        "wins": wins,
        "losses": losses,
        "ties": int((graded["actual_winner"] == "TIE").sum()),
        "win_acc": _rate(wins, losses),
        "score_mae": float(
            pd.concat([graded["home_score_err"], graded["away_score_err"]]).mean()
        ),
        "margin_mae": float(graded["margin_err"].mean()),
        "total_mae": float(graded["total_err"].mean()),
        "ats_w": int(ats.get("win", 0)),
        "ats_l": int(ats.get("loss", 0)),
        "ats_p": int(ats.get("push", 0)),
        "ats_rate": _rate(int(ats.get("win", 0)), int(ats.get("loss", 0))),
        "ou_w": int(ou.get("win", 0)),
        "ou_l": int(ou.get("loss", 0)),
        "ou_p": int(ou.get("push", 0)),
        "ou_rate": _rate(int(ou.get("win", 0)), int(ou.get("loss", 0))),
    }

    decisive = graded["actual_winner"] != "TIE"
    if "win_prob" in graded.columns:
        p = graded.loc[decisive, "win_prob"].astype(float)
        y = (graded.loc[decisive, "actual_winner"] == graded.loc[decisive, "home_team"]).astype(float)
        summary["brier_win"] = float(((p - y) ** 2).mean()) if len(p) else float("nan")
    if "cover_prob" in graded.columns:
        no_push = cover != 0
        p = graded.loc[no_push, "cover_prob"].astype(float)
        y = (cover[no_push] > 0).astype(float)
        summary["brier_cover"] = float(((p - y) ** 2).mean()) if len(p) else float("nan")
    if "over_prob" in graded.columns:
        no_push = graded["total"] != graded["market_total"]
        p = graded.loc[no_push, "over_prob"].astype(float)
        y = (graded.loc[no_push, "total"] > graded.loc[no_push, "market_total"]).astype(float)
        summary["brier_over"] = float(((p - y) ** 2).mean()) if len(p) else float("nan")
    return graded, summary


def actuals_from_schedule(season: int, week: int) -> pd.DataFrame:
    """Final scores straight from the nflverse schedule.

    The weekly grader uses this so it can grade as soon as results are posted,
    without waiting for the full team-game/feature rebuild.
    """
    games = pd.read_parquet(RAW / "games.parquet")
    g = games[
        (games.season == season) & (games.week == week) & (games.game_type == "REG")
    ].copy()
    g["home_score"] = pd.to_numeric(g["home_score"], errors="coerce")
    g["away_score"] = pd.to_numeric(g["away_score"], errors="coerce")
    g["margin"] = g["home_score"] - g["away_score"]
    g["total"] = g["home_score"] + g["away_score"]
    return g[["game_id", "home_score", "away_score", "margin", "total"]]


def complete_weeks(season: int, games: pd.DataFrame | None = None) -> list[int]:
    """Completed weeks, newest first, for one regular season."""
    if games is None:
        games = pd.read_parquet(RAW / "games.parquet")
    g = games[(games.season == season) & (games.game_type == "REG")].copy()
    if g.empty:
        return []

    g["home_score"] = pd.to_numeric(g["home_score"], errors="coerce")
    scheduled = g.groupby("week").size()
    finals = g.dropna(subset=["home_score"]).groupby("week").size()
    weeks = [
        int(week) for week in scheduled.index if finals.get(week, 0) == scheduled[week]
    ]
    return sorted(weeks, reverse=True)


def grade_saved(
    season: int,
    week: int,
    path: str | None = None,
    actual: pd.DataFrame | None = None,
) -> dict:
    """Grade one saved prediction file and print the summary."""
    pred = pd.read_csv(path or prediction_path(week))
    if "season" not in pred.columns:
        pred["season"] = season

    if actual is None:
        games = pd.read_parquet(PROCESSED / "game_features.parquet")
        actual = games[(games.season == season) & (games.week == week)][
            ["game_id", "home_score", "away_score", "margin", "total"]
        ]

    graded, summary = grade_frame(pred, actual)

    print(
        f"{season} Week {week} - graded {summary['graded']} of {summary['n']} games "
        f"({summary['pending']} pending)"
    )
    if summary["graded"] == 0:
        print("No finals yet. Re-run after the slate.")
        return summary

    out = OUTPUTS / f"week{week:02d}_graded.csv"
    graded.to_csv(out, index=False)

    print(
        f"Winner accuracy: {summary['wins']}-{summary['losses']} "
        f"({summary['win_acc']:.1%})"
    )
    print(
        f"Score MAE: {summary['score_mae']:.2f} | "
        f"Margin MAE: {summary['margin_mae']:.2f} | "
        f"Total MAE: {summary['total_mae']:.2f}"
    )
    print(
        f"ATS: {summary['ats_w']}-{summary['ats_l']}"
        + (f"-{summary['ats_p']}" if summary["ats_p"] else "")
        + f" ({summary['ats_rate']:.1%})"
    )
    print(
        f"O/U: {summary['ou_w']}-{summary['ou_l']}"
        + (f"-{summary['ou_p']}" if summary["ou_p"] else "")
        + f" ({summary['ou_rate']:.1%})"
    )

    cols = [
        "away_team",
        "home_team",
        "pred_winner",
        "pred_home_score",
        "pred_away_score",
        "home_score",
        "away_score",
        "pick_correct",
        "ats_result",
        "ou_result",
    ]
    print()
    print(graded[cols].round(1).to_string(index=False))
    print(f"\nWrote {out}")
    return summary


def main(
    season: int | None = None, week: int | None = None, path: str | None = None
) -> None:
    games = pd.read_parquet(PROCESSED / "game_features.parquet")

    if season is None:
        season = int(games["season"].max())
    if week is None:
        raise SystemExit("Pass --week (for example --week 9).")

    grade_saved(season, week, path)


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Grade a saved weekly prediction file."
    )
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()
    main(args.season, args.week, args.file)


if __name__ == "__main__":
    _cli()
