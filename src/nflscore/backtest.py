"""Walk-forward backtests for the NFL score model.

Run with:
    $env:PYTHONPATH="src"; python -m nflscore.backtest
"""

from __future__ import annotations

from math import erf, sqrt

import pandas as pd

from .config import OUTPUTS, PROCESSED, TRAIN_START
from .metrics import calibration, evaluate
from .model import NFLModel, add_market_baseline


def _prob_from_margin(margin: pd.Series, sigma: float = 13.5) -> pd.Series:
    return pd.Series(
        [0.5 * (1 + erf((m / sigma) / sqrt(2))) for m in margin], index=margin.index
    )


def _add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    out = add_market_baseline(df)
    out["market_win_prob"] = out["market_home_win_prob"]
    out["naive_win_prob"] = _prob_from_margin(out["naive_margin"])
    out["pure_win_prob"] = _prob_from_margin(out["pure_margin"])
    return out


SLIM_COLS = [
    "game_id",
    "season",
    "week",
    "test_week",
    "gameday",
    "away_team",
    "home_team",
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
    "pure_home_score",
    "pure_away_score",
    "pure_win_prob",
    "naive_margin",
    "naive_total",
    "naive_win_prob",
    "model_edge",
    "total_edge",
    "ats_pick",
    "ou_pick",
    "home_qb_name",
    "away_qb_name",
    "home_qb_change",
    "away_qb_change",
    "home_qb_out",
    "away_qb_out",
    "train_rows",
]


def _slim(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in SLIM_COLS if c in df.columns]].copy()


def _summary_row(df: pd.DataFrame, prefix: str, label: str) -> dict:
    row = evaluate(
        df,
        pred_margin=f"{prefix}margin" if prefix else "pred_margin",
        pred_total=f"{prefix}total" if prefix else "pred_total",
        win_prob=f"{prefix}win_prob" if prefix else "win_prob",
        home_score=f"{prefix}home_score" if prefix else "pred_home_score",
        away_score=f"{prefix}away_score" if prefix else "pred_away_score",
        label=label,
    )
    return row


def season_validation(
    games: pd.DataFrame, seasons=range(2021, 2026)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on all prior seasons, test on each held-out season."""
    rows = []
    preds = []
    for season in seasons:
        train = games[(games.season >= TRAIN_START) & (games.season < season)]
        test = games[games.season == season].copy()
        model = NFLModel().fit(train)
        pred = _add_baselines(model.predict(test))
        pred["test_season"] = season
        preds.append(pred)
        rows.append(_summary_row(pred, "market_", f"{season} market"))
        rows.append(_summary_row(pred, "naive_", f"{season} naive"))
        rows.append(_summary_row(pred, "pure_", f"{season} pure_stats"))
        rows.append(_summary_row(pred, "", f"{season} hybrid"))
    summary = pd.DataFrame(rows)
    pred_df = pd.concat(preds, ignore_index=True)
    return summary, pred_df


def weekly_2026(games: pd.DataFrame, weeks=range(1, 9)) -> pd.DataFrame:
    """Weekly walk-forward: only games before the target week are used."""
    preds = []
    for week in weeks:
        train = games[
            (games.season >= TRAIN_START)
            & ((games.season < 2026) | ((games.season == 2026) & (games.week < week)))
        ]
        test = games[(games.season == 2026) & (games.week == week)].copy()
        model = NFLModel().fit(train)
        pred = _add_baselines(model.predict(test))
        pred["test_week"] = week
        pred["train_rows"] = model.train_rows_
        preds.append(pred)
    return pd.concat(preds, ignore_index=True)


def run() -> None:
    games = pd.read_parquet(PROCESSED / "game_features.parquet")

    # ---- 2021-2025 season-level validation (model selection) ----
    val_summary, val_preds = season_validation(games)
    val_summary.to_csv(OUTPUTS / "validation_2021_2025_summary.csv", index=False)
    _slim(val_preds).to_csv(
        OUTPUTS / "validation_2021_2025_predictions.csv", index=False
    )

    # ---- 2026 Weeks 1-8 weekly walk-forward ----
    bt = weekly_2026(games)
    _slim(bt).to_csv(OUTPUTS / "backtest_2026_weeks1_8.csv", index=False)

    # ---- Summaries ----
    summary_rows = []
    for prefix, label in [
        ("market_", "Market closing line"),
        ("naive_", "Naive scoring baseline"),
        ("pure_", "Pure stats ridge"),
        ("", "Hybrid market+stats"),
    ]:
        summary_rows.append(_summary_row(bt, prefix, label))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUTS / "backtest_summary.csv", index=False)

    # Per-week table for the three main models.
    week_rows = []
    for week, g in bt.groupby("test_week"):
        row = {"week": int(week), "n": len(g)}
        for prefix, name in [("market_", "market"), ("pure_", "pure"), ("", "hybrid")]:
            s = _summary_row(g, prefix, name)
            row[f"{name}_win_acc"] = s["win_acc"]
            row[f"{name}_margin_mae"] = s["margin_mae"]
            row[f"{name}_total_mae"] = s["total_mae"]
            row[f"{name}_score_mae"] = s["score_mae"]
        week_rows.append(row)
    week_table = pd.DataFrame(week_rows)
    week_table.to_csv(OUTPUTS / "backtest_per_week.csv", index=False)

    cal = calibration(bt, "win_prob")
    cal.to_csv(OUTPUTS / "backtest_calibration.csv", index=False)

    # ---- Print ----
    pd.set_option("display.width", 240)
    print("\n=== 2021-2025 season-level validation (average) ===")
    v = val_summary.copy()
    v["base"] = v["model"].str.replace(r"^\d{4} ", "", regex=True)
    print(
        v.groupby("base")[
            ["win_acc", "margin_mae", "total_mae", "score_mae", "brier", "logloss"]
        ]
        .mean()
        .round(3)
        .to_string()
    )
    print("\n=== 2026 Weeks 1-8 pooled ===")
    print(summary.round(3).to_string(index=False))
    print("\n=== 2026 per week ===")
    print(week_table.round(3).to_string(index=False))
    print("\n=== 2026 calibration ===")
    print(cal.round(3).to_string(index=False))
    print(f"\nWrote outputs to {OUTPUTS}")


if __name__ == "__main__":
    run()
