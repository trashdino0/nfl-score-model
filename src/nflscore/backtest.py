"""Walk-forward backtests for the NFL score model.

Run with:
    $env:PYTHONPATH="src"; python -m nflscore.backtest
"""

from __future__ import annotations

import pandas as pd

from .config import OUTPUTS, PROCESSED, TRAIN_START
from .metrics import calibration, evaluate
from .model import NFLModel, add_market_baseline, prob_from_margin


def _prob_from_margin(margin: pd.Series, sigma: float = 13.5) -> pd.Series:
    return pd.Series(
        prob_from_margin(margin.to_numpy(dtype=float), sigma), index=margin.index
    )


def _add_baselines(df: pd.DataFrame, model: NFLModel | None = None) -> pd.DataFrame:
    out = add_market_baseline(df)
    out["market_win_prob"] = out["market_home_win_prob"]
    out["naive_win_prob"] = _prob_from_margin(out["naive_margin"])
    pure_sigma = model.pure_margin_sigma_ if model is not None else 13.5
    out["pure_win_prob"] = _prob_from_margin(out["pure_margin"], pure_sigma)
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
    "market_line_missing",
    "line_source",
    "pred_margin",
    "pred_total",
    "pred_home_score",
    "pred_away_score",
    "pred_winner",
    "pred_home_win",
    "win_prob",
    "win_prob_source",
    "pick_win_prob",
    "model_win_prob",
    "market_win_prob",
    "sim_win_prob",
    "cover_prob",
    "push_prob",
    "over_prob",
    "total_push_prob",
    "pred_home_score_low",
    "pred_home_score_med",
    "pred_home_score_high",
    "pred_away_score_low",
    "pred_away_score_med",
    "pred_away_score_high",
    "pick_cover_prob",
    "pure_margin",
    "pure_total",
    "pure_home_score",
    "pure_away_score",
    "pure_win_prob",
    "naive_margin",
    "naive_total",
    "naive_win_prob",
    "model_edge",
    "ats_pick",
    "ats_play",
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
        pred = _add_baselines(model.predict(test), model)
        pred["test_season"] = season
        preds.append(pred)
        rows.append(_summary_row(pred, "market_", f"{season} market"))
        rows.append(_summary_row(pred, "naive_", f"{season} naive"))
        rows.append(_summary_row(pred, "pure_", f"{season} pure_stats"))
        rows.append(_summary_row(pred, "", f"{season} hybrid"))
    summary = pd.DataFrame(rows)
    pred_df = pd.concat(preds, ignore_index=True)
    return summary, pred_df


def weekly_2026(
    games: pd.DataFrame, weeks=range(1, 9), season: int = 2026
) -> pd.DataFrame:
    """Game-day walk-forward that mirrors the live prediction path.

    The live job trains on ``gameday < the first unplayed kickoff``, so a
    Sunday slate already knows the Thursday result.  Grouping by game day (not
    by week) reproduces that: Thursday's model trains through Wednesday,
    Sunday's model trains through Saturday, and Monday's through Sunday.
    """
    preds = []
    for week in weeks:
        test_week = games[(games.season == season) & (games.week == week)].copy()
        if test_week.empty:
            continue
        for gameday, slate in test_week.groupby("gameday", sort=True):
            train = games[
                (games.season >= TRAIN_START) & games.home_score.notna()
            ].copy()
            if pd.notna(gameday):
                train = train[train["gameday"] < gameday]
            model = NFLModel().fit(train)
            pred = _add_baselines(model.predict(slate), model)
            pred["test_week"] = week
            pred["train_rows"] = model.train_rows_
            preds.append(pred)
    if not preds:
        return pd.DataFrame()
    return pd.concat(preds, ignore_index=True)


def run() -> None:
    games = pd.read_parquet(PROCESSED / "game_features.parquet")

    # ---- 2021-2025 season-level validation (model selection) ----
    val_summary, val_preds = season_validation(games)
    val_summary.to_csv(OUTPUTS / "validation_2021_2025_summary.csv", index=False)
    _slim(val_preds).to_csv(
        OUTPUTS / "validation_2021_2025_predictions.csv", index=False
    )

    # ---- Current-season weekly walk-forward, every completed week ----
    season = int(games["season"].max())
    completed = sorted(
        int(week)
        for week, g in games[games.season == season].groupby("week")
        if len(g) and g["home_score"].notna().all()
    )
    bt = weekly_2026(games, weeks=completed, season=season)
    _slim(bt).to_csv(OUTPUTS / f"backtest_{season}.csv", index=False)
    early = bt[bt["test_week"] <= 8].copy()
    _slim(early).to_csv(OUTPUTS / f"backtest_{season}_weeks1_8.csv", index=False)
    summary = pd.DataFrame(
        [
            _summary_row(bt, prefix, label)
            for prefix, label in [
                ("market_", "Market closing line"),
                ("naive_", "Naive scoring baseline"),
                ("pure_", "Pure stats ridge"),
                ("", "Hybrid market+stats"),
            ]
        ]
    )
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
    cal_all = pd.concat(
        [
            calibration(bt, "win_prob", label="blend"),
            calibration(bt, "model_win_prob", label="model"),
            calibration(bt, "market_win_prob", label="market"),
        ],
        ignore_index=True,
    )
    cal_all.to_csv(OUTPUTS / "backtest_calibration_all.csv", index=False)

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
    print(f"\n=== {season} weekly walk-forward, Weeks {min(completed)}-{max(completed)} pooled ===")
    print(summary.round(3).to_string(index=False))
    print("\n=== 2026 per week ===")
    print(week_table.round(3).to_string(index=False))
    print("\n=== 2026 calibration (blend) ===")
    print(cal.round(3).to_string(index=False))
    print(f"\nWrote outputs to {OUTPUTS}")


if __name__ == "__main__":
    run()
