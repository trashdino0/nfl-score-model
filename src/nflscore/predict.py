"""Generate predictions for the next unplayed NFL week.

Run with:
    $env:PYTHONPATH="src"; python -m nflscore.predict
    $env:PYTHONPATH="src"; python -m nflscore.predict --season 2026 --week 10
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import OUTPUTS, PROCESSED
from .model import NFLModel, add_market_baseline, feature_label, prob_from_margin


def resolve_target(
    season: int | None = None, week: int | None = None
) -> tuple[int, int]:
    """Pick the season/week to predict when the caller does not specify them.

    Default behaviour is the earliest week in the latest season that still has
    unplayed games, so the same command works every week of the season.
    """
    games = pd.read_parquet(PROCESSED / "game_features.parquet")
    open_games = games[games.home_score.isna()]

    if season is None:
        if open_games.empty:
            raise SystemExit("No unplayed games found in game_features.parquet.")
        season = int(open_games["season"].max())

    season_open = open_games[open_games.season == season]
    if season_open.empty:
        raise SystemExit(f"No unplayed games found for the {season} season.")

    if week is None:
        week = int(season_open["week"].min())

    if season_open[season_open.week == week].empty:
        raise SystemExit(f"No unplayed games found for {season} Week {week}.")

    return season, week


def prediction_path(week: int):
    return OUTPUTS / f"week{week:02d}_predictions.csv"


def _player_names() -> dict:
    players = pd.read_parquet(PROCESSED.parent / "raw" / "players.parquet")
    return dict(zip(players["gsis_id"], players["display_name"]))


def _context(row, names: dict) -> str:
    parts = []

    for side in ["home", "away"]:
        qb = row.get(f"{side}_qb_name")
        last = row.get(f"{side}_last_qb_name")
        change = row.get(f"{side}_qb_change")
        out = row.get(f"{side}_qb_out")
        if pd.notna(qb):
            if change == 1 and pd.notna(last):
                parts.append(
                    f"{side.upper()} QB: {qb} in for {last}"
                    + (" (out)" if out == 1 else "")
                )
            elif out == 1:
                parts.append(f"{side.upper()} QB: {qb} (listed out)")
            else:
                parts.append(f"{side.upper()} QB: {qb}")

    for grp in ["wr", "rb", "te"]:
        for side, suffix in [("home", "h"), ("away", "a")]:
            out = row.get(f"{grp}_out_{suffix}")
            questionable = row.get(f"{grp}_questionable_{suffix}")
            if out == 1 or questionable == 1:
                pid = row.get(f"{grp}_top_id_{suffix}")
                name = names.get(pid, f"top {grp.upper()}")
                team = row.get(f"{side}_team")
                status = "out" if out == 1 else "questionable"
                parts.append(f"{team} {grp.upper()} {name} {status}")

    wind = row.get("wind")
    temp = row.get("temp")
    if pd.notna(wind) and wind >= 15:
        parts.append(f"wind {wind:.0f} mph")
    if pd.notna(temp) and temp <= 32:
        parts.append(f"temp {temp:.0f} F")
    if row.get("is_dome") == 1:
        parts.append("dome")

    return "; ".join(parts)


def _drivers(contrib: pd.DataFrame, row_idx, n: int = 4) -> str:
    s = contrib.loc[row_idx]
    top = s.abs().sort_values(ascending=False).head(n)
    pieces = []
    for feat in top.index:
        val = s[feat]
        if abs(val) < 0.01:
            continue
        pieces.append(f"{feature_label(feat)} {val:+.2f}")
    return "; ".join(pieces) if pieces else "market-anchored"


def predict_slate(season: int | None = None, week: int | None = None) -> pd.DataFrame:
    season, week = resolve_target(season, week)
    games = pd.read_parquet(PROCESSED / "game_features.parquet")
    test = games[
        (games.season == season) & (games.week == week) & games.home_score.isna()
    ].copy()
    if test.empty:
        raise SystemExit(f"No unplayed games found for {season} Week {week}.")

    # Leakage guard: when backfilling an earlier week, only train on games played
    # before that slate. For the normal next-week run this is every completed game.
    target_day = test["gameday"].min()
    train = games[(games.season >= 2018) & games.home_score.notna()].copy()
    if pd.notna(target_day):
        train = train[train["gameday"] < target_day].copy()

    # Keep the raw market state: the model distinguishes a real closing line
    # from a pure-model fallback, and the card must not call the latter a line.
    if "market_margin" in test.columns:
        test["market_line_missing"] = test["market_margin"].isna()
    else:
        test["market_line_missing"] = True

    model = NFLModel().fit(train)
    pred = add_market_baseline(model.predict(test))
    pred["pure_win_prob"] = prob_from_margin(
        pred["pure_margin"], model.pure_margin_sigma_
    )
    contrib = model.margin_contributions(test) * model.config.margin_shrink

    names = _player_names()
    pred["key_drivers"] = [_drivers(contrib, idx) for idx in pred.index]
    pred["context"] = [_context(row, names) for _, row in pred.iterrows()]

    pred["market_pick"] = [
        home if m > 0 else away
        for home, away, m in zip(pred.home_team, pred.away_team, pred.market_margin)
    ]
    pred["pure_pick"] = [
        home if m > 0 else away
        for home, away, m in zip(pred.home_team, pred.away_team, pred.pure_margin)
    ]
    pred["model_vs_market"] = pred["model_edge"].map(lambda x: f"{x:+.1f} pts")
    pred["pred_score"] = pred.apply(
        lambda r: f"{r.pred_away_score:.1f} - {r.pred_home_score:.1f}", axis=1
    )
    pred["market_score"] = pred.apply(
        lambda r: f"{r.market_away_score:.1f} - {r.market_home_score:.1f}", axis=1
    )
    pred["pure_score"] = pred.apply(
        lambda r: f"{r.pure_away_score:.1f} - {r.pure_home_score:.1f}", axis=1
    )
    pred["pred_home_score_r"] = pred["pred_home_score"].round().astype(int)
    pred["pred_away_score_r"] = pred["pred_away_score"].round().astype(int)
    pred["pure_pick_win_prob"] = np.where(
        pred["pure_margin"] > 0, pred["pure_win_prob"], 1 - pred["pure_win_prob"]
    )

    # 80% score ranges come from the joint residual bootstrap fitted on
    # out-of-fold errors, so they are conditional on this game's projection
    # and preserve the margin/total correlation.
    for side in ["home", "away"]:
        for q in ["low", "med", "high"]:
            col = f"pred_{side}_score_{q}"
            pred[col] = pred[col].round().astype(int)

    keep = [
        "game_id",
        "gameday",
        "away_team",
        "home_team",
        "pred_winner",
        "pred_margin",
        "pred_total",
        "pred_home_score",
        "pred_away_score",
        "pred_score",
        "pred_home_score_r",
        "pred_away_score_r",
        "pred_home_score_low",
        "pred_home_score_med",
        "pred_home_score_high",
        "pred_away_score_low",
        "pred_away_score_med",
        "pred_away_score_high",
        "win_prob",
        "win_prob_source",
        "pick_win_prob",
        "model_win_prob",
        "market_win_prob",
        "sim_win_prob",
        "model_confidence",
        "cover_prob",
        "push_prob",
        "over_prob",
        "total_push_prob",
        "pick_cover_prob",
        "pure_margin",
        "pure_total",
        "pure_home_score",
        "pure_away_score",
        "pure_score",
        "pure_pick",
        "pure_win_prob",
        "pure_pick_win_prob",
        "market_line_missing",
        "line_source",
        "market_margin",
        "market_total",
        "market_home_score",
        "market_away_score",
        "market_score",
        "market_pick",
        "model_edge",
        "model_vs_market",
        "ats_pick",
        "ats_play",
        "home_qb_name",
        "away_qb_name",
        "home_last_qb_name",
        "away_last_qb_name",
        "home_qb_change",
        "away_qb_change",
        "home_qb_out",
        "away_qb_out",
        "home_projected_qb"
        if "home_projected_qb" in pred.columns
        else "home_qb_projected",
        "away_projected_qb"
        if "away_projected_qb" in pred.columns
        else "away_qb_projected",
        "key_drivers",
        "context",
    ]
    keep = [c for c in keep if c in pred.columns]
    out = pred[keep].copy()
    out = out.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return out, model


def main(season: int | None = None, week: int | None = None) -> None:
    season, week = resolve_target(season, week)
    out, model = predict_slate(season, week)
    path = prediction_path(week)
    out.to_csv(path, index=False)

    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 60)
    print(f"Trained on {model.train_rows_} completed games (2018+).")
    print(
        f"Margin sigma: {model.margin_sigma_:.2f}; "
        f"margin shrink: {model.config.margin_shrink:.2f}; "
        f"total shrink: {model.config.total_shrink:.2f}; "
        f"win-prob model weight: {model.win_prob_model_weight_:.2f}"
    )
    print(f"\n=== {season} Week {week}: {len(out)} unplayed games ===")
    cols = [
        "gameday",
        "away_team",
        "home_team",
        "pred_winner",
        "pred_score",
        "pick_win_prob",
        "market_win_prob",
        "cover_prob",
        "market_margin",
        "market_total",
        "ats_play",
        "key_drivers",
    ]
    print(out[cols].to_string(index=False))
    print(f"\nWrote {path}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Predict one unplayed NFL week.")
    parser.add_argument("--season", type=int, default=None, help="e.g. 2026")
    parser.add_argument("--week", type=int, default=None, help="e.g. 10")
    args = parser.parse_args()
    main(args.season, args.week)


if __name__ == "__main__":
    _cli()
