"""Evaluation metrics for game-level predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error


def _winner_mask(df: pd.DataFrame) -> pd.Series:
    if "home_win" in df.columns:
        return df["home_win"].notna()
    return df["margin"].notna() & (df["margin"] != 0)


def _home_win(df: pd.DataFrame) -> pd.Series:
    if "home_win" in df.columns:
        return df["home_win"]
    return (df["margin"] > 0).astype(float)


def evaluate(
    df: pd.DataFrame,
    pred_margin: str = "pred_margin",
    pred_total: str = "pred_total",
    win_prob: str | None = "win_prob",
    home_score: str = "pred_home_score",
    away_score: str = "pred_away_score",
    label: str = "model",
) -> dict:
    """Compute accuracy, error, correlation, and market-relative metrics."""
    d = df.copy()
    valid = d["home_score"].notna() & d["away_score"].notna()
    d = d[valid]
    if d.empty:
        return {"model": label, "n": 0}

    pm = d[pred_margin].astype(float)
    pt = d[pred_total].astype(float)
    ph = d[home_score].astype(float)
    pa = d[away_score].astype(float)

    wmask = _winner_mask(d)
    actual_win = _home_win(d)
    pred_win = (pm > 0).astype(float)
    win_acc = (
        float((pred_win[wmask] == actual_win[wmask]).mean()) if wmask.any() else np.nan
    )

    # ATS: model side vs closing line. Positive margin = home favored.
    # A pure market baseline has zero edge and therefore no ATS pick; report
    # NaN rather than silently picking one side of every game.
    ats_rate = np.nan
    ats_n = 0
    if "market_margin" in d.columns:
        has_edge = not np.allclose(
            pm.to_numpy(dtype=float),
            d["market_margin"].to_numpy(dtype=float),
            equal_nan=True,
        )
        if has_edge:
            edge = pm - d["market_margin"]
            actual_edge = d["margin"] - d["market_margin"]
            push = actual_edge == 0
            pick_home = edge > 0
            ats_win = (
                (pick_home & (actual_edge > 0)) | (~pick_home & (actual_edge < 0))
            ) & ~push
            ats_n = int((~push).sum())
            if ats_n:
                ats_rate = float(ats_win.sum() / ats_n)

    # Over/under: model total vs closing total.
    ou_rate = np.nan
    ou_n = 0
    if "market_total" in d.columns:
        has_edge = not np.allclose(
            pt.to_numpy(dtype=float),
            d["market_total"].to_numpy(dtype=float),
            equal_nan=True,
        )
        if has_edge:
            pick_over = pt > d["market_total"]
            actual_over = d["total"] - d["market_total"]
            push = actual_over == 0
            ou_win = (
                (pick_over & (actual_over > 0)) | (~pick_over & (actual_over < 0))
            ) & ~push
            ou_n = int((~push).sum())
            if ou_n:
                ou_rate = float(ou_win.sum() / ou_n)

    brier = np.nan
    logloss = np.nan
    if win_prob and win_prob in d.columns and wmask.any():
        p = d.loc[wmask, win_prob].astype(float).clip(1e-6, 1 - 1e-6)
        y = actual_win[wmask].astype(float)
        brier = float(np.mean((p - y) ** 2))
        logloss = float(log_loss(y, p, labels=[0, 1]))

    scores = pd.concat([d["home_score"], d["away_score"]], ignore_index=True)
    pred_scores = pd.concat([ph, pa], ignore_index=True)

    return {
        "model": label,
        "n": len(d),
        "win_acc": win_acc,
        "margin_mae": float(mean_absolute_error(d["margin"], pm)),
        "margin_rmse": float(mean_squared_error(d["margin"], pm) ** 0.5),
        "total_mae": float(mean_absolute_error(d["total"], pt)),
        "total_rmse": float(mean_squared_error(d["total"], pt) ** 0.5),
        "home_score_mae": float(mean_absolute_error(d["home_score"], ph)),
        "away_score_mae": float(mean_absolute_error(d["away_score"], pa)),
        "score_mae": float(mean_absolute_error(scores, pred_scores)),
        "margin_corr": float(np.corrcoef(d["margin"], pm)[0, 1]),
        "total_corr": float(np.corrcoef(d["total"], pt)[0, 1]),
        "ats_rate": ats_rate,
        "ats_n": ats_n,
        "ou_rate": ou_rate,
        "ou_n": ou_n,
        "brier": brier,
        "logloss": logloss,
    }


def per_week(df: pd.DataFrame, pred_margin: str = "pred_margin") -> pd.DataFrame:
    """Winner accuracy and MAE by week for one model."""
    rows = []
    for week, g in df.groupby("week", sort=True):
        valid = g["home_score"].notna() & g["home_win"].notna()
        g = g[valid]
        if g.empty:
            continue
        pred_win = (g[pred_margin] > 0).astype(float)
        rows.append(
            {
                "week": int(week),
                "n": len(g),
                "win_acc": float((pred_win == g["home_win"]).mean()),
                "margin_mae": float(mean_absolute_error(g["margin"], g[pred_margin])),
                "total_mae": float(mean_absolute_error(g["total"], g["pred_total"]))
                if "pred_total" in g.columns
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calibration(df: pd.DataFrame, prob_col: str = "win_prob", bins=5) -> pd.DataFrame:
    d = df[df["home_win"].notna()].copy()
    d = d[d[prob_col].notna()]
    edges = np.linspace(0, 1, bins + 1)
    d["bin"] = pd.cut(d[prob_col], edges, include_lowest=True)
    out = (
        d.groupby("bin", observed=True)
        .agg(
            n=("home_win", "size"),
            predicted=(prob_col, "mean"),
            actual=("home_win", "mean"),
        )
        .reset_index()
    )
    out["bin"] = out["bin"].astype(str)
    return out
