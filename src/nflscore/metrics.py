"""Evaluation metrics for game-level predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error

from .evaluation import wilson_interval


def _roi(wins: int, losses: int, odds: float = -110) -> float:
    """Return on investment for 1-unit bets at American ``odds``."""
    n = wins + losses
    if not n:
        return float("nan")
    payout = 100 / abs(odds) if odds < 0 else odds / 100
    return (wins * payout - losses) / n


def _winner_mask(df: pd.DataFrame) -> pd.Series:
    if "home_win" in df.columns:
        return df["home_win"].notna()
    return df["margin"].notna() & (df["margin"] != 0)


def _home_win(df: pd.DataFrame) -> pd.Series:
    if "home_win" in df.columns:
        return df["home_win"]
    return (df["margin"] > 0).astype(float)


def _safe_corr(a, b) -> float:
    """Correlation that returns NaN instead of warning on tiny/constant samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


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
        return {
            "model": label,
            "n": 0,
            **{
                key: float("nan")
                for key in [
                    "win_acc",
                    "margin_mae",
                    "margin_rmse",
                    "total_mae",
                    "total_rmse",
                    "home_score_mae",
                    "away_score_mae",
                    "score_mae",
                    "margin_corr",
                    "total_corr",
                    "ats_rate",
                    "ats_n",
                    "ats_w",
                    "ats_l",
                    "ats_p",
                    "ats_roi",
                    "ou_rate",
                    "ou_n",
                    "ou_w",
                    "ou_l",
                    "ou_p",
                    "ou_roi",
                    "brier",
                    "logloss",
                ]
            },
        }

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
    ats_w = ats_l = ats_p = 0
    ats_roi = np.nan
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
            ats_p = int(push.sum())
            ats_w = int(ats_win.sum())
            ats_n = int((~push).sum())
            ats_l = ats_n - ats_w
            if ats_n:
                ats_rate = ats_w / ats_n
                ats_roi = _roi(ats_w, ats_l)

    # Over/under: model total vs closing total.
    ou_rate = np.nan
    ou_n = 0
    ou_w = ou_l = ou_p = 0
    ou_roi = np.nan
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
            ou_p = int(push.sum())
            ou_w = int(ou_win.sum())
            ou_n = int((~push).sum())
            ou_l = ou_n - ou_w
            if ou_n:
                ou_rate = ou_w / ou_n
                ou_roi = _roi(ou_w, ou_l)

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
        "margin_corr": _safe_corr(d["margin"], pm),
        "total_corr": _safe_corr(d["total"], pt),
        "ats_rate": ats_rate,
        "ats_n": ats_n,
        "ats_w": ats_w,
        "ats_l": ats_l,
        "ats_p": ats_p,
        "ats_roi": ats_roi,
        "ou_rate": ou_rate,
        "ou_n": ou_n,
        "ou_w": ou_w,
        "ou_l": ou_l,
        "ou_p": ou_p,
        "ou_roi": ou_roi,
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


def calibration(
    df: pd.DataFrame,
    prob_col: str = "win_prob",
    bins: int = 8,
    label: str | None = None,
) -> pd.DataFrame:
    """Calibration bins with Wilson intervals on the observed win rate."""
    d = df[df["home_win"].notna()].copy()
    d = d[d[prob_col].notna()]
    if d.empty:
        return pd.DataFrame(columns=["bin", "n", "predicted", "actual", "lo", "hi"])
    edges = np.linspace(0, 1, bins + 1)
    d["bin"] = pd.cut(d[prob_col], edges, include_lowest=True)
    rows = []
    for label_bin, g in d.groupby("bin", observed=True):
        wins = int(g["home_win"].sum())
        n = len(g)
        lo, hi = wilson_interval(wins, n)
        rows.append(
            {
                "bin": str(label_bin),
                "n": n,
                "predicted": float(g[prob_col].mean()),
                "actual": wins / n,
                "lo": lo,
                "hi": hi,
            }
        )
    out = pd.DataFrame(rows)
    if label is not None:
        out.insert(0, "model", label)
    return out
