"""Evaluation utilities with paired uncertainty.

Every comparison here resamples **games**, not models.  Two predictions on the
same game share the same outcome, so an unpaired test overstates precision.
The harness reports a point estimate, a 95% bootstrap interval, and (where
appropriate) a paired test.  If an interval spans zero, the honest conclusion
is "no detectable difference", not "small edge".

Implemented with numpy/pandas only; no scipy dependency.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

DEFAULT_BOOT = 10_000
DEFAULT_SEED = 20261114


def _finite(*arrays) -> tuple[np.ndarray, ...]:
    cols = [np.asarray(a, dtype=float) for a in arrays]
    mask = np.ones(len(cols[0]), dtype=bool)
    for c in cols:
        mask &= np.isfinite(c)
    return tuple(c[mask] for c in cols)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(
    values,
    statistic=np.mean,
    n_boot: int = DEFAULT_BOOT,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Percentile bootstrap interval for one statistic."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(x.size)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    stats = statistic(x[idx], axis=1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": float(statistic(x)),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(x.size),
    }


def paired_bootstrap_diff(
    a,
    b,
    statistic=np.mean,
    n_boot: int = DEFAULT_BOOT,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Bootstrap ``statistic(a) - statistic(b)`` with paired resampling.

    A negative difference means ``a`` is better when the statistic is an error
    (MAE, Brier, log loss) and worse when it is accuracy.  Read the sign in the
    context of the metric.
    """
    aa, bb = _finite(a, b)
    n = len(aa)
    if n < 2:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "n": n}
    point = float(statistic(aa) - statistic(bb))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = statistic(aa[idx], axis=1) - statistic(bb[idx], axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p = 2 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return {
        "diff": point,
        "lo": float(lo),
        "hi": float(hi),
        "p": float(min(1.0, p)),
        "n": n,
    }


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test for two classifiers on the same games."""
    y, a, b = _finite(y_true, pred_a, pred_b)
    a_correct = a == y
    b_correct = b == y
    a_only = int((a_correct & ~b_correct).sum())
    b_only = int((b_correct & ~a_correct).sum())
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "n": 0, "p": 1.0}
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return {
        "a_only": a_only,
        "b_only": b_only,
        "n": n,
        "p": float(min(1.0, 2 * tail)),
    }


def diebold_mariano(loss_a, loss_b, h: int = 1) -> dict:
    """Diebold-Mariano test for equal predictive accuracy.

    Uses the Harvey-Leybourne-Newbold small-sample correction and a normal
    reference distribution (accurate for the sample sizes here).
    """
    a, b = _finite(loss_a, loss_b)
    n = len(a)
    if n < 3:
        return {"stat": float("nan"), "p": float("nan"), "mean_diff": float("nan"), "n": n}
    d = a - b
    mean_d = float(d.mean())
    gamma0 = float(np.var(d, ddof=0))
    var = gamma0
    for k in range(1, h):
        gamma = float(np.cov(d[k:], d[:-k], ddof=0)[0, 1])
        var += 2 * (1 - k / h) * gamma
    if var <= 0:
        return {"stat": float("nan"), "p": float("nan"), "mean_diff": mean_d, "n": n}
    stat = mean_d / math.sqrt(var / n)
    correction = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat *= correction
    p = math.erfc(abs(stat) / math.sqrt(2))
    return {
        "stat": float(stat),
        "p": float(p),
        "mean_diff": mean_d,
        "n": n,
    }


def brier(y_true, prob) -> float:
    y, p = _finite(y_true, prob)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def log_loss(y_true, prob) -> float:
    y, p = _finite(y_true, prob)
    if not len(y):
        return float("nan")
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_table(
    df: pd.DataFrame,
    prob_col: str,
    y_col: str = "home_win",
    n_bins: int = 8,
) -> pd.DataFrame:
    """Calibration bins with Wilson intervals on the observed rate."""
    d = df[[prob_col, y_col]].dropna()
    if d.empty:
        return pd.DataFrame(columns=["bin", "n", "predicted", "actual", "lo", "hi"])
    edges = np.linspace(0, 1, n_bins + 1)
    d = d.assign(bin=pd.cut(d[prob_col], edges, include_lowest=True))
    rows = []
    for label, g in d.groupby("bin", observed=True):
        k = float(g[y_col].sum())
        n = len(g)
        lo, hi = wilson_interval(int(k), n)
        rows.append(
            {
                "bin": str(label),
                "n": n,
                "predicted": float(g[prob_col].mean()),
                "actual": k / n,
                "lo": lo,
                "hi": hi,
            }
        )
    return pd.DataFrame(rows)


def _bet_profit(win: np.ndarray, odds: float) -> np.ndarray:
    """Profit in units for a 1-unit bet at American ``odds``."""
    if odds < 0:
        payout = 100 / abs(odds)
    else:
        payout = odds / 100
    return np.where(win, payout, -1.0)


def ats_summary(
    df: pd.DataFrame,
    pick_col: str = "ats_pick",
    home_col: str = "home_team",
    line_col: str = "market_margin",
    margin_col: str = "margin",
    odds: float = -110,
) -> dict:
    """ATS record, win rate, and ROI at the given price.

    Pushes are excluded from the rate and the ROI, as they are in the market.
    """
    d = df[[pick_col, home_col, line_col, margin_col]].dropna()
    if d.empty:
        return {"n": 0, "wins": 0, "losses": 0, "pushes": 0}
    cover = d[margin_col].astype(float) - d[line_col].astype(float)
    pick_home = d[pick_col] == d[home_col]
    push = cover == 0
    win = np.where(pick_home, cover > 0, cover < 0)
    graded = ~push.to_numpy()
    win = win[graded]
    n = int(graded.sum())
    wins = int(win.sum())
    losses = n - wins
    profit = _bet_profit(win, odds)
    roi = bootstrap_ci(profit, n_boot=DEFAULT_BOOT)
    rate_lo, rate_hi = wilson_interval(wins, n)
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "pushes": int(push.sum()),
        "win_rate": wins / n if n else float("nan"),
        "win_rate_lo": rate_lo,
        "win_rate_hi": rate_hi,
        "units": float(profit.sum()),
        "roi": roi["estimate"],
        "roi_lo": roi["lo"],
        "roi_hi": roi["hi"],
        "odds": odds,
    }


def over_summary(
    df: pd.DataFrame,
    pick_col: str = "ou_diagnostic",
    line_col: str = "market_total",
    total_col: str = "total",
    odds: float = -110,
) -> dict:
    """Over/under record and ROI.  Kept as a diagnostic, not a recommendation."""
    if pick_col not in df.columns:
        return {"n": 0}
    d = df[[pick_col, line_col, total_col]].dropna()
    if d.empty:
        return {"n": 0}
    diff = d[total_col].astype(float) - d[line_col].astype(float)
    push = diff == 0
    win = np.where(d[pick_col] == "Over", diff > 0, diff < 0)
    graded = ~push.to_numpy()
    win = win[graded]
    n = int(graded.sum())
    wins = int(win.sum())
    profit = _bet_profit(win, odds)
    roi = bootstrap_ci(profit)
    lo, hi = wilson_interval(wins, n)
    return {
        "n": n,
        "wins": wins,
        "losses": n - wins,
        "pushes": int(push.sum()),
        "win_rate": wins / n if n else float("nan"),
        "win_rate_lo": lo,
        "win_rate_hi": hi,
        "units": float(profit.sum()),
        "roi": roi["estimate"],
        "roi_lo": roi["lo"],
        "roi_hi": roi["hi"],
        "odds": odds,
    }


def coverage_summary(df: pd.DataFrame, prob_col: str = "cover_prob") -> dict:
    """Brier score and reliability of cover probabilities."""
    if prob_col not in df.columns or "market_margin" not in df.columns:
        return {"n": 0}
    d = df.dropna(subset=[prob_col, "market_margin", "margin"]).copy()
    if d.empty:
        return {"n": 0}
    diff = d["margin"] - d["market_margin"]
    d = d[diff != 0]
    y = ((d["margin"] - d["market_margin"]) > 0).astype(float)
    p = d[prob_col].astype(float)
    return {
        "n": len(d),
        "brier": brier(y, p),
        "mean_prob": float(p.mean()),
        "actual_rate": float(y.mean()),
    }
