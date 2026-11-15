"""Joint residual bootstrap for coherent score distributions.

The model predicts a margin and a total.  Turning those two point estimates
into a score distribution needs the *joint* error structure: margin errors and
total errors are correlated (a shootout moves both), and a team's score is

    home = (total + margin) / 2
    away = (total - margin) / 2

so independent intervals for each team can produce impossible combinations.
This module resamples observed ``(margin_error, total_error)`` pairs from
out-of-fold predictions, adds them to the point forecast, and reports coherent
win / cover / push / over probabilities plus per-team score quantiles.

Everything is empirical: no normal assumption, no fixed +/- constants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_DRAWS = 4000
DEFAULT_SEED = 20261114
_CHUNK = 256


@dataclass
class SimulationResult:
    """Per-game simulation summaries.  All arrays are aligned to the input."""

    home_win_prob: np.ndarray
    cover_prob: np.ndarray
    push_prob: np.ndarray
    over_prob: np.ndarray
    total_push_prob: np.ndarray
    margin_sd: np.ndarray
    total_sd: np.ndarray
    home_low: np.ndarray
    home_med: np.ndarray
    home_high: np.ndarray
    away_low: np.ndarray
    away_med: np.ndarray
    away_high: np.ndarray


def _draw(pool: np.ndarray, n_games: int, n_draws: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n_draws`` residual pairs for each of ``n_games`` games."""
    idx = rng.integers(0, len(pool), size=(n_games, n_draws))
    return pool[idx, 0], pool[idx, 1]


def simulate_scores(
    pred_margin: np.ndarray,
    pred_total: np.ndarray,
    resid_pool: np.ndarray,
    fallback_pool: np.ndarray | None = None,
    use_fallback: np.ndarray | None = None,
    market_margin: np.ndarray | None = None,
    market_total: np.ndarray | None = None,
    n_draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
) -> SimulationResult:
    """Bootstrap scores from out-of-fold ``(margin, total)`` residual pairs.

    Parameters
    ----------
    pred_margin, pred_total
        Point forecasts, in points (home perspective).
    resid_pool
        ``(n, 2)`` array of out-of-fold ``actual - prediction`` pairs for the
        market-anchored model.
    fallback_pool
        Optional pool for games with no market line (the pure model's errors).
    use_fallback
        Boolean mask selecting games that should draw from ``fallback_pool``.
    market_margin, market_total
        Closing line, used for cover/over probabilities.  Missing values yield
        NaN probabilities rather than a fake 50/50.
    """
    pm = np.asarray(pred_margin, dtype=float)
    pt = np.asarray(pred_total, dtype=float)
    n = len(pm)
    if len(pt) != n:
        raise ValueError("pred_margin and pred_total must have the same length")
    if resid_pool is None or len(resid_pool) < 30:
        raise ValueError("residual pool needs at least 30 observations")

    mm = None if market_margin is None else np.asarray(market_margin, dtype=float)
    mt = None if market_total is None else np.asarray(market_total, dtype=float)
    fb = None if use_fallback is None else np.asarray(use_fallback, dtype=bool)

    keys = [
        "home_win_prob",
        "cover_prob",
        "push_prob",
        "over_prob",
        "total_push_prob",
        "margin_sd",
        "total_sd",
        "home_low",
        "home_med",
        "home_high",
        "away_low",
        "away_med",
        "away_high",
    ]
    out = {k: np.full(n, np.nan) for k in keys}
    rng = np.random.default_rng(seed)

    for start in range(0, n, _CHUNK):
        sl = slice(start, min(start + _CHUNK, n))
        m0 = pm[sl]
        t0 = pt[sl]
        size = len(m0)

        draw_m, draw_t = _draw(resid_pool, size, n_draws, rng)
        if fallback_pool is not None and fb is not None:
            local_fb = fb[sl]
            if local_fb.any():
                fbm, fbt = _draw(fallback_pool, int(local_fb.sum()), n_draws, rng)
                draw_m[local_fb] = fbm
                draw_t[local_fb] = fbt

        sim_margin = m0[:, None] + draw_m
        sim_total = t0[:, None] + draw_t
        home = (sim_total + sim_margin) / 2
        away = (sim_total - sim_margin) / 2

        # Probabilities use the continuous draws.  Rounding each score first
        # would create a ~4% tie rate purely from discretisation (real NFL ties
        # are ~0.4%), and would bias cover/push probabilities on integer lines.
        # The reported score quantiles are rounded so the card shows integers.
        out["home_win_prob"][sl] = (home > away).mean(axis=1)
        out["margin_sd"][sl] = sim_margin.std(axis=1)
        out["total_sd"][sl] = sim_total.std(axis=1)

        q_home = np.clip(
            np.quantile(np.rint(home), [0.10, 0.50, 0.90], axis=1), 0, None
        )
        q_away = np.clip(
            np.quantile(np.rint(away), [0.10, 0.50, 0.90], axis=1), 0, None
        )
        out["home_low"][sl], out["home_med"][sl], out["home_high"][sl] = q_home
        out["away_low"][sl], out["away_med"][sl], out["away_high"][sl] = q_away

        if mm is not None:
            line = mm[sl]
            ok = np.isfinite(line)
            if ok.any():
                diff = sim_margin[ok] - line[ok, None]
                cover = np.full(size, np.nan)
                push = np.full(size, np.nan)
                cover[ok] = (diff > 0).mean(axis=1)
                push[ok] = (diff == 0).mean(axis=1)
                out["cover_prob"][sl] = cover
                out["push_prob"][sl] = push

        if mt is not None:
            line = mt[sl]
            ok = np.isfinite(line)
            if ok.any():
                diff = sim_total[ok] - line[ok, None]
                over = np.full(size, np.nan)
                push = np.full(size, np.nan)
                over[ok] = (diff > 0).mean(axis=1)
                push[ok] = (diff == 0).mean(axis=1)
                out["over_prob"][sl] = over
                out["total_push_prob"][sl] = push

    return SimulationResult(**out)
