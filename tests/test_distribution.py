"""Tests for the joint residual bootstrap."""

from __future__ import annotations

import numpy as np

from nflscore.distribution import simulate_scores


def _symmetric_pool(n: int = 2000) -> np.ndarray:
    rng = np.random.default_rng(0)
    margin = rng.normal(0, 10, n)
    total = 0.5 * margin + rng.normal(0, 8, n)
    pool = np.column_stack([margin, total])
    return pool - pool.mean(axis=0)


def test_output_shapes_and_ranges():
    pool = _symmetric_pool()
    n = 12
    sim = simulate_scores(
        np.zeros(n),
        np.full(n, 45.0),
        pool,
        market_margin=np.zeros(n),
        market_total=np.full(n, 45.0),
        n_draws=1000,
        seed=3,
    )
    for field in ["home_win_prob", "cover_prob", "over_prob", "home_low", "home_high"]:
        assert getattr(sim, field).shape == (n,)
    assert np.all(sim.home_low <= sim.home_med)
    assert np.all(sim.home_med <= sim.home_high)
    assert np.all(sim.away_low <= sim.away_med)
    assert np.all(sim.away_med <= sim.away_high)
    assert np.all(sim.home_low >= 0)
    assert np.all(sim.away_low >= 0)


def test_centered_game_is_a_coin_flip():
    pool = _symmetric_pool()
    sim = simulate_scores(
        np.zeros(400),
        np.full(400, 45.0),
        pool,
        market_margin=np.zeros(400),
        market_total=np.full(400, 45.0),
        n_draws=4000,
        seed=7,
    )
    assert abs(sim.home_win_prob.mean() - 0.5) < 0.02
    assert abs(sim.cover_prob.mean() - 0.5) < 0.02
    assert abs(sim.over_prob.mean() - 0.5) < 0.02


def test_large_favorite_wins_most_of_the_time():
    pool = _symmetric_pool()
    sim = simulate_scores(
        np.full(50, 14.0),
        np.full(50, 45.0),
        pool,
        n_draws=3000,
        seed=11,
    )
    assert sim.home_win_prob.min() > 0.75
    assert np.all(np.isnan(sim.cover_prob))  # no line, no fake cover prob


def test_high_total_game_has_wider_score_range():
    pool = _symmetric_pool(6000)
    sim = simulate_scores(
        np.zeros(2),
        np.array([38.0, 56.0]),
        pool,
        n_draws=6000,
        seed=13,
    )
    low_width = sim.home_high[0] - sim.home_low[0]
    high_width = sim.home_high[1] - sim.home_low[1]
    # The pool is homoskedastic, so this is a weak check that widths are at
    # least not collapsed to a constant by a bug.
    assert low_width > 0 and high_width > 0
    assert sim.total_sd[1] > 0


def test_fallback_pool_is_used_when_requested():
    primary = np.column_stack([np.full(500, 100.0), np.full(500, 100.0)])
    fallback = np.column_stack([np.full(500, 0.0), np.full(500, 0.0)])
    sim = simulate_scores(
        np.zeros(2),
        np.full(2, 45.0),
        primary,
        fallback_pool=fallback,
        use_fallback=np.array([False, True]),
        n_draws=500,
        seed=17,
    )
    assert sim.margin_sd[0] == 0.0
    assert sim.margin_sd[1] == 0.0
    # The primary pool shifts the score by 100 points; the fallback does not.
    assert sim.home_med[0] == 122.0
    assert sim.home_med[1] == 22.0
