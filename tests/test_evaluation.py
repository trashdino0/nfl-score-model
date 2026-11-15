"""Tests for the paired evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nflscore.evaluation import (
    ats_summary,
    diebold_mariano,
    mcnemar,
    paired_bootstrap_diff,
    wilson_interval,
)


def test_wilson_interval_known_value():
    lo, hi = wilson_interval(5, 10)
    assert abs(lo - 0.2366) < 0.01
    assert abs(hi - 0.7634) < 0.01


def test_wilson_interval_edges():
    assert wilson_interval(0, 10)[0] == 0.0
    assert abs(wilson_interval(10, 10)[1] - 1.0) < 1e-9
    lo, hi = wilson_interval(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_mcnemar_exact():
    y = np.array([1.0, 1.0, 1.0, 1.0])
    a = np.array([1.0, 1.0, 1.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 0.0])
    result = mcnemar(y, a, b)
    assert result["a_only"] == 3
    assert result["b_only"] == 0
    assert abs(result["p"] - 0.25) < 1e-9


def test_paired_bootstrap_identical_is_zero():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    result = paired_bootstrap_diff(a, a, np.mean, n_boot=500)
    assert result["diff"] == 0
    assert result["lo"] == 0
    assert result["hi"] == 0
    assert result["p"] == 1.0


def test_paired_bootstrap_detects_large_shift():
    rng = np.random.default_rng(0)
    b = rng.normal(0, 1, 400)
    a = b + 1.0
    result = paired_bootstrap_diff(a, b, np.mean, n_boot=2000)
    assert result["diff"] > 0.9
    assert result["lo"] > 0.8
    assert result["p"] < 0.01


def test_diebold_mariano_flags_shift():
    rng = np.random.default_rng(1)
    b = rng.normal(1.0, 1.0, 300)
    a = b - 0.5 + rng.normal(0, 0.1, 300)
    result = diebold_mariano(a, b)
    assert result["mean_diff"] < -0.4
    assert result["p"] < 0.01


def test_ats_summary_counts_and_roi():
    df = pd.DataFrame(
        {
            "ats_pick": ["H", "H", "A", "A", "H"],
            "home_team": ["H", "H", "H", "H", "H"],
            "market_margin": [3.0, 3.0, -2.0, -2.0, 0.0],
            "margin": [7.0, 3.0, 0.0, -10.0, 0.0],
        }
    )
    result = ats_summary(df)
    # Game 1 home covers, game 2 push, game 3 away loses (home covers),
    # game 4 away covers, game 5 push.
    assert result["n"] == 3
    assert result["wins"] == 2
    assert result["losses"] == 1
    assert result["pushes"] == 2
    assert abs(result["win_rate"] - 2 / 3) < 1e-9
    assert abs(result["roi"] - (2 * 100 / 110 - 1) / 3) < 1e-9
