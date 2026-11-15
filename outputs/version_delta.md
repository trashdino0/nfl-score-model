# Version delta — current model vs earlier revision

*Base `d39c571` · current `main` · 2026 Weeks 1–9 · 136 games aligned by `game_id`.*

Training windows are identical on both sides (`gameday < slate date`), so
this isolates model changes from the backtest-alignment fix. For error
metrics a negative Δ means the current model is better.

**Summary:** none of the 6 tested metrics moved beyond noise.

| Metric | Before | After | Δ | 95% CI | p | Verdict |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Margin MAE | 9.489 | 9.489 | +0.000 | [+0.000, +0.000] | 1.000 | unchanged |
| Total MAE | 10.654 | 10.596 | -0.058 | [-0.129, +0.012] | 0.109 | no detectable change |
| Score MAE | 7.260 | 7.253 | -0.007 | [-0.032, +0.018] | 0.565 | no detectable change |
| Winner accuracy | 70.4% | 70.4% | +0.00 pp | — | 1.000 | no detectable change |
| Brier (win prob) | 21.5% | 21.5% | +0.01 pp | [-0.001, +0.001] | 0.699 | no detectable change |
| Log loss (win prob) | 62.2% | 62.2% | +0.04 pp | [-0.001, +0.002] | 0.612 | no detectable change |
| 80% interval coverage (home) | 87.5% | 86.8% | -0.74 pp | — | — | closer to 80% |

## Betting diagnostics

- ATS win rate: 52.3% → 52.3% (n = 130/130). pushes after/before: 6/6.
- ATS ROI (before): -0.1% [-16.3%, 16.0%] over 130 bets.
- ATS ROI (after): -0.1% [-16.3%, 16.0%] over 130 bets.
- Over/under win rate: 40.4% → 48.5% (n = 136/136). after derived from P(over) > 0.5; the model publishes no lean.

## What a delta does and does not prove

- **It does** isolate the model change: same games, same training windows,
  same evaluation code on both sides.
- **It does not** prove the change generalises. 136 games is one season's
  sample; a 0.05-point MAE move is smaller than the seed-to-seed variation
  of the bootstrap. Re-run on 2021–2025 before believing a small win.
- **It does not** capture harness improvements. The Tier-1 release mostly
  changed *measurement* (honest fallback labels, paired intervals,
  bootstrap score ranges), which by construction cannot appear as a
  prediction delta.
