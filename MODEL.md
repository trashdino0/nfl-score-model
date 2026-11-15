# The machine learning, explained

*Written 2026-11-14 against `main` (Tier 1). Numbers are reproducible with
`scripts/evaluate.py` and `scripts/version_delta.py`.*

This document explains what the model actually is, why it is built that way, and
what the diagnostics say about whether any of it works. It is deliberately
blunt: the interesting result is a negative one, and understanding *why* it is
negative is more useful than another feature.

---

## 1. The learning problem

Two regression targets per game, both from the home team's perspective:

```
margin = home_score - away_score          (points)
total  = home_score + away_score          (points)
```

Scores are then recovered algebraically:

```
home = (total + margin) / 2
away = (total - margin) / 2
```

Why not predict each team's score directly? Because margin and total are
**approximately independent** quantities with different drivers. Margin is
driven by relative strength; total is driven by pace, play volume, and
environment. Modelling `home` and `away` separately forces the model to learn
"home offence vs away defence" and "away offence vs home defence" twice, with
double the noise and no guarantee that the implied margin is coherent. The
two-target decomposition is a variance-reduction trick, not a stylistic choice.

**Sample:** 2,264 completed games, 2018–2026 Week 9. 2018 is the floor because
nflverse coverage/pressure data starts there. That is small. A season is ~272
games, so the model has seen about eight and a half seasons of *game-level*
outcomes — roughly the sample size of a single moderately sized Kaggle tabular
problem, but with far lower signal-to-noise.

**Split:** strictly walk-forward. For a slate on date `d`, training uses every
game with `gameday < d`. No random train/test split, ever. A random split on
time-series sports data leaks future information through team strength,
schedule, and rule changes, and it inflates every metric.

---

## 2. What the model is allowed to see

158 candidate features, built from play-by-play, participation, injuries, depth
charts, and the closing line. Four families:

### 2.1 Team state (rolling, no leakage)

Each team-week gets a rolling average of ~80 metrics over the last 4 games, and
a subset over the last 8. The critical detail is **`shift(1)`**:

```python
game_state[f"{m}_before4"] = g[m].transform(
    lambda s: s.shift(1).rolling(4, min_periods=1).mean()
)
```

The state used for week *w* excludes week *w*. Without that shift, the model
would see the game it is predicting.

Two further subtleties that matter more than they look:

- **Windows are over games, not weeks.** A bye must not shorten the window or
  pull in an older game. The grid is team × week 1–18, and byes forward-fill the
  state *after* the last played game, rather than interpolating.
- **Prior-season means are a separate feature**, computed from the previous
  season's full average. This gives Week 1 a prior instead of a null. It is
  legal because the prior season is complete before the current one starts.

### 2.2 Matchup features (the "who beats whom" layer)

Rather than 158 raw columns, the important ones are **differences of
differences**, which is what actually encodes a matchup:

```
home_adv = home_offense_roll4 - away_defense_roll4
away_adv = away_offense_roll4 - home_defense_roll4
net      = home_adv - away_adv
```

`net` is the quantity that should predict margin: how much better the home
offence matches up against the away defence than the reverse. There are 26
pairs, including EPA/pass/rush, success rate, explosive rate, red zone, pace,
QB-vs-pass-defense, WR1-vs-coverage, RB1-vs-run-defense, and
pass-rush-vs-protection.

This is the layer where your scheme/coaching idea would plug in, and it is the
right place for it: as another `offense_metric - defense_metric` pair with a
defensible prior.

### 2.3 Availability (QB and injuries)

Injuries are not just counts. Each injured player is weighted by their **snap
share**, so losing a 90%-snap left tackle matters more than losing a
special-teamer:

```
lost = 1 - status_score
lost_off = lost * off_snap_pct_roll4
lost_def = lost * def_snap_pct_roll4
```

`status_score` maps the official report to an availability prior:
Out = 0.0, Doubtful = 0.2, Questionable = 0.7, and practice participation
downgrades it further (DNP = 0.35, Limited = 0.8). This is a hand-built prior,
not a learned parameter — a defensible simplification given the sample size.

QB gets special treatment because the position dominates: separate features for
the *projected starter* (from depth chart plus injury status), QB change flags,
and the starter's own rolling EPA/CPOE.

### 2.4 Market context

The closing spread, total, and no-vig win probability. These are not "cheating"
— they are the thing being forecast *around*. The model's job is to say
something the line does not already say.

---

## 3. The estimator: ridge regression, deliberately

Every model is the same three-step pipeline:

```python
Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
    ("ridge",  Ridge(alpha=100.0)),
])
```

**Why ridge and not gradient boosting?** Four reasons, in order of importance:

1. **The signal is approximately linear and tiny.** With a true R² around 0.17,
   a high-capacity model mostly fits noise. The gradient-boosting experiments in
   `scripts/experiment_*.py` did not beat ridge out of sample.
2. **Regularisation is the whole game.** 158 correlated features on 2,264 rows
   is a p/n ratio where OLS is unstable. Ridge's L2 penalty shrinks correlated
   coefficients together instead of picking one arbitrarily — which is exactly
   the behaviour you want for "off EPA" vs "pass EPA" vs "points per drive",
   which are ~0.9 correlated.
3. **Coefficients are interpretable.** The card shows "key drivers". A ridge
   coefficient times a standardised feature is a readable contribution; a
   tree ensemble's SHAP values on 2,264 rows are not.
4. **Stability across seasons.** Ridge refits are nearly identical week to week.
   That matters when the same model is used to publish a card every Sunday.

`alpha=100` is a strong penalty (the feature matrix is standardised, so 100 is
large). It was chosen by the grid in `outputs/tune_validation.csv`. Note the
caveat in §7: that grid selected and evaluated on the same window, so it is
mildly optimistic.

Median imputation rather than mean: injury and depth-chart columns are
right-skewed and have real missingness, and the median is a more robust neutral
value. Crucially, imputation happens **inside the pipeline**, so it is fit on
training folds only — no test-set statistics leak into preprocessing.

---

## 4. The architectural decision that matters: anchor on the line

This is the single most important design choice, and the reason the model looks
underwhelming by design.

### 4.1 Why not just predict the score?

Because the closing line is a better predictor than anything derivable from
public play-by-play. Out of sample on 2026 Weeks 1–9 (136 games):

| Target | Market R² | Pure stats R² |
| :--- | ---: | ---: |
| Margin | **0.172** | 0.086 |
| Total | **0.135** | −0.039 |

Public stats recover about **half** the market's explanatory power on margin,
and on totals the pure model is *worse than predicting the league average*.
That is not a bug in the feature engineering; it is what an efficient betting
market looks like. The line aggregates public stats, injury news, weather,
travel, sharp money, and information that never appears in a box score.

So the architecture is:

```
prediction = closing_line + shrink * f(features)
```

where `f` is a ridge model trained to predict `actual - closing_line`, and
`shrink` is a scalar in [0, 1].

### 4.2 Why the residual formulation helps

Two mechanical benefits, independent of whether `f` has signal:

- **Variance reduction.** Even a useless `f` with shrink 0 collapses to the
  market, which is a strong baseline. A pure model has no such floor.
- **Stationarity.** `actual - line` has mean ≈ 0 and is far more stable across
  seasons than raw scoring, which drifts with rule changes and offensive era.
  A model trained on 2018 residuals is more transferable than one trained on
  2018 raw points.

### 4.3 The uncomfortable diagnostic

Out-of-fold, with 5-fold CV inside the training window:

| Residual target | OOF R² | Residual correlation |
| :--- | ---: | ---: |
| Margin − line | **−0.033** | 0.025 |
| Total − line | **−0.031** | 0.061 |

A negative R² means the model's 128-feature prediction of the residual is
**worse than predicting zero**. In other words: after the closing line, the
features add nothing that survives out of sample. The correlation of 0.03 is
noise at n = 2,264 (standard error ≈ 0.02).

This is why `total_shrink = 0` in Tier 1. The total model was adding 10% of a
negative-R² signal to the closing total, which is strictly harmful in
expectation.

**A subtlety worth understanding.** The fitted optimal scalar shrink is *not*
zero: regressing the actual residual on the predicted residual gives β ≈ 0.12
(margin) and 0.25 (total). Both are positive. This is not a contradiction. R²
penalises the full 128-dimensional prediction, which overfits; the *univariate
projection* onto that prediction can still have a slightly positive slope. The
practical size of the effect is what settles it:

| Shrink | Margin MAE | Total MAE |
| ---: | ---: | ---: |
| 0.00 | 9.8200 | 10.3966 |
| 0.10 | 9.8166 | 10.3871 |
| 0.20 | 9.8172 | 10.3809 |
| fitted | 9.8163 (β=0.12) | 10.3807 (β=0.25) |

The entire range from 0 to the fitted optimum moves MAE by **0.004 points**
(margin) and **0.016 points** (total). That is 0.04% and 0.15% of the error. It
is far below the sampling noise of a 136-game backtest, whose 95% interval on
the MAE difference is roughly ±0.12.

The honest conclusion: the shrink parameter is not identifiable from this
sample. Set it from theory (0 when the signal is unproven), not from a marginal
MAE win that will not replicate.

---

## 5. Where cross-validation is actually used

The OOF predictions are not just for reporting. They drive four separate things,
and using in-sample versions of any of them would make the model overconfident:

1. **Error scale (`margin_sigma_`).** The normal-approximation win probability
   needs a standard deviation. In-sample residuals give ~11.9; OOF gives 12.73.
   Using the in-sample value would systematically overstate confidence.
2. **Residual pool.** The bootstrap resamples OOF `(margin_error, total_error)`
   pairs. In-sample errors are too small, so score intervals would be too
   narrow.
3. **Blend weight.** The market/model log-odds weight (0.43) is chosen by
   minimising OOF log loss. A weight fitted in-sample would over-weight the
   model.
4. **Calibration.** The reliability bins that judge whether the probabilities
   are honest are computed on OOF predictions.

`KFold(shuffle=True)` inside the training window is acceptable here because the
unit is a game and the window is fixed — the shuffle does not cross the
walk-forward boundary, so no future game enters any training fold for a past
prediction. (If folds were constructed across time instead, adjacent weeks would
share team state and leak.)

---

## 6. From two numbers to a distribution

A point forecast is not a product. The card needs P(win), P(cover), P(push),
P(over), and a score range. Three layers:

### 6.1 Win probability

Normal approximation on the predicted margin:

```
P(home win) = Φ(pred_margin / sigma)
```

with `sigma = 12.73` (OOF). This ignores ties (real rate ~0.4%) and assumes
margin errors are Gaussian-ish. The empirical residual distribution is close
enough — slightly leptokurtic, which makes the model a touch overconfident in
the tails.

Then a **log-odds blend** with the market's no-vig probability:

```
logit(p) = (1 - w) * logit(p_market) + w * logit(p_model),   w = 0.43
```

`w` is fitted by minimising OOF log loss, bounded to [0, 1]. Log-odds rather
than probability space because probabilities are bounded and blending them
linearly distorts the tails.

**Honest caveat:** because `pred_margin` is the line plus 20% of a near-zero
residual, `p_model` is almost identical to `p_market` (correlation 0.9996). The
blend is therefore nearly a no-op — which is why Brier barely moves. The fitted
weight is real, but the input it weights is not independent.

### 6.2 The joint residual bootstrap

For each game, draw `n = 4,000` residual pairs from the OOF pool:

```
sim_margin = pred_margin + e_margin
sim_total  = pred_total  + e_total
home = (sim_total + sim_margin) / 2
away = (sim_total - sim_margin) / 2
```

Why joint rather than independent intervals per team? Because the score
arithmetic links them: `home` and `away` are functions of the same two draws.
Independent intervals can produce a "range" where the implied margin contradicts
the projected margin. Sampling the pair preserves the arithmetic.

**An honest correction to the original rationale:** margin and total errors turn
out to be nearly uncorrelated in this data (r ≈ 0.007), despite the intuition
that a shootout moves both. So the practical benefit is *coherence and
distributional shape*, not correlation modelling. The pool is also centred
before use, so the bootstrap adds noise around the point forecast rather than
silently re-adding a bias.

Probabilities are computed on **continuous** draws and only the displayed
quantiles are rounded. Rounding first manufactured a ~4% tie rate (real: ~0.4%)
and distorted push probabilities on integer lines. That was caught by a unit
test, which is the argument for having unit tests on simulation code.

### 6.3 What the intervals are worth

| | Nominal | Actual coverage (2026 W1–9) |
| :--- | ---: | ---: |
| Pre-Tier-1 fixed ±offsets | 80% | 87.5% |
| Tier-1 bootstrap quantiles | 80% | 86.8% |

Both are **too wide**, and the bootstrap only improved this by 0.7 pp. The
reason is homoskedasticity: one residual pool is applied to every game, so a
projected 45-point game and a projected 55-point game get the same width, and
the pool is dominated by high-variance games. A conditional (quantile regression
or per-game variance) approach is the obvious next step, and it is in Tier 2.

---

## 7. Known weaknesses, ranked by how much they matter

1. **No identifiable signal beyond the line.** This dominates everything else.
   No amount of feature engineering has moved OOF residual R² above zero. Until
   something does, the model is a well-calibrated restatement of the market.
2. **Homoskedastic intervals.** Coverage is 87% against a nominal 80%. Game-level
   variance should depend on the matchup (pace, total, QB quality).
3. **Small sample, high-dimensional feature set.** 2,264 rows, 158 features. The
   alpha grid was selected and evaluated on the same 2021–2025 window
   (`outputs/tune_validation.csv`), so its headline number is optimistic. A
   nested selection would be honest but expensive.
4. **Hand-built priors.** Injury status scores (0.2/0.35/0.7/0.8) and the Elo
   K-factor/MOV multiplier are judgment calls, not fitted. They are defensible
   but unvalidated.
5. **Normal approximation.** Margin residuals are slightly fat-tailed, so tail
   probabilities are a little too confident.
6. **No opponent adjustment for injuries.** Losing a CB1 is weighted by snap
   share, not by how much the opposing WR1 exploits it.

---

## 8. What would actually change the answer

In rough order of expected value per unit of work:

1. **Conditional variance / quantile regression.** Improves intervals and
   cover probabilities without needing new signal. This is engineering, not
   discovery, and it will work.
2. **Player-prop targets instead of game outcomes.** The scheme/matchup features
   have measurable split-half reliability at the player level (WR vs man 0.48,
   zone 0.29) but not at the game level. Games are the wrong frequency to
   detect them.
3. **Situational subsets.** The residual signal may be concentrated: backup-QB
   games, coordinator changes, short weeks, extreme weather. A single global
   linear model averages these away. Interaction terms or a mixture model are
   the natural test.
4. **Nonlinearity, only after (1)–(3).** Gradient boosting on 2,264 rows with
   near-zero signal will fit noise. It becomes interesting only if a subset
   shows real signal.
5. **Better injury/availability data.** Free data cannot attribute a target to a
   specific defender. Exact "WR1 vs a bad-zone corner" needs PFF/SIS/NGS
   tracking, not nflverse participation.

---

## 9. One-paragraph summary

The model is a set of ridge regressions that predict **how wrong the closing
line will be**, using leakage-safe rolling team state, offense-vs-defense
matchup nets, snap-share-weighted injuries, QB availability, Elo, and weather.
Those residual predictions are shrunk toward zero and added back to the line,
then converted into a joint score distribution by resampling out-of-fold
`(margin, total)` errors. The evaluation is walk-forward, paired, and
bootstrap-inferenced. Out of sample, the residual models have **negative R²** —
they add nothing beyond the line — which is why the total shrink is zero and why
the published product is a calibrated distribution and an honest market
comparison, not a betting edge. That is a real result about the efficiency of
NFL closing lines, not a failure of the pipeline.
