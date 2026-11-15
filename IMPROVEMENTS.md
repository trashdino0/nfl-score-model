# NFL Score Model — Improvement Plan and Scheme/Coaching Assessment

*Date: 2026-11-14*
*Scope: `src/nflscore`, 2026 season, 2021–2025 validation + 2026 Weeks 1–8 backtest*
*Status: Tier 1 implemented (see §0). The pre-Tier-1 numbers in §1–§7 are kept
as the audit record that motivated each change; §0 supersedes them.*

---

## 0. Status update — Tier 1 implemented (2026-11-14)

Tier 1 is now in the repository. The original plan below is kept as the
rationale; this section records what changed and what the fresh numbers say.

**Code changes**

| Area | Change |
| :--- | :--- |
| Evaluation | New `src/nflscore/evaluation.py` and `scripts/evaluate.py`: paired bootstrap CIs, McNemar, Diebold–Mariano, Wilson intervals, ATS/over ROI at −110, calibration bins. Writes `outputs/evaluation.md` / `.json`. |
| Total model | `total_shrink = 0.0`. `ou_pick` and `total_edge` are removed from new outputs. The projected total is the market total; no O/U lean is published. |
| No-line fallback | `_safe_market` no longer fills a missing spread with the naive baseline. `predict` uses the pure model as the anchor and labels every row `line_source = market` or `pure_fallback`. |
| Distribution | New `src/nflscore/distribution.py`: joint residual bootstrap from out-of-fold `(margin, total)` errors. Score 10/50/90 quantiles, P(cover), P(push), P(over), margin/total SD. Replaces the fixed ±12.9/11.0 intervals. |
| Win probability | `margin_sigma_` now comes from out-of-fold error. The market/model log-odds weight is fitted on out-of-fold log loss (2026 fit: 0.43 model weight). The 50/50 blend is gone. |
| Backtest/live alignment | `weekly_2026` is now **game-day** walk-forward: Thursday's model trains through Wednesday, Sunday's through Saturday, Monday's through Sunday — exactly the live `gameday < first unplayed kickoff` rule. |
| Metrics | `evaluate` reports ATS/over W–L–P and ROI; `calibration` uses 8 bins with Wilson intervals; correlations are guarded against tiny samples. |
| Tests | `tests/test_evaluation.py`, `tests/test_distribution.py` (12 tests, all passing). |

**Fresh 2026 Weeks 1–9 walk-forward (136 games, 135 decisive)**

| Model | Winner acc. | Margin MAE | Total MAE | Score MAE | Brier |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Market closing line | 68.9% | 9.49 | 10.60 | 7.26 | 0.215 |
| Pure stats ridge | 67.4% | 9.73 | 12.07 | 7.78 | 0.213 |
| Hybrid | 70.4% | 9.49 | 10.60 | 7.25 | 0.215 |

**What the harness says**

- Hybrid − market: margin MAE **+0.004** [−0.113, +0.120]; score MAE **−0.008**
  [−0.050, +0.033]; Brier **−0.000** [−0.003, +0.002]. No detectable difference.
- Winner accuracy **+1.48 pp**; McNemar p = 0.50 (2 discordant games).
- Total MAE is identical by design (`total_shrink = 0`).
- ATS every lean: 68–62–6 = **52.3%** [43.8, 60.7], ROI **−0.1%** [−16.3, +16.0].
  ATS plays only (edge ≥ 1): 10–11–1 = 47.6%, ROI −9.1%. Not a strategy.
- Calibration is reasonable in the middle bins; the 0.25–0.375 bin is the worst
  (21 games, 33.7% predicted vs 14.3% actual, wide CI).

The original plan's conclusion stands: the residual model has no measurable
game-level edge. Tier 1 makes that visible instead of hiding it behind a fixed
interval and an O/U pick.

---

## 0. TL;DR

1. **The model is well-built, but the residual model has no measurable edge over the closing line.** The hybrid's 2026 winner accuracy is 67.5% vs 65.8% for the market — a two-game difference (McNemar p ≈ 0.5). Paired bootstrap differences in margin/total MAE are centered on zero. The predicted-residual correlation is ~0.03–0.06, and the predicted-residual deciles are non-monotonic.

2. **The biggest wins are not more team-efficiency features.** They are:
   - stop publishing the total/O-U opinion (it is worse than the market),
   - fix the no-line fallback (it currently applies a market-trained residual to a naive anchor),
   - replace point estimates and fixed intervals with a calibrated joint distribution,
   - evaluate everything with paired confidence intervals,
   - then chase information the closing line may not fully price (late injuries, weather, travel/rest, line movement).

3. **Your scheme/coaching idea is sound and the data exists — with caveats.**
   - `pbp_participation` (nflverse, 2016–2025) has `defense_man_zone_type`, `defense_coverage_type` (Cover 0/1/2/3/4/6/9, 2-Man), personnel, box count, pass rushers, pressure, route, and player IDs.
   - **No 2026 in-season participation file exists yet.** Man/zone data is effectively a *prior-season* signal for 2026. FTN charting (2022–2026, updated weekly) provides current-season motion/play-action/RPO/blitz/box data but **not** man/zone.
   - **There is a 2023–24 labeling regime break.** Zone rate was 62.8% (2019), 65.6% (2020), 69.7% (2021), 71.2% (2022), **58.5% (2023), 51.4% (2024)**, 69.0% (2025). The 2023–24 files were produced by a different pipeline (updated 2025-09-04 vs 2023-12-19 for earlier files). Cross-season pooling without handling this break is unsafe.
   - **Team scheme tendencies are stable** (split-half reliability: zone rate 0.86, blitz rate 0.82, box count 0.76, pass rushers 0.72, pressure 0.67), but **player-level man/zone splits are noisy** (one-season split-half reliability: QB vs zone 0.28, QB vs man 0.10, WR vs zone 0.29, WR vs man 0.48). Player-level features need multi-season hierarchical shrinkage.
   - **First incremental test:** using 2021, 2022, and 2025 Weeks 4+ (n=671), prior-game rolling scheme features, the scheme matchup model had in-sample R² = 0.0094 for margin and 0.0032 for total. Out-of-sample MAE deltas vs the market/hybrid were ~0.00 points and the coefficient sign was unstable across seasons. Only `zone_rate_diff` was borderline (t ≈ 1.8–1.9), which is not robust with eight features.
   - **Verdict:** the idea does not beat the closing line at the game level in its current form. Its best chances are **player props** (WR1/QB vs scheme), **coordinator-change spots**, and **situational scheme** (third down, red zone, two-minute), not overall game margin/total.

4. **Recommended sequence:** evaluation harness → total/no-line fixes → joint distribution + calibration → injury/travel/weather/neutral-script features → scheme/coaching pipeline → nonlinear/hierarchical models.

---

## 1. Current state and diagnostics

### 1.1 Headline metrics

**2026 Weeks 1–8 (121 games, 120 decisive):**

| Model | Winner acc. | Margin MAE | Total MAE | Score MAE |
| :--- | ---: | ---: | ---: | ---: |
| Market closing line | 65.8% (79/120) | 9.57 | 10.66 | 7.33 |
| Pure stats ridge | 64.2% (77/120) | 9.85 | 12.10 | 7.86 |
| Hybrid market+stats | 67.5% (81/120) | 9.58 | 10.71 | 7.33 |

**2021–2025 season-level validation (average):**

| Model | Winner acc. | Margin MAE | Total MAE | Score MAE |
| :--- | ---: | ---: | ---: | ---: |
| Market closing line | 66.6% | 9.75 | 10.31 | 7.24 |
| Pure stats ridge | 63.0% | 10.39 | 10.85 | 7.58 |
| Hybrid market+stats | 66.7% | 9.74 | 10.29 | 7.23 |

### 1.2 Paired diagnostics (computed from `outputs/backtest_2026_weeks1_8.csv`)

| Test | Result |
| :--- | :--- |
| Hybrid minus market, margin MAE | **+0.010** points, 95% CI **[−0.121, +0.139]** |
| Hybrid minus market, total MAE | **+0.050** points, 95% CI **[−0.026, +0.126]** |
| Hybrid minus market, score MAE | **−0.003** points, 95% CI **[−0.055, +0.049]** |
| Winner accuracy, discordant games | Hybrid wins 2, market wins 0; McNemar p ≈ 0.5 |
| ATS record | 59–56–6 = **51.8%**, 95% CI **[42.6%, 60.9%]** |
| O/U record | 51–70 = **42.1%**, 95% CI **[33.4%, 50.9%]** |
| Mean absolute model edge | **0.59 pts** (median 0.49); 18% of games > 1 pt |
| Brier, 2026 | market **0.2270**, hybrid **0.2260**, model-only **0.2255** |
| Brier, 2021–25 average | market **0.212**, hybrid **0.211** |
| Fitted market/model prob blend | weight 0.43 on 2021–24; 2025 Brier 0.2115 vs 0.2114 for fixed 50/50 — no improvement |

### 1.3 The residual model has almost no signal

Predicted residual = `(pred_margin − market_margin) / 0.20` (and the analogous total). The correlation with the actual residual:

| Sample | Margin residual corr. | Total residual corr. |
| :--- | ---: | ---: |
| 2021–2025 validation (n=1,359) | **0.036** | **0.061** |
| 2026 Weeks 1–8 (n=121) | **0.034** | **−0.040** |

The 2021–25 margin correlation's 95% CI includes zero. The total correlation is marginally positive in 2021–25 but negative in 2026. Predicted-residual deciles are non-monotonic; in 2026 the most positive margin-residual decile had an actual residual of **−3.9 points**.

### 1.4 The fixed shrink is not stable

Optimal shrink by season (regression of actual residual on predicted residual, without shrink):

| Season | Margin | Total |
| :--- | ---: | ---: |
| 2021 | 0.260 | 0.423 |
| 2022 | −0.107 | −0.079 |
| 2023 | 0.030 | 0.378 |
| 2024 | 0.102 | 0.024 |
| 2025 | 0.576 | 0.523 |
| 2026 (W1–8) | 0.109 | −0.124 |

The production values are 0.20 (margin) and 0.10 (total). They are reasonable central values but are presented as if estimated; they are not stable enough for that.

### 1.5 What this means

The model is essentially **the closing line plus a small nudge**. That is a legitimate design for a score/win-probability model, but it is not a betting edge. Improvements should target:
- the **output layer** (distributions, calibration, evaluation),
- **information the close does not have** (timing, weather, travel, line movement),
- **input quality** (garbage-time/opponent adjustment, sample-size shrinkage),
- and **honesty** (stop publishing total/O-U opinions that are worse than the market).

---

## 2. Tier 1 — Fix what is currently misleading (highest value, lowest effort)

### 2.1 Stop shipping the total model's opinion

**Change:** set `total_shrink = 0.0` in `ModelConfig` and use the market total as the score anchor. Remove `ou_pick` from the card/CSV or rename it `ou_diagnostic`. Do not present O/U leans as recommendations.

**Evidence:**
- 2026 total MAE: 10.71 hybrid vs 10.66 market.
- 2026 O/U: 42.1% (51–70), 95% CI [33.4%, 50.9%].
- 2021–25 total MAE: 10.29 hybrid vs 10.31 market (a hair better, not robust).
- Optimal total shrink is negative in 2026 (−0.124) and ranges −0.12 to +0.52 across seasons.

**If you want a real total model:** rebuild it with total-specific inputs (weather forecast, pace × efficiency, special teams, red-zone regression, referee/penalty tendencies, kicker quality), use quantile regression, and require it to beat the market total on at least two independent seasons before publishing leans.

### 2.2 Fix the no-line fallback

`_safe_market()` fills missing `market_margin`/`market_total` with `naive_margin`/`naive_total`, then `predict()` adds a residual trained against the **market** line to a **naive** anchor. That residual is not calibrated for a naive anchor. It also imputes `market_home_win_prob` from the naive margin (sigma 13.5) and blends it 50/50.

**Change:**
- If `market_margin` is missing, use `pure_margin`/`pure_total` (trained on actual outcomes), and derive `win_prob` from the pure model's distribution.
- Add `line_source = market | pure_fallback` to the output.
- Drop training rows with missing market lines, or train a separate naive-residual model.
- Do not call a naive-anchored prediction "market-anchored."

### 2.3 Make the shrink defensible

**Options:**
- Hierarchical/Bayesian partial pooling of the shrink slope across seasons.
- Pre-register a conservative value (e.g., 0.10–0.15 margin, 0 total) and report a sensitivity table at 0, 0.1, 0.2, 0.3.
- Fit an intercept/calibration term. The market has season-specific bias: the validation intercept is **+0.55 points** (2021–25) and the 2026 intercept is **−1.31 points**. A rolling bias term may capture some of it, but test it out-of-sample.
- Report results under `shrink = 0` as the honest baseline. If the model cannot beat it, say so.

### 2.4 Replace the normal win probability and fixed intervals with a joint residual bootstrap

Current:
- `P(home win) = Φ(pred_margin / σ)`, σ ≈ 12.74, then a 50/50 blend with the market probability.
- 80% score intervals are global constants (home −12.9/+11.0, away −11.9/+11.3), not conditioned on the game; they can produce negative lows.

**Change:** sample empirical `(margin_residual, total_residual)` pairs from out-of-fold errors, add them to the market line, and derive:
- score percentiles (10/50/90) for each team,
- P(home win), P(cover), P(push), P(over),
- coherent score-pair distributions,
- correlated score ranges (instead of independent ± constants).

Use an empirical or Student-t distribution for the tails. This preserves the margin/total correlation, handles heteroskedasticity (a 45-total game and a 55-total game do not have the same variance), and eliminates negative interval lows.

### 2.5 Fit the win-probability layer out-of-sample, or drop the blend

The 50/50 blend is arbitrary. Evidence:
- 2026 Brier: market 0.2270, hybrid 0.2260, model-only 0.2255.
- 2021–25: market 0.212, hybrid 0.211.
- A fitted blend weight of 0.43 on 2021–24 produces 2025 Brier 0.2115 vs 0.2114 for the fixed blend — no meaningful improvement.

**Change:** fit a Platt/isotonic calibration or a logistic blend on inner folds. If it does not beat the market no-vig probability out-of-sample, publish the market probability and use the model only for the score projection. Add a tie probability (~0.4% of games); the normal currently ignores ties.

### 2.6 Add paired uncertainty to every comparison, and fix the backtest/live mismatch

**Evaluation harness:**
- Bootstrap CIs for MAE/RMSE/accuracy/Brier/log loss.
- McNemar test for winner accuracy.
- Diebold–Mariano test for MAE differences.
- ATS/O-U ROI at −110 with CIs, not just win rates.
- Reliability diagram with more bins and CIs (5 bins on 121 games is unstable).
- Report "no detectable edge" when the CI spans zero.

**Backtest/live mismatch:**
- `backtest.weekly_2026()` trains on `week < w`, while `predict.predict_slate()` trains on `gameday < target_day`, which **includes the Thursday game of the same week**. The report says the Week 9 model was retrained through Thursday; the backtest does not. Fix the backtest to use the same `gameday` filter.
- The live Saturday run uses whatever line is in `games.parquet` at that moment; the backtest uses the final `spread_line`. If `spread_line` is the closing line, the backtest's market baseline is stronger than the live model's anchor. Store the line and timestamp at prediction time and evaluate against that, or label the backtest "vs. closing line."
- `SLIM_COLS` in `backtest.py` omits `market_line_missing`; add it so future unlined games are auditable.

**Model selection:**
- `outputs/tune_validation.csv` has 561 configurations, selected and evaluated on the same 2021–25 window. Use nested walk-forward or pre-register a small grid. Report the selection uncertainty.

---

## 3. Tier 2 — New signal the closing line may not fully price

### 3.1 Injury timing and availability

- Week 1 injury weights are zero because `_player_snap_state` groups by `season, team, player`; carry prior-season snap share.
- Add practice-report trend (DNP → LP → FP), position-specific replacement value, QB-out probability, and historical play rates for "questionable" players.
- Late inactives are the most plausible residual edge. nflverse's `report_status` alone is coarse; the inactive list is published ~90 minutes before kickoff.

### 3.2 Travel, rest, and schedule spots

`rest_diff` and `short_week_diff` exist. Add:
- travel distance and time-zone shift,
- consecutive road games,
- off-bye (own and opponent),
- Thursday/Monday short weeks,
- international games,
- look-ahead/letdown spots.

These are orthogonal to team quality — exactly what a residual model needs.

### 3.3 Weather forecast, not historical weather

For unplayed games, wind/temp are often missing or stale. Use a forecast API keyed to stadium and kickoff; wind is the main total driver. This is the most concrete total-model fix.

### 3.4 Neutral-script and opponent-adjusted EPA

The current build aggregates all `pass`/`run` plays with equal weight, including fourth-quarter blowouts and prevent defense. Add:
- neutral-script EPA (win probability 20–80%, first three quarters),
- early-down EPA,
- opponent-adjusted ratings (iterative ridge or a simple team-rating model),
- exclusions for kneels, spikes, aborted snaps.

It will not create an edge by itself because the market does this too, but it makes the residual model's inputs less noisy.

### 3.5 Sample-size-aware rolling priors

`min_periods=1` gives a one-game sample the same weight as a four-game sample. Use:
- empirical-Bayes shrinkage toward the prior-season/league mean with weight k ≈ 2–4 games,
- exponential decay (half-life 4–6 games),
- stronger shrinkage for noisy rates (red zone, turnovers, FG%) than for stable ones (EPA, pace).

This improves the pure model and probably the residual model.

### 3.6 Special teams and 4th-down/coaching

Add ST EPA, punt/kickoff/return efficiency, distance-adjusted FG; 4th-down go rate, 2-point rate, and pace when trailing. Special teams are roughly a third of hidden yards and are underrepresented in the current feature set.

### 3.7 Market microstructure (if betting is the goal)

Add opening line, line movement, reverse line movement, and closing-line value. This is likely a bigger edge than any team-stat feature, but it requires a new data source. Also verify that `spread_line` is actually the close, not an earlier snapshot.

---

## 4. Tier 3 — Model class (test, but expect null)

1. **Nonlinear residual models:** HistGradientBoosting (`HGB_PARAMS` is already in config but unused), ElasticNet, Bayesian ridge, small MLP. Use nested CV and require improvement larger than bootstrap noise.
2. **Hierarchical team/season effects:** partial-pooled team ratings and team-specific residual intercepts (ridge dummies or a Bayesian state-space model). Could improve the pure fallback and capture persistent ATS biases.
3. **Quantile/distributional total model:** if you keep a total model, predict quantiles (pinball loss) or a gamma/lognormal-ish distribution; evaluate with CRPS, not just MAE.
4. **Empirical score-pair distribution:** instead of rounding continuous scores, sample historical score pairs conditioned on predicted margin/total (nearest-neighbor or bivariate distribution). This won't improve MAE much, but it makes exact-score and within-3/7 metrics realistic and gives a proper mode.

---

## 5. Tier 4 — Engineering correctness

- Put the season in prediction/card/graded filenames (`weekXX` collides across seasons).
- Add `generated_at` and `data_as_of` to the prediction CSV.
- Fix the stale "through Week 8" card text in `weekly.py`.
- Use depth-chart snapshots as of the target date, not the latest snapshot.
- Add leakage tests: mutating future games must not change a past game's features.
- Pin dependencies and initialize git.
- Define `pred_winner` from `win_prob` (or report both); currently `pred_margin > 0` and the blended probability can disagree in principle.
- Report ATS ROI at −110 with a CI, not just win rate; add a no-pick band for edges under ~1 point.
- Fix `report.py`'s rounded win recovery (`round(win_acc * n)`) by storing wins/losses explicitly.
- Parameterize `report.py` if you want a weekly report; it is currently a one-off.

---

## 6. Your ideas: OC/DC norms, coaching, and man/zone matchups

This section is the answer to "what about advanced analytics like OC/DC norms, coaching, and WR-vs-zone / DB-vs-zone matchups?"

### 6.1 What data is actually available

| Source | Coverage | What it gives you | Limits |
| :--- | :--- | :--- | :--- |
| `pbp_participation` (nflverse) | 2016–2025; **no 2026 file yet** | `defense_man_zone_type` (MAN/ZONE), `defense_coverage_type` (Cover 0/1/2/3/4/6/9, 2-Man, Combo, Blown), `defense_personnel`, `offense_personnel`, `defenders_in_box`, `number_of_pass_rushers`, `was_pressure`, `time_to_throw`, `route`, all 11 player IDs/positions | No per-defender target responsibility. The man/zone label is one call per play. **2023–24 label distribution is a regime break.** No 2026 in-season data. |
| `ftn_charting` (nflverse) | 2022–2026, updated weekly | motion, play action, RPO, screen, trick play, no-huddle, backfield count, box count, blitzers, pass rushers, QB location, read thrown, contested/catchable/drop | **No man/zone or coverage shell.** The cached file only has 2025; 2026 is available. |
| PFR advanced defense (`advstats_week_def`) | 2018–2026 | individual DB coverage: targets, completions allowed, yards allowed, passer rating allowed, ADOT, missed tackles | Overall, **not split by man/zone**. |
| PFR team pages | 2005–present | head coach, offensive coordinator, defensive coordinator, offensive scheme, defensive alignment | Scraping; rate limits. Coordinator titles do not always equal play-caller. |
| nflverse `games.parquet` | 1999–present | head coach, rest, roof, surface, weather, spread/total, moneylines | Head coach only; no OC/DC. |
| Tracking data (NGS) | 2016–present | coverage responsibility, route running, separation | Not in nflverse; paid/limited access. |

**Key data conclusion:** the free man/zone data is a **historical/prior-season** signal, not a weekly current-season signal. For 2026 live predictions you can use 2025 scheme priors plus current-season FTN proxies (blitz rate, box count, pass rushers, motion, play action, RPO), but not 2026 man/zone rates. If you need in-season man/zone, you need PFF/SIS/NGS or another charting source.

### 6.2 What is directly buildable

**Team scheme profile (stable, useful as a prior):**
- `def_zone_rate`, `def_man_rate`
- `def_cover0_rate`, `def_cover1_rate`, `def_cover2_rate`, `def_cover3_rate`, `def_cover4_rate`, `def_cover6_rate`, `def_2man_rate`
- `def_blitz_rate` (pass rushers > 4), `def_pressure_rate`, `def_box_avg`, `def_pass_rushers_avg`
- `def_nickel_rate`, `def_dime_rate`, `def_base_rate` (from personnel)
- `off_motion_rate`, `off_play_action_rate`, `off_rpo_rate`, `off_screen_rate`, `off_no_huddle_rate` (FTN)
- `off_early_down_pass_rate`, `off_shotgun_rate`, `off_pace`

**Team performance by scheme:**
- `off_epa_vs_zone`, `off_epa_vs_man`
- `off_epa_vs_cover1/2/3/4`
- `def_epa_vs_zone_allowed`, `def_epa_vs_man_allowed`
- `def_epa_vs_cover1/2/3/4_allowed`

**Player performance by scheme (from PBP `receiver_player_id` / `passer_player_id` joined to participation):**
- QB: EPA/dropback, CPOE, success rate vs man/zone; sack rate vs man/zone.
- WR/TE/RB: targets, EPA/target, yards/target, target share vs man/zone; route-specific EPA.
- DB group: team defense EPA allowed by scheme, weighted by which DBs were on the field (`defense_players`, `defense_positions`). This is **unit-level**, not individual responsibility.
- Offensive line: pressure/sack rate vs blitz and vs four-man rush.

**Matchup features:**
```
home_expected_epa = away_def_zone_rate * home_off_epa_vs_zone
                  + (1 - away_def_zone_rate) * home_off_epa_vs_man
away_expected_epa = home_def_zone_rate * away_off_epa_vs_zone
                  + (1 - home_def_zone_rate) * away_off_epa_vs_man
scheme_net = home_expected_epa - away_expected_epa

zone_edge_net = (home_off_epa_vs_zone - away_def_epa_vs_zone_allowed)
              - (away_off_epa_vs_zone - home_def_epa_vs_zone_allowed)
man_edge_net  = analogous
zone_rate_diff = away_def_zone_rate - home_def_zone_rate
```
Add interactions with QB/WR quality, target share, and route mix. Use empirical-Bayes shrinkage toward the league mean; do not use raw one-season player splits.

**Coordinator/coaching features:**
- Team-season-role mapping from PFR (`/teams/{team}/coaches.htm`, table `coaches_year`: HC, OC, DC).
- `new_hc`, `new_oc`, `new_dc` flags; coordinator tenure; play-caller continuity.
- Coordinator prior profile: the coordinator's prior-team scheme tendencies (zone rate, blitz rate, pace, play-action, motion, RPO), shrunk toward the current team's prior.
- Head-coach in-game tendencies from PBP: 4th-down go rate, 2-point rate, timeout usage, pace when trailing, aggressiveness in neutral situations.

### 6.3 What is not directly buildable

- **"WR1 is good vs zone while the opponent's DBs are bad at zone."** The participation data has the receiver on the play and the coverage label, so you can compute **WR1 vs zone** and **team defense vs zone**. It does **not** identify which defender was responsible for the target. Without tracking/target attribution, "DB X is bad at zone" is not available. You can approximate:
  - team/DB-unit zone performance (not individual),
  - individual DB overall coverage quality from PFR,
  - and a matchup proxy: `WR1_epa_vs_zone × opponent_zone_rate × opponent_DB_unit_epa_vs_zone`.
  Be explicit that it is a unit-level proxy, not a player-on-player matchup.
- **In-season 2026 man/zone rates.** No 2026 participation file. Use 2025 priors or a paid source.
- **Stable player-level man/zone splits from one season.** The reliability is too low. Use 2–3 seasons, hierarchical shrinkage, and only target players with large samples.

### 6.4 Stability evidence (2025 season)

**Team scheme tendencies (split-half reliability, Spearman–Brown corrected):**

| Metric | Reliability |
| :--- | ---: |
| Defensive zone rate | **0.855** |
| Blitz rate (>4 rushers) | **0.821** |
| Defenders in box | **0.758** |
| Pass rushers | **0.718** |
| Pressure rate | **0.669** |
| Cover 1 rate | **0.867** |
| Cover 2 rate | **0.830** |
| Cover 3 rate | **0.699** |
| Cover 4 rate | **0.683** |
| 2-Man rate | **0.849** |
| Offensive no-huddle rate | **0.945** |
| Offensive shotgun rate | **0.906** |
| Offensive motion rate (FTN) | **0.861** |
| Offensive RPO rate (FTN) | **0.915** |
| Offensive play-action rate (FTN) | **0.620** |
| Offensive screen rate (FTN) | **0.784** |

**Performance by scheme (split-half reliability):**

| Metric | Reliability |
| :--- | ---: |
| Offense EPA vs zone | **0.623** |
| Offense EPA vs man | **0.459** |
| Defense EPA allowed vs zone | **0.485** |
| Defense EPA allowed vs man | **0.436** |
| QB EPA/dropback vs zone | **0.280** |
| QB EPA/dropback vs man | **0.100** |
| Receiver EPA/target vs zone | **0.292** |
| Receiver EPA/target vs man | **0.482** |

**Route × coverage (2025 league-wide, EPA/target):**

| Route | vs Man | vs Zone | Zone − Man |
| :--- | ---: | ---: | ---: |
| HITCH/CURL | −0.044 | +0.188 | **+0.231** |
| GO | +0.246 | +0.429 | **+0.183** |
| CORNER | +0.298 | +0.472 | **+0.175** |
| WHEEL | +0.201 | +0.496 | **+0.294** |
| SCREEN | −0.092 | −0.045 | +0.047 |
| POST | +0.603 | +0.603 | 0.000 |
| SHALLOW CROSS/DRAG | +0.071 | +0.067 | −0.004 |
| IN/DIG | +0.272 | +0.254 | −0.019 |
| DEEP OUT | +0.310 | +0.249 | −0.061 |
| SLANT | +0.265 | +0.201 | −0.064 |
| QUICK OUT | +0.134 | +0.035 | **−0.098** |
| SWING | +0.069 | −0.114 | **−0.183** |

These route effects are intuitive (hitch/curl, go, corner, wheel beat zone; quick out and swing beat man) and could be useful for **prop/route-matchup** models. At the game level they are already largely priced.

**Label regime break:**

| Season | Zone rate | Cover 1 | Cover 2 | Cover 3 | Cover 4 | 2-Man |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2019 | 62.8% | 30.4% | 13.3% | 30.0% | 12.3% | 2.3% |
| 2020 | 65.6% | 27.4% | 13.2% | 31.7% | 13.1% | 2.1% |
| 2021 | 69.7% | 23.8% | 13.7% | 32.9% | 14.2% | 1.8% |
| 2022 | 71.2% | 22.8% | 14.1% | 32.2% | 15.9% | 1.8% |
| **2023** | **58.5%** | **32.0%** | **15.4%** | **18.3%** | **12.1%** | **6.1%** |
| **2024** | **51.4%** | **36.5%** | **20.6%** | **17.7%** | **6.0%** | **7.5%** |
| 2025 | 69.0% | 22.4% | 22.9% | 27.7% | 10.1% | 4.7% |

The 2023–24 files are from a different pipeline (updated 2025-09-04; earlier files 2023-12-19). Treat 2023–24 as a separate labeling regime, or use a consistent paid source. Do not pool raw labels across the break without season-standardization or regime dummies.

### 6.5 First incremental test (game level)

**Design:**
- Seasons 2021, 2022, 2025 (exclude 2023–24 due to the label break), Weeks 4–18.
- Prior-game rolling features only (8-game window; no current-game information).
- `scheme_net`, `zone_edge_net`, `man_edge_net`, `zone_rate_diff`, and four zone-minus-man features.
- Target: `actual margin − market margin`, `actual total − market total`, and the hybrid residuals.
- n = 671 games.

**In-sample fit:**

| Target | R² | Largest |t| |
| :--- | ---: | :--- |
| Margin residual | **0.0094** | `zone_rate_diff` t = 1.83 |
| Total residual | **0.0032** | `home_def_zone_minus_man` t = 0.92 |
| Hybrid margin residual | **0.0100** | `zone_rate_diff` t = 1.90 |
| Hybrid total residual | **0.0033** | `home_def_zone_minus_man` t = 0.93 |

**Out-of-sample (train two seasons, test the third), univariate `scheme_net`:**

| Target | 2021 ΔMAE | 2022 ΔMAE | 2025 ΔMAE | Pooled ΔMAE |
| :--- | ---: | ---: | ---: | ---: |
| Margin | −0.0085 | +0.0049 | −0.0045 | **−0.0027** |
| Total | +0.0050 | +0.0096 | +0.0032 | **+0.0059** |
| Hybrid margin | −0.0075 | −0.0004 | −0.0037 | **−0.0039** |
| Hybrid total | +0.0070 | +0.0168 | −0.0018 | **+0.0073** |

The coefficient sign is unstable (margin β: +0.88, +0.48, −1.07 across test seasons). The only borderline result is `zone_rate_diff` (t ≈ 1.8–1.9), which is not robust with eight features and multiple testing.

**Conclusion:** at the **game** level, team-level man/zone matchup features do not add measurable value beyond the closing line in this sample. That does not invalidate the idea; it means the edge, if any, is likely at the **player/prop** level or in **specific coordinator-change/situational spots**, not in the game margin/total.

### 6.6 Where the scheme/coaching edge is most likely

1. **Player props, not game lines.** The game market is extremely efficient; prop markets are less so. A WR1 with a large, stable man/zone split facing a zone-heavy defense is a more plausible edge than the same information applied to the game spread. Test:
   - WR1 receiving yards vs opponent zone rate,
   - QB passing yards vs opponent man/zone mix,
   - TE/RB targets and receptions vs scheme,
   - route-specific props (e.g., hitch/curl, crossers).
2. **Coordinator changes and early-season scheme shifts.** The market may be slower to price a new OC/DC's tendencies, especially Weeks 1–4. Use the coordinator's prior-team profile shrunk toward the current team's prior; compare to the market's implied team total.
3. **Situational scheme, not overall rates.** Overall zone/blitz rates are well known. Look at:
   - third-down man rate,
   - red-zone coverage shell,
   - two-minute drill tendencies,
   - blitz rate by down/distance,
   - scheme changes when leading/trailing.
4. **Availability × scheme.** A WR1 out changes the offense's zone-beating ability; a slot receiver's absence changes the route mix. Combine injury features with scheme.
5. **Line movement / market microstructure.** If a coordinator change or scheme shift is not priced, the opening-to-close move may be the trade; this is likely a bigger edge than the scheme feature itself.

### 6.7 Leakage and sample-size rules

- Use only prior games for rolling scheme features; never the current game's participation.
- For Week 1, use prior-season/coordinator-history priors; for Weeks 2–3, blend the prior with the current-season sample; for Week 4+, use current-season rolling with a prior weight of ~2–4 games.
- For player-level splits, use 2–3 seasons and empirical-Bayes shrinkage; do not use a one-season split.
- For coordinator features, use the coordinator's prior seasons only; do not use the current season's results to define the prior.
- Handle the 2023–24 label break with season standardization, regime dummies, or exclusion.
- Validate on 2021–2022 and 2025 separately; treat 2023–24 as a different regime.
- With ~400 zone dropbacks per team-season and ~40–120 targets per player split, expect wide CIs. Require an effect larger than the noise.

### 6.8 Proposed experiment plan

1. **Data layer (1–2 days):** add `pbp_participation` to `download.py`; build a `scheme.py` module with team-game and player-game aggregates by man/zone and shell. Cache to `data/processed/`.
2. **Team-level scheme features (2–3 days):** zone rate, shell mix, blitz rate, box, pressure, nickel/dime, motion/PA/RPO (FTN), rolling with shrinkage. Add to the residual model; nested walk-forward; ablations (tendencies only, performance by scheme, interactions).
3. **Player-level features (1 week):** QB/WR/TE/RB splits vs man/zone with multi-season priors; route × coverage; test on prop-style targets (receiving yards, receptions, passing yards). Use hierarchical shrinkage.
4. **Coordinator layer (3–5 days):** scrape PFR team coaches pages (HC/OC/DC by year); build `new_oc`/`new_dc`, tenure, prior scheme profile; add to the model and test early-season spots.
5. **Market microstructure (if betting):** opening lines, line movement, CLV; test whether scheme/coordinator news predicts movement.
6. **Evaluation:** paired bootstrap, McNemar, Diebold–Mariano, CRPS for distributions; require out-of-sample improvement on multiple seasons.

### 6.9 Verdict on your ideas

- **Good idea, right instinct.** Scheme and coaching are under-modeled in the current feature set, and the data is available for a serious version.
- **But the public data has three constraints:** no 2026 in-season man/zone, a 2023–24 labeling break, and noisy player-level splits.
- **The first game-level test is null.** Do not add these features to the game model expecting a margin/total edge. The better targets are player props, coordinator-change spots, and situational scheme.
- **The exact "WR1 vs zone, DB bad at zone" idea is not directly supported** because there is no per-defender target attribution in the free data. The feasible version is WR1 vs opponent zone rate/unit performance, not WR1 vs a specific DB.
- **If you want the exact version,** you need a tracking/charting source with coverage responsibility (PFF, SIS, NGS). That is a data acquisition decision, not a modeling one.

---

## 7. Prioritized roadmap

| Priority | Work | Effort | Expected impact |
| :--- | :--- | :--- | :--- |
| 1 | Evaluation harness: bootstrap CIs, McNemar, DM, ROI, calibration; fix `gameday` backtest mismatch; add `market_line_missing` | 2–3 days | High (prevents overfitting; makes every later decision honest) |
| 2 | `total_shrink = 0`, drop O/U picks, fix no-line fallback, add `line_source` | 1 day | High (removes a false signal; fixes unlined-game bias) |
| 3 | Joint residual bootstrap; replace fixed intervals and 50/50 blend; add tie probability | 2–3 days | High (calibrated scores/win prob/cover prob) |
| 4 | Injury week-1 snap shares, practice trends, inactive-list timing | 3–5 days | High (most plausible residual edge) |
| 5 | Travel/rest/bye/spot features | 2–3 days | Medium (cheap, orthogonal to team quality) |
| 6 | Weather forecast integration | 2–3 days | Medium (total side) |
| 7 | Neutral-script/opponent-adjusted EPA; sample-size priors | 1–2 weeks | Medium (better inputs; may not move the market-anchored residual) |
| 8 | Scheme/coaching pipeline + player-prop research | 2–4 weeks | Medium (best chance in props/coordinator spots, not game lines) |
| 9 | Nonlinear/hierarchical models; market microstructure | 2–4 weeks | Low–medium (likely null at game level; confirm with nested CV) |

---

## 8. Data acquisition details

### 8.1 Participation (man/zone, personnel, route)

```
https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{season}.parquet
```

- Available 2016–2025; 2026 not yet published (as of 2026-11-14).
- ~2.8–4.5 MB per season.
- Join to PBP on `nflverse_game_id = game_id` and `play_id`.

### 8.2 FTN charting (motion, play action, RPO, blitz, box)

```
https://github.com/nflverse/nflverse-data/releases/download/ftn_charting/ftn_charting_{season}.parquet
```

- Available 2022–2026; 2026 updated within 48 hours.
- Join on `nflverse_game_id = game_id` and `nflverse_play_id = play_id`.
- No man/zone columns in the public subset.

### 8.3 Coordinators (PFR)

- Team coaches page: `https://www.pro-football-reference.com/teams/{team}/coaches.htm`
- Table `coaches_year`: year, coach, OC, DC.
- Team season page: `https://www.pro-football-reference.com/teams/{team}/{season}.htm` — includes "Offensive Coordinator", "Defensive Coordinator", "Offensive Scheme", "Defensive Alignment".
- Scrape with a 2–4 second delay; respect PFR terms. Build a `team, season, role, coach` table.

### 8.4 Head coaches

- Already in `games.parquet`: `home_coach`, `away_coach`.

### 8.5 Tracking / coverage responsibility (only if you need exact DB matchups)

- NFL Next Gen Stats tracking data (coverage responsibility), PFF, Sports Info Solutions.
- These are paid/limited; not part of nflverse.

---

## 9. Files added for the GitHub upload

- `IMPROVEMENTS.md` — this document.
- `.gitignore` — excludes `data/raw/`, `data/processed/`, caches, virtualenvs, and local secrets.
- `requirements.txt` — minimal runtime dependencies (pandas, numpy, scikit-learn, requests, pyarrow). Not pinned; pin later for reproducibility.

No model code has been changed in this commit. The existing `README.md`, `analysis.md`, `learned.md`, `outputs/`, and `src/` are unchanged.

---

## Appendix A — Scheme experiment details

**Data build:**
- Seasons 2019–2025; PBP + participation joined on `game_id`/`play_id`.
- Regular season dropbacks only; exclude kneels, spikes, aborted plays; require a man/zone label.
- 2025: 19,830 labeled dropbacks; 69.0% zone.
- Team-game aggregates: offense EPA/dropback vs man/zone; defense EPA allowed/dropback vs man/zone; zone rate.
- Rolling features: 8-game prior window, shifted one game; Weeks 1–3 use prior-season averages; Weeks 4+ use current-season rolling only (avoids the 2024 label break for the 2025 test).
- Game features: expected EPA, zone edge, man edge, zone-rate difference, zone-minus-man interactions.
- Targets: actual minus market, actual minus hybrid prediction.

**Test sample:** 2021, 2022, 2025; Weeks 4–18; n=671.

**Results:** see §6.5.

**Caveats:**
- Team-level features only; no player-level or coordinator-level features in this test.
- 2023–24 excluded due to the label break; this reduces the sample and may hide a real effect.
- The scheme features are correlated with team quality, which the market already prices; the test is intentionally on the residual, which is the hard part.
- The test uses prior-game rolling features, not the current game's participation; no leakage.
- A null result at the game level does not rule out a player-prop or coordinator-change edge.

---

## Appendix B — References

- nflverse `pbp_participation`: <https://github.com/nflverse/nflverse-data/releases/tag/pbp_participation>
- nflverse `ftn_charting`: <https://github.com/nflverse/nflverse-data/releases/tag/ftn_charting>
- nflreadr participation dictionary: <https://nflreadr.nflverse.com/articles/dictionary_participation.html>
- nflreadr FTN charting dictionary: <https://nflreadr.nflverse.com/articles/dictionary_ftn_charting.html>
- PFR team coaches: <https://www.pro-football-reference.com/teams/>
- Existing project docs: `README.md`, `analysis.md`, `outputs/report.md`
