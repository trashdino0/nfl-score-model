# Learned — FF Stream Lab

Everything worth carrying forward from building weekly kicker and team-defense
projections. Written so a future version of me (or another agent) can rebuild
this, avoid the same traps, and reuse the good parts for an NFL game-score model.

---

## TL;DR

- **Team defense is a real streaming edge. Kicker is close to a coin flip.**
- The pipeline is deliberately simple: nflverse data → leakage-safe rolling
  features → regularized linear models → weekly board with floor/ceiling.
- The hard part is not the model. It is **scoring definitions, leakage control,
  and honest evaluation by week**.
- No small feature blend or model tweak fixed the kicker signal. The remaining
  upside is operational: injuries, late kicker changes, and live weather.
- For game scores, model **margin and total**, not two independent team scores.
  Benchmark against the closing line, not just naive averages.

---

## 1. The data

### Source

All data comes from **nflverse GitHub releases**. No API keys, no scraping.

| File | Release | Use |
| --- | --- | --- |
| `games.parquet` | `schedules` | Schedule, final scores, spread, total, rest, roof, surface, weather |
| `stats_player_week_{season}.parquet` | `stats_player` | Kicker FG/PAT stats |
| `play_by_play_{season}.parquet` | `pbp` | Team offense/defense metrics, drives, red zone, pressure |

Cached under `data/raw/`; processed tables under `data/processed/`.

### Coverage

- Seasons: **2014–2026**
- 2026 was partial during development (Weeks 1–8 complete, Week 9 in progress)
- `data/processed/team_games.parquet`: **6,846 rows**
- `data/processed/kicker_games.parquet`: **6,528 rows**
- Walk-forward evaluation: 2021–2025 = 2,712 kicker rows / 2,718 DST rows
- 2026 partial evaluation: 244 rows each

### Why 2014?

It is a practical cutoff, not a requirement.

- Gives ~12 seasons of consistent nflverse weekly/PBP data.
- Enough rows to estimate 77 kicker features and 248 DST features.
- Rare events (DST touchdowns, safeties, blocks, long FGs) need many seasons.
- Provides burn-in before the 2021–2025 test window.
- The oldest seasons matter least; recent form dominates. Cutting to 2018+
  would likely produce similar results. 2014 is cheap to keep.

### Data gotchas

1. **nflverse `fantasy_points` excludes kickers and team defenses.** Compute
   scoring yourself from `config/scoring.json`.
2. **The current week may be partially complete.** Filter out games that
   already have a final score (`team_score.notna()`) unless explicitly asked.
3. **A Thursday game can finish before Sunday.** `latest_kicker_rows` must
   ignore the target week's stats, or Thursday results leak into Sunday rows.
4. **Future games have no stats, only schedule context.** That is intentional:
   rolling features then represent only prior information.
5. **Weather in `games.parquet` is game-time, not a forecast.** For future
   weeks it is often missing or stale. Do not pretend it is a live forecast.
6. **Byes, postponements, and international games** create missing weeks.
   Rolling windows must tolerate gaps (`min_periods=1`).
7. **Team abbreviations change over time** (e.g., LA/LAR, WAS). Normalize or
   verify before joining across seasons.
8. **`build` is global.** Running `ffstream build --seasons 2026` overwrites
   the processed tables with only 2026 data and destroys training history.
   Run `ffstream build` with no season filter to rebuild all seasons.

---

## 2. Scoring

`config/scoring.json` is the single source of truth. Change it and rerun
`build`; features that depend on scoring (kicker points, DST points) are
recomputed.

Kicker scoring used:

- FG distance buckets (0–19, 20–29, 30–39, 40–49, 50–59, 60+)
- PAT made/missed
- Game-winning FG bonus (if your league uses it)

DST scoring used:

- Points allowed buckets
- Sacks, interceptions, fumble recoveries, safeties, defensive TDs
- Blocked kicks / return TDs where applicable

**Lesson:** define scoring before building features. A scoring change alters the
target, the rolling history, and the backtest. Rerun `build` and `backtest`
after any change.

---

## 3. Pipeline architecture

```
refresh  ->  data/raw/*.parquet
build    ->  data/processed/team_games.parquet
             data/processed/kicker_games.parquet
features ->  rolling, schedule, opponent, prior-season features
train    ->  regularized model + uncertainty
week     ->  outputs/week_YYYY_WW_report.html + CSVs
backtest ->  walk-forward metrics, predictions, feature importance
```

Commands:

```powershell
$env:PYTHONPATH="src"

python -m ffstream refresh --seasons 2014-2026
python -m ffstream build
python -m ffstream week --season 2026 --week 10 --top 20
python -m ffstream backtest --seasons 2021-2025
python -m ffstream train --season 2026 --week 10
```

Windows shortcut: double-click `weekly.cmd`. It refreshes the current season,
rebuilds all seasons, and writes the next board.

`week` retrains automatically before building the board, so `train` is only
needed when you explicitly want persisted model artifacts.

---

## 4. Feature engineering rules

### The one rule that matters

**Every rolling feature is shifted one game within its group.**

```python
shifted = df.groupby("team")[metric].shift(1)
df[f"{metric}_last4"] = shifted.groupby(df["team"]).transform(
    lambda s: s.rolling(4, min_periods=1).mean()
)
```

If you skip the shift, the model sees the game it is predicting. Every
backtest will look great and every live week will disappoint.

### Feature families

- **Form:** rolling 4-game and 8-game averages, season-to-date, prior season,
  career (kickers).
- **Offense:** points, yards, plays, EPA, success rate, pass/rush EPA, pass
  rate, early-down pass rate, turnovers, sacks, QB hits, explosive rate,
  red-zone trips and TD rate, FG attempts, points/drive, plays/drive.
- **Defense:** points/yards/EPA allowed, success allowed, sacks, interceptions,
  fumble recoveries, TDs, safeties, takeaways, QB hits, explosive rate allowed,
  red-zone TD rate allowed, FG attempts allowed, pressure rate.
- **Kicker:** FG attempts/makes by distance, long, PAT attempts, points,
  volatility (`std_last8`), games played, whether the kicker is new.
- **Schedule/market:** spread, total, implied team total, implied margin,
  home/away, rest, rest differential, short week, division game, dome, turf,
  wind, cold, altitude, implied-total ranks, missing-line flag.

### Implied totals

With `total_line` and `spread_line` where the spread is from the home team's
perspective (negative = home favored):

```python
home_implied = total_line / 2 - spread_line / 2
away_implied = total_line / 2 + spread_line / 2
```

Then select the team's own implied total and the opponent's based on `is_home`.
This is the strongest single input for both positions and a good baseline.

### Kicker-specific features that matter

- Team implied total (game environment)
- Opponent implied total and rank
- Implied margin / game closeness
- Dome, wind, cold, altitude
- Own offense: red-zone trips, red-zone TD rate (TDs kill FG chances),
  FG attempts, points per drive, pace
- Opponent defense: FG attempts allowed, points/EPA allowed, pressure, sacks
- Kicker form: attempts and makes, 40+ and 50+ attempts, long, volatility

### DST-specific features that matter

- Opponent implied total (lower is better)
- Implied margin / being a favorite
- Total line
- Own pressure rate, sacks, takeaways, defensive TDs
- Opponent turnovers, sacks allowed, red-zone trips, pace
- Prior-season DST scoring and defensive EPA

### Feature-importance caution

Standardized linear coefficients are directional, not causal. Correlated
features split credit arbitrarily. The kicker model's largest coefficient was
`implied_margin` **negative**, which sounds like "underdogs produce kicker
points" but is more likely a noisy game-script proxy. Do not turn coefficients
into betting rules without out-of-sample testing.

---

## 5. Models

### Kicker

- **ElasticNet** (`alpha=0.1`, `l1_ratio=0.5`), 77 curated features.
- Pipeline: median imputer → standard scaler → ElasticNet.
- **50/50 blend** with a Vegas-implied baseline: a univariate linear regression
  of kicker points on the team's implied total.
- Blend weight: `KICKER_BLEND = 0.5`.
- Rationale: kicker scoring is extremely noisy; the market total is a stable
  prior. The blend was more stable in the 2026 holdout without hurting
  2021–2025.

### Team defense

- **Ridge** (`alpha=500`), 248 features.
- Model-only (`DST_BLEND = 1.0`).
- Rationale: DST has a materially stronger signal and the full rolling feature
  set; blending with Vegas or form did not help.

### Uncertainty and ranking

```python
sigma = residual_std * clip(recent_vol / median_vol, 0.6, 1.8)
p10 = mean - 1.2816 * sigma
p90 = mean + 1.2816 * sigma
boom_prob = P(score >= threshold)   # 10 for K, 12 for DST
stream_score = mean + 0.12 * max(p90 - mean, 0) + boom_prob
```

- Kicker board ranks by `stream_score`.
- DST board ranks by projected mean.
- The ceiling tilt helped the kicker top-1 in the 2026 sample (8.25 vs 7.25)
  but slightly hurt the top-5 group (8.88 vs 9.00). Treat it as a tie-breaker,
  not a large edge.

### Why linear models?

Gradient boosting (LightGBM) was tried and removed. With ~6,500 kicker rows and
a very noisy target, it was unstable and did not beat the regularized linear
model. Simpler models also retrain in seconds, which makes weekly walk-forward
testing painless.

**Lesson:** on small, noisy tabular data, regularized linear models are a strong
baseline. Try boosting only after you have a trustworthy validation setup.

---

## 6. Evaluation

### Metrics

- MAE and RMSE for point projections.
- Spearman and Pearson correlation across players/teams within a week.
- Top-1/3/5 actual points vs league average and vs the oracle top-k.
- Top-k hit rate (overlap between predicted and actual top-k).
- Bias (mean prediction minus mean actual).
- Baselines: Vegas-only, recent form, Vegas + form.

### Validation design

- **Walk-forward by season** (2021–2025): train on all prior seasons, predict
  the next season.
- **Weekly replay** (2026): retrain before each week using only prior weeks.
- Never random-split time-series data.
- Evaluate per week first, then pool. Pooled correlation can hide a model that
  is good in one week and terrible in another.

### Ties and ranks

Rank **within each week** with `method="min"`. If several players tie at the
top-5 cutoff, include all of them in the "actual top 5" table and note the
extra rows. Ranking globally across weeks produces absurd ranks (e.g., `T216`)
and makes the review unreadable.

---

## 7. Results

### 2021–2025 walk-forward

| Position | MAE | Spearman | Top-5 actual | Gain vs league |
| --- | ---: | ---: | ---: | ---: |
| Kicker | — | 0.165 | 8.78 | +0.98 |
| DST | — | 0.326 | 10.47 | +3.63 |

### 2026 partial season (Weeks 1–8 replay)

| Position | MAE | Spearman | Top-5 actual | League avg | Gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kicker | 3.81 | 0.025–0.04 | 9.00 | 8.15 | +0.85 |
| DST | 4.40 | 0.345–0.355 | 9.30 | 6.25 | +3.05 |

### Hit rates (2026 Weeks 1–8)

| Measure | Kicker | DST |
| --- | ---: | ---: |
| Top-5 group beat league, by week | 62.5% | 87.5% |
| Individual top-5 picks beat league | 47.5% | 67.5% |
| Top-5 picks finished top-10 | 37.5% | 57.5% |
| Top-5 picks finished top-5 | 27.5% | 40.0% |
| Average percentile of top-5 picks | 56th | 69th |
| Top-1 pick beat league | 25.0% | 62.5% |
| Average percentile of top-1 pick | 43rd | 70th |
| Pooled Spearman | 0.025 | 0.345 |

### Interpretation

- **DST is actionable.** Roughly two-thirds of individual top-5 picks beat the
  weekly league average; the group beat the league in 7 of 8 weeks.
- **Kicker is a ranking aid, not a projection.** The top-5 group can beat the
  league while the single top pick does not. One big week carries the average.
- **Exact top-5 overlap is harsh but informative:** 1.4/5 kickers (28%) and
  1.9/5 defenses (38%).
- The 8-week sample is small. Treat all 2026 numbers as noisy.

---

## 8. What did not work

Tested on both 2021–2025 and the 2026 weekly replay:

- Blending kicker projections with team implied total, opponent implied total,
  recent form, or dome/weather flags: neutral to worse.
- Blending DST with opponent implied total, total line, or recent form: neutral
  to worse.
- Gradient boosting / LightGBM: removed; no reliable gain, less stable.
- Ranking kickers by pure ceiling: helps the top-1 in one sample, hurts the
  top-5 group.

**Conclusion:** there is no small model tweak that turns kickers into a strong
signal. The remaining improvements are data improvements, not model
improvements:

1. Live injury/inactive news.
2. Late kicker signings and depth-chart changes.
3. Actual weather forecasts.
4. Roster/transaction data to drop kickers who are no longer on the team.
5. Better handling of short-week and backup-QB games.

---

## 9. Operational lessons

- **Run the weekly routine after Monday night**, once the week's games are
  final. Double-click `weekly.cmd` or run the three commands manually.
- **`week` auto-detects the current slate** as the first week with unplayed
  games. If Monday's game is still marked unplayed, it will rebuild the same
  week. Pass `--week` to force the next one.
- **Completed games are excluded** unless `--include-completed` is passed.
- **Late kicker changes are invisible.** The board uses the last kicker with a
  recorded game. If a team signs a new kicker, the model may still list the old
  one. Check the news before locking a lineup.
- **The replay scores the kicker who actually played.** A real lineup decision
  made before kickoff can be worse. Do not treat the backtest as a lineup
  simulator.
- **Keep prediction logs.** Save the board before games start; otherwise you
  cannot evaluate honestly later.
- **Model artifacts are cheap.** Linear models retrain in seconds; there is no
  need for a model registry or incremental training.
- **Check the output directory.** Reports are self-contained HTML; CSVs are the
  machine-readable version.

---

## 10. Repo map

```
config/scoring.json          League scoring rules
src/ffstream/data.py         nflverse download/cache, current-week detection
src/ffstream/build.py        PBP aggregation, team/kicker tables, DST scoring
src/ffstream/scoring.py      Kicker and DST point functions
src/ffstream/features.py     Leakage-safe rolling features, prediction rows
src/ffstream/model.py        Training, Vegas blend, uncertainty, walk-forward
src/ffstream/backtest.py     Backtest orchestration and artifacts
src/ffstream/report.py       Tiers, CSV, self-contained HTML
src/ffstream/cli.py          refresh / build / backtest / week / train
scripts/backtest_review.py   Model top-5 vs actual top-5 review
scripts/weekly.ps1           Weekly refresh + build + board
weekly.cmd                   Double-click Windows launcher
tests/                       Pytest suite
```

Ignored: `data/raw/`, `data/processed/`, `models/`, `outputs/`.

---

## 11. If you build an NFL game-score model next

### Predict margin and total, not two independent scores

Define:

```
margin = home_score - away_score
total  = home_score + away_score
```

Then derive:

```
home_score = total / 2 + margin / 2
away_score = total / 2 - margin / 2
```

Why: margin and total are closer to the market's language, are less correlated
with each other than the two raw scores, and guarantee the derived scores add
up. Predicting each team's score independently often produces inconsistent
margin/total combinations and wastes data.

### Start with the market baseline

The closing spread and total are extremely efficient. Convert them to implied
scores:

```
home_implied = total / 2 - spread / 2
away_implied = total / 2 + spread / 2
```

Use that as your baseline. A model that cannot beat the closing line can still
be useful for analysis and player projections, but it is not a betting model.

### Targets and metrics

- Margin: MAE, RMSE, correlation, sign accuracy.
- Total: MAE, RMSE, correlation.
- Win probability: Brier score and log loss (better than accuracy alone).
- Betting: ATS cover rate, over/under rate, ROI at -110, and closing-line value.
- Evaluate by week and by season; a single hot streak is not an edge.

Rough expectations: a decent model lands in the **10–14 point MAE** range for
margin and total. Beating the closing line consistently is much harder than
beating a naive average. Treat any ATS rate above ~55% over a small sample as
suspicious until proven over multiple seasons.

### Features that transfer from this project

- Offensive and defensive EPA per play, success rate, pass/rush splits.
- Pace, plays per drive, points per drive.
- Turnovers, sacks, pressure rate, explosive rate.
- Red-zone trips and TD rate.
- Rest, short week, travel, division, dome, wind, cold, altitude.
- Prior-season priors and returning production.
- Market lines and implied totals.

### New features worth adding

- **Quarterback availability and quality.** The biggest single swing factor.
- **Injuries along the offensive line and at skill positions.**
- **Coaching changes and scheme turnover.**
- **Travel distance and time-zone shifts.**
- **Short-week and Thursday-game effects.**
- **Referee tendencies** (weak, but cheap to test).
- **Weather forecast** at kickoff, not game-time historical values.
- **Line movement** from open to close, if you can timestamp it.

### Leakage traps for a game model

- Using same-week box scores or final scores.
- Using the closing line when your prediction time was earlier.
- Letting Thursday results leak into Sunday predictions.
- Random train/test splits instead of walk-forward.
- Using season-long averages that include the game being predicted.
- Mixing regular season and playoffs without a flag.
- Forgetting that injuries and weather change after your prediction timestamp.

### Validation plan

1. Build a naive baseline: season-to-date scoring averages + home-field edge.
2. Build a market baseline: closing spread/total implied scores.
3. Train Ridge/ElasticNet on 2014–2020, test 2021–2025.
4. Add gradient boosting and compare.
5. Run a weekly replay for 2026: retrain before each week.
6. Track predictions with timestamps before kickoff.
7. Only then consider betting or public-facing picks.

### Suggested project shape

```
src/nflscore/
  data.py          reuse the nflverse download/cache layer
  build.py         team-game table + game-level home/away rows
  features.py      rolling team metrics, rest, weather, market
  model.py         margin model, total model, uncertainty, win probability
  backtest.py      walk-forward, market baseline, ATS/O-U metrics
  report.py        weekly game board with projected score and intervals
  cli.py           refresh / build / backtest / week
```

Reuse `team_games.parquet` and most of `features.py`. The main new work is
reshaping to one row per game with home/away perspectives and building a
credible market baseline.

### Realistic mindset

- The market is the opponent. Your first goal is to match it, not beat it.
- Injury and weather information is where most of the remaining edge lives.
- A good projection model is useful even if it does not beat the spread: it
  powers win probabilities, player projections, and content.
- Keep the evaluation honest. A model that is fun to talk about but cannot beat
  a naive baseline is not ready.

---

## 12. Reusable checklist

- [ ] Define scoring before building features.
- [ ] Verify the data's season/week coverage.
- [ ] Shift every rolling feature by one game.
- [ ] Exclude the target week's completed games.
- [ ] Compute implied totals from spread and total.
- [ ] Build a naive baseline and a market baseline.
- [ ] Evaluate per week, then pool.
- [ ] Rank within week, not globally.
- [ ] Include tied players at the cutoff.
- [ ] Report MAE, correlation, top-k gain, and hit rate.
- [ ] Keep a prediction log with timestamps.
- [ ] Check late injuries, kicker changes, and weather manually.
- [ ] Rerun `build` and `backtest` after any scoring change.
- [ ] Do not overfit to one partial season.
