# NFL Score Model — How It Works

*An in-depth walkthrough of the pipeline, model, validation, and weekly workflow.*
*Analysis date: 2026-11-12.*

---

## The one-sentence version

This is **not** a model that predicts NFL scores from scratch. It is a model that predicts **how the closing line will be wrong**, shrinks that prediction heavily toward zero, and then converts the adjusted margin/total into a score.

That distinction explains almost everything about the design and the results.

---

## 1. End-to-end architecture

```text
nflverse GitHub releases
        │  download.py
        ▼
data/raw/*.parquet
        │  build.py
        ▼
data/processed/team_games.parquet          one row per team-game
        │  features.py
        ▼
data/processed/game_features.parquet       one row per game, 538 columns
        │  model.py
        ▼
predicted margin + total
        │  backtest.py / predict.py / weekly.py
        ▼
outputs/*.csv, weekXX_card.md, report.md
        │  grade.py
        ▼
outputs/weekXX_graded.csv
```

The stages are deliberately separated:

| Path | Role |
| :--- | :--- |
| `src/nflscore/config.py` | Paths, seasons, training start, constants |
| `data.py` | Load cached nflverse parquet files |
| `download.py` | Download/refresh nflverse releases |
| `build.py` | Build team-game table from PBP, players, coverage, snaps, injuries, Elo |
| `features.py` | Leakage-safe weekly state and game-level features |
| `model.py` | Ridge residual model, pure stats model, win probability |
| `metrics.py` | MAE, RMSE, accuracy, Brier, log loss, ATS, O/U |
| `backtest.py` | 2021–2025 season validation + 2026 weekly walk-forward |
| `predict.py` | Next unplayed week predictions |
| `weekly.py` | Human-readable markdown card |
| `grade.py` | Post-week grading against finals |
| `report.py` | Fixed methodology/backtest report |
| `pipeline.py` | End-to-end weekly run |

`learned.md` is not documentation for this project — it is a retrospective from an earlier **FF Stream Lab** kicker/DST project. The current project's authoritative docs are `README.md` and `outputs/report.md`.

---

## 2. Data layer

Everything comes from **nflverse GitHub releases**. No API keys, no scraping.

Cached files include:

- `stats_team_week_{season}.parquet` — team box-score stats
- `stats_player_week_{season}.parquet` — player box-score stats
- `injuries_{season}.parquet` — weekly injury reports
- `snap_counts_{season}.parquet` — snap counts
- `advstats_week_def_{season}.parquet` — PFR advanced defense (coverage/pressure)
- `play_by_play_{season}.parquet` — play-by-play
- `games.parquet` — schedule, finals, spreads, totals, rest, weather, QB IDs
- `depth_charts_{season}.parquet` — depth charts
- `roster_{season}.parquet` — roster (loaded but currently unused)

**Seasons:** 2014–2026 downloaded. The main model trains from **2018** because coverage/pressure data begins there. 2014–2017 remain available as prior-season priors.

**Scope:** regular season only (`game_type == "REG"`). No playoffs, seeding, or playoff rest.

**Refresh behavior:**

- `refresh_season(2026)` force-refreshes the current season's weekly files plus the shared schedule/depth chart/roster.
- `refresh_schedule()` only refreshes `games.parquet` — the fast path used by grading.

**Critical sign convention:** in the cached nflverse file, `spread_line` is **positive when the home team is favored**. The code sets `market_margin = spread_line`. The README says this was verified against moneylines. A sign flip here would wreck the model, so it is worth re-verifying whenever the data source changes.

**Data timing caveats:**

- Closing lines and injury reports are only as fresh as the last refresh.
- Weather in `games.parquet` is game-time historical, not a forecast. For future games it is often missing or stale.
- The README recommends running Saturday after final injury reports and before the first kickoff.

---

## 3. Build layer: from play-by-play to team-games

`build.py` produces `team_games.parquet`, one row per team per game.

### 3.1 Schedule and box score

It merges team-week stats with the schedule and computes:

- `points_for`, `points_allowed`, `margin`, `total_points`, `win`
- Per-play rates: pass/rush yards per play, pass TD rate, INT rate, sack rate, turnover rate
- Turnover margin, field-goal percentage, punting/return efficiency
- Home/away flags, rest, roof, surface, weather, QB IDs/names, referee

### 3.2 Play-by-play aggregation

It views the same scrimmage plays from both sides:

- **Offense:** plays, pass/rush plays, EPA, success rate, first downs, sacks, QB hits, turnovers, explosive plays, pass/rush yards, air yards, YAC, pass over expected
- **Defense:** the same metrics allowed
- **Drives:** one row per drive, then aggregated to offense and defense — drives, red-zone trips, TD/FG/score counts, plays, seconds, starting field position
- **Penalties**

Only `play_type in {"pass", "run"}` is used for scrimmage metrics, so special teams and kickoffs do not contaminate EPA/success.

### 3.3 Player usage

For each team-game:

- **QB:** top QB by attempts; attempts, completions, yards, TDs, INTs, sacks, passing EPA, CPOE, carries, rushing EPA, fumbles lost
- **WR1:** top WR by targets; targets, receptions, yards, TDs, receiving EPA, air yards, YAC, target share, air-yards share, WOPR
- **WR corps:** top 3 WRs by targets, summed
- **TE1:** top TE by targets
- **RB1:** top RB by carries
- **RB room:** all RBs summed

Then it derives rates such as QB EPA/DB, INT rate, sack rate, rush EPA/carry, WR1 EPA/target, RB1 yards/carry.

### 3.4 Coverage and pressure

From PFR advanced defense:

- Team targets, completions allowed, yards allowed, TDs allowed, INTs, pressures, hurries, QB hits, sacks, blitzes
- Target-weighted passer rating allowed and ADOT
- Coverage yards/target, completion %, TD rate
- Top CB by targets
- Top coverage defender by targets
- Top pass rusher by pressures

This is how the model gets "QB vs pass defense" and "WR1 vs coverage" style features.

### 3.5 Snaps

Team offense/defense/special-teams snaps. Also a player-level rolling snap share used to weight injuries: a player who plays 90% of snaps matters more than a bench player.

### 3.6 Injuries

Injury status is converted to a 0–1 availability score:

| Status | Score |
| :--- | ---: |
| Out | 0.00 |
| Doubtful | 0.20 |
| Questionable | 0.70 |
| Did Not Participate | 0.35 |
| Limited | 0.80 |
| Full Participation | 1.00 |

`lost = 1 - status_score`, then weighted by the player's prior 4-game snap share. The result is aggregated by position group:

- `inj_lost_off`, `inj_lost_def`
- `inj_lost_qb`, `inj_lost_rb`, `inj_lost_wr`, `inj_lost_te`, `inj_lost_ol`
- `inj_lost_dl`, `inj_lost_lb`, `inj_lost_db`
- Out/doubtful/questionable counts

It also builds:

- **QB availability flags:** whether the team's most recent starting QB is out/questionable for the current week
- **Skill-position flags:** whether the top recent WR/RB/TE is out/questionable

### 3.7 Elo

A FiveThirtyEight-style Elo:

- Pre-game rating for every game
- Margin-of-victory multiplier: `log(mov + 1) * 2.2 / (elo_diff * 0.001 + 2.2)`, capped at 3
- K = 20
- Home advantage = 65 Elo points
- Between seasons, ratings regress 33% toward 1500 (`1500 + 0.67 * (rating - 1500)`)

The feature layer recomputes Elo state separately, so the Elo columns merged into `team_games` are largely redundant.

---

## 4. Feature layer: the leakage firewall

This is the most important part of the codebase. `features.py` builds `team_week_state.parquet` and `game_features.parquet`.

### 4.1 Rolling team state

For every team/season/week, it computes rolling averages over **games, not calendar weeks**. This matters because a bye must not shorten the window or pull in an older game.

For each metric:

```python
before4 = group[metric].shift(1).rolling(4, min_periods=1).mean()
after4  = group[metric].rolling(4, min_periods=1).mean()
```

- `before4` = average of the previous 4 games
- `after4` = average of the current game plus the previous 3

Then:

- If the team has a game in week W: `roll4 = before4` — i.e. the state **before** the game.
- If the team is on a bye or the week is in the future: `roll4 = after4` forward-filled — i.e. the state **after the last game played**.
- If week 1 has no prior games: fall back to the prior season's average (`_prev`).

It builds a complete team-week grid for weeks 1–18, so byes and future weeks always have a state. `roll8` does the same for key metrics.

This is the core leakage control. Every model feature is based on information available before the target game.

### 4.2 Elo state

Pre-game Elo for teams that play; last post-game Elo for byes/future weeks.

### 4.3 QB state

For the **actual starting QB** in each game, it computes prior-4-game:

- QB EPA/DB
- CPOE
- INT rate
- Sack rate
- Rush EPA/carry
- Attempts
- Career attempts
- Games started

The merge is `merge_asof` on `time_idx = season * 100 + week`, by QB ID, with `allow_exact_matches=False`. That means the model sees the QB's stats **through the prior week only**, never the current game.

If the starter is a rookie/backup with no prior attempts, it falls back to the team's QB aggregate. The `qb_change` flag still tells the model this is a replacement situation.

### 4.4 Injury features

Current-week injury report, weighted by prior snap share, aggregated by position group. QB and skill-position flags are merged separately.

### 4.5 Market features

- `market_margin = spread_line` (positive = home favored)
- `market_total = total_line`
- `market_home_win_prob` from no-vig moneylines
- Fallback win probability from the spread with `sigma = 13.5`
- `home_implied = market_total / 2 + market_margin / 2`
- `away_implied = market_total / 2 - market_margin / 2`

### 4.6 Prediction-time leakage guard

`predict.py` trains only on games with `gameday < target_day`:

```python
train = games[(games.season >= 2018) & games.home_score.notna()]
if pd.notna(target_day):
    train = train[train["gameday"] < target_day]
```

For a normal next-week run, that is every completed game. For a backfill of an earlier week, it excludes anything played on or after the target slate. This is what lets the model retrain through a Thursday game and still predict the Sunday slate honestly.

---

## 5. Game-level feature assembly

`build_game_features` turns the team-week state into one row per game.

### 5.1 Home/away state

It merges the state table twice — once as `home_*`, once as `away_*`.

### 5.2 Matchup features

For each `(offensive_metric, defensive_metric)` pair:

```text
home_adv = home_offense - away_defense
away_adv = away_offense - home_defense
net      = home_adv - away_adv
```

There are 26 pairs, including:

- Overall EPA, pass EPA, rush EPA
- Success rate, pass/rush success
- Explosive pass/rush
- Points/drive, red-zone TD rate, third down
- Turnovers, sacks, QB hits
- Pass/rush yards per play
- Plays/drive, pace
- QB vs pass defense, QB CPOE vs coverage
- WR1 vs coverage, WR corps vs coverage
- RB1 vs run defense
- Pass rush vs protection

It also adds 8-week versions and prior-season versions of the most important metrics, plus straight home-minus-away differences and home-plus-away sums (pace, plays/drive, seconds/play, pass rate, snaps, time of possession).

### 5.3 Injuries and availability

- Directional injury differences: `inj_lost_*_diff`
- Game-level injury sums: `inj_lost_off_sum`, `inj_lost_def_sum`
- Skill-position injury edge: `inj_lost_skill_diff`
- QB out/questionable/change flags and differences
- WR/RB/TE top-player out/questionable flags and differences

### 5.4 Actual starting QB state

The prior-4-game QB stats are merged as-of the prior game, then differences are computed:

- `start_qb_epa_diff`
- `start_qb_cpoe_diff`
- `start_qb_int_diff`
- `start_qb_sack_diff`
- `start_qb_attempts_diff`
- `start_qb_career_diff`
- `start_qb_new_diff`

### 5.5 Situational features

- Rest differential
- Short week
- Dome
- Turf
- Denver altitude
- Division game
- Wind
- Temperature

### 5.6 Naive baseline

```text
naive_home = (home_points_for_roll4 + away_points_allowed_roll4) / 2 + 1.5
naive_away = (away_points_for_roll4 + home_points_allowed_roll4) / 2 - 1.5
```

This is the sanity-check baseline: rolling scoring averages plus home field.

### 5.7 Targets

- `margin = home_score - away_score`
- `total = home_score + away_score`
- `home_win = 1 / 0 / NaN`

The resulting feature table has **3,423 game rows and 538 columns**. The model uses **128 margin features, 152 total features, and 158 stats-only features**.

---

## 6. Model layer

### 6.1 Why margin and total, not two scores

The model predicts:

```text
margin = home_score - away_score
total  = home_score + away_score
```

Then derives:

```text
home_score = total / 2 + margin / 2
away_score = total / 2 - margin / 2
```

This is closer to the market's language, less correlated than two independent team scores, and guarantees the derived scores add up.

### 6.2 The market anchor

The primary model is a **market-anchored residual ridge**:

1. The closing line supplies `market_margin` and `market_total`.
2. Ridge regression predicts the residual:

```text
resid_margin = actual_margin - market_margin
resid_total  = actual_total  - market_total
```

3. The residuals are shrunk before being added back:

```text
pred_margin = market_margin + 0.20 * resid_margin_pred
pred_total  = market_total  + 0.10 * resid_total_pred
```

4. Scores are derived from the adjusted margin/total.

The pipeline for each ridge is:

```text
median imputation → standard scaling → Ridge(alpha=100)
```

### 6.3 Why anchor to the market?

Because the closing line is extremely efficient. A pure stats model sees real signal, but also schedule effects, garbage time, and small-sample player performance. Anchoring to the line and learning only the residual keeps the model from making large, low-confidence predictions.

The cost is that the model will rarely disagree with the market on the winner. That is visible in the Week 9 card: it agrees with the market favorite in **14 of 14** games.

### 6.4 Win probability

```text
model_win_prob = Φ(pred_margin / 12.74)
```

where 12.74 is the historical standard deviation of `actual_margin - market_margin` in the training pool.

The reported probability is a **50/50 blend**:

```text
win_prob = 0.5 * model_win_prob + 0.5 * market_win_prob
```

`market_win_prob` is the no-vig probability implied by the home/away moneylines.

### 6.5 The pure stats comparison

A second pair of ridges predicts `margin` and `total` directly from the non-market features. It is included to show how much of the result comes from stats versus the market anchor.

It is also the fallback when no line is available.

### 6.6 ATS and O/U leans

```text
model_edge = pred_margin - market_margin
total_edge = pred_total - market_total
```

- ATS pick = home if `model_edge > 0`, else away
- O/U pick = Over if `total_edge > 0`, else Under

The card explicitly warns that edges under a point are noise.

---

## 7. Validation layer

### 7.1 2021–2025 season-level walk-forward

For each season 2021–2025:

- Train on all prior seasons from 2018
- Test the held-out season

This is used for model/hyperparameter selection.

### 7.2 2026 weekly walk-forward

For Week *w*:

- Train on 2018–2025 plus 2026 Weeks 1…*w*−1
- Test Week *w*

This mirrors what would have been known before each slate.

### 7.3 Results

**2021–2025 validation averages:**

| Model | Winner acc. | Margin MAE | Total MAE | Score MAE | Brier | Log loss |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Market closing line | 66.6% | 9.75 | 10.31 | 7.24 | 0.212 | 0.611 |
| Naive scoring baseline | 58.8% | 10.80 | 11.15 | 7.83 | 0.240 | 0.674 |
| Pure stats ridge | 63.0% | 10.39 | 10.85 | 7.58 | 0.228 | 0.648 |
| Hybrid market+stats | 66.7% | 9.74 | 10.29 | 7.23 | 0.211 | 0.610 |

**2026 Weeks 1–8 pooled:**

| Model | Winner acc. | Margin MAE | Total MAE | Score MAE | ATS | O/U |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Market closing line | 65.8% | 9.57 | 10.66 | 7.33 | — | — |
| Naive scoring baseline | 55.0% | 10.97 | 12.82 | 8.56 | 46.1% | 39.7% |
| Pure stats ridge | 64.2% | 9.85 | 12.10 | 7.86 | 51.3% | 43.0% |
| Hybrid market+stats | 67.5% | 9.58 | 10.71 | 7.33 | 51.3% | 42.1% |

### 7.4 What the numbers actually mean

- The hybrid model's 2026 winner edge is **81 vs 79 wins out of 120** — a two-game difference. The report itself calls it noise.
- Margin MAE is essentially tied: **9.58 vs 9.57**.
- The total model is slightly **worse** than simply using the market total: **10.71 vs 10.66**.
- ATS **51.3%** and O/U **42.1%** are not edges. The O/U result is below a coin flip.
- The pure stats model is clearly weaker on totals (**12.10 MAE**), confirming that the market anchor is doing real work.
- Calibration is reasonable in the middle bins, but the 0.2–0.4 bin overpredicts the home team (34.0% predicted vs 23.8% actual), and the 0.8–1.0 bin has only five games.
- Exact scores: the model never hits both exact scores in the 121-game sample; it hits at least one team's exact score about 8% of the time. Median team-score error is about 6 points; 90th percentile about 15. The score projection is a **distributional estimate**, not a final-score promise.

The report is unusually honest about this. That is a strength.

---

## 8. Weekly operation

### 8.1 Predict

`run_predictions.cmd` → `scripts/predict_now.py` → `pipeline.run_week(refresh=True, build=True)`:

1. Refresh the current season's nflverse files.
2. Rebuild `team_games.parquet`.
3. Rebuild `team_week_state.parquet` and `game_features.parquet`.
4. Resolve the next unplayed week (earliest week in the latest season with unplayed games).
5. Train on completed games before the target day.
6. Predict.
7. Write `outputs/weekXX_predictions.csv` and `outputs/weekXX_card.md`.

Equivalent CLI:

```powershell
$env:PYTHONPATH="src"
python scripts/weekly.py --refresh
```

### 8.2 Grade

`grade_last_week.cmd` → `scripts/grade_last_week.py`:

1. Refresh only `games.parquet` — a small download.
2. Find completed weeks (all scheduled games have finals), newest first.
3. Pick the newest completed week with a saved prediction file.
4. Grade against finals.
5. Write `outputs/weekXX_graded.csv`.

Grading uses the market line stored in the prediction file, so it is a true pre-game evaluation, not a line-shopping exercise after the fact.

### 8.3 Full rebuild

`scripts/run_all.py` runs build → features → backtest → predict → report.

### 8.4 Card anatomy

The weekly card contains:

- Generated timestamp, training rows, margin sigma, number of games
- Already-final games
- Agreement with the market favorite
- Main table: game, pick, win probability, projected score, 80% score range, line, ATS lean, O/U lean
- Confidence tiers: High / Medium / Slight / Coin flip
- Top 5 model-vs-market edges
- Availability and weather notes
- "How to read this card"

---

## 9. Critical assessment

### What works well

- **Clean separation of stages.** Data, build, features, model, evaluation, prediction, grading are distinct and inspectable.
- **The leakage firewall is thoughtful.** Rolling windows are over games, byes carry forward the post-game state, Week 1 uses prior-season priors, QB state is as-of the prior game, and prediction-time training filters by `gameday`.
- **Honest evaluation.** Market baseline, naive baseline, pure stats, walk-forward, per-week results, calibration, exact-score accuracy, and an explicit "not betting advice" disclaimer.
- **Simple, fast model.** Ridge retrains in seconds. No model registry, no incremental training. Coefficients and per-feature contributions are interpretable.
- **Good weekly ergonomics.** One command refreshes, rebuilds, predicts, and writes both CSV and card. Grading is separate and uses the saved line.
- **Edge cases are handled.** Ties, pushes, missing lines, byes, QB changes, depth-chart promotions, prior-season fallbacks.

### Operational bugs / staleness

1. **The weekly card contains hardcoded Week 8 text.**
   `weekly.py` says "Backtest through Week 8" and "2026 Weeks 1–8 went 51.3% ATS and 42.1% on totals" regardless of the target week. Once you predict Week 10, the card will be stale and misleading. This is the most urgent fix.

2. **Prediction filenames omit the season.**
   `prediction_path(week)` → `weekXX_predictions.csv`; `write_card` → `weekXX_card.md`; `grade_saved` → `weekXX_graded.csv`. A 2025 Week 9 file and a 2026 Week 9 file collide. `grade_last_week.py` can read the wrong season's file if the current week was never predicted. Include the season in the filename.

3. **The prediction CSV has no timestamp.**
   The card has a generated timestamp, but the CSV does not. For auditability, add `generated_at` and `data_as_of` columns. Right now, proving the CSV was written before kickoff relies on file metadata.

4. **`report.py` is a one-off.**
   It reads `week09_predictions.csv` and contains Week 9-specific narrative ("JAX 33, BAL 10", "Week 9 forecasts", "121-game sample"). It is not a general report generator. If you want a report each week, parameterize it.

5. **`_safe_market` silently fills missing lines with the naive baseline.**
   For unlined games, the "market anchor" is not the market — it is the naive scoring baseline, and `win_prob` is blended with a naive-derived probability. The card flags `market_line_missing`, but the CSV and backtest semantics should be explicit. The current 2021–2025 and 2026 backtest samples have no missing lines, so the reported metrics are clean, but future unlined games are a different regime.

6. **The 80% score intervals are global constants.**
   `predict.py` uses fixed offsets: home −12.9/+11.0, away −11.9/+11.3. They are not conditioned on the game, predicted total, or team volatility. They can also produce negative lows for low-scoring projections because the code does not clamp at zero. Compute empirical quantiles from residuals and clamp.

### Statistical / methodological risks

7. **The claimed edge is not statistically established.**
   The 2026 hybrid-vs-market difference is two games out of 120; the 2021–2025 difference is ~0.01 MAE. Both are well within noise. The report says this, but the headline can still be read as "beats the market." It doesn't, meaningfully.

8. **Model selection is noisy.**
   `outputs/tune_validation.csv` contains 561 configurations. The best margin MAE differences are ~0.01 points. Picking the best config on the validation window risks overfitting the window. A simpler pre-registered config, or nested CV with bootstrap confidence intervals, would be more defensible.

9. **The total model is worse than the market.**
   Total MAE 10.71 vs 10.66; O/U 42.1%. The 10% shrink is a compromise. Publishing O/U leans at all is questionable. The card already says "treat leans as diagnostics," but the CSV still contains `ou_pick`.

10. **The win-probability blend is arbitrary.**
    The 50/50 model/market blend is a heuristic, not a fitted calibration layer. The model probability itself uses a single normal sigma. The calibration bins show some miscalibration, especially in the 0.2–0.4 bin and the tiny 0.8–1.0 bin.

11. **No closing-line value or line-movement modeling.**
    The model uses the closing line only. It does not model open-to-close movement, which is where a lot of market information lives. If the goal is betting, this is a missed opportunity.

12. **Weather is historical, not forecast.**
    For future games, wind/temp may be missing or stale. The model imputes medians. The README recommends Saturday, but the feature itself is not a forecast.

### Data / feature risks

13. **Week 1 injury weights are zero.**
    `_player_snap_state` groups by `season, team, player`, so Week 1 has no prior snap share within the season; `off_weight`/`def_weight` are filled with 0. That means `inj_lost_*` features are zero at Week 1, exactly when injury information matters most. Carrying prior-season snap share would fix this.

14. **Depth-chart timing can leak in backfills.**
    `_project_future_qbs` takes the latest depth-chart snapshot per team in the file (`dc.groupby("team")["dt"].transform("max")`), not the snapshot as of the target week. For live weekly predictions this is current, but for backfilling an old week from a refreshed file it can use future information. Filter to `dt <= target_date`.

15. **Injury report timing is fragile.**
    `report_status` is often missing; the model falls back to practice status. Late-week activations and surprise scratches are not captured. The card says this, but the model can silently see zero injuries if the current week's report is not posted yet.

16. **The QB-change flag depends on the schedule's listed starter.**
    nflverse may update the listed QB after the game. For a live prediction, it is a snapshot; for a backfill, it may reflect the actual starter. This is fine for live use but worth knowing for historical replay.

### Engineering hygiene

17. **No dependency pinning or packaging.**
    There is no `requirements.txt`, `pyproject.toml`, or lockfile. The local Python 3.12 environment does not have pandas installed, while the `__pycache__` files are `cpython-315`. Reproducing the exact environment is currently guesswork. Add pinned dependencies.

18. **No tests.**
    For a pipeline with subtle leakage logic, this is a real risk. The repo has many one-off `scripts/check_*.py` and `inspect_*.py` files, but no automated regression tests. At minimum, add tests that:
    - Mutating future games does not change a past game's features.
    - Rolling state uses only prior games.
    - No post-game columns appear in feature lists.
    - A saved prediction file grades correctly, including pushes and ties.

19. **No git repository.**
    The workspace is not a git repo. With this many experiments and generated outputs, version control would help. Initialize git and ignore `data/raw`, `data/processed`, and large outputs as appropriate.

20. **Dead config and unused files.**
    `ID_COLS`, `MARKET_COLS`, `RIDGE_ALPHAS`, `HGB_PARAMS`, `LEAGUE_POINTS`, and `HOME_FIELD_POINTS` are defined but unused. `load_roster` is unused. `ftn_charting_2025.parquet` is present but not used. Clean up or wire them in.

21. **`_context` is fragile.**
    `predict._context` reads columns like `wr_top_id_h` that are not in the final CSV keep list. It works because context is built before the keep list is applied, but a future refactor could break it silently. Build context from a small dedicated frame instead.

22. **`market_line_missing` is not in the backtest outputs.**
    `SLIM_COLS` in `backtest.py` omits it. For future backtests with missing lines, you cannot tell whether a "market" row was a real line or the naive fallback. Add the flag.

23. **`report.py` computes wins by rounding.**
    `srow` uses `round(r['win_acc'] * n_decisive)` to recover wins. That can be off by one. Store wins/losses explicitly in the summary.

---

## 10. Where I'd focus next

If the goal is to make this reliable for weekly use, in order:

1. **Fix the stale card text.** Compute the backtest paragraph from the latest summary, or remove it. The card should never claim "through Week 8" in Week 10.
2. **Add season to prediction/card/graded filenames.** This prevents cross-season collisions and makes grading unambiguous.
3. **Add `generated_at` and `data_as_of` to the prediction CSV.** This is the cheapest auditability improvement.
4. **Clamp and derive the 80% intervals.** Use residual quantiles and `max(0, low)`.
5. **Carry prior-season snap share into Week 1 injury weights.** This is a real signal fix, not cosmetic.
6. **Add a small test suite for leakage invariants.** This is the highest-value engineering investment.
7. **Decide what to do with O/U leans.** Either drop them, or increase the total shrink to zero and present the market total as the baseline. The current model does not beat the market total.
8. **Parameterize `report.py`.** If you want a weekly report, make it take season/week and compute the narrative from data.

The model itself is sound for its stated purpose: a structured second opinion anchored to the closing line, not a market-beating oracle. The main risks are operational staleness and reproducibility, not the ridge math.
