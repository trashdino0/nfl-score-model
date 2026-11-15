# NFL Score Model — 2026 Backtest and Week 9 Forecast

*Generated 2026-11-06. All predictions are model estimates, not betting advice.*

## 1. Executive summary

The model predicts **margin and total separately**, then converts them into a final score:

```
home_score = total / 2 + margin / 2
away_score = total / 2 - margin / 2
```

It is trained on nflverse play-by-play, team, player, injury, snap-count, depth-chart,
advanced-defense, Elo, and market data. The primary model is **market-anchored**: it
predicts how much the actual result should differ from the closing line and shrinks that
adjustment heavily toward zero.

**2026 Weeks 1–8 backtest (121 games, 120 decisive):**

- **Hybrid model: 67.5% winner accuracy**
  (81/120), margin MAE
  9.58, total MAE
  10.71, score MAE
  7.33.
- **Market closing line: 65.8%**
  (79/120), margin MAE
  9.57, total MAE
  10.66, score MAE
  7.33.
- **Pure stats model: 64.2%**, margin MAE
  9.85, total MAE
  12.10.

The result is deliberately sobering: the closing market is extremely efficient. The
hybrid model edges it slightly on winner accuracy in this sample (by about two games) and
is essentially tied on margin and score error. It does **not** beat the market total; the
total adjustment is kept very small for that reason. The stats-only model is respectable
but clearly weaker than the market.

**Week 9:** the model forecasts the 14 remaining games. It agrees with the market favorite
in every game, so the useful output is the projected score, win probability, and the size
of the model's lean against the line — not a set of contrarian winners. The largest leans
are BUF -3 (BUF +0.87); LAC +6 (LAC +0.81); LA -4.5 (LA +0.76); SEA -8.5 (SEA +0.76); DAL -4.5 (DAL +0.41).

## 2. Data and feature construction

- **Source:** nflverse GitHub release data only (no API keys).
- **Seasons:** 2014–2026 downloaded; the main model trains from **2018** because
  coverage/pressure data begins there. 2014–2017 are retained as prior-season priors.
- **Feature table:** 3,423 game rows and 538 columns.
- **Training pool:** 2,249 completed games from 2018–2026 (through the
  Thursday Week 9 game).
- **Feature counts:** 128 margin features,
  152 total features, 158 stats-only
  features.

Feature families:

- **Team offense/defense:** EPA per play, pass/rush EPA, success rate, explosiveness,
  points/drive, plays/drive, pace, red-zone efficiency, turnovers, sacks, pressure.
- **Player-level:** starting-QB EPA/CPOE/interception/sack rates and career experience;
  WR1 and WR corps vs coverage; RB1 vs run defense; top pass-rusher pressure/hurries/sacks;
  skill-position usage and availability.
- **Matchups:** each offensive metric is paired with the opposing defense and expressed as
  a home advantage, away advantage, and net edge.
- **Injuries:** current-week out/questionable/doubtful counts and lost-production weights
  by position group; QB availability and QB-change flags.
- **Situational:** rest, short week, dome/turf, altitude, division game, wind, temperature.
- **Market:** closing spread, total, no-vig win probability, and implied scores.
  Sign convention: in this cached nflverse file, `spread_line` is positive when the
  home team is favored (verified against the home/away moneylines), so
  `market_margin = spread_line`.
- **Elo:** pre-game Elo state and difference.

### Leakage controls

- Rolling team and player features are shifted one game and use only prior games.
- The team-week grid includes byes; a bye carries forward the state after the last game
  rather than shortening the window.
- Week 1 uses the prior season's average.
- QB state is merged as-of the prior game, never the current game.
- Current-week injury reports and closing market lines are treated as pre-game information.
- Post-game columns (scores, results, overtime, win/loss) are excluded from all feature
  lists.

## 3. Model design

### 3.1 Market-anchored residual model

For each game:

1. The closing line supplies `market_margin` and `market_total`.
2. A ridge regression predicts the **residual** `actual_margin - market_margin`.
3. A second ridge predicts `actual_total - market_total`.
4. The residuals are shrunk before being added back:

```
pred_margin = market_margin + 0.20 * residual_margin
pred_total  = market_total  + 0.10 * residual_total
```

The shrink factors were selected on 2021–2025 season-level walk-forward validation, not on
the 2026 backtest. The margin model uses matchup, QB, and directional injury features. The
total model adds team, pace, situational, and Elo features.

### 3.2 Why anchor to the market?

A pure stats model is noisy: it sees real signal, but it also sees schedule effects,
garbage time, and small-sample player performance. The closing line aggregates far more
information than any public model. Anchoring to it and learning only the residual keeps the
model from making large, low-confidence predictions. The cost is that the model will rarely
disagree with the market on the winner.

### 3.3 Win probability and scores

Win probability comes from a normal approximation to the predicted margin:

```
P(home win) = Φ(pred_margin / σ)
```

with σ = 12.74, estimated from the historical distribution of
`actual_margin - market_margin` in the training pool. The reported `win_prob` is a 50/50
blend of this model probability and the no-vig market probability. Final scores use the
margin/total conversion above.

### 3.4 Pure stats comparison

The pure stats model uses the same ridge machinery but predicts margin and total directly
from the non-market features. It is included to show how much of the result comes from the
stats versus the market anchor.

## 4. Backtest methodology

- **2021–2025 season-level walk-forward:** train on all prior seasons from 2018, test on the
  next season. Used for model/hyperparameter selection.
- **2026 weekly walk-forward:** for Week *w*, train only on 2018–2025 plus 2026 Weeks
  1…*w*−1. This mirrors what would have been known before each slate.
- **Metrics:** winner accuracy (ties excluded), margin/total MAE and RMSE, home/away/score
  MAE, correlation, Brier score, log loss, ATS and over/under rates.
- **ATS/O-U:** model side versus the closing line. A pure market baseline has zero edge and
  therefore no ATS/O-U pick; those cells are shown as “—”.

## 5. Validation: 2021–2025

| Model | Winner acc. | Margin MAE | Total MAE | Score MAE | Brier | Log loss |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Market closing line | 66.6% | 9.75 | 10.31 | 7.24 | 0.212 | 0.611 |
| Naive scoring baseline | 58.8% | 10.80 | 11.15 | 7.83 | 0.240 | 0.674 |
| Pure stats ridge | 63.0% | 10.39 | 10.85 | 7.58 | 0.228 | 0.648 |
| Hybrid market+stats | 66.7% | 9.74 | 10.29 | 7.23 | 0.211 | 0.610 |

The hybrid model is a hair better than the market on every average error metric, but the
gaps are tiny (0.01–0.02 points/percentage points). This is the expected size of a real
edge in a highly efficient market; it is not a guarantee that the edge persists.

## 6. Backtest: 2026 Weeks 1–8

| Model | Winner acc. | Margin MAE | Total MAE | Score MAE | Brier | Log loss | ATS | O/U |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Market closing line | 65.8% (79/120) | 9.57 | 10.66 | 7.33 | 0.227 | 0.648 | — | — |
| Naive scoring baseline | 55.0% (66/120) | 10.97 | 12.82 | 8.56 | 0.263 | 0.732 | 46.1% (115) | 39.7% (121) |
| Pure stats ridge | 64.2% (77/120) | 9.85 | 12.10 | 7.86 | 0.223 | 0.637 | 51.3% (115) | 43.0% (121) |
| Hybrid market+stats | 67.5% (81/120) | 9.58 | 10.71 | 7.33 | 0.226 | 0.646 | 51.3% (115) | 42.1% (121) |

**Reading the table:**

- The hybrid model wins 67.5% of decisive games,
  versus 65.8% for the closing line. That is a
  two-game difference out of 120; treat it as noise, not proof of a persistent edge.
- Margin MAE is essentially tied (9.58 vs
  9.57).
- The total model is slightly worse than simply using the market total
  (10.71 vs
  10.66). This is why the total shrink is only
  10%.
- The model's ATS rate is 51.3% and its O/U rate
  is 42.1%. Neither is a betting edge; the O/U
  result is below a coin flip. Do not use the O/U leans as recommendations.
- The pure stats model is clearly weaker on totals
  (12.10 MAE) and slightly weaker on winners
  (64.2%), confirming that the market anchor is doing
  real work.

### Per-week results

| Week | Games | Market acc. | Hybrid acc. | Pure acc. | Market margin MAE | Hybrid margin MAE | Hybrid total MAE |
| :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 75.0% | 75.0% | 50.0% | 11.28 | 11.41 | 11.64 |
| 2 | 16 | 68.8% | 68.8% | 50.0% | 11.84 | 11.64 | 10.84 |
| 3 | 16 | 56.2% | 62.5% | 62.5% | 8.09 | 8.31 | 10.86 |
| 4 | 16 | 56.2% | 62.5% | 81.2% | 7.31 | 6.97 | 9.66 |
| 5 | 15 | 64.3% | 64.3% | 78.6% | 10.40 | 10.41 | 10.32 |
| 6 | 14 | 57.1% | 57.1% | 57.1% | 6.64 | 6.93 | 10.43 |
| 7 | 14 | 71.4% | 71.4% | 57.1% | 10.07 | 10.30 | 10.67 |
| 8 | 14 | 78.6% | 78.6% | 78.6% | 10.82 | 10.63 | 11.25 |

### How accurate are the exact scores?

Winner accuracy and exact-score accuracy are very different questions. The model predicts
a continuous score, but an NFL final score is a narrow integer target with a wide
distribution. Rounding the projections gives:

| Score metric | Hybrid model | Market baseline |
| :--- | ---: | ---: |
| Exact score, both teams | 0 (0.0%) | 0 (0.0%) |
| Exact score, at least one team | 10 (8.3%) | 10 (8.3%) |
| Exact margin | 7 (5.8%) | 8 (6.6%) |
| Exact total | 1 (0.8%) | 1 (0.8%) |
| Both scores within 3 points | 10 (8.3%) | 9 (7.4%) |
| Both scores within 7 points | 41 (33.9%) | 39 (32.2%) |
| Per-team score within 3 points | 23.6% | 23.6% |
| Per-team score within 7 points | 57.9% | 57.0% |
| Median team-score error | 6.0 pts | 6.0 pts |
| 90th percentile team-score error | 15.0 pts | 15.0 pts |
| Score MAE (rounded) | 7.36 | 7.34 |

The model never hits both exact scores in this 121-game sample, and the market doesn't
either. Hitting one team's exact score happens about 8% of the time for both. The typical
team-score error is about 7 points, with a 90th percentile around 15 points. In other words,
the score projection is a **distributional estimate** — useful for the implied game script
and for ranges — not a precise final-score forecast. The hybrid and market score errors are
effectively identical.

### Calibration

| Predicted win-prob. bin | Games | Predicted | Actual |
| :--- | ---: | ---: | ---: |
| (-0.001, 0.2] | 1 | 19.4% | 0.0% |
| (0.2, 0.4] | 21 | 34.0% | 23.8% |
| (0.4, 0.6] | 47 | 50.3% | 48.9% |
| (0.6, 0.8] | 46 | 68.9% | 67.4% |
| (0.8, 1.0] | 5 | 83.9% | 60.0% |

Calibration is reasonable in the middle bins. The 0.2–0.4 bin overpredicts the home team
(34.0% predicted vs 23.8% actual), and the 0.8–1.0 bin has only five games, so its 60%
actual rate is noisy. The model is not systematically overconfident in the extreme
favorites; the small samples simply make the bins unstable.

## 7. Week 9 forecasts

The Thursday game is already final: **JAX 33, BAL 10**. The model was retrained through that
game and forecasts the 14 remaining Sunday/Monday games.

| Date | Matchup | Pick | Predicted score | Pick win prob. | Market line | ATS lean | O/U lean | Model lean |
| :--- | :--- | :--- | :--- | ---: | :--- | :--- | :--- | :--- |
| Sun 11/08 | ARI @ SEA | SEA | ARI 15.1 – SEA 24.3 | 77.9% | SEA -8.5 | SEA -8.5 | Under 39.5 | SEA +0.76 |
| Sun 11/08 | CIN @ ATL | CIN | CIN 27.8 – ATL 25.1 | 60.0% | CIN -3 | ATL +3 | Over 52.5 | ATL +0.33 |
| Sun 11/08 | CLE @ NO | NO | CLE 20.0 – NO 23.2 | 60.1% | NO -3.5 | CLE +3.5 | Over 42.5 | CLE +0.30 |
| Sun 11/08 | DAL @ IND | DAL | DAL 28.2 – IND 23.3 | 65.7% | DAL -4.5 | DAL -4.5 | Under 51.5 | DAL +0.41 |
| Sun 11/08 | DEN @ CAR | DEN | DEN 24.2 – CAR 21.0 | 59.0% | DEN -3 | DEN -3 | Over 44.5 | DEN +0.15 |
| Sun 11/08 | DET @ MIA | DET | DET 28.1 – MIA 21.5 | 70.5% | DET -6.5 | DET -6.5 | Over 49.5 | DET +0.07 |
| Sun 11/08 | GB @ NE | NE | GB 21.1 – NE 23.8 | 59.4% | NE -3 | GB +3 | Over 44.5 | GB +0.30 |
| Sun 11/08 | HOU @ LAC | HOU | HOU 22.9 – LAC 17.7 | 66.7% | HOU -6 | LAC +6 | Over 40.5 | LAC +0.81 |
| Sun 11/08 | LA @ WAS | LA | LA 26.6 – WAS 21.3 | 65.9% | LA -4.5 | LA -4.5 | Over 47.5 | LA +0.76 |
| Sun 11/08 | LV @ SF | SF | LV 20.4 – SF 30.6 | 79.5% | SF -10 | SF -10 | Under 51.5 | SF +0.26 |
| Sun 11/08 | NYG @ PHI | NYG | NYG 20.6 – PHI 18.0 | 58.2% | NYG -3 | PHI +3 | Over 38.5 | PHI +0.39 |
| Sun 11/08 | NYJ @ KC | KC | NYJ 19.7 – KC 28.9 | 78.2% | KC -9.5 | NYJ +9.5 | Over 48.5 | NYJ +0.30 |
| Sun 11/08 | TB @ CHI | CHI | TB 20.8 – CHI 26.8 | 69.4% | CHI -6 | CHI -6 | Over 47.5 | CHI +0.03 |
| Mon 11/09 | BUF @ MIN | BUF | BUF 24.1 – MIN 20.2 | 61.1% | BUF -3 | BUF -3 | Over 43.5 | BUF +0.87 |

*Predicted score is shown as away–home. “Model lean” is the model's margin minus the
market margin; the named side is the side the model likes relative to the line.*

The stats-only model picks the same winner in all 14 games; its projected scores are in
`outputs/week09_predictions.csv` (`pure_score`).

**Most confident picks:** SF 79.5% (LV @ SF); KC 78.2% (NYJ @ KC); SEA 77.9% (ARI @ SEA); DET 70.5% (DET @ MIA); CHI 69.4% (TB @ CHI).

**Closest picks:** NYG 58.2% (NYG @ PHI); DEN 59.0% (DEN @ CAR); NE 59.4% (GB @ NE); CIN 60.0% (CIN @ ATL); NO 60.1% (CLE @ NO).

### Key matchups and availability

*Matchup edges are standardized against the 2018–2026 training distribution. Positive σ = edge toward the home team; negative σ = edge toward the away team. These are descriptive inputs, not causal claims.*

- **ARI @ SEA** — SEA 15.1–24.3 (77.9%). Matchup edges: Turnover margin +1.72σ (home); Overall EPA -0.97σ (away). HOME QB: Drew Lock in for Sam Darnold (out); AWAY QB: Gardner Minshew in for Jacoby Brissett (out); SEA WR Jaxon Smith-Njigba questionable; ARI WR Michael Wilson questionable.
- **CIN @ ATL** — CIN 27.8–25.1 (60.0%). Matchup edges: QB vs pass defense -2.64σ (away); Pass EPA -2.61σ (away). HOME QB: Michael Penix Jr.; AWAY QB: Joe Burrow.
- **CLE @ NO** — NO 20.0–23.2 (60.1%). Matchup edges: Pass rush vs protection -1.62σ (away); WR1 vs coverage +1.51σ (home). HOME QB: Tyler Shough; AWAY QB: Shedeur Sanders in for Deshaun Watson (out); CLE WR Denzel Boston questionable; CLE TE Harold Fannin Jr. questionable; dome.
- **DAL @ IND** — DAL 28.2–23.3 (65.7%). Matchup edges: Red-zone efficiency +1.96σ (home); RB1 vs run defense -1.34σ (away). HOME QB: Daniel Jones; AWAY QB: Dak Prescott; DAL TE Jake Ferguson out.
- **DEN @ CAR** — DEN 24.2–21.0 (59.0%). Matchup edges: RB1 vs run defense +0.90σ (home); Points per drive +0.85σ (home). HOME QB: Bryce Young; AWAY QB: Bo Nix.
- **DET @ MIA** — DET 28.1–21.5 (70.5%). Matchup edges: RB1 vs run defense +1.32σ (home); Rush EPA +0.99σ (home). HOME QB: Malik Willis; AWAY QB: Jared Goff; DET WR Amon-Ra St. Brown questionable.
- **GB @ NE** — NE 21.1–23.8 (59.4%). Matchup edges: QB vs pass defense +1.39σ (home); WR corps vs coverage +1.35σ (home). HOME QB: Drake Maye; AWAY QB: Jordan Love; NE WR A.J. Brown out; NE TE Hunter Henry out.
- **HOU @ LAC** — HOU 22.9–17.7 (66.7%). Matchup edges: Pass EPA -2.45σ (away); QB vs pass defense -2.31σ (away). HOME QB: Justin Herbert; AWAY QB: C.J. Stroud; HOU RB David Montgomery out; dome.
- **LA @ WAS** — LA 26.6–21.3 (65.9%). Matchup edges: Pass rush vs protection +1.46σ (home); Red-zone efficiency +1.39σ (home). HOME QB: Jayden Daniels; AWAY QB: Matthew Stafford; WAS WR Stefon Diggs out.
- **LV @ SF** — SF 20.4–30.6 (79.5%). Matchup edges: WR1 vs coverage -1.52σ (away); Points per drive -1.41σ (away). HOME QB: Brock Purdy; AWAY QB: Kirk Cousins; SF WR Mike Evans out; SF RB Christian McCaffrey questionable.
- **NYG @ PHI** — NYG 20.6–18.0 (58.2%). Matchup edges: Turnover margin -1.30σ (away); Red-zone efficiency -1.28σ (away). HOME QB: Joe Milton III in for Tanner McKee; AWAY QB: J.J. McCarthy; PHI WR DeVonta Smith out; NYG WR Malik Nabers out; NYG RB Cam Skattebo out.
- **NYJ @ KC** — KC 19.7–28.9 (78.2%). Matchup edges: Points per drive -0.95σ (away); QB vs pass defense -0.67σ (away). HOME QB: Patrick Mahomes; AWAY QB: Geno Smith; NYJ WR Garrett Wilson out.
- **TB @ CHI** — CHI 20.8–26.8 (69.4%). Matchup edges: QB CPOE vs coverage +3.37σ (home); QB vs pass defense +2.52σ (home). HOME QB: Caleb Williams; AWAY QB: Baker Mayfield; TB RB Bucky Irving out.
- **BUF @ MIN** — BUF 24.1–20.2 (61.1%). Matchup edges: Points per drive -1.99σ (away); Pass EPA -1.89σ (away). HOME QB: Carson Wentz in for Kyler Murray (out); AWAY QB: Josh Allen; BUF WR DJ Moore questionable; BUF TE Dalton Kincaid questionable; dome.

## 8. Limitations

- **The market is the main signal.** The hybrid model's edge over the closing line is
  small and sample-dependent. The model is best used as a structured second opinion, not
  as a market-beating oracle.
- **Totals are hard.** The stats-only total model is materially worse than the market, and
  the residual total model did not beat the market in the 2026 sample. The O/U leans are
  included for completeness, not as recommendations.
- **Injury timing.** Week 9 statuses are as of Friday. `report_status` was often missing;
  the model treats “did not participate” as out/doubtful. Late-week activations and
  surprise scratches can change the picture.
- **Small 2026 sample.** 121 games is enough to sanity-check the pipeline, not enough
  to establish a durable edge.
- **Player matchups are aggregated.** WR1/RB1 are usage-based top players, not a full
  route-by-route matchup model. Pass-rush pressure is team/top-rusher level, not a full
  offensive-line vs defensive-line simulation.
- **No special-teams or coaching model** beyond basic kicking/return features and coach
  identity. Weather is handled through wind/temp/dome flags, not a full stadium model.
- **Not betting advice.** ATS/O-U results are descriptive backtest metrics, not a
  profitable strategy.

## 9. Files

- `outputs/backtest_2026_weeks1_8.csv` — per-game 2026 backtest predictions.
- `outputs/backtest_summary.csv` — pooled 2026 metrics by model.
- `outputs/backtest_per_week.csv` — 2026 week-by-week metrics.
- `outputs/backtest_calibration.csv` — win-probability calibration bins.
- `outputs/validation_2021_2025_summary.csv` — season-level validation metrics.
- `outputs/week09_predictions.csv` — final Week 9 forecasts and context.
- `outputs/week09_card.md` — the readable Week 9 card.
- `outputs/report.md` — this report.
- `src/nflscore/model.py` — model definition and feature lists.
- `src/nflscore/backtest.py` — walk-forward backtests.
- `src/nflscore/predict.py` — weekly forecast generation.
- `src/nflscore/weekly.py` — prediction card renderer.
- `src/nflscore/grade.py` — post-week grading.
- `src/nflscore/report.py` — report builder.
