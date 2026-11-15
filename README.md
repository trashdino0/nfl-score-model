# NFL Score Model

Predicts NFL game winners and exact scores from nflverse data: team efficiency,
player performance, player-on-player matchups, injuries, weather, Elo, and
market context.

The model is built for **margin and total**, then converts those into scores:

```
home_score = total / 2 + margin / 2
away_score = total / 2 - margin / 2
```

## Headline result

2026 Weeks 1–8 walk-forward backtest (121 games, 120 decisive):

| Model | Winner accuracy | Margin MAE | Total MAE | Score MAE |
| --- | ---: | ---: | ---: | ---: |
| Market closing line | 65.8% | 9.57 | 10.66 | 7.33 |
| Pure stats ridge | 64.2% | 9.85 | 12.10 | 7.86 |
| **Hybrid market+stats** | **67.5%** | **9.58** | 10.71 | **7.33** |

The closing market is extremely efficient. The hybrid model edges it on winner
accuracy in this sample but is essentially tied on margin/score error and does
not beat the market total. The full write-up is in
[`outputs/report.md`](outputs/report.md).

## Quick start

Data is cached under `data/raw/`. To rebuild everything from cached nflverse
files:

```powershell
$env:PYTHONPATH="src"
python scripts/run_all.py
```

Or run the stages individually:

```powershell
$env:PYTHONPATH="src"
python -m nflscore.build       # team-game table from play-by-play
python -m nflscore.features    # leakage-safe weekly state + game features
python -m nflscore.backtest    # 2021-2025 validation + 2026 Weeks 1-8 replay
python -m nflscore.predict     # next unplayed week (CSV)
python -m nflscore.weekly      # next unplayed week (CSV + markdown card)
python -m nflscore.grade --week 9   # grade saved picks after finals
python -m nflscore.report      # outputs/report.md (fixed Weeks 1-8 study)
```

## One-click scripts (Windows)

If you do not want to remember commands, double-click either file in the
project root:

| File | What it does |
| :--- | :--- |
| `run_predictions.cmd` | Refreshes data, rebuilds features, predicts the next unplayed week, and writes the CSV + card (about 2–3 minutes). |
| `grade_last_week.cmd` | Refreshes results, finds the newest fully completed week with a saved prediction file, and grades it. |

Both pause when finished so the window stays open. They are thin wrappers around
`scripts/predict_now.py` and `scripts/grade_last_week.py`, which can also be run
directly.

`grade_last_week.cmd` only needs a small schedule/results refresh, not a full
rebuild, so it finishes in seconds.

## Weekly workflow (in season)

One command runs the whole weekly loop:

```powershell
$env:PYTHONPATH="src"
python scripts/weekly.py --refresh
```

It re-downloads the current season's nflverse files and schedule, rebuilds
`team_games.parquet` and `game_features.parquet`, finds the next week with
unplayed games, and writes `outputs/weekXX_predictions.csv` plus a readable
`outputs/weekXX_card.md`.

| Command | What it does |
| :--- | :--- |
| `python scripts/weekly.py --refresh` | Full refresh, rebuild, predict the next unplayed week |
| `python scripts/weekly.py --week 10` | Rebuild from cached data and predict Week 10 |
| `python scripts/weekly.py --week 10 --skip-build` | Reuse the feature table; predict and re-render only |
| `python -m nflscore.predict --week 10` | Predictions CSV only |
| `python -m nflscore.grade --week 9` | Grade saved Week 9 picks after the finals |

**When to run it.** Saturday, after the final injury reports and before the
first kickoff, is the best snapshot: the schedule file has the most complete
lines and the injury report has the latest statuses. Run again if a line moves
or a starter is ruled out. If a line is not posted yet, the card marks it
`no line` instead of pretending the model has market information.

**Do not pass `--week` unless you want to.** The default resolves to the
earliest week in the latest season that still has unplayed games, so the same
command moves from Week 9 to Week 10 automatically once Week 9 is final.

**After the games.** Grade the saved card:

```powershell
$env:PYTHONPATH="src"
python -m nflscore.grade --week 9
```

This prints winner accuracy, score/margin/total MAE, and ATS/O-U records, and
writes `outputs/week09_graded.csv`. Grading uses the market line stored in the
prediction file, so it is a true pre-game evaluation, not a line-shopping
exercise after the fact.

To re-download nflverse files without running the model:

```powershell
python scripts/download_all.py                  # fill in anything missing
python scripts/download_all.py --refresh        # refresh 2026 + schedule
python scripts/download_all.py --refresh --season 2026
```

## Outputs

- `outputs/report.md` — the fixed methodology/validation/backtest study
  (2021–2025 validation plus 2026 Weeks 1–8), not regenerated each week.
- `outputs/weekXX_card.md` — the weekly card: picks, projected scores, win
  probabilities, 80% score ranges, market leans, and availability notes.
- `outputs/weekXX_predictions.csv` — the same slate in machine-readable form,
  with pure-stats projections and feature drivers.
- `outputs/weekXX_graded.csv` — post-week grading, created by
  `python -m nflscore.grade --week N`.
- `outputs/backtest_2026_weeks1_8.csv` — per-game 2026 backtest predictions.
- `outputs/backtest_summary.csv` — pooled metrics by model.
- `outputs/backtest_per_week.csv` — week-by-week accuracy/error.
- `outputs/backtest_calibration.csv` — win-probability calibration.
- `outputs/validation_2021_2025_summary.csv` — season-level validation.

## Model in one paragraph

The primary model is a **market-anchored residual ridge**. It predicts
`actual_margin - closing_margin` and `actual_total - closing_total`, then shrinks
those residuals (20% for margin, 10% for total) before adding them back to the
closing line. A pure stats ridge predicts margin and total directly as a
comparison. Win probability comes from a normal approximation to the predicted
margin (`σ ≈ 12.7`), blended 50/50 with the no-vig market probability. Feature
families include EPA/pass/rush splits, success rate, pace, points/drive,
red-zone efficiency, QB availability and quality, WR1/RB1/pass-rush matchups,
injuries, rest/weather, Elo, and the closing line.

## Notes

- Regular season only. The model filters to `game_type == "REG"` and does not
  handle playoff teams, seeding, or playoff rest.
- Training starts in 2018 because coverage/pressure data begins there; 2014–2017
  remain available as prior-season priors.
- The cached nflverse `spread_line` is **positive when the home team is
  favored** (verified against moneylines). `market_margin = spread_line`.
- Availability and lines are only as fresh as the last `--refresh`; late injury
  news and line movement can change the picture.
- At the start of a new season, add the year to `SEASONS` in
  `src/nflscore/config.py` and pass `--season <year>` once to the refresh.
- This is a model/backtest project, not betting advice.
