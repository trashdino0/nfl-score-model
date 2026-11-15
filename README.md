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

2026 Weeks 1–9 walk-forward backtest (136 games, 135 decisive):

| Model | Winner accuracy | Margin MAE | Total MAE | Score MAE |
| --- | ---: | ---: | ---: | ---: |
| Market closing line | 68.9% | 9.49 | 10.60 | 7.26 |
| Pure stats ridge | 67.4% | 9.73 | 12.07 | 7.78 |
| **Hybrid market+stats** | **70.4%** | 9.49 | 10.60 | **7.25** |

The closing market is extremely efficient. The hybrid model's winner edge is a
two-game difference (McNemar p ≈ 0.5), and its margin/total/Brier intervals all
span zero. **It is not a demonstrated betting edge.** The paired evaluation —
bootstrap intervals, McNemar, Diebold–Mariano, ATS/over ROI at −110, and
calibration bins — is in [`outputs/evaluation.md`](outputs/evaluation.md).
[`outputs/report.md`](outputs/report.md) is the earlier narrative study and
reflects the pre-Tier-1 model.

For the method — features, estimator, cross-validation, and the diagnostics
behind every design choice — read [`MODEL.md`](MODEL.md).

## Version deltas

Every change is measured against the previous tag on **identical games**. The
harness checks out the old commit into a temporary worktree, runs that code on
the current walk-forward windows, and reports paired deltas:

```powershell
$env:PYTHONPATH="src"; python scripts/version_delta.py --base <git-ref>
```

- [`outputs/version_delta.md`](outputs/version_delta.md) — the latest before/after
- `outputs/version_history.csv` — one row per metric per comparison, appended

Training windows are matched on both sides, so the delta isolates model changes
from harness changes. See §1 of `version_delta.md` for what a delta does *not*
prove.


## Quick start

Install the dependencies once (the project uses a local venv in this repo):

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

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
python -m nflscore.backtest    # 2021-2025 validation + current-season replay
python scripts/evaluate.py     # paired CIs, McNemar, DM, ATS ROI, calibration
python -m nflscore.predict     # next unplayed week (CSV)
python -m nflscore.weekly      # next unplayed week (CSV + markdown card)
python -m nflscore.grade --week 9   # grade saved picks after finals
python -m nflscore.report      # legacy fixed study (pre-Tier-1 model)
```

Run the evaluation/unit tests:

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest tests -q
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

- `outputs/evaluation.md` / `outputs/evaluation.json` — the current paired
  evaluation: bootstrap CIs, McNemar, Diebold–Mariano, ATS/over ROI, and
  calibration for the blend, model, and market probabilities.
- `outputs/report.md` — the earlier fixed methodology/validation/backtest
  narrative. It describes the pre-Tier-1 model; use `evaluation.md` for current
  numbers.
- `outputs/weekXX_card.md` — the weekly card: picks, projected scores, model
  and market win probabilities, bootstrap score ranges, cover probabilities,
  and availability notes.
- `outputs/weekXX_predictions.csv` — the same slate in machine-readable form,
  with pure-stats projections, `line_source`, and feature drivers.
- `outputs/weekXX_graded.csv` — post-week grading, created by
  `python -m nflscore.grade --week N`.
- `outputs/backtest_2026.csv` — per-game current-season backtest predictions.
- `outputs/backtest_2026_weeks1_8.csv` — the Weeks 1–8 slice of the current
  backtest, kept for comparison with earlier write-ups.
- `outputs/backtest_summary.csv` — pooled metrics by model.
- `outputs/backtest_per_week.csv` — week-by-week accuracy/error.
- `outputs/backtest_calibration.csv` — blend calibration bins.
- `outputs/backtest_calibration_all.csv` — blend/model/market calibration.
- `outputs/validation_2021_2025_summary.csv` — season-level validation.

## Model in one paragraph

The primary model is a **market-anchored residual ridge**. It predicts
`actual_margin - closing_margin` and `actual_total - closing_total`, then shrinks
those residuals (20% for margin, **0% for total**) before adding them back to the
closing line. The total shrink is zero because the total model has not beaten the
closing total out of sample; the projected total is the market total, and no
over/under pick is published. A pure stats ridge predicts margin and total
directly as a comparison and as the fallback when no line is posted. Win
probability comes from a normal approximation using the **out-of-fold** error
scale, blended with the no-vig market in log-odds space at a weight fitted on
out-of-fold log loss. Score ranges and cover/over probabilities come from a
**joint residual bootstrap** that resamples `(margin, total)` error pairs from
out-of-fold predictions, so the distribution is coherent and conditional on the
game. Feature families include EPA/pass/rush splits, success rate, pace,
points/drive, red-zone efficiency, QB availability and quality, WR1/RB1/pass-rush
matchups, injuries, rest/weather, Elo, and the closing line.

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
