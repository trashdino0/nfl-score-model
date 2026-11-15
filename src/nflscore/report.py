"""Build the markdown report from the saved backtest and prediction outputs.

Run with:
    $env:PYTHONPATH="src"; python -m nflscore.report
"""

from __future__ import annotations

import pandas as pd

from .config import OUTPUTS, PROCESSED
from .model import NFLModel

NETS = {
    "epa_net": "Overall EPA",
    "pass_epa_net": "Pass EPA",
    "rush_epa_net": "Rush EPA",
    "qb_vs_pass_net": "QB vs pass defense",
    "qb_cpoe_vs_coverage_net": "QB CPOE vs coverage",
    "wr1_vs_coverage_net": "WR1 vs coverage",
    "wr_corps_vs_coverage_net": "WR corps vs coverage",
    "rb1_vs_run_net": "RB1 vs run defense",
    "pass_rush_vs_protection_net": "Pass rush vs protection",
    "turnovers_net": "Turnover margin",
    "red_zone_net": "Red-zone efficiency",
    "points_drive_net": "Points per drive",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def pct(x, nd=1):
    return "—" if pd.isna(x) else f"{x * 100:.{nd}f}%"


def num(x, nd=2):
    return "—" if pd.isna(x) else f"{x:.{nd}f}"


def signed(x, nd=2):
    return "—" if pd.isna(x) else f"{x:+.{nd}f}"


def md_table(headers, rows, aligns=None):
    if aligns is None:
        aligns = ["---"] * len(headers)
    align_map = {"l": ":---", "c": ":---:", "r": "---:"}
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(align_map.get(a, a) for a in aligns) + " |"
    body = []
    for row in rows:
        cells = []
        for cell in row:
            text = "" if cell is None else str(cell)
            text = text.replace("|", "\\|")
            cells.append(text)
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


def market_line(row, team=None):
    m = row["market_margin"]
    if pd.isna(m) or m == 0:
        return "Pick'em"
    if team is None:
        team = row["home_team"] if m > 0 else row["away_team"]
    if team == row["home_team"]:
        sign = "-" if m > 0 else "+"
        return f"{team} {sign}{abs(m):g}"
    sign = "-" if m < 0 else "+"
    return f"{team} {sign}{abs(m):g}"


def pick_line(row):
    return market_line(row, row["ats_pick"])


def edge_lean(row):
    e = row["model_edge"]
    if pd.isna(e) or abs(e) < 0.005:
        return "level"
    side = row["home_team"] if e > 0 else row["away_team"]
    return f"{side} +{abs(e):.2f}"


def predicted_score(row):
    return (
        f"{row['away_team']} {row['pred_away_score']:.1f} – "
        f"{row['home_team']} {row['pred_home_score']:.1f}"
    )


def score_accuracy(df: pd.DataFrame, model: str = "hybrid") -> dict:
    """Exact-score and score-error rates using rounded score predictions."""
    d = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    if model == "market":
        ph = ((d["market_total"] + d["market_margin"]) / 2).round().astype(int)
        pa = ((d["market_total"] - d["market_margin"]) / 2).round().astype(int)
    else:
        ph = d["pred_home_score"].round().astype(int)
        pa = d["pred_away_score"].round().astype(int)
    ah = d["home_score"].astype(int)
    aa = d["away_score"].astype(int)
    eh = (ph - ah).abs()
    ea = (pa - aa).abs()
    err = pd.concat([eh, ea], ignore_index=True)
    both3 = (eh <= 3) & (ea <= 3)
    both7 = (eh <= 7) & (ea <= 7)
    return {
        "n": len(d),
        "exact_both": int(((eh == 0) & (ea == 0)).sum()),
        "exact_either": int(((eh == 0) | (ea == 0)).sum()),
        "exact_margin": int((((ph - pa) - (ah - aa)).abs() == 0).sum()),
        "exact_total": int((((ph + pa) - (ah + aa)).abs() == 0).sum()),
        "both3": int(both3.sum()),
        "both7": int(both7.sum()),
        "team3": float((err <= 3).mean()),
        "team7": float((err <= 7).mean()),
        "median": float(err.median()),
        "p90": float(err.quantile(0.90)),
        "mae": float(err.mean()),
    }


def top_edges(row, stats, n=2):
    items = []
    for col, label in NETS.items():
        if col not in row.index or col not in stats:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        mu, sd = stats[col]
        if not sd or sd <= 0:
            continue
        z = (val - mu) / sd
        items.append((abs(z), label, z))
    items.sort(reverse=True)
    parts = []
    for _, label, z in items[:n]:
        side = "home" if z > 0 else "away"
        parts.append(f"{label} {z:+.2f}σ ({side})")
    return "; ".join(parts) if parts else "no clear edge"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report() -> str:
    games = pd.read_parquet(PROCESSED / "game_features.parquet")
    bt = pd.read_csv(OUTPUTS / "backtest_2026_weeks1_8.csv")
    summary = pd.read_csv(OUTPUTS / "backtest_summary.csv")
    week_table = pd.read_csv(OUTPUTS / "backtest_per_week.csv")
    cal = pd.read_csv(OUTPUTS / "backtest_calibration.csv")
    val = pd.read_csv(OUTPUTS / "validation_2021_2025_summary.csv")
    w9 = pd.read_csv(OUTPUTS / "week09_predictions.csv")
    raw_nets = games[games.game_id.isin(w9.game_id)][["game_id"] + list(NETS)]
    w9 = w9.merge(raw_nets, on="game_id", how="left")

    train = games[(games.season >= 2018) & games.home_score.notna()].copy()
    model = NFLModel().fit(train)
    stats = {}
    for col in NETS:
        if col in train.columns:
            stats[col] = (train[col].mean(), train[col].std())

    n_games = len(bt)
    n_decisive = int(bt["home_win"].notna().sum())
    decisive = bt["home_win"].notna()
    hybrid_wins = int(
        (((bt.loc[decisive, "pred_margin"] > 0).astype(float)) == bt.loc[decisive, "home_win"]).sum()
    )
    market_wins = int(
        (((bt.loc[decisive, "market_margin"] > 0).astype(float)) == bt.loc[decisive, "home_win"]).sum()
    )

    # Validation averages
    val = val.copy()
    val["base"] = val["model"].str.replace(r"^\d{4} ", "", regex=True)
    val_avg = (
        val.groupby("base")[
            ["win_acc", "margin_mae", "total_mae", "score_mae", "brier", "logloss"]
        ]
        .mean()
        .reset_index()
    )

    # 2026 summary by model label
    s = summary.set_index("model")

    def srow(label):
        r = s.loc[label]
        ats = (
            "—"
            if pd.isna(r["ats_rate"])
            else f"{pct(r['ats_rate'])} ({int(r['ats_n'])})"
        )
        ou = "—" if pd.isna(r["ou_rate"]) else f"{pct(r['ou_rate'])} ({int(r['ou_n'])})"
        return [
            label,
            f"{pct(r['win_acc'])} ({round(r['win_acc'] * n_decisive)}/{n_decisive})",
            num(r["margin_mae"]),
            num(r["total_mae"]),
            num(r["score_mae"]),
            num(r["brier"], 3),
            num(r["logloss"], 3),
            ats,
            ou,
        ]

    report_rows = [
        srow("Market closing line"),
        srow("Naive scoring baseline"),
        srow("Pure stats ridge"),
        srow("Hybrid market+stats"),
    ]

    val_rows = []
    for label, name in [
        ("market", "Market closing line"),
        ("naive", "Naive scoring baseline"),
        ("pure_stats", "Pure stats ridge"),
        ("hybrid", "Hybrid market+stats"),
    ]:
        r = val_avg[val_avg["base"] == label].iloc[0]
        val_rows.append(
            [
                name,
                pct(r["win_acc"]),
                num(r["margin_mae"]),
                num(r["total_mae"]),
                num(r["score_mae"]),
                num(r["brier"], 3),
                num(r["logloss"], 3),
            ]
        )

    week_rows = []
    for _, r in week_table.iterrows():
        week_rows.append(
            [
                int(r["week"]),
                int(r["n"]),
                pct(r["market_win_acc"]),
                pct(r["hybrid_win_acc"]),
                pct(r["pure_win_acc"]),
                num(r["market_margin_mae"]),
                num(r["hybrid_margin_mae"]),
                num(r["hybrid_total_mae"]),
            ]
        )

    cal_rows = []
    for _, r in cal.iterrows():
        cal_rows.append([r["bin"], int(r["n"]), pct(r["predicted"]), pct(r["actual"])])

    sa_h = score_accuracy(bt, "hybrid")
    sa_m = score_accuracy(bt, "market")
    score_rows = [
        [
            "Exact score, both teams",
            f"{sa_h['exact_both']} ({pct(sa_h['exact_both'] / sa_h['n'])})",
            f"{sa_m['exact_both']} ({pct(sa_m['exact_both'] / sa_m['n'])})",
        ],
        [
            "Exact score, at least one team",
            f"{sa_h['exact_either']} ({pct(sa_h['exact_either'] / sa_h['n'])})",
            f"{sa_m['exact_either']} ({pct(sa_m['exact_either'] / sa_m['n'])})",
        ],
        [
            "Exact margin",
            f"{sa_h['exact_margin']} ({pct(sa_h['exact_margin'] / sa_h['n'])})",
            f"{sa_m['exact_margin']} ({pct(sa_m['exact_margin'] / sa_m['n'])})",
        ],
        [
            "Exact total",
            f"{sa_h['exact_total']} ({pct(sa_h['exact_total'] / sa_h['n'])})",
            f"{sa_m['exact_total']} ({pct(sa_m['exact_total'] / sa_m['n'])})",
        ],
        [
            "Both scores within 3 points",
            f"{sa_h['both3']} ({pct(sa_h['both3'] / sa_h['n'])})",
            f"{sa_m['both3']} ({pct(sa_m['both3'] / sa_m['n'])})",
        ],
        [
            "Both scores within 7 points",
            f"{sa_h['both7']} ({pct(sa_h['both7'] / sa_h['n'])})",
            f"{sa_m['both7']} ({pct(sa_m['both7'] / sa_m['n'])})",
        ],
        ["Per-team score within 3 points", pct(sa_h["team3"]), pct(sa_m["team3"])],
        ["Per-team score within 7 points", pct(sa_h["team7"]), pct(sa_m["team7"])],
        [
            "Median team-score error",
            f"{sa_h['median']:.1f} pts",
            f"{sa_m['median']:.1f} pts",
        ],
        [
            "90th percentile team-score error",
            f"{sa_h['p90']:.1f} pts",
            f"{sa_m['p90']:.1f} pts",
        ],
        ["Score MAE (rounded)", f"{sa_h['mae']:.2f}", f"{sa_m['mae']:.2f}"],
    ]

    # Week 9 main table
    w9_rows = []
    for _, r in w9.iterrows():
        w9_rows.append(
            [
                pd.to_datetime(r["gameday"]).strftime("%a %m/%d"),
                f"{r['away_team']} @ {r['home_team']}",
                r["pred_winner"],
                predicted_score(r),
                pct(r["pick_win_prob"]),
                market_line(r),
                pick_line(r),
                f"{r['ou_pick']} {r['market_total']:g}",
                edge_lean(r),
            ]
        )

    # Week 9 notes
    note_lines = []
    for _, r in w9.iterrows():
        edges = top_edges(r, stats, n=2)
        context = r.get("context", "")
        if pd.isna(context):
            context = ""
        note_lines.append(
            f"- **{r['away_team']} @ {r['home_team']}** — {r['pred_winner']} "
            f"{r['pred_away_score']:.1f}–{r['pred_home_score']:.1f} "
            f"({pct(r['pick_win_prob'])}). Matchup edges: {edges}. "
            f"{context}."
        )

    # Most/least confident
    w9_conf = w9.copy()
    w9_conf["pick_conf"] = w9_conf["pick_win_prob"]
    most = w9_conf.sort_values("pick_conf", ascending=False).head(5)
    least = w9_conf.sort_values("pick_conf").head(5)

    def conf_list(df):
        return "; ".join(
            f"{r['pred_winner']} {pct(r['pick_conf'])} ({r['away_team']} @ {r['home_team']})"
            for _, r in df.iterrows()
        )

    # Biggest model-vs-market leans
    leans = w9.reindex(w9["model_edge"].abs().sort_values(ascending=False).index).head(
        5
    )
    lean_list = "; ".join(
        f"{pick_line(r)} ({edge_lean(r)})" for _, r in leans.iterrows()
    )

    report = f"""# NFL Score Model — 2026 Backtest and Week 9 Forecast

> **Legacy report (pre-Tier-1).** This document describes the model before the
> Tier 1 evaluation, fallback, and distribution changes. For current numbers,
> see `outputs/evaluation.md`. The model now sets `total_shrink = 0`, labels
> no-line games as `pure_fallback`, and uses a joint residual bootstrap instead
> of fixed score intervals.

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

**2026 Weeks 1–8 backtest ({n_games} games, {n_decisive} decisive):**

- **Hybrid model: {pct(s.loc["Hybrid market+stats", "win_acc"])} winner accuracy**
  ({hybrid_wins}/{n_decisive}), margin MAE
  {num(s.loc["Hybrid market+stats", "margin_mae"])}, total MAE
  {num(s.loc["Hybrid market+stats", "total_mae"])}, score MAE
  {num(s.loc["Hybrid market+stats", "score_mae"])}.
- **Market closing line: {pct(s.loc["Market closing line", "win_acc"])}**
  ({market_wins}/{n_decisive}), margin MAE
  {num(s.loc["Market closing line", "margin_mae"])}, total MAE
  {num(s.loc["Market closing line", "total_mae"])}, score MAE
  {num(s.loc["Market closing line", "score_mae"])}.
- **Pure stats model: {pct(s.loc["Pure stats ridge", "win_acc"])}**, margin MAE
  {num(s.loc["Pure stats ridge", "margin_mae"])}, total MAE
  {num(s.loc["Pure stats ridge", "total_mae"])}.

The result is deliberately sobering: the closing market is extremely efficient. The
hybrid model edges it slightly on winner accuracy in this sample (by about two games) and
is essentially tied on margin and score error. It does **not** beat the market total; the
total adjustment is kept very small for that reason. The stats-only model is respectable
but clearly weaker than the market.

**Week 9:** the model forecasts the 14 remaining games. It agrees with the market favorite
in every game, so the useful output is the projected score, win probability, and the size
of the model's lean against the line — not a set of contrarian winners. The largest leans
are {lean_list}.

## 2. Data and feature construction

- **Source:** nflverse GitHub release data only (no API keys).
- **Seasons:** 2014–2026 downloaded; the main model trains from **2018** because
  coverage/pressure data begins there. 2014–2017 are retained as prior-season priors.
- **Feature table:** {len(games):,} game rows and {games.shape[1]} columns.
- **Training pool:** {model.train_rows_:,} completed games from 2018–2026 (through the
  Thursday Week 9 game).
- **Feature counts:** {len(model.margin_features_)} margin features,
  {len(model.total_features_)} total features, {len(model.pure_features_)} stats-only
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

with σ = {model.margin_sigma_:.2f}, estimated from the historical distribution of
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

{
        md_table(
            [
                "Model",
                "Winner acc.",
                "Margin MAE",
                "Total MAE",
                "Score MAE",
                "Brier",
                "Log loss",
            ],
            val_rows,
            ["l", "r", "r", "r", "r", "r", "r"],
        )
    }

The hybrid model is a hair better than the market on every average error metric, but the
gaps are tiny (0.01–0.02 points/percentage points). This is the expected size of a real
edge in a highly efficient market; it is not a guarantee that the edge persists.

## 6. Backtest: 2026 Weeks 1–8

{
        md_table(
            [
                "Model",
                "Winner acc.",
                "Margin MAE",
                "Total MAE",
                "Score MAE",
                "Brier",
                "Log loss",
                "ATS",
                "O/U",
            ],
            report_rows,
            ["l", "r", "r", "r", "r", "r", "r", "r", "r"],
        )
    }

**Reading the table:**

- The hybrid model wins {
        pct(s.loc["Hybrid market+stats", "win_acc"])
    } of decisive games,
  versus {pct(s.loc["Market closing line", "win_acc"])} for the closing line. That is a
  two-game difference out of {
        n_decisive
    }; treat it as noise, not proof of a persistent edge.
- Margin MAE is essentially tied ({num(s.loc["Hybrid market+stats", "margin_mae"])} vs
  {num(s.loc["Market closing line", "margin_mae"])}).
- The total model is slightly worse than simply using the market total
  ({num(s.loc["Hybrid market+stats", "total_mae"])} vs
  {
        num(s.loc["Market closing line", "total_mae"])
    }). This is why the total shrink is only
  10%.
- The model's ATS rate is {
        pct(s.loc["Hybrid market+stats", "ats_rate"])
    } and its O/U rate
  is {pct(s.loc["Hybrid market+stats", "ou_rate"])}. Neither is a betting edge; the O/U
  result is below a coin flip. Do not use the O/U leans as recommendations.
- The pure stats model is clearly weaker on totals
  ({num(s.loc["Pure stats ridge", "total_mae"])} MAE) and slightly weaker on winners
  ({
        pct(s.loc["Pure stats ridge", "win_acc"])
    }), confirming that the market anchor is doing
  real work.

### Per-week results

{
        md_table(
            [
                "Week",
                "Games",
                "Market acc.",
                "Hybrid acc.",
                "Pure acc.",
                "Market margin MAE",
                "Hybrid margin MAE",
                "Hybrid total MAE",
            ],
            week_rows,
            ["c", "r", "r", "r", "r", "r", "r", "r"],
        )
    }

### How accurate are the exact scores?

Winner accuracy and exact-score accuracy are very different questions. The model predicts
a continuous score, but an NFL final score is a narrow integer target with a wide
distribution. Rounding the projections gives:

{
        md_table(
            ["Score metric", "Hybrid model", "Market baseline"],
            score_rows,
            ["l", "r", "r"],
        )
    }

The model never hits both exact scores in this 121-game sample, and the market doesn't
either. Hitting one team's exact score happens about 8% of the time for both. The typical
team-score error is about 7 points, with a 90th percentile around 15 points. In other words,
the score projection is a **distributional estimate** — useful for the implied game script
and for ranges — not a precise final-score forecast. The hybrid and market score errors are
effectively identical.

### Calibration

{
        md_table(
            ["Predicted win-prob. bin", "Games", "Predicted", "Actual"],
            cal_rows,
            ["l", "r", "r", "r"],
        )
    }

Calibration is reasonable in the middle bins. The 0.2–0.4 bin overpredicts the home team
(34.0% predicted vs 23.8% actual), and the 0.8–1.0 bin has only five games, so its 60%
actual rate is noisy. The model is not systematically overconfident in the extreme
favorites; the small samples simply make the bins unstable.

## 7. Week 9 forecasts

The Thursday game is already final: **JAX 33, BAL 10**. The model was retrained through that
game and forecasts the 14 remaining Sunday/Monday games.

{
        md_table(
            [
                "Date",
                "Matchup",
                "Pick",
                "Predicted score",
                "Pick win prob.",
                "Market line",
                "ATS lean",
                "O/U lean",
                "Model lean",
            ],
            w9_rows,
            ["l", "l", "l", "l", "r", "l", "l", "l", "l"],
        )
    }

*Predicted score is shown as away–home. “Model lean” is the model's margin minus the
market margin; the named side is the side the model likes relative to the line.*

The stats-only model picks the same winner in all 14 games; its projected scores are in
`outputs/week09_predictions.csv` (`pure_score`).

**Most confident picks:** {conf_list(most)}.

**Closest picks:** {conf_list(least)}.

### Key matchups and availability

*Matchup edges are standardized against the 2018–2026 training distribution. Positive σ = edge toward the home team; negative σ = edge toward the away team. These are descriptive inputs, not causal claims.*

{chr(10).join(note_lines)}

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
- **Small 2026 sample.** {
        n_games
    } games is enough to sanity-check the pipeline, not enough
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
"""

    return report


def main() -> None:
    text = build_report()
    path = OUTPUTS / "report.md"
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path} ({len(text):,} chars)")


if __name__ == "__main__":
    main()
