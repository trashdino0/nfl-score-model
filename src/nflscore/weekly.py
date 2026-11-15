"""Render a readable weekly prediction card.

The card is the human-facing output; the CSV next to it is the machine-readable
version. Both are produced from the same ``predict_slate`` call.

Run with:
    $env:PYTHONPATH="src"; python -m nflscore.weekly
    $env:PYTHONPATH="src"; python -m nflscore.weekly --season 2026 --week 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUTPUTS, PROCESSED
from .predict import predict_slate, prediction_path, resolve_target


def _spread(margin, team: str, home: str) -> str:
    """Format a spread from one team's perspective (positive = favored)."""
    if pd.isna(margin):
        return "—"
    value = float(margin) if team == home else -float(margin)
    if value > 0:
        return f"{team} -{value:g}"
    if value < 0:
        return f"{team} +{abs(value):g}"
    return f"{team} PK"


def _game_line(margin, home: str, away: str) -> str:
    """Format the game line the usual way: favorite -X, or PK."""
    if pd.isna(margin):
        return "—"
    value = float(margin)
    if value > 0:
        return f"{home} -{value:g}"
    if value < 0:
        return f"{away} -{abs(value):g}"
    return "PK"


def _day(gameday) -> str:
    ts = pd.to_datetime(gameday, errors="coerce")
    return ts.strftime("%a %m/%d") if pd.notna(ts) else ""


def _tier(prob: float) -> str:
    if prob >= 0.75:
        return "High"
    if prob >= 0.65:
        return "Medium"
    if prob >= 0.55:
        return "Slight"
    return "Coin flip"


def _range_for(row, pick: str) -> str:
    if pick == row.home_team:
        low, high = row.pred_home_score_low, row.pred_home_score_high
    else:
        low, high = row.pred_away_score_low, row.pred_away_score_high
    return f"{int(low)}–{int(high)}"


def _context_text(row) -> str:
    text = "" if pd.isna(row.context) else str(row.context)
    if not text or text.lower() == "nan":
        return "No listed QB, injury, or weather flags."
    text = text.replace("HOME QB", f"{row.home_team} QB")
    text = text.replace("AWAY QB", f"{row.away_team} QB")
    return text


def _table(headers: list[str], rows: list[list[str]], align: list[str]) -> str:
    sep = {
        "l": ":---",
        "r": "---:",
        "c": ":---:",
    }
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(sep[a] for a in align) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_card(
    pred: pd.DataFrame,
    model,
    season: int,
    week: int,
    completed: pd.DataFrame | None = None,
) -> str:
    pred = pred.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    generated = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    has_line = ~pred["market_line_missing"].astype(bool)
    n_line = int(has_line.sum())
    home_favs = int(((pred["market_margin"] > 0) & has_line).sum())
    agree = int(
        (
            (np.sign(pred["pred_margin"]) == np.sign(pred["market_margin"])) & has_line
        ).sum()
    )

    lines = [
        f"# {season} Week {week} — NFL predictions",
        "",
        f"*Generated {generated}. Model trained on {model.train_rows_} completed games "
        f"(2018+); margin σ = {model.margin_sigma_:.2f}. "
        f"This card covers {len(pred)} unplayed games.*",
        "",
    ]

    if completed is not None and not completed.empty:
        finals = []
        for _, r in completed.iterrows():
            finals.append(
                f"{r.away_team} {int(r.away_score)} – {r.home_team} {int(r.home_score)}"
            )
        lines += [f"**Already final:** {'; '.join(finals)}.", ""]

    lines += [
        (
            f"The model agrees with the market favorite in **{agree} of {n_line}** games "
            f"with a posted line ({home_favs} home favorites). Win probabilities are "
            "blended 50/50 with the no-vig market when a line is posted, so they are "
            "deliberately close to the line."
        ),
        "",
        (
            "**Backtest through Week 8:** 2026 walk-forward — 67.5% winner accuracy "
            "(81/120 decisive games) vs 65.8% for the closing line; margin MAE 9.58 "
            "vs 9.57. The difference is within noise, not a proven edge."
        ),
        "",
        "## The card",
        "",
        _table(
            [
                "Game",
                "Pick",
                "Win prob",
                "Projected score (away–home)",
                "Pick's 80% score range",
                "Line",
                "ATS lean",
                "O/U lean",
            ],
            [
                [
                    f"{_day(r.gameday)} · {r.away_team} @ {r.home_team}",
                    f"**{r.pred_winner}**",
                    f"{r.pick_win_prob:.0%}",
                    f"{r.away_team} {int(round(r.pred_away_score))}–"
                    f"{int(round(r.pred_home_score))} {r.home_team}",
                    _range_for(r, r.pred_winner),
                    "no line"
                    if r.market_line_missing
                    else _game_line(r.market_margin, r.home_team, r.away_team),
                    "—"
                    if r.market_line_missing
                    else _spread(r.market_margin, r.ats_pick, r.home_team),
                    "—" if r.market_line_missing else f"{r.ou_pick} {r.market_total:g}",
                ]
                for r in pred.itertuples()
            ],
            ["l", "l", "r", "l", "c", "l", "l", "l"],
        ),
        "",
        "ATS leans are against the spread, not the moneyline: a team can be the "
        "pick to win and still not cover. O/U leans compare the model total to the "
        "market total.",
        "",
        "## Confidence tiers",
        "",
        _table(
            ["Tier", "Pick win prob", "Picks"],
            [
                [
                    tier,
                    bound,
                    ", ".join(
                        f"{r.pred_winner} ({r.pick_win_prob:.0%})"
                        for r in pred[
                            pred["pick_win_prob"].map(_tier) == tier
                        ].itertuples()
                    )
                    or "—",
                ]
                for tier, bound in [
                    ("High", "≥ 75%"),
                    ("Medium", "65–75%"),
                    ("Slight", "55–65%"),
                    ("Coin flip", "< 55%"),
                ]
            ],
            ["l", "l", "l"],
        ),
        "",
    ]

    lined = pred[has_line]
    edge = lined.reindex(
        lined["model_edge"].abs().sort_values(ascending=False).index
    ).head(5)

    lines += ["## Where the model differs from the market", ""]
    if edge.empty:
        lines += ["No market lines are posted yet, so there is nothing to compare.", ""]
    else:
        lines += [
            _table(
                ["Game", "Edge vs line", "Value side", "Line"],
                [
                    [
                        f"{r.away_team} @ {r.home_team}",
                        f"{abs(r.model_edge):.1f} pts",
                        r.home_team if r.model_edge > 0 else r.away_team,
                        _spread(
                            r.market_margin,
                            r.home_team if r.model_edge > 0 else r.away_team,
                            r.home_team,
                        ),
                    ]
                    for r in edge.itertuples()
                ],
                ["l", "r", "l", "l"],
            ),
            "",
            "The value side is where the model is higher than the market, in points. "
            "It is not always the same as the pick: the model can project a team to win "
            "without covering. Edges under a point are noise, not an invitation to bet.",
            "",
        ]

    lines += ["## Availability and weather notes", ""]

    for r in pred.itertuples():
        lines.append(f"- **{r.away_team} @ {r.home_team}** — {_context_text(r)}")

    lines += [
        "",
        "## How to read this card",
        "",
        "- **Pick** is the team with the higher projected score. **Win prob** is the "
        "chance that pick wins, blended 50/50 between the model and the no-vig market "
        "when a line is posted.",
        "- **Projected score** is a continuous estimate, not a final-score promise. "
        "The 80% range comes from 2021–2025 walk-forward residuals; the median "
        "team-score miss in the 2026 backtest was about 6 points.",
        "- **Line / ATS / O-U** use the market snapshot in the data. The model does "
        "not have a proven betting edge: 2026 Weeks 1–8 went 51.3% ATS and 42.1% "
        "on totals. Treat leans as diagnostics, not recommendations.",
        "- **Availability** reflects the injury report and depth chart at build time. "
        "Late scratches and line movement are not captured; rebuild if either changes.",
        "",
        f"*CSV: `{prediction_path(week).relative_to(OUTPUTS.parent).as_posix()}`*",
        "",
    ]
    return "\n".join(lines)


def write_card(season: int, week: int, pred: pd.DataFrame, model) -> Path:
    games = pd.read_parquet(PROCESSED / "game_features.parquet")
    completed = games[
        (games.season == season) & (games.week == week) & games.home_score.notna()
    ].copy()
    text = render_card(pred, model, season, week, completed)
    path = OUTPUTS / f"week{week:02d}_card.md"
    path.write_text(text, encoding="utf-8")
    return path


def main(season: int | None = None, week: int | None = None) -> None:
    season, week = resolve_target(season, week)
    pred, model = predict_slate(season, week)

    csv_path = prediction_path(week)
    pred.to_csv(csv_path, index=False)
    card_path = write_card(season, week, pred, model)

    print(f"{season} Week {week}: {len(pred)} unplayed games")
    print(f"Wrote {csv_path}")
    print(f"Wrote {card_path}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Write a weekly prediction card.")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    args = parser.parse_args()
    main(args.season, args.week)


if __name__ == "__main__":
    _cli()
