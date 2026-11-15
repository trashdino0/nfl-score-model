"""Leakage-safe weekly state and game-level feature construction."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import PROCESSED, SEASONS
from . import data as nfl
from .build import _status_score, build_injury_features, build_qb_injury_flags

# Metrics that get a 4-week rolling average (as of before the target week).
ROLL_METRICS = [
    # offense
    "points_for",
    "off_epa_play",
    "off_pass_epa_db",
    "off_rush_epa_carry",
    "off_success_rate",
    "off_pass_success",
    "off_rush_success",
    "off_pass_rate",
    "off_early_down_pass_rate",
    "off_explosive_pass_rate",
    "off_explosive_rush_rate",
    "off_points_per_drive",
    "off_plays_per_drive",
    "off_seconds_per_play",
    "off_red_zone_td_rate",
    "off_red_zone_trips",
    "off_third_down_rate",
    "off_turnover_rate",
    "off_sack_rate",
    "off_qb_hit_rate",
    "off_penalty_yards_play",
    "off_pass_yards_play",
    "off_rush_yards_play",
    "off_time_of_possession",
    # defense
    "points_allowed",
    "def_epa_play",
    "def_pass_epa_db",
    "def_rush_epa_carry",
    "def_success_rate",
    "def_pass_success",
    "def_rush_success",
    "def_explosive_pass_rate",
    "def_explosive_rush_rate",
    "def_points_per_drive",
    "def_red_zone_td_rate",
    "def_third_down_rate",
    "def_takeaway_rate",
    "def_sack_rate",
    "def_qb_hit_rate",
    "def_pass_yards_play",
    "def_rush_yards_play",
    "def_plays_per_drive",
    "def_seconds_per_play",
    # special teams / turnovers
    "fg_pct",
    "punt_net_yards_per_punt",
    "punt_return_yards_per_return",
    "kickoff_return_yards_per_return",
    "turnover_margin",
    # QB team aggregate
    "qb_epa_db",
    "qb_cpoe",
    "qb_int_rate",
    "qb_sack_rate",
    "qb_rush_epa_carry",
    "qb_attempts",
    # skill players
    "wr1_target_share",
    "wr1_epa_target",
    "wr1_yards_target",
    "wr1_air_yards_share",
    "wr1_wopr",
    "wr_corps_epa_target",
    "wr_corps_yards_target",
    "wr_corps_target_share",
    "te1_epa_target",
    "te1_target_share",
    "rb1_rush_epa_carry",
    "rb1_yards_carry",
    "rb1_target_share",
    "rb1_epa_target",
    "rb_room_rush_epa_carry",
    "rb_room_yards_carry",
    # coverage / pressure
    "coverage_yards_per_target",
    "coverage_completion_pct",
    "coverage_passer_rating",
    "coverage_adot",
    "cb1_yards_target",
    "cb1_passer_rating",
    "cb1_completion_pct",
    "cov1_yards_target",
    "cov1_passer_rating",
    "prush1_pressures",
    "prush1_sacks",
    "prush1_hurries",
    "adv_pressures",
    "adv_hurries",
    "adv_qb_hits",
    "adv_sacks",
    "adv_blitzes",
    "pressure_rate",
    "pressure_rate_allowed",
    # usage / pace
    "team_off_snaps",
    "team_def_snaps",
]

ROLL8_METRICS = [
    "off_epa_play",
    "def_epa_play",
    "off_pass_epa_db",
    "def_pass_epa_db",
    "points_for",
    "points_allowed",
    "qb_epa_db",
    "coverage_passer_rating",
    "pressure_rate",
    "off_points_per_drive",
    "def_points_per_drive",
]

PREV_METRICS = [
    "off_epa_play",
    "def_epa_play",
    "off_pass_epa_db",
    "def_pass_epa_db",
    "points_for",
    "points_allowed",
    "qb_epa_db",
    "coverage_passer_rating",
    "pressure_rate",
    "off_points_per_drive",
    "def_points_per_drive",
]

# (offensive metric, defensive metric, label)
MATCHUP_PAIRS = [
    ("off_epa_play", "def_epa_play", "epa"),
    ("off_pass_epa_db", "def_pass_epa_db", "pass_epa"),
    ("off_rush_epa_carry", "def_rush_epa_carry", "rush_epa"),
    ("off_success_rate", "def_success_rate", "success"),
    ("off_pass_success", "def_pass_success", "pass_success"),
    ("off_rush_success", "def_rush_success", "rush_success"),
    ("off_explosive_pass_rate", "def_explosive_pass_rate", "explosive_pass"),
    ("off_explosive_rush_rate", "def_explosive_rush_rate", "explosive_rush"),
    ("off_points_per_drive", "def_points_per_drive", "points_drive"),
    ("off_red_zone_td_rate", "def_red_zone_td_rate", "red_zone"),
    ("off_third_down_rate", "def_third_down_rate", "third_down"),
    ("off_turnover_rate", "def_takeaway_rate", "turnovers"),
    ("off_sack_rate", "def_sack_rate", "sacks"),
    ("off_qb_hit_rate", "def_qb_hit_rate", "qb_hits"),
    ("off_pass_yards_play", "def_pass_yards_play", "pass_yards"),
    ("off_rush_yards_play", "def_rush_yards_play", "rush_yards"),
    ("off_plays_per_drive", "def_plays_per_drive", "plays_drive"),
    ("off_seconds_per_play", "def_seconds_per_play", "pace"),
    ("qb_epa_db", "def_pass_epa_db", "qb_vs_pass"),
    ("qb_cpoe", "coverage_passer_rating", "qb_cpoe_vs_coverage"),
    ("qb_rush_epa_carry", "def_rush_epa_carry", "qb_rush"),
    ("wr1_epa_target", "coverage_yards_per_target", "wr1_vs_coverage"),
    ("wr_corps_epa_target", "coverage_yards_per_target", "wr_corps_vs_coverage"),
    ("rb1_rush_epa_carry", "def_rush_epa_carry", "rb1_vs_run"),
    ("rb1_yards_carry", "def_rush_yards_play", "rb1_yards_vs_run"),
    ("pressure_rate", "pressure_rate_allowed", "pass_rush_vs_protection"),
]

DIFF_METRICS = [
    "points_for",
    "points_allowed",
    "off_epa_play",
    "def_epa_play",
    "qb_epa_db",
    "qb_cpoe",
    "qb_int_rate",
    "qb_sack_rate",
    "wr1_epa_target",
    "wr_corps_epa_target",
    "rb1_rush_epa_carry",
    "coverage_yards_per_target",
    "coverage_passer_rating",
    "pressure_rate",
    "turnover_margin",
    "fg_pct",
    "punt_net_yards_per_punt",
]

SUM_METRICS = [
    "off_plays_per_drive",
    "off_seconds_per_play",
    "off_pass_rate",
    "team_off_snaps",
    "team_def_snaps",
    "off_time_of_possession",
]

INJURY_COLS = [
    "inj_lost_off",
    "inj_lost_def",
    "inj_lost_qb",
    "inj_lost_rb",
    "inj_lost_wr",
    "inj_lost_te",
    "inj_lost_ol",
    "inj_lost_dl",
    "inj_lost_lb",
    "inj_lost_db",
    "inj_out_count",
    "inj_doubtful_count",
    "inj_questionable_count",
]


# ---------------------------------------------------------------------------
# Team weekly state
# ---------------------------------------------------------------------------
def build_team_week_state(
    team_games: pd.DataFrame, seasons: Iterable[int] = SEASONS
) -> pd.DataFrame:
    metrics = [m for m in ROLL_METRICS if m in team_games.columns]
    tg = team_games[["season", "team", "week"] + metrics].copy()
    tg = tg.groupby(["season", "team", "week"], as_index=False)[metrics].mean()
    tg = tg.sort_values(["season", "team", "week"]).reset_index(drop=True)

    # Rolling windows are over *games*, not calendar weeks. A bye must not
    # shorten the window or pull in an older game.
    g = tg.groupby(["season", "team"], sort=False)
    game_state = tg[["season", "team", "week"]].copy()
    game_state["has_game"] = 1.0
    for m in metrics:
        game_state[f"{m}_before4"] = g[m].transform(
            lambda s: s.shift(1).rolling(4, min_periods=1).mean()
        )
        game_state[f"{m}_after4"] = g[m].transform(
            lambda s: s.rolling(4, min_periods=1).mean()
        )
        if m in ROLL8_METRICS:
            game_state[f"{m}_before8"] = g[m].transform(
                lambda s: s.shift(1).rolling(8, min_periods=1).mean()
            )
            game_state[f"{m}_after8"] = g[m].transform(
                lambda s: s.rolling(8, min_periods=1).mean()
            )

    # Complete team-week grid (includes byes and future weeks).
    teams = tg[["season", "team"]].drop_duplicates()
    grid = []
    for row in teams.itertuples(index=False):
        for week in range(1, 19):
            grid.append((row.season, row.team, week))
    grid = pd.DataFrame(grid, columns=["season", "team", "week"])
    state = grid.merge(game_state, on=["season", "team", "week"], how="left")
    state = state.sort_values(["season", "team", "week"]).reset_index(drop=True)

    # Prior-season means, known before the season starts.
    season_mean = tg.groupby(["season", "team"])[metrics].mean().reset_index()
    prev = season_mean.copy()
    prev["season"] = prev["season"] + 1
    prev = prev.rename(columns={m: f"{m}_prev" for m in metrics})
    state = state.merge(prev, on=["season", "team"], how="left")

    has_game = state["has_game"].fillna(0.0) == 1.0
    for m in metrics:
        # For a bye/future week, carry forward the state after the last game.
        state[f"{m}_after4"] = state.groupby(["season", "team"], sort=False)[
            f"{m}_after4"
        ].ffill()
        state[f"{m}_roll4"] = np.where(
            has_game, state[f"{m}_before4"], state[f"{m}_after4"]
        )
        state[f"{m}_roll4"] = state[f"{m}_roll4"].fillna(state[f"{m}_prev"])
        if m in ROLL8_METRICS:
            state[f"{m}_after8"] = state.groupby(["season", "team"], sort=False)[
                f"{m}_after8"
            ].ffill()
            state[f"{m}_roll8"] = np.where(
                has_game, state[f"{m}_before8"], state[f"{m}_after8"]
            )
            state[f"{m}_roll8"] = state[f"{m}_roll8"].fillna(state[f"{m}_prev"])

    keep = ["season", "team", "week"]
    keep += [f"{m}_roll4" for m in metrics]
    keep += [f"{m}_roll8" for m in ROLL8_METRICS]
    keep += [f"{m}_prev" for m in PREV_METRICS]
    state = state[keep]
    return state


# ---------------------------------------------------------------------------
# Elo state
# ---------------------------------------------------------------------------
def build_elo_state(games: pd.DataFrame) -> pd.DataFrame:
    g = games[games.home_score.notna() & games.away_score.notna()].copy()
    g = g.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    ratings: dict[str, float] = {}
    rows = []
    current_season = None
    for row in g.itertuples(index=False):
        if current_season is None:
            current_season = row.season
        if row.season != current_season:
            for team in list(ratings):
                ratings[team] = 1500.0 + 0.67 * (ratings[team] - 1500.0)
            current_season = row.season

        home, away = row.home_team, row.away_team
        h, a = ratings.get(home, 1500.0), ratings.get(away, 1500.0)
        margin = float(row.home_score - row.away_score)
        winner_elo = h if margin > 0 else a
        loser_elo = a if margin > 0 else h
        elo_diff = winner_elo - loser_elo
        mov = max(abs(margin), 1)
        multiplier = min(np.log(mov + 1.0) * 2.2 / ((elo_diff * 0.001) + 2.2), 3.0)
        expected_home = 1.0 / (1.0 + 10 ** ((a - h - 65.0) / 400.0))
        actual_home = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
        change = 20.0 * multiplier * (actual_home - expected_home)

        rows.append((row.season, row.week, home, h, h + change))
        rows.append((row.season, row.week, away, a, a - change))
        ratings[home] = h + change
        ratings[away] = a - change

    elo_tg = pd.DataFrame(
        rows, columns=["season", "week", "team", "elo_pre", "elo_post"]
    )
    elo_tg = elo_tg.groupby(["season", "week", "team"], as_index=False)[
        ["elo_pre", "elo_post"]
    ].mean()

    teams = elo_tg[["season", "team"]].drop_duplicates()
    grid = []
    for row in teams.itertuples(index=False):
        for week in range(1, 19):
            grid.append((row.season, row.team, week))
    grid = pd.DataFrame(grid, columns=["season", "team", "week"])
    state = grid.merge(elo_tg, on=["season", "team", "week"], how="left")
    state = state.sort_values(["season", "team", "week"]).reset_index(drop=True)

    # Pre-game rating if the team plays; otherwise the last post-game rating.
    last_post = state.groupby(["season", "team"], sort=False)["elo_post"].ffill()
    state["elo_state"] = state["elo_pre"].where(state["elo_pre"].notna(), last_post)
    return state[["season", "team", "week", "elo_state"]]


# ---------------------------------------------------------------------------
# QB state (actual starting QB, using only prior games)
# ---------------------------------------------------------------------------
def build_qb_state(seasons: Iterable[int] = SEASONS) -> pd.DataFrame:
    pw = nfl.load_player_week(seasons)
    qb = pw[pw.position == "QB"].copy()
    qb["time_idx"] = qb["season"] * 100 + qb["week"]
    qb["attempts"] = pd.to_numeric(qb["attempts"], errors="coerce").fillna(0.0)
    qb["sacks_suffered"] = pd.to_numeric(qb["sacks_suffered"], errors="coerce").fillna(
        0.0
    )
    qb["passing_epa"] = pd.to_numeric(qb["passing_epa"], errors="coerce")
    qb["passing_cpoe"] = pd.to_numeric(qb["passing_cpoe"], errors="coerce")
    qb["passing_interceptions"] = pd.to_numeric(
        qb["passing_interceptions"], errors="coerce"
    ).fillna(0.0)
    qb["rushing_epa"] = pd.to_numeric(qb["rushing_epa"], errors="coerce")
    qb["carries"] = pd.to_numeric(qb["carries"], errors="coerce").fillna(0.0)

    qb = qb.sort_values(
        ["player_id", "time_idx", "attempts"], ascending=[True, True, False]
    )
    qb = qb.drop_duplicates(["player_id", "time_idx"])
    qb["qb_epa_db"] = qb["passing_epa"] / (
        qb["attempts"] + qb["sacks_suffered"]
    ).replace(0, np.nan)
    qb["qb_cpoe"] = qb["passing_cpoe"]
    qb["qb_int_rate"] = qb["passing_interceptions"] / qb["attempts"].replace(0, np.nan)
    qb["qb_sack_rate"] = qb["sacks_suffered"] / (
        qb["attempts"] + qb["sacks_suffered"]
    ).replace(0, np.nan)
    qb["qb_rush_epa_carry"] = qb["rushing_epa"] / qb["carries"].replace(0, np.nan)

    g = qb.groupby("player_id", sort=False)
    for m in [
        "qb_epa_db",
        "qb_cpoe",
        "qb_int_rate",
        "qb_sack_rate",
        "qb_rush_epa_carry",
        "attempts",
    ]:
        qb[f"start_{m}_roll4"] = g[m].transform(
            lambda s: s.shift(1).rolling(4, min_periods=1).mean()
        )
    qb["start_career_attempts"] = g["attempts"].transform(lambda s: s.shift(1).cumsum())
    qb["start_games"] = g.cumcount()

    keep = ["player_id", "time_idx", "season", "week"] + [
        "start_qb_epa_db_roll4",
        "start_qb_cpoe_roll4",
        "start_qb_int_rate_roll4",
        "start_qb_sack_rate_roll4",
        "start_qb_rush_epa_carry_roll4",
        "start_attempts_roll4",
        "start_career_attempts",
        "start_games",
    ]
    return qb[keep]


# ---------------------------------------------------------------------------
# Skill-position injury flags
# ---------------------------------------------------------------------------
def build_skill_injury_flags(seasons: Iterable[int] = SEASONS) -> pd.DataFrame:
    """Flag whether the team's recent top WR/RB/TE is out or questionable."""
    pw = nfl.load_player_week(seasons)
    p = pw[pw.position.isin(["WR", "RB", "TE"])].copy()
    p["time_idx"] = p["season"] * 100 + p["week"]
    p["targets"] = pd.to_numeric(p["targets"], errors="coerce").fillna(0.0)
    p["carries"] = pd.to_numeric(p["carries"], errors="coerce").fillna(0.0)
    p = p.sort_values(["player_id", "time_idx"])
    g = p.groupby("player_id", sort=False)
    p["targets_roll4"] = g["targets"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    p["carries_roll4"] = g["carries"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )

    # Expand each player to all weeks so a player who is out still has a state.
    players = p[["season", "team", "player_id", "position"]].drop_duplicates()
    grid = []
    for row in players.itertuples(index=False):
        for week in range(1, 19):
            grid.append((row.season, row.team, row.player_id, row.position, week))
    grid = pd.DataFrame(
        grid, columns=["season", "team", "player_id", "position", "week"]
    )
    state = grid.merge(
        p[["season", "team", "player_id", "week", "targets_roll4", "carries_roll4"]],
        on=["season", "team", "player_id", "week"],
        how="left",
    )
    state = state.sort_values(["season", "team", "player_id", "week"])
    state["targets_roll4"] = state.groupby(["season", "team", "player_id"])[
        "targets_roll4"
    ].ffill()
    state["carries_roll4"] = state.groupby(["season", "team", "player_id"])[
        "carries_roll4"
    ].ffill()

    # Pick the top recent player per position group and team-week.
    out = []
    for group, col in [
        ("WR", "targets_roll4"),
        ("RB", "carries_roll4"),
        ("TE", "targets_roll4"),
    ]:
        sub = state[state.position == group].copy()
        sub = sub.sort_values(
            ["season", "team", "week", col], ascending=[True, True, True, False]
        )
        top = sub.drop_duplicates(["season", "team", "week"])
        top = top[["season", "team", "week", "player_id", "position"] + [col]]
        top = top.rename(
            columns={
                "player_id": f"{group.lower()}_top_id",
                col: f"{group.lower()}_top_usage",
            }
        )
        out.append(top)

    flags = out[0]
    for d in out[1:]:
        flags = flags.merge(d, on=["season", "team", "week"], how="left")

    inj = nfl.load_injuries(seasons)
    inj["status_score"] = [
        (
            0.0
            if rs == "Out"
            else 0.20
            if rs == "Doubtful"
            else 0.70
            if rs == "Questionable"
            else 0.35
            if isinstance(ps, str) and "Did Not Participate" in ps
            else 0.80
            if isinstance(ps, str) and "Limited" in ps
            else 1.0
        )
        for rs, ps in zip(inj["report_status"], inj["practice_status"])
    ]
    inj = inj[["season", "week", "team", "gsis_id", "status_score"]]
    inj["season"] = pd.to_numeric(inj["season"], errors="coerce").astype("int64")
    inj["week"] = pd.to_numeric(inj["week"], errors="coerce").astype("int64")
    flags["season"] = pd.to_numeric(flags["season"], errors="coerce").astype("int64")
    flags["week"] = pd.to_numeric(flags["week"], errors="coerce").astype("int64")
    inj = inj.groupby(["season", "week", "team", "gsis_id"], as_index=False)[
        "status_score"
    ].min()

    for group in ["wr", "rb", "te"]:
        col = f"{group}_top_id"
        flags = flags.merge(
            inj.rename(columns={"gsis_id": col, "status_score": f"{group}_top_status"}),
            on=["season", "week", "team", col],
            how="left",
        )
        flags[f"{group}_top_status"] = flags[f"{group}_top_status"].fillna(1.0)
        flags[f"{group}_out"] = (flags[f"{group}_top_status"] < 0.70).astype(float)
        flags[f"{group}_questionable"] = (
            (flags[f"{group}_top_status"] >= 0.70)
            & (flags[f"{group}_top_status"] < 0.90)
        ).astype(float)

    keep = ["season", "team", "week"] + [
        "wr_top_id",
        "wr_top_usage",
        "wr_out",
        "wr_questionable",
        "rb_top_id",
        "rb_top_usage",
        "rb_out",
        "rb_questionable",
        "te_top_id",
        "te_top_usage",
        "te_out",
        "te_questionable",
    ]
    return flags[keep]


# ---------------------------------------------------------------------------
# Game-level features
# ---------------------------------------------------------------------------
def _market_features(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()

    # American moneyline -> no-vig probability.
    def ml_prob(ml):
        ml = pd.to_numeric(ml, errors="coerce")
        out = np.where(ml > 0, 100.0 / (ml + 100.0), (-ml) / ((-ml) + 100.0))
        return out

    p_home = ml_prob(g.get("home_moneyline"))
    p_away = ml_prob(g.get("away_moneyline"))
    p_home = pd.Series(p_home, index=g.index)
    p_away = pd.Series(p_away, index=g.index)
    total_p = p_home + p_away
    g["market_home_win_prob"] = p_home / total_p.replace(0, np.nan)

    g["market_margin"] = pd.to_numeric(g["spread_line"], errors="coerce")
    g["market_total"] = pd.to_numeric(g["total_line"], errors="coerce")
    g["home_implied"] = g["market_total"] / 2 + g["market_margin"] / 2
    g["away_implied"] = g["market_total"] / 2 - g["market_margin"] / 2

    # Fallback win probability from the spread.
    missing = g["market_home_win_prob"].isna() & g["market_margin"].notna()
    if missing.any():
        from math import erf, sqrt

        sigma = 13.5
        g.loc[missing, "market_home_win_prob"] = [
            0.5 * (1 + erf((m / sigma) / sqrt(2)))
            for m in g.loc[missing, "market_margin"]
        ]
    return g


def _project_future_qbs(g: pd.DataFrame) -> pd.DataFrame:
    """For unplayed games, replace a listed starter who is ruled out.

    nflverse's games file can still list the preseason depth-chart starter in a
    future game even after he appears on the injury report.  Use the latest
    depth chart to promote the next available QB so the player-level features
    describe who is actually expected to play.
    """
    future = g["home_score"].isna()
    if not future.any():
        return g

    seasons = sorted(g.loc[future, "season"].dropna().unique().tolist())
    inj = nfl.load_injuries(seasons)
    if inj.empty:
        return g
    inj["status_score"] = [
        _status_score(rs, ps)
        for rs, ps in zip(inj["report_status"], inj["practice_status"])
    ]
    inj = inj.groupby(["season", "week", "team", "gsis_id"], as_index=False)[
        "status_score"
    ].min()

    dc_frames = []
    for season in seasons:
        try:
            dc = nfl.load_depth_charts(int(season))
        except FileNotFoundError:
            continue
        if dc.empty or "pos_abb" not in dc.columns:
            continue
        dc = dc[dc["pos_abb"] == "QB"].copy()
        dc["dt"] = pd.to_datetime(dc["dt"], errors="coerce")
        dc = dc.dropna(subset=["team", "gsis_id", "dt"])
        latest = dc.groupby("team")["dt"].transform("max")
        dc = dc[dc["dt"] == latest].copy()
        dc["pos_rank"] = pd.to_numeric(dc["pos_rank"], errors="coerce").fillna(99)
        dc = dc.sort_values(["team", "pos_rank"]).drop_duplicates(["team", "gsis_id"])
        dc["season"] = int(season)
        dc_frames.append(dc[["season", "team", "gsis_id", "player_name", "pos_rank"]])
    if not dc_frames:
        return g
    depth = pd.concat(dc_frames, ignore_index=True)

    status = {
        (int(r.season), int(r.week), r.team, r.gsis_id): float(r.status_score)
        for r in inj.itertuples(index=False)
    }
    depth_by_team = {
        (int(r.season), r.team): []
        for r in depth[["season", "team"]].drop_duplicates().itertuples(index=False)
    }
    for r in depth.itertuples(index=False):
        depth_by_team[(int(r.season), r.team)].append(
            (r.gsis_id, r.player_name, float(r.pos_rank))
        )
    for key in depth_by_team:
        depth_by_team[key].sort(key=lambda x: x[2])

    for side in ["home", "away"]:
        g[f"{side}_qb_projected"] = 0.0

    rows = g.index[future]
    for idx in rows:
        season = int(g.at[idx, "season"])
        week = int(g.at[idx, "week"])
        for side in ["home", "away"]:
            team = g.at[idx, f"{side}_team"]
            qid = g.at[idx, f"{side}_qb_id"]
            if pd.isna(qid):
                continue
            if status.get((season, week, team, qid), 1.0) >= 0.70:
                continue
            for cand_id, cand_name, _ in depth_by_team.get((season, team), []):
                if cand_id == qid:
                    continue
                if status.get((season, week, team, cand_id), 1.0) >= 0.70:
                    g.at[idx, f"{side}_qb_id"] = cand_id
                    g.at[idx, f"{side}_qb_name"] = cand_name
                    g.at[idx, f"{side}_qb_projected"] = 1.0
                    break
    return g


def build_game_features(
    games: pd.DataFrame,
    state: pd.DataFrame,
    elo_state: pd.DataFrame,
    qb_state: pd.DataFrame,
    injuries: pd.DataFrame,
    qb_flags: pd.DataFrame,
    skill_flags: pd.DataFrame,
) -> pd.DataFrame:
    g = _market_features(games).copy()
    g = _project_future_qbs(g)

    # Home and away team state
    state_cols = [c for c in state.columns if c not in {"season", "team", "week"}]
    hs = state.rename(columns={c: f"home_{c}" for c in state_cols})
    as_ = state.rename(columns={c: f"away_{c}" for c in state_cols})
    g = g.merge(
        hs,
        left_on=["season", "week", "home_team"],
        right_on=["season", "week", "team"],
        how="left",
        suffixes=("", "_hs"),
    )
    g = g.drop(columns=[c for c in ["team"] if c in g.columns])
    g = g.merge(
        as_,
        left_on=["season", "week", "away_team"],
        right_on=["season", "week", "team"],
        how="left",
        suffixes=("", "_as"),
    )
    g = g.drop(columns=[c for c in ["team"] if c in g.columns])

    # Elo
    g = g.merge(
        elo_state.rename(columns={"team": "home_team", "elo_state": "home_elo_state"}),
        on=["season", "week", "home_team"],
        how="left",
    )
    g = g.merge(
        elo_state.rename(columns={"team": "away_team", "elo_state": "away_elo_state"}),
        on=["season", "week", "away_team"],
        how="left",
    )
    g["elo_diff"] = g["home_elo_state"] - g["away_elo_state"]

    # Matchup features
    for off_m, def_m, label in MATCHUP_PAIRS:
        h_off = f"home_{off_m}_roll4"
        a_off = f"away_{off_m}_roll4"
        h_def = f"home_{def_m}_roll4"
        a_def = f"away_{def_m}_roll4"
        if all(c in g.columns for c in [h_off, a_off, h_def, a_def]):
            g[f"{label}_home_adv"] = g[h_off] - g[a_def]
            g[f"{label}_away_adv"] = g[a_off] - g[h_def]
            g[f"{label}_net"] = g[f"{label}_home_adv"] - g[f"{label}_away_adv"]

    # 8-week versions of the most important matchup nets
    for m in ROLL8_METRICS:
        h = f"home_{m}_roll8"
        a = f"away_{m}_roll8"
        if h in g.columns and a in g.columns:
            g[f"{m}_roll8_diff"] = g[h] - g[a]

    # Prior-season differences
    for m in PREV_METRICS:
        h = f"home_{m}_prev"
        a = f"away_{m}_prev"
        if h in g.columns and a in g.columns:
            g[f"{m}_prev_diff"] = g[h] - g[a]

    # Straight differences and sums
    for m in DIFF_METRICS:
        h = f"home_{m}_roll4"
        a = f"away_{m}_roll4"
        if h in g.columns and a in g.columns:
            g[f"{m}_diff"] = g[h] - g[a]
    for m in SUM_METRICS:
        h = f"home_{m}_roll4"
        a = f"away_{m}_roll4"
        if h in g.columns and a in g.columns:
            g[f"{m}_sum"] = g[h] + g[a]

    # Injuries (current week, known pre-game)
    inj_cols = INJURY_COLS
    inj_h = injuries.rename(
        columns={"team": "home_team", **{c: f"home_{c}" for c in inj_cols}}
    )
    inj_a = injuries.rename(
        columns={"team": "away_team", **{c: f"away_{c}" for c in inj_cols}}
    )
    g = g.merge(inj_h, on=["season", "week", "home_team"], how="left")
    g = g.merge(inj_a, on=["season", "week", "away_team"], how="left")
    for c in inj_cols:
        h, a = f"home_{c}", f"away_{c}"
        if h in g.columns and a in g.columns:
            g[f"{c}_diff"] = g[h] - g[a]
    if "home_inj_lost_off" in g.columns and "away_inj_lost_off" in g.columns:
        g["inj_lost_off_sum"] = g["home_inj_lost_off"] + g["away_inj_lost_off"]
        g["inj_lost_def_sum"] = g["home_inj_lost_def"] + g["away_inj_lost_def"]
        g["inj_lost_skill_diff"] = (
            g["home_inj_lost_rb"]
            + g["home_inj_lost_wr"]
            + g["home_inj_lost_te"]
            - g["away_inj_lost_rb"]
            - g["away_inj_lost_wr"]
            - g["away_inj_lost_te"]
        )

    # QB availability / change
    qf_h = qb_flags.rename(
        columns={
            "team": "home_team",
            "qb_out": "home_qb_out",
            "qb_questionable": "home_qb_questionable",
            "qb_status_score": "home_qb_status_score",
            "last_qb_id": "home_last_qb_id",
            "last_qb_name": "home_last_qb_name",
            "last_qb_attempts": "home_last_qb_attempts",
            "qb_report_status": "home_qb_report_status",
        }
    )
    qf_a = qb_flags.rename(
        columns={
            "team": "away_team",
            "qb_out": "away_qb_out",
            "qb_questionable": "away_qb_questionable",
            "qb_status_score": "away_qb_status_score",
            "last_qb_id": "away_last_qb_id",
            "last_qb_name": "away_last_qb_name",
            "last_qb_attempts": "away_last_qb_attempts",
            "qb_report_status": "away_qb_report_status",
        }
    )
    g = g.merge(qf_h, on=["season", "week", "home_team"], how="left")
    g = g.merge(qf_a, on=["season", "week", "away_team"], how="left")
    g["qb_out_diff"] = g["home_qb_out"] - g["away_qb_out"]
    g["qb_questionable_diff"] = g["home_qb_questionable"] - g["away_qb_questionable"]

    # Skill-position injury flags
    sf_cols = [c for c in skill_flags.columns if c not in {"season", "week", "team"}]
    sf_h = skill_flags.rename(
        columns={"team": "home_team", **{c: f"{c}_h" for c in sf_cols}}
    )
    sf_a = skill_flags.rename(
        columns={"team": "away_team", **{c: f"{c}_a" for c in sf_cols}}
    )
    g = g.merge(sf_h, on=["season", "week", "home_team"], how="left")
    g = g.merge(sf_a, on=["season", "week", "away_team"], how="left")
    for grp in ["wr", "rb", "te"]:
        h, a = f"{grp}_out_h", f"{grp}_out_a"
        if h in g.columns and a in g.columns:
            g[f"{grp}_out_diff"] = g[h] - g[a]
            g[f"{grp}_questionable_diff"] = (
                g[f"{grp}_questionable_h"] - g[f"{grp}_questionable_a"]
            )

    # Actual starting QB state (prior games only)
    qb_metric_cols = [
        "start_qb_epa_db_roll4",
        "start_qb_cpoe_roll4",
        "start_qb_int_rate_roll4",
        "start_qb_sack_rate_roll4",
        "start_qb_rush_epa_carry_roll4",
        "start_attempts_roll4",
        "start_career_attempts",
        "start_games",
    ]
    for side in ["home", "away"]:
        q = g[["game_id", "season", "week", f"{side}_qb_id"]].copy()
        q["time_idx"] = q["season"] * 100 + q["week"]
        q = q.sort_values("time_idx")
        qb_sorted = qb_state[["player_id", "time_idx"] + qb_metric_cols].sort_values(
            "time_idx"
        )
        merged = pd.merge_asof(
            q,
            qb_sorted,
            on="time_idx",
            left_by=f"{side}_qb_id",
            right_by="player_id",
            direction="backward",
            allow_exact_matches=False,
        )
        merged = merged.drop(
            columns=["player_id", "time_idx", f"{side}_qb_id"],
            errors="ignore",
        )
        merged = merged.rename(
            columns={c: f"{side}_{c}" for c in merged.columns if c not in {"game_id"}}
        )
        g = g.merge(merged, on="game_id", how="left")

    # If the starting QB has no prior attempts (rookie/backup), fall back to
    # the team's QB aggregate. The qb_out / qb_change flags still tell the
    # model this is a replacement situation.
    fallback = {
        "start_qb_epa_db_roll4": "qb_epa_db_roll4",
        "start_qb_cpoe_roll4": "qb_cpoe_roll4",
        "start_qb_int_rate_roll4": "qb_int_rate_roll4",
        "start_qb_sack_rate_roll4": "qb_sack_rate_roll4",
        "start_qb_rush_epa_carry_roll4": "qb_rush_epa_carry_roll4",
        "start_attempts_roll4": "qb_attempts_roll4",
    }
    for side in ["home", "away"]:
        for start_col, team_col in fallback.items():
            s, t = f"{side}_{start_col}", f"{side}_{team_col}"
            if s in g.columns and t in g.columns:
                g[s] = g[s].fillna(g[t])

    g["start_qb_epa_diff"] = g.get("home_start_qb_epa_db_roll4") - g.get(
        "away_start_qb_epa_db_roll4"
    )
    g["start_qb_cpoe_diff"] = g.get("home_start_qb_cpoe_roll4") - g.get(
        "away_start_qb_cpoe_roll4"
    )
    g["start_qb_int_diff"] = g.get("home_start_qb_int_rate_roll4") - g.get(
        "away_start_qb_int_rate_roll4"
    )
    g["start_qb_sack_diff"] = g.get("home_start_qb_sack_rate_roll4") - g.get(
        "away_start_qb_sack_rate_roll4"
    )
    g["start_qb_attempts_diff"] = g.get("home_start_attempts_roll4") - g.get(
        "away_start_attempts_roll4"
    )
    g["start_qb_career_diff"] = g.get("home_start_career_attempts") - g.get(
        "away_start_career_attempts"
    )
    g["home_start_qb_is_new"] = (
        g.get("home_start_career_attempts").fillna(0) < 100
    ).astype(float)
    g["away_start_qb_is_new"] = (
        g.get("away_start_career_attempts").fillna(0) < 100
    ).astype(float)
    g["start_qb_new_diff"] = g["home_start_qb_is_new"] - g["away_start_qb_is_new"]

    # QB change flag (target starter vs the team's last starter)
    for side in ["home", "away"]:
        last_id = f"{side}_last_qb_id"
        actual_id = f"{side}_qb_id"
        if last_id in g.columns and actual_id in g.columns:
            g[f"{side}_qb_change"] = (g[actual_id] != g[last_id]).astype(float)
            g.loc[g[last_id].isna(), f"{side}_qb_change"] = np.nan
    if "home_qb_change" in g.columns and "away_qb_change" in g.columns:
        g["qb_change_diff"] = g["home_qb_change"] - g["away_qb_change"]

    # Situational / weather
    g["rest_diff"] = pd.to_numeric(g["home_rest"], errors="coerce") - pd.to_numeric(
        g["away_rest"], errors="coerce"
    )
    g["home_short_week"] = (pd.to_numeric(g["home_rest"], errors="coerce") <= 4).astype(
        float
    )
    g["away_short_week"] = (pd.to_numeric(g["away_rest"], errors="coerce") <= 4).astype(
        float
    )
    g["short_week_diff"] = g["home_short_week"] - g["away_short_week"]
    g["is_dome"] = g["roof"].isin(["dome", "closed"]).astype(float)
    g["is_turf"] = (
        g["surface"]
        .astype(str)
        .str.contains("turf", case=False, na=False)
        .astype(float)
    )
    g["home_altitude"] = (g["home_team"] == "DEN").astype(float)
    g["div_game"] = pd.to_numeric(g["div_game"], errors="coerce").fillna(0.0)
    g["wind"] = pd.to_numeric(g["wind"], errors="coerce")
    g["temp"] = pd.to_numeric(g["temp"], errors="coerce")

    # Naive baseline: rolling scoring averages + home field.
    lg = 22.0
    hp = g.get("home_points_for_roll4")
    ha = g.get("home_points_allowed_roll4")
    ap = g.get("away_points_for_roll4")
    aa = g.get("away_points_allowed_roll4")
    g["naive_home_score"] = (hp.fillna(lg) + aa.fillna(lg)) / 2 + 1.5
    g["naive_away_score"] = (ap.fillna(lg) + ha.fillna(lg)) / 2 - 1.5
    g["naive_margin"] = g["naive_home_score"] - g["naive_away_score"]
    g["naive_total"] = g["naive_home_score"] + g["naive_away_score"]

    # Targets
    g["margin"] = pd.to_numeric(g["home_score"], errors="coerce") - pd.to_numeric(
        g["away_score"], errors="coerce"
    )
    g["total"] = pd.to_numeric(g["home_score"], errors="coerce") + pd.to_numeric(
        g["away_score"], errors="coerce"
    )
    g["home_win"] = np.where(
        g["margin"] > 0, 1.0, np.where(g["margin"] < 0, 0.0, np.nan)
    )

    return g


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_all(seasons: Iterable[int] = SEASONS) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_games = pd.read_parquet(PROCESSED / "team_games.parquet")
    games = nfl.load_games(seasons)

    print("Building weekly team state...")
    state = build_team_week_state(team_games, seasons)
    print("Building Elo state...")
    elo_state = build_elo_state(games)
    print("Building QB state...")
    qb_state = build_qb_state(seasons)
    print("Building injury features...")
    injuries = build_injury_features(seasons)
    qb_flags = build_qb_injury_flags(seasons)
    print("Building skill-position injury flags...")
    skill_flags = build_skill_injury_flags(seasons)

    print("Building game features...")
    features = build_game_features(
        games, state, elo_state, qb_state, injuries, qb_flags, skill_flags
    )

    state.to_parquet(PROCESSED / "team_week_state.parquet", index=False)
    features.to_parquet(PROCESSED / "game_features.parquet", index=False)
    print(f"Game features: {features.shape}")
    print(f"Completed games: {features[features.home_score.notna()].shape[0]:,}")
    print(f"Future games: {features[features.home_score.isna()].shape[0]:,}")
    return features, state


if __name__ == "__main__":
    build_all()
