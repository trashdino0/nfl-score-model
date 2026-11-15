"""Build team-game and team-week tables from nflverse data."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PROCESSED, SEASONS, SEASON_TYPE
from . import data as nfl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / b.replace(0, np.nan)


def _parse_clock(value) -> float:
    """Parse '3:33' -> 213 seconds."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    m = re.match(r"^\s*(\d+):(\d+)\s*$", str(value))
    if not m:
        return np.nan
    return float(int(m.group(1)) * 60 + int(m.group(2)))


def _parse_yardline(value) -> float:
    """Parse nflverse yard line strings into yards to the opponent's goal."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if s in {"50", "MID"}:
        return 50.0
    m = re.match(r"^([A-Z]{2,3})\s+(\d+)$", s)
    if not m:
        return np.nan
    prefix, num = m.group(1), float(m.group(2))
    # 'ARI 22' means the offense is on its own 22 -> 78 yards to goal.
    # 'OPP 22' means 22 yards to goal.
    if prefix == "OPP":
        return num
    return 100.0 - num


def _status_score(report_status, practice_status) -> float:
    """Pre-game availability score: 1 = fully available, 0 = out."""
    if report_status == "Out":
        return 0.0
    if report_status == "Doubtful":
        return 0.20
    if report_status == "Questionable":
        return 0.70
    if isinstance(practice_status, str):
        if "Did Not Participate" in practice_status:
            return 0.35
        if "Limited" in practice_status:
            return 0.80
        if "Full Participation" in practice_status:
            return 1.0
    return 1.0


def _pos_group(pos) -> str:
    pos = str(pos).upper()
    if pos == "QB":
        return "QB"
    if pos in {"RB", "FB", "HB"}:
        return "RB"
    if pos == "WR":
        return "WR"
    if pos == "TE":
        return "TE"
    if pos in {"T", "G", "C", "OL"}:
        return "OL"
    if pos in {"DE", "DT", "NT", "DL", "EDGE"}:
        return "DL"
    if pos == "LB":
        return "LB"
    if pos in {"CB", "S", "DB", "FS", "SS"}:
        return "DB"
    if pos == "K":
        return "K"
    if pos == "P":
        return "P"
    return "OTHER"


# ---------------------------------------------------------------------------
# PBP aggregation
# ---------------------------------------------------------------------------
def _pbp_season_aggregates(pbp: pd.DataFrame) -> pd.DataFrame:
    keys = ["game_id", "season", "week", "team"]
    pbp = pbp.copy()
    pbp["one"] = 1.0
    pbp["is_explosive_pass"] = (
        (pbp["pass"] == 1) & (pbp["yards_gained"] >= 20)
    ).astype(float)
    pbp["is_explosive_rush"] = (
        (pbp["rush"] == 1) & (pbp["yards_gained"] >= 10)
    ).astype(float)
    pbp["is_turnover"] = (
        (pbp["interception"] == 1) | (pbp["fumble_lost"] == 1)
    ).astype(float)
    pbp["is_early"] = (
        pbp["down"].isin([1, 2]) & pbp["play_type"].isin(["pass", "run"])
    ).astype(float)
    pbp["is_early_pass"] = (pbp["is_early"] == 1) & (pbp["pass"] == 1)
    pbp["is_third"] = (pbp["down"] == 3).astype(float)
    pbp["is_third_conv"] = (pbp["is_third"] == 1) & (pbp["third_down_converted"] == 1)

    scrimmage = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

    # Offense
    off = (
        scrimmage.assign(team=scrimmage["posteam"])
        .groupby(keys, as_index=False)
        .agg(
            pbp_plays=("one", "sum"),
            pbp_pass_plays=("pass", "sum"),
            pbp_rush_plays=("rush", "sum"),
            pbp_epa=("epa", "mean"),
            pbp_success=("success", "mean"),
            pbp_first_downs=("first_down", "sum"),
            pbp_sacks=("sack", "sum"),
            pbp_qb_hits=("qb_hit", "sum"),
            pbp_turnovers=("is_turnover", "sum"),
            pbp_explosive_pass=("is_explosive_pass", "sum"),
            pbp_explosive_rush=("is_explosive_rush", "sum"),
            pbp_pass_yards=("passing_yards", "sum"),
            pbp_rush_yards=("rushing_yards", "sum"),
            pbp_air_yards=("air_yards", "sum"),
            pbp_yac=("yards_after_catch", "sum"),
            pbp_pass_oe=("pass_oe", "mean"),
        )
    )
    pass_off = (
        scrimmage[scrimmage["pass"] == 1]
        .assign(team=lambda d: d["posteam"])
        .groupby(keys, as_index=False)
        .agg(pbp_pass_epa=("epa", "mean"), pbp_pass_success=("success", "mean"))
    )
    rush_off = (
        scrimmage[scrimmage["rush"] == 1]
        .assign(team=lambda d: d["posteam"])
        .groupby(keys, as_index=False)
        .agg(pbp_rush_epa=("epa", "mean"), pbp_rush_success=("success", "mean"))
    )
    early = (
        scrimmage[scrimmage["is_early"] == 1]
        .assign(team=lambda d: d["posteam"])
        .groupby(keys, as_index=False)
        .agg(pbp_early_plays=("one", "sum"), pbp_early_pass=("is_early_pass", "sum"))
    )
    third = (
        scrimmage[scrimmage["is_third"] == 1]
        .assign(team=lambda d: d["posteam"])
        .groupby(keys, as_index=False)
        .agg(pbp_third_plays=("one", "sum"), pbp_third_conv=("is_third_conv", "sum"))
    )
    penalties = (
        pbp[pbp["posteam"].notna()]
        .assign(team=lambda d: d["posteam"])
        .groupby(keys, as_index=False)
        .agg(
            pbp_penalties=("penalty", "sum"), pbp_penalty_yards=("penalty_yards", "sum")
        )
    )
    off = (
        off.merge(pass_off, on=keys, how="left")
        .merge(rush_off, on=keys, how="left")
        .merge(early, on=keys, how="left")
        .merge(third, on=keys, how="left")
        .merge(penalties, on=keys, how="left")
    )

    # Defense (same plays viewed from the defense)
    def_ = (
        scrimmage.assign(team=scrimmage["defteam"])
        .groupby(keys, as_index=False)
        .agg(
            pbp_def_plays=("one", "sum"),
            pbp_def_pass_plays=("pass", "sum"),
            pbp_def_rush_plays=("rush", "sum"),
            pbp_def_epa=("epa", "mean"),
            pbp_def_success=("success", "mean"),
            pbp_def_first_downs=("first_down", "sum"),
            pbp_def_sacks=("sack", "sum"),
            pbp_def_qb_hits=("qb_hit", "sum"),
            pbp_def_turnovers=("is_turnover", "sum"),
            pbp_def_explosive_pass=("is_explosive_pass", "sum"),
            pbp_def_explosive_rush=("is_explosive_rush", "sum"),
            pbp_def_pass_yards=("passing_yards", "sum"),
            pbp_def_rush_yards=("rushing_yards", "sum"),
            pbp_def_pass_oe=("pass_oe", "mean"),
        )
    )
    pass_def = (
        scrimmage[scrimmage["pass"] == 1]
        .assign(team=lambda d: d["defteam"])
        .groupby(keys, as_index=False)
        .agg(pbp_def_pass_epa=("epa", "mean"), pbp_def_pass_success=("success", "mean"))
    )
    rush_def = (
        scrimmage[scrimmage["rush"] == 1]
        .assign(team=lambda d: d["defteam"])
        .groupby(keys, as_index=False)
        .agg(pbp_def_rush_epa=("epa", "mean"), pbp_def_rush_success=("success", "mean"))
    )
    third_def = (
        scrimmage[scrimmage["is_third"] == 1]
        .assign(team=lambda d: d["defteam"])
        .groupby(keys, as_index=False)
        .agg(
            pbp_def_third_plays=("one", "sum"),
            pbp_def_third_conv=("is_third_conv", "sum"),
        )
    )
    def_ = (
        def_.merge(pass_def, on=keys, how="left")
        .merge(rush_def, on=keys, how="left")
        .merge(third_def, on=keys, how="left")
    )

    out = off.merge(def_, on=keys, how="outer")

    # Drives: one row per drive, then aggregate to offense and defense.
    drives = pbp[pbp["fixed_drive"].notna() & pbp["posteam"].notna()].copy()
    if len(drives):
        drives = drives.sort_values(["game_id", "posteam", "fixed_drive", "play_id"])
        drive = drives.groupby(
            ["game_id", "season", "week", "posteam", "fixed_drive"], as_index=False
        ).agg(
            defteam=("defteam", "first"),
            drive_play_count=("drive_play_count", "max"),
            drive_time=("drive_time_of_possession", "first"),
            drive_inside20=("drive_inside20", "max"),
            drive_end=(
                "drive_end_transition",
                lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan,
            ),
            drive_start=("drive_start_yard_line", "first"),
        )
        drive["drive_seconds"] = drive["drive_time"].map(_parse_clock)
        drive["drive_start_yards_to_goal"] = drive["drive_start"].map(_parse_yardline)
        drive["drive_td"] = (drive["drive_end"] == "TOUCHDOWN").astype(float)
        drive["drive_fg"] = (drive["drive_end"] == "FIELD_GOAL").astype(float)
        drive["drive_score"] = (
            (drive["drive_td"] == 1) | (drive["drive_fg"] == 1)
        ).astype(float)
        drive["one"] = 1.0
        drive["drive_plays"] = pd.to_numeric(
            drive["drive_play_count"], errors="coerce"
        ).fillna(0.0)

        dr_off = (
            drive.groupby(["game_id", "season", "week", "posteam"], as_index=False)
            .agg(
                pbp_drives=("one", "sum"),
                pbp_drive_inside20=("drive_inside20", "sum"),
                pbp_drive_td=("drive_td", "sum"),
                pbp_drive_fg=("drive_fg", "sum"),
                pbp_drive_score=("drive_score", "sum"),
                pbp_drive_plays=("drive_plays", "sum"),
                pbp_drive_seconds=("drive_seconds", "sum"),
                pbp_drive_start=("drive_start_yards_to_goal", "mean"),
            )
            .rename(columns={"posteam": "team"})
        )
        dr_def = (
            drive.groupby(["game_id", "season", "week", "defteam"], as_index=False)
            .agg(
                pbp_def_drives=("one", "sum"),
                pbp_def_drive_inside20=("drive_inside20", "sum"),
                pbp_def_drive_td=("drive_td", "sum"),
                pbp_def_drive_fg=("drive_fg", "sum"),
                pbp_def_drive_score=("drive_score", "sum"),
                pbp_def_drive_plays=("drive_plays", "sum"),
                pbp_def_drive_seconds=("drive_seconds", "sum"),
            )
            .rename(columns={"defteam": "team"})
        )
        out = out.merge(dr_off, on=keys, how="left").merge(dr_def, on=keys, how="left")

    return out


def build_pbp_team_games(seasons: Iterable[int] = SEASONS) -> pd.DataFrame:
    frames = []
    for season, pbp in nfl.iter_pbp(seasons):
        print(f"  PBP {season}: {len(pbp):,} plays")
        frames.append(_pbp_season_aggregates(pbp))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Player and usage aggregation
# ---------------------------------------------------------------------------
def _top_player(
    df: pd.DataFrame, group_keys, sort_col: str, prefix: str, cols: dict
) -> pd.DataFrame:
    """Select the top player per team-game by sort_col and rename columns."""
    if df.empty:
        return pd.DataFrame(columns=group_keys)
    d = df.copy()
    d[sort_col] = pd.to_numeric(d[sort_col], errors="coerce").fillna(0.0)
    d = d.sort_values(
        group_keys + [sort_col], ascending=[True] * len(group_keys) + [False]
    )
    d = d.drop_duplicates(group_keys)
    keep = dict(cols)
    keep[sort_col] = sort_col
    out = d[group_keys + list(keep.keys())].rename(
        columns={k: f"{prefix}_{v}" for k, v in keep.items()}
    )
    return out


def build_player_team_games(seasons: Iterable[int] = SEASONS) -> pd.DataFrame:
    pw = nfl.load_player_week(seasons)
    keys = ["game_id", "season", "week", "team"]

    # QB: most attempts
    qb_cols = {
        "player_id": "id",
        "player_display_name": "name",
        "attempts": "attempts",
        "completions": "completions",
        "passing_yards": "passing_yards",
        "passing_tds": "passing_tds",
        "passing_interceptions": "interceptions",
        "sacks_suffered": "sacks",
        "passing_epa": "passing_epa",
        "passing_cpoe": "cpoe",
        "carries": "carries",
        "rushing_yards": "rushing_yards",
        "rushing_epa": "rushing_epa",
        "rushing_tds": "rushing_tds",
        "fumbles_lost_total": "fumbles_lost",
    }
    qb = _top_player(pw[pw.position == "QB"], keys, "attempts", "qb", qb_cols)
    qb["qb_epa_db"] = _safe_div(
        qb["qb_passing_epa"], qb["qb_attempts"] + qb["qb_sacks"]
    )
    qb["qb_int_rate"] = _safe_div(qb["qb_interceptions"], qb["qb_attempts"])
    qb["qb_sack_rate"] = _safe_div(qb["qb_sacks"], qb["qb_attempts"] + qb["qb_sacks"])
    qb["qb_rush_epa_carry"] = _safe_div(qb["qb_rushing_epa"], qb["qb_carries"])
    qb["qb_yards_attempt"] = _safe_div(qb["qb_passing_yards"], qb["qb_attempts"])

    # WR1 by targets
    wr_cols = {
        "player_id": "id",
        "player_display_name": "name",
        "targets": "targets",
        "receptions": "receptions",
        "receiving_yards": "receiving_yards",
        "receiving_tds": "receiving_tds",
        "receiving_epa": "receiving_epa",
        "receiving_air_yards": "air_yards",
        "receiving_yards_after_catch": "yac",
        "target_share": "target_share",
        "air_yards_share": "air_yards_share",
        "wopr": "wopr",
    }
    wr1 = _top_player(pw[pw.position == "WR"], keys, "targets", "wr1", wr_cols)
    wr1["wr1_epa_target"] = _safe_div(wr1["wr1_receiving_epa"], wr1["wr1_targets"])
    wr1["wr1_yards_target"] = _safe_div(wr1["wr1_receiving_yards"], wr1["wr1_targets"])

    # WR corps: top 3 by targets
    wrs = pw[pw.position == "WR"].copy()
    wrs["targets"] = pd.to_numeric(wrs["targets"], errors="coerce").fillna(0.0)
    wrs["wr_rank"] = wrs.groupby(keys)["targets"].rank(ascending=False, method="first")
    wrs3 = wrs[wrs["wr_rank"] <= 3]
    wr_corps = wrs3.groupby(keys, as_index=False).agg(
        wr_corps_targets=("targets", "sum"),
        wr_corps_receptions=("receptions", "sum"),
        wr_corps_yards=("receiving_yards", "sum"),
        wr_corps_tds=("receiving_tds", "sum"),
        wr_corps_epa=("receiving_epa", "sum"),
        wr_corps_air_yards=("receiving_air_yards", "sum"),
        wr_corps_target_share=("target_share", "sum"),
        wr_corps_air_yards_share=("air_yards_share", "sum"),
    )
    wr_corps["wr_corps_epa_target"] = _safe_div(
        wr_corps["wr_corps_epa"], wr_corps["wr_corps_targets"]
    )
    wr_corps["wr_corps_yards_target"] = _safe_div(
        wr_corps["wr_corps_yards"], wr_corps["wr_corps_targets"]
    )

    # TE1
    te_cols = {
        "player_id": "id",
        "player_display_name": "name",
        "targets": "targets",
        "receptions": "receptions",
        "receiving_yards": "receiving_yards",
        "receiving_epa": "receiving_epa",
        "target_share": "target_share",
    }
    te1 = _top_player(pw[pw.position == "TE"], keys, "targets", "te1", te_cols)
    te1["te1_epa_target"] = _safe_div(te1["te1_receiving_epa"], te1["te1_targets"])
    te1["te1_yards_target"] = _safe_div(te1["te1_receiving_yards"], te1["te1_targets"])

    # RB1 by carries
    rb_cols = {
        "player_id": "id",
        "player_display_name": "name",
        "carries": "carries",
        "rushing_yards": "rushing_yards",
        "rushing_tds": "rushing_tds",
        "rushing_epa": "rushing_epa",
        "targets": "targets",
        "receptions": "receptions",
        "receiving_yards": "receiving_yards",
        "receiving_epa": "receiving_epa",
        "target_share": "target_share",
    }
    rb1 = _top_player(pw[pw.position == "RB"], keys, "carries", "rb1", rb_cols)
    rb1["rb1_rush_epa_carry"] = _safe_div(rb1["rb1_rushing_epa"], rb1["rb1_carries"])
    rb1["rb1_yards_carry"] = _safe_div(rb1["rb1_rushing_yards"], rb1["rb1_carries"])
    rb1["rb1_epa_target"] = _safe_div(rb1["rb1_receiving_epa"], rb1["rb1_targets"])

    # RB room
    rbs = pw[pw.position == "RB"].copy()
    rb_room = rbs.groupby(keys, as_index=False).agg(
        rb_room_carries=("carries", "sum"),
        rb_room_rush_yards=("rushing_yards", "sum"),
        rb_room_rush_epa=("rushing_epa", "sum"),
        rb_room_targets=("targets", "sum"),
        rb_room_receiving_epa=("receiving_epa", "sum"),
    )
    rb_room["rb_room_rush_epa_carry"] = _safe_div(
        rb_room["rb_room_rush_epa"], rb_room["rb_room_carries"]
    )
    rb_room["rb_room_yards_carry"] = _safe_div(
        rb_room["rb_room_rush_yards"], rb_room["rb_room_carries"]
    )

    out = qb
    for d in (wr1, wr_corps, te1, rb1, rb_room):
        out = out.merge(d, on=keys, how="outer")
    return out


# ---------------------------------------------------------------------------
# Coverage / pressure aggregation (PFR advanced defense, 2018+)
# ---------------------------------------------------------------------------
def build_advdef_team_games(seasons: Iterable[int] = SEASONS) -> pd.DataFrame:
    adv = nfl.load_advdef(seasons)
    if adv.empty:
        return pd.DataFrame()
    keys = ["game_id", "season", "week", "team"]
    players = nfl.load_players()[["pfr_id", "position"]].drop_duplicates("pfr_id")
    adv = adv.merge(players, left_on="pfr_player_id", right_on="pfr_id", how="left")
    adv["pos"] = adv["position"].fillna("UNK")
    adv["is_cb"] = adv["pos"].isin(["CB", "DB", "FS", "SS", "S"])
    adv["is_front"] = adv["pos"].isin(["DE", "DT", "NT", "DL", "LB", "EDGE"])

    for src, dst in [
        ("def_targets", "targets"),
        ("def_completions_allowed", "completions_allowed"),
        ("def_yards_allowed", "yards_allowed"),
        ("def_receiving_td_allowed", "td_allowed"),
        ("def_ints", "ints"),
        ("def_pressures", "pressures"),
        ("def_times_hurried", "hurries"),
        ("def_times_hitqb", "qb_hits"),
        ("def_sacks", "sacks"),
        ("def_times_blitzed", "blitzes"),
        ("def_adot", "adot"),
        ("def_passer_rating_allowed", "passer_rating"),
    ]:
        adv[dst] = pd.to_numeric(adv.get(src), errors="coerce").fillna(0.0)

    team = adv.groupby(keys, as_index=False).agg(
        adv_targets=("targets", "sum"),
        adv_completions_allowed=("completions_allowed", "sum"),
        adv_yards_allowed=("yards_allowed", "sum"),
        adv_td_allowed=("td_allowed", "sum"),
        adv_ints=("ints", "sum"),
        adv_pressures=("pressures", "sum"),
        adv_hurries=("hurries", "sum"),
        adv_qb_hits=("qb_hits", "sum"),
        adv_sacks=("sacks", "sum"),
        adv_blitzes=("blitzes", "sum"),
        adv_rating_num=("passer_rating", lambda s: np.nan),
    )
    # Weighted averages need the target weights, so compute separately.
    adv["rating_x_targets"] = adv["passer_rating"] * adv["targets"]
    adv["adot_x_targets"] = adv["adot"] * adv["targets"]
    wavg = adv.groupby(keys, as_index=False).agg(
        rating_num=("rating_x_targets", "sum"),
        adot_num=("adot_x_targets", "sum"),
        target_sum=("targets", "sum"),
    )
    team = team.drop(columns=["adv_rating_num"]).merge(wavg, on=keys, how="left")
    team["coverage_passer_rating"] = _safe_div(team["rating_num"], team["target_sum"])
    team["coverage_adot"] = _safe_div(team["adot_num"], team["target_sum"])
    team["coverage_yards_per_target"] = _safe_div(
        team["adv_yards_allowed"], team["adv_targets"]
    )
    team["coverage_completion_pct"] = _safe_div(
        team["adv_completions_allowed"], team["adv_targets"]
    )
    team["coverage_td_rate"] = _safe_div(team["adv_td_allowed"], team["adv_targets"])

    # Top CB by targets
    cb = adv[adv["is_cb"]].sort_values(
        keys + ["targets"], ascending=[True] * len(keys) + [False]
    )
    cb = cb.drop_duplicates(keys)
    cb_out = cb[
        keys
        + [
            "pfr_player_name",
            "targets",
            "yards_allowed",
            "passer_rating",
            "completions_allowed",
            "adot",
        ]
    ].copy()
    cb_out["cb1_yards_target"] = _safe_div(cb_out["yards_allowed"], cb_out["targets"])
    cb_out["cb1_completion_pct"] = _safe_div(
        cb_out["completions_allowed"], cb_out["targets"]
    )
    cb_out = cb_out.rename(
        columns={
            "pfr_player_name": "cb1_name",
            "targets": "cb1_targets",
            "passer_rating": "cb1_passer_rating",
            "adot": "cb1_adot",
        }
    ).drop(columns=["yards_allowed", "completions_allowed"])

    # Top coverage defender overall
    cov = adv.sort_values(
        keys + ["targets"], ascending=[True] * len(keys) + [False]
    ).drop_duplicates(keys)
    cov_out = cov[
        keys + ["pfr_player_name", "targets", "yards_allowed", "passer_rating"]
    ].copy()
    cov_out["cov1_yards_target"] = _safe_div(
        cov_out["yards_allowed"], cov_out["targets"]
    )
    cov_out = cov_out.rename(
        columns={
            "pfr_player_name": "cov1_name",
            "targets": "cov1_targets",
            "passer_rating": "cov1_passer_rating",
        }
    ).drop(columns=["yards_allowed"])

    # Top pass rusher by pressures
    pr = (
        adv[adv["is_front"]]
        .sort_values(keys + ["pressures"], ascending=[True] * len(keys) + [False])
        .drop_duplicates(keys)
    )
    pr_out = pr[
        keys + ["pfr_player_name", "pressures", "hurries", "qb_hits", "sacks"]
    ].rename(
        columns={
            "pfr_player_name": "prush1_name",
            "pressures": "prush1_pressures",
            "hurries": "prush1_hurries",
            "qb_hits": "prush1_qb_hits",
            "sacks": "prush1_sacks",
        }
    )

    out = team
    for d in (cb_out, cov_out, pr_out):
        out = out.merge(d, on=keys, how="left")
    return out


# ---------------------------------------------------------------------------
# Snaps, injuries, Elo
# ---------------------------------------------------------------------------
def build_snap_team_games(seasons: Iterable[int] = SEASONS) -> pd.DataFrame:
    snaps = nfl.load_snaps(seasons)
    keys = ["game_id", "season", "week", "team"]
    team = snaps.groupby(keys, as_index=False).agg(
        team_off_snaps=("offense_snaps", "max"),
        team_def_snaps=("defense_snaps", "max"),
        team_st_snaps=("st_snaps", "max"),
    )
    return team


def _player_snap_state(seasons: Iterable[int]) -> pd.DataFrame:
    """Rolling snap share as of before a given week, for injury weighting."""
    snaps = nfl.load_snaps(seasons)
    p = snaps[
        ["season", "week", "team", "pfr_player_id", "offense_pct", "defense_pct"]
    ].copy()
    p["offense_pct"] = pd.to_numeric(p["offense_pct"], errors="coerce")
    p["defense_pct"] = pd.to_numeric(p["defense_pct"], errors="coerce")
    p = p.sort_values(["season", "team", "pfr_player_id", "week"])
    g = p.groupby(["season", "team", "pfr_player_id"])
    p["off_pct_roll4"] = g["offense_pct"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    p["def_pct_roll4"] = g["defense_pct"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    return p[
        ["season", "week", "team", "pfr_player_id", "off_pct_roll4", "def_pct_roll4"]
    ]


def build_injury_features(seasons: Iterable[int] = SEASONS) -> pd.DataFrame:
    inj = nfl.load_injuries(seasons)
    players = nfl.load_players()[["gsis_id", "pfr_id", "position"]].drop_duplicates(
        "gsis_id"
    )
    inj = inj.merge(players, on="gsis_id", how="left", suffixes=("", "_player"))
    inj["position"] = inj["position"].fillna(inj.get("position_player"))
    inj["pos_group"] = inj["position"].map(_pos_group)
    inj["status_score"] = [
        _status_score(rs, ps)
        for rs, ps in zip(inj["report_status"], inj["practice_status"])
    ]
    inj["lost"] = 1.0 - inj["status_score"]

    state = _player_snap_state(seasons)
    inj["season"] = pd.to_numeric(inj["season"], errors="coerce").astype("int64")
    inj["week"] = pd.to_numeric(inj["week"], errors="coerce").astype("int64")
    state["season"] = pd.to_numeric(state["season"], errors="coerce").astype("int64")
    state["week"] = pd.to_numeric(state["week"], errors="coerce").astype("int64")

    # Expand each player's snap state to every week, then forward-fill. This
    # gives the last known role for a player who has not played recently.
    players_weeks = state[["season", "team", "pfr_player_id"]].drop_duplicates()
    grid = []
    for row in players_weeks.itertuples(index=False):
        for week in range(1, 19):
            grid.append((row.season, row.team, row.pfr_player_id, week))
    grid = pd.DataFrame(grid, columns=["season", "team", "pfr_player_id", "week"])
    state_full = grid.merge(
        state, on=["season", "team", "pfr_player_id", "week"], how="left"
    )
    state_full = state_full.sort_values(["season", "team", "pfr_player_id", "week"])
    for col in ["off_pct_roll4", "def_pct_roll4"]:
        state_full[col] = state_full.groupby(
            ["season", "team", "pfr_player_id"], sort=False
        )[col].ffill()
    state_full = state_full.rename(columns={"pfr_player_id": "pfr_id"})
    inj = inj.merge(state_full, on=["season", "week", "team", "pfr_id"], how="left")
    inj["off_weight"] = pd.to_numeric(inj["off_pct_roll4"], errors="coerce").fillna(0.0)
    inj["def_weight"] = pd.to_numeric(inj["def_pct_roll4"], errors="coerce").fillna(0.0)
    inj["lost_off"] = inj["lost"] * inj["off_weight"]
    inj["lost_def"] = inj["lost"] * inj["def_weight"]

    keys = ["season", "week", "team"]
    out = inj.groupby(keys, as_index=False).agg(
        inj_lost_off=("lost_off", "sum"),
        inj_lost_def=("lost_def", "sum"),
        inj_out_count=("report_status", lambda s: (s == "Out").sum()),
        inj_doubtful_count=("report_status", lambda s: (s == "Doubtful").sum()),
        inj_questionable_count=("report_status", lambda s: (s == "Questionable").sum()),
    )
    for group, col, prefix in [
        ("QB", "lost_off", "inj_lost_qb"),
        ("RB", "lost_off", "inj_lost_rb"),
        ("WR", "lost_off", "inj_lost_wr"),
        ("TE", "lost_off", "inj_lost_te"),
        ("OL", "lost_off", "inj_lost_ol"),
        ("DL", "lost_def", "inj_lost_dl"),
        ("LB", "lost_def", "inj_lost_lb"),
        ("DB", "lost_def", "inj_lost_db"),
    ]:
        sub = inj[inj["pos_group"] == group].groupby(keys, as_index=False)[col].sum()
        sub = sub.rename(columns={col: prefix})
        out = out.merge(sub, on=keys, how="left")
    return out.fillna({c: 0.0 for c in out.columns if c not in keys})


def build_qb_injury_flags(seasons: Iterable[int] = SEASONS) -> pd.DataFrame:
    """Flag whether the team's most recent starting QB is ruled out/questionable."""
    pw = nfl.load_player_week(seasons)
    qbs = pw[pw.position == "QB"].copy()
    qbs["attempts"] = pd.to_numeric(qbs["attempts"], errors="coerce").fillna(0.0)
    qbs = qbs.sort_values(
        ["season", "week", "team", "attempts"], ascending=[True, True, True, False]
    )
    qbs = qbs.drop_duplicates(["season", "week", "team"])
    qbs = qbs[
        ["season", "week", "team", "player_id", "player_display_name", "attempts"]
    ].rename(
        columns={
            "player_id": "qb_id",
            "player_display_name": "qb_name",
            "attempts": "qb_attempts",
        }
    )
    qbs["season"] = pd.to_numeric(qbs["season"], errors="coerce").astype("int64")
    qbs["week"] = pd.to_numeric(qbs["week"], errors="coerce").astype("int64")

    # Build a complete team-week grid so future weeks (and byes) get the most
    # recent prior starter. Shift within team across seasons so Week 1 inherits
    # the previous season's final starter.
    seasons_seen = sorted(qbs["season"].unique())
    teams = sorted(qbs["team"].unique())
    grid = []
    for season in seasons_seen:
        for team in teams:
            for week in range(1, 19):
                grid.append((season, team, week))
    grid = pd.DataFrame(grid, columns=["season", "team", "week"])
    state = grid.merge(qbs, on=["season", "team", "week"], how="left")
    state = state.sort_values(["team", "season", "week"]).reset_index(drop=True)
    for col in ["qb_id", "qb_name", "qb_attempts"]:
        state[f"{col}_ffill"] = state.groupby("team", sort=False)[col].ffill()
        state[f"last_{col}"] = state.groupby("team", sort=False)[f"{col}_ffill"].shift(
            1
        )
    state = state.rename(
        columns={
            "last_qb_id": "last_qb_id",
            "last_qb_name": "last_qb_name",
            "last_qb_attempts": "last_qb_attempts",
        }
    )

    inj = nfl.load_injuries(seasons)
    inj["status_score"] = [
        _status_score(rs, ps)
        for rs, ps in zip(inj["report_status"], inj["practice_status"])
    ]
    inj = inj[
        [
            "season",
            "week",
            "team",
            "gsis_id",
            "status_score",
            "report_status",
            "practice_status",
        ]
    ]
    inj["season"] = pd.to_numeric(inj["season"], errors="coerce").astype("int64")
    inj["week"] = pd.to_numeric(inj["week"], errors="coerce").astype("int64")
    inj = inj.groupby(["season", "week", "team", "gsis_id"], as_index=False).agg(
        status_score=("status_score", "min"),
        report_status=("report_status", "first"),
    )
    state = state.merge(
        inj,
        left_on=["season", "week", "team", "last_qb_id"],
        right_on=["season", "week", "team", "gsis_id"],
        how="left",
    )
    state["qb_out"] = (state["status_score"] < 0.70).astype(float)
    state["qb_questionable"] = (
        (state["status_score"] >= 0.70) & (state["status_score"] < 0.90)
    ).astype(float)
    state["qb_status_score"] = state["status_score"].fillna(1.0)
    return state[
        [
            "season",
            "week",
            "team",
            "last_qb_id",
            "last_qb_name",
            "last_qb_attempts",
            "qb_out",
            "qb_questionable",
            "qb_status_score",
            "report_status",
        ]
    ].rename(columns={"report_status": "qb_report_status"})


def build_elo(seasons: Iterable[int] = SEASONS) -> pd.DataFrame:
    """Pre-game Elo ratings for every regular-season game."""
    games = nfl.load_games(seasons)
    games = games[games.home_score.notna() & games.away_score.notna()].copy()
    games = games.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    ratings: dict[str, float] = {}
    rows = []
    current_season = None
    for row in games.itertuples(index=False):
        if current_season is None:
            current_season = row.season
        if row.season != current_season:
            # Regress toward the mean between seasons.
            for team in list(ratings):
                ratings[team] = 1500.0 + 0.67 * (ratings[team] - 1500.0)
            current_season = row.season

        home = row.home_team
        away = row.away_team
        h = ratings.get(home, 1500.0)
        a = ratings.get(away, 1500.0)
        rows.append(
            {
                "game_id": row.game_id,
                "season": row.season,
                "week": row.week,
                "home_team": home,
                "away_team": away,
                "home_elo": h,
                "away_elo": a,
                "elo_diff": h - a,
            }
        )

        # FiveThirtyEight-style margin-of-victory multiplier.
        margin = float(row.home_score - row.away_score)
        winner_elo = h if margin > 0 else a
        loser_elo = a if margin > 0 else h
        elo_diff = winner_elo - loser_elo
        mov = max(abs(margin), 1)
        multiplier = np.log(mov + 1.0) * 2.2 / ((elo_diff * 0.001) + 2.2)
        multiplier = min(multiplier, 3.0)
        expected_home = 1.0 / (1.0 + 10 ** ((a - h - 65.0) / 400.0))
        actual_home = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
        k = 20.0
        change = k * multiplier * (actual_home - expected_home)
        ratings[home] = h + change
        ratings[away] = a - change

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main team-game table
# ---------------------------------------------------------------------------
def build_team_games(seasons: Iterable[int] = SEASONS) -> pd.DataFrame:
    print("Loading team-week stats...")
    team = nfl.load_team_week(seasons)
    games = nfl.load_games(seasons)

    sched_cols = [
        "game_id",
        "season",
        "week",
        "gameday",
        "weekday",
        "gametime",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "spread_line",
        "total_line",
        "away_moneyline",
        "home_moneyline",
        "away_rest",
        "home_rest",
        "div_game",
        "roof",
        "surface",
        "temp",
        "wind",
        "away_qb_id",
        "home_qb_id",
        "away_qb_name",
        "home_qb_name",
        "stadium",
        "location",
        "referee",
    ]
    sched = games[[c for c in sched_cols if c in games.columns]].copy()
    df = team.merge(
        sched, on=["game_id", "season", "week"], how="left", suffixes=("", "_sched")
    )
    df["is_home"] = (df["team"] == df["home_team"]).astype(int)
    df["points_for"] = np.where(df["is_home"] == 1, df["home_score"], df["away_score"])
    df["points_allowed"] = np.where(
        df["is_home"] == 1, df["away_score"], df["home_score"]
    )
    df["margin"] = df["points_for"] - df["points_allowed"]
    df["total_points"] = df["points_for"] + df["points_allowed"]
    df["win"] = (df["margin"] > 0).astype(float)
    df.loc[df["margin"] == 0, "win"] = np.nan

    # Basic per-play efficiency from team stats (PBP will override where available).
    df["plays"] = (
        pd.to_numeric(df["attempts"], errors="coerce").fillna(0)
        + pd.to_numeric(df["sacks_suffered"], errors="coerce").fillna(0)
        + pd.to_numeric(df["carries"], errors="coerce").fillna(0)
    )
    df["off_pass_yards_play"] = _safe_div(df["passing_yards"], df["plays"])
    df["off_rush_yards_play"] = _safe_div(df["rushing_yards"], df["plays"])
    df["off_pass_td_rate"] = _safe_div(df["passing_tds"], df["attempts"])
    df["off_int_rate"] = _safe_div(df["passing_interceptions"], df["attempts"])
    df["off_sack_rate"] = _safe_div(
        df["sacks_suffered"], df["attempts"] + df["sacks_suffered"]
    )
    df["off_turnover_rate"] = _safe_div(
        pd.to_numeric(df["passing_interceptions"], errors="coerce").fillna(0)
        + pd.to_numeric(df["fumbles_lost_total"], errors="coerce").fillna(0),
        df["plays"],
    )
    df["off_cpoe"] = pd.to_numeric(df["passing_cpoe"], errors="coerce")
    df["def_takeaway_rate"] = _safe_div(
        pd.to_numeric(df["def_interceptions"], errors="coerce").fillna(0)
        + pd.to_numeric(df["fumble_recovery_opp"], errors="coerce").fillna(0),
        df["plays"],
    )
    df["turnover_margin"] = (
        pd.to_numeric(df["def_interceptions"], errors="coerce").fillna(0)
        + pd.to_numeric(df["fumble_recovery_opp"], errors="coerce").fillna(0)
        - pd.to_numeric(df["passing_interceptions"], errors="coerce").fillna(0)
        - pd.to_numeric(df["fumbles_lost_total"], errors="coerce").fillna(0)
    )
    df["fg_pct"] = pd.to_numeric(df["fg_pct"], errors="coerce")
    df["punt_net_yards_per_punt"] = _safe_div(df["pt_net_yards"], df["pt_att"])
    df["punt_return_yards_per_return"] = _safe_div(
        df["punt_return_yards"], df["punt_returns"]
    )
    df["kickoff_return_yards_per_return"] = _safe_div(
        df["kickoff_return_yards"], df["kickoff_returns"]
    )

    print("Aggregating play-by-play...")
    pbp_team = build_pbp_team_games(seasons)
    df = df.merge(pbp_team, on=["game_id", "season", "week", "team"], how="left")

    print("Aggregating player usage...")
    player_team = build_player_team_games(seasons)
    df = df.merge(player_team, on=["game_id", "season", "week", "team"], how="left")

    print("Aggregating coverage / pressure...")
    adv_team = build_advdef_team_games(seasons)
    if not adv_team.empty:
        df = df.merge(adv_team, on=["game_id", "season", "week", "team"], how="left")

    print("Aggregating snaps...")
    snap_team = build_snap_team_games(seasons)
    df = df.merge(snap_team, on=["game_id", "season", "week", "team"], how="left")

    # PBP-derived rates
    df["off_epa_play"] = pd.to_numeric(df["pbp_epa"], errors="coerce")
    df["off_pass_epa_db"] = pd.to_numeric(df["pbp_pass_epa"], errors="coerce")
    df["off_rush_epa_carry"] = pd.to_numeric(df["pbp_rush_epa"], errors="coerce")
    df["off_success_rate"] = pd.to_numeric(df["pbp_success"], errors="coerce")
    df["off_pass_success"] = pd.to_numeric(df["pbp_pass_success"], errors="coerce")
    df["off_rush_success"] = pd.to_numeric(df["pbp_rush_success"], errors="coerce")
    df["off_pass_rate"] = _safe_div(df["pbp_pass_plays"], df["pbp_plays"])
    df["off_early_down_pass_rate"] = _safe_div(
        df["pbp_early_pass"], df["pbp_early_plays"]
    )
    df["off_explosive_pass_rate"] = _safe_div(
        df["pbp_explosive_pass"], df["pbp_pass_plays"]
    )
    df["off_explosive_rush_rate"] = _safe_div(
        df["pbp_explosive_rush"], df["pbp_rush_plays"]
    )
    df["off_third_down_rate"] = _safe_div(df["pbp_third_conv"], df["pbp_third_plays"])
    df["off_penalty_yards_play"] = _safe_div(df["pbp_penalty_yards"], df["pbp_plays"])
    df["off_pass_yards_play"] = _safe_div(df["pbp_pass_yards"], df["pbp_plays"])
    df["off_rush_yards_play"] = _safe_div(df["pbp_rush_yards"], df["pbp_plays"])
    df["off_points_per_drive"] = _safe_div(df["points_for"], df["pbp_drives"])
    df["off_plays_per_drive"] = _safe_div(df["pbp_drive_plays"], df["pbp_drives"])
    df["off_seconds_per_play"] = _safe_div(
        df["pbp_drive_seconds"], df["pbp_drive_plays"]
    )
    df["off_time_of_possession"] = pd.to_numeric(
        df["pbp_drive_seconds"], errors="coerce"
    )
    df["off_red_zone_trips"] = pd.to_numeric(df["pbp_drive_inside20"], errors="coerce")
    df["off_red_zone_td_rate"] = _safe_div(df["pbp_drive_td"], df["pbp_drive_inside20"])
    df["off_red_zone_score_rate"] = _safe_div(
        df["pbp_drive_score"], df["pbp_drive_inside20"]
    )
    df["off_turnover_rate"] = _safe_div(df["pbp_turnovers"], df["pbp_plays"])
    df["off_sack_rate"] = _safe_div(df["pbp_sacks"], df["pbp_pass_plays"])
    df["off_qb_hit_rate"] = _safe_div(df["pbp_qb_hits"], df["pbp_pass_plays"])

    df["def_epa_play"] = pd.to_numeric(df["pbp_def_epa"], errors="coerce")
    df["def_pass_epa_db"] = pd.to_numeric(df["pbp_def_pass_epa"], errors="coerce")
    df["def_rush_epa_carry"] = pd.to_numeric(df["pbp_def_rush_epa"], errors="coerce")
    df["def_success_rate"] = pd.to_numeric(df["pbp_def_success"], errors="coerce")
    df["def_pass_success"] = pd.to_numeric(df["pbp_def_pass_success"], errors="coerce")
    df["def_rush_success"] = pd.to_numeric(df["pbp_def_rush_success"], errors="coerce")
    df["def_explosive_pass_rate"] = _safe_div(
        df["pbp_def_explosive_pass"], df["pbp_def_pass_plays"]
    )
    df["def_explosive_rush_rate"] = _safe_div(
        df["pbp_def_explosive_rush"], df["pbp_def_rush_plays"]
    )
    df["def_third_down_rate"] = _safe_div(
        df["pbp_def_third_conv"], df["pbp_def_third_plays"]
    )
    df["def_pass_yards_play"] = _safe_div(df["pbp_def_pass_yards"], df["pbp_def_plays"])
    df["def_rush_yards_play"] = _safe_div(df["pbp_def_rush_yards"], df["pbp_def_plays"])
    df["def_points_per_drive"] = _safe_div(df["points_allowed"], df["pbp_def_drives"])
    df["def_plays_per_drive"] = _safe_div(
        df["pbp_def_drive_plays"], df["pbp_def_drives"]
    )
    df["def_seconds_per_play"] = _safe_div(
        df["pbp_def_drive_seconds"], df["pbp_def_drive_plays"]
    )
    df["def_red_zone_trips"] = pd.to_numeric(
        df["pbp_def_drive_inside20"], errors="coerce"
    )
    df["def_red_zone_td_rate"] = _safe_div(
        df["pbp_def_drive_td"], df["pbp_def_drive_inside20"]
    )
    df["def_takeaway_rate"] = _safe_div(df["pbp_def_turnovers"], df["pbp_def_plays"])
    df["def_sack_rate"] = _safe_div(df["pbp_def_sacks"], df["pbp_def_pass_plays"])
    df["def_qb_hit_rate"] = _safe_div(df["pbp_def_qb_hits"], df["pbp_def_pass_plays"])

    df["pressure_rate"] = _safe_div(df["adv_pressures"], df["team_def_snaps"])
    df["pressure_rate_allowed"] = _safe_div(
        df.groupby("game_id")["adv_pressures"].transform("sum") - df["adv_pressures"],
        df["team_off_snaps"],
    )

    print("Computing Elo...")
    elo = build_elo(seasons)
    df = df.merge(
        elo[["game_id", "home_elo", "away_elo", "elo_diff"]], on="game_id", how="left"
    )
    df["elo"] = np.where(df["is_home"] == 1, df["home_elo"], df["away_elo"])
    df["opp_elo"] = np.where(df["is_home"] == 1, df["away_elo"], df["home_elo"])

    print(f"Team-games: {len(df):,} rows, {df.shape[1]} columns")
    return df.sort_values(["season", "week", "gameday", "game_id", "team"]).reset_index(
        drop=True
    )


if __name__ == "__main__":
    df = build_team_games()
    out = PROCESSED / "team_games.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
