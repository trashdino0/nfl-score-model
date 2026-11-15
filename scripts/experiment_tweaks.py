import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

pd.set_option("display.width", 240)

f = pd.read_parquet("data/processed/game_features.parquet")
f = f[f.home_score.notna()].copy()

MATCHUP = [c for c in f.columns if c.endswith("_net")]
MATCHUP += [
    f"{lab}_{side}"
    for lab in [
        "epa",
        "pass_epa",
        "rush_epa",
        "qb_vs_pass",
        "qb_cpoe_vs_coverage",
        "wr1_vs_coverage",
        "wr_corps_vs_coverage",
        "rb1_vs_run",
        "pass_rush_vs_protection",
        "turnovers",
        "red_zone",
        "points_drive",
    ]
    for side in ["home_adv", "away_adv"]
    if f"{lab}_{side}" in f.columns
]
MATCHUP += [c for c in f.columns if c.endswith(("_roll8_diff", "_prev_diff"))]
TEAM = [
    c
    for c in f.columns
    if c.endswith("_diff") and not c.startswith(("inj_", "qb_", "start_qb_"))
]
QB = [
    c
    for c in f.columns
    if c.startswith("start_qb_")
    or c
    in [
        "home_qb_out",
        "away_qb_out",
        "qb_out_diff",
        "qb_questionable_diff",
        "home_qb_change",
        "away_qb_change",
        "qb_change_diff",
        "home_start_qb_is_new",
        "away_start_qb_is_new",
        "start_qb_new_diff",
    ]
]
INJ = [c for c in f.columns if c.startswith("inj_") and c.endswith("_diff")]
INJ += ["inj_lost_off_sum", "inj_lost_def_sum"]
INJ += [
    c
    for c in f.columns
    if c.endswith(("_out_diff", "_questionable_diff"))
    and c.split("_")[0] in ["wr", "rb", "te"]
]
SIT = [
    c
    for c in [
        "rest_diff",
        "short_week_diff",
        "is_dome",
        "is_turf",
        "home_altitude",
        "div_game",
        "wind",
        "temp",
    ]
    if c in f.columns
]
ELO = ["elo_diff"]
SUM = [c for c in f.columns if c.endswith("_sum")]
BASE_M = list(dict.fromkeys(MATCHUP + QB + INJ))
BASE_T = list(dict.fromkeys(MATCHUP + TEAM + QB + INJ + SIT + ELO))
PURE = list(dict.fromkeys(BASE_T + SUM))


def ridge(alpha=100):
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=alpha)),
        ]
    )


def add_interactions(train, test, cols):
    interactions = []
    pairs = [
        ("qb_out_diff", "start_qb_epa_diff"),
        ("qb_change_diff", "start_qb_epa_diff"),
        ("inj_lost_off_diff", "off_epa_play_diff"),
        ("inj_lost_def_diff", "def_epa_play_diff"),
        ("pass_rush_vs_protection_net", "qb_out_diff"),
        ("wind", "pass_epa_net"),
        ("temp", "pass_epa_net"),
    ]
    for a, b in pairs:
        if a in cols and b in cols:
            name = f"{a}_x_{b}"
            train[name] = train[a] * train[b]
            test[name] = test[a] * test[b]
            interactions.append(name)
    return cols + interactions


configs = {
    "base_m": (BASE_M, "margin"),
    "base_m+mp": (BASE_M + ["market_home_win_prob"], "margin"),
    "base_m+inter": (BASE_M, "margin_inter"),
    "base_t": (BASE_T, "total"),
    "base_t+mp": (BASE_T + ["market_home_win_prob"], "total"),
    "base_t+inter": (BASE_T, "total_inter"),
    "pure": (PURE, "pure"),
}

seasons = [2021, 2022, 2023, 2024, 2025]
rows = []
for name, (cols, kind) in configs.items():
    preds = []
    for season in seasons:
        train = f[(f.season >= 2018) & (f.season < season)].copy()
        test = f[f.season == season].copy()
        use = [c for c in cols if train[c].nunique(dropna=True) > 1]
        if "inter" in kind:
            use = add_interactions(train, test, use)
        if "pure" in kind:
            for target, prefix in [("margin", "rm"), ("total", "rt")]:
                m = ridge()
                m.fit(train[use], train[target])
                test[prefix] = m.predict(test[use])
        else:
            train["res_m"] = train.margin - train.market_margin
            train["res_t"] = train.total - train.market_total
            for target, prefix in [("res_m", "rm"), ("res_t", "rt")]:
                m = ridge()
                m.fit(train[use], train[target])
                test[prefix] = m.predict(test[use])
        preds.append(
            test[
                [
                    "game_id",
                    "season",
                    "week",
                    "margin",
                    "total",
                    "home_win",
                    "market_margin",
                    "market_total",
                    "rm",
                    "rt",
                ]
            ]
        )
    p = pd.concat(preds, ignore_index=True)
    if "pure" in kind:
        pm = p.rm
        pt = p.rt
        ms = 0
        ts = 0
    elif "_m" in name or name == "base_m" or name.startswith("base_m"):
        pm = p.market_margin + 0.2 * p.rm
        pt = p.market_total
        ms = 0.2
        ts = 0
    else:
        pm = p.market_margin
        pt = p.market_total + 0.25 * p.rt
        ms = 0
        ts = 0.25
    v = p.home_win.notna()
    rows.append(
        {
            "config": name,
            "n_features": len(cols),
            "ms": ms,
            "ts": ts,
            "margin_mae": mean_absolute_error(p.margin, pm),
            "total_mae": mean_absolute_error(p.total, pt),
            "win": ((pm > 0).astype(float)[v] == p.loc[v, "home_win"]).mean(),
        }
    )
print(pd.DataFrame(rows).round(4).to_string(index=False))
