import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, brier_score_loss, log_loss

pd.set_option("display.width", 260)

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
GROUPS = {
    "matchup": MATCHUP,
    "team": TEAM,
    "qb": QB,
    "inj": INJ,
    "sit": SIT,
    "elo": ELO,
}

SETS = {
    "matchup": ["matchup"],
    "m_qb_inj": ["matchup", "qb", "inj"],
    "m_all": ["matchup", "team", "qb", "inj", "sit", "elo"],
}
SETS = {
    k: list(dict.fromkeys([c for g in v for c in GROUPS[g]])) for k, v in SETS.items()
}


def ridge(alpha):
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=alpha)),
        ]
    )


def logit():
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", LogisticRegression(C=0.1, max_iter=2000)),
        ]
    )


seasons = [2021, 2022, 2023, 2024, 2025]
# Build residuals per season for each feature set / alpha
store = {}
for sname, cols in SETS.items():
    for alpha in [30, 100, 300]:
        preds = []
        for season in seasons:
            train = f[(f.season >= 2018) & (f.season < season)].copy()
            test = f[f.season == season].copy()
            train["res_m"] = train.margin - train.market_margin
            train["res_t"] = train.total - train.market_total
            use = [c for c in cols if train[c].nunique(dropna=True) > 1]
            for target, name in [("res_m", "rm"), ("res_t", "rt")]:
                m = ridge(alpha)
                m.fit(train[use], train[target])
                test[name] = m.predict(test[use])
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
                        "market_home_win_prob",
                        "rm",
                        "rt",
                    ]
                ]
            )
        store[(sname, alpha)] = pd.concat(preds, ignore_index=True)


def eval_blend(p, ms, ts):
    pm = p.market_margin + ms * p.rm
    pt = p.market_total + ts * p.rt
    v = p.home_win.notna()
    return (
        mean_absolute_error(p.margin, pm),
        mean_absolute_error(p.total, pt),
        ((pm > 0).astype(float)[v] == p.loc[v, "home_win"]).mean(),
    )


print("=== single configs ===")
rows = []
for (sname, alpha), p in store.items():
    for ms in [0.15, 0.2, 0.25, 0.3, 0.35]:
        for ts in [0.0, 0.1, 0.2, 0.25]:
            ma, ta, wa = eval_blend(p, ms, ts)
            rows.append(
                {
                    "set": sname,
                    "alpha": alpha,
                    "ms": ms,
                    "ts": ts,
                    "margin_mae": ma,
                    "total_mae": ta,
                    "win": wa,
                }
            )
r = pd.DataFrame(rows)
print(r.sort_values("margin_mae").head(10).round(4).to_string(index=False))
print(r.sort_values("total_mae").head(10).round(4).to_string(index=False))
print(r.sort_values("win", ascending=False).head(10).round(4).to_string(index=False))

# Ensemble average of residuals across sets/alphas
print("\n=== ensembles ===")
rows = []
for combo_name, keys in [
    ("m_all_a100", [("m_all", 100)]),
    ("m_qb_inj_a100", [("m_qb_inj", 100)]),
    ("matchup_a100", [("matchup", 100)]),
    ("avg_m_all_m_qb", [("m_all", 100), ("m_qb_inj", 100)]),
    ("avg_3", [("m_all", 100), ("m_qb_inj", 100), ("matchup", 100)]),
    (
        "avg_6",
        [
            ("m_all", 100),
            ("m_qb_inj", 100),
            ("matchup", 100),
            ("m_all", 30),
            ("m_qb_inj", 30),
            ("matchup", 300),
        ],
    ),
]:
    p = store[keys[0]].copy()
    p["rm"] = np.mean([store[k].rm.values for k in keys], axis=0)
    p["rt"] = np.mean([store[k].rt.values for k in keys], axis=0)
    for ms in [0.15, 0.2, 0.25, 0.3, 0.35, 0.5]:
        for ts in [0.0, 0.1, 0.2, 0.25]:
            ma, ta, wa = eval_blend(p, ms, ts)
            rows.append(
                {
                    "combo": combo_name,
                    "ms": ms,
                    "ts": ts,
                    "margin_mae": ma,
                    "total_mae": ta,
                    "win": wa,
                }
            )
r = pd.DataFrame(rows)
print(r.sort_values("margin_mae").head(15).round(4).to_string(index=False))
print(r.sort_values("total_mae").head(10).round(4).to_string(index=False))
print(r.sort_values("win", ascending=False).head(10).round(4).to_string(index=False))

# Direct win classifier on pure stats, and blend with market prob
print("\n=== win probability ===")
prob_rows = []
for season in seasons:
    train = f[(f.season >= 2018) & (f.season < season)].copy()
    train = train[train.home_win.notna()]
    test = f[f.season == season].copy()
    test = test[test.home_win.notna()]
    cols = SETS["m_qb_inj"]
    use = [c for c in cols if train[c].nunique(dropna=True) > 1]
    clf = logit()
    clf.fit(train[use], train.home_win)
    test["p_logit"] = clf.predict_proba(test[use])[:, 1]
    p = store[("m_qb_inj", 100)]
    p = p[(p.season == season) & p.home_win.notna()].copy()
    p = p.merge(test[["game_id", "p_logit"]], on="game_id", how="left")
    # margin prob with sigma from train residuals
    pm = p.market_margin + 0.2 * p.rm
    sigma = np.std(train.margin - train.market_margin)
    from math import erf, sqrt

    p["p_margin"] = [0.5 * (1 + erf((x / sigma) / sqrt(2))) for x in pm]
    p["p_market"] = p.market_home_win_prob
    p["p_blend"] = 0.5 * p.p_market + 0.5 * p.p_margin
    p["p_blend_logit"] = 0.4 * p.p_market + 0.3 * p.p_margin + 0.3 * p.p_logit
    for name in ["p_market", "p_margin", "p_logit", "p_blend", "p_blend_logit"]:
        prob_rows.append(
            {
                "season": season,
                "model": name,
                "brier": brier_score_loss(p.home_win, p[name]),
                "logloss": log_loss(p.home_win, p[name]),
            }
        )
pr = pd.DataFrame(prob_rows)
print(pr.groupby("model")[["brier", "logloss"]].mean().round(4).to_string())
