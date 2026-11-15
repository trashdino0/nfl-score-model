import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

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
MARKET = [
    "market_margin",
    "market_total",
    "market_home_win_prob",
    "spread_line",
    "total_line",
]

GROUPS = {
    "matchup": MATCHUP,
    "team": TEAM,
    "qb": QB,
    "inj": INJ,
    "sit": SIT,
    "elo": ELO,
}
GROUPS = {
    k: list(dict.fromkeys([c for c in v if c in f.columns])) for k, v in GROUPS.items()
}
for k, v in GROUPS.items():
    print(k, len(v))

FEATURE_SETS = {
    "matchup": ["matchup"],
    "matchup+qb": ["matchup", "qb"],
    "matchup+qb+inj": ["matchup", "qb", "inj"],
    "matchup+all": ["matchup", "team", "qb", "inj", "sit", "elo"],
}


def ridge(alpha):
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=alpha)),
        ]
    )


def hgb():
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )


def get_cols(gs):
    cols = []
    for g in gs:
        cols += GROUPS[g]
    return list(dict.fromkeys(cols))


seasons = [2021, 2022, 2023, 2024, 2025]
rows = []
for set_name, gs in FEATURE_SETS.items():
    cols = get_cols(gs)
    for model_name, maker in [("ridge", ridge), ("hgb", hgb)]:
        for alpha in [10, 30, 100, 300] if model_name == "ridge" else [0]:
            preds = []
            for season in seasons:
                train = f[(f.season >= 2018) & (f.season < season)].copy()
                test = f[f.season == season].copy()
                train["res_m"] = train.margin - train.market_margin
                train["res_t"] = train.total - train.market_total
                use = [c for c in cols if train[c].nunique(dropna=True) > 1]
                for target, prefix in [("res_m", "rm"), ("res_t", "rt")]:
                    m = maker(alpha) if model_name == "ridge" else maker()
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
            for ms in [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.5]:
                for ts in [0.0, 0.1, 0.2, 0.25]:
                    pm = p.market_margin + ms * p.rm
                    pt = p.market_total + ts * p.rt
                    valid = p.home_win.notna()
                    rows.append(
                        {
                            "set": set_name,
                            "model": model_name,
                            "alpha": alpha,
                            "ms": ms,
                            "ts": ts,
                            "margin_mae": mean_absolute_error(p.margin, pm),
                            "total_mae": mean_absolute_error(p.total, pt),
                            "win_acc": (
                                (pm > 0).astype(float)[valid]
                                == p.loc[valid, "home_win"]
                            ).mean(),
                            "n": len(p),
                        }
                    )
res = pd.DataFrame(rows)
print("\n=== best margin configs (all total shrinks) ===")
print(
    res.groupby(["set", "model", "alpha", "ms"])
    .agg(
        margin_mae=("margin_mae", "mean"),
        total_mae=("total_mae", "mean"),
        win_acc=("win_acc", "mean"),
    )
    .reset_index()
    .sort_values("margin_mae")
    .head(20)
    .round(4)
    .to_string(index=False)
)
print("\n=== best total configs ===")
print(
    res.groupby(["set", "model", "alpha", "ts"])
    .agg(
        margin_mae=("margin_mae", "mean"),
        total_mae=("total_mae", "mean"),
        win_acc=("win_acc", "mean"),
    )
    .reset_index()
    .sort_values("total_mae")
    .head(20)
    .round(4)
    .to_string(index=False)
)
print("\n=== best win acc ===")
print(
    res.groupby(["set", "model", "alpha", "ms", "ts"])
    .agg(
        margin_mae=("margin_mae", "mean"),
        total_mae=("total_mae", "mean"),
        win_acc=("win_acc", "mean"),
    )
    .reset_index()
    .sort_values("win_acc", ascending=False)
    .head(20)
    .round(4)
    .to_string(index=False)
)

# market baselines
base = []
for season in seasons:
    t = f[f.season == season]
    v = t.home_win.notna()
    base.append(
        {
            "margin_mae": mean_absolute_error(t.margin, t.market_margin),
            "total_mae": mean_absolute_error(t.total, t.market_total),
            "win_acc": (
                (t.market_margin > 0).astype(float)[v] == t.loc[v, "home_win"]
            ).mean(),
        }
    )
print("\nMarket baseline:", pd.DataFrame(base).mean().round(4).to_dict())
res.to_csv("outputs/tune_validation.csv", index=False)
