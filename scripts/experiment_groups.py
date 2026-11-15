import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

pd.set_option("display.width", 260)

f = pd.read_parquet("data/processed/game_features.parquet")
f = f[f.home_score.notna()].copy()


def feat(pred):
    return [c for c in f.columns if c.endswith(pred)]


groups = {
    "elo": ["elo_diff"],
    "matchup": [c for c in f.columns if c.endswith("_net")]
    + [c for c in f.columns if c.endswith(("_roll8_diff", "_prev_diff"))],
    "team_diff": [
        c
        for c in f.columns
        if c.endswith("_diff") and not c.startswith(("inj_", "qb_", "start_qb_"))
    ],
    "qb": [
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
    ],
    "injury": [c for c in f.columns if c.startswith("inj_") and c.endswith("_diff")]
    + ["inj_lost_off_sum", "inj_lost_def_sum"]
    + [
        c
        for c in f.columns
        if c.endswith(("_out_diff", "_questionable_diff"))
        and c.split("_")[0] in ["wr", "rb", "te"]
    ],
    "situational": [
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
    ],
    "market": ["spread_line", "total_line", "market_home_win_prob"],
}
for k, v in groups.items():
    groups[k] = list(dict.fromkeys([c for c in v if c in f.columns]))
    print(k, len(groups[k]))

configs = {
    "elo": ["elo"],
    "qb": ["qb"],
    "injury": ["injury"],
    "situational": ["situational"],
    "elo+qb": ["elo", "qb"],
    "elo+qb+inj": ["elo", "qb", "injury"],
    "elo+qb+inj+sit": ["elo", "qb", "injury", "situational"],
    "matchup": ["matchup"],
    "matchup+elo+qb+inj+sit": [
        "matchup",
        "team_diff",
        "elo",
        "qb",
        "injury",
        "situational",
    ],
}


def ridge(alpha):
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=alpha)),
        ]
    )


def run(cols, alpha, shrink):
    rows = []
    for season in [2021, 2022, 2023, 2024, 2025]:
        train = f[(f.season >= 2018) & (f.season < season)].copy()
        test = f[f.season == season].copy()
        train["res_m"] = train.margin - train.market_margin
        train["res_t"] = train.total - train.market_total
        use = [c for c in cols if train[c].nunique(dropna=True) > 1]
        if not use:
            return None
        mm = ridge(alpha)
        mm.fit(train[use], train.res_m)
        tm = ridge(alpha)
        tm.fit(train[use], train.res_t)
        pm = test.market_margin + shrink * mm.predict(test[use])
        pt = test.market_total + shrink * tm.predict(test[use])
        valid = test.home_win.notna()
        rows.append(
            {
                "margin_mae": mean_absolute_error(test.margin, pm),
                "total_mae": mean_absolute_error(test.total, pt),
                "win_acc": (
                    (pm > 0).astype(float)[valid] == test.loc[valid, "home_win"]
                ).mean(),
            }
        )
    return pd.DataFrame(rows).mean()


results = []
for name, gs in configs.items():
    cols = []
    for g in gs:
        cols += groups[g]
    cols = list(dict.fromkeys(cols))
    for alpha in [10, 30, 100, 300]:
        for shrink in [0.1, 0.25, 0.5]:
            r = run(cols, alpha, shrink)
            if r is not None:
                results.append(
                    {
                        "config": name,
                        "n_features": len(cols),
                        "alpha": alpha,
                        "shrink": shrink,
                        **r.to_dict(),
                    }
                )
res = pd.DataFrame(results)
print("\nTop 25 by margin MAE:")
print(res.sort_values("margin_mae").head(25).round(4).to_string(index=False))
print("\nTop 15 by total MAE:")
print(res.sort_values("total_mae").head(15).round(4).to_string(index=False))
print("\nTop 15 by win acc:")
print(
    res.sort_values("win_acc", ascending=False).head(15).round(4).to_string(index=False)
)

# Baselines
base = []
for season in [2021, 2022, 2023, 2024, 2025]:
    test = f[f.season == season].copy()
    valid = test.home_win.notna()
    base.append(
        {
            "market_margin_mae": mean_absolute_error(test.margin, test.market_margin),
            "market_total_mae": mean_absolute_error(test.total, test.market_total),
            "market_win_acc": (
                (test.market_margin > 0).astype(float)[valid]
                == test.loc[valid, "home_win"]
            ).mean(),
        }
    )
print("\nMarket baseline avg:", pd.DataFrame(base).mean().round(4).to_dict())
