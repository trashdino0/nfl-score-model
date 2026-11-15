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

# Candidate numeric features: exclude IDs/targets/market/post-game.
exclude = {
    "season",
    "week",
    "home_score",
    "away_score",
    "margin",
    "total",
    "home_win",
    "overtime",
    "result",
    "spread_line",
    "total_line",
    "home_implied",
    "away_implied",
    "market_margin",
    "market_total",
    "market_home_win_prob",
    "naive_margin",
    "naive_total",
    "naive_home_score",
    "naive_away_score",
    "home_moneyline",
    "away_moneyline",
    "home_spread_odds",
    "away_spread_odds",
    "over_odds",
    "under_odds",
}
cand = [
    c for c in f.columns if c not in exclude and pd.api.types.is_numeric_dtype(f[c])
]
print("candidates", len(cand))


def ridge(alpha=100):
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=alpha)),
        ]
    )


def run(target_resid, seasons, k, alpha, shrink, scope):
    rows = []
    for season in seasons:
        train = f[(f.season >= 2018) & (f.season < season)].copy()
        test = f[f.season == season].copy()
        if target_resid == "res_m":
            train["y"] = train.margin - train.market_margin
        else:
            train["y"] = train.total - train.market_total
        # correlation selection on train
        cors = train[cand].corrwith(train["y"]).abs().sort_values(ascending=False)
        cols = [c for c in cors.index[:k] if train[c].nunique(dropna=True) > 1]
        use = [c for c in cols if train[c].nunique(dropna=True) > 1]
        m = ridge(alpha)
        m.fit(train[use], train["y"])
        if target_resid == "res_m":
            pred = test.market_margin + shrink * m.predict(test[use])
            actual = test.margin
        else:
            pred = test.market_total + shrink * m.predict(test[use])
            actual = test.total
        rows.append(
            {
                "season": season,
                "mae": mean_absolute_error(actual, pred),
                "win": (
                    ((test.market_margin + shrink * m.predict(test[use])) > 0).astype(
                        float
                    )
                    == test.home_win
                ).mean()
                if target_resid == "res_m"
                else np.nan,
                "cols": ",".join(use[:8]),
            }
        )
    return pd.DataFrame(rows)


print("\n=== total residual, top-k correlation features ===")
for k in [5, 10, 15, 20, 30, 50, 80]:
    for alpha in [10, 30, 100, 300]:
        r = run("res_t", [2021, 2022, 2023, 2024, 2025], k, alpha, 0.25, "val")
        print(f"k={k:3d} a={alpha:3d} total_mae={r.mae.mean():.4f}")
print("\nBest k feature lists (last season):")
for k in [10, 20, 30]:
    r = run("res_t", [2025], k, 100, 0.25, "val")
    print(k, r.iloc[0].cols)

print("\n=== margin residual, top-k correlation features ===")
for k in [5, 10, 15, 20, 30, 50, 80]:
    for alpha in [10, 30, 100, 300]:
        r = run("res_m", [2021, 2022, 2023, 2024, 2025], k, alpha, 0.2, "val")
        print(
            f"k={k:3d} a={alpha:3d} margin_mae={r.mae.mean():.4f} win={r.win.mean():.4f}"
        )
