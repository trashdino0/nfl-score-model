import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from nflscore.features import DIFF_METRICS, SUM_METRICS, MATCHUP_PAIRS

pd.set_option("display.width", 240)

f = pd.read_parquet("data/processed/game_features.parquet")
f = f[f.home_score.notna()].copy()

MARKET = [
    "spread_line",
    "total_line",
    "home_implied",
    "away_implied",
    "market_margin",
    "market_total",
    "market_home_win_prob",
]


def feature_cols(df, market=False):
    cols = []
    # all matchup nets
    cols += [c for c in df.columns if c.endswith("_net")]
    # home/away advantages for key matchups
    key_labels = [
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
    for lab in key_labels:
        for side in ["home_adv", "away_adv"]:
            c = f"{lab}_{side}"
            if c in df.columns:
                cols.append(c)
    cols += [c for c in df.columns if c.endswith("_roll8_diff")]
    cols += [c for c in df.columns if c.endswith("_prev_diff")]
    cols += [f"{m}_diff" for m in DIFF_METRICS if f"{m}_diff" in df.columns]
    cols += [f"{m}_sum" for m in SUM_METRICS if f"{m}_sum" in df.columns]
    cols += [c for c in df.columns if c.startswith("inj_") and c.endswith("_diff")]
    cols += [c for c in ["inj_lost_off_sum", "inj_lost_def_sum"] if c in df.columns]
    for grp in ["wr", "rb", "te"]:
        cols += [
            c
            for c in [f"{grp}_out_diff", f"{grp}_questionable_diff"]
            if c in df.columns
        ]
    cols += [
        c
        for c in [
            "home_qb_out",
            "away_qb_out",
            "qb_out_diff",
            "qb_questionable_diff",
            "home_qb_change",
            "away_qb_change",
            "qb_change_diff",
            "start_qb_epa_diff",
            "start_qb_cpoe_diff",
            "start_qb_int_diff",
            "start_qb_sack_diff",
            "start_qb_attempts_diff",
            "start_qb_career_diff",
            "start_qb_new_diff",
            "home_start_qb_is_new",
            "away_start_qb_is_new",
        ]
        if c in df.columns
    ]
    cols += [
        c
        for c in [
            "elo_diff",
            "rest_diff",
            "short_week_diff",
            "is_dome",
            "is_turf",
            "home_altitude",
            "div_game",
            "wind",
            "temp",
        ]
        if c in df.columns
    ]
    cols = list(dict.fromkeys(cols))
    if market:
        cols += [c for c in MARKET if c in df.columns]
    return list(dict.fromkeys(cols))


pure_cols = feature_cols(f, market=False)
hyb_cols = feature_cols(f, market=True)
print("pure features", len(pure_cols), "hybrid", len(hyb_cols))


def ridge(alpha=30.0):
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
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )


def fit_predict(train, test, cols, target, model):
    # HistGradientBoosting chokes on constant columns in this sklearn version.
    use = [c for c in cols if train[c].nunique(dropna=True) > 1]
    m = model
    m.fit(train[use], train[target])
    return m.predict(test[use])


def evaluate(test, pred_m, pred_t, name):
    pred_home = (pred_t + pred_m) / 2
    pred_away = (pred_t - pred_m) / 2
    pred_win = (pred_m > 0).astype(float)
    valid = test.home_win.notna()
    acc = (pred_win[valid] == test.loc[valid, "home_win"]).mean()
    return {
        "model": name,
        "margin_mae": mean_absolute_error(test.margin, pred_m),
        "total_mae": mean_absolute_error(test.total, pred_t),
        "home_mae": mean_absolute_error(test.home_score, pred_home),
        "away_mae": mean_absolute_error(test.away_score, pred_away),
        "win_acc": acc,
        "n": int(valid.sum()),
    }


rows = []
for season in [2021, 2022, 2023, 2024, 2025]:
    train = f[(f.season >= 2018) & (f.season < season)]
    test = f[f.season == season].copy()
    print(f"\nseason {season}: train {len(train)}, test {len(test)}")

    # Market / naive
    rows.append(
        evaluate(test, test.market_margin, test.market_total, f"{season} market")
    )
    rows.append(evaluate(test, test.naive_margin, test.naive_total, f"{season} naive"))

    # Pure ridge
    pm = fit_predict(train, test, pure_cols, "margin", ridge(30))
    pt = fit_predict(train, test, pure_cols, "total", ridge(30))
    rows.append(evaluate(test, pm, pt, f"{season} pure_ridge"))

    # Pure HGB
    pm = fit_predict(train, test, pure_cols, "margin", hgb())
    pt = fit_predict(train, test, pure_cols, "total", hgb())
    rows.append(evaluate(test, pm, pt, f"{season} pure_hgb"))

    # Hybrid ridge
    pm = fit_predict(train, test, hyb_cols, "margin", ridge(30))
    pt = fit_predict(train, test, hyb_cols, "total", ridge(30))
    rows.append(evaluate(test, pm, pt, f"{season} hybrid_ridge"))

    # Hybrid HGB
    pm = fit_predict(train, test, hyb_cols, "margin", hgb())
    pt = fit_predict(train, test, hyb_cols, "total", hgb())
    rows.append(evaluate(test, pm, pt, f"{season} hybrid_hgb"))

    # Residual ridge: market + shrink * predicted residual
    rm = fit_predict(train, test, pure_cols, "margin", ridge(30))
    # Note: target is residual, so need a residual target in train
    tr = train.copy()
    tr["res_m"] = tr.margin - tr.market_margin
    tr["res_t"] = tr.total - tr.market_total
    rm = fit_predict(tr, test, pure_cols, "res_m", ridge(30))
    rt = fit_predict(tr, test, pure_cols, "res_t", ridge(30))
    for shrink in [0.25, 0.5, 0.75]:
        pm = test.market_margin + shrink * rm
        pt = test.market_total + shrink * rt
        rows.append(evaluate(test, pm, pt, f"{season} residual_ridge_s{shrink}"))

res = pd.DataFrame(rows)
print("\n=== Season-level results ===")
print(res.to_string(index=False))

print("\n=== Averages by model ===")
res["base"] = res.model.str.replace(r"^\d+ ", "", regex=True)
avg = (
    res.groupby("base")[["margin_mae", "total_mae", "home_mae", "away_mae", "win_acc"]]
    .mean()
    .sort_values("margin_mae")
)
print(avg.round(3).to_string())

print("\n=== Pooled ===")
# Re-run one pooled set of predictions for best candidates to get pooled winner accuracy
all_preds = []
for season in [2021, 2022, 2023, 2024, 2025]:
    train = f[(f.season >= 2018) & (f.season < season)]
    test = f[f.season == season].copy()
    tr = train.copy()
    tr["res_m"] = tr.margin - tr.market_margin
    tr["res_t"] = tr.total - tr.market_total
    out = test[
        [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "margin",
            "total",
            "home_win",
            "market_margin",
            "market_total",
            "market_home_win_prob",
            "naive_margin",
            "naive_total",
        ]
    ].copy()
    out["pure_ridge_m"] = fit_predict(train, test, pure_cols, "margin", ridge(30))
    out["pure_ridge_t"] = fit_predict(train, test, pure_cols, "total", ridge(30))
    out["hybrid_ridge_m"] = fit_predict(train, test, hyb_cols, "margin", ridge(30))
    out["hybrid_ridge_t"] = fit_predict(train, test, hyb_cols, "total", ridge(30))
    out["resid_m"] = fit_predict(tr, test, pure_cols, "res_m", ridge(30))
    out["resid_t"] = fit_predict(tr, test, pure_cols, "res_t", ridge(30))
    all_preds.append(out)
p = pd.concat(all_preds, ignore_index=True)
for name, m, t in [
    ("market", p.market_margin, p.market_total),
    ("naive", p.naive_margin, p.naive_total),
    ("pure_ridge", p.pure_ridge_m, p.pure_ridge_t),
    ("hybrid_ridge", p.hybrid_ridge_m, p.hybrid_ridge_t),
    ("resid_s0.5", p.market_margin + 0.5 * p.resid_m, p.market_total + 0.5 * p.resid_t),
    (
        "resid_s0.75",
        p.market_margin + 0.75 * p.resid_m,
        p.market_total + 0.75 * p.resid_t,
    ),
]:
    valid = p.home_win.notna()
    acc = ((m > 0).astype(float)[valid] == p.loc[valid, "home_win"]).mean()
    print(
        f"{name:16s} margin {mean_absolute_error(p.margin, m):6.2f} total {mean_absolute_error(p.total, t):6.2f} win {acc:.4f}"
    )
