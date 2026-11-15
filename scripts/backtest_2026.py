import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    brier_score_loss,
    log_loss,
)

from nflscore.features import DIFF_METRICS, SUM_METRICS

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 80)

f = pd.read_parquet("data/processed/game_features.parquet")
f = f[f.home_score.notna()].copy()


def feature_cols(df, market=False):
    cols = []
    cols += [c for c in df.columns if c.endswith("_net")]
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
        cols += [
            "spread_line",
            "total_line",
            "home_implied",
            "away_implied",
            "market_margin",
            "market_total",
            "market_home_win_prob",
        ]
    return list(dict.fromkeys(cols))


cols = feature_cols(f)
print("features", len(cols))


def ridge(alpha=30):
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=alpha)),
        ]
    )


def fit_pred(train, test, cols, target, model):
    use = [c for c in cols if train[c].nunique(dropna=True) > 1]
    model.fit(train[use], train[target])
    return model.predict(test[use]), use


def run_week(target_week, alpha=30, shrink=0.25):
    train = f[
        (f.season >= 2018)
        & ((f.season < 2026) | ((f.season == 2026) & (f.week < target_week)))
    ].copy()
    test = f[(f.season == 2026) & (f.week == target_week)].copy()
    if len(test) == 0:
        return None
    tr = train.copy()
    tr["res_m"] = tr.margin - tr.market_margin
    tr["res_t"] = tr.total - tr.market_total

    # Pure model
    pure_m, use = fit_pred(train, test, cols, "margin", ridge(alpha))
    pure_t, _ = fit_pred(train, test, cols, "total", ridge(alpha))
    # Residual model
    res_m, _ = fit_pred(tr, test, cols, "res_m", ridge(alpha))
    res_t, _ = fit_pred(tr, test, cols, "res_t", ridge(alpha))

    out = test[
        [
            "game_id",
            "season",
            "week",
            "gameday",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "margin",
            "total",
            "home_win",
            "spread_line",
            "total_line",
            "market_margin",
            "market_total",
            "market_home_win_prob",
            "naive_margin",
            "naive_total",
            "home_qb_out",
            "away_qb_out",
            "home_qb_change",
            "away_qb_change",
            "home_qb_name",
            "away_qb_name",
        ]
    ].copy()
    out["pure_margin"] = pure_m
    out["pure_total"] = pure_t
    out["resid_margin"] = out["market_margin"] + shrink * res_m
    out["resid_total"] = out["market_total"] + shrink * res_t
    out["blend_margin"] = out["market_margin"] + shrink * (
        pure_m - out["market_margin"]
    )
    out["blend_total"] = out["market_total"] + shrink * (pure_t - out["market_total"])
    out["pure_home_score"] = (out.pure_total + out.pure_margin) / 2
    out["pure_away_score"] = (out.pure_total - out.pure_margin) / 2
    out["resid_home_score"] = (out.resid_total + out.resid_margin) / 2
    out["resid_away_score"] = (out.resid_total - out.resid_margin) / 2
    return out


all_weeks = []
for week in range(1, 9):
    out = run_week(week)
    all_weeks.append(out)
    print(f"week {week}: {len(out)} games")
bt = pd.concat(all_weeks, ignore_index=True)


def metrics(df, mcol, tcol, name):
    df = df[df.home_win.notna()].copy()
    pm, pt = df[mcol], df[tcol]
    ph = (pt + pm) / 2
    pa = (pt - pm) / 2
    pred_win = (pm > 0).astype(float)
    acc = (pred_win == df.home_win).mean()
    # ATS
    ats_pick_home = pm > df.market_margin
    ats_push = pm == df.market_margin
    ats_win = (ats_pick_home & (df.margin > df.market_margin)) | (
        ~ats_pick_home & (df.margin < df.market_margin)
    )
    ats_win = ats_win & ~ats_push
    ats_n = (~ats_push).sum()
    ats_rate = ats_win.sum() / ats_n if ats_n else np.nan
    # O/U
    ou_over = pt > df.market_total
    ou_push = pt == df.market_total
    ou_win = (
        (ou_over & (df.total > df.market_total))
        | (~ou_over & (df.total < df.market_total))
    ) & ~ou_push
    ou_n = (~ou_push).sum()
    ou_rate = ou_win.sum() / ou_n if ou_n else np.nan
    # Winner Brier using normal CDF sigma 13.5
    from math import erf, sqrt

    prob = np.array([0.5 * (1 + erf((m / 13.5) / sqrt(2))) for m in pm])
    brier = np.mean((prob - df.home_win) ** 2)
    ll = -np.mean(
        df.home_win * np.log(np.clip(prob, 1e-6, 1 - 1e-6))
        + (1 - df.home_win) * np.log(np.clip(1 - prob, 1e-6, 1 - 1e-6))
    )
    return {
        "model": name,
        "margin_mae": mean_absolute_error(df.margin, pm),
        "margin_rmse": mean_squared_error(df.margin, pm) ** 0.5,
        "total_mae": mean_absolute_error(df.total, pt),
        "total_rmse": mean_squared_error(df.total, pt) ** 0.5,
        "home_mae": mean_absolute_error(df.home_score, ph),
        "away_mae": mean_absolute_error(df.away_score, pa),
        "score_mae": mean_absolute_error(
            pd.concat([df.home_score, df.away_score]), pd.concat([ph, pa])
        ),
        "win_acc": acc,
        "ats_rate": ats_rate,
        "ou_rate": ou_rate,
        "brier": brier,
        "logloss": ll,
        "n": len(df),
    }


rows = []
for name, m, t in [
    ("market", "market_margin", "market_total"),
    ("naive", "naive_margin", "naive_total"),
    ("pure", "pure_margin", "pure_total"),
    ("blend_0.25", "blend_margin", "blend_total"),
    ("resid_0.25", "resid_margin", "resid_total"),
]:
    rows.append(metrics(bt, m, t, name))
summary = pd.DataFrame(rows)
print("\n=== 2026 Weeks 1-8 pooled ===")
print(summary.round(3).to_string(index=False))

# Per-week winner accuracy / margin MAE
print("\n=== Per-week: market vs pure vs resid ===")
per = []
for week in range(1, 9):
    d = bt[bt.week == week]
    row = {"week": week, "n": len(d)}
    for name, m, t in [
        ("market", "market_margin", "market_total"),
        ("pure", "pure_margin", "pure_total"),
        ("resid", "resid_margin", "resid_total"),
    ]:
        dd = d[d.home_win.notna()]
        row[f"{name}_acc"] = ((dd[m] > 0).astype(float) == dd.home_win).mean()
        row[f"{name}_m_mae"] = mean_absolute_error(dd.margin, dd[m])
        row[f"{name}_t_mae"] = mean_absolute_error(dd.total, dd[t])
    per.append(row)
print(pd.DataFrame(per).round(3).to_string(index=False))

bt.to_parquet("outputs/backtest_2026_raw.parquet", index=False)
print("\nwrote outputs/backtest_2026_raw.parquet")
print("\nSample predictions:")
print(
    bt[
        [
            "week",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "market_margin",
            "pure_margin",
            "resid_margin",
            "market_total",
            "pure_total",
            "resid_total",
        ]
    ]
    .head(12)
    .round(1)
    .to_string(index=False)
)
