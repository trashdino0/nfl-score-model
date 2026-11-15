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
cols = list(dict.fromkeys(MATCHUP + TEAM + QB + INJ + SIT + ELO))


def ridge(alpha=100):
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=alpha)),
        ]
    )


rows = []
for season in [2021, 2022, 2023, 2024, 2025, 2026]:
    if season == 2026:
        train = f[
            (f.season >= 2018)
            & ((f.season < 2026) | ((f.season == 2026) & (f.week <= 8)))
        ]
        # weekly walk forward for 2026
        preds = []
        for week in range(1, 9):
            tr = f[
                (f.season >= 2018)
                & ((f.season < 2026) | ((f.season == 2026) & (f.week < week)))
            ]
            te = f[(f.season == 2026) & (f.week == week)]
            preds.append((tr, te))
    else:
        preds = [(f[(f.season >= 2018) & (f.season < season)], f[f.season == season])]

    out = []
    for tr, te in preds:
        use = [c for c in cols if tr[c].nunique(dropna=True) > 1]
        tr2 = tr.copy()
        tr2["res_m"] = tr2.margin - tr2.market_margin
        tr2["res_t"] = tr2.total - tr2.market_total
        te2 = te.copy()
        # margin/total residual
        for target, name in [
            ("res_m", "rm"),
            ("res_t", "rt"),
            ("home_score", "hs"),
            ("away_score", "ascore"),
        ]:
            m = ridge(100)
            m.fit(tr2[use], tr2[target])
            te2[name] = m.predict(te2[use])
        out.append(te2)
    o = pd.concat(out, ignore_index=True)
    # hybrid
    pm = o.market_margin + 0.2 * o.rm
    pt = o.market_total + 0.1 * o.rt
    ph = (pt + pm) / 2
    pa = (pt - pm) / 2
    # direct scores
    dh = o.hs
    da = o.ascore
    dm = dh - da
    dt = dh + da
    rows.append(
        {
            "season": season,
            "n": len(o),
            "hybrid_score_mae": mean_absolute_error(
                pd.concat([o.home_score, o.away_score]), pd.concat([ph, pa])
            ),
            "direct_score_mae": mean_absolute_error(
                pd.concat([o.home_score, o.away_score]), pd.concat([dh, da])
            ),
            "hybrid_margin_mae": mean_absolute_error(o.margin, pm),
            "direct_margin_mae": mean_absolute_error(o.margin, dm),
            "hybrid_total_mae": mean_absolute_error(o.total, pt),
            "direct_total_mae": mean_absolute_error(o.total, dt),
            "hybrid_win": ((pm > 0).astype(float) == o.home_win).mean(),
            "direct_win": ((dm > 0).astype(float) == o.home_win).mean(),
        }
    )
print(pd.DataFrame(rows).round(3).to_string(index=False))
