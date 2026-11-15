import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

pd.set_option("display.width", 260)

bt = pd.read_parquet("outputs/backtest_2026_raw.parquet")
bt = bt[bt.home_win.notna()].copy()

# Recover raw residual predictions (resid = market + 0.25 * res)
bt["res_m"] = (bt.resid_margin - bt.market_margin) / 0.25
bt["res_t"] = (bt.resid_total - bt.market_total) / 0.25

rows = []
for ms in [0.0, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]:
    for ts in [0.0, 0.10, 0.15, 0.25, 0.50]:
        pm = bt.market_margin + ms * bt.res_m
        pt = bt.market_total + ts * bt.res_t
        pred_win = (pm > 0).astype(float)
        acc = (pred_win == bt.home_win).mean()
        ph = (pt + pm) / 2
        pa = (pt - pm) / 2
        score_mae = mean_absolute_error(
            pd.concat([bt.home_score, bt.away_score]), pd.concat([ph, pa])
        )
        # OU win rate
        over = pt > bt.market_total
        push = pt == bt.market_total
        ou_win = (
            (over & (bt.total > bt.market_total))
            | (~over & (bt.total < bt.market_total))
        ) & ~push
        ou_rate = ou_win.sum() / (~push).sum() if (~push).sum() else np.nan
        rows.append(
            {
                "margin_shrink": ms,
                "total_shrink": ts,
                "margin_mae": mean_absolute_error(bt.margin, pm),
                "total_mae": mean_absolute_error(bt.total, pt),
                "score_mae": score_mae,
                "win_acc": acc,
                "ou_rate": ou_rate,
            }
        )
res = pd.DataFrame(rows)
print(
    res.sort_values(
        ["win_acc", "margin_mae", "total_mae"], ascending=[False, True, True]
    )
    .round(4)
    .head(30)
    .to_string(index=False)
)

print("\nSelected:")
for ms, ts in [
    (0.25, 0.0),
    (0.25, 0.10),
    (0.25, 0.15),
    (0.25, 0.25),
    (0.15, 0.10),
    (0.35, 0.10),
]:
    r = res[(res.margin_shrink == ms) & (res.total_shrink == ts)].iloc[0]
    print(
        f"margin {ms:.2f} total {ts:.2f}: win {r.win_acc:.4f} margin {r.margin_mae:.3f} total {r.total_mae:.3f} score {r.score_mae:.3f} OU {r.ou_rate:.3f}"
    )

# Also evaluate pure total vs market for reference
print("\nPure total MAE:", mean_absolute_error(bt.total, bt.pure_total))
print("Market total MAE:", mean_absolute_error(bt.total, bt.market_total))
