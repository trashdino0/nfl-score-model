import numpy as np
import pandas as pd

pd.set_option("display.width", 240)

b = pd.read_csv("outputs/backtest_2026_weeks1_8.csv")
b = b[b.home_score.notna() & b.away_score.notna()].copy()

models = {
    "Hybrid": ("pred_home_score", "pred_away_score"),
    "Market": ("market_total", "market_margin"),
    "Pure stats": ("pure_home_score", "pure_away_score"),
    "Naive": ("naive_total", "naive_margin"),
}

rows = []
for name, (hc, ac) in models.items():
    if name == "Market":
        ph = ((b[hc] + b[ac]) / 2).round().astype(int)
        pa = ((b[hc] - b[ac]) / 2).round().astype(int)
    elif name == "Naive":
        ph = ((b[hc] + b[ac]) / 2).round().astype(int)
        pa = ((b[hc] - b[ac]) / 2).round().astype(int)
    else:
        ph = b[hc].round().astype(int)
        pa = b[ac].round().astype(int)
    ah = b.home_score.astype(int)
    aa = b.away_score.astype(int)
    eh = (ph - ah).abs()
    ea = (pa - aa).abs()
    both = (eh == 0) & (ea == 0)
    either = (eh == 0) | (ea == 0)
    # margin/total from rounded scores
    pm = ph - pa
    pt = ph + pa
    am = ah - aa
    at = ah + aa
    em = (pm - am).abs()
    et = (pt - at).abs()
    rows.append(
        {
            "model": name,
            "exact_both": both.mean(),
            "exact_either": either.mean(),
            "exact_margin": (em == 0).mean(),
            "exact_total": (et == 0).mean(),
            "home_exact": (eh == 0).mean(),
            "away_exact": (ea == 0).mean(),
            "home_within3": (eh <= 3).mean(),
            "away_within3": (ea <= 3).mean(),
            "both_within3": ((eh <= 3) & (ea <= 3)).mean(),
            "both_within7": ((eh <= 7) & (ea <= 7)).mean(),
            "margin_within3": (em <= 3).mean(),
            "total_within3": (et <= 3).mean(),
            "total_within7": (et <= 7).mean(),
            "mean_abs_home": eh.mean(),
            "mean_abs_away": ea.mean(),
            "median_abs_both": pd.concat([eh, ea]).median(),
            "p90_abs_both": pd.concat([eh, ea]).quantile(0.90),
        }
    )

res = pd.DataFrame(rows)
print(res.round(4).to_string(index=False))

# winner-correct subset for hybrid
ph = b.pred_home_score.round().astype(int)
pa = b.pred_away_score.round().astype(int)
ah = b.home_score.astype(int)
aa = b.away_score.astype(int)
pred_win = b.pred_margin > 0
actual_win = b.margin > 0
correct = pred_win == actual_win
d = b[correct]
print("\nHybrid on games where it picked the winner correctly: n=", len(d))
eh = (d.pred_home_score.round().astype(int) - d.home_score.astype(int)).abs()
ea = (d.pred_away_score.round().astype(int) - d.away_score.astype(int)).abs()
print(
    "exact both",
    ((eh == 0) & (ea == 0)).mean(),
    "within3 both",
    ((eh <= 3) & (ea <= 3)).mean(),
    "score MAE",
    pd.concat([eh, ea]).mean(),
)

# exact-score distribution: how often does any single team's score hit
print("\nHybrid per-team score error distribution:")
err = pd.concat([eh, ea])
for q in [0.25, 0.5, 0.75, 0.9, 0.95]:
    print(q, err.quantile(q))
print(
    "within 1",
    (err <= 1).mean(),
    "within 3",
    (err <= 3).mean(),
    "within 7",
    (err <= 7).mean(),
    "within 10",
    (err <= 10).mean(),
)
