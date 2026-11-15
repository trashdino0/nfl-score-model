import pandas as pd

b = pd.read_csv("outputs/backtest_2026_weeks1_8.csv")
b = b[b.home_score.notna() & b.away_score.notna()].copy()

ph = b.pred_home_score.round().astype(int)
pa = b.pred_away_score.round().astype(int)
ah = b.home_score.astype(int)
aa = b.away_score.astype(int)
eh = (ph - ah).abs()
ea = (pa - aa).abs()
both = (eh == 0) & (ea == 0)
either = (eh == 0) | (ea == 0)
em = ((ph - pa) - (ah - aa)).abs()
et = ((ph + pa) - (ah + aa)).abs()
err = pd.concat([eh, ea], ignore_index=True)

n = len(b)
print("n", n)
print("exact both", both.sum(), both.mean())
print("exact either", either.sum(), either.mean())
print("exact margin", (em == 0).sum(), (em == 0).mean())
print("exact total", (et == 0).sum(), (et == 0).mean())
print("both within 3", ((eh <= 3) & (ea <= 3)).sum(), ((eh <= 3) & (ea <= 3)).mean())
print("both within 7", ((eh <= 7) & (ea <= 7)).sum(), ((eh <= 7) & (ea <= 7)).mean())
print("per-team within 1", (err <= 1).sum(), (err <= 1).mean())
print("per-team within 3", (err <= 3).sum(), (err <= 3).mean())
print("per-team within 7", (err <= 7).sum(), (err <= 7).mean())
print("per-team within 10", (err <= 10).sum(), (err <= 10).mean())
print("median", err.median(), "p90", err.quantile(0.9))
print("mean rounded", err.mean())
print(
    "mean continuous",
    (
        pd.concat([b.pred_home_score, b.pred_away_score], ignore_index=True)
        - pd.concat([b.home_score, b.away_score], ignore_index=True)
    )
    .abs()
    .mean(),
)

# market
mh = ((b.market_total + b.market_margin) / 2).round().astype(int)
ma = ((b.market_total - b.market_margin) / 2).round().astype(int)
meh = (mh - ah).abs()
mea = (ma - aa).abs()
merr = pd.concat([meh, mea], ignore_index=True)
print(
    "\nmarket exact either",
    ((meh == 0) | (mea == 0)).sum(),
    ((meh == 0) | (mea == 0)).mean(),
)
print(
    "market both within 3",
    ((meh <= 3) & (mea <= 3)).sum(),
    ((meh <= 3) & (mea <= 3)).mean(),
)
print(
    "market both within 7",
    ((meh <= 7) & (mea <= 7)).sum(),
    ((meh <= 7) & (mea <= 7)).mean(),
)
print("market per-team within 3", (merr <= 3).mean(), "within 7", (merr <= 7).mean())
print("market median", merr.median(), "p90", merr.quantile(0.9), "mean", merr.mean())
