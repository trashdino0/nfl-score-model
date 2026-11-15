import pandas as pd

w = pd.read_csv("outputs/week09_predictions.csv")
for _, r in w.iterrows():
    print(
        f"{r.away_team:>3} @ {r.home_team:<3} | {r.pred_winner:<3} | "
        f"{r.pred_away_score:4.1f}-{r.pred_home_score:<4.1f} | "
        f"{r.pick_win_prob * 100:4.1f}% | line {r.market_margin:+5.1f} / {r.market_total:4.1f} | "
        f"edge {r.model_edge:+5.2f} | ATS {r.ats_pick:<3} | {r.ou_pick}"
    )
