# 2026 Week 11 — NFL predictions

*Generated 2026-11-17 18:12. Model trained on 2277 completed games (2018+); margin σ = 12.73. This card covers 13 unplayed games.*

The model agrees with the market favorite in **13 of 13** games with a posted line (10 home favorites). Win probabilities are the model's estimate; the market's no-vig probability is shown beside it for comparison. The CSV's `win_prob` is a log-odds blend whose market/model weight is fitted on out-of-fold data.

**Backtest (136 games):** the hybrid model picks 70.4% of decisive winners vs 68.9% for the closing line; margin MAE 9.49 vs 9.49. A gap this small is within sampling noise, not a proven edge.

## The card

| Game | Pick | Model win % | Market win % | Projected score (away–home) | Pick's 80% score range | Line | Pick cover % |
| :--- | :--- | ---: | ---: | :--- | :---: | :--- | ---: |
| Thu 11/19 · IND @ HOU | **HOU** | 84% | 84% | IND 16–28 HOU | 17–40 | HOU -12.5 | 52% |
| Sun 11/22 · ARI @ KC | **KC** | 90% | 89% | ARI 13–29 KC | 17–41 | KC -16.5 | 52% |
| Sun 11/22 · BAL @ CAR | **CAR** | 56% | 56% | BAL 21–23 CAR | 12–35 | CAR -2.5 | 55% |
| Sun 11/22 · JAX @ NYG | **JAX** | 70% | 70% | JAX 27–20 NYG | 15–39 | JAX -6 | 54% |
| Sun 11/22 · LV @ DEN | **DEN** | 78% | 82% | LV 15–25 DEN | 14–37 | DEN -10 | 52% |
| Sun 11/22 · MIA @ BUF | **BUF** | 91% | 90% | MIA 14–31 BUF | 20–43 | BUF -16.5 | 50% |
| Sun 11/22 · MIN @ SF | **SF** | 61% | 59% | MIN 21–25 SF | 13–36 | SF -3 | 52% |
| Sun 11/22 · NO @ CHI | **CHI** | 83% | 84% | NO 19–31 CHI | 19–43 | CHI -11.5 | 53% |
| Sun 11/22 · NYJ @ LAC | **LAC** | 60% | 62% | NYJ 19–22 LAC | 11–34 | LAC -3.5 | 52% |
| Sun 11/22 · PIT @ PHI | **PIT** | 65% | 64% | PIT 22–17 PHI | 11–34 | PIT -4.5 | 54% |
| Sun 11/22 · TB @ DET | **DET** | 71% | 73% | TB 19–26 DET | 14–38 | DET -7 | 50% |
| Sun 11/22 · TEN @ DAL | **DAL** | 74% | 76% | TEN 23–31 DAL | 20–43 | DAL -8.5 | 52% |
| Mon 11/23 · CIN @ WAS | **CIN** | 70% | 71% | CIN 30–23 WAS | 19–42 | CIN -6.5 | 52% |

Cover probabilities come from a joint residual bootstrap (margin and total errors resampled together), so they are consistent with the projected score. The model does not publish over/under leans: its total model has not beaten the closing total out of sample, so the projected total is the market total.

## Confidence tiers

| Tier | Model win prob | Picks |
| :--- | :--- | :--- |
| High | ≥ 75% | HOU (84%), KC (90%), DEN (78%), BUF (91%), CHI (83%) |
| Medium | 65–75% | JAX (70%), PIT (65%), DET (71%), DAL (74%), CIN (70%) |
| Slight | 55–65% | CAR (56%), SF (61%), LAC (60%) |
| Coin flip | < 55% | — |

## Where the model differs from the market

| Game | Edge vs line | Value side | Line | Play? |
| :--- | ---: | :--- | :--- | :--- |
| NO @ CHI | 0.7 pts | CHI | CHI -11.5 | lean only |
| JAX @ NYG | 0.7 pts | JAX | JAX -6 | lean only |
| BAL @ CAR | 0.7 pts | BAL | BAL +2.5 | lean only |
| MIN @ SF | 0.5 pts | SF | SF -3 | lean only |
| PIT @ PHI | 0.5 pts | PIT | PIT -4.5 | lean only |

The value side is where the model is higher than the market, in points. It is not always the same as the pick: the model can project a team to win without covering. A side is labelled a play only when the edge is at least 1 point; smaller edges are noise. There is no demonstrated betting edge.

## Availability and weather notes

- **IND @ HOU** — HOU QB: C.J. Stroud; IND QB: Daniel Jones
- **ARI @ KC** — KC QB: Patrick Mahomes; ARI QB: Gardner Minshew in for Carson Beck
- **BAL @ CAR** — CAR QB: Bryce Young; BAL QB: Tyler Huntley in for Jacoby Brissett
- **JAX @ NYG** — NYG QB: J.J. McCarthy; JAX QB: Trevor Lawrence
- **LV @ DEN** — DEN QB: Bo Nix; LV QB: Fernando Mendoza
- **MIA @ BUF** — BUF QB: Josh Allen; MIA QB: Kyle McCord
- **MIN @ SF** — SF QB: Brock Purdy; MIN QB: Kirk Cousins
- **NO @ CHI** — CHI QB: Caleb Williams; NO QB: Tyler Shough
- **NYJ @ LAC** — LAC QB: Justin Herbert; NYJ QB: Geno Smith; dome
- **PIT @ PHI** — PHI QB: Joe Milton III; PIT QB: Aaron Rodgers
- **TB @ DET** — DET QB: Jared Goff; TB QB: Baker Mayfield; dome
- **TEN @ DAL** — DAL QB: Dak Prescott; TEN QB: Cam Ward
- **CIN @ WAS** — WAS QB: Jayden Daniels; CIN QB: Joe Burrow

## How to read this card

- **Pick** is the team with the higher projected score. **Model win %** is the probability the model assigns to that pick; **market win %** is the no-vig implied probability for the same side. When they disagree, the market is the better-calibrated number.
- **Projected score** is a continuous estimate, not a final-score promise. The 80% range is this game's 10th–90th percentile from the joint residual bootstrap; exact NFL scores are inherently noisy.
- **Line / cover %** use the market snapshot in the data. A cover probability above 50% is the model's lean against the spread, not a recommendation. Only edges ≥ 1 point are labelled plays, and the backtest has not shown a profitable ATS strategy.
- **No over/under lean is published.** The total model did not beat the closing total out of sample, so the projected total is the market total.
- **Availability** reflects the injury report and depth chart at build time. Late scratches and line movement are not captured; rebuild if either changes.

*CSV: `outputs/week11_predictions.csv`*
