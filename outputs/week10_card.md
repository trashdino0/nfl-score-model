# 2026 Week 10 — NFL predictions

*Generated 2026-11-14 20:13. Model trained on 2264 completed games (2018+); margin σ = 12.73. This card covers 13 unplayed games.*

**Already final:** WAS 24 – NYG 21.

The model agrees with the market favorite in **13 of 13** games with a posted line (6 home favorites). Win probabilities are the model's estimate; the market's no-vig probability is shown beside it for comparison. The CSV's `win_prob` is a log-odds blend whose market/model weight is fitted on out-of-fold data.

**Backtest (136 games):** the hybrid model picks 70.4% of decisive winners vs 68.9% for the closing line; margin MAE 9.49 vs 9.49. A gap this small is within sampling noise, not a proven edge.

## The card

| Game | Pick | Model win % | Market win % | Projected score (away–home) | Pick's 80% score range | Line | Pick cover % |
| :--- | :--- | ---: | ---: | :--- | :---: | :--- | ---: |
| Sun 11/15 · BUF @ NYJ | **BUF** | 74% | 78% | BUF 29–21 NYJ | 18–42 | BUF -8.5 | 50% |
| Sun 11/15 · CAR @ NO | **NO** | 59% | 55% | CAR 23–26 NO | 14–38 | NO -2.5 | 51% |
| Sun 11/15 · HOU @ CLE | **HOU** | 64% | 68% | HOU 23–18 CLE | 11–35 | HOU -4.5 | 52% |
| Sun 11/15 · JAX @ TEN | **JAX** | 79% | 84% | JAX 28–18 TEN | 17–40 | JAX -10.5 | 51% |
| Sun 11/15 · KC @ ATL | **KC** | 58% | 61% | KC 25–22 ATL | 14–37 | KC -3 | 52% |
| Sun 11/15 · LA @ ARI | **LA** | 78% | 80% | LA 25–15 ARI | 13–37 | LA -9.5 | 54% |
| Sun 11/15 · MIA @ IND | **IND** | 59% | 66% | MIA 18–21 IND | 9–33 | IND -3.5 | 52% |
| Sun 11/15 · MIN @ GB | **GB** | 56% | 57% | MIN 20–22 GB | 11–34 | GB -2.5 | 55% |
| Sun 11/15 · NE @ DET | **DET** | 60% | 57% | NE 23–26 DET | 15–38 | DET -2.5 | 54% |
| Sun 11/15 · PIT @ CIN | **CIN** | 62% | 62% | PIT 23–27 CIN | 15–38 | CIN -3.5 | 50% |
| Sun 11/15 · SEA @ LV | **SEA** | 72% | 72% | SEA 24–17 LV | 12–36 | SEA -6.5 | 54% |
| Sun 11/15 · SF @ DAL | **SF** | 66% | 66% | SF 30–25 DAL | 19–43 | SF -4.5 | 54% |
| Mon 11/16 · LAC @ BAL | **BAL** | 64% | 66% | LAC 17–22 BAL | 10–34 | BAL -4.5 | 50% |

Cover probabilities come from a joint residual bootstrap (margin and total errors resampled together), so they are consistent with the projected score. The model does not publish over/under leans: its total model has not beaten the closing total out of sample, so the projected total is the market total.

## Confidence tiers

| Tier | Model win prob | Picks |
| :--- | :--- | :--- |
| High | ≥ 75% | JAX (79%), LA (78%) |
| Medium | 65–75% | BUF (74%), SEA (72%), SF (66%) |
| Slight | 55–65% | NO (59%), HOU (64%), KC (58%), IND (59%), GB (56%), DET (60%), CIN (62%), BAL (64%) |
| Coin flip | < 55% | — |

## Where the model differs from the market

| Game | Edge vs line | Value side | Line | Play? |
| :--- | ---: | :--- | :--- | :--- |
| NE @ DET | 0.8 pts | DET | DET -2.5 | lean only |
| SEA @ LV | 0.8 pts | SEA | SEA -6.5 | lean only |
| MIN @ GB | 0.7 pts | MIN | MIN +2.5 | lean only |
| SF @ DAL | 0.7 pts | SF | SF -4.5 | lean only |
| KC @ ATL | 0.5 pts | ATL | ATL +3 | lean only |

The value side is where the model is higher than the market, in points. It is not always the same as the pick: the model can project a team to win without covering. A side is labelled a play only when the edge is at least 1 point; smaller edges are noise. There is no demonstrated betting edge.

## Availability and weather notes

- **BUF @ NYJ** — NYJ QB: Geno Smith; BUF QB: Josh Allen; BUF TE Dalton Kincaid questionable
- **CAR @ NO** — NO QB: Tyler Shough; CAR QB: Bryce Young; dome
- **HOU @ CLE** — CLE QB: Deshaun Watson in for Shedeur Sanders; HOU QB: C.J. Stroud; CLE WR Denzel Boston out; HOU RB Woody Marks out; CLE TE Harold Fannin Jr. questionable
- **JAX @ TEN** — TEN QB: Cam Ward; JAX QB: Trevor Lawrence; TEN WR Carnell Tate out; TEN RB Tony Pollard out; JAX RB Bhayshul Tuten out
- **KC @ ATL** — ATL QB: Michael Penix Jr.; KC QB: Patrick Mahomes
- **LA @ ARI** — ARI QB: Gardner Minshew; LA QB: Matthew Stafford; LA WR Puka Nacua out; LA TE Terrance Ferguson out
- **MIA @ IND** — IND QB: Daniel Jones; MIA QB: Malik Willis
- **MIN @ GB** — GB QB: Jordan Love; MIN QB: Kirk Cousins in for Carson Wentz; GB RB Josh Jacobs out; MIN TE T.J. Hockenson out
- **NE @ DET** — DET QB: Jared Goff; NE QB: Drake Maye; DET WR Amon-Ra St. Brown questionable; NE WR A.J. Brown questionable; NE RB Rhamondre Stevenson questionable; NE TE Hunter Henry questionable; dome
- **PIT @ CIN** — CIN QB: Joe Burrow; PIT QB: Aaron Rodgers
- **SEA @ LV** — LV QB: Fernando Mendoza in for Kirk Cousins; SEA QB: Drew Lock; dome
- **SF @ DAL** — DAL QB: Dak Prescott; SF QB: Brock Purdy; SF WR Mike Evans out; DAL TE Jake Ferguson out
- **LAC @ BAL** — BAL QB: Tyler Huntley in for Joe Fagnano; LAC QB: Justin Herbert

## How to read this card

- **Pick** is the team with the higher projected score. **Model win %** is the probability the model assigns to that pick; **market win %** is the no-vig implied probability for the same side. When they disagree, the market is the better-calibrated number.
- **Projected score** is a continuous estimate, not a final-score promise. The 80% range is this game's 10th–90th percentile from the joint residual bootstrap; exact NFL scores are inherently noisy.
- **Line / cover %** use the market snapshot in the data. A cover probability above 50% is the model's lean against the spread, not a recommendation. Only edges ≥ 1 point are labelled plays, and the backtest has not shown a profitable ATS strategy.
- **No over/under lean is published.** The total model did not beat the closing total out of sample, so the projected total is the market total.
- **Availability** reflects the injury report and depth chart at build time. Late scratches and line movement are not captured; rebuild if either changes.

*CSV: `outputs/week10_predictions.csv`*
