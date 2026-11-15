# 2026 Week 10 — NFL predictions

*Generated 2026-11-13 22:39. Model trained on 2264 completed games (2018+); margin σ = 12.73. This card covers 13 unplayed games.*

**Already final:** WAS 24 – NYG 21 (Thu). Saved pre-game TNF pick (screenshot, 11/12): NYG 22–20, 55% — the NYG winner pick lost outright; the WAS +2.5 ATS lean and over 41.5 both hit. Grade: `outputs/week10_tnf_graded.csv`.

The model agrees with the market favorite in **13 of 13** games with a posted line (6 home favorites). Win probabilities are blended 50/50 with the no-vig market when a line is posted, so they are deliberately close to the line.

**Backtest through Week 8:** 2026 walk-forward — 67.5% winner accuracy (81/120 decisive games) vs 65.8% for the closing line; margin MAE 9.58 vs 9.57. The difference is within noise, not a proven edge.

## The card

| Game | Pick | Win prob | Projected score (away–home) | Pick's 80% score range | Line | ATS lean | O/U lean |
| :--- | :--- | ---: | :--- | :---: | :--- | :--- | :--- |
| Sun 11/15 · BUF @ NYJ | **BUF** | 76% | BUF 29–21 NYJ | 17–41 | BUF -8.5 | NYJ +8.5 | Under 50.5 |
| Sun 11/15 · CAR @ NO | **NO** | 57% | CAR 23–26 NO | 13–37 | NO -2.5 | NO -2.5 | Over 48.5 |
| Sun 11/15 · HOU @ CLE | **HOU** | 66% | HOU 22–18 CLE | 11–34 | HOU -4.5 | HOU -4.5 | Under 40.5 |
| Sun 11/15 · JAX @ TEN | **JAX** | 82% | JAX 28–18 TEN | 17–40 | JAX -10.5 | TEN +10.5 | Over 46.5 |
| Sun 11/15 · KC @ ATL | **KC** | 59% | KC 25–23 ATL | 13–37 | KC -3 | ATL +3 | Over 47.5 |
| Sun 11/15 · LA @ ARI | **LA** | 79% | LA 25–15 ARI | 13–36 | LA -9.5 | LA -9.5 | Over 39.5 |
| Sun 11/15 · MIA @ IND | **IND** | 63% | MIA 18–21 IND | 8–32 | IND -3.5 | MIA +3.5 | Over 38.5 |
| Sun 11/15 · MIN @ GB | **GB** | 56% | MIN 20–22 GB | 9–33 | GB -2.5 | MIN +2.5 | Under 42.5 |
| Sun 11/15 · NE @ DET | **DET** | 59% | NE 24–27 DET | 14–38 | DET -2.5 | DET -2.5 | Over 49.5 |
| Sun 11/15 · PIT @ CIN | **CIN** | 62% | PIT 23–27 CIN | 14–38 | CIN -3.5 | CIN -3.5 | Over 49.5 |
| Sun 11/15 · SEA @ LV | **SEA** | 72% | SEA 23–16 LV | 11–35 | SEA -6.5 | SEA -6.5 | Under 40.5 |
| Sun 11/15 · SF @ DAL | **SF** | 66% | SF 31–26 DAL | 19–42 | SF -4.5 | SF -4.5 | Over 55.5 |
| Mon 11/16 · LAC @ BAL | **BAL** | 65% | LAC 17–22 BAL | 9–33 | BAL -4.5 | BAL -4.5 | Under 38.5 |

ATS leans are against the spread, not the moneyline: a team can be the pick to win and still not cover. O/U leans compare the model total to the market total.

## Confidence tiers

| Tier | Pick win prob | Picks |
| :--- | :--- | :--- |
| High | ≥ 75% | BUF (76%), JAX (82%), LA (79%) |
| Medium | 65–75% | HOU (66%), SEA (72%), SF (66%), BAL (65%) |
| Slight | 55–65% | NO (57%), KC (59%), IND (63%), GB (56%), DET (59%), CIN (62%) |
| Coin flip | < 55% | — |

## Where the model differs from the market

| Game | Edge vs line | Value side | Line |
| :--- | ---: | :--- | :--- |
| NE @ DET | 0.8 pts | DET | DET -2.5 |
| SEA @ LV | 0.8 pts | SEA | SEA -6.5 |
| MIN @ GB | 0.7 pts | MIN | MIN +2.5 |
| SF @ DAL | 0.7 pts | SF | SF -4.5 |
| KC @ ATL | 0.5 pts | ATL | ATL +3 |

The value side is where the model is higher than the market, in points. It is not always the same as the pick: the model can project a team to win without covering. Edges under a point are noise, not an invitation to bet.

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

- **Pick** is the team with the higher projected score. **Win prob** is the chance that pick wins, blended 50/50 between the model and the no-vig market when a line is posted.
- **Projected score** is a continuous estimate, not a final-score promise. The 80% range comes from 2021–2025 walk-forward residuals; the median team-score miss in the 2026 backtest was about 6 points.
- **Line / ATS / O-U** use the market snapshot in the data. The model does not have a proven betting edge: 2026 Weeks 1–8 went 51.3% ATS and 42.1% on totals. Treat leans as diagnostics, not recommendations.
- **Availability** reflects the injury report and depth chart at build time. Late scratches and line movement are not captured; rebuild if either changes.

*CSV: `outputs/week10_predictions.csv`*
