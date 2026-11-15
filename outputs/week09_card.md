# 2026 Week 9 — NFL predictions

> **Pre-Tier-1 card (historical).** Generated before the evaluation/fallback and
> joint-distribution changes. It shows a 50/50 market blend, fixed score ranges,
> and an over/under lean that the current model no longer publishes. Kept as a
> record of what was published; the Week 9 grade is in the git history only.

*Generated 2026-11-06 23:02. Model trained on 2249 completed games (2018+); margin σ = 12.74. This card covers 14 unplayed games.*

**Already final:** JAX 33 – BAL 10.

The model agrees with the market favorite in **14 of 14** games with a posted line (6 home favorites). Win probabilities are blended 50/50 with the no-vig market when a line is posted, so they are deliberately close to the line.

**Backtest through Week 8:** 2026 walk-forward — 67.5% winner accuracy (81/120 decisive games) vs 65.8% for the closing line; margin MAE 9.58 vs 9.57. The difference is within noise, not a proven edge.

## The card

| Game | Pick | Win prob | Projected score (away–home) | Pick's 80% score range | Line | ATS lean | O/U lean |
| :--- | :--- | ---: | :--- | :---: | :--- | :--- | :--- |
| Sun 11/08 · ARI @ SEA | **SEA** | 78% | ARI 15–24 SEA | 11–35 | SEA -8.5 | SEA -8.5 | Under 39.5 |
| Sun 11/08 · CIN @ ATL | **CIN** | 60% | CIN 28–25 ATL | 16–39 | CIN -3 | ATL +3 | Over 52.5 |
| Sun 11/08 · CLE @ NO | **NO** | 60% | CLE 20–23 NO | 10–34 | NO -3.5 | CLE +3.5 | Over 42.5 |
| Sun 11/08 · DAL @ IND | **DAL** | 66% | DAL 28–23 IND | 16–39 | DAL -4.5 | DAL -4.5 | Under 51.5 |
| Sun 11/08 · DEN @ CAR | **DEN** | 59% | DEN 24–21 CAR | 12–35 | DEN -3 | DEN -3 | Over 44.5 |
| Sun 11/08 · DET @ MIA | **DET** | 71% | DET 28–22 MIA | 16–39 | DET -6.5 | DET -6.5 | Over 49.5 |
| Sun 11/08 · GB @ NE | **NE** | 59% | GB 21–24 NE | 11–35 | NE -3 | GB +3 | Over 44.5 |
| Sun 11/08 · HOU @ LAC | **HOU** | 67% | HOU 23–18 LAC | 11–34 | HOU -6 | LAC +6 | Over 40.5 |
| Sun 11/08 · LA @ WAS | **LA** | 66% | LA 27–21 WAS | 15–38 | LA -4.5 | LA -4.5 | Over 47.5 |
| Sun 11/08 · LV @ SF | **SF** | 79% | LV 20–31 SF | 18–42 | SF -10 | SF -10 | Under 51.5 |
| Sun 11/08 · NYG @ PHI | **NYG** | 58% | NYG 21–18 PHI | 9–32 | NYG -3 | PHI +3 | Over 38.5 |
| Sun 11/08 · NYJ @ KC | **KC** | 78% | NYJ 20–29 KC | 16–40 | KC -9.5 | NYJ +9.5 | Over 48.5 |
| Sun 11/08 · TB @ CHI | **CHI** | 69% | TB 21–27 CHI | 14–38 | CHI -6 | CHI -6 | Over 47.5 |
| Mon 11/09 · BUF @ MIN | **BUF** | 61% | BUF 24–20 MIN | 12–35 | BUF -3 | BUF -3 | Over 43.5 |

ATS leans are against the spread, not the moneyline: a team can be the pick to win and still not cover. O/U leans compare the model total to the market total.

## Confidence tiers

| Tier | Pick win prob | Picks |
| :--- | :--- | :--- |
| High | ≥ 75% | SEA (78%), SF (79%), KC (78%) |
| Medium | 65–75% | DAL (66%), DET (71%), HOU (67%), LA (66%), CHI (69%) |
| Slight | 55–65% | CIN (60%), NO (60%), DEN (59%), NE (59%), NYG (58%), BUF (61%) |
| Coin flip | < 55% | — |

## Where the model differs from the market

| Game | Edge vs line | Value side | Line |
| :--- | ---: | :--- | :--- |
| BUF @ MIN | 0.9 pts | BUF | BUF -3 |
| HOU @ LAC | 0.8 pts | LAC | LAC +6 |
| LA @ WAS | 0.8 pts | LA | LA -4.5 |
| ARI @ SEA | 0.8 pts | SEA | SEA -8.5 |
| DAL @ IND | 0.4 pts | DAL | DAL -4.5 |

The value side is where the model is higher than the market, in points. It is not always the same as the pick: the model can project a team to win without covering. Edges under a point are noise, not an invitation to bet.

## Availability and weather notes

- **ARI @ SEA** — SEA QB: Drew Lock in for Sam Darnold (out); ARI QB: Gardner Minshew in for Jacoby Brissett (out); SEA WR Jaxon Smith-Njigba questionable; ARI WR Michael Wilson questionable
- **CIN @ ATL** — ATL QB: Michael Penix Jr.; CIN QB: Joe Burrow
- **CLE @ NO** — NO QB: Tyler Shough; CLE QB: Shedeur Sanders in for Deshaun Watson (out); CLE WR Denzel Boston questionable; CLE TE Harold Fannin Jr. questionable; dome
- **DAL @ IND** — IND QB: Daniel Jones; DAL QB: Dak Prescott; DAL TE Jake Ferguson out
- **DEN @ CAR** — CAR QB: Bryce Young; DEN QB: Bo Nix
- **DET @ MIA** — MIA QB: Malik Willis; DET QB: Jared Goff; DET WR Amon-Ra St. Brown questionable
- **GB @ NE** — NE QB: Drake Maye; GB QB: Jordan Love; NE WR A.J. Brown out; NE TE Hunter Henry out
- **HOU @ LAC** — LAC QB: Justin Herbert; HOU QB: C.J. Stroud; HOU RB David Montgomery out; dome
- **LA @ WAS** — WAS QB: Jayden Daniels; LA QB: Matthew Stafford; WAS WR Stefon Diggs out
- **LV @ SF** — SF QB: Brock Purdy; LV QB: Kirk Cousins; SF WR Mike Evans out; SF RB Christian McCaffrey questionable
- **NYG @ PHI** — PHI QB: Joe Milton III in for Tanner McKee; NYG QB: J.J. McCarthy; PHI WR DeVonta Smith out; NYG WR Malik Nabers out; NYG RB Cam Skattebo out
- **NYJ @ KC** — KC QB: Patrick Mahomes; NYJ QB: Geno Smith; NYJ WR Garrett Wilson out
- **TB @ CHI** — CHI QB: Caleb Williams; TB QB: Baker Mayfield; TB RB Bucky Irving out
- **BUF @ MIN** — MIN QB: Carson Wentz in for Kyler Murray (out); BUF QB: Josh Allen; BUF WR DJ Moore questionable; BUF TE Dalton Kincaid questionable; dome

## How to read this card

- **Pick** is the team with the higher projected score. **Win prob** is the chance that pick wins, blended 50/50 between the model and the no-vig market when a line is posted.
- **Projected score** is a continuous estimate, not a final-score promise. The 80% range comes from 2021–2025 walk-forward residuals; the median team-score miss in the 2026 backtest was about 6 points.
- **Line / ATS / O-U** use the market snapshot in the data. The model does not have a proven betting edge: 2026 Weeks 1–8 went 51.3% ATS and 42.1% on totals. Treat leans as diagnostics, not recommendations.
- **Availability** reflects the injury report and depth chart at build time. Late scratches and line movement are not captured; rebuild if either changes.

*CSV: `outputs/week09_predictions.csv`*
