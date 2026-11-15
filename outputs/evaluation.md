# Evaluation — paired, out-of-sample

*Generated 2026-11-14 20:14. Bootstrap intervals resample games, so paired predictions keep their dependence.*

**How to read this:** a negative Δ for MAE/Brier means the first model is better. A result is only called detectable when the 95% interval excludes zero and the paired test agrees. Everything else is noise.

## 2026 (136 games)

| Model | n | Winner acc. | Margin MAE | Total MAE | Score MAE | Brier | Log loss |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Market closing line | 136 | 68.9% | 9.49 | 10.60 | 7.26 | 0.215 | 0.622 |
| Pure stats ridge | 136 | 67.4% | 9.73 | 12.07 | 7.78 | 0.213 | 0.615 |
| Hybrid market+stats | 136 | 70.4% | 9.49 | 10.60 | 7.25 | 0.215 | 0.622 |

### Paired comparisons

| Comparison | Metric | Δ (A−B) | 95% CI | p | Verdict |
| :--- | :--- | ---: | ---: | ---: | :--- |
| hybrid vs market | margin MAE | +0.004 | [-0.113, +0.120] | 0.941 | no detectable difference |
| hybrid vs market | total MAE | +0.000 | [+0.000, +0.000] | 1.000 | identical by design |
| hybrid vs market | score MAE | -0.008 | [-0.050, +0.033] | 0.686 | no detectable difference |
| hybrid vs market | winner accuracy (pp) | +1.481 | — | 0.500 | no detectable difference |
| hybrid vs market | Brier | -0.000 | [-0.003, +0.002] | 0.840 | no detectable difference |
| hybrid vs pure | margin MAE | -0.240 | [-0.920, +0.433] | 0.467 | no detectable difference |
| hybrid vs pure | total MAE | -1.479 | [-2.295, -0.667] | 0.000 | detectable |
| hybrid vs pure | score MAE | -0.529 | [-0.894, -0.163] | 0.005 | detectable |
| hybrid vs pure | winner accuracy (pp) | +2.963 | — | 0.585 | no detectable difference |
| hybrid vs pure | Brier | +0.002 | [-0.018, +0.022] | 0.855 | no detectable difference |
| pure vs market | margin MAE | +0.244 | [-0.497, +0.994] | 0.504 | no detectable difference |
| pure vs market | total MAE | +1.479 | [+0.667, +2.295] | 0.000 | detectable |
| pure vs market | score MAE | +0.521 | [+0.143, +0.903] | 0.008 | detectable |
| pure vs market | winner accuracy (pp) | -1.481 | — | 0.860 | no detectable difference |
| pure vs market | Brier | -0.002 | [-0.024, +0.019] | 0.843 | no detectable difference |

### Betting diagnostics (flat 1-unit bets at −110)

| Strategy | n | W–L–P | Win rate (95% CI) | ROI (95% CI) |
| :--- | ---: | ---: | ---: | ---: |
| ATS — every lean | 130 | 68–62–6 | 52.3% [43.8%, 60.7%] | -0.1% [-16.3%, 16.0%] |
| ATS — plays only (edge ≥ 1) | 21 | 10–11–1 | 47.6% [28.3%, 67.6%] | -9.1% [-45.5%, 27.3%] |

**ATS by model edge:**

| Min |edge| | n | W–L–P | Win rate (95% CI) | ROI (95% CI) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 130 | 68–62–6 | 52.3% [43.8%, 60.7%] | -0.1% [-16.3%, 16.0%] |
| 0.5 | 58 | 34–24–1 | 58.6% [45.8%, 70.4%] | 11.9% [-11.1%, 35.0%] |
| 1 | 21 | 10–11–1 | 47.6% [28.3%, 67.6%] | -9.1% [-45.5%, 27.3%] |
| 1.5 | 7 | 5–2–0 | 71.4% [35.9%, 91.8%] | 36.4% [-18.2%, 90.9%] |
| 2 | 2 | 2–0–0 | 100.0% [34.2%, 100.0%] | 90.9% [90.9%, 90.9%] |

### Calibration

| Probability | Bin | n | Predicted | Actual | 95% CI |
| :--- | :--- | ---: | ---: | ---: | :--- |
| blend | (0.125, 0.25] | 3 | 22.1% | 33.3% | [6.1%, 79.2%] |
| blend | (0.25, 0.375] | 21 | 33.7% | 14.3% | [5.0%, 34.6%] |
| blend | (0.375, 0.5] | 30 | 42.7% | 33.3% | [19.2%, 51.2%] |
| blend | (0.5, 0.625] | 36 | 57.9% | 58.3% | [42.2%, 72.9%] |
| blend | (0.625, 0.75] | 28 | 69.2% | 71.4% | [52.9%, 84.7%] |
| blend | (0.75, 0.875] | 17 | 79.2% | 70.6% | [46.9%, 86.7%] |
| model | (0.125, 0.25] | 3 | 23.2% | 33.3% | [6.1%, 79.2%] |
| model | (0.25, 0.375] | 19 | 34.2% | 15.8% | [5.5%, 37.6%] |
| model | (0.375, 0.5] | 32 | 42.3% | 28.1% | [15.6%, 45.4%] |
| model | (0.5, 0.625] | 39 | 58.0% | 59.0% | [43.4%, 72.9%] |
| model | (0.625, 0.75] | 29 | 69.4% | 79.3% | [61.6%, 90.2%] |
| model | (0.75, 0.875] | 13 | 79.2% | 61.5% | [35.5%, 82.3%] |
| market | (0.125, 0.25] | 3 | 20.6% | 33.3% | [6.1%, 79.2%] |
| market | (0.25, 0.375] | 19 | 32.0% | 10.5% | [2.9%, 31.4%] |
| market | (0.375, 0.5] | 31 | 42.5% | 35.5% | [21.1%, 53.1%] |
| market | (0.5, 0.625] | 32 | 57.2% | 56.2% | [39.3%, 71.8%] |
| market | (0.625, 0.75] | 31 | 69.1% | 71.0% | [53.4%, 83.9%] |
| market | (0.75, 0.875] | 19 | 79.9% | 68.4% | [46.0%, 84.6%] |

### Model edge over the market

Mean absolute edge **0.55 pts** (median 0.44, p90 1.21). 16.2% of games have an edge above 1 point; 1.5% above 2.

## 2021-2025 (1359 games)

| Model | n | Winner acc. | Margin MAE | Total MAE | Score MAE | Brier | Log loss |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Market closing line | 1359 | 66.6% | 9.75 | 10.31 | 7.24 | 0.212 | 0.611 |
| Pure stats ridge | 1359 | 63.0% | 10.39 | 10.85 | 7.58 | 0.227 | 0.647 |
| Hybrid market+stats | 1359 | 66.7% | 9.74 | 10.31 | 7.23 | 0.212 | 0.610 |

### Paired comparisons

| Comparison | Metric | Δ (A−B) | 95% CI | p | Verdict |
| :--- | :--- | ---: | ---: | ---: | :--- |
| hybrid vs market | margin MAE | -0.009 | [-0.038, +0.021] | 0.569 | no detectable difference |
| hybrid vs market | total MAE | +0.000 | [+0.000, +0.000] | 1.000 | identical by design |
| hybrid vs market | score MAE | -0.009 | [-0.020, +0.002] | 0.092 | no detectable difference |
| hybrid vs market | winner accuracy (pp) | +0.074 | — | 1.000 | no detectable difference |
| hybrid vs market | Brier | -0.000 | [-0.001, +0.000] | 0.492 | no detectable difference |
| hybrid vs pure | margin MAE | -0.645 | [-0.837, -0.456] | 0.000 | detectable |
| hybrid vs pure | total MAE | -0.538 | [-0.783, -0.289] | 0.000 | detectable |
| hybrid vs pure | score MAE | -0.344 | [-0.455, -0.228] | 0.000 | detectable |
| hybrid vs pure | winner accuracy (pp) | +3.690 | — | 0.004 | detectable |
| hybrid vs pure | Brier | -0.016 | [-0.022, -0.010] | 0.000 | detectable |
| pure vs market | margin MAE | +0.637 | [+0.431, +0.842] | 0.000 | detectable |
| pure vs market | total MAE | +0.538 | [+0.289, +0.783] | 0.000 | detectable |
| pure vs market | score MAE | +0.335 | [+0.216, +0.449] | 0.000 | detectable |
| pure vs market | winner accuracy (pp) | -3.616 | — | 0.005 | detectable |
| pure vs market | Brier | +0.016 | [+0.009, +0.022] | 0.000 | detectable |

### Betting diagnostics (flat 1-unit bets at −110)

| Strategy | n | W–L–P | Win rate (95% CI) | ROI (95% CI) |
| :--- | ---: | ---: | ---: | ---: |
| ATS — every lean | 1326 | 686–640–33 | 51.7% [49.0%, 54.4%] | -1.2% [-6.4%, 3.8%] |
| ATS — plays only (edge ≥ 1) | 104 | 58–46–1 | 55.8% [46.2%, 64.9%] | 6.5% [-11.9%, 24.8%] |

**ATS by model edge:**

| Min |edge| | n | W–L–P | Win rate (95% CI) | ROI (95% CI) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1326 | 686–640–33 | 51.7% [49.0%, 54.4%] | -1.2% [-6.4%, 3.8%] |
| 0.5 | 451 | 236–215–12 | 52.3% [47.7%, 56.9%] | -0.1% [-8.6%, 8.4%] |
| 1 | 104 | 58–46–1 | 55.8% [46.2%, 64.9%] | 6.5% [-11.9%, 24.8%] |
| 1.5 | 21 | 14–7–0 | 66.7% [45.4%, 82.8%] | 27.3% [-9.1%, 63.6%] |
| 2 | 6 | 4–2–0 | 66.7% [30.0%, 90.3%] | 27.3% [-36.4%, 90.9%] |

### Calibration

| Probability | Bin | n | Predicted | Actual | 95% CI |
| :--- | :--- | ---: | ---: | ---: | :--- |
| blend | (-0.001, 0.125] | 3 | 10.8% | 33.3% | [6.1%, 79.2%] |
| blend | (0.125, 0.25] | 65 | 20.0% | 23.1% | [14.5%, 34.6%] |
| blend | (0.25, 0.375] | 185 | 31.6% | 29.7% | [23.6%, 36.7%] |
| blend | (0.375, 0.5] | 287 | 42.8% | 39.7% | [34.2%, 45.5%] |
| blend | (0.5, 0.625] | 345 | 57.7% | 57.4% | [52.1%, 62.5%] |
| blend | (0.625, 0.75] | 290 | 68.7% | 66.2% | [60.6%, 71.4%] |
| blend | (0.75, 0.875] | 160 | 80.6% | 86.2% | [80.1%, 90.7%] |
| blend | (0.875, 1.0] | 20 | 89.3% | 100.0% | [83.9%, 100.0%] |
| model | (-0.001, 0.125] | 2 | 9.8% | 0.0% | [0.0%, 65.8%] |
| model | (0.125, 0.25] | 53 | 19.5% | 22.6% | [13.5%, 35.5%] |
| model | (0.25, 0.375] | 179 | 31.6% | 29.1% | [22.9%, 36.1%] |
| model | (0.375, 0.5] | 311 | 42.3% | 39.5% | [34.3%, 45.1%] |
| model | (0.5, 0.625] | 377 | 58.1% | 58.4% | [53.3%, 63.2%] |
| model | (0.625, 0.75] | 280 | 68.4% | 67.5% | [61.8%, 72.7%] |
| model | (0.75, 0.875] | 136 | 80.6% | 88.2% | [81.7%, 92.6%] |
| model | (0.875, 1.0] | 17 | 89.9% | 100.0% | [81.6%, 100.0%] |
| market | (-0.001, 0.125] | 5 | 11.2% | 20.0% | [3.6%, 62.4%] |
| market | (0.125, 0.25] | 80 | 20.3% | 25.0% | [16.8%, 35.5%] |
| market | (0.25, 0.375] | 185 | 31.9% | 29.2% | [23.1%, 36.1%] |
| market | (0.375, 0.5] | 271 | 43.5% | 40.6% | [34.9%, 46.5%] |
| market | (0.5, 0.625] | 317 | 57.2% | 56.2% | [50.6%, 61.5%] |
| market | (0.625, 0.75] | 282 | 68.4% | 67.4% | [61.7%, 72.6%] |
| market | (0.75, 0.875] | 187 | 80.1% | 81.3% | [75.1%, 86.2%] |
| market | (0.875, 1.0] | 28 | 89.3% | 100.0% | [87.9%, 100.0%] |

### Model edge over the market

Mean absolute edge **0.44 pts** (median 0.35, p90 0.92). 7.7% of games have an edge above 1 point; 0.4% above 2.

## What this means

- The hybrid model is **not detectably better than the closing line** on margin, score, or Brier. Its total is the market total by construction (`total_shrink = 0`), so the total comparison is an identity, not a result.
- Winner accuracy is the easiest metric to over-read. A two- or three-game difference is a McNemar coin flip, not evidence.
- ATS/over ROI intervals are wide enough to contain any plausible edge. Do not treat a positive point estimate as a strategy.
- The useful product is the calibrated score distribution and the honest market comparison, not a claim of a betting edge.
