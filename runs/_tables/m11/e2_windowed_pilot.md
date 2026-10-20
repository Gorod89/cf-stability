# Windowed-gain penalty, pilot: choice of the weight (M11, D128)

- Runs: 12 of 12 expected runs exist; missing runs, files and values: missing.txt.
- Pilot of the windowed-gain penalty (D128, kind windowed_gain, every 4th step x 4, 16 speeds per epoch): weights 0.1, 1, 10, follownet_highd, fold 0, seed 0; one run per row.
- val / test RMSE s: mean spacing RMSE of the validation / test events (m), the test one with its interval over the drivers (events); change vs E1: relative change of the mean against the E1 run of fold 0, seed 0, paired over the events.
- Band shares of the audit (band_numerical of the grid speeds in support): stable, unstable, outside, none; not stable (poles incl.): with a pole of the full-history loop on or outside the unit circle counted unstable (full_history.json); unstable among equilibria: share_unstable_numerical; max gain: the largest numerical gain of the audit.
- Validation penalty of the best epoch (unweighted, at the grid speeds 5-30 m/s): its value, the mean relu(0.5 - M_w), the mean gain term mean_w relu(|G_w| - 0.98)^2, the mean pole term relu(rho - 0.995), the smallest M_w (1/s^2) and the largest spectral radius over the anchored equilibria; feasible epochs: validation penalty <= 0.01 (D89).
- Long window: the rollout gain over the last 252 s of a 380-s rollout at 0.02, 0.05 and 0.1 rad/s at the equilibria inside the band (long_window.json, scripts/analysis/long_window_check.py); holds: below 1.02 everywhere.
- Chosen (D85): among the weights with stable >= 0.9 the smallest validation RMSE; without such a weight the largest stable share; written to e2_windowed_chosen.json.
- Intervals: 95 % percentile bootstrap over the units, 1000 resamples, seed 0; complete: every run of the design is there and audited.

| architecture | weight | runs | val RMSE s (m) | test RMSE s (m) | test RMSE change vs E1 | stable | unstable | outside | none | not stable (poles incl.) | unstable among equilibria | max gain | epochs | best epoch | feasible epochs | val penalty (best epoch) | relu(0.5 - M_w) | gain term | pole term | min M_w | max radius | long window: max gain | long window holds | collided profiles | training (h) | chosen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| gru | 0.1 | 1 | 2.44 | 2.42 [2.34, 2.50] | +17.4 % [+14.1 %, +20.5 %] | 0.23 | 0.77 | 0.00 | 0.00 | 0.77 | 0.77 | 1.364 | 17 | 7 | 0 | 0.548 | 0.500 | 0.0052 | 0.0000 | -0.0024 | 0.9947 | 1.339 |  | 4 | 0.11 | yes |
| gru | 1 | 1 | 2.91 | 2.98 [2.86, 3.09] | +44.2 % [+39.3 %, +49.5 %] | 0.23 | 0.59 | 0.18 | 0.00 | 0.86 | 0.73 | 2.066 | 34 | 17 | 0 | 0.283 | 0.158 | 0.0141 | 0.0073 | 0.0714 | 1.0885 | 0.997 | yes | 3 | 0.28 |  |
| gru | 10 | 1 | 2.93 | 2.94 [2.84, 3.05] | +42.6 % [+37.9 %, +47.9 %] | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.113 | 21 | 11 | 9 | 0.003 | 0.000 | 0.0016 | 0.0000 | 1.2602 | 0.9952 | 0.997 | yes | 1 | 0.19 |  |
| lstm | 0.1 | 1 | 2.55 | 2.54 [2.44, 2.62] | +13.8 % [+10.7 %, +17.1 %] | 0.41 | 0.59 | 0.00 | 0.00 | 0.59 | 0.59 | 11.142 | 35 | 11 | 0 | 0.470 | 0.414 | 0.0060 | 0.0040 | -0.0388 | 1.0662 | 1.474 |  | 5 | 0.28 |  |
| lstm | 1 | 1 | 2.84 | 2.82 [2.73, 2.92] | +26.7 % [+23.2 %, +30.5 %] | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.711 | 33 | 6 | 0 | 0.500 | 0.417 | 0.0745 | 0.0007 | -0.0318 | 0.9964 | 1.159 |  | 2 | 0.26 |  |
| lstm | 10 | 1 | 2.98 | 2.94 [2.83, 3.04] | +31.8 % [+27.7 %, +35.8 %] | 0.45 | 0.55 | 0.00 | 0.00 | 0.55 | 0.55 | 1.108 | 58 | 48 | 16 | 0.002 | 0.000 | 0.0002 | 0.0015 | 0.5056 | 0.9968 | 0.995 | yes | 3 | 0.58 | yes |
| perl | 0.1 | 1 | 3.19 | 3.14 [3.02, 3.26] | +29.1 % [+24.9 %, +34.2 %] | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.321 | 14 | 3 | 0 | 0.564 | 0.498 | 0.0236 | 0.0002 | 0.0004 | 0.9970 | 1.154 |  | 5 | 0.17 |  |
| perl | 1 | 1 | 3.18 | 3.09 [2.97, 3.20] | +26.9 % [+22.6 %, +31.4 %] | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 45.693 | 32 | 22 | 0 | 0.507 | 0.500 | 0.0008 | 0.0061 | -0.0005 | 1.0038 | 2.960 |  | 5 | 0.34 | yes |
| perl | 10 | 1 | 3.27 | 3.17 [3.06, 3.29] | +30.3 % [+26.4 %, +35.1 %] | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 9.239 | 22 | 12 | 0 | 0.439 | 0.383 | 0.0183 | 0.0166 | -0.0002 | 1.0630 | 4.199 |  | 5 | 0.24 |  |

Chosen weights: gru 0.1, lstm 10, perl 1.
