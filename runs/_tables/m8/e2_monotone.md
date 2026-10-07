# E2 control: monotonicity terms only (D117)

- Runs: 9 of 9 expected runs exist; missing runs, files and values: missing.txt.
- Control of M8 (D117): penalty kind monotone = relu(-f_s) + relu(f_dv) + relu(f_v) at the anchored equilibria of E2, without string term (the monotonicity constraints of RACER, dv = v - v_lead), experiment e2_monotone_w1; next to E1 and the chosen weight of E2 (D85) of the same runs: follownet_highd, fold(s) 0, seed 0.
- RMSE s: spacing RMSE of the test part (m), unit driver, with its interval over the drivers; RMSE change vs E1: relative change of the mean RMSE against E1 of the same fold and seed, paired over the drivers; p: Wilcoxon signed-rank test.
- Band shares: band_numerical of the grid speeds in support (not stable = 1 - stable); unstable among equilibria: share_unstable_numerical; max gain: largest measured gain of the audit; collided: OpenACC platoon profiles that collided, of profiles (CSV). Unit run: one run per row, so the intervals of the shares are the values themselves.
- Missing runs and files: runs/_tables/m4/missing.txt.
- Intervals: 95 % percentile bootstrap over the units, 1000 resamples, seed 0; complete: every run of the design is there and audited.

| architecture | arm | runs | drivers | RMSE s (m) | RMSE change vs E1 | p | audited | stable | unstable | outside | none | not stable | unstable among equilibria | max gain | collided profiles | complete |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mlp | E1 | 1 | 2503 | 2.84 [2.74, 2.93] |  |  | 1 | 0.09 | 0.91 [0.91, 0.91] | 0.00 | 0.00 | 0.91 [0.91, 0.91] | 0.91 [0.91, 0.91] | 1.202 | 1 | yes |
| mlp | E2 (jacobian, weight 1) | 1 | 2503 | 2.77 [2.67, 2.86] | -2.6 % [-3.6 %, -1.5 %] | <0.001 | 1 | 1.00 | 0.00 [0.00, 0.00] | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.995 | 2 | yes |
| mlp | monotone (D117) | 1 | 2503 | 2.91 [2.80, 3.00] | +2.3 % [+1.6 %, +3.0 %] | <0.001 | 1 | 0.00 | 1.00 [1.00, 1.00] | 0.00 | 0.00 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.152 | 2 | yes |
| pidl | E1 | 1 | 2503 | 2.83 [2.73, 2.92] |  |  | 1 | 0.14 | 0.86 [0.86, 0.86] | 0.00 | 0.00 | 0.86 [0.86, 0.86] | 0.86 [0.86, 0.86] | 1.201 | 0 | yes |
| pidl | E2 (jacobian, weight 10) | 1 | 2503 | 2.80 [2.70, 2.90] | -0.9 % [-2.2 %, +0.3 %] | 0.212 | 1 | 1.00 | 0.00 [0.00, 0.00] | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.996 | 0 | yes |
| pidl | monotone (D117) | 1 | 2503 | 2.86 [2.76, 2.95] | +1.1 % [+0.3 %, +1.9 %] | 0.004 | 1 | 0.05 | 0.95 [0.95, 0.95] | 0.00 | 0.00 | 0.95 [0.95, 0.95] | 0.95 [0.95, 0.95] | 1.380 | 1 | yes |
| residual_idm | E1 | 1 | 2503 | 2.71 [2.62, 2.80] |  |  | 1 | 0.73 | 0.27 [0.27, 0.27] | 0.00 | 0.00 | 0.27 [0.27, 0.27] | 0.27 [0.27, 0.27] | 1.040 | 0 | yes |
| residual_idm | E2 (jacobian, weight 0.1) | 1 | 2503 | 2.77 [2.68, 2.86] | +2.0 % [+0.8 %, +3.2 %] | <0.001 | 1 | 1.00 | 0.00 [0.00, 0.00] | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.009 | 0 | yes |
| residual_idm | monotone (D117) | 1 | 2503 | 2.78 [2.69, 2.87] | +2.5 % [+1.6 %, +3.5 %] | <0.001 | 1 | 0.50 | 0.50 [0.50, 0.50] | 0.00 | 0.00 | 0.50 [0.50, 0.50] | 0.50 [0.50, 0.50] | 1.048 | 0 | yes |
