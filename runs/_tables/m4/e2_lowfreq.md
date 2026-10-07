# E2, low-frequency arm: combined penalty of the recurrent models (D110)

- Runs: 36 of 36 expected runs exist; missing runs, files and values: missing.txt.
- Low-frequency arm of E2 (D110): gru, lstm, perl on follownet_highd with the combined penalty (rollout gain penalty of E2 plus the Jacobian penalty of the memoryless view and the needle guard), experiments e2_combined_j0.1 etc.; the pilot runs every weight on fold(s) 0, the weight chosen by the rule of D85 on the pilot runs (or given in the configuration) on 5 folds; seed 0.
- RMSE s: spacing RMSE of the test parts (m), unit driver; RMSE change vs E1: relative change of the mean RMSE against the E1 runs of the same folds and seed 0, paired over the common drivers; p: Wilcoxon signed-rank test.
- Band shares: band_numerical of the grid speeds in support (not stable = 1 - stable); unstable among equilibria: share_unstable_numerical; unit run. max gain: median over the runs. collided: OpenACC platoon profiles that collided, of profiles, summed over the runs.
- H1.2 (combined), on the chosen weight: the rule of H1.2 (E2): confirmed when the upper end of the band share not stable < 0.1 and the upper end of the RMSE change <= +10 %; refuted when the lower end of the RMSE change > +20 % or the lower end of the share > 0.25; otherwise open. The verdict of H1.2 of table e2 stays.
- Intervals: 95 % percentile bootstrap over the units, 1000 resamples, seed 0; complete: every run of the design is there and audited.

| architecture | Jacobian weight | chosen | runs | drivers | RMSE s (m) | RMSE change vs E1 | p | audited | stable | unstable | outside | none | not stable | unstable among equilibria | max gain (median) | collided | profiles | H1.2 (combined) | complete |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| gru | 0.1 | yes | 5 | 12512 | 2.45 [2.41, 2.49] | +17.3 % [+15.1 %, +19.3 %] | <0.001 | 5 | 0.55 | 0.29 [0.16, 0.46] | 0.14 | 0.02 | 0.45 [0.25, 0.65] | 0.34 [0.21, 0.53] | 1.234 | 19 | 25 | open | yes |
| gru | 1 |  | 1 | 2503 | 2.75 [2.65, 2.84] | +33.2 % [+29.1 %, +37.4 %] | <0.001 | 1 | 0.45 | 0.45 [0.45, 0.45] | 0.09 | 0.00 | 0.55 [0.55, 0.55] | 0.50 [0.50, 0.50] | 1.123 | 3 | 5 |  | yes |
| lstm | 0.1 | yes | 5 | 12512 | 2.81 [2.76, 2.85] | +29.0 % [+27.1 %, +30.6 %] | <0.001 | 5 | 0.50 | 0.38 [0.14, 0.67] | 0.11 | 0.01 | 0.50 [0.24, 0.76] | 0.42 [0.19, 0.70] | 1.237 | 18 | 25 | refuted | yes |
| lstm | 1 |  | 1 | 2503 | 2.83 [2.73, 2.93] | +27.1 % [+23.5 %, +30.7 %] | <0.001 | 1 | 0.32 | 0.27 [0.27, 0.27] | 0.27 | 0.14 | 0.68 [0.68, 0.68] | 0.47 [0.47, 0.47] | 1.737 | 4 | 5 |  | yes |
| perl | 0.1 |  | 1 | 2503 | 2.90 [2.78, 3.01] | +19.2 % [+15.5 %, +23.4 %] | <0.001 | 1 | 0.00 | 0.00 [0.00, 0.00] | 0.00 | 1.00 | 1.00 [1.00, 1.00] |  |  | 5 | 5 |  | yes |
| perl | 1 | yes | 5 | 12512 | 3.04 [2.99, 3.09] | +28.3 % [+26.2 %, +30.2 %] | <0.001 | 5 | 0.03 | 0.14 [0.00, 0.32] | 0.19 | 0.65 | 0.97 [0.92, 1.00] | 0.91 [0.82, 1.00] | 1.188 | 15 | 25 | refuted | yes |
