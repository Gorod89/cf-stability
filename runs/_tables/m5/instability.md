# Instability of the members and macro error of the laws

- Runs: 1370 of 1370 expected runs have a macro.json (I-80: 22 laws x 3 scenarios x 10 seeds + 4 laws (idm_global_p0, residual_idm_certified_p0, mlp_p0, gru_p0) x 2 scenarios (i80_p1, i80_p2) x 10 seeds; US-101: 21 laws x 3 scenarios x 10 seeds); missing runs, files and values: missing.txt.
- Analysis window (scenario.json): 180-840 s, 90-870 s.
- Unstable among equilibria: mean over the member runs of the audit share share_unstable_numerical (string-unstable equilibria among the equilibria found, speeds in support); band share not stable: mean over the members of 1 - band_numerical stable (speeds in support). idm_heterogeneous and idm_heterogeneous_all: per parameter set (members) the share of the grid speeds in the support of the data at which the exact discrete-time gain (dt 0.1 s, 25 frequencies 0.02-2 rad/s) exceeds 1.02, speeds without equilibrium left out, mean over the sets (audited: sets with an equilibrium in the support); the same for the certified per-event cores of residual_idm_certified_het (D118; members: its cores, the residual left out) and for an IDM law of calibrations without member runs (idm_global_p0, D120). The members are the same on every corridor.
- macro error and macro error (dynamic: FD, wave speed, waves, wave amplitude only): mean over the runs of the law on the corridor (unit run) with its interval. collides: mean collisions per 1000 vehicle-km above 0. The residual amplitudes of D111 are not in this table (correlation_pooled of M8 has them); the temporal hold-out laws of D120 (idm_global_p0, residual_idm_certified_p0, mlp_p0, gru_p0) have rows over their scenarios (column scenarios) but are not in the correlations (in the correlation).
- Correlations below: per corridor over the laws, 1000 bootstrap resamples of the laws (resamples with a constant variable have no correlation: valid resamples); p (permutation): two-sided, 10000 permutations of the instability over the laws, (1 + hits) / (1 + permutations).
- Factorial ablation of the certified hybrid (review of M8; rows only, not in the correlations nor the verdicts of H12): idm_core_margin: the IDM cores of the fold members of residual_idm_certified (margin 0.2, calibrated on follownet_highd) with the residual switched off, one core per member drawn per vehicle as residual_idm_certified draws its members; idm_margin_i80: the global IDM of ngsim_i80 calibrated per fold with the stability margin 0.2; residual_idm_margin_free_r0.3: the margin core with a residual of r_max 0.3 without certificate, fine-tuned on ngsim_i80. Their instability: the exact gain of the parameter sets for the IDM laws (members: the parameter sets), the member audits for the hybrid.
- Intervals: 95 % percentile bootstrap, 1000 resamples, seed 0.

| corridor | law | kind | members | audited | unstable among equilibria | band share not stable | runs | scenarios | macro error | macro error (dynamic) | collisions / 1000 veh-km | collides | in the correlation |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| I-80 | idm_global | idm | 5 | 5 | 0.000 | 0.000 | 30 | all | 0.182 [0.176, 0.189] | 0.345 [0.326, 0.366] | 0.000 |  | yes |
| I-80 | idm_heterogeneous | idm_heterogeneous | 2840 | 2840 | 0.092 |  | 30 | all | 0.302 [0.255, 0.357] | 0.520 [0.421, 0.632] | 0.000 |  | yes |
| I-80 | knn | models | 5 | 5 | 0.924 | 0.924 | 30 | all | 0.490 [0.460, 0.522] | 0.508 [0.460, 0.559] | 2140.503 | yes | yes |
| I-80 | mlp | models | 5 | 5 | 0.957 | 0.957 | 30 | all | 0.308 [0.274, 0.349] | 0.485 [0.411, 0.576] | 912.508 | yes | yes |
| I-80 | pidl | models | 5 | 5 | 0.908 | 0.923 | 30 | all | 0.274 [0.245, 0.311] | 0.387 [0.332, 0.466] | 1800.653 | yes | yes |
| I-80 | gru | models | 5 | 5 | 0.715 | 0.731 | 30 | all | 0.192 [0.164, 0.223] | 0.312 [0.261, 0.373] | 205.968 | yes | yes |
| I-80 | lstm | models | 5 | 5 | 0.733 | 0.748 | 30 | all | 0.716 [0.662, 0.774] | 0.657 [0.577, 0.754] | 4664.165 | yes | yes |
| I-80 | perl | models | 5 | 5 | 0.940 | 0.940 | 30 | all | 0.431 [0.396, 0.464] | 0.652 [0.586, 0.718] | 11683.262 | yes | yes |
| I-80 | residual_idm | models | 5 | 5 | 0.030 | 0.030 | 30 | all | 0.158 [0.126, 0.207] | 0.268 [0.204, 0.369] | 0.000 |  | yes |
| I-80 | residual_idm_certified | models | 5 | 5 | 0.000 | 0.000 | 30 | all | 0.125 [0.109, 0.144] | 0.190 [0.159, 0.228] | 0.000 |  | yes |
| I-80 | mlp_penalty | models | 5 | 5 | 0.000 | 0.000 | 30 | all | 0.273 [0.254, 0.289] | 0.423 [0.386, 0.459] | 1588.453 | yes | yes |
| I-80 | gru_penalty | models | 5 | 5 | 0.359 | 0.359 | 30 | all | 0.235 [0.185, 0.296] | 0.363 [0.283, 0.462] | 730.579 | yes | yes |
| I-80 | lstm_penalty | models | 5 | 5 | 0.499 | 0.499 | 30 | all | 0.734 [0.673, 0.795] | 0.684 [0.601, 0.772] | 8041.715 | yes | yes |
| I-80 | idm_heterogeneous_all | idm_heterogeneous | 7651 | 7651 | 0.165 |  | 30 | all | 0.336 [0.276, 0.406] | 0.573 [0.433, 0.738] | 0.000 |  | yes |
| I-80 | residual_idm_certified_het | residual_heterogeneous | 3337 | 3337 | 0.000 |  | 30 | all | 0.200 [0.176, 0.226] | 0.368 [0.303, 0.437] | 0.000 |  | yes |
| I-80 | idm_global_p0 | idm | 1 | 1 | 0.000 |  | 20 | i80_p1, i80_p2 | 0.186 [0.172, 0.199] | 0.322 [0.296, 0.347] | 0.000 |  |  |
| I-80 | residual_idm_certified_p0 | models | 5 | 5 | 0.000 | 0.000 | 20 | i80_p1, i80_p2 | 0.173 [0.146, 0.202] | 0.282 [0.223, 0.344] | 0.000 |  |  |
| I-80 | mlp_p0 | models | 5 | 5 | 0.941 | 0.963 | 20 | i80_p1, i80_p2 | 0.230 [0.213, 0.249] | 0.338 [0.308, 0.372] | 687.094 | yes |  |
| I-80 | gru_p0 | models | 5 | 5 | 0.791 | 0.812 | 20 | i80_p1, i80_p2 | 0.107 [0.092, 0.123] | 0.182 [0.148, 0.216] | 196.277 | yes |  |
| I-80 | idm_core_margin | idm | 5 | 5 | 0.000 |  | 30 | all | 0.198 [0.174, 0.226] | 0.301 [0.254, 0.356] | 0.000 |  |  |
| I-80 | idm_margin_i80 | idm | 5 | 5 | 0.000 |  | 30 | all | 0.239 [0.188, 0.298] | 0.479 [0.355, 0.624] | 0.000 |  |  |
| I-80 | residual_idm_margin_free_r0.3 | models | 5 | 5 | 0.000 | 0.000 | 30 | all | 0.176 [0.150, 0.202] | 0.281 [0.229, 0.337] | 0.110 | yes |  |
| US-101 | idm_global | idm | 5 | 5 | 0.000 | 0.000 | 30 | all | 0.133 [0.110, 0.163] | 0.229 [0.183, 0.288] | 0.030 | yes | yes |
| US-101 | idm_heterogeneous | idm_heterogeneous | 2840 | 2840 | 0.092 |  | 30 | all | 0.125 [0.100, 0.150] | 0.196 [0.152, 0.243] | 0.155 | yes | yes |
| US-101 | knn | models | 5 | 5 | 0.924 | 0.924 | 30 | all | 0.628 [0.574, 0.690] | 0.583 [0.505, 0.682] | 1955.966 | yes | yes |
| US-101 | mlp | models | 5 | 5 | 0.957 | 0.957 | 30 | all | 0.428 [0.395, 0.464] | 0.651 [0.544, 0.775] | 725.901 | yes | yes |
| US-101 | pidl | models | 5 | 5 | 0.908 | 0.923 | 30 | all | 0.412 [0.367, 0.462] | 0.526 [0.452, 0.612] | 1293.825 | yes | yes |
| US-101 | gru | models | 5 | 5 | 0.715 | 0.731 | 30 | all | 0.421 [0.338, 0.515] | 0.517 [0.403, 0.652] | 211.229 | yes | yes |
| US-101 | lstm | models | 5 | 5 | 0.733 | 0.748 | 30 | all | 0.888 [0.784, 1.004] | 0.837 [0.698, 0.998] | 2857.501 | yes | yes |
| US-101 | perl | models | 5 | 5 | 0.940 | 0.940 | 30 | all | 0.661 [0.621, 0.704] | 0.685 [0.642, 0.734] | 13481.471 | yes | yes |
| US-101 | residual_idm | models | 5 | 5 | 0.030 | 0.030 | 30 | all | 0.156 [0.135, 0.176] | 0.256 [0.213, 0.297] | 0.029 | yes | yes |
| US-101 | residual_idm_certified | models | 5 | 5 | 0.000 | 0.000 | 30 | all | 0.113 [0.107, 0.119] | 0.122 [0.107, 0.135] | 0.227 | yes | yes |
| US-101 | mlp_penalty | models | 5 | 5 | 0.000 | 0.000 | 30 | all | 0.380 [0.334, 0.428] | 0.518 [0.440, 0.603] | 1160.574 | yes | yes |
| US-101 | gru_penalty | models | 5 | 5 | 0.359 | 0.359 | 30 | all | 0.347 [0.278, 0.428] | 0.458 [0.354, 0.581] | 752.931 | yes | yes |
| US-101 | lstm_penalty | models | 5 | 5 | 0.499 | 0.499 | 30 | all | 0.886 [0.785, 1.020] | 0.786 [0.663, 0.938] | 5540.222 | yes | yes |
| US-101 | residual_idm_certified_het | residual_heterogeneous | 3337 | 3337 | 0.000 |  | 30 | all | 0.174 [0.159, 0.189] | 0.266 [0.234, 0.300] | 0.671 | yes | yes |
| US-101 | idm_core_margin | idm | 5 | 5 | 0.000 |  | 30 | all | 0.177 [0.168, 0.185] | 0.222 [0.207, 0.237] | 0.422 | yes |  |
| US-101 | idm_margin_i80 | idm | 5 | 5 | 0.000 |  | 30 | all | 0.132 [0.115, 0.154] | 0.210 [0.179, 0.248] | 0.068 | yes |  |
| US-101 | residual_idm_margin_free_r0.3 | models | 5 | 5 | 0.000 | 0.000 | 30 | all | 0.147 [0.138, 0.155] | 0.189 [0.174, 0.203] | 0.296 | yes |  |

## Correlation of instability and macro error over the laws

- Correlation of the mean macro error (all eight components, or the dynamic four) of a law with the instability of its members, per corridor over the laws that have both; 95 % percentile interval from 1000 resamples of the laws; p (permutation): two-sided permutation test, 10000 permutations of the instability over the laws.

| corridor | error | instability measure | subset | method | laws | correlation | valid resamples | p (permutation) |
|---|---|---|---|---|---:|---:|---:|---:|
| I-80 | macro error | unstable among equilibria | all laws | spearman | 15 | 0.652 [0.287, 0.834] | 1000 | 0.009 |
| I-80 | macro error | unstable among equilibria | all laws | pearson | 15 | 0.483 [0.223, 0.789] | 1000 | 0.067 |
| I-80 | macro error | unstable among equilibria | laws without collisions | spearman | 6 | 0.698 [-0.426, 1.000] | 980 | 0.172 |
| I-80 | macro error | unstable among equilibria | laws without collisions | pearson | 6 | 0.907 [-0.339, 0.995] | 980 | 0.024 |
| I-80 | macro error | band share not stable | all laws | spearman | 12 | 0.648 [0.177, 0.883] | 1000 | 0.026 |
| I-80 | macro error | band share not stable | all laws | pearson | 12 | 0.467 [0.072, 0.823] | 1000 | 0.125 |
| I-80 | macro error | band share not stable | laws without collisions | spearman | 3 | 0.000 [-1.000, 1.000] | 649 | 1.000 |
| I-80 | macro error | band share not stable | laws without collisions | pearson | 3 | 0.081 [-1.000, 1.000] | 649 | 1.000 |
| I-80 | macro error (dynamic) | unstable among equilibria | all laws | spearman | 15 | 0.496 [0.071, 0.762] | 1000 | 0.059 |
| I-80 | macro error (dynamic) | unstable among equilibria | all laws | pearson | 15 | 0.457 [0.075, 0.734] | 1000 | 0.084 |
| I-80 | macro error (dynamic) | unstable among equilibria | laws without collisions | spearman | 6 | 0.698 [-0.426, 1.000] | 980 | 0.172 |
| I-80 | macro error (dynamic) | unstable among equilibria | laws without collisions | pearson | 6 | 0.852 [-0.361, 0.997] | 980 | 0.024 |
| I-80 | macro error (dynamic) | band share not stable | all laws | spearman | 12 | 0.570 [0.074, 0.855] | 1000 | 0.056 |
| I-80 | macro error (dynamic) | band share not stable | all laws | pearson | 12 | 0.572 [0.169, 0.837] | 1000 | 0.055 |
| I-80 | macro error (dynamic) | band share not stable | laws without collisions | spearman | 3 | 0.000 [-1.000, 1.000] | 649 | 1.000 |
| I-80 | macro error (dynamic) | band share not stable | laws without collisions | pearson | 3 | 0.004 [-1.000, 1.000] | 649 | 1.000 |
| US-101 | macro error | unstable among equilibria | all laws | spearman | 14 | 0.744 [0.372, 0.912] | 1000 | 0.003 |
| US-101 | macro error | unstable among equilibria | all laws | pearson | 14 | 0.689 [0.433, 0.928] | 1000 | 0.008 |
| US-101 | macro error | unstable among equilibria | laws without collisions | spearman | 0 |  | 0 |  |
| US-101 | macro error | unstable among equilibria | laws without collisions | pearson | 0 |  | 0 |  |
| US-101 | macro error | band share not stable | all laws | spearman | 12 | 0.718 [0.261, 0.933] | 1000 | 0.012 |
| US-101 | macro error | band share not stable | all laws | pearson | 12 | 0.630 [0.246, 0.900] | 1000 | 0.033 |
| US-101 | macro error | band share not stable | laws without collisions | spearman | 0 |  | 0 |  |
| US-101 | macro error | band share not stable | laws without collisions | pearson | 0 |  | 0 |  |
| US-101 | macro error (dynamic) | unstable among equilibria | all laws | spearman | 14 | 0.733 [0.366, 0.933] | 1000 | 0.004 |
| US-101 | macro error (dynamic) | unstable among equilibria | all laws | pearson | 14 | 0.765 [0.562, 0.951] | 1000 | 0.002 |
| US-101 | macro error (dynamic) | unstable among equilibria | laws without collisions | spearman | 0 |  | 0 |  |
| US-101 | macro error (dynamic) | unstable among equilibria | laws without collisions | pearson | 0 |  | 0 |  |
| US-101 | macro error (dynamic) | band share not stable | all laws | spearman | 12 | 0.711 [0.297, 0.964] | 1000 | 0.013 |
| US-101 | macro error (dynamic) | band share not stable | all laws | pearson | 12 | 0.719 [0.384, 0.941] | 1000 | 0.011 |
| US-101 | macro error (dynamic) | band share not stable | laws without collisions | spearman | 0 |  | 0 |  |
| US-101 | macro error (dynamic) | band share not stable | laws without collisions | pearson | 0 |  | 0 |  |
