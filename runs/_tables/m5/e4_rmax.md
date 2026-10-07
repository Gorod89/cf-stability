# E4: residual amplitude, certificate, accuracy and corridor (D111)

- Residual-amplitude sweep of the hybrid (D111): per r_max the runs on follownet_highd (certified: core with margin, certified budget; free: no certificate) and their fine-tuning on ngsim_i80, 5 folds x 5 seeds each (25 runs per experiment), and the corridor law of the fine-tuned runs. r_max 0.3 certified: e4_stable, e4_stable_ft, residual_idm_certified; free 1.0: the ResidualIDM of E1, e4_free_ft, residual_idm.
- a priori holds / holds after fine-tuning: runs whose a priori certificate (certificate.json) holds, on HighD and after the fine-tuning (no certificate step in E1). unstable among eq.: share_unstable_numerical, unit run, mean over the runs with interval. RMSE s: spacing RMSE of the test parts (m), unit driver.
- Corridor: macro error, dynamic macro error and collisions per 1000 vehicle-km of the law (table laws), unit run (scenario and seed), mean with interval.
- Extra rows of the factorial ablation of the certified hybrid (review of M8; r_max 0: no residual; not in the verdicts): idm_core_margin (r_max 0, certified core alone: no training run of its own, empty cells); idm_margin_i80 (r_max 0, I-80 margin core: no training run of its own, empty cells); residual_idm_margin_free_r0.3 (r_max 0.3, margin, no certificate: e4_margin_free_r0.3, e4_margin_free_r0.3_ft).
- Intervals: 95 % percentile bootstrap, 1000 resamples, seed 0; missing runs and files: missing.txt.

| corridor | r_max | core | runs HighD | a priori holds | runs NGSIM | holds after fine-tuning | unstable among eq. HighD | unstable among eq. NGSIM | RMSE s HighD (m) | RMSE s NGSIM (m) | corridor runs | macro error | macro error (dynamic) | collisions / 1000 veh-km |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I-80 | 0.1 | certified | 25 | 25 | 25 | 25 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 4.54 [4.48, 4.61] | 5.38 [5.26, 5.52] | 30 | 0.197 [0.177, 0.218] | 0.301 [0.261, 0.346] | 0.000 [0.000, 0.000] |
| I-80 | 0.2 | certified | 25 | 25 | 25 | 25 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 4.51 [4.44, 4.58] | 5.26 [5.14, 5.39] | 30 | 0.181 [0.158, 0.204] | 0.277 [0.232, 0.322] | 0.000 [0.000, 0.000] |
| I-80 | 0.3 | certified | 25 | 25 | 25 | 25 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 4.51 [4.45, 4.57] | 5.16 [5.04, 5.28] | 30 | 0.125 [0.109, 0.144] | 0.190 [0.159, 0.228] | 0.000 [0.000, 0.000] |
| I-80 | 0.5 | certified | 25 | 25 | 25 | 25 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 4.54 [4.48, 4.60] | 5.07 [4.95, 5.18] | 30 | 0.108 [0.092, 0.126] | 0.164 [0.128, 0.202] | 0.110 [0.000, 0.274] |
| I-80 | 0.3 | free | 25 | 0 | 25 | 0 | 0.22 [0.21, 0.23] | 0.07 [0.07, 0.08] | 2.81 [2.77, 2.86] | 5.18 [5.09, 5.28] | 30 | 0.209 [0.176, 0.249] | 0.356 [0.289, 0.437] | 0.000 [0.000, 0.000] |
| I-80 | 1 | free | 25 |  | 25 | 0 | 0.31 [0.28, 0.34] | 0.04 [0.02, 0.06] | 2.70 [2.66, 2.75] | 4.49 [4.40, 4.58] | 30 | 0.158 [0.126, 0.207] | 0.268 [0.204, 0.369] | 0.000 [0.000, 0.000] |
| I-80 | 0 | certified core alone |  |  |  |  |  |  |  |  | 30 | 0.198 [0.174, 0.226] | 0.301 [0.254, 0.356] | 0.000 [0.000, 0.000] |
| I-80 | 0 | I-80 margin core |  |  |  |  |  |  |  |  | 30 | 0.239 [0.188, 0.298] | 0.479 [0.355, 0.624] | 0.000 [0.000, 0.000] |
| I-80 | 0.3 | margin, no certificate | 25 | 0 | 25 | 0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 4.39 [4.32, 4.45] | 5.16 [5.04, 5.28] | 30 | 0.176 [0.150, 0.202] | 0.281 [0.229, 0.337] | 0.110 [0.000, 0.274] |
| US-101 | 0.1 | certified | 25 | 25 | 25 | 25 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 4.54 [4.48, 4.61] | 5.38 [5.26, 5.52] | 30 | 0.177 [0.166, 0.191] | 0.228 [0.210, 0.250] | 0.228 [0.094, 0.382] |
| US-101 | 0.2 | certified | 25 | 25 | 25 | 25 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 4.51 [4.44, 4.58] | 5.26 [5.14, 5.39] | 30 | 0.147 [0.139, 0.157] | 0.184 [0.169, 0.199] | 0.059 [0.000, 0.147] |
| US-101 | 0.3 | certified | 25 | 25 | 25 | 25 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 4.51 [4.45, 4.57] | 5.16 [5.04, 5.28] | 30 | 0.113 [0.107, 0.119] | 0.122 [0.107, 0.135] | 0.227 [0.094, 0.389] |
| US-101 | 0.5 | certified | 25 | 25 | 25 | 25 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 4.54 [4.48, 4.60] | 5.07 [4.95, 5.18] | 30 | 0.104 [0.096, 0.113] | 0.136 [0.119, 0.152] | 0.161 [0.060, 0.292] |
| US-101 | 0.3 | free | 25 | 0 | 25 | 0 | 0.22 [0.21, 0.23] | 0.07 [0.07, 0.08] | 2.81 [2.77, 2.86] | 5.18 [5.09, 5.28] | 30 | 0.194 [0.171, 0.217] | 0.312 [0.261, 0.363] | 0.035 [0.000, 0.104] |
| US-101 | 1 | free | 25 |  | 25 | 0 | 0.31 [0.28, 0.34] | 0.04 [0.02, 0.06] | 2.70 [2.66, 2.75] | 4.49 [4.40, 4.58] | 30 | 0.156 [0.135, 0.176] | 0.256 [0.213, 0.297] | 0.029 [0.000, 0.088] |
| US-101 | 0 | certified core alone |  |  |  |  |  |  |  |  | 30 | 0.177 [0.168, 0.185] | 0.222 [0.207, 0.237] | 0.422 [0.162, 0.748] |
| US-101 | 0 | I-80 margin core |  |  |  |  |  |  |  |  | 30 | 0.132 [0.115, 0.154] | 0.210 [0.179, 0.248] | 0.068 [0.000, 0.169] |
| US-101 | 0.3 | margin, no certificate | 25 | 0 | 25 | 0 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 4.39 [4.32, 4.45] | 5.16 [5.04, 5.28] | 30 | 0.147 [0.138, 0.155] | 0.189 [0.174, 0.203] | 0.296 [0.128, 0.483] |
