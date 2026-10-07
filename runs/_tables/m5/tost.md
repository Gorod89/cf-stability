# Equivalence of residual_idm_certified and idm_global (TOST)

- Runs: 1190 of 1190 expected runs have a macro.json (I-80: 19 laws x 3 scenarios x 10 seeds + 4 laws (idm_global_p0, residual_idm_certified_p0, mlp_p0, gru_p0) x 2 scenarios (i80_p1, i80_p2) x 10 seeds; US-101: 18 laws x 3 scenarios x 10 seeds); missing runs, files and values: missing.txt.
- Analysis window (scenario.json): 180-840 s, 90-870 s.
- TOST: residual_idm_certified against idm_global per corridor, paired by scenario and seed (pairs); relative difference mean(residual_idm_certified) / mean(idm_global) - 1 with its interval over the pairs and the Wilcoxon signed-rank test.
- Two one-sided paired t tests of the margin +-10 % of the mean of idm_global (cf_stability.eval.stats.tost_relative); equivalent when the larger p-value (p TOST) is below 0.05. A metric with a negative reference mean (wave speed) is tested on its magnitude; one whose reference mean is 0 has no relative margin.
- Intervals: 95 % percentile bootstrap, 1000 resamples, seed 0.

| corridor | metric | pairs | idm_global | residual_idm_certified | relative difference | p (Wilcoxon) | p lower | p upper | p TOST | equivalent within 10 % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| I-80 | throughput (veh/h) | 30 | 7323.273 | 7246.364 | -1.1 % [-1.4 %, -0.7 %] | <0.001 | <0.001 | <0.001 | <0.001 | yes |
| I-80 | mean speed (m/s) | 30 | 6.269 | 6.384 | +1.8 % [+1.4 %, +2.3 %] | <0.001 | <0.001 | <0.001 | <0.001 | yes |
| I-80 | queue discharge (veh/h) | 30 | 7256.654 | 6895.900 | -5.0 % [-6.7 %, -3.0 %] | <0.001 | <0.001 | <0.001 | <0.001 | yes |
| I-80 | peak 2-min flow (veh/h) | 30 | 8582.000 | 8691.000 | +1.3 % [+0.9 %, +1.6 %] | <0.001 | <0.001 | <0.001 | <0.001 | yes |
| I-80 | capacity drop | 30 | 0.154 | 0.206 | +33.7 % [+23.8 %, +41.7 %] | <0.001 | <0.001 | 1.000 | 1.000 |  |
| I-80 | congested share | 30 | 0.724 | 0.295 | -59.2 % [-63.1 %, -55.9 %] | <0.001 | 1.000 | <0.001 | 1.000 |  |
| I-80 | FD scatter (veh/h) | 30 | 581.827 | 893.092 | +53.5 % [+42.8 %, +64.7 %] | <0.001 | <0.001 | 1.000 | 1.000 |  |
| I-80 | waves | 30 | 1.100 | 2.600 | +136.4 % [+90.5 %, +220.8 %] | <0.001 | <0.001 | 1.000 | 1.000 |  |
| I-80 | wave speed, cross-correlation (m/s) | 30 | -4.282 | -5.589 | +30.5 % [+23.1 %, +38.2 %] | <0.001 | <0.001 | 1.000 | 1.000 |  |
| I-80 | wave speed, leading edges (m/s) | 20 | -7.037 | -8.277 | +17.6 % [-11.8 %, +57.8 %] | 0.202 | 0.038 | 0.669 | 0.669 |  |
| I-80 | wave amplitude (m/s) | 20 | 3.755 | 4.441 | +18.3 % [+7.0 %, +33.5 %] | 0.006 | <0.001 | 0.881 | 0.881 |  |
| I-80 | travel time mean (s) | 30 | 68.858 | 68.013 | -1.2 % [-1.5 %, -1.0 %] | <0.001 | <0.001 | <0.001 | <0.001 | yes |
| I-80 | travel time median (s) | 30 | 65.322 | 64.962 | -0.6 % [-1.4 %, +0.4 %] | 0.213 | <0.001 | <0.001 | <0.001 | yes |
| I-80 | collisions / 1000 veh-km | 30 | 0.000 | 0.000 |  |  |  |  |  |  |
| I-80 | vehicles in contact | 30 | 0.000 | 0.000 |  |  |  |  |  |  |
| I-80 | inserted share | 30 | 1.000 | 1.000 | +0.0 % [+0.0 %, +0.0 %] | 1.000 | <0.001 | <0.001 | <0.001 | yes |
| I-80 | insertion delay (s) | 30 | 0.243 | 0.240 | -1.6 % [-23.6 %, +21.0 %] | 0.871 | 0.236 | 0.168 | 0.236 |  |
| US-101 | throughput (veh/h) | 30 | 7627.231 | 7630.154 | +0.0 % [-0.1 %, +0.2 %] | 0.698 | <0.001 | <0.001 | <0.001 | yes |
| US-101 | mean speed (m/s) | 30 | 9.813 | 10.573 | +7.7 % [+7.1 %, +8.4 %] | <0.001 | <0.001 | <0.001 | <0.001 | yes |
| US-101 | queue discharge (veh/h) | 20 | 7201.070 | 8418.000 | +16.9 % [+13.4 %, +20.2 %] | <0.001 | <0.001 | 1.000 | 1.000 |  |
| US-101 | peak 2-min flow (veh/h) | 30 | 8531.000 | 8814.000 | +3.3 % [+3.0 %, +3.6 %] | <0.001 | <0.001 | <0.001 | <0.001 | yes |
| US-101 | capacity drop | 20 | 0.135 | 0.022 | -83.8 % [-92.4 %, -72.6 %] | <0.001 | 1.000 | <0.001 | 1.000 |  |
| US-101 | congested share | 30 | 0.445 | 0.027 | -93.9 % [-95.3 %, -92.6 %] | <0.001 | 1.000 | <0.001 | 1.000 |  |
| US-101 | FD scatter (veh/h) | 30 | 757.523 | 944.350 | +24.7 % [+20.1 %, +29.3 %] | <0.001 | <0.001 | 1.000 | 1.000 |  |
| US-101 | waves | 30 | 2.233 | 2.533 | +13.4 % [-1.4 %, +29.3 %] | 0.074 | 0.002 | 0.670 | 0.670 |  |
| US-101 | wave speed, cross-correlation (m/s) | 30 | -6.660 | -6.186 | -7.1 % [-25.0 %, +13.9 %] | 0.490 | 0.382 | 0.081 | 0.382 |  |
| US-101 | wave speed, leading edges (m/s) | 30 | -3.531 | -5.269 | +49.2 % [+39.7 %, +56.9 %] | <0.001 | <0.001 | 1.000 | 1.000 |  |
| US-101 | wave amplitude (m/s) | 30 | 8.392 | 7.319 | -12.8 % [-15.9 %, -9.9 %] | <0.001 | 0.953 | <0.001 | 0.953 |  |
| US-101 | travel time mean (s) | 30 | 62.858 | 58.788 | -6.5 % [-7.2 %, -5.9 %] | <0.001 | <0.001 | <0.001 | <0.001 | yes |
| US-101 | travel time median (s) | 30 | 60.652 | 56.236 | -7.3 % [-8.5 %, -6.2 %] | <0.001 | <0.001 | <0.001 | <0.001 | yes |
| US-101 | collisions / 1000 veh-km | 30 | 0.030 | 0.227 | +662.7 % | 0.028 | 0.005 | 0.993 | 0.993 |  |
| US-101 | vehicles in contact | 30 | 0.000 | 0.000 | +658.0 % | 0.028 | 0.005 | 0.993 | 0.993 |  |
| US-101 | inserted share | 30 | 1.000 | 1.000 | +0.0 % [+0.0 %, +0.0 %] | 1.000 | <0.001 | <0.001 | <0.001 | yes |
| US-101 | insertion delay (s) | 30 | 1.863 | 0.037 | -98.0 % [-98.3 %, -97.6 %] | <0.001 | 1.000 | <0.001 | 1.000 |  |
