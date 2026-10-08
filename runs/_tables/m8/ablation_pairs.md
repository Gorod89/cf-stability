# Factorial ablation of the certified hybrid: paired differences between its variants (review of M8)

- Runs: 1370 of 1370 expected runs have a macro.json (I-80: 22 laws x 3 scenarios x 10 seeds + 4 laws (idm_global_p0, residual_idm_certified_p0, mlp_p0, gru_p0) x 2 scenarios (i80_p1, i80_p2) x 10 seeds; US-101: 21 laws x 3 scenarios x 10 seeds); missing runs, files and values: missing.txt.
- Analysis window (scenario.json): 180-840 s, 90-870 s.
- Factorial ablation of the certified hybrid (review of M8; exploratory, not in the verdicts of H12): its variants against each other per corridor, paired by scenario and seed (unit run). Variants: A idm_global: the global IDM of ngsim_i80, no stability margin, no residual; B idm_core_margin: the IDM cores of the fold members of residual_idm_certified (margin 0.2, calibrated on follownet_highd) with the residual switched off, one core per member drawn per vehicle as residual_idm_certified draws its members; B' idm_margin_i80: the global IDM of ngsim_i80 calibrated per fold with the stability margin 0.2; C residual_idm_free_r0.3: the free core (no stability margin) with a residual of r_max 0.3 without certificate, fine-tuned on ngsim_i80 (the control of D111); D residual_idm_margin_free_r0.3: the margin core with a residual of r_max 0.3 without certificate, fine-tuned on ngsim_i80; E residual_idm_certified: the certified hybrid: the margin core with a residual of r_max 0.3 under the a priori certificate, fine-tuned on ngsim_i80.
- pair: candidate - reference by the letters of the variants, candidate and reference: their laws (ablation_pairs: [candidate, reference]; a pair with a law not in the tables has no rows, missing_corridor.txt); corridor: the corridor of the runs; metric: macro error (mean of the absolute components of the macro-error vector against the ground truth of the run's scenario, as in laws), macro error (dynamic) (of FD, wave speed, waves and wave amplitude only) and collisions / 1000 veh-km (contact episodes of followers per 1000 vehicle-km inside the window, as in laws).
- pairs: the runs (scenario and seed) of both laws with a value of the metric; mean candidate, mean reference: the means over these pairs; difference: candidate - reference per pair, mean with its interval over the pairs (negative: the candidate has the smaller error or fewer collisions); relative: mean candidate / mean reference - 1 with its interval over the pairs, only with a reference mean above 0 (no interval when resamples have a reference mean of 0); p (Wilcoxon): two-sided signed-rank test of the differences (1 when every difference is 0; not adjusted for the multiple comparisons); outcome: lower / higher when the interval of the difference lies below / above 0, else no difference.
- Intervals: 95 % percentile bootstrap, 1000 resamples, seed 0.

| corridor | pair | candidate | reference | metric | pairs | mean candidate | mean reference | difference | relative | p (Wilcoxon) | outcome |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| I-80 | E - D | residual_idm_certified | residual_idm_margin_free_r0.3 | macro error | 30 | 0.125 | 0.176 | -0.051 [-0.073, -0.028] | -29.0 % [-38.4 %, -17.8 %] | <0.001 | lower |
| I-80 | E - D | residual_idm_certified | residual_idm_margin_free_r0.3 | macro error (dynamic) | 30 | 0.190 | 0.281 | -0.092 [-0.137, -0.045] | -32.6 % [-43.7 %, -18.9 %] | <0.001 | lower |
| I-80 | E - D | residual_idm_certified | residual_idm_margin_free_r0.3 | collisions / 1000 veh-km | 30 | 0.000 | 0.110 | -0.110 [-0.274, 0.000] | -100.0 % | 0.180 | no difference |
| I-80 | E - C | residual_idm_certified | residual_idm_free_r0.3 | macro error | 30 | 0.125 | 0.209 | -0.084 [-0.130, -0.043] | -40.3 % [-52.7 %, -24.2 %] | <0.001 | lower |
| I-80 | E - C | residual_idm_certified | residual_idm_free_r0.3 | macro error (dynamic) | 30 | 0.190 | 0.356 | -0.166 [-0.254, -0.086] | -46.7 % [-60.1 %, -28.4 %] | <0.001 | lower |
| I-80 | E - C | residual_idm_certified | residual_idm_free_r0.3 | collisions / 1000 veh-km | 30 | 0.000 | 0.000 | 0.000 [0.000, 0.000] |  | 1.000 | no difference |
| I-80 | E - B | residual_idm_certified | idm_core_margin | macro error | 30 | 0.125 | 0.198 | -0.073 [-0.103, -0.044] | -36.9 % [-46.8 %, -24.6 %] | <0.001 | lower |
| I-80 | E - B | residual_idm_certified | idm_core_margin | macro error (dynamic) | 30 | 0.190 | 0.301 | -0.111 [-0.171, -0.056] | -37.0 % [-49.5 %, -20.6 %] | <0.001 | lower |
| I-80 | E - B | residual_idm_certified | idm_core_margin | collisions / 1000 veh-km | 30 | 0.000 | 0.000 | 0.000 [0.000, 0.000] |  | 1.000 | no difference |
| I-80 | D - C | residual_idm_margin_free_r0.3 | residual_idm_free_r0.3 | macro error | 30 | 0.176 | 0.209 | -0.033 [-0.082, 0.009] | -16.0 % [-34.2 %, +5.1 %] | 0.360 | no difference |
| I-80 | D - C | residual_idm_margin_free_r0.3 | residual_idm_free_r0.3 | macro error (dynamic) | 30 | 0.281 | 0.356 | -0.074 [-0.170, 0.010] | -20.9 % [-40.9 %, +3.2 %] | 0.177 | no difference |
| I-80 | D - C | residual_idm_margin_free_r0.3 | residual_idm_free_r0.3 | collisions / 1000 veh-km | 30 | 0.110 | 0.000 | 0.110 [0.000, 0.274] |  | 0.180 | no difference |
| I-80 | D - B | residual_idm_margin_free_r0.3 | idm_core_margin | macro error | 30 | 0.176 | 0.198 | -0.022 [-0.049, 0.002] | -11.1 % [-22.9 %, +1.4 %] | 0.045 | no difference |
| I-80 | D - B | residual_idm_margin_free_r0.3 | idm_core_margin | macro error (dynamic) | 30 | 0.281 | 0.301 | -0.020 [-0.075, 0.031] | -6.5 % [-22.3 %, +11.0 %] | 0.205 | no difference |
| I-80 | D - B | residual_idm_margin_free_r0.3 | idm_core_margin | collisions / 1000 veh-km | 30 | 0.110 | 0.000 | 0.110 [0.000, 0.274] |  | 0.180 | no difference |
| I-80 | B - A | idm_core_margin | idm_global | macro error | 30 | 0.198 | 0.182 | 0.015 [-0.013, 0.047] | +8.5 % [-7.1 %, +26.1 %] | 0.968 | no difference |
| I-80 | B - A | idm_core_margin | idm_global | macro error (dynamic) | 30 | 0.301 | 0.345 | -0.044 [-0.105, 0.022] | -12.8 % [-29.1 %, +6.6 %] | 0.040 | no difference |
| I-80 | B - A | idm_core_margin | idm_global | collisions / 1000 veh-km | 30 | 0.000 | 0.000 | 0.000 [0.000, 0.000] |  | 1.000 | no difference |
| I-80 | B' - A | idm_margin_i80 | idm_global | macro error | 30 | 0.239 | 0.182 | 0.057 [0.010, 0.112] | +31.3 % [+5.4 %, +60.3 %] | 0.824 | higher |
| I-80 | B' - A | idm_margin_i80 | idm_global | macro error (dynamic) | 30 | 0.479 | 0.345 | 0.134 [0.024, 0.265] | +39.0 % [+6.9 %, +73.6 %] | 0.919 | higher |
| I-80 | B' - A | idm_margin_i80 | idm_global | collisions / 1000 veh-km | 30 | 0.000 | 0.000 | 0.000 [0.000, 0.000] |  | 1.000 | no difference |
| US-101 | E - D | residual_idm_certified | residual_idm_margin_free_r0.3 | macro error | 30 | 0.113 | 0.147 | -0.034 [-0.043, -0.024] | -23.1 % [-28.4 %, -17.5 %] | <0.001 | lower |
| US-101 | E - D | residual_idm_certified | residual_idm_margin_free_r0.3 | macro error (dynamic) | 30 | 0.122 | 0.189 | -0.067 [-0.085, -0.051] | -35.5 % [-43.2 %, -28.0 %] | <0.001 | lower |
| US-101 | E - D | residual_idm_certified | residual_idm_margin_free_r0.3 | collisions / 1000 veh-km | 30 | 0.227 | 0.296 | -0.069 [-0.271, 0.123] | -23.3 % [-68.5 %, +77.6 %] | 0.388 | no difference |
| US-101 | E - C | residual_idm_certified | residual_idm_free_r0.3 | macro error | 30 | 0.113 | 0.194 | -0.081 [-0.103, -0.059] | -41.9 % [-47.9 %, -34.2 %] | <0.001 | lower |
| US-101 | E - C | residual_idm_certified | residual_idm_free_r0.3 | macro error (dynamic) | 30 | 0.122 | 0.312 | -0.190 [-0.232, -0.146] | -60.9 % [-65.6 %, -54.4 %] | <0.001 | lower |
| US-101 | E - C | residual_idm_certified | residual_idm_free_r0.3 | collisions / 1000 veh-km | 30 | 0.227 | 0.035 | 0.192 [0.064, 0.345] | +554.0 % | 0.028 | higher |
| US-101 | E - B | residual_idm_certified | idm_core_margin | macro error | 30 | 0.113 | 0.177 | -0.064 [-0.074, -0.053] | -36.1 % [-40.5 %, -31.2 %] | <0.001 | lower |
| US-101 | E - B | residual_idm_certified | idm_core_margin | macro error (dynamic) | 30 | 0.122 | 0.222 | -0.100 [-0.118, -0.083] | -45.1 % [-51.6 %, -38.5 %] | <0.001 | lower |
| US-101 | E - B | residual_idm_certified | idm_core_margin | collisions / 1000 veh-km | 30 | 0.227 | 0.422 | -0.195 [-0.550, 0.123] | -46.3 % [-82.8 %, +67.6 %] | 0.463 | no difference |
| US-101 | D - C | residual_idm_margin_free_r0.3 | residual_idm_free_r0.3 | macro error | 30 | 0.147 | 0.194 | -0.047 [-0.073, -0.021] | -24.4 % [-34.0 %, -12.3 %] | 0.014 | lower |
| US-101 | D - C | residual_idm_margin_free_r0.3 | residual_idm_free_r0.3 | macro error (dynamic) | 30 | 0.189 | 0.312 | -0.123 [-0.174, -0.071] | -39.3 % [-48.4 %, -26.8 %] | <0.001 | lower |
| US-101 | D - C | residual_idm_margin_free_r0.3 | residual_idm_free_r0.3 | collisions / 1000 veh-km | 30 | 0.296 | 0.035 | 0.261 [0.093, 0.444] | +753.1 % | 0.017 | higher |
| US-101 | D - B | residual_idm_margin_free_r0.3 | idm_core_margin | macro error | 30 | 0.147 | 0.177 | -0.030 [-0.042, -0.018] | -16.9 % [-23.3 %, -10.6 %] | <0.001 | lower |
| US-101 | D - B | residual_idm_margin_free_r0.3 | idm_core_margin | macro error (dynamic) | 30 | 0.189 | 0.222 | -0.033 [-0.055, -0.013] | -14.8 % [-23.5 %, -6.1 %] | 0.008 | lower |
| US-101 | D - B | residual_idm_margin_free_r0.3 | idm_core_margin | collisions / 1000 veh-km | 30 | 0.296 | 0.422 | -0.126 [-0.525, 0.220] | -29.9 % [-75.8 %, +116.7 %] | 0.776 | no difference |
| US-101 | B - A | idm_core_margin | idm_global | macro error | 30 | 0.177 | 0.133 | 0.043 [0.014, 0.066] | +32.6 % [+8.7 %, +58.9 %] | 0.001 | higher |
| US-101 | B - A | idm_core_margin | idm_global | macro error (dynamic) | 30 | 0.222 | 0.229 | -0.006 [-0.069, 0.040] | -2.8 % [-23.9 %, +21.3 %] | 0.088 | no difference |
| US-101 | B - A | idm_core_margin | idm_global | collisions / 1000 veh-km | 30 | 0.422 | 0.030 | 0.392 [0.132, 0.717] | +1319.5 % | 0.017 | higher |
| US-101 | B' - A | idm_margin_i80 | idm_global | macro error | 30 | 0.132 | 0.133 | -0.001 [-0.018, 0.018] | -0.6 % [-12.4 %, +15.2 %] | 0.177 | no difference |
| US-101 | B' - A | idm_margin_i80 | idm_global | macro error (dynamic) | 30 | 0.210 | 0.229 | -0.019 [-0.053, 0.017] | -8.3 % [-20.1 %, +8.9 %] | 0.033 | no difference |
| US-101 | B' - A | idm_margin_i80 | idm_global | collisions / 1000 veh-km | 30 | 0.068 | 0.030 | 0.038 [-0.059, 0.140] | +128.2 % | 0.285 | no difference |
