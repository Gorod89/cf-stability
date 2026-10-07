# Temporal hold-out: laws fine-tuned on period 0 against the laws of D97 (D120)

- Temporal hold-out on I-80 (D120): the laws fine-tuned on period 0 only (view ngsim_i80_p0: idm_global_p0, residual_idm_certified_p0, mlp_p0, gru_p0) against the same laws of D97 fine-tuned on all periods, both on i80_p1, i80_p2 (periods the period-0 laws did not see) x seeds 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.
- macro error and dynamic macro error: means over the runs (unit run) with intervals; difference: period-0 law - law of D97, paired by scenario and seed, mean with interval; relative: mean ratio - 1 with interval and the Wilcoxon signed-rank p; period 0 only: worse / better when the interval of the difference excludes 0.
- Intervals: 95 % percentile bootstrap, 1000 resamples, seed 0; missing runs: missing_corridor.txt.

| law (period 0) | law of D97 | runs | runs D97 | macro error | macro error D97 | difference | relative | p | dynamic | dynamic D97 | dynamic difference | collisions / 1000 veh-km | collisions D97 | period 0 only |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| idm_global_p0 | idm_global | 20 | 20 | 0.186 [0.172, 0.199] | 0.174 [0.170, 0.178] | 0.013 [-0.002, 0.026] | +7.2 % [-1.1 %, +15.2 %] | 0.202 | 0.322 [0.296, 0.347] | 0.310 [0.301, 0.319] | 0.013 [-0.016, 0.040] | 0.000 | 0.000 | no difference |
| residual_idm_certified_p0 | residual_idm_certified | 20 | 20 | 0.173 [0.146, 0.202] | 0.136 [0.116, 0.161] | 0.037 [0.010, 0.072] | +27.3 % [+6.8 %, +59.0 %] | 0.040 | 0.282 [0.223, 0.344] | 0.207 [0.165, 0.259] | 0.075 [0.022, 0.145] | 0.000 | 0.000 | worse |
| mlp_p0 | mlp | 20 | 20 | 0.230 [0.213, 0.249] | 0.259 [0.236, 0.282] | -0.029 [-0.052, -0.005] | -11.3 % [-18.7 %, -2.1 %] | 0.015 | 0.338 [0.308, 0.372] | 0.376 [0.335, 0.416] | -0.038 [-0.082, 0.009] | 687.094 | 871.054 | better |
| gru_p0 | gru | 20 | 20 | 0.107 [0.092, 0.123] | 0.147 [0.132, 0.162] | -0.039 [-0.054, -0.025] | -26.8 % [-35.4 %, -17.4 %] | <0.001 | 0.182 [0.148, 0.216] | 0.240 [0.206, 0.270] | -0.058 [-0.086, -0.026] | 196.277 | 137.249 | better |
