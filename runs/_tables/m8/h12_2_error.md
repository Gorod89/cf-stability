# H12.2 on the errors: residual_idm_certified against idm_global (exploratory, review of M8)

- Runs: 1370 of 1370 expected runs have a macro.json (I-80: 22 laws x 3 scenarios x 10 seeds + 4 laws (idm_global_p0, residual_idm_certified_p0, mlp_p0, gru_p0) x 2 scenarios (i80_p1, i80_p2) x 10 seeds; US-101: 21 laws x 3 scenarios x 10 seeds); missing runs, files and values: missing.txt.
- Analysis window (scenario.json): 180-840 s, 90-870 s.
- Exploratory comparison on the errors (review of M8), not a verdict: the pre-specified H12.2 used the TOST of the raw metrics (table tost, D108: equivalence of residual_idm_certified and idm_global within +-10 % of the mean of idm_global) and stays as reported (verdicts).
- Per corridor, residual_idm_certified against idm_global, paired by scenario and seed (pairs): the absolute value |e| of every component of the macro-error vector (signed relative errors against the ground truth of the run's scenario, as in components; the macro triple first: throughput, travel time (W1), wave speed by cross-correlation) and of the macro error and the dynamic macro error (means of the absolute components; kind summary).
- difference of |e| = |e_residual_idm_certified| - |e_idm_global| per pair, mean with its interval over the pairs (negative: residual_idm_certified closer to the data); relative: mean |e_residual_idm_certified| / mean |e_idm_global| - 1 with its interval; p (Wilcoxon): signed-rank test of the differences (two-sided); p (Holm): over the eight components of a corridor; last column: smaller / larger error when the interval of the difference excludes 0, else no difference.
- Intervals: 95 % percentile bootstrap, 1000 resamples, seed 0.

| corridor | component | macro triple | kind | pairs | \|e\| idm_global | \|e\| residual_idm_certified | difference of \|e\| | relative | p (Wilcoxon) | p (Holm) | residual_idm_certified |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| I-80 | throughput | yes | anchored | 30 | 0.019 | 0.029 | 0.010 [0.006, 0.013] | +50.1 % [+44.7 %, +59.5 %] | <0.001 | <0.001 | larger error |
| I-80 | travel time (W1) | yes | anchored | 30 | 0.072 | 0.080 | 0.007 [0.005, 0.010] | +10.2 % [+6.1 %, +14.8 %] | <0.001 | <0.001 | larger error |
| I-80 | wave speed (xcorr) | yes | dynamic | 30 | 0.233 | 0.100 | -0.132 [-0.175, -0.086] | -56.9 % [-66.3 %, -43.3 %] | <0.001 | <0.001 | smaller error |
| I-80 | mean speed |  | anchored | 30 | 0.033 | 0.050 | 0.017 [0.013, 0.021] | +51.6 % [+39.2 %, +65.4 %] | <0.001 | <0.001 | larger error |
| I-80 | queue discharge |  | anchored | 30 | 0.026 | 0.081 | 0.055 [0.040, 0.068] | +208.2 % [+137.5 %, +290.6 %] | <0.001 | <0.001 | larger error |
| I-80 | FD |  | dynamic | 30 | 0.106 | 0.113 | 0.007 [0.000, 0.013] | +6.3 % [+0.1 %, +12.4 %] | 0.064 | 0.127 | larger error |
| I-80 | waves |  | dynamic | 30 | 0.736 | 0.293 | -0.442 [-0.531, -0.349] | -60.1 % [-68.1 %, -50.6 %] | <0.001 | <0.001 | smaller error |
| I-80 | wave amplitude |  | dynamic | 20 | 0.250 | 0.322 | 0.072 [-0.036, 0.217] | +28.8 % [-14.7 %, +94.4 %] | 0.571 | 0.571 | no difference |
| I-80 | macro error |  | summary | 30 | 0.182 | 0.125 | -0.058 [-0.075, -0.036] | -31.6 % [-40.6 %, -19.8 %] | <0.001 |  | smaller error |
| I-80 | macro error (dynamic) |  | summary | 30 | 0.345 | 0.190 | -0.155 [-0.196, -0.109] | -45.0 % [-54.7 %, -32.5 %] | <0.001 |  | smaller error |
| US-101 | throughput | yes | anchored | 30 | 0.014 | 0.013 | -0.001 [-0.003, 0.001] | -7.8 % [-18.2 %, +4.9 %] | 0.201 | 0.201 | no difference |
| US-101 | travel time (W1) | yes | anchored | 30 | 0.066 | 0.124 | 0.058 [0.051, 0.065] | +87.5 % [+80.2 %, +94.8 %] | <0.001 | <0.001 | larger error |
| US-101 | wave speed (xcorr) | yes | dynamic | 30 | 0.446 | 0.092 | -0.354 [-0.586, -0.203] | -79.4 % [-86.8 %, -69.6 %] | <0.001 | <0.001 | smaller error |
| US-101 | mean speed |  | anchored | 30 | 0.054 | 0.134 | 0.080 [0.073, 0.087] | +146.3 % [+130.9 %, +165.8 %] | <0.001 | <0.001 | larger error |
| US-101 | queue discharge |  | anchored | 20 | 0.016 | 0.162 | 0.146 [0.122, 0.171] | +918.0 % [+734.1 %, +1244.0 %] | <0.001 | <0.001 | larger error |
| US-101 | FD |  | dynamic | 30 | 0.107 | 0.096 | -0.011 [-0.019, -0.003] | -10.3 % [-16.8 %, -2.9 %] | 0.043 | 0.171 | smaller error |
| US-101 | waves |  | dynamic | 30 | 0.273 | 0.182 | -0.091 [-0.211, -0.002] | -33.3 % [-59.9 %, -1.1 %] | 0.063 | 0.189 | smaller error |
| US-101 | wave amplitude |  | dynamic | 30 | 0.089 | 0.119 | 0.030 [-0.002, 0.062] | +34.1 % [-1.8 %, +88.4 %] | 0.067 | 0.189 | no difference |
| US-101 | macro error |  | summary | 30 | 0.133 | 0.113 | -0.020 [-0.053, 0.003] | -15.2 % [-32.1 %, +2.9 %] | 0.871 |  | no difference |
| US-101 | macro error (dynamic) |  | summary | 30 | 0.229 | 0.122 | -0.106 [-0.170, -0.059] | -46.6 % [-60.4 %, -31.4 %] | <0.001 |  | smaller error |
