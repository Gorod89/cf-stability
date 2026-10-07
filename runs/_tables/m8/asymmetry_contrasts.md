# Asymmetry: penalised and certified laws against the free laws (D121)

- Penalties and certificate against the free laws (D121): every pair (law, free law), paired by scenario and seed (pairs); differences law - free law of the asymmetry index, the accelerating share, the spectral centroid, the peak frequency and the band RMS, mean with interval over the pairs; changed: the interval of the index difference excludes 0.
- |error| difference: |e_law| - |e_free| of the signed relative error of the index against the ground truth (negative: the law is closer to the truth); to the truth: closer / further when its interval excludes 0.
- Pairs: mlp_penalty vs mlp, gru_penalty vs gru, lstm_penalty vs lstm, residual_idm_certified vs residual_idm, residual_idm_certified vs residual_idm_free_r0.3. Intervals: 95 % percentile bootstrap, 1000 resamples, seed 0.

| corridor | law | free law | pairs | index | index (free) | index difference | changed | |error| difference | to the truth | accelerating share difference | centroid difference (Hz) | peak frequency difference (Hz) | band RMS difference (m/s) |
|---|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|
| I-80 | mlp_penalty | mlp | 30 | 0.350 | 0.359 | -0.008 [-0.027, 0.011] |  | 0.008 [-0.012, 0.027] | no difference | -0.014 [-0.023, -0.005] | -0.0000 [-0.0010, 0.0010] | 0.0000 [0.0000, 0.0000] | 0.299 [0.214, 0.383] |
| I-80 | gru_penalty | gru | 30 | 0.410 | 0.377 | 0.033 [0.020, 0.050] | yes | -0.035 [-0.052, -0.021] | closer | 0.028 [0.011, 0.048] | 0.0002 [-0.0005, 0.0010] | 0.0000 [0.0000, 0.0000] | 0.243 [0.187, 0.313] |
| I-80 | lstm_penalty | lstm | 30 | 0.698 | 0.610 | 0.087 [0.053, 0.122] | yes | -0.090 [-0.125, -0.054] | closer | -0.036 [-0.099, 0.028] | 0.0001 [-0.0025, 0.0026] | 0.0000 [0.0000, 0.0000] | 0.316 [-0.436, 0.887] |
| I-80 | residual_idm_certified | residual_idm | 30 | 1.281 | 0.651 | 0.630 [0.616, 0.642] | yes | -0.010 [-0.049, 0.028] | no difference | -0.138 [-0.143, -0.133] | 0.0037 [0.0031, 0.0043] | 0.0000 [0.0000, 0.0000] | 0.121 [0.083, 0.158] |
| I-80 | residual_idm_certified | residual_idm_free_r0.3 | 30 | 1.281 | 0.625 | 0.656 [0.639, 0.672] | yes | -0.037 [-0.072, -0.005] | closer | -0.161 [-0.166, -0.156] | 0.0027 [0.0021, 0.0033] | 0.0000 [0.0000, 0.0000] | -0.059 [-0.106, -0.010] |
| US-101 | mlp_penalty | mlp | 30 | 0.283 | 0.274 | 0.009 [-0.009, 0.030] |  | -0.009 [-0.030, 0.009] | no difference | -0.055 [-0.085, -0.026] | -0.0002 [-0.0008, 0.0005] | 0.0000 [0.0000, 0.0000] | 0.472 [0.376, 0.567] |
| US-101 | gru_penalty | gru | 30 | 0.344 | 0.319 | 0.024 [-0.007, 0.050] |  | -0.024 [-0.049, 0.007] | no difference | 0.060 [0.023, 0.103] | -0.0003 [-0.0006, -0.0000] | 0.0000 [0.0000, 0.0000] | 0.259 [0.181, 0.332] |
| US-101 | lstm_penalty | lstm | 30 | 0.686 | 0.619 | 0.067 [0.039, 0.095] | yes | -0.066 [-0.094, -0.039] | closer | -0.038 [-0.073, -0.002] | 0.0001 [-0.0007, 0.0010] | 0.0000 [0.0000, 0.0000] | 0.181 [0.121, 0.241] |
| US-101 | residual_idm_certified | residual_idm | 30 | 1.488 | 0.496 | 0.992 [0.975, 1.010] | yes | -0.038 [-0.072, -0.006] | closer | -0.280 [-0.293, -0.268] | 0.0013 [0.0012, 0.0014] | 0.0000 [0.0000, 0.0000] | 0.174 [0.152, 0.199] |
| US-101 | residual_idm_certified | residual_idm_free_r0.3 | 30 | 1.488 | 0.525 | 0.963 [0.955, 0.972] | yes | -0.008 [-0.035, 0.015] | no difference | -0.260 [-0.268, -0.252] | 0.0006 [0.0004, 0.0008] | 0.0000 [0.0000, 0.0000] | 0.102 [0.070, 0.131] |
