# Sensitivity to the downstream boundary and the lane-change model (D113)

- Sensitivity of the corridor results (D113): variants of i80_p1 (scenario i80_p1_<variant>: gain0.01, gain0.04, nofeedback, lc_low, lc_high) and the scenario itself (baseline), laws idm_global, residual_idm_certified, gru, seeds 0, 1, 2, 3, 4 (the baseline with the same seeds).
- macro error and the macro triple (signed relative errors of throughput, travel time (W1) and wave speed by cross-correlation against the ground truth of the variant's scenario), collisions per 1000 vehicle-km: unit run (seed), mean with interval.
- rank: 1 = smallest mean macro error within the variant; the last column marks the variants whose ranking is residual_idm_certified < idm_global < gru (empty while a law has no runs).
- Intervals: 95 % percentile bootstrap, 1000 resamples, seed 0; missing runs and files: missing.txt.

| variant | law | runs | macro error | throughput | travel time (W1) | wave speed (xcorr) | collisions / 1000 veh-km | rank | ranking by macro error | residual_idm_certified < idm_global < gru |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| baseline | idm_global | 5 | 0.177 [0.169, 0.184] | -0.004 [-0.005, -0.004] | 0.072 [0.069, 0.076] | 0.208 [0.178, 0.241] | 0.000 [0.000, 0.000] | 3 | residual_idm_certified < gru < idm_global |  |
| baseline | residual_idm_certified | 5 | 0.131 [0.098, 0.164] | -0.007 [-0.008, -0.007] | 0.079 [0.076, 0.082] | 0.112 [0.064, 0.168] | 0.000 [0.000, 0.000] | 1 | residual_idm_certified < gru < idm_global |  |
| baseline | gru | 5 | 0.170 [0.161, 0.183] | -0.042 [-0.091, -0.010] | 0.146 [0.102, 0.207] | 0.272 [0.236, 0.321] | 173.325 [138.960, 204.050] | 2 | residual_idm_certified < gru < idm_global |  |
| gain0.01 | idm_global | 5 | 0.165 [0.141, 0.178] | -0.004 [-0.004, -0.003] | 0.110 [0.106, 0.115] | 0.265 [0.254, 0.276] | 0.000 [0.000, 0.000] | 1 | idm_global < gru < residual_idm_certified |  |
| gain0.01 | residual_idm_certified | 5 | 0.230 [0.198, 0.276] | -0.010 [-0.011, -0.009] | 0.122 [0.120, 0.124] | 0.176 [0.143, 0.211] | 0.000 [0.000, 0.000] | 3 | idm_global < gru < residual_idm_certified |  |
| gain0.01 | gru | 5 | 0.183 [0.150, 0.212] | -0.026 [-0.047, -0.007] | 0.121 [0.083, 0.158] | 0.239 [0.236, 0.241] | 169.670 [134.083, 204.041] | 2 | idm_global < gru < residual_idm_certified |  |
| gain0.04 | idm_global | 5 | 0.161 [0.153, 0.172] | -0.002 [-0.003, -0.002] | 0.063 [0.060, 0.068] | 0.278 [0.197, 0.355] | 0.000 [0.000, 0.000] | 1 | idm_global < residual_idm_certified < gru |  |
| gain0.04 | residual_idm_certified | 5 | 0.197 [0.164, 0.224] | -0.005 [-0.006, -0.004] | 0.063 [0.061, 0.065] | 0.102 [0.007, 0.192] | 0.000 [0.000, 0.000] | 2 | idm_global < residual_idm_certified < gru |  |
| gain0.04 | gru | 5 | 0.222 [0.205, 0.237] | -0.040 [-0.091, 0.008] | 0.149 [0.125, 0.174] | 0.271 [0.256, 0.293] | 176.914 [138.122, 231.205] | 3 | idm_global < residual_idm_certified < gru |  |
| nofeedback | idm_global | 5 | 0.273 [0.271, 0.275] | 0.044 [0.042, 0.046] | 0.303 [0.295, 0.310] | 0.243 [0.230, 0.258] | 0.000 [0.000, 0.000] | 2 | gru < idm_global < residual_idm_certified |  |
| nofeedback | residual_idm_certified | 5 | 0.369 [0.363, 0.375] | 0.042 [0.040, 0.044] | 0.444 [0.440, 0.449] | -0.072 [-0.105, -0.038] | 0.000 [0.000, 0.000] | 3 | gru < idm_global < residual_idm_certified |  |
| nofeedback | gru | 5 | 0.202 [0.187, 0.220] | -0.058 [-0.080, -0.040] | 0.158 [0.113, 0.195] | 0.244 [0.224, 0.260] | 176.086 [137.974, 210.385] | 1 | gru < idm_global < residual_idm_certified |  |
| lc_low | idm_global | 5 | 0.184 [0.173, 0.194] | -0.004 [-0.005, -0.004] | 0.146 [0.136, 0.157] | 0.319 [0.270, 0.369] | 0.000 [0.000, 0.000] | 1 | idm_global < residual_idm_certified < gru |  |
| lc_low | residual_idm_certified | 5 | 0.227 [0.178, 0.273] | -0.009 [-0.010, -0.009] | 0.139 [0.132, 0.147] | 0.104 [-0.012, 0.219] | 0.000 [0.000, 0.000] | 2 | idm_global < residual_idm_certified < gru |  |
| lc_low | gru | 5 | 0.252 [0.211, 0.306] | -0.091 [-0.184, -0.014] | 0.185 [0.176, 0.194] | 0.109 [-0.107, 0.321] | 133.160 [98.297, 175.094] | 3 | idm_global < residual_idm_certified < gru |  |
| lc_high | idm_global | 5 | 0.156 [0.151, 0.160] | -0.005 [-0.005, -0.004] | 0.046 [0.043, 0.050] | 0.207 [0.162, 0.235] | 0.000 [0.000, 0.000] | 2 | gru < idm_global < residual_idm_certified |  |
| lc_high | residual_idm_certified | 5 | 0.201 [0.164, 0.247] | -0.008 [-0.009, -0.007] | 0.060 [0.057, 0.061] | 0.189 [0.156, 0.222] | 0.000 [0.000, 0.000] | 3 | gru < idm_global < residual_idm_certified |  |
| lc_high | gru | 5 | 0.156 [0.121, 0.192] | -0.029 [-0.049, -0.009] | 0.105 [0.085, 0.125] | 0.249 [0.230, 0.263] | 310.180 [299.425, 320.830] | 1 | gru < idm_global < residual_idm_certified |  |
