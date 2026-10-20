# FD-matched weights of the heterogeneous certified hybrid (D132)

- Law residual_idm_certified_het (D118): 3337 certified per-event cores of data/calibration/ngsim_i80/idm_event_margin0.2_1b5493c173f7_certified.json (those that pass the recheck), the residuals of its five fold members; weights w_i ~ exp(beta_T T_i + beta_s s0_i) (D132).
- Truth: the binned fundamental diagram of i80_p1 (macro.json fd: 31 bins of 10 veh/km, all lanes; mean flow of the cells 7012 veh/h). Mixture: mean spacing S(v) = sum_i w_i s_i(v) + L, L = 4.972 m (mean length of the demand), s_i(v) = mean over the members of the equilibrium gap of core i with the member's residual (the cores alone: without residual); total density k = n_lanes / S(v), n_lanes = 6.208 (mean lane count of the cells of the diagram: 6.00, 7.00, 6.04, 6.00, 6.00); flow q = k v at the bin centres.
- fd_rmse_rel: RMSE of the flow over the bins / mean flow of the truth (the fd component of the macro error); fd_sse: the sum of squares that is minimised (veh²/h²). Fit: L-BFGS-B from beta = 0 within |beta| <= 5; grid best: the smallest of a 41 x 41 grid on [-5, 5]² (a check); effective cores: 1 / sum w_i²; mean T, s0: weighted means; mean spacing at 5 and 10 m/s (m).

| mixture | residual | β_T (1/s) | β_s (1/m) | FD RMSE / mean flow | FD SSE | effective cores | mean T (s) | mean s0 (m) | spacing at 5 m/s (m) | spacing at 10 m/s (m) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | members | 0.000 | 0.000 | 0.1258 | 24132344 | 3337 | 1.625 | 2.819 | 16.71 | 25.50 |
| tilted | members | 0.436 | -0.548 | 0.0673 | 6906316 | 1868 | 1.904 | 1.010 | 16.27 | 26.55 |
| grid best | members | 0.500 | -0.750 | 0.0680 | 7039013 | 1626 | 1.949 | 0.834 | 16.32 | 26.85 |
| uniform, cores alone | none | 0.000 | 0.000 | 0.1222 | 22774791 | 3337 | 1.625 | 2.819 | 15.92 | 24.10 |
| tilted, cores alone | none | 0.436 | -0.548 | 0.1030 | 16178692 | 1868 | 1.904 | 1.010 | 15.50 | 25.08 |
| tilted on the cores alone | none | 0.755 | -0.463 | 0.0673 | 6907017 | 1664 | 2.043 | 1.084 | 16.27 | 26.55 |
