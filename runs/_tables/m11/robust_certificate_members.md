# Robust certificate per member (D130)

- Robust (non-equilibrium) certificate of D130 (cf_stability/stability/robust.py): the bounded-real LMI with one common quadratic storage function P (2 x 2) over the vertices of the polytope of the follower's perturbation dynamics e' = w - u, u' = f_s e + f_dv (u - w) + f_v u (output u; dv = v - v_lead, the convention of the project; the contract writes the f_dv term in the other sign convention), P = L L^T, gamma by bisection (relative tolerance 1e-4) above the largest vertex H-infinity norm, the largest vertex eigenvalue minimised with scipy (Nelder-Mead from several starts, then BFGS), every certificate verified in float64 (all vertex eigenvalues <= -1e-9, P positive definite). infeasible: a vertex not Hurwitz (f_s <= 0 or f_dv + f_v >= 0) or no verified certificate up to gamma 1e4.
- Every vertex with f_s > 0 has G(0) = 1: the L2 gain of the polytope is at least 1 and gamma < 1 cannot be certified; string stable = a verified certificate at gamma = 1 + 1e-3 (gamma reaches 1 only for a single frozen system: a common storage function over different systems needs a little more, e.g. 1.005 for the two ends of the derivative curve of the margin core at 5 m/s on I(v)).
- Regions of states (s, 0, v) at the grid speeds 5-30 m/s: band = the spacing band of the member's training context (NGSIM I-80; the band of the audit) where it exists; I(v) = the feasible gaps of the certificate of the member's core at its r_max (r_max 0 for B and the IDM: the equilibrium spacing). box (the rule): the core's derivative ranges over the spacings (exact: the ends of the interval) widened by the residual's Jacobian bounds (r_max lipschitz prod(norms) / (20, 2, 10)); MLP and PIDL: the ranges of their partials at 20000 states drawn uniformly over the band (a necessary-condition check: the sampled range is inside the true one). hull (supplementary, not the rule): the polytope holding the core's derivative curve over the spacings (chord and 9 tangents; the derivatives move together with s) summed with the residual box. per speed: one polytope per grid speed, gamma = the largest over the speeds; whole: one polytope over all speeds of the region (box: the union of the ranges; hull: the convex hull of the union).
- Per member, region, polytope and scope: gamma (the largest over the speeds for per speed).

| law | member | run | region | polytope | scope | status | gamma | lower bound | polytopes |
|---|---:|---|---|---|---|---|---:|---:|---:|
| E, r_max 0.1 | 0 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold0_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| E, r_max 0.1 | 0 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold0_seed0 | band | box | whole | infeasible |  | inf | 1 |
| E, r_max 0.1 | 0 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold0_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| E, r_max 0.1 | 0 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold0_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| E, r_max 0.1 | 0 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | box | per speed | certified | 1.0426 | 1.0084 | 26 |
| E, r_max 0.1 | 0 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | box | whole | certified | 12.0794 | 1.3281 | 1 |
| E, r_max 0.1 | 0 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | hull | per speed | certified | 1.0268 | 1.0000 | 26 |
| E, r_max 0.1 | 1 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold1_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| E, r_max 0.1 | 0 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | hull | whole | certified | 1.3687 | 1.0000 | 1 |
| E, r_max 0.1 | 1 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold1_seed0 | band | box | whole | infeasible |  | inf | 1 |
| E, r_max 0.1 | 1 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold1_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| E, r_max 0.1 | 1 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold1_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| E, r_max 0.1 | 1 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | box | per speed | certified | 1.0444 | 1.0083 | 26 |
| E, r_max 0.1 | 1 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | box | whole | certified | 15.8641 | 1.3336 | 1 |
| E, r_max 0.1 | 1 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | hull | per speed | certified | 1.0281 | 1.0000 | 26 |
| E, r_max 0.1 | 2 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold2_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| E, r_max 0.1 | 1 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | hull | whole | certified | 1.3827 | 1.0000 | 1 |
| E, r_max 0.1 | 2 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold2_seed0 | band | box | whole | infeasible |  | inf | 1 |
| E, r_max 0.1 | 2 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold2_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| E, r_max 0.1 | 2 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold2_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| E, r_max 0.1 | 2 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | box | per speed | certified | 1.0443 | 1.0083 | 26 |
| E, r_max 0.1 | 2 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | box | whole | certified | 15.3334 | 1.3330 | 1 |
| E, r_max 0.1 | 2 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | hull | per speed | certified | 1.0280 | 1.0000 | 26 |
| E, r_max 0.1 | 3 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold3_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| E, r_max 0.1 | 2 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | hull | whole | certified | 1.3812 | 1.0000 | 1 |
| E, r_max 0.1 | 3 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold3_seed0 | band | box | whole | infeasible |  | inf | 1 |
| E, r_max 0.1 | 3 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold3_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| E, r_max 0.1 | 3 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold3_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| E, r_max 0.1 | 3 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | box | per speed | certified | 1.0444 | 1.0083 | 26 |
| E, r_max 0.1 | 3 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | box | whole | certified | 14.4541 | 1.3321 | 1 |
| E, r_max 0.1 | 3 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | hull | per speed | certified | 1.0281 | 1.0000 | 26 |
| E, r_max 0.1 | 4 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold4_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| E, r_max 0.1 | 3 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | hull | whole | certified | 1.3790 | 1.0000 | 1 |
| E, r_max 0.1 | 4 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold4_seed0 | band | box | whole | infeasible |  | inf | 1 |
| E, r_max 0.1 | 4 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold4_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| E, r_max 0.1 | 4 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold4_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| E, r_max 0.1 | 4 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | box | per speed | certified | 1.0412 | 1.0084 | 26 |
| E, r_max 0.1 | 4 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | box | whole | certified | 8.9396 | 1.3204 | 1 |
| E, r_max 0.1 | 4 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | hull | per speed | certified | 1.0260 | 1.0000 | 26 |
| E, r_max 0.2 | 0 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold0_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| E, r_max 0.1 | 4 | runs/e4_stable_ft_r0.1/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | hull | whole | certified | 1.3509 | 1.0000 | 1 |
| E, r_max 0.2 | 0 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold0_seed0 | band | box | whole | infeasible |  | inf | 1 |
| E, r_max 0.2 | 0 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold0_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| E, r_max 0.2 | 0 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold0_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| E, r_max 0.2 | 0 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | box | per speed | certified | 1.0943 | 1.0321 | 26 |
| E, r_max 0.2 | 0 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | hull | per speed | certified | 1.0366 | 1.0000 | 26 |
| E, r_max 0.2 | 0 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | box | whole | infeasible |  | 1.4542 | 1 |
| E, r_max 0.2 | 1 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold1_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| E, r_max 0.2 | 0 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | hull | whole | certified | 1.6059 | 1.0000 | 1 |
| E, r_max 0.2 | 1 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold1_seed0 | band | box | whole | infeasible |  | inf | 1 |
| E, r_max 0.2 | 1 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold1_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| E, r_max 0.2 | 1 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold1_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| E, r_max 0.2 | 1 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | box | per speed | certified | 1.0981 | 1.0318 | 26 |
| E, r_max 0.2 | 1 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | box | whole | infeasible |  | 1.4617 | 1 |
| E, r_max 0.2 | 1 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | hull | per speed | certified | 1.0385 | 1.0000 | 26 |
| E, r_max 0.2 | 2 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold2_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| E, r_max 0.2 | 1 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | hull | whole | certified | 1.6286 | 1.0000 | 1 |
| E, r_max 0.2 | 2 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold2_seed0 | band | box | whole | infeasible |  | inf | 1 |
| E, r_max 0.2 | 2 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold2_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| E, r_max 0.2 | 2 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold2_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| E, r_max 0.2 | 2 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | box | per speed | certified | 1.0977 | 1.0318 | 26 |
| E, r_max 0.2 | 2 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | box | whole | infeasible |  | 1.4608 | 1 |
| E, r_max 0.2 | 2 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | hull | per speed | certified | 1.0383 | 1.0000 | 26 |
| E, r_max 0.2 | 3 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold3_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| E, r_max 0.2 | 2 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | hull | whole | certified | 1.6260 | 1.0000 | 1 |
| E, r_max 0.2 | 3 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold3_seed0 | band | box | whole | infeasible |  | inf | 1 |
| E, r_max 0.2 | 3 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold3_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| E, r_max 0.2 | 3 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold3_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| E, r_max 0.2 | 3 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | box | per speed | certified | 1.0974 | 1.0317 | 26 |
| E, r_max 0.2 | 3 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | hull | per speed | certified | 1.0384 | 1.0000 | 26 |
| E, r_max 0.2 | 3 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | box | whole | infeasible |  | 1.4588 | 1 |
| E, r_max 0.2 | 4 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold4_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| E, r_max 0.2 | 3 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | hull | whole | certified | 1.6208 | 1.0000 | 1 |
| E, r_max 0.2 | 4 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold4_seed0 | band | box | whole | infeasible |  | inf | 1 |
| E, r_max 0.2 | 4 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold4_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| E, r_max 0.2 | 4 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold4_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| E, r_max 0.2 | 4 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | box | per speed | certified | 1.0899 | 1.0319 | 26 |
| E, r_max 0.2 | 4 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | hull | per speed | certified | 1.0352 | 1.0000 | 26 |
| E, r_max 0.2 | 4 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | box | whole | infeasible |  | 1.4419 | 1 |
| E (r_max 0.3) | 0 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | band | box | per speed | infeasible |  | 7.7521 | 16 |
| E, r_max 0.2 | 4 | runs/e4_stable_ft_r0.2/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | hull | whole | certified | 1.5717 | 1.0000 | 1 |
| E (r_max 0.3) | 0 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | band | box | whole | infeasible |  | 18.0777 | 1 |
| E (r_max 0.3) | 0 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | band | hull | per speed | infeasible |  | 1.2632 | 16 |
| E (r_max 0.3) | 0 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | box | per speed | certified | 1.2387 | 1.0654 | 26 |
| E (r_max 0.3) | 0 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | band | hull | whole | infeasible |  | 1.2632 | 1 |
| E (r_max 0.3) | 0 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | hull | per speed | certified | 1.0522 | 1.0000 | 26 |
| E (r_max 0.3) | 0 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | box | whole | infeasible |  | 1.6376 | 1 |
| E (r_max 0.3) | 1 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| E (r_max 0.3) | 0 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | hull | whole | certified | 2.0716 | 1.0000 | 1 |
| E (r_max 0.3) | 1 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | band | box | whole | infeasible |  | inf | 1 |
| E (r_max 0.3) | 1 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| E (r_max 0.3) | 1 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| E (r_max 0.3) | 1 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | box | per speed | certified | 1.2484 | 1.0649 | 26 |
| E (r_max 0.3) | 1 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | hull | per speed | certified | 1.0550 | 1.0000 | 26 |
| E (r_max 0.3) | 1 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | box | whole | infeasible |  | 1.6480 | 1 |
| E (r_max 0.3) | 2 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| E (r_max 0.3) | 1 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | hull | whole | certified | 2.1153 | 1.0000 | 1 |
| E (r_max 0.3) | 2 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | band | box | whole | infeasible |  | inf | 1 |
| E (r_max 0.3) | 2 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| E (r_max 0.3) | 2 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| E (r_max 0.3) | 2 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | box | per speed | certified | 1.2473 | 1.0649 | 26 |
| E (r_max 0.3) | 2 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | hull | per speed | certified | 1.0548 | 1.0000 | 26 |
| E (r_max 0.3) | 2 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | box | whole | infeasible |  | 1.6464 | 1 |
| E (r_max 0.3) | 3 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| E (r_max 0.3) | 2 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | hull | whole | certified | 2.1094 | 1.0000 | 1 |
| E (r_max 0.3) | 3 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | band | box | whole | infeasible |  | inf | 1 |
| E (r_max 0.3) | 3 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| E (r_max 0.3) | 3 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| E (r_max 0.3) | 3 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | box | per speed | certified | 1.2446 | 1.0648 | 26 |
| E (r_max 0.3) | 3 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | hull | per speed | certified | 1.0547 | 1.0000 | 26 |
| E (r_max 0.3) | 3 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | box | whole | infeasible |  | 1.6427 | 1 |
| E (r_max 0.3) | 4 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| E (r_max 0.3) | 3 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | hull | whole | certified | 2.0958 | 1.0000 | 1 |
| E (r_max 0.3) | 4 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | band | box | whole | infeasible |  | inf | 1 |
| E (r_max 0.3) | 4 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| E (r_max 0.3) | 4 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| E (r_max 0.3) | 4 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | box | per speed | certified | 1.2221 | 1.0652 | 26 |
| E (r_max 0.3) | 4 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | box | whole | infeasible |  | 1.6171 | 1 |
| E (r_max 0.3) | 4 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | hull | per speed | certified | 1.0496 | 1.0000 | 26 |
| E, r_max 0.5 | 0 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold0_seed0 | band | box | per speed | infeasible |  | 6.7061 | 16 |
| E (r_max 0.3) | 4 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | hull | whole | certified | 1.9966 | 1.0000 | 1 |
| E, r_max 0.5 | 0 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold0_seed0 | band | box | whole | infeasible |  | 13.1963 | 1 |
| E, r_max 0.5 | 0 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold0_seed0 | band | hull | per speed | infeasible |  | 1.0775 | 16 |
| E, r_max 0.5 | 0 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | box | per speed | certified | 7.0669 | 1.1611 | 26 |
| E, r_max 0.5 | 0 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold0_seed0 | band | hull | whole | infeasible |  | 1.0636 | 1 |
| E, r_max 0.5 | 0 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | hull | per speed | certified | 1.1586 | 1.0000 | 26 |
| E, r_max 0.5 | 0 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | box | whole | infeasible |  | 2.3812 | 1 |
| E, r_max 0.5 | 1 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold1_seed0 | band | box | per speed | infeasible |  | 6.6977 | 16 |
| E, r_max 0.5 | 0 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | hull | whole | certified | 9.7827 | 1.0000 | 1 |
| E, r_max 0.5 | 1 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold1_seed0 | band | box | whole | infeasible |  | 15.1191 | 1 |
| E, r_max 0.5 | 1 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold1_seed0 | band | hull | per speed | infeasible |  | 1.0830 | 16 |
| E, r_max 0.5 | 1 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | box | per speed | certified | 8.2422 | 1.1601 | 26 |
| E, r_max 0.5 | 1 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold1_seed0 | band | hull | whole | infeasible |  | 1.0698 | 1 |
| E, r_max 0.5 | 1 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | hull | per speed | certified | 1.1579 | 1.0000 | 26 |
| E, r_max 0.5 | 1 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | box | whole | infeasible |  | 2.4029 | 1 |
| E, r_max 0.5 | 2 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold2_seed0 | band | box | per speed | infeasible |  | 6.7746 | 16 |
| E, r_max 0.5 | 1 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | hull | whole | certified | 11.0638 | 1.0000 | 1 |
| E, r_max 0.5 | 2 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold2_seed0 | band | box | whole | infeasible |  | 12.3386 | 1 |
| E, r_max 0.5 | 2 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold2_seed0 | band | hull | per speed | infeasible |  | 1.0853 | 16 |
| E, r_max 0.5 | 2 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | box | per speed | certified | 8.0220 | 1.1601 | 26 |
| E, r_max 0.5 | 2 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold2_seed0 | band | hull | whole | infeasible |  | 1.0716 | 1 |
| E, r_max 0.5 | 2 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | box | whole | infeasible |  | 2.3982 | 1 |
| E, r_max 0.5 | 2 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | hull | per speed | certified | 1.1574 | 1.0000 | 26 |
| E, r_max 0.5 | 3 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold3_seed0 | band | box | per speed | infeasible |  | 6.5475 | 16 |
| E, r_max 0.5 | 2 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | hull | whole | certified | 10.8287 | 1.0000 | 1 |
| E, r_max 0.5 | 3 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold3_seed0 | band | box | whole | infeasible |  | 15.4665 | 1 |
| E, r_max 0.5 | 3 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold3_seed0 | band | hull | per speed | infeasible |  | 1.0775 | 16 |
| E, r_max 0.5 | 3 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | box | per speed | certified | 7.4113 | 1.1598 | 26 |
| E, r_max 0.5 | 3 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold3_seed0 | band | hull | whole | infeasible |  | 1.0645 | 1 |
| E, r_max 0.5 | 3 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | hull | per speed | certified | 1.1544 | 1.0000 | 26 |
| E, r_max 0.5 | 3 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | box | whole | infeasible |  | 2.3825 | 1 |
| E, r_max 0.5 | 4 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold4_seed0 | band | box | per speed | infeasible |  | 7.3943 | 16 |
| E, r_max 0.5 | 3 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | hull | whole | certified | 10.1974 | 1.0000 | 1 |
| E, r_max 0.5 | 4 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold4_seed0 | band | box | whole | infeasible |  | 16.6243 | 1 |
| E, r_max 0.5 | 4 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold4_seed0 | band | hull | per speed | infeasible |  | 1.0782 | 16 |
| E, r_max 0.5 | 4 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | box | per speed | certified | 5.1943 | 1.1608 | 26 |
| E, r_max 0.5 | 4 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold4_seed0 | band | hull | whole | infeasible |  | 1.0657 | 1 |
| E, r_max 0.5 | 4 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | hull | per speed | certified | 1.1483 | 1.0000 | 26 |
| E, r_max 0.5 | 4 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | box | whole | infeasible |  | 2.3146 | 1 |
| C | 0 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| C | 0 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | band | box | whole | infeasible |  | inf | 1 |
| C | 0 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| C | 0 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| C | 0 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | box | per speed | infeasible |  | inf | 26 |
| C | 0 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | box | whole | infeasible |  | inf | 1 |
| C | 0 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | hull | per speed | infeasible |  | inf | 26 |
| C | 1 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| C | 0 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | hull | whole | infeasible |  | inf | 1 |
| C | 1 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | band | box | whole | infeasible |  | inf | 1 |
| C | 1 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| C | 1 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | box | per speed | infeasible |  | inf | 26 |
| C | 1 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | box | whole | infeasible |  | inf | 1 |
| C | 1 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| C | 1 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | hull | per speed | infeasible |  | inf | 26 |
| C | 2 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| C | 2 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | band | box | whole | infeasible |  | inf | 1 |
| C | 2 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| C | 1 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | hull | whole | infeasible |  | inf | 1 |
| C | 2 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | box | per speed | infeasible |  | inf | 26 |
| C | 2 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| C | 2 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | box | whole | infeasible |  | inf | 1 |
| C | 2 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | hull | per speed | infeasible |  | inf | 26 |
| C | 3 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| C | 3 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | band | box | whole | infeasible |  | inf | 1 |
| C | 3 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| C | 2 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | hull | whole | infeasible |  | inf | 1 |
| C | 3 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | box | per speed | infeasible |  | inf | 26 |
| C | 3 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| C | 3 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | box | whole | infeasible |  | inf | 1 |
| C | 3 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | hull | per speed | infeasible |  | inf | 26 |
| C | 4 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| C | 4 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | band | box | whole | infeasible |  | inf | 1 |
| C | 4 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| C | 3 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | hull | whole | infeasible |  | inf | 1 |
| C | 4 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | box | per speed | infeasible |  | inf | 26 |
| C | 4 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| C | 4 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | box | whole | infeasible |  | inf | 1 |
| C | 4 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | hull | per speed | infeasible |  | inf | 26 |
| E, r_max 0.5 | 4 | runs/e4_stable_ft_r0.5/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | hull | whole | certified | 7.6372 | 1.0000 | 1 |
| C | 4 | runs/e4_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | hull | whole | infeasible |  | inf | 1 |
| D | 0 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| D | 0 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | band | box | whole | infeasible |  | inf | 1 |
| D | 0 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| D | 0 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| D | 0 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | box | per speed | infeasible |  | inf | 26 |
| D | 0 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | box | whole | infeasible |  | inf | 1 |
| D | 0 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | hull | per speed | infeasible |  | inf | 26 |
| D | 0 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | hull | whole | infeasible |  | inf | 1 |
| D | 1 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| D | 1 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | band | box | whole | infeasible |  | inf | 1 |
| D | 1 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| D | 1 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| D | 1 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | box | per speed | infeasible |  | inf | 26 |
| D | 1 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | box | whole | infeasible |  | inf | 1 |
| D | 1 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | hull | per speed | infeasible |  | inf | 26 |
| D | 1 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | hull | whole | infeasible |  | inf | 1 |
| D | 2 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| D | 2 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | band | box | whole | infeasible |  | inf | 1 |
| D | 2 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| D | 2 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| D | 2 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | box | per speed | infeasible |  | inf | 26 |
| D | 2 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | box | whole | infeasible |  | inf | 1 |
| D | 2 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | hull | per speed | infeasible |  | inf | 26 |
| D | 2 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | hull | whole | infeasible |  | inf | 1 |
| D | 3 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| D | 3 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | band | box | whole | infeasible |  | inf | 1 |
| D | 3 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| D | 3 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| D | 3 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | box | per speed | infeasible |  | inf | 26 |
| D | 3 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | box | whole | infeasible |  | inf | 1 |
| D | 3 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | hull | per speed | infeasible |  | inf | 26 |
| D | 3 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | hull | whole | infeasible |  | inf | 1 |
| D | 4 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| D | 4 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | band | box | whole | infeasible |  | inf | 1 |
| D | 4 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | band | hull | per speed | infeasible |  | inf | 16 |
| D | 4 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | band | hull | whole | infeasible |  | inf | 1 |
| D | 4 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | box | per speed | infeasible |  | inf | 26 |
| D | 4 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | box | whole | infeasible |  | inf | 1 |
| D | 4 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | hull | per speed | infeasible |  | inf | 26 |
| D | 4 | runs/e4_margin_free_r0.3_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | hull | whole | infeasible |  | inf | 1 |
| B | 0 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | band | box | per speed | infeasible |  | 6.4020 | 16 |
| B | 0 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | band | box | whole | infeasible |  | 12.0482 | 1 |
| B | 0 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | band | hull | per speed | infeasible |  | 1.0543 | 16 |
| B | 0 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | box | per speed | certified | 1.0001 | 1.0000 | 26 |
| B | 0 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | hull | per speed | certified | 1.0001 | 1.0000 | 26 |
| B | 0 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | band | hull | whole | infeasible |  | 1.0543 | 1 |
| B | 0 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | box | whole | certified | 1.5461 | 1.1496 | 1 |
| B | 0 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold0_seed0 | I(v) | hull | whole | certified | 1.0971 | 1.0000 | 1 |
| B | 1 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | band | box | per speed | infeasible |  | 6.3565 | 16 |
| B | 1 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | band | box | whole | infeasible |  | 13.4531 | 1 |
| B | 1 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | band | hull | per speed | infeasible |  | 1.0585 | 16 |
| B | 1 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | box | per speed | certified | 1.0001 | 1.0000 | 26 |
| B | 1 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | hull | per speed | certified | 1.0001 | 1.0000 | 26 |
| B | 1 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | band | hull | whole | infeasible |  | 1.0585 | 1 |
| B | 1 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | box | whole | certified | 1.5615 | 1.1516 | 1 |
| B | 1 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold1_seed0 | I(v) | hull | whole | certified | 1.1006 | 1.0000 | 1 |
| B | 2 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | band | box | per speed | infeasible |  | 6.4218 | 16 |
| B | 2 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | band | box | whole | infeasible |  | 11.1980 | 1 |
| B | 2 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | band | hull | per speed | infeasible |  | 1.0599 | 16 |
| B | 2 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | box | per speed | certified | 1.0001 | 1.0000 | 26 |
| B | 2 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | band | hull | whole | infeasible |  | 1.0599 | 1 |
| B | 2 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | hull | per speed | certified | 1.0001 | 1.0000 | 26 |
| B | 2 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | box | whole | certified | 1.5597 | 1.1513 | 1 |
| B | 3 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | band | box | per speed | infeasible |  | 6.2165 | 16 |
| B | 2 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold2_seed0 | I(v) | hull | whole | certified | 1.1002 | 1.0000 | 1 |
| B | 3 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | band | hull | per speed | infeasible |  | 1.0540 | 16 |
| B | 3 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | band | box | whole | infeasible |  | 13.7032 | 1 |
| B | 3 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | box | per speed | certified | 1.0001 | 1.0000 | 26 |
| B | 3 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | band | hull | whole | infeasible |  | 1.0540 | 1 |
| B | 3 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | hull | per speed | certified | 1.0001 | 1.0000 | 26 |
| B | 3 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | box | whole | certified | 1.5572 | 1.1507 | 1 |
| B | 3 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold3_seed0 | I(v) | hull | whole | certified | 1.0998 | 1.0000 | 1 |
| B | 4 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | band | box | per speed | infeasible |  | 7.0309 | 16 |
| B | 4 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | band | box | whole | infeasible |  | 14.8665 | 1 |
| B | 4 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | band | hull | per speed | infeasible |  | 1.0553 | 16 |
| B | 4 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | box | per speed | certified | 1.0001 | 1.0000 | 26 |
| B | 4 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | band | hull | whole | infeasible |  | 1.0553 | 1 |
| B | 4 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | hull | per speed | certified | 1.0001 | 1.0000 | 26 |
| B | 4 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | box | whole | certified | 1.5280 | 1.1462 | 1 |
| B | 4 | runs/e4_stable_ft/ngsim_i80/residual_idm/driver_fold4_seed0 | I(v) | hull | whole | certified | 1.0938 | 1.0000 | 1 |
| IDM (A) | 0 | runs/e3_reference/ngsim_i80/idm/driver_fold0_seed0 | band | box | per speed | infeasible |  | 6.0823 | 16 |
| IDM (A) | 0 | runs/e3_reference/ngsim_i80/idm/driver_fold0_seed0 | band | box | whole | infeasible |  | 9.4002 | 1 |
| IDM (A) | 0 | runs/e3_reference/ngsim_i80/idm/driver_fold0_seed0 | band | hull | per speed | infeasible |  | 1.0237 | 16 |
| IDM (A) | 0 | runs/e3_reference/ngsim_i80/idm/driver_fold0_seed0 | I(v) | box | per speed | certified | 1.0001 | 1.0000 | 22 |
| IDM (A) | 0 | runs/e3_reference/ngsim_i80/idm/driver_fold0_seed0 | band | hull | whole | infeasible |  | 1.0231 | 1 |
| IDM (A) | 0 | runs/e3_reference/ngsim_i80/idm/driver_fold0_seed0 | I(v) | hull | per speed | certified | 1.0001 | 1.0000 | 22 |
| IDM (A) | 0 | runs/e3_reference/ngsim_i80/idm/driver_fold0_seed0 | I(v) | box | whole | infeasible |  | 2.7907 | 1 |
| IDM (A) | 0 | runs/e3_reference/ngsim_i80/idm/driver_fold0_seed0 | I(v) | hull | whole | certified | 87.6608 | 1.0000 | 1 |
| IDM (A) | 1 | runs/e3_reference/ngsim_i80/idm/driver_fold1_seed0 | band | box | per speed | infeasible |  | 6.0722 | 16 |
| IDM (A) | 1 | runs/e3_reference/ngsim_i80/idm/driver_fold1_seed0 | band | box | whole | infeasible |  | 10.4639 | 1 |
| IDM (A) | 1 | runs/e3_reference/ngsim_i80/idm/driver_fold1_seed0 | band | hull | per speed | infeasible |  | 1.0294 | 16 |
| IDM (A) | 1 | runs/e3_reference/ngsim_i80/idm/driver_fold1_seed0 | I(v) | box | per speed | certified | 1.0001 | 1.0000 | 22 |
| IDM (A) | 1 | runs/e3_reference/ngsim_i80/idm/driver_fold1_seed0 | band | hull | whole | infeasible |  | 1.0085 | 1 |
| IDM (A) | 1 | runs/e3_reference/ngsim_i80/idm/driver_fold1_seed0 | I(v) | hull | per speed | certified | 1.0001 | 1.0000 | 22 |
| IDM (A) | 1 | runs/e3_reference/ngsim_i80/idm/driver_fold1_seed0 | I(v) | box | whole | infeasible |  | 1.9422 | 1 |
| IDM (A) | 1 | runs/e3_reference/ngsim_i80/idm/driver_fold1_seed0 | I(v) | hull | whole | certified | 2.8366 | 1.0000 | 1 |
| IDM (A) | 2 | runs/e3_reference/ngsim_i80/idm/driver_fold2_seed0 | band | box | per speed | infeasible |  | 6.1246 | 16 |
| IDM (A) | 2 | runs/e3_reference/ngsim_i80/idm/driver_fold2_seed0 | band | box | whole | infeasible |  | 8.7600 | 1 |
| IDM (A) | 2 | runs/e3_reference/ngsim_i80/idm/driver_fold2_seed0 | band | hull | per speed | infeasible |  | 1.0295 | 16 |
| IDM (A) | 2 | runs/e3_reference/ngsim_i80/idm/driver_fold2_seed0 | I(v) | box | per speed | certified | 1.0001 | 1.0000 | 22 |
| IDM (A) | 2 | runs/e3_reference/ngsim_i80/idm/driver_fold2_seed0 | I(v) | hull | per speed | certified | 1.0001 | 1.0000 | 22 |
| IDM (A) | 2 | runs/e3_reference/ngsim_i80/idm/driver_fold2_seed0 | band | hull | whole | infeasible |  | 1.0295 | 1 |
| IDM (A) | 2 | runs/e3_reference/ngsim_i80/idm/driver_fold2_seed0 | I(v) | box | whole | infeasible |  | 1.8874 | 1 |
| IDM (A) | 2 | runs/e3_reference/ngsim_i80/idm/driver_fold2_seed0 | I(v) | hull | whole | certified | 2.5730 | 1.0000 | 1 |
| IDM (A) | 3 | runs/e3_reference/ngsim_i80/idm/driver_fold3_seed0 | band | box | per speed | infeasible |  | 5.9554 | 16 |
| IDM (A) | 3 | runs/e3_reference/ngsim_i80/idm/driver_fold3_seed0 | band | box | whole | infeasible |  | 10.4400 | 1 |
| IDM (A) | 3 | runs/e3_reference/ngsim_i80/idm/driver_fold3_seed0 | band | hull | per speed | infeasible |  | 1.0272 | 16 |
| IDM (A) | 3 | runs/e3_reference/ngsim_i80/idm/driver_fold3_seed0 | I(v) | box | per speed | certified | 1.0001 | 1.0000 | 21 |
| IDM (A) | 3 | runs/e3_reference/ngsim_i80/idm/driver_fold3_seed0 | band | hull | whole | infeasible |  | 1.0039 | 1 |
| IDM (A) | 3 | runs/e3_reference/ngsim_i80/idm/driver_fold3_seed0 | I(v) | hull | per speed | certified | 1.0001 | 1.0000 | 21 |
| IDM (A) | 3 | runs/e3_reference/ngsim_i80/idm/driver_fold3_seed0 | I(v) | box | whole | infeasible |  | 2.1548 | 1 |
| IDM (A) | 3 | runs/e3_reference/ngsim_i80/idm/driver_fold3_seed0 | I(v) | hull | whole | certified | 4.2693 | 1.0000 | 1 |
| IDM (A) | 4 | runs/e3_reference/ngsim_i80/idm/driver_fold4_seed0 | band | box | per speed | infeasible |  | 6.7158 | 16 |
| IDM (A) | 4 | runs/e3_reference/ngsim_i80/idm/driver_fold4_seed0 | band | box | whole | infeasible |  | 11.5220 | 1 |
| IDM (A) | 4 | runs/e3_reference/ngsim_i80/idm/driver_fold4_seed0 | band | hull | per speed | infeasible |  | 1.0266 | 16 |
| IDM (A) | 4 | runs/e3_reference/ngsim_i80/idm/driver_fold4_seed0 | I(v) | box | per speed | certified | 1.0001 | 1.0000 | 22 |
| IDM (A) | 4 | runs/e3_reference/ngsim_i80/idm/driver_fold4_seed0 | band | hull | whole | infeasible |  | 1.0007 | 1 |
| IDM (A) | 4 | runs/e3_reference/ngsim_i80/idm/driver_fold4_seed0 | I(v) | hull | per speed | certified | 1.0001 | 1.0000 | 22 |
| IDM (A) | 4 | runs/e3_reference/ngsim_i80/idm/driver_fold4_seed0 | I(v) | box | whole | infeasible |  | 2.8781 | 1 |
| MLP | 0 | runs/e4_free_ft/ngsim_i80/mlp/driver_fold0_seed0 | band | box | per speed | infeasible |  | 1.9317 | 16 |
| IDM (A) | 4 | runs/e3_reference/ngsim_i80/idm/driver_fold4_seed0 | I(v) | hull | whole | infeasible |  | 1.0000 | 1 |
| MLP | 1 | runs/e4_free_ft/ngsim_i80/mlp/driver_fold1_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| MLP | 0 | runs/e4_free_ft/ngsim_i80/mlp/driver_fold0_seed0 | band | box | whole | infeasible |  | 3.4799 | 1 |
| MLP | 1 | runs/e4_free_ft/ngsim_i80/mlp/driver_fold1_seed0 | band | box | whole | infeasible |  | inf | 1 |
| MLP | 2 | runs/e4_free_ft/ngsim_i80/mlp/driver_fold2_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| MLP | 2 | runs/e4_free_ft/ngsim_i80/mlp/driver_fold2_seed0 | band | box | whole | infeasible |  | inf | 1 |
| MLP | 3 | runs/e4_free_ft/ngsim_i80/mlp/driver_fold3_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| MLP | 3 | runs/e4_free_ft/ngsim_i80/mlp/driver_fold3_seed0 | band | box | whole | infeasible |  | inf | 1 |
| MLP | 4 | runs/e4_free_ft/ngsim_i80/mlp/driver_fold4_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| MLP | 4 | runs/e4_free_ft/ngsim_i80/mlp/driver_fold4_seed0 | band | box | whole | infeasible |  | inf | 1 |
| PIDL | 0 | runs/m5_ft/ngsim_i80/pidl/driver_fold0_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| PIDL | 0 | runs/m5_ft/ngsim_i80/pidl/driver_fold0_seed0 | band | box | whole | infeasible |  | inf | 1 |
| PIDL | 1 | runs/m5_ft/ngsim_i80/pidl/driver_fold1_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| PIDL | 1 | runs/m5_ft/ngsim_i80/pidl/driver_fold1_seed0 | band | box | whole | infeasible |  | inf | 1 |
| PIDL | 2 | runs/m5_ft/ngsim_i80/pidl/driver_fold2_seed0 | band | box | per speed | infeasible |  | 5.1798 | 16 |
| PIDL | 3 | runs/m5_ft/ngsim_i80/pidl/driver_fold3_seed0 | band | box | per speed | infeasible |  | inf | 16 |
| PIDL | 2 | runs/m5_ft/ngsim_i80/pidl/driver_fold2_seed0 | band | box | whole | infeasible |  | 6.7244 | 1 |
| PIDL | 3 | runs/m5_ft/ngsim_i80/pidl/driver_fold3_seed0 | band | box | whole | infeasible |  | inf | 1 |
| PIDL | 4 | runs/m5_ft/ngsim_i80/pidl/driver_fold4_seed0 | band | box | per speed | infeasible |  | 7.4087 | 16 |
| PIDL | 4 | runs/m5_ft/ngsim_i80/pidl/driver_fold4_seed0 | band | box | whole | infeasible |  | 9.9014 | 1 |
