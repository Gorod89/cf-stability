# Robust (non-equilibrium) certificate of the laws (D130, A3)

- Robust (non-equilibrium) certificate of D130 (cf_stability/stability/robust.py): the bounded-real LMI with one common quadratic storage function P (2 x 2) over the vertices of the polytope of the follower's perturbation dynamics e' = w - u, u' = f_s e + f_dv (u - w) + f_v u (output u; dv = v - v_lead, the convention of the project; the contract writes the f_dv term in the other sign convention), P = L L^T, gamma by bisection (relative tolerance 1e-4) above the largest vertex H-infinity norm, the largest vertex eigenvalue minimised with scipy (Nelder-Mead from several starts, then BFGS), every certificate verified in float64 (all vertex eigenvalues <= -1e-9, P positive definite). infeasible: a vertex not Hurwitz (f_s <= 0 or f_dv + f_v >= 0) or no verified certificate up to gamma 1e4.
- Every vertex with f_s > 0 has G(0) = 1: the L2 gain of the polytope is at least 1 and gamma < 1 cannot be certified; string stable = a verified certificate at gamma = 1 + 1e-3 (gamma reaches 1 only for a single frozen system: a common storage function over different systems needs a little more, e.g. 1.005 for the two ends of the derivative curve of the margin core at 5 m/s on I(v)).
- Regions of states (s, 0, v) at the grid speeds 5-30 m/s: band = the spacing band of the member's training context (NGSIM I-80; the band of the audit) where it exists; I(v) = the feasible gaps of the certificate of the member's core at its r_max (r_max 0 for B and the IDM: the equilibrium spacing). box (the rule): the core's derivative ranges over the spacings (exact: the ends of the interval) widened by the residual's Jacobian bounds (r_max lipschitz prod(norms) / (20, 2, 10)); MLP and PIDL: the ranges of their partials at 20000 states drawn uniformly over the band (a necessary-condition check: the sampled range is inside the true one). hull (supplementary, not the rule): the polytope holding the core's derivative curve over the spacings (chord and 9 tangents; the derivatives move together with s) summed with the residual box. per speed: one polytope per grid speed, gamma = the largest over the speeds; whole: one polytope over all speeds of the region (box: the union of the ranges; hull: the convex hull of the union).
- Per law: gamma = the largest over its members (and speeds); status infeasible when one polytope of a member has no certificate; lower bound: the largest vertex H-infinity norm; polytopes string stable: those with a certificate at 1 + 1e-3. Macro error: the law's mean over its runs on each corridor (runs/_tables/m5/laws.csv).
- Rule of D130 (band, box): explains when E has a string-stable certificate (gamma <= 1 + 1e-3, the reading of 'gamma < 1' above) and D has none (gamma > 1 + 1e-3 or infeasible). band, box, whole: E gamma infeasible (string stable: False), D gamma infeasible -> does not explain; band, box, per speed: E gamma infeasible (string stable: False), D gamma infeasible -> does not explain; band, hull, whole: E gamma infeasible (string stable: False), D gamma infeasible -> does not explain (supplementary variant); band, hull, per speed: E gamma infeasible (string stable: False), D gamma infeasible -> does not explain (supplementary variant).

| law | law file | region | polytope | scope | members | polytopes | status | gamma | lower bound | string stable | polytopes string stable | worst speed | macro error I-80 | macro error US-101 |
|---|---|---|---|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| E, r_max 0.1 | residual_idm_certified_r0.1 | band | box | per speed | 5 | 80 | infeasible |  | inf |  | 0 | 5 | 0.197 | 0.177 |
| E, r_max 0.1 | residual_idm_certified_r0.1 | band | box | whole | 5 | 5 | infeasible |  | inf |  | 0 |  | 0.197 | 0.177 |
| E, r_max 0.1 | residual_idm_certified_r0.1 | band | hull | per speed | 5 | 80 | infeasible |  | inf |  | 0 | 5 | 0.197 | 0.177 |
| E, r_max 0.1 | residual_idm_certified_r0.1 | band | hull | whole | 5 | 5 | infeasible |  | inf |  | 0 |  | 0.197 | 0.177 |
| E, r_max 0.1 | residual_idm_certified_r0.1 | I(v) | box | per speed | 5 | 130 | certified | 1.0444 | 1.0084 |  | 0 | 30 | 0.197 | 0.177 |
| E, r_max 0.1 | residual_idm_certified_r0.1 | I(v) | box | whole | 5 | 5 | certified | 15.8641 | 1.3336 |  | 0 |  | 0.197 | 0.177 |
| E, r_max 0.1 | residual_idm_certified_r0.1 | I(v) | hull | per speed | 5 | 130 | certified | 1.0281 | 1.0000 |  | 0 | 30 | 0.197 | 0.177 |
| E, r_max 0.1 | residual_idm_certified_r0.1 | I(v) | hull | whole | 5 | 5 | certified | 1.3827 | 1.0000 |  | 0 |  | 0.197 | 0.177 |
| E, r_max 0.2 | residual_idm_certified_r0.2 | band | box | per speed | 5 | 80 | infeasible |  | inf |  | 0 | 5 | 0.181 | 0.147 |
| E, r_max 0.2 | residual_idm_certified_r0.2 | band | box | whole | 5 | 5 | infeasible |  | inf |  | 0 |  | 0.181 | 0.147 |
| E, r_max 0.2 | residual_idm_certified_r0.2 | band | hull | per speed | 5 | 80 | infeasible |  | inf |  | 0 | 6 | 0.181 | 0.147 |
| E, r_max 0.2 | residual_idm_certified_r0.2 | band | hull | whole | 5 | 5 | infeasible |  | inf |  | 0 |  | 0.181 | 0.147 |
| E, r_max 0.2 | residual_idm_certified_r0.2 | I(v) | box | per speed | 5 | 130 | certified | 1.0981 | 1.0321 |  | 0 | 30 | 0.181 | 0.147 |
| E, r_max 0.2 | residual_idm_certified_r0.2 | I(v) | hull | per speed | 5 | 130 | certified | 1.0385 | 1.0000 |  | 0 | 30 | 0.181 | 0.147 |
| E, r_max 0.2 | residual_idm_certified_r0.2 | I(v) | box | whole | 5 | 5 | infeasible |  | 1.4617 |  | 0 |  | 0.181 | 0.147 |
| E, r_max 0.2 | residual_idm_certified_r0.2 | I(v) | hull | whole | 5 | 5 | certified | 1.6286 | 1.0000 |  | 0 |  | 0.181 | 0.147 |
| E (r_max 0.3) | residual_idm_certified | band | box | per speed | 5 | 80 | infeasible |  | inf |  | 0 | 5 | 0.125 | 0.113 |
| E (r_max 0.3) | residual_idm_certified | band | box | whole | 5 | 5 | infeasible |  | inf |  | 0 |  | 0.125 | 0.113 |
| E (r_max 0.3) | residual_idm_certified | band | hull | per speed | 5 | 80 | infeasible |  | inf |  | 0 | 6 | 0.125 | 0.113 |
| E (r_max 0.3) | residual_idm_certified | I(v) | box | per speed | 5 | 130 | certified | 1.2484 | 1.0654 |  | 0 | 30 | 0.125 | 0.113 |
| E (r_max 0.3) | residual_idm_certified | band | hull | whole | 5 | 5 | infeasible |  | inf |  | 0 |  | 0.125 | 0.113 |
| E (r_max 0.3) | residual_idm_certified | I(v) | hull | per speed | 5 | 130 | certified | 1.0550 | 1.0000 |  | 0 | 30 | 0.125 | 0.113 |
| E (r_max 0.3) | residual_idm_certified | I(v) | box | whole | 5 | 5 | infeasible |  | 1.6480 |  | 0 |  | 0.125 | 0.113 |
| E (r_max 0.3) | residual_idm_certified | I(v) | hull | whole | 5 | 5 | certified | 2.1153 | 1.0000 |  | 0 |  | 0.125 | 0.113 |
| E, r_max 0.5 | residual_idm_certified_r0.5 | band | box | per speed | 5 | 80 | infeasible |  | 7.3943 |  | 0 | 6 | 0.108 | 0.104 |
| E, r_max 0.5 | residual_idm_certified_r0.5 | band | box | whole | 5 | 5 | infeasible |  | 16.6243 |  | 0 |  | 0.108 | 0.104 |
| E, r_max 0.5 | residual_idm_certified_r0.5 | band | hull | per speed | 5 | 80 | infeasible |  | 1.0853 |  | 0 | 5 | 0.108 | 0.104 |
| E, r_max 0.5 | residual_idm_certified_r0.5 | I(v) | box | per speed | 5 | 130 | certified | 8.2422 | 1.1611 |  | 0 | 30 | 0.108 | 0.104 |
| E, r_max 0.5 | residual_idm_certified_r0.5 | band | hull | whole | 5 | 5 | infeasible |  | 1.0716 |  | 0 |  | 0.108 | 0.104 |
| E, r_max 0.5 | residual_idm_certified_r0.5 | I(v) | hull | per speed | 5 | 130 | certified | 1.1586 | 1.0000 |  | 0 | 30 | 0.108 | 0.104 |
| E, r_max 0.5 | residual_idm_certified_r0.5 | I(v) | box | whole | 5 | 5 | infeasible |  | 2.4029 |  | 0 |  | 0.108 | 0.104 |
| E, r_max 0.5 | residual_idm_certified_r0.5 | I(v) | hull | whole | 5 | 5 | certified | 11.0638 | 1.0000 |  | 0 |  | 0.108 | 0.104 |
| C | residual_idm_free_r0.3 | band | box | per speed | 5 | 80 | infeasible |  | inf |  | 0 | 5 | 0.209 | 0.194 |
| C | residual_idm_free_r0.3 | band | box | whole | 5 | 5 | infeasible |  | inf |  | 0 |  | 0.209 | 0.194 |
| C | residual_idm_free_r0.3 | band | hull | per speed | 5 | 80 | infeasible |  | inf |  | 0 | 5 | 0.209 | 0.194 |
| C | residual_idm_free_r0.3 | band | hull | whole | 5 | 5 | infeasible |  | inf |  | 0 |  | 0.209 | 0.194 |
| C | residual_idm_free_r0.3 | I(v) | box | per speed | 5 | 130 | infeasible |  | inf |  | 0 | 5 | 0.209 | 0.194 |
| C | residual_idm_free_r0.3 | I(v) | box | whole | 5 | 5 | infeasible |  | inf |  | 0 |  | 0.209 | 0.194 |
| C | residual_idm_free_r0.3 | I(v) | hull | per speed | 5 | 130 | infeasible |  | inf |  | 0 | 5 | 0.209 | 0.194 |
| C | residual_idm_free_r0.3 | I(v) | hull | whole | 5 | 5 | infeasible |  | inf |  | 0 |  | 0.209 | 0.194 |
| D | residual_idm_margin_free_r0.3 | band | box | per speed | 5 | 80 | infeasible |  | inf |  | 0 | 5 | 0.176 | 0.147 |
| D | residual_idm_margin_free_r0.3 | band | box | whole | 5 | 5 | infeasible |  | inf |  | 0 |  | 0.176 | 0.147 |
| D | residual_idm_margin_free_r0.3 | band | hull | per speed | 5 | 80 | infeasible |  | inf |  | 0 | 6 | 0.176 | 0.147 |
| D | residual_idm_margin_free_r0.3 | band | hull | whole | 5 | 5 | infeasible |  | inf |  | 0 |  | 0.176 | 0.147 |
| D | residual_idm_margin_free_r0.3 | I(v) | box | per speed | 5 | 130 | infeasible |  | inf |  | 0 | 30 | 0.176 | 0.147 |
| D | residual_idm_margin_free_r0.3 | I(v) | box | whole | 5 | 5 | infeasible |  | inf |  | 0 |  | 0.176 | 0.147 |
| D | residual_idm_margin_free_r0.3 | I(v) | hull | per speed | 5 | 130 | infeasible |  | inf |  | 0 | 30 | 0.176 | 0.147 |
| D | residual_idm_margin_free_r0.3 | I(v) | hull | whole | 5 | 5 | infeasible |  | inf |  | 0 |  | 0.176 | 0.147 |
| B | idm_core_margin | band | box | per speed | 5 | 80 | infeasible |  | 7.0309 |  | 0 | 6 | 0.198 | 0.177 |
| B | idm_core_margin | band | box | whole | 5 | 5 | infeasible |  | 14.8665 |  | 0 |  | 0.198 | 0.177 |
| B | idm_core_margin | band | hull | per speed | 5 | 80 | infeasible |  | 1.0599 |  | 0 | 5 | 0.198 | 0.177 |
| B | idm_core_margin | I(v) | box | per speed | 5 | 130 | certified | 1.0001 | 1.0000 | yes | 130 | 5 | 0.198 | 0.177 |
| B | idm_core_margin | I(v) | hull | per speed | 5 | 130 | certified | 1.0001 | 1.0000 | yes | 130 | 5 | 0.198 | 0.177 |
| B | idm_core_margin | band | hull | whole | 5 | 5 | infeasible |  | 1.0599 |  | 0 |  | 0.198 | 0.177 |
| B | idm_core_margin | I(v) | box | whole | 5 | 5 | certified | 1.5615 | 1.1516 |  | 0 |  | 0.198 | 0.177 |
| B | idm_core_margin | I(v) | hull | whole | 5 | 5 | certified | 1.1006 | 1.0000 |  | 0 |  | 0.198 | 0.177 |
| IDM (A) | idm_global | band | box | per speed | 5 | 80 | infeasible |  | 6.7158 |  | 0 | 6 | 0.182 | 0.133 |
| IDM (A) | idm_global | band | box | whole | 5 | 5 | infeasible |  | 11.5220 |  | 0 |  | 0.182 | 0.133 |
| IDM (A) | idm_global | band | hull | per speed | 5 | 80 | infeasible |  | 1.0295 |  | 0 | 5 | 0.182 | 0.133 |
| IDM (A) | idm_global | I(v) | box | per speed | 5 | 109 | certified | 1.0001 | 1.0000 | yes | 109 | 5 | 0.182 | 0.133 |
| IDM (A) | idm_global | band | hull | whole | 5 | 5 | infeasible |  | 1.0295 |  | 0 |  | 0.182 | 0.133 |
| IDM (A) | idm_global | I(v) | hull | per speed | 5 | 109 | certified | 1.0001 | 1.0000 | yes | 109 | 5 | 0.182 | 0.133 |
| IDM (A) | idm_global | I(v) | box | whole | 5 | 5 | infeasible |  | 2.8781 |  | 0 |  | 0.182 | 0.133 |
| IDM (A) | idm_global | I(v) | hull | whole | 5 | 5 | infeasible |  | 1.0000 |  | 0 |  | 0.182 | 0.133 |
| MLP | mlp | band | box | per speed | 5 | 80 | infeasible |  | inf |  | 0 | 9 | 0.308 | 0.428 |
| MLP | mlp | band | box | whole | 5 | 5 | infeasible |  | inf |  | 0 |  | 0.308 | 0.428 |
| PIDL | pidl | band | box | per speed | 5 | 80 | infeasible |  | inf |  | 0 | 8 | 0.274 | 0.412 |
| PIDL | pidl | band | box | whole | 5 | 5 | infeasible |  | inf |  | 0 |  | 0.274 | 0.412 |
