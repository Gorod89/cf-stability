# Full-history linear analysis of the windowed-gain penalty (M11, D128)

- Runs: GRU, LSTM and PERL on follownet_highd, seed 0 (M11, D128). E1 (seed 0): no penalty, five folds. E2 windowed (chosen weight): the windowed-gain penalty (kind windowed_gain: mean over the anchored equilibria of mean_w relu(|G_w| - 0.98)^2 at the 25 audit frequencies + relu(0.5 - M_w) + relu(rho - 0.995), plus the existence term) at the weight chosen by the rule of D85 on the pilot (gru e2_windowed_w0.1, lstm e2_windowed_w10, perl e2_windowed_w1), five folds; E2 windowed w<weight> (pilot): the pilot weights 0.1, 1 and 10 on fold 0 (the chosen one is also in the row above). Speeds: every grid speed with an equilibrium in the stored audit (status ok, multiple or outside: inside or outside the band), at the stored equilibrium; part all = every such speed, support = those in the speed range of the training data.
- Linearisation: the history Jacobian J [30, 3] of the acceleration with respect to the 30 states of the window (history_jacobian). The networks are re-run over the last 30 states at every step, without a hidden state carried between steps, so J is the exact linearisation of the rollout. State-space form of the two-vehicle loop with the leader speed as input and the follower speed as output, semi-implicit Euler at 0.1 s as in the rollout (windowed_state_space, cf_stability/stability/analytic.py): state (e[k], u[k-29..k], w[k-29..k]) of the gap, follower and leader speed deviations.
- poles inside: share of the speeds whose 31 closed-loop poles (the eigenvalues of the block of e and u; the other 30 are 0) all lie strictly inside the unit circle: local stability of the full history model. memoryless locally stable (reference): f_s > 0 and f_v + f_dv < 0 of the memoryless view (derivatives summed over the window).
- windowed > 1.02: share of the speeds whose windowed gain |G(e^(iω dt))| (transfer_windowed) exceeds 1.02 at some of the 25 audit frequencies (0.02-2 rad/s; strict inequality, the rule of the audit). numerical unstable (reference): the audit's verdict from the rollouts. agreement: share of the speeds where the two verdicts are equal (a speed whose rollout broke down has no numerical verdict and is left out). \|num - win\| at 0.02: median (90 % quantile) over the speeds with all poles inside and an unflagged rollout at 0.02 rad/s (pooled over the runs) of the difference between the audit's numerical gain and the windowed gain at that frequency.
- Low frequency: sign(M) vs \|G_w(0.02)\| is the share of the speeds where M < 0 (M = f_v² + 2 f_v f_dv - 2 f_s of the memoryless view) exactly when the windowed gain at the lowest audit frequency 0.02 rad/s is above 1; sign(M_w) the same for M_w = M + 2 (f_v m_s - f_s m_v) - dt f_s f_v, the exact coefficient of the low-frequency expansion of the windowed gain, |G|² = 1 - ω² M_w / f_s² + O(ω⁴) (windowed_margin; m_x = sum over the lags of lag x J_x, the first moments of the history Jacobian; the last term is the Euler step); sign(M) = sign(M_w): the memoryless criterion gives the ω -> 0 limit of the windowed gain. M is the coefficient only when f_v m_s = f_s m_v (no memory, or a lag common to spacing and speed). At 0.02 rad/s the higher-order terms can decide the side of 1 when |M_w| is small: PERL (Newell's feed-forward of the leader acceleration has m_dv = -m_v = 1, so M_w ~ M + 2 f_s) under the combined penalty.
- Unit run: the shares per run over its speeds, mean over the runs with the 95 % percentile bootstrap interval over the runs (1000 resamples, seed 0); a run without speeds in the part is left out of it.
- Checks: the recomputed maximum windowed gain equals the stored analytic_max_gain.windowed of the audits to 7.9e-12 (largest absolute difference); the agreement column equals the audit's agreement_gain to 0.0e+00; at 0.0001 rad/s the windowed gain is above 1 exactly when M_w < 0 at a share 1.000 or more of the speeds (mean over the runs, smallest row). Per-run details (poles, gains, Jacobian): full_history.json next to stability.json.

Poles and gain verdicts:

| architecture | arm | part | runs | speeds | poles inside | memoryless locally stable | windowed > 1.02 | numerical unstable | agreement windowed vs numerical | \|num - win\| at 0.02 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gru | E1 (seed 0) | all | 5 | 124 | 0.97 [0.91, 1.00] | 0.98 | 0.87 [0.75, 0.98] | 0.87 | 1.00 [1.00, 1.00] | 5.7e-05 (5.9e-04) |
| gru | E1 (seed 0) | support | 5 | 104 | 1.00 [1.00, 1.00] | 0.98 | 0.89 [0.79, 0.99] | 0.89 | 1.00 [1.00, 1.00] | 5.5e-05 (8.0e-04) |
| gru | E2 windowed (chosen weight) | all | 5 | 127 | 1.00 [1.00, 1.00] | 1.00 | 0.86 [0.76, 0.94] | 0.85 | 0.99 [0.98, 1.00] | 9.6e-06 (2.6e-04) |
| gru | E2 windowed (chosen weight) | support | 5 | 110 | 1.00 [1.00, 1.00] | 1.00 | 0.91 [0.84, 0.96] | 0.91 | 1.00 [1.00, 1.00] | 6.1e-06 (2.2e-04) |
| gru | E2 windowed w0.1 (pilot) | all | 1 | 25 | 1.00 [1.00, 1.00] | 1.00 | 0.68 [0.68, 0.68] | 0.68 | 1.00 [1.00, 1.00] | 5.5e-06 (2.0e-04) |
| gru | E2 windowed w0.1 (pilot) | support | 1 | 22 | 1.00 [1.00, 1.00] | 1.00 | 0.77 [0.77, 0.77] | 0.77 | 1.00 [1.00, 1.00] | 4.8e-06 (1.9e-04) |
| gru | E2 windowed w1 (pilot) | all | 1 | 22 | 0.73 [0.73, 0.73] | 1.00 | 0.82 [0.82, 0.82] | 0.73 | 0.82 [0.82, 0.82] | 1.4e-06 (2.3e-05) |
| gru | E2 windowed w1 (pilot) | support | 1 | 22 | 0.73 [0.73, 0.73] | 1.00 | 0.82 [0.82, 0.82] | 0.73 | 0.82 [0.82, 0.82] | 1.4e-06 (2.3e-05) |
| gru | E2 windowed w10 (pilot) | all | 1 | 26 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | 1.4e-07 (5.0e-07) |
| gru | E2 windowed w10 (pilot) | support | 1 | 22 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | 1.4e-07 (2.9e-07) |
| lstm | E1 (seed 0) | all | 5 | 121 | 0.87 [0.80, 0.94] | 0.85 | 0.83 [0.79, 0.88] | 0.87 | 0.96 [0.93, 1.00] | 1.9e-04 (2.1e-03) |
| lstm | E1 (seed 0) | support | 5 | 101 | 0.94 [0.88, 1.00] | 0.92 | 0.81 [0.75, 0.88] | 0.84 | 0.97 [0.92, 1.00] | 1.8e-04 (2.3e-03) |
| lstm | E2 windowed (chosen weight) | all | 5 | 130 | 1.00 [1.00, 1.00] | 1.00 | 0.45 [0.15, 0.78] | 0.35 | 0.89 [0.80, 0.98] | 1.5e-07 (3.9e-07) |
| lstm | E2 windowed (chosen weight) | support | 5 | 110 | 1.00 [1.00, 1.00] | 1.00 | 0.45 [0.14, 0.77] | 0.34 | 0.89 [0.78, 0.98] | 1.4e-07 (3.2e-07) |
| lstm | E2 windowed w0.1 (pilot) | all | 1 | 23 | 0.96 [0.96, 0.96] | 0.96 | 0.57 [0.57, 0.57] | 0.61 | 0.96 [0.96, 0.96] | 5.3e-05 (2.9e-03) |
| lstm | E2 windowed w0.1 (pilot) | support | 1 | 22 | 1.00 [1.00, 1.00] | 1.00 | 0.59 [0.59, 0.59] | 0.59 | 1.00 [1.00, 1.00] | 5.3e-05 (2.9e-03) |
| lstm | E2 windowed w1 (pilot) | all | 1 | 26 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | 2.6e-06 (7.1e-06) |
| lstm | E2 windowed w1 (pilot) | support | 1 | 22 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | 1.8e-06 (4.0e-06) |
| lstm | E2 windowed w10 (pilot) | all | 1 | 26 | 1.00 [1.00, 1.00] | 1.00 | 0.69 [0.69, 0.69] | 0.46 | 0.77 [0.77, 0.77] | 2.8e-07 (5.4e-07) |
| lstm | E2 windowed w10 (pilot) | support | 1 | 22 | 1.00 [1.00, 1.00] | 1.00 | 0.73 [0.73, 0.73] | 0.55 | 0.82 [0.82, 0.82] | 2.5e-07 (3.9e-07) |
| perl | E1 (seed 0) | all | 5 | 118 | 0.89 [0.82, 0.96] | 0.41 | 0.93 [0.88, 0.97] | 0.93 | 0.99 [0.98, 1.00] | 6.1e-05 (9.0e-04) |
| perl | E1 (seed 0) | support | 5 | 106 | 0.94 [0.90, 0.98] | 0.42 | 0.93 [0.87, 0.97] | 0.93 | 1.00 [1.00, 1.00] | 7.9e-05 (9.1e-04) |
| perl | E2 windowed (chosen weight) | all | 5 | 130 | 0.68 [0.28, 1.00] | 0.72 | 0.85 [0.64, 1.00] | 1.00 | 0.85 [0.64, 1.00] | 3.8e-07 (8.8e-06) |
| perl | E2 windowed (chosen weight) | support | 5 | 110 | 0.70 [0.30, 1.00] | 0.74 | 0.90 [0.72, 1.00] | 1.00 | 0.90 [0.72, 1.00] | 2.9e-07 (2.0e-06) |
| perl | E2 windowed w0.1 (pilot) | all | 1 | 26 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | 1.1e-07 (1.7e-07) |
| perl | E2 windowed w0.1 (pilot) | support | 1 | 22 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] | 1.0e-07 (1.6e-07) |
| perl | E2 windowed w1 (pilot) | all | 1 | 26 | 0.00 [0.00, 0.00] | 0.15 | 0.81 [0.81, 0.81] | 1.00 | 0.81 [0.81, 0.81] | n/a |
| perl | E2 windowed w1 (pilot) | support | 1 | 22 | 0.00 [0.00, 0.00] | 0.18 | 0.95 [0.95, 0.95] | 1.00 | 0.95 [0.95, 0.95] | n/a |
| perl | E2 windowed w10 (pilot) | all | 1 | 26 | 0.58 [0.58, 0.58] | 0.58 | 0.69 [0.69, 0.69] | 1.00 | 0.69 [0.69, 0.69] | 5.6e-08 (1.1e-05) |
| perl | E2 windowed w10 (pilot) | support | 1 | 22 | 0.68 [0.68, 0.68] | 0.68 | 0.73 [0.73, 0.73] | 1.00 | 0.73 [0.73, 0.73] | 5.6e-08 (1.1e-05) |

Low frequency:

| architecture | arm | part | sign(M) vs \|G_w(0.02)\| | sign(M_w) vs \|G_w(0.02)\| | sign(M) = sign(M_w) |
|---|---|---|---:|---:|---:|
| gru | E1 (seed 0) | all | 0.73 [0.63, 0.82] | 0.98 [0.97, 1.00] | 0.71 [0.60, 0.81] |
| gru | E1 (seed 0) | support | 0.76 [0.69, 0.85] | 0.98 [0.96, 1.00] | 0.75 [0.65, 0.84] |
| gru | E2 windowed (chosen weight) | all | 0.73 [0.64, 0.83] | 1.00 [1.00, 1.00] | 0.73 [0.64, 0.83] |
| gru | E2 windowed (chosen weight) | support | 0.72 [0.63, 0.81] | 1.00 [1.00, 1.00] | 0.72 [0.63, 0.81] |
| gru | E2 windowed w0.1 (pilot) | all | 0.60 [0.60, 0.60] | 1.00 [1.00, 1.00] | 0.60 [0.60, 0.60] |
| gru | E2 windowed w0.1 (pilot) | support | 0.68 [0.68, 0.68] | 1.00 [1.00, 1.00] | 0.68 [0.68, 0.68] |
| gru | E2 windowed w1 (pilot) | all | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| gru | E2 windowed w1 (pilot) | support | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| gru | E2 windowed w10 (pilot) | all | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| gru | E2 windowed w10 (pilot) | support | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| lstm | E1 (seed 0) | all | 0.75 [0.70, 0.79] | 1.00 [1.00, 1.00] | 0.75 [0.70, 0.79] |
| lstm | E1 (seed 0) | support | 0.73 [0.70, 0.76] | 1.00 [1.00, 1.00] | 0.73 [0.70, 0.76] |
| lstm | E2 windowed (chosen weight) | all | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| lstm | E2 windowed (chosen weight) | support | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| lstm | E2 windowed w0.1 (pilot) | all | 0.83 [0.83, 0.83] | 1.00 [1.00, 1.00] | 0.83 [0.83, 0.83] |
| lstm | E2 windowed w0.1 (pilot) | support | 0.82 [0.82, 0.82] | 1.00 [1.00, 1.00] | 0.82 [0.82, 0.82] |
| lstm | E2 windowed w1 (pilot) | all | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| lstm | E2 windowed w1 (pilot) | support | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| lstm | E2 windowed w10 (pilot) | all | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| lstm | E2 windowed w10 (pilot) | support | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| perl | E1 (seed 0) | all | 0.81 [0.75, 0.89] | 1.00 [1.00, 1.00] | 0.81 [0.75, 0.89] |
| perl | E1 (seed 0) | support | 0.85 [0.82, 0.90] | 1.00 [1.00, 1.00] | 0.85 [0.82, 0.90] |
| perl | E2 windowed (chosen weight) | all | 0.23 [0.03, 0.46] | 0.94 [0.82, 1.00] | 0.17 [0.01, 0.42] |
| perl | E2 windowed (chosen weight) | support | 0.24 [0.04, 0.51] | 0.95 [0.86, 1.00] | 0.19 [0.00, 0.50] |
| perl | E2 windowed w0.1 (pilot) | all | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| perl | E2 windowed w0.1 (pilot) | support | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| perl | E2 windowed w1 (pilot) | all | 0.65 [0.65, 0.65] | 1.00 [1.00, 1.00] | 0.65 [0.65, 0.65] |
| perl | E2 windowed w1 (pilot) | support | 0.77 [0.77, 0.77] | 1.00 [1.00, 1.00] | 0.77 [0.77, 0.77] |
| perl | E2 windowed w10 (pilot) | all | 0.38 [0.38, 0.38] | 1.00 [1.00, 1.00] | 0.38 [0.38, 0.38] |
| perl | E2 windowed w10 (pilot) | support | 0.32 [0.32, 0.32] | 1.00 [1.00, 1.00] | 0.32 [0.32, 0.32] |
