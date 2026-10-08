# Full-history linear analysis of the recurrent laws (M9)

- Runs: GRU, LSTM and PERL; arms E1: gru e1, lstm e1, perl e1; E2: gru e2_gain_w0.1, lstm e2_gain_w0.1, perl e2_gain_w0.1; E2 combined: gru e2_combined_j0.1, lstm e2_combined_j0.1, perl e2_combined_j1; E2 long window: gru e2_gain_long_w0.1, lstm e2_gain_long_w0.1; E5 ACC: gru e5, lstm e5 on openacc_acc; E5 ACC penalised: gru e5_gain_w0.1, lstm e5_gain_w0.1 on openacc_acc; E5 human: gru e5, lstm e5 on openacc_human; E5 human penalised: gru e5_gain_w0.1, lstm e5_gain_w0.1 on openacc_human; on follownet_highd where no view is named (E1 and E2 five folds x five seeds, the combined arm and E5 five folds x seed 0, the long window fold 0, seed 0). E2 long window: the gain penalty of E2 (weight 0.1) measured over the last 252 s of a 380-s rollout, the pilot of the revision; E5: the control on the OpenACC views (D87) without penalty and with the chosen gain weight 0.1. Speeds: every grid speed with an equilibrium in the stored audit (status ok, multiple or outside: inside or outside the band), at the stored equilibrium; part all = every such speed, support = those in the speed range of the training data.
- Linearisation: the history Jacobian J [30, 3] of the acceleration with respect to the 30 states of the window (history_jacobian). The networks are re-run over the last 30 states at every step, without a hidden state carried between steps, so J is the exact linearisation of the rollout. State-space form of the two-vehicle loop with the leader speed as input and the follower speed as output, semi-implicit Euler at 0.1 s as in the rollout (windowed_state_space, cf_stability/stability/analytic.py): state (e[k], u[k-29..k], w[k-29..k]) of the gap, follower and leader speed deviations.
- poles inside: share of the speeds whose 31 closed-loop poles (the eigenvalues of the block of e and u; the other 30 are 0) all lie strictly inside the unit circle: local stability of the full history model. memoryless locally stable (reference): f_s > 0 and f_v + f_dv < 0 of the memoryless view (derivatives summed over the window).
- windowed > 1.02: share of the speeds whose windowed gain |G(e^(iω dt))| (transfer_windowed) exceeds 1.02 at some of the 25 audit frequencies (0.02-2 rad/s; strict inequality, the rule of the audit). numerical unstable (reference): the audit's verdict from the rollouts. agreement: share of the speeds where the two verdicts are equal (a speed whose rollout broke down has no numerical verdict and is left out). \|num - win\| at 0.02: median (90 % quantile) over the speeds with all poles inside and an unflagged rollout at 0.02 rad/s (pooled over the runs) of the difference between the audit's numerical gain and the windowed gain at that frequency.
- Low frequency: sign(M) vs \|G_w(0.02)\| is the share of the speeds where M < 0 (M = f_v² + 2 f_v f_dv - 2 f_s of the memoryless view) exactly when the windowed gain at the lowest audit frequency 0.02 rad/s is above 1; sign(M_w) the same for M_w = M + 2 (f_v m_s - f_s m_v) - dt f_s f_v, the exact coefficient of the low-frequency expansion of the windowed gain, |G|² = 1 - ω² M_w / f_s² + O(ω⁴) (windowed_margin; m_x = sum over the lags of lag x J_x, the first moments of the history Jacobian; the last term is the Euler step); sign(M) = sign(M_w): the memoryless criterion gives the ω -> 0 limit of the windowed gain. M is the coefficient only when f_v m_s = f_s m_v (no memory, or a lag common to spacing and speed). At 0.02 rad/s the higher-order terms can decide the side of 1 when |M_w| is small: PERL (Newell's feed-forward of the leader acceleration has m_dv = -m_v = 1, so M_w ~ M + 2 f_s) under the combined penalty.
- Unit run: the shares per run over its speeds, mean over the runs with the 95 % percentile bootstrap interval over the runs (1000 resamples, seed 0); a run without speeds in the part is left out of it.
- Checks: the recomputed maximum windowed gain equals the stored analytic_max_gain.windowed of the audits to 9.7e-12 (largest absolute difference); the agreement column equals the audit's agreement_gain to 0.0e+00; at 0.0001 rad/s the windowed gain is above 1 exactly when M_w < 0 at a share 1.000 or more of the speeds (mean over the runs, smallest row). Per-run details (poles, gains, Jacobian): full_history.json next to stability.json.

Poles and gain verdicts:

| architecture | arm | part | runs | speeds | poles inside | memoryless locally stable | windowed > 1.02 | numerical unstable | agreement windowed vs numerical | \|num - win\| at 0.02 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gru | E1 | all | 25 | 620 | 0.96 [0.93, 0.98] | 0.93 | 0.85 [0.80, 0.91] | 0.86 | 0.99 [0.98, 1.00] | 8.2e-05 (8.6e-04) |
| gru | E1 | support | 25 | 523 | 0.99 [0.98, 1.00] | 0.95 | 0.86 [0.81, 0.91] | 0.86 | 1.00 [0.99, 1.00] | 7.8e-05 (8.8e-04) |
| gru | E2 | all | 25 | 641 | 0.74 [0.64, 0.83] | 0.81 | 0.88 [0.82, 0.93] | 0.64 | 0.74 [0.66, 0.82] | 1.3e-05 (3.4e-04) |
| gru | E2 | support | 25 | 546 | 0.75 [0.64, 0.85] | 0.81 | 0.90 [0.84, 0.95] | 0.63 | 0.71 [0.62, 0.79] | 1.0e-05 (3.4e-04) |
| gru | E2 combined | all | 5 | 124 | 1.00 [1.00, 1.00] | 1.00 | 0.66 [0.48, 0.78] | 0.36 | 0.68 [0.47, 0.90] | 2.8e-05 (2.8e-04) |
| gru | E2 combined | support | 5 | 108 | 1.00 [1.00, 1.00] | 1.00 | 0.71 [0.53, 0.86] | 0.34 | 0.64 [0.38, 0.88] | 1.9e-05 (2.2e-04) |
| gru | E2 long window | all | 1 | 24 | 1.00 [1.00, 1.00] | 1.00 | 0.04 [0.04, 0.04] | 0.04 | 1.00 [1.00, 1.00] | 7.1e-06 (5.4e-05) |
| gru | E2 long window | support | 1 | 22 | 1.00 [1.00, 1.00] | 1.00 | 0.05 [0.05, 0.05] | 0.05 | 1.00 [1.00, 1.00] | 6.2e-06 (3.3e-05) |
| gru | E5 ACC | all | 5 | 127 | 1.00 [1.00, 1.00] | 1.00 | 0.96 [0.92, 1.00] | 0.96 | 1.00 [1.00, 1.00] | 5.0e-06 (1.1e-04) |
| gru | E5 ACC | support | 5 | 127 | 1.00 [1.00, 1.00] | 1.00 | 0.96 [0.92, 1.00] | 0.96 | 1.00 [1.00, 1.00] | 5.0e-06 (1.1e-04) |
| gru | E5 ACC penalised | all | 5 | 130 | 1.00 [1.00, 1.00] | 1.00 | 0.88 [0.65, 1.00] | 0.57 | 0.68 [0.33, 1.00] | 3.4e-06 (5.1e-05) |
| gru | E5 ACC penalised | support | 5 | 130 | 1.00 [1.00, 1.00] | 1.00 | 0.88 [0.65, 1.00] | 0.57 | 0.68 [0.33, 1.00] | 3.4e-06 (5.1e-05) |
| gru | E5 human | all | 5 | 130 | 0.97 [0.91, 1.00] | 1.00 | 0.92 [0.83, 1.00] | 0.92 | 1.00 [1.00, 1.00] | 4.1e-06 (1.1e-04) |
| gru | E5 human | support | 5 | 130 | 0.97 [0.91, 1.00] | 1.00 | 0.92 [0.83, 1.00] | 0.92 | 1.00 [1.00, 1.00] | 4.1e-06 (1.1e-04) |
| gru | E5 human penalised | all | 5 | 125 | 1.00 [1.00, 1.00] | 1.00 | 0.73 [0.50, 0.93] | 0.67 | 0.94 [0.81, 1.00] | 7.0e-06 (2.4e-05) |
| gru | E5 human penalised | support | 5 | 125 | 1.00 [1.00, 1.00] | 1.00 | 0.73 [0.50, 0.93] | 0.67 | 0.94 [0.81, 1.00] | 7.0e-06 (2.4e-05) |
| lstm | E1 | all | 25 | 580 | 0.90 [0.86, 0.94] | 0.89 | 0.88 [0.84, 0.91] | 0.91 | 0.96 [0.93, 0.98] | 1.4e-04 (2.4e-03) |
| lstm | E1 | support | 25 | 482 | 0.93 [0.89, 0.96] | 0.93 | 0.89 [0.84, 0.93] | 0.92 | 0.96 [0.93, 0.99] | 1.4e-04 (2.1e-03) |
| lstm | E2 | all | 25 | 644 | 0.98 [0.96, 0.99] | 0.99 | 0.63 [0.51, 0.73] | 0.37 | 0.72 [0.61, 0.83] | 4.3e-06 (1.1e-04) |
| lstm | E2 | support | 25 | 548 | 0.99 [0.96, 1.00] | 1.00 | 0.61 [0.49, 0.73] | 0.33 | 0.70 [0.57, 0.82] | 2.6e-06 (8.2e-05) |
| lstm | E2 combined | all | 5 | 129 | 0.98 [0.93, 1.00] | 1.00 | 0.51 [0.31, 0.72] | 0.47 | 0.95 [0.87, 1.00] | 5.7e-06 (1.5e-04) |
| lstm | E2 combined | support | 5 | 109 | 0.97 [0.91, 1.00] | 1.00 | 0.47 [0.23, 0.71] | 0.42 | 0.94 [0.84, 1.00] | 2.9e-06 (6.3e-05) |
| lstm | E2 long window | all | 1 | 26 | 1.00 [1.00, 1.00] | 1.00 | 0.15 [0.15, 0.15] | 0.15 | 1.00 [1.00, 1.00] | 9.3e-06 (7.5e-04) |
| lstm | E2 long window | support | 1 | 22 | 1.00 [1.00, 1.00] | 1.00 | 0.18 [0.18, 0.18] | 0.18 | 1.00 [1.00, 1.00] | 9.7e-06 (9.8e-04) |
| lstm | E5 ACC | all | 5 | 121 | 0.93 [0.78, 1.00] | 0.93 | 0.93 [0.86, 0.99] | 0.96 | 0.97 [0.93, 1.00] | 9.1e-06 (1.1e-04) |
| lstm | E5 ACC | support | 5 | 121 | 0.93 [0.78, 1.00] | 0.93 | 0.93 [0.86, 0.99] | 0.96 | 0.97 [0.93, 1.00] | 9.1e-06 (1.1e-04) |
| lstm | E5 ACC penalised | all | 5 | 130 | 0.98 [0.93, 1.00] | 1.00 | 0.65 [0.38, 0.91] | 0.30 | 0.63 [0.25, 0.97] | 9.3e-06 (1.8e-04) |
| lstm | E5 ACC penalised | support | 5 | 130 | 0.98 [0.93, 1.00] | 1.00 | 0.65 [0.38, 0.91] | 0.30 | 0.63 [0.25, 0.97] | 9.3e-06 (1.8e-04) |
| lstm | E5 human | all | 5 | 107 | 0.94 [0.87, 1.00] | 0.95 | 0.78 [0.67, 0.90] | 0.79 | 0.99 [0.97, 1.00] | 8.3e-06 (1.1e-04) |
| lstm | E5 human | support | 5 | 107 | 0.94 [0.87, 1.00] | 0.95 | 0.78 [0.67, 0.90] | 0.79 | 0.99 [0.97, 1.00] | 8.3e-06 (1.1e-04) |
| lstm | E5 human penalised | all | 5 | 122 | 1.00 [1.00, 1.00] | 1.00 | 0.71 [0.33, 0.98] | 0.71 | 1.00 [1.00, 1.00] | 2.6e-05 (6.2e-05) |
| lstm | E5 human penalised | support | 5 | 122 | 1.00 [1.00, 1.00] | 1.00 | 0.71 [0.33, 0.98] | 0.71 | 1.00 [1.00, 1.00] | 2.6e-05 (6.2e-05) |
| perl | E1 | all | 25 | 620 | 0.86 [0.83, 0.90] | 0.64 | 0.93 [0.90, 0.96] | 0.96 | 0.97 [0.95, 0.99] | 4.1e-05 (1.2e-03) |
| perl | E1 | support | 25 | 540 | 0.92 [0.89, 0.96] | 0.65 | 0.94 [0.90, 0.97] | 0.95 | 0.98 [0.96, 1.00] | 3.9e-05 (1.0e-03) |
| perl | E2 | all | 25 | 612 | 0.12 [0.03, 0.22] | 0.14 | 0.59 [0.43, 0.76] | 0.98 | 0.60 [0.43, 0.77] | 1.2e-04 (2.4e-03) |
| perl | E2 | support | 25 | 522 | 0.12 [0.04, 0.23] | 0.14 | 0.58 [0.42, 0.76] | 0.98 | 0.59 [0.43, 0.77] | 8.7e-05 (2.1e-03) |
| perl | E2 combined | all | 5 | 43 | 1.00 [1.00, 1.00] | 1.00 | 0.91 [0.82, 1.00] | 0.91 | 1.00 [1.00, 1.00] | 1.1e-03 (1.8e-02) |
| perl | E2 combined | support | 5 | 39 | 1.00 [1.00, 1.00] | 1.00 | 0.91 [0.82, 1.00] | 0.91 | 1.00 [1.00, 1.00] | 1.2e-03 (1.9e-02) |

Low frequency:

| architecture | arm | part | sign(M) vs \|G_w(0.02)\| | sign(M_w) vs \|G_w(0.02)\| | sign(M) = sign(M_w) |
|---|---|---|---:|---:|---:|
| gru | E1 | all | 0.67 [0.63, 0.72] | 0.99 [0.99, 1.00] | 0.67 [0.63, 0.72] |
| gru | E1 | support | 0.69 [0.65, 0.74] | 0.99 [0.98, 1.00] | 0.69 [0.65, 0.74] |
| gru | E2 | all | 0.87 [0.81, 0.92] | 1.00 [0.99, 1.00] | 0.87 [0.81, 0.92] |
| gru | E2 | support | 0.85 [0.79, 0.91] | 1.00 [0.99, 1.00] | 0.85 [0.79, 0.92] |
| gru | E2 combined | all | 0.84 [0.75, 0.91] | 1.00 [1.00, 1.00] | 0.84 [0.75, 0.91] |
| gru | E2 combined | support | 0.83 [0.76, 0.90] | 1.00 [1.00, 1.00] | 0.83 [0.76, 0.90] |
| gru | E2 long window | all | 0.17 [0.17, 0.17] | 1.00 [1.00, 1.00] | 0.17 [0.17, 0.17] |
| gru | E2 long window | support | 0.09 [0.09, 0.09] | 1.00 [1.00, 1.00] | 0.09 [0.09, 0.09] |
| gru | E5 ACC | all | 0.69 [0.50, 0.87] | 0.97 [0.94, 1.00] | 0.66 [0.44, 0.85] |
| gru | E5 ACC | support | 0.69 [0.50, 0.87] | 0.97 [0.94, 1.00] | 0.66 [0.44, 0.85] |
| gru | E5 ACC penalised | all | 0.96 [0.92, 1.00] | 1.00 [1.00, 1.00] | 0.96 [0.92, 1.00] |
| gru | E5 ACC penalised | support | 0.96 [0.92, 1.00] | 1.00 [1.00, 1.00] | 0.96 [0.92, 1.00] |
| gru | E5 human | all | 0.89 [0.81, 0.98] | 0.99 [0.98, 1.00] | 0.90 [0.82, 0.98] |
| gru | E5 human | support | 0.89 [0.81, 0.98] | 0.99 [0.98, 1.00] | 0.90 [0.82, 0.98] |
| gru | E5 human penalised | all | 0.99 [0.98, 1.00] | 1.00 [1.00, 1.00] | 0.99 [0.98, 1.00] |
| gru | E5 human penalised | support | 0.99 [0.98, 1.00] | 1.00 [1.00, 1.00] | 0.99 [0.98, 1.00] |
| lstm | E1 | all | 0.71 [0.67, 0.75] | 1.00 [0.99, 1.00] | 0.71 [0.68, 0.75] |
| lstm | E1 | support | 0.69 [0.65, 0.74] | 1.00 [0.99, 1.00] | 0.70 [0.65, 0.74] |
| lstm | E2 | all | 0.97 [0.91, 1.00] | 1.00 [0.99, 1.00] | 0.97 [0.91, 1.00] |
| lstm | E2 | support | 0.97 [0.90, 1.00] | 1.00 [0.99, 1.00] | 0.97 [0.90, 1.00] |
| lstm | E2 combined | all | 0.97 [0.92, 1.00] | 1.00 [1.00, 1.00] | 0.97 [0.92, 1.00] |
| lstm | E2 combined | support | 0.96 [0.91, 1.00] | 1.00 [1.00, 1.00] | 0.96 [0.91, 1.00] |
| lstm | E2 long window | all | 0.42 [0.42, 0.42] | 1.00 [1.00, 1.00] | 0.42 [0.42, 0.42] |
| lstm | E2 long window | support | 0.32 [0.32, 0.32] | 1.00 [1.00, 1.00] | 0.32 [0.32, 0.32] |
| lstm | E5 ACC | all | 0.86 [0.72, 0.98] | 1.00 [1.00, 1.00] | 0.86 [0.72, 0.98] |
| lstm | E5 ACC | support | 0.86 [0.72, 0.98] | 1.00 [1.00, 1.00] | 0.86 [0.72, 0.98] |
| lstm | E5 ACC penalised | all | 0.99 [0.98, 1.00] | 1.00 [1.00, 1.00] | 0.99 [0.98, 1.00] |
| lstm | E5 ACC penalised | support | 0.99 [0.98, 1.00] | 1.00 [1.00, 1.00] | 0.99 [0.98, 1.00] |
| lstm | E5 human | all | 0.92 [0.81, 0.99] | 1.00 [1.00, 1.00] | 0.92 [0.81, 0.99] |
| lstm | E5 human | support | 0.92 [0.81, 0.99] | 1.00 [1.00, 1.00] | 0.92 [0.81, 0.99] |
| lstm | E5 human penalised | all | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| lstm | E5 human penalised | support | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| perl | E1 | all | 0.72 [0.66, 0.77] | 0.99 [0.99, 1.00] | 0.72 [0.66, 0.77] |
| perl | E1 | support | 0.74 [0.68, 0.80] | 0.99 [0.99, 1.00] | 0.74 [0.68, 0.80] |
| perl | E2 | all | 0.41 [0.28, 0.55] | 0.82 [0.71, 0.91] | 0.54 [0.40, 0.68] |
| perl | E2 | support | 0.41 [0.28, 0.56] | 0.81 [0.71, 0.91] | 0.54 [0.40, 0.68] |
| perl | E2 combined | all | 1.00 [1.00, 1.00] | 0.29 [0.00, 0.58] | 0.29 [0.00, 0.58] |
| perl | E2 combined | support | 1.00 [1.00, 1.00] | 0.34 [0.00, 0.68] | 0.34 [0.00, 0.68] |
