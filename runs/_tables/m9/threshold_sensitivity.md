# Threshold sensitivity of the audit's numerical rule (M9)

- Runs: every E1 run on follownet_highd (e1: the learned models five folds x five seeds, the baselines five folds x seed 0); the stored audits (stability.json, threshold 1.02), no new rollouts. Per run and threshold the numerical flag of every equilibrium is recomputed from the stored largest gain over the 25 audit frequencies and the summary by the audit's own summarise_audit.
- Rule: unstable = (largest gain > threshold), a strict inequality as in the audit (FrequencyConfig.threshold): a largest gain exactly at the threshold counts as not unstable (stable). A speed whose largest gain is undefined (no equilibrium; a rollout that broke down, NaN gain) has no flag: it is left out of the H1.1 share and counted as undefined in the band shares. Largest gains of the stored audits exactly equal to one of the thresholds: 0.
- H1.1: string-unstable equilibria among the equilibria the audit analysed, inside the band or outside it, speeds in support (share_unstable_numerical, D92). Band shares: band_numerical of the grid speeds in support (unstable inside the band; not stable = 1 - stable; outside and none do not depend on the threshold). flagged: equilibria in support above the threshold, summed over the runs.
- Unit run; mean over the runs with the 95 % percentile bootstrap interval over the runs (1000 resamples, seed 0). Change vs 1.02: the H1.1 statistic minus that of the same run at 1.02, mean over the runs with its interval. H1.1 share rule: share >= 0.5 with the lower end > 0.3 (cf_stability/eval/tables.py; the verdict of H1.1 concerns mlp, gru and lstm and also needs the RMSE part).
- Check: at the stored threshold the recomputed H1.1 statistic and band shares equal the stored summaries for 175 of 175 runs (largest difference 0.0e+00).

| architecture | threshold | runs | H1.1: unstable among equilibria | change vs 1.02 | band unstable | band not stable | stable | outside | none | flagged | H1.1 share rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mlp | 1.00 | 25 | 0.99 [0.98, 1.00] | +0.05 [+0.03, +0.07] | 0.99 [0.98, 1.00] | 0.99 [0.98, 1.00] | 0.01 | 0.00 | 0.00 | 545 | met |
| mlp | 1.01 | 25 | 0.98 [0.96, 0.99] | +0.04 [+0.02, +0.06] | 0.98 [0.96, 0.99] | 0.98 [0.96, 0.99] | 0.02 | 0.00 | 0.00 | 538 | met |
| mlp | 1.02 | 25 | 0.94 [0.91, 0.97] |  | 0.94 [0.91, 0.96] | 0.94 [0.91, 0.97] | 0.06 | 0.00 | 0.00 | 518 | met |
| mlp | 1.05 | 25 | 0.35 [0.30, 0.40] | -0.59 [-0.64, -0.54] | 0.35 [0.29, 0.40] | 0.35 [0.30, 0.40] | 0.65 | 0.00 | 0.00 | 191 | not met |
| pidl | 1.00 | 25 | 0.98 [0.97, 0.99] | +0.14 [+0.10, +0.18] | 0.98 [0.97, 0.99] | 0.98 [0.97, 0.99] | 0.02 | 0.00 | 0.00 | 541 | met |
| pidl | 1.01 | 25 | 0.95 [0.93, 0.97] | +0.11 [+0.07, +0.14] | 0.95 [0.93, 0.97] | 0.95 [0.93, 0.97] | 0.05 | 0.00 | 0.00 | 523 | met |
| pidl | 1.02 | 25 | 0.84 [0.81, 0.88] |  | 0.84 [0.81, 0.88] | 0.84 [0.81, 0.88] | 0.16 | 0.00 | 0.00 | 464 | met |
| pidl | 1.05 | 25 | 0.52 [0.48, 0.57] | -0.32 [-0.36, -0.29] | 0.52 [0.48, 0.57] | 0.52 [0.48, 0.57] | 0.48 | 0.00 | 0.00 | 286 | met |
| residual_idm | 1.00 | 25 | 0.91 [0.89, 0.92] | +0.59 [+0.57, +0.62] | 0.91 [0.89, 0.92] | 0.91 [0.89, 0.92] | 0.09 | 0.00 | 0.00 | 498 | met |
| residual_idm | 1.01 | 25 | 0.64 [0.61, 0.66] | +0.33 [+0.31, +0.35] | 0.64 [0.61, 0.66] | 0.64 [0.61, 0.66] | 0.36 | 0.00 | 0.00 | 352 | met |
| residual_idm | 1.02 | 25 | 0.31 [0.28, 0.34] |  | 0.31 [0.28, 0.34] | 0.31 [0.28, 0.34] | 0.69 | 0.00 | 0.00 | 171 | not met |
| residual_idm | 1.05 | 25 | 0.00 [0.00, 0.00] | -0.31 [-0.34, -0.28] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.00 | 0.00 | 0.00 | 0 | not met |
| gru | 1.00 | 25 | 0.91 [0.88, 0.95] | +0.06 [+0.04, +0.08] | 0.58 [0.52, 0.65] | 0.94 [0.91, 0.97] | 0.06 | 0.31 | 0.05 | 477 | met |
| gru | 1.01 | 25 | 0.89 [0.84, 0.92] | +0.03 [+0.01, +0.04] | 0.56 [0.49, 0.62] | 0.92 [0.88, 0.95] | 0.08 | 0.31 | 0.05 | 462 | met |
| gru | 1.02 | 25 | 0.86 [0.80, 0.91] |  | 0.53 [0.47, 0.60] | 0.89 [0.84, 0.93] | 0.11 | 0.31 | 0.05 | 447 | met |
| gru | 1.05 | 25 | 0.81 [0.75, 0.88] | -0.04 [-0.07, -0.02] | 0.49 [0.43, 0.56] | 0.85 [0.79, 0.91] | 0.15 | 0.31 | 0.05 | 424 | met |
| lstm | 1.00 | 25 | 0.94 [0.91, 0.96] | +0.01 [+0.01, +0.02] | 0.44 [0.36, 0.52] | 0.96 [0.94, 0.98] | 0.04 | 0.39 | 0.12 | 450 | met |
| lstm | 1.01 | 25 | 0.92 [0.89, 0.96] | +0.00 [+0.00, +0.00] | 0.43 [0.35, 0.51] | 0.95 [0.93, 0.97] | 0.05 | 0.39 | 0.12 | 443 | met |
| lstm | 1.02 | 25 | 0.92 [0.89, 0.96] |  | 0.43 [0.35, 0.51] | 0.95 [0.93, 0.97] | 0.05 | 0.39 | 0.12 | 443 | met |
| lstm | 1.05 | 25 | 0.89 [0.85, 0.93] | -0.04 [-0.06, -0.02] | 0.40 [0.33, 0.48] | 0.92 [0.89, 0.95] | 0.08 | 0.39 | 0.12 | 424 | met |
| perl | 1.00 | 25 | 0.98 [0.96, 0.99] | +0.02 [+0.00, +0.05] | 0.67 [0.59, 0.75] | 0.98 [0.97, 0.99] | 0.02 | 0.30 | 0.02 | 527 | met |
| perl | 1.01 | 25 | 0.96 [0.94, 0.99] | +0.01 [+0.00, +0.03] | 0.66 [0.59, 0.74] | 0.97 [0.96, 0.99] | 0.03 | 0.30 | 0.02 | 521 | met |
| perl | 1.02 | 25 | 0.95 [0.92, 0.98] |  | 0.66 [0.59, 0.74] | 0.97 [0.96, 0.99] | 0.03 | 0.30 | 0.02 | 516 | met |
| perl | 1.05 | 25 | 0.94 [0.90, 0.97] | -0.02 [-0.03, -0.01] | 0.64 [0.57, 0.72] | 0.96 [0.94, 0.98] | 0.04 | 0.30 | 0.02 | 506 | met |
| idm | 1.00 | 5 | 0.65 [0.64, 0.66] | +0.49 [+0.47, +0.50] | 0.62 [0.60, 0.64] | 0.68 [0.68, 0.68] | 0.32 | 0.06 | 0.00 | 71 | met |
| idm | 1.01 | 5 | 0.33 [0.32, 0.35] | +0.17 [+0.15, +0.18] | 0.30 [0.28, 0.32] | 0.36 [0.36, 0.36] | 0.64 | 0.06 | 0.00 | 36 | not met |
| idm | 1.02 | 5 | 0.15 [0.14, 0.17] |  | 0.14 [0.10, 0.17] | 0.20 [0.18, 0.22] | 0.80 | 0.06 | 0.00 | 17 | not met |
| idm | 1.05 | 5 | 0.00 [0.00, 0.00] | -0.15 [-0.17, -0.14] | 0.00 [0.00, 0.00] | 0.06 [0.05, 0.08] | 0.94 | 0.06 | 0.00 | 0 | not met |
| knn | 1.00 | 5 | 0.95 [0.92, 0.97] | +0.05 [+0.02, +0.07] | 0.95 [0.92, 0.97] | 0.95 [0.92, 0.97] | 0.05 | 0.00 | 0.00 | 104 | met |
| knn | 1.01 | 5 | 0.95 [0.92, 0.97] | +0.05 [+0.02, +0.07] | 0.95 [0.92, 0.97] | 0.95 [0.92, 0.97] | 0.05 | 0.00 | 0.00 | 104 | met |
| knn | 1.02 | 5 | 0.90 [0.87, 0.93] |  | 0.90 [0.87, 0.93] | 0.90 [0.87, 0.93] | 0.10 | 0.00 | 0.00 | 99 | met |
| knn | 1.05 | 5 | 0.84 [0.77, 0.90] | -0.06 [-0.10, -0.03] | 0.84 [0.77, 0.90] | 0.84 [0.77, 0.90] | 0.16 | 0.00 | 0.00 | 92 | met |
| newell | 1.00 | 5 | 1.00 [1.00, 1.00] | +1.00 [+1.00, +1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 | 0.00 | 0.00 | 110 | met |
| newell | 1.01 | 5 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 | 0.00 | 0.00 | 0 | not met |
| newell | 1.02 | 5 | 0.00 [0.00, 0.00] |  | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 | 0.00 | 0.00 | 0 | not met |
| newell | 1.05 | 5 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 | 0.00 | 0.00 | 0 | not met |
| ovm | 1.00 | 5 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] | 0.86 [0.86, 0.86] | 1.00 [1.00, 1.00] | 0.00 | 0.00 | 0.14 | 95 | met |
| ovm | 1.01 | 5 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] | 0.86 [0.86, 0.86] | 1.00 [1.00, 1.00] | 0.00 | 0.00 | 0.14 | 95 | met |
| ovm | 1.02 | 5 | 1.00 [1.00, 1.00] |  | 0.86 [0.86, 0.86] | 1.00 [1.00, 1.00] | 0.00 | 0.00 | 0.14 | 95 | met |
| ovm | 1.05 | 5 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] | 0.86 [0.86, 0.86] | 1.00 [1.00, 1.00] | 0.00 | 0.00 | 0.14 | 95 | met |
| persistence | 1.00 | 5 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 | 0.00 | 0.00 | 0 | not met |
| persistence | 1.01 | 5 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 | 0.00 | 0.00 | 0 | not met |
| persistence | 1.02 | 5 | 0.00 [0.00, 0.00] |  | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 | 0.00 | 0.00 | 0 | not met |
| persistence | 1.05 | 5 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 | 0.00 | 0.00 | 0 | not met |
