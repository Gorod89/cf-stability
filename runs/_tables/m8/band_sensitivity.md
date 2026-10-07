# Band sensitivity of the audit (D119)

- Runs: e1 on follownet_highd, learned models, five folds x five seeds; one audit per band (1/99: stability_q01_99.json, 5/95: stability.json, the band of D78, 10/90: stability_q10_90.json).
- Band: quantiles of the spacing of the near-steady training samples (|dv| < 0.5 m/s, |a| < 0.3 m/s^2, speed within 0.5 m/s) per grid speed, computed with cf_stability/train/tensors.py; the equilibria are searched inside the band first (D79), so another band can move the equilibrium itself, not only its status.
- stable, unstable, outside, none: band_numerical of the grid speeds in support (numerical rule); H1.1 = unstable among the equilibria (share_unstable_numerical, D92): the equilibrium inside the band where there is one, the first one of [1, 200] m otherwise.
- Unit run; mean over the runs with the 95 % percentile bootstrap interval over the runs (1000 resamples, seed 0). Change vs 5/95: the H1.1 statistic minus that of the same run with the band of D78, mean over the runs with its interval.
- H1.1 share rule: the band-dependent part of the verdict of H1.1, share >= 0.5 with the lower end > 0.3 (cf_stability/eval/tables.py; the RMSE part does not depend on the band).

| architecture | band | runs | stable | unstable | outside | none | H1.1: unstable among equilibria | change vs 5/95 | H1.1 share rule |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp | 1/99 | 25 | 0.06 [0.03, 0.09] | 0.94 [0.91, 0.96] | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.94 [0.91, 0.97] | +0.00 [+0.00, +0.00] | met |
| mlp | 5/95 | 25 | 0.06 [0.03, 0.09] | 0.94 [0.91, 0.96] | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.94 [0.91, 0.97] |  | met |
| mlp | 10/90 | 25 | 0.06 [0.03, 0.09] | 0.92 [0.89, 0.95] | 0.02 [0.01, 0.04] | 0.00 [0.00, 0.00] | 0.94 [0.91, 0.97] | +0.00 [+0.00, +0.00] | met |
| pidl | 1/99 | 25 | 0.16 [0.12, 0.19] | 0.84 [0.81, 0.88] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.84 [0.81, 0.88] | +0.00 [+0.00, +0.00] | met |
| pidl | 5/95 | 25 | 0.16 [0.12, 0.19] | 0.84 [0.81, 0.88] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.84 [0.81, 0.88] |  | met |
| pidl | 10/90 | 25 | 0.15 [0.12, 0.19] | 0.80 [0.76, 0.85] | 0.04 [0.02, 0.07] | 0.00 [0.00, 0.00] | 0.84 [0.81, 0.88] | +0.00 [+0.00, +0.00] | met |
| residual_idm | 1/99 | 25 | 0.69 [0.66, 0.72] | 0.31 [0.28, 0.34] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.31 [0.28, 0.34] | +0.00 [+0.00, +0.00] | not met |
| residual_idm | 5/95 | 25 | 0.69 [0.66, 0.72] | 0.31 [0.28, 0.34] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.31 [0.28, 0.34] |  | not met |
| residual_idm | 10/90 | 25 | 0.58 [0.53, 0.63] | 0.28 [0.24, 0.31] | 0.14 [0.09, 0.19] | 0.00 [0.00, 0.00] | 0.31 [0.28, 0.34] | +0.00 [+0.00, +0.00] | not met |
| gru | 1/99 | 25 | 0.14 [0.09, 0.19] | 0.68 [0.63, 0.74] | 0.13 [0.09, 0.18] | 0.05 [0.02, 0.08] | 0.85 [0.80, 0.90] | -0.00 [-0.01, +0.00] | met |
| gru | 5/95 | 25 | 0.11 [0.07, 0.16] | 0.53 [0.47, 0.60] | 0.31 [0.24, 0.38] | 0.05 [0.02, 0.08] | 0.86 [0.80, 0.91] |  | met |
| gru | 10/90 | 25 | 0.08 [0.05, 0.12] | 0.42 [0.35, 0.49] | 0.45 [0.37, 0.52] | 0.05 [0.02, 0.08] | 0.86 [0.80, 0.91] | +0.00 [+0.00, +0.00] | met |
| lstm | 1/99 | 25 | 0.06 [0.03, 0.09] | 0.55 [0.47, 0.63] | 0.27 [0.21, 0.32] | 0.12 [0.06, 0.19] | 0.92 [0.89, 0.96] | +0.00 [+0.00, +0.00] | met |
| lstm | 5/95 | 25 | 0.05 [0.03, 0.07] | 0.43 [0.35, 0.51] | 0.39 [0.33, 0.46] | 0.12 [0.06, 0.19] | 0.92 [0.89, 0.96] |  | met |
| lstm | 10/90 | 25 | 0.05 [0.03, 0.07] | 0.37 [0.30, 0.45] | 0.45 [0.39, 0.52] | 0.12 [0.06, 0.19] | 0.92 [0.89, 0.96] | +0.00 [+0.00, +0.00] | met |
| perl | 1/99 | 25 | 0.03 [0.01, 0.05] | 0.81 [0.76, 0.87] | 0.14 [0.09, 0.19] | 0.02 [0.01, 0.03] | 0.96 [0.92, 0.99] | +0.00 [+0.00, +0.01] | met |
| perl | 5/95 | 25 | 0.03 [0.01, 0.04] | 0.66 [0.59, 0.74] | 0.30 [0.22, 0.37] | 0.02 [0.01, 0.03] | 0.95 [0.92, 0.98] |  | met |
| perl | 10/90 | 25 | 0.02 [0.01, 0.04] | 0.52 [0.45, 0.60] | 0.44 [0.35, 0.52] | 0.02 [0.01, 0.03] | 0.95 [0.92, 0.98] | +0.00 [-0.01, +0.01] | met |
