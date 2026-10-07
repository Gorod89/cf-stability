# E2 sweep: choice of the penalty weight (D85)

- Runs: 120 of 120 expected runs exist; missing runs, files and values: missing.txt.
- E2 sweep: penalty weights 0.01, 0.1, 1, 10 on follownet_highd, 5 folds, seed 0; unit run, means over the folds (column runs; audited for the share).
- val / test RMSE s: mean over the runs of the mean spacing RMSE of the validation / test events (m). stable: band_numerical of the grid speeds in support. feasible best epoch: runs whose best epoch meets the penalty (D89).
- Chosen (D85): among the weights with stable >= 0.9 the smallest validation RMSE; without such a weight the largest stable share. Written to chosen_weights.json.
- Intervals: 95 % percentile bootstrap over the units, 1000 resamples, seed 0; complete: every run of the design is there and audited.

| architecture | penalty | weight | runs | val RMSE s (m) | test RMSE s (m) | audited | stable | feasible best epoch | chosen |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlp | jacobian | 0.01 | 5 | 2.84 | 2.84 | 5 | 0.12 | 0 |  |
| mlp | jacobian | 0.1 | 5 | 2.85 | 2.86 | 5 | 1.00 | 5 |  |
| mlp | jacobian | 1 | 5 | 2.80 | 2.80 | 5 | 1.00 | 5 | yes |
| mlp | jacobian | 10 | 5 | 2.94 | 2.94 | 5 | 1.00 | 5 |  |
| pidl | jacobian | 0.01 | 5 | 2.80 | 2.80 | 5 | 0.11 | 0 |  |
| pidl | jacobian | 0.1 | 5 | 2.79 | 2.81 | 5 | 0.48 | 1 |  |
| pidl | jacobian | 1 | 5 | 2.80 | 2.80 | 5 | 1.00 | 5 |  |
| pidl | jacobian | 10 | 5 | 2.78 | 2.78 | 5 | 1.00 | 5 | yes |
| gru | gain | 0.01 | 5 | 2.19 | 2.20 | 5 | 0.22 | 0 |  |
| gru | gain | 0.1 | 5 | 2.32 | 2.33 | 5 | 0.24 | 1 | yes |
| gru | gain | 1 | 5 | 2.91 | 2.91 | 5 | 0.00 | 5 |  |
| gru | gain | 10 | 5 | 2.69 | 2.69 | 5 | 0.04 | 5 |  |
| lstm | gain | 0.01 | 5 | 2.28 | 2.29 | 5 | 0.31 | 0 |  |
| lstm | gain | 0.1 | 5 | 2.73 | 2.74 | 5 | 0.55 | 0 | yes |
| lstm | gain | 1 | 5 | 2.69 | 2.70 | 5 | 0.10 | 4 |  |
| lstm | gain | 10 | 5 | 2.73 | 2.73 | 5 | 0.03 | 4 |  |
| perl | gain | 0.01 | 5 | 3.02 | 3.03 | 5 | 0.00 | 0 |  |
| perl | gain | 0.1 | 5 | 2.98 | 3.00 | 5 | 0.00 | 0 | yes |
| perl | gain | 1 | 5 | 3.06 | 3.03 | 5 | 0.00 | 0 |  |
| perl | gain | 10 | 5 | 3.30 | 3.30 | 5 | 0.00 | 0 |  |
| residual_idm | jacobian | 0.01 | 5 | 2.71 | 2.71 | 5 | 0.85 | 0 |  |
| residual_idm | jacobian | 0.1 | 5 | 2.74 | 2.74 | 5 | 0.99 | 0 | yes |
| residual_idm | jacobian | 1 | 5 | 3.39 | 3.39 | 5 | 0.87 | 0 |  |
| residual_idm | jacobian | 10 | 5 | 3.50 | 3.51 | 5 | 0.87 | 0 |  |

Chosen weights: mlp 1, pidl 10, gru 0.1, lstm 0.1, perl 0.1, residual_idm 0.1.
