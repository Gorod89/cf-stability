# H12.3 over all laws, per corridor and pooled (D122)

- H12.3 over the laws (D122): Spearman correlation of the mean macro error of a law on a corridor (all components, or the dynamic ones) with the share of unstable equilibria among the equilibria of its members (table instability; the residual amplitudes of D111 from their member audits, the closed-form laws from their parameter sets).
- Laws: all laws with runs (D122: idm_global, idm_heterogeneous, knn, mlp, pidl, gru, lstm, perl, residual_idm, residual_idm_certified, mlp_penalty, gru_penalty, lstm_penalty, idm_heterogeneous_all, residual_idm_certified_het, residual_idm_certified_r0.1, residual_idm_certified_r0.2, residual_idm_certified_r0.5, residual_idm_free_r0.3) and, for comparison, the laws of the verdicts of H12.3 (D108: idm_global, idm_heterogeneous, knn, mlp, pidl, gru, lstm, perl, residual_idm, residual_idm_certified, mlp_penalty, gru_penalty, lstm_penalty, idm_heterogeneous_all, residual_idm_certified_het); a law without runs or without instability is left out (missing_corridor.txt).
- Per corridor over its laws (n laws); pooled with the corridor as a stratum: the ranks of the instability and of the error within every corridor, scaled to (0, 1) as (rank - 0.5) / n, then the Pearson correlation of the pooled ranks over the (law, corridor) pairs (n); bootstrap: the laws resampled within every corridor; p (permutation): two-sided, the instability permuted within every corridor, 10000 permutations, (1 + hits) / (1 + permutations).
- rule of H12.3 (for information, the verdicts stay those of D108): r > 0.6, p < 0.05 and n >= 10: confirmed; r < 0.3: refuted; otherwise open.
- Intervals: 95 % percentile bootstrap, 1000 resamples, seed 0.

| corridor | error | laws | n | Spearman | valid resamples | p (permutation) | rule of H12.3 |
|---|---|---|---:|---:|---:|---:|---|
| I-80 | macro error | all laws with runs (D122) | 19 | 0.734 [0.462, 0.869] | 1000 | <0.001 | confirmed |
| US-101 | macro error | all laws with runs (D122) | 18 | 0.777 [0.534, 0.902] | 1000 | <0.001 | confirmed |
| pooled (corridor as stratum) | macro error | all laws with runs (D122) | 37 | 0.755 [0.546, 0.850] | 1000 | <0.001 | confirmed |
| I-80 | macro error (dynamic) | all laws with runs (D122) | 19 | 0.651 [0.338, 0.820] | 1000 | 0.003 | confirmed |
| US-101 | macro error (dynamic) | all laws with runs (D122) | 18 | 0.788 [0.535, 0.925] | 1000 | <0.001 | confirmed |
| pooled (corridor as stratum) | macro error (dynamic) | all laws with runs (D122) | 37 | 0.718 [0.503, 0.828] | 1000 | <0.001 | confirmed |
| I-80 | macro error | laws of H12.3 (D108) | 15 | 0.652 [0.287, 0.834] | 1000 | 0.009 | confirmed |
| US-101 | macro error | laws of H12.3 (D108) | 14 | 0.744 [0.372, 0.912] | 1000 | 0.003 | confirmed |
| pooled (corridor as stratum) | macro error | laws of H12.3 (D108) | 29 | 0.697 [0.462, 0.819] | 1000 | <0.001 | confirmed |
| I-80 | macro error (dynamic) | laws of H12.3 (D108) | 15 | 0.496 [0.071, 0.762] | 1000 | 0.059 | open |
| US-101 | macro error (dynamic) | laws of H12.3 (D108) | 14 | 0.733 [0.366, 0.933] | 1000 | 0.004 | confirmed |
| pooled (corridor as stratum) | macro error (dynamic) | laws of H12.3 (D108) | 29 | 0.610 [0.338, 0.780] | 1000 | <0.001 | confirmed |

## The laws of the pooled correlation (D122)

- Per corridor and law the values that enter correlation_pooled: the share of unstable equilibria among the equilibria of its members and its mean macro errors over its runs (unit run).

| corridor | law | unstable among equilibria | macro error | macro error (dynamic) |
|---|---|---:|---:|---:|
| I-80 | idm_global | 0.000 | 0.182 | 0.345 |
| I-80 | idm_heterogeneous | 0.092 | 0.302 | 0.520 |
| I-80 | knn | 0.924 | 0.490 | 0.508 |
| I-80 | mlp | 0.957 | 0.308 | 0.485 |
| I-80 | pidl | 0.908 | 0.274 | 0.387 |
| I-80 | gru | 0.715 | 0.192 | 0.312 |
| I-80 | lstm | 0.733 | 0.716 | 0.657 |
| I-80 | perl | 0.940 | 0.431 | 0.652 |
| I-80 | residual_idm | 0.030 | 0.158 | 0.268 |
| I-80 | residual_idm_certified | 0.000 | 0.125 | 0.190 |
| I-80 | mlp_penalty | 0.000 | 0.273 | 0.423 |
| I-80 | gru_penalty | 0.359 | 0.235 | 0.363 |
| I-80 | lstm_penalty | 0.499 | 0.734 | 0.684 |
| I-80 | idm_heterogeneous_all | 0.165 | 0.336 | 0.573 |
| I-80 | residual_idm_certified_het | 0.000 | 0.200 | 0.368 |
| I-80 | residual_idm_certified_r0.1 | 0.000 | 0.197 | 0.301 |
| I-80 | residual_idm_certified_r0.2 | 0.000 | 0.181 | 0.277 |
| I-80 | residual_idm_certified_r0.5 | 0.000 | 0.108 | 0.164 |
| I-80 | residual_idm_free_r0.3 | 0.075 | 0.209 | 0.356 |
| US-101 | idm_global | 0.000 | 0.133 | 0.229 |
| US-101 | idm_heterogeneous | 0.092 | 0.125 | 0.196 |
| US-101 | knn | 0.924 | 0.628 | 0.583 |
| US-101 | mlp | 0.957 | 0.428 | 0.651 |
| US-101 | pidl | 0.908 | 0.412 | 0.526 |
| US-101 | gru | 0.715 | 0.421 | 0.517 |
| US-101 | lstm | 0.733 | 0.888 | 0.837 |
| US-101 | perl | 0.940 | 0.661 | 0.685 |
| US-101 | residual_idm | 0.030 | 0.156 | 0.256 |
| US-101 | residual_idm_certified | 0.000 | 0.113 | 0.122 |
| US-101 | mlp_penalty | 0.000 | 0.380 | 0.518 |
| US-101 | gru_penalty | 0.359 | 0.347 | 0.458 |
| US-101 | lstm_penalty | 0.499 | 0.886 | 0.786 |
| US-101 | residual_idm_certified_het | 0.000 | 0.174 | 0.266 |
| US-101 | residual_idm_certified_r0.1 | 0.000 | 0.177 | 0.228 |
| US-101 | residual_idm_certified_r0.2 | 0.000 | 0.147 | 0.184 |
| US-101 | residual_idm_certified_r0.5 | 0.000 | 0.104 | 0.136 |
| US-101 | residual_idm_free_r0.3 | 0.075 | 0.194 | 0.312 |
