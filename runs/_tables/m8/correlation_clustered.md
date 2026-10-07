# H12.3 with clustered inference: laws and architecture families as units (review of M8)

- Clustered inference of H12.3 (review of M8): the rows of correlation_pooled, one per law and corridor (the share of unstable equilibria among the equilibria of the law's members against its mean macro error over its runs; correlation_pooled_laws), are not independent: a law has the same instability on both corridors, and the variants of one architecture (penalised, certified, residual amplitudes) are related. Spearman: as correlation_pooled (ranks within every corridor scaled to (0, 1), Pearson of the pooled ranks; per corridor: the Spearman correlation over its laws).
- law clusters: bootstrap of the laws as clusters (the rows of a law on both corridors move together); family clusters: bootstrap of the architecture families as clusters (families: IDM: idm_global, idm_heterogeneous, idm_heterogeneous_all; ResidualIDM: residual_idm, residual_idm_certified, residual_idm_certified_r0.1, residual_idm_certified_r0.2, residual_idm_certified_r0.5, residual_idm_free_r0.3, residual_idm_certified_het; k-NN: knn; MLP: mlp, mlp_penalty; PIDL: pidl; GRU: gru, gru_penalty; LSTM: lstm, lstm_penalty; PERL: perl). A cluster drawn k times enters with all its rows k times; ranks within the corridors of every resample; resamples with a constant variable have no correlation. pairs independent (D122): the rows resampled within every corridor as in correlation_pooled, with the seed and the resamples of this table, for comparison. Per corridor the law clusters are the laws themselves.
- Percentile intervals (95 %), 5000 resamples each; one generator numpy default_rng(20261007) per row draws the laws (resamples x laws, the clusters numbered in the order of their first appearance: corridor, then the law order of the design), then from the same stream the families, then the rows within the corridors.
- p (laws permuted): two-sided permutation test of 10000 permutations (own generator default_rng(20261007)) of the instability values over the laws as wholes: a law takes another law's value on every corridor it runs on (linked across the corridors), the ranks are recomputed; (1 + hits) / (1 + permutations). Per corridor it is the permutation of the instability over its laws.
- Subsets: all laws with runs (those of correlation_pooled); without <family>: the rows of one family left out, ranks recomputed (leave-one-family-out); learned laws only: without the families IDM, ResidualIDM. n: rows (law-corridor pairs).
- rule of H12.3 (for information; the verdicts of D108 stay as reported): r > 0.6, p (laws permuted) < 0.05 and n >= 10: confirmed; r < 0.3: refuted; otherwise open.

| corridor | error | laws | n | laws (clusters) | families (clusters) | Spearman | law clusters | family clusters | pairs independent (D122) | p (laws permuted) | rule of H12.3 |
|---|---|---|---:|---:|---:|---:|---|---|---|---:|---|
| I-80 | macro error | all laws with runs | 19 | 19 | 8 | 0.734 | [0.451, 0.868] | [0.029, 0.850] | [0.457, 0.870] | <0.001 | confirmed |
| I-80 | macro error | without IDM | 16 | 16 | 7 | 0.713 | [0.375, 0.875] | [-0.392, 0.842] | [0.380, 0.874] | 0.003 | confirmed |
| I-80 | macro error | without k-NN | 18 | 18 | 7 | 0.716 | [0.414, 0.869] | [0.060, 0.852] | [0.416, 0.871] | 0.001 | confirmed |
| I-80 | macro error | without MLP | 17 | 17 | 7 | 0.770 | [0.479, 0.908] | [-0.148, 0.883] | [0.475, 0.910] | <0.001 | confirmed |
| I-80 | macro error | without PIDL | 18 | 18 | 7 | 0.733 | [0.427, 0.884] | [0.046, 0.876] | [0.433, 0.886] | <0.001 | confirmed |
| I-80 | macro error | without GRU | 17 | 17 | 7 | 0.795 | [0.553, 0.901] | [-0.321, 0.883] | [0.537, 0.899] | <0.001 | confirmed |
| I-80 | macro error | without LSTM | 17 | 17 | 7 | 0.737 | [0.396, 0.894] | [0.301, 0.876] | [0.394, 0.893] | 0.001 | confirmed |
| I-80 | macro error | without PERL | 18 | 18 | 7 | 0.716 | [0.413, 0.873] | [0.056, 0.850] | [0.408, 0.868] | 0.002 | confirmed |
| I-80 | macro error | without ResidualIDM | 12 | 12 | 7 | 0.473 | [-0.090, 0.796] | [-0.342, 0.852] | [-0.088, 0.794] | 0.121 | open |
| I-80 | macro error | learned laws only (without IDM, ResidualIDM) | 9 | 9 | 6 | 0.317 | [-0.590, 0.795] | [-0.852, 0.875] | [-0.615, 0.810] | 0.407 | open |
| US-101 | macro error | all laws with runs | 18 | 18 | 8 | 0.777 | [0.509, 0.896] | [0.155, 0.901] | [0.510, 0.905] | <0.001 | confirmed |
| US-101 | macro error | without IDM | 16 | 16 | 7 | 0.809 | [0.555, 0.928] | [-0.185, 0.945] | [0.564, 0.931] | <0.001 | confirmed |
| US-101 | macro error | without k-NN | 17 | 17 | 7 | 0.764 | [0.431, 0.904] | [0.226, 0.888] | [0.438, 0.902] | <0.001 | confirmed |
| US-101 | macro error | without MLP | 16 | 16 | 7 | 0.814 | [0.533, 0.936] | [-0.012, 0.960] | [0.539, 0.939] | <0.001 | confirmed |
| US-101 | macro error | without PIDL | 17 | 17 | 7 | 0.764 | [0.436, 0.907] | [0.253, 0.901] | [0.438, 0.904] | <0.001 | confirmed |
| US-101 | macro error | without GRU | 16 | 16 | 7 | 0.722 | [0.370, 0.867] | [-0.161, 0.871] | [0.381, 0.865] | 0.002 | confirmed |
| US-101 | macro error | without LSTM | 16 | 16 | 7 | 0.771 | [0.352, 0.942] | [0.428, 0.960] | [0.356, 0.943] | <0.001 | confirmed |
| US-101 | macro error | without PERL | 17 | 17 | 7 | 0.764 | [0.433, 0.903] | [0.228, 0.888] | [0.446, 0.902] | <0.001 | confirmed |
| US-101 | macro error | without ResidualIDM | 11 | 11 | 7 | 0.638 | [0.014, 0.888] | [-0.185, 0.966] | [0.014, 0.888] | 0.038 | confirmed |
| US-101 | macro error | learned laws only (without IDM, ResidualIDM) | 9 | 9 | 6 | 0.417 | [-0.431, 0.878] | [-0.580, 1.000] | [-0.500, 0.879] | 0.269 | open |
| pooled (corridor as stratum) | macro error | all laws with runs | 37 | 19 | 8 | 0.755 | [0.534, 0.859] | [0.123, 0.843] | [0.563, 0.843] | <0.001 | confirmed |
| pooled (corridor as stratum) | macro error | without IDM | 32 | 16 | 7 | 0.761 | [0.490, 0.894] | [-0.275, 0.869] | [0.543, 0.856] | <0.001 | confirmed |
| pooled (corridor as stratum) | macro error | without k-NN | 35 | 18 | 7 | 0.739 | [0.498, 0.861] | [0.171, 0.814] | [0.524, 0.842] | <0.001 | confirmed |
| pooled (corridor as stratum) | macro error | without MLP | 33 | 17 | 7 | 0.791 | [0.562, 0.899] | [-0.075, 0.873] | [0.588, 0.885] | <0.001 | confirmed |
| pooled (corridor as stratum) | macro error | without PIDL | 35 | 18 | 7 | 0.748 | [0.507, 0.859] | [0.198, 0.807] | [0.534, 0.852] | <0.001 | confirmed |
| pooled (corridor as stratum) | macro error | without GRU | 33 | 17 | 7 | 0.759 | [0.522, 0.855] | [-0.253, 0.846] | [0.545, 0.842] | <0.001 | confirmed |
| pooled (corridor as stratum) | macro error | without LSTM | 33 | 17 | 7 | 0.753 | [0.448, 0.894] | [0.523, 0.873] | [0.499, 0.874] | <0.001 | confirmed |
| pooled (corridor as stratum) | macro error | without PERL | 35 | 18 | 7 | 0.739 | [0.505, 0.860] | [0.171, 0.814] | [0.521, 0.839] | <0.001 | confirmed |
| pooled (corridor as stratum) | macro error | without ResidualIDM | 23 | 12 | 7 | 0.552 | [-0.004, 0.802] | [-0.245, 0.839] | [0.160, 0.764] | 0.052 | open |
| pooled (corridor as stratum) | macro error | learned laws only (without IDM, ResidualIDM) | 18 | 9 | 6 | 0.367 | [-0.513, 0.810] | [-0.703, 0.870] | [-0.226, 0.719] | 0.307 | open |
| I-80 | macro error (dynamic) | all laws with runs | 19 | 19 | 8 | 0.651 | [0.328, 0.817] | [-0.065, 0.802] | [0.345, 0.818] | 0.003 | confirmed |
| I-80 | macro error (dynamic) | without IDM | 16 | 16 | 7 | 0.679 | [0.334, 0.864] | [-0.376, 0.855] | [0.340, 0.858] | 0.004 | confirmed |
| I-80 | macro error (dynamic) | without k-NN | 18 | 18 | 7 | 0.645 | [0.286, 0.832] | [-0.072, 0.812] | [0.310, 0.832] | 0.005 | confirmed |
| I-80 | macro error (dynamic) | without MLP | 17 | 17 | 7 | 0.710 | [0.391, 0.874] | [-0.196, 0.865] | [0.392, 0.873] | 0.002 | confirmed |
| I-80 | macro error (dynamic) | without PIDL | 18 | 18 | 7 | 0.667 | [0.333, 0.831] | [-0.047, 0.811] | [0.342, 0.833] | 0.004 | confirmed |
| I-80 | macro error (dynamic) | without GRU | 17 | 17 | 7 | 0.719 | [0.439, 0.855] | [-0.346, 0.845] | [0.435, 0.853] | 0.002 | confirmed |
| I-80 | macro error (dynamic) | without LSTM | 17 | 17 | 7 | 0.640 | [0.260, 0.845] | [0.011, 0.858] | [0.263, 0.844] | 0.007 | confirmed |
| I-80 | macro error (dynamic) | without PERL | 18 | 18 | 7 | 0.618 | [0.274, 0.806] | [-0.122, 0.771] | [0.280, 0.806] | 0.009 | confirmed |
| I-80 | macro error (dynamic) | without ResidualIDM | 12 | 12 | 7 | 0.235 | [-0.326, 0.685] | [-0.324, 0.726] | [-0.321, 0.674] | 0.462 | refuted |
| I-80 | macro error (dynamic) | learned laws only (without IDM, ResidualIDM) | 9 | 9 | 6 | 0.250 | [-0.561, 0.735] | [-0.800, 0.962] | [-0.572, 0.722] | 0.519 | refuted |
| US-101 | macro error (dynamic) | all laws with runs | 18 | 18 | 8 | 0.788 | [0.530, 0.920] | [0.185, 0.942] | [0.532, 0.932] | <0.001 | confirmed |
| US-101 | macro error (dynamic) | without IDM | 16 | 16 | 7 | 0.809 | [0.556, 0.950] | [-0.147, 0.980] | [0.556, 0.952] | <0.001 | confirmed |
| US-101 | macro error (dynamic) | without k-NN | 17 | 17 | 7 | 0.772 | [0.459, 0.927] | [0.242, 0.925] | [0.484, 0.921] | <0.001 | confirmed |
| US-101 | macro error (dynamic) | without MLP | 16 | 16 | 7 | 0.847 | [0.604, 0.958] | [0.068, 0.991] | [0.603, 0.955] | <0.001 | confirmed |
| US-101 | macro error (dynamic) | without PIDL | 17 | 17 | 7 | 0.772 | [0.459, 0.926] | [0.286, 0.924] | [0.484, 0.921] | <0.001 | confirmed |
| US-101 | macro error (dynamic) | without GRU | 16 | 16 | 7 | 0.758 | [0.448, 0.897] | [-0.142, 0.928] | [0.452, 0.889] | 0.002 | confirmed |
| US-101 | macro error (dynamic) | without LSTM | 16 | 16 | 7 | 0.786 | [0.392, 0.974] | [0.500, 0.991] | [0.385, 0.971] | <0.001 | confirmed |
| US-101 | macro error (dynamic) | without PERL | 17 | 17 | 7 | 0.775 | [0.463, 0.930] | [0.286, 0.934] | [0.483, 0.926] | <0.001 | confirmed |
| US-101 | macro error (dynamic) | without ResidualIDM | 11 | 11 | 7 | 0.633 | [0.090, 0.924] | [-0.148, 0.991] | [0.088, 0.934] | 0.042 | confirmed |
| US-101 | macro error (dynamic) | learned laws only (without IDM, ResidualIDM) | 9 | 9 | 6 | 0.433 | [-0.339, 0.945] | [-0.580, 1.000] | [-0.421, 0.947] | 0.247 | open |
| pooled (corridor as stratum) | macro error (dynamic) | all laws with runs | 37 | 19 | 8 | 0.718 | [0.463, 0.847] | [0.098, 0.834] | [0.513, 0.829] | <0.001 | confirmed |
| pooled (corridor as stratum) | macro error (dynamic) | without IDM | 32 | 16 | 7 | 0.744 | [0.452, 0.898] | [-0.255, 0.910] | [0.520, 0.855] | <0.001 | confirmed |
| pooled (corridor as stratum) | macro error (dynamic) | without k-NN | 35 | 18 | 7 | 0.707 | [0.427, 0.851] | [0.150, 0.811] | [0.479, 0.828] | <0.001 | confirmed |
| pooled (corridor as stratum) | macro error (dynamic) | without MLP | 33 | 17 | 7 | 0.776 | [0.556, 0.888] | [-0.038, 0.923] | [0.576, 0.874] | <0.001 | confirmed |
| pooled (corridor as stratum) | macro error (dynamic) | without PIDL | 35 | 18 | 7 | 0.718 | [0.445, 0.847] | [0.161, 0.811] | [0.497, 0.830] | <0.001 | confirmed |
| pooled (corridor as stratum) | macro error (dynamic) | without GRU | 33 | 17 | 7 | 0.738 | [0.503, 0.844] | [-0.249, 0.838] | [0.529, 0.831] | <0.001 | confirmed |
| pooled (corridor as stratum) | macro error (dynamic) | without LSTM | 33 | 17 | 7 | 0.710 | [0.376, 0.888] | [0.356, 0.910] | [0.449, 0.858] | <0.001 | confirmed |
| pooled (corridor as stratum) | macro error (dynamic) | without PERL | 35 | 18 | 7 | 0.694 | [0.424, 0.841] | [0.120, 0.805] | [0.467, 0.815] | <0.001 | confirmed |
| pooled (corridor as stratum) | macro error (dynamic) | without ResidualIDM | 23 | 12 | 7 | 0.425 | [-0.064, 0.740] | [-0.228, 0.819] | [0.029, 0.693] | 0.140 | open |
| pooled (corridor as stratum) | macro error (dynamic) | learned laws only (without IDM, ResidualIDM) | 18 | 9 | 6 | 0.342 | [-0.445, 0.826] | [-0.679, 0.981] | [-0.202, 0.689] | 0.351 | open |
