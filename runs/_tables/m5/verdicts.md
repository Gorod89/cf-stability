# Verdicts of H12.1-H12.3 (corridor)

- Verdicts of H12.1-H12.3 of Part A of the plan (D108); an empty verdict lacks data; complete: drawn from every run of the design on every corridor of the configuration (the corridors used are named).
- H12.1: table h12_1. H12.2: macro part = TOST of residual_idm_certified against idm_global on the macro triple (table tost: throughput, mean travel time, wave speed by cross-correlation), holds when all three are equivalent within +-10 % on every corridor, worse by > 10 % when the interval of a relative difference lies beyond the margin on the bad side (lower throughput, longer travel time, slower wave speed in magnitude) on a corridor; micro part = table h12_2; confirmed when both parts hold, refuted when the macro part is worse, otherwise open.
- H12.3: Spearman over the laws of the macro error against the share of unstable equilibria (table instability_correlation, all laws): confirmed when r > 0.6, p (permutation) < 0.05 and at least 10 laws, refuted when r < 0.3, otherwise open; overall on every corridor. Pearson and the dynamic macro error: the same rule, for information.

| hypothesis | unit | corridors | verdict | complete | basis | information |
|---|---|---|---|---|---|---|
| H12.1 | mlp | I-80 | open | yes | degradation vs idm_global: throughput +0.147 [+0.135, +0.157] degraded; travel time +0.082 [+0.075, +0.091] degraded; wave speed +0.553 [+0.364, +0.809] degraded by >= 0.15 |  |
| H12.1 | gru | I-80 | open | yes | degradation vs idm_global: throughput +0.027 [+0.006, +0.051] degraded; travel time +0.078 [+0.050, +0.108] degraded; wave speed +0.126 [+0.005, +0.296] degraded |  |
| H12.1 | lstm | I-80 | confirmed | yes | degradation vs idm_global: throughput +0.754 [+0.688, +0.817] degraded by >= 0.15; travel time +0.710 [+0.556, +0.866] degraded by >= 0.15; wave speed +0.429 [+0.257, +0.656] degraded by >= 0.15 |  |
| H12.1 | mlp | US-101 | confirmed | yes | degradation vs idm_global: throughput +0.230 [+0.214, +0.248] degraded by >= 0.15; travel time +0.265 [+0.220, +0.313] degraded by >= 0.15; wave speed +0.559 [+0.191, +0.913] degraded by >= 0.15 |  |
| H12.1 | gru | US-101 | confirmed | yes | degradation vs idm_global: throughput +0.208 [+0.176, +0.249] degraded by >= 0.15; travel time +0.472 [+0.340, +0.631] degraded by >= 0.15; wave speed -0.131 [-0.359, +0.084] not degraded |  |
| H12.1 | lstm | US-101 | confirmed | yes | degradation vs idm_global: throughput +0.741 [+0.686, +0.790] degraded by >= 0.15; travel time +1.387 [+1.050, +1.751] degraded by >= 0.15; wave speed +0.156 [-0.114, +0.369] not degraded |  |
| H12.1 | overall | I-80 and US-101 | open | yes | confirmed for >= 2 of 3 networks on every corridor: I-80: mlp open, gru open, lstm confirmed; US-101: mlp confirmed, gru confirmed, lstm confirmed |  |
| H12.2 | residual_idm_certified | I-80 and US-101 | open | yes | fails: macro part (TOST) and micro part; macro (TOST +-10 %): I-80: throughput -1.1 % [-1.4 %, -0.7 %] equivalent, travel time -1.2 % [-1.5 %, -1.0 %] equivalent, wave speed +30.5 % [+23.1 %, +38.2 %] not equivalent (faster); US-101: throughput +0.0 % [-0.1 %, +0.2 %] equivalent, travel time -6.5 % [-7.2 %, -5.9 %] equivalent, wave speed -7.1 % [-25.0 %, +13.9 %] not equivalent (slower); micro: RMSE vs idm +4.8 % [+3.7 %, +5.9 %] over 5521 drivers, not superior | reference rows (no verdict): e4_free_ft residual_idm: RMSE vs IDM -8.8 % [-9.8 %, -7.8 %]; e4_free_ft mlp: RMSE vs IDM +21.2 % [+19.4 %, +23.2 %] |
| H12.3 | Spearman | I-80 | confirmed | yes | r +0.652 [+0.287, +0.834] over 15 laws, p (permutation) 0.0095 | decides |
| H12.3 | Pearson (does not decide) | I-80 | open | yes | r +0.483 [+0.223, +0.789] over 15 laws, p (permutation) 0.0668 | information |
| H12.3 | Spearman, dynamic macro error (does not decide) | I-80 | open | yes | r +0.496 [+0.071, +0.762] over 15 laws, p (permutation) 0.0591 | information |
| H12.3 | Pearson, dynamic macro error (does not decide) | I-80 | open | yes | r +0.457 [+0.075, +0.734] over 15 laws, p (permutation) 0.0837 | information |
| H12.3 | Spearman | US-101 | confirmed | yes | r +0.744 [+0.372, +0.912] over 14 laws, p (permutation) 0.0030 | decides |
| H12.3 | Pearson (does not decide) | US-101 | confirmed | yes | r +0.689 [+0.433, +0.928] over 14 laws, p (permutation) 0.0077 | information |
| H12.3 | Spearman, dynamic macro error (does not decide) | US-101 | confirmed | yes | r +0.733 [+0.366, +0.933] over 14 laws, p (permutation) 0.0037 | information |
| H12.3 | Pearson, dynamic macro error (does not decide) | US-101 | confirmed | yes | r +0.765 [+0.562, +0.951] over 14 laws, p (permutation) 0.0023 | information |
| H12.3 | overall | I-80 and US-101 | confirmed | yes | Spearman, r > 0.6, p < 0.05 and >= 10 laws on every corridor: I-80 confirmed, US-101 confirmed |  |
