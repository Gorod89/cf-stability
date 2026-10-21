# Long-window check of the windowed-gain penalty with the maximum over the frequencies (M12, D134)

- Long-window check of D128 (M12, D134, hypothesis H-N3'): the rollout gain of the gain penalty with the rollout of e2_horizon_pilot, 380 s after a warm-up of one window at the equilibrium, the follower speed fitted on sin, cos and 1 over the last 252 s (two periods of 0.05 rad/s), leader amplitude 0.2 m/s, at omega 0.02, 0.05, 0.1 rad/s; float64, eval mode; follownet_highd, fold 0, seed 0.
- Equilibria: those of the stored audit (stability.json) at the grid speeds in the support of the training data; inside band: status ok or multiple (the speeds of the stable band share); outside band: reported beside. max gain at a frequency: the largest over the speeds inside the band; share: of those speeds with a gain above 1.02 at one of the three frequencies; the check holds when there is none.
- audit: the largest numerical gain of the audit (discard max(60 s, one period), whole periods >= max(60 s, two periods)) at its frequencies <= 0.1 rad/s at the same speeds, for reference.
- Arms: E1 (no penalty); windowed mean w<weight> (M11): the windowed-gain penalty of D128 (the mean over the frequencies) at the pilot weights of M11; windowed max w<weight>: the penalty with the maximum over the frequencies of D134 (frequency_reduce max) at the pilot weights of M12. Per run: long_window_wmax_runs.json next to this table (and long_window.json next to stability.json of the runs of the maximum).

| architecture | arm | speeds (inside band, support) | max gain at 0.02 rad/s | max gain at 0.05 rad/s | max gain at 0.1 rad/s | max gain | share > 1.02 | check holds (all < 1.02) | audit: max gain at omega <= 0.1 | speeds (outside band) | max gain outside |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| gru | E1 | 4 | 1.153 | 1.214 | 1.199 | 1.214 | 1.00 |  | 1.215 | 14 | 2.402 |
| gru | windowed mean w0.1 (M11) | 22 | 1.125 | 1.339 | 1.310 | 1.339 | 0.64 |  | 1.364 | 0 |  |
| gru | windowed mean w1 (M11) | 18 | 0.997 | 0.992 | 0.990 | 0.997 | 0.00 | yes | 0.997 | 4 | 1.117 |
| gru | windowed mean w10 (M11) | 22 | 0.997 | 0.989 | 0.983 | 0.997 | 0.00 | yes | 0.997 | 0 |  |
| gru | windowed max w0.1 | 21 | 1.137 | 1.207 | 1.177 | 1.207 | 0.52 |  | 1.208 | 1 | 0.967 |
| gru | windowed max w1 | 22 | 0.996 | 0.986 | 0.978 | 0.996 | 0.00 | yes | 0.996 | 0 |  |
| gru | windowed max w10 | 22 | 0.998 | 0.992 | 0.987 | 0.998 | 0.00 | yes | 0.998 | 0 |  |
| gru | windowed max w100 | 22 | 0.997 | 0.989 | 0.982 | 0.997 | 0.00 | yes | 0.997 | 0 |  |
| lstm | E1 | 12 | 1.760 | 1.877 | 1.155 | 1.877 | 0.50 |  | 2.103 | 5 | 117.164 |
| lstm | windowed mean w0.1 (M11) | 22 | 1.102 | 1.265 | 1.474 | 1.474 | 0.23 |  | 1.470 | 0 |  |
| lstm | windowed mean w1 (M11) | 22 | 1.050 | 1.159 | 1.144 | 1.159 | 0.59 |  | 1.177 | 0 |  |
| lstm | windowed mean w10 (M11) | 22 | 0.995 | 0.985 | 0.980 | 0.995 | 0.00 | yes | 0.995 | 0 |  |
| lstm | windowed max w0.1 | 21 | 1.276 | 1.430 | 1.507 | 1.507 | 0.29 |  | 1.495 | 1 | 0.979 |
| lstm | windowed max w1 | 22 | 0.998 | 0.990 | 0.973 | 0.998 | 0.00 | yes | 0.998 | 0 |  |
| lstm | windowed max w10 | 21 | 0.998 | 0.990 | 0.977 | 0.998 | 0.00 | yes | 0.998 | 1 | 0.978 |
| lstm | windowed max w100 | 22 | 0.999 | 0.996 | 0.988 | 0.999 | 0.00 | yes | 0.999 | 0 |  |
| perl | E1 | 19 | 1.106 | 1.333 | 2.915 | 2.915 | 0.79 |  | 3.546 | 3 | 2.636 |
| perl | windowed mean w0.1 (M11) | 22 | 0.999 | 0.998 | 1.154 | 1.154 | 0.45 |  | 1.130 | 0 |  |
| perl | windowed mean w1 (M11) | 22 | 1.015 | 1.731 | 2.960 | 2.960 | 0.59 |  | 3.385 | 0 |  |
| perl | windowed mean w10 (M11) | 22 | 1.052 | 4.199 | 1.256 | 4.199 | 0.45 |  | 5.918 | 0 |  |
| perl | windowed max w0.1 | 20 | 1.000 | 1.036 | 1.042 | 1.042 | 0.90 |  | 1.141 | 2 | 1.015 |
| perl | windowed max w1 | 22 | 1.463 | 1.432 | 4.673 | 4.673 | 1.00 |  | 3.303 | 0 |  |
| perl | windowed max w10 | 21 | 0.999 | 0.990 | 1.139 | 1.139 | 0.52 |  | 1.134 | 1 | 1.065 |
| perl | windowed max w100 | 18 | 1.006 | 6.153 | 1.038 | 6.153 | 0.56 |  | 6.377 | 4 | 0.992 |
