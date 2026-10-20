# Long-window check of the windowed-gain penalty (M11, D128)

- Long-window check of D128 (M11, hypothesis H-N3): the rollout gain of the gain penalty with the rollout of e2_horizon_pilot, 380 s after a warm-up of one window at the equilibrium, the follower speed fitted on sin, cos and 1 over the last 252 s (two periods of 0.05 rad/s), leader amplitude 0.2 m/s, at omega 0.02, 0.05, 0.1 rad/s; float64, eval mode; follownet_highd, fold 0, seed 0.
- Equilibria: those of the stored audit (stability.json) at the grid speeds in the support of the training data; inside band: status ok or multiple (the speeds of the stable band share); outside band: reported beside. max gain at a frequency: the largest over the speeds inside the band; share: of those speeds with a gain above 1.02 at one of the three frequencies; the check holds when there is none.
- audit: the largest numerical gain of the audit (discard max(60 s, one period), whole periods >= max(60 s, two periods)) at its frequencies <= 0.1 rad/s at the same speeds, for reference.
- Arms: E1 (no penalty); E2 (gain penalty, weight 0.1, rollout 40 s, last 20 s); E2 long window (the same with the rollout of this check, the revision of 7 October 2026); windowed w<weight>: the windowed-gain penalty of D128 at the pilot weights. Per run: long_window_runs.json next to this table (and long_window.json next to stability.json of the runs of the windowed penalty).

| architecture | arm | speeds (inside band, support) | max gain at 0.02 rad/s | max gain at 0.05 rad/s | max gain at 0.1 rad/s | max gain | share > 1.02 | check holds (all < 1.02) | audit: max gain at omega <= 0.1 | speeds (outside band) | max gain outside |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| gru | E1 | 4 | 1.153 | 1.214 | 1.199 | 1.214 | 1.00 |  | 1.215 | 14 | 2.402 |
| gru | E2 (gain 0.1, rollout 40 s) | 22 | 1.042 | 1.629 | 1.070 | 1.629 | 1.00 |  | 1.641 | 0 |  |
| gru | E2 long window (gain 0.1, rollout 380 s) | 17 | 0.993 | 0.970 | 0.942 | 0.993 | 0.00 | yes | 0.993 | 5 | 1.055 |
| gru | windowed w0.1 | 22 | 1.125 | 1.339 | 1.310 | 1.339 | 0.64 |  | 1.364 | 0 |  |
| gru | windowed w1 | 18 | 0.997 | 0.992 | 0.990 | 0.997 | 0.00 | yes | 0.997 | 4 | 1.117 |
| gru | windowed w10 | 22 | 0.997 | 0.989 | 0.983 | 0.997 | 0.00 | yes | 0.997 | 0 |  |
| lstm | E1 | 12 | 1.760 | 1.877 | 1.155 | 1.877 | 0.50 |  | 2.103 | 5 | 117.164 |
| lstm | E2 (gain 0.1, rollout 40 s) | 19 | 1.001 | 0.998 | 0.998 | 1.001 | 0.00 | yes | 1.013 | 3 | 0.924 |
| lstm | E2 long window (gain 0.1, rollout 380 s) | 22 | 1.233 | 1.489 | 1.335 | 1.489 | 0.18 |  | 1.492 | 0 |  |
| lstm | windowed w0.1 | 22 | 1.102 | 1.265 | 1.474 | 1.474 | 0.23 |  | 1.470 | 0 |  |
| lstm | windowed w1 | 22 | 1.050 | 1.159 | 1.144 | 1.159 | 0.59 |  | 1.177 | 0 |  |
| lstm | windowed w10 | 22 | 0.995 | 0.985 | 0.980 | 0.995 | 0.00 | yes | 0.995 | 0 |  |
| perl | E1 | 19 | 1.106 | 1.333 | 2.915 | 2.915 | 0.79 |  | 3.546 | 3 | 2.636 |
| perl | E2 (gain 0.1, rollout 40 s) | 9 | 32.149 | 6.710 | 3.128 | 32.149 | 1.00 |  | 22.556 | 13 | 18.677 |
| perl | windowed w0.1 | 22 | 0.999 | 0.998 | 1.154 | 1.154 | 0.45 |  | 1.130 | 0 |  |
| perl | windowed w1 | 22 | 1.015 | 1.731 | 2.960 | 2.960 | 0.59 |  | 3.385 | 0 |  |
| perl | windowed w10 | 22 | 1.052 | 4.199 | 1.256 | 4.199 | 0.45 |  | 5.918 | 0 |  |
