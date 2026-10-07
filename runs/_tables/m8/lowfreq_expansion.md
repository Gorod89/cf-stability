# Low-frequency expansion of the gain (D115)

- Low-frequency expansion of the gain (D115): fold-0 runs (seed 0, follownet_highd) of E1 and of the chosen E2 weight (D85, chosen_weights.json; idm: E1 only). Per grid speed with an equilibrium (status ok, multiple or outside: inside or outside the band) the numerical gain of the audit (stability.json, audit.equilibria[].gain) at the 3 lowest audit frequencies (0.02, 0.02423, 0.02936 rad/s) against the expansion |G| = sqrt(max(0, 1 - ω² M / f_s²)), M = f_v² + 2 f_v f_dv - 2 f_s, from the stored partial derivatives f_s, f_dv, f_v (the memoryless view: derivatives summed over the window, for GRU, LSTM and PERL; D66).
- Speeds whose rollout at that frequency was clipped, stopped or collided (audit flags) are left out (flagged): their gain is not a linear response. speeds = speeds with equilibrium - flagged.
- |num - exp|: absolute difference of the numerical gain and the expansion, median and 90 % quantile over the speeds; same sign of |G| - 1: share of the speeds where sign(numerical - 1) = sign(expansion - 1); of which -ω² < M < 0: sign differences where the exact memoryless gain (1) is below 1 while the expansion is above (the band 0 < -M < ω² of docs/theory_notes.md, section 2).
- |num - (1)|: median absolute difference of the numerical gain and the exact gain |G(iω)| of the continuous memoryless linearisation (equation (1) of docs/theory_notes.md, transfer_continuous): the part of |num - exp| that is not the truncation of the expansion (memory of the recurrent models, the discrete integration). slow pole: |f_s| / |f_dv + f_v| (rad/s); the expansion holds for ω well below it.
- Speeds without a linear stationary response, kept in the numbers above: locally unstable = the memoryless view is not locally stable (f_s <= 0 or f_v + f_dv >= 0); fit residual > 0.1 = the projection of the follower speed on sin, cos and 1 leaves more than 10 % of its norm (audit.equilibria[].residual).

| architecture | arm | experiment | ω (rad/s) | speeds with equilibrium | flagged (left out) | speeds | \|num - exp\| median | \|num - exp\| q90 | same sign of \|G\| - 1 | sign differs | of which -ω² < M < 0 | \|num - (1)\| median | slow pole median (rad/s) | locally unstable | fit residual > 0.1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| idm | E1 | e1 | 0.0200 | 25 | 0 | 25 | 0.0002 | 0.0568 | 0.960 | 1 | 1 | 0.0000 | 0.0446 | 0 | 0 |
| idm | E1 | e1 | 0.0242 | 25 | 0 | 25 | 0.0004 | 0.0993 | 0.960 | 1 | 1 | 0.0000 | 0.0446 | 0 | 0 |
| idm | E1 | e1 | 0.0294 | 25 | 0 | 25 | 0.0008 | 0.1718 | 0.960 | 1 | 1 | 0.0000 | 0.0446 | 0 | 0 |
| mlp | E1 | e1 | 0.0200 | 26 | 0 | 26 | 0.0099 | 0.0454 | 1.000 | 0 | 0 | 0.0000 | 0.0294 | 0 | 0 |
| mlp | E1 | e1 | 0.0242 | 26 | 0 | 26 | 0.0185 | 0.0763 | 1.000 | 0 | 0 | 0.0000 | 0.0294 | 0 | 0 |
| mlp | E1 | e1 | 0.0294 | 26 | 0 | 26 | 0.0335 | 0.1247 | 1.000 | 0 | 0 | 0.0000 | 0.0294 | 0 | 0 |
| mlp | E2 | e2_jacobian_w1 | 0.0200 | 26 | 0 | 26 | 0.0084 | 0.0503 | 1.000 | 0 | 0 | 0.0000 | 0.0176 | 0 | 0 |
| mlp | E2 | e2_jacobian_w1 | 0.0242 | 26 | 0 | 26 | 0.0146 | 0.0828 | 1.000 | 0 | 0 | 0.0000 | 0.0176 | 0 | 0 |
| mlp | E2 | e2_jacobian_w1 | 0.0294 | 26 | 0 | 26 | 0.0245 | 0.1346 | 1.000 | 0 | 0 | 0.0000 | 0.0176 | 0 | 0 |
| pidl | E1 | e1 | 0.0200 | 26 | 0 | 26 | 0.0021 | 0.0505 | 1.000 | 0 | 0 | 0.0000 | 0.0403 | 0 | 0 |
| pidl | E1 | e1 | 0.0242 | 26 | 0 | 26 | 0.0041 | 0.0907 | 1.000 | 0 | 0 | 0.0000 | 0.0403 | 0 | 0 |
| pidl | E1 | e1 | 0.0294 | 26 | 0 | 26 | 0.0078 | 0.1580 | 1.000 | 0 | 0 | 0.0001 | 0.0403 | 0 | 0 |
| pidl | E2 | e2_jacobian_w10 | 0.0200 | 26 | 0 | 26 | 0.0073 | 0.0149 | 1.000 | 0 | 0 | 0.0000 | 0.0214 | 0 | 0 |
| pidl | E2 | e2_jacobian_w10 | 0.0242 | 26 | 0 | 26 | 0.0128 | 0.0263 | 1.000 | 0 | 0 | 0.0000 | 0.0214 | 0 | 0 |
| pidl | E2 | e2_jacobian_w10 | 0.0294 | 26 | 0 | 26 | 0.0217 | 0.0448 | 1.000 | 0 | 0 | 0.0000 | 0.0214 | 0 | 0 |
| residual_idm | E1 | e1 | 0.0200 | 26 | 0 | 26 | 0.0010 | 0.0046 | 1.000 | 0 | 0 | 0.0000 | 0.0369 | 0 | 0 |
| residual_idm | E1 | e1 | 0.0242 | 26 | 0 | 26 | 0.0018 | 0.0080 | 1.000 | 0 | 0 | 0.0000 | 0.0369 | 0 | 0 |
| residual_idm | E1 | e1 | 0.0294 | 26 | 0 | 26 | 0.0034 | 0.0135 | 1.000 | 0 | 0 | 0.0000 | 0.0369 | 0 | 0 |
| residual_idm | E2 | e2_jacobian_w0.1 | 0.0200 | 24 | 0 | 24 | 0.0006 | 0.0099 | 1.000 | 0 | 0 | 0.0000 | 0.0334 | 0 | 0 |
| residual_idm | E2 | e2_jacobian_w0.1 | 0.0242 | 24 | 0 | 24 | 0.0011 | 0.0170 | 0.958 | 1 | 0 | 0.0000 | 0.0334 | 0 | 0 |
| residual_idm | E2 | e2_jacobian_w0.1 | 0.0294 | 24 | 0 | 24 | 0.0020 | 0.0284 | 0.958 | 1 | 1 | 0.0000 | 0.0334 | 0 | 0 |
| gru | E1 | e1 | 0.0200 | 22 | 0 | 22 | 0.1877 | 0.6074 | 0.773 | 5 | 0 | 0.1946 | 0.0385 | 0 | 0 |
| gru | E1 | e1 | 0.0242 | 22 | 0 | 22 | 0.2658 | 0.7967 | 0.773 | 5 | 1 | 0.2557 | 0.0385 | 0 | 1 |
| gru | E1 | e1 | 0.0294 | 22 | 0 | 22 | 0.3726 | 1.0393 | 0.773 | 5 | 1 | 0.3098 | 0.0385 | 0 | 1 |
| gru | E2 | e2_gain_w0.1 | 0.0200 | 26 | 0 | 26 | 0.0229 | 0.0386 | 1.000 | 0 | 0 | 0.0234 | 0.1456 | 2 | 3 |
| gru | E2 | e2_gain_w0.1 | 0.0242 | 26 | 0 | 26 | 0.0287 | 0.0540 | 1.000 | 0 | 0 | 0.0310 | 0.1456 | 2 | 3 |
| gru | E2 | e2_gain_w0.1 | 0.0294 | 26 | 0 | 26 | 0.0423 | 0.0760 | 1.000 | 0 | 0 | 0.0447 | 0.1456 | 2 | 3 |
| lstm | E1 | e1 | 0.0200 | 21 | 3 | 18 | 0.2793 | 0.4463 | 0.722 | 5 | 0 | 0.1443 | 0.0301 | 3 | 3 |
| lstm | E1 | e1 | 0.0242 | 21 | 3 | 18 | 0.3969 | 0.6105 | 0.722 | 5 | 0 | 0.1698 | 0.0301 | 3 | 3 |
| lstm | E1 | e1 | 0.0294 | 21 | 3 | 18 | 0.5856 | 0.8319 | 0.722 | 5 | 0 | 0.1353 | 0.0301 | 3 | 3 |
| lstm | E2 | e2_gain_w0.1 | 0.0200 | 26 | 0 | 26 | 0.0079 | 0.1518 | 0.962 | 1 | 0 | 0.0002 | 0.0222 | 0 | 2 |
| lstm | E2 | e2_gain_w0.1 | 0.0242 | 26 | 0 | 26 | 0.0135 | 0.2756 | 0.962 | 1 | 0 | 0.0002 | 0.0222 | 0 | 2 |
| lstm | E2 | e2_gain_w0.1 | 0.0294 | 26 | 0 | 26 | 0.0223 | 0.5453 | 0.962 | 1 | 0 | 0.0001 | 0.0222 | 0 | 2 |
| perl | E1 | e1 | 0.0200 | 26 | 2 | 24 | 0.0955 | 0.1595 | 0.792 | 5 | 0 | 0.1128 | 0.1189 | 11 | 2 |
| perl | E1 | e1 | 0.0242 | 26 | 2 | 24 | 0.1373 | 0.2217 | 0.792 | 5 | 1 | 0.1732 | 0.1189 | 11 | 2 |
| perl | E1 | e1 | 0.0294 | 26 | 2 | 24 | 0.1940 | 0.3171 | 0.792 | 5 | 1 | 0.2563 | 0.1189 | 11 | 2 |
| perl | E2 | e2_gain_w0.1 | 0.0200 | 24 | 11 | 13 | 0.3035 | 19.1619 | 0.615 | 5 | 0 | 0.2833 | 0.0299 | 11 | 11 |
| perl | E2 | e2_gain_w0.1 | 0.0242 | 24 | 9 | 15 | 3.8486 | 14.1380 | 0.933 | 1 | 0 | 3.9152 | 0.0286 | 13 | 13 |
| perl | E2 | e2_gain_w0.1 | 0.0294 | 24 | 10 | 14 | 6.7409 | 8.4615 | 1.000 | 0 | 0 | 6.8938 | 0.0292 | 13 | 13 |
