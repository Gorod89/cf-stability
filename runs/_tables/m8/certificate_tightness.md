# Tightness of the certificate of E4 (D116)

- Tightness of the certificate of E4 (D116): every run of e4_stable, e4_stable_ft and the r_max variants e4_stable_r{0.1,0.2,0.5}, e4_stable_ft_r{0.1,0.2,0.5} (ResidualIDM, core with margin 0.2, certified budget; folds [0, 1, 2, 3, 4] x seeds [0, 1, 2, 3, 4]; before fine-tuning on follownet_highd, after on ngsim_i80).
- Per grid speed: audited = the margin M = f_v² + 2 f_v f_dv - 2 f_s of the hybrid at its equilibrium (analytic criterion of the audit, stability.json audit.equilibria[].margin); bound a priori = certificate.json per_speed[].a_priori_margin (minimum over every spacing that can be an equilibrium of a residual of amplitude r_max and over the box of the residual derivatives), compared at every speed with an equilibrium (inside or outside the band); bound at the equilibria = per_speed[].guaranteed_margin (core derivatives at the hybrid's equilibrium plus the worst residual derivatives), compared at the anchored equilibria the certificate uses (found), at the spacing of the audit.
- slack = audited - bound: median, 10 % quantile and minimum over the speeds of all runs; negative slack would violate the certificate (expected 0).

| r_max (m/s²) | stage | bound | experiment | data | runs | speeds | bound median (s⁻²) | audited M median (s⁻²) | slack median | slack q10 | slack min | negative slack | share negative |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | before | a priori | e4_stable_r0.1 | follownet_highd | 25 | 650 | 0.0812 | 0.1980 | 0.1134 | 0.0914 | 0.0773 | 0 | 0.000 |
| 0.1 | before | at the equilibria | e4_stable_r0.1 | follownet_highd | 25 | 583 | 0.1021 | 0.1974 | 0.0914 | 0.0804 | 0.0730 | 0 | 0.000 |
| 0.1 | after | a priori | e4_stable_ft_r0.1 | ngsim_i80 | 25 | 650 | 0.0812 | 0.1863 | 0.1011 | 0.0811 | 0.0723 | 0 | 0.000 |
| 0.1 | after | at the equilibria | e4_stable_ft_r0.1 | ngsim_i80 | 25 | 650 | 0.0813 | 0.1863 | 0.1010 | 0.0811 | 0.0723 | 0 | 0.000 |
| 0.2 | before | a priori | e4_stable_r0.2 | follownet_highd | 25 | 650 | 0.0792 | 0.1849 | 0.1033 | 0.0786 | 0.0643 | 0 | 0.000 |
| 0.2 | before | at the equilibria | e4_stable_r0.2 | follownet_highd | 25 | 570 | 0.1040 | 0.1846 | 0.0774 | 0.0673 | 0.0555 | 0 | 0.000 |
| 0.2 | after | a priori | e4_stable_ft_r0.2 | ngsim_i80 | 25 | 650 | 0.0792 | 0.1538 | 0.0753 | 0.0593 | 0.0523 | 0 | 0.000 |
| 0.2 | after | at the equilibria | e4_stable_ft_r0.2 | ngsim_i80 | 25 | 650 | 0.0800 | 0.1538 | 0.0746 | 0.0591 | 0.0522 | 0 | 0.000 |
| 0.3 | before | a priori | e4_stable | follownet_highd | 25 | 650 | 0.0764 | 0.1478 | 0.0728 | 0.0563 | 0.0465 | 0 | 0.000 |
| 0.3 | before | at the equilibria | e4_stable | follownet_highd | 25 | 570 | 0.0917 | 0.1472 | 0.0539 | 0.0463 | 0.0373 | 0 | 0.000 |
| 0.3 | after | a priori | e4_stable_ft | ngsim_i80 | 25 | 650 | 0.0764 | 0.1238 | 0.0499 | 0.0397 | 0.0346 | 0 | 0.000 |
| 0.3 | after | at the equilibria | e4_stable_ft | ngsim_i80 | 25 | 650 | 0.0764 | 0.1238 | 0.0499 | 0.0397 | 0.0346 | 0 | 0.000 |
| 0.5 | before | a priori | e4_stable_r0.5 | follownet_highd | 25 | 650 | 0.0660 | 0.1158 | 0.0499 | 0.0406 | 0.0351 | 0 | 0.000 |
| 0.5 | before | at the equilibria | e4_stable_r0.5 | follownet_highd | 25 | 545 | 0.0983 | 0.1121 | 0.0139 | 0.0116 | 0.0087 | 0 | 0.000 |
| 0.5 | after | a priori | e4_stable_ft_r0.5 | ngsim_i80 | 25 | 650 | 0.0660 | 0.0877 | 0.0216 | 0.0181 | 0.0161 | 0 | 0.000 |
| 0.5 | after | at the equilibria | e4_stable_ft_r0.5 | ngsim_i80 | 25 | 650 | 0.0746 | 0.0877 | 0.0129 | 0.0098 | 0.0081 | 0 | 0.000 |
