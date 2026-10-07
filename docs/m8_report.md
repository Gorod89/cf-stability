# Milestone M8 report: strengthening arms

Date: 2026-10-07. Contract: `docs/m8_contract.md`. Decisions: `docs/decisions.md`, D115-D124.
Theory: `docs/theory_notes.md` (thesis, the gain expansion, the certificate as a proposition,
the price of a common equilibrium, the loophole table). Tables: `runs/_tables/m8/`; figures:
`runs/_report/supplement/figures/` (21 + the refreshed corridor grids); the generated report
`runs/_report/report.md` section 11. Git: `1b91820` (M8 code) and the commit of this report
with the run outputs.

## 1. Commands run

```bash
python scripts/make_figures_supplement.py                                     # D115, D116, D123: 3 tables, 21 figures
python scripts/run_experiment.py configs/queue/e2_monotone.yaml                # 3 runs (D117)
python scripts/calibrate_idm.py dataset=ngsim_i80_p0 per_event=false           # global IDM of period 0 (D120)
python scripts/run_experiment.py configs/queue/m8_temporal.yaml                # 15 runs (D120)
python scripts/calibrate_idm.py dataset=ngsim_i80 per_event_margin=0.2 global_fit=false   # 7 651 cores with margin 0.2 (D118), 12 min
python scripts/certify_cores.py folds=[0,1,2,3,4]                              # 3 337 of 7 651 cores certified (43.6 %)
python scripts/run_experiment.py configs/queue/audit_bands.yaml                # 300 audits with the bands 1/99 and 10/90 (D119)
python scripts/analysis/band_sensitivity.py                                    # runs/_tables/m8/band_sensitivity
python scripts/build_corridor.py scenarios=false                               # laws *_p0 and residual_idm_certified_het
python scripts/run_corridor.py scenarios=[i80_p1,i80_p2] laws=[idm_global_p0,residual_idm_certified_p0,mlp_p0,gru_p0] seeds=[0..9] workers=4   # 80
python scripts/run_corridor.py scenarios=[i80_p0,i80_p1,i80_p2,us101_p0,us101_p1,us101_p2] laws=[residual_idm_certified_het] seeds=[0..9] workers=4   # 60
python scripts/corridor_metrics.py; python scripts/corridor_asymmetry.py workers=4           # D121: 1 276 asymmetry files
python scripts/make_tables.py; python scripts/corridor_metrics.py compute=false tables=true; python scripts/make_report.py
./.venv/Scripts/python.exe -m pytest                                           # 607 tests: 583 passed, 24 skipped (work package C2, after the M8 code)
```

## 2. Theory checks (D115, D116, D123 a)

- **Gain expansion.** The audits store the partial derivatives and the gain per speed and
  frequency, so nothing was approximated. For the memoryless laws the numerical gain of the
  audit equals the exact linearised gain |G|^2 = 1 - omega^2 (M + omega^2) / D of the theory
  notes to better than 1e-4 (median). The truncated expansion 1 - omega^2 M / f_s^2 is further
  away (median 0.0002-0.0099 at 0.02 rad/s, 90 % quantile up to 0.057) because the slow pole
  f_s / |f_dv + f_v| of the audited laws is 0.018-0.045 rad/s, the order of the audit's lowest
  frequencies; the sign of |G| - 1 agrees with the sign of M at 96-100 % of the speeds, the
  exceptions lying in -omega^2 < M < 0 where (1) says so. The recurrent laws of E1 are 0.10-0.28
  away from the memoryless gain (the memory effect); with the chosen E2 weight GRU and LSTM
  are at 0.023 and 0.008 (same sign 100 % and 96 %), PERL at 0.30 with half its speeds locally
  unstable.
- **Certificate tightness.** 200 runs, four amplitudes, before and after fine-tuning: no
  negative slack in 5 200 a priori and 4 868 at-equilibrium comparisons (smallest slack 0.008
  s^-2). Median slack (audited minus certified margin) before / after fine-tuning: 0.113 /
  0.101 (r_max 0.1), 0.103 / 0.075 (0.2), 0.073 / 0.050 (0.3), 0.050 / 0.022 (0.5): the bound
  tightens as the amplitude grows because the admissible derivative budget shrinks
  (0.18-0.20 at r_max 0.1, 0.026-0.031 at 0.5); after fine-tuning the equilibria sit at the
  upper end of the admissible spacing interval (all of them at r_max 0.3).
- **Band width.** Relative width (s_95 - s_5) / s_50 of the spacing band: 1.13-2.51 (median
  1.67) on HighD, 1.20-2.06 (median 1.63) on I-80; the per-driver dispersion of the
  equilibrium spacing is 0.33-0.74 (HighD) and 0.30-0.52 (I-80) of the median, and 3.29 times
  that dispersion divided by the band width has a median of 1.05 / 0.97: the band of D78 is
  the spread of the drivers' own equilibria, which bounds what memory can gain on the
  near-steady part of the data.

## 3. RACER-type control (D117)

Fold 0, seed 0, weight 1, the monotonicity terms relu(-f_s) + relu(f_dv) + relu(f_v) alone:

| architecture | E1 unstable | monotone unstable | Jacobian (E2) unstable | RMSE change monotone / Jacobian |
|---|---:|---:|---:|---|
| MLP | 0.91 | 1.00 | 0.00 | +2.3 % / -2.6 % |
| PIDL | 0.86 | 0.95 | 0.00 | +1.1 % / -0.9 % |
| ResidualIDM | 0.27 | 0.50 | 0.00 | +2.5 % / +2.0 % |

The sign constraints are met (penalty below 0.01 within two epochs) and leave the string
instability where it was or worse: the string term is what stabilises, the rational
constraints are not a substitute.

## 4. Band sensitivity of the audit (D119)

Offline reclassification is not exact: the anchored search of D79 moves the equilibrium
itself when the band changes (32 of 3 300 speeds; e.g. a GRU equilibrium outside the band at
7.4 m becomes one inside at 81.2 m), so the 300 audits were rerun with the bands 1/99 and
10/90 (150 E1 runs of the learned models, seed 0-4, folds 0-4).

| architecture | unstable among equilibria, 1/99 / 5/95 / 10/90 | unstable in band | outside the band |
|---|---|---|---|
| MLP | 0.94 / 0.94 / 0.94 | 0.94 / 0.94 / 0.92 | 0.00 / 0.00 / 0.02 |
| PIDL | 0.84 / 0.84 / 0.84 | 0.84 / 0.84 / 0.80 | 0.00 / 0.00 / 0.04 |
| ResidualIDM | 0.31 / 0.31 / 0.31 | 0.31 / 0.31 / 0.28 | 0.00 / 0.00 / 0.14 |
| GRU | 0.85 / 0.86 / 0.86 | 0.68 / 0.53 / 0.42 | 0.13 / 0.31 / 0.45 |
| LSTM | 0.92 / 0.92 / 0.92 | 0.55 / 0.43 / 0.37 | 0.27 / 0.39 / 0.45 |
| PERL | 0.96 / 0.95 / 0.95 | 0.81 / 0.66 / 0.52 | 0.14 / 0.30 / 0.44 |

The H1.1 statistic (the share of unstable equilibria among the equilibria found, D92) does
not depend on the band to two decimals for any architecture: the verdict of H1.1 does not
hinge on D78. The band decides only how the recurrent models' equilibria split between
"unstable" and "outside": with the narrow 10/90 band 44-45 % of their speeds have the
equilibrium outside, with the wide 1/99 band 13-27 %; the memoryless laws keep their
equilibria inside every band (at most 4 %, the hybrid 14 % at 10/90).

## 5. Heterogeneous certified hybrid (D118)

Per-event IDM cores of I-80 calibrated with the margin 0.2 (7 651 events, 12 min on the GPU;
the margin costs about 13 % of the fit objective and raises the median spacing RMSE of the
per-event fit from 1.33 to 1.86 m in the 200-event smoke test; all five parameters free). The
a priori certificate with the shared certified residual (r_max 0.3, the five fold members)
holds for 3 337 of the 7 651 cores (43.6 %); the law draws from those and rechecks every core
against the largest member bound (3 337 hold).

| law | macro error I-80 | macro error US-101 | dynamic I-80 / US-101 | collisions / 1000 veh-km I-80, US-101 |
|---|---:|---:|---|---|
| residual_idm_certified (one global core) | 0.125 [0.109, 0.144] | 0.113 [0.107, 0.119] | 0.190 / 0.122 | 0, 0.23 |
| residual_idm_certified_het (per-event cores) | 0.200 [0.176, 0.226] | 0.174 [0.159, 0.189] | 0.368 / 0.266 | 0, 0.67 |
| idm_heterogeneous | 0.302 | 0.125 | 0.520 / 0.196 | 0, 0.15 |
| idm_global | 0.182 | 0.133 | 0.345 / 0.229 | 0, 0.03 |

Fleet heterogeneity of the certified cores does not pay: the law with per-event cores is
worse than the single global core on both corridors (its share of unstable equilibria is 0,
every core is certified), better than the heterogeneous IDM on I-80 and worse than it on
US-101. The certificate scales to a population (3 337 cores certified against the same
residual budget), the corridor fidelity does not: a population of margin-constrained cores
spreads the equilibrium spacings and the waves more than the data do.

## 6. Temporal hold-out on I-80 (D120)

View `ngsim_i80_p0` (2 635 events of period 0); the global IDM, the certified hybrid (from
e4_stable), the MLP and the GRU (from E1) fine-tuned on it (15 runs; the hybrid's a priori
certificate holds at 26 of 26 speeds after fine-tuning). Macro error on periods 1-2 (20 runs
each), the law fine-tuned on period 0 against the same law fine-tuned on all periods (D97),
paired by scenario and seed:

| law | fine-tuned on period 0 | fine-tuned on all periods | difference | outcome |
|---|---:|---:|---|---|
| idm_global | 0.186 [0.172, 0.199] | 0.174 [0.170, 0.178] | +7 % [-1 %, +15 %], p 0.20 | no difference |
| residual_idm_certified | 0.173 [0.146, 0.202] | 0.136 [0.116, 0.161] | +27 % [+7 %, +59 %], p 0.04 | worse |
| mlp | 0.230 [0.213, 0.249] | 0.259 [0.236, 0.282] | -11 % [-19 %, -2 %], p 0.015 | better |
| gru | 0.107 [0.092, 0.123] | 0.147 [0.132, 0.162] | -27 % [-35 %, -17 %], p 0.0001 | better |

The temporal hold-out within I-80 removes part of the certified hybrid's in-sample advantage
(its error on the held-out periods rises by a quarter), leaves the IDM where it was and
improves the networks, so that on periods 1-2 the GRU fine-tuned on period 0 has the lowest
macro error of the four (0.107, with 196 collision episodes per 1 000 veh-km) ahead of the
certified hybrid (0.173, no collisions) and the IDM (0.186). Together with US-101 the picture
is: across sites the certified hybrid keeps its fidelity and the networks collapse; across
periods of one site a network with collisions can match or beat it on the macro error. The
collision-free property, not the macro error alone, is what separates the certified hybrid.

## 7. Asymmetry and spectrum (D121)

| law | asymmetry index I-80 / US-101 (data 0.94-0.99 / 1.00-1.03) | share of time accelerating (data 38-44 %) |
|---|---|---|
| idm_global | 1.23 [1.21, 1.24] / 1.13 [1.04, 1.21] | |
| residual_idm_certified | 1.28 [1.25, 1.31] / 1.49 [1.47, 1.50] | |
| residual_idm (free) | 0.65 / 0.50 | |
| mlp, gru, lstm | 0.36 / 0.27, 0.38 / 0.32, 0.61 / 0.62 | 56-65 % (mlp, gru, pidl) |

The penalties change the asymmetry little (mlp_penalty -0.008 [-0.027, 0.011] on I-80,
gru_penalty +0.033 [0.020, 0.050], lstm_penalty +0.087 towards the data); the certificate
reverses the asymmetry of the free core (+0.63 / +0.99 against `residual_idm`, +0.66 / +0.96
against the free core at r_max 0.3) and ends about as far from the data on the other side. So
the claim of the strengthening note ("the penalty and the certificate do not erase the
asymmetry of human driving") is false as a claim about the certificate: the learned laws
over-decelerate, the certified hybrid over-accelerates, and neither is the data. The peak
frequency of the detector spectra sits on the lowest line (1/128 Hz) for data and runs
alike; the spectral centroid separates the laws: data 0.0197 / 0.0116 Hz, certified hybrid
+5.8 % / -0.1 %, IDM -8.9 % / -11.5 %, GRU -25.4 % / +12.6 %.

## 8. Pooled correlation and power (D122)

H12.3 over every law with runs (19 on I-80, 18 on US-101, the heterogeneous certified hybrid
included): Spearman 0.734 (p 0.0008) and 0.777 (p 0.0006); pooled with the corridor as
stratum 0.755, p 0.0001 over 37 law-corridor pairs (0.682 with the D108 laws only; before the
het law: 0.738 [0.439, 0.882], 0.785 [0.552, 0.910], pooled 0.761 [0.576, 0.854]; dynamic
macro error pooled 0.731). Power: the macro error's standard deviation over seeds within a scenario
is 0.014 (IDM) and 0.046 (certified) on I-80, 0.064 and 0.016 on US-101; the 10-seed design
detects a 16.6 % (I-80) and 27.9 % (US-101) difference between the two laws at alpha 0.05 and
power 0.8, so the certified hybrid's margin over the IDM is significant on I-80 (31 %) and not
on US-101 (15 %); a 10 % difference would need 27 (I-80) or 73 (US-101) seeds per scenario.
Throughput and mean travel time need 2-3 seeds.

## 9. Release (D124)

`requirements.txt` with the exact versions of the environment; `README.md` section
"Reproduce" with the data access and the commands of M1-M8; the manifest of the report. No
licence file.

## 10. Deviations

D115-D124 as in the contract, with: the corridor grids drawn by new grid functions (the M6
functions hard-code 2-3 columns); the band sensitivity rerun on the GPU (D119, offline
reclassification not exact); the monotone penalty keeps the existence term of D74; the
temporal fine-tuning mirrors e4_stable_ft (core and budget from the checkpoint, no
re-calibration); the asymmetry uses the wave-field lanes and 1 s speed differences (1 Hz
trajectories); the corridor grids of the supplement hold the ground truth and 19 laws per
panel set, so the four `*_p0` laws of periods 1-2 are left out of them (their numbers are in
the `temporal` table). Smoke runs under `runs/m8_smoke/` are kept out of every table.
