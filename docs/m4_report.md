# Milestone M4 report: experiments E1-E5

Date: 2026-09-30 (runs of 2026-09-29 and 2026-09-30). Contract: `docs/m4_contract.md`. Decisions:
`docs/decisions.md`, D78-D93. The GPU and the CPU were shared with other jobs during every run;
timings are as measured. Nothing is committed to git.

Tables with intervals: `runs/_tables/m4/*.md` (Markdown) and `*.csv`, written by
`python scripts/make_tables.py`; the verdicts of the hypotheses in `runs/_tables/m4/verdicts.md`.
The basis of every verdict is named in `configs/make_tables.yaml`. Intervals are 95 % percentile
bootstrap intervals (1000 resamples); the unit is the driver for RMSE-type numbers (mean over the
events and the seeds of a driver) and the run (fold and seed) for numbers of a model (D88).

## 1. Commands run

```bash
./.venv/Scripts/python.exe -m pytest                     # 412 tests: 412 passed, 24 skipped (GPU variants)

# previews on the reference fold (fold 0, seed 0), before the experiments
python scripts/run_experiment.py configs/queue/preview.yaml            # anchored penalties, 5 runs
python scripts/run_experiment.py configs/queue/e4_pilot.yaml           # residual amplitude of the certified hybrid, 4 runs
python scripts/run_experiment.py configs/queue/preview_gain.yaml       # rollout penalty, 3 runs
python scripts/run_experiment.py configs/queue/preview_existence.yaml  # stand-in for the existence term, 2 runs

# experiments (restartable queues: complete steps are skipped, a second call retries failed jobs)
python scripts/run_experiment.py configs/queue/e1.yaml                 # 175 jobs
python scripts/run_experiment.py configs/queue/e2_sweep.yaml           # 120 jobs
python scripts/make_tables.py tables=[e2_sweep]                        # chosen weights (D85) -> e2_chosen.yaml, e5_penalised.yaml
python scripts/run_experiment.py configs/queue/e2_chosen.yaml          # 150 jobs
python scripts/run_experiment.py configs/queue/e2_existence.yaml       # 15 jobs
python scripts/run_experiment.py configs/queue/e4.yaml                 # 100 jobs
python scripts/run_experiment.py configs/queue/ngsim_idm.yaml          # 5 jobs
python scripts/run_experiment.py configs/queue/e5.yaml                 # 40 jobs
python scripts/run_experiment.py configs/queue/e5_penalised.yaml       # 30 jobs
python scripts/make_tables.py                                          # all tables and verdicts
```

Every job is a chain of steps: `scripts/train.py` (metrics.json, model.pt, test_events.parquet),
`scripts/audit_stability.py` (stability.json), `scripts/platoon_test.py` (platoon.json), and, for
seed 0, `scripts/evaluate_transfer.py` (transfer.json) or, in E4, `scripts/certificate.py`
(certificate.json). 635 jobs in all; the queues ran from 2026-09-29 19:27 to 2026-09-30 18:24 with
two to six trainings side by side. Six jobs (all with the rollout penalty or the existence term
for a recurrent model, 6 of about 145) ended with a transient CUDA error ("illegal memory access")
while the machine was short of memory; every one of them succeeded when retried, and one of them
was also repeated in a scratch directory: weights identical bit by bit. Five runs of E1 that were
trained while the code was still being edited were repeated with the final code: weights
identical bit by bit.

## 2. What was built

| Part | Files | Content |
|---|---|---|
| Spacing band of the data | `train/tensors.py`, `stability/equilibrium.py` | 5-50-95 % quantiles of the spacing of the near-steady training samples per grid speed (D78); `Band`, equilibria searched inside the band, status `outside` (D79) |
| Penalties | `stability/penalties.py`, `train/trainer.py` | existence term at the band edges (D80), `aggregate: max` (D81), kind `existence`, penalty on every n-th step (D90), rollout penalty replayed from a CUDA graph (0.29 s instead of 3.75 s per step), penalty-aware choice of the best epoch (D89) |
| Audit | `stability/audit.py`, `scripts/audit_stability.py` | band shares stable / unstable / outside / none next to the shares of D75 |
| Queue | `scripts/run_experiment.py`, `eval/experiment_queue.py`, `configs/queue/*.yaml` | restartable queue of separate processes, hash check of every training, timeouts, status file (D82) |
| Platoon | `scripts/platoon_test.py`, `stability/platoon.py` | `platoon.json` per run, hysteresis loop area (D83), start at the anchored equilibrium (D91) |
| Transfer | `scripts/evaluate_transfer.py` | closed loop on `ngsim_i80`, `ngsim_us101`, `waymo`: whole events and the first 15 s (D84) |
| Certificate | `scripts/certificate.py`, `stability/certificate.py` | `certificate.json`; admissible budget at the model's own `r_max` |
| Training script | `scripts/train.py` | views of an event set (`openacc_acc`, `openacc_human`, D87), fine-tuning `init_from`, stable core, certified budget (D86), origin of the events in the cached calibrations |
| Collection and statistics | `eval/collect.py`, `eval/stats.py`, `eval/tables.py`, `scripts/collect_results.py`, `scripts/make_tables.py` | one row per run; bootstrap over drivers or runs, Wilcoxon, Holm, TOST (D88); the tables of this report |
| Newell | `models/newell.py` | a gap that is not finite gives NaN instead of an undefined index (found by a diverged run) |

## 3. Previews on the reference fold

Fold 0, seed 0, FollowNet HighD, penalty weight 1. Shares over the 22 grid speeds inside the
data. "Stable" and "unstable": equilibrium inside the band; the test RMSE of the spacing is
compared with the run without penalty of the same fold (MLP 2.84 m, ResidualIDM 2.71 m, GRU
2.06 m, LSTM 2.23 m, PERL 2.44 m).

| Model | Penalty | Test RMSE (change) | Stable / unstable / outside / none (%) | Largest gain | Numerical = analytic | Reading |
|---|---|---:|---|---:|---:|---|
| MLP | Jacobian | 2.77 (-2.5 %) | 100 / 0 / 0 / 0 | 0.995 | 100 % | the anchored penalty works |
| MLP | linearised gain | 2.89 (+1.8 %) | 95 / 0 / 5 / 0 | 0.996 | 96 % | |
| ResidualIDM | Jacobian | 3.41 (+26 %) | 86 / 0 / 14 / 0 | 1.000 | 100 % | margin 0.5 not reached (penalty 0.48) |
| GRU | linearised gain | 3.17 (+54 %) | 0 / 100 / 0 / 0 | 5.2 | 0 % | derivatives of -35 to -237 1/s at the constant history, measured gain 1.2-2.7 |
| LSTM | linearised gain | 2.98 (+34 %) | 59 / 41 / 0 / 0 | 3.8 | 50 % | locally unstable equilibria with a gain below 1 on the unit circle |
| GRU | rollout gain (specification) | 2.99 (+45 %) | 0 / 100 / 0 / 0 | 1.59 | 100 % | gains 1.05-1.59 at 0.043-0.052 rad/s |
| LSTM | rollout gain | 2.67 (+20 %) | 36 / 55 / 9 / 0 | 1.77 | 73 % | |
| PERL | rollout gain | 3.05 (+25 %) | 0 / 73 / 27 / 0 | 83 | 100 % | |

Two findings decided the design of E2:

1. **The linearised gain penalty is met by the recurrent models without string stability.** The
   GRU builds a needle at the constant history of the equilibrium (summed derivatives of -35 to
   -237 1/s, all sign conditions met, gain 0.99 on the unit circle) while the measured response
   at 0.2 m/s amplifies by 20-170 %; the LSTM has equilibria that are locally unstable (poles
   outside the unit circle) with a gain below 1 on the unit circle. A penalty on constant
   histories tests a recurrent law on inputs it never sees. E2 uses the rollout penalty of the
   specification for GRU, LSTM and PERL (D90), made affordable by moving its step into a CUDA
   graph and by evaluating it on every 8th step.
2. **The rollout penalty of the specification cannot see the frequencies below 0.05 rad/s**: its
   40 s rollouts hold a sixth of the period there, and the recurrent models keep their gains of
   1.05-1.8 at 0.04-0.05 rad/s. E2 inherits this limitation of the specification's penalty.

A diagnostic of the preview (a stand-in for the existence term) suggested that a steady state
inside the data costs the recurrent models 25-41 % of accuracy; the arm `e2_existence` of E2
refuted this (section 5.3, D93).

Pilot of the certified hybrid (D86): core with margin 0.2, `r_max` 0.1 / 0.2 / 0.3 / 0.5 m/s^2
give a validation RMSE of 4.60 / 4.57 / 4.56 / 4.58 m against 4.72 m of the core alone and 3.12 m
of the free IDM; `r_max = 0.3` is used. With `r_max = 1` (the first version of D86) no residual
budget can be certified: the residual could place the equilibrium at gaps where the core itself
is string unstable.

## 4. E1: audit of the unconstrained models (H1.1)

FollowNet HighD, driver split, 5 folds x 5 seeds for the learned models, 5 folds for the
calibrated laws; 175 runs. `runs/_tables/m4/e1.md`.

| Model | Test RMSE of the spacing (m) | Against the IDM | Unstable among the equilibria | Stable / unstable / outside / none (band) | Largest gain (median) |
|---|---:|---:|---:|---|---:|
| IDM | 3.08 [3.04, 3.13] | - | 0.15 [0.14, 0.17] | 0.80 / 0.14 / 0.06 / 0.00 | 1.04 |
| kNN | 3.08 [3.03, 3.13] | +0.0 % | 0.90 [0.87, 0.93] | 0.10 / 0.90 / 0.00 / 0.00 | 3.04 |
| MLP | 2.85 [2.80, 2.89] | -7.7 % | 0.94 [0.91, 0.97] | 0.06 / 0.94 / 0.00 / 0.00 | 1.19 |
| PIDL | 2.79 [2.75, 2.83] | -9.5 % | 0.84 [0.81, 0.88] | 0.16 / 0.84 / 0.00 / 0.00 | 1.22 |
| ResidualIDM (rollout loss) | 2.70 [2.66, 2.75] | -12.3 % | 0.31 [0.28, 0.34] | 0.69 / 0.31 / 0.00 / 0.00 | 1.04 |
| GRU | 2.08 [2.04, 2.12] | -32.6 % | 0.86 [0.80, 0.91] | 0.11 / 0.53 / 0.31 / 0.05 | 4.0 |
| LSTM | 2.19 [2.16, 2.22] | -29.0 % | 0.92 [0.89, 0.96] | 0.05 / 0.43 / 0.39 / 0.12 | 14 |
| PERL | 2.39 [2.36, 2.43] | -22.4 % | 0.95 [0.92, 0.98] | 0.03 / 0.66 / 0.30 / 0.02 | 28 |
| Newell | 3.44 [3.38, 3.49] | +11.4 % | every gap is an equilibrium | | 1.00 |
| OVM | 8.00 [7.89, 8.12] | +160 % | 1.00 | 0.00 / 0.86 / 0.00 / 0.14 | 1.13 |
| persistence | 5.94 [5.82, 6.05] | +93 % | every gap is an equilibrium | | 0 |

**H1.1 holds** (D92: share of unstable equilibria at least 0.5 with the lower end of the
interval above 0.3, and RMSE below the IDM, for at least two of MLP, GRU, LSTM): all three meet
it; every difference to the IDM has p < 0.001 (Wilcoxon over the drivers). The verdict is the
same on the band share "not stable". The band shares repeat the picture of M3: the recurrent
models have an equilibrium inside the band at 16-56 % of the speeds only.

## 5. E2: penalties (H1.2)

### 5.1 Sweep of the weight (5 folds, seed 0)

Cells: mean test RMSE of the spacing (m) / mean share of the speeds that are stable inside the
band. `runs/_tables/m4/e2_sweep.md`.

| Architecture | Penalty | Weight 0.01 | 0.1 | 1 | 10 | Chosen (D85) |
|---|---|---|---|---|---|---|
| MLP | Jacobian | 2.84 / 0.12 | 2.86 / 1.00 | 2.80 / 1.00 | 2.94 / 1.00 | 1 |
| PIDL | Jacobian | 2.80 / 0.11 | 2.81 / 0.48 | 2.80 / 1.00 | 2.78 / 1.00 | 10 |
| ResidualIDM | Jacobian | 2.71 / 0.85 | 2.74 / 0.99 | 3.39 / 0.87 | 3.51 / 0.87 | 0.1 |
| GRU | rollout gain | 2.20 / 0.22 | 2.33 / 0.24 | 2.91 / 0.00 | 2.69 / 0.04 | 0.1 |
| LSTM | rollout gain | 2.29 / 0.31 | 2.74 / 0.55 | 2.70 / 0.10 | 2.73 / 0.03 | 0.1 |
| PERL | rollout gain | 3.03 / 0.00 | 3.00 / 0.00 | 3.03 / 0.00 | 3.30 / 0.00 | 0.1 |

No weight of the recurrent models reaches the share 0.9 of the rule; their chosen weight is the
one with the largest share. The share of the recurrent models does not grow with the weight: the
rollout penalty removes the amplification at 0.05-0.8 rad/s that it measures, and the
amplification at 0.02-0.05 rad/s that the audit measures stays.

### 5.2 Chosen weights, 5 folds x 5 seeds

Pairs with the runs of E1 (fold and seed for the numbers of a model, driver for the RMSE; 300
runs). "Not stable" = 1 - stable (band). p: Wilcoxon over the drivers, Holm over the six
architectures. `runs/_tables/m4/e2.md`.

| Architecture | Penalty, weight | RMSE change | Not stable, E1 -> E2 | Unstable among the equilibria, E2 | Growth error, E1 -> E2 | Hysteresis of the pulse, E1 -> E2 | H1.2 |
|---|---|---:|---|---:|---|---|---|
| MLP | Jacobian, 1 | -0.5 % [-0.9 %, -0.1 %] | 0.94 -> 0.00 [0.00, 0.00] | 0.00 | 0.170 -> 0.104 (-39 %) | 54 -> 69 m^2/s (+27 %) | confirmed |
| PIDL | Jacobian, 10 | -0.3 % [-0.7 %, +0.0 %] | 0.84 -> 0.00 [0.00, 0.00] | 0.00 | 0.255 -> 0.162 (-37 %) | 79 -> 86 (+8 %) | confirmed |
| ResidualIDM | Jacobian, 0.1 | +1.5 % [+1.1 %, +2.0 %] | 0.31 -> 0.00 [0.00, 0.01] | 0.00 | 0.289 -> 0.255 (-12 %) | 56 -> 43 (-24 %) | confirmed |
| GRU | rollout gain, 0.1 | +14.7 % [+13.2 %, +15.9 %] | 0.89 -> 0.66 [0.55, 0.76] | 0.63 | 11.7 -> 0.70 (-94 %) | 69 -> 585 | refuted |
| LSTM | rollout gain, 0.1 | +25.0 % [+23.7 %, +26.3 %] | 0.95 -> 0.39 [0.31, 0.48] | 0.33 | 5.9 -> 0.87 (-85 %) | 55 -> 115 | refuted |
| PERL | rollout gain, 0.1 | +26.9 % [+25.5 %, +28.3 %] | 0.97 -> 1.00 [0.99, 1.00] | 0.98 | 6.7 -> 1.29 (-81 %) | 108 -> 124 | refuted |

All p < 0.001 after Holm except PIDL (0.84: no change of the RMSE). **H1.2 is confirmed for the
memoryless and the residual architectures** (MLP, PIDL, ResidualIDM: not stable 0.00 at a change
of the RMSE between -0.5 % and +1.5 %) **and refuted for the recurrent ones** (GRU, LSTM, PERL:
the RMSE rises by 15-27 %, above the 10 % of the hypothesis, and 39-100 % of the speeds stay
without a stable equilibrium inside the band). The growth error of the recurrent models falls by
81-94 % because the unpenalised models collide in the platoons (growth errors of 6-12); their
platoons still collide on 4-6 of the 6 profiles with the penalty.

### 5.3 The existence term alone (arm `e2_existence`, 5 folds, seed 0)

| Architecture | Test RMSE against E1 | Stable / unstable / outside / none | Largest gain (median) | Profiles with a collision |
|---|---:|---|---:|---:|
| GRU | 2.12 against 2.09 (+1.8 %) | 0.00 / 1.00 / 0.00 / 0.00 | 64 | 5.2 of 6 |
| LSTM | 2.32 against 2.18 (+6.7 %) | 0.00 / 1.00 / 0.00 / 0.00 | 4.3 | 4.2 of 6 |
| PERL | 2.55 against 2.37 (+7.6 %) | 0.00 / 1.00 / 0.00 / 0.00 | 44 | 5.2 of 6 |

A steady state inside the band costs the recurrent models 2-8 % of accuracy, not the 25-41 % the
stand-in of the preview suggested (D93). The accuracy the penalised recurrent models lose in 5.2
is the price of the stability terms of the rollout penalty, which nevertheless leave the low
frequencies unstable: the penalty of the specification is not the right tool for them.

## 6. E3: transfer to NGSIM I-80, US-101 and Waymo

Models of seed 0 of the five folds, evaluated on all events of the target (5 521 drivers of
I-80, 5 866 of US-101, 849 of Waymo). Relative degradation: spacing RMSE of the first 15 s of the
target events over the RMSE of the HighD test parts, minus 1 (D84). `runs/_tables/m4/e3.md`.

| Architecture | Penalty | I-80: RMSE 15 s / whole (m) | I-80: degradation | Waymo: RMSE 15 s / whole (m) | Waymo: degradation |
|---|---|---|---:|---|---:|
| IDM | - | 3.43 / 7.41 | +11 % | 5.17 / 7.52 | +68 % |
| MLP | none | 3.13 / 6.07 | +10 % | 3.92 / 5.44 | +38 % |
| MLP | Jacobian, 1 | 3.19 / 5.47 | +14 % | 3.88 / 5.34 | +39 % |
| PIDL | none | 3.17 / 6.82 | +13 % | 4.68 / 6.75 | +67 % |
| PIDL | Jacobian, 10 | 3.19 / 6.44 | +15 % | 4.54 / 6.50 | +63 % |
| ResidualIDM | none | 3.01 / 5.58 | +12 % | 3.76 / 5.17 | +39 % |
| ResidualIDM | Jacobian, 0.1 | 2.92 / 4.74 | +6 % | 3.13 / 4.15 | +14 % |
| GRU | none | 5.51 / 22.5 | +164 % | 4.33 / 6.72 | +107 % |
| GRU | rollout gain, 0.1 | 4.24 / 7.82 | +82 % | 3.57 / 5.02 | +53 % |
| LSTM | none | 4.67 / 21.2 | +114 % | 4.63 / 7.23 | +113 % |
| LSTM | rollout gain, 0.1 | 3.70 / 8.31 | +35 % | 4.59 / 6.70 | +68 % |
| PERL | none | 5.64 / 11.4 | +138 % | 3.50 / 4.85 | +48 % |
| PERL | rollout gain, 0.1 | 7.51 / 19.7 | +150 % | 4.92 / 6.91 | +64 % |

US-101 behaves like I-80 (GRU without penalty +90 %, with penalty +33 %; LSTM +59 % and +13 %; whole
events 26-31 m without penalty). The unconstrained recurrent models lose their advantage in
the transfer: on the whole 48 s events of I-80 the GRU and the LSTM reach 21-22 m of error
(the MLP 6 m, the IDM 7 m). The penalty halves the degradation of GRU and LSTM (differences
-82 and -79 points on I-80, p < 0.001 over the drivers), and the penalised ResidualIDM
transfers best of all (degradation +6 % on I-80, +14 % on Waymo); the penalised memoryless
models transfer like the unpenalised ones (+1 to +3 points).

## 7. E4: certificate of the hybrid (H1.5)

Core with margin 0.2 (`v0` 35.1-35.3, `T` 0.98-0.99, `s0` 3.19-3.28, `a` 1.57-1.58, `b` 0.5 on
the five folds), `r_max = 0.3 m/s^2`, `lipschitz = 0.9 * b / r_max` with the admissible product
`b = 0.100-0.108`; 25 runs per row. Fine-tuning: the training of M2 continued on the training
parts of the driver folds of `ngsim_i80` (same fold number), scaler and core kept.
`runs/_tables/m4/e4.md`.

| Model | Data | A priori certificate holds | Unstable (band) | Not stable | Spacing RMSE of the test part (m) |
|---|---|---:|---:|---:|---:|
| ResidualIDM, free core (E1) | HighD | - | 0.31 [0.28, 0.34] | 0.31 | 2.70 [2.66, 2.75] |
| ResidualIDM, free core, fine-tuned | NGSIM I-80 | 0 of 25 | 0.04 [0.02, 0.06] | 0.04 | 4.49 [4.40, 4.58] |
| MLP (E1) | HighD | - | 0.94 [0.91, 0.96] | 0.94 | 2.85 [2.80, 2.89] |
| MLP, fine-tuned | NGSIM I-80 | - | 0.98 [0.96, 0.99] | 0.99 | 5.96 [5.87, 6.07] |
| ResidualIDM, certified | HighD | 25 of 25 | 0.00 [0.00, 0.00] | 0.05 | 4.51 [4.45, 4.57] |
| ResidualIDM, certified, fine-tuned | NGSIM I-80 | 25 of 25 | 0.00 [0.00, 0.00] | 0.00 | 5.16 [5.04, 5.28] |

**H1.5 holds**: after the fine-tuning the certified hybrid is string unstable at 0 % of the
speeds (a priori certificate and certificate at the equilibria in 25 of 25 runs), the fine-tuned
MLP at 98 %. The price of the certificate is the core: 4.51 m against 2.70 m of the free hybrid
on HighD (+67 %) and 3.08 m of the free IDM. On NGSIM the order changes: the certified hybrid
(5.16 m) beats the fine-tuned MLP (5.96 m); the free hybrid fine-tuned on NGSIM becomes almost
stable by itself (unstable 4 %), because the congested data pull its equilibrium to small gaps
where the IDM core is stable.

## 8. E5: control on OpenACC and the growth of oscillations (H1.3)

Views `openacc_acc` (24 followers driving in ACC mode) and `openacc_human` (23 followers), without
the five runs that give the leader profiles of the platoon test, 5 folds, seed 0; the penalised
MLP, GRU and LSTM with the weights chosen on HighD. Events last up to 60 s, so the RMSE is
larger than on HighD. `runs/_tables/m4/e5.md`.

| View | Model | Penalty | Spacing RMSE (m) | Stable / unstable / outside / none | Growth error on the profiles of the view | Reduction by the penalty |
|---|---|---|---:|---|---:|---:|
| ACC | IDM | - | 7.45 | 0.87 / 0.05 / 0.08 / 0.00 | 0.105 | |
| ACC | MLP | none | 7.56 | 0.00 / 0.92 / 0.08 / 0.00 | 0.087 | |
| ACC | MLP | Jacobian, 1 | 7.84 | 1.00 / 0.00 / 0.00 / 0.00 | 0.089 | -3 % [-29 %, +15 %] |
| ACC | GRU | none | 8.45 | 0.03 / 0.78 / 0.16 / 0.02 | 0.092 | |
| ACC | GRU | rollout gain, 0.1 | 7.18 | 0.38 / 0.51 / 0.12 / 0.00 | 0.093 | -1 % [-6 %, +6 %] |
| ACC | LSTM | none | 8.62 | 0.02 / 0.60 / 0.32 / 0.07 | 0.123 | |
| ACC | LSTM | rollout gain, 0.1 | 8.05 | 0.55 / 0.24 / 0.21 / 0.00 | 0.097 | +21 % [-3 %, +41 %] |
| human | IDM | - | 9.68 | 0.94 / 0.00 / 0.06 / 0.00 | 0.089 | |
| human | MLP | none | 11.03 | 0.10 / 0.74 / 0.17 / 0.00 | 0.111 | |
| human | MLP | Jacobian, 1 | 11.61 | 0.20 / 0.80 / 0.00 / 0.00 | 0.149 | -34 % [-91 %, +5 %] |
| human | GRU | none | 10.02 | 0.05 / 0.73 / 0.22 / 0.00 | 4.47 | |
| human | GRU | rollout gain, 0.1 | 11.76 | 0.24 / 0.40 / 0.34 / 0.02 | 0.079 | +98 % [+8 %, +99 %] |
| human | LSTM | none | 10.22 | 0.12 / 0.51 / 0.19 / 0.18 | 1.24 | |
| human | LSTM | rollout gain, 0.1 | 10.09 | 0.13 / 0.49 / 0.34 / 0.05 | 0.070 | +94 % [+28 %, +98 %] |

**Control (E5):** the audit flags every learned model of the ACC data as string unstable at
81-100 % of the grid speeds in every fold (`configs/e5_expected.yaml`, `tests/test_e5_expected.py`:
MLP at least 0.9, GRU and LSTM at least 0.8, rule "each"). The IDM calibrated on the ACC data is
not flagged (unstable 0-12 %): an IDM cannot follow the delayed response of the controller, its
gains stay at 1.03. The 664 human events give models with larger errors (9.7-11.6 m) and no
usable steady state.

**H1.3** (reduction of the growth error by at least 30 %, interval without 0): **refuted on the
ACC profiles** (pooled reduction +8 % [-6 %, +22 %]: the unpenalised models already reproduce the
weak growth of the ACC platoons, errors of 0.09-0.12), **confirmed on the human profiles** for
GRU and LSTM (+98 % and +94 %) and pooled (+95 % [+72 %, +98 %]), because the unpenalised recurrent
models collide in the human platoons (growth errors 4.5 and 1.2) and the penalised ones do not
(0.07-0.08); for the MLP the penalty makes the growth error worse (-34 % [-91 %, +5 %]).
The verdict of H1.3 therefore rests on the collapse of the unconstrained recurrent models, not
on a better shape of the growth curve.

**Addendum of 2026-10-01 (D107).** The growth error of a platoon that collides was computed
from the speeds after the collision (standard deviations of 100-500 m/s); the "confirmed"
verdicts above rest on such numbers. With the rule of D107 (no growth error after a collision,
pairs with both values only, at least three pairs) every verdict of H1.3 is **open**: the
OpenACC-trained laws collide on 7-14 of their 10-15 profiles without and with penalty, and no
comparison has three pairs. The penalty removes the platoon collisions of the MLP on the ACC
profiles (7 of 10 without, 0 of 10 with; growth error -31 % over the 2 pairs) and changes
nothing for GRU and LSTM. `runs/_tables/m4/verdicts.md` carries the corrected verdicts.

## 9. Deviations from the specification

| Id | Deviation | Reason |
|---|---|---|
| D79, D80 | Equilibria are searched inside the spacing band of the data; the existence term demands braking at its lower and acceleration at its upper edge. | Penalties were met with equilibria at 100-190 m (M3). |
| D81 | `linear_gain` and `gain` take the maximum over the frequencies. | A mean over 25 frequencies hides one violated frequency. |
| D82 | Queue of processes instead of hydra multirun. | Reboots and a shared machine. |
| D85 | E2: four weights on seed 0, seeds 1-4 for the chosen weight. | 240 instead of 600 runs (accepted at the M3 review). |
| D86 | Certified hybrid with `r_max = 0.3 m/s^2` and core margin 0.2. | With `r_max = 1` no budget can be certified. |
| D87 | OpenACC models on views without the profile runs; H1.3 judged on them. | Leakage between the training events and the empirical curves. |
| D89 | Best epoch under a penalty: the feasible epoch with the smallest validation RMSE. | The RMSE alone returns epochs in which the penalty is not met. |
| D90 | Rollout penalty for the recurrent models on every 8th step, 16 speeds, replayed from a CUDA graph; the linearised penalty of M3 stays a negative result. | The linearised penalty is gamed by recurrent models (section 3). |
| D92 | H1.1 judged on the share of unstable equilibria among the equilibria found. | The band share would call a law without steady state stable. |
| D93 | Arm `e2_existence`. | Price of a steady state inside the band. |

## 10. Open points for the review

1. **Recurrent models in M5.** The penalised GRU and LSTM stay string unstable at 39-66 % of the
   speeds and collide on 4-6 of 6 platoon profiles. Proposal for the corridor: keep GRU, LSTM
   and PERL with and without penalty in the list of laws (as accepted at M3), expect collapses,
   and report them as results; the certified hybrid, the penalised MLP and the penalised
   ResidualIDM are the laws that can be compared with the IDM.
2. **A penalty that sees the low frequencies.** The rollout penalty of the specification is
   blind below 0.05 rad/s. A rollout of 250 s at 0.02-0.05 rad/s costs six times the present
   penalty step; with the CUDA graph that is about 2 s per penalty step, affordable on every 8th
   step. Proposal: one arm on fold 0 for GRU and LSTM after M5, not before, as it does not change
   the verdict of H1.2 (the accuracy cost of stability is already above the budget).
3. **Laws of the corridor.** The I-80 corridor has speeds of 0-15 m/s; the HighD models cover
   4.7-26.8 m/s. Proposal: the learned laws of the corridor are the E4 fine-tuned models on
   `ngsim_i80` (free and certified hybrid, MLP) plus GRU/LSTM trained on `ngsim_i80` (5 folds,
   seed 0, with and without penalty: 20 runs, about 3 hours), and the calibrated IDM of
   `ngsim_i80` (D49). The HighD models would be extrapolating for most of the simulation.
4. **Hysteresis.** The loop area of the pulse grows with the Jacobian penalty for MLP and PIDL
   (+27 %, +8 %) and shrinks for ResidualIDM (-24 %); for the recurrent models it explodes with
   the penalty because their platoons no longer collide before the pulse ends. Without an
   empirical reference the number describes the laws only. Proposal: report it, no hypothesis.

## 11. Tests

412 tests: 412 passed, 24 skipped (GPU variants, which pass with `CF_GPU_TESTS=1`). New in M4:
band of the training context and interpolation; equilibria inside the band, `outside`, several
crossings, speeds without band; existence term at the band edges and its gradient;
`aggregate: max`; kind `existence`; choice of the best epoch on prescribed sequences; penalty on
every n-th step and the graphs (mixed mode equals eager bit by bit); audit keys and band shares;
queue (expansion, templates, restart, failures, timeouts, `init_from`, dry run); platoon script,
hysteresis, anchored start; transfer script and horizon; certificate script and budget; views
and the extraction guard; fine-tuning; collection with missing files; statistics against
hand-computed values; tables with hand-made run trees and swapped verdict bases; Newell with a
gap that is not finite; expected outcome of E5.
