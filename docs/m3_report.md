# Milestone M3 report: stability module

Date: 2026-09-29 (runs of 2026-09-28). Contract: `docs/m3_contract.md`. Decisions:
`docs/decisions.md`, D63-D77 and E7. All stability computations in float64. GPU and CPU were
shared with three other projects during every run reported here; timings are as measured.
Nothing is committed to git.

## 1. Commands run

```bash
./.venv/Scripts/python.exe -m pytest                       # 306 tests: 300 passed, 6 skipped (146 s)

# trainings with a penalty (lambda = 1), reference fold and seed
python scripts/train.py data=follownet_highd fold=0 seed=0 model=mlp          train.penalty.kind=jacobian    train.penalty.weight=1.0 experiment=m3_jacobian
python scripts/train.py data=follownet_highd fold=0 seed=0 model=residual_idm train.penalty.kind=jacobian    train.penalty.weight=1.0 experiment=m3_jacobian
python scripts/train.py data=follownet_highd fold=0 seed=0 model=mlp          train.penalty.kind=linear_gain train.penalty.weight=1.0 experiment=m3_linear_gain
python scripts/train.py data=follownet_highd fold=0 seed=0 model=gru          train.penalty.kind=linear_gain train.penalty.weight=1.0 experiment=m3_linear_gain
python scripts/train.py data=follownet_highd fold=0 seed=0 model=lstm         train.penalty.kind=linear_gain train.penalty.weight=1.0 experiment=m3_linear_gain

# audit of the penalised models and of the reference models of M2 (stability.json next to every model.pt)
python scripts/audit_stability.py experiment=m3_jacobian    data=follownet_highd force=true
python scripts/audit_stability.py experiment=m3_linear_gain data=follownet_highd force=true
python scripts/audit_stability.py experiment=m2_acc0        data=follownet_highd force=true
python scripts/audit_stability.py experiment=m2             data=follownet_highd force=true
```

The five trainings were run twice: the first pass showed two ways around the penalties, the
penalties were corrected (D74, D76), and the numbers of section 3.3 are those of the second pass.
The platoon test, the certificate and the stability-constrained calibration were run through
their functions (`cf_stability.stability.platoon.platoon_test`, `certificate.*`,
`calibrate_global` with `stability_margin`); their entry into the experiment scripts is part of M4.

**Incident (E7).** The training of the penalised LSTM ended and wrote its weights and the frame
of the test events; the process then hung in a `git` child process and never wrote
`metrics.json`. Its metrics were rebuilt from the saved weights (the repeated evaluation equals
the stored frame of the test events exactly); the training history of this one run is lost.
`git_revision()` no longer starts a child process (D77).

## 2. What was built

| Part | Files | Content |
|---|---|---|
| Equilibria | `stability/equilibrium.py` | scan and bisection on the speed grid 5-30 m/s, status `ok / multiple / none / indifferent` |
| Linearisation | `stability/analytic.py` | partial derivatives (autograd, finite differences for kNN), Jacobian with respect to the history, criterion, transfer functions (continuous, discrete, with history window) |
| Frequency response | `stability/frequency.py` | two-vehicle rollouts, least-squares amplitude, 25 frequencies, flags for clipping, stops and collisions |
| Audit | `stability/audit.py`, `scripts/audit_stability.py` | `stability.json` per model: per equilibrium and summary, all speeds and speeds inside the data, shares over the grid speeds |
| Penalties | `stability/penalties.py`, trainer | Jacobian penalty, gain penalty by rollouts, linearised gain penalty, existence term; `train.penalty.*`; they run inside the CUDA-graph training step |
| Certificate | `stability/certificate.py` | guaranteed margin of ResidualIDM, a posteriori and a priori, admissible residual budget |
| Stable core | `models/idm.py`, `train/calibration.py` | closed-form IDM derivatives, global calibration with a lower bound on the string-stability margin |
| Platoon | `stability/platoon.py`, `configs/stability/platoon.yaml` | 50 followers stepped together, OpenACC leader profiles and braking pulse, growth error against the empirical curve |

## 3. Key numbers

### 3.1 Agreement of the analytic and the numerical test

Test of the specification (`tests/test_frequency.py`, `tests/test_audit.py`): for the calibrated
IDM the numerical flag equals the flag of the analytic gain at every equilibrium and the sign
criterion at every equilibrium that is not marginal; for an MLP trained on IDM samples the two
flags agree on more than 95 % of the equilibria and the largest gains within 2 %. For linear laws
the measured gain equals the exact discrete-time transfer function within 1e-3 at all 25
frequencies, also for a law with memory.

On the trained reference models (speeds inside the data) the numerical flag and the flag of the
analytic gain agree on 100 % of the equilibria for IDM, OVM, MLP, PIDL, ResidualIDM, GRU and PERL,
on 88 % for the LSTM (its locally unstable equilibria diverge in the rollout) and on 68 % for
kNN (piecewise constant law).

**The sign criterion and the rule "gain > 1.02" are different questions.** For the calibrated
IDM the sign criterion calls 15 of 25 equilibria unstable, the rule of the specification 4: the
other 11 amplify by 0.5-1.4 % only. Both numbers are reported everywhere (D65).

### 3.2 Audit of the reference models

FollowNet HighD, driver fold 0, seed 0. The 22 grid speeds inside the 1 % - 99 % speed range of
the training data (4.7-26.8 m/s). Shares of these speeds (D75): stable equilibrium / unstable
equilibrium / no equilibrium. "Numerical": unstable = largest measured gain above 1.02. Last
column: speeds at which the audited equilibrium lies inside the spacing band of the data
(section 3.6).

| Model | Loss | Test RMSE of the spacing (m) | Numerical: stable / unstable / none (%) | Sign criterion: stable / unstable / none (%) | Locally unstable (%) | Largest gain (rad/s; m/s) | Median of the largest gains | Equilibrium inside the band |
|---|---|---:|---|---|---:|---|---:|---:|
| IDM, calibrated | - | 3.13 | 82 / 18 / 0 | 32 / 68 / 0 | 0 | 1.040 (0.20; 5) | 1.005 | 21 of 22 |
| OVM, calibrated | - | 7.99 | 0 / 86 / 14 | 0 / 86 / 14 | 0 | 1.126 (0.93; 22) | 1.126 | 19 |
| Newell (PERL form) | - | 3.33 | every gap is an equilibrium | | | 1.006 (0.17; 21) | 1.003 | - |
| kNN | - | 3.08 | 14 / 86 / 0 | 9 / 91 / 0 | 14 | 3.04 (0.09; 6) | 1.27 | 13 |
| MLP | specification | 2.84 | 9 / 91 / 0 | 0 / 100 / 0 | 0 | 1.202 (0.04; 24) | 1.052 | 22 |
| PIDL | specification | 2.83 | 14 / 86 / 0 | 5 / 95 / 0 | 0 | 1.201 (0.08; 23) | 1.051 | 22 |
| ResidualIDM | specification | 3.58 | 50 / 50 / 0 | 9 / 91 / 0 | 0 | 1.161 (0.20; 5) | 1.022 | 12 |
| GRU | specification | 2.06 | 5 / 77 / 18 | 0 / 82 / 18 | 0 | 2.40 (0.05; 24) | 1.34 | 4 |
| LSTM | specification | 2.23 | 14 / 64 / 23 | 14 / 64 / 23 | 24 | 113 (diverges) | 1.26 | 12 |
| PERL | specification | 2.44 | 0 / 100 / 0 | 0 / 100 / 0 | 45 | 3.55 (0.09; 24) | 1.28 | 19 |
| MLP | rollout only | 2.54 | 50 / 50 / 0 | 5 / 95 / 0 | 0 | 1.036 (0.02; 22) | 1.021 | 22 |
| ResidualIDM | rollout only | 2.71 | 73 / 27 / 0 | 5 / 95 / 0 | 0 | 1.040 (0.14; 5) | 1.015 | 22 |
| GRU | rollout only | 1.96 | 27 / 73 / 0 | 5 / 95 / 0 | 0 | 6.85 (0.09; 25) | 1.055 | 12 |

The OVM is the known-unstable reference: unstable at every equilibrium, as its closed form
requires. Newell's law is string-neutral (gain 1.003-1.006, the excess is the discretisation).
The margins of the learned laws are small: -0.024 to -0.001 for the MLP against -0.115 to +0.021
for the IDM. Audit time per model: 2-25 s memoryless and hybrid, 90-109 s recurrent, 440 s kNN.

### 3.3 Penalties

Gradient check of the specification (`tests/test_penalties.py`): the gradients of the three
penalties with respect to every parameter equal central finite differences (relative error below
1e-4; MLP for the Jacobian penalty, GRU for the two gain penalties).

Cost per training step on the shared GPU, without penalty / Jacobian / linearised gain / gain by
rollouts: MLP 38 / 48 / 55 / 576 ms, ResidualIDM 105 / 165 / 147 / 3 037 ms, GRU 66 / 137 /
170 / 6 823 ms. With the first two penalties the step runs in the CUDA graph and gives bit by bit
the weights of the eager step.

Effect in a real training (lambda = 1, reference fold and seed, second pass). Shares over the 22
grid speeds inside the data, as in section 3.2. The reference of ResidualIDM is the rollout-only
training (decision of the M2 review).

| Model | Penalty | Epochs (best) | Test RMSE of the spacing, mean / median (m) | Change of the mean | Collisions (%) | Numerical: stable / unstable / none (%) | Sign criterion: stable / unstable / none (%) | Largest gain (rad/s; m/s) | Equilibrium inside the band | Penalty at the best epoch | s per epoch |
|---|---|---|---:|---:|---:|---|---|---|---:|---:|---:|
| MLP | none | 21 (11) | 2.84 / 2.20 | | 0.1 | 9 / 91 / 0 | 0 / 100 / 0 | 1.202 (0.04; 24) | 22 of 22 | - | 9.3 |
| MLP | Jacobian | 29 (19) | 2.80 / 2.12 | -1.3 % | 0.2 | 100 / 0 / 0 | 100 / 0 / 0 | 0.995 (0.02; 11) | 20 | 0.0007 | 16.4 |
| MLP | linearised gain | 52 (42) | 2.81 / 2.14 | -1.2 % | 0.0 | 91 / 9 / 0 | 77 / 23 / 0 | 1.025 (0.02; 21) | 18 | 0.0030 | 17.8 |
| ResidualIDM | none | 45 (35) | 2.71 / 2.01 | | 0.0 | 73 / 27 / 0 | 5 / 95 / 0 | 1.040 (0.14; 5) | 22 | - | 21.1 |
| ResidualIDM | Jacobian | 22 (12) | 3.89 / 3.34 | +43.2 % | 0.0 | 100 / 0 / 0 | 100 / 0 / 0 | 1.000 (0.02; 5) | 11 | 0.3828 | 52.8 |
| GRU | none | 21 (11) | 2.06 / 1.54 | | 0.0 | 5 / 77 / 18 | 0 / 82 / 18 | 2.40 (0.05; 24) | 4 | - | 12.1 |
| GRU | linearised gain | 48 (38) | 2.06 / 1.55 | -0.2 % | 0.0 | 86 / 14 / 0 | 50 / 50 / 0 | 1.354 (0.02; 18) | 6 | 0.0028 | 37.9 |
| LSTM | none | 31 (21) | 2.23 / 1.73 | | 0.0 | 14 / 64 / 23 | 14 / 64 / 23 | 113 (diverges) | 12 | - | 12.2 |
| LSTM | linearised gain | lost (E7) | 2.57 / 2.01 | +15.2 % | 0.0 | 86 / 14 / 0 | 86 / 14 / 0 | 1.951 (0.20; 22) | 13 | 0.0003 (saved weights, 26 grid speeds) | 27 min in total |

Three ways around a penalty that is evaluated at the equilibria only were found:

1. **Losing the equilibria** (first pass): the penalised GRU kept an equilibrium at 2 of 26
   speeds, the LSTM at 1, with penalties of 0.0002. Closed by the existence term (D74): in the
   second pass every model has an equilibrium at every speed inside the data.
2. **Moving the amplification below the lowest penalised frequency** (first pass): the MLP had a
   gain of 1.354 at 0.02 rad/s. Closed by the frequencies of the audit with a margin that
   shrinks with the square of the frequency (D76): 1.025 in the second pass.
3. **Moving the equilibria out of the data** (second pass, open): see section 3.6. The penalised
   GRU has its equilibrium at 145 m for 20 m/s (time gap 7 s), the LSTM at 139 m; the data hold
   13-61 m there.

Further observations:

* The Jacobian penalty with the margin 0.5 of the specification is met by a steep reaction to
  the relative speed: `f_dv` of the penalised MLP is -2.6 to -6.7 1/s (median -6.2), against
  -0.13 to -0.38 of the reference MLP and -0.45 to -0.97 of the calibrated IDM.
* ResidualIDM (fixed core, `r_max = 1`) reaches margins of +0.07 to +0.28 at every speed, not
  the margin 0.5 of the penalty, which stays at 0.38. The bounded residual is spent on the
  derivatives: the open-loop acceleration error rises to 1.38 m/s^2 and the spacing RMSE by
  43 %, above the calibrated IDM (3.13 m) and as much as a stable core costs (section 3.4).
* The linearised gain penalty is a mean over speeds and frequencies: one speed with a gain of
  1.35 at one frequency contributes 0.001. The speed 18 m/s of the GRU is such a case (second
  equilibrium at 7.3 m).
* The LSTM has two equilibria at 22-24 m/s; the rollout of the frequency test leaves the audited
  one (gain 1.07-1.95 measured, 0.93-0.95 by the linearisation). These three speeds are the
  14 % "unstable" and the disagreement of the two tests (86 % agreement).

### 3.4 Certificate of ResidualIDM and the stable core

Reference ResidualIDM (`r_max * lipschitz = 1`, exact layer norms 1.0000-1.0002): bounds of the
residual derivatives `B = (0.050, 0.500, 0.100)` for `(s, dv, v)`. The empirical margin of the
hybrid is never below the guaranteed margin (soundness, also tested after adversarial training).

**The certificate holds at 0 of 26 speeds**, in both forms. Reason: the calibrated IDM core is
string unstable itself below 19 m/s (margin -0.12 at 5 m/s) and marginal above (at most +0.02),
so the admissible budget `r_max * lipschitz` is 0 up to 18 m/s and 0.01-0.21 above (a priori:
at most 0.03).

Global IDM calibration with a lower bound on the margin at every grid speed (FollowNet HighD,
fold 0, test part, rollouts of whole events):

| Lower bound of the margin | v0 | T | s0 | a | b | Spacing RMSE, mean / median (m) | Against the free fit |
|---|---:|---:|---:|---:|---:|---:|---:|
| none (free fit) | 29.6 | 0.68 | 2.02 | 0.35 | 0.50 | 3.81 / 2.89 | - |
| 0 | 30.0 | 0.94 | 0.50 | 0.51 | 0.50 | 4.12 / 3.27 | +8 % |
| 0.05 | 33.0 | 0.98 | 0.50 | 0.71 | 0.50 | 4.52 / 3.56 | +19 % |
| 0.2 | 35.2 | 0.98 | 3.19 | 1.57 | 0.50 | 5.41 / 4.34 | +42 % |
| 0.5 | 35.1 | 0.94 | 4.71 | 2.74 | 0.50 | 6.05 / 4.94 | +59 % |

A certified hybrid therefore needs a core with margin and a small residual budget; whether the
residual wins back the accuracy the constraint costs is the question of E4.

### 3.5 Platoon growth test

Leader + 50 identical followers, start at the time gap `2 m + 1.5 s * v`. Leader profiles: five
OpenACC runs and a braking pulse. Empirical standard deviation of the speed by platoon position
(leader first):

| Profile | Drivers | Length | Leader speed, 5 % / median / 95 % (m/s) | Empirical curve (m/s) |
|---|---|---:|---|---|
| ZalaZone handling_part30 | human | 206 s | 6.8 / 7.9 / 8.4, start from rest | 1.38, 1.56, 1.70, 1.86, 2.00, 2.08, 2.22, -, 2.41 |
| Vicolungo JRC-VC_280219_part4_highway | human | 556 s | 19.4 / 30.0 / 35.8 | 5.17, 5.21, 5.31, 5.60, 5.59 |
| Vicolungo JRC-VC_280219_part2 | human | 374 s | 22.7 / 31.2 / 35.0 | 4.13, 4.31, 4.35, 4.05, 3.90 |
| ZalaZone handling_part32 | ACC | 510 s | 0.0 / 7.9 / 8.3, stop at the end | 2.23, 2.29, 2.34, 2.44, 2.44, 2.52, 2.61, 3.09, 3.18, 3.28 |
| ZalaZone handling_part13 | ACC | 519 s | 0.0 / 7.9 / 8.5, stop at the end | 2.41, 2.43, 2.47, 2.47, 2.49, 2.47, 2.45, 2.43, 2.40, 2.49 |

The values of the 25-car experiment of Jiang et al. are not public (`configs/empirical_growth.yaml`
has the place for them).

**The profiles leave the speed range of the training data** (4.7-26.8 m/s): the ZalaZone leaders
stand or drive below 5 m/s for 4-11 % of the time, the Vicolungo leaders drive above 26.8 m/s for
66-75 %. The models of this report are trained on FollowNet HighD, so the platoon test also
measures their extrapolation.

Model platoons, standard deviation of the speed at positions 1 / 5 / 10 / 25 / 50 on the human
ZalaZone profile and behind the braking pulse, collisions over all six profiles, growth error
(range over the five OpenACC profiles):

| Model | Loss, penalty | Speed std at 1 / 5 / 10 / 25 / 50, ZalaZone human (m/s) | The same, braking pulse | Profiles with a collision | Growth error |
|---|---|---|---|---:|---:|
| IDM, calibrated | - | 2.1 / 2.9 / 3.5 / 4.9 / 6.6 | 1.1 / 0.8 / 0.7 / 1.3 / 1.5 | 0 of 6 | 0.07-0.74 |
| MLP | specification | 1.7 / 2.6 / 3.5 / 5.9 / 3.1 | 1.0 / 1.0 / 1.0 / 1.0 / 0.5 | 1 of 6 (vehicle 19) | 0.04-0.18 |
| MLP | rollout only | 1.6 / 2.2 / 2.8 / 4.0 / 4.9 | 1.0 / 0.8 / 0.6 / 0.4 / 0.9 | 0 of 6 | 0.02-0.14 |
| MLP | Jacobian penalty | 1.7 / 2.3 / 2.7 / 3.1 / 2.4 | 1.0 / 0.6 / 0.3 / 0.4 / 0.6 | 2 of 6 (first follower, final stop of the ACC leaders) | 0.01-0.30 |
| MLP | linearised gain penalty | 1.7 / 2.4 / 3.0 / 4.0 / 4.2 | 1.0 / 1.0 / 1.0 / 1.4 / 1.6 | 0 of 6 | 0.02-0.20 |
| ResidualIDM | rollout only | 1.7 / 2.8 / 3.7 / 5.8 / 4.7 | 1.1 / 0.9 / 0.8 / 1.4 / 2.3 | 0 of 6 | 0.06-0.54 |
| ResidualIDM | Jacobian penalty | 1.5 / 1.7 / 2.0 / 2.8 / 3.9 | 0.9 / 0.6 / 0.4 / 0.1 / 0.1 | 0 of 6 | 0.02-0.40 |
| GRU | specification | collision | 1.1 / 1.6 / 2.2 / 10.2 / 10.8 | 6 of 6 (one of the first three followers after 13-160 s) | 5.8-15.1 |
| GRU | linearised gain penalty | 1.9 / 2.8 / 3.7 / 5.6 / 5.9 | 1.2 / 2.8 / 3.2 / 4.6 / 5.0 | 4 of 6 (vehicles 4-32 after 100-441 s) | 0.07-0.32 |
| LSTM | specification | collision | 1.0 / 1.4 / 2.6 / 7.6 / 7.4 | 5 of 6 (first follower after 31-192 s) | 3.3-7.7 |
| LSTM | linearised gain penalty | 1.6 / 2.0 / 2.3 / 2.5 / 1.9 | 1.1 / 1.8 / 2.5 / 3.8 / 5.5 | 5 of 6 (vehicles 23-39 after 76-97 s; first follower at the final stop) | 0.04-2.15 |

Six profiles take 2-7 s (MLP, IDM), 3-19 s (ResidualIDM), 15-63 s (GRU, LSTM), depending on the
load of the GPU.

### 3.6 Where the equilibria lie

Spacing band of the data: 5 % and 95 % quantile of the spacing of the near-steady training
samples (`|dv| < 0.5 m/s`, `|a| < 0.3 m/s^2`, 40 % of the samples) within 0.5 m/s of the grid
speed. Examples: 4.5-18.9 m at 5 m/s, 7.8-35.9 m at 10 m/s, 13.0-61.4 m at 20 m/s (median
27.9 m, time gap 1.4 s), 11.4-37.5 m at 25 m/s.

Shares of the 22 grid speeds (numerical rule), computed from `stability.json` and the band:

| Model | Loss, penalty | Stable, inside the band (%) | Unstable, inside the band (%) | Equilibrium outside the band (%) | No equilibrium (%) |
|---|---|---:|---:|---:|---:|
| IDM, calibrated | - | 77 | 18 | 5 | 0 |
| MLP | specification | 9 | 91 | 0 | 0 |
| PIDL | specification | 14 | 86 | 0 | 0 |
| GRU | specification | 0 | 18 | 64 | 18 |
| LSTM | specification | 14 | 41 | 23 | 23 |
| PERL | specification | 0 | 86 | 14 | 0 |
| MLP | rollout only | 50 | 50 | 0 | 0 |
| ResidualIDM | rollout only | 73 | 27 | 0 | 0 |
| GRU | rollout only | 18 | 36 | 45 | 0 |
| MLP | Jacobian penalty | 91 | 0 | 9 | 0 |
| MLP | linearised gain penalty | 73 | 9 | 18 | 0 |
| ResidualIDM | Jacobian penalty | 50 | 0 | 50 | 0 |
| GRU | linearised gain penalty | 27 | 0 | 73 | 0 |
| LSTM | linearised gain penalty | 50 | 9 | 41 | 0 |

The audit analyses the first equilibrium it finds from small gaps. A scan of all equilibria
shows that this choice hardly matters: another equilibrium lies inside the band at 1 speed of
the MLP with the Jacobian penalty, at 8 speeds of kNN and at no speed of the other models.

Static slope of the acceleration in the gap at the median of the band (mean over the speeds):
IDM 0.0148, MLP 0.0094, PIDL 0.0099 1/s^2; GRU 0.0005, LSTM 0.0009, PERL 0.0004 1/s^2. In steady
state the recurrent models hardly react to the gap.

## 4. Reading of the numbers

1. **Premise of H1.1.** On this fold and seed the unconstrained networks are more accurate than
   the IDM (M2) and string unstable at 64-91 % of the grid speeds (MLP 91 %, GRU 77 %, LSTM
   64 %); GRU and LSTM have no equilibrium at a further 18-23 %. The calibrated IDM is unstable
   at 18 % with gains up to 1.04.
2. **The size of the gain matters more than the share.** MLP and PIDL amplify by 5 % in the
   median, the recurrent models by 26-34 %, with peaks of 2.4 and more.
3. **The training loss changes the stability.** Without the one-step acceleration term the MLP is
   unstable at 50 % instead of 91 % of the speeds and its largest gain falls from 1.20 to 1.04.
4. **GRU and LSTM have no usable steady state**, with and without penalty: an equilibrium
   inside the band of the data at 4-13 of 22 speeds, a static reaction to the gap 10-30 times
   weaker than that of the IDM and the MLP. On the 12 s events of the closed-loop evaluation
   they had no collision at all; behind a real leader the reference GRU and LSTM collide within
   minutes.
5. **The penalties work for the MLP.** Jacobian penalty: stable at every grid speed, accuracy
   unchanged (-1.3 %), a braking pulse decays along the platoon. H1.2 holds on this fold, with
   two reservations: the equilibria at 24 and 26 m/s lie outside the band, and the first
   follower runs into the ACC leader that brakes to a stop (speeds below the data).
6. **For the recurrent models the penalties repair the numbers more than the behaviour.** The
   penalised GRU keeps its accuracy and no longer collides at once, but 73 % of its equilibria
   lie outside the data and it still collides on 4 of 6 profiles. The penalised LSTM loses
   15 % of accuracy, more than H1.2 allows.
7. **Damping is not the same as reproducing the empirical growth.** The human and ACC platoons
   of OpenACC amplify themselves; for the MLP the growth error does not fall with the penalties
   (mean over the five profiles 0.10 without, 0.14 and 0.11 with). It falls where the reference
   collapses (GRU 5.8-15.1 to 0.07-0.32). H1.3 should be read with this in mind.
8. **The certificate is sound but empty with the calibrated core**, because the core itself is
   not string stable; the Jacobian penalty repairs it through the bounded residual only at a
   loss of 43 % of accuracy.
9. **The gain penalty of the specification is too expensive for recurrent models** (6.8 s per
   step for the GRU) and biased at low frequency; the linearised gain penalty costs 0.17 s.

## 5. Deviations from the specification

| Id | Deviation | Reason |
|---|---|---|
| D64 | Frequencies 0.02-2.0 rad/s log-spaced; rollout length per frequency (discard max(60 s, 1 period), measure whole periods, at least max(60 s, 2 periods)); least-squares amplitude instead of an FFT line. | 60 s hold a fifth of the period at 0.02 rad/s; transients of 30 s time constant. |
| D65 | Three flags: numerical, analytic gain, sign criterion; the agreement test compares the first two. | The sign criterion and the 1.02 rule differ on marginal equilibria by construction. |
| D66 | Recurrent models are analysed with the linearisation that keeps the history window. | The memoryless view overstates the gain (8.7 against 2.4 for the GRU). |
| D68 | kNN derivatives by finite differences. | Piecewise constant law. |
| D70, D76 | Additional linearised gain penalty, on the 25 frequencies of the audit with a margin that shrinks below 0.1 rad/s. | Cost and bias of the rollout penalty; amplification below the lowest penalised frequency. |
| D71, D72 | Certificate in two forms; stability-constrained calibration of the IDM core. | Survival of fine-tuning; a certificate needs a stable core. |
| D73 | Platoon starts at a time gap, not at the equilibrium of the model. | Unphysical equilibria of learned laws. |
| D74 | Existence term in every penalty (`existence_weight = 0` restores the specification). | Penalties were met by losing the equilibria. |
| D75 | Primary stability numbers are shares of the grid speeds: stable / unstable / no equilibrium. | The share among the equilibria found rewards a law for losing equilibria. |
| D76 | With a penalty the initial weights are not eligible as the best epoch. | Epoch 0 of ResidualIDM is the unpenalised IDM. |
| D77 | The git revision is read from the files of `.git`. | A hanging `git` child blocked a finished training run. |

## 6. Open points for the review

1. **Equilibria outside the data (new).** Proposal: (a) the spacing band of section 3.6 becomes
   part of the training context; (b) the existence term demands braking at the lower edge of
   the band and acceleration at its upper edge, at every sampled speed, instead of 1 m and
   200 m; (c) the audit analyses the equilibrium inside the band and reports the shares stable /
   unstable / no equilibrium inside the band; (d) H1.2 is judged on the share of speeds that
   are not "stable inside the band". A third pass of the five trainings on the reference fold
   (about 1.7 hours) checks it before E2 starts.
2. **Penalty for the recurrent models in E2.** Proposal: linearised gain penalty for GRU, LSTM
   and PERL in the full experiment; the gain penalty of the specification on one fold and one
   seed of the GRU for comparison (about 13 hours).
3. **Core of the certified hybrid in E4.** Proposal: IDM core calibrated with margin 0.2 and
   residual budget `r_max * lipschitz` chosen so that the a priori certificate holds on the whole
   grid; the free core stays as the uncertified comparison.
4. **Compute of E1 and E2.** E1 (8 trainable configurations, 5 folds x 5 seeds, audits and
   platoon test): about 26 hours sequentially. E2 with the measured costs (MLP 8-15 min,
   ResidualIDM 19 min, GRU 30 min, LSTM 27 min per run at lambda = 1): six architectures x four
   weights x 25 runs = 600 runs, about 210 hours sequentially, 70 hours with three runs in
   parallel. Proposal: all four weights on the five folds with seed 0, the seeds 1-4 only for
   the weight chosen on the validation parts: 240 runs, about 84 hours sequentially, 28 hours
   with three in parallel.
5. **Growth error across data sets.** The empirical curves come from OpenACC, the models from
   HighD, and the leader profiles leave the speed range of HighD. Proposal: H1.3 is judged on
   models trained on OpenACC (they are needed for E5 anyway); the HighD models are reported
   next to them.
6. **Recurrent models in the corridor (M5).** They will collide in a simulation of 15 minutes.
   Proposal: keep them in the list of laws, report the collapse as a result, and let the
   penalised versions show whether the penalty repairs it.

## 7. Tests

306 tests: 300 passed, 6 skipped (the GPU variants of the platoon and CUDA-graph tests, which run
with `CF_GPU_TESTS=1`). New in M3:

* equilibria: IDM and OVM against their closed forms, `none` above the desired speed,
  `indifferent` and `multiple` laws, windowed models, float32 and float64;
* linearisation: IDM derivatives against the closed form, OVM margin `-1 / tau^2`, derivatives
  differentiable with respect to the parameters (MLP, GRU), finite differences against autograd,
  the band of amplification `0 < w < sqrt(-M)`, discrete against continuous transfer function,
  Jacobian of the history against the summed derivatives;
* frequency response: exact agreement with the discrete-time transfer function for linear laws
  with and without memory, recovery of amplitude, phase and offset, **the agreement test of the
  specification for the calibrated IDM and for an MLP**;
* audit: documented keys, JSON output, counts and grid shares of the summary, script end to end;
* penalties: **gradients against finite differences for the three penalties**, zero for a law
  with margin, positive for the OVM, linearised penalty against the transfer function, existence
  term, training with penalty ends with a larger margin, the initial weights are not chosen
  under a penalty, reproducibility, identical weights with and without CUDA graph;
* certificate: guaranteed margin against a brute-force minimum over the box, a priori never above
  a posteriori, soundness after adversarial training and after further training on other data;
* stable core: closed-form IDM margin against autograd, the constrained calibration keeps the margin;
* platoon: stable law does not grow and unstable law grows, vehicle `n` equals a two-vehicle
  rollout behind vehicle `n - 1`, batched equals sequential, growth error on hand-made curves;
* helpers: git revision from the files of a repository, of a packed reference, of a work tree.
