# Milestone M2 report: models and closed-loop evaluation

Date: 2026-09-28. Contract: `docs/m2_contract.md`. Decisions: `docs/decisions.md`, D51-D62 and E6.
Environment: `.venv` (Python 3.11.8, torch 2.11.0+cu128), now shared by the four sibling
projects. GPU RTX 6000 Ada, shared with the other sessions during every run reported here.
Nothing is committed to git.

## 1. Commands run

```powershell
.\.venv\Scripts\python.exe -m pytest                       # 231 passed (169 s on CPU)

# reference runs: every model, FollowNet HighD, driver split, fold 0, seed 0
powershell -ExecutionPolicy Bypass -File scripts\run_m2.ps1
#   = for each <model>:
.\.venv\Scripts\python.exe scripts\train.py data=follownet_highd model=<model> fold=0 seed=0 experiment=m2

# weight of the one-step acceleration term
.\.venv\Scripts\python.exe scripts\train.py data=follownet_highd model=residual_idm experiment=m2_acc0 train.acc_weight=0.0
.\.venv\Scripts\python.exe scripts\train.py data=follownet_highd model=residual_idm experiment=m2_acc01 train.acc_weight=0.1
.\.venv\Scripts\python.exe scripts\train.py data=follownet_highd model=mlp experiment=m2_acc0 train.acc_weight=0.0
.\.venv\Scripts\python.exe scripts\train.py data=follownet_highd model=gru experiment=m2_acc0 train.acc_weight=0.0
.\.venv\Scripts\python.exe scripts\train.py data=follownet_highd model=residual_idm experiment=m2_epoch0
```

Outputs: `runs/<experiment>/follownet_highd/<model>/driver_fold0_seed0/{metrics.json, model.pt,
test_events.parquet}`.

## 2. What was built

| Part | Files | Content |
|---|---|---|
| Models | `cf_stability/models/` | interface with internal input scaling, IDM (plain and bounded), OVM, Newell (PERL form), persistence, kNN, MLP, GRU, LSTM, PIDL, PERL, ResidualIDM, factory and checkpoints |
| Training | `cf_stability/train/{tensors,trainer}.py` | event tensors, acceleration loss + rollout loss + model loss, early stopping on the closed-loop validation error, seeds, CUDA-graph step |
| Evaluation | `cf_stability/train/{closed_loop,evaluate}.py` | full-event closed-loop rollouts, per-event and summary metrics, one-step acceleration error |
| Entry point | `scripts/train.py`, `configs/train.yaml`, `configs/model/*.yaml` | one run = one model, one fold, one seed; hydra multirun |

Code 4 699 lines, tests 3 001 lines in 27 files.

## 3. Key numbers

FollowNet HighD, 12 512 events of 15 s; driver split, fold 0: 7 506 / 2 503 / 2 503 events
(training / validation / test); seed 0. Every model gets the first 3 s of an event as observed
history and is rolled from sample 29 to the end (12 s, 121 samples scored).

| Model | Trainable parameters | Epochs (best) | Spacing RMSE, mean / median (m) | Speed RMSE, mean / median (m/s) | Collisions | One-step acceleration RMSE (m/s^2) | Time (s) |
|---|---:|---|---:|---:|---:|---:|---:|
| persistence | 0 | - | 5.62 / 3.71 | 1.42 / 0.97 | 7.4 % | 0.013 | 2 |
| OVM, global | 0 | - | 7.99 / 6.18 | 1.52 / 1.25 | 0.0 % | 8.920 | 25 |
| Newell (PERL form), global | 0 | - | 3.33 / 2.36 | 0.69 / 0.53 | 2.3 % | 0.252 | 2 |
| IDM, global | 0 | - | 3.13 / 2.42 | 0.62 / 0.50 | 0.0 % | 1.119 | 61 |
| kNN | 0 | - | 3.08 / 2.35 | 0.64 / 0.51 | 0.5 % | 0.313 | 42 |
| MLP | 4 481 | 21 (11) | 2.84 / 2.20 | 0.58 / 0.47 | 0.1 % | 0.292 | 198 |
| PIDL | 4 481 | 23 (13) | 2.83 / 2.15 | 0.58 / 0.47 | 0.0 % | 0.508 | 249 |
| GRU | 13 313 | 21 (11) | 2.06 / 1.54 | 0.48 / 0.38 | 0.0 % | 0.056 | 259 |
| LSTM | 17 729 | 31 (21) | 2.23 / 1.73 | 0.50 / 0.42 | 0.0 % | 0.077 | 399 |
| PERL | 17 729 | 29 (19) | 2.44 / 1.77 | 0.55 / 0.42 | 0.0 % | 0.112 | 663 |
| ResidualIDM | 4 481 | 29 (19) | 3.58 / 2.63 | 0.74 / 0.56 | 0.0 % | 0.965 | 723 |

Test part; validation errors differ by less than 0.1 m. Fitted on the training part: IDM
`v0 = 29.6, T = 0.68, s0 = 2.0, a = 0.35, b = 0.50`; OVM `v0 = 23.9, tau = 0.71, s0 = 5.9`;
Newell wave speed `w = 12.4` m/s.

Weight of the one-step acceleration term in the loss (rollout weight 1 in all runs):

| Model | Weight 1 (specification) | Weight 0.1 | Weight 0 (rollout loss only) |
|---|---:|---:|---:|
| MLP | 2.84 | - | 2.54 |
| GRU | 2.06 | - | 1.96 |
| ResidualIDM | 3.58 | 3.07 | 2.71 |
| ResidualIDM with epoch 0 in the selection | 3.13 (= IDM, best epoch 0) | - | - |

Spacing RMSE, mean over the test events, m.

## 4. Reading of the numbers

1. **The one-step acceleration error does not rank models.** Persistence has the smallest
   one-step error (0.013 m/s^2) and the second-worst closed-loop error; the IDM has the largest
   one-step error among the usable models (1.12 m/s^2) and no collision. Only closed-loop
   metrics are reported as results.
2. **Learned models beat the global IDM in closed loop**, recurrent ones by a third (GRU 2.06 m
   against 3.13 m). This is the premise of hypothesis H1.1; whether they are string stable is
   the question of M3.
3. **The recurrent models use the autocorrelation of the acceleration** (one-step error 0.06-0.08
   m/s^2, close to persistence), the memoryless ones cannot (0.29 m/s^2). The stability analysis
   has to treat the two groups differently: constant-history equilibrium for the analytic
   criterion, frequency response for the rest.
4. **ResidualIDM trained with the loss of the specification is worse than its own IDM.** The
   acceleration term (1.37 (m/s^2)^2) dominates the loss; the residual learns to cancel the strong
   reaction of the IDM to the speed difference, which is what makes the IDM accurate in closed
   loop. Trained on the rollout loss alone it is the best memoryless model (2.71 m, 13 % below
   the IDM).
5. **PIDL gains nothing over the MLP** in closed loop (2.83 m against 2.84 m): the physics term
   pulls the network towards the same high-gain IDM (one-step error 0.51 against 0.29 m/s^2).
6. **The Newell law of PERL has no feedback on the gap**: alone it collides in 2.3 % of the
   12 s rollouts; in PERL the LSTM residual provides the regulation (no collision, 2.44 m).

## 5. Deviations from the specification and decisions

| Id | Item | Reason |
|---|---|---|
| D22 | Training set is FollowNet HighD, not raw highD. | No raw highD on the machine; none of the sessions can download it. |
| D52, D58 | "Newell" is the adapted Newell model of the PERL paper (delayed leader acceleration, wave speed `w`); the relaxation form first implemented is kept as the extra model `ovm`. | The relaxation form is string unstable at every congested equilibrium by construction and is not the physics of PERL. |
| D53 | The IDM acceleration in the physics term of PIDL is clipped to [-8, 4] m/s^2. | Unclipped it reaches -72 m/s^2 in the corners of the collocation box. |
| D55 | The residual of ResidualIDM starts at zero through an antisymmetric initialisation. | Spectral normalisation removes the scale of the weights. |
| D56 | kNN uses at most 200 000 training samples. | Cost of the search in closed loop. |
| D57 | All models are scored from sample 29 (3 s of observed history). | Same samples for all windows. |
| D59 | Rollout loss normalised by the variances of spacing and speed, weight 1; epoch capped at 300 steps; CUDA-graph training step. | Left open by the specification. |
| D61 | Initial weights take part in the model selection (epoch 0). | A hybrid must not end worse than its physics. |
| E6 | One Python environment for the four projects. | User's instruction. |

## 6. Open points for the review

1. **Training loss of the experiments (D62).** Options: (a) loss of the specification for all
   models; (b) loss of the specification for the audited architectures and rollout loss only for
   ResidualIDM, the proposed model; (c) rollout loss only for all. Recommendation: (b), plus the
   weight-0 runs of MLP, GRU and LSTM as a sensitivity analysis on one fold, because the stability
   of a learned model may depend on the loss and the audit should say so.
2. **Compute for E1.** One fold and seed of the eight trainable configurations takes about
   45 minutes on the shared GPU; 5 folds x 5 seeds are about 19 hours sequentially, about 6 hours
   with three runs in parallel. E2 multiplies this by the four penalty weights.
3. **Short events.** HighD events are 15 s: 3 s of history and 12 s of rollout. Collision rates
   and spacing errors on 12 s understate what happens in long platoon runs; the platoon test of
   M3 and the corridor of M5 are the long-horizon checks.
4. **Heterogeneous IDM for the corridor.** With `v0` and `b` fixed, 37-42 % of the per-event
   estimates of `T, s0, a` are inside the bounds; proposal: sample the vehicles of M5 from these
   interior estimates only (docs/m1_report.md, section 6a).

## 7. Tests

231 tests. New in M2: model interface and checkpoints for the eleven models, scaler inside the
model, bounded IDM, OVM and Newell formulas and calibration, generic global calibration, kNN
against numpy, PIDL physics term, PERL initialisation, ResidualIDM bound of the Jacobian against
autograd at 2 000 states after adversarial training steps, event tensors, closed-loop evaluation
(generating IDM reproduces its events, scoring starts at sample 29, batching invariance, collision
flag), trainer (reproducibility per seed, early stopping, rollout loss gradients, model loss term),
training script end to end.
