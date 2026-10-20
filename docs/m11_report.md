# Milestone M11 report: extensions after the submission (H-N1 to H-N5, A1 to A3)

Contract: `docs/m11_contract.md` (D127-D133). Every agent writes its own section; the main session assembles the
verdicts and the decision log.

## Verdicts (main session, 20 October 2026)

Six pre-specified rules, none met; three results that are useful anyway, two design flaws found and diagnosed, one
negative result that closes a line of thought. The manuscript (paper/v2, submitted 9 October) is not touched; the
corridor tables of `runs/_tables/m5` and `m8` carry the five new laws as further rows (the correlations and the
verdicts of H12 are unchanged); the state of the manuscript's tables is the tag v2.0.2.

| Item | Rule (contract) | Numbers | Verdict |
|---|---|---|---|
| H-N1 predictive validity (D127, section C.1) | a single-vehicle predictor beats the share of unstable equilibria (paired interval of rho_P - rho_(a) above 0) and ranks E better than D | rho_(a) 0.755 [0.527, 0.863]; closest predictor the share not stable in the band, +0.012 [-0.016, +0.072]; the frozen gain along trajectories -0.049 [-0.294, +0.187] (share) and -0.443 [-0.953, -0.021] (mean log), and it ranks D better than E (0.144 / 0.081 against 0.230 / 0.164); only the contact rate ranks E better than D, at +0.001 [-0.296, +0.253] | **refuted**: the pre-specified statistic is as good a predictor of the corridor error as anything linearised we tried; what the derivative budget buys in the corridor (the ablation E - D) is not visible in the laws' linearisations along equilibria or along trajectories |
| H-N3 windowed-gain penalty (D128, section A) | GRU and LSTM stable share >= 0.9 at <= +10 % RMSE and the long-window check holds | GRU 0.09 at +17.3 % (check fails at the chosen weight 0.1, holds at 1 and 10); LSTM 0.66 at +36.2 % (holds); PERL 0.00 | **refuted**, with the cause diagnosed: the gain term is the mean over 25 frequencies of the squared excess, so a broad amplification of 1.1 near 1 rad/s costs 0.002, below the feasibility tolerance 0.01 (the weakness D81 found for `linear_gain`). What the penalty does: it closes the low-frequency blind band of the rollout penalties (M_w >= 0.5, poles inside, the 380-s check holds at weights 1 and 10), raises the stable share of the fine-tuned corridor members (GRU 0.63 against 0.27, LSTM 0.91 against 0.25 at +1.5 % and +5.2 % RMSE on I-80) and gives the best GRU law of the study in the corridor: macro error 0.132 on I-80 (free GRU 0.192, rollout penalty 0.235; the certified hybrid 0.125) and 0.260 on US-101 (0.421, 0.347), every vehicle inserted, but still 270-300 contacts per 1 000 vehicle-km |
| H-N2 two-timescale certified law (D129, section B.2) | RMSE <= 1.05 x GRU, 100 % stable equilibria at every frozen z, certificate at all 21 z after fine-tuning | RMSE 4.218 m = 2.02 x GRU (0.935 x the certified hybrid E4); frozen-z audits 100 % stable on every fold before and after fine-tuning; exact certificate at 13 of 21 z (z >= -0.2) and on the whole reachable range of the latent; corridor: best of 26 laws on US-101 (0.090 against E 0.113, -20 % [-25, -14]), no difference on I-80 (-0.007 [-0.027, +0.012]) | **refuted** by the accuracy criterion, for a reason the contract built in: the latent restarts at z = 0 with every 30-state window, so it can never leave [-0.18, +0.18] (T within +-7.7 %) at tau_z = 15 s; the "slow" variable is a window feature, not a state carried along the event. Even so the law is 6.5 % more accurate than E4 on highD and the best law on the out-of-sample corridor; the budget had to be enforced over the reachable range because the margin core cannot be certified below T = 0.92 T_c (margin 0 at T_c / 1.5) |
| A3 robust certificate (D130, section C.2) | E certified string stable on the band, D not | band, box: every law infeasible, the IDM and the margin core included (no common quadratic storage function over the derivative box of a band); I(v) per speed: B and IDM certified at 1.0001, E certified with gamma 1.044 / 1.098 / 1.248 / 8.24 at r_max 0.1 / 0.2 / 0.3 / 0.5 (box) and 1.028-1.159 (hull), C and D infeasible | **does not explain** as specified; the criterion is too conservative on the band to separate anything. On I(v) it separates the laws with a derivative budget from those without, in the direction of the hypothesis, but gamma rises with the budget while the corridor error falls with it, so the certified gain is not a proxy of fidelity either. Budget allocation: degenerate under the LMI; under the pointwise certificate of the manuscript the admissible total budget is 1.34-1.37 x the uniform one, all of it on s and dv |
| A3 safety filter (D130, section C.2) | contacts of the three networks zero on both scenarios, E's error within 0.02 | contacts per 1 000 vehicle-km: MLP 814 -> 0 and 587 -> 0.23, GRU 173 -> 0.36 and 301 -> 0, LSTM 6 305 -> 0.37 and 2 978 -> 2.27; E +0.017 / -0.008 | **does not work** by the letter (4 of 6 cells above zero), works in substance: contacts fall by 99.9-100 % at an unchanged certified hybrid and a lower macro error of the LSTM and of the GRU on US-101 (the GRU on I-80 gets worse, +0.067). The residue comes from the constant-leader-speed assumption of the barrier: a leader braking at 3 m/s^2 or more still produces a contact |
| H-N4 empirical gain of the drivers (D131, section C.3) | the networks' gain above the drivers' at 0.3-2 rad/s for >= 4 of 5 networks, the IDM below | NGSIM I-80, 3 592 events >= 40 s: excess log gain MLP -0.92, PIDL -0.79, GRU -0.58, LSTM -0.82, PERL -0.36, IDM -0.31, k-NN -0.64, every interval below zero; highD events are 14.9 s long and cannot be used | **refuted**: at 0.3-2 rad/s the learned laws amplify less than the drivers they were fitted to; their string instability lives at lower frequencies, which events of 15-40 s cannot resolve. Caveat: the leader-follower coherence is below 0.25 above 0.78 rad/s and the pooled estimator would say "open"; on the coherent cells both estimators say refuted |
| H-N5 FD-matched member weights (D132, section B.3) | the tilted het law below the uniform one on both sites | the tilt (beta_T +0.44, beta_s -0.55) lowers the FD distance from 0.126 to 0.067; macro error +0.015 [-0.014, +0.043] on I-80 and +0.080 [+0.051, +0.116] on US-101 against the uniform law; +0.090 and +0.141 against E | **not confirmed**: the capacity (fundamental-diagram) mismatch is not what makes the heterogeneous law worse than the single core |

What follows from this for the next steps, in the order of cost: (1) A1': the windowed penalty with the maximum (or a
per-frequency tolerance) instead of the mean over the frequencies, a one-pilot change that targets the diagnosed
weakness; (2) A2': a latent carried along the vehicle (and along the event in training) instead of restarting at every
window, with the core certified over the range the latent may reach (T >= 0.92 T_c, or a core calibrated with the
margin at every T of the range); (3) the barrier of the safety filter with the leader's acceleration (or a bound on
it); (4) the explanation of the ablation remains open: it is not in the linearised gains at equilibria, along
trajectories or over derivative boxes, so the next candidates are nonlinear: the behaviour at merges, lane changes and
the insertion at the boundary, where the derivative budget bounds the response to large perturbations.

Deviations from the contract, all recorded by the agents in their sections and in D127-D133: `every` = 4 for the
windowed penalty (cost factor 5.6-8.8); the eigenvalue derivative from the dominant pole's eigenvectors; two
hash-neutral margin keys; the budget of the tt law enforced over the reachable latent range; the fine-tuning of the tt
law inside `e4_tt.yaml`; the f_dv sign convention of the project in the robust certificate; "gamma < 1" read as a
verified certificate at 1 + 1e-3 (every vertex has unit gain at zero frequency); the hull polytope as a supplementary
variant; the platoon transient growth with collided platoons counted as +inf; the empirical gain on NGSIM I-80 only;
the filtered runs as `_filter` law files; (h') added as a predictor. Agent C's files were converted to LF line endings.

## A: windowed-gain penalty (D128, H-N3)

Status: complete (2026-10-20 15:45). Every step of D128 was run: implementation and tests, pilot, choice of the weights,
five folds, full-history audit, long-window check, fine-tuning on NGSIM I-80, law files, 180 corridor runs, metrics,
asymmetry and tables. **H-N3: the numbers fall in the branch "refuted" of the decision rule** (A.6); the verdict is left
to the main session.

Summary: no architecture reaches a stable band share of 0.9 at any pilot weight. At the chosen weights (D85) the
five-fold means on highD are GRU 0.09 at +17.3 % test RMSE (long-window check fails), LSTM 0.66 at +36.2 % (check
holds), PERL 0.00 at +31.9 % (reported). Where the penalty is strong enough it repairs the low-frequency band (M_w >= 0.5,
poles inside, the long-window check holds for GRU 1 and 10 and LSTM 10), but an amplification of about 1.1 near 1 rad/s
stays or appears, which its frequency-averaged squared gain term barely sees (A.9, item 8). In the corridor the windowed
GRU and LSTM laws have the lowest macro error of their architectures (rows only, A.8).

### A.1 What was implemented

- **Penalty kind `windowed_gain`** (`cf_stability/stability/penalties.py`). At every penalty step, for the 16 speeds of
  the epoch (drawn as for the gain penalty), the equilibria are found with the band of the training data, the model is
  run over its window W = 30 on the constant equilibrium history (s*, 0, v*) and differentiated once with respect to
  the whole history tensor [W, 3] (`window_jacobian`, one backward pass for all speeds, `create_graph`): the lag
  Jacobians of Appendix A. From them (`windowed_response`, `windowed_terms`, float64): the windowed transfer function
  G_w of equation (A2) at the 25 audit frequencies 0.02-2 rad/s (`transfer_windowed`; the frequencies are
  `linear_omega_min/max/n_omega`, log-spaced, equal to the audit's), the coefficient M_w in closed form
  (`windowed_margin`) and the spectral radius rho of the (e, u) block of the closed loop (`spectral_radius`). Penalty =
  mean over the anchored equilibria of mean_w relu(|G_w| - 1 + 0.02)^2 + relu(0.5 - M_w) + relu(rho - 0.995), plus the
  existence term of the band exactly as in `gain` / `combined`; the trainer multiplies the sum by `train.penalty.weight`
  (and by `every` on the penalty steps). Info in the history: `n_no_equilibrium`, `max_gain`, `min_margin_window`,
  `max_radius`, the means of the three terms, `existence`.
- **Configuration.** Margins: `train.penalty.window_gain_margin` = 0.02 and `train.penalty.pole_margin` = 0.995 (new keys of
  `configs/train.yaml`, hash-neutral at these defaults: two entries appended to `LATE_KEYS` in `cf_stability/utils.py`;
  they are refused for every other kind); the margin of M_w is `train.penalty.margin` = 0.5, the margin of the Jacobian
  penalty, as D128 says. `n_equilibria` defaults to 16 for the kind.
- **CUDA graphs.** The kind is not in `CAPTURABLE` (`NOT_CAPTURABLE` in `penalties.py`), so the trainer never tries to
  capture its penalty steps: they run eagerly, and with `every > 1` the steps without penalty are replayed from their
  graph (`cuda_graph` stays true in metrics.json, `penalty_graph` false, `plain_graph` true). Reason, checked on
  2026-10-19: a capture that meets `torch.linalg.eigvals` fails and leaves the CUDA context of the process unusable
  ("operation failed due to a previous error during capture"), so the fallback of `_GraphedStep` (eager after a failed
  capture) cannot be used for this kind.
- **Eigenvalues and their derivative.** rho is computed with `torch.linalg.eig` (the LAPACK routine of `eigvals`) on the
  CPU in float64 without gradient (6.5 ms for the forward and backward of 16 matrices 31 x 31, against 60 ms on the
  GPU); its derivative is that of the dominant pole, d lambda = w^T dA x / (w^T x) with the right and left eigenvectors
  (|w^T x| bounded below by 1e-8), applied on the device of the model. This equals the derivative of
  `torch.linalg.eigvals` (relative difference 1e-11 on the stored E1 runs of GRU, LSTM and PERL, fold 0, seed 0; 1e-8 on
  random Jacobians) and stays finite and exact where other poles are nearly defective (a law that ignores its oldest
  states has a Jordan block at zero; tested against central differences). The backward pass never leaves the GPU:
  with a CPU branch in it the order in which the gradients were summed depended on the timing of the host thread, and
  two trainings with the same seed differed in the last bits.
- **Long-window check.** `rollout_gain` was factored out of `gain_penalty` (the same operations in the same order;
  the GPU tests of the trainer that compare graph and eager weights bit by bit pass after the change) and is used by
  the new script `scripts/analysis/long_window_check.py` (the rollout of `e2_horizon_pilot`: 380 s after the warm-up,
  the amplitude fitted over the last 252 s, leader amplitude 0.2 m/s, at 0.02, 0.05 and 0.1 rad/s, at the equilibria of
  the stored audit; float64, eval mode; `long_window.json` per run, table `runs/_tables/m11/long_window`).
- **Full-history audit.** `scripts/analysis/full_history_audit.py --arm-set m11`: the arms of `M11_ARMS` (E1 seed 0, the
  chosen weight on five folds, the three pilot weights on fold 0) into `runs/_tables/m11/full_history_windowed.{csv,md}`;
  `full_history.json` is written only into the runs of `e2_windowed*`. The default arm set and its table are unchanged.
- **Tables.** `scripts/analysis/windowed_gain.py` (`choose`: the pilot table `e2_windowed_pilot` and
  `e2_windowed_chosen.json`; `tables`: also the five-fold table `e2_windowed` with the fine-tuned members and the
  branch of the rule of H-N3 in `e2_windowed_branch.md`), built on `TableMaker` of `cf_stability/eval/tables.py`
  (intervals, paired RMSE changes, poles as in the M4 tables).
- **Corridor tables.** `cf_stability/eval/corridor_tables.py`: a design key `windowed_laws` (`gru_windowed`,
  `lstm_windowed`, `perl_windowed`) handled as `ablation_laws`: optional, rows of laws, components, instability, asymmetry
  and contacts_absolute, not in the correlations nor the verdicts (my block of insertions; agent B added `m11_laws` the
  same way next to it).

### A.2 Hash neutrality and reproducibility checks

- Dry runs of all 22 earlier queue files (`run_experiment.py --dry-run`) before and after the new keys: identical
  output, 1 131 jobs.
- `python scripts/make_tables.py` before the change (2026-10-19) and after all runs of this section on highD and NGSIM
  (2026-10-20 14:35): `runs/_tables/m4` (and `runs/_tables/m8/e2_monotone`) byte-identical to git both times.
- `python scripts/analysis/full_history_audit.py --no-json --no-figure` with the extended script: `runs/_tables/m9/
  full_history.{csv,md}` byte-identical to git (207 runs).

### A.3 Tests

New: `tests/test_penalties_windowed.py` (18 tests: configuration and late keys; a linear law with geometric memory
whose window polynomials, M_w (closed-form lag moments) and poles (roots of the characteristic polynomial) are known:
the response equals them to 1e-10, the penalty with margins 1 / -100 / 1 equals the mean squared gain of its known
transfer function, with the D128 margins the closed-form value; its windowed gain equals the gain of the project's
rollout to 1e-6; the memoryless limit; central-difference gradients of each term and of all three together on a small
GRU; the derivative of rho against `eigvals` and at nearly defective poles; M_w, rho and |G_w| equal to those of
`full_history_audit.analyse` on three stored runs (E1 GRU, E2 LSTM, E1 PERL, fold 0) to 1e-9; anchoring and the
existence term; a training run on synthetic data; on CUDA (opt-in `CF_GPU_TESTS=1`) no capture attempt, plain steps
from the graph, bitwise the eager weights). `tests/test_windowed_analysis.py` (6 tests: the m11 arm set, the long-window
check against the windowed transfer function of linear laws, its summary, the script, the branch of the rule and the
table helpers). `tests/test_corridor_tables_windowed.py` (2 tests: the rows of the windowed laws in laws, components and instability with the member audits, not in the correlations; an absent law left out with a note). Changed: `tests/test_utils.py` (the two late keys), `tests/test_run_experiment.py` (my three queue files). Results: see A.10.

### A.4 `every`

Smoke runs of 2026-10-19 (follownet_highd fold 0, 2 epochs of 40 steps, weight 1, the GPU otherwise idle; outputs in
the scratchpad): an eager penalty step took 0.173 / 0.324 / 0.413 s, a step without penalty replayed from its graph
0.031 / 0.037 / 0.052 s (GRU / LSTM / PERL): 5.6 / 8.8 / 7.9 times, above the factor 3 of D128, so `every` = 4 (the
penalty multiplied by 4 on every 4th step; mean cost per step 2.2-2.9 times a plain step, against 3.3-4.9 with 2).
Most of the cost of the penalty step is the eager rollout loss, not the penalty (0.05 s for 16 speeds).

### A.5 Pilot (`configs/queue/e2_windowed_pilot.yaml`, 9 runs, follownet_highd fold 0, seed 0)

Table `runs/_tables/m11/e2_windowed_pilot.md` (unit run; RMSE change against the E1 run of fold 0, seed 0, paired over
the 2 503 test events; stable = band share of the audit, speeds in support):

| arch | weight | val RMSE | test RMSE | change vs E1 | stable | unstable | outside | max gain | feasible epochs | val penalty | min M_w | max rho |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GRU | 0.1 | 2.44 | 2.42 | +17.4 % | 0.23 | 0.77 | 0.00 | 1.364 | 0 | 0.548 | -0.002 | 0.995 |
| GRU | 1 | 2.91 | 2.98 | +44.2 % | 0.23 | 0.59 | 0.18 | 2.066 | 0 | 0.283 | 0.071 | 1.089 |
| GRU | 10 | 2.93 | 2.94 | +42.6 % | 0.00 | 1.00 | 0.00 | 1.113 | 9 | 0.003 | 1.260 | 0.995 |
| LSTM | 0.1 | 2.55 | 2.54 | +13.8 % | 0.41 | 0.59 | 0.00 | 11.142 | 0 | 0.470 | -0.039 | 1.066 |
| LSTM | 1 | 2.84 | 2.82 | +26.7 % | 0.00 | 1.00 | 0.00 | 1.711 | 0 | 0.500 | -0.032 | 0.996 |
| LSTM | 10 | 2.98 | 2.94 | +31.8 % | 0.45 | 0.55 | 0.00 | 1.108 | 16 | 0.002 | 0.506 | 0.997 |
| PERL | 0.1 | 3.19 | 3.14 | +29.1 % | 0.00 | 1.00 | 0.00 | 1.321 | 0 | 0.564 | 0.000 | 0.997 |
| PERL | 1 | 3.18 | 3.09 | +26.9 % | 0.00 | 1.00 | 0.00 | 45.693 | 0 | 0.507 | -0.001 | 1.004 |
| PERL | 10 | 3.27 | 3.17 | +30.3 % | 0.00 | 1.00 | 0.00 | 9.239 | 0 | 0.439 | -0.000 | 1.063 |

(val penalty, min M_w, max rho: the validation penalty of the best epoch at the grid speeds and its parts.)

**Chosen weights (rule of D85):** no weight reaches a stable share of 0.9 for any architecture, so the largest share
decides: GRU 0.1 (0.23, tied with weight 1, the smaller validation RMSE), LSTM 10 (0.45), PERL 1 (0.00 at every weight,
the smallest validation RMSE). Written to `runs/_tables/m11/e2_windowed_chosen.json` and `configs/queue/e2_windowed.yaml`.

Observations (no verdict):
- At weights 0.1 and 1 the term relu(0.5 - M_w) dominates the penalty (about 0.5: M_w stays near 0, as in E1) and no
  epoch is feasible (D89), so the best epoch is the one of the smallest validation RMSE.
- At weight 10 GRU and LSTM meet the penalty at the validation grid (penalty 0.003 and 0.002, 9 and 16 feasible epochs,
  M_w >= 0.5 everywhere), yet the audit finds the GRU string unstable at every speed in support and the LSTM at 55 %:
  a broad resonance with |G_w| about 1.1 near 1.1 rad/s (GRU: windowed gain at the audit equilibria 1.087-1.109,
  numerical gain 1.078-1.113). Its cost in the penalty is the mean over the 25 frequencies of the squared excess,
  about 0.002, below the feasibility tolerance 0.01 of D89: the observation of D81 (one violated frequency weighs 1/25
  in a mean over the frequencies) holds for the squared hinge of D128 as well. The low-frequency band is repaired
  (M_w > 0), the amplification moves to about 1 rad/s.
- PERL: no weight gives a stable equilibrium in the band; at weights 1 and 10 poles outside the unit circle (rho up to
  1.06) and audit gains up to 46.

### A.6 Five folds (`configs/queue/e2_windowed.yaml`)

Queue with the chosen weights (gru 0.1, lstm 10, perl 1; every 4, 16 speeds): 15 jobs, of which the 3 of fold 0 are
the pilot runs (skipped); 12 runs trained, audited and platoon-tested on 2026-10-20 11:20-12:38 with 2 workers, none
failed (9-17 min per job). Table `runs/_tables/m11/e2_windowed.md` (`python scripts/analysis/windowed_gain.py tables`;
follownet_highd, seed 0, five folds; RMSE change against the E1 runs of the same folds and seed, paired over the 12 512
test events; stable = mean over the runs of the stable band share, interval over the runs):

| arch | arm | test RMSE (m) | change vs E1 | stable | unstable | not stable (poles incl.) | unstable among eq. | max gain (median) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GRU | E1 | 2.09 [2.05, 2.13] | | 0.09 [0.00, 0.18] | 0.50 | 0.91 | 0.89 | 2.577 |
| GRU | windowed 0.1 | 2.45 [2.41, 2.48] | +17.3 % [+15.1, +19.2] | 0.09 [0.04, 0.16] | 0.91 | 0.91 | 0.91 | 1.370 |
| LSTM | E1 | 2.18 [2.15, 2.21] | | 0.10 [0.05, 0.15] | 0.36 | 0.90 | 0.84 | 133.894 |
| LSTM | windowed 10 | 2.97 [2.92, 3.01] | +36.2 % [+34.2, +38.3] | 0.66 [0.35, 0.97] | 0.34 | 0.34 | 0.34 | 1.081 |
| PERL | E1 | 2.37 [2.33, 2.41] | | 0.06 [0.03, 0.11] | 0.80 | 0.94 | 0.93 | 39.321 |
| PERL | windowed 1 | 3.13 [3.08, 3.18] | +31.9 % [+30.0, +33.7] | 0.00 [0.00, 0.00] | 0.99 | 1.00 | 1.00 | 1.429 |

Per fold (stable share; test RMSE): GRU 0.23, 0.09, 0.00, 0.09, 0.05 (2.41-2.54 m); LSTM 0.45, 1.00, 1.00, 0.00, 0.86
(2.94-3.01 m; fold 3 unstable at every speed with a largest gain of 1.081); PERL 0.00 at every fold (3.09-3.31 m;
largest gains 1.33-45.7). With the poles of the full-history loop the shares do not change for GRU and LSTM (every
pole inside at the audited speeds in support); PERL has a pole on or outside the unit circle at 30 % of its speeds in
support (A.7).

**Branch of the decision rule of H-N3** (`runs/_tables/m11/e2_windowed_branch.md`): GRU stable share 0.091 (< 0.9),
RMSE change +17.3 % (> 10 %), long-window check fails; LSTM 0.664 (< 0.9), +36.2 % (> 10 %), the check holds; PERL (not
judged) 0.000, +31.9 %, fails. Neither GRU nor LSTM reaches the share: the numbers fall in the branch "refuted"
(otherwise) of the rule.

### A.7 Full-history audit and long-window check

**Long-window check** (`python scripts/analysis/long_window_check.py --device cuda`, 2026-10-20; table
`runs/_tables/m11/long_window.md`, per-equilibrium results in `long_window_runs.json`, `long_window.json` in the runs of
the windowed penalty only): rollout gain over the last 252 s of a 380-s rollout at 0.02, 0.05 and 0.1 rad/s, at the
audit equilibria inside the band at the speeds in support, fold 0, seed 0 (a first run of the script on the CPU had
also written the file into the reference runs of E1; those three files were moved to the scratchpad and the script now
writes only into `e2_windowed*`).

| arch | arm | speeds | max gain 0.02 / 0.05 / 0.1 rad/s | share > 1.02 | holds |
|---|---|---:|---|---:|---|
| GRU | E1 | 4 | 1.153 / 1.214 / 1.199 | 1.00 | no |
| GRU | E2 long window (gain 0.1, 380 s) | 17 | 0.993 / 0.970 / 0.942 | 0.00 | yes |
| GRU | windowed 0.1 (chosen) | 22 | 1.125 / 1.339 / 1.310 | 0.64 | no |
| GRU | windowed 1 | 18 | 0.997 / 0.992 / 0.990 | 0.00 | yes |
| GRU | windowed 10 | 22 | 0.997 / 0.989 / 0.983 | 0.00 | yes |
| LSTM | E1 | 12 | 1.760 / 1.877 / 1.155 | 0.50 | no |
| LSTM | windowed 0.1 | 22 | 1.102 / 1.265 / 1.474 | 0.23 | no |
| LSTM | windowed 1 | 22 | 1.050 / 1.159 / 1.144 | 0.59 | no |
| LSTM | windowed 10 (chosen) | 22 | 0.995 / 0.985 / 0.980 | 0.00 | yes |
| PERL | E1 | 19 | 1.106 / 1.333 / 2.915 | 0.79 | no |
| PERL | windowed 0.1 | 22 | 0.999 / 0.998 / 1.154 | 0.45 | no |
| PERL | windowed 1 (chosen) | 22 | 1.015 / 1.731 / 2.960 | 0.59 | no |
| PERL | windowed 10 | 22 | 1.052 / 4.199 / 1.256 | 0.45 | no |

The rollout gains agree with the audit's numerical gains at its frequencies <= 0.1 rad/s (column audit of the table:
e.g. GRU w0.1 1.339 against 1.364, LSTM w10 0.995 against 0.995). At the chosen weights the check holds for the LSTM
(weight 10) and fails for the GRU (weight 0.1) and PERL (weight 1); for the GRU it holds at weights 1 and 10, which the
rule of D85 did not choose (their stable band shares 0.23 and 0.00: the amplification of GRU 10 sits at about 1 rad/s,
above the frequencies of this check).

**Full-history audit** (`python scripts/analysis/full_history_audit.py --arm-set m11`, 39 runs: E1 seed 0 and the chosen
weights on five folds, the three pilot weights on fold 0; table `runs/_tables/m11/full_history_windowed.md`, speeds in
support, mean over the runs with the interval over the runs):

| arch | arm | runs | speeds | poles inside | windowed > 1.02 | numerical unstable | agreement | sign(M_w) vs G_w(0.02) | sign(M) = sign(M_w) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GRU | E1 | 5 | 104 | 1.00 | 0.89 | 0.89 | 1.00 | 0.98 | 0.75 |
| GRU | windowed 0.1 | 5 | 110 | 1.00 | 0.91 | 0.91 | 1.00 | 1.00 | 0.72 |
| LSTM | E1 | 5 | 101 | 0.94 | 0.81 | 0.84 | 0.97 | 1.00 | 0.73 |
| LSTM | windowed 10 | 5 | 110 | 1.00 | 0.45 | 0.34 | 0.89 | 1.00 | 1.00 |
| PERL | E1 | 5 | 106 | 0.94 | 0.93 | 0.93 | 1.00 | 1.00 | 0.85 |
| PERL | windowed 1 | 5 | 110 | 0.70 [0.30, 1.00] | 0.90 | 1.00 | 0.90 | 0.95 | 0.19 |

The windowed (full-history) gain agrees with the numerical audit at 89-100 % of the speeds; for the LSTM at weight 10 the
linearisation flags more speeds (0.45) than the rollouts of the audit (0.34): gains of 1.02-1.08 near 1 rad/s, where
the response to the amplitude 0.2 m/s of the audit is a little smaller than the linear one. The low-frequency
coefficient decides the side of 1 of the windowed gain at 0.02 rad/s at 95-100 % of the speeds; the memoryless margin M
agrees in sign with M_w at 72-85 % of the speeds of GRU and E1 (the memory, D66), at all speeds of the penalised LSTM
(its M_w >= 0.5 makes both positive) and at 19 % for the penalised PERL (Newell's feed-forward, M_w ~ M + 2 f_s, as for
the combined penalty of M9). Under the penalty the LSTM has every pole inside; PERL at weight 1 has a pole on or outside
the unit circle at 30 % of its speeds (E1: 6 %).

### A.8 Fine-tuning on NGSIM I-80 and corridor

Prepared: `configs/queue/m11_ft_windowed.yaml` (15 runs, ngsim_i80, `init_from` the e2_windowed run of the same fold,
the penalty continued with the same kind, weight, `every` and speeds, steps train and audit, as `m5_laws`); my block of
three laws appended to `configs/corridor/laws.yaml` (`gru_windowed`, `lstm_windowed`, `perl_windowed`: kind models,
cuda, members `runs/m11_ft_windowed_w{0.1,10,1}/ngsim_i80/{gru,lstm,perl}/driver_fold{fold}_seed0`); the block
`windowed_laws` appended to the design of `configs/corridor_metrics.yaml` next to `ablation_laws` and `m11_laws`; test
`tests/test_corridor_tables_windowed.py` (rows of laws, components and instability with the member audits, not in the
correlations, the absent law left out with a note).

**Fine-tuning** (`configs/queue/m11_ft_windowed.yaml`, 15 runs, 2026-10-20 12:38-14:19 with 2 workers, 4-29 min per
job): 14 done, 1 failed — LSTM fold 0 stopped after 555 s with "CUDA error: an illegal memory access was encountered"
(raised at the synchronisation of `torch.isfinite(loss)` in the trainer; the GPU was shared with the corridor and
training jobs of the other agents). Rerun by restarting the queue (14 jobs skipped as complete): done in 12.2 min, no
change of anything. A transient fault: the same job, code and data passed on the second attempt. Rows of
`runs/_tables/m11/e2_windowed.md` (ngsim_i80, five folds, seed 0; RMSE change against `m5_ft`, the E1 models fine-tuned
in M5, paired over the 5 521 test drivers; for information, the members of the corridor laws):

| arch | arm | test RMSE (m) | change vs m5_ft | stable | unstable | max gain (median) |
|---|---|---:|---:|---:|---:|---:|
| GRU | m5_ft | 4.53 [4.45, 4.62] | | 0.27 [0.12, 0.43] | 0.64 | 1694 |
| GRU | windowed 0.1, fine-tuned | 4.59 [4.49, 4.70] | +1.5 % [+0.0, +3.1] (p 0.11) | 0.63 [0.50, 0.76] | 0.37 | 1.141 |
| LSTM | m5_ft | 4.72 [4.62, 4.83] | | 0.25 [0.06, 0.45] | 0.67 | 861 |
| LSTM | windowed 10, fine-tuned | 4.97 [4.86, 5.10] | +5.2 % [+3.5, +7.1] | 0.91 [0.78, 1.00] | 0.09 | 1.052 |
| PERL | m5_ft | 6.47 [5.87, 7.50] | | 0.06 [0.01, 0.11] | 0.81 | 785 |
| PERL | windowed 1, fine-tuned | 8.38 [7.72, 9.38] | +29.4 % [+23.2, +36.6] (p 0.09) | 0.17 [0.00, 0.43] | 0.83 | 1.500 |

(PERL fold 3 has a test RMSE of 15.9 m on NGSIM I-80.) The penalty, continued during the fine-tuning on the NGSIM band,
raises the stable share of the members on I-80 for GRU and LSTM at a small cost in accuracy; this is information on the
corridor members, not part of the rule of H-N3 (highD).

**Law files**: `python scripts/build_corridor.py scenarios=false` (14:19, inside the chain) wrote `gru_windowed` and
`perl_windowed` (5/5 members, window 30, support 0.0-18.1 m/s, cuda); `lstm_windowed` was not written then (its fold-0
member was missing) and was exported after the rerun with the same export function restricted to that law
(`cf_stability.corridor.laws.export_laws(..., only=["lstm_windowed"])`), so that the law files of the corridor runs
in progress (mine and those of agents B and C) were not rewritten while they were being read.

**Corridor runs**: 180 runs (3 laws x 6 scenarios x 10 seeds), none failed: `gru_windowed` and `perl_windowed` 14:20-15:15,
`lstm_windowed` 15:15-15:38, workers 3 (GRU and LSTM 53-60 s per run, PERL about 150 s); `corridor_metrics.py law=<law>`
and `corridor_asymmetry.py law=<law> workers=2` for each law: 180 macro.json and 180 asymmetry.json, all OK (the
ground truths kept). Then `python scripts/make_tables.py` (m4 unchanged, see A.2) and `python scripts/corridor_metrics.py
compute=false tables=true` (15:40): the three laws are further rows of `runs/_tables/m5/laws`, `components`,
`instability` and of `runs/_tables/m8/asymmetry`, `contacts_absolute` (1 670 runs in the tables); the correlations
(`instability_correlation`, `correlation_pooled`, `correlation_clustered`), the verdicts of H12 and every other table
are unchanged apart from the run count of their header line.

| corridor | law | macro error | dynamic macro error | contacts / 1000 veh-km | inserted | members: unstable among eq. | asymmetry index |
|---|---|---:|---:|---:|---:|---:|---:|
| I-80 | idm_global | 0.182 [0.176, 0.189] | 0.345 | 0.0 | 1.000 | 0.000 | 1.225 |
| I-80 | residual_idm_certified | 0.125 [0.109, 0.144] | 0.190 | 0.0 | 1.000 | 0.000 | 1.281 |
| I-80 | gru | 0.192 [0.164, 0.223] | 0.312 | 206 | 0.989 | 0.715 | 0.377 |
| I-80 | gru_penalty | 0.235 [0.185, 0.296] | 0.363 | 731 | 0.988 | 0.359 | |
| I-80 | **gru_windowed** | 0.132 [0.112, 0.153] | 0.233 | 270 | 1.000 | 0.370 | 0.623 |
| I-80 | lstm | 0.716 [0.662, 0.774] | 0.657 | 4 664 | 0.335 | 0.733 | 0.610 |
| I-80 | **lstm_windowed** | 0.423 [0.385, 0.465] | 0.511 | 2 666 | 0.950 | 0.092 | 0.450 |
| I-80 | perl | 0.431 [0.396, 0.464] | 0.652 | 11 683 | 0.989 | 0.940 | 0.584 |
| I-80 | **perl_windowed** | 0.574 [0.544, 0.611] | 0.471 | 12 560 | 0.849 | 0.829 | 0.612 |
| US-101 | idm_global | 0.133 [0.110, 0.163] | 0.229 | 0.03 | 1.000 | 0.000 | 1.128 |
| US-101 | residual_idm_certified | 0.113 [0.107, 0.119] | 0.122 | 0.23 | 1.000 | 0.000 | 1.488 |
| US-101 | gru | 0.421 [0.338, 0.515] | 0.517 | 211 | 0.909 | 0.715 | 0.319 |
| US-101 | gru_penalty | 0.347 [0.278, 0.428] | 0.458 | 753 | 0.950 | 0.359 | |
| US-101 | **gru_windowed** | 0.260 [0.222, 0.304] | 0.481 | 298 | 1.000 | 0.370 | 0.390 |
| US-101 | lstm | 0.888 [0.784, 1.004] | 0.837 | 2 858 | 0.382 | 0.733 | 0.619 |
| US-101 | **lstm_windowed** | 0.679 [0.607, 0.762] | 0.654 | 2 836 | 0.781 | 0.092 | 0.485 |
| US-101 | perl | 0.661 [0.621, 0.704] | 0.685 | 13 481 | 0.891 | 0.940 | 0.594 |
| US-101 | **perl_windowed** | 0.904 [0.821, 0.997] | 0.767 | 12 350 | 0.706 | 0.829 | 0.622 |

(macro error with its interval over the 30 runs of the law on the corridor; contacts: mean over the runs; inserted:
share of the planned vehicles inserted; members: mean over the five fold members of their audit share of unstable
equilibria, the same on both corridors; asymmetry: the acceleration asymmetry index of D121, ground truths 0.94-1.03.)
Descriptive, no paired test and no verdict (rows only, by design): with the windowed penalty the GRU and the LSTM have
the lower macro error of their architecture on both corridors (GRU I-80 0.132 against 0.192 free and 0.235 with the
rollout penalty, US-101 0.260 against 0.421 and 0.347; LSTM 0.423 / 0.679 against 0.716 / 0.888), the GRU inserts every
vehicle and the LSTM most of them (0.95 / 0.78 against 0.34 / 0.38 free); both still have contacts (270-298 and
2 666-2 836 per 1000 vehicle-km against none for the IDM and the certified hybrid). PERL with the penalty is worse than
free PERL on both corridors and keeps colliding.

### A.9 Deviations and observations for the decision log

1. `every` = 4 instead of 1, as D128 allows (the eager penalty step cost 5.6-8.8 times a plain step; A.4, queue files).
2. The derivative of rho is that of the dominant pole from its eigenvectors (the value from `torch.linalg.eig`, the
   routine of `eigvals`, on the CPU), not the autograd of `torch.linalg.eigvals`: equal where both are accurate, finite
   at nearly defective poles, and the backward pass stays on the GPU (bitwise reproducible training). The penalty steps
   are never captured in a CUDA graph (the kind is not in `CAPTURABLE`).
3. Two hash-neutral keys (`window_gain_margin`, `pole_margin`) instead of fixed constants, so that the margins of D128 are
   visible in `configs/train.yaml` and in the `train_config` of every run; `margin` and the frequencies reuse existing
   keys.
4. D85 ties: GRU 0.23 at weights 0.1 and 1 (the smaller validation RMSE: 0.1); PERL 0.00 at every weight (the smallest
   validation RMSE: 1). No weight of any architecture reached 0.9, so the weight of the largest share was chosen, as D85
   prescribes.
5. Long-window check: the equilibria are those of the stored audits (inside the band at the grid speeds in support;
   outside the band reported beside); the check holds when no gain reaches 1.02 at the three frequencies. A first CPU
   run of the script wrote `long_window.json` into the three E1 reference runs; the files were moved to the scratchpad,
   the script now writes only into `e2_windowed*` and keeps every result in `runs/_tables/m11/long_window_runs.json`.
6. Fine-tuning: one transient CUDA failure (illegal memory access, LSTM fold 0), rerun unchanged and complete. The law
   file `lstm_windowed` was then exported with `export_laws(only=[...])` instead of a second full `build_corridor.py`, so
   that the law files read by running corridor jobs (mine and those of agents B and C) were not rewritten.
7. `corridor_metrics.py` and `corridor_asymmetry.py` were run with `law=<law>` for the three new laws (the runs of the
   other agents untouched); the tables were then written for all runs (`compute=false tables=true`).
8. A property of the penalty as specified (not a deviation; relevant for H-N3 and any follow-up): its gain term is the
   mean over the 25 frequencies of the squared excess. At weight 10 a broad amplification of about 1.1 near 1 rad/s
   costs about 0.002 (GRU pilot: the windowed penalty is met, 9 feasible epochs, M_w >= 1.26, yet every speed is string
   unstable with gains up to 1.11), below the feasibility tolerance 0.01 of D89: the low-frequency coefficient and the
   poles are repaired, the amplification moves to higher frequencies. D81 made the same observation for `linear_gain` and
   took the maximum over the frequencies.
9. On CUDA the first training of a process differs in the last bits from the later ones for every penalty kind on this
   machine (lazy initialisation of the kernels): the bitwise GPU test of the windowed kind discards a first eager
   training. The earlier GPU tests compare trainings of other configurations and are not affected.
10. Shared files edited by appending my own block only: `configs/corridor/laws.yaml` (three laws), `configs/corridor_metrics.yaml`
    (`windowed_laws`), `cf_stability/eval/corridor_tables.py` (`WINDOWED_LAWS`, `WINDOWED_TEXT`, the field
    `windowed_laws`, `windowed_note()` and the law set in `table_laws()`, `present()` and the instability rows),
    `cf_stability/utils.py` (two `LATE_KEYS`), `tests/test_utils.py` (the two keys), `tests/test_run_experiment.py`
    (my three queue files and their assertions). `cf_stability/eval/tables.py` was not changed: the M11 tables are
    written by `scripts/analysis/windowed_gain.py` with its `TableMaker`.

### A.10 Commands run, run counts and durations

```bash
# smoke timing (2026-10-19; scratchpad, not in runs/)
python scripts/train.py data=follownet_highd model={gru,lstm,perl} fold=0 seed=0 experiment=m11_smoke_every1 \
    train.penalty.kind=windowed_gain train.penalty.weight=1 train.max_epochs=2 train.steps_per_epoch=40 paths.runs_root=<scratchpad>
python scripts/train.py ... experiment=m11_smoke_plain train.penalty.every=100000 ...      # the plain step from its graph
for q in configs/queue/*.yaml; do python scripts/run_experiment.py $q --dry-run; done      # before and after the new keys
python scripts/make_tables.py                                                             # before: m4 unchanged
python scripts/analysis/full_history_audit.py --no-json --no-figure                       # m9 table unchanged
python scripts/run_experiment.py configs/queue/e2_windowed_pilot.yaml                      # 9 runs, 2026-10-19 19:08-20:45
python scripts/analysis/windowed_gain.py choose                                           # D85: gru 0.1, lstm 10, perl 1
python scripts/run_experiment.py configs/queue/e2_windowed.yaml                           # 12 runs, 2026-10-20 11:20-12:38
python scripts/run_experiment.py configs/queue/m11_ft_windowed.yaml                       # 15 runs, from 12:38 (chained)
python scripts/analysis/long_window_check.py --device cuda                                # 17 runs (9 pilot + 8 reference)
python scripts/analysis/full_history_audit.py --arm-set m11                               # 39 runs
python scripts/analysis/windowed_gain.py tables                                           # pilot, five folds, branch
python scripts/run_experiment.py configs/queue/m11_ft_windowed.yaml                       # rerun of the failed job
python scripts/build_corridor.py scenarios=false                                          # gru_windowed, perl_windowed
python -c "...export_laws(laws_cfg, 'runs/corridor/laws', only=['lstm_windowed'])"      # after the rerun
python scripts/run_corridor.py scenarios=[i80_p0,i80_p1,i80_p2,us101_p0,us101_p1,us101_p2] \
    laws=[gru_windowed,perl_windowed] seeds=[0,1,2,3,4,5,6,7,8,9] workers=3               # 120 runs
python scripts/run_corridor.py scenarios=[...the same six...] laws=[lstm_windowed] seeds=[0,...,9] workers=3   # 60 runs
python scripts/corridor_metrics.py law=gru_windowed   # and lstm_windowed, perl_windowed: 180 macro.json
python scripts/corridor_asymmetry.py law=gru_windowed workers=2   # and the other two: 180 asymmetry.json
python scripts/make_tables.py                                                             # m4 unchanged
python scripts/corridor_metrics.py compute=false tables=true                              # rows of the windowed laws
python -m pytest tests/test_penalties.py tests/test_penalties_windowed.py tests/test_analysis_m9.py     tests/test_windowed_analysis.py tests/test_corridor_tables_windowed.py tests/test_utils.py tests/test_trainer.py
CF_GPU_TESTS=1 python -m pytest tests/test_penalties_windowed.py tests/test_trainer.py -k "graph or capture or eager"
```

Run counts and durations:

| step | runs | wall time | per job |
|---|---:|---|---|
| smoke timing (scratchpad) | 6 | 2026-10-19 | 7-20 s |
| e2_windowed_pilot (train, audit, platoon) | 9 | 2026-10-19 19:08-20:45 | 10-38 min (training 6.8-35.1 min, 14-58 epochs of 222 steps) |
| e2_windowed (train, audit, platoon) | 12 new + 3 pilot | 2026-10-20 11:20-12:38 | 9-17 min |
| long-window check | 17 | about 15 min (GPU) | |
| full-history audit, arm set m11 | 39 | about 5 min (CPU) | |
| m11_ft_windowed (train, audit) | 15 (one rerun) | 12:38-14:19, rerun 14:21-14:34 | 4-29 min |
| corridor runs | 180 | 14:20-15:38 (workers 3) | 53-150 s |
| macro.json and asymmetry.json | 180 + 180 | about 1 min | |

The pilot queue waited once for free memory (5.4 GB < 6 GB). Tests (2026-10-20, CPU): `python -m pytest tests/test_penalties.py tests/test_penalties_windowed.py tests/test_analysis_m9.py tests/test_windowed_analysis.py tests/test_corridor_tables_windowed.py tests/test_utils.py tests/test_run_experiment.py tests/test_corridor_tables.py tests/test_corridor_tables_ablation.py tests/test_corridor_tables_m8.py tests/test_corridor_tables_review.py`: 140 passed, 1 skipped (the opt-in GPU test) in 12.7 min; earlier the same day `tests/test_trainer.py` with the penalty suites (117 passed, 25 GPU tests skipped) and, with `CF_GPU_TESTS=1`, the GPU test of the windowed kind and the 23 graph and capture tests of the trainer: passed.

## B: two-timescale certified law (D129, H-N2) and FD-matched member weights (D132, H-N5)

Status (20 October, 13:10): complete. Every step of D129 and D132 is done: training, audits, certificates, the
law files, 120 corridor runs with their metrics and asymmetry files, the corridor tables (`make_tables.py`,
`corridor_metrics.py compute=false tables=true`, the M11 laws as rows of `runs/_tables/m5` and `runs/_tables/m8`), the
paired comparisons and the tables of `runs/_tables/m11` (certificate_tt, tt_pilot, tt_highd, tt_ngsim, tt_pairs,
het_fd_fit, het_fd_pairs). Branches of the rules: H-N2 **refuted** (RMSE 2.02 x the GRU's); H-N5 **not confirmed** (the
tilted law is not below the uniform one on either site).

### B.1 What was done (code)

- `cf_stability/models/residual_idm_tt.py` (new), registered as `residual_idm_tt` in `cf_stability/models/__init__.py`;
  model config `configs/model/residual_idm_tt.yaml` (as `residual_idm`: rollout loss only, `acc_weight` 0, D63).
  `ResidualIDMTT` subclasses `ResidualIDM`: core time gap `T(z) = T_c exp(z ln 1.5)`, `s0` and the other parameters
  fixed; the residual `r_max tanh(lipschitz g(s/20, dv/2, v/10, z/z_scale))` takes `z` as a fourth input of its own
  scale (`z_scale` 1), spectral normalisation over the four inputs together, so the bound of `|dr/d(s, dv, v)|`
  (`jacobian_bound`, that of ResidualIDM) is the same at every `z`; the latent
  `z_{t+1} = z_t + (dt/tau_z)(tanh(g_phi(h_t)) - z_t)`, `z_0 = 0`, with `h_t` a one-layer GRU (hidden 16) over the
  window of 30 states (data scaler of the training part), linear head `g_phi`, gradients through `z`. The latent is
  integrated over the window the model is called with, so the model is a function of its window like the GRU of the
  zoo: training rollouts, audit, platoon test, fine-tuning with `init_from` and the corridor loop (the history of every
  vehicle, as for the GRU laws) need no special case. `frozen_z` (None: online latent) freezes the latent: the model is
  then the memoryless hybrid with the core at `T(z)` (`window` 1). `at_z(z)` gives the same hybrid as a
  `FrozenLatentHybrid` (a ResidualIDM whose `idm` is the core at `T(z)`), which the certificate code takes as it takes
  the E4 hybrid. The head of the latent starts at zero (`z = 0`, `T = T_c`) and the residual at zero (antisymmetric
  start), so the untrained model is the margin core (epoch 0 of the E4 recipe). `from_residual_idm` builds the model
  from an E4 hybrid (the weights of `z` zero): at `frozen_z = 0` it computes that hybrid.
- `cf_stability/stability/certificate.py` (extension only): `LATENT_SWEEP` (z = -1, -0.9, ..., 1), `has_latent`,
  `at_latent`, `reachable_latent`, a keyword `z=` of `certificate_a_priori`, `certificate_at_equilibria`,
  `empirical_margin`, `core_margin`, `max_residual_budget` (the core of `T(z)`, the residual bounds unchanged),
  `latent_core_params` and `a_priori_exact_latent` (the exact a priori certificate of M9 at every z from the stored
  inputs), and `enforce_budget` for a latent model: the certified budget of D86 is the smallest over 21 frozen
  latents of the range the online latent can reach (record `latent` in `metrics.json["certificate_budget"]`; the
  record of a ResidualIDM is unchanged).
- `scripts/train.py`: `residual_idm_tt` gets the IDM calibration (`NEEDS_IDM`) and may use `certificate.enforce`.
  `scripts/certificate.py`: for a latent model `certificate.json` holds the entries of the hybrid at `z = 0`
  (`frozen_z: 0.0`) and a block `latent` (per z of the sweep: core, a priori, at the equilibria, per speed; the a priori
  certificate on the reachable range; counts). `scripts/audit_stability.py` + `configs/audit_stability.yaml`: option
  `frozen_z=<z>` (output `stability_z<z>.json`, hashed only when set; a model without latent fails with that message).
- `cf_stability/corridor/laws.py`: kind `residual_heterogeneous` takes `tilt` (D132): the weights
  `exp(beta_T T_i + beta_s s0_i)` of the cores (betas from the tilt file, which must have been fitted on the same cores)
  go to the table (`weight`) and the betas to `source.tilt`; `ResidualHetLaw` draws a vehicle's core with
  `Generator.choice(p=weights)` when the table has weights and with `Generator.integers` (as in M8) when not.
  `residual_idm_tt` needs nothing of its own (kind `models`).
- Analysis scripts: `scripts/analysis/het_fd_tilt.py` (the FD fit of D132), `scripts/analysis/certificate_tt.py` (the
  machinery of `certificate_exact.py` at the 21 frozen latents), `scripts/analysis/tt_summary.py` (tables tt_pilot,
  tt_highd, tt_ngsim and the branch of the rule of D129), `scripts/analysis/m11_pairs.py` (tt_pairs, het_fd_pairs:
  the builder of `ablation_pairs` with these pairs).
- Queues `configs/queue/e4_tt_pilot.yaml` (3 runs) and `configs/queue/e4_tt.yaml` (e4_tt and its fine-tuning e4_tt_ft
  in one file, so that a fine-tuning job waits in the queue for its source run; the queue `m11_ft_tt` of D133 was not
  needed as a file of its own).
- Appended blocks of shared files: `configs/corridor/laws.yaml` (the block "M11 (agent B)": laws
  `residual_idm_certified_het_fd` and `residual_idm_tt`), `configs/corridor_metrics.yaml` (`design.m11_laws`), and in
  `cf_stability/eval/corridor_tables.py` the field `m11_laws` handled as `ablation_laws` / the `windowed_laws` of agent
  A (optional, rows of laws, components, instability, asymmetry and contacts_absolute, `m11_note`; not in the
  correlations nor the verdicts).
- Tests (new): `tests/test_residual_idm_tt.py` (11), `tests/test_certificate_latent.py` (10),
  `tests/test_corridor_laws_m11.py` (8). `tests/test_models.py` (shared): `residual_idm_tt` added to its tables of the
  registry (5 570 trainable parameters, differentiable) and, for the generic interface test, its latent head given
  random weights (the untrained latent is zero by design, so the window would not count); `tests/test_run_experiment.py`
  (shared): the queue files `e4_tt_pilot` and `e4_tt` added to its list.

### B.2 D129 (A2, H-N2): two-timescale certified law

**Two facts of the design that shape every number below.**

1. *Reach of the latent.* With `z_0 = 0` at the start of the window of 30 states and `|tanh| < 1`,
   `|z| <= 1 - (1 - dt/tau_z)^30`: 0.182 (tau_z 15 s), 0.095 (30 s), 0.049 (60 s); T moves by at most ±7.7 %, ±3.9 %,
   ±2.0 % around `T_c`. On the observed windows of the highD test events (32 526-32 539 windows per fold) the latent of
   e4_tt sits at this bound: its 1 % and 99 % quantiles are -0.182 and +0.182 on every fold, its median -0.125 to
   -0.174 (table tt_highd): the training pushes it to the ends of its range.
2. *The margin core of E4 at small time gaps.* At `T_c / 1.5` (z = -1) the core of e4_stable has a string-stability
   margin of about 0 (minimum over 5-30 m/s between -0.001 and +0.000 on the five folds, against 0.200 at z = 0) and the
   admissible residual budget of the a priori certificate is 0 on every fold (fold 0: 0.012 at z = -0.5, 0.077 at
   z = -0.2, 0.118 at z = 0, 0.205 at z = 0.5). A budget enforced over z in [-1, 1] would be 0 (the training would stop
   with the error of D86). The budget is therefore enforced over the reachable range (21 values of z in
   [-reach, reach]): admissible 0.080-0.083 for tau_z 15 s (E4: 0.112), lipschitz 0.24 instead of 0.33.

**Pilot** (`e4_tt_pilot_tau{15,30,60}`, highD fold 0, seed 0; table `m11/tt_pilot`):

| tau_z | val RMSE s (m) | test RMSE s (m) | epochs (best) | reach of z | admissible budget | unstable eq. at z = -1, -0.5, 0, 0.5, 1 | band stable at z = -1 ... 1 |
|---:|---:|---:|---|---:|---:|---|---|
| 15 s | 4.271 | 4.170 | 39 (29) | 0.182 | 0.0802 | 0, 0, 0, 0, 0 | 1.000, 0.955, 0.955, 0.909, 0.864 |
| 30 s | 4.421 | 4.321 | 24 (14) | 0.095 | 0.0976 | 0, 0, 0, 0, 0 | 1.000, 0.955, 0.955, 0.909, 0.864 |
| 60 s | 4.488 | 4.392 | 28 (18) | 0.049 | 0.1068 | 0, 0, 0, 0, 0 | 1.000, 0.955, 0.955, 0.909, 0.864 |

"Unstable eq." is the share of unstable equilibria (numerical rule, gain > 1.02) among the equilibria in the support of
the speeds (the H1.1 statistic); all 26 grid speeds have an equilibrium at every frozen z; the band share "stable
inside the band" is below 1 only through equilibria outside the band at larger T (none unstable). Rule of D129: all
three runs have 100 % stable equilibria at every frozen z, so tau_z = the smallest validation RMSE = **15 s** (the band
shares are equal for the three runs, so a reading on the band share gives the same choice).

**Five folds on highD** (`e4_tt`, seed 0, tau_z 15 s; table `m11/tt_highd`):

| fold | test RMSE s (m) | GRU E1 (m) | E4 e4_stable (m) | unstable eq. at the 5 frozen z | max gain z = -1 | online: unstable eq., max gain | certificate: z of 21 (scan) | reachable range |
|---:|---:|---:|---:|---|---:|---|---:|---|
| 0 | 4.170 | 2.063 | 4.470 | 0 at all | 1.003 | 0, 1.000 | 13 | holds (min 0.0056) |
| 1 | 4.295 | 2.139 | 4.581 | 0 at all | 1.003 | 0, 1.001 | 13 | holds (min 0.0057) |
| 2 | 4.191 | 2.086 | 4.481 | 0 at all | 1.003 | 0, 1.004 | 13 | holds (min 0.0057) |
| 3 | 4.304 | 2.197 | 4.603 | 0 at all | 1.003 | 0, 1.005 | 13 | holds (min 0.0057) |
| 4 | 4.131 | 1.952 | 4.425 | 0 at all | 1.003 | 0, 1.000 | 13 | holds (min 0.0049) |
| mean | 4.218 | 2.087 | 4.512 | | | | | |

5-fold mean ratio to the GRU of E1: 4.218 / 2.087 = **2.02**; to E4: 0.935 (6.5 % lower RMSE than the certified hybrid
of E4). Platoon growth error 0.166-0.171, no collision. Band shares at the frozen z: 1.000 (z = -1), 0.955, 0.909-0.955,
0.909, 0.864 (equilibria outside the band at larger T, none unstable).

**Fine-tuning on NGSIM I-80** (`e4_tt_ft`, `init_from` the e4_tt run of the same fold; table `m11/tt_ngsim`): test
RMSE 4.810, 4.818, 4.797, 4.760, 4.746 m (mean 4.786) against 4.887 of `e4_stable_ft` (ratio 0.979); 19-34 epochs; no
unstable equilibrium at any frozen z, band share stable 1.000 at every frozen z and fold, max gain at most 1.003 (z = -1);
online audit max gain 0.9997-1.0083, no unstable equilibrium.

**Exact certificate at 21 frozen latents** (`scripts/analysis/certificate_tt.py`, table `m11/certificate_tt`; the
machinery of `certificate_exact.py`: exact minimum at the grid speeds and on the sweep of 2 501 speeds): before and after
fine-tuning the certificate holds for all 5 runs at the 13 latents z = -0.2, -0.1, ..., 1 and for none at
z = -1, ..., -0.3 (exact minimum -0.116 s⁻² at z = -1, -0.011 at z = -0.3; +0.003 at z = -0.2, +0.028 at z = 0). Exact,
scan and sweep minima agree (largest |exact - scan| < 1e-5; the stored scans are reproduced from the stored inputs).
On the reachable range |z| <= 0.182 (the online latent) the certificate holds for every run (min guaranteed margin
0.0049-0.0058 s⁻²).

**Rule of H-N2** (D129; 5-fold means on highD): test spacing RMSE 4.218 m = **2.02 x** the GRU's E1 RMSE (2.087 m), above
1.15 x: the numbers fall in the branch **refuted**. Of the two stability conditions of the branch "confirmed", the
frozen-z audits hold (no unstable equilibrium at any of the five frozen z, every fold, before and after fine-tuning) and
the certificate at all 21 z after fine-tuning does not (13 of 21 z for every fold; it holds on the reachable range of the
online latent).

**B.2b Corridor.** Law `residual_idm_tt` (`runs/corridor/laws/residual_idm_tt.json`, fingerprint `535eff82cab2`; the five
fine-tuned folds, drawn per vehicle as `residual_idm_certified` draws them, inference on the GPU), 60 runs (i80_p0-p2,
us101_p0-p2, seeds 0-9), all complete (8 contact episodes in all, no teleport). Paired by scenario and seed against E
(`scripts/analysis/m11_pairs.py`, table `m11/tt_pairs`; the rows of the laws in `runs/_tables/m5/laws.md`):

| corridor | residual_idm_tt | E (residual_idm_certified) | difference tt - E [95 % interval] | p (Wilcoxon) | dynamic macro error, tt / E | rank of tt by macro error |
|---|---:|---:|---|---:|---|---|
| I-80 | 0.1178 | 0.1248 | -0.0069 [-0.0270, +0.0118] (no difference) | 0.46 | 0.182 / 0.190 | 3 of 31 laws |
| US-101 | 0.0902 | 0.1129 | -0.0227 [-0.0289, -0.0153] (lower) | < 0.0001 | 0.082 / 0.122 | 1 of 26 laws |

The components are close to those of E (same core and residual family); the differences lie in the wave amplitude
(I-80 +0.092 against +0.162), the wave speed (US-101 -0.020 against -0.092) and the fundamental diagram (US-101 0.082
against 0.096). Instability of the members (their audits with the online latent): no unstable equilibrium, band share
not stable 0. Collisions per 1000 vehicle-km: I-80 0 (E 0), US-101 0.245 (E 0.227).

### B.3 D132 (H-N5): FD-matched member weights

`python scripts/analysis/het_fd_tilt.py` (table and figure `m11/het_fd_fit`, tilt file
`data/calibration/ngsim_i80/het_fd_tilt.json`): the 3 337 certified cores of `residual_idm_certified_het`, `s_i(v)` = the
equilibrium gap of core i with the residual of each of the five fold members (mean over the members; the fine-tuned
residuals sit at -r_max at steady states, so the mean hybrid gap is 0.8 m (5 m/s) to 1.4 m (10 m/s) larger than that
of the cores alone), vehicle length 4.972 m (mean of the demand of i80_p1), 6.208 lanes (mean over the five 100-m cells
of the diagram), the 31 density bins of the truth at their centres. L-BFGS-B from beta = 0 within |beta| <= 5:

| | beta_T (1/s) | beta_s (1/m) | FD RMSE / mean flow | effective cores | mean T (s) | mean s0 (m) |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 0 | 0 | 0.1258 | 3 337 | 1.625 | 2.819 |
| tilted (fitted) | +0.4357 | -0.5481 | 0.0673 | 1 868 | 1.904 | 1.010 |

The 41 x 41 grid on [-5, 5]² has its smallest distance at (0.5, -0.75), 0.0680 (the fit is the better point). Fitted on
the cores alone (without residual) the betas would be (+0.755, -0.463). The law `residual_idm_certified_het_fd` was built
with `build_corridor.py scenarios=false`: 3 337 cores (the same as the uniform law), tilted, effective cores 1 868. The
uniform law is unchanged: its re-export is byte-identical (JSON and npz), its fingerprint `3897c1e2d4d7` and the config
hash of its runs (`df2235354235` for i80_p1 seed 0, checked on 4 runs) are those of M8, and with the code of M8 and
of M11 its members, cores and accelerations are identical for seeds 0, 3, 9. The tilted law draws the same members as the
uniform law and `residual_idm_certified` for a seed.

Corridor runs: 60 (i80_p0-p2, us101_p0-p2, seeds 0-9), all complete (no failure); `corridor_metrics.py` and
`corridor_asymmetry.py workers=2` for this law: 60 macro.json, 60 asymmetry.json. Paired by scenario and seed (30 pairs
per corridor; `scripts/analysis/m11_pairs.py`, table `m11/het_fd_pairs`; the rows of the law in
`runs/_tables/m5/laws.md`: rank 16 of 31 laws on I-80, 14 of 26 on US-101):

| corridor | tilted (het_fd) | uniform (het) | difference het_fd - het [95 % interval] | p (Wilcoxon) | E (residual_idm_certified) | het_fd - E [interval] |
|---|---:|---:|---|---:|---:|---|
| I-80 | 0.2149 | 0.2003 | +0.0145 [-0.0143, +0.0426] | 0.21 | 0.1248 | +0.0901 [+0.0705, +0.1079] |
| US-101 | 0.2538 | 0.1737 | +0.0801 [+0.0511, +0.1161] | < 0.0001 | 0.1129 | +0.1409 [+0.1143, +0.1722] |

The fd component of the macro error (mean over the runs): I-80 0.115 (tilted) against 0.129 (uniform), the corridor of
the fit, US-101 0.135 against 0.128; the tilted law also has the smaller mean-speed error on I-80 (+0.067 against
+0.095), but fewer waves than the uniform law (component of the number of waves -0.80 against -0.61 on I-80, -0.77
against -0.25 on US-101) and larger errors of the wave amplitude (+0.25 against +0.09, +0.42 against +0.20) and, on
US-101, of the throughput and the travel time. Collisions per 1000 vehicle-km: I-80 0 (both), US-101 0.43 (tilted)
against 0.67 (uniform).

**Rule of H-N5** (D132): confirmed when the macro error of the tilted law is below that of the uniform het law on both
sites with paired-by-seed intervals excluding zero. The numbers: I-80 +0.0145 [-0.0143, +0.0426] (no difference), US-101
+0.0801 [+0.0511, +0.1161] (higher): the condition of "confirmed" is met on neither site (the rule names no other
branch). Against E the tilted law is higher on both sites (+0.090 and +0.141, intervals above zero).

### B.4 Deviations and choices left open by the contract

1. **Latent integrated over the model's window** (D129: "a one-layer GRU over the model's window, z_0 = 0"): z starts at 0
   at the first of the 30 states the model is called with and is integrated over them, so the model stays a function
   of its window and runs in the training rollouts, the audit, the platoon test and the corridor (the history of every
   vehicle, as the GRU laws) without a special case. Consequence: |z| <= 1 - (1 - dt/tau_z)^30 (0.182 at tau_z = 15 s);
   the latent cannot reach the range [-1, 1] that the frozen audits and the 21-z certificate cover, and on the test
   windows it sits at its bound (B.2).
2. **Certified budget over the reachable latent, not over [-1, 1].** With the margin core of E4 the admissible budget of
   the a priori certificate is 0 at z = -1 (core margin about 0 at T_c / 1.5) and below the E4 budget for z < 0; over [-1, 1]
   the training would stop with the error of D86. `enforce_budget` takes the smallest admissible budget over 21 values
   of z in [-reach, reach] (lipschitz = 0.9 b / r_max as in D86), so the certificate holds on every z the online latent
   can take; at z <= -0.3 it fails for every run (B.2, table certificate_tt).
3. z enters the residual with scale 1 (`z_scale`); the GRU of the latent reads the states with the data scaler of the
   training part (as the GRU of the zoo); the head of the latent starts at zero (the untrained model is the margin core,
   epoch 0 eligible as in E4); `frozen_z` makes the model memoryless (`window` 1).
4. "100 % stable equilibria at every frozen z" read as: no unstable equilibrium (numerical rule) among the equilibria in
   the support of the speeds (the H1.1 statistic, `share_unstable_numerical.support` = 0) with an equilibrium at every
   grid speed; the band shares "stable inside the band" are reported beside it (below 1 at z >= -0.5 on highD only
   through equilibria outside the band; 1 at every z after fine-tuning). "The certificate holds at all 21 z" read as: the
   a priori certificate (exact, at the grid speeds and on the 2 501-speed sweep) holds for every fine-tuned fold at every
   z of the sweep.
5. e4_tt fold 0 was trained again under its own experiment name (identical to the pilot run of tau_z 15 s: the same
   validation and test RMSE). The fine-tuning group e4_tt_ft is in `configs/queue/e4_tt.yaml` (a fine-tuning job waits
   only for a source of its own queue); no separate file `m11_ft_tt.yaml`.
6. D132: `s_i(v)` is the equilibrium gap of the hybrid (core i with the residual of every fold member, mean over the
   members), not of the core alone (the fine-tuned residuals sit at -r_max at steady states; the cores alone would give
   the betas (+0.755, -0.463)); vehicle length = mean length of the demand of i80_p1 (4.972 m); the total density of the
   diagram is n_lanes / S(v) with n_lanes = 6.208 (mean lane count of its five 100-m cells, 6 or 7 lanes); the bins at
   their centres; distance = unweighted sum of squares over the 31 bins (minimiser of the fd component of the macro
   error); L-BFGS-B. The instability of the tilted law in the corridor tables is that of its cores, unweighted (the
   cores are those of the uniform law).
7. The law `residual_idm_tt` runs its inference on the GPU (`device: cuda`, as the recurrent laws gru, lstm, perl); under
   the load of 19 October the CPU inference of a hybrid law took about 1 s per step for 400 vehicles.
8. Shared files edited outside the appended blocks: `scripts/train.py` (two lines), `scripts/audit_stability.py` and
   `configs/audit_stability.yaml` (`frozen_z`, hash-neutral at null), `scripts/certificate.py` (latent branch),
   `cf_stability/eval/corridor_tables.py` (`m11_laws`, next to the `windowed_laws` of agent A), `tests/test_models.py`
   (the registry tables) and `tests/test_run_experiment.py` (the list of queue files). No key was added to
   `configs/train.yaml` or `configs/run_corridor.yaml`; the dry runs of the queues e1, e4, e4_rmax and m8_temporal stay
   complete (175, 100, 200, 15 jobs, none to run) and every law file of M5-M8 keeps its fingerprint after both builds.

### B.5 Commands run

```bash
python scripts/train.py data=follownet_highd model=residual_idm_tt fold=0 seed=0 calibration.stability_margin=0.2 \
    certificate.enforce=true model.r_max=0.3 model.tau_z=15 train.max_epochs=2 train.steps_per_epoch=10 \
    experiment=m11_smoke                                                    # smoke (runs/m11_smoke), then its audit,
python scripts/audit_stability.py run=runs/m11_smoke/... [frozen_z=-1]      # frozen audit, certificate and platoon
python scripts/run_experiment.py configs/queue/e4_tt_pilot.yaml             # 3 runs, steps train, audit, platoon, certificate
bash frozen_audits.sh <run>...                                              # scripts/audit_stability.py run=<run> frozen_z=<z>,
                                                                            # z in -1 -0.5 0 0.5 1 (65 audits: 13 runs)
python scripts/run_experiment.py configs/queue/e4_tt.yaml                   # 10 runs: e4_tt (5) and e4_tt_ft (5)
python scripts/analysis/het_fd_tilt.py                                      # tilt file, table and figure het_fd_fit
python scripts/check_environment.py                                         # 17 OK (twice)
python scripts/build_corridor.py scenarios=false                            # 19 Oct: residual_idm_certified_het_fd
                                                                            # (residual_idm_tt not yet: members missing);
                                                                            # 20 Oct: residual_idm_tt
python scripts/run_corridor.py scenarios=[i80_p0,i80_p1,i80_p2,us101_p0,us101_p1,us101_p2] \
    laws=[residual_idm_certified_het_fd] seeds=[0,1,2,3,4,5,6,7,8,9] workers=3       # 60 runs
python scripts/run_corridor.py scenarios=[...the six...] laws=[residual_idm_tt] seeds=[0..9] workers=3   # 60 runs
python scripts/corridor_metrics.py law=residual_idm_certified_het_fd
python scripts/corridor_asymmetry.py law=residual_idm_certified_het_fd workers=2
python scripts/corridor_metrics.py law=residual_idm_tt
python scripts/corridor_asymmetry.py law=residual_idm_tt workers=2
python scripts/make_tables.py                                               # runs/_tables/m4 (and e2_monotone in m8)
python scripts/corridor_metrics.py compute=false tables=true                # runs/_tables/m5, m8: 51 laws, M11 rows
python scripts/analysis/certificate_tt.py                                   # table certificate_tt
python scripts/analysis/tt_summary.py                                       # tables tt_pilot, tt_highd, tt_ngsim
python scripts/analysis/m11_pairs.py                                        # tables het_fd_pairs, tt_pairs
python scripts/run_experiment.py configs/queue/{e1,e4,e4_rmax,m8_temporal}.yaml --dry-run   # hash checks
python -m pytest tests/test_certificate*.py tests/test_corridor_laws*.py tests/test_residual_idm_tt.py \
    tests/test_models.py ...                                                # targeted (see B.7)
python -m pytest -x -k "not slow"                                           # the quick full set (see B.7)
```

### B.6 Run counts and durations

- Training (GPU shared with agents A and C): pilot 3 runs (train 9.2, 5.8, 4.5 min; with audit, platoon and
  certificate 24 min of steps); e4_tt 5 runs (train 12.5-17.1 min, 72 min in all) and e4_tt_ft 5 runs (train 9.2-13.6
  min, 55 min in all); the queue e4_tt took 213 min of steps over 2 workers (19 Oct 19:12-21:00). The fine-tuning
  stopped early after 19-34 epochs (e4_stable_ft ran its 100).
- Audits at frozen latents: 65 (13 runs x 5 z), 9-47 s each.
- Corridor: residual_idm_certified_het_fd 60 runs (30 s to 30 min each under the load of 19 October, 3.2 h of run time
  in all, 12 contact episodes, 3 teleports); residual_idm_tt 60 runs on 20 October 11:20-12:57 (3.1-6.2 min each,
  4.6 h of run time over 3 workers). Metrics: 120 macro.json and 120 asymmetry.json (a few minutes in all); tables
  chain (make_tables, corridor tables, m11 tables) 3 min.
- The smoke run `runs/m11_smoke` (2 epochs, 10 steps each) is a leftover of the smoke test of 19 October and can be
  deleted.

### B.7 Tests

- New tests: `tests/test_residual_idm_tt.py` (11), `tests/test_certificate_latent.py` (10; its script tests run on
  three speeds to stay short), `tests/test_corridor_laws_m11.py` (8), with `tests/test_models.py`: 74 passed.
- `tests/test_certificate*.py`, `tests/test_corridor_laws*.py`, `tests/test_residual_idm.py`, `tests/test_train_script.py`,
  `tests/test_audit.py`: passed (with the new files).
- The quick full set (`pytest -x -k "not slow"`, in three passes because of `-x`): every test passes except
  `tests/test_check_release.py::test_the_checkout_passes`, which compares the counts of the trained runs (1 014) and of
  the corridor runs (1 577 at the time) with the manifest of the generated report (983, 1 445): the new runs of M11 (of
  all three agents) are not in that manifest until the report is generated again (main session).
  `tests/test_run_experiment.py::test_queue_files_of_the_experiments` lists the queue files: `e4_tt_pilot` (3 jobs) and
  `e4_tt` (10) added next to those of agent A. Passes: 84 passed before the release check; 596 passed, 1 skipped (that
  check deselected; stopped at the queue-file test, then fixed); 154 passed, 24 skipped (the files from
  `test_run_experiment.py` on).

## C: predictive validity (D127, H-N1), robust certificate and safety filter (D130, A3), empirical gain (D131, H-N4)

Status (updated after every step): D127 done; D131 done; D130: module, tests, safety filter, the 40 filtered
corridor runs, their metrics and the table safety_filter done; the robust-certificate tables and the budget allocation
are running (`scripts/analysis/robust_certificate.py`, 3 workers, started 20 October 11:35; sections 2.2 and 2.3 are
filled in when it finishes).

### 1. D127, H-N1: predictive validity of single-vehicle metrics for the macro error

**Done.** `scripts/analysis/predictive_validity.py` (new; tests `tests/test_predictive_validity.py`). Units: the 37
(law, corridor) pairs of `runs/_tables/m8/correlation_pooled_laws.csv` (19 laws on I-80, 18 on US-101) and, as the
separate check, these plus the six pairs of the ablation laws (`idm_core_margin`, `idm_margin_i80`,
`residual_idm_margin_free_r0.3` on both corridors): 43 pairs. Response: `macro_error` of `runs/_tables/m5/laws.csv`.
Tables `runs/_tables/m11/predictive_validity.{csv,md}` (statistics) and `predictive_validity_pairs.{csv,md}` (the
predictor values of every pair).

Predictors as implemented (details the contract leaves open are stated):

* (a) `unstable_eq` of `correlation_pooled_laws.csv`, for the ablation laws of `runs/_tables/m5/instability.csv`.
* (b) `1 - band_numerical["support"]["stable"]` of the member audits, mean over the members; equal to `not_stable` of
  the instability table for all 26 pairs that have it (largest difference 1.1e-16). For the laws without member runs
  (`idm_heterogeneous`, `idm_heterogeneous_all`, `idm_core_margin`, `idm_margin_i80`) and for the cores of
  `residual_idm_certified_het` (residual left out, as its (a) in the tables) the same categories in closed form: the
  IDM equilibrium spacing of every parameter set against the band of the training contexts of the runs the law file
  names (`core_runs`, `support_runs`, the members), stable = inside the band and exact discrete gain <= 1.02; mean
  over the sets and contexts. (a) recomputed this way agrees with the tables to 7.6e-3 (9 pairs; the per-fold support
  differs slightly from the law file's).
* (c), (d) the largest numerical gain of the member audits over the equilibria in support (inside or outside the
  band: the speeds of the H1.1 statistic) and the 25 audit frequencies, resp. the 9 frequencies <= 0.1 rad/s, mean
  over the members; closed form: the exact discrete gain of `idm_unstable_share`.
* (e) none of the 85 member runs of the 17 laws with member runs had `platoon.json`: the platoon step was run for all
  of them (`scripts/platoon_test.py run=<run>`, the step `platoon` of `configs/queue/e1.yaml`/`e5.yaml`; 85 files, 0
  errors). Statistic: the transient growth behind the braking pulse of the platoon test, `max_n sigma_n / sigma_1`
  over the 50 followers; a pulse platoon with a collision counts as +inf (ranked above every finite value), mean over
  the members (+inf once one collided). Collided member platoons: GRU 5/5, PERL 5/5, k-NN 4/5, GRU penalised 2/5,
  MLP 1/5, LSTM 1/5, all others 0. Laws without member runs: the pulse simulated with `simulate_platoons` and the
  options of `configs/stability/platoon.yaml` for all their parameter sets (`idm_core_margin`, `idm_margin_i80`: 5
  each) or 100 drawn uniformly (`idm_heterogeneous`, `idm_heterogeneous_all`; the (core, member) pairs of
  `residual_idm_certified_het`, drawn as the law draws them).
* (f) frozen gain along the trajectories: the follower states of the seed-0 runs of the pair's three scenarios rebuilt
  from `trajectories.npz` (the leader = the next vehicle ahead in the lane at the same whole second, `s = x_lead -
  length_lead - x`, kept: `0 < s <= 50 m`; 297 870-773 274 states per pair), 20 000 drawn uniformly per pair; the
  partials of the law that drove the vehicle (`law.assign` with the generator of seed 0 reproduces
  `vehicles.npz["member"]`, checked for every run; per-vehicle IDM parameters and het cores as drawn); autograd in
  float64 (on the GPU: equal to the CPU to 3e-15), the memoryless view of the audit for GRU/LSTM/PERL, central
  differences of the audit's step for k-NN; frozen gain = `max_w |transfer_discrete(f_s, f_dv, f_v)|` over the 25
  audit frequencies (dt 0.1 s); (f1) share of the states > 1.02, (f2) mean log gain.
* (g) `collisions_per_1000_vkm` of `laws.csv`; (h) `asymmetry_index` of `runs/_tables/m8/asymmetry.csv`; (h')
  `abs_error_asymmetry_index` of the same table (added: the raw index has no "worse" direction).

Statistics as specified: Spearman per corridor and pooled (corridor as stratum), cluster bootstrap over the laws (5000
resamples), permutation p (10 000; whole laws permuted, `linked_permutation_p`, for the per-law (a)-(e); within the
corridor, `stratified_permutation_p`, for the per-pair (f)-(h), whose values differ between the corridors), paired
cluster-bootstrap interval of rho_P - rho_(a) (the same 5000 law resamples for every predictor,
`default_rng(20261019)`), D-vs-E check `P(E) < P(D)` strictly on both corridors. Reading of the rule: "predicts
better" = the interval of rho_P - rho_(a) lies above zero (an interval below zero means worse) and the D-vs-E check
holds.

**Numbers against the rule (37 pairs; the 43-pair check behaves the same):**

| | predictor | pooled rho [law clusters] | rho - rho(a) [paired] | P(E) vs P(D) (I-80 / US-101) |
|---|---|---|---|---|
| a | unstable equilibria (reference) | 0.755 [0.527, 0.863] | - | 0 vs 0 |
| b | grid speeds not stable in the band | 0.767 [0.540, 0.880] | +0.012 [-0.016, +0.072] | 0 vs 0 (tie) |
| c | largest numerical gain | 0.678 [0.312, 0.834] | -0.077 [-0.320, +0.147] | 0.99979 vs 0.99971 |
| d | largest gain <= 0.1 rad/s | 0.677 [0.284, 0.849] | -0.078 [-0.327, +0.170] | 0.99979 vs 0.99971 |
| e | platoon transient growth | 0.708 [0.422, 0.862] | -0.047 [-0.187, +0.116] | 1 vs 1 (tie) |
| f1 | frozen gain, share > 1.02 | 0.706 [0.414, 0.855] | -0.049 [-0.294, +0.187] | 0.230 / 0.164 vs 0.144 / 0.081 |
| f2 | frozen gain, mean log gain | 0.312 [-0.179, 0.676] | -0.443 [-0.953, -0.021] | 0.123 / 0.149 vs 0.013 / -0.004 |
| g | contact rate per 1000 veh-km | 0.756 [0.437, 0.887] | +0.001 [-0.296, +0.253] | 0.000 / 0.227 vs 0.110 / 0.296 (E better) |
| h | asymmetry index | -0.596 [-0.810, -0.158] | -1.351 [-1.631, -0.798] | 1.281 / 1.488 vs 1.268 / 1.441 |
| h' | asymmetry, distance to the drivers | 0.308 [-0.089, 0.611] | -0.447 [-0.770, -0.091] | 0.319 / 0.472 vs 0.306 / 0.426 |

All permutation p < 0.001 except f2 (0.066) and h' (0.067). With the ablation laws (43 pairs): (a) 0.735 [0.507,
0.839]; differences b +0.009 [-0.016, +0.066], c -0.123, d -0.127, e -0.040, f1 -0.073 [-0.295, +0.128], f2 -0.405
[-0.859, -0.003], g -0.030 [-0.297, +0.190], h -1.351, h' -0.362. No interval of rho_P - rho_(a) lies above zero,
and only (g) ranks E better than D (on both sites). **Branch of the rule: no predictor predicts better (the "refuted"
branch of H-N1; the frozen-gain statistic (f) neither as f1 nor as f2).** Observation: E has a larger frozen-gain
share than D on both corridors (0.230 vs 0.144, 0.164 vs 0.081).

### 2. D130, A3: robust certificate, budget allocation, safety filter

**2.1 Implemented (done).** `cf_stability/stability/robust.py` (new; tests `tests/test_robust.py`, 14): the
follower's perturbation dynamics in the project's convention `dv = v - v_lead`, `e' = w - u`, `u' = f_s e + f_dv (u -
w) + f_v u` (the contract's `f_dv (w - u)` is the same system for `dv = v_lead - v`; the transfer function equals
`analytic.transfer_continuous`, tested); the box's eight corners as vertices; the bounded-real LMI `[[A^T P + P A +
C^T C, P B], [B^T P, -gamma^2]] < 0` with `P = L L^T`, gamma by bisection, the largest vertex eigenvalue minimised with
scipy (Nelder-Mead starts, BFGS), every certificate verified in float64 (all vertex eigenvalues <= -1e-9, P positive
definite); `robust_gain(box)` returns gamma or "infeasible"; `law_boxes` builds the boxes of a law over a region
(exact IDM core ranges at the ends of the spacing interval plus the residual budget; empirical ranges of the partials
at 20 000 states for networks). Tests: a known stable LTI system gives its H-infinity norm within 1 % (four systems),
an unstable box is infeasible, the verification rejects a wrong P, the IDM box equals the range of the sampled
derivatives, the budget allocation keeps the certificate.

Safety filter: `cf_stability/corridor/safety.py` (new; tests `tests/test_safety.py`, 13, among them a run of the loop
on the synthetic scenario of `tests/test_corridor_loop.py`), the hook in `cf_stability/corridor/loop.py` (the law's
accelerations of the controlled vehicles clipped to `min(a, a_safe)` before the loop's clip to `[a_min, a_max]`; the
counts of the active steps in `run.json` of filtered runs only) and the key `sim.safety_filter: null` in
`configs/run_corridor.yaml` (true or a mapping `{s_min, tau_s, b, alpha}`). Hash neutrality: `SimConfig.from_mapping`
returns the unchanged `SimConfig` while the key is null (its fields and hence `dataclasses.asdict` as before) and a
subclass `FilteredSimConfig` with the parameters as a field otherwise; the stored `sim` blocks and config hashes of
all 1445 corridor runs are reproduced by the new code (with and without the key at its default): 0 changed. Run
directories: the law suffix `_filter` (`runs/corridor/<scenario>/<law>_filter/seed<s>`): four law files that are
copies of `mlp`, `gru`, `lstm`, `residual_idm_certified` (same members and devices, hence the same member draws and
SUMO seed per seed), from a block appended to `configs/corridor/laws.yaml` (`mlp_filter`, `gru_filter`, `lstm_filter`,
`residual_idm_certified_filter`) and exported with `export_laws(..., only=[these four])` (no other law file was
rewritten); the loop refuses a filtered run of a law without the suffix and an unfiltered run of a law with it.
Checks: at every band equilibrium of the members' audits of the four laws `h > 0` (smallest h: GRU 0.61 m, LSTM 1.41,
MLP 3.55, certified hybrid 4.50) and the bound is inactive there and 0.25 m / 0.1 m/s around (tested with the
models); two-vehicle toy with the loop's arithmetic: no contact behind a leader braking at 2 m/s^2 (the platoon test's
pulse) to a lower speed or to standstill, for followers that ignore the leader, started anywhere with h >= 0 (smallest
gap 1.09 m; tested); with sustained leader braking of 3 m/s^2 and more, contacts do occur in that toy (the barrier
assumes a constant leader speed): a limit of the filter as specified, not tested as a property.

Runs: `python scripts/check_environment.py` (17 OK), then the 40 runs (laws `*_filter` on `i80_p1` and `us101_p1`,
seeds 0-4, 3 workers): 40 done, 0 failed, 19 October 19:02-20:18 (wall times 33-2104 s per run, 11 277 s in all; the
CPU was shared with two other agents).

**2.2 Robust certificate (D130 rule).** Run on 20 October (3 648 polytopes, 1 996 s, 3 workers; table `runs/_tables/m11/robust_certificate` with its member-level twin `robust_certificate_members` and the polytope list `robust_certificate_polytopes`). On the region of the rule (the spacing band of the NGSIM I-80 data at the grid speeds, box polytope, per speed) no law has a certificate: all 80 polytopes of every law are infeasible, including E at every r_max, the margin core alone (B) and the IDM (vertex H-infinity lower bounds 6.7-16.6 for the IDM and B). A common quadratic storage function over the box of the core's derivative ranges across the band does not exist even for laws that are string stable at every equilibrium of that band, so the criterion of the rule is too conservative to separate anything; the rule's branch is therefore **"does not explain"** (E infeasible, D infeasible), but for a reason that says nothing about E against D. On the feasible-gap set I(v) per speed the picture is informative: the laws without a residual are certified at gamma 1.0001 at every polytope (B 130/130, IDM 109/109: string stable in the reading above), E is certified with a finite gamma that grows with the residual budget (box: 1.044, 1.098, 1.248, 8.24 at r_max 0.1, 0.2, 0.3, 0.5; hull: 1.028, 1.039, 1.055, 1.159), and C and D, the variants without a derivative budget, are infeasible (D at 5 of 130 polytopes, at 30 m/s; C everywhere). The derivative budget is thus what makes a common storage function exist at all, in the direction of the hypothesis, but the certified gain of E stays above 1 + 1e-3 at every budget, and it moves against the corridor error: the tightest budget has the best gamma and the worst macro error of the E variants (r_max 0.1: gamma 1.044, error 0.197/0.177; r_max 0.5: gamma 8.24, error 0.108/0.104). A single polytope over the whole speed range certifies nothing useful (gamma 1.4-16 or infeasible). The memoryless networks with their empirical boxes on the band are infeasible at 59 (MLP) and 67 (PIDL) of 80 polytopes, consistent with their audits.

**2.3 Budget allocation.** Table `runs/_tables/m11/budget_allocation`. Under the criterion of the rule (a common storage function at gamma <= 1 + 1e-3 on the band or on I(v)) the largest feasible budget is zero for every fold member of E, on every polytope and scope: the core alone is not certified on the band, and on I(v) its certificate sits at 1.0001 with no slack for a residual, so the LMI-based allocation is degenerate. Under the pointwise criterion of the manuscript's a priori certificate (the sufficient condition on I(v) at 400 spacings per speed, without a common storage function) the ascent that maximises B_s + B_dv + B_v allocates B_s 0.014-0.015, B_dv 0.074-0.077 and B_v 0 (totals 0.089-0.091 against the uniform totals 0.065-0.067 of r_max 0.3: ratios 1.34-1.37): the pointwise certificate admits about a third more total derivative budget than the uniform allocation, all of it on the spacing and speed-difference inputs and none on the follower's own speed. Whether that allocation buys accuracy is untested (the stretch goal of D130 was not run).

**2.4 Safety filter (D130 rule).** `scripts/corridor_metrics.py scenario=<s> law=<law>_filter` for the 40 runs (40
macro.json, about 1 min), then `scripts/analysis/safety_filter.py` (new) -> `runs/_tables/m11/safety_filter.{csv,md}`
(per law and scenario, paired by seed with the unfiltered runs of the corridor: means off/on and the paired change
with its bootstrap interval over the 5 seeds) and `safety_filter_runs.{csv,md}` (per run).

| scenario | law | contacts / 1000 veh-km, off -> on | episodes per run, off -> on | macro error, off -> on (change [interval]) | filter active |
|---|---|---|---|---|---|
| i80_p1 | MLP | 813.9 -> 0.000 | 431.6 -> 0.0 | 0.233 -> 0.201 (-0.033 [-0.087, +0.042]) | 4.4 % |
| i80_p1 | GRU | 173.3 -> 0.364 | 101.4 -> 0.2 | 0.170 -> 0.238 (+0.067 [+0.035, +0.089]) | 1.3 % |
| i80_p1 | LSTM | 6305 -> 0.370 | 410.4 -> 0.2 | 0.683 -> 0.422 (-0.261 [-0.378, -0.096]) | 9.6 % |
| i80_p1 | certified hybrid E | 0 -> 0 | 0 -> 0 | 0.131 -> 0.148 (+0.017 [-0.023, +0.065]) | 0.0 % |
| us101_p1 | MLP | 587.1 -> 0.231 | 483.4 -> 0.2 | 0.558 -> 0.528 (-0.030 [-0.055, +0.005]) | 5.0 % |
| us101_p1 | GRU | 300.8 -> 0.000 | 181.6 -> 0.0 | 0.399 -> 0.225 (-0.174 [-0.451, -0.005]) | 0.9 % |
| us101_p1 | LSTM | 2978 -> 2.273 | 837.8 -> 0.4 | 0.831 -> 0.671 (-0.160 [-0.279, -0.050]) | 10.9 % |
| us101_p1 | certified hybrid E | 0 -> 0 | 0 -> 0 | 0.115 -> 0.108 (-0.008 [-0.024, +0.005]) | 0.06 % |

The filter removes 99.9-100 % of the networks' contact episodes, but one or two episodes remain in 4 of the 6
network-scenario cells (GRU and LSTM on i80_p1, MLP and LSTM on us101_p1; 1-2 of the 5 runs each); the certified
hybrid's macro error changes by +0.017 (i80_p1) and -0.008 (us101_p1), within 0.02. **Branch of the rule: the filter
does not work (the contact rate of the three networks does not fall to zero on both scenarios; the condition on the
certified hybrid holds).** The networks' macro error falls in 5 of 6 cells (MLP -0.033/-0.030, LSTM -0.261/-0.160,
GRU on us101_p1 -0.174) and rises for the GRU on i80_p1 (+0.067, interval above 0). The filter also changes the
demand the LSTM lets in (inserted share 0.37 -> 0.63 on i80_p1, 0.38 -> 0.44 on us101_p1; the slow LSTM stays below
the demand with and without the filter).

### 3. D131, H-N4: empirical gain of the drivers against the laws

**Done.** `scripts/analysis/empirical_gain.py` (new; tests `tests/test_empirical_gain.py`); tables
`runs/_tables/m11/empirical_gain.{csv,md}` (excess per data set, estimator and law), `empirical_gain_cells.{csv,md}`
(per cell: events, the drivers' median and 25-75 % range, the pooled estimate and its coherence, every law's gain),
`empirical_gain_rule.csv` (the branch per estimator and interval); figure
`runs/_report/supplement/figures/empirical_gain.{png,pdf}` with its caption `empirical_gain.txt`.

Drivers: `EventSet.from_parquet` of `data/events/ngsim_i80`, the 3592 of 7651 events of at least 40 s; per event the
H1 estimate `|P_xy| / P_xx` (`scipy.signal.csd`/`welch`: Hann, 40-s segments, 50 % overlap, linear detrending per
segment, density) at the Welch lines k/40 Hz in 0.3-2 rad/s (0.31-1.88 rad/s, 11 lines); 5-m/s bins of the event's
mean follower speed (0-5 m/s: 1522 events, 5-10: 1946, 10-15: 124); per cell the median and the 25-75 % range. Laws:
the numerical gains of `stability.json` of the E1 runs (`runs/e1/follownet_highd/<model>/driver_fold<k>_seed0`,
folds 0-4) of MLP, PIDL, GRU, LSTM, PERL, the IDM and k-NN (entries of clipped, stopped or collided rollouts left
out), the log gain interpolated linearly in log frequency and in speed to the cell (at the median mean speed of its
events; no extrapolation: the 0-5 m/s bin has no law value), mean over the folds. Excess = mean over the 22 cells with
both values of `log|G_law| - log|G_emp,median|`; intervals (2000 resamples, `default_rng(20261019)`): the events
resampled (law side fixed), the folds resampled (drivers fixed), both (joint).

**Numbers against the rule (NGSIM I-80, the statistic of D131):** excess [events interval; folds interval] MLP -0.918
[-0.963, -0.880; -0.937, -0.903], PIDL -0.788 [-0.833, -0.749; -0.854, -0.730], GRU -0.578 [-0.622, -0.539; -0.734,
-0.430], LSTM -0.821 [-0.865, -0.782; -1.308, -0.403], PERL -0.355 [-0.400, -0.317; -0.432, -0.265], IDM -0.310
[-0.355, -0.271; -0.318, -0.303], k-NN -0.644 [-0.689, -0.605; -0.820, -0.478]; the joint intervals likewise exclude
zero. 0 of 5 networks exceed the drivers. **Branch: "refuted" (fewer than three networks exceed the drivers), with
every interval (events, folds, joint).**

Diagnostic and sensitivity (not under the rule): the drivers' median per-event gain stays at 0.77-0.81 at 1-2 rad/s
in the 5-10 m/s bin where the laws fall to 0.2-0.5, while the coherence of the pooled spectra of the bin drops below
0.25 above 0.78 rad/s (below 0.05 above 1.1 rad/s): with one to three segments per event the per-event ratio is close
to `|Y| / |X|` and contains the follower's fluctuations that the leader does not cause. The pooled H1 estimate of a
cell (`|sum P_xy| / sum P_xx` over all segments of its events) removes these but is lowered by the phase differences
between the drivers: with it the excess is positive for every law (MLP +0.270, PIDL +0.400, GRU +0.611, LSTM +0.368,
PERL +0.833, IDM +0.878, k-NN +0.544; 5 of 5 networks above with the event intervals, 4 of 5 with the fold and joint
intervals; the IDM positive: branch "open"). On the cells with a pooled coherence >= 0.5 (0.31 and 0.47 rad/s, 4
cells) both estimators give the "refuted" branch (median: every excess negative, GRU -0.022, PERL -0.013 the
largest; pooled: GRU +0.122, PERL +0.131 above, the others below). The result of D131's statistic rests on the
0.6-2 rad/s cells, where the drivers' response is not coherent with the leader.

HighD (deviation): the FollowNet HighD events are 15-s windows (14.9 s, D19); none reaches 40 s, so the rule runs on
NGSIM I-80 alone. With one Hann segment of the whole event (lines 0.42, 0.84, 1.27, 1.69 rad/s; 12 512 events, 6
speed bins) the excess is negative for every law as well (MLP -1.433 ... PERL -0.552, IDM -0.659; branch "refuted");
pooled: PERL +0.419, IDM +0.312, GRU +0.067 above, the others below (branch "refuted").

Stretch goal `gain_match`: not done (the penalty would go to `cf_stability/stability/penalties.py`, owned by agent A).

### 4. Deviations (summary)

* D127: (e) defined as the pulse transient growth with collisions as +inf; (h') added; the per-pair predictors use the
  within-corridor permutation (the linked one needs one value per law); closed forms for the laws without member runs.
* D130: the sign convention of the f_dv term; "gamma < 1" read as a verified certificate at gamma = 1 + 1e-3 (2.2);
  the filtered runs named by the law suffix `_filter`; the filter's no-contact property holds for leader braking up to
  2 m/s^2 (tested), not for 3 m/s^2 and more.
* D131: HighD has no event >= 40 s (D19): the rule runs on NGSIM I-80; HighD with 14.9-s segments and the pooled H1
  estimator are reported as sensitivities.

### 5. Commands run (repository root, Git Bash, `./.venv/Scripts/python.exe` as python)

```
python scripts/check_environment.py
python scripts/platoon_test.py run=<run>          # the 85 member runs without platoon.json, one call each
python -c "export_laws(laws.yaml, runs/corridor/laws, only=[mlp_filter, gru_filter, lstm_filter, residual_idm_certified_filter])"
python scripts/run_corridor.py "scenarios=[i80_p1,us101_p1]" \
    "laws=[mlp_filter,gru_filter,lstm_filter,residual_idm_certified_filter]" "seeds=[0,1,2,3,4]" workers=3 \
    sim.safety_filter=true
python scripts/analysis/predictive_validity.py --device cuda
python scripts/analysis/predictive_validity.py --from-pairs runs/_tables/m11/predictive_validity_pairs.csv
python scripts/analysis/empirical_gain.py
python scripts/corridor_metrics.py scenario=<i80_p1|us101_p1> law=<mlp|gru|lstm|residual_idm_certified>_filter   # 8 calls
python scripts/analysis/safety_filter.py
python scripts/analysis/robust_certificate.py --workers 3
python -m pytest tests/test_analytic.py tests/test_corridor_tables_ablation.py tests/test_corridor_macro.py \
    tests/test_corridor_loop.py tests/test_robust.py tests/test_safety.py tests/test_predictive_validity.py \
    tests/test_empirical_gain.py -q
```

Counts and durations: platoon step 85 runs (1305 s); predictive validity 43 pairs, 22 laws, 20 000 states per pair
(849 s; the statistics alone from the pairs table 145 s); empirical gain 3592 NGSIM I-80 and 12 512 HighD events
(47 s); 40 filtered corridor runs (1 h 16 min on 19 October); their 40 macro.json and the table safety_filter (1 min
on 20 October); tests: 81 passed on 19 October (with tests/test_corridor_loop.py, 1511 s on the shared CPU) and,
after the changes of 20 October to robust.py and robust_certificate.py, 73 passed (tests/test_analytic.py,
test_corridor_tables_ablation.py, test_corridor_macro.py and the four test files of agent C; 173 s).

Incidents: the session ended with the API outage of 19 October about 21:00; the 40 filtered runs, the platoon step,
the predictive-validity and empirical-gain tables had finished before it. The first full run of
robust_certificate.py (19 October, 10 workers) stopped on a `LinAlgError: Singular matrix` of a numerically singular
storage function inside the Schur-gain search: `schur_gain` now treats a singular or non-finite matrix as no
certificate (tested), every polytope's exception is recorded as status `error` instead of stopping the run, the
results are appended to a checkpoint file in chunks (a restart continues), torch is no longer imported by the worker
processes (130 MB instead of 585 MB each), and the run was restarted on 20 October with 3 workers. The section of
this report had been duplicated once by a concurrent rewrite of the file that dropped its start marker; the writer of
the section now removes every earlier copy.

Shared files appended to: `configs/corridor/laws.yaml` (the block "M11, D130 (agent C)" with the four `*_filter`
laws). Files of agent C: `cf_stability/stability/robust.py`, `cf_stability/corridor/safety.py`,
`cf_stability/corridor/loop.py` (the hook, `SimConfig.from_mapping`, `FilteredSimConfig`), `configs/run_corridor.yaml`
(the key), `scripts/analysis/predictive_validity.py`, `robust_certificate.py`, `empirical_gain.py`, `safety_filter.py`
(new: the table m11/safety_filter), `tests/test_robust.py`, `test_safety.py`, `test_predictive_validity.py`,
`test_empirical_gain.py`.
<!-- section C: end -->
