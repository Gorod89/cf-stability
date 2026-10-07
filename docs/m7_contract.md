# M7 contract: revision arms (the gaps of the M6 review)

Basis: the review of M6 on 2026-10-05 asked to (1) formalise the verdicts of H12.1-H12.3 of
Part A of the plan and (2) close the seven gaps named in that review. Numbers in this file come
from `runs/_report/report.md` of 2026-10-01. Decisions of M7 are D108-D114 (one number per item,
fixed below, so that three work packages can append to `docs/decisions.md` without colliding: re-read
the file before every edit and append at the end of the table). Nothing is retrained of the
M4-M6 runs; `configs/train.yaml` keeps every hash of the existing runs (section 3.4).

Three work packages own disjoint files:

| work package | items | files |
|---|---|---|
| A (statistics, tables, report) | 1, 2, and the tables of 3-7 | `cf_stability/eval/*`, `configs/make_tables.yaml`, `configs/corridor_metrics.yaml`, `configs/make_report.yaml`, `scripts/make_tables.py`, `scripts/make_report.py`, `scripts/corridor_metrics.py` (tables part only), their tests |
| B (training) | 3, 4 (training side) | `cf_stability/stability/penalties.py`, `cf_stability/train/*`, `cf_stability/utils.py`, `configs/train.yaml`, `configs/queue/*`, `scripts/train.py`, their tests |
| C (corridor) | 4 (laws), 5, 6, 7 | `cf_stability/corridor/*`, `scripts/build_corridor.py`, `scripts/run_corridor.py`, `scripts/corridor_metrics.py` (metrics part), `configs/corridor/*`, `configs/build_corridor.yaml`, `configs/run_corridor.yaml`, their tests |

The long queues (sections 3-7) are started by the lead, not by the work packages; the work packages
run smoke tests only (one fold, few epochs, one short corridor run).

## 1. Verdicts of H12.1-H12.3 (work package A, D108)

Part A of the plan states, for the corridor:

| hypothesis | statement | confirmed | refuted |
|---|---|---|---|
| H12.1 | the pure network degrades >= 2 of 3 macro metrics by >= 15 % relative to the IDM | on both corridors | no metric degraded |
| H12.2 | the stable hybrid is not inferior to the IDM in macro metrics and superior in micro metrics | TOST equivalence (+-10 %) in macro and superiority in micro | macro worse by > 10 % |
| H12.3 | the macro error correlates with the share of unstable equilibria, r > 0.6 over >= 10 models | r > 0.6, p < 0.05 | r < 0.3 |

Definitions (D108):

- The macro triple: throughput at `x_out`, travel time (Wasserstein-1), wave speed by
  cross-correlation: one anchored flow quantity, one delay quantity, one dynamic quantity (the
  headline wave number of D105). The signed relative errors are the components of the
  macro-error vector of D99.
- H12.1, per pure network (`mlp`, `gru`, `lstm`: the unpenalised networks, I-80 laws of D97) and
  per corridor: degradation of a metric = |e_net| - |e_idm_global| with e the signed relative
  error, paired by scenario and seed (unit run), 95 % percentile bootstrap. A metric is
  "degraded" when the lower bound of the interval is above 0, "degraded by >= 15 %" when in
  addition the point estimate is >= 0.15. Verdict per network and corridor: confirmed with >= 2
  of 3 metrics degraded by >= 15 %, refuted with no metric degraded, otherwise open. Overall:
  confirmed when confirmed for >= 2 of 3 networks on every available corridor; the row names the
  corridors used ("I-80 only" until US-101 exists).
- H12.2: macro part = the TOST of `residual_idm_certified` against `idm_global` (margin +-10 %,
  the table `tost`) on the macro triple: holds when all three are equivalent; "macro worse by
  > 10 %" when the interval of a relative difference lies beyond +-10 % on the bad side (lower
  throughput, longer travel time, slower wave speed in magnitude). Micro part = closed-loop
  spacing RMSE on the test parts of `ngsim_i80`, paired over drivers, `e4_stable_ft` (certified
  hybrid after fine-tuning) against `e3_reference` IDM of the same fold (the same split): superior
  when the upper bound of the relative difference is below 0. Verdict: confirmed when both parts
  hold; refuted when the macro part is worse by > 10 %; otherwise open, with the text stating
  which part fails. Reference rows (no verdict): `e4_free_ft` residual_idm and mlp against the
  same IDM. The current numbers (means of `rmse_s_mean` over the runs): IDM 4.66 m, certified
  hybrid 4.89 m, free hybrid 4.27 m, fine-tuned MLP 5.57 m; the micro part is expected to fail.
- H12.3: Spearman over the laws of the macro error (all components) against the share of
  unstable equilibria among the equilibria found (table `instability_correlation`, all laws),
  with the existing bootstrap interval and a new permutation p-value (10 000 permutations of the
  instability values over the laws, two-sided). Holds when r_S > 0.6, p < 0.05 and n >= 10;
  refuted when r_S < 0.3; otherwise open. Pearson is reported in a second row with the same rule
  but does not decide (a monotone relation with collapsed laws is a rank statement); the dynamic
  macro error gets the same two rows as secondary information.
- Output: `runs/_tables/m5/verdicts.{csv,md}` (rows H12.1 per network and corridor plus overall,
  H12.2, H12.3 Spearman and Pearson), the tables `h12_1` (degradations) and `h12_2` (the micro
  comparison), the permutation p-value in `instability_correlation`. `make_report` section 2
  shows the M4 and the M5 verdict tables one after the other and section 8 the new tables.
  The US-101 scenarios of item 5 enter as a second corridor when their `macro.json` files exist
  (table rows per corridor, "I-80 only" otherwise).

## 2. H1.3 with a collision-free outcome (work package A, D109)

D107 left every H1.3 verdict open because a collided platoon has no growth error and the
OpenACC-trained laws collide on 7-14 of their 10-15 profiles. D109: the growth error of a
profile is computed on the collision-free prefix: with empirical positions 1..P (P = 3..10) and
the first collided position c of the model platoon (none: c = infinity), the comparison uses the
positions 1..min(P, c - 1) when these are at least 3, with the same normalisation (RMSE between
the model and the empirical curve divided by the maximum of the empirical curve over the
positions used); fewer than 3 positions: undefined. Two descriptive outcomes per run: the mean
first collided position over the profiles (a platoon without collision counts as 51, one past
the last vehicle) and the collision-free share of the profiles. The verdict rule of H1.3 is
unchanged (reduction >= 30 % with the interval excluding 0: confirmed; < 10 %: refuted; fewer
than 3 pairs: open); the D107 full-curve numbers stay as secondary columns. The same prefix
rule applies to the growth-error columns of the E2 table. The `platoon.json` files are reread
(they hold the curves and the collision positions; verify; if the collided position is not
stored, say so: the platoon step of E5 (70 runs) is then rerun by the lead).

## 3. Low-frequency penalty arm for the recurrent models (work package B, D110)

The rollout gain penalty of the specification is blind below 0.05 rad/s (its lowest frequency,
40 s rollout), and the linearised gain penalty of M3 was gamed by needles (f_dv of -35 to -237
1/s) and by locally unstable equilibria. The Jacobian criterion M = f_v^2 + 2 f_v f_dv - 2 f_s
>= 0 is the omega -> 0 limit of |G(i omega)| <= 1 (|G|^2 = 1 - omega^2 M / f_s^2 + O(omega^4)),
so it is the low-frequency complement of the rollout penalty. D110: penalty kind `combined`
for the recurrent models = the rollout gain penalty of E2 (5 frequencies, 16 speeds, `every` 8,
weight `train.penalty.weight` = 0.1 as chosen) plus the Jacobian penalty of the specification
applied to the memoryless view of the model (the window filled with the constant state; the
derivatives are the sums over the window positions, i.e. the static gains; margin 0.5, the
local terms relu(-f_s) + relu(f_dv) + relu(f_v + f_dv), equilibria anchored to the band as in
E2) with weight `train.penalty.jacobian_weight`, plus a needle guard relu(|f_dv| - c) +
relu(|f_v| - c) with c = `train.penalty.guard` = 3 1/s (the calibrated IDM has |f_dv|, |f_v|
below 1 1/s at every grid speed; the needles of M3 were an order of magnitude beyond).

- Implementation: `penalties.py` gets the memoryless-view Jacobian for recurrent models (reuse
  the equilibrium finder's constant-history evaluation) and the guard; the trainer applies the
  Jacobian part at every step (cheap) and the rollout part every n-th step as now; the penalty
  of the epoch choice (D89) is the sum. Finite-difference test of the gradient of the new terms;
  a test that the memoryless-view derivatives of a GRU with constant history equal the finite
  differences of its output under a common shift of the window.
- Queues: `e2_lowfreq_pilot.yaml` (fold 0, seed 0, gru/lstm/perl, jacobian_weight in {0.1, 1},
  6 runs, steps train, audit, platoon; experiment `e2_combined_j{jacobian_weight}`), then
  `e2_lowfreq.yaml` (folds 0-4, seed 0, the weight chosen per architecture by the rule of D85
  from the pilot, written into the queue file by hand, 12 further runs; the same experiment
  names). Runs of the combined kind cost what the E2 gain runs cost (66-89 min).
- Table `e2_lowfreq` (work package A): per architecture and weight, test RMSE and its change against
  E1 (paired over drivers), band shares, unstable among equilibria, max gain, collided profiles,
  and a verdict row "H1.2 (combined)" per architecture by the H1.2 rule; the E2 verdict stays.

### 3.4 Hash-neutral defaults

`configs/train.yaml` gets the keys `train.penalty.jacobian_weight: 0.0` and
`train.penalty.guard: 0.0` (0 = off). A new key changes the config hash of every run and would
make the restartable queues retrain everything, so the config hash drops these keys while they
hold their defaults (a list of late additions in `cf_stability/utils.py` or wherever the hash is
computed), and a test recomputes the hash of an existing run (`metrics.json` of
`runs/e1/follownet_highd/mlp/driver_fold0_seed0`) from its stored config and the current code
and asserts equality. `python scripts/run_experiment.py configs/queue/e1.yaml --dry-run` (or
the equivalent) must report every job complete after the change.

## 4. Residual-amplitude sweep of the certified hybrid (work packages B and C, D111)

The corridor result of the certified hybrid (macro error 0.125, the best law) rests on one
amplitude, r_max = 0.3, chosen on fold 0. D111: E4 is repeated at r_max in {0.1, 0.2, 0.5}
(`e4_stable_r{r}` on `follownet_highd`, margin 0.2, certificate enforced; `e4_stable_ft_r{r}`
fine-tuned on `ngsim_i80`; 25 runs each) and a control that separates the amplitude from the
certificate: `e4_free_r0.3` (free core, no certificate, r_max 0.3) and `e4_free_r0.3_ft`. Queue
`e4_rmax.yaml`, 200 runs, steps train, audit, certificate. The existing e4 runs are the r = 0.3
and r = 1.0 (free) points.

Corridor (work package C): law files `residual_idm_certified_r0.1`, `..._r0.2`, `..._r0.5` and
`residual_idm_free_r0.3` from the fine-tuned runs (five folds each, as the existing
`residual_idm_certified`), entries in `configs/corridor/laws.yaml`; 4 laws x 3 periods x 10 seeds
= 120 runs on I-80 (and on US-101 when it exists).

Table `e4_rmax` (work package A): per r_max (0.1, 0.2, 0.3, 0.5, free 0.3, free 1.0): certificate
holds a priori / after fine-tuning (runs), unstable among equilibria, RMSE on HighD, RMSE on
NGSIM, and, from the corridor tables, the macro error, the dynamic macro error and the
collisions per 1000 veh-km: the certificate-accuracy-corridor curve.

## 5. Second corridor: NGSIM US-101 (work package C, D112)

Source `data/raw/ngsim/cache/ngsim_us-101_reconstructed.parquet` (the reconstruction of M1,
D42-D46; the same columns as I-80). Geometry from the data, not from memory: 640 m study
section, lanes 1-5 through lanes, lane 6 the auxiliary lane between the on-ramp (lane 7) and
the off-ramp (lane 8); derive the x-ranges of lanes 6, 7, 8 from the positions where they are
occupied, and the three 15-minute periods from the file. Edges as for I-80 (`main_a` lanes
1-5, `main_b` with the auxiliary lane as index 0, `main_c` lanes 1-5, buffer `out`) plus an
`offramp` edge that leaves the end of the auxiliary lane: vehicles that leave by the off-ramp in
the data get a route that ends on it (LC2013 strategic changes bring them to the auxiliary
lane); vehicles of the on-ramp enter on the auxiliary lane at the on-ramp position with their
data speeds. `hov_lane: null` unless the data show a restricted lane. Everything else as I-80
(D94-D104: boundary feedback, leader range, contact episodes, insertion, analysis window
chosen from the loading of the section). Scenarios `us101_p{0,1,2}`, ground truth and
`boundary.npz` as for I-80; the builder reports the ground-truth throughput, mean speed, wave
speed and the number of vehicles per period, and a pilot run of `idm_global` on period 1 (one
seed) before the queue.

Laws: the thirteen I-80 laws of D97 unchanged (fine-tuned on I-80: US-101 is a transfer to
another site; nothing is retrained) plus the r_max laws of item 4. Runs: 13 x 3 x 10 = 390
(+120). Metrics: `corridor_metrics.py` with the scenario names; tables (work package A): every M5 table
gets a `corridor` column (I-80, US-101) and the H12 verdicts use both corridors.

## 6. Sensitivity to the downstream boundary and to the lane-change model (work package C, D113)

The anchored macro quantities follow the boundary feedback of D95 (D105 says so). D113:
scenario variants of I-80 period 1 built with overrides: `gain 0.01`, `gain 0.04`, `feedback
false` (speed-only boundary), and two LC2013 settings, `lc_low` (lcSpeedGain 0.5, lcCooperative
0.5, lcAssertive 0.5) and `lc_high` (lcSpeedGain 2.0, lcCooperative 1.0, lcAssertive 1.5),
named `i80_p1_<variant>`; laws `idm_global`, `residual_idm_certified`, `gru`; 5 seeds; 5
variants x 3 laws x 5 seeds = 75 runs (the baseline is the existing period 1). The builder
must accept the variant name and the overrides from the command line (hydra) and write the
variant into `scenario.json`. Table `sensitivity` (work package A): per variant and law the macro
error, the macro triple and the collisions, and the rank of the three laws per variant; the
report states whether the ranking certified < IDM < GRU survives every variant.

## 7. Heterogeneous IDM from all estimates (work package C, D114)

`idm_heterogeneous` draws its parameter sets from the interior per-event estimates of
`ngsim_i80` only (D63). D114: law `idm_heterogeneous_all` draws from all per-event estimates
(bounds-hitting ones included; the same sampling otherwise); 30 runs on I-80 (3 periods x 10
seeds); a row in the laws tables and in the instability table (exact gain of its parameter
sets, as for `idm_heterogeneous`).

## 8. Not closable here: raw highD

Raw highD is not on the machine (levelXdata application pending, D22). The FollowNet HighD
events stand in; the paper cites FollowNet as the source of the events. Nothing to do in M7.

## 9. Order of execution and reporting

1. Work packages A, B, C in parallel (code, tests, smoke tests); every work package reports the commands
   run, the tests, and the exact names of the experiments, laws and tables it created.
2. Lead: GPU queues `e2_lowfreq_pilot` -> `e2_lowfreq`, `e4_rmax` (2 workers); corridor
   queues on the CPU: item 7 (30), item 6 (75), item 4 laws on I-80 (120), US-101 (390, then 120).
3. `python scripts/make_tables.py`, `python scripts/corridor_metrics.py tables=true`,
   `python scripts/make_report.py refresh=true`, `docs/m7_report.md`, commit, stop for review.
